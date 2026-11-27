import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import build_yolov10_candidate_cache, yolov10_candidate_mask_from_cache
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import (
    _class_loss_tensor,
    build_detector,
    expand_layer_names,
    map_grad_tensor_to_numbers,
    resolve_layer_parameter,
)
from models.yolov10.core import xywh2xyxy


def _bbox_offset_from_xyxy(boxes_xyxy, source_points):
    px = source_points[:, 0]
    py = source_points[:, 1]
    return torch.stack(
        [
            px - boxes_xyxy[:, 0],
            py - boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - px,
            boxes_xyxy[:, 3] - py,
        ],
        dim=1,
    )


def _bbox_offset_loss(final_xyxy, source_point, mode):
    source_point = source_point.view(1, 2)
    final_xyxy = final_xyxy.view(1, 4)
    target = torch.zeros_like(_bbox_offset_from_xyxy(final_xyxy, source_point))
    offset = _bbox_offset_from_xyxy(final_xyxy, source_point)
    diff = offset - target
    if mode == "l1":
        return diff.abs().sum()
    if mode == "l2":
        return diff.pow(2).sum()
    raise ValueError("bbox_loss must be l1 or l2.")


def _candidate_bbox_offset_loss(candidate_xyxy, candidate_source_points, final_xyxy, mode):
    candidate_offset = _bbox_offset_from_xyxy(candidate_xyxy, candidate_source_points)
    final_xyxy = final_xyxy.detach().view(1, 4).expand(candidate_xyxy.shape[0], -1)
    target_offset = _bbox_offset_from_xyxy(final_xyxy, candidate_source_points).detach()
    diff = candidate_offset - target_offset
    if mode == "l1":
        return diff.abs().sum()
    if mode == "l2":
        return diff.pow(2).sum()
    raise ValueError("bbox_loss must be l1 or l2.")


def _zero_grad_features(row, params_by_scalar, scalars, reductions):
    for scalar_name in scalars:
        for layer_name, _param in params_by_scalar[scalar_name]:
            safe = layer_name.replace(".", "_")
            for metric in reductions:
                row[f"{safe}_{scalar_name}_{metric}"] = 0.0


def _write_grad_features(row, params, grads, scalar_name, reductions):
    for (layer_name, _param), grad in zip(params, grads):
        safe = layer_name.replace(".", "_")
        stats = map_grad_tensor_to_numbers(grad)
        for metric in reductions:
            value = stats.get(metric, 0.0)
            row[f"{safe}_{scalar_name}_{metric}"] = float(value.detach().cpu().item()) if isinstance(value, torch.Tensor) else float(value)


def _build_null_scalar_terms(forward, item, layer_cfg, device):
    raw_box_idx = item["raw_box_idx"]
    source_point = forward.source_points[raw_box_idx].to(device=device, dtype=torch.float32)
    final_xyxy = item["box"][:4].float()
    scalar_terms = {}
    if "bbox_loss" in layer_cfg["scalar"]:
        scalar_terms["bbox_loss"] = _bbox_offset_loss(final_xyxy, source_point, layer_cfg["bbox_loss"])
    if "cls_loss" in layer_cfg["scalar"]:
        cls_logits = forward.raw_logits[item["sample_idx"], raw_box_idx]
        target_value = 0.5 if layer_cfg["cls_loss"] == "bcewithlogits" else 1.0 / float(max(1, cls_logits.numel()))
        cls_target = torch.full_like(cls_logits, target_value)
        scalar_terms["cls_loss"] = _class_loss_tensor(
            cls_logits,
            cls_target,
            class_idx=None,
            mode=layer_cfg["cls_loss"],
            direction=layer_cfg["cls_direction"],
            reduction="sum",
        )
    return scalar_terms


def _build_candidate_scalar_terms(forward, item, candidate_indices, layer_cfg, device):
    scalar_terms = {}
    if candidate_indices.numel() == 0:
        return scalar_terms
    sample_idx = item["sample_idx"]
    if "bbox_loss" in layer_cfg["scalar"]:
        candidate_xyxy = xywh2xyxy(forward.decoded_prediction[sample_idx, candidate_indices, :4].float())
        candidate_source_points = forward.source_points[candidate_indices].to(device=candidate_xyxy.device, dtype=torch.float32)
        final_xyxy = item["box"][:4].to(device=candidate_xyxy.device, dtype=torch.float32)
        scalar_terms["bbox_loss"] = _candidate_bbox_offset_loss(
            candidate_xyxy,
            candidate_source_points,
            final_xyxy,
            layer_cfg["bbox_loss"],
        )
    if "cls_loss" in layer_cfg["scalar"]:
        cls_logits = forward.raw_logits[sample_idx, candidate_indices].to(device=device, dtype=torch.float32)
        cls_target = torch.zeros_like(cls_logits)
        cls_target[:, int(item["raw_class_idx"])] = 1.0
        scalar_terms["cls_loss"] = _class_loss_tensor(
            cls_logits,
            cls_target,
            class_idx=int(item["raw_class_idx"]),
            mode=layer_cfg["cls_loss"],
            direction=layer_cfg["cls_direction"],
            reduction="sum",
        )
    return scalar_terms


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    layer_cfg = parsed["layer_grad"]
    target_mode = layer_cfg["target"]
    detector, device = build_detector(config)
    params_by_scalar = {}
    for scalar_name in layer_cfg["scalar"]:
        layers = expand_layer_names(detector.model, layer_cfg["layers_by_scalar"][scalar_name])
        params_by_scalar[scalar_name] = [(layer_name, resolve_layer_parameter(detector.model, layer_name)) for layer_name in layers]
    reductions = layer_cfg["reduction"]
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"]
    for scalar_name in layer_cfg["scalar"]:
        for layer_name, _param in params_by_scalar[scalar_name]:
            safe = layer_name.replace(".", "_")
            for metric in reductions:
                fieldnames.append(f"{safe}_{scalar_name}_{metric}")
    output_csv = run_dir / "layer_grad.csv"
    dataloader = create_dataloader(config, split=split)
    stages = ["detector_inference_sec"]
    if target_mode == "cand_target":
        stages.append("candidate_search_sec")
    stages.extend(["loss_compute_sec", "backpropagation_sec", "feature_compute_sec"])
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=stages, device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            forward = run_yolov10_forward(detector, infer_batch, timing=timing, grad=True)
            candidate_search_sec = 0.0
            loss_compute_sec = 0.0
            backpropagation_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            all_items = list(iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device))
            contexts = []
            if target_mode == "cand_target":
                candidate_caches = {}
                for sample_idx in range(len(image_list)):
                    t_candidate = timing.start()
                    candidate_caches[sample_idx] = build_yolov10_candidate_cache(forward, sample_idx)
                    candidate_search_sec += timing.elapsed(t_candidate)
                for item in all_items:
                    cache = candidate_caches[item["sample_idx"]]
                    t_candidate = timing.start()
                    cand_mask, _ious = yolov10_candidate_mask_from_cache(
                        cache,
                        item["box"][:4],
                        item["raw_class_idx"],
                        layer_cfg["cand_score_threshold"],
                        layer_cfg["cand_iou_threshold"],
                    )
                    candidate_search_sec += timing.elapsed(t_candidate)
                    contexts.append({"item": item, "candidate_indices": torch.where(cand_mask)[0]})
            else:
                contexts = [{"item": item, "candidate_indices": None} for item in all_items]
            total_grad_calls = 0
            for ctx in contexts:
                if target_mode == "cand_target" and ctx["candidate_indices"].numel() == 0:
                    continue
                total_grad_calls += len(layer_cfg["scalar"])
            grad_call_idx = 0
            for ctx in contexts:
                item = ctx["item"]
                row = dict(item["base_row"])
                if target_mode == "cand_target" and ctx["candidate_indices"].numel() == 0:
                    _zero_grad_features(row, params_by_scalar, layer_cfg["scalar"], reductions)
                    writer.writerow(row)
                    batch_items += 1
                    continue
                t_loss = timing.start()
                scalar_terms = (
                    _build_candidate_scalar_terms(forward, item, ctx["candidate_indices"], layer_cfg, device)
                    if target_mode == "cand_target"
                    else _build_null_scalar_terms(forward, item, layer_cfg, device)
                )
                loss_compute_sec += timing.elapsed(t_loss)
                for scalar_name, target_scalar in scalar_terms.items():
                    grad_call_idx += 1
                    t_back = timing.start()
                    grads = torch.autograd.grad(
                        target_scalar,
                        [param for _layer_name, param in params_by_scalar[scalar_name]],
                        retain_graph=grad_call_idx < total_grad_calls,
                    )
                    backpropagation_sec += timing.elapsed(t_back)
                    t_feature = timing.start()
                    _write_grad_features(row, params_by_scalar[scalar_name], grads, scalar_name, reductions)
                    feature_compute_sec += timing.elapsed(t_feature)
                writer.writerow(row)
                batch_items += 1
            stage_seconds = {
                "detector_inference_sec": forward.detector_inference_sec,
                "loss_compute_sec": loss_compute_sec,
                "backpropagation_sec": backpropagation_sec,
                "feature_compute_sec": feature_compute_sec,
            }
            if target_mode == "cand_target":
                stage_seconds["candidate_search_sec"] = candidate_search_sec
            timing.record(len(image_list), batch_items, stage_seconds)
            del infer_batch, forward, all_items, contexts
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_layer_grad_csv"]
