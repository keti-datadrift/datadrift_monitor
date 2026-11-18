import csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.layer_grad_utils import collect_faster_rcnn_roi_layer_grads_per_target
from commands.utils.predict_utils import build_detector, expand_layer_names, map_grad_tensor_to_numbers


def _safe_npz_key(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _gradient_to_np_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _faster_rcnn_target_layer_map(target_values, target_layers):
    defaults = {
        "roi_cls_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.cls_score"],
        "roi_bbox_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.bbox_pred"],
        "rpn_obj_loss": ["rpn.head.conv", "rpn.head.cls_logits"],
        "rpn_bbox_loss": ["rpn.head.conv", "rpn.head.bbox_pred"],
        "bbox_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.bbox_pred"],
        "obj_loss": ["rpn.head.conv", "rpn.head.cls_logits"],
        "cls_loss": ["roi_heads.box_head.fc7", "roi_heads.box_predictor.cls_score"],
    }
    target_layer_map = {}
    for target_value in target_values:
        if target_value not in defaults:
            target_layer_map[target_value] = []
            continue
        if target_value in {"roi_cls_loss", "cls_loss"}:
            selected = [
                layer
                for layer in target_layers
                if layer.endswith("box_head.fc7") or layer.endswith("box_predictor.cls_score")
            ]
        elif target_value in {"roi_bbox_loss", "bbox_loss"}:
            selected = [
                layer
                for layer in target_layers
                if layer.endswith("box_head.fc7") or layer.endswith("box_predictor.bbox_pred")
            ]
        elif target_value in {"rpn_obj_loss", "obj_loss"}:
            selected = [
                layer
                for layer in target_layers
                if layer.endswith("rpn.head.conv") or layer.endswith("rpn.head.cls_logits")
            ]
        else:
            selected = [
                layer
                for layer in target_layers
                if layer.endswith("rpn.head.conv") or layer.endswith("rpn.head.bbox_pred")
            ]
        target_layer_map[target_value] = selected or defaults[target_value]
    return target_layer_map


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = [str(v) for v in parsed["layer_target_values"]]
    target_layers = parsed["layer_target_layers"]
    configured_target_layer_map = parsed.get("layer_target_layer_map", {})
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_gradient_reduction = parsed["layer_gradient_reduction"]
    layer_pseudo_gt = parsed.get("layer_pseudo_gt", "cand")
    layer_cand_score_threshold = float(parsed.get("layer_cand_score_threshold", 0.01))
    layer_roi_cand_scalar = [str(v) for v in parsed.get("layer_roi_cand_scalar", [])]
    layer_rpn_cand_scalar = [str(v) for v in parsed.get("layer_rpn_cand_scalar", [])]
    layer_roi_null_scalar = [str(v) for v in parsed.get("layer_roi_null_scalar", [])]
    layer_rpn_null_scalar = [str(v) for v in parsed.get("layer_rpn_null_scalar", [])]
    layer_bbox_loss = parsed.get("layer_bbox_loss", "box_l1")
    layer_cls_loss = parsed.get("layer_cls_loss", "bcewithlogits")
    layer_obj_loss = parsed.get("layer_obj_loss", "bcewithlogits")
    layer_bbox_direction = parsed.get("layer_bbox_direction", "pred_to_target")
    layer_cls_direction = parsed.get("layer_cls_direction", "pred_to_target")
    layer_obj_direction = parsed.get("layer_obj_direction", "pred_to_target")
    layer_roi_cand_enabled = bool(parsed.get("layer_roi_cand_enabled", True))
    layer_roi_cand_score_threshold = float(parsed.get("layer_roi_cand_score_threshold", layer_cand_score_threshold))
    layer_roi_bbox_loss = parsed.get("layer_roi_bbox_loss", layer_bbox_loss)
    layer_roi_cls_loss = parsed.get("layer_roi_cls_loss", layer_cls_loss)
    layer_roi_bbox_direction = parsed.get("layer_roi_bbox_direction", layer_bbox_direction)
    layer_roi_cls_direction = parsed.get("layer_roi_cls_direction", layer_cls_direction)
    layer_rpn_cand_enabled = bool(parsed.get("layer_rpn_cand_enabled", False))
    layer_rpn_cand_obj_threshold = float(parsed.get("layer_rpn_cand_obj_threshold", 0.0))
    layer_rpn_bbox_loss = parsed.get("layer_rpn_bbox_loss", "offset_l1")
    layer_rpn_obj_loss = parsed.get("layer_rpn_obj_loss", "bcewithlogits")
    layer_rpn_bbox_direction = parsed.get("layer_rpn_bbox_direction", "pred_to_target")
    layer_rpn_obj_direction = parsed.get("layer_rpn_obj_direction", "pred_to_target")
    layer_roi_null_enabled = bool(parsed.get("layer_roi_null_enabled", True))
    layer_roi_null_bbox_loss = parsed.get("layer_roi_null_bbox_loss", "l1")
    layer_roi_null_cls_loss = parsed.get("layer_roi_null_cls_loss", "bcewithlogits")
    layer_roi_null_bbox_direction = parsed.get("layer_roi_null_bbox_direction", "pred_to_target")
    layer_roi_null_cls_direction = parsed.get("layer_roi_null_cls_direction", "pred_to_target")
    layer_rpn_null_enabled = bool(parsed.get("layer_rpn_null_enabled", True))
    layer_rpn_null_bbox_loss = parsed.get("layer_rpn_null_bbox_loss", "offset_l1")
    layer_rpn_null_obj_loss = parsed.get("layer_rpn_null_obj_loss", "bcewithlogits")
    layer_rpn_null_bbox_direction = parsed.get("layer_rpn_null_bbox_direction", "pred_to_target")
    layer_rpn_null_obj_direction = parsed.get("layer_rpn_null_obj_direction", "pred_to_target")
    layer_frcnn_null_scalar = [str(v) for v in parsed.get("layer_frcnn_null_scalar", [])]
    layer_null_scalar = [str(v) for v in parsed.get("layer_null_scalar", [])]
    layer_null_bbox_loss = parsed.get("layer_null_bbox_loss", "l1")
    layer_null_cls_loss = parsed.get("layer_null_cls_loss", "bcewithlogits")
    layer_null_obj_loss = parsed.get("layer_null_obj_loss", "bcewithlogits")
    layer_null_bbox_direction = parsed.get("layer_null_bbox_direction", "pred_to_target")
    layer_null_cls_direction = parsed.get("layer_null_cls_direction", "pred_to_target")
    layer_null_obj_direction = parsed.get("layer_null_obj_direction", "pred_to_target")

    if not save_csv:
        return

    output_csv = run_dir / "layer_grad.csv"
    save_raw_gradients = not layer_gradient_reduction
    gradients_dir = run_dir / "gradients"
    if save_raw_gradients:
        gradients_dir.mkdir(parents=True, exist_ok=True)

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    is_faster_rcnn = bool(getattr(detector, "is_faster_rcnn", False))
    if (not is_faster_rcnn) and layer_pseudo_gt == "uniform":
        layer_bbox_loss = parsed.get("layer_null_bbox_loss", layer_bbox_loss)
        layer_cls_loss = parsed.get("layer_null_cls_loss", layer_cls_loss)
        layer_obj_loss = parsed.get("layer_null_obj_loss", layer_obj_loss)
        layer_bbox_direction = parsed.get("layer_null_bbox_direction", layer_bbox_direction)
        layer_cls_direction = parsed.get("layer_null_cls_direction", layer_cls_direction)
        layer_obj_direction = parsed.get("layer_null_obj_direction", layer_obj_direction)
    if is_faster_rcnn:
        target_mode = "frcnn_null" if layer_pseudo_gt == "frcnn_null" else "cand"
        alias = {
            "bbox_loss": "roi_bbox_loss",
            "cls_loss": "roi_cls_loss",
            "obj_loss": "rpn_obj_loss",
        }
        nested_target_values = []
        if target_mode == "cand" and layer_rpn_cand_enabled:
            nested_target_values.extend(layer_rpn_cand_scalar)
        if target_mode == "cand" and layer_roi_cand_enabled:
            nested_target_values.extend(layer_roi_cand_scalar)
        if target_mode == "frcnn_null":
            nested_target_values.extend(layer_frcnn_null_scalar)
        if nested_target_values:
            target_values = list(dict.fromkeys(nested_target_values))
        else:
            target_values = list(dict.fromkeys(alias.get(v, v) for v in target_values))
        allowed = (
            {"roi_bbox_loss", "roi_cls_loss", "rpn_bbox_loss", "rpn_obj_loss"}
        )
        target_values = [v for v in target_values if v in allowed]
        if not target_values:
            target_values = (
                ["roi_bbox_loss", "roi_cls_loss"]
                if target_mode == "cand"
                else ["rpn_obj_loss", "rpn_bbox_loss", "roi_cls_loss", "roi_bbox_loss"]
                if target_mode == "frcnn_null"
                else ["roi_bbox_loss", "roi_cls_loss"]
            )
        layer_roi_cand_enabled = layer_roi_null_enabled = any(v.startswith("roi_") for v in target_values)
        layer_rpn_cand_enabled = layer_rpn_null_enabled = any(v.startswith("rpn_") for v in target_values)
    timing_stages = [
        "detector_inference_sec",
        "loss_compute_sec",
        "backpropagation_sec",
        "feature_compute_sec",
    ]
    if (not is_faster_rcnn) or target_mode == "cand":
        timing_stages.insert(1, "candidate_search_sec")

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=timing_stages,
        device=device,
    )
    target_layers = expand_layer_names(detector.model, target_layers)
    target_layer_map = None
    if is_faster_rcnn:
        if configured_target_layer_map:
            default_target_layer_map = _faster_rcnn_target_layer_map(target_values, target_layers)
            target_layer_map = {
                target_value: expand_layer_names(
                    detector.model,
                    configured_target_layer_map.get(target_value, []) or default_target_layer_map.get(target_value, []),
                )
                for target_value in target_values
            }
        else:
            target_layer_map = _faster_rcnn_target_layer_map(target_values, target_layers)

    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
    ]
    for target_value in target_values:
        layers_for_target = target_layer_map.get(target_value, []) if target_layer_map is not None else target_layers
        for layer_name in layers_for_target:
            grad_key = f"{target_value}_{layer_name}"
            if save_raw_gradients:
                fieldnames.append(grad_key)
            else:
                fieldnames.extend(f"{grad_key}_{metric}" for metric in layer_gradient_reduction)

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file_handle:
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for batch_idx, (images, targets) in enumerate(tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        )):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            stage_seconds = {
                "detector_inference_sec": 0.0,
                "loss_compute_sec": 0.0,
                "backpropagation_sec": 0.0,
                "feature_compute_sec": 0.0,
            }
            if "candidate_search_sec" in timing_stages:
                stage_seconds["candidate_search_sec"] = 0.0
            batch_items = 0
            batch_csv_rows = []
            batch_grad_arrays = {}
            if save_raw_gradients:
                npz_name = f"layer_grad_batch_{batch_idx:06d}.npz"
                npz_rel_path = (Path("gradients") / npz_name).as_posix()
                npz_path = gradients_dir / npz_name

            if is_faster_rcnn:
                bbox_rows = collect_faster_rcnn_roi_layer_grads_per_target(
                    detector=detector,
                    input_tensor=infer_batch,
                    target_values=target_values,
                    target_layers=target_layers,
                    target_layer_map=target_layer_map,
                    map_reduction=layer_map_reduction,
                    vector_reduction=layer_gradient_reduction,
                    target_mode=target_mode,
                    roi_cand_enabled=layer_roi_cand_enabled,
                    roi_cand_score_threshold=layer_roi_cand_score_threshold,
                    roi_bbox_loss=layer_roi_bbox_loss,
                    roi_cls_loss=layer_roi_cls_loss,
                    roi_bbox_direction=layer_roi_bbox_direction,
                    roi_cls_direction=layer_roi_cls_direction,
                    rpn_cand_enabled=layer_rpn_cand_enabled,
                    rpn_cand_obj_threshold=layer_rpn_cand_obj_threshold,
                    rpn_bbox_loss=layer_rpn_bbox_loss,
                    rpn_obj_loss=layer_rpn_obj_loss,
                    rpn_bbox_direction=layer_rpn_bbox_direction,
                    rpn_obj_direction=layer_rpn_obj_direction,
                    roi_null_enabled=layer_roi_null_enabled,
                    roi_null_bbox_loss=layer_roi_null_bbox_loss,
                    roi_null_cls_loss=layer_roi_null_cls_loss,
                    roi_null_bbox_direction=layer_roi_null_bbox_direction,
                    roi_null_cls_direction=layer_roi_null_cls_direction,
                    rpn_null_enabled=layer_rpn_null_enabled,
                    rpn_null_bbox_loss=layer_rpn_null_bbox_loss,
                    rpn_null_obj_loss=layer_rpn_null_obj_loss,
                    rpn_null_bbox_direction=layer_rpn_null_bbox_direction,
                    rpn_null_obj_direction=layer_rpn_null_obj_direction,
                    null_bbox_loss=layer_null_bbox_loss,
                    null_cls_loss=layer_null_cls_loss,
                    null_obj_loss=layer_null_obj_loss,
                    null_bbox_direction=layer_null_bbox_direction,
                    null_cls_direction=layer_null_cls_direction,
                    null_obj_direction=layer_null_obj_direction,
                    timing_accumulator=stage_seconds,
                    timing_device=device,
                )
                for bbox_row in bbox_rows:
                    sample_idx = int(bbox_row.get("sample_idx", 0))
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    output_pred_idx = bbox_row["pred_idx"]
                    output_raw_pred_idx = bbox_row["raw_pred_idx"]
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": output_pred_idx,
                        "raw_pred_idx": output_raw_pred_idx,
                        "xmin": bbox_row["xmin"],
                        "ymin": bbox_row["ymin"],
                        "xmax": bbox_row["xmax"],
                        "ymax": bbox_row["ymax"],
                        "score": bbox_row["score"],
                        "pred_class": bbox_row["pred_class"],
                    }
                    for grad_key, grad_value in bbox_row["grad_stats"].items():
                        if save_raw_gradients:
                            array_key = (
                                f"s{sample_idx:03d}_p{int(output_pred_idx):06d}_"
                                f"r{int(output_raw_pred_idx):06d}_{_safe_npz_key(grad_key)}"
                            )
                            batch_grad_arrays[array_key] = _gradient_to_np_array(grad_value)
                            row[grad_key] = f"{npz_rel_path}::{array_key}"
                        else:
                            for metric in layer_gradient_reduction:
                                value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                                row[f"{grad_key}_{metric}"] = _scalar_to_float(value)
                    batch_csv_rows.append(row)
                batch_items += int(len(bbox_rows))
                del bbox_rows
            else:
                bbox_rows = collect_bbox_layer_grads_per_target(
                    detector=detector,
                    input_tensor=infer_batch,
                    target_values=target_values,
                    target_layers=target_layers,
                    map_reduction=layer_map_reduction,
                    vector_reduction=layer_gradient_reduction,
                    pseudo_gt=layer_pseudo_gt,
                    cand_score_threshold=layer_cand_score_threshold,
                    bbox_loss=layer_bbox_loss,
                    cls_loss=layer_cls_loss,
                    obj_loss=layer_obj_loss,
                    bbox_direction=layer_bbox_direction,
                    cls_direction=layer_cls_direction,
                    obj_direction=layer_obj_direction,
                    timing_accumulator=stage_seconds,
                    timing_device=device,
                )
                for bbox_row in bbox_rows:
                    sample_idx = int(bbox_row.get("sample_idx", 0))
                    target = targets[sample_idx]
                    image_id = int(target["image_id"][0].item())
                    image_path = target["path"]
                    output_pred_idx = bbox_row["pred_idx"]
                    output_raw_pred_idx = bbox_row["raw_pred_idx"]
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": output_pred_idx,
                        "raw_pred_idx": output_raw_pred_idx,
                        "xmin": bbox_row["xmin"],
                        "ymin": bbox_row["ymin"],
                        "xmax": bbox_row["xmax"],
                        "ymax": bbox_row["ymax"],
                        "score": bbox_row["score"],
                        "pred_class": bbox_row["pred_class"],
                    }
                    for grad_key, grad_value in bbox_row["grad_stats"].items():
                        if save_raw_gradients:
                            array_key = (
                                f"s{sample_idx:03d}_p{int(output_pred_idx):06d}_"
                                f"r{int(output_raw_pred_idx):06d}_{_safe_npz_key(grad_key)}"
                            )
                            batch_grad_arrays[array_key] = _gradient_to_np_array(grad_value)
                            row[grad_key] = f"{npz_rel_path}::{array_key}"
                        else:
                            for metric in layer_gradient_reduction:
                                value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                                row[f"{grad_key}_{metric}"] = _scalar_to_float(value)
                    batch_csv_rows.append(row)
                batch_items += int(len(bbox_rows))
                del bbox_rows

            if save_raw_gradients and batch_grad_arrays:
                np.savez(npz_path, **batch_grad_arrays)
            for row in batch_csv_rows:
                csv_writer.writerow(row)
            csv_file_handle.flush()

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds=stage_seconds,
            )
            del infer_batch

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    if save_raw_gradients:
        print(f"Saved gradient arrays: {gradients_dir}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_layer_grad_csv"]
