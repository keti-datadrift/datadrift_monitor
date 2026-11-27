import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, create_dataloader
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.forward import run_faster_rcnn_intermediate_forward
from commands.predict.faster_rcnn.features import (
    faster_rcnn_null_reference_boxes,
    shape_diff_features,
    tensor_to_float,
    zero_shape_diff_features,
)
from commands.predict.faster_rcnn.rows import pred_class_name
from commands.utils.predict_utils import _class_loss_tensor, _objectness_loss_tensor, _resize_boxes_xyxy_tensor, build_detector


def _zero(device, dtype=torch.float32):
    return torch.zeros((), dtype=dtype, device=device)


def _loss_cfg(null_cfg, section, key, default):
    value = null_cfg.get(section, {})
    if isinstance(value, dict) and key in value:
        return value.get(key, default)
    return null_cfg.get(f"{section}_{key}", null_cfg.get(key, default))


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    null_cfg = config.get("output", {}).get("null_detect", {}) or {}
    rpn_obj_loss = str(_loss_cfg(null_cfg, "rpn", "obj_loss", parsed["null_detect_obj_loss"])).strip().lower()
    rpn_obj_direction = str(_loss_cfg(null_cfg, "rpn", "obj_direction", parsed["null_detect_obj_direction"])).strip().lower()
    roi_cls_loss = str(_loss_cfg(null_cfg, "roi", "cls_loss", parsed["null_detect_cls_loss"])).strip().lower()
    roi_cls_direction = str(_loss_cfg(null_cfg, "roi", "cls_direction", parsed["null_detect_cls_direction"])).strip().lower()
    feature_set = parsed["null_detect_feature_set"]

    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not bool(getattr(detector, "is_faster_rcnn", False)):
        raise NotImplementedError("faster_rcnn null_detect runner requires a Faster R-CNN detector.")

    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = [] if feature_set == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
    rpn_xywh_names = ["rpn_x_loss", "rpn_y_loss", "rpn_w_loss", "rpn_h_loss"]
    roi_xywh_names = ["roi_x_loss", "roi_y_loss", "roi_w_loss", "roi_h_loss"]
    rpn_shape_diff_names = ["rpn_size_diff", "rpn_circum_diff", "rpn_size_circum_diff"]
    roi_shape_diff_names = ["roi_size_diff", "roi_circum_diff", "roi_size_circum_diff"]
    null_feature_names = (
        [
            *rpn_xywh_names,
            "rpn_obj_loss",
            *roi_xywh_names,
            "roi_cls_loss",
        ]
        if feature_set == "losses_only"
        else [
            "final_score", "size", "circum", "size_circum",
            *rpn_shape_diff_names,
            *rpn_xywh_names,
            "rpn_obj_loss",
            *roi_shape_diff_names,
            *roi_xywh_names,
            "roi_cls_loss",
        ]
    )
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        *output_feature_names,
        *null_feature_names,
    ]
    output_csv = run_dir / "null_detect.csv"

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "feature_compute_sec"],
        device=device,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
            result = run_faster_rcnn_intermediate_forward(detector, image_list, device, timing)
            feature_compute_sec = 0.0
            batch_items = 0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det_b = result.selected_preds[sample_idx] if result.selected_preds and sample_idx < len(result.selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = result.selected_indices[sample_idx] if result.selected_indices and sample_idx < len(result.selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_img = result.raw_prediction[sample_idx]
                logit_img = result.raw_logits[sample_idx] if result.raw_logits is not None else None
                proposals_img = _resize_boxes_xyxy_tensor(
                    result.proposals[sample_idx],
                    result.transformed_images.image_sizes[sample_idx],
                    result.original_image_sizes[sample_idx],
                )
                proposal_to_rpn_raw_idx_img = (
                    result.proposal_to_rpn_raw_indices[sample_idx]
                    if result.proposal_to_rpn_raw_indices and sample_idx < len(result.proposal_to_rpn_raw_indices)
                    else None
                )
                proposal_indices_img = result.proposal_indices_by_img[sample_idx] if sample_idx < len(result.proposal_indices_by_img) else None
                rpn_objectness_img = result.rpn_objectness_flat[sample_idx]
                rpn_anchors_img = result.rpn_anchors[sample_idx]

                batch_items += int(det_b.shape[0])
                for pred_idx, box in enumerate(det_b):
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item()) if pred_idx < int(raw_keep_b.shape[0]) else pred_idx
                    t_feature = timing.start()
                    prob_values = {}
                    if feature_set != "losses_only":
                        if 0 <= raw_pred_idx < int(pred_img.shape[0]) and pred_img.shape[1] > 6:
                            pred_probs = pred_img[raw_pred_idx, 6:].detach().float()
                        else:
                            pred_probs = torch.zeros((0,), dtype=torch.float32, device=device)
                        prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else _zero(device)}
                        for prob_idx in range(max(0, num_classes)):
                            prob_values[f"prob_{prob_idx}"] = (
                                pred_probs[prob_idx]
                                if prob_idx < int(pred_probs.shape[0])
                                else _zero(device)
                            )

                    shape_values = {}
                    if feature_set != "losses_only":
                        width = torch.abs(box[2] - box[0])
                        height = torch.abs(box[3] - box[1])
                        size = width * height
                        circum = width + height
                        shape_values = {
                            "final_score": box[4],
                            "size": size,
                            "circum": circum,
                            "size_circum": size / circum.clamp(min=1e-12),
                        }
                    rpn_reference, roi_reference, rpn_raw_idx = faster_rcnn_null_reference_boxes(
                        raw_pred_idx=raw_pred_idx,
                        proposal_indices_img=proposal_indices_img,
                        proposal_to_rpn_raw_idx=proposal_to_rpn_raw_idx_img,
                        proposals_xyxy=proposals_img,
                        rpn_anchors=rpn_anchors_img,
                        from_size=result.transformed_images.image_sizes[sample_idx],
                        to_size=result.original_image_sizes[sample_idx],
                        device=device,
                    )
                    rpn_shape_values = shape_diff_features(box[:4], rpn_reference, "rpn", device)
                    roi_shape_values = shape_diff_features(box[:4], roi_reference, "roi", device)

                    rpn_obj_loss_value = _zero(device)
                    if rpn_raw_idx is not None:
                        rpn_objectness_flat = rpn_objectness_img.reshape(-1)
                        if 0 <= int(rpn_raw_idx) < int(rpn_objectness_flat.shape[0]):
                            selected_obj_logit = rpn_objectness_flat[int(rpn_raw_idx)]
                            obj_target = torch.full_like(selected_obj_logit, 0.5)
                            rpn_obj_loss_value = _objectness_loss_tensor(
                                selected_obj_logit,
                                obj_target,
                                mode=rpn_obj_loss,
                                direction=rpn_obj_direction,
                                reduction="sum",
                            )
                    roi_cls_loss_value = _zero(device)
                    if logit_img is not None and 0 <= raw_pred_idx < int(logit_img.shape[0]):
                        cls_logits = logit_img[raw_pred_idx]
                        target_value = 0.5 if str(roi_cls_loss).strip().lower() == "bcewithlogits" else 1.0 / float(max(1, cls_logits.numel()))
                        cls_target = torch.full_like(cls_logits, target_value)
                        roi_cls_loss_value = _class_loss_tensor(
                            cls_logits,
                            cls_target,
                            class_idx=None,
                            mode=roi_cls_loss,
                            direction=roi_cls_direction,
                            reduction="sum",
                        )
                    if rpn_reference is None:
                        rpn_shape_values = zero_shape_diff_features("rpn", device)
                    if roi_reference is None:
                        roi_shape_values = zero_shape_diff_features("roi", device)
                    feature_compute_sec += timing.elapsed(t_feature)

                    cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                    pred_class = pred_class_name(detector, cls_idx)

                    writer.writerow(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_pred_idx,
                            "xmin": tensor_to_float(box[0]),
                            "ymin": tensor_to_float(box[1]),
                            "xmax": tensor_to_float(box[2]),
                            "ymax": tensor_to_float(box[3]),
                            "score": tensor_to_float(box[4]),
                            "pred_class": pred_class,
                            **{key: tensor_to_float(value) for key, value in prob_values.items()},
                            **{key: tensor_to_float(value) for key, value in shape_values.items()},
                            **{
                                key: tensor_to_float(value)
                                for key, value in {**rpn_shape_values, **roi_shape_values}.items()
                                if feature_set != "losses_only" or key in set(rpn_xywh_names + roi_xywh_names)
                            },
                            "rpn_obj_loss": tensor_to_float(rpn_obj_loss_value),
                            "roi_cls_loss": tensor_to_float(roi_cls_loss_value),
                        }
                    )

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": result.detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            output_file.flush()
            del result

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
