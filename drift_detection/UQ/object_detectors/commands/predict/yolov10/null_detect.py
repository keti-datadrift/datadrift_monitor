import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import source_point_box
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import _class_loss_tensor, build_detector


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _xyxy_shape_features(final_xyxy, reference_xyxy):
    final_x = 0.5 * (final_xyxy[0] + final_xyxy[2])
    final_y = 0.5 * (final_xyxy[1] + final_xyxy[3])
    final_w = torch.abs(final_xyxy[2] - final_xyxy[0])
    final_h = torch.abs(final_xyxy[3] - final_xyxy[1])
    ref_x = 0.5 * (reference_xyxy[0] + reference_xyxy[2])
    ref_y = 0.5 * (reference_xyxy[1] + reference_xyxy[3])
    ref_w = torch.abs(reference_xyxy[2] - reference_xyxy[0])
    ref_h = torch.abs(reference_xyxy[3] - reference_xyxy[1])
    final_size = final_w * final_h
    final_circum = final_w + final_h
    final_size_circum = final_size / final_circum.clamp(min=1e-12)
    ref_size = ref_w * ref_h
    ref_circum = ref_w + ref_h
    ref_size_circum = ref_size / ref_circum.clamp(min=1e-12)
    return {
        "size": final_size,
        "circum": final_circum,
        "size_circum": final_size_circum,
        "size_diff": torch.abs(final_size - ref_size),
        "circum_diff": torch.abs(final_circum - ref_circum),
        "size_circum_diff": torch.abs(final_size_circum - ref_size_circum),
        "x_loss": torch.abs(final_x - ref_x),
        "y_loss": torch.abs(final_y - ref_y),
        "w_loss": torch.abs(final_w - ref_w),
        "h_loss": torch.abs(final_h - ref_h),
    }


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    cls_loss = parsed["null_detect_cls_loss"]
    cls_direction = parsed["null_detect_cls_direction"]
    feature_set = parsed["null_detect_feature_set"]
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_feature_names = [] if feature_set == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
    null_feature_names = (
        ["x_loss", "y_loss", "w_loss", "h_loss", "cls_loss"]
        if feature_set == "losses_only"
        else [
            "final_score",
            "size",
            "size_diff",
            "circum",
            "circum_diff",
            "size_circum",
            "size_circum_diff",
            "x_loss",
            "y_loss",
            "w_loss",
            "h_loss",
            "cls_loss",
        ]
    )
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", *output_feature_names, *null_feature_names]
    output_csv = run_dir / "null_detect.csv"
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec", "feature_compute_sec"], device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch, timing=timing)
            feature_compute_sec = 0.0
            batch_items = 0
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                t_feature = timing.start()
                raw_box_idx = item["raw_box_idx"]
                point_box = source_point_box(forward.source_points, raw_box_idx, device)
                final_xyxy = item["box"][:4].detach().float()
                shape = _xyxy_shape_features(final_xyxy, point_box)
                cls_logits = forward.raw_logits[item["sample_idx"], raw_box_idx].to(device=device, dtype=torch.float32)
                target_value = 0.5 if str(cls_loss).strip().lower() == "bcewithlogits" else 1.0 / float(max(1, cls_logits.numel()))
                cls_target = torch.full_like(cls_logits, target_value)
                cls_loss_value = _class_loss_tensor(cls_logits, cls_target, class_idx=None, mode=cls_loss, direction=cls_direction, reduction="sum")
                row = dict(item["base_row"])
                if feature_set != "losses_only":
                    prob_vec = torch.sigmoid(cls_logits)
                    row["prob_sum"] = _to_float(prob_vec.sum())
                    for class_idx in range(num_classes):
                        row[f"prob_{class_idx}"] = _to_float(prob_vec[class_idx]) if class_idx < prob_vec.numel() else 0.0
                    row.update(
                        {
                            "final_score": _to_float(item["box"][4]),
                            "size": _to_float(shape["size"]),
                            "size_diff": _to_float(shape["size_diff"]),
                            "circum": _to_float(shape["circum"]),
                            "circum_diff": _to_float(shape["circum_diff"]),
                            "size_circum": _to_float(shape["size_circum"]),
                            "size_circum_diff": _to_float(shape["size_circum_diff"]),
                        }
                    )
                row.update(
                    {
                        "x_loss": _to_float(shape["x_loss"]),
                        "y_loss": _to_float(shape["y_loss"]),
                        "w_loss": _to_float(shape["w_loss"]),
                        "h_loss": _to_float(shape["h_loss"]),
                        "cls_loss": _to_float(cls_loss_value),
                    }
                )
                feature_compute_sec += timing.elapsed(t_feature)
                writer.writerow(row)
                batch_items += 1
            timing.record(len(image_list), batch_items, {"detector_inference_sec": forward.detector_inference_sec, "feature_compute_sec": feature_compute_sec})
            del infer_batch, forward
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_null_detect_csv"]
