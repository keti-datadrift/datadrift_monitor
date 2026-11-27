from pathlib import Path

import csv
import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import gather_yolov10_feature_matrix
from commands.predict.yolov10.forward import run_yolov10_forward, run_yolov10_raw_forward
from commands.predict.yolov10.rows import (
    iter_yolov10_detection_rows,
)
from commands.utils.predict_utils import build_detector, resolve_project_path


def _resolve_path(path):
    p = Path(path)
    if not p.is_absolute():
        p = resolve_project_path(p)
    return p


def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    weights = config.get("output", {}).get("ensemble", {}).get("weights", [])
    if not weights:
        raise ValueError("output.ensemble.weights must contain at least one weight.")
    model_weight = config.get("model", {}).get("weights", "")
    if _resolve_path(weights[0]) != _resolve_path(model_weight):
        raise ValueError("YOLOv10 ensemble requires output.ensemble.weights[0] == model.weights.")
    dataloader = create_dataloader(config, split=split)
    detectors = [build_detector(config, model_weight=w)[0] for w in weights]
    device = next(detectors[0].parameters()).device
    num_classes = len(detectors[0].names) if detectors[0].names is not None else 80
    base_architecture = getattr(detectors[0], "architecture", None)
    for detector in detectors[1:]:
        if getattr(detector, "architecture", None) != base_architecture:
            raise ValueError(
                "YOLOv10 ensemble architecture mismatch: "
                f"{getattr(detector, 'architecture', None)} != {base_architecture}"
            )
        other_classes = len(detector.names) if detector.names is not None else 80
        if other_classes != num_classes:
            raise ValueError(f"YOLOv10 ensemble class count mismatch: {other_classes} != {num_classes}")
    fieldnames = ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean", "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std"]
    for class_idx in range(num_classes):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")
    output_csv = run_dir / "ensemble.csv"
    timing = StageTimingProfiler(run_dir=run_dir, uncertainty=uncertainty, unit=parsed["unit"], stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"], device=device)
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detectors[0], image_list, device, auto=False)
            detector_inference_sec = 0.0
            feature_compute_sec = 0.0
            with torch.no_grad():
                t_detector = timing.start()
                base_cache = detectors[0].prepare_feature_cache(infer_batch)
                base = run_yolov10_forward(detectors[0], feature_cache=base_cache)
                source_points = base.source_points
                detector_inference_sec += timing.elapsed(t_detector)
            base_items = list(iter_yolov10_detection_rows(detectors[0], targets, base.selected_preds, base.selected_indices, device))
            feat_dim = 4 + num_classes
            feat_sum = torch.zeros((len(base_items), feat_dim), dtype=torch.float32, device=device)
            feat_sumsq = torch.zeros_like(feat_sum)
            for detector in detectors:
                with torch.no_grad():
                    if detector is detectors[0]:
                        forward = base
                    else:
                        t_detector = timing.start()
                        feature_cache = detector.prepare_feature_cache(infer_batch)
                        forward = run_yolov10_raw_forward(
                            detector,
                            feature_cache=feature_cache,
                            source_points=source_points,
                        )
                        detector_inference_sec += timing.elapsed(t_detector)
                if detector is not detectors[0]:
                    if forward.decoded_prediction.shape != base.decoded_prediction.shape:
                        raise ValueError(
                            "YOLOv10 ensemble raw output shape mismatch: "
                            f"{tuple(forward.decoded_prediction.shape)} != {tuple(base.decoded_prediction.shape)}"
                        )
                t_feature = timing.start()
                feat = gather_yolov10_feature_matrix(forward, base_items, device)
                feat_sum += feat
                feat_sumsq += feat.pow(2)
                feature_compute_sec += timing.elapsed(t_feature)
            t_feature = timing.start()
            n = float(len(detectors))
            feat_mean = feat_sum / n
            feat_var = (feat_sumsq / n - feat_mean.pow(2)).clamp(min=0)
            feat_std = feat_var.sqrt()
            feature_compute_sec += timing.elapsed(t_feature)
            t_match = timing.start()
            batch_items = 0
            for item_idx, item in enumerate(base_items):
                raw_class_idx = item["raw_class_idx"]
                mean_vec = feat_mean[item_idx].detach().cpu()
                std_vec = feat_std[item_idx].detach().cpu()
                score_offset = 4 + raw_class_idx
                row = dict(item["base_row"])
                row.update(
                    {
                        "xmin_mean": float(mean_vec[0].item()),
                        "ymin_mean": float(mean_vec[1].item()),
                        "xmax_mean": float(mean_vec[2].item()),
                        "ymax_mean": float(mean_vec[3].item()),
                        "score_mean": float(mean_vec[score_offset].item()),
                        "xmin_std": float(std_vec[0].item()),
                        "ymin_std": float(std_vec[1].item()),
                        "xmax_std": float(std_vec[2].item()),
                        "ymax_std": float(std_vec[3].item()),
                        "score_std": float(std_vec[score_offset].item()),
                    }
                )
                for class_idx in range(num_classes):
                    offset = 4 + class_idx
                    row[f"prob_{class_idx}_mean"] = float(mean_vec[offset].item()) if offset < mean_vec.numel() else 0.0
                    row[f"prob_{class_idx}_std"] = float(std_vec[offset].item()) if offset < std_vec.numel() else 0.0
                writer.writerow(row)
                batch_items += 1
            prediction_matching_sec = timing.elapsed(t_match)
            timing.record(len(image_list), batch_items, {"detector_inference_sec": detector_inference_sec, "prediction_matching_sec": prediction_matching_sec, "feature_compute_sec": feature_compute_sec})
            del infer_batch, base_cache, base, base_items, feat_sum, feat_sumsq, feat_mean, feat_std
    for detector in detectors:
        del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_ensemble_csv"]
