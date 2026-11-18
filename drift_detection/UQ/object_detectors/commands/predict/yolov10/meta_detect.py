import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import build_yolov10_candidate_cache, yolov10_candidate_mask_from_cache
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import build_detector


def _stats_tensor(values, device):
    if values is None or values.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, zero, zero, zero
    values = values.detach().float().reshape(-1)
    return values.min(), values.max(), values.mean(), values.std(unbiased=False)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def run_meta_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "meta_detect"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"]:
        return
    score_threshold = float(parsed["meta_detect_score_threshold"])
    iou_threshold = float(parsed["meta_detect_iou_threshold"])
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    output_feature_names = ["prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
    meta_feature_names = [
        "num_candidate_boxes",
        "x_min", "x_max", "x_mean", "x_std",
        "y_min", "y_max", "y_mean", "y_std",
        "w_min", "w_max", "w_mean", "w_std",
        "h_min", "h_max", "h_mean", "h_std",
        "size", "size_min", "size_max", "size_mean", "size_std",
        "circum", "circum_min", "circum_max", "circum_mean", "circum_std",
        "size_circum", "size_circum_min", "size_circum_max", "size_circum_mean", "size_circum_std",
        "score_min", "score_max", "score_mean", "score_std",
        "iou_pb_min", "iou_pb_max", "iou_pb_mean", "iou_pb_std",
    ]
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        *output_feature_names,
        *meta_feature_names,
    ]
    output_csv = run_dir / "meta_detect.csv"
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=parsed["unit"],
        stages=["detector_inference_sec", "candidate_search_sec", "feature_compute_sec"],
        device=device,
    )
    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch, timing=timing)
            candidate_search_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            candidate_caches = {}
            for sample_idx in range(len(image_list)):
                t_candidate = timing.start()
                candidate_caches[sample_idx] = build_yolov10_candidate_cache(forward, sample_idx)
                candidate_search_sec += timing.elapsed(t_candidate)
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                cache = candidate_caches[item["sample_idx"]]
                t_candidate = timing.start()
                cand_mask, ious = yolov10_candidate_mask_from_cache(
                    cache,
                    item["box"][:4],
                    item["raw_class_idx"],
                    score_threshold,
                    iou_threshold,
                )
                candidate_search_sec += timing.elapsed(t_candidate)
                t_feature = timing.start()
                cand_boxes = cache.raw_xyxy[cand_mask]
                cand_scores = cache.raw_probs[:, item["raw_class_idx"]][cand_mask]
                cand_ious = ious[cand_mask]
                raw_probs = torch.sigmoid(forward.raw_logits[item["sample_idx"], item["raw_box_idx"]].detach().float())
                prob_values = {"prob_sum": raw_probs.sum()}
                for class_idx in range(num_classes):
                    prob_values[f"prob_{class_idx}"] = raw_probs[class_idx] if class_idx < raw_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)
                x = 0.5 * (cand_boxes[:, 0] + cand_boxes[:, 2])
                y = 0.5 * (cand_boxes[:, 1] + cand_boxes[:, 3])
                w = torch.abs(cand_boxes[:, 2] - cand_boxes[:, 0])
                h = torch.abs(cand_boxes[:, 3] - cand_boxes[:, 1])
                size_vals = w * h
                circum_vals = w + h
                size_circum_vals = size_vals / circum_vals.clamp(min=1e-12)
                iou_pb = torch.where(cand_ious == 1.0, torch.zeros_like(cand_ious), cand_ious)
                iou_pb_pos = iou_pb[iou_pb > 0]
                final_xyxy = item["box"][:4].detach().float()
                fw = torch.abs(final_xyxy[2] - final_xyxy[0])
                fh = torch.abs(final_xyxy[3] - final_xyxy[1])
                fsize = fw * fh
                fcircum = fw + fh
                fsize_circum = fsize / fcircum.clamp(min=1e-12)
                x_min, x_max, x_mean, x_std = _stats_tensor(x, device)
                y_min, y_max, y_mean, y_std = _stats_tensor(y, device)
                w_min, w_max, w_mean, w_std = _stats_tensor(w, device)
                h_min, h_max, h_mean, h_std = _stats_tensor(h, device)
                size_min, size_max, size_mean, size_std = _stats_tensor(size_vals, device)
                circum_min, circum_max, circum_mean, circum_std = _stats_tensor(circum_vals, device)
                size_circum_min, size_circum_max, size_circum_mean, size_circum_std = _stats_tensor(size_circum_vals, device)
                score_min, score_max, score_mean, score_std = _stats_tensor(cand_scores, device)
                iou_pb_min, iou_pb_max, iou_pb_mean, iou_pb_std = _stats_tensor(iou_pb_pos, device)
                feature_row = {
                    "num_candidate_boxes": float(cand_boxes.shape[0]),
                    "x_min": x_min, "x_max": x_max, "x_mean": x_mean, "x_std": x_std,
                    "y_min": y_min, "y_max": y_max, "y_mean": y_mean, "y_std": y_std,
                    "w_min": w_min, "w_max": w_max, "w_mean": w_mean, "w_std": w_std,
                    "h_min": h_min, "h_max": h_max, "h_mean": h_mean, "h_std": h_std,
                    "size": fsize, "size_min": size_min, "size_max": size_max, "size_mean": size_mean, "size_std": size_std,
                    "circum": fcircum, "circum_min": circum_min, "circum_max": circum_max, "circum_mean": circum_mean, "circum_std": circum_std,
                    "size_circum": fsize_circum, "size_circum_min": size_circum_min, "size_circum_max": size_circum_max,
                    "size_circum_mean": size_circum_mean, "size_circum_std": size_circum_std,
                    "score_min": score_min, "score_max": score_max, "score_mean": score_mean, "score_std": score_std,
                    "iou_pb_min": iou_pb_min, "iou_pb_max": iou_pb_max, "iou_pb_mean": iou_pb_mean, "iou_pb_std": iou_pb_std,
                }
                row = dict(item["base_row"])
                row.update({key: _to_float(value) for key, value in prob_values.items()})
                row.update({key: _to_float(value) for key, value in feature_row.items()})
                feature_compute_sec += timing.elapsed(t_feature)
                writer.writerow(row)
                batch_items += 1
            timing.record(
                len(image_list),
                batch_items,
                {
                    "detector_inference_sec": forward.detector_inference_sec,
                    "candidate_search_sec": candidate_search_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del infer_batch, forward, candidate_caches
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing.save()
    print(f"Saved results CSV: {output_csv}")


__all__ = ["run_meta_detect_csv"]
