import csv
import queue
import threading
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _mc_dropout_single_csv_writer, _resolve_detector_nms_kwargs, create_dataloader
from commands.predict.faster_rcnn.candidates import build_faster_rcnn_roi_candidate_cache, match_same_class_highest_iou
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.features import roi_feature_vector_from_cache
from commands.predict.faster_rcnn.rows import iter_faster_rcnn_detection_rows
from commands.utils.predict_utils import build_detector


def _feature_dim(class_count):
    return 5 + int(class_count)


def _row_feature_or_base(cache, row, class_count, device):
    matched_idx = match_same_class_highest_iou(row.box[:4], row.cls_idx, cache)
    if matched_idx is None:
        return None
    return roi_feature_vector_from_cache(cache, int(matched_idx.detach().cpu().item()), class_count, device)


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    num_runs = int(parsed["mc_num_runs"])
    dropout_rate = float(parsed["mc_dropout_rate"])
    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if not hasattr(detector, "prepare_roi_cache") or not hasattr(detector, "forward_from_roi_cache"):
        raise NotImplementedError("Faster R-CNN MC-dropout requires ROI cache helpers.")
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    class_count = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 80))
    output_csv = run_dir / "mc_dropout.csv"
    fieldnames = [
        "image_id",
        "image_path",
        "pred_idx",
        "raw_pred_idx",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "score",
        "pred_class",
        "xmin_mean",
        "ymin_mean",
        "xmax_mean",
        "ymax_mean",
        "score_mean",
        "xmin_std",
        "ymin_std",
        "xmax_std",
        "ymax_std",
        "score_std",
    ]
    for class_idx in range(class_count):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )
    write_queue = queue.Queue()
    writer_thread = threading.Thread(target=_mc_dropout_single_csv_writer, args=(write_queue, output_csv, fieldnames), daemon=True)
    writer_thread.start()
    try:
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
            detector_inference_sec = 0.0
            prediction_matching_sec = 0.0
            feature_compute_sec = 0.0

            t_detector = timing.start()
            with torch.no_grad():
                roi_cache = detector.prepare_roi_cache(image_list)
                base_output = detector.forward_from_roi_cache(roi_cache)
                base_raw_prediction = base_output[0] if isinstance(base_output, (tuple, list)) else base_output
                base_raw_logits = base_output[1] if isinstance(base_output, (tuple, list)) and len(base_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=base_raw_prediction,
                    logits=base_raw_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )
            detector_inference_sec += timing.elapsed(t_detector)

            rows = list(iter_faster_rcnn_detection_rows(detector, targets, selected_preds, selected_indices, device))
            sums = [torch.zeros((_feature_dim(class_count),), dtype=torch.float32, device=device) for _ in rows]
            sums_sq = [torch.zeros_like(v) for v in sums]
            counts = [0 for _ in rows]
            base_vectors = []
            for row in rows:
                logits = base_raw_logits[row.sample_idx] if base_raw_logits is not None and row.sample_idx < len(base_raw_logits) else None
                base_cache = build_faster_rcnn_roi_candidate_cache(base_raw_prediction[row.sample_idx], logits, detach=True)
                base_vec = roi_feature_vector_from_cache(base_cache, row.raw_pred_idx, class_count, device)
                if base_vec is None:
                    base_vec = torch.zeros((_feature_dim(class_count),), dtype=torch.float32, device=device)
                    base_vec[:4] = row.box[:4].detach().float().to(device)
                    base_vec[4] = float(row.score)
                base_vectors.append(base_vec)

            detector.set_dropout_rate(dropout_rate)
            try:
                with torch.no_grad():
                    for _run_idx in range(num_runs):
                        t_detector = timing.start()
                        model_output = detector.forward_from_roi_cache(roi_cache)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                        detector_inference_sec += timing.elapsed(t_detector)

                        caches = []
                        t_feature = timing.start()
                        for sample_idx in range(len(image_list)):
                            logits = raw_logits[sample_idx] if raw_logits is not None and sample_idx < len(raw_logits) else None
                            caches.append(build_faster_rcnn_roi_candidate_cache(raw_prediction[sample_idx], logits, detach=True))
                        feature_compute_sec += timing.elapsed(t_feature)

                        t_matching = timing.start()
                        for row_idx, row in enumerate(rows):
                            vec = _row_feature_or_base(caches[row.sample_idx], row, class_count, device)
                            if vec is None:
                                continue
                            sums[row_idx] += vec
                            sums_sq[row_idx] += vec * vec
                            counts[row_idx] += 1
                        prediction_matching_sec += timing.elapsed(t_matching)
            finally:
                detector.set_dropout_rate(0.0)

            batch_rows = []
            t_feature = timing.start()
            for row_idx, row in enumerate(rows):
                if counts[row_idx] > 0:
                    mean_vec = sums[row_idx] / float(counts[row_idx])
                    var_vec = (sums_sq[row_idx] / float(counts[row_idx]) - mean_vec * mean_vec).clamp(min=0.0)
                    std_vec = torch.sqrt(var_vec)
                else:
                    mean_vec = base_vectors[row_idx]
                    std_vec = torch.zeros_like(mean_vec)
                mean_cpu = mean_vec.detach().float().cpu()
                std_cpu = std_vec.detach().float().cpu()
                out = {
                    **row.base,
                    "xmin_mean": float(mean_cpu[0].item()),
                    "ymin_mean": float(mean_cpu[1].item()),
                    "xmax_mean": float(mean_cpu[2].item()),
                    "ymax_mean": float(mean_cpu[3].item()),
                    "score_mean": float(mean_cpu[4].item()),
                    "xmin_std": float(std_cpu[0].item()),
                    "ymin_std": float(std_cpu[1].item()),
                    "xmax_std": float(std_cpu[2].item()),
                    "ymax_std": float(std_cpu[3].item()),
                    "score_std": float(std_cpu[4].item()),
                }
                for class_idx in range(class_count):
                    out[f"prob_{class_idx}_mean"] = float(mean_cpu[5 + class_idx].item())
                    out[f"prob_{class_idx}_std"] = float(std_cpu[5 + class_idx].item())
                batch_rows.append(out)
            feature_compute_sec += timing.elapsed(t_feature)

            write_queue.put(batch_rows)
            timing.record(
                num_images=len(image_list),
                num_predictions=len(rows),
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "prediction_matching_sec": prediction_matching_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del roi_cache, base_output, base_raw_prediction, base_raw_logits, selected_preds, selected_indices, rows
    finally:
        write_queue.put(None)
        writer_thread.join()
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_mc_dropout_csv"]
