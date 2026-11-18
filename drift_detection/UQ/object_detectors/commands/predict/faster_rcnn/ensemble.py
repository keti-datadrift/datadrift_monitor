import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _resolve_detector_nms_kwargs, create_dataloader
from commands.predict.faster_rcnn.candidates import build_faster_rcnn_roi_candidate_cache, match_same_class_highest_iou
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.features import roi_feature_vector_from_cache
from commands.predict.faster_rcnn.rows import iter_faster_rcnn_detection_rows
from commands.utils.predict_utils import build_detector, resolve_project_path


def _normalize_weight_path(value, key):
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{key} must be a non-empty string for Faster R-CNN ensemble row alignment.")
    return Path(resolve_project_path(text)).resolve()


def _feature_dim(class_count):
    return 5 + int(class_count)


def _run_raw_from_roi_cache(detector, image_list, timing):
    t_detector = timing.start()
    with torch.no_grad():
        roi_cache = detector.prepare_roi_cache(image_list)
        model_output = detector.forward_from_roi_cache(roi_cache)
        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
    del roi_cache
    return raw_prediction, raw_logits, timing.elapsed(t_detector)


def _feature_from_match(cache, row, class_count, device):
    matched_idx = match_same_class_highest_iou(row.box[:4], row.cls_idx, cache)
    if matched_idx is None:
        return None
    return roi_feature_vector_from_cache(cache, int(matched_idx.detach().cpu().item()), class_count, device)


def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    if not save_csv:
        return

    ensemble_cfg = config.get("output", {}).get("ensemble", {})
    weights_cfg = ensemble_cfg.get("weights", [])
    if isinstance(weights_cfg, str):
        weight_paths = [weights_cfg]
    elif isinstance(weights_cfg, (list, tuple)):
        weight_paths = [str(w) for w in weights_cfg if str(w).strip()]
    else:
        weight_paths = []
    if not weight_paths:
        raise ValueError("output.uncertainty='ensemble' requires output.ensemble.weights to be a non-empty string/list.")

    model_weights = config.get("model", {}).get("weights", "")
    if isinstance(model_weights, (list, tuple)):
        raise ValueError("model.weights must be a single weight path when running Faster R-CNN ensemble.")
    model_weight_path = _normalize_weight_path(model_weights, "model.weights")
    first_weight_path = _normalize_weight_path(weight_paths[0], "output.ensemble.weights[0]")
    if model_weight_path != first_weight_path:
        raise ValueError(
            "Faster R-CNN ensemble requires output.ensemble.weights[0] to match model.weights. "
            f"model.weights={model_weight_path}; output.ensemble.weights[0]={first_weight_path}"
        )

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    output_csv = run_dir / "ensemble.csv"
    detectors = []
    device = torch.device("cpu")
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )
    try:
        class_count = None
        for model_weight in weight_paths:
            detector, device = build_detector(config, model_weight=model_weight)
            current_class_count = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 80))
            if class_count is None:
                class_count = current_class_count
            elif int(current_class_count) != int(class_count):
                raise ValueError(f"All ensemble weights must have the same class count: {class_count} vs {current_class_count}.")
            detectors.append(detector)
        timing.device = device
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

        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
                image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
                detector_inference_sec = 0.0
                prediction_matching_sec = 0.0
                feature_compute_sec = 0.0
                base_detector = detectors[0]
                base_raw_prediction, base_raw_logits, elapsed = _run_raw_from_roi_cache(base_detector, image_list, timing)
                detector_inference_sec += elapsed
                nms_kwargs = _resolve_detector_nms_kwargs(base_detector)
                t_detector = timing.start()
                with torch.no_grad():
                    selected_preds, _selected_logits, _selected_objectness, selected_indices = base_detector.non_max_suppression(
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
                rows = list(iter_faster_rcnn_detection_rows(base_detector, targets, selected_preds, selected_indices, device))
                sums = [torch.zeros((_feature_dim(class_count),), dtype=torch.float32, device=device) for _ in rows]
                sums_sq = [torch.zeros_like(v) for v in sums]
                counts = [0 for _ in rows]

                member_outputs = [(base_detector, base_raw_prediction, base_raw_logits)]
                for detector in detectors[1:]:
                    raw_prediction, raw_logits, elapsed = _run_raw_from_roi_cache(detector, image_list, timing)
                    detector_inference_sec += elapsed
                    member_outputs.append((detector, raw_prediction, raw_logits))

                for detector, raw_prediction, raw_logits in member_outputs:
                    t_feature = timing.start()
                    caches = []
                    for sample_idx in range(len(image_list)):
                        logits = raw_logits[sample_idx] if raw_logits is not None and sample_idx < len(raw_logits) else None
                        caches.append(build_faster_rcnn_roi_candidate_cache(raw_prediction[sample_idx], logits, detach=True))
                    feature_compute_sec += timing.elapsed(t_feature)
                    t_matching = timing.start()
                    for row_idx, row in enumerate(rows):
                        if detector is base_detector:
                            vec = roi_feature_vector_from_cache(caches[row.sample_idx], row.raw_pred_idx, class_count, device)
                        else:
                            vec = _feature_from_match(caches[row.sample_idx], row, class_count, device)
                        if vec is None:
                            continue
                        sums[row_idx] += vec
                        sums_sq[row_idx] += vec * vec
                        counts[row_idx] += 1
                    prediction_matching_sec += timing.elapsed(t_matching)

                t_feature = timing.start()
                for row_idx, row in enumerate(rows):
                    if counts[row_idx] <= 0:
                        continue
                    mean_vec = sums[row_idx] / float(counts[row_idx])
                    var_vec = (sums_sq[row_idx] / float(counts[row_idx]) - mean_vec * mean_vec).clamp(min=0.0)
                    std_vec = torch.sqrt(var_vec)
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
                    writer.writerow(out)
                feature_compute_sec += timing.elapsed(t_feature)
                timing.record(
                    num_images=len(image_list),
                    num_predictions=len(rows),
                    stage_seconds={
                        "detector_inference_sec": detector_inference_sec,
                        "prediction_matching_sec": prediction_matching_sec,
                        "feature_compute_sec": feature_compute_sec,
                    },
                )
                del base_raw_prediction, base_raw_logits, selected_preds, selected_indices, rows
    finally:
        for detector in detectors:
            del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_ensemble_csv"]
