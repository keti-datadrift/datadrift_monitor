import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, create_dataloader
from commands.predict.faster_rcnn.candidates import (
    build_faster_rcnn_roi_candidate_cache,
    faster_rcnn_roi_candidate_mask_from_cache,
)
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.features import (
    build_meta_feature_values,
    meta_feature_names,
    selected_probs_from_cache,
    tensor_to_float,
)
from commands.predict.faster_rcnn.forward import run_faster_rcnn_forward
from commands.predict.faster_rcnn.rows import iter_faster_rcnn_detection_rows
from commands.utils.predict_utils import build_detector


def run_meta_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "meta_detect"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    score_threshold = float(parsed["meta_detect_score_threshold"])
    iou_threshold = float(parsed["meta_detect_iou_threshold"])
    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
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
        *output_feature_names,
        *meta_feature_names(),
    ]
    output_csv = run_dir / "meta_detect.csv"
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "candidate_search_sec", "feature_compute_sec"],
        device=device,
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)):
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
            result = run_faster_rcnn_forward(detector, image_list, device, timing, score_threshold=score_threshold)
            candidate_search_sec = 0.0
            feature_compute_sec = 0.0
            batch_items = 0
            candidate_caches = {}
            for sample_idx in range(len(image_list)):
                t_candidate = timing.start()
                logits = result.raw_logits[sample_idx] if result.raw_logits is not None and sample_idx < len(result.raw_logits) else None
                candidate_caches[sample_idx] = build_faster_rcnn_roi_candidate_cache(
                    result.raw_prediction[sample_idx],
                    logits,
                    detach=True,
                )
                candidate_search_sec += timing.elapsed(t_candidate)

            for det_row in iter_faster_rcnn_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                cache = candidate_caches[det_row.sample_idx]
                t_candidate = timing.start()
                cand_mask, ious = faster_rcnn_roi_candidate_mask_from_cache(
                    cache,
                    det_row.box[:4],
                    det_row.cls_idx,
                    score_threshold,
                    iou_threshold,
                )
                candidate_search_sec += timing.elapsed(t_candidate)

                t_feature = timing.start()
                cand_boxes = cache.raw_xyxy[cand_mask]
                cand_scores = cache.scores[cand_mask]
                cand_ious = ious[cand_mask]
                pred_probs = selected_probs_from_cache(cache, det_row.raw_pred_idx, num_classes, device)
                prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)}
                for prob_idx in range(max(0, num_classes)):
                    prob_values[f"prob_{prob_idx}"] = (
                        pred_probs[prob_idx]
                        if prob_idx < int(pred_probs.shape[0])
                        else torch.zeros((), dtype=torch.float32, device=device)
                    )
                feature_row = build_meta_feature_values(cand_boxes, cand_scores, cand_ious, det_row.box[:4], device)
                feature_compute_sec += timing.elapsed(t_feature)

                writer.writerow(
                    {
                        **det_row.base,
                        **{key: tensor_to_float(value) for key, value in prob_values.items()},
                        **{key: tensor_to_float(value) for key, value in feature_row.items()},
                    }
                )
                batch_items += 1

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": result.detector_inference_sec,
                    "candidate_search_sec": candidate_search_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del result, candidate_caches

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_meta_detect_csv"]
