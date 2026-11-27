import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, _as_image_list, _prepare_infer_batch, create_dataloader
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.features import yolov10_raw_logits_for_item, yolov10_raw_probs_for_item
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import build_detector


def _save_csv_enabled(config, uncertainty):
    output = config.get("output", {})
    cfg = output.get(uncertainty, {}) if isinstance(output, dict) else {}
    save_cfg = cfg.get("save_csv", {}) if isinstance(cfg, dict) else {}
    if isinstance(save_cfg, bool):
        return save_cfg
    return bool(save_cfg.get("enabled", False)) if isinstance(save_cfg, dict) else False


def run_deterministic_uncertainties_csv(config, run_dir, uncertainties=None):
    mode = str(config.get("mode", "predict"))
    supported = ["score", "class_probability", "entropy", "energy"]
    requested = supported if uncertainties is None else [str(v).strip().lower() for v in uncertainties]
    active = [name for name in supported if name in requested and _save_csv_enabled(config, name)]
    if not active:
        return
    if isinstance(run_dir, dict):
        run_dirs = {str(k): Path(v) for k, v in run_dir.items()}
    else:
        base_run_dir = Path(run_dir)
        run_dirs = {name: base_run_dir for name in active}
    for name in active:
        run_dirs[name].mkdir(parents=True, exist_ok=True)
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80
    fieldnames_by_uncertainty = {
        "score": ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"],
        "class_probability": ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"] + [f"prob_{i}" for i in range(num_classes)],
        "entropy": ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "entropy"],
        "energy": ["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "energy"],
    }
    handles = {}
    writers = {}
    for key, fieldnames in fieldnames_by_uncertainty.items():
        if key not in active:
            continue
        path = run_dirs[key] / f"{key}.csv"
        handles[key] = open(path, "w", newline="", encoding="utf-8")
        writers[key] = csv.DictWriter(handles[key], fieldnames=fieldnames)
        writers[key].writeheader()
    profilers = {
        name: StageTimingProfiler(
            run_dir=run_dirs[name],
            uncertainty=name,
            unit=parsed["unit"],
            stages=["detector_inference_sec"],
            device=device,
        )
        for name in active
    }
    try:
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - deterministic)", total=len(dataloader)):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                first_profiler = profilers[active[0]]
                forward = run_yolov10_forward(detector, infer_batch, timing=first_profiler)
            batch_items = 0
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                base = dict(item["base_row"])
                logits = yolov10_raw_logits_for_item(forward, item, device)
                if "score" in writers:
                    writers["score"].writerow(base)
                if "class_probability" in writers:
                    row_prob = dict(base)
                    values = yolov10_raw_probs_for_item(forward, item, device).detach().cpu().tolist()
                    for class_idx in range(num_classes):
                        row_prob[f"prob_{class_idx}"] = float(values[class_idx]) if class_idx < len(values) else 0.0
                    writers["class_probability"].writerow(row_prob)
                if "entropy" in writers:
                    row_entropy = dict(base)
                    soft = torch.softmax(logits, dim=-1)
                    row_entropy["entropy"] = float((-(soft * soft.clamp(min=1e-12).log()).sum()).detach().cpu().item())
                    writers["entropy"].writerow(row_entropy)
                if "energy" in writers:
                    row_energy = dict(base)
                    row_energy["energy"] = float((-torch.logsumexp(logits, dim=-1)).detach().cpu().item())
                    writers["energy"].writerow(row_energy)
                batch_items += 1
            stage_seconds = {"detector_inference_sec": forward.detector_inference_sec}
            for profiler in profilers.values():
                profiler.record(len(image_list), batch_items, stage_seconds)
            del infer_batch, forward
    finally:
        for handle in handles.values():
            handle.close()
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for profiler in profilers.values():
        profiler.save()


__all__ = ["run_deterministic_uncertainties_csv"]
