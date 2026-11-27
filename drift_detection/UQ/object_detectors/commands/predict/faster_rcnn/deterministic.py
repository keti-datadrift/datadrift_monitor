import csv
from pathlib import Path

import torch
from tqdm import tqdm

from commands.predict.common import StageTimingProfiler, create_dataloader
from commands.predict.faster_rcnn.config import parse_faster_rcnn_output_config
from commands.predict.faster_rcnn.forward import run_faster_rcnn_forward
from commands.predict.faster_rcnn.rows import iter_faster_rcnn_detection_rows
from commands.utils.predict_utils import build_detector


def _save_csv_enabled(config, uncertainty):
    output = config.get("output", {})
    cfg = output.get(uncertainty, {}) if isinstance(output, dict) else {}
    save_cfg = cfg.get("save_csv", {}) if isinstance(cfg, dict) else {}
    if isinstance(save_cfg, bool):
        return save_cfg
    return bool(save_cfg.get("enabled", False)) if isinstance(save_cfg, dict) else False


def _class_tensor(num_rows, num_classes, device):
    return torch.zeros((int(num_rows), int(num_classes)), dtype=torch.float32, device=device)


def _selected_probs_and_logits(result, sample_idx, num_rows, num_classes, device):
    logits = (
        result.selected_logits[sample_idx].to(device=device, dtype=torch.float32)
        if result.selected_logits and sample_idx < len(result.selected_logits)
        else _class_tensor(num_rows, num_classes, device)
    )
    if int(logits.shape[0]) != int(num_rows):
        out = _class_tensor(num_rows, num_classes, device)
        n = min(int(num_rows), int(logits.shape[0]))
        c = min(int(num_classes), int(logits.shape[-1])) if logits.ndim == 2 else 0
        if n > 0 and c > 0:
            out[:n, :c] = logits[:n, :c]
        logits = out
    elif int(logits.shape[-1]) != int(num_classes):
        out = _class_tensor(num_rows, num_classes, device)
        c = min(int(num_classes), int(logits.shape[-1]))
        if c > 0:
            out[:, :c] = logits[:, :c]
        logits = out
    probs = torch.softmax(logits, dim=-1) if logits.numel() else logits
    return probs, logits


def _energy_from_logits(logits):
    if not logits.numel():
        return torch.zeros((0,), dtype=torch.float32, device=logits.device)
    return -100.0 * torch.log(torch.clamp(torch.sum(torch.exp(logits / 100.0), dim=-1), min=1e-8))


def run_deterministic_uncertainties_csv(config, run_dir, uncertainties=None):
    if uncertainties is None:
        uncertainties = ["score", "class_probability", "entropy", "energy"]
    run_dirs = {str(k): Path(v) for k, v in run_dir.items()} if isinstance(run_dir, dict) else {str(u): Path(run_dir) for u in uncertainties}
    mode = str(config.get("mode", "predict"))
    requested = [str(u).strip().lower() for u in uncertainties]
    active = [u for u in ["score", "class_probability", "entropy", "energy"] if u in requested and _save_csv_enabled(config, u)]
    if not active:
        return

    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_faster_rcnn_output_config(config.get("output", {}))
    unit = parsed["unit"]
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 80))

    writers = {}
    files = {}
    outputs = {}
    profilers = {}
    try:
        if "score" in active:
            out_dir = run_dirs["score"]
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs["score"] = out_dir / "score.csv"
            files["score"] = open(outputs["score"], "w", newline="", encoding="utf-8")
            writers["score"] = csv.DictWriter(
                files["score"],
                fieldnames=["image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class"],
            )
            writers["score"].writeheader()
            profilers["score"] = StageTimingProfiler(out_dir, "score", unit, stages=["detector_inference_sec"], device=device)

        if "class_probability" in active:
            out_dir = run_dirs["class_probability"]
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs["class_probability"] = out_dir / "class_probability.csv"
            files["class_probability"] = open(outputs["class_probability"], "w", newline="", encoding="utf-8")
            writers["class_probability"] = csv.DictWriter(
                files["class_probability"],
                fieldnames=[
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
                    *[f"prob_{i}" for i in range(num_classes)],
                ],
            )
            writers["class_probability"].writeheader()
            profilers["class_probability"] = StageTimingProfiler(
                out_dir, "class_probability", unit, stages=["detector_inference_sec"], device=device
            )

        for uncertainty in ("entropy", "energy"):
            if uncertainty not in active:
                continue
            out_dir = run_dirs[uncertainty]
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs[uncertainty] = out_dir / f"{uncertainty}.csv"
            files[uncertainty] = open(outputs[uncertainty], "w", newline="", encoding="utf-8")
            writers[uncertainty] = csv.DictWriter(
                files[uncertainty],
                fieldnames=[
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
                    uncertainty,
                ],
            )
            writers[uncertainty].writeheader()
            profilers[uncertainty] = StageTimingProfiler(
                out_dir, uncertainty, unit, stages=["detector_inference_sec", "feature_compute_sec"], device=device
            )

        first_profiler = next(iter(profilers.values()))
        for images, targets in tqdm(dataloader, desc=f"Object Detector ({mode} - deterministic)", total=len(dataloader)):
            image_list = images if isinstance(images, list) else [images[i] for i in range(images.shape[0])]
            result = run_faster_rcnn_forward(detector, image_list, device, first_profiler)
            batch_items = 0
            feature_secs = {"entropy": 0.0, "energy": 0.0}
            probs_by_sample = {}
            logits_by_sample = {}
            entropy_by_sample = {}
            energy_by_sample = {}
            for sample_idx in range(len(image_list)):
                det = (
                    result.selected_preds[sample_idx]
                    if result.selected_preds and sample_idx < len(result.selected_preds)
                    else torch.zeros((0, 6), dtype=torch.float32, device=device)
                )
                probs, logits = _selected_probs_and_logits(result, sample_idx, int(det.shape[0]), num_classes, device)
                probs_by_sample[sample_idx] = probs
                logits_by_sample[sample_idx] = logits
                if "entropy" in active:
                    t_feature = profilers["entropy"].start()
                    entropy_by_sample[sample_idx] = (
                        -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                        if probs.numel()
                        else torch.zeros((0,), device=device)
                    )
                    feature_secs["entropy"] += profilers["entropy"].elapsed(t_feature)
                if "energy" in active:
                    t_feature = profilers["energy"].start()
                    energy_by_sample[sample_idx] = _energy_from_logits(logits)
                    feature_secs["energy"] += profilers["energy"].elapsed(t_feature)

            for row in iter_faster_rcnn_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                batch_items += 1
                if "score" in active:
                    writers["score"].writerow(row.base)
                if "class_probability" in active:
                    probs = probs_by_sample[row.sample_idx]
                    out = dict(row.base)
                    for class_idx in range(num_classes):
                        out[f"prob_{class_idx}"] = (
                            float(probs[row.pred_idx, class_idx].detach().cpu().item())
                            if row.pred_idx < int(probs.shape[0]) and class_idx < int(probs.shape[1])
                            else 0.0
                        )
                    writers["class_probability"].writerow(out)
                if "entropy" in active:
                    values = entropy_by_sample.get(row.sample_idx)
                    out = dict(row.base)
                    out["entropy"] = (
                        float(values[row.pred_idx].detach().cpu().item())
                        if values is not None and row.pred_idx < int(values.shape[0])
                        else 0.0
                    )
                    writers["entropy"].writerow(out)
                if "energy" in active:
                    values = energy_by_sample.get(row.sample_idx)
                    out = dict(row.base)
                    out["energy"] = (
                        float(values[row.pred_idx].detach().cpu().item())
                        if values is not None and row.pred_idx < int(values.shape[0])
                        else 0.0
                    )
                    writers["energy"].writerow(out)

            for uncertainty, profiler in profilers.items():
                stage_seconds = {"detector_inference_sec": result.detector_inference_sec}
                if uncertainty in feature_secs:
                    stage_seconds["feature_compute_sec"] = feature_secs[uncertainty]
                profiler.record(len(image_list), batch_items, stage_seconds)
            del result
    finally:
        for f in files.values():
            f.close()
        del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for uncertainty, profiler in profilers.items():
        timing_csv, timing_json = profiler.save()
        print(f"Saved results CSV: {outputs[uncertainty]}")
        print(f"Saved timing: {timing_csv}")
        print(f"Saved timing summary: {timing_json}")


__all__ = ["run_deterministic_uncertainties_csv"]
