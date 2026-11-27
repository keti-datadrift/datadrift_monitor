from commands.predict.common import *
from commands.predict.fcos.utils import (
    iter_fcos_detection_rows,
    run_fcos_forward_nms,
    selected_fcos_class_logits,
    selected_fcos_class_probs,
)


def _energy_from_logits(logits):
    if not logits.numel():
        return torch.zeros((0,), dtype=torch.float32, device=logits.device)
    return -100.0 * torch.log(
        torch.clamp(torch.sum(torch.exp(logits / 100.0), dim=-1), min=1e-8)
    )


def run_deterministic_uncertainties_csv(config, run_dir, uncertainties=None):
    if uncertainties is None:
        uncertainties = ["score", "class_probability", "entropy", "energy"]
    if isinstance(run_dir, dict):
        run_dirs = {str(k): Path(v) for k, v in run_dir.items()}
    else:
        base_run_dir = Path(run_dir)
        run_dirs = {str(u): base_run_dir for u in uncertainties}

    mode = str(config.get("mode", "predict"))
    requested = [str(u).strip().lower() for u in uncertainties]
    supported = ["score", "class_probability", "entropy", "energy"]
    active = [u for u in supported if u in requested]
    if not active:
        return

    split = config.get("dataset", {}).get("split", "val")
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    num_classes = len(detector.names) if detector.names is not None else 80

    writers = {}
    files = {}
    outputs = {}
    profilers = {}
    try:
        if "score" in active:
            score_dir = run_dirs["score"]
            score_dir.mkdir(parents=True, exist_ok=True)
            outputs["score"] = score_dir / "score.csv"
            files["score"] = open(outputs["score"], "w", newline="", encoding="utf-8")
            writers["score"] = csv.DictWriter(
                files["score"],
                fieldnames=[
                    "image_id", "image_path", "pred_idx", "raw_pred_idx",
                    "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
                ],
            )
            writers["score"].writeheader()
            profilers["score"] = StageTimingProfiler(
                run_dir=score_dir, uncertainty="score", unit="bbox", stages=["detector_inference_sec"], device=device
            )

        if "class_probability" in active:
            out_dir = run_dirs["class_probability"]
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs["class_probability"] = out_dir / "class_probability.csv"
            files["class_probability"] = open(outputs["class_probability"], "w", newline="", encoding="utf-8")
            writers["class_probability"] = csv.DictWriter(
                files["class_probability"],
                fieldnames=[
                    "image_id", "image_path", "pred_idx", "raw_pred_idx",
                    "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
                    *[f"prob_{i}" for i in range(num_classes)],
                ],
            )
            writers["class_probability"].writeheader()
            profilers["class_probability"] = StageTimingProfiler(
                run_dir=out_dir,
                uncertainty="class_probability",
                unit="bbox",
                stages=["detector_inference_sec"],
                device=device,
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
                    "image_id", "image_path", "pred_idx", "raw_pred_idx",
                    "xmin", "ymin", "xmax", "ymax", "score", "pred_class", uncertainty,
                ],
            )
            writers[uncertainty].writeheader()
            profilers[uncertainty] = StageTimingProfiler(
                run_dir=out_dir,
                uncertainty=uncertainty,
                unit="bbox",
                stages=["detector_inference_sec", "feature_compute_sec"],
                device=device,
            )

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - deterministic)", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            result = run_fcos_forward_nms(
                detector=detector,
                image_list=image_list,
                device=device,
                timing=next(iter(profilers.values())),
                keep_pre_nms=False,
                keep_class_outputs=True,
            )

            batch_items = 0
            entropy_feature_sec = 0.0
            energy_feature_sec = 0.0
            probs_by_sample = {}
            logits_by_sample = {}
            entropy_by_sample = {}
            energy_by_sample = {}

            if "class_probability" in active or "entropy" in active:
                for sample_idx in range(len(image_list)):
                    probs_by_sample[sample_idx] = selected_fcos_class_probs(result, sample_idx, num_classes, device)
            if "energy" in active:
                for sample_idx in range(len(image_list)):
                    logits_by_sample[sample_idx] = selected_fcos_class_logits(result, sample_idx, num_classes, device)
            if "entropy" in active:
                t_feature = profilers["entropy"].start()
                for sample_idx, probs in probs_by_sample.items():
                    entropy_by_sample[sample_idx] = (
                        -torch.sum(probs * torch.log(probs.clamp(min=1e-12)), dim=-1)
                        if probs.numel()
                        else torch.zeros((0,), device=device)
                    )
                entropy_feature_sec = profilers["entropy"].elapsed(t_feature)
            if "energy" in active:
                t_feature = profilers["energy"].start()
                for sample_idx, logits in logits_by_sample.items():
                    energy_by_sample[sample_idx] = _energy_from_logits(logits)
                energy_feature_sec = profilers["energy"].elapsed(t_feature)

            for det_row in iter_fcos_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                base_row = dict(det_row.base)
                batch_items += 1
                if "score" in active:
                    writers["score"].writerow(base_row)
                if "class_probability" in active:
                    row = dict(base_row)
                    probs = probs_by_sample[det_row.sample_idx]
                    for class_idx in range(num_classes):
                        row[f"prob_{class_idx}"] = (
                            float(probs[det_row.pred_idx, class_idx].detach().cpu().item())
                            if det_row.pred_idx < int(probs.shape[0]) and class_idx < int(probs.shape[1])
                            else 0.0
                        )
                    writers["class_probability"].writerow(row)
                if "entropy" in active:
                    row = dict(base_row)
                    entropy = entropy_by_sample[det_row.sample_idx]
                    row["entropy"] = (
                        float(entropy[det_row.pred_idx].detach().cpu().item())
                        if det_row.pred_idx < int(entropy.shape[0])
                        else 0.0
                    )
                    writers["entropy"].writerow(row)
                if "energy" in active:
                    row = dict(base_row)
                    energy = energy_by_sample[det_row.sample_idx]
                    row["energy"] = (
                        float(energy[det_row.pred_idx].detach().cpu().item())
                        if det_row.pred_idx < int(energy.shape[0])
                        else 0.0
                    )
                    writers["energy"].writerow(row)

            for uncertainty, profiler in profilers.items():
                stage_seconds = {"detector_inference_sec": result.detector_inference_sec}
                if uncertainty == "entropy":
                    stage_seconds["feature_compute_sec"] = entropy_feature_sec
                elif uncertainty == "energy":
                    stage_seconds["feature_compute_sec"] = energy_feature_sec
                profiler.record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_seconds,
                )
            del result, probs_by_sample, logits_by_sample, entropy_by_sample, energy_by_sample
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
