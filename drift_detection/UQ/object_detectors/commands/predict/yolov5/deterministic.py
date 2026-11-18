from commands.predict.common import *
from commands.predict.yolov5.utils import (
    run_yolo_forward_nms,
    selected_class_logits,
    selected_sigmoid_class_outputs,
    selected_softmax_class_probs,
    yolo_pred_class_name,
)


def _pred_class_name(detector, cls_idx):
    return yolo_pred_class_name(detector, cls_idx)


def _empty_class_tensor(num_classes, device):
    return torch.zeros((0, num_classes), dtype=torch.float32, device=device)


def _selected_class_probs_and_logits(detector, raw_prediction, raw_logits, sample_idx, raw_keep_b, num_classes, device):
    probs = selected_sigmoid_class_outputs(raw_prediction, sample_idx, raw_keep_b, num_classes, device)
    logits = selected_class_logits(raw_logits, raw_prediction, sample_idx, raw_keep_b, num_classes, device)
    return probs, logits


def _energy_from_probs(probs):
    if not probs.numel():
        return torch.zeros((0,), dtype=torch.float32, device=probs.device)
    probs_clipped = probs.clamp(min=1e-8, max=1.0 - 1e-8)
    pseudo_logits = torch.log(probs_clipped / (1.0 - probs_clipped))
    return -100.0 * torch.log(
        torch.clamp(torch.sum(torch.exp(pseudo_logits / 100.0), dim=-1), min=1e-8)
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

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
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
            class_probability_dir = run_dirs["class_probability"]
            class_probability_dir.mkdir(parents=True, exist_ok=True)
            outputs["class_probability"] = class_probability_dir / "class_probability.csv"
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
                run_dir=class_probability_dir,
                uncertainty="class_probability",
                unit="bbox",
                stages=["detector_inference_sec"],
                device=device,
            )

        for uncertainty in ("entropy", "energy"):
            if uncertainty not in active:
                continue
            uncertainty_dir = run_dirs[uncertainty]
            uncertainty_dir.mkdir(parents=True, exist_ok=True)
            outputs[uncertainty] = uncertainty_dir / f"{uncertainty}.csv"
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
                run_dir=uncertainty_dir,
                uncertainty=uncertainty,
                unit="bbox",
                stages=["detector_inference_sec", "feature_compute_sec"],
                device=device,
            )

        needs_warmup = True
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - deterministic)", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)

            if needs_warmup:
                with torch.no_grad():
                    warmup_output = detector.model(infer_batch, augment=False)
                    warmup_prediction = warmup_output[0] if isinstance(warmup_output, (tuple, list)) else warmup_output
                    warmup_logits = (
                        warmup_output[1]
                        if isinstance(warmup_output, (tuple, list)) and len(warmup_output) > 1
                        else None
                    )
                    warmup_nms_logits = _resolve_nms_logits(warmup_prediction, warmup_logits)
                    detector.non_max_suppression(
                        prediction=warmup_prediction,
                        logits=warmup_nms_logits,
                        conf_thres=nms_kwargs["conf_thres"],
                        iou_thres=nms_kwargs["iou_thres"],
                        classes=nms_kwargs["classes"],
                        agnostic=nms_kwargs["agnostic"],
                        max_det=nms_kwargs["max_det"],
                        return_indices=True,
                    )
                _sync_timing_device(device)
                del warmup_output, warmup_prediction, warmup_logits, warmup_nms_logits
                needs_warmup = False

            with torch.no_grad():
                forward = run_yolo_forward_nms(
                    detector,
                    infer_batch,
                    nms_kwargs,
                    timing=next(iter(profilers.values())),
                    num_classes_hint=num_classes,
                )
            detector_inference_sec = forward.detector_inference_sec

            batch_items = 0
            entropy_feature_sec = 0.0
            energy_feature_sec = 0.0
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                det = (
                    forward.selected_preds[sample_idx]
                    if forward.selected_preds and sample_idx < len(forward.selected_preds)
                    else torch.zeros((0, 6), device=device)
                )
                raw_keep_b = (
                    forward.selected_indices[sample_idx]
                    if forward.selected_indices and sample_idx < len(forward.selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                batch_items += int(det.shape[0])

                selected_probs, selected_logits = _selected_class_probs_and_logits(
                    detector, forward.raw_prediction, forward.raw_logits, sample_idx, raw_keep_b, num_classes, device
                )
                entropy_probs = selected_softmax_class_probs(
                    forward.raw_logits, forward.raw_prediction, sample_idx, raw_keep_b, num_classes, device
                )

                pred_entropy = None
                pred_energy = None
                if "entropy" in active:
                    t_feature = profilers["entropy"].start()
                    pred_entropy = (
                        -torch.sum(entropy_probs * torch.log(entropy_probs.clamp(min=1e-12)), dim=-1)
                        if entropy_probs.numel()
                        else torch.zeros((0,), device=device)
                    )
                    entropy_feature_sec += profilers["entropy"].elapsed(t_feature)
                if "energy" in active:
                    t_feature = profilers["energy"].start()
                    pred_energy = _energy_from_logits(selected_logits)
                    energy_feature_sec += profilers["energy"].elapsed(t_feature)

                for pred_idx, box in enumerate(det):
                    if pred_idx >= int(raw_keep_b.shape[0]):
                        raise RuntimeError(
                            "YOLO deterministic selected_indices is shorter than selected predictions. "
                            f"sample_idx={sample_idx}, pred_idx={pred_idx}, indices={int(raw_keep_b.shape[0])}"
                        )
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item())
                    cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
                    base_row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "pred_idx": pred_idx,
                        "raw_pred_idx": raw_pred_idx,
                        "xmin": float(box[0].detach().cpu().item()),
                        "ymin": float(box[1].detach().cpu().item()),
                        "xmax": float(box[2].detach().cpu().item()),
                        "ymax": float(box[3].detach().cpu().item()),
                        "score": float(box[4].detach().cpu().item()),
                        "pred_class": _pred_class_name(detector, cls_idx),
                    }
                    if "score" in active:
                        writers["score"].writerow(base_row)
                    if "class_probability" in active:
                        row = dict(base_row)
                        for class_idx in range(num_classes):
                            row[f"prob_{class_idx}"] = (
                                float(selected_probs[pred_idx, class_idx].detach().cpu().item())
                                if pred_idx < selected_probs.shape[0] and class_idx < selected_probs.shape[1]
                                else 0.0
                            )
                        writers["class_probability"].writerow(row)
                    if "entropy" in active:
                        row = dict(base_row)
                        row["entropy"] = (
                            float(pred_entropy[pred_idx].detach().cpu().item())
                            if pred_entropy is not None and pred_idx < pred_entropy.shape[0]
                            else 0.0
                        )
                        writers["entropy"].writerow(row)
                    if "energy" in active:
                        row = dict(base_row)
                        row["energy"] = (
                            float(pred_energy[pred_idx].detach().cpu().item())
                            if pred_energy is not None and pred_idx < pred_energy.shape[0]
                            else 0.0
                        )
                        writers["energy"].writerow(row)

            for uncertainty, profiler in profilers.items():
                stage_seconds = {"detector_inference_sec": detector_inference_sec}
                if uncertainty == "entropy":
                    stage_seconds["feature_compute_sec"] = entropy_feature_sec
                elif uncertainty == "energy":
                    stage_seconds["feature_compute_sec"] = energy_feature_sec
                profiler.record(
                    num_images=len(image_list),
                    num_predictions=batch_items,
                    stage_seconds=stage_seconds,
                )
            del infer_batch, forward
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
