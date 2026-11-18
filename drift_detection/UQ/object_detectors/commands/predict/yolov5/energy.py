from commands.predict.common import *
from commands.predict.yolov5.utils import (
    iter_yolo_detection_rows,
    run_yolo_forward_nms,
    selected_class_logits,
)

def run_energy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "energy"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]

    if not save_csv:
        return

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "feature_compute_sec"],
        device=device,
    )
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "energy.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "energy",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolo_forward_nms(detector, infer_batch, nms_kwargs, timing=timing)
            detector_inference_sec = forward.detector_inference_sec

            t_feature = timing.start()
            batch_items = 0
            energy_by_sample = {}
            for sample_idx in range(len(image_list)):
                raw_keep_b = (
                    forward.selected_indices[sample_idx]
                    if forward.selected_indices and sample_idx < len(forward.selected_indices)
                    else torch.zeros((0,), dtype=torch.long, device=device)
                )
                selected_logits = selected_class_logits(
                    forward.raw_logits, forward.raw_prediction, sample_idx, raw_keep_b, num_classes, device
                )
                if selected_logits.numel():
                    pred_energy = -100.0 * torch.log(torch.clamp(torch.sum(torch.exp(selected_logits / 100.0), dim=-1), min=1e-8))
                else:
                    pred_energy = torch.zeros((0,), device=device)
                energy_by_sample[sample_idx] = pred_energy
            for item in iter_yolo_detection_rows(
                detector, targets, forward.selected_preds, forward.selected_indices, device
            ):
                batch_items += 1
                pred_energy = energy_by_sample[item["sample_idx"]]
                pred_idx = item["pred_idx"]
                row = dict(item["base_row"])
                row["energy"] = float(pred_energy[pred_idx].detach().cpu().item()) if pred_idx < pred_energy.shape[0] else 0.0
                writer.writerow(row)
            feature_compute_sec = timing.elapsed(t_feature)
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del infer_batch, forward

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_energy_csv"]
