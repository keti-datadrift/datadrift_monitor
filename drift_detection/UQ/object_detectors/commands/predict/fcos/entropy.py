from commands.predict.common import *
from commands.predict.fcos.utils import iter_fcos_detection_rows, run_fcos_forward_nms, selected_fcos_class_probs

def run_entropy_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "entropy"

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
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "feature_compute_sec"],
        device=device,
    )
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "entropy.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class", "entropy",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            result = run_fcos_forward_nms(
                detector=detector,
                image_list=image_list,
                device=device,
                timing=timing,
                keep_pre_nms=False,
                keep_class_outputs=True,
            )

            t_feature = timing.start()
            batch_items = 0
            entropy_by_sample = {}
            for sample_idx in range(len(image_list)):
                selected_probs = selected_fcos_class_probs(result, sample_idx, num_classes, device)
                pred_entropy = -torch.sum(selected_probs * torch.log(selected_probs.clamp(min=1e-12)), dim=-1) if selected_probs.numel() else torch.zeros((0,), device=device)
                entropy_by_sample[sample_idx] = pred_entropy
            for det_row in iter_fcos_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                pred_entropy = entropy_by_sample[det_row.sample_idx]
                row = dict(det_row.base)
                row["entropy"] = float(pred_entropy[det_row.pred_idx].detach().cpu().item()) if det_row.pred_idx < int(pred_entropy.shape[0]) else 0.0
                writer.writerow(row)
                batch_items += 1
            feature_compute_sec = timing.elapsed(t_feature)
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": result.detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            del result, entropy_by_sample

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_entropy_csv"]
