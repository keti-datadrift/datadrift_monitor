from commands.predict.common import *
from commands.predict.fcos.utils import iter_fcos_detection_rows, run_fcos_forward_nms, selected_fcos_class_probs

def run_class_probability_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "class_probability"

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
        stages=["detector_inference_sec"],
        device=device,
    )
    num_classes = len(detector.names) if detector.names is not None else 80
    output_csv = run_dir / "class_probability.csv"
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
    ] + [f"prob_{i}" for i in range(num_classes)]

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

            batch_items = 0
            probs_by_sample = {
                sample_idx: selected_fcos_class_probs(result, sample_idx, num_classes, device)
                for sample_idx in range(len(image_list))
            }
            for det_row in iter_fcos_detection_rows(detector, targets, result.selected_preds, result.selected_indices, device):
                row = dict(det_row.base)
                pred_probs = probs_by_sample[det_row.sample_idx]
                probs = pred_probs[det_row.pred_idx].detach().cpu().tolist() if det_row.pred_idx < int(pred_probs.shape[0]) else [0.0] * num_classes
                for class_idx in range(num_classes):
                    row[f"prob_{class_idx}"] = float(probs[class_idx]) if class_idx < len(probs) else 0.0
                writer.writerow(row)
                batch_items += 1
            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={"detector_inference_sec": result.detector_inference_sec},
            )
            del result, probs_by_sample

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")

__all__ = ["run_class_probability_csv"]
