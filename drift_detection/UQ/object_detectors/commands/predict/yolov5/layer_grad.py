from commands.predict.common import *


def _safe_npz_key(value):
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def _gradient_to_np_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _scalar_to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def run_layer_grad_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "layer_grad"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    target_values = [str(v) for v in parsed["layer_target_values"]]
    target_layers = parsed["layer_target_layers"]
    layer_map_reduction = parsed["layer_map_reduction"]
    layer_gradient_reduction = parsed["layer_gradient_reduction"]
    layer_pseudo_gt = parsed.get("layer_pseudo_gt", "cand")
    layer_cand_score_threshold = float(parsed.get("layer_cand_score_threshold", 0.01))
    layer_bbox_loss = parsed.get("layer_bbox_loss", "box_l1")
    layer_cls_loss = parsed.get("layer_cls_loss", "bcewithlogits")
    layer_obj_loss = parsed.get("layer_obj_loss", "bcewithlogits")
    layer_bbox_direction = parsed.get("layer_bbox_direction", "pred_to_target")
    layer_cls_direction = parsed.get("layer_cls_direction", "pred_to_target")
    layer_obj_direction = parsed.get("layer_obj_direction", "pred_to_target")

    if not save_csv:
        return

    output_csv = run_dir / "layer_grad.csv"
    save_raw_gradients = not layer_gradient_reduction
    gradients_dir = run_dir / "gradients"
    if save_raw_gradients:
        gradients_dir.mkdir(parents=True, exist_ok=True)

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    if layer_pseudo_gt == "uniform":
        layer_bbox_loss = parsed.get("layer_null_bbox_loss", layer_bbox_loss)
        layer_cls_loss = parsed.get("layer_null_cls_loss", layer_cls_loss)
        layer_obj_loss = parsed.get("layer_null_obj_loss", layer_obj_loss)
        layer_bbox_direction = parsed.get("layer_null_bbox_direction", layer_bbox_direction)
        layer_cls_direction = parsed.get("layer_null_cls_direction", layer_cls_direction)
        layer_obj_direction = parsed.get("layer_null_obj_direction", layer_obj_direction)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=[
            "detector_inference_sec",
            "candidate_search_sec",
            "loss_compute_sec",
            "backpropagation_sec",
            "feature_compute_sec",
        ],
        device=device,
    )
    target_layers = expand_layer_names(detector.model, target_layers)

    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
    ]
    for target_value in target_values:
        for layer_name in target_layers:
            grad_key = f"{target_value}_{layer_name}"
            if save_raw_gradients:
                fieldnames.append(grad_key)
            else:
                fieldnames.extend(f"{grad_key}_{metric}" for metric in layer_gradient_reduction)

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file_handle:
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=fieldnames)
        csv_writer.writeheader()

        for batch_idx, (images, targets) in enumerate(tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        )):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            stage_seconds = {
                "detector_inference_sec": 0.0,
                "candidate_search_sec": 0.0,
                "loss_compute_sec": 0.0,
                "backpropagation_sec": 0.0,
                "feature_compute_sec": 0.0,
            }
            batch_items = 0
            batch_csv_rows = []
            batch_grad_arrays = {}
            if save_raw_gradients:
                npz_name = f"layer_grad_batch_{batch_idx:06d}.npz"
                npz_rel_path = (Path("gradients") / npz_name).as_posix()
                npz_path = gradients_dir / npz_name

            bbox_rows = collect_bbox_layer_grads_per_target(
                detector=detector,
                input_tensor=infer_batch,
                target_values=target_values,
                target_layers=target_layers,
                map_reduction=layer_map_reduction,
                vector_reduction=layer_gradient_reduction,
                pseudo_gt=layer_pseudo_gt,
                cand_score_threshold=layer_cand_score_threshold,
                bbox_loss=layer_bbox_loss,
                cls_loss=layer_cls_loss,
                obj_loss=layer_obj_loss,
                bbox_direction=layer_bbox_direction,
                cls_direction=layer_cls_direction,
                obj_direction=layer_obj_direction,
                timing_accumulator=stage_seconds,
                timing_device=device,
            )
            for bbox_row in bbox_rows:
                sample_idx = int(bbox_row.get("sample_idx", 0))
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]
                output_pred_idx = bbox_row["pred_idx"]
                output_raw_pred_idx = bbox_row["raw_pred_idx"]
                row = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "pred_idx": output_pred_idx,
                    "raw_pred_idx": output_raw_pred_idx,
                    "xmin": bbox_row["xmin"],
                    "ymin": bbox_row["ymin"],
                    "xmax": bbox_row["xmax"],
                    "ymax": bbox_row["ymax"],
                    "score": bbox_row["score"],
                    "pred_class": bbox_row["pred_class"],
                }
                for grad_key, grad_value in bbox_row["grad_stats"].items():
                    if save_raw_gradients:
                        array_key = (
                            f"s{sample_idx:03d}_p{int(output_pred_idx):06d}_"
                            f"r{int(output_raw_pred_idx):06d}_{_safe_npz_key(grad_key)}"
                        )
                        batch_grad_arrays[array_key] = _gradient_to_np_array(grad_value)
                        row[grad_key] = f"{npz_rel_path}::{array_key}"
                    else:
                        for metric in layer_gradient_reduction:
                            value = grad_value.get(metric, 0.0) if isinstance(grad_value, dict) else 0.0
                            row[f"{grad_key}_{metric}"] = _scalar_to_float(value)
                batch_csv_rows.append(row)
            batch_items += int(len(bbox_rows))
            del bbox_rows

            if save_raw_gradients and batch_grad_arrays:
                np.savez(npz_path, **batch_grad_arrays)
            for row in batch_csv_rows:
                csv_writer.writerow(row)
            csv_file_handle.flush()

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds=stage_seconds,
            )
            del infer_batch

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    if save_raw_gradients:
        print(f"Saved gradient arrays: {gradients_dir}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_layer_grad_csv"]
