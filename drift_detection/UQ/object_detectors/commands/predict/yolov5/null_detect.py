from commands.predict.common import *
from commands.utils.predict_utils import (
    _class_loss_tensor,
    _flatten_raw_prediction_layers,
    _objectness_loss_tensor,
    _xywh_to_xyxy_tensor,
)
from commands.predict.yolov5.utils import (
    iter_yolo_detection_rows,
    run_yolo_forward_nms,
)


def run_null_detect_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "null_detect"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    cls_loss = parsed["null_detect_cls_loss"]
    obj_loss = parsed["null_detect_obj_loss"]
    cls_direction = parsed["null_detect_cls_direction"]
    obj_direction = parsed["null_detect_obj_direction"]
    feature_set = parsed["null_detect_feature_set"]

    if not save_csv:
        return

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _xyxy_shape_features(final_xyxy: torch.Tensor, reference_xyxy: torch.Tensor):
        final_x = 0.5 * (final_xyxy[0] + final_xyxy[2])
        final_y = 0.5 * (final_xyxy[1] + final_xyxy[3])
        final_w = torch.abs(final_xyxy[2] - final_xyxy[0])
        final_h = torch.abs(final_xyxy[3] - final_xyxy[1])
        ref_x = 0.5 * (reference_xyxy[0] + reference_xyxy[2])
        ref_y = 0.5 * (reference_xyxy[1] + reference_xyxy[3])
        ref_w = torch.abs(reference_xyxy[2] - reference_xyxy[0])
        ref_h = torch.abs(reference_xyxy[3] - reference_xyxy[1])

        final_size = final_w * final_h
        final_circum = final_w + final_h
        final_size_circum = final_size / final_circum.clamp(min=1e-12)
        ref_size = ref_w * ref_h
        ref_circum = ref_w + ref_h
        ref_size_circum = ref_size / ref_circum.clamp(min=1e-12)

        return {
            "size": final_size,
            "circum": final_circum,
            "size_circum": final_size_circum,
            "size_diff": torch.abs(final_size - ref_size),
            "circum_diff": torch.abs(final_circum - ref_circum),
            "size_circum_diff": torch.abs(final_size_circum - ref_size_circum),
            "x_loss": torch.abs(final_x - ref_x),
            "y_loss": torch.abs(final_y - ref_y),
            "w_loss": torch.abs(final_w - ref_w),
            "h_loss": torch.abs(final_h - ref_h),
        }

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    num_classes = len(detector.names) if detector.names is not None else int(config.get("model", {}).get("num_classes", 0))
    output_feature_names = [] if feature_set == "losses_only" else ["prob_sum"] + [f"prob_{i}" for i in range(max(0, num_classes))]
    null_feature_names = (
        ["x_loss", "y_loss", "w_loss", "h_loss", "obj_loss", "cls_loss"]
        if feature_set == "losses_only"
        else [
            "final_score",
            "size",
            "size_diff",
            "circum",
            "circum_diff",
            "size_circum",
            "size_circum_diff",
            "x_loss",
            "y_loss",
            "w_loss",
            "h_loss",
            "obj_loss",
            "cls_loss",
        ]
    )
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        *output_feature_names,
        *null_feature_names,
    ]
    output_csv = run_dir / "null_detect.csv"

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "feature_compute_sec"],
        device=device,
    )

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
                forward = run_yolo_forward_nms(
                    detector,
                    infer_batch,
                    nms_kwargs,
                    timing=timing,
                    num_classes_hint=max(1, num_classes),
                )

                if forward.raw_anchor_priors is None:
                    raise RuntimeError("null_detect requires YOLO anchor priors, but detector.model() did not return them.")
                t_feature = timing.start()
                raw_flat = _flatten_raw_prediction_layers(forward.raw_layers)
                if raw_flat is None:
                    raise RuntimeError("null_detect requires raw YOLO prediction layers, but detector.model() did not return them.")
                feature_compute_sec = timing.elapsed(t_feature)

            batch_items = 0
            sample_cache = {}
            for sample_idx in range(len(image_list)):
                t_feature = timing.start()
                pred_img = forward.raw_prediction[sample_idx].float()
                raw_img = raw_flat[sample_idx]
                anchor_img = (
                    forward.raw_anchor_priors[sample_idx]
                    if forward.raw_anchor_priors.ndim >= 3
                    else forward.raw_anchor_priors
                    if forward.raw_anchor_priors.ndim == 2 and len(image_list) == 1
                    else None
                )
                if anchor_img is None:
                    raise RuntimeError("YOLO null_detect could not align anchor priors with the current batch.")
                sample_cache[sample_idx] = {
                    "pred_img": pred_img,
                    "raw_img": raw_img,
                    "anchor_img": anchor_img,
                    "anchor_xyxy": _xywh_to_xyxy_tensor(anchor_img.to(dtype=pred_img.dtype, device=pred_img.device)),
                }
                feature_compute_sec += timing.elapsed(t_feature)

            for item in iter_yolo_detection_rows(
                detector, targets, forward.selected_preds, forward.selected_indices, device
            ):
                sample_idx = item["sample_idx"]
                raw_pred_idx = item["raw_pred_idx"]
                box = item["box"]
                cache = sample_cache[sample_idx]
                pred_img = cache["pred_img"]
                raw_img = cache["raw_img"]
                anchor_img = cache["anchor_img"]
                if raw_pred_idx >= int(pred_img.shape[0]) or raw_pred_idx >= int(raw_img.shape[0]) or raw_pred_idx >= int(anchor_img.shape[0]):
                    raise RuntimeError(
                        "YOLO null_detect raw_pred_idx is out of range for raw prediction/raw layer/anchor tensors. "
                        f"raw_pred_idx={raw_pred_idx}, pred={int(pred_img.shape[0])}, "
                        f"raw={int(raw_img.shape[0])}, anchors={int(anchor_img.shape[0])}"
                    )

                t_feature = timing.start()
                pred_row = pred_img[raw_pred_idx]
                raw_row = raw_img[raw_pred_idx]
                anchor_xyxy = cache["anchor_xyxy"][raw_pred_idx]
                prob_values = {}
                if feature_set != "losses_only":
                    pred_probs = pred_row[5:].detach().float() if pred_row.shape[0] > 5 else torch.zeros((0,), dtype=torch.float32, device=device)
                    prob_values = {"prob_sum": pred_probs.sum() if pred_probs.numel() else torch.zeros((), dtype=torch.float32, device=device)}
                    for prob_idx in range(max(0, num_classes)):
                        prob_values[f"prob_{prob_idx}"] = (
                            pred_probs[prob_idx]
                            if prob_idx < int(pred_probs.shape[0])
                            else torch.zeros((), dtype=torch.float32, device=device)
                        )

                final_xyxy = box[:4].detach().float()
                box_diff_values = _xyxy_shape_features(final_xyxy, anchor_xyxy)
                shape_values = {}
                if feature_set != "losses_only":
                    shape_values = {
                        "final_score": box[4],
                        "size": box_diff_values["size"],
                        "size_diff": box_diff_values["size_diff"],
                        "circum": box_diff_values["circum"],
                        "circum_diff": box_diff_values["circum_diff"],
                        "size_circum": box_diff_values["size_circum"],
                        "size_circum_diff": box_diff_values["size_circum_diff"],
                    }
                obj_target = torch.full_like(raw_row[4], 0.5)
                obj_loss_value = _objectness_loss_tensor(
                    raw_row[4],
                    obj_target,
                    mode=obj_loss,
                    direction=obj_direction,
                    reduction="sum",
                )
                cls_logits = raw_row[5:]
                if cls_logits.numel() == 0:
                    cls_loss_value = torch.zeros((), dtype=torch.float32, device=device)
                else:
                    cls_target_value = (
                        0.5
                        if str(cls_loss).strip().lower() == "bcewithlogits"
                        else 1.0 / float(cls_logits.numel())
                    )
                    uniform_target = torch.full_like(cls_logits, cls_target_value)
                    cls_loss_value = _class_loss_tensor(
                        cls_logits,
                        uniform_target,
                        class_idx=None,
                        mode=cls_loss,
                        direction=cls_direction,
                        reduction="sum",
                    )
                feature_compute_sec += timing.elapsed(t_feature)

                row = dict(item["base_row"])
                row.update({key: _to_float(value) for key, value in prob_values.items()})
                row.update({key: _to_float(value) for key, value in shape_values.items()})
                row.update(
                    {
                        "x_loss": _to_float(box_diff_values["x_loss"]),
                        "y_loss": _to_float(box_diff_values["y_loss"]),
                        "w_loss": _to_float(box_diff_values["w_loss"]),
                        "h_loss": _to_float(box_diff_values["h_loss"]),
                        "obj_loss": _to_float(obj_loss_value),
                        "cls_loss": _to_float(cls_loss_value),
                    }
                )
                writer.writerow(row)
                batch_items += 1

            timing.record(
                num_images=len(image_list),
                num_predictions=batch_items,
                stage_seconds={
                    "detector_inference_sec": forward.detector_inference_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )
            output_file.flush()
            del infer_batch, forward, raw_flat, sample_cache

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_null_detect_csv"]
