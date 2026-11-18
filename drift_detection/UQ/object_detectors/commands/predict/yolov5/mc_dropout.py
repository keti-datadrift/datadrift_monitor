from commands.predict.common import *
from commands.predict.yolov5.utils import iter_yolo_detection_rows


def _get_yolov5_detect_module(detector):
    model = getattr(detector, "model", None)
    modules = getattr(model, "model", None)
    if modules is None or len(modules) == 0:
        raise RuntimeError("YOLOv5 MC-dropout requires detector.model.model modules.")
    detect_module = modules[-1]
    if not (hasattr(detect_module, "m") and hasattr(detect_module, "nc") and hasattr(detect_module, "na")):
        raise RuntimeError("YOLOv5 MC-dropout requires a YOLO Detect head as the final module.")
    return detect_module


def _forward_yolov5_features_to_head(model, img):
    modules = model.model
    detect_module = modules[-1]
    if not (hasattr(detect_module, "m") and hasattr(detect_module, "nc") and hasattr(detect_module, "na")):
        raise RuntimeError("YOLOv5 MC-dropout requires a YOLO Detect head as the final module.")
    y = []
    x = img
    for module in modules[:-1]:
        if module.f != -1:
            x = y[module.f] if isinstance(module.f, int) else [x if j == -1 else y[j] for j in module.f]
        x = module(x)
        y.append(x if module.i in model.save else None)
    head_input = y[detect_module.f] if isinstance(detect_module.f, int) else [x if j == -1 else y[j] for j in detect_module.f]
    if isinstance(head_input, torch.Tensor):
        return [head_input.detach()]
    return [feature.detach() for feature in head_input]


def _forward_yolov5_head_from_cache(detector, cached_features):
    features = [feature.clone() for feature in cached_features]
    return detector.model.model[-1](features)


def _mc_feature_tensor(detector, raw_prediction):
    pred_batch = raw_prediction.detach().float()
    bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
    score_vec = pred_batch[..., 4].unsqueeze(-1)
    prob_mat = get_prediction_class_probs(detector, pred_batch).detach().float()
    return torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2)


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]
    num_runs = int(parsed["mc_num_runs"])
    dropout_rate = float(parsed["mc_dropout_rate"])

    if not save_csv:
        return
    if num_runs <= 0:
        raise ValueError("mc_dropout.num_runs must be positive.")

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    _get_yolov5_detect_module(detector)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )
    n_classes_hint = len(detector.names) if detector.names is not None else 80

    output_csv = run_dir / "mc_dropout.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean",
        "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std",
    ]
    for class_idx in range(n_classes_hint):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")

    probe_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)
    if len(probe_handles) == 0:
        print("[WARN] YOLOv5 detect head not found for forced MC-dropout hooks.")
    for handle in probe_handles:
        handle.remove()

    write_queue: queue.Queue = queue.Queue()
    writer_thread = threading.Thread(
        target=_mc_dropout_single_csv_writer,
        args=(write_queue, output_csv, fieldnames),
        daemon=True,
    )
    writer_thread.start()

    try:
        for images, targets in tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        ):
            image_list = _as_image_list(images)
            infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                detector, image_list, device, auto=False
            )
            detector_inference_sec = 0.0
            prediction_matching_sec = 0.0
            feature_compute_sec = 0.0

            with torch.no_grad():
                t_detector = timing.start()
                cached_features = _forward_yolov5_features_to_head(detector.model, infer_batch)
                det_output = _forward_yolov5_head_from_cache(detector, cached_features)
                det_raw_pred = det_output[0] if isinstance(det_output, (tuple, list)) else det_output
                det_raw_logits = det_output[1] if isinstance(det_output, (tuple, list)) and len(det_output) > 1 else None
                det_nms_logits = _resolve_nms_logits(det_raw_pred, det_raw_logits, num_classes_hint=n_classes_hint)
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=det_raw_pred,
                    logits=det_nms_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )
                detector_inference_sec += timing.elapsed(t_detector)

            feature_runs = []
            n_candidates = None
            n_classes = None
            mc_handles = enable_forced_mc_dropout_on_yolov5_head(detector.model, dropout_rate)
            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        detector.zero_grad(set_to_none=True)
                        t_detector = timing.start()
                        model_output = _forward_yolov5_head_from_cache(detector, cached_features)
                        raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                        detector_inference_sec += timing.elapsed(t_detector)

                        t_feature = timing.start()
                        run_features = _mc_feature_tensor(detector, raw_prediction)
                        if n_candidates is None:
                            n_candidates = int(run_features.shape[1])
                            n_classes = int(run_features.shape[2] - 5)
                        if int(run_features.shape[1]) != n_candidates:
                            raise RuntimeError(
                                "YOLOv5 raw candidate count changed across MC runs; expected fixed pre-NMS candidates."
                            )
                        feature_runs.append(run_features.detach())
                        feature_compute_sec += timing.elapsed(t_feature)
            finally:
                for handle in mc_handles:
                    handle.remove()

            if n_candidates is None:
                del infer_batch, cached_features, selected_preds, selected_indices
                continue

            t_feature = timing.start()
            runs_tensor = torch.stack(feature_runs, dim=0)
            feat_mean = runs_tensor.mean(dim=0)
            feat_std = runs_tensor.std(dim=0, unbiased=False)
            feature_compute_sec += timing.elapsed(t_feature)
            del runs_tensor

            batch_rows = []
            t_matching = timing.start()
            row_items = list(iter_yolo_detection_rows(detector, targets, selected_preds, selected_indices, device))
            for item in row_items:
                sample_idx = item["sample_idx"]
                pred_idx = item["pred_idx"]
                raw_idx = item["raw_pred_idx"]
                if raw_idx >= int(n_candidates):
                    raise RuntimeError(
                        "YOLOv5 MC-dropout raw_pred_idx is out of range for MC feature tensor. "
                        f"raw_pred_idx={raw_idx}, candidates={int(n_candidates)}"
                    )
                mean_vec = feat_mean[sample_idx, raw_idx].detach().float().cpu()
                std_vec = feat_std[sample_idx, raw_idx].detach().float().cpu()
                row = dict(item["base_row"])
                row.update(
                    {
                        "xmin_mean": float(mean_vec[0].item()),
                        "ymin_mean": float(mean_vec[1].item()),
                        "xmax_mean": float(mean_vec[2].item()),
                        "ymax_mean": float(mean_vec[3].item()),
                        "score_mean": float(mean_vec[4].item()),
                        "xmin_std": float(std_vec[0].item()),
                        "ymin_std": float(std_vec[1].item()),
                        "xmax_std": float(std_vec[2].item()),
                        "ymax_std": float(std_vec[3].item()),
                        "score_std": float(std_vec[4].item()),
                    }
                )
                for class_idx in range(n_classes_hint):
                    if class_idx < int(n_classes):
                        row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                        row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
                    else:
                        row[f"prob_{class_idx}_mean"] = 0.0
                        row[f"prob_{class_idx}_std"] = 0.0
                batch_rows.append(row)
            prediction_matching_sec += timing.elapsed(t_matching)

            write_queue.put(batch_rows)
            timing.record(
                num_images=len(image_list),
                num_predictions=len(batch_rows),
                stage_seconds={
                    "detector_inference_sec": detector_inference_sec,
                    "prediction_matching_sec": prediction_matching_sec,
                    "feature_compute_sec": feature_compute_sec,
                },
            )

            del infer_batch, cached_features, selected_preds, selected_indices
            del feature_runs, feat_mean, feat_std
    except Exception:
        raise
    finally:
        write_queue.put(None)
        writer_thread.join()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()

    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_mc_dropout_csv"]
