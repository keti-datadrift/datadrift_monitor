from commands.predict.common import *
from commands.utils.predict_utils import resolve_project_path


def _normalize_weight_path_for_compare(value, key):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string for YOLOv5 ensemble row alignment.")
    return Path(resolve_project_path(value.strip())).resolve()


def _same_weight_path(left, right):
    return str(left).lower() == str(right).lower()


def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    unit = parsed["unit"]

    if not save_csv:
        return

    ensemble_cfg = config.get("output", {}).get("ensemble", {})
    weights_cfg = ensemble_cfg.get("weights", [])
    if isinstance(weights_cfg, str):
        weight_paths = [weights_cfg]
    elif isinstance(weights_cfg, (list, tuple)):
        weight_paths = [str(w) for w in weights_cfg if str(w).strip()]
    else:
        weight_paths = []
    if not weight_paths:
        raise ValueError("output.uncertainty='ensemble' requires output.ensemble.weights to be a non-empty string/list.")

    model_weights = config.get("model", {}).get("weights", "")
    if isinstance(model_weights, (list, tuple)):
        raise ValueError("model.weights must be a single weight path when running YOLOv5 ensemble.")
    model_weight_path = _normalize_weight_path_for_compare(model_weights, "model.weights")
    first_ensemble_weight_path = _normalize_weight_path_for_compare(
        weight_paths[0], "output.ensemble.weights[0]"
    )
    if not _same_weight_path(model_weight_path, first_ensemble_weight_path):
        raise ValueError(
            "YOLOv5 ensemble requires output.ensemble.weights[0] to match model.weights so ensemble.csv "
            "uses the same deterministic row basis as gt.csv. "
            f"model.weights={model_weight_path}; output.ensemble.weights[0]={first_ensemble_weight_path}"
        )

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    output_csv = run_dir / "ensemble.csv"

    n_classes_hint = None
    class_names_hint = None
    n_classes_actual = None
    device = torch.device("cpu")
    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )

    detectors = []
    try:
        for model_weight in weight_paths:
            detector, device = build_detector(config, model_weight=model_weight)
            if n_classes_hint is None:
                n_classes_hint = len(detector.names) if detector.names is not None else 80
                class_names_hint = detector.names
            detectors.append(detector)
        timing.device = device

        if n_classes_hint is None:
            n_classes_hint = 80

        fieldnames = [
            "image_id", "image_path", "pred_idx", "raw_pred_idx",
            "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
            "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean",
            "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std",
        ]
        for class_idx in range(n_classes_hint):
            fieldnames.append(f"prob_{class_idx}_mean")
            fieldnames.append(f"prob_{class_idx}_std")

        with open(output_csv, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for images, targets in tqdm(
                dataloader,
                desc=f"Object Detector ({mode} - {uncertainty})",
                total=len(dataloader),
            ):
                base_detector = detectors[0]
                infer_batch, _ratios, _pads, _resized_chws = _prepare_infer_batch(
                    base_detector, images, device, auto=False
                )
                batch_size = int(infer_batch.shape[0])
                image_ids = [int(targets[i]["image_id"][0].item()) for i in range(batch_size)]
                image_paths = [targets[i]["path"] for i in range(batch_size)]

                feature_sum = None
                feature_sq_sum = None
                feature_count = 0
                det_boxes = None
                raw_keep_indices = None
                detector_inference_total_sec = 0.0
                prediction_matching_sec = 0.0
                feature_compute_sec = 0.0

                for det_idx, detector in enumerate(detectors):
                    t_detector = timing.start()
                    with torch.no_grad():
                        det_output = detector.model(infer_batch, augment=False)
                        det_raw_pred = det_output[0] if isinstance(det_output, (tuple, list)) else det_output
                        det_raw_logits = det_output[1] if isinstance(det_output, (tuple, list)) and len(det_output) > 1 else None
                        if det_idx == 0:
                            nms_logits = _resolve_nms_logits(det_raw_pred, det_raw_logits, num_classes_hint=n_classes_hint)
                            nms_kwargs = _resolve_detector_nms_kwargs(detector)
                            selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                                det_raw_pred,
                                nms_logits,
                                conf_thres=nms_kwargs["conf_thres"],
                                iou_thres=nms_kwargs["iou_thres"],
                                classes=nms_kwargs["classes"],
                                agnostic=nms_kwargs["agnostic"],
                                max_det=nms_kwargs["max_det"],
                                return_indices=True,
                            )
                    detector_inference_total_sec += timing.elapsed(t_detector)

                    t_feature = timing.start()
                    pred_batch = det_raw_pred.detach().float()
                    bbox_xyxy = _xywh_to_xyxy_tensor(pred_batch[..., :4])
                    score_vec = pred_batch[..., 4].unsqueeze(-1)
                    prob_mat = get_prediction_class_probs(detector, pred_batch).detach().float()
                    if prob_mat.numel() == 0 and det_raw_logits is not None:
                        prob_mat = torch.sigmoid(det_raw_logits.detach().float())
                    run_features = torch.cat([bbox_xyxy, score_vec, prob_mat], dim=2).detach()
                    class_count = int(run_features.shape[2] - 5)
                    if n_classes_actual is None:
                        n_classes_actual = class_count
                    elif n_classes_actual != class_count:
                        raise ValueError(
                            f"All ensemble weights must have the same class count: {n_classes_actual} vs {class_count}."
                        )
                    if feature_sum is None:
                        feature_sum = run_features.clone()
                        feature_sq_sum = run_features.square()
                    else:
                        if tuple(feature_sum.shape) != tuple(run_features.shape):
                            raise RuntimeError(
                                "YOLOv5 ensemble raw prediction shape mismatch across weights: "
                                f"{tuple(feature_sum.shape)} vs {tuple(run_features.shape)}"
                            )
                        feature_sum = feature_sum + run_features
                        feature_sq_sum = feature_sq_sum + run_features.square()
                    feature_count += 1
                    feature_compute_sec += timing.elapsed(t_feature)
                    del run_features

                    if det_idx == 0:
                        t_matching = timing.start()
                        det_boxes = []
                        raw_keep_indices = []
                        for b in range(batch_size):
                            det_b = selected_preds[b] if selected_preds and b < len(selected_preds) else torch.zeros((0, 6), device=device)
                            raw_keep_b = (
                                selected_indices[b]
                                if selected_indices and b < len(selected_indices)
                                else torch.zeros((0,), dtype=torch.long, device=device)
                            )
                            det_boxes.append(det_b.detach().cpu())
                            raw_keep_indices.append([int(v) for v in raw_keep_b.detach().cpu().tolist()])
                        prediction_matching_sec += timing.elapsed(t_matching)

                t_feature = timing.start()
                if feature_sum is None or feature_sq_sum is None or feature_count <= 0:
                    raise RuntimeError("YOLOv5 ensemble did not produce any feature tensor.")
                mean = feature_sum / float(feature_count)
                var = feature_sq_sum / float(feature_count) - mean.square()
                std = torch.sqrt(torch.clamp(var, min=0.0))
                feature_compute_sec += timing.elapsed(t_feature)
                del infer_batch, feature_sum, feature_sq_sum, var

                batch_items = 0
                t_matching = timing.start()
                mean_cpu = mean.detach().float().cpu()
                std_cpu = std.detach().float().cpu()
                for b in range(len(image_ids)):
                    image_id = int(image_ids[b])
                    image_path = str(image_paths[b])
                    mean_b = mean_cpu[b]
                    std_b = std_cpu[b]
                    n_candidates = int(mean_b.shape[0])
                    det_b = det_boxes[b]
                    raw_keep_b = [int(v) for v in raw_keep_indices[b]]
                    num_final = int(det_b.shape[0])
                    for pred_idx in range(num_final):
                        if pred_idx >= len(raw_keep_b):
                            raise RuntimeError(
                                "YOLOv5 ensemble selected_indices is shorter than selected predictions. "
                                f"pred_idx={pred_idx}, selected_indices={len(raw_keep_b)}"
                            )
                        raw_idx = int(raw_keep_b[pred_idx])
                        if raw_idx < 0 or raw_idx >= n_candidates:
                            raise RuntimeError(
                                "YOLOv5 ensemble raw_pred_idx is out of range for ensemble feature tensor. "
                                f"raw_pred_idx={raw_idx}, candidates={n_candidates}"
                            )
                        mean_vec = mean_b[raw_idx]
                        std_vec = std_b[raw_idx]
                        cls_idx = int(det_b[pred_idx, 5].item()) if det_b.shape[1] > 5 else -1
                        row = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "pred_idx": pred_idx,
                            "raw_pred_idx": raw_idx,
                            "xmin": float(det_b[pred_idx, 0].item()),
                            "ymin": float(det_b[pred_idx, 1].item()),
                            "xmax": float(det_b[pred_idx, 2].item()),
                            "ymax": float(det_b[pred_idx, 3].item()),
                            "score": float(det_b[pred_idx, 4].item()) if det_b.shape[1] > 4 else 0.0,
                            "pred_class": (
                                class_names_hint[cls_idx]
                                if (class_names_hint is not None and cls_idx >= 0 and cls_idx < len(class_names_hint))
                                else int(cls_idx)
                            ),
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
                        for class_idx in range(n_classes_hint):
                            if class_idx < n_classes_actual:
                                row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                                row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
                            else:
                                row[f"prob_{class_idx}_mean"] = 0.0
                                row[f"prob_{class_idx}_std"] = 0.0
                        writer.writerow(row)
                    batch_items += int(num_final)
                prediction_matching_sec += timing.elapsed(t_matching)
                timing.record(
                    num_images=batch_size,
                    num_predictions=batch_items,
                    stage_seconds={
                        "detector_inference_sec": detector_inference_total_sec,
                        "prediction_matching_sec": prediction_matching_sec,
                        "feature_compute_sec": feature_compute_sec,
                    },
                )
                del mean, std, mean_cpu, std_cpu
    finally:
        for detector in detectors:
            del detector
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    timing_csv, timing_json = timing.save()
    print(f"Saved results CSV: {output_csv}")
    print(f"Saved timing: {timing_csv}")
    print(f"Saved timing summary: {timing_json}")


__all__ = ["run_ensemble_csv"]
