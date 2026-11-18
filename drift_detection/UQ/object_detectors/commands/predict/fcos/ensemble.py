from commands.predict.common import *
from commands.predict.fcos.mc_dropout import (
    _compute_fcos_locations,
    _features_from_fcos_dense_outputs,
    _run_fcos_head_dense_from_cache,
    _source_specs_from_detections,
)
from commands.predict.fcos.common import select_fcos_post_nms
from commands.predict.fcos.utils import iter_fcos_detection_rows
from commands.utils.predict_utils import resolve_project_path


def _normalize_weight_path_for_compare(value, key):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string for FCOS ensemble row alignment.")
    return Path(resolve_project_path(value.strip())).resolve()


def _same_weight_path(left, right):
    return str(left).lower() == str(right).lower()


def _run_fcos_post_nms_from_dense(detector, cache, locations, box_cls, box_regression, centerness):
    rpn = detector.detector_model.rpn
    detector.detector_model.eval()
    detector._set_postprocessor_flags(keep_class_outputs=True)
    detections, _losses = rpn._forward_test(
        locations,
        box_cls,
        box_regression,
        centerness,
        cache["images"].image_sizes,
    )
    raw_prediction, raw_logits, raw_indices = detector._boxlists_to_contract(
        detections,
        include_logits=True,
        include_indices=True,
        include_probs=True,
    )
    selected = select_fcos_post_nms(
        detector,
        raw_prediction,
        raw_logits,
        raw_indices,
    )
    return detections, selected[0], selected[3]


def run_ensemble_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "ensemble"

    split = config.get("dataset", {}).get("split", "val")
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
        raise ValueError("model.weights must be a single weight path when running FCOS ensemble.")
    model_weight_path = _normalize_weight_path_for_compare(model_weights, "model.weights")
    first_ensemble_weight_path = _normalize_weight_path_for_compare(
        weight_paths[0], "output.ensemble.weights[0]"
    )
    if not _same_weight_path(model_weight_path, first_ensemble_weight_path):
        raise ValueError(
            "FCOS ensemble requires output.ensemble.weights[0] to match model.weights so ensemble.csv "
            "uses the same deterministic row basis as gt.csv. "
            f"model.weights={model_weight_path}; output.ensemble.weights[0]={first_ensemble_weight_path}"
        )

    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    output_csv = run_dir / "ensemble.csv"

    n_classes_hint = None
    class_names_hint = None
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
            if not bool(getattr(detector, "is_fcos", False)):
                raise ValueError("commands.predict.fcos.ensemble requires model.type=fcos.")
            if n_classes_hint is None:
                n_classes_hint = len(detector.names) if detector.names is not None else 80
                class_names_hint = detector.names
            else:
                class_count = len(detector.names) if detector.names is not None else 80
                if int(class_count) != int(n_classes_hint):
                    raise ValueError(
                        f"All ensemble weights must have the same class count: {n_classes_hint} vs {class_count}."
                    )
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
                image_list = _as_image_list(images)
                infer_batch = _prepare_infer_batch(detectors[0], image_list, device, auto=False)[0]

                detector_inference_sec = 0.0
                prediction_matching_sec = 0.0
                feature_compute_sec = 0.0

                with torch.no_grad():
                    base_preprocessed = detectors[0].preprocess_images(infer_batch)
                    t_detector = timing.start()
                    base_feature_cache = detectors[0].prepare_feature_cache(base_preprocessed)
                    base_locations = _compute_fcos_locations(detectors[0], base_feature_cache)
                    base_box_cls, base_box_regression, base_centerness = _run_fcos_head_dense_from_cache(
                        detectors[0],
                        base_feature_cache,
                    )
                    detections, selected_preds, selected_indices = _run_fcos_post_nms_from_dense(
                        detectors[0],
                        base_feature_cache,
                        base_locations,
                        base_box_cls,
                        base_box_regression,
                        base_centerness,
                    )
                detector_inference_sec += timing.elapsed(t_detector)

                t_matching = timing.start()
                det_rows = list(iter_fcos_detection_rows(detectors[0], targets, selected_preds, selected_indices, device))
                source_specs = _source_specs_from_detections(detections, det_rows)
                prediction_matching_sec += timing.elapsed(t_matching)

                feature_sum = None
                feature_sq_sum = None
                feature_count = 0
                for det_idx, detector in enumerate(detectors):
                    if det_idx == 0:
                        feature_cache = base_feature_cache
                        locations = base_locations
                        box_cls = base_box_cls
                        box_regression = base_box_regression
                        centerness = base_centerness
                    else:
                        with torch.no_grad():
                            processed = detector.preprocess_images(infer_batch)
                            t_detector = timing.start()
                            feature_cache = detector.prepare_feature_cache(processed)
                            locations = _compute_fcos_locations(detector, feature_cache)
                        detector_inference_sec += timing.elapsed(t_detector)

                        t_detector = timing.start()
                        with torch.no_grad():
                            box_cls, box_regression, centerness = _run_fcos_head_dense_from_cache(detector, feature_cache)
                        detector_inference_sec += timing.elapsed(t_detector)

                    t_feature = timing.start()
                    run_features = _features_from_fcos_dense_outputs(
                        box_cls=box_cls,
                        box_regression=box_regression,
                        centerness=centerness,
                        locations=locations,
                        source_specs=source_specs,
                        num_classes=n_classes_hint,
                        device=device,
                    )
                    run_features = run_features.detach()
                    if feature_sum is None:
                        feature_sum = run_features.clone()
                        feature_sq_sum = run_features.square()
                    else:
                        if tuple(feature_sum.shape) != tuple(run_features.shape):
                            raise RuntimeError(
                                "FCOS ensemble feature row shape mismatch across weights: "
                                f"{tuple(feature_sum.shape)} vs {tuple(run_features.shape)}"
                            )
                        feature_sum = feature_sum + run_features
                        feature_sq_sum = feature_sq_sum + run_features.square()
                    feature_count += 1
                    feature_compute_sec += timing.elapsed(t_feature)

                    if det_idx != 0:
                        del processed, feature_cache, locations
                    del box_cls, box_regression, centerness, run_features

                t_feature = timing.start()
                if feature_sum is not None and feature_sq_sum is not None and feature_count > 0:
                    feat_mean = feature_sum / float(feature_count)
                    feat_var = feature_sq_sum / float(feature_count) - feat_mean.square()
                    feat_std = torch.sqrt(torch.clamp(feat_var, min=0.0))
                    del feat_var
                else:
                    feat_mean = torch.zeros((0, 5 + int(n_classes_hint)), dtype=torch.float32, device=device)
                    feat_std = torch.zeros_like(feat_mean)
                feature_compute_sec += timing.elapsed(t_feature)

                t_matching = timing.start()
                if int(feat_mean.shape[0]) != len(det_rows):
                    raise RuntimeError(
                        f"FCOS ensemble feature row mismatch: feature_rows={int(feat_mean.shape[0])}, detections={len(det_rows)}"
                    )
                for row_idx, det_row in enumerate(det_rows):
                    mean_vec = feat_mean[row_idx].detach().float().cpu()
                    std_vec = feat_std[row_idx].detach().float().cpu()
                    cls_idx = int(det_row.cls_idx)
                    row = dict(det_row.base)
                    row.update(
                        {
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
                    )
                    for class_idx in range(n_classes_hint):
                        row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                        row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
                    writer.writerow(row)
                prediction_matching_sec += timing.elapsed(t_matching)

                timing.record(
                    num_images=len(image_list),
                    num_predictions=len(det_rows),
                    stage_seconds={
                        "detector_inference_sec": detector_inference_sec,
                        "prediction_matching_sec": prediction_matching_sec,
                        "feature_compute_sec": feature_compute_sec,
                    },
                )
                del infer_batch, base_preprocessed, base_feature_cache, base_locations
                del base_box_cls, base_box_regression, base_centerness
                del detections, selected_preds, selected_indices, det_rows, source_specs
                del feature_sum, feature_sq_sum, feat_mean, feat_std
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
