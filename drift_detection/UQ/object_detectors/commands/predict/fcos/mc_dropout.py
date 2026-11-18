from commands.predict.common import *
from commands.predict.fcos.common import select_fcos_post_nms
from commands.predict.fcos.utils import iter_fcos_detection_rows


def _flatten_fcos_level_output(tensor, image_idx):
    return tensor[image_idx].permute(1, 2, 0).reshape(-1, tensor.shape[1])


def _flatten_fcos_centerness(tensor, image_idx):
    return tensor[image_idx].permute(1, 2, 0).reshape(-1)


def _decode_fcos_xyxy(location_xy, ltrb):
    return torch.stack(
        [
            location_xy[0] - ltrb[0],
            location_xy[1] - ltrb[1],
            location_xy[0] + ltrb[2],
            location_xy[1] + ltrb[3],
        ]
    )


def _run_fcos_head_post_nms_from_cache(detector, cache):
    rpn = detector.detector_model.rpn
    detector.detector_model.eval()
    detector._set_postprocessor_flags(keep_class_outputs=True)
    detections, _losses = rpn(cache["images"], cache["features"], None)
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


def _run_fcos_head_dense_from_cache(detector, cache):
    rpn = detector.detector_model.rpn
    detector.detector_model.eval()
    return rpn.head(cache["features"])


def _compute_fcos_locations(detector, cache):
    return detector.detector_model.rpn.compute_locations(cache["features"])


def _source_specs_from_detections(detections, det_rows):
    specs = []
    for det_row in det_rows:
        if det_row.sample_idx >= len(detections):
            raise RuntimeError(f"FCOS MC-dropout missing detection BoxList for image_idx={det_row.sample_idx}.")
        boxlist = detections[det_row.sample_idx]
        if det_row.pred_idx >= len(boxlist):
            raise RuntimeError(
                f"FCOS MC-dropout source row out of range: image_idx={det_row.sample_idx}, "
                f"pred_idx={det_row.pred_idx}, rows={len(boxlist)}"
            )
        required_fields = ["pre_nms_level", "pre_nms_location_idx", "pre_nms_candidate_idx"]
        for field in required_fields:
            if not boxlist.has_field(field):
                raise RuntimeError(f"FCOS MC-dropout requires detection field '{field}'.")
        source_raw_idx = int(boxlist.get_field("pre_nms_candidate_idx")[det_row.pred_idx].detach().cpu().item())
        if source_raw_idx != int(det_row.raw_pred_idx):
            raise RuntimeError(
                f"FCOS MC-dropout raw_pred_idx mismatch: image_idx={det_row.sample_idx}, "
                f"pred_idx={det_row.pred_idx}, row_raw_idx={det_row.raw_pred_idx}, source_raw_idx={source_raw_idx}"
            )
        specs.append(
            {
                "sample_idx": int(det_row.sample_idx),
                "level": int(boxlist.get_field("pre_nms_level")[det_row.pred_idx].detach().cpu().item()),
                "location_idx": int(boxlist.get_field("pre_nms_location_idx")[det_row.pred_idx].detach().cpu().item()),
                "class_idx": int(det_row.cls_idx),
            }
        )
    return specs


def _features_from_fcos_dense_outputs(
    *,
    box_cls,
    box_regression,
    centerness,
    locations,
    source_specs,
    num_classes,
    device,
):
    flat_cache = {}
    rows = []
    for spec in source_specs:
        sample_idx = spec["sample_idx"]
        level = spec["level"]
        loc_idx = spec["location_idx"]
        class_idx = spec["class_idx"]
        if level < 0 or level >= len(box_cls):
            raise RuntimeError(f"FCOS MC-dropout source level out of range: level={level}, levels={len(box_cls)}")
        key = (sample_idx, level)
        if key not in flat_cache:
            flat_cache[key] = {
                "cls": _flatten_fcos_level_output(box_cls[level], sample_idx),
                "bbox": _flatten_fcos_level_output(box_regression[level], sample_idx),
                "cnt": _flatten_fcos_centerness(centerness[level], sample_idx),
            }
        flat = flat_cache[key]
        if loc_idx < 0 or loc_idx >= int(flat["cls"].shape[0]):
            raise RuntimeError(
                f"FCOS MC-dropout source location out of range: image_idx={sample_idx}, "
                f"level={level}, location_idx={loc_idx}, locations={int(flat['cls'].shape[0])}"
            )
        if class_idx < 0 or class_idx >= int(num_classes):
            raise RuntimeError(
                f"FCOS MC-dropout source class out of range: class_idx={class_idx}, num_classes={int(num_classes)}"
            )
        cls_logits = flat["cls"][loc_idx].view(-1)
        prob_vec = torch.sigmoid(cls_logits[:num_classes])
        ltrb = flat["bbox"][loc_idx].view(4)
        cnt_prob = torch.sigmoid(flat["cnt"][loc_idx].view(()))
        loc_xy = locations[level][loc_idx].to(device=device, dtype=ltrb.dtype)
        box_xyxy = _decode_fcos_xyxy(loc_xy, ltrb)
        score = torch.sqrt((prob_vec[class_idx] * cnt_prob).clamp(min=0.0))
        rows.append(torch.cat([box_xyxy, score.view(1), prob_vec], dim=0))
    if not rows:
        return torch.zeros((0, 5 + int(num_classes)), dtype=torch.float32, device=device)
    return torch.stack(rows, dim=0).to(device=device, dtype=torch.float32)


def run_mc_dropout_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "mc_dropout"

    split = config.get("dataset", {}).get("split", "val")
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
    if not bool(getattr(detector, "is_fcos", False)):
        raise ValueError("commands.predict.fcos.mc_dropout requires model.type=fcos.")

    timing = StageTimingProfiler(
        run_dir=run_dir,
        uncertainty=uncertainty,
        unit=unit,
        stages=["detector_inference_sec", "prediction_matching_sec", "feature_compute_sec"],
        device=device,
    )
    num_classes = len(detector.names) if detector.names is not None else 80

    output_csv = run_dir / "mc_dropout.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx",
        "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "xmin_mean", "ymin_mean", "xmax_mean", "ymax_mean", "score_mean",
        "xmin_std", "ymin_std", "xmax_std", "ymax_std", "score_std",
    ]
    for class_idx in range(num_classes):
        fieldnames.append(f"prob_{class_idx}_mean")
        fieldnames.append(f"prob_{class_idx}_std")

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
            infer_batch = _prepare_infer_batch(detector, image_list, device, auto=False)[0]

            detector_inference_sec = 0.0
            prediction_matching_sec = 0.0
            feature_compute_sec = 0.0

            with torch.no_grad():
                fcos_preprocessed = detector.preprocess_images(infer_batch)
                t_detector = timing.start()
                feature_cache = detector.prepare_feature_cache(fcos_preprocessed)
                locations = _compute_fcos_locations(detector, feature_cache)
                detector.set_dropout_rate(0.0)
                detections, selected_preds, selected_indices = _run_fcos_head_post_nms_from_cache(detector, feature_cache)
            detector_inference_sec += timing.elapsed(t_detector)

            t_matching = timing.start()
            det_rows = list(iter_fcos_detection_rows(detector, targets, selected_preds, selected_indices, device))
            source_specs = _source_specs_from_detections(detections, det_rows)
            prediction_matching_sec += timing.elapsed(t_matching)

            run_feature_rows = []
            detector.set_dropout_rate(dropout_rate)
            try:
                with torch.no_grad():
                    for _ in range(num_runs):
                        detector.zero_grad(set_to_none=True)
                        t_detector = timing.start()
                        box_cls, box_regression, centerness = _run_fcos_head_dense_from_cache(detector, feature_cache)
                        detector_inference_sec += timing.elapsed(t_detector)

                        t_feature = timing.start()
                        run_features = _features_from_fcos_dense_outputs(
                            box_cls=box_cls,
                            box_regression=box_regression,
                            centerness=centerness,
                            locations=locations,
                            source_specs=source_specs,
                            num_classes=num_classes,
                            device=device,
                        )
                        run_feature_rows.append(run_features.detach())
                        feature_compute_sec += timing.elapsed(t_feature)
            finally:
                detector.set_dropout_rate(0.0)

            t_feature = timing.start()
            if run_feature_rows:
                runs_tensor = torch.stack(run_feature_rows, dim=0)
                feat_mean = runs_tensor.mean(dim=0)
                feat_std = runs_tensor.std(dim=0, unbiased=False)
                del runs_tensor
            else:
                feat_mean = torch.zeros((0, 5 + num_classes), dtype=torch.float32, device=device)
                feat_std = torch.zeros_like(feat_mean)
            feature_compute_sec += timing.elapsed(t_feature)

            t_matching = timing.start()
            batch_rows = []
            if int(feat_mean.shape[0]) != len(det_rows):
                raise RuntimeError(
                    f"FCOS MC-dropout feature row mismatch: feature_rows={int(feat_mean.shape[0])}, detections={len(det_rows)}"
                )
            for row_idx, det_row in enumerate(det_rows):
                mean_vec = feat_mean[row_idx].detach().float().cpu()
                std_vec = feat_std[row_idx].detach().float().cpu()
                row = dict(det_row.base)
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
                for class_idx in range(num_classes):
                    row[f"prob_{class_idx}_mean"] = float(mean_vec[5 + class_idx].item())
                    row[f"prob_{class_idx}_std"] = float(std_vec[5 + class_idx].item())
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

            del infer_batch, fcos_preprocessed, feature_cache, locations
            del detections, selected_preds, selected_indices, det_rows, source_specs
            del run_feature_rows, feat_mean, feat_std, batch_rows
    finally:
        detector.set_dropout_rate(0.0)
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
