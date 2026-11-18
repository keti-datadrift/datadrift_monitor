from commands.predict.common import *

def run_fn_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    iou_match_threshold = parsed["gt_iou_match_threshold"]
    save_image = parsed["save_image_enabled"]
    image_step = parsed["save_image_gt_step"]
    image_max_num = parsed["save_image_gt_max_num"]

    output_csv = run_dir / "fn.csv"
    summary_json = run_dir / "summary.json"

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)

    total_images = 0
    fn_images = 0
    csv_writer = None
    csv_file_handle = None
    if save_csv:
        csv_file_handle = open(output_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file_handle, fieldnames=["image_id", "image_path", "fn"])
        csv_writer.writeheader()

    try:
        for step_idx, (images, targets) in enumerate(
            tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader))
        ):
            image_list = _as_image_list(images)
            batch_size = len(image_list)
            should_save_step = save_image and (step_idx % image_step == 0)
            step_dir = run_dir / "images" / f"0_{step_idx}"
            if should_save_step:
                step_dir.mkdir(parents=True, exist_ok=True)

            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                preds, _logits, _objectness, _features = detector(infer_batch)

            for sample_idx in range(batch_size):
                target = targets[sample_idx]
                row = {
                    "image_id": int(target["image_id"][0].item()),
                    "image_path": target["path"],
                }

                pred_boxes = preds[0][sample_idx]
                pred_class_names = preds[2][sample_idx]
                pred_scores = preds[3][sample_idx]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                fn = has_fn_for_image(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    iou_match_threshold=iou_match_threshold,
                )
                row["fn"] = fn
                fn_images += int(fn)

                if csv_writer is not None:
                    csv_writer.writerow(row)

                if should_save_step and sample_idx < image_max_num:
                    image_chw = resized_chws[sample_idx]
                    vis_image = draw_predictions(image_chw, pred_boxes, pred_class_names, pred_scores)
                    fn_gt_indices = get_fn_gt_indices(
                        gt_boxes=gt_boxes,
                        gt_class_names=gt_class_names,
                        pred_boxes=pred_boxes,
                        pred_class_names=pred_class_names,
                        iou_match_threshold=iou_match_threshold,
                    )
                    for gt_idx in fn_gt_indices:
                        gt_box = gt_boxes[gt_idx]
                        gt_name = gt_class_names[gt_idx]
                        x1, y1, x2, y2 = [int(v) for v in gt_box]
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            vis_image,
                            f"{gt_name}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    out_path = step_dir / f"{row['image_id']}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                total_images += 1
            del infer_batch, preds, _logits, _objectness, _features
    finally:
        if csv_file_handle is not None:
            csv_file_handle.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    summary = _build_summary(total_images, fn_images, output_csv if save_csv else "")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    print(f"Saved summary: {summary_json}")

def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"

    dataset_cfg = config.get("dataset", {})
    split = dataset_cfg.get("split", "val")
    parsed = parse_output_config(config.get("output", {}))
    save_csv = parsed["save_csv_enabled"]
    save_image = parsed["save_image_enabled"]
    image_step = max(1, int(parsed["save_image_gt_step"]))
    image_max_num = max(0, int(parsed["save_image_gt_max_num"]))
    iou_match_threshold = parsed["gt_iou_match_threshold"]

    if not save_csv and not save_image:
        return

    output_csv = run_dir / "tp.csv"
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
        "max_iou",
        "gt_iou",
        "tp",
        "error_type",
    ]

    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    if len(dataloader.dataset) == 0:
        raise ValueError("Loaded 0 images. Check dataset root/image_dir/split configuration in YAML.")

    detector, device = build_detector(config)
    nms_kwargs = _resolve_detector_nms_kwargs(detector)

    output_file = open(output_csv, "w", newline="", encoding="utf-8") if save_csv else None
    writer = csv.DictWriter(output_file, fieldnames=fieldnames) if output_file is not None else None
    if writer is not None:
        writer.writeheader()

    try:
        for step_idx, (images, targets) in enumerate(tqdm(
            dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader)
        )):
            image_list = _as_image_list(images)
            detector.zero_grad(set_to_none=True)
            infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                model_output = detector.model(infer_batch, augment=False)
                raw_prediction = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                raw_logits = model_output[1] if isinstance(model_output, (tuple, list)) and len(model_output) > 1 else None
                selected_preds, _selected_logits, _selected_objectness, selected_indices = detector.non_max_suppression(
                    prediction=raw_prediction,
                    logits=raw_logits,
                    conf_thres=nms_kwargs["conf_thres"],
                    iou_thres=nms_kwargs["iou_thres"],
                    classes=nms_kwargs["classes"],
                    agnostic=nms_kwargs["agnostic"],
                    max_det=nms_kwargs["max_det"],
                    return_indices=True,
                )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                det_b = selected_preds[sample_idx] if selected_preds and sample_idx < len(selected_preds) else torch.zeros((0, 6), device=device)
                raw_keep_b = selected_indices[sample_idx] if selected_indices and sample_idx < len(selected_indices) else torch.zeros((0,), dtype=torch.long, device=device)
                pred_boxes_t = det_b[:, :4]
                pred_scores_t = det_b[:, 4] if det_b.shape[1] > 4 else torch.zeros((det_b.shape[0],), device=device)
                pred_cls_ids = det_b[:, 5].long() if det_b.shape[1] > 5 else torch.zeros((det_b.shape[0],), dtype=torch.long, device=device)
                pred_boxes = pred_boxes_t.detach().cpu().tolist()
                pred_scores = pred_scores_t.detach().cpu().tolist()
                pred_class_names = [
                    detector.names[int(c.item())] if detector.names is not None else int(c.item())
                    for c in pred_cls_ids
                ]
                gt_boxes_tensor = target["boxes"]
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)

                error_rows = analyze_prediction_error_types(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=iou_match_threshold,
                )
                tp_flags = [row["tp"] for row in error_rows]
                best_ious = [row["gt_iou"] for row in error_rows]

                should_save_image = (
                    save_image
                    and step_idx % image_step == 0
                    and sample_idx < image_max_num
                )
                if should_save_image:
                    step_dir = run_dir / "images" / f"0_{step_idx}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    image_chw = resized_chws[sample_idx]
                    vis_image = np.transpose(image_chw, (1, 2, 0)).copy()
                    for gt_box, gt_name in zip(gt_boxes, gt_class_names):
                        x1, y1, x2, y2 = [int(v) for v in gt_box]
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (64, 128, 255), 2)
                        cv2.putText(
                            vis_image,
                            f"GT:{gt_name}",
                            (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (64, 128, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    for pred_idx, (box, score, pred_class) in enumerate(zip(pred_boxes, pred_scores, pred_class_names)):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        is_tp = int(tp_flags[pred_idx]) == 1
                        color = (0, 220, 0) if is_tp else (255, 64, 64)
                        tag = "TP" if is_tp else "FP"
                        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            vis_image,
                            f"{tag}:{pred_class}:{float(score):.2f}/IoU{float(best_ious[pred_idx]):.2f}",
                            (x1, min(vis_image.shape[0] - 1, y2 + 14)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            1,
                            cv2.LINE_AA,
                        )
                    out_path = step_dir / f"{image_id}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                for pred_idx, (box, score, pred_class) in enumerate(
                    zip(pred_boxes, pred_scores, pred_class_names)
                ):
                    if pred_idx >= int(raw_keep_b.shape[0]):
                        raise RuntimeError(
                            "YOLOv5 gt selected_indices is shorter than selected predictions. "
                            f"pred_idx={pred_idx}, selected_indices={int(raw_keep_b.shape[0])}"
                        )
                    raw_pred_idx = int(raw_keep_b[pred_idx].detach().cpu().item())
                    if writer is not None:
                        error_row = error_rows[pred_idx]
                        writer.writerow(
                            {
                                "image_id": image_id,
                                "image_path": image_path,
                                "pred_idx": pred_idx,
                                "raw_pred_idx": raw_pred_idx,
                                "xmin": float(box[0]),
                                "ymin": float(box[1]),
                                "xmax": float(box[2]),
                                "ymax": float(box[3]),
                                "score": float(score),
                                "pred_class": pred_class,
                                "max_iou": float(error_row["max_iou"]),
                                "gt_iou": float(error_row["gt_iou"]),
                                "tp": int(error_row["tp"]),
                                "error_type": error_row["error_type"],
                            }
                        )
            del infer_batch, model_output, raw_prediction, raw_logits, selected_preds, selected_indices
    finally:
        if output_file is not None:
            output_file.close()

    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if save_csv:
        print(f"Saved results CSV: {output_csv}")
    if save_image:
        print(f"Saved images: {run_dir / 'images'}")

__all__ = ["run_fn_csv", "run_tp_csv"]
