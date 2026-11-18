from commands.predict.common import *
from commands.predict.fcos.utils import iter_fcos_detection_rows, run_fcos_forward_nms

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
            dummy_timing = type("_DummyTiming", (), {"start": staticmethod(time.perf_counter), "elapsed": staticmethod(lambda t: time.perf_counter() - t)})()
            result = run_fcos_forward_nms(
                detector=detector,
                image_list=image_list,
                device=device,
                timing=dummy_timing,
                keep_pre_nms=False,
                keep_class_outputs=False,
            )

            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                image_id = int(target["image_id"][0].item())
                image_path = target["path"]

                det_b = result.selected_preds[sample_idx] if result.selected_preds and sample_idx < len(result.selected_preds) else torch.zeros((0, 6), device=device)
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
                gt_boxes = map_boxes_to_letterbox(gt_boxes_tensor, result.ratios[sample_idx], result.pads[sample_idx])
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
                    if result.resized_chws is None and bool(getattr(detector, "is_fcos", False)):
                        image_chw = detector.resize_image_for_display(image_list[sample_idx])
                    elif result.resized_chws is None:
                        image_chw = np.ascontiguousarray(
                            np.clip(image_list[sample_idx].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                        )
                    else:
                        image_chw = result.resized_chws[sample_idx]
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

                for det_row in iter_fcos_detection_rows(detector, [target], [det_b], [result.selected_indices[sample_idx]], device):
                    pred_idx = det_row.pred_idx
                    if writer is not None:
                        error_row = error_rows[pred_idx]
                        writer.writerow(
                            {
                                **det_row.base,
                                "max_iou": float(error_row["max_iou"]),
                                "gt_iou": float(error_row["gt_iou"]),
                                "tp": int(error_row["tp"]),
                                "error_type": error_row["error_type"],
                            }
                        )
            del result
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

__all__ = ["run_tp_csv"]
