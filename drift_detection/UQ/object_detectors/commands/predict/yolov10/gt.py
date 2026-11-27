import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from commands.predict.common import (
    _as_image_list,
    _prepare_infer_batch,
    _resolve_gt_class_names,
    create_dataloader,
)
from commands.predict.yolov10.config import parse_yolov10_output_config
from commands.predict.yolov10.forward import run_yolov10_forward
from commands.predict.yolov10.rows import iter_yolov10_detection_rows
from commands.utils.predict_utils import (
    analyze_prediction_error_types,
    build_detector,
    load_gt_category_maps,
    map_boxes_to_letterbox,
)


def run_tp_csv(config, run_dir):
    run_dir = Path(run_dir)
    mode = str(config.get("mode", "predict"))
    uncertainty = "gt"
    split = config.get("dataset", {}).get("split", "val")
    parsed = parse_yolov10_output_config(config)
    if not parsed["save_csv_enabled"] and not parsed["save_image_enabled"]:
        return
    image_step = max(1, int(parsed["save_image_gt_step"]))
    image_max_num = max(0, int(parsed["save_image_gt_max_num"]))
    output_csv = run_dir / "tp.csv"
    fieldnames = [
        "image_id", "image_path", "pred_idx", "raw_pred_idx", "xmin", "ymin", "xmax", "ymax", "score", "pred_class",
        "max_iou", "gt_iou", "tp", "error_type",
    ]
    catid_to_name = load_gt_category_maps(config, split)
    dataloader = create_dataloader(config, split=split)
    detector, device = build_detector(config)
    output_file = open(output_csv, "w", newline="", encoding="utf-8") if parsed["save_csv_enabled"] else None
    writer = csv.DictWriter(output_file, fieldnames=fieldnames) if output_file is not None else None
    if writer is not None:
        writer.writeheader()
    try:
        for step_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Object Detector ({mode} - {uncertainty})", total=len(dataloader))):
            image_list = _as_image_list(images)
            infer_batch, ratios, pads, resized_chws = _prepare_infer_batch(detector, image_list, device, auto=False)
            with torch.no_grad():
                forward = run_yolov10_forward(detector, infer_batch)
            items_by_sample = {i: [] for i in range(len(image_list))}
            for item in iter_yolov10_detection_rows(detector, targets, forward.selected_preds, forward.selected_indices, device):
                items_by_sample[item["sample_idx"]].append(item)
            for sample_idx in range(len(image_list)):
                target = targets[sample_idx]
                sample_items = items_by_sample[sample_idx]
                pred_boxes = [item["box"][:4].detach().cpu().tolist() for item in sample_items]
                pred_scores = [float(item["box"][4].detach().cpu().item()) for item in sample_items]
                pred_class_names = [item["base_row"]["pred_class"] for item in sample_items]
                gt_boxes = map_boxes_to_letterbox(target["boxes"], ratios[sample_idx], pads[sample_idx])
                gt_class_names = _resolve_gt_class_names(target, catid_to_name)
                error_rows = analyze_prediction_error_types(
                    gt_boxes=gt_boxes,
                    gt_class_names=gt_class_names,
                    pred_boxes=pred_boxes,
                    pred_class_names=pred_class_names,
                    pred_scores=pred_scores,
                    iou_match_threshold=parsed["gt_iou_match_threshold"],
                )

                should_save_image = (
                    parsed["save_image_enabled"]
                    and step_idx % image_step == 0
                    and sample_idx < image_max_num
                )
                if should_save_image:
                    step_dir = run_dir / "images" / f"0_{step_idx}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    image_chw = resized_chws[sample_idx]
                    vis_image = np.transpose(image_chw, (1, 2, 0)).copy()
                    tp_flags = [row["tp"] for row in error_rows]
                    best_ious = [row["gt_iou"] for row in error_rows]
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
                    out_path = step_dir / f"{int(target['image_id'][0].item())}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                if writer is None:
                    continue
                for item, row_info in zip(sample_items, error_rows):
                    row = dict(item["base_row"])
                    row.update(
                        {
                            "max_iou": row_info["max_iou"],
                            "gt_iou": row_info["gt_iou"],
                            "tp": row_info["tp"],
                            "error_type": row_info["error_type"],
                        }
                    )
                    writer.writerow(row)
            del infer_batch, forward
    finally:
        if output_file is not None:
            output_file.close()
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if parsed["save_csv_enabled"]:
        print(f"Saved results CSV: {output_csv}")
    if parsed["save_image_enabled"]:
        print(f"Saved images: {run_dir / 'images'}")


__all__ = ["run_tp_csv"]
