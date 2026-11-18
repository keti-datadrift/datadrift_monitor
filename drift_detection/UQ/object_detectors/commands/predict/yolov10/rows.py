import torch


def split_yolov10_raw_pred_idx(raw_pred_idx, num_classes):
    raw_pred_idx = int(raw_pred_idx)
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("YOLOv10 num_classes must be positive.")
    return raw_pred_idx // num_classes, raw_pred_idx % num_classes


def yolov10_class_name(detector, cls_idx):
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def iter_yolov10_detection_rows(detector, targets, selected_preds, selected_indices, device):
    num_classes = len(detector.names) if detector.names is not None else getattr(detector, "num_classes", 80)
    for sample_idx in range(len(targets)):
        target = targets[sample_idx]
        image_id = int(target["image_id"][0].item())
        image_path = target["path"]
        det = (
            selected_preds[sample_idx]
            if selected_preds and sample_idx < len(selected_preds)
            else torch.zeros((0, 6), dtype=torch.float32, device=device)
        )
        raw_keep = (
            selected_indices[sample_idx]
            if selected_indices and sample_idx < len(selected_indices)
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        for pred_idx in range(int(det.shape[0])):
            if pred_idx >= int(raw_keep.shape[0]):
                raise RuntimeError(
                    "YOLOv10 selected_indices is shorter than selected predictions. "
                    f"sample_idx={sample_idx}, pred_idx={pred_idx}, indices={int(raw_keep.shape[0])}"
                )
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            raw_box_idx, raw_class_idx = split_yolov10_raw_pred_idx(raw_pred_idx, num_classes)
            box = det[pred_idx]
            cls_idx = int(box[5].detach().cpu().item()) if box.shape[0] > 5 else 0
            if raw_class_idx != cls_idx:
                raise RuntimeError(
                    "YOLOv10 raw_pred_idx class component does not match selected prediction class. "
                    f"raw_pred_idx={raw_pred_idx}, raw_class_idx={raw_class_idx}, pred_class_idx={cls_idx}"
                )
            yield {
                "sample_idx": sample_idx,
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "raw_box_idx": raw_box_idx,
                "raw_class_idx": raw_class_idx,
                "box": box,
                "base_row": {
                    "image_id": image_id,
                    "image_path": image_path,
                    "pred_idx": pred_idx,
                    "raw_pred_idx": raw_pred_idx,
                    "xmin": float(box[0].detach().cpu().item()),
                    "ymin": float(box[1].detach().cpu().item()),
                    "xmax": float(box[2].detach().cpu().item()),
                    "ymax": float(box[3].detach().cpu().item()),
                    "score": float(box[4].detach().cpu().item()),
                    "pred_class": yolov10_class_name(detector, cls_idx),
                },
            }


__all__ = ["iter_yolov10_detection_rows", "split_yolov10_raw_pred_idx", "yolov10_class_name"]
