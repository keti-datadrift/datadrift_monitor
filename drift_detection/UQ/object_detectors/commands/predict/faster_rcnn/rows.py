from dataclasses import dataclass

import torch


@dataclass
class FasterRCNNDetectionRow:
    sample_idx: int
    pred_idx: int
    raw_pred_idx: int
    box: torch.Tensor
    cls_idx: int
    score: float
    pred_class: object
    base: dict


def pred_class_name(detector, cls_idx):
    cls_idx = int(cls_idx)
    if isinstance(detector.names, dict):
        return detector.names.get(cls_idx, str(cls_idx))
    if isinstance(detector.names, list) and 0 <= cls_idx < len(detector.names):
        return detector.names[cls_idx]
    return str(cls_idx)


def iter_faster_rcnn_detection_rows(detector, targets, selected_preds, selected_indices, device):
    for sample_idx, target in enumerate(targets):
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
        for pred_idx, row in enumerate(det):
            if pred_idx >= int(raw_keep.shape[0]):
                raise RuntimeError(
                    "Faster R-CNN selected_indices is shorter than selected predictions. "
                    f"sample_idx={sample_idx}, pred_idx={pred_idx}, indices={int(raw_keep.shape[0])}"
                )
            raw_pred_idx = int(raw_keep[pred_idx].detach().cpu().item())
            cls_idx = int(row[5].detach().cpu().item()) if row.shape[0] > 5 else -1
            score = float(row[4].detach().cpu().item()) if row.shape[0] > 4 else 0.0
            pred_class = pred_class_name(detector, cls_idx)
            base = {
                "image_id": image_id,
                "image_path": image_path,
                "pred_idx": pred_idx,
                "raw_pred_idx": raw_pred_idx,
                "xmin": float(row[0].detach().cpu().item()),
                "ymin": float(row[1].detach().cpu().item()),
                "xmax": float(row[2].detach().cpu().item()),
                "ymax": float(row[3].detach().cpu().item()),
                "score": score,
                "pred_class": pred_class,
            }
            yield FasterRCNNDetectionRow(
                sample_idx=sample_idx,
                pred_idx=pred_idx,
                raw_pred_idx=raw_pred_idx,
                box=row,
                cls_idx=cls_idx,
                score=score,
                pred_class=pred_class,
                base=base,
            )


__all__ = ["FasterRCNNDetectionRow", "iter_faster_rcnn_detection_rows", "pred_class_name"]
