from dataclasses import dataclass
from types import MethodType

import torch
import torch.nn.functional as F

from models.yolov10.core import xywh2xyxy


@dataclass
class YoloV10CandidateCache:
    raw_xyxy: torch.Tensor
    raw_probs: torch.Tensor


def _box_iou_1vn(box, boxes):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=box.dtype, device=box.device)
    x1 = torch.maximum(box[:, 0], boxes[:, 0])
    y1 = torch.maximum(box[:, 1], boxes[:, 1])
    x2 = torch.minimum(box[:, 2], boxes[:, 2])
    y2 = torch.minimum(box[:, 3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-12)


def build_yolov10_candidate_cache(forward, sample_idx):
    with torch.no_grad():
        boxes_xywh = forward.decoded_prediction[int(sample_idx), :, :4].detach().float()
        return YoloV10CandidateCache(
            raw_xyxy=xywh2xyxy(boxes_xywh),
            raw_probs=torch.sigmoid(forward.raw_logits[int(sample_idx)].detach().float()),
        )


def yolov10_candidate_mask_from_cache(cache, final_xyxy, class_idx, score_threshold, iou_threshold):
    class_idx = int(class_idx)
    if class_idx < 0 or class_idx >= int(cache.raw_probs.shape[1]):
        raise RuntimeError(f"YOLOv10 candidate class index out of range: {class_idx}")
    scores = cache.raw_probs[:, class_idx]
    score_mask = scores >= float(score_threshold)
    ious = torch.zeros((cache.raw_xyxy.shape[0],), dtype=cache.raw_xyxy.dtype, device=cache.raw_xyxy.device)
    candidate_mask = torch.zeros((cache.raw_xyxy.shape[0],), dtype=torch.bool, device=cache.raw_xyxy.device)
    if bool(score_mask.any()):
        candidate_indices = torch.nonzero(score_mask, as_tuple=False).flatten()
        candidate_ious = _box_iou_1vn(final_xyxy.detach().float().view(1, 4).to(cache.raw_xyxy.device), cache.raw_xyxy[candidate_indices])
        ious[candidate_indices] = candidate_ious
        keep_indices = candidate_indices[candidate_ious > float(iou_threshold)]
        if keep_indices.numel() > 0:
            candidate_mask[keep_indices] = True
    return candidate_mask, ious


def yolov10_raw_logits_for_item(forward, item, device):
    return forward.raw_logits[item["sample_idx"], item["raw_box_idx"]].to(device=device, dtype=torch.float32)


def yolov10_raw_probs_for_item(forward, item, device):
    return torch.sigmoid(yolov10_raw_logits_for_item(forward, item, device))


def source_point_box(source_points, raw_box_idx, device):
    point = source_points[int(raw_box_idx)].to(device=device, dtype=torch.float32)
    return torch.stack([point[0], point[1], point[0], point[1]])


def gather_yolov10_feature_matrix(forward, items, device):
    num_classes = int(forward.raw_logits.shape[-1])
    if not items:
        return torch.zeros((0, 4 + num_classes), dtype=torch.float32, device=device)
    source_device = forward.decoded_prediction.device
    sample_idx = torch.as_tensor([item["sample_idx"] for item in items], dtype=torch.long, device=source_device)
    raw_box_idx = torch.as_tensor([item["raw_box_idx"] for item in items], dtype=torch.long, device=source_device)
    boxes_xywh = forward.decoded_prediction[sample_idx, raw_box_idx, :4].detach().float()
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    probs = torch.sigmoid(forward.raw_logits[sample_idx, raw_box_idx].detach().float())
    return torch.cat([boxes_xyxy.to(device=device), probs.to(device=device)], dim=1)


def enable_forced_yolov10_dropout(model, dropout_rate):
    head = model.model[-1]
    original = head.forward_feat
    p = float(dropout_rate)

    def patched(self, x, cv2, cv3):
        outputs = original(x, cv2, cv3)
        return [F.dropout(out, p=p, training=True) for out in outputs]

    head.forward_feat = MethodType(patched, head)

    def restore():
        head.forward_feat = original

    return restore


__all__ = [
    "YoloV10CandidateCache",
    "build_yolov10_candidate_cache",
    "enable_forced_yolov10_dropout",
    "gather_yolov10_feature_matrix",
    "source_point_box",
    "yolov10_candidate_mask_from_cache",
    "yolov10_raw_logits_for_item",
    "yolov10_raw_probs_for_item",
]
