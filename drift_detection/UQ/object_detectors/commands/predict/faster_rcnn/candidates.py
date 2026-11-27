from dataclasses import dataclass

import torch

from commands.predict.common import _xywh_to_xyxy_tensor


def _box_iou_1vN_tensor(box_xyxy, boxes_xyxy):
    if boxes_xyxy is None or boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=box_xyxy.dtype, device=box_xyxy.device)
    box = box_xyxy.view(1, 4).to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    x1 = torch.maximum(box[:, 0], boxes_xyxy[:, 0])
    y1 = torch.maximum(box[:, 1], boxes_xyxy[:, 1])
    x2 = torch.minimum(box[:, 2], boxes_xyxy[:, 2])
    y2 = torch.minimum(box[:, 3], boxes_xyxy[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)
    area2 = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * (
        boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
    ).clamp(min=0)
    return inter / (area1 + area2 - inter).clamp(min=1e-12)


@dataclass
class FasterRCNNROICandidateCache:
    raw_xyxy: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    probs: torch.Tensor
    logits: torch.Tensor | None = None


@dataclass
class FasterRCNNRPNCandidateCache:
    boxes_xyxy: torch.Tensor
    objectness_logits: torch.Tensor
    anchors: torch.Tensor
    bbox_deltas: torch.Tensor


def build_faster_rcnn_roi_candidate_cache(raw_prediction_img, raw_logits_img=None, detach=True):
    pred = raw_prediction_img.detach() if detach else raw_prediction_img
    pred = pred.float()
    if pred.numel() == 0:
        device = pred.device
        logits = None
        if raw_logits_img is not None:
            logits = raw_logits_img.detach().float() if detach else raw_logits_img.float()
        return FasterRCNNROICandidateCache(
            raw_xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            scores=torch.zeros((0,), dtype=torch.float32, device=device),
            labels=torch.zeros((0,), dtype=torch.long, device=device),
            probs=torch.zeros((0, 0), dtype=torch.float32, device=device),
            logits=logits,
        )
    probs = pred[:, 6:].float() if pred.shape[1] > 6 else torch.zeros((pred.shape[0], 0), dtype=torch.float32, device=pred.device)
    logits = None
    if raw_logits_img is not None:
        logits = raw_logits_img.detach().float() if detach else raw_logits_img.float()
    return FasterRCNNROICandidateCache(
        raw_xyxy=_xywh_to_xyxy_tensor(pred[:, :4]),
        scores=pred[:, 4].float(),
        labels=pred[:, 5].long() if pred.shape[1] > 5 else torch.zeros((pred.shape[0],), dtype=torch.long, device=pred.device),
        probs=probs,
        logits=logits,
    )


def faster_rcnn_roi_candidate_mask_from_cache(cache, final_xyxy, class_idx, score_threshold, iou_threshold):
    class_idx = int(class_idx)
    score_class_mask = (cache.scores >= float(score_threshold)) & (cache.labels == class_idx)
    ious = torch.zeros((cache.raw_xyxy.shape[0],), dtype=cache.raw_xyxy.dtype, device=cache.raw_xyxy.device)
    candidate_mask = torch.zeros((cache.raw_xyxy.shape[0],), dtype=torch.bool, device=cache.raw_xyxy.device)
    if bool(score_class_mask.any()):
        candidate_indices = torch.nonzero(score_class_mask, as_tuple=False).flatten()
        candidate_ious = _box_iou_1vN_tensor(
            final_xyxy.detach().float().view(1, 4).to(cache.raw_xyxy.device),
            cache.raw_xyxy[candidate_indices],
        )
        ious[candidate_indices] = candidate_ious
        keep_indices = candidate_indices[candidate_ious > float(iou_threshold)]
        if keep_indices.numel() > 0:
            candidate_mask[keep_indices] = True
    return candidate_mask, ious


def build_faster_rcnn_rpn_candidate_cache(boxes_xyxy, objectness_logits, anchors, bbox_deltas, detach=True):
    return FasterRCNNRPNCandidateCache(
        boxes_xyxy=boxes_xyxy.detach().float() if detach else boxes_xyxy.float(),
        objectness_logits=objectness_logits.detach().float() if detach else objectness_logits.float(),
        anchors=anchors.detach().float() if detach else anchors.float(),
        bbox_deltas=bbox_deltas.detach().float() if detach else bbox_deltas.float(),
    )


def faster_rcnn_rpn_candidate_mask_from_cache(cache, final_xyxy, obj_threshold, iou_threshold):
    scores = torch.sigmoid(cache.objectness_logits.reshape(-1))
    score_mask = scores >= float(obj_threshold)
    ious = torch.zeros((cache.boxes_xyxy.shape[0],), dtype=cache.boxes_xyxy.dtype, device=cache.boxes_xyxy.device)
    candidate_mask = torch.zeros((cache.boxes_xyxy.shape[0],), dtype=torch.bool, device=cache.boxes_xyxy.device)
    if bool(score_mask.any()):
        candidate_indices = torch.nonzero(score_mask, as_tuple=False).flatten()
        candidate_ious = _box_iou_1vN_tensor(
            final_xyxy.detach().float().view(1, 4).to(cache.boxes_xyxy.device),
            cache.boxes_xyxy[candidate_indices],
        )
        ious[candidate_indices] = candidate_ious
        keep_indices = candidate_indices[candidate_ious > float(iou_threshold)]
        if keep_indices.numel() > 0:
            candidate_mask[keep_indices] = True
    return candidate_mask, ious


def match_same_class_highest_iou(final_xyxy, class_idx, cache):
    class_mask = cache.labels == int(class_idx)
    if not bool(class_mask.any()):
        return None
    candidate_indices = torch.nonzero(class_mask, as_tuple=False).flatten()
    ious = _box_iou_1vN_tensor(
        final_xyxy.detach().float().view(1, 4).to(cache.raw_xyxy.device),
        cache.raw_xyxy[candidate_indices],
    )
    best_pos = int(torch.argmax(ious).detach().cpu().item())
    return candidate_indices[best_pos]


__all__ = [
    "FasterRCNNROICandidateCache",
    "FasterRCNNRPNCandidateCache",
    "build_faster_rcnn_roi_candidate_cache",
    "build_faster_rcnn_rpn_candidate_cache",
    "faster_rcnn_roi_candidate_mask_from_cache",
    "faster_rcnn_rpn_candidate_mask_from_cache",
    "match_same_class_highest_iou",
]
