"""Detection-to-ground-truth matching helpers for active learning methods."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in bbox[:4]]
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def bbox_iou_xywh(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = xywh_to_xyxy(lhs)
    rx1, ry1, rx2, ry2 = xywh_to_xyxy(rhs)

    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0

    lhs_area = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0)
    rhs_area = max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0)
    union = lhs_area + rhs_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def annotations_from_coco(coco_or_annotations: Any) -> List[Dict[str, Any]]:
    if isinstance(coco_or_annotations, dict):
        return list(coco_or_annotations.get('annotations', []))
    return list(coco_or_annotations)


def _group_ground_truth(
    annotations: Iterable[Dict[str, Any]],
) -> Dict[Tuple[Any, Any], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if 'image_id' not in ann or 'category_id' not in ann or 'bbox' not in ann:
            continue
        grouped[(ann['image_id'], ann['category_id'])].append(ann)
    return grouped


def _gt_match_key(gt: Dict[str, Any]) -> Any:
    return gt['id'] if gt.get('id') is not None else id(gt)


def _best_unmatched_gt(
    det: Dict[str, Any],
    gt_group: List[Dict[str, Any]],
    matched_gt_ids: set,
) -> Tuple[Optional[Dict[str, Any]], float]:
    best_gt = None
    best_iou = 0.0
    for gt in gt_group:
        gt_key = _gt_match_key(gt)
        if gt_key in matched_gt_ids:
            continue
        iou = bbox_iou_xywh(det['bbox'], gt['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    return best_gt, best_iou


def match_detections_to_ground_truth(
    detections: Iterable[Dict[str, Any]],
    ground_truth: Any,
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Label detections as TP/FP using greedy COCO-style class-wise matching.

    Detections are grouped by `(image_id, category_id)` and processed by
    descending confidence. Each ground-truth box can be matched at most once.
    """

    gt_by_key = _group_ground_truth(annotations_from_coco(ground_truth))
    dets = [dict(det) for det in detections]
    dets.sort(key=lambda det: float(det.get('score', det.get('confidence', 0.0))), reverse=True)

    matched_by_key: Dict[Tuple[Any, Any], set] = defaultdict(set)
    matched_records = []
    for det in dets:
        key = (det.get('image_id'), det.get('category_id'))
        target = 0
        matched_gt_id = None
        matched_iou = 0.0

        if det.get('bbox') is not None:
            best_gt, best_iou = _best_unmatched_gt(det, gt_by_key.get(key, []), matched_by_key[key])
            if best_gt is not None and best_iou >= iou_threshold:
                target = 1
                matched_gt_id = best_gt.get('id')
                matched_iou = best_iou
                matched_by_key[key].add(_gt_match_key(best_gt))

        record = dict(det)
        record['target'] = target
        record['matched_gt_id'] = matched_gt_id
        record['matched_iou'] = matched_iou
        matched_records.append(record)

    return matched_records
