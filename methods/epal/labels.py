"""TIDE-style training labels for EPAL predictors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from methods.common.matching import bbox_iou_xywh
from methods.epal.features import (
    classification_feature_vector,
    localization_feature_vector,
    miss_feature_vector,
)


def _group_annotations_by_image(coco_data: Mapping[str, Any]) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        if ann.get('image_id') is None or ann.get('bbox') is None or ann.get('category_id') is None:
            continue
        grouped[ann['image_id']].append(dict(ann))
    return dict(grouped)


def _gt_key(index: int, gt: Mapping[str, Any]) -> Any:
    return gt.get('id') if gt.get('id') is not None else index


def _best_gt_overlaps(
    detection: Mapping[str, Any],
    ground_truth: Iterable[Mapping[str, Any]],
) -> Tuple[float, float, float, Optional[Any]]:
    category_id = detection.get('category_id')
    u_same = 0.0
    u_diff = 0.0
    best_iou = 0.0
    best_gt_id = None
    for index, gt in enumerate(ground_truth):
        iou = bbox_iou_xywh(detection['bbox'], gt['bbox'])
        if iou > best_iou:
            best_iou = iou
            best_gt_id = _gt_key(index, gt)
        if gt.get('category_id') == category_id:
            u_same = max(u_same, iou)
        else:
            u_diff = max(u_diff, iou)
    return u_same, u_diff, max(u_same, u_diff), best_gt_id


def build_training_labels(
    feature_records: Iterable[Mapping[str, Any]],
    labeled_pool: Mapping[str, Any],
    foreground_iou_threshold: float = 0.5,
    background_iou_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Create predictor samples and labeled error-count statistics."""

    gt_by_image = _group_annotations_by_image(labeled_pool)
    detection_examples = []
    image_examples = []

    for record in feature_records:
        image_id = record['image_id']
        detections = list(record.get('final_detections', []) or [])
        ground_truth = gt_by_image.get(image_id, [])
        explained_gt_ids = set()
        n_cls = 0
        n_loc = 0

        for detection in detections:
            u_same, u_diff, u_max, best_gt_id = _best_gt_overlaps(detection, ground_truth)
            is_foreground = u_max >= background_iou_threshold
            y_cls = int(u_diff > u_same) if is_foreground else 0
            y_loc = int(u_max < foreground_iou_threshold) if is_foreground else 0
            if is_foreground and best_gt_id is not None:
                explained_gt_ids.add(best_gt_id)
            if is_foreground:
                n_cls += y_cls
                n_loc += y_loc

            detection_examples.append({
                'image_id': image_id,
                'category_id': detection.get('category_id'),
                'classification_feature': classification_feature_vector(detection),
                'localization_feature': localization_feature_vector(detection),
                'y_cls': y_cls,
                'y_loc': y_loc,
                'is_foreground': bool(is_foreground),
                'u_same': float(u_same),
                'u_diff': float(u_diff),
                'u_max': float(u_max),
            })

        gt_ids = {_gt_key(index, gt) for index, gt in enumerate(ground_truth)}
        n_miss = len(gt_ids.difference(explained_gt_ids))
        image_examples.append({
            'image_id': image_id,
            'miss_feature': miss_feature_vector(record),
            'n_cls': int(n_cls),
            'n_loc': int(n_loc),
            'n_miss': int(n_miss),
            'gt_count': len(ground_truth),
            'final_detection_count': len(detections),
        })

    foreground_count = sum(1 for example in detection_examples if example['is_foreground'])
    return {
        'detection_examples': detection_examples,
        'image_examples': image_examples,
        'summary': {
            'image_count': len(image_examples),
            'detection_count': len(detection_examples),
            'foreground_detection_count': foreground_count,
            'foreground_iou_threshold': float(foreground_iou_threshold),
            'background_iou_threshold': float(background_iou_threshold),
        },
    }
