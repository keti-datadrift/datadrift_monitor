"""Core-set k-center acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from methods.common.coco_pool import image_ids, read_coco_json
from methods.common.feature_artifacts import filter_feature_artifact, load_feature_artifact
from methods.common.kcenter import greedy_k_center_select


def select_coreset_images(
    labeled_pool: Dict[str, Any],
    unlabeled_pool: Dict[str, Any],
    labeled_feature_artifact: Any,
    unlabeled_feature_artifact: Any,
    budget: int,
    batch_size: int = 512,
    center_batch_size: int = 2048,
) -> Dict[str, Any]:
    """Select unlabeled images using greedy k-center."""

    labeled_ids = image_ids(labeled_pool)
    unlabeled_ids = image_ids(unlabeled_pool)
    labeled_features = filter_feature_artifact(
        labeled_feature_artifact,
        labeled_ids,
        artifact_name='Core-set labeled feature artifact',
        require_all=True,
    )
    unlabeled_features = filter_feature_artifact(
        unlabeled_feature_artifact,
        unlabeled_ids,
        artifact_name='Core-set unlabeled feature artifact',
        require_all=True,
    )
    if labeled_features.image_features is None or unlabeled_features.image_features is None:
        raise ValueError('Core-set acquisition requires image_features in feature artifacts')
    selection = greedy_k_center_select(
        candidate_image_ids=unlabeled_features.image_ids,
        candidate_features=unlabeled_features.image_features,
        center_features=labeled_features.image_features,
        budget=budget,
        batch_size=batch_size,
        center_batch_size=center_batch_size,
    )

    metrics = dict(selection.get('metrics', {}))
    diagnostics = {
        'mode': 'coreset',
        'stage': 'kcenter',
        'budget': int(budget),
        'pool_counts': {
            'labeled_images': len(labeled_ids),
            'unlabeled_images': len(unlabeled_ids),
            'labeled_feature_records': labeled_features.image_count(),
            'unlabeled_feature_records': unlabeled_features.image_count(),
        },
        'feature_dim': int(unlabeled_features.image_features.shape[1])
        if unlabeled_features.image_features.ndim == 2 else None,
        'distance_summary': metrics.get('distance_summary', {}),
        'selected_count': len(selection['selected_image_ids']),
        'selected_image_ids': selection['selected_image_ids'],
    }
    return {
        'selected_image_ids': selection['selected_image_ids'],
        'candidate_records': selection['candidate_records'],
        'diagnostics': diagnostics,
        'mode': 'coreset',
        'stage': 'kcenter',
    }


def sample_coreset_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_features_npz: Path,
    unlabeled_features_npz: Path,
    budget: int,
    batch_size: int = 512,
    center_batch_size: int = 2048,
) -> Dict[str, Any]:
    """Run Core-set acquisition from COCO pools and feature artifacts."""

    labeled_pool = read_coco_json(Path(labeled_pool_json))
    unlabeled_pool = read_coco_json(Path(unlabeled_pool_json))
    labeled_features = load_feature_artifact(Path(labeled_features_npz))
    unlabeled_features = load_feature_artifact(Path(unlabeled_features_npz))
    result = select_coreset_images(
        labeled_pool=labeled_pool,
        unlabeled_pool=unlabeled_pool,
        labeled_feature_artifact=labeled_features,
        unlabeled_feature_artifact=unlabeled_features,
        budget=budget,
        batch_size=batch_size,
        center_batch_size=center_batch_size,
    )
    result['diagnostics']['inputs'] = {
        'labeled_pool_json': str(labeled_pool_json),
        'unlabeled_pool_json': str(unlabeled_pool_json),
        'labeled_features_npz': str(labeled_features_npz),
        'unlabeled_features_npz': str(unlabeled_features_npz),
    }
    return result
