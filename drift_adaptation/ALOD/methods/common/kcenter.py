"""Method-neutral greedy k-center selection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from methods.common.vectors import (
    squared_euclidean_to_centers,
    squared_euclidean_to_vector,
)


def _as_feature_matrix(features: Any, name: str) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError('%s must be a 2D feature matrix' % name)
    if not np.all(np.isfinite(values)):
        raise ValueError('%s contains non-finite values' % name)
    return values


def _distance_summary(values: np.ndarray, prefix: str) -> Dict[str, Optional[float]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            '%s_min' % prefix: None,
            '%s_mean' % prefix: None,
            '%s_max' % prefix: None,
        }
    return {
        '%s_min' % prefix: float(finite.min()),
        '%s_mean' % prefix: float(finite.mean()),
        '%s_max' % prefix: float(finite.max()),
    }


def greedy_k_center_select(
    candidate_image_ids: Sequence[Any],
    candidate_features: Any,
    budget: int,
    center_features: Optional[Any] = None,
    batch_size: int = 512,
    center_batch_size: int = 2048,
) -> Dict[str, Any]:
    """Select candidates by farthest-first k-center greedy.

    ``center_features`` represents already labeled images and any selected
    images are added to this center set after each greedy step.
    """

    candidate_ids = list(candidate_image_ids)
    features = _as_feature_matrix(candidate_features, 'candidate_features')
    if len(candidate_ids) != features.shape[0]:
        raise ValueError(
            'candidate_image_ids has %d ids but candidate_features has %d rows'
            % (len(candidate_ids), features.shape[0]))

    centers = None
    if center_features is not None:
        centers = _as_feature_matrix(center_features, 'center_features')
        if centers.shape[1] != features.shape[1]:
            raise ValueError(
                'center feature dim %d does not match candidate feature dim %d'
                % (centers.shape[1], features.shape[1]))
    target = min(int(budget), len(candidate_ids))
    if target <= 0:
        return {
            'selected_image_ids': [],
            'candidate_records': [],
            'metrics': {
                'selected_count': 0,
                'distance_summary': {},
            },
        }

    if centers is None or centers.shape[0] == 0:
        min_distances = np.full((features.shape[0], ), np.inf, dtype=np.float64)
    else:
        min_distances = squared_euclidean_to_centers(
            features,
            centers,
            batch_size=batch_size,
            center_batch_size=center_batch_size,
        )
    initial_min_distances = min_distances.copy()

    selected_indices: List[int] = []
    available = np.ones((features.shape[0], ), dtype=np.bool_)
    for rank in range(1, target + 1):
        masked = np.where(available, min_distances, -np.inf)
        selected_index = int(np.argmax(masked))
        if not np.isfinite(masked[selected_index]):
            remaining = np.flatnonzero(available)
            if remaining.size == 0:
                break
            selected_index = int(remaining[0])
        selected_indices.append(selected_index)
        available[selected_index] = False
        new_distances = squared_euclidean_to_vector(features, features[selected_index])
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[selected_indices] = -np.inf

    candidate_records = []
    final_distances = min_distances.copy()
    for index, image_id in enumerate(candidate_ids):
        candidate_records.append({
            'image_id': image_id,
            'rank': index + 1,
            'score': float(initial_min_distances[index])
            if np.isfinite(initial_min_distances[index]) else None,
            'source': 'kcenter',
            'components': {
                'final_min_distance': float(final_distances[index])
                if np.isfinite(final_distances[index]) else None,
            },
        })

    return {
        'selected_image_ids': [candidate_ids[index] for index in selected_indices],
        'candidate_records': candidate_records,
        'metrics': {
            'selected_count': len(selected_indices),
            'distance_summary': {
                **_distance_summary(initial_min_distances, 'initial'),
                **_distance_summary(final_distances, 'final'),
            },
        },
    }
