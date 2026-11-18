"""Pure PAL GUIDE scoring helpers.

This module intentionally has no MMDetection dependency. It operates on PAL
detection JSON records produced by inference and returns plain Python data
structures that can be consumed by ``methods.pal.acquisition``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from methods.common.selection import image_id_sort_key


DEFAULT_ALPHA = 0.9
DEFAULT_BETA = 0.04
DEFAULT_GAMMA = 0.02


def _candidate_sort_key(candidate: Mapping[str, Any], score_key: str) -> tuple:
    return (
        -float(candidate.get(score_key, 0.0)),
        image_id_sort_key(candidate.get('image_id')),
        image_id_sort_key(candidate.get('category_id')),
    )


def _as_probability_vector(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError('class_scores must contain at least one value')
    if not np.all(np.isfinite(values)):
        raise ValueError('class_scores must be finite')
    if np.any(values < 0.0):
        raise ValueError('class_scores must be non-negative')

    total = float(values.sum())
    if total <= 0.0:
        return np.zeros_like(values)
    return values / total


def compute_class_weights(
    labeled_counts: Mapping[Any, int],
    unlabeled_counts: Mapping[Any, int],
    categories: Optional[Iterable[Any]] = None,
) -> Dict[Any, float]:
    """Compute PAL Eq. 5 class rarity weights."""

    if categories is None:
        category_ids = set(labeled_counts) | set(unlabeled_counts)
    else:
        category_ids = set(categories)

    total_labeled = float(sum(max(int(count), 0) for count in labeled_counts.values()))
    total_unlabeled = float(sum(max(int(count), 0) for count in unlabeled_counts.values()))

    weights: Dict[Any, float] = {}
    for category_id in sorted(category_ids, key=image_id_sort_key):
        labeled_ratio = (
            max(int(labeled_counts.get(category_id, 0)), 0) / total_labeled
            if total_labeled > 0.0 else 0.0
        )
        unlabeled_ratio = (
            max(int(unlabeled_counts.get(category_id, 0)), 0) / total_unlabeled
            if total_unlabeled > 0.0 else 0.0
        )
        weights[category_id] = max(1.0 - 0.5 * (labeled_ratio + unlabeled_ratio), 0.0)
    return weights


def allocate_class_budgets(
    class_weights: Mapping[Any, float],
    class_capacities: Mapping[Any, int],
    total_budget: int,
) -> Dict[Any, int]:
    """Allocate integer per-class budgets with deterministic largest remainder."""

    category_ids = sorted(class_capacities, key=image_id_sort_key)
    budgets = {category_id: 0 for category_id in category_ids}
    if total_budget <= 0 or not category_ids:
        return budgets

    positive_weights = {
        category_id: max(float(class_weights.get(category_id, 0.0)), 0.0)
        for category_id in category_ids
        if int(class_capacities.get(category_id, 0)) > 0
    }
    weight_sum = sum(positive_weights.values())
    if weight_sum <= 0.0:
        positive_weights = {
            category_id: 1.0
            for category_id in category_ids
            if int(class_capacities.get(category_id, 0)) > 0
        }
        weight_sum = float(len(positive_weights))
    if weight_sum <= 0.0:
        return budgets

    raw = {
        category_id: total_budget * positive_weights.get(category_id, 0.0) / weight_sum
        for category_id in category_ids
    }
    for category_id in category_ids:
        capacity = max(int(class_capacities.get(category_id, 0)), 0)
        budgets[category_id] = min(int(math.floor(raw[category_id])), capacity)

    remaining = total_budget - sum(budgets.values())
    ranked = sorted(
        category_ids,
        key=lambda category_id: (
            -(raw[category_id] - math.floor(raw[category_id])),
            -positive_weights.get(category_id, 0.0),
            image_id_sort_key(category_id),
        ),
    )
    while remaining > 0:
        progressed = False
        for category_id in ranked:
            capacity = max(int(class_capacities.get(category_id, 0)), 0)
            if budgets[category_id] < capacity:
                budgets[category_id] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return budgets


def class_entropy_from_scores(class_scores: Sequence[float]) -> float:
    """Compute entropy from a detection's per-class score vector."""

    probabilities = _as_probability_vector(class_scores)
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def compute_cwie(
    image_detections: Iterable[Mapping[str, Any]],
    class_weights: Mapping[Any, float],
    class_scores_key: str = 'class_scores',
) -> float:
    """Compute PAL Eq. 7 CWIE for one image from detection class scores."""

    score = 0.0
    for det in image_detections:
        if class_scores_key not in det:
            raise KeyError('Detection is missing %s for CWIE' % class_scores_key)
        category_id = det.get('category_id')
        weight = max(float(class_weights.get(category_id, 0.0)), 0.0)
        score += weight * class_entropy_from_scores(det[class_scores_key])
    return float(score)


def unique_categories_from_detections(
    image_detections: Iterable[Mapping[str, Any]],
) -> List[Any]:
    categories = {
        det.get('category_id')
        for det in image_detections
        if det.get('category_id') is not None
    }
    return sorted(categories, key=image_id_sort_key)


def compute_rcdi(
    image_categories: Iterable[Any],
    class_weights: Mapping[Any, float],
) -> float:
    """Compute PAL Eq. 9 RCDI for one image from unique predicted classes."""

    return float(sum(max(float(class_weights.get(category_id, 0.0)), 0.0)
                     for category_id in set(image_categories)))


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    first_vector = np.asarray(first, dtype=np.float64).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float64).reshape(-1)
    if first_vector.size == 0 or second_vector.size == 0:
        raise ValueError('Embeddings must be non-empty')
    if first_vector.shape != second_vector.shape:
        raise ValueError('Embedding shapes must match')
    first_norm = float(np.linalg.norm(first_vector))
    second_norm = float(np.linalg.norm(second_vector))
    if first_norm <= 0.0 or second_norm <= 0.0:
        raise ValueError('Embeddings must have non-zero norm')
    return float(np.dot(first_vector, second_vector) / (first_norm * second_norm))


def compute_rcsp(
    ranked_image_ids: Sequence[Any],
    image_embeddings: Mapping[Any, Sequence[float]],
) -> Dict[Any, float]:
    """Compute PAL Eq. 10 RCSP for one LIUS-ranked candidate list."""

    scores: Dict[Any, float] = {}
    previous_image_ids: List[Any] = []
    for image_id in ranked_image_ids:
        if image_id not in image_embeddings:
            raise KeyError('Missing embedding for image_id %s' % image_id)
        if not previous_image_ids:
            scores[image_id] = 1.0
        else:
            max_similarity = max(
                cosine_similarity(image_embeddings[image_id], image_embeddings[previous_id])
                for previous_id in previous_image_ids
            )
            scores[image_id] = float(1.0 - max_similarity)
        previous_image_ids.append(image_id)
    return scores


def min_max_normalize_dict(
    values: Mapping[Any, float],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Dict[Any, float]:
    """Return deterministic min-max normalized values for a mapping."""

    if not values:
        return {}
    value_list = [float(value) for value in values.values()]
    min_value = min(value_list) if minimum is None else float(minimum)
    max_value = max(value_list) if maximum is None else float(maximum)
    denom = max_value - min_value
    if denom <= 0.0:
        return {key: 0.0 for key in values}
    return {
        key: max((float(value) - min_value) / denom, 0.0)
        for key, value in values.items()
    }


def compute_pal_score(
    lius: float,
    cwie: float,
    rcdi: float,
    rcsp: float,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Compute PAL Eq. 11 final candidate score."""

    return float(alpha * lius + gamma * rcsp + beta * (cwie + rcdi))


def group_detections_by_image(
    detections: Iterable[Mapping[str, Any]],
) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for det in detections:
        image_id = det.get('image_id')
        if image_id is not None:
            grouped[image_id].append(dict(det))
    return dict(grouped)


def build_class_candidates(
    scored_detections: Iterable[Mapping[str, Any]],
    class_budgets: Mapping[Any, int],
    candidate_multiplier: int = 2,
    score_key: str = 'lius_score',
) -> Dict[Any, List[Dict[str, Any]]]:
    """Build top ``candidate_multiplier * b_c`` image candidates per class."""

    best_by_class_image: Dict[Any, Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    for det in scored_detections:
        category_id = det.get('category_id')
        image_id = det.get('image_id')
        if category_id is None or image_id is None:
            continue
        record = dict(det)
        score = float(record.get(score_key, 0.0))
        current = best_by_class_image[category_id].get(image_id)
        if current is None or score > float(current.get(score_key, 0.0)):
            best_by_class_image[category_id][image_id] = record

    candidates: Dict[Any, List[Dict[str, Any]]] = {}
    for category_id in sorted(class_budgets, key=image_id_sort_key):
        limit = max(int(class_budgets.get(category_id, 0)), 0) * candidate_multiplier
        if limit <= 0:
            candidates[category_id] = []
            continue
        ranked = sorted(
            best_by_class_image.get(category_id, {}).values(),
            key=lambda record: _candidate_sort_key(record, score_key),
        )
        class_candidates = []
        for rank, record in enumerate(ranked[:limit], start=1):
            candidate = dict(record)
            candidate['candidate_class_id'] = category_id
            candidate['candidate_rank'] = rank
            class_candidates.append(candidate)
        candidates[category_id] = class_candidates
    return candidates


def score_guide_candidates(
    candidates_by_class: Mapping[Any, Sequence[Mapping[str, Any]]],
    image_detections: Mapping[Any, Sequence[Mapping[str, Any]]],
    class_weights: Mapping[Any, float],
    image_embeddings: Mapping[Any, Sequence[float]],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    lius_key: str = 'lius_score',
) -> Dict[Any, List[Dict[str, Any]]]:
    """Attach CWIE, RCDI, RCSP, and final PAL scores to class candidates."""

    scored_by_class: Dict[Any, List[Dict[str, Any]]] = {}
    for category_id in sorted(candidates_by_class, key=image_id_sort_key):
        class_candidates = [dict(candidate) for candidate in candidates_by_class[category_id]]
        ranked_image_ids = [candidate.get('image_id') for candidate in class_candidates]
        rcsp_values = compute_rcsp(ranked_image_ids, image_embeddings) if ranked_image_ids else {}

        cwie_raw = {}
        rcdi_raw = {}
        for candidate in class_candidates:
            image_id = candidate.get('image_id')
            detections = image_detections.get(image_id, [candidate])
            cwie_raw[image_id] = compute_cwie(detections, class_weights)
            rcdi_raw[image_id] = compute_rcdi(
                unique_categories_from_detections(detections),
                class_weights,
            )

        cwie_norm = min_max_normalize_dict(cwie_raw, minimum=0.0)
        rcdi_norm = min_max_normalize_dict(rcdi_raw, minimum=0.0)

        scored = []
        for candidate in class_candidates:
            image_id = candidate.get('image_id')
            record = dict(candidate)
            record['guide_cwie'] = cwie_norm.get(image_id, 0.0)
            record['guide_rcdi'] = rcdi_norm.get(image_id, 0.0)
            record['guide_rcsp'] = rcsp_values.get(image_id, 0.0)
            record['pal_score'] = compute_pal_score(
                lius=float(record.get(lius_key, 0.0)),
                cwie=record['guide_cwie'],
                rcdi=record['guide_rcdi'],
                rcsp=record['guide_rcsp'],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            scored.append(record)
        scored_by_class[category_id] = sorted(
            scored,
            key=lambda record: _candidate_sort_key(record, 'pal_score'),
        )
    return scored_by_class


def select_deduplicated_candidates(
    candidates_by_class: Mapping[Any, Sequence[Mapping[str, Any]]],
    class_budgets: Mapping[Any, int],
    total_budget: int,
    class_weights: Optional[Mapping[Any, float]] = None,
    score_key: str = 'pal_score',
) -> Dict[str, Any]:
    """Select unique images from per-class PAL candidates and refill deficits."""

    if class_weights is None:
        class_weights = {}

    selected_image_ids = []
    selected_images = set()
    selected_candidates = []
    class_selected_counts = {category_id: 0 for category_id in class_budgets}

    class_order = sorted(
        class_budgets,
        key=lambda category_id: (-float(class_weights.get(category_id, 0.0)),
                                 image_id_sort_key(category_id)),
    )

    def add_candidate(candidate: Mapping[str, Any], category_id: Any, reason: str) -> bool:
        image_id = candidate.get('image_id')
        if image_id is None or image_id in selected_images:
            return False
        record = dict(candidate)
        record['selected_class_id'] = category_id
        record['selection_reason'] = reason
        selected_images.add(image_id)
        selected_image_ids.append(image_id)
        selected_candidates.append(record)
        return True

    for category_id in class_order:
        class_budget = max(int(class_budgets.get(category_id, 0)), 0)
        if class_budget <= 0:
            continue
        ranked = sorted(
            candidates_by_class.get(category_id, []),
            key=lambda candidate: _candidate_sort_key(candidate, score_key),
        )
        for candidate in ranked:
            if class_selected_counts[category_id] >= class_budget:
                break
            if add_candidate(candidate, category_id, 'class_budget'):
                class_selected_counts[category_id] += 1
                if len(selected_image_ids) >= total_budget:
                    break
        if len(selected_image_ids) >= total_budget:
            break

    if len(selected_image_ids) < total_budget:
        best_by_image: Dict[Any, Dict[str, Any]] = {}
        for category_id, class_candidates in candidates_by_class.items():
            for candidate in class_candidates:
                image_id = candidate.get('image_id')
                if image_id is None or image_id in selected_images:
                    continue
                record = dict(candidate)
                record['candidate_class_id'] = category_id
                current = best_by_image.get(image_id)
                if (
                    current is None
                    or _candidate_sort_key(record, score_key) < _candidate_sort_key(current, score_key)
                ):
                    best_by_image[image_id] = record

        refill_candidates = sorted(
            best_by_image.values(),
            key=lambda candidate: _candidate_sort_key(candidate, score_key),
        )
        for candidate in refill_candidates:
            add_candidate(candidate, candidate.get('candidate_class_id'), 'refill')
            if len(selected_image_ids) >= total_budget:
                break

    return {
        'selected_image_ids': selected_image_ids[:total_budget],
        'selected_candidates': selected_candidates[:total_budget],
        'class_selected_counts': class_selected_counts,
        'unfilled_budget': max(total_budget - len(selected_image_ids), 0),
    }
