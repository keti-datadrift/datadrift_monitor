"""Deterministic image-id selection helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence


def unique_in_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def image_id_sort_key(image_id: Any) -> tuple:
    if isinstance(image_id, int):
        return (0, image_id)
    return (1, str(image_id))


def deterministic_random_sample(
    image_ids: Iterable[Any],
    budget: int,
    seed: int = 0,
) -> List[Any]:
    candidates = sorted(unique_in_order(image_ids), key=image_id_sort_key)
    if budget <= 0:
        return []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:min(budget, len(candidates))]


def top_k_by_score(scores: Dict[Any, float], budget: int) -> List[Any]:
    if budget <= 0:
        return []
    return ranked_ids_by_score(scores)[:budget]


def ranked_ids_by_score(scores: Dict[Any, float]) -> List[Any]:
    ranked = sorted(
        scores.items(),
        key=lambda item: (-float(item[1]), image_id_sort_key(item[0])),
    )
    return [image_id for image_id, _ in ranked]


def fill_to_budget(
    selected: Iterable[Any],
    candidate_image_ids: Iterable[Any],
    budget: int,
    seed: int = 0,
) -> List[Any]:
    """Return selected ids, deterministically refilled from remaining ids."""

    output = unique_in_order(selected)
    if budget <= 0:
        return []
    if len(output) >= budget:
        return output[:budget]

    selected_set = set(output)
    remaining = [
        image_id for image_id in candidate_image_ids
        if image_id not in selected_set
    ]
    output.extend(
        deterministic_random_sample(remaining, budget - len(output), seed=seed)
    )
    return output[:budget]


def ordered_scores_for_ids(image_ids: Sequence[Any], default_score: float = 0.0) -> Dict[Any, float]:
    return {image_id: default_score for image_id in image_ids}
