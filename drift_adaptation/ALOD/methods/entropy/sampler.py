"""Image-level entropy sampler for detector bbox JSON outputs.

The preferred input is a list of per-detection dictionaries containing
``image_id`` and an uncertainty-like field such as ``cls_uncertainty``. If only
``score`` is available, the sampler uses binary entropy of the confidence as a
detector-agnostic fallback. If no result file is available, it falls back to a
seeded random sample so the acquisition loop remains deterministic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from methods.common.coco_pool import image_ids, load_coco_pool
from methods.common.detections import load_detection_records
from methods.common.selection import (
    deterministic_random_sample,
    ordered_scores_for_ids,
    top_k_by_score,
)

PoolInput = Union[Path, Dict[str, Any]]


def _binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _distribution_entropy(scores: Iterable[float]) -> float:
    values = [max(float(score), 0.0) for score in scores]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values if value > 0.0)


def _detection_uncertainty(det: Dict[str, Any]) -> float:
    for key in ('cls_uncertainty', 'uncertainty', 'entropy', 'score_uncertainty'):
        if key in det:
            return float(det[key])
    if 'scores' in det and isinstance(det['scores'], list):
        return _distribution_entropy(det['scores'])
    if 'score' in det:
        return _binary_entropy(float(det['score']))
    return 0.0


def _load_results(results_json: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if results_json is None or not results_json.exists():
        return None
    return load_detection_records(results_json, schema_name='entropy result')


def aggregate_image_scores(
    pool: PoolInput,
    results_json: Optional[Path],
) -> Optional[Dict[Any, float]]:
    pool_ids = image_ids(load_coco_pool(pool))
    pool_id_set = set(pool_ids)
    results = _load_results(results_json)
    if results is None:
        return None

    scores = ordered_scores_for_ids(pool_ids, default_score=0.0)
    for det in results:
        image_id = det.get('image_id')
        if image_id in pool_id_set:
            scores[image_id] += _detection_uncertainty(det)
    return scores


def sample(
    pool: PoolInput,
    budget: int,
    results_json: Optional[Path] = None,
    seed: int = 0,
) -> List[Any]:
    scores = aggregate_image_scores(pool, results_json)
    if scores is None:
        return deterministic_random_sample(image_ids(load_coco_pool(pool)), budget, seed=seed)
    return top_k_by_score(scores, budget)
