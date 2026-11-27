"""MIAL image scoring helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence

from methods.common.image_identity import normalize_image_id
from methods.common.selection import ranked_ids_by_score


def _score_summary(scores: Iterable[float]) -> Dict[str, float]:
    values = [float(score) for score in scores]
    if not values:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0}
    return {
        'min': min(values),
        'max': max(values),
        'mean': sum(values) / len(values),
    }


def rank_mial_candidates(
    records: Sequence[Mapping[str, Any]],
    unlabeled_image_ids: Iterable[Any],
) -> Dict[str, Any]:
    """Rank unlabeled images by MI-AOD instance-discrepancy uncertainty."""

    pool_ids = [normalize_image_id(image_id) for image_id in unlabeled_image_ids]
    pool_id_set = set(pool_ids)
    scores: Dict[Any, float] = {}
    record_by_id: Dict[Any, Mapping[str, Any]] = {}
    duplicates = []
    for record in records:
        image_id = normalize_image_id(record['image_id'])
        if image_id not in pool_id_set:
            continue
        if image_id in scores:
            duplicates.append(image_id)
            continue
        scores[image_id] = float(record['score'])
        record_by_id[image_id] = record

    if duplicates:
        raise ValueError(
            'MIAL uncertainty artifact contains duplicate image ids: %s'
            % sorted(set(duplicates), key=str)[:10]
        )
    missing = [image_id for image_id in pool_ids if image_id not in scores]
    if missing:
        raise ValueError(
            'MIAL uncertainty artifact is missing unlabeled image ids: %s'
            % sorted(missing, key=str)[:10]
        )

    ranked_ids = ranked_ids_by_score(scores)
    candidates: List[Dict[str, Any]] = []
    for rank, image_id in enumerate(ranked_ids, start=1):
        source = record_by_id[image_id]
        candidates.append({
            'image_id': image_id,
            'rank': rank,
            'score': scores[image_id],
            'components': dict(source.get('components', {})),
            'metadata': dict(source.get('metadata', {})),
        })
    return {
        'ranked_image_ids': ranked_ids,
        'candidate_records': candidates,
        'score_summary': _score_summary(scores.values()),
    }
