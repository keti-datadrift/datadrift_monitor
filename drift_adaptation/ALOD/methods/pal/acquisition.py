"""PAL acquisition orchestration for LIUS-only and full PAL."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from methods.common.coco_pool import category_counts, image_ids, read_coco_json
from methods.common.matching import match_detections_to_ground_truth
from methods.common.selection import fill_to_budget, ranked_ids_by_score, unique_in_order
from methods.pal.embeddings import build_image_embeddings, read_embedding_cache
from methods.pal.guide import (
    allocate_class_budgets as allocate_guide_class_budgets,
    build_class_candidates,
    compute_class_weights as compute_guide_class_weights,
    group_detections_by_image,
    score_guide_candidates,
    select_deduplicated_candidates,
)
from methods.pal.inference import (
    detection_confidence,
    detection_pre_nms_count,
    filter_pool_detections,
    load_detection_records,
)
from methods.pal.lius import score_unlabeled_detections, train_classwise_models


__all__ = [
    'allocate_class_budgets',
    'compute_class_weights',
    'sample_lius_only_from_files',
    'sample_pal_from_files',
    'select_full_pal_images',
    'select_lius_images',
]


compute_class_weights = compute_guide_class_weights


def _candidate_image_scores_by_class(
    scored_detections: Iterable[Dict[str, Any]],
) -> Dict[Any, Dict[Any, float]]:
    scores: Dict[Any, Dict[Any, float]] = defaultdict(dict)
    for det in scored_detections:
        category_id = det.get('category_id')
        image_id = det.get('image_id')
        if category_id is None or image_id is None:
            continue
        score = float(det.get('lius_score', 0.0))
        current = scores[category_id].get(image_id)
        if current is None or score > current:
            scores[category_id][image_id] = score
    return scores


def _candidate_records_by_class(
    scored_detections: Iterable[Dict[str, Any]],
) -> Dict[Any, Dict[Any, Dict[str, Any]]]:
    candidates: Dict[Any, Dict[Any, Dict[str, Any]]] = defaultdict(dict)
    for det in scored_detections:
        category_id = det.get('category_id')
        image_id = det.get('image_id')
        if category_id is None or image_id is None:
            continue
        record = {
            'category_id': category_id,
            'image_id': image_id,
            'lius_score': float(det.get('lius_score', 0.0)),
            'tp_probability': float(det.get('tp_probability', 0.5)),
            'score': detection_confidence(det),
            'pre_nms_count': detection_pre_nms_count(det),
        }
        current = candidates[category_id].get(image_id)
        if current is None or (
                record['lius_score'], record['score']) > (
                    current['lius_score'], current['score']):
            candidates[category_id][image_id] = record
    return candidates


def allocate_class_budgets(
    budget: int,
    labeled_counts: Dict[Any, int],
    unlabeled_counts: Dict[Any, int],
    class_capacities: Dict[Any, int],
) -> Dict[Any, int]:
    """Compatibility wrapper for PAL class-budget allocation."""

    class_weights = compute_class_weights(
        labeled_counts,
        unlabeled_counts,
        class_capacities,
    )
    return allocate_guide_class_budgets(class_weights, class_capacities, budget)


def _load_candidate_embeddings(
    embedding_source: str,
    embedding_path: Optional[Path],
    detections: Sequence[Dict[str, Any]],
    candidate_image_ids: Sequence[Any],
) -> Dict[Any, Any]:
    source = embedding_source.lower().replace('_', '-')
    if source in ('external', 'cache'):
        if embedding_path is None:
            raise ValueError('PAL full mode requires pal_embedding_path when embedding_source=%s'
                             % embedding_source)
        embeddings = read_embedding_cache(embedding_path)
    elif source in ('detection', 'detector', 'class-scores', 'class-score'):
        embeddings = build_image_embeddings(
            detections,
            image_ids=candidate_image_ids,
            backend='detection',
        )
    elif source in ('vit', 'vision-transformer'):
        embeddings = build_image_embeddings(
            detections,
            image_ids=candidate_image_ids,
            backend='vit',
        )
    else:
        raise ValueError('Unsupported PAL embedding source: %s' % embedding_source)

    missing = [image_id for image_id in candidate_image_ids if image_id not in embeddings]
    if missing:
        raise ValueError(
            'Missing PAL RCSP embeddings for %d candidate image(s), first ids: %s'
            % (len(missing), missing[:10]))
    return embeddings


def _append_selected(
    selected: List[Any],
    selected_details: List[Dict[str, Any]],
    selected_set: set,
    record: Dict[str, Any],
    stage: str,
) -> None:
    image_id = record['image_id']
    selected.append(image_id)
    selected_set.add(image_id)
    detail = dict(record)
    detail['selection_stage'] = stage
    selected_details.append(detail)


def _pal_candidate_record(record: Dict[str, Any], rank: int) -> Dict[str, Any]:
    components = {}
    for key in (
            'lius_score',
            'tp_probability',
            'guide_cwie',
            'guide_rcdi',
            'guide_rcsp',
            'pal_score'):
        if key in record:
            components[key] = float(record[key])

    metadata = {}
    for key in ('candidate_class_id', 'candidate_rank', 'selection_stage'):
        if key in record:
            metadata[key] = record[key]

    score = record.get('pal_score', record.get('lius_score', record.get('score')))
    return {
        'image_id': record['image_id'],
        'category_id': record.get('category_id'),
        'rank': rank,
        'score': float(score) if score is not None else None,
        'source': 'guide',
        'components': components,
        'metadata': metadata,
    }


def select_lius_images(
    labeled_pool: Dict[str, Any],
    unlabeled_pool: Dict[str, Any],
    labeled_detections: Sequence[Dict[str, Any]],
    unlabeled_detections: Sequence[Dict[str, Any]],
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
    candidate_multiplier: int = 1,
) -> Dict[str, Any]:
    candidate_multiplier = int(candidate_multiplier)
    if candidate_multiplier < 1:
        raise ValueError('PAL LIUS candidate_multiplier must be positive')

    labeled_ids = set(image_ids(labeled_pool))
    unlabeled_ids = image_ids(unlabeled_pool)

    labeled_dets = filter_pool_detections(labeled_detections, labeled_ids)
    unlabeled_dets = filter_pool_detections(unlabeled_detections, unlabeled_ids)

    matched = match_detections_to_ground_truth(
        labeled_dets,
        labeled_pool,
        iou_threshold=iou_threshold,
    )
    models = train_classwise_models(matched)
    scored = score_unlabeled_detections(unlabeled_dets, models)

    class_image_scores = _candidate_image_scores_by_class(scored)
    labeled_counts = category_counts(labeled_pool.get('annotations', []))
    unlabeled_counts = category_counts(unlabeled_dets)
    class_capacities = {
        category_id: len(image_scores)
        for category_id, image_scores in class_image_scores.items()
    }
    class_budgets = allocate_class_budgets(
        budget,
        labeled_counts,
        unlabeled_counts,
        class_capacities,
    )

    candidate_scores: Dict[Any, float] = {}
    for category_id, class_budget in class_budgets.items():
        if class_budget <= 0:
            continue
        class_candidates = ranked_ids_by_score(class_image_scores.get(category_id, {}))
        for image_id in class_candidates[:candidate_multiplier * class_budget]:
            score = class_image_scores[category_id][image_id]
            if image_id not in candidate_scores or score > candidate_scores[image_id]:
                candidate_scores[image_id] = score

    selected = ranked_ids_by_score(candidate_scores)[:budget]
    global_scores: Dict[Any, float] = {}
    if len(selected) < budget:
        global_scores = defaultdict(float)
        for det in scored:
            image_id = det.get('image_id')
            if image_id is not None:
                global_scores[image_id] = max(global_scores[image_id], float(det.get('lius_score', 0.0)))
        for image_id in ranked_ids_by_score(global_scores):
            if image_id not in selected:
                selected.append(image_id)
                candidate_scores.setdefault(image_id, float(global_scores[image_id]))
            if len(selected) == budget:
                break

    if len(selected) < budget:
        before_refill = len(selected)
        selected = fill_to_budget(selected, unlabeled_ids, budget, seed=seed)
        for image_id in selected[before_refill:]:
            candidate_scores.setdefault(image_id, 0.0)

    candidate_records = []
    for rank, image_id in enumerate(ranked_ids_by_score(candidate_scores), start=1):
        score = float(candidate_scores[image_id])
        candidate_records.append({
            'image_id': image_id,
            'rank': rank,
            'score': score,
            'source': 'lius',
            'components': {'lius_score': score},
            'metadata': {},
        })

    return {
        'selected_image_ids': selected[:budget],
        'mode': 'lius',
        'candidate_multiplier': candidate_multiplier,
        'class_budgets': class_budgets,
        'matched_detection_count': len(matched),
        'scored_detection_count': len(scored),
        'candidate_records': candidate_records,
    }


def select_full_pal_images(
    labeled_pool: Dict[str, Any],
    unlabeled_pool: Dict[str, Any],
    labeled_detections: Sequence[Dict[str, Any]],
    unlabeled_detections: Sequence[Dict[str, Any]],
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
    alpha: float = 0.9,
    beta: float = 0.04,
    gamma: float = 0.02,
    embedding_source: str = 'external',
    embedding_path: Optional[Path] = None,
) -> Dict[str, Any]:
    labeled_ids = set(image_ids(labeled_pool))
    unlabeled_ids = image_ids(unlabeled_pool)

    labeled_dets = filter_pool_detections(labeled_detections, labeled_ids)
    unlabeled_dets = filter_pool_detections(unlabeled_detections, unlabeled_ids)

    matched = match_detections_to_ground_truth(
        labeled_dets,
        labeled_pool,
        iou_threshold=iou_threshold,
    )
    models = train_classwise_models(matched)
    scored = score_unlabeled_detections(unlabeled_dets, models)

    labeled_counts = category_counts(labeled_pool.get('annotations', []))
    unlabeled_counts = category_counts(unlabeled_dets)
    class_candidates = _candidate_records_by_class(scored)
    class_capacities = {
        category_id: len(image_records)
        for category_id, image_records in class_candidates.items()
    }
    all_categories = set(labeled_counts) | set(unlabeled_counts) | set(class_capacities)
    class_weights = compute_guide_class_weights(labeled_counts, unlabeled_counts, all_categories)
    class_budgets = allocate_guide_class_budgets(class_weights, class_capacities, budget)

    candidates_by_class = build_class_candidates(
        scored,
        class_budgets,
        candidate_multiplier=2,
        score_key='lius_score',
    )
    candidate_image_ids = [
        candidate['image_id']
        for candidates in candidates_by_class.values()
        for candidate in candidates
    ]
    unique_candidate_image_ids = unique_in_order(candidate_image_ids)
    embeddings = _load_candidate_embeddings(
        embedding_source,
        embedding_path,
        scored,
        unique_candidate_image_ids,
    ) if unique_candidate_image_ids else {}

    image_detections = group_detections_by_image(scored)
    scored_by_class = score_guide_candidates(
        candidates_by_class,
        image_detections,
        class_weights,
        embeddings,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lius_key='lius_score',
    )
    all_candidates = [
        candidate
        for candidates in scored_by_class.values()
        for candidate in candidates
    ]
    selection = select_deduplicated_candidates(
        scored_by_class,
        class_budgets,
        budget,
        class_weights=class_weights,
        score_key='pal_score',
    )
    selected = list(selection['selected_image_ids'])
    selected_details = list(selection['selected_candidates'])
    selected_set = set(selected)

    if len(selected) < budget:
        global_scores: Dict[Any, float] = defaultdict(float)
        for det in scored:
            image_id = det.get('image_id')
            if image_id is not None:
                global_scores[image_id] = max(global_scores[image_id], float(det.get('lius_score', 0.0)))
        for image_id in ranked_ids_by_score(global_scores):
            if image_id in selected_set:
                continue
            record = {
                'image_id': image_id,
                'category_id': None,
                'lius_score': float(global_scores[image_id]),
                'pal_score': float(global_scores[image_id]),
            }
            _append_selected(selected, selected_details, selected_set, record, 'global_lius_refill')
            if len(selected) == budget:
                break

    if len(selected) < budget:
        refilled = fill_to_budget(selected, unlabeled_ids, budget, seed=seed)
        for image_id in refilled[len(selected):]:
            record = {
                'image_id': image_id,
                'category_id': None,
                'lius_score': 0.0,
                'pal_score': 0.0,
            }
            _append_selected(selected, selected_details, selected_set, record, 'random_refill')

    candidate_records = [
        _pal_candidate_record(candidate, rank)
        for rank, candidate in enumerate(all_candidates, start=1)
    ]
    candidate_image_set = {
        candidate.get('image_id')
        for candidate in candidate_records
        if candidate.get('image_id') is not None
    }
    for detail in selected_details:
        image_id = detail.get('image_id')
        if image_id is not None and image_id not in candidate_image_set:
            candidate_records.append(
                _pal_candidate_record(detail, len(candidate_records) + 1)
            )
            candidate_image_set.add(image_id)

    return {
        'selected_image_ids': selected[:budget],
        'mode': 'full',
        'class_budgets': class_budgets,
        'class_weights': class_weights,
        'matched_detection_count': len(matched),
        'scored_detection_count': len(scored),
        'candidate_image_count': len(unique_candidate_image_ids),
        'embedding_source': embedding_source,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'class_selected_counts': selection['class_selected_counts'],
        'unfilled_guide_budget': selection['unfilled_budget'],
        'candidate_records': candidate_records,
    }


def sample_lius_only_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_detections_json: Path,
    unlabeled_detections_json: Path,
    budget: int,
    iou_threshold: float = 0.5,
    seed: int = 0,
    candidate_multiplier: int = 1,
) -> Dict[str, Any]:
    labeled_pool = read_coco_json(labeled_pool_json)
    unlabeled_pool = read_coco_json(unlabeled_pool_json)
    labeled_detections = load_detection_records(labeled_detections_json)
    unlabeled_detections = load_detection_records(unlabeled_detections_json)
    return select_lius_images(
        labeled_pool=labeled_pool,
        unlabeled_pool=unlabeled_pool,
        labeled_detections=labeled_detections,
        unlabeled_detections=unlabeled_detections,
        budget=budget,
        iou_threshold=iou_threshold,
        seed=seed,
        candidate_multiplier=candidate_multiplier,
    )


def sample_pal_from_files(
    labeled_pool_json: Path,
    unlabeled_pool_json: Path,
    labeled_detections_json: Path,
    unlabeled_detections_json: Path,
    budget: int,
    mode: str = 'lius',
    iou_threshold: float = 0.5,
    seed: int = 0,
    candidate_multiplier: int = 1,
    alpha: float = 0.9,
    beta: float = 0.04,
    gamma: float = 0.02,
    embedding_source: str = 'external',
    embedding_path: Optional[Path] = None,
) -> Dict[str, Any]:
    labeled_pool = read_coco_json(labeled_pool_json)
    unlabeled_pool = read_coco_json(unlabeled_pool_json)
    labeled_detections = load_detection_records(labeled_detections_json)
    unlabeled_detections = load_detection_records(unlabeled_detections_json)
    normalized_mode = mode.lower()
    if normalized_mode in ('lius', 'lius_only', 'lius-only'):
        return select_lius_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=budget,
            iou_threshold=iou_threshold,
            seed=seed,
            candidate_multiplier=candidate_multiplier,
        )
    if normalized_mode in ('full', 'guide'):
        return select_full_pal_images(
            labeled_pool=labeled_pool,
            unlabeled_pool=unlabeled_pool,
            labeled_detections=labeled_detections,
            unlabeled_detections=unlabeled_detections,
            budget=budget,
            iou_threshold=iou_threshold,
            seed=seed,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            embedding_source=embedding_source,
            embedding_path=embedding_path,
        )
    raise ValueError('Unsupported PAL mode: %s' % mode)
