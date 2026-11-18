"""Small COCO-style annotation pool helpers.

These utilities intentionally avoid MMDetection imports so they can be used by
lightweight samplers before detector dependencies load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from methods.common.io import read_json, write_json

JsonDict = Dict[str, Any]
PoolInput = Union[Path, JsonDict]


def read_coco_json(path: Path) -> JsonDict:
    return read_json(Path(path))


def write_coco_json(data: JsonDict, path: Path) -> None:
    write_json(path, data)


def load_coco_pool(pool: PoolInput) -> JsonDict:
    if isinstance(pool, Path):
        return read_coco_json(pool)
    return pool


def image_ids(coco_data: JsonDict) -> List[Any]:
    return [image['id'] for image in coco_data.get('images', [])]


def _validate_known_unique_ids(
    oracle_data: JsonDict,
    selected_image_ids: Iterable[Any],
    label: str,
) -> List[Any]:
    ids = list(selected_image_ids)
    oracle_id_set = set(image_ids(oracle_data))
    unknown = [image_id for image_id in ids if image_id not in oracle_id_set]
    if unknown:
        raise ValueError(
            '%s contains image ids not found in oracle: %s'
            % (label, sorted(set(unknown), key=str)[:10])
        )
    if len(set(ids)) != len(ids):
        duplicates = []
        seen = set()
        for image_id in ids:
            if image_id in seen:
                duplicates.append(image_id)
            seen.add(image_id)
        raise ValueError(
            '%s contains duplicate image ids: %s'
            % (label, sorted(set(duplicates), key=str)[:10])
        )
    return ids


def category_counts(records: Iterable[Dict[str, Any]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for record in records:
        category_id = record.get('category_id')
        if category_id is not None:
            counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def _metadata_from(oracle_data: JsonDict) -> JsonDict:
    subset = {'images': [], 'categories': oracle_data.get('categories', [])}
    for optional_key in ('info', 'licenses'):
        if optional_key in oracle_data:
            subset[optional_key] = oracle_data[optional_key]
    return subset


def build_coco_subset(
    oracle_data: JsonDict,
    selected_image_ids: Iterable[Any],
    include_annotations: bool,
) -> JsonDict:
    selected = set(selected_image_ids)
    subset = _metadata_from(oracle_data)
    subset['images'] = [
        image for image in oracle_data.get('images', [])
        if image.get('id') in selected
    ]
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle_data.get('annotations', [])
            if ann.get('image_id') in selected
        ]
    return subset


def build_coco_subset_ordered(
    oracle_data: JsonDict,
    selected_image_ids: Iterable[Any],
    include_annotations: bool,
) -> JsonDict:
    """Build a COCO subset preserving the supplied image-id order."""

    subset = _metadata_from(oracle_data)
    image_lookup = {
        image.get('id'): image for image in oracle_data.get('images', [])
    }
    ordered_ids = [
        image_id for image_id in selected_image_ids
        if image_id in image_lookup
    ]
    selected = set(ordered_ids)
    subset['images'] = [image_lookup[image_id] for image_id in ordered_ids]
    if include_annotations:
        subset['annotations'] = [
            ann for ann in oracle_data.get('annotations', [])
            if ann.get('image_id') in selected
        ]
    return subset


def write_candidate_pool_from_selection(
    oracle: PoolInput,
    last_labeled: PoolInput,
    candidate_image_ids: Iterable[Any],
    out_candidate_json: Path,
    out_remainder_json: Optional[Path] = None,
    candidate_include_annotations: bool = False,
) -> Tuple[List[Any], List[Any]]:
    """Write a candidate pool and optionally its unlabeled remainder."""

    oracle_data = load_coco_pool(oracle)
    last_labeled_data = load_coco_pool(last_labeled)
    oracle_ids = image_ids(oracle_data)
    oracle_id_set = set(oracle_ids)
    last_labeled_ids = set(image_ids(last_labeled_data))
    candidate_ids = _validate_known_unique_ids(
        oracle_data,
        candidate_image_ids,
        'candidate pool',
    )

    already_labeled = set(candidate_ids).intersection(last_labeled_ids)
    if already_labeled:
        raise ValueError(
            'Candidate pool contains already labeled image ids: %s'
            % sorted(already_labeled, key=str)[:10]
        )
    unknown_labeled = [
        image_id for image_id in last_labeled_ids
        if image_id not in oracle_id_set
    ]
    if unknown_labeled:
        raise ValueError(
            'Last labeled pool contains image ids not found in oracle: %s'
            % sorted(set(unknown_labeled), key=str)[:10]
        )

    candidate_set = set(candidate_ids)
    remainder_ids = [
        image_id for image_id in oracle_ids
        if image_id not in last_labeled_ids and image_id not in candidate_set
    ]
    overlap = candidate_set.intersection(remainder_ids)
    if overlap:
        raise ValueError(
            'Candidate and remainder pools overlap: %s'
            % sorted(overlap, key=str)[:10]
        )

    write_coco_json(
        build_coco_subset_ordered(
            oracle_data,
            candidate_ids,
            include_annotations=candidate_include_annotations,
        ),
        out_candidate_json,
    )
    if out_remainder_json is not None:
        write_coco_json(
            build_coco_subset_ordered(
                oracle_data,
                remainder_ids,
                include_annotations=False,
            ),
            out_remainder_json,
        )
    return candidate_ids, remainder_ids


def write_next_round_pool_split(
    oracle: PoolInput,
    last_labeled: PoolInput,
    selected_image_ids: Sequence[Any],
    out_labeled_json: Path,
    out_unlabeled_json: Path,
) -> Tuple[List[Any], List[Any]]:
    """Create next labeled/unlabeled JSONs from oracle annotations.

    Returns the final labeled and unlabeled image id lists in oracle image order.
    Selected ids already present in the labeled set are ignored.
    """

    oracle_data = load_coco_pool(oracle)
    last_labeled_data = load_coco_pool(last_labeled)

    oracle_ids = image_ids(oracle_data)
    oracle_id_set = set(oracle_ids)
    last_labeled_set = set(image_ids(last_labeled_data))
    unknown = [
        image_id for image_id in selected_image_ids
        if image_id not in oracle_id_set
    ]
    if unknown:
        raise ValueError(
            'Selected image ids not found in oracle: %s'
            % sorted(set(unknown), key=str)[:10]
        )

    new_selected = [
        image_id for image_id in selected_image_ids
        if image_id in oracle_id_set and image_id not in last_labeled_set
    ]
    labeled_set = last_labeled_set.union(new_selected)

    labeled_ids = [image_id for image_id in oracle_ids if image_id in labeled_set]
    unlabeled_ids = [image_id for image_id in oracle_ids if image_id not in labeled_set]

    write_coco_json(
        build_coco_subset(oracle_data, labeled_ids, include_annotations=True),
        out_labeled_json,
    )
    write_coco_json(
        build_coco_subset(oracle_data, unlabeled_ids, include_annotations=False),
        out_unlabeled_json,
    )
    return labeled_ids, unlabeled_ids
