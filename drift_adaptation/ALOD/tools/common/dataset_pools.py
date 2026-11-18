"""Shared initial-pool writers for COCO-style annotation datasets."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from methods.common.coco_pool import build_coco_subset
from methods.common.io import read_json, write_json


SeededSplitRecord = Dict[str, Any]


def _integer_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError('%s must be an integer, found bool' % label)
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError('%s must be an integer: %r' % (label, value)) from exc


def _normalize_seeds(seeds: Sequence[int]) -> List[int]:
    if not seeds:
        raise ValueError('At least one seed is required')

    normalized: List[int] = []
    seen = set()
    for value in seeds:
        seed = _integer_id(value, 'seed')
        if seed < 0 or seed > np.iinfo(np.uint32).max:
            raise ValueError('seed must be between 0 and 2**32 - 1: %d' % seed)
        if seed not in seen:
            normalized.append(seed)
            seen.add(seed)
    return normalized


def _validate_split_request(
    oracle: Dict[str, Any],
    n_labeled: int,
    dataset_prefix: str,
) -> List[Dict[str, Any]]:
    if not isinstance(oracle, dict):
        raise ValueError('COCO-style oracle root must be an object')
    images = oracle.get('images')
    if not isinstance(images, list):
        raise ValueError('COCO-style oracle must contain an images list')
    annotations = oracle.get('annotations')
    if not isinstance(annotations, list):
        raise ValueError('COCO-style oracle must contain an annotations list')
    categories = oracle.get('categories')
    if not isinstance(categories, list):
        raise ValueError('COCO-style oracle must contain a categories list')
    if n_labeled <= 0:
        raise ValueError('n_labeled must be positive')
    if n_labeled >= len(images):
        raise ValueError('n_labeled must be smaller than image count')
    if not dataset_prefix:
        raise ValueError('dataset_prefix must not be empty')

    image_ids = []
    for index, image in enumerate(images):
        if not isinstance(image, dict) or 'id' not in image:
            raise ValueError('Oracle image %d must contain an integer id' % index)
        image_ids.append(_integer_id(image['id'], 'oracle image id'))
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('COCO-style oracle contains duplicate image ids')

    category_ids = []
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or 'id' not in category:
            raise ValueError('Oracle category %d must contain an integer id' % index)
        category_ids.append(_integer_id(category['id'], 'oracle category id'))
    if len(set(category_ids)) != len(category_ids):
        raise ValueError('COCO-style oracle contains duplicate category ids')

    image_id_set = set(image_ids)
    category_id_set = set(category_ids)
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise ValueError('Oracle annotation %d must be an object' % index)
        if 'image_id' not in annotation or 'category_id' not in annotation:
            raise ValueError(
                'Oracle annotation %d must contain image_id and category_id'
                % index
            )
        image_id = _integer_id(
            annotation['image_id'], 'oracle annotation image_id')
        category_id = _integer_id(
            annotation['category_id'], 'oracle annotation category_id')
        if image_id not in image_id_set:
            raise ValueError(
                'Oracle annotation %d references unknown image id %d'
                % (index, image_id)
            )
        if category_id not in category_id_set:
            raise ValueError(
                'Oracle annotation %d references unknown category id %d'
                % (index, category_id)
            )
    return images


def _seeded_split_paths(
    output_dir: Path,
    dataset_prefix: str,
    n_labeled: int,
    seed: int,
) -> SplitPaths:
    stem = '%s_%d' % (dataset_prefix, n_labeled)
    output_dir = Path(output_dir)
    return (
        output_dir / ('%s_labeled_seed_%d.json' % (stem, seed)),
        output_dir / ('%s_unlabeled_seed_%d.json' % (stem, seed)),
    )


def _sample_image_ids(
    images: Sequence[Dict[str, Any]],
    n_labeled: int,
    rng: np.random.RandomState,
) -> Tuple[List[int], List[int]]:
    permutation = rng.permutation(len(images))
    labeled_indices = set(int(index) for index in permutation[:n_labeled])
    labeled_ids = [int(images[index]['id']) for index in labeled_indices]
    unlabeled_ids = [
        int(image['id']) for index, image in enumerate(images)
        if index not in labeled_indices
    ]
    return labeled_ids, unlabeled_ids


def _validate_pool_payload(
    payload: Any,
    path: Path,
    oracle: Dict[str, Any],
    expected_ids: Sequence[int],
    require_annotations: bool,
) -> List[int]:
    if not isinstance(payload, dict):
        raise ValueError('COCO pool root must be an object: %s' % path)

    images = payload.get('images')
    categories = payload.get('categories')
    if not isinstance(images, list):
        raise ValueError('COCO pool must contain an images list: %s' % path)
    if not isinstance(categories, list):
        raise ValueError('COCO pool must contain a categories list: %s' % path)
    if categories != oracle['categories']:
        raise ValueError('COCO pool categories do not match oracle: %s' % path)

    expected_payload = build_coco_subset(
        oracle,
        expected_ids,
        include_annotations=require_annotations,
    )
    if images != expected_payload['images']:
        raise ValueError(
            'COCO pool image records do not match the oracle subset: %s' % path
        )

    image_ids = []
    for index, image in enumerate(images):
        if not isinstance(image, dict) or 'id' not in image:
            raise ValueError('COCO pool image %d is missing id: %s' % (index, path))
        image_ids.append(_integer_id(image['id'], 'pool image id'))
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('COCO pool contains duplicate image ids: %s' % path)

    expected_id_set = set(expected_ids)
    if len(image_ids) != len(expected_ids) or set(image_ids) != expected_id_set:
        raise ValueError(
            'COCO pool membership does not match its seed: %s' % path
        )

    annotations = payload.get('annotations')
    if require_annotations:
        if not isinstance(annotations, list):
            raise ValueError('Labeled COCO pool must contain annotations: %s' % path)
        category_ids = {
            _integer_id(category['id'], 'oracle category id')
            for category in oracle['categories']
        }
        image_id_set = set(image_ids)
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                raise ValueError(
                    'Labeled pool annotation %d must be an object: %s'
                    % (index, path)
                )
            if 'image_id' not in annotation or 'category_id' not in annotation:
                raise ValueError(
                    'Labeled pool annotation %d must contain image_id and '
                    'category_id: %s' % (index, path)
                )
            image_id = _integer_id(
                annotation['image_id'], 'labeled annotation image_id')
            category_id = _integer_id(
                annotation['category_id'], 'labeled annotation category_id')
            if image_id not in image_id_set:
                raise ValueError(
                    'Labeled pool annotation %d references unknown image id: %s'
                    % (index, path)
                )
            if category_id not in category_ids:
                raise ValueError(
                    'Labeled pool annotation %d references unknown category id: %s'
                    % (index, path)
                )
        if annotations != expected_payload['annotations']:
            raise ValueError(
                'Labeled pool annotations do not match the oracle subset: %s'
                % path
            )
    else:
        if annotations is not None and not isinstance(annotations, list):
            raise ValueError(
                'Unlabeled COCO pool annotations must be a list when present: %s'
                % path
            )
        if annotations:
            raise ValueError('Unlabeled COCO pool must not contain annotations: %s'
                             % path)
    return image_ids


def _validate_split_pair(
    labeled_payload: Any,
    unlabeled_payload: Any,
    labeled_path: Path,
    unlabeled_path: Path,
    oracle: Dict[str, Any],
    expected_labeled_ids: Sequence[int],
    expected_unlabeled_ids: Sequence[int],
    n_labeled: int,
) -> None:
    labeled_ids = _validate_pool_payload(
        labeled_payload,
        labeled_path,
        oracle,
        expected_labeled_ids,
        require_annotations=True,
    )
    unlabeled_ids = _validate_pool_payload(
        unlabeled_payload,
        unlabeled_path,
        oracle,
        expected_unlabeled_ids,
        require_annotations=False,
    )
    if len(labeled_ids) != n_labeled:
        raise ValueError(
            'Labeled pool must contain %d images, found %d: %s'
            % (n_labeled, len(labeled_ids), labeled_path)
        )

    labeled_id_set = set(labeled_ids)
    unlabeled_id_set = set(unlabeled_ids)
    overlap = labeled_id_set.intersection(unlabeled_id_set)
    if overlap:
        raise ValueError(
            'Labeled and unlabeled pools overlap: %s'
            % sorted(overlap)[:10]
        )
    oracle_ids = {
        _integer_id(image['id'], 'oracle image id')
        for image in oracle['images']
    }
    if labeled_id_set.union(unlabeled_id_set) != oracle_ids:
        raise ValueError(
            'Labeled and unlabeled pools do not cover the oracle: %s, %s'
            % (labeled_path, unlabeled_path)
        )


def ensure_seeded_initial_splits(
    oracle: Dict[str, Any],
    output_dir: Path,
    dataset_prefix: str,
    n_labeled: int,
    seeds: Sequence[int],
) -> List[SeededSplitRecord]:
    """Validate or create deterministic initial pools for actual seed values.

    Existing files are never overwritten. A valid partial pair is completed by
    creating only its missing counterpart; any invalid existing file raises.
    """

    images = _validate_split_request(oracle, n_labeled, dataset_prefix)
    normalized_seeds = _normalize_seeds(seeds)
    planned = []

    for seed in normalized_seeds:
        labeled_path, unlabeled_path = _seeded_split_paths(
            output_dir,
            dataset_prefix,
            n_labeled,
            seed,
        )
        rng = np.random.RandomState(seed)
        labeled_ids, unlabeled_ids = _sample_image_ids(images, n_labeled, rng)

        for path in (labeled_path, unlabeled_path):
            if path.exists() and not path.is_file():
                raise ValueError('Initial-pool path is not a file: %s' % path)

        labeled_payload = read_json(labeled_path) if labeled_path.is_file() else None
        unlabeled_payload = (
            read_json(unlabeled_path) if unlabeled_path.is_file() else None
        )
        if labeled_payload is not None:
            _validate_pool_payload(
                labeled_payload,
                labeled_path,
                oracle,
                labeled_ids,
                require_annotations=True,
            )
        if unlabeled_payload is not None:
            _validate_pool_payload(
                unlabeled_payload,
                unlabeled_path,
                oracle,
                unlabeled_ids,
                require_annotations=False,
            )
        planned.append((
            seed,
            labeled_path,
            unlabeled_path,
            labeled_ids,
            unlabeled_ids,
            labeled_payload,
            unlabeled_payload,
        ))

    records: List[SeededSplitRecord] = []
    for (
        seed,
        labeled_path,
        unlabeled_path,
        labeled_ids,
        unlabeled_ids,
        labeled_payload,
        unlabeled_payload,
    ) in planned:
        labeled_status = 'kept'
        unlabeled_status = 'kept'
        if labeled_payload is None:
            labeled_payload = build_coco_subset(
                oracle, labeled_ids, include_annotations=True)
            write_json(labeled_path, labeled_payload)
            labeled_status = 'created'
        if unlabeled_payload is None:
            unlabeled_payload = build_coco_subset(
                oracle, unlabeled_ids, include_annotations=False)
            write_json(unlabeled_path, unlabeled_payload)
            unlabeled_status = 'created'

        _validate_split_pair(
            labeled_payload,
            unlabeled_payload,
            labeled_path,
            unlabeled_path,
            oracle,
            labeled_ids,
            unlabeled_ids,
            n_labeled,
        )
        records.append({
            'seed': seed,
            'labeled': labeled_path,
            'unlabeled': unlabeled_path,
            'status': (
                'kept'
                if labeled_status == 'kept' and unlabeled_status == 'kept'
                else 'created'
            ),
            'labeled_status': labeled_status,
            'unlabeled_status': unlabeled_status,
        })
    return records
