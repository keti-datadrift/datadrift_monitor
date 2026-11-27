"""PAL image embedding utilities.

The PAL paper uses image-level embeddings for RCSP. This module provides a
lightweight backend interface plus a deterministic detection-record fallback so
the PAL pipeline can run before an optional ViT backend is added.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from methods.common.image_identity import normalize_image_id
from methods.common.io import read_json, write_json
from methods.common.selection import image_id_sort_key
from methods.common.vectors import l2_normalize
from methods.pal.inference import detection_confidence, detection_pre_nms_count


EmbeddingMap = Dict[Any, np.ndarray]


def _sort_image_id(image_id: Any) -> tuple:
    return image_id_sort_key(normalize_image_id(image_id))


def _json_image_id(image_id: Any) -> str:
    return json.dumps(normalize_image_id(image_id), sort_keys=True)


def _loads_image_id(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return json.loads(str(value))


def _as_float_vector(values: Sequence[Any], dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return vector
    raw = np.asarray(list(values), dtype=np.float32).reshape(-1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    width = min(dim, raw.shape[0])
    vector[:width] = raw[:width]
    return vector


class ImageEmbeddingBackend:
    """Base interface for image embedding backends."""

    name = 'base'

    def embed_records(
        self,
        detections: Iterable[Mapping[str, Any]],
        image_ids: Optional[Iterable[Any]] = None,
    ) -> EmbeddingMap:
        raise NotImplementedError


class DetectionEmbeddingBackend(ImageEmbeddingBackend):
    """Deterministic fallback embedding backend based on detector records.

    Each image embedding concatenates a score-weighted class-score mean, a
    per-class max score, and simple detection statistics. If a record has no
    ``class_scores`` field, its ``category_id`` is used as a deterministic
    one-hot fallback.
    """

    name = 'detection'

    def __init__(
        self,
        num_classes: Optional[int] = None,
        include_detection_stats: bool = True,
        normalize: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.include_detection_stats = include_detection_stats
        self.normalize = normalize

    def embed_records(
        self,
        detections: Iterable[Mapping[str, Any]],
        image_ids: Optional[Iterable[Any]] = None,
    ) -> EmbeddingMap:
        records = [dict(record) for record in detections]
        ordered_image_ids = self._ordered_image_ids(records, image_ids)
        class_dim, category_to_index = self._class_space(records)
        embedding_dim = 2 * class_dim
        if self.include_detection_stats:
            embedding_dim += 5

        accumulators = {
            image_id: self._new_accumulator(class_dim)
            for image_id in ordered_image_ids
        }
        for record in records:
            image_id = normalize_image_id(record.get('image_id'))
            if image_id not in accumulators:
                continue
            class_vector = self._record_class_vector(
                record,
                class_dim=class_dim,
                category_to_index=category_to_index,
            )
            score = max(detection_confidence(record), 0.0)
            pre_nms_count = max(detection_pre_nms_count(record), 0.0)
            accumulator = accumulators[image_id]
            accumulator['weighted_class_sum'] += class_vector * score
            accumulator['class_max'] = np.maximum(
                accumulator['class_max'], class_vector)
            accumulator['count'] += 1.0
            accumulator['score_sum'] += score
            accumulator['score_max'] = max(accumulator['score_max'], score)
            accumulator['pre_nms_sum'] += pre_nms_count
            accumulator['pre_nms_max'] = max(
                accumulator['pre_nms_max'], pre_nms_count)

        embeddings = {}
        for image_id in ordered_image_ids:
            vector = self._finalize_accumulator(
                accumulators[image_id],
                class_dim=class_dim,
                embedding_dim=embedding_dim,
            )
            embeddings[image_id] = l2_normalize(vector) if self.normalize else vector
        return embeddings

    def _ordered_image_ids(
        self,
        records: Sequence[Mapping[str, Any]],
        image_ids: Optional[Iterable[Any]],
    ) -> List[Any]:
        if image_ids is not None:
            seen = set()
            ordered = []
            for image_id in image_ids:
                normalized = normalize_image_id(image_id)
                if normalized not in seen:
                    ordered.append(normalized)
                    seen.add(normalized)
            return ordered

        ids = {
            normalize_image_id(record.get('image_id'))
            for record in records
            if record.get('image_id') is not None
        }
        return sorted(ids, key=_sort_image_id)

    def _class_space(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> tuple:
        max_class_scores = 0
        categories = set()
        for record in records:
            class_scores = record.get('class_scores')
            if class_scores is not None:
                max_class_scores = max(max_class_scores, len(class_scores))
            if record.get('category_id') is not None:
                categories.add(normalize_image_id(record['category_id']))

        if self.num_classes is not None:
            class_dim = int(self.num_classes)
        elif max_class_scores > 0:
            class_dim = max_class_scores
        else:
            class_dim = len(categories)

        category_to_index = self._category_to_index(categories, class_dim)
        return class_dim, category_to_index

    def _category_to_index(self, categories: Iterable[Any], class_dim: int) -> Dict[Any, int]:
        categories = sorted(categories, key=_sort_image_id)
        if class_dim <= 0:
            return {}
        if all(isinstance(category_id, int) and 0 <= category_id < class_dim
               for category_id in categories):
            return {category_id: category_id for category_id in categories}
        if all(isinstance(category_id, int) and 1 <= category_id <= class_dim
               for category_id in categories):
            return {category_id: category_id - 1 for category_id in categories}
        return {
            category_id: index
            for index, category_id in enumerate(categories)
            if index < class_dim
        }

    def _record_class_vector(
        self,
        record: Mapping[str, Any],
        class_dim: int,
        category_to_index: Mapping[Any, int],
    ) -> np.ndarray:
        if record.get('class_scores') is not None:
            return _as_float_vector(record['class_scores'], class_dim)

        vector = np.zeros(class_dim, dtype=np.float32)
        category_id = normalize_image_id(record.get('category_id'))
        index = category_to_index.get(category_id)
        if index is not None and 0 <= index < class_dim:
            vector[index] = 1.0
        return vector

    def _new_accumulator(self, class_dim: int) -> Dict[str, Any]:
        return {
            'weighted_class_sum': np.zeros(class_dim, dtype=np.float32),
            'class_max': np.zeros(class_dim, dtype=np.float32),
            'count': 0.0,
            'score_sum': 0.0,
            'score_max': 0.0,
            'pre_nms_sum': 0.0,
            'pre_nms_max': 0.0,
        }

    def _finalize_accumulator(
        self,
        accumulator: Mapping[str, Any],
        class_dim: int,
        embedding_dim: int,
    ) -> np.ndarray:
        count = float(accumulator['count'])
        if count <= 0.0:
            return np.zeros(embedding_dim, dtype=np.float32)

        score_sum = float(accumulator['score_sum'])
        class_normalizer = score_sum if score_sum > 1e-12 else count
        class_mean = accumulator['weighted_class_sum'] / class_normalizer
        parts = [class_mean.astype(np.float32), accumulator['class_max']]
        if self.include_detection_stats:
            mean_score = score_sum / count
            mean_pre_nms = float(accumulator['pre_nms_sum']) / count
            stats = np.asarray(
                [
                    math.log1p(count),
                    mean_score,
                    float(accumulator['score_max']),
                    math.log1p(mean_pre_nms),
                    math.log1p(float(accumulator['pre_nms_max'])),
                ],
                dtype=np.float32,
            )
            parts.append(stats)
        if class_dim == 0 and not self.include_detection_stats:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32, copy=False)


class VitEmbeddingBackend(ImageEmbeddingBackend):
    """Placeholder for a future ViT image backend."""

    name = 'vit'

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def embed_records(
        self,
        detections: Iterable[Mapping[str, Any]],
        image_ids: Optional[Iterable[Any]] = None,
    ) -> EmbeddingMap:
        raise NotImplementedError(
            'Direct ViT embedding extraction is handled by the ALOD runner '
            'automatic preparation path. Use backend="detection" for '
            'deterministic detector-record fallback embeddings, or set '
            'pal_embedding_source="external" and let the runner prepare the cache.')


def build_embedding_backend(name: str = 'detection', **kwargs: Any) -> ImageEmbeddingBackend:
    normalized = name.lower().replace('_', '-')
    if normalized in ('detection', 'detector', 'fallback'):
        return DetectionEmbeddingBackend(**kwargs)
    if normalized in ('vit', 'vision-transformer'):
        return VitEmbeddingBackend(**kwargs)
    raise ValueError('Unsupported PAL embedding backend: %s' % name)


def build_image_embeddings(
    detections: Iterable[Mapping[str, Any]],
    image_ids: Optional[Iterable[Any]] = None,
    backend: str = 'detection',
    **backend_kwargs: Any,
) -> EmbeddingMap:
    embedding_backend = build_embedding_backend(backend, **backend_kwargs)
    return embedding_backend.embed_records(detections, image_ids=image_ids)


def _validate_embedding_map(embeddings: Mapping[Any, np.ndarray]) -> EmbeddingMap:
    normalized = {}
    expected_shape = None
    for image_id, embedding in embeddings.items():
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if expected_shape is None:
            expected_shape = vector.shape
        elif vector.shape != expected_shape:
            raise ValueError('All PAL image embeddings must have the same shape')
        normalized[normalize_image_id(image_id)] = vector
    return normalized


def write_embeddings_json(embeddings: Mapping[Any, np.ndarray], path: Path) -> None:
    normalized = _validate_embedding_map(embeddings)
    payload = {
        'format': 'pal_image_embeddings_v1',
        'embeddings': [
            {
                'image_id': image_id,
                'embedding': normalized[image_id].tolist(),
            }
            for image_id in sorted(normalized, key=_sort_image_id)
        ],
    }
    write_json(Path(path), payload)


def read_embeddings_json(path: Path) -> EmbeddingMap:
    path = Path(path)
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get('embeddings'), list):
        records = payload['embeddings']
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError('Unsupported PAL embedding JSON schema: %s' % path)

    embeddings = {}
    for record in records:
        image_id = normalize_image_id(record['image_id'])
        embeddings[image_id] = np.asarray(record['embedding'], dtype=np.float32)
    return embeddings


def write_embeddings_npy(embeddings: Mapping[Any, np.ndarray], path: Path) -> None:
    normalized = _validate_embedding_map(embeddings)
    ids = [_json_image_id(image_id) for image_id in sorted(normalized, key=_sort_image_id)]
    max_id_length = max([1] + [len(image_id) for image_id in ids])
    embedding_dim = len(next(iter(normalized.values()))) if normalized else 0
    dtype = np.dtype([
        ('image_id', 'U%d' % max_id_length),
        ('embedding', np.float32, (embedding_dim, )),
    ])
    rows = np.empty(len(ids), dtype=dtype)
    for row_index, image_id_json in enumerate(ids):
        image_id = _loads_image_id(image_id_json)
        rows[row_index]['image_id'] = image_id_json
        rows[row_index]['embedding'] = normalized[image_id]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), rows, allow_pickle=False)


def read_embeddings_npy(path: Path) -> EmbeddingMap:
    rows = np.load(str(path), allow_pickle=False)
    if rows.dtype.names != ('image_id', 'embedding'):
        raise ValueError('Unsupported PAL embedding NPY schema: %s' % path)

    embeddings = {}
    for row in rows:
        image_id = _loads_image_id(row['image_id'])
        embeddings[image_id] = np.asarray(row['embedding'], dtype=np.float32)
    return embeddings


def write_embedding_cache(embeddings: Mapping[Any, np.ndarray], path: Path) -> None:
    path = Path(path)
    if path.suffix.lower() == '.json':
        write_embeddings_json(embeddings, path)
        return
    if path.suffix.lower() == '.npy':
        write_embeddings_npy(embeddings, path)
        return
    raise ValueError('PAL embedding cache must use .json or .npy: %s' % path)


def read_embedding_cache(path: Path) -> EmbeddingMap:
    path = Path(path)
    if path.suffix.lower() == '.json':
        return read_embeddings_json(path)
    if path.suffix.lower() == '.npy':
        return read_embeddings_npy(path)
    raise ValueError('PAL embedding cache must use .json or .npy: %s' % path)


def stack_embeddings(
    embeddings: Mapping[Any, np.ndarray],
    image_ids: Optional[Iterable[Any]] = None,
) -> tuple:
    ordered_ids = (
        [normalize_image_id(image_id) for image_id in image_ids]
        if image_ids is not None
        else sorted(embeddings, key=_sort_image_id)
    )
    vectors = [np.asarray(embeddings[image_id], dtype=np.float32) for image_id in ordered_ids]
    if not vectors:
        return ordered_ids, np.zeros((0, 0), dtype=np.float32)
    return ordered_ids, np.stack(vectors, axis=0).astype(np.float32, copy=False)
