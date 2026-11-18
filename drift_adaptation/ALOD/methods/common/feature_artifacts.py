"""Shared image/detection feature artifact loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from methods.common.image_identity import normalize_image_id, normalize_image_ids


@dataclass(frozen=True)
class FeatureArtifact:
    path: Path
    image_ids: List[Any]
    image_features: Optional[np.ndarray] = None
    det_labels: Optional[np.ndarray] = None
    det_scores: Optional[np.ndarray] = None
    det_features: Optional[np.ndarray] = None
    det_valid: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def image_count(self) -> int:
        return len(self.image_ids)


def _metadata_from_npz(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    text = str(np.asarray(value).item())
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError('Feature artifact metadata_json must decode to an object')
    return payload


def _require_2d_float(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError('%s must be a 2D array in %s' % (name, path))
    if not np.all(np.isfinite(value)):
        raise ValueError('%s contains non-finite values in %s' % (name, path))
    return value


def _validate_unique_image_ids(image_ids: List[Any], path: Path) -> None:
    seen = set()
    duplicates = []
    for image_id in image_ids:
        if image_id in seen:
            duplicates.append(image_id)
        seen.add(image_id)
    if duplicates:
        raise ValueError(
            'Feature artifact contains duplicate image ids in %s: %s'
            % (path, sorted(set(duplicates), key=str)[:10]))


def load_feature_artifact(
    path: Any,
    require_detection_features: bool = False,
    require_image_features: bool = True,
) -> FeatureArtifact:
    """Load a generic ALOD feature artifact from ``np.savez`` output."""

    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as data:
        required = ['image_ids']
        if require_image_features:
            required.append('image_features')
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError('Feature artifact %s is missing keys: %s' % (artifact_path, ', '.join(missing)))

        image_ids = normalize_image_ids(np.asarray(data['image_ids']).reshape(-1).tolist())
        image_features = None
        if 'image_features' in data:
            image_features = _require_2d_float(data['image_features'], 'image_features', artifact_path)
            if image_features.shape[0] != len(image_ids):
                raise ValueError(
                    'Feature artifact %s has %d image ids but %d image feature rows'
                    % (artifact_path, len(image_ids), image_features.shape[0]))
        _validate_unique_image_ids(image_ids, artifact_path)

        det_labels = det_scores = det_features = det_valid = None
        detection_keys = ('det_labels', 'det_scores', 'det_features', 'det_valid')
        has_detection = all(key in data for key in detection_keys)
        if require_detection_features and not has_detection:
            missing = [key for key in detection_keys if key not in data]
            raise KeyError(
                'Feature artifact %s is missing detection feature keys: %s'
                % (artifact_path, ', '.join(missing)))
        if has_detection:
            det_labels = np.asarray(data['det_labels'], dtype=np.int64)
            det_scores = np.asarray(data['det_scores'], dtype=np.float32)
            det_features = np.asarray(data['det_features'], dtype=np.float32)
            det_valid = np.asarray(data['det_valid'], dtype=np.bool_)
            if det_labels.ndim != 2 or det_scores.ndim != 2 or det_valid.ndim != 2:
                raise ValueError('Detection label/score/valid arrays must be 2D in %s' % artifact_path)
            if det_features.ndim != 3:
                raise ValueError('det_features must be a 3D array in %s' % artifact_path)
            if not (
                det_labels.shape == det_scores.shape == det_valid.shape
                and det_features.shape[:2] == det_labels.shape
                and det_labels.shape[0] == len(image_ids)
            ):
                raise ValueError('Detection feature array shapes do not align in %s' % artifact_path)
            if not np.all(np.isfinite(det_scores)) or not np.all(np.isfinite(det_features)):
                raise ValueError('Detection feature arrays contain non-finite values in %s' % artifact_path)

        metadata = _metadata_from_npz(data['metadata_json']) if 'metadata_json' in data else {}

    return FeatureArtifact(
        path=artifact_path,
        image_ids=image_ids,
        image_features=image_features,
        det_labels=det_labels,
        det_scores=det_scores,
        det_features=det_features,
        det_valid=det_valid,
        metadata=metadata,
    )


def filter_feature_artifact(
    artifact: FeatureArtifact,
    image_ids: Iterable[Any],
    artifact_name: str = 'feature artifact',
    require_all: bool = True,
) -> FeatureArtifact:
    """Return ``artifact`` rows ordered by ``image_ids``."""

    requested = [normalize_image_id(image_id) for image_id in image_ids]
    index = {image_id: offset for offset, image_id in enumerate(artifact.image_ids)}
    missing = [image_id for image_id in requested if image_id not in index]
    if missing and require_all:
        raise ValueError(
            '%s is missing %d requested image id(s), first ids: %s'
            % (artifact_name, len(missing), missing[:10]))
    kept_ids = [image_id for image_id in requested if image_id in index]
    rows = np.asarray([index[image_id] for image_id in kept_ids], dtype=np.int64)

    def take_optional(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if value is None:
            return None
        return value[rows]

    return FeatureArtifact(
        path=artifact.path,
        image_ids=kept_ids,
        image_features=take_optional(artifact.image_features),
        det_labels=take_optional(artifact.det_labels),
        det_scores=take_optional(artifact.det_scores),
        det_features=take_optional(artifact.det_features),
        det_valid=take_optional(artifact.det_valid),
        metadata=dict(artifact.metadata or {}),
    )
