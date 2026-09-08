"""Feature contract for versioned EPAL artifacts."""

from typing import Any, Mapping, Sequence

import numpy as np

FEATURE_SCHEMA = 'epal'
FEATURE_SCHEMA_VERSION = 1
CLASSIFICATION_FEATURE_KEYS = ('score', 'E_cls')
LOCALIZATION_FEATURE_KEYS = ('score', 'E_loc')
MISS_FEATURE_KEYS = ('R_amt', 'R_conf')


def _feature_vector(record: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    vector = np.asarray([float(record[key]) for key in keys], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError('EPAL feature vector contains non-finite values')
    return vector


def classification_feature_vector(detection: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(detection, CLASSIFICATION_FEATURE_KEYS)


def localization_feature_vector(detection: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(detection, LOCALIZATION_FEATURE_KEYS)


def miss_feature_vector(image_record: Mapping[str, Any]) -> np.ndarray:
    return _feature_vector(image_record['miss_features'], MISS_FEATURE_KEYS)
