"""EUA entropy scoring for active EPAL components."""

import math
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from methods.common.selection import image_id_sort_key
from methods.epal.components import ERROR_KEYS
from methods.epal.features import (
    classification_feature_vector, localization_feature_vector, miss_feature_vector,
)
from methods.epal.predictors import EPALPredictors


def bernoulli_entropy(probability: float) -> float:
    p = float(probability)
    if not np.isfinite(p):
        raise ValueError('Bernoulli entropy probability must be finite')
    p = min(max(p, 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))


def poisson_entropy(mean: float) -> float:
    value = float(mean)
    if not np.isfinite(value):
        raise ValueError('Poisson entropy mean must be finite')
    if value <= 0.0:
        return 0.0
    try:
        from scipy.stats import poisson
    except ImportError as exc:
        raise ImportError(
            'scipy is required for EPAL EUA Poisson entropy. '
            'Install it with `pip install -r requirements.txt`.'
        ) from exc
    entropy = float(poisson.entropy(value))
    if not np.isfinite(entropy):
        raise ValueError('Poisson entropy produced a non-finite value')
    return entropy


def predict_eua_profile(record: Mapping[str, Any], predictors: EPALPredictors) -> Dict[str, float]:
    profile = {}
    for name, model in predictors.models.items():
        if name == 'mocp':
            value = poisson_entropy(model.predict_count(miss_feature_vector(record)))
        else:
            feature_fn = classification_feature_vector if name == 'cecp' else localization_feature_vector
            value = sum(
                bernoulli_entropy(model.predict_probability(feature_fn(detection)))
                for detection in record['final_detections']
            )
        profile['u_' + ERROR_KEYS[name]] = float(value)
    return profile


def mean_eua_profile(
    records: Iterable[Mapping[str, Any]], predictors: EPALPredictors,
) -> Dict[str, float]:
    totals = {ERROR_KEYS[name]: 0.0 for name in predictors.models}
    count = 0
    for record in records:
        profile = predict_eua_profile(record, predictors)
        for key in totals:
            totals[key] += profile['u_' + key]
        count += 1
    return {key: value / count if count else 0.0 for key, value in totals.items()}


def inverse_scale_weights(
    means: Mapping[str, float], weight_eps: float = 1e-6,
) -> Dict[str, float]:
    if not means or not np.isfinite(weight_eps) or weight_eps <= 0:
        raise ValueError('EPAL weights require active components and positive finite epsilon')
    if any(not np.isfinite(value) or value < 0 for value in means.values()):
        raise ValueError('EPAL mean uncertainty must be finite and nonnegative')
    raw = {key: 1.0 / (value + weight_eps) for key, value in means.items()}
    total = sum(raw.values())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('EPAL inverse-scale weights are non-finite')
    return {key: len(raw) * value / total for key, value in raw.items()}


def score_unlabeled_records(
    records: Iterable[Mapping[str, Any]], predictors: EPALPredictors,
    weights: Mapping[str, float],
) -> List[Dict[str, Any]]:
    if set(weights) != {ERROR_KEYS[name] for name in predictors.models}:
        raise ValueError('EPAL weights must match active predictors')
    scored = []
    for record in records:
        profile = predict_eua_profile(record, predictors)
        weighted = {key: weights[key] * profile['u_' + key] for key in weights}
        score = float(sum(weighted.values()))
        if not np.isfinite(score):
            raise ValueError('EPAL score must be finite')
        scored.append({
            'image_id': record['image_id'],
            'score': score,
            'profile': profile,
            'weighted_profile': weighted,
            'final_detection_count': len(record['final_detections']),
        })
    return sorted(scored, key=lambda item: (-item['score'], image_id_sort_key(item['image_id'])))
