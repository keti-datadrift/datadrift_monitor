"""Fit the selected EPAL error predictors."""

from dataclasses import dataclass
from typing import Any, Dict

from methods.common.predictors import BinaryLogisticModel, PoissonCountModel
from methods.epal.components import COMPONENT_ORDER, DEFAULT_COMPONENTS, normalize_components
from methods.epal.features import (
    CLASSIFICATION_FEATURE_KEYS, LOCALIZATION_FEATURE_KEYS, MISS_FEATURE_KEYS,
)

FEATURE_KEYS = {
    'cecp': CLASSIFICATION_FEATURE_KEYS,
    'lecp': LOCALIZATION_FEATURE_KEYS,
    'mocp': MISS_FEATURE_KEYS,
}


@dataclass
class EPALPredictors:
    models: Dict[str, Any]
    diagnostics: Dict[str, Any]


def fit_predictors(
    label_data: Dict[str, Any],
    components: str = DEFAULT_COMPONENTS,
    min_samples: int = 4,
) -> EPALPredictors:
    active = normalize_components(components).split('+')
    foreground = [
        example for example in label_data['detection_examples']
        if example['is_foreground']
    ]
    sources = {
        'cecp': (foreground, 'classification_feature', 'y_cls'),
        'lecp': (foreground, 'localization_feature', 'y_loc'),
        'mocp': (label_data['image_examples'], 'miss_feature', 'n_miss'),
    }
    models = {}
    diagnostics = {}
    for name in COMPONENT_ORDER:
        if name not in active:
            diagnostics[name] = {'enabled': False}
            continue
        examples, feature_key, target_key = sources[name]
        features = [example[feature_key] for example in examples]
        targets = [example[target_key] for example in examples]
        model_type = PoissonCountModel if name == 'mocp' else BinaryLogisticModel
        model = model_type.fit(features, targets, min_samples=min_samples, n_features=2)
        models[name] = model
        details = {
            'enabled': True,
            'feature_keys': list(FEATURE_KEYS[name]),
            'sample_count': len(targets),
            'target_mean': float(sum(targets) / len(targets)) if targets else None,
            'fallback_reason': model.fallback_reason,
            'scaler_mean': model.mean.tolist(),
            'scaler_scale': model.scale.tolist(),
            'coefficients': model.weights.tolist(),
            'intercept': float(model.bias),
            'constant_value': (
                model.constant_count if name == 'mocp' else model.constant_probability
            ),
        }
        if name != 'mocp':
            details['positive_rate'] = details['target_mean']
        diagnostics[name] = details
    return EPALPredictors(models=models, diagnostics=diagnostics)
