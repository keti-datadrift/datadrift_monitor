"""PAL LIUS scoring."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence

from methods.common.predictors import BinaryLogisticModel
from methods.pal.inference import lius_feature


def binary_entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def train_classwise_models(
    matched_detections: Iterable[Dict[str, Any]],
    min_samples: int = 4,
) -> Dict[Any, BinaryLogisticModel]:
    features_by_class: Dict[Any, List[Sequence[float]]] = defaultdict(list)
    targets_by_class: Dict[Any, List[int]] = defaultdict(list)
    for det in matched_detections:
        category_id = det.get('category_id')
        if category_id is None:
            continue
        features_by_class[category_id].append(lius_feature(det))
        targets_by_class[category_id].append(int(det.get('target', 0)))

    models = {}
    for category_id, features in features_by_class.items():
        models[category_id] = BinaryLogisticModel.fit(
            features,
            targets_by_class[category_id],
            min_samples=min_samples,
        )
    return models


def score_unlabeled_detections(
    detections: Iterable[Dict[str, Any]],
    models: Dict[Any, BinaryLogisticModel],
) -> List[Dict[str, Any]]:
    scored = []
    for det in detections:
        record = dict(det)
        model = models.get(record.get('category_id'))
        probability = model.predict_probability(lius_feature(record)) if model else 0.5
        record['tp_probability'] = probability
        record['lius_score'] = binary_entropy(probability)
        scored.append(record)
    return scored
