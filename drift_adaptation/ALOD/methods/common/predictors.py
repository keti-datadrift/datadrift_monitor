"""Small scikit-learn backed predictor wrappers for AL methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _feature_array(features: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _target_array(targets: Sequence[float]) -> np.ndarray:
    return np.asarray(targets, dtype=np.float64).reshape(-1)


def _normalize_features(values: np.ndarray) -> tuple:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale, mean, scale


@dataclass
class BinaryLogisticModel:
    """Binary logistic model with PAL-compatible fallback behavior."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    constant_probability: Optional[float] = None
    estimator: Optional[Any] = None
    fallback_reason: Optional[str] = None

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]],
        targets: Sequence[int],
        learning_rate: float = 0.1,
        max_iter: int = 300,
        l2: float = 1e-4,
        min_samples: int = 4,
        n_features: Optional[int] = None,
    ) -> 'BinaryLogisticModel':
        del learning_rate
        x = _feature_array(features)
        y = _target_array(targets)
        if x.ndim != 2 or x.shape[0] == 0:
            return cls.constant(0.5, n_features=n_features or 2, reason='no_samples')
        if x.shape[0] != y.shape[0]:
            raise ValueError('features and targets must have the same length')
        if x.shape[0] < min_samples or len(set(y.astype(int).tolist())) < 2:
            probability = float((y.sum() + 1.0) / (len(y) + 2.0))
            return cls.constant(probability, n_features=x.shape[1],
                                reason='insufficient_or_single_class')

        x_norm, mean, scale = _normalize_features(x)
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:
            raise ImportError(
                'scikit-learn is required for PAL/EPAL predictor fitting. '
                'Install it with `pip install -r requirements.txt`.'
            ) from exc

        estimator = LogisticRegression(
            penalty='l2',
            C=1.0 / max(float(l2), 1e-12),
            solver='lbfgs',
            max_iter=max_iter,
            random_state=0,
        )
        estimator.fit(x_norm, y.astype(int))
        weights = estimator.coef_.reshape(-1).astype(np.float64, copy=False)
        bias = float(estimator.intercept_[0])
        return cls(
            weights=weights,
            bias=bias,
            mean=mean,
            scale=scale,
            estimator=estimator,
        )

    @classmethod
    def constant(
        cls,
        probability: float,
        n_features: int = 2,
        reason: Optional[str] = None,
    ) -> 'BinaryLogisticModel':
        return cls(
            weights=np.zeros(n_features, dtype=np.float64),
            bias=0.0,
            mean=np.zeros(n_features, dtype=np.float64),
            scale=np.ones(n_features, dtype=np.float64),
            constant_probability=float(min(max(probability, 1e-6), 1.0 - 1e-6)),
            fallback_reason=reason,
        )

    @property
    def used_fallback(self) -> bool:
        return self.constant_probability is not None

    def predict_probability(self, feature: Sequence[float]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        x = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        x_norm = (x - self.mean) / self.scale
        if self.estimator is not None:
            return float(self.estimator.predict_proba(x_norm)[0, 1])
        logit = float(np.dot(x_norm.reshape(-1), self.weights) + self.bias)
        return float(_sigmoid(np.asarray([logit]))[0])


@dataclass
class PoissonCountModel:
    """Poisson count model for image-wise error-count prediction."""

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    constant_count: Optional[float] = None
    estimator: Optional[Any] = None
    max_count: float = 1e6
    fallback_reason: Optional[str] = None

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]],
        targets: Sequence[float],
        max_iter: int = 300,
        l2: float = 1e-4,
        min_samples: int = 4,
        max_count: float = 1e6,
        n_features: Optional[int] = None,
    ) -> 'PoissonCountModel':
        x = _feature_array(features)
        y = _target_array(targets)
        if x.ndim != 2 or x.shape[0] == 0:
            return cls.constant(0.0, n_features=n_features or 2,
                                max_count=max_count, reason='no_samples')
        if x.shape[0] != y.shape[0]:
            raise ValueError('features and targets must have the same length')
        y = np.maximum(y, 0.0)
        if x.shape[0] < min_samples or float(np.std(y)) <= 1e-12:
            count = float(y.mean()) if y.size else 0.0
            return cls.constant(count, n_features=x.shape[1], max_count=max_count,
                                reason='insufficient_or_constant_target')

        x_norm, mean, scale = _normalize_features(x)
        try:
            from sklearn.linear_model import PoissonRegressor
        except ImportError as exc:
            raise ImportError(
                'scikit-learn is required for EPAL Poisson predictor fitting. '
                'Install it with `pip install -r requirements.txt`.'
            ) from exc

        estimator = PoissonRegressor(
            alpha=max(float(l2), 0.0),
            max_iter=max_iter,
        )
        estimator.fit(x_norm, y)
        weights = estimator.coef_.reshape(-1).astype(np.float64, copy=False)
        bias = float(estimator.intercept_)
        return cls(
            weights=weights,
            bias=bias,
            mean=mean,
            scale=scale,
            estimator=estimator,
            max_count=float(max_count),
        )

    @classmethod
    def constant(
        cls,
        count: float,
        n_features: int = 2,
        max_count: float = 1e6,
        reason: Optional[str] = None,
    ) -> 'PoissonCountModel':
        return cls(
            weights=np.zeros(n_features, dtype=np.float64),
            bias=0.0,
            mean=np.zeros(n_features, dtype=np.float64),
            scale=np.ones(n_features, dtype=np.float64),
            constant_count=float(min(max(count, 0.0), max_count)),
            max_count=float(max_count),
            fallback_reason=reason,
        )

    @property
    def used_fallback(self) -> bool:
        return self.constant_count is not None

    def predict_count(self, feature: Sequence[float]) -> float:
        if self.constant_count is not None:
            return self.constant_count
        x = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        x_norm = (x - self.mean) / self.scale
        if self.estimator is not None:
            value = float(self.estimator.predict(x_norm)[0])
        else:
            value = float(np.exp(np.dot(x_norm.reshape(-1), self.weights) + self.bias))
        return float(min(max(value, 0.0), self.max_count))
