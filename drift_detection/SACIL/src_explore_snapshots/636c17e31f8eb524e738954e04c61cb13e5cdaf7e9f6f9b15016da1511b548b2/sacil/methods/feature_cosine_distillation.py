from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_feature_distillation_loss(
    current_features: Tensor,
    reference_features: Tensor,
    *,
    sample_weights: Tensor | None = None,
    epsilon: float = 1e-12,
) -> Tensor:
    """LUCIR-style cosine embedding loss on matched feature rows.

    The reference branch is always detached.  Optional row weights are
    normalized by their sum so an all-one mask is exactly the unweighted
    objective; this hook is reserved for a hierarchy-routed follow-up.
    """

    if current_features.shape != reference_features.shape:
        raise ValueError("current and reference features must have one shape")
    if current_features.ndim != 2:
        raise ValueError("features must have shape [batch, feature_dim]")
    if not torch.isfinite(current_features).all() or not torch.isfinite(
        reference_features
    ).all():
        raise ValueError("feature distillation inputs must be finite")
    losses = 1.0 - F.cosine_similarity(
        current_features,
        reference_features.detach(),
        dim=1,
        eps=float(epsilon),
    )
    if sample_weights is None:
        return losses.mean()
    if sample_weights.ndim != 1 or sample_weights.shape[0] != losses.shape[0]:
        raise ValueError("sample weights must have shape [batch]")
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("sample weights must be finite and non-negative")
    denominator = weights.sum()
    if float(denominator.detach()) <= 0:
        raise ValueError("sample weights must have positive total mass")
    return (weights * losses).sum() / denominator.clamp_min(float(epsilon))


def normalized_cosine_classifier_logits(
    features: Tensor,
    class_weights: Tensor,
    *,
    scale: float,
    epsilon: float = 1e-12,
) -> Tensor:
    """Cosine classifier used by LUCIR/GeoDL-style training controls."""

    if features.ndim != 2 or class_weights.ndim != 2:
        raise ValueError("features and class weights must be matrices")
    if features.shape[1] != class_weights.shape[1]:
        raise ValueError("feature and class-weight dimensions do not match")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("cosine classifier scale must be finite and positive")
    return float(scale) * F.linear(
        F.normalize(features, dim=1, eps=float(epsilon)),
        F.normalize(class_weights, dim=1, eps=float(epsilon)),
    )


def cosine_imprinted_weights(
    class_features: list[Tensor],
    old_weights: Tensor,
    *,
    epsilon: float = 1e-12,
) -> Tensor:
    """Initialize new cosine templates from normalized class means."""

    if old_weights.ndim != 2 or old_weights.shape[0] == 0:
        raise ValueError("old class weights must be a non-empty matrix")
    old_norm = old_weights.detach().norm(dim=1).mean()
    values = []
    for features in class_features:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("each new class requires a non-empty feature matrix")
        if features.shape[1] != old_weights.shape[1]:
            raise ValueError("imprinting feature dimension does not match")
        prototype = F.normalize(features, dim=1, eps=float(epsilon)).mean(
            dim=0, keepdim=True
        )
        values.append(F.normalize(prototype, dim=1, eps=float(epsilon)) * old_norm)
    return torch.cat(values, dim=0)
