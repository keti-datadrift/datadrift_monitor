from __future__ import annotations

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
