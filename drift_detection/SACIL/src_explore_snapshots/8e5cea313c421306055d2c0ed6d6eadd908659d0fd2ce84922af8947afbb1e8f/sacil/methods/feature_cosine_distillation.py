from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class HierarchyRoutedFeatureWeights:
    """Per-row feature-distillation routing derived from a fixed BGS reference."""

    sample_weights: Tensor
    old_conflict_mask: Tensor
    old_outside_mask: Tensor
    new_mask: Tensor

    def detached_metrics(self) -> dict[str, float | int]:
        weights = self.sample_weights.detach()
        metrics: dict[str, float | int] = {
            "sample_count": int(weights.numel()),
            "weight_sum": float(weights.sum()),
            "mean_weight": float(weights.mean()),
        }
        total_weight = float(metrics["weight_sum"])
        for name, mask in (
            ("old_conflict", self.old_conflict_mask),
            ("old_outside", self.old_outside_mask),
            ("new", self.new_mask),
        ):
            detached_mask = mask.detach()
            count = int(detached_mask.sum())
            weight_sum = float(weights[detached_mask].sum())
            metrics[f"{name}_count"] = count
            metrics[f"{name}_weight_sum"] = weight_sum
            metrics[f"{name}_mean_weight"] = (
                0.0 if count == 0 else weight_sum / count
            )
            metrics[f"{name}_effective_weight"] = (
                0.0 if total_weight == 0.0 else weight_sum / total_weight
            )
        return metrics


def hierarchy_routed_feature_sample_weights(
    targets: Tensor,
    replay_mask: Tensor,
    *,
    known_classes: int,
    sample_region_ids: Sequence[str | None],
    old_conflict_weight: float,
    old_outside_weight: float,
    new_weight: float,
) -> HierarchyRoutedFeatureWeights:
    """Route feature-KD rows using the frozen BGS old-class partition.

    An old replay row is a conflict row exactly when its old class has a
    non-``None`` entry in ``BGSReference.sample_region_ids``.  Non-replay
    rows are incoming-new rows and never index the old-class mapping.
    """

    if targets.ndim != 1 or replay_mask.ndim != 1:
        raise ValueError("targets and replay_mask must have shape [batch]")
    if targets.shape != replay_mask.shape:
        raise ValueError("targets and replay_mask must have one shape")
    if replay_mask.dtype != torch.bool:
        raise ValueError("replay_mask must be boolean")
    if known_classes <= 0:
        raise ValueError("known_classes must be positive")
    regions = tuple(sample_region_ids)
    if len(regions) != known_classes:
        raise ValueError(
            "BGS sample_region_ids must contain one entry per known class"
        )
    configured = {
        "old_conflict_weight": float(old_conflict_weight),
        "old_outside_weight": float(old_outside_weight),
        "new_weight": float(new_weight),
    }
    for name, value in configured.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if sum(configured.values()) <= 0:
        raise ValueError("hierarchy-routed weights need positive configured mass")

    old_targets = targets[replay_mask]
    if old_targets.numel() and bool(
        ((old_targets < 0) | (old_targets >= known_classes)).any()
    ):
        raise ValueError("replay targets must index known old classes")
    new_targets = targets[~replay_mask]
    if new_targets.numel() and bool((new_targets < known_classes).any()):
        raise ValueError("non-replay targets must index incoming new classes")

    conflict_lookup = torch.tensor(
        [region_id is not None for region_id in regions],
        device=targets.device,
        dtype=torch.bool,
    )
    old_conflict_mask = torch.zeros_like(replay_mask)
    if old_targets.numel():
        old_conflict_mask[replay_mask] = conflict_lookup[old_targets.long()]
    old_outside_mask = replay_mask & ~old_conflict_mask
    new_mask = ~replay_mask

    weights = torch.full(
        targets.shape,
        configured["new_weight"],
        device=targets.device,
        dtype=torch.float32,
    )
    weights[old_outside_mask] = configured["old_outside_weight"]
    weights[old_conflict_mask] = configured["old_conflict_weight"]
    if float(weights.sum()) <= 0:
        raise ValueError("hierarchy-routed batch has zero sample-weight mass")
    return HierarchyRoutedFeatureWeights(
        sample_weights=weights,
        old_conflict_mask=old_conflict_mask,
        old_outside_mask=old_outside_mask,
        new_mask=new_mask,
    )


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
    # Preserve the exact legacy reduction (including its gradient graph) for
    # the all-one control, rather than merely returning a close equivalent.
    if torch.equal(weights, torch.ones_like(weights)):
        return losses.mean()
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
