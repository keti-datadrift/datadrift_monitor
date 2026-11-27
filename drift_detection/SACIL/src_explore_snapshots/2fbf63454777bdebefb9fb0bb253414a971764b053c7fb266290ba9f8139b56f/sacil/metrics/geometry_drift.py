from __future__ import annotations

from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> float | None:
    if values.numel() == 0 or not bool(mask.any()):
        return None
    return float(values[mask].mean().item())


def summarize_geometry_drift(
    per_anchor_drift: dict[str, Tensor],
    leaf_weights: Tensor,
    internal_weights: Tensor,
    conflict_threshold: float = 1.0 - 1e-6,
    stable_threshold: float = 1.0 - 1e-6,
) -> dict:
    leaf_drift = per_anchor_drift["leaf"].detach().cpu()
    internal_drift = per_anchor_drift["internal"].detach().cpu()
    leaf_weights = leaf_weights.detach().cpu()
    internal_weights = internal_weights.detach().cpu()
    result = {
        "leaf_mean": (
            None if leaf_drift.numel() == 0 else float(leaf_drift.mean().item())
        ),
        "leaf_conflict_mean": _masked_mean(
            leaf_drift, leaf_weights < conflict_threshold
        ),
        "leaf_stable_mean": _masked_mean(
            leaf_drift, leaf_weights >= stable_threshold
        ),
        "internal_mean": (
            None
            if internal_drift.numel() == 0
            else float(internal_drift.mean().item())
        ),
        "internal_conflict_mean": _masked_mean(
            internal_drift, internal_weights < conflict_threshold
        ),
        "internal_stable_mean": _masked_mean(
            internal_drift, internal_weights >= stable_threshold
        ),
    }
    return result
