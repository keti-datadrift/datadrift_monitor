from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProjectedGradientResult:
    gradients: tuple[Tensor | None, ...]
    conflict: bool
    cosine: float
    projection_coefficient: float
    insertion_retained_ratio: float


def project_insertion_gradient(
    stability_gradients: Sequence[Tensor | None],
    insertion_gradients: Sequence[Tensor | None],
    *,
    epsilon: float = 1e-12,
) -> ProjectedGradientResult:
    """Project a conflicting insertion gradient off the stability gradient.

    Both inputs must already include their configured loss coefficients.  The
    returned gradients are the optimizer-ready sum of stability and projected
    insertion gradients.  Missing gradients are handled without fabricating
    tensors for parameters that neither objective uses.
    """

    if len(stability_gradients) != len(insertion_gradients):
        raise ValueError("gradient sequences must have identical lengths")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("projection epsilon must be finite and positive")

    device = next(
        (
            gradient.device
            for gradient in (*stability_gradients, *insertion_gradients)
            if gradient is not None
        ),
        torch.device("cpu"),
    )
    dot = torch.zeros((), device=device, dtype=torch.float64)
    stability_norm_sq = torch.zeros_like(dot)
    insertion_norm_sq = torch.zeros_like(dot)
    for stability, insertion in zip(stability_gradients, insertion_gradients):
        if stability is not None:
            stability_norm_sq += stability.detach().double().square().sum()
        if insertion is not None:
            insertion_norm_sq += insertion.detach().double().square().sum()
        if stability is not None and insertion is not None:
            dot += (stability.detach().double() * insertion.detach().double()).sum()

    stability_norm = torch.sqrt(stability_norm_sq)
    insertion_norm = torch.sqrt(insertion_norm_sq)
    denominator = stability_norm * insertion_norm
    cosine = (
        float((dot / denominator).item())
        if float(denominator.item()) > epsilon
        else 0.0
    )
    conflict = bool(dot.item() < 0.0 and stability_norm_sq.item() > epsilon)
    coefficient = (
        float((dot / (stability_norm_sq + epsilon)).item())
        if conflict
        else 0.0
    )

    combined: list[Tensor | None] = []
    projected_insertion_norm_sq = torch.zeros_like(dot)
    for stability, insertion in zip(stability_gradients, insertion_gradients):
        projected = insertion
        if conflict and insertion is not None and stability is not None:
            projected = insertion - coefficient * stability
        if projected is not None:
            projected_insertion_norm_sq += projected.detach().double().square().sum()
        if stability is None:
            combined.append(None if projected is None else projected.detach())
        elif projected is None:
            combined.append(stability.detach())
        else:
            combined.append((stability + projected).detach())

    retained_ratio = (
        float((torch.sqrt(projected_insertion_norm_sq) / insertion_norm).item())
        if float(insertion_norm.item()) > epsilon
        else 0.0
    )
    return ProjectedGradientResult(
        gradients=tuple(combined),
        conflict=conflict,
        cosine=cosine,
        projection_coefficient=coefficient,
        insertion_retained_ratio=retained_ratio,
    )
