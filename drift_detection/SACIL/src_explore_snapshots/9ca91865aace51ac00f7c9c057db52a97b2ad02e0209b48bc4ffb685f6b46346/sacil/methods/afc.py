from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def afc_nca_loss(
    similarities: Tensor,
    targets: Tensor,
    scale: Tensor | float,
    *,
    margin: float = 0.6,
    exclude_positive_denominator: bool = True,
) -> Tensor:
    scaled = scale * (similarities - float(margin))
    if not exclude_positive_denominator:
        return F.cross_entropy(scaled, targets)
    scaled = scaled - scaled.max(1)[0].view(-1, 1)
    target_values = scaled.gather(1, targets[:, None]).squeeze(1)
    denominator = scaled.clone()
    denominator.scatter_(1, targets[:, None], 0.0)
    losses = target_values - torch.log(torch.exp(denominator).sum(-1))
    return -losses.mean()


def afc_pod_loss(
    reference_attentions: Sequence[Tensor],
    current_attentions: Sequence[Tensor],
    importance: Sequence[Tensor],
) -> Tensor:
    if not (
        len(reference_attentions)
        == len(current_attentions)
        == len(importance)
    ):
        raise ValueError("AFC POD inputs must have the same layer count")
    if not reference_attentions:
        raise ValueError("AFC POD requires at least one attention layer")
    loss = reference_attentions[0].new_zeros(())
    for reference, current, layer_importance in zip(
        reference_attentions, current_attentions, importance
    ):
        if reference.shape != current.shape or reference.ndim != 4:
            raise ValueError("AFC attention shapes do not match")
        reference_pixels = F.normalize(
            torch.pow(reference, 2).view(
                reference.shape[0], reference.shape[1], -1
            ),
            dim=2,
            p=2,
        )
        current_pixels = F.normalize(
            torch.pow(current, 2).view(
                current.shape[0], current.shape[1], -1
            ),
            dim=2,
            p=2,
        )
        per_channel = torch.frobenius_norm(
            reference_pixels - current_pixels, dim=-1
        )
        if layer_importance.numel() != per_channel.shape[1]:
            raise ValueError("AFC importance dimension does not match attention")
        loss = loss + (
            per_channel
            * layer_importance.detach().reshape(1, -1)
        ).mean()
    return loss / len(reference_attentions)


def scheduled_afc_factor(
    seen_class_count: int,
    new_class_count: int,
    base_factor: float = 4.0,
) -> float:
    if seen_class_count <= 0 or new_class_count <= 0:
        raise ValueError("class counts must be positive")
    return float(base_factor) * math.sqrt(
        float(seen_class_count) / float(new_class_count)
    )
