from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def podnet_nca_loss(
    similarities: Tensor,
    targets: Tensor,
    *,
    scale: Tensor | float = 1.0,
    margin: float = 0.6,
    exclude_positive_denominator: bool = True,
) -> Tensor:
    """PODNet's Local Similarity Classifier NCA objective."""

    if similarities.ndim != 2 or targets.ndim != 1:
        raise ValueError("similarities must be a matrix and targets a vector")
    if similarities.shape[0] != targets.shape[0]:
        raise ValueError("similarities and targets have different batch sizes")
    if targets.numel() and (
        int(targets.min()) < 0
        or int(targets.max()) >= similarities.shape[1]
    ):
        raise ValueError("target is outside the similarity matrix")

    # Keep the released PODNet/AFC NCA behavior exactly: the scalar margin is
    # subtracted from every entry, and the positive denominator entry is
    # replaced by zero after row-wise stabilization.
    adjusted = scale * (similarities - float(margin))
    if not exclude_positive_denominator:
        return F.cross_entropy(adjusted, targets)

    stable = adjusted - adjusted.max(1)[0].view(-1, 1)
    row = torch.arange(targets.shape[0], device=targets.device)
    numerator = stable[row, targets]
    denominator_terms = stable.clone()
    denominator_terms[row, targets] = 0.0
    losses = numerator - torch.log(torch.exp(denominator_terms).sum(-1))
    return -losses.mean()


def pod_spatial_loss(
    current_maps: Sequence[Tensor],
    reference_maps: Sequence[Tensor],
    *,
    normalize: bool = True,
) -> Tensor:
    """POD-spatial loss from PODNet.

    Each squared activation map is pooled independently over width and height,
    concatenated, optionally L2-normalized, and compared to the teacher.
    """

    if len(current_maps) != len(reference_maps):
        raise ValueError("current and reference map counts differ")
    if not current_maps:
        raise ValueError("POD-spatial requires at least one feature map")
    loss = current_maps[0].new_zeros(())
    for current, reference in zip(current_maps, reference_maps):
        if current.shape != reference.shape or current.ndim != 4:
            raise ValueError("POD-spatial map shapes do not match")
        current_sq = torch.pow(current, 2)
        reference_sq = torch.pow(reference, 2)
        current_h = current_sq.sum(dim=3).flatten(start_dim=1)
        current_w = current_sq.sum(dim=2).flatten(start_dim=1)
        reference_h = reference_sq.sum(dim=3).flatten(start_dim=1)
        reference_w = reference_sq.sum(dim=2).flatten(start_dim=1)
        current_pooled = torch.cat((current_h, current_w), dim=1)
        reference_pooled = torch.cat((reference_h, reference_w), dim=1)
        if normalize:
            current_pooled = F.normalize(current_pooled, dim=1)
            reference_pooled = F.normalize(reference_pooled, dim=1)
        loss = loss + torch.mean(
            torch.frobenius_norm(
                current_pooled - reference_pooled, dim=-1
            )
        )
    return loss / len(current_maps)


def pod_flat_loss(current_features: Tensor, reference_features: Tensor) -> Tensor:
    """POD-flat cosine embedding loss."""

    if (
        current_features.shape != reference_features.shape
        or current_features.ndim != 2
    ):
        raise ValueError("POD-flat feature shapes do not match")
    labels = torch.ones(
        current_features.shape[0],
        device=current_features.device,
        dtype=current_features.dtype,
    )
    return F.cosine_embedding_loss(
        current_features, reference_features.detach(), labels
    )
