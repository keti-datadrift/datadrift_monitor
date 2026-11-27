from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_similarity_matrix(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("similarity inputs must have matching features")
    return F.normalize(left, dim=1) @ F.normalize(right, dim=1).T


def cross_space_clustering_loss(
    current_features: Tensor,
    reference_features: Tensor,
    targets: Tensor,
) -> Tensor:
    """CSCCT cross-space clustering objective.

    Same-class pairs are pulled together while different-class pairs are
    pushed apart, matching the signed pair mask in the official release.
    """

    if current_features.shape != reference_features.shape:
        raise ValueError("current and reference feature shapes differ")
    if targets.ndim != 1 or targets.shape[0] != current_features.shape[0]:
        raise ValueError("invalid targets for CSC loss")
    signed = torch.where(
        targets[:, None] == targets[None, :],
        torch.ones((), device=targets.device),
        -torch.ones((), device=targets.device),
    )
    similarity = cosine_similarity_matrix(
        current_features, reference_features.detach()
    )
    return ((1.0 - similarity) * signed).mean()


def controlled_transfer_loss(
    current_features: Tensor,
    reference_features: Tensor,
    targets: Tensor,
    *,
    known_classes: int,
    temperature: float = 2.0,
) -> Tensor:
    """CSCCT controlled-transfer relation distillation."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if current_features.shape != reference_features.shape:
        raise ValueError("current and reference feature shapes differ")
    new_mask = targets >= int(known_classes)
    old_mask = ~new_mask
    if not bool(new_mask.any()) or not bool(old_mask.any()):
        return current_features.sum() * 0.0
    current_relation = cosine_similarity_matrix(
        current_features[new_mask], current_features[old_mask]
    )
    reference_relation = cosine_similarity_matrix(
        reference_features[new_mask].detach(),
        reference_features[old_mask].detach(),
    )
    return (
        F.kl_div(
            F.log_softmax(current_relation / temperature, dim=1),
            F.softmax(reference_relation / temperature, dim=1),
            # The CSCCT release uses ``nn.KLDivLoss()`` with its historical
            # default reduction.  That is element-wise mean, not batchmean.
            reduction="mean",
        )
        * temperature**2
    )
