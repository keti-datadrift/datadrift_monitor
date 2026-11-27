from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def cosine_soft_confusion(
    features: Tensor,
    targets: Tensor,
    classifier_weights: Tensor,
    temperature: float = 0.2,
) -> Tensor:
    """Class-conditional soft confusion from unscaled cosine logits."""
    if features.ndim != 2 or classifier_weights.ndim != 2:
        raise ValueError("features and classifier weights must be matrices")
    if features.shape[0] != targets.numel():
        raise ValueError("feature and target counts do not match")
    if features.shape[1] != classifier_weights.shape[1]:
        raise ValueError("feature dimensions do not match")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    num_classes = classifier_weights.shape[0]
    if targets.numel() == 0:
        raise ValueError("soft confusion requires at least one sample")
    if targets.min().item() < 0 or targets.max().item() >= num_classes:
        raise ValueError("target outside classifier range")

    cosine = F.normalize(features.float(), dim=1) @ F.normalize(
        classifier_weights.detach().float(), dim=1
    ).t()
    probabilities = torch.softmax(cosine / float(temperature), dim=1)
    confusion = torch.zeros(
        num_classes,
        num_classes,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    counts = torch.zeros(
        num_classes, dtype=probabilities.dtype, device=probabilities.device
    )
    confusion.index_add_(0, targets.long(), probabilities)
    counts.index_add_(
        0,
        targets.long(),
        torch.ones_like(targets, dtype=probabilities.dtype),
    )
    confusion = confusion / counts.clamp_min(1).unsqueeze(1)
    confusion.fill_diagonal_(0.0)
    return confusion


def symmetric_affinity(confusion: Tensor) -> Tensor:
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    affinity = 0.5 * (confusion + confusion.t())
    affinity.fill_diagonal_(0.0)
    return affinity

