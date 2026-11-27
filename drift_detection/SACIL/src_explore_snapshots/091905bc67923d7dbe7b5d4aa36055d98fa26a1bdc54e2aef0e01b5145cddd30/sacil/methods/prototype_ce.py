from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def prototype_logits(
    features: Tensor,
    prototypes: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Cosine logits whose decision rule is normalized nearest-class-mean.

    For normalized vectors, maximizing cosine similarity is exactly
    equivalent to minimizing squared Euclidean distance.  The temperature
    changes the training gradients but not the predicted class.
    """

    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must be matrices")
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("feature and prototype dimensions differ")
    if prototypes.shape[0] == 0:
        raise ValueError("at least one prototype is required")
    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    normalized_features = F.normalize(features.float(), dim=1)
    normalized_prototypes = F.normalize(
        prototypes.detach().float(), dim=1
    )
    return (
        normalized_features @ normalized_prototypes.T
    ) / float(temperature)


def prototype_cross_entropy(
    features: Tensor,
    targets: Tensor,
    prototypes: Tensor,
    *,
    temperature: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Cross-entropy over prototype logits.

    Prototypes are stop-gradient statistics recomputed from the current
    representation.  The returned logits are also used for training accuracy,
    avoiding dependence on a separate parametric classifier.
    """

    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("targets must match the feature batch")
    if targets.numel() and (
        int(targets.min()) < 0
        or int(targets.max()) >= prototypes.shape[0]
    ):
        raise ValueError("target is outside the prototype bank")
    logits = prototype_logits(
        features, prototypes, temperature=temperature
    )
    return F.cross_entropy(logits, targets), logits
