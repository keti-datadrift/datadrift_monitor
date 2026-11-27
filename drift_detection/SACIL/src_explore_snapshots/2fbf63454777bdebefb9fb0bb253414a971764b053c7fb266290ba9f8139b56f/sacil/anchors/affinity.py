from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def anchor_affinity(features: Tensor, anchors: Tensor) -> Tensor:
    if features.ndim != 2 or anchors.ndim != 2:
        raise ValueError("features and anchors must be matrices")
    if features.shape[1] != anchors.shape[1]:
        raise ValueError("feature and anchor dimensions do not match")
    return F.normalize(features, dim=1) @ F.normalize(anchors, dim=1).t()

