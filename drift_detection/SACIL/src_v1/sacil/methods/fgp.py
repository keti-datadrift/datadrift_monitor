from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RectifiedCosineLinear(nn.Module):
    """FGP-ICL's cosine classifier with an appended learnable bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("classifier dimensions must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 1))
        else:
            self.register_parameter("bias", None)
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("scale", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        if self.scale is not None:
            nn.init.ones_(self.scale)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.in_features:
            raise ValueError("FGP classifier received an invalid feature shape")
        weights = self.weight
        if self.bias is not None:
            ones = features.new_ones((features.shape[0], 1))
            features = torch.cat((features, ones), dim=1)
            weights = torch.cat((weights, self.bias), dim=1)
        logits = F.linear(
            F.normalize(features, dim=1),
            F.normalize(weights, dim=1),
        )
        return logits if self.scale is None else self.scale * logits


def pairwise_squared_euclidean(left: Tensor, right: Tensor) -> Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("pairwise distance inputs must have matching features")
    # FGP author release ``FGP._Euclidean``.  Keep the addmm computation
    # instead of replacing it with torch.cdist: the latter has multiple
    # kernels and does not guarantee the same floating-point trajectory.
    distances = left.pow(2).sum(dim=1, keepdim=True).expand(
        len(left), len(right)
    ) + right.pow(2).sum(dim=1, keepdim=True).expand(
        len(right), len(left)
    ).T
    return distances.addmm(left, right.T, beta=1.0, alpha=-2.0)


def fgp_graph_preservation_loss(
    current_features: Tensor,
    reference_features: Tensor,
    current_old_weights: Tensor,
    reference_old_weights: Tensor,
    *,
    distance_weight: float = 0.5,
) -> Tensor:
    """Weighted-Euclidean feature-graph preservation from FGP-ICL."""

    if current_features.shape != reference_features.shape:
        raise ValueError("current and reference feature shapes differ")
    if current_old_weights.shape != reference_old_weights.shape:
        raise ValueError("current and reference classifier shapes differ")
    current_distance = pairwise_squared_euclidean(
        F.normalize(current_features, dim=1),
        F.normalize(current_old_weights, dim=1),
    )
    reference_distance = pairwise_squared_euclidean(
        F.normalize(reference_features.detach(), dim=1),
        F.normalize(reference_old_weights.detach(), dim=1),
    )
    weights = torch.exp(-float(distance_weight) * reference_distance)
    return (weights * (current_distance - reference_distance).square()).sum(
    ) / current_features.shape[0]


def scheduled_fgp_weight(
    known_classes: int,
    total_classes: int,
    *,
    base_weight: float = 0.1,
) -> float:
    if known_classes <= 0 or total_classes < known_classes:
        raise ValueError("invalid FGP class counts")
    return float(base_weight) * math.sqrt(known_classes / total_classes)
