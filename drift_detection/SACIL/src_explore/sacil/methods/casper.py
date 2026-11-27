"""CaSpeR-IL spectral objective copied from the author release.

This module keeps the Neural Nearest Neighbours relaxation and
``xitorch.linalg.symeig`` partial eigensolver used by:

* ``CaSpeR-IL/utils/knn.py``
* ``CaSpeR-IL/utils/spectral_analysis.py``
* ``CaSpeR-IL/models/utils/egap_model.py``

Only device-safe tensor construction and the function-style wrapper expected
by the unified runner are added.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from xitorch import LinearOperator
from xitorch.linalg import symeig


def pairwise_feature_distance(features: Tensor) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    return ((features.unsqueeze(0) - features.unsqueeze(1)) ** 2).sum(-1)


def _log_one_minus_exp(values: Tensor, expm1_guard: float = 1e-7) -> Tensor:
    threshold = values < math.log(0.5)
    outputs = torch.zeros_like(values)
    outputs[threshold] = torch.log1p(-values[threshold].exp())
    expxm1 = torch.expm1(values[~threshold])
    forward = (-expxm1).log()
    backward = (-expxm1 + expm1_guard).log()
    outputs[~threshold] = forward.detach() + (
        backward - backward.detach()
    )
    return outputs


class NeuralNearestNeighbors(nn.Module):
    """Neural k-NN layer bundled with the official CaSpeR release."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
        self.k = int(k)

    def forward(self, distances: Tensor) -> Tensor:
        batch, rows, columns = distances.shape
        temperature = self.log_temp.view(1, 1, 1).exp()
        scaled = distances.clone()
        if self.training:
            finite = scaled.data > -float("inf")
            scaled[finite] = scaled[finite] / temperature[0, 0, 0]
        else:
            scaled = scaled / temperature
        logits = scaled.view(batch * rows, columns)
        samples = []
        for _ in range(self.k):
            weights = F.log_softmax(logits, dim=1)
            samples.append(weights.exp().view(batch, rows, columns))
            logits = logits + _log_one_minus_exp(weights)
        return torch.stack(samples, dim=3)


def neural_knn_weights(
    negative_distances: Tensor,
    k: int,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Compatibility entry point using the release's exact k-NN layer."""

    if temperature != 1.0:
        raise ValueError("official CaSpeR uses NeuralNN temperature=1")
    if negative_distances.ndim != 2:
        raise ValueError("distance scores must be a matrix")
    layer = NeuralNearestNeighbors(k).to(negative_distances.device)
    return layer(negative_distances.unsqueeze(0)).squeeze(0).sum(-1)


def _author_affinity(distances: Tensor, k: int) -> tuple[Tensor, Tensor]:
    masked = distances.clone()
    diagonal = torch.eye(
        len(masked), device=masked.device, dtype=torch.bool
    )
    masked[diagonal] = math.inf
    neural_knn = NeuralNearestNeighbors(k).to(masked.device)

    hard = torch.zeros_like(masked)
    indices = masked.topk(k, largest=False)[1]
    hard[
        torch.arange(len(indices), device=indices.device).unsqueeze(1),
        indices,
    ] = 1
    soft = neural_knn(-masked.unsqueeze(0)).squeeze().sum(-1)
    hard = ((hard + hard.T) > 0).float()
    soft = soft + soft.T
    affinity = hard.detach() + (soft - soft.detach())
    degree_values = affinity.sum(1)
    if bool((degree_values == 0).any()):
        raise RuntimeError("CaSpeR affinity produced a zero degree")
    degree = torch.diag(degree_values)
    return affinity, degree


def _normalize_affinity(affinity: Tensor, degree: Tensor) -> Tensor:
    mask = torch.eye(len(degree), device=degree.device, dtype=torch.bool)
    inverse = torch.diag(degree[mask].pow(-0.5))
    if bool(torch.isinf(inverse).any()):
        raise RuntimeError("CaSpeR degree normalization produced infinity")
    return inverse @ affinity @ inverse


def casper_spectral_loss(
    features: Tensor,
    *,
    num_classes: int,
    k: int = 10,
    temperature: float = 1.0,
    solver: str = "xitorch",
) -> Tensor:
    """Author-release CaSpeR eigengap loss."""

    if temperature != 1.0:
        raise ValueError("official CaSpeR uses NeuralNN temperature=1")
    if solver not in {"xitorch", "partial"}:
        raise ValueError("official CaSpeR requires xitorch.symeig")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if k <= 0 or k >= len(features):
        raise ValueError("k must be in [1, number_of_samples)")

    distances = pairwise_feature_distance(features)
    affinity, degree = _author_affinity(distances, int(k))
    laplacian = torch.eye(
        len(affinity), device=affinity.device, dtype=affinity.dtype
    ) - _normalize_affinity(affinity, degree)
    n = int(num_classes)
    pair_count = min(2 * n, len(laplacian))
    eigenvalues, _ = symeig(
        LinearOperator.m(laplacian, True), pair_count
    )
    if len(eigenvalues) <= n + 1:
        raise ValueError("CaSpeR replay graph is too small for its eigengap")
    return eigenvalues[: n + 1].sum() - eigenvalues[n + 1]
