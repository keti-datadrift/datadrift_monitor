from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch

# Avoid loky's platform-specific physical-core probe, which can emit a
# non-fatal UnicodeDecodeError on Korean Windows installations.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.nn import functional as F


class AFCMultiProxyClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        class_chunks: Sequence[int],
        *,
        proxies_per_class: int = 10,
        gamma: float = 1.0,
        distance_scale: float = 3.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or proxies_per_class <= 0:
            raise ValueError("feature and proxy counts must be positive")
        if not class_chunks or any(int(value) <= 0 for value in class_chunks):
            raise ValueError("class chunks must be positive")
        self.feature_dim = int(feature_dim)
        self.proxies_per_class = int(proxies_per_class)
        self.gamma = float(gamma)
        self.distance_scale = float(distance_scale)
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        self.class_chunks = [int(value) for value in class_chunks]
        self._weights = nn.ParameterList()
        for class_count in self.class_chunks:
            weights = nn.Parameter(
                torch.empty(
                    class_count * self.proxies_per_class,
                    self.feature_dim,
                )
            )
            nn.init.kaiming_normal_(weights, nonlinearity="linear")
            self._weights.append(weights)

    @property
    def num_classes(self) -> int:
        return sum(self.class_chunks)

    @property
    def weights(self) -> Tensor:
        return torch.cat(tuple(self._weights), dim=0)

    @property
    def weight(self) -> Tensor:
        """One representative vector per class for hierarchy construction."""
        normalized = F.normalize(self.weights, dim=1)
        return normalized.view(
            self.num_classes,
            self.proxies_per_class,
            self.feature_dim,
        ).mean(dim=1)

    @property
    def new_weights(self) -> nn.Parameter:
        return self._weights[-1]

    @property
    def old_weights(self) -> tuple[nn.Parameter, ...]:
        return tuple(self._weights[:-1])

    def forward(self, features: Tensor) -> Tensor:
        normalized_features = self.distance_scale * F.normalize(
            features, p=2, dim=-1
        )
        normalized_weights = self.distance_scale * F.normalize(
            self.weights, p=2, dim=-1
        )
        # Exact AFC ``-stable_cosine_distance`` implementation, including
        # its clamp/mask behavior around zero distances.
        combined = torch.cat((normalized_features, normalized_weights))
        squared = torch.add(
            combined.pow(2).sum(dim=1, keepdim=True).expand(combined.size(0), -1),
            combined.T.pow(2).sum(dim=0, keepdim=True).expand(combined.size(0), -1),
        ) - 2 * (combined @ combined.T)
        squared = torch.clamp(squared, min=0.0)
        error_mask = squared <= 0.0
        squared = squared * (~error_mask).float()
        off_diagonal = 1 - torch.eye(
            *squared.size(), device=squared.device
        )
        squared = squared * off_diagonal
        proxy_similarities = -squared[
            : normalized_features.shape[0], normalized_features.shape[0] :
        ]
        per_class = proxy_similarities.view(
            features.shape[0],
            self.num_classes,
            self.proxies_per_class,
        )
        attention = torch.softmax(self.gamma * per_class, dim=-1)
        return (per_class * attention).sum(dim=-1)

    def append_imprinted(self, weights: Tensor) -> None:
        if weights.ndim != 3:
            raise ValueError("imprinted weights must have shape [classes, proxies, D]")
        if (
            weights.shape[1] != self.proxies_per_class
            or weights.shape[2] != self.feature_dim
        ):
            raise ValueError("imprinted proxy shape does not match classifier")
        flattened = weights.reshape(-1, self.feature_dim).detach().clone()
        self._weights.append(nn.Parameter(flattened))
        self.class_chunks.append(int(weights.shape[0]))


def kmeans_imprinted_weights(
    class_features: Sequence[Tensor],
    reference_weights: Tensor,
    *,
    proxies_per_class: int,
    random_state: int | None = None,
) -> Tensor:
    if not class_features:
        raise ValueError("at least one class feature tensor is required")
    average_norm = reference_weights.detach().float().norm(dim=1).mean()
    result = []
    for features in class_features:
        if features.ndim != 2:
            raise ValueError("class features must be matrices")
        if features.shape[0] < proxies_per_class:
            raise ValueError("not enough samples for requested proxy count")
        normalized = F.normalize(features.detach().float(), dim=1)
        clusterer = KMeans(
            n_clusters=int(proxies_per_class),
        )
        centers = clusterer.fit(normalized.cpu().numpy()).cluster_centers_
        centers_tensor = torch.from_numpy(
            np.asarray(centers, dtype=np.float32)
        )
        result.append(centers_tensor * average_norm.cpu())
    return torch.stack(result, dim=0)
