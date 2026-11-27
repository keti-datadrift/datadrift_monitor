from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RipsNet(nn.Module):
    """Permutation-invariant point-cloud to persistence-image regressor."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] | None = None,
        output_dim: int = 400,
        operator: str = "mean",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (64, 32, 32, 50, 100, 200)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(hidden_dims) != 6:
            raise ValueError("RipsNet requires six hidden dimensions")
        if operator not in {"mean", "sum"}:
            raise ValueError("RipsNet operator must be mean or sum")
        dimensions = (int(feature_dim),) + hidden_dims + (int(output_dim),)
        self.point_layers = nn.ModuleList(
            [
                nn.Linear(dimensions[0], dimensions[1], bias=True),
                nn.Linear(dimensions[1], dimensions[2], bias=True),
                nn.Linear(dimensions[2], dimensions[3], bias=True),
            ]
        )
        self.cloud_layers = nn.ModuleList(
            [
                nn.Linear(dimensions[3], dimensions[4], bias=False),
                nn.Linear(dimensions[4], dimensions[5], bias=False),
                nn.Linear(dimensions[5], dimensions[6], bias=False),
                nn.Linear(dimensions[6], dimensions[7], bias=False),
            ]
        )
        self.feature_dim = int(feature_dim)
        self.output_dim = int(output_dim)
        self.operator = operator

    def forward(self, point_cloud: Tensor) -> Tensor:
        if point_cloud.ndim not in {2, 3}:
            raise ValueError("point cloud must have shape [N,D] or [B,N,D]")
        if point_cloud.shape[-1] != self.feature_dim:
            raise ValueError("point-cloud feature dimension mismatch")
        unbatched = point_cloud.ndim == 2
        if unbatched:
            point_cloud = point_cloud.unsqueeze(0)
        hidden = point_cloud
        for layer in self.point_layers:
            hidden = F.relu(layer(hidden), inplace=False)
        if self.operator == "mean":
            hidden = hidden.mean(dim=1)
        else:
            hidden = hidden.sum(dim=1)
        for layer in self.cloud_layers[:-1]:
            hidden = F.relu(layer(hidden), inplace=False)
        output = torch.sigmoid(self.cloud_layers[-1](hidden))
        return output.squeeze(0) if unbatched else output


def load_frozen_ripsnet(
    checkpoint_path: str | Path,
    *,
    expected_feature_dim: int,
) -> tuple[RipsNet, dict]:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    metadata = dict(checkpoint.get("metadata", {}))
    feature_dim = int(metadata.get("feature_dim", expected_feature_dim))
    if feature_dim != int(expected_feature_dim):
        raise ValueError(
            f"RipsNet expects {feature_dim} features, "
            f"model emits {expected_feature_dim}"
        )
    network = RipsNet(
        feature_dim=feature_dim,
        hidden_dims=metadata.get("hidden_dims"),
        output_dim=int(metadata.get("output_dim", 400)),
        operator=str(metadata.get("operator", "mean")),
    )
    state = checkpoint.get("model", checkpoint.get("state_dict"))
    if state is None:
        raise KeyError("RipsNet checkpoint has no model state")
    network.load_state_dict(state)
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network, metadata


class TopologyDistillationLoss(nn.Module):
    def __init__(self, ripsnet: RipsNet) -> None:
        super().__init__()
        self.ripsnet = ripsnet

    def train(self, mode: bool = True):
        super().train(False)
        self.ripsnet.eval()
        return self

    def forward(
        self,
        current_features: Tensor,
        reference_features: Tensor,
    ) -> Tensor:
        if current_features.shape != reference_features.shape:
            raise ValueError("current and reference feature shapes differ")
        if current_features.ndim != 2:
            raise ValueError("TopKD expects a feature point cloud [N,D]")
        if current_features.shape[0] < 2:
            return current_features.sum() * 0.0
        with torch.no_grad():
            reference_topology = self.ripsnet(reference_features)
        current_topology = self.ripsnet(current_features)
        return F.mse_loss(current_topology, reference_topology)
