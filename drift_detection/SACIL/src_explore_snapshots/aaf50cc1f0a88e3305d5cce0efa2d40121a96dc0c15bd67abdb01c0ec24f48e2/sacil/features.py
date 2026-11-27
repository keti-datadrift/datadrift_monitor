from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class FeatureCollection:
    features: Tensor
    targets: Tensor
    original_targets: Tensor
    indices: Tensor

    def normalized(self) -> "FeatureCollection":
        return FeatureCollection(
            features=F.normalize(self.features, dim=1),
            targets=self.targets,
            original_targets=self.original_targets,
            indices=self.indices,
        )


@torch.inference_mode()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> FeatureCollection:
    was_training = model.training
    model.eval()
    features: list[Tensor] = []
    targets: list[Tensor] = []
    original_targets: list[Tensor] = []
    indices: list[Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_features = model.extract_features(images)
        features.append(batch_features.detach().cpu())
        targets.append(batch["target"].detach().cpu().long())
        original_targets.append(
            batch["original_target"].detach().cpu().long()
        )
        indices.append(batch["index"].detach().cpu().long())
    if was_training:
        model.train()
    if not features:
        raise ValueError("cannot collect features from an empty loader")
    return FeatureCollection(
        features=torch.cat(features, dim=0),
        targets=torch.cat(targets, dim=0),
        original_targets=torch.cat(original_targets, dim=0),
        indices=torch.cat(indices, dim=0),
    )

