from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CosineClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        initial_scale: float = 10.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        scale = torch.tensor(float(initial_scale))
        if learnable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.learnable_scale = bool(learnable_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def cosine_logits(self, features: Tensor) -> Tensor:
        normalized_features = F.normalize(features, dim=1)
        normalized_weights = F.normalize(self.weight, dim=1)
        return normalized_features @ normalized_weights.t()

    def forward(self, features: Tensor, apply_scale: bool = True) -> Tensor:
        logits = self.cosine_logits(features)
        if apply_scale:
            logits = logits * self.scale.clamp_min(1e-6)
        return logits

    def expanded(self, num_classes: int) -> "CosineClassifier":
        target_classes = int(num_classes)
        if target_classes < self.num_classes:
            raise ValueError("classifier cannot shrink")
        if target_classes == self.num_classes:
            return self
        expanded = CosineClassifier(
            self.in_features,
            target_classes,
            initial_scale=float(self.scale.detach().item()),
            learnable_scale=self.learnable_scale,
        ).to(self.weight.device)
        with torch.no_grad():
            expanded.weight[: self.num_classes].copy_(self.weight)
            expanded.scale.copy_(self.scale)
        return expanded

    def initialize_rows(self, start: int, prototypes: Tensor) -> None:
        stop = int(start) + prototypes.shape[0]
        if prototypes.ndim != 2 or prototypes.shape[1] != self.in_features:
            raise ValueError("prototype shape does not match classifier")
        if start < 0 or stop > self.num_classes:
            raise ValueError("classifier initialization range is invalid")
        with torch.no_grad():
            self.weight[start:stop].copy_(F.normalize(prototypes, dim=1))

