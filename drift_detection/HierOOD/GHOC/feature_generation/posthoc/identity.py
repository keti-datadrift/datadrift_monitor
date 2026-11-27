from __future__ import annotations

import torch

from feature_generation.base import BaseFeatureGenerator


class IdentityFeatureGenerator(BaseFeatureGenerator):
    name = "identity"

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        return {depth: features.clone() for depth in range(1, self.num_depths + 1)}
