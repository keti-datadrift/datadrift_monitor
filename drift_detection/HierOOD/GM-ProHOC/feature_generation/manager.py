from __future__ import annotations

from typing import Any

import torch

from feature_generation.base import BaseFeatureGenerator


class FeatureGenerationManager:
    def __init__(self, generator: BaseFeatureGenerator, hierarchy_depths: int):
        self.generator = generator
        self.hierarchy_depths = int(hierarchy_depths)

    def fit(
        self,
        train_features: torch.Tensor,
        labels_by_depth: dict[int, torch.Tensor] | None,
        metadata: dict[str, Any] | None = None,
    ) -> "FeatureGenerationManager":
        self.generator.fit(train_features, labels_by_depth=labels_by_depth, metadata=metadata)
        return self

    def transform_split(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        depth_features = self.generator.transform(features)
        expected = set(range(1, self.hierarchy_depths + 1))
        actual = set(depth_features.keys())
        if actual != expected:
            raise ValueError(f"generator returned depths {sorted(actual)}, expected {sorted(expected)}")
        return depth_features

    def state_dict(self) -> dict[str, Any]:
        return {
            "hierarchy_depths": self.hierarchy_depths,
            "generator_name": self.generator.name,
            "generator_state": self.generator.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> "FeatureGenerationManager":
        self.hierarchy_depths = int(state["hierarchy_depths"])
        self.generator.load_state_dict(state["generator_state"])
        return self
