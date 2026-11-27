from __future__ import annotations

from typing import Any

import torch

from feature_generation.base import BaseFeatureGenerator


class PCAFeatureGenerator(BaseFeatureGenerator):
    name = "pca_projection"

    def __init__(self, num_depths: int, config: dict[str, Any]):
        super().__init__(num_depths=num_depths)
        self.config = config
        self.per_depth = bool(config.get("pca", {}).get("per_depth", True))
        self.whiten = bool(config.get("pca", {}).get("whiten", False))
        self.output_dim_by_depth = {
            int(k): int(v) for k, v in (config.get("output_dim_by_depth", {}) or {}).items()
        }
        self.models: dict[int, dict[str, torch.Tensor]] = {}

    def fit(self, train_features: torch.Tensor, labels_by_depth=None, metadata=None) -> "PCAFeatureGenerator":
        del labels_by_depth, metadata
        if self.per_depth:
            for depth in range(1, self.num_depths + 1):
                self.models[depth] = self._fit_one(train_features, self._target_dim(depth))
        else:
            shared = self._fit_one(train_features, max(self._target_dim(depth) for depth in range(1, self.num_depths + 1)))
            for depth in range(1, self.num_depths + 1):
                self.models[depth] = shared
        return self

    def _target_dim(self, depth: int) -> int:
        return min(self.output_dim_by_depth.get(depth, self.output_dim_by_depth.get(str(depth), 0) or 0) or 0, 10**9)

    def _fit_one(self, features: torch.Tensor, target_dim: int) -> dict[str, torch.Tensor]:
        feats = features.float().cpu()
        mean = feats.mean(dim=0)
        centered = feats - mean
        _, singular_values, v = torch.linalg.svd(centered, full_matrices=False)
        max_rank = v.shape[0]
        if target_dim <= 0:
            target_dim = max_rank
        target_dim = min(target_dim, max_rank)
        components = v[:target_dim].contiguous()
        explained_var = (singular_values[:target_dim].pow(2) / max(centered.shape[0] - 1, 1)).contiguous()
        return {
            "mean": mean,
            "components": components,
            "explained_var": explained_var,
            "target_dim": torch.tensor(target_dim, dtype=torch.long),
        }

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs = {}
        for depth in range(1, self.num_depths + 1):
            model = self.models[depth]
            mean = model["mean"].to(features.device, dtype=features.dtype)
            components = model["components"].to(features.device, dtype=features.dtype)
            transformed = (features - mean) @ components.T
            if self.whiten:
                scale = torch.sqrt(model["explained_var"].to(features.device, dtype=features.dtype).clamp_min(1e-12))
                transformed = transformed / scale
            outputs[depth] = transformed
        return outputs

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "config": self.config,
            "per_depth": self.per_depth,
            "whiten": self.whiten,
            "output_dim_by_depth": self.output_dim_by_depth,
            "models": self.models,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "PCAFeatureGenerator":
        super().load_state_dict(state)
        self.config = state["config"]
        self.per_depth = bool(state["per_depth"])
        self.whiten = bool(state["whiten"])
        self.output_dim_by_depth = {int(k): int(v) for k, v in state["output_dim_by_depth"].items()}
        self.models = {int(k): v for k, v in state["models"].items()}
        return self
