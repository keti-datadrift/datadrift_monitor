from __future__ import annotations

from typing import Any

import torch

from feature_generation.base import BaseFeatureGenerator
from feature_generation.utils.mask_utils import build_mask_from_scores
from feature_generation.utils.statistics import variance_scores


class VarianceMaskFeatureGenerator(BaseFeatureGenerator):
    name = "variance_mask"

    def __init__(self, num_depths: int, config: dict[str, Any]):
        super().__init__(num_depths=num_depths)
        self.config = config
        self.mask_cfg = dict(config.get("mask", {}))
        self.scores: dict[int, torch.Tensor] = {}
        self.masks: dict[int, torch.Tensor] = {}

    def fit(self, train_features: torch.Tensor, labels_by_depth=None, metadata=None) -> "VarianceMaskFeatureGenerator":
        del metadata
        if labels_by_depth is None:
            raise ValueError("variance_mask requires labels_by_depth")
        feats = train_features.float().cpu()
        for depth in range(1, self.num_depths + 1):
            scores = variance_scores(feats, labels_by_depth[depth].long().cpu())
            self.scores[depth] = scores
            self.masks[depth] = build_mask_from_scores(scores, self.mask_cfg)
        return self

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        return {
            depth: features * self.masks[depth].to(features.device, dtype=features.dtype)
            for depth in range(1, self.num_depths + 1)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "config": self.config,
            "mask_cfg": self.mask_cfg,
            "scores": self.scores,
            "masks": self.masks,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "VarianceMaskFeatureGenerator":
        super().load_state_dict(state)
        self.config = state["config"]
        self.mask_cfg = dict(state["mask_cfg"])
        self.scores = {int(k): v for k, v in state["scores"].items()}
        self.masks = {int(k): v for k, v in state["masks"].items()}
        return self
