from __future__ import annotations

from typing import Any

import torch

from feature_generation.base import BaseFeatureGenerator


class LDAProjectionFeatureGenerator(BaseFeatureGenerator):
    name = "lda_projection"

    def __init__(self, num_depths: int, config: dict[str, Any]):
        super().__init__(num_depths=num_depths)
        self.config = config
        self.selected_depth = int(config.get("lda", {}).get("label_depth", num_depths))
        self.model: dict[str, torch.Tensor] = {}

    def fit(
        self,
        train_features: torch.Tensor,
        labels_by_depth: dict[int, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LDAProjectionFeatureGenerator":
        del metadata
        if labels_by_depth is None or self.selected_depth not in labels_by_depth:
            raise ValueError(
                f"lda_projection requires labels for depth {self.selected_depth}"
            )
        features = train_features.detach().double().cpu()
        labels = labels_by_depth[self.selected_depth].long().cpu()
        valid = labels >= 0
        features = features[valid]
        labels = labels[valid]
        classes = torch.unique(labels, sorted=True)
        if classes.numel() < 2:
            raise ValueError("lda_projection requires at least two classes")

        global_mean = features.mean(dim=0)
        within_scatter = torch.zeros(
            (features.shape[1], features.shape[1]),
            dtype=features.dtype,
        )
        between_scatter = torch.zeros_like(within_scatter)
        for class_id in classes.tolist():
            class_features = features[labels == class_id]
            class_mean = class_features.mean(dim=0)
            centered = class_features - class_mean
            within_scatter += centered.transpose(0, 1) @ centered
            mean_delta = class_mean - global_mean
            between_scatter += class_features.shape[0] * torch.outer(mean_delta, mean_delta)

        within_covariance = within_scatter / max(features.shape[0] - classes.numel(), 1)
        between_covariance = between_scatter / max(features.shape[0], 1)
        within_covariance = 0.5 * (within_covariance + within_covariance.transpose(0, 1))
        eigenvalues, eigenvectors = torch.linalg.eigh(within_covariance)
        eigen_floor = (
            torch.finfo(features.dtype).eps
            * features.shape[1]
            * eigenvalues.max().clamp_min(1.0)
        )
        whitening = (
            eigenvectors
            * eigenvalues.clamp_min(eigen_floor).rsqrt().unsqueeze(0)
        ) @ eigenvectors.transpose(0, 1)
        whitened_between = whitening.transpose(0, 1) @ between_covariance @ whitening
        whitened_between = 0.5 * (
            whitened_between + whitened_between.transpose(0, 1)
        )
        fisher_values, fisher_vectors = torch.linalg.eigh(whitened_between)
        target_dim = min(int(classes.numel() - 1), features.shape[1])
        order = torch.argsort(fisher_values, descending=True)[:target_dim]
        projection = whitening @ fisher_vectors[:, order]

        self.model = {
            "mean": global_mean.float(),
            "projection": projection.float().contiguous(),
            "fisher_values": fisher_values[order].float().contiguous(),
            "target_dim": torch.tensor(target_dim, dtype=torch.long),
            "num_classes": torch.tensor(classes.numel(), dtype=torch.long),
        }
        return self

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        if not self.model:
            raise RuntimeError("lda_projection must be fit before transform")
        mean = self.model["mean"].to(features.device, dtype=features.dtype)
        projection = self.model["projection"].to(features.device, dtype=features.dtype)
        transformed = (features - mean) @ projection
        return {
            depth: transformed for depth in range(1, self.num_depths + 1)
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "config": self.config,
            "selected_depth": self.selected_depth,
            "model": self.model,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "LDAProjectionFeatureGenerator":
        super().load_state_dict(state)
        self.config = state["config"]
        self.selected_depth = int(state["selected_depth"])
        self.model = state["model"]
        return self
