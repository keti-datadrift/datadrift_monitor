from __future__ import annotations

from typing import Any

import torch

from feature_generation.base import BaseFeatureGenerator


class UMAPFeatureGenerator(BaseFeatureGenerator):
    name = "umap_projection"

    def __init__(self, num_depths: int, config: dict[str, Any]):
        super().__init__(num_depths=num_depths)
        self.config = config
        self.output_dim_by_depth = {
            int(k): int(v) for k, v in (config.get("output_dim_by_depth", {}) or {}).items()
        }
        self.umap_cfg = dict(config.get("umap", {}))
        self.models: dict[int, Any] = {}

    def fit(self, train_features: torch.Tensor, labels_by_depth=None, metadata=None) -> "UMAPFeatureGenerator":
        del metadata
        try:
            import umap
        except ImportError as exc:
            raise ImportError(
                "UMAP feature generation requires the optional 'umap-learn' package."
            ) from exc

        feats = train_features.float().cpu().numpy()
        supervised = bool(self.umap_cfg.get("supervised", True))
        for depth in range(1, self.num_depths + 1):
            n_components = self.output_dim_by_depth.get(depth, self.umap_cfg.get("n_components", feats.shape[1]))
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=self.umap_cfg.get("n_neighbors", 15),
                min_dist=self.umap_cfg.get("min_dist", 0.1),
                metric=self.umap_cfg.get("metric", "cosine"),
                random_state=self.umap_cfg.get("random_state", 42),
            )
            labels = None
            if supervised and labels_by_depth is not None:
                labels = labels_by_depth[depth].long().cpu().numpy()
            model.fit(feats, y=labels)
            self.models[depth] = model
        return self

    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        outputs = {}
        feats = features.float().cpu().numpy()
        for depth in range(1, self.num_depths + 1):
            transformed = self.models[depth].transform(feats)
            outputs[depth] = torch.from_numpy(transformed).to(features.dtype)
        return outputs

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "config": self.config,
            "output_dim_by_depth": self.output_dim_by_depth,
            "umap_cfg": self.umap_cfg,
            "models": self.models,
        }

    def load_state_dict(self, state: dict[str, Any]) -> "UMAPFeatureGenerator":
        super().load_state_dict(state)
        self.config = state["config"]
        self.output_dim_by_depth = {int(k): int(v) for k, v in state["output_dim_by_depth"].items()}
        self.umap_cfg = dict(state["umap_cfg"])
        self.models = {int(k): v for k, v in state["models"].items()}
        return self
