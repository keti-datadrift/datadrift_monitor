from __future__ import annotations

from typing import Any

from feature_generation.posthoc.fisher_mask import FisherMaskFeatureGenerator
from feature_generation.posthoc.identity import IdentityFeatureGenerator
from feature_generation.posthoc.pca_projection import PCAFeatureGenerator
from feature_generation.posthoc.umap_projection import UMAPFeatureGenerator
from feature_generation.posthoc.variance_mask import VarianceMaskFeatureGenerator


def build_feature_generator(config: dict[str, Any], hierarchy_info) -> object:
    fg_cfg = config.get("feature_generation", {})
    fg_type = fg_cfg.get("type", "identity").lower()
    num_depths = int(hierarchy_info.max_depth)

    if fg_type == "identity":
        return IdentityFeatureGenerator(num_depths=num_depths)
    if fg_type == "pca_projection":
        return PCAFeatureGenerator(num_depths=num_depths, config=fg_cfg)
    if fg_type == "umap_projection":
        return UMAPFeatureGenerator(num_depths=num_depths, config=fg_cfg)
    if fg_type == "variance_mask":
        return VarianceMaskFeatureGenerator(num_depths=num_depths, config=fg_cfg)
    if fg_type == "fisher_mask":
        return FisherMaskFeatureGenerator(num_depths=num_depths, config=fg_cfg)
    raise ValueError(f"Unsupported feature_generation.type: {fg_type}")
