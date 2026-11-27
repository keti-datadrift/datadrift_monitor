from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_merged_config, save_config
from core.density import (
    fit_depth_masked_node_distributions,
    fit_depth_fisher_precision_node_distributions,
    fit_depth_reweighted_node_distributions,
    fit_node_distributions,
)
from core.feature_io import load_artifact, save_artifact
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    parser.add_argument("--output", help="Density artifact path. Default: <experiment_dir>/node_density.pt")
    args = parser.parse_args()

    config = load_merged_config(args.config, args.feature_gen_config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config.yaml")

    train_artifact = load_artifact(experiment_dir / "features_train.pt")
    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    density_cfg = config["density"]
    train_features, feature_meta = resolve_feature_tensor(config, experiment_dir, "train")
    feature_mask_type = density_cfg.get("feature_mask_type", "none")
    if str(feature_mask_type).lower() == "depth_fisher":
        density = fit_depth_masked_node_distributions(
            train_features,
            train_artifact["targets"].long(),
            hierarchy,
            train_artifact["class_names"],
            mask_dim=int(density_cfg.get("feature_mask_dim", 64)),
            covariance_type=density_cfg.get("covariance_type", "diag"),
            eps=density_cfg["eps"],
            covariance_shrinkage=density_cfg.get("covariance_shrinkage", 0.0),
        )
    elif str(density_cfg.get("covariance_type", "")).lower() == "fisher_precision_shared_full":
        density = fit_depth_fisher_precision_node_distributions(
            train_features,
            train_artifact["targets"].long(),
            hierarchy,
            train_artifact["class_names"],
            precision_strength=float(density_cfg.get("fisher_precision_strength", 0.25)),
            min_weight=float(density_cfg.get("fisher_precision_min", 0.5)),
            max_weight=float(density_cfg.get("fisher_precision_max", 2.0)),
            eps=density_cfg["eps"],
            covariance_shrinkage=density_cfg.get("covariance_shrinkage", 0.0),
        )
    elif str(feature_mask_type).lower() == "depth_fisher_reweight":
        density = fit_depth_reweighted_node_distributions(
            train_features,
            train_artifact["targets"].long(),
            hierarchy,
            train_artifact["class_names"],
            gamma=float(density_cfg.get("feature_weight_gamma", 0.25)),
            min_weight=float(density_cfg.get("feature_weight_min", 0.5)),
            max_weight=float(density_cfg.get("feature_weight_max", 2.0)),
            covariance_type=density_cfg.get("covariance_type", "diag"),
            eps=density_cfg["eps"],
            covariance_shrinkage=density_cfg.get("covariance_shrinkage", 0.0),
        )
    else:
        density = fit_node_distributions(
            train_features,
            train_artifact["targets"].long(),
            hierarchy,
            train_artifact["class_names"],
            covariance_type=density_cfg.get("covariance_type", "diag"),
            eps=density_cfg["eps"],
            covariance_shrinkage=density_cfg.get("covariance_shrinkage", 0.0),
        )
    density["config"] = density_cfg
    density["feature_source"] = feature_meta
    output_path = Path(args.output) if args.output else experiment_dir / "node_density.pt"
    save_artifact(density, output_path)


if __name__ == "__main__":
    main()
