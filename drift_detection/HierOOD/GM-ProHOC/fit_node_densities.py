from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_merged_config, save_config
from core.density import fit_node_distributions
from core.feature_io import load_artifact, save_artifact
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
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
    density = fit_node_distributions(
        train_features,
        train_artifact["targets"].long(),
        hierarchy,
        train_artifact["class_names"],
        covariance_type=density_cfg.get("covariance_type", "diag"),
        eps=density_cfg["eps"],
    )
    density["config"] = density_cfg
    density["feature_source"] = feature_meta
    save_artifact(density, experiment_dir / "node_density.pt")


if __name__ == "__main__":
    main()
