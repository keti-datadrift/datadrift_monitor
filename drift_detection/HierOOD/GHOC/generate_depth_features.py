from __future__ import annotations

import argparse
from pathlib import Path

from core.config import load_merged_config, save_config
from core.feature_io import load_artifact, save_artifact
from core.hierarchy_depth_labels import (
    build_depth_label_matrix,
    depth_labels_to_dict,
    targets_to_depth_labels,
)
from feature_generation import FeatureGenerationManager, build_feature_generator
from feature_generation.utils.io import make_generated_split_payload
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def _load_split_artifact(experiment_dir: Path, split: str):
    return load_artifact(experiment_dir / f"features_{split}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    args = parser.parse_args()

    config = load_merged_config(args.config, args.feature_gen_config)
    fg_cfg = config.get("feature_generation", {})
    if not fg_cfg.get("enable", False):
        raise ValueError("feature_generation.enable must be true to run generate_depth_features.py")

    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config_feature_gen.yaml")

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    train_artifact = _load_split_artifact(experiment_dir, "train")
    depth_label_matrix, depth_meta = build_depth_label_matrix(
        hierarchy,
        train_artifact["class_names"],
        use_pruned=config.get("hierarchy", {}).get("use_pruned_for_loss_labels", True),
        hierarchy_path=dataset_cfg["hierarchy"],
    )
    train_depth_labels = targets_to_depth_labels(train_artifact["targets"], depth_label_matrix)
    labels_by_depth = depth_labels_to_dict(train_depth_labels)

    generator = build_feature_generator(config, hierarchy)
    manager = FeatureGenerationManager(generator, hierarchy_depths=hierarchy.max_depth)
    manager.fit(
        train_artifact["features"].float(),
        labels_by_depth=labels_by_depth,
        metadata={"depth_label_meta": depth_meta, "class_names": train_artifact["class_names"]},
    )

    generator_payload = {
        "generator_type": generator.name,
        "generator_config": fg_cfg,
        "depths": list(range(1, hierarchy.max_depth + 1)),
        "state": manager.state_dict(),
        "fit_metadata": {
            "depth_label_meta": depth_meta,
            "train_feature_artifact": str(experiment_dir / "features_train.pt"),
            "source_config": config.get("_meta", {}),
        },
    }
    save_artifact(generator_payload, experiment_dir / "feature_generator.pt")

    split_artifact_paths = {
        "train": experiment_dir / "features_train.pt",
        "val": experiment_dir / "features_val.pt",
        "ood": experiment_dir / "features_ood.pt",
    }
    for farood_name in dataset_cfg.get("farood_sets", []) or []:
        split_artifact_paths[f"farood_{farood_name.replace('/', '-')}"] = (
            experiment_dir / f"features_farood_{farood_name.replace('/', '-')}.pt"
        )

    for split, artifact_path in split_artifact_paths.items():
        split_artifact = load_artifact(artifact_path)
        depth_features = manager.transform_split(split_artifact["features"].float())
        payload = make_generated_split_payload(
            split_artifact,
            depth_features=depth_features,
            generator_type=generator.name,
            generator_config=fg_cfg,
            base_feature_path=str(artifact_path),
        )
        payload["depth_label_meta"] = depth_meta
        save_artifact(payload, experiment_dir / f"generated_features_{split}.pt")


if __name__ == "__main__":
    main()
