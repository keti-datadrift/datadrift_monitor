from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config, save_config
from core.feature_io import save_artifact
from core.hierarchy_labels import build_leaf_path_matrix
from core.model_factory import (
    backbone_summary,
    build_backbone,
    load_backbone_checkpoint,
    maybe_wrap_dataparallel,
)
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import gen_custom_dataset, gen_datasets, get_id_classes


def dataset_transform_kwargs(config):
    dataset_cfg = config["dataset"]
    backbone_type = config["backbone"]["type"].lower()
    preset = dataset_cfg.get("transform_preset")
    if preset is None:
        preset = "clip" if backbone_type == "clip" else "imagenet"
    return {
        "preset": preset,
        "mean": dataset_cfg.get("mean"),
        "std": dataset_cfg.get("std"),
        "resize": dataset_cfg.get("resize"),
        "cropsize": dataset_cfg.get("cropsize"),
        "model_name": config["backbone"].get("model_name"),
    }


def collect_features(model, loader, device):
    feats = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in tqdm(loader, desc="Extract"):
            inputs = inputs.to(device)
            features = model(inputs)
            feats.append(features.cpu())
            targets.append(batch_targets.long().cpu())
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)


def split_payload(split_name, features, targets, class_names, node_targets, config, extra=None):
    payload = {
        "features": features,
        "targets": targets,
        "class_names": class_names,
        "node_targets": node_targets,
        "split": split_name,
        "backbone_config": backbone_summary(config),
    }
    if extra:
        payload.update(extra)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config.yaml")

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    transform_kwargs = dataset_transform_kwargs(config)
    train_ds, val_ds, ood_ds = gen_datasets(
        dataset_cfg["datadir"],
        id_classes,
        hierarchy.ood_train_classes,
        **transform_kwargs,
    )

    model = build_backbone(config).to(device)
    load_backbone_checkpoint(model, experiment_dir / "checkpoint_backbone.pt", map_location=device)
    model = maybe_wrap_dataparallel(model, config)
    model.eval()

    batch_size = config["dataloader"].get("eval_batch_size", config["dataloader"]["batch_size"])
    num_workers = config["dataloader"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    use_pruned = config.get("hierarchy", {}).get("use_pruned_for_loss_labels", True)
    leaf_path_matrix, path_meta = build_leaf_path_matrix(
        hierarchy,
        train_ds.classes,
        use_pruned=use_pruned,
        hierarchy_path=dataset_cfg["hierarchy"],
    )

    extra_meta = {}
    feature_cfg = config.get("feature_extraction", {})
    if feature_cfg.get("save_leaf_path_matrix", False):
        extra_meta["leaf_path_matrix"] = leaf_path_matrix
    if feature_cfg.get("save_index_mappings", False):
        extra_meta["leaf_name_to_idx"] = {name: idx for idx, name in enumerate(train_ds.classes)}
        extra_meta["node_name_to_idx"] = {name: idx for idx, name in enumerate(hierarchy.id_node_list)}
    extra_meta["path_label_config"] = path_meta

    if feature_cfg.get("save_train", True):
        train_features, train_targets = collect_features(model, train_loader, device)
        save_artifact(
            split_payload(
                "train",
                train_features,
                train_targets,
                train_ds.classes,
                hierarchy.gen_ds2node_map(train_ds.classes)[train_targets],
                config,
                extra=extra_meta,
            ),
            experiment_dir / "features_train.pt",
        )

    if feature_cfg.get("save_val", True):
        val_features, val_targets = collect_features(model, val_loader, device)
        save_artifact(
            split_payload(
                "val",
                val_features,
                val_targets,
                val_ds.classes,
                hierarchy.gen_ds2node_map(val_ds.classes)[val_targets],
                config,
                extra=extra_meta,
            ),
            experiment_dir / "features_val.pt",
        )

    if feature_cfg.get("save_ood", True):
        ood_features, ood_targets = collect_features(model, ood_loader, device)
        save_artifact(
            split_payload(
                "ood",
                ood_features,
                ood_targets,
                ood_ds.classes,
                hierarchy.gen_ds2node_map(ood_ds.classes)[ood_targets],
                config,
                extra=extra_meta,
            ),
            experiment_dir / "features_ood.pt",
        )

    for farood_name in dataset_cfg.get("farood_sets", []) or []:
        if not feature_cfg.get("save_farood", True):
            break
        farood_ds = gen_custom_dataset(
            dataset_cfg["datadir"],
            farood_name,
            hierarchy.ood_train_classes,
            **transform_kwargs,
        )
        farood_loader = DataLoader(farood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        farood_features, farood_targets = collect_features(model, farood_loader, device)
        root_idx = hierarchy.id_node_list.index("root")
        root_targets = torch.full((farood_targets.shape[0],), root_idx, dtype=torch.long)
        save_artifact(
            split_payload(
                farood_name,
                farood_features,
                farood_targets,
                farood_ds.classes,
                root_targets,
                config,
                extra=extra_meta,
            ),
            experiment_dir / f"features_farood_{farood_name.replace('/', '-')}.pt",
        )


if __name__ == "__main__":
    main()
