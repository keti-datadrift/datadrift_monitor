from __future__ import annotations

import argparse
import hashlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ProHOC.libs.utils.score_util import entcompprob
from negzerohoc.cf_rpep import (
    fold_weight_identity,
    fit_shared_route_scalars,
    parent_descendant_mass,
    route_preserving_terminal,
    streaming_file_identity,
)
from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.crossfit_class_holdout import (
    RemappedSubset,
    canonical_hash,
    tensor_partitions_hash,
    validate_topology_holdout_manifest,
)
from negzerohoc.evaluation import (
    build_hierarchy,
    get_results,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hierarchical_support import (
    expected_hierarchy_distance_predictions,
)
from negzerohoc.multidepth_fusion import (
    get_multidepth_classes,
    multidepth_targets,
    multidepth_unknown_probabilities,
)
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from scripts.evaluate_dual_expert_support import (
    load_frozen_vision,
    make_eval_loader,
    release_cuda,
)
from scripts.train_crossfit_class_holdout_lora import (
    CHECKPOINT_STAGE as FOLD_CHECKPOINT_STAGE,
    build_fold_metric_topology,
)
from scripts.train_idea3_joint_vision_lora import build_transforms
from scripts.train_idea4_unknown_prompts import encode_dataset_features
from scripts.train_multidepth_feature_heads import (
    MultiDepthLinearHeads,
    payload_logits,
    probability_list,
)
from scripts.train_paper_negprompt_ablation import json_ready


STAGE = "crossfit_route_preserving_entcomp_oof"
CROSSFIT_LEVEL = "non_nested_screening"
CROSSFIT_LIMITATION = (
    "Source-fold model weights can have been trained on the target fold's "
    "classes and images. The exclusion audit removes target identities only "
    "from scalar-calibration episodes; it does not make source model weights "
    "target-disjoint. A nested cross-fit or an independent meta-calibration "
    "set is required before any confirmatory gate may unlock official OOD."
)
REQUIRED_CONFIRMATORY_CALIBRATION = (
    "nested_crossfit_or_independent_meta_calibration"
)


def screening_audit_metadata() -> dict:
    return {
        "crossfit_level": CROSSFIT_LEVEL,
        "strict_confirmatory_gate": False,
        "may_unlock_official_ood": False,
        "crossfit_limitation": CROSSFIT_LIMITATION,
        "required_confirmatory_calibration": (
            REQUIRED_CONFIRMATORY_CALIBRATION
        ),
    }


def nested_confirmatory_metadata() -> dict:
    return {
        "crossfit_level": "nested_class_and_image_crossfit",
        "strict_confirmatory_gate": True,
        "may_unlock_official_ood": True,
        "inner_fold_count": 3,
        "scalar_parameter_scope": "two_global_scalars_a_b",
        "actual_ood_still_excluded_from_this_stage": True,
    }


def fold_query_timing_metadata() -> dict:
    return {
        "query_encoded_after_this_fold_model_selection": True,
    }


class OriginalIndexedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [
            int(value)
            for value in torch.as_tensor(
                indices, dtype=torch.long
            ).flatten().tolist()
        ]
        self.classes = list(dataset.classes)
        self.targets = [
            int(dataset.targets[index]) for index in self.indices
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment = cfg.get("experiment", {})
    runtime = cfg.get("runtime", {})
    dataset = cfg.get("dataset", {})
    clip = cfg.get("clip", {})
    dataloader = cfg.get("dataloader", {})
    stage = cfg.get("cf_rpep", {})
    heads = stage.get("heads", {})
    temperature = stage.get("temperature", {})
    experiment_name = str(
        experiment.get("name", "cf-rpep-oof")
    )
    output_root = Path(experiment.get("output_root", "outputs"))
    checkpoints = [
        str(value) for value in stage.get("fold_checkpoints", [])
    ]
    if len(checkpoints) != 4:
        raise ValueError("cf_rpep.fold_checkpoints must contain four paths")

    def artifact(configured, kind, filename):
        return str(resolve_experiment_artifact(
            configured,
            output_root=output_root,
            experiment_name=experiment_name,
            kind=kind,
            default_filename=filename,
        ))

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        dataset=dataset.get("name", "fgvc-aircraft"),
        datadir=str(dataset.get("datadir", "")),
        hierarchy=dataset.get(
            "hierarchy", "hierarchies/fgvc-aircraft.json"
        ),
        id_split=dataset.get(
            "id_split", "data/fgvc-aircraft-id-labels.csv"
        ),
        clip_model=clip.get(
            "model", "openai/clip-vit-base-patch16"
        ),
        tokenizer_model=clip.get(
            "tokenizer_model",
            clip.get("model", "openai/clip-vit-base-patch16"),
        ),
        local_files_only=bool(clip.get("local_files_only", True)),
        augmentation=cfg.get("augmentation", {}),
        num_workers=int(dataloader.get("num_workers", 0)),
        device=configured_device(runtime),
        seed=int(runtime.get("seed", 0)),
        deterministic=bool(runtime.get("deterministic", True)),
        precision=str(stage.get("precision", "fp16")).lower(),
        inference_batch_size=max(
            1, int(stage.get("inference_batch_size", 128))
        ),
        support_lora_enabled=True,
        fold_checkpoints=checkpoints,
        head_epochs=max(1, int(heads.get("epochs", 200))),
        head_batch_size=max(1, int(heads.get("batch_size", 512))),
        head_lr=float(heads.get("lr", 0.01)),
        head_weight_decay=float(
            heads.get("weight_decay", 1e-4)
        ),
        head_patience=max(1, int(heads.get("patience", 30))),
        temperature_max_iter=max(
            1, int(temperature.get("max_iter", 100))
        ),
        scalar_max_iter=100,
        checkpoint_path=artifact(
            stage.get("checkpoint"),
            "checkpoints",
            f"{experiment_name}.pt",
        ),
        result_path=artifact(
            stage.get("result_path"),
            "results",
            f"{experiment_name}.result",
        ),
        diagnostics_path=artifact(
            stage.get("diagnostics_path"),
            "diagnostics",
            f"{experiment_name}.json",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def build_eval_train_dataset(args):
    from negzerohoc.prohoc_compat.utils.dataset_util import (
        SubsetImageFolder,
        get_id_classes,
    )

    _, transform = build_transforms(args)
    return SubsetImageFolder(
        Path(args.datadir) / "train",
        get_id_classes(args.id_split),
        transform=transform,
    )


def load_fold_checkpoint_with_identity(path: str | Path):
    before = streaming_file_identity(path)
    checkpoint = load_idea3_checkpoint(path, map_location="cpu")
    after = streaming_file_identity(path)
    if before != after:
        raise RuntimeError(
            f"Fold checkpoint changed while loading: {before['canonical_path']}"
        )
    return checkpoint, {
        "file": after,
        "weights": fold_weight_identity(checkpoint),
    }


def verify_fold_checkpoint_identities(
    folds: dict[int, dict],
    identities: dict[int, dict],
) -> None:
    if set(folds) != set(identities):
        raise ValueError("Fold checkpoint identity coverage is incomplete")
    for fold in sorted(folds):
        current_file = streaming_file_identity(folds[fold]["path"])
        if current_file != identities[fold]["file"]:
            raise RuntimeError(
                f"Fold {fold} best checkpoint was overwritten during CF-RPEP"
            )
        current_weights = fold_weight_identity(
            folds[fold]["checkpoint"]
        )
        if current_weights != identities[fold]["weights"]:
            raise RuntimeError(
                f"Fold {fold} loaded LoRA/proxy state changed in memory"
            )


def _validate_partition_indices(
    checkpoint,
    dataset_targets: torch.Tensor,
    heldout_original_targets: list[int],
) -> dict[str, torch.Tensor]:
    partitions = checkpoint.get("crossfit_split_indices")
    expected_names = {
        "representation_train",
        "model_selection",
        "known_query",
        "heldout_query",
    }
    if not isinstance(partitions, dict) or set(partitions) != expected_names:
        raise ValueError("Fold checkpoint split partitions are incomplete")
    partitions = {
        name: torch.as_tensor(indices, dtype=torch.long).flatten()
        for name, indices in partitions.items()
    }
    if tensor_partitions_hash(partitions) != checkpoint.get(
        "crossfit_split_hash"
    ):
        raise ValueError("Fold checkpoint split hash is invalid")
    combined = torch.cat(list(partitions.values()))
    if bool(
        ((combined < 0) | (combined >= int(dataset_targets.numel()))).any()
    ):
        raise ValueError("Fold checkpoint contains out-of-range image index")
    if int(torch.unique(combined).numel()) != int(combined.numel()):
        raise ValueError("Fold checkpoint image partitions overlap")
    if not torch.equal(
        combined.sort().values,
        torch.arange(int(dataset_targets.numel())),
    ):
        raise ValueError("Fold checkpoint image partitions are incomplete")
    heldout_mask = torch.zeros_like(dataset_targets, dtype=torch.bool)
    for target in heldout_original_targets:
        heldout_mask |= dataset_targets == int(target)
    expected_heldout = torch.nonzero(
        heldout_mask, as_tuple=False
    ).flatten()
    if not torch.equal(
        partitions["heldout_query"].sort().values,
        expected_heldout,
    ):
        raise ValueError("Heldout-query indices do not match heldout classes")
    for name in (
        "representation_train",
        "model_selection",
        "known_query",
    ):
        if bool(heldout_mask[partitions[name]].any()):
            raise ValueError(f"Heldout image leaked into {name}")
    return partitions


def validate_fold_checkpoints(
    args,
    checkpoints: list[dict],
    paths: list[str],
    full_hierarchy,
    dataset,
) -> tuple[dict, dict[int, dict]]:
    if len(checkpoints) != 4 or len(paths) != 4:
        raise ValueError("Exactly four fold checkpoints are required")
    classes = list(dataset.classes)
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    manifests = [checkpoint.get("crossfit_manifest") for checkpoint in checkpoints]
    if any(not isinstance(manifest, dict) for manifest in manifests):
        raise ValueError("Fold checkpoint is missing its manifest")
    manifest = manifests[0]
    validate_topology_holdout_manifest(
        full_hierarchy, classes, manifest
    )
    if any(value != manifest for value in manifests[1:]):
        raise ValueError("Fold checkpoints do not share one exact manifest")
    records = {}
    for checkpoint, path in zip(checkpoints, paths):
        if Path(path).name != "best.pt":
            raise ValueError("CF-RPEP accepts only fold best.pt checkpoints")
        if checkpoint.get("stage") != FOLD_CHECKPOINT_STAGE:
            raise ValueError("Unexpected fold checkpoint stage")
        expected_header = {
            "dataset": args.dataset,
            "clip_model": args.clip_model,
            "hierarchy": args.hierarchy,
            "id_split": args.id_split,
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected_header.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Fold checkpoint header mismatch: {mismatches}")
        if checkpoint.get("crossfit_manifest_hash") != manifest[
            "manifest_hash"
        ]:
            raise ValueError("Fold checkpoint manifest hash mismatch")
        fold_record = checkpoint.get("crossfit_fold")
        fold = int((fold_record or {}).get("fold", -1))
        if fold < 0 or fold >= 4 or fold in records:
            raise ValueError("Fold checkpoint IDs are missing or duplicated")
        if fold_record != manifest["folds"][fold]:
            raise ValueError("Fold checkpoint record differs from manifest")
        heldout_targets = fold_record[
            "heldout_original_class_indices"
        ]
        heldout_set = set(heldout_targets)
        retained_targets = [
            index for index in range(len(classes))
            if index not in heldout_set
        ]
        retained_classes = [classes[index] for index in retained_targets]
        if checkpoint.get("metric_proxy_classes") != retained_classes:
            raise ValueError("Fold metric proxy class ordering is invalid")
        proxies = checkpoint.get("metric_proxies")
        if (
            not isinstance(proxies, torch.Tensor)
            or int(proxies.shape[0]) != len(retained_classes)
        ):
            raise ValueError("Fold metric proxy tensor is invalid")
        if not checkpoint.get("vision_lora_state_dict"):
            raise ValueError("Fold checkpoint is missing LoRA state")
        if checkpoint.get("training_state") is not None:
            raise ValueError("Fold best checkpoint must be finalized/compact")
        partitions = _validate_partition_indices(
            checkpoint, targets, heldout_targets
        )
        metrics = checkpoint.get("metrics") or {}
        required_false = (
            "used_heldout_class_images_for_representation_training",
            "used_heldout_class_images_for_proxy_initialization",
            "used_heldout_class_images_for_model_selection",
            "used_known_query_for_training_or_selection",
            "used_official_test_for_training_or_selection",
        )
        if any(metrics.get(key) is not False for key in required_false):
            raise ValueError("Fold checkpoint is not finalized leakage-safe")
        if metrics.get("split_hash") != checkpoint.get(
            "crossfit_split_hash"
        ):
            raise ValueError("Fold metrics/split hash disagree")
        _, _, expected_provenance = build_fold_metric_topology(
            args, full_hierarchy, retained_classes
        )
        if checkpoint.get(
            "crossfit_hierarchy_provenance"
        ) != expected_provenance:
            raise ValueError("Fold hierarchy provenance is invalid")
        records[fold] = {
            "path": path,
            "checkpoint": checkpoint,
            "fold_record": fold_record,
            "partitions": partitions,
            "retained_targets": retained_targets,
            "retained_classes": retained_classes,
            "hierarchy_provenance": expected_provenance,
        }
    if set(records) != {0, 1, 2, 3}:
        raise ValueError("Fold checkpoints do not cover folds 0..3")
    return manifest, records


@torch.no_grad()
def split_nll(model, features, targets_by_depth, device) -> float:
    model.eval()
    logits = model(features.float().to(device))
    return float(sum(
        F.cross_entropy(value, target.long().to(device))
        for value, target in zip(logits, targets_by_depth)
    ).cpu())


def train_heads(
    args,
    train_features,
    train_targets,
    selection_features,
    selection_targets,
    class_counts,
    *,
    fold,
    device,
):
    model = MultiDepthLinearHeads(
        int(train_features.shape[1]), class_counts
    ).to(device)
    train_features = train_features.float().to(device)
    train_targets = [
        value.long().to(device) for value in train_targets
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.head_lr,
        weight_decay=args.head_weight_decay,
    )
    generator = torch.Generator().manual_seed(
        int(args.seed) + 1_000_003 * int(fold)
    )
    best_nll = float("inf")
    best_epoch = None
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        permutation = torch.randperm(
            int(train_features.shape[0]), generator=generator
        ).to(device)
        running = 0.0
        steps = 0
        for start in range(
            0, int(permutation.numel()), args.head_batch_size
        ):
            index = permutation[start:start + args.head_batch_size]
            logits = model(train_features.index_select(0, index))
            loss = sum(
                F.cross_entropy(
                    value, target.index_select(0, index)
                )
                for value, target in zip(logits, train_targets)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
        selection_nll = split_nll(
            model,
            selection_features,
            selection_targets,
            device,
        )
        if selection_nll < best_nll - 1e-7:
            best_nll = selection_nll
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            history.append({
                "epoch": epoch,
                "train_nll": running / max(1, steps),
                "model_selection_nll": selection_nll,
            })
        if stale >= args.head_patience:
            break
    if best_state is None:
        raise RuntimeError("CF-RPEP multi-depth head produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_model_selection_nll": best_nll,
        "completed_epochs": epoch,
        "history": history,
    }


def fit_temperatures(
    logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    max_iter: int,
) -> tuple[torch.Tensor, dict]:
    logits = [value.double() for value in logits]
    targets = [value.long() for value in targets]
    log_temperature = nn.Parameter(
        torch.zeros(len(logits), dtype=torch.float64)
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=1.0,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def objective(backward):
        temperatures = log_temperature.exp().clamp(0.05, 20.0)
        loss = sum(
            F.cross_entropy(
                value / temperatures[depth], targets[depth]
            )
            for depth, value in enumerate(logits)
        )
        if backward:
            loss.backward()
        return loss

    initial = float(objective(False).detach())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        return objective(True)

    optimizer.step(closure)
    temperatures = log_temperature.detach().exp().clamp(0.05, 20.0)
    return temperatures.float(), {
        "initial_model_selection_nll": initial,
        "final_model_selection_nll": float(objective(False).detach()),
        "temperatures": temperatures.tolist(),
        "fit_split": "model_selection",
    }


def proxy_logits(features, proxies):
    return (
        F.normalize(features.float(), dim=-1)
        @ F.normalize(proxies.float(), dim=-1).t()
    )


def encode_subset(args, clip_model, dataset, device, description):
    return encode_dataset_features(
        args,
        clip_model,
        dataset,
        make_eval_loader(args, dataset),
        device,
        description,
    )


def make_oof_bundle(
    full_hierarchy,
    fold_hierarchy,
    fold_record,
    retained_classes,
    proxies,
    leaf_temperature,
    heads,
    head_temperatures,
    known_payload,
    heldout_payload,
    known_original_indices,
    heldout_original_indices,
    device,
):
    full_node_to_index = {
        node: index
        for index, node in enumerate(full_hierarchy.id_node_list)
    }
    parent_nodes = [
        node for node in full_hierarchy.id_node_list
        if node != "root" and node in fold_hierarchy.parent2children
    ]
    multidepth_classes = get_multidepth_classes(
        fold_hierarchy, retained_classes
    )

    def evidence(payload):
        leaf = F.softmax(
            proxy_logits(payload["features"], proxies)
            / float(leaf_temperature),
            dim=1,
        )
        logits = payload_logits(heads, payload, device)
        probabilities = probability_list(logits, head_temperatures)
        unknown_by_parent = multidepth_unknown_probabilities(
            probabilities,
            fold_hierarchy,
            multidepth_classes,
            entcompprob,
        )
        unknown = torch.stack([
            unknown_by_parent[parent] for parent in parent_nodes
        ], dim=1)
        mass = parent_descendant_mass(
            leaf,
            fold_hierarchy,
            retained_classes,
            parent_nodes,
        )
        return leaf, mass, unknown

    known_leaf, known_mass, known_unknown = evidence(known_payload)
    pseudo_leaf, pseudo_mass, pseudo_unknown = evidence(heldout_payload)
    known_class_names = [
        retained_classes[int(target)]
        for target in known_payload["targets"].tolist()
    ]
    heldout_class_names = [
        heldout_payload["classes"][int(target)]
        for target in heldout_payload["targets"].tolist()
    ]
    known_targets = torch.tensor([
        full_node_to_index[name] for name in known_class_names
    ], dtype=torch.long)
    mapped = fold_record["mapped_unknown_nodes"]
    pseudo_target_names = [mapped[name] for name in heldout_class_names]
    pseudo_targets = torch.tensor([
        full_node_to_index[name] for name in pseudo_target_names
    ], dtype=torch.long)
    leaf_indices = torch.tensor([
        full_node_to_index[name] for name in retained_classes
    ], dtype=torch.long)
    parent_indices = torch.tensor([
        full_node_to_index[name] for name in parent_nodes
    ], dtype=torch.long)
    return {
        "leaf_probabilities": torch.cat(
            [known_leaf, pseudo_leaf], dim=0
        ),
        "parent_mass": torch.cat(
            [known_mass, pseudo_mass], dim=0
        ),
        "entcomp_unknown": torch.cat(
            [known_unknown, pseudo_unknown], dim=0
        ),
        "leaf_node_indices": leaf_indices,
        "parent_node_indices": parent_indices,
        "node_count": len(full_hierarchy.id_node_list),
        "target_node_indices": torch.cat(
            [known_targets, pseudo_targets]
        ),
        "kinds": (
            ["known"] * len(known_class_names)
            + ["pseudo"] * len(heldout_class_names)
        ),
        "target_groups": known_class_names + pseudo_target_names,
        "class_names": known_class_names + heldout_class_names,
        "original_indices": (
            [int(value) for value in known_original_indices.tolist()]
            + [int(value) for value in heldout_original_indices.tolist()]
        ),
        "known_count": len(known_class_names),
        "pseudo_count": len(heldout_class_names),
        "parent_nodes": parent_nodes,
        "retained_classes": retained_classes,
    }


def subset_bundle(bundle, keep: torch.Tensor) -> dict:
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    result = {
        key: value.index_select(0, indices)
        for key, value in bundle.items()
        if isinstance(value, torch.Tensor)
        and int(value.ndim) > 0
        and int(value.shape[0]) == len(bundle["kinds"])
    }
    result.update({
        "leaf_node_indices": bundle["leaf_node_indices"],
        "parent_node_indices": bundle["parent_node_indices"],
        "node_count": bundle["node_count"],
        "kinds": [
            bundle["kinds"][int(index)] for index in indices
        ],
        "target_groups": [
            bundle["target_groups"][int(index)] for index in indices
        ],
    })
    for key in ("retained_classes", "parent_nodes"):
        if key in bundle:
            result[key] = bundle[key]
    return result


def calibration_bundles_for_target(
    target_fold: int,
    bundles: dict[int, dict],
    manifest: dict,
) -> tuple[list[dict], dict]:
    target_classes = set(
        manifest["folds"][target_fold]["heldout_leaves"]
    )
    target_query_indices = set(
        int(value) for value in bundles[target_fold]["original_indices"]
    )
    selected = []
    excluded = []
    included_count = 0
    for source_fold in sorted(bundles):
        if source_fold == target_fold:
            continue
        bundle = bundles[source_fold]
        keep_values = []
        for index, class_name in enumerate(bundle["class_names"]):
            original_index = int(bundle["original_indices"][index])
            reasons = []
            if class_name in target_classes:
                reasons.append("target_heldout_class_identity")
            if original_index in target_query_indices:
                reasons.append("target_fold_query_image_identity")
            keep = not reasons
            keep_values.append(keep)
            if keep:
                included_count += 1
            else:
                excluded.append({
                    "source_fold": source_fold,
                    "kind": bundle["kinds"][index],
                    "class_name": class_name,
                    "original_index": original_index,
                    "reasons": reasons,
                })
        selected.append(subset_bundle(
            bundle, torch.tensor(keep_values, dtype=torch.bool)
        ))
    excluded = sorted(
        excluded,
        key=lambda row: (
            row["source_fold"],
            row["original_index"],
            row["class_name"],
            row["kind"],
        ),
    )
    return selected, {
        "scope": "episode_level_only",
        "weight_level_target_disjoint": False,
        "target_fold": int(target_fold),
        "target_heldout_classes": sorted(target_classes),
        "target_query_index_count": len(target_query_indices),
        "target_query_indices_hash": hashlib.sha256(
            torch.tensor(
                sorted(target_query_indices), dtype=torch.long
            ).numpy().tobytes()
        ).hexdigest(),
        "source_folds": [
            fold for fold in sorted(bundles) if fold != target_fold
        ],
        "included_episode_count": included_count,
        "excluded_episode_count": len(excluded),
        "excluded_known_count": sum(
            row["kind"] == "known" for row in excluded
        ),
        "excluded_pseudo_count": sum(
            row["kind"] == "pseudo" for row in excluded
        ),
        "excluded_for_target_class_identity_count": sum(
            "target_heldout_class_identity" in row["reasons"]
            for row in excluded
        ),
        "excluded_for_target_query_image_identity_count": sum(
            "target_fold_query_image_identity" in row["reasons"]
            for row in excluded
        ),
        "excluded_class_counts": {
            name: sum(row["class_name"] == name for row in excluded)
            for name in sorted({row["class_name"] for row in excluded})
        },
        "excluded_records_hash": canonical_hash({
            "target_fold": int(target_fold),
            "records": excluded,
        }),
    }


def evaluate_target_fold(
    full_hierarchy,
    bundle,
    scalar_fit,
    distance_matrix,
    dists_mats,
    *,
    decoder: str = "expected_hdist",
):
    terminal = route_preserving_terminal(
        bundle["leaf_probabilities"],
        bundle["parent_mass"],
        bundle["entcomp_unknown"],
        leaf_node_indices=bundle["leaf_node_indices"],
        parent_node_indices=bundle["parent_node_indices"],
        node_count=bundle["node_count"],
        a=scalar_fit["a"],
        b=scalar_fit["b"],
    ).float()
    known_count = int(bundle["known_count"])
    known_terminal = terminal[:known_count]
    pseudo_terminal = terminal[known_count:]
    known_targets = bundle["target_node_indices"][:known_count]
    pseudo_targets = bundle["target_node_indices"][known_count:]
    leaf_winner = bundle["leaf_probabilities"][:known_count].argmax(
        dim=1
    )
    classifier_predictions = bundle["leaf_node_indices"].index_select(
        0, leaf_winner
    )
    if decoder == "expected_hdist":
        posterior_predictions = expected_hierarchy_distance_predictions(
            known_terminal, distance_matrix
        )
        pseudo_predictions = expected_hierarchy_distance_predictions(
            pseudo_terminal, distance_matrix
        )
    elif decoder == "map":
        posterior_predictions = known_terminal.argmax(dim=1)
        pseudo_predictions = pseudo_terminal.argmax(dim=1)
    else:
        raise ValueError(f"Unsupported CF-RPEP decoder: {decoder!r}")
    classifier_metrics = get_results(
        classifier_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        posterior_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    internal_indices = torch.tensor([
        index
        for index, node in enumerate(full_hierarchy.id_node_list)
        if node != "root" and node in full_hierarchy.parent2children
    ])
    return {
        "terminal": terminal,
        "known_terminal": known_terminal,
        "pseudo_terminal": pseudo_terminal,
        "known_targets": known_targets,
        "pseudo_targets": pseudo_targets,
        "classifier_predictions": classifier_predictions,
        "known_predictions": posterior_predictions,
        "pseudo_predictions": pseudo_predictions,
        "known_unknown_mass": known_terminal.index_select(
            1, internal_indices
        ).sum(dim=1),
        "pseudo_unknown_mass": pseudo_terminal.index_select(
            1, internal_indices
        ).sum(dim=1),
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": float(
            (terminal.sum(dim=1) - 1.0).abs().max()
        ),
        "decoder": decoder,
    }


def nested_inner_assignments(
    bundle,
    hierarchy,
    *,
    num_inner_folds: int = 3,
) -> tuple[torch.Tensor, dict]:
    """Assign disjoint known images and pseudo classes to inner folds."""
    if int(num_inner_folds) != 3:
        raise ValueError("CF-RPEP confirmatory protocol requires 3 inner folds")
    count = len(bundle["kinds"])
    if not (
        len(bundle["class_names"])
        == len(bundle["target_groups"])
        == len(bundle["original_indices"])
        == count
    ):
        raise ValueError("CF-RPEP episode metadata is misaligned")
    assignments = torch.full((count,), -1, dtype=torch.long)

    known_by_class = {}
    pseudo_parent_by_class = {}
    pseudo_indices_by_class = {}
    for index, kind in enumerate(bundle["kinds"]):
        class_name = bundle["class_names"][index]
        if kind == "known":
            known_by_class.setdefault(class_name, []).append(index)
        elif kind == "pseudo":
            parent = bundle["target_groups"][index]
            previous = pseudo_parent_by_class.setdefault(class_name, parent)
            if previous != parent:
                raise ValueError(
                    f"Pseudo class {class_name!r} maps to multiple parents"
                )
            pseudo_indices_by_class.setdefault(class_name, []).append(index)
        else:
            raise ValueError(f"Unsupported CF-RPEP episode kind: {kind!r}")
    if len(pseudo_indices_by_class) < int(num_inner_folds):
        raise ValueError("Each outer fold needs at least three pseudo classes")

    # Every known class contributes image-disjoint calibration/evaluation
    # samples to all three inner folds.
    for class_name in sorted(known_by_class):
        ordered = sorted(
            known_by_class[class_name],
            key=lambda index: (
                int(bundle["original_indices"][index]),
                index,
            ),
        )
        for rank, index in enumerate(ordered):
            assignments[index] = rank % int(num_inner_folds)

    pseudo_classes = sorted(pseudo_indices_by_class)
    capacities = [
        len(pseudo_classes) // int(num_inner_folds)
        + (
            1
            if fold < len(pseudo_classes) % int(num_inner_folds)
            else 0
        )
        for fold in range(int(num_inner_folds))
    ]
    totals = [0] * int(num_inner_folds)
    parent_counts = [dict() for _ in range(int(num_inner_folds))]
    depth_counts = [dict() for _ in range(int(num_inner_folds))]

    def pseudo_sort_key(class_name):
        parent = pseudo_parent_by_class[class_name]
        depth = len(hierarchy.node_ancestors[parent])
        # Depth-2 classes are assigned first. Each audited outer fold has
        # exactly three, so the greedy depth count places one in each group.
        return (-depth, parent, class_name)

    pseudo_group = {}
    for class_name in sorted(pseudo_classes, key=pseudo_sort_key):
        parent = pseudo_parent_by_class[class_name]
        depth = len(hierarchy.node_ancestors[parent])
        available = [
            fold for fold in range(int(num_inner_folds))
            if totals[fold] < capacities[fold]
        ]
        if not available:
            raise RuntimeError("Nested pseudo-class fold capacity overflow")
        chosen = min(
            available,
            key=lambda fold: (
                parent_counts[fold].get(parent, 0),
                depth_counts[fold].get(depth, 0),
                totals[fold],
                fold,
            ),
        )
        pseudo_group[class_name] = chosen
        totals[chosen] += 1
        parent_counts[chosen][parent] = (
            parent_counts[chosen].get(parent, 0) + 1
        )
        depth_counts[chosen][depth] = (
            depth_counts[chosen].get(depth, 0) + 1
        )
        for index in pseudo_indices_by_class[class_name]:
            assignments[index] = chosen

    if bool((assignments < 0).any()):
        raise RuntimeError("Nested CF-RPEP left episodes unassigned")
    if totals != capacities:
        raise RuntimeError("Nested pseudo-class folds are not size-balanced")
    audit = {
        "num_inner_folds": int(num_inner_folds),
        "pseudo_class_capacities": capacities,
        "pseudo_class_counts": totals,
        "pseudo_class_groups": {
            str(fold): sorted(
                class_name
                for class_name, assigned in pseudo_group.items()
                if assigned == fold
            )
            for fold in range(int(num_inner_folds))
        },
        "mapped_parent_counts": {
            str(fold): parent_counts[fold]
            for fold in range(int(num_inner_folds))
        },
        "mapped_parent_depth_counts": {
            str(fold): {
                str(depth): value
                for depth, value in depth_counts[fold].items()
            }
            for fold in range(int(num_inner_folds))
        },
        "known_image_counts": {
            str(fold): int(sum(
                bundle["kinds"][index] == "known"
                and int(assignments[index]) == fold
                for index in range(count)
            ))
            for fold in range(int(num_inner_folds))
        },
        "assignment_hash": hashlib.sha256(
            assignments.numpy().tobytes()
        ).hexdigest(),
    }
    return assignments, audit


def ordered_evaluation_subset(bundle, keep: torch.Tensor) -> dict:
    """Subset episodes while retaining the known-then-pseudo convention."""
    keep = torch.as_tensor(keep, dtype=torch.bool).flatten()
    if int(keep.numel()) != len(bundle["kinds"]):
        raise ValueError("Nested evaluation mask has the wrong length")
    known_indices = [
        index for index, kind in enumerate(bundle["kinds"])
        if kind == "known" and bool(keep[index])
    ]
    pseudo_indices = [
        index for index, kind in enumerate(bundle["kinds"])
        if kind == "pseudo" and bool(keep[index])
    ]
    if not known_indices or not pseudo_indices:
        raise ValueError(
            "Every nested evaluation fold needs known and pseudo episodes"
        )
    indices = torch.tensor(
        known_indices + pseudo_indices, dtype=torch.long
    )
    episode_count = len(bundle["kinds"])
    result = {
        key: value.index_select(0, indices)
        for key, value in bundle.items()
        if isinstance(value, torch.Tensor)
        and int(value.ndim) > 0
        and int(value.shape[0]) == episode_count
    }
    result.update({
        "leaf_node_indices": bundle["leaf_node_indices"],
        "parent_node_indices": bundle["parent_node_indices"],
        "node_count": bundle["node_count"],
        "kinds": [
            bundle["kinds"][int(index)] for index in indices
        ],
        "target_groups": [
            bundle["target_groups"][int(index)] for index in indices
        ],
        "class_names": [
            bundle["class_names"][int(index)] for index in indices
        ],
        "original_indices": [
            bundle["original_indices"][int(index)] for index in indices
        ],
        "known_count": len(known_indices),
        "pseudo_count": len(pseudo_indices),
    })
    for key in ("retained_classes", "parent_nodes"):
        if key in bundle:
            result[key] = bundle[key]
    return result


def evaluate_nested_target_fold(
    full_hierarchy,
    bundle,
    distance_matrix,
    dists_mats,
    *,
    max_iter: int,
    decoder: str = "expected_hdist",
) -> dict:
    assignments, assignment_audit = nested_inner_assignments(
        bundle, full_hierarchy
    )
    inner_results = []
    scalar_fits = []
    seen_eval_indices = []
    for inner_fold in range(3):
        evaluation_mask = assignments == inner_fold
        calibration_mask = ~evaluation_mask
        calibration = subset_bundle(bundle, calibration_mask)
        scalar_fit = fit_shared_route_scalars(
            [calibration], max_iter=max_iter
        )
        evaluation_bundle = ordered_evaluation_subset(
            bundle, evaluation_mask
        )
        evaluation = evaluate_target_fold(
            full_hierarchy,
            evaluation_bundle,
            scalar_fit,
            distance_matrix,
            dists_mats,
            decoder=decoder,
        )
        evaluation["scalar_fit"] = scalar_fit
        evaluation["inner_fold"] = inner_fold
        evaluation["calibration_episode_count"] = int(
            calibration_mask.sum()
        )
        evaluation["evaluation_episode_count"] = int(
            evaluation_mask.sum()
        )
        inner_results.append(evaluation)
        scalar_fits.append(scalar_fit)
        seen_eval_indices.extend(
            torch.nonzero(
                evaluation_mask, as_tuple=False
            ).flatten().tolist()
        )
    if sorted(seen_eval_indices) != list(range(len(bundle["kinds"]))):
        raise RuntimeError(
            "Nested CF-RPEP did not evaluate every episode exactly once"
        )

    classifier_targets = torch.cat([
        value["known_targets"] for value in inner_results
    ])
    classifier_predictions = torch.cat([
        value["classifier_predictions"] for value in inner_results
    ])
    known_predictions = torch.cat([
        value["known_predictions"] for value in inner_results
    ])
    pseudo_targets = torch.cat([
        value["pseudo_targets"] for value in inner_results
    ])
    pseudo_predictions = torch.cat([
        value["pseudo_predictions"] for value in inner_results
    ])
    classifier_metrics = get_results(
        classifier_predictions,
        classifier_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        classifier_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    return {
        "classifier_predictions": classifier_predictions,
        "known_predictions": known_predictions,
        "pseudo_predictions": pseudo_predictions,
        "known_targets": classifier_targets,
        "pseudo_targets": pseudo_targets,
        "known_unknown_mass": torch.cat([
            value["known_unknown_mass"] for value in inner_results
        ]),
        "pseudo_unknown_mass": torch.cat([
            value["pseudo_unknown_mass"] for value in inner_results
        ]),
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": max(
            value["normalization_error"] for value in inner_results
        ),
        "inner_scalar_fits": scalar_fits,
        "assignment_audit": assignment_audit,
        "target_disjoint_audit": {
            "encoder_excludes_all_pseudo_classes": True,
            "scalar_fit_excludes_evaluated_pseudo_classes": True,
            "scalar_fit_excludes_evaluated_known_images": True,
            "every_episode_evaluated_exactly_once": True,
        },
        "decoder": decoder,
    }


def aggregate_evaluation_summary(
    evaluations: dict[int, dict],
    hierarchy,
    dists_mats,
) -> dict:
    folds = sorted(evaluations)
    known_targets = torch.cat([
        evaluations[fold]["known_targets"] for fold in folds
    ])
    classifier_metrics = get_results(
        torch.cat([
            evaluations[fold]["classifier_predictions"]
            for fold in folds
        ]),
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        torch.cat([
            evaluations[fold]["known_predictions"] for fold in folds
        ]),
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_targets = torch.cat([
        evaluations[fold]["pseudo_targets"] for fold in folds
    ])
    pseudo_metrics = get_results(
        torch.cat([
            evaluations[fold]["pseudo_predictions"] for fold in folds
        ]),
        pseudo_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    unknown_binary = binary_ood_metrics(
        torch.cat([
            evaluations[fold]["known_unknown_mass"] for fold in folds
        ]).numpy(),
        torch.cat([
            evaluations[fold]["pseudo_unknown_mass"] for fold in folds
        ]).numpy(),
    )
    return {
        "classifier_known": classifier_metrics,
        "posterior_known": known_metrics,
        "pseudo_mapped_parent": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "unknown_mass_binary": unknown_binary,
        "normalization_error": max(
            evaluations[fold]["normalization_error"] for fold in folds
        ),
        "per_fold": {
            str(fold): {
                "classifier_known": evaluations[fold][
                    "classifier_metrics"
                ],
                "posterior_known": evaluations[fold]["known_metrics"],
                "pseudo_mapped_parent": evaluations[fold][
                    "pseudo_metrics"
                ],
                "mixed": evaluations[fold]["mixed"],
            }
            for fold in folds
        },
    }


def atomic_torch_save(payload, path: Path):
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    if not args.datadir:
        raise ValueError("Missing dataset.datadir")
    if args.num_workers != 0:
        raise ValueError("CF-RPEP deterministic encoding requires num_workers=0")
    configure_reproducibility(
        args.seed, deterministic=args.deterministic
    )
    device = available_device(args.device)
    full_hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    dataset = build_eval_train_dataset(args)
    loaded_pairs = [
        load_fold_checkpoint_with_identity(path)
        for path in args.fold_checkpoints
    ]
    loaded = [checkpoint for checkpoint, _ in loaded_pairs]
    manifest, folds = validate_fold_checkpoints(
        args,
        loaded,
        args.fold_checkpoints,
        full_hierarchy,
        dataset,
    )
    input_checkpoint_identities = {}
    for checkpoint, (_, identity) in zip(loaded, loaded_pairs):
        fold = int(checkpoint["crossfit_fold"]["fold"])
        input_checkpoint_identities[fold] = identity

    bundles = {}
    fold_models = {}
    for fold in range(4):
        configure_reproducibility(
            args.seed + 1_000_003 * fold,
            deterministic=args.deterministic,
        )
        record = folds[fold]
        checkpoint = record["checkpoint"]
        partitions = record["partitions"]
        retained_classes = record["retained_classes"]
        original_to_compact = {
            original: compact
            for compact, original in enumerate(
                record["retained_targets"]
            )
        }
        representation_dataset = RemappedSubset(
            dataset,
            partitions["representation_train"],
            original_to_compact,
            retained_classes,
        )
        selection_dataset = RemappedSubset(
            dataset,
            partitions["model_selection"],
            original_to_compact,
            retained_classes,
        )
        known_dataset = RemappedSubset(
            dataset,
            partitions["known_query"],
            original_to_compact,
            retained_classes,
        )
        heldout_dataset = OriginalIndexedSubset(
            dataset, partitions["heldout_query"]
        )
        args.support_checkpoint = record["path"]
        _, clip_model = load_frozen_vision(
            args, record["path"], device
        )
        representation_payload = encode_subset(
            args,
            clip_model,
            representation_dataset,
            device,
            f"CF-RPEP fold {fold} representation_train",
        )
        selection_payload = encode_subset(
            args,
            clip_model,
            selection_dataset,
            device,
            f"CF-RPEP fold {fold} model_selection",
        )
        fold_hierarchy, _, _ = build_fold_metric_topology(
            args, full_hierarchy, retained_classes
        )
        multidepth_classes = get_multidepth_classes(
            fold_hierarchy, retained_classes
        )
        representation_targets = multidepth_targets(
            fold_hierarchy,
            retained_classes,
            representation_payload["targets"],
            multidepth_classes,
        )
        selection_targets = multidepth_targets(
            fold_hierarchy,
            retained_classes,
            selection_payload["targets"],
            multidepth_classes,
        )
        heads, head_training = train_heads(
            args,
            representation_payload["features"],
            representation_targets,
            selection_payload["features"],
            selection_targets,
            [len(nodes) for nodes in multidepth_classes],
            fold=fold,
            device=device,
        )
        selection_head_logits = payload_logits(
            heads, selection_payload, device
        )
        head_temperatures, head_temperature_fit = fit_temperatures(
            selection_head_logits,
            selection_targets,
            max_iter=args.temperature_max_iter,
        )
        proxies = checkpoint["metric_proxies"].float()
        selection_proxy_logits = proxy_logits(
            selection_payload["features"], proxies
        )
        leaf_temperature, leaf_temperature_fit = fit_temperatures(
            [selection_proxy_logits],
            [selection_payload["targets"]],
            max_iter=args.temperature_max_iter,
        )
        meta_selection_leaf_probabilities = F.softmax(
            selection_proxy_logits / float(leaf_temperature[0]),
            dim=1,
        ).detach().cpu()
        # Query encoding begins only after every representation/head/proxy
        # checkpoint and temperature decision for this fold is frozen.
        known_payload = encode_subset(
            args,
            clip_model,
            known_dataset,
            device,
            f"CF-RPEP fold {fold} known_query final OOF",
        )
        heldout_payload = encode_subset(
            args,
            clip_model,
            heldout_dataset,
            device,
            f"CF-RPEP fold {fold} heldout_query final OOF",
        )
        bundles[fold] = make_oof_bundle(
            full_hierarchy,
            fold_hierarchy,
            record["fold_record"],
            retained_classes,
            proxies,
            float(leaf_temperature[0]),
            heads,
            head_temperatures,
            known_payload,
            heldout_payload,
            partitions["known_query"],
            partitions["heldout_query"],
            device,
        )
        fold_models[fold] = {
            "multidepth_classes": multidepth_classes,
            "head_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in heads.state_dict().items()
            },
            "head_training": head_training,
            "head_temperatures": head_temperatures,
            "head_temperature_fit": head_temperature_fit,
            "leaf_temperature": float(leaf_temperature[0]),
            "leaf_temperature_fit": leaf_temperature_fit,
            "class_loco_meta_selection": {
                "leaf_probabilities": (
                    meta_selection_leaf_probabilities
                ),
                "targets": selection_payload["targets"].detach().cpu(),
                "original_indices": partitions[
                    "model_selection"
                ].detach().cpu(),
                "classes": list(retained_classes),
                "source_split": "model_selection",
                "used_outer_heldout_classes": False,
                "used_known_query": False,
            },
            "head_checkpoint_selection_split": "model_selection",
            "temperature_fit_split": "model_selection",
            **fold_query_timing_metadata(),
        }
        del clip_model, heads
        release_cuda()

    dists_mats = make_distance_mats(full_hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    scalar_fits = {}
    exclusions = {}
    evaluations = {}
    for target_fold in range(4):
        calibration, exclusion = calibration_bundles_for_target(
            target_fold, bundles, manifest
        )
        scalar_fit = fit_shared_route_scalars(
            calibration, max_iter=args.scalar_max_iter
        )
        scalar_fits[target_fold] = scalar_fit
        exclusions[target_fold] = exclusion
        evaluations[target_fold] = evaluate_target_fold(
            full_hierarchy,
            bundles[target_fold],
            scalar_fit,
            distance_matrix,
            dists_mats,
        )
    nested_evaluations = {
        fold: evaluate_nested_target_fold(
            full_hierarchy,
            bundles[fold],
            distance_matrix,
            dists_mats,
            max_iter=args.scalar_max_iter,
        )
        for fold in range(4)
    }
    nested_map_evaluations = {
        fold: evaluate_nested_target_fold(
            full_hierarchy,
            bundles[fold],
            distance_matrix,
            dists_mats,
            max_iter=args.scalar_max_iter,
            decoder="map",
        )
        for fold in range(4)
    }
    raw_scalar = {"a": 1.0, "b": 0.0}
    raw_expected_evaluations = {
        fold: evaluate_target_fold(
            full_hierarchy,
            bundles[fold],
            raw_scalar,
            distance_matrix,
            dists_mats,
            decoder="expected_hdist",
        )
        for fold in range(4)
    }
    raw_map_evaluations = {
        fold: evaluate_target_fold(
            full_hierarchy,
            bundles[fold],
            raw_scalar,
            distance_matrix,
            dists_mats,
            decoder="map",
        )
        for fold in range(4)
    }

    classifier_targets = torch.cat([
        evaluations[fold]["known_targets"] for fold in range(4)
    ])
    classifier_predictions = torch.cat([
        evaluations[fold]["classifier_predictions"] for fold in range(4)
    ])
    known_targets = classifier_targets
    known_predictions = torch.cat([
        evaluations[fold]["known_predictions"] for fold in range(4)
    ])
    pseudo_targets = torch.cat([
        evaluations[fold]["pseudo_targets"] for fold in range(4)
    ])
    pseudo_predictions = torch.cat([
        evaluations[fold]["pseudo_predictions"] for fold in range(4)
    ])
    classifier_metrics = get_results(
        classifier_predictions,
        classifier_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    mixed = mixed_summary(known_metrics, pseudo_metrics)
    unknown_binary = binary_ood_metrics(
        torch.cat([
            evaluations[fold]["known_unknown_mass"]
            for fold in range(4)
        ]).numpy(),
        torch.cat([
            evaluations[fold]["pseudo_unknown_mass"]
            for fold in range(4)
        ]).numpy(),
    )
    per_fold_degradation = {
        str(fold): (
            float(evaluations[fold]["classifier_metrics"]["balanced_acc"])
            - float(evaluations[fold]["known_metrics"]["balanced_acc"])
        )
        for fold in range(4)
    }
    mean_degradation = sum(per_fold_degradation.values()) / 4.0
    max_normalization_error = max(
        evaluations[fold]["normalization_error"]
        for fold in range(4)
    )
    thresholds = {
        "classifier_known_balanced_acc_min": 0.800,
        "posterior_known_balanced_acc_min": 0.780,
        "mean_known_degradation_max": 0.020,
        "per_fold_known_degradation_max": 0.030,
        "pseudo_mapped_parent_balanced_acc_min": 0.227,
        "mixed_balanced_acc_strict_min": 0.503,
        "mixed_balanced_hdist_max": 0.810,
        "unknown_mass_auroc_min": 0.750,
        "normalization_error_max": 1e-5,
    }
    values = {
        "classifier_known_balanced_acc": float(
            classifier_metrics["balanced_acc"]
        ),
        "posterior_known_balanced_acc": float(
            known_metrics["balanced_acc"]
        ),
        "mean_known_degradation": mean_degradation,
        "max_per_fold_known_degradation": max(
            per_fold_degradation.values()
        ),
        "pseudo_mapped_parent_balanced_acc": float(
            pseudo_metrics["balanced_acc"]
        ),
        "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
        "mixed_balanced_hdist": float(
            mixed["mixed_balanced_hdist"]
        ),
        "unknown_mass_auroc": float(unknown_binary["auroc"]),
        "normalization_error": max_normalization_error,
    }
    checks = {
        "classifier_known_balanced_acc": (
            values["classifier_known_balanced_acc"]
            >= thresholds["classifier_known_balanced_acc_min"]
        ),
        "posterior_known_balanced_acc": (
            values["posterior_known_balanced_acc"]
            >= thresholds["posterior_known_balanced_acc_min"]
        ),
        "mean_known_degradation": (
            values["mean_known_degradation"]
            <= thresholds["mean_known_degradation_max"]
        ),
        "per_fold_known_degradation": (
            values["max_per_fold_known_degradation"]
            <= thresholds["per_fold_known_degradation_max"]
        ),
        "pseudo_mapped_parent_balanced_acc": (
            values["pseudo_mapped_parent_balanced_acc"]
            >= thresholds["pseudo_mapped_parent_balanced_acc_min"]
        ),
        "mixed_balanced_acc": (
            values["mixed_balanced_acc"]
            > thresholds["mixed_balanced_acc_strict_min"]
        ),
        "mixed_balanced_hdist": (
            values["mixed_balanced_hdist"]
            <= thresholds["mixed_balanced_hdist_max"]
        ),
        "unknown_mass_auroc": (
            values["unknown_mass_auroc"]
            >= thresholds["unknown_mass_auroc_min"]
        ),
        "normalization": (
            values["normalization_error"]
            <= thresholds["normalization_error_max"]
        ),
    }
    gate = {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
    }
    nested_classifier_targets = torch.cat([
        nested_evaluations[fold]["known_targets"] for fold in range(4)
    ])
    nested_classifier_predictions = torch.cat([
        nested_evaluations[fold]["classifier_predictions"]
        for fold in range(4)
    ])
    nested_known_predictions = torch.cat([
        nested_evaluations[fold]["known_predictions"]
        for fold in range(4)
    ])
    nested_pseudo_targets = torch.cat([
        nested_evaluations[fold]["pseudo_targets"] for fold in range(4)
    ])
    nested_pseudo_predictions = torch.cat([
        nested_evaluations[fold]["pseudo_predictions"]
        for fold in range(4)
    ])
    nested_classifier_metrics = get_results(
        nested_classifier_predictions,
        nested_classifier_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    nested_known_metrics = get_results(
        nested_known_predictions,
        nested_classifier_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    nested_pseudo_metrics = get_results(
        nested_pseudo_predictions,
        nested_pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    nested_mixed = mixed_summary(
        nested_known_metrics, nested_pseudo_metrics
    )
    nested_unknown_binary = binary_ood_metrics(
        torch.cat([
            nested_evaluations[fold]["known_unknown_mass"]
            for fold in range(4)
        ]).numpy(),
        torch.cat([
            nested_evaluations[fold]["pseudo_unknown_mass"]
            for fold in range(4)
        ]).numpy(),
    )
    nested_per_fold_degradation = {
        str(fold): (
            float(
                nested_evaluations[fold][
                    "classifier_metrics"
                ]["balanced_acc"]
            )
            - float(
                nested_evaluations[fold][
                    "known_metrics"
                ]["balanced_acc"]
            )
        )
        for fold in range(4)
    }
    nested_values = {
        "classifier_known_balanced_acc": float(
            nested_classifier_metrics["balanced_acc"]
        ),
        "posterior_known_balanced_acc": float(
            nested_known_metrics["balanced_acc"]
        ),
        "mean_known_degradation": (
            sum(nested_per_fold_degradation.values()) / 4.0
        ),
        "max_per_fold_known_degradation": max(
            nested_per_fold_degradation.values()
        ),
        "pseudo_mapped_parent_balanced_acc": float(
            nested_pseudo_metrics["balanced_acc"]
        ),
        "mixed_balanced_acc": float(
            nested_mixed["mixed_balanced_acc"]
        ),
        "mixed_balanced_hdist": float(
            nested_mixed["mixed_balanced_hdist"]
        ),
        "unknown_mass_auroc": float(nested_unknown_binary["auroc"]),
        "normalization_error": max(
            nested_evaluations[fold]["normalization_error"]
            for fold in range(4)
        ),
    }
    nested_checks = {
        "classifier_known_balanced_acc": (
            nested_values["classifier_known_balanced_acc"]
            >= thresholds["classifier_known_balanced_acc_min"]
        ),
        "posterior_known_balanced_acc": (
            nested_values["posterior_known_balanced_acc"]
            >= thresholds["posterior_known_balanced_acc_min"]
        ),
        "mean_known_degradation": (
            nested_values["mean_known_degradation"]
            <= thresholds["mean_known_degradation_max"]
        ),
        "per_fold_known_degradation": (
            nested_values["max_per_fold_known_degradation"]
            <= thresholds["per_fold_known_degradation_max"]
        ),
        "pseudo_mapped_parent_balanced_acc": (
            nested_values["pseudo_mapped_parent_balanced_acc"]
            >= thresholds["pseudo_mapped_parent_balanced_acc_min"]
        ),
        "mixed_balanced_acc": (
            nested_values["mixed_balanced_acc"]
            > thresholds["mixed_balanced_acc_strict_min"]
        ),
        "mixed_balanced_hdist": (
            nested_values["mixed_balanced_hdist"]
            <= thresholds["mixed_balanced_hdist_max"]
        ),
        "unknown_mass_auroc": (
            nested_values["unknown_mass_auroc"]
            >= thresholds["unknown_mass_auroc_min"]
        ),
        "normalization": (
            nested_values["normalization_error"]
            <= thresholds["normalization_error_max"]
        ),
    }
    nested_gate = {
        "passed": all(nested_checks.values()),
        "thresholds": thresholds,
        "values": nested_values,
        "checks": nested_checks,
    }
    fold_results = {
        str(fold): {
            "classifier_known": evaluations[fold][
                "classifier_metrics"
            ],
            "posterior_known": evaluations[fold]["known_metrics"],
            "pseudo_mapped_parent": evaluations[fold]["pseudo_metrics"],
            "mixed": evaluations[fold]["mixed"],
            "known_degradation": per_fold_degradation[str(fold)],
            "normalization_error": evaluations[fold][
                "normalization_error"
            ],
            "scalar_fit": scalar_fits[fold],
            "episode_level_crossfold_exclusion": exclusions[fold],
        }
        for fold in range(4)
    }
    nested_fold_results = {
        str(fold): {
            "classifier_known": nested_evaluations[fold][
                "classifier_metrics"
            ],
            "posterior_known": nested_evaluations[fold][
                "known_metrics"
            ],
            "pseudo_mapped_parent": nested_evaluations[fold][
                "pseudo_metrics"
            ],
            "mixed": nested_evaluations[fold]["mixed"],
            "known_degradation": nested_per_fold_degradation[str(fold)],
            "normalization_error": nested_evaluations[fold][
                "normalization_error"
            ],
            "inner_scalar_fits": nested_evaluations[fold][
                "inner_scalar_fits"
            ],
            "assignment_audit": nested_evaluations[fold][
                "assignment_audit"
            ],
            "target_disjoint_audit": nested_evaluations[fold][
                "target_disjoint_audit"
            ],
        }
        for fold in range(4)
    }
    decoder_diagnostics = {
        "raw_entcomp_expected_hdist": aggregate_evaluation_summary(
            raw_expected_evaluations, full_hierarchy, dists_mats
        ),
        "raw_entcomp_map": aggregate_evaluation_summary(
            raw_map_evaluations, full_hierarchy, dists_mats
        ),
        "nested_nll_fitted_map": aggregate_evaluation_summary(
            nested_map_evaluations, full_hierarchy, dists_mats
        ),
    }
    result = {
        "status": (
            "nested_oof_go"
            if nested_gate["passed"]
            else "nested_oof_no_go"
        ),
        "method": "cross_fitted_route_preserving_entcomp_posterior",
        "stage": STAGE,
        **nested_confirmatory_metadata(),
        "manifest_hash": manifest["manifest_hash"],
        "input_fold_checkpoints": args.fold_checkpoints,
        "input_checkpoint_identities": input_checkpoint_identities,
        "strict_input_validation_passed": True,
        "actual_ood_loader_loaded": False,
        "actual_ood_dataset_loaded": False,
        "actual_ood_encoded": False,
        "actual_ood_evaluation_implemented_in_this_stage": False,
        "classifier_known": nested_classifier_metrics,
        "posterior_known": nested_known_metrics,
        "pseudo_mapped_parent": nested_pseudo_metrics,
        "mixed": nested_mixed,
        "unknown_mass_binary": nested_unknown_binary,
        "per_fold": nested_fold_results,
        "gate": nested_gate,
        "decoder_diagnostics_not_used_for_locked_gate": (
            decoder_diagnostics
        ),
        "non_nested_screening": {
            **screening_audit_metadata(),
            "classifier_known": classifier_metrics,
            "posterior_known": known_metrics,
            "pseudo_mapped_parent": pseudo_metrics,
            "mixed": mixed,
            "unknown_mass_binary": unknown_binary,
            "per_fold": fold_results,
            "gate": gate,
        },
    }
    transfer_scalar_fit = None
    if nested_gate["passed"]:
        transfer_scalar_fit = fit_shared_route_scalars(
            [bundles[fold] for fold in range(4)],
            max_iter=args.scalar_max_iter,
        )
    checkpoint_payload = {
        "stage": STAGE,
        **nested_confirmatory_metadata(),
        "manifest": manifest,
        "manifest_hash": manifest["manifest_hash"],
        "input_fold_checkpoints": args.fold_checkpoints,
        "input_checkpoint_identities": input_checkpoint_identities,
        "input_split_hashes": {
            str(fold): folds[fold]["checkpoint"][
                "crossfit_split_hash"
            ]
            for fold in range(4)
        },
        "fold_models": fold_models,
        "nested_inner_scalar_fits": {
            str(fold): nested_evaluations[fold]["inner_scalar_fits"]
            for fold in range(4)
        },
        "nested_assignment_audits": {
            str(fold): nested_evaluations[fold]["assignment_audit"]
            for fold in range(4)
        },
        "decoder_diagnostics_not_used_for_locked_gate": (
            decoder_diagnostics
        ),
        "oof_bundles_for_threshold_free_method_development": bundles,
        "transfer_scalar_fit": transfer_scalar_fit,
        "gate": nested_gate,
        "non_nested_screening": {
            **screening_audit_metadata(),
            "target_fold_scalar_fits": scalar_fits,
            "episode_level_crossfold_exclusions": exclusions,
            "gate": gate,
        },
        "actual_ood_encoded": False,
    }
    checkpoint_path = Path(args.checkpoint_path)
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    # Long head/scalar fitting can overlap an upstream fold finalization.
    # Refuse to publish artifacts if any content-addressed input changed.
    verify_fold_checkpoint_identities(
        folds, input_checkpoint_identities
    )
    atomic_torch_save(checkpoint_payload, checkpoint_path)
    atomic_torch_save(result, result_path)
    ensure_dir(diagnostics_path.parent)
    save_json(diagnostics_path, json_ready(result))
    print(
        f"CF-RPEP non-nested screen="
        f"{'GO' if gate['passed'] else 'NO-GO'}: "
        f"classifier={values['classifier_known_balanced_acc']:.6f}, "
        f"posterior={values['posterior_known_balanced_acc']:.6f}, "
        f"pseudo={values['pseudo_mapped_parent_balanced_acc']:.6f}, "
        f"mix={values['mixed_balanced_acc']:.6f}, "
        f"BMHD={values['mixed_balanced_hdist']:.6f}, "
        f"AUROC={values['unknown_mass_auroc']:.6f}"
    )
    print(
        f"CF-RPEP nested gate="
        f"{'GO' if nested_gate['passed'] else 'NO-GO'}: "
        f"classifier="
        f"{nested_values['classifier_known_balanced_acc']:.6f}, "
        f"posterior="
        f"{nested_values['posterior_known_balanced_acc']:.6f}, "
        f"pseudo="
        f"{nested_values['pseudo_mapped_parent_balanced_acc']:.6f}, "
        f"mix={nested_values['mixed_balanced_acc']:.6f}, "
        f"BMHD={nested_values['mixed_balanced_hdist']:.6f}, "
        f"AUROC={nested_values['unknown_mass_auroc']:.6f}"
    )
    for name, row in decoder_diagnostics.items():
        print(
            f"CF-RPEP diagnostic {name}: "
            f"ID={float(row['posterior_known']['balanced_acc']):.6f}, "
            f"pseudo="
            f"{float(row['pseudo_mapped_parent']['balanced_acc']):.6f}, "
            f"mix={float(row['mixed']['mixed_balanced_acc']):.6f}, "
            f"BMHD="
            f"{float(row['mixed']['mixed_balanced_hdist']):.6f}, "
            f"AUROC={float(row['unknown_mass_binary']['auroc']):.6f}"
        )
    print(
        "Official OOD loader/dataset was not loaded; "
        f"saved: {result_path}"
    )


if __name__ == "__main__":
    main()
