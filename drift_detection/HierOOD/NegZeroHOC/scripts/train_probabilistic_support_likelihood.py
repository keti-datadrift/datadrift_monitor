from __future__ import annotations

import argparse
import hashlib
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ProHOC.libs.utils.score_util import entcompprob
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    get_results,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hierarchical_support import (
    build_hierarchical_support_calibration,
    expand_to_reference_sample_prototypes,
    expected_hierarchy_distance_predictions,
)
from negzerohoc.multidepth_fusion import (
    multidepth_route_conditionals,
    multidepth_unknown_probabilities,
)
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.probabilistic_support import (
    SharedMaskedCategoricalLikelihood,
    SharedMaskedSupportLikelihood,
    build_energy_episodes,
    build_global_leaf_energy_episodes,
    categorical_episode_targets,
    energy_unknown_probabilities_by_parent,
    global_unknown_probability,
    latent_knownness_terminal_distribution,
    masked_subtree_terminal_distribution,
    mixture_conditionals_by_parent,
    prior_corrected_product_unknown,
    probabilistic_terminal_distribution,
    reference_only_partitions_from_checkpoints,
    stratified_four_way_split,
    weighted_binary_nll,
    weighted_categorical_nll,
)
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from scripts.evaluate_dual_expert_support import (
    build_eval_datasets,
    load_frozen_vision,
    make_eval_loader,
    release_cuda,
)
from scripts.train_idea4_unknown_prompts import encode_dataset_features
from scripts.train_multidepth_feature_heads import (
    MultiDepthLinearHeads,
    payload_logits,
    predictions_from_leaf_probabilities,
    probability_list,
)
from scripts.train_paper_negprompt_ablation import json_ready


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    train_cfg = cfg.get("probabilistic_support_likelihood", {})
    inference_cfg = cfg.get("inference", {})
    experiment_name = str(
        experiment_cfg.get("name", "probabilistic-support-likelihood")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    support_checkpoint = train_cfg.get("support_checkpoint")
    multidepth_checkpoint = train_cfg.get("multidepth_checkpoint")
    split_metadata_checkpoint = train_cfg.get(
        "split_metadata_checkpoint"
    )
    forbidden = {
        key
        for key in train_cfg
        if any(token in key.lower() for token in (
            "alpha", "threshold", "acceptance", "terminal_weight"
        ))
    }
    if forbidden:
        raise ValueError(
            "Threshold-free likelihood config contains forbidden keys: "
            f"{sorted(forbidden)}"
        )
    posterior_form = str(
        train_cfg.get("posterior_form", "factorized")
    ).lower()
    if posterior_form not in {"factorized", "categorical_mixture"}:
        raise ValueError(
            "posterior_form must be factorized or categorical_mixture"
        )
    prototype_mode = str(
        train_cfg.get("prototype_mode", "class_centroid")
    ).lower()
    if prototype_mode not in {"class_centroid", "reference_samples"}:
        raise ValueError(
            "prototype_mode must be class_centroid or reference_samples"
        )
    if not support_checkpoint:
        raise ValueError("support_checkpoint is required")
    if posterior_form == "factorized" and not multidepth_checkpoint:
        raise ValueError(
            "multidepth_checkpoint is required for factorized posterior"
        )

    def artifact(configured, kind: str, filename: str) -> str:
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
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=str(dataset_cfg.get("datadir", "")),
        hierarchy=dataset_cfg.get(
            "hierarchy", "hierarchies/fgvc-aircraft.json"
        ),
        id_split=dataset_cfg.get(
            "id_split", "data/fgvc-aircraft-id-labels.csv"
        ),
        clip_model=clip_cfg.get(
            "model", "openai/clip-vit-base-patch16"
        ),
        tokenizer_model=clip_cfg.get(
            "tokenizer_model",
            clip_cfg.get("model", "openai/clip-vit-base-patch16"),
        ),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        augmentation=cfg.get("augmentation", {}),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        support_checkpoint=str(support_checkpoint),
        split_metadata_checkpoint=(
            str(split_metadata_checkpoint)
            if split_metadata_checkpoint
            else None
        ),
        support_lora_enabled=bool(
            train_cfg.get("support_lora_enabled", True)
        ),
        posterior_form=posterior_form,
        prototype_mode=prototype_mode,
        enforce_id_only_gate=bool(
            train_cfg.get("enforce_id_only_gate", False)
        ),
        multidepth_checkpoint=(
            str(multidepth_checkpoint)
            if multidepth_checkpoint
            else None
        ),
        split_fractions=tuple(
            float(value)
            for value in train_cfg.get(
                "split_fractions", [0.6, 0.2, 0.1, 0.1]
            )
        ),
        epochs=max(1, int(train_cfg.get("epochs", 1000))),
        patience=max(1, int(train_cfg.get("patience", 100))),
        lr=float(train_cfg.get("lr", 0.05)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        initial_concentration=float(
            train_cfg.get("initial_concentration", 5.0)
        ),
        initial_base_energy=float(
            train_cfg.get("initial_base_energy", 2.5)
        ),
        inference_batch_size=max(
            1, int(inference_cfg.get("batch_size", 128))
        ),
        checkpoint_path=artifact(
            train_cfg.get("checkpoint"),
            "checkpoints",
            f"{experiment_name}.pt",
        ),
        result_path=artifact(
            train_cfg.get("result_path"),
            "results",
            f"{experiment_name}.result",
        ),
        diagnostics_path=artifact(
            train_cfg.get("diagnostics_path"),
            "diagnostics",
            f"{experiment_name}.json",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def indices_hash(partitions) -> str:
    digest = hashlib.sha256()
    for name, indices in partitions.items():
        digest.update(name.encode("utf-8"))
        digest.update(indices.detach().long().cpu().numpy().tobytes())
    return digest.hexdigest()


def sample_leaf_nodes(hierarchy, payload) -> list[str]:
    mapping = hierarchy.gen_ds2node_map(payload["classes"])
    node_indices = mapping[payload["targets"].long().cpu()]
    return [
        hierarchy.id_node_list[int(index)]
        for index in node_indices.tolist()
    ]


def move_episodes(episodes, device):
    return {
        "child_similarities": episodes.child_similarities.to(device),
        "prototype_mask": episodes.prototype_mask.to(device),
        "child_mask": episodes.child_mask.to(device),
        "targets": episodes.targets.to(device),
        "categorical_targets": categorical_episode_targets(
            episodes
        ).to(device),
        "weights": episodes.weights.to(device),
    }


@torch.no_grad()
def episode_diagnostics(model, episodes, device, posterior_form):
    model.eval()
    payload = move_episodes(episodes, device)
    masked = episodes.masked.to(device)
    if posterior_form == "categorical_mixture":
        categorical_logits = model.categorical_logits(
            payload["child_similarities"],
            payload["prototype_mask"],
            payload["child_mask"],
        )
        probabilities = torch.softmax(
            categorical_logits, dim=1
        )[:, -1]
        logits = torch.logit(
            probabilities.clamp(1e-7, 1.0 - 1e-7)
        )
        weighted_nll = weighted_categorical_nll(
            categorical_logits,
            payload["categorical_targets"],
            payload["weights"],
        )
    else:
        logits = model(
            payload["child_similarities"],
            payload["prototype_mask"],
            payload["child_mask"],
        )
        probabilities = torch.sigmoid(logits)
        weighted_nll = weighted_binary_nll(
            logits, payload["targets"], payload["weights"]
        )
    binary = binary_ood_metrics(
        logits[~masked].detach().cpu().numpy(),
        logits[masked].detach().cpu().numpy(),
    )
    return {
        "weighted_nll": float(weighted_nll.cpu()),
        "full_unknown_probability_mean": float(
            probabilities[~masked].mean().cpu()
        ),
        "masked_unknown_probability_mean": float(
            probabilities[masked].mean().cpu()
        ),
        "full_vs_masked_auroc": float(binary["auroc"]),
        "full_vs_masked_fpr95": float(binary["fpr95"]),
    }


def train_likelihood(
    args,
    train_episodes,
    validation_episodes,
    device,
):
    model_class = (
        SharedMaskedCategoricalLikelihood
        if args.posterior_form == "categorical_mixture"
        else SharedMaskedSupportLikelihood
    )
    model = model_class(
        initial_concentration=args.initial_concentration,
        initial_base_energy=args.initial_base_energy,
    ).to(device)
    train_payload = move_episodes(train_episodes, device)
    validation_payload = move_episodes(validation_episodes, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.posterior_form == "categorical_mixture":
            logits = model.categorical_logits(
                train_payload["child_similarities"],
                train_payload["prototype_mask"],
                train_payload["child_mask"],
            )
            loss = weighted_categorical_nll(
                logits,
                train_payload["categorical_targets"],
                train_payload["weights"],
            )
        else:
            logits = model(
                train_payload["child_similarities"],
                train_payload["prototype_mask"],
                train_payload["child_mask"],
            )
            loss = weighted_binary_nll(
                logits,
                train_payload["targets"],
                train_payload["weights"],
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            if args.posterior_form == "categorical_mixture":
                validation_logits = model.categorical_logits(
                    validation_payload["child_similarities"],
                    validation_payload["prototype_mask"],
                    validation_payload["child_mask"],
                )
                validation_loss = weighted_categorical_nll(
                    validation_logits,
                    validation_payload["categorical_targets"],
                    validation_payload["weights"],
                )
            else:
                validation_logits = model(
                    validation_payload["child_similarities"],
                    validation_payload["prototype_mask"],
                    validation_payload["child_mask"],
                )
                validation_loss = weighted_binary_nll(
                    validation_logits,
                    validation_payload["targets"],
                    validation_payload["weights"],
                )
        validation_value = float(validation_loss.cpu())
        if validation_value < best_loss - 1e-9:
            best_loss = validation_value
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            history.append({
                "epoch": epoch,
                "train_nll": float(loss.detach().cpu()),
                "validation_nll": validation_value,
                "concentration": float(model.concentration.detach().cpu()),
                "base_energy": float(model.base_energy.detach().cpu()),
            })
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Likelihood training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "best_validation_nll": best_loss,
        "history": history,
        "train": episode_diagnostics(
            model, train_episodes, device, args.posterior_form
        ),
        "validation": episode_diagnostics(
            model, validation_episodes, device, args.posterior_form
        ),
        "concentration": float(model.concentration.detach().cpu()),
        "base_energy": float(model.base_energy.detach().cpu()),
    }


def load_multidepth_heads(path: str, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "frozen_feature_multidepth_heads":
        raise ValueError(
            f"Unexpected multi-depth checkpoint stage: {checkpoint.get('stage')}"
        )
    multidepth_classes = checkpoint["multidepth_classes"]
    first_weight = checkpoint["model_state_dict"]["heads.0.weight"]
    model = MultiDepthLinearHeads(
        int(first_weight.shape[1]),
        [len(nodes) for nodes in multidepth_classes],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return checkpoint, model, multidepth_classes


def evaluate_predictions(
    hierarchy,
    id_payload,
    ood_payload,
    id_predictions,
    ood_predictions,
    dists_mats,
):
    _, id_metrics = evaluate_split(
        hierarchy, id_payload, id_predictions, dists_mats=dists_mats
    )
    _, ood_metrics = evaluate_split(
        hierarchy, ood_payload, ood_predictions, dists_mats=dists_mats
    )
    return {
        "id": id_metrics,
        "ood": ood_metrics,
        "mixed": mixed_summary(id_metrics, ood_metrics),
    }


def probability_diagnostics(
    hierarchy,
    id_terminal,
    ood_terminal,
):
    internal_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node != "root" and node in hierarchy.parent2children
    ])
    id_unknown = id_terminal[:, internal_indices].sum(dim=1)
    ood_unknown = ood_terminal[:, internal_indices].sum(dim=1)
    binary = binary_ood_metrics(
        id_unknown.numpy(), ood_unknown.numpy()
    )
    return {
        "id_unknown_mass_mean": float(id_unknown.mean()),
        "ood_unknown_mass_mean": float(ood_unknown.mean()),
        "unknown_mass_auroc_diagnostic": float(binary["auroc"]),
        "unknown_mass_fpr95_diagnostic": float(binary["fpr95"]),
        "id_terminal_sum_range": [
            float(id_terminal.sum(dim=1).min()),
            float(id_terminal.sum(dim=1).max()),
        ],
        "ood_terminal_sum_range": [
            float(ood_terminal.sum(dim=1).min()),
            float(ood_terminal.sum(dim=1).max()),
        ],
    }


def terminal_evaluation_row(
    hierarchy,
    id_payload,
    ood_payload,
    id_terminal,
    ood_terminal,
    dists_mats,
    distance_matrix,
):
    return {
        "map": evaluate_predictions(
            hierarchy,
            id_payload,
            ood_payload,
            id_terminal.argmax(dim=1),
            ood_terminal.argmax(dim=1),
            dists_mats,
        ),
        "expected_hdist": evaluate_predictions(
            hierarchy,
            id_payload,
            ood_payload,
            expected_hierarchy_distance_predictions(
                id_terminal, distance_matrix
            ),
            expected_hierarchy_distance_predictions(
                ood_terminal, distance_matrix
            ),
            dists_mats,
        ),
        "probability_diagnostics": probability_diagnostics(
            hierarchy, id_terminal, ood_terminal
        ),
    }


def leaf_probability_terminal(
    hierarchy,
    probabilities_by_depth,
    multidepth_classes,
):
    finest = probabilities_by_depth[-1].detach().float().cpu()
    finest_nodes = multidepth_classes[-1]
    if int(finest.shape[1]) != len(finest_nodes):
        raise ValueError("Finest probability width differs from classes")
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    terminal = torch.zeros(
        int(finest.shape[0]),
        len(hierarchy.id_node_list),
        dtype=torch.float32,
    )
    for class_index, node in enumerate(finest_nodes):
        if node in hierarchy.parent2children:
            raise ValueError(f"Finest class {node!r} is not a leaf")
        terminal[:, node_to_index[node]] = finest[:, class_index]
    return terminal / terminal.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)


def id_only_categorical_gate(
    hierarchy,
    model,
    train_payload,
    validation_indices,
    train_leaf_nodes,
    calibration,
):
    """Evaluate Idea I without encoding either official ID or OOD test data."""
    validation_indices = validation_indices.detach().long().cpu()
    validation_features = train_payload["features"].index_select(
        0, validation_indices
    )
    validation_payload = {
        "features": validation_features,
        "targets": train_payload["targets"].index_select(
            0, validation_indices
        ),
        "classes": train_payload["classes"],
    }
    validation_leaf_nodes = [
        train_leaf_nodes[int(index)]
        for index in validation_indices.tolist()
    ]
    dists_mats = make_distance_mats(hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()

    full_routes, full_unknown = mixture_conditionals_by_parent(
        model, hierarchy, validation_features, calibration
    )
    full_terminal = probabilistic_terminal_distribution(
        hierarchy, full_routes, full_unknown
    )
    full_predictions = expected_hierarchy_distance_predictions(
        full_terminal, distance_matrix
    )
    _, full_metrics = evaluate_split(
        hierarchy,
        validation_payload,
        full_predictions,
        dists_mats=dists_mats,
    )

    masked = masked_subtree_terminal_distribution(
        model,
        hierarchy,
        validation_features,
        validation_leaf_nodes,
        calibration,
    )
    masked_predictions = expected_hierarchy_distance_predictions(
        masked["terminal"], distance_matrix
    )
    masked_metrics = get_results(
        masked_predictions,
        masked["targets"],
        hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_mixed = mixed_summary(full_metrics, masked_metrics)
    full_normalization_error = float(
        (full_terminal.sum(dim=1) - 1.0).abs().max()
    )
    thresholds = {
        "full_balanced_acc_min": 0.780,
        "masked_parent_balanced_acc_min": 0.227,
        "pseudo_mixed_balanced_acc_strict_min": 0.503,
        "pseudo_mixed_balanced_hdist_max": 0.81,
        "normalization_error_max": 1e-5,
        "masked_subtree_mass_max": 1e-6,
    }
    values = {
        "full_balanced_acc": float(full_metrics["balanced_acc"]),
        "masked_parent_balanced_acc": float(
            masked_metrics["balanced_acc"]
        ),
        "pseudo_mixed_balanced_acc": float(
            pseudo_mixed["mixed_balanced_acc"]
        ),
        "pseudo_mixed_balanced_hdist": float(
            pseudo_mixed["mixed_balanced_hdist"]
        ),
        "full_normalization_error": full_normalization_error,
        "masked_normalization_error": float(
            masked["max_normalization_error"]
        ),
        "masked_subtree_mass": float(
            masked["max_masked_subtree_mass"]
        ),
    }
    checks = {
        "full_balanced_acc": (
            values["full_balanced_acc"]
            >= thresholds["full_balanced_acc_min"]
        ),
        "masked_parent_balanced_acc": (
            values["masked_parent_balanced_acc"]
            >= thresholds["masked_parent_balanced_acc_min"]
        ),
        "pseudo_mixed_balanced_acc": (
            values["pseudo_mixed_balanced_acc"]
            > thresholds["pseudo_mixed_balanced_acc_strict_min"]
        ),
        "pseudo_mixed_balanced_hdist": (
            values["pseudo_mixed_balanced_hdist"]
            <= thresholds["pseudo_mixed_balanced_hdist_max"]
        ),
        "full_normalization": (
            values["full_normalization_error"]
            <= thresholds["normalization_error_max"]
        ),
        "masked_normalization": (
            values["masked_normalization_error"]
            <= thresholds["normalization_error_max"]
        ),
        "masked_subtree_zero_mass": (
            values["masked_subtree_mass"]
            <= thresholds["masked_subtree_mass_max"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "selection_data": "original_calibration_posterior_val_only",
        "decoder": "expected_hdist",
        "episode_weighting": "uniform_terminal",
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
        "full_metrics": full_metrics,
        "masked_parent_metrics": masked_metrics,
        "pseudo_mixed": pseudo_mixed,
        "masked_episode_count": int(masked["episodes"]),
        "masked_group_count": int(masked["groups"]),
    }


def save_fitted_likelihood_checkpoint(
    args,
    partitions,
    fitted_models,
    fit_diagnostics,
    *,
    lineage,
    id_only_gate,
    extra_models=None,
):
    checkpoint_path = Path(args.checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    checkpoint_stage = (
        "shared_masked_categorical_likelihood"
        if args.posterior_form == "categorical_mixture"
        else "shared_masked_support_likelihood"
    )
    torch.save({
        "stage": checkpoint_stage,
        "posterior_form": args.posterior_form,
        "prototype_mode": args.prototype_mode,
        "threshold_free": True,
        "support_checkpoint": args.support_checkpoint,
        "split_metadata_checkpoint": args.split_metadata_checkpoint,
        "multidepth_checkpoint": args.multidepth_checkpoint,
        "split_indices": partitions,
        "split_hash": indices_hash(partitions),
        "split_lineage": lineage,
        "id_only_gate": id_only_gate,
        "models": {
            name: {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            for name, model in fitted_models.items()
        },
        "extra_models": {
            name: {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            for name, model in (extra_models or {}).items()
        },
        "fit_diagnostics": fit_diagnostics,
    }, checkpoint_path)


def main():
    args = parse_args()
    configure_reproducibility(
        args.seed, deterministic=args.deterministic
    )
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    train_dataset, id_dataset, ood_dataset = build_eval_datasets(
        args, hierarchy
    )
    support_checkpoint, clip_model = load_frozen_vision(
        args, args.support_checkpoint, device
    )

    # Fit the complete support posterior before encoding either official test
    # split. This call ordering prevents ID-test/OOD data from entering fitting.
    train_payload = encode_dataset_features(
        args,
        clip_model,
        train_dataset,
        make_eval_loader(args, train_dataset),
        device,
        "encode likelihood ID train",
    )
    metadata_checkpoint = None
    if args.split_metadata_checkpoint:
        metadata_checkpoint = torch.load(
            args.split_metadata_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    reference_only = bool(
        (support_checkpoint.get("args") or {}).get(
            "reference_only_training", False
        )
    ) or metadata_checkpoint is not None
    if reference_only:
        partitions, split_lineage = (
            reference_only_partitions_from_checkpoints(
                support_checkpoint,
                train_payload["targets"],
                expected_seed=args.seed,
                metadata_checkpoint=metadata_checkpoint,
                posterior_train_per_class=8,
                posterior_val_per_class=5,
            )
        )
        support_calibration_indices = partitions["posterior_train"]
    else:
        partitions_tuple = stratified_four_way_split(
            train_payload["targets"],
            fractions=args.split_fractions,
            seed=args.seed,
        )
        partitions = dict(zip(
            (
                "reference",
                "support_calibration",
                "posterior_train",
                "posterior_val",
            ),
            partitions_tuple,
        ))
        support_calibration_indices = partitions["support_calibration"]
        split_lineage = {
            "training_split_source": "new_stratified_four_way_split",
            "split_fractions": args.split_fractions,
        }
    calibration = build_hierarchical_support_calibration(
        hierarchy,
        train_payload["features"],
        train_payload["classes"],
        train_payload["targets"],
        reference_indices=partitions["reference"],
        calibration_indices=support_calibration_indices,
    )
    if args.prototype_mode == "reference_samples":
        calibration = expand_to_reference_sample_prototypes(
            hierarchy,
            calibration,
            train_payload["features"],
            train_payload["classes"],
            train_payload["targets"],
        )
    train_leaf_nodes = sample_leaf_nodes(hierarchy, train_payload)
    fitted_models = {}
    fit_diagnostics = {}
    for weighting in ("uniform_terminal", "paired_view"):
        train_episodes = build_energy_episodes(
            hierarchy,
            train_payload["features"],
            partitions["posterior_train"],
            train_leaf_nodes,
            calibration,
            weighting=weighting,
        )
        validation_episodes = build_energy_episodes(
            hierarchy,
            train_payload["features"],
            partitions["posterior_val"],
            train_leaf_nodes,
            calibration,
            weighting=weighting,
        )
        model, diagnostics = train_likelihood(
            args, train_episodes, validation_episodes, device
        )
        fitted_models[weighting] = model
        fit_diagnostics[weighting] = diagnostics

    global_model = None
    if args.posterior_form == "categorical_mixture":
        global_train_episodes = build_global_leaf_energy_episodes(
            hierarchy,
            train_payload["features"],
            partitions["posterior_train"],
            train_leaf_nodes,
            calibration,
        )
        global_validation_episodes = build_global_leaf_energy_episodes(
            hierarchy,
            train_payload["features"],
            partitions["posterior_val"],
            train_leaf_nodes,
            calibration,
        )
        global_model, global_diagnostics = train_likelihood(
            args,
            global_train_episodes,
            global_validation_episodes,
            device,
        )
        fit_diagnostics["global_structural"] = global_diagnostics

    id_only_gate = None
    if args.posterior_form == "categorical_mixture":
        id_only_gate = id_only_categorical_gate(
            hierarchy,
            fitted_models["uniform_terminal"],
            train_payload,
            partitions["posterior_val"],
            train_leaf_nodes,
            calibration,
        )
    save_fitted_likelihood_checkpoint(
        args,
        partitions,
        fitted_models,
        fit_diagnostics,
        lineage=split_lineage,
        id_only_gate=id_only_gate,
        extra_models=(
            {"global_structural": global_model}
            if global_model is not None
            else None
        ),
    )
    if (
        args.enforce_id_only_gate
        and id_only_gate is not None
        and not id_only_gate["passed"]
    ):
        release_cuda(clip_model)
        result = {
            "status": "id_only_no_go",
            "method": "shared_masked_categorical_mixture_likelihood",
            "posterior_form": args.posterior_form,
            "used_actual_ood_for_posterior_training_or_selection": False,
            "official_id_test_encoded": False,
            "official_ood_test_encoded": False,
            "support_checkpoint": args.support_checkpoint,
            "split_metadata_checkpoint": args.split_metadata_checkpoint,
            "split_lineage": split_lineage,
            "split_hash": indices_hash(partitions),
            "split_sizes": {
                key: int(value.numel())
                for key, value in partitions.items()
            },
            "fit_diagnostics": fit_diagnostics,
            "id_only_gate": id_only_gate,
        }
        result_path = Path(args.result_path)
        diagnostics_path = Path(args.diagnostics_path)
        ensure_dir(result_path.parent)
        ensure_dir(diagnostics_path.parent)
        torch.save(result, result_path)
        save_json(diagnostics_path, json_ready(result))
        print(
            "ID-only gate failed; official ID/OOD encoding skipped. "
            f"saved: {result_path}"
        )
        return

    id_payload = encode_dataset_features(
        args,
        clip_model,
        id_dataset,
        make_eval_loader(args, id_dataset),
        device,
        "encode likelihood official ID test",
    )
    ood_payload = encode_dataset_features(
        args,
        clip_model,
        ood_dataset,
        make_eval_loader(args, ood_dataset),
        device,
        "encode likelihood official OOD test",
    )
    release_cuda(clip_model)

    dists_mats = make_distance_mats(hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    results = {}
    if args.posterior_form == "categorical_mixture":
        route_rows = {}
        mixture_cache = {}
        for weighting, model in fitted_models.items():
            id_routes, id_unknown = mixture_conditionals_by_parent(
                model,
                hierarchy,
                id_payload["features"],
                calibration,
            )
            ood_routes, ood_unknown = mixture_conditionals_by_parent(
                model,
                hierarchy,
                ood_payload["features"],
                calibration,
            )
            id_terminal = probabilistic_terminal_distribution(
                hierarchy, id_routes, id_unknown
            )
            ood_terminal = probabilistic_terminal_distribution(
                hierarchy, ood_routes, ood_unknown
            )
            mixture_cache[weighting] = {
                "id_routes": id_routes,
                "ood_routes": ood_routes,
                "id_terminal": id_terminal,
                "ood_terminal": ood_terminal,
            }
            route_rows[weighting] = terminal_evaluation_row(
                hierarchy,
                id_payload,
                ood_payload,
                id_terminal,
                ood_terminal,
                dists_mats,
                distance_matrix,
            )
        results["prototype_mixture"] = route_rows
        if global_model is not None:
            id_global_unknown = global_unknown_probability(
                global_model,
                hierarchy,
                id_payload["features"],
                calibration,
            )
            ood_global_unknown = global_unknown_probability(
                global_model,
                hierarchy,
                ood_payload["features"],
                calibration,
            )
            global_rows = {}
            for weighting, cached in mixture_cache.items():
                id_zero_unknown = {
                    parent: torch.zeros(
                        int(id_payload["features"].shape[0])
                    )
                    for parent in hierarchy.parent2children
                    if parent != "root"
                }
                ood_zero_unknown = {
                    parent: torch.zeros(
                        int(ood_payload["features"].shape[0])
                    )
                    for parent in hierarchy.parent2children
                    if parent != "root"
                }
                id_leaf_conditional = (
                    probabilistic_terminal_distribution(
                        hierarchy,
                        cached["id_routes"],
                        id_zero_unknown,
                    )
                )
                ood_leaf_conditional = (
                    probabilistic_terminal_distribution(
                        hierarchy,
                        cached["ood_routes"],
                        ood_zero_unknown,
                    )
                )
                id_global_terminal = (
                    latent_knownness_terminal_distribution(
                        hierarchy,
                        id_global_unknown,
                        id_leaf_conditional,
                        cached["id_terminal"],
                    )
                )
                ood_global_terminal = (
                    latent_knownness_terminal_distribution(
                        hierarchy,
                        ood_global_unknown,
                        ood_leaf_conditional,
                        cached["ood_terminal"],
                    )
                )
                global_rows[weighting] = terminal_evaluation_row(
                    hierarchy,
                    id_payload,
                    ood_payload,
                    id_global_terminal,
                    ood_global_terminal,
                    dists_mats,
                    distance_matrix,
                )
            results["global_latent_prototype"] = global_rows
        if args.multidepth_checkpoint:
            multidepth_checkpoint, heads, multidepth_classes = (
                load_multidepth_heads(
                    args.multidepth_checkpoint, device
                )
            )
            id_logits = payload_logits(heads, id_payload, device)
            ood_logits = payload_logits(heads, ood_payload, device)
            temperatures = multidepth_checkpoint["temperatures"]
            for calibration_name, route_temperatures in (
                ("uncalibrated", None),
                ("id_train_temperature", temperatures),
            ):
                id_probabilities = probability_list(
                    id_logits, route_temperatures
                )
                ood_probabilities = probability_list(
                    ood_logits, route_temperatures
                )
                id_ent_unknown = multidepth_unknown_probabilities(
                    id_probabilities,
                    hierarchy,
                    multidepth_classes,
                    entcompprob,
                )
                ood_ent_unknown = multidepth_unknown_probabilities(
                    ood_probabilities,
                    hierarchy,
                    multidepth_classes,
                    entcompprob,
                )
                cross_rows = {}
                product_rows = {}
                global_head_mixture_rows = {}
                global_head_ent_rows = {}
                id_head_leaf = leaf_probability_terminal(
                    hierarchy, id_probabilities, multidepth_classes
                )
                ood_head_leaf = leaf_probability_terminal(
                    hierarchy, ood_probabilities, multidepth_classes
                )
                for weighting, model in fitted_models.items():
                    id_routes, id_density_unknown = (
                        mixture_conditionals_by_parent(
                            model,
                            hierarchy,
                            id_payload["features"],
                            calibration,
                        )
                    )
                    ood_routes, ood_density_unknown = (
                        mixture_conditionals_by_parent(
                            model,
                            hierarchy,
                            ood_payload["features"],
                            calibration,
                        )
                    )
                    id_cross_terminal = (
                        probabilistic_terminal_distribution(
                            hierarchy, id_routes, id_ent_unknown
                        )
                    )
                    ood_cross_terminal = (
                        probabilistic_terminal_distribution(
                            hierarchy, ood_routes, ood_ent_unknown
                        )
                    )
                    cross_rows[weighting] = terminal_evaluation_row(
                        hierarchy,
                        id_payload,
                        ood_payload,
                        id_cross_terminal,
                        ood_cross_terminal,
                        dists_mats,
                        distance_matrix,
                    )
                    if global_model is not None:
                        id_global_head_mixture = (
                            latent_knownness_terminal_distribution(
                                hierarchy,
                                id_global_unknown,
                                id_head_leaf,
                                mixture_cache[weighting]["id_terminal"],
                            )
                        )
                        ood_global_head_mixture = (
                            latent_knownness_terminal_distribution(
                                hierarchy,
                                ood_global_unknown,
                                ood_head_leaf,
                                mixture_cache[weighting]["ood_terminal"],
                            )
                        )
                        global_head_mixture_rows[weighting] = (
                            terminal_evaluation_row(
                                hierarchy,
                                id_payload,
                                ood_payload,
                                id_global_head_mixture,
                                ood_global_head_mixture,
                                dists_mats,
                                distance_matrix,
                            )
                        )
                        id_global_head_ent = (
                            latent_knownness_terminal_distribution(
                                hierarchy,
                                id_global_unknown,
                                id_head_leaf,
                                id_cross_terminal,
                            )
                        )
                        ood_global_head_ent = (
                            latent_knownness_terminal_distribution(
                                hierarchy,
                                ood_global_unknown,
                                ood_head_leaf,
                                ood_cross_terminal,
                            )
                        )
                        global_head_ent_rows[weighting] = (
                            terminal_evaluation_row(
                                hierarchy,
                                id_payload,
                                ood_payload,
                                id_global_head_ent,
                                ood_global_head_ent,
                                dists_mats,
                                distance_matrix,
                            )
                        )
                    id_product_unknown = (
                        prior_corrected_product_unknown(
                            hierarchy,
                            id_density_unknown,
                            id_ent_unknown,
                            prior_mode=weighting,
                        )
                    )
                    ood_product_unknown = (
                        prior_corrected_product_unknown(
                            hierarchy,
                            ood_density_unknown,
                            ood_ent_unknown,
                            prior_mode=weighting,
                        )
                    )
                    id_product_terminal = (
                        probabilistic_terminal_distribution(
                            hierarchy, id_routes, id_product_unknown
                        )
                    )
                    ood_product_terminal = (
                        probabilistic_terminal_distribution(
                            hierarchy, ood_routes, ood_product_unknown
                        )
                    )
                    product_rows[weighting] = terminal_evaluation_row(
                        hierarchy,
                        id_payload,
                        ood_payload,
                        id_product_terminal,
                        ood_product_terminal,
                        dists_mats,
                        distance_matrix,
                    )
                results[
                    f"cross_entcompprob_{calibration_name}"
                ] = cross_rows
                results[
                    f"product_ent_density_{calibration_name}"
                ] = product_rows
                if global_model is not None:
                    results[
                        "global_latent_head_mixture_"
                        f"{calibration_name}"
                    ] = global_head_mixture_rows
                    results[
                        "global_latent_head_ent_"
                        f"{calibration_name}"
                    ] = global_head_ent_rows
        primary_route = "prototype_mixture"
        route_calibration = "joint_prototype_mixture"
    else:
        multidepth_checkpoint, heads, multidepth_classes = (
            load_multidepth_heads(args.multidepth_checkpoint, device)
        )
        id_logits = payload_logits(heads, id_payload, device)
        ood_logits = payload_logits(heads, ood_payload, device)
        temperatures = multidepth_checkpoint["temperatures"]
        for route_name, route_temperatures in (
            ("uncalibrated", None),
            ("id_train_temperature", temperatures),
        ):
            id_probabilities = probability_list(
                id_logits, route_temperatures
            )
            ood_probabilities = probability_list(
                ood_logits, route_temperatures
            )
            id_routes = multidepth_route_conditionals(
                id_probabilities, hierarchy, multidepth_classes
            )
            ood_routes = multidepth_route_conditionals(
                ood_probabilities, hierarchy, multidepth_classes
            )
            route_rows = {
                "leaf": evaluate_predictions(
                    hierarchy,
                    id_payload,
                    ood_payload,
                    predictions_from_leaf_probabilities(
                        id_probabilities, hierarchy, multidepth_classes
                    ),
                    predictions_from_leaf_probabilities(
                        ood_probabilities, hierarchy, multidepth_classes
                    ),
                    dists_mats,
                )
            }
            for weighting, model in fitted_models.items():
                id_unknown = energy_unknown_probabilities_by_parent(
                    model,
                    hierarchy,
                    id_payload["features"],
                    calibration,
                )
                ood_unknown = energy_unknown_probabilities_by_parent(
                    model,
                    hierarchy,
                    ood_payload["features"],
                    calibration,
                )
                id_terminal = probabilistic_terminal_distribution(
                    hierarchy, id_routes, id_unknown
                )
                ood_terminal = probabilistic_terminal_distribution(
                    hierarchy, ood_routes, ood_unknown
                )
                route_rows[weighting] = terminal_evaluation_row(
                    hierarchy,
                    id_payload,
                    ood_payload,
                    id_terminal,
                    ood_terminal,
                    dists_mats,
                    distance_matrix,
                )
            results[route_name] = route_rows
        primary_route = "id_train_temperature"
        route_calibration = "id_train_temperature"

    primary = results[primary_route]["uniform_terminal"][
        "expected_hdist"
    ]
    method_name = (
        "shared_masked_categorical_mixture_likelihood"
        if args.posterior_form == "categorical_mixture"
        else "shared_masked_support_likelihood"
    )
    global_knownness_diagnostics = None
    if args.posterior_form == "categorical_mixture" and global_model is not None:
        global_binary = binary_ood_metrics(
            id_global_unknown.numpy(), ood_global_unknown.numpy()
        )
        global_knownness_diagnostics = {
            "id_unknown_probability_mean": float(
                id_global_unknown.mean()
            ),
            "ood_unknown_probability_mean": float(
                ood_global_unknown.mean()
            ),
            "auroc_diagnostic": float(global_binary["auroc"]),
            "fpr95_diagnostic": float(global_binary["fpr95"]),
        }
    result = {
        "status": "completed",
        "method": method_name,
        "posterior_form": args.posterior_form,
        "prototype_mode": args.prototype_mode,
        "threshold_free_inference": True,
        "inference_thresholds": [],
        "node_specific_parameters": False,
        "depth_specific_parameters": False,
        "used_actual_ood_for_posterior_training_or_selection": False,
        "official_test_encoded_after_posterior_checkpoint_saved": True,
        "development_benchmark_caveat": (
            "Frozen backbones used official test ID for earlier checkpoint "
            "selection and this OOD split has been adaptively inspected."
        ),
        "support_checkpoint": args.support_checkpoint,
        "support_checkpoint_stage": support_checkpoint.get("stage"),
        "split_metadata_checkpoint": args.split_metadata_checkpoint,
        "multidepth_checkpoint": args.multidepth_checkpoint,
        "split_fractions": args.split_fractions,
        "split_lineage": split_lineage,
        "split_hash": indices_hash(partitions),
        "split_sizes": {
            key: int(value.numel()) for key, value in partitions.items()
        },
        "fit_diagnostics": fit_diagnostics,
        "global_knownness_diagnostics": global_knownness_diagnostics,
        "id_only_gate": id_only_gate,
        "primary_spec": {
            "parameter_sharing": "global",
            "episode_weighting": "uniform_terminal",
            "route_calibration": route_calibration,
            "decoder": "expected_hdist",
        },
        "primary": primary,
        "results": results,
    }
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    ensure_dir(result_path.parent)
    ensure_dir(diagnostics_path.parent)
    torch.save(result, result_path)
    save_json(diagnostics_path, json_ready(result))
    print(
        "primary ID/OOD/Mixed BAcc="
        f"{float(primary['id']['balanced_acc']):.6f}/"
        f"{float(primary['ood']['balanced_acc']):.6f}/"
        f"{float(primary['mixed']['mixed_balanced_acc']):.6f}, "
        f"Mixed BMHD="
        f"{float(primary['mixed']['mixed_balanced_hdist']):.6f}"
    )
    for route_name, route_rows in results.items():
        for weighting in ("uniform_terminal", "paired_view"):
            for decoder in ("map", "expected_hdist"):
                row = route_rows[weighting][decoder]
                print(
                    f"{route_name}/{weighting}/{decoder}: "
                    f"{float(row['id']['balanced_acc']):.6f}/"
                    f"{float(row['ood']['balanced_acc']):.6f}/"
                    f"{float(row['mixed']['mixed_balanced_acc']):.6f}, "
                    f"BMHD="
                    f"{float(row['mixed']['mixed_balanced_hdist']):.6f}"
                )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
