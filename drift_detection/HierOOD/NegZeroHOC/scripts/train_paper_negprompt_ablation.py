from __future__ import annotations

import argparse
from argparse import Namespace
from collections import defaultdict
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.paper_negprompt import (
    TransferableNegativePromptLearner,
    negprompt_loss,
    negprompt_mcm_confidence,
)
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from scripts.train_idea3_joint_vision_lora import (
    build_datasets,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    load_frozen_positive_stack,
)


CHECKPOINT_STAGE = "literature_negprompt_frozen_p64k4"


def float_list(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [
        float(item.strip())
        for item in str(value).split(",")
        if item.strip()
    ]


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    positive_cfg = cfg.get("positive", {})
    train_cfg = cfg.get("negative_training", {})
    prompt_cfg = train_cfg.get("prompt", {})
    loss_cfg = train_cfg.get("loss", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = str(
        experiment_cfg.get("name", "literature-negprompt")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")
    distance_mode = str(loss_cfg.get("distance_mode", "attractive"))
    if distance_mode not in {"attractive", "repulsive"}:
        raise ValueError(
            "negative_training.loss.distance_mode must be attractive or "
            "repulsive"
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
        datadir=str(datadir),
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
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        augmentation=cfg.get("augmentation", {}),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        positive_checkpoint=str(positive_checkpoint),
        epochs=max(1, int(train_cfg.get("epochs", 15))),
        batch_size=max(1, int(train_cfg.get("batch_size", 64))),
        shots_per_class=max(
            1, int(train_cfg.get("shots_per_class", 16))
        ),
        lr=float(train_cfg.get("lr", 0.1)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(
            train_cfg.get("gradient_clip_norm", 1.0)
        ),
        text_gradient_checkpointing=bool(
            train_cfg.get("text_gradient_checkpointing", True)
        ),
        num_negative_prompts=max(
            1, int(prompt_cfg.get("count", 2))
        ),
        init_noise=float(prompt_cfg.get("init_noise", 1e-2)),
        distance_mode=distance_mode,
        beta=float(loss_cfg.get("beta", 0.1)),
        gamma=float(loss_cfg.get("gamma", 0.05)),
        training_logit_scale=loss_cfg.get(
            "logit_scale", "clip"
        ),
        inference_temperature=float(
            inference_cfg.get("temperature", 1.0)
        ),
        id_acceptance=float(
            inference_cfg.get("id_acceptance", 0.95)
        ),
        id_acceptance_grid=float_list(
            inference_cfg.get(
                "id_acceptance_grid",
                [0.95, 0.90, 0.80, 0.70],
            )
        ),
        inference_batch_size=max(
            1, int(inference_cfg.get("batch_size", 128))
        ),
        checkpoint=artifact(
            train_cfg.get("checkpoint"),
            "checkpoints",
            "negative.pt",
        ),
        last_checkpoint=artifact(
            train_cfg.get("last_checkpoint"),
            "checkpoints",
            "negative-last.pt",
        ),
        result_path=artifact(
            train_cfg.get("result_path"),
            "results",
            "negprompt.result",
        ),
        diagnostics_path=artifact(
            train_cfg.get("diagnostics_path"),
            "diagnostics",
            "negprompt.json",
        ),
        automatic_resume=bool(
            train_cfg.get("resume", {}).get("automatic", True)
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def all_edge_pairs(hierarchy) -> list[tuple[str, str]]:
    parents = sorted(
        hierarchy.parent2children,
        key=lambda node: (
            len(hierarchy.node_ancestors.get(node, [])),
            node,
        ),
    )
    return [
        (parent, child)
        for parent in parents
        for child in hierarchy.parent2children[parent]
    ]


def leaf_nodes(hierarchy) -> list[str]:
    return [
        node
        for node in hierarchy.id_node_list
        if node not in hierarchy.parent2children
    ]


def node_targets(hierarchy, payload: dict) -> list[str]:
    mapping = hierarchy.gen_ds2node_map(payload["classes"])
    indices = mapping[payload["targets"].long().cpu()]
    return [
        hierarchy.id_node_list[int(index)]
        for index in indices.tolist()
    ]


def select_few_shot_indices(
    targets: list[int],
    *,
    shots_per_class: int,
    seed: int,
) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    rng = random.Random(seed)
    selected = []
    for target in sorted(by_class):
        candidates = list(by_class[target])
        rng.shuffle(candidates)
        selected.extend(candidates[:shots_per_class])
    rng.shuffle(selected)
    return selected


def prompt_state(negative) -> dict[str, torch.Tensor]:
    return {
        "context_offsets": (
            negative.context_offsets.detach().cpu().clone()
        )
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        return (
            value.item()
            if value.numel() == 1
            else value.detach().cpu().tolist()
        )
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_prompt_state(negative, state: dict) -> None:
    with torch.no_grad():
        negative.context_offsets.copy_(
            state["context_offsets"].to(
                negative.context_offsets.device,
                dtype=negative.context_offsets.dtype,
            )
        )


def save_training_checkpoint(
    path: str,
    *,
    args,
    positive_checkpoint: dict,
    negative,
    epoch: int,
    history: list[dict],
    optimizer=None,
    scheduler=None,
    complete: bool,
) -> None:
    checkpoint = {
        "stage": CHECKPOINT_STAGE,
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
        "positive_checkpoint": args.positive_checkpoint,
        "positive_checkpoint_stage": positive_checkpoint.get("stage"),
        "distance_mode": args.distance_mode,
        "num_negative_prompts": args.num_negative_prompts,
        "negative_state_dict": prompt_state(negative),
        "epoch": int(epoch),
        "history": history,
        "training_complete": bool(complete),
        "args": vars(args),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    ensure_dir(Path(path).parent)
    torch.save(checkpoint, path)


def positive_only_confidence(
    images: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = F.normalize(images.float(), dim=-1)
    positives = F.normalize(positives.float(), dim=-1)
    probabilities = F.softmax(
        images @ positives.t() / float(temperature),
        dim=1,
    )
    return probabilities.max(dim=1)


@torch.no_grad()
def flat_mcm_diagnostics(
    hierarchy,
    id_payload: dict,
    ood_payload: dict,
    positive_by_edge: dict,
    negative_by_edge: dict,
    temperature: float,
) -> dict:
    leaves = leaf_nodes(hierarchy)
    pairs = [(hierarchy.child2parent[leaf], leaf) for leaf in leaves]
    positives = torch.stack([positive_by_edge[pair] for pair in pairs])
    negatives = torch.stack([negative_by_edge[pair] for pair in pairs])

    id_base, _ = positive_only_confidence(
        id_payload["features"], positives, temperature
    )
    ood_base, _ = positive_only_confidence(
        ood_payload["features"], positives, temperature
    )
    id_negative, _ = negprompt_mcm_confidence(
        id_payload["features"],
        positives,
        negatives,
        temperature=temperature,
    )
    ood_negative, _ = negprompt_mcm_confidence(
        ood_payload["features"],
        positives,
        negatives,
        temperature=temperature,
    )
    return {
        "definition": (
            "NegPrompt/MCM maximum positive probability over all ID leaves; "
            "larger 1-confidence is more OOD-like"
        ),
        "positive_only": binary_ood_metrics(
            (1.0 - id_base).cpu().numpy(),
            (1.0 - ood_base).cpu().numpy(),
        ),
        "with_negative_prompts": binary_ood_metrics(
            (1.0 - id_negative).cpu().numpy(),
            (1.0 - ood_negative).cpu().numpy(),
        ),
    }


@torch.no_grad()
def oracle_parent_local_diagnostics(
    hierarchy,
    id_payload: dict,
    ood_payload: dict,
    positive_by_edge: dict,
    negative_by_edge: dict,
    temperature: float,
) -> dict:
    id_nodes = node_targets(hierarchy, id_payload)
    ood_nodes = node_targets(hierarchy, ood_payload)
    by_parent = {}
    for parent in sorted(hierarchy.parent2children):
        if parent == "root":
            continue
        ood_indices = [
            index
            for index, node in enumerate(ood_nodes)
            if node == parent
        ]
        if not ood_indices:
            continue
        parent_index = hierarchy.id_node_list.index(parent)
        id_indices = [
            index
            for index, node in enumerate(id_nodes)
            if parent_index in hierarchy.node_ancestors.get(node, [])
        ]
        if not id_indices:
            continue

        children = list(hierarchy.parent2children[parent])
        pairs = [(parent, child) for child in children]
        positives = torch.stack([positive_by_edge[pair] for pair in pairs])
        negatives = torch.stack([negative_by_edge[pair] for pair in pairs])
        id_features = id_payload["features"][id_indices]
        ood_features = ood_payload["features"][ood_indices]
        id_base, _ = positive_only_confidence(
            id_features, positives, temperature
        )
        ood_base, _ = positive_only_confidence(
            ood_features, positives, temperature
        )
        id_negative, _ = negprompt_mcm_confidence(
            id_features,
            positives,
            negatives,
            temperature=temperature,
        )
        ood_negative, _ = negprompt_mcm_confidence(
            ood_features,
            positives,
            negatives,
            temperature=temperature,
        )
        by_parent[parent] = {
            "id_samples": len(id_indices),
            "ood_samples": len(ood_indices),
            "positive_only": binary_ood_metrics(
                (1.0 - id_base).cpu().numpy(),
                (1.0 - ood_base).cpu().numpy(),
            ),
            "with_negative_prompts": binary_ood_metrics(
                (1.0 - id_negative).cpu().numpy(),
                (1.0 - ood_negative).cpu().numpy(),
            ),
        }

    keys = (
        "auroc",
        "fpr95",
        "best_balanced_acc_diagnostic_only",
    )
    macro = {}
    for method in ("positive_only", "with_negative_prompts"):
        macro[method] = {
            key: float(np.mean([
                item[method][key] for item in by_parent.values()
            ]))
            for key in keys
        }
    return {
        "definition": (
            "oracle rejection parent; ID descendants versus OOD samples "
            "whose ground-truth rejection node is that parent"
        ),
        "num_supported_parents": len(by_parent),
        "macro": macro,
        "by_parent": by_parent,
    }


def local_probability_tables(
    features: torch.Tensor,
    hierarchy,
    positive_by_edge: dict,
    negative_by_edge: dict,
    temperature: float,
) -> dict[str, torch.Tensor]:
    tables = {}
    for parent, children in hierarchy.parent2children.items():
        if parent == "root":
            continue
        pairs = [(parent, child) for child in children]
        positives = torch.stack([positive_by_edge[pair] for pair in pairs])
        negatives = torch.stack([negative_by_edge[pair] for pair in pairs])
        probabilities, _ = negprompt_mcm_confidence(
            features,
            positives,
            negatives,
            temperature=temperature,
        )
        # The decoder needs every child probability, not only the maximum.
        images = F.normalize(features.float(), dim=-1)
        positive_matrix = F.normalize(positives.float(), dim=-1)
        negative_matrix = F.normalize(negatives.float(), dim=-1)
        positive_logits = (
            images @ positive_matrix.t() / float(temperature)
        )
        negative_logits = torch.einsum(
            "bd,ckd->bck", images, negative_matrix
        ) / float(temperature)
        denominator = torch.logsumexp(
            torch.cat(
                [positive_logits, negative_logits.flatten(1)],
                dim=1,
            ),
            dim=1,
            keepdim=True,
        )
        tables[parent] = torch.exp(positive_logits - denominator)
        del probabilities
    return tables


@torch.no_grad()
def decode_hierarchical_mcm(
    features: torch.Tensor,
    hierarchy,
    positive_by_edge: dict,
    negative_by_edge: dict,
    *,
    temperature: float,
    threshold: float | None,
) -> dict:
    leaves = leaf_nodes(hierarchy)
    leaf_pairs = [
        (hierarchy.child2parent[leaf], leaf) for leaf in leaves
    ]
    leaf_features = torch.stack([
        positive_by_edge[pair] for pair in leaf_pairs
    ])
    images = F.normalize(features.float(), dim=-1)
    leaf_features = F.normalize(leaf_features.float(), dim=-1)
    leaf_indices = (images @ leaf_features.t()).argmax(dim=1)
    predicted_leaves = [leaves[int(index)] for index in leaf_indices]
    tables = local_probability_tables(
        features,
        hierarchy,
        positive_by_edge,
        negative_by_edge,
        temperature,
    )

    minimum_confidences = []
    weakest_parents = []
    for sample_index, leaf in enumerate(predicted_leaves):
        path = [
            hierarchy.id_node_list[index]
            for index in hierarchy.node_ancestors.get(leaf, [])
        ] + [leaf]
        candidates = []
        for parent, child in zip(path[:-1], path[1:]):
            if parent == "root":
                continue
            child_index = list(
                hierarchy.parent2children[parent]
            ).index(child)
            candidates.append((
                float(tables[parent][sample_index, child_index]),
                parent,
            ))
        if not candidates:
            # Some pruned FGVC leaves attach directly to root. Root rejection
            # is disabled by protocol, so these routes are always accepted.
            confidence, weakest_parent = 1.0, "root"
        else:
            confidence, weakest_parent = min(candidates)
        minimum_confidences.append(confidence)
        weakest_parents.append(weakest_parent)

    if threshold is None:
        predicted_nodes = predicted_leaves
    else:
        predicted_nodes = [
            parent if confidence < float(threshold) else leaf
            for leaf, parent, confidence in zip(
                predicted_leaves,
                weakest_parents,
                minimum_confidences,
            )
        ]
    node_to_index = {
        node: index
        for index, node in enumerate(hierarchy.id_node_list)
    }
    preds = torch.tensor([
        node_to_index[node] for node in predicted_nodes
    ], dtype=torch.long)
    rejected = [
        node in hierarchy.parent2children
        for node in predicted_nodes
    ]
    return {
        "preds": preds,
        "minimum_path_confidence": torch.tensor(minimum_confidences),
        "predicted_leaves": predicted_leaves,
        "weakest_parents": weakest_parents,
        "predicted_nodes": predicted_nodes,
        "unknown_selection_rate": float(np.mean(rejected)),
    }


@torch.no_grad()
def evaluate_all(
    args,
    hierarchy,
    positive,
    negative,
    val_payload: dict,
    ood_payload: dict,
) -> tuple[dict, dict]:
    pairs = all_edge_pairs(hierarchy)
    positive_features = positive.encode_edges(pairs).float().cpu()
    negative_features = negative.encode_edges(pairs).float().cpu()
    positive_by_edge = {
        pair: positive_features[index]
        for index, pair in enumerate(pairs)
    }
    negative_by_edge = {
        pair: negative_features[index]
        for index, pair in enumerate(pairs)
    }
    flat = flat_mcm_diagnostics(
        hierarchy,
        val_payload,
        ood_payload,
        positive_by_edge,
        negative_by_edge,
        args.inference_temperature,
    )
    local = oracle_parent_local_diagnostics(
        hierarchy,
        val_payload,
        ood_payload,
        positive_by_edge,
        negative_by_edge,
        args.inference_temperature,
    )

    val_unthresholded = decode_hierarchical_mcm(
        val_payload["features"],
        hierarchy,
        positive_by_edge,
        negative_by_edge,
        temperature=args.inference_temperature,
        threshold=None,
    )
    distance_mats = make_distance_mats(hierarchy)
    acceptance_grid = list(dict.fromkeys(
        [args.id_acceptance] + args.id_acceptance_grid
    ))
    decoder_grid = []
    full_decodes = {}
    for acceptance in acceptance_grid:
        quantile = max(0.0, min(1.0, 1.0 - float(acceptance)))
        threshold = float(torch.quantile(
            val_unthresholded["minimum_path_confidence"],
            quantile,
        ))
        val_decoded = decode_hierarchical_mcm(
            val_payload["features"],
            hierarchy,
            positive_by_edge,
            negative_by_edge,
            temperature=args.inference_temperature,
            threshold=threshold,
        )
        ood_decoded = decode_hierarchical_mcm(
            ood_payload["features"],
            hierarchy,
            positive_by_edge,
            negative_by_edge,
            temperature=args.inference_temperature,
            threshold=threshold,
        )
        val_targets, val_metrics = evaluate_split(
            hierarchy,
            val_payload,
            val_decoded["preds"],
            dists_mats=distance_mats,
        )
        ood_targets, ood_metrics = evaluate_split(
            hierarchy,
            ood_payload,
            ood_decoded["preds"],
            dists_mats=distance_mats,
        )
        ood_expected_nodes = [
            hierarchy.id_node_list[int(index)]
            for index in ood_targets.tolist()
        ]
        ood_rejected = [
            node in hierarchy.parent2children
            for node in ood_decoded["predicted_nodes"]
        ]
        correct_parent_given_rejection = [
            predicted == expected
            for predicted, expected, rejected in zip(
                ood_decoded["predicted_nodes"],
                ood_expected_nodes,
                ood_rejected,
            )
            if rejected
        ]
        row = {
            "id_acceptance_target": float(acceptance),
            "threshold_quantile": quantile,
            "threshold": threshold,
            "val_balanced_acc": float(val_metrics["balanced_acc"]),
            "val_balanced_hdist": float(
                val_metrics["balanced_hdist"]
            ),
            "val_unknown_selection_rate": (
                val_decoded["unknown_selection_rate"]
            ),
            "ood_balanced_acc": float(ood_metrics["balanced_acc"]),
            "ood_balanced_hdist": float(
                ood_metrics["balanced_hdist"]
            ),
            "ood_unknown_selection_rate": (
                ood_decoded["unknown_selection_rate"]
            ),
            "ood_correct_parent_precision_given_rejection": (
                float(np.mean(correct_parent_given_rejection))
                if correct_parent_given_rejection
                else 0.0
            ),
            **mixed_summary(val_metrics, ood_metrics),
        }
        decoder_grid.append(row)
        full_decodes[float(acceptance)] = (
            val_decoded,
            ood_decoded,
            val_targets,
            ood_targets,
            val_metrics,
            ood_metrics,
        )
    (
        val_decoded,
        ood_decoded,
        val_targets,
        ood_targets,
        val_metrics,
        ood_metrics,
    ) = full_decodes[float(args.id_acceptance)]
    primary = next(
        row for row in decoder_grid
        if row["id_acceptance_target"] == float(args.id_acceptance)
    )
    decoder = {
        "definition": (
            "positive leaf max-cosine route; reject at the weakest non-root "
            "local NegPrompt/MCM edge when its positive probability is below "
            "an ID-validation-only threshold"
        ),
        "id_acceptance_target": args.id_acceptance,
        "threshold_quantile": primary["threshold_quantile"],
        "threshold": primary["threshold"],
        "used_ood_for_threshold_selection": False,
        "id_only_operating_point_grid": decoder_grid,
        "val": {
            "metrics": val_metrics,
            "unknown_selection_rate": (
                val_decoded["unknown_selection_rate"]
            ),
        },
        "ood": {
            "metrics": ood_metrics,
            "unknown_selection_rate": (
                ood_decoded["unknown_selection_rate"]
            ),
            "correct_parent_precision_given_rejection": (
                primary[
                    "ood_correct_parent_precision_given_rejection"
                ]
            ),
        },
        "mixed": mixed_summary(val_metrics, ood_metrics),
    }
    result = {
        "method": "literature_negprompt_ablation",
        "distance_mode": args.distance_mode,
        "used_actual_ood_for_training_or_selection": False,
        "flat_mcm": flat,
        "oracle_parent_local": local,
        "hierarchical_decoder": decoder,
        "outputs": {
            "val_preds": val_decoded["preds"],
            "val_targets": val_targets,
            "ood_preds": ood_decoded["preds"],
            "ood_targets": ood_targets,
        },
    }
    diagnostics = {
        key: value
        for key, value in result.items()
        if key != "outputs"
    }
    return result, diagnostics


def main() -> None:
    args = parse_args()
    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
    )
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(
        REPO_ROOT,
        args.id_split,
        args.hierarchy,
    )
    train_dataset, val_dataset, ood_dataset = build_datasets(
        args, hierarchy
    )
    selected_indices = select_few_shot_indices(
        train_dataset.targets,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
    )
    train_loader = make_loader(
        Subset(train_dataset, selected_indices),
        args.batch_size,
        args.num_workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.inference_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    ood_loader = make_loader(
        ood_dataset,
        args.inference_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    (
        positive_checkpoint,
        clip_model,
        _text_encoder,
        _prompt_cfg,
        positive,
        replaced_modules,
    ) = load_frozen_positive_stack(
        args,
        hierarchy,
        device,
    )
    negative = TransferableNegativePromptLearner(
        positive,
        num_negative_prompts=args.num_negative_prompts,
        init_noise=args.init_noise,
    ).to(device)
    if args.text_gradient_checkpointing:
        if not hasattr(clip_model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "This CLIP implementation does not support text gradient "
                "checkpointing"
            )
        clip_model.gradient_checkpointing_enable()
    optimizer = torch.optim.SGD(
        negative.trainable_parameters(),
        lr=args.lr,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
    )
    if args.training_logit_scale == "clip":
        training_logit_scale = float(
            clip_model.logit_scale.exp().detach().cpu()
        )
    else:
        training_logit_scale = float(args.training_logit_scale)

    pairs = all_edge_pairs(hierarchy)
    with torch.no_grad():
        positive_features = positive.encode_edges(pairs).detach()
    start_epoch = 1
    history = []
    if args.automatic_resume and Path(args.last_checkpoint).exists():
        resume = torch.load(
            args.last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if (
            resume.get("stage") == CHECKPOINT_STAGE
            and resume.get("distance_mode") == args.distance_mode
        ):
            load_prompt_state(negative, resume["negative_state_dict"])
            history = list(resume.get("history", []))
            start_epoch = int(resume["epoch"]) + 1
            if not resume.get("training_complete", False):
                optimizer.load_state_dict(
                    resume["optimizer_state_dict"]
                )
                scheduler.load_state_dict(
                    resume["scheduler_state_dict"]
                )

    print(
        "Literature NegPrompt training: "
        f"device={device}, mode={args.distance_mode}, "
        f"frozen_lora_modules={len(replaced_modules)}, "
        f"shots={args.shots_per_class}, samples={len(selected_indices)}, "
        f"batch={args.batch_size}, epochs={args.epochs}, "
        f"negative_prompts={args.num_negative_prompts}, "
        f"text_checkpointing={args.text_gradient_checkpointing}, "
        f"beta={args.beta}, gamma={args.gamma}, "
        f"logit_scale={training_logit_scale:.6f}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        negative.train()
        clip_model.eval()
        if args.text_gradient_checkpointing:
            # Transformers only activates encoder checkpointing in train mode.
            positive.text_encoder.text_model.train()
        epoch_stats = []
        for images, _targets in train_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(
                    pixel_values=images
                ).float()
            negative_features = negative.encode_edges(pairs)
            loss, stats = negprompt_loss(
                image_features,
                positive_features,
                negative_features,
                logit_scale=training_logit_scale,
                beta=args.beta,
                gamma=args.gamma,
                distance_mode=args.distance_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                negative.trainable_parameters(),
                args.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            epoch_stats.append(stats)
        averaged = {
            key: float(np.mean([item[key] for item in epoch_stats]))
            for key in epoch_stats[0]
        }
        averaged.update({
            "epoch": epoch,
            "steps": len(epoch_stats),
            "lr": optimizer.param_groups[0]["lr"],
        })
        history.append(averaged)
        save_training_checkpoint(
            args.last_checkpoint,
            args=args,
            positive_checkpoint=positive_checkpoint,
            negative=negative,
            epoch=epoch,
            history=history,
            optimizer=optimizer,
            scheduler=scheduler,
            complete=epoch == args.epochs,
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={averaged['loss']:.6f}, "
            f"nis_excess={averaged['nis_excess']:.6f}, "
            f"npd={averaged['npd_loss']:.6f}, "
            f"nnd={averaged['nnd_loss']:.6f}, "
            f"pos_neg_cos={averaged['positive_negative_cosine']:.6f}"
        )

    save_training_checkpoint(
        args.checkpoint,
        args=args,
        positive_checkpoint=positive_checkpoint,
        negative=negative,
        epoch=args.epochs,
        history=history,
        complete=True,
    )
    negative.eval()
    positive.text_encoder.text_model.eval()
    val_payload = encode_dataset_features(
        args,
        clip_model,
        val_dataset,
        val_loader,
        device,
        "encode ID val",
    )
    ood_payload = encode_dataset_features(
        args,
        clip_model,
        ood_dataset,
        ood_loader,
        device,
        "encode OOD",
    )
    result, diagnostics = evaluate_all(
        args,
        hierarchy,
        positive,
        negative,
        val_payload,
        ood_payload,
    )
    result.update({
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "training_history": history,
    })
    diagnostics.update({
        "config": args.config,
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "training_history": history,
    })
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    save_json(args.diagnostics_path, json_ready(diagnostics))

    flat = diagnostics["flat_mcm"]
    local = diagnostics["oracle_parent_local"]["macro"]
    decoder = diagnostics["hierarchical_decoder"]
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved result: {args.result_path}")
    print(f"saved diagnostics: {args.diagnostics_path}")
    print(
        "Flat AUROC: "
        f"positive={flat['positive_only']['auroc']:.6f}, "
        f"negative={flat['with_negative_prompts']['auroc']:.6f}"
    )
    print(
        "Local macro AUROC: "
        f"positive={local['positive_only']['auroc']:.6f}, "
        f"negative={local['with_negative_prompts']['auroc']:.6f}"
    )
    print(
        "HOC: "
        f"ID BAcc={float(decoder['val']['metrics']['balanced_acc']):.6f}, "
        f"OOD BAcc={float(decoder['ood']['metrics']['balanced_acc']):.6f}, "
        f"Mixed BAcc={float(decoder['mixed']['mixed_balanced_acc']):.6f}"
    )


if __name__ == "__main__":
    main()
