from __future__ import annotations

import argparse
from argparse import Namespace
import random
import sys
from pathlib import Path

import torch
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import save_idea3_checkpoint
from negzerohoc.clip_backend import ClipBackend
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.image_adapters import ImageAdapterConfig, build_image_adapter
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.losses import (
    depthwise_prompt_metric_loss,
    positive_ce_loss,
    prompt_metric_loss,
    sparse_path_bottleneck_loss,
)
from negzerohoc.output_layout import resolve_experiment_artifact, resolve_shared_feature_dir
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.soft_prompting import SoftPromptTextEncoder, soft_prompt_smoke_test
from negzerohoc.training_data import build_positive_edge_examples, gather_image_features, group_examples_by_parent, node_path, sample_examples


def load_config(path):
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    features_cfg = cfg.get("features", {})
    prompt_cfg = cfg.get("prompt", {})
    image_adapter_cfg = cfg.get("image_adapter", cfg.get("image_adaptation", {}))
    train_cfg = cfg.get("positive_training", {})
    loss_cfg = train_cfg.get("loss", {})
    validation_cfg = train_cfg.get("validation", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea3")
    output_root = experiment_cfg.get("output_root", "outputs")
    checkpoint = resolve_experiment_artifact(
        train_cfg.get("checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}-positive.pt",
    )
    checkpoint_path = Path(checkpoint)
    default_last_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}-last{checkpoint_path.suffix}")
    result_path = resolve_experiment_artifact(
        train_cfg.get("result_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="results",
        default_filename=f"{experiment_name}-positive_child_only.result",
    )
    diagnostics_path = resolve_experiment_artifact(
        train_cfg.get("diagnostics_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="diagnostics",
        default_filename=f"{experiment_name}-positive-diagnostics.json",
    )
    last_checkpoint = resolve_experiment_artifact(
        validation_cfg.get("last_checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=default_last_checkpoint.name,
    )

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=output_root,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        features_dir=str(resolve_shared_feature_dir(
            features_cfg.get("dir") or inference_cfg.get("features_dir"),
            output_root=output_root,
        )) if features_cfg.get("dir") or inference_cfg.get("features_dir") else None,
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        prompt=prompt_cfg,
        image_adapter=image_adapter_cfg,
        epochs=int(train_cfg.get("epochs", 20)),
        batch_size=int(train_cfg.get("batch_size", 1024)),
        parents_per_step=int(train_cfg.get("parents_per_step", 4)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        loss_name=loss_cfg.get("name", "ce"),
        loss_prompt_scope=loss_cfg.get("prompt_scope", "parent_local"),
        loss_alpha=float(loss_cfg.get("alpha", 2.0)),
        loss_beta=float(loss_cfg.get("beta", 50.0)),
        loss_lam=float(loss_cfg.get("lam", 0.5)),
        loss_mining_margin=float(loss_cfg.get("mining_margin", 0.1)),
        loss_minimum_mode=loss_cfg.get("minimum_mode", "sample"),
        loss_dist_scale=float(loss_cfg.get("dist_scale", 2.0)),
        loss_dist_pow=float(loss_cfg.get("dist_pow", 1.0)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        metric_weight=float(loss_cfg.get("metric_weight", 1.0)),
        loss_microbatch_size=int(loss_cfg.get("microbatch_size", train_cfg.get("batch_size", 1024))),
        loss_execution=loss_cfg.get("execution", "microbatch"),
        loss_bottleneck_weight=float(loss_cfg.get("bottleneck_weight", 0.5)),
        loss_bottleneck_temperature=float(loss_cfg.get("bottleneck_temperature", 0.5)),
        loss_route_margin=float(loss_cfg.get("route_margin", 0.05)),
        loss_margin_weight=float(loss_cfg.get("margin_weight", 0.25)),
        eval_after_training=bool(train_cfg.get("eval_after_training", True)),
        eval_inference_mode=train_cfg.get("eval_mode", inference_cfg.get("mode", "positive_child_only")),
        validation_enabled=bool(validation_cfg.get("enabled", False)),
        validation_every_n_epochs=max(1, int(validation_cfg.get("every_n_epochs", 1))),
        validation_start_epoch=max(1, int(validation_cfg.get("start_epoch", 1))),
        validation_inference_mode=validation_cfg.get(
            "mode",
            train_cfg.get("eval_mode", inference_cfg.get("mode", "positive_child_only")),
        ),
        validation_metric=validation_cfg.get("metric", "balanced_acc"),
        validation_save_best=bool(validation_cfg.get("save_best", True)),
        last_checkpoint=str(last_checkpoint),
        inference_batch_size=int(inference_cfg.get("batch_size", 1024)),
        inference_tau=float(inference_cfg.get("tau", 1.0)),
        checkpoint=str(checkpoint),
        result_path=str(result_path),
        diagnostics_path=str(diagnostics_path),
    )


def _path_label_tensor(hierarchy, nodes: list[str], max_len: int, device: str | torch.device) -> torch.Tensor:
    node_to_idx = {node: idx for idx, node in enumerate(hierarchy.id_node_list)}
    rows = []
    for node in nodes:
        path = node_path(hierarchy, node)
        if len(path) < max_len:
            path = path + [path[-1]] * (max_len - len(path))
        elif len(path) > max_len:
            path = path[:max_len]
        rows.append([node_to_idx[item] for item in path])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _variable_path_label_tensor(hierarchy, nodes: list[str], max_len: int, device: str | torch.device) -> torch.Tensor:
    node_to_idx = {node: idx for idx, node in enumerate(hierarchy.id_node_list)}
    rows = []
    for node in nodes:
        path = node_path(hierarchy, node)
        row = [-1] * max_len
        for depth, item in enumerate(path[:max_len]):
            row[depth] = node_to_idx[item]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


def build_depth_prompt_sets(hierarchy) -> dict[int, list[str]]:
    by_depth = {}
    for node in hierarchy.id_node_list:
        depth = len(hierarchy.node_ancestors.get(node, []))
        if depth <= 0:
            continue
        by_depth.setdefault(depth, []).append(node)
    return {depth: sorted(nodes) for depth, nodes in sorted(by_depth.items())}


def build_depth_prompt_metadata(hierarchy, depth_nodes: dict[int, list[str]], device: str | torch.device) -> dict:
    max_len = hierarchy.max_depth + 1
    node_to_idx = {node: idx for idx, node in enumerate(hierarchy.id_node_list)}
    return {
        "node_labels": {
            depth: torch.tensor([node_to_idx[node] for node in nodes], dtype=torch.long, device=device)
            for depth, nodes in depth_nodes.items()
        },
        "path_labels": {
            depth: _variable_path_label_tensor(hierarchy, nodes, max_len, device)
            for depth, nodes in depth_nodes.items()
        },
    }


def encode_depth_prompts(learner, hierarchy, depth_nodes: dict[int, list[str]]) -> dict[int, torch.Tensor]:
    encoded = {}
    for depth, nodes in depth_nodes.items():
        pairs = [(hierarchy.child2parent[node], node) for node in nodes]
        encoded[depth] = learner.encode_edges(pairs)
    return encoded


def depthwise_ce_loss(
    image_features: torch.Tensor,
    prompt_features_by_depth: dict[int, torch.Tensor],
    image_node_labels_by_depth: torch.Tensor,
    prompt_node_labels_by_depth: dict[int, torch.Tensor],
    tau: float = 0.07,
) -> tuple[torch.Tensor, dict]:
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    device = image_features.device
    losses = []
    correct = 0
    total = 0

    for depth in sorted(prompt_features_by_depth):
        if depth <= 0:
            continue
        image_labels = image_node_labels_by_depth[:, depth].to(device)
        valid = image_labels >= 0
        if not bool(valid.any()):
            continue
        prompt_features = torch.nn.functional.normalize(prompt_features_by_depth[depth].float(), dim=-1)
        prompt_labels = prompt_node_labels_by_depth[depth].to(device)
        node_to_pos = {int(node): idx for idx, node in enumerate(prompt_labels.detach().cpu().tolist())}
        target_positions = torch.tensor(
            [node_to_pos[int(node)] for node in image_labels[valid].detach().cpu().tolist()],
            dtype=torch.long,
            device=device,
        )
        logits = image_features[valid] @ prompt_features.t() / float(tau)
        losses.append(torch.nn.functional.cross_entropy(logits, target_positions))
        correct += int((logits.argmax(dim=1) == target_positions).sum().detach().cpu())
        total += int(target_positions.numel())

    if not losses:
        zero = image_features.sum() * 0.0
        return zero, {"acc": 0.0, "loss": 0.0}

    loss = torch.stack(losses).mean()
    return loss, {"acc": correct / max(1, total), "loss": float(loss.detach().cpu())}


def compute_positive_loss(
    args,
    hierarchy,
    learner,
    image_features,
    child_features,
    parent_examples,
    children,
    targets,
    device,
):
    loss_name = args.loss_name.lower()
    logits = image_features @ child_features.t() / float(args.tau)
    local_acc = (logits.argmax(dim=1) == targets.to(logits.device)).float().mean()

    if loss_name == "ce":
        loss, stats = positive_ce_loss(image_features, child_features, targets, tau=args.tau)
        stats["metric_loss"] = 0.0
        return loss, stats

    metric_names = {"ms", "hims_min", "himsmin", "weihims", "hims_min_wei"}
    combo_names = {"ce_ms", "ce_hims_min", "ce_himsmin", "ce_weihims"}
    if loss_name not in metric_names and loss_name not in combo_names:
        raise ValueError(f"Unsupported positive_training.loss.name: {args.loss_name}")

    metric_name = loss_name
    use_ce = False
    if loss_name.startswith("ce_"):
        use_ce = True
        metric_name = loss_name[3:]

    max_len = hierarchy.max_depth + 1
    image_nodes = [example.child for example in parent_examples]
    image_path_labels = _path_label_tensor(hierarchy, image_nodes, max_len, device)
    prompt_path_labels = _path_label_tensor(hierarchy, children, max_len, device)
    metric_loss, metric_stats = prompt_metric_loss(
        image_features,
        child_features,
        image_path_labels,
        prompt_path_labels,
        loss_name=metric_name,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        lam=args.loss_lam,
        mining_margin=args.loss_mining_margin,
        minimum_mode=args.loss_minimum_mode,
        dist_scale=args.loss_dist_scale,
        dist_pow=args.loss_dist_pow,
    )

    if use_ce:
        ce_loss, _ = positive_ce_loss(image_features, child_features, targets, tau=args.tau)
        loss = args.ce_weight * ce_loss + args.metric_weight * metric_loss
        ce_value = float(ce_loss.detach().cpu())
    else:
        loss = metric_loss
        ce_value = 0.0

    return loss, {
        "acc": float(local_acc.detach().cpu()),
        "loss": float(loss.detach().cpu()),
        "ce_loss": ce_value,
        "metric_loss": metric_stats["metric_loss"],
    }


def compute_depth_global_positive_loss(
    args,
    image_features,
    image_node_labels_by_depth,
    prompt_features_by_depth,
    prompt_node_labels_by_depth,
    prompt_path_labels_by_depth,
):
    loss_name = args.loss_name.lower()
    metric_names = {"ms", "hims_min", "himsmin", "weihims", "hims_min_wei"}
    combo_names = {"ce_ms", "ce_hims_min", "ce_himsmin", "ce_weihims"}
    if loss_name not in metric_names and loss_name not in combo_names:
        raise ValueError(f"Unsupported depth-global positive_training.loss.name: {args.loss_name}")

    metric_name = loss_name
    use_ce = False
    if loss_name.startswith("ce_"):
        use_ce = True
        metric_name = loss_name[3:]

    metric_loss, metric_stats = depthwise_prompt_metric_loss(
        image_features,
        prompt_features_by_depth,
        image_node_labels_by_depth,
        prompt_node_labels_by_depth,
        prompt_path_labels_by_depth,
        loss_name=metric_name,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        lam=args.loss_lam,
        mining_margin=args.loss_mining_margin,
        minimum_mode=args.loss_minimum_mode,
        dist_scale=args.loss_dist_scale,
        dist_pow=args.loss_dist_pow,
    )
    ce_value = 0.0
    if use_ce:
        ce_loss, ce_stats = depthwise_ce_loss(
            image_features,
            prompt_features_by_depth,
            image_node_labels_by_depth,
            prompt_node_labels_by_depth,
            tau=args.tau,
        )
        loss = args.ce_weight * ce_loss + args.metric_weight * metric_loss
        acc = ce_stats["acc"]
        ce_value = float(ce_loss.detach().cpu())
    else:
        loss = metric_loss
        _, ce_stats = depthwise_ce_loss(
            image_features,
            prompt_features_by_depth,
            image_node_labels_by_depth,
            prompt_node_labels_by_depth,
            tau=args.tau,
        )
        acc = ce_stats["acc"]

    return loss, {
        "acc": float(acc),
        "loss": float(loss.detach().cpu()),
        "ce_loss": ce_value,
        "metric_loss": metric_stats["metric_loss"],
        "pair_mode": metric_stats["pair_mode"],
    }


def build_sparse_path_decisions(
    hierarchy,
    learner,
    image_features: torch.Tensor,
    leaf_nodes: list[str],
    tau: float,
) -> tuple[list[list[torch.Tensor]], list[list[int]], dict]:
    """Build exhaustive local sibling decisions only on active target paths."""
    if len(leaf_nodes) != int(image_features.shape[0]):
        raise ValueError("leaf_nodes must align with image_features")

    active = {}
    for sample_index, leaf in enumerate(leaf_nodes):
        path = node_path(hierarchy, leaf)
        for parent, child in zip(path[:-1], path[1:]):
            children = list(hierarchy.parent2children.get(parent, []))
            if child not in children:
                continue
            active.setdefault(parent, []).append((sample_index, child))

    ordered_parents = sorted(
        active,
        key=lambda parent: (len(hierarchy.node_ancestors.get(parent, [])), parent),
    )
    all_edges = []
    parent_slices = {}
    cursor = 0
    for parent in ordered_parents:
        children = list(hierarchy.parent2children[parent])
        all_edges.extend((parent, child) for child in children)
        parent_slices[parent] = (cursor, cursor + len(children), children)
        cursor += len(children)

    prompt_features = learner.encode_edges(all_edges)
    image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
    prompt_features = torch.nn.functional.normalize(prompt_features.float(), dim=-1)
    decisions = [[] for _ in leaf_nodes]
    targets = [[] for _ in leaf_nodes]

    for parent in ordered_parents:
        start, end, children = parent_slices[parent]
        child_to_index = {child: index for index, child in enumerate(children)}
        entries = active[parent]
        sample_indices = torch.tensor(
            [sample_index for sample_index, _ in entries],
            dtype=torch.long,
            device=image_features.device,
        )
        logits = (
            image_features.index_select(0, sample_indices)
            @ prompt_features[start:end].t()
            / float(tau)
        )
        for row, (sample_index, child) in enumerate(entries):
            decisions[sample_index].append(logits[row])
            targets[sample_index].append(child_to_index[child])

    return decisions, targets, {
        "active_parents": len(ordered_parents),
        "active_prompts": len(all_edges),
    }


def compute_sparse_path_positive_loss(
    args,
    hierarchy,
    learner,
    image_features: torch.Tensor,
    leaf_nodes: list[str],
) -> tuple[torch.Tensor, dict]:
    decisions, targets, active_stats = build_sparse_path_decisions(
        hierarchy,
        learner,
        image_features,
        leaf_nodes,
        tau=args.tau,
    )
    loss, stats = sparse_path_bottleneck_loss(
        decisions,
        targets,
        bottleneck_weight=args.loss_bottleneck_weight,
        bottleneck_temperature=args.loss_bottleneck_temperature,
        route_margin=args.loss_route_margin / float(args.tau),
        margin_weight=args.loss_margin_weight,
    )
    stats.update(active_stats)
    return loss, stats


def build_active_path_entries(hierarchy, leaf_nodes: list[str]) -> tuple[dict, list[int]]:
    active = {}
    path_lengths = []
    for sample_index, leaf in enumerate(leaf_nodes):
        sample_edges = []
        path = node_path(hierarchy, leaf)
        for parent, child in zip(path[:-1], path[1:]):
            children = list(hierarchy.parent2children.get(parent, []))
            if child not in children:
                continue
            sample_edges.append((parent, child))
        path_lengths.append(len(sample_edges))
        for depth_slot, (parent, child) in enumerate(sample_edges):
            active.setdefault(parent, []).append((sample_index, child, depth_slot))
    return active, path_lengths


def backward_sparse_path_bottleneck_streaming(
    args,
    hierarchy,
    learner,
    image_adapter,
    raw_image_features: torch.Tensor,
    leaf_nodes: list[str],
    grad_scaler=None,
) -> dict:
    """Backpropagate exact bottleneck-path gradients one parent at a time.

    The first pass computes dL/d(local CE) without retaining text-encoder
    activations. The second pass recomputes one exhaustive sibling decision at
    a time and applies those chain-rule coefficients. Prompt activations are
    released after every parent while the resulting gradient is identical to
    differentiating the complete sample-wise bottleneck loss at once.
    """
    active, path_lengths = build_active_path_entries(hierarchy, leaf_nodes)
    if not active or any(length <= 0 for length in path_lengths):
        raise ValueError("Streaming sparse path loss received an empty target path")

    ordered_parents = sorted(
        active,
        key=lambda parent: (len(hierarchy.node_ancestors.get(parent, [])), parent),
    )
    adapted_features = torch.nn.functional.normalize(
        image_adapter(raw_image_features).float(), dim=-1
    )
    feature_leaf = adapted_features.detach().requires_grad_(True)
    batch_size = len(leaf_nodes)
    records = [[] for _ in leaf_nodes]
    local_correct = 0
    local_total = 0
    active_prompts = 0

    with torch.no_grad():
        for parent in ordered_parents:
            children = list(hierarchy.parent2children[parent])
            active_prompts += len(children)
            child_to_index = {child: index for index, child in enumerate(children)}
            entries = active[parent]
            sample_indices = torch.tensor(
                [sample_index for sample_index, _, _ in entries],
                dtype=torch.long,
                device=feature_leaf.device,
            )
            child_features = torch.nn.functional.normalize(
                learner.encode_children(parent, children).float(), dim=-1
            )
            logits = (
                feature_leaf.detach().index_select(0, sample_indices)
                @ child_features.t()
                / float(args.tau)
            )
            target_tensor = torch.tensor(
                [child_to_index[child] for _, child, _ in entries],
                dtype=torch.long,
                device=feature_leaf.device,
            )
            ce_values = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
            predictions = torch.argmax(logits, dim=1)
            local_correct += int((predictions == target_tensor).sum().cpu())
            local_total += int(target_tensor.numel())

            margin_values = torch.zeros_like(ce_values)
            if len(children) > 1 and (args.loss_margin_weight > 0.0 or args.loss_route_margin > 0.0):
                negative_mask = torch.ones_like(logits, dtype=torch.bool)
                negative_mask.scatter_(1, target_tensor.unsqueeze(1), False)
                negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
                margin_values = torch.nn.functional.softplus(
                    torch.logsumexp(negative_logits, dim=1)
                    - logits.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
                    + args.loss_route_margin / float(args.tau)
                )

            for row, (sample_index, _, depth_slot) in enumerate(entries):
                records[sample_index].append(
                    (parent, row, depth_slot, ce_values[row], margin_values[row])
                )

    ce_coefficients = {
        parent: torch.zeros(len(entries), dtype=feature_leaf.dtype, device=feature_leaf.device)
        for parent, entries in active.items()
    }
    margin_coefficients = {
        parent: torch.zeros(len(entries), dtype=feature_leaf.dtype, device=feature_leaf.device)
        for parent, entries in active.items()
    }
    detached_sample_losses = []
    temperature = float(args.loss_bottleneck_temperature)

    for sample_records in records:
        sample_records.sort(key=lambda record: record[2])
        local_ce = torch.stack([record[3] for record in sample_records])
        local_margin = torch.stack([record[4] for record in sample_records])
        num_decisions = int(local_ce.numel())
        mean_ce = local_ce.mean()
        smooth_worst = temperature * (
            torch.logsumexp(local_ce / temperature, dim=0)
            - torch.log(torch.tensor(float(num_decisions), device=local_ce.device))
        )
        sample_loss = (
            (1.0 - args.loss_bottleneck_weight) * mean_ce
            + args.loss_bottleneck_weight * smooth_worst
            + args.loss_margin_weight * local_margin.mean()
        )
        detached_sample_losses.append(sample_loss)

        ce_weights = (
            (1.0 - args.loss_bottleneck_weight) / float(num_decisions)
            + args.loss_bottleneck_weight * torch.softmax(local_ce / temperature, dim=0)
        ) / float(batch_size)
        margin_weight = args.loss_margin_weight / float(num_decisions * batch_size)
        for decision_index, (parent, row, _, _, _) in enumerate(sample_records):
            ce_coefficients[parent][row] = ce_weights[decision_index]
            margin_coefficients[parent][row] = margin_weight

    sample_correct = [True] * batch_size
    for parent in ordered_parents:
        children = list(hierarchy.parent2children[parent])
        child_to_index = {child: index for index, child in enumerate(children)}
        entries = active[parent]
        sample_indices = torch.tensor(
            [sample_index for sample_index, _, _ in entries],
            dtype=torch.long,
            device=feature_leaf.device,
        )
        child_features = torch.nn.functional.normalize(
            learner.encode_children(parent, children).float(), dim=-1
        )
        logits = (
            feature_leaf.index_select(0, sample_indices)
            @ child_features.t()
            / float(args.tau)
        )
        target_tensor = torch.tensor(
            [child_to_index[child] for _, child, _ in entries],
            dtype=torch.long,
            device=feature_leaf.device,
        )
        ce_values = torch.nn.functional.cross_entropy(logits, target_tensor, reduction="none")
        weighted_loss = torch.sum(ce_coefficients[parent] * ce_values)

        if len(children) > 1 and args.loss_margin_weight > 0.0:
            negative_mask = torch.ones_like(logits, dtype=torch.bool)
            negative_mask.scatter_(1, target_tensor.unsqueeze(1), False)
            negative_logits = logits.masked_fill(~negative_mask, float("-inf"))
            margin_values = torch.nn.functional.softplus(
                torch.logsumexp(negative_logits, dim=1)
                - logits.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
                + args.loss_route_margin / float(args.tau)
            )
            weighted_loss = weighted_loss + torch.sum(
                margin_coefficients[parent] * margin_values
            )

        predictions = torch.argmax(logits.detach(), dim=1)
        for row, (sample_index, _, _) in enumerate(entries):
            sample_correct[sample_index] = sample_correct[sample_index] and (
                int(predictions[row]) == int(target_tensor[row])
            )
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(weighted_loss).backward()
        else:
            weighted_loss.backward()

    if adapted_features.requires_grad and feature_leaf.grad is not None:
        adapted_features.backward(feature_leaf.grad)
    path_correct = sum(int(correct) for correct in sample_correct)

    return {
        "loss": float(torch.stack(detached_sample_losses).mean().cpu()),
        "local_acc": local_correct / max(1, local_total),
        "path_acc": path_correct / max(1, batch_size),
        "active_parents": len(ordered_parents),
        "active_prompts": active_prompts,
        "peak_parent_prompts": max(len(hierarchy.parent2children[parent]) for parent in ordered_parents),
        "num_samples": batch_size,
        "num_decisions": local_total,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


@torch.no_grad()
def evaluate_positive(
    args,
    hierarchy,
    learner,
    image_adapter,
    features_dir: Path,
    device: str,
    split_names: tuple[str, ...] = ("val", "ood"),
    eval_mode: str | None = None,
) -> dict:
    dists_mats = make_distance_mats(hierarchy)
    eval_mode = eval_mode or args.eval_inference_mode
    semantic_index = build_idea3_semantic_index(hierarchy, learner, mode=eval_mode)
    image_adapter.eval()
    results = {
        "args": vars(args),
        "mode": eval_mode,
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
    }
    for split_name in split_names:
        payload = load_feature_file(features_dir / f"{split_name}-features.pt")
        preds = []
        stop_depth_counts = {}
        stop_node_counts = {}
        features = payload["features"]
        for start in range(0, int(features.shape[0]), args.inference_batch_size):
            end = min(start + args.inference_batch_size, int(features.shape[0]))
            image_features = image_adapter(features[start:end].to(device))
            out = predict_features_idea3(
                image_features,
                hierarchy,
                semantic_index,
                mode=eval_mode,
                tau=args.inference_tau,
            )
            preds.append(out["preds"].cpu())
            for key, value in out["diagnostics"]["stop_depth_counts"].items():
                stop_depth_counts[int(key)] = stop_depth_counts.get(int(key), 0) + int(value)
            for key, value in out["diagnostics"]["stop_node_counts"].items():
                stop_node_counts[str(key)] = stop_node_counts.get(str(key), 0) + int(value)
        preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
        node_labels, metrics = evaluate_split(hierarchy, payload, preds.cpu(), dists_mats=dists_mats)
        results[split_name] = {
            "preds": preds.cpu(),
            "targets": node_labels.cpu(),
            "metrics": metrics,
            "diagnostics": {
                "stop_depth_counts": dict(sorted(stop_depth_counts.items())),
                "stop_node_counts": dict(sorted(stop_node_counts.items(), key=lambda x: x[1], reverse=True)),
            },
        }
    if "val" in results and "ood" in results:
        results["mixed"] = mixed_summary(results["val"]["metrics"], results["ood"]["metrics"])
    return results


def main():
    args = parse_args()
    if not args.features_dir:
        raise ValueError("Missing features.dir or inference.features_dir in config")

    configure_reproducibility(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    features_dir = Path(args.features_dir)
    train_payload = load_feature_file(features_dir / "train-features.pt")
    train_features = train_payload["features"].float()
    image_dim = int(train_features.shape[1])
    examples = build_positive_edge_examples(hierarchy, train_payload)
    grouped = group_examples_by_parent(examples)
    parents = sorted(grouped)
    if not parents:
        raise RuntimeError("No positive edge examples were built from train-features.pt")

    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    smoke = soft_prompt_smoke_test(backend, max_length=int(args.prompt.get("max_length", 77)))
    if not smoke["ok"]:
        raise RuntimeError(f"Soft prompt smoke test failed: {smoke}")

    text_encoder = SoftPromptTextEncoder(
        backend.model,
        backend.processor.tokenizer,
        max_length=int(args.prompt.get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(args.prompt)
    learner = PositivePromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    adapter_cfg = ImageAdapterConfig.from_dict(args.image_adapter)
    image_adapter = build_image_adapter(image_dim, adapter_cfg.to_dict()).to(device)
    trainable_params = list(learner.parameters()) + [p for p in image_adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    print(
        "image adapter: "
        f"mode={adapter_cfg.mode}, input_dim={image_dim}, output_dim={image_adapter.output_dim}, "
        f"trainable_params={sum(p.numel() for p in image_adapter.parameters() if p.requires_grad)}"
    )
    rng = random.Random(args.seed)

    history = []
    sparse_path_names = {"sparse_path_bottleneck", "sparse_path", "spb"}
    use_sparse_path = args.loss_name.lower() in sparse_path_names
    use_depth_global = args.loss_prompt_scope == "depth_global" and args.loss_name.lower() != "ce"
    if args.loss_prompt_scope not in {"parent_local", "depth_global", "path_local"}:
        raise ValueError(
            "positive_training.loss.prompt_scope must be one of: parent_local, depth_global, path_local"
        )
    if use_sparse_path and args.loss_prompt_scope != "path_local":
        raise ValueError("sparse_path_bottleneck requires positive_training.loss.prompt_scope: path_local")

    if use_depth_global or use_sparse_path:
        node_labels = hierarchy.gen_ds2node_map(train_payload["classes"])[train_payload["targets"].long()]
        train_nodes = [hierarchy.id_node_list[int(idx)] for idx in node_labels.tolist()]

    if use_depth_global:
        depth_nodes = build_depth_prompt_sets(hierarchy)
        depth_prompt_metadata = build_depth_prompt_metadata(hierarchy, depth_nodes, device)
        train_path_labels = _variable_path_label_tensor(hierarchy, train_nodes, hierarchy.max_depth + 1, device="cpu")
        print(
            "using depth-global prompt metric loss: "
            + ", ".join(f"depth {depth}={len(nodes)} prompts" for depth, nodes in depth_nodes.items())
        )
    elif use_sparse_path:
        if args.loss_microbatch_size <= 0:
            raise ValueError("positive_training.loss.microbatch_size must be positive")
        if args.loss_execution not in {"parent_stream", "microbatch"}:
            raise ValueError("sparse path loss execution must be one of: parent_stream, microbatch")
        execution_detail = (
            f", microbatch_size={args.loss_microbatch_size}"
            if args.loss_execution == "microbatch"
            else ""
        )
        print(
            "using sparse path bottleneck loss: "
            f"batch_size={args.batch_size}, execution={args.loss_execution}, "
            f"bottleneck_weight={args.loss_bottleneck_weight}, "
            f"bottleneck_temperature={args.loss_bottleneck_temperature}, "
            f"route_margin={args.loss_route_margin}, margin_weight={args.loss_margin_weight}"
            f"{execution_detail}"
        )
    else:
        n_per_parent = max(1, args.batch_size // max(1, args.parents_per_step))

    if args.validation_metric not in {"balanced_acc", "balanced_hdist"}:
        raise ValueError(
            "positive_training.validation.metric must be one of: balanced_acc, balanced_hdist"
        )
    best_validation_score = float("-inf")
    best_validation = None
    best_positive_state = None
    best_adapter_state = None

    for epoch in range(1, args.epochs + 1):
        learner.train()
        image_adapter.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_local_acc = 0.0
        epoch_active_prompts = 0.0
        steps = 0

        if use_sparse_path:
            order = list(range(int(train_features.shape[0])))
            rng.shuffle(order)
            starts = range(0, len(order), args.batch_size)
            iterator = tqdm(starts, desc=f"positive epoch {epoch}/{args.epochs}", leave=False) if tqdm else starts
            for start in iterator:
                batch_indices = order[start:start + args.batch_size]
                if not batch_indices:
                    continue
                optimizer.zero_grad(set_to_none=True)
                batch_count = len(batch_indices)
                if args.loss_execution == "parent_stream":
                    index_tensor = torch.tensor(batch_indices, dtype=torch.long)
                    raw_image_features = train_features.index_select(0, index_tensor).to(device)
                    leaf_nodes = [train_nodes[index] for index in batch_indices]
                    stats = backward_sparse_path_bottleneck_streaming(
                        args,
                        hierarchy,
                        learner,
                        image_adapter,
                        raw_image_features,
                        leaf_nodes,
                    )
                    batch_loss = stats["loss"]
                    batch_path_acc = stats["path_acc"]
                    batch_local_acc = stats["local_acc"]
                    batch_active_prompts = stats["active_prompts"]
                elif args.loss_execution == "microbatch":
                    batch_loss = 0.0
                    batch_path_acc = 0.0
                    batch_local_acc = 0.0
                    batch_active_prompts = 0.0
                    for micro_start in range(0, batch_count, args.loss_microbatch_size):
                        micro_indices = batch_indices[micro_start:micro_start + args.loss_microbatch_size]
                        index_tensor = torch.tensor(micro_indices, dtype=torch.long)
                        image_features = image_adapter(train_features.index_select(0, index_tensor).to(device))
                        leaf_nodes = [train_nodes[index] for index in micro_indices]
                        loss, stats = compute_sparse_path_positive_loss(
                            args,
                            hierarchy,
                            learner,
                            image_features,
                            leaf_nodes,
                        )
                        micro_weight = len(micro_indices) / float(batch_count)
                        (loss * micro_weight).backward()
                        batch_loss += float(loss.detach().cpu()) * micro_weight
                        batch_path_acc += stats["path_acc"] * micro_weight
                        batch_local_acc += stats["local_acc"] * micro_weight
                        batch_active_prompts += stats["active_prompts"] * micro_weight
                else:
                    raise ValueError(
                        "sparse path loss execution must be one of: parent_stream, microbatch"
                    )

                optimizer.step()
                epoch_loss += batch_loss
                epoch_acc += batch_path_acc
                epoch_local_acc += batch_local_acc
                epoch_active_prompts += batch_active_prompts
                steps += 1
        elif use_depth_global:
            order = list(range(int(train_features.shape[0])))
            rng.shuffle(order)
            starts = range(0, len(order), args.batch_size)
            iterator = tqdm(starts, desc=f"positive epoch {epoch}/{args.epochs}", leave=False) if tqdm else starts
            for start in iterator:
                batch_indices = order[start:start + args.batch_size]
                if not batch_indices:
                    continue
                optimizer.zero_grad(set_to_none=True)
                index_tensor = torch.tensor(batch_indices, dtype=torch.long)
                image_features = image_adapter(train_features.index_select(0, index_tensor).to(device))
                image_path_labels = train_path_labels.index_select(0, index_tensor).to(device)
                prompt_features_by_depth = encode_depth_prompts(learner, hierarchy, depth_nodes)
                loss, stats = compute_depth_global_positive_loss(
                    args,
                    image_features,
                    image_path_labels,
                    prompt_features_by_depth,
                    depth_prompt_metadata["node_labels"],
                    depth_prompt_metadata["path_labels"],
                )
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())
                epoch_acc += stats["acc"]
                steps += 1
        else:
            shuffled = list(parents)
            rng.shuffle(shuffled)
            chunks = [
                shuffled[i:i + args.parents_per_step]
                for i in range(0, len(shuffled), args.parents_per_step)
            ]
            iterator = tqdm(chunks, desc=f"positive epoch {epoch}/{args.epochs}", leave=False) if tqdm else chunks

            for parent_chunk in iterator:
                optimizer.zero_grad(set_to_none=True)
                losses = []
                step_accs = []
                for parent in parent_chunk:
                    parent_examples = sample_examples(grouped[parent], n_per_parent, rng)
                    if not parent_examples:
                        continue
                    children = list(hierarchy.parent2children[parent])
                    child_to_idx = {child: idx for idx, child in enumerate(children)}
                    targets = torch.tensor([child_to_idx[ex.child] for ex in parent_examples], dtype=torch.long, device=device)
                    image_features = image_adapter(gather_image_features(train_features, parent_examples, device))
                    child_features = learner.encode_children(parent, children)
                    loss, stats = compute_positive_loss(
                        args,
                        hierarchy,
                        learner,
                        image_features,
                        child_features,
                        parent_examples,
                        children,
                        targets,
                        device,
                    )
                    losses.append(loss)
                    step_accs.append(stats["acc"])

                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())
                epoch_acc += sum(step_accs) / max(1, len(step_accs))
                steps += 1

        epoch_stats = {"epoch": epoch, "loss": epoch_loss / max(1, steps), "steps": steps}
        if use_sparse_path:
            epoch_stats.update({
                "path_acc": epoch_acc / max(1, steps),
                "local_acc": epoch_local_acc / max(1, steps),
                "mean_active_prompts": epoch_active_prompts / max(1, steps),
            })
        else:
            epoch_stats["depth_acc" if use_depth_global else "local_acc"] = epoch_acc / max(1, steps)

        validation_due = (
            args.validation_enabled
            and epoch >= args.validation_start_epoch
            and (epoch - args.validation_start_epoch) % args.validation_every_n_epochs == 0
        )
        is_best_epoch = False
        if validation_due:
            validation_result = evaluate_positive(
                args,
                hierarchy,
                learner,
                image_adapter,
                features_dir,
                device,
                split_names=("val",),
                eval_mode=args.validation_inference_mode,
            )
            val_metrics = validation_result["val"]["metrics"]
            val_bacc = float(val_metrics["balanced_acc"])
            val_bmhd = float(val_metrics["balanced_hdist"])
            epoch_stats.update({
                "val_balanced_acc": val_bacc,
                "val_balanced_hdist": val_bmhd,
                "val_inference_mode": args.validation_inference_mode,
            })
            validation_score = val_bacc if args.validation_metric == "balanced_acc" else -val_bmhd
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_validation = {
                    "epoch": epoch,
                    "metric": args.validation_metric,
                    "score": val_bacc if args.validation_metric == "balanced_acc" else val_bmhd,
                    "val_balanced_acc": val_bacc,
                    "val_balanced_hdist": val_bmhd,
                    "inference_mode": args.validation_inference_mode,
                }
                best_positive_state = state_dict_to_cpu(learner)
                best_adapter_state = state_dict_to_cpu(image_adapter)
                is_best_epoch = True

        history.append(epoch_stats)
        if use_sparse_path:
            message = (
                f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, "
                f"path_acc={epoch_stats['path_acc']:.6f}, "
                f"local_acc={epoch_stats['local_acc']:.6f}, "
                f"active_prompts={epoch_stats['mean_active_prompts']:.1f}"
            )
        else:
            acc_key = "depth_acc" if use_depth_global else "local_acc"
            message = f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, {acc_key}={epoch_stats[acc_key]:.6f}"
        if validation_due:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"val_bmhd={epoch_stats['val_balanced_hdist']:.6f}"
            )
            if is_best_epoch:
                message += " [best]"
        print(message)

        if is_best_epoch and args.validation_save_best:
            save_idea3_checkpoint(
                args.checkpoint,
                stage="positive",
                dataset=args.dataset,
                clip_model=args.clip_model,
                hierarchy=args.hierarchy,
                id_split=args.id_split,
                prompt_config=prompt_cfg.to_dict(),
                positive_state_dict=best_positive_state,
                image_adapter_config=adapter_cfg.to_dict(),
                image_adapter_state_dict=best_adapter_state,
                metrics={
                    "train_history": history,
                    "soft_prompt_smoke": smoke,
                    "best_validation": best_validation,
                },
                args=vars(args),
            )
            print(f"saved best checkpoint: {args.checkpoint}")

    metrics = {
        "train_history": history,
        "soft_prompt_smoke": smoke,
        "best_validation": best_validation,
    }

    if args.validation_enabled:
        last_ckpt_path = save_idea3_checkpoint(
            args.last_checkpoint,
            stage="positive",
            dataset=args.dataset,
            clip_model=args.clip_model,
            hierarchy=args.hierarchy,
            id_split=args.id_split,
            prompt_config=prompt_cfg.to_dict(),
            positive_state_dict=learner.state_dict(),
            image_adapter_config=adapter_cfg.to_dict(),
            image_adapter_state_dict=image_adapter.state_dict(),
            metrics=metrics,
            args=vars(args),
        )
        print(f"saved last checkpoint: {last_ckpt_path}")

        if best_positive_state is None or best_adapter_state is None:
            best_positive_state = state_dict_to_cpu(learner)
            best_adapter_state = state_dict_to_cpu(image_adapter)
            best_validation = {
                "epoch": args.epochs,
                "metric": args.validation_metric,
                "score": None,
                "val_balanced_acc": None,
                "val_balanced_hdist": None,
                "inference_mode": args.validation_inference_mode,
            }
            metrics["best_validation"] = best_validation
        learner.load_state_dict(best_positive_state)
        image_adapter.load_state_dict(best_adapter_state)
        print(f"restored best epoch: {best_validation['epoch']}")

    result = None
    if args.eval_after_training:
        result = evaluate_positive(args, hierarchy, learner, image_adapter, features_dir, device)
        ensure_dir(Path(args.result_path).parent)
        torch.save(result, args.result_path)
        metrics["final"] = {
            "val_balanced_acc": float(result["val"]["metrics"]["balanced_acc"]),
            "ood_balanced_acc": float(result["ood"]["metrics"]["balanced_acc"]),
            "mixed_balanced_acc": float(result["mixed"]["mixed_balanced_acc"]),
            "val_balanced_hdist": float(result["val"]["metrics"]["balanced_hdist"]),
            "ood_balanced_hdist": float(result["ood"]["metrics"]["balanced_hdist"]),
            "mixed_balanced_hdist": float(result["mixed"]["mixed_balanced_hdist"]),
        }
        print(f"saved result: {args.result_path}")

    ckpt_path = save_idea3_checkpoint(
        args.checkpoint,
        stage="positive",
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=learner.state_dict(),
        image_adapter_config=adapter_cfg.to_dict(),
        image_adapter_state_dict=image_adapter.state_dict(),
        metrics=metrics,
        args=vars(args),
    )
    save_json(args.diagnostics_path, metrics)
    if args.validation_enabled:
        print(f"saved best checkpoint: {ckpt_path}")
    else:
        print(f"saved checkpoint: {ckpt_path}")
    print(f"saved diagnostics: {args.diagnostics_path}")


if __name__ == "__main__":
    main()
