from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ProHOC.libs.utils.score_util import compprob, entcompprob
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hierarchical_support import (
    build_hierarchical_support_calibration,
    conformal_p_values,
    expected_hierarchy_distance_predictions,
    mondrian_support_p_values,
    node_support_p_values,
    stratified_reference_calibration_split,
)
from negzerohoc.multidepth_fusion import (
    fuse_multidepth_probabilities,
    get_multidepth_classes,
    multidepth_targets,
)
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
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
from scripts.train_paper_negprompt_ablation import json_ready


class MultiDepthLinearHeads(nn.Module):
    def __init__(self, feature_dim: int, class_counts: list[int]):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Linear(feature_dim, count) for count in class_counts
        )

    def forward(self, features: torch.Tensor) -> list[torch.Tensor]:
        features = F.normalize(features.float(), dim=-1)
        return [head(features) for head in self.heads]


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    train_cfg = cfg.get("multidepth_heads", {})
    inference_cfg = cfg.get("inference", {})
    experiment_name = str(
        experiment_cfg.get("name", "multidepth-feature-heads")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    support_checkpoint = train_cfg.get("support_checkpoint")
    if not support_checkpoint:
        raise ValueError("multidepth_heads.support_checkpoint is required")

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
        support_lora_enabled=bool(
            train_cfg.get("support_lora_enabled", True)
        ),
        train_fraction=float(train_cfg.get("train_fraction", 0.8)),
        epochs=max(1, int(train_cfg.get("epochs", 200))),
        batch_size=max(1, int(train_cfg.get("batch_size", 512))),
        lr=float(train_cfg.get("lr", 0.01)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        patience=max(1, int(train_cfg.get("patience", 30))),
        temperature_steps=max(
            1, int(train_cfg.get("temperature_steps", 300))
        ),
        alpha_grid=sorted(set(
            float(value)
            for value in train_cfg.get(
                "diagnostic_alpha_grid",
                [0.01, 0.025, 0.05, 0.075, 0.10],
            )
        )),
        fisher_id_acceptances=sorted(set(
            float(value)
            for value in train_cfg.get(
                "fisher_id_acceptances",
                [0.90, 0.925, 0.95, 0.975],
            )
        )),
        inference_batch_size=max(
            1, int(inference_cfg.get("batch_size", 128))
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
        checkpoint_path=artifact(
            train_cfg.get("checkpoint"),
            "checkpoints",
            f"{experiment_name}.pt",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


@torch.no_grad()
def validation_loss(
    model,
    features,
    targets_by_depth,
) -> float:
    model.eval()
    logits = model(features)
    return float(sum(
        F.cross_entropy(value, target)
        for value, target in zip(logits, targets_by_depth)
    ).cpu())


def train_heads(
    args,
    features,
    targets_by_depth,
    train_indices,
    calibration_indices,
    class_counts,
    device,
):
    model = MultiDepthLinearHeads(
        int(features.shape[1]), class_counts
    ).to(device)
    features = features.float().to(device)
    targets_by_depth = [
        value.long().to(device) for value in targets_by_depth
    ]
    train_indices = train_indices.long().to(device)
    calibration_indices = calibration_indices.long().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = train_indices.cpu()[
            torch.randperm(int(train_indices.numel()), generator=generator)
        ].to(device)
        running_loss = 0.0
        steps = 0
        for start in range(0, int(permutation.numel()), args.batch_size):
            index = permutation[start:start + args.batch_size]
            logits = model(features.index_select(0, index))
            loss = sum(
                F.cross_entropy(
                    value, target.index_select(0, index)
                )
                for value, target in zip(logits, targets_by_depth)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            steps += 1
        cal_loss = validation_loss(
            model,
            features.index_select(0, calibration_indices),
            [
                target.index_select(0, calibration_indices)
                for target in targets_by_depth
            ],
        )
        if cal_loss < best_loss - 1e-7:
            best_loss = cal_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            history.append({
                "epoch": epoch,
                "train_loss": running_loss / max(1, steps),
                "calibration_loss": cal_loss,
            })
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("Multi-depth head training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "completed_epochs": epoch,
        "history": history,
    }


@torch.no_grad()
def payload_logits(model, payload, device):
    model.eval()
    features = payload["features"].float()
    chunks = [[] for _ in model.heads]
    for start in range(0, int(features.shape[0]), 1024):
        output = model(
            features[start:start + 1024].to(device)
        )
        for depth, value in enumerate(output):
            chunks[depth].append(value.float().cpu())
    return [torch.cat(values) for values in chunks]


def fit_temperatures(
    args,
    calibration_logits,
    calibration_targets,
    device,
):
    logits = [value.float().to(device) for value in calibration_logits]
    targets = [value.long().to(device) for value in calibration_targets]
    log_temperatures = nn.Parameter(
        torch.zeros(len(logits), device=device)
    )
    optimizer = torch.optim.Adam([log_temperatures], lr=0.05)
    for _ in range(args.temperature_steps):
        temperatures = log_temperatures.exp().clamp(0.05, 20.0)
        loss = sum(
            F.cross_entropy(value / temperatures[depth], targets[depth])
            for depth, value in enumerate(logits)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return (
        log_temperatures.detach().exp().clamp(0.05, 20.0).cpu()
    )


def probability_list(logits, temperatures=None):
    if temperatures is None:
        temperatures = torch.ones(len(logits))
    return [
        F.softmax(value.float() / float(temperatures[depth]), dim=1)
        for depth, value in enumerate(logits)
    ]


def predictions_from_leaf_probabilities(
    probabilities,
    hierarchy,
    multidepth_classes,
):
    leaf_nodes = multidepth_classes[-1]
    winners = probabilities[-1].argmax(dim=1)
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    return torch.tensor([
        node_to_index[leaf_nodes[int(index)]]
        for index in winners.tolist()
    ])


def evaluate_prediction_pair(
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


def fusion_results(
    hierarchy,
    multidepth_classes,
    id_probabilities,
    ood_probabilities,
    id_payload,
    ood_payload,
    dists_mats,
):
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    result = {}
    for name, method in (
        ("compprob", compprob),
        ("entcompprob", entcompprob),
    ):
        id_fused = fuse_multidepth_probabilities(
            id_probabilities, hierarchy, multidepth_classes, method
        )
        ood_fused = fuse_multidepth_probabilities(
            ood_probabilities, hierarchy, multidepth_classes, method
        )
        for decoder in ("map", "expected_hdist"):
            if decoder == "map":
                id_predictions = id_fused.argmax(dim=1)
                ood_predictions = ood_fused.argmax(dim=1)
            else:
                id_predictions = expected_hierarchy_distance_predictions(
                    id_fused, distance_matrix
                )
                ood_predictions = expected_hierarchy_distance_predictions(
                    ood_fused, distance_matrix
                )
            row = evaluate_prediction_pair(
                hierarchy,
                id_payload,
                ood_payload,
                id_predictions,
                ood_predictions,
                dists_mats,
            )
            row["id_probability_sum_range"] = [
                float(id_fused.sum(dim=1).min()),
                float(id_fused.sum(dim=1).max()),
            ]
            row["ood_probability_sum_range"] = [
                float(ood_fused.sum(dim=1).min()),
                float(ood_fused.sum(dim=1).max()),
            ]
            result[f"{name}_{decoder}"] = row
    return result


def density_gated_multidepth_results(
    hierarchy,
    multidepth_classes,
    id_probabilities,
    ood_probabilities,
    id_support_p_values,
    ood_support_p_values,
    id_payload,
    ood_payload,
    dists_mats,
    alpha_grid,
    gate_name,
):
    leaf_id_predictions = predictions_from_leaf_probabilities(
        id_probabilities, hierarchy, multidepth_classes
    )
    leaf_ood_predictions = predictions_from_leaf_probabilities(
        ood_probabilities, hierarchy, multidepth_classes
    )
    internal_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node != "root" and node in hierarchy.parent2children
    ])
    result = {}
    for method_name, method in (
        ("compprob", compprob),
        ("entcompprob", entcompprob),
    ):
        id_fused = fuse_multidepth_probabilities(
            id_probabilities, hierarchy, multidepth_classes, method
        )
        ood_fused = fuse_multidepth_probabilities(
            ood_probabilities, hierarchy, multidepth_classes, method
        )
        id_internal = internal_indices[
            id_fused[:, internal_indices].argmax(dim=1)
        ]
        ood_internal = internal_indices[
            ood_fused[:, internal_indices].argmax(dim=1)
        ]
        method_rows = {}
        for alpha in alpha_grid:
            id_reject = (
                id_support_p_values["root"] <= float(alpha)
            )
            ood_reject = (
                ood_support_p_values["root"] <= float(alpha)
            )
            id_predictions = leaf_id_predictions.clone()
            ood_predictions = leaf_ood_predictions.clone()
            id_predictions[id_reject] = id_internal[id_reject]
            ood_predictions[ood_reject] = ood_internal[ood_reject]
            row = evaluate_prediction_pair(
                hierarchy,
                id_payload,
                ood_payload,
                id_predictions,
                ood_predictions,
                dists_mats,
            )
            row["id_rejection_rate"] = float(
                id_reject.float().mean()
            )
            row["ood_rejection_rate"] = float(
                ood_reject.float().mean()
            )
            method_rows[str(alpha)] = row
        result[f"{gate_name}_{method_name}"] = method_rows

    node_to_depth_position = {}
    for depth, nodes in enumerate(multidepth_classes):
        for position, node in enumerate(nodes):
            if (
                node != "root"
                and node in hierarchy.parent2children
                and node not in node_to_depth_position
            ):
                node_to_depth_position[node] = (depth, position)
    internal_nodes = sorted(
        node_to_depth_position,
        key=lambda node: (
            len(hierarchy.node_ancestors.get(node, [])),
            node,
        ),
    )

    def residual_predictions(probabilities, support_values):
        columns = []
        for node in internal_nodes:
            depth, position = node_to_depth_position[node]
            route_reach = probabilities[depth][:, position]
            novelty = 1.0 - support_values[node]
            columns.append(route_reach * novelty)
        scores = torch.stack(columns, dim=1)
        winners = scores.argmax(dim=1)
        return torch.tensor([
            hierarchy.id_node_list.index(internal_nodes[int(index)])
            for index in winners.tolist()
        ])

    id_residual_parent = residual_predictions(
        id_probabilities, id_support_p_values
    )
    ood_residual_parent = residual_predictions(
        ood_probabilities, ood_support_p_values
    )
    residual_rows = {}
    for alpha in alpha_grid:
        id_reject = id_support_p_values["root"] <= float(alpha)
        ood_reject = ood_support_p_values["root"] <= float(alpha)
        id_predictions = leaf_id_predictions.clone()
        ood_predictions = leaf_ood_predictions.clone()
        id_predictions[id_reject] = id_residual_parent[id_reject]
        ood_predictions[ood_reject] = ood_residual_parent[ood_reject]
        row = evaluate_prediction_pair(
            hierarchy,
            id_payload,
            ood_payload,
            id_predictions,
            ood_predictions,
            dists_mats,
        )
        row["id_rejection_rate"] = float(id_reject.float().mean())
        row["ood_rejection_rate"] = float(
            ood_reject.float().mean()
        )
        residual_rows[str(alpha)] = row
    result[f"{gate_name}_residual_density"] = residual_rows
    return result


def fisher_combined_gate_results(
    hierarchy,
    multidepth_classes,
    calibration_probabilities,
    id_probabilities,
    ood_probabilities,
    calibration_density_p,
    id_density_p,
    ood_density_p,
    id_payload,
    ood_payload,
    dists_mats,
    id_acceptances,
):
    calibration_leaf_confidence = (
        calibration_probabilities[-1].max(dim=1).values
    )
    id_leaf_confidence = id_probabilities[-1].max(dim=1).values
    ood_leaf_confidence = ood_probabilities[-1].max(dim=1).values
    calibration_confidence_p = conformal_p_values(
        calibration_leaf_confidence,
        calibration_leaf_confidence,
    )
    id_confidence_p = conformal_p_values(
        id_leaf_confidence, calibration_leaf_confidence
    )
    ood_confidence_p = conformal_p_values(
        ood_leaf_confidence, calibration_leaf_confidence
    )

    def fisher(density_p, confidence_p):
        return -2.0 * (
            density_p.clamp_min(1e-12).log()
            + confidence_p.clamp_min(1e-12).log()
        )

    calibration_scores = fisher(
        calibration_density_p, calibration_confidence_p
    )
    id_scores = fisher(id_density_p, id_confidence_p)
    ood_scores = fisher(ood_density_p, ood_confidence_p)

    id_leaf_predictions = predictions_from_leaf_probabilities(
        id_probabilities, hierarchy, multidepth_classes
    )
    ood_leaf_predictions = predictions_from_leaf_probabilities(
        ood_probabilities, hierarchy, multidepth_classes
    )
    id_fused = fuse_multidepth_probabilities(
        id_probabilities,
        hierarchy,
        multidepth_classes,
        entcompprob,
    )
    ood_fused = fuse_multidepth_probabilities(
        ood_probabilities,
        hierarchy,
        multidepth_classes,
        entcompprob,
    )
    internal_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node != "root" and node in hierarchy.parent2children
    ])
    id_parent_predictions = internal_indices[
        id_fused[:, internal_indices].argmax(dim=1)
    ]
    ood_parent_predictions = internal_indices[
        ood_fused[:, internal_indices].argmax(dim=1)
    ]
    rows = {}
    for acceptance in id_acceptances:
        threshold = float(torch.quantile(
            calibration_scores, float(acceptance)
        ))
        id_reject = id_scores > threshold
        ood_reject = ood_scores > threshold
        id_predictions = id_leaf_predictions.clone()
        ood_predictions = ood_leaf_predictions.clone()
        id_predictions[id_reject] = id_parent_predictions[id_reject]
        ood_predictions[ood_reject] = ood_parent_predictions[ood_reject]
        row = evaluate_prediction_pair(
            hierarchy,
            id_payload,
            ood_payload,
            id_predictions,
            ood_predictions,
            dists_mats,
        )
        row.update({
            "train_id_acceptance_target": float(acceptance),
            "train_calibration_threshold": threshold,
            "id_rejection_rate": float(id_reject.float().mean()),
            "ood_rejection_rate": float(ood_reject.float().mean()),
        })
        rows[str(acceptance)] = row
    return {
        "binary_ood": binary_ood_metrics(
            id_scores.numpy(), ood_scores.numpy()
        ),
        "rows": rows,
    }


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
    train_payload = encode_dataset_features(
        args,
        clip_model,
        train_dataset,
        make_eval_loader(args, train_dataset),
        device,
        "encode multi-depth ID train",
    )
    id_payload = encode_dataset_features(
        args,
        clip_model,
        id_dataset,
        make_eval_loader(args, id_dataset),
        device,
        "encode multi-depth ID test",
    )
    ood_payload = encode_dataset_features(
        args,
        clip_model,
        ood_dataset,
        make_eval_loader(args, ood_dataset),
        device,
        "encode multi-depth OOD test",
    )
    release_cuda(clip_model)

    leaf_classes = [
        hierarchy.id_node_list[int(index)]
        for index in hierarchy.gen_ds2node_map(
            train_payload["classes"]
        ).tolist()
    ]
    multidepth_classes = get_multidepth_classes(
        hierarchy, leaf_classes
    )
    train_targets = multidepth_targets(
        hierarchy,
        train_payload["classes"],
        train_payload["targets"],
        multidepth_classes,
    )
    train_indices, calibration_indices = (
        stratified_reference_calibration_split(
            train_payload["targets"],
            reference_fraction=args.train_fraction,
            seed=args.seed,
        )
    )
    support_calibration = build_hierarchical_support_calibration(
        hierarchy,
        train_payload["features"],
        train_payload["classes"],
        train_payload["targets"],
        reference_fraction=args.train_fraction,
        seed=args.seed,
    )
    id_support_p_values = node_support_p_values(
        id_payload["features"], support_calibration
    )
    ood_support_p_values = node_support_p_values(
        ood_payload["features"], support_calibration
    )
    calibration_support_p_values = node_support_p_values(
        train_payload["features"].index_select(
            0, calibration_indices
        ),
        support_calibration,
    )
    id_mondrian_support_p_values = dict(id_support_p_values)
    ood_mondrian_support_p_values = dict(ood_support_p_values)
    id_mondrian_support_p_values["root"] = (
        mondrian_support_p_values(
            id_payload["features"], support_calibration
        )
    )
    ood_mondrian_support_p_values["root"] = (
        mondrian_support_p_values(
            ood_payload["features"], support_calibration
        )
    )
    model, training = train_heads(
        args,
        train_payload["features"],
        train_targets,
        train_indices,
        calibration_indices,
        [len(nodes) for nodes in multidepth_classes],
        device,
    )
    train_logits = payload_logits(model, train_payload, device)
    id_logits = payload_logits(model, id_payload, device)
    ood_logits = payload_logits(model, ood_payload, device)
    calibration_logits = [
        value.index_select(0, calibration_indices)
        for value in train_logits
    ]
    calibration_targets = [
        value.index_select(0, calibration_indices)
        for value in train_targets
    ]
    temperatures = fit_temperatures(
        args,
        calibration_logits,
        calibration_targets,
        device,
    )
    dists_mats = make_distance_mats(hierarchy)
    results = {}
    for calibration_name, used_temperatures in (
        ("uncalibrated", None),
        ("id_train_temperature", temperatures),
    ):
        id_probabilities = probability_list(
            id_logits, used_temperatures
        )
        ood_probabilities = probability_list(
            ood_logits, used_temperatures
        )
        calibration_probabilities = [
            value.index_select(0, calibration_indices)
            for value in probability_list(
                train_logits, used_temperatures
            )
        ]
        leaf_id_predictions = predictions_from_leaf_probabilities(
            id_probabilities, hierarchy, multidepth_classes
        )
        leaf_ood_predictions = predictions_from_leaf_probabilities(
            ood_probabilities, hierarchy, multidepth_classes
        )
        rows = {
            "leaf": evaluate_prediction_pair(
                hierarchy,
                id_payload,
                ood_payload,
                leaf_id_predictions,
                leaf_ood_predictions,
                dists_mats,
            )
        }
        rows.update(fusion_results(
            hierarchy,
            multidepth_classes,
            id_probabilities,
            ood_probabilities,
            id_payload,
            ood_payload,
            dists_mats,
        ))
        density_rows = density_gated_multidepth_results(
                hierarchy,
                multidepth_classes,
                id_probabilities,
                ood_probabilities,
                id_support_p_values,
                ood_support_p_values,
                id_payload,
                ood_payload,
                dists_mats,
                args.alpha_grid,
                "pooled",
        )
        density_rows.update(density_gated_multidepth_results(
            hierarchy,
            multidepth_classes,
            id_probabilities,
            ood_probabilities,
            id_mondrian_support_p_values,
            ood_mondrian_support_p_values,
            id_payload,
            ood_payload,
            dists_mats,
            args.alpha_grid,
            "mondrian",
        ))
        rows["density_gated_localizer"] = density_rows
        rows["fisher_combined_gate"] = fisher_combined_gate_results(
            hierarchy,
            multidepth_classes,
            calibration_probabilities,
            id_probabilities,
            ood_probabilities,
            calibration_support_p_values["root"],
            id_support_p_values["root"],
            ood_support_p_values["root"],
            id_payload,
            ood_payload,
            dists_mats,
            args.fisher_id_acceptances,
        )
        results[calibration_name] = rows

    result = {
        "method": "frozen_feature_multidepth_prohoc_fusion",
        "used_actual_ood_for_training_calibration_or_selection": False,
        "official_test_used_only_after_id_train_internal_selection": True,
        "support_checkpoint": args.support_checkpoint,
        "support_checkpoint_stage": support_checkpoint.get("stage"),
        "support_lora_enabled": args.support_lora_enabled,
        "multidepth_classes": multidepth_classes,
        "train_samples": int(train_indices.numel()),
        "calibration_samples": int(calibration_indices.numel()),
        "training": training,
        "temperatures": temperatures,
        "density_gate_alpha_grid_diagnostic_not_for_selection": (
            args.alpha_grid
        ),
        "fisher_id_acceptances": args.fisher_id_acceptances,
        "results": results,
    }
    checkpoint_path = Path(args.checkpoint_path)
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(result_path.parent)
    ensure_dir(diagnostics_path.parent)
    torch.save({
        "stage": "frozen_feature_multidepth_heads",
        "support_checkpoint": args.support_checkpoint,
        "multidepth_classes": multidepth_classes,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "temperatures": temperatures,
        "training": training,
    }, checkpoint_path)
    torch.save(result, result_path)
    save_json(diagnostics_path, json_ready(result))
    for calibration_name, rows in results.items():
        for method_name, row in rows.items():
            if method_name == "density_gated_localizer":
                for localizer_name, alpha_rows in row.items():
                    for alpha, alpha_row in alpha_rows.items():
                        print(
                            f"{calibration_name}/density_gate_"
                            f"{localizer_name}@{alpha}: "
                            f"ID/OOD/Mix="
                            f"{float(alpha_row['id']['balanced_acc']):.6f}/"
                            f"{float(alpha_row['ood']['balanced_acc']):.6f}/"
                            f"{float(alpha_row['mixed']['mixed_balanced_acc']):.6f}, "
                            f"Mix BMHD="
                            f"{float(alpha_row['mixed']['mixed_balanced_hdist']):.6f}"
                        )
                continue
            if method_name == "fisher_combined_gate":
                print(
                    f"{calibration_name}/fisher binary="
                    f"{row['binary_ood']}"
                )
                for acceptance, fisher_row in row["rows"].items():
                    print(
                        f"{calibration_name}/fisher@{acceptance}: "
                        f"ID/OOD/Mix="
                        f"{float(fisher_row['id']['balanced_acc']):.6f}/"
                        f"{float(fisher_row['ood']['balanced_acc']):.6f}/"
                        f"{float(fisher_row['mixed']['mixed_balanced_acc']):.6f}, "
                        f"Mix BMHD="
                        f"{float(fisher_row['mixed']['mixed_balanced_hdist']):.6f}"
                    )
                continue
            print(
                f"{calibration_name}/{method_name}: "
                f"ID/OOD/Mix="
                f"{float(row['id']['balanced_acc']):.6f}/"
                f"{float(row['ood']['balanced_acc']):.6f}/"
                f"{float(row['mixed']['mixed_balanced_acc']):.6f}, "
                f"Mix BMHD="
                f"{float(row['mixed']['mixed_balanced_hdist']):.6f}"
            )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
