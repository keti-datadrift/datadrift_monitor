from __future__ import annotations

import argparse
import gc
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
    node_labels_from_feature_targets,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hierarchical_support import (
    build_hierarchical_support_calibration,
    expected_hierarchy_distance_predictions,
    factorized_terminal_probabilities,
    global_gate_route_stop_predictions,
    mondrian_support_p_values,
    nearest_support_prototype_predictions,
    node_support_p_values,
    positive_route_conditionals,
)
from negzerohoc.idea3_inference import predict_features_idea3
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_enabled,
    set_vision_lora_train_mode,
)
from scripts.train_idea3_joint_vision_lora import (
    build_transforms,
    load_clip_and_tokenizer,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import (
    build_positive_semantic_index,
    encode_dataset_features,
    freeze_module,
    load_frozen_positive_stack,
)
from scripts.train_paper_negprompt_ablation import json_ready


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    method_cfg = cfg.get("dual_expert_support", {})
    inference_cfg = cfg.get("inference", {})
    experiment_name = str(
        experiment_cfg.get("name", "dual-expert-hierarchical-support")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))

    routing_checkpoint = method_cfg.get("routing_checkpoint")
    support_checkpoint = method_cfg.get("support_checkpoint")
    if not routing_checkpoint or not support_checkpoint:
        raise ValueError(
            "dual_expert_support.routing_checkpoint and support_checkpoint "
            "are required"
        )
    alpha = float(method_cfg.get("alpha", 0.05))
    alpha_grid = [
        float(value)
        for value in method_cfg.get(
            "diagnostic_alpha_grid", [0.01, 0.025, 0.05, 0.075, 0.10]
        )
    ]
    if alpha not in alpha_grid:
        alpha_grid.append(alpha)
    alpha_grid = sorted(set(alpha_grid))

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
        precision=str(method_cfg.get("precision", "fp16")).lower(),
        routing_checkpoint=str(routing_checkpoint),
        positive_checkpoint=str(routing_checkpoint),
        support_checkpoint=str(support_checkpoint),
        support_lora_enabled=bool(
            method_cfg.get("support_lora_enabled", True)
        ),
        reference_fraction=float(method_cfg.get("reference_fraction", 0.8)),
        alpha=alpha,
        alpha_grid=alpha_grid,
        inference_batch_size=max(
            1, int(inference_cfg.get("batch_size", 128))
        ),
        inference_tau=float(inference_cfg.get("tau", 14.285714)),
        result_path=artifact(
            method_cfg.get("result_path"),
            "results",
            f"{experiment_name}.result",
        ),
        diagnostics_path=artifact(
            method_cfg.get("diagnostics_path"),
            "diagnostics",
            f"{experiment_name}.json",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def build_eval_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import (
        SubsetImageFolder,
        get_id_classes,
    )

    _, transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    train_dataset = SubsetImageFolder(
        datadir / "train", id_classes, transform=transform
    )
    id_dataset = SubsetImageFolder(
        datadir / "val", id_classes, transform=transform
    )
    ood_dataset = SubsetImageFolder(
        datadir / "val", hierarchy.ood_train_classes, transform=transform
    )
    return train_dataset, id_dataset, ood_dataset


def make_eval_loader(args, dataset):
    return make_loader(
        dataset,
        args.inference_batch_size,
        args.num_workers,
        False,
        args.seed,
    )


def validate_vision_checkpoint(args, checkpoint):
    expected = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
    }
    mismatches = {
        key: (checkpoint.get(key), expected_value)
        for key, expected_value in expected.items()
        if checkpoint.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Support checkpoint/config mismatch: {mismatches}")
    for key in ("vision_lora_config", "vision_lora_state_dict"):
        if not checkpoint.get(key):
            raise ValueError(f"Support checkpoint is missing {key}")


def load_frozen_vision(args, checkpoint_path: str, device: str):
    checkpoint = load_idea3_checkpoint(checkpoint_path, map_location="cpu")
    validate_vision_checkpoint(args, checkpoint)
    clip_model, _ = load_clip_and_tokenizer(args, device)
    lora_cfg = VisionLoRAConfig.from_dict(
        checkpoint["vision_lora_config"]
    )
    inject_clip_vision_lora(clip_model, lora_cfg)
    load_vision_lora_state_dict(
        clip_model, checkpoint["vision_lora_state_dict"]
    )
    set_vision_lora_enabled(
        clip_model, bool(args.support_lora_enabled)
    )
    freeze_module(clip_model)
    set_vision_lora_train_mode(clip_model, False)
    return checkpoint, clip_model


def release_cuda(*objects):
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_aligned(first, second, name: str):
    if first["classes"] != second["classes"]:
        raise RuntimeError(f"{name} payload class order differs between experts")
    if not torch.equal(first["targets"], second["targets"]):
        raise RuntimeError(f"{name} payload target order differs between experts")


def evaluate_predictions(hierarchy, payload, predictions, dists_mats):
    _, metrics = evaluate_split(
        hierarchy, payload, predictions, dists_mats=dists_mats
    )
    return metrics


def metric_row(
    hierarchy,
    id_payload,
    ood_payload,
    id_predictions,
    ood_predictions,
    dists_mats,
):
    id_metrics = evaluate_predictions(
        hierarchy, id_payload, id_predictions, dists_mats
    )
    ood_metrics = evaluate_predictions(
        hierarchy, ood_payload, ood_predictions, dists_mats
    )
    return {
        "id": id_metrics,
        "ood": ood_metrics,
        "mixed": mixed_summary(id_metrics, ood_metrics),
    }


def routing_predictions(features, hierarchy, semantic_index, args):
    return predict_features_idea3(
        features,
        hierarchy,
        semantic_index,
        mode="positive_global_path",
        tau=args.inference_tau,
    )["preds"]


def factorized_rows(
    hierarchy,
    id_route,
    ood_route,
    id_support,
    ood_support,
    alpha,
    dists_mats,
    id_payload,
    ood_payload,
):
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    rows = {}
    for gate in ("hard", "conformal_ramp"):
        id_probabilities = factorized_terminal_probabilities(
            hierarchy, id_route, id_support, alpha=alpha, gate=gate
        )
        ood_probabilities = factorized_terminal_probabilities(
            hierarchy, ood_route, ood_support, alpha=alpha, gate=gate
        )
        for decoder in ("map", "expected_hdist"):
            if decoder == "map":
                id_predictions = id_probabilities.argmax(dim=1)
                ood_predictions = ood_probabilities.argmax(dim=1)
            else:
                id_predictions = expected_hierarchy_distance_predictions(
                    id_probabilities, distance_matrix
                )
                ood_predictions = expected_hierarchy_distance_predictions(
                    ood_probabilities, distance_matrix
                )
            row = metric_row(
                hierarchy,
                id_payload,
                ood_payload,
                id_predictions,
                ood_predictions,
                dists_mats,
            )
            row["id_probability_sum_range"] = [
                float(id_probabilities.sum(dim=1).min()),
                float(id_probabilities.sum(dim=1).max()),
            ]
            row["ood_probability_sum_range"] = [
                float(ood_probabilities.sum(dim=1).min()),
                float(ood_probabilities.sum(dim=1).max()),
            ]
            rows[f"{gate}_{decoder}"] = row
    return rows


def localizer_diagnostics(
    hierarchy,
    ood_payload,
    predictions,
    positive_leaf_predictions,
    rejected,
):
    targets = node_labels_from_feature_targets(
        hierarchy, ood_payload["classes"], ood_payload["targets"]
    )
    route_contains_target = []
    for target_index, leaf_index in zip(
        targets.tolist(), positive_leaf_predictions.tolist()
    ):
        target = hierarchy.id_node_list[int(target_index)]
        leaf = hierarchy.id_node_list[int(leaf_index)]
        route = [
            hierarchy.id_node_list[int(index)]
            for index in hierarchy.node_ancestors.get(leaf, [])
        ] + [leaf]
        route_contains_target.append(target in route)
    route_contains_target = torch.tensor(route_contains_target)
    rejected = rejected.bool().cpu()
    return {
        "positive_route_contains_true_parent": float(
            route_contains_target.float().mean()
        ),
        "rejected_count": int(rejected.sum()),
        "rejected_parent_accuracy": (
            float((predictions[rejected] == targets[rejected]).float().mean())
            if bool(rejected.any())
            else 0.0
        ),
        "route_contains_true_parent_among_rejected": (
            float(route_contains_target[rejected].float().mean())
            if bool(rejected.any())
            else 0.0
        ),
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
    print(
        f"# train={len(train_dataset)}, ID test={len(id_dataset)}, "
        f"OOD test={len(ood_dataset)}"
    )

    (
        routing_checkpoint,
        routing_clip,
        routing_text_encoder,
        _routing_prompt_cfg,
        routing_positive,
        _routing_replaced,
    ) = load_frozen_positive_stack(args, hierarchy, device)
    semantic_index = build_positive_semantic_index(
        hierarchy, routing_positive
    )
    id_route_payload = encode_dataset_features(
        args,
        routing_clip,
        id_dataset,
        make_eval_loader(args, id_dataset),
        device,
        "encode routing ID test",
    )
    ood_route_payload = encode_dataset_features(
        args,
        routing_clip,
        ood_dataset,
        make_eval_loader(args, ood_dataset),
        device,
        "encode routing OOD test",
    )
    release_cuda(
        routing_positive, routing_text_encoder, routing_clip
    )

    support_checkpoint, support_clip = load_frozen_vision(
        args, args.support_checkpoint, device
    )
    train_support_payload = encode_dataset_features(
        args,
        support_clip,
        train_dataset,
        make_eval_loader(args, train_dataset),
        device,
        "encode support ID train",
    )
    id_support_payload = encode_dataset_features(
        args,
        support_clip,
        id_dataset,
        make_eval_loader(args, id_dataset),
        device,
        "encode support ID test",
    )
    ood_support_payload = encode_dataset_features(
        args,
        support_clip,
        ood_dataset,
        make_eval_loader(args, ood_dataset),
        device,
        "encode support OOD test",
    )
    release_cuda(support_clip)
    ensure_aligned(id_route_payload, id_support_payload, "ID")
    ensure_aligned(ood_route_payload, ood_support_payload, "OOD")

    calibration = build_hierarchical_support_calibration(
        hierarchy,
        train_support_payload["features"],
        train_support_payload["classes"],
        train_support_payload["targets"],
        reference_fraction=args.reference_fraction,
        seed=args.seed,
    )
    id_support = node_support_p_values(
        id_support_payload["features"], calibration
    )
    ood_support = node_support_p_values(
        ood_support_payload["features"], calibration
    )
    id_support_mondrian = dict(id_support)
    ood_support_mondrian = dict(ood_support)
    id_support_mondrian["root"] = mondrian_support_p_values(
        id_support_payload["features"], calibration
    )
    ood_support_mondrian["root"] = mondrian_support_p_values(
        ood_support_payload["features"], calibration
    )
    id_routes = positive_route_conditionals(
        id_route_payload["features"],
        hierarchy,
        semantic_index,
        logit_scale=args.inference_tau,
    )
    ood_routes = positive_route_conditionals(
        ood_route_payload["features"],
        hierarchy,
        semantic_index,
        logit_scale=args.inference_tau,
    )
    id_positive_predictions = routing_predictions(
        id_route_payload["features"], hierarchy, semantic_index, args
    )
    ood_positive_predictions = routing_predictions(
        ood_route_payload["features"], hierarchy, semantic_index, args
    )
    id_support_route_predictions = (
        nearest_support_prototype_predictions(
            id_support_payload["features"], calibration, hierarchy
        )
    )
    ood_support_route_predictions = (
        nearest_support_prototype_predictions(
            ood_support_payload["features"], calibration, hierarchy
        )
    )
    dists_mats = make_distance_mats(hierarchy)

    positive_baseline = metric_row(
        hierarchy,
        id_route_payload,
        ood_route_payload,
        id_positive_predictions,
        ood_positive_predictions,
        dists_mats,
    )
    root_binary = binary_ood_metrics(
        (-id_support["root"]).numpy(),
        (-ood_support["root"]).numpy(),
    )
    grid = {}
    primary_localizer_diagnostics = {}
    for alpha in args.alpha_grid:
        alpha_rows = {}
        for gate_name, id_gate_support, ood_gate_support in (
            ("pooled", id_support, ood_support),
            (
                "mondrian",
                id_support_mondrian,
                ood_support_mondrian,
            ),
        ):
            for route_name, id_route_predictions, ood_route_predictions in (
                (
                    "positive_route",
                    id_positive_predictions,
                    ood_positive_predictions,
                ),
                (
                    "support_route",
                    id_support_route_predictions,
                    ood_support_route_predictions,
                ),
            ):
                for localizer in (
                    "first_unsupported",
                    "weakest_support",
                    "deepest",
                ):
                    id_predictions, id_diag = (
                        global_gate_route_stop_predictions(
                            hierarchy,
                            id_route_predictions,
                            id_gate_support,
                            alpha=alpha,
                            localizer=localizer,
                        )
                    )
                    ood_predictions, ood_diag = (
                        global_gate_route_stop_predictions(
                            hierarchy,
                            ood_route_predictions,
                            ood_gate_support,
                            alpha=alpha,
                            localizer=localizer,
                        )
                    )
                    id_rejected = (
                        id_diag["root_support_p_values"] <= float(alpha)
                    )
                    ood_rejected = (
                        ood_diag["root_support_p_values"] <= float(alpha)
                    )
                    if route_name == "support_route":
                        id_predictions[~id_rejected] = (
                            id_positive_predictions[~id_rejected]
                        )
                        ood_predictions[~ood_rejected] = (
                            ood_positive_predictions[~ood_rejected]
                        )
                    row = metric_row(
                        hierarchy,
                        id_route_payload,
                        ood_route_payload,
                        id_predictions,
                        ood_predictions,
                        dists_mats,
                    )
                    row["id_rejection_rate"] = id_diag["rejection_rate"]
                    row["ood_rejection_rate"] = (
                        ood_diag["rejection_rate"]
                    )
                    key = f"{gate_name}_{route_name}_{localizer}"
                    row["localizer_diagnostics"] = localizer_diagnostics(
                        hierarchy,
                        ood_route_payload,
                        ood_predictions,
                        ood_route_predictions,
                        ood_rejected,
                    )
                    alpha_rows[key] = row
                    if (
                        gate_name == "pooled"
                        and route_name == "positive_route"
                        and alpha == args.alpha
                        and localizer == "first_unsupported"
                    ):
                        primary_localizer_diagnostics = (
                            row["localizer_diagnostics"]
                        )
                        # Preserve the original short key as the locked
                        # primary protocol for result compatibility.
                        alpha_rows[localizer] = row
        alpha_rows["factorized"] = factorized_rows(
            hierarchy,
            id_routes,
            ood_routes,
            id_support,
            ood_support,
            alpha,
            dists_mats,
            id_route_payload,
            ood_route_payload,
        )
        grid[str(alpha)] = alpha_rows

    primary = grid[str(args.alpha)]["first_unsupported"]
    result = {
        "method": "dual_expert_hierarchical_support_factorization",
        "used_actual_ood_for_training_calibration_or_selection": False,
        "official_test_used_only_after_train_internal_calibration": True,
        "primary_protocol": {
            "alpha": args.alpha,
            "support_gate": "global_train_conformal",
            "localizer": "positive_route_first_unsupported",
            "decoder": "hard_gate_route_stop",
        },
        "routing_checkpoint": args.routing_checkpoint,
        "routing_checkpoint_stage": routing_checkpoint.get("stage"),
        "support_checkpoint": args.support_checkpoint,
        "support_checkpoint_stage": support_checkpoint.get("stage"),
        "support_lora_enabled": args.support_lora_enabled,
        "reference_fraction": args.reference_fraction,
        "reference_samples": int(calibration.reference_indices.numel()),
        "calibration_samples": int(
            calibration.calibration_indices.numel()
        ),
        "positive_baseline": positive_baseline,
        "root_support_binary_ood": root_binary,
        "primary": primary,
        "primary_localizer_diagnostics": primary_localizer_diagnostics,
        "diagnostic_alpha_grid_not_for_selection": grid,
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
        f"Mixed BMHD={float(primary['mixed']['mixed_balanced_hdist']):.6f}"
    )
    print(
        "root support AUROC="
        f"{float(root_binary['auroc']):.6f}, "
        f"FPR95={float(root_binary['fpr95']):.6f}"
    )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
