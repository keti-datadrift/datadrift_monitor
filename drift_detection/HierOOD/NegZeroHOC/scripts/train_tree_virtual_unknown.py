from __future__ import annotations

import argparse
from argparse import Namespace
import sys
from pathlib import Path

import numpy as np
import torch
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
from negzerohoc.metric_terminal import build_metric_terminal_specs
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.tree_virtual_unknown import (
    VirtualSiblingPromptLearner,
    augmented_unknown_distance_matrix,
    leaf_unknown_distance_matrix,
    predict_tree_complement_terminals,
    tree_complement_terminal_scores,
    tree_virtual_unknown_loss,
)
from scripts.train_idea3_joint_vision_lora import (
    build_datasets,
    make_loader,
    target_leaf_nodes,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    load_frozen_positive_stack,
)
from scripts.train_paper_negprompt_ablation import (
    all_edge_pairs,
    json_ready,
    select_few_shot_indices,
)


CHECKPOINT_STAGE = "tree_consistent_virtual_sibling_unknown"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    positive_cfg = cfg.get("positive", {})
    train_cfg = cfg.get("unknown_training", {})
    prompt_cfg = train_cfg.get("prompt", {})
    loss_cfg = train_cfg.get("loss", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = str(
        experiment_cfg.get("name", "tree-virtual-unknown")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

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
        epochs=max(1, int(train_cfg.get("epochs", 30))),
        batch_size=max(1, int(train_cfg.get("batch_size", 64))),
        shots_per_class=max(
            1, int(train_cfg.get("shots_per_class", 16))
        ),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(
            train_cfg.get("gradient_clip_norm", 1.0)
        ),
        text_gradient_checkpointing=bool(
            train_cfg.get("text_gradient_checkpointing", True)
        ),
        num_unknown_prompts=max(
            1, int(prompt_cfg.get("count", 2))
        ),
        init_noise=float(prompt_cfg.get("init_noise", 1e-2)),
        lambda_id=float(loss_cfg.get("lambda_id", 1.0)),
        lambda_tree=float(loss_cfg.get("lambda_tree", 1.0)),
        lambda_shell=float(loss_cfg.get("lambda_shell", 1.0)),
        local_margin=float(loss_cfg.get("local_margin", 0.05)),
        distance_margin=float(
            loss_cfg.get("distance_margin", 0.02)
        ),
        id_temperature=float(
            loss_cfg.get("id_temperature", 0.05)
        ),
        ordinal_margin_per_step=float(
            loss_cfg.get("ordinal_margin_per_step", 0.02)
        ),
        ordinal_temperature=float(
            loss_cfg.get("ordinal_temperature", 0.05)
        ),
        terminal_weight=float(
            inference_cfg.get("terminal_weight", 0.75)
        ),
        bottleneck_temperature=float(
            inference_cfg.get("bottleneck_temperature", 0.1)
        ),
        unknown_temperature=float(
            inference_cfg.get("unknown_temperature", 0.07)
        ),
        child_temperature=float(
            inference_cfg.get("child_temperature", 0.07)
        ),
        complement_weight=float(
            inference_cfg.get("complement_weight", 0.5)
        ),
        inference_batch_size=max(
            1, int(inference_cfg.get("batch_size", 128))
        ),
        checkpoint=artifact(
            train_cfg.get("checkpoint"),
            "checkpoints",
            "unknown.pt",
        ),
        last_checkpoint=artifact(
            train_cfg.get("last_checkpoint"),
            "checkpoints",
            "unknown-last.pt",
        ),
        result_path=artifact(
            train_cfg.get("result_path"),
            "results",
            "tree-complement.result",
        ),
        diagnostics_path=artifact(
            train_cfg.get("diagnostics_path"),
            "diagnostics",
            "tree-complement.json",
        ),
        automatic_resume=bool(
            train_cfg.get("resume", {}).get("automatic", True)
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def prompt_state(learner) -> dict[str, torch.Tensor]:
    return {
        "context_offsets": (
            learner.context_offsets.detach().cpu().clone()
        )
    }


def load_prompt_state(learner, state: dict) -> None:
    with torch.no_grad():
        learner.context_offsets.copy_(
            state["context_offsets"].to(
                learner.context_offsets.device,
                dtype=learner.context_offsets.dtype,
            )
        )


def save_checkpoint(
    path: str,
    *,
    args,
    positive_checkpoint: dict,
    learner,
    epoch: int,
    history: list[dict],
    complete: bool,
    optimizer=None,
    scheduler=None,
) -> None:
    payload = {
        "stage": CHECKPOINT_STAGE,
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
        "positive_checkpoint": args.positive_checkpoint,
        "positive_checkpoint_stage": positive_checkpoint.get("stage"),
        "unknown_state_dict": prompt_state(learner),
        "epoch": int(epoch),
        "history": history,
        "training_complete": bool(complete),
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def positive_feature_layout(hierarchy, positive, device):
    pairs = all_edge_pairs(hierarchy)
    with torch.no_grad():
        feature_tensor = positive.encode_edges(pairs).float().detach()
    edge_features = {
        pair: feature_tensor[index]
        for index, pair in enumerate(pairs)
    }
    positive_nodes = [
        node for node in hierarchy.id_node_list if node != "root"
    ]
    node_features = torch.stack([
        edge_features[(hierarchy.child2parent[node], node)]
        for node in positive_nodes
    ]).to(device)
    child_features = {
        parent: torch.stack([
            edge_features[(parent, child)]
            for child in hierarchy.parent2children[parent]
        ]).to(device)
        for parent in hierarchy.parent2children
        if parent != "root"
    }
    return edge_features, positive_nodes, node_features, child_features


def score_kwargs(args) -> dict:
    return {
        "terminal_weight": args.terminal_weight,
        "bottleneck_temperature": args.bottleneck_temperature,
        "unknown_temperature": args.unknown_temperature,
        "child_temperature": args.child_temperature,
        "complement_weight": args.complement_weight,
    }


@torch.no_grad()
def evaluate_payload(
    args,
    hierarchy,
    payload: dict,
    edge_features: dict,
    terminal_specs,
    unknown_features_by_parent: dict,
    unknown_threshold: float = 0.0,
) -> dict:
    predictions = []
    unknown_count = 0
    for start in range(
        0, len(payload["features"]), args.inference_batch_size
    ):
        features = payload["features"][
            start:start + args.inference_batch_size
        ]
        output = predict_tree_complement_terminals(
            features,
            hierarchy,
            edge_features,
            terminal_specs,
            unknown_features_by_parent,
            unknown_threshold=unknown_threshold,
            **score_kwargs(args),
        )
        predictions.append(output["preds"])
        unknown_count += output["diagnostics"][
            "candidate_type_counts"
        ].get("unknown", 0)
    preds = torch.cat(predictions)
    targets, metrics = evaluate_split(
        hierarchy,
        payload,
        preds,
        dists_mats=make_distance_mats(hierarchy),
    )
    return {
        "preds": preds,
        "targets": targets,
        "metrics": metrics,
        "unknown_selection_rate": (
            unknown_count / max(1, len(payload["features"]))
        ),
    }


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
    ) = load_frozen_positive_stack(args, hierarchy, device)
    learner = VirtualSiblingPromptLearner(
        positive,
        num_unknown_prompts=args.num_unknown_prompts,
        init_noise=args.init_noise,
    ).to(device)
    if args.text_gradient_checkpointing:
        if not hasattr(clip_model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "CLIP text gradient checkpointing is unavailable"
            )
        clip_model.gradient_checkpointing_enable()

    (
        edge_features,
        positive_nodes,
        positive_node_features,
        child_features_by_parent,
    ) = positive_feature_layout(hierarchy, positive, device)
    unknown_parents = sorted(
        parent
        for parent in hierarchy.parent2children
        if parent != "root"
    )
    terminal_specs = build_metric_terminal_specs(
        hierarchy,
        unknown_parents=unknown_parents,
        allow_root_unknown=False,
    )
    unknown_node_distances = augmented_unknown_distance_matrix(
        hierarchy,
        unknown_parents,
        positive_nodes,
    ).to(device)
    training_leaves = list(hierarchy.train_classes)
    leaf_distance_lookup = {
        leaf: row
        for leaf, row in zip(
            training_leaves,
            leaf_unknown_distance_matrix(
                hierarchy,
                training_leaves,
                unknown_parents,
            ),
        )
    }
    optimizer = torch.optim.AdamW(
        learner.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * len(train_loader)),
    )
    start_epoch = 1
    history = []
    if args.automatic_resume and Path(args.last_checkpoint).exists():
        resume = torch.load(
            args.last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if resume.get("stage") == CHECKPOINT_STAGE:
            load_prompt_state(learner, resume["unknown_state_dict"])
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
        "Tree virtual-unknown training: "
        f"device={device}, frozen_lora_modules={len(replaced_modules)}, "
        f"parents={len(unknown_parents)}, K={args.num_unknown_prompts}, "
        f"shots={args.shots_per_class}, batch={args.batch_size}, "
        f"epochs={args.epochs}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        learner.train()
        clip_model.eval()
        if args.text_gradient_checkpointing:
            positive.text_encoder.text_model.train()
        stats_this_epoch = []
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                image_features = clip_model.get_image_features(
                    pixel_values=images
                ).float()
            leaves = target_leaf_nodes(
                hierarchy,
                train_dataset.classes,
                targets,
            )
            unknown_tensor = learner.encode_parents(unknown_parents)
            unknown_features_by_parent = {
                parent: unknown_tensor[index]
                for index, parent in enumerate(unknown_parents)
            }
            scores = tree_complement_terminal_scores(
                image_features,
                hierarchy,
                edge_features,
                terminal_specs,
                unknown_features_by_parent,
                **score_kwargs(args),
            )
            batch_leaf_distances = torch.stack([
                leaf_distance_lookup[leaf] for leaf in leaves
            ]).to(device)
            loss, stats = tree_virtual_unknown_loss(
                score_output=scores,
                target_leaves=leaves,
                leaf_unknown_distances=batch_leaf_distances,
                unknown_feature_tensor=unknown_tensor,
                positive_node_features=positive_node_features,
                unknown_node_distances=unknown_node_distances,
                unknown_features_by_parent=unknown_features_by_parent,
                child_features_by_parent=child_features_by_parent,
                lambda_id=args.lambda_id,
                lambda_tree=args.lambda_tree,
                lambda_shell=args.lambda_shell,
                local_margin=args.local_margin,
                distance_margin=args.distance_margin,
                id_temperature=args.id_temperature,
                ordinal_margin_per_step=(
                    args.ordinal_margin_per_step
                ),
                ordinal_temperature=args.ordinal_temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                learner.trainable_parameters(),
                args.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
            stats_this_epoch.append(stats)
        averaged = {
            key: float(np.mean([
                stats[key] for stats in stats_this_epoch
            ]))
            for key in stats_this_epoch[0]
            if isinstance(stats_this_epoch[0][key], (int, float))
        }
        averaged.update({
            "epoch": epoch,
            "steps": len(stats_this_epoch),
            "lr": optimizer.param_groups[0]["lr"],
        })
        history.append(averaged)
        save_checkpoint(
            args.last_checkpoint,
            args=args,
            positive_checkpoint=positive_checkpoint,
            learner=learner,
            epoch=epoch,
            history=history,
            complete=epoch == args.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={averaged['loss']:.6f}, "
            f"id={averaged['decoder_id_loss']:.6f}, "
            f"tree={averaged['tree_ordinal_loss']:.6f}, "
            f"shell={averaged['virtual_sibling_shell_loss']:.6f}, "
            f"id_unk_win={averaged['decoder_id_unknown_win_rate']:.6f}"
        )

    save_checkpoint(
        args.checkpoint,
        args=args,
        positive_checkpoint=positive_checkpoint,
        learner=learner,
        epoch=args.epochs,
        history=history,
        complete=True,
    )
    learner.eval()
    positive.text_encoder.text_model.eval()
    with torch.no_grad():
        unknown_tensor = learner.encode_parents(
            unknown_parents
        ).float().cpu()
    unknown_features_by_parent = {
        parent: unknown_tensor[index]
        for index, parent in enumerate(unknown_parents)
    }
    edge_features_cpu = {
        edge: feature.float().cpu()
        for edge, feature in edge_features.items()
    }
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
    val_result = evaluate_payload(
        args,
        hierarchy,
        val_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    ood_result = evaluate_payload(
        args,
        hierarchy,
        ood_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    mixed = mixed_summary(
        val_result["metrics"],
        ood_result["metrics"],
    )
    result = {
        "method": "tree_consistent_virtual_sibling_unknown",
        "used_actual_ood_for_training_or_selection": False,
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "training_history": history,
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    diagnostics = {
        "method": result["method"],
        "used_actual_ood_for_training_or_selection": False,
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "training_history": history,
        "val_metrics": val_result["metrics"],
        "val_unknown_selection_rate": (
            val_result["unknown_selection_rate"]
        ),
        "ood_metrics": ood_result["metrics"],
        "ood_unknown_selection_rate": (
            ood_result["unknown_selection_rate"]
        ),
        "mixed": mixed,
    }
    save_json(args.diagnostics_path, json_ready(diagnostics))
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved result: {args.result_path}")
    print(
        "HOC: "
        f"ID BAcc={float(val_result['metrics']['balanced_acc']):.6f}, "
        f"OOD BAcc={float(ood_result['metrics']['balanced_acc']):.6f}, "
        f"Mixed BAcc={float(mixed['mixed_balanced_acc']):.6f}"
    )


if __name__ == "__main__":
    main()
