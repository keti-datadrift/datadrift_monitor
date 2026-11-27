from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.evaluation import build_hierarchy, mixed_summary
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.metric_terminal import build_metric_terminal_specs
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
)
from negzerohoc.training_data import (
    UNKNOWN_LABEL,
    build_positive_edge_examples,
    group_examples_by_parent_child,
    sample_leave_child_out_episode,
)
from negzerohoc.tree_loco import (
    ParentSpecificVirtualSiblingPromptLearner,
    balanced_slot_assignment_loss,
    leave_one_child_out_global_recall,
    loco_pseudo_unknown_loss,
    prune_terminal_specs_for_hidden_child,
)
from negzerohoc.tree_virtual_unknown import (
    augmented_unknown_distance_matrix,
    calibrate_unknown_gap_threshold,
    decoder_aligned_hierarchical_id_loss,
    leaf_unknown_distance_matrix,
    tree_complement_terminal_scores,
    tree_ordinal_prompt_loss,
    virtual_sibling_shell_loss,
)
from scripts.train_idea3_joint_vision_lora import (
    build_datasets,
    make_loader,
    target_leaf_nodes,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    encode_selected_train_images,
    load_frozen_positive_stack,
)
from scripts.train_paper_negprompt_ablation import (
    json_ready,
    select_few_shot_indices,
)
from scripts.train_tree_virtual_unknown import (
    evaluate_payload,
    load_config as load_base_config,
    positive_feature_layout,
    score_kwargs,
)

CHECKPOINT_STAGE = "tree_loco_multiprompt_unknown"


def load_config(path: str | Path):
    args = load_base_config(path)
    cfg = args.raw_config
    loco_cfg = cfg.get("loco_training", {})
    loss_cfg = loco_cfg.get("loss", {})
    prompt_cfg = loco_cfg.get("prompt", {})
    args.num_unknown_prompts = max(
        1,
        int(prompt_cfg.get("count", args.num_unknown_prompts)),
    )
    args.shared_init_noise = float(
        prompt_cfg.get("shared_init_noise", 1e-2)
    )
    args.parent_init_noise = float(
        prompt_cfg.get("parent_init_noise", 1e-3)
    )
    args.parents_per_step = max(
        1, int(loco_cfg.get("parents_per_step", 4))
    )
    args.max_examples_per_parent = max(
        2, int(loco_cfg.get("max_examples_per_parent", 64))
    )
    args.vision_batch_size = max(
        1, int(loco_cfg.get("vision_batch_size", args.batch_size))
    )
    args.hide_strategy = str(
        loco_cfg.get("hide_strategy", "hide_one_child")
    )
    args.lambda_pseudo = float(loss_cfg.get("lambda_pseudo", 1.0))
    args.lambda_slot = float(loss_cfg.get("lambda_slot", 0.1))
    args.lambda_parent_offset = float(
        loss_cfg.get("lambda_parent_offset", 1e-4)
    )
    args.pseudo_local_margin = float(
        loss_cfg.get("pseudo_local_margin", 0.05)
    )
    args.pseudo_distance_margin = float(
        loss_cfg.get("pseudo_distance_margin", 0.02)
    )
    args.pseudo_temperature = float(
        loss_cfg.get("pseudo_temperature", 0.05)
    )
    args.slot_temperature = float(
        loss_cfg.get("slot_temperature", 0.05)
    )
    args.id_acceptance = float(
        cfg.get("inference", {}).get("id_acceptance", 0.95)
    )
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def save_checkpoint(
    path: str,
    *,
    args,
    positive_checkpoint: dict,
    learner,
    optimizer,
    scheduler,
    epoch: int,
    history: list[dict],
    rng: random.Random,
    complete: bool,
) -> None:
    payload = {
        "stage": CHECKPOINT_STAGE,
        "positive_checkpoint": args.positive_checkpoint,
        "positive_checkpoint_stage": positive_checkpoint.get("stage"),
        "unknown_state_dict": learner.prompt_state(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": int(epoch),
        "history": history,
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "training_complete": bool(complete),
        "args": vars(args),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    ensure_dir(Path(path).parent)
    torch.save(payload, path)


def restore_rng_state(checkpoint: dict, rng: random.Random) -> None:
    if "rng_state" in checkpoint:
        rng.setstate(checkpoint["rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])


def terminal_ood_scores(
    payload: dict,
    args,
    hierarchy,
    edge_features: dict,
    terminal_specs,
    unknown_features_by_parent: dict,
) -> torch.Tensor:
    chunks = []
    leaf_mask = torch.tensor([
        spec.unknown_parent is None for spec in terminal_specs
    ], dtype=torch.bool)
    unknown_mask = ~leaf_mask
    for start in range(
        0, len(payload["features"]), args.inference_batch_size
    ):
        output = tree_complement_terminal_scores(
            payload["features"][
                start:start + args.inference_batch_size
            ],
            hierarchy,
            edge_features,
            terminal_specs,
            unknown_features_by_parent,
            **score_kwargs(args),
        )
        scores = output["score_matrix"].cpu()
        chunks.append(
            scores[:, unknown_mask].max(dim=1).values
            - scores[:, leaf_mask].max(dim=1).values
        )
    return torch.cat(chunks)


def aggregate_epoch_stats(rows: list[dict], epoch: int, lr: float) -> dict:
    result = {
        key: float(np.mean([
            row[key] for row in rows if key in row
        ]))
        for key in sorted({
            key for row in rows for key in row
        })
    }
    result.update({
        "epoch": int(epoch),
        "steps": len(rows),
        "lr": float(lr),
    })
    return result


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
    learner = ParentSpecificVirtualSiblingPromptLearner(
        positive,
        unknown_parents,
        num_unknown_prompts=args.num_unknown_prompts,
        shared_init_noise=args.shared_init_noise,
        parent_init_noise=args.parent_init_noise,
    ).to(device)
    if args.text_gradient_checkpointing:
        if not hasattr(clip_model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "CLIP text gradient checkpointing is unavailable"
            )
        clip_model.gradient_checkpointing_enable()

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

    full_train_payload = {
        "classes": list(train_dataset.classes),
        "targets": torch.tensor(
            train_dataset.targets,
            dtype=torch.long,
        ),
    }
    selected_indices = set(select_few_shot_indices(
        train_dataset.targets,
        shots_per_class=args.shots_per_class,
        seed=args.seed,
    ))
    selected_examples = [
        example
        for example in build_positive_edge_examples(
            hierarchy,
            full_train_payload,
        )
        if example.image_index in selected_indices
    ]
    train_groups = group_examples_by_parent_child(selected_examples)
    episode_parents = [
        parent
        for parent in unknown_parents
        if len(train_groups.get(parent, {})) >= 2
    ]
    if not episode_parents:
        raise RuntimeError("No non-root LOCO parent has two ID children")

    parameters = learner.trainable_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(episode_parents) / args.parents_per_step
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs * steps_per_epoch),
    )
    rng = random.Random(args.seed)
    start_epoch = 1
    history = []
    if args.automatic_resume and Path(args.last_checkpoint).exists():
        resume = torch.load(
            args.last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if resume.get("stage") == CHECKPOINT_STAGE:
            learner.load_prompt_state(resume["unknown_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            history = list(resume.get("history", []))
            start_epoch = int(resume["epoch"]) + 1
            restore_rng_state(resume, rng)

    print(
        "Tree LOCO multi-prompt training: "
        f"device={device}, frozen_lora_modules={len(replaced_modules)}, "
        f"parents={len(unknown_parents)}, episode_parents="
        f"{len(episode_parents)}, K={args.num_unknown_prompts}, "
        f"shots={args.shots_per_class}, parents_per_step="
        f"{args.parents_per_step}, epochs={args.epochs}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        learner.train()
        clip_model.eval()
        if args.text_gradient_checkpointing:
            positive.text_encoder.text_model.train()
        shuffled_parents = list(episode_parents)
        rng.shuffle(shuffled_parents)
        parent_chunks = [
            shuffled_parents[
                start:start + args.parents_per_step
            ]
            for start in range(
                0, len(shuffled_parents), args.parents_per_step
            )
        ]
        epoch_rows = []
        for parent_chunk in parent_chunks:
            episodes = [
                sample_leave_child_out_episode(
                    parent,
                    train_groups[parent],
                    strategy=args.hide_strategy,
                    max_examples=args.max_examples_per_parent,
                    rng=rng,
                )
                for parent in parent_chunk
            ]
            episodes = [
                episode for episode in episodes
                if episode is not None
            ]
            if not episodes:
                continue
            image_indices = [
                example.image_index
                for episode in episodes
                for example in episode.examples
            ]
            feature_by_index = encode_selected_train_images(
                args,
                clip_model,
                train_dataset,
                image_indices,
                device,
            )
            unknown_tensor = learner.encode_parents(unknown_parents)
            unknown_features_by_parent = {
                parent: unknown_tensor[index]
                for index, parent in enumerate(unknown_parents)
            }
            tree_loss, tree_stats = tree_ordinal_prompt_loss(
                unknown_tensor,
                positive_node_features,
                unknown_node_distances,
                margin_per_step=args.ordinal_margin_per_step,
                temperature=args.ordinal_temperature,
            )
            shell_loss, shell_stats = virtual_sibling_shell_loss(
                unknown_features_by_parent,
                child_features_by_parent,
            )

            id_losses = []
            pseudo_losses = []
            slot_losses = []
            id_stats_rows = []
            pseudo_stats_rows = []
            slot_stats_rows = []
            for episode in episodes:
                image_features = torch.stack([
                    feature_by_index[example.image_index]
                    for example in episode.examples
                ])
                leaves = [example.leaf for example in episode.examples]
                full_scores = tree_complement_terminal_scores(
                    image_features,
                    hierarchy,
                    edge_features,
                    terminal_specs,
                    unknown_features_by_parent,
                    **score_kwargs(args),
                )
                leaf_distances = torch.stack([
                    leaf_distance_lookup[leaf] for leaf in leaves
                ]).to(device)
                id_loss, id_stats = (
                    decoder_aligned_hierarchical_id_loss(
                        full_scores,
                        leaves,
                        leaf_distances,
                        local_margin=args.local_margin,
                        distance_margin=args.distance_margin,
                        temperature=args.id_temperature,
                    )
                )

                hidden_mask = torch.tensor(
                    [
                        label == UNKNOWN_LABEL
                        for label in episode.labels
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                hidden_features = image_features[hidden_mask]
                if int(hidden_features.shape[0]) == 0:
                    continue
                if len(episode.hidden_children) != 1:
                    raise RuntimeError(
                        "This experiment requires hide_one_child"
                    )
                hidden_child = episode.hidden_children[0]
                pruned_specs = prune_terminal_specs_for_hidden_child(
                    hierarchy,
                    terminal_specs,
                    episode.parent,
                    hidden_child,
                )
                pseudo_scores = tree_complement_terminal_scores(
                    hidden_features,
                    hierarchy,
                    edge_features,
                    pruned_specs,
                    unknown_features_by_parent,
                    excluded_children_by_parent={
                        episode.parent: {hidden_child}
                    },
                    **score_kwargs(args),
                )
                pseudo_loss, pseudo_stats = loco_pseudo_unknown_loss(
                    pseudo_scores,
                    hierarchy,
                    pruned_specs,
                    episode.parent,
                    local_margin=args.pseudo_local_margin,
                    distance_margin=args.pseudo_distance_margin,
                    temperature=args.pseudo_temperature,
                )
                slot_loss, slot_stats = balanced_slot_assignment_loss(
                    hidden_features,
                    unknown_features_by_parent[episode.parent],
                    temperature=args.slot_temperature,
                )
                id_losses.append(id_loss)
                pseudo_losses.append(pseudo_loss)
                slot_losses.append(slot_loss)
                id_stats_rows.append(id_stats)
                pseudo_stats_rows.append(pseudo_stats)
                slot_stats_rows.append(slot_stats)

            if not pseudo_losses:
                continue
            id_loss = torch.stack(id_losses).mean()
            pseudo_loss = torch.stack(pseudo_losses).mean()
            slot_loss = torch.stack(slot_losses).mean()
            parent_offset_loss = learner.parent_offset_regularizer()
            loss = (
                args.lambda_id * id_loss
                + args.lambda_pseudo * pseudo_loss
                + args.lambda_slot * slot_loss
                + args.lambda_tree * tree_loss
                + args.lambda_shell * shell_loss
                + args.lambda_parent_offset * parent_offset_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                args.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()

            def mean_stat(rows, key):
                return float(np.mean([row[key] for row in rows]))

            epoch_rows.append({
                "loss": float(loss.detach().cpu()),
                "decoder_id_loss": float(id_loss.detach().cpu()),
                "loco_pseudo_loss": float(pseudo_loss.detach().cpu()),
                "slot_loss": float(slot_loss.detach().cpu()),
                "tree_ordinal_loss": tree_stats["tree_ordinal_loss"],
                "virtual_sibling_shell_loss": (
                    shell_stats["virtual_sibling_shell_loss"]
                ),
                "parent_offset_loss": float(
                    parent_offset_loss.detach().cpu()
                ),
                "decoder_id_unknown_win_rate": mean_stat(
                    id_stats_rows,
                    "decoder_id_unknown_win_rate",
                ),
                "loco_target_win_rate": mean_stat(
                    pseudo_stats_rows,
                    "loco_target_win_rate",
                ),
                "loco_margin_violation_rate": mean_stat(
                    pseudo_stats_rows,
                    "loco_margin_violation_rate",
                ),
                "slot_effective_count": mean_stat(
                    slot_stats_rows,
                    "slot_effective_count",
                ),
                "slot_max_usage": mean_stat(
                    slot_stats_rows,
                    "slot_max_usage",
                ),
            })
        if not epoch_rows:
            raise RuntimeError(f"Epoch {epoch} produced no LOCO update")
        averaged = aggregate_epoch_stats(
            epoch_rows,
            epoch,
            optimizer.param_groups[0]["lr"],
        )
        history.append(averaged)
        save_checkpoint(
            args.last_checkpoint,
            args=args,
            positive_checkpoint=positive_checkpoint,
            learner=learner,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            history=history,
            rng=rng,
            complete=epoch == args.epochs,
        )
        print(
            f"epoch {epoch}/{args.epochs}: "
            f"loss={averaged['loss']:.6f}, "
            f"id={averaged['decoder_id_loss']:.6f}, "
            f"loco={averaged['loco_pseudo_loss']:.6f}, "
            f"slot_eff={averaged['slot_effective_count']:.3f}, "
            f"id_unk_win="
            f"{averaged['decoder_id_unknown_win_rate']:.4f}, "
            f"loco_win={averaged['loco_target_win_rate']:.4f}"
        )

    save_checkpoint(
        args.checkpoint,
        args=args,
        positive_checkpoint=positive_checkpoint,
        learner=learner,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        history=history,
        rng=rng,
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
    raw_val_result = evaluate_payload(
        args,
        hierarchy,
        val_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    raw_ood_result = evaluate_payload(
        args,
        hierarchy,
        ood_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    raw_mixed = mixed_summary(
        raw_val_result["metrics"],
        raw_ood_result["metrics"],
    )
    id_scores = terminal_ood_scores(
        val_payload,
        args,
        hierarchy,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    ood_scores = terminal_ood_scores(
        ood_payload,
        args,
        hierarchy,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
    )
    binary = binary_ood_metrics(
        id_scores.numpy(),
        ood_scores.numpy(),
    )
    calibrated_threshold = calibrate_unknown_gap_threshold(
        id_scores,
        id_acceptance=args.id_acceptance,
    )
    val_result = evaluate_payload(
        args,
        hierarchy,
        val_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
        unknown_threshold=calibrated_threshold,
    )
    ood_result = evaluate_payload(
        args,
        hierarchy,
        ood_payload,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
        unknown_threshold=calibrated_threshold,
    )
    mixed = mixed_summary(
        val_result["metrics"],
        ood_result["metrics"],
    )
    val_leaves = target_leaf_nodes(
        hierarchy,
        val_payload["classes"],
        val_payload["targets"],
    )
    pseudo_validation = leave_one_child_out_global_recall(
        val_payload["features"],
        val_leaves,
        hierarchy,
        edge_features_cpu,
        terminal_specs,
        unknown_features_by_parent,
        **score_kwargs(args),
    )
    result = {
        "method": "tree_loco_multiprompt_unknown",
        "used_actual_ood_for_training_or_selection": False,
        "args": vars(args),
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "training_history": history,
        "pseudo_loco_validation": pseudo_validation,
        "binary_ood": binary,
        "inference_calibration": {
            "source": "ID validation only",
            "id_acceptance": args.id_acceptance,
            "unknown_gap_threshold": calibrated_threshold,
        },
        "raw_threshold_free": {
            "val": raw_val_result,
            "ood": raw_ood_result,
            "mixed": raw_mixed,
        },
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    diagnostics = {
        key: value for key, value in result.items()
        if key not in {"val", "ood"}
    }
    diagnostics.update({
        "val_metrics": val_result["metrics"],
        "val_unknown_selection_rate": (
            val_result["unknown_selection_rate"]
        ),
        "ood_metrics": ood_result["metrics"],
        "ood_unknown_selection_rate": (
            ood_result["unknown_selection_rate"]
        ),
    })
    save_json(args.diagnostics_path, json_ready(diagnostics))
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved result: {args.result_path}")
    print(
        "Final: "
        f"AUROC={binary['auroc']:.6f}, "
        f"FPR95={binary['fpr95']:.6f}, "
        f"ID-only threshold={calibrated_threshold:.6f}, "
        f"ID BAcc={float(val_result['metrics']['balanced_acc']):.6f}, "
        f"OOD BAcc={float(ood_result['metrics']['balanced_acc']):.6f}, "
        f"Mixed BAcc={float(mixed['mixed_balanced_acc']):.6f}, "
        f"LOCO fold recall="
        f"{pseudo_validation['fold_macro_recall']:.6f}"
    )


if __name__ == "__main__":
    main()
