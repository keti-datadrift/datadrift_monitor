from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import replace
import random
import sys
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import save_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, mixed_summary
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hc_negprompt import (
    global_decoder_safety_loss,
    hierarchy_constrained_negprompt_loss,
)
from negzerohoc.hier_negprompt import (
    build_differentiable_hier_negprompt_semantic_index,
    build_hier_negprompt_semantic_index,
    hierarchical_negprompt_loss,
)
from negzerohoc.idea4_inference import terminal_global_path_scores
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.oracle_parent import oracle_parent_diagnostics
from negzerohoc.prompt_models import HierNegativePromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.training_data import (
    build_positive_edge_examples,
    group_examples_by_parent_child,
    sample_parent_known_episode,
)
from negzerohoc.vision_lora import set_vision_lora_train_mode
from scripts.train_idea3_joint_vision_lora import make_loader, prompt_only_state_dict
from scripts.train_idea4_unknown_prompts import (
    GREEDY_INFERENCE_MODE,
    PRIMARY_INFERENCE_MODE,
    build_positive_semantic_index,
    dataset_payload,
    encode_dataset_features,
    encode_selected_train_images,
    evaluate_feature_payload,
    load_frozen_positive_stack,
    prompt_parameters,
    scalar_metric_summary,
)
from scripts.train_idea3_joint_vision_lora import build_datasets


VALID_METHODS = {"hnp_paper", "hnp_stop", "hc_negprompt"}


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
    validation_cfg = train_cfg.get("validation", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea5-hier-negprompt")
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    method = str(train_cfg.get("method", "hnp_stop"))
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported HierNegPrompt method: {method!r}")
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

    inference_mode = inference_cfg.get("mode", PRIMARY_INFERENCE_MODE)
    if inference_mode != PRIMARY_INFERENCE_MODE:
        raise ValueError(
            f"HierNegPrompt primary inference must be {PRIMARY_INFERENCE_MODE!r}"
        )
    allow_root_unknown = bool(inference_cfg.get("allow_root_unknown", False))
    if allow_root_unknown:
        raise ValueError("The FGVC HierNegPrompt protocol disables root unknown")
    unknown_aggregation = str(
        inference_cfg.get(
            "unknown_aggregation",
            "logsumexp" if method == "hc_negprompt" else "logmeanexp",
        )
    )
    safety_mode = str(loss_cfg.get("safety_mode", "softplus"))
    safety_scope = str(loss_cfg.get("safety_scope", "local"))
    if safety_mode not in {"softplus", "squared_hinge"}:
        raise ValueError(f"Unsupported negative_training.loss.safety_mode: {safety_mode!r}")
    if safety_scope not in {"local", "global_decoder"}:
        raise ValueError(
            f"Unsupported negative_training.loss.safety_scope: {safety_scope!r}"
        )
    if method != "hc_negprompt" and safety_scope != "local":
        raise ValueError("Global decoder safety is only supported by hc_negprompt")
    if unknown_aggregation not in {"logmeanexp", "logsumexp"}:
        raise ValueError(
            f"Unsupported inference.unknown_aggregation: {unknown_aggregation!r}"
        )
    if method == "hc_negprompt" and unknown_aggregation != "logsumexp":
        raise ValueError("hc_negprompt requires unknown_aggregation='logsumexp'")

    checkpoint = resolve_experiment_artifact(
        train_cfg.get("checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}.pt",
    )
    result_path = resolve_experiment_artifact(
        train_cfg.get("result_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="results",
        default_filename=f"{experiment_name}-global-path.result",
    )
    diagnostics_path = resolve_experiment_artifact(
        train_cfg.get("diagnostics_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="diagnostics",
        default_filename=f"{experiment_name}-diagnostics.json",
    )
    oracle_diagnostics_path = resolve_experiment_artifact(
        cfg.get("oracle_diagnostics", {}).get("path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="diagnostics",
        default_filename=f"{experiment_name}-oracle-parent.json",
    )

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        method=method,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=str(datadir),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch16"),
        tokenizer_model=clip_cfg.get(
            "tokenizer_model", clip_cfg.get("model", "openai/clip-vit-base-patch16")
        ),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        augmentation=cfg.get("augmentation", {}),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        positive_checkpoint=str(positive_checkpoint),
        epochs=max(1, int(train_cfg.get("epochs", 10))),
        parents_per_step=max(1, int(train_cfg.get("parents_per_step", 4))),
        max_examples_per_parent=max(1, int(train_cfg.get("max_examples_per_parent", 64))),
        vision_batch_size=max(1, int(train_cfg.get("vision_batch_size", 16))),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        negative_prompts=max(1, int(prompt_cfg.get("count_per_child", 2))),
        negative_prototype_ctx_tokens=max(
            0, int(prompt_cfg.get("prototype_ctx_tokens", 2))
        ),
        lambda_nis=float(loss_cfg.get("lambda_nis", 1.0)),
        lambda_npd=float(loss_cfg.get("lambda_npd", 0.1)),
        lambda_nnd=float(loss_cfg.get("lambda_nnd", 0.05)),
        lambda_stop=float(loss_cfg.get("lambda_stop", 0.0)),
        lambda_parent=float(loss_cfg.get("lambda_parent", 0.0)),
        lambda_hnis=float(loss_cfg.get("lambda_hnis", 1.0)),
        lambda_safe=float(loss_cfg.get("lambda_safe", 1.0)),
        lambda_shell=float(loss_cfg.get("lambda_shell", 0.1)),
        lambda_diversity=float(loss_cfg.get("lambda_diversity", 0.05)),
        lambda_route=float(loss_cfg.get("lambda_route", 0.5)),
        lambda_balance=float(loss_cfg.get("lambda_balance", 0.1)),
        hierarchy_tau=float(loss_cfg.get("hierarchy_tau", 0.07)),
        safety_margin=float(loss_cfg.get("safety_margin", 0.0)),
        safety_mode=safety_mode,
        safety_scope=safety_scope,
        depth_balanced=bool(loss_cfg.get("depth_balanced", True)),
        text_gradient_checkpointing=bool(
            train_cfg.get("text_gradient_checkpointing", False)
        ),
        global_safety_batch_size=max(
            1, int(train_cfg.get("global_safety_batch_size", 64))
        ),
        validation_every_n_epochs=max(
            1, int(validation_cfg.get("every_n_epochs", 1))
        ),
        validation_batch_size=max(1, int(validation_cfg.get("batch_size", 64))),
        max_id_bacc_drop=float(validation_cfg.get("max_id_bacc_drop", 0.04)),
        inference_mode=inference_mode,
        inference_batch_size=max(1, int(inference_cfg.get("batch_size", 64))),
        inference_tau=float(
            inference_cfg.get("tau", 1.0 / float(train_cfg.get("tau", 0.07)))
        ),
        allow_root_unknown=allow_root_unknown,
        unknown_aggregation=unknown_aggregation,
        greedy_ablation=bool(inference_cfg.get("greedy_ablation", True)),
        checkpoint=str(checkpoint),
        result_path=str(result_path),
        diagnostics_path=str(diagnostics_path),
        oracle_diagnostics_path=str(oracle_diagnostics_path),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def average_stats(stats: list[dict]) -> dict:
    if not stats:
        return {}
    keys = stats[0].keys()
    return {key: sum(float(item[key]) for item in stats) / len(stats) for key in keys}


def checkpoint_stage(args) -> str:
    if args.method == "hc_negprompt":
        return "idea6_hc_negprompt_frozen_positive_lora"
    return "hier_negprompt_frozen_positive_lora"


def save_checkpoint(args, positive_checkpoint, prompt_cfg, negative, metrics):
    return save_idea3_checkpoint(
        args.checkpoint,
        stage=checkpoint_stage(args),
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=positive_checkpoint["positive_state_dict"],
        unknown_state_dict=prompt_only_state_dict(negative),
        vision_lora_config=positive_checkpoint["vision_lora_config"],
        vision_lora_state_dict=positive_checkpoint["vision_lora_state_dict"],
        positive_checkpoint=args.positive_checkpoint,
        metrics=metrics,
        args=vars(args),
    )


def main():
    args = parse_args()
    configure_reproducibility(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    train_dataset, val_dataset, ood_dataset = build_datasets(args, hierarchy)
    val_loader = make_loader(
        val_dataset,
        args.validation_batch_size,
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

    positive_checkpoint, clip_model, text_encoder, prompt_cfg, positive, replaced_modules = (
        load_frozen_positive_stack(args, hierarchy, device)
    )
    if args.text_gradient_checkpointing:
        if not hasattr(clip_model, "gradient_checkpointing_enable"):
            raise RuntimeError("This CLIP implementation does not support gradient checkpointing")
        clip_model.gradient_checkpointing_enable()
    positive_index = build_positive_semantic_index(hierarchy, positive)
    prompt_cfg = replace(
        prompt_cfg,
        negative_prompts=args.negative_prompts,
        negative_prototype_ctx_tokens=args.negative_prototype_ctx_tokens,
    )
    negative = HierNegativePromptLearner(
        args.dataset,
        hierarchy,
        text_encoder,
        prompt_cfg,
    ).to(device)
    trainable_params = prompt_parameters(negative)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    train_payload = dataset_payload(train_dataset)
    val_payload = encode_dataset_features(
        args,
        clip_model,
        val_dataset,
        val_loader,
        device,
        "encode ID val",
    )
    train_groups = group_examples_by_parent_child(
        build_positive_edge_examples(hierarchy, train_payload)
    )
    parents = sorted(parent for parent in train_groups if parent != "root")
    if not parents:
        raise RuntimeError("HierNegPrompt found no eligible non-root parents")
    depth_counts = {}
    for parent in parents:
        depth = len(hierarchy.node_ancestors.get(parent, []))
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    num_training_depths = len(depth_counts)
    parent_loss_weights = {
        parent: len(parents)
        / (
            num_training_depths
            * depth_counts[len(hierarchy.node_ancestors.get(parent, []))]
        )
        if args.depth_balanced and args.method == "hc_negprompt"
        else 1.0
        for parent in parents
    }

    positive_baseline = evaluate_feature_payload(
        args,
        hierarchy,
        val_payload,
        positive_index,
        "val",
        mode="positive_global_path",
    )
    positive_baseline_bacc = float(positive_baseline["metrics"]["balanced_acc"])
    id_bacc_floor = positive_baseline_bacc - args.max_id_bacc_drop
    print(
        "HierNegPrompt training: "
        f"method={args.method}, vision_lora_modules={len(replaced_modules)}, "
        f"trainable_params={sum(parameter.numel() for parameter in trainable_params)}, "
        f"negative_prompts_per_child={args.negative_prompts}, "
        f"parents={len(parents)}, fixed_epochs={args.epochs}, "
        f"safety_scope={args.safety_scope}, "
        f"text_gradient_checkpointing={args.text_gradient_checkpointing}, "
        f"positive_val_bacc={positive_baseline_bacc:.6f}, "
        f"id_bacc_floor={id_bacc_floor:.6f}"
    )

    rng = random.Random(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        negative.train()
        positive.eval()
        clip_model.eval()
        set_vision_lora_train_mode(clip_model, False)
        if args.text_gradient_checkpointing:
            # Transformers activates encoder checkpointing only in train mode.
            text_encoder.text_model.train()
        shuffled_parents = list(parents)
        rng.shuffle(shuffled_parents)
        parent_chunks = [
            shuffled_parents[start:start + args.parents_per_step]
            for start in range(0, len(shuffled_parents), args.parents_per_step)
        ]
        iterator = (
            tqdm(parent_chunks, desc=f"{args.method} epoch {epoch}/{args.epochs}", leave=False)
            if tqdm
            else parent_chunks
        )
        step_stats = []
        for parent_chunk in iterator:
            episodes = [
                sample_parent_known_episode(
                    parent,
                    train_groups[parent],
                    args.max_examples_per_parent,
                    rng,
                )
                for parent in parent_chunk
            ]
            episodes = [episode for episode in episodes if episode is not None]
            if not episodes:
                continue

            image_indices = [
                example.image_index
                for episode in episodes
                for example in episode.examples
            ]
            global_image_indices = []
            if args.method == "hc_negprompt" and args.safety_scope == "global_decoder":
                global_image_indices = rng.sample(
                    range(len(train_dataset)),
                    min(args.global_safety_batch_size, len(train_dataset)),
                )
                image_indices.extend(global_image_indices)
            feature_by_index = encode_selected_train_images(
                args,
                clip_model,
                train_dataset,
                image_indices,
                device,
            )

            optimizer.zero_grad(set_to_none=True)
            global_semantic_index = None
            negative_features_by_parent = {}
            if args.method == "hc_negprompt" and args.safety_scope == "global_decoder":
                (
                    global_semantic_index,
                    negative_features_by_parent,
                ) = build_differentiable_hier_negprompt_semantic_index(
                    hierarchy,
                    positive_index,
                    negative,
                )
            parent_losses = []
            loss_weights = []
            parent_stats = []
            for episode in episodes:
                local = positive_index[episode.parent]
                model_children = (
                    list(local.children)
                    if args.method == "hc_negprompt"
                    else list(episode.children)
                )
                child_to_position = {
                    child: index for index, child in enumerate(local.children)
                }
                positive_features = torch.stack([
                    local.child_features[child_to_position[child]]
                    for child in model_children
                ]).to(device)
                if episode.parent in negative_features_by_parent:
                    negative_features = negative_features_by_parent[episode.parent]
                else:
                    negative_features = negative.encode_negative_prototypes(
                        episode.parent,
                        model_children,
                    )
                child_to_target = {
                    child: index for index, child in enumerate(model_children)
                }
                targets = torch.tensor(
                    [child_to_target[label] for label in episode.labels],
                    dtype=torch.long,
                    device=device,
                )
                image_features = torch.stack([
                    feature_by_index[example.image_index]
                    for example in episode.examples
                ])
                if args.method == "hc_negprompt":
                    parent_path = [
                        hierarchy.id_node_list[index]
                        for index in hierarchy.node_ancestors.get(
                            episode.parent, []
                        )
                    ] + [episode.parent]
                    route_levels = []
                    for ancestor, route_child in zip(
                        parent_path[:-1], parent_path[1:]
                    ):
                        route_local = positive_index[ancestor]
                        route_levels.append((
                            route_local.child_features.to(device),
                            list(route_local.children).index(route_child),
                        ))
                    loss, stats = hierarchy_constrained_negprompt_loss(
                        image_features,
                        positive_features,
                        negative_features,
                        targets,
                        negative._parent_feature(episode.parent),
                        route_levels,
                        tau=args.tau,
                        hierarchy_tau=args.hierarchy_tau,
                        safety_margin=args.safety_margin,
                        lambda_hnis=args.lambda_hnis,
                        lambda_safe=(
                            args.lambda_safe if args.safety_scope == "local" else 0.0
                        ),
                        lambda_shell=args.lambda_shell,
                        lambda_diversity=args.lambda_diversity,
                        lambda_route=args.lambda_route,
                        lambda_balance=args.lambda_balance,
                        safety_mode=args.safety_mode,
                    )
                else:
                    loss, stats = hierarchical_negprompt_loss(
                        image_features,
                        positive_features,
                        negative_features,
                        targets,
                        negative._parent_feature(episode.parent),
                        tau=args.tau,
                        lambda_nis=args.lambda_nis,
                        lambda_npd=args.lambda_npd,
                        lambda_nnd=args.lambda_nnd,
                        lambda_stop=args.lambda_stop,
                        lambda_parent=args.lambda_parent,
                    )
                parent_losses.append(loss)
                loss_weights.append(parent_loss_weights[episode.parent])
                parent_stats.append(stats)

            placement_loss = torch.stack([
                parent_loss * weight
                for parent_loss, weight in zip(parent_losses, loss_weights)
            ]).mean()
            step_stat = average_stats(parent_stats)
            loss = placement_loss
            if global_semantic_index is not None:
                global_image_features = torch.stack([
                    feature_by_index[index] for index in global_image_indices
                ])
                terminal_scores = terminal_global_path_scores(
                    global_image_features,
                    hierarchy,
                    global_semantic_index,
                    logit_scale=args.inference_tau,
                    allow_root_unknown=args.allow_root_unknown,
                    unknown_aggregation=args.unknown_aggregation,
                )
                global_safe_loss, global_stats = global_decoder_safety_loss(
                    terminal_scores["score_matrix"],
                    terminal_scores["candidate_kinds"],
                    margin=args.safety_margin,
                    safety_mode=args.safety_mode,
                )
                loss = placement_loss + args.lambda_safe * global_safe_loss
                step_stat.update(global_stats)
            step_stat["loss"] = float(loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
            optimizer.step()
            step_stats.append(step_stat)

        scheduler.step()
        epoch_stats = average_stats(step_stats)
        epoch_stats.update({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "steps": len(step_stats),
        })

        if epoch % args.validation_every_n_epochs == 0 or epoch == args.epochs:
            semantic_index = build_hier_negprompt_semantic_index(
                hierarchy,
                positive_index,
                negative,
            )
            val_result = evaluate_feature_payload(
                args,
                hierarchy,
                val_payload,
                semantic_index,
                "val",
                mode=PRIMARY_INFERENCE_MODE,
            )
            epoch_stats.update({
                "val_balanced_acc": float(val_result["metrics"]["balanced_acc"]),
                "val_balanced_hdist": float(val_result["metrics"]["balanced_hdist"]),
                "val_unknown_selection_rate": float(
                    val_result["diagnostics"]["unknown_selection_rate"]
                ),
                "id_guard_passed": float(
                    val_result["metrics"]["balanced_acc"]
                ) >= id_bacc_floor,
            })
        history.append(epoch_stats)
        if args.method == "hc_negprompt":
            if args.safety_scope == "global_decoder":
                message = (
                    f"epoch {epoch}: loss={epoch_stats.get('loss', 0.0):.6f}, "
                    f"hnis_excess={epoch_stats.get('hnis_excess', 0.0):.6f}, "
                    f"global_safe={epoch_stats.get('global_safe_loss', 0.0):.6f}, "
                    f"global_active={epoch_stats.get('global_safety_violation_rate', 0.0):.6f}, "
                    f"global_unk_win={epoch_stats.get('global_unknown_win_rate', 0.0):.6f}, "
                    f"route_acc={epoch_stats.get('route_acc', 0.0):.6f}"
                )
            else:
                message = (
                    f"epoch {epoch}: loss={epoch_stats.get('loss', 0.0):.6f}, "
                    f"hnis_excess={epoch_stats.get('hnis_excess', 0.0):.6f}, "
                    f"safe={epoch_stats.get('safe_loss', 0.0):.6f}, "
                    f"safe_active={epoch_stats.get('safety_violation_rate', 0.0):.6f}, "
                    f"route_acc={epoch_stats.get('route_acc', 0.0):.6f}, "
                    f"id_neg_win={epoch_stats.get('id_unknown_mass_win_rate', 0.0):.6f}"
                )
        else:
            message = (
                f"epoch {epoch}: loss={epoch_stats.get('loss', 0.0):.6f}, "
                f"nis_excess={epoch_stats.get('nis_excess', 0.0):.6f}, "
                f"stop_loss={epoch_stats.get('stop_loss', 0.0):.6f}, "
                f"known_acc={epoch_stats.get('known_acc', 0.0):.6f}"
            )
        if "val_balanced_acc" in epoch_stats:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"id_guard={epoch_stats['id_guard_passed']}"
            )
        print(message)

    semantic_index = build_hier_negprompt_semantic_index(
        hierarchy,
        positive_index,
        negative,
    )
    val_result = evaluate_feature_payload(
        args,
        hierarchy,
        val_payload,
        semantic_index,
        "val",
        mode=PRIMARY_INFERENCE_MODE,
    )
    ood_payload = encode_dataset_features(
        args,
        clip_model,
        ood_dataset,
        ood_loader,
        device,
        "encode OOD",
    )
    ood_result = evaluate_feature_payload(
        args,
        hierarchy,
        ood_payload,
        semantic_index,
        "ood",
        mode=PRIMARY_INFERENCE_MODE,
    )
    oracle_result = oracle_parent_diagnostics(
        ood_payload["features"],
        ood_payload["classes"],
        ood_payload["targets"],
        hierarchy,
        semantic_index,
        logit_scale=args.inference_tau,
        allow_root_unknown=args.allow_root_unknown,
        unknown_aggregation=args.unknown_aggregation,
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])
    id_guard_passed = float(val_result["metrics"]["balanced_acc"]) >= id_bacc_floor

    ablations = {}
    if args.greedy_ablation:
        greedy_val = evaluate_feature_payload(
            args,
            hierarchy,
            val_payload,
            semantic_index,
            "val",
            mode=GREEDY_INFERENCE_MODE,
        )
        greedy_ood = evaluate_feature_payload(
            args,
            hierarchy,
            ood_payload,
            semantic_index,
            "ood",
            mode=GREEDY_INFERENCE_MODE,
        )
        ablations[GREEDY_INFERENCE_MODE] = {
            "val": greedy_val,
            "ood": greedy_ood,
            "mixed": mixed_summary(greedy_val["metrics"], greedy_ood["metrics"]),
        }

    final_metrics = {
        "train_history": history,
        "positive_baseline": scalar_metric_summary(positive_baseline),
        "selection": {
            "policy": "fixed_last_epoch",
            "selected_epoch": args.epochs,
            "used_ood_for_selection": False,
            "id_bacc_floor": id_bacc_floor,
            "id_guard_passed": id_guard_passed,
        },
        "final": {
            "val": scalar_metric_summary(val_result),
            "ood": scalar_metric_summary(ood_result),
            "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
            "mixed_balanced_hdist": float(mixed["mixed_balanced_hdist"]),
        },
        "oracle_parent": oracle_result,
    }
    if args.greedy_ablation:
        greedy_result = ablations[GREEDY_INFERENCE_MODE]
        final_metrics["greedy_ablation"] = {
            "val": scalar_metric_summary(greedy_result["val"]),
            "ood": scalar_metric_summary(greedy_result["ood"]),
            "mixed_balanced_acc": float(greedy_result["mixed"]["mixed_balanced_acc"]),
            "mixed_balanced_hdist": float(greedy_result["mixed"]["mixed_balanced_hdist"]),
        }

    save_checkpoint(
        args,
        positive_checkpoint,
        prompt_cfg,
        negative,
        final_metrics,
    )
    result = {
        "args": vars(args),
        "mode": PRIMARY_INFERENCE_MODE,
        "method": args.method,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint_stage(args),
        "positive_checkpoint": args.positive_checkpoint,
        "selected_epoch": args.epochs,
        "selection_policy": "fixed_last_epoch",
        "selection_used_ood": False,
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "oracle_parent": oracle_result,
        "mixed": mixed,
        "ablations": ablations,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    save_json(args.diagnostics_path, final_metrics)
    save_json(args.oracle_diagnostics_path, {
        "diagnostic_only": True,
        "used_for_checkpoint_selection": False,
        "checkpoint": args.checkpoint,
        "oracle_parent": oracle_result,
    })

    print(f"fixed selected epoch: {args.epochs} (OOD was not used for selection)")
    print(f"ID guard passed: {id_guard_passed}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved result: {args.result_path}")
    print(f"saved oracle-parent diagnostics: {args.oracle_diagnostics_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"OOD BAcc: {float(ood_result['metrics']['balanced_acc']):.6f}")
    print(f"Mixed BAcc: {float(mixed['mixed_balanced_acc']):.6f}")
    print(f"Mixed BMHD: {float(mixed['mixed_balanced_hdist']):.6f}")
    print(
        "Oracle-parent: "
        f"local_unknown={oracle_result['oracle_unknown_balanced_recall']:.6f}, "
        f"positive_route={oracle_result['positive_route_balanced_reach_rate']:.6f}, "
        f"joint={oracle_result['joint_student_balanced_exact_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
