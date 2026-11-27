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
from negzerohoc.oracle_parent import oracle_parent_diagnostics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.prompt_models import UnknownPromptLearner
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
from negzerohoc.virtual_open_negprompt import (
    refine_virtual_open_features,
    spherical_sibling_mixup,
    virtual_open_negprompt_loss,
)
from negzerohoc.vision_lora import set_vision_lora_train_mode
from scripts.train_idea3_joint_vision_lora import (
    build_datasets,
    load_prompt_only_state_dict,
    make_loader,
    prompt_only_state_dict,
)
from scripts.train_idea4_unknown_prompts import (
    GREEDY_INFERENCE_MODE,
    PRIMARY_INFERENCE_MODE,
    build_positive_semantic_index,
    build_unknown_semantic_index,
    clone_prompt_state,
    dataset_payload,
    encode_dataset_features,
    encode_selected_train_images,
    evaluate_feature_payload,
    load_frozen_positive_stack,
    prompt_parameters,
    scalar_metric_summary,
)


CHECKPOINT_STAGE = "idea7_virtual_open_negprompt_frozen_positive_lora"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    positive_cfg = cfg.get("positive", {})
    train_cfg = cfg.get("virtual_open_training", {})
    prompt_cfg = train_cfg.get("prompt", {})
    virtual_cfg = train_cfg.get("virtual_features", {})
    loss_cfg = train_cfg.get("loss", {})
    validation_cfg = train_cfg.get("validation", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = str(
        experiment_cfg.get("name", "idea7-virtual-open-negprompt")
    )
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

    inference_mode = str(inference_cfg.get("mode", PRIMARY_INFERENCE_MODE))
    if inference_mode != PRIMARY_INFERENCE_MODE:
        raise ValueError(
            f"Idea7 primary inference must be {PRIMARY_INFERENCE_MODE!r}"
        )
    allow_root_unknown = bool(inference_cfg.get("allow_root_unknown", False))
    if allow_root_unknown:
        raise ValueError("The FGVC Idea7 protocol disables root unknown")
    unknown_aggregation = str(
        inference_cfg.get("unknown_aggregation", "logmeanexp")
    )
    if unknown_aggregation != "logmeanexp":
        raise ValueError("Idea7 requires unknown_aggregation='logmeanexp'")

    def artifact(configured, kind: str, filename: str) -> str:
        return str(resolve_experiment_artifact(
            configured,
            output_root=output_root,
            experiment_name=experiment_name,
            kind=kind,
            default_filename=filename,
        ))

    mix_min = float(virtual_cfg.get("mix_min", 0.35))
    mix_max = float(virtual_cfg.get("mix_max", 0.65))
    if not 0.0 <= mix_min <= mix_max <= 1.0:
        raise ValueError("virtual_features mix range must lie inside [0, 1]")

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
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
        epochs=max(1, int(train_cfg.get("epochs", 30))),
        parents_per_step=max(1, int(train_cfg.get("parents_per_step", 4))),
        max_examples_per_parent=max(
            2, int(train_cfg.get("max_examples_per_parent", 64))
        ),
        vision_batch_size=max(1, int(train_cfg.get("vision_batch_size", 16))),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        num_unknown_prompts=max(1, int(prompt_cfg.get("count", 4))),
        unknown_prototype_ctx_tokens=max(
            0, int(prompt_cfg.get("prototype_ctx_tokens", 2))
        ),
        virtual_samples_per_parent=max(
            1, int(virtual_cfg.get("samples_per_parent", 32))
        ),
        mix_min=mix_min,
        mix_max=mix_max,
        refinement_steps=max(0, int(virtual_cfg.get("refinement_steps", 5))),
        refinement_step_size=float(virtual_cfg.get("refinement_step_size", 0.1)),
        refinement_child_temperature=float(
            virtual_cfg.get("child_temperature", 0.07)
        ),
        refinement_parent_weight=float(virtual_cfg.get("parent_weight", 0.5)),
        refinement_anchor_weight=float(virtual_cfg.get("anchor_weight", 0.25)),
        lambda_virtual=float(loss_cfg.get("lambda_virtual", 1.0)),
        lambda_id_teacher=float(loss_cfg.get("lambda_id_teacher", 1.0)),
        lambda_coverage=float(loss_cfg.get("lambda_coverage", 0.2)),
        lambda_diversity=float(loss_cfg.get("lambda_diversity", 0.05)),
        diversity_margin=float(loss_cfg.get("diversity_margin", 0.2)),
        depth_balanced=bool(loss_cfg.get("depth_balanced", True)),
        validation_every_n_epochs=max(
            1, int(validation_cfg.get("every_n_epochs", 1))
        ),
        validation_batch_size=max(1, int(validation_cfg.get("batch_size", 64))),
        max_id_bacc_drop=float(validation_cfg.get("max_id_bacc_drop", 0.08)),
        inference_mode=inference_mode,
        inference_batch_size=max(1, int(inference_cfg.get("batch_size", 64))),
        inference_tau=float(
            inference_cfg.get("tau", 1.0 / float(train_cfg.get("tau", 0.07)))
        ),
        allow_root_unknown=allow_root_unknown,
        unknown_aggregation=unknown_aggregation,
        greedy_ablation=bool(inference_cfg.get("greedy_ablation", True)),
        checkpoint=artifact(
            train_cfg.get("checkpoint"),
            "checkpoints",
            f"{experiment_name}.pt",
        ),
        result_path=artifact(
            train_cfg.get("result_path"),
            "results",
            f"{experiment_name}-global-path.result",
        ),
        diagnostics_path=artifact(
            train_cfg.get("diagnostics_path"),
            "diagnostics",
            f"{experiment_name}-diagnostics.json",
        ),
        oracle_diagnostics_path=artifact(
            cfg.get("oracle_diagnostics", {}).get("path"),
            "diagnostics",
            f"{experiment_name}-oracle-parent.json",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def average_stats(stats: list[dict]) -> dict:
    if not stats:
        return {}
    keys = set.intersection(*(set(item) for item in stats))
    return {
        key: sum(float(item[key]) for item in stats) / len(stats)
        for key in sorted(keys)
    }


def save_checkpoint(args, positive_checkpoint, prompt_cfg, unknown, metrics):
    return save_idea3_checkpoint(
        args.checkpoint,
        stage=CHECKPOINT_STAGE,
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=positive_checkpoint["positive_state_dict"],
        unknown_state_dict=prompt_only_state_dict(unknown),
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
    positive_index = build_positive_semantic_index(hierarchy, positive)
    prompt_cfg = replace(
        prompt_cfg,
        unknown_prompts=args.num_unknown_prompts,
        unknown_prototype_ctx_tokens=args.unknown_prototype_ctx_tokens,
    )
    unknown = UnknownPromptLearner(
        args.dataset, hierarchy, text_encoder, prompt_cfg
    ).to(device)
    trainable_params = prompt_parameters(unknown)
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise RuntimeError("Idea7 requires every CLIP backbone parameter to be frozen")
    if any(parameter.requires_grad for parameter in positive.parameters()):
        raise RuntimeError("Idea7 requires every positive prompt parameter to be frozen")
    if not trainable_params:
        raise RuntimeError("Idea7 found no trainable negative prompt parameters")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    train_payload = dataset_payload(train_dataset)
    val_payload = encode_dataset_features(
        args, clip_model, val_dataset, val_loader, device, "encode ID val"
    )
    train_groups = group_examples_by_parent_child(
        build_positive_edge_examples(hierarchy, train_payload)
    )
    parents = sorted(
        parent
        for parent, child_map in train_groups.items()
        if parent != "root" and len(child_map) >= 2
    )
    if not parents:
        raise RuntimeError("Idea7 found no non-root parent with at least two children")

    depth_counts = {}
    for parent in parents:
        depth = len(hierarchy.node_ancestors.get(parent, []))
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    parent_weights = {
        parent: len(parents) / (len(depth_counts) * depth_counts[
            len(hierarchy.node_ancestors.get(parent, []))
        ])
        if args.depth_balanced else 1.0
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
        "Idea7 virtual-open negative prompt training: "
        f"vision_lora_modules={len(replaced_modules)}, "
        f"trainable_negative_params={sum(p.numel() for p in trainable_params)}, "
        "negative_only_training=True, "
        f"parents={len(parents)}, unknown_prompts={args.num_unknown_prompts}, "
        f"positive_val_bacc={positive_baseline_bacc:.6f}, "
        f"id_bacc_floor={id_bacc_floor:.6f}"
    )

    rng = random.Random(args.seed)
    history = []
    best_guarded = None
    best_id = None
    for epoch in range(1, args.epochs + 1):
        unknown.train()
        positive.eval()
        clip_model.eval()
        set_vision_lora_train_mode(clip_model, False)
        shuffled = list(parents)
        rng.shuffle(shuffled)
        chunks = [
            shuffled[start:start + args.parents_per_step]
            for start in range(0, len(shuffled), args.parents_per_step)
        ]
        iterator = (
            tqdm(chunks, desc=f"idea7 epoch {epoch}/{args.epochs}", leave=False)
            if tqdm else chunks
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
            feature_by_index = encode_selected_train_images(
                args, clip_model, train_dataset, image_indices, device
            )

            optimizer.zero_grad(set_to_none=True)
            unknown_banks = unknown.encode_unknown_prototypes(
                [episode.parent for episode in episodes]
            )
            losses = []
            weights = []
            parent_stats = []
            for bank_index, episode in enumerate(episodes):
                local = positive_index[episode.parent]
                children = list(local.children)
                positive_features = local.child_features.to(device)
                child_to_target = {
                    child: index for index, child in enumerate(children)
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
                virtual = spherical_sibling_mixup(
                    image_features,
                    targets,
                    num_samples=args.virtual_samples_per_parent,
                    mix_min=args.mix_min,
                    mix_max=args.mix_max,
                    rng=rng,
                )
                virtual, refine_stats = refine_virtual_open_features(
                    virtual,
                    positive_features,
                    unknown._parent_feature(episode.parent),
                    steps=args.refinement_steps,
                    step_size=args.refinement_step_size,
                    child_temperature=args.refinement_child_temperature,
                    parent_weight=args.refinement_parent_weight,
                    anchor_weight=args.refinement_anchor_weight,
                )
                loss, stats = virtual_open_negprompt_loss(
                    image_features,
                    positive_features,
                    unknown_banks[bank_index],
                    virtual,
                    tau=args.tau,
                    lambda_virtual=args.lambda_virtual,
                    lambda_id_teacher=args.lambda_id_teacher,
                    lambda_coverage=args.lambda_coverage,
                    lambda_diversity=args.lambda_diversity,
                    diversity_margin=args.diversity_margin,
                )
                stats.update(refine_stats)
                losses.append(loss)
                weights.append(parent_weights[episode.parent])
                parent_stats.append(stats)

            batch_loss = torch.stack([
                loss * weight for loss, weight in zip(losses, weights)
            ]).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
            optimizer.step()
            stats = average_stats(parent_stats)
            stats["loss"] = float(batch_loss.detach().cpu())
            step_stats.append(stats)

        scheduler.step()
        epoch_stats = average_stats(step_stats)
        epoch_stats.update({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "steps": len(step_stats),
        })

        if epoch % args.validation_every_n_epochs == 0 or epoch == args.epochs:
            semantic_index = build_unknown_semantic_index(
                hierarchy, positive_index, unknown
            )
            val_result = evaluate_feature_payload(
                args,
                hierarchy,
                val_payload,
                semantic_index,
                "val",
                mode=PRIMARY_INFERENCE_MODE,
            )
            val_bacc = float(val_result["metrics"]["balanced_acc"])
            guard_passed = val_bacc >= id_bacc_floor
            epoch_stats.update({
                "val_balanced_acc": val_bacc,
                "val_balanced_hdist": float(
                    val_result["metrics"]["balanced_hdist"]
                ),
                "val_unknown_selection_rate": float(
                    val_result["diagnostics"]["unknown_selection_rate"]
                ),
                "id_guard_passed": guard_passed,
            })
            state = clone_prompt_state(unknown)
            if best_id is None or val_bacc > best_id[0]:
                best_id = (val_bacc, epoch, state)
            open_objective = float(epoch_stats.get("open_objective", float("inf")))
            candidate = (open_objective, -val_bacc, epoch, state)
            if guard_passed and (best_guarded is None or candidate[:3] < best_guarded[:3]):
                best_guarded = candidate

        history.append(epoch_stats)
        message = (
            f"epoch {epoch}: loss={epoch_stats.get('loss', 0.0):.6f}, "
            f"open={epoch_stats.get('open_objective', 0.0):.6f}, "
            f"virtual_recall={epoch_stats.get('virtual_unknown_recall', 0.0):.6f}, "
            f"id_unknown={epoch_stats.get('id_unknown_selection_rate', 0.0):.6f}, "
            f"refined_child_energy={epoch_stats.get('virtual_refined_child_energy', 0.0):.6f}"
        )
        if "val_balanced_acc" in epoch_stats:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"id_guard={epoch_stats['id_guard_passed']}"
            )
        print(message)

    if best_guarded is not None:
        selection_policy = "min_virtual_open_objective_under_id_guard"
        selected_objective, neg_selected_bacc, selected_epoch, selected_state = best_guarded
        selected_val_bacc = -neg_selected_bacc
        selection_guard_passed = True
    elif best_id is not None:
        selection_policy = "fallback_max_id_bacc_no_guarded_epoch"
        selected_val_bacc, selected_epoch, selected_state = best_id
        selected_objective = next(
            float(item.get("open_objective", float("nan")))
            for item in history if item["epoch"] == selected_epoch
        )
        selection_guard_passed = False
    else:
        raise RuntimeError("Idea7 did not produce a validation checkpoint")
    load_prompt_only_state_dict(unknown, selected_state)

    semantic_index = build_unknown_semantic_index(hierarchy, positive_index, unknown)
    val_result = evaluate_feature_payload(
        args,
        hierarchy,
        val_payload,
        semantic_index,
        "val",
        mode=PRIMARY_INFERENCE_MODE,
    )
    ood_payload = encode_dataset_features(
        args, clip_model, ood_dataset, ood_loader, device, "encode OOD"
    )
    ood_result = evaluate_feature_payload(
        args,
        hierarchy,
        ood_payload,
        semantic_index,
        "ood",
        mode=PRIMARY_INFERENCE_MODE,
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])
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

    ablations = {}
    if args.greedy_ablation:
        greedy_val = evaluate_feature_payload(
            args, hierarchy, val_payload, semantic_index, "val", mode=GREEDY_INFERENCE_MODE
        )
        greedy_ood = evaluate_feature_payload(
            args, hierarchy, ood_payload, semantic_index, "ood", mode=GREEDY_INFERENCE_MODE
        )
        ablations[GREEDY_INFERENCE_MODE] = {
            "val": greedy_val,
            "ood": greedy_ood,
            "mixed": mixed_summary(greedy_val["metrics"], greedy_ood["metrics"]),
        }

    selection = {
        "policy": selection_policy,
        "selected_epoch": selected_epoch,
        "selected_open_objective": selected_objective,
        "selected_validation_bacc": selected_val_bacc,
        "used_ood_for_selection": False,
        "id_bacc_floor": id_bacc_floor,
        "id_guard_passed": selection_guard_passed,
    }
    final_metrics = {
        "train_history": history,
        "positive_baseline": scalar_metric_summary(positive_baseline),
        "selection": selection,
        "final": {
            "val": scalar_metric_summary(val_result),
            "ood": scalar_metric_summary(ood_result),
            "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
            "mixed_balanced_hdist": float(mixed["mixed_balanced_hdist"]),
        },
        "oracle_parent": oracle_result,
    }
    save_checkpoint(args, positive_checkpoint, prompt_cfg, unknown, final_metrics)
    result = {
        "args": vars(args),
        "mode": PRIMARY_INFERENCE_MODE,
        "method": "virtual_open_negprompt",
        "checkpoint": args.checkpoint,
        "checkpoint_stage": CHECKPOINT_STAGE,
        "positive_checkpoint": args.positive_checkpoint,
        "selected_epoch": selected_epoch,
        "selection_policy": selection_policy,
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

    print(
        f"selected epoch: {selected_epoch} ({selection_policy}; OOD not used)"
    )
    print(f"ID guard passed: {selection_guard_passed}")
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
