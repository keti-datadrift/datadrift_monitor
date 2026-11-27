from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import replace
import math
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

from negzerohoc.checkpointing import (
    load_idea3_checkpoint_with_fallback,
    save_idea3_checkpoint,
)
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.metric_terminal import (
    build_metric_terminal_specs,
    metric_terminal_scores,
)
from negzerohoc.negative_metric_terminal import (
    global_metric_terminal_negprompt_loss,
    leave_one_child_out_terminal_recall,
    threshold_terminal_winner_indices,
)
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.prompt_models import UnknownPromptLearner
from negzerohoc.prompt_text import UNKNOWN_TEXT_VARIANTS
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
from negzerohoc.virtual_open_negprompt import spherical_sibling_mixup
from negzerohoc.vision_lora import set_vision_lora_train_mode
from scripts.infer_metric_terminal_positive import encode_all_positive_edges
from scripts.train_idea3_joint_vision_lora import (
    build_transforms,
    load_prompt_only_state_dict,
    make_loader,
    prompt_only_state_dict,
)
from scripts.train_idea4_unknown_prompts import (
    clone_prompt_state,
    dataset_payload,
    encode_dataset_features,
    load_frozen_positive_stack,
    prompt_parameters,
)


CHECKPOINT_STAGE = "negative_text_global_metric_terminal"


def comma_floats(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = [
            float(item.strip())
            for item in str(value).split(",")
            if item.strip()
        ]
    if not values:
        raise ValueError("Expected at least one float")
    return values


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
    virtual_cfg = train_cfg.get("virtual_features", {})
    loss_cfg = train_cfg.get("loss", {})
    validation_cfg = train_cfg.get("validation", {})
    decoder_cfg = cfg.get("metric_terminal_decoder", {})
    resume_cfg = train_cfg.get("resume", {})

    experiment_name = str(experiment_cfg.get(
        "name", "negative-text-metric-terminal"
    ))
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")
    text_variant = str(prompt_cfg.get(
        "text_variant", "parent_conditioned"
    ))
    if text_variant not in UNKNOWN_TEXT_VARIANTS:
        raise ValueError(
            f"Unsupported negative text variant {text_variant!r}; "
            f"expected one of {sorted(UNKNOWN_TEXT_VARIANTS)}"
        )
    if bool(decoder_cfg.get("allow_root_unknown", False)):
        raise ValueError("This protocol disables root unknown")

    def artifact(configured, kind: str, filename: str) -> str:
        return str(resolve_experiment_artifact(
            configured,
            output_root=output_root,
            experiment_name=experiment_name,
            kind=kind,
            default_filename=filename,
        ))

    checkpoint = artifact(
        train_cfg.get("checkpoint"),
        "checkpoints",
        "negative.pt",
    )
    last_checkpoint = artifact(
        validation_cfg.get("last_checkpoint"),
        "checkpoints",
        "negative-last.pt",
    )
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
        epochs=max(1, int(train_cfg.get("epochs", 50))),
        parents_per_step=max(
            1, int(train_cfg.get("parents_per_step", 4))
        ),
        max_examples_per_parent=max(
            2, int(train_cfg.get("max_examples_per_parent", 64))
        ),
        feature_batch_size=max(
            1, int(train_cfg.get("feature_batch_size", 64))
        ),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(
            train_cfg.get("gradient_clip_norm", 1.0)
        ),
        num_unknown_prompts=max(
            1, int(prompt_cfg.get("count", 1))
        ),
        unknown_prototype_ctx_tokens=max(
            0, int(prompt_cfg.get("prototype_ctx_tokens", 2))
        ),
        unknown_text_variant=text_variant,
        virtual_samples_per_parent=max(
            1, int(virtual_cfg.get("samples_per_parent", 32))
        ),
        mix_min=float(virtual_cfg.get("mix_min", 0.35)),
        mix_max=float(virtual_cfg.get("mix_max", 0.65)),
        loss_temperature=float(loss_cfg.get("temperature", 0.07)),
        loss_terminal_weight=float(
            loss_cfg.get("terminal_weight", 1.0)
        ),
        loss_bottleneck_temperature=float(
            loss_cfg.get("bottleneck_temperature", 0.1)
        ),
        unknown_temperature=float(
            loss_cfg.get("unknown_temperature", 0.07)
        ),
        lambda_virtual=float(loss_cfg.get("lambda_virtual", 1.0)),
        lambda_id_teacher=float(
            loss_cfg.get("lambda_id_teacher", 1.0)
        ),
        lambda_coverage=float(
            loss_cfg.get("lambda_coverage", 0.2)
        ),
        lambda_diversity=float(
            loss_cfg.get("lambda_diversity", 0.05)
        ),
        diversity_margin=float(
            loss_cfg.get("diversity_margin", 0.2)
        ),
        depth_balanced=bool(loss_cfg.get("depth_balanced", True)),
        validation_every_n_epochs=max(
            1, int(validation_cfg.get("every_n_epochs", 5))
        ),
        validation_batch_size=max(
            1, int(validation_cfg.get("batch_size", 128))
        ),
        max_id_bacc_drop=float(
            validation_cfg.get("max_id_bacc_drop", 0.03)
        ),
        terminal_weights=comma_floats(
            decoder_cfg.get("terminal_weights", [0.75, 1.0])
        ),
        bottleneck_temperatures=comma_floats(
            decoder_cfg.get("bottleneck_temperatures", [0.1])
        ),
        unknown_thresholds=comma_floats(
            decoder_cfg.get(
                "unknown_thresholds",
                [-0.1, -0.075, -0.05, -0.025, 0.0,
                 0.025, 0.05, 0.075, 0.1, 0.15],
            )
        ),
        allow_root_unknown=False,
        checkpoint=checkpoint,
        last_checkpoint=last_checkpoint,
        result_path=artifact(
            train_cfg.get("result_path"),
            "results",
            "validation.result",
        ),
        diagnostics_path=artifact(
            train_cfg.get("diagnostics_path"),
            "diagnostics",
            "validation.json",
        ),
        automatic_resume=bool(resume_cfg.get("automatic", True)),
        resume_checkpoint=str(
            resume_cfg.get("checkpoint") or last_checkpoint
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def build_id_datasets(args):
    from negzerohoc.prohoc_compat.utils.dataset_util import (
        SubsetImageFolder,
        get_id_classes,
    )

    train_transform, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    train_dataset = SubsetImageFolder(
        datadir / "train", id_classes, transform=train_transform
    )
    val_dataset = SubsetImageFolder(
        datadir / "val", id_classes, transform=eval_transform
    )
    empty_files = sorted({
        path
        for dataset in (train_dataset, val_dataset)
        for path, _ in dataset.samples
        if Path(path).stat().st_size == 0
    })
    if empty_files:
        raise RuntimeError(
            f"Found {len(empty_files)} zero-byte ID image files; "
            f"examples: {empty_files[:3]}"
        )
    print(f"# ID Train: {len(train_dataset)}")
    print(f"# ID Val: {len(val_dataset)}")
    print("# OOD: not constructed during training/model selection")
    return train_dataset, val_dataset


def average_stats(stats: list[dict]) -> dict:
    if not stats:
        return {}
    keys = set.intersection(*(set(item) for item in stats))
    return {
        key: sum(float(item[key]) for item in stats) / len(stats)
        for key in sorted(keys)
    }


def metric_summary(metrics: dict) -> dict:
    return {
        "acc": float(metrics["acc"]),
        "balanced_acc": float(metrics["balanced_acc"]),
        "avg_hdist": float(metrics["avg_hdist"]),
        "balanced_hdist": float(metrics["balanced_hdist"]),
    }


def retained_target_names(hierarchy, payload: dict) -> list[str]:
    mapping = hierarchy.gen_ds2node_map(list(payload["classes"]))
    indices = mapping[payload["targets"].long()].tolist()
    return [hierarchy.id_node_list[int(index)] for index in indices]


def terminal_indices_to_node_predictions(
    hierarchy,
    terminal_specs,
    winner_indices: torch.Tensor,
) -> torch.Tensor:
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    return torch.tensor([
        node_to_index[terminal_specs[int(index)].node]
        for index in winner_indices.detach().cpu().tolist()
    ], dtype=torch.long)


def encode_all_unknowns(unknown, parents: list[str]) -> dict[str, torch.Tensor]:
    banks = unknown.encode_unknown_prototypes(parents)
    return {
        parent: banks[index]
        for index, parent in enumerate(parents)
    }


def decoder_grid(
    args,
    hierarchy,
    val_payload,
    retained_targets,
    edge_features,
    terminal_specs,
    unknown_features_by_parent,
    distance_mats,
) -> tuple[dict, list[dict]]:
    rows = []
    selected = None
    selected_order = None
    seen_score_settings = set()
    for terminal_weight in args.terminal_weights:
        bottleneck_values = (
            [args.bottleneck_temperatures[0]]
            if math.isclose(float(terminal_weight), 1.0)
            else args.bottleneck_temperatures
        )
        for bottleneck_temperature in bottleneck_values:
            score_key = (
                float(terminal_weight),
                float(bottleneck_temperature),
            )
            if score_key in seen_score_settings:
                continue
            seen_score_settings.add(score_key)
            score_matrix = metric_terminal_scores(
                val_payload["features"],
                edge_features,
                terminal_specs,
                unknown_features_by_parent=unknown_features_by_parent,
                terminal_weight=terminal_weight,
                bottleneck_temperature=bottleneck_temperature,
                unknown_temperature=args.unknown_temperature,
            )["score_matrix"]
            for unknown_threshold in args.unknown_thresholds:
                winner_indices = threshold_terminal_winner_indices(
                    score_matrix,
                    terminal_specs,
                    unknown_threshold=unknown_threshold,
                )
                predictions = terminal_indices_to_node_predictions(
                    hierarchy, terminal_specs, winner_indices
                )
                _, metrics = evaluate_split(
                    hierarchy,
                    val_payload,
                    predictions,
                    dists_mats=distance_mats,
                )
                summary = metric_summary(metrics)
                loo = leave_one_child_out_terminal_recall(
                    score_matrix,
                    terminal_specs,
                    hierarchy,
                    retained_targets,
                    unknown_threshold=unknown_threshold,
                )
                candidate_kinds = [
                    terminal_specs[int(index)].kind
                    for index in winner_indices.detach().cpu().tolist()
                ]
                row = {
                    "terminal_weight": float(terminal_weight),
                    "bottleneck_temperature": float(
                        bottleneck_temperature
                    ),
                    "unknown_threshold": float(unknown_threshold),
                    **summary,
                    "unknown_selection_rate": (
                        sum(kind == "unknown" for kind in candidate_kinds)
                        / len(candidate_kinds)
                    ),
                    "loo_fold_macro_recall": float(
                        loo["fold_macro_recall"]
                    ),
                    "loo_parent_macro_recall": float(
                        loo["parent_macro_recall"]
                    ),
                    "loo_sample_recall": float(loo["sample_recall"]),
                    "loo_fold_count": int(loo["fold_count"]),
                    "id_guard_passed": (
                        summary["balanced_acc"] >= args.id_bacc_floor
                    ),
                }
                rows.append(row)
                if row["id_guard_passed"]:
                    order = (
                        row["loo_fold_macro_recall"],
                        row["loo_parent_macro_recall"],
                        row["balanced_acc"],
                        -row["balanced_hdist"],
                    )
                    if selected_order is None or order > selected_order:
                        selected = row
                        selected_order = order

    if selected is None:
        selected = max(
            rows,
            key=lambda row: (
                row["balanced_acc"],
                -row["balanced_hdist"],
                row["loo_fold_macro_recall"],
            ),
        )
        selected = dict(selected)
        selected["selection_fallback"] = (
            "no_decoder_setting_passed_id_guard"
        )
    else:
        selected = dict(selected)
        selected["selection_fallback"] = None

    final_scores = metric_terminal_scores(
        val_payload["features"],
        edge_features,
        terminal_specs,
        unknown_features_by_parent=unknown_features_by_parent,
        terminal_weight=selected["terminal_weight"],
        bottleneck_temperature=selected["bottleneck_temperature"],
        unknown_temperature=args.unknown_temperature,
    )["score_matrix"]
    selected["loo"] = leave_one_child_out_terminal_recall(
        final_scores,
        terminal_specs,
        hierarchy,
        retained_targets,
        unknown_threshold=selected["unknown_threshold"],
    )
    return selected, rows


def checkpoint_signature(args) -> dict:
    return {
        "config": str(Path(args.config).resolve()),
        "positive_checkpoint": str(
            Path(args.positive_checkpoint).resolve()
        ),
        "text_variant": args.unknown_text_variant,
        "unknown_prompts": args.num_unknown_prompts,
        "epochs": args.epochs,
    }


def save_negative_checkpoint(
    args,
    positive_checkpoint,
    prompt_cfg,
    unknown_state,
    metrics,
    path,
    training_state=None,
):
    return save_idea3_checkpoint(
        path,
        stage=CHECKPOINT_STAGE,
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=positive_checkpoint["positive_state_dict"],
        unknown_state_dict=unknown_state,
        vision_lora_config=positive_checkpoint["vision_lora_config"],
        vision_lora_state_dict=positive_checkpoint[
            "vision_lora_state_dict"
        ],
        positive_checkpoint=args.positive_checkpoint,
        metrics=metrics,
        args=vars(args),
        training_state=training_state,
    )


def main():
    args = parse_args()
    configure_reproducibility(
        args.seed, deterministic=args.deterministic
    )
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    train_dataset, val_dataset = build_id_datasets(args)
    train_loader = make_loader(
        train_dataset,
        args.feature_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    val_loader = make_loader(
        val_dataset,
        args.validation_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    (
        positive_checkpoint,
        clip_model,
        text_encoder,
        positive_prompt_cfg,
        positive,
        replaced_modules,
    ) = load_frozen_positive_stack(args, hierarchy, device)
    prompt_cfg = replace(
        positive_prompt_cfg,
        unknown_prompts=args.num_unknown_prompts,
        unknown_prototype_ctx_tokens=(
            args.unknown_prototype_ctx_tokens
        ),
        unknown_text_variant=args.unknown_text_variant,
    )
    unknown = UnknownPromptLearner(
        args.dataset, hierarchy, text_encoder, prompt_cfg
    ).to(device)
    trainable_params = prompt_parameters(unknown)
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise RuntimeError("Every CLIP parameter must be frozen")
    if any(parameter.requires_grad for parameter in positive.parameters()):
        raise RuntimeError("Every positive prompt parameter must be frozen")
    if not trainable_params:
        raise RuntimeError("No trainable negative prompt parameters found")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    train_payload = encode_dataset_features(
        args, clip_model, train_dataset, train_loader, device,
        "encode frozen train features",
    )
    val_payload = encode_dataset_features(
        args, clip_model, val_dataset, val_loader, device,
        "encode frozen ID validation features",
    )
    edge_features_cpu = encode_all_positive_edges(
        hierarchy, positive
    )
    edge_features = {
        edge: feature.to(device)
        for edge, feature in edge_features_cpu.items()
    }
    parents = sorted(
        (
            parent for parent in hierarchy.parent2children
            if parent != "root"
        ),
        key=lambda node: (
            len(hierarchy.node_ancestors.get(node, [])),
            node,
        ),
    )
    terminal_specs = build_metric_terminal_specs(
        hierarchy,
        parents,
        allow_root_unknown=args.allow_root_unknown,
    )
    distance_mats = make_distance_mats(hierarchy)
    retained_targets = retained_target_names(hierarchy, val_payload)

    known_specs = [
        spec for spec in terminal_specs if spec.unknown_parent is None
    ]
    positive_scores = metric_terminal_scores(
        val_payload["features"],
        edge_features,
        known_specs,
        terminal_weight=1.0,
        bottleneck_temperature=args.loss_bottleneck_temperature,
    )["score_matrix"]
    positive_winners = positive_scores.argmax(dim=1)
    positive_predictions = terminal_indices_to_node_predictions(
        hierarchy, known_specs, positive_winners
    )
    _, positive_metrics = evaluate_split(
        hierarchy,
        val_payload,
        positive_predictions,
        dists_mats=distance_mats,
    )
    positive_baseline = metric_summary(positive_metrics)
    args.id_bacc_floor = (
        positive_baseline["balanced_acc"] - args.max_id_bacc_drop
    )

    train_groups = group_examples_by_parent_child(
        build_positive_edge_examples(hierarchy, train_payload)
    )
    parents = [
        parent for parent in parents
        if parent in train_groups and len(train_groups[parent]) >= 2
    ]
    if len(parents) != 28:
        print(
            f"warning: expected 28 trainable non-root parents, got "
            f"{len(parents)}"
        )
    depth_counts = {}
    for parent in parents:
        depth = len(hierarchy.node_ancestors.get(parent, []))
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    parent_weights = {
        parent: (
            len(parents)
            / (
                len(depth_counts)
                * depth_counts[
                    len(hierarchy.node_ancestors.get(parent, []))
                ]
            )
            if args.depth_balanced else 1.0
        )
        for parent in parents
    }

    history = []
    best_state = None
    best_selection = None
    best_epoch = None
    best_rank = None
    start_epoch = 1
    rng = random.Random(args.seed)
    resume_path = Path(args.resume_checkpoint)
    if args.automatic_resume and (
        resume_path.exists()
        or resume_path.with_name(
            f"{resume_path.stem}-previous{resume_path.suffix}"
        ).exists()
    ):
        resume_payload, loaded_path = load_idea3_checkpoint_with_fallback(
            resume_path, map_location="cpu"
        )
        state = resume_payload["training_state"]
        if state.get("resume_signature") != checkpoint_signature(args):
            raise ValueError(
                "Negative training resume signature mismatch: "
                f"{loaded_path}"
            )
        load_prompt_only_state_dict(
            unknown, resume_payload["unknown_state_dict"]
        )
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = list(state["history"])
        best_state = state.get("best_unknown_state_dict")
        best_selection = state.get("best_selection")
        best_epoch = state.get("best_epoch")
        best_rank = state.get("best_rank")
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"resumed {loaded_path} at epoch {state['epoch']}; "
            f"next={start_epoch}"
        )

    print(
        "negative text metric-terminal training: "
        f"device={device}, lora_modules={len(replaced_modules)}, "
        f"text={args.unknown_text_variant}, K={args.num_unknown_prompts}, "
        f"negative_only=True, actual_ood_images_encoded=False, "
        f"positive_ID_BAcc={positive_baseline['balanced_acc']:.6f}, "
        f"ID_floor={args.id_bacc_floor:.6f}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
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
            tqdm(
                chunks,
                desc=f"negative epoch {epoch}/{args.epochs}",
                leave=False,
            )
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
            episodes = [episode for episode in episodes if episode]
            if not episodes:
                continue
            id_features = []
            virtual_features = []
            virtual_parents = []
            virtual_weights = []
            for episode in episodes:
                indices = torch.tensor(
                    [
                        example.image_index
                        for example in episode.examples
                    ],
                    dtype=torch.long,
                )
                images = train_payload["features"].index_select(
                    0, indices
                ).to(device)
                child_to_target = {
                    child: index
                    for index, child in enumerate(episode.children)
                }
                targets = torch.tensor(
                    [
                        child_to_target[label]
                        for label in episode.labels
                    ],
                    dtype=torch.long,
                    device=device,
                )
                virtual = spherical_sibling_mixup(
                    images,
                    targets,
                    num_samples=args.virtual_samples_per_parent,
                    mix_min=args.mix_min,
                    mix_max=args.mix_max,
                    rng=rng,
                )
                weight = parent_weights[episode.parent]
                id_features.append(images)
                virtual_features.append(virtual)
                virtual_parents.extend(
                    [episode.parent] * int(virtual.shape[0])
                )
                virtual_weights.append(weight)

            optimizer.zero_grad(set_to_none=True)
            unknown_banks = encode_all_unknowns(unknown, parents)
            loss, stats = global_metric_terminal_negprompt_loss(
                torch.cat(id_features, dim=0),
                torch.cat(virtual_features, dim=0),
                virtual_parents,
                edge_features,
                terminal_specs,
                unknown_banks,
                loss_temperature=args.loss_temperature,
                terminal_weight=args.loss_terminal_weight,
                bottleneck_temperature=(
                    args.loss_bottleneck_temperature
                ),
                unknown_temperature=args.unknown_temperature,
                lambda_virtual=args.lambda_virtual,
                lambda_id_teacher=args.lambda_id_teacher,
                lambda_coverage=args.lambda_coverage,
                lambda_diversity=args.lambda_diversity,
                diversity_margin=args.diversity_margin,
            )
            mean_parent_weight = sum(virtual_weights) / len(
                virtual_weights
            )
            weighted_loss = loss * mean_parent_weight
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params, args.gradient_clip_norm
            )
            optimizer.step()
            stats["loss"] = float(weighted_loss.detach().cpu())
            step_stats.append(stats)

        scheduler.step()
        epoch_stats = average_stats(step_stats)
        epoch_stats.update({
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "steps": len(step_stats),
        })
        is_validation_epoch = (
            epoch % args.validation_every_n_epochs == 0
            or epoch == args.epochs
        )
        if is_validation_epoch:
            unknown.eval()
            with torch.no_grad():
                unknown_banks = encode_all_unknowns(unknown, parents)
                selection, grid = decoder_grid(
                    args,
                    hierarchy,
                    val_payload,
                    retained_targets,
                    edge_features,
                    terminal_specs,
                    unknown_banks,
                    distance_mats,
                )
            epoch_stats["decoder_selection"] = {
                key: value
                for key, value in selection.items()
                if key != "loo"
            }
            epoch_stats["decoder_grid"] = grid
            rank = (
                float(selection["loo_fold_macro_recall"]),
                float(selection["loo_parent_macro_recall"]),
                float(selection["balanced_acc"]),
                -float(selection["balanced_hdist"]),
            )
            if best_rank is None or tuple(rank) > tuple(best_rank):
                best_rank = rank
                best_state = clone_prompt_state(unknown)
                best_selection = selection
                best_epoch = epoch

        history.append(epoch_stats)
        message = (
            f"epoch {epoch}: loss={epoch_stats.get('loss', 0):.6f}, "
            f"virtual={epoch_stats.get('virtual_exact_parent_recall', 0):.6f}, "
            f"id_unknown={epoch_stats.get('id_unknown_selection_rate', 0):.6f}"
        )
        if "decoder_selection" in epoch_stats:
            selected = epoch_stats["decoder_selection"]
            message += (
                f", ID_BAcc={selected['balanced_acc']:.6f}, "
                f"LOO={selected['loo_fold_macro_recall']:.6f}, "
                f"threshold={selected['unknown_threshold']:.3f}"
            )
            if epoch == best_epoch:
                message += " [best]"
        print(message)

        training_state = {
            "version": 1,
            "epoch": epoch,
            "resume_signature": checkpoint_signature(args),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
            "history": history,
            "best_unknown_state_dict": best_state,
            "best_selection": best_selection,
            "best_epoch": best_epoch,
            "best_rank": best_rank,
            "training_loop_complete": epoch >= args.epochs,
        }
        save_negative_checkpoint(
            args,
            positive_checkpoint,
            prompt_cfg,
            prompt_only_state_dict(unknown),
            {
                "positive_baseline": positive_baseline,
                "id_bacc_floor": args.id_bacc_floor,
                "best_epoch": best_epoch,
                "best_selection": best_selection,
            },
            args.last_checkpoint,
            training_state=training_state,
        )

    if best_state is None or best_selection is None:
        raise RuntimeError("No validation checkpoint was selected")
    load_prompt_only_state_dict(unknown, best_state)
    final_metrics = {
        "positive_baseline": positive_baseline,
        "id_bacc_floor": args.id_bacc_floor,
        "selection_policy": (
            "maximize leave-one-child-out fold-macro exact parent-unknown "
            "recall under the ID BAcc guard; parent-macro recall, ID BAcc, "
            "and ID BMHD are tie-breakers"
        ),
        "selection_used_actual_ood": False,
        "selected_epoch": best_epoch,
        "selected_decoder": best_selection,
        "train_history": history,
    }
    save_negative_checkpoint(
        args,
        positive_checkpoint,
        prompt_cfg,
        best_state,
        final_metrics,
        args.checkpoint,
    )
    result = {
        "method": "negative_text_global_metric_terminal",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "positive_checkpoint": args.positive_checkpoint,
        "negative_text_variant": args.unknown_text_variant,
        "unknown_prompt_count": args.num_unknown_prompts,
        "selected_epoch": best_epoch,
        "selected_decoder": best_selection,
        "positive_baseline": positive_baseline,
        "actual_ood_evaluated": False,
        "actual_ood_used_for_training_or_selection": False,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    save_json(args.diagnostics_path, final_metrics)
    print(f"selected epoch: {best_epoch}")
    print(
        "selected decoder: "
        f"w={best_selection['terminal_weight']}, "
        f"route_tau={best_selection['bottleneck_temperature']}, "
        f"unknown_threshold={best_selection['unknown_threshold']}, "
        f"ID_BAcc={best_selection['balanced_acc']:.6f}, "
        f"LOO={best_selection['loo_fold_macro_recall']:.6f}"
    )
    print("actual OOD was not encoded or evaluated")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved validation result: {args.result_path}")


if __name__ == "__main__":
    main()
