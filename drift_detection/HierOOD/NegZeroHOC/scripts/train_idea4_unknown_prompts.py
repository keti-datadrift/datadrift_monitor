from __future__ import annotations

import argparse
from argparse import Namespace
from dataclasses import replace
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint, save_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.idea4_inference import predict_features_terminal_global_path
from negzerohoc.losses import unknown_regularization
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner, UnknownPromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.semantic_index import LocalSemanticCandidates
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.training_data import (
    UNKNOWN_LABEL,
    build_positive_edge_examples,
    group_examples_by_parent_child,
    sample_leave_child_out_episode,
)
from negzerohoc.unknown_scoring import grouped_unknown_logits
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_train_mode,
)
from scripts.train_idea3_joint_vision_lora import (
    autocast_context,
    build_datasets,
    load_clip_and_tokenizer,
    load_prompt_only_state_dict,
    make_loader,
    prompt_only_state_dict,
)


PRIMARY_INFERENCE_MODE = "parent_unknown_global_path"
GREEDY_INFERENCE_MODE = "parent_unknown"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    dataloader_cfg = cfg.get("dataloader", {})
    augmentation_cfg = cfg.get("augmentation", {})
    positive_cfg = cfg.get("positive", {})
    train_cfg = cfg.get("unknown_training", {})
    unknown_prompt_cfg = train_cfg.get("prompt", {})
    validation_cfg = train_cfg.get("validation", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea4-parent-unknown")
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    positive_checkpoint = positive_cfg.get("checkpoint")
    if not positive_checkpoint:
        raise ValueError(f"Missing positive.checkpoint in {path}")
    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

    inference_mode = inference_cfg.get("mode", PRIMARY_INFERENCE_MODE)
    if inference_mode != PRIMARY_INFERENCE_MODE:
        raise ValueError(
            f"Idea 4 primary inference must be {PRIMARY_INFERENCE_MODE!r}, got {inference_mode!r}"
        )
    allow_root_unknown = bool(inference_cfg.get("allow_root_unknown", False))
    if allow_root_unknown:
        raise ValueError("Idea 4 FGVC protocol does not allow a root-level unknown candidate")
    unknown_aggregation = str(unknown_prompt_cfg.get("aggregation", "logmeanexp"))
    if unknown_aggregation != "logmeanexp":
        raise ValueError(
            "Idea 4 multiple unknown prompts currently require aggregation='logmeanexp'"
        )

    checkpoint = resolve_experiment_artifact(
        train_cfg.get("checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}-parent-unknown.pt",
    )
    last_checkpoint = resolve_experiment_artifact(
        validation_cfg.get("last_checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{checkpoint.stem}-last{checkpoint.suffix}",
    )

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
        tokenizer_model=clip_cfg.get("tokenizer_model", clip_cfg.get("model", "openai/clip-vit-base-patch16")),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        augmentation=augmentation_cfg,
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        positive_checkpoint=str(positive_checkpoint),
        epochs=int(train_cfg.get("epochs", 100)),
        parents_per_step=max(1, int(train_cfg.get("parents_per_step", 4))),
        max_examples_per_parent=max(2, int(train_cfg.get("max_examples_per_parent", 64))),
        vision_batch_size=max(1, int(train_cfg.get("vision_batch_size", 16))),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        hide_strategy=train_cfg.get("hide_strategy", "hide_one_child"),
        num_unknown_prompts=max(1, int(unknown_prompt_cfg.get("count", 1))),
        unknown_prototype_ctx_tokens=max(
            0, int(unknown_prompt_cfg.get("prototype_ctx_tokens", 1))
        ),
        unknown_aggregation=unknown_aggregation,
        unknown_target_weight=float(train_cfg.get("unknown_target_weight", 1.0)),
        lambda_anchor=float(train_cfg.get("lambda_anchor", 0.1)),
        lambda_child_sep=float(train_cfg.get("lambda_child_sep", 0.1)),
        lambda_prototype_diversity=float(
            train_cfg.get("lambda_prototype_diversity", 0.0)
        ),
        prototype_diversity_margin=float(
            train_cfg.get("prototype_diversity_margin", 0.2)
        ),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        validation_every_n_epochs=max(1, int(validation_cfg.get("every_n_epochs", 1))),
        validation_start_epoch=max(1, int(validation_cfg.get("start_epoch", 1))),
        validation_batch_size=max(1, int(validation_cfg.get("batch_size", 64))),
        max_id_bacc_drop=float(validation_cfg.get("max_id_bacc_drop", 0.04)),
        pseudo_max_examples_per_child=max(0, int(validation_cfg.get("pseudo_max_examples_per_child", 0))),
        inference_mode=inference_mode,
        inference_batch_size=max(1, int(inference_cfg.get("batch_size", 64))),
        inference_tau=float(inference_cfg.get("tau", 1.0 / float(train_cfg.get("tau", 0.07)))),
        allow_root_unknown=allow_root_unknown,
        greedy_ablation=bool(inference_cfg.get("greedy_ablation", True)),
        checkpoint=str(checkpoint),
        last_checkpoint=str(last_checkpoint),
        result_path=str(resolve_experiment_artifact(
            train_cfg.get("result_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="results",
            default_filename=f"{experiment_name}-parent-unknown-global-path.result",
        )),
        diagnostics_path=str(resolve_experiment_artifact(
            train_cfg.get("diagnostics_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="diagnostics",
            default_filename=f"{experiment_name}-parent-unknown-diagnostics.json",
        )),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def validate_positive_checkpoint(args, checkpoint: dict) -> None:
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
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, config={expected_value!r}"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Idea 4 positive checkpoint/config mismatch: {details}")
    stage = str(checkpoint.get("stage", ""))
    if not stage.startswith("positive_joint_vision_lora"):
        raise ValueError(
            "Frozen-positive training requires a positive_joint_vision_lora "
            f"checkpoint family, got {checkpoint.get('stage')!r}"
        )
    for key in (
        "positive_state_dict",
        "vision_lora_config",
        "vision_lora_state_dict",
        "prompt_config",
    ):
        if not checkpoint.get(key):
            raise ValueError(f"Positive checkpoint is missing {key}")


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def prompt_parameters(learner: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in learner.named_parameters()
        if not name.startswith("text_encoder.") and parameter.requires_grad
    ]


def load_frozen_positive_stack(args, hierarchy, device, checkpoint: dict | None = None):
    checkpoint = checkpoint or load_idea3_checkpoint(args.positive_checkpoint, map_location="cpu")
    validate_positive_checkpoint(args, checkpoint)

    clip_model, tokenizer = load_clip_and_tokenizer(args, device)
    lora_cfg = VisionLoRAConfig.from_dict(checkpoint["vision_lora_config"])
    replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    load_vision_lora_state_dict(clip_model, checkpoint["vision_lora_state_dict"])
    freeze_module(clip_model)
    set_vision_lora_train_mode(clip_model, False)

    prompt_cfg = HierPromptConfig.from_dict(checkpoint["prompt_config"])
    text_encoder = SoftPromptTextEncoder(
        clip_model,
        tokenizer,
        max_length=prompt_cfg.max_length,
    )
    positive = PositivePromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    load_prompt_only_state_dict(positive, checkpoint["positive_state_dict"])
    freeze_module(positive)
    return checkpoint, clip_model, text_encoder, prompt_cfg, positive, replaced_modules


def dataset_payload(dataset, features: torch.Tensor | None = None) -> dict:
    payload = {
        "classes": list(dataset.classes),
        "targets": torch.tensor(dataset.targets, dtype=torch.long),
    }
    if features is not None:
        payload["features"] = features
    return payload


@torch.no_grad()
def encode_dataset_features(args, clip_model, dataset, loader, device, description: str) -> dict:
    clip_model.eval()
    set_vision_lora_train_mode(clip_model, False)
    feature_chunks = []
    target_chunks = []
    iterator = tqdm(loader, desc=description, leave=False) if tqdm else loader
    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            image_features = clip_model.get_image_features(pixel_values=images)
        feature_chunks.append(image_features.float().cpu())
        target_chunks.append(targets.long().cpu())
    return {
        "classes": list(dataset.classes),
        "targets": torch.cat(target_chunks),
        "features": torch.cat(feature_chunks),
    }


@torch.no_grad()
def encode_selected_train_images(args, clip_model, dataset, image_indices: list[int], device):
    unique_indices = list(dict.fromkeys(int(index) for index in image_indices))
    chunks = []
    for start in range(0, len(unique_indices), args.vision_batch_size):
        batch_indices = unique_indices[start:start + args.vision_batch_size]
        images = torch.stack([dataset[index][0] for index in batch_indices]).to(device)
        with autocast_context(args, device):
            features = clip_model.get_image_features(pixel_values=images)
        chunks.append(features.float())
    all_features = torch.cat(chunks, dim=0)
    return {
        image_index: all_features[position]
        for position, image_index in enumerate(unique_indices)
    }


@torch.no_grad()
def build_positive_semantic_index(hierarchy, positive):
    return build_idea3_semantic_index(
        hierarchy,
        positive,
        mode="positive_child_only",
    )


@torch.no_grad()
def build_unknown_semantic_index(hierarchy, positive_index, unknown):
    was_training = unknown.training
    unknown.eval()
    index = {}
    for parent, local in positive_index.items():
        unknown_feature = None
        candidate_names = list(local.children)
        prompts = dict(local.prompts)
        if parent != "root":
            unknown_feature = unknown.encode_unknown_prototypes([parent])[0].detach().cpu()
            unknown_name = f"__unknown__:{parent}"
            candidate_names.append(unknown_name)
            prompts[unknown_name] = [
                unknown.unknown_text(parent)
                for _ in range(unknown.num_unknown_prompts)
            ]
        index[parent] = LocalSemanticCandidates(
            parent=parent,
            children=list(local.children),
            child_features=local.child_features,
            unknown_feature=unknown_feature,
            candidate_names=candidate_names,
            prompts=prompts,
        )
    if was_training:
        unknown.train()
    return index


def weighted_unknown_ce(
    image_features: torch.Tensor,
    known_features: torch.Tensor,
    unknown_features: torch.Tensor,
    targets: torch.Tensor,
    tau: float,
    unknown_target_weight: float,
) -> tuple[torch.Tensor, dict]:
    logits = grouped_unknown_logits(
        image_features,
        known_features,
        unknown_features,
        logit_scale=1.0 / float(tau),
    )
    class_weights = torch.ones(logits.shape[1], dtype=logits.dtype, device=logits.device)
    class_weights[-1] = float(unknown_target_weight)
    targets = targets.long().to(logits.device)
    loss = F.cross_entropy(logits, targets, weight=class_weights)
    predictions = logits.argmax(dim=1)
    unknown_index = logits.shape[1] - 1
    known_mask = targets != unknown_index
    unknown_mask = ~known_mask
    return loss, {
        "acc": float((predictions == targets).float().mean().detach().cpu()),
        "known_acc": float((predictions[known_mask] == targets[known_mask]).float().mean().detach().cpu())
        if known_mask.any() else 0.0,
        "hidden_unknown_recall": float((predictions[unknown_mask] == unknown_index).float().mean().detach().cpu())
        if unknown_mask.any() else 0.0,
        "known_count": int(known_mask.sum().item()),
        "unknown_count": int(unknown_mask.sum().item()),
    }


def evaluate_feature_payload(
    args,
    hierarchy,
    payload: dict,
    semantic_index,
    split_name: str,
    mode: str,
) -> dict:
    features = payload["features"]
    prediction_chunks = []
    diagnostics = {
        "stop_depth_counts": {},
        "stop_node_counts": {},
        "candidate_type_counts": {},
    }
    unknown_selected = 0
    total = 0
    for start in range(0, int(features.shape[0]), args.inference_batch_size):
        batch = features[start:start + args.inference_batch_size]
        if mode == PRIMARY_INFERENCE_MODE:
            output = predict_features_terminal_global_path(
                batch,
                hierarchy,
                semantic_index,
                logit_scale=args.inference_tau,
                allow_root_unknown=args.allow_root_unknown,
                unknown_aggregation=getattr(args, "unknown_aggregation", "logmeanexp"),
            )
        elif mode == GREEDY_INFERENCE_MODE:
            output = predict_features_idea3(
                batch,
                hierarchy,
                semantic_index,
                mode="parent_unknown",
                tau=args.inference_tau,
                unknown_aggregation=getattr(args, "unknown_aggregation", "logmeanexp"),
            )
        elif mode == "positive_global_path":
            output = predict_features_idea3(
                batch,
                hierarchy,
                semantic_index,
                mode="positive_global_path",
                tau=args.inference_tau,
            )
        else:
            raise ValueError(f"Unsupported Idea 4 evaluation mode: {mode}")
        prediction_chunks.append(output["preds"].cpu())
        batch_diagnostics = output.get("diagnostics", {})
        batch_size = int(batch.shape[0])
        if mode == GREEDY_INFERENCE_MODE:
            predicted_nodes = [
                hierarchy.id_node_list[int(index)]
                for index in output["preds"].detach().cpu().tolist()
            ]
            batch_unknown_selected = sum(
                node in hierarchy.parent2children and node != "root"
                for node in predicted_nodes
            )
            batch_diagnostics = dict(batch_diagnostics)
            batch_diagnostics["candidate_type_counts"] = {
                "unknown": batch_unknown_selected,
                "leaf": batch_size - batch_unknown_selected,
            }
            batch_diagnostics["unknown_selection_rate"] = (
                batch_unknown_selected / max(1, batch_size)
            )
        for key in ("stop_depth_counts", "stop_node_counts", "candidate_type_counts"):
            for name, count in batch_diagnostics.get(key, {}).items():
                diagnostics[key][name] = diagnostics[key].get(name, 0) + int(count)
        unknown_selected += int(round(float(batch_diagnostics.get("unknown_selection_rate", 0.0)) * batch_size))
        total += batch_size

    preds = torch.cat(prediction_chunks)
    node_targets, metrics = evaluate_split(
        hierarchy,
        payload,
        preds,
        dists_mats=make_distance_mats(hierarchy),
    )
    diagnostics["unknown_selection_rate"] = unknown_selected / max(1, total)
    return {
        "preds": preds,
        "targets": node_targets.cpu(),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "split": split_name,
    }


@torch.no_grad()
def evaluate_pseudo_leave_child_out(
    args,
    hierarchy,
    payload: dict,
    grouped_examples,
    positive_index,
    unknown_index,
) -> dict:
    known_correct = 0
    known_total = 0
    hidden_correct = 0
    hidden_total = 0
    parent_metrics = {}

    for parent in sorted(grouped_examples):
        if parent == "root" or parent not in unknown_index:
            continue
        child_examples = grouped_examples[parent]
        children = [child for child in positive_index[parent].children if child in child_examples]
        if len(children) < 2:
            continue
        child_to_feature = {
            child: positive_index[parent].child_features[index]
            for index, child in enumerate(positive_index[parent].children)
        }
        unknown_feature = unknown_index[parent].unknown_feature
        parent_known_correct = 0
        parent_known_total = 0
        parent_hidden_correct = 0
        parent_hidden_total = 0

        for hidden_child in children:
            known_children = [child for child in children if child != hidden_child]
            known_features = torch.stack(
                [child_to_feature[child] for child in known_children]
            ).float()
            selected_examples = []
            labels = []
            child_to_index = {child: index for index, child in enumerate(known_children)}
            for child in children:
                examples = child_examples[child]
                if args.pseudo_max_examples_per_child > 0:
                    examples = examples[:args.pseudo_max_examples_per_child]
                selected_examples.extend(examples)
                target = len(known_children) if child == hidden_child else child_to_index[child]
                labels.extend([target] * len(examples))
            if not selected_examples:
                continue

            image_indices = torch.tensor([example.image_index for example in selected_examples], dtype=torch.long)
            image_features = payload["features"].index_select(0, image_indices)
            logits = grouped_unknown_logits(
                image_features,
                known_features,
                unknown_feature,
                logit_scale=1.0,
            )
            predictions = logits.argmax(dim=1)
            targets = torch.tensor(labels, dtype=torch.long)
            hidden_mask = targets == len(known_children)
            known_mask = ~hidden_mask
            parent_known_correct += int((predictions[known_mask] == targets[known_mask]).sum())
            parent_known_total += int(known_mask.sum())
            parent_hidden_correct += int((predictions[hidden_mask] == targets[hidden_mask]).sum())
            parent_hidden_total += int(hidden_mask.sum())

        known_correct += parent_known_correct
        known_total += parent_known_total
        hidden_correct += parent_hidden_correct
        hidden_total += parent_hidden_total
        parent_known_acc = parent_known_correct / max(1, parent_known_total)
        parent_hidden_recall = parent_hidden_correct / max(1, parent_hidden_total)
        parent_metrics[parent] = {
            "known_acc": parent_known_acc,
            "hidden_unknown_recall": parent_hidden_recall,
            "balanced_acc": 0.5 * (parent_known_acc + parent_hidden_recall),
        }

    known_acc = known_correct / max(1, known_total)
    hidden_recall = hidden_correct / max(1, hidden_total)
    harmonic_mean = (
        2.0 * known_acc * hidden_recall / (known_acc + hidden_recall)
        if known_acc + hidden_recall > 0.0 else 0.0
    )
    return {
        "known_acc": known_acc,
        "hidden_unknown_recall": hidden_recall,
        "balanced_acc": 0.5 * (known_acc + hidden_recall),
        "harmonic_mean": harmonic_mean,
        "known_count": known_total,
        "hidden_count": hidden_total,
        "by_parent": parent_metrics,
    }


def clone_prompt_state(learner) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in prompt_only_state_dict(learner).items()
    }


def save_unknown_checkpoint(
    args,
    path,
    positive_checkpoint,
    prompt_cfg,
    unknown,
    metrics,
):
    return save_idea3_checkpoint(
        path,
        stage="idea4_parent_unknown_frozen_positive_lora",
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


def scalar_metric_summary(result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "balanced_acc": float(metrics["balanced_acc"]),
        "balanced_hdist": float(metrics["balanced_hdist"]),
        "unknown_selection_rate": float(result.get("diagnostics", {}).get("unknown_selection_rate", 0.0)),
    }


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
    unknown = UnknownPromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    trainable_params = prompt_parameters(unknown)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    train_payload = dataset_payload(train_dataset)
    val_payload = encode_dataset_features(args, clip_model, val_dataset, val_loader, device, "encode ID val")
    train_groups = group_examples_by_parent_child(build_positive_edge_examples(hierarchy, train_payload))
    val_groups = group_examples_by_parent_child(build_positive_edge_examples(hierarchy, val_payload))
    parents = sorted(
        parent
        for parent, child_map in train_groups.items()
        if parent != "root" and len(child_map) >= 2
    )
    if not parents:
        raise RuntimeError("Idea 4 found no eligible non-root parents for leave-child-out training")

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
        "Idea 4 frozen-positive unknown training: "
        f"vision_lora_modules={len(replaced_modules)}, "
        f"unknown_prompt_params={sum(parameter.numel() for parameter in trainable_params)}, "
        f"unknown_prototypes={args.num_unknown_prompts}, "
        f"hide_strategy={args.hide_strategy}, "
        f"eligible_parents={len(parents)}, positive_val_bacc={positive_baseline_bacc:.6f}, "
        f"id_bacc_floor={id_bacc_floor:.6f}"
    )

    rng = random.Random(args.seed)
    history = []
    best_guarded_score = float("-inf")
    best_guarded_state = None
    best_guarded_epoch = None
    best_guarded_validation = None
    fallback_key = (float("-inf"), float("-inf"))
    fallback_state = None
    fallback_epoch = None
    fallback_validation = None

    for epoch in range(1, args.epochs + 1):
        unknown.train()
        clip_model.eval()
        positive.eval()
        set_vision_lora_train_mode(clip_model, False)
        shuffled_parents = list(parents)
        rng.shuffle(shuffled_parents)
        parent_chunks = [
            shuffled_parents[start:start + args.parents_per_step]
            for start in range(0, len(shuffled_parents), args.parents_per_step)
        ]
        iterator = tqdm(parent_chunks, desc=f"idea4 epoch {epoch}/{args.epochs}", leave=False) if tqdm else parent_chunks
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_reg = 0.0
        epoch_diversity = 0.0
        epoch_known_acc = 0.0
        epoch_hidden_recall = 0.0
        steps = 0

        for parent_chunk in iterator:
            episodes = []
            for parent in parent_chunk:
                episode = sample_leave_child_out_episode(
                    parent,
                    train_groups[parent],
                    strategy=args.hide_strategy,
                    max_examples=args.max_examples_per_parent,
                    rng=rng,
                )
                if episode is not None:
                    episodes.append(episode)
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

            optimizer.zero_grad(set_to_none=True)
            unknown_features = unknown.encode_unknown_prototypes(
                [episode.parent for episode in episodes]
            )
            losses = []
            ce_values = []
            reg_values = []
            diversity_values = []
            known_acc_values = []
            hidden_recall_values = []

            for episode_index, episode in enumerate(episodes):
                local = positive_index[episode.parent]
                child_to_position = {child: index for index, child in enumerate(local.children)}
                known_features = torch.stack([
                    local.child_features[child_to_position[child]]
                    for child in episode.known_children
                ]).to(device)
                all_child_features = local.child_features.to(device)
                unknown_feature = unknown_features[episode_index]
                child_to_target = {child: index for index, child in enumerate(episode.known_children)}
                targets = torch.tensor([
                    len(episode.known_children) if label == UNKNOWN_LABEL else child_to_target[label]
                    for label in episode.labels
                ], dtype=torch.long, device=device)
                image_features = torch.stack([
                    feature_by_index[example.image_index]
                    for example in episode.examples
                ])

                ce_loss, ce_stats = weighted_unknown_ce(
                    image_features,
                    known_features,
                    unknown_feature,
                    targets,
                    tau=args.tau,
                    unknown_target_weight=args.unknown_target_weight,
                )
                regularizer, reg_stats = unknown_regularization(
                    unknown_feature,
                    unknown._parent_feature(episode.parent),
                    all_child_features,
                    lambda_anchor=args.lambda_anchor,
                    lambda_child_sep=args.lambda_child_sep,
                    lambda_prototype_diversity=args.lambda_prototype_diversity,
                    prototype_diversity_margin=args.prototype_diversity_margin,
                )
                losses.append(ce_loss + regularizer)
                ce_values.append(float(ce_loss.detach().cpu()))
                reg_values.append(reg_stats["regularizer"])
                diversity_values.append(reg_stats["prototype_diversity_loss"])
                known_acc_values.append(ce_stats["known_acc"])
                hidden_recall_values.append(ce_stats["hidden_unknown_recall"])

            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_ce += sum(ce_values) / len(ce_values)
            epoch_reg += sum(reg_values) / len(reg_values)
            epoch_diversity += sum(diversity_values) / len(diversity_values)
            epoch_known_acc += sum(known_acc_values) / len(known_acc_values)
            epoch_hidden_recall += sum(hidden_recall_values) / len(hidden_recall_values)
            steps += 1

        scheduler.step()
        epoch_stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "ce_loss": epoch_ce / max(1, steps),
            "regularizer": epoch_reg / max(1, steps),
            "prototype_diversity_loss": epoch_diversity / max(1, steps),
            "known_acc": epoch_known_acc / max(1, steps),
            "hidden_unknown_recall": epoch_hidden_recall / max(1, steps),
            "lr": optimizer.param_groups[0]["lr"],
            "steps": steps,
        }

        validation_due = (
            epoch >= args.validation_start_epoch
            and (epoch - args.validation_start_epoch) % args.validation_every_n_epochs == 0
        )
        is_best = False
        if validation_due:
            unknown_index = build_unknown_semantic_index(hierarchy, positive_index, unknown)
            pseudo = evaluate_pseudo_leave_child_out(
                args,
                hierarchy,
                val_payload,
                val_groups,
                positive_index,
                unknown_index,
            )
            val_result = evaluate_feature_payload(
                args,
                hierarchy,
                val_payload,
                unknown_index,
                "val",
                mode=PRIMARY_INFERENCE_MODE,
            )
            val_bacc = float(val_result["metrics"]["balanced_acc"])
            val_bmhd = float(val_result["metrics"]["balanced_hdist"])
            guarded = val_bacc >= id_bacc_floor
            validation_summary = {
                "epoch": epoch,
                "val_balanced_acc": val_bacc,
                "val_balanced_hdist": val_bmhd,
                "id_guard_passed": guarded,
                "unknown_selection_rate": float(val_result["diagnostics"]["unknown_selection_rate"]),
                "pseudo_known_acc": pseudo["known_acc"],
                "pseudo_hidden_unknown_recall": pseudo["hidden_unknown_recall"],
                "pseudo_balanced_acc": pseudo["balanced_acc"],
                "pseudo_harmonic_mean": pseudo["harmonic_mean"],
            }
            epoch_stats.update(validation_summary)

            fallback_candidate = (val_bacc, pseudo["harmonic_mean"])
            if fallback_candidate > fallback_key:
                fallback_key = fallback_candidate
                fallback_state = clone_prompt_state(unknown)
                fallback_epoch = epoch
                fallback_validation = validation_summary
            if guarded and pseudo["harmonic_mean"] > best_guarded_score:
                best_guarded_score = pseudo["harmonic_mean"]
                best_guarded_state = clone_prompt_state(unknown)
                best_guarded_epoch = epoch
                best_guarded_validation = validation_summary
                is_best = True

        history.append(epoch_stats)
        message = (
            f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, "
            f"known_acc={epoch_stats['known_acc']:.6f}, "
            f"hidden_recall={epoch_stats['hidden_unknown_recall']:.6f}"
        )
        if validation_due:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"pseudo_hmean={epoch_stats['pseudo_harmonic_mean']:.6f}, "
                f"id_guard={epoch_stats['id_guard_passed']}"
            )
            if is_best:
                message += " [best]"
        print(message)

        if is_best:
            save_unknown_checkpoint(
                args,
                args.checkpoint,
                positive_checkpoint,
                prompt_cfg,
                unknown,
                {
                    "train_history": history,
                    "positive_baseline": scalar_metric_summary(positive_baseline),
                    "best_validation": best_guarded_validation,
                },
            )

    save_unknown_checkpoint(
        args,
        args.last_checkpoint,
        positive_checkpoint,
        prompt_cfg,
        unknown,
        {
            "train_history": history,
            "positive_baseline": scalar_metric_summary(positive_baseline),
            "best_validation": best_guarded_validation,
        },
    )

    selection_used_guard = best_guarded_state is not None
    selected_state = best_guarded_state if selection_used_guard else fallback_state
    selected_epoch = best_guarded_epoch if selection_used_guard else fallback_epoch
    selected_validation = best_guarded_validation if selection_used_guard else fallback_validation
    if selected_state is None:
        selected_state = clone_prompt_state(unknown)
        selected_epoch = args.epochs
        selected_validation = None
    load_prompt_only_state_dict(unknown, selected_state)

    unknown_index = build_unknown_semantic_index(hierarchy, positive_index, unknown)
    val_result = evaluate_feature_payload(
        args, hierarchy, val_payload, unknown_index, "val", mode=PRIMARY_INFERENCE_MODE
    )
    ood_payload = encode_dataset_features(args, clip_model, ood_dataset, ood_loader, device, "encode OOD")
    ood_result = evaluate_feature_payload(
        args, hierarchy, ood_payload, unknown_index, "ood", mode=PRIMARY_INFERENCE_MODE
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])

    ablations = {}
    if args.greedy_ablation:
        greedy_val = evaluate_feature_payload(
            args, hierarchy, val_payload, unknown_index, "val", mode=GREEDY_INFERENCE_MODE
        )
        greedy_ood = evaluate_feature_payload(
            args, hierarchy, ood_payload, unknown_index, "ood", mode=GREEDY_INFERENCE_MODE
        )
        ablations[GREEDY_INFERENCE_MODE] = {
            "val": greedy_val,
            "ood": greedy_ood,
            "mixed": mixed_summary(greedy_val["metrics"], greedy_ood["metrics"]),
        }

    result = {
        "args": vars(args),
        "mode": PRIMARY_INFERENCE_MODE,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": "idea4_parent_unknown_frozen_positive_lora",
        "positive_checkpoint": args.positive_checkpoint,
        "selected_epoch": selected_epoch,
        "selection_used_id_guard": selection_used_guard,
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
        "ablations": ablations,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)

    final_metrics = {
        "train_history": history,
        "positive_baseline": scalar_metric_summary(positive_baseline),
        "selection": {
            "selected_epoch": selected_epoch,
            "used_id_guard": selection_used_guard,
            "id_bacc_floor": id_bacc_floor,
            "validation": selected_validation,
        },
        "final": {
            "val": scalar_metric_summary(val_result),
            "ood": scalar_metric_summary(ood_result),
            "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
            "mixed_balanced_hdist": float(mixed["mixed_balanced_hdist"]),
        },
    }
    if args.greedy_ablation:
        greedy_result = ablations[GREEDY_INFERENCE_MODE]
        final_metrics["greedy_ablation"] = {
            "val": scalar_metric_summary(greedy_result["val"]),
            "ood": scalar_metric_summary(greedy_result["ood"]),
            "mixed_balanced_acc": float(greedy_result["mixed"]["mixed_balanced_acc"]),
            "mixed_balanced_hdist": float(greedy_result["mixed"]["mixed_balanced_hdist"]),
        }

    save_unknown_checkpoint(
        args,
        args.checkpoint,
        positive_checkpoint,
        prompt_cfg,
        unknown,
        final_metrics,
    )
    save_json(args.diagnostics_path, final_metrics)

    print(f"selected epoch: {selected_epoch} (ID guard used: {selection_used_guard})")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved last checkpoint: {args.last_checkpoint}")
    print(f"saved result: {args.result_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"OOD BAcc: {float(ood_result['metrics']['balanced_acc']):.6f}")
    print(f"Mixed BAcc: {float(mixed['mixed_balanced_acc']):.6f}")
    print(f"Mixed BMHD: {float(mixed['mixed_balanced_hdist']):.6f}")


if __name__ == "__main__":
    main()
