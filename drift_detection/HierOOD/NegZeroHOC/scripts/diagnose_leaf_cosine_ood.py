from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.ood_diagnostics import (
    balanced_label_accuracy,
    binary_ood_metrics,
    macro_class_ood_metrics,
    max_cosine_scores,
    score_summary,
)
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.runtime import available_device
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_train_mode,
)
from scripts.train_idea3_joint_vision_lora import (
    build_eval_datasets,
    load_clip_and_tokenizer,
    load_prompt_only_state_dict,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    freeze_module,
)
from scripts.train_idea7_virtual_open_negprompt import load_config


OUTPUT_ROOT = Path(
    "outputs/experiments/ablate-positive-leaf-cosine-fgvc-aircraft-b16-r16-qkvo/"
    "diagnostics"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--vision", choices=["base", "lora"], default="lora")
    parser.add_argument("--text", choices=["plain", "learned"], default="learned")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--positive-checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parsed = parser.parse_args()
    args = load_config(parsed.config)
    if parsed.gpu_id is not None:
        args.device = f"cuda:{parsed.gpu_id}"
    if parsed.positive_checkpoint is not None:
        args.positive_checkpoint = parsed.positive_checkpoint
    output_path = (
        Path(parsed.out)
        if parsed.out
        else OUTPUT_ROOT / f"positive-leaf-cosine-{parsed.vision}-{parsed.text}.json"
    )
    return args, output_path, parsed.vision, parsed.text


def node_targets(hierarchy, payload: dict) -> list[str]:
    mapping = hierarchy.gen_ds2node_map(payload["classes"])
    indices = mapping[payload["targets"].long().cpu()]
    return [hierarchy.id_node_list[int(index)] for index in indices.tolist()]


@torch.no_grad()
def encode_edge_features(positive, pairs: list[tuple[str, str]], text_variant: str) -> torch.Tensor:
    if text_variant == "learned":
        return positive.encode_edges(pairs).detach().cpu()
    if text_variant == "plain":
        texts = [positive.edge_text(parent, child) for parent, child in pairs]
        return positive.text_encoder.encode_plain_texts(texts).detach().cpu()
    raise ValueError(f"Unsupported text variant: {text_variant!r}")


@torch.no_grad()
def encode_leaf_features(
    hierarchy,
    positive,
    text_variant: str,
) -> tuple[list[str], torch.Tensor]:
    leaves = list(hierarchy.train_classes)
    pairs = [(hierarchy.child2parent[leaf], leaf) for leaf in leaves]
    return leaves, encode_edge_features(positive, pairs, text_variant)


def load_ablation_stack(
    args,
    hierarchy,
    device: str,
    vision_variant: str,
    text_variant: str,
):
    checkpoint = load_idea3_checkpoint(args.positive_checkpoint, map_location="cpu")
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
        raise ValueError(f"Positive checkpoint/config mismatch: {mismatches}")
    if text_variant == "learned" and (
        not checkpoint.get("prompt_config") or not checkpoint.get("positive_state_dict")
    ):
        raise ValueError(
            "Learned-text diagnostic requires prompt_config and positive_state_dict"
        )

    clip_model, tokenizer = load_clip_and_tokenizer(args, device)
    replaced_modules = []
    if vision_variant == "lora":
        if not checkpoint.get("vision_lora_config") or not checkpoint.get("vision_lora_state_dict"):
            raise ValueError("LoRA diagnostic requires Vision LoRA config and state")
        lora_cfg = VisionLoRAConfig.from_dict(checkpoint["vision_lora_config"])
        replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
        load_vision_lora_state_dict(clip_model, checkpoint["vision_lora_state_dict"])
    freeze_module(clip_model)
    set_vision_lora_train_mode(clip_model, False)
    prompt_cfg = HierPromptConfig.from_dict(checkpoint.get("prompt_config"))
    text_encoder = SoftPromptTextEncoder(
        clip_model, tokenizer, max_length=prompt_cfg.max_length
    )
    positive = PositivePromptLearner(
        args.dataset, hierarchy, text_encoder, prompt_cfg
    ).to(device)
    if checkpoint.get("positive_state_dict"):
        load_prompt_only_state_dict(positive, checkpoint["positive_state_dict"])
    freeze_module(positive)
    return checkpoint, clip_model, text_encoder, prompt_cfg, positive, len(replaced_modules)


def label_names(payload: dict) -> list[str]:
    return [payload["classes"][int(index)] for index in payload["targets"].tolist()]


@torch.no_grad()
def parent_local_leaf_diagnostics(
    hierarchy,
    positive,
    id_payload: dict,
    ood_payload: dict,
    text_variant: str,
) -> dict:
    id_nodes = node_targets(hierarchy, id_payload)
    ood_rejection_nodes = node_targets(hierarchy, ood_payload)
    leaf_parents = [
        parent
        for parent, children in hierarchy.parent2children.items()
        if parent != "root"
        and children
        and all(child not in hierarchy.parent2children for child in children)
    ]

    by_parent = {}
    for parent in sorted(leaf_parents):
        id_indices = [
            index
            for index, node in enumerate(id_nodes)
            if hierarchy.child2parent.get(node) == parent
        ]
        ood_indices = [
            index
            for index, node in enumerate(ood_rejection_nodes)
            if node == parent
        ]
        if not id_indices or not ood_indices:
            continue

        children = list(hierarchy.parent2children[parent])
        child_features = encode_edge_features(
            positive,
            [(parent, child) for child in children],
            text_variant,
        )
        id_tensor = torch.tensor(id_indices, dtype=torch.long)
        ood_tensor = torch.tensor(ood_indices, dtype=torch.long)
        id_max, id_predictions = max_cosine_scores(
            id_payload["features"].index_select(0, id_tensor), child_features
        )
        ood_max, _ = max_cosine_scores(
            ood_payload["features"].index_select(0, ood_tensor), child_features
        )
        target_indices = [children.index(id_nodes[index]) for index in id_indices]
        metrics = binary_ood_metrics(-id_max.numpy(), -ood_max.numpy())
        metrics.update({
            "id_samples": len(id_indices),
            "ood_samples": len(ood_indices),
            "children": children,
            "id_local_leaf_balanced_acc": balanced_label_accuracy(
                target_indices, id_predictions.tolist()
            ),
            "id_max_cosine": score_summary(id_max.numpy()),
            "ood_max_cosine": score_summary(ood_max.numpy()),
        })
        by_parent[parent] = metrics

    macro_keys = (
        "auroc",
        "aupr_out",
        "aupr_in",
        "fpr95",
        "best_balanced_acc_diagnostic_only",
        "id_local_leaf_balanced_acc",
    )
    macro = {
        key: float(np.mean([metrics[key] for metrics in by_parent.values()]))
        for key in macro_keys
    } if by_parent else {}
    return {
        "definition": "oracle rejection parents whose retained ID children are terminal leaves",
        "num_supported_parents": len(by_parent),
        "supported_id_samples": sum(item["id_samples"] for item in by_parent.values()),
        "supported_ood_samples": sum(item["ood_samples"] for item in by_parent.values()),
        "macro": macro,
        "by_parent": by_parent,
    }


def main():
    args, output_path, vision_variant, text_variant = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    val_dataset, ood_dataset = build_eval_datasets(args, hierarchy)
    val_loader = make_loader(
        val_dataset, args.inference_batch_size, args.num_workers, False, args.seed
    )
    ood_loader = make_loader(
        ood_dataset, args.inference_batch_size, args.num_workers, False, args.seed
    )
    checkpoint, clip_model, _, _, positive, vision_lora_modules = load_ablation_stack(
        args, hierarchy, device, vision_variant, text_variant
    )
    val_payload = encode_dataset_features(
        args, clip_model, val_dataset, val_loader, device, "encode ID val"
    )
    ood_payload = encode_dataset_features(
        args, clip_model, ood_dataset, ood_loader, device, "encode OOD"
    )

    leaf_nodes, leaf_features = encode_leaf_features(
        hierarchy, positive, text_variant
    )
    id_max, id_predictions = max_cosine_scores(val_payload["features"], leaf_features)
    ood_max, _ = max_cosine_scores(ood_payload["features"], leaf_features)
    id_nodes = node_targets(hierarchy, val_payload)
    leaf_to_index = {leaf: index for index, leaf in enumerate(leaf_nodes)}
    id_target_indices = [leaf_to_index[node] for node in id_nodes]
    ood_class_names = label_names(ood_payload)

    global_metrics = binary_ood_metrics(-id_max.numpy(), -ood_max.numpy())
    global_metrics.update({
        "definition": "negative maximum cosine to all learned ID leaf positive text features",
        "num_leaf_prototypes": len(leaf_nodes),
        "id_leaf_balanced_acc": balanced_label_accuracy(
            id_target_indices, id_predictions.tolist()
        ),
        "id_max_cosine": score_summary(id_max.numpy()),
        "ood_max_cosine": score_summary(ood_max.numpy()),
        "class_balanced": macro_class_ood_metrics(
            -id_max.numpy(), -ood_max.numpy(), ood_class_names
        ),
    })
    local_metrics = parent_local_leaf_diagnostics(
        hierarchy, positive, val_payload, ood_payload, text_variant
    )

    result = {
        "method": "positive_leaf_max_cosine_ood_diagnostic",
        "diagnostic_only": True,
        "used_negative_prompts": False,
        "used_ood_for_training_or_selection": False,
        "vision_variant": vision_variant,
        "text_variant": text_variant,
        "plain_text_definition": (
            "the same path-aware edge sentence as learned prompts, encoded without soft context"
        ),
        "config": args.config,
        "positive_checkpoint": args.positive_checkpoint,
        "positive_checkpoint_stage": checkpoint["stage"],
        "device": device,
        "vision_lora_modules": vision_lora_modules,
        "global_leaf": global_metrics,
        "oracle_parent_local_leaf": local_metrics,
    }
    ensure_dir(output_path.parent)
    save_json(output_path, result)
    print(f"saved: {output_path}")
    print(
        f"{vision_variant}+{text_variant} global leaf max-cosine OOD: "
        f"AUROC={global_metrics['auroc']:.6f}, "
        f"AUPR-Out={global_metrics['aupr_out']:.6f}, "
        f"FPR95={global_metrics['fpr95']:.6f}, "
        f"ID leaf BAcc={global_metrics['id_leaf_balanced_acc']:.6f}"
    )
    if local_metrics["macro"]:
        print(
            "Oracle parent-local leaf OOD macro: "
            f"parents={local_metrics['num_supported_parents']}, "
            f"AUROC={local_metrics['macro']['auroc']:.6f}, "
            f"FPR95={local_metrics['macro']['fpr95']:.6f}"
        )


if __name__ == "__main__":
    main()
