from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.evaluation import build_hierarchy, mixed_summary
from negzerohoc.feature_io import ensure_dir
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner, UnknownPromptLearner
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
    GREEDY_INFERENCE_MODE,
    PRIMARY_INFERENCE_MODE,
    build_positive_semantic_index,
    build_unknown_semantic_index,
    encode_dataset_features,
    evaluate_feature_payload,
    freeze_module,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def validate_unknown_checkpoint(args, checkpoint: dict) -> None:
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
        raise ValueError(f"Idea 4 checkpoint/config mismatch: {details}")
    if checkpoint.get("stage") != "idea4_parent_unknown_frozen_positive_lora":
        raise ValueError(
            "Expected an idea4_parent_unknown_frozen_positive_lora checkpoint, "
            f"got {checkpoint.get('stage')!r}"
        )
    for key in (
        "positive_state_dict",
        "unknown_state_dict",
        "vision_lora_config",
        "vision_lora_state_dict",
        "prompt_config",
    ):
        if not checkpoint.get(key):
            raise ValueError(f"Idea 4 checkpoint is missing {key}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    checkpoint = load_idea3_checkpoint(args.checkpoint, map_location="cpu")
    validate_unknown_checkpoint(args, checkpoint)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    val_dataset, ood_dataset = build_eval_datasets(args, hierarchy)
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
    unknown = UnknownPromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    load_prompt_only_state_dict(positive, checkpoint["positive_state_dict"])
    load_prompt_only_state_dict(unknown, checkpoint["unknown_state_dict"])
    freeze_module(positive)
    freeze_module(unknown)

    positive_index = build_positive_semantic_index(hierarchy, positive)
    unknown_index = build_unknown_semantic_index(hierarchy, positive_index, unknown)
    val_payload = encode_dataset_features(args, clip_model, val_dataset, val_loader, device, "encode ID val")
    ood_payload = encode_dataset_features(args, clip_model, ood_dataset, ood_loader, device, "encode OOD")

    val_result = evaluate_feature_payload(
        args, hierarchy, val_payload, unknown_index, "val", mode=PRIMARY_INFERENCE_MODE
    )
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

    checkpoint_metrics = checkpoint.get("metrics", {})
    selection = checkpoint_metrics.get("selection", {})
    result = {
        "args": vars(args),
        "mode": PRIMARY_INFERENCE_MODE,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint["stage"],
        "positive_checkpoint": checkpoint.get("positive_checkpoint"),
        "selected_epoch": selection.get("selected_epoch"),
        "selection_used_id_guard": selection.get("used_id_guard"),
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
        "ablations": ablations,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)

    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"reconstructed Vision LoRA modules: {len(replaced_modules)}")
    print(f"saved result: {args.result_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"ID BMHD: {float(val_result['metrics']['balanced_hdist']):.6f}")
    print(f"OOD BAcc: {float(ood_result['metrics']['balanced_acc']):.6f}")
    print(f"OOD BMHD: {float(ood_result['metrics']['balanced_hdist']):.6f}")
    print(f"Mixed BAcc: {float(mixed['mixed_balanced_acc']):.6f}")
    print(f"Mixed BMHD: {float(mixed['mixed_balanced_hdist']):.6f}")


if __name__ == "__main__":
    main()
