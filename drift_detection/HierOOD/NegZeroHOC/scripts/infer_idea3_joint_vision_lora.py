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
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.runtime import available_device
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
)
from scripts.train_idea3_joint_vision_lora import (
    build_eval_datasets,
    evaluate_split_raw,
    load_clip_and_tokenizer,
    load_config,
    load_prompt_only_state_dict,
    make_loader,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def validate_checkpoint(args, checkpoint: dict) -> None:
    expected = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={actual!r}, config={expected_value!r}"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Joint Vision LoRA checkpoint/config mismatch: {details}")
    if checkpoint.get("stage") != "positive_joint_vision_lora":
        raise ValueError(
            "Expected a positive_joint_vision_lora checkpoint, "
            f"got {checkpoint.get('stage')!r}"
        )
    if not checkpoint.get("positive_state_dict"):
        raise ValueError("Joint checkpoint does not contain learned positive prompts")
    if not checkpoint.get("vision_lora_state_dict"):
        raise ValueError("Joint checkpoint does not contain CLIP Vision LoRA weights")
    if not checkpoint.get("prompt_config"):
        raise ValueError("Joint checkpoint does not contain the prompt configuration")
    if not checkpoint.get("vision_lora_config"):
        raise ValueError("Joint checkpoint does not contain the Vision LoRA configuration")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    checkpoint = load_idea3_checkpoint(args.checkpoint, map_location="cpu")
    validate_checkpoint(args, checkpoint)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    val_dataset, ood_dataset = build_eval_datasets(args, hierarchy)
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    ood_loader = make_loader(
        ood_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    clip_model, tokenizer = load_clip_and_tokenizer(args, device)
    lora_cfg = VisionLoRAConfig.from_dict(checkpoint.get("vision_lora_config"))
    replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    load_vision_lora_state_dict(clip_model, checkpoint["vision_lora_state_dict"])

    checkpoint_prompt_cfg = checkpoint["prompt_config"]
    text_encoder = SoftPromptTextEncoder(
        clip_model,
        tokenizer,
        max_length=int(checkpoint_prompt_cfg.get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(checkpoint_prompt_cfg)
    learner = PositivePromptLearner(
        args.dataset,
        hierarchy,
        text_encoder,
        prompt_cfg,
    ).to(device)
    load_prompt_only_state_dict(learner, checkpoint["positive_state_dict"])

    val_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        val_dataset,
        val_loader,
        device,
        "val",
    )
    ood_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        ood_dataset,
        ood_loader,
        device,
        "ood",
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])
    result = {
        "args": vars(args),
        "mode": args.inference_mode,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint["stage"],
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
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
