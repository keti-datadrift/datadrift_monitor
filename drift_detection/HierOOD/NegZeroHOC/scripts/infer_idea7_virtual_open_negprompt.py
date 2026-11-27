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
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.oracle_parent import oracle_parent_diagnostics
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
)
from scripts.train_idea7_virtual_open_negprompt import CHECKPOINT_STAGE, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def validate_checkpoint(args, checkpoint: dict) -> None:
    expected = {
        "stage": CHECKPOINT_STAGE,
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
        raise ValueError(f"Idea7 checkpoint/config mismatch: {details}")
    for key in (
        "positive_state_dict",
        "unknown_state_dict",
        "vision_lora_config",
        "vision_lora_state_dict",
        "prompt_config",
    ):
        if not checkpoint.get(key):
            raise ValueError(f"Idea7 checkpoint is missing {key}")


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
        clip_model, tokenizer, max_length=prompt_cfg.max_length
    )
    positive = PositivePromptLearner(
        args.dataset, hierarchy, text_encoder, prompt_cfg
    ).to(device)
    unknown = UnknownPromptLearner(
        args.dataset, hierarchy, text_encoder, prompt_cfg
    ).to(device)
    load_prompt_only_state_dict(positive, checkpoint["positive_state_dict"])
    load_prompt_only_state_dict(unknown, checkpoint["unknown_state_dict"])
    freeze_module(positive)
    freeze_module(unknown)

    positive_index = build_positive_semantic_index(hierarchy, positive)
    semantic_index = build_unknown_semantic_index(
        hierarchy, positive_index, unknown
    )
    val_payload = encode_dataset_features(
        args, clip_model, val_dataset, val_loader, device, "encode ID val"
    )
    ood_payload = encode_dataset_features(
        args, clip_model, ood_dataset, ood_loader, device, "encode OOD"
    )
    val_result = evaluate_feature_payload(
        args, hierarchy, val_payload, semantic_index, "val", mode=PRIMARY_INFERENCE_MODE
    )
    ood_result = evaluate_feature_payload(
        args, hierarchy, ood_payload, semantic_index, "ood", mode=PRIMARY_INFERENCE_MODE
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

    selection = checkpoint.get("metrics", {}).get("selection", {})
    result = {
        "args": vars(args),
        "mode": PRIMARY_INFERENCE_MODE,
        "method": "virtual_open_negprompt",
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint["stage"],
        "positive_checkpoint": checkpoint.get("positive_checkpoint"),
        "selected_epoch": selection.get("selected_epoch"),
        "selection_policy": selection.get("policy"),
        "selection_used_ood": selection.get("used_ood_for_selection"),
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "oracle_parent": oracle_result,
        "mixed": mixed,
        "ablations": ablations,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)
    save_json(args.oracle_diagnostics_path, {
        "diagnostic_only": True,
        "used_for_checkpoint_selection": False,
        "checkpoint": args.checkpoint,
        "oracle_parent": oracle_result,
    })

    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"reconstructed Vision LoRA modules: {len(replaced_modules)}")
    print(f"saved result: {args.result_path}")
    print(f"saved oracle-parent diagnostics: {args.oracle_diagnostics_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"ID BMHD: {float(val_result['metrics']['balanced_hdist']):.6f}")
    print(f"OOD BAcc: {float(ood_result['metrics']['balanced_acc']):.6f}")
    print(f"OOD BMHD: {float(ood_result['metrics']['balanced_hdist']):.6f}")
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
