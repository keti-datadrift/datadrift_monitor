from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.metric_terminal import (
    build_metric_terminal_specs,
    predict_features_metric_terminal,
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
    load_config,
    load_prompt_only_state_dict,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import encode_dataset_features, freeze_module


def comma_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--terminal-weights", type=comma_floats, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--bottleneck-temperatures",
        type=comma_floats,
        default=[0.02, 0.05, 0.1, 0.2, 0.5],
    )
    parser.add_argument("--out-result", default=None)
    parser.add_argument("--out-diagnostics", default=None)
    parsed = parser.parse_args()
    args = load_config(parsed.config)
    if parsed.gpu_id is not None:
        args.device = f"cuda:{parsed.gpu_id}"
    checkpoint = parsed.checkpoint or args.last_checkpoint
    experiment_dir = Path(args.output_root) / "experiments" / args.experiment_name
    result_path = Path(parsed.out_result) if parsed.out_result else (
        experiment_dir / "results" / f"{args.experiment_name}-metric-terminal-positive-only.result"
    )
    diagnostics_path = Path(parsed.out_diagnostics) if parsed.out_diagnostics else (
        experiment_dir / "diagnostics" / f"{args.experiment_name}-metric-terminal-positive-only.json"
    )
    return (
        args,
        checkpoint,
        parsed.terminal_weights,
        parsed.bottleneck_temperatures,
        result_path,
        diagnostics_path,
    )


def validate_checkpoint(args, checkpoint: dict) -> None:
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
    for key in (
        "positive_state_dict",
        "vision_lora_config",
        "vision_lora_state_dict",
        "prompt_config",
    ):
        if not checkpoint.get(key):
            raise ValueError(f"Positive checkpoint is missing {key}")


@torch.no_grad()
def encode_all_positive_edges(hierarchy, positive) -> dict[tuple[str, str], torch.Tensor]:
    pairs = [
        (parent, child)
        for parent in sorted(
            hierarchy.parent2children,
            key=lambda node: (len(hierarchy.node_ancestors.get(node, [])), node),
        )
        for child in hierarchy.parent2children[parent]
    ]
    features = positive.encode_edges(pairs).float().cpu()
    return {pair: features[index] for index, pair in enumerate(pairs)}


def metric_summary(metrics: dict) -> dict:
    return {
        "acc": float(metrics["acc"]),
        "balanced_acc": float(metrics["balanced_acc"]),
        "avg_hdist": float(metrics["avg_hdist"]),
        "balanced_hdist": float(metrics["balanced_hdist"]),
    }


def main():
    (
        args,
        checkpoint_path,
        terminal_weights,
        bottleneck_temperatures,
        result_path,
        diagnostics_path,
    ) = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = available_device(args.device)

    checkpoint = load_idea3_checkpoint(checkpoint_path, map_location="cpu")
    validate_checkpoint(args, checkpoint)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    val_dataset, _ = build_eval_datasets(args, hierarchy)
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
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
    load_prompt_only_state_dict(positive, checkpoint["positive_state_dict"])
    freeze_module(positive)

    val_payload = encode_dataset_features(
        args,
        clip_model,
        val_dataset,
        val_loader,
        device,
        "encode ID val",
    )
    edge_features = encode_all_positive_edges(hierarchy, positive)
    terminal_specs = build_metric_terminal_specs(hierarchy)
    distance_mats = make_distance_mats(hierarchy)

    grid = []
    full_outputs = {}
    best_key = None
    best_order = None
    for terminal_weight in terminal_weights:
        for bottleneck_temperature in bottleneck_temperatures:
            key = f"w={terminal_weight:g},tau={bottleneck_temperature:g}"
            output = predict_features_metric_terminal(
                val_payload["features"],
                hierarchy,
                edge_features,
                terminal_specs,
                terminal_weight=terminal_weight,
                bottleneck_temperature=bottleneck_temperature,
            )
            targets, metrics = evaluate_split(
                hierarchy,
                val_payload,
                output["preds"],
                dists_mats=distance_mats,
            )
            summary = metric_summary(metrics)
            row = {
                "key": key,
                "terminal_weight": float(terminal_weight),
                "bottleneck_temperature": float(bottleneck_temperature),
                **summary,
            }
            grid.append(row)
            full_outputs[key] = {
                "preds": output["preds"],
                "targets": targets.cpu(),
                "metrics": metrics,
                "diagnostics": output["diagnostics"],
            }
            order = (summary["balanced_acc"], -summary["balanced_hdist"])
            if best_order is None or order > best_order:
                best_key = key
                best_order = order

    assert best_key is not None
    selected = next(row for row in grid if row["key"] == best_key)
    result = {
        "method": "global_metric_terminal_decoder",
        "positive_only": True,
        "used_negative_prompts": False,
        "negative_ready_candidate_definition": (
            "known leaves plus enabled parent-unknown terminals in one global metric space"
        ),
        "selection": "maximum ID validation balanced accuracy; BMHD tie-break",
        "used_ood_for_training_or_selection": False,
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "checkpoint_stage": checkpoint.get("stage"),
        "device": device,
        "vision_lora_modules": len(replaced_modules),
        "num_positive_edges": len(edge_features),
        "num_leaf_terminals": len(terminal_specs),
        "grid": grid,
        "selected": selected,
        "outputs": full_outputs,
    }
    ensure_dir(result_path.parent)
    torch.save(result, result_path)
    save_json(
        diagnostics_path,
        {key: value for key, value in result.items() if key != "outputs"},
    )
    print(f"saved result: {result_path}")
    print(f"saved diagnostics: {diagnostics_path}")
    print(
        f"selected {best_key}: ID BAcc={selected['balanced_acc']:.6f}, "
        f"ID BMHD={selected['balanced_hdist']:.6f}"
    )


if __name__ == "__main__":
    main()
