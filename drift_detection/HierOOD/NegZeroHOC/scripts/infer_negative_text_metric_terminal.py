from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.metric_terminal import (
    build_metric_terminal_specs,
    metric_terminal_scores,
)
from negzerohoc.negative_metric_terminal import (
    threshold_terminal_winner_indices,
)
from negzerohoc.prompt_models import HierPromptConfig, UnknownPromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
)
from scripts.infer_metric_terminal_positive import encode_all_positive_edges
from scripts.train_idea3_joint_vision_lora import (
    build_eval_datasets,
    load_prompt_only_state_dict,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    load_frozen_positive_stack,
)
from scripts.train_negative_text_metric_terminal import (
    CHECKPOINT_STAGE,
    load_config,
    metric_summary,
    terminal_indices_to_node_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--out-result", default=None)
    parser.add_argument("--out-diagnostics", default=None)
    parsed = parser.parse_args()
    args = load_config(parsed.config)
    if parsed.gpu_id is not None:
        args.device = f"cuda:{parsed.gpu_id}"
    checkpoint_path = parsed.checkpoint or args.checkpoint
    experiment_dir = (
        Path(args.output_root) / "experiments" / args.experiment_name
    )
    result_path = (
        Path(parsed.out_result) if parsed.out_result
        else experiment_dir / "results" / "actual-ood.result"
    )
    diagnostics_path = (
        Path(parsed.out_diagnostics) if parsed.out_diagnostics
        else experiment_dir / "diagnostics" / "actual-ood.json"
    )
    return args, checkpoint_path, result_path, diagnostics_path


def split_output(
    hierarchy,
    payload,
    score_matrix,
    terminal_specs,
    unknown_threshold,
    distance_mats,
) -> dict:
    winner_indices = threshold_terminal_winner_indices(
        score_matrix,
        terminal_specs,
        unknown_threshold=unknown_threshold,
    )
    predictions = terminal_indices_to_node_predictions(
        hierarchy, terminal_specs, winner_indices
    )
    targets, metrics = evaluate_split(
        hierarchy,
        payload,
        predictions,
        dists_mats=distance_mats,
    )
    winner_specs = [
        terminal_specs[int(index)]
        for index in winner_indices.detach().cpu().tolist()
    ]
    kinds = Counter(spec.kind for spec in winner_specs)
    nodes = Counter(spec.node for spec in winner_specs)
    return {
        "preds": predictions,
        "targets": targets.cpu(),
        "metrics": metrics,
        "diagnostics": {
            "unknown_selection_rate": (
                kinds.get("unknown", 0) / max(1, len(winner_specs))
            ),
            "candidate_type_counts": dict(kinds),
            "stop_node_counts": dict(nodes.most_common()),
        },
    }


def main():
    args, checkpoint_path, result_path, diagnostics_path = parse_args()
    configure_reproducibility(
        args.seed, deterministic=args.deterministic
    )
    device = available_device(args.device)
    negative_checkpoint = load_idea3_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    if negative_checkpoint.get("stage") != CHECKPOINT_STAGE:
        raise ValueError(
            f"Expected checkpoint stage {CHECKPOINT_STAGE!r}, got "
            f"{negative_checkpoint.get('stage')!r}"
        )

    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    val_dataset, ood_dataset = build_eval_datasets(args, hierarchy)
    val_loader = make_loader(
        val_dataset,
        args.validation_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    ood_loader = make_loader(
        ood_dataset,
        args.validation_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    (
        _positive_checkpoint,
        clip_model,
        text_encoder,
        _positive_prompt_cfg,
        positive,
        replaced_modules,
    ) = load_frozen_positive_stack(args, hierarchy, device)
    negative_prompt_cfg = HierPromptConfig.from_dict(
        negative_checkpoint["prompt_config"]
    )
    unknown = UnknownPromptLearner(
        args.dataset, hierarchy, text_encoder, negative_prompt_cfg
    ).to(device)
    load_prompt_only_state_dict(
        unknown, negative_checkpoint["unknown_state_dict"]
    )
    unknown.eval()

    val_payload = encode_dataset_features(
        args, clip_model, val_dataset, val_loader, device, "encode ID val"
    )
    ood_payload = encode_dataset_features(
        args, clip_model, ood_dataset, ood_loader, device, "encode actual OOD"
    )
    edge_features = {
        edge: feature.to(device)
        for edge, feature in encode_all_positive_edges(
            hierarchy, positive
        ).items()
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
        hierarchy, parents, allow_root_unknown=False
    )
    with torch.no_grad():
        banks = unknown.encode_unknown_prototypes(parents)
        unknown_features = {
            parent: banks[index]
            for index, parent in enumerate(parents)
        }

    selected = negative_checkpoint["metrics"]["selected_decoder"]
    score_kwargs = {
        "unknown_features_by_parent": unknown_features,
        "terminal_weight": float(selected["terminal_weight"]),
        "bottleneck_temperature": float(
            selected["bottleneck_temperature"]
        ),
        "unknown_temperature": args.unknown_temperature,
    }
    val_scores = metric_terminal_scores(
        val_payload["features"],
        edge_features,
        terminal_specs,
        **score_kwargs,
    )["score_matrix"]
    ood_scores = metric_terminal_scores(
        ood_payload["features"],
        edge_features,
        terminal_specs,
        **score_kwargs,
    )["score_matrix"]
    distance_mats = make_distance_mats(hierarchy)
    val_output = split_output(
        hierarchy,
        val_payload,
        val_scores,
        terminal_specs,
        float(selected["unknown_threshold"]),
        distance_mats,
    )
    ood_output = split_output(
        hierarchy,
        ood_payload,
        ood_scores,
        terminal_specs,
        float(selected["unknown_threshold"]),
        distance_mats,
    )
    mixed = mixed_summary(
        val_output["metrics"], ood_output["metrics"]
    )
    summary = {
        "method": "negative_text_global_metric_terminal",
        "config": args.config,
        "checkpoint": str(checkpoint_path),
        "positive_checkpoint": args.positive_checkpoint,
        "negative_text_variant": (
            negative_prompt_cfg.unknown_text_variant
        ),
        "unknown_prompt_count": (
            negative_prompt_cfg.unknown_prompts
        ),
        "selected_epoch": negative_checkpoint["metrics"][
            "selected_epoch"
        ],
        "selected_decoder": selected,
        "selection_used_actual_ood": False,
        "actual_ood_evaluated_only_after_selection": True,
        "vision_lora_modules": len(replaced_modules),
        "val": metric_summary(val_output["metrics"]),
        "val_unknown_selection_rate": val_output["diagnostics"][
            "unknown_selection_rate"
        ],
        "ood": metric_summary(ood_output["metrics"]),
        "ood_unknown_selection_rate": ood_output["diagnostics"][
            "unknown_selection_rate"
        ],
        "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
        "mixed_balanced_hdist": float(
            mixed["mixed_balanced_hdist"]
        ),
    }
    result = {
        **summary,
        "val_output": val_output,
        "ood_output": ood_output,
    }
    ensure_dir(result_path.parent)
    torch.save(result, result_path)
    save_json(diagnostics_path, summary)
    print(f"saved result: {result_path}")
    print(f"saved diagnostics: {diagnostics_path}")
    print(
        f"ID BAcc={summary['val']['balanced_acc']:.6f}, "
        f"OOD BAcc={summary['ood']['balanced_acc']:.6f}, "
        f"Mixed BAcc={summary['mixed_balanced_acc']:.6f}, "
        f"Mixed BMHD={summary['mixed_balanced_hdist']:.6f}"
    )


if __name__ == "__main__":
    main()
