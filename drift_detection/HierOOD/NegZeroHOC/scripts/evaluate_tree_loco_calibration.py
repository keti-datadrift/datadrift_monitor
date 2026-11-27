from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.metric_terminal import build_metric_terminal_specs
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.runtime import available_device, configure_reproducibility
from negzerohoc.tree_loco import (
    ParentSpecificVirtualSiblingPromptLearner,
)
from negzerohoc.tree_virtual_unknown import (
    tree_complement_terminal_scores,
)
from scripts.train_idea3_joint_vision_lora import (
    build_datasets,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import (
    encode_dataset_features,
    load_frozen_positive_stack,
)
from scripts.train_paper_negprompt_ablation import json_ready
from scripts.train_tree_loco_multiprompt import load_config
from scripts.train_tree_virtual_unknown import (
    positive_feature_layout,
    score_kwargs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--id-acceptances",
        nargs="+",
        type=float,
        default=[0.99, 0.975, 0.95, 0.90, 0.80],
    )
    return parser.parse_args()


@torch.no_grad()
def score_payload(
    payload,
    args,
    hierarchy,
    edge_features,
    terminal_specs,
    unknown_features_by_parent,
):
    chunks = []
    for start in range(
        0, len(payload["features"]), args.inference_batch_size
    ):
        output = tree_complement_terminal_scores(
            payload["features"][
                start:start + args.inference_batch_size
            ],
            hierarchy,
            edge_features,
            terminal_specs,
            unknown_features_by_parent,
            **score_kwargs(args),
        )
        chunks.append(output["score_matrix"].cpu())
    return torch.cat(chunks)


def decode_at_threshold(
    hierarchy,
    payload,
    score_matrix,
    terminal_specs,
    threshold,
):
    leaf_indices = torch.tensor([
        index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is None
    ])
    unknown_indices = torch.tensor([
        index
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is not None
    ])
    leaf_scores, leaf_local = score_matrix[:, leaf_indices].max(dim=1)
    unknown_scores, unknown_local = score_matrix[
        :, unknown_indices
    ].max(dim=1)
    leaf_winners = leaf_indices[leaf_local]
    unknown_winners = unknown_indices[unknown_local]
    reject = unknown_scores - leaf_scores >= float(threshold)
    winners = torch.where(reject, unknown_winners, leaf_winners)
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    preds = torch.tensor([
        node_to_index[terminal_specs[int(index)].node]
        for index in winners.tolist()
    ])
    _, metrics = evaluate_split(
        hierarchy,
        payload,
        preds,
        dists_mats=make_distance_mats(hierarchy),
    )
    return {
        "preds": preds,
        "metrics": metrics,
        "unknown_selection_rate": float(reject.float().mean()),
        "ood_score": unknown_scores - leaf_scores,
    }


def prototype_usage(
    features,
    score_matrix,
    terminal_specs,
    unknown_features_by_parent,
):
    unknown_specs = [
        (index, spec)
        for index, spec in enumerate(terminal_specs)
        if spec.unknown_parent is not None
    ]
    columns = torch.tensor([index for index, _ in unknown_specs])
    parent_order = [spec.unknown_parent for _, spec in unknown_specs]
    best_parent_local = score_matrix[:, columns].argmax(dim=1)
    global_counts = Counter()
    by_parent = defaultdict(Counter)
    for parent_index, parent in enumerate(parent_order):
        sample_indices = (
            best_parent_local == parent_index
        ).nonzero(as_tuple=False).flatten()
        if int(sample_indices.numel()) == 0:
            continue
        images = F.normalize(
            features.index_select(0, sample_indices).float(),
            dim=-1,
        )
        prototypes = F.normalize(
            unknown_features_by_parent[parent].float(),
            dim=-1,
        )
        slots = (images @ prototypes.t()).argmax(dim=1).tolist()
        for slot in slots:
            global_counts[int(slot)] += 1
            by_parent[parent][int(slot)] += 1

    def effective_count(counts):
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probabilities = torch.tensor(
            [count / total for count in counts.values()],
            dtype=torch.float64,
        )
        entropy = -(
            probabilities * probabilities.clamp_min(1e-12).log()
        ).sum()
        return float(entropy.exp())

    parent_effective = {
        parent: effective_count(counts)
        for parent, counts in by_parent.items()
    }
    return {
        "global_slot_counts": dict(sorted(global_counts.items())),
        "global_effective_count": effective_count(global_counts),
        "mean_parent_effective_count": (
            sum(parent_effective.values()) / max(1, len(parent_effective))
        ),
        "parent_effective_count": parent_effective,
    }


def main():
    cli = parse_args()
    args = load_config(cli.config)
    configure_reproducibility(
        args.seed,
        deterministic=args.deterministic,
    )
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(
        REPO_ROOT,
        args.id_split,
        args.hierarchy,
    )
    _, val_dataset, ood_dataset = build_datasets(args, hierarchy)
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
    (
        _positive_checkpoint,
        clip_model,
        _text_encoder,
        _prompt_cfg,
        positive,
        _replaced_modules,
    ) = load_frozen_positive_stack(args, hierarchy, device)
    edge_features, _, _, _ = positive_feature_layout(
        hierarchy,
        positive,
        device,
    )
    unknown_parents = sorted(
        parent
        for parent in hierarchy.parent2children
        if parent != "root"
    )
    learner = ParentSpecificVirtualSiblingPromptLearner(
        positive,
        unknown_parents,
        num_unknown_prompts=args.num_unknown_prompts,
        shared_init_noise=args.shared_init_noise,
        parent_init_noise=args.parent_init_noise,
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    learner.load_prompt_state(checkpoint["unknown_state_dict"])
    learner.eval()
    positive.text_encoder.text_model.eval()
    with torch.no_grad():
        unknown_tensor = learner.encode_parents(
            unknown_parents
        ).float().cpu()
    unknown_features = {
        parent: unknown_tensor[index]
        for index, parent in enumerate(unknown_parents)
    }
    edge_features = {
        edge: feature.float().cpu()
        for edge, feature in edge_features.items()
    }
    terminal_specs = build_metric_terminal_specs(
        hierarchy,
        unknown_parents=unknown_parents,
        allow_root_unknown=False,
    )
    val_payload = encode_dataset_features(
        args,
        clip_model,
        val_dataset,
        val_loader,
        device,
        "encode ID calibration",
    )
    ood_payload = encode_dataset_features(
        args,
        clip_model,
        ood_dataset,
        ood_loader,
        device,
        "encode OOD analysis",
    )
    val_scores = score_payload(
        val_payload,
        args,
        hierarchy,
        edge_features,
        terminal_specs,
        unknown_features,
    )
    ood_scores = score_payload(
        ood_payload,
        args,
        hierarchy,
        edge_features,
        terminal_specs,
        unknown_features,
    )
    raw_val = decode_at_threshold(
        hierarchy,
        val_payload,
        val_scores,
        terminal_specs,
        0.0,
    )
    raw_ood = decode_at_threshold(
        hierarchy,
        ood_payload,
        ood_scores,
        terminal_specs,
        0.0,
    )
    binary = binary_ood_metrics(
        raw_val["ood_score"].numpy(),
        raw_ood["ood_score"].numpy(),
    )
    rows = []
    for acceptance in cli.id_acceptances:
        threshold = float(torch.quantile(
            raw_val["ood_score"],
            float(acceptance),
        ))
        val = decode_at_threshold(
            hierarchy,
            val_payload,
            val_scores,
            terminal_specs,
            threshold,
        )
        ood = decode_at_threshold(
            hierarchy,
            ood_payload,
            ood_scores,
            terminal_specs,
            threshold,
        )
        mixed = mixed_summary(val["metrics"], ood["metrics"])
        rows.append({
            "id_acceptance_target": float(acceptance),
            "threshold": threshold,
            "id_unknown_selection_rate": (
                val["unknown_selection_rate"]
            ),
            "ood_unknown_selection_rate": (
                ood["unknown_selection_rate"]
            ),
            "id_balanced_acc": float(
                val["metrics"]["balanced_acc"]
            ),
            "ood_balanced_acc": float(
                ood["metrics"]["balanced_acc"]
            ),
            "mixed_balanced_acc": float(
                mixed["mixed_balanced_acc"]
            ),
            "id_balanced_hdist": float(
                val["metrics"]["balanced_hdist"]
            ),
            "ood_balanced_hdist": float(
                ood["metrics"]["balanced_hdist"]
            ),
            "mixed_balanced_hdist": float(
                mixed["mixed_balanced_hdist"]
            ),
        })
    result = {
        "method": "tree_loco_id_calibration_analysis",
        "used_actual_ood_for_training_or_selection": False,
        "binary_ood": binary,
        "calibration_rows": rows,
        "id_prototype_usage": prototype_usage(
            val_payload["features"],
            val_scores,
            terminal_specs,
            unknown_features,
        ),
        "ood_prototype_usage": prototype_usage(
            ood_payload["features"],
            ood_scores,
            terminal_specs,
            unknown_features,
        ),
    }
    output_dir = Path(args.result_path).parent
    result_path = output_dir / "calibration-analysis.result"
    json_path = Path(args.diagnostics_path).parent / (
        "calibration-analysis.json"
    )
    ensure_dir(output_dir)
    torch.save(result, result_path)
    save_json(json_path, json_ready(result))
    print("binary", binary)
    for row in rows:
        print(row)
    print("ID slots", result["id_prototype_usage"])
    print("OOD slots", result["ood_prototype_usage"])
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
