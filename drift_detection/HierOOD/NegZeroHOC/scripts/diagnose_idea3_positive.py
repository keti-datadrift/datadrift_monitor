from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_idea3 import load_config, load_models
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import load_feature_file, save_json
from negzerohoc.idea3_inference import build_idea3_semantic_index
from negzerohoc.runtime import available_device
from negzerohoc.training_data import build_positive_edge_examples, group_examples_by_parent


def node_path(hierarchy, node: str) -> list[str]:
    return [hierarchy.id_node_list[idx] for idx in hierarchy.node_ancestors.get(node, [])] + [node]


@torch.no_grad()
def oracle_local_accuracy_by_depth(args, hierarchy, positive, payload, device: str) -> dict:
    examples = build_positive_edge_examples(hierarchy, payload)
    grouped = group_examples_by_parent(examples)
    features = payload["features"].float()
    depth_total = Counter()
    depth_correct = Counter()
    parent_total = Counter()
    parent_correct = Counter()

    for parent, parent_examples in grouped.items():
        children = list(hierarchy.parent2children[parent])
        child_to_idx = {child: idx for idx, child in enumerate(children)}
        child_features = positive.encode_children(parent, children).to(device)
        for start in range(0, len(parent_examples), args.batch_size):
            batch = parent_examples[start:start + args.batch_size]
            image_indices = torch.tensor([ex.image_index for ex in batch], dtype=torch.long)
            image_features = features.index_select(0, image_indices).to(device)
            logits = image_features @ child_features.t()
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            target_idx = [child_to_idx[ex.child] for ex in batch]
            depth = batch[0].parent_depth
            for pred, target, ex in zip(pred_idx, target_idx, batch):
                depth_total[depth] += 1
                parent_total[ex.parent] += 1
                if pred == target:
                    depth_correct[depth] += 1
                    parent_correct[ex.parent] += 1

    by_depth = {
        str(depth): {
            "acc": depth_correct[depth] / depth_total[depth],
            "correct": int(depth_correct[depth]),
            "total": int(depth_total[depth]),
        }
        for depth in sorted(depth_total)
    }
    by_parent = {
        parent: {
            "depth": len(hierarchy.node_ancestors.get(parent, [])),
            "acc": parent_correct[parent] / parent_total[parent],
            "correct": int(parent_correct[parent]),
            "total": int(parent_total[parent]),
        }
        for parent in sorted(parent_total)
    }
    worst_parents = sorted(by_parent.items(), key=lambda item: (item[1]["acc"], -item[1]["total"]))[:20]
    return {"by_depth": by_depth, "worst_parents": dict(worst_parents)}


def greedy_prefix_accuracy(hierarchy, payload, result_split: dict) -> dict:
    targets = result_split["targets"].tolist()
    preds = result_split["preds"].tolist()
    max_depth = hierarchy.max_depth
    depth_total = Counter()
    depth_correct = Counter()
    first_error_depth = Counter()

    for target_idx, pred_idx in zip(targets, preds):
        target_node = hierarchy.id_node_list[target_idx]
        pred_node = hierarchy.id_node_list[pred_idx]
        target_path = node_path(hierarchy, target_node)
        pred_path = node_path(hierarchy, pred_node)
        first_error = None
        for depth in range(max_depth + 1):
            if depth >= len(target_path):
                continue
            depth_total[depth] += 1
            ok = depth < len(pred_path) and pred_path[depth] == target_path[depth]
            if ok:
                depth_correct[depth] += 1
            elif first_error is None:
                first_error = depth
        if first_error is None and pred_node != target_node:
            first_error = len(target_path)
        first_error_depth[first_error if first_error is not None else "none"] += 1

    return {
        "prefix_by_depth": {
            str(depth): {
                "acc": depth_correct[depth] / depth_total[depth],
                "correct": int(depth_correct[depth]),
                "total": int(depth_total[depth]),
            }
            for depth in sorted(depth_total)
        },
        "first_error_depth_counts": dict(first_error_depth),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "ood"])
    parser.add_argument("--result", default=None)
    parser.add_argument("--out", default=None)
    args_ns = parser.parse_args()

    args = load_config(args_ns.config, mode_override="positive_child_only")
    args.batch_size = int(args.batch_size)
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    _, positive, _ = load_models(args, hierarchy, device)

    payload = load_feature_file(Path(args.features_dir) / f"{args_ns.split}-features.pt")
    diagnostics = {
        "split": args_ns.split,
        "oracle_local": oracle_local_accuracy_by_depth(args, hierarchy, positive, payload, device),
    }

    result_path = Path(args_ns.result) if args_ns.result else Path(args.result_path)
    if result_path.exists() and args_ns.split in {"val", "ood"}:
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        diagnostics["greedy"] = greedy_prefix_accuracy(hierarchy, payload, result[args_ns.split])

    print(diagnostics)
    if args_ns.out:
        save_json(args_ns.out, diagnostics)
        print(f"saved: {args_ns.out}")


if __name__ == "__main__":
    main()
