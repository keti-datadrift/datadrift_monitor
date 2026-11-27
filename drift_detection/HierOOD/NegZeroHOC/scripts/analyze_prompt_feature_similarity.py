from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_child_only import encode_leaf_text_features, load_config, path_edges
from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.evaluation import build_hierarchy, node_labels_from_feature_targets
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.output_layout import experiment_artifact_path
from negzerohoc.runtime import available_device
from negzerohoc.semantic_index import build_semantic_index

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--split", default="val", choices=["val", "ood"])
    parser.add_argument("--top-k-classes", type=int, default=20)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def summarize(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    quantiles = torch.quantile(tensor, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)) if tensor.numel() > 1 else 0.0,
        "min": float(tensor.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(tensor.max()),
    }


def labels_to_nodes(hierarchy, node_labels: torch.Tensor) -> list[str]:
    return [hierarchy.id_node_list[int(idx)] for idx in node_labels.tolist()]


@torch.no_grad()
def flat_prompt_alignment(
    features: torch.Tensor,
    true_nodes: list[str],
    leaf_names: list[str],
    leaf_features: torch.Tensor,
    batch_size: int,
    device: str,
) -> tuple[dict, dict]:
    leaf_to_idx = {leaf: idx for idx, leaf in enumerate(leaf_names)}
    leaf_features = leaf_features.to(device)
    true_sims = []
    top_sims = []
    margins = []
    ranks = []
    per_class = defaultdict(lambda: {"true_sims": [], "ranks": [], "top1": 0, "total": 0})
    skipped = 0

    starts = range(0, features.shape[0], batch_size)
    if tqdm is not None:
        starts = tqdm(starts, total=(features.shape[0] + batch_size - 1) // batch_size, desc="flat prompt sims", leave=False)

    for start in starts:
        end = min(start + batch_size, features.shape[0])
        batch_nodes = true_nodes[start:end]
        valid_positions = [(pos, leaf_to_idx[node], node) for pos, node in enumerate(batch_nodes) if node in leaf_to_idx]
        skipped += len(batch_nodes) - len(valid_positions)
        if not valid_positions:
            continue

        batch = F.normalize(features[start:end].float().to(device), dim=-1)
        sims = batch @ leaf_features.T
        for pos, true_idx, node in valid_positions:
            row = sims[pos].detach().cpu()
            true_sim = float(row[true_idx])
            top_idx = int(torch.argmax(row).item())
            top_sim = float(row[top_idx])
            rank = int((row > row[true_idx]).sum().item()) + 1
            margin = top_sim - true_sim

            true_sims.append(true_sim)
            top_sims.append(top_sim)
            margins.append(margin)
            ranks.append(float(rank))
            per_class[node]["true_sims"].append(true_sim)
            per_class[node]["ranks"].append(float(rank))
            per_class[node]["top1"] += int(top_idx == true_idx)
            per_class[node]["total"] += 1

    rank_tensor = torch.tensor(ranks, dtype=torch.float32) if ranks else torch.empty(0)
    aggregate = {
        "num_valid_samples": int(len(true_sims)),
        "num_skipped_samples": int(skipped),
        "true_prompt_cosine": summarize(true_sims),
        "top_prompt_cosine": summarize(top_sims),
        "top_minus_true_margin": summarize(margins),
        "true_prompt_rank": summarize(ranks),
        "recall_at_1": float((rank_tensor <= 1).float().mean()) if rank_tensor.numel() else None,
        "recall_at_5": float((rank_tensor <= 5).float().mean()) if rank_tensor.numel() else None,
        "recall_at_10": float((rank_tensor <= 10).float().mean()) if rank_tensor.numel() else None,
    }

    per_class_summary = {}
    for node, stats in per_class.items():
        total = stats["total"]
        per_class_summary[node] = {
            "count": int(total),
            "mean_true_prompt_cosine": float(torch.tensor(stats["true_sims"]).mean()),
            "mean_true_prompt_rank": float(torch.tensor(stats["ranks"]).mean()),
            "flat_top1_acc": float(stats["top1"] / total) if total else None,
        }

    return aggregate, per_class_summary


@torch.no_grad()
def class_centroid_alignment(
    features: torch.Tensor,
    true_nodes: list[str],
    leaf_names: list[str],
    leaf_features: torch.Tensor,
    top_k: int,
) -> dict:
    leaf_to_idx = {leaf: idx for idx, leaf in enumerate(leaf_names)}
    sample_indices_by_leaf = defaultdict(list)
    for sample_idx, node in enumerate(true_nodes):
        if node in leaf_to_idx:
            sample_indices_by_leaf[node].append(sample_idx)

    rows = []
    for leaf, sample_indices in sample_indices_by_leaf.items():
        prompt = leaf_features[leaf_to_idx[leaf]]
        class_features = F.normalize(features[sample_indices].float(), dim=-1)
        centroid = F.normalize(class_features.mean(dim=0), dim=0)
        sample_sims = class_features @ prompt
        rows.append({
            "class": leaf,
            "count": int(len(sample_indices)),
            "centroid_prompt_cosine": float(torch.dot(centroid, prompt)),
            "mean_sample_prompt_cosine": float(sample_sims.mean()),
            "std_sample_prompt_cosine": float(sample_sims.std(unbiased=False)) if len(sample_indices) > 1 else 0.0,
        })

    rows_by_low_centroid = sorted(rows, key=lambda row: row["centroid_prompt_cosine"])
    rows_by_high_centroid = sorted(rows, key=lambda row: row["centroid_prompt_cosine"], reverse=True)
    return {
        "num_classes_with_samples": len(rows),
        "centroid_prompt_cosine": summarize([row["centroid_prompt_cosine"] for row in rows]),
        "mean_sample_prompt_cosine": summarize([row["mean_sample_prompt_cosine"] for row in rows]),
        "lowest_centroid_classes": rows_by_low_centroid[:top_k],
        "highest_centroid_classes": rows_by_high_centroid[:top_k],
    }


@torch.no_grad()
def oracle_local_prompt_alignment(
    features: torch.Tensor,
    true_nodes: list[str],
    hierarchy,
    semantic_index,
    batch_size: int,
    device: str,
    top_k: int,
) -> dict:
    parent_to_entries = defaultdict(list)
    for sample_idx, node in enumerate(true_nodes):
        for parent, child in path_edges(hierarchy, node):
            if parent in semantic_index:
                parent_to_entries[parent].append((sample_idx, child))

    by_depth = defaultdict(lambda: {"true_sims": [], "ranks": [], "margins": []})
    by_parent = defaultdict(lambda: {"true_sims": [], "ranks": [], "margins": []})

    parents = list(parent_to_entries)
    if tqdm is not None:
        parents = tqdm(parents, desc="local prompt sims", leave=False)

    for parent in parents:
        entries = parent_to_entries[parent]
        local = semantic_index[parent]
        child_to_idx = {child: idx for idx, child in enumerate(local.children)}
        valid_entries = [(sample_idx, child_to_idx[child]) for sample_idx, child in entries if child in child_to_idx]
        if not valid_entries:
            continue

        depth = len(hierarchy.node_ancestors.get(parent, []))
        child_features = local.child_features.to(device)
        for start in range(0, len(valid_entries), batch_size):
            batch_entries = valid_entries[start:start + batch_size]
            sample_indices = [idx for idx, _ in batch_entries]
            true_local_indices = [idx for _, idx in batch_entries]
            batch = F.normalize(features[sample_indices].float().to(device), dim=-1)
            sims = (batch @ child_features.T).detach().cpu()
            for row_idx, true_idx in enumerate(true_local_indices):
                row = sims[row_idx]
                true_sim = float(row[true_idx])
                top_sim = float(row.max())
                rank = int((row > row[true_idx]).sum().item()) + 1
                margin = top_sim - true_sim
                by_depth[depth]["true_sims"].append(true_sim)
                by_depth[depth]["ranks"].append(float(rank))
                by_depth[depth]["margins"].append(margin)
                by_parent[parent]["true_sims"].append(true_sim)
                by_parent[parent]["ranks"].append(float(rank))
                by_parent[parent]["margins"].append(margin)

    def finish(stats: dict) -> dict:
        rank_tensor = torch.tensor(stats["ranks"], dtype=torch.float32) if stats["ranks"] else torch.empty(0)
        return {
            "num_edges": int(len(stats["ranks"])),
            "true_child_cosine": summarize(stats["true_sims"]),
            "true_child_rank": summarize(stats["ranks"]),
            "top_minus_true_margin": summarize(stats["margins"]),
            "recall_at_1": float((rank_tensor <= 1).float().mean()) if rank_tensor.numel() else None,
        }

    parent_rows = [(parent, finish(stats)) for parent, stats in by_parent.items()]
    parent_rows.sort(key=lambda item: (
        item[1]["recall_at_1"] if item[1]["recall_at_1"] is not None else 1.0,
        item[1]["true_child_cosine"]["mean"] if item[1]["true_child_cosine"]["mean"] is not None else 1.0,
    ))

    all_true_sims = []
    all_ranks = []
    all_margins = []
    for stats in by_depth.values():
        all_true_sims.extend(stats["true_sims"])
        all_ranks.extend(stats["ranks"])
        all_margins.extend(stats["margins"])
    all_rank_tensor = torch.tensor(all_ranks, dtype=torch.float32) if all_ranks else torch.empty(0)

    return {
        "overall": {
            "num_edges": int(len(all_ranks)),
            "true_child_cosine": summarize(all_true_sims),
            "true_child_rank": summarize(all_ranks),
            "top_minus_true_margin": summarize(all_margins),
            "recall_at_1": float((all_rank_tensor <= 1).float().mean()) if all_rank_tensor.numel() else None,
        },
        "by_parent_depth": {str(depth): finish(stats) for depth, stats in sorted(by_depth.items())},
        "hardest_parents": {parent: stats for parent, stats in parent_rows[:top_k]},
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = available_device(cfg["device"])

    hierarchy, _ = build_hierarchy(REPO_ROOT, cfg["id_split"], cfg["hierarchy"])
    payload = load_feature_file(Path(cfg["features_dir"]) / f"{args.split}-features.pt")
    node_labels = node_labels_from_feature_targets(hierarchy, payload["classes"], payload["targets"])
    true_nodes = labels_to_nodes(hierarchy, node_labels)

    backend = ClipBackend(cfg["clip_model"], device=device, local_files_only=cfg["local_files_only"])
    leaf_names, leaf_features = encode_leaf_text_features(cfg["dataset"], hierarchy, backend)
    semantic_index = build_semantic_index(cfg["dataset"], hierarchy, backend, mode="child_only")

    flat_alignment, per_class_alignment = flat_prompt_alignment(
        payload["features"],
        true_nodes,
        leaf_names,
        leaf_features,
        cfg["batch_size"],
        device,
    )
    centroid_alignment = class_centroid_alignment(
        payload["features"],
        true_nodes,
        leaf_names,
        leaf_features,
        args.top_k_classes,
    )
    local_alignment = oracle_local_prompt_alignment(
        payload["features"],
        true_nodes,
        hierarchy,
        semantic_index,
        cfg["batch_size"],
        device,
        args.top_k_classes,
    )

    output = {
        "config": cfg["config"],
        "dataset": cfg["dataset"],
        "split": args.split,
        "num_samples": int(node_labels.numel()),
        "num_id_leaf_classes": len(hierarchy.train_classes),
        "clip_model": cfg["clip_model"],
        "flat_leaf_prompt_alignment": flat_alignment,
        "class_centroid_prompt_alignment": centroid_alignment,
        "oracle_local_prompt_alignment": local_alignment,
        "per_class_leaf_prompt_alignment": per_class_alignment,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        model_key = safe_model_name(cfg["clip_model"])
        out_path = experiment_artifact_path(
            cfg["outdir"],
            cfg["experiment_name"],
            "diagnostics",
            f"{cfg['dataset']}-clip_{model_key}-{args.split}-prompt-feature-similarity.json",
        )

    ensure_dir(out_path.parent)
    save_json(out_path, output)
    print(f"saved: {out_path}")
    print("flat true prompt cosine mean:", f"{flat_alignment['true_prompt_cosine']['mean']:.6f}")
    print("flat true prompt rank mean:", f"{flat_alignment['true_prompt_rank']['mean']:.6f}")
    print("flat recall@1:", f"{flat_alignment['recall_at_1']:.6f}")
    print("class centroid cosine mean:", f"{centroid_alignment['centroid_prompt_cosine']['mean']:.6f}")
    print("local true child cosine mean:", f"{local_alignment['overall']['true_child_cosine']['mean']:.6f}")
    print("local recall@1:", f"{local_alignment['overall']['recall_at_1']:.6f}")


if __name__ == "__main__":
    main()
