from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    get_results,
    make_distance_mats,
    node_labels_from_feature_targets,
)
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.inference import predict_features
from negzerohoc.output_layout import experiment_artifact_path, resolve_shared_feature_dir
from negzerohoc.prompts import build_positive_prompts, infer_node_role, node_path_names
from negzerohoc.runtime import available_device, configured_device
from negzerohoc.semantic_index import build_semantic_index


def load_config(path: str | Path) -> dict:
    cfg = load_yaml_config(path)

    dataset_cfg = cfg.get("dataset", {})
    runtime_cfg = cfg.get("runtime", {})
    clip_cfg = cfg.get("clip", {})
    inference_cfg = cfg.get("inference", {})
    experiment_cfg = cfg.get("experiment", {})

    output_root = experiment_cfg.get("output_root", "outputs")
    features_dir = resolve_shared_feature_dir(
        inference_cfg.get("features_dir"),
        output_root=output_root,
    )
    if not features_dir:
        raise ValueError(f"Missing inference.features_dir in {path}")

    return {
        "config": str(path),
        "experiment_name": experiment_cfg.get("name", "child-only-diagnostics"),
        "dataset": dataset_cfg.get("name", "fgvc-aircraft"),
        "hierarchy": dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        "id_split": dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        "features_dir": str(features_dir),
        "clip_model": clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        "local_files_only": clip_cfg.get("local_files_only", True),
        "device": configured_device(runtime_cfg),
        "batch_size": int(inference_cfg.get("batch_size", 1024)),
        "outdir": output_root,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--split", default="val", choices=["val", "ood"])
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path. Defaults to the experiment diagnostics directory.",
    )
    return parser.parse_args()


def metric_summary(metrics: dict) -> dict:
    return {
        "acc": float(metrics["acc"]),
        "balanced_acc": float(metrics["balanced_acc"]),
        "avg_hdist": float(metrics["avg_hdist"]),
        "balanced_hdist": float(metrics["balanced_hdist"]),
    }


def labels_to_nodes(hierarchy, node_labels: torch.Tensor) -> list[str]:
    return [hierarchy.id_node_list[int(idx)] for idx in node_labels.tolist()]


def path_edges(hierarchy, node: str) -> list[tuple[str, str]]:
    ancestors = [hierarchy.id_node_list[i] for i in hierarchy.node_ancestors.get(node, [])]
    path = ancestors + [node]
    return list(zip(path[:-1], path[1:]))


@torch.no_grad()
def encode_leaf_text_features(dataset_name: str, hierarchy, backend: ClipBackend) -> tuple[list[str], torch.Tensor]:
    leaf_names = list(hierarchy.train_classes)
    leaf_features = []

    iterator = leaf_names
    if tqdm is not None:
        iterator = tqdm(iterator, desc="encode flat leaf prompts", leave=False)

    for leaf in iterator:
        parent = hierarchy.child2parent.get(leaf)
        path = node_path_names(hierarchy, leaf, include_self=True, dataset_name=dataset_name)
        role = infer_node_role(dataset_name, leaf)
        prompts = build_positive_prompts(dataset_name, leaf, parent, path, role)
        leaf_features.append(backend.encode_prompt_ensemble(prompts).cpu())

    return leaf_names, torch.stack(leaf_features, dim=0)


@torch.no_grad()
def predict_flat_leaf(
    features: torch.Tensor,
    leaf_features: torch.Tensor,
    leaf_names: list[str],
    hierarchy,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    leaf_features = leaf_features.to(device)
    preds = []
    starts = range(0, features.shape[0], batch_size)
    if tqdm is not None:
        starts = tqdm(starts, total=(features.shape[0] + batch_size - 1) // batch_size, desc="flat leaf", leave=False)

    for start in starts:
        end = min(start + batch_size, features.shape[0])
        batch = F.normalize(features[start:end].float().to(device), dim=-1)
        winners = torch.argmax(batch @ leaf_features.T, dim=1).cpu()
        preds.extend(hierarchy.id_node_list.index(leaf_names[int(idx)]) for idx in winners.tolist())
    return torch.tensor(preds, dtype=torch.long)


@torch.no_grad()
def predict_greedy_child_only(
    payload: dict,
    hierarchy,
    semantic_index,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, dict]:
    features = payload["features"]
    preds = []
    stop_nodes = []
    starts = range(0, features.shape[0], batch_size)
    if tqdm is not None:
        starts = tqdm(starts, total=(features.shape[0] + batch_size - 1) // batch_size, desc="greedy child-only", leave=False)

    for start in starts:
        end = min(start + batch_size, features.shape[0])
        out = predict_features(
            features[start:end].to(device),
            hierarchy,
            semantic_index,
            mode="child_only",
            return_trace=False,
        )
        batch_preds = out["preds"].cpu()
        preds.append(batch_preds)
        stop_nodes.extend(hierarchy.id_node_list[int(idx)] for idx in batch_preds.tolist())

    preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
    depth_counts = Counter(len(hierarchy.node_ancestors.get(node, [])) for node in stop_nodes)
    node_counts = Counter(stop_nodes)
    return preds, {
        "stop_depth_counts": dict(sorted(depth_counts.items())),
        "top_stop_nodes": dict(node_counts.most_common(20)),
    }


@torch.no_grad()
def oracle_path_local_diagnostics(
    features: torch.Tensor,
    true_nodes: list[str],
    hierarchy,
    semantic_index,
    batch_size: int,
    device: str,
) -> dict:
    parent_to_samples = defaultdict(list)
    sample_edge_counts = [0] * len(true_nodes)

    for sample_idx, node in enumerate(true_nodes):
        for parent, child in path_edges(hierarchy, node):
            if parent in semantic_index:
                parent_to_samples[parent].append((sample_idx, child))
                sample_edge_counts[sample_idx] += 1

    sample_all_correct = [count > 0 for count in sample_edge_counts]
    by_depth = defaultdict(lambda: {"correct": 0, "total": 0})
    by_parent = defaultdict(lambda: {"correct": 0, "total": 0})
    first_wrong_depth = Counter()

    parents = list(parent_to_samples)
    if tqdm is not None:
        parents = tqdm(parents, desc="oracle local", leave=False)

    for parent in parents:
        entries = parent_to_samples[parent]
        children = semantic_index[parent].children
        child_to_local = {child: i for i, child in enumerate(children)}
        valid_entries = [(idx, child) for idx, child in entries if child in child_to_local]
        if not valid_entries:
            continue

        parent_depth = len(hierarchy.node_ancestors.get(parent, []))
        child_features = semantic_index[parent].child_features.to(device)

        for start in range(0, len(valid_entries), batch_size):
            batch_entries = valid_entries[start:start + batch_size]
            sample_indices = [idx for idx, _ in batch_entries]
            target_local = torch.tensor([child_to_local[child] for _, child in batch_entries], dtype=torch.long)
            batch_features = F.normalize(features[sample_indices].float().to(device), dim=-1)
            pred_local = torch.argmax(batch_features @ child_features.T, dim=1).cpu()
            correct = pred_local.eq(target_local)

            for local_pos, ok in enumerate(correct.tolist()):
                sample_idx = sample_indices[local_pos]
                by_depth[parent_depth]["total"] += 1
                by_parent[parent]["total"] += 1
                if ok:
                    by_depth[parent_depth]["correct"] += 1
                    by_parent[parent]["correct"] += 1
                else:
                    sample_all_correct[sample_idx] = False
                    first_wrong_depth[parent_depth] += 1

    def finish(counter: dict) -> dict:
        return {
            "correct": int(counter["correct"]),
            "total": int(counter["total"]),
            "acc": float(counter["correct"] / counter["total"]) if counter["total"] else None,
        }

    parent_rows = [
        (parent, finish(stats))
        for parent, stats in by_parent.items()
    ]
    parent_rows.sort(key=lambda item: (item[1]["acc"] if item[1]["acc"] is not None else 1.0, -item[1]["total"]))

    total_edges = sum(stats["total"] for stats in by_depth.values())
    correct_edges = sum(stats["correct"] for stats in by_depth.values())
    return {
        "local_edge_acc": float(correct_edges / total_edges) if total_edges else None,
        "path_exact_acc": float(sum(sample_all_correct) / len(sample_all_correct)) if sample_all_correct else None,
        "by_parent_depth": {str(depth): finish(stats) for depth, stats in sorted(by_depth.items())},
        "first_wrong_depth_counts": dict(sorted(first_wrong_depth.items())),
        "hardest_parents": {
            parent: stats for parent, stats in parent_rows[:20]
        },
    }


def build_leaf_edge_map(hierarchy) -> tuple[list[str], dict[str, list[tuple[int, str]]], torch.Tensor]:
    leaf_names = list(hierarchy.train_classes)
    edge_map = defaultdict(list)
    path_lengths = []
    for leaf_idx, leaf in enumerate(leaf_names):
        edges = path_edges(hierarchy, leaf)
        path_lengths.append(max(len(edges), 1))
        for parent, child in edges:
            edge_map[parent].append((leaf_idx, child))
    return leaf_names, edge_map, torch.tensor(path_lengths, dtype=torch.float32)


@torch.no_grad()
def predict_path_score_leaf(
    features: torch.Tensor,
    hierarchy,
    semantic_index,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    leaf_names, edge_map, path_lengths = build_leaf_edge_map(hierarchy)
    node_indices = torch.tensor([hierarchy.id_node_list.index(leaf) for leaf in leaf_names], dtype=torch.long)
    preds = []
    starts = range(0, features.shape[0], batch_size)
    if tqdm is not None:
        starts = tqdm(starts, total=(features.shape[0] + batch_size - 1) // batch_size, desc="path-score leaf", leave=False)

    for start in starts:
        end = min(start + batch_size, features.shape[0])
        batch = F.normalize(features[start:end].float().to(device), dim=-1)
        scores = torch.zeros((batch.shape[0], len(leaf_names)), device=device)
        for parent, leaf_child_pairs in edge_map.items():
            if parent not in semantic_index:
                continue
            local = semantic_index[parent]
            child_to_local = {child: i for i, child in enumerate(local.children)}
            valid_pairs = [(leaf_idx, child_to_local[child]) for leaf_idx, child in leaf_child_pairs if child in child_to_local]
            if not valid_pairs:
                continue
            local_scores = batch @ local.child_features.to(device).T
            leaf_indices = torch.tensor([leaf_idx for leaf_idx, _ in valid_pairs], dtype=torch.long, device=device)
            child_indices = torch.tensor([child_idx for _, child_idx in valid_pairs], dtype=torch.long, device=device)
            scores[:, leaf_indices] += local_scores[:, child_indices]
        scores = scores / path_lengths.to(device).unsqueeze(0)
        preds.append(node_indices[torch.argmax(scores.cpu(), dim=1)])
    return torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)
    device = available_device(cfg["device"])

    hierarchy, _ = build_hierarchy(REPO_ROOT, cfg["id_split"], cfg["hierarchy"])
    dists_mats = make_distance_mats(hierarchy)
    payload = load_feature_file(Path(cfg["features_dir"]) / f"{cli_args.split}-features.pt")
    node_labels = node_labels_from_feature_targets(hierarchy, payload["classes"], payload["targets"])
    true_nodes = labels_to_nodes(hierarchy, node_labels)

    backend = ClipBackend(cfg["clip_model"], device=device, local_files_only=cfg["local_files_only"])
    semantic_index = build_semantic_index(cfg["dataset"], hierarchy, backend, mode="child_only")

    leaf_names, leaf_features = encode_leaf_text_features(cfg["dataset"], hierarchy, backend)
    flat_preds = predict_flat_leaf(
        payload["features"],
        leaf_features,
        leaf_names,
        hierarchy,
        cfg["batch_size"],
        device,
    )
    greedy_preds, greedy_diag = predict_greedy_child_only(
        payload,
        hierarchy,
        semantic_index,
        cfg["batch_size"],
        device,
    )
    path_score_preds = predict_path_score_leaf(
        payload["features"],
        hierarchy,
        semantic_index,
        cfg["batch_size"],
        device,
    )
    oracle_diag = oracle_path_local_diagnostics(
        payload["features"],
        true_nodes,
        hierarchy,
        semantic_index,
        cfg["batch_size"],
        device,
    )

    diagnostics = {
        "config": cfg["config"],
        "dataset": cfg["dataset"],
        "split": cli_args.split,
        "num_samples": int(node_labels.numel()),
        "num_id_leaf_classes": len(hierarchy.train_classes),
        "num_id_nodes": len(hierarchy.id_node_list),
        "clip_model": cfg["clip_model"],
        "flat_leaf": metric_summary(get_results(flat_preds, node_labels, hierarchy, dists_mats=dists_mats)),
        "greedy_child_only": {
            "metrics": metric_summary(get_results(greedy_preds, node_labels, hierarchy, dists_mats=dists_mats)),
            "diagnostics": greedy_diag,
        },
        "path_score_leaf": metric_summary(get_results(path_score_preds, node_labels, hierarchy, dists_mats=dists_mats)),
        "oracle_path_local": oracle_diag,
    }

    if cli_args.out:
        out_path = Path(cli_args.out)
    else:
        model_key = safe_model_name(cfg["clip_model"])
        out_path = experiment_artifact_path(
            cfg["outdir"],
            cfg["experiment_name"],
            "diagnostics",
            f"{cfg['dataset']}-clip_{model_key}-{cli_args.split}-child-only-diagnostics.json",
        )

    ensure_dir(out_path.parent)
    save_json(out_path, diagnostics)
    print(f"saved: {out_path}")
    print(f"flat leaf BAcc: {diagnostics['flat_leaf']['balanced_acc']:.6f}")
    print(f"greedy child-only BAcc: {diagnostics['greedy_child_only']['metrics']['balanced_acc']:.6f}")
    print(f"path-score leaf BAcc: {diagnostics['path_score_leaf']['balanced_acc']:.6f}")
    print(f"oracle local edge acc: {diagnostics['oracle_path_local']['local_edge_acc']:.6f}")
    print(f"oracle path exact acc: {diagnostics['oracle_path_local']['path_exact_acc']:.6f}")


if __name__ == "__main__":
    main()
