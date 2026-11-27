import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from libs.hierarchy import Hierarchy
from libs.utils.hierarchy_utils import get_multidepth_classes


EPS = 1e-12


def get_id_classes(id_classes_fn):
    with open(id_classes_fn, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return sorted(line for line in lines if line)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_full_hierarchy(hierarchy_fn):
    with open(hierarchy_fn, "r", encoding="utf-8") as f:
        root = json.load(f)

    parent_to_children = {}
    child_to_parent = {}
    all_nodes = []

    def dfs(node, parent=None):
        name = node["name"]
        all_nodes.append(name)
        if parent is not None:
            child_to_parent[name] = parent
        children = [child["name"] for child in node.get("children", [])]
        parent_to_children[name] = children
        for child in node.get("children", []):
            dfs(child, name)

    dfs(root)
    return {
        "root": root["name"],
        "all_nodes": all_nodes,
        "parent_to_children": parent_to_children,
        "child_to_parent": child_to_parent,
    }


def build_leaf_descendant_map(full_tree):
    parent_to_children = full_tree["parent_to_children"]
    memo = {}

    def get_leaf_descendants(node_name):
        if node_name in memo:
            return memo[node_name]
        children = parent_to_children.get(node_name, [])
        if not children:
            memo[node_name] = {node_name}
            return memo[node_name]
        leaves = set()
        for child in children:
            leaves.update(get_leaf_descendants(child))
        memo[node_name] = leaves
        return leaves

    for node_name in full_tree["all_nodes"]:
        get_leaf_descendants(node_name)
    return memo


def compute_local_entropy_and_comp(probs, child_indices):
    child_probs = probs[:, child_indices]
    child_sums = child_probs.sum(dim=-1)
    local_probs = child_probs / child_sums.unsqueeze(-1).clamp_min(EPS)
    local_probs = local_probs.clamp_min(EPS)
    entropy = -(local_probs * torch.log(local_probs)).sum(dim=-1)
    p_comp = (1.0 - child_sums).clamp(min=0.0, max=1.0)
    return entropy, p_comp


def stats_or_nan(values):
    if not values:
        return math.nan, math.nan, 0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean()), float(tensor.std(unbiased=False)), int(tensor.numel())


def compute_ece(probs, labels, num_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, dtype=torch.float32)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower) & confidences.le(bin_upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece.item())


def compute_nll(probs, labels):
    true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(EPS)
    return float((-torch.log(true_probs)).mean().item())


def compute_brier(probs, labels):
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return float(brier.item())


def build_node_analysis_plan(hierarchy, id_classes, ood_classes, full_leaf_descendants):
    internal_nodes_by_depth = defaultdict(list)
    available_child_names = {}

    for node_name, children in hierarchy.parent2children.items():
        depth = len(hierarchy.node_ancestors[node_name])
        if depth == 0:
            continue
        internal_nodes_by_depth[depth].append(node_name)
        available_child_names[node_name] = list(children)

    id_leaf_sets = {}
    local_ood_leaf_sets = {}

    for node_name in hierarchy.parent2children.keys():
        reduced_children = hierarchy.parent2children[node_name]

        id_leaves = set()
        child_leaf_union = set()
        for child_name in reduced_children:
            child_full_leaves = full_leaf_descendants[child_name]
            child_leaf_union.update(child_full_leaves)
            id_leaves.update(set(id_classes).intersection(child_full_leaves))

        full_node_leaves = full_leaf_descendants[node_name]
        ood_leaves = (set(ood_classes).intersection(full_node_leaves)) - child_leaf_union

        id_leaf_sets[node_name] = id_leaves
        local_ood_leaf_sets[node_name] = ood_leaves

    return internal_nodes_by_depth, available_child_names, id_leaf_sets, local_ood_leaf_sets


def classifier_depth_to_logits_index(max_depth, parent_depth):
    return max_depth - parent_depth - 1


def load_predictions(basedir, max_depth, split, device):
    logits_by_height = []
    targets_by_height = []
    for height in range(max_depth):
        path = os.path.join(basedir, f"H{height}", f"{split}-preds.out")
        obj = torch.load(path, map_location=device, weights_only=False)
        logits_by_height.append(obj["logits"].to(device))
        targets_by_height.append(obj["targets"].long().to(device))
    return logits_by_height, targets_by_height


def compute_entropy_and_comp_rows(hierarchy,
                                  multi_classes,
                                  val_logits_by_height,
                                  val_targets_by_height,
                                  ood_logits_by_height,
                                  ood_targets_by_height,
                                  temperature_candidates,
                                  internal_nodes_by_depth,
                                  available_child_names,
                                  id_leaf_sets,
                                  local_ood_leaf_sets,
                                  id_classes,
                                  ood_classes,
                                  device):
    entropy_rows = []
    comp_rows = []
    entropy_by_child_count = defaultdict(list)
    comp_by_child_count = defaultdict(list)
    comp_by_complement_count = defaultdict(list)

    id_class_names = [id_classes[int(x)] for x in val_targets_by_height[0].tolist()]
    ood_class_names = [ood_classes[int(x)] for x in ood_targets_by_height[0].tolist()]

    analysis_depths = sorted(internal_nodes_by_depth.keys())
    loop_items = [(depth, temp) for depth in analysis_depths for temp in temperature_candidates]

    for depth, temperature in tqdm(loop_items, desc="Entropy/Comp", unit="setting"):
        logits_index = classifier_depth_to_logits_index(hierarchy._max_depth, depth)
        probs_val = torch.softmax(val_logits_by_height[logits_index] / temperature, dim=-1)
        probs_ood = torch.softmax(ood_logits_by_height[logits_index] / temperature, dim=-1)
        classifier_classes = multi_classes[depth]

        entropy_id_values = []
        entropy_ood_values = []
        comp_id_values = []
        comp_ood_values = []

        nodes_with_id = 0
        nodes_with_ood = 0

        for node_name in internal_nodes_by_depth[depth]:
            child_names = [child_name for child_name in available_child_names[node_name]
                           if child_name in classifier_classes]
            if len(child_names) <= 1:
                continue
            num_children = len(child_names)
            num_complements = len(local_ood_leaf_sets[node_name])
            child_indices = [classifier_classes.index(child_name) for child_name in child_names]

            entropy_val, comp_val = compute_local_entropy_and_comp(probs_val, child_indices)
            entropy_ood, comp_ood = compute_local_entropy_and_comp(probs_ood, child_indices)

            id_mask = torch.tensor(
                [class_name in id_leaf_sets[node_name] for class_name in id_class_names],
                dtype=torch.bool,
                device=device,
            )
            ood_mask = torch.tensor(
                [class_name in local_ood_leaf_sets[node_name] for class_name in ood_class_names],
                dtype=torch.bool,
                device=device,
            )

            if id_mask.any():
                nodes_with_id += 1
                entropy_id_node_values = entropy_val[id_mask].detach().cpu().tolist()
                comp_id_node_values = comp_val[id_mask].detach().cpu().tolist()
                entropy_id_values.extend(entropy_id_node_values)
                comp_id_values.extend(comp_id_node_values)
                entropy_by_child_count[(depth, temperature, num_children, "id")].extend(entropy_id_node_values)
                comp_by_child_count[(depth, temperature, num_children, "id")].extend(comp_id_node_values)
                comp_by_complement_count[(depth, temperature, num_complements, "id")].extend(comp_id_node_values)

            if ood_mask.any():
                nodes_with_ood += 1
                entropy_ood_node_values = entropy_ood[ood_mask].detach().cpu().tolist()
                comp_ood_node_values = comp_ood[ood_mask].detach().cpu().tolist()
                entropy_ood_values.extend(entropy_ood_node_values)
                comp_ood_values.extend(comp_ood_node_values)
                entropy_by_child_count[(depth, temperature, num_children, "ood")].extend(entropy_ood_node_values)
                comp_by_child_count[(depth, temperature, num_children, "ood")].extend(comp_ood_node_values)
                comp_by_complement_count[(depth, temperature, num_complements, "ood")].extend(comp_ood_node_values)

        mean_ent_id, std_ent_id, count_ent_id = stats_or_nan(entropy_id_values)
        mean_ent_ood, std_ent_ood, count_ent_ood = stats_or_nan(entropy_ood_values)
        mean_comp_id, std_comp_id, count_comp_id = stats_or_nan(comp_id_values)
        mean_comp_ood, std_comp_ood, count_comp_ood = stats_or_nan(comp_ood_values)

        entropy_rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_nodes": len(internal_nodes_by_depth[depth]),
            "nodes_with_id": nodes_with_id,
            "nodes_with_ood": nodes_with_ood,
            "num_id_samples": count_ent_id,
            "num_ood_samples": count_ent_ood,
            "mean_entropy_id": mean_ent_id,
            "std_entropy_id": std_ent_id,
            "mean_entropy_ood": mean_ent_ood,
            "std_entropy_ood": std_ent_ood,
            "entropy_gap": mean_ent_ood - mean_ent_id if not math.isnan(mean_ent_id) and not math.isnan(mean_ent_ood) else math.nan,
        })

        comp_rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_nodes": len(internal_nodes_by_depth[depth]),
            "nodes_with_id": nodes_with_id,
            "nodes_with_ood": nodes_with_ood,
            "num_id_samples": count_comp_id,
            "num_ood_samples": count_comp_ood,
            "mean_comp_id": mean_comp_id,
            "std_comp_id": std_comp_id,
            "mean_comp_ood": mean_comp_ood,
            "std_comp_ood": std_comp_ood,
            "comp_gap": mean_comp_ood - mean_comp_id if not math.isnan(mean_comp_id) and not math.isnan(mean_comp_ood) else math.nan,
        })

    entropy_child_rows = []
    grouped_keys = sorted(set((depth, temperature, num_children)
                              for depth, temperature, num_children, _ in entropy_by_child_count.keys()))
    for depth, temperature, num_children in grouped_keys:
        id_values = entropy_by_child_count.get((depth, temperature, num_children, "id"), [])
        ood_values = entropy_by_child_count.get((depth, temperature, num_children, "ood"), [])
        mean_id, std_id, count_id = stats_or_nan(id_values)
        mean_ood, std_ood, count_ood = stats_or_nan(ood_values)
        entropy_child_rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_children": num_children,
            "max_entropy_logK": math.log(num_children),
            "num_id_samples": count_id,
            "num_ood_samples": count_ood,
            "mean_entropy_id": mean_id,
            "std_entropy_id": std_id,
            "mean_entropy_ood": mean_ood,
            "std_entropy_ood": std_ood,
            "entropy_gap": mean_ood - mean_id if not math.isnan(mean_id) and not math.isnan(mean_ood) else math.nan,
            "mean_norm_entropy_id": mean_id / math.log(num_children) if not math.isnan(mean_id) else math.nan,
            "mean_norm_entropy_ood": mean_ood / math.log(num_children) if not math.isnan(mean_ood) else math.nan,
        })

    comp_child_rows = []
    grouped_comp_keys = sorted(set((depth, temperature, num_children)
                                   for depth, temperature, num_children, _ in comp_by_child_count.keys()))
    for depth, temperature, num_children in grouped_comp_keys:
        id_values = comp_by_child_count.get((depth, temperature, num_children, "id"), [])
        ood_values = comp_by_child_count.get((depth, temperature, num_children, "ood"), [])
        mean_id, std_id, count_id = stats_or_nan(id_values)
        mean_ood, std_ood, count_ood = stats_or_nan(ood_values)
        comp_child_rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_children": num_children,
            "num_id_samples": count_id,
            "num_ood_samples": count_ood,
            "mean_comp_id": mean_id,
            "std_comp_id": std_id,
            "mean_comp_ood": mean_ood,
            "std_comp_ood": std_ood,
            "comp_gap": mean_ood - mean_id if not math.isnan(mean_id) and not math.isnan(mean_ood) else math.nan,
        })

    comp_complement_rows = []
    grouped_complement_keys = sorted(set((depth, temperature, num_complements)
                                         for depth, temperature, num_complements, _ in comp_by_complement_count.keys()))
    for depth, temperature, num_complements in grouped_complement_keys:
        id_values = comp_by_complement_count.get((depth, temperature, num_complements, "id"), [])
        ood_values = comp_by_complement_count.get((depth, temperature, num_complements, "ood"), [])
        mean_id, std_id, count_id = stats_or_nan(id_values)
        mean_ood, std_ood, count_ood = stats_or_nan(ood_values)
        comp_complement_rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_complements": num_complements,
            "num_id_samples": count_id,
            "num_ood_samples": count_ood,
            "mean_comp_id": mean_id,
            "std_comp_id": std_id,
            "mean_comp_ood": mean_ood,
            "std_comp_ood": std_ood,
            "comp_gap": mean_ood - mean_id if not math.isnan(mean_id) and not math.isnan(mean_ood) else math.nan,
        })

    return entropy_rows, comp_rows, entropy_child_rows, comp_child_rows, comp_complement_rows


def compute_calibration_rows(val_logits_by_height, val_targets_by_height, temperature_candidates, device):
    calibration_rows = []

    loop_items = [(height, temp) for height in range(len(val_logits_by_height)) for temp in temperature_candidates]
    for height, temperature in tqdm(loop_items, desc="Calibration", unit="setting"):
        logits = val_logits_by_height[height]
        labels = val_targets_by_height[height]
        probs = torch.softmax(logits / temperature, dim=-1)

        calibration_rows.append({
            "classifier_height": height,
            "temperature": temperature,
            "num_samples": int(labels.numel()),
            "ece": compute_ece(probs, labels),
            "nll": compute_nll(probs, labels),
            "brier": compute_brier(probs, labels),
        })

    return calibration_rows


def summarize_best_temperatures(entropy_rows, comp_rows, calibration_rows):
    summary_rows = []

    entropy_by_depth = defaultdict(list)
    comp_by_depth = defaultdict(list)
    calib_by_height = defaultdict(list)

    for row in entropy_rows:
        entropy_by_depth[row["depth"]].append(row)
    for row in comp_rows:
        comp_by_depth[row["depth"]].append(row)
    for row in calibration_rows:
        calib_by_height[row["classifier_height"]].append(row)

    for depth, rows in sorted(entropy_by_depth.items()):
        best_entropy = max(
            [row for row in rows if not math.isnan(row["entropy_gap"])],
            key=lambda row: row["entropy_gap"],
            default=None,
        )
        best_comp = max(
            [row for row in comp_by_depth[depth] if not math.isnan(row["comp_gap"])],
            key=lambda row: row["comp_gap"],
            default=None,
        )

        classifier_height = len(calib_by_height) - depth - 1
        best_ece = min(calib_by_height[classifier_height], key=lambda row: row["ece"])
        best_nll = min(calib_by_height[classifier_height], key=lambda row: row["nll"])
        best_brier = min(calib_by_height[classifier_height], key=lambda row: row["brier"])

        summary_rows.append({
            "depth": depth,
            "best_temperature_entropy_gap": best_entropy["temperature"] if best_entropy else math.nan,
            "best_entropy_gap": best_entropy["entropy_gap"] if best_entropy else math.nan,
            "best_temperature_comp_gap": best_comp["temperature"] if best_comp else math.nan,
            "best_comp_gap": best_comp["comp_gap"] if best_comp else math.nan,
            "classifier_height": classifier_height,
            "best_temperature_ece": best_ece["temperature"],
            "best_ece": best_ece["ece"],
            "best_temperature_nll": best_nll["temperature"],
            "best_nll": best_nll["nll"],
            "best_temperature_brier": best_brier["temperature"],
            "best_brier": best_brier["brier"],
        })

    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--temperatures",
                        nargs="+",
                        type=float,
                        default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available")

    device = args.device
    id_classes = get_id_classes(args.id_split)
    hierarchy = Hierarchy(id_classes, args.hierarchy)
    ood_classes = hierarchy.ood_train_classes
    multi_classes = get_multidepth_classes(hierarchy, id_classes)

    full_tree = read_full_hierarchy(args.hierarchy)
    full_leaf_descendants = build_leaf_descendant_map(full_tree)
    internal_nodes_by_depth, available_child_names, id_leaf_sets, local_ood_leaf_sets = build_node_analysis_plan(
        hierarchy,
        id_classes,
        ood_classes,
        full_leaf_descendants,
    )

    val_logits_by_height, val_targets_by_height = load_predictions(
        args.basedir, hierarchy._max_depth, "val", device
    )
    ood_logits_by_height, ood_targets_by_height = load_predictions(
        args.basedir, hierarchy._max_depth, "ood", device
    )

    entropy_rows, comp_rows, entropy_child_rows, comp_child_rows, comp_complement_rows = compute_entropy_and_comp_rows(
        hierarchy,
        multi_classes,
        val_logits_by_height,
        val_targets_by_height,
        ood_logits_by_height,
        ood_targets_by_height,
        args.temperatures,
        internal_nodes_by_depth,
        available_child_names,
        id_leaf_sets,
        local_ood_leaf_sets,
        id_classes,
        ood_classes,
        device,
    )

    calibration_rows = compute_calibration_rows(
        val_logits_by_height,
        val_targets_by_height,
        args.temperatures,
        device,
    )

    summary_rows = summarize_best_temperatures(entropy_rows, comp_rows, calibration_rows)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    write_csv(os.path.join(output_dir, "entropy_by_depth.csv"), entropy_rows)
    write_csv(os.path.join(output_dir, "entropy_by_child_count.csv"), entropy_child_rows)
    write_csv(os.path.join(output_dir, "comp_prob_by_depth.csv"), comp_rows)
    write_csv(os.path.join(output_dir, "comp_prob_by_child_count.csv"), comp_child_rows)
    write_csv(os.path.join(output_dir, "comp_prob_by_complement_count.csv"), comp_complement_rows)
    write_csv(os.path.join(output_dir, "calibration_by_depth.csv"), calibration_rows)
    write_csv(os.path.join(output_dir, "best_temperature_summary.csv"), summary_rows)

    print(f"Saved analysis CSVs to {output_dir}")


if __name__ == "__main__":
    main()
