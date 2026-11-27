import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FixedLocator

from libs.hierarchy import Hierarchy
from libs.utils.hierarchy_utils import get_multidepth_classes
from analyze.temperature_sweep_analysis import (
    build_leaf_descendant_map,
    build_node_analysis_plan,
    classifier_depth_to_logits_index,
    compute_local_entropy_and_comp,
    get_id_classes,
    load_predictions,
    read_full_hierarchy,
)


DATASET_PRESETS = {
    "fgvc-aircraft": {
        "basedir": "ckpts/fgvc-aircraft",
        "id_split": "data/fgvc-aircraft-id-labels.csv",
        "hierarchy": "hierarchies/fgvc-aircraft.json",
    },
    "inat19": {
        "basedir": "ckpts/inat19",
        "id_split": "data/inat19-id-labels.csv",
        "hierarchy": "hierarchies/inat19.json",
    },
}


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_temp_vector(raw):
    values = json.loads(raw)
    if not isinstance(values, list) or not values:
        raise ValueError("temperature vector must be a non-empty JSON list")
    return [float(x) for x in values]


def resolve_temperature_from_vector(local_depth, temp_vector):
    if local_depth < 0 or local_depth >= len(temp_vector):
        raise ValueError(
            f"local_depth={local_depth} out of range for temperature vector length {len(temp_vector)}"
        )
    return float(temp_vector[local_depth])


def stats(values):
    if not values:
        return float("nan"), float("nan"), 0
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.mean().item()), float(tensor.std(unbiased=False).item()), int(tensor.numel())


def compute_rows(args, temp_vector):
    device = args.device
    id_classes = get_id_classes(args.id_split)
    hierarchy = Hierarchy(id_classes, args.hierarchy)
    ood_classes = hierarchy.ood_train_classes
    multi_classes = get_multidepth_classes(hierarchy, id_classes)

    expected_len = hierarchy._max_depth - 1
    if len(temp_vector) != expected_len:
        raise ValueError(
            f"temperature vector length {len(temp_vector)} does not match expected score depths {expected_len}"
        )

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

    id_class_names = [id_classes[int(x)] for x in val_targets_by_height[0].tolist()]
    ood_class_names = [ood_classes[int(x)] for x in ood_targets_by_height[0].tolist()]

    rows = []
    for depth in sorted(internal_nodes_by_depth.keys()):
        local_depth = depth - 1
        temperature = resolve_temperature_from_vector(local_depth, temp_vector)
        logits_index = classifier_depth_to_logits_index(hierarchy._max_depth, depth)
        probs_val = torch.softmax(val_logits_by_height[logits_index] / temperature, dim=-1)
        probs_ood = torch.softmax(ood_logits_by_height[logits_index] / temperature, dim=-1)
        classifier_classes = multi_classes[depth]

        entropy_id_values = []
        entropy_ood_values = []
        comp_id_values = []
        comp_ood_values = []
        score_id_values = []
        score_ood_values = []

        nodes_with_id = 0
        nodes_with_ood = 0

        for node_name in internal_nodes_by_depth[depth]:
            child_names = [
                child_name
                for child_name in available_child_names[node_name]
                if child_name in classifier_classes
            ]
            if len(child_names) <= 1:
                continue

            child_indices = [classifier_classes.index(child_name) for child_name in child_names]
            entropy_val, comp_val = compute_local_entropy_and_comp(probs_val, child_indices)
            entropy_ood, comp_ood = compute_local_entropy_and_comp(probs_ood, child_indices)
            score_val = entropy_val + comp_val
            score_ood = entropy_ood + comp_ood

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
                entropy_id_values.extend(entropy_val[id_mask].detach().cpu().tolist())
                comp_id_values.extend(comp_val[id_mask].detach().cpu().tolist())
                score_id_values.extend(score_val[id_mask].detach().cpu().tolist())

            if ood_mask.any():
                nodes_with_ood += 1
                entropy_ood_values.extend(entropy_ood[ood_mask].detach().cpu().tolist())
                comp_ood_values.extend(comp_ood[ood_mask].detach().cpu().tolist())
                score_ood_values.extend(score_ood[ood_mask].detach().cpu().tolist())

        mean_entropy_id, std_entropy_id, num_entropy_id = stats(entropy_id_values)
        mean_entropy_ood, std_entropy_ood, num_entropy_ood = stats(entropy_ood_values)
        mean_comp_id, std_comp_id, num_comp_id = stats(comp_id_values)
        mean_comp_ood, std_comp_ood, num_comp_ood = stats(comp_ood_values)
        mean_score_id, std_score_id, num_score_id = stats(score_id_values)
        mean_score_ood, std_score_ood, num_score_ood = stats(score_ood_values)

        rows.append({
            "depth": depth,
            "local_depth": local_depth,
            "temperature": temperature,
            "nodes_with_id": nodes_with_id,
            "nodes_with_ood": nodes_with_ood,
            "mean_entropy_id": mean_entropy_id,
            "std_entropy_id": std_entropy_id,
            "num_entropy_id": num_entropy_id,
            "mean_entropy_ood": mean_entropy_ood,
            "std_entropy_ood": std_entropy_ood,
            "num_entropy_ood": num_entropy_ood,
            "mean_comp_id": mean_comp_id,
            "std_comp_id": std_comp_id,
            "num_comp_id": num_comp_id,
            "mean_comp_ood": mean_comp_ood,
            "std_comp_ood": std_comp_ood,
            "num_comp_ood": num_comp_ood,
            "mean_score_id": mean_score_id,
            "std_score_id": std_score_id,
            "num_score_id": num_score_id,
            "mean_score_ood": mean_score_ood,
            "std_score_ood": std_score_ood,
            "num_score_ood": num_score_ood,
        })

    return rows


def plot_metric(rows, id_key, ood_key, y_label, output_path):
    depths = [row["depth"] for row in rows]
    id_values = [row[id_key] for row in rows]
    ood_values = [row[ood_key] for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(depths, id_values, marker="o", label="ID")
    ax.plot(depths, ood_values, marker="o", label="OOD")
    ax.set_xlabel("Local parent depth")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(depths))))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), required=True)
    parser.add_argument("--basedir", type=str, default=None)
    parser.add_argument("--id_split", type=str, default=None)
    parser.add_argument("--hierarchy", type=str, default=None)
    parser.add_argument("--temperature_vector", required=True, type=str)
    parser.add_argument("--profile_name", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    args.basedir = args.basedir or preset["basedir"]
    args.id_split = args.id_split or preset["id_split"]
    args.hierarchy = args.hierarchy or preset["hierarchy"]
    args.output_dir = args.output_dir or os.path.join(
        "results", "temperature_analysis", args.dataset, "custom_temperature_profiles", args.profile_name
    )

    temp_vector = parse_temp_vector(args.temperature_vector)
    rows = compute_rows(args, temp_vector)
    os.makedirs(args.output_dir, exist_ok=True)

    write_csv(os.path.join(args.output_dir, "metrics_by_depth.csv"), rows)
    with open(os.path.join(args.output_dir, "temperature_vector.json"), "w", encoding="utf-8") as f:
        json.dump(temp_vector, f, ensure_ascii=False, indent=2)

    plot_metric(
        rows,
        "mean_entropy_id",
        "mean_entropy_ood",
        "Mean entropy",
        os.path.join(args.output_dir, "entropy_id_ood_by_depth.png"),
    )
    plot_metric(
        rows,
        "mean_comp_id",
        "mean_comp_ood",
        "Mean complementary probability",
        os.path.join(args.output_dir, "comp_prob_id_ood_by_depth.png"),
    )
    plot_metric(
        rows,
        "mean_score_id",
        "mean_score_ood",
        "Mean OOD score",
        os.path.join(args.output_dir, "ood_score_id_ood_by_depth.png"),
    )
    print(f"Saved custom temperature profile plots to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
