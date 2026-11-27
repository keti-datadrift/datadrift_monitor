import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from libs.hierarchy import Hierarchy
from libs.utils.hierarchy_utils import get_multidepth_classes
from analyze.temperature_sweep_analysis import (
    EPS,
    build_leaf_descendant_map,
    build_node_analysis_plan,
    classifier_depth_to_logits_index,
    compute_local_entropy_and_comp,
    get_id_classes,
    load_predictions,
    read_full_hierarchy,
)


SCHEDULES = [
    {
        "name": "lambda_constant__temp_constant_1p0",
        "lambda_schedule": "constant",
        "temperature_schedule": "constant",
        "temperature": 1.0,
    },
    {
        "name": "lambda_linear_decay_0p5__temp_constant_1p0",
        "lambda_schedule": "linear_decay",
        "temperature_schedule": "constant",
        "temperature": 1.0,
    },
    {
        "name": "lambda_linear_decay_0p5__temp_linear_1p0_0p5",
        "lambda_schedule": "linear_decay",
        "temperature_schedule": "linear_increase",
        "temperature_t0": 1.0,
        "temperature_k": 0.5,
    },
]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_lambda(local_depth, schedule):
    if schedule["lambda_schedule"] == "constant":
        return 1.0
    if schedule["lambda_schedule"] == "linear_decay":
        return max(1.0 - 0.5 * local_depth, 0.0)
    raise ValueError(f"Unknown lambda schedule: {schedule['lambda_schedule']}")


def resolve_temperature(local_depth, schedule):
    if schedule["temperature_schedule"] == "constant":
        return float(schedule["temperature"])
    if schedule["temperature_schedule"] == "linear_increase":
        return float(schedule.get("temperature_t0", 1.5)) + float(schedule.get("temperature_k", 0.5)) * local_depth
    raise ValueError(f"Unknown temperature schedule: {schedule['temperature_schedule']}")


def stats(values):
    if not values:
        return {
            "count": 0,
            "mean_score": float("nan"),
            "mean_max_child_prob": float("nan"),
            "mean_margin": float("nan"),
            "ood_selected_rate": float("nan"),
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    score = tensor[:, 0]
    max_child = tensor[:, 1]
    margin = score - max_child
    return {
        "count": int(tensor.size(0)),
        "mean_score": float(score.mean().item()),
        "mean_max_child_prob": float(max_child.mean().item()),
        "mean_margin": float(margin.mean().item()),
        "ood_selected_rate": float((margin > 0).float().mean().item()),
    }


def binary_detection_stats(id_values, ood_values):
    if not id_values or not ood_values:
        return {
            "num_id": len(id_values),
            "num_ood": len(ood_values),
            "auroc": float("nan"),
            "aupr": float("nan"),
            "balanced_acc_at_zero": float("nan"),
            "id_correct_rate_at_zero": float("nan"),
            "ood_correct_rate_at_zero": float("nan"),
            "fpr_at_zero": float("nan"),
            "tpr_at_zero": float("nan"),
        }

    id_tensor = torch.tensor(id_values, dtype=torch.float32)
    ood_tensor = torch.tensor(ood_values, dtype=torch.float32)
    id_margin = id_tensor[:, 0] - id_tensor[:, 1]
    ood_margin = ood_tensor[:, 0] - ood_tensor[:, 1]

    y_true = torch.cat([
        torch.zeros_like(id_margin, dtype=torch.long),
        torch.ones_like(ood_margin, dtype=torch.long),
    ])
    y_score = torch.cat([id_margin, ood_margin])
    y_pred = (y_score > 0).long()

    y_true_np = y_true.numpy()
    y_score_np = y_score.numpy()
    y_pred_np = y_pred.numpy()

    id_correct = float((id_margin <= 0).float().mean().item())
    ood_correct = float((ood_margin > 0).float().mean().item())

    return {
        "num_id": int(id_margin.numel()),
        "num_ood": int(ood_margin.numel()),
        "auroc": float(roc_auc_score(y_true_np, y_score_np)),
        "aupr": float(average_precision_score(y_true_np, y_score_np)),
        "balanced_acc_at_zero": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "id_correct_rate_at_zero": id_correct,
        "ood_correct_rate_at_zero": ood_correct,
        "fpr_at_zero": 1.0 - id_correct,
        "tpr_at_zero": ood_correct,
    }


def make_schedules(temp_t0):
    return [
        {
            "name": f"lambda_constant__temp_constant_{str(temp_t0).replace('.', 'p')}",
            "lambda_schedule": "constant",
            "temperature_schedule": "constant",
            "temperature": float(temp_t0),
        },
        {
            "name": f"lambda_linear_decay_0p5__temp_constant_{str(temp_t0).replace('.', 'p')}",
            "lambda_schedule": "linear_decay",
            "temperature_schedule": "constant",
            "temperature": float(temp_t0),
        },
        {
            "name": f"lambda_constant__temp_linear_{str(temp_t0).replace('.', 'p')}_0p5",
            "lambda_schedule": "constant",
            "temperature_schedule": "linear_increase",
            "temperature_t0": float(temp_t0),
            "temperature_k": 0.5,
        },
        {
            "name": f"lambda_linear_decay_0p5__temp_linear_{str(temp_t0).replace('.', 'p')}_0p5",
            "lambda_schedule": "linear_decay",
            "temperature_schedule": "linear_increase",
            "temperature_t0": float(temp_t0),
            "temperature_k": 0.5,
        },
    ]


def compute_rows(args, schedules):
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

    id_class_names = [id_classes[int(x)] for x in val_targets_by_height[0].tolist()]
    ood_class_names = [ood_classes[int(x)] for x in ood_targets_by_height[0].tolist()]

    rows = []
    binary_rows = []
    loop_items = [(schedule, depth) for schedule in schedules for depth in sorted(internal_nodes_by_depth.keys())]

    for schedule, depth in tqdm(loop_items, desc="Score vs child max", unit="setting"):
        local_depth = depth - 1
        lambda_value = resolve_lambda(local_depth, schedule)
        temperature = resolve_temperature(local_depth, schedule)
        logits_index = classifier_depth_to_logits_index(hierarchy._max_depth, depth)
        classifier_classes = multi_classes[depth]

        probs_val = torch.softmax(val_logits_by_height[logits_index] / temperature, dim=-1)
        probs_ood = torch.softmax(ood_logits_by_height[logits_index] / temperature, dim=-1)

        values_by_split = defaultdict(list)

        for node_name in internal_nodes_by_depth[depth]:
            child_names = [
                child_name
                for child_name in available_child_names[node_name]
                if child_name in classifier_classes
            ]
            if len(child_names) <= 1:
                continue

            child_indices = [classifier_classes.index(child_name) for child_name in child_names]
            child_indices_tensor = torch.tensor(child_indices, dtype=torch.long, device=device)

            entropy_val, comp_val = compute_local_entropy_and_comp(probs_val, child_indices)
            entropy_ood, comp_ood = compute_local_entropy_and_comp(probs_ood, child_indices)

            score_val = entropy_val + lambda_value * comp_val
            score_ood = entropy_ood + lambda_value * comp_ood
            max_child_val = probs_val.index_select(1, child_indices_tensor).max(dim=1).values
            max_child_ood = probs_ood.index_select(1, child_indices_tensor).max(dim=1).values

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
                values_by_split["ID"].extend(
                    torch.stack([score_val[id_mask], max_child_val[id_mask]], dim=1).detach().cpu().tolist()
                )
            if ood_mask.any():
                values_by_split["OOD"].extend(
                    torch.stack([score_ood[ood_mask], max_child_ood[ood_mask]], dim=1).detach().cpu().tolist()
                )

        for split in ("ID", "OOD"):
            summary = stats(values_by_split[split])
            rows.append({
                "schedule_name": schedule["name"],
                "split": split,
                "hierarchy_depth": depth,
                "local_depth": local_depth,
                "temperature": temperature,
                "lambda": lambda_value,
                **summary,
            })

        binary_rows.append({
            "schedule_name": schedule["name"],
            "hierarchy_depth": depth,
            "local_depth": local_depth,
            "temperature": temperature,
            "lambda": lambda_value,
            **binary_detection_stats(values_by_split["ID"], values_by_split["OOD"]),
        })

    return rows, binary_rows


def plot_score_vs_child_max(df, output_path):
    schedules = list(df["schedule_name"].unique())
    fig, axes = plt.subplots(1, len(schedules), figsize=(6 * len(schedules), 4.5), sharey=True)
    if len(schedules) == 1:
        axes = [axes]

    for ax, schedule_name in zip(axes, schedules):
        group = df[df["schedule_name"] == schedule_name].sort_values(["split", "hierarchy_depth"])
        split_colors = {
            "ID": "tab:blue",
            "OOD": "tab:red",
        }
        for split, split_group in group.groupby("split"):
            marker = "o" if split == "ID" else "s"
            color = split_colors.get(split)
            ax.plot(
                split_group["hierarchy_depth"],
                split_group["mean_score"],
                marker=marker,
                linestyle="-",
                color=color,
                label=f"{split} OOD score",
            )
            ax.plot(
                split_group["hierarchy_depth"],
                split_group["mean_max_child_prob"],
                marker=marker,
                linestyle=":",
                color=color,
                label=f"{split} max child prob",
            )
        ax.set_title(schedule_name)
        ax.set_xlabel("Hierarchy parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Mean value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_score_vs_child_max_single_schedule_flat(df, schedule_name, output_path):
    group = df[df["schedule_name"] == schedule_name].sort_values(["split", "hierarchy_depth"])
    if group.empty:
        raise ValueError(f"No rows found for schedule: {schedule_name}")

    fig, ax = plt.subplots(figsize=(8.2, 3.5))
    split_colors = {
        "ID": "tab:blue",
        "OOD": "tab:red",
    }
    labels = {
        ("ID", "mean_score"): "ID OOD score",
        ("ID", "mean_max_child_prob"): "ID max child prob",
        ("OOD", "mean_score"): "OOD OOD score",
        ("OOD", "mean_max_child_prob"): "OOD max child prob",
    }

    for split, split_group in group.groupby("split"):
        split_group = split_group.sort_values("hierarchy_depth")
        for column in ["mean_score", "mean_max_child_prob"]:
            linestyle = "-" if column == "mean_score" else (0, (2.2, 1.6))
            ax.plot(
                split_group["hierarchy_depth"],
                split_group[column],
                marker="o",
                linestyle=linestyle,
                color=split_colors.get(split),
                linewidth=1.8,
                markersize=5,
                label=labels[(split, column)],
            )

    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean value")
    ax.xaxis.set_major_locator(FixedLocator(sorted(group["hierarchy_depth"].unique())))
    ax.set_ylim(0.0, 1.2)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.xaxis.label.set_size(12.5)
    ax.yaxis.label.set_size(12.5)
    ax.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linestyle="-", linewidth=1.8, marker="o", markersize=5, label="ID OOD score"),
        Line2D([0], [0], color="tab:blue", linestyle=(0, (2.2, 1.6)), linewidth=1.8, marker="o", markersize=5, label="ID max child prob"),
        Line2D([0], [0], color="tab:red", linestyle="-", linewidth=1.8, marker="o", markersize=5, label="OOD OOD score"),
        Line2D([0], [0], color="tab:red", linestyle=(0, (2.2, 1.6)), linewidth=1.8, marker="o", markersize=5, label="OOD max child prob"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=10.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.31),
        handlelength=2.8,
        columnspacing=1.4,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_vs_child_max_for_split(df, split, output_path):
    split_df = df[df["split"] == split]
    schedules = list(split_df["schedule_name"].unique())
    fig, axes = plt.subplots(1, len(schedules), figsize=(6 * len(schedules), 4.5), sharey=True)
    if len(schedules) == 1:
        axes = [axes]

    color = "tab:blue" if split == "ID" else "tab:red"
    for ax, schedule_name in zip(axes, schedules):
        group = split_df[split_df["schedule_name"] == schedule_name].sort_values("hierarchy_depth")
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score"],
            marker="o",
            linestyle="-",
            color=color,
            label="OOD score",
        )
        ax.plot(
            group["hierarchy_depth"],
            group["mean_max_child_prob"],
            marker="o",
            linestyle=":",
            color=color,
            label="Max child probability",
        )
        ax.set_title(schedule_name)
        ax.set_xlabel("Hierarchy parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f"{split} mean value")
    fig.suptitle(f"{split}: OOD Score vs Max Child Probability", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_margin(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant__temp_constant_1p0": "black",
        "lambda_linear_decay_0p5__temp_constant_1p0": "tab:orange",
        "lambda_linear_decay_0p5__temp_linear_1p0_0p5": "tab:green",
    }

    for (schedule_name, split), group in df.groupby(["schedule_name", "split"]):
        group = group.sort_values("hierarchy_depth")
        marker = "o" if split == "ID" else "s"
        linestyle = "-" if split == "ID" else "--"
        ax.plot(
            group["hierarchy_depth"],
            group["mean_margin"],
            marker=marker,
            linestyle=linestyle,
            color=colors.get(schedule_name),
            label=f"{schedule_name} {split}",
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("OOD Score Margin Over Max Child Probability")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("mean(score - max_child_prob)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ood_selected_rate(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant__temp_constant_1p0": "black",
        "lambda_linear_decay_0p5__temp_constant_1p0": "tab:orange",
        "lambda_linear_decay_0p5__temp_linear_1p0_0p5": "tab:green",
    }

    for (schedule_name, split), group in df.groupby(["schedule_name", "split"]):
        group = group.sort_values("hierarchy_depth")
        marker = "o" if split == "ID" else "s"
        linestyle = "-" if split == "ID" else "--"
        ax.plot(
            group["hierarchy_depth"],
            group["ood_selected_rate"],
            marker=marker,
            linestyle=linestyle,
            color=colors.get(schedule_name),
            label=f"{schedule_name} {split}",
        )

    ax.set_title("Fraction Where OOD Score Exceeds Max Child Probability")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("P(score > max_child_prob)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_binary_detection_by_depth(df, output_path):
    schedules = list(df["schedule_name"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for schedule_name, group in df.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        axes[0].plot(
            group["hierarchy_depth"],
            group["auroc"],
            marker="o",
            label=schedule_name,
        )
        axes[1].plot(
            group["hierarchy_depth"],
            group["balanced_acc_at_zero"],
            marker="o",
            label=schedule_name,
        )

    axes[0].set_title("ID/OOD Detection AUROC by Depth")
    axes[0].set_ylabel("AUROC")
    axes[1].set_title("ID/OOD Detection Balanced Acc at score > max child")
    axes[1].set_ylabel("Balanced accuracy")

    for ax in axes:
        ax.set_xlabel("Hierarchy parent depth")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_threshold_rates_by_depth(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant__temp_constant_1p0": "black",
        "lambda_linear_decay_0p5__temp_constant_1p0": "tab:orange",
        "lambda_linear_decay_0p5__temp_linear_1p0_0p5": "tab:green",
    }

    for schedule_name, group in df.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        color = colors.get(schedule_name)
        ax.plot(
            group["hierarchy_depth"],
            group["id_correct_rate_at_zero"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{schedule_name} ID correct",
        )
        ax.plot(
            group["hierarchy_depth"],
            group["ood_correct_rate_at_zero"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{schedule_name} OOD correct",
        )

    ax.set_title("Correct ID/OOD Decisions at score > max child")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_max_child_prob_id_ood(df, output_path):
    schedules = list(df["schedule_name"].unique())
    fig, axes = plt.subplots(1, len(schedules), figsize=(6 * len(schedules), 4.5), sharey=True)
    if len(schedules) == 1:
        axes = [axes]

    split_colors = {
        "ID": "tab:blue",
        "OOD": "tab:red",
    }

    for ax, schedule_name in zip(axes, schedules):
        group = df[df["schedule_name"] == schedule_name].sort_values(["split", "hierarchy_depth"])
        for split, split_group in group.groupby("split"):
            ax.plot(
                split_group["hierarchy_depth"],
                split_group["mean_max_child_prob"],
                marker="o",
                color=split_colors.get(split),
                label=split,
            )
        ax.set_title(schedule_name)
        ax.set_xlabel("Hierarchy parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Mean max child probability")
    fig.suptitle("Maximum Child Probability by Depth", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument(
        "--output_dir",
        default="results/temperature_analysis/fgvc-aircraft/Tgrid_default/plots",
        type=str,
    )
    parser.add_argument("--temp_t0", default=1.0, type=float)
    parser.add_argument("--output_suffix", default="", type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    schedules = make_schedules(args.temp_t0)
    rows, binary_rows = compute_rows(args, schedules)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    csv_path = os.path.join(output_dir, f"ood_score_vs_child_max{suffix}.csv")
    write_csv(csv_path, rows)
    write_csv(os.path.join(output_dir, f"ood_detection_binary_by_depth{suffix}.csv"), binary_rows)

    df = pd.DataFrame(rows)
    binary_df = pd.DataFrame(binary_rows)
    plot_score_vs_child_max(df, os.path.join(output_dir, f"ood_score_vs_child_max_by_depth{suffix}.png"))
    plot_score_vs_child_max_single_schedule_flat(
        df,
        schedules[0]["name"],
        os.path.join(output_dir, f"ood_score_vs_child_max_baseline_flat{suffix}.png"),
    )
    plot_score_vs_child_max_for_split(
        df,
        "ID",
        os.path.join(output_dir, f"id_score_vs_child_max_by_depth{suffix}.png"),
    )
    plot_score_vs_child_max_for_split(
        df,
        "OOD",
        os.path.join(output_dir, f"ood_only_score_vs_child_max_by_depth{suffix}.png"),
    )
    plot_margin(df, os.path.join(output_dir, f"ood_score_margin_vs_child_max{suffix}.png"))
    plot_ood_selected_rate(df, os.path.join(output_dir, f"ood_score_exceeds_child_max_rate{suffix}.png"))
    plot_max_child_prob_id_ood(df, os.path.join(output_dir, f"max_child_prob_id_ood_by_depth{suffix}.png"))
    plot_binary_detection_by_depth(
        binary_df,
        os.path.join(output_dir, f"ood_detection_binary_metrics_by_depth{suffix}.png"),
    )
    plot_threshold_rates_by_depth(
        binary_df,
        os.path.join(output_dir, f"ood_detection_threshold_rates_by_depth{suffix}.png"),
    )
    print(f"Saved OOD score vs child max diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
