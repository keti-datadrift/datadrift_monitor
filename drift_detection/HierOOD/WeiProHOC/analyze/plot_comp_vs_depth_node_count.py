import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


def count_full_internal_nodes_by_depth(hierarchy_path):
    root = json.load(open(hierarchy_path, encoding="utf-8"))
    counts = Counter()
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        children = node.get("children", [])
        if children:
            counts[depth] += 1
            for child in children:
                stack.append((child, depth + 1))
    return dict(sorted(counts.items()))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_t1(df, x_col, x_label, output_path):
    group = df[df["temperature"] == 1.0].sort_values(x_col)
    if group.empty:
        raise ValueError("No rows found for temperature=1.0")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group[x_col], group["mean_comp_id"], marker="o", label="ID")
    ax.plot(group[x_col], group["mean_comp_ood"], marker="o", label="OOD")
    ax.set_title(f"Complementary Probability by {x_label} at T=1")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean complementary probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gap(df, x_col, x_label, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for temp, group in df.groupby("temperature"):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group["comp_gap"], marker="o", label=f"T={temp:g}")
    ax.set_title(f"Complementary Probability ID/OOD Gap by {x_label}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("mean_ood - mean_id")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "plots")
    ensure_dir(output_dir)

    comp_df = pd.read_csv(os.path.join(args.input_dir, "comp_prob_by_depth.csv"))
    full_counts = count_full_internal_nodes_by_depth(args.hierarchy)
    comp_df["full_num_nodes"] = comp_df["depth"].map(full_counts)
    comp_df["pruned_num_nodes"] = comp_df["num_nodes"]

    csv_path = os.path.join(output_dir, "comp_prob_by_depth_with_node_counts.csv")
    comp_df.to_csv(csv_path, index=False)

    plot_t1(
        comp_df,
        "pruned_num_nodes",
        "Number of pruned internal nodes at depth",
        os.path.join(output_dir, "comp_prob_id_ood_by_pruned_depth_node_count_T1.png"),
    )
    plot_gap(
        comp_df,
        "pruned_num_nodes",
        "Number of pruned internal nodes at depth",
        os.path.join(output_dir, "comp_prob_gap_by_pruned_depth_node_count.png"),
    )

    if comp_df["full_num_nodes"].notna().all():
        plot_t1(
            comp_df,
            "full_num_nodes",
            "Number of full-hierarchy internal nodes at depth",
            os.path.join(output_dir, "comp_prob_id_ood_by_full_depth_node_count_T1.png"),
        )
        plot_gap(
            comp_df,
            "full_num_nodes",
            "Number of full-hierarchy internal nodes at depth",
            os.path.join(output_dir, "comp_prob_gap_by_full_depth_node_count.png"),
        )

    print(f"Saved node-count comp_prob plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
