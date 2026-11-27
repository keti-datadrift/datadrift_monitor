import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_gap_by_depth(entropy_df, comp_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    for temp, group in entropy_df.groupby("temperature"):
        group = group.sort_values("depth")
        axes[0].plot(group["depth"], group["entropy_gap"], marker="o", label=f"T={temp:g}")

    for temp, group in comp_df.groupby("temperature"):
        group = group.sort_values("depth")
        axes[1].plot(group["depth"], group["comp_gap"], marker="o", label=f"T={temp:g}")

    axes[0].set_title("Entropy ID/OOD Gap by Depth")
    axes[0].set_ylabel("mean_ood - mean_id")
    axes[1].set_title("Complementary Probability ID/OOD Gap by Depth")

    for ax in axes:
        ax.set_xlabel("Local parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_temperature(df, metric_name, output_path):
    depths = sorted(df["depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 4.5), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    id_col = f"mean_{metric_name}_id"
    ood_col = f"mean_{metric_name}_ood"
    y_label = "Mean entropy" if metric_name == "entropy" else "Mean complementary probability"

    for ax, depth in zip(axes, depths):
        group = df[df["depth"] == depth].sort_values("temperature")
        ax.plot(group["temperature"], group[id_col], marker="o", label="ID")
        ax.plot(group["temperature"], group[ood_col], marker="o", label="OOD")
        ax.set_title(f"{y_label} at Depth {depth}")
        ax.set_xlabel("Temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_depth(df, metric_name, output_path, selected_temperatures=None):
    if selected_temperatures is None:
        selected_temperatures = sorted(df["temperature"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    id_col = f"mean_{metric_name}_id"
    ood_col = f"mean_{metric_name}_ood"
    y_label = "Mean entropy" if metric_name == "entropy" else "Mean complementary probability"

    for temp in selected_temperatures:
        group = df[df["temperature"] == temp].sort_values("depth")
        if group.empty:
            continue
        axes[0].plot(group["depth"], group[id_col], marker="o", label=f"T={temp:g}")
        axes[1].plot(group["depth"], group[ood_col], marker="o", label=f"T={temp:g}")

    axes[0].set_title(f"{y_label} by Depth: ID")
    axes[1].set_title(f"{y_label} by Depth: OOD")

    for ax in axes:
        ax.set_xlabel("Local parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel(y_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_t1_metric_by_depth(df, metric_name, output_path, temperature=1.0):
    group = df[df["temperature"] == temperature].sort_values("depth")
    if group.empty:
        raise ValueError(f"No rows found for temperature={temperature}")

    id_col = f"mean_{metric_name}_id"
    ood_col = f"mean_{metric_name}_ood"
    y_label = "Mean entropy" if metric_name == "entropy" else "Mean complementary probability"

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(group["depth"], group[id_col], marker="o", label="ID")
    ax.plot(group["depth"], group[ood_col], marker="o", label="OOD")
    ax.set_title(f"{y_label} by Depth at T={temperature:g}")
    ax.set_xlabel("Local parent depth")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(FixedLocator(sorted(group["depth"].unique())))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_entropy_by_child_count(df, output_path, temperature=1.0):
    group = df[df["temperature"] == temperature].sort_values("num_children")
    if group.empty:
        raise ValueError(f"No child-count entropy rows found for temperature={temperature}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group["num_children"], group["mean_entropy_id"], marker="o", label="ID")
    ax.plot(group["num_children"], group["mean_entropy_ood"], marker="o", label="OOD")
    if "max_entropy_logK" in group.columns:
        ax.plot(
            group["num_children"],
            group["max_entropy_logK"],
            marker="o",
            linestyle="--",
            color="black",
            label="Max entropy log(K)",
        )
    ax.set_title(f"Entropy by Number of Children at T={temperature:g}")
    ax.set_xlabel("Number of children")
    ax.set_ylabel("Mean entropy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_entropy_gap_by_child_count(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for temp, group in df.groupby("temperature"):
        group = group.sort_values("num_children")
        ax.plot(group["num_children"], group["entropy_gap"], marker="o", label=f"T={temp:g}")

    ax.set_title("Entropy ID/OOD Gap by Number of Children")
    ax.set_xlabel("Number of children")
    ax.set_ylabel("mean_ood - mean_id")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comp_by_child_count(df, output_path, temperature=1.0):
    group = df[df["temperature"] == temperature].sort_values("num_children")
    if group.empty:
        raise ValueError(f"No child-count complementary probability rows found for temperature={temperature}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group["num_children"], group["mean_comp_id"], marker="o", label="ID")
    ax.plot(group["num_children"], group["mean_comp_ood"], marker="o", label="OOD")
    ax.set_title(f"Complementary Probability by Number of Children at T={temperature:g}")
    ax.set_xlabel("Number of children")
    ax.set_ylabel("Mean complementary probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comp_gap_by_child_count(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for temp, group in df.groupby("temperature"):
        group = group.sort_values("num_children")
        ax.plot(group["num_children"], group["comp_gap"], marker="o", label=f"T={temp:g}")

    ax.set_title("Complementary Probability ID/OOD Gap by Number of Children")
    ax.set_xlabel("Number of children")
    ax.set_ylabel("mean_ood - mean_id")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comp_by_complement_count(df, output_path, temperature=1.0):
    group = df[df["temperature"] == temperature].sort_values("num_complements")
    if group.empty:
        raise ValueError(f"No complementary-count rows found for temperature={temperature}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group["num_complements"], group["mean_comp_id"], marker="o", label="ID")
    ax.plot(group["num_complements"], group["mean_comp_ood"], marker="o", label="OOD")
    ax.set_title(f"Complementary Probability by Number of Complements at T={temperature:g}")
    ax.set_xlabel("Number of complementary leaves/classes")
    ax.set_ylabel("Mean complementary probability")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_comp_gap_by_complement_count(df, output_path):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for temp, group in df.groupby("temperature"):
        group = group.sort_values("num_complements")
        ax.plot(group["num_complements"], group["comp_gap"], marker="o", label=f"T={temp:g}")

    ax.set_title("Complementary Probability ID/OOD Gap by Number of Complements")
    ax.set_xlabel("Number of complementary leaves/classes")
    ax.set_ylabel("mean_ood - mean_id")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def weighted_mean(values, weights):
    weights = weights.clip(lower=0)
    total = weights.sum()
    if total == 0:
        return float(values.mean())
    return float((values * weights).sum() / total)


def aggregate_child_count_entropy_by_depth(df):
    rows = []
    for (depth, temperature), group in df.groupby(["depth", "temperature"]):
        rows.append({
            "depth": depth,
            "temperature": temperature,
            "num_child_count_bins": len(group),
            "num_id_samples": int(group["num_id_samples"].sum()),
            "num_ood_samples": int(group["num_ood_samples"].sum()),
            "mean_entropy_id": weighted_mean(group["mean_entropy_id"], group["num_id_samples"]),
            "mean_entropy_ood": weighted_mean(group["mean_entropy_ood"], group["num_ood_samples"]),
            "mean_norm_entropy_id": weighted_mean(group["mean_norm_entropy_id"], group["num_id_samples"]),
            "mean_norm_entropy_ood": weighted_mean(group["mean_norm_entropy_ood"], group["num_ood_samples"]),
        })
    out = pd.DataFrame(rows)
    out["entropy_gap"] = out["mean_entropy_ood"] - out["mean_entropy_id"]
    out["norm_entropy_gap"] = out["mean_norm_entropy_ood"] - out["mean_norm_entropy_id"]
    return out.sort_values(["temperature", "depth"])


def aggregate_child_count_entropy_by_num_children(df):
    rows = []
    for (num_children, temperature), group in df.groupby(["num_children", "temperature"]):
        rows.append({
            "num_children": num_children,
            "temperature": temperature,
            "num_depth_bins": len(group),
            "max_entropy_logK": float(group["max_entropy_logK"].iloc[0]),
            "num_id_samples": int(group["num_id_samples"].sum()),
            "num_ood_samples": int(group["num_ood_samples"].sum()),
            "mean_entropy_id": weighted_mean(group["mean_entropy_id"], group["num_id_samples"]),
            "mean_entropy_ood": weighted_mean(group["mean_entropy_ood"], group["num_ood_samples"]),
            "mean_norm_entropy_id": weighted_mean(group["mean_norm_entropy_id"], group["num_id_samples"]),
            "mean_norm_entropy_ood": weighted_mean(group["mean_norm_entropy_ood"], group["num_ood_samples"]),
        })
    out = pd.DataFrame(rows)
    out["entropy_gap"] = out["mean_entropy_ood"] - out["mean_entropy_id"]
    out["norm_entropy_gap"] = out["mean_norm_entropy_ood"] - out["mean_norm_entropy_id"]
    return out.sort_values(["temperature", "num_children"])


def aggregate_child_count_comp_by_num_children(df):
    rows = []
    for (num_children, temperature), group in df.groupby(["num_children", "temperature"]):
        rows.append({
            "num_children": num_children,
            "temperature": temperature,
            "num_depth_bins": len(group),
            "num_id_samples": int(group["num_id_samples"].sum()),
            "num_ood_samples": int(group["num_ood_samples"].sum()),
            "mean_comp_id": weighted_mean(group["mean_comp_id"], group["num_id_samples"]),
            "mean_comp_ood": weighted_mean(group["mean_comp_ood"], group["num_ood_samples"]),
        })
    out = pd.DataFrame(rows)
    out["comp_gap"] = out["mean_comp_ood"] - out["mean_comp_id"]
    return out.sort_values(["temperature", "num_children"])


def aggregate_complement_count_comp(df):
    rows = []
    for (num_complements, temperature), group in df.groupby(["num_complements", "temperature"]):
        rows.append({
            "num_complements": num_complements,
            "temperature": temperature,
            "num_depth_bins": len(group),
            "num_id_samples": int(group["num_id_samples"].sum()),
            "num_ood_samples": int(group["num_ood_samples"].sum()),
            "mean_comp_id": weighted_mean(group["mean_comp_id"], group["num_id_samples"]),
            "mean_comp_ood": weighted_mean(group["mean_comp_ood"], group["num_ood_samples"]),
        })
    out = pd.DataFrame(rows)
    out["comp_gap"] = out["mean_comp_ood"] - out["mean_comp_id"]
    return out.sort_values(["temperature", "num_complements"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="results/temperature_analysis/fgvc-aircraft/Tgrid_default",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Defaults to <input_dir>/plots.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "plots")
    ensure_dir(output_dir)

    entropy_df = pd.read_csv(os.path.join(input_dir, "entropy_by_depth.csv"))
    comp_df = pd.read_csv(os.path.join(input_dir, "comp_prob_by_depth.csv"))
    entropy_child_df = pd.read_csv(os.path.join(input_dir, "entropy_by_child_count.csv"))
    comp_child_df = pd.read_csv(os.path.join(input_dir, "comp_prob_by_child_count.csv"))
    comp_complement_df = pd.read_csv(os.path.join(input_dir, "comp_prob_by_complement_count.csv"))
    entropy_child_depth_df = aggregate_child_count_entropy_by_depth(entropy_child_df)
    entropy_child_depth_df.to_csv(
        os.path.join(output_dir, "entropy_by_depth_from_child_count_agg.csv"),
        index=False,
    )
    entropy_child_count_df = aggregate_child_count_entropy_by_num_children(entropy_child_df)
    entropy_child_count_df.to_csv(
        os.path.join(output_dir, "entropy_by_child_count_agg.csv"),
        index=False,
    )
    comp_child_count_df = aggregate_child_count_comp_by_num_children(comp_child_df)
    comp_child_count_df.to_csv(
        os.path.join(output_dir, "comp_prob_by_child_count_agg.csv"),
        index=False,
    )
    comp_complement_count_df = aggregate_complement_count_comp(comp_complement_df)
    comp_complement_count_df.to_csv(
        os.path.join(output_dir, "comp_prob_by_complement_count_agg.csv"),
        index=False,
    )

    plot_gap_by_depth(
        entropy_df,
        comp_df,
        os.path.join(output_dir, "gap_by_depth.png"),
    )
    plot_metric_by_temperature(
        entropy_df,
        "entropy",
        os.path.join(output_dir, "entropy_id_ood_by_temperature.png"),
    )
    plot_metric_by_temperature(
        comp_df,
        "comp",
        os.path.join(output_dir, "comp_prob_id_ood_by_temperature.png"),
    )
    plot_metric_by_depth(
        entropy_df,
        "entropy",
        os.path.join(output_dir, "entropy_id_ood_by_depth.png"),
    )
    plot_metric_by_depth(
        comp_df,
        "comp",
        os.path.join(output_dir, "comp_prob_id_ood_by_depth.png"),
    )
    plot_t1_metric_by_depth(
        entropy_df,
        "entropy",
        os.path.join(output_dir, "entropy_id_ood_by_depth_T1.png"),
    )
    plot_t1_metric_by_depth(
        comp_df,
        "comp",
        os.path.join(output_dir, "comp_prob_id_ood_by_depth_T1.png"),
    )
    plot_entropy_by_child_count(
        entropy_child_df,
        os.path.join(output_dir, "entropy_id_ood_by_child_count_T1.png"),
    )
    plot_entropy_gap_by_child_count(
        entropy_child_df,
        os.path.join(output_dir, "entropy_gap_by_child_count.png"),
    )
    plot_t1_metric_by_depth(
        entropy_child_depth_df,
        "entropy",
        os.path.join(output_dir, "entropy_id_ood_by_depth_T1_child_count_agg.png"),
    )
    plot_entropy_by_child_count(
        entropy_child_count_df,
        os.path.join(output_dir, "entropy_id_ood_by_child_count_T1_agg.png"),
    )
    plot_entropy_gap_by_child_count(
        entropy_child_count_df,
        os.path.join(output_dir, "entropy_gap_by_child_count_agg.png"),
    )
    plot_comp_by_child_count(
        comp_child_df,
        os.path.join(output_dir, "comp_prob_id_ood_by_child_count_T1.png"),
    )
    plot_comp_gap_by_child_count(
        comp_child_df,
        os.path.join(output_dir, "comp_prob_gap_by_child_count.png"),
    )
    plot_comp_by_child_count(
        comp_child_count_df,
        os.path.join(output_dir, "comp_prob_id_ood_by_child_count_T1_agg.png"),
    )
    plot_comp_gap_by_child_count(
        comp_child_count_df,
        os.path.join(output_dir, "comp_prob_gap_by_child_count_agg.png"),
    )
    plot_comp_by_complement_count(
        comp_complement_df,
        os.path.join(output_dir, "comp_prob_id_ood_by_complement_count_T1.png"),
    )
    plot_comp_gap_by_complement_count(
        comp_complement_df,
        os.path.join(output_dir, "comp_prob_gap_by_complement_count.png"),
    )
    plot_comp_by_complement_count(
        comp_complement_count_df,
        os.path.join(output_dir, "comp_prob_id_ood_by_complement_count_T1_agg.png"),
    )
    plot_comp_gap_by_complement_count(
        comp_complement_count_df,
        os.path.join(output_dir, "comp_prob_gap_by_complement_count_agg.png"),
    )

    print(f"Saved plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
