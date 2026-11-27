import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


SCHEDULE_LABELS = {
    "beta_constant__temp_constant_1p0": "Baseline: T=1",
    "beta_constant__temp_constant_2p0": "Global: T=2",
    "beta_constant__temp_linear_1p5_0p5": "Depth-wise: T=1.5+0.5d",
}


def load_rows(input_csv, run):
    df = pd.read_csv(input_csv)
    df = df[
        (df["split"] == "val+ood")
        & (df["run"] == run)
        & (df["schedule_name"].isin(SCHEDULE_LABELS))
    ].copy()
    if df.empty:
        raise ValueError("No matching rows found")
    df["schedule_label"] = df["schedule_name"].map(SCHEDULE_LABELS)
    return df


def plot_absolute_recalls(df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    for schedule_name, group in df.groupby("schedule_name", sort=False):
        group = group.sort_values("depth")
        label = SCHEDULE_LABELS[schedule_name]
        axes[0].plot(group["depth"], group["tpr_ood_recall"], marker="o", label=label)
        axes[1].plot(group["depth"], group["tnr_id_recall"], marker="o", label=label)

    axes[0].set_title("OOD Recall by Depth")
    axes[0].set_ylabel("TPR: OOD as OOD")
    axes[1].set_title("ID Recall by Depth")
    axes[1].set_ylabel("TNR: ID as ID")

    for ax in axes:
        ax.set_xlabel("Local parent depth")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_tradeoff_vs_baseline(df, output_path):
    baseline = df[df["schedule_name"] == "beta_constant__temp_constant_1p0"][
        ["depth", "tpr_ood_recall", "tnr_id_recall"]
    ].rename(columns={
        "tpr_ood_recall": "baseline_ood_recall",
        "tnr_id_recall": "baseline_id_recall",
    })

    comp = df[df["schedule_name"] != "beta_constant__temp_constant_1p0"].merge(
        baseline,
        on="depth",
    )
    comp["ood_recall_gain"] = comp["tpr_ood_recall"] - comp["baseline_ood_recall"]
    comp["id_recall_cost"] = comp["baseline_id_recall"] - comp["tnr_id_recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for schedule_name, group in comp.groupby("schedule_name", sort=False):
        group = group.sort_values("depth")
        label = SCHEDULE_LABELS[schedule_name]
        axes[0].plot(group["depth"], group["ood_recall_gain"], marker="o", label=label)
        axes[1].plot(group["depth"], group["id_recall_cost"], marker="o", label=label)

    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_title("OOD Recall Gain vs Baseline")
    axes[0].set_ylabel("Delta OOD recall")
    axes[1].set_title("ID Recall Cost vs Baseline")
    axes[1].set_ylabel("Baseline ID recall - current ID recall")

    for ax in axes:
        ax.set_xlabel("Local parent depth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="results/schedule_experiments/fgvc-aircraft/scheduled_raw_full_local_ood_detection/local_ood_detection_by_depth.csv",
        type=str,
    )
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--run", default="scheduled_raw_argmax", type=str)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input_csv)
    os.makedirs(output_dir, exist_ok=True)

    df = load_rows(args.input_csv, args.run)
    df.to_csv(os.path.join(output_dir, "depthwise_temperature_tradeoff_rows.csv"), index=False)
    plot_absolute_recalls(
        df,
        os.path.join(output_dir, "depthwise_temperature_absolute_recalls.png"),
    )
    comp = plot_tradeoff_vs_baseline(
        df,
        os.path.join(output_dir, "depthwise_temperature_tradeoff_vs_baseline.png"),
    )
    comp.to_csv(os.path.join(output_dir, "depthwise_temperature_tradeoff_vs_baseline.csv"), index=False)
    print(f"Saved depth-wise temperature tradeoff plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
