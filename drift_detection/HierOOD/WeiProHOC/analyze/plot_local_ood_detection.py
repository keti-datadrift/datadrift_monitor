import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_tpr_tnr_temperature1(df, output_path):
    schedules = list(df["schedule_name"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    for schedule_name in schedules:
        group = df[df["schedule_name"] == schedule_name].sort_values("depth")
        axes[0].plot(group["depth"], group["tpr_ood_recall"], marker="o", label=schedule_name)
        axes[1].plot(group["depth"], group["tnr_id_recall"], marker="o", label=schedule_name)

    axes[0].set_title("OOD Recall by Local OOD Depth at T=1")
    axes[0].set_ylabel("TPR: OOD as OOD")
    axes[1].set_title("ID Recall by Local OOD Depth at T=1")
    axes[1].set_ylabel("TNR: ID as ID")

    for ax in axes:
        ax.set_xlabel("Local parent depth")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_baseline_id_ood_same_axes(df, output_path):
    group = df[df["schedule_name"] == "beta_constant__temp_constant_1p0"].sort_values("depth")
    if group.empty:
        raise ValueError("No baseline rows found for beta_constant__temp_constant_1p0")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(
        group["depth"],
        group["tpr_ood_recall"],
        marker="o",
        color="tab:red",
        label="OOD recall (OOD as OOD)",
    )
    ax.plot(
        group["depth"],
        group["tnr_id_recall"],
        marker="o",
        color="tab:blue",
        label="ID recall (ID as ID)",
    )
    ax.set_title("Baseline Local OOD Detection at T=1")
    ax.set_xlabel("Local parent depth")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.2, 0.3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_baseline_tpr_only(df, output_path):
    group = df[df["schedule_name"] == "beta_constant__temp_constant_1p0"].sort_values("depth")
    if group.empty:
        raise ValueError("No baseline rows found for beta_constant__temp_constant_1p0")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        group["depth"],
        group["tpr_ood_recall"],
        marker="o",
        color="tab:red",
        label="OOD recall",
    )
    ax.set_title("Baseline OOD Recall by Local OOD Depth at T=1")
    ax.set_xlabel("Local parent depth")
    ax.set_ylabel("TPR: OOD as OOD")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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

    df = pd.read_csv(args.input_csv)
    filtered = df[
        (df["split"] == "val+ood")
        & (df["run"] == args.run)
        & (df["temperature_schedule"] == "constant")
        & (df["temperature_t0"] == 1.0)
    ].copy()

    if filtered.empty:
        raise ValueError("No rows found for split=val+ood, run, and T=1 constant temperature")

    filtered.to_csv(os.path.join(output_dir, "local_ood_detection_T1_val_ood.csv"), index=False)
    baseline = filtered[filtered["schedule_name"] == "beta_constant__temp_constant_1p0"].copy()
    baseline.to_csv(
        os.path.join(output_dir, "local_ood_detection_T1_val_ood_baseline.csv"),
        index=False,
    )
    plot_tpr_tnr_temperature1(
        filtered,
        os.path.join(output_dir, "local_ood_detection_tpr_tnr_T1_val_ood.png"),
    )
    plot_baseline_id_ood_same_axes(
        filtered,
        os.path.join(output_dir, "local_ood_detection_tpr_tnr_T1_val_ood_baseline.png"),
    )
    plot_baseline_tpr_only(
        filtered,
        os.path.join(output_dir, "local_ood_detection_tpr_T1_val_ood_baseline.png"),
    )
    print(f"Saved local OOD detection plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
