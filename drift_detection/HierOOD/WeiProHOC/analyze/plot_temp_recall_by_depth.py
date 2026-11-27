import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_by_depth(df, output_path):
    depths = sorted(df["depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 4.8), sharey=True)
    if len(depths) == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        group = df[df["depth"] == depth].sort_values("temperature_t0")
        ax.plot(
            group["temperature_t0"],
            group["tnr_id_recall"],
            marker="o",
            color="tab:blue",
            label="ID recall",
        )
        ax.plot(
            group["temperature_t0"],
            group["tpr_ood_recall"],
            marker="o",
            color="tab:red",
            label="OOD recall",
        )
        ax.plot(
            group["temperature_t0"],
            group["balanced_acc"],
            marker="o",
            color="tab:green",
            label="Mixed balanced acc",
        )
        ax.set_title(f"Local OOD Depth {depth}")
        ax.set_xlabel("Constant softmax temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Recall / balanced accuracy")
    axes[0].set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_by_depth_lines(df, metric, output_path, ylabel):
    fig, ax = plt.subplots(figsize=(7, 4.8))

    for depth, group in df.groupby("depth"):
        group = group.sort_values("temperature_t0")
        ax.plot(
            group["temperature_t0"],
            group[metric],
            marker="o",
            label=f"depth {depth}",
        )

    ax.set_title(f"{ylabel} vs Temperature by Depth")
    ax.set_xlabel("Constant softmax temperature")
    ax.set_ylabel(ylabel)
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
        default="results/schedule_experiments/fgvc-aircraft/temp_constant_local_ood_detection/temp_constant_local_ood_detection_by_depth.csv",
        type=str,
    )
    parser.add_argument("--output_dir", default=None, type=str)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input_csv)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    df = df[(df["split"] == "val+ood") & (df["run"] == "scheduled_raw_argmax")].copy()
    df = df.sort_values(["depth", "temperature_t0"])
    df.to_csv(os.path.join(output_dir, "temp_constant_local_ood_detection_by_depth_plot_rows.csv"), index=False)

    plot_by_depth(
        df,
        os.path.join(output_dir, "temp_constant_recall_by_depth.png"),
    )
    plot_metric_by_depth_lines(
        df,
        "tpr_ood_recall",
        os.path.join(output_dir, "temp_constant_ood_recall_by_depth.png"),
        "OOD recall",
    )
    plot_metric_by_depth_lines(
        df,
        "tnr_id_recall",
        os.path.join(output_dir, "temp_constant_id_recall_by_depth.png"),
        "ID recall",
    )
    print(f"Saved temperature recall by depth plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
