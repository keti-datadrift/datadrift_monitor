import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


BETA_SCHEDULES = [
    {
        "name": "lambda_linear_decay_0p5",
        "beta_schedule": "linear_decay",
        "beta0": 1.0,
        "beta_k": 0.5,
        "beta_min": 0.0,
    },
]


def resolve_beta(local_depth, schedule):
    beta_schedule = schedule["beta_schedule"]
    beta0 = float(schedule.get("beta0", 1.0))
    d = local_depth + 1

    if beta_schedule == "constant":
        return beta0
    if beta_schedule == "inverse_depth":
        return beta0 / d
    if beta_schedule == "exp_decay":
        return beta0 * (float(schedule.get("beta_gamma", 0.5)) ** (d - 1))
    if beta_schedule == "linear_decay":
        value = beta0 - float(schedule.get("beta_k", 0.5)) * (d - 1)
        return max(value, float(schedule.get("beta_min", 0.0)))
    raise ValueError(f"Unknown beta schedule: {beta_schedule}")


def load_depth_stats(input_dir, temperature):
    entropy = pd.read_csv(os.path.join(input_dir, "entropy_by_depth.csv"))
    comp = pd.read_csv(os.path.join(input_dir, "comp_prob_by_depth.csv"))

    entropy = entropy[entropy["temperature"] == temperature]
    comp = comp[comp["temperature"] == temperature]
    merged = entropy.merge(
        comp,
        on=["depth", "temperature", "num_nodes", "nodes_with_id", "nodes_with_ood"],
        suffixes=("", "_comp"),
    )
    return merged.sort_values("depth")


def load_all_depth_stats(input_dir):
    entropy = pd.read_csv(os.path.join(input_dir, "entropy_by_depth.csv"))
    comp = pd.read_csv(os.path.join(input_dir, "comp_prob_by_depth.csv"))
    merged = entropy.merge(
        comp,
        on=["depth", "temperature", "num_nodes", "nodes_with_id", "nodes_with_ood"],
        suffixes=("", "_comp"),
    )
    return merged.sort_values(["depth", "temperature"])


def resolve_temperature(local_depth, schedule_name, t0=1.5, k=0.5, r=1.25):
    if schedule_name == "constant_1p0":
        return 1.0
    if schedule_name == "linear_increase":
        return float(t0) + float(k) * local_depth
    if schedule_name == "exp_increase":
        return float(t0) * (float(r) ** local_depth)
    raise ValueError(f"Unknown temperature schedule: {schedule_name}")


def build_temperature_increase_rows(all_depth_stats):
    rows = []
    comparison_schedules = [
        {
            "name": "lambda_constant__temp_constant_1p0",
            "lambda_schedule": "constant",
            "temperature_schedule": "constant_1p0",
        },
        {
            "name": "lambda_linear_decay_0p5__temp_constant_1p0",
            "lambda_schedule": "linear_decay",
            "temperature_schedule": "constant_1p0",
        },
        {
            "name": "lambda_linear_decay_0p5__temp_linear_1p5_0p5",
            "lambda_schedule": "linear_decay",
            "temperature_schedule": "linear_increase",
            "temperature_t0": 1.5,
            "temperature_k": 0.5,
        },
    ]

    for hierarchy_depth in sorted(all_depth_stats["depth"].unique()):
        local_depth = int(hierarchy_depth) - 1
        for schedule in comparison_schedules:
            temperature = resolve_temperature(
                local_depth,
                schedule["temperature_schedule"],
                t0=schedule.get("temperature_t0", 1.5),
                k=schedule.get("temperature_k", 0.5),
            )
            row = all_depth_stats[
                (all_depth_stats["depth"] == hierarchy_depth)
                & (all_depth_stats["temperature"] == temperature)
            ]
            if row.empty:
                raise ValueError(
                    f"No rows for depth={hierarchy_depth}, temperature={temperature}. "
                    "Check the temperature grid."
                )
            row = row.iloc[0]

            if schedule["lambda_schedule"] == "constant":
                lambda_value = 1.0
            else:
                lambda_value = resolve_beta(local_depth, {
                    "beta_schedule": "linear_decay",
                    "beta0": 1.0,
                    "beta_k": 0.5,
                    "beta_min": 0.0,
                })

            score_id = row["mean_entropy_id"] + lambda_value * row["mean_comp_id"]
            score_ood = row["mean_entropy_ood"] + lambda_value * row["mean_comp_ood"]
            rows.append({
                "schedule_name": schedule["name"],
                "hierarchy_depth": int(hierarchy_depth),
                "local_depth": local_depth,
                "temperature": temperature,
                "lambda": lambda_value,
                "mean_entropy_id": row["mean_entropy_id"],
                "mean_entropy_ood": row["mean_entropy_ood"],
                "mean_comp_id": row["mean_comp_id"],
                "mean_comp_ood": row["mean_comp_ood"],
                "mean_score_id": score_id,
                "mean_score_ood": score_ood,
                "score_gap": score_ood - score_id,
            })
    return pd.DataFrame(rows)


def build_score_rows(depth_stats, include_constant=False):
    rows = []
    for _, row in depth_stats.iterrows():
        hierarchy_depth = int(row["depth"])
        local_depth = hierarchy_depth - 1
        schedules = [
            {
                "name": "lambda_constant",
                "beta_schedule": "constant",
                "beta0": 1.0,
            },
            *BETA_SCHEDULES,
        ]
        for schedule in schedules:
            beta = resolve_beta(local_depth, schedule)
            score_id = row["mean_entropy_id"] + beta * row["mean_comp_id"]
            score_ood = row["mean_entropy_ood"] + beta * row["mean_comp_ood"]
            rows.append({
                "schedule_name": schedule["name"],
                "lambda_schedule": schedule["beta_schedule"],
                "hierarchy_depth": hierarchy_depth,
                "local_depth": local_depth,
                "temperature": row["temperature"],
                "lambda": beta,
                "mean_entropy_id": row["mean_entropy_id"],
                "mean_entropy_ood": row["mean_entropy_ood"],
                "mean_comp_id": row["mean_comp_id"],
                "mean_comp_ood": row["mean_comp_ood"],
                "mean_score_id": score_id,
                "mean_score_ood": score_ood,
                "score_gap": score_ood - score_id,
            })

    out = pd.DataFrame(rows)
    baseline = out[out["schedule_name"] == "lambda_constant"][
        ["hierarchy_depth", "mean_score_id", "mean_score_ood"]
    ].rename(columns={
        "mean_score_id": "constant_score_id",
        "mean_score_ood": "constant_score_ood",
    })
    out = out.merge(baseline, on="hierarchy_depth")
    out["ood_score_drop_vs_constant"] = out["constant_score_ood"] - out["mean_score_ood"]
    out["id_score_drop_vs_constant"] = out["constant_score_id"] - out["mean_score_id"]
    if include_constant:
        return out.copy()
    return out[out["schedule_name"] != "lambda_constant"].copy()


def plot_ood_score(rows, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        ax.plot(group["hierarchy_depth"], group["mean_score_ood"], marker="o", label=schedule_name)

    ax.set_title("Mean OOD Score Under Lambda Decay Schedules")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean OOD score: H + lambda(d) * p_comp")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_id_score(rows, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        ax.plot(group["hierarchy_depth"], group["mean_score_id"], marker="o", label=schedule_name)

    ax.set_title("Mean ID Score Under Lambda Decay Schedules")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean ID score: H + lambda(d) * p_comp")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_id_ood_scores_together(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant": "black",
        "lambda_linear_decay_0p5": "tab:orange",
    }

    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        color = colors.get(schedule_name)
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score_id"],
            marker="o",
            linestyle="-",
            color=color,
            linewidth=2.2 if schedule_name == "lambda_constant" else 1.5,
            label=f"{schedule_name} ID",
        )
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score_ood"],
            marker="s",
            linestyle="--",
            color=color,
            linewidth=2.2 if schedule_name == "lambda_constant" else 1.5,
            label=f"{schedule_name} OOD",
        )

    ax.set_title("Mean ID/OOD Score Under Lambda Decay Schedules")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean score: H + lambda(d) * p_comp")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temperature_increase_ood_score(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant__temp_constant_1p0": "black",
        "lambda_linear_decay_0p5__temp_constant_1p0": "tab:orange",
        "lambda_linear_decay_0p5__temp_linear_1p5_0p5": "tab:green",
    }

    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score_ood"],
            marker="o",
            color=colors.get(schedule_name),
            linewidth=2.0 if "temp_linear" in schedule_name else 1.5,
            label=schedule_name,
        )

    ax.set_title("Mean OOD Score With Lambda Decay and Temperature Increase")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean OOD score: H_T + lambda(d) * p_comp_T")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_temperature_increase_id_ood_score(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_constant__temp_constant_1p0": "black",
        "lambda_linear_decay_0p5__temp_constant_1p0": "tab:orange",
        "lambda_linear_decay_0p5__temp_linear_1p5_0p5": "tab:green",
    }

    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        color = colors.get(schedule_name)
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score_id"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{schedule_name} ID",
        )
        ax.plot(
            group["hierarchy_depth"],
            group["mean_score_ood"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{schedule_name} OOD",
        )

    ax.set_title("ID/OOD Score With Lambda Decay and Temperature Increase")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean score: H_T + lambda(d) * p_comp_T")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_id_ood_score_drops_together(rows, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "lambda_linear_decay_0p5": "tab:orange",
    }

    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        color = colors.get(schedule_name)
        ax.plot(
            group["hierarchy_depth"],
            group["id_score_drop_vs_constant"],
            marker="o",
            linestyle="-",
            color=color,
            label=f"{schedule_name} ID",
        )
        ax.plot(
            group["hierarchy_depth"],
            group["ood_score_drop_vs_constant"],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{schedule_name} OOD",
        )

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("ID/OOD Score Drop Relative to Constant Lambda")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("constant score - decayed score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_score_drop(rows, output_path, split):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    drop_col = f"{split}_score_drop_vs_constant"
    for schedule_name, group in rows.groupby("schedule_name"):
        group = group.sort_values("hierarchy_depth")
        ax.plot(group["hierarchy_depth"], group[drop_col], marker="o", label=schedule_name)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_title(f"{split.upper()} Score Drop Relative to Constant Lambda")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("constant score - decayed score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_id_ood_score(rows, output_path, schedule_name):
    group = rows[rows["schedule_name"] == schedule_name].sort_values("hierarchy_depth")
    if group.empty:
        raise ValueError(f"No rows for schedule: {schedule_name}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group["hierarchy_depth"], group["mean_score_id"], marker="o", label="ID")
    ax.plot(group["hierarchy_depth"], group["mean_score_ood"], marker="o", label="OOD")
    ax.set_title(f"ID/OOD Score by Depth: {schedule_name}")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean score: H + lambda(d) * p_comp")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_baseline_id_ood_score(rows_with_constant, output_path):
    group = rows_with_constant[
        rows_with_constant["schedule_name"] == "lambda_constant"
    ].sort_values("hierarchy_depth")
    if group.empty:
        raise ValueError("No lambda_constant rows found")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(group["hierarchy_depth"], group["mean_score_id"], marker="o", color="tab:blue", label="ID")
    ax.plot(group["hierarchy_depth"], group["mean_score_ood"], marker="o", color="tab:red", label="OOD")
    ax.set_title("ID/OOD Score by Depth: No Decay, T=1")
    ax.set_xlabel("Hierarchy parent depth")
    ax.set_ylabel("Mean score: H + p_comp")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="results/temperature_analysis/fgvc-aircraft/Tgrid_default",
        type=str,
    )
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--output_dir", default=None, type=str)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    depth_stats = load_depth_stats(args.input_dir, args.temperature)
    all_depth_stats = load_all_depth_stats(args.input_dir)
    rows = build_score_rows(depth_stats)
    rows_with_constant = build_score_rows(depth_stats, include_constant=True)
    temp_increase_rows = build_temperature_increase_rows(all_depth_stats)
    rows.to_csv(os.path.join(output_dir, "lambda_decay_score_effect_T1.csv"), index=False)
    temp_increase_rows.to_csv(
        os.path.join(output_dir, "lambda_temp_increase_score_effect.csv"),
        index=False,
    )

    plot_ood_score(rows, os.path.join(output_dir, "lambda_decay_ood_score_by_depth_T1.png"))
    plot_id_score(rows, os.path.join(output_dir, "lambda_decay_id_score_by_depth_T1.png"))
    plot_id_ood_scores_together(
        rows_with_constant,
        os.path.join(output_dir, "lambda_decay_id_ood_score_by_depth_T1.png"),
    )
    plot_id_ood_score_drops_together(
        rows,
        os.path.join(output_dir, "lambda_decay_id_ood_score_drop_T1.png"),
    )
    plot_score_drop(rows, os.path.join(output_dir, "lambda_decay_ood_score_drop_T1.png"), "ood")
    plot_score_drop(rows, os.path.join(output_dir, "lambda_decay_id_score_drop_T1.png"), "id")
    plot_id_ood_score(
        rows,
        os.path.join(output_dir, "lambda_linear_decay_id_ood_score_by_depth_T1.png"),
        "lambda_linear_decay_0p5",
    )
    plot_baseline_id_ood_score(
        rows_with_constant,
        os.path.join(output_dir, "baseline_id_ood_score_by_depth_T1.png"),
    )
    plot_temperature_increase_ood_score(
        temp_increase_rows,
        os.path.join(output_dir, "lambda_temp_increase_ood_score_by_depth.png"),
    )
    plot_temperature_increase_id_ood_score(
        temp_increase_rows,
        os.path.join(output_dir, "lambda_temp_increase_id_ood_score_by_depth.png"),
    )

    print(f"Saved lambda decay score plots to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
