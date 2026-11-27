import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from gather_hinference import HInferenceEvaluator
from libs.utils import score_util
from analyze.schedule_local_ood_detection import (
    add_detection_rows,
    get_split_class_names,
    merge_split_counts,
    update_counts_for_split,
)
from analyze.temperature_sweep_analysis import (
    build_leaf_descendant_map,
    build_node_analysis_plan,
    get_id_classes,
    read_full_hierarchy,
)


warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_base_args(args):
    return SimpleNamespace(
        hierarchy=args.hierarchy,
        basedir=args.basedir,
        id_split=args.id_split,
        uncertainty_methods=["scheduled_raw"],
        betas=[0.0],
        farood=[],
        depth_alpha=None,
        depth_beta=None,
        beta_rule="ones",
        node_alpha_json=None,
        node_beta_json=None,
        result_suffix="",
        output_path=None,
        device=args.device,
        beta_schedule="constant",
        schedule_beta0=1.0,
        beta_gamma=0.5,
        beta_k=0.5,
        beta_min=0.0,
        temperature_schedule="constant",
        temperature_t0=1.0,
        temperature_k=0.5,
        temperature_r=1.25,
    )


def make_schedule(temperature):
    tag = str(temperature).replace(".", "p")
    return {
        "name": f"lambda_constant__temp_constant_{tag}",
        "beta_schedule": "constant",
        "schedule_beta0": 1.0,
        "temperature_schedule": "constant",
        "temperature_t0": float(temperature),
    }


def make_u_args():
    return {
        "beta_schedule": "constant",
        "beta0": 1.0,
        "schedule_beta0": 1.0,
        "beta_gamma": 0.5,
        "beta_k": 0.5,
        "beta_min": 0.0,
    }


def plot_temperature_sweep(rows, output_path):
    temps = [row["temperature_t0"] for row in rows]
    ood = [row["tpr_ood_recall"] for row in rows]
    id_ = [row["tnr_id_recall"] for row in rows]
    mixed = [row["balanced_acc"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(temps, id_, marker="o", color="tab:blue", label="ID recall")
    ax.plot(temps, ood, marker="o", color="tab:red", label="OOD recall")
    ax.plot(temps, mixed, marker="o", color="tab:green", label="Mixed balanced acc")
    ax.set_title("Local OOD Detection vs Constant Temperature")
    ax.set_xlabel("Softmax temperature")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0, 5.0])
    args = parser.parse_args()

    id_classes = get_id_classes(args.id_split)
    evaluator = HInferenceEvaluator(make_base_args(args))
    ood_classes = evaluator.hierarchy.ood_train_classes

    full_tree = read_full_hierarchy(args.hierarchy)
    full_leaf_descendants = build_leaf_descendant_map(full_tree)
    internal_nodes_by_depth, _, id_leaf_sets, local_ood_leaf_sets = build_node_analysis_plan(
        evaluator.hierarchy,
        id_classes,
        ood_classes,
        full_leaf_descendants,
    )

    class_names = {
        split: get_split_class_names(evaluator, id_classes, ood_classes, split)
        for split in evaluator.dsets
    }

    detail_rows = []
    summary_rows = []
    method_fn = score_util.scheduled_raw
    u_args = make_u_args()

    for temperature in args.temperatures:
        schedule = make_schedule(temperature)
        print(f"Temperature sweep local OOD detection: T={temperature:g}")
        temperature_args = {
            "temperature_schedule": "constant",
            "temperature_t0": float(temperature),
            "temperature_k": 0.5,
            "temperature_r": 1.25,
        }

        counts_by_split = {}
        for split in evaluator.dsets:
            preds = evaluator.multi_predict(
                evaluator.logits[split],
                evaluator.softmax[split],
                u_method=method_fn,
                u_args=u_args,
                temperature_args=temperature_args,
                min_hdist=False,
                beta=0.0,
            ).to("cpu")
            counts_by_split[split] = update_counts_for_split(
                evaluator,
                internal_nodes_by_depth,
                id_leaf_sets,
                local_ood_leaf_sets,
                class_names[split],
                preds,
                split,
            )

        merged_counts = merge_split_counts(counts_by_split["val"], counts_by_split["ood"])
        total_counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for depth, counts in merged_counts.items():
            add_detection_rows(detail_rows, schedule, "scheduled_raw_argmax", "val+ood", depth, counts)
            for key in total_counts:
                total_counts[key] += counts[key]

        before = len(summary_rows)
        add_detection_rows(summary_rows, schedule, "scheduled_raw_argmax", "val+ood_all_depths", -1, total_counts)
        summary_rows[before]["temperature_t0"] = float(temperature)

    summary_rows.sort(key=lambda row: row["temperature_t0"])

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "temp_constant_local_ood_detection_summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "temp_constant_local_ood_detection_by_depth.csv"), detail_rows)
    plot_temperature_sweep(
        summary_rows,
        os.path.join(output_dir, "temp_constant_local_ood_detection_summary.png"),
    )
    print(f"Saved temperature local OOD detection results to {output_dir}")


if __name__ == "__main__":
    main()
