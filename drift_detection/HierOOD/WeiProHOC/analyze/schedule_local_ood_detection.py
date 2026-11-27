import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gather_hinference import HInferenceEvaluator, build_uncertainty_args
from libs.utils import score_util
from schedule_experiment_runner import build_schedules, make_args
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


def safe_div(num, den):
    return float(num / den) if den else float("nan")


def add_detection_rows(rows,
                       schedule,
                       run_name,
                       split,
                       depth,
                       counts):
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    num_id = tn + fp
    num_ood = tp + fn
    total = num_id + num_ood

    tpr = safe_div(tp, num_ood)
    tnr = safe_div(tn, num_id)
    fpr = safe_div(fp, num_id)
    fnr = safe_div(fn, num_ood)
    acc = safe_div(tp + tn, total)
    balanced_acc = 0.5 * (tpr + tnr) if num_id and num_ood else float("nan")

    rows.append({
        "schedule_name": schedule["name"],
        "run": run_name,
        "split": split,
        "depth": depth,
        "num_id": num_id,
        "num_ood": num_ood,
        "tp_ood_as_ood": tp,
        "tn_id_as_id": tn,
        "fp_id_as_ood": fp,
        "fn_ood_as_id": fn,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "tpr_ood_recall": tpr,
        "tnr_id_recall": tnr,
        "fpr_id_to_ood": fpr,
        "fnr_ood_to_id": fnr,
        "beta_schedule": schedule.get("beta_schedule", "constant"),
        "schedule_beta0": schedule.get("schedule_beta0", 1.0),
        "beta_gamma": schedule.get("beta_gamma", ""),
        "beta_k": schedule.get("beta_k", ""),
        "beta_min": schedule.get("beta_min", ""),
        "temperature_schedule": schedule.get("temperature_schedule", "constant"),
        "temperature_t0": schedule.get("temperature_t0", 1.0),
        "temperature_k": schedule.get("temperature_k", ""),
        "temperature_r": schedule.get("temperature_r", ""),
    })


def get_split_class_names(evaluator, id_classes, ood_classes, split):
    raw_targets = evaluator.data[split][0]["targets"].long().tolist()
    if split == "val":
        return [id_classes[int(target)] for target in raw_targets]
    if split == "ood":
        return [ood_classes[int(target)] for target in raw_targets]
    raise ValueError(f"Unknown split: {split}")


def init_counts_by_depth(internal_nodes_by_depth):
    return {
        int(depth): {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for depth in sorted(internal_nodes_by_depth.keys())
    }


def update_counts_for_split(evaluator,
                            internal_nodes_by_depth,
                            id_leaf_sets,
                            local_ood_leaf_sets,
                            class_names,
                            preds,
                            split):
    pred_names = [evaluator.hierarchy.id_node_list[int(pred)] for pred in preds.tolist()]
    counts_by_depth = init_counts_by_depth(internal_nodes_by_depth)

    for depth, node_names in internal_nodes_by_depth.items():
        counts = counts_by_depth[int(depth)]
        for node_name in node_names:
            pred_ood = [pred_name == node_name for pred_name in pred_names]

            if split == "val":
                # Val samples are global ID, but only samples under this local parent
                # contribute to the local ID-vs-OOD detection task.
                local_id_mask = [class_name in id_leaf_sets[node_name] for class_name in class_names]
                for is_id, is_pred_ood in zip(local_id_mask, pred_ood):
                    if not is_id:
                        continue
                    if is_pred_ood:
                        counts["fp"] += 1
                    else:
                        counts["tn"] += 1
            elif split == "ood":
                local_ood_mask = [class_name in local_ood_leaf_sets[node_name] for class_name in class_names]
                for is_ood, is_pred_ood in zip(local_ood_mask, pred_ood):
                    if not is_ood:
                        continue
                    if is_pred_ood:
                        counts["tp"] += 1
                    else:
                        counts["fn"] += 1

    return counts_by_depth


def merge_split_counts(val_counts, ood_counts):
    merged = {}
    for depth in sorted(set(val_counts.keys()).union(ood_counts.keys())):
        merged[depth] = {
            key: val_counts.get(depth, {}).get(key, 0) + ood_counts.get(depth, {}).get(key, 0)
            for key in ("tp", "tn", "fp", "fn")
        }
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--method", choices=["scheduled_raw", "scheduled_norm"], default="scheduled_raw")
    parser.add_argument("--exp_beta_gamma", default=0.5, type=float)
    parser.add_argument("--temp_t0", default=1.5, type=float)
    parser.add_argument("--temp_linear_k", default=0.5, type=float)
    parser.add_argument("--temp_exp_r", default=1.25, type=float)
    args = parser.parse_args()

    id_classes = get_id_classes(args.id_split)
    schedules = build_schedules(
        args.exp_beta_gamma,
        args.temp_t0,
        args.temp_linear_k,
        args.temp_exp_r,
    )
    evaluator = HInferenceEvaluator(make_args(args, schedules[0]))
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

    method_fn = getattr(score_util, args.method)
    rows = []

    for schedule in schedules:
        print(f"Local OOD detection: {schedule['name']}")
        schedule_args = make_args(args, schedule)
        u_args = build_uncertainty_args(schedule_args, evaluator, args.method)
        temperature_args = {
            "temperature_schedule": schedule_args.temperature_schedule,
            "temperature_t0": schedule_args.temperature_t0,
            "temperature_k": schedule_args.temperature_k,
            "temperature_r": schedule_args.temperature_r,
        }

        for min_hdist in (False, True):
            run_name = f"{args.method}_{'minhdist' if min_hdist else 'argmax'}"
            counts_by_split = {}

            for split in evaluator.dsets:
                preds = evaluator.multi_predict(
                    evaluator.logits[split],
                    evaluator.softmax[split],
                    u_method=method_fn,
                    u_args=u_args,
                    temperature_args=temperature_args,
                    min_hdist=min_hdist,
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
            for depth, counts in merged_counts.items():
                add_detection_rows(rows, schedule, run_name, "val+ood", depth, counts)
            for split, split_counts in counts_by_split.items():
                for depth, counts in split_counts.items():
                    add_detection_rows(rows, schedule, run_name, split, depth, counts)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "local_ood_detection_by_depth.csv"), rows)
    print(f"Saved local OOD detection breakdown to {output_dir}")


if __name__ == "__main__":
    main()
