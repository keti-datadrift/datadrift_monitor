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

import torch
from sklearn.metrics import balanced_accuracy_score

from gather_hinference import HInferenceEvaluator, build_uncertainty_args, get_results
from libs.utils import score_util
from schedule_experiment_runner import build_schedules, make_args, scalar


warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def node_depths(hierarchy, labels):
    depths = []
    for label in labels.tolist():
        node_name = hierarchy.id_node_list[int(label)]
        depths.append(len(hierarchy.node_ancestors[node_name]))
    return torch.tensor(depths, dtype=torch.long)


def safe_balanced_acc(targets, preds):
    if targets.numel() == 0:
        return float("nan")
    return float(balanced_accuracy_score(targets.cpu().numpy(), preds.cpu().numpy()))


def calc_avg_hdist(gt_dists_mat, pred_dists_mat, targets, preds):
    if targets.numel() == 0:
        return float("nan")
    hdists = gt_dists_mat[targets, preds] + pred_dists_mat[targets, preds]
    return float(hdists.float().mean().item())


def add_depth_rows(rows,
                   evaluator,
                   schedule,
                   run_name,
                   split,
                   preds,
                   targets):
    target_depths = node_depths(evaluator.hierarchy, targets)
    pred_depths = node_depths(evaluator.hierarchy, preds)

    for depth_kind, depths in (("target_depth", target_depths), ("pred_depth", pred_depths)):
        for depth in sorted(depths.unique().tolist()):
            mask = depths == depth
            t = targets[mask]
            p = preds[mask]
            rows.append({
                "schedule_name": schedule["name"],
                "run": run_name,
                "split": split,
                "depth_kind": depth_kind,
                "depth": int(depth),
                "num_samples": int(mask.sum().item()),
                "acc": float(p.eq(t).float().mean().item()),
                "balanced_acc": safe_balanced_acc(t, p),
                "avg_hdist": calc_avg_hdist(evaluator.gt_dists_mat, evaluator.pred_dists_mat, t, p),
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


def add_overall_rows(rows, evaluator, schedule, run_name, split, preds, targets):
    result = get_results(
        preds.to("cpu"),
        targets,
        evaluator.hierarchy,
        dists_mats=(evaluator.gt_dists_mat, evaluator.pred_dists_mat),
    )
    rows.append({
        "schedule_name": schedule["name"],
        "run": run_name,
        "split": split,
        "acc": scalar(result["acc"]),
        "balanced_acc": scalar(result["balanced_acc"]),
        "avg_hdist": scalar(result["avg_hdist"]),
        "balanced_hdist": scalar(result["balanced_hdist"]),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--method", choices=["scheduled_raw", "scheduled_norm"], default="scheduled_raw")
    parser.add_argument("--exp_beta_gamma", default=0.5, type=float)
    args = parser.parse_args()

    schedules = build_schedules(args.exp_beta_gamma)
    evaluator = HInferenceEvaluator(make_args(args, schedules[0]))
    method_fn = getattr(score_util, args.method)

    overall_rows = []
    depth_rows = []

    for schedule in schedules:
        print(f"Depth breakdown: {schedule['name']}")
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
                targets = evaluator.node_labels[split]
                add_overall_rows(overall_rows, evaluator, schedule, run_name, split, preds, targets)
                add_depth_rows(depth_rows, evaluator, schedule, run_name, split, preds, targets)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "schedule_overall.csv"), overall_rows)
    write_csv(os.path.join(output_dir, "schedule_depth_breakdown.csv"), depth_rows)
    print(f"Saved depth breakdown to {output_dir}")


if __name__ == "__main__":
    main()
