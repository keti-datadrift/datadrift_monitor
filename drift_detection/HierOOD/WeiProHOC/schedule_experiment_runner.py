import argparse
import csv
import os
import warnings
from types import SimpleNamespace

import torch

from gather_hinference import HInferenceEvaluator, build_uncertainty_args
from libs.utils import score_util


warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


BETA_CONFIGS = [
    {
        "name": "beta_constant",
        "beta_schedule": "constant",
        "schedule_beta0": 1.0,
    },
    {
        "name": "beta_inverse_depth",
        "beta_schedule": "inverse_depth",
        "schedule_beta0": 1.0,
    },
    {
        "name": "beta_exp_decay_0p5",
        "beta_schedule": "exp_decay",
        "schedule_beta0": 1.0,
        "beta_gamma": 0.5,
    },
    {
        "name": "beta_linear_decay_0p5",
        "beta_schedule": "linear_decay",
        "schedule_beta0": 1.0,
        "beta_k": 0.5,
        "beta_min": 0.0,
    },
]

TEMPERATURE_CONFIGS = [
    {
        "name": "temp_constant_1p0",
        "temperature_schedule": "constant",
        "temperature_t0": 1.0,
    },
    {
        "name": "temp_constant_1p5",
        "temperature_schedule": "constant",
        "temperature_t0": 1.5,
    },
    {
        "name": "temp_linear_1p5_0p5",
        "temperature_schedule": "linear_increase",
        "temperature_t0": 1.5,
        "temperature_k": 0.5,
    },
    {
        "name": "temp_exp_1p5_1p25",
        "temperature_schedule": "exp_increase",
        "temperature_t0": 1.5,
        "temperature_r": 1.25,
    },
]


def format_float_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def build_schedules(exp_beta_gamma=0.5, temp_t0=1.5, temp_linear_k=0.5, temp_exp_r=1.25):
    schedules = []
    temperature_configs = []
    for temperature_config in TEMPERATURE_CONFIGS:
        temperature_config = dict(temperature_config)
        if temperature_config.get("name") == "temp_constant_1p5":
            temperature_config["temperature_t0"] = float(temp_t0)
            temperature_config["name"] = f"temp_constant_{format_float_tag(temp_t0)}"
        if temperature_config.get("temperature_schedule") == "linear_increase":
            temperature_config["temperature_t0"] = float(temp_t0)
            temperature_config["temperature_k"] = float(temp_linear_k)
            temperature_config["name"] = f"temp_linear_{format_float_tag(temp_t0)}_{format_float_tag(temp_linear_k)}"
        if temperature_config.get("temperature_schedule") == "exp_increase":
            temperature_config["temperature_t0"] = float(temp_t0)
            temperature_config["temperature_r"] = float(temp_exp_r)
            temperature_config["name"] = f"temp_exp_{format_float_tag(temp_t0)}_{format_float_tag(temp_exp_r)}"
        temperature_configs.append(temperature_config)

    for beta_config in BETA_CONFIGS:
        beta_config = dict(beta_config)
        if beta_config.get("beta_schedule") == "exp_decay":
            beta_config["beta_gamma"] = float(exp_beta_gamma)
            beta_config["name"] = f"beta_exp_decay_{format_float_tag(exp_beta_gamma)}"
        for temperature_config in temperature_configs:
            schedule = {**beta_config, **temperature_config}
            schedule["name"] = f"{beta_config['name']}__{temperature_config['name']}"
            schedules.append(schedule)

    unique_schedules = []
    seen = set()
    for schedule in schedules:
        key = (
            schedule.get("beta_schedule"),
            schedule.get("schedule_beta0", 1.0),
            schedule.get("beta_gamma", ""),
            schedule.get("beta_k", ""),
            schedule.get("beta_min", ""),
            schedule.get("temperature_schedule"),
            schedule.get("temperature_t0", 1.0),
            schedule.get("temperature_k", ""),
            schedule.get("temperature_r", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_schedules.append(schedule)
    return unique_schedules


SCHEDULES = build_schedules()


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_args(args, schedule):
    values = {
        "hierarchy": args.hierarchy,
        "basedir": args.basedir,
        "id_split": args.id_split,
        "uncertainty_methods": [args.method],
        "betas": [0.0],
        "farood": [],
        "depth_alpha": None,
        "depth_beta": None,
        "beta_rule": "ones",
        "node_alpha_json": None,
        "node_beta_json": None,
        "result_suffix": "",
        "output_path": None,
        "device": args.device,
        "beta_schedule": schedule.get("beta_schedule", "constant"),
        "schedule_beta0": schedule.get("schedule_beta0", 1.0),
        "beta_gamma": schedule.get("beta_gamma", 0.5),
        "beta_k": schedule.get("beta_k", 0.5),
        "beta_min": schedule.get("beta_min", 0.0),
        "temperature_schedule": schedule.get("temperature_schedule", "constant"),
        "temperature_t0": schedule.get("temperature_t0", 1.0),
        "temperature_k": schedule.get("temperature_k", 0.5),
        "temperature_r": schedule.get("temperature_r", 1.5),
    }
    return SimpleNamespace(**values)


def scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def add_rows(rows, schedule_name, schedule, run_name, result):
    for split in ("val", "ood"):
        metrics = result[split]
        rows.append({
            "schedule_name": schedule_name,
            "run": run_name,
            "split": split,
            "balanced_acc": scalar(metrics["balanced_acc"]),
            "balanced_hdist": scalar(metrics["balanced_hdist"]),
            "avg_hdist": scalar(metrics["avg_hdist"]),
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


def add_mean_rows(summary_rows, detailed_rows):
    grouped = {}
    for row in detailed_rows:
        key = (row["schedule_name"], row["run"])
        grouped.setdefault(key, {})[row["split"]] = row

    for (schedule_name, run), split_rows in grouped.items():
        if "val" not in split_rows or "ood" not in split_rows:
            continue
        val = split_rows["val"]
        ood = split_rows["ood"]
        summary_rows.append({
            "schedule_name": schedule_name,
            "run": run,
            "mean_balanced_acc": 0.5 * (val["balanced_acc"] + ood["balanced_acc"]),
            "val_balanced_acc": val["balanced_acc"],
            "ood_balanced_acc": ood["balanced_acc"],
            "mean_balanced_hdist": 0.5 * (val["balanced_hdist"] + ood["balanced_hdist"]),
            "val_balanced_hdist": val["balanced_hdist"],
            "ood_balanced_hdist": ood["balanced_hdist"],
            "beta_schedule": val["beta_schedule"],
            "schedule_beta0": val["schedule_beta0"],
            "beta_gamma": val["beta_gamma"],
            "beta_k": val["beta_k"],
            "beta_min": val["beta_min"],
            "temperature_schedule": val["temperature_schedule"],
            "temperature_t0": val["temperature_t0"],
            "temperature_k": val["temperature_k"],
            "temperature_r": val["temperature_r"],
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
    parser.add_argument("--temp_t0", default=1.5, type=float)
    parser.add_argument("--temp_linear_k", default=0.5, type=float)
    parser.add_argument("--temp_exp_r", default=1.25, type=float)
    args = parser.parse_args()

    schedules = build_schedules(args.exp_beta_gamma, args.temp_t0, args.temp_linear_k, args.temp_exp_r)

    base_args = make_args(args, schedules[0])
    evaluator = HInferenceEvaluator(base_args)
    method_fn = getattr(score_util, args.method)

    detail_rows = []
    summary_rows = []

    for schedule in schedules:
        schedule_name = schedule["name"]
        print(f"Evaluating schedule: {schedule_name}")
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
            result = evaluator.predict_and_eval(
                u_method=method_fn,
                u_args=u_args,
                temperature_args=temperature_args,
                min_hdist=min_hdist,
                beta=0.0,
            )
            add_rows(detail_rows, schedule_name, schedule, run_name, result)

    add_mean_rows(summary_rows, detail_rows)
    summary_rows.sort(key=lambda row: (-row["mean_balanced_acc"], row["mean_balanced_hdist"]))

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, "schedule_detail.csv"), detail_rows)
    write_csv(os.path.join(output_dir, "schedule_summary.csv"), summary_rows)
    print(f"Saved schedule results to {output_dir}")


if __name__ == "__main__":
    main()
