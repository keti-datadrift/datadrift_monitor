import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from gather_hinference import HInferenceEvaluator, build_uncertainty_args
from libs.utils import score_util
from schedule_experiment_runner import make_args


DATASET_PRESETS = {
    "fgvc-aircraft": {
        "basedir": "ckpts/fgvc-aircraft",
        "id_split": "data/fgvc-aircraft-id-labels.csv",
        "hierarchy": "hierarchies/fgvc-aircraft.json",
        "output_dir": "results/temperature_experiments/fgvc-aircraft",
    },
    "inat19": {
        "basedir": "ckpts/inat19",
        "id_split": "data/inat19-id-labels.csv",
        "hierarchy": "hierarchies/inat19.json",
        "output_dir": "results/temperature_experiments/inat19",
    },
}


def parse_candidates(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def format_float_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_csv(path, rows):
    if rows:
        write_csv(path, rows)


def scalar(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)


def build_base_namespace(args):
    return argparse.Namespace(
        hierarchy=args.hierarchy,
        basedir=args.basedir,
        id_split=args.id_split,
        uncertainty_methods=[args.method],
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
        temperature_vector_json=None,
    )


def evaluate_temperature_setting(evaluator, method_fn, schedule_args, temperature_args, run_name):
    u_args = build_uncertainty_args(schedule_args, evaluator, schedule_args.uncertainty_methods[0])
    min_hdist = run_name.endswith("minhdist")
    result = evaluator.predict_and_eval(
        u_method=method_fn,
        u_args=u_args,
        temperature_args=temperature_args,
        min_hdist=min_hdist,
        beta=0.0,
    )
    return {
        "val_balanced_acc": scalar(result["val"]["balanced_acc"]),
        "ood_balanced_acc": scalar(result["ood"]["balanced_acc"]),
        "mixed_balanced_acc": 0.5 * (
            scalar(result["val"]["balanced_acc"]) + scalar(result["ood"]["balanced_acc"])
        ),
        "val_balanced_hdist": scalar(result["val"]["balanced_hdist"]),
        "ood_balanced_hdist": scalar(result["ood"]["balanced_hdist"]),
        "mixed_balanced_hdist": 0.5 * (
            scalar(result["val"]["balanced_hdist"]) + scalar(result["ood"]["balanced_hdist"])
        ),
    }


def make_temperature_vector_name(vector):
    return "__".join(format_float_tag(v) for v in vector)


def run_global_sweep(args, evaluator, method_fn, candidates, score_depths):
    rows = []
    run_names = [f"{args.method}_argmax", f"{args.method}_minhdist"]
    schedule_args = build_base_namespace(args)

    loop = list(candidates)
    for temperature in tqdm(loop, desc="Global sweep", unit="temp"):
        temperature_args = {
            "temperature_schedule": "constant",
            "temperature_t0": float(temperature),
            "temperature_k": 0.5,
            "temperature_r": 1.25,
            "temperature_vector": None,
        }
        for run_name in run_names:
            metrics = evaluate_temperature_setting(evaluator, method_fn, schedule_args, temperature_args, run_name)
            row = {
                "setting_type": "global",
                "run": run_name,
                "temperature": float(temperature),
                "temperature_spec": f"T={temperature:g}",
                "score_depths": score_depths,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def iterate_depthwise_vectors(candidates, score_depths, monotonic="none"):
    product_iter = itertools.product(candidates, repeat=score_depths)
    for vector in product_iter:
        if monotonic == "increasing" and any(vector[i] > vector[i + 1] for i in range(len(vector) - 1)):
            continue
        if monotonic == "decreasing" and any(vector[i] < vector[i + 1] for i in range(len(vector) - 1)):
            continue
        yield tuple(float(x) for x in vector)


def count_depthwise_vectors(candidates, score_depths, monotonic="none", num_shards=1, shard_index=0):
    if monotonic == "none" and num_shards == 1:
        return len(candidates) ** score_depths
    count = 0
    for idx, _ in enumerate(iterate_depthwise_vectors(candidates, score_depths, monotonic=monotonic)):
        if idx % num_shards != shard_index:
            continue
        count += 1
    return count


def run_depthwise_search(args,
                         evaluator,
                         method_fn,
                         candidates,
                         score_depths,
                         output_dir,
                         monotonic="none",
                         num_shards=1,
                         shard_index=0,
                         batch_size=100):
    rows = []
    run_names = [f"{args.method}_argmax", f"{args.method}_minhdist"]
    schedule_args = build_base_namespace(args)
    total = count_depthwise_vectors(
        candidates,
        score_depths,
        monotonic=monotonic,
        num_shards=num_shards,
        shard_index=shard_index,
    )

    iterator = iterate_depthwise_vectors(candidates, score_depths, monotonic=monotonic)
    processed = 0
    partial_path = os.path.join(output_dir, f"depthwise_temperature_search.shard_{shard_index}_partial.csv")
    final_path = os.path.join(output_dir, f"depthwise_temperature_search.shard_{shard_index}.csv")

    for idx, vector in enumerate(tqdm(iterator, total=None, desc="Depth-wise search", unit="grid")):
        if idx % num_shards != shard_index:
            continue
        temperature_args = {
            "temperature_schedule": "depth_vector",
            "temperature_t0": float(vector[0]),
            "temperature_k": 0.0,
            "temperature_r": 1.0,
            "temperature_vector": list(vector),
        }
        vector_name = make_temperature_vector_name(vector)
        for run_name in run_names:
            metrics = evaluate_temperature_setting(evaluator, method_fn, schedule_args, temperature_args, run_name)
            row = {
                "setting_type": "depthwise",
                "run": run_name,
                "temperature_spec": vector_name,
                "score_depths": score_depths,
            }
            for idx, value in enumerate(vector, start=1):
                row[f"T_depth{idx}"] = float(value)
            row.update(metrics)
            rows.append(row)
        processed += 1
        if batch_size > 0 and processed % batch_size == 0:
            rows.sort(key=lambda r: (r["run"], r["temperature_spec"]))
            maybe_write_csv(partial_path, rows)

    rows.sort(key=lambda r: (r["run"], r["temperature_spec"]))
    maybe_write_csv(final_path, rows)
    maybe_write_csv(partial_path, rows)
    return rows


def summarize_best(global_rows, depth_rows):
    rows = []
    for run_name in sorted({row["run"] for row in global_rows + depth_rows}):
        global_candidates = [row for row in global_rows if row["run"] == run_name]
        depth_candidates = [row for row in depth_rows if row["run"] == run_name]
        best_global = max(global_candidates, key=lambda row: row["mixed_balanced_acc"])
        best_depth = max(depth_candidates, key=lambda row: row["mixed_balanced_acc"])

        rows.append({
            "run": run_name,
            "setting_type": "best_global",
            "temperature_spec": best_global["temperature_spec"],
            "val_balanced_acc": best_global["val_balanced_acc"],
            "ood_balanced_acc": best_global["ood_balanced_acc"],
            "mixed_balanced_acc": best_global["mixed_balanced_acc"],
            "val_balanced_hdist": best_global["val_balanced_hdist"],
            "ood_balanced_hdist": best_global["ood_balanced_hdist"],
            "mixed_balanced_hdist": best_global["mixed_balanced_hdist"],
        })
        rows.append({
            "run": run_name,
            "setting_type": "best_depthwise",
            "temperature_spec": best_depth["temperature_spec"],
            "val_balanced_acc": best_depth["val_balanced_acc"],
            "ood_balanced_acc": best_depth["ood_balanced_acc"],
            "mixed_balanced_acc": best_depth["mixed_balanced_acc"],
            "val_balanced_hdist": best_depth["val_balanced_hdist"],
            "ood_balanced_hdist": best_depth["ood_balanced_hdist"],
            "mixed_balanced_hdist": best_depth["mixed_balanced_hdist"],
        })
        rows.append({
            "run": run_name,
            "setting_type": "depthwise_minus_global",
            "temperature_spec": f"{best_depth['temperature_spec']} - {best_global['temperature_spec']}",
            "val_balanced_acc": best_depth["val_balanced_acc"] - best_global["val_balanced_acc"],
            "ood_balanced_acc": best_depth["ood_balanced_acc"] - best_global["ood_balanced_acc"],
            "mixed_balanced_acc": best_depth["mixed_balanced_acc"] - best_global["mixed_balanced_acc"],
            "val_balanced_hdist": best_depth["val_balanced_hdist"] - best_global["val_balanced_hdist"],
            "ood_balanced_hdist": best_depth["ood_balanced_hdist"] - best_global["ood_balanced_hdist"],
            "mixed_balanced_hdist": best_depth["mixed_balanced_hdist"] - best_global["mixed_balanced_hdist"],
        })
    return rows


def plot_global_sweep(rows, output_path):
    run_names = sorted({row["run"] for row in rows})
    fig, axes = plt.subplots(1, len(run_names), figsize=(6 * len(run_names), 4.5), sharey=True)
    if len(run_names) == 1:
        axes = [axes]

    for ax, run_name in zip(axes, run_names):
        group = sorted((row for row in rows if row["run"] == run_name), key=lambda row: row["temperature"])
        temps = [row["temperature"] for row in group]
        ax.plot(temps, [row["val_balanced_acc"] for row in group], marker="o", label="Val")
        ax.plot(temps, [row["ood_balanced_acc"] for row in group], marker="o", label="OOD")
        ax.plot(temps, [row["mixed_balanced_acc"] for row in group], marker="o", label="Mixed")
        ax.set_title(run_name)
        ax.set_xlabel("Global temperature")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Balanced accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_best_comparison(summary_rows, output_path):
    run_names = sorted({row["run"] for row in summary_rows if row["setting_type"] == "best_global"})
    fig, axes = plt.subplots(1, len(run_names), figsize=(6 * len(run_names), 4.8), sharey=True)
    if len(run_names) == 1:
        axes = [axes]

    for ax, run_name in zip(axes, run_names):
        group = [row for row in summary_rows if row["run"] == run_name and row["setting_type"] in {"best_global", "best_depthwise"}]
        group = sorted(group, key=lambda row: row["setting_type"])
        labels = ["Global", "Depth-wise"]
        x = [0, 1]
        width = 0.22

        ax.bar([i - width for i in x], [group[i]["val_balanced_acc"] for i in range(2)], width=width, label="Val")
        ax.bar(x, [group[i]["ood_balanced_acc"] for i in range(2)], width=width, label="OOD")
        ax.bar([i + width for i in x], [group[i]["mixed_balanced_acc"] for i in range(2)], width=width, label="Mixed")
        ax.set_xticks(x, labels)
        ax.set_title(run_name)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Balanced accuracy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_best_depth_profiles(summary_rows, output_path):
    best_rows = [row for row in summary_rows if row["setting_type"] == "best_depthwise"]
    if not best_rows:
        return

    fig, axes = plt.subplots(1, len(best_rows), figsize=(6 * len(best_rows), 4.5), sharey=True)
    if len(best_rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, best_rows):
        vector = [float(x.replace("p", ".").replace("m", "-")) for x in row["temperature_spec"].split("__")]
        depths = list(range(1, len(vector) + 1))
        ax.plot(depths, vector, marker="o")
        ax.set_title(row["run"])
        ax.set_xlabel("Score depth index")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Temperature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), default=None)
    parser.add_argument("--basedir", type=str, default=None)
    parser.add_argument("--id_split", type=str, default=None)
    parser.add_argument("--hierarchy", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--method", choices=["scheduled_raw", "scheduled_norm"], default="scheduled_raw")
    parser.add_argument(
        "--temperature_candidates",
        type=str,
        default="0.1,0.2,0.3,0.5,1.0,1.5,2.0,3.0,5.0,10.0",
    )
    parser.add_argument("--search_mode", choices=["full_grid"], default="full_grid")
    parser.add_argument("--monotonic", choices=["none", "increasing", "decreasing"], default="none")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--skip_global", action="store_true")
    args = parser.parse_args()

    if args.dataset is not None:
        preset = DATASET_PRESETS[args.dataset]
        args.basedir = args.basedir or preset["basedir"]
        args.id_split = args.id_split or preset["id_split"]
        args.hierarchy = args.hierarchy or preset["hierarchy"]
        args.output_dir = args.output_dir or preset["output_dir"]

    required = ["basedir", "id_split", "hierarchy", "output_dir"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    candidates = parse_candidates(args.temperature_candidates)
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_args = build_base_namespace(args)
    evaluator = HInferenceEvaluator(base_args)
    method_fn = getattr(score_util, args.method)
    score_depths = evaluator.score_depths

    global_rows = []
    if not args.skip_global and args.shard_index == 0:
        global_rows = run_global_sweep(args, evaluator, method_fn, candidates, score_depths)
        write_csv(os.path.join(output_dir, "global_temperature_sweep.csv"), global_rows)
        plot_global_sweep(global_rows, os.path.join(output_dir, "global_temperature_sweep.png"))

    depth_rows = run_depthwise_search(
        args,
        evaluator,
        method_fn,
        candidates,
        score_depths,
        output_dir,
        monotonic=args.monotonic,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        batch_size=args.batch_size,
    )

    if args.num_shards == 1:
        write_csv(os.path.join(output_dir, "depthwise_temperature_search.csv"), depth_rows)
        if global_rows:
            summary_rows = summarize_best(global_rows, depth_rows)
            write_csv(os.path.join(output_dir, "temperature_comparison_summary.csv"), summary_rows)
            plot_best_comparison(summary_rows, os.path.join(output_dir, "global_vs_depthwise_best.png"))
            plot_best_depth_profiles(summary_rows, os.path.join(output_dir, "best_depthwise_temperature_profiles.png"))

    metadata_rows = [{
        "dataset": args.dataset or "",
        "basedir": args.basedir,
        "id_split": args.id_split,
        "hierarchy": args.hierarchy,
        "device": args.device,
        "method": args.method,
        "temperature_candidates": ",".join(str(x) for x in candidates),
        "search_mode": args.search_mode,
        "monotonic": args.monotonic,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "score_depths": score_depths,
        "global_count": len(global_rows) // 2,
        "depthwise_grid_count": len(depth_rows) // 2,
    }]
    write_csv(os.path.join(output_dir, f"experiment_metadata.shard_{args.shard_index}.csv"), metadata_rows)

    print(f"Saved experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
