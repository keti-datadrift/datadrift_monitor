import argparse
import csv
import itertools
import os
import warnings
from types import SimpleNamespace

import torch
from tqdm import tqdm

from gather_hinference import HInferenceEvaluator
from libs.utils import score_util


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_method(evaluator, method_name, method_fn, depth_alpha, depth_beta, min_hdist):
    result = evaluator.predict_and_eval(
        u_method=method_fn,
        u_args={
            "depth_alpha": list(depth_alpha),
            "depth_beta": list(depth_beta),
        },
        min_hdist=min_hdist,
        beta=0.0,
    )

    val_bacc = float(result["val"]["balanced_acc"])
    ood_bacc = float(result["ood"]["balanced_acc"])
    mean_bacc = 0.5 * (val_bacc + ood_bacc)
    val_bhd = float(result["val"]["balanced_hdist"])
    ood_bhd = float(result["ood"]["balanced_hdist"])
    mean_bhd = 0.5 * (val_bhd + ood_bhd)

    return {
        "method": method_name,
        "inference_mode": "minhdist" if min_hdist else "argmax",
        "alpha_0": depth_alpha[0],
        "alpha_1": depth_alpha[1],
        "beta_0": depth_beta[0],
        "beta_1": depth_beta[1],
        "val_balanced_acc": val_bacc,
        "ood_balanced_acc": ood_bacc,
        "mean_balanced_acc": mean_bacc,
        "val_balanced_hdist": val_bhd,
        "ood_balanced_hdist": ood_bhd,
        "mean_balanced_hdist": mean_bhd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", required=True, type=str)
    parser.add_argument("--id_split", required=True, type=str)
    parser.add_argument("--hierarchy", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--grid_values", nargs="+", type=float, default=[0.1, 0.5, 1.0, 3.0, 5.0, 10.0])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="y_pred contains classes not in y_true",
        category=UserWarning,
        module=r"sklearn\.metrics\._classification",
    )

    eval_args = SimpleNamespace(
        hierarchy=args.hierarchy,
        basedir=args.basedir,
        id_split=args.id_split,
        uncertainty_methods=[],
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
    )

    evaluator = HInferenceEvaluator(eval_args)
    methods = {
        "depth_weighted_raw": score_util.depth_weighted_raw,
        "depth_weighted_norm": score_util.depth_weighted_norm,
    }

    rows = []
    grid = list(itertools.product(args.grid_values, repeat=4))
    progress = tqdm(grid, desc="Grid search", unit="combo")

    for a0, a1, b0, b1 in progress:
        depth_alpha = (a0, a1)
        depth_beta = (b0, b1)
        progress.set_postfix({
            "a": f"{a0:g},{a1:g}",
            "b": f"{b0:g},{b1:g}",
        })

        for method_name, method_fn in methods.items():
            for min_hdist in (False, True):
                rows.append(
                    run_method(
                        evaluator,
                        method_name,
                        method_fn,
                        depth_alpha,
                        depth_beta,
                        min_hdist,
                    )
                )

    rows.sort(key=lambda row: (-row["mean_balanced_acc"], row["mean_balanced_hdist"]))
    top5_rows = rows[:5]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    full_csv = os.path.join(output_dir, "grid_search_all_results.csv")
    top5_csv = os.path.join(output_dir, "top5_mean_balanced_acc.csv")
    write_csv(full_csv, rows)
    write_csv(top5_csv, top5_rows)

    summary_txt = os.path.join(output_dir, "top5_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Ranking metric: mean_balanced_acc = (val_balanced_acc + ood_balanced_acc) / 2\n")
        f.write("Methods searched: depth_weighted_raw, depth_weighted_norm\n")
        f.write("Inference modes: argmax, minhdist\n\n")
        for rank, row in enumerate(top5_rows, start=1):
            f.write(
                f"{rank}. method={row['method']}, mode={row['inference_mode']}, "
                f"alpha=[{row['alpha_0']}, {row['alpha_1']}], "
                f"beta=[{row['beta_0']}, {row['beta_1']}], "
                f"mean_balanced_acc={row['mean_balanced_acc']:.6f}, "
                f"val_balanced_acc={row['val_balanced_acc']:.6f}, "
                f"ood_balanced_acc={row['ood_balanced_acc']:.6f}\n"
            )

    print(f"Saved all results to {full_csv}")
    print(f"Saved top-5 summary CSV to {top5_csv}")
    print(f"Saved text summary to {summary_txt}")


if __name__ == "__main__":
    main()
