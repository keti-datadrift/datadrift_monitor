import argparse
import csv
import json
import os

import numpy as np
import torch


def to_python_scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_summary_rows(result_obj, include_metadata=False):
    metadata = result_obj.get("metadata", {})
    rows = []

    for run_name, run_data in result_obj.items():
        if run_name == "metadata":
            continue

        for split_name, metrics in run_data.items():
            row = {
                "run": run_name,
                "split": split_name,
                "acc": to_python_scalar(metrics.get("acc")),
                "balanced_acc": to_python_scalar(metrics.get("balanced_acc")),
                "avg_hdist": to_python_scalar(metrics.get("avg_hdist")),
                "balanced_hdist": to_python_scalar(metrics.get("balanced_hdist")),
            }

            hdist = metrics.get("hdist")
            if hdist is not None:
                hdist = to_python_scalar(hdist)
                row["hdist_shape"] = json.dumps(np.array(hdist).shape)
            else:
                row["hdist_shape"] = ""

            if include_metadata:
                row["depth_alpha"] = json.dumps(metadata.get("depth_alpha", []))
                row["depth_beta"] = json.dumps(metadata.get("depth_beta", []))
                row["beta_rule"] = metadata.get("beta_rule", "")
                row["device"] = metadata.get("device", "")

            rows.append(row)

    return rows


def build_class_hdist_rows(result_obj):
    rows = []

    for run_name, run_data in result_obj.items():
        if run_name == "metadata":
            continue

        for split_name, metrics in run_data.items():
            class_hdists = metrics.get("class_hdists", {})
            if not isinstance(class_hdists, dict):
                continue

            for class_name, class_hdist in class_hdists.items():
                rows.append({
                    "run": run_name,
                    "split": split_name,
                    "class_name": class_name,
                    "class_hdist": to_python_scalar(class_hdist),
                })

    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(result_path):
    base, _ = os.path.splitext(result_path)
    return f"{base}.csv"


def default_class_output_path(result_path):
    base, _ = os.path.splitext(result_path)
    return f"{base}.class_hdists.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", required=True, type=str)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--class_hdists_csv", type=str, default=None)
    parser.add_argument("--include_metadata", action="store_true")
    args = parser.parse_args()

    result_obj = torch.load(args.result_path, map_location="cpu", weights_only=False)

    summary_rows = build_summary_rows(result_obj, include_metadata=args.include_metadata)
    output_csv = args.output_csv or default_output_path(args.result_path)
    write_csv(output_csv, summary_rows)
    print(f"Saved summary CSV to {output_csv}")

    class_rows = build_class_hdist_rows(result_obj)
    if args.class_hdists_csv or class_rows:
        class_output = args.class_hdists_csv or default_class_output_path(args.result_path)
        write_csv(class_output, class_rows)
        print(f"Saved class-hdist CSV to {class_output}")


if __name__ == "__main__":
    main()
