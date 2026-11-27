from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "experiment_name",
    "dataset",
    "score_type",
    "covariance_type",
    "temperature_vector",
    "alpha_vector",
    "beta_vector",
    "kappa",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="hyp_tuning/all/tuning_results.csv",
        help="Path to tuning_results.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Default: next to input as picked_hparams.csv",
    )
    parser.add_argument("--min_ood_acc", type=float, default=0.2)
    parser.add_argument("--min_val_acc", type=float, default=0.7)
    parser.add_argument("--sort_by", default="ood_acc", choices=["ood_acc", "val_acc", "ood_balanced_acc", "val_balanced_acc"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name("picked_hparams.csv")

    df = pd.read_csv(input_path)
    required_columns = set(KEY_COLUMNS + ["split", "acc", "balanced_acc", "avg_hdist", "balanced_hdist"])
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    val_df = df[df["split"] == "val"].copy()
    ood_df = df[df["split"] == "ood"].copy()

    val_df = val_df.rename(
        columns={
            "acc": "val_acc",
            "balanced_acc": "val_balanced_acc",
            "avg_hdist": "val_avg_hdist",
            "balanced_hdist": "val_balanced_hdist",
        }
    )
    ood_df = ood_df.rename(
        columns={
            "acc": "ood_acc",
            "balanced_acc": "ood_balanced_acc",
            "avg_hdist": "ood_avg_hdist",
            "balanced_hdist": "ood_balanced_hdist",
        }
    )

    keep_val_cols = KEY_COLUMNS + [
        "val_acc",
        "val_balanced_acc",
        "val_avg_hdist",
        "val_balanced_hdist",
    ] + [col for col in val_df.columns if col.startswith("tau_depth_") or col.startswith("alpha_depth_") or col.startswith("beta_depth_")]
    keep_ood_cols = KEY_COLUMNS + [
        "ood_acc",
        "ood_balanced_acc",
        "ood_avg_hdist",
        "ood_balanced_hdist",
    ]

    merged = pd.merge(
        ood_df[keep_ood_cols],
        val_df[keep_val_cols],
        on=KEY_COLUMNS,
        how="inner",
    )

    picked = merged[
        (merged["ood_acc"] > args.min_ood_acc)
        & (merged["val_acc"] > args.min_val_acc)
    ].copy()

    ascending = args.sort_by.endswith("hdist")
    picked = picked.sort_values(by=args.sort_by, ascending=ascending)
    picked.to_csv(output_path, index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Matched hyperparameter sets: {len(merged)}")
    print(
        f"Picked sets with ood_acc > {args.min_ood_acc} and "
        f"val_acc > {args.min_val_acc}: {len(picked)}"
    )
    print(f"Saved filtered results to: {output_path}")

    if not picked.empty:
        preview_cols = [
            "temperature_vector",
            "alpha_vector",
            "beta_vector",
            "ood_acc",
            "val_acc",
            "ood_balanced_acc",
            "val_balanced_acc",
            "ood_avg_hdist",
            "val_avg_hdist",
        ]
        preview_cols = [col for col in preview_cols if col in picked.columns]
        print()
        print(picked[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
