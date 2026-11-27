from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ood import compute_metrics, gaussian_scores, knn_scores, mahalanobis_scores
from utils.io import append_csv


def evaluate_features(
    feature_path: str | Path,
    output_csv: str | Path,
    config: dict,
    extra_fields: dict | None = None,
):
    data = np.load(feature_path)
    train_features = data["train_features"]
    train_labels = data["train_labels"]
    id_features = data["id_test_features"]
    ood_features = data["ood_test_features"]

    k = int(config.get("ood", {}).get("k", 5))
    cov_reg = float(config.get("ood", {}).get("cov_reg", 1e-5))
    dataset = config["dataset"]["name"]
    model = config["model"].get("type", "auto")

    methods = {
        "kNN": (
            knn_scores(train_features, id_features, k),
            knn_scores(train_features, ood_features, k),
        ),
        "Mahalanobis": (
            mahalanobis_scores(train_features, train_labels, id_features, cov_reg),
            mahalanobis_scores(train_features, train_labels, ood_features, cov_reg),
        ),
        "Gaussian": (
            gaussian_scores(train_features, train_labels, id_features, cov_reg),
            gaussian_scores(train_features, train_labels, ood_features, cov_reg),
        ),
    }

    rows = []
    for method, (id_scores, ood_scores) in methods.items():
        metrics = compute_metrics(id_scores, ood_scores)
        row = {
            **(extra_fields or {}),
            "Dataset": dataset,
            "Feature Extractor": model,
            "OOD Score": method,
            **metrics,
        }
        append_csv(output_csv, row)
        rows.append(row)
    return rows


def main():
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    evaluate_features(args.features, args.output_csv, config)


if __name__ == "__main__":
    main()
