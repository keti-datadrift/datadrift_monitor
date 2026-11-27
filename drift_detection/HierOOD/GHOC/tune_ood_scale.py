from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path

from tune_temperature_greedy import (
    evaluate_vector,
    format_vector,
    move_tensors_to_device,
    parse_float_list,
    precompute_node_scores,
)

import torch

from core.config import load_config
from core.feature_io import load_artifact
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def write_rows(rows, path: Path) -> None:
    fields = [
        "ood_scale",
        "temperature_vector",
        "val_acc",
        "val_balanced_acc",
        "val_avg_hdist",
        "val_balanced_hdist",
        "ood_acc",
        "ood_balanced_acc",
        "ood_avg_hdist",
        "ood_balanced_hdist",
        "mixed_balanced_acc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", required=True)
    parser.add_argument("--scales", default="0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,5,7,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--cache-scores-on-cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("cgm", {})["enabled"] = False
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    output_dir = Path(args.output_dir)

    device = torch.device(args.device)
    cache_device = torch.device("cpu") if args.cache_scores_on_cpu else device
    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    density_payload = move_tensors_to_device(load_artifact(experiment_dir / "node_density.pt"), device)
    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, _ = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, _ = resolve_feature_tensor(config, experiment_dir, "ood")
    inference_cfg = deepcopy(config["inference"])
    inference_cfg["score_type"] = "gaussian_loglik"
    temperature = parse_float_list(args.temperature)
    inference_cfg["temperature"] = temperature

    print("Precomputing val node scores...", flush=True)
    val_scores = precompute_node_scores(
        val_features,
        density_payload,
        inference_cfg,
        device,
        args.score_batch_size,
        cache_device,
    )
    print("Precomputing ood node scores...", flush=True)
    ood_scores = precompute_node_scores(
        ood_features,
        density_payload,
        inference_cfg,
        device,
        args.score_batch_size,
        cache_device,
    )
    artifacts = {
        "val_artifact": val_artifact,
        "ood_artifact": ood_artifact,
        "val_scores": val_scores,
        "ood_scores": ood_scores,
    }

    rows = []
    for scale in parse_float_list(args.scales):
        inference_cfg["ood_scale"] = scale
        result = evaluate_vector(
            temperature,
            artifacts,
            inference_cfg,
            hierarchy,
            density_payload,
            device,
            args.eval_batch_size,
        )
        row = {
            "ood_scale": scale,
            "temperature_vector": format_vector(temperature),
            "val_acc": result["val"]["acc"],
            "val_balanced_acc": result["val"]["balanced_acc"],
            "val_avg_hdist": result["val"]["avg_hdist"],
            "val_balanced_hdist": result["val"]["balanced_hdist"],
            "ood_acc": result["ood"]["acc"],
            "ood_balanced_acc": result["ood"]["balanced_acc"],
            "ood_avg_hdist": result["ood"]["avg_hdist"],
            "ood_balanced_hdist": result["ood"]["balanced_hdist"],
            "mixed_balanced_acc": 0.5 * (result["val"]["balanced_acc"] + result["ood"]["balanced_acc"]),
        }
        rows.append(row)
        write_rows(rows, output_dir / "ood_scale_results.csv")
        print(row, flush=True)

    best = max(rows, key=lambda row: row["mixed_balanced_acc"])
    write_rows([best], output_dir / "ood_scale_best.csv")
    print("Best:", best, flush=True)


if __name__ == "__main__":
    main()
