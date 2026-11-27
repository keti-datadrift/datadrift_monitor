from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from evaluate_ood import evaluate_features
from extract_features import extract_features
from train import train_model
from utils.io import load_config


def make_run_dir(config: dict, base_dir: str | Path = "runs") -> Path:
    dataset = config["dataset"]["name"].lower()
    model = config["model"].get("type", "auto").lower()
    emb = config["model"].get("embedding_dim", 128)
    seed = config.get("seed", 42)
    run_name = f"{dataset}_{model}_emb{emb}_seed{seed}"
    return Path(base_dir) / run_name


def run_experiment(config_path: str | Path, run_dir: str | Path | None = None):
    config_path = Path(config_path)
    config = load_config(config_path)
    run_dir = Path(run_dir) if run_dir else make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    checkpoint = train_model(config, run_dir)
    features = extract_features(config, checkpoint, run_dir / "features.npz")
    rows = evaluate_features(features, run_dir / "results.csv", config)
    tensorboard_cfg = config.get("tensorboard", {})
    if bool(tensorboard_cfg.get("enabled", True)):
        writer = SummaryWriter(run_dir / tensorboard_cfg.get("log_dir", "tensorboard"))
        for row in rows:
            method = row["OOD Score"]
            for metric in ("AUROC", "FPR@95", "Detection Accuracy", "F1"):
                writer.add_scalar(f"ood/{method}/{metric}", row[metric], 0)
        writer.flush()
        writer.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    rows = run_experiment(args.config, args.run_dir)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
