from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data import build_dataloaders
from models import build_model
from utils.gpu import get_device, maybe_data_parallel
from utils.io import load_config


def _base_state_dict(checkpoint: dict):
    state = checkpoint.get("base_model_state", checkpoint["model_state"])
    return {
        key.removeprefix("module."): value
        for key, value in state.items()
    }


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    features = []
    labels = []
    for x, y in loader:
        x = x.to(device)
        z = model(x).cpu().numpy()
        features.append(z)
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def extract_features(config: dict, checkpoint_path: str | Path, output_path: str | Path):
    device = get_device(config)
    datasets, loaders = build_dataloaders(config)
    model = build_model(
        config,
        datasets["input_shape"],
        datasets["input_dim"],
        datasets["input_kind"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(_base_state_dict(checkpoint))
    model = model.to(device)
    model = maybe_data_parallel(model, config)

    split = datasets["split_info"]
    arrays = {}
    for key in ("train", "val", "id_test", "ood_test"):
        arrays[f"{key}_features"], arrays[f"{key}_labels"] = _collect(model, loaders[key], device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **arrays,
        id_classes=np.array(split.id_classes, dtype=np.int64),
        ood_classes=np.array(split.ood_classes, dtype=np.int64),
    )
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_features(load_config(args.config), args.checkpoint, args.output)


if __name__ == "__main__":
    main()
