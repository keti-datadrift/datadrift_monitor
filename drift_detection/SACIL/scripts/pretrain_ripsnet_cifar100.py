from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from torch import nn
from torch.optim import Adamax
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sacil.data import ClassOrderProtocol, build_data_module  # noqa: E402
from sacil.models import (  # noqa: E402
    AFCIncrementalNet,
    IncrementalNet,
    TaKPIncrementalNet,
)
from sacil.methods import RipsNet  # noqa: E402
from sacil.utils import atomic_torch_save, make_generator, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain an H0 persistence-image RipsNet for SACIL"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "experiment_configs/class_orders/"
            "cifar100_b50_t10_afc_order1.json"
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/ripsnet_cifar100_resnet32_h0/seed_1/best.pt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-family",
        choices=("standard", "afc", "takp"),
        default="standard",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--point-clouds", type=int, default=19968)
    parser.add_argument("--cloud-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def collect_point_clouds(
    model: IncrementalNet | AFCIncrementalNet | TaKPIncrementalNet,
    data,
    *,
    count: int,
    cloud_size: int,
    device: torch.device,
    seed: int,
    num_workers: int,
) -> np.ndarray:
    dataset = data.new_train_dataset(0, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=cloud_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        generator=make_generator(seed),
    )
    clouds: list[np.ndarray] = []
    model.eval()
    while len(clouds) < count:
        for batch in loader:
            features = model.extract_features(
                batch["image"].to(device, non_blocking=True)
            )
            clouds.append(features.float().cpu().numpy())
            if len(clouds) >= count:
                break
    return np.stack(clouds).astype(np.float32, copy=False)


def h0_death_times(point_clouds: np.ndarray) -> np.ndarray:
    count, cloud_size, _ = point_clouds.shape
    deaths = np.zeros((count, cloud_size - 1), dtype=np.float32)
    for index, cloud in enumerate(point_clouds):
        distances = squareform(pdist(cloud, metric="euclidean"))
        tree = minimum_spanning_tree(distances)
        values = np.asarray(tree.data, dtype=np.float32)
        values.sort()
        deaths[index, : values.size] = values
        if (index + 1) % 128 == 0 or index + 1 == count:
            print(f"MST {index + 1}/{count}", flush=True)
    return deaths


def persistence_images(deaths: np.ndarray, resolution: int = 20):
    flattened = deaths.reshape(-1)
    finite = flattened[np.isfinite(flattened) & (flattened > 0)]
    if finite.size == 0:
        raise RuntimeError("H0 persistence diagrams are empty")
    first = finite[: min(200, finite.size)]
    pairwise = np.abs(first[:, None] - first[None, :]).reshape(-1)
    positive = pairwise[pairwise > 1e-5]
    sigma = float(np.quantile(positive, 0.2)) if positive.size else 1e-3
    sigma = max(sigma, 1e-6)
    persistence_min = float(finite.min())
    persistence_max = float(finite.max())
    y_grid = np.linspace(
        persistence_min,
        persistence_max,
        resolution,
        dtype=np.float32,
    )

    images = np.empty(
        (deaths.shape[0], resolution * resolution), dtype=np.float32
    )
    gaussian_scale = 1.0 / (2.0 * sigma * sigma)
    for start in range(0, deaths.shape[0], 128):
        stop = min(start + 128, deaths.shape[0])
        values = deaths[start:stop]
        weights = 10.0 * np.tanh(values)
        delta = values[:, :, None] - y_grid[None, None, :]
        vertical = (
            weights[:, :, None]
            * np.exp(-(delta * delta) * gaussian_scale)
        ).sum(axis=1)
        image = np.repeat(vertical[:, None, :], resolution, axis=1)
        images[start:stop] = image.reshape(stop - start, -1)
    maximum = float(images.max())
    if maximum > 0:
        images /= maximum
    metadata = {
        "homology_dimension": 0,
        "resolution": [resolution, resolution],
        "bandwidth": sigma,
        "persistence_range": [persistence_min, persistence_max],
        "normalization_max": maximum,
    }
    return images, metadata


def train_ripsnet(
    point_clouds: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(point_clouds.shape[0])
    split = int(0.75 * indices.size)
    train_indices, validation_indices = indices[:split], indices[split:]
    train_dataset = TensorDataset(
        torch.from_numpy(point_clouds[train_indices]),
        torch.from_numpy(targets[train_indices]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(point_clouds[validation_indices]),
        torch.from_numpy(targets[validation_indices]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=make_generator(seed + 1),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    feature_dim = int(point_clouds.shape[-1])
    units = {
        16: (16, 16, 16, 50, 100, 200),
        32: (32, 16, 16, 50, 100, 200),
        64: (64, 32, 32, 50, 100, 200),
        128: (128, 64, 64, 50, 100, 200),
        256: (256, 128, 128, 50, 100, 200),
        512: (512, 256, 256, 50, 100, 200),
        1024: (512, 256, 256, 50, 100, 200),
        2048: (1024, 512, 256, 50, 100, 200),
    }
    if feature_dim not in units:
        raise ValueError(
            f"no public TaKP RipsNet width map for {feature_dim}-D features"
        )
    hidden_dims = units[feature_dim]
    model = RipsNet(
        feature_dim=point_clouds.shape[-1],
        hidden_dims=hidden_dims,
        output_dim=targets.shape[-1],
        operator="mean",
    ).to(device)
    optimizer = Adamax(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for clouds, labels in train_loader:
            clouds = clouds.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(clouds), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().item()) * clouds.shape[0]
            train_count += clouds.shape[0]
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for clouds, labels in validation_loader:
                clouds = clouds.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                loss = criterion(model(clouds), labels)
                validation_total += (
                    float(loss.detach().item()) * clouds.shape[0]
                )
                validation_count += clouds.shape[0]
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % 10 == 0 or stale_epochs == 0:
            print(
                f"epoch={epoch} train={train_loss:.8f} "
                f"val={validation_loss:.8f} best={best_loss:.8f}",
                flush=True,
            )
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("RipsNet training did not produce a checkpoint")
    validation_targets = targets[validation_indices]
    zero_mse = float(np.square(validation_targets).mean())
    mean_target = targets[train_indices].mean(axis=0, keepdims=True)
    mean_predictor_mse = float(
        np.square(validation_targets - mean_target).mean()
    )
    return (
        best_state,
        best_loss,
        history,
        hidden_dims,
        zero_mse,
        mean_predictor_mse,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device)
    protocol_path = (PROJECT_ROOT / args.protocol).resolve()
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    checkpoint_path = (PROJECT_ROOT / args.base_checkpoint).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    protocol = ClassOrderProtocol.from_json(protocol_path)
    if args.model_family == "afc":
        model = AFCIncrementalNet(
            num_classes=int(checkpoint["num_classes"]),
            initial_size=protocol.session(0).size,
            increment_size=protocol.session(1).size,
            proxies_per_class=int(
                checkpoint.get("config", {})
                .get("model", {})
                .get("proxies_per_class", 10)
            ),
            classifier_scale=float(
                checkpoint.get("config", {})
                .get("model", {})
                .get("classifier_scale", 1.0)
            ),
        )
    elif args.model_family == "takp":
        model_config = checkpoint.get("config", {}).get("model", {})
        model = TaKPIncrementalNet(
            num_classes=int(checkpoint["num_classes"]),
            mix_scale=float(model_config.get("mix_scale", 2.0)),
            stem=str(model_config.get("stem", "imagenet")),
        )
    else:
        model_config = (
            checkpoint.get("config", {}).get("model", {})
        )
        model = IncrementalNet(
            num_classes=int(checkpoint["num_classes"]),
            backbone=str(model_config.get("backbone", "resnet32")),
            classifier_scale=float(
                model_config.get("classifier_scale", 10.0)
            ),
            learnable_scale=bool(
                model_config.get("learnable_scale", True)
            ),
        )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    data = build_data_module("cifar100", data_root, protocol)

    point_clouds = collect_point_clouds(
        model,
        data,
        count=args.point_clouds,
        cloud_size=args.cloud_size,
        device=device,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    deaths = h0_death_times(point_clouds)
    targets, persistence_metadata = persistence_images(deaths)
    (
        state,
        best_loss,
        history,
        hidden_dims,
        zero_mse,
        mean_predictor_mse,
    ) = train_ripsnet(
        point_clouds,
        targets,
        device=device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    metadata = {
        "feature_dim": int(point_clouds.shape[-1]),
        "hidden_dims": list(hidden_dims),
        "output_dim": int(targets.shape[-1]),
        "operator": "mean",
        "point_clouds": int(point_clouds.shape[0]),
        "cloud_size": int(point_clouds.shape[1]),
        "seed": args.seed,
        "model_family": args.model_family,
        "base_checkpoint": str(checkpoint_path),
        "protocol": str(protocol_path),
        "best_validation_loss": best_loss,
        "zero_predictor_validation_mse": zero_mse,
        "mean_predictor_validation_mse": mean_predictor_mse,
        "training_epochs_completed": len(history),
        "persistence_image": persistence_metadata,
        "provenance": (
            "Locally retrained from the public TaKP RipsNet architecture "
            "and H0 persistence-image recipe; the authors did not publish "
            "their trained RipsNet checkpoint."
        ),
    }
    atomic_torch_save(
        {"model": state, "metadata": metadata, "history": history},
        output_path,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
