from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from MS import MS_loss
from data import build_dataloaders
from models import build_model
from ood.gaussian import gaussian_scores
from ood.knn import knn_scores
from ood.mahalanobis import mahalanobis_scores
from utils.gpu import get_device, maybe_data_parallel, unwrap_model
from utils.io import load_config, save_json
from utils.seed import set_seed


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    max_epochs: int,
    show_progress: bool,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_count = 0
    phase = "train" if train else "val"
    iterator = tqdm(
        loader,
        desc=f"epoch {epoch}/{max_epochs} {phase}",
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )

    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            features = model(x)
            loss = MS_loss(features, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * x.size(0)
        total_count += x.size(0)
        running_loss = total_loss / max(total_count, 1)
        iterator.set_postfix(loss=f"{running_loss:.4f}")
        if writer is not None and train:
            writer.add_scalar("batch/train_loss", float(loss.detach()), global_step)
            writer.add_scalar("batch/train_running_loss", running_loss, global_step)
            global_step += 1

    return total_loss / max(total_count, 1), global_step


@torch.no_grad()
def _collect_embeddings(model, loader, device, show_progress: bool, desc: str):
    model.eval()
    features = []
    labels = []
    iterator = tqdm(
        loader,
        desc=desc,
        leave=False,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for x, y in iterator:
        x = x.to(device)
        features.append(model(x).detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def _score_ood(config: dict, train_features, train_labels, test_features):
    method = config.get("model_selection", {}).get("ood_score", "knn").lower()
    k = int(config.get("ood", {}).get("k", 5))
    cov_reg = float(config.get("ood", {}).get("cov_reg", 1e-5))

    if method in {"knn", "kNN".lower()}:
        return knn_scores(train_features, test_features, k)
    if method == "mahalanobis":
        return mahalanobis_scores(train_features, train_labels, test_features, cov_reg)
    if method in {"gaussian", "gaussian_likelihood"}:
        return gaussian_scores(train_features, train_labels, test_features, cov_reg)
    raise ValueError(f"Unsupported model_selection.ood_score: {method}")


def _validation_ood_auroc(config: dict, model, loaders, device, epoch: int, show_progress: bool):
    train_features, train_labels = _collect_embeddings(
        model, loaders["train"], device, show_progress, f"epoch {epoch} OOD bank"
    )
    id_features, _ = _collect_embeddings(
        model, loaders["val"], device, show_progress, f"epoch {epoch} OOD ID-val"
    )
    ood_features, _ = _collect_embeddings(
        model, loaders["ood_test"], device, show_progress, f"epoch {epoch} OOD OOD-val"
    )

    id_scores = _score_ood(config, train_features, train_labels, id_features)
    ood_scores = _score_ood(config, train_features, train_labels, ood_features)
    labels = np.concatenate(
        [np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64)]
    )
    scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(labels, scores))


def train_model(config: dict, run_dir: str | Path):
    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(config)

    datasets, loaders = build_dataloaders(config)
    model = build_model(
        config,
        datasets["input_shape"],
        datasets["input_dim"],
        datasets["input_kind"],
    ).to(device)
    model = maybe_data_parallel(model, config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    max_epochs = int(config["training"].get("epochs", 50))
    show_progress = bool(config["training"].get("progress_bar", True))
    selection_cfg = config.get("model_selection", {})
    default_checkpoint = selection_cfg.get("default_checkpoint", "auroc")
    best_val_loss = float("inf")
    best_val_auroc = -float("inf")
    best_train_loss = float("inf")
    history = []
    checkpoint_paths = {
        "train_loss": run_dir / "checkpoint_best_train_loss.pt",
        "val_loss": run_dir / "checkpoint_best_val_loss.pt",
        "auroc": run_dir / "checkpoint_best_val_ood_auroc.pt",
    }
    tensorboard_cfg = config.get("tensorboard", {})
    use_tensorboard = bool(tensorboard_cfg.get("enabled", True))
    writer = SummaryWriter(run_dir / tensorboard_cfg.get("log_dir", "tensorboard")) if use_tensorboard else None
    global_step = 0

    def save_checkpoint(path: Path, selection_metric: str, selection_score: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "base_model_state": unwrap_model(model).state_dict(),
                "config": config,
                "input_shape": datasets["input_shape"],
                "input_dim": datasets["input_dim"],
                "input_kind": datasets["input_kind"],
                "data_parallel": isinstance(model, torch.nn.DataParallel),
                "selection_metric": selection_metric,
                "selection_score": selection_score,
            },
            path,
        )

    epoch_iter = tqdm(
        range(1, max_epochs + 1),
        desc="training",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    try:
        for epoch in epoch_iter:
            train_loss, global_step = _run_epoch(
                model,
                loaders["train"],
                optimizer,
                device,
                epoch,
                max_epochs,
                show_progress,
                writer,
                global_step,
            )
            val_loss, global_step = _run_epoch(
                model, loaders["val"], None, device, epoch, max_epochs, show_progress, None, global_step
            )
            if writer is not None:
                writer.add_scalar("epoch/train_loss", train_loss, epoch)
                writer.add_scalar("epoch/val_loss", val_loss, epoch)
                writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("epoch/best_val_loss", min(best_val_loss, val_loss), epoch)
            val_auroc = _validation_ood_auroc(
                config, model, loaders, device, epoch, show_progress
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ood_auroc": val_auroc,
                }
            )
            if writer is not None:
                writer.add_scalar("selection/val_ood_auroc", val_auroc, epoch)
                writer.add_scalar("selection/best_val_ood_auroc", max(best_val_auroc, val_auroc), epoch)
                writer.add_scalar("selection/best_train_loss", min(best_train_loss, train_loss), epoch)
                writer.add_scalar("selection/best_val_loss", min(best_val_loss, val_loss), epoch)
            epoch_iter.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_auroc=f"{val_auroc:.4f}",
                best_auroc=f"{max(best_val_auroc, val_auroc):.4f}",
            )

            saved = []
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                save_checkpoint(
                    checkpoint_paths["train_loss"], "min_train_loss", best_train_loss
                )
                saved.append("train_loss")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_paths["val_loss"], "min_val_loss", best_val_loss)
                saved.append("val_loss")

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                save_checkpoint(
                    checkpoint_paths["auroc"], "max_val_ood_auroc", best_val_auroc
                )
                saved.append("auroc")

            status = f"saved: {', '.join(saved)}" if saved else "no checkpoint update"
            tqdm.write(
                f"epoch {epoch}/{max_epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, val_ood_auroc={val_auroc:.4f}, "
                f"best_val_ood_auroc={best_val_auroc:.4f}, {status}"
            )
    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    split = datasets["split_info"]
    save_json(
        run_dir / "metadata.json",
        {
            "id_classes": list(split.id_classes),
            "ood_classes": list(split.ood_classes),
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_val_ood_auroc": best_val_auroc,
            "checkpoint_selection": "all_epochs_best_train_loss_best_val_loss_best_val_ood_auroc",
            "default_checkpoint": default_checkpoint,
            "checkpoints": {
                key: str(path)
                for key, path in checkpoint_paths.items()
            },
            "selection_ood_score": config.get("model_selection", {}).get("ood_score", "knn"),
            "selection_ood_split": "ood_test",
            "epochs_ran": len(history),
            "device": str(device),
            "data_parallel": isinstance(model, torch.nn.DataParallel),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    )
    save_json(run_dir / "train_log.json", {"history": history})
    if default_checkpoint not in checkpoint_paths:
        raise ValueError(
            f"Unknown model_selection.default_checkpoint={default_checkpoint}. "
            f"Choose one of {list(checkpoint_paths)}."
        )
    return checkpoint_paths[default_checkpoint]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    train_model(load_config(args.config), args.run_dir)


if __name__ == "__main__":
    main()
