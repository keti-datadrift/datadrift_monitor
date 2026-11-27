from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config, save_config
from core.density import fit_node_distributions, score_nodes
from core.feature_io import save_artifact
from core.hierarchy_labels import build_leaf_path_matrix, targets_to_path_labels
from core.metric_losses import build_metric_loss
from core.model_factory import backbone_summary, build_backbone, maybe_wrap_dataparallel, unwrap_model
from core.hierarchy_inference import build_depth_maps
from core.gradient_cache import (
    GradientCacheConfig,
    autocast_context,
    configure_batch_norm,
    gradient_cache_step,
)
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import gen_datasets, get_id_classes


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_transform_kwargs(config):
    dataset_cfg = config["dataset"]
    backbone_type = config["backbone"]["type"].lower()
    preset = dataset_cfg.get("transform_preset")
    if preset is None:
        preset = "clip" if backbone_type == "clip" else "imagenet"
    return {
        "preset": preset,
        "mean": dataset_cfg.get("mean"),
        "std": dataset_cfg.get("std"),
        "resize": dataset_cfg.get("resize"),
        "cropsize": dataset_cfg.get("cropsize"),
        "model_name": config["backbone"].get("model_name"),
    }


def build_optimizer(model, config):
    opt_cfg = config["optimizer"]
    name = opt_cfg["name"].lower()
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            nesterov=opt_cfg.get("nesterov", False),
        )
    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")


def build_scheduler(optimizer, config):
    sched_cfg = config.get("scheduler", {})
    name = sched_cfg.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=sched_cfg.get("eta_min", 0.0),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg.get('name')}")


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "model", "state_dict", "backbone_state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Could not find model weights in checkpoint.")


def load_model_weights(model, checkpoint):
    state_dict = extract_model_state(checkpoint)
    target_model = unwrap_model(model)
    try:
        target_model.load_state_dict(state_dict)
        return
    except RuntimeError:
        stripped = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        target_model.load_state_dict(stripped)


def load_resume_checkpoint(model, optimizer, scheduler, scaler, config, device):
    resume_from = config["training"].get("resume_from")
    if not resume_from:
        return 0, float("inf"), None

    checkpoint = torch.load(resume_from, map_location=device)
    load_model_weights(model, checkpoint)

    start_epoch = 0
    best_val_loss = float("inf")
    history = None

    if isinstance(checkpoint, dict):
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = checkpoint.get("history")

    print(f"Resumed from checkpoint: {resume_from}")
    if start_epoch > 0:
        print(f"Continuing at epoch {start_epoch + 1}")
    else:
        print("Loaded model weights only; optimizer and epoch state were not found.")
    return start_epoch, best_val_loss, history


def save_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_loss,
    history,
    config,
    path_meta,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "history": history,
        "config": config,
        "path_metadata": path_meta,
    }
    torch.save(payload, path)


def evaluate_loss(
    model,
    loader,
    loss_fn,
    leaf_path_matrix,
    device,
    amp_enabled=False,
    batch_norm_mode="standard",
    batch_norm_freeze_affine=False,
    desc="Val loss",
):
    model.eval()
    configure_batch_norm(
        model,
        mode="frozen" if batch_norm_mode == "frozen" else "standard",
        freeze_affine=batch_norm_freeze_affine,
    )
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast_context(inputs.device.type, amp_enabled):
                features = model(inputs)
            path_labels = targets_to_path_labels(targets, leaf_path_matrix, device=device)
            loss = loss_fn(features.float(), path_labels)
            total_loss += float(loss.item())
            steps += 1
    return total_loss / max(steps, 1)


def collect_features_and_targets(model, loader, device, desc, amp_enabled=False):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for inputs, batch_targets in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            with autocast_context(inputs.device.type, amp_enabled):
                feats = model(inputs)
            features.append(feats.detach().cpu())
            targets.append(batch_targets.long().cpu())
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def compute_depthwise_accuracy(
    model,
    train_loader,
    val_loader,
    hierarchy,
    train_classes,
    leaf_path_matrix,
    covariance_type,
    density_eps,
    score_type,
    temperature,
    kappa,
    device,
    amp_enabled=False,
    score_batch_size=None,
):
    train_features, train_targets = collect_features_and_targets(
        model,
        train_loader,
        device,
        desc="Depth eval train features",
        amp_enabled=amp_enabled,
    )
    val_features, val_targets = collect_features_and_targets(
        model,
        val_loader,
        device,
        desc="Depth eval val features",
        amp_enabled=amp_enabled,
    )

    density = fit_node_distributions(
        train_features.float(),
        train_targets.long(),
        hierarchy,
        train_classes,
        covariance_type=covariance_type,
        eps=density_eps,
    )
    score_batch_size = int(score_batch_size or val_features.shape[0])
    if score_batch_size <= 0:
        raise ValueError("validation.score_batch_size must be a positive integer.")

    score_kwargs = {
        "means": density["means"].float(),
        "variances": density.get("variances").float()
        if density.get("variances") is not None
        else None,
        "covariance_matrices": density.get("covariance_matrices").float()
        if density.get("covariance_matrices") is not None
        else None,
        "shared_covariance": density.get("shared_covariance").float()
        if density.get("shared_covariance") is not None
        else None,
        "mean_directions": density.get("mean_directions", None).float()
        if density.get("mean_directions", None) is not None
        else None,
        "covariance_type": density.get("covariance_type", covariance_type),
        "score_type": score_type,
        "kappa": kappa,
    }
    node_scores = torch.cat(
        [
            score_nodes(feature_chunk.float(), **score_kwargs).cpu()
            for feature_chunk in tqdm(
                val_features.split(score_batch_size, dim=0),
                desc="Depth eval node scoring",
            )
        ],
        dim=0,
    )
    nodes_by_depth = build_depth_maps(hierarchy)
    val_class_names = [train_classes[int(idx)] for idx in val_targets.long().tolist()]

    depth_metrics = {}
    top1_values = []
    top5_values = []

    for depth in sorted(nodes_by_depth.keys()):
        if depth == 0:
            continue
        depth_indices = nodes_by_depth[depth]
        logits = node_scores[:, depth_indices]
        depth_index_map = {node_idx: pos for pos, node_idx in enumerate(depth_indices)}
        valid_rows = []
        true_local_list = []

        for row_idx, class_name in enumerate(val_class_names):
            ancestors = hierarchy.node_ancestors[class_name].copy()
            actual_depth = len(ancestors)
            if actual_depth < depth:
                continue

            if depth == actual_depth:
                node_idx = hierarchy.id_node_list.index(class_name)
            else:
                node_idx = ancestors[depth]

            if node_idx not in depth_index_map:
                continue

            valid_rows.append(row_idx)
            true_local_list.append(depth_index_map[node_idx])

        if not valid_rows:
            continue

        valid_logits = logits[valid_rows]
        true_local = torch.tensor(true_local_list, dtype=torch.long)
        top1 = valid_logits.argmax(dim=1)
        top1_acc = (top1.cpu() == true_local).float().mean().item()
        k = min(5, valid_logits.shape[1])
        topk = torch.topk(valid_logits, k=k, dim=1).indices.cpu()
        top5_acc = (topk == true_local.unsqueeze(1)).any(dim=1).float().mean().item()

        depth_metrics[depth] = {
            "acc_top1": top1_acc,
            "acc_top5": top5_acc,
            "num_classes": len(depth_indices),
            "num_samples": len(valid_rows),
        }
        top1_values.append(top1_acc)
        top5_values.append(top5_acc)

    summary = {
        "depth_metrics": depth_metrics,
        "acc_top1_mean": sum(top1_values) / max(len(top1_values), 1),
        "acc_top5_mean": sum(top5_values) / max(len(top5_values), 1),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])
    device = config["experiment"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, experiment_dir / "resolved_config.yaml")
    tensorboard_dir = experiment_dir / "tensorboard"
    use_tensorboard = config.get("logging", {}).get("use_tensorboard", False)
    writer = SummaryWriter(log_dir=str(tensorboard_dir)) if use_tensorboard else None

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    train_ds, val_ds, _ = gen_datasets(
        dataset_cfg["datadir"],
        id_classes,
        hierarchy.ood_train_classes,
        **dataset_transform_kwargs(config),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=config["dataloader"].get("shuffle_train", True),
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"].get("pin_memory", False),
        drop_last=config["dataloader"].get("drop_last", False),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["dataloader"].get("eval_batch_size", config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"].get("pin_memory", False),
        drop_last=False,
    )
    eval_train_loader = DataLoader(
        train_ds,
        batch_size=config["dataloader"].get("eval_batch_size", config["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=config["dataloader"].get("pin_memory", False),
        drop_last=False,
    )

    use_pruned = config.get("hierarchy", {}).get("use_pruned_for_loss_labels", True)
    leaf_path_matrix, path_meta = build_leaf_path_matrix(
        hierarchy,
        train_ds.classes,
        use_pruned=use_pruned,
        hierarchy_path=dataset_cfg["hierarchy"],
    )

    model = build_backbone(config).to(device)
    model = maybe_wrap_dataparallel(model, config)
    loss_cfg = dict(config["loss"])
    loss_name = loss_cfg.pop("name")
    loss_fn = build_metric_loss(loss_name, **loss_cfg)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    training_cfg = config["training"]
    amp_requested = bool(training_cfg.get("amp", False))
    amp_enabled = amp_requested and str(device).startswith("cuda")
    if amp_requested and not amp_enabled:
        print("AMP requested but CUDA is not active; continuing without AMP.")
    amp_initial_scale = float(training_cfg.get("amp_initial_scale", 65536.0))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=amp_initial_scale,
    )

    grad_cache_cfg = training_cfg.get("gradient_cache", {})
    grad_cache_enabled = bool(grad_cache_cfg.get("enabled", False))
    batch_norm_cfg = training_cfg.get("batch_norm", {})
    batch_norm_mode = batch_norm_cfg.get(
        "mode", "chunk" if grad_cache_enabled else "standard"
    )
    batch_norm_freeze_affine = bool(batch_norm_cfg.get("freeze_affine", False))
    configure_batch_norm(
        model,
        mode=batch_norm_mode,
        freeze_affine=batch_norm_freeze_affine,
    )
    gradient_cache_config = None
    if grad_cache_enabled:
        gradient_cache_config = GradientCacheConfig(
            chunk_size=int(grad_cache_cfg.get("chunk_size", 32)),
            batch_norm_mode=batch_norm_mode,
        )
        print(
            "Gradient Cache enabled: "
            f"logical_batch={config['dataloader']['batch_size']}, "
            f"chunk_size={gradient_cache_config.chunk_size}, "
            f"batch_norm={batch_norm_mode}, amp={amp_enabled}"
        )

    epochs = config["training"]["epochs"]
    history = {"train_loss": [], "val_loss": []}
    start_epoch, best_val_loss, resume_history = load_resume_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        device,
    )
    if resume_history is not None:
        history = resume_history
    validation_cfg = config.get("validation", {})
    enable_depth_eval = validation_cfg.get("enable_depth_eval", False)
    depth_eval_every = max(int(validation_cfg.get("depth_eval_every", 1)), 1)

    for epoch in range(start_epoch, epochs):
        model.train()
        configure_batch_norm(
            model,
            mode=batch_norm_mode,
            freeze_affine=batch_norm_freeze_affine,
        )
        running_loss = 0.0
        steps = 0
        for inputs, targets in tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            path_labels = targets_to_path_labels(targets, leaf_path_matrix, device=device)
            optimizer.zero_grad(set_to_none=True)

            if grad_cache_enabled:
                loss = gradient_cache_step(
                    model,
                    inputs,
                    path_labels,
                    loss_fn,
                    gradient_cache_config,
                    amp_enabled=amp_enabled,
                    scaler=scaler,
                )
            else:
                with autocast_context(inputs.device.type, amp_enabled):
                    features = model(inputs)
                loss = loss_fn(features.float(), path_labels)
                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if amp_enabled:
                scaler.unscale_(optimizer)
            if config["training"].get("grad_clip_norm"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            running_loss += float(loss.item())
            steps += 1

        train_loss = running_loss / max(steps, 1)
        val_loss = evaluate_loss(
            model,
            val_loader,
            loss_fn,
            leaf_path_matrix,
            device,
            amp_enabled=amp_enabled,
            batch_norm_mode=batch_norm_mode,
            batch_norm_freeze_affine=batch_norm_freeze_affine,
            desc=f"Val loss {epoch+1}/{epochs}",
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        depth_eval_summary = None
        if enable_depth_eval and ((epoch + 1) % depth_eval_every == 0):
            depth_eval_summary = compute_depthwise_accuracy(
                model,
                eval_train_loader,
                val_loader,
                hierarchy,
                train_ds.classes,
                leaf_path_matrix,
                config["density"].get("covariance_type", "diag"),
                config["density"]["eps"],
                config["inference"].get("score_type", "gaussian_loglik"),
                config["inference"].get("temperature", 1.0),
                config["inference"].get("kappa", 20.0),
                device,
                amp_enabled=amp_enabled,
                score_batch_size=validation_cfg.get(
                    "score_batch_size",
                    config["dataloader"].get(
                        "eval_batch_size",
                        config["dataloader"]["batch_size"],
                    ),
                ),
            )
            history.setdefault("depth_eval", []).append({"epoch": epoch, **depth_eval_summary})

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={current_lr:.6g}"
        )

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("test/loss", val_loss, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)
            if depth_eval_summary is not None:
                writer.add_scalar("test/acc_top1_mean", depth_eval_summary["acc_top1_mean"], epoch)
                writer.add_scalar("test/acc_top5_mean", depth_eval_summary["acc_top5_mean"], epoch)
                for depth, metrics in depth_eval_summary["depth_metrics"].items():
                    writer.add_scalar(f"test/acc_top1_depth_{depth}", metrics["acc_top1"], epoch)
                    writer.add_scalar(f"test/acc_top5_depth_{depth}", metrics["acc_top5"], epoch)
            writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unwrap_model(model).state_dict(), experiment_dir / "checkpoint_backbone.pt")
            save_training_checkpoint(
                experiment_dir / "checkpoint_best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                history,
                config,
                path_meta,
            )

        save_training_checkpoint(
            experiment_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_val_loss,
            history,
            config,
            path_meta,
        )

    save_artifact(
        {
            "history": history,
            "best_val_loss": best_val_loss,
            "path_metadata": path_meta,
            "backbone": backbone_summary(config),
            "runtime": config.get("runtime", {}),
            "tensorboard_dir": str(tensorboard_dir) if use_tensorboard else None,
        },
        experiment_dir / "train_metrics.pt",
    )
    save_artifact(
        {
            "config": config,
            "best_val_loss": best_val_loss,
            "path_metadata": path_meta,
        },
        experiment_dir / "train_config.pt",
    )
    if writer is not None:
        writer.close()
        print(f"TensorBoard logs saved to: {tensorboard_dir}")


if __name__ == "__main__":
    main()
