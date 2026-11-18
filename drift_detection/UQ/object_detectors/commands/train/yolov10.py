import json
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from commands.train.common import (
    autocast_context,
    count_total_params,
    count_trainable_params,
    make_preprocess_cache,
    make_grad_scaler,
    merge_epoch_timing,
    prepare_yolo_batch,
    resolve_train_class_names,
    to_yolo_targets,
    trainable_parameters,
    training_options,
)
from dataloaders.yolov10 import create_dataloader
from losses.registry import build_loss
from models.yolov10 import YOLOV10TorchObjectDetector

def _build_model(config, device):
    model_cfg = config.get("model", {})
    names = resolve_train_class_names(config)
    detector = YOLOV10TorchObjectDetector(
        model_weight=model_cfg.get("weights") or None,
        device=device,
        img_size=(int(model_cfg.get("img_size", 640)), int(model_cfg.get("img_size", 640)))
        if isinstance(model_cfg.get("img_size", 640), int)
        else tuple(model_cfg.get("img_size", [640, 640])),
        names=names,
        mode="train",
        confidence=float(model_cfg.get("confidence_threshold", 0.25)),
        max_det=int(model_cfg.get("max_det", 300)),
    )
    model = detector.model
    model.architecture = detector.architecture
    loss_cfg = config.get("loss", {})
    model.args.box = float(loss_cfg.get("box", 7.5))
    model.args.cls = float(loss_cfg.get("cls", 0.5))
    model.args.dfl = float(loss_cfg.get("dfl", 1.5))
    model.requires_grad_(True)
    model.float().to(device)
    model.train()
    return model, build_loss(model_cfg.get("type", "yolov10"), model, config)


def _freeze_feature_extractor(model):
    for module in model.model[:-1]:
        module.requires_grad_(False)
        module.eval()
    model.model[-1].requires_grad_(True)
    model.model[-1].train()


def _apply_frozen_module_modes(model, freeze_feature_extractor):
    if not freeze_feature_extractor:
        return
    for module in model.model[:-1]:
        module.eval()
    model.model[-1].train()


def _targets_to_batch(yolo_targets):
    if yolo_targets.numel() == 0:
        return {
            "batch_idx": yolo_targets.new_zeros((0,)),
            "cls": yolo_targets.new_zeros((0,)),
            "bboxes": yolo_targets.new_zeros((0, 4)),
        }
    return {
        "batch_idx": yolo_targets[:, 0],
        "cls": yolo_targets[:, 1],
        "bboxes": yolo_targets[:, 2:6],
    }


def _raise_if_nonfinite_loss(loss, loss_items):
    if torch.isfinite(loss).all():
        return
    if torch.is_tensor(loss_items):
        flat_items = loss_items.detach().float().reshape(-1).cpu().tolist()
        values = {
            f"loss_item_{idx}": float(value)
            for idx, value in enumerate(flat_items)
        }
    else:
        values = {"loss_items": str(loss_items)}
    raise FloatingPointError(f"Non-finite YOLOv10 loss encountered: {values}")


def _forward_loss_preds(model, infer_batch, freeze_feature_extractor=False):
    was_training = model.training
    model.train()
    _apply_frozen_module_modes(model, freeze_feature_extractor)
    try:
        return model(infer_batch)
    finally:
        model.train(was_training)
        if was_training:
            _apply_frozen_module_modes(model, freeze_feature_extractor)


def _run_one_epoch(
    model,
    dataloader,
    loss_fn,
    optimizer,
    img_size,
    device,
    train_mode=True,
    amp=False,
    scaler=None,
    freeze_feature_extractor=False,
    log_timing=False,
    grad_clip_norm=0.0,
    grad_params=None,
    preprocess_cache=None,
    preprocess_cache_size=0,
):
    if train_mode:
        model.train()
        _apply_frozen_module_modes(model, freeze_feature_extractor)
    else:
        model.eval()
    timing = {"data_sec": 0.0, "target_sec": 0.0, "forward_loss_sec": 0.0, "backward_step_sec": 0.0}
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    next_data_start = time.perf_counter()
    for images, targets in pbar:
        if log_timing:
            timing["data_sec"] += time.perf_counter() - next_data_start
        t_target = time.perf_counter()
        infer_batch, ratios, pads = prepare_yolo_batch(
            images,
            targets,
            img_size=img_size,
            device=device,
            preprocess_cache=preprocess_cache,
            preprocess_cache_size=preprocess_cache_size,
        )
        yolo_targets = to_yolo_targets(targets, ratios, pads, img_h=int(infer_batch.shape[2]), img_w=int(infer_batch.shape[3]), device=device)
        batch = _targets_to_batch(yolo_targets)
        if log_timing:
            timing["target_sec"] += time.perf_counter() - t_target
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            t_forward = time.perf_counter()
            with autocast_context(device, amp):
                preds = model(infer_batch)
                loss, _loss_items = loss_fn(preds, batch)
            _raise_if_nonfinite_loss(loss, _loss_items)
            if log_timing:
                timing["forward_loss_sec"] += time.perf_counter() - t_forward
            t_backward = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(grad_params, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(grad_params, grad_clip_norm)
                optimizer.step()
            if log_timing:
                timing["backward_step_sec"] += time.perf_counter() - t_backward
        else:
            with torch.no_grad():
                t_forward = time.perf_counter()
                with autocast_context(device, amp):
                    preds = _forward_loss_preds(model, infer_batch, freeze_feature_extractor)
                    loss, _loss_items = loss_fn(preds, batch)
                _raise_if_nonfinite_loss(loss, _loss_items)
                if log_timing:
                    timing["forward_loss_sec"] += time.perf_counter() - t_forward
        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        total_steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        del infer_batch, yolo_targets, batch, preds, loss
        next_data_start = time.perf_counter()
    mean_loss = total_loss / total_steps if total_steps else 0.0
    if log_timing:
        return mean_loss, {key: value / max(total_steps, 1) for key, value in timing.items()}
    return mean_loss, {}


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss, config, save_optimizer=True):
    model_cfg = config.get("model", {})
    torch.save(
        {
            "epoch": int(epoch),
            "model_type": "yolov10",
            "architecture": getattr(model, "architecture", None),
            "num_classes": int(getattr(model, "nc", 0)),
            "names": list(getattr(model, "names", [])),
            "img_size": model_cfg.get("img_size", 640),
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if save_optimizer else None,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "date": datetime.now().isoformat(),
        },
        path,
    )


def run_train(config, run_dir, device, epochs, lr, weight_decay):
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    options = training_options(config, device)
    train_loader = create_dataloader(config, split="train")
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None
    model, loss_fn = _build_model(config, device)
    if options["freeze_feature_extractor"]:
        _freeze_feature_extractor(model)
    trainable_params = trainable_parameters(model)
    if not trainable_params:
        raise ValueError("No trainable YOLOv10 parameters remain after applying freeze settings.")
    print(
        f"[train] trainable_params={count_trainable_params(model)} total_params={count_total_params(model)} "
        f"amp={options['amp']} freeze_feature_extractor={options['freeze_feature_extractor']}"
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(device, options["amp"])
    img_size = config.get("model", {}).get("img_size", 640)
    preprocess_cache_size = int(options["preprocess_cache_size"])
    preprocess_cache = make_preprocess_cache(preprocess_cache_size)
    best_metric = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_timing = _run_one_epoch(
            model, train_loader, loss_fn, optimizer, img_size, device, True, options["amp"], scaler,
            options["freeze_feature_extractor"], options["log_timing"], options["grad_clip_norm"],
            trainable_params, preprocess_cache, preprocess_cache_size,
        )
        val_loss = None
        val_timing = {}
        do_val = val_loader is not None and (epoch % options["val_interval"] == 0 or epoch == epochs)
        if do_val:
            val_loss, val_timing = _run_one_epoch(
                model, val_loader, loss_fn, optimizer, img_size, device, False, options["amp"], None,
                options["freeze_feature_extractor"], options["log_timing"], 0.0,
                None, preprocess_cache, preprocess_cache_size,
            )
        metric = val_loss if val_loss is not None else train_loss
        can_update_best = val_loss is not None or val_loader is None
        if epoch % options["save_last_interval"] == 0 or epoch == epochs:
            _save_ckpt(weights_dir / "last.pt", epoch, model, optimizer, train_loss, val_loss, config, options["save_optimizer"])
        is_best = can_update_best and metric <= best_metric
        if is_best:
            best_metric = metric
        if options["save_best"] and is_best:
            _save_ckpt(weights_dir / "best.pt", epoch, model, optimizer, train_loss, val_loss, config, options["save_optimizer"])
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
        }
        if options["log_timing"]:
            row["train_timing"] = train_timing
            row["val_timing"] = val_timing
            row["timing"] = merge_epoch_timing(train_timing, val_timing)
        history.append(row)
        print(f"[train] epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss if val_loss is not None else 'NA'}")
    summary = {
        "model_type": "yolov10",
        "epochs": int(epochs),
        "best_metric": float(best_metric),
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "amp": bool(options["amp"]),
        "freeze_feature_extractor": bool(options["freeze_feature_extractor"]),
        "grad_clip_norm": float(options["grad_clip_norm"]),
        "preprocess_cache_size": int(preprocess_cache_size),
        "history": history,
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["run_train"]
