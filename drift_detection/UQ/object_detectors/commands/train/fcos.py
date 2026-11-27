import json
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

from commands.train.common import (
    active_dataset_names,
    autocast_context,
    count_total_params,
    count_trainable_params,
    make_grad_scaler,
    map_coco91_to_80,
    merge_epoch_timing,
    resolve_train_class_names,
    trainable_parameters,
    training_options,
)
from dataloaders.fcos import create_dataloader
from dataloaders.core.class_names import DATASET_CLASS_NAMES
from models.fcos import FCOSTorchObjectDetector
from models.fcos.core.structures.bounding_box import BoxList


def _resolve_class_names(config):
    names = resolve_train_class_names(config)
    if names is not None:
        return list(names)
    active = active_dataset_names(config)
    if active == ["coco"]:
        return list(DATASET_CLASS_NAMES["coco"])
    return None


def _target_to_fcos(target, source_image, processed_image, device):
    boxes = target.get("boxes")
    labels = target.get("labels")
    if boxes is None:
        boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
    else:
        boxes = boxes.to(device=device, dtype=torch.float32, non_blocking=True)
    if labels is None:
        labels = torch.zeros((0,), dtype=torch.int64, device=device)
    else:
        labels = labels.to(device=device, dtype=torch.int64, non_blocking=True)

    dataset_name = str(target.get("dataset_name", "")).lower()
    if dataset_name == "coco":
        mapped, keep = map_coco91_to_80(labels, offset=1)
        if keep.any():
            boxes = boxes[keep]
            labels = mapped.to(dtype=torch.int64, device=device)
        else:
            boxes = boxes[:0]
            labels = labels[:0]
    else:
        labels = labels + 1

    if boxes.numel():
        src_h, src_w = int(source_image.shape[-2]), int(source_image.shape[-1])
        dst_h, dst_w = int(processed_image.shape[-2]), int(processed_image.shape[-1])
        ratio_w = float(dst_w) / float(max(src_w, 1))
        ratio_h = float(dst_h) / float(max(src_h, 1))
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_h
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0.0, max=float(dst_w))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0.0, max=float(dst_h))
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (labels > 0)
        boxes = boxes[valid]
        labels = labels[valid]
    width = int(processed_image.shape[-1])
    height = int(processed_image.shape[-2])
    boxlist = BoxList(boxes, (width, height), mode="xyxy")
    boxlist.add_field("labels", labels)
    return boxlist


def _has_valid_target(boxlist):
    labels = boxlist.get_field("labels")
    return bool(boxlist.bbox.numel() and labels.numel())


def _raise_if_nonfinite_loss(loss, loss_dict):
    if torch.isfinite(loss).all():
        return
    values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
    raise FloatingPointError(f"Non-finite FCOS loss encountered: {values}")


def _build_model(config, device):
    model_cfg = config.get("model", {})
    class_names = _resolve_class_names(config)
    if not class_names:
        raise ValueError("Could not resolve FCOS class names for training.")

    weights_path = str(model_cfg.get("weights", "") or "").strip()
    use_finetune = bool(weights_path)
    detector = FCOSTorchObjectDetector(
        model_weight=weights_path if weights_path else None,
        device=str(device),
        names=class_names,
        mode="train",
        confidence=float(model_cfg.get("confidence_threshold", 0.05)),
        iou_thresh=float(model_cfg.get("iou_threshold", 0.6)),
    )
    model = detector.detector_model
    model.to(device)
    model.train()
    return detector, model, use_finetune, class_names, detector.cfg


def _freeze_feature_extractor(model):
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.rpn.head.requires_grad_(True)
    model.rpn.head.train()


def _apply_frozen_module_modes(model, freeze_feature_extractor):
    if not freeze_feature_extractor:
        return
    model.backbone.eval()
    model.rpn.head.train()


def _run_one_epoch(
    detector,
    model,
    dataloader,
    optimizer,
    device,
    train_mode=True,
    amp=False,
    scaler=None,
    freeze_feature_extractor=False,
    log_timing=False,
    grad_clip_norm=0.0,
    grad_params=None,
):
    model.train()
    _apply_frozen_module_modes(model, freeze_feature_extractor)
    timing = {"data_sec": 0.0, "target_sec": 0.0, "forward_loss_sec": 0.0, "backward_step_sec": 0.0}
    skipped_images = 0
    skipped_batches = 0
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    next_data_start = time.perf_counter()

    for images, targets in pbar:
        if log_timing:
            timing["data_sec"] += time.perf_counter() - next_data_start

        t_target = time.perf_counter()
        candidate_image_list = detector.preprocess_images(images)
        candidate_target_list = [
            _target_to_fcos(t, src_img, proc_img, device)
            for src_img, proc_img, t in zip(images, candidate_image_list, targets)
        ]
        keep_indices = [idx for idx, target in enumerate(candidate_target_list) if _has_valid_target(target)]
        if not keep_indices:
            skipped_images += len(images)
            skipped_batches += 1
            next_data_start = time.perf_counter()
            continue
        skipped_images += len(images) - len(keep_indices)
        if len(keep_indices) == len(images):
            image_list = candidate_image_list
            target_list = candidate_target_list
        else:
            image_list = [candidate_image_list[idx] for idx in keep_indices]
            target_list = [candidate_target_list[idx] for idx in keep_indices]
        if log_timing:
            timing["target_sec"] += time.perf_counter() - t_target

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            t_forward = time.perf_counter()
            with autocast_context(device, amp):
                loss_dict = model(image_list, target_list)
                loss = sum(v for v in loss_dict.values())
            _raise_if_nonfinite_loss(loss, loss_dict)
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
                    loss_dict = model(image_list, target_list)
                    loss = sum(v for v in loss_dict.values())
                _raise_if_nonfinite_loss(loss, loss_dict)
                if log_timing:
                    timing["forward_loss_sec"] += time.perf_counter() - t_forward

        loss_value = float(loss.detach().cpu().item())
        total_loss += loss_value
        total_steps += 1
        pbar.set_postfix(loss=f"{loss_value:.4f}")
        del image_list, target_list, loss_dict, loss
        next_data_start = time.perf_counter()

    mean_loss = (total_loss / total_steps) if total_steps else 0.0
    if log_timing:
        result = {key: value / max(total_steps, 1) for key, value in timing.items()}
        result["skipped_images"] = float(skipped_images)
        result["skipped_batches"] = float(skipped_batches)
        return mean_loss, result
    return mean_loss, {}


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss, class_names, cfg, save_optimizer=True):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if save_optimizer else None,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "class_names": list(class_names),
            "num_classes": int(len(class_names) + 1),
            "fcos_cfg": cfg.dump() if hasattr(cfg, "dump") else None,
            "date": datetime.now().isoformat(),
        },
        path,
    )


def run_train(config, run_dir, device, epochs, lr, weight_decay):
    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    training_cfg = config.get("training", {})
    options = training_options(config, device)

    train_loader = create_dataloader(config, split="train")
    try:
        val_loader = create_dataloader(config, split="val")
    except Exception:
        val_loader = None

    detector, model, use_finetune, class_names, cfg = _build_model(config, device)
    if options["freeze_feature_extractor"]:
        _freeze_feature_extractor(model)
    trainable_params = trainable_parameters(model)
    if not trainable_params:
        raise ValueError("No trainable FCOS parameters remain after applying freeze settings.")

    print(
        f"[train] trainable_params={count_trainable_params(model)} total_params={count_total_params(model)} "
        f"amp={options['amp']} freeze_feature_extractor={options['freeze_feature_extractor']}"
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scaler = make_grad_scaler(device, options["amp"])

    best_metric = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_loss, train_timing = _run_one_epoch(
            detector=detector,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            train_mode=True,
            amp=options["amp"],
            scaler=scaler,
            freeze_feature_extractor=options["freeze_feature_extractor"],
            log_timing=options["log_timing"],
            grad_clip_norm=options["grad_clip_norm"],
            grad_params=trainable_params,
        )
        val_loss = None
        val_timing = {}
        do_val = val_loader is not None and (epoch % options["val_interval"] == 0 or epoch == epochs)
        if do_val:
            val_loss, val_timing = _run_one_epoch(
                detector=detector,
                model=model,
                dataloader=val_loader,
                optimizer=optimizer,
                device=device,
                train_mode=False,
                amp=options["amp"],
                scaler=None,
                freeze_feature_extractor=options["freeze_feature_extractor"],
                log_timing=options["log_timing"],
                grad_clip_norm=0.0,
                grad_params=trainable_params,
            )

        metric = val_loss if val_loss is not None else train_loss
        can_update_best = val_loss is not None or val_loader is None
        if epoch % options["save_last_interval"] == 0 or epoch == epochs:
            _save_ckpt(
                weights_dir / "last.pt",
                epoch,
                model,
                optimizer,
                train_loss,
                val_loss,
                class_names,
                cfg,
                save_optimizer=options["save_optimizer"],
            )
        is_best = can_update_best and metric <= best_metric
        if is_best:
            best_metric = metric
        if options["save_best"] and is_best:
            _save_ckpt(
                weights_dir / "best.pt",
                epoch,
                model,
                optimizer,
                train_loss,
                val_loss,
                class_names,
                cfg,
                save_optimizer=options["save_optimizer"],
            )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "best_metric": float(best_metric),
        }
        if options["log_timing"]:
            row = merge_epoch_timing(row, {f"train_{k}": v for k, v in train_timing.items()})
            row = merge_epoch_timing(row, {f"val_{k}": v for k, v in val_timing.items()})
        history.append(row)
        print(
            f"[train] epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={'none' if val_loss is None else f'{val_loss:.6f}'}"
        )

    summary = {
        "mode": "train",
        "model_type": "fcos",
        "epochs": int(epochs),
        "seed": None if training_cfg.get("seed") is None else int(training_cfg.get("seed")),
        "device": str(device),
        "finetune": bool(use_finetune),
        "num_classes": int(len(class_names)),
        "num_classes_with_background": int(len(class_names) + 1),
        "class_names": list(class_names),
        "history": history,
        "best_metric": float(best_metric),
        "training_options": options,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "weights": {
            "last": str((weights_dir / "last.pt").resolve()),
            "best": str((weights_dir / "best.pt").resolve()) if options["save_best"] else None,
        },
    }
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = ["run_train"]
