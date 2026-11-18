import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from commands.train.common import (
    active_dataset_names,
    autocast_context,
    count_total_params,
    count_trainable_params,
    load_matching_state_dict,
    make_grad_scaler,
    map_coco91_to_80,
    merge_epoch_timing,
    resolve_train_class_names,
    torch_load,
    trainable_parameters,
    training_options,
)
from dataloaders.faster_rcnn import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTER_RCNN_COCO_WEIGHT = (
    PROJECT_ROOT / "models" / "faster_rcnn" / "weights" / "coco" / "fasterrcnn_resnet50_fpn_coco.pth"
)


def _ensure_default_coco_weight(path: Path = DEFAULT_FASTER_RCNN_COCO_WEIGHT) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url, str(path), progress=True)
    return path


def _default_coco_names():
    categories = list(FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta.get("categories", []))
    return categories[1:] if categories and categories[0] == "__background__" else categories


def _resolve_class_names(config):
    names = resolve_train_class_names(config)
    if names is not None:
        return list(names)
    active = active_dataset_names(config)
    if active == ["coco"]:
        return _default_coco_names()
    return None


def _target_to_faster_rcnn(target, device):
    boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).to(device=device, dtype=torch.float32)
    labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).to(device=device, dtype=torch.int64)
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
    valid = (
        (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]) & (labels > 0)
        if boxes.numel()
        else torch.zeros((0,), dtype=torch.bool, device=device)
    )
    return {"boxes": boxes[valid], "labels": labels[valid]}


def _build_model(config, device):
    model_cfg = config.get("model", {})
    pretrained = bool(model_cfg.get("pretrained", True))
    weights = None
    img_size = int(model_cfg.get("img_size", 640))
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None if weights is None else None,
        min_size=img_size,
        max_size=img_size,
        box_score_thresh=float(model_cfg.get("confidence_threshold", 0.25)),
        box_nms_thresh=float(model_cfg.get("iou_threshold", 0.45)),
        box_detections_per_img=int(model_cfg.get("max_det", 300)),
    )
    class_names = _resolve_class_names(config)
    if not class_names:
        raise ValueError("Could not resolve Faster R-CNN class names for training.")
    num_classes = len(class_names) + 1
    current_classes = int(model.roi_heads.box_predictor.cls_score.out_features)
    if current_classes != num_classes:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    weights_path = str(model_cfg.get("weights", "") or "").strip()
    use_finetune = False
    if weights_path and Path(weights_path).is_file():
        payload = torch_load(weights_path, map_location=device)
        state_dict = payload.get("model_state_dict") if isinstance(payload, dict) else None
        if state_dict is None and isinstance(payload, dict):
            state_dict = payload.get("state_dict", payload)
        load_matching_state_dict(model, state_dict)
        use_finetune = True
    elif weights_path and pretrained and Path(weights_path).name == DEFAULT_FASTER_RCNN_COCO_WEIGHT.name:
        state_dict = torch_load(_ensure_default_coco_weight(Path(weights_path)), map_location=device)
        load_matching_state_dict(model, state_dict)
    elif pretrained:
        state_dict = torch_load(_ensure_default_coco_weight(), map_location=device)
        load_matching_state_dict(model, state_dict)
    model.to(device)
    model.train()
    return model, use_finetune, class_names


def _freeze_feature_extractor(model):
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.rpn.requires_grad_(True)
    model.rpn.train()
    model.roi_heads.requires_grad_(True)
    model.roi_heads.train()


def _apply_frozen_module_modes(model, freeze_feature_extractor):
    if not freeze_feature_extractor:
        return
    model.backbone.eval()
    model.rpn.train()
    model.roi_heads.train()


def _set_norm_eval(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()


def _raise_if_nonfinite_loss(loss, loss_dict):
    if torch.isfinite(loss).all():
        return
    values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
    raise FloatingPointError(f"Non-finite Faster R-CNN loss encountered: {values}")


def _run_one_epoch(
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
    was_training = model.training
    model.train()
    _apply_frozen_module_modes(model, freeze_feature_extractor)
    if not train_mode:
        _set_norm_eval(model)

    timing = {"data_sec": 0.0, "target_sec": 0.0, "forward_loss_sec": 0.0, "backward_step_sec": 0.0}
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="train" if train_mode else "val")
    next_data_start = time.perf_counter()

    try:
        for images, targets in pbar:
            if log_timing:
                timing["data_sec"] += time.perf_counter() - next_data_start

            t_target = time.perf_counter()
            image_list = [img.to(device=device, dtype=torch.float32, non_blocking=True) for img in images]
            target_list = [_target_to_faster_rcnn(t, device) for t in targets]
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
    finally:
        if not train_mode:
            model.train(was_training)
            if was_training:
                _apply_frozen_module_modes(model, freeze_feature_extractor)

    mean_loss = (total_loss / total_steps) if total_steps else 0.0
    if log_timing:
        return mean_loss, {key: value / max(total_steps, 1) for key, value in timing.items()}
    return mean_loss, {}


def _save_ckpt(path, epoch, model, optimizer, train_loss, val_loss, class_names, save_optimizer=True):
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if save_optimizer else None,
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "class_names": list(class_names),
            "num_classes": int(len(class_names) + 1),
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
    model, use_finetune, class_names = _build_model(config, device)
    if options["freeze_feature_extractor"]:
        _freeze_feature_extractor(model)
    trainable_params = trainable_parameters(model)
    if not trainable_params:
        raise ValueError("No trainable Faster R-CNN parameters remain after applying freeze settings.")

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
        "model_type": "faster_rcnn",
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
