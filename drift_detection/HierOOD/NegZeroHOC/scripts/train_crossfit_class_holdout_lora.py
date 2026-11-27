from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import (
    load_idea3_checkpoint_with_fallback,
    previous_checkpoint_path,
    save_idea3_checkpoint,
)
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.crossfit_class_holdout import (
    RemappedSubset,
    atomic_save_json,
    build_topology_holdout_manifest,
    canonical_hash,
    hierarchy_topology_record,
    stratified_retained_image_split,
    tensor_partitions_hash,
    validate_topology_holdout_manifest,
)
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.image_metric import (
    PKBatchSampler,
    batch_hard_hierarchical_triplet_loss,
    class_tree_distance_matrix,
    cosine_proxy_loss,
    supervised_contrastive_loss,
)
from negzerohoc.output_layout import experiment_dir
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    seed_data_loader_worker,
)
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_enabled,
    set_vision_lora_train_mode,
    vision_lora_parameters,
    vision_lora_state_dict,
)
from scripts.train_idea3_joint_vision_lora import (
    autocast_context,
    build_transforms,
    load_clip_and_tokenizer,
    make_grad_scaler,
    make_loader,
)
from scripts.train_image_metric_vision_lora import (
    capture_rng_state,
    clone_state,
    encode_base_features,
    evaluate_proxy,
    initialize_proxies,
    next_epoch_from_training_state,
    restore_training_components,
)
from scripts.train_image_metric_vision_lora import (
    load_config as load_image_metric_config,
)


CHECKPOINT_STAGE = "crossfit_class_holdout_image_metric_lora"
TRIPLET_DISTANCE_SOURCE = (
    "fold_specific_released_prohoc_retained_only_unary_pruned"
)


def load_config(path: str | Path) -> Namespace:
    args = load_image_metric_config(path)
    cfg = load_yaml_config(path)
    crossfit = cfg.get("crossfit_class_holdout", {})
    folds = tuple(int(value) for value in crossfit.get("folds", []))
    num_folds = int(crossfit.get("num_folds", 4))
    if not folds:
        raise ValueError("crossfit_class_holdout.folds is required")
    if len(folds) != len(set(folds)) or any(
        fold < 0 or fold >= num_folds for fold in folds
    ):
        raise ValueError("Configured cross-fit folds are invalid")
    fractions = tuple(
        float(value)
        for value in crossfit.get(
            "retained_split_fractions", [0.6, 0.2, 0.2]
        )
    )
    root = Path(crossfit.get(
        "fold_output_root",
        experiment_dir(args.output_root, args.experiment_name),
    ))
    manifest_path = Path(crossfit.get(
        "manifest_path",
        root / "manifests" / f"folds-{'-'.join(map(str, folds))}.json",
    ))
    args.crossfit_folds = folds
    args.crossfit_num_folds = num_folds
    args.crossfit_requested_fold_size = int(
        crossfit.get("requested_fold_size", 16)
    )
    args.crossfit_manifest_seed = int(
        crossfit.get("manifest_seed", args.seed)
    )
    args.retained_split_fractions = fractions
    args.fold_output_root = str(root)
    args.manifest_path = str(manifest_path)
    args.resume_enabled = bool(
        crossfit.get("resume", {}).get("enabled", True)
    )
    return args


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def fold_paths(args, fold: int) -> dict[str, Path]:
    root = Path(args.fold_output_root) / f"fold-{fold}"
    return {
        "best": root / "checkpoints" / "best.pt",
        "last": root / "checkpoints" / "last.pt",
        "diagnostics": root / "diagnostics" / "training.json",
    }


def fold_seed(args, fold: int) -> int:
    return int(args.seed) + 1_000_003 * int(fold)


def build_fold_metric_topology(
    args,
    full_hierarchy,
    retained_classes: list[str],
):
    """Rebuild released ProHOC topology after removing heldout leaves."""
    from negzerohoc.prohoc_compat.hierarchy import Hierarchy

    hierarchy_path = Path(args.hierarchy)
    if not hierarchy_path.is_absolute():
        hierarchy_path = REPO_ROOT / hierarchy_path
    fold_hierarchy = Hierarchy(
        list(retained_classes), str(hierarchy_path)
    )
    class_node_indices = fold_hierarchy.gen_ds2node_map(
        retained_classes
    )
    class_nodes = [
        fold_hierarchy.id_node_list[int(index)]
        for index in class_node_indices.tolist()
    ]
    distances = class_tree_distance_matrix(
        fold_hierarchy, class_nodes
    )
    full_record = hierarchy_topology_record(full_hierarchy)
    fold_record = hierarchy_topology_record(fold_hierarchy)
    if full_record["topology_hash"] == fold_record["topology_hash"]:
        raise RuntimeError(
            "Fold hierarchy unexpectedly equals the full 80-ID hierarchy"
        )
    provenance = {
        "triplet_distance_source": TRIPLET_DISTANCE_SOURCE,
        "full_hierarchy_role": "holdout_manifest_only",
        "fold_hierarchy_role": (
            "retained_class_mapping_and_triplet_distance"
        ),
        "released_prohoc_rebuilt_from_retained_classes": True,
        "heldout_topology_excluded_from_triplet_distance": True,
        "full_hierarchy": full_record,
        "fold_hierarchy": fold_record,
    }
    return fold_hierarchy, distances, provenance


def fold_resume_signature(
    args,
    *,
    fold: int,
    manifest_hash: str,
    split_hash: str,
    retained_classes: list[str],
    fold_hierarchy_hash: str,
) -> dict:
    return {
        "version": 1,
        "stage": CHECKPOINT_STAGE,
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "datadir": args.datadir,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
        "clip_model": args.clip_model,
        "vision_lora": args.vision_lora,
        "fold": int(fold),
        "fold_seed": fold_seed(args, fold),
        "manifest_hash": manifest_hash,
        "split_hash": split_hash,
        "fold_hierarchy_hash": fold_hierarchy_hash,
        "triplet_distance_source": TRIPLET_DISTANCE_SOURCE,
        "retained_classes": list(retained_classes),
        "deterministic": args.deterministic,
        "num_workers": args.num_workers,
        "augmentation": args.augmentation,
        "epochs": args.epochs,
        "classes_per_batch": args.classes_per_batch,
        "examples_per_class": args.examples_per_class,
        "eval_batch_size": args.eval_batch_size,
        "lora_lr": args.lora_lr,
        "proxy_lr": args.proxy_lr,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_clip_norm": args.gradient_clip_norm,
        "supcon_temperature": args.supcon_temperature,
        "proxy_temperature": args.proxy_temperature,
        "proxy_margin": args.proxy_margin,
        "triplet_base_margin": args.triplet_base_margin,
        "triplet_hierarchy_margin": args.triplet_hierarchy_margin,
        "lambda_supcon": args.lambda_supcon,
        "lambda_triplet": args.lambda_triplet,
        "lambda_proxy": args.lambda_proxy,
        "lambda_retention": args.lambda_retention,
        "validation_every_n_epochs": args.validation_every_n_epochs,
        "retained_split_fractions": args.retained_split_fractions,
    }


def make_fold_training_state(
    signature,
    *,
    epoch,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    device,
    history,
    best_epoch,
    best_bacc,
    best_lora_state,
    best_proxy_state,
    training_loop_complete,
) -> dict:
    return {
        "version": 1,
        "epoch": int(epoch),
        "training_loop_complete": bool(training_loop_complete),
        "resume_signature": signature,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(train_loader, device),
        "sampler_state_dict": None,
        "history": list(history),
        "best_epoch": best_epoch,
        "best_bacc": (
            float(best_bacc) if math.isfinite(best_bacc) else None
        ),
        "best_lora_state_dict": best_lora_state,
        "best_proxy_state": best_proxy_state,
    }


def save_fold_checkpoint(
    args,
    path: Path,
    lora_cfg,
    clip_model,
    proxies,
    retained_classes,
    metrics,
    *,
    manifest,
    fold_record,
    split_indices,
    split_hash,
    hierarchy_provenance,
    training_state=None,
):
    return save_idea3_checkpoint(
        path,
        stage=CHECKPOINT_STAGE,
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config={},
        vision_lora_config=lora_cfg.to_dict(),
        vision_lora_state_dict=vision_lora_state_dict(clip_model),
        metrics=metrics,
        args=vars(args),
        training_state=training_state,
        extra_payload={
            "metric_proxies": proxies.detach().cpu().clone(),
            "metric_proxy_classes": list(retained_classes),
            "crossfit_manifest": manifest,
            "crossfit_manifest_hash": manifest["manifest_hash"],
            "crossfit_fold": fold_record,
            "crossfit_split_indices": {
                name: indices.detach().long().cpu().clone()
                for name, indices in split_indices.items()
            },
            "crossfit_split_hash": split_hash,
            "crossfit_hierarchy_provenance": hierarchy_provenance,
        },
    )


def save_or_validate_manifest(
    args,
    hierarchy,
    classes: list[str],
    manifest: dict,
) -> None:
    path = Path(args.manifest_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as source:
            saved = json.load(source)
        validate_topology_holdout_manifest(
            hierarchy, classes, saved
        )
        if saved != manifest:
            raise ValueError(
                f"Existing cross-fit manifest differs: {path}"
            )
        return
    atomic_save_json(path, manifest)


def build_train_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import (
        SubsetImageFolder,
        get_id_classes,
    )

    train_transform, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    augmented = SubsetImageFolder(
        Path(args.datadir) / "train",
        id_classes,
        transform=train_transform,
    )
    evaluation = SubsetImageFolder(
        Path(args.datadir) / "train",
        id_classes,
        transform=eval_transform,
    )
    if (
        augmented.classes != evaluation.classes
        or augmented.targets != evaluation.targets
    ):
        raise RuntimeError("Augmented/eval train dataset ordering differs")
    hierarchy.gen_ds2node_map(augmented.classes)
    return augmented, evaluation


def fold_partitions(
    targets: torch.Tensor,
    fold_record: dict,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[dict[str, torch.Tensor], list[int]]:
    heldout = sorted(
        int(value)
        for value in fold_record["heldout_original_class_indices"]
    )
    heldout_set = set(heldout)
    retained = sorted(
        set(targets.detach().long().cpu().tolist()) - heldout_set
    )
    retained_partitions = stratified_retained_image_split(
        targets,
        retained,
        fractions=fractions,
        seed=seed,
    )
    heldout_mask = torch.zeros_like(targets, dtype=torch.bool)
    for target in heldout:
        heldout_mask |= targets == target
    partitions = {
        **retained_partitions,
        "heldout_query": torch.nonzero(
            heldout_mask, as_tuple=False
        ).flatten(),
    }
    return partitions, retained


def train_fold(
    args,
    hierarchy,
    augmented_dataset,
    evaluation_dataset,
    manifest,
    fold: int,
    device: str,
) -> dict:
    configure_reproducibility(
        fold_seed(args, fold), deterministic=args.deterministic
    )
    fold_record = manifest["folds"][fold]
    targets = torch.tensor(
        augmented_dataset.targets, dtype=torch.long
    )
    partitions, retained_original_targets = fold_partitions(
        targets,
        fold_record,
        seed=fold_seed(args, fold),
        fractions=args.retained_split_fractions,
    )
    split_hash = tensor_partitions_hash(partitions)
    retained_classes = [
        augmented_dataset.classes[target]
        for target in retained_original_targets
    ]
    original_to_compact = {
        original: compact
        for compact, original in enumerate(retained_original_targets)
    }
    representation_dataset = RemappedSubset(
        augmented_dataset,
        partitions["representation_train"],
        original_to_compact,
        retained_classes,
    )
    proxy_dataset = RemappedSubset(
        evaluation_dataset,
        partitions["representation_train"],
        original_to_compact,
        retained_classes,
    )
    selection_dataset = RemappedSubset(
        evaluation_dataset,
        partitions["model_selection"],
        original_to_compact,
        retained_classes,
    )
    if args.classes_per_batch > len(retained_classes):
        raise ValueError(
            "classes_per_batch exceeds retained class count"
        )
    batch_sampler = PKBatchSampler(
        representation_dataset.targets,
        classes_per_batch=args.classes_per_batch,
        examples_per_class=args.examples_per_class,
        seed=fold_seed(args, fold),
    )
    train_loader = DataLoader(
        representation_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(fold_seed(args, fold)),
        worker_init_fn=seed_data_loader_worker,
    )
    proxy_init_loader = make_loader(
        proxy_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=fold_seed(args, fold),
    )
    selection_loader = make_loader(
        selection_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=fold_seed(args, fold),
    )
    fold_hierarchy, class_distances, hierarchy_provenance = (
        build_fold_metric_topology(
            args, hierarchy, retained_classes
        )
    )
    signature = fold_resume_signature(
        args,
        fold=fold,
        manifest_hash=manifest["manifest_hash"],
        split_hash=split_hash,
        retained_classes=retained_classes,
        fold_hierarchy_hash=hierarchy_provenance[
            "fold_hierarchy"
        ]["topology_hash"],
    )
    paths = fold_paths(args, fold)

    # Loading here, inside the fold, guarantees every fold starts from the
    # exact base CLIP checkpoint rather than the preceding fold's LoRA state.
    clip_model, _ = load_clip_and_tokenizer(args, device)
    if args.gradient_checkpointing and hasattr(
        clip_model, "gradient_checkpointing_enable"
    ):
        clip_model.gradient_checkpointing_enable()
    lora_cfg = VisionLoRAConfig.from_dict(args.vision_lora)
    replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    lora_params = vision_lora_parameters(clip_model)
    feature_dim = int(clip_model.config.projection_dim)

    resume_payload = None
    loaded_resume_path = None
    previous_path = previous_checkpoint_path(paths["last"])
    if args.resume_enabled and (
        paths["last"].exists() or previous_path.exists()
    ):
        resume_payload, loaded_resume_path = (
            load_idea3_checkpoint_with_fallback(
                paths["last"], map_location="cpu"
            )
        )
        if resume_payload.get("stage") != CHECKPOINT_STAGE:
            raise ValueError("Cross-fit resume checkpoint stage mismatch")
        saved_signature = resume_payload["training_state"].get(
            "resume_signature"
        )
        if saved_signature != signature:
            raise ValueError(
                "Cross-fit resume signature mismatch: "
                f"saved={saved_signature}, current={signature}"
            )
        if resume_payload.get("crossfit_manifest_hash") != (
            manifest["manifest_hash"]
        ):
            raise ValueError("Cross-fit resume manifest hash mismatch")
        if resume_payload.get("crossfit_split_hash") != split_hash:
            raise ValueError("Cross-fit resume image split hash mismatch")
        saved_provenance = resume_payload.get(
            "crossfit_hierarchy_provenance"
        )
        if saved_provenance != hierarchy_provenance:
            raise ValueError(
                "Cross-fit resume fold hierarchy provenance mismatch"
            )
        if resume_payload.get("metric_proxy_classes") != retained_classes:
            raise ValueError("Cross-fit resume proxy classes mismatch")
        saved_proxies = resume_payload.get("metric_proxies")
        expected_shape = (len(retained_classes), feature_dim)
        if (
            not isinstance(saved_proxies, torch.Tensor)
            or tuple(saved_proxies.shape) != expected_shape
        ):
            raise ValueError(
                "Cross-fit resume proxy shape mismatch: "
                f"expected={expected_shape}"
            )
        load_vision_lora_state_dict(
            clip_model, resume_payload["vision_lora_state_dict"]
        )
        initial_proxies = saved_proxies.float().to(device)
    else:
        if args.resume_enabled:
            print(
                f"fold {fold}: no resume checkpoint; starting from base CLIP"
            )
        initial_proxies = initialize_proxies(
            args,
            clip_model,
            proxy_init_loader,
            len(retained_classes),
            feature_dim,
            device,
        )
    proxies = nn.Parameter(initial_proxies)
    class_distances = class_distances.to(device)
    optimizer = torch.optim.AdamW([
        {
            "params": lora_params,
            "lr": args.lora_lr,
            "group_name": "vision_lora",
        },
        {
            "params": [proxies],
            "lr": args.proxy_lr,
            "group_name": "metric_proxy",
        },
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = make_grad_scaler(
        device.startswith("cuda")
        and args.precision in {"fp16", "float16"}
    )
    trainable_params = lora_params + [proxies]
    print(
        f"cross-fit fold {fold}: heldout={len(fold_record['heldout_leaves'])}, "
        f"retained={len(retained_classes)}, "
        f"rep/model/known/heldout="
        f"{len(partitions['representation_train'])}/"
        f"{len(partitions['model_selection'])}/"
        f"{len(partitions['known_query'])}/"
        f"{len(partitions['heldout_query'])}, "
        f"P={args.classes_per_batch}, K={args.examples_per_class}, "
        f"LoRA modules={len(replaced_modules)}"
    )

    history = []
    best_bacc = float("-inf")
    best_epoch = None
    best_lora_state = None
    best_proxy_state = None
    start_epoch = 1
    if resume_payload is not None:
        training_state = resume_payload["training_state"]
        restore_training_components(
            training_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
        )
        history = list(training_state.get("history", []))
        best_epoch = training_state.get("best_epoch")
        saved_best_bacc = training_state.get("best_bacc")
        best_bacc = (
            float(saved_best_bacc)
            if saved_best_bacc is not None
            else float("-inf")
        )
        best_lora_state = training_state.get(
            "best_lora_state_dict"
        )
        best_proxy_state = training_state.get("best_proxy_state")
        start_epoch = next_epoch_from_training_state(
            training_state, args.epochs
        )
        print(
            f"fold {fold}: resumed {loaded_resume_path}; "
            f"next={start_epoch if start_epoch <= args.epochs else 'finalize'}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        batch_sampler.set_epoch(epoch)
        clip_model.eval()
        set_vision_lora_enabled(clip_model, True)
        set_vision_lora_train_mode(clip_model, True)
        totals: dict[str, float] = {}
        steps = 0
        iterator = (
            tqdm(
                train_loader,
                desc=f"crossfit f{fold} epoch {epoch}/{args.epochs}",
                leave=False,
            )
            if tqdm else train_loader
        )
        for images, batch_targets in iterator:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            base_features = encode_base_features(
                args, clip_model, images, device
            )
            with autocast_context(args, device):
                tuned_features = clip_model.get_image_features(
                    pixel_values=images
                )
            supcon_loss, supcon_stats = supervised_contrastive_loss(
                tuned_features,
                batch_targets,
                temperature=args.supcon_temperature,
            )
            triplet_loss, triplet_stats = (
                batch_hard_hierarchical_triplet_loss(
                    tuned_features,
                    batch_targets,
                    class_distances,
                    base_margin=args.triplet_base_margin,
                    hierarchy_margin=args.triplet_hierarchy_margin,
                )
            )
            proxy_loss, proxy_stats = cosine_proxy_loss(
                tuned_features,
                proxies,
                batch_targets,
                temperature=args.proxy_temperature,
                margin=args.proxy_margin,
            )
            retention_loss = (
                1.0 - torch.nn.functional.cosine_similarity(
                    tuned_features.float(),
                    base_features.detach().float(),
                    dim=-1,
                )
            ).mean()
            loss = (
                args.lambda_supcon * supcon_loss
                + args.lambda_triplet * triplet_loss
                + args.lambda_proxy * proxy_loss
                + args.lambda_retention * retention_loss
            )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.gradient_clip_norm
                )
                optimizer.step()
            stats = {
                **supcon_stats,
                **triplet_stats,
                **proxy_stats,
                "loss": float(loss.detach().cpu()),
                "retention_loss": float(retention_loss.detach().cpu()),
            }
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            steps += 1
        scheduler.step()
        epoch_stats = {
            key: value / max(1, steps)
            for key, value in totals.items()
        }
        epoch_stats.update({"epoch": epoch, "steps": steps})
        for group in optimizer.param_groups:
            epoch_stats[f"{group['group_name']}_lr"] = group["lr"]
        if (
            epoch % args.validation_every_n_epochs == 0
            or epoch == args.epochs
        ):
            validation = evaluate_proxy(
                args,
                clip_model,
                proxies,
                selection_loader,
                device,
            )
            epoch_stats.update({
                f"val_{key}": value
                for key, value in validation.items()
            })
            if validation["balanced_acc"] > best_bacc:
                best_bacc = validation["balanced_acc"]
                best_epoch = epoch
                best_lora_state = clone_state(
                    vision_lora_state_dict(clip_model)
                )
                best_proxy_state = proxies.detach().cpu().clone()
                save_fold_checkpoint(
                    args,
                    paths["best"],
                    lora_cfg,
                    clip_model,
                    proxies,
                    retained_classes,
                    {
                        "train_history": history + [epoch_stats],
                        "best_validation": {
                            "epoch": epoch,
                            **validation,
                        },
                    },
                    manifest=manifest,
                    fold_record=fold_record,
                    split_indices=partitions,
                    split_hash=split_hash,
                    hierarchy_provenance=hierarchy_provenance,
                )
        history.append(epoch_stats)
        training_state = make_fold_training_state(
            signature,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            history=history,
            best_epoch=best_epoch,
            best_bacc=best_bacc,
            best_lora_state=best_lora_state,
            best_proxy_state=best_proxy_state,
            training_loop_complete=epoch >= args.epochs,
        )
        save_fold_checkpoint(
            args,
            paths["last"],
            lora_cfg,
            clip_model,
            proxies,
            retained_classes,
            {
                "train_history": history,
                "best_validation": {
                    "epoch": best_epoch,
                    "balanced_acc": (
                        best_bacc if math.isfinite(best_bacc) else None
                    ),
                },
            },
            manifest=manifest,
            fold_record=fold_record,
            split_indices=partitions,
            split_hash=split_hash,
            hierarchy_provenance=hierarchy_provenance,
            training_state=training_state,
        )
        print(
            f"fold {fold} epoch {epoch}: "
            f"loss={epoch_stats['loss']:.6f}, "
            f"val_bacc={epoch_stats.get('val_balanced_acc', float('nan')):.6f}"
        )

    if best_lora_state is None or best_proxy_state is None:
        raise RuntimeError(
            f"Cross-fit fold {fold} produced no validation checkpoint"
        )
    load_vision_lora_state_dict(clip_model, best_lora_state)
    with torch.no_grad():
        proxies.copy_(best_proxy_state.to(device))
    final_selection = evaluate_proxy(
        args, clip_model, proxies, selection_loader, device
    )
    metrics = {
        "train_history": history,
        "best_validation": {
            "epoch": best_epoch,
            **final_selection,
        },
        "selection_split": "retained_model_selection",
        "fold": int(fold),
        "fold_seed": fold_seed(args, fold),
        "manifest_hash": manifest["manifest_hash"],
        "split_hash": split_hash,
        "split_sizes": {
            name: int(indices.numel())
            for name, indices in partitions.items()
        },
        "heldout_classes": fold_record["heldout_leaves"],
        "retained_classes": retained_classes,
        "hierarchy_provenance": hierarchy_provenance,
        "triplet_distance_source": TRIPLET_DISTANCE_SOURCE,
        "used_heldout_class_images_for_representation_training": False,
        "used_heldout_class_images_for_proxy_initialization": False,
        "used_heldout_class_images_for_model_selection": False,
        "used_known_query_for_training_or_selection": False,
        "used_official_test_for_training_or_selection": False,
    }
    save_fold_checkpoint(
        args,
        paths["best"],
        lora_cfg,
        clip_model,
        proxies,
        retained_classes,
        metrics,
        manifest=manifest,
        fold_record=fold_record,
        split_indices=partitions,
        split_hash=split_hash,
        hierarchy_provenance=hierarchy_provenance,
    )
    atomic_save_json(paths["diagnostics"], metrics)
    print(
        f"fold {fold} complete: best_epoch={best_epoch}, "
        f"selection_BAcc={final_selection['balanced_acc']:.6f}, "
        f"saved={paths['best']}"
    )
    del (
        proxies,
        optimizer,
        scheduler,
        scaler,
        clip_model,
        fold_hierarchy,
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    args = parse_args()
    if not args.datadir:
        raise ValueError("Missing dataset.datadir")
    if args.num_workers != 0:
        raise ValueError(
            "Exact cross-fit resume requires dataloader.num_workers=0"
        )
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    augmented_dataset, evaluation_dataset = build_train_datasets(
        args, hierarchy
    )
    manifest = build_topology_holdout_manifest(
        hierarchy,
        augmented_dataset.classes,
        num_folds=args.crossfit_num_folds,
        requested_fold_size=args.crossfit_requested_fold_size,
        seed=args.crossfit_manifest_seed,
    )
    save_or_validate_manifest(
        args, hierarchy, augmented_dataset.classes, manifest
    )
    print(
        "cross-fit manifest: "
        f"requested={manifest['requested_fold_size']}, "
        f"feasible={manifest['feasible_fold_sizes']}, "
        f"eligible={manifest['eligible_leaf_count']}/"
        f"{manifest['leaf_count']}, "
        f"hash={manifest['manifest_hash']}"
    )
    run_summary = {}
    for fold in args.crossfit_folds:
        run_summary[str(fold)] = train_fold(
            args,
            hierarchy,
            augmented_dataset,
            evaluation_dataset,
            manifest,
            fold,
            device,
        )
    summary_path = Path(args.fold_output_root) / "diagnostics" / (
        f"folds-{'-'.join(map(str, args.crossfit_folds))}.json"
    )
    atomic_save_json(summary_path, {
        "manifest_hash": manifest["manifest_hash"],
        "folds": list(args.crossfit_folds),
        "metrics": run_summary,
        "summary_hash": canonical_hash({
            "manifest_hash": manifest["manifest_hash"],
            "folds": list(args.crossfit_folds),
        }),
    })


if __name__ == "__main__":
    main()
