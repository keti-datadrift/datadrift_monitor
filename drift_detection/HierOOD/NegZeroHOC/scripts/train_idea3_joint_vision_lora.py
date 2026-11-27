from __future__ import annotations

import argparse
from argparse import Namespace
import gc
import math
import sys
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
    load_idea3_checkpoint,
    load_idea3_checkpoint_with_fallback,
    previous_checkpoint_path,
    save_idea3_checkpoint,
)
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.grad_cache import (
    build_parallel_image_encoder,
    grad_cache_forward_backward,
    is_cuda_out_of_memory,
)
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.image_metric import (
    GloballyBalancedPKBatchSampler,
    HierarchyPKBatchSampler,
    MSLossPKBatchSampler,
)
from negzerohoc.losses import dual_weihims_positive_loss
from negzerohoc.metric_terminal import (
    build_metric_terminal_specs,
    predict_features_metric_terminal,
)
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
    seed_data_loader_worker,
)
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_train_mode,
    vision_lora_parameters,
    vision_lora_state_dict,
)
from scripts.train_idea3_positive_prompts import (
    backward_sparse_path_bottleneck_streaming,
    build_depth_prompt_metadata,
    build_depth_prompt_sets,
    compute_sparse_path_positive_loss,
    encode_depth_prompts,
)


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    prompt_cfg = cfg.get("prompt", {})
    dataloader_cfg = cfg.get("dataloader", {})
    augmentation_cfg = cfg.get("augmentation", {})
    lora_cfg = cfg.get("vision_lora", {})
    train_cfg = cfg.get("joint_training", {})
    loss_cfg = train_cfg.get("loss", {})
    grad_cache_cfg = train_cfg.get("grad_cache", {})
    validation_cfg = train_cfg.get("validation", {})
    early_stopping_cfg = validation_cfg.get("early_stopping", {})
    resume_cfg = train_cfg.get("resume", {})
    inference_cfg = cfg.get("inference", {})
    runtime_gpu_ids = runtime_cfg.get("gpu_ids")
    if runtime_gpu_ids is None:
        configured = configured_device(runtime_cfg)
        runtime_gpu_ids = (
            [int(configured.split(":", 1)[1])]
            if configured.startswith("cuda:")
            else [0]
            if configured == "cuda"
            else []
        )

    experiment_name = experiment_cfg.get("name", "idea3-joint-vision-lora")
    output_root = Path(experiment_cfg.get("output_root", "outputs"))
    checkpoint = resolve_experiment_artifact(
        train_cfg.get("checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}-positive.pt",
    )
    last_checkpoint = resolve_experiment_artifact(
        validation_cfg.get("last_checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{checkpoint.stem}-last{checkpoint.suffix}",
    )
    resume_checkpoint = resume_cfg.get("checkpoint") or last_checkpoint

    datadir = dataset_cfg.get("datadir")
    if not datadir:
        raise ValueError(f"Missing dataset.datadir in {path}")

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        datadir=str(datadir),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch16"),
        tokenizer_model=clip_cfg.get("tokenizer_model", clip_cfg.get("model", "openai/clip-vit-base-patch16")),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        device=configured_device(runtime_cfg),
        gpu_ids=tuple(int(gpu_id) for gpu_id in runtime_gpu_ids),
        seed=int(runtime_cfg.get("seed", 0)),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        prompt=prompt_cfg,
        vision_lora=lora_cfg,
        augmentation=augmentation_cfg,
        train_batch_size=int(train_cfg.get("batch_size", dataloader_cfg.get("batch_size", 16))),
        eval_batch_size=int(validation_cfg.get("batch_size", inference_cfg.get("batch_size", 64))),
        num_workers=int(dataloader_cfg.get("num_workers", 4)),
        epochs=int(train_cfg.get("epochs", 50)),
        prompt_lr=float(train_cfg.get("prompt_lr", 1e-3)),
        vision_lora_lr=float(train_cfg.get("vision_lora_lr", 1e-4)),
        train_prompt=bool(train_cfg.get("train_prompt", True)),
        train_vision_lora=bool(train_cfg.get("train_vision_lora", True)),
        positive_text_variant=str(train_cfg.get("positive_text_variant", "learned")).lower(),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        tau=float(train_cfg.get("tau", 0.07)),
        precision=str(train_cfg.get("precision", "fp16")).lower(),
        gradient_clip_norm=float(train_cfg.get("gradient_clip_norm", 1.0)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        grad_cache_mode=str(grad_cache_cfg.get("mode", "never")).lower(),
        grad_cache_micro_batch_size=int(grad_cache_cfg.get("micro_batch_size", 16)),
        loss_name=str(loss_cfg.get("name", "sparse_path_bottleneck")).lower(),
        loss_bottleneck_weight=float(loss_cfg.get("bottleneck_weight", 0.5)),
        loss_bottleneck_temperature=float(loss_cfg.get("bottleneck_temperature", 0.5)),
        loss_route_margin=float(loss_cfg.get("route_margin", 0.05)),
        loss_margin_weight=float(loss_cfg.get("margin_weight", 0.25)),
        loss_classes_per_batch=int(loss_cfg.get("classes_per_batch", 4)),
        loss_examples_per_class=int(loss_cfg.get("examples_per_class", 4)),
        loss_sampler=str(loss_cfg.get("sampler", "hierarchy_independent")).lower(),
        loss_image_weight=float(loss_cfg.get("image_weight", 1.0)),
        loss_alignment_weight=float(loss_cfg.get("alignment_weight", 1.0)),
        loss_alpha=float(loss_cfg.get("alpha", 2.0)),
        loss_beta=float(loss_cfg.get("beta", 50.0)),
        loss_lam=float(loss_cfg.get("lam", 0.5)),
        loss_mining_margin=float(loss_cfg.get("mining_margin", 0.1)),
        loss_minimum_mode=str(loss_cfg.get("minimum_mode", "sample")).lower(),
        loss_dist_scale=float(loss_cfg.get("dist_scale", 2.0)),
        loss_dist_pow=float(loss_cfg.get("dist_pow", 1.0)),
        validation_enabled=bool(validation_cfg.get("enabled", True)),
        validation_every_n_epochs=max(1, int(validation_cfg.get("every_n_epochs", 1))),
        validation_start_epoch=max(1, int(validation_cfg.get("start_epoch", 1))),
        validation_mode=validation_cfg.get("mode", "positive_global_path"),
        early_stopping_enabled=bool(early_stopping_cfg.get("enabled", False)),
        early_stopping_patience=max(1, int(early_stopping_cfg.get("patience", 10))),
        early_stopping_min_delta=max(0.0, float(early_stopping_cfg.get("min_delta", 0.0))),
        resume_enabled=bool(resume_cfg.get("enabled", False)),
        resume_checkpoint=str(resume_checkpoint),
        inference_mode=inference_cfg.get("mode", "positive_global_path"),
        inference_tau=float(inference_cfg.get("tau", 1.0 / float(train_cfg.get("tau", 0.07)))),
        metric_terminal_weight=float(
            validation_cfg.get(
                "terminal_weight",
                inference_cfg.get("terminal_weight", 1.0),
            )
        ),
        metric_terminal_temperature=float(
            validation_cfg.get(
                "bottleneck_temperature",
                inference_cfg.get("bottleneck_temperature", 0.5),
            )
        ),
        checkpoint=str(checkpoint),
        last_checkpoint=str(last_checkpoint),
        result_path=str(resolve_experiment_artifact(
            train_cfg.get("result_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="results",
            default_filename=f"{experiment_name}-positive_global_path.result",
        )),
        diagnostics_path=str(resolve_experiment_artifact(
            train_cfg.get("diagnostics_path"),
            output_root=output_root,
            experiment_name=experiment_name,
            kind="diagnostics",
            default_filename=f"{experiment_name}-positive-diagnostics.json",
        )),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def build_transforms(args):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_size = int(args.augmentation.get("image_size", 224))
    resize_size = int(args.augmentation.get("eval_resize", image_size))
    scale = tuple(float(value) for value in args.augmentation.get("random_resized_crop_scale", [0.7, 1.0]))
    flip_probability = float(args.augmentation.get("horizontal_flip_probability", 0.5))
    jitter_cfg = args.augmentation.get("color_jitter", {})
    jitter_probability = float(jitter_cfg.get("probability", 0.0))

    train_steps = [
        transforms.RandomResizedCrop(
            image_size,
            scale=scale,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=flip_probability),
    ]
    if jitter_probability > 0.0:
        jitter = transforms.ColorJitter(
            brightness=float(jitter_cfg.get("brightness", 0.0)),
            contrast=float(jitter_cfg.get("contrast", 0.0)),
            saturation=float(jitter_cfg.get("saturation", 0.0)),
            hue=float(jitter_cfg.get("hue", 0.0)),
        )
        train_steps.append(transforms.RandomApply([jitter], p=jitter_probability))
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])
    return transforms.Compose(train_steps), eval_transform


def build_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    train_transform, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    train_dataset = SubsetImageFolder(datadir / "train", id_classes, transform=train_transform)
    val_dataset = SubsetImageFolder(datadir / "val", id_classes, transform=eval_transform)
    ood_dataset = SubsetImageFolder(
        datadir / "val",
        hierarchy.ood_train_classes,
        transform=eval_transform,
    )
    empty_files = sorted({
        path
        for dataset in (train_dataset, val_dataset, ood_dataset)
        for path, _ in dataset.samples
        if Path(path).stat().st_size == 0
    })
    if empty_files:
        preview = ", ".join(empty_files[:3])
        raise RuntimeError(
            f"Raw-image joint training found {len(empty_files)} zero-byte image files "
            f"under {datadir}. Populate the real dataset before training. Examples: {preview}"
        )
    print(f"# ID Train: {len(train_dataset)}")
    print(f"# ID Val: {len(val_dataset)}")
    print(f"# OOD: {len(ood_dataset)}")
    return train_dataset, val_dataset, ood_dataset


def build_eval_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    _, eval_transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    val_dataset = SubsetImageFolder(datadir / "val", id_classes, transform=eval_transform)
    ood_dataset = SubsetImageFolder(
        datadir / "val",
        hierarchy.ood_train_classes,
        transform=eval_transform,
    )
    empty_files = sorted({
        path
        for dataset in (val_dataset, ood_dataset)
        for path, _ in dataset.samples
        if Path(path).stat().st_size == 0
    })
    if empty_files:
        preview = ", ".join(empty_files[:3])
        raise RuntimeError(
            f"Raw-image inference found {len(empty_files)} zero-byte image files "
            f"under {datadir}. Populate the real dataset before inference. Examples: {preview}"
        )
    print(f"# ID Val: {len(val_dataset)}")
    print(f"# OOD: {len(ood_dataset)}")
    return val_dataset, ood_dataset


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
        worker_init_fn=seed_data_loader_worker,
    )


def class_paths_by_dataset_target(hierarchy, classes: list[str]) -> dict[int, tuple[str, ...]]:
    paths = {}
    for target, class_name in enumerate(classes):
        node = class_name
        while node not in hierarchy.id_node_list:
            node = hierarchy.child2parent[node]
        paths[target] = tuple(
            [hierarchy.id_node_list[index] for index in hierarchy.node_ancestors.get(node, [])]
            + [node]
        )
    return paths


def make_hierarchy_metric_loader(dataset, hierarchy, args):
    expected_batch_size = args.loss_classes_per_batch * args.loss_examples_per_class
    if expected_batch_size != args.train_batch_size:
        raise ValueError(
            "Dual WeiHiMS requires batch_size == classes_per_batch * examples_per_class; "
            f"got {args.train_batch_size} != {expected_batch_size}"
        )
    sampler_kwargs = {
        "targets": dataset.targets,
        "classes_per_batch": args.loss_classes_per_batch,
        "examples_per_class": args.loss_examples_per_class,
        "seed": args.seed,
    }
    if args.loss_sampler == "hierarchy_independent":
        batch_sampler = HierarchyPKBatchSampler(
            **sampler_kwargs,
            class_paths=class_paths_by_dataset_target(
                hierarchy,
                list(dataset.classes),
            ),
        )
    elif args.loss_sampler == "ms_loss":
        batch_sampler = MSLossPKBatchSampler(**sampler_kwargs)
    elif args.loss_sampler == "globally_balanced":
        batch_sampler = GloballyBalancedPKBatchSampler(**sampler_kwargs)
    else:
        raise ValueError(
            "joint_training.loss.sampler must be one of: "
            "hierarchy_independent, ms_loss, globally_balanced"
        )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=generator,
        worker_init_fn=seed_data_loader_worker,
    )


def leaf_path_label_tensor(hierarchy, leaf_nodes: list[str], device) -> torch.Tensor:
    node_to_index = {node: index for index, node in enumerate(hierarchy.id_node_list)}
    max_length = hierarchy.max_depth + 1
    rows = []
    for leaf in leaf_nodes:
        path = [
            hierarchy.id_node_list[index]
            for index in hierarchy.node_ancestors.get(leaf, [])
        ] + [leaf]
        row = [-1] * max_length
        for depth, node in enumerate(path[:max_length]):
            row[depth] = node_to_index[node]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.no_grad()
def depthwise_alignment_accuracy(
    image_features: torch.Tensor,
    image_path_labels: torch.Tensor,
    prompt_features_by_depth: dict[int, torch.Tensor],
    prompt_node_labels_by_depth: dict[int, torch.Tensor],
) -> tuple[float, float]:
    images = torch.nn.functional.normalize(image_features.float(), dim=-1)
    sample_correct = torch.ones(images.shape[0], dtype=torch.bool, device=images.device)
    sample_active = torch.zeros_like(sample_correct)
    local_correct = 0
    local_total = 0
    for depth in sorted(prompt_features_by_depth):
        if depth <= 0 or depth >= image_path_labels.shape[1]:
            continue
        labels = image_path_labels[:, depth]
        valid = labels >= 0
        if not bool(valid.any()):
            continue
        prompt_labels = prompt_node_labels_by_depth[depth].to(images.device)
        label_to_position = {
            int(label): position
            for position, label in enumerate(prompt_labels.detach().cpu().tolist())
        }
        targets = torch.tensor(
            [label_to_position[int(label)] for label in labels[valid].detach().cpu().tolist()],
            dtype=torch.long,
            device=images.device,
        )
        prompts = torch.nn.functional.normalize(
            prompt_features_by_depth[depth].detach().float(), dim=-1
        )
        predictions = (images[valid] @ prompts.t()).argmax(dim=1)
        correct = predictions == targets
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        sample_correct[valid_indices] &= correct
        sample_active[valid_indices] = True
        local_correct += int(correct.sum().cpu())
        local_total += int(correct.numel())
    path_mask = sample_active
    path_acc = float(sample_correct[path_mask].float().mean().cpu()) if bool(path_mask.any()) else 0.0
    return local_correct / max(1, local_total), path_acc


def load_clip_and_tokenizer(args, device):
    from transformers import CLIPModel, CLIPTokenizerFast

    clip_model = CLIPModel.from_pretrained(
        args.clip_model,
        local_files_only=args.local_files_only,
    ).to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained(
        args.tokenizer_model,
        local_files_only=args.local_files_only,
    )
    return clip_model, tokenizer


def prompt_only_state_dict(learner: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in learner.state_dict().items()
        if not key.startswith("text_encoder.")
    }


def load_prompt_only_state_dict(learner: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = learner.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected positive prompt checkpoint keys: {unexpected}")


def prompt_parameters(learner: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in learner.named_parameters()
        if not name.startswith("text_encoder.") and parameter.requires_grad
    ]


def target_leaf_nodes(hierarchy, classes: list[str], targets: torch.Tensor) -> list[str]:
    ds_to_node = hierarchy.gen_ds2node_map(classes)
    node_indices = ds_to_node[targets.detach().cpu().long()]
    return [hierarchy.id_node_list[int(index)] for index in node_indices.tolist()]


def set_joint_train_mode(clip_model, learner, train_prompt: bool = True, train_vision_lora: bool = True):
    learner.train(train_prompt)
    clip_model.eval()
    set_vision_lora_train_mode(clip_model, train_vision_lora)


def autocast_context(args, device):
    enabled = device.startswith("cuda") and args.precision in {"fp16", "float16"}
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except AttributeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:  # pragma: no cover - compatibility with older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_dual_weihims_batch(
    args,
    hierarchy,
    learner,
    depth_nodes,
    depth_prompt_metadata,
    image_features: torch.Tensor,
    leaf_nodes: list[str],
) -> tuple[torch.Tensor, dict]:
    image_path_labels = leaf_path_label_tensor(
        hierarchy,
        leaf_nodes,
        image_features.device,
    )
    prompt_features_by_depth = encode_depth_prompts(
        learner,
        hierarchy,
        depth_nodes,
    )
    loss, stats = dual_weihims_positive_loss(
        image_features,
        image_path_labels,
        prompt_features_by_depth,
        depth_prompt_metadata["node_labels"],
        depth_prompt_metadata["path_labels"],
        image_weight=args.loss_image_weight,
        alignment_weight=args.loss_alignment_weight,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        lam=args.loss_lam,
        mining_margin=args.loss_mining_margin,
        minimum_mode=args.loss_minimum_mode,
        dist_scale=args.loss_dist_scale,
        dist_pow=args.loss_dist_pow,
    )
    local_acc, path_acc = depthwise_alignment_accuracy(
        image_features.detach(),
        image_path_labels,
        prompt_features_by_depth,
        depth_prompt_metadata["node_labels"],
    )
    stats["local_acc"] = local_acc
    stats["path_acc"] = path_acc
    return loss, stats


def release_cuda_cache(optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resume_signature(args) -> dict:
    return {
        "experiment_name": args.experiment_name,
        "dataset": args.dataset,
        "hierarchy": args.hierarchy,
        "id_split": args.id_split,
        "clip_model": args.clip_model,
        "loss_name": args.loss_name,
        "batch_size": args.train_batch_size,
        "classes_per_batch": args.loss_classes_per_batch,
        "examples_per_class": args.loss_examples_per_class,
        "sampler": args.loss_sampler,
        "prompt_lr": args.prompt_lr,
        "vision_lora_lr": args.vision_lora_lr,
    }


def capture_rng_state(train_loader) -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    loader_generator = getattr(train_loader, "generator", None)
    if loader_generator is not None:
        state["train_loader_generator"] = loader_generator.get_state()
    return state


def restore_rng_state(state: dict | None, train_loader) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    loader_generator = getattr(train_loader, "generator", None)
    if loader_generator is not None and state.get("train_loader_generator") is not None:
        loader_generator.set_state(state["train_loader_generator"])


def make_training_state(
    args,
    *,
    epoch,
    optimizer,
    scheduler,
    scaler,
    train_loader,
    history,
    best_epoch,
    best_bacc,
    best_prompt_state,
    best_lora_state,
    validations_without_improvement,
    grad_cache_active,
    active_micro_batch_size,
    training_loop_complete,
) -> dict:
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    sampler_state = (
        batch_sampler.state_dict()
        if batch_sampler is not None and hasattr(batch_sampler, "state_dict")
        else None
    )
    return {
        "version": 1,
        "epoch": int(epoch),
        "resume_signature": resume_signature(args),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(train_loader),
        "sampler_state_dict": sampler_state,
        "history": history,
        "best_epoch": best_epoch,
        "best_bacc": best_bacc if math.isfinite(best_bacc) else None,
        "best_prompt_state_dict": best_prompt_state,
        "best_vision_lora_state_dict": best_lora_state,
        "validations_without_improvement": int(validations_without_improvement),
        "grad_cache_active": bool(grad_cache_active),
        "active_micro_batch_size": int(active_micro_batch_size),
        "training_loop_complete": bool(training_loop_complete),
    }


@torch.no_grad()
def evaluate_split_raw(
    args,
    hierarchy,
    clip_model,
    learner,
    dataset,
    loader,
    device,
    split_name: str,
    inference_mode: str | None = None,
    image_encoder=None,
):
    inference_mode = inference_mode or args.inference_mode
    clip_model.eval()
    learner.eval()
    set_vision_lora_train_mode(clip_model, False)
    if inference_mode == "positive_metric_terminal":
        terminal_specs = build_metric_terminal_specs(hierarchy)
        edge_pairs = list(dict.fromkeys(
            edge for spec in terminal_specs for edge in spec.route_edges
        ))
        encoded_edges = learner.encode_edges(edge_pairs).detach()
        positive_edge_features = {
            edge: encoded_edges[index]
            for index, edge in enumerate(edge_pairs)
        }
        semantic_index = None
    else:
        terminal_specs = None
        positive_edge_features = None
        semantic_index = build_idea3_semantic_index(
            hierarchy,
            learner,
            mode=inference_mode,
        )

    predictions = []
    targets = []
    for images, batch_targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(args, device):
            image_features = (
                image_encoder(images)
                if image_encoder is not None
                else clip_model.get_image_features(pixel_values=images)
            )
        if inference_mode == "positive_metric_terminal":
            output = predict_features_metric_terminal(
                image_features,
                hierarchy,
                positive_edge_features,
                terminal_specs,
                terminal_weight=args.metric_terminal_weight,
                bottleneck_temperature=args.metric_terminal_temperature,
            )
        else:
            output = predict_features_idea3(
                image_features,
                hierarchy,
                semantic_index,
                mode=inference_mode,
                tau=args.inference_tau,
            )
        predictions.append(output["preds"].cpu())
        targets.append(batch_targets.cpu())

    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    payload = {
        "classes": list(dataset.classes),
        "targets": target_tensor,
    }
    node_targets, metrics = evaluate_split(
        hierarchy,
        payload,
        prediction_tensor,
        dists_mats=make_distance_mats(hierarchy),
    )
    return {
        "preds": prediction_tensor,
        "targets": node_targets.cpu(),
        "metrics": metrics,
        "split": split_name,
    }


def save_joint_checkpoint(
    args,
    path,
    hierarchy,
    prompt_cfg,
    lora_cfg,
    learner,
    clip_model,
    metrics,
    training_state=None,
):
    stage = (
        "positive_joint_vision_lora_dual_weihims"
        if args.loss_name == "dual_weihims"
        else "positive_joint_vision_lora"
        if args.train_prompt and args.train_vision_lora
        else "positive_prompt_only_base_vision"
        if args.train_prompt
        else "positive_vision_lora_plain_text"
    )
    return save_idea3_checkpoint(
        path,
        stage=stage,
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=prompt_only_state_dict(learner),
        vision_lora_config=lora_cfg.to_dict() if args.train_vision_lora else None,
        vision_lora_state_dict=(
            vision_lora_state_dict(clip_model) if args.train_vision_lora else None
        ),
        metrics=metrics,
        args=vars(args),
        training_state=training_state,
    )


def main():
    args = parse_args()
    supported_losses = {"sparse_path_bottleneck", "dual_weihims"}
    if args.loss_name not in supported_losses:
        raise ValueError(
            f"Unsupported joint_training.loss.name: {args.loss_name}. "
            f"Expected one of {sorted(supported_losses)}"
        )
    if args.positive_text_variant not in {"learned", "plain"}:
        raise ValueError(
            "joint_training.positive_text_variant must be one of: learned, plain"
        )
    if args.train_prompt != (args.positive_text_variant == "learned"):
        raise ValueError(
            "A proper training ablation requires train_prompt=true with learned text "
            "or train_prompt=false with plain text"
        )
    if not args.train_prompt and not args.train_vision_lora:
        raise ValueError("At least one of train_prompt or train_vision_lora must be true")
    if args.loss_name == "dual_weihims" and not (args.train_prompt and args.train_vision_lora):
        raise ValueError("dual_weihims requires both positive prompts and Vision LoRA to train")
    if args.grad_cache_mode not in {"never", "auto", "always"}:
        raise ValueError("joint_training.grad_cache.mode must be never, auto, or always")
    if args.grad_cache_micro_batch_size <= 0:
        raise ValueError("joint_training.grad_cache.micro_batch_size must be positive")
    if args.grad_cache_mode != "never" and args.loss_name != "dual_weihims":
        raise ValueError("GradCache is currently supported only for dual_weihims")
    if args.loss_sampler not in {
        "hierarchy_independent",
        "ms_loss",
        "globally_balanced",
    }:
        raise ValueError(
            "joint_training.loss.sampler must be one of: "
            "hierarchy_independent, ms_loss, globally_balanced"
        )
    configure_reproducibility(args.seed, deterministic=args.deterministic)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    train_dataset, val_dataset, ood_dataset = build_datasets(args, hierarchy)
    if args.loss_name == "dual_weihims":
        train_loader = make_hierarchy_metric_loader(train_dataset, hierarchy, args)
    else:
        train_loader = make_loader(
            train_dataset,
            args.train_batch_size,
            args.num_workers,
            shuffle=True,
            seed=args.seed,
        )
    val_loader = make_loader(
        val_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )
    ood_loader = make_loader(
        ood_dataset,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        seed=args.seed,
    )

    clip_model, tokenizer = load_clip_and_tokenizer(args, device)
    if args.gradient_checkpointing and hasattr(clip_model, "gradient_checkpointing_enable"):
        clip_model.gradient_checkpointing_enable()

    lora_cfg = VisionLoRAConfig.from_dict(args.vision_lora)
    if args.train_vision_lora:
        replaced_modules = inject_clip_vision_lora(clip_model, lora_cfg)
    else:
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)
        replaced_modules = []
    text_encoder = SoftPromptTextEncoder(
        clip_model,
        tokenizer,
        max_length=int(args.prompt.get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(args.prompt)
    learner = PositivePromptLearner(
        args.dataset,
        hierarchy,
        text_encoder,
        prompt_cfg,
    ).to(device)
    learner.set_text_variant(args.positive_text_variant)
    image_encoder, active_gpu_ids = build_parallel_image_encoder(
        clip_model,
        device,
        args.gpu_ids,
    )
    depth_nodes = build_depth_prompt_sets(hierarchy) if args.loss_name == "dual_weihims" else None
    depth_prompt_metadata = (
        build_depth_prompt_metadata(hierarchy, depth_nodes, device)
        if depth_nodes is not None
        else None
    )

    prompt_params = prompt_parameters(learner) if args.train_prompt else []
    if not args.train_prompt:
        for name, parameter in learner.named_parameters():
            if not name.startswith("text_encoder."):
                parameter.requires_grad_(False)
    lora_params = vision_lora_parameters(clip_model) if args.train_vision_lora else []
    parameter_groups = []
    if prompt_params:
        parameter_groups.append({"params": prompt_params, "lr": args.prompt_lr, "group_name": "prompt"})
    if lora_params:
        parameter_groups.append({
            "params": lora_params,
            "lr": args.vision_lora_lr,
            "group_name": "vision_lora",
        })
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    use_scaler = device.startswith("cuda") and args.precision in {"fp16", "float16"}
    scaler = make_grad_scaler(use_scaler)
    identity_adapter = nn.Identity()
    grad_cache_active = args.grad_cache_mode == "always"
    active_micro_batch_size = min(
        args.train_batch_size,
        args.grad_cache_micro_batch_size,
    )

    print(
        "positive vision/text training: "
        f"text_variant={args.positive_text_variant}, "
        f"train_prompt={args.train_prompt}, train_vision_lora={args.train_vision_lora}, "
        f"replaced_modules={len(replaced_modules)}, "
        f"prompt_params={sum(parameter.numel() for parameter in prompt_params)}, "
        f"vision_lora_params={sum(parameter.numel() for parameter in lora_params)}, "
        f"batch_size={args.train_batch_size}, precision={args.precision}, "
        f"loss={args.loss_name}, sampler={args.loss_sampler}, "
        f"seed={args.seed}, deterministic={args.deterministic}, "
        f"gpu_ids={list(active_gpu_ids)}, "
        f"data_parallel={len(active_gpu_ids) > 1}, "
        f"grad_cache={args.grad_cache_mode}, "
        f"grad_cache_micro_batch={active_micro_batch_size}"
    )

    history = []
    best_bacc = float("-inf")
    best_epoch = None
    best_prompt_state = None
    best_lora_state = None
    validations_without_improvement = 0
    start_epoch = 1
    last_completed_epoch = 0

    resume_path = Path(args.resume_checkpoint)
    previous_resume_path = previous_checkpoint_path(resume_path)
    if args.resume_enabled and (resume_path.exists() or previous_resume_path.exists()):
        resume_checkpoint, loaded_resume_path = load_idea3_checkpoint_with_fallback(
            resume_path,
            map_location="cpu",
        )
        training_state = resume_checkpoint.get("training_state")
        if not training_state:
            raise ValueError(
                f"Resume checkpoint has no training_state: {resume_path}"
            )
        saved_signature = training_state.get("resume_signature")
        current_signature = resume_signature(args)
        if saved_signature is not None and "sampler" not in saved_signature:
            saved_signature = {
                **saved_signature,
                "sampler": "hierarchy_independent",
            }
        if saved_signature != current_signature:
            raise ValueError(
                "Resume checkpoint configuration mismatch: "
                f"saved={saved_signature}, current={current_signature}"
            )
        load_prompt_only_state_dict(
            learner,
            resume_checkpoint["positive_state_dict"],
        )
        load_vision_lora_state_dict(
            clip_model,
            resume_checkpoint["vision_lora_state_dict"],
        )
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        scheduler.load_state_dict(training_state["scheduler_state_dict"])
        scaler.load_state_dict(training_state.get("scaler_state_dict", {}))
        history = list(training_state.get("history", []))
        best_epoch = training_state.get("best_epoch")
        saved_best_bacc = training_state.get("best_bacc")
        best_bacc = (
            float(saved_best_bacc)
            if saved_best_bacc is not None
            else float("-inf")
        )
        best_prompt_state = training_state.get("best_prompt_state_dict")
        best_lora_state = training_state.get("best_vision_lora_state_dict")
        validations_without_improvement = int(
            training_state.get("validations_without_improvement", 0)
        )
        grad_cache_active = bool(
            training_state.get("grad_cache_active", grad_cache_active)
        )
        active_micro_batch_size = int(
            training_state.get(
                "active_micro_batch_size",
                active_micro_batch_size,
            )
        )
        last_completed_epoch = int(training_state["epoch"])
        start_epoch = last_completed_epoch + 1
        if training_state.get("training_loop_complete", False):
            start_epoch = args.epochs + 1
        sampler_state = training_state.get("sampler_state_dict")
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if sampler_state is not None and hasattr(batch_sampler, "load_state_dict"):
            batch_sampler.load_state_dict(sampler_state)
        restore_rng_state(training_state.get("rng_state"), train_loader)
        if loaded_resume_path != resume_path:
            print(
                f"primary resume checkpoint was unavailable or invalid; "
                f"using previous checkpoint: {loaded_resume_path}"
            )
        print(
            f"resumed checkpoint: {loaded_resume_path}, "
            f"completed_epoch={last_completed_epoch}, "
            f"next_epoch={start_epoch if start_epoch <= args.epochs else 'finalize'}, "
            f"best_epoch={best_epoch}, best_val_bacc={best_bacc:.6f}"
        )
    elif args.resume_enabled:
        print(f"resume checkpoint not found; starting fresh: {resume_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        set_joint_train_mode(
            clip_model,
            learner,
            train_prompt=args.train_prompt,
            train_vision_lora=args.train_vision_lora,
        )
        epoch_loss = 0.0
        epoch_path_acc = 0.0
        epoch_local_acc = 0.0
        epoch_image_weihims = 0.0
        epoch_image_prompt_weihims = 0.0
        steps = 0
        iterator = tqdm(
            train_loader,
            desc=f"joint epoch {epoch}/{args.epochs}",
            leave=False,
        ) if tqdm else train_loader

        for images, batch_targets in iterator:
            images = images.to(device, non_blocking=True)
            leaf_nodes = target_leaf_nodes(hierarchy, train_dataset.classes, batch_targets)
            optimizer.zero_grad(set_to_none=True)

            if args.loss_name == "dual_weihims":
                def loss_closure(cached_image_features):
                    return compute_dual_weihims_batch(
                        args,
                        hierarchy,
                        learner,
                        depth_nodes,
                        depth_prompt_metadata,
                        cached_image_features,
                        leaf_nodes,
                    )

                direct_completed = False
                if not grad_cache_active:
                    try:
                        with autocast_context(args, device):
                            image_features = image_encoder(images)
                            loss, stats = loss_closure(image_features)
                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        direct_completed = True
                    except RuntimeError as error:
                        if args.grad_cache_mode != "auto" or not is_cuda_out_of_memory(error):
                            raise
                        print(
                            "direct logical batch ran out of CUDA memory; "
                            f"switching permanently to GradCache with micro-batch "
                            f"{active_micro_batch_size}"
                        )
                        grad_cache_active = True
                        release_cuda_cache(optimizer)

                if not direct_completed:
                    while True:
                        try:
                            _, stats = grad_cache_forward_backward(
                                images,
                                image_encoder,
                                loss_closure,
                                micro_batch_size=active_micro_batch_size,
                                scaler=scaler,
                                autocast_factory=lambda: autocast_context(args, device),
                                cuda_devices=active_gpu_ids,
                            )
                            break
                        except RuntimeError as error:
                            if (
                                not is_cuda_out_of_memory(error)
                                or active_micro_batch_size <= max(1, len(active_gpu_ids))
                            ):
                                raise
                            release_cuda_cache(optimizer)
                            active_micro_batch_size = max(
                                max(1, len(active_gpu_ids)),
                                active_micro_batch_size // 2,
                            )
                            print(
                                "GradCache micro-batch ran out of CUDA memory; "
                                f"retrying with micro-batch {active_micro_batch_size}"
                            )
            else:
                with autocast_context(args, device):
                    image_features = image_encoder(images)
                if args.train_prompt and args.train_vision_lora:
                    stats = backward_sparse_path_bottleneck_streaming(
                        args,
                        hierarchy,
                        learner,
                        identity_adapter,
                        image_features,
                        leaf_nodes,
                        grad_scaler=scaler,
                    )
                else:
                    loss, stats = compute_sparse_path_positive_loss(
                        args,
                        hierarchy,
                        learner,
                        image_features,
                        leaf_nodes,
                    )
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(prompt_params + lora_params, args.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(prompt_params + lora_params, args.gradient_clip_norm)
                optimizer.step()

            epoch_loss += stats["loss"]
            epoch_path_acc += stats["path_acc"]
            epoch_local_acc += stats["local_acc"]
            epoch_image_weihims += stats.get("image_weihims_loss", 0.0)
            epoch_image_prompt_weihims += stats.get("image_prompt_weihims_loss", 0.0)
            steps += 1

        scheduler.step()
        epoch_stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "path_acc": epoch_path_acc / max(1, steps),
            "local_acc": epoch_local_acc / max(1, steps),
            "steps": steps,
        }
        if args.loss_name == "dual_weihims":
            epoch_stats["image_weihims_loss"] = epoch_image_weihims / max(1, steps)
            epoch_stats["image_prompt_weihims_loss"] = (
                epoch_image_prompt_weihims / max(1, steps)
            )
            epoch_stats["path_ce_loss"] = 0.0
            epoch_stats["grad_cache_active"] = bool(grad_cache_active)
            epoch_stats["grad_cache_micro_batch_size"] = int(
                active_micro_batch_size
            )
            epoch_stats["sampler"] = args.loss_sampler
            if hasattr(train_loader.batch_sampler, "exposure_stats"):
                epoch_stats["sampler_exposure"] = (
                    train_loader.batch_sampler.exposure_stats()
                )
        for group in optimizer.param_groups:
            epoch_stats[f"{group['group_name']}_lr"] = group["lr"]

        validation_due = (
            args.validation_enabled
            and epoch >= args.validation_start_epoch
            and (epoch - args.validation_start_epoch) % args.validation_every_n_epochs == 0
        )
        is_best = False
        should_stop = False
        if validation_due:
            val_result = evaluate_split_raw(
                args,
                hierarchy,
                clip_model,
                learner,
                val_dataset,
                val_loader,
                device,
                "val",
                inference_mode=args.validation_mode,
                image_encoder=image_encoder,
            )
            val_bacc = float(val_result["metrics"]["balanced_acc"])
            val_bmhd = float(val_result["metrics"]["balanced_hdist"])
            epoch_stats["val_balanced_acc"] = val_bacc
            epoch_stats["val_balanced_hdist"] = val_bmhd
            if val_bacc > best_bacc + args.early_stopping_min_delta:
                best_bacc = val_bacc
                best_epoch = epoch
                best_prompt_state = prompt_only_state_dict(learner)
                best_lora_state = vision_lora_state_dict(clip_model)
                validations_without_improvement = 0
                is_best = True
            else:
                validations_without_improvement += 1
            epoch_stats["early_stopping_bad_validations"] = (
                validations_without_improvement
            )
            should_stop = (
                args.early_stopping_enabled
                and validations_without_improvement >= args.early_stopping_patience
            )
            epoch_stats["early_stopping_triggered"] = should_stop

        history.append(epoch_stats)
        message = (
            f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, "
            f"path_acc={epoch_stats['path_acc']:.6f}, "
            f"local_acc={epoch_stats['local_acc']:.6f}"
        )
        if args.loss_name == "dual_weihims":
            message += (
                f", image_weihims={epoch_stats['image_weihims_loss']:.6f}, "
                f"image_prompt_weihims={epoch_stats['image_prompt_weihims_loss']:.6f}, "
                "path_ce=0.000000, "
                f"grad_cache={epoch_stats['grad_cache_active']}, "
                f"micro_batch={epoch_stats['grad_cache_micro_batch_size']}"
            )
        if validation_due:
            message += (
                f", val_bacc={epoch_stats['val_balanced_acc']:.6f}, "
                f"val_bmhd={epoch_stats['val_balanced_hdist']:.6f}"
            )
            if is_best:
                message += " [best]"
        print(message)

        if is_best:
            save_joint_checkpoint(
                args,
                args.checkpoint,
                hierarchy,
                prompt_cfg,
                lora_cfg,
                learner,
                clip_model,
                {
                    "train_history": history,
                    "best_validation": {
                        "epoch": best_epoch,
                        "val_balanced_acc": best_bacc,
                    },
                },
            )
        last_completed_epoch = epoch
        last_metrics = {
            "train_history": history,
            "best_validation": {
                "epoch": best_epoch,
                "val_balanced_acc": (
                    best_bacc if math.isfinite(best_bacc) else None
                ),
            },
        }
        resume_training_state = make_training_state(
            args,
            epoch=last_completed_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            history=history,
            best_epoch=best_epoch,
            best_bacc=best_bacc,
            best_prompt_state=best_prompt_state,
            best_lora_state=best_lora_state,
            validations_without_improvement=validations_without_improvement,
            grad_cache_active=grad_cache_active,
            active_micro_batch_size=active_micro_batch_size,
            training_loop_complete=should_stop or epoch >= args.epochs,
        )
        save_joint_checkpoint(
            args,
            args.last_checkpoint,
            hierarchy,
            prompt_cfg,
            lora_cfg,
            learner,
            clip_model,
            last_metrics,
            training_state=resume_training_state,
        )
        if should_stop:
            print(
                "early stopping: "
                f"no val_bacc improvement greater than "
                f"{args.early_stopping_min_delta:g} for "
                f"{validations_without_improvement} validation runs; "
                f"best epoch={best_epoch}, best val_bacc={best_bacc:.6f}"
            )
            break

    last_metrics = {
        "train_history": history,
        "best_validation": {
            "epoch": best_epoch,
            "val_balanced_acc": best_bacc if math.isfinite(best_bacc) else None,
        },
    }

    if best_prompt_state is None or best_lora_state is None:
        best_epoch = last_completed_epoch
        best_prompt_state = prompt_only_state_dict(learner)
        best_lora_state = vision_lora_state_dict(clip_model)
    load_prompt_only_state_dict(learner, best_prompt_state)
    load_vision_lora_state_dict(clip_model, best_lora_state)

    val_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        val_dataset,
        val_loader,
        device,
        "val",
        image_encoder=image_encoder,
    )
    ood_result = evaluate_split_raw(
        args,
        hierarchy,
        clip_model,
        learner,
        ood_dataset,
        ood_loader,
        device,
        "ood",
        image_encoder=image_encoder,
    )
    mixed = mixed_summary(val_result["metrics"], ood_result["metrics"])
    result = {
        "args": vars(args),
        "mode": args.inference_mode,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": (
            "positive_joint_vision_lora_dual_weihims"
            if args.loss_name == "dual_weihims"
            else "positive_joint_vision_lora"
            if args.train_prompt and args.train_vision_lora
            else "positive_prompt_only_base_vision"
            if args.train_prompt
            else "positive_vision_lora_plain_text"
        ),
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "val": val_result,
        "ood": ood_result,
        "mixed": mixed,
    }
    ensure_dir(Path(args.result_path).parent)
    torch.save(result, args.result_path)

    final_metrics = {
        "train_history": history,
        "best_validation": {
            "epoch": best_epoch,
            "val_balanced_acc": float(val_result["metrics"]["balanced_acc"]),
            "val_balanced_hdist": float(val_result["metrics"]["balanced_hdist"]),
        },
        "final": {
            "val_balanced_acc": float(val_result["metrics"]["balanced_acc"]),
            "ood_balanced_acc": float(ood_result["metrics"]["balanced_acc"]),
            "mixed_balanced_acc": float(mixed["mixed_balanced_acc"]),
            "val_balanced_hdist": float(val_result["metrics"]["balanced_hdist"]),
            "ood_balanced_hdist": float(ood_result["metrics"]["balanced_hdist"]),
            "mixed_balanced_hdist": float(mixed["mixed_balanced_hdist"]),
        },
    }
    save_joint_checkpoint(
        args,
        args.checkpoint,
        hierarchy,
        prompt_cfg,
        lora_cfg,
        learner,
        clip_model,
        final_metrics,
    )
    save_json(args.diagnostics_path, final_metrics)

    print(f"best epoch: {best_epoch}")
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved last checkpoint: {args.last_checkpoint}")
    print(f"saved result: {args.result_path}")
    print(f"ID BAcc: {float(val_result['metrics']['balanced_acc']):.6f}")
    print(f"ID BMHD: {float(val_result['metrics']['balanced_hdist']):.6f}")


if __name__ == "__main__":
    main()
