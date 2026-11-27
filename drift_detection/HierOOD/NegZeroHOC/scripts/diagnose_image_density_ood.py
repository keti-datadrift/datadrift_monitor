from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.runtime import available_device
from negzerohoc.vision_lora import (
    VisionLoRAConfig,
    inject_clip_vision_lora,
    load_vision_lora_state_dict,
    set_vision_lora_enabled,
    set_vision_lora_train_mode,
)
from scripts.train_idea3_joint_vision_lora import (
    build_transforms,
    load_clip_and_tokenizer,
    make_loader,
)
from scripts.train_idea4_unknown_prompts import encode_dataset_features, freeze_module
from scripts.train_image_metric_vision_lora import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--out", required=True)
    parsed = parser.parse_args()
    args = load_config(parsed.config)
    if parsed.checkpoint:
        args.checkpoint = parsed.checkpoint
    if parsed.gpu_id is not None:
        args.device = f"cuda:{parsed.gpu_id}"
    return args, parsed


def build_eval_datasets(args, hierarchy):
    from negzerohoc.prohoc_compat.utils.dataset_util import SubsetImageFolder, get_id_classes

    _, transform = build_transforms(args)
    id_classes = get_id_classes(args.id_split)
    datadir = Path(args.datadir)
    return (
        SubsetImageFolder(datadir / "train", id_classes, transform=transform),
        SubsetImageFolder(datadir / "val", id_classes, transform=transform),
        SubsetImageFolder(datadir / "val", hierarchy.ood_train_classes, transform=transform),
    )


def node_targets(hierarchy, payload: dict) -> list[str]:
    mapping = hierarchy.gen_ds2node_map(payload["classes"])
    indices = mapping[payload["targets"].long().cpu()]
    return [hierarchy.id_node_list[int(index)] for index in indices.tolist()]


def class_centroids(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(features.float(), dim=-1)
    centroids = []
    for target in sorted(set(targets.long().tolist())):
        centroids.append(F.normalize(normalized[targets == int(target)].mean(dim=0), dim=0))
    return torch.stack(centroids)


def max_centroid_cosine(features: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return (F.normalize(features.float(), dim=-1) @ F.normalize(centroids.float(), dim=-1).t()).max(dim=1).values


def mean_knn_cosine(features: torch.Tensor, bank: torch.Tensor, k: int, chunk_size: int = 256) -> torch.Tensor:
    bank = F.normalize(bank.float(), dim=-1)
    k = min(max(1, int(k)), int(bank.shape[0]))
    chunks = []
    for start in range(0, int(features.shape[0]), chunk_size):
        query = F.normalize(features[start:start + chunk_size].float(), dim=-1)
        chunks.append((query @ bank.t()).topk(k, dim=1).values.mean(dim=1))
    return torch.cat(chunks)


def density_metrics(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> dict:
    metrics = binary_ood_metrics(-id_scores.numpy(), -ood_scores.numpy())
    metrics.update({
        "id_mean_similarity": float(id_scores.mean()),
        "ood_mean_similarity": float(ood_scores.mean()),
    })
    return metrics


def local_density_diagnostics(hierarchy, train_payload, id_payload, ood_payload, k: int) -> dict:
    train_nodes = node_targets(hierarchy, train_payload)
    id_nodes = node_targets(hierarchy, id_payload)
    ood_nodes = node_targets(hierarchy, ood_payload)
    by_parent = {}
    leaf_parents = [
        parent for parent, children in hierarchy.parent2children.items()
        if parent != "root" and children
        and all(child not in hierarchy.parent2children for child in children)
    ]
    for parent in sorted(leaf_parents):
        children = list(hierarchy.parent2children[parent])
        train_indices = [i for i, node in enumerate(train_nodes) if node in children]
        id_indices = [i for i, node in enumerate(id_nodes) if node in children]
        ood_indices = [i for i, node in enumerate(ood_nodes) if node == parent]
        if not train_indices or not id_indices or not ood_indices:
            continue
        train_index = torch.tensor(train_indices, dtype=torch.long)
        id_index = torch.tensor(id_indices, dtype=torch.long)
        ood_index = torch.tensor(ood_indices, dtype=torch.long)
        bank = train_payload["features"].index_select(0, train_index)
        child_targets = torch.tensor(
            [children.index(train_nodes[index]) for index in train_indices], dtype=torch.long
        )
        centroids = class_centroids(bank, child_targets)
        id_features = id_payload["features"].index_select(0, id_index)
        ood_features = ood_payload["features"].index_select(0, ood_index)
        by_parent[parent] = {
            "id_samples": len(id_indices),
            "ood_samples": len(ood_indices),
            "centroid": density_metrics(
                max_centroid_cosine(id_features, centroids),
                max_centroid_cosine(ood_features, centroids),
            ),
            "knn": density_metrics(
                mean_knn_cosine(id_features, bank, k),
                mean_knn_cosine(ood_features, bank, k),
            ),
        }
    macro = {}
    for method in ("centroid", "knn"):
        macro[method] = {
            key: float(np.mean([value[method][key] for value in by_parent.values()]))
            for key in ("auroc", "fpr95", "best_balanced_acc_diagnostic_only")
        }
    return {"macro": macro, "by_parent": by_parent}


def main():
    args, parsed = parse_args()
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    train_dataset, val_dataset, ood_dataset = build_eval_datasets(args, hierarchy)
    train_loader = make_loader(train_dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    val_loader = make_loader(val_dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    ood_loader = make_loader(ood_dataset, args.eval_batch_size, args.num_workers, False, args.seed)
    checkpoint = load_idea3_checkpoint(args.checkpoint, map_location="cpu")
    clip_model, _ = load_clip_and_tokenizer(args, device)
    lora_cfg = VisionLoRAConfig.from_dict(checkpoint["vision_lora_config"])
    inject_clip_vision_lora(clip_model, lora_cfg)
    load_vision_lora_state_dict(clip_model, checkpoint["vision_lora_state_dict"])
    freeze_module(clip_model)
    set_vision_lora_enabled(clip_model, True)
    set_vision_lora_train_mode(clip_model, False)
    train_payload = encode_dataset_features(args, clip_model, train_dataset, train_loader, device, "encode ID train")
    id_payload = encode_dataset_features(args, clip_model, val_dataset, val_loader, device, "encode ID val")
    ood_payload = encode_dataset_features(args, clip_model, ood_dataset, ood_loader, device, "encode OOD")
    centroids = class_centroids(train_payload["features"], train_payload["targets"])
    global_metrics = {
        "centroid": density_metrics(
            max_centroid_cosine(id_payload["features"], centroids),
            max_centroid_cosine(ood_payload["features"], centroids),
        ),
        "knn": density_metrics(
            mean_knn_cosine(id_payload["features"], train_payload["features"], parsed.knn_k),
            mean_knn_cosine(ood_payload["features"], train_payload["features"], parsed.knn_k),
        ),
    }
    result = {
        "method": "image_density_ood_diagnostic",
        "diagnostic_only": True,
        "used_ood_for_training_or_selection": False,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint.get("stage"),
        "knn_k": parsed.knn_k,
        "global": global_metrics,
        "oracle_parent_local": local_density_diagnostics(
            hierarchy, train_payload, id_payload, ood_payload, parsed.knn_k
        ),
    }
    output_path = Path(parsed.out)
    ensure_dir(output_path.parent)
    save_json(output_path, result)
    print(f"saved: {output_path}")
    print(
        f"global centroid AUROC={global_metrics['centroid']['auroc']:.6f}, "
        f"global kNN AUROC={global_metrics['knn']['auroc']:.6f}, "
        f"local centroid AUROC={result['oracle_parent_local']['macro']['centroid']['auroc']:.6f}, "
        f"local kNN AUROC={result['oracle_parent_local']['macro']['knn']['auroc']:.6f}"
    )


if __name__ == "__main__":
    main()
