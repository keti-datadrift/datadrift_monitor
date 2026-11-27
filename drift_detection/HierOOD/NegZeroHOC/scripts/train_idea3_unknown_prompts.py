from __future__ import annotations

import argparse
from argparse import Namespace
import random
import sys
from pathlib import Path

import torch
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.checkpointing import load_idea3_checkpoint, save_idea3_checkpoint
from negzerohoc.clip_backend import ClipBackend
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, load_feature_file, save_json
from negzerohoc.image_adapters import build_image_adapter
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.losses import unknown_ce_loss, unknown_regularization
from negzerohoc.output_layout import resolve_experiment_artifact, resolve_shared_feature_dir
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner, UnknownPromptLearner
from negzerohoc.runtime import (
    available_device,
    configure_reproducibility,
    configured_device,
)
from negzerohoc.soft_prompting import SoftPromptTextEncoder
from negzerohoc.training_data import (
    UNKNOWN_LABEL,
    build_positive_edge_examples,
    gather_image_features,
    group_examples_by_parent_child,
    sample_leave_child_out_episode,
)


def load_config(path):
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    features_cfg = cfg.get("features", {})
    inference_cfg = cfg.get("inference", {})
    unknown_cfg = cfg.get("unknown_training", {})

    experiment_name = experiment_cfg.get("name", "idea3")
    output_root = experiment_cfg.get("output_root", "outputs")
    positive_checkpoint = resolve_experiment_artifact(
        unknown_cfg.get("positive_checkpoint") or inference_cfg.get("positive_checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}-positive.pt",
    )
    checkpoint = resolve_experiment_artifact(
        unknown_cfg.get("checkpoint"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=f"{experiment_name}-parent_unknown.pt",
    )
    result_path = resolve_experiment_artifact(
        unknown_cfg.get("result_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="results",
        default_filename=f"{experiment_name}-parent_unknown.result",
    )
    diagnostics_path = resolve_experiment_artifact(
        unknown_cfg.get("diagnostics_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="diagnostics",
        default_filename=f"{experiment_name}-unknown-diagnostics.json",
    )

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=output_root,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        local_files_only=bool(clip_cfg.get("local_files_only", True)),
        features_dir=str(resolve_shared_feature_dir(
            features_cfg.get("dir") or inference_cfg.get("features_dir"),
            output_root=output_root,
        )) if features_cfg.get("dir") or inference_cfg.get("features_dir") else None,
        device=configured_device(runtime_cfg),
        seed=int(runtime_cfg.get("seed", 0)),
        epochs=int(unknown_cfg.get("epochs", 20)),
        batch_size=int(unknown_cfg.get("batch_size", 1024)),
        parents_per_step=int(unknown_cfg.get("parents_per_step", 4)),
        lr=float(unknown_cfg.get("lr", 1e-3)),
        weight_decay=float(unknown_cfg.get("weight_decay", 1e-4)),
        tau=float(unknown_cfg.get("tau", 0.07)),
        hide_strategy=unknown_cfg.get("hide_strategy", "hide_one_child"),
        lambda_anchor=float(unknown_cfg.get("lambda_anchor", 0.1)),
        lambda_child_sep=float(unknown_cfg.get("lambda_child_sep", 0.1)),
        eval_after_training=bool(unknown_cfg.get("eval_after_training", True)),
        inference_batch_size=int(inference_cfg.get("batch_size", 1024)),
        inference_tau=float(inference_cfg.get("tau", 1.0)),
        allow_root_unknown=bool(inference_cfg.get("allow_root_unknown", False)),
        positive_checkpoint=str(positive_checkpoint),
        checkpoint=str(checkpoint),
        result_path=str(result_path),
        diagnostics_path=str(diagnostics_path),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def load_positive_image_adapter(positive_ckpt: dict, input_dim: int, device):
    adapter = build_image_adapter(input_dim, positive_ckpt.get("image_adapter_config") or {"mode": "none"}).to(device)
    state = positive_ckpt.get("image_adapter_state_dict")
    if state is not None:
        adapter.load_state_dict(state)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)
    return adapter


@torch.no_grad()
def evaluate_parent_unknown(args, hierarchy, positive, unknown, image_adapter, features_dir: Path, device: str) -> dict:
    dists_mats = make_distance_mats(hierarchy)
    semantic_index = build_idea3_semantic_index(
        hierarchy,
        positive,
        unknown,
        mode="parent_unknown",
        allow_root_unknown=args.allow_root_unknown,
    )
    results = {
        "args": vars(args),
        "mode": "parent_unknown",
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
    }
    for split_name in ["val", "ood"]:
        payload = load_feature_file(features_dir / f"{split_name}-features.pt")
        preds = []
        stop_depth_counts = {}
        stop_node_counts = {}
        features = payload["features"]
        for start in range(0, int(features.shape[0]), args.inference_batch_size):
            end = min(start + args.inference_batch_size, int(features.shape[0]))
            image_features = image_adapter(features[start:end].to(device))
            out = predict_features_idea3(
                image_features,
                hierarchy,
                semantic_index,
                mode="parent_unknown",
                tau=args.inference_tau,
            )
            preds.append(out["preds"].cpu())
            for key, value in out["diagnostics"]["stop_depth_counts"].items():
                stop_depth_counts[int(key)] = stop_depth_counts.get(int(key), 0) + int(value)
            for key, value in out["diagnostics"]["stop_node_counts"].items():
                stop_node_counts[str(key)] = stop_node_counts.get(str(key), 0) + int(value)
        preds = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long)
        node_labels, metrics = evaluate_split(hierarchy, payload, preds.cpu(), dists_mats=dists_mats)
        results[split_name] = {
            "preds": preds.cpu(),
            "targets": node_labels.cpu(),
            "metrics": metrics,
            "diagnostics": {
                "stop_depth_counts": dict(sorted(stop_depth_counts.items())),
                "stop_node_counts": dict(sorted(stop_node_counts.items(), key=lambda x: x[1], reverse=True)),
            },
        }
    results["mixed"] = mixed_summary(results["val"]["metrics"], results["ood"]["metrics"])
    return results


def main():
    args = parse_args()
    if not args.features_dir:
        raise ValueError("Missing features.dir or inference.features_dir in config")

    configure_reproducibility(args.seed)
    device = available_device(args.device)

    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    features_dir = Path(args.features_dir)
    train_payload = load_feature_file(features_dir / "train-features.pt")
    train_features = train_payload["features"].float()
    image_dim = int(train_features.shape[1])
    examples = build_positive_edge_examples(hierarchy, train_payload)
    grouped = group_examples_by_parent_child(examples)
    parents = sorted(parent for parent in grouped if parent != "root" and len(grouped[parent]) >= 2)
    if not parents:
        raise RuntimeError("No eligible non-root parents for leave-child-out unknown training")

    positive_ckpt = load_idea3_checkpoint(args.positive_checkpoint, map_location="cpu")
    prompt_cfg = HierPromptConfig.from_dict(positive_ckpt.get("prompt_config", {}))
    image_adapter = load_positive_image_adapter(positive_ckpt, image_dim, device)

    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    text_encoder = SoftPromptTextEncoder(
        backend.model,
        backend.processor.tokenizer,
        max_length=prompt_cfg.max_length,
    )
    positive = PositivePromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    positive.load_state_dict(positive_ckpt["positive_state_dict"])
    positive.eval()
    for param in positive.parameters():
        param.requires_grad_(False)

    unknown = UnknownPromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    optimizer = torch.optim.AdamW(unknown.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    max_examples = max(2, args.batch_size // max(1, args.parents_per_step))
    history = []

    for epoch in range(1, args.epochs + 1):
        unknown.train()
        shuffled = list(parents)
        rng.shuffle(shuffled)
        chunks = [
            shuffled[i:i + args.parents_per_step]
            for i in range(0, len(shuffled), args.parents_per_step)
        ]
        iterator = tqdm(chunks, desc=f"unknown epoch {epoch}/{args.epochs}", leave=False) if tqdm else chunks
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        for parent_chunk in iterator:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            step_accs = []
            for parent in parent_chunk:
                episode = sample_leave_child_out_episode(
                    parent,
                    grouped[parent],
                    strategy=args.hide_strategy,
                    max_examples=max_examples,
                    rng=rng,
                )
                if episode is None:
                    continue

                image_features = image_adapter(gather_image_features(train_features, episode.examples, device))
                with torch.no_grad():
                    child_features = positive.encode_children(parent, episode.known_children).detach()
                unknown_feature = unknown.encode_unknown(parent)
                candidate_features = torch.cat([child_features, unknown_feature.unsqueeze(0)], dim=0)
                child_to_idx = {child: idx for idx, child in enumerate(episode.known_children)}
                targets = torch.tensor(
                    [
                        len(episode.known_children) if label == UNKNOWN_LABEL else child_to_idx[label]
                        for label in episode.labels
                    ],
                    dtype=torch.long,
                    device=device,
                )
                ce_loss, stats = unknown_ce_loss(image_features, candidate_features, targets, tau=args.tau)
                parent_feature = unknown._parent_feature(parent)
                reg_loss, _ = unknown_regularization(
                    unknown_feature,
                    parent_feature,
                    child_features,
                    lambda_anchor=args.lambda_anchor,
                    lambda_child_sep=args.lambda_child_sep,
                )
                loss = ce_loss + reg_loss
                losses.append(loss)
                step_accs.append(stats["acc"])

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_acc += sum(step_accs) / max(1, len(step_accs))
            steps += 1

        epoch_stats = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, steps),
            "episode_acc": epoch_acc / max(1, steps),
            "steps": steps,
        }
        history.append(epoch_stats)
        print(f"epoch {epoch}: loss={epoch_stats['loss']:.6f}, episode_acc={epoch_stats['episode_acc']:.6f}")

    metrics = {"train_history": history}
    result = None
    if args.eval_after_training:
        result = evaluate_parent_unknown(args, hierarchy, positive, unknown, image_adapter, features_dir, device)
        ensure_dir(Path(args.result_path).parent)
        torch.save(result, args.result_path)
        metrics["final"] = {
            "val_balanced_acc": float(result["val"]["metrics"]["balanced_acc"]),
            "ood_balanced_acc": float(result["ood"]["metrics"]["balanced_acc"]),
            "mixed_balanced_acc": float(result["mixed"]["mixed_balanced_acc"]),
            "val_balanced_hdist": float(result["val"]["metrics"]["balanced_hdist"]),
            "ood_balanced_hdist": float(result["ood"]["metrics"]["balanced_hdist"]),
            "mixed_balanced_hdist": float(result["mixed"]["mixed_balanced_hdist"]),
        }
        print(f"saved result: {args.result_path}")

    ckpt_path = save_idea3_checkpoint(
        args.checkpoint,
        stage="unknown",
        dataset=args.dataset,
        clip_model=args.clip_model,
        hierarchy=args.hierarchy,
        id_split=args.id_split,
        prompt_config=prompt_cfg.to_dict(),
        positive_state_dict=positive.state_dict(),
        unknown_state_dict=unknown.state_dict(),
        image_adapter_config=positive_ckpt.get("image_adapter_config") or {"mode": "none"},
        image_adapter_state_dict=image_adapter.state_dict(),
        positive_checkpoint=args.positive_checkpoint,
        metrics=metrics,
        args=vars(args),
    )
    save_json(args.diagnostics_path, metrics)
    print(f"saved checkpoint: {ckpt_path}")
    print(f"saved diagnostics: {args.diagnostics_path}")


if __name__ == "__main__":
    main()
