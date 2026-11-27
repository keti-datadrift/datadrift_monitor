from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter
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

from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.clip_backend import ClipBackend
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import build_hierarchy, evaluate_split, make_distance_mats, mixed_summary
from negzerohoc.feature_io import ensure_dir, load_feature_file
from negzerohoc.image_adapters import build_image_adapter
from negzerohoc.idea3_inference import build_idea3_semantic_index, predict_features_idea3
from negzerohoc.output_layout import resolve_experiment_artifact, resolve_shared_feature_dir
from negzerohoc.prompt_models import HierPromptConfig, PositivePromptLearner, UnknownPromptLearner
from negzerohoc.runtime import available_device, configured_device
from negzerohoc.soft_prompting import SoftPromptTextEncoder


def load_config(path, mode_override=None):
    cfg = load_yaml_config(path)
    experiment_cfg = cfg.get("experiment", {})
    runtime_cfg = cfg.get("runtime", {})
    dataset_cfg = cfg.get("dataset", {})
    clip_cfg = cfg.get("clip", {})
    features_cfg = cfg.get("features", {})
    inference_cfg = cfg.get("inference", {})

    experiment_name = experiment_cfg.get("name", "idea3")
    output_root = experiment_cfg.get("output_root", "outputs")
    mode = mode_override or inference_cfg.get("mode", "positive_child_only")
    if mode not in {
        "positive_child_only",
        "positive_global_path",
        "positive_pathscore_diagnostic",
        "parent_unknown",
    }:
        raise ValueError(f"Unsupported Idea 3 inference mode: {mode}")

    checkpoint = inference_cfg.get("checkpoint")
    if mode == "parent_unknown":
        checkpoint = inference_cfg.get("unknown_checkpoint") or checkpoint
    if not checkpoint:
        checkpoint = inference_cfg.get("positive_checkpoint")
    checkpoint = resolve_experiment_artifact(
        checkpoint,
        output_root=output_root,
        experiment_name=experiment_name,
        kind="checkpoints",
        default_filename=(
            f"{experiment_name}-parent_unknown.pt"
            if mode == "parent_unknown"
            else f"{experiment_name}-positive.pt"
        ),
    )
    result_path = resolve_experiment_artifact(
        inference_cfg.get("result_path"),
        output_root=output_root,
        experiment_name=experiment_name,
        kind="results",
        default_filename=f"{experiment_name}-{mode}.result",
    )

    return Namespace(
        config=str(path),
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
        checkpoint=str(checkpoint),
        result_path=str(result_path),
        mode=mode,
        batch_size=int(inference_cfg.get("batch_size", 1024)),
        tau=float(inference_cfg.get("tau", 1.0)),
        allow_root_unknown=bool(inference_cfg.get("allow_root_unknown", False)),
        save_trace=bool(inference_cfg.get("save_trace", False)),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default=None)
    parsed = parser.parse_args()
    return load_config(parsed.config, parsed.mode)


def load_models(args, hierarchy, device):
    checkpoint = load_idea3_checkpoint(args.checkpoint, map_location="cpu")
    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    text_encoder = SoftPromptTextEncoder(
        backend.model,
        backend.processor.tokenizer,
        max_length=int(checkpoint.get("prompt_config", {}).get("max_length", 77)),
    )
    prompt_cfg = HierPromptConfig.from_dict(checkpoint.get("prompt_config", {}))

    positive = PositivePromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
    positive_state = checkpoint.get("positive_state_dict")
    if positive_state is None:
        raise ValueError(f"Checkpoint has no positive_state_dict: {args.checkpoint}")
    positive.load_state_dict(positive_state)
    positive.eval()

    unknown = None
    if args.mode == "parent_unknown":
        unknown_state = checkpoint.get("unknown_state_dict")
        if unknown_state is None:
            raise ValueError(
                f"parent_unknown mode requires unknown_state_dict. "
                f"Use an unknown checkpoint, got: {args.checkpoint}"
            )
        unknown = UnknownPromptLearner(args.dataset, hierarchy, text_encoder, prompt_cfg).to(device)
        unknown.load_state_dict(unknown_state)
        unknown.eval()

    return checkpoint, positive, unknown


def load_image_adapter(checkpoint, input_dim: int, device):
    cfg = checkpoint.get("image_adapter_config") or {"mode": "none"}
    adapter = build_image_adapter(input_dim, cfg).to(device)
    state = checkpoint.get("image_adapter_state_dict")
    if state is not None:
        adapter.load_state_dict(state)
    adapter.eval()
    return adapter


@torch.no_grad()
def predict_payload_in_batches(args, payload, hierarchy, semantic_index, image_adapter, device, split_name):
    features = payload["features"]
    num_features = int(features.shape[0])
    starts = range(0, num_features, args.batch_size)
    if tqdm is not None:
        total = (num_features + args.batch_size - 1) // args.batch_size
        starts = tqdm(starts, total=total, desc=f"infer {split_name}", leave=False)

    preds = []
    traces = [] if args.save_trace else None
    stop_depth_counts = Counter()
    stop_node_counts = Counter()

    for start in starts:
        end = min(start + args.batch_size, num_features)
        image_features = image_adapter(features[start:end].to(device))
        out = predict_features_idea3(
            image_features,
            hierarchy,
            semantic_index,
            mode=args.mode,
            tau=args.tau,
            return_trace=args.save_trace,
        )
        preds.append(out["preds"].cpu())
        stop_depth_counts.update({int(k): int(v) for k, v in out["diagnostics"]["stop_depth_counts"].items()})
        stop_node_counts.update({str(k): int(v) for k, v in out["diagnostics"]["stop_node_counts"].items()})
        if args.save_trace:
            traces.extend(out["traces"])

    return {
        "preds": torch.cat(preds) if preds else torch.empty(0, dtype=torch.long),
        "traces": traces,
        "diagnostics": {
            "stop_depth_counts": dict(sorted(stop_depth_counts.items())),
            "stop_node_counts": dict(stop_node_counts.most_common()),
        },
    }


def main():
    args = parse_args()
    if not args.features_dir:
        raise ValueError("Missing features.dir or inference.features_dir in config")

    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    dists_mats = make_distance_mats(hierarchy)
    checkpoint, positive, unknown = load_models(args, hierarchy, device)
    semantic_index = build_idea3_semantic_index(
        hierarchy,
        positive,
        unknown,
        mode=args.mode,
        allow_root_unknown=args.allow_root_unknown,
    )

    features_dir = Path(args.features_dir)
    val_payload = load_feature_file(features_dir / "val-features.pt")
    ood_payload = load_feature_file(features_dir / "ood-features.pt")
    image_adapter = load_image_adapter(checkpoint, int(val_payload["features"].shape[1]), device)

    results = {
        "args": vars(args),
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "checkpoint_stage": checkpoint.get("stage"),
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "semantic_prompts": {parent: local.prompts for parent, local in semantic_index.items()},
    }

    for split_name, payload in [("val", val_payload), ("ood", ood_payload)]:
        pred_out = predict_payload_in_batches(args, payload, hierarchy, semantic_index, image_adapter, device, split_name)
        preds = pred_out["preds"].cpu()
        node_labels, metrics = evaluate_split(hierarchy, payload, preds, dists_mats=dists_mats)
        results[split_name] = {
            "preds": preds,
            "targets": node_labels.cpu(),
            "metrics": metrics,
            "diagnostics": pred_out["diagnostics"],
        }
        if args.save_trace:
            results[split_name]["trace"] = pred_out["traces"]

    results["mixed"] = mixed_summary(results["val"]["metrics"], results["ood"]["metrics"])

    save_path = Path(args.result_path)
    ensure_dir(save_path.parent)
    torch.save(results, save_path)

    print(f"saved: {save_path}")
    print(f"ID BAcc: {results['val']['metrics']['balanced_acc']:.6f}")
    print(f"OOD BAcc: {results['ood']['metrics']['balanced_acc']:.6f}")
    print(f"Mixed BAcc: {results['mixed']['mixed_balanced_acc']:.6f}")
    print(f"ID Balanced H-Dist: {float(results['val']['metrics']['balanced_hdist']):.6f}")
    print(f"OOD Balanced H-Dist: {float(results['ood']['metrics']['balanced_hdist']):.6f}")
    print(f"Mixed Balanced H-Dist: {results['mixed']['mixed_balanced_hdist']:.6f}")


if __name__ == "__main__":
    main()
