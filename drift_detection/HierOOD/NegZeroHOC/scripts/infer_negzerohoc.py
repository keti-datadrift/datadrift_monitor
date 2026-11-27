from __future__ import annotations

import argparse
from argparse import Namespace
from collections import Counter
import sys
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.clip_backend import ClipBackend, safe_model_name
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.evaluation import (
    build_hierarchy,
    evaluate_split,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, load_feature_file
from negzerohoc.inference import predict_features
from negzerohoc.output_layout import experiment_artifact_path, resolve_shared_feature_dir
from negzerohoc.runtime import available_device, configured_device
from negzerohoc.semantic_index import build_semantic_index


def load_config(path):
    cfg = load_yaml_config(path)
    dataset_cfg = cfg.get("dataset", {})
    runtime_cfg = cfg.get("runtime", {})
    clip_cfg = cfg.get("clip", {})
    inference_cfg = cfg.get("inference", {})
    experiment_cfg = cfg.get("experiment", {})

    mode = inference_cfg.get("mode")
    if mode not in {"child_only", "manual_unknown"}:
        raise ValueError(
            f"Invalid or missing inference.mode in {path}: {mode}. "
            "Expected one of: child_only, manual_unknown"
        )
    experiment_name = experiment_cfg.get("name", f"fgvc-aircraft-{mode}")
    output_root = experiment_cfg.get("output_root", "outputs")
    features_dir = resolve_shared_feature_dir(
        inference_cfg.get("features_dir"),
        output_root=output_root,
    )
    if not features_dir:
        raise ValueError(f"Missing required config key: inference.features_dir in {path}")
    batch_size = int(inference_cfg.get("batch_size", 1024))
    if batch_size <= 0:
        raise ValueError(f"inference.batch_size must be a positive integer in {path}")

    return Namespace(
        config=str(path),
        experiment_name=experiment_name,
        dataset=dataset_cfg.get("name", "fgvc-aircraft"),
        hierarchy=dataset_cfg.get("hierarchy", "hierarchies/fgvc-aircraft.json"),
        id_split=dataset_cfg.get("id_split", "data/fgvc-aircraft-id-labels.csv"),
        features_dir=str(features_dir),
        clip_model=clip_cfg.get("model", "openai/clip-vit-base-patch32"),
        mode=mode,
        outdir=output_root,
        device=configured_device(runtime_cfg),
        tau=inference_cfg.get("tau", 1.0),
        batch_size=batch_size,
        allow_root_unknown=inference_cfg.get("allow_root_unknown", False),
        local_files_only=clip_cfg.get("local_files_only", True),
        save_trace=inference_cfg.get("save_trace", False),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    config_arg = parser.parse_args()
    return load_config(config_arg.config)


@torch.no_grad()
def predict_payload_in_batches(
    payload,
    hierarchy,
    semantic_index,
    mode: str,
    tau: float,
    batch_size: int,
    device: str,
    split_name: str,
    return_trace: bool,
):
    features = payload["features"]
    num_features = int(features.shape[0])
    preds = []
    traces = [] if return_trace else None
    stop_depth_counts = Counter()
    stop_node_counts = Counter()

    starts = range(0, num_features, batch_size)
    if tqdm is not None:
        total = (num_features + batch_size - 1) // batch_size
        starts = tqdm(starts, total=total, desc=f"infer {split_name}", leave=False)

    for start in starts:
        end = min(start + batch_size, num_features)
        batch_features = features[start:end].to(device, non_blocking=True)
        pred_out = predict_features(
            batch_features,
            hierarchy,
            semantic_index,
            mode=mode,
            tau=tau,
            return_trace=return_trace,
        )
        preds.append(pred_out["preds"].cpu())

        diagnostics = pred_out["diagnostics"]
        stop_depth_counts.update({
            int(depth): int(count)
            for depth, count in diagnostics["stop_depth_counts"].items()
        })
        stop_node_counts.update({
            str(node): int(count)
            for node, count in diagnostics["stop_node_counts"].items()
        })
        if return_trace:
            traces.extend(pred_out["traces"])

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
    device = available_device(args.device)
    hierarchy, _ = build_hierarchy(REPO_ROOT, args.id_split, args.hierarchy)
    dists_mats = make_distance_mats(hierarchy)

    backend = ClipBackend(args.clip_model, device=device, local_files_only=args.local_files_only)
    semantic_index = build_semantic_index(
        args.dataset,
        hierarchy,
        backend,
        args.mode,
        allow_root_unknown=args.allow_root_unknown,
    )

    features_dir = Path(args.features_dir)
    val_payload = load_feature_file(features_dir / "val-features.pt")
    ood_payload = load_feature_file(features_dir / "ood-features.pt")

    results = {
        "args": vars(args),
        "mode": args.mode,
        "hierarchy_id_node_list": list(hierarchy.id_node_list),
        "semantic_prompts": {
            parent: local.prompts for parent, local in semantic_index.items()
        },
    }

    for split_name, payload in [("val", val_payload), ("ood", ood_payload)]:
        pred_out = predict_payload_in_batches(
            payload,
            hierarchy,
            semantic_index,
            mode=args.mode,
            tau=args.tau,
            batch_size=args.batch_size,
            device=device,
            split_name=split_name,
            return_trace=args.save_trace,
        )
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

    model_key = f"clip_{safe_model_name(args.clip_model)}"
    save_path = experiment_artifact_path(
        args.outdir,
        args.experiment_name,
        "results",
        f"negzerohoc-{args.dataset}-{model_key}-{args.mode}.result",
    )
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
