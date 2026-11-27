from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from core.config import load_merged_config
from core.density import gaussian_logpdf
from core.feature_io import load_artifact, save_artifact
from core.hierarchy_inference import _cgm_child_weights
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def move_tensors_to_device(payload: Any, device: torch.device) -> Any:
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    if isinstance(payload, dict):
        return {key: move_tensors_to_device(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [move_tensors_to_device(value, device) for value in payload]
    if isinstance(payload, tuple):
        return tuple(move_tensors_to_device(value, device) for value in payload)
    return payload


def build_parent_samples(train_artifact: dict[str, Any], hierarchy: Hierarchy):
    sample_ids_by_parent: dict[str, list[int]] = defaultdict(list)
    child_ids_by_parent: dict[str, list[int]] = defaultdict(list)
    node_name_to_idx = {name: idx for idx, name in enumerate(hierarchy.id_node_list)}
    class_names = train_artifact["class_names"]
    targets = train_artifact["targets"].tolist()

    for sample_idx, leaf_target in enumerate(targets):
        leaf_name = class_names[int(leaf_target)]
        if leaf_name not in node_name_to_idx:
            continue
        leaf_idx = node_name_to_idx[leaf_name]
        path = list(hierarchy.node_ancestors[leaf_name]) + [leaf_idx]
        for pos in range(len(path) - 1):
            parent_name = hierarchy.id_node_list[path[pos]]
            child_idx = int(path[pos + 1])
            sample_ids_by_parent[parent_name].append(sample_idx)
            child_ids_by_parent[parent_name].append(child_idx)

    return sample_ids_by_parent, child_ids_by_parent


def fit_thresholds(
    train_features: torch.Tensor,
    train_artifact: dict[str, Any],
    hierarchy: Hierarchy,
    density_payload: dict[str, Any],
    coverage: float,
    score_mode: str,
    batch_size: int,
    device: torch.device,
):
    if coverage <= 0.0 or coverage >= 1.0:
        raise ValueError("coverage must satisfy 0 < coverage < 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    sample_ids_by_parent, child_ids_by_parent = build_parent_samples(train_artifact, hierarchy)
    score_mode = score_mode.lower()
    if score_mode not in {"posterior", "logpdf"}:
        raise ValueError("score_mode must be one of: posterior, logpdf")
    thresholds = {}
    counts = {}
    node_score_values: dict[str, list[torch.Tensor]] = defaultdict(list)
    means = density_payload["means"]
    variances = density_payload.get("variances")
    covariance_matrices = density_payload.get("covariance_matrices")
    shared_covariance = density_payload.get("shared_covariance")
    covariance_type = density_payload.get("covariance_type", "diag")

    for parent_name in sorted(hierarchy.parent2children):
        children = hierarchy.parent2children[parent_name]
        child_indices = [hierarchy.id_node_list.index(child) for child in children]
        if len(child_indices) <= 1 or parent_name not in sample_ids_by_parent:
            thresholds[parent_name] = 1.0
            counts[parent_name] = 0
            continue

        sample_ids = torch.tensor(sample_ids_by_parent[parent_name], dtype=torch.long)
        true_children = torch.tensor(child_ids_by_parent[parent_name], dtype=torch.long)
        child_local_index = {child_idx: local for local, child_idx in enumerate(child_indices)}
        true_local = torch.tensor(
            [child_local_index[int(child_idx)] for child_idx in true_children.tolist()],
            dtype=torch.long,
        )
        scores = []
        child_weights = _cgm_child_weights(
            child_indices,
            density_payload,
            "count",
            dtype=train_features.dtype,
            device=device,
        )
        log_child_weights = torch.log(child_weights.clamp_min(1e-12)).unsqueeze(0)

        for start in range(0, int(sample_ids.numel()), batch_size):
            batch_ids = sample_ids[start : start + batch_size]
            batch_features = train_features[batch_ids].to(device)
            child_logpdf = gaussian_logpdf(
                batch_features,
                means,
                variances,
                covariance_matrices=covariance_matrices,
                shared_covariance=shared_covariance,
                covariance_type=covariance_type,
                node_indices=child_indices,
            )
            local_targets = true_local[start : start + batch_size].to(device)
            if score_mode == "posterior":
                posteriors = torch.softmax(child_logpdf + log_child_weights, dim=1)
                true_probs = posteriors.gather(1, local_targets.unsqueeze(1)).squeeze(1)
                batch_scores = 1.0 - true_probs
            else:
                true_logpdf = child_logpdf.gather(1, local_targets.unsqueeze(1)).squeeze(1)
                batch_scores = -true_logpdf
                absolute_children = true_children[start : start + batch_size]
                for child_idx in absolute_children.unique().tolist():
                    child_mask = absolute_children == int(child_idx)
                    child_name = hierarchy.id_node_list[int(child_idx)]
                    node_score_values[child_name].append(batch_scores[child_mask].detach().cpu())
            scores.append(batch_scores.detach().cpu())

        parent_scores = torch.cat(scores, dim=0)
        thresholds[parent_name] = float(torch.quantile(parent_scores, coverage).item())
        counts[parent_name] = int(parent_scores.numel())

    calibration = {
        "method": f"conformal_{score_mode}_stop",
        "coverage": float(coverage),
        "score": "1 - posterior_true_child" if score_mode == "posterior" else "-log p_true_child(x)",
        "thresholds": thresholds,
        "counts": counts,
    }
    if score_mode == "logpdf":
        node_thresholds = {}
        node_counts = {}
        for node_name, values in sorted(node_score_values.items()):
            node_scores = torch.cat(values, dim=0)
            node_thresholds[node_name] = float(torch.quantile(node_scores, coverage).item())
            node_counts[node_name] = int(node_scores.numel())
        calibration["node_thresholds"] = node_thresholds
        calibration["node_counts"] = node_counts
    return calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    parser.add_argument("--density-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--score-mode", choices=["posterior", "logpdf"], default="posterior")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_merged_config(args.config, args.feature_gen_config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    train_artifact = load_artifact(experiment_dir / "features_train.pt")
    train_features, _ = resolve_feature_tensor(config, experiment_dir, "train")
    density_payload = load_artifact(args.density_file)
    device_name = args.device or config.get("experiment", {}).get("device", "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    density_payload = move_tensors_to_device(density_payload, device)
    train_features = train_features.float()
    calibration = fit_thresholds(
        train_features,
        train_artifact,
        hierarchy,
        density_payload,
        coverage=float(args.coverage),
        score_mode=args.score_mode,
        batch_size=int(args.batch_size),
        device=device,
    )
    density_payload = move_tensors_to_device(density_payload, torch.device("cpu"))
    density_payload.setdefault("cgm_calibration", {})
    density_payload["cgm_calibration"][calibration["method"]] = calibration
    save_artifact(density_payload, args.output)
    print(f"Saved conformal calibration to {args.output}")


if __name__ == "__main__":
    main()
