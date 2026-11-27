from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from core.config import load_config
from core.density import score_nodes
from core.eval import evaluate_predictions
from core.feature_io import load_artifact
from core.hierarchy_inference import hierarchical_node_probabilities, predict_from_probabilities
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


SCALAR_METRICS = ["acc", "balanced_acc", "avg_hdist", "balanced_hdist"]
OBJECTIVES = SCALAR_METRICS + ["mixed_balanced_acc"]
HIGHER_IS_BETTER = {"acc", "balanced_acc", "mixed_balanced_acc"}
LOWER_IS_BETTER = {"avg_hdist", "balanced_hdist"}


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def format_vector(values: list[float] | tuple[float, ...]) -> str:
    return "[" + ",".join(f"{float(v):.6g}" for v in values) + "]"


def metric_better(candidate: dict[str, float], incumbent: dict[str, float] | None, metric: str) -> bool:
    if incumbent is None:
        return True
    if metric in HIGHER_IS_BETTER:
        return candidate[metric] > incumbent[metric]
    if metric in LOWER_IS_BETTER:
        return candidate[metric] < incumbent[metric]
    raise ValueError(f"Unsupported metric: {metric}")


def objective_metrics(result: dict[str, dict[str, float]], target_split: str, metric: str) -> dict[str, float]:
    if metric == "mixed_balanced_acc":
        return {"mixed_balanced_acc": 0.5 * (result["val"]["balanced_acc"] + result["ood"]["balanced_acc"])}
    return result[target_split]


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


def precompute_node_scores(features, density_payload, inference_cfg, device, batch_size, cache_device):
    scores = []
    n_samples = int(features.shape[0])
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch_features = features[start : start + batch_size].to(device)
            batch_scores = score_nodes(
                batch_features,
                density_payload["means"],
                density_payload.get("variances"),
                covariance_matrices=density_payload.get("covariance_matrices"),
                shared_covariance=density_payload.get("shared_covariance"),
                mean_directions=density_payload.get("mean_directions"),
                covariance_type=density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type", "diag")),
                score_type=inference_cfg.get("score_type", "gaussian_loglik"),
                kappa=inference_cfg.get("kappa", 20.0),
            )
            scores.append(batch_scores.to(cache_device))
    return torch.cat(scores, dim=0)


def evaluate_from_scores(split_artifact, node_scores, hierarchy, density_payload, inference_cfg, device, batch_size):
    preds = []
    n_samples = int(node_scores.shape[0])
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch_scores = node_scores[start : start + batch_size].to(device)
            final_probs, _ = hierarchical_node_probabilities(
                None,
                hierarchy,
                density_payload,
                score_type=inference_cfg.get("score_type", "gaussian_loglik"),
                temperature=inference_cfg.get("temperature", 1.0),
                kappa=inference_cfg.get("kappa", 20.0),
                include_debug=False,
                node_scores=batch_scores,
                cgm_cfg={"enabled": False},
                ood_scale=inference_cfg.get("ood_scale", 1.0),
            )
            batch_preds = predict_from_probabilities(
                final_probs,
                hierarchy,
                mode=inference_cfg.get("prediction_mode", "argmax"),
            )
            preds.append(batch_preds.cpu())
    preds = torch.cat(preds, dim=0)
    metrics = evaluate_predictions(preds, split_artifact["node_targets"], hierarchy)
    return {
        metric: float(metrics[metric].item() if isinstance(metrics[metric], torch.Tensor) else metrics[metric])
        for metric in SCALAR_METRICS
    }


def evaluate_vector(temp_vector, artifacts, base_inference_cfg, hierarchy, density_payload, device, batch_size):
    inference_cfg = deepcopy(base_inference_cfg)
    inference_cfg["temperature"] = list(temp_vector)
    return {
        "temperature_vector": format_vector(temp_vector),
        "val": evaluate_from_scores(
            artifacts["val_artifact"],
            artifacts["val_scores"],
            hierarchy,
            density_payload,
            inference_cfg,
            device,
            batch_size,
        ),
        "ood": evaluate_from_scores(
            artifacts["ood_artifact"],
            artifacts["ood_scores"],
            hierarchy,
            density_payload,
            inference_cfg,
            device,
            batch_size,
        ),
    }


def write_rows(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "depth",
        "candidate_tau",
        "temperature_vector",
        "val_acc",
        "val_balanced_acc",
        "val_avg_hdist",
        "val_balanced_hdist",
        "ood_acc",
        "ood_balanced_acc",
        "ood_avg_hdist",
        "ood_balanced_hdist",
        "mixed_balanced_acc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--taus", default="0.5,1,2,3,5,7,10,15,20,30,50")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--initial-temperature", default=None, help="Comma-separated initial vector. Defaults to config inference.temperature.")
    parser.add_argument("--depths", default=None, help="Comma-separated 1-based depths to tune. Defaults to all depths.")
    parser.add_argument("--target-split", default="ood", choices=["val", "ood"])
    parser.add_argument("--metric", default="acc", choices=OBJECTIVES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--cache-scores-on-cpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("cgm", {})["enabled"] = False
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "temperature_greedy"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    cache_device = torch.device("cpu") if args.cache_scores_on_cpu else device

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    density_payload = move_tensors_to_device(load_artifact(experiment_dir / "node_density.pt"), device)

    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, _ = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, _ = resolve_feature_tensor(config, experiment_dir, "ood")
    base_inference_cfg = dict(config["inference"])
    base_inference_cfg["score_type"] = "gaussian_loglik"

    print("Precomputing val node scores...", flush=True)
    val_scores = precompute_node_scores(
        val_features,
        density_payload,
        base_inference_cfg,
        device,
        args.score_batch_size,
        cache_device,
    )
    print("Precomputing ood node scores...", flush=True)
    ood_scores = precompute_node_scores(
        ood_features,
        density_payload,
        base_inference_cfg,
        device,
        args.score_batch_size,
        cache_device,
    )

    artifacts = {
        "val_artifact": val_artifact,
        "ood_artifact": ood_artifact,
        "val_scores": val_scores,
        "ood_scores": ood_scores,
    }
    taus = parse_float_list(args.taus)
    n_depths = hierarchy.max_depth
    initial_temperature = parse_float_list(args.initial_temperature) if args.initial_temperature else base_inference_cfg.get("temperature", 1.0)
    if isinstance(initial_temperature, (list, tuple)):
        if len(initial_temperature) != n_depths:
            raise ValueError(
                f"Initial temperature length {len(initial_temperature)} does not match max depth {n_depths}"
            )
        current = [float(temp) for temp in initial_temperature]
    else:
        current = [float(initial_temperature) for _ in range(n_depths)]
    tune_depth_positions = list(range(n_depths))
    if args.depths:
        tune_depth_positions = [int(depth) - 1 for depth in args.depths.split(",") if depth.strip()]
        if any(depth < 0 or depth >= n_depths for depth in tune_depth_positions):
            raise ValueError(f"--depths must contain values in [1, {n_depths}]")
    rows = []
    best = None

    for pass_idx in range(args.passes):
        improved = False
        for depth_pos in tune_depth_positions:
            depth_best = None
            depth_best_vector = current
            for tau in taus:
                candidate = list(current)
                candidate[depth_pos] = tau
                result = evaluate_vector(candidate, artifacts, base_inference_cfg, hierarchy, density_payload, device, args.eval_batch_size)
                row = {
                    "stage": f"pass_{pass_idx + 1}",
                    "depth": depth_pos + 1,
                    "candidate_tau": tau,
                    "temperature_vector": result["temperature_vector"],
                    "val_acc": result["val"]["acc"],
                    "val_balanced_acc": result["val"]["balanced_acc"],
                    "val_avg_hdist": result["val"]["avg_hdist"],
                    "val_balanced_hdist": result["val"]["balanced_hdist"],
                    "ood_acc": result["ood"]["acc"],
                    "ood_balanced_acc": result["ood"]["balanced_acc"],
                    "ood_avg_hdist": result["ood"]["avg_hdist"],
                    "ood_balanced_hdist": result["ood"]["balanced_hdist"],
                    "mixed_balanced_acc": 0.5 * (result["val"]["balanced_acc"] + result["ood"]["balanced_acc"]),
                }
                rows.append(row)
                write_rows(rows, output_dir / "temperature_greedy_results.csv")
                target_metrics = objective_metrics(result, args.target_split, args.metric)
                if metric_better(target_metrics, depth_best, args.metric):
                    depth_best = target_metrics
                    depth_best_vector = candidate
                if metric_better(target_metrics, best, args.metric):
                    best = target_metrics
            if depth_best_vector != current:
                improved = True
            current = depth_best_vector
            print(
                f"pass={pass_idx + 1} depth={depth_pos + 1} current={format_vector(current)} "
                f"{args.metric}={depth_best[args.metric]:.6g}",
                flush=True,
            )
        if not improved:
            break

    final_result = evaluate_vector(current, artifacts, base_inference_cfg, hierarchy, density_payload, device, args.eval_batch_size)
    summary = {
        "stage": "final",
        "depth": 0,
        "candidate_tau": "",
        "temperature_vector": final_result["temperature_vector"],
        "val_acc": final_result["val"]["acc"],
        "val_balanced_acc": final_result["val"]["balanced_acc"],
        "val_avg_hdist": final_result["val"]["avg_hdist"],
        "val_balanced_hdist": final_result["val"]["balanced_hdist"],
        "ood_acc": final_result["ood"]["acc"],
        "ood_balanced_acc": final_result["ood"]["balanced_acc"],
        "ood_avg_hdist": final_result["ood"]["avg_hdist"],
        "ood_balanced_hdist": final_result["ood"]["balanced_hdist"],
        "mixed_balanced_acc": 0.5 * (final_result["val"]["balanced_acc"] + final_result["ood"]["balanced_acc"]),
    }
    rows.append(summary)
    write_rows(rows, output_dir / "temperature_greedy_results.csv")
    write_rows([summary], output_dir / "temperature_greedy_best.csv")
    print("Best temperature:", summary["temperature_vector"], flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
