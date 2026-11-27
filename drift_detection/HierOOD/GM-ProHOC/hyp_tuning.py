from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from compare_results import load_result_rows
from core.config import load_config
from core.eval import evaluate_predictions
from core.feature_io import load_artifact
from core.hierarchy_inference import hierarchical_node_probabilities, predict_from_probabilities
from core.density import score_nodes
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


SCALAR_METRICS = ["acc", "balanced_acc", "avg_hdist", "balanced_hdist"]
HIGHER_IS_BETTER = {"acc", "balanced_acc"}
LOWER_IS_BETTER = {"avg_hdist", "balanced_hdist"}


def parse_list_arg(raw: str | None, default_values: list[float]) -> list[float]:
    if raw is None:
        return default_values
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def default_log_grid(start: float, end: float, num: int) -> list[float]:
    if num <= 1:
        return [float(start)]
    values = torch.logspace(
        start=torch.log10(torch.tensor(float(start))),
        end=torch.log10(torch.tensor(float(end))),
        steps=num,
    )
    return [float(v.item()) for v in values]


def metric_sort(metric: str):
    if metric in HIGHER_IS_BETTER:
        return False
    if metric in LOWER_IS_BETTER:
        return True
    raise ValueError(f"Unsupported metric: {metric}")


def format_temperature_vector(temperature_vector: list[float] | tuple[float, ...]) -> str:
    return "[" + ",".join(f"{float(t):.6g}" for t in temperature_vector) + "]"


def format_depth_vector(values: list[float] | tuple[float, ...]) -> str:
    return "[" + ",".join(f"{float(v):.6g}" for v in values) + "]"


def build_inference_cfg(
    base_cfg: dict[str, Any],
    temperature: list[float],
    alpha: list[float],
    beta: list[float],
) -> dict[str, Any]:
    inference_cfg = dict(base_cfg)
    inference_cfg["temperature"] = [float(t) for t in temperature]
    inference_cfg["alpha"] = [float(a) for a in alpha]
    inference_cfg["beta"] = [float(b) for b in beta]
    return inference_cfg


def metric_lookup(
    df: pd.DataFrame,
    split: str,
    temperature_vector: str,
    alpha_vector: str,
    beta_vector: str,
) -> pd.Series | None:
    matched = df[
        (df["split"] == split)
        & (df["temperature_vector"] == temperature_vector)
        & (df["alpha_vector"] == alpha_vector)
        & (df["beta_vector"] == beta_vector)
    ]
    if matched.empty:
        return None
    return matched.iloc[0]


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


def evaluate_split_for_tuning(
    split_artifact,
    features: torch.Tensor,
    hierarchy,
    density_payload,
    inference_cfg,
    feature_meta,
    device: torch.device,
    eval_batch_size: int,
    node_scores: torch.Tensor | None = None,
):
    preds = []
    n_samples = int(features.shape[0])
    batch_size = n_samples if eval_batch_size <= 0 else int(eval_batch_size)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch_features = None if node_scores is not None else features[start : start + batch_size].to(device)
            batch_scores = None if node_scores is None else node_scores[start : start + batch_size].to(device)
            final_probs, _ = hierarchical_node_probabilities(
                batch_features,
                hierarchy,
                density_payload,
                score_type=inference_cfg.get("score_type", "gaussian_loglik"),
                temperature=inference_cfg.get("temperature", 1.0),
                kappa=inference_cfg.get("kappa", 20.0),
                alpha=inference_cfg.get("alpha", 1.0),
                beta=inference_cfg.get("beta", 1.0),
                include_debug=False,
                node_scores=batch_scores,
            )
            batch_preds = predict_from_probabilities(
                final_probs,
                hierarchy,
                mode=inference_cfg.get("prediction_mode", "argmax"),
            )
            preds.append(batch_preds.cpu())

    preds = torch.cat(preds, dim=0)
    metrics = evaluate_predictions(preds, split_artifact["node_targets"], hierarchy)
    metrics.update(
        {
            "num_samples": n_samples,
            "prediction_mode": inference_cfg.get("prediction_mode", "argmax"),
            "score_type": inference_cfg.get("score_type", "gaussian_loglik"),
            "temperature": inference_cfg.get("temperature", 1.0),
            "kappa": inference_cfg.get("kappa", 20.0),
            "alpha": inference_cfg.get("alpha", 1.0),
            "beta": inference_cfg.get("beta", 1.0),
            "collapsed_ood": inference_cfg.get("collapse_ood_to_parent", True),
            "feature_source": feature_meta,
        }
    )
    return metrics


def precompute_node_scores(
    features: torch.Tensor,
    density_payload,
    inference_cfg,
    device: torch.device,
    eval_batch_size: int,
    cache_device: torch.device,
) -> torch.Tensor:
    scores = []
    n_samples = int(features.shape[0])
    batch_size = n_samples if eval_batch_size <= 0 else int(eval_batch_size)
    with torch.no_grad():
        for start in tqdm(range(0, n_samples, batch_size), desc="Precompute node scores"):
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


def write_tuning_outputs(
    tuning_df: pd.DataFrame,
    output_dir: Path,
    target_split: str,
    topk: int,
    tuned_depths: list[int],
    experiment_dir: Path,
    config: dict[str, Any],
    prohoc: str | None = None,
):
    tuning_csv = output_dir / "tuning_results.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    top_rows: list[pd.DataFrame] = []
    ranking_df = tuning_df[tuning_df["split"] == target_split].copy()
    for metric in SCALAR_METRICS:
        ascending = metric_sort(metric)
        ranked = ranking_df.sort_values(
            by=[metric]
            + [f"tau_depth_{depth}" for depth in tuned_depths]
            + [f"alpha_depth_{depth}" for depth in tuned_depths]
            + [f"beta_depth_{depth}" for depth in tuned_depths],
            ascending=[ascending] + [True] * (len(tuned_depths) * 3),
        ).head(topk).copy()
        ranked.insert(0, "rank_metric", metric)
        display_rows = []
        for _, row in ranked.iterrows():
            combined = row.to_dict()
            for metric_name in SCALAR_METRICS:
                combined[f"{target_split}_{metric_name}"] = combined.pop(metric_name)
            paired_split = "val" if target_split == "ood" else "ood"
            paired_row = metric_lookup(
                tuning_df,
                paired_split,
                row["temperature_vector"],
                row["alpha_vector"],
                row["beta_vector"],
            )
            if paired_row is not None:
                for metric_name in SCALAR_METRICS:
                    combined[f"{paired_split}_{metric_name}"] = paired_row[metric_name]
            display_rows.append(combined)
        ranked = pd.DataFrame(display_rows)
        top_rows.append(ranked)

    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    top_csv = output_dir / f"top_{topk}_{target_split}.csv"
    top_df.to_csv(top_csv, index=False)

    summary_rows = []
    for metric in SCALAR_METRICS:
        metric_top = top_df[top_df["rank_metric"] == metric]
        for _, row in metric_top.iterrows():
            summary_rows.append(
                {
                    "metric": metric,
                    "split": target_split,
                    "temperature_vector": row["temperature_vector"],
                    "alpha_vector": row["alpha_vector"],
                    "beta_vector": row["beta_vector"],
                    "score_type": row["score_type"],
                    "covariance_type": row["covariance_type"],
                    f"{target_split}_{metric}": row[f"{target_split}_{metric}"],
                }
            )
            for depth in tuned_depths:
                summary_rows[-1][f"tau_depth_{depth}"] = row[f"tau_depth_{depth}"]
                summary_rows[-1][f"alpha_depth_{depth}"] = row[f"alpha_depth_{depth}"]
                summary_rows[-1][f"beta_depth_{depth}"] = row[f"beta_depth_{depth}"]
            paired_split = "val" if target_split == "ood" else "ood"
            for metric_name in SCALAR_METRICS:
                col = f"{paired_split}_{metric_name}"
                if col in row:
                    summary_rows[-1][col] = row[col]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"best_hparams_summary_{target_split}.csv"
    summary_df.to_csv(summary_csv, index=False)

    if prohoc:
        prohoc_rows = load_result_rows([Path(p) for p in _discover_paths(prohoc)], source_hint="prohoc")
        gm_compare_rows = []
        for _, row in ranking_df.iterrows():
            gm_compare_rows.append(
                {
                    "source": "gm_prohoc",
                    "result_file": str(experiment_dir / "hinference_density.result"),
                    "dataset": row["dataset"],
                    "method": (
                        f"{config['experiment']['name']}"
                        f"_tau{row['temperature_vector']}"
                        f"_a{row['alpha_vector']}"
                        f"_b{row['beta_vector']}"
                    ),
                    "split": row["split"],
                    "acc": row["acc"],
                    "balanced_acc": row["balanced_acc"],
                    "avg_hdist": row["avg_hdist"],
                    "balanced_hdist": row["balanced_hdist"],
                }
            )
        compare_df = pd.DataFrame(prohoc_rows + gm_compare_rows)
        compare_df.to_csv(output_dir / "comparison_with_prohoc.csv", index=False)

    print(f"Saved tuning results to: {tuning_csv}")
    print(f"Saved top-{topk} hyperparameters to: {top_csv}")
    print(f"Saved summary to: {summary_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--merge_results", default=None, help="Comma-separated tuning_results.csv files to merge and rank.")
    parser.add_argument("--prohoc", default=None, help="Optional ProHOC .result file or directory for comparison rows.")
    parser.add_argument("--target_split", default="ood", choices=["val", "ood"], help="Split used to rank top-5 hyperparameters.")
    parser.add_argument("--taus", default=None, help="Comma-separated temperatures. Default: logspace in [1, 10].")
    parser.add_argument("--alphas", default=None, help="Comma-separated alpha values. Default: logspace in [0.1, 10].")
    parser.add_argument("--betas", default=None, help="Comma-separated beta values. Default: logspace in [0.1, 50].")
    parser.add_argument("--num_taus", type=int, default=5)
    parser.add_argument("--num_alphas", type=int, default=5)
    parser.add_argument("--num_betas", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="Device for tensor inference, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--eval_batch_size", type=int, default=0, help="Batch size for evaluation. Use 0 to evaluate each split at once.")
    parser.add_argument("--no_cache_node_scores", action="store_true", help="Recompute Gaussian node scores for every grid combination.")
    parser.add_argument("--cache_node_scores_on_cpu", action="store_true", help="Keep cached node scores on CPU instead of the inference device.")
    parser.add_argument("--save_every", type=int, default=0, help="Write partial tuning results every N grid combinations. Use 0 to save only at the end.")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the hyperparameter grid into this many shards.")
    parser.add_argument("--shard_index", type=int, default=0, help="Zero-based shard index to run.")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "hyperparam_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    taus = parse_list_arg(args.taus, default_log_grid(1.0, 10.0, args.num_taus))
    alphas = parse_list_arg(args.alphas, default_log_grid(0.1, 10.0, args.num_alphas))
    betas = parse_list_arg(args.betas, default_log_grid(0.1, 50.0, args.num_betas))

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])
    tuned_depths = list(range(1, hierarchy.max_depth + 1))

    if args.merge_results:
        merge_paths = [Path(token.strip()) for token in args.merge_results.split(",") if token.strip()]
        if not merge_paths:
            raise ValueError("--merge_results did not include any paths")
        tuning_df = pd.concat((pd.read_csv(path) for path in merge_paths), ignore_index=True)
        tuning_df = tuning_df.drop_duplicates(
            subset=["split", "temperature_vector", "alpha_vector", "beta_vector"],
            keep="last",
        )
        write_tuning_outputs(
            tuning_df,
            output_dir,
            args.target_split,
            args.topk,
            tuned_depths,
            experiment_dir,
            config,
            prohoc=args.prohoc,
        )
        return

    density_payload = load_artifact(experiment_dir / "node_density.pt")
    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, val_feature_meta = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, ood_feature_meta = resolve_feature_tensor(config, experiment_dir, "ood")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    density_payload = move_tensors_to_device(density_payload, device)
    if args.eval_batch_size <= 0:
        val_features = val_features.to(device)
        ood_features = ood_features.to(device)
    val_node_scores = None
    ood_node_scores = None
    if not args.no_cache_node_scores:
        cache_device = torch.device("cpu") if args.cache_node_scores_on_cpu else device
        val_node_scores = precompute_node_scores(
            val_features,
            density_payload,
            config["inference"],
            device,
            args.eval_batch_size,
            cache_device,
        )
        ood_node_scores = precompute_node_scores(
            ood_features,
            density_payload,
            config["inference"],
            device,
            args.eval_batch_size,
            cache_device,
        )

    rows: list[dict[str, Any]] = []

    tau_vectors = list(itertools.product(taus, repeat=len(tuned_depths)))
    alpha_vectors = list(itertools.product(alphas, repeat=len(tuned_depths)))
    beta_vectors = list(itertools.product(betas, repeat=len(tuned_depths)))
    combinations = list(itertools.product(tau_vectors, alpha_vectors, beta_vectors))
    total_combinations = len(combinations)
    combinations = combinations[args.shard_index::args.num_shards]
    print(
        f"Running shard {args.shard_index + 1}/{args.num_shards}: "
        f"{len(combinations)} of {total_combinations} combinations on {device}"
    )
    progress = tqdm(combinations, desc="Hyperparameter tuning", total=len(combinations))

    for combo_idx, (temperature_vector, alpha_vector, beta_vector) in enumerate(progress, start=1):
        progress.set_postfix(
            tau="[" + ",".join(f"{tau:.3g}" for tau in temperature_vector) + "]",
            alpha="[" + ",".join(f"{a:.2g}" for a in alpha_vector) + "]",
            beta="[" + ",".join(f"{b:.2g}" for b in beta_vector) + "]",
        )
        inference_cfg = build_inference_cfg(
            config["inference"],
            list(temperature_vector),
            list(alpha_vector),
            list(beta_vector),
        )
        val_metrics = evaluate_split_for_tuning(
            val_artifact,
            val_features,
            hierarchy,
            density_payload,
            inference_cfg,
            val_feature_meta,
            device,
            args.eval_batch_size,
            node_scores=val_node_scores,
        )
        ood_metrics = evaluate_split_for_tuning(
            ood_artifact,
            ood_features,
            hierarchy,
            density_payload,
            inference_cfg,
            ood_feature_meta,
            device,
            args.eval_batch_size,
            node_scores=ood_node_scores,
        )

        for split_name, metrics in [("val", val_metrics), ("ood", ood_metrics)]:
            row = {
                "experiment_name": config["experiment"]["name"],
                "dataset": dataset_cfg.get("name", Path(dataset_cfg["id_split"]).stem),
                "split": split_name,
                "score_type": inference_cfg.get("score_type"),
                "covariance_type": density_payload.get("covariance_type", density_payload.get("config", {}).get("covariance_type")),
                "temperature_vector": format_temperature_vector(temperature_vector),
                "alpha_vector": format_depth_vector(alpha_vector),
                "beta_vector": format_depth_vector(beta_vector),
                "kappa": float(inference_cfg.get("kappa", 20.0)),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            }
            for depth, tau in zip(tuned_depths, temperature_vector):
                row[f"tau_depth_{depth}"] = float(tau)
            for depth, alpha in zip(tuned_depths, alpha_vector):
                row[f"alpha_depth_{depth}"] = float(alpha)
            for depth, beta in zip(tuned_depths, beta_vector):
                row[f"beta_depth_{depth}"] = float(beta)
            for metric in SCALAR_METRICS:
                value = metrics.get(metric)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                row[metric] = value
            rows.append(row)

        if args.save_every > 0 and combo_idx % args.save_every == 0:
            partial_csv = output_dir / "tuning_results_partial.csv"
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            status_txt = output_dir / "progress_status.txt"
            status_txt.write_text(
                "\n".join(
                    [
                        f"completed_combinations={combo_idx}",
                        f"shard_combinations={len(combinations)}",
                        f"total_combinations={total_combinations}",
                        f"num_shards={args.num_shards}",
                        f"shard_index={args.shard_index}",
                        f"current_temperature={format_temperature_vector(temperature_vector)}",
                        f"current_alpha={format_depth_vector(alpha_vector)}",
                        f"current_beta={format_depth_vector(beta_vector)}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

    tuning_df = pd.DataFrame(rows)
    write_tuning_outputs(
        tuning_df,
        output_dir,
        args.target_split,
        args.topk,
        tuned_depths,
        experiment_dir,
        config,
        prohoc=args.prohoc,
    )


def _discover_paths(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.result"))


if __name__ == "__main__":
    main()
