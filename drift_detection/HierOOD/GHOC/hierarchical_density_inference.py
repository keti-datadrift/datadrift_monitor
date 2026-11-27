from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from core.config import load_merged_config, save_config
from core.eval import evaluate_predictions
from core.feature_io import load_artifact, save_artifact
from core.hierarchy_inference import hierarchical_node_probabilities, predict_from_probabilities
from libs.utils.hierarchy_utils import get_hdist_matrix
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


def evaluate_split(
    split_artifact,
    features,
    hierarchy,
    density_payload,
    inference_cfg,
    feature_meta,
    device: torch.device,
    eval_batch_size: int,
    include_debug: bool = False,
    save_probabilities: bool = False,
    hdist_mat: torch.Tensor | None = None,
):
    preds = []
    probabilities = []
    debug_batches = []
    n_samples = int(features.shape[0])
    batch_size = n_samples if eval_batch_size <= 0 else int(eval_batch_size)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            batch_features = features[start : start + batch_size].to(device)
            final_probs, debug = hierarchical_node_probabilities(
                batch_features,
                hierarchy,
                density_payload,
                score_type=inference_cfg.get("score_type", "gaussian_loglik"),
                temperature=inference_cfg.get("temperature", 1.0),
                kappa=inference_cfg.get("kappa", 20.0),
                include_debug=include_debug,
                cgm_cfg=inference_cfg.get("cgm", {}),
                ood_scale=inference_cfg.get("ood_scale", 1.0),
            )
            batch_preds = predict_from_probabilities(
                final_probs,
                hierarchy,
                mode=inference_cfg.get("prediction_mode", "argmax"),
                hdist_mat=hdist_mat,
            )
            preds.append(batch_preds.cpu())
            if save_probabilities:
                probabilities.append(final_probs.cpu())
            if include_debug:
                debug_batches.append(debug)

    preds = torch.cat(preds, dim=0)
    metrics = evaluate_predictions(preds, split_artifact["node_targets"], hierarchy)
    metrics.update(
        {
            "num_samples": n_samples,
            "prediction_mode": inference_cfg.get("prediction_mode", "argmax"),
            "score_type": inference_cfg.get("score_type", "gaussian_loglik"),
            "temperature": inference_cfg.get("temperature", 1.0),
            "kappa": inference_cfg.get("kappa", 20.0),
            "ood_scale": inference_cfg.get("ood_scale", 1.0),
            "cgm": inference_cfg.get("cgm", {"enabled": False}),
            "collapsed_ood": inference_cfg.get("collapse_ood_to_parent", True),
            "predictions": preds,
            "feature_source": feature_meta,
        }
    )
    if save_probabilities:
        metrics["probabilities"] = torch.cat(probabilities, dim=0)
    if include_debug:
        metrics["debug_batches"] = debug_batches
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    parser.add_argument("--density-file", help="Density artifact path. Default: <experiment_dir>/node_density.pt")
    parser.add_argument("--output", help="Result artifact path. Default: <experiment_dir>/hinference_density.result")
    parser.add_argument("--device", default=None, help="Inference device. Default: config experiment.device, falling back to cpu.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for density inference. Default: dataloader.eval_batch_size.")
    parser.add_argument("--include-debug", action="store_true", help="Store debug tensors. This can be very memory intensive.")
    parser.add_argument("--save-probabilities", action="store_true", help="Store final probability matrix in the result artifact.")
    args = parser.parse_args()

    config = load_merged_config(args.config, args.feature_gen_config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config.yaml")

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    density_path = Path(args.density_file) if args.density_file else experiment_dir / "node_density.pt"
    density_payload = load_artifact(density_path)
    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, val_feature_meta = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, ood_feature_meta = resolve_feature_tensor(config, experiment_dir, "ood")
    device_name = args.device or config.get("experiment", {}).get("device", "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    eval_batch_size = args.eval_batch_size
    if eval_batch_size is None:
        eval_batch_size = int(config.get("dataloader", {}).get("eval_batch_size", 0))
    density_payload = move_tensors_to_device(density_payload, device)
    prediction_mode = config["inference"].get("prediction_mode", "argmax")
    hdist_mat = None
    if prediction_mode == "min_hdist":
        hdist_mat = get_hdist_matrix(hierarchy, range(len(hierarchy.id_node_list)))

    results = {
        "val": evaluate_split(
            val_artifact,
            val_features,
            hierarchy,
            density_payload,
            {**config["inference"], "cgm": config.get("cgm", {"enabled": False})},
            val_feature_meta,
            device,
            eval_batch_size,
            include_debug=args.include_debug,
            save_probabilities=args.save_probabilities,
            hdist_mat=hdist_mat,
        ),
        "ood": evaluate_split(
            ood_artifact,
            ood_features,
            hierarchy,
            density_payload,
            {**config["inference"], "cgm": config.get("cgm", {"enabled": False})},
            ood_feature_meta,
            device,
            eval_batch_size,
            include_debug=args.include_debug,
            save_probabilities=args.save_probabilities,
            hdist_mat=hdist_mat,
        ),
    }

    farood_features = {}
    for farood_name in dataset_cfg.get("farood_sets", []) or []:
        artifact_path = experiment_dir / f"features_farood_{farood_name.replace('/', '-')}.pt"
        if artifact_path.exists():
            farood_artifact = load_artifact(artifact_path)
            farood_tensor, farood_feature_meta = resolve_feature_tensor(
                config,
                experiment_dir,
                f"farood_{farood_name.replace('/', '-')}",
            )
            results[farood_name] = evaluate_split(
                farood_artifact,
                farood_tensor,
                hierarchy,
                density_payload,
                {**config["inference"], "cgm": config.get("cgm", {"enabled": False})},
                farood_feature_meta,
                device,
                eval_batch_size,
                include_debug=args.include_debug,
                save_probabilities=args.save_probabilities,
                hdist_mat=hdist_mat,
            )
            farood_features[farood_name] = str(artifact_path)

    result_payload = {
        "experiment_name": config["experiment"]["name"],
        "dataset": dataset_cfg.get("name", Path(dataset_cfg["id_split"]).stem),
        "config": config,
        "artifacts": {
            "backbone_checkpoint": str(experiment_dir / "checkpoint_backbone.pt"),
            "train_features": str(experiment_dir / "features_train.pt"),
            "val_features": str(experiment_dir / "features_val.pt"),
            "ood_features": str(experiment_dir / "features_ood.pt"),
            "density_file": str(density_path),
            "feature_generator": str(experiment_dir / "feature_generator.pt"),
            "farood_features": farood_features,
        },
        "results": results,
    }
    output_path = Path(args.output) if args.output else experiment_dir / "hinference_density.result"
    save_artifact(result_payload, output_path)


if __name__ == "__main__":
    main()
