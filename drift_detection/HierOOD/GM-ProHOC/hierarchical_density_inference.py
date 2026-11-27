from __future__ import annotations

import argparse
from pathlib import Path

import torch

from core.config import load_merged_config, save_config
from core.eval import evaluate_predictions
from core.feature_io import load_artifact, save_artifact
from core.hierarchy_inference import hierarchical_node_probabilities, predict_from_probabilities
from feature_generation.utils.io import resolve_feature_tensor
from libs.hierarchy import Hierarchy
from libs.utils.dataset_util import get_id_classes


def evaluate_split(split_artifact, features, hierarchy, density_payload, inference_cfg, feature_meta, include_debug=True):
    final_probs, debug = hierarchical_node_probabilities(
        features,
        hierarchy,
        density_payload,
        score_type=inference_cfg.get("score_type", "gaussian_loglik"),
        temperature=inference_cfg.get("temperature", 1.0),
        kappa=inference_cfg.get("kappa", 20.0),
        alpha=inference_cfg.get("alpha", 1.0),
        beta=inference_cfg.get("beta", 1.0),
        include_debug=include_debug,
    )
    preds = predict_from_probabilities(final_probs, hierarchy, mode=inference_cfg.get("prediction_mode", "argmax"))
    metrics = evaluate_predictions(preds, split_artifact["node_targets"], hierarchy)
    metrics.update(
        {
            "num_samples": int(features.shape[0]),
            "prediction_mode": inference_cfg.get("prediction_mode", "argmax"),
            "score_type": inference_cfg.get("score_type", "gaussian_loglik"),
            "temperature": inference_cfg.get("temperature", 1.0),
            "kappa": inference_cfg.get("kappa", 20.0),
            "alpha": inference_cfg.get("alpha", 1.0),
            "beta": inference_cfg.get("beta", 1.0),
            "collapsed_ood": inference_cfg.get("collapse_ood_to_parent", True),
            "predictions": preds,
            "probabilities": final_probs.cpu(),
            "debug": debug,
            "feature_source": feature_meta,
        }
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-gen-config")
    args = parser.parse_args()

    config = load_merged_config(args.config, args.feature_gen_config)
    experiment_dir = Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    save_config(config, experiment_dir / "resolved_config.yaml")

    dataset_cfg = config["dataset"]
    id_classes = get_id_classes(dataset_cfg["id_split"])
    hierarchy = Hierarchy(id_classes, dataset_cfg["hierarchy"])

    density_payload = load_artifact(experiment_dir / "node_density.pt")
    val_artifact = load_artifact(experiment_dir / "features_val.pt")
    ood_artifact = load_artifact(experiment_dir / "features_ood.pt")
    val_features, val_feature_meta = resolve_feature_tensor(config, experiment_dir, "val")
    ood_features, ood_feature_meta = resolve_feature_tensor(config, experiment_dir, "ood")

    results = {
        "val": evaluate_split(val_artifact, val_features, hierarchy, density_payload, config["inference"], val_feature_meta),
        "ood": evaluate_split(ood_artifact, ood_features, hierarchy, density_payload, config["inference"], ood_feature_meta),
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
                config["inference"],
                farood_feature_meta,
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
            "density_file": str(experiment_dir / "node_density.pt"),
            "feature_generator": str(experiment_dir / "feature_generator.pt"),
            "farood_features": farood_features,
        },
        "results": results,
    }
    save_artifact(result_payload, experiment_dir / "hinference_density.result")


if __name__ == "__main__":
    main()
