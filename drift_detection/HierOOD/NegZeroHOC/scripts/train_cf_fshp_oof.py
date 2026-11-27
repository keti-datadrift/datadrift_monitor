from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.cf_fshp import (
    LOCKED_MAX_ITER,
    apply_cf_fshp,
    fit_cf_fshp,
    known_favoring_map,
    rejection_features,
)
from negzerohoc.cf_rpep import streaming_file_identity
from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.crossfit_class_holdout import (
    canonical_hash,
    validate_topology_holdout_manifest,
)
from negzerohoc.evaluation import (
    build_hierarchy,
    get_results,
    make_distance_mats,
    mixed_summary,
)
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.hierarchical_support import (
    expected_hierarchy_distance_predictions,
)
from negzerohoc.ood_diagnostics import binary_ood_metrics
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.runtime import configure_reproducibility
from scripts.train_cf_rpep_oof import (
    STAGE as CF_RPEP_STAGE,
    atomic_torch_save,
    nested_inner_assignments,
    ordered_evaluation_subset,
    subset_bundle,
)
from scripts.train_crossfit_class_holdout_lora import (
    build_fold_metric_topology,
)
from scripts.train_paper_negprompt_ablation import json_ready


STAGE = "crossfit_factorized_selective_hierarchical_posterior"
METHOD = "CF-FSHP"
SOURCE_PATHS = {
    "train_cf_fshp_oof": REPO_ROOT / "scripts/train_cf_fshp_oof.py",
    "cf_fshp": REPO_ROOT / "negzerohoc/cf_fshp.py",
    "train_cf_rpep_oof": REPO_ROOT / "scripts/train_cf_rpep_oof.py",
    "cf_rpep": REPO_ROOT / "negzerohoc/cf_rpep.py",
    "multidepth_fusion": REPO_ROOT / "negzerohoc/multidepth_fusion.py",
    "prohoc_score_util": REPO_ROOT / "ProHOC/libs/utils/score_util.py",
    "train_crossfit_class_holdout_lora": (
        REPO_ROOT / "scripts/train_crossfit_class_holdout_lora.py"
    ),
}
FGVC_EXPECTED_AUGMENTED_EDGE_COUNTS = {
    0: 119,
    1: 117,
    2: 114,
    3: 118,
}


def method_development_metadata() -> dict:
    return {
        "crossfit_level": (
            "nested_class_and_image_crossfit_method_development_screen"
        ),
        "strict_episode_disjoint": True,
        "strict_confirmatory_gate": False,
        "may_unlock_official_ood": False,
        "actual_ood_still_excluded_from_this_stage": True,
        "adaptivity_limitation": (
            "Idea V CF-FSHP was designed after inspecting prior methods on "
            "this outer "
            "manifest. A pass is method-development evidence only."
        ),
        "idea": "V",
        "required_before_official_ood": (
            "preregistered_fresh_topology_manifest_or_independent_id_hierarchy"
        ),
        "primary_decoder": "categorical_map",
        "diagnostic_decoder": "expected_hierarchy_distance",
        "parameter_scope": "six_global_shared_scalars",
        "loss": "parameter_free_augmented_tree_brier",
        "inner_fold_count": 3,
    }


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment = cfg.get("experiment", {})
    runtime = cfg.get("runtime", {})
    dataset = cfg.get("dataset", {})
    stage = cfg.get("cf_fshp", {})
    experiment_name = str(
        experiment.get("name", "ideaV-cf-fshp-oof-screen")
    )
    output_root = Path(experiment.get("output_root", "outputs"))
    input_checkpoint = str(stage.get("cf_rpep_checkpoint", ""))
    if not input_checkpoint:
        raise ValueError("cf_fshp.cf_rpep_checkpoint is required")
    forbidden_keys = [
        key for key in stage
        if "actual_ood" in key.lower() or "official_ood" in key.lower()
    ]
    if forbidden_keys:
        raise ValueError(
            f"CF-FSHP config cannot contain OOD paths: {forbidden_keys}"
        )
    max_iter = int(stage.get("optimizer", {}).get("max_iter", 100))
    if max_iter != LOCKED_MAX_ITER:
        raise ValueError("CF-FSHP optimizer.max_iter is locked to 100")
    if str(stage.get("primary_decoder", "map")).lower() != "map":
        raise ValueError("CF-FSHP primary decoder is locked to MAP")
    if str(stage.get("loss", "augmented_tree_brier")).lower() != (
        "augmented_tree_brier"
    ):
        raise ValueError("CF-FSHP loss is locked to augmented_tree_brier")

    def artifact(configured, kind, filename):
        return str(resolve_experiment_artifact(
            configured,
            output_root=output_root,
            experiment_name=experiment_name,
            kind=kind,
            default_filename=filename,
        ))

    return Namespace(
        config=str(path),
        raw_config=cfg,
        experiment_name=experiment_name,
        output_root=str(output_root),
        dataset=dataset.get("name", "fgvc-aircraft"),
        hierarchy=dataset.get(
            "hierarchy", "hierarchies/fgvc-aircraft.json"
        ),
        id_split=dataset.get(
            "id_split", "data/fgvc-aircraft-id-labels.csv"
        ),
        seed=int(runtime.get("seed", 0)),
        deterministic=bool(runtime.get("deterministic", True)),
        cf_rpep_checkpoint=input_checkpoint,
        max_iter=max_iter,
        checkpoint_path=artifact(
            stage.get("checkpoint"),
            "checkpoints",
            f"{experiment_name}.pt",
        ),
        result_path=artifact(
            stage.get("result_path"),
            "results",
            f"{experiment_name}.result",
        ),
        diagnostics_path=artifact(
            stage.get("diagnostics_path"),
            "diagnostics",
            f"{experiment_name}.json",
        ),
    )


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return load_config(parser.parse_args().config)


def capture_file_set(paths: dict[str, Path]) -> dict:
    records = {
        name: streaming_file_identity(path)
        for name, path in sorted(paths.items())
    }
    return {
        "files": records,
        "content_hash": canonical_hash({
            name: {
                "file_size": record["file_size"],
                "sha256": record["sha256"],
            }
            for name, record in records.items()
        }),
    }


def capture_run_provenance(args) -> dict:
    source = capture_file_set(SOURCE_PATHS)
    config_file = streaming_file_identity(args.config)
    resolved_config_hash = canonical_hash(args.raw_config)
    input_artifact = streaming_file_identity(args.cf_rpep_checkpoint)
    return {
        "source": source,
        "config_file": config_file,
        "resolved_config_hash": resolved_config_hash,
        "input_cf_rpep_artifact": input_artifact,
    }


def verify_run_provenance(args, expected: dict) -> None:
    current = capture_run_provenance(args)
    if current != expected:
        raise RuntimeError(
            "CF-FSHP source, resolved config, or input artifact changed "
            "during execution"
        )
    reloaded_hash = canonical_hash(load_yaml_config(args.config))
    if reloaded_hash != expected["resolved_config_hash"]:
        raise RuntimeError("CF-FSHP inherited config changed during execution")


def load_input_artifact(args, full_hierarchy, id_classes):
    before = streaming_file_identity(args.cf_rpep_checkpoint)
    checkpoint = load_idea3_checkpoint(
        args.cf_rpep_checkpoint, map_location="cpu"
    )
    after = streaming_file_identity(args.cf_rpep_checkpoint)
    if before != after:
        raise RuntimeError("CF-RPEP input artifact changed while loading")
    if checkpoint.get("stage") != CF_RPEP_STAGE:
        raise ValueError("CF-FSHP input is not a CF-RPEP checkpoint")
    if checkpoint.get("actual_ood_encoded") is not False:
        raise ValueError("CF-FSHP input must be actual-OOD-free")
    manifest = checkpoint.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("CF-FSHP input is missing the outer manifest")
    validate_topology_holdout_manifest(
        full_hierarchy, id_classes, manifest
    )
    if checkpoint.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("CF-FSHP input manifest hash is inconsistent")
    bundles = checkpoint.get(
        "oof_bundles_for_threshold_free_method_development"
    )
    if not isinstance(bundles, dict) or set(bundles) != {0, 1, 2, 3}:
        raise ValueError("CF-FSHP input must contain four OOF bundles")
    upstream_identities = checkpoint.get("input_checkpoint_identities")
    if not isinstance(upstream_identities, dict) or set(
        int(key) for key in upstream_identities
    ) != {0, 1, 2, 3}:
        raise ValueError("CF-FSHP input checkpoint identities are incomplete")
    for fold, bundle in bundles.items():
        required = {
            "leaf_probabilities",
            "parent_mass",
            "entcomp_unknown",
            "leaf_node_indices",
            "parent_node_indices",
            "node_count",
            "target_node_indices",
            "kinds",
            "target_groups",
            "class_names",
            "original_indices",
            "known_count",
            "pseudo_count",
            "retained_classes",
            "parent_nodes",
        }
        if not required.issubset(bundle):
            raise ValueError(f"CF-FSHP fold {fold} bundle is incomplete")
        count = len(bundle["kinds"])
        if not (
            len(bundle["target_groups"])
            == len(bundle["class_names"])
            == len(bundle["original_indices"])
            == count
        ):
            raise ValueError(f"CF-FSHP fold {fold} episode rows misalign")
        if int(bundle["target_node_indices"].numel()) != count:
            raise ValueError(f"CF-FSHP fold {fold} targets misalign")
    return checkpoint, manifest, bundles


def exact_assignment_audit(bundle, assignments, base_audit) -> dict:
    rows = [
        {
            "episode_position": index,
            "inner_fold": int(assignments[index]),
            "kind": bundle["kinds"][index],
            "class_name": bundle["class_names"][index],
            "target_group": bundle["target_groups"][index],
            "original_index": int(bundle["original_indices"][index]),
            "target_node_index": int(
                bundle["target_node_indices"][index]
            ),
        }
        for index in range(len(bundle["kinds"]))
    ]
    row_hash = canonical_hash({"rows": rows})
    if row_hash != canonical_hash({"rows": list(rows)}):
        raise RuntimeError("CF-FSHP assignment hash is unstable")
    unique_original_image_count = len({
        row["original_index"] for row in rows
    })
    if unique_original_image_count != len(rows):
        raise RuntimeError(
            "CF-FSHP OOF bundle repeats an original query image"
        )
    return {
        **base_audit,
        "assignment_scope": "every_episode_exactly_once",
        "exact_episode_assignments": rows,
        "exact_episode_assignments_hash": row_hash,
        "episode_count": len(rows),
        "unique_original_image_count": unique_original_image_count,
    }


def evaluate_inner_bundle(
    full_hierarchy,
    fold_hierarchy,
    bundle,
    fit,
    distance_matrix,
    dists_mats,
) -> dict:
    terminal, unknown_mass, _ = apply_cf_fshp(
        bundle,
        fold_hierarchy,
        fit,
        global_node_names=full_hierarchy.id_node_list,
    )
    known_count = int(bundle["known_count"])
    known_terminal = terminal[:known_count]
    pseudo_terminal = terminal[known_count:]
    known_targets = bundle["target_node_indices"][:known_count]
    pseudo_targets = bundle["target_node_indices"][known_count:]
    classifier_predictions = bundle["leaf_node_indices"].index_select(
        0, bundle["leaf_probabilities"][:known_count].argmax(dim=1)
    )
    known_predictions = known_favoring_map(
        known_terminal,
        bundle["leaf_node_indices"],
        bundle["parent_node_indices"],
    )
    pseudo_predictions = known_favoring_map(
        pseudo_terminal,
        bundle["leaf_node_indices"],
        bundle["parent_node_indices"],
    )
    expected_known_predictions = (
        expected_hierarchy_distance_predictions(
            known_terminal, distance_matrix
        )
    )
    expected_pseudo_predictions = (
        expected_hierarchy_distance_predictions(
            pseudo_terminal, distance_matrix
        )
    )
    classifier_metrics = get_results(
        classifier_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    return {
        "classifier_predictions": classifier_predictions,
        "known_predictions": known_predictions,
        "pseudo_predictions": pseudo_predictions,
        "expected_known_predictions": expected_known_predictions,
        "expected_pseudo_predictions": expected_pseudo_predictions,
        "known_targets": known_targets,
        "pseudo_targets": pseudo_targets,
        "known_unknown_mass": unknown_mass[:known_count],
        "pseudo_unknown_mass": unknown_mass[known_count:],
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": float(
            (terminal.sum(dim=1) - 1.0).abs().max()
        ),
    }


def evaluate_nested_fold(
    full_hierarchy,
    fold_hierarchy,
    bundle,
    distance_matrix,
    dists_mats,
    *,
    max_iter,
) -> dict:
    assignments, base_audit = nested_inner_assignments(
        bundle, full_hierarchy
    )
    assignment_audit = exact_assignment_audit(
        bundle, assignments, base_audit
    )
    inner_results = []
    inner_fits = []
    evaluated_positions = []
    for inner_fold in range(3):
        evaluation_mask = assignments == inner_fold
        calibration_mask = ~evaluation_mask
        calibration = subset_bundle(bundle, calibration_mask)
        fit = fit_cf_fshp(
            calibration,
            fold_hierarchy,
            global_node_names=full_hierarchy.id_node_list,
            max_iter=max_iter,
        )
        expected_edge_count = (
            len(fold_hierarchy.id_node_list) - 1
            + int(bundle["parent_node_indices"].numel())
        )
        if fit["edge_count"] != expected_edge_count:
            raise RuntimeError(
                "CF-FSHP fit did not use the exact fold-pruned "
                "augmented-tree edges"
            )
        if fit["feature_normalization"]["sample_count"] != int(
            calibration_mask.sum()
        ):
            raise RuntimeError(
                "CF-FSHP feature normalization used a non-calibration row"
            )
        evaluation_bundle = ordered_evaluation_subset(
            bundle, evaluation_mask
        )
        evaluation = evaluate_inner_bundle(
            full_hierarchy,
            fold_hierarchy,
            evaluation_bundle,
            fit,
            distance_matrix,
            dists_mats,
        )
        calibration_positions = torch.nonzero(
            calibration_mask, as_tuple=False
        ).flatten().tolist()
        evaluation_positions = torch.nonzero(
            evaluation_mask, as_tuple=False
        ).flatten().tolist()
        if set(calibration_positions) & set(evaluation_positions):
            raise RuntimeError("CF-FSHP inner calibration/evaluation overlap")
        fit["inner_fold"] = inner_fold
        fit["calibration_episode_positions"] = calibration_positions
        fit["evaluation_episode_positions"] = evaluation_positions
        fit["calibration_episode_positions_hash"] = canonical_hash({
            "positions": calibration_positions
        })
        fit["evaluation_episode_positions_hash"] = canonical_hash({
            "positions": evaluation_positions
        })
        fit["normalization_fit_on_calibration_only"] = True
        evaluation["fit"] = fit
        inner_results.append(evaluation)
        inner_fits.append(fit)
        evaluated_positions.extend(evaluation_positions)
    if sorted(evaluated_positions) != list(range(len(bundle["kinds"]))):
        raise RuntimeError(
            "CF-FSHP did not evaluate every episode exactly once"
        )

    known_targets = torch.cat([
        value["known_targets"] for value in inner_results
    ])
    pseudo_targets = torch.cat([
        value["pseudo_targets"] for value in inner_results
    ])
    classifier_predictions = torch.cat([
        value["classifier_predictions"] for value in inner_results
    ])
    known_predictions = torch.cat([
        value["known_predictions"] for value in inner_results
    ])
    pseudo_predictions = torch.cat([
        value["pseudo_predictions"] for value in inner_results
    ])
    classifier_metrics = get_results(
        classifier_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        known_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        full_hierarchy,
        dists_mats=dists_mats,
    )
    return {
        "classifier_predictions": classifier_predictions,
        "known_predictions": known_predictions,
        "pseudo_predictions": pseudo_predictions,
        "expected_known_predictions": torch.cat([
            value["expected_known_predictions"]
            for value in inner_results
        ]),
        "expected_pseudo_predictions": torch.cat([
            value["expected_pseudo_predictions"]
            for value in inner_results
        ]),
        "known_targets": known_targets,
        "pseudo_targets": pseudo_targets,
        "known_unknown_mass": torch.cat([
            value["known_unknown_mass"] for value in inner_results
        ]),
        "pseudo_unknown_mass": torch.cat([
            value["pseudo_unknown_mass"] for value in inner_results
        ]),
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": max(
            value["normalization_error"] for value in inner_results
        ),
        "inner_fits": inner_fits,
        "assignment_audit": assignment_audit,
        "target_disjoint_audit": {
            "encoder_excludes_all_pseudo_classes": True,
            "fit_excludes_evaluated_pseudo_classes": True,
            "fit_excludes_evaluated_known_images": True,
            "normalization_excludes_all_evaluation_episodes": True,
            "every_episode_evaluated_exactly_once": True,
        },
    }


def aggregate(evaluations, hierarchy, dists_mats, *, diagnostic=False):
    folds = sorted(evaluations)
    known_targets = torch.cat([
        evaluations[fold]["known_targets"] for fold in folds
    ])
    pseudo_targets = torch.cat([
        evaluations[fold]["pseudo_targets"] for fold in folds
    ])
    known_key = (
        "expected_known_predictions" if diagnostic else "known_predictions"
    )
    pseudo_key = (
        "expected_pseudo_predictions"
        if diagnostic else "pseudo_predictions"
    )
    classifier_metrics = get_results(
        torch.cat([
            evaluations[fold]["classifier_predictions"] for fold in folds
        ]),
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        torch.cat([evaluations[fold][known_key] for fold in folds]),
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        torch.cat([evaluations[fold][pseudo_key] for fold in folds]),
        pseudo_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    unknown_binary = binary_ood_metrics(
        torch.cat([
            evaluations[fold]["known_unknown_mass"] for fold in folds
        ]).numpy(),
        torch.cat([
            evaluations[fold]["pseudo_unknown_mass"] for fold in folds
        ]).numpy(),
    )
    return {
        "classifier_known": classifier_metrics,
        "posterior_known": known_metrics,
        "pseudo_mapped_parent": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "unknown_mass_binary": unknown_binary,
        "normalization_error": max(
            evaluations[fold]["normalization_error"] for fold in folds
        ),
    }


def locked_gate(primary, evaluations):
    per_fold_degradation = {
        str(fold): (
            float(evaluations[fold]["classifier_metrics"]["balanced_acc"])
            - float(evaluations[fold]["known_metrics"]["balanced_acc"])
        )
        for fold in sorted(evaluations)
    }
    thresholds = {
        "classifier_known_balanced_acc_min": 0.800,
        "posterior_known_balanced_acc_min": 0.780,
        "mean_known_degradation_max": 0.020,
        "per_fold_known_degradation_max": 0.030,
        "pseudo_mapped_parent_balanced_acc_min": 0.227,
        "mixed_balanced_acc_strict_min": 0.503,
        "mixed_balanced_hdist_max": 0.810,
        "unknown_mass_auroc_min": 0.750,
        "normalization_error_max": 1e-5,
    }
    values = {
        "classifier_known_balanced_acc": float(
            primary["classifier_known"]["balanced_acc"]
        ),
        "posterior_known_balanced_acc": float(
            primary["posterior_known"]["balanced_acc"]
        ),
        "mean_known_degradation": (
            sum(per_fold_degradation.values())
            / float(len(per_fold_degradation))
        ),
        "max_per_fold_known_degradation": max(
            per_fold_degradation.values()
        ),
        "pseudo_mapped_parent_balanced_acc": float(
            primary["pseudo_mapped_parent"]["balanced_acc"]
        ),
        "mixed_balanced_acc": float(
            primary["mixed"]["mixed_balanced_acc"]
        ),
        "mixed_balanced_hdist": float(
            primary["mixed"]["mixed_balanced_hdist"]
        ),
        "unknown_mass_auroc": float(
            primary["unknown_mass_binary"]["auroc"]
        ),
        "normalization_error": float(primary["normalization_error"]),
    }
    checks = {
        "classifier_known_balanced_acc": (
            values["classifier_known_balanced_acc"]
            >= thresholds["classifier_known_balanced_acc_min"]
        ),
        "posterior_known_balanced_acc": (
            values["posterior_known_balanced_acc"]
            >= thresholds["posterior_known_balanced_acc_min"]
        ),
        "mean_known_degradation": (
            values["mean_known_degradation"]
            <= thresholds["mean_known_degradation_max"]
        ),
        "per_fold_known_degradation": (
            values["max_per_fold_known_degradation"]
            <= thresholds["per_fold_known_degradation_max"]
        ),
        "pseudo_mapped_parent_balanced_acc": (
            values["pseudo_mapped_parent_balanced_acc"]
            >= thresholds["pseudo_mapped_parent_balanced_acc_min"]
        ),
        "mixed_balanced_acc": (
            values["mixed_balanced_acc"]
            > thresholds["mixed_balanced_acc_strict_min"]
        ),
        "mixed_balanced_hdist": (
            values["mixed_balanced_hdist"]
            <= thresholds["mixed_balanced_hdist_max"]
        ),
        "unknown_mass_auroc": (
            values["unknown_mass_auroc"]
            >= thresholds["unknown_mass_auroc_min"]
        ),
        "normalization": (
            values["normalization_error"]
            <= thresholds["normalization_error_max"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "values": values,
        "checks": checks,
        "per_fold_known_degradation": per_fold_degradation,
        "gate_scope": "method_development_screen_only",
        "may_unlock_official_ood": False,
    }


def fixed_feature_diagnostics(bundles) -> dict:
    known = [[], []]
    pseudo = [[], []]
    for fold in sorted(bundles):
        bundle = bundles[fold]
        values = rejection_features(
            bundle["leaf_probabilities"],
            bundle["parent_mass"],
            bundle["entcomp_unknown"],
        ).cpu()
        known_count = int(bundle["known_count"])
        for feature in range(2):
            known[feature].append(values[:known_count, feature])
            pseudo[feature].append(values[known_count:, feature])
    names = ("route_mass_mean_entcomp_odds", "leaf_entropy")
    result = {}
    for feature, name in enumerate(names):
        known_values = torch.cat(known[feature]).numpy()
        pseudo_values = torch.cat(pseudo[feature]).numpy()
        result[name] = {
            "known_mean": float(known_values.mean()),
            "pseudo_mean": float(pseudo_values.mean()),
            "known_count": int(known_values.size),
            "pseudo_count": int(pseudo_values.size),
            "binary_metrics": binary_ood_metrics(
                known_values, pseudo_values
            ),
            "used_for_specification_change": False,
            "used_for_gate": False,
        }
    return result


def main():
    args = parse_args()
    configure_reproducibility(
        args.seed, deterministic=args.deterministic
    )
    provenance = capture_run_provenance(args)
    hierarchy, id_classes = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    source_checkpoint, manifest, bundles = load_input_artifact(
        args, hierarchy, id_classes
    )
    dists_mats = make_distance_mats(hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    evaluations = {}
    fold_topology_audits = {}
    for fold in range(4):
        configure_reproducibility(
            args.seed + 1_000_003 * fold,
            deterministic=args.deterministic,
        )
        fold_hierarchy, _, topology_provenance = (
            build_fold_metric_topology(
                args, hierarchy, bundles[fold]["retained_classes"]
            )
        )
        if set(bundles[fold]["retained_classes"]) - set(
            fold_hierarchy.id_node_list
        ):
            raise RuntimeError(
                f"CF-FSHP fold {fold} retained classes are absent"
            )
        if set(bundles[fold]["parent_nodes"]) != set(
            node for node in fold_hierarchy.parent2children
            if node != "root"
        ):
            raise RuntimeError(
                f"CF-FSHP fold {fold} parent terminals/topology differ"
            )
        augmented_edge_count = (
            len(fold_hierarchy.id_node_list) - 1
            + len(bundles[fold]["parent_nodes"])
        )
        if (
            args.dataset == "fgvc-aircraft"
            and augmented_edge_count
            != FGVC_EXPECTED_AUGMENTED_EDGE_COUNTS[fold]
        ):
            raise RuntimeError(
                f"CF-FSHP fold {fold} augmented edge count changed: "
                f"{augmented_edge_count}"
            )
        fold_topology_audits[str(fold)] = {
            "fold_node_count": len(fold_hierarchy.id_node_list),
            "unknown_terminal_count": len(
                bundles[fold]["parent_nodes"]
            ),
            "augmented_edge_count": augmented_edge_count,
            "expected_augmented_edge_count": (
                FGVC_EXPECTED_AUGMENTED_EDGE_COUNTS.get(fold)
            ),
            "topology_provenance": topology_provenance,
        }
        evaluations[fold] = evaluate_nested_fold(
            hierarchy,
            fold_hierarchy,
            bundles[fold],
            distance_matrix,
            dists_mats,
            max_iter=args.max_iter,
        )
    primary = aggregate(evaluations, hierarchy, dists_mats)
    expected_diagnostic = aggregate(
        evaluations, hierarchy, dists_mats, diagnostic=True
    )
    feature_diagnostics = fixed_feature_diagnostics(bundles)
    gate = locked_gate(primary, evaluations)
    per_fold = {
        str(fold): {
            "classifier_known": evaluations[fold]["classifier_metrics"],
            "posterior_known": evaluations[fold]["known_metrics"],
            "pseudo_mapped_parent": evaluations[fold]["pseudo_metrics"],
            "mixed": evaluations[fold]["mixed"],
            "known_degradation": gate[
                "per_fold_known_degradation"
            ][str(fold)],
            "normalization_error": evaluations[fold][
                "normalization_error"
            ],
            "inner_fits": evaluations[fold]["inner_fits"],
            "assignment_audit": evaluations[fold]["assignment_audit"],
            "target_disjoint_audit": evaluations[fold][
                "target_disjoint_audit"
            ],
        }
        for fold in range(4)
    }
    common = {
        "method": METHOD,
        "stage": STAGE,
        **method_development_metadata(),
        "manifest_hash": manifest["manifest_hash"],
        "source_and_config_provenance": provenance,
        "input_cf_rpep_checkpoint": args.cf_rpep_checkpoint,
        "input_cf_rpep_checkpoint_identity": provenance[
            "input_cf_rpep_artifact"
        ],
        "upstream_fold_checkpoint_identities": source_checkpoint[
            "input_checkpoint_identities"
        ],
        "upstream_split_hashes": source_checkpoint["input_split_hashes"],
        "fold_pruned_augmented_topology_audits": fold_topology_audits,
        "strict_input_validation_passed": True,
        "actual_ood_loader_loaded": False,
        "actual_ood_dataset_loaded": False,
        "actual_ood_encoded": False,
        "actual_ood_evaluation_implemented_in_this_stage": False,
    }
    result = {
        "status": (
            "method_development_screen_go"
            if gate["passed"]
            else "method_development_screen_no_go"
        ),
        **common,
        "classifier_known": primary["classifier_known"],
        "posterior_known": primary["posterior_known"],
        "pseudo_mapped_parent": primary["pseudo_mapped_parent"],
        "mixed": primary["mixed"],
        "unknown_mass_binary": primary["unknown_mass_binary"],
        "normalization_error": primary["normalization_error"],
        "per_fold": per_fold,
        "gate": gate,
        "expected_hierarchy_distance_diagnostic_not_used_for_gate": (
            expected_diagnostic
        ),
        "fixed_feature_diagnostics_not_used_for_gate": (
            feature_diagnostics
        ),
    }
    checkpoint_payload = {
        **common,
        "manifest": manifest,
        "nested_inner_fits": {
            str(fold): evaluations[fold]["inner_fits"]
            for fold in range(4)
        },
        "exact_episode_assignment_audits": {
            str(fold): evaluations[fold]["assignment_audit"]
            for fold in range(4)
        },
        "gate": gate,
        "expected_hierarchy_distance_diagnostic_not_used_for_gate": (
            expected_diagnostic
        ),
        "fixed_feature_diagnostics_not_used_for_gate": (
            feature_diagnostics
        ),
        "transfer_fit": None,
    }
    verify_run_provenance(args, provenance)
    checkpoint_path = Path(args.checkpoint_path)
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    atomic_torch_save(checkpoint_payload, checkpoint_path)
    atomic_torch_save(result, result_path)
    ensure_dir(diagnostics_path.parent)
    save_json(diagnostics_path, json_ready(result))
    values = gate["values"]
    print(
        f"CF-FSHP development screen="
        f"{'GO' if gate['passed'] else 'NO-GO'}: "
        f"classifier={values['classifier_known_balanced_acc']:.6f}, "
        f"posterior={values['posterior_known_balanced_acc']:.6f}, "
        f"pseudo={values['pseudo_mapped_parent_balanced_acc']:.6f}, "
        f"mix={values['mixed_balanced_acc']:.6f}, "
        f"BMHD={values['mixed_balanced_hdist']:.6f}, "
        f"AUROC={values['unknown_mass_auroc']:.6f}"
    )
    print(
        "Official OOD loader/dataset was not loaded; this screen cannot "
        f"unlock it. Saved: {result_path}"
    )


if __name__ == "__main__":
    main()
