from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.cf_rpep import (
    fit_shared_hazard_scalars,
    fit_shared_route_scalars,
    hierarchical_hazard_terminal,
    leaf_coherent_entcomp_unknown,
)
from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
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
from scripts.train_cf_rpep_oof import (
    STAGE as CF_RPEP_STAGE,
    evaluate_target_fold,
    nested_inner_assignments,
    ordered_evaluation_subset,
    subset_bundle,
)
from scripts.train_paper_negprompt_ablation import json_ready


STAGE = "crossfit_hierarchical_hazard_oof"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment = cfg.get("experiment", {})
    stage = cfg.get("hierarchical_hazard", {})
    dataset = cfg.get("dataset", {})
    experiment_name = str(
        experiment.get("name", "crossfit-hierarchical-hazard")
    )
    output_root = Path(experiment.get("output_root", "outputs"))

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
        hierarchy=dataset.get(
            "hierarchy", "hierarchies/fgvc-aircraft.json"
        ),
        id_split=dataset.get(
            "id_split", "data/fgvc-aircraft-id-labels.csv"
        ),
        input_checkpoint=str(stage.get("input_checkpoint", "")),
        scalar_max_iter=max(1, int(stage.get("scalar_max_iter", 100))),
        primary_decoder=str(
            stage.get("primary_decoder", "expected_hdist")
        ),
        primary_row=str(
            stage.get(
                "primary_row",
                "coherent_raw_hazard/expected_hdist",
            )
        ),
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


def atomic_torch_save(payload, path: Path):
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as output:
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_hazard_bundle(
    hierarchy,
    bundle,
    scalar_fit,
    distance_matrix,
    dists_mats,
    *,
    decoder,
):
    terminal = hierarchical_hazard_terminal(
        bundle["leaf_probabilities"],
        bundle["entcomp_unknown"],
        hierarchy,
        leaf_nodes=bundle["retained_classes"],
        parent_nodes=bundle["parent_nodes"],
        leaf_node_indices=bundle["leaf_node_indices"],
        parent_node_indices=bundle["parent_node_indices"],
        node_count=int(bundle["node_count"]),
        a=scalar_fit["a"],
        b=scalar_fit["b"],
    ).float()
    known_count = int(bundle["known_count"])
    known_terminal = terminal[:known_count]
    pseudo_terminal = terminal[known_count:]
    known_targets = bundle["target_node_indices"][:known_count]
    pseudo_targets = bundle["target_node_indices"][known_count:]
    classifier_predictions = bundle["leaf_node_indices"].index_select(
        0,
        bundle["leaf_probabilities"][:known_count].argmax(dim=1),
    )
    if decoder == "expected_hdist":
        known_predictions = expected_hierarchy_distance_predictions(
            known_terminal, distance_matrix
        )
        pseudo_predictions = expected_hierarchy_distance_predictions(
            pseudo_terminal, distance_matrix
        )
    elif decoder == "map":
        known_predictions = known_terminal.argmax(dim=1)
        pseudo_predictions = pseudo_terminal.argmax(dim=1)
    else:
        raise ValueError(f"Unsupported HHP decoder: {decoder!r}")
    internal_indices = torch.tensor([
        index
        for index, node in enumerate(hierarchy.id_node_list)
        if node != "root" and node in hierarchy.parent2children
    ])
    classifier_metrics = get_results(
        classifier_predictions,
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    return {
        "known_targets": known_targets,
        "pseudo_targets": pseudo_targets,
        "classifier_predictions": classifier_predictions,
        "known_predictions": known_predictions,
        "pseudo_predictions": pseudo_predictions,
        "known_unknown_mass": known_terminal.index_select(
            1, internal_indices
        ).sum(dim=1),
        "pseudo_unknown_mass": pseudo_terminal.index_select(
            1, internal_indices
        ).sum(dim=1),
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": float(
            (terminal.sum(dim=1) - 1.0).abs().max()
        ),
        "decoder": decoder,
    }


def combine_evaluations(evaluations, hierarchy, dists_mats):
    known_targets = torch.cat([
        value["known_targets"] for value in evaluations
    ])
    pseudo_targets = torch.cat([
        value["pseudo_targets"] for value in evaluations
    ])
    classifier_predictions = torch.cat([
        value["classifier_predictions"] for value in evaluations
    ])
    known_predictions = torch.cat([
        value["known_predictions"] for value in evaluations
    ])
    pseudo_predictions = torch.cat([
        value["pseudo_predictions"] for value in evaluations
    ])
    classifier_metrics = get_results(
        classifier_predictions,
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    known_metrics = get_results(
        known_predictions,
        known_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    pseudo_metrics = get_results(
        pseudo_predictions,
        pseudo_targets,
        hierarchy,
        dists_mats=dists_mats,
    )
    return {
        "known_targets": known_targets,
        "pseudo_targets": pseudo_targets,
        "classifier_predictions": classifier_predictions,
        "known_predictions": known_predictions,
        "pseudo_predictions": pseudo_predictions,
        "known_unknown_mass": torch.cat([
            value["known_unknown_mass"] for value in evaluations
        ]),
        "pseudo_unknown_mass": torch.cat([
            value["pseudo_unknown_mass"] for value in evaluations
        ]),
        "classifier_metrics": classifier_metrics,
        "known_metrics": known_metrics,
        "pseudo_metrics": pseudo_metrics,
        "mixed": mixed_summary(known_metrics, pseudo_metrics),
        "normalization_error": max(
            value["normalization_error"] for value in evaluations
        ),
    }


def nested_hazard_fold(
    hierarchy,
    bundle,
    distance_matrix,
    dists_mats,
    *,
    decoder,
    max_iter,
):
    assignments, assignment_audit = nested_inner_assignments(
        bundle, hierarchy
    )
    evaluations = []
    fits = []
    for inner_fold in range(3):
        evaluation_mask = assignments == inner_fold
        scalar_fit = fit_shared_hazard_scalars(
            [subset_bundle(bundle, ~evaluation_mask)],
            hierarchy,
            max_iter=max_iter,
        )
        evaluation = evaluate_hazard_bundle(
            hierarchy,
            ordered_evaluation_subset(bundle, evaluation_mask),
            scalar_fit,
            distance_matrix,
            dists_mats,
            decoder=decoder,
        )
        evaluations.append(evaluation)
        fits.append(scalar_fit)
    combined = combine_evaluations(evaluations, hierarchy, dists_mats)
    combined.update({
        "inner_scalar_fits": fits,
        "assignment_audit": assignment_audit,
        "target_disjoint": True,
        "decoder": decoder,
    })
    return combined


def nested_route_fold(
    hierarchy,
    bundle,
    distance_matrix,
    dists_mats,
    *,
    decoder,
    max_iter,
):
    assignments, assignment_audit = nested_inner_assignments(
        bundle, hierarchy
    )
    evaluations = []
    fits = []
    for inner_fold in range(3):
        evaluation_mask = assignments == inner_fold
        scalar_fit = fit_shared_route_scalars(
            [subset_bundle(bundle, ~evaluation_mask)],
            max_iter=max_iter,
        )
        evaluation = evaluate_target_fold(
            hierarchy,
            ordered_evaluation_subset(bundle, evaluation_mask),
            scalar_fit,
            distance_matrix,
            dists_mats,
            decoder=decoder,
        )
        evaluations.append(evaluation)
        fits.append(scalar_fit)
    combined = combine_evaluations(evaluations, hierarchy, dists_mats)
    combined.update({
        "inner_scalar_fits": fits,
        "assignment_audit": assignment_audit,
        "target_disjoint": True,
        "decoder": decoder,
    })
    return combined


def summarize_outer(evaluations, hierarchy, dists_mats):
    combined = combine_evaluations(
        [evaluations[fold] for fold in sorted(evaluations)],
        hierarchy,
        dists_mats,
    )
    unknown_binary = binary_ood_metrics(
        combined["known_unknown_mass"].numpy(),
        combined["pseudo_unknown_mass"].numpy(),
    )
    return {
        "classifier_known": combined["classifier_metrics"],
        "posterior_known": combined["known_metrics"],
        "pseudo_mapped_parent": combined["pseudo_metrics"],
        "mixed": combined["mixed"],
        "unknown_mass_binary": unknown_binary,
        "normalization_error": combined["normalization_error"],
        "per_outer_fold": {
            str(fold): {
                "classifier_known": evaluations[fold][
                    "classifier_metrics"
                ],
                "posterior_known": evaluations[fold]["known_metrics"],
                "pseudo_mapped_parent": evaluations[fold][
                    "pseudo_metrics"
                ],
                "mixed": evaluations[fold]["mixed"],
                "inner_scalar_fits": evaluations[fold].get(
                    "inner_scalar_fits"
                ),
                "assignment_audit": evaluations[fold].get(
                    "assignment_audit"
                ),
            }
            for fold in sorted(evaluations)
        },
    }


def locked_gate(summary):
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
    degradations = {
        fold: (
            float(value["classifier_known"]["balanced_acc"])
            - float(value["posterior_known"]["balanced_acc"])
        )
        for fold, value in summary["per_outer_fold"].items()
    }
    values = {
        "classifier_known_balanced_acc": float(
            summary["classifier_known"]["balanced_acc"]
        ),
        "posterior_known_balanced_acc": float(
            summary["posterior_known"]["balanced_acc"]
        ),
        "mean_known_degradation": (
            sum(degradations.values()) / len(degradations)
        ),
        "max_per_fold_known_degradation": max(degradations.values()),
        "pseudo_mapped_parent_balanced_acc": float(
            summary["pseudo_mapped_parent"]["balanced_acc"]
        ),
        "mixed_balanced_acc": float(
            summary["mixed"]["mixed_balanced_acc"]
        ),
        "mixed_balanced_hdist": float(
            summary["mixed"]["mixed_balanced_hdist"]
        ),
        "unknown_mass_auroc": float(
            summary["unknown_mass_binary"]["auroc"]
        ),
        "normalization_error": float(summary["normalization_error"]),
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
        "per_fold_known_degradation": degradations,
    }


def main():
    args = parse_args()
    if not args.input_checkpoint:
        raise ValueError("Missing hierarchical_hazard.input_checkpoint")
    if args.primary_decoder not in {"expected_hdist", "map"}:
        raise ValueError("HHP primary decoder must be expected_hdist or map")
    source = load_idea3_checkpoint(
        args.input_checkpoint, map_location="cpu"
    )
    if source.get("stage") != CF_RPEP_STAGE:
        raise ValueError("HHP input is not a CF-RPEP OOF checkpoint")
    if source.get("actual_ood_encoded") is not False:
        raise ValueError("HHP input must be actual-OOD-free")
    bundles = source.get(
        "oof_bundles_for_threshold_free_method_development"
    )
    if not isinstance(bundles, dict):
        raise ValueError("CF-RPEP checkpoint has no cached OOF bundles")
    bundles = {int(key): value for key, value in bundles.items()}
    if set(bundles) != {0, 1, 2, 3}:
        raise ValueError("HHP requires four complete OOF bundles")

    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    dists_mats = make_distance_mats(hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    raw = {"a": 1.0, "b": 0.0}
    coherent_bundles = {}
    for fold, bundle in bundles.items():
        coherent = dict(bundle)
        coherent["entcomp_unknown"] = leaf_coherent_entcomp_unknown(
            bundle["leaf_probabilities"],
            hierarchy,
            leaf_nodes=bundle["retained_classes"],
            parent_nodes=bundle["parent_nodes"],
        )
        coherent_bundles[fold] = coherent
    evidence_sources = {
        "multidepth": bundles,
        "coherent": coherent_bundles,
    }
    rows = {}
    for evidence_name, used_bundles in evidence_sources.items():
        for decoder in ("expected_hdist", "map"):
            raw_hazard = {
                fold: evaluate_hazard_bundle(
                    hierarchy,
                    used_bundles[fold],
                    raw,
                    distance_matrix,
                    dists_mats,
                    decoder=decoder,
                )
                for fold in range(4)
            }
            rows[
                f"{evidence_name}_raw_hazard/{decoder}"
            ] = summarize_outer(raw_hazard, hierarchy, dists_mats)
            nested_hazard = {
                fold: nested_hazard_fold(
                    hierarchy,
                    used_bundles[fold],
                    distance_matrix,
                    dists_mats,
                    decoder=decoder,
                    max_iter=args.scalar_max_iter,
                )
                for fold in range(4)
            }
            rows[
                f"{evidence_name}_nested_hazard/{decoder}"
            ] = summarize_outer(nested_hazard, hierarchy, dists_mats)
            raw_route = {
                fold: evaluate_target_fold(
                    hierarchy,
                    used_bundles[fold],
                    raw,
                    distance_matrix,
                    dists_mats,
                    decoder=decoder,
                )
                for fold in range(4)
            }
            rows[
                f"{evidence_name}_raw_route/{decoder}"
            ] = summarize_outer(raw_route, hierarchy, dists_mats)
            nested_route = {
                fold: nested_route_fold(
                    hierarchy,
                    used_bundles[fold],
                    distance_matrix,
                    dists_mats,
                    decoder=decoder,
                    max_iter=args.scalar_max_iter,
                )
                for fold in range(4)
            }
            rows[
                f"{evidence_name}_nested_route/{decoder}"
            ] = summarize_outer(nested_route, hierarchy, dists_mats)

    primary_name = args.primary_row
    if primary_name not in rows:
        raise ValueError(f"Configured HHP primary row is absent: {primary_name}")
    gate = locked_gate(rows[primary_name])
    transfer_scalar_fit = raw if gate["passed"] else None
    if gate["passed"] and "_nested_hazard/" in primary_name:
        selected_bundles = (
            coherent_bundles
            if primary_name.startswith("coherent_")
            else bundles
        )
        transfer_scalar_fit = fit_shared_hazard_scalars(
            [selected_bundles[fold] for fold in range(4)],
            hierarchy,
            max_iter=args.scalar_max_iter,
        )
    result = {
        "status": "nested_oof_go" if gate["passed"] else "nested_oof_no_go",
        "stage": STAGE,
        "method": "hierarchical_hazard_posterior",
        "threshold_free_inference": True,
        "node_specific_parameters": False,
        "depth_specific_parameters": False,
        "actual_ood_loaded_or_encoded": False,
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "primary": primary_name,
        "primary_decoder": args.primary_decoder,
        "primary_evidence_source": (
            "leaf_coherent"
            if primary_name.startswith("coherent_")
            else "multidepth_heads"
        ),
        "rows": rows,
        "gate": gate,
        "transfer_scalar_fit": transfer_scalar_fit,
    }
    checkpoint_payload = {
        "stage": STAGE,
        "method": result["method"],
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "primary": primary_name,
        "gate": gate,
        "transfer_scalar_fit": transfer_scalar_fit,
        "actual_ood_encoded": False,
    }
    checkpoint_path = Path(args.checkpoint_path)
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    atomic_torch_save(checkpoint_payload, checkpoint_path)
    atomic_torch_save(result, result_path)
    ensure_dir(diagnostics_path.parent)
    save_json(diagnostics_path, json_ready(result))
    for name, row in rows.items():
        print(
            f"HHP {name}: "
            f"ID={float(row['posterior_known']['balanced_acc']):.6f}, "
            f"pseudo="
            f"{float(row['pseudo_mapped_parent']['balanced_acc']):.6f}, "
            f"mix={float(row['mixed']['mixed_balanced_acc']):.6f}, "
            f"BMHD={float(row['mixed']['mixed_balanced_hdist']):.6f}, "
            f"AUROC={float(row['unknown_mass_binary']['auroc']):.6f}"
        )
    print(
        f"HHP nested gate={'GO' if gate['passed'] else 'NO-GO'}; "
        "actual OOD was not loaded."
    )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
