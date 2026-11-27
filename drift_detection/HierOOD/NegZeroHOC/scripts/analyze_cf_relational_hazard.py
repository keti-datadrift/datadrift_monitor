from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.relational_hazard import (
    fit_shared_relational_hazard,
    relational_hazard_terminal,
)
from scripts.analyze_cf_hierarchical_hazard import (
    atomic_torch_save,
    combine_evaluations,
    locked_gate,
    summarize_outer,
)
from scripts.train_cf_rpep_oof import (
    STAGE as CF_RPEP_STAGE,
    nested_inner_assignments,
    ordered_evaluation_subset,
    subset_bundle,
)
from scripts.train_paper_negprompt_ablation import json_ready


STAGE = "crossfit_shared_relational_hazard_oof"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment = cfg.get("experiment", {})
    stage = cfg.get("relational_hazard", {})
    dataset = cfg.get("dataset", {})
    experiment_name = str(
        experiment.get("name", "crossfit-relational-hazard")
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
        max_iter=max(1, int(stage.get("max_iter", 150))),
        l2_weight=float(stage.get("l2_weight", 1e-3)),
        primary_decoder=str(
            stage.get("primary_decoder", "expected_hdist")
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


def evaluate_relational_bundle(
    hierarchy,
    bundle,
    model,
    distance_matrix,
    dists_mats,
    *,
    decoder,
):
    terminal = relational_hazard_terminal(
        bundle, hierarchy, model
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
        raise ValueError(
            f"Unsupported relational-hazard decoder: {decoder!r}"
        )
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
    }


def nested_relational_fold(
    hierarchy,
    bundle,
    distance_matrix,
    dists_mats,
    *,
    decoder,
    max_iter,
    l2_weight,
):
    assignments, assignment_audit = nested_inner_assignments(
        bundle, hierarchy
    )
    evaluations = []
    models = []
    for inner_fold in range(3):
        evaluation_mask = assignments == inner_fold
        model = fit_shared_relational_hazard(
            [subset_bundle(bundle, ~evaluation_mask)],
            hierarchy,
            max_iter=max_iter,
            l2_weight=l2_weight,
        )
        evaluation = evaluate_relational_bundle(
            hierarchy,
            ordered_evaluation_subset(bundle, evaluation_mask),
            model,
            distance_matrix,
            dists_mats,
            decoder=decoder,
        )
        evaluations.append(evaluation)
        models.append(model)
    combined = combine_evaluations(evaluations, hierarchy, dists_mats)
    combined.update({
        "inner_models": models,
        "assignment_audit": assignment_audit,
        "target_disjoint": True,
        "decoder": decoder,
    })
    return combined


def main():
    args = parse_args()
    if not args.input_checkpoint:
        raise ValueError("Missing relational_hazard.input_checkpoint")
    if args.primary_decoder not in {"expected_hdist", "map"}:
        raise ValueError(
            "Relational-hazard primary decoder must be expected_hdist or map"
        )
    source = load_idea3_checkpoint(
        args.input_checkpoint, map_location="cpu"
    )
    if source.get("stage") != CF_RPEP_STAGE:
        raise ValueError("Relational input is not a CF-RPEP checkpoint")
    if source.get("actual_ood_encoded") is not False:
        raise ValueError("Relational input must be actual-OOD-free")
    bundles = source.get(
        "oof_bundles_for_threshold_free_method_development"
    )
    if not isinstance(bundles, dict):
        raise ValueError("Relational input has no cached OOF bundles")
    bundles = {int(key): value for key, value in bundles.items()}
    if set(bundles) != {0, 1, 2, 3}:
        raise ValueError("Relational input requires folds 0..3")

    hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    dists_mats = make_distance_mats(hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    outer_by_decoder = {}
    rows = {}
    for decoder in ("expected_hdist", "map"):
        outer = {
            fold: nested_relational_fold(
                hierarchy,
                bundles[fold],
                distance_matrix,
                dists_mats,
                decoder=decoder,
                max_iter=args.max_iter,
                l2_weight=args.l2_weight,
            )
            for fold in range(4)
        }
        outer_by_decoder[decoder] = outer
        rows[decoder] = summarize_outer(
            outer, hierarchy, dists_mats
        )
    gate = locked_gate(rows[args.primary_decoder])
    transfer_model = None
    if gate["passed"]:
        transfer_model = fit_shared_relational_hazard(
            [bundles[fold] for fold in range(4)],
            hierarchy,
            max_iter=args.max_iter,
            l2_weight=args.l2_weight,
        )
    result = {
        "status": "nested_oof_go" if gate["passed"] else "nested_oof_no_go",
        "stage": STAGE,
        "method": "shared_linear_relational_hazard",
        "threshold_free_inference": True,
        "node_specific_parameters": False,
        "depth_specific_parameters": False,
        "actual_ood_loaded_or_encoded": False,
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "l2_weight": args.l2_weight,
        "primary_decoder": args.primary_decoder,
        "rows": rows,
        "gate": gate,
        "transfer_model": transfer_model,
    }
    checkpoint_payload = {
        "stage": STAGE,
        "method": result["method"],
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "l2_weight": args.l2_weight,
        "primary_decoder": args.primary_decoder,
        "gate": gate,
        "transfer_model": transfer_model,
        "actual_ood_encoded": False,
    }
    checkpoint_path = Path(args.checkpoint_path)
    result_path = Path(args.result_path)
    diagnostics_path = Path(args.diagnostics_path)
    atomic_torch_save(checkpoint_payload, checkpoint_path)
    atomic_torch_save(result, result_path)
    ensure_dir(diagnostics_path.parent)
    save_json(diagnostics_path, json_ready(result))
    for decoder, row in rows.items():
        print(
            f"SRHC {decoder}: "
            f"ID={float(row['posterior_known']['balanced_acc']):.6f}, "
            f"pseudo="
            f"{float(row['pseudo_mapped_parent']['balanced_acc']):.6f}, "
            f"mix={float(row['mixed']['mixed_balanced_acc']):.6f}, "
            f"BMHD={float(row['mixed']['mixed_balanced_hdist']):.6f}, "
            f"AUROC={float(row['unknown_mass_binary']['auroc']):.6f}"
        )
    print(
        f"SRHC nested gate={'GO' if gate['passed'] else 'NO-GO'}; "
        "actual OOD was not loaded."
    )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
