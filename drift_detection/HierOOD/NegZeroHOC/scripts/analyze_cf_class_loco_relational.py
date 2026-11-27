from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.cf_rpep import leaf_coherent_entcomp_unknown
from negzerohoc.checkpointing import load_idea3_checkpoint
from negzerohoc.config_utils import load_yaml_config
from negzerohoc.crossfit_class_holdout import mapped_retained_ancestor
from negzerohoc.evaluation import build_hierarchy, make_distance_mats
from negzerohoc.feature_io import ensure_dir, save_json
from negzerohoc.output_layout import resolve_experiment_artifact
from negzerohoc.relational_hazard import fit_shared_relational_hazard
from scripts.analyze_cf_hierarchical_hazard import (
    atomic_torch_save,
    locked_gate,
    summarize_outer,
)
from scripts.analyze_cf_relational_hazard import (
    evaluate_relational_bundle,
)
from scripts.train_cf_rpep_oof import STAGE as CF_RPEP_STAGE
from scripts.train_crossfit_class_holdout_lora import (
    build_fold_metric_topology,
)
from scripts.train_paper_negprompt_ablation import json_ready


STAGE = "crossfit_class_loco_relational_meta_head"


def load_config(path: str | Path) -> Namespace:
    cfg = load_yaml_config(path)
    experiment = cfg.get("experiment", {})
    stage = cfg.get("class_loco_relational", {})
    dataset = cfg.get("dataset", {})
    experiment_name = str(
        experiment.get("name", "class-loco-relational")
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


def parent_nodes(hierarchy):
    return [
        node for node in hierarchy.id_node_list
        if node != "root" and node in hierarchy.parent2children
    ]


def base_bundle(
    probabilities,
    hierarchy,
    classes,
    target_names,
    kinds,
    target_groups,
    class_names,
    original_indices,
):
    parents = parent_nodes(hierarchy)
    node_to_index = {
        node: index for index, node in enumerate(hierarchy.id_node_list)
    }
    probabilities = probabilities.float()
    return {
        "_hierarchy": hierarchy,
        "leaf_probabilities": probabilities,
        "entcomp_unknown": leaf_coherent_entcomp_unknown(
            probabilities,
            hierarchy,
            leaf_nodes=classes,
            parent_nodes=parents,
        ),
        "leaf_node_indices": torch.tensor([
            node_to_index[name] for name in classes
        ], dtype=torch.long),
        "parent_node_indices": torch.tensor([
            node_to_index[name] for name in parents
        ], dtype=torch.long),
        "retained_classes": list(classes),
        "parent_nodes": parents,
        "node_count": len(hierarchy.id_node_list),
        "target_node_indices": torch.tensor([
            node_to_index[name] for name in target_names
        ], dtype=torch.long),
        "kinds": list(kinds),
        "target_groups": list(target_groups),
        "class_names": list(class_names),
        "original_indices": [
            int(value) for value in original_indices
        ],
    }


def build_meta_loco_bundles(args, full_hierarchy, meta):
    probabilities = meta["leaf_probabilities"].float()
    targets = meta["targets"].long()
    original_indices = meta["original_indices"].long()
    classes = list(meta["classes"])
    if probabilities.shape != (
        int(targets.numel()),
        len(classes),
    ):
        raise ValueError("CLRM meta probabilities are misaligned")
    fold_hierarchy, _, _ = build_fold_metric_topology(
        args, full_hierarchy, classes
    )
    class_names = [classes[int(value)] for value in targets]
    bundles = [base_bundle(
        probabilities,
        fold_hierarchy,
        classes,
        class_names,
        ["known"] * len(class_names),
        class_names,
        class_names,
        original_indices.tolist(),
    )]
    eligible = []
    for class_index, class_name in enumerate(classes):
        visible = [
            value for value in classes if value != class_name
        ]
        mapped = mapped_retained_ancestor(
            fold_hierarchy, class_name, visible
        )
        if mapped is None:
            continue
        row_indices = torch.nonzero(
            targets == class_index, as_tuple=False
        ).flatten()
        if int(row_indices.numel()) == 0:
            continue
        visible_indices = torch.tensor([
            index for index in range(len(classes))
            if index != class_index
        ], dtype=torch.long)
        visible_probabilities = probabilities.index_select(
            0, row_indices
        ).index_select(1, visible_indices)
        visible_probabilities = (
            visible_probabilities
            / visible_probabilities.sum(dim=1, keepdim=True).clamp_min(
                1e-12
            )
        )
        pruned_hierarchy, _, _ = build_fold_metric_topology(
            args, full_hierarchy, visible
        )
        if mapped not in pruned_hierarchy.id_node_list:
            raise RuntimeError("CLRM mapped parent disappeared after pruning")
        count = int(row_indices.numel())
        bundles.append(base_bundle(
            visible_probabilities,
            pruned_hierarchy,
            visible,
            [mapped] * count,
            ["pseudo"] * count,
            [mapped] * count,
            [class_name] * count,
            original_indices.index_select(0, row_indices).tolist(),
        ))
        eligible.append(class_name)
    if len(eligible) < 20:
        raise RuntimeError("CLRM has too few non-root meta-LOCO classes")
    return bundles, {
        "known_episode_count": int(targets.numel()),
        "pseudo_episode_count": sum(
            len(bundle["kinds"]) for bundle in bundles[1:]
        ),
        "eligible_class_count": len(eligible),
        "eligible_classes": eligible,
        "source_split": meta["source_split"],
        "used_outer_heldout_classes": False,
        "used_known_query": False,
    }


def main():
    args = parse_args()
    if not args.input_checkpoint:
        raise ValueError(
            "Missing class_loco_relational.input_checkpoint"
        )
    if args.primary_decoder not in {"expected_hdist", "map"}:
        raise ValueError("CLRM primary decoder is invalid")
    source = load_idea3_checkpoint(
        args.input_checkpoint, map_location="cpu"
    )
    if source.get("stage") != CF_RPEP_STAGE:
        raise ValueError("CLRM input is not a CF-RPEP checkpoint")
    if source.get("actual_ood_encoded") is not False:
        raise ValueError("CLRM input must be actual-OOD-free")
    oof_bundles = source.get(
        "oof_bundles_for_threshold_free_method_development"
    )
    fold_models = source.get("fold_models")
    if not isinstance(oof_bundles, dict) or not isinstance(
        fold_models, dict
    ):
        raise ValueError("CLRM input lacks fold development payloads")
    oof_bundles = {int(key): value for key, value in oof_bundles.items()}
    fold_models = {int(key): value for key, value in fold_models.items()}
    if set(oof_bundles) != {0, 1, 2, 3} or set(
        fold_models
    ) != {0, 1, 2, 3}:
        raise ValueError("CLRM requires folds 0..3")

    full_hierarchy, _ = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    dists_mats = make_distance_mats(full_hierarchy)
    distance_matrix = (dists_mats[0] + dists_mats[1]).float()
    outer_models = {}
    meta_audits = {}
    evaluations = {
        "expected_hdist": {},
        "map": {},
    }
    for fold in range(4):
        meta = fold_models[fold].get("class_loco_meta_selection")
        if not isinstance(meta, dict):
            raise ValueError(
                "CLRM input predates class-LOCO meta payload support"
            )
        meta_bundles, audit = build_meta_loco_bundles(
            args, full_hierarchy, meta
        )
        model = fit_shared_relational_hazard(
            meta_bundles,
            full_hierarchy,
            max_iter=args.max_iter,
            l2_weight=args.l2_weight,
        )
        outer_models[fold] = model
        meta_audits[fold] = audit
        outer_bundle = dict(oof_bundles[fold])
        outer_bundle["entcomp_unknown"] = (
            leaf_coherent_entcomp_unknown(
                outer_bundle["leaf_probabilities"],
                full_hierarchy,
                leaf_nodes=outer_bundle["retained_classes"],
                parent_nodes=outer_bundle["parent_nodes"],
            )
        )
        for decoder in evaluations:
            evaluations[decoder][fold] = evaluate_relational_bundle(
                full_hierarchy,
                outer_bundle,
                model,
                distance_matrix,
                dists_mats,
                decoder=decoder,
            )
    rows = {
        decoder: summarize_outer(
            values, full_hierarchy, dists_mats
        )
        for decoder, values in evaluations.items()
    }
    gate = locked_gate(rows[args.primary_decoder])
    result = {
        "status": "outer_oof_go" if gate["passed"] else "outer_oof_no_go",
        "stage": STAGE,
        "method": "class_loco_shared_relational_meta_head",
        "threshold_free_inference": True,
        "node_specific_parameters": False,
        "depth_specific_parameters": False,
        "actual_ood_loaded_or_encoded": False,
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "l2_weight": args.l2_weight,
        "primary_decoder": args.primary_decoder,
        "meta_audits": meta_audits,
        "outer_models": outer_models,
        "rows": rows,
        "gate": gate,
    }
    checkpoint_payload = {
        "stage": STAGE,
        "method": result["method"],
        "input_checkpoint": args.input_checkpoint,
        "input_manifest_hash": source.get("manifest_hash"),
        "l2_weight": args.l2_weight,
        "primary_decoder": args.primary_decoder,
        "meta_audits": meta_audits,
        "outer_models": outer_models,
        "gate": gate,
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
            f"CLRM {decoder}: "
            f"ID={float(row['posterior_known']['balanced_acc']):.6f}, "
            f"pseudo="
            f"{float(row['pseudo_mapped_parent']['balanced_acc']):.6f}, "
            f"mix={float(row['mixed']['mixed_balanced_acc']):.6f}, "
            f"BMHD={float(row['mixed']['mixed_balanced_hdist']):.6f}, "
            f"AUROC={float(row['unknown_mass_binary']['auroc']):.6f}"
        )
    print(
        f"CLRM outer gate={'GO' if gate['passed'] else 'NO-GO'}; "
        "actual OOD was not loaded."
    )
    print(f"saved: {result_path}")


if __name__ == "__main__":
    main()
