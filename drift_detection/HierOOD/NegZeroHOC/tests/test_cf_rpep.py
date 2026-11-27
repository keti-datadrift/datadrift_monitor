from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import tempfile

import torch

from ProHOC.libs.utils.score_util import entcompprob
from negzerohoc.cf_rpep import (
    canonical_named_tensor_hash,
    fit_shared_hazard_scalars,
    fit_shared_route_scalars,
    fold_weight_identity,
    hierarchical_hazard_terminal,
    leaf_coherent_entcomp_unknown,
    macro_terminal_weights,
    route_preserving_terminal,
    streaming_file_identity,
)
from negzerohoc.multidepth_fusion import (
    get_multidepth_classes,
    multidepth_unknown_probabilities,
)
from negzerohoc.relational_hazard import (
    fit_shared_relational_hazard,
    relational_hazard_terminal,
)
from negzerohoc.crossfit_class_holdout import (
    build_topology_holdout_manifest,
    tensor_partitions_hash,
)
from negzerohoc.evaluation import build_hierarchy
from scripts.train_cf_rpep_oof import (
    calibration_bundles_for_target,
    nested_confirmatory_metadata,
    nested_inner_assignments,
    ordered_evaluation_subset,
    load_fold_checkpoint_with_identity,
    fold_query_timing_metadata,
    load_config,
    screening_audit_metadata,
    validate_fold_checkpoints,
    verify_fold_checkpoint_identities,
)
from scripts.train_crossfit_class_holdout_lora import (
    CHECKPOINT_STAGE,
    build_fold_metric_topology,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_route_preserving_terminal_preserves_leaf_and_parent_odds():
    leaf = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    parent_mass = torch.tensor([[0.5]], dtype=torch.float64)
    evidence = torch.tensor([[0.4]], dtype=torch.float64)
    terminal = route_preserving_terminal(
        leaf,
        parent_mass,
        evidence,
        leaf_node_indices=torch.tensor([2, 3, 4]),
        parent_node_indices=torch.tensor([1]),
        node_count=5,
        a=1.7,
        b=-0.2,
    )
    assert torch.allclose(
        terminal.sum(dim=1), torch.ones(1, dtype=torch.float64)
    )
    assert torch.allclose(
        terminal[0, 2] / terminal[0, 3],
        leaf[0, 0] / leaf[0, 1],
    )
    descendant_known = terminal[0, 2] + terminal[0, 3]
    expected_odds = torch.exp(
        1.7 * torch.logit(torch.tensor(0.4, dtype=torch.float64))
        - 0.2
    )
    assert torch.allclose(
        terminal[0, 1] / descendant_known, expected_odds
    )
    assert terminal[0, 0] == 0


def test_hierarchical_hazard_terminal_telescopes_without_global_competition():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "p", "q", "a", "b"],
        node_ancestors={
            "a": [0, 1],
            "b": [0, 1, 2],
        },
    )
    leaf = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    evidence = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    terminal = hierarchical_hazard_terminal(
        leaf,
        evidence,
        hierarchy,
        leaf_nodes=["a", "b"],
        parent_nodes=["p", "q"],
        leaf_node_indices=torch.tensor([3, 4]),
        parent_node_indices=torch.tensor([1, 2]),
        node_count=5,
        a=1.0,
        b=0.0,
    )
    expected = torch.tensor([[
        0.0,
        0.2,
        0.75 * 0.8 * 0.4,
        0.25 * 0.8,
        0.75 * 0.8 * 0.6,
    ]], dtype=torch.float64)
    assert torch.allclose(terminal, expected)
    assert torch.allclose(
        terminal.sum(dim=1), torch.ones(1, dtype=torch.float64)
    )


def test_hazard_scalar_fit_improves_macro_terminal_nll():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "p", "a", "b"],
        node_ancestors={"a": [0, 1], "b": [0, 1]},
    )
    bundle = {
        "leaf_probabilities": torch.tensor([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        "entcomp_unknown": torch.tensor([
            [0.05], [0.05], [0.95], [0.95]
        ]),
        "leaf_node_indices": torch.tensor([2, 3]),
        "parent_node_indices": torch.tensor([1]),
        "retained_classes": ["a", "b"],
        "parent_nodes": ["p"],
        "node_count": 4,
        "target_node_indices": torch.tensor([2, 3, 1, 1]),
        "kinds": ["known", "known", "pseudo", "pseudo"],
        "target_groups": ["a", "b", "p", "p"],
    }
    result = fit_shared_hazard_scalars(
        [bundle], hierarchy, max_iter=100
    )
    assert result["a"] > 0.0
    assert result["final_nll"] < result["initial_nll"]


def test_leaf_coherent_entcomp_uses_visible_immediate_branches():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "p", "x", "a", "b", "c"],
        node_ancestors={
            "a": [0, 1],
            "b": [0, 1, 2],
            "c": [0, 1, 2],
        },
    )
    leaf = torch.tensor([[0.5, 0.3, 0.2]])
    evidence = leaf_coherent_entcomp_unknown(
        leaf,
        hierarchy,
        leaf_nodes=["a", "b", "c"],
        parent_nodes=["p", "x"],
    )
    p_branches = torch.tensor([0.5, 0.5])
    p_entropy = -(p_branches * p_branches.log()).sum()
    x_branches = torch.tensor([0.6, 0.4])
    x_entropy = -(x_branches * x_branches.log()).sum()
    expected = torch.tensor([[
        p_entropy / (1.0 + p_entropy),
        (x_entropy + 0.5) / (1.0 + x_entropy),
    ]])
    assert torch.allclose(evidence, expected, atol=1e-6)


def test_shared_relational_hazard_fits_and_normalizes():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "p", "a", "b"],
        parent2children={"root": ["p"], "p": ["a", "b"]},
        node_ancestors={
            "root": [],
            "p": [0],
            "a": [0, 1],
            "b": [0, 1],
        },
    )
    bundle = {
        "leaf_probabilities": torch.tensor([
            [0.90, 0.10],
            [0.10, 0.90],
            [0.55, 0.45],
            [0.45, 0.55],
        ]),
        "entcomp_unknown": torch.tensor([
            [0.05], [0.05], [0.90], [0.90]
        ]),
        "leaf_node_indices": torch.tensor([2, 3]),
        "parent_node_indices": torch.tensor([1]),
        "retained_classes": ["a", "b"],
        "parent_nodes": ["p"],
        "node_count": 4,
        "target_node_indices": torch.tensor([2, 3, 1, 1]),
        "kinds": ["known", "known", "pseudo", "pseudo"],
        "target_groups": ["a", "b", "p", "p"],
    }
    model = fit_shared_relational_hazard(
        [bundle], hierarchy, max_iter=100, l2_weight=1e-3
    )
    assert model["final_objective"] < model["initial_objective"]
    assert model["weight"].numel() == 10
    terminal = relational_hazard_terminal(
        bundle, hierarchy, model
    )
    assert torch.allclose(
        terminal.sum(dim=1), torch.ones(4), atol=1e-5
    )


def test_macro_weights_equalize_known_pseudo_and_target_groups():
    kinds = ["known"] * 3 + ["pseudo"] * 4
    groups = ["a", "a", "b", "p", "p", "p", "q"]
    weights = macro_terminal_weights(kinds, groups)
    half = torch.tensor(0.5, dtype=torch.float64)
    assert torch.isclose(weights[:3].sum(), half)
    assert torch.isclose(weights[3:].sum(), half)
    assert torch.isclose(weights[:2].sum(), weights[2])
    assert torch.isclose(weights[3:6].sum(), weights[6])


def test_full_batch_lbfgs_fits_positive_shared_a_and_improves_nll():
    bundle = {
        "leaf_probabilities": torch.tensor([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.5, 0.5],
        ]),
        "parent_mass": torch.ones(4, 1),
        "entcomp_unknown": torch.tensor([
            [0.05], [0.05], [0.95], [0.95]
        ]),
        "leaf_node_indices": torch.tensor([1, 2]),
        "parent_node_indices": torch.tensor([0]),
        "node_count": 3,
        "target_node_indices": torch.tensor([1, 2, 0, 0]),
        "kinds": ["known", "known", "pseudo", "pseudo"],
        "target_groups": ["a", "b", "p", "p"],
    }
    result = fit_shared_route_scalars([bundle], max_iter=100)
    assert result["a"] > 0.0
    assert result["final_nll"] < result["initial_nll"]
    assert result["optimizer"] == "LBFGS"
    assert result["line_search"] == "strong_wolfe"
    assert abs(result["known_weight_sum"] - 0.5) < 1e-8
    assert abs(result["pseudo_weight_sum"] - 0.5) < 1e-8


def minimal_bundle(class_names, original_indices, kinds):
    count = len(class_names)
    return {
        "leaf_probabilities": torch.full((count, 2), 0.5),
        "parent_mass": torch.ones(count, 1),
        "entcomp_unknown": torch.full((count, 1), 0.5),
        "target_node_indices": torch.zeros(count, dtype=torch.long),
        "leaf_node_indices": torch.tensor([1, 2]),
        "parent_node_indices": torch.tensor([0]),
        "node_count": 3,
        "kinds": list(kinds),
        "target_groups": [
            name if kind == "known" else "parent"
            for name, kind in zip(class_names, kinds)
        ],
        "class_names": list(class_names),
        "original_indices": list(original_indices),
    }


def test_crossfold_fit_excludes_target_classes_and_query_image_ids():
    bundles = {
        0: minimal_bundle(
            ["a", "z"], [1, 2], ["pseudo", "known"]
        ),
        1: minimal_bundle(
            ["a", "b", "c"], [10, 1, 11],
            ["known", "known", "pseudo"],
        ),
        2: minimal_bundle(
            ["d", "e"], [12, 13], ["known", "pseudo"]
        ),
        3: minimal_bundle(
            ["f", "g"], [14, 15], ["known", "pseudo"]
        ),
    }
    manifest = {
        "folds": [
            {"heldout_leaves": ["a"]},
            {"heldout_leaves": ["b"]},
            {"heldout_leaves": ["c"]},
            {"heldout_leaves": ["d"]},
        ]
    }
    selected, audit = calibration_bundles_for_target(
        0, bundles, manifest
    )
    assert sum(len(bundle["kinds"]) for bundle in selected) == 5
    assert audit["excluded_episode_count"] == 2
    assert audit["excluded_known_count"] == 2
    assert audit["excluded_for_target_class_identity_count"] == 1
    assert audit["excluded_for_target_query_image_identity_count"] == 1
    assert audit["scope"] == "episode_level_only"
    assert audit["weight_level_target_disjoint"] is False
    assert len(audit["excluded_records_hash"]) == 64
    assert len(audit["target_query_indices_hash"]) == 64


def test_crossfit_gate_is_explicitly_non_nested_screening_only():
    metadata = screening_audit_metadata()
    assert metadata["crossfit_level"] == "non_nested_screening"
    assert metadata["strict_confirmatory_gate"] is False
    assert metadata["may_unlock_official_ood"] is False
    assert metadata["required_confirmatory_calibration"] == (
        "nested_crossfit_or_independent_meta_calibration"
    )
    limitation = metadata["crossfit_limitation"]
    assert "model weights" in limitation
    assert "target-disjoint" in limitation
    assert "official OOD" in limitation


def test_nested_protocol_is_strict_and_can_unlock_only_after_gate():
    metadata = nested_confirmatory_metadata()
    assert metadata["crossfit_level"] == (
        "nested_class_and_image_crossfit"
    )
    assert metadata["strict_confirmatory_gate"] is True
    assert metadata["may_unlock_official_ood"] is True
    assert metadata["actual_ood_still_excluded_from_this_stage"] is True


def test_nested_assignment_is_image_disjoint_and_pseudo_class_disjoint():
    hierarchy = SimpleNamespace(node_ancestors={
        "p0": [0],
        "p1": [0],
        "p2": [0, 1],
    })
    kinds = ["known"] * 9 + ["pseudo"] * 6
    class_names = (
        ["a"] * 3 + ["b"] * 3 + ["c"] * 3
        + ["u0"] * 2 + ["u1"] * 2 + ["u2"] * 2
    )
    target_groups = (
        class_names[:9]
        + ["p0"] * 2 + ["p1"] * 2 + ["p2"] * 2
    )
    bundle = {
        "kinds": kinds,
        "class_names": class_names,
        "target_groups": target_groups,
        "original_indices": list(range(len(kinds))),
        "leaf_probabilities": torch.full((15, 2), 0.5),
        "parent_mass": torch.ones(15, 1),
        "entcomp_unknown": torch.full((15, 1), 0.5),
        "target_node_indices": torch.zeros(15, dtype=torch.long),
        "leaf_node_indices": torch.tensor([1, 2]),
        "parent_node_indices": torch.tensor([0]),
        "node_count": 3,
    }
    assignments, audit = nested_inner_assignments(bundle, hierarchy)
    assert sorted(torch.unique(assignments).tolist()) == [0, 1, 2]
    assert audit["pseudo_class_counts"] == [1, 1, 1]
    for class_name in ("u0", "u1", "u2"):
        indices = [
            index for index, value in enumerate(class_names)
            if value == class_name
        ]
        assert len(torch.unique(assignments[indices])) == 1
    for class_name in ("a", "b", "c"):
        indices = [
            index for index, value in enumerate(class_names)
            if value == class_name
        ]
        assert sorted(assignments[indices].tolist()) == [0, 1, 2]
    for inner_fold in range(3):
        subset = ordered_evaluation_subset(
            bundle, assignments == inner_fold
        )
        assert subset["known_count"] == 3
        assert subset["pseudo_count"] == 2
        assert subset["kinds"] == ["known"] * 3 + ["pseudo"] * 2


def test_query_timing_is_fold_local_not_global_two_pass():
    metadata = fold_query_timing_metadata()
    assert metadata == {
        "query_encoded_after_this_fold_model_selection": True
    }
    assert "query_encoded_after_all_fold_model_selection" not in metadata


def test_direct_entcomp_extraction_and_terminal_embedding_regression():
    hierarchy = SimpleNamespace(
        id_node_list=["root", "p", "q", "a", "b", "c", "d"],
        parent2children={
            "root": ["p", "q"],
            "p": ["a", "b"],
            "q": ["c", "d"],
        },
        node_ancestors={
            "root": [],
            "p": [0],
            "q": [0],
            "a": [0, 1],
            "b": [0, 1],
            "c": [0, 2],
            "d": [0, 2],
        },
        max_depth=2,
    )
    multidepth = get_multidepth_classes(
        hierarchy, ["a", "b", "c", "d"]
    )
    leaf = torch.tensor([[0.30, 0.10, 0.20, 0.40]])
    probabilities = [torch.tensor([[0.60, 0.40]]), leaf]
    extracted = multidepth_unknown_probabilities(
        probabilities, hierarchy, multidepth, entcompprob
    )

    def direct_unknown(group):
        group_sum = group.sum(dim=1)
        local = group / group_sum[:, None]
        entropy = -(local * local.log()).sum(dim=1)
        score = entropy + (1.0 - group_sum)
        return score / (group_sum + score)

    direct = torch.stack([
        direct_unknown(leaf[:, :2]),
        direct_unknown(leaf[:, 2:]),
    ], dim=1)
    extracted_matrix = torch.stack(
        [extracted["p"], extracted["q"]], dim=1
    )
    assert torch.allclose(extracted_matrix, direct, atol=1e-7)

    parent_mass = torch.tensor([[0.40, 0.60]])
    terminal = route_preserving_terminal(
        leaf,
        parent_mass,
        extracted_matrix,
        leaf_node_indices=torch.tensor([3, 4, 5, 6]),
        parent_node_indices=torch.tensor([1, 2]),
        node_count=7,
        a=1.0,
        b=0.0,
    )
    parent_weight = parent_mass * direct / (1.0 - direct)
    normalizer = 1.0 + parent_weight.sum(dim=1, keepdim=True)
    expected = torch.zeros(1, 7)
    expected[:, 1:3] = parent_weight / normalizer
    expected[:, 3:] = leaf / normalizer
    assert torch.allclose(terminal, expected, atol=1e-7)
    assert torch.allclose(terminal.sum(dim=1), torch.ones(1))


def make_synthetic_final_fold_checkpoints(args):
    hierarchy, classes = build_hierarchy(
        REPO_ROOT, args.id_split, args.hierarchy
    )
    manifest = build_topology_holdout_manifest(
        hierarchy, classes, seed=0
    )
    targets = torch.arange(80).repeat_interleave(3)
    dataset = SimpleNamespace(classes=classes, targets=targets.tolist())
    checkpoints = []
    paths = []
    for fold, fold_record in enumerate(manifest["folds"]):
        heldout = set(
            fold_record["heldout_original_class_indices"]
        )
        retained = [
            index for index in range(80) if index not in heldout
        ]
        partitions = {
            "representation_train": torch.tensor([
                3 * target for target in retained
            ]),
            "model_selection": torch.tensor([
                3 * target + 1 for target in retained
            ]),
            "known_query": torch.tensor([
                3 * target + 2 for target in retained
            ]),
            "heldout_query": torch.tensor([
                3 * target + offset
                for target in sorted(heldout)
                for offset in range(3)
            ]),
        }
        split_hash = tensor_partitions_hash(partitions)
        retained_classes = [classes[index] for index in retained]
        _, _, provenance = build_fold_metric_topology(
            args, hierarchy, retained_classes
        )
        false_flags = {
            "used_heldout_class_images_for_representation_training": False,
            "used_heldout_class_images_for_proxy_initialization": False,
            "used_heldout_class_images_for_model_selection": False,
            "used_known_query_for_training_or_selection": False,
            "used_official_test_for_training_or_selection": False,
        }
        checkpoints.append({
            "stage": CHECKPOINT_STAGE,
            "dataset": args.dataset,
            "clip_model": args.clip_model,
            "hierarchy": args.hierarchy,
            "id_split": args.id_split,
            "vision_lora_state_dict": {"lora": torch.ones(1)},
            "metric_proxies": torch.zeros(len(retained), 4),
            "metric_proxy_classes": retained_classes,
            "training_state": None,
            "metrics": {"split_hash": split_hash, **false_flags},
            "crossfit_manifest": manifest,
            "crossfit_manifest_hash": manifest["manifest_hash"],
            "crossfit_fold": fold_record,
            "crossfit_split_indices": partitions,
            "crossfit_split_hash": split_hash,
            "crossfit_hierarchy_provenance": provenance,
        })
        paths.append(f"synthetic/fold-{fold}/checkpoints/best.pt")
    return hierarchy, dataset, checkpoints, paths


def test_strict_fold_validation_checks_all_provenance():
    args = load_config(
        REPO_ROOT
        / "configs/19_cf_rpep/fgvc_aircraft_oof_gate_gpu0.yaml"
    )
    hierarchy, dataset, checkpoints, paths = (
        make_synthetic_final_fold_checkpoints(args)
    )
    manifest, records = validate_fold_checkpoints(
        args, checkpoints, paths, hierarchy, dataset
    )
    assert manifest["eligible_leaf_count"] == 50
    assert set(records) == {0, 1, 2, 3}
    checkpoints[0]["crossfit_split_hash"] = "tampered"
    try:
        validate_fold_checkpoints(
            args, checkpoints, paths, hierarchy, dataset
        )
    except ValueError as error:
        assert "split hash" in str(error)
    else:
        raise AssertionError("Tampered split provenance was accepted")


def test_cf_rpep_config_locks_four_best_checkpoints_and_no_ood_path():
    args = load_config(
        REPO_ROOT
        / "configs/19_cf_rpep/fgvc_aircraft_oof_gate_gpu0.yaml"
    )
    assert args.device == "cuda:0"
    assert len(args.fold_checkpoints) == 4
    assert all(
        Path(path).name == "best.pt" for path in args.fold_checkpoints
    )
    assert args.scalar_max_iter == 100
    assert not any(
        "ood" in key.lower()
        for key in args.raw_config.get("cf_rpep", {})
    )


def test_streaming_and_tensor_identities_are_content_addressed():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "weights.bin"
        content = b"cross-fit-checkpoint" * 101
        path.write_bytes(content)
        identity = streaming_file_identity(path, chunk_size=17)
        assert identity["canonical_path"] == str(path.resolve())
        assert identity["file_size"] == len(content)
        assert identity["sha256"] == hashlib.sha256(content).hexdigest()

    first = {
        "b": torch.tensor([[1.0, 2.0]]),
        "a": torch.tensor([3], dtype=torch.int64),
    }
    reordered = {"a": first["a"], "b": first["b"]}
    changed = {"a": first["a"], "b": first["b"] + 1.0}
    assert canonical_named_tensor_hash(first) == (
        canonical_named_tensor_hash(reordered)
    )
    assert canonical_named_tensor_hash(first) != (
        canonical_named_tensor_hash(changed)
    )
    weights = fold_weight_identity({
        "vision_lora_state_dict": first,
        "metric_proxies": torch.eye(2),
    })
    assert all(len(value) == 64 for value in weights.values())


def test_loaded_checkpoint_identity_detects_later_overwrite():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        checkpoint = {
            "vision_lora_state_dict": {
                "lora": torch.tensor([1.0, 2.0])
            },
            "metric_proxies": torch.eye(2),
        }
        torch.save(checkpoint, path)
        loaded, identity = load_fold_checkpoint_with_identity(path)
        folds = {0: {"path": str(path), "checkpoint": loaded}}
        identities = {0: identity}
        verify_fold_checkpoint_identities(folds, identities)
        torch.save({
            **checkpoint,
            "metric_proxies": torch.ones(2, 2),
        }, path)
        try:
            verify_fold_checkpoint_identities(folds, identities)
        except RuntimeError as error:
            assert "overwritten" in str(error)
        else:
            raise AssertionError("Overwritten fold checkpoint was accepted")
