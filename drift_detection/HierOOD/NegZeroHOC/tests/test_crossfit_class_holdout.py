from __future__ import annotations

from pathlib import Path

import torch

from negzerohoc.crossfit_class_holdout import (
    RemappedSubset,
    build_topology_holdout_manifest,
    stratified_retained_image_split,
    tensor_partitions_hash,
    validate_topology_holdout_manifest,
)
from negzerohoc.evaluation import build_hierarchy
from negzerohoc.image_metric import class_tree_distance_matrix
from scripts.train_crossfit_class_holdout_lora import (
    TRIPLET_DISTANCE_SOURCE,
    build_fold_metric_topology,
    fold_partitions,
    fold_resume_signature,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def aircraft_hierarchy():
    return build_hierarchy(
        REPO_ROOT,
        REPO_ROOT / "data/fgvc-aircraft-id-labels.csv",
        REPO_ROOT / "hierarchies/fgvc-aircraft.json",
    )


def test_aircraft_loco_manifest_matches_audited_feasible_design():
    hierarchy, classes = aircraft_hierarchy()
    manifest = build_topology_holdout_manifest(
        hierarchy,
        classes,
        num_folds=4,
        requested_fold_size=16,
        seed=0,
    )
    repeated = build_topology_holdout_manifest(
        hierarchy,
        classes,
        num_folds=4,
        requested_fold_size=16,
        seed=0,
    )
    assert manifest == repeated
    assert manifest["eligible_leaf_count"] == 50
    assert len(manifest["ineligible_leaves"]) == 30
    assert manifest["feasible_fold_sizes"] == [13, 13, 12, 12]
    assert manifest["unassigned_eligible_leaves"] == []
    assert manifest["manifest_hash"] == (
        "b35c06cd0bee51ad0356de393d30d618c"
        "f617f1a15d9a3d1226e8e6862f6b847"
    )
    assert [fold["fold_hash"] for fold in manifest["folds"]] == [
        "fd8cafd5b79d7abcfb8f8b40c7695a254fb66a04dba7e0ea55d78ccf8fa4c475",
        "c0c38c469af5f08eacd580795013b84fa2d9fcae04f3ae15d9871b4041cb6871",
        "616264990b77ada3c2f4d30d2c353581d2a92839cee787e8d2fd37f6fb4675ae",
        "1a3a829d8b290cc3c8069269b947fe4a89bf01b7e9b2080c415930d3d20a10c7",
    ]
    all_heldout = [
        leaf
        for fold in manifest["folds"]
        for leaf in fold["heldout_leaves"]
    ]
    assert len(all_heldout) == len(set(all_heldout)) == 50
    for fold in manifest["folds"]:
        parents = set(fold["mapped_unknown_nodes"].values())
        depth1 = {
            parent for parent in parents
            if len(hierarchy.node_ancestors[parent]) == 1
        }
        depth2 = {
            parent for parent in parents
            if len(hierarchy.node_ancestors[parent]) == 2
        }
        assert len(parents) >= 10
        assert len(depth1) >= 7
        assert len(depth2) >= 2
        assert sum(
            depth == 2
            for depth in fold["mapped_parent_depths"].values()
        ) == 3
    validate_topology_holdout_manifest(
        hierarchy, classes, manifest
    )


def test_retained_split_is_complete_disjoint_stratified_and_stable():
    targets = torch.tensor(
        [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    )
    first = stratified_retained_image_split(
        targets, [0, 1, 2], seed=17
    )
    second = stratified_retained_image_split(
        targets, [0, 1, 2], seed=17
    )
    assert all(
        torch.equal(first[name], second[name]) for name in first
    )
    assert [
        torch.bincount(
            targets[first[name]], minlength=4
        ).tolist()
        for name in (
            "representation_train",
            "model_selection",
            "known_query",
        )
    ] == [
        [6, 6, 6, 0],
        [2, 2, 2, 0],
        [2, 2, 2, 0],
    ]
    combined = torch.cat(list(first.values()))
    assert len(combined) == len(torch.unique(combined)) == 30
    assert not bool((targets[combined] == 3).any())
    assert tensor_partitions_hash(first) == tensor_partitions_hash(
        second
    )


def test_fold_partitions_keep_heldout_out_of_all_retained_uses():
    targets = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
    partitions, retained = fold_partitions(
        targets,
        {"heldout_original_class_indices": [1]},
        seed=3,
    )
    assert retained == [0, 2]
    for name in (
        "representation_train",
        "model_selection",
        "known_query",
    ):
        assert not bool((targets[partitions[name]] == 1).any())
    assert bool((targets[partitions["heldout_query"]] == 1).all())
    combined = torch.cat(list(partitions.values()))
    assert len(combined) == len(torch.unique(combined)) == len(targets)


def test_remapped_subset_exposes_only_compact_retained_targets():
    class ToyDataset:
        targets = [0, 1, 2, 0, 1, 2]

        def __getitem__(self, index):
            return torch.tensor(index), self.targets[index]

        def __len__(self):
            return len(self.targets)

    subset = RemappedSubset(
        ToyDataset(),
        [0, 2, 3, 5],
        {0: 0, 2: 1},
        ["zero", "two"],
    )
    assert subset.classes == ["zero", "two"]
    assert subset.targets == [0, 1, 0, 1]
    assert [subset[index][1] for index in range(len(subset))] == [
        0, 1, 0, 1
    ]


def test_gpu_configs_share_manifest_design_and_partition_folds():
    gpu0 = load_config(
        REPO_ROOT
        / "configs/18_crossfit_class_holdout/"
        "fgvc_aircraft_loco_folds02_gpu0.yaml"
    )
    gpu1 = load_config(
        REPO_ROOT
        / "configs/18_crossfit_class_holdout/"
        "fgvc_aircraft_loco_folds13_gpu1.yaml"
    )
    assert gpu0.device == "cuda:0"
    assert gpu1.device == "cuda:1"
    assert gpu0.crossfit_folds == (0, 2)
    assert gpu1.crossfit_folds == (1, 3)
    assert set(gpu0.crossfit_folds).isdisjoint(gpu1.crossfit_folds)
    assert sorted(gpu0.crossfit_folds + gpu1.crossfit_folds) == [
        0, 1, 2, 3
    ]
    assert gpu0.crossfit_manifest_seed == 0
    assert gpu1.crossfit_manifest_seed == 0
    assert gpu0.resume_enabled is True
    assert gpu1.resume_enabled is True


def test_triplet_distance_uses_retained_only_pruned_hierarchy():
    args = load_config(
        REPO_ROOT
        / "configs/18_crossfit_class_holdout/"
        "fgvc_aircraft_loco_folds02_gpu0.yaml"
    )
    full_hierarchy, classes = aircraft_hierarchy()
    manifest = build_topology_holdout_manifest(
        full_hierarchy,
        classes,
        num_folds=4,
        requested_fold_size=16,
        seed=0,
    )
    heldout = set(manifest["folds"][0]["heldout_leaves"])
    retained = [leaf for leaf in classes if leaf not in heldout]
    fold_hierarchy, fold_distances, provenance = (
        build_fold_metric_topology(
            args, full_hierarchy, retained
        )
    )
    full_mapping = full_hierarchy.gen_ds2node_map(retained)
    full_nodes = [
        full_hierarchy.id_node_list[int(index)]
        for index in full_mapping.tolist()
    ]
    full_distances = class_tree_distance_matrix(
        full_hierarchy, full_nodes
    )
    assert fold_distances.shape == full_distances.shape
    assert not torch.equal(fold_distances, full_distances)
    assert int((fold_distances != full_distances).sum()) == 132
    assert provenance["triplet_distance_source"] == (
        TRIPLET_DISTANCE_SOURCE
    )
    assert provenance["full_hierarchy_role"] == "holdout_manifest_only"
    assert provenance["fold_hierarchy_role"] == (
        "retained_class_mapping_and_triplet_distance"
    )
    assert provenance[
        "heldout_topology_excluded_from_triplet_distance"
    ] is True
    assert provenance["full_hierarchy"]["node_count"] == 109
    assert provenance["fold_hierarchy"]["node_count"] == 94
    assert provenance["fold_hierarchy"]["node_list"] == (
        fold_hierarchy.id_node_list
    )
    assert provenance["full_hierarchy"]["topology_hash"] != (
        provenance["fold_hierarchy"]["topology_hash"]
    )
    signature = fold_resume_signature(
        args,
        fold=0,
        manifest_hash=manifest["manifest_hash"],
        split_hash="split",
        retained_classes=retained,
        fold_hierarchy_hash=provenance[
            "fold_hierarchy"
        ]["topology_hash"],
    )
    assert signature["triplet_distance_source"] == (
        TRIPLET_DISTANCE_SOURCE
    )
    assert signature["fold_hierarchy_hash"] == provenance[
        "fold_hierarchy"
    ]["topology_hash"]
