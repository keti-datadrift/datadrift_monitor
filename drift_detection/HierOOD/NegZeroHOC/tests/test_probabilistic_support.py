from __future__ import annotations

from types import SimpleNamespace

import torch

from negzerohoc.probabilistic_support import (
    SharedMaskedCategoricalLikelihood,
    SharedMaskedSupportLikelihood,
    SharedSupportPosterior,
    build_energy_episodes,
    build_global_leaf_energy_episodes,
    build_support_episodes,
    categorical_episode_targets,
    masked_subtree_terminal_distribution,
    mixture_conditionals_by_parent,
    global_unknown_probability,
    latent_knownness_terminal_distribution,
    prior_corrected_product_unknown,
    probabilistic_terminal_distribution,
    reference_only_partitions_from_checkpoints,
    stratified_calibration_train_val_split,
    stratified_four_way_split,
    support_evidence,
    validate_reference_only_split,
)


def toy_hierarchy():
    return SimpleNamespace(
        id_node_list=["root", "p-a", "p-b", "a1", "a2", "b1", "b2"],
        parent2children={
            "root": ["p-a", "p-b"],
            "p-a": ["a1", "a2"],
            "p-b": ["b1", "b2"],
        },
        node_ancestors={
            "root": [],
            "p-a": [0],
            "p-b": [0],
            "a1": [0, 1],
            "a2": [0, 1],
            "b1": [0, 2],
            "b2": [0, 2],
        },
    )


def test_four_way_split_is_complete_disjoint_and_stratified():
    targets = torch.tensor([0] * 20 + [1] * 20)
    partitions = stratified_four_way_split(targets, seed=3)
    flattened = [index for part in partitions for index in part.tolist()]
    assert sorted(flattened) == list(range(40))
    assert len(flattened) == len(set(flattened))
    assert [
        torch.bincount(targets[part], minlength=2).tolist()
        for part in partitions
    ] == [[12, 12], [4, 4], [2, 2], [2, 2]]


def test_saved_calibration_is_deterministically_split_8_per_5():
    targets = torch.tensor([0] * 20 + [1] * 20)
    reference = torch.tensor(
        list(range(7)) + list(range(20, 27))
    )
    calibration = torch.tensor(
        list(range(7, 20)) + list(range(27, 40))
    )
    validated_reference, validated_calibration = (
        validate_reference_only_split(
            targets, reference, calibration
        )
    )
    posterior_train, posterior_val = (
        stratified_calibration_train_val_split(
            targets, validated_calibration, seed=11
        )
    )
    repeated_train, repeated_val = (
        stratified_calibration_train_val_split(
            targets, validated_calibration, seed=11
        )
    )
    assert torch.equal(validated_reference, reference.sort().values)
    assert torch.equal(posterior_train, repeated_train)
    assert torch.equal(posterior_val, repeated_val)
    assert torch.bincount(
        targets[posterior_train], minlength=2
    ).tolist() == [8, 8]
    assert torch.bincount(
        targets[posterior_val], minlength=2
    ).tolist() == [5, 5]
    assert torch.equal(
        torch.cat([posterior_train, posterior_val]).sort().values,
        validated_calibration,
    )


def test_reference_only_lineage_falls_back_to_finalized_best_metadata():
    targets = torch.tensor([0] * 20 + [1] * 20)
    reference = list(range(7)) + list(range(20, 27))
    calibration = list(range(7, 20)) + list(range(27, 40))
    common = {
        "stage": "image_metric_vision_lora",
        "dataset": "fgvc-aircraft",
        "clip_model": "clip",
        "hierarchy": "hierarchy.json",
        "id_split": "id.csv",
        "metric_proxy_classes": ["zero", "one"],
        "args": {
            "experiment_name": "reference-only-seed0",
            "seed": 0,
            "reference_only_training": True,
        },
    }
    support = {**common, "metrics": {"train_history": []}}
    metadata = {
        **common,
        "metrics": {
            "training_split": {
                "reference_only": True,
                "reference_fraction": 0.35,
                "seed": 0,
                "reference_samples": len(reference),
                "calibration_samples": len(calibration),
                "reference_indices": reference,
                "calibration_indices": calibration,
            }
        },
    }
    partitions, lineage = reference_only_partitions_from_checkpoints(
        support,
        targets,
        expected_seed=0,
        metadata_checkpoint=metadata,
    )
    assert lineage["training_split_source"] == "split_metadata_checkpoint"
    assert int(partitions["reference"].numel()) == 14
    assert int(partitions["original_calibration"].numel()) == 26
    assert int(partitions["posterior_train"].numel()) == 16
    assert int(partitions["posterior_val"].numel()) == 10


def test_reference_only_lineage_rejects_a_different_run():
    targets = torch.tensor([0] * 20 + [1] * 20)
    common = {
        "stage": "image_metric_vision_lora",
        "dataset": "fgvc-aircraft",
        "clip_model": "clip",
        "hierarchy": "hierarchy.json",
        "id_split": "id.csv",
        "metric_proxy_classes": ["zero", "one"],
        "metrics": {},
        "args": {
            "experiment_name": "seed0",
            "seed": 0,
            "reference_only_training": True,
        },
    }
    metadata = {
        **common,
        "args": {**common["args"], "experiment_name": "seed1"},
    }
    try:
        reference_only_partitions_from_checkpoints(
            common,
            targets,
            expected_seed=0,
            metadata_checkpoint=metadata,
        )
    except ValueError as error:
        assert "different run" in str(error)
    else:
        raise AssertionError("Different checkpoint lineage was accepted")


def test_support_evidence_increases_when_support_falls():
    route = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    support = torch.tensor([[0.8, 0.2], [0.05, 0.02]])
    evidence = support_evidence(support, route)
    assert evidence[1, 0] > evidence[0, 0]
    assert torch.allclose(evidence[:, 1], torch.ones(2))


def test_posterior_is_monotone_in_nonconformity():
    model = SharedSupportPosterior(use_entropy=False)
    logits = model(torch.tensor([[0.1, 0.0], [2.0, 0.0]]))
    assert logits[1] > logits[0]


def test_masked_episode_has_unknown_target_and_terminal_prior_weight():
    hierarchy = toy_hierarchy()
    leaf_nodes = ["a1"]
    support = {
        "a1": torch.tensor([0.9]),
        "a2": torch.tensor([0.1]),
        "b1": torch.tensor([0.1]),
        "b2": torch.tensor([0.1]),
    }
    routes = {
        "root": torch.tensor([[0.9, 0.1]]),
        "p-a": torch.tensor([[0.8, 0.2]]),
        "p-b": torch.tensor([[0.5, 0.5]]),
    }
    episodes = build_support_episodes(
        hierarchy, torch.tensor([0]), leaf_nodes, support, routes
    )
    assert episodes.targets.tolist() == [0.0, 1.0]
    assert episodes.weights.tolist() == [1.0, 0.5]
    assert episodes.evidence[1, 0] > episodes.evidence[0, 0]


def test_recursive_posterior_sums_to_one():
    hierarchy = toy_hierarchy()
    routes = {
        "root": torch.tensor([[0.6, 0.4]]),
        "p-a": torch.tensor([[0.7, 0.3]]),
        "p-b": torch.tensor([[0.2, 0.8]]),
    }
    unknown = {
        "p-a": torch.tensor([0.25]),
        "p-b": torch.tensor([0.5]),
    }
    terminal = probabilistic_terminal_distribution(
        hierarchy, routes, unknown
    )
    assert torch.allclose(terminal.sum(dim=1), torch.ones(1))
    assert torch.isclose(
        terminal[0, hierarchy.id_node_list.index("p-a")],
        torch.tensor(0.15),
    )


def test_masked_vmf_energy_removes_true_child_bank():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "a2", "b1", "b2"),
        prototypes=torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]),
        node_prototype_indices={
            "a1": torch.tensor([0]),
            "a2": torch.tensor([1]),
            "b1": torch.tensor([2]),
            "b2": torch.tensor([3]),
            "p-a": torch.tensor([0, 1]),
            "p-b": torch.tensor([2, 3]),
            "root": torch.tensor([0, 1, 2, 3]),
        },
        node_calibration_scores={},
        prototype_calibration_scores=tuple(),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    episodes = build_energy_episodes(
        hierarchy,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([0]),
        ["a1"],
        calibration,
        weighting="uniform_terminal",
    )
    model = SharedMaskedSupportLikelihood(
        initial_concentration=5.0,
        initial_base_energy=2.5,
    )
    logits = model(
        episodes.child_similarities,
        episodes.prototype_mask,
        episodes.child_mask,
    )
    assert logits[1] > logits[0]
    assert episodes.targets.tolist() == [0.0, 1.0]


def test_vmf_likelihood_has_only_two_global_parameters():
    model = SharedMaskedSupportLikelihood()
    assert set(dict(model.named_parameters())) == {
        "raw_concentration",
        "base_energy",
    }


def test_categorical_likelihood_targets_child_then_unknown():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "a2", "b1", "b2"),
        prototypes=torch.eye(4),
        node_prototype_indices={
            "a1": torch.tensor([0]),
            "a2": torch.tensor([1]),
            "b1": torch.tensor([2]),
            "b2": torch.tensor([3]),
            "p-a": torch.tensor([0, 1]),
            "p-b": torch.tensor([2, 3]),
            "root": torch.tensor([0, 1, 2, 3]),
        },
        node_calibration_scores={},
        prototype_calibration_scores=tuple(),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    episodes = build_energy_episodes(
        hierarchy,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([0]),
        ["a1"],
        calibration,
    )
    assert categorical_episode_targets(episodes).tolist() == [0, 2]
    model = SharedMaskedCategoricalLikelihood(
        initial_concentration=5.0,
        initial_base_energy=2.5,
    )
    logits = model.categorical_logits(
        episodes.child_similarities,
        episodes.prototype_mask,
        episodes.child_mask,
    )
    assert logits.shape == (2, 3)
    assert torch.isneginf(logits[1, 0])
    assert logits[0, 0] > logits[0, -1]
    assert logits[1, -1] > logits[1, 1]


def test_categorical_mixture_conditionals_form_normalized_terminal():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "a2", "b1", "b2"),
        prototypes=torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ]),
        node_prototype_indices={
            "a1": torch.tensor([0]),
            "a2": torch.tensor([1]),
            "b1": torch.tensor([2]),
            "b2": torch.tensor([3]),
            "p-a": torch.tensor([0, 1]),
            "p-b": torch.tensor([2, 3]),
            "root": torch.tensor([0, 1, 2, 3]),
        },
        node_calibration_scores={},
        prototype_calibration_scores=tuple(),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    model = SharedMaskedCategoricalLikelihood()
    routes, unknown = mixture_conditionals_by_parent(
        model,
        hierarchy,
        torch.tensor([[1.0, 0.0]]),
        calibration,
    )
    assert "root" not in unknown
    assert set(unknown) == {"p-a", "p-b"}
    assert all(
        torch.allclose(value.sum(dim=1), torch.ones(1))
        for value in routes.values()
    )
    terminal = probabilistic_terminal_distribution(
        hierarchy, routes, unknown
    )
    assert torch.allclose(terminal.sum(dim=1), torch.ones(1))


def test_prior_corrected_product_counts_uniform_prior_once():
    hierarchy = toy_hierarchy()
    prior = 1.0 / 3.0
    left = {
        "p-a": torch.tensor([prior]),
        "p-b": torch.tensor([prior]),
    }
    combined = prior_corrected_product_unknown(
        hierarchy,
        left,
        left,
        prior_mode="uniform_terminal",
    )
    assert torch.allclose(combined["p-a"], torch.tensor([prior]))
    paired = {
        "p-a": torch.tensor([0.5]),
        "p-b": torch.tensor([0.5]),
    }
    paired_combined = prior_corrected_product_unknown(
        hierarchy,
        paired,
        paired,
        prior_mode="paired_view",
    )
    assert torch.allclose(paired_combined["p-b"], torch.tensor([0.5]))


def test_global_leaf_episodes_use_structural_terminal_prior():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "a2", "b1", "b2"),
        prototypes=torch.eye(4),
        node_prototype_indices={
            "a1": torch.tensor([0]),
            "a2": torch.tensor([1]),
            "b1": torch.tensor([2]),
            "b2": torch.tensor([3]),
        },
        node_calibration_scores={},
        prototype_calibration_scores=tuple(),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    episodes = build_global_leaf_energy_episodes(
        hierarchy,
        torch.eye(4),
        torch.arange(4),
        ["a1", "a2", "b1", "b2"],
        calibration,
    )
    assert episodes.child_mask.shape == (8, 4)
    assert episodes.masked.tolist() == [
        False, True, False, True, False, True, False, True
    ]
    assert categorical_episode_targets(episodes).tolist() == [
        0, 4, 1, 4, 2, 4, 3, 4
    ]
    # Two non-root internal terminals among four leaves imply prior 2/6.
    masked_mass = episodes.weights[episodes.masked].sum()
    full_mass = episodes.weights[~episodes.masked].sum()
    assert torch.isclose(masked_mass, torch.tensor(2.0 / 6.0))
    assert torch.isclose(full_mass, torch.tensor(4.0 / 6.0))
    model = SharedMaskedCategoricalLikelihood()
    unknown = global_unknown_probability(
        model, hierarchy, torch.eye(4), calibration
    )
    assert unknown.shape == (4,)


def test_latent_knownness_distribution_has_requested_partition_mass():
    hierarchy = toy_hierarchy()
    leaf = torch.zeros(1, len(hierarchy.id_node_list))
    leaf[0, hierarchy.id_node_list.index("a1")] = 0.75
    leaf[0, hierarchy.id_node_list.index("b1")] = 0.25
    parent = torch.zeros_like(leaf)
    parent[0, hierarchy.id_node_list.index("p-a")] = 0.2
    parent[0, hierarchy.id_node_list.index("p-b")] = 0.8
    terminal = latent_knownness_terminal_distribution(
        hierarchy, torch.tensor([0.3]), leaf, parent
    )
    internal = torch.tensor([
        hierarchy.id_node_list.index("p-a"),
        hierarchy.id_node_list.index("p-b"),
    ])
    assert torch.allclose(terminal.sum(dim=1), torch.ones(1))
    assert torch.isclose(terminal[:, internal].sum(), torch.tensor(0.3))


def test_masked_categorical_terminal_has_zero_hidden_subtree_mass():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "a2", "b1", "b2"),
        prototypes=torch.tensor([
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ]),
        node_prototype_indices={
            "a1": torch.tensor([0]),
            "a2": torch.tensor([1]),
            "b1": torch.tensor([2]),
            "b2": torch.tensor([3]),
            "p-a": torch.tensor([0, 1]),
            "p-b": torch.tensor([2, 3]),
            "root": torch.tensor([0, 1, 2, 3]),
        },
        node_calibration_scores={},
        prototype_calibration_scores=tuple(),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    masked = masked_subtree_terminal_distribution(
        SharedMaskedCategoricalLikelihood(),
        hierarchy,
        torch.tensor([[1.0, 0.0]]),
        ["a1"],
        calibration,
    )
    assert masked["targets"].tolist() == [
        hierarchy.id_node_list.index("p-a")
    ]
    assert torch.allclose(
        masked["terminal"].sum(dim=1), torch.ones(1)
    )
    assert masked["max_normalization_error"] <= 1e-6
    assert masked["max_masked_subtree_mass"] <= 1e-7
    assert torch.isclose(
        masked["terminal"][
            0, hierarchy.id_node_list.index("a1")
        ],
        torch.tensor(0.0),
    )
