from __future__ import annotations

from types import SimpleNamespace

import torch

from negzerohoc.hierarchical_support import (
    conformal_p_values,
    expected_hierarchy_distance_predictions,
    factorized_terminal_probabilities,
    global_gate_route_stop_predictions,
    mondrian_support_p_values,
    nearest_support_prototype_predictions,
    stratified_reference_calibration_split,
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


def test_stratified_split_is_disjoint_complete_and_reproducible():
    targets = torch.tensor([0] * 5 + [1] * 5)
    first = stratified_reference_calibration_split(
        targets, reference_fraction=0.6, seed=7
    )
    second = stratified_reference_calibration_split(
        targets, reference_fraction=0.6, seed=7
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert set(first[0].tolist()).isdisjoint(first[1].tolist())
    assert sorted(first[0].tolist() + first[1].tolist()) == list(range(10))
    assert torch.bincount(targets[first[0]]).tolist() == [3, 3]
    assert torch.bincount(targets[first[1]]).tolist() == [2, 2]


def test_conformal_p_value_increases_with_support():
    calibration = torch.tensor([0.1, 0.2, 0.3, 0.4])
    values = conformal_p_values(
        torch.tensor([0.05, 0.25, 0.50]), calibration
    )
    assert torch.allclose(values, torch.tensor([0.2, 0.6, 1.0]))


def test_factorized_distribution_is_normalized_and_stops_at_parent():
    hierarchy = toy_hierarchy()
    routes = {
        "root": torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
        "p-a": torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
        "p-b": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    }
    support = {
        "p-a": torch.tensor([0.9, 0.01]),
        "p-b": torch.tensor([0.9, 0.9]),
    }
    probabilities = factorized_terminal_probabilities(
        hierarchy, routes, support, alpha=0.05, gate="hard"
    )
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    assert float(probabilities[0, hierarchy.id_node_list.index("p-a")]) == 0.0
    assert torch.isclose(
        probabilities[1, hierarchy.id_node_list.index("p-a")],
        torch.tensor(0.9),
    )


def test_global_gate_stops_at_first_unsupported_parent():
    hierarchy = toy_hierarchy()
    leaf_predictions = torch.tensor([
        hierarchy.id_node_list.index("a1"),
        hierarchy.id_node_list.index("a1"),
    ])
    support = {
        "root": torch.tensor([0.9, 0.01]),
        "p-a": torch.tensor([0.9, 0.01]),
        "p-b": torch.tensor([0.9, 0.9]),
    }
    predictions, diagnostics = global_gate_route_stop_predictions(
        hierarchy,
        leaf_predictions,
        support,
        alpha=0.05,
        localizer="first_unsupported",
    )
    assert predictions.tolist() == [
        hierarchy.id_node_list.index("a1"),
        hierarchy.id_node_list.index("p-a"),
    ]
    assert diagnostics["rejected"] == 1


def test_expected_distance_decoder_uses_candidate_rows():
    probabilities = torch.tensor([[0.1, 0.9]])
    distances = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    predictions = expected_hierarchy_distance_predictions(
        probabilities, distances
    )
    assert predictions.tolist() == [1]


def test_nearest_support_prototype_maps_to_hierarchy_node():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    hierarchy = toy_hierarchy()
    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "b1"),
        prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        node_prototype_indices={},
        node_calibration_scores={},
        prototype_calibration_scores=(
            torch.tensor([0.7, 0.8]),
            torch.tensor([0.7, 0.8]),
        ),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    predictions = nearest_support_prototype_predictions(
        torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
        calibration,
        hierarchy,
    )
    assert predictions.tolist() == [
        hierarchy.id_node_list.index("a1"),
        hierarchy.id_node_list.index("b1"),
    ]


def test_mondrian_support_uses_nearest_class_distribution():
    from negzerohoc.hierarchical_support import (
        HierarchicalSupportCalibration,
    )

    calibration = HierarchicalSupportCalibration(
        prototype_nodes=("a1", "b1"),
        prototypes=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        node_prototype_indices={},
        node_calibration_scores={},
        prototype_calibration_scores=(
            torch.tensor([0.8, 0.9]),
            torch.tensor([0.2, 0.3]),
        ),
        reference_indices=torch.tensor([], dtype=torch.long),
        calibration_indices=torch.tensor([], dtype=torch.long),
    )
    values = mondrian_support_p_values(
        torch.tensor([[0.85, 0.5268], [0.5268, 0.85]]),
        calibration,
    )
    assert torch.allclose(values, torch.tensor([2.0 / 3.0, 1.0]))
