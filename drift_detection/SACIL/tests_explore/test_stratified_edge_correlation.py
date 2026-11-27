from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    UnifiedTable1Trainer,
    base_recipe_signature,
    resolve_edge_topology_options,
)
from sacil.methods import (
    STRATIFIED_EDGE_GROUP_NAMES,
    StratifiedHierarchicalEdgeCorrelationLoss,
    StratifiedHierarchicalEdgeReference,
    conflict_union_membership,
    pairwise_cosine_edge_vector,
    stratified_edge_group_ids,
    weighted_global_edge_correlation_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_strata_are_disjoint_complete_and_in_upper_triangle_order() -> None:
    # Representative order: [outside, inside, inside, outside]. Edges are
    # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    membership = torch.tensor([False, True, True, False])
    groups = stratified_edge_group_ids(membership)
    torch.testing.assert_close(groups, torch.tensor([1, 1, 0, 2, 1, 1]))
    masks = tuple(groups == group_id for group_id in range(3))
    assert torch.stack(masks).sum(0).eq(1).all()
    assert sum(int(mask.sum()) for mask in masks) == 6


def test_overlapping_subtrees_form_order_independent_union() -> None:
    class_ids = (0, 0, 1, 2, 3, 4)
    first = conflict_union_membership(
        class_ids, ((1, 2), (2, 3), (3, 3))
    )
    second = conflict_union_membership(
        class_ids, ((3, 2), (2, 1), (1, 2))
    )
    expected = torch.tensor([False, False, True, True, True, False])
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


def test_stratified_loss_matches_three_independent_pearsons() -> None:
    torch.manual_seed(31)
    reference_features = torch.randn(6, 5)
    current_features = (
        reference_features + 0.35 * torch.randn_like(reference_features)
    ).requires_grad_()
    reference_edges = pairwise_cosine_edge_vector(reference_features)
    group_ids = stratified_edge_group_ids(
        torch.tensor([False, False, False, True, True, True])
    )
    module = StratifiedHierarchicalEdgeCorrelationLoss(
        reference_edges,
        group_ids,
        beta_boundary=1.0,
        gamma_conflict=0.1,
    )
    result = module(current_features)
    current_edges = pairwise_cosine_edge_vector(current_features)
    expected_groups = torch.stack(
        [
            weighted_global_edge_correlation_loss(
                current_edges,
                reference_edges,
                (group_ids == group_id).float(),
            )
            for group_id in range(3)
        ]
    )
    torch.testing.assert_close(result.group_losses, expected_groups)
    torch.testing.assert_close(
        result.loss,
        expected_groups[0] + expected_groups[1] + 0.1 * expected_groups[2],
    )
    assert result.active_groups.tolist() == [True, True, True]
    result.loss.backward()
    assert current_features.grad is not None
    assert torch.isfinite(current_features.grad).all()
    assert float(current_features.grad.abs().sum()) > 0.0


def test_empty_and_low_count_groups_are_explicitly_guarded() -> None:
    torch.manual_seed(32)
    reference_features = torch.randn(4, 5)
    group_ids = stratified_edge_group_ids(torch.zeros(4, dtype=torch.bool))
    module = StratifiedHierarchicalEdgeCorrelationLoss(
        pairwise_cosine_edge_vector(reference_features), group_ids
    )
    result = module(reference_features + 0.2 * torch.randn_like(reference_features))
    assert result.group_counts.tolist() == [6, 0, 0]
    assert result.reference_active_groups.tolist() == [True, False, False]
    assert result.active_groups.tolist() == [True, False, False]
    assert float(result.group_losses[1]) == 0.0
    assert float(result.group_losses[2]) == 0.0

    # Valid partition with only one stable and one conflict edge. Pearson is
    # undefined for those groups, so both must be inactive rather than NaN.
    small_groups = stratified_edge_group_ids(
        torch.tensor([False, True, True, False])
    )
    guarded = StratifiedHierarchicalEdgeCorrelationLoss(
        torch.linspace(-0.8, 0.7, 6), small_groups
    )(torch.randn(4, 5))
    assert guarded.group_counts.tolist() == [1, 4, 1]
    assert guarded.reference_active_groups.tolist() == [False, True, False]
    assert torch.isfinite(guarded.loss)


def test_zero_current_variance_is_guarded_and_reported() -> None:
    torch.manual_seed(33)
    reference_features = torch.randn(6, 4)
    groups = stratified_edge_group_ids(
        torch.tensor([False, False, False, True, True, True])
    )
    module = StratifiedHierarchicalEdgeCorrelationLoss(
        pairwise_cosine_edge_vector(reference_features), groups
    )
    collapsed = torch.ones(6, 4, requires_grad=True)
    result = module(collapsed)
    assert result.reference_active_groups.tolist() == [True, True, True]
    assert result.current_active_groups.tolist() == [False, False, False]
    assert result.active_groups.tolist() == [False, False, False]
    assert float(result.loss) == 0.0
    result.loss.backward()
    assert collapsed.grad is not None
    assert torch.isfinite(collapsed.grad).all()


def test_stratified_reference_round_trip_preserves_partition() -> None:
    reference = StratifiedHierarchicalEdgeReference(
        session_id=1,
        representatives_per_class=2,
        representative_indices=(10, 11, 20, 21),
        representative_class_ids=(0, 0, 1, 1),
        reference_edges=torch.linspace(-0.8, 0.7, 6),
        edge_group_ids=torch.tensor([1, 1, 0, 2, 1, 1]),
        beta_boundary=1.0,
        gamma_conflict=0.1,
        conflict_node_ids=("n1", "n3"),
    )
    restored = StratifiedHierarchicalEdgeReference.from_state_dict(
        reference.state_dict()
    )
    assert restored.group_counts == {
        "stable": 1,
        "boundary": 4,
        "conflict": 1,
    }
    torch.testing.assert_close(restored.edge_group_ids, reference.edge_group_ids)
    assert isinstance(
        restored.loss_module(), StratifiedHierarchicalEdgeCorrelationLoss
    )


class _TinyFeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images)


def test_trainer_component_exposes_group_diagnostics() -> None:
    torch.manual_seed(34)
    trainer = object.__new__(UnifiedTable1Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = _TinyFeatureModel()
    images = torch.randn(6, 3)
    with torch.no_grad():
        reference_features = (
            trainer.model.extract_features(images)
            + 0.15 * torch.randn(6, 4)
        )
    groups = stratified_edge_group_ids(
        torch.tensor([False, False, False, True, True, True])
    )
    trainer.edge_topology_options = {
        "objective": "stratified_hierarchical_edge_correlation",
        "lambda_edge": 0.5,
        "update_interval_steps": 10,
    }
    trainer._edge_topology_images = images
    trainer._edge_topology_loss = StratifiedHierarchicalEdgeCorrelationLoss(
        pairwise_cosine_edge_vector(reference_features), groups
    )
    value = trainer._edge_topology_component(0)
    assert value is not None and value.requires_grad
    assert trainer._last_edge_topology_stats["stable_edge_count"] == 3
    assert trainer._last_edge_topology_stats["boundary_edge_count"] == 9
    assert trainer._last_edge_topology_stats["conflict_edge_count"] == 3
    assert trainer._edge_topology_component(1) is None
    assert trainer._last_edge_topology_stats == {}


def test_p2_config_and_option_validation() -> None:
    control = load_config_tree(
        PROJECT_ROOT
        / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(
        PROJECT_ROOT
        / "configs/explore/cifar100/"
        "icarl_edgecorr_stratified_r20_lambda05.yaml"
    )
    options = resolve_edge_topology_options("icarl", candidate["method"])
    assert options == {
        "enabled": True,
        "objective": "stratified_hierarchical_edge_correlation",
        "representatives_per_class": 20,
        "lambda_edge": 0.5,
        "edge_weighting": "stratified_hierarchy",
        "update_interval_steps": 10,
        "min_edge_weight": 0.1,
        "conflict_branches_per_new_class": 1,
        "beta_boundary": 1.0,
        "gamma_conflict": 0.1,
    }
    assert base_recipe_signature(candidate) == base_recipe_signature(control)

    bad = {
        "edge_topology": {
            "enabled": True,
            "objective": "stratified_hierarchical_edge_correlation",
            "edge_weighting": "global",
        }
    }
    with pytest.raises(ValueError, match="stratified_hierarchy"):
        resolve_edge_topology_options("icarl", bad)
    bad["edge_topology"]["edge_weighting"] = "stratified_hierarchy"
    bad["edge_topology"]["gamma_conflict"] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        resolve_edge_topology_options("icarl", bad)


def test_group_name_contract_is_stable() -> None:
    assert STRATIFIED_EDGE_GROUP_NAMES == ("stable", "boundary", "conflict")
