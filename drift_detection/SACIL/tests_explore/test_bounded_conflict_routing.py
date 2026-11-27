from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import resolve_bgs_options
from sacil.hierarchy import HierarchyTree
from sacil.hierarchy.tree import TreeNode
from sacil.methods import (
    bounded_conflict_union_diagnostics,
    effective_bounded_branch_cap,
    nearest_leaf_bounded_ancestor_branches,
)


ROOT = Path(__file__).resolve().parents[1]


def _unbalanced_tree() -> HierarchyTree:
    nodes = {
        f"leaf:{index}": TreeNode(
            f"leaf:{index}", (index,), class_id=index
        )
        for index in range(8)
    }
    nodes.update(
        {
            "node:01": TreeNode(
                "node:01", (0, 1), "leaf:0", "leaf:1"
            ),
            "node:012": TreeNode(
                "node:012", (0, 1, 2), "node:01", "leaf:2"
            ),
            "node:0123": TreeNode(
                "node:0123", (0, 1, 2, 3), "node:012", "leaf:3"
            ),
            "node:45": TreeNode(
                "node:45", (4, 5), "leaf:4", "leaf:5"
            ),
            "node:456": TreeNode(
                "node:456", (4, 5, 6), "node:45", "leaf:6"
            ),
            "node:4567": TreeNode(
                "node:4567", (4, 5, 6, 7), "node:456", "leaf:7"
            ),
            "root": TreeNode(
                "root",
                tuple(range(8)),
                "node:0123",
                "node:4567",
            ),
        }
    )
    return HierarchyTree(nodes, "root", tuple(range(8)))


def test_effective_cap_enforces_s1_union_upper_bound() -> None:
    cap = effective_bounded_branch_cap(8, 0.60, 50, 5)
    assert cap == 6
    assert cap * 5 / 50 <= 0.60


def test_bounded_selection_is_deterministic_and_handles_leaf_only_case() -> None:
    tree = _unbalanced_tree()
    old_prototypes = torch.eye(8)
    incoming = old_prototypes[[0, 3, 4]].clone()
    first = nearest_leaf_bounded_ancestor_branches(
        tree,
        tuple(range(8)),
        old_prototypes,
        incoming,
        max_branch_leaves=3,
    )
    second = nearest_leaf_bounded_ancestor_branches(
        tree,
        tuple(range(8)),
        old_prototypes,
        incoming,
        max_branch_leaves=3,
    )
    assert first.nearest_leaf_original_ids == (0, 3, 4)
    assert first.selected_branch_nodes == (
        "node:012",
        "leaf:3",
        "node:456",
    )
    assert first.selected_branch_leaf_counts == (3, 1, 3)
    assert max(first.selected_branch_leaf_counts) <= 3
    assert first.primary_internal_nodes == (
        "node:012",
        "node:0123",
        "node:456",
    )
    assert first.selected_branch_nodes == second.selected_branch_nodes
    assert first.primary_internal_nodes == second.primary_internal_nodes
    assert torch.equal(first.nearest_leaf_scores, second.nearest_leaf_scores)


def test_realized_union_coverage_is_checked_after_selection() -> None:
    tree = _unbalanced_tree()
    nodes = ("node:012", "leaf:3", "node:456")
    diagnostics = bounded_conflict_union_diagnostics(
        tree,
        nodes,
        tuple(range(8)),
        max_conflict_leaf_coverage=0.90,
    )
    assert diagnostics == {
        "conflict_union_leaf_count": 7,
        "conflict_union_coverage": 0.875,
    }
    with pytest.raises(AssertionError, match="coverage exceeds"):
        bounded_conflict_union_diagnostics(
            tree,
            nodes,
            tuple(range(8)),
            max_conflict_leaf_coverage=0.80,
        )


def test_bounded_routed_config_is_explicit_and_session_cap_is_six() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_routed_bounded_bgs.yaml"
    )
    options = resolve_bgs_options("icarl", config["method"])
    assert options["branch_source"] == "nearest_leaf_bounded_ancestor"
    assert options["max_branch_leaves"] == 8
    assert options["max_conflict_leaf_coverage"] == pytest.approx(0.60)
    assert effective_bounded_branch_cap(
        options["max_branch_leaves"],
        options["max_conflict_leaf_coverage"],
        50,
        5,
    ) == 6
    assert options["geometry"]["mask_mode"] == "structured"
    assert not options["insertion"]["enabled"]
    routing = config["method"]["feature_cosine_distillation"][
        "hierarchy_routing"
    ]
    assert routing["old_conflict_weight"] == pytest.approx(0.1)
    assert routing["old_outside_weight"] == pytest.approx(1.0)
    assert routing["new_weight"] == pytest.approx(0.1)


def test_legacy_i2_bgs_resolver_contract_is_exactly_unchanged() -> None:
    config = load_config_tree(
        ROOT
        / "configs/explore/cifar100/icarl_feature_cosine_lucir_cosinehead_bgs.yaml"
    )
    options = resolve_bgs_options("icarl", config["method"])
    assert options == {
        "enabled": True,
        "spec_version": "bgs_v1",
        "branch_source": "i2_teacher_internal_top1",
        "branches_per_new_class": 1,
        "geometry": {
            "lambda": 16.0,
            "inside_weight": 0.1,
            "boundary_weight": 1.0,
            "mask_mode": "structured",
            "use_leaf": True,
            "use_internal_without_root": True,
            "denominator": "old_sample_count_x_anchor_count",
            "objective": "fixed_anchor_mse",
        },
        "insertion": {
            "enabled": False,
            "lambda": 0.0,
            "negatives_per_class": 5,
            "temperature": 0.1,
            "prototype_refresh": "epoch_start_full_new_unaugmented",
            "separation_enabled": False,
            "parent_weight": 0.0,
            "parent_slack": 0.05,
            "negative_scope": "branch_local",
            "gradient_projection": False,
            "projection_epsilon": 1e-12,
        },
    }
