from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import base_recipe_signature, resolve_bgs_options
from sacil.hierarchy import HierarchyTree
from sacil.hierarchy.tree import TreeNode
from sacil.methods import (
    BOUNDARY,
    INSIDE,
    OUTSIDE,
    BoundaryGraphSurgeryLoss,
    BoundaryGraphSurgeryReference,
    bgs_insertion_loss,
    canonical_regions,
    endpoint_regions,
    negative_candidate_positions,
    pair_mask_summary,
    pair_types_and_weights,
    row_permuted_random_weights,
)


ROOT = Path(__file__).resolve().parents[1]


def _tree() -> HierarchyTree:
    nodes = {
        f"leaf:{i}": TreeNode(f"leaf:{i}", (i,), class_id=i)
        for i in range(4)
    }
    nodes.update(
        {
            "node:a": TreeNode("node:a", (0, 1), "leaf:0", "leaf:1"),
            "node:b": TreeNode("node:b", (2, 3), "leaf:2", "leaf:3"),
            "root": TreeNode(
                "root",
                (0, 1, 2, 3),
                "node:a",
                "node:b",
            ),
        }
    )
    return HierarchyTree(nodes, "root", (0, 1, 2, 3))


def test_canonical_regions_duplicate_nested_and_disjoint_invariants() -> None:
    tree = _tree()
    canonical, mapping = canonical_regions(
        tree,
        ["node:a", "node:a", "leaf:0", "node:b"],
    )
    assert canonical == ("node:a", "node:b")
    assert mapping["leaf:0"] == "node:a"
    assert set(tree.descendants(canonical[0])).isdisjoint(tree.descendants(canonical[1]))


def test_pair_partition_boundary_ancestor_and_stored_order() -> None:
    tree = _tree()
    sample, leaf = endpoint_regions(
        tree,
        ("node:a",),
        (0, 1, 2, 3),
        [(0,), (1,), (2,), (3,)],
    )
    _, internal = endpoint_regions(
        tree,
        ("node:a",),
        (0, 1, 2, 3),
        [(0, 1), (2, 3), (0, 1, 2, 3)],
    )
    types, weights = pair_types_and_weights(
        sample,
        leaf,
        inside_weight=0.1,
        boundary_weight=1.0,
        mask_mode="structured",
    )
    assert set(types.flatten().tolist()) == {INSIDE, BOUNDARY, OUTSIDE}
    assert types[0, 0] == INSIDE and types[0, 2] == BOUNDARY
    assert types[2, 2] == OUTSIDE and types[2, 0] == BOUNDARY
    internal_types, _ = pair_types_and_weights(
        sample,
        internal,
        inside_weight=0.1,
        boundary_weight=1.0,
        mask_mode="structured",
    )
    assert internal_types[0, 2] == BOUNDARY  # root ancestor crosses cut
    assert weights[0, 0] == pytest.approx(0.1)


def test_absolute_denominator_and_random_exact_row_budget() -> None:
    weights = torch.tensor([[0.1, 0.1, 1.0, 1.0], [1.0, 0.1, 1.0, 0.1]])
    randomized, seeds, permutations = row_permuted_random_weights(
        weights,
        (7, 9),
        experiment_seed=1,
        session_id=2,
        group="leaf",
    )
    assert len(set(seeds)) == 2 and len(permutations) == 2
    torch.testing.assert_close(weights.sort(1).values, randomized.sort(1).values)
    torch.testing.assert_close((1 - weights).sum(1), (1 - randomized).sum(1))
    constant_error = torch.ones_like(weights)
    absolute = (weights * constant_error).sum() / weights.numel()
    expected = 1.0 - (1 - weights).sum() / weights.numel()
    torch.testing.assert_close(absolute, expected)


def _reference(weight: float = 1.0) -> BoundaryGraphSurgeryReference:
    leaf = torch.eye(4)
    internal = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    internal = torch.nn.functional.normalize(internal, dim=1)
    return BoundaryGraphSurgeryReference(
        session_id=1,
        old_original_ids=(0, 1, 2, 3),
        new_original_ids=(4,),
        raw_branch_nodes=("node:a",),
        raw_branch_scores=torch.tensor([0.8]),
        canonical_nodes=("node:a",),
        raw_to_canonical={"node:a": "node:a"},
        tree_state=_tree().state_dict(),
        anchor_state={
            "leaf_class_ids": [0, 1, 2, 3],
            "leaf_anchors": leaf,
            "internal_node_ids": ["node:a", "node:b", "root"],
            "internal_anchors": internal,
            "root_id": "root",
        },
        leaf_pair_types=torch.zeros(4, 4, dtype=torch.int8),
        internal_pair_types=torch.zeros(4, 2, dtype=torch.int8),
        leaf_weights=torch.full((4, 4), weight),
        internal_weights=torch.full((4, 2), weight),
        random_seeds={},
        random_permutations={},
        incoming_teacher_prototypes=torch.randn(1, 4),
        negative_class_positions=((0, 1),),
        parent_thresholds=torch.tensor([0.7]),
        options={"spec_version": "bgs_v1"},
    )


def test_global_identity_group_mean_grad_and_checkpoint_roundtrip() -> None:
    reference = _reference(1.0)
    restored = BoundaryGraphSurgeryReference.from_state_dict(reference.state_dict())
    module = BoundaryGraphSurgeryLoss(restored)
    current = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4)
    loss, stats = module(current, teacher, torch.tensor([0,1,2]))
    manual_leaf = (
        (
            (
                torch.nn.functional.normalize(current, dim=1)
                @ torch.eye(4).T
            )
            - (
                torch.nn.functional.normalize(teacher, dim=1)
                @ torch.eye(4).T
            )
        )
        ** 2
    ).mean()
    torch.testing.assert_close(loss, (manual_leaf + stats["internal"]) / 2)
    loss.backward()
    assert current.grad is not None
    assert torch.equal(restored.leaf_weights, reference.leaf_weights)


def test_insertion_routes_new_only_and_frozen_artifacts() -> None:
    query = torch.randn(2,4,requires_grad=True)
    positives = torch.nn.functional.normalize(torch.randn(1,4),dim=1).requires_grad_()
    leaf = torch.eye(4).requires_grad_()
    internal = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        dim=1,
    ).requires_grad_()
    total, sep, parent, _ = bgs_insertion_loss(
        query,
        torch.zeros(2, dtype=torch.long),
        positives,
        leaf,
        internal,
        ((0, 1),),
        torch.tensor([0]),
        torch.tensor([0.5], requires_grad=True),
        temperature=0.1,
        separation_enabled=True,
        parent_weight=1.0,
    )
    total.backward()
    assert query.grad is not None
    assert positives.grad is None and leaf.grad is None and internal.grad is None
    zero, *_ = bgs_insertion_loss(
        query[:0],
        torch.empty(0, dtype=torch.long),
        positives,
        leaf,
        internal,
        ((0, 1),),
        torch.tensor([0]),
        torch.tensor([0.5]),
        temperature=0.1,
        separation_enabled=True,
        parent_weight=1.0,
    )
    assert float(zero) == 0.0 and sep.isfinite() and parent.isfinite()


def test_geometry_flags_control_the_executed_anchor_groups() -> None:
    reference = _reference(1.0)
    reference.options = {
        "geometry": {
            "use_leaf": False,
            "use_internal_without_root": True,
        }
    }
    module = BoundaryGraphSurgeryLoss(reference)
    current = torch.randn(3, 4, requires_grad=True)
    teacher = torch.randn(3, 4)
    loss, stats = module(current, teacher, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(loss, stats["internal"])
    assert float(stats["leaf"]) == 0.0


def test_random_pair_diagnostics_keep_semantic_types_separate() -> None:
    pair_types = torch.tensor(
        [[INSIDE, INSIDE, BOUNDARY, OUTSIDE]],
        dtype=torch.int8,
    )
    structured = torch.tensor([[0.1, 0.1, 1.0, 1.0]])
    randomized = structured[:, torch.tensor([2, 0, 3, 1])]
    summary = pair_mask_summary(
        pair_types,
        randomized,
        inside_weight=0.1,
        boundary_weight=1.0,
    )
    assert summary["inside"]["pair_count"] == 2
    assert summary["structured_weight_mismatch_count"] > 0
    assert summary["relaxation_deficit"] == pytest.approx(1.8)


def test_a5_and_a6_execute_only_the_requested_insertion_terms() -> None:
    query = torch.randn(3, 4, requires_grad=True)
    positives = torch.nn.functional.normalize(torch.randn(1, 4), dim=1)
    leaf = torch.eye(4)
    internal = torch.nn.functional.normalize(
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        dim=1,
    )
    arguments = (
        query,
        torch.zeros(3, dtype=torch.long),
        positives,
        leaf,
        internal,
        ((0, 1),),
        torch.tensor([0]),
        torch.tensor([0.5]),
    )
    a5_total, a5_separation, a5_parent, _ = bgs_insertion_loss(
        *arguments,
        temperature=0.1,
        separation_enabled=True,
        parent_weight=0.0,
    )
    a6_total, a6_separation, a6_parent, _ = bgs_insertion_loss(
        *arguments,
        temperature=0.1,
        separation_enabled=False,
        parent_weight=1.0,
    )
    torch.testing.assert_close(a5_total, a5_separation)
    assert a5_parent.isfinite()
    torch.testing.assert_close(a6_total, a6_parent)
    assert float(a6_separation) == 0.0


def test_a8_all_old_negative_scope_changes_the_candidate_pool() -> None:
    tree = _tree()
    local = negative_candidate_positions(
        tree,
        (0, 1, 2, 3),
        "node:a",
        "branch_local",
    )
    all_old = negative_candidate_positions(
        tree,
        (0, 1, 2, 3),
        "node:a",
        "all_old",
    )
    assert local == (0, 1)
    assert all_old == (0, 1, 2, 3)


def test_checkpoint_roundtrip_is_bitwise_on_a_fixed_cpu_probe() -> None:
    reference = _reference(0.1)
    restored = BoundaryGraphSurgeryReference.from_state_dict(
        reference.state_dict()
    )
    current = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
    )
    teacher = torch.tensor(
        [[1.5, 2.0, 2.5, 4.0], [3.5, 3.0, 2.0, 1.5]]
    )
    targets = torch.tensor([0, 3])
    before = BoundaryGraphSurgeryLoss(reference)(current, teacher, targets)[0]
    after = BoundaryGraphSurgeryLoss(restored)(current, teacher, targets)[0]
    assert torch.equal(before, after)
    assert restored.negative_class_positions == reference.negative_class_positions
    assert restored.raw_to_canonical == reference.raw_to_canonical


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
@pytest.mark.parametrize("row_mode", ["old_only", "new_only", "mixed"])
def test_finite_geometry_and_insertion_smoke(device: str, row_mode: str) -> None:
    reference = _reference(0.1)
    geometry = BoundaryGraphSurgeryLoss(reference).to(device)
    all_features = torch.randn(4, 4, device=device, requires_grad=True)
    if row_mode == "old_only":
        old_features = all_features
        new_features = all_features[:0]
    elif row_mode == "new_only":
        old_features = all_features[:0]
        new_features = all_features
    else:
        old_features = all_features[:2]
        new_features = all_features[2:]
    geometry_loss, _ = geometry(
        old_features,
        torch.randn_like(old_features),
        torch.arange(old_features.shape[0], device=device),
    )
    insertion_loss, *_ = bgs_insertion_loss(
        new_features,
        torch.zeros(new_features.shape[0], dtype=torch.long, device=device),
        torch.nn.functional.normalize(torch.randn(1, 4, device=device), dim=1),
        torch.eye(4, device=device),
        torch.nn.functional.normalize(
            torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=device),
            dim=1,
        ),
        ((0, 1),),
        torch.tensor([0], device=device),
        torch.tensor([0.5], device=device),
        temperature=0.1,
        separation_enabled=True,
        parent_weight=1.0,
    )
    total = geometry_loss + insertion_loss
    assert total.isfinite()
    total.backward()
    assert all_features.grad is not None
    assert torch.isfinite(all_features.grad).all()


@pytest.mark.parametrize(
    "name",
    [
        "icarl_bgs_a1_global.yaml",
        "icarl_bgs_a4.yaml",
        "icarl_bgs_a7.yaml",
        "icarl_bgs_a2_incident.yaml",
        "icarl_bgs_a3_random_pair.yaml",
        "icarl_bgs_a5_separation.yaml",
        "icarl_bgs_a6_parent.yaml",
        "icarl_bgs_a8_global_negatives.yaml",
    ],
)
def test_bgs_configs_are_explicit_and_base_compatible(name: str) -> None:
    control = load_config_tree(
        ROOT / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    candidate = load_config_tree(ROOT / "configs/explore/cifar100" / name)
    options = resolve_bgs_options("icarl", candidate["method"])
    assert options["enabled"] and options["geometry"]["lambda"] == 16.0
    assert candidate["output"]["directory"] == "outputs/explore/boundary_graph_surgery"
    assert base_recipe_signature(candidate) == base_recipe_signature(control)
    assert candidate["method"].get("geometry", {}).get("mode", "none") == "none"
    assert not candidate["method"].get("edge_topology", {}).get(
        "enabled", False
    )
    assert not candidate["method"].get("branch_masked_kd", {}).get(
        "enabled", False
    )
    assert not candidate["method"].get("selective_kd", {}).get(
        "enabled", False
    )


def test_a4_a5_a6_a7_a8_config_objectives_are_exact() -> None:
    names = {
        "a1": "icarl_bgs_a1_global.yaml",
        "a4": "icarl_bgs_a4.yaml",
        "a5": "icarl_bgs_a5_separation.yaml",
        "a6": "icarl_bgs_a6_parent.yaml",
        "a7": "icarl_bgs_a7.yaml",
        "a8": "icarl_bgs_a8_global_negatives.yaml",
    }
    options = {
        key: resolve_bgs_options(
            "icarl",
            load_config_tree(ROOT / "configs/explore/cifar100" / name)[
                "method"
            ],
        )
        for key, name in names.items()
    }
    assert options["a1"]["geometry"]["mask_mode"] == "global"
    assert options["a1"]["geometry"]["inside_weight"] == 1.0
    assert not options["a4"]["insertion"]["enabled"]
    assert options["a5"]["insertion"]["separation_enabled"]
    assert options["a5"]["insertion"]["parent_weight"] == 0.0
    assert not options["a6"]["insertion"]["separation_enabled"]
    assert options["a6"]["insertion"]["parent_weight"] == 1.0
    assert options["a7"]["insertion"]["separation_enabled"]
    assert options["a7"]["insertion"]["parent_weight"] == 1.0
    assert options["a8"]["insertion"]["negative_scope"] == "all_old"


def test_bgs_strict_validation_rejects_unsupported_v1_variants() -> None:
    config = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_bgs_a7.yaml"
    )
    config["method"]["boundary_graph_surgery"][
        "branches_per_new_class"
    ] = 2
    with pytest.raises(ValueError, match="exactly one branch"):
        resolve_bgs_options("icarl", config["method"])
