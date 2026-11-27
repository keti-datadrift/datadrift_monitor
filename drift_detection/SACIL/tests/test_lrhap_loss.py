from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank
from sacil.hierarchy import GriffinPeronaGreedy
from sacil.methods import (
    AnchorGeometryLoss,
    global_preservation_weights,
    hierarchical_triplet_rank_loss,
    inverse_angular_dispersion_reliability,
    weighted_anchor_correlation_loss,
)


def _bank() -> tuple[HierarchicalAnchorBank, object]:
    prototypes = F.normalize(torch.randn(4, 8), dim=1)
    affinity = torch.tensor(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.1, 0.1],
            [0.1, 0.1, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ]
    )
    tree = GriffinPeronaGreedy().build((0, 1, 2, 3), affinity)
    bank = HierarchicalAnchorBank.from_tree(
        PrototypeBank((0, 1, 2, 3), prototypes), tree
    )
    return bank, tree


def _loss() -> AnchorGeometryLoss:
    bank, _ = _bank()
    weights = global_preservation_weights(bank)
    return AnchorGeometryLoss(
        bank, weights.leaf_weights, weights.internal_weights
    )


def test_geometry_loss_is_zero_for_identical_features() -> None:
    loss = _loss()
    current = torch.randn(5, 8, requires_grad=True)
    reference = current.detach().clone()
    value = loss(current, reference)
    assert torch.allclose(value, torch.tensor(0.0), atol=1e-8)
    value.backward()
    assert current.grad is not None


def test_geometry_loss_increases_after_feature_perturbation() -> None:
    loss = _loss()
    reference = torch.randn(5, 8)
    identical = loss(reference.clone(), reference)
    perturbed = loss(reference + torch.randn_like(reference), reference)
    assert perturbed > identical


def test_absolute_weight_normalization_does_not_amplify_other_anchors() -> None:
    bank, _ = _bank()
    weights = torch.tensor([1.0, 0.5, 0.25, 0.0])
    reference = torch.randn(5, bank.feature_dim)
    current = reference + torch.randn_like(reference)
    squared_error = (
        F.normalize(current, dim=1)
        @ F.normalize(bank.leaf_anchors, dim=1).T
        - F.normalize(reference, dim=1)
        @ F.normalize(bank.leaf_anchors, dim=1).T
    ).square()
    relative = AnchorGeometryLoss(
        bank,
        weights,
        torch.ones(bank.internal_without_root()[1].shape[0]),
        use_internal_anchors=False,
    )(current, reference)
    absolute = AnchorGeometryLoss(
        bank,
        weights,
        torch.ones(bank.internal_without_root()[1].shape[0]),
        use_internal_anchors=False,
        weight_normalization="anchor_count",
    )(current, reference)
    expected_relative = (squared_error * weights).sum() / (
        current.shape[0] * weights.sum()
    )
    expected_absolute = (squared_error * weights).sum() / (
        current.shape[0] * weights.numel()
    )
    torch.testing.assert_close(relative, expected_relative)
    torch.testing.assert_close(absolute, expected_absolute)
    assert absolute < relative


def test_empty_old_batch_is_safe() -> None:
    loss = _loss()
    empty = torch.empty(0, 8, requires_grad=True)
    value = loss(empty, empty.detach())
    assert value.item() == 0.0


def test_weighted_correlation_ignores_positive_affine_rescaling() -> None:
    reference = torch.tensor([[0.1, 0.4, 0.9], [0.8, 0.2, -0.1]])
    current = (2.5 * reference + 0.3).requires_grad_()
    weights = torch.tensor([1.0, 0.5, 0.2])
    value = weighted_anchor_correlation_loss(
        current, reference, weights
    )
    assert value < 1e-7
    value.backward()
    assert current.grad is not None


def test_weighted_correlation_penalizes_reordered_anchors() -> None:
    reference = torch.tensor([[0.1, 0.4, 0.9]])
    reordered = torch.tensor([[0.9, 0.4, 0.1]], requires_grad=True)
    value = weighted_anchor_correlation_loss(
        reordered, reference, torch.ones(3)
    )
    assert value > 1.0


def test_hierarchical_triplet_rank_is_zero_for_teacher_order() -> None:
    reference = torch.tensor([[0.9, 0.5, 0.1]])
    current = reference.clone().requires_grad_()
    value = hierarchical_triplet_rank_loss(
        current, reference, torch.ones(3)
    )
    assert value.item() == 0.0
    value.backward()
    assert current.grad is not None


def test_hierarchical_triplet_rank_penalizes_order_violation() -> None:
    reference = torch.tensor([[0.9, 0.5, 0.1]])
    current = torch.tensor([[0.1, 0.5, 0.9]], requires_grad=True)
    value = hierarchical_triplet_rank_loss(
        current, reference, torch.ones(3)
    )
    assert value > 0.0


def test_absolute_triplet_weights_reduce_strength_without_renormalizing() -> None:
    reference = torch.tensor([[0.9, 0.5, 0.1]])
    current = torch.tensor([[0.1, 0.5, 0.9]])
    full = hierarchical_triplet_rank_loss(
        current,
        reference,
        torch.ones(3),
        weight_normalization="anchor_count",
    )
    quarter = hierarchical_triplet_rank_loss(
        current,
        reference,
        torch.full((3,), 0.25),
        weight_normalization="anchor_count",
    )
    relative_quarter = hierarchical_triplet_rank_loss(
        current,
        reference,
        torch.full((3,), 0.25),
        weight_normalization="weight_sum",
    )
    torch.testing.assert_close(quarter, 0.25 * full)
    torch.testing.assert_close(relative_quarter, full)


def test_absolute_correlation_weights_scale_group_preservation_strength() -> None:
    bank, _ = _bank()
    reference = torch.randn(7, bank.feature_dim)
    current = reference + torch.randn_like(reference)
    internal_count = bank.internal_without_root()[1].shape[0]
    full = AnchorGeometryLoss(
        bank,
        torch.ones(4),
        torch.ones(internal_count),
        use_internal_anchors=False,
        objective="correlation",
        weight_normalization="anchor_count",
    )(current, reference)
    quarter = AnchorGeometryLoss(
        bank,
        torch.full((4,), 0.25),
        torch.ones(internal_count),
        use_internal_anchors=False,
        objective="correlation",
        weight_normalization="anchor_count",
    )(current, reference)
    torch.testing.assert_close(quarter, 0.25 * full)


def test_anchor_geometry_correlation_and_triplet_objectives() -> None:
    bank, _ = _bank()
    weights = global_preservation_weights(bank)
    reference = torch.randn(7, bank.feature_dim)
    for objective in ("correlation", "triplet_rank"):
        module = AnchorGeometryLoss(
            bank,
            weights.leaf_weights,
            weights.internal_weights,
            objective=objective,
        )
        current = reference.clone().requires_grad_()
        value = module(current, reference)
        assert value < 1e-7
        value.backward()
        assert current.grad is not None


def test_co_moving_anchor_loss_is_invariant_to_joint_rotation() -> None:
    torch.manual_seed(7)
    bank, _ = _bank()
    weights = global_preservation_weights(bank)
    co_moving = AnchorGeometryLoss(
        bank,
        weights.leaf_weights,
        weights.internal_weights,
        anchor_frame="co_moving",
    )
    fixed = AnchorGeometryLoss(
        bank,
        weights.leaf_weights,
        weights.internal_weights,
        anchor_frame="fixed",
    )
    rotation, _ = torch.linalg.qr(torch.randn(bank.feature_dim, bank.feature_dim))
    rotated_bank = HierarchicalAnchorBank(
        bank.leaf_class_ids,
        bank.leaf_anchors @ rotation,
        bank.internal_node_ids,
        bank.internal_anchors @ rotation,
        bank.root_id,
    )
    co_moving.update_current_anchors(rotated_bank)
    reference = torch.randn(12, bank.feature_dim)
    current = (reference @ rotation).requires_grad_()
    assert co_moving(current, reference) < 1e-7
    assert fixed(current, reference) > 1e-5


def test_hybrid_anchor_loss_is_convex_mixture() -> None:
    torch.manual_seed(11)
    bank, _ = _bank()
    weights = global_preservation_weights(bank)
    rotation, _ = torch.linalg.qr(torch.randn(bank.feature_dim, bank.feature_dim))
    rotated_bank = HierarchicalAnchorBank(
        bank.leaf_class_ids,
        bank.leaf_anchors @ rotation,
        bank.internal_node_ids,
        bank.internal_anchors @ rotation,
        bank.root_id,
    )
    losses = {}
    reference = torch.randn(9, bank.feature_dim)
    current = reference @ rotation
    for frame in ("fixed", "co_moving", "hybrid"):
        module = AnchorGeometryLoss(
            bank,
            weights.leaf_weights,
            weights.internal_weights,
            anchor_frame=frame,
            fixed_mix=0.5,
        )
        module.update_current_anchors(rotated_bank)
        losses[frame] = module(current, reference)
    assert torch.allclose(
        losses["hybrid"],
        0.5 * (losses["fixed"] + losses["co_moving"]),
        atol=1e-7,
    )


def test_inverse_dispersion_downweights_noisy_anchor() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tree = GriffinPeronaGreedy().build(
        (0, 1), torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    )
    bank = HierarchicalAnchorBank.from_tree(
        PrototypeBank((0, 1), prototypes), tree
    )
    features = torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [1.0, 0.0], [-1.0, 0.0]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    leaf, internal = inverse_angular_dispersion_reliability(
        features,
        targets,
        bank,
        tree,
        min_weight=0.1,
        max_weight=10.0,
    )
    assert leaf[0] > leaf[1]
    assert internal.numel() == 0
