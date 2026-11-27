from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.anchors import (
    HierarchicalAnchorBank,
    PrototypeBank,
    compute_prototypes,
)
from sacil.hierarchy import GriffinPeronaGreedy


def _bank() -> tuple[HierarchicalAnchorBank, object]:
    features = F.normalize(torch.randn(12, 8), dim=1)
    targets = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    prototypes = PrototypeBank(
        (0, 1, 2), compute_prototypes(features, targets, (0, 1, 2))
    )
    affinity = torch.tensor(
        [[0.0, 0.8, 0.1], [0.8, 0.0, 0.2], [0.1, 0.2, 0.0]]
    )
    tree = GriffinPeronaGreedy().build((0, 1, 2), affinity)
    return HierarchicalAnchorBank.from_tree(prototypes, tree), tree


def test_anchor_bank_is_normalized_non_parametric_and_roundtrips() -> None:
    bank, _ = _bank()
    assert torch.allclose(
        bank.leaf_anchors.norm(dim=1),
        torch.ones(bank.leaf_anchors.shape[0]),
        atol=1e-5,
    )
    assert torch.allclose(
        bank.internal_anchors.norm(dim=1),
        torch.ones(bank.internal_anchors.shape[0]),
        atol=1e-5,
    )
    assert not bank.leaf_anchors.requires_grad
    assert not bank.internal_anchors.requires_grad
    restored = HierarchicalAnchorBank.from_state_dict(bank.state_dict())
    assert restored.leaf_class_ids == bank.leaf_class_ids
    assert restored.internal_node_ids == bank.internal_node_ids
    assert torch.equal(restored.leaf_anchors, bank.leaf_anchors)
    assert torch.equal(restored.internal_anchors, bank.internal_anchors)

