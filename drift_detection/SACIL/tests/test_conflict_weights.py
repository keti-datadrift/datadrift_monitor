from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.anchors import HierarchicalAnchorBank, PrototypeBank
from sacil.hierarchy import GriffinPeronaGreedy
from sacil.methods import compute_conflict_weights


def _fixture():
    prototypes = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.1, 0.9],
            ]
        ),
        dim=1,
    )
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
    return prototypes, tree, bank


def test_nearest_leaf_is_relaxed_and_weights_are_bounded() -> None:
    prototypes, tree, bank = _fixture()
    new = F.normalize(prototypes[0:1] + 0.001, dim=1)
    weights = compute_conflict_weights(
        new,
        bank,
        tree,
        max_neighbors=1,
        old_class_ratio=0.1,
        temperature=0.05,
        min_preservation_weight=0.1,
        ancestor_decay=0.5,
    )
    assert 0.1 <= float(weights.leaf_weights.min()) <= 1.0
    assert float(weights.leaf_weights.max()) <= 1.0
    assert weights.leaf_weights[0] < weights.leaf_weights[2]
    assert weights.leaf_weights[2] == 1.0
    assert torch.all(weights.internal_weights >= 0.1)
    assert torch.all(weights.internal_weights <= 1.0)
    assert bank.root_id not in weights.internal_node_ids


def test_ancestor_relaxation_decays_with_tree_distance() -> None:
    prototypes, tree, bank = _fixture()
    new = prototypes[0:1]
    weights = compute_conflict_weights(
        new, bank, tree, max_neighbors=1, ancestor_decay=0.5
    )
    activation = {
        node_id: float(weights.internal_activations[position])
        for position, node_id in enumerate(weights.internal_node_ids)
    }
    parent = tree.parent(tree.leaf_node_id(0))
    assert parent is not None
    if parent != tree.root_id:
        assert activation[parent] <= float(weights.leaf_activations[0])
