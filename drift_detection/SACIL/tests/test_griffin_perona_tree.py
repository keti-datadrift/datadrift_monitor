from __future__ import annotations

import torch

from sacil.hierarchy import GriffinPeronaGreedy, HierarchyTree


def _example_tree() -> HierarchyTree:
    affinity = torch.tensor(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.2, 0.1],
            [0.1, 0.2, 0.0, 0.8],
            [0.1, 0.1, 0.8, 0.0],
        ]
    )
    return GriffinPeronaGreedy().build((10, 20, 30, 40), affinity)


def test_expected_pair_merges_first_and_tree_is_binary() -> None:
    tree = _example_tree()
    assert tree.nodes["node:0000"].members == (10, 20)
    assert tree.nodes["node:0001"].members == (30, 40)
    assert tree.num_leaves == 4
    assert len(tree.internal_node_ids()) == 3
    assert tree.descendants(tree.root_id) == (10, 20, 30, 40)


def test_tree_is_deterministic_and_roundtrips() -> None:
    first = _example_tree()
    second = _example_tree()
    assert first.state_dict() == second.state_dict()
    restored = HierarchyTree.from_state_dict(first.state_dict())
    assert restored.state_dict() == first.state_dict()
    assert (
        restored.distance_from_ancestor(
            restored.root_id, restored.leaf_node_id(10)
        )
        == 2
    )

