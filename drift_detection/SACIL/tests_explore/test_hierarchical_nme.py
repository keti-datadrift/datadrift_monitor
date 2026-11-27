from __future__ import annotations

import torch

from sacil.methods.hierarchical_nme import hierarchical_shrink_nme_means


def test_hierarchical_nme_is_normalized_bounded_and_deterministic() -> None:
    means = torch.tensor(
        [[1.0, 0.0], [0.98, 0.20], [-1.0, 0.0], [-0.98, -0.20]]
    )
    means = torch.nn.functional.normalize(means, dim=1)
    features = torch.cat(
        [
            means[0].repeat(4, 1),
            torch.tensor([[0.7, 0.7], [0.9, 0.4], [0.8, 0.6], [1.0, 0.1]]),
            means[2].repeat(4, 1),
            torch.tensor([[-0.7, -0.7], [-0.9, -0.4], [-0.8, -0.6], [-1.0, -0.1]]),
        ]
    )
    targets = torch.arange(4).repeat_interleave(4)

    first, diagnostics, tree = hierarchical_shrink_nme_means(
        means, features, targets, (10, 11, 12, 13), max_shrinkage=0.25
    )
    second, second_diagnostics, second_tree = hierarchical_shrink_nme_means(
        means, features, targets, (10, 11, 12, 13), max_shrinkage=0.25
    )

    assert torch.allclose(first.norm(dim=1), torch.ones(4), atol=1.0e-6)
    assert torch.equal(first, second)
    assert diagnostics == second_diagnostics
    assert tree == second_tree
    assert all(0.0 <= value <= 0.25 for value in diagnostics.shrinkage)
    assert diagnostics.shrinkage[0] < diagnostics.shrinkage[1]


def test_hierarchical_nme_rejects_missing_incremental_label() -> None:
    means = torch.eye(3)
    features = torch.eye(3)[:2]
    targets = torch.tensor([0, 1])
    try:
        hierarchical_shrink_nme_means(means, features, targets, (0, 1, 2))
    except ValueError as error:
        assert "contiguous incremental labels" in str(error)
    else:
        raise AssertionError("missing class must be rejected")
