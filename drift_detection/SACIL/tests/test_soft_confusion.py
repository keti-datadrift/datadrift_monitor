from __future__ import annotations

import torch

from sacil.hierarchy import cosine_soft_confusion, symmetric_affinity


def test_soft_confusion_has_zero_diagonal_and_symmetric_affinity() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    confusion = cosine_soft_confusion(
        features, targets, weights, temperature=0.2
    )
    affinity = symmetric_affinity(confusion)
    assert confusion.shape == (2, 2)
    assert torch.equal(torch.diag(confusion), torch.zeros(2))
    assert torch.equal(torch.diag(affinity), torch.zeros(2))
    assert torch.allclose(affinity, affinity.t())
    assert torch.isfinite(affinity).all()

