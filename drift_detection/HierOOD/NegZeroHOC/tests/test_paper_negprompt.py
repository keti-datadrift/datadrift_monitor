import pytest
import torch

from negzerohoc.paper_negprompt import (
    negprompt_loss,
    negprompt_probabilities,
)


def test_negprompt_probabilities_share_one_joint_denominator():
    images = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negatives = torch.tensor([
        [[-1.0, 0.0], [0.0, -1.0]],
        [[-1.0, 0.0], [0.0, -1.0]],
    ])
    positive, negative = negprompt_probabilities(
        images,
        positives,
        negatives,
        temperature=1.0,
    )
    assert positive.shape == (1, 2)
    assert negative.shape == (1, 2, 2)
    assert torch.allclose(
        positive.sum() + negative.sum(),
        torch.tensor(1.0),
    )
    assert int(positive.argmax(dim=1).item()) == 0


def test_published_and_repulsive_npd_have_opposite_signs():
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negatives = positives.unsqueeze(1).clone()
    published, published_stats = negprompt_loss(
        images,
        positives,
        negatives,
        logit_scale=1.0,
        beta=1.0,
        gamma=0.0,
        distance_mode="attractive",
    )
    repulsive, repulsive_stats = negprompt_loss(
        images,
        positives,
        negatives,
        logit_scale=1.0,
        beta=1.0,
        gamma=0.0,
        distance_mode="repulsive",
    )
    assert published_stats["npd_loss"] == pytest.approx(-1.0)
    assert repulsive_stats["npd_loss"] == pytest.approx(1.0)
    assert published < repulsive


def test_negprompt_loss_rejects_invalid_mode():
    with pytest.raises(ValueError, match="distance_mode"):
        negprompt_loss(
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 1, 3),
            logit_scale=1.0,
            beta=0.1,
            gamma=0.0,
            distance_mode="unknown",
        )
