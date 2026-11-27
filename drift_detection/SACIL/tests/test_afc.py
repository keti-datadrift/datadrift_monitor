from __future__ import annotations

import torch

from sacil.methods import afc_nca_loss, afc_pod_loss
from sacil.models import AFCIncrementalNet


def test_afc_model_forward_and_expansion_shapes() -> None:
    model = AFCIncrementalNet(
        4,
        initial_size=4,
        increment_size=2,
        proxies_per_class=3,
    )
    images = torch.randn(2, 3, 32, 32)
    output = model.forward_detailed(images)
    assert output.logits.shape == (2, 4)
    assert output.features.shape == (2, 64)
    assert [tuple(value.shape) for value in output.attentions] == [
        (2, 16, 32, 32),
        (2, 32, 16, 16),
        (2, 64, 8, 8),
        (2, 64, 8, 8),
    ]
    old_weights = model.classifier.weights.detach().clone()
    model.expand_classes(6, torch.randn(2, 3, 64))
    assert model.num_classes == 6
    assert torch.equal(model.classifier.weights[:12], old_weights)
    assert model(images).shape == (2, 6)


def test_afc_nca_and_pod_losses_are_finite() -> None:
    logits = torch.randn(5, 4, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 0])
    nca = afc_nca_loss(logits, targets, scale)
    assert torch.isfinite(nca)
    nca.backward()
    assert logits.grad is not None
    assert scale.grad is not None

    reference = (
        torch.randn(2, 3, 4, 4),
        torch.randn(2, 5, 2, 2),
    )
    importance = (torch.ones(3), torch.ones(5))
    assert torch.equal(
        afc_pod_loss(reference, reference, importance),
        torch.tensor(0.0),
    )
    changed = (reference[0] + 0.1 * torch.randn_like(reference[0]), reference[1])
    assert afc_pod_loss(reference, changed, importance).item() > 0


def test_afc_importance_collection_normalizes_each_layer() -> None:
    model = AFCIncrementalNet(
        4,
        initial_size=4,
        increment_size=2,
        proxies_per_class=2,
    )
    model.backbone.reset_importance()
    model.backbone.start_importance_collection()
    logits = model(torch.randn(3, 3, 32, 32))
    logits.square().mean().backward()
    model.backbone.stop_importance_collection()
    model.backbone.normalize_importance()
    for layer in model.backbone.importance_layers:
        assert torch.allclose(
            layer.importance.mean(), torch.tensor(1.0), atol=1e-5
        )


def test_afc_dual_branch_starts_identical_and_receives_gradients() -> None:
    model = AFCIncrementalNet(
        4,
        initial_size=4,
        increment_size=2,
        proxies_per_class=2,
    )
    model.eval()
    images = torch.randn(3, 3, 32, 32)
    conventional = model.forward_detailed(images)
    model.enable_rebalancing_branch()
    rebalancing = model.forward_detailed(
        images, branch="rebalancing"
    )
    assert torch.allclose(
        conventional.features, rebalancing.features, atol=1e-6
    )
    logits = model(images)
    assert torch.allclose(logits, conventional.logits, atol=1e-6)
    logits.square().mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.backbone.rebalancing_stage4.parameters()
    )
