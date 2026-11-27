from __future__ import annotations

import torch
from torch.nn import functional as F

from sacil.models import IncrementalNet


def test_incremental_model_expands_without_changing_old_weights() -> None:
    model = IncrementalNet(5)
    old_weight = model.classifier.weight.detach().clone()
    model.expand_classes(8)
    assert model.num_classes == 8
    assert torch.equal(model.classifier.weight[:5], old_weight)
    prototypes = F.normalize(torch.randn(3, model.feature_dim), dim=1)
    model.classifier.initialize_rows(5, prototypes)
    assert torch.allclose(
        F.normalize(model.classifier.weight[5:], dim=1),
        prototypes,
        atol=1e-6,
    )


def test_resnet32_forward_shapes() -> None:
    model = IncrementalNet(5)
    images = torch.randn(2, 3, 32, 32)
    logits, features = model(images, return_features=True)
    assert features.shape == (2, 64)
    assert logits.shape == (2, 5)


def test_cifar_resnet18_forward_shapes() -> None:
    model = IncrementalNet(5, backbone="resnet18")
    images = torch.randn(2, 3, 32, 32)
    logits, features = model(images, return_features=True)
    assert features.shape == (2, 512)
    assert logits.shape == (2, 5)
