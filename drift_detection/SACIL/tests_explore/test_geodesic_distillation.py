from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sacil.config import load_config_tree
from sacil.engine.table1_trainer import (
    base_recipe_signature,
    resolve_geodesic_distillation_options,
)
from sacil.methods import (
    geodesic_distillation_loss,
    geodesic_flow_kernel,
    principal_subspace,
)


ROOT = Path(__file__).resolve().parents[1]


def test_principal_subspace_is_orthonormal() -> None:
    features = torch.randn(20, 8)
    basis = principal_subspace(features, 4)
    assert basis.shape == (8, 4)
    assert torch.allclose(basis.T @ basis, torch.eye(4), atol=1e-5)


def test_geodesic_kernel_is_symmetric_positive_semidefinite() -> None:
    source = principal_subspace(torch.randn(24, 8), 3)
    target = principal_subspace(torch.randn(24, 8), 3)
    kernel = geodesic_flow_kernel(source, target)
    assert torch.allclose(kernel, kernel.T, atol=1e-6)
    assert torch.linalg.eigvalsh(kernel).min().item() >= -1e-5


def test_geodesic_loss_has_finite_current_feature_gradient() -> None:
    reference = torch.randn(24, 8)
    current = (reference + 0.1 * torch.randn_like(reference)).requires_grad_()
    loss = geodesic_distillation_loss(
        current, reference, subspace_rank=4
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
    assert current.grad.abs().sum().item() > 0


def test_identical_features_have_zero_geodesic_distillation() -> None:
    features = torch.randn(24, 8)
    loss = geodesic_distillation_loss(
        features, features, subspace_rank=4
    )
    assert loss.item() == pytest.approx(0.0, abs=2e-5)


def test_geodl_config_is_incremental_only_resnet32_control() -> None:
    canonical = load_config_tree(
        ROOT / "configs/table1/cifar100/icarl_nme_b50_inc5_resnet32.yaml"
    )
    config = load_config_tree(
        ROOT / "configs/explore/cifar100/icarl_geodl_equation.yaml"
    )
    options = resolve_geodesic_distillation_options(
        "icarl", config["method"]
    )
    assert config["model"]["backbone"] == "resnet32"
    assert config["evaluation"]["classifier"] == "nme"
    assert config["memory"]["exemplars_per_class"] == 20
    assert config["method"]["kd_weight"] == 0.0
    assert options["enabled"]
    assert options["subspace_rank"] == 32
    assert options["lambda"] == pytest.approx(6.0)
    assert base_recipe_signature(config) == base_recipe_signature(canonical)
