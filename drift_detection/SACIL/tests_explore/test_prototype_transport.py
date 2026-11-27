from __future__ import annotations

import torch

from sacil.engine.table1_trainer import resolve_prototype_transport_options

from sacil.hierarchy import GriffinPeronaGreedy
from sacil.methods.prototype_transport import (
    affine_ridge_transport,
    empirical_bayes_residual_transport,
    orthogonal_procrustes_transport,
    rigid_procrustes_transport,
    similarity_procrustes_transport,
    transport_class_prototypes,
    weighted_rigid_procrustes_transport,
)


def test_global_transport_applies_one_drift_and_normalizes() -> None:
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    old = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    current = old + torch.tensor([0.0, 0.2])
    result = transport_class_prototypes(
        prototypes,
        old,
        current,
        torch.tensor([0, 0, 1, 1]),
        (10, 11),
        mode="global",
        sigma=1.0,
    )
    assert torch.allclose(result.drifts[0], result.drifts[1])
    assert torch.allclose(result.prototypes.norm(dim=1), torch.ones(2))
    assert result.support_counts.tolist() == [4, 4]


def test_hierarchy_parent_transport_uses_local_support() -> None:
    prototypes = torch.eye(4)
    old = prototypes.repeat_interleave(2, dim=0)
    current = old.clone()
    current[:4, 0] += 0.2
    current[4:, 3] += 0.2
    targets = torch.arange(4).repeat_interleave(2)
    affinity = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    tree = GriffinPeronaGreedy().build((10, 11, 12, 13), affinity)
    result = transport_class_prototypes(
        prototypes,
        old,
        current,
        targets,
        (10, 11, 12, 13),
        mode="hierarchy_parent",
        tree=tree,
        sigma=1.0,
    )
    assert result.support_counts.tolist() == [4, 4, 4, 4]
    assert torch.allclose(result.drifts[0], result.drifts[1])
    assert torch.allclose(result.drifts[2], result.drifts[3])
    assert not torch.allclose(result.drifts[0], result.drifts[2])


def test_class_transport_uses_only_same_class_support() -> None:
    prototypes = torch.eye(2)
    old = prototypes.repeat_interleave(3, dim=0)
    current = old.clone()
    current[:3, 1] += 0.2
    current[3:, 0] -= 0.2
    result = transport_class_prototypes(
        prototypes,
        old,
        current,
        torch.tensor([0, 0, 0, 1, 1, 1]),
        (10, 11),
        mode="class",
        sigma=1.0,
    )
    assert result.support_counts.tolist() == [3, 3]
    assert not torch.allclose(result.drifts[0], result.drifts[1])


def test_orthogonal_procrustes_recovers_rotation_and_preserves_geometry() -> None:
    torch.manual_seed(7)
    old = torch.randn(80, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    current = old @ q
    prototypes = torch.nn.functional.normalize(torch.randn(5, 6), dim=1)
    moved, rotation, residual = orthogonal_procrustes_transport(
        prototypes, old, current
    )
    torch.testing.assert_close(rotation, q, atol=1.0e-5, rtol=1.0e-5)
    torch.testing.assert_close(
        moved @ moved.T,
        prototypes @ prototypes.T,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert residual < 1.0e-10


def test_rigid_procrustes_recovers_rotation_and_translation() -> None:
    torch.manual_seed(9)
    old = torch.nn.functional.normalize(torch.randn(96, 5), dim=1)
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    translation = torch.tensor([0.03, -0.02, 0.01, 0.04, -0.01])
    current = old @ q + translation
    prototypes = torch.nn.functional.normalize(torch.randn(4, 5), dim=1)
    moved, rotation, fitted_translation, residual = rigid_procrustes_transport(
        prototypes, old, current
    )
    assert moved.shape == prototypes.shape
    assert rotation.shape == q.shape
    assert fitted_translation.shape == translation.shape
    assert torch.isfinite(moved).all()
    assert residual >= 0.0


def test_affine_ridge_transport_is_finite_and_fits_pairs() -> None:
    torch.manual_seed(11)
    old = torch.randn(128, 4)
    current = old @ torch.tensor(
        [
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.1, 0.1],
            [0.1, 0.0, 0.0, 1.0],
        ]
    )
    prototypes = torch.randn(3, 4)
    moved, mapping, residual = affine_ridge_transport(
        prototypes, old, current, ridge=1.0e-2
    )
    assert moved.shape == prototypes.shape
    assert mapping.shape == (5, 4)
    assert torch.isfinite(moved).all()
    assert residual >= 0.0


def test_empirical_bayes_residual_transport_estimates_relaxation() -> None:
    torch.manual_seed(13)
    old = torch.nn.functional.normalize(torch.randn(120, 5), dim=1)
    targets = torch.arange(4).repeat_interleave(30)
    class_shift = torch.zeros(4, 5)
    class_shift[0, 0] = 0.08
    class_shift[1, 1] = -0.08
    current = torch.nn.functional.normalize(old + class_shift[targets], dim=1)
    prototypes = torch.nn.functional.normalize(torch.randn(4, 5), dim=1)
    moved, _, _, residual, shrinkage, residual_means = (
        empirical_bayes_residual_transport(
            prototypes, old, current, targets
        )
    )
    assert moved.shape == prototypes.shape
    assert residual_means.shape == prototypes.shape
    assert 0.0 <= shrinkage <= 1.0
    assert shrinkage > 0.0
    assert residual >= 0.0


def test_weighted_rigid_procrustes_returns_normalized_prototypes() -> None:
    torch.manual_seed(17)
    old = torch.nn.functional.normalize(torch.randn(80, 6), dim=1)
    current = torch.nn.functional.normalize(
        old + 0.01 * torch.randn_like(old), dim=1
    )
    targets = torch.arange(4).repeat_interleave(20)
    prototypes = torch.nn.functional.normalize(torch.randn(4, 6), dim=1)
    moved, rotation, translation, residual, weights = (
        weighted_rigid_procrustes_transport(
            prototypes, old, current, targets, sigma=0.5
        )
    )
    assert rotation.shape == (6, 6)
    assert translation.shape == (6,)
    torch.testing.assert_close(moved.norm(dim=1), torch.ones(4))
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    assert residual >= 0.0


def test_similarity_procrustes_recovers_isotropic_scale() -> None:
    torch.manual_seed(19)
    old = torch.nn.functional.normalize(torch.randn(100, 5), dim=1)
    q, _ = torch.linalg.qr(torch.randn(5, 5))
    current = 0.9 * old @ q + torch.tensor([0.02, 0.0, -0.01, 0.01, 0.0])
    prototypes = torch.nn.functional.normalize(torch.randn(4, 5), dim=1)
    moved, rotation, translation, scale, residual = (
        similarity_procrustes_transport(prototypes, old, current)
    )
    assert moved.shape == prototypes.shape
    assert rotation.shape == q.shape
    assert translation.shape == (5,)
    assert scale > 0.0
    assert residual >= 0.0


def test_rigid_flip_transport_config_resolves_explicitly() -> None:
    options = resolve_prototype_transport_options(
        "icarl",
        {
            "prototype_transport": {
                "enabled": True,
                "mode": "procrustes_rigid_flip",
                "sigma": 0.2,
            }
        },
    )
    assert options["enabled"] is True
    assert options["mode"] == "procrustes_rigid_flip"
    assert options["horizontal_flip_consistent"] is True
