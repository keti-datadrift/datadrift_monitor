from __future__ import annotations

import torch

from sacil.methods import (
    RipsNet,
    TopologyDistillationLoss,
    old_logit_kl_loss,
)
from sacil.methods.replay_ce import (
    method_uses_afc,
    method_uses_dual_rebalancing,
    method_uses_geometry,
    method_uses_topkd,
)


def test_old_logit_kl_is_zero_for_matching_old_logits() -> None:
    reference = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    current = torch.cat(
        [reference.clone(), torch.tensor([[9.0], [-3.0]])], dim=1
    )
    loss = old_logit_kl_loss(current, reference, temperature=4.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_old_logit_kl_updates_only_student() -> None:
    current = torch.randn(4, 5, requires_grad=True)
    reference = torch.randn(4, 3, requires_grad=True)
    loss = old_logit_kl_loss(current, reference)
    loss.backward()
    assert current.grad is not None
    assert reference.grad is None
    assert torch.count_nonzero(current.grad[:, 3:]) == 0


def test_ripsnet_is_permutation_invariant() -> None:
    torch.manual_seed(3)
    network = RipsNet(feature_dim=8)
    points = torch.randn(12, 8)
    permutation = torch.randperm(points.shape[0])
    assert torch.allclose(
        network(points),
        network(points[permutation]),
        atol=1e-6,
    )


def test_topology_loss_freezes_ripsnet_but_updates_current_features() -> None:
    torch.manual_seed(4)
    network = RipsNet(feature_dim=8)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    loss_function = TopologyDistillationLoss(network)
    current = torch.randn(12, 8, requires_grad=True)
    reference = torch.randn(12, 8, requires_grad=True)
    loss = loss_function(current, reference)
    loss.backward()
    assert current.grad is not None
    assert reference.grad is None
    assert all(parameter.grad is None for parameter in network.parameters())


def test_afc_topkd_method_flags() -> None:
    assert method_uses_afc("afc_topkd")
    assert method_uses_topkd("afc_topkd")
    assert not method_uses_geometry("afc_topkd")


def test_afc_topkd_dual_rebalancing_method_flags() -> None:
    name = "afc_topkd_dual_rebalance"
    assert method_uses_afc(name)
    assert method_uses_topkd(name)
    assert method_uses_dual_rebalancing(name)
    assert not method_uses_geometry(name)
