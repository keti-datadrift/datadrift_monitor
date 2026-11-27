from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.gradient_cache import (
    GradientCacheConfig,
    configure_batch_norm,
    gradient_cache_step,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, inputs):
        return nn.functional.normalize(self.layers(inputs), dim=-1)


class TinyBatchNormEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, inputs):
        return nn.functional.normalize(self.layers(inputs), dim=-1)


def pair_loss(features, labels):
    similarity = features @ features.t()
    target = (labels[:, None] == labels[None, :]).float()
    return ((similarity - target) ** 2).mean()


def collect_gradients(model):
    return [parameter.grad.detach().clone() for parameter in model.parameters()]


def main():
    torch.manual_seed(7)
    inputs = torch.randn(12, 5)
    labels = torch.arange(12) // 2

    direct_model = TinyEncoder()
    cached_model = deepcopy(direct_model)

    direct_loss = pair_loss(direct_model(inputs), labels)
    direct_loss.backward()
    direct_gradients = collect_gradients(direct_model)

    cached_loss = gradient_cache_step(
        cached_model,
        inputs,
        labels,
        pair_loss,
        GradientCacheConfig(chunk_size=3, batch_norm_mode="standard"),
        amp_enabled=False,
        scaler=None,
    )
    cached_gradients = collect_gradients(cached_model)

    assert torch.allclose(direct_loss.detach(), cached_loss, atol=1e-7, rtol=1e-6)
    for direct, cached in zip(direct_gradients, cached_gradients):
        assert torch.allclose(direct, cached, atol=1e-6, rtol=1e-5)

    try:
        GradientCacheConfig(chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid chunk size was not rejected.")

    bn_model = TinyBatchNormEncoder()
    bn_model.train()
    configure_batch_norm(bn_model, mode="chunk")
    bn_layer = bn_model.layers[1]
    before = int(bn_layer.num_batches_tracked.item())
    gradient_cache_step(
        bn_model,
        inputs,
        labels,
        pair_loss,
        GradientCacheConfig(chunk_size=3, batch_norm_mode="chunk"),
        amp_enabled=False,
        scaler=None,
    )
    after = int(bn_layer.num_batches_tracked.item())
    assert after - before == 4, "Chunk BN statistics should update only once per chunk."

    if torch.cuda.is_available():
        amp_model = TinyEncoder().cuda()
        amp_inputs = inputs.cuda()
        amp_labels = labels.cuda()
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        optimizer = torch.optim.SGD(amp_model.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        amp_loss = gradient_cache_step(
            amp_model,
            amp_inputs,
            amp_labels,
            pair_loss,
            GradientCacheConfig(chunk_size=3, batch_norm_mode="standard"),
            amp_enabled=True,
            scaler=scaler,
        )
        scaler.unscale_(optimizer)
        assert torch.isfinite(amp_loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in amp_model.parameters()
        )
        scaler.step(optimizer)
        scaler.update()

    print("Gradient Cache smoke checks passed")


if __name__ == "__main__":
    main()
