from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """Replay the RNG state used by the graph-less forward pass."""

    def __init__(self, *tensors: Tensor):
        self.cpu_state = torch.get_rng_state()
        self.gpu_devices, self.gpu_states = get_device_states(*tensors)
        self._fork = None

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu_state)
        set_device_states(self.gpu_devices, self.gpu_states)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._fork.__exit__(exc_type, exc_value, traceback)
        self._fork = None


def iter_batch_norm_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            yield module


def configure_batch_norm(model: nn.Module, mode: str, freeze_affine: bool = False):
    if mode not in {"chunk", "frozen", "standard"}:
        raise ValueError(
            f"Unsupported batch_norm mode: {mode}. "
            "Expected 'chunk', 'frozen', or 'standard'."
        )

    for module in iter_batch_norm_modules(model):
        if mode == "frozen":
            module.eval()
        if module.weight is not None:
            module.weight.requires_grad_(not freeze_affine)
        if module.bias is not None:
            module.bias.requires_grad_(not freeze_affine)


def snapshot_batch_norm_state(model: nn.Module):
    state = []
    for module in iter_batch_norm_modules(model):
        state.append(
            (
                module,
                module.running_mean.detach().clone()
                if module.running_mean is not None
                else None,
                module.running_var.detach().clone()
                if module.running_var is not None
                else None,
                module.num_batches_tracked.detach().clone()
                if module.num_batches_tracked is not None
                else None,
            )
        )
    return state


def restore_batch_norm_state(state):
    with torch.no_grad():
        for module, running_mean, running_var, num_batches_tracked in state:
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if num_batches_tracked is not None:
                module.num_batches_tracked.copy_(num_batches_tracked)


def autocast_context(device_type: str, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=torch.float16)


@dataclass
class GradientCacheConfig:
    chunk_size: int
    batch_norm_mode: str = "chunk"

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("gradient_cache.chunk_size must be a positive integer.")
        if self.batch_norm_mode not in {"chunk", "frozen", "standard"}:
            raise ValueError(
                "batch_norm.mode must be 'chunk', 'frozen', or 'standard'."
            )


def gradient_cache_step(
    model: nn.Module,
    inputs: Tensor,
    path_labels: Tensor,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    config: GradientCacheConfig,
    *,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
) -> Tensor:
    """Populate model gradients for one logical batch using cached feature gradients."""

    input_chunks = list(inputs.split(config.chunk_size, dim=0))
    device_type = inputs.device.type
    random_states = []
    feature_chunks = []

    bn_state = (
        snapshot_batch_norm_state(model)
        if config.batch_norm_mode == "chunk"
        else None
    )

    with torch.no_grad():
        for input_chunk in input_chunks:
            random_states.append(RandContext(input_chunk))
            with autocast_context(device_type, amp_enabled):
                feature_chunks.append(model(input_chunk))

    if bn_state is not None:
        # The graph-less pass must not count as a second BN statistics update.
        restore_batch_norm_state(bn_state)

    cached_features = torch.cat(feature_chunks, dim=0).detach().float()
    cached_features.requires_grad_(True)
    loss = loss_fn(cached_features, path_labels)

    if amp_enabled:
        if scaler is None:
            raise ValueError("AMP gradient cache requires a GradScaler.")
        scaler.scale(loss).backward()
    else:
        loss.backward()

    cached_gradients = cached_features.grad.split(config.chunk_size, dim=0)

    for input_chunk, random_state, cached_gradient in zip(
        input_chunks, random_states, cached_gradients
    ):
        with random_state:
            with autocast_context(device_type, amp_enabled):
                features = model(input_chunk)
        surrogate = torch.sum(features.float() * cached_gradient)
        surrogate.backward()

    return loss.detach()
