from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


class ImageFeatureEncoder(nn.Module):
    """Expose ``CLIPModel.get_image_features`` as a regular forward method."""

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_image_features(pixel_values=pixel_values)


def build_parallel_image_encoder(
    clip_model: nn.Module,
    device: str,
    gpu_ids: tuple[int, ...] | list[int],
) -> tuple[nn.Module, tuple[int, ...]]:
    encoder = ImageFeatureEncoder(clip_model)
    if not device.startswith("cuda"):
        return encoder, ()

    requested = tuple(dict.fromkeys(int(gpu_id) for gpu_id in gpu_ids))
    if not requested:
        requested = (int(device.split(":", 1)[1]) if ":" in device else 0,)
    available = torch.cuda.device_count()
    invalid = [gpu_id for gpu_id in requested if gpu_id < 0 or gpu_id >= available]
    if invalid:
        raise ValueError(
            f"Requested CUDA devices {invalid}, but torch sees {available} GPUs"
        )
    primary = int(device.split(":", 1)[1]) if ":" in device else 0
    if requested[0] != primary:
        raise ValueError(
            f"Primary device {device} must match the first runtime.gpu_ids entry "
            f"({requested[0]})"
        )
    if len(requested) == 1:
        return encoder, requested
    return (
        nn.DataParallel(
            encoder,
            device_ids=list(requested),
            output_device=primary,
        ),
        requested,
    )


@dataclass
class RNGSnapshot:
    cpu_state: torch.Tensor
    cuda_states: dict[int, torch.Tensor]


def capture_rng_state(cuda_devices: tuple[int, ...] | list[int] = ()) -> RNGSnapshot:
    cuda_states = {}
    if torch.cuda.is_available():
        for device_id in cuda_devices:
            cuda_states[int(device_id)] = torch.cuda.get_rng_state(int(device_id))
    return RNGSnapshot(
        cpu_state=torch.get_rng_state(),
        cuda_states=cuda_states,
    )


def restore_rng_state(snapshot: RNGSnapshot) -> None:
    torch.set_rng_state(snapshot.cpu_state)
    if torch.cuda.is_available():
        for device_id, state in snapshot.cuda_states.items():
            torch.cuda.set_rng_state(state, int(device_id))


def is_cuda_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def grad_cache_forward_backward(
    images: torch.Tensor,
    image_encoder: nn.Module,
    loss_closure: Callable[[torch.Tensor], tuple[torch.Tensor, dict]],
    *,
    micro_batch_size: int,
    scaler=None,
    autocast_factory: Callable[[], object] | None = None,
    cuda_devices: tuple[int, ...] | list[int] = (),
) -> tuple[torch.Tensor, dict]:
    """Backpropagate an exact logical-batch representation loss in chunks.

    This follows GradCache's three computation phases: graph-less chunked
    encoding, full logical-batch loss and representation-gradient caching,
    then chunked encoder replay with the cached representation gradients.
    RNG states are restored for each replay so stochastic encoder layers use
    the same masks as the graph-less forward.
    """
    if int(micro_batch_size) <= 0:
        raise ValueError("micro_batch_size must be positive")
    if images.dim() < 1 or int(images.shape[0]) == 0:
        raise ValueError("GradCache requires a non-empty logical batch")
    autocast_factory = autocast_factory or nullcontext
    batch_size = int(images.shape[0])
    slices = [
        slice(start, min(batch_size, start + int(micro_batch_size)))
        for start in range(0, batch_size, int(micro_batch_size))
    ]

    snapshots = []
    cached_chunks = []
    with torch.no_grad():
        for chunk_slice in slices:
            snapshots.append(capture_rng_state(cuda_devices))
            with autocast_factory():
                chunk_features = image_encoder(images[chunk_slice])
            cached_chunks.append(chunk_features.detach())

    cached_features = torch.cat(cached_chunks, dim=0).detach().requires_grad_(True)
    loss, stats = loss_closure(cached_features)
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    if cached_features.grad is None:
        raise RuntimeError("GradCache loss did not produce representation gradients")
    representation_gradients = cached_features.grad.detach()

    for chunk_slice, snapshot in zip(slices, snapshots):
        restore_rng_state(snapshot)
        with autocast_factory():
            replayed_features = image_encoder(images[chunk_slice])
        torch.autograd.backward(
            replayed_features,
            grad_tensors=representation_gradients[chunk_slice],
        )

    return loss.detach(), stats
