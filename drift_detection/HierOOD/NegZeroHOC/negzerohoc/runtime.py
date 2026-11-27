from __future__ import annotations

import os
import random

import numpy as np
import torch


def configured_device(runtime_cfg: dict | None, default: str = "cuda") -> str:
    runtime_cfg = dict(runtime_cfg or {})
    device = str(runtime_cfg.get("device", default))
    gpu_id = runtime_cfg.get("gpu_id")
    if gpu_id is not None and device == "cuda":
        device = f"cuda:{int(gpu_id)}"
    return device


def available_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG and optionally require deterministic PyTorch execution.

    This must run before the first CUDA operation so cuBLAS observes its
    deterministic workspace configuration.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_data_loader_worker(worker_id: int) -> None:
    """Deterministically seed Python and NumPy in a DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
