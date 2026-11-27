from __future__ import annotations

import torch


def get_device(config: dict) -> torch.device:
    requested = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested CUDA, but this Python environment cannot access CUDA. "
            "Use the DD conda environment or set device: cpu."
        )
    return torch.device(requested)


def configured_device_ids(config: dict) -> list[int]:
    multi_gpu = config.get("multi_gpu", {})
    if isinstance(multi_gpu, dict):
        device_ids = multi_gpu.get("device_ids", config.get("device_ids"))
    else:
        device_ids = config.get("device_ids")
    if device_ids is None:
        return list(range(torch.cuda.device_count()))
    return [int(device_id) for device_id in device_ids]


def multi_gpu_enabled(config: dict) -> bool:
    multi_gpu = config.get("multi_gpu", {})
    if isinstance(multi_gpu, dict):
        return bool(multi_gpu.get("enabled", False))
    return multi_gpu in {"data_parallel", "dp", True}


def multi_gpu_backend(config: dict) -> str:
    multi_gpu = config.get("multi_gpu", {})
    if isinstance(multi_gpu, dict):
        return str(multi_gpu.get("backend", "data_parallel"))
    if multi_gpu in {"data_parallel", "dp", True}:
        return "data_parallel"
    return "none"


def maybe_data_parallel(model: torch.nn.Module, config: dict) -> torch.nn.Module:
    if config.get("device") != "cuda":
        return model
    if not multi_gpu_enabled(config):
        return model
    if multi_gpu_backend(config) != "data_parallel":
        raise ValueError(f"Unsupported multi_gpu backend: {multi_gpu_backend(config)}")
    if torch.cuda.device_count() < 2:
        return model

    device_ids = configured_device_ids(config)
    if len(device_ids) < 2:
        return model
    if torch.cuda.device_count() < len(device_ids):
        raise RuntimeError(
            f"Config requested device_ids={device_ids}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )
    return torch.nn.DataParallel(model, device_ids=device_ids)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model
