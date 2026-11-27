from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import torch
from torch import nn


@dataclass
class VisionLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    layers: str | tuple[int, ...] = "all"

    @classmethod
    def from_dict(cls, data: dict | None) -> "VisionLoRAConfig":
        data = dict(data or {})
        target_modules = tuple(data.get("target_modules", ("q_proj", "v_proj")))
        layers = data.get("layers", "all")
        if isinstance(layers, list):
            layers = tuple(int(layer) for layer in layers)
        return cls(
            rank=int(data.get("rank", 8)),
            alpha=float(data.get("alpha", 16.0)),
            dropout=float(data.get("dropout", 0.05)),
            target_modules=target_modules,
            layers=layers,
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        if isinstance(self.layers, tuple):
            data["layers"] = list(self.layers)
        return data


class LoRALinear(nn.Module):
    """Low-rank residual update around a frozen linear projection."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base layer")

        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

        self.rank = min(int(rank), base_layer.in_features, base_layer.out_features)
        self.scaling = float(alpha) / float(self.rank)
        self.enabled = True
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base_layer.out_features, bias=False)
        self.lora_a.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.lora_b.to(
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        if not self.enabled:
            return base
        delta = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        return base + delta


def _module_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _selected_layer(module_name: str, layers: str | tuple[int, ...]) -> bool:
    if layers == "all":
        return True
    match = re.search(r"vision_model\.encoder\.layers\.(\d+)\.", module_name)
    return match is not None and int(match.group(1)) in set(layers)


def inject_clip_vision_lora(clip_model: nn.Module, cfg: VisionLoRAConfig) -> list[str]:
    """Insert LoRA into selected CLIP vision attention projections."""
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    replacements = []
    for module_name, module in list(clip_model.named_modules()):
        if not module_name.startswith("vision_model.encoder.layers."):
            continue
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in cfg.target_modules:
            continue
        if not _selected_layer(module_name, cfg.layers):
            continue
        parent, attribute = _module_parent(clip_model, module_name)
        setattr(
            parent,
            attribute,
            LoRALinear(
                module,
                rank=cfg.rank,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
            ),
        )
        replacements.append(module_name)

    if not replacements:
        raise RuntimeError(
            "No CLIP vision modules matched the LoRA configuration. "
            f"targets={cfg.target_modules}, layers={cfg.layers}"
        )
    return replacements


def iter_vision_lora_modules(module: nn.Module):
    for child in module.modules():
        if isinstance(child, LoRALinear):
            yield child


def vision_lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    parameters = []
    for lora_module in iter_vision_lora_modules(module):
        parameters.extend(lora_module.lora_a.parameters())
        parameters.extend(lora_module.lora_b.parameters())
    return parameters


def vision_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
        if ".lora_a." in key or ".lora_b." in key
    }


def load_vision_lora_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    incompatible = module.load_state_dict(state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"Unexpected Vision LoRA checkpoint keys: {unexpected}")


def set_vision_lora_train_mode(module: nn.Module, training: bool) -> None:
    for lora_module in iter_vision_lora_modules(module):
        lora_module.train(training)


def set_vision_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for lora_module in iter_vision_lora_modules(module):
        lora_module.enabled = bool(enabled)
