from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ImageAdapterConfig:
    mode: str = "none"
    output_dim: int | None = None
    hidden_dim: int = 1024
    dropout: float = 0.0
    rank: int = 16
    alpha: float = 16.0
    residual: bool = True
    normalize_output: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "ImageAdapterConfig":
        data = dict(data or {})
        return cls(
            mode=str(data.get("mode", "none")).lower(),
            output_dim=data.get("output_dim"),
            hidden_dim=int(data.get("hidden_dim", 1024)),
            dropout=float(data.get("dropout", 0.0)),
            rank=int(data.get("rank", 16)),
            alpha=float(data.get("alpha", 16.0)),
            residual=bool(data.get("residual", True)),
            normalize_output=bool(data.get("normalize_output", True)),
        )

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "rank": self.rank,
            "alpha": self.alpha,
            "residual": self.residual,
            "normalize_output": self.normalize_output,
        }


class ImageFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, cfg: ImageAdapterConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cfg = cfg
        self.mode = cfg.mode
        self.output_dim = int(cfg.output_dim or input_dim)

        if self.mode in {"none", "identity", "no_adaptation"}:
            if self.output_dim != self.input_dim:
                raise ValueError("image_adapter.output_dim must match input_dim when mode is none")
            self.net = nn.Identity()
        elif self.mode == "linear":
            self.net = nn.Linear(self.input_dim, self.output_dim)
        elif self.mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, self.output_dim),
            )
        elif self.mode == "lora":
            if self.output_dim != self.input_dim:
                raise ValueError("feature-space lora adapter requires output_dim == input_dim")
            rank = max(1, min(int(cfg.rank), self.input_dim))
            self.dropout = nn.Dropout(cfg.dropout)
            self.lora_a = nn.Linear(self.input_dim, rank, bias=False)
            self.lora_b = nn.Linear(rank, self.input_dim, bias=False)
            self.scaling = float(cfg.alpha) / float(rank)
            nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
            nn.init.zeros_(self.lora_b.weight)
            self.net = None
        else:
            raise ValueError(
                "image_adapter.mode must be one of: none, linear, mlp, lora. "
                f"Got {cfg.mode!r}"
            )

    @property
    def is_identity(self) -> bool:
        return self.mode in {"none", "identity", "no_adaptation"}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        if self.mode == "lora":
            delta = self.lora_b(self.lora_a(self.dropout(features))) * self.scaling
            out = features + delta if self.cfg.residual else delta
        else:
            out = self.net(features)
            if self.cfg.residual and self.mode in {"linear", "mlp"} and out.shape[-1] == features.shape[-1]:
                out = out + features

        if self.cfg.normalize_output:
            out = F.normalize(out, dim=-1)
        return out


def build_image_adapter(input_dim: int, cfg_dict: dict | None) -> ImageFeatureAdapter:
    cfg = ImageAdapterConfig.from_dict(cfg_dict)
    return ImageFeatureAdapter(input_dim, cfg)
