from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseFeatureGenerator(ABC):
    name = "base"

    def __init__(self, num_depths: int):
        self.num_depths = int(num_depths)

    def fit(
        self,
        train_features: torch.Tensor,
        labels_by_depth: dict[int, torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "BaseFeatureGenerator":
        return self

    @abstractmethod
    def transform(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        return {"num_depths": self.num_depths}

    def load_state_dict(self, state: dict[str, Any]) -> "BaseFeatureGenerator":
        self.num_depths = int(state.get("num_depths", self.num_depths))
        return self
