from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def compute_prototypes(
    features: Tensor,
    original_targets: Tensor,
    class_ids: Sequence[int],
) -> Tensor:
    if features.ndim != 2:
        raise ValueError("features must have shape [N, D]")
    if features.shape[0] != original_targets.numel():
        raise ValueError("feature and target counts do not match")
    normalized = F.normalize(features.float(), dim=1)
    prototypes = []
    for class_id in class_ids:
        mask = original_targets.long() == int(class_id)
        if not bool(mask.any()):
            raise ValueError(f"no features found for class {class_id}")
        prototype = F.normalize(
            normalized[mask].mean(dim=0, keepdim=True), dim=1
        )[0]
        prototypes.append(prototype)
    return torch.stack(prototypes, dim=0)


class PrototypeBank:
    def __init__(self, class_ids: Sequence[int], prototypes: Tensor) -> None:
        self.class_ids = tuple(int(value) for value in class_ids)
        self.prototypes = prototypes.detach().cpu().float().clone()
        if self.prototypes.ndim != 2:
            raise ValueError("prototypes must be a matrix")
        if len(self.class_ids) != self.prototypes.shape[0]:
            raise ValueError("class ID and prototype counts do not match")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise ValueError("prototype class IDs must be unique")
        norms = self.prototypes.norm(dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
            raise ValueError("prototypes must be normalized")
        self._position = {
            class_id: position
            for position, class_id in enumerate(self.class_ids)
        }

    def for_class(self, class_id: int) -> Tensor:
        return self.prototypes[self._position[int(class_id)]]

    def state_dict(self) -> dict:
        return {
            "class_ids": list(self.class_ids),
            "prototypes": self.prototypes.clone(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "PrototypeBank":
        return cls(state["class_ids"], state["prototypes"])

