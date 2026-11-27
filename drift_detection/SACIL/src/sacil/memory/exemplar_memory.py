from __future__ import annotations

from collections.abc import Mapping


class ExemplarMemory:
    """Original-dataset indices grouped by original class ID."""

    def __init__(self, exemplars_per_class: int) -> None:
        if exemplars_per_class <= 0:
            raise ValueError("exemplars_per_class must be positive")
        self.exemplars_per_class = int(exemplars_per_class)
        self._indices: dict[int, list[int]] = {}

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._indices))

    def __len__(self) -> int:
        return sum(len(indices) for indices in self._indices.values())

    def indices_for_class(self, class_id: int) -> tuple[int, ...]:
        return tuple(self._indices.get(int(class_id), ()))

    def all_indices(self, class_order: tuple[int, ...] | None = None) -> list[int]:
        order = class_order if class_order is not None else self.class_ids
        result: list[int] = []
        for class_id in order:
            result.extend(self._indices.get(int(class_id), ()))
        return result

    def set_class_indices(self, class_id: int, indices: list[int]) -> None:
        normalized = [int(index) for index in indices]
        if len(normalized) > self.exemplars_per_class:
            raise ValueError("too many exemplars for class")
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate exemplar index")
        self._indices[int(class_id)] = normalized

    def resize_limit(self, exemplars_per_class: int) -> None:
        """Change the per-class limit and deterministically trim old sets.

        This is needed for methods whose official protocol keeps a fixed
        total budget (for example PODNet/AFC with M=2,000).  Existing exemplar
        order is the herding priority order, so retaining the prefix matches
        the reference implementations.
        """

        limit = int(exemplars_per_class)
        if limit <= 0:
            raise ValueError("exemplars_per_class must be positive")
        self.exemplars_per_class = limit
        for class_id in tuple(self._indices):
            self._indices[class_id] = self._indices[class_id][:limit]

    def state_dict(self) -> dict:
        return {
            "exemplars_per_class": self.exemplars_per_class,
            "indices": {
                str(class_id): list(indices)
                for class_id, indices in sorted(self._indices.items())
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "ExemplarMemory":
        memory = cls(int(state["exemplars_per_class"]))
        for class_id, indices in state["indices"].items():
            memory.set_class_indices(int(class_id), list(indices))
        return memory
