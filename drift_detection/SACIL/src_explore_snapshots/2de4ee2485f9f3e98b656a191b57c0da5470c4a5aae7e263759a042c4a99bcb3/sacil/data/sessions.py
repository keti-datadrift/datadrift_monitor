from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class SessionSpec:
    session_id: int
    kind: str
    start: int
    stop: int
    class_ids: tuple[int, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


class ClassOrderProtocol:
    """Validated class order and session slicing contract."""

    def __init__(
        self,
        protocol_id: str,
        dataset: str,
        num_classes: int,
        class_order: Sequence[int],
        session_slices: Sequence[dict[str, int | str]],
        memory_policy: dict | None = None,
    ) -> None:
        self.protocol_id = str(protocol_id)
        self.dataset = str(dataset)
        self.num_classes = int(num_classes)
        self.class_order = tuple(int(value) for value in class_order)
        self.memory_policy = dict(memory_policy or {})
        self._incremental_by_original = {
            original: incremental
            for incremental, original in enumerate(self.class_order)
        }
        self.sessions = tuple(
            SessionSpec(
                session_id=int(item["session_id"]),
                kind=str(item["kind"]),
                start=int(item["start"]),
                stop=int(item["stop"]),
                class_ids=tuple(
                    self.class_order[int(item["start"]) : int(item["stop"])]
                ),
            )
            for item in session_slices
        )
        self._validate()

    @classmethod
    def from_json(cls, path: str | Path) -> "ClassOrderProtocol":
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            protocol_id=payload["protocol_id"],
            dataset=payload["dataset"],
            num_classes=payload["num_classes"],
            class_order=payload["class_order"],
            session_slices=payload["session_slices"],
            memory_policy=payload.get("memory_policy"),
        )

    def _validate(self) -> None:
        if len(self.class_order) != self.num_classes:
            raise ValueError("class order length does not match num_classes")
        if set(self.class_order) != set(range(self.num_classes)):
            raise ValueError("class order must be a permutation of 0..C-1")
        if not self.sessions:
            raise ValueError("at least one session is required")
        cursor = 0
        for expected_id, session in enumerate(self.sessions):
            if session.session_id != expected_id:
                raise ValueError("session IDs must be contiguous from zero")
            if session.start != cursor or session.stop <= session.start:
                raise ValueError("session slices must be contiguous and non-empty")
            cursor = session.stop
        if cursor != self.num_classes:
            raise ValueError("session slices must cover the complete class order")

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    def session(self, session_id: int) -> SessionSpec:
        return self.sessions[int(session_id)]

    def classes_for_session(self, session_id: int) -> tuple[int, ...]:
        return self.session(session_id).class_ids

    def seen_classes(self, session_id: int) -> tuple[int, ...]:
        return self.class_order[: self.session(session_id).stop]

    def old_classes(self, session_id: int) -> tuple[int, ...]:
        return self.class_order[: self.session(session_id).start]

    def incremental_label(self, original_label: int) -> int:
        try:
            return self._incremental_by_original[int(original_label)]
        except KeyError as error:
            raise KeyError(f"unknown original class ID: {original_label}") from error

    def original_label(self, incremental_label: int) -> int:
        return self.class_order[int(incremental_label)]

    def map_original_labels(self, labels: Iterable[int]) -> list[int]:
        return [self.incremental_label(label) for label in labels]

    def session_for_incremental_label(self, label: int) -> int:
        position = int(label)
        for session in self.sessions:
            if session.start <= position < session.stop:
                return session.session_id
        raise KeyError(f"incremental label outside protocol: {label}")

