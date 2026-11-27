from __future__ import annotations

from collections.abc import Mapping

from sacil.engine.evaluator import EvaluationResult


class CILMetricsTracker:
    def __init__(self, records: list[dict] | None = None) -> None:
        self.records = list(records or [])

    def update(
        self, session_id: int, evaluation: EvaluationResult
    ) -> dict:
        record = {
            "session_id": int(session_id),
            **evaluation.to_dict(),
        }
        self.records.append(record)
        return record

    @property
    def average_incremental_accuracy(self) -> float:
        if not self.records:
            return 0.0
        return sum(record["accuracy"] for record in self.records) / len(
            self.records
        )

    @property
    def final_accuracy(self) -> float | None:
        return None if not self.records else self.records[-1]["accuracy"]

    def average_forgetting(self) -> float:
        if len(self.records) < 2:
            return 0.0
        final_per_class = self.records[-1]["per_class_accuracy"]
        forgetting = []
        for class_id, final_value in final_per_class.items():
            previous = [
                record["per_class_accuracy"][class_id]
                for record in self.records[:-1]
                if class_id in record["per_class_accuracy"]
            ]
            if previous:
                forgetting.append(max(previous) - final_value)
        return 0.0 if not forgetting else sum(forgetting) / len(forgetting)

    def summary(self) -> dict:
        return {
            "average_incremental_accuracy": self.average_incremental_accuracy,
            "final_accuracy": self.final_accuracy,
            "average_forgetting": self.average_forgetting(),
            "num_sessions_completed": len(self.records),
        }

    def state_dict(self) -> dict:
        return {"records": self.records}

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "CILMetricsTracker":
        records = []
        for record in state.get("records", []):
            normalized = dict(record)
            normalized["per_class_accuracy"] = {
                str(key): float(value)
                for key, value in normalized["per_class_accuracy"].items()
            }
            records.append(normalized)
        return cls(records)

