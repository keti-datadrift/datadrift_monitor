from __future__ import annotations

from pathlib import Path
from typing import Any

from collections.abc import Callable

Runner = Callable[[dict[str, Any], Path], Path]


def resolve_runner(task: str, mode: str) -> Runner:
    normalized_task = str(task).strip().lower()
    normalized_mode = str(mode).strip().lower()
    if normalized_task == "meta_classifier":
        if normalized_mode == "train":
            from meta_models.commands.meta_classifier.train import run_train as run_classifier_train

            return run_classifier_train
        if normalized_mode == "test":
            from meta_models.commands.meta_classifier.test import run_test as run_classifier_test

            return run_classifier_test
        if normalized_mode == "compare":
            from meta_models.commands.meta_classifier.compare import run_compare as run_classifier_compare

            return run_classifier_compare
        raise ValueError(f"Unsupported meta_classifier mode: {mode}")
    if normalized_task == "meta_regressor":
        if normalized_mode == "train":
            from meta_models.commands.meta_regressor.train import run_train as run_regressor_train

            return run_regressor_train
        if normalized_mode == "test":
            from meta_models.commands.meta_regressor.test import run_test as run_regressor_test

            return run_regressor_test
        if normalized_mode == "compare":
            raise ValueError("mode='compare' is only supported for task='meta_classifier'.")
        raise ValueError(f"Unsupported meta_regressor mode: {mode}")
    raise ValueError(f"Unsupported meta model task: {task}")
