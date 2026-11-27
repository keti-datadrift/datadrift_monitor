from __future__ import annotations

import os
from pathlib import Path

import torch

from .feature_io import ensure_dir
from .prompt_text import TEXT_TEMPLATES_VERSION


def save_idea3_checkpoint(
    path: str | Path,
    *,
    stage: str,
    dataset: str,
    clip_model: str,
    hierarchy: str,
    id_split: str,
    prompt_config: dict,
    positive_state_dict: dict | None = None,
    unknown_state_dict: dict | None = None,
    image_adapter_config: dict | None = None,
    image_adapter_state_dict: dict | None = None,
    vision_lora_config: dict | None = None,
    vision_lora_state_dict: dict | None = None,
    positive_checkpoint: str | None = None,
    metrics: dict | None = None,
    args: dict | None = None,
    training_state: dict | None = None,
    extra_payload: dict | None = None,
) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "stage": stage,
        "dataset": dataset,
        "clip_model": clip_model,
        "hierarchy": hierarchy,
        "id_split": id_split,
        "prompt_config": prompt_config,
        "positive_state_dict": positive_state_dict,
        "unknown_state_dict": unknown_state_dict,
        "image_adapter_config": image_adapter_config,
        "image_adapter_state_dict": image_adapter_state_dict,
        "vision_lora_config": vision_lora_config,
        "vision_lora_state_dict": vision_lora_state_dict,
        "positive_checkpoint": positive_checkpoint,
        "text_templates_version": TEXT_TEMPLATES_VERSION,
        "metrics": metrics or {},
        "args": args or {},
        "training_state": training_state,
    }
    if extra_payload:
        reserved_keys = sorted(set(payload) & set(extra_payload))
        if reserved_keys:
            raise ValueError(
                "extra_payload cannot replace checkpoint fields: "
                + ", ".join(reserved_keys)
            )
        payload.update(extra_payload)
    temporary_path = path.with_name(f".{path.name}.tmp")
    previous_path = path.with_name(f"{path.stem}-previous{path.suffix}")
    try:
        with temporary_path.open("wb") as checkpoint_file:
            torch.save(payload, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

        validated_payload = load_idea3_checkpoint(temporary_path)
        if validated_payload.get("stage") != stage:
            raise RuntimeError(
                "Checkpoint validation failed: "
                f"expected stage={stage}, got {validated_payload.get('stage')}"
            )
        if training_state is not None:
            validated_state = validated_payload.get("training_state")
            if not validated_state:
                raise RuntimeError(
                    "Checkpoint validation failed: resumable training state is missing"
                )
            if validated_state.get("epoch") != training_state.get("epoch"):
                raise RuntimeError(
                    "Checkpoint validation failed: training epoch mismatch"
                )

        keep_previous = training_state is not None
        if keep_previous and path.exists():
            os.replace(path, previous_path)
        try:
            os.replace(temporary_path, path)
        except BaseException:
            if keep_previous and not path.exists() and previous_path.exists():
                os.replace(previous_path, path)
            raise
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_idea3_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def previous_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}-previous{path.suffix}")


def load_idea3_checkpoint_with_fallback(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict, Path]:
    path = Path(path)
    candidates = [path, previous_checkpoint_path(path)]
    errors = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = load_idea3_checkpoint(candidate, map_location=map_location)
            if not payload.get("training_state"):
                raise ValueError("resumable training state is missing")
            return payload, candidate
        except Exception as error:
            errors.append(f"{candidate}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError(
            "No valid resumable checkpoint was found. " + " | ".join(errors)
        )
    raise FileNotFoundError(
        "No resumable checkpoint was found at "
        + " or ".join(str(candidate) for candidate in candidates)
    )
