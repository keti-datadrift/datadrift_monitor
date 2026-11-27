from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg.setdefault("_meta", {})
    cfg["_meta"]["config_path"] = str(config_path.resolve())
    return cfg


def load_merged_config(
    config_path: str | Path,
    feature_gen_config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    if feature_gen_config_path is None:
        config["_meta"]["feature_gen_config_path"] = None
        return config

    feature_gen_config_path = Path(feature_gen_config_path)
    with feature_gen_config_path.open("r", encoding="utf-8") as handle:
        fg_cfg = yaml.safe_load(handle) or {}
    merged = apply_overrides(config, fg_cfg)
    merged.setdefault("_meta", {})
    merged["_meta"]["config_path"] = str(Path(config_path).resolve())
    merged["_meta"]["feature_gen_config_path"] = str(feature_gen_config_path.resolve())
    return merged


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in overrides.items() if v is not None}
    result = deepcopy(config)
    return _deep_update(result, clean)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
