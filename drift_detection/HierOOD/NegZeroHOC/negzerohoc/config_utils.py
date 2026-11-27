from __future__ import annotations

from pathlib import Path

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.get("base_config")
    if not base_path:
        return cfg

    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base_cfg = load_yaml_config(base_path)
    return _deep_update(base_cfg, cfg)
