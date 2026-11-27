from __future__ import annotations

from pathlib import Path


ARTIFACT_KINDS = frozenset({"checkpoints", "results", "diagnostics", "logs"})


def _validate_experiment_name(experiment_name: str) -> str:
    name = str(experiment_name).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid experiment name: {experiment_name!r}")
    return name


def experiment_dir(output_root: str | Path, experiment_name: str) -> Path:
    name = _validate_experiment_name(experiment_name)
    return Path(output_root) / "experiments" / name


def experiment_artifact_dir(
    output_root: str | Path,
    experiment_name: str,
    kind: str,
) -> Path:
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"Unsupported experiment artifact kind: {kind!r}")
    return experiment_dir(output_root, experiment_name) / kind


def experiment_artifact_path(
    output_root: str | Path,
    experiment_name: str,
    kind: str,
    filename: str,
) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Artifact filename must be a basename: {filename!r}")
    return experiment_artifact_dir(output_root, experiment_name, kind) / filename


def resolve_experiment_artifact(
    configured_path: str | Path | None,
    *,
    output_root: str | Path,
    experiment_name: str,
    kind: str,
    default_filename: str,
) -> Path:
    default_path = experiment_artifact_path(
        output_root,
        experiment_name,
        kind,
        default_filename,
    )
    if not configured_path:
        return default_path

    candidate = Path(configured_path)
    legacy_dir = Path(output_root) / kind
    if candidate.parent == legacy_dir:
        return default_path.with_name(candidate.name)
    return candidate


def shared_features_root(output_root: str | Path) -> Path:
    return Path(output_root) / "shared" / "features"


def shared_feature_dir(
    output_root: str | Path,
    dataset: str,
    model_key: str,
) -> Path:
    return shared_features_root(output_root) / dataset / model_key


def resolve_shared_feature_dir(
    configured_path: str | Path | None,
    *,
    output_root: str | Path,
) -> Path | None:
    if not configured_path:
        return None

    candidate = Path(configured_path)
    legacy_root = Path(output_root) / "features"
    try:
        relative = candidate.relative_to(legacy_root)
    except ValueError:
        return candidate
    return shared_features_root(output_root) / relative
