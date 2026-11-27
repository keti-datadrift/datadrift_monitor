from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from negzerohoc.config_utils import load_yaml_config
from negzerohoc.output_layout import experiment_artifact_path, shared_features_root


LEGACY_PREFIX_ALIASES = {
    "idea3-fgvc-aircraft-weihims": "idea3-fgvc-aircraft-weihims",
    "fgvc-aircraft-clip-b16-feature-knn": "fgvc-aircraft-clip-b16-feature-knn",
    "fgvc-aircraft-clip-feature-knn": "fgvc-aircraft-clip-feature-knn",
    "fgvc-aircraft-clip-feature-probe": "fgvc-aircraft-clip-leaf-probe",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _assert_within(path: Path, root: Path) -> None:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing path outside output root: {resolved_path}") from exc


def experiment_names(config_root: Path) -> list[str]:
    names = set(LEGACY_PREFIX_ALIASES.values())
    for path in config_root.rglob("*.yaml"):
        cfg = load_yaml_config(path)
        name = cfg.get("experiment", {}).get("name")
        if name:
            names.add(str(name))
    return sorted(names, key=len, reverse=True)


def classify_experiment(filename: str, names: list[str]) -> str | None:
    for prefix, experiment_name in sorted(
        LEGACY_PREFIX_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if filename.startswith(prefix):
            return experiment_name
    return next((name for name in names if filename.startswith(name)), None)


def artifact_moves(output_root: Path, names: list[str]) -> list[tuple[Path, Path]]:
    moves = []
    unmatched = []
    for kind in ("checkpoints", "results", "diagnostics"):
        legacy_dir = output_root / kind
        if not legacy_dir.exists():
            continue
        for source in sorted(legacy_dir.iterdir()):
            if not source.is_file():
                unmatched.append(source)
                continue
            experiment_name = classify_experiment(source.name, names)
            if experiment_name is None:
                unmatched.append(source)
                continue
            target = experiment_artifact_path(
                output_root,
                experiment_name,
                kind,
                source.name,
            )
            moves.append((source, target))
    if unmatched:
        formatted = "\n".join(f"- {path}" for path in unmatched)
        raise RuntimeError(f"Unmapped output artifacts; no files were moved:\n{formatted}")
    return moves


def feature_moves(output_root: Path) -> list[tuple[Path, Path]]:
    legacy_root = output_root / "features"
    if not legacy_root.exists():
        return []
    target_root = shared_features_root(output_root)
    return [
        (source, target_root / source.relative_to(legacy_root))
        for source in sorted(legacy_root.rglob("*"))
        if source.is_file()
    ]


def rewrite_config(path: Path, experiment_name: str, output_root: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    updated = text.replace(
        f"{output_root.as_posix()}/features/",
        f"{output_root.as_posix()}/shared/features/",
    )
    for kind in ("checkpoints", "results", "diagnostics"):
        updated = updated.replace(
            f"{output_root.as_posix()}/{kind}/",
            f"{output_root.as_posix()}/experiments/{experiment_name}/{kind}/",
        )
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def remove_empty_legacy_dirs(output_root: Path) -> None:
    legacy_features = output_root / "features"
    if legacy_features.exists():
        for path in sorted(legacy_features.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        if not any(legacy_features.iterdir()):
            legacy_features.rmdir()
    for kind in ("checkpoints", "results", "diagnostics"):
        path = output_root / kind
        if path.exists() and not any(path.iterdir()):
            path.rmdir()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    config_root = Path(args.config_root)
    names = experiment_names(config_root)
    moves = artifact_moves(output_root, names) + feature_moves(output_root)

    for source, target in moves:
        _assert_within(source, output_root)
        _assert_within(target, output_root)
        if target.exists():
            raise FileExistsError(f"Migration target already exists: {target}")
        print(f"{source} -> {target}")

    config_changes = []
    for path in sorted(config_root.rglob("*.yaml")):
        cfg = load_yaml_config(path)
        name = cfg.get("experiment", {}).get("name")
        if name:
            config_changes.append((path, str(name)))

    print(f"planned file moves: {len(moves)}")
    print(f"configs checked for path updates: {len(config_changes)}")
    if not args.apply:
        print("dry run only; pass --apply to execute")
        return

    manifest = []
    for source, target in moves:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.replace(target)
        manifest.append({"from": str(source), "to": str(target)})

    rewritten = 0
    for path, name in config_changes:
        rewritten += int(rewrite_config(path, name, output_root))

    remove_empty_legacy_dirs(output_root)
    manifest_path = output_root / "migration_manifest.json"
    manifest_path.write_text(
        json.dumps({"moves": manifest, "rewritten_configs": rewritten}, indent=2),
        encoding="utf-8",
    )
    print(f"moved files: {len(manifest)}")
    print(f"rewritten configs: {rewritten}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
