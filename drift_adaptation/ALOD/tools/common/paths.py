"""Path helpers shared by ALOD tooling."""

from __future__ import annotations

import os
from pathlib import Path

from methods.common.paths import is_relative_to


def assert_not_code_refs(path: Path, root: Path) -> None:
    resolved = Path(path).resolve()
    code_refs = (Path(root) / 'code_refs').resolve()
    if is_relative_to(resolved, code_refs):
        raise ValueError('Refusing to read/write runtime output under code_refs: %s' % path)


def resolve_repo_path(
    value: str,
    root: Path,
    must_be_relative: bool = True,
) -> Path:
    path = Path(value)
    if must_be_relative and path.is_absolute():
        raise ValueError('Config path must be relative to the repo root: %s' % value)
    resolved = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    repo_root = Path(root).resolve()
    if not is_relative_to(resolved, repo_root):
        raise ValueError('Config path must stay inside the repo root: %s' % value)
    assert_not_code_refs(resolved, repo_root)
    return resolved


def resolve_repo_input_path(value: str, root: Path) -> Path:
    """Resolve a repo-relative input while allowing an external junction target."""

    path = Path(value)
    if path.is_absolute():
        raise ValueError('Config input path must be relative to the repo root: %s' % value)
    repo_root = Path(root).resolve()
    lexical = Path(os.path.abspath(str(Path(root) / path)))
    if not is_relative_to(lexical, repo_root):
        raise ValueError('Config input path must start inside the repo root: %s' % value)
    assert_not_code_refs(lexical, repo_root)
    assert_not_code_refs(lexical.resolve(), repo_root)
    return lexical


def display_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
