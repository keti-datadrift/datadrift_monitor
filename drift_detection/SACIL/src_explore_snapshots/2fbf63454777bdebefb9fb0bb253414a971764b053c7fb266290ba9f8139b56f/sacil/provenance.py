from __future__ import annotations

import hashlib
from pathlib import Path


def _python_manifest(root: Path) -> dict[str, str]:
    root = root.resolve()
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _manifest_digest(manifest: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_digest in sorted(manifest.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_exploration_provenance(
    source_root: str | Path,
    preserved_source_root: str | Path,
) -> dict:
    """Fingerprint exploration Python sources against the preserved tree."""

    source = Path(source_root).expanduser().resolve()
    preserved = Path(preserved_source_root).expanduser().resolve()
    source_manifest = _python_manifest(source)
    preserved_manifest = _python_manifest(preserved)
    changed_files = []
    for relative in sorted(set(source_manifest) | set(preserved_manifest)):
        current = source_manifest.get(relative)
        baseline = preserved_manifest.get(relative)
        if current == baseline:
            continue
        status = "added" if baseline is None else "deleted" if current is None else "modified"
        changed_files.append(
            {
                "path": relative,
                "status": status,
                "source_sha256": current,
                "preserved_sha256": baseline,
            }
        )
    return {
        "algorithm": "sha256",
        "scope": "python_sources_only",
        "source_root": str(source),
        "source_digest": _manifest_digest(source_manifest),
        "source_file_count": len(source_manifest),
        "preserved_source_root": str(preserved),
        "preserved_src_digest": _manifest_digest(preserved_manifest),
        "preserved_file_count": len(preserved_manifest),
        "changed_files": changed_files,
    }
