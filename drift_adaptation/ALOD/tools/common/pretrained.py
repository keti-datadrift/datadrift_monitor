"""Pretrained model preparation helpers."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Dict


RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
RESNET50_SHA256_PREFIX = '19c8e357'
RESNET50_FILENAME = 'resnet50-19c8e357.pth'


def sha256_prefix(path: Path, prefix_length: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:prefix_length]


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(output)


def ensure_resnet50(output_path: Path) -> Dict[str, object]:
    """Ensure the ResNet-50 backbone checkpoint exists and has the expected hash."""

    output = Path(output_path)
    if output.exists():
        actual = sha256_prefix(output, len(RESNET50_SHA256_PREFIX))
        if actual != RESNET50_SHA256_PREFIX:
            raise RuntimeError(
                'Existing file hash prefix mismatch: %s, expected %s for %s'
                % (actual, RESNET50_SHA256_PREFIX, output)
            )
        return {
            'component': 'pretrained',
            'type': 'resnet50',
            'status': 'ready',
            'action': 'kept',
            'path': str(output),
        }

    _download(RESNET50_URL, output)
    actual = sha256_prefix(output, len(RESNET50_SHA256_PREFIX))
    if actual != RESNET50_SHA256_PREFIX:
        raise RuntimeError(
            'Downloaded file hash prefix mismatch: %s, expected %s'
            % (actual, RESNET50_SHA256_PREFIX)
        )
    return {
        'component': 'pretrained',
        'type': 'resnet50',
        'status': 'ready',
        'action': 'downloaded',
        'path': str(output),
    }


def ensure_pretrained(pretrained_cfg: Dict[str, object], root: Path) -> Dict[str, object]:
    kind = str(pretrained_cfg.get('type', '')).lower()
    if kind != 'resnet50':
        raise ValueError('Unsupported pretrained model type: %s' % pretrained_cfg.get('type'))
    output_path = pretrained_cfg.get('output_path')
    if not output_path:
        output_path = 'data/pretrain_models/%s' % RESNET50_FILENAME
    output = Path(str(output_path))
    if not output.is_absolute():
        output = Path(root) / output
    return ensure_resnet50(output.resolve())

