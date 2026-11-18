"""PAL image embedding preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from methods.pal.embeddings import write_embedding_cache
from methods.pal.vit_embeddings import (
    DEFAULT_GOOGLE_VIT_MODEL,
    extract_vit_embeddings,
    metadata_payload,
    missing_image_paths,
    read_coco_image_paths,
    write_metadata,
)


def default_metadata_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + '.meta.json')


def validate_cache_output_path(output_path: Path) -> None:
    if Path(output_path).suffix.lower() not in ('.npy', '.json'):
        raise ValueError('PAL embedding cache output must use .npy or .json: %s'
                         % output_path)


def build_pal_vit_embedding_cache(
    annotation_paths: Sequence[Path],
    image_root: Path,
    output_path: Path,
    metadata_output: Optional[Path] = None,
    model_name: str = DEFAULT_GOOGLE_VIT_MODEL,
    batch_size: int = 16,
    device: str = 'auto',
    embedding_output: str = 'pooler',
    normalize: bool = True,
    skip_missing: bool = False,
    max_images: Optional[int] = None,
    progress: bool = False,
) -> Dict[str, object]:
    validate_cache_output_path(output_path)
    metadata_path = metadata_output or default_metadata_path(output_path)

    records = read_coco_image_paths(annotation_paths, image_root)
    missing = missing_image_paths(records)
    if missing and not skip_missing:
        first = '\n'.join(str(path) for path in missing[:10])
        raise RuntimeError(
            'Missing %d image file(s) for PAL embeddings. First paths:\n%s'
            % (len(missing), first))
    if missing and skip_missing:
        missing_set = set(missing)
        records = [record for record in records if record.path not in missing_set]

    if max_images is not None:
        if max_images <= 0:
            raise ValueError('max_images must be positive')
        records = records[:max_images]
    if not records:
        raise RuntimeError('No images available for PAL embedding extraction')

    embeddings = extract_vit_embeddings(
        records,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        embedding_output=embedding_output,
        normalize=normalize,
        progress=progress,
    )
    write_embedding_cache(embeddings, output_path)

    first_vector = next(iter(embeddings.values()))
    payload = metadata_payload(
        records=records,
        annotation_paths=annotation_paths,
        image_root=image_root,
        output_path=output_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        embedding_output=embedding_output,
        normalize=normalize,
        embedding_dim=len(first_vector),
        skipped_missing=len(missing) if skip_missing else 0,
    )
    write_metadata(payload, metadata_path)
    return {
        'component': 'pal_embeddings',
        'type': 'google_vit',
        'status': 'ready',
        'action': 'created',
        'path': str(output_path),
        'metadata_path': str(metadata_path),
        'image_count': len(embeddings),
        'embedding_dim': len(first_vector),
    }


def ensure_pal_embeddings(
    embedding_cfg: Mapping[str, object],
    annotation_paths: Iterable[Path],
    image_root: Path,
    root: Path,
) -> Dict[str, object]:
    kind = str(embedding_cfg.get('type', '')).lower()
    if kind != 'google_vit':
        raise ValueError('Unsupported PAL embedding preparation type: %s'
                         % embedding_cfg.get('type'))
    output_value = embedding_cfg.get('output_path')
    if not output_value:
        raise ValueError('PAL embedding preparation requires output_path')
    output_path = Path(str(output_value))
    if not output_path.is_absolute():
        output_path = Path(root) / output_path
    metadata_output = embedding_cfg.get('metadata_output')
    metadata_path = Path(str(metadata_output)) if metadata_output else None
    if metadata_path is not None and not metadata_path.is_absolute():
        metadata_path = Path(root) / metadata_path

    validate_cache_output_path(output_path)
    if output_path.exists():
        return {
            'component': 'pal_embeddings',
            'type': 'google_vit',
            'status': 'ready',
            'action': 'kept',
            'path': str(output_path),
            'metadata_path': str(metadata_path or default_metadata_path(output_path)),
        }

    return build_pal_vit_embedding_cache(
        annotation_paths=list(annotation_paths),
        image_root=image_root,
        output_path=output_path,
        metadata_output=metadata_path,
        model_name=str(embedding_cfg.get('model_name', DEFAULT_GOOGLE_VIT_MODEL)),
        batch_size=int(embedding_cfg.get('batch_size', 16)),
        device=str(embedding_cfg.get('device', 'auto')),
        embedding_output=str(embedding_cfg.get('embedding_output', 'pooler')),
        normalize=bool(embedding_cfg.get('normalize', True)),
        skip_missing=bool(embedding_cfg.get('skip_missing', False)),
        max_images=(
            int(embedding_cfg['max_images'])
            if embedding_cfg.get('max_images') is not None
            else None
        ),
        progress=bool(embedding_cfg.get('progress', False)),
    )
