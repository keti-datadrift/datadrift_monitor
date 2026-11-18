"""PAL ViT embedding cache helpers.

This module intentionally avoids importing image/model dependencies at module
import time so tests and PAL samplers do not require PIL, torch, or
transformers. Heavy dependencies are imported only by the extraction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from methods.common.io import read_json, write_json
from methods.common.paths import is_relative_to
from methods.common.vectors import l2_normalize


DEFAULT_GOOGLE_VIT_MODEL = 'google/vit-base-patch16-224-in21k'
EMBEDDING_OUTPUTS = ('pooler', 'cls', 'mean')


@dataclass(frozen=True)
class CocoImagePath:
    image_id: Any
    file_name: str
    path: Path


def resolve_coco_image_path(image_root: Path, file_name: str) -> Path:
    relative_path = Path(file_name)
    if relative_path.is_absolute():
        raise ValueError('COCO file_name must be relative to image_root: %s'
                         % file_name)

    root = Path(image_root).resolve()
    resolved = (root / relative_path).resolve()
    if not is_relative_to(resolved, root):
        raise ValueError('COCO file_name escapes image_root: %s' % file_name)
    return resolved


def read_coco_image_paths(annotation_paths: Sequence[Path], image_root: Path) -> List[CocoImagePath]:
    """Collect unique COCO image ids and image paths from annotation files.

    ``file_name`` values must be relative to ``image_root``. If an image id is
    repeated across pools it must point at the same relative file name.
    """

    if not annotation_paths:
        raise ValueError('At least one COCO annotation JSON is required')

    image_root = Path(image_root)
    by_id: Dict[Any, CocoImagePath] = {}
    order: List[Any] = []
    for annotation_path in annotation_paths:
        annotation_path = Path(annotation_path)
        payload = read_json(annotation_path)

        images = payload.get('images')
        if not isinstance(images, list):
            raise ValueError('COCO annotation JSON must contain an images list: %s'
                             % annotation_path)

        for image in images:
            if not isinstance(image, dict):
                raise ValueError('COCO image entries must be objects: %s'
                                 % annotation_path)
            if 'id' not in image:
                raise ValueError('COCO image entry is missing id: %s' % annotation_path)
            file_name = image.get('file_name')
            if not isinstance(file_name, str) or not file_name:
                raise ValueError('COCO image entry is missing file_name: %s'
                                 % annotation_path)

            image_id = image['id']
            record = CocoImagePath(
                image_id=image_id,
                file_name=file_name,
                path=resolve_coco_image_path(image_root, file_name),
            )
            existing = by_id.get(image_id)
            if existing is not None:
                if existing.file_name != file_name:
                    raise ValueError(
                        'Conflicting file_name values for image_id %s: %s != %s'
                        % (image_id, existing.file_name, file_name))
                continue

            by_id[image_id] = record
            order.append(image_id)

    return [by_id[image_id] for image_id in order]


def missing_image_paths(records: Iterable[CocoImagePath]) -> List[Path]:
    """Return image paths that do not exist, preserving record order."""

    return [record.path for record in records if not record.path.is_file()]


def batched(records: Sequence[CocoImagePath], batch_size: int) -> Iterator[List[CocoImagePath]]:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    for index in range(0, len(records), batch_size):
        yield list(records[index:index + batch_size])


def _dependency_error(package: str, install_hint: str) -> ImportError:
    return ImportError(
        'PAL Google ViT embedding extraction requires %s. Install optional '
        'dependencies with: %s' % (package, install_hint))


def _import_image_dependencies():
    try:
        from PIL import Image
    except ImportError as exc:
        raise _dependency_error('Pillow', 'pip install Pillow') from exc
    return Image


def _import_model_dependencies():
    try:
        import torch
    except ImportError as exc:
        raise _dependency_error('PyTorch', 'install the PPAL/PAL torch environment') from exc

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        try:
            from transformers import AutoFeatureExtractor as AutoImageProcessor
            from transformers import AutoModel
        except ImportError as exc:
            raise _dependency_error(
                'transformers',
                'pip install transformers Pillow',
            ) from exc

    return torch, AutoImageProcessor, AutoModel


def resolve_device(torch_module: Any, requested_device: str) -> Any:
    requested = str(requested_device).lower()
    if requested == 'auto':
        requested = 'cuda' if torch_module.cuda.is_available() else 'cpu'
    return torch_module.device(requested)


def load_rgb_image(path: Path) -> Any:
    Image = _import_image_dependencies()
    with Image.open(path) as image:
        return image.convert('RGB')


def load_google_vit(model_name: str, device: str = 'auto') -> Dict[str, Any]:
    """Load a Hugging Face ViT processor/model pair for embedding extraction."""

    torch, AutoImageProcessor, AutoModel = _import_model_dependencies()
    device_obj = resolve_device(torch, device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device_obj)
    return {
        'torch': torch,
        'processor': processor,
        'model': model,
        'device': device_obj,
    }


def select_output_tensor(outputs: Any, embedding_output: str) -> Any:
    """Select one image-level tensor from Hugging Face ViT outputs."""

    output_name = embedding_output.lower()
    if output_name not in EMBEDDING_OUTPUTS:
        raise ValueError('Unsupported embedding output: %s' % embedding_output)

    last_hidden_state = getattr(outputs, 'last_hidden_state', None)
    if output_name == 'pooler':
        pooler_output = getattr(outputs, 'pooler_output', None)
        if pooler_output is not None:
            return pooler_output
        output_name = 'cls'

    if last_hidden_state is None:
        raise ValueError('ViT model output is missing last_hidden_state')

    if output_name == 'cls':
        return last_hidden_state[:, 0]
    if output_name == 'mean':
        return last_hidden_state.mean(dim=1)
    raise ValueError('Unsupported embedding output: %s' % embedding_output)


def extract_vit_embeddings(
    records: Sequence[CocoImagePath],
    model_name: str = DEFAULT_GOOGLE_VIT_MODEL,
    batch_size: int = 16,
    device: str = 'auto',
    embedding_output: str = 'pooler',
    normalize: bool = True,
    progress: bool = True,
) -> Dict[Any, np.ndarray]:
    """Extract image-level Google ViT embeddings keyed by COCO image id."""

    if embedding_output.lower() not in EMBEDDING_OUTPUTS:
        raise ValueError('Unsupported embedding output: %s' % embedding_output)
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    components = load_google_vit(model_name, device=device)
    torch = components['torch']
    processor = components['processor']
    model = components['model']
    device_obj = components['device']

    embeddings: Dict[Any, np.ndarray] = {}
    total = len(records)
    for batch_index, batch_records in enumerate(batched(records, batch_size), start=1):
        images = [load_rgb_image(record.path) for record in batch_records]
        inputs = processor(images=images, return_tensors='pt')
        inputs = {
            key: value.to(device_obj)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
            tensor = select_output_tensor(outputs, embedding_output)
        vectors = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        for record, vector in zip(batch_records, vectors):
            flat = np.asarray(vector, dtype=np.float32).reshape(-1)
            embeddings[record.image_id] = l2_normalize(flat) if normalize else flat

        if progress:
            done = min(batch_index * batch_size, total)
            print('embedded %d/%d images' % (done, total), flush=True)

    return embeddings


def metadata_payload(
    records: Sequence[CocoImagePath],
    annotation_paths: Sequence[Path],
    image_root: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    device: str,
    embedding_output: str,
    normalize: bool,
    embedding_dim: Optional[int] = None,
    skipped_missing: int = 0,
) -> Mapping[str, Any]:
    return {
        'format': 'pal_vit_embedding_cache_metadata_v1',
        'model_name': model_name,
        'annotation_files': [str(Path(path)) for path in annotation_paths],
        'image_root': str(Path(image_root)),
        'output_path': str(Path(output_path)),
        'image_count': len(records),
        'embedding_dim': embedding_dim,
        'batch_size': int(batch_size),
        'device': str(device),
        'embedding_output': embedding_output,
        'normalized': bool(normalize),
        'skipped_missing_image_count': int(skipped_missing),
    }


def write_metadata(payload: Mapping[str, Any], path: Path) -> None:
    write_json(Path(path), dict(payload), indent=2)
