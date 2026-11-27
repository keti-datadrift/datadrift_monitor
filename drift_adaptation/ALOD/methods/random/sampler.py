"""Random active learning sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from methods.common.coco_pool import image_ids, load_coco_pool
from methods.common.selection import deterministic_random_sample

PoolInput = Union[Path, Dict[str, Any]]


def sample(pool: PoolInput, budget: int, seed: int = 0) -> List[Any]:
    """Return a deterministic random subset of image ids from a COCO pool."""

    return deterministic_random_sample(image_ids(load_coco_pool(pool)), budget, seed=seed)
