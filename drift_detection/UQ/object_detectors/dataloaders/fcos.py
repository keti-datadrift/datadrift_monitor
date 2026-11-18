from dataloaders.core.factory import build_dataset, detection_collate_fn
from dataloaders.core.loader import create_detection_dataloader


def create_dataloader(config, split="train"):
    return create_detection_dataloader(config, split=split, default_aspect_ratio_grouping=True)


__all__ = ["build_dataset", "create_dataloader", "detection_collate_fn"]
