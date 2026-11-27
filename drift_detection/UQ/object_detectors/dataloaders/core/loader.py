from torch.utils.data import DataLoader

from dataloaders.core.factory import build_dataset, detection_collate_fn, sort_dataset_by_aspect_ratio


def is_train_split(split):
    if isinstance(split, (list, tuple)):
        split_values = [str(v).strip().lower() for v in split if str(v).strip()]
        return bool(split_values) and all(v == "train" for v in split_values)
    return str(split).strip().lower() == "train"


def create_detection_dataloader(config, split="train", default_aspect_ratio_grouping=False):
    dataset = build_dataset(config, split=split)
    dl_cfg = config["dataloader"]
    shuffle = dl_cfg["shuffle_train"] if is_train_split(split) else dl_cfg["shuffle_eval"]
    aspect_grouping = dl_cfg.get("aspect_ratio_grouping", default_aspect_ratio_grouping)
    if bool(aspect_grouping) and not bool(shuffle):
        dataset = sort_dataset_by_aspect_ratio(dataset)
    num_workers = int(dl_cfg["num_workers"])
    loader_kwargs = {
        "batch_size": dl_cfg["batch_size"],
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": dl_cfg["pin_memory"],
        "collate_fn": detection_collate_fn,
    }
    if num_workers > 0:
        if "persistent_workers" in dl_cfg:
            loader_kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", False))
        if "prefetch_factor" in dl_cfg and dl_cfg.get("prefetch_factor") is not None:
            loader_kwargs["prefetch_factor"] = int(dl_cfg["prefetch_factor"])
    return DataLoader(dataset, **loader_kwargs)
