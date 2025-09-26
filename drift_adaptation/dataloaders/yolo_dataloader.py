from ultralytics.data import build_dataloader
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from .datasets.yolo_dataset import build_dataset

def get_dataloader(dataset_path, batch_size, args, data, model, rank=-1, mode="train"):
    assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
    with torch_distributed_zero_first(rank):
        dataset = build_dataset(args, dataset_path, mode, batch_size, data, model)
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = args.workers if mode == "train" else args.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)
