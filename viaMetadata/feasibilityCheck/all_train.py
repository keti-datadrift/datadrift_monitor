import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms


import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from lightning.pytorch.callbacks.progress.rich_progress import (
    RichProgressBar,
    RichProgressBarTheme,
)

import glob
import shutil

from net import Detector
import numpy as np

import gc

from datetime import datetime

from rich.console import Console


from transformers import AutoModelForImageClassification


DEBUGGING = True
DEBUGGING = False


# hyperparameters
BATCH_SIZE = 64
MODEL_NAME = "Detector"

LEARNING_RATE = 1e-4
LR_FACTOR = 0.1
LR_PATIENCE = 4
STOP_PATIENCE = 8

EVAL_DICT = {
    "metric": "accuracy",
    "mode": "max",
    "min_delta": 0.1,
    "valid_step": 500,
}


SCHEDULER_DICT = None
SCHEDULER_DICT = {
    "scheduler": "ReduceLROnPlateau",
    "mode": EVAL_DICT["mode"],
    "factor": LR_FACTOR,
    "patience": LR_PATIENCE,
    "threshold": EVAL_DICT["min_delta"],
    "threshold_mode": "abs",
    "monitor": EVAL_DICT["metric"],
    "interval": "step",
    "frequency": EVAL_DICT["valid_step"],
}


MODEL_LIST = [
    "microsoft/resnet-50",
    "microsoft/resnet-101",
    "microsoft/resnet-152",
    "WinKawaks/vit-small-patch16-224",
    "google/vit-base-patch16-224",
    "google/vit-large-patch16-224",
]


class CustomProgressBar(RichProgressBar):

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


class AddGaussianNoise:
    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std
        return tensor + noise


class DatasetWithLabel(Dataset):
    def __init__(self, root_dir, label, transform):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.images = glob.glob(f"{root_dir}\\*.JPEG")
        # self.images = glob.glob(f"{root_dir}\\*.png")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.label, dtype=torch.float32).unsqueeze(0)
        image = self.transform(image)

        return image, label


def train(model_name, version, eval_dict, train_loader, valid_loader, network):

    console_temp = Console()
    console_temp.print(f"\n[bold red]{version}[/bold red] [bold]is now training[/bold]")
    print(f"is now training ...")

    monitor_metric = eval_dict["metric"]
    monitor_mode = eval_dict["mode"]

    if DEBUGGING:
        limit_train = 50
        limit_val = 10
        val_check = 10
        log_step = 10

    else:
        limit_train = None
        limit_val = 200
        val_check = eval_dict["valid_step"]
        log_step = val_check // 10

    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=model_name,
        version=version,
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=eval_dict["min_delta"],
        patience=STOP_PATIENCE,
        verbose=False,
        mode=monitor_mode,
        check_on_train_epoch_end=False,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        monitor=monitor_metric,
        mode=monitor_mode,
        filename="{epoch},{step}," + f"{{{monitor_metric}:.3f}}",
    )

    # progressbar_callback = RichProgressBar(
    progressbar_callback = CustomProgressBar(
        theme=RichProgressBarTheme(
            progress_bar="bold green",
            progress_bar_finished="bold green",
            batch_progress="bold cyan",
            time="bold magenta",
            processing_speed="bold red",
            metrics="bold yellow",
            metrics_text_delimiter=" | ",
            metrics_format=".3f",
        )
    )

    trainer = L.Trainer(
        logger=logger,
        benchmark=True,  # speed up if input sizes don't change.
        # check_val_every_n_epoch=1,
        val_check_interval=val_check,  # val check for every INT/FLOAT batch/epoch. use tpgether with check_val_every_n_epoch=None
        limit_train_batches=limit_train,
        limit_val_batches=limit_val,  # How much of validation dataset to check.
        log_every_n_steps=log_step,  # does not write to disk.
        max_epochs=-1,  # To enable infinite training, set max_epochs = -1.
        # max_time="00:01:00:00", # use together with min_steps or min_epochs.
        num_sanity_val_steps=0,  # runs n batches of val before starting the training routine. The Trainer uses 2 steps by default.
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            progressbar_callback,
        ],  # callbacks
    )

    trainer.fit(network, train_loader, valid_loader)

    del network
    del trainer
    del logger
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    console = Console()

    L.seed_everything(4758)

    # Tensor Core 있으면 medium 적용해서 좀 더 빠르게 학습시킬 수 있음
    torch.set_float32_matmul_precision("high")

    # 현재 시간을 얻습니다
    now = datetime.now()

    # 시간을 원하는 형식으로 출력합니다
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    console.print("#" * 79, style="bold yellow")
    console.print(f" Start Time: {current_time}", style="bold yellow")
    console.print("#" * 79, style="bold yellow")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    transform_drift = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(0.1),
        ]
    )

    train_dataset_list = []
    for i in range(5):
        train_folder = os.path.join("imageNet", f"train_images_{i}")
        train_dataset_list.append(DatasetWithLabel(train_folder, 0, transform))
        train_dataset_list.append(DatasetWithLabel(train_folder, 1, transform_drift))
    train_dataset = ConcatDataset(train_dataset_list)

    valid_dataset_list = []
    valid_folder = os.path.join("imageNet", "val_images")
    valid_dataset_list.append(DatasetWithLabel(valid_folder, 0, transform))
    valid_dataset_list.append(DatasetWithLabel(valid_folder, 1, transform_drift))
    valid_dataset = ConcatDataset(valid_dataset_list)

    print("Train:", len(train_dataset))
    print("Test:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=24,
        persistent_workers=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=24,
        persistent_workers=True,
    )

    for pretrained in MODEL_LIST:
        VERSION = pretrained.replace("/", "-")
        classifier = AutoModelForImageClassification.from_pretrained(pretrained)
        detector = Detector(
            classifier=classifier,
            learning_rate=LEARNING_RATE,
            scheduler_opt=SCHEDULER_DICT,
        )

        train(
            model_name=MODEL_NAME,
            version=VERSION,
            eval_dict=EVAL_DICT,
            train_loader=train_loader,
            valid_loader=valid_loader,
            network=detector,
        )
