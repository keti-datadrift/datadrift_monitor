import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import yaml
import numpy as np
import random

from model import VehiInfoRet
from dataloader2 import VehicleDataModule


def train_config(config):
    seed_everything(config["seed"])

    datamodule = VehicleDataModule(
        data_root=config["data"]["data_root"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    model = VehiInfoRet(
        pretrained=config["training"]["pretrained"],
        model_name=config["training"]["model_name"],
        exemplar_k=config["training"]["exemplar_k"],
        knn=config["training"]["knn"],
        loss=config["training"]["loss"],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        dist_scale=config["training"]["dist_scale"],
        dist_pow=config["training"]["dist_pow"],
    )

    save_dir = Path(config["logging"]["log_dir"])
    exp_name = config["logging"]["exp_name"]
    loss_name = config["training"]["loss"]
    
    root_dir = save_dir / exp_name
    existing_versions = []
    
    if root_dir.exists():
        for d in root_dir.iterdir():
            if d.is_dir() and d.name.startswith("version_"):
                try:
                    ver_num = int(d.name.split("-")[0].split("_")[1])
                    existing_versions.append(ver_num)
                except (IndexError, ValueError):
                    continue
    
    next_ver = max(existing_versions) + 1 if existing_versions else 0
    version_name = f"version_{next_ver}-{loss_name}"

    logger = TensorBoardLogger(
        config["logging"]["log_dir"], name=exp_name, version=version_name
    )

    checkpoint_callback_loss = ModelCheckpoint(
        monitor="val_loss",
        filename="best-loss-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    checkpoint_callback_map = ModelCheckpoint(
        monitor="val_map_r",
        filename="best-map-{epoch:02d}-{val_map_r:.4f}",
        save_top_k=1,
        mode="max",
    )
    checkpoint_callback_prec = ModelCheckpoint(
        monitor="val_prec1",
        filename="best-prec-{epoch:02d}-{val_prec1:.4f}",
        save_top_k=1,
        mode="max",
    )

    trainer = Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else "auto",
        logger=logger,
        callbacks=[checkpoint_callback_loss, checkpoint_callback_map, checkpoint_callback_prec],
        log_every_n_steps=10,
    )

    print("Starting model training...")
    trainer.fit(model, datamodule)
    print("Training finished.")

    save_path = config["training"]["save_path"]
    save_dir = Path(save_path).parent
    exemplar_save_path = save_dir / "exemplar_set.pt"

    if checkpoint_callback_prec.best_model_path:
        print(f"Saving the best feature extractor weights (based on Prec@1) to {save_path}")
        best_model = VehiInfoRet.load_from_checkpoint(checkpoint_callback_prec.best_model_path)
        torch.save(best_model.feature_extractor.state_dict(), save_path)

        if (
            best_model.exemplar_features is not None
            and best_model.exemplar_labels is not None
        ):
            print(f"Saving the corresponding exemplar set to {exemplar_save_path}")
            torch.save(
                {
                    "exemplar_features": best_model.exemplar_features,
                    "exemplar_labels": best_model.exemplar_labels,
                },
                exemplar_save_path,
            )
        else:
            print("Warning: Exemplar set not found in the best model. Skipping save.")
            
    print("Script finished successfully.")
    
    return trainer.log_dir


if __name__ == "__main__":
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        exit()
    train_config(config)
