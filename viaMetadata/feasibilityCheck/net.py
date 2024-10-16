import os
import torch

import lightning as L

import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
import torchmetrics as tm

import torch.optim.lr_scheduler as LRS


import numpy as np


class Detector(L.LightningModule):

    def __init__(
        self,
        classifier,
        learning_rate,
        scheduler_opt=None,
    ):
        super().__init__()
        self.save_hyperparameters(
            "learning_rate",
            "scheduler_opt",
        )
        self.classifier = classifier

        self.learning_rate = learning_rate
        self.schedulerOpt = scheduler_opt
        self.main_metric = scheduler_opt["monitor"]

        self.classifier.eval()
        for param in self.classifier.parameters():
            param.requires_grad = False

        self.detector = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 1),
        )

        self.acc = tm.Accuracy(task="binary")
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        with torch.no_grad():
            outputs = self.classifier(x)
        logits = outputs.logits
        pred = self.detector(logits)
        return pred

    def training_step(self, batch, batch_idx):
        x, label = batch
        pred = self(x)
        loss = self.loss(pred, label)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        pred = self(x)
        acc = self.acc(pred, label)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr (x1e4)", lr * 1e4, prog_bar=True, logger=True)
        self.log(self.main_metric, acc * 100, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": LRS.ReduceLROnPlateau(
                optimizer,
                mode=self.schedulerOpt["mode"],
                factor=self.schedulerOpt["factor"],
                patience=self.schedulerOpt["patience"],
                threshold=self.schedulerOpt["threshold"],
                threshold_mode=self.schedulerOpt["threshold_mode"],
            ),
            "monitor": self.schedulerOpt["monitor"],
            "interval": self.schedulerOpt["interval"],
            "frequency": self.schedulerOpt["frequency"],
        }
        return [optimizer], [scheduler]
