import torch
from torchvision import models
import pytorch_lightning as pl

class CifarModel(pl.LightningModule):
    def __init__(self):
        super(CifarModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)
    
class DriftModel(pl.LightningModule):
    def __init__(self):
        super(DriftModel, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)