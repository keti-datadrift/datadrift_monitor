from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], embedding_dim: int):
        super().__init__()
        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return F.normalize(self.proj(x), dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return F.normalize(self.net(x), dim=1)


class CLIPTextEncoder(nn.Module):
    def __init__(self, config: dict, embedding_dim: int):
        super().__init__()
        try:
            from transformers import CLIPTextModel
        except ImportError as exc:
            raise ImportError("clip_text model requires transformers.") from exc

        model_name = config["model"].get("clip_model_name", "openai/clip-vit-base-patch32")
        local_files_only = bool(config["model"].get("local_files_only", False))
        self.text_model = CLIPTextModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        if bool(config["model"].get("freeze_clip", True)):
            for param in self.text_model.parameters():
                param.requires_grad = False
        hidden_size = self.text_model.config.hidden_size
        self.proj = nn.Linear(hidden_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        input_ids = x[:, 0, :]
        attention_mask = x[:, 1, :]
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.proj(out.pooler_output), dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18Encoder(nn.Module):
    def __init__(self, input_shape: tuple[int, ...], embedding_dim: int):
        super().__init__()
        channels = input_shape[0]
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embedding_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return F.normalize(self.proj(x), dim=1)


def build_model(config: dict, input_shape: tuple[int, ...], input_dim: int, input_kind: str):
    embedding_dim = int(config["model"].get("embedding_dim", 128))
    model_type = config["model"].get("type", "auto").lower()
    if model_type == "auto":
        model_type = "cnn" if input_kind == "image" else "mlp"

    if model_type == "cnn":
        return SmallCNN(input_shape, embedding_dim)
    if model_type == "resnet18":
        if input_kind != "image":
            raise ValueError("resnet18 requires image input")
        return ResNet18Encoder(input_shape, embedding_dim)
    if model_type == "mlp":
        return MLPEncoder(input_dim, embedding_dim)
    if model_type == "tfidf_mlp":
        return MLPEncoder(input_dim, embedding_dim)
    if model_type == "clip_text":
        if input_kind != "clip_text":
            raise ValueError("clip_text requires Reuters raw text token input")
        return CLIPTextEncoder(config, embedding_dim)
    raise ValueError(f"Unknown model type: {config['model']['type']}")
