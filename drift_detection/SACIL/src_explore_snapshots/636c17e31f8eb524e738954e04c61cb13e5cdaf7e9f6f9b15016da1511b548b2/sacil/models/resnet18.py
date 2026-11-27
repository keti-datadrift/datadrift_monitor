from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + self.shortcut(inputs), inplace=True)


class CifarResNet18(nn.Module):
    """CIFAR-stem ResNet-18 used by TaKP-style CIFAR experiments."""

    output_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_stage(64, blocks=2, stride=1)
        self.layer2 = self._make_stage(128, blocks=2, stride=2)
        self.layer3 = self._make_stage(256, blocks=2, stride=2)
        self.layer4 = self._make_stage(512, blocks=2, stride=2)
        self._initialize()

    def _make_stage(
        self, out_channels: int, *, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(
                BasicBlock(
                    self.in_channels, out_channels, block_stride
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = F.adaptive_avg_pool2d(outputs, output_size=1)
        return torch.flatten(outputs, 1)


def resnet18() -> CifarResNet18:
    return CifarResNet18()
