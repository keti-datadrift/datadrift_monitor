"""PyCIL's CIFAR ResNet-32, exposed through the SACIL model contract.

The convolutional implementation below is kept structurally identical to
``ref_codes/00_frameworks/PyCIL/convs/cifar_resnet.py``.  Only the thin
``forward_detailed``/tensor-returning ``forward`` adapter is added so the
in-repo runner can consume the official feature extractor without executing
the reference checkout.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DownsampleA(nn.Module):
    def __init__(self, n_in: int, n_out: int, stride: int) -> None:
        super().__init__()
        if stride != 2:
            raise ValueError("PyCIL DownsampleA requires stride=2")
        if n_out != 2 * n_in:
            raise ValueError("PyCIL DownsampleA expects doubled channels")
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.avg(inputs)
        return torch.cat((outputs, outputs.mul(0)), dim=1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = F.relu(self.bn_a(self.conv_a(inputs)), inplace=True)
        outputs = self.bn_b(self.conv_b(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return F.relu(residual + outputs, inplace=True)


class CifarResNet(nn.Module):
    """CIFAR ResNet used by stock PyCIL ``convnet_type=resnet32``."""

    def __init__(self, block: type[ResNetBasicblock], depth: int, channels: int = 3) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must be 6n+2")
        layer_blocks = (depth - 2) // 6
        self.conv_1_3x3 = nn.Conv2d(
            channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(16)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.output_dim = self.out_dim
        # Present in the official convnet even though BaseNet uses an external
        # incremental classifier and never calls this layer in ``forward``.
        self.fc = nn.Linear(self.out_dim, 10)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def _make_layer(
        self,
        block: type[ResNetBasicblock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(
                self.inplanes, planes * block.expansion, stride
            )
        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_detailed(self, inputs: Tensor) -> dict[str, Tensor | tuple[Tensor, ...]]:
        outputs = F.relu(self.bn_1(self.conv_1_3x3(inputs)), inplace=True)
        map1 = self.stage_1(outputs)
        map2 = self.stage_2(map1)
        map3 = self.stage_3(map2)
        pooled = self.avgpool(map3)
        features = pooled.view(pooled.size(0), -1)
        return {"fmaps": (map1, map2, map3), "features": features}

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_detailed(inputs)["features"]  # type: ignore[return-value]

    @property
    def last_conv(self) -> nn.Conv2d:
        return self.stage_3[-1].conv_b


def resnet32() -> CifarResNet:
    return CifarResNet(ResNetBasicblock, 32)
