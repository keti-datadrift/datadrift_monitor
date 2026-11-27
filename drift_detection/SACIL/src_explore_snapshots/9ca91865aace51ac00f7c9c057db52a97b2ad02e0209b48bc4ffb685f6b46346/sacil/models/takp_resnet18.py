from __future__ import annotations

from dataclasses import dataclass

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
        self.relu = nn.ReLU(inplace=True)
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
            self.downsample: nn.Module | None = nn.Sequential(
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
            self.downsample = None

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(outputs + identity)


class TaKPResNet18Backbone(nn.Module):
    """The released TaKP/BBN dual-branch layout adapted to ResNet-18.

    TaKP publishes both an ImageNet-stem ResNet-18 and a CIFAR-stem
    ResNet-18, but no ResNet-18 dual-branch class.  ``stem`` makes that
    otherwise undocumented choice explicit while keeping the public BBN
    split: the first seven residual blocks are shared and each branch owns
    the final 512-channel residual block.
    """

    branch_feature_dim = 512
    output_dim = 1024

    def __init__(self, *, stem: str = "imagenet") -> None:
        super().__init__()
        if stem not in {"cifar", "imagenet"}:
            raise ValueError(f"unknown TaKP ResNet-18 stem: {stem}")
        self.stem = stem
        if stem == "cifar":
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.pool: nn.Module = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_stage(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_stage(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_stage(128, 256, blocks=2, stride=2)
        # ResNet-18 layer4 has two blocks.  BBN shares the first and splits
        # the last block into the conventional/rebalancing branches.
        self.shared_layer4 = BasicBlock(256, 512, stride=2)
        self.conventional_block = BasicBlock(512, 512, stride=1)
        self.rebalancing_block = BasicBlock(512, 512, stride=1)
        self._initialize()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride=stride)
        ]
        layers.extend(
            BasicBlock(out_channels, out_channels, stride=1)
            for _ in range(blocks - 1)
        )
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

    def shared_features(self, inputs: Tensor) -> Tensor:
        outputs = self.pool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        return self.shared_layer4(outputs)

    @staticmethod
    def _pool(features: Tensor) -> Tensor:
        return torch.flatten(
            F.adaptive_avg_pool2d(features, output_size=1), 1
        )

    def branch_features(self, inputs: Tensor, branch: str) -> Tensor:
        shared = self.shared_features(inputs)
        if branch == "conventional":
            return self._pool(self.conventional_block(shared))
        if branch == "rebalancing":
            return self._pool(self.rebalancing_block(shared))
        raise ValueError(f"unknown TaKP branch: {branch}")

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.shared_features(inputs)
        conventional = self._pool(self.conventional_block(shared))
        rebalancing = self._pool(self.rebalancing_block(shared))
        return conventional, rebalancing


@dataclass
class TaKPForwardOutput:
    logits: Tensor
    features: Tensor
    conventional_features: Tensor
    rebalancing_features: Tensor


class TaKPIncrementalNet(nn.Module):
    """Paper-faithful TaKP classifier with the released BBN mixing rule."""

    def __init__(
        self,
        num_classes: int,
        mix_scale: float = 2.0,
        *,
        stem: str = "imagenet",
    ) -> None:
        super().__init__()
        self.backbone_name = "takp_resnet18"
        self.backbone = TaKPResNet18Backbone(stem=stem)
        self.stem = stem
        self.feature_dim = self.backbone.output_dim
        self.mix_scale = float(mix_scale)
        self.classifier = nn.Linear(
            self.feature_dim, int(num_classes), bias=True
        )

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    def mixed_features(
        self,
        conventional_features: Tensor,
        rebalancing_features: Tensor,
        *,
        alpha: float,
    ) -> Tensor:
        if conventional_features.shape != rebalancing_features.shape:
            raise ValueError("TaKP branch feature shapes differ")
        alpha = float(alpha)
        return self.mix_scale * torch.cat(
            (
                alpha * conventional_features,
                (1.0 - alpha) * rebalancing_features,
            ),
            dim=1,
        )

    def forward_mixed(
        self,
        conventional_images: Tensor,
        rebalancing_images: Tensor,
        *,
        alpha: float,
    ) -> TaKPForwardOutput:
        conventional = self.backbone.branch_features(
            conventional_images, "conventional"
        )
        rebalancing = self.backbone.branch_features(
            rebalancing_images, "rebalancing"
        )
        features = self.mixed_features(
            conventional, rebalancing, alpha=alpha
        )
        return TaKPForwardOutput(
            logits=self.classifier(features),
            features=features,
            conventional_features=conventional,
            rebalancing_features=rebalancing,
        )

    def forward_detailed(
        self, images: Tensor, *, alpha: float = 0.5
    ) -> TaKPForwardOutput:
        conventional, rebalancing = self.backbone(images)
        features = self.mixed_features(
            conventional, rebalancing, alpha=alpha
        )
        return TaKPForwardOutput(
            logits=self.classifier(features),
            features=features,
            conventional_features=conventional,
            rebalancing_features=rebalancing,
        )

    def extract_features(self, images: Tensor) -> Tensor:
        # The public BBN evaluation path concatenates both branches, which is
        # equivalent to mix_scale=2 and alpha=0.5.
        return self.forward_detailed(images, alpha=0.5).features

    def forward(
        self,
        images: Tensor,
        return_features: bool = False,
        *,
        alpha: float = 0.5,
    ) -> Tensor | tuple[Tensor, Tensor]:
        output = self.forward_detailed(images, alpha=alpha)
        if return_features:
            return output.logits, output.features
        return output.logits

    def expand_classes(self, num_classes: int) -> None:
        num_classes = int(num_classes)
        if num_classes < self.num_classes:
            raise ValueError("cannot shrink a TaKP classifier")
        if num_classes == self.num_classes:
            return
        old_classifier = self.classifier
        expanded = nn.Linear(
            old_classifier.in_features,
            num_classes,
            bias=old_classifier.bias is not None,
            device=old_classifier.weight.device,
            dtype=old_classifier.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight[: self.num_classes].copy_(
                old_classifier.weight
            )
            if old_classifier.bias is not None:
                expanded.bias[: self.num_classes].copy_(
                    old_classifier.bias
                )
        self.classifier = expanded
