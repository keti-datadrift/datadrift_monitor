"""AFC's author-released importance-aware ImageNet ResNet-18.

The layer layout mirrors
``ref_codes/02_stability_plasticity_selective_preservation/AFC/``
``inclearn/convnet/resnet_importance.py``.  Only the hook lifecycle and return
value are adapted to the unified in-repository model contract.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F

from .afc_resnet32 import AFCBackboneOutput, ChannelImportance


class AFCImageNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        downsample: nn.Module | None = None,
        last_relu: bool = True,
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
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.last_relu = bool(last_relu)

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        outputs = outputs + identity
        return self.relu(outputs) if self.last_relu else outputs


class AFCResNet18(nn.Module):
    """ImageNet-100 ``resnet18_importance`` from the AFC release."""

    output_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 64
        # The released B50 ImageNet-100 config leaves initial_kernel at its
        # default of 3 and uses stride 1 before the max-pool.
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2)
        self.stage1_importance = ChannelImportance(64)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.stage2_importance = ChannelImportance(128)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.stage3_importance = ChannelImportance(256)
        self.layer4 = self._make_layer(512, blocks=2, stride=2, last=True)
        self.stage4_importance = ChannelImportance(512)
        self.raw_features_importance = ChannelImportance(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize()

    def _make_layer(
        self,
        out_channels: int,
        *,
        blocks: int,
        stride: int = 1,
        last: bool = False,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers: list[nn.Module] = [
            AFCImageNetBasicBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_channels = out_channels
        for index in range(1, blocks):
            # This reproduces the author code: with two-block stages the
            # second residual block has no final ReLU in every stage.
            layers.append(
                AFCImageNetBasicBlock(
                    self.in_channels,
                    out_channels,
                    last_relu=not (index == blocks - 1 or last),
                )
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
        # AFC's ResNet constructor defaults zero_init_residual=True.
        for module in self.modules():
            if isinstance(module, AFCImageNetBasicBlock):
                nn.init.zeros_(module.bn2.weight)

    @property
    def importance_layers(self) -> tuple[ChannelImportance, ...]:
        return (
            self.stage1_importance,
            self.stage2_importance,
            self.stage3_importance,
            self.stage4_importance,
            self.raw_features_importance,
        )

    @property
    def has_rebalancing_branch(self) -> bool:
        return False

    def enable_rebalancing_branch(self) -> None:
        raise RuntimeError("AFC ImageNet ResNet-18 has no rebalancing branch")

    def forward(
        self,
        inputs: Tensor,
        *,
        branch: str = "conventional",
    ) -> AFCBackboneOutput:
        if branch != "conventional":
            raise ValueError("AFC ImageNet ResNet-18 only has one branch")
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        map1 = self.stage1_importance(self.layer1(outputs))
        map2 = self.stage2_importance(self.layer2(map1))
        map3 = self.stage3_importance(self.layer3(map2))
        map4 = self.stage4_importance(self.layer4(map3))
        raw_features = self.raw_features_importance(
            F.adaptive_avg_pool2d(map4, output_size=1).flatten(1)
        )
        features = F.adaptive_avg_pool2d(
            F.relu(map4, inplace=False), output_size=1
        ).flatten(1)
        return AFCBackboneOutput(
            raw_features=raw_features,
            features=features,
            attentions=(map1, map2, map3, map4),
            importance=tuple(
                layer.importance for layer in self.importance_layers
            ),
        )

    def reset_importance(self) -> None:
        for layer in self.importance_layers:
            layer.reset()

    def start_importance_collection(self) -> None:
        for layer in self.importance_layers:
            layer.start()

    def stop_importance_collection(self) -> None:
        for layer in self.importance_layers:
            layer.stop()

    def normalize_importance(self) -> None:
        for layer in self.importance_layers:
            layer.normalize()


def afc_resnet18() -> AFCResNet18:
    return AFCResNet18()
