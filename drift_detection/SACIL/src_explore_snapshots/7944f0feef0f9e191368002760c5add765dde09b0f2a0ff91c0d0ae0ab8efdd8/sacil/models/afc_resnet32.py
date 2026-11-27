from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class AFCBackboneOutput:
    raw_features: Tensor
    features: Tensor
    attentions: tuple[Tensor, ...]
    importance: tuple[Tensor, ...]


class ChannelImportance(nn.Module):
    """Identity layer that accumulates squared output-gradient importance."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        # Keep the author implementation's RNG consumption as well as its
        # final value.  Constructing with ones changes the random stream for
        # every convolution initialized after this module.
        self.scale = nn.Parameter(
            torch.randn(num_channels), requires_grad=False
        )
        nn.init.constant_(self.scale, 1.0)
        self.register_buffer("importance", torch.zeros_like(self.scale))
        self._collecting = False

    def _accumulate(self, gradient: Tensor) -> None:
        values = gradient.detach().float().square()
        if values.ndim == 4:
            values = values.sum(dim=(2, 3))
        self.importance.add_(values.mean(dim=0))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 4:
            outputs = inputs * self.scale.reshape(1, -1, 1, 1)
        else:
            outputs = inputs * self.scale.reshape(1, -1)
        if self._collecting and outputs.requires_grad:
            outputs.register_hook(self._accumulate)
        return outputs

    def reset(self) -> None:
        self.importance.zero_()

    def start(self) -> None:
        self._collecting = True

    def stop(self) -> None:
        self._collecting = False

    def normalize(self, epsilon: float = 1e-12) -> None:
        self.importance.div_(self.importance.mean().clamp_min(epsilon))


class DownsampleStride(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs[..., ::2, ::2]


class AFCResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        increase_dim: bool = False,
        last_relu: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels * 2 if increase_dim else in_channels
        stride = 2 if increase_dim else 1
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
        self.increase_dim = bool(increase_dim)
        self.downsample = DownsampleStride() if increase_dim else nn.Identity()
        self.last_relu = bool(last_relu)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.downsample(inputs)
        if self.increase_dim:
            residual = torch.cat((residual, torch.zeros_like(residual)), dim=1)
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs))
        outputs = outputs + residual
        if self.last_relu:
            outputs = F.relu(outputs, inplace=True)
        return outputs


class AFCStage(nn.Module):
    def __init__(self, blocks: list[AFCResidualBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        intermediates = []
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
            intermediates.append(outputs)
        return tuple(intermediates), outputs


class AFCResNet32(nn.Module):
    """The Rebuffi CIFAR ResNet used by the official AFC implementation."""

    output_dim = 64

    def __init__(self, blocks_per_stage: int = 5) -> None:
        super().__init__()
        if blocks_per_stage < 3:
            raise ValueError("AFC ResNet requires at least three blocks")
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.stage1 = AFCStage(
            [AFCResidualBlock(16) for _ in range(blocks_per_stage)]
        )
        self.stage1_importance = ChannelImportance(16)
        self.stage2 = AFCStage(
            [AFCResidualBlock(16, increase_dim=True)]
            + [AFCResidualBlock(32) for _ in range(blocks_per_stage - 1)]
        )
        self.stage2_importance = ChannelImportance(32)
        self.stage3 = AFCStage(
            [AFCResidualBlock(32, increase_dim=True)]
            + [AFCResidualBlock(64) for _ in range(blocks_per_stage - 2)]
        )
        self.stage3_importance = ChannelImportance(64)
        self.stage4 = AFCResidualBlock(64)
        self.stage4_importance = ChannelImportance(64)
        self.rebalancing_stage4: AFCResidualBlock | None = None
        self.raw_features_importance = ChannelImportance(64)
        self._initialize()

    @property
    def importance_layers(self) -> tuple[ChannelImportance, ...]:
        return (
            self.stage1_importance,
            self.stage2_importance,
            self.stage3_importance,
            self.stage4_importance,
            self.raw_features_importance,
        )

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, AFCResidualBlock):
                nn.init.zeros_(module.bn2.weight)

    @property
    def has_rebalancing_branch(self) -> bool:
        return self.rebalancing_stage4 is not None

    def enable_rebalancing_branch(self) -> None:
        """Clone the final residual block for TaKP/BBN-style rebalancing."""
        if self.rebalancing_stage4 is None:
            self.rebalancing_stage4 = copy.deepcopy(self.stage4)

    def forward(
        self,
        inputs: Tensor,
        *,
        branch: str = "conventional",
    ) -> AFCBackboneOutput:
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        stage1_features, outputs = self.stage1(outputs)
        outputs = self.stage1_importance(outputs)
        stage2_features, outputs = self.stage2(outputs)
        outputs = self.stage2_importance(outputs)
        stage3_features, outputs = self.stage3(outputs)
        outputs = self.stage3_importance(outputs)
        if branch == "conventional":
            outputs = self.stage4(outputs)
        elif branch == "rebalancing":
            if self.rebalancing_stage4 is None:
                raise RuntimeError("rebalancing branch has not been enabled")
            outputs = self.rebalancing_stage4(outputs)
        else:
            raise ValueError(f"unknown AFC branch: {branch}")
        outputs = self.stage4_importance(outputs)

        raw_features = torch.flatten(
            F.adaptive_avg_pool2d(outputs, output_size=1), 1
        )
        features = torch.flatten(
            F.adaptive_avg_pool2d(F.relu(outputs, inplace=False), 1), 1
        )
        raw_features = self.raw_features_importance(raw_features)
        attentions = (
            stage1_features[-1],
            stage2_features[-1],
            stage3_features[-1],
            outputs,
        )
        return AFCBackboneOutput(
            raw_features=raw_features,
            features=features,
            attentions=attentions,
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


def afc_resnet32() -> AFCResNet32:
    return AFCResNet32(blocks_per_stage=5)
