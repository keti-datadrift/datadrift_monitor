from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sacil.methods.create import ClasswiseAutoencoderClassifier
from sacil.methods.fgp import RectifiedCosineLinear

from .resnet18 import imagenet_resnet18, resnet18
from .resnet32 import resnet32
from .pycil_linear import PyCILSimpleLinear


@dataclass
class Table1ForwardOutput:
    logits: Tensor
    features: Tensor
    attentions: tuple[Tensor, ...] = ()


class ExpandableLinearNet(nn.Module):
    """Plain expanding linear-head network for CE/iCaRL controls."""

    def __init__(self, num_classes: int, backbone: str = "resnet32") -> None:
        super().__init__()
        if backbone == "resnet32":
            self.backbone = resnet32()
        elif backbone == "resnet18":
            self.backbone = resnet18()
        elif backbone == "resnet18_imagenet":
            self.backbone = imagenet_resnet18()
        elif backbone == "resnet18_imagenet_no_last_relu":
            self.backbone = imagenet_resnet18(remove_last_relu=True)
        else:
            raise ValueError(f"unsupported linear backbone: {backbone}")
        self.feature_dim = int(self.backbone.output_dim)
        self.classifier = PyCILSimpleLinear(
            self.feature_dim, int(num_classes)
        )

    @staticmethod
    def _reset_classifier(classifier: PyCILSimpleLinear) -> None:
        classifier.reset_parameters()

    @property
    def num_classes(self) -> int:
        return int(self.classifier.out_features)

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward_detailed(self, images: Tensor) -> Table1ForwardOutput:
        features = self.extract_features(images)
        return Table1ForwardOutput(
            self.classifier(features)["logits"], features
        )

    def forward(
        self, images: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        output = self.forward_detailed(images)
        return (
            (output.logits, output.features)
            if return_features
            else output.logits
        )

    def expand_classes(self, num_classes: int) -> None:
        target = int(num_classes)
        if target <= self.num_classes:
            raise ValueError("classifier expansion must add classes")
        expanded = PyCILSimpleLinear(self.feature_dim, target).to(
            device=self.classifier.weight.device,
            dtype=self.classifier.weight.dtype,
        )
        with torch.no_grad():
            expanded.weight[: self.num_classes].copy_(self.classifier.weight)
            expanded.bias[: self.num_classes].copy_(self.classifier.bias)
        self.classifier = expanded


class FGPBasicBlock(nn.Module):
    """FGP CIFAR block; the final residual block omits its last ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        last: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.last = bool(last)
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def _shortcut(self, inputs: Tensor) -> Tensor:
        if self.stride == 1 and self.in_channels == self.out_channels:
            return inputs
        reduced = F.avg_pool2d(inputs, kernel_size=1, stride=self.stride)
        padding = self.out_channels - reduced.shape[1]
        if padding < 0:
            raise RuntimeError("FGP shortcut cannot reduce channels")
        return torch.cat(
            (
                reduced,
                reduced.new_zeros(
                    reduced.shape[0],
                    padding,
                    reduced.shape[2],
                    reduced.shape[3],
                ),
            ),
            dim=1,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)), inplace=True)
        outputs = self.bn2(self.conv2(outputs)) + self._shortcut(inputs)
        return outputs if self.last else F.relu(outputs, inplace=True)


class FGPResNet32(nn.Module):
    output_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._stage(16, 16, 5, stride=1)
        self.layer2 = self._stage(16, 32, 5, stride=2)
        self.layer3 = self._stage(32, 64, 5, stride=2, final=True)
        self.avgpool = nn.AvgPool2d(8)
        self._initialize()

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        *,
        stride: int,
        final: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            FGPBasicBlock(
                in_channels, out_channels, stride=stride, last=False
            )
        ]
        for index in range(1, blocks):
            layers.append(
                FGPBasicBlock(
                    out_channels,
                    out_channels,
                    last=final and index == blocks - 1,
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

    def forward(self, images: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(self.conv1(images)), inplace=True)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.avgpool(outputs)
        return outputs.view(outputs.size(0), -1)


class FGPIncrementalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "fgp_resnet32_no_last_relu",
    ) -> None:
        super().__init__()
        if backbone == "fgp_resnet32_no_last_relu":
            self.backbone = FGPResNet32()
        elif backbone == "fgp_resnet18_imagenet_no_last_relu":
            # The author ImageNet path sets zero_init_residual=True and
            # removes the final residual ReLU.
            self.backbone = imagenet_resnet18(
                remove_last_relu=True,
                zero_init_residual=True,
            )
        else:
            raise ValueError(f"unsupported FGP backbone: {backbone}")
        self.feature_dim = self.backbone.output_dim
        self.classifier = RectifiedCosineLinear(
            self.feature_dim, int(num_classes), bias=True
        )

    @property
    def num_classes(self) -> int:
        return int(self.classifier.out_features)

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward_detailed(self, images: Tensor) -> Table1ForwardOutput:
        features = self.extract_features(images)
        return Table1ForwardOutput(self.classifier(features), features)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_detailed(images).logits

    def expand_classes(self, num_classes: int) -> None:
        target = int(num_classes)
        if target <= self.num_classes:
            raise ValueError("FGP classifier expansion must add classes")
        expanded = RectifiedCosineLinear(
            self.feature_dim, target, bias=True
        ).to(self.classifier.weight.device)
        with torch.no_grad():
            expanded.weight[: self.num_classes].copy_(self.classifier.weight)
            expanded.bias[: self.num_classes].copy_(self.classifier.bias)
            if self.classifier.scale is not None and expanded.scale is not None:
                expanded.scale.copy_(self.classifier.scale)
        self.classifier = expanded


class CREATEIncrementalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        hidden_layers: tuple[int, ...] = (),
        latent_features: int = 32,
        reconstruction_scale: float = 0.1,
        backbone: str = "resnet32",
    ) -> None:
        super().__init__()
        if backbone == "resnet32":
            self.backbone = resnet32()
        elif backbone == "resnet18":
            self.backbone = resnet18()
        else:
            raise ValueError(f"unsupported CREATE backbone: {backbone}")
        self.feature_dim = int(self.backbone.output_dim)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.latent_features = int(latent_features)
        self.reconstruction_scale = float(reconstruction_scale)
        self.classifier = self._new_classifier(int(num_classes))

    def _new_classifier(self, num_classes: int) -> ClasswiseAutoencoderClassifier:
        return ClasswiseAutoencoderClassifier(
            self.feature_dim,
            num_classes,
            hidden_layers=self.hidden_layers,
            latent_features=self.latent_features,
            reconstruction_scale=self.reconstruction_scale,
        )

    @property
    def num_classes(self) -> int:
        return int(self.classifier.num_classes)

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward_detailed(self, images: Tensor) -> dict[str, Tensor]:
        return self.classifier(self.extract_features(images))

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_detailed(images)["logits"]

    def expand_classes(self, num_classes: int) -> None:
        target = int(num_classes)
        if target <= self.num_classes:
            raise ValueError("CREATE classifier expansion must add classes")
        expanded = self._new_classifier(target).to(
            next(self.parameters()).device
        )
        for class_id in range(self.num_classes):
            expanded.class_autoencoders[class_id] = copy.deepcopy(
                self.classifier.class_autoencoders[class_id]
            )
        self.classifier = expanded


class ChunkedCosineClassifier(nn.Module):
    """CSCCT split cosine head with independently freezable task chunks."""

    def __init__(self, feature_dim: int, class_chunks: list[int]) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.class_chunks = [int(value) for value in class_chunks]
        self.weights = nn.ParameterList()
        for count in self.class_chunks:
            parameter = nn.Parameter(torch.empty(count, self.feature_dim))
            nn.init.uniform_(
                parameter,
                -1.0 / math.sqrt(self.feature_dim),
                1.0 / math.sqrt(self.feature_dim),
            )
            self.weights.append(parameter)
        self.sigma = nn.Parameter(torch.ones(1))

    @property
    def num_classes(self) -> int:
        return sum(self.class_chunks)

    @property
    def weight(self) -> Tensor:
        return torch.cat(tuple(self.weights), dim=0)

    @property
    def old_weights(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.weights[:-1])

    @property
    def new_weights(self) -> nn.Parameter:
        return self.weights[-1]

    def forward(self, features: Tensor) -> Tensor:
        return self.sigma * F.linear(
            F.normalize(features, dim=1), F.normalize(self.weight, dim=1)
        )

    def append_imprinted(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError("invalid CSCCT imprinted classifier weights")
        self.weights.append(nn.Parameter(values.detach().clone()))
        self.class_chunks.append(int(values.shape[0]))


class _CSCCTBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        last: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.last = bool(last)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs)) + self.shortcut(inputs)
        return outputs if self.last else self.relu(outputs)


class CSCCTResNet32(nn.Module):
    output_dim = 64

    def __init__(self, *, initialize: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._stage(16, 16, stride=1)
        self.layer2 = self._stage(16, 32, stride=2)
        self.layer3 = self._stage(32, 64, stride=2, final=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        if initialize:
            self.initialize_parameters()

    @staticmethod
    def _stage(
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        final: bool = False,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            _CSCCTBasicBlock(in_channels, out_channels, stride=stride)
        ]
        for index in range(1, 5):
            blocks.append(
                _CSCCTBasicBlock(
                    out_channels, out_channels, last=final and index == 4
                )
            )
        return nn.Sequential(*blocks)

    def initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def stem(self, images: Tensor) -> Tensor:
        return self.relu(self.bn1(self.conv1(images)))

    def pool(self, features: Tensor) -> Tensor:
        return torch.flatten(self.avgpool(features), 1)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        map1 = self.layer1(self.stem(images))
        map2 = self.layer2(map1)
        map3 = self.layer3(map2)
        return self.pool(map3), (map1, map2, map3)


class CSCCTImageNetResNet18(nn.Module):
    """Author-aligned CSCCT ResNet-18 backbone for ImageNet-Subset.

    This mirrors ``models/modified_resnet.py`` from the CSCCT release: the
    ImageNet stem is followed by four residual stages, the last residual
    block omits its final ReLU, and 224px inputs are pooled with a 7x7 average
    pool.  The classifier remains in :class:`CSCCTIncrementalNet` so one
    shared split-cosine implementation can serve both datasets.
    """

    output_dim = 512

    def __init__(self, *, initialize: bool = True) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stage(64, blocks=2, stride=1)
        self.layer2 = self._stage(128, blocks=2, stride=2)
        self.layer3 = self._stage(256, blocks=2, stride=2)
        self.layer4 = self._stage(
            512, blocks=2, stride=2, remove_final_relu=True
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if initialize:
            self.initialize_parameters()

    def _stage(
        self,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
        remove_final_relu: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index in range(blocks):
            layers.append(
                _CSCCTBasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=stride if index == 0 else 1,
                    last=remove_final_relu and index == blocks - 1,
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def stem(self, images: Tensor) -> Tensor:
        return self.maxpool(self.relu(self.bn1(self.conv1(images))))

    def pool(self, features: Tensor) -> Tensor:
        return torch.flatten(self.avgpool(features), 1)

    def forward(self, images: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        map1 = self.layer1(self.stem(images))
        map2 = self.layer2(map1)
        map3 = self.layer3(map2)
        map4 = self.layer4(map3)
        return self.pool(map4), (map1, map2, map3, map4)


class ScaleShiftConv2d(nn.Conv2d):
    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        )
        self.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.weight.copy_(source.weight)
            if self.bias is not None and source.bias is not None:
                self.bias.copy_(source.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.mtl_weight = nn.Parameter(
            torch.ones(
                self.out_channels,
                self.in_channels // self.groups,
                1,
                1,
                device=source.weight.device,
                dtype=source.weight.dtype,
            )
        )
        if self.bias is None:
            self.register_parameter("mtl_bias", None)
        else:
            self.mtl_bias = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight * self.mtl_weight.expand_as(self.weight)
        bias = None if self.bias is None else self.bias + self.mtl_bias
        return F.conv2d(
            inputs,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _convert_scale_shift(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, ScaleShiftConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, ScaleShiftConv2d(child))
        else:
            _convert_scale_shift(child)


class CSCCTIncrementalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "cscct_modified_resnet32",
    ) -> None:
        super().__init__()
        # The reference ResNet constructs its cosine FC before applying the
        # explicit Kaiming initialization to convolution layers.  Defer the
        # backbone initialization until after the separate wrapper head has
        # been constructed so the seeded parameter stream is identical.
        self.backbone_name = str(backbone)
        if self.backbone_name == "cscct_modified_resnet32":
            backbone_type: type[CSCCTResNet32 | CSCCTImageNetResNet18] = (
                CSCCTResNet32
            )
        elif self.backbone_name == "cscct_modified_resnet18_imagenet":
            backbone_type = CSCCTImageNetResNet18
        else:
            raise ValueError(f"unsupported CSCCT backbone: {self.backbone_name}")
        self.first = backbone_type(initialize=False)
        self.second: CSCCTResNet32 | CSCCTImageNetResNet18 | None = None
        self.feature_dim = self.first.output_dim
        self.classifier = ChunkedCosineClassifier(
            self.feature_dim, [int(num_classes)]
        )
        self.first.initialize_parameters()
        fusion_levels = 4 if isinstance(self.first, CSCCTImageNetResNet18) else 3
        self.fusion = nn.ParameterList(
            nn.Parameter(torch.tensor([0.5])) for _ in range(fusion_levels)
        )

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    def _mix(self, index: int, left: Tensor, right: Tensor) -> Tensor:
        alpha = self.fusion[index]
        return alpha * left + (1.0 - alpha) * right

    def extract_detailed(self, images: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        if self.second is None:
            return self.first(images)
        left1 = self.first.layer1(self.first.stem(images))
        right1 = self.second.layer1(self.second.stem(images))
        map1 = self._mix(0, left1, right1)
        map2 = self._mix(
            1, self.first.layer2(map1), self.second.layer2(map1)
        )
        map3 = self._mix(
            2, self.first.layer3(map2), self.second.layer3(map2)
        )
        if isinstance(self.first, CSCCTImageNetResNet18):
            if not isinstance(self.second, CSCCTImageNetResNet18):
                raise TypeError("CSCCT ImageNet branches have different types")
            map4 = self._mix(
                3, self.first.layer4(map3), self.second.layer4(map3)
            )
            return self.first.pool(map4), (map1, map2, map3, map4)
        return self.first.pool(map3), (map1, map2, map3)

    def extract_features(self, images: Tensor) -> Tensor:
        return self.extract_detailed(images)[0]

    def forward_detailed(self, images: Tensor) -> Table1ForwardOutput:
        features, attentions = self.extract_detailed(images)
        return Table1ForwardOutput(
            self.classifier(features), features, attentions
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_detailed(images).logits

    def expand_classes(self, imprinted_weights: Tensor) -> None:
        if self.second is None:
            self.second = copy.deepcopy(self.first)
            _convert_scale_shift(self.first)
        self.classifier.append_imprinted(imprinted_weights)

    def main_parameters(self) -> list[nn.Parameter]:
        fusion_ids = {id(parameter) for parameter in self.fusion}
        old_ids = {
            id(parameter) for parameter in self.classifier.old_weights
        }
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
            and id(parameter) not in fusion_ids
            and id(parameter) not in old_ids
        ]
