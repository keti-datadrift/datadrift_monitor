"""Stock PyCIL PODNet classifier/expansion with a thin runner adapter.

The classifier definitions and update lifecycle mirror:

* ``ref_codes/00_frameworks/PyCIL/convs/linears.py``
* ``ref_codes/00_frameworks/PyCIL/utils/inc_net.py::CosineIncrementalNet``

PODNet deliberately does not inherit from or reuse AFC's classifier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .resnet18 import ImageNetResNet18, imagenet_resnet18
from .resnet32 import CifarResNet, resnet32


def reduce_proxies(outputs: Tensor, nb_proxy: int) -> Tensor:
    if nb_proxy == 1:
        return outputs
    batch_size = outputs.shape[0]
    nb_classes = outputs.shape[1] / nb_proxy
    if not nb_classes.is_integer():
        raise ValueError("proxy output shape is not divisible by proxy count")
    similarities = outputs.view(batch_size, int(nb_classes), nb_proxy)
    attentions = F.softmax(similarities, dim=-1)
    return (attentions * similarities).sum(-1)


class PyCILCosineLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nb_proxy: int = 1,
        to_reduce: bool = False,
        sigma: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features) * int(nb_proxy)
        self.nb_proxy = int(nb_proxy)
        self.to_reduce = bool(to_reduce)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.empty(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        outputs = F.linear(
            F.normalize(inputs, p=2, dim=1),
            F.normalize(self.weight, p=2, dim=1),
        )
        if self.to_reduce:
            outputs = reduce_proxies(outputs, self.nb_proxy)
        if self.sigma is not None:
            outputs = self.sigma * outputs
        return {"logits": outputs}


class PyCILSplitCosineLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features1: int,
        out_features2: int,
        nb_proxy: int = 1,
        sigma: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = (
            int(out_features1) + int(out_features2)
        ) * int(nb_proxy)
        self.nb_proxy = int(nb_proxy)
        self.fc1 = PyCILCosineLinear(
            in_features, out_features1, nb_proxy, False, False
        )
        self.fc2 = PyCILCosineLinear(
            in_features, out_features2, nb_proxy, False, False
        )
        if sigma:
            self.sigma = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("sigma", None)

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        old = self.fc1(inputs)["logits"]
        new = self.fc2(inputs)["logits"]
        outputs = reduce_proxies(
            torch.cat((old, new), dim=1), self.nb_proxy
        )
        if self.sigma is not None:
            outputs = self.sigma * outputs
        return {
            "old_scores": reduce_proxies(old, self.nb_proxy),
            "new_scores": reduce_proxies(new, self.nb_proxy),
            "logits": outputs,
        }


@dataclass
class PyCILPODNetOutput:
    logits: Tensor
    features: Tensor
    attentions: tuple[Tensor, ...]


class PyCILPODNet(nn.Module):
    """PyCIL ``CosineIncrementalNet`` with a dataset-native backbone."""

    def __init__(
        self,
        num_classes: int,
        *,
        proxies_per_class: int = 10,
        backbone: str = "resnet32",
    ) -> None:
        super().__init__()
        if backbone == "resnet32":
            self.backbone: CifarResNet | ImageNetResNet18 = resnet32()
        elif backbone == "resnet18_imagenet":
            self.backbone = imagenet_resnet18()
        else:
            raise ValueError(f"unsupported PODNet backbone: {backbone}")
        self.backbone_name = str(backbone)
        self.feature_dim = int(self.backbone.output_dim)
        self.nb_proxy = int(proxies_per_class)
        self.fc: PyCILCosineLinear | PyCILSplitCosineLinear = PyCILCosineLinear(
            self.feature_dim,
            int(num_classes),
            self.nb_proxy,
            to_reduce=True,
        )

    @property
    def classifier(self) -> PyCILCosineLinear | PyCILSplitCosineLinear:
        return self.fc

    @property
    def num_classes(self) -> int:
        return int(self.fc.out_features // self.nb_proxy)

    def expand_classes(self, num_classes: int) -> None:
        target = int(num_classes)
        previous = self.num_classes
        if target <= previous:
            raise ValueError("PODNet expansion must add classes")
        expanded = PyCILSplitCosineLinear(
            self.feature_dim,
            previous,
            target - previous,
            self.nb_proxy,
        ).to(device=next(self.parameters()).device)
        with torch.no_grad():
            if isinstance(self.fc, PyCILCosineLinear):
                expanded.fc1.weight.copy_(self.fc.weight)
                expanded.sigma.copy_(self.fc.sigma)
            else:
                previous_first = self.fc.fc1.out_features
                expanded.fc1.weight[:previous_first].copy_(self.fc.fc1.weight)
                expanded.fc1.weight[previous_first:].copy_(self.fc.fc2.weight)
                expanded.sigma.copy_(self.fc.sigma)
        self.fc = expanded

    def freeze_old_classifier(self) -> None:
        if isinstance(self.fc, PyCILSplitCosineLinear):
            for parameter in self.fc.fc1.parameters():
                parameter.requires_grad_(False)

    def main_trainable_parameters(self) -> list[nn.Parameter]:
        self.freeze_old_classifier()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward_detailed(self, images: Tensor) -> PyCILPODNetOutput:
        backbone = self.backbone.forward_detailed(images)
        features = backbone["features"]
        attentions = backbone["fmaps"]
        if not isinstance(features, Tensor) or not isinstance(attentions, tuple):
            raise TypeError("invalid PyCIL backbone adapter output")
        logits = self.fc(features)["logits"]
        return PyCILPODNetOutput(logits, features, attentions)

    def forward(self, images: Tensor, return_features: bool = False):
        output = self.forward_detailed(images)
        if return_features:
            return output.logits, output.features
        return output.logits
