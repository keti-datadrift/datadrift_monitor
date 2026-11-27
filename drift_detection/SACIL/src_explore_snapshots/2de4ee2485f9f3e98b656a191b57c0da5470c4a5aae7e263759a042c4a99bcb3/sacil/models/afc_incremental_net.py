from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .afc_classifier import AFCMultiProxyClassifier
from .afc_resnet32 import AFCResNet32, afc_resnet32


@dataclass
class AFCForwardOutput:
    logits: Tensor
    features: Tensor
    attentions: tuple[Tensor, ...]
    importance: tuple[Tensor, ...]


def _class_chunks(
    num_classes: int, initial_size: int, increment_size: int
) -> list[int]:
    if num_classes < initial_size:
        raise ValueError("AFC model cannot contain fewer than base classes")
    remainder = num_classes - initial_size
    if remainder % increment_size != 0:
        raise ValueError("class count is incompatible with the CIL protocol")
    return [initial_size] + [increment_size] * (remainder // increment_size)


class AFCIncrementalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        initial_size: int,
        increment_size: int,
        proxies_per_class: int = 10,
        classifier_scale: float = 1.0,
        distance_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.backbone: AFCResNet32 = afc_resnet32()
        self.feature_dim = self.backbone.output_dim
        self.initial_size = int(initial_size)
        self.increment_size = int(increment_size)
        self.classifier = AFCMultiProxyClassifier(
            self.feature_dim,
            _class_chunks(num_classes, initial_size, increment_size),
            proxies_per_class=proxies_per_class,
            distance_scale=distance_scale,
        )

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    def extract_features(self, images: Tensor) -> Tensor:
        if self.backbone.has_rebalancing_branch:
            conventional = self.backbone(
                images, branch="conventional"
            ).raw_features
            rebalancing = self.backbone(
                images, branch="rebalancing"
            ).raw_features
            return 0.5 * (conventional + rebalancing)
        return self.backbone(images).raw_features

    def enable_rebalancing_branch(self) -> None:
        self.backbone.enable_rebalancing_branch()

    @property
    def has_rebalancing_branch(self) -> bool:
        return self.backbone.has_rebalancing_branch

    def forward_detailed(
        self,
        images: Tensor,
        *,
        branch: str = "conventional",
    ) -> AFCForwardOutput:
        backbone_output = self.backbone(images, branch=branch)
        logits = self.classifier(backbone_output.raw_features)
        return AFCForwardOutput(
            logits=logits,
            features=backbone_output.raw_features,
            attentions=backbone_output.attentions,
            importance=backbone_output.importance,
        )

    def forward(
        self, images: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.has_rebalancing_branch:
            conventional = self.forward_detailed(
                images, branch="conventional"
            )
            rebalancing = self.forward_detailed(
                images, branch="rebalancing"
            )
            logits = 0.5 * (
                conventional.logits + rebalancing.logits
            )
            features = 0.5 * (
                conventional.features + rebalancing.features
            )
            if return_features:
                return logits, features
            return logits
        output = self.forward_detailed(images, branch="conventional")
        if return_features:
            return output.logits, output.features
        return output.logits

    def expand_classes(
        self, num_classes: int, imprinted_weights: Tensor
    ) -> None:
        expected_new = int(num_classes) - self.num_classes
        if expected_new <= 0:
            raise ValueError("AFC classifier can only expand")
        if imprinted_weights.shape[0] != expected_new:
            raise ValueError("imprinted class count does not match expansion")
        self.classifier.append_imprinted(imprinted_weights)

    def main_trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [
            parameter
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        ]
        parameters.append(self.classifier.new_weights)
        return parameters

    def classifier_parameters(self) -> list[nn.Parameter]:
        return list(self.classifier.parameters())
