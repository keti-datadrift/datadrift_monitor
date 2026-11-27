from __future__ import annotations

from torch import Tensor, nn

from .cosine_classifier import CosineClassifier
from .resnet18 import resnet18
from .resnet32 import resnet32


class IncrementalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet32",
        classifier_scale: float = 10.0,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in {"resnet18", "resnet32"}:
            raise ValueError(f"unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.backbone = (
            resnet18() if backbone == "resnet18" else resnet32()
        )
        self.feature_dim = self.backbone.output_dim
        self.classifier = CosineClassifier(
            self.feature_dim,
            num_classes,
            initial_scale=classifier_scale,
            learnable_scale=learnable_scale,
        )

    @property
    def num_classes(self) -> int:
        return self.classifier.num_classes

    def extract_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def forward(
        self, images: Tensor, return_features: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        features = self.extract_features(images)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

    def expand_classes(self, num_classes: int) -> None:
        self.classifier = self.classifier.expanded(num_classes)
