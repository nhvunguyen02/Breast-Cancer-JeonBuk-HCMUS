import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)


class DenseNet121MeanLogits(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        self.backbone = densenet121(
            weights=weights,
        )

        feature_dim = (
            self.backbone.classifier.in_features
        )

        self.backbone.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

        self.num_classes = num_classes
        self.num_views = 4

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                "Expected input shape "
                "[batch, views, channels, height, width]."
            )

        (
            batch_size,
            num_views,
            channels,
            height,
            width,
        ) = images.shape

        if num_views != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} views, "
                f"received {num_views}."
            )

        images = images.reshape(
            batch_size * num_views,
            channels,
            height,
            width,
        )

        view_logits = self.backbone(
            images
        )

        view_logits = view_logits.reshape(
            batch_size,
            num_views,
            self.num_classes,
        )

        case_logits = view_logits.mean(
            dim=1,
        )

        return case_logits


def build_model(
    num_classes: int = 4,
    pretrained: bool = True,
) -> DenseNet121MeanLogits:
    return DenseNet121MeanLogits(
        num_classes=num_classes,
        pretrained=pretrained,
    )