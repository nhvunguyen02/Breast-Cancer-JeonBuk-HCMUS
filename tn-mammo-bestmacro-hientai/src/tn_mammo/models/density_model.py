from __future__ import annotations

from dataclasses import dataclass

import torch
from coral_pytorch.layers import CoralLayer
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    DenseNet121_Weights,
    densenet121,
)

from tn_mammo.constants import (
    FEATURE_DIM,
    NUM_CLASSES,
    VIEW_ORDER,
)
from tn_mammo.models.fusion import (
    build_four_view_fusion,
)


@dataclass(frozen=True)
class ModelOptions:
    use_ordinal_head: bool = False
    use_binary_head: bool = False
    imagenet_init: bool = False
    fusion: str = "mean"
    fusion_dropout: float = 0.1
    control_hidden_dim: int = 608
    bilateral_bottleneck_dim: int = 256


class FourViewDensityModel(nn.Module):
    """Shared DenseNet121 four-view breast-density model.

    E0:
        Mean fusion and flat A/B/C/D head.

    E1:
        E0 plus CORAL ordinal head.

    E2:
        E1 plus A/B-versus-C/D auxiliary head.

    E3:
        Mean control, parameter-matched MLP control, or shared
        ipsilateral gated relational fusion.

    The default mean-fusion model introduces no additional state
    dictionary parameters and therefore preserves strict Phase-G
    checkpoint compatibility.
    """

    def __init__(
        self,
        options: ModelOptions,
    ) -> None:
        super().__init__()

        weights = (
            DenseNet121_Weights.IMAGENET1K_V1
            if options.imagenet_init
            else None
        )

        self.options = options

        self.backbone = densenet121(
            weights=weights
        )

        in_features = int(
            self.backbone.classifier.in_features
        )

        if in_features != FEATURE_DIM:
            raise RuntimeError(
                "Unexpected DenseNet121 feature "
                f"dimension: {in_features}"
            )

        self.backbone.classifier = nn.Linear(
            FEATURE_DIM,
            NUM_CLASSES,
        )

        self.fusion_module = (
            build_four_view_fusion(
                name=options.fusion,
                feature_dim=FEATURE_DIM,
                dropout=options.fusion_dropout,
                control_hidden_dim=(
                    options.control_hidden_dim
                ),
                bilateral_bottleneck_dim=(
                    options.bilateral_bottleneck_dim
                ),
            )
        )

        if options.use_ordinal_head:
            self.ordinal_head: nn.Module | None = (
                CoralLayer(
                    FEATURE_DIM,
                    NUM_CLASSES,
                )
            )
        else:
            self.ordinal_head = None

        if options.use_binary_head:
            self.binary_head: nn.Module | None = (
                nn.Linear(
                    FEATURE_DIM,
                    2,
                )
            )
        else:
            self.binary_head = None

    def encode_images(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone.features(
            images
        )

        # Required to match torchvision DenseNet forward.
        features = F.relu(
            features,
            inplace=False,
        )

        features = F.adaptive_avg_pool2d(
            features,
            output_size=(1, 1),
        )

        return torch.flatten(
            features,
            start_dim=1,
        )

    def encode_views(
        self,
        views: torch.Tensor,
    ) -> torch.Tensor:
        if views.ndim != 5:
            raise ValueError(
                "views must have shape "
                "[B, 4, 3, H, W]."
            )

        batch_size, num_views = views.shape[:2]

        if num_views != len(VIEW_ORDER):
            raise ValueError(
                f"Expected {len(VIEW_ORDER)} views, "
                f"received {num_views}."
            )

        flattened = views.reshape(
            batch_size * num_views,
            *views.shape[2:],
        )

        encoded = self.encode_images(
            flattened
        )

        return encoded.reshape(
            batch_size,
            num_views,
            FEATURE_DIM,
        )

    def phaseg_mean_logits(
        self,
        views: torch.Tensor,
    ) -> torch.Tensor:
        """Direct reproduction of Phase-G mean-logit computation."""
        batch_size, num_views = views.shape[:2]

        flattened = views.reshape(
            batch_size * num_views,
            *views.shape[2:],
        )

        logits = self.backbone(
            flattened
        )

        return logits.reshape(
            batch_size,
            num_views,
            NUM_CLASSES,
        ).mean(dim=1)

    def forward(
        self,
        views: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | None,
    ]:
        view_features = self.encode_views(
            views
        )

        fusion_result = self.fusion_module(
            view_features
        )

        exam_features = (
            fusion_result.exam_features
        )

        flat_logits = (
            self.backbone.classifier(
                exam_features
            )
        )

        ordinal_logits = (
            self.ordinal_head(exam_features)
            if self.ordinal_head is not None
            else None
        )

        binary_logits = (
            self.binary_head(exam_features)
            if self.binary_head is not None
            else None
        )

        return {
            "flat_logits": flat_logits,
            "ordinal_logits": ordinal_logits,
            "binary_logits": binary_logits,
            "exam_features": exam_features,
            "view_features": view_features,
            "left_features": (
                fusion_result.left_features
            ),
            "right_features": (
                fusion_result.right_features
            ),
            "left_gate_weights": (
                fusion_result.left_gate_weights
            ),
            "right_gate_weights": (
                fusion_result.right_gate_weights
            ),
            "bilateral_gate_weights": (
                fusion_result.bilateral_gate_weights
            ),
        }
