# -*- coding: utf-8 -*-
"""Model E1: DenseNet121 chia sẻ + mean fusion + flat head + CORAL head."""
from __future__ import annotations

import torch
from coral_pytorch.layers import CoralLayer
from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet121_Weights, densenet121

from tn_mammo.constants import FEATURE_DIM, NUM_CLASSES


class FourViewDensityModel(nn.Module):
    """E1: mean fusion + flat A/B/C/D head + CORAL ordinal head phụ trợ.

    Mean fusion không thêm tham số nào ngoài backbone, nên state_dict
    tương thích chặt với checkpoint Phase-G/E0.
    """

    def __init__(
        self,
        use_ordinal_head: bool = True,
        imagenet_init: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
            if imagenet_init
            else None
        )
        self.backbone.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

        self.ordinal_head = (
            CoralLayer(FEATURE_DIM, NUM_CLASSES)
            if use_ordinal_head
            else None
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(images)
        features = F.relu(features)  # khớp forward chuẩn của torchvision
        features = F.adaptive_avg_pool2d(features, (1, 1))
        return torch.flatten(features, 1)  # [N, 1024]

    def forward(self, views: torch.Tensor) -> dict[str, torch.Tensor | None]:
        # views: [B, 4, 3, H, W] theo thứ tự L_CC, L_MLO, R_CC, R_MLO
        batch_size, num_views = views.shape[:2]

        view_features = self.encode_images(
            views.reshape(batch_size * num_views, *views.shape[2:])
        ).reshape(batch_size, num_views, FEATURE_DIM)

        # Mean fusion: trung bình 2 view mỗi bên, rồi trung bình 2 bên.
        left = view_features[:, 0:2].mean(dim=1)
        right = view_features[:, 2:4].mean(dim=1)
        exam_features = (left + right) / 2.0

        return {
            "flat_logits": self.backbone.classifier(exam_features),
            "ordinal_logits": (
                self.ordinal_head(exam_features)
                if self.ordinal_head is not None
                else None
            ),
            "exam_features": exam_features,
        }
