"""Model definitions."""

import torch.nn as nn
from torchvision import models


class DenseNet121MeanFusion(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [B, 4, 3, H, W]
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        logits = self.backbone(x)
        logits = logits.view(b, v, -1)
        return logits.mean(dim=1)
