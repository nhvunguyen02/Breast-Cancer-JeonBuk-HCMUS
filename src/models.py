"""Model definitions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ImageNet channel-0 stats, used to recover the grayscale value from the
# normalized input so the breast mask (background was zeroed) can be derived.
_MEAN0 = 0.485
_STD0 = 0.229


class DenseNet121MeanFusion(nn.Module):
    def __init__(self, num_classes=4, masked_pool=False, mask_thresh=0.02):
        """masked_pool: pool the conv features only over the breast region
        (derived from the input: background is zeroed by preprocessing) instead
        of a plain global average pool, so background/edge/pectoral features
        cannot drive the prediction."""
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.masked_pool = masked_pool
        self.mask_thresh = mask_thresh

    def _masked_logits(self, x):
        """x: [N, 3, H, W] normalized. Return per-image logits using a breast-
        masked global average pool over the conv feature maps."""
        feat = self.backbone.features(x)                 # [N, 1024, h', w']
        feat = F.relu(feat, inplace=True)

        gray = x[:, 0] * _STD0 + _MEAN0                  # recover grayscale in ~[0,1]
        m = (gray > self.mask_thresh).float().unsqueeze(1)   # [N, 1, H, W]
        m = F.adaptive_avg_pool2d(m, feat.shape[-2:])    # breast fraction per cell

        num = (feat * m).sum(dim=(2, 3))
        den = m.sum(dim=(2, 3)).clamp_min(1e-6)
        pooled = num / den                               # [N, 1024]
        return self.backbone.classifier(pooled)          # [N, num_classes]

    def forward(self, x):
        # x: [B, 4, 3, H, W]
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        if self.masked_pool:
            logits = self._masked_logits(x)
        else:
            logits = self.backbone(x)
        logits = logits.view(b, v, -1)
        return logits.mean(dim=1)
