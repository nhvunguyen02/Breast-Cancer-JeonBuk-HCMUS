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
    def __init__(self, num_classes=4, masked_pool=False, mask_thresh=0.02,
                 fusion="mean", attn_dim=128):
        """4-view DenseNet121 with a shared backbone.

        masked_pool: pool the conv features only over the breast region (derived
        from the input: background is zeroed by preprocessing) instead of a plain
        global average pool.
        fusion: 'mean' averages the 4 per-view logits; 'gated' learns a
        gated-attention weight per view (Ilse et al. 2018) and aggregates the
        per-view features before classifying, so it can weight views (L/R, CC/MLO)."""
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.masked_pool = masked_pool
        self.mask_thresh = mask_thresh
        self.fusion = fusion

        if fusion == "gated":
            self.attn_V = nn.Linear(in_features, attn_dim)
            self.attn_U = nn.Linear(in_features, attn_dim)
            self.attn_w = nn.Linear(attn_dim, 1)
        elif fusion != "mean":
            raise ValueError(f"Unknown fusion: {fusion}")

    def _features_and_mask(self, x):
        """x: [N, 3, H, W] normalized. Return (feat [N,C,h,w] after relu,
        mask [N,1,h,w] breast fraction per feature cell)."""
        feat = self.backbone.features(x)                 # [N, 1024, h', w']
        feat = F.relu(feat, inplace=True)

        gray = x[:, 0] * _STD0 + _MEAN0                  # recover grayscale in ~[0,1]
        m = (gray > self.mask_thresh).float().unsqueeze(1)   # [N, 1, H, W]
        m = F.adaptive_avg_pool2d(m, feat.shape[-2:])    # breast fraction per cell
        return feat, m

    def forward(self, x, return_attn=False):
        # x: [B, 4, 3, H, W]
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)

        feat, m = self._features_and_mask(x)             # [N,C,h,w], [N,1,h,w]
        if self.masked_pool:
            num = (feat * m).sum(dim=(2, 3))
            den = m.sum(dim=(2, 3)).clamp_min(1e-6)
            pooled = num / den
        else:
            pooled = feat.mean(dim=(2, 3))               # plain global avg pool

        if self.fusion == "gated":
            h_v = pooled.view(b, v, -1)                  # [B, V, C]
            gate = torch.tanh(self.attn_V(h_v)) * torch.sigmoid(self.attn_U(h_v))
            a = torch.softmax(self.attn_w(gate), dim=1)  # [B, V, 1] weights per view
            z = (a * h_v).sum(dim=1)                     # [B, C] weighted feature
            logits = self.backbone.classifier(z)         # [B, num_classes]
        else:                                            # mean of per-view logits
            logits = self.backbone.classifier(pooled)
            logits = logits.view(b, v, -1).mean(dim=1)

        if return_attn:
            attn = feat.mean(dim=1)                       # [N, h, w] saliency proxy
            return logits, attn, m.squeeze(1)            # [N,h,w]
        return logits
