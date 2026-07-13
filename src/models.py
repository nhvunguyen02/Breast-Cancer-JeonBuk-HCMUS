"""Model definitions: a multi-view classifier with a selectable backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ImageNet channel-0 stats, used to recover the grayscale value from the
# normalized input so the breast mask (background was zeroed) can be derived.
_MEAN0 = 0.485
_STD0 = 0.229


def build_backbone(name):
    """Return (feature_extractor, feat_dim, needs_relu) for a torchvision backbone.

    feature_extractor maps [N, 3, H, W] -> spatial feature maps [N, C, h, w].
    needs_relu: whether a ReLU must be applied to the feature maps (DenseNet
    applies it in its own forward; EfficientNet/ResNet features are already
    activated)."""
    name = name.lower()
    if name == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1")
        return m.features, m.classifier.in_features, True
    if name == "densenet169":
        m = models.densenet169(weights="IMAGENET1K_V1")
        return m.features, m.classifier.in_features, True
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1")
        return m.features, m.classifier[1].in_features, False
    if name == "efficientnet_b3":
        m = models.efficientnet_b3(weights="IMAGENET1K_V1")
        return m.features, m.classifier[1].in_features, False
    if name == "efficientnet_b5":
        m = models.efficientnet_b5(weights="IMAGENET1K_V1")
        return m.features, m.classifier[1].in_features, False
    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V1")
        return nn.Sequential(*list(m.children())[:-2]), m.fc.in_features, False
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1")
        return m.features, m.classifier[2].in_features, False
    if name == "convnext_small":
        m = models.convnext_small(weights="IMAGENET1K_V1")
        return m.features, m.classifier[2].in_features, False
    if name == "convnext_base":
        m = models.convnext_base(weights="IMAGENET1K_V1")
        return m.features, m.classifier[2].in_features, False
    raise ValueError(f"Unknown backbone: {name}")


class MultiViewModel(nn.Module):
    def __init__(self, num_classes=4, backbone="densenet121", masked_pool=False,
                 mask_thresh=0.02, fusion="mean", attn_dim=128, ordinal=False):
        """4-view classifier with a shared, selectable backbone.

        backbone: 'densenet121' | 'densenet169' | 'efficientnet_b0/b3/b5' | 'resnet50'.
        masked_pool: pool conv features only over the breast region (background
        was zeroed by preprocessing) instead of a plain global average pool.
        fusion: 'mean' averages the 4 per-view logits; 'gated' learns a
        gated-attention weight per view (Ilse et al. 2018) and aggregates the
        per-view features before classifying."""
        super().__init__()
        self.features, feat_dim, self._needs_relu = build_backbone(backbone)
        self.num_classes = num_classes
        self.ordinal = ordinal
        if ordinal:
            # CORAL head: one shared projection + (K-1) ordered bias thresholds.
            self.classifier = nn.Linear(feat_dim, 1, bias=False)
            self.ordinal_bias = nn.Parameter(torch.zeros(num_classes - 1))
        else:
            self.classifier = nn.Linear(feat_dim, num_classes)
        # ConvNeXt's head LayerNorm is dropped with its classifier; restore a
        # norm on the pooled feature so the fresh Linear trains stably.
        self.head_norm = nn.LayerNorm(feat_dim) if backbone.startswith("convnext") else nn.Identity()
        self.backbone_name = backbone
        self.masked_pool = masked_pool
        self.mask_thresh = mask_thresh
        self.fusion = fusion

        if fusion == "gated":
            self.attn_V = nn.Linear(feat_dim, attn_dim)
            self.attn_U = nn.Linear(feat_dim, attn_dim)
            self.attn_w = nn.Linear(attn_dim, 1)
        elif fusion != "mean":
            raise ValueError(f"Unknown fusion: {fusion}")

    def _features_and_mask(self, x):
        """x: [N, 3, H, W] normalized. Return (feat [N,C,h,w], mask [N,1,h,w])."""
        feat = self.features(x)
        if self._needs_relu:
            feat = F.relu(feat, inplace=True)

        gray = x[:, 0] * _STD0 + _MEAN0                  # recover grayscale in ~[0,1]
        m = (gray > self.mask_thresh).float().unsqueeze(1)   # [N, 1, H, W]
        m = F.adaptive_avg_pool2d(m, feat.shape[-2:])    # breast fraction per cell
        return feat, m

    def _head(self, z):
        """Map a pooled/aggregated feature to logits: [., K] normally, or the
        [., K-1] CORAL cumulative logits (shared projection + ordered biases)."""
        out = self.classifier(z)
        if self.ordinal:
            out = out + self.ordinal_bias                # [.,1] + [K-1] -> [.,K-1]
        return out

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

        pooled = self.head_norm(pooled)                  # no-op unless ConvNeXt

        if self.fusion == "gated":
            h_v = pooled.view(b, v, -1)                  # [B, V, C]
            gate = torch.tanh(self.attn_V(h_v)) * torch.sigmoid(self.attn_U(h_v))
            a = torch.softmax(self.attn_w(gate), dim=1)  # [B, V, 1] per-view weights
            z = (a * h_v).sum(dim=1)                     # [B, C]
            logits = self._head(z)
        else:                                            # mean of per-view logits
            logits = self._head(pooled)
            logits = logits.view(b, v, -1).mean(dim=1)

        if return_attn:
            attn = feat.mean(dim=1)                       # [N, h, w] saliency proxy
            return logits, attn, m.squeeze(1)
        return logits


# Backward-compatible alias (older callers / checkpoints used this name).
DenseNet121MeanFusion = MultiViewModel
