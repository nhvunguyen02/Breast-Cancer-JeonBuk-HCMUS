"""Loss functions and class-weight helpers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        return loss.mean()


def coral_loss(logits, targets, num_classes):
    """CORAL ordinal loss (Cao et al. 2020). logits: [N, K-1] cumulative
    thresholds P(y>k); targets: [N] class index. Encodes that A<B<C<D are ordered,
    so predicting an adjacent class (C vs D) is penalized less than a distant one."""
    levels = (targets.view(-1, 1) >
              torch.arange(num_classes - 1, device=targets.device).view(1, -1)).float()
    return F.binary_cross_entropy_with_logits(logits, levels, reduction="mean")


def coral_probs(logits):
    """Convert CORAL cumulative logits [N, K-1] to a per-class distribution [N, K]."""
    p_gt = torch.sigmoid(logits)                       # P(y > k), k=0..K-2
    n = p_gt.shape[0]
    ones = torch.ones(n, 1, device=logits.device)
    zeros = torch.zeros(n, 1, device=logits.device)
    left = torch.cat([ones, p_gt], dim=1)              # [1, P(y>0), ...]
    right = torch.cat([p_gt, zeros], dim=1)            # [P(y>0), ..., 0]
    return (left - right).clamp_min(0.0)               # class probs, sum ~ 1


def compute_class_weights(tn_counts, cb_beta):
    """Return (ce_weights, cb_weights) as numpy arrays, derived from TN-train
    class counts only (VinDr is excluded on purpose)."""
    tn_counts = np.asarray(tn_counts, dtype=np.float32)

    ce_weights = tn_counts.sum() / (len(tn_counts) * tn_counts)

    effective_num = 1.0 - np.power(cb_beta, tn_counts)
    cb_weights = (1.0 - cb_beta) / effective_num
    cb_weights = cb_weights / cb_weights.mean()

    return ce_weights, cb_weights


def build_criterion(loss_type, ce_weights, cb_weights, focal_gamma):
    """ce_weights / cb_weights are expected to be torch tensors already on the
    target device."""
    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=ce_weights)
    elif loss_type == "focal":
        return FocalLoss(alpha=ce_weights, gamma=focal_gamma)
    elif loss_type == "cb_focal":
        return FocalLoss(alpha=cb_weights, gamma=focal_gamma)
    raise ValueError(f"Unknown loss type: {loss_type}")
