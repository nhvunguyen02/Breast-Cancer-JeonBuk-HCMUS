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
