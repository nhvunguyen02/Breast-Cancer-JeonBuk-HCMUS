# -*- coding: utf-8 -*-
"""Contracts cho nhãn ordinal CORAL và nhãn binary phụ trợ."""
from __future__ import annotations

import torch

from tn_mammo.constants import NUM_CLASSES


def make_ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    """Nhãn lớp k -> vector 3 mức CORAL: mức j = 1 nếu label > j."""
    thresholds = torch.arange(NUM_CLASSES - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def make_binary_targets(labels: torch.Tensor) -> torch.Tensor:
    """Nhãn phụ A/B (0) với C/D (1) — chỉ dùng khi bật binary head (E2)."""
    return (labels >= 2).long()


def decode_coral_logits(
    logits: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Decode CORAL: đếm số ngưỡng vượt qua. KHÔNG dùng cho dự đoán cuối E1."""
    return (torch.sigmoid(logits) > threshold).sum(dim=1)
