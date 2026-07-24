# -*- coding: utf-8 -*-
"""Loss: class-balanced focal (chính) + CORAL ordinal (phụ, lambda=0.5)."""
from __future__ import annotations

import torch
from coral_pytorch.losses import coral_loss
from torch import nn
from torch.nn import functional as F

from tn_mammo.constants import TN_CLASS_COUNTS
from tn_mammo.data.contracts import make_ordinal_targets


def class_balanced_weights(
    class_counts: list[int], beta: float = 0.99
) -> torch.Tensor:
    """Trọng số 'effective number of samples' (Cui et al. 2019)."""
    counts = torch.tensor(class_counts, dtype=torch.float64)
    weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: list[int],
        beta: float = 0.99,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer(
            "class_weights", class_balanced_weights(class_counts, beta)
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        pt = torch.softmax(logits, dim=1).gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)
        return (torch.pow(1.0 - pt, self.gamma) * ce).mean()


class MultiTaskCriterion(nn.Module):
    def __init__(
        self,
        class_counts: list[int] = TN_CLASS_COUNTS,
        beta: float = 0.99,
        gamma: float = 2.0,
        lambda_ordinal: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_ordinal = lambda_ordinal
        self.flat_loss = ClassBalancedFocalLoss(class_counts, beta, gamma)

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        total = self.flat_loss(outputs["flat_logits"], labels)

        if self.lambda_ordinal > 0 and outputs["ordinal_logits"] is not None:
            total = total + self.lambda_ordinal * coral_loss(
                outputs["ordinal_logits"],
                make_ordinal_targets(labels),
            )

        return total
