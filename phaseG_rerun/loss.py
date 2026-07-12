from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_balanced_weights(
    class_counts: Sequence[int],
    beta: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    counts = torch.as_tensor(
        class_counts,
        dtype=torch.float32,
        device=device,
    )

    if counts.ndim != 1:
        raise ValueError(
            "class_counts must be a one-dimensional sequence."
        )

    if torch.any(counts <= 0):
        raise ValueError(
            "Every class count must be greater than zero."
        )

    if not 0.0 <= beta < 1.0:
        raise ValueError(
            "beta must satisfy 0 <= beta < 1."
        )

    effective_numbers = 1.0 - torch.pow(
        torch.tensor(
            beta,
            dtype=torch.float32,
            device=device,
        ),
        counts,
    )

    weights = (1.0 - beta) / effective_numbers

    weights = weights / weights.sum()
    weights = weights * len(class_counts)

    return weights


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.99,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if gamma < 0.0:
            raise ValueError(
                "gamma must be greater than or equal to zero."
            )

        class_weights = compute_class_balanced_weights(
            class_counts=class_counts,
            beta=beta,
        )

        self.register_buffer(
            "class_weights",
            class_weights,
        )

        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [batch, classes]."
            )

        if targets.ndim != 1:
            raise ValueError(
                "targets must have shape [batch]."
            )

        if logits.size(0) != targets.size(0):
            raise ValueError(
                "Batch size mismatch between logits and targets."
            )

        log_probabilities = F.log_softmax(
            logits,
            dim=1,
        )

        probabilities = log_probabilities.exp()

        target_log_probabilities = log_probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)

        target_probabilities = probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1),
        ).squeeze(1)

        target_class_weights = self.class_weights[
            targets
        ]

        focal_factor = torch.pow(
            1.0 - target_probabilities,
            self.gamma,
        )

        loss = (
            -target_class_weights
            * focal_factor
            * target_log_probabilities
        )

        return loss.mean()