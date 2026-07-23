from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from coral_pytorch.losses import coral_loss
from torch import nn
from torch.nn import functional as F

from tn_mammo.data.contracts import (
    make_binary_targets,
    make_ordinal_targets,
)


def compute_class_balanced_weights(
    class_counts: list[int] | tuple[int, ...],
    *,
    beta: float,
) -> torch.Tensor:
    counts = torch.tensor(
        class_counts,
        dtype=torch.float64,
    )

    if torch.any(counts <= 0):
        raise ValueError(
            "Class counts must be positive."
        )

    if not 0.0 <= beta < 1.0:
        raise ValueError(
            "beta must lie in [0, 1)."
        )

    effective_number = (
        1.0 - torch.pow(beta, counts)
    )

    weights = (
        (1.0 - beta)
        / effective_number
    )

    weights = (
        weights
        / weights.sum()
        * len(class_counts)
    )

    return weights.to(torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        class_counts: list[int] | tuple[int, ...],
        *,
        beta: float = 0.99,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        self.gamma = float(gamma)

        weights = (
            compute_class_balanced_weights(
                class_counts,
                beta=beta,
            )
        )

        self.register_buffer(
            "class_weights",
            weights,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        pt = probabilities.gather(
            1,
            targets.unsqueeze(1),
        ).squeeze(1)

        focal_factor = torch.pow(
            1.0 - pt,
            self.gamma,
        )

        return (
            focal_factor * ce
        ).mean()


@dataclass(frozen=True)
class MultiTaskLossOptions:
    lambda_ordinal: float = 0.0
    lambda_binary: float = 0.0


class MultiTaskCriterion(nn.Module):
    def __init__(
        self,
        *,
        flat_class_counts: list[int],
        binary_class_counts: list[int],
        beta: float,
        gamma: float,
        options: MultiTaskLossOptions,
    ) -> None:
        super().__init__()

        self.options = options

        self.flat_loss = (
            ClassBalancedFocalLoss(
                flat_class_counts,
                beta=beta,
                gamma=gamma,
            )
        )

        binary_weights = (
            compute_class_balanced_weights(
                binary_class_counts,
                beta=beta,
            )
        )

        self.register_buffer(
            "binary_weights",
            binary_weights,
        )

    def forward(
        self,
        outputs: Mapping[
            str,
            torch.Tensor | None
        ],
        labels: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        flat_logits = outputs.get(
            "flat_logits"
        )

        if flat_logits is None:
            raise ValueError(
                "flat_logits are required."
            )

        flat = self.flat_loss(
            flat_logits,
            labels,
        )

        total = flat

        parts: dict[str, torch.Tensor] = {
            "flat": flat.detach(),
        }

        if self.options.lambda_ordinal > 0:
            ordinal_logits = outputs.get(
                "ordinal_logits"
            )

            if ordinal_logits is None:
                raise ValueError(
                    "Ordinal loss enabled but "
                    "ordinal_logits are missing."
                )

            levels = make_ordinal_targets(
                labels
            )

            ordinal = coral_loss(
                ordinal_logits,
                levels,
            )

            total = (
                total
                + self.options.lambda_ordinal
                * ordinal
            )

            parts["ordinal"] = (
                ordinal.detach()
            )

        if self.options.lambda_binary > 0:
            binary_logits = outputs.get(
                "binary_logits"
            )

            if binary_logits is None:
                raise ValueError(
                    "Binary loss enabled but "
                    "binary_logits are missing."
                )

            binary_targets = (
                make_binary_targets(labels)
            )

            binary = F.cross_entropy(
                binary_logits,
                binary_targets,
                weight=self.binary_weights,
            )

            total = (
                total
                + self.options.lambda_binary
                * binary
            )

            parts["binary"] = binary.detach()

        parts["total"] = total.detach()

        return total, parts
