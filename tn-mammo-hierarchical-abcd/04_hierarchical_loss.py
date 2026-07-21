#!/usr/bin/env python3
"""Losses for soft hierarchy, C/D residual, and C/D-specific fusion branches."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS = ["A", "B", "C", "D"]


def class_balanced_weights(class_counts: list[int] | tuple[int, ...], beta: float) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float64)
    if counts.numel() == 0 or torch.any(counts <= 0):
        raise ValueError(f"Every training class must have a positive count, got {class_counts}")
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0,1), got {beta}")
    if beta == 0.0:
        weights = torch.ones_like(counts)
    else:
        effective = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), counts)
        weights = (1.0 - beta) / effective.clamp_min(1e-12)
    weights = weights / weights.mean()
    return weights.to(torch.float32)


def class_balanced_focal_nll(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    selected = log_probs.gather(1, targets[:, None]).squeeze(1)
    probability = selected.exp().clamp(0.0, 1.0)
    weights = class_weights.to(log_probs.device, log_probs.dtype)[targets]
    return (-weights * torch.pow(1.0 - probability, float(gamma)) * selected).mean()


def graph_safe_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def conditional_corn_loss(
    logits: torch.Tensor,
    y4: torch.Tensor,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError(f"CORN logits must be Bx3, got {tuple(logits.shape)}")
    losses: list[torch.Tensor] = []
    mask_counts: list[int] = []
    positive_counts: list[int] = []
    for threshold in range(3):
        conditional_mask = y4 > (threshold - 1)
        mask_counts.append(int(conditional_mask.sum().item()))
        if conditional_mask.any():
            targets = (y4[conditional_mask] > threshold).to(logits.dtype)
            positive_counts.append(int(targets.sum().item()))
            losses.append(
                F.binary_cross_entropy_with_logits(
                    logits[conditional_mask, threshold],
                    targets,
                )
            )
        else:
            positive_counts.append(0)
    loss = torch.stack(losses).mean() if losses else graph_safe_zero(logits)
    details = {"mask_counts": mask_counts, "target_positive_counts": positive_counts}
    return (loss, details) if return_details else loss


class HierarchicalTaskLoss(nn.Module):
    """Primary direct loss plus soft hierarchical auxiliary supervision.

    The auxiliary heads never compose the final A/B/C/D probability. In R2/R3,
    the C/D head contributes only a bounded residual to the direct C/D logits.
    """

    def __init__(self, config: dict[str, Any], class_counts: dict[str, int]) -> None:
        super().__init__()
        self.config = config
        loss_cfg = config["loss"]
        beta = float(loss_cfg.get("cb_beta", 0.99))
        self.gamma = float(loss_cfg.get("focal_gamma", 2.0))
        self.lambda_coarse = float(loss_cfg.get("lambda_coarse", 0.20))
        self.lambda_ab = float(loss_cfg.get("lambda_ab", 0.20))
        self.lambda_cd = float(loss_cfg.get("lambda_cd", 0.50))
        self.lambda_corn = float(loss_cfg.get("lambda_corn", 0.10))
        self.aux_warmup_epochs = max(1, int(loss_cfg.get("aux_warmup_epochs", 3)))

        counts4 = [int(class_counts[label]) for label in LABELS]
        self.register_buffer("weights4", class_balanced_weights(counts4, beta))
        self.register_buffer(
            "weights_coarse",
            class_balanced_weights(
                [counts4[0] + counts4[1], counts4[2] + counts4[3]],
                beta,
            ),
        )
        self.register_buffer("weights_ab", class_balanced_weights(counts4[:2], beta))
        self.register_buffer("weights_cd", class_balanced_weights(counts4[2:], beta))

    def forward(
        self,
        outputs: dict[str, torch.Tensor | None],
        y4: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float | int | list[int]]]:
        final_log_probs = outputs.get("final_log_probs")
        if final_log_probs is None:
            raise RuntimeError("Model did not provide final_log_probs")

        final_loss = class_balanced_focal_nll(
            final_log_probs,
            y4,
            self.weights4,
            gamma=self.gamma,
        )

        logits_coarse = outputs.get("logits_coarse")
        logits_ab = outputs.get("logits_ab")
        logits_cd = outputs.get("logits_cd")
        if logits_coarse is None or logits_ab is None or logits_cd is None:
            raise RuntimeError("Soft hierarchy requires coarse, A/B, and C/D heads")

        coarse_target = (y4 >= 2).to(torch.long)
        coarse_loss = F.cross_entropy(
            logits_coarse,
            coarse_target,
            weight=self.weights_coarse.to(logits_coarse.device),
        )

        ab_mask = y4 < 2
        cd_mask = y4 >= 2
        if ab_mask.any():
            ab_loss = F.cross_entropy(
                logits_ab[ab_mask],
                y4[ab_mask],
                weight=self.weights_ab.to(logits_ab.device),
            )
        else:
            ab_loss = graph_safe_zero(logits_ab)

        if cd_mask.any():
            cd_loss = F.cross_entropy(
                logits_cd[cd_mask],
                y4[cd_mask] - 2,
                weight=self.weights_cd.to(logits_cd.device),
            )
        else:
            cd_loss = graph_safe_zero(logits_cd)

        corn_loss = graph_safe_zero(final_log_probs)
        corn_details = {"mask_counts": [0, 0, 0], "target_positive_counts": [0, 0, 0]}
        if bool(self.config["experiment"].get("use_corn", True)):
            corn_logits = outputs.get("corn_logits")
            if corn_logits is None:
                raise RuntimeError("CORN was requested but the model has no CORN head")
            corn_loss, corn_details = conditional_corn_loss(
                corn_logits,
                y4,
                return_details=True,
            )

        aux_scale = min(float(epoch + 1) / float(self.aux_warmup_epochs), 1.0)
        auxiliary = (
            self.lambda_coarse * coarse_loss
            + self.lambda_ab * ab_loss
            + self.lambda_cd * cd_loss
            + self.lambda_corn * corn_loss
        )
        total = final_loss + aux_scale * auxiliary

        components: dict[str, float | int | list[int]] = {
            "loss_total": float(total.detach().item()),
            "loss_final": float(final_loss.detach().item()),
            "loss_coarse": float(coarse_loss.detach().item()),
            "loss_ab": float(ab_loss.detach().item()),
            "loss_cd": float(cd_loss.detach().item()),
            "loss_corn": float(corn_loss.detach().item()),
            "aux_scale": aux_scale,
            "lambda_coarse": self.lambda_coarse,
            "lambda_ab": self.lambda_ab,
            "lambda_cd": self.lambda_cd,
            "lambda_corn": self.lambda_corn,
            "corn_mask_counts": corn_details["mask_counts"],
            "corn_target_positive_counts": corn_details["target_positive_counts"],
        }
        return total, components
