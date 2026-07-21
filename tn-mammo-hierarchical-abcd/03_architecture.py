#!/usr/bin/env python3
"""Three incremental TN-Mammo improvement branches.

R1: direct A/B/C/D prediction + soft hierarchical auxiliary heads.
R2: R1 + bounded C/D residual correction on final C/D logits.
R3: R2 + C/D-specific bilateral/gated fusion for the C/D specialist.

The direct four-class head remains the primary path in every branch.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

VIEW_ORDER = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
SOFT_ARCHITECTURES = {
    "soft_multitask",
    "soft_multitask_cd_residual",
    "soft_multitask_cd_fusion",
}


def masked_mean(features: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    weights = mask.to(features.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (features * weights).sum(dim=dim) / denominator


class SharedDenseNet121Encoder(nn.Module):
    def __init__(self, pretrained: bool, allow_random_init_if_unavailable: bool) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            network = densenet121(weights=weights)
            self.initialization = "imagenet1k" if weights is not None else "random"
        except Exception as exc:
            if pretrained and allow_random_init_if_unavailable:
                network = densenet121(weights=None)
                self.initialization = f"random_fallback:{exc!r}"
            else:
                raise RuntimeError(
                    "DenseNet121 ImageNet weights were requested but unavailable. "
                    "Populate the torchvision cache or explicitly allow random initialization."
                ) from exc
        self.features = network.features
        self.out_dim = int(network.classifier.in_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        feature_map = F.relu(feature_map, inplace=False)
        return F.adaptive_avg_pool2d(feature_map, output_size=(1, 1)).flatten(1)


class ImprovedTask1DenseNet(nn.Module):
    def __init__(self, config: dict[str, Any], pretrained_override: bool | None = None) -> None:
        super().__init__()
        experiment = config["experiment"]
        model_cfg = config["model"]

        self.architecture = str(experiment["architecture"])
        if self.architecture not in SOFT_ARCHITECTURES:
            raise ValueError(f"Unsupported improvement architecture={self.architecture}")

        self.fusion = str(experiment.get("fusion", "mean"))
        if self.fusion != "mean":
            raise ValueError("The primary path is locked to mean fusion for fair incremental ablation")

        self.use_corn = bool(experiment.get("use_corn", True))
        self.use_cd_residual = self.architecture in {
            "soft_multitask_cd_residual",
            "soft_multitask_cd_fusion",
        }
        self.use_cd_specific_fusion = self.architecture == "soft_multitask_cd_fusion"
        self.cd_residual_alpha = float(model_cfg.get("cd_residual_alpha", 0.35))
        if not 0.0 <= self.cd_residual_alpha <= 0.5:
            raise ValueError("cd_residual_alpha must be in [0, 0.5]")

        pretrained = bool(model_cfg.get("imagenet_pretrained", True))
        if pretrained_override is not None:
            pretrained = bool(pretrained_override)
        self.encoder = SharedDenseNet121Encoder(
            pretrained=pretrained,
            allow_random_init_if_unavailable=bool(
                model_cfg.get("allow_random_init_if_unavailable", False)
            ),
        )

        feature_dim = self.encoder.out_dim
        hidden_dim = int(model_cfg.get("hidden_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.25))
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.exam_project = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.direct_head = nn.Linear(hidden_dim, 4)
        self.head_coarse = nn.Linear(hidden_dim, 2)
        self.head_ab = nn.Linear(hidden_dim, 2)
        self.head_cd = nn.Linear(hidden_dim, 2)
        self.corn_head = nn.Linear(hidden_dim, 3) if self.use_corn else None

        self.cd_view_embedding = nn.Embedding(4, feature_dim)
        nn.init.normal_(self.cd_view_embedding.weight, mean=0.0, std=0.02)
        self.cd_gate_fc = nn.Linear(feature_dim, 1)
        self.missing_side_token = nn.Parameter(torch.zeros(2, feature_dim))
        nn.init.normal_(self.missing_side_token, mean=0.0, std=0.02)
        self.cd_project = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _encode_views(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1] != 4 or views.shape[2] != 3:
            raise ValueError(f"views must have shape Bx4x3xHxW, got {tuple(views.shape)}")
        batch, num_views, channels, height, width = views.shape
        flat = views.reshape(batch * num_views, channels, height, width)
        return self.encoder(flat).reshape(batch, num_views, self.feature_dim)

    @staticmethod
    def _mean_fusion(
        features: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = view_mask.to(features.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(features * weights.unsqueeze(-1), dim=1), weights

    def _side_representation(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
        indices: tuple[int, int],
        token_id: int,
    ) -> torch.Tensor:
        side_features = features[:, list(indices), :]
        side_mask = view_mask[:, list(indices)]
        average = masked_mean(side_features, side_mask, dim=1)
        present = side_mask.any(dim=1, keepdim=True)
        token = self.missing_side_token[token_id].unsqueeze(0).expand_as(average)
        return torch.where(present, average, token)

    def _cd_representation(
        self,
        features: torch.Tensor,
        view_mask: torch.Tensor,
        main_exam_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_cd_specific_fusion:
            uniform = view_mask.to(features.dtype)
            uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp_min(1.0)
            return main_exam_repr, uniform

        ids = torch.arange(4, device=features.device)
        gate_input = features + self.cd_view_embedding(ids).unsqueeze(0)
        gate_logits = self.cd_gate_fc(gate_input).squeeze(-1)
        dtype_min = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill(~view_mask.bool(), dtype_min)
        cd_gate_weights = torch.softmax(gate_logits, dim=1)
        cd_fused = torch.sum(features * cd_gate_weights.unsqueeze(-1), dim=1)

        left = self._side_representation(features, view_mask, (0, 1), token_id=0)
        right = self._side_representation(features, view_mask, (2, 3), token_id=1)
        cd_input = torch.cat([cd_fused, left, right, torch.abs(left - right)], dim=1)
        return self.cd_project(cd_input), cd_gate_weights

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if view_mask.ndim != 2 or view_mask.shape[1] != 4:
            raise ValueError(f"view_mask must be Bx4, got {tuple(view_mask.shape)}")
        view_mask = view_mask.bool()
        if torch.any(view_mask.sum(dim=1) < 1):
            raise ValueError("All-view-missing cases are forbidden before fusion")

        features = self._encode_views(views)
        fused_all, gate_weights = self._mean_fusion(features, view_mask)
        gate_entropy = -(
            gate_weights.clamp_min(1e-12) * gate_weights.clamp_min(1e-12).log()
        ).sum(dim=1)

        exam_repr = self.exam_project(fused_all)
        base_direct_logits = self.direct_head(exam_repr)
        logits_coarse = self.head_coarse(exam_repr)
        logits_ab = self.head_ab(exam_repr)

        cd_repr, cd_gate_weights = self._cd_representation(features, view_mask, exam_repr)
        logits_cd = self.head_cd(cd_repr)

        final_logits = base_direct_logits
        if self.use_cd_residual:
            centered_cd = logits_cd - logits_cd.mean(dim=1, keepdim=True)
            correction = torch.zeros_like(base_direct_logits)
            correction[:, 2:4] = self.cd_residual_alpha * centered_cd
            final_logits = base_direct_logits + correction

        return {
            "features": features,
            "gate_weights": gate_weights,
            "cd_gate_weights": cd_gate_weights,
            "gate_entropy": gate_entropy,
            "exam_repr": exam_repr,
            "cd_repr": cd_repr,
            "base_direct_logits": base_direct_logits,
            "final_logits": final_logits,
            "final_log_probs": F.log_softmax(final_logits, dim=1),
            "logits_coarse": logits_coarse,
            "logits_ab": logits_ab,
            "logits_cd": logits_cd,
            "corn_logits": self.corn_head(exam_repr) if self.corn_head is not None else None,
        }


def build_model(
    config: dict[str, Any],
    pretrained_override: bool | None = None,
) -> ImprovedTask1DenseNet:
    return ImprovedTask1DenseNet(config, pretrained_override=pretrained_override)
