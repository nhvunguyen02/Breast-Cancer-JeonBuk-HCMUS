from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FusionResult:
    """Common output contract for all four-view fusion modules."""

    exam_features: torch.Tensor
    left_features: torch.Tensor
    right_features: torch.Tensor
    left_gate_weights: torch.Tensor
    right_gate_weights: torch.Tensor
    bilateral_gate_weights: torch.Tensor | None = None


def _validate_view_features(
    view_features: torch.Tensor,
    feature_dim: int,
) -> None:
    if view_features.ndim != 3:
        raise ValueError(
            "view_features must have shape "
            "[batch, 4, feature_dim]."
        )

    if view_features.shape[1] != 4:
        raise ValueError(
            "Exactly four views are required in order "
            "[L_CC, L_MLO, R_CC, R_MLO]."
        )

    if view_features.shape[2] != feature_dim:
        raise ValueError(
            "Unexpected feature dimension: "
            f"{view_features.shape[2]}."
        )


class MeanFourViewFusion(nn.Module):
    """Parameter-free Phase-G-compatible mean fusion."""

    def __init__(
        self,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)

    def forward(
        self,
        view_features: torch.Tensor,
    ) -> FusionResult:
        _validate_view_features(
            view_features,
            self.feature_dim,
        )

        left = view_features[:, 0:2].mean(dim=1)
        right = view_features[:, 2:4].mean(dim=1)
        exam = (left + right) / 2.0

        batch_size = view_features.shape[0]

        fixed_gates = view_features.new_full(
            (batch_size, 2),
            0.5,
        )

        return FusionResult(
            exam_features=exam,
            left_features=left,
            right_features=right,
            left_gate_weights=fixed_gates,
            right_gate_weights=fixed_gates.clone(),
        )


class ParameterMatchedMLPFusion(nn.Module):
    """Unstructured four-view MLP parameter control.

    The module concatenates all four 1024-d view features. It has
    approximately the same parameter count as the ipsilateral module
    but contains no explicit CC-MLO or left-right structure.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 608,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(
                self.feature_dim * 4,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_dim,
                self.feature_dim,
            ),
        )

        self.output_norm = nn.LayerNorm(
            self.feature_dim
        )

    def forward(
        self,
        view_features: torch.Tensor,
    ) -> FusionResult:
        _validate_view_features(
            view_features,
            self.feature_dim,
        )

        left = view_features[:, 0:2].mean(dim=1)
        right = view_features[:, 2:4].mean(dim=1)

        baseline_mean = view_features.mean(dim=1)

        flattened = view_features.reshape(
            view_features.shape[0],
            self.feature_dim * 4,
        )

        update = self.mlp(flattened)

        exam = self.output_norm(
            baseline_mean + update
        )

        batch_size = view_features.shape[0]

        fixed_gates = view_features.new_full(
            (batch_size, 2),
            0.5,
        )

        return FusionResult(
            exam_features=exam,
            left_features=left,
            right_features=right,
            left_gate_weights=fixed_gates,
            right_gate_weights=fixed_gates.clone(),
        )


class SharedIpsilateralPairFusion(nn.Module):
    """Shared gated relational fusion for one breast.

    The same module instance is applied to:
      - L_CC and L_MLO
      - R_CC and R_MLO

    Relation terms:
      - case-specific gated CC/MLO average
      - absolute CC-MLO difference
      - element-wise CC-MLO product
    """

    def __init__(
        self,
        feature_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)

        self.view_gate = nn.Linear(
            self.feature_dim,
            1,
        )

        self.relation_projection = nn.Linear(
            self.feature_dim * 3,
            self.feature_dim,
        )

        self.output_norm = nn.LayerNorm(
            self.feature_dim
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cc_features: torch.Tensor,
        mlo_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if cc_features.shape != mlo_features.shape:
            raise ValueError(
                "CC and MLO feature shapes must match."
            )

        if (
            cc_features.ndim != 2
            or cc_features.shape[1]
            != self.feature_dim
        ):
            raise ValueError(
                "Pair features must have shape "
                "[batch, feature_dim]."
            )

        pair = torch.stack(
            [
                cc_features,
                mlo_features,
            ],
            dim=1,
        )

        gate_logits = self.view_gate(
            pair
        ).squeeze(-1)

        gate_weights = torch.softmax(
            gate_logits,
            dim=1,
        )

        weighted_pair = (
            gate_weights.unsqueeze(-1)
            * pair
        ).sum(dim=1)

        absolute_difference = torch.abs(
            cc_features - mlo_features
        )

        elementwise_product = (
            cc_features * mlo_features
        )

        relation_features = torch.cat(
            [
                weighted_pair,
                absolute_difference,
                elementwise_product,
            ],
            dim=1,
        )

        relation_update = (
            self.relation_projection(
                relation_features
            )
        )

        relation_update = F.gelu(
            relation_update
        )

        relation_update = self.dropout(
            relation_update
        )

        fused = self.output_norm(
            weighted_pair + relation_update
        )

        return fused, gate_weights


class IpsilateralFourViewFusion(nn.Module):
    """Ipsilateral-first hierarchical four-view fusion."""

    def __init__(
        self,
        feature_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)

        # One shared module for both breasts.
        self.ipsilateral_fusion = (
            SharedIpsilateralPairFusion(
                feature_dim=self.feature_dim,
                dropout=dropout,
            )
        )

    def forward(
        self,
        view_features: torch.Tensor,
    ) -> FusionResult:
        _validate_view_features(
            view_features,
            self.feature_dim,
        )

        left, left_gates = (
            self.ipsilateral_fusion(
                view_features[:, 0],
                view_features[:, 1],
            )
        )

        right, right_gates = (
            self.ipsilateral_fusion(
                view_features[:, 2],
                view_features[:, 3],
            )
        )

        exam = (left + right) / 2.0

        return FusionResult(
            exam_features=exam,
            left_features=left,
            right_features=right,
            left_gate_weights=left_gates,
            right_gate_weights=right_gates,
        )



class LightweightBilateralRelationFusion(nn.Module):
    """Fuse left and right breast representations.

    Inputs:
        left_features:  [B, D]
        right_features: [B, D]

    Relations:
        - learned left/right gate
        - bilateral mean
        - absolute bilateral difference
        - element-wise bilateral product

    No hard symmetry or asymmetry loss is applied.
    """

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(
            bottleneck_dim
        )

        # Shared scalar scorer applied to both sides.
        self.side_gate = nn.Linear(
            self.feature_dim,
            1,
        )

        self.relation_mlp = nn.Sequential(
            nn.Linear(
                self.feature_dim * 3,
                self.bottleneck_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.bottleneck_dim,
                self.feature_dim,
            ),
        )

        self.output_norm = nn.LayerNorm(
            self.feature_dim
        )

    def forward(
        self,
        left_features: torch.Tensor,
        right_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if left_features.shape != right_features.shape:
            raise ValueError(
                "Left and right feature shapes "
                "must match."
            )

        if (
            left_features.ndim != 2
            or left_features.shape[1]
            != self.feature_dim
        ):
            raise ValueError(
                "Bilateral features must have shape "
                "[batch, feature_dim]."
            )

        sides = torch.stack(
            [
                left_features,
                right_features,
            ],
            dim=1,
        )

        gate_logits = self.side_gate(
            sides
        ).squeeze(-1)

        gate_weights = torch.softmax(
            gate_logits,
            dim=1,
        )

        gated_bilateral = (
            gate_weights.unsqueeze(-1)
            * sides
        ).sum(dim=1)

        bilateral_mean = (
            left_features + right_features
        ) / 2.0

        bilateral_difference = torch.abs(
            left_features - right_features
        )

        bilateral_product = (
            left_features * right_features
        )

        relation_features = torch.cat(
            [
                bilateral_mean,
                bilateral_difference,
                bilateral_product,
            ],
            dim=1,
        )

        relation_update = self.relation_mlp(
            relation_features
        )

        exam_features = self.output_norm(
            gated_bilateral + relation_update
        )

        return exam_features, gate_weights


class BilateralFourViewFusion(nn.Module):
    """Shared ipsilateral fusion followed by bilateral fusion."""

    def __init__(
        self,
        feature_dim: int,
        dropout: float = 0.1,
        bilateral_bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)

        # This name intentionally matches the E3 module path,
        # allowing exact loading of E3 ipsilateral weights.
        self.ipsilateral_fusion = (
            SharedIpsilateralPairFusion(
                feature_dim=self.feature_dim,
                dropout=dropout,
            )
        )

        self.bilateral_fusion = (
            LightweightBilateralRelationFusion(
                feature_dim=self.feature_dim,
                bottleneck_dim=(
                    bilateral_bottleneck_dim
                ),
                dropout=dropout,
            )
        )

    def forward(
        self,
        view_features: torch.Tensor,
    ) -> FusionResult:
        _validate_view_features(
            view_features,
            self.feature_dim,
        )

        left, left_gates = (
            self.ipsilateral_fusion(
                view_features[:, 0],
                view_features[:, 1],
            )
        )

        right, right_gates = (
            self.ipsilateral_fusion(
                view_features[:, 2],
                view_features[:, 3],
            )
        )

        exam, bilateral_gates = (
            self.bilateral_fusion(
                left,
                right,
            )
        )

        return FusionResult(
            exam_features=exam,
            left_features=left,
            right_features=right,
            left_gate_weights=left_gates,
            right_gate_weights=right_gates,
            bilateral_gate_weights=(
                bilateral_gates
            ),
        )

def build_four_view_fusion(
    *,
    name: str,
    feature_dim: int,
    dropout: float = 0.1,
    control_hidden_dim: int = 608,
    bilateral_bottleneck_dim: int = 256,
) -> nn.Module:
    normalized = str(name).strip().lower()

    if normalized == "mean":
        return MeanFourViewFusion(
            feature_dim=feature_dim
        )

    if normalized in {
        "control",
        "parameter_control",
        "parameter_matched_mlp",
    }:
        return ParameterMatchedMLPFusion(
            feature_dim=feature_dim,
            hidden_dim=control_hidden_dim,
            dropout=dropout,
        )

    if normalized in {
        "ipsilateral",
        "ipsilateral_gated_relational",
    }:
        return IpsilateralFourViewFusion(
            feature_dim=feature_dim,
            dropout=dropout,
        )

    if normalized in {
        "bilateral",
        "bilateral_gated_relational",
    }:
        return BilateralFourViewFusion(
            feature_dim=feature_dim,
            dropout=dropout,
            bilateral_bottleneck_dim=(
                bilateral_bottleneck_dim
            ),
        )

    raise ValueError(
        "Unsupported fusion name: "
        f"{name!r}. Expected mean, control, "
        "or ipsilateral."
    )
