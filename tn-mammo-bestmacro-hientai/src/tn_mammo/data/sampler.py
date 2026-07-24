# -*- coding: utf-8 -*-
"""Sampler trộn domain TN / VinDr (tn_domain_ratio = 0.6)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_domain_sampler(
    domains: list[str],
    tn_ratio: float,
    num_samples: int,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Trọng số sao cho tổng xác suất lấy mẫu domain TN đúng bằng tn_ratio."""
    if not 0.0 < tn_ratio < 1.0:
        raise ValueError(f"tn_ratio phải nằm trong (0, 1), nhận được {tn_ratio}")
    if num_samples <= 0:
        raise ValueError("num_samples phải > 0")

    domains_array = np.char.upper(np.char.strip(np.asarray(domains, dtype=str)))
    tn_mask = domains_array == "TN"
    tn_count = int(tn_mask.sum())
    other_count = len(domains) - tn_count
    if tn_count == 0 or other_count == 0:
        raise ValueError(
            f"Sampler cần cả TN và domain ngoài; tn_count={tn_count}, "
            f"other_count={other_count}"
        )

    weights = np.where(
        tn_mask,
        tn_ratio / tn_count,
        (1.0 - tn_ratio) / other_count,
    )

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
