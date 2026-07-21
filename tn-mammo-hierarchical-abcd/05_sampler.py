#!/usr/bin/env python3
"""Hierarchical case sampler: domain -> AB/CD -> class -> case."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

LABEL_TO_GROUP = {"A": "AB", "B": "AB", "C": "CD", "D": "CD"}
GROUP_CLASSES = {"AB": ["A", "B"], "CD": ["C", "D"]}


@dataclass(frozen=True)
class SamplerAudit:
    target_domain: dict[str, float]
    target_group: dict[str, float]
    target_class: dict[str, float]
    observed_domain: dict[str, float]
    observed_group: dict[str, float]
    observed_class: dict[str, float]
    max_abs_class_error: float


def _normalize_distribution(values: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0:
        raise ValueError("Distribution mass must be positive")
    return {str(key): float(value) / total for key, value in values.items()}


def compute_hierarchical_sample_weights(
    frame: pd.DataFrame,
    domain_targets: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
    required = {"dataset", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Sampler frame is missing columns: {sorted(missing)}")
    labels = set(frame["label"].astype(str))
    if not labels.issubset(set(LABEL_TO_GROUP)):
        raise ValueError(f"Unsupported labels for sampler: {sorted(labels)}")

    domains = sorted(frame["dataset"].astype(str).unique().tolist())
    if domain_targets is None:
        domain_distribution = {domain: 1.0 / len(domains) for domain in domains}
    else:
        domain_distribution = _normalize_distribution({domain: float(domain_targets[domain]) for domain in domains})

    target_group = {"AB": 0.5, "CD": 0.5}
    target_class_within_group = {
        "A": 0.5,
        "B": 0.5,
        "C": 0.5,
        "D": 0.5,
    }
    counts = frame.groupby([frame["dataset"].astype(str), frame["label"].astype(str)]).size().to_dict()
    weights = []
    target_class_global: dict[str, float] = {label: 0.0 for label in LABEL_TO_GROUP}

    for _, row in frame.iterrows():
        domain = str(row["dataset"])
        label = str(row["label"])
        group = LABEL_TO_GROUP[label]
        count = int(counts[(domain, label)])
        desired_mass = (
            domain_distribution[domain]
            * target_group[group]
            * target_class_within_group[label]
        )
        weights.append(desired_mass / max(count, 1))
        target_class_global[label] += desired_mass / len(domains)

    tensor = torch.tensor(weights, dtype=torch.double)
    if not torch.isfinite(tensor).all() or torch.any(tensor <= 0):
        raise ValueError("Sampler produced non-positive or non-finite weights")
    audit_targets = {
        "domain": domain_distribution,
        "group": target_group,
        "class": {label: 0.25 for label in ["A", "B", "C", "D"]},
    }
    return tensor, audit_targets


def build_hierarchical_sampler(
    frame: pd.DataFrame,
    seed: int,
    num_samples: int | None = None,
    domain_targets: Mapping[str, float] | None = None,
) -> tuple[WeightedRandomSampler, dict[str, dict[str, float]]]:
    weights, targets = compute_hierarchical_sample_weights(frame, domain_targets=domain_targets)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=int(num_samples or len(frame)),
        replacement=True,
        generator=generator,
    )
    return sampler, targets


def audit_sampler(
    frame: pd.DataFrame,
    seed: int = 42,
    draws: int = 20_000,
    domain_targets: Mapping[str, float] | None = None,
) -> SamplerAudit:
    weights, targets = compute_hierarchical_sample_weights(frame, domain_targets=domain_targets)
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.multinomial(weights / weights.sum(), num_samples=int(draws), replacement=True, generator=generator)
    sampled = frame.iloc[indices.cpu().numpy()].copy()
    sampled["group"] = sampled["label"].map(LABEL_TO_GROUP)

    def frequencies(series: pd.Series) -> dict[str, float]:
        values = series.astype(str).value_counts(normalize=True).sort_index()
        return {str(key): float(value) for key, value in values.items()}

    observed_domain = frequencies(sampled["dataset"])
    observed_group = frequencies(sampled["group"])
    observed_class = frequencies(sampled["label"])
    max_error = max(abs(observed_class.get(label, 0.0) - 0.25) for label in ["A", "B", "C", "D"])
    return SamplerAudit(
        target_domain=targets["domain"],
        target_group=targets["group"],
        target_class=targets["class"],
        observed_domain=observed_domain,
        observed_group=observed_group,
        observed_class=observed_class,
        max_abs_class_error=float(max_error),
    )
