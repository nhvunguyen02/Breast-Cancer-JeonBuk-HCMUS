"""Misc training utilities: seeding, param counting, model size, benchmark IO."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_model_size_mb(model, path):
    torch.save(model.state_dict(), path)
    mb = Path(path).stat().st_size / (1024 ** 2)
    try:
        Path(path).unlink()
    except Exception:
        pass
    return mb


def append_benchmark(row, benchmark_path):
    benchmark_path = Path(benchmark_path)
    if benchmark_path.exists():
        old = pd.read_csv(benchmark_path)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])
    new.to_csv(benchmark_path, index=False)
    return new
