# -*- coding: utf-8 -*-
"""Đánh giá checkpoint đã chọn trên một manifest.

Bản gốc inference.py còn kiểm SHA256 checkpoint, đối chiếu lại metric
validation trước khi mở test khóa đúng MỘT lần — ở đây rút gọn thành
đánh giá thường.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tn_mammo.data.dataset import FourViewManifestDataset
from tn_mammo.models.density_model import FourViewDensityModel
from tn_mammo.training.engine import evaluate


def run_eval(checkpoint_path: Path, manifest_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    use_ordinal_head = bool(model_config.get("use_ordinal_head", True))
    image_size = int(data_config.get("image_size", 224))
    batch_size = int(training_config.get("eval_batch_size", 2))
    num_workers = int(training_config.get("num_workers", 4))

    model = FourViewDensityModel(use_ordinal_head=use_ordinal_head)
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint), strict=True
    )
    model = model.to(device)

    loader = DataLoader(
        FourViewManifestDataset(
            manifest_path, image_size=image_size, training=False
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    metrics = evaluate(model, loader, device, amp=device.type == "cuda")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
