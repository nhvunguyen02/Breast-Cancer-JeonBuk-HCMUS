# -*- coding: utf-8 -*-
"""Vòng lặp train/validate, chọn checkpoint theo Macro-F1 validation."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from tn_mammo.data.dataset import FourViewManifestDataset
from tn_mammo.data.sampler import build_domain_sampler
from tn_mammo.losses.multitask import MultiTaskCriterion
from tn_mammo.metrics.classification import compute_metrics
from tn_mammo.models.density_model import FourViewDensityModel
from tn_mammo.utils.seeding import seed_everything, seed_worker


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp):
    model.train()
    total_loss, total_samples = 0.0, 0

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            loss = criterion(model(views), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach()) * labels.shape[0]
        total_samples += labels.shape[0]

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        views = batch["views"].to(device, non_blocking=True)
        labels = batch["label"].long()

        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            outputs = model(views)

        # Dự đoán cuối = argmax flat head (CORAL không tham gia decode).
        predictions = outputs["flat_logits"].float().argmax(dim=1).cpu()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

    return compute_metrics(y_true, y_pred)


def run_training(config: dict, output_dir: Path) -> None:
    seed = int(config["experiment"]["seed"])
    deterministic = bool(config["experiment"].get("deterministic", True))
    seed_everything(seed, deterministic=deterministic)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config["data"]["image_size"])
    tn_train = FourViewManifestDataset(
        config["data"]["train"]["tn_manifest"], image_size, training=True
    )
    vindr_train = FourViewManifestDataset(
        config["data"]["train"]["vindr_manifest"], image_size, training=True
    )
    tn_valid = FourViewManifestDataset(
        config["data"]["validation"]["manifest"], image_size, training=False
    )

    combined = ConcatDataset([tn_train, vindr_train])
    sampler = build_domain_sampler(
        ["TN"] * len(tn_train) + ["VinDr"] * len(vindr_train),
        tn_ratio=float(config["data"]["tn_domain_ratio"]),
        num_samples=int(
            config["training"].get("sampler_num_samples", len(combined))
        ),
        generator=generator,
    )

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        combined, batch_size=batch_size, sampler=sampler, **loader_kwargs
    )
    valid_loader = DataLoader(
        tn_valid, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    model = FourViewDensityModel(
        use_ordinal_head=bool(config["model"].get("use_ordinal_head", True)),
        imagenet_init=bool(config["model"].get("imagenet_init", False)),
    )

    # E1 khởi tạo từ checkpoint E0 (Phase-G): thiếu key ordinal_head là hợp lệ.
    init_checkpoint = config["model"].get("initialization_checkpoint")
    if init_checkpoint:
        state = torch.load(
            init_checkpoint, map_location="cpu", weights_only=True
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            state.get("model_state_dict", state), strict=False
        )
        allowed_missing = {
            key for key in missing_keys if key.startswith("ordinal_head.")
        }
        disallowed_missing = set(missing_keys) - allowed_missing
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint khởi tạo không tương thích: "
                f"missing={sorted(disallowed_missing)}, "
                f"unexpected={sorted(unexpected_keys)}"
            )

    model = model.to(device)
    criterion = MultiTaskCriterion(
        beta=float(config["loss"]["flat"]["beta"]),
        gamma=float(config["loss"]["flat"]["gamma"]),
        lambda_ordinal=float(config["loss"]["lambda_ordinal"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["training"]["scheduler"]["step_size"]),
        gamma=float(config["training"]["scheduler"]["gamma"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_macro_f1, patience_counter = -math.inf, 0
    patience = int(config["training"]["early_stopping_patience"])

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp
        )
        metrics = evaluate(model, valid_loader, device, amp)
        scheduler.step()

        improved = metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = metrics["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "valid_metrics": metrics,
                    "config": config,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                output_dir / "best_checkpoint.pt",
            )
        else:
            patience_counter += 1

        print(json.dumps({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "valid_macro_f1": round(metrics["macro_f1"], 4),
            "best_macro_f1": round(best_macro_f1, 4),
            "improved": improved,
        }))

        if patience_counter >= patience:
            print(f"[EARLY_STOP] epoch={epoch}")
            break
