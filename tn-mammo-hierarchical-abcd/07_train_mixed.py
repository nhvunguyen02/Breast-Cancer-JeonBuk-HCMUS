#!/usr/bin/env python3
"""Target-aware mixed VinDr+TN training with TN-only validation.

All three branches start independently from the same ImageNet initialization.
Sampling contract: TN domain mass 0.60, VinDr mass 0.40, then uniform class
mass inside each domain.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dataset_mod = importlib.import_module("02_dataset")
architecture_mod = importlib.import_module("03_architecture")
loss_mod = importlib.import_module("04_hierarchical_loss")
metrics_mod = importlib.import_module("06_metrics")

LABELS = ["A", "B", "C", "D"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_torch_checkpoint(path: Path, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().to_dict()
    result = {label: int(counts.get(label, 0)) for label in LABELS}
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"Training split must contain every class: {result}")
    return result


def preprocess_config(config: dict[str, Any], training: bool) -> Any:
    data = config["data"]
    return dataset_mod.PreprocessConfig(
        image_size=int(data["image_size"]),
        interpolation=str(data.get("interpolation", "bilinear")),
        foreground_crop=bool(data.get("foreground_crop", False)),
        normalize=str(data.get("normalize", "robust_foreground")),
        view_dropout_prob=float(data.get("view_dropout_prob", 0.0)) if training else 0.0,
    )


def build_domain_class_sampler(
    frame: pd.DataFrame,
    tn_ratio: float,
    seed: int,
    num_samples: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    if not 0.0 < tn_ratio < 1.0:
        raise ValueError("tn_ratio must be in (0,1)")
    domain_mass = {"TN": float(tn_ratio), "VinDr": float(1.0 - tn_ratio)}
    counts = frame.groupby(["dataset", "label"]).size().to_dict()
    weights = []
    for row in frame.itertuples(index=False):
        key = (str(row.dataset), str(row.label))
        count = int(counts.get(key, 0))
        if count <= 0:
            raise ValueError(f"Missing domain/class group: {key}")
        weights.append(domain_mass[key[0]] / (4.0 * float(count)))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )
    audit = {
        "tn_ratio_target": tn_ratio,
        "vindr_ratio_target": 1.0 - tn_ratio,
        "num_samples_per_epoch": int(num_samples),
        "raw_domain_class_counts": {
            f"{domain}:{label}": int(value)
            for (domain, label), value in sorted(counts.items())
        },
    }
    return sampler, audit


def make_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    training: bool,
    seed: int,
    smoke: bool = False,
) -> DataLoader:
    data_cfg = config["data"]
    dataset = dataset_mod.FourViewMammoDataset(
        frame,
        preprocess=preprocess_config(config, training=training),
        training=training,
        seed=seed,
        strict_paths=True,
    )
    workers = 0 if smoke else int(data_cfg.get("num_workers", 4))
    batch_size = int(data_cfg["train_batch_size"] if training else data_cfg["eval_batch_size"])
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available(),
        "collate_fn": dataset_mod.collate_cases,
        "drop_last": False,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
    if training:
        samples_per_epoch = 32 if smoke else int(
            config["optimization"].get("samples_per_epoch", len(frame))
        )
        sampler, _ = build_domain_class_sampler(
            frame,
            tn_ratio=float(config["optimization"].get("tn_domain_ratio", 0.60)),
            seed=seed,
            num_samples=samples_per_epoch,
        )
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = False
    return DataLoader(**kwargs)


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    clipped = 0
    start = time.time()
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        views = batch["views"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        y4 = batch["y4"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            outputs = model(views, view_mask)
            total_loss, components = criterion(outputs, y4, epoch=epoch)
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch} batch={batch_index}")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(max_grad_norm)
        )
        if float(gradient_norm) > float(max_grad_norm):
            clipped += 1
        scaler.step(optimizer)
        scaler.update()
        count += 1
        for key, value in components.items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
    if count == 0:
        raise RuntimeError("Training loader produced zero batches")
    return {
        **{key: value / count for key, value in sums.items()},
        "batches": count,
        "gradient_clip_count": clipped,
        "seconds": time.time() - start,
    }


@torch.no_grad()
def infer_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    y_true: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    coarse: list[np.ndarray] = []
    ab: list[np.ndarray] = []
    cd: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    cd_gate_weights: list[np.ndarray] = []
    original_masks: list[np.ndarray] = []
    case_ids: list[str] = []
    labels: list[str] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        views = batch["views"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        with autocast_context(amp_enabled):
            outputs = model(views, view_mask)
        log_probs = outputs["final_log_probs"]
        if log_probs is None:
            raise RuntimeError("Model did not return final log probabilities")
        y_true.append(batch["y4"].cpu().numpy())
        probabilities.append(log_probs.exp().float().cpu().numpy())
        scores.append(log_probs.float().cpu().numpy())
        gate_weights.append(outputs["gate_weights"].float().cpu().numpy())
        cd_gate_weights.append(outputs["cd_gate_weights"].float().cpu().numpy())
        original_masks.append(batch["original_view_mask"].cpu().numpy().astype(np.int64))
        case_ids.extend(batch["case_id"])
        labels.extend(batch["label"])
        coarse.append(torch.softmax(outputs["logits_coarse"], dim=1).float().cpu().numpy())
        ab.append(torch.softmax(outputs["logits_ab"], dim=1).float().cpu().numpy())
        cd.append(torch.softmax(outputs["logits_cd"], dim=1).float().cpu().numpy())

    if not y_true:
        raise RuntimeError("Evaluation loader produced zero batches")
    return {
        "y_true": np.concatenate(y_true),
        "probabilities": np.concatenate(probabilities),
        "scores": np.concatenate(scores),
        "coarse_probabilities": np.concatenate(coarse),
        "ab_probabilities": np.concatenate(ab),
        "cd_probabilities": np.concatenate(cd),
        "gate_weights": np.concatenate(gate_weights),
        "cd_gate_weights": np.concatenate(cd_gate_weights),
        "original_view_mask": np.concatenate(original_masks),
        "case_id": case_ids,
        "label": labels,
    }


def compute_metrics_from_inference(
    config: dict[str, Any],
    inference: dict[str, Any],
    smoke: bool = False,
    bootstrap_samples_override: int | None = None,
) -> dict[str, Any]:
    selection = config["selection"]
    bootstrap_samples = (
        20
        if smoke
        else int(bootstrap_samples_override)
        if bootstrap_samples_override is not None
        else int(selection.get("bootstrap_samples", 1000))
    )
    final = metrics_mod.compute_case_metrics(
        inference["y_true"],
        inference["probabilities"],
        ece_bins=int(selection.get("ece_bins", 15)),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(config["experiment"].get("seed", 42)),
    )
    final["branches"] = metrics_mod.compute_branch_metrics(
        inference["y_true"],
        inference["coarse_probabilities"],
        inference["ab_probabilities"],
        inference["cd_probabilities"],
    )
    final["mean_gate_weights"] = {
        view: float(inference["gate_weights"][:, index].mean())
        for index, view in enumerate(dataset_mod.VIEW_ORDER)
    }
    final["mean_cd_gate_weights"] = {
        view: float(inference["cd_gate_weights"][:, index].mean())
        for index, view in enumerate(dataset_mod.VIEW_ORDER)
    }
    return final


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["qwk"]),
    )


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    counts: dict[str, int],
    epoch: int,
    metrics: dict[str, Any],
    vindr_manifest: Path,
    tn_manifest: Path,
) -> dict[str, Any]:
    return {
        "document_contract": "TNM-T1-IMPROVE-R1R3-2026-01",
        "stage": "MIXED_TN_TARGET_TRAINING",
        "state_dict": model.state_dict(),
        "config": config,
        "class_counts": counts,
        "best_epoch": int(epoch),
        "selection_metrics": metrics,
        "vindr_development_manifest": str(vindr_manifest),
        "vindr_development_manifest_sha256": sha256_file(vindr_manifest),
        "tn_development_manifest": str(tn_manifest),
        "tn_development_manifest_sha256": sha256_file(tn_manifest),
        "encoder_initialization": getattr(model.encoder, "initialization", "unknown"),
        "view_order": dataset_mod.VIEW_ORDER,
        "label_map": dataset_mod.LABEL_MAP,
        "selection_dataset": "TN validation only",
        "tn_test_used_during_training": False,
    }


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.run_dir / "resolved_config.yaml")

    vindr_manifest = args.resolved_dir / "resolved_vindr_dev.csv"
    tn_manifest = args.resolved_dir / "resolved_tn_dev.csv"
    if not vindr_manifest.is_file() or not tn_manifest.is_file():
        raise FileNotFoundError("Resolved development manifests are missing")

    vindr = pd.read_csv(vindr_manifest)
    tn = pd.read_csv(tn_manifest)
    if (vindr["split"] == "test").any() or (tn["split"] == "test").any():
        raise RuntimeError("Development manifests contain test rows")

    vindr_train = vindr[vindr["split"] == "train"].copy().reset_index(drop=True)
    tn_train = tn[tn["split"] == "train"].copy().reset_index(drop=True)
    tn_valid = tn[tn["split"] == "valid"].copy().reset_index(drop=True)
    if len(vindr_train) != 3975 or len(tn_train) != 411 or len(tn_valid) != 133:
        raise ValueError(
            f"Split mismatch VinDr={len(vindr_train)} TN train={len(tn_train)} valid={len(tn_valid)}"
        )

    vindr_train["dataset"] = "VinDr"
    tn_train["dataset"] = "TN"
    tn_valid["dataset"] = "TN"
    mixed_train = pd.concat([tn_train, vindr_train], ignore_index=True)

    if args.smoke:
        mixed_train = (
            mixed_train.groupby(["dataset", "label"], group_keys=False)
            .head(2)
            .reset_index(drop=True)
        )
        tn_valid = tn_valid.groupby("label", group_keys=False).head(1).reset_index(drop=True)

    mixed_train.to_csv(args.run_dir / "mixed_train.csv", index=False)
    tn_valid.to_csv(args.run_dir / "tn_valid.csv", index=False)

    samples_per_epoch = 32 if args.smoke else int(
        config["optimization"].get("samples_per_epoch", len(mixed_train))
    )
    _, sampler_audit = build_domain_class_sampler(
        mixed_train,
        tn_ratio=float(config["optimization"].get("tn_domain_ratio", 0.60)),
        seed=seed,
        num_samples=samples_per_epoch,
    )
    atomic_json(args.run_dir / "mixed_sampler_audit.json", sampler_audit)

    train_loader = make_loader(mixed_train, config, training=True, seed=seed, smoke=args.smoke)
    valid_loader = make_loader(tn_valid, config, training=False, seed=seed + 1, smoke=args.smoke)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = architecture_mod.build_model(config).to(device)
    counts = class_counts(mixed_train)
    criterion = loss_mod.HierarchicalTaskLoss(config, counts).to(device)

    optimization = config["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["lr"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    epochs = 1 if args.smoke else int(optimization["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(optimization.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    patience = 1 if args.smoke else int(optimization.get("early_stopping_patience", 6))

    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = args.run_dir / ("smoke_best_checkpoint.pt" if args.smoke else "best_tn_checkpoint.pt")
    metrics_path = args.run_dir / "epoch_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            max_grad_norm=float(optimization.get("max_grad_norm", 1.0)),
            amp_enabled=amp_enabled,
            max_batches=2 if args.smoke else args.max_train_batches,
        )
        scheduler.step()
        inference = infer_loader(
            model,
            valid_loader,
            device,
            amp_enabled=amp_enabled,
            max_batches=1 if args.smoke else args.max_valid_batches,
        )
        valid_metrics = compute_metrics_from_inference(
            config,
            inference,
            smoke=args.smoke,
            bootstrap_samples_override=0,
        )
        current_key = selection_key(valid_metrics)
        improved = current_key > best_key
        if improved:
            best_key = current_key
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                checkpoint_payload(
                    model,
                    config,
                    counts,
                    epoch,
                    valid_metrics,
                    vindr_manifest,
                    tn_manifest,
                ),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        record = {
            "stage": "MIXED_TN_TARGET_TRAINING",
            "experiment_id": config["experiment"]["id"],
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_stats,
            "valid": valid_metrics,
            "improved": improved,
            "best_epoch": best_epoch,
            "tn_test_used": False,
        }
        append_jsonl(metrics_path, record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= patience:
            print(f"[EARLY_STOP] epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("No checkpoint was saved")
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    reloaded = infer_loader(model, valid_loader, device, amp_enabled=amp_enabled)
    metrics = compute_metrics_from_inference(
        config,
        reloaded,
        smoke=args.smoke,
        bootstrap_samples_override=0,
    )
    pointer = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics_after_reload": metrics,
        "selection_dataset": "TN validation only",
        "tn_test_used": False,
    }
    marker = args.run_dir / ("SMOKE_PASS.json" if args.smoke else "TRAINING_DONE.json")
    atomic_json(marker, {"status": "PASS", **pointer})
    if not args.smoke:
        atomic_json(args.run_dir / "tn_checkpoint_pointer.json", pointer)
    print(json.dumps({"status": "PASS", "checkpoint": str(checkpoint_path), "smoke": args.smoke}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--resolved-dir", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--max-train-batches", type=int)
    result.add_argument("--max-valid-batches", type=int)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
