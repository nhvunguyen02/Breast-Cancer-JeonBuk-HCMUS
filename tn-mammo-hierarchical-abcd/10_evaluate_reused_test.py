#!/usr/bin/env python3
"""Exploratory TN-test evaluation after all three branches were frozen.

This is deliberately not named or reported as a new locked-test estimate because
TN test was already exposed in the earlier Task-1 H0 evaluation.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

architecture_mod = importlib.import_module("03_architecture")
metrics_mod = importlib.import_module("06_metrics")
train_mod = importlib.import_module("07_train_mixed")
valid_mod = importlib.import_module("08_evaluate_valid")

LABELS = ["A", "B", "C", "D"]


def require_frozen(root: Path, run_dir: Path, checkpoint: Path, config: Path) -> dict:
    freeze_path = root / "FROZEN_THREE_BRANCHES_BEFORE_REUSED_TEST.json"
    if not freeze_path.is_file():
        raise PermissionError(f"Missing freeze file: {freeze_path}")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("branch_design_fixed_before_test") is not True:
        raise PermissionError("Branch design was not frozen before test")
    matches = [item for item in freeze["branches"] if Path(item["checkpoint"]) == checkpoint]
    if len(matches) != 1:
        raise PermissionError("Checkpoint is not one of the frozen branches")
    item = matches[0]
    if train_mod.sha256_file(checkpoint) != item["checkpoint_sha256"]:
        raise PermissionError("Frozen checkpoint hash mismatch")
    if train_mod.sha256_file(config) != item["config_sha256"]:
        raise PermissionError("Frozen config hash mismatch")
    if Path(item["checkpoint"]).parent != run_dir:
        raise PermissionError("Run directory does not match frozen record")
    return freeze


def run(args: argparse.Namespace) -> None:
    freeze = require_frozen(args.root, args.run_dir, args.checkpoint, args.config)
    output_path = args.run_dir / "reused_test_metrics.json"
    if output_path.exists():
        raise RuntimeError(f"Reused-test evaluation already exists: {output_path}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = train_mod.load_torch_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "MIXED_TN_TARGET_TRAINING":
        raise RuntimeError("Checkpoint stage mismatch")
    if checkpoint["config"]["experiment"]["id"] != config["experiment"]["id"]:
        raise RuntimeError("Config/checkpoint experiment mismatch")

    manifest = args.resolved_dir / "resolved_tn_locked_test.csv"
    frame = pd.read_csv(manifest)
    if len(frame) != 132 or set(frame["split"].astype(str)) != {"test"}:
        raise ValueError(f"TN test contract mismatch: n={len(frame)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = architecture_mod.build_model(config, pretrained_override=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    loader = train_mod.make_loader(
        frame,
        config,
        training=False,
        seed=int(config["experiment"]["seed"]),
        smoke=False,
    )
    amp_enabled = bool(config["optimization"].get("amp", True)) and device.type == "cuda"
    inference = train_mod.infer_loader(model, loader, device, amp_enabled=amp_enabled)
    metrics = train_mod.compute_metrics_from_inference(config, inference, smoke=False)

    valid_mod.prediction_frame(inference).to_csv(
        args.run_dir / "reused_test_predictions.csv", index=False
    )
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        args.run_dir / "reused_test_confusion_matrix.csv"
    )
    valid_mod.plot_confusion(
        matrix,
        args.run_dir / "reused_test_confusion_matrix.png",
        "Exploratory reused TN-test confusion matrix",
    )
    report = {
        "status": "EXPLORATORY_REUSED_TN_TEST_COMPLETE",
        "scientific_status": "NOT_A_NEW_LOCKED_UNBIASED_ESTIMATE",
        "reason": "TN test was already exposed in the earlier Task-1 H0 run.",
        "all_three_branches_frozen_before_any_new_test": True,
        "freeze_timestamp_utc": freeze["timestamp_utc"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": train_mod.sha256_file(args.checkpoint),
        "metrics": metrics,
        "no_post_test_tuning_permitted": True,
    }
    train_mod.atomic_json(output_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--resolved-dir", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
