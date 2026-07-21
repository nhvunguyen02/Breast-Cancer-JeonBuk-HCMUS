#!/usr/bin/env python3
"""Freeze all three branches before any reused TN-test evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

BRANCHES = ["R1_soft_hierarchy", "R2_cd_residual", "R3_cd_specific_fusion"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path) -> None:
    records = []
    for branch in BRANCHES:
        run_dir = root / branch
        config_path = run_dir / "resolved_config.yaml"
        checkpoint_path = run_dir / "best_tn_checkpoint.pt"
        valid_path = run_dir / "valid_metrics.json"
        done_path = run_dir / "VALIDATION_DONE.json"
        for path in [config_path, checkpoint_path, valid_path, done_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        if (run_dir / "reused_test_metrics.json").exists():
            raise RuntimeError(f"Reused-test result already exists before freeze: {run_dir}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        valid = json.loads(valid_path.read_text(encoding="utf-8"))
        records.append(
            {
                "branch": branch,
                "experiment_id": config["experiment"]["id"],
                "architecture": config["experiment"]["architecture"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "config": str(config_path),
                "config_sha256": sha256(config_path),
                "valid_metrics": str(valid_path),
                "valid_metrics_sha256": sha256(valid_path),
                "tn_valid_macro_f1": float(valid["macro_f1"]),
                "tn_valid_balanced_accuracy": float(valid["balanced_accuracy"]),
                "tn_valid_qwk": float(valid["qwk"]),
            }
        )

    payload = {
        "status": "ALL_THREE_BRANCHES_FROZEN_BEFORE_REUSED_TEST",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_dataset": "TN validation only",
        "branch_design_fixed_before_test": True,
        "test_status": (
            "TN test was already opened in the earlier Task-1 H0 run. "
            "Any new TN-test results are exploratory reused-test measurements, "
            "not an unbiased locked-test estimate."
        ),
        "no_post_test_tuning_permitted": True,
        "branches": records,
    }
    output = root / "FROZEN_THREE_BRANCHES_BEFORE_REUSED_TEST.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "branches": records}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.root)
