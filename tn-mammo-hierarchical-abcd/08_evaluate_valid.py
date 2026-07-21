#!/usr/bin/env python3
"""TN validation-only evaluation for the three improvement branches."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

LABELS = ["A", "B", "C", "D"]
VIEW_ORDER = ["L_CC", "L_MLO", "R_CC", "R_MLO"]


def plot_confusion(matrix: np.ndarray, output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(np.arange(4), labels=LABELS)
    axis.set_yticks(np.arange(4), labels=LABELS)
    axis.set_xlabel("Predicted density")
    axis.set_ylabel("True density")
    axis.set_title(title)
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(4):
        for column in range(4):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def prediction_frame(inference: dict) -> pd.DataFrame:
    probabilities = inference["probabilities"]
    predictions = probabilities.argmax(axis=1)
    records = pd.DataFrame(
        {
            "case_id": inference["case_id"],
            "true_label": [LABELS[index] for index in inference["y_true"]],
            "true_id": inference["y_true"],
            "pred_label": [LABELS[index] for index in predictions],
            "pred_id": predictions,
            "prob_A": probabilities[:, 0],
            "prob_B": probabilities[:, 1],
            "prob_C": probabilities[:, 2],
            "prob_D": probabilities[:, 3],
            "confidence": probabilities.max(axis=1),
            "ordinal_abs_error": np.abs(predictions - inference["y_true"]),
        }
    )
    for index, view in enumerate(VIEW_ORDER):
        records[f"gate_{view}"] = inference["gate_weights"][:, index]
        records[f"cd_gate_{view}"] = inference["cd_gate_weights"][:, index]
        records[f"present_{view}"] = inference["original_view_mask"][:, index]
    records["prob_AB"] = inference["coarse_probabilities"][:, 0]
    records["prob_CD"] = inference["coarse_probabilities"][:, 1]
    records["prob_A_given_AB"] = inference["ab_probabilities"][:, 0]
    records["prob_B_given_AB"] = inference["ab_probabilities"][:, 1]
    records["prob_C_given_CD"] = inference["cd_probabilities"][:, 0]
    records["prob_D_given_CD"] = inference["cd_probabilities"][:, 1]
    return records


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = train_mod.load_torch_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint.get("stage") != "MIXED_TN_TARGET_TRAINING":
        raise RuntimeError("Checkpoint was not produced by mixed TN-target training")
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None or checkpoint_config["experiment"]["id"] != config["experiment"]["id"]:
        raise RuntimeError("Evaluation config does not match checkpoint")
    if checkpoint.get("tn_test_used_during_training", False):
        raise RuntimeError("Checkpoint metadata indicates TN test use during development")

    manifest = args.resolved_dir / "resolved_tn_dev.csv"
    frame = pd.read_csv(manifest)
    if (frame["split"] == "test").any():
        raise RuntimeError("Development manifest contains test rows")
    valid_frame = frame[frame["split"] == "valid"].copy().reset_index(drop=True)
    if len(valid_frame) != 133:
        raise ValueError(f"Expected 133 TN validation cases, got {len(valid_frame)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = architecture_mod.build_model(config, pretrained_override=False).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    loader = train_mod.make_loader(
        valid_frame,
        config,
        training=False,
        seed=int(config["experiment"]["seed"]),
        smoke=False,
    )
    amp_enabled = bool(config["optimization"].get("amp", True)) and device.type == "cuda"
    inference = train_mod.infer_loader(model, loader, device, amp_enabled=amp_enabled)
    metrics = train_mod.compute_metrics_from_inference(config, inference, smoke=False)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame(inference).to_csv(args.run_dir / "valid_predictions.csv", index=False)
    train_mod.atomic_json(args.run_dir / "valid_metrics.json", metrics)
    pd.DataFrame(metrics_mod.classwise_table(metrics)).to_csv(
        args.run_dir / "classwise_report.csv", index=False
    )
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        args.run_dir / "valid_confusion_matrix.csv"
    )
    plot_confusion(
        matrix,
        args.run_dir / "valid_confusion_matrix.png",
        "TN validation confusion matrix",
    )
    report = {
        "status": "PASS",
        "selection_metric": "TN validation macro-F1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": train_mod.sha256_file(args.checkpoint),
        "tn_test_used": False,
        "metrics": metrics,
    }
    train_mod.atomic_json(args.run_dir / "VALIDATION_DONE.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--resolved-dir", type=Path, required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
