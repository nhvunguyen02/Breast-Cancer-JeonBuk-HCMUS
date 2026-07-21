#!/usr/bin/env python3
"""Generate validation dashboards and combined validation/reused-test tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.backends.backend_pdf import PdfPages

LABELS = ["A", "B", "C", "D"]
BRANCHES = ["R1_soft_hierarchy", "R2_cd_residual", "R3_cd_specific_fusion"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> pd.DataFrame:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("stage") == "MIXED_TN_TARGET_TRAINING":
                rows.append(
                    {
                        "epoch": record.get("epoch"),
                        "macro_f1": record.get("valid", {}).get("macro_f1"),
                        "balanced_accuracy": record.get("valid", {}).get("balanced_accuracy"),
                        "qwk": record.get("valid", {}).get("qwk"),
                        "loss_total": record.get("train", {}).get("loss_total"),
                    }
                )
    return pd.DataFrame(rows)


def run_dashboard(run_dir: Path) -> Path:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    metrics = read_json(run_dir / "valid_metrics.json")
    history = read_history(run_dir / "epoch_metrics.jsonl")
    output = run_dir / "validation_dashboard.pdf"
    with PdfPages(output) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3))
        figure.suptitle("TN-Mammo hierarchy improvement - validation dashboard", fontsize=17)
        lines = [
            f"Experiment: {config['experiment']['id']}",
            f"Architecture: {config['experiment']['architecture']}",
            f"Primary fusion: {config['experiment']['fusion']}",
            f"C/D residual alpha: {config['model'].get('cd_residual_alpha', 0.0)}",
            f"TN domain ratio: {config['optimization'].get('tn_domain_ratio')}",
            "",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Macro-F1: {metrics['macro_f1']:.4f}",
            f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
            f"QWK: {metrics['qwk']:.4f}",
            f"C->D / D->C: {metrics['C_to_D']} / {metrics['D_to_C']}",
            f"Recall A/B/C/D: " + "/".join(f"{metrics['class_recall'][x]:.3f}" for x in LABELS),
            "",
            "Selection boundary: TN validation only.",
            "No TN-test result was used for training or checkpoint selection.",
        ]
        figure.text(0.07, 0.88, "\n".join(lines), va="top", family="monospace", fontsize=11)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        figure, axis = plt.subplots(figsize=(8.0, 6.5))
        image = axis.imshow(matrix)
        figure.colorbar(image, ax=axis)
        axis.set_xticks(np.arange(4), labels=LABELS)
        axis.set_yticks(np.arange(4), labels=LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("TN validation confusion matrix")
        for row in range(4):
            for col in range(4):
                axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if not history.empty:
            figure, axis = plt.subplots(figsize=(10.0, 5.8))
            axis.plot(history["epoch"], history["macro_f1"], marker="o", label="Macro-F1")
            axis.plot(history["epoch"], history["balanced_accuracy"], marker="o", label="Balanced accuracy")
            axis.plot(history["epoch"], history["qwk"], marker="o", label="QWK")
            axis.set_xlabel("Epoch")
            axis.set_ylim(-0.05, 1.0)
            axis.set_title("TN validation trajectory")
            axis.legend()
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    return output


def aggregate(root: Path) -> Path:
    rows = []
    for branch in BRANCHES:
        run_dir = root / branch
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        valid = read_json(run_dir / "valid_metrics.json")
        test_path = run_dir / "reused_test_metrics.json"
        test = read_json(test_path)["metrics"] if test_path.is_file() else None
        row = {
            "branch": branch,
            "experiment_id": config["experiment"]["id"],
            "architecture": config["experiment"]["architecture"],
            "valid_accuracy": valid["accuracy"],
            "valid_macro_f1": valid["macro_f1"],
            "valid_balanced_accuracy": valid["balanced_accuracy"],
            "valid_qwk": valid["qwk"],
            "valid_C_to_D": valid["C_to_D"],
            "valid_D_to_C": valid["D_to_C"],
            "valid_recall_A": valid["class_recall"]["A"],
            "valid_recall_B": valid["class_recall"]["B"],
            "valid_recall_C": valid["class_recall"]["C"],
            "valid_recall_D": valid["class_recall"]["D"],
        }
        if test is not None:
            row.update(
                {
                    "reused_test_accuracy": test["accuracy"],
                    "reused_test_macro_f1": test["macro_f1"],
                    "reused_test_balanced_accuracy": test["balanced_accuracy"],
                    "reused_test_qwk": test["qwk"],
                    "reused_test_C_to_D": test["C_to_D"],
                    "reused_test_D_to_C": test["D_to_C"],
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["valid_macro_f1", "valid_balanced_accuracy", "valid_qwk"],
        ascending=False,
    )
    output = root / "three_branch_summary.csv"
    frame.to_csv(output, index=False)

    pdf_path = root / "three_branch_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        figure, axis = plt.subplots(figsize=(11.7, 6.5))
        positions = np.arange(len(frame))
        axis.bar(positions, frame["valid_macro_f1"].to_numpy())
        axis.set_xticks(positions, labels=frame["branch"].tolist(), rotation=15, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("TN validation macro-F1")
        axis.set_title("Branch ranking is validation-only")
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=(11.7, 8.3))
        figure.text(0.03, 0.95, frame.to_string(index=False), va="top", family="monospace", fontsize=6.7)
        figure.text(
            0.03,
            0.04,
            "TN test is reused exploratory reporting only. It is not used for ranking, selection, or tuning.",
            fontsize=10,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--aggregate-root", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    output = run_dashboard(args.run_dir) if args.run_dir else aggregate(args.aggregate_root)
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))
