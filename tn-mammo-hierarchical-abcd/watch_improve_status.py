#!/usr/bin/env python3

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])

metric_files = list(run_root.rglob("epoch_metrics.jsonl"))

if not metric_files:
    print("Không tìm thấy epoch_metrics.jsonl")
    raise SystemExit(0)

metrics_path = max(metric_files, key=lambda p: p.stat().st_mtime)

rows = []

for raw_line in metrics_path.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines():
    raw_line = raw_line.strip()

    if not raw_line:
        continue

    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        continue

    valid = data.get("valid", {})
    train = data.get("train", {})
    branches = valid.get("branches", {})
    cd_oracle = branches.get("cd_oracle", {})

    rows.append({
        "epoch": data.get("epoch"),
        "lr": data.get("lr"),
        "train_loss": train.get("loss_total"),
        "minutes": (
            train.get("seconds", 0) / 60
            if train.get("seconds") is not None
            else None
        ),
        "acc": valid.get("accuracy"),
        "macro_f1": valid.get("macro_f1"),
        "bal_acc": valid.get("balanced_accuracy"),
        "qwk": valid.get("qwk"),
        "ece": valid.get("ece"),
        "c_to_d": valid.get("C_to_D"),
        "d_to_c": valid.get("D_to_C"),
        "cd_total": valid.get("CD_total"),
        "c_recall": valid.get("class_recall", {}).get("C"),
        "d_recall": valid.get("class_recall", {}).get("D"),
        "cd_auc": cd_oracle.get("roc_auc"),
        "improved": data.get("improved", False),
        "best_epoch": data.get("best_epoch"),
        "test_used": data.get("tn_test_used"),
        "confusion": valid.get("confusion_matrix"),
    })

print("=" * 116)
print("TASK 1 — IMPROVE HIERARCHY")
print(f"RUN     : {run_root}")
print(f"BRANCH  : {metrics_path.parent.name}")
print(f"METRICS : {metrics_path}")
print("=" * 116)

if not rows:
    print("File metrics chưa có epoch hoàn chỉnh.")
    raise SystemExit(0)

latest_best_epoch = rows[-1]["best_epoch"]

header = (
    f"{'':1} {'EP':>3} {'LR':>9} {'LOSS':>8} {'MIN':>6} "
    f"{'ACC':>7} {'MF1':>7} {'BACC':>7} {'QWK':>7} {'ECE':>7} "
    f"{'C-R':>7} {'D-R':>7} {'C>D':>4} {'D>C':>4} {'CD':>4} "
    f"{'CD-AUC':>7}"
)

print(header)
print("-" * len(header))

for row in rows[-15:]:
    marker = "*" if row["epoch"] == latest_best_epoch else " "

    def fmt(value, width=7, digits=4):
        if value is None:
            return f"{'NA':>{width}}"
        return f"{value:>{width}.{digits}f}"

    print(
        f"{marker} "
        f"{row['epoch']:>3} "
        f"{row['lr']:>9.2e} "
        f"{fmt(row['train_loss'], 8)} "
        f"{fmt(row['minutes'], 6, 1)} "
        f"{fmt(row['acc'])} "
        f"{fmt(row['macro_f1'])} "
        f"{fmt(row['bal_acc'])} "
        f"{fmt(row['qwk'])} "
        f"{fmt(row['ece'])} "
        f"{fmt(row['c_recall'])} "
        f"{fmt(row['d_recall'])} "
        f"{str(row['c_to_d']):>4} "
        f"{str(row['d_to_c']):>4} "
        f"{str(row['cd_total']):>4} "
        f"{fmt(row['cd_auc'])}"
    )

best_rows = [
    row for row in rows
    if row["epoch"] == latest_best_epoch
]

best = best_rows[-1] if best_rows else max(
    rows,
    key=lambda row: (
        row["macro_f1"]
        if row["macro_f1"] is not None
        else float("-inf")
    ),
)

latest = rows[-1]

print()
print("* = checkpoint tốt nhất hiện tại theo logic của training script")
print(
    f"BEST   : epoch={best['epoch']} | "
    f"macro-F1={best['macro_f1']:.4f} | "
    f"accuracy={best['acc']:.4f} | "
    f"balanced-acc={best['bal_acc']:.4f} | "
    f"QWK={best['qwk']:.4f} | "
    f"CD errors={best['cd_total']}"
)

print(
    f"LATEST : epoch={latest['epoch']} | "
    f"macro-F1={latest['macro_f1']:.4f} | "
    f"accuracy={latest['acc']:.4f} | "
    f"QWK={latest['qwk']:.4f} | "
    f"CD errors={latest['cd_total']} | "
    f"TN test used={latest['test_used']}"
)

print()
print("LATEST CONFUSION MATRIX — rows=true, columns=pred")
print("             Pred A  Pred B  Pred C  Pred D")
labels = ["True A", "True B", "True C", "True D"]

for label, matrix_row in zip(
    labels,
    latest["confusion"] or [],
):
    print(
        f"{label:>8} : "
        + " ".join(f"{value:>7}" for value in matrix_row)
    )
