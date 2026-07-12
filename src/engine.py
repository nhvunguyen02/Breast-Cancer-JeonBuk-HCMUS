"""Train/eval loops and test-time metric computation."""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from constants import CLASS_NAMES


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    losses = []
    all_true = []
    all_pred = []

    start = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        pred = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    elapsed = time.time() - start

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "elapsed_sec": elapsed,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def evaluate_test(model, loader, criterion, device):
    model.eval()

    losses = []
    all_true = []
    all_pred = []

    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            losses.append(loss.item())
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

    elapsed = time.time() - start

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "macro_precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(all_true, all_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(all_true, all_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "elapsed_sec": elapsed,
        "sec_per_exam": elapsed / max(1, len(all_true)),
    }

    report = classification_report(
        all_true,
        all_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    # Per-class accuracy = recall từng class
    per_class_rows = []

    for i, class_name in enumerate(CLASS_NAMES):
        support = int(cm[i, :].sum())
        correct = int(cm[i, i])
        acc = correct / support if support > 0 else 0.0

        metrics[f"{class_name}_correct"] = correct
        metrics[f"{class_name}_support"] = support
        metrics[f"{class_name}_acc"] = acc

        per_class_rows.append({
            "class": class_name,
            "correct": correct,
            "support": support,
            "accuracy": acc,
        })

    per_class_df = pd.DataFrame(per_class_rows)

    return metrics, report, cm_df, per_class_df
