#!/usr/bin/env python3
"""Case-level Task-1 metrics with explicit A/B/C/D label order."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)

LABELS = ["A", "B", "C", "D"]
LABEL_IDS = np.arange(4)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray, num_classes: int = 4) -> float:
    one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def _bootstrap_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    n = len(y_true)
    for _ in range(int(samples)):
        indices = rng.integers(0, n, size=n)
        try:
            value = float(metric(y_true[indices], probabilities[indices]))
            if np.isfinite(value):
                values.append(value)
        except Exception:
            continue
    if not values:
        return {"low": None, "high": None, "valid_replicates": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "low": float(np.quantile(array, 0.025)),
        "high": float(np.quantile(array, 0.975)),
        "valid_replicates": int(len(array)),
    }


def compute_case_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    ece_bins: int = 15,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(f"probabilities must be Nx4, got {probabilities.shape}")
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities length mismatch")
    row_sums = probabilities.sum(axis=1)
    if not np.all(np.isfinite(probabilities)) or not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Probabilities are non-finite or not normalized")

    y_pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    class_recall = recall_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=LABEL_IDS, weights="quadratic")),
        "within_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "severe_error_rate": float(np.mean(np.abs(y_true - y_pred) >= 2)),
        "nll": float(log_loss(y_true, probabilities, labels=LABEL_IDS)),
        "brier": multiclass_brier(y_true, probabilities, num_classes=4),
        "ece": expected_calibration_error(y_true, probabilities, bins=ece_bins),
        "confusion_matrix": cm.astype(int).tolist(),
        "class_recall": {LABELS[i]: float(class_recall[i]) for i in range(4)},
        "class_f1": {LABELS[i]: float(class_f1[i]) for i in range(4)},
        "C_to_D": int(cm[2, 3]),
        "D_to_C": int(cm[3, 2]),
        "CD_total": int(cm[2, 3] + cm[3, 2]),
        "A_to_D": int(cm[0, 3]),
        "D_to_A": int(cm[3, 0]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_IDS,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    metric_functions = {
        "accuracy": lambda truth, probs: accuracy_score(truth, probs.argmax(axis=1)),
        "macro_f1": lambda truth, probs: f1_score(
            truth, probs.argmax(axis=1), labels=LABEL_IDS, average="macro", zero_division=0
        ),
        "balanced_accuracy": lambda truth, probs: balanced_accuracy_score(truth, probs.argmax(axis=1)),
        "qwk": lambda truth, probs: cohen_kappa_score(
            truth, probs.argmax(axis=1), labels=LABEL_IDS, weights="quadratic"
        ),
    }
    metrics["bootstrap_95ci"] = {
        name: _bootstrap_interval(y_true, probabilities, fn, bootstrap_samples, bootstrap_seed + offset)
        for offset, (name, fn) in enumerate(metric_functions.items())
    }
    class_recall_ci: dict[str, dict[str, float | int | None]] = {}
    for class_id, label in enumerate(LABELS):
        def recall_metric(truth: np.ndarray, probs: np.ndarray, class_id: int = class_id) -> float:
            present = truth == class_id
            if not present.any():
                raise ValueError("Bootstrap replicate omitted the class")
            pred = probs.argmax(axis=1)
            return float(np.mean(pred[present] == class_id))
        class_recall_ci[label] = _bootstrap_interval(
            y_true,
            probabilities,
            recall_metric,
            bootstrap_samples,
            bootstrap_seed + 100 + class_id,
        )
    metrics["class_recall_bootstrap_95ci"] = class_recall_ci
    return metrics


def compute_branch_metrics(
    y_true: np.ndarray,
    coarse_probabilities: np.ndarray | None = None,
    ab_probabilities: np.ndarray | None = None,
    cd_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    result: dict[str, Any] = {}
    if coarse_probabilities is not None:
        coarse = np.asarray(coarse_probabilities, dtype=np.float64)
        target = (y_true >= 2).astype(np.int64)
        pred = coarse.argmax(axis=1)
        cm = confusion_matrix(target, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        result["coarse"] = {
            "accuracy": float(accuracy_score(target, pred)),
            "macro_f1": float(f1_score(target, pred, labels=[0, 1], average="macro", zero_division=0)),
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "roc_auc": safe_roc_auc(target, coarse[:, 1]),
            "confusion_matrix": cm.astype(int).tolist(),
        }
    if ab_probabilities is not None:
        mask = y_true < 2
        if mask.any():
            probs = np.asarray(ab_probabilities, dtype=np.float64)[mask]
            target = y_true[mask]
            pred = probs.argmax(axis=1)
            result["ab_oracle"] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(target, pred)),
                "macro_f1": float(f1_score(target, pred, labels=[0, 1], average="macro", zero_division=0)),
                "roc_auc": safe_roc_auc(target, probs[:, 1]),
            }
    if cd_probabilities is not None:
        mask = y_true >= 2
        if mask.any():
            probs = np.asarray(cd_probabilities, dtype=np.float64)[mask]
            target = y_true[mask] - 2
            pred = probs.argmax(axis=1)
            cm = confusion_matrix(target, pred, labels=[0, 1])
            result["cd_oracle"] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(target, pred)),
                "macro_f1": float(f1_score(target, pred, labels=[0, 1], average="macro", zero_division=0)),
                "roc_auc": safe_roc_auc(target, probs[:, 1]),
                "C_to_D": int(cm[0, 1]),
                "D_to_C": int(cm[1, 0]),
                "confusion_matrix": cm.astype(int).tolist(),
            }
    return result


def classwise_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(LABELS):
        support = int(cm[index].sum())
        correct = int(cm[index, index])
        rows.append(
            {
                "label": label,
                "support": support,
                "correct": correct,
                "recall": float(metrics["class_recall"][label]),
                "f1": float(metrics["class_f1"][label]),
            }
        )
    return rows
