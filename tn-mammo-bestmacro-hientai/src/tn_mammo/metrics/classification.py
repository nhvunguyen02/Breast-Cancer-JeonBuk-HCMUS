# -*- coding: utf-8 -*-
"""Metrics: Macro-F1 (chính) + các chỉ số thứ tự (QWK, within-one, severe)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from tn_mammo.constants import INDEX_TO_LABEL, NUM_CLASSES


def compute_metrics(y_true, y_pred) -> dict:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(NUM_CLASSES))

    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    distance = np.abs(truth - prediction)

    return {
        "num_samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(truth, prediction)
        ),
        "macro_f1": float(
            f1_score(truth, prediction, average="macro", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(
                truth, prediction, weights="quadratic", labels=labels
            )
        ),
        "within_one": float(np.mean(distance <= 1)),
        "severe_error_count": int(np.sum(distance >= 2)),
        "confusion_matrix": confusion_matrix(
            truth, prediction, labels=labels
        ).tolist(),
        "per_class": {
            INDEX_TO_LABEL[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(class_f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
    }
