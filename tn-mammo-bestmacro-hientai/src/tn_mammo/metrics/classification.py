from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from tn_mammo.constants import (
    INDEX_TO_LABEL,
    NUM_CLASSES,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, object]:
    truth = np.asarray(
        y_true,
        dtype=np.int64,
    )
    prediction = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    if truth.shape != prediction.shape:
        raise ValueError(
            "y_true and y_pred shape mismatch."
        )

    labels = list(range(NUM_CLASSES))

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            truth,
            prediction,
            labels=labels,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        truth,
        prediction,
        labels=labels,
    )

    absolute_error = np.abs(
        truth - prediction
    )

    per_class = {}

    for index in labels:
        per_class[
            INDEX_TO_LABEL[index]
        ] = {
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                class_f1[index]
            ),
            "support": int(
                support[index]
            ),
        }

    return {
        "num_samples": int(len(truth)),
        "accuracy": float(
            accuracy_score(
                truth,
                prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                truth,
                prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "qwk": float(
            cohen_kappa_score(
                truth,
                prediction,
                weights="quadratic",
                labels=labels,
            )
        ),
        "within_one": float(
            np.mean(absolute_error <= 1)
        ),
        "severe_error_rate": float(
            np.mean(absolute_error >= 2)
        ),
        "severe_error_count": int(
            np.sum(absolute_error >= 2)
        ),
        "c_to_d": int(
            np.sum(
                (truth == 2)
                & (prediction == 3)
            )
        ),
        "d_to_c": int(
            np.sum(
                (truth == 3)
                & (prediction == 2)
            )
        ),
        "b_to_c": int(
            np.sum(
                (truth == 1)
                & (prediction == 2)
            )
        ),
        "c_to_b": int(
            np.sum(
                (truth == 2)
                & (prediction == 1)
            )
        ),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
