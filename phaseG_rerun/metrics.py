from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = ["A", "B", "C", "D"]


def compute_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: List[str] | None = None,
) -> Dict[str, object]:
    if class_names is None:
        class_names = CLASS_NAMES

    targets_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    predictions_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    if targets_array.ndim != 1:
        raise ValueError(
            "targets must be one-dimensional."
        )

    if predictions_array.ndim != 1:
        raise ValueError(
            "predictions must be one-dimensional."
        )

    if len(targets_array) != len(predictions_array):
        raise ValueError(
            "targets and predictions must have equal length."
        )

    if len(targets_array) == 0:
        raise ValueError(
            "targets and predictions cannot be empty."
        )

    labels = list(range(len(class_names)))

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
        labels=labels,
    )

    accuracy = accuracy_score(
        targets_array,
        predictions_array,
    )

    balanced_accuracy = balanced_accuracy_score(
        targets_array,
        predictions_array,
    )

    macro_f1 = f1_score(
        targets_array,
        predictions_array,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        targets_array,
        predictions_array,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets_array,
        predictions_array,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class_accuracy = {}

    for class_index, class_name in enumerate(class_names):
        class_total = int(matrix[class_index].sum())

        if class_total == 0:
            class_accuracy = 0.0
        else:
            class_accuracy = (
                float(matrix[class_index, class_index])
                / class_total
            )

        per_class_accuracy[class_name] = class_accuracy

    c_index = class_names.index("C")
    d_index = class_names.index("D")

    c_to_d = int(matrix[c_index, d_index])
    d_to_c = int(matrix[d_index, c_index])

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": {
            class_name: float(per_class_f1[index])
            for index, class_name in enumerate(
                class_names
            )
        },
        "per_class_accuracy": per_class_accuracy,
        "c_to_d": c_to_d,
        "d_to_c": d_to_c,
        "cd_total": c_to_d + d_to_c,
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(len(targets_array)),
    }

    return metrics


def format_metrics(
    metrics: Dict[str, object],
) -> str:
    per_class_f1 = metrics["per_class_f1"]
    per_class_accuracy = metrics[
        "per_class_accuracy"
    ]

    lines = [
        f"accuracy: {metrics['accuracy']:.6f}",
        (
            "balanced_accuracy: "
            f"{metrics['balanced_accuracy']:.6f}"
        ),
        f"macro_f1: {metrics['macro_f1']:.6f}",
        f"weighted_f1: {metrics['weighted_f1']:.6f}",
        (
            "per_class_f1: "
            + ", ".join(
                f"{label}={per_class_f1[label]:.6f}"
                for label in CLASS_NAMES
            )
        ),
        (
            "per_class_accuracy: "
            + ", ".join(
                f"{label}={per_class_accuracy[label]:.6f}"
                for label in CLASS_NAMES
            )
        ),
        f"C_to_D: {metrics['c_to_d']}",
        f"D_to_C: {metrics['d_to_c']}",
        f"CD_total: {metrics['cd_total']}",
    ]

    return "\n".join(lines)