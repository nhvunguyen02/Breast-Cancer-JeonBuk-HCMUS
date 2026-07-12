import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from phaseG_rerun.config import config
from phaseG_rerun.dataset import build_datasets
from phaseG_rerun.metrics import (
    CLASS_NAMES,
    compute_classification_metrics,
    format_metrics,
)
from phaseG_rerun.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.output_dir / "best_model.pt",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Dict[str, object], pd.DataFrame]:
    model.eval()

    rows: List[Dict[str, object]] = []
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for batch in loader:
        images = batch["images"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=config.use_amp,
        ):
            logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1,
        )

        labels_cpu = labels.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        probabilities_cpu = (
            probabilities
            .float()
            .cpu()
            .tolist()
        )

        all_targets.extend(labels_cpu)
        all_predictions.extend(predictions_cpu)

        for index in range(len(labels_cpu)):
            true_index = labels_cpu[index]
            predicted_index = predictions_cpu[index]
            probability_row = probabilities_cpu[index]

            rows.append(
                {
                    "case_id": str(
                        batch["case_id"][index]
                    ),
                    "domain": str(
                        batch["domain"][index]
                    ),
                    "true_label_idx": true_index,
                    "true_label": CLASS_NAMES[
                        true_index
                    ],
                    "predicted_label_idx": (
                        predicted_index
                    ),
                    "predicted_label": CLASS_NAMES[
                        predicted_index
                    ],
                    "correct": (
                        true_index
                        == predicted_index
                    ),
                    "prob_A": probability_row[0],
                    "prob_B": probability_row[1],
                    "prob_C": probability_row[2],
                    "prob_D": probability_row[3],
                    "confidence": max(
                        probability_row
                    ),
                }
            )

    metrics = compute_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
    )

    predictions_df = pd.DataFrame(rows)

    return metrics, predictions_df


def main() -> None:
    args = parse_args()

    config.validate()
    config.create_directories()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    _, _, test_dataset = build_datasets(
        config
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=False,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        num_classes=config.num_classes,
        pretrained=False,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    metrics, predictions_df = evaluate_test(
        model=model,
        loader=test_loader,
        device=device,
    )

    output_dir = (
        config.output_dir
        / "test_evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_dir
        / "test_predictions.csv"
    )

    metrics_path = (
        output_dir
        / "test_metrics.json"
    )

    confusion_matrix_path = (
        output_dir
        / "test_confusion_matrix.csv"
    )

    cd_errors_path = (
        output_dir
        / "test_cd_errors.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    confusion_matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            f"true_{label}"
            for label in CLASS_NAMES
        ],
        columns=[
            f"pred_{label}"
            for label in CLASS_NAMES
        ],
    )

    confusion_matrix_df.to_csv(
        confusion_matrix_path,
        index=True,
    )

    cd_errors_df = predictions_df[
        (
            (
                predictions_df["true_label"]
                == "C"
            )
            & (
                predictions_df[
                    "predicted_label"
                ]
                == "D"
            )
        )
        |
        (
            (
                predictions_df["true_label"]
                == "D"
            )
            & (
                predictions_df[
                    "predicted_label"
                ]
                == "C"
            )
        )
    ].copy()

    cd_errors_df = (
        cd_errors_df
        .sort_values(
            [
                "true_label",
                "confidence",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    cd_errors_df.to_csv(
        cd_errors_path,
        index=False,
    )

    result = {
        "checkpoint": str(
            args.checkpoint
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_best_macro_f1": float(
            checkpoint["best_macro_f1"]
        ),
        "num_test_cases": int(
            len(predictions_df)
        ),
        "metrics": metrics,
        "outputs": {
            "predictions": str(
                predictions_path
            ),
            "metrics": str(
                metrics_path
            ),
            "confusion_matrix": str(
                confusion_matrix_path
            ),
            "cd_errors": str(
                cd_errors_path
            ),
        },
    }

    metrics_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "checkpoint best valid macro_f1:",
        checkpoint["best_macro_f1"],
    )

    print(
        "test cases:",
        len(predictions_df),
    )

    print()
    print(format_metrics(metrics))

    print()
    print("confusion matrix:")
    print(
        confusion_matrix_df.to_string()
    )

    print()
    print(
        "C/D error cases:",
        len(cd_errors_df),
    )

    print()
    print("created:")
    print(predictions_path)
    print(metrics_path)
    print(confusion_matrix_path)
    print(cd_errors_path)

    assert len(predictions_df) == 132
    assert metrics["num_samples"] == 132
    assert (
        len(cd_errors_df)
        == metrics["cd_total"]
    )

    print()
    print("[PASS] TN test evaluation")


if __name__ == "__main__":
    main()
    