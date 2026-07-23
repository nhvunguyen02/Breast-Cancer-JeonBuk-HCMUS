from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.nn import functional as F

from tn_mammo.data.contracts import (
    decode_coral_logits,
)
from tn_mammo.training.engine import (
    build_dataloaders,
    build_model,
)


ROOT = Path(
    "/mnt/hcmus/breast_vn/code/new_implement"
)

OUTPUTS = ROOT / "outputs"
MANIFESTS = ROOT / "manifests"

SELECTED_OUTPUT_POINTER = (
    OUTPUTS / "CURRENT_SELECTED_OUTPUT.txt"
)

SELECTED_CHECKPOINT_POINTER = (
    OUTPUTS / "CURRENT_SELECTED_CHECKPOINT.txt"
)

TEST_MANIFEST = (
    MANIFESTS / "tn_locked_test132.csv"
)

TEST_ADAPTER_MANIFEST = (
    MANIFESTS / "tn_final_eval132_phaseg.csv"
)

GLOBAL_MARKER = (
    OUTPUTS
    / "TN_LOCKED_TEST132_FINAL_EVALUATED.json"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "7b80c4cd36f4377f87f0dbfbc337ba0d0f58fa8ac60cca9043790ddd5b43b22b"
)

LABELS = ["A", "B", "C", "D"]
LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}
INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    raise TypeError(
        f"Unsupported JSON type: {type(value).__name__}"
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=json_ready,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def normalize_label(value: Any) -> str:
    text = str(value).strip().upper()

    numeric_mapping = {
        "0": "A",
        "0.0": "A",
        "1": "A",
        "1.0": "A",
        "2": "B",
        "2.0": "B",
        "3": "C",
        "3.0": "C",
        "4": "D",
        "4.0": "D",
    }

    if text in LABEL_TO_INDEX:
        return text

    if text in numeric_mapping:
        return numeric_mapping[text]

    for label in LABELS:
        if text.endswith(label):
            return label

    raise ValueError(
        f"Unsupported density label: {value!r}"
    )


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str:
    lookup = {
        str(column).strip().lower(): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        "Missing required column. "
        f"Candidates={candidates}; "
        f"columns={list(frame.columns)}"
    )


def load_manifest_identity(
    path: Path,
) -> tuple[pd.DataFrame, str, str]:
    frame = pd.read_csv(
        path,
        dtype=str,
    )

    id_column = find_column(
        frame,
        [
            "case_id",
            "study_id",
            "exam_id",
            "patient_id",
            "id",
        ],
    )

    label_column = find_column(
        frame,
        [
            "label",
            "density",
            "breast_density",
            "birads_density",
            "label_idx",
        ],
    )

    return (
        frame,
        id_column,
        label_column,
    )


def manifest_case_ids(
    path: Path,
) -> set[str]:
    frame, id_column, _ = (
        load_manifest_identity(path)
    )

    return set(
        frame[id_column]
        .astype(str)
        .str.strip()
        .tolist()
    )


def extract_batch_ids(
    batch: dict[str, Any],
) -> list[str] | None:
    for key in (
        "case_id",
        "study_id",
        "exam_id",
        "patient_id",
        "id",
    ):
        if key not in batch:
            continue

        values = batch[key]

        if isinstance(values, torch.Tensor):
            return [
                str(value)
                for value in values
                .detach()
                .cpu()
                .tolist()
            ]

        if isinstance(
            values,
            (list, tuple),
        ):
            return [
                str(value)
                for value in values
            ]

        return [str(values)]

    return None


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    precision, recall, f1_values, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            zero_division=0,
        )
    )

    distance = np.abs(
        y_true - y_pred
    )

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        weights="quadratic",
    )

    per_class = {}

    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(
                precision[index]
            ),
            "recall": float(
                recall[index]
            ),
            "f1": float(
                f1_values[index]
            ),
            "support": int(
                support[index]
            ),
        }

    return {
        "num_samples": int(
            len(y_true)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=[0, 1, 2, 3],
                average="weighted",
                zero_division=0,
            )
        ),
        "qwk": float(qwk),
        "within_one": float(
            (distance <= 1).mean()
        ),
        "severe_error_rate": float(
            (distance >= 2).mean()
        ),
        "severe_error_count": int(
            (distance >= 2).sum()
        ),
        "mean_absolute_class_distance": float(
            distance.mean()
        ),
        "c_to_d": int(cm[2, 3]),
        "d_to_c": int(cm[3, 2]),
        "b_to_c": int(cm[1, 2]),
        "c_to_b": int(cm[2, 1]),
        "cd_total": int(
            cm[2, 3] + cm[3, 2]
        ),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


@torch.no_grad()
def run_inference(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    manifest_case_order: list[str],
) -> dict[str, Any]:
    model.eval()

    all_true: list[int] = []
    all_flat_pred: list[int] = []
    all_ordinal_pred: list[int] = []

    all_flat_probs: list[list[float]] = []
    all_ordinal_threshold_probs: list[
        list[float]
    ] = []

    all_case_ids: list[str] = []

    fallback_cursor = 0

    for batch in loader:
        views = batch["views"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        ).long()

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(views)

        flat_logits = outputs[
            "flat_logits"
        ]

        ordinal_logits = outputs[
            "ordinal_logits"
        ]

        if ordinal_logits is None:
            raise RuntimeError(
                "Selected E1 checkpoint does not "
                "provide ordinal logits."
            )

        flat_probs = F.softmax(
            flat_logits,
            dim=1,
        )

        flat_pred = flat_probs.argmax(
            dim=1
        )

        ordinal_threshold_probs = (
            torch.sigmoid(
                ordinal_logits
            )
        )

        ordinal_pred = decode_coral_logits(
            ordinal_logits,
            threshold=0.5,
        )

        batch_size = int(
            labels.shape[0]
        )

        batch_ids = extract_batch_ids(
            batch
        )

        if batch_ids is None:
            batch_ids = manifest_case_order[
                fallback_cursor:
                fallback_cursor + batch_size
            ]

        fallback_cursor += batch_size

        if len(batch_ids) != batch_size:
            raise RuntimeError(
                "Could not align batch case IDs."
            )

        all_case_ids.extend(
            batch_ids
        )

        all_true.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        all_flat_pred.extend(
            flat_pred.detach()
            .cpu()
            .tolist()
        )

        all_ordinal_pred.extend(
            ordinal_pred.detach()
            .cpu()
            .tolist()
        )

        all_flat_probs.extend(
            flat_probs.detach()
            .float()
            .cpu()
            .tolist()
        )

        all_ordinal_threshold_probs.extend(
            ordinal_threshold_probs.detach()
            .float()
            .cpu()
            .tolist()
        )

    return {
        "case_ids": all_case_ids,
        "true": np.asarray(
            all_true,
            dtype=np.int64,
        ),
        "flat_pred": np.asarray(
            all_flat_pred,
            dtype=np.int64,
        ),
        "ordinal_pred": np.asarray(
            all_ordinal_pred,
            dtype=np.int64,
        ),
        "flat_probs": np.asarray(
            all_flat_probs,
            dtype=np.float64,
        ),
        "ordinal_threshold_probs": np.asarray(
            all_ordinal_threshold_probs,
            dtype=np.float64,
        ),
    }


def metric_distance(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> float:
    keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "qwk",
        "within_one",
    ]

    return float(
        sum(
            abs(
                float(observed[key])
                - float(expected[key])
            )
            for key in keys
        )
    )


def main() -> None:
    if GLOBAL_MARKER.exists():
        existing = json.loads(
            GLOBAL_MARKER.read_text(
                encoding="utf-8"
            )
        )

        print(
            "[REFUSED] TN locked test has "
            "already been evaluated."
        )

        print(
            json.dumps(
                existing,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise RuntimeError(
            "One-time locked-test marker already exists."
        )

    if not SELECTED_OUTPUT_POINTER.is_file():
        raise FileNotFoundError(
            SELECTED_OUTPUT_POINTER
        )

    if not SELECTED_CHECKPOINT_POINTER.is_file():
        raise FileNotFoundError(
            SELECTED_CHECKPOINT_POINTER
        )

    selected_output = Path(
        SELECTED_OUTPUT_POINTER.read_text(
            encoding="utf-8"
        ).strip()
    )

    selected_checkpoint = Path(
        SELECTED_CHECKPOINT_POINTER.read_text(
            encoding="utf-8"
        ).strip()
    )

    if not selected_output.is_dir():
        raise FileNotFoundError(
            selected_output
        )

    if not selected_checkpoint.is_file():
        raise FileNotFoundError(
            selected_checkpoint
        )

    checkpoint_sha256 = sha256_file(
        selected_checkpoint
    )

    if (
        checkpoint_sha256
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Selected checkpoint SHA256 changed. "
            f"Expected={EXPECTED_CHECKPOINT_SHA256}; "
            f"observed={checkpoint_sha256}"
        )

    if (
        selected_output.name
        != "E1_sequential_seed42_20260719_145848"
    ):
        raise RuntimeError(
            "Final selected output is not the "
            "approved E1 champion: "
            f"{selected_output}"
        )

    config_path = (
        selected_output
        / "run_config.json"
    )

    valid_metrics_path = (
        selected_output
        / "best_valid_metrics.json"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    if not valid_metrics_path.is_file():
        raise FileNotFoundError(
            valid_metrics_path
        )

    raw_config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    def locate_training_config(
        node: Any,
    ) -> dict[str, Any] | None:
        if isinstance(node, dict):
            required_keys = {
                "data",
                "model",
                "training",
            }

            if required_keys.issubset(
                node.keys()
            ):
                return node

            preferred_wrappers = (
                "config",
                "resolved_config",
                "run_config",
                "configuration",
                "settings",
            )

            for key in preferred_wrappers:
                if key not in node:
                    continue

                found = locate_training_config(
                    node[key]
                )

                if found is not None:
                    return found

            for value in node.values():
                found = locate_training_config(
                    value
                )

                if found is not None:
                    return found

        elif isinstance(node, list):
            for value in node:
                found = locate_training_config(
                    value
                )

                if found is not None:
                    return found

        return None

    config = locate_training_config(
        raw_config
    )

    config_source = str(
        config_path
    )

    if config is None:
        import yaml

        path_candidates: list[Path] = []

        def collect_config_paths(
            node: Any,
        ) -> None:
            if isinstance(node, dict):
                for value in node.values():
                    collect_config_paths(
                        value
                    )

            elif isinstance(node, list):
                for value in node:
                    collect_config_paths(
                        value
                    )

            elif isinstance(node, str):
                candidate = Path(node)

                if candidate.suffix.lower() in {
                    ".yaml",
                    ".yml",
                    ".json",
                }:
                    path_candidates.append(
                        candidate
                    )

        collect_config_paths(
            raw_config
        )

        runtime_directory = (
            ROOT
            / "configs"
            / "runtime_e1_e2"
        )

        known_runtime = (
            runtime_directory
            / "e1_runtime_20260719_145848.yaml"
        )

        if known_runtime.is_file():
            path_candidates.append(
                known_runtime
            )

        if runtime_directory.is_dir():
            path_candidates.extend(
                sorted(
                    runtime_directory.glob(
                        "e1_runtime_*.yaml"
                    ),
                    key=lambda item: (
                        item.stat().st_mtime
                    ),
                    reverse=True,
                )
            )

        examined: set[Path] = set()

        for candidate in path_candidates:
            candidate = candidate.expanduser()

            if not candidate.is_absolute():
                candidate = (
                    ROOT / candidate
                ).resolve()

            if candidate in examined:
                continue

            examined.add(candidate)

            if not candidate.is_file():
                continue

            if candidate.suffix.lower() == ".json":
                candidate_data = json.loads(
                    candidate.read_text(
                        encoding="utf-8"
                    )
                )
            else:
                candidate_data = yaml.safe_load(
                    candidate.read_text(
                        encoding="utf-8"
                    )
                )

            found = locate_training_config(
                candidate_data
            )

            if found is not None:
                config = found
                config_source = str(
                    candidate
                )
                break

    if config is None:
        root_keys = (
            sorted(raw_config.keys())
            if isinstance(
                raw_config,
                dict,
            )
            else []
        )

        raise RuntimeError(
            "Could not resolve a training config "
            "containing data/model/training. "
            f"run_config_root_keys={root_keys}"
        )

    config = json.loads(
        json.dumps(config)
    )

    print(
        "RESOLVED_CONFIG_SOURCE="
        f"{config_source}"
    )

    print(
        "RESOLVED_CONFIG_KEYS="
        f"{sorted(config.keys())}"
    )

    print(
        "RESOLVED_MODEL_KEYS="
        f"{sorted(config['model'].keys())}"
    )

    print(
        "RESOLVED_DATA_KEYS="
        f"{sorted(config['data'].keys())}"
    )

    print(
        "RESOLVED_TRAINING_KEYS="
        f"{sorted(config['training'].keys())}"
    )

    expected_valid_metrics = json.loads(
        valid_metrics_path.read_text(
            encoding="utf-8"
        )
    )

    config["model"][
        "initialization_checkpoint"
    ] = str(selected_checkpoint)

    config["model"]["imagenet_init"] = False
    config["model"]["fusion"] = "mean"
    config["model"]["use_ordinal_head"] = True
    config["model"]["use_binary_head"] = False

    config["training"]["batch_size"] = 2
    config["training"]["num_workers"] = 4
    config["training"]["amp"] = True

    valid_manifest = Path(
        config["data"][
            "validation"
        ]["manifest"]
    )

    if not valid_manifest.is_file():
        raise FileNotFoundError(
            valid_manifest
        )

    valid_frame, valid_id_column, _ = (
        load_manifest_identity(
            valid_manifest
        )
    )

    valid_case_order = (
        valid_frame[valid_id_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print("=" * 76)
    print("STEP 1 — VALIDATION REPRODUCTION BEFORE TEST UNLOCK")
    print("=" * 76)
    print(f"SELECTED_OUTPUT={selected_output}")
    print(f"SELECTED_CHECKPOINT={selected_checkpoint}")
    print(f"CHECKPOINT_SHA256={checkpoint_sha256}")
    print(f"VALID_MANIFEST={valid_manifest}")
    print("TEST_MANIFEST_READ=False")

    _, valid_loader, valid_audit = (
        build_dataloaders(config)
    )

    model, initialization_report = (
        build_model(config)
    )

    checkpoint = torch.load(
        selected_checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get(
            "state_dict",
            checkpoint,
        ),
    )

    strict_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        strict_result.missing_keys
        or strict_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint load failed."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    valid_inference = run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        amp_enabled=bool(
            config["training"].get(
                "amp",
                True,
            )
        ),
        manifest_case_order=valid_case_order,
    )

    flat_valid_metrics = compute_metrics(
        valid_inference["true"],
        valid_inference["flat_pred"],
    )

    ordinal_valid_metrics = compute_metrics(
        valid_inference["true"],
        valid_inference["ordinal_pred"],
    )

    flat_distance = metric_distance(
        flat_valid_metrics,
        expected_valid_metrics,
    )

    ordinal_distance = metric_distance(
        ordinal_valid_metrics,
        expected_valid_metrics,
    )

    print(
        "FLAT_VALID_MACRO_F1="
        f"{flat_valid_metrics['macro_f1']:.10f}"
    )

    print(
        "ORDINAL_VALID_MACRO_F1="
        f"{ordinal_valid_metrics['macro_f1']:.10f}"
    )

    print(
        "EXPECTED_VALID_MACRO_F1="
        f"{expected_valid_metrics['macro_f1']:.10f}"
    )

    print(
        "FLAT_METRIC_DISTANCE="
        f"{flat_distance:.12e}"
    )

    print(
        "ORDINAL_METRIC_DISTANCE="
        f"{ordinal_distance:.12e}"
    )

    decoder_candidates = {
        "flat_argmax": flat_distance,
        "coral_threshold_0p5": ordinal_distance,
    }

    selected_decoder = min(
        decoder_candidates,
        key=decoder_candidates.get,
    )

    selected_distance = (
        decoder_candidates[
            selected_decoder
        ]
    )

    if selected_distance > 1e-7:
        raise RuntimeError(
            "Neither decoder reproduces the "
            "approved validation metrics. "
            "Locked test remains unopened."
        )

    print(
        "VALIDATION_REPRODUCTION_PASS=True"
    )

    print(
        f"LOCKED_DECODER={selected_decoder}"
    )

    print("TEST_MANIFEST_READ=False")

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_dir = (
        OUTPUTS
        / (
            "FINAL_TN_LOCKED_TEST132_"
            f"E1_{timestamp}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Only after validation reproduction succeeds
    # is the test manifest accessed.
    if not TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            TEST_MANIFEST
        )

    test_frame, test_id_column, test_label_column = (
        load_manifest_identity(
            TEST_MANIFEST
        )
    )

    test_case_order = (
        test_frame[test_id_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    normalized_test_labels = (
        test_frame[test_label_column]
        .map(normalize_label)
    )

    test_counts = (
        normalized_test_labels
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    expected_test_counts = {
        "A": 4,
        "B": 26,
        "C": 57,
        "D": 45,
    }

    if len(test_frame) != 132:
        raise RuntimeError(
            "Locked test row count mismatch: "
            f"{len(test_frame)}"
        )

    if test_counts != expected_test_counts:
        raise RuntimeError(
            "Locked test class count mismatch: "
            f"{test_counts}"
        )

    train_manifest = Path(
        config["data"]["train"][
            "tn_manifest"
        ]
    )

    train_cases = manifest_case_ids(
        train_manifest
    )

    valid_cases = manifest_case_ids(
        valid_manifest
    )

    test_cases = set(
        test_case_order
    )

    train_test_overlap = (
        train_cases & test_cases
    )

    valid_test_overlap = (
        valid_cases & test_cases
    )

    if train_test_overlap:
        raise RuntimeError(
            "TN train/test case overlap: "
            f"{len(train_test_overlap)}"
        )

    if valid_test_overlap:
        raise RuntimeError(
            "TN valid/test case overlap: "
            f"{len(valid_test_overlap)}"
        )

    test_manifest_sha256 = sha256_file(
        TEST_MANIFEST
    )

    locked_config = {
        "status": "LOCKED_BEFORE_TEST_INFERENCE",
        "locked_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "selected_experiment": "E1",
        "selected_output": str(
            selected_output
        ),
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": (
            checkpoint_sha256
        ),
        "checkpoint_best_epoch": int(
            checkpoint.get(
                "epoch",
                2,
            )
        ),
        "validation_manifest": str(
            valid_manifest
        ),
        "validation_manifest_sha256": (
            sha256_file(
                valid_manifest
            )
        ),
        "validation_macro_f1": float(
            expected_valid_metrics[
                "macro_f1"
            ]
        ),
        "validation_reproduction_distance": float(
            selected_distance
        ),
        "decoder": selected_decoder,
        "coral_threshold": (
            0.5
            if selected_decoder
            == "coral_threshold_0p5"
            else None
        ),
        "test_manifest": str(
            TEST_MANIFEST
        ),
        "test_manifest_sha256": (
            test_manifest_sha256
        ),
        "expected_test_rows": 132,
        "expected_test_class_counts": (
            expected_test_counts
        ),
        "train_test_case_overlap": 0,
        "valid_test_case_overlap": 0,
        "view_order": [
            "L_CC",
            "L_MLO",
            "R_CC",
            "R_MLO",
        ],
        "image_size": int(
            config["data"][
                "image_size"
            ]
        ),
        "batch_size": 2,
        "amp": True,
        "physical_gpu": int(
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                "1",
            ).split(",")[0]
        ),
        "model_changes_after_lock_allowed": False,
        "test_time_augmentation": False,
        "calibration_on_test": False,
        "threshold_tuning_on_test": False,
    }

    write_json(
        output_dir
        / "LOCKED_CONFIG.json",
        locked_config,
    )

    print()
    print("=" * 76)
    print("STEP 2 — TN LOCKED TEST132 OPENED ONCE")
    print("=" * 76)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"TEST_MANIFEST={TEST_MANIFEST}")
    print(f"TEST_MANIFEST_SHA256={test_manifest_sha256}")
    print(f"TEST_COUNTS={test_counts}")
    print("TN_LOCKED_TEST_OPENED=True")
    print("VINDR_LOCKED_TEST_OPENED=False")

    test_config = json.loads(
        json.dumps(config)
    )

    test_config["data"][
        "validation"
    ]["manifest"] = str(
        TEST_ADAPTER_MANIFEST
    )

    print(
        "TEST_ADAPTER_MANIFEST="
        f"{TEST_ADAPTER_MANIFEST}"
    )

    _, test_loader, test_audit = (
        build_dataloaders(
            test_config
        )
    )

    test_inference = run_inference(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=True,
        manifest_case_order=test_case_order,
    )

    if len(test_inference["true"]) != 132:
        raise RuntimeError(
            "Inference did not return 132 cases."
        )

    if selected_decoder == "flat_argmax":
        selected_predictions = (
            test_inference[
                "flat_pred"
            ]
        )
    else:
        selected_predictions = (
            test_inference[
                "ordinal_pred"
            ]
        )

    test_metrics = compute_metrics(
        test_inference["true"],
        selected_predictions,
    )

    predictions = pd.DataFrame({
        "case_id": (
            test_inference[
                "case_ids"
            ]
        ),
        "true_index": (
            test_inference[
                "true"
            ]
        ),
        "true_label": [
            INDEX_TO_LABEL[int(value)]
            for value in (
                test_inference[
                    "true"
                ]
            )
        ],
        "pred_index": (
            selected_predictions
        ),
        "pred_label": [
            INDEX_TO_LABEL[int(value)]
            for value in (
                selected_predictions
            )
        ],
        "correct": (
            test_inference["true"]
            == selected_predictions
        ).astype(int),
        "absolute_class_distance": (
            np.abs(
                test_inference["true"]
                - selected_predictions
            )
        ),
        "selected_decoder": (
            selected_decoder
        ),
    })

    flat_probs = test_inference[
        "flat_probs"
    ]

    for index, label in enumerate(
        LABELS
    ):
        predictions[
            f"flat_prob_{label}"
        ] = flat_probs[:, index]

    ordinal_probs = test_inference[
        "ordinal_threshold_probs"
    ]

    predictions[
        "ordinal_prob_gt_A"
    ] = ordinal_probs[:, 0]

    predictions[
        "ordinal_prob_gt_B"
    ] = ordinal_probs[:, 1]

    predictions[
        "ordinal_prob_gt_C"
    ] = ordinal_probs[:, 2]

    predictions.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
    )

    confusion_frame = pd.DataFrame(
        test_metrics[
            "confusion_matrix"
        ],
        index=[
            f"true_{label}"
            for label in LABELS
        ],
        columns=[
            f"pred_{label}"
            for label in LABELS
        ],
    )

    confusion_frame.to_csv(
        output_dir
        / "test_confusion_matrix.csv",
    )

    predictions.loc[
        predictions[
            "absolute_class_distance"
        ] >= 2
    ].to_csv(
        output_dir
        / "test_severe_errors.csv",
        index=False,
    )

    predictions.loc[
        (
            predictions["true_label"]
            == "C"
        )
        & (
            predictions["pred_label"]
            == "D"
        )
    ].to_csv(
        output_dir
        / "test_c_to_d_errors.csv",
        index=False,
    )

    predictions.loc[
        (
            predictions["true_label"]
            == "D"
        )
        & (
            predictions["pred_label"]
            == "C"
        )
    ].to_csv(
        output_dir
        / "test_d_to_c_errors.csv",
        index=False,
    )

    result_payload = {
        "status": "PASS",
        "evaluation_protocol": (
            "FINAL_ONE_TIME_TN_LOCKED_TEST132"
        ),
        "selected_experiment": "E1",
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": (
            checkpoint_sha256
        ),
        "decoder": selected_decoder,
        "metrics": test_metrics,
        "tn_test_evaluated": True,
        "vindr_test_evaluated": False,
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json(
        output_dir
        / "test_metrics.json",
        result_payload,
    )

    write_json(
        output_dir
        / "validation_reproduction.json",
        {
            "expected": (
                expected_valid_metrics
            ),
            "flat_decoder_metrics": (
                flat_valid_metrics
            ),
            "ordinal_decoder_metrics": (
                ordinal_valid_metrics
            ),
            "flat_metric_distance": (
                flat_distance
            ),
            "ordinal_metric_distance": (
                ordinal_distance
            ),
            "selected_decoder": (
                selected_decoder
            ),
            "reproduction_pass": True,
        },
    )

    write_json(
        output_dir
        / "data_audit.json",
        {
            "valid_loader_audit": (
                valid_audit
            ),
            "test_loader_audit": (
                test_audit
            ),
            "test_rows": 132,
            "test_class_counts": (
                test_counts
            ),
            "train_test_case_overlap": 0,
            "valid_test_case_overlap": 0,
        },
    )

    pipeline_done = "\n".join([
        "STATUS=PASS",
        "EVALUATION_PROTOCOL=FINAL_ONE_TIME_TN_LOCKED_TEST132",
        "SELECTED_EXPERIMENT=E1",
        (
            "SELECTED_CHECKPOINT="
            f"{selected_checkpoint}"
        ),
        (
            "CHECKPOINT_SHA256="
            f"{checkpoint_sha256}"
        ),
        (
            "DECODER="
            f"{selected_decoder}"
        ),
        (
            "TEST_ACCURACY="
            f"{test_metrics['accuracy']:.10f}"
        ),
        (
            "TEST_BALANCED_ACCURACY="
            f"{test_metrics['balanced_accuracy']:.10f}"
        ),
        (
            "TEST_MACRO_F1="
            f"{test_metrics['macro_f1']:.10f}"
        ),
        (
            "TEST_WEIGHTED_F1="
            f"{test_metrics['weighted_f1']:.10f}"
        ),
        (
            "TEST_QWK="
            f"{test_metrics['qwk']:.10f}"
        ),
        (
            "TEST_C_TO_D="
            f"{test_metrics['c_to_d']}"
        ),
        (
            "TEST_D_TO_C="
            f"{test_metrics['d_to_c']}"
        ),
        (
            "TEST_CD_TOTAL="
            f"{test_metrics['cd_total']}"
        ),
        (
            "TEST_SEVERE_ERRORS="
            f"{test_metrics['severe_error_count']}"
        ),
        "TN_TEST_EVALUATED=True",
        "VINDR_TEST_EVALUATED=False",
        (
            "COMPLETED_AT="
            + datetime.now(
                timezone.utc
            ).isoformat()
        ),
    ]) + "\n"

    (
        output_dir
        / "PIPELINE_DONE.txt"
    ).write_text(
        pipeline_done,
        encoding="utf-8",
    )

    marker_payload = {
        "status": "EVALUATED",
        "one_time_evaluation": True,
        "output_dir": str(
            output_dir
        ),
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": (
            checkpoint_sha256
        ),
        "test_manifest": str(
            TEST_MANIFEST
        ),
        "test_manifest_sha256": (
            test_manifest_sha256
        ),
        "decoder": selected_decoder,
        "test_macro_f1": (
            test_metrics["macro_f1"]
        ),
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "further_test_driven_model_selection_allowed": False,
    }

    write_json(
        GLOBAL_MARKER,
        marker_payload,
    )

    (
        OUTPUTS
        / "FINAL_LOCKED_TEST_OUTPUT.txt"
    ).write_text(
        str(output_dir) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("FINAL TN LOCKED TEST RESULTS")
    print("=" * 76)
    print(
        "ACCURACY="
        f"{test_metrics['accuracy']:.10f}"
    )
    print(
        "BALANCED_ACCURACY="
        f"{test_metrics['balanced_accuracy']:.10f}"
    )
    print(
        "MACRO_F1="
        f"{test_metrics['macro_f1']:.10f}"
    )
    print(
        "WEIGHTED_F1="
        f"{test_metrics['weighted_f1']:.10f}"
    )
    print(
        "QWK="
        f"{test_metrics['qwk']:.10f}"
    )
    print(
        "WITHIN_ONE="
        f"{test_metrics['within_one']:.10f}"
    )
    print(
        "SEVERE_ERRORS="
        f"{test_metrics['severe_error_count']}"
    )
    print(
        "C_TO_D="
        f"{test_metrics['c_to_d']}"
    )
    print(
        "D_TO_C="
        f"{test_metrics['d_to_c']}"
    )
    print(
        "CD_TOTAL="
        f"{test_metrics['cd_total']}"
    )
    print(
        "CONFUSION_MATRIX="
        f"{test_metrics['confusion_matrix']}"
    )

    for label in LABELS:
        values = test_metrics[
            "per_class"
        ][label]

        print(
            f"CLASS_{label}: "
            f"precision={values['precision']:.6f} "
            f"recall={values['recall']:.6f} "
            f"f1={values['f1']:.6f} "
            f"support={values['support']}"
        )

    print(f"OUTPUT_DIR={output_dir}")
    print("TN_TEST_EVALUATED=True")
    print("VINDR_TEST_EVALUATED=False")


if __name__ == "__main__":
    main()
