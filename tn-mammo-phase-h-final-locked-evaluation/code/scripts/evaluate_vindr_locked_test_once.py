from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from tn_mammo.training.engine import (
    build_dataloaders,
    build_model,
)


ROOT = Path("/mnt/hcmus/breast_vn/code/new_implement")
OUTPUTS = ROOT / "outputs"

TEST_MANIFEST = (
    ROOT
    / "manifests"
    / "vindr_final_eval992_phaseg.csv"
)

MARKER = (
    OUTPUTS
    / "VINDR_LOCKED_TEST992_FINAL_EVALUATED.json"
)

TN_HELPER_PATH = (
    ROOT
    / "scripts"
    / "evaluate_tn_locked_test_once.py"
)

SELECTED_OUTPUT_POINTER = (
    OUTPUTS
    / "CURRENT_SELECTED_OUTPUT.txt"
)

SELECTED_CHECKPOINT_POINTER = (
    OUTPUTS
    / "CURRENT_SELECTED_CHECKPOINT.txt"
)

FINAL_TN_OUTPUT_POINTER = (
    OUTPUTS
    / "FINAL_LOCKED_TEST_OUTPUT.txt"
)


def load_helper_module():
    spec = importlib.util.spec_from_file_location(
        "tn_final_eval_helper",
        TN_HELPER_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot import TN evaluation helper."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


helper = load_helper_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def locate_training_config(
    node: Any,
) -> dict[str, Any] | None:
    if isinstance(node, dict):
        required = {
            "data",
            "model",
            "training",
        }

        if required.issubset(
            node.keys()
        ):
            return node

        for key in (
            "config",
            "resolved_config",
            "run_config",
            "configuration",
            "settings",
        ):
            if key in node:
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


def main() -> None:
    if MARKER.is_file():
        print(
            "VINDR_LOCKED_TEST_ALREADY_EVALUATED=True"
        )

        print(
            MARKER.read_text(
                encoding="utf-8"
            )
        )

        raise RuntimeError(
            "VinDr locked test marker already exists."
        )

    required_paths = [
        TEST_MANIFEST,
        SELECTED_OUTPUT_POINTER,
        SELECTED_CHECKPOINT_POINTER,
        FINAL_TN_OUTPUT_POINTER,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

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

    final_tn_output = Path(
        FINAL_TN_OUTPUT_POINTER.read_text(
            encoding="utf-8"
        ).strip()
    )

    locked_config_path = (
        final_tn_output
        / "LOCKED_CONFIG.json"
    )

    if not locked_config_path.is_file():
        raise FileNotFoundError(
            locked_config_path
        )

    locked_config = json.loads(
        locked_config_path.read_text(
            encoding="utf-8"
        )
    )

    checkpoint_sha256 = sha256_file(
        selected_checkpoint
    )

    if (
        checkpoint_sha256
        != locked_config["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "Checkpoint does not match final TN lock."
        )

    if (
        locked_config["decoder"]
        != "flat_argmax"
    ):
        raise RuntimeError(
            "Expected locked decoder flat_argmax."
        )

    run_config_path = (
        selected_output
        / "run_config.json"
    )

    raw_config = json.loads(
        run_config_path.read_text(
            encoding="utf-8"
        )
    )

    config = locate_training_config(
        raw_config
    )

    if config is None:
        raise RuntimeError(
            "Could not resolve E1 training config."
        )

    config = json.loads(
        json.dumps(config)
    )

    config["model"][
        "initialization_checkpoint"
    ] = str(selected_checkpoint)

    config["model"]["imagenet_init"] = False
    config["model"]["fusion"] = "mean"
    config["model"][
        "use_ordinal_head"
    ] = True
    config["model"][
        "use_binary_head"
    ] = False

    config["training"]["batch_size"] = 2
    config["training"]["num_workers"] = 4
    config["training"]["amp"] = True

    valid_manifest = Path(
        config["data"][
            "validation"
        ]["manifest"]
    )

    valid_frame, valid_id_column, _ = (
        helper.load_manifest_identity(
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
    print("STEP 1 — REPRODUCE LOCKED TN VALIDATION")
    print("=" * 76)
    print(f"CHECKPOINT={selected_checkpoint}")
    print(f"CHECKPOINT_SHA256={checkpoint_sha256}")
    print("LOCKED_DECODER=flat_argmax")
    print("VINDR_TEST_MANIFEST_READ=False")

    _, valid_loader, _ = (
        build_dataloaders(config)
    )

    model, _ = build_model(config)

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

    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading failed."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    valid_inference = helper.run_inference(
        model=model,
        loader=valid_loader,
        device=device,
        amp_enabled=True,
        manifest_case_order=valid_case_order,
    )

    valid_metrics = helper.compute_metrics(
        valid_inference["true"],
        valid_inference["flat_pred"],
    )

    expected_valid_macro_f1 = float(
        locked_config[
            "validation_macro_f1"
        ]
    )

    observed_valid_macro_f1 = float(
        valid_metrics["macro_f1"]
    )

    print(
        "EXPECTED_VALID_MACRO_F1="
        f"{expected_valid_macro_f1:.10f}"
    )

    print(
        "OBSERVED_VALID_MACRO_F1="
        f"{observed_valid_macro_f1:.10f}"
    )

    if (
        abs(
            observed_valid_macro_f1
            - expected_valid_macro_f1
        )
        > 1e-10
    ):
        raise RuntimeError(
            "Validation reproduction failed. "
            "VinDr test remains unopened."
        )

    print("VALIDATION_REPRODUCTION_PASS=True")
    print("VINDR_TEST_MANIFEST_READ=False")

    test_frame, test_id_column, _ = (
        helper.load_manifest_identity(
            TEST_MANIFEST
        )
    )

    if len(test_frame) != 992:
        raise RuntimeError(
            f"Expected 992 VinDr cases, "
            f"found {len(test_frame)}."
        )

    test_case_order = (
        test_frame[test_id_column]
        .astype(str)
        .str.strip()
        .tolist()
    )

    test_manifest_sha256 = sha256_file(
        TEST_MANIFEST
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_dir = (
        OUTPUTS
        / f"FINAL_VINDR_LOCKED_TEST992_E1_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    lock_payload = {
        "status": "LOCKED_BEFORE_VINDR_TEST_INFERENCE",
        "selected_experiment": "E1",
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "decoder": "flat_argmax",
        "test_manifest": str(
            TEST_MANIFEST
        ),
        "test_manifest_sha256": (
            test_manifest_sha256
        ),
        "expected_test_rows": 992,
        "batch_size": 2,
        "physical_gpu": 1,
        "test_time_augmentation": False,
        "threshold_tuning": False,
        "calibration_on_test": False,
        "model_changed_after_tn_test": False,
        "decoder_changed_after_tn_test": False,
        "tn_test_result_used_for_selection": False,
        "locked_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    write_json(
        output_dir
        / "LOCKED_CONFIG.json",
        lock_payload,
    )

    print()
    print("=" * 76)
    print("STEP 2 — VINDR LOCKED TEST992 OPENED ONCE")
    print("=" * 76)
    print(f"OUTPUT_DIR={output_dir}")
    print(f"TEST_MANIFEST={TEST_MANIFEST}")
    print(
        "TEST_MANIFEST_SHA256="
        f"{test_manifest_sha256}"
    )
    print("VINDR_LOCKED_TEST_OPENED=True")

    test_config = json.loads(
        json.dumps(config)
    )

    test_config["data"][
        "validation"
    ]["manifest"] = str(
        TEST_MANIFEST
    )

    _, test_loader, test_audit = (
        build_dataloaders(
            test_config
        )
    )

    test_inference = helper.run_inference(
        model=model,
        loader=test_loader,
        device=device,
        amp_enabled=True,
        manifest_case_order=test_case_order,
    )

    if len(test_inference["true"]) != 992:
        raise RuntimeError(
            "VinDr inference did not return 992 cases."
        )

    predictions_array = (
        test_inference["flat_pred"]
    )

    test_metrics = helper.compute_metrics(
        test_inference["true"],
        predictions_array,
    )

    labels = ["A", "B", "C", "D"]

    predictions = pd.DataFrame({
        "case_id": (
            test_inference["case_ids"]
        ),
        "true_index": (
            test_inference["true"]
        ),
        "true_label": [
            labels[int(value)]
            for value in test_inference[
                "true"
            ]
        ],
        "pred_index": predictions_array,
        "pred_label": [
            labels[int(value)]
            for value in predictions_array
        ],
        "correct": (
            test_inference["true"]
            == predictions_array
        ).astype(int),
        "absolute_class_distance": np.abs(
            test_inference["true"]
            - predictions_array
        ),
        "decoder": "flat_argmax",
    })

    flat_probs = (
        test_inference["flat_probs"]
    )

    for index, label in enumerate(labels):
        predictions[
            f"prob_{label}"
        ] = flat_probs[:, index]

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
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ],
    )

    confusion_frame.to_csv(
        output_dir
        / "test_confusion_matrix.csv"
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

    result_payload = {
        "status": "PASS",
        "evaluation_protocol": (
            "FINAL_ONE_TIME_VINDR_LOCKED_TEST992"
        ),
        "selected_experiment": "E1",
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "decoder": "flat_argmax",
        "metrics": test_metrics,
        "tn_test_evaluated": True,
        "vindr_test_evaluated": True,
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
        / "data_audit.json",
        {
            "test_loader_audit": test_audit,
            "test_rows": 992,
            "test_manifest_sha256": (
                test_manifest_sha256
            ),
        },
    )

    pipeline_done = "\n".join([
        "STATUS=PASS",
        (
            "EVALUATION_PROTOCOL="
            "FINAL_ONE_TIME_VINDR_LOCKED_TEST992"
        ),
        "SELECTED_EXPERIMENT=E1",
        (
            "SELECTED_CHECKPOINT="
            f"{selected_checkpoint}"
        ),
        (
            "CHECKPOINT_SHA256="
            f"{checkpoint_sha256}"
        ),
        "DECODER=flat_argmax",
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
        "VINDR_TEST_EVALUATED=True",
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
        "output_dir": str(output_dir),
        "selected_checkpoint": str(
            selected_checkpoint
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "test_manifest": str(
            TEST_MANIFEST
        ),
        "test_manifest_sha256": (
            test_manifest_sha256
        ),
        "decoder": "flat_argmax",
        "test_macro_f1": (
            test_metrics["macro_f1"]
        ),
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "further_test_driven_selection_allowed": False,
    }

    write_json(
        MARKER,
        marker_payload,
    )

    (
        OUTPUTS
        / "FINAL_VINDR_LOCKED_TEST_OUTPUT.txt"
    ).write_text(
        str(output_dir) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("FINAL VINDR LOCKED TEST RESULTS")
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

    for label in labels:
        class_metrics = (
            test_metrics[
                "per_class"
            ][label]
        )

        print(
            f"CLASS_{label}: "
            f"precision={class_metrics['precision']:.6f} "
            f"recall={class_metrics['recall']:.6f} "
            f"f1={class_metrics['f1']:.6f} "
            f"support={class_metrics['support']}"
        )

    print(f"OUTPUT_DIR={output_dir}")
    print("TN_TEST_EVALUATED=True")
    print("VINDR_TEST_EVALUATED=True")


if __name__ == "__main__":
    main()
