from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
)

from tn_mammo.constants import (
    INDEX_TO_LABEL,
    VIEW_ORDER,
)
from tn_mammo.data import (
    PhaseGDatasetAdapter,
    build_target_aware_sampler,
    compute_domain_sample_weights,
    realized_domain_mass,
)
from tn_mammo.losses import (
    MultiTaskCriterion,
    MultiTaskLossOptions,
)
from tn_mammo.metrics import (
    compute_classification_metrics,
)
from tn_mammo.models import (
    FourViewDensityModel,
    ModelOptions,
)


TN_CLASS_COUNTS = [
    12,
    81,
    178,
    140,
]

TN_BINARY_COUNTS = [
    12 + 81,
    178 + 140,
]


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def set_global_seed(
    seed: int,
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(
    worker_id: int,
) -> None:
    worker_seed = (
        torch.initial_seed()
        % (2**32)
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def flatten_config_paths(
    config: dict[str, Any],
) -> list[str]:
    paths: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(
            value,
            (list, tuple),
        ):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            paths.append(value)

    walk(config)

    return paths


def enforce_no_locked_test(
    config: dict[str, Any],
) -> None:
    forbidden = (
        "tn_locked_test132",
        "vindr_locked_test992",
        "locked_test",
        "test132.csv",
        "test992.csv",
    )

    relevant = {
        "train": config.get(
            "data",
            {},
        ).get("train", {}),
        "validation": config.get(
            "data",
            {},
        ).get("validation", {}),
    }

    for value in flatten_config_paths(
        relevant
    ):
        lower = value.lower()

        if any(
            token in lower
            for token in forbidden
        ):
            raise RuntimeError(
                "Locked-test reference found "
                f"in training configuration: {value}"
            )


def dataset_summary(
    dataset: PhaseGDatasetAdapter,
) -> dict[str, Any]:
    dataframe = dataset.dataframe

    class_counts = Counter(
        str(value)
        for value in dataframe[
            "label"
        ].tolist()
    )

    source_column = (
        "domain"
        if "domain" in dataframe.columns
        else "source"
    )

    source_counts = Counter(
        str(value)
        for value in dataframe[
            source_column
        ].tolist()
    )

    duplicate_count = int(
        dataframe[
            "case_id"
        ].astype(str).duplicated().sum()
    )

    return {
        "manifest": str(
            dataset.manifest_path
        ),
        "rows": len(dataset),
        "class_counts": dict(
            class_counts
        ),
        "source_counts": dict(
            source_counts
        ),
        "duplicate_case_ids": (
            duplicate_count
        ),
        "reference_transform": (
            dataset.reference_transform
        ),
        "reference_signature": (
            dataset.reference_signature
        ),
    }


def assert_disjoint_dataframes(
    first: PhaseGDatasetAdapter,
    second: PhaseGDatasetAdapter,
) -> None:
    first_cases = set(
        first.dataframe[
            "case_id"
        ].astype(str)
    )

    second_cases = set(
        second.dataframe[
            "case_id"
        ].astype(str)
    )

    overlap = first_cases & second_cases

    if overlap:
        raise RuntimeError(
            "Train-validation case overlap: "
            f"{sorted(overlap)[:20]}"
        )


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[
    DataLoader,
    DataLoader,
    dict[str, Any],
]:
    data_config = config["data"]
    training_config = config["training"]

    image_size = int(
        data_config["image_size"]
    )

    tn_train = PhaseGDatasetAdapter(
        data_config[
            "train"
        ][
            "tn_manifest"
        ],
        image_size=image_size,
        training=True,
    )

    vindr_train = PhaseGDatasetAdapter(
        data_config[
            "train"
        ][
            "vindr_manifest"
        ],
        image_size=image_size,
        training=True,
    )

    tn_valid = PhaseGDatasetAdapter(
        data_config[
            "validation"
        ][
            "manifest"
        ],
        image_size=image_size,
        training=False,
    )

    assert_disjoint_dataframes(
        tn_train,
        tn_valid,
    )

    combined_train = ConcatDataset([
        tn_train,
        vindr_train,
    ])

    domains = (
        ["TN"] * len(tn_train)
        + ["VinDr"] * len(vindr_train)
    )

    seed = int(
        config[
            "experiment"
        ][
            "seed"
        ]
    )

    sampler_generator = (
        torch.Generator()
    )
    sampler_generator.manual_seed(seed)

    num_samples = int(
        training_config.get(
            "sampler_num_samples",
            len(combined_train),
        )
    )

    tn_ratio = float(
        data_config[
            "tn_domain_ratio"
        ]
    )

    sampler = build_target_aware_sampler(
        domains,
        tn_ratio=tn_ratio,
        num_samples=num_samples,
        generator=sampler_generator,
    )

    weights = (
        compute_domain_sample_weights(
            domains,
            tn_ratio=tn_ratio,
        )
    )

    theoretical_mass = (
        realized_domain_mass(
            domains,
            weights,
        )
    )

    loader_generator = (
        torch.Generator()
    )
    loader_generator.manual_seed(
        seed + 1000
    )

    batch_size = int(
        training_config[
            "batch_size"
        ]
    )

    num_workers = int(
        training_config.get(
            "num_workers",
            4,
        )
    )

    common_loader_kwargs: dict[
        str,
        Any,
    ] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": (
            num_workers > 0
        ),
        "generator": loader_generator,
    }

    if num_workers > 0:
        common_loader_kwargs[
            "prefetch_factor"
        ] = int(
            training_config.get(
                "prefetch_factor",
                2,
            )
        )

    train_loader = DataLoader(
        combined_train,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    valid_loader = DataLoader(
        tn_valid,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    audit = {
        "tn_train": dataset_summary(
            tn_train
        ),
        "vindr_train": dataset_summary(
            vindr_train
        ),
        "tn_validation": dataset_summary(
            tn_valid
        ),
        "combined_train_rows": len(
            combined_train
        ),
        "sampler_num_samples": (
            num_samples
        ),
        "sampler_theoretical_mass": (
            theoretical_mass
        ),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "view_order": list(VIEW_ORDER),
        "test_manifest_read": False,
    }

    return (
        train_loader,
        valid_loader,
        audit,
    )


def build_model(
    config: dict[str, Any],
) -> tuple[
    FourViewDensityModel,
    dict[str, Any],
]:
    model_config = config["model"]

    options = ModelOptions(
        use_ordinal_head=bool(
            model_config.get(
                "use_ordinal_head",
                False,
            )
        ),
        use_binary_head=bool(
            model_config.get(
                "use_binary_head",
                False,
            )
        ),
        imagenet_init=bool(
            model_config.get(
                "imagenet_init",
                False,
            )
        ),
        fusion=str(
            model_config.get(
                "fusion",
                "mean",
            )
        ),
        fusion_dropout=float(
            model_config.get(
                "fusion_dropout",
                0.1,
            )
        ),
        control_hidden_dim=int(
            model_config.get(
                "control_hidden_dim",
                608,
            )
        ),
        bilateral_bottleneck_dim=int(
            model_config.get(
                "bilateral_bottleneck_dim",
                256,
            )
        ),
    )

    model = FourViewDensityModel(
        options
    )

    initialization_report: dict[
        str,
        Any,
    ] = {
        "imagenet_init": (
            options.imagenet_init
        ),
        "checkpoint": None,
        "checkpoint_loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }

    checkpoint_path_raw = (
        model_config.get(
            "initialization_checkpoint"
        )
    )

    if checkpoint_path_raw:
        checkpoint_path = Path(
            checkpoint_path_raw
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Initialization checkpoint "
                f"not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
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
            strict=False,
        )

        initialization_report.update({
            "checkpoint": str(
                checkpoint_path
            ),
            "checkpoint_loaded": True,
            "missing_keys": list(
                load_result.missing_keys
            ),
            "unexpected_keys": list(
                load_result.unexpected_keys
            ),
        })

        allowed_missing_prefixes = (
            "ordinal_head.",
            "binary_head.",
            "fusion_module.",
        )

        illegal_missing = [
            key
            for key in load_result.missing_keys
            if not key.startswith(
                allowed_missing_prefixes
            )
        ]

        if illegal_missing:
            raise RuntimeError(
                "Illegal missing keys during "
                f"initialization: {illegal_missing}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint keys: "
                f"{load_result.unexpected_keys}"
            )

    return (
        model,
        initialization_report,
    )


def build_criterion(
    config: dict[str, Any],
) -> MultiTaskCriterion:
    loss_config = config["loss"]

    return MultiTaskCriterion(
        flat_class_counts=(
            TN_CLASS_COUNTS
        ),
        binary_class_counts=(
            TN_BINARY_COUNTS
        ),
        beta=float(
            loss_config[
                "flat"
            ][
                "beta"
            ]
        ),
        gamma=float(
            loss_config[
                "flat"
            ][
                "gamma"
            ]
        ),
        options=MultiTaskLossOptions(
            lambda_ordinal=float(
                loss_config.get(
                    "lambda_ordinal",
                    0.0,
                )
            ),
            lambda_binary=float(
                loss_config.get(
                    "lambda_binary",
                    0.0,
                )
            ),
        ),
    )


def move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    views = batch[
        "views"
    ].to(
        device,
        non_blocking=True,
    )

    labels = batch[
        "label"
    ].to(
        device,
        non_blocking=True,
    ).long()

    if views.ndim != 5:
        raise RuntimeError(
            "Expected batched views with shape "
            "[B,4,3,H,W], received "
            f"{tuple(views.shape)}."
        )

    if tuple(
        views.shape[1:3]
    ) != (4, 3):
        raise RuntimeError(
            "Four RGB views are required; "
            f"received {tuple(views.shape)}."
        )

    return views, labels


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskCriterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    gradient_clip_norm: float | None,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_samples = 0
    component_sums: Counter[str] = (
        Counter()
    )

    for batch_index, batch in enumerate(
        loader
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        views, labels = move_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(views)

            loss, parts = criterion(
                outputs,
                labels,
            )

        scaler.scale(loss).backward()

        if (
            gradient_clip_norm is not None
            and gradient_clip_norm > 0
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )

        scaler.step(optimizer)
        scaler.update()

        batch_size = int(
            labels.shape[0]
        )

        total_loss += float(
            loss.detach().item()
        ) * batch_size

        total_samples += batch_size

        for key, value in parts.items():
            component_sums[key] += (
                float(value.item())
                * batch_size
            )

    if total_samples == 0:
        raise RuntimeError(
            "No training samples were processed."
        )

    result = {
        "loss": (
            total_loss / total_samples
        ),
        "samples": float(
            total_samples
        ),
    }

    for key, value in component_sums.items():
        result[
            f"component_{key}"
        ] = value / total_samples

    return result


@torch.no_grad()
def validate_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiTaskCriterion,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true: list[int] = []
    y_pred: list[int] = []
    prediction_rows: list[
        dict[str, Any]
    ] = []

    for batch_index, batch in enumerate(
        loader
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        views, labels = move_batch(
            batch,
            device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(views)

            loss, _ = criterion(
                outputs,
                labels,
            )

        flat_logits = outputs[
            "flat_logits"
        ]

        if flat_logits is None:
            raise RuntimeError(
                "flat_logits are missing."
            )

        probabilities = torch.softmax(
            flat_logits.float(),
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1
        )

        labels_cpu = labels.detach().cpu()
        predictions_cpu = (
            predictions.detach().cpu()
        )
        probabilities_cpu = (
            probabilities.detach().cpu()
        )

        batch_size = int(
            labels.shape[0]
        )

        total_loss += float(
            loss.item()
        ) * batch_size
        total_samples += batch_size

        y_true.extend(
            labels_cpu.tolist()
        )
        y_pred.extend(
            predictions_cpu.tolist()
        )

        case_ids = list(
            batch["case_id"]
        )

        sources = list(
            batch["source"]
        )

        for item_index in range(
            batch_size
        ):
            true_index = int(
                labels_cpu[item_index]
            )
            predicted_index = int(
                predictions_cpu[
                    item_index
                ]
            )

            row = {
                "case_id": str(
                    case_ids[item_index]
                ),
                "source": str(
                    sources[item_index]
                ),
                "true_index": true_index,
                "true_label": (
                    INDEX_TO_LABEL[
                        true_index
                    ]
                ),
                "predicted_index": (
                    predicted_index
                ),
                "predicted_label": (
                    INDEX_TO_LABEL[
                        predicted_index
                    ]
                ),
            }

            for class_index in range(4):
                row[
                    f"prob_{INDEX_TO_LABEL[class_index]}"
                ] = float(
                    probabilities_cpu[
                        item_index,
                        class_index,
                    ]
                )

            prediction_rows.append(row)

    if total_samples == 0:
        raise RuntimeError(
            "No validation samples were processed."
        )

    metrics = (
        compute_classification_metrics(
            y_true,
            y_pred,
        )
    )

    metrics["loss"] = (
        total_loss / total_samples
    )

    return metrics, prediction_rows


def write_prediction_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_history_csv(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return

    fieldnames: list[str] = []

    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def save_checkpoint(
    *,
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    valid_metrics: dict[str, Any],
    best_macro_f1: float,
    data_audit: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
        ),
        "scaler_state_dict": (
            scaler.state_dict()
        ),
        "best_macro_f1": (
            best_macro_f1
        ),
        "valid_metrics": (
            valid_metrics
        ),
        "config": config,
        "class_counts": (
            TN_CLASS_COUNTS
        ),
        "class_weight_source": (
            "TN train only"
        ),
        "training_domains": [
            "TN",
            "VinDr",
        ],
        "tn_domain_ratio": (
            config[
                "data"
            ][
                "tn_domain_ratio"
            ]
        ),
        "label_to_index": {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
        },
        "view_order": list(
            VIEW_ORDER
        ),
        "model_name": (
            config[
                "model"
            ][
                "backbone"
            ]
        ),
        "fusion": (
            config[
                "model"
            ][
                "fusion"
            ]
        ),
        "image_size": (
            config[
                "data"
            ][
                "image_size"
            ]
        ),
        "data_audit": data_audit,
        "test_evaluated": False,
        "saved_at": utc_now(),
    }

    torch.save(
        checkpoint,
        path,
    )


def make_optimizer_and_scheduler(
    *,
    model: nn.Module,
    config: dict[str, Any],
) -> tuple[
    torch.optim.Optimizer,
    Any,
]:
    training_config = config[
        "training"
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training_config[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training_config[
                "weight_decay"
            ]
        ),
    )

    scheduler_config = (
        training_config.get(
            "scheduler",
            {},
        )
    )

    scheduler_name = str(
        scheduler_config.get(
            "name",
            "step_lr",
        )
    ).lower()

    if scheduler_name == "step_lr":
        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(
                    scheduler_config.get(
                        "step_size",
                        5,
                    )
                ),
                gamma=float(
                    scheduler_config.get(
                        "gamma",
                        0.5,
                    )
                ),
            )
        )
    else:
        raise ValueError(
            "Unsupported scheduler: "
            f"{scheduler_name}"
        )

    return optimizer, scheduler


def run_training(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    max_epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_valid_batches: int | None = None,
    require_cuda: bool = True,
) -> dict[str, Any]:
    enforce_no_locked_test(config)

    output_path = Path(
        output_dir
    )
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    running_marker = (
        output_path / "RUNNING.txt"
    )
    done_marker = (
        output_path / "PIPELINE_DONE.txt"
    )
    failed_marker = (
        output_path / "PIPELINE_FAILED.txt"
    )

    running_marker.write_text(
        (
            f"started_at={utc_now()}\n"
            f"pid={os.getpid()}\n"
            "test_evaluated=False\n"
        ),
        encoding="utf-8",
    )

    if done_marker.exists():
        done_marker.unlink()

    if failed_marker.exists():
        failed_marker.unlink()

    seed = int(
        config[
            "experiment"
        ][
            "seed"
        ]
    )

    set_global_seed(seed)

    if (
        require_cuda
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is required but unavailable."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    training_config = config[
        "training"
    ]

    amp_enabled = bool(
        training_config.get(
            "amp",
            True,
        )
        and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    train_loader, valid_loader, data_audit = (
        build_dataloaders(config)
    )

    atomic_write_json(
        output_path / "data_audit.json",
        data_audit,
    )

    model, initialization_report = (
        build_model(config)
    )

    model = model.to(device)

    criterion = build_criterion(
        config
    ).to(device)

    optimizer, scheduler = (
        make_optimizer_and_scheduler(
            model=model,
            config=config,
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    configured_epochs = int(
        training_config[
            "epochs"
        ]
    )

    epochs = (
        int(max_epochs_override)
        if max_epochs_override is not None
        else configured_epochs
    )

    patience = int(
        training_config.get(
            "early_stopping_patience",
            10,
        )
    )

    gradient_clip_raw = (
        training_config.get(
            "gradient_clip_norm",
            5.0,
        )
    )

    gradient_clip_norm = (
        float(gradient_clip_raw)
        if gradient_clip_raw is not None
        else None
    )

    run_config = {
        "created_at": utc_now(),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "torch_version": (
            torch.__version__
        ),
        "amp_enabled": amp_enabled,
        "max_epochs_override": (
            max_epochs_override
        ),
        "max_train_batches": (
            max_train_batches
        ),
        "max_valid_batches": (
            max_valid_batches
        ),
        "config": config,
        "initialization": (
            initialization_report
        ),
        "test_evaluated": False,
    }

    atomic_write_json(
        output_path / "run_config.json",
        run_config,
    )

    best_macro_f1 = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    best_checkpoint = (
        output_path
        / "best_checkpoint.pt"
    )

    last_checkpoint = (
        output_path
        / "last_checkpoint.pt"
    )

    print(
        "[START]",
        json.dumps({
            "output_dir": str(
                output_path
            ),
            "device": str(device),
            "cuda_name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else None
            ),
            "epochs": epochs,
            "train_batches": len(
                train_loader
            ),
            "valid_batches": len(
                valid_loader
            ),
            "tn_ratio": config[
                "data"
            ][
                "tn_domain_ratio"
            ],
            "test_evaluated": False,
        }),
        flush=True,
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        epoch_start = time.time()

        learning_rate = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            gradient_clip_norm=(
                gradient_clip_norm
            ),
            max_batches=(
                max_train_batches
            ),
        )

        valid_metrics, prediction_rows = (
            validate_one_epoch(
                model=model,
                loader=valid_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                max_batches=(
                    max_valid_batches
                ),
            )
        )

        scheduler.step()

        current_macro_f1 = float(
            valid_metrics[
                "macro_f1"
            ]
        )

        improved = (
            current_macro_f1
            > best_macro_f1 + 1e-12
        )

        if improved:
            best_macro_f1 = (
                current_macro_f1
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_checkpoint,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                valid_metrics=(
                    valid_metrics
                ),
                best_macro_f1=(
                    best_macro_f1
                ),
                data_audit=data_audit,
            )

            atomic_write_json(
                output_path
                / "best_valid_metrics.json",
                valid_metrics,
            )

            write_prediction_csv(
                output_path
                / "best_valid_predictions.csv",
                prediction_rows,
            )

            print(
                "[BEST]",
                json.dumps({
                    "epoch": epoch,
                    "macro_f1": (
                        best_macro_f1
                    ),
                    "balanced_accuracy": (
                        valid_metrics[
                            "balanced_accuracy"
                        ]
                    ),
                    "accuracy": (
                        valid_metrics[
                            "accuracy"
                        ]
                    ),
                }),
                flush=True,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=last_checkpoint,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            valid_metrics=valid_metrics,
            best_macro_f1=best_macro_f1,
            data_audit=data_audit,
        )

        elapsed = (
            time.time() - epoch_start
        )

        record = {
            "epoch": epoch,
            "learning_rate": (
                learning_rate
            ),
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_samples": int(
                train_metrics[
                    "samples"
                ]
            ),
            "valid_loss": (
                valid_metrics["loss"]
            ),
            "valid_samples": (
                valid_metrics[
                    "num_samples"
                ]
            ),
            "valid_accuracy": (
                valid_metrics[
                    "accuracy"
                ]
            ),
            "valid_balanced_accuracy": (
                valid_metrics[
                    "balanced_accuracy"
                ]
            ),
            "valid_macro_f1": (
                valid_metrics[
                    "macro_f1"
                ]
            ),
            "valid_weighted_f1": (
                valid_metrics[
                    "weighted_f1"
                ]
            ),
            "valid_qwk": (
                valid_metrics["qwk"]
            ),
            "valid_within_one": (
                valid_metrics[
                    "within_one"
                ]
            ),
            "valid_severe_error_count": (
                valid_metrics[
                    "severe_error_count"
                ]
            ),
            "valid_c_to_d": (
                valid_metrics[
                    "c_to_d"
                ]
            ),
            "valid_d_to_c": (
                valid_metrics[
                    "d_to_c"
                ]
            ),
            "valid_cd_total": (
                valid_metrics[
                    "c_to_d"
                ]
                + valid_metrics[
                    "d_to_c"
                ]
            ),
            "best_macro_f1": (
                best_macro_f1
            ),
            "best_epoch": best_epoch,
            "improved": improved,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "epoch_seconds": elapsed,
        }

        history.append(record)

        write_history_csv(
            output_path / "history.csv",
            history,
        )

        atomic_write_json(
            output_path
            / "latest_valid_metrics.json",
            valid_metrics,
        )

        print(
            "[EPOCH]",
            json.dumps(record),
            flush=True,
        )

        if (
            max_epochs_override is None
            and epochs_without_improvement
            >= patience
        ):
            print(
                "[EARLY_STOP]",
                json.dumps({
                    "epoch": epoch,
                    "patience": patience,
                    "best_epoch": best_epoch,
                    "best_macro_f1": (
                        best_macro_f1
                    ),
                }),
                flush=True,
            )
            break

    peak_gpu_memory = None

    if device.type == "cuda":
        peak_gpu_memory = int(
            torch.cuda.max_memory_allocated()
        )

    final_summary = {
        "status": "PASS",
        "experiment": (
            config["experiment"]
        ),
        "output_dir": str(
            output_path
        ),
        "best_epoch": best_epoch,
        "best_valid_macro_f1": (
            best_macro_f1
        ),
        "epochs_completed": (
            history[-1]["epoch"]
            if history
            else 0
        ),
        "peak_gpu_memory_bytes": (
            peak_gpu_memory
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
        "last_checkpoint": str(
            last_checkpoint
        ),
        "tn_test_evaluated": False,
        "vindr_test_evaluated": False,
        "completed_at": utc_now(),
    }

    atomic_write_json(
        output_path / "final_summary.json",
        final_summary,
    )

    done_marker.write_text(
        (
            "STATUS=PASS\n"
            f"BEST_EPOCH={best_epoch}\n"
            "BEST_VALID_MACRO_F1="
            f"{best_macro_f1:.10f}\n"
            "TN_TEST_EVALUATED=False\n"
            "VINDR_TEST_EVALUATED=False\n"
        ),
        encoding="utf-8",
    )

    if running_marker.exists():
        running_marker.unlink()

    print(
        "[PIPELINE_DONE]",
        json.dumps(final_summary),
        flush=True,
    )

    return final_summary


def run_with_failure_record(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    max_epochs_override: int | None,
    max_train_batches: int | None,
    max_valid_batches: int | None,
    require_cuda: bool,
) -> dict[str, Any]:
    output_path = Path(
        output_dir
    )
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        return run_training(
            config=config,
            output_dir=output_path,
            max_epochs_override=(
                max_epochs_override
            ),
            max_train_batches=(
                max_train_batches
            ),
            max_valid_batches=(
                max_valid_batches
            ),
            require_cuda=require_cuda,
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": (
                type(exc).__name__
            ),
            "error": str(exc),
            "traceback": (
                traceback.format_exc()
            ),
            "tn_test_evaluated": False,
            "vindr_test_evaluated": False,
            "failed_at": utc_now(),
        }

        atomic_write_json(
            output_path
            / "failure_report.json",
            failure,
        )

        (
            output_path
            / "PIPELINE_FAILED.txt"
        ).write_text(
            (
                "STATUS=FAIL\n"
                "ERROR_TYPE="
                f"{type(exc).__name__}\n"
                f"ERROR={exc}\n"
                "TN_TEST_EVALUATED=False\n"
                "VINDR_TEST_EVALUATED=False\n"
            ),
            encoding="utf-8",
        )

        print(
            "[PIPELINE_FAILED]",
            json.dumps(
                failure,
                ensure_ascii=False,
            ),
            flush=True,
        )

        raise
