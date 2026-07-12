import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

from phaseG_rerun.config import config
from phaseG_rerun.dataset import (
    build_datasets,
    build_domain_sampler,
    load_dataframes,
)
from phaseG_rerun.loss import ClassBalancedFocalLoss
from phaseG_rerun.metrics import (
    compute_classification_metrics,
    format_metrics,
)
from phaseG_rerun.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders() -> Tuple[
    DataLoader,
    DataLoader,
    pd.DataFrame,
]:
    train_df, valid_df, _ = load_dataframes(
        config
    )

    train_dataset, valid_dataset, _ = (
        build_datasets(
            config
        )
    )

    sampler = build_domain_sampler(
        dataframe=train_df,
        tn_domain_ratio=config.tn_domain_ratio,
        seed=config.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(
        config.seed
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=(
            config.num_workers > 0
        ),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )

    return (
        train_loader,
        valid_loader,
        train_df,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    smoke_test: bool = False,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    domain_counts = {
        "TN": 0,
        "VinDr": 0,
    }

    for batch_index, batch in enumerate(
        loader
    ):
        images = batch["images"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            non_blocking=True,
        )

        for domain in batch["domain"]:
            if domain not in domain_counts:
                raise ValueError(
                    f"Unsupported domain: {domain}"
                )

            domain_counts[domain] += 1

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=config.use_amp,
        ):
            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        predictions = logits.argmax(
            dim=1
        )

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += (
            batch_size
        )

        if smoke_test and batch_index >= 1:
            break

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "accuracy": (
            total_correct
            / total_samples
        ),
        "num_samples": total_samples,
        "tn_samples": domain_counts["TN"],
        "vindr_samples": domain_counts["VinDr"],
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    smoke_test: bool = False,
) -> Dict[str, object]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []

    for batch_index, batch in enumerate(
        loader
    ):
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
            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels,
            )

        predictions = logits.argmax(
            dim=1
        )

        batch_size = labels.size(
            0
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        if smoke_test and batch_index >= 1:
            break

    metrics = compute_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
    )

    metrics["loss"] = (
        total_loss
        / total_samples
    )

    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_macro_f1: float,
    valid_metrics: Dict[str, object],
    class_counts: list[int],
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
        "class_counts": (
            class_counts
        ),
        "class_weight_source": (
            "TN train only"
        ),
        "training_domains": [
            "TN",
            "VinDr",
        ],
        "tn_domain_ratio": (
            config.tn_domain_ratio
        ),
        "label_to_index": (
            config.label_to_index
        ),
        "view_columns": (
            config.view_columns
        ),
        "model_name": (
            config.model_name
        ),
        "fusion": (
            config.fusion
        ),
        "image_size": (
            config.image_size
        ),
    }

    torch.save(
        checkpoint,
        path,
    )


def config_to_dict() -> Dict[str, object]:
    values = asdict(
        config
    )

    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(
                value
            )

    values["class_weight_source"] = (
        "TN train only"
    )

    values["training_domains"] = [
        "TN",
        "VinDr",
    ]

    values["model_selection_split"] = (
        "TN validation"
    )

    values["test_used_during_training"] = (
        False
    )

    return values


def main() -> None:
    args = parse_args()

    if args.epochs is not None:
        config.epochs = (
            args.epochs
        )

    if args.batch_size is not None:
        config.batch_size = (
            args.batch_size
        )

    if args.num_workers is not None:
        config.num_workers = (
            args.num_workers
        )

    config.validate()
    config.create_directories()

    set_seed(
        config.seed
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        config.device
    )

    run_config_path = (
        config.output_dir
        / "run_config.json"
    )

    run_config_path.write_text(
        json.dumps(
            config_to_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    train_loader, valid_loader, train_df = (
        build_loaders()
    )

    tn_train_df = (
        train_df[
            train_df["domain"] == "TN"
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(tn_train_df) != 411:
        raise RuntimeError(
            "Expected 411 TN train cases, "
            f"found {len(tn_train_df)}."
        )

    class_counts = (
        tn_train_df["label_idx"]
        .value_counts()
        .reindex(
            range(
                config.num_classes
            ),
            fill_value=0,
        )
        .tolist()
    )

    expected_class_counts = [
        12,
        81,
        178,
        140,
    ]

    if class_counts != expected_class_counts:
        raise RuntimeError(
            "Unexpected TN train class counts. "
            f"Expected {expected_class_counts}, "
            f"found {class_counts}."
        )

    model = build_model(
        num_classes=config.num_classes,
        pretrained=True,
    ).to(
        device
    )

    criterion = ClassBalancedFocalLoss(
        class_counts=class_counts,
        beta=config.cb_beta,
        gamma=config.focal_gamma,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step_size,
        gamma=config.scheduler_gamma,
    )

    scaler = GradScaler(
        device="cuda",
        enabled=config.use_amp,
    )

    print(
        f"device: {device}"
    )

    print(
        "gpu:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "model:",
        config.model_name,
    )

    print(
        "fusion:",
        config.fusion,
    )

    print(
        "train cases:",
        len(
            train_loader.dataset
        ),
    )

    print(
        "valid cases:",
        len(
            valid_loader.dataset
        ),
    )

    print(
        "TN train cases:",
        len(
            tn_train_df
        ),
    )

    print(
        "class weight source: TN train only"
    )

    print(
        "class counts:",
        class_counts,
    )

    print(
        "class weights:",
        criterion.class_weights
        .detach()
        .cpu()
        .tolist(),
    )

    print(
        "TN domain ratio:",
        config.tn_domain_ratio,
    )

    print(
        "batch size:",
        config.batch_size,
    )

    print(
        "num workers:",
        config.num_workers,
    )

    print(
        "learning rate:",
        config.learning_rate,
    )

    print(
        "scheduler step size:",
        config.scheduler_step_size,
    )

    print(
        "scheduler gamma:",
        config.scheduler_gamma,
    )

    print(
        "AMP:",
        config.use_amp,
    )

    if args.smoke_test:
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            smoke_test=True,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            smoke_test=True,
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        scheduler.step()

        print(
            "\nsmoke train:"
        )

        print(
            train_metrics
        )

        print(
            "\nsmoke valid:"
        )

        print(
            format_metrics(
                valid_metrics
            )
        )

        print(
            f"loss: "
            f"{valid_metrics['loss']:.6f}"
        )

        print(
            "learning rate before step: "
            f"{current_learning_rate:.8f}"
        )

        print(
            "learning rate after step: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

        print(
            "\n[PASS] Phase G old-like smoke test"
        )

        return

    history = []

    best_macro_f1 = float(
        "-inf"
    )

    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = (
        config.output_dir
        / "best_model.pt"
    )

    last_model_path = (
        config.output_dir
        / "last_model.pt"
    )

    history_path = (
        config.output_dir
        / "history.csv"
    )

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        start_time = time.time()

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        valid_metrics = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        epoch_seconds = (
            time.time()
            - start_time
        )

        current_macro_f1 = float(
            valid_metrics["macro_f1"]
        )

        improved = (
            current_macro_f1
            > best_macro_f1
            + config.min_delta
        )

        if improved:
            best_macro_f1 = (
                current_macro_f1
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_macro_f1=best_macro_f1,
                valid_metrics=valid_metrics,
                class_counts=class_counts,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            path=last_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_macro_f1=best_macro_f1,
            valid_metrics=valid_metrics,
            class_counts=class_counts,
        )

        history_row = {
            "epoch": epoch,
            "learning_rate": (
                current_learning_rate
            ),
            "train_loss": (
                train_metrics["loss"]
            ),
            "train_accuracy": (
                train_metrics["accuracy"]
            ),
            "train_samples": (
                train_metrics["num_samples"]
            ),
            "train_tn_samples": (
                train_metrics["tn_samples"]
            ),
            "train_vindr_samples": (
                train_metrics[
                    "vindr_samples"
                ]
            ),
            "valid_loss": (
                valid_metrics["loss"]
            ),
            "valid_accuracy": (
                valid_metrics["accuracy"]
            ),
            "valid_balanced_accuracy": (
                valid_metrics[
                    "balanced_accuracy"
                ]
            ),
            "valid_macro_f1": (
                valid_metrics["macro_f1"]
            ),
            "valid_weighted_f1": (
                valid_metrics[
                    "weighted_f1"
                ]
            ),
            "valid_c_to_d": (
                valid_metrics["c_to_d"]
            ),
            "valid_d_to_c": (
                valid_metrics["d_to_c"]
            ),
            "valid_cd_total": (
                valid_metrics["cd_total"]
            ),
            "best_macro_f1": (
                best_macro_f1
            ),
            "best_epoch": (
                best_epoch
            ),
            "improved": (
                improved
            ),
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "epoch_seconds": (
                epoch_seconds
            ),
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        print()

        print(
            f"epoch {epoch}/{config.epochs}"
        )

        print(
            "learning rate: "
            f"{current_learning_rate:.8f}"
        )

        print(
            "train loss: "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            "train accuracy: "
            f"{train_metrics['accuracy']:.6f}"
        )

        print(
            "sampled domains: "
            f"TN={train_metrics['tn_samples']}, "
            f"VinDr={train_metrics['vindr_samples']}"
        )

        print(
            "valid loss: "
            f"{valid_metrics['loss']:.6f}"
        )

        print(
            format_metrics(
                valid_metrics
            )
        )

        print(
            f"best epoch: {best_epoch}"
        )

        print(
            "best macro_f1: "
            f"{best_macro_f1:.6f}"
        )

        print(
            "epochs without improvement: "
            f"{epochs_without_improvement}"
        )

        print(
            "epoch seconds: "
            f"{epoch_seconds:.2f}"
        )

        scheduler.step()

        if (
            epochs_without_improvement
            >= config.patience
        ):
            print()

            print(
                "early stopping at epoch "
                f"{epoch}"
            )

            break

    print()

    print(
        "training completed. "
        f"best epoch: {best_epoch}, "
        "best valid macro_f1: "
        f"{best_macro_f1:.6f}"
    )

    print(
        "best checkpoint: "
        f"{best_model_path}"
    )


if __name__ == "__main__":
    main()