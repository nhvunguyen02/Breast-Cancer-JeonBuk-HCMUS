#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SAM2UNet import SAM2UNet


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TN-Mammo SAM2-UNet training with locked fold-0 validation."
        )
    )

    parser.add_argument("--train-image-dir", type=Path, required=True)
    parser.add_argument("--train-mask-dir", type=Path, required=True)
    parser.add_argument("--valid-image-dir", type=Path, required=True)
    parser.add_argument("--valid-mask-dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Fixed in advance. TN-Mammo is not used to tune this threshold.
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--smoke-test", action="store_true")

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)



def letterbox_pair(
    image: Image.Image,
    mask: Image.Image,
    size: int,
) -> tuple[Image.Image, Image.Image]:
    """Resize without anatomical distortion, then pad with black."""
    width, height = image.size

    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image_resized = image.resize(
        (new_width, new_height),
        resample=Image.Resampling.BILINEAR,
    )

    mask_resized = mask.resize(
        (new_width, new_height),
        resample=Image.Resampling.NEAREST,
    )

    image_canvas = Image.new(
        "RGB",
        (size, size),
        color=(0, 0, 0),
    )

    mask_canvas = Image.new(
        "L",
        (size, size),
        color=0,
    )

    left = (size - new_width) // 2
    top = (size - new_height) // 2

    image_canvas.paste(image_resized, (left, top))
    mask_canvas.paste(mask_resized, (left, top))

    return image_canvas, mask_canvas


class MammogramMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int,
        train: bool,
        limit: int | None = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.train = train

        self.images = sorted(
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if limit is not None:
            self.images = self.images[:limit]

        if not self.images:
            raise ValueError(f"No images found in {image_dir}")

        missing_masks = [
            path.name
            for path in self.images
            if not (mask_dir / path.name).is_file()
        ]

        if missing_masks:
            raise FileNotFoundError(
                f"Missing masks for {len(missing_masks)} images: "
                f"{missing_masks[:10]}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.images[index]
        mask_path = self.mask_dir / image_path.name

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"{image.size} vs {mask.size}"
            )

        # Horizontal flip only. Vertical flip is anatomically implausible.
        if self.train and random.random() < 0.5:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        image, mask = letterbox_pair(
            image,
            mask,
            self.image_size,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).permute(2, 0, 1)

        image_tensor = (
            image_tensor - IMAGENET_MEAN
        ) / IMAGENET_STD

        mask_array = (
            np.asarray(mask, dtype=np.uint8) > 0
        ).astype(np.float32)

        mask_tensor = torch.from_numpy(
            mask_array
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.name,
        }


def structure_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    weight = 1 + 5 * torch.abs(
        F.avg_pool2d(
            target,
            kernel_size=31,
            stride=1,
            padding=15,
        ) - target
    )

    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )

    weighted_bce = (
        weight * weighted_bce
    ).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    probability = torch.sigmoid(logits)

    intersection = (
        (probability * target) * weight
    ).sum(dim=(2, 3))

    union = (
        (probability + target) * weight
    ).sum(dim=(2, 3))

    weighted_iou = 1 - (
        intersection + 1
    ) / (
        union - intersection + 1
    )

    return (weighted_bce + weighted_iou).mean()


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prediction = (
        torch.sigmoid(logits) >= threshold
    ).float()

    target = (target >= 0.5).float()

    dimensions = (1, 2, 3)
    eps = 1e-7

    true_positive = (
        prediction * target
    ).sum(dimensions)

    false_positive = (
        prediction * (1 - target)
    ).sum(dimensions)

    false_negative = (
        (1 - prediction) * target
    ).sum(dimensions)

    dice = (
        2 * true_positive + eps
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + eps
    )

    iou = (
        true_positive + eps
    ) / (
        true_positive
        + false_positive
        + false_negative
        + eps
    )

    precision = (
        true_positive + eps
    ) / (
        true_positive
        + false_positive
        + eps
    )

    recall = (
        true_positive + eps
    ) / (
        true_positive
        + false_negative
        + eps
    )

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=enabled
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train(train)

    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }

    samples = 0
    start = time.time()

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["mask"].to(
            device,
            non_blocking=True,
        )

        batch_size = int(images.shape[0])

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output, auxiliary_1, auxiliary_2 = model(images)

                loss = (
                    structure_loss(output, targets)
                    + structure_loss(auxiliary_1, targets)
                    + structure_loss(auxiliary_2, targets)
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()

        metrics = segmentation_metrics(
            output.detach(),
            targets,
            threshold,
        )

        totals["loss"] += (
            float(loss.detach().item()) * batch_size
        )

        for key, value in metrics.items():
            totals[key] += value * batch_size

        samples += batch_size

        if train and (
            batch_index == 1
            or batch_index % 25 == 0
            or batch_index == len(loader)
        ):
            print(
                f"[TRAIN BATCH] {batch_index}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"dice={metrics['dice']:.4f}",
                flush=True,
            )

    if samples == 0:
        raise RuntimeError("Epoch processed zero samples")

    results = {
        key: value / samples
        for key, value in totals.items()
    }

    results["seconds"] = time.time() - start
    return results


def save_json(
    path: Path,
    data: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for SAM2-UNet training"
        )

    if not args.pretrained_checkpoint.is_file():
        raise FileNotFoundError(
            args.pretrained_checkpoint
        )

    if args.smoke_test:
        args.epochs = 1
        args.patience = 1
        args.num_workers = 0
        args.resume = False

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cuda:0")
    amp_enabled = bool(
        args.amp and device.type == "cuda"
    )

    train_dataset = MammogramMaskDataset(
        args.train_image_dir,
        args.train_mask_dir,
        args.image_size,
        train=True,
        limit=8 if args.smoke_test else None,
    )

    valid_dataset = MammogramMaskDataset(
        args.valid_image_dir,
        args.valid_mask_dir,
        args.image_size,
        train=False,
        limit=4 if args.smoke_test else None,
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        train=True,
        seed=args.seed,
    )

    valid_loader = make_loader(
        valid_dataset,
        args.batch_size,
        args.num_workers,
        train=False,
        seed=args.seed,
    )

    print(
        f"[DATA] train={len(train_dataset)} "
        f"valid={len(valid_dataset)} "
        f"batch={args.batch_size} "
        f"image_size={args.image_size}",
        flush=True,
    )

    print(
        f"[GPU] {torch.cuda.get_device_name(0)} "
        f"torch={torch.__version__} "
        f"cuda={torch.version.cuda}",
        flush=True,
    )

    model = SAM2UNet(
        str(args.pretrained_checkpoint)
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[MODEL] total_parameters={total_parameters:,} "
        f"trainable_parameters={trainable_parameters:,}",
        flush=True,
    )

    optimizer = AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=1e-6,
    )

    scaler = make_scaler(amp_enabled)

    last_checkpoint = (
        args.output_dir / "last_checkpoint.pt"
    )
    best_checkpoint = (
        args.output_dir / "best_model.pt"
    )
    best_state_dict = (
        args.output_dir / "best_model_state_dict.pth"
    )
    history_path = (
        args.output_dir / "history.csv"
    )
    config_path = (
        args.output_dir / "run_config.json"
    )
    summary_path = (
        args.output_dir / "training_summary.json"
    )

    config = vars(args).copy()
    config.update(
        {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "fixed_threshold": args.threshold,
            "model_selection_metric": "validation_dice",
        }
    )

    save_json(
        config_path,
        {
            key: str(value)
            if isinstance(value, Path)
            else value
            for key, value in config.items()
        },
    )

    start_epoch = 0
    best_val_dice = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(
            last_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_epoch = int(checkpoint["epoch"])
        best_val_dice = float(
            checkpoint["best_val_dice"]
        )
        best_epoch = int(
            checkpoint["best_epoch"]
        )
        epochs_without_improvement = int(
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        print(
            f"[RESUME] start_epoch={start_epoch} "
            f"best_epoch={best_epoch} "
            f"best_val_dice={best_val_dice:.6f}",
            flush=True,
        )

    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_precision",
        "train_recall",
        "valid_loss",
        "valid_dice",
        "valid_iou",
        "valid_precision",
        "valid_recall",
        "train_seconds",
        "valid_seconds",
    ]

    write_header = (
        not history_path.exists()
        or start_epoch == 0
    )

    if start_epoch == 0 and history_path.exists():
        history_path.unlink()

    completed_epoch = start_epoch
    stopped_early = False

    for epoch_index in range(
        start_epoch,
        args.epochs,
    ):
        epoch = epoch_index + 1
        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\n===== EPOCH {epoch}/{args.epochs} "
            f"lr={current_lr:.8f} =====",
            flush=True,
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            args.threshold,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        valid_metrics = run_epoch(
            model,
            valid_loader,
            device,
            args.threshold,
            train=False,
            optimizer=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        scheduler.step()
        completed_epoch = epoch

        row = {
            "epoch": epoch,
            "lr": current_lr,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            **{
                f"valid_{key}": value
                for key, value in valid_metrics.items()
            },
        }

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            if write_header:
                writer.writeheader()
                write_header = False

            writer.writerow(
                {
                    key: row[key]
                    for key in fieldnames
                }
            )

        improved = (
            valid_metrics["dice"]
            > best_val_dice + 1e-6
        )

        if improved:
            best_val_dice = valid_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "best_epoch": best_epoch,
            "epochs_without_improvement": (
                epochs_without_improvement
            ),
            "args": {
                key: str(value)
                if isinstance(value, Path)
                else value
                for key, value in vars(args).items()
            },
            "validation_metrics": valid_metrics,
        }

        torch.save(
            checkpoint,
            last_checkpoint,
        )

        if improved:
            torch.save(
                checkpoint,
                best_checkpoint,
            )
            torch.save(
                model.state_dict(),
                best_state_dict,
            )

            print(
                f"[BEST] epoch={epoch} "
                f"val_dice={best_val_dice:.6f} "
                f"saved={best_checkpoint}",
                flush=True,
            )

        print(
            "[EPOCH SUMMARY] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_dice={valid_metrics['dice']:.4f} "
            f"valid_iou={valid_metrics['iou']:.4f} "
            f"valid_precision="
            f"{valid_metrics['precision']:.4f} "
            f"valid_recall={valid_metrics['recall']:.4f} "
            f"patience="
            f"{epochs_without_improvement}/{args.patience}",
            flush=True,
        )

        if (
            not args.smoke_test
            and epochs_without_improvement >= args.patience
        ):
            stopped_early = True
            print(
                "[EARLY STOP] no validation-Dice "
                f"improvement for {args.patience} epochs",
                flush=True,
            )
            break

    summary = {
        "status": "PASS",
        "smoke_test": args.smoke_test,
        "completed_epoch": completed_epoch,
        "requested_epochs": args.epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_dice": best_val_dice,
        "best_checkpoint": str(best_checkpoint),
        "best_state_dict": str(best_state_dict),
        "last_checkpoint": str(last_checkpoint),
        "history_csv": str(history_path),
        "fixed_threshold": args.threshold,
        "model_selection_metric": "validation_dice",
    }

    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
