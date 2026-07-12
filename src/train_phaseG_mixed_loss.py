"""Phase G: mixed TN + VinDr training of a DenseNet121 mean-fusion model for
4-class breast density classification.

Run:
    python src/train_phaseG_mixed_loss.py --loss-type cb_focal --gpu 1
"""

import os
import sys
import json
import time
from pathlib import Path

# Make the sibling modules importable no matter the current working directory
# (so `python src/train_phaseG_mixed_loss.py` works from the repo root too).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import parse_args

args = parse_args()

# Must happen before torch is imported (transitively via the modules below).
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
import torch

from constants import LABEL2IDX  # noqa: F401  (kept for parity / downstream use)
from utils import seed_everything, count_params, save_model_size_mb, append_benchmark
from data import standardize_split_df, MultiViewDataset, make_domain_balanced_sampler
from models import DenseNet121MeanFusion
from losses import compute_class_weights, build_criterion
from engine import run_one_epoch, evaluate_test

from torch.utils.data import DataLoader


def main():
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    loss_tag = args.loss_type
    if args.loss_type == "focal":
        loss_tag = f"{args.loss_type}_gamma{args.focal_gamma}"
    elif args.loss_type == "cb_focal":
        loss_tag = f"{args.loss_type}_gamma{args.focal_gamma}_beta{args.cb_beta}"

    phase_dir = out_dir / "phaseG_mixed_loss" / f"densenet121_mean_tnratio{args.tn_domain_ratio}_{loss_tag}_pp{args.preprocess}_seed{args.seed}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print("Phase G Mixed TN + VinDr Loss Training", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"gpu physical: {args.gpu}", flush=True)
    print(f"tn_domain_ratio: {args.tn_domain_ratio}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch_size: {args.batch_size}", flush=True)
    print(f"Num_workers: {args.num_workers}", flush=True)

    tn_df = standardize_split_df(args.tn_split_csv, "TN")

    tn_train = tn_df[tn_df["_split"].isin(["train", "training"])].copy()
    tn_valid = tn_df[tn_df["_split"].isin(["valid", "val", "validation"])].copy()
    tn_test = tn_df[tn_df["_split"].isin(["test", "testing"])].copy()

    # VinDr is optional: only mix it in when the split CSV is actually present.
    use_vindr = bool(args.vindr_split_csv) and Path(args.vindr_split_csv).exists()
    if use_vindr:
        vindr_df = standardize_split_df(args.vindr_split_csv, "VinDr")
        vindr_train = vindr_df[vindr_df["_split"].isin(["train", "training"])].copy()
    else:
        print(f"VinDr split CSV not found ({args.vindr_split_csv}) -> TN-only training", flush=True)
        vindr_train = tn_train.iloc[0:0].copy()  # empty, same schema

    mixed_train = pd.concat([tn_train, vindr_train], ignore_index=True)

    print(f"TN train: {len(tn_train)}", flush=True)
    print(f"TN valid: {len(tn_valid)}", flush=True)
    print(f"TN test: {len(tn_test)}", flush=True)
    print(f"VinDr train: {len(vindr_train)}", flush=True)
    print(f"Mixed train: {len(mixed_train)}", flush=True)

    print("\nTN train labels:", tn_train["label_idx"].value_counts().sort_index().to_dict(), flush=True)
    print("TN valid labels:", tn_valid["label_idx"].value_counts().sort_index().to_dict(), flush=True)
    print("TN test labels:", tn_test["label_idx"].value_counts().sort_index().to_dict(), flush=True)
    print("VinDr train labels:", vindr_train["label_idx"].value_counts().sort_index().to_dict(), flush=True)
    print("Mixed domain counts:", mixed_train["domain"].value_counts().to_dict(), flush=True)

    train_ds = MultiViewDataset(mixed_train, img_size=args.img_size, train=True, preprocess=args.preprocess)
    valid_ds = MultiViewDataset(tn_valid, img_size=args.img_size, train=False, preprocess=args.preprocess)
    test_ds = MultiViewDataset(tn_test, img_size=args.img_size, train=False, preprocess=args.preprocess)

    # Domain-balanced sampling only makes sense with two domains; otherwise
    # fall back to plain shuffling of the TN-only training set.
    if use_vindr:
        sampler = make_domain_balanced_sampler(mixed_train, tn_ratio=args.tn_domain_ratio)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Train batches: {len(train_loader)}", flush=True)
    print(f"Valid batches: {len(valid_loader)}", flush=True)
    print(f"Test batches: {len(test_loader)}", flush=True)

    model = DenseNet121MeanFusion(num_classes=4).to(device)

    total_params, trainable_params = count_params(model)
    model_size_mb = save_model_size_mb(model, phase_dir / "temp_model_size.pt")

    print(f"Params total: {total_params:,}", flush=True)
    print(f"Params trainable: {trainable_params:,}", flush=True)
    print(f"Model size MB: {model_size_mb:.2f}", flush=True)

    # Loss weights based on TN train only, not VinDr.
    tn_counts = tn_train["label_idx"].value_counts().sort_index().values.astype(np.float32)

    ce_weights_np, cb_weights_np = compute_class_weights(tn_counts, args.cb_beta)
    ce_weights = torch.tensor(ce_weights_np, dtype=torch.float32).to(device)
    cb_weights = torch.tensor(cb_weights_np, dtype=torch.float32).to(device)

    print(f"TN train counts: {tn_counts}", flush=True)
    print(f"CE weights from TN train: {ce_weights}", flush=True)
    print(f"CB weights from TN train beta={args.cb_beta}: {cb_weights}", flush=True)
    print(f"Loss type: {args.loss_type}, focal_gamma={args.focal_gamma}", flush=True)

    criterion = build_criterion(args.loss_type, ce_weights, cb_weights, args.focal_gamma)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, args.epochs // 2),
        gamma=0.5,
    )

    best_valid_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    best_ckpt = phase_dir / "best_model.pt"

    history = []

    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"[EPOCH] {epoch}/{args.epochs}", flush=True)

        train_m = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        valid_m = run_one_epoch(model, valid_loader, criterion, optimizer, device, train=False)

        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "train_macro_f1": train_m["macro_f1"],
            "train_weighted_f1": train_m["weighted_f1"],
            "valid_loss": valid_m["loss"],
            "valid_acc": valid_m["acc"],
            "valid_macro_f1": valid_m["macro_f1"],
            "valid_weighted_f1": valid_m["weighted_f1"],
            "lr": lr_now,
            "train_elapsed_sec": train_m["elapsed_sec"],
            "valid_elapsed_sec": valid_m["elapsed_sec"],
        }
        history.append(row)

        print(
            "[METRIC] "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_acc={row['train_acc']:.4f} | "
            f"train_macro_f1={row['train_macro_f1']:.4f} | "
            f"valid_acc={row['valid_acc']:.4f} | "
            f"valid_macro_f1={row['valid_macro_f1']:.4f} | "
            f"valid_weighted_f1={row['valid_weighted_f1']:.4f} | "
            f"lr={lr_now}",
            flush=True,
        )

        improved = valid_m["macro_f1"] > (best_valid_macro_f1 + args.min_delta)

        if improved:
            best_valid_macro_f1 = valid_m["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "densenet121",
                    "fusion": "mean",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_macro_f1": best_valid_macro_f1,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print("[SAVE] Saved new best model.", flush=True)
        else:
            patience_counter += 1
            print(
                f"[EARLY] no improvement: {patience_counter}/{args.early_stop_patience} | "
                f"best_epoch={best_epoch} | best_valid_macro_f1={best_valid_macro_f1:.4f}",
                flush=True,
            )

        scheduler.step()
        pd.DataFrame(history).to_csv(phase_dir / "history.csv", index=False)

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(
                f"[EARLY] Stop at epoch {epoch}. "
                f"Best epoch={best_epoch}, best_valid_macro_f1={best_valid_macro_f1:.4f}",
                flush=True,
            )
            break

    total_train_time = time.time() - train_start
    print(f"\nTotal train time: {total_train_time:.2f} sec", flush=True)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, report, cm_df, per_class_df = evaluate_test(model, test_loader, criterion, device)

    print("\n[TEST] Metrics:", flush=True)
    for k, v in test_metrics.items():
        if isinstance(v, int):
            print(f"{k}: {v}", flush=True)
        else:
            print(f"{k}: {v:.4f}", flush=True)

    print("\n[TEST] Per-class accuracy:", flush=True)
    for _, row in per_class_df.iterrows():
        print(
            f"{row['class']}: {int(row['correct'])}/{int(row['support'])} = {row['accuracy']:.4f}",
            flush=True,
        )

    print("\n[TEST] Classification report:", flush=True)
    print(report, flush=True)

    print("\n[TEST] Confusion matrix:", flush=True)
    print(cm_df, flush=True)

    with open(phase_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(phase_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm_df.to_csv(phase_dir / "confusion_matrix.csv")
    per_class_df.to_csv(phase_dir / "per_class_accuracy.csv", index=False)

    benchmark_row = {
        "experiment": f"phaseG_mixed_TN_VinDr_densenet121_mean_tnratio{args.tn_domain_ratio}_{loss_tag}_seed{args.seed}",
        "model": "densenet121",
        "fusion": "mean_4_views",
        "input": f"JPEG_or_PNG_resize_{args.img_size}",
        "preprocessing": ("brm_cropbreast_inmaskP2P98norm_resize_imagenet_norm"
                          if args.preprocess == "brm"
                          else "nocrop_noCLAHE_resize_imagenet_norm"),
        "split": "TN_seed42_train411_valid133_test132_plus_VinDr_train_external",
        "gpu": f"physical_gpu{args.gpu}",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sampler": f"domain_balanced_tn_ratio_{args.tn_domain_ratio}",
        "loss": "weighted_CE_TN_train_weights",
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "macro_auc": test_metrics["macro_auc"],
        "weighted_auc": test_metrics["weighted_auc"],
        "params_total": total_params,
        "params_trainable": trainable_params,
        "model_size_mb": model_size_mb,
        "total_train_time_sec": total_train_time,
        "test_time_sec": test_metrics["elapsed_sec"],
        "test_sec_per_exam": test_metrics["sec_per_exam"],
        "best_valid_macro_f1": best_valid_macro_f1,
        "best_epoch": best_epoch,
        "early_stop_patience": args.early_stop_patience,
        "img_size": args.img_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "tn_domain_ratio": args.tn_domain_ratio,
        "tn_train_cases": len(tn_train),
        "vindr_train_cases": len(vindr_train),
    }

    benchmark_path = out_dir / "benchmark_results.csv"
    append_benchmark(benchmark_row, benchmark_path)

    print(f"\nOutputs: {phase_dir}", flush=True)
    print(f"Benchmark updated: {benchmark_path}", flush=True)
    print("Done - Phase G mixed TN+VinDr loss training finished.", flush=True)


if __name__ == "__main__":
    main()
