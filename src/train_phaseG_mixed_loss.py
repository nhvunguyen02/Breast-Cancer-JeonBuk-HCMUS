import os
import argparse
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tn-split-csv",
        type=str,
        default="/media/hongcat/HONGCAT/BC/code/1dieuchoae/outputs_phaseA/splits_seed42/all_splits_with_paths.csv",
    )
    parser.add_argument(
        "--vindr-split-csv",
        type=str,
        default="/media/hongcat/HONGCAT/BC/code/1dieuchoae/outputs_phaseA/phaseE_vindr/vindr_png_4view_density_split_seed42_clean.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/media/hongcat/HONGCAT/BC/code/1dieuchoae/outputs_phaseA",
    )

    parser.add_argument(
        "--tn-domain-ratio",
        type=float,
        default=0.5,
        help="Sampling mass for TN train. 0.5 means TN and VinDr appear equally often per epoch.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="cb_focal",
        choices=["ce", "focal", "cb_focal"],
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--cb-beta",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()


args = parse_args()

# set before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models


LABEL2IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX2LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
VIEW_NAMES = ["left_cc", "left_mlo", "right_cc", "right_mlo"]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def standardize_split_df(csv_path, source_name):
    df = pd.read_csv(csv_path)

    split_col = find_col(df, ["split", "set", "subset"])
    label_col = find_col(df, ["Labels", "label", "density", "breast_density", "breast density"])
    id_col = find_col(df, ["ID", "id", "case_id", "exam_id", "study_id", "patient_id"])

    if split_col is None:
        raise ValueError(f"Missing split column in {csv_path}. Columns={list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Missing label column in {csv_path}. Columns={list(df.columns)}")
    if id_col is None:
        raise ValueError(f"Missing ID column in {csv_path}. Columns={list(df.columns)}")

    out = df.copy()
    out["_split"] = out[split_col].astype(str).str.lower().str.strip()
    out["_id"] = out[id_col].astype(str).str.strip()
    out["Labels"] = out[label_col].astype(str).str.strip()
    out["label_idx"] = out["Labels"].map(LABEL2IDX)

    if out["label_idx"].isna().any():
        bad = out[out["label_idx"].isna()]["Labels"].unique()
        raise ValueError(f"Unknown labels in {csv_path}: {bad}")

    for view in VIEW_NAMES:
        path_col = find_col(out, [f"{view}_path", view, f"path_{view}"])
        if path_col is None:
            raise ValueError(f"Missing {view}_path column in {csv_path}. Columns={list(out.columns)}")
        out[f"{view}_path_final"] = out[path_col].astype(str)

    out["domain"] = source_name

    # Check existence quickly
    missing = []
    for _, row in out.iterrows():
        for view in VIEW_NAMES:
            p = Path(row[f"{view}_path_final"])
            if not p.exists():
                missing.append(str(p))
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break

    if missing:
        raise FileNotFoundError("Missing image paths, examples:\n" + "\n".join(missing))

    keep_cols = (
        ["_id", "_split", "Labels", "label_idx", "domain"]
        + [f"{v}_path_final" for v in VIEW_NAMES]
    )
    return out[keep_cols].copy()


class MultiViewDataset(Dataset):
    def __init__(self, df, img_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.eval_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = torch.tensor(int(row["label_idx"]), dtype=torch.long)

        imgs = []
        for view in VIEW_NAMES:
            p = row[f"{view}_path_final"]
            img = Image.open(p).convert("RGB")
            if self.train:
                img = self.train_tf(img)
            else:
                img = self.eval_tf(img)
            imgs.append(img)

        x = torch.stack(imgs, dim=0)
        return x, y


class DenseNet121MeanFusion(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: [B, 4, 3, H, W]
        b, v, c, h, w = x.shape
        x = x.view(b * v, c, h, w)
        logits = self.backbone(x)
        logits = logits.view(b, v, -1)
        return logits.mean(dim=1)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_model_size_mb(model, path):
    torch.save(model.state_dict(), path)
    mb = Path(path).stat().st_size / (1024 ** 2)
    try:
        Path(path).unlink()
    except Exception:
        pass
    return mb


def make_domain_balanced_sampler(train_df, tn_ratio=0.5):
    domains = train_df["domain"].tolist()
    n_tn = sum(d == "TN" for d in domains)
    n_vindr = sum(d == "VinDr" for d in domains)

    if n_tn == 0 or n_vindr == 0:
        raise ValueError(f"Invalid domain counts: TN={n_tn}, VinDr={n_vindr}")

    vindr_ratio = 1.0 - tn_ratio

    weights = []
    for d in domains:
        if d == "TN":
            weights.append(tn_ratio / n_tn)
        else:
            weights.append(vindr_ratio / n_vindr)

    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_df),
        replacement=True,
    )
    return sampler


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        loss = -((1.0 - target_probs) ** self.gamma) * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        return loss.mean()


def run_one_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    losses = []
    all_true = []
    all_pred = []

    start = time.time()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        pred = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    elapsed = time.time() - start

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "elapsed_sec": elapsed,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def evaluate_test(model, loader, criterion, device):
    model.eval()

    losses = []
    all_true = []
    all_pred = []

    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            pred = torch.argmax(logits, dim=1)

            losses.append(loss.item())
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())

    elapsed = time.time() - start

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "macro_precision": precision_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(all_true, all_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(all_true, all_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(all_true, all_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "elapsed_sec": elapsed,
        "sec_per_exam": elapsed / max(1, len(all_true)),
    }

    report = classification_report(
        all_true,
        all_pred,
        target_names=["A", "B", "C", "D"],
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"])

    # Per-class accuracy = recall từng class
    class_names = ["A", "B", "C", "D"]
    per_class_rows = []

    for i, class_name in enumerate(class_names):
        support = int(cm[i, :].sum())
        correct = int(cm[i, i])
        acc = correct / support if support > 0 else 0.0

        metrics[f"{class_name}_correct"] = correct
        metrics[f"{class_name}_support"] = support
        metrics[f"{class_name}_acc"] = acc

        per_class_rows.append({
            "class": class_name,
            "correct": correct,
            "support": support,
            "accuracy": acc,
        })

    per_class_df = pd.DataFrame(per_class_rows)

    return metrics, report, cm_df, per_class_df


def append_benchmark(row, benchmark_path):
    benchmark_path = Path(benchmark_path)
    if benchmark_path.exists():
        old = pd.read_csv(benchmark_path)
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])
    new.to_csv(benchmark_path, index=False)
    return new


def main():
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    loss_tag = args.loss_type
    if args.loss_type == "focal":
        loss_tag = f"{args.loss_type}_gamma{args.focal_gamma}"
    elif args.loss_type == "cb_focal":
        loss_tag = f"{args.loss_type}_gamma{args.focal_gamma}_beta{args.cb_beta}"

    phase_dir = out_dir / "phaseG_mixed_loss" / f"densenet121_mean_tnratio{args.tn_domain_ratio}_{loss_tag}_seed{args.seed}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    print("Phase G Mixed TN + VinDr Loss Training", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"gpu physical: {args.gpu}", flush=True)
    print(f"tn_domain_ratio: {args.tn_domain_ratio}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch_size: {args.batch_size}", flush=True)
    print(f"Num_workers: {args.num_workers}", flush=True)

    tn_df = standardize_split_df(args.tn_split_csv, "TN")
    vindr_df = standardize_split_df(args.vindr_split_csv, "VinDr")

    tn_train = tn_df[tn_df["_split"].isin(["train", "training"])].copy()
    tn_valid = tn_df[tn_df["_split"].isin(["valid", "val", "validation"])].copy()
    tn_test = tn_df[tn_df["_split"].isin(["test", "testing"])].copy()

    vindr_train = vindr_df[vindr_df["_split"].isin(["train", "training"])].copy()

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

    train_ds = MultiViewDataset(mixed_train, img_size=args.img_size, train=True)
    valid_ds = MultiViewDataset(tn_valid, img_size=args.img_size, train=False)
    test_ds = MultiViewDataset(tn_test, img_size=args.img_size, train=False)

    sampler = make_domain_balanced_sampler(mixed_train, tn_ratio=args.tn_domain_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
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

    ce_weights_np = tn_counts.sum() / (len(tn_counts) * tn_counts)
    ce_weights = torch.tensor(ce_weights_np, dtype=torch.float32).to(device)

    effective_num = 1.0 - np.power(args.cb_beta, tn_counts)
    cb_weights_np = (1.0 - args.cb_beta) / effective_num
    cb_weights_np = cb_weights_np / cb_weights_np.mean()
    cb_weights = torch.tensor(cb_weights_np, dtype=torch.float32).to(device)

    print(f"TN train counts: {tn_counts}", flush=True)
    print(f"CE weights from TN train: {ce_weights}", flush=True)
    print(f"CB weights from TN train beta={args.cb_beta}: {cb_weights}", flush=True)
    print(f"Loss type: {args.loss_type}, focal_gamma={args.focal_gamma}", flush=True)

    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss(weight=ce_weights)
    elif args.loss_type == "focal":
        criterion = FocalLoss(alpha=ce_weights, gamma=args.focal_gamma)
    elif args.loss_type == "cb_focal":
        criterion = FocalLoss(alpha=cb_weights, gamma=args.focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
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
        "preprocessing": "nocrop_noCLAHE_resize_imagenet_norm",
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
