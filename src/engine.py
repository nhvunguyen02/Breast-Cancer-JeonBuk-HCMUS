"""Train/eval loops and test-time metric computation."""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from constants import CLASS_NAMES


def attention_outside_loss(attn, mask, eps=1e-6):
    """Fraction of the per-image attention energy that falls OUTSIDE the breast
    mask. attn, mask: [N, h, w]. Minimizing this pushes activations into the
    breast (background/edge/pectoral suppression)."""
    a = attn.flatten(1)
    a = a / (a.sum(dim=1, keepdim=True) + eps)
    outside = (a * (1.0 - mask.flatten(1))).sum(dim=1)
    return outside.mean()


def run_one_epoch(model, loader, criterion, optimizer, device, train=True, attn_weight=0.0):
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
            if attn_weight > 0:
                logits, attn, mask = model(x, return_attn=True)
                loss = criterion(logits, y) + attn_weight * attention_outside_loss(attn, mask)
            else:
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
    all_prob = []

    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)

            losses.append(loss.item())
            all_true.extend(y.detach().cpu().numpy().tolist())
            all_pred.extend(pred.detach().cpu().numpy().tolist())
            all_prob.append(prob.detach().cpu().numpy())

    elapsed = time.time() - start

    all_prob = np.concatenate(all_prob, axis=0) if all_prob else np.zeros((0, len(CLASS_NAMES)))

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

    # AUC (one-vs-rest, needs the softmax probabilities). Guarded because a
    # class absent from y_true makes roc_auc_score raise.
    y_true_arr = np.asarray(all_true)
    y_bin = label_binarize(y_true_arr, classes=list(range(len(CLASS_NAMES))))
    for avg in ("macro", "weighted"):
        try:
            metrics[f"{avg}_auc"] = float(
                roc_auc_score(y_bin, all_prob, average=avg, multi_class="ovr")
            )
        except ValueError:
            metrics[f"{avg}_auc"] = float("nan")

    report = classification_report(
        all_true,
        all_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    # Per-class accuracy = recall từng class
    per_class_rows = []

    for i, class_name in enumerate(CLASS_NAMES):
        support = int(cm[i, :].sum())
        correct = int(cm[i, i])
        acc = correct / support if support > 0 else 0.0

        try:
            class_auc = float(roc_auc_score(y_bin[:, i], all_prob[:, i]))
        except (ValueError, IndexError):
            class_auc = float("nan")

        metrics[f"{class_name}_correct"] = correct
        metrics[f"{class_name}_support"] = support
        metrics[f"{class_name}_acc"] = acc
        metrics[f"{class_name}_auc"] = class_auc

        per_class_rows.append({
            "class": class_name,
            "correct": correct,
            "support": support,
            "accuracy": acc,
            "auc": class_auc,
        })

    per_class_df = pd.DataFrame(per_class_rows)

    return metrics, report, cm_df, per_class_df
