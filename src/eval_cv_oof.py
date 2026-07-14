"""Aggregate out-of-fold predictions from k-fold CV into one honest estimate.

For each fold, load its trained model and predict (with TTA) on that fold's own
TEST split -- exams the model never trained on. Concatenating across folds gives
one held-out prediction for every exam in the dataset.

Run:
    python src/eval_cv_oof.py --k 5 --cv-dir outputs_cv
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

from data import standardize_split_df, MultiViewDataset
from constants import CLASS_NAMES
from eval_ensemble import find_ckpt, load_model, probs_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cv-dir", default="outputs_cv")
    ap.add_argument("--split-csv", default="data/splits/fold5_{i}.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_true, all_prob = [], []
    for i in range(args.k):
        ckpt = find_ckpt(os.path.join(args.cv_dir, f"fold_{i}"))
        assert ckpt, f"no checkpoint for fold {i}"
        model, a = load_model(ckpt, device)
        df = standardize_split_df(args.split_csv.format(i=i), "TN")
        test = df[df["_split"].isin(["test", "testing"])].copy()
        y = np.array(test["label_idx"].astype(int))
        loader = DataLoader(MultiViewDataset(test, img_size=a.get("img_size", 224), train=False,
                                             preprocess="none"), batch_size=16, num_workers=8)
        p = probs_for(model, loader, device, tta=True)
        all_true.append(y); all_prob.append(p)
        print(f"fold {i}: {len(y)} held-out exams, acc {accuracy_score(y, p.argmax(1)):.3f}", flush=True)

    y = np.concatenate(all_true); probs = np.concatenate(all_prob); pred = probs.argmax(1)
    yb = label_binarize(y, classes=[0, 1, 2, 3])
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])

    print(f"\n===== 5-FOLD OUT-OF-FOLD ({len(y)} exams, all held-out) =====")
    print(f"  accuracy    : {accuracy_score(y, pred):.4f}")
    print(f"  balanced_acc: {balanced_accuracy_score(y, pred):.4f}")
    print(f"  macro_f1    : {f1_score(y, pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted_f1 : {f1_score(y, pred, average='weighted', zero_division=0):.4f}")
    print(f"  macro_auc   : {roc_auc_score(yb, probs, average='macro', multi_class='ovr'):.4f}")
    print("\n  Confusion (rows true A/B/C/D):")
    print("         pA   pB   pC   pD | tot  correct")
    for i, c in enumerate(CLASS_NAMES):
        tot = cm[i].sum()
        print(f"    {c}   {cm[i][0]:4d} {cm[i][1]:4d} {cm[i][2]:4d} {cm[i][3]:4d} | {tot:3d}  {cm[i][i]:3d} ({cm[i][i]/max(1,tot)*100:.0f}%)")
    print(f"  TOTAL correct: {np.trace(cm)}/{len(y)}")


if __name__ == "__main__":
    main()
