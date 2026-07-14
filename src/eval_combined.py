"""Combined eval: main 4-class ensemble refined by the binary C-vs-D expert.

The main ensemble (with TTA) gives 4-class probabilities. Whenever the main
prediction is C or D, the C-vs-D specialist ensemble decides between them. This
targets the dominant C<->D confusion without touching A/B.

Run:
    python src/eval_combined.py --main <dir...> --cd <expert.pt...>
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

from data import standardize_split_df, MultiViewDataset
from models import MultiViewModel
from constants import CLASS_NAMES
from eval_ensemble import find_ckpt, load_model, probs_for


def report(name, y, pred, probs=None):
    print(f"\n=== {name} ===")
    print(f"  accuracy    : {accuracy_score(y, pred):.4f}")
    print(f"  balanced_acc: {balanced_accuracy_score(y, pred):.4f}")
    print(f"  macro_f1    : {f1_score(y, pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted_f1 : {f1_score(y, pred, average='weighted', zero_division=0):.4f}")
    if probs is not None:
        yb = label_binarize(y, classes=[0, 1, 2, 3])
        print(f"  macro_auc   : {roc_auc_score(yb, probs, average='macro', multi_class='ovr'):.4f}")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2, 3])
    pc = {c: f"{cm[i, i] / max(1, (np.array(y) == i).sum()):.2f}" for i, c in enumerate(CLASS_NAMES)}
    print(f"  per-class   : {pc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", nargs="+", required=True)
    ap.add_argument("--cd", nargs="+", required=True)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main_ckpts = [find_ckpt(p) for p in a.main]
    main_ckpts = [c for c in main_ckpts if c]
    cd_ckpts = [find_ckpt(p) for p in a.cd]
    cd_ckpts = [c for c in cd_ckpts if c]

    _, a0 = load_model(main_ckpts[0], device)
    df = standardize_split_df(a0["tn_split_csv"], "TN")
    test = df[df["_split"].isin(["test", "testing"])].copy()
    y = np.array(test["label_idx"].astype(int).tolist())
    loader = DataLoader(MultiViewDataset(test, img_size=a0.get("img_size", 224), train=False,
                                         preprocess=a0.get("preprocess", "none")),
                        batch_size=16, num_workers=8)

    # main 4-class ensemble + TTA
    main_probs = np.mean([probs_for(load_model(c, device)[0], loader, device, tta=True)
                          for c in main_ckpts], axis=0)
    main_pred = main_probs.argmax(1)
    report("MAIN ensemble + TTA (baseline)", y, main_pred, main_probs)

    # C-vs-D expert ensemble + TTA -> P(D)
    pD = []
    for c in cd_ckpts:
        m, _ = load_model(c, device)
        pD.append(probs_for(m, loader, device, tta=True)[:, 1])
    pD = np.mean(pD, axis=0)

    # refine: where main says C(2) or D(3), let the expert decide
    final = main_pred.copy()
    mask = np.isin(main_pred, [2, 3])
    final[mask] = np.where(pD[mask] >= 0.5, 3, 2)
    report("COMBINED (main + C/D expert)", y, final, main_probs)

    changed = int((final != main_pred).sum())
    print(f"\nExpert overrode {changed}/{mask.sum()} of the C/D predictions.")


if __name__ == "__main__":
    main()
