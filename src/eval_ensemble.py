"""Ensemble + TTA evaluation of trained checkpoints on the TN test set.

Averages softmax probabilities across (a) horizontal-flip test-time augmentation
and (b) several seed checkpoints, then reports the pooled metrics. Each model is
rebuilt from the `args` stored in its checkpoint, so mixed configs can be pooled.

Run:
    python src/eval_ensemble.py <ckpt_dir_or_pt> [<ckpt_dir_or_pt> ...]
"""

import os
import sys
import glob

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


def find_ckpt(p):
    if p.endswith(".pt"):
        return p
    hits = glob.glob(os.path.join(p, "**", "best_model.pt"), recursive=True)
    return hits[0] if hits else None


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    m = MultiViewModel(num_classes=a.get("num_classes", 4), backbone=a.get("backbone", "densenet121"),
                       masked_pool=a.get("masked_pool", False),
                       fusion=a.get("fusion", "mean"), ordinal=a.get("ordinal", False)).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m, a


@torch.no_grad()
def probs_for(model, loader, device, tta=True):
    out = []
    for x, _ in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1)
        if tta:
            p = p + F.softmax(model(torch.flip(x, dims=[-1])), dim=1)  # horizontal flip
            p = p / 2.0
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def report(name, y, probs):
    pred = probs.argmax(1)
    yb = label_binarize(y, classes=[0, 1, 2, 3])
    print(f"\n=== {name} ===")
    print(f"  accuracy    : {accuracy_score(y, pred):.4f}")
    print(f"  balanced_acc: {balanced_accuracy_score(y, pred):.4f}")
    print(f"  macro_f1    : {f1_score(y, pred, average='macro', zero_division=0):.4f}")
    print(f"  weighted_f1 : {f1_score(y, pred, average='weighted', zero_division=0):.4f}")
    print(f"  macro_auc   : {roc_auc_score(yb, probs, average='macro', multi_class='ovr'):.4f}")
    pc = {c: f"{(confusion_matrix(y,pred,labels=[0,1,2,3])[i,i]/max(1,(np.array(y)==i).sum())):.2f}"
          for i, c in enumerate(CLASS_NAMES)}
    print(f"  per-class   : {pc}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = [find_ckpt(p) for p in sys.argv[1:]]
    ckpts = [c for c in ckpts if c]
    assert ckpts, "no checkpoints found"

    # dataset from the first model's split csv
    _, a0 = load_model(ckpts[0], device)
    df = standardize_split_df(a0["tn_split_csv"], "TN")
    test = df[df["_split"].isin(["test", "testing"])].copy()
    y = test["label_idx"].astype(int).tolist()
    loader = DataLoader(MultiViewDataset(test, img_size=a0.get("img_size", 224), train=False,
                                         preprocess=a0.get("preprocess", "none")),
                        batch_size=16, num_workers=8)

    all_probs = []
    for c in ckpts:
        m, a = load_model(c, device)
        p_notta = probs_for(m, loader, device, tta=False)
        p_tta = probs_for(m, loader, device, tta=True)
        tag = os.path.basename(os.path.dirname(c))[-28:]
        report(f"single [{tag}] (no TTA)", y, p_notta)
        report(f"single [{tag}] (+TTA)", y, p_tta)
        all_probs.append(p_tta)

    ens = np.mean(all_probs, axis=0)
    report(f"ENSEMBLE {len(ckpts)} models + TTA", y, ens)


if __name__ == "__main__":
    main()
