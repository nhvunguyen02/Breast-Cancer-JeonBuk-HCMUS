"""Binary C-vs-D specialist.

The 4-class model's main confusion is C (heterogeneously dense) vs D (extremely
dense). This trains a dedicated binary classifier on only the C and D exams
(C=0, D=1), which learns the C/D boundary without being distracted by A/B. At
inference it refines the main model whenever the main prediction is C or D.

Run:
    python src/train_cd_expert.py --init-weights outputs_ssl/ssl_dn121.pt --gpu 3 --seed 1
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="3")
    p.add_argument("--backbone", type=str, default="densenet121")
    p.add_argument("--tn-split-csv", type=str, default="data/splits/all_splits_with_paths.csv")
    p.add_argument("--init-weights", type=str, default="")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs_cd")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from utils import seed_everything
from data import standardize_split_df, MultiViewDataset
from models import MultiViewModel
from engine import run_one_epoch


def main():
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = standardize_split_df(args.tn_split_csv, "TN")
    df = df[df["Labels"].isin(["C", "D"])].copy()
    df["label_idx"] = df["Labels"].map({"C": 0, "D": 1})

    tr = df[df["_split"].isin(["train", "training"])].copy()
    va = df[df["_split"].isin(["valid", "val", "validation"])].copy()
    te = df[df["_split"].isin(["test", "testing"])].copy()
    print(f"C-vs-D  train {len(tr)} valid {len(va)} test {len(te)}", flush=True)
    print("train C/D:", tr["label_idx"].value_counts().sort_index().to_dict(), flush=True)

    mk = lambda d, t: DataLoader(MultiViewDataset(d, img_size=args.img_size, train=t, preprocess="none"),
                                 batch_size=args.batch_size, shuffle=t, num_workers=args.num_workers,
                                 pin_memory=True, persistent_workers=(args.num_workers > 0))
    train_loader, valid_loader, test_loader = mk(tr, True), mk(va, False), mk(te, False)

    model = MultiViewModel(num_classes=2, backbone=args.backbone, fusion="mean").to(device)
    if args.init_weights:
        ck = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.features.load_state_dict(ck["features_state_dict"], strict=False)
        print(f"Loaded SSL backbone from {args.init_weights}", flush=True)

    counts = tr["label_idx"].value_counts().sort_index().values.astype(np.float32)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1, best_epoch, patience = -1.0, 0, 0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"cd_expert_seed{args.seed}.pt")

    for epoch in range(1, args.epochs + 1):
        run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vm = run_one_epoch(model, valid_loader, criterion, optimizer, device, train=False)
        if vm["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, patience = vm["macro_f1"], epoch, 0
            torch.save({"model_state_dict": model.state_dict(),
                        "args": {**vars(args), "num_classes": 2, "fusion": "mean", "cd_expert": True}},
                       ckpt_path)
        else:
            patience += 1
        if patience >= args.early_stop_patience:
            break
    print(f"Best valid C/D macro-F1 {best_f1:.4f} @ epoch {best_epoch}", flush=True)

    # test the standalone expert
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            prob = torch.softmax(model(x.to(device)), dim=1)[:, 1]  # P(D)
            ps.extend(prob.cpu().numpy().tolist()); ys.extend(y.numpy().tolist())
    ys, ps = np.array(ys), np.array(ps)
    pred = (ps >= 0.5).astype(int)
    print(f"[CD-expert test] acc {(pred==ys).mean():.4f} | macro-F1 "
          f"{f1_score(ys,pred,average='macro',zero_division=0):.4f} | AUC {roc_auc_score(ys,ps):.4f}", flush=True)
    print("confusion (rows C,D):\n", confusion_matrix(ys, pred, labels=[0, 1]), flush=True)
    print(f"Saved -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
