"""Self-supervised (SimCLR) pretraining of the backbone on unlabeled mammograms.

Domain-specific pretraining: contrastive-learn the backbone on the mammogram
views themselves (no density labels), then load the weights into the classifier
via `train_phaseG_mixed_loss.py --init-weights <ckpt>` and fine-tune.

By default SSL uses only the TRAIN+VALID view images (keeps the test set unseen).
On a small corpus this may not beat ImageNet init (see literature); the value
grows with more unlabeled data (e.g. VinDr).

Run (from repo root):
    python src/ssl_pretrain.py --backbone densenet121 --gpu 3 --epochs 200
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="3")
    p.add_argument("--backbone", type=str, default="densenet121")
    p.add_argument("--split-csv", type=str, default="data/splits/all_splits_with_paths.csv")
    p.add_argument("--splits", type=str, default="train,valid",
                   help="Comma-sep splits whose images feed SSL (test kept unseen).")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs_ssl/ssl_backbone.pt")
    return p.parse_args()


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from constants import VIEW_NAMES
from data import standardize_split_df
from models import build_backbone


class TwoCropDataset(Dataset):
    """Return two independently-augmented crops of each mammogram view."""

    def __init__(self, paths, img_size):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.08, 0.08))], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.tf(img)


class SimCLRNet(nn.Module):
    def __init__(self, backbone, proj_dim=128):
        super().__init__()
        self.features, feat_dim, self.needs_relu = build_backbone(backbone)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        f = self.features(x)
        if self.needs_relu:
            f = F.relu(f, inplace=True)
        z = self.proj(f.mean(dim=(2, 3)))
        return F.normalize(z, dim=1)


def nt_xent(z, temp):
    """SimCLR NT-Xent. z: [2N, d] L2-normalized; z[i] & z[i+N] are positives."""
    n2 = z.shape[0]
    n = n2 // 2
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, n2), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = standardize_split_df(args.split_csv, "TN")
    keep = [s.strip() for s in args.splits.split(",")]
    sub = df[df["_split"].isin(keep)]
    paths = [str(sub.iloc[i][f"{v}_path_final"]) for i in range(len(sub)) for v in VIEW_NAMES]
    print(f"SSL images: {len(paths)} ({len(sub)} exams x 4 views), splits={keep}", flush=True)

    loader = DataLoader(
        TwoCropDataset(paths, args.img_size), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = SimCLRNet(args.backbone, args.proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"Backbone: {args.backbone} | epochs {args.epochs} | batch {args.batch_size}", flush=True)
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for a, b in loader:
            x = torch.cat([a, b], dim=0).to(device, non_blocking=True)
            z = model(x)
            loss = nt_xent(z, args.temp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[SSL] epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f} lr={opt.param_groups[0]['lr']:.2e}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": args.backbone, "features_state_dict": model.features.state_dict()}, out)
    print(f"Saved SSL backbone -> {out}", flush=True)


if __name__ == "__main__":
    main()
