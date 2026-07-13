"""Dataset, split-CSV standardization, and domain-balanced sampler."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from constants import LABEL2IDX, VIEW_NAMES

# view column -> (view code, breast side) for BRM preprocessing.
VIEW_META = {
    "left_cc": ("CC", "L"),
    "left_mlo": ("MLO", "L"),
    "right_cc": ("CC", "R"),
    "right_mlo": ("MLO", "R"),
}


def _brm_to_rgb(pil_img, view, side, remove_pectoral=False):
    """Run BRM stage0 preprocessing on a PIL image and return an RGB PIL image."""
    from preprocess import preprocess_view

    gray = np.asarray(pil_img.convert("L"), dtype=np.float32)
    out = preprocess_view(gray, view=view, side=side, remove_pectoral=remove_pectoral)

    # Keep the ORIGINAL 8-bit intensities (input JPEGs are already 0-255). Do NOT
    # contrast-stretch per image: that would blow out low-contrast fatty breasts
    # and destroy the density signal we are trying to classify.
    arr8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr8, mode="L").convert("RGB")


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
    def __init__(self, df, img_size=224, train=False, preprocess="none",
                 brm_pectoral=False, aug="basic"):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.train = train
        self.preprocess = preprocess      # "none" (raw resize) or "brm" (crop+normalize)
        self.brm_pectoral = brm_pectoral  # remove pectoral muscle in MLO views (brm only)
        self.aug = aug                    # "basic" (flip only) or "strong"

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if aug == "affine":
            # Geometry-only augmentation (no erasing, which would occlude the
            # fibroglandular tissue that defines density). Milder than 'strong'.
            self.train_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), fill=0),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                self.normalize,
            ])
        elif aug == "strong":
            # Mammography-appropriate augmentation. Geometry (affine) fills with
            # black to match the zeroed background; intensity jitter is kept MILD
            # so the density signal (relative fibroglandular brightness) survives.
            self.train_tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                        scale=(0.9, 1.1), fill=0),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                self.normalize,
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), value=0.0),
            ])
        else:  # "basic"
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
            img = Image.open(p)
            if self.preprocess == "brm":
                view_code, side = VIEW_META[view]
                img = _brm_to_rgb(img, view_code, side, remove_pectoral=self.brm_pectoral)
            else:
                img = img.convert("RGB")
            if self.train:
                img = self.train_tf(img)
            else:
                img = self.eval_tf(img)
            imgs.append(img)

        x = torch.stack(imgs, dim=0)
        return x, y


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
