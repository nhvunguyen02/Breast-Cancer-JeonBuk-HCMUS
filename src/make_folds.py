"""Generate stratified k-fold split CSVs for cross-validation.

For each fold i: test = fold i; from the rest, a stratified slice is held out as
valid (for early stopping), the remainder is train. Writes data/splits/foldK_i.csv
with the standard columns (ID, Labels, split, <view>_path...).

Run:
    python src/make_folds.py --k 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", default="data/splits/all_splits_with_paths.csv")
    p.add_argument("--out-dir", default="data/splits")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--valid-frac", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.in_csv).reset_index(drop=True)
    y = df["Labels"].values

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    out_dir = Path(args.out_dir)

    for i, (rest_idx, test_idx) in enumerate(skf.split(df, y)):
        rest = df.iloc[rest_idx].copy()
        tr_idx, va_idx = train_test_split(
            np.arange(len(rest)), test_size=args.valid_frac,
            stratify=rest["Labels"].values, random_state=args.seed,
        )
        split = np.empty(len(df), dtype=object)
        split[test_idx] = "test"
        split[rest_idx[tr_idx]] = "train"
        split[rest_idx[va_idx]] = "valid"

        out = df.copy()
        out["split"] = split
        path = out_dir / f"fold{args.k}_{i}.csv"
        out.to_csv(path, index=False)
        vc = out["split"].value_counts().to_dict()
        print(f"fold {i}: {vc} -> {path}", flush=True)


if __name__ == "__main__":
    main()
