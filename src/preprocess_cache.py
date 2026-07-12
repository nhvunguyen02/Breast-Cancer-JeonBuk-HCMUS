"""Precompute BRM preprocessing once and cache it to disk.

Reads a split CSV (ID, Labels, split, <view>_path columns), runs preprocess_view
on every view image, and writes the result as an 8-bit PNG under <out-dir>/<ID>/.
Emits a NEW split CSV whose <view>_path columns point at the cached PNGs, so
training can then run with --preprocess none (just resize + normalize) and read
already-preprocessed images -> no per-epoch recompute.

Run (from repo root):
    python src/preprocess_cache.py                       # pectoral OFF (default)
    python src/preprocess_cache.py --remove-pectoral     # pectoral removal ON
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from PIL import Image

from constants import VIEW_NAMES
from data import VIEW_META
from preprocess import preprocess_view


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", default="data/splits/all_splits_with_paths.csv")
    p.add_argument("--out-dir", default=None,
                   help="Cache dir (default data/cache/brm[_pec]).")
    p.add_argument("--out-csv", default=None,
                   help="New split CSV (default <out-dir>/all_splits_brm[_pec].csv).")
    p.add_argument("--remove-pectoral", action="store_true")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip in-mask p2-p98 normalization; keep original intensities (crop only).")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def _process_one(task):
    """task = (src_path, dst_path, view_code, side, remove_pectoral, normalize)."""
    src, dst, view_code, side, remove_pectoral, normalize = task
    if os.path.exists(dst):
        return dst
    gray = np.asarray(Image.open(src).convert("L"), dtype=np.float32)
    out = preprocess_view(gray, view=view_code, side=side,
                          remove_pectoral=remove_pectoral, normalize=normalize)
    arr8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(arr8, mode="L").save(dst)
    return dst


def main():
    args = parse_args()

    tag = "brm_pec" if args.remove_pectoral else "brm"
    if args.no_normalize:
        tag += "_nonorm"
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/cache") / tag
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / f"all_splits_{tag}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.split_csv)
    id_col = "ID" if "ID" in df.columns else df.columns[0]

    tasks = []
    new_df = df.copy()
    for i, row in df.iterrows():
        case_id = str(row[id_col]).strip()
        for view in VIEW_NAMES:
            src = str(row[f"{view}_path"])
            dst = str((out_dir / case_id / f"{view}.png").resolve())
            view_code, side = VIEW_META[view]
            tasks.append((src, dst, view_code, side, args.remove_pectoral, not args.no_normalize))
            new_df.at[i, f"{view}_path"] = dst

    print(f"Caching {len(tasks)} images ({len(df)} exams x 4 views) -> {out_dir}", flush=True)
    print(f"remove_pectoral = {args.remove_pectoral} | workers = {args.workers}", flush=True)

    done = 0
    with Pool(args.workers) as pool:
        for _ in pool.imap_unordered(_process_one, tasks, chunksize=8):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)

    new_df.to_csv(out_csv, index=False)
    print(f"Done. Cached PNGs in {out_dir}", flush=True)
    print(f"New split CSV: {out_csv}", flush=True)
    print(f"Train with:  --tn-split-csv {out_csv} --preprocess none", flush=True)


if __name__ == "__main__":
    main()
