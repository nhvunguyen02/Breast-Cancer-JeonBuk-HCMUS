"""Generate the TN-Mammo 4-view split CSV consumed by train_phaseG_mixed_loss.py.

Reads the label file (ID, Labels, Age) and, for each case, resolves the four
standard view images under <images-dir>/<ID>/. Produces a stratified
train/valid/test split (default 60/20/20, seed 42) written with ABSOLUTE image
paths, so the resulting CSV can be consumed from any working directory.

Run (from repo root):
    python src/make_splits.py

Output columns match what data.standardize_split_df expects:
    ID, Labels, split, left_cc_path, left_mlo_path, right_cc_path, right_mlo_path
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# View key (as in constants.VIEW_NAMES) -> filename on disk.
VIEW_FILES = {
    "left_cc": "Left - CC.jpg",
    "left_mlo": "Left - MLO.jpg",
    "right_cc": "Right - CC.jpg",
    "right_mlo": "Right - MLO.jpg",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-csv", default="data/TN-Mammo_labels.csv")
    p.add_argument("--images-dir", default="data/images")
    p.add_argument("--out-csv", default="data/splits/all_splits_with_paths.csv")
    p.add_argument("--valid-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_rows(labels, images_dir):
    """Return (rows, missing): one row per case that has all four views."""
    rows = []
    missing = []
    for _, r in labels.iterrows():
        case_id = str(r["ID"]).strip()
        case_dir = images_dir / case_id

        paths = {}
        ok = True
        for view, fname in VIEW_FILES.items():
            fp = case_dir / fname
            if not fp.exists():
                missing.append(str(fp))
                ok = False
            paths[f"{view}_path"] = str(fp.resolve())

        if not ok:
            continue

        row = {"ID": case_id, "Labels": str(r["Labels"]).strip()}
        row.update(paths)
        rows.append(row)

    return rows, missing


def main():
    args = parse_args()

    labels = pd.read_csv(args.labels_csv)
    images_dir = Path(args.images_dir)

    rows, missing = build_rows(labels, images_dir)

    if missing:
        print(f"WARNING: {len(missing)} missing view files, e.g.:")
        for m in missing[:5]:
            print("  ", m)

    df = pd.DataFrame(rows)
    print(f"Cases with all 4 views: {len(df)} / {len(labels)}")

    # Stratified split: first carve out (valid + test), then split that into
    # valid and test. Stratify on Labels so every class keeps its proportion.
    val_test_frac = args.valid_frac + args.test_frac
    train_df, tmp_df = train_test_split(
        df,
        test_size=val_test_frac,
        stratify=df["Labels"],
        random_state=args.seed,
    )
    rel_test = args.test_frac / val_test_frac
    valid_df, test_df = train_test_split(
        tmp_df,
        test_size=rel_test,
        stratify=tmp_df["Labels"],
        random_state=args.seed,
    )

    train_df = train_df.assign(split="train")
    valid_df = valid_df.assign(split="valid")
    test_df = test_df.assign(split="test")

    out = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    col_order = ["ID", "Labels", "split"] + [f"{v}_path" for v in VIEW_FILES]
    out = out[col_order]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print("Split sizes:", out["split"].value_counts().reindex(["train", "valid", "test"]).to_dict())
    print("Per-split label distribution:")
    dist = out.groupby(["split", "Labels"]).size().unstack(fill_value=0)
    dist = dist.reindex(["train", "valid", "test"])
    print(dist)


if __name__ == "__main__":
    main()
