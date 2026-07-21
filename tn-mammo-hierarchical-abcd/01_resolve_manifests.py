#!/usr/bin/env python3
"""Resolve and canonicalize TN-Mammo/VinDr case-level manifests.

The resolver reads metadata only. It does not open any TN locked-test image.
Development outputs and locked-test outputs are physically separated so that
training scripts cannot accidentally consume locked-test rows.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

VIEW_ORDER = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
LABELS = ["A", "B", "C", "D"]

CASE_CANDIDATES = ["case_id", "study_id", "exam_id", "patient_id", "studyinstanceuid", "id"]
LABEL_CANDIDATES = ["label", "density", "density_label", "breast_density", "birads_density", "breastdensity"]
SPLIT_CANDIDATES = ["split", "subset", "set", "partition", "data_split"]
VIEW_CANDIDATES = {
    "L_CC": ["l_cc", "lcc", "left_cc", "leftcc", "path_l_cc", "l_cc_path", "left_cc_path", "image_l_cc"],
    "L_MLO": ["l_mlo", "lmlo", "left_mlo", "leftmlo", "path_l_mlo", "l_mlo_path", "left_mlo_path", "image_l_mlo"],
    "R_CC": ["r_cc", "rcc", "right_cc", "rightcc", "path_r_cc", "r_cc_path", "right_cc_path", "image_r_cc"],
    "R_MLO": ["r_mlo", "rmlo", "right_mlo", "rightmlo", "path_r_mlo", "r_mlo_path", "right_mlo_path", "image_r_mlo"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns: Iterable[str], candidates: list[str], required: bool = True) -> str | None:
    lookup = {normalized_name(col): col for col in columns}
    for candidate in candidates:
        hit = lookup.get(normalized_name(candidate))
        if hit is not None:
            return hit
    # Conservative substring fallback for path columns and verbose schemas.
    for candidate in candidates:
        key = normalized_name(candidate)
        for norm_col, original in lookup.items():
            if key and (norm_col.endswith(key) or key.endswith(norm_col)):
                return original
    if required:
        raise KeyError(f"Could not resolve a required column from candidates={candidates}; columns={list(columns)}")
    return None


def normalize_label(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("Missing density label")
    text = str(value).strip().upper()
    text = text.replace("DENSITY", "").replace("BI-RADS", "").replace("BIRADS", "").strip(" _:-")
    mapping = {
        "0": "A",
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "I": "A",
        "II": "B",
        "III": "C",
        "IV": "D",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    if text in mapping:
        return mapping[text]
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Unsupported label value: {value!r}")




def normalize_label_column(series: pd.Series) -> pd.Series:
    stripped = series.astype(str).str.strip()
    numeric = pd.to_numeric(stripped, errors="coerce")
    if numeric.notna().all():
        values = set(int(value) for value in numeric.tolist())
        if values.issubset({0, 1, 2, 3}):
            mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
            return numeric.astype(int).map(mapping)
        if values.issubset({1, 2, 3, 4}):
            mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
            return numeric.astype(int).map(mapping)
    return series.map(normalize_label)

def normalize_split(value: Any, default_split: str) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return default_split
    text = normalized_name(str(value))
    mapping = {
        "train": "train",
        "training": "train",
        "tr": "train",
        "valid": "valid",
        "validation": "valid",
        "val": "valid",
        "dev": "valid",
        "calib": "calib",
        "calibration": "calib",
        "test": "test",
        "testing": "test",
        "lockedtest": "test",
    }
    return mapping.get(text, str(value).strip().lower())


def resolve_globs(patterns: list[str]) -> list[Path]:
    results: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for item in sorted(glob.glob(pattern, recursive=True)):
            path = Path(item).resolve()
            if path.is_file() and str(path) not in seen:
                results.append(path)
                seen.add(str(path))
    return results


def choose_manifest(paths: list[Path], role: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No manifest candidate found for {role}")
    # Prefer canonical/resolved files, then latest modification time.
    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        quality = 0
        quality += 10 if "canonical" in name else 0
        quality += 6 if "resolved" in name else 0
        quality += 3 if "caselevel" in name or "case_level" in name else 0
        quality -= 8 if role.endswith("train") and "test" in name else 0
        quality += 8 if role.endswith("test") and "test" in name else 0
        return quality, path.stat().st_mtime

    return sorted(paths, key=score, reverse=True)[0]


def canonicalize(path: Path, dataset: str, default_split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    case_col = find_column(frame.columns, CASE_CANDIDATES)
    label_col = find_column(frame.columns, LABEL_CANDIDATES)
    split_col = find_column(frame.columns, SPLIT_CANDIDATES, required=False)
    view_cols = {view: find_column(frame.columns, candidates) for view, candidates in VIEW_CANDIDATES.items()}

    out = pd.DataFrame(index=frame.index)
    out["dataset"] = dataset
    out["case_id"] = frame[case_col].astype(str).str.strip()
    out["label"] = normalize_label_column(frame[label_col])
    if split_col is None:
        out["split"] = default_split
    else:
        out["split"] = frame[split_col].map(lambda value: normalize_split(value, default_split))
    for view in VIEW_ORDER:
        source = view_cols[view]
        out[view] = frame[source].fillna("").astype(str).str.strip()
    out["source_manifest"] = str(path)

    duplicated = out["case_id"].duplicated(keep=False)
    if duplicated.any():
        duplicates = out.loc[duplicated, "case_id"].tolist()[:20]
        raise ValueError(f"Duplicate case IDs in {path}: {duplicates}")
    if (out["case_id"] == "").any():
        raise ValueError(f"Blank case IDs in {path}")
    if not set(out["label"]).issubset(set(LABELS)):
        raise ValueError(f"Unexpected labels in {path}: {sorted(set(out['label']))}")
    return out


def infer_path_policy(frame: pd.DataFrame) -> str:
    tokens = ("sam2", "masked", "bbox", "breast_only", "breast-only", "segment", "crop")
    paths: list[str] = []
    for view in VIEW_ORDER:
        paths.extend(frame[view].fillna("").astype(str).str.lower().tolist())
    paths = [value for value in paths if value and value not in {"nan", "none", "null"}]
    if not paths:
        return "unknown"
    flagged = sum(any(token in value for token in tokens) for value in paths)
    return "breast_only_or_segmented" if flagged / len(paths) >= 0.50 else "full_raw_candidate"


def count_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = (
        frame.groupby(["dataset", "split", "label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return result.sort_values(["dataset", "split", "label"]).reset_index(drop=True)


def verify_tn_counts(frame: pd.DataFrame, expected: dict[str, dict[str, int]]) -> None:
    for split, label_counts in expected.items():
        subset = frame[frame["split"] == split]
        observed = subset["label"].value_counts().to_dict()
        for label, expected_count in label_counts.items():
            actual = int(observed.get(label, 0))
            if actual != int(expected_count):
                raise ValueError(
                    f"TN count mismatch split={split} label={label}: observed={actual}, expected={expected_count}"
                )


def verify_no_overlap(frame: pd.DataFrame, dataset: str) -> None:
    split_sets = {split: set(part["case_id"].astype(str)) for split, part in frame.groupby("split")}
    names = sorted(split_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"{dataset} split overlap {left}/{right}: {sorted(overlap)[:20]}")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vindr-manifest", type=Path)
    parser.add_argument("--vindr-locked-test-manifest", type=Path)
    parser.add_argument("--tn-manifest", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    protocol_cfg = config["protocol"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vindr_train_candidates = resolve_globs(data_cfg.get("vindr_manifest_candidates", []))
    vindr_test_candidates = resolve_globs(data_cfg.get("vindr_locked_test_manifest_candidates", []))
    tn_candidates = resolve_globs(data_cfg.get("tn_manifest_candidates", []))

    vindr_path = args.vindr_manifest.resolve() if args.vindr_manifest else choose_manifest(vindr_train_candidates, "vindr_train")
    tn_path = args.tn_manifest.resolve() if args.tn_manifest else choose_manifest(tn_candidates, "tn")

    vindr = canonicalize(vindr_path, "VinDr", default_split="train")
    if (vindr["split"] == "test").any():
        vindr_all = vindr
        selected_vindr_test_path = vindr_path
    else:
        selected_vindr_test_path = (
            args.vindr_locked_test_manifest.resolve()
            if args.vindr_locked_test_manifest
            else choose_manifest(vindr_test_candidates, "vindr_test")
        )
        vindr_test = canonicalize(selected_vindr_test_path, "VinDr", default_split="test")
        vindr_test["split"] = "test"
        duplicate_between = set(vindr["case_id"]) & set(vindr_test["case_id"])
        if duplicate_between:
            raise ValueError(f"VinDr train/test overlap: {sorted(duplicate_between)[:20]}")
        vindr_all = pd.concat([vindr, vindr_test], ignore_index=True)

    tn_all = canonicalize(tn_path, "TN", default_split="train")
    verify_no_overlap(vindr_all, "VinDr")
    verify_no_overlap(tn_all, "TN")
    verify_tn_counts(tn_all, protocol_cfg["expected_tn_counts"])

    vindr_policy = infer_path_policy(vindr_all[vindr_all["split"] != "test"])
    tn_policy = infer_path_policy(tn_all[tn_all["split"].isin(["train", "valid"])])
    expected_policy = str(config["experiment"].get("input_policy", "")).strip()
    if vindr_policy != tn_policy:
        raise ValueError(f"Preprocessing parity violation: VinDr={vindr_policy}, TN={tn_policy}")
    if expected_policy and expected_policy != vindr_policy:
        raise ValueError(
            f"Input policy mismatch: config={expected_policy}, inferred manifests={vindr_policy}. "
            "Use matching raw/raw or breast-only/breast-only manifests and record the policy explicitly."
        )

    vindr_dev = vindr_all[vindr_all["split"] != "test"].copy().reset_index(drop=True)
    vindr_locked = vindr_all[vindr_all["split"] == "test"].copy().reset_index(drop=True)
    tn_dev = tn_all[tn_all["split"].isin(["train", "valid", "calib"])].copy().reset_index(drop=True)
    tn_locked = tn_all[tn_all["split"] == "test"].copy().reset_index(drop=True)

    expected_vindr_dev = int(protocol_cfg.get("expected_vindr_train_pool", len(vindr_dev)))
    expected_vindr_test = int(protocol_cfg.get("expected_vindr_locked_test", len(vindr_locked)))
    if len(vindr_dev) != expected_vindr_dev:
        raise ValueError(f"VinDr development pool mismatch: observed={len(vindr_dev)} expected={expected_vindr_dev}")
    if len(vindr_locked) != expected_vindr_test:
        raise ValueError(f"VinDr locked test mismatch: observed={len(vindr_locked)} expected={expected_vindr_test}")

    outputs = {
        "vindr_dev": args.output_dir / "resolved_vindr_dev.csv",
        "vindr_locked_test": args.output_dir / "resolved_vindr_locked_test.csv",
        "tn_dev": args.output_dir / "resolved_tn_dev.csv",
        "tn_locked_test": args.output_dir / "resolved_tn_locked_test.csv",
    }
    write_csv(vindr_dev, outputs["vindr_dev"])
    write_csv(vindr_locked, outputs["vindr_locked_test"])
    write_csv(tn_dev, outputs["tn_dev"])
    write_csv(tn_locked, outputs["tn_locked_test"])

    summary = count_table(pd.concat([vindr_all, tn_all], ignore_index=True))
    summary.to_csv(args.output_dir / "split_summary.csv", index=False)

    hashes = {
        "config": sha256_file(args.config),
        "source_vindr": sha256_file(vindr_path),
        "source_vindr_locked_test": sha256_file(selected_vindr_test_path),
        "source_tn": sha256_file(tn_path),
        **{f"resolved_{key}": sha256_file(path) for key, path in outputs.items()},
    }
    (args.output_dir / "manifest_hashes.json").write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "status": "PASS",
        "view_order": VIEW_ORDER,
        "label_map": {label: index for index, label in enumerate(LABELS)},
        "selected_sources": {
            "vindr": str(vindr_path),
            "vindr_locked_test": str(selected_vindr_test_path),
            "tn": str(tn_path),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
        "input_policy": {"configured": expected_policy, "vindr_inferred": vindr_policy, "tn_inferred": tn_policy},
        "counts": {
            "vindr_dev": len(vindr_dev),
            "vindr_locked_test": len(vindr_locked),
            "tn_dev": len(tn_dev),
            "tn_locked_test": len(tn_locked),
        },
        "locked_test_policy": "Locked files are resolved but never consumed by development scripts.",
    }
    (args.output_dir / "manifest_resolution.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
