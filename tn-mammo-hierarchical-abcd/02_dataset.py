#!/usr/bin/env python3
"""Case-level four-view dataset with fixed view order and mask-aware preprocessing."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

VIEW_ORDER = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    interpolation: str = "bilinear"
    foreground_crop: bool = False
    normalize: str = "robust_foreground"
    view_dropout_prob: float = 0.0


def _pil_resample(name: str) -> int:
    name = name.lower()
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported interpolation: {name}")
    return mapping[name]


def load_grayscale(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if image_path.suffix.lower() in {".dcm", ".dicom"}:
        try:
            import pydicom
        except Exception as exc:
            raise RuntimeError("pydicom is required for DICOM input") from exc
        ds = pydicom.dcmread(str(image_path), force=True)
        array = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        array = array * slope + intercept
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric == "MONOCHROME1":
            array = float(array.max()) + float(array.min()) - array
    else:
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mammogram, got shape={array.shape} at {image_path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite image values at {image_path}")
    return array


def foreground_mask(array: np.ndarray) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=bool)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if hi <= lo:
        return array > lo
    threshold = lo + 0.03 * (hi - lo)
    mask = array > threshold
    if mask.mean() < 0.005:
        mask = array > lo
    return mask


def robust_normalize(array: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mask = foreground_mask(array)
    values = array[mask]
    if values.size < 16:
        values = array.reshape(-1)
    p1, p50, p99 = [float(x) for x in np.percentile(values, [1, 50, 99])]
    scale = max(p99 - p1, 1e-6)
    normalized = np.clip((array - p1) / scale, 0.0, 1.0).astype(np.float32)
    return normalized, {
        "p1": p1,
        "p50": p50,
        "p99": p99,
        "foreground_area_ratio": float(mask.mean()),
    }


def crop_foreground(array: np.ndarray) -> np.ndarray:
    mask = foreground_mask(array)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return array
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if x1 <= x0 or y1 <= y0:
        return array
    return array[y0:y1, x0:x1]


def to_tensor(
    array: np.ndarray,
    cfg: PreprocessConfig,
    augmentation: dict[str, float | bool] | None = None,
) -> torch.Tensor:
    if cfg.foreground_crop:
        array = crop_foreground(array)
    if cfg.normalize != "robust_foreground":
        raise ValueError(f"Unsupported normalization policy: {cfg.normalize}")
    normalized, _ = robust_normalize(array)
    image = Image.fromarray(np.uint8(np.round(normalized * 255.0)), mode="L")
    if augmentation:
        angle = float(augmentation.get("angle", 0.0))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        image = ImageEnhance.Brightness(image).enhance(float(augmentation.get("brightness", 1.0)))
        image = ImageEnhance.Contrast(image).enhance(float(augmentation.get("contrast", 1.0)))
    image = image.resize((cfg.image_size, cfg.image_size), resample=_pil_resample(cfg.interpolation))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1)
    # ImageNet normalization is applied only after image-intensity normalization and channel conversion.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class FourViewMammoDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        preprocess: PreprocessConfig,
        training: bool = False,
        seed: int = 42,
        strict_paths: bool = True,
    ) -> None:
        self.frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        required = {"case_id", "label", "split", *VIEW_ORDER}
        missing = required - set(self.frame.columns)
        if missing:
            raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
        self.frame = self.frame.reset_index(drop=True)
        self.preprocess = preprocess
        self.training = bool(training)
        self.seed = int(seed)
        self.strict_paths = bool(strict_paths)
        labels = set(self.frame["label"].astype(str))
        if not labels.issubset(LABEL_MAP):
            raise ValueError(f"Unexpected labels: {sorted(labels)}")

    def __len__(self) -> int:
        return len(self.frame)

    def _augmentation(self, index: int) -> dict[str, float]:
        if not self.training:
            return {"angle": 0.0, "brightness": 1.0, "contrast": 1.0}
        rng = random.Random(self.seed * 1_000_003 + index)
        return {
            "angle": rng.uniform(-7.0, 7.0),
            "brightness": rng.uniform(0.92, 1.08),
            "contrast": rng.uniform(0.92, 1.08),
        }

    def _apply_view_dropout(self, mask: torch.Tensor, index: int) -> torch.Tensor:
        probability = float(self.preprocess.view_dropout_prob)
        if not self.training or probability <= 0:
            return mask.clone()
        rng = random.Random(self.seed * 2_000_003 + index)
        result = mask.clone()
        candidates = [i for i in range(4) if result[i] > 0 and rng.random() < probability]
        for view_id in candidates:
            trial = result.clone()
            trial[view_id] = 0
            if int(trial.sum().item()) < 1:
                continue
            left_present = bool(trial[0] > 0 or trial[1] > 0)
            right_present = bool(trial[2] > 0 or trial[3] > 0)
            # Never drop both breasts simultaneously; at least one side must remain.
            if not left_present and not right_present:
                continue
            result = trial
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        original_mask = torch.zeros(4, dtype=torch.bool)
        arrays: list[np.ndarray | None] = []
        for view_id, view in enumerate(VIEW_ORDER):
            path = str(row[view]).strip()
            if path and path.lower() not in {"nan", "none", "null"}:
                if self.strict_paths and not Path(path).is_file():
                    raise FileNotFoundError(f"Non-empty manifest path is missing: case={row['case_id']} view={view} path={path}")
                if Path(path).is_file():
                    arrays.append(load_grayscale(path))
                    original_mask[view_id] = True
                else:
                    arrays.append(None)
            else:
                arrays.append(None)
        if int(original_mask.sum().item()) < 1:
            raise ValueError(f"All views are missing for case_id={row['case_id']}")

        effective_mask = self._apply_view_dropout(original_mask, index)
        augmentation = self._augmentation(index)
        placeholder = torch.zeros(3, self.preprocess.image_size, self.preprocess.image_size, dtype=torch.float32)
        tensors: list[torch.Tensor] = []
        for view_id, array in enumerate(arrays):
            if array is None or not bool(effective_mask[view_id]):
                tensors.append(placeholder.clone())
            else:
                tensors.append(to_tensor(array, self.preprocess, augmentation=augmentation))
        views = torch.stack(tensors, dim=0)
        y4 = LABEL_MAP[str(row["label"])]
        return {
            "views": views,
            "view_mask": effective_mask,
            "original_view_mask": original_mask,
            "y4": torch.tensor(y4, dtype=torch.long),
            "case_id": str(row["case_id"]),
            "split": str(row["split"]),
            "label": str(row["label"]),
        }


def collate_cases(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "views": torch.stack([item["views"] for item in batch]),
        "view_mask": torch.stack([item["view_mask"] for item in batch]),
        "original_view_mask": torch.stack([item["original_view_mask"] for item in batch]),
        "y4": torch.stack([item["y4"] for item in batch]),
        "case_id": [item["case_id"] for item in batch],
        "split": [item["split"] for item in batch],
        "label": [item["label"] for item in batch],
    }


def preprocessing_audit(manifests: list[Path], output: Path, max_cases: int = 0) -> None:
    frames = [pd.read_csv(path) for path in manifests]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["dataset", "split", "case_id"]).reset_index(drop=True)
    if max_cases > 0:
        group_cols = [column for column in ["dataset", "split", "label"] if column in frame.columns]
        if group_cols:
            frame = (
                frame.groupby(group_cols, group_keys=False, sort=True)
                .head(max_cases)
                .reset_index(drop=True)
            )
        else:
            frame = frame.head(max_cases).reset_index(drop=True)

    intensity_samples: list[float] = []
    area_ratios: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    missing = {view: 0 for view in VIEW_ORDER}
    readable = {view: 0 for view in VIEW_ORDER}
    failures: list[dict[str, str]] = []

    for _, row in frame.iterrows():
        for view in VIEW_ORDER:
            path = str(row[view]).strip()
            if not path or path.lower() in {"nan", "none", "null"}:
                missing[view] += 1
                continue
            try:
                array = load_grayscale(path)
                height, width = array.shape
                widths.append(int(width))
                heights.append(int(height))
                aspect_ratios.append(float(width / max(height, 1)))
                normalized, stats = robust_normalize(array)
                area_ratios.append(stats["foreground_area_ratio"])
                mask = foreground_mask(array)
                values = normalized[mask]
                if values.size:
                    stride = max(1, values.size // 4096)
                    intensity_samples.extend(values[::stride][:4096].astype(float).tolist())
                readable[view] += 1
            except Exception as exc:
                failures.append({"case_id": str(row["case_id"]), "view": view, "path": path, "error": repr(exc)})

    def quantiles(values: list[float], probs: list[float]) -> dict[str, float | None]:
        if not values:
            return {str(prob): None for prob in probs}
        array = np.asarray(values, dtype=np.float64)
        return {str(prob): float(np.quantile(array, prob)) for prob in probs}

    report = {
        "status": "PASS" if not failures else "FAIL_IMAGE_READ",
        "cases_scanned": int(len(frame)),
        "images_readable_by_view": readable,
        "missing_by_view": missing,
        "foreground_intensity_quantiles": quantiles(intensity_samples, [0.01, 0.5, 0.99]),
        "foreground_area_ratio_quantiles": quantiles(area_ratios, [0.01, 0.5, 0.99]),
        "width_quantiles": quantiles([float(v) for v in widths], [0.01, 0.5, 0.99]),
        "height_quantiles": quantiles([float(v) for v in heights], [0.01, 0.5, 0.99]),
        "aspect_ratio_quantiles": quantiles(aspect_ratios, [0.01, 0.5, 0.99]),
        "class_counts_by_split": (
            frame.groupby(["dataset", "split", "label"]).size().rename("count").reset_index().to_dict("records")
        ),
        "failure_count": len(failures),
        "failures": failures[:200],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "cases_scanned": len(frame)}, indent=2))
    if failures:
        raise RuntimeError(f"Preprocessing audit found {len(failures)} unreadable images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.audit:
        if not args.manifest or args.output is None:
            raise ValueError("--audit requires at least one --manifest and --output")
        preprocessing_audit(args.manifest, args.output, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
