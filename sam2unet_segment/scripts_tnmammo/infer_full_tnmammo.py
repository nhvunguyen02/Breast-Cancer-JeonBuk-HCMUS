#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SAM2UNet import SAM2UNet  # noqa: E402


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)

EXPECTED_VIEWS = [
    "L_CC",
    "L_MLO",
    "R_CC",
    "R_MLO",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inference and non-destructive QC for full TN-Mammo "
            "using the locked SAM2-UNet checkpoint."
        )
    )

    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-images", type=int, default=2704)
    parser.add_argument("--expected-cases", type=int, default=676)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def safe_token(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("._-")

    if not text:
        text = "unnamed"

    return text


def case_folder(case_id: str) -> str:
    digest = hashlib.sha1(
        case_id.encode("utf-8")
    ).hexdigest()[:8]

    return f"{safe_token(case_id)}__{digest}"


def letterbox_image(
    image: Image.Image,
    size: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    original_width, original_height = image.size

    scale = min(
        size / original_width,
        size / original_height,
    )

    resized_width = max(
        1,
        int(round(original_width * scale)),
    )

    resized_height = max(
        1,
        int(round(original_height * scale)),
    )

    resized = image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        color=(0, 0, 0),
    )

    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    canvas.paste(resized, (left, top))

    array = np.asarray(
        canvas,
        dtype=np.float32,
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(2, 0, 1)

    tensor = (
        tensor - IMAGENET_MEAN
    ) / IMAGENET_STD

    metadata = {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "letterbox_left": int(left),
        "letterbox_top": int(top),
        "letterbox_scale": float(scale),
    }

    return tensor, metadata


def restore_probability(
    padded_probability: np.ndarray,
    metadata: dict[str, int | float],
) -> np.ndarray:
    left = int(metadata["letterbox_left"])
    top = int(metadata["letterbox_top"])

    resized_width = int(
        metadata["resized_width"]
    )

    resized_height = int(
        metadata["resized_height"]
    )

    original_width = int(
        metadata["original_width"]
    )

    original_height = int(
        metadata["original_height"]
    )

    cropped = padded_probability[
        top : top + resized_height,
        left : left + resized_width,
    ]

    restored = cv2.resize(
        cropped,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    return np.clip(
        restored.astype(np.float32),
        0.0,
        1.0,
    )


def image_statistics(
    rgb_image: np.ndarray,
) -> dict[str, float]:
    gray = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY,
    )

    height, width = gray.shape

    border_height = max(
        5,
        int(round(height * 0.02)),
    )

    border_width = max(
        5,
        int(round(width * 0.02)),
    )

    border_pixels = np.concatenate(
        [
            gray[:border_height, :].reshape(-1),
            gray[
                height - border_height :,
                :,
            ].reshape(-1),
            gray[
                border_height:
                height - border_height,
                :border_width,
            ].reshape(-1),
            gray[
                border_height:
                height - border_height,
                width - border_width :,
            ].reshape(-1),
        ]
    )

    return {
        "image_zero_fraction": float(
            np.mean(gray == 0)
        ),
        "image_dark_fraction_le5": float(
            np.mean(gray <= 5)
        ),
        "image_intensity_p01": float(
            np.percentile(gray, 1)
        ),
        "image_intensity_p50": float(
            np.percentile(gray, 50)
        ),
        "image_intensity_p99": float(
            np.percentile(gray, 99)
        ),
        "border_intensity_p50": float(
            np.percentile(border_pixels, 50)
        ),
        "border_intensity_p99": float(
            np.percentile(border_pixels, 99)
        ),
        "border_nonzero_fraction": float(
            np.mean(border_pixels > 0)
        ),
    }


def mask_statistics(
    mask: np.ndarray,
) -> dict[str, object]:
    binary = mask.astype(np.uint8)
    height, width = binary.shape

    total_pixels = int(height * width)
    foreground_pixels = int(binary.sum())

    if foreground_pixels == 0:
        return {
            "prediction_pixels": 0,
            "prediction_area_ratio": 0.0,
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
            "bbox_x": None,
            "bbox_y": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_width_ratio": 0.0,
            "bbox_height_ratio": 0.0,
            "centroid_x_normalized": None,
            "centroid_y_normalized": None,
            "edge_band_px": 0,
            "left_edge_fraction": 0.0,
            "right_edge_fraction": 0.0,
            "top_edge_fraction": 0.0,
            "bottom_edge_fraction": 0.0,
            "touches_left_edge": False,
            "touches_right_edge": False,
            "touches_top_edge": False,
            "touches_bottom_edge": False,
            "dominant_vertical_edge": "none",
        }

    component_count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    foreground_component_count = (
        component_count - 1
    )

    component_areas = (
        stats[1:, cv2.CC_STAT_AREA]
        if foreground_component_count > 0
        else np.asarray([], dtype=np.int64)
    )

    largest_component_pixels = (
        int(component_areas.max())
        if len(component_areas)
        else 0
    )

    largest_component_ratio = (
        largest_component_pixels
        / foreground_pixels
        if foreground_pixels > 0
        else 0.0
    )

    foreground_y, foreground_x = np.where(
        binary > 0
    )

    x_min = int(foreground_x.min())
    x_max = int(foreground_x.max())
    y_min = int(foreground_y.min())
    y_max = int(foreground_y.max())

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    centroid_x = float(
        foreground_x.mean() / max(width - 1, 1)
    )

    centroid_y = float(
        foreground_y.mean() / max(height - 1, 1)
    )

    edge_band = max(
        5,
        int(round(min(height, width) * 0.005)),
    )

    left_fraction = float(
        binary[:, :edge_band].mean()
    )

    right_fraction = float(
        binary[:, width - edge_band :].mean()
    )

    top_fraction = float(
        binary[:edge_band, :].mean()
    )

    bottom_fraction = float(
        binary[
            height - edge_band :,
            :,
        ].mean()
    )

    touches_left = bool(
        binary[:, :edge_band].any()
    )

    touches_right = bool(
        binary[
            :,
            width - edge_band :,
        ].any()
    )

    touches_top = bool(
        binary[:edge_band, :].any()
    )

    touches_bottom = bool(
        binary[
            height - edge_band :,
            :,
        ].any()
    )

    if left_fraction > right_fraction:
        dominant_vertical_edge = "left"
    elif right_fraction > left_fraction:
        dominant_vertical_edge = "right"
    else:
        dominant_vertical_edge = "equal"

    return {
        "prediction_pixels": foreground_pixels,
        "prediction_area_ratio": float(
            foreground_pixels / total_pixels
        ),
        "component_count": int(
            foreground_component_count
        ),
        "largest_component_pixels": (
            largest_component_pixels
        ),
        "largest_component_ratio": float(
            largest_component_ratio
        ),
        "bbox_x": x_min,
        "bbox_y": y_min,
        "bbox_width": int(bbox_width),
        "bbox_height": int(bbox_height),
        "bbox_width_ratio": float(
            bbox_width / width
        ),
        "bbox_height_ratio": float(
            bbox_height / height
        ),
        "centroid_x_normalized": centroid_x,
        "centroid_y_normalized": centroid_y,
        "edge_band_px": int(edge_band),
        "left_edge_fraction": left_fraction,
        "right_edge_fraction": right_fraction,
        "top_edge_fraction": top_fraction,
        "bottom_edge_fraction": bottom_fraction,
        "touches_left_edge": touches_left,
        "touches_right_edge": touches_right,
        "touches_top_edge": touches_top,
        "touches_bottom_edge": touches_bottom,
        "dominant_vertical_edge": (
            dominant_vertical_edge
        ),
    }


def fixed_qc_flags(
    metrics: dict[str, object],
) -> list[str]:
    flags: list[str] = []

    area = float(
        metrics["prediction_area_ratio"]
    )

    components = int(
        metrics["component_count"]
    )

    largest_ratio = float(
        metrics["largest_component_ratio"]
    )

    touches_left = bool(
        metrics["touches_left_edge"]
    )

    touches_right = bool(
        metrics["touches_right_edge"]
    )

    touches_top = bool(
        metrics["touches_top_edge"]
    )

    touches_bottom = bool(
        metrics["touches_bottom_edge"]
    )

    if int(metrics["prediction_pixels"]) == 0:
        flags.append("empty_prediction")

    if 0 < area < 0.05:
        flags.append("very_small_mask")

    if area > 0.95:
        flags.append("very_large_mask")

    if components > 3:
        flags.append("fragmented_mask")

    if components > 0 and largest_ratio < 0.97:
        flags.append(
            "low_largest_component_ratio"
        )

    if not touches_left and not touches_right:
        flags.append(
            "no_vertical_border_contact"
        )

    if touches_left and touches_right:
        flags.append(
            "both_vertical_borders_contacted"
        )

    if touches_top and touches_bottom:
        flags.append(
            "top_and_bottom_contacted"
        )

    if (
        float(metrics["bbox_width_ratio"]) > 0.99
        and float(metrics["bbox_height_ratio"]) > 0.99
    ):
        flags.append("near_full_frame_bbox")

    return flags


def robust_zscore(
    series: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = values.median()
    mad = (
        values - median
    ).abs().median()

    if (
        not np.isfinite(mad)
        or mad < 1e-12
    ):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=np.float64,
        )

    return (
        0.67448975
        * (values - median)
        / mad
    )


def add_statistical_qc(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    metrics = [
        "prediction_area_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
        "largest_component_ratio",
        "border_nonzero_fraction",
    ]

    z_columns: list[str] = []

    for metric in metrics:
        z_column = f"robust_z_{metric}"
        z_columns.append(z_column)

        dataframe[z_column] = (
            dataframe.groupby(
                "view_key",
                group_keys=False,
            )[metric]
            .apply(robust_zscore)
            .astype(float)
        )

    dataframe["max_absolute_robust_z"] = (
        dataframe[z_columns]
        .abs()
        .max(axis=1)
    )

    final_flags: list[str] = []
    fixed_counts: list[int] = []

    for _, row in dataframe.iterrows():
        fixed = [
            flag
            for flag in str(
                row["qc_flags_fixed"]
            ).split(";")
            if flag
        ]

        fixed_counts.append(len(fixed))

        statistical: list[str] = []

        for metric in metrics:
            z_value = float(
                row[
                    f"robust_z_{metric}"
                ]
            )

            if abs(z_value) > 4.5:
                direction = (
                    "high"
                    if z_value > 0
                    else "low"
                )

                statistical.append(
                    f"outlier_{metric}_{direction}"
                )

        combined = fixed + statistical
        final_flags.append(
            ";".join(combined)
        )

    dataframe["fixed_qc_flag_count"] = fixed_counts
    dataframe["qc_flags"] = final_flags

    dataframe["is_flagged"] = (
        (dataframe["fixed_qc_flag_count"] > 0)
        | (
            dataframe[
                "max_absolute_robust_z"
            ]
            > 4.5
        )
    )

    dataframe["qc_score"] = (
        dataframe[
            "fixed_qc_flag_count"
        ]
        * 10.0
        + dataframe[
            "max_absolute_robust_z"
        ].clip(
            lower=0.0,
            upper=20.0,
        )
        + (
            dataframe[
                "component_count"
            ]
            .clip(lower=1)
            - 1
        )
        * 0.5
    )

    return dataframe


def make_qc_overlay(
    image_path: Path,
    mask_path: Path,
    title: str,
    output_path: Path,
    max_dimension: int = 1600,
) -> None:
    image = np.asarray(
        Image.open(
            image_path
        ).convert("RGB"),
        dtype=np.uint8,
    )

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Cannot read prediction mask: {mask_path}"
        )

    mask = (mask > 0).astype(np.uint8)

    height, width = image.shape[:2]
    scale = min(
        1.0,
        max_dimension / max(height, width),
    )

    if scale < 1.0:
        new_width = max(
            1,
            int(round(width * scale)),
        )

        new_height = max(
            1,
            int(round(height * scale)),
        )

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

        mask = cv2.resize(
            mask,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    overlay = image.copy()

    # RGB cyan contour for standalone SAM2-UNet QC.
    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 255),
        4,
    )

    font_scale = max(
        0.6,
        min(
            overlay.shape[0],
            overlay.shape[1],
        )
        / 1400,
    )

    thickness = max(
        2,
        int(round(font_scale * 2)),
    )

    cv2.putText(
        overlay,
        title,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        "SAM2-UNet prediction: cyan",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(overlay).save(
        output_path
    )


def attach_overlays(
    dataframe: pd.DataFrame,
    overlay_dir: Path,
) -> pd.DataFrame:
    output = dataframe.copy()
    paths: list[str] = []

    for _, row in output.iterrows():
        folder = case_folder(
            str(row["case_id"])
        )

        overlay_path = (
            overlay_dir
            / folder
            / f"{safe_token(row['view_key'])}.png"
        )

        if not overlay_path.is_file():
            title = (
                f"{row['case_id']} | {row['view_key']} | "
                f"area={row['prediction_area_ratio']:.3f} | "
                f"components={int(row['component_count'])} | "
                f"QC={row['qc_score']:.2f}"
            )

            make_qc_overlay(
                Path(row["image_path"]),
                Path(
                    row[
                        "binary_prediction_path"
                    ]
                ),
                title,
                overlay_path,
            )

        paths.append(
            str(overlay_path.resolve())
        )

    output["overlay_path"] = paths
    return output


def make_contact_sheet(
    dataframe: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    count = min(
        24,
        len(dataframe),
    )

    if count == 0:
        return

    columns = 4
    rows = int(
        math.ceil(count / columns)
    )

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(20, 5 * rows),
    )

    axes = np.atleast_1d(
        axes
    ).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for axis, (_, row) in zip(
        axes,
        dataframe.head(count).iterrows(),
    ):
        image = plt.imread(
            row["overlay_path"]
        )

        axis.imshow(image)

        flags = str(
            row["qc_flags"]
        )

        if len(flags) > 60:
            flags = flags[:57] + "..."

        axis.set_title(
            f"{row['case_id']} | {row['view_key']}\n"
            f"Area={row['prediction_area_ratio']:.3f} | "
            f"Comp={int(row['component_count'])}\n"
            f"Score={row['qc_score']:.2f} | {flags}",
            fontsize=9,
        )

        axis.axis("off")

    figure.suptitle(
        title,
        fontsize=18,
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.975)
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)


def describe_metric(
    series: pd.Series,
) -> dict[str, float]:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=0)),
        "p01": float(
            np.percentile(values, 1)
        ),
        "p05": float(
            np.percentile(values, 5)
        ),
        "p95": float(
            np.percentile(values, 95)
        ),
        "p99": float(
            np.percentile(values, 99)
        ),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def make_dashboard(
    dataframe: pd.DataFrame,
    flagged_samples: pd.DataFrame,
    summary: dict[str, object],
    output_path: Path,
) -> None:
    figure = plt.figure(
        figsize=(22, 18)
    )

    grid = figure.add_gridspec(
        4,
        4,
        height_ratios=[
            0.7,
            1.3,
            2.2,
            2.2,
        ],
    )

    title_axis = figure.add_subplot(
        grid[0, :]
    )

    title_axis.axis("off")

    title_axis.text(
        0.5,
        0.5,
        (
            "SAM2-UNet Full TN-Mammo Inference & QC\n"
            f"Images={summary['n_images']} | "
            f"Cases={summary['n_cases']} | "
            f"Threshold={summary['threshold']} | "
            f"Flagged={summary['n_flagged']} | "
            f"Empty={summary['empty_predictions']}\n"
            "Raw prediction only — no morphological or "
            "connected-component post-processing"
        ),
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    area_axis = figure.add_subplot(
        grid[1, 0]
    )

    area_axis.hist(
        dataframe[
            "prediction_area_ratio"
        ],
        bins=35,
    )

    area_axis.set_title(
        "Predicted breast area ratio"
    )

    area_axis.set_xlabel(
        "Mask area / image area"
    )

    area_axis.set_ylabel("Images")
    area_axis.grid(alpha=0.25)

    component_axis = figure.add_subplot(
        grid[1, 1]
    )

    component_counts = (
        dataframe[
            "component_count"
        ]
        .clip(upper=10)
        .value_counts()
        .sort_index()
    )

    component_axis.bar(
        component_counts.index.astype(str),
        component_counts.values,
    )

    component_axis.set_title(
        "Connected-component count"
    )

    component_axis.set_xlabel(
        "Components (10 = 10+)"
    )

    component_axis.set_ylabel("Images")
    component_axis.grid(
        axis="y",
        alpha=0.25,
    )

    view_axis = figure.add_subplot(
        grid[1, 2]
    )

    view_area = (
        dataframe.groupby(
            "view_key"
        )["prediction_area_ratio"]
        .mean()
        .reindex(EXPECTED_VIEWS)
    )

    view_axis.bar(
        view_area.index,
        view_area.values,
    )

    view_axis.set_title(
        "Mean mask area by view"
    )

    view_axis.tick_params(
        axis="x",
        rotation=30,
    )

    view_axis.grid(
        axis="y",
        alpha=0.25,
    )

    intensity_axis = figure.add_subplot(
        grid[1, 3]
    )

    intensity_axis.scatter(
        dataframe[
            "border_nonzero_fraction"
        ],
        dataframe[
            "prediction_area_ratio"
        ],
        s=8,
        alpha=0.5,
    )

    intensity_axis.set_title(
        "Background/intensity domain diagnostic"
    )

    intensity_axis.set_xlabel(
        "Border non-zero fraction"
    )

    intensity_axis.set_ylabel(
        "Prediction area ratio"
    )

    intensity_axis.grid(alpha=0.25)

    samples = flagged_samples.head(8)

    for index, (_, row) in enumerate(
        samples.iterrows()
    ):
        axis = figure.add_subplot(
            grid[
                2 + index // 4,
                index % 4,
            ]
        )

        axis.imshow(
            plt.imread(
                row["overlay_path"]
            )
        )

        axis.set_title(
            f"{row['case_id']} | {row['view_key']}\n"
            f"QC={row['qc_score']:.2f} | "
            f"Area={row['prediction_area_ratio']:.3f}\n"
            f"{str(row['qc_flags'])[:70]}",
            fontsize=9,
        )

        axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(figure)


def prepare_manifest(
    manifest_path: Path,
    expected_images: int,
    expected_cases: int,
) -> pd.DataFrame:
    dataframe = pd.read_csv(
        manifest_path
    )

    required = {
        "case_id",
        "view_key",
        "image_path",
        "relative_path",
        "filename",
        "width",
        "height",
    }

    missing = sorted(
        required - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    if len(dataframe) != expected_images:
        raise ValueError(
            f"Expected {expected_images} images, "
            f"found {len(dataframe)}"
        )

    if (
        dataframe["case_id"]
        .astype(str)
        .nunique()
        != expected_cases
    ):
        raise ValueError(
            f"Expected {expected_cases} cases"
        )

    duplicated_pair = dataframe.duplicated(
        subset=[
            "case_id",
            "view_key",
        ]
    )

    if duplicated_pair.any():
        raise ValueError(
            "Duplicate case_id/view_key pairs found"
        )

    if (
        dataframe[
            "image_path"
        ]
        .astype(str)
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Duplicate image_path values found"
        )

    view_counts = (
        dataframe[
            "view_key"
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    expected_view_counts = {
        view: expected_cases
        for view in EXPECTED_VIEWS
    }

    if view_counts != expected_view_counts:
        raise ValueError(
            f"Unexpected view distribution: "
            f"{view_counts}"
        )

    case_view_counts = (
        dataframe.groupby(
            "case_id"
        )["view_key"]
        .nunique()
    )

    if (
        case_view_counts.min() != 4
        or case_view_counts.max() != 4
    ):
        raise ValueError(
            "Each case must contain exactly four views"
        )

    missing_paths = [
        path
        for path in dataframe[
            "image_path"
        ].astype(str)
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} images missing; "
            f"examples={missing_paths[:5]}"
        )

    dataframe = dataframe.copy()

    dataframe["case_id"] = (
        dataframe["case_id"].astype(str)
    )

    dataframe["view_key"] = (
        dataframe["view_key"].astype(str)
    )

    dataframe["sample_key"] = (
        dataframe["case_id"]
        + "|"
        + dataframe["view_key"]
    )

    return (
        dataframe.sort_values(
            [
                "case_id",
                "view_key",
            ]
        )
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    complete_lock = (
        args.output_dir
        / "FULL_TNMAMMO_INFERENCE_COMPLETE.json"
    )

    if complete_lock.is_file():
        print(
            "[STOP] Full TN-Mammo inference is already complete.",
            flush=True,
        )

        print(
            complete_lock.read_text(
                encoding="utf-8"
            ),
            flush=True,
        )

        return

    for path in [
        args.manifest,
        args.checkpoint,
        args.pretrained_checkpoint,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    binary_dir = (
        args.output_dir
        / "binary_masks_original_size"
    )

    probability_dir = (
        args.output_dir
        / "probability_u16_modelspace_1024"
    )

    reports_dir = (
        args.output_dir
        / "reports"
    )

    overlay_dir = (
        args.output_dir
        / "qc_overlays"
    )

    for directory in [
        binary_dir,
        probability_dir,
        reports_dir,
        overlay_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    input_manifest = prepare_manifest(
        args.manifest,
        args.expected_images,
        args.expected_cases,
    )

    print(
        "===== MANIFEST LOCKED =====",
        flush=True,
    )

    print(
        f"images={len(input_manifest)} "
        f"cases={input_manifest['case_id'].nunique()}",
        flush=True,
    )

    print(
        input_manifest[
            "view_key"
        ].value_counts().to_dict(),
        flush=True,
    )

    partial_csv = (
        reports_dir
        / "prediction_manifest.partial.csv"
    )

    final_csv = (
        reports_dir
        / "prediction_manifest.csv"
    )

    completed_keys: set[str] = set()

    if partial_csv.is_file():
        partial = pd.read_csv(
            partial_csv
        )

        valid_rows = []

        for _, row in partial.iterrows():
            binary_exists = Path(
                row[
                    "binary_prediction_path"
                ]
            ).is_file()

            probability_exists = Path(
                row[
                    "probability_modelspace_path"
                ]
            ).is_file()

            if binary_exists and probability_exists:
                valid_rows.append(row)
                completed_keys.add(
                    str(row["sample_key"])
                )

        if valid_rows:
            pd.DataFrame(
                valid_rows
            ).drop_duplicates(
                subset=["sample_key"],
                keep="last",
            ).to_csv(
                partial_csv,
                index=False,
            )
        else:
            partial_csv.unlink(
                missing_ok=True
            )

        print(
            f"[RESUME] valid completed images="
            f"{len(completed_keys)}",
            flush=True,
        )

    pending = input_manifest[
        ~input_manifest[
            "sample_key"
        ].isin(completed_keys)
    ].reset_index(drop=True)

    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for full inference"
        )

    checkpoint_payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint_payload:
        raise KeyError(
            "Checkpoint does not contain model_state_dict"
        )

    model = SAM2UNet(
        str(args.pretrained_checkpoint)
    )

    model.load_state_dict(
        checkpoint_payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    amp_enabled = bool(
        args.amp
        and device.type == "cuda"
    )

    current_batch_size = args.batch_size
    batch_fallbacks = 0
    processed_this_run = 0
    start_time = time.time()
    cursor = 0

    print(
        "===== FULL TN-MAMMO INFERENCE =====",
        flush=True,
    )

    print(
        f"pending={len(pending)} "
        f"preferred_batch={current_batch_size} "
        f"image_size={args.image_size} "
        f"threshold={args.threshold}",
        flush=True,
    )

    while cursor < len(pending):
        batch_rows = pending.iloc[
            cursor : cursor + current_batch_size
        ]

        tensors = []
        metadata_list = []
        image_arrays = []

        for _, row in batch_rows.iterrows():
            image_path = Path(
                row["image_path"]
            )

            image_pil = Image.open(
                image_path
            ).convert("RGB")

            if (
                image_pil.width
                != int(row["width"])
                or image_pil.height
                != int(row["height"])
            ):
                raise ValueError(
                    f"Manifest dimension mismatch: "
                    f"{image_path}"
                )

            tensor, metadata = (
                letterbox_image(
                    image_pil,
                    args.image_size,
                )
            )

            tensors.append(tensor)
            metadata_list.append(metadata)

            image_arrays.append(
                np.asarray(
                    image_pil,
                    dtype=np.uint8,
                )
            )

        batch_tensor = torch.stack(
            tensors,
            dim=0,
        ).to(
            device,
            non_blocking=True,
        )

        try:
            torch.cuda.synchronize()
            batch_start = time.time()

            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits, _, _ = model(
                        batch_tensor
                    )

                    logits = F.interpolate(
                        logits,
                        size=(
                            args.image_size,
                            args.image_size,
                        ),
                        mode="bilinear",
                        align_corners=False,
                    )

                    probabilities = (
                        torch.sigmoid(
                            logits
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )

            torch.cuda.synchronize()

            batch_seconds = (
                time.time() - batch_start
            )

        except torch.OutOfMemoryError:
            del batch_tensor
            torch.cuda.empty_cache()

            if current_batch_size <= 1:
                raise

            new_batch_size = max(
                1,
                current_batch_size // 2,
            )

            batch_fallbacks += 1

            print(
                f"[OOM FALLBACK] inference batch "
                f"{current_batch_size} -> {new_batch_size}",
                flush=True,
            )

            current_batch_size = new_batch_size
            continue

        rows_to_append = []

        for batch_index, (
            (_, row),
            metadata,
            rgb_image,
        ) in enumerate(
            zip(
                batch_rows.iterrows(),
                metadata_list,
                image_arrays,
            )
        ):
            probability_1024 = probabilities[
                batch_index,
                0,
            ]

            probability_original = (
                restore_probability(
                    probability_1024,
                    metadata,
                )
            )

            binary_prediction = (
                probability_original
                >= args.threshold
            ).astype(np.uint8)

            case_id = str(
                row["case_id"]
            )

            view_key = str(
                row["view_key"]
            )

            folder = case_folder(case_id)

            binary_path = (
                binary_dir
                / folder
                / f"{safe_token(view_key)}.png"
            )

            probability_path = (
                probability_dir
                / folder
                / f"{safe_token(view_key)}.png"
            )

            binary_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            probability_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            binary_ok = cv2.imwrite(
                str(binary_path),
                binary_prediction * 255,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3,
                ],
            )

            probability_u16 = np.round(
                probability_1024 * 65535
            ).astype(np.uint16)

            probability_ok = cv2.imwrite(
                str(probability_path),
                probability_u16,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    3,
                ],
            )

            if not binary_ok:
                raise RuntimeError(
                    f"Failed writing {binary_path}"
                )

            if not probability_ok:
                raise RuntimeError(
                    f"Failed writing {probability_path}"
                )

            mask_metrics = mask_statistics(
                binary_prediction
            )

            image_metrics = image_statistics(
                rgb_image
            )

            fixed_flags = fixed_qc_flags(
                mask_metrics
            )

            rows_to_append.append(
                {
                    "sample_key": row[
                        "sample_key"
                    ],
                    "case_id": case_id,
                    "view_key": view_key,
                    "image_path": str(
                        Path(
                            row["image_path"]
                        ).resolve()
                    ),
                    "relative_path": row[
                        "relative_path"
                    ],
                    "filename": row["filename"],
                    "original_width": int(
                        metadata[
                            "original_width"
                        ]
                    ),
                    "original_height": int(
                        metadata[
                            "original_height"
                        ]
                    ),
                    "model_image_size": (
                        args.image_size
                    ),
                    "threshold": (
                        args.threshold
                    ),
                    **metadata,
                    "binary_prediction_path": str(
                        binary_path.resolve()
                    ),
                    "probability_modelspace_path": str(
                        probability_path.resolve()
                    ),
                    "probability_storage": (
                        "uint16 sigmoid probability "
                        "in padded 1024x1024 model space"
                    ),
                    "postprocessing_applied": False,
                    "inference_seconds_per_image": float(
                        batch_seconds
                        / len(batch_rows)
                    ),
                    **mask_metrics,
                    **image_metrics,
                    "qc_flags_fixed": ";".join(
                        fixed_flags
                    ),
                }
            )

        append_dataframe = pd.DataFrame(
            rows_to_append
        )

        append_dataframe.to_csv(
            partial_csv,
            mode="a",
            header=not partial_csv.exists(),
            index=False,
        )

        cursor += len(batch_rows)
        processed_this_run += len(batch_rows)

        total_completed = (
            len(completed_keys)
            + processed_this_run
        )

        if (
            total_completed % 25 < len(batch_rows)
            or total_completed
            == len(input_manifest)
        ):
            elapsed = (
                time.time() - start_time
            )

            rate = (
                processed_this_run / elapsed
                if elapsed > 0
                else 0.0
            )

            print(
                f"[INFERENCE] "
                f"{total_completed}/{len(input_manifest)} "
                f"batch={current_batch_size} "
                f"rate={rate:.2f} images/s",
                flush=True,
            )

        del batch_tensor
        del probabilities
        torch.cuda.empty_cache()

    predictions = pd.read_csv(
        partial_csv
    )

    predictions = (
        predictions.drop_duplicates(
            subset=["sample_key"],
            keep="last",
        )
        .sort_values(
            [
                "case_id",
                "view_key",
            ]
        )
        .reset_index(drop=True)
    )

    if len(predictions) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} prediction rows, "
            f"found {len(predictions)}"
        )

    if (
        predictions[
            "sample_key"
        ].nunique()
        != args.expected_images
    ):
        raise RuntimeError(
            "Prediction sample_key inventory is not unique"
        )

    missing_binary = [
        path
        for path in predictions[
            "binary_prediction_path"
        ].astype(str)
        if not Path(path).is_file()
    ]

    missing_probability = [
        path
        for path in predictions[
            "probability_modelspace_path"
        ].astype(str)
        if not Path(path).is_file()
    ]

    if missing_binary or missing_probability:
        raise RuntimeError(
            "Prediction files are missing after inference"
        )

    predictions = add_statistical_qc(
        predictions
    )

    predictions.to_csv(
        final_csv,
        index=False,
    )

    case_summary = (
        predictions.groupby(
            "case_id"
        )
        .agg(
            n_views=(
                "view_key",
                "nunique",
            ),
            flagged_views=(
                "is_flagged",
                "sum",
            ),
            max_qc_score=(
                "qc_score",
                "max",
            ),
            mean_prediction_area_ratio=(
                "prediction_area_ratio",
                "mean",
            ),
            max_component_count=(
                "component_count",
                "max",
            ),
            min_largest_component_ratio=(
                "largest_component_ratio",
                "min",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "flagged_views",
                "max_qc_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    case_summary_path = (
        reports_dir
        / "case_qc_summary.csv"
    )

    case_summary.to_csv(
        case_summary_path,
        index=False,
    )

    random_generator = np.random.default_rng(
        args.seed
    )

    random_parts = []

    for view in EXPECTED_VIEWS:
        subset = predictions[
            predictions["view_key"] == view
        ]

        chosen_indices = random_generator.choice(
            subset.index.to_numpy(),
            size=6,
            replace=False,
        )

        random_parts.append(
            predictions.loc[
                chosen_indices
            ]
        )

    random_24 = pd.concat(
        random_parts,
        ignore_index=True,
    )

    flagged_24 = (
        predictions.sort_values(
            [
                "qc_score",
                "prediction_area_ratio",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(24)
        .copy()
    )

    smallest_12 = (
        predictions.sort_values(
            "prediction_area_ratio",
            ascending=True,
        )
        .head(12)
    )

    largest_12 = (
        predictions.sort_values(
            "prediction_area_ratio",
            ascending=False,
        )
        .head(12)
    )

    extreme_24 = (
        pd.concat(
            [
                smallest_12,
                largest_12,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=["sample_key"],
            keep="first",
        )
        .head(24)
    )

    random_24 = attach_overlays(
        random_24,
        overlay_dir,
    )

    flagged_24 = attach_overlays(
        flagged_24,
        overlay_dir,
    )

    extreme_24 = attach_overlays(
        extreme_24,
        overlay_dir,
    )

    random_csv = (
        reports_dir
        / "qc_random_24.csv"
    )

    flagged_csv = (
        reports_dir
        / "qc_flagged_24.csv"
    )

    extreme_csv = (
        reports_dir
        / "qc_extreme_24.csv"
    )

    random_24.to_csv(
        random_csv,
        index=False,
    )

    flagged_24.to_csv(
        flagged_csv,
        index=False,
    )

    extreme_24.to_csv(
        extreme_csv,
        index=False,
    )

    random_sheet = (
        reports_dir
        / "qc_random_24_contact_sheet.png"
    )

    flagged_sheet = (
        reports_dir
        / "qc_flagged_24_contact_sheet.png"
    )

    extreme_sheet = (
        reports_dir
        / "qc_extreme_24_contact_sheet.png"
    )

    make_contact_sheet(
        random_24,
        random_sheet,
        (
            "SAM2-UNet TN-Mammo QC — "
            "Random 24 (6 per view)"
        ),
    )

    make_contact_sheet(
        flagged_24,
        flagged_sheet,
        (
            "SAM2-UNet TN-Mammo QC — "
            "Top 24 flagged"
        ),
    )

    make_contact_sheet(
        extreme_24,
        extreme_sheet,
        (
            "SAM2-UNet TN-Mammo QC — "
            "12 smallest + 12 largest masks"
        ),
    )

    per_view_summary = {}

    for view, group in predictions.groupby(
        "view_key"
    ):
        per_view_summary[str(view)] = {
            "n": int(len(group)),
            "flagged": int(
                group["is_flagged"].sum()
            ),
            "prediction_area_ratio": (
                describe_metric(
                    group[
                        "prediction_area_ratio"
                    ]
                )
            ),
            "largest_component_ratio": (
                describe_metric(
                    group[
                        "largest_component_ratio"
                    ]
                )
            ),
            "component_count": (
                describe_metric(
                    group[
                        "component_count"
                    ]
                )
            ),
            "border_nonzero_fraction": (
                describe_metric(
                    group[
                        "border_nonzero_fraction"
                    ]
                )
            ),
        }

    flag_counts: dict[str, int] = {}

    for flags in predictions[
        "qc_flags"
    ].astype(str):
        for flag in flags.split(";"):
            if flag:
                flag_counts[flag] = (
                    flag_counts.get(
                        flag,
                        0,
                    )
                    + 1
                )

    elapsed_seconds = (
        time.time() - start_time
    )

    summary = {
        "status": "PASS",
        "stage": (
            "S8 full inference + "
            "S9 non-destructive QC + "
            "S10 output lock"
        ),
        "completed_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "n_images": int(
            len(predictions)
        ),
        "n_cases": int(
            predictions[
                "case_id"
            ].nunique()
        ),
        "view_distribution": {
            str(key): int(value)
            for key, value in predictions[
                "view_key"
            ].value_counts().to_dict().items()
        },
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_sha256": sha256_file(
            args.checkpoint
        ),
        "checkpoint_best_epoch": int(
            checkpoint_payload.get(
                "best_epoch",
                -1,
            )
        ),
        "checkpoint_best_validation_dice": float(
            checkpoint_payload.get(
                "best_val_dice",
                float("nan"),
            )
        ),
        "pretrained_checkpoint": str(
            args.pretrained_checkpoint.resolve()
        ),
        "pretrained_checkpoint_sha256": (
            sha256_file(
                args.pretrained_checkpoint
            )
        ),
        "input_manifest": str(
            args.manifest.resolve()
        ),
        "input_manifest_sha256": sha256_file(
            args.manifest
        ),
        "git_commit": git_commit(
            PROJECT_ROOT
        ),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0)
        ),
        "python_version": (
            platform.python_version()
        ),
        "image_size": args.image_size,
        "preprocessing": (
            "resize longest side to 1024, "
            "preserve aspect ratio, black padding, "
            "ImageNet normalization"
        ),
        "threshold": args.threshold,
        "threshold_provenance": (
            "locked at 0.5 before full TN-Mammo inference"
        ),
        "preferred_batch_size": args.batch_size,
        "final_batch_size": current_batch_size,
        "batch_fallback_count": batch_fallbacks,
        "postprocessing_applied": False,
        "postprocessing_note": (
            "Masks are raw thresholded model outputs. "
            "No largest-component filtering, hole filling, "
            "morphology, smoothing, or contour correction."
        ),
        "empty_predictions": int(
            (
                predictions[
                    "prediction_pixels"
                ]
                == 0
            ).sum()
        ),
        "n_flagged": int(
            predictions[
                "is_flagged"
            ].sum()
        ),
        "flag_counts": dict(
            sorted(
                flag_counts.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        ),
        "overall_statistics": {
            "prediction_area_ratio": (
                describe_metric(
                    predictions[
                        "prediction_area_ratio"
                    ]
                )
            ),
            "largest_component_ratio": (
                describe_metric(
                    predictions[
                        "largest_component_ratio"
                    ]
                )
            ),
            "component_count": (
                describe_metric(
                    predictions[
                        "component_count"
                    ]
                )
            ),
            "border_nonzero_fraction": (
                describe_metric(
                    predictions[
                        "border_nonzero_fraction"
                    ]
                )
            ),
            "inference_seconds_per_image": (
                describe_metric(
                    predictions[
                        "inference_seconds_per_image"
                    ]
                )
            ),
        },
        "per_view_statistics": per_view_summary,
        "total_runtime_seconds_this_run": float(
            elapsed_seconds
        ),
        "prediction_manifest": str(
            final_csv.resolve()
        ),
        "case_qc_summary": str(
            case_summary_path.resolve()
        ),
        "binary_mask_folder": str(
            binary_dir.resolve()
        ),
        "probability_folder": str(
            probability_dir.resolve()
        ),
        "random_24_csv": str(
            random_csv.resolve()
        ),
        "flagged_24_csv": str(
            flagged_csv.resolve()
        ),
        "extreme_24_csv": str(
            extreme_csv.resolve()
        ),
        "random_24_contact_sheet": str(
            random_sheet.resolve()
        ),
        "flagged_24_contact_sheet": str(
            flagged_sheet.resolve()
        ),
        "extreme_24_contact_sheet": str(
            extreme_sheet.resolve()
        ),
        "qc_limitations": [
            (
                "QC flags are screening heuristics, "
                "not ground-truth segmentation errors."
            ),
            (
                "No TN-Mammo mask labels are used to "
                "alter predictions or tune threshold."
            ),
            (
                "High Dice on the labeled source test "
                "does not guarantee absence of domain-shift "
                "errors on TN-Mammo."
            ),
        ],
    }

    summary_path = (
        reports_dir
        / "full_tnmammo_summary.json"
    )

    dashboard_path = (
        reports_dir
        / "full_tnmammo_qc_dashboard.png"
    )

    make_dashboard(
        predictions,
        flagged_24,
        summary,
        dashboard_path,
    )

    summary[
        "qc_dashboard"
    ] = str(
        dashboard_path.resolve()
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    lock_payload = {
        "locked": True,
        "purpose": (
            "Lock full TN-Mammo SAM2-UNet predictions "
            "and provenance before manual review or "
            "cross-model comparison."
        ),
        "completed_at": summary[
            "completed_at"
        ],
        "n_images": summary["n_images"],
        "n_cases": summary["n_cases"],
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary[
            "checkpoint_sha256"
        ],
        "input_manifest": summary[
            "input_manifest"
        ],
        "input_manifest_sha256": summary[
            "input_manifest_sha256"
        ],
        "threshold": summary["threshold"],
        "image_size": summary["image_size"],
        "postprocessing_applied": False,
        "prediction_manifest": summary[
            "prediction_manifest"
        ],
        "summary": str(
            summary_path.resolve()
        ),
        "dashboard": str(
            dashboard_path.resolve()
        ),
    }

    complete_lock.write_text(
        json.dumps(
            lock_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== FULL TN-MAMMO COMPLETE =====",
        flush=True,
    )

    print(
        json.dumps(
            {
                "n_images": summary[
                    "n_images"
                ],
                "n_cases": summary[
                    "n_cases"
                ],
                "empty_predictions": (
                    summary[
                        "empty_predictions"
                    ]
                ),
                "n_flagged": summary[
                    "n_flagged"
                ],
                "final_batch_size": (
                    summary[
                        "final_batch_size"
                    ]
                ),
                "prediction_manifest": (
                    summary[
                        "prediction_manifest"
                    ]
                ),
                "dashboard": summary[
                    "qc_dashboard"
                ],
                "flagged_contact_sheet": (
                    summary[
                        "flagged_24_contact_sheet"
                    ]
                ),
                "lock_file": str(
                    complete_lock.resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
