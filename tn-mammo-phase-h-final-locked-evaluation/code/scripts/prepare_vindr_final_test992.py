from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/mnt/hcmus/breast_vn/code/new_implement")
MANIFEST_DIR = ROOT / "manifests"

OUTPUT_MANIFEST = (
    MANIFEST_DIR
    / "vindr_final_eval992_phaseg.csv"
)

AUDIT_PATH = (
    MANIFEST_DIR
    / "vindr_final_eval992_audit.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lookup = {
        normalize_name(column): str(column)
        for column in frame.columns
    }

    for candidate in candidates:
        normalized = normalize_name(candidate)

        if normalized in lookup:
            return lookup[normalized]

    if required:
        raise RuntimeError(
            "Missing required column. "
            f"Candidates={candidates}; "
            f"columns={list(frame.columns)}"
        )

    return None


candidate_paths = [
    MANIFEST_DIR / "vindr_locked_test992.csv",
    MANIFEST_DIR / "phaseg_vindr_test992.csv",
    MANIFEST_DIR / "vindr_test992.csv",
    MANIFEST_DIR / "vindr_four_view_canonical.csv",
]

candidate_paths.extend(
    sorted(
        MANIFEST_DIR.glob("*vindr*test*.csv")
    )
)

candidate_paths.extend(
    sorted(
        MANIFEST_DIR.glob("*vindr*canonical*.csv")
    )
)

unique_candidates: list[Path] = []
seen: set[Path] = set()

for candidate in candidate_paths:
    candidate = candidate.resolve()

    if candidate in seen:
        continue

    seen.add(candidate)

    if candidate.is_file():
        unique_candidates.append(candidate)

selected_source: Path | None = None
selected_frame: pd.DataFrame | None = None
selection_notes: list[str] = []

for candidate in unique_candidates:
    try:
        frame = pd.read_csv(
            candidate,
            dtype=str,
        )
    except Exception as error:
        selection_notes.append(
            f"{candidate}: read error={error!r}"
        )
        continue

    working = frame.copy()

    split_column = find_column(
        working,
        [
            "split",
            "subset",
            "partition",
            "set",
        ],
        required=False,
    )

    if len(working) != 992 and split_column:
        split_values = (
            working[split_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        test_mask = split_values.isin(
            {
                "test",
                "locked_test",
                "locked-test",
                "test_locked",
            }
        )

        filtered = working.loc[
            test_mask
        ].copy()

        if len(filtered) == 992:
            working = filtered

    if len(working) == 992:
        selected_source = candidate
        selected_frame = (
            working
            .reset_index(drop=True)
        )
        break

    selection_notes.append(
        f"{candidate}: rows={len(working)}"
    )

if selected_source is None or selected_frame is None:
    raise RuntimeError(
        "Không tìm được VinDr test manifest có 992 ca. "
        f"Candidates={selection_notes}"
    )

frame = selected_frame

id_column = find_column(
    frame,
    [
        "case_id",
        "study_id",
        "exam_id",
        "patient_id",
        "study_uid",
        "id",
    ],
)

label_column = find_column(
    frame,
    [
        "label",
        "density",
        "breast_density",
        "birads_density",
        "density_label",
        "label_idx",
    ],
)

view_mapping = {
    "left_cc_path": [
        "left_cc_path",
        "left_cc",
        "l_cc_path",
        "l_cc",
        "L_CC",
        "path_l_cc",
        "image_l_cc",
    ],
    "left_mlo_path": [
        "left_mlo_path",
        "left_mlo",
        "l_mlo_path",
        "l_mlo",
        "L_MLO",
        "path_l_mlo",
        "image_l_mlo",
    ],
    "right_cc_path": [
        "right_cc_path",
        "right_cc",
        "r_cc_path",
        "r_cc",
        "R_CC",
        "path_r_cc",
        "image_r_cc",
    ],
    "right_mlo_path": [
        "right_mlo_path",
        "right_mlo",
        "r_mlo_path",
        "r_mlo",
        "R_MLO",
        "path_r_mlo",
        "image_r_mlo",
    ],
}

resolved_views: dict[str, str] = {}

for target, candidates in view_mapping.items():
    source_column = find_column(
        frame,
        candidates,
    )

    resolved_views[target] = source_column
    frame[target] = frame[source_column]


raw_labels = (
    frame[label_column]
    .astype(str)
    .str.strip()
    .str.upper()
)

unique_labels = set(
    raw_labels.unique().tolist()
)

if unique_labels.issubset(
    {"A", "B", "C", "D"}
):
    normalized_labels = raw_labels

elif unique_labels.issubset(
    {"0", "1", "2", "3", "0.0", "1.0", "2.0", "3.0"}
):
    mapping = {
        "0": "A",
        "0.0": "A",
        "1": "B",
        "1.0": "B",
        "2": "C",
        "2.0": "C",
        "3": "D",
        "3.0": "D",
    }

    normalized_labels = raw_labels.map(
        mapping
    )

elif unique_labels.issubset(
    {"1", "2", "3", "4", "1.0", "2.0", "3.0", "4.0"}
):
    mapping = {
        "1": "A",
        "1.0": "A",
        "2": "B",
        "2.0": "B",
        "3": "C",
        "3.0": "C",
        "4": "D",
        "4.0": "D",
    }

    normalized_labels = raw_labels.map(
        mapping
    )

else:
    raise RuntimeError(
        f"Unsupported VinDr labels: {sorted(unique_labels)}"
    )

frame["case_id"] = (
    frame[id_column]
    .astype(str)
    .str.strip()
)

frame["label"] = normalized_labels
frame["domain"] = "VinDr"

candidate_roots = [
    Path(
        "/mnt/hcmus/breast_vn/data/"
        "VinDrMammo/extracted_full/images_png"
    ),
    Path(
        "/mnt/hcmus/breast_vn/data/"
        "VinDrMammo"
    ),
    Path(
        "/mnt/hcmus/breast_vn/data"
    ),
    selected_source.parent,
    ROOT,
]


def resolve_path(value: str) -> str:
    text = str(value).strip()
    path = Path(text).expanduser()

    if path.is_absolute():
        return str(path)

    for root in candidate_roots:
        candidate = (
            root / path
        ).resolve()

        if candidate.is_file():
            return str(candidate)

    return str(path)


path_columns = [
    "left_cc_path",
    "left_mlo_path",
    "right_cc_path",
    "right_mlo_path",
]

for column in path_columns:
    frame[column] = (
        frame[column]
        .map(resolve_path)
    )

missing_paths: list[dict[str, object]] = []

for row_index, row in frame.iterrows():
    for column in path_columns:
        image_path = Path(
            str(row[column]).strip()
        )

        if not image_path.is_file():
            missing_paths.append(
                {
                    "row": int(row_index),
                    "column": column,
                    "path": str(image_path),
                }
            )

if missing_paths:
    print(
        "MISSING_PATH_EXAMPLES="
        f"{missing_paths[:20]}"
    )

    raise RuntimeError(
        "VinDr test có đường dẫn ảnh không tồn tại: "
        f"{len(missing_paths)}"
    )

if frame["case_id"].duplicated().any():
    raise RuntimeError(
        "VinDr test contains duplicate case IDs."
    )

if len(frame) != 992:
    raise RuntimeError(
        f"VinDr test rows={len(frame)}, expected=992."
    )

class_counts = (
    frame["label"]
    .value_counts()
    .reindex(
        ["A", "B", "C", "D"],
        fill_value=0,
    )
    .astype(int)
    .to_dict()
)

source_train_candidates = [
    MANIFEST_DIR / "phaseg_vindr_train3975.csv",
    MANIFEST_DIR / "e5_vindr_source_train_seed42.csv",
]

train_test_overlap: int | None = None
train_manifest_used: str | None = None

for train_manifest in source_train_candidates:
    if not train_manifest.is_file():
        continue

    train_frame = pd.read_csv(
        train_manifest,
        dtype=str,
    )

    train_id_column = find_column(
        train_frame,
        [
            "case_id",
            "study_id",
            "exam_id",
            "patient_id",
            "study_uid",
            "id",
        ],
    )

    train_ids = set(
        train_frame[train_id_column]
        .astype(str)
        .str.strip()
    )

    test_ids = set(
        frame["case_id"]
    )

    train_test_overlap = len(
        train_ids & test_ids
    )

    train_manifest_used = str(
        train_manifest
    )

    if train_test_overlap != 0:
        raise RuntimeError(
            "VinDr train/test overlap detected: "
            f"{train_test_overlap}"
        )

    break

preferred_columns = [
    "case_id",
    "label",
    "domain",
    "left_cc_path",
    "left_mlo_path",
    "right_cc_path",
    "right_mlo_path",
]

remaining_columns = [
    column
    for column in frame.columns
    if column not in preferred_columns
]

frame = frame[
    preferred_columns
    + remaining_columns
]

frame.to_csv(
    OUTPUT_MANIFEST,
    index=False,
)

audit = {
    "status": "PASS",
    "source_manifest": str(
        selected_source
    ),
    "source_manifest_sha256": sha256_file(
        selected_source
    ),
    "adapter_manifest": str(
        OUTPUT_MANIFEST
    ),
    "adapter_manifest_sha256": sha256_file(
        OUTPUT_MANIFEST
    ),
    "rows": int(len(frame)),
    "image_paths_verified": int(
        len(frame) * 4
    ),
    "class_counts": class_counts,
    "id_source_column": id_column,
    "label_source_column": label_column,
    "view_mapping": resolved_views,
    "train_manifest_checked": (
        train_manifest_used
    ),
    "train_test_overlap": (
        train_test_overlap
    ),
    "tn_test_metrics_used_for_selection": False,
    "vindr_test_evaluated": False,
}

AUDIT_PATH.write_text(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)

print("VINDR_TEST_MANIFEST_PREPARED=True")
print("VINDR_TEST_ROWS=992")
print("IMAGE_PATHS_VERIFIED=3968")
