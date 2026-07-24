# -*- coding: utf-8 -*-
"""Dataset 4 view đọc manifest CSV.

Manifest cần cột: case_id, label, L_CC, L_MLO, R_CC, R_MLO, [source].
Đường dẫn tương đối trong manifest được resolve theo thư mục chứa manifest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from tn_mammo.constants import LABEL_TO_INDEX, VIEW_ORDER
from tn_mammo.data.transforms import build_four_view_transform


class FourViewManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        training: bool = False,
        validate_paths: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy manifest: {self.manifest_path}")

        self.dataframe = pd.read_csv(self.manifest_path, dtype=str)
        required_columns = {"case_id", "label", *VIEW_ORDER}
        missing_columns = sorted(required_columns - set(self.dataframe.columns))
        if missing_columns:
            raise ValueError(f"Manifest thiếu cột bắt buộc: {missing_columns}")
        if self.dataframe.empty:
            raise ValueError(f"Manifest rỗng: {self.manifest_path}")
        if self.dataframe["case_id"].isna().any():
            raise ValueError("Manifest có case_id bị thiếu.")
        duplicated = self.dataframe["case_id"].duplicated(keep=False)
        if duplicated.any():
            examples = self.dataframe.loc[duplicated, "case_id"].head(5).tolist()
            raise ValueError(f"Manifest có case_id trùng, ví dụ: {examples}")

        labels = self.dataframe["label"].str.strip().str.upper()
        invalid_labels = sorted(set(labels.dropna()) - set(LABEL_TO_INDEX))
        if invalid_labels or labels.isna().any():
            raise ValueError(f"Label không hợp lệ hoặc bị thiếu: {invalid_labels}")
        self.dataframe["label"] = labels

        manifest_dir = self.manifest_path.parent
        for view in VIEW_ORDER:
            if self.dataframe[view].isna().any():
                raise ValueError(f"Manifest có đường dẫn bị thiếu ở view {view}.")
            self.dataframe[view] = self.dataframe[view].map(
                lambda value: str(
                    (manifest_dir / str(value)).resolve()
                    if not Path(str(value)).expanduser().is_absolute()
                    else Path(str(value)).expanduser().resolve()
                )
            )

        if validate_paths:
            missing_paths = [
                path
                for view in VIEW_ORDER
                for path in self.dataframe[view].tolist()
                if not Path(path).is_file()
            ]
            if missing_paths:
                preview = missing_paths[:5]
                raise FileNotFoundError(
                    f"Thiếu {len(missing_paths)} file ảnh; ví dụ: {preview}"
                )

        self.transform = build_four_view_transform(image_size, training)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict:
        row = self.dataframe.iloc[index]
        image_tensors = []
        for view in VIEW_ORDER:
            image_path = Path(str(row[view]))
            try:
                with Image.open(image_path) as image:
                    image_tensors.append(self.transform(image.convert("RGB")))
            except Exception as exc:
                raise RuntimeError(
                    f"Không thể đọc case={row['case_id']} view={view}: {image_path}"
                ) from exc

        return {
            "views": torch.stack(image_tensors),
            "label": LABEL_TO_INDEX[str(row["label"])],
            "case_id": str(row["case_id"]),
            "source": str(row.get("source", "TN")).strip(),
        }
