# -*- coding: utf-8 -*-
"""Hằng số dùng chung: thứ tự view, ánh xạ nhãn, kích thước feature."""
from __future__ import annotations

VIEW_ORDER = ("L_CC", "L_MLO", "R_CC", "R_MLO")
LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
NUM_CLASSES = 4
FEATURE_DIM = 1024  # đầu ra DenseNet121 sau global average pooling

# Phân bố lớp của tập train TN (411 ca) — nguồn duy nhất để tính class weight.
TN_CLASS_COUNTS = [12, 81, 178, 140]
