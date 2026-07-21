from __future__ import annotations

VIEW_ORDER: tuple[str, ...] = (
    "L_CC",
    "L_MLO",
    "R_CC",
    "R_MLO",
)

LABEL_TO_INDEX: dict[str, int] = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
}

INDEX_TO_LABEL: dict[int, str] = {
    value: key
    for key, value in LABEL_TO_INDEX.items()
}

NUM_CLASSES = 4
NUM_ORDINAL_THRESHOLDS = NUM_CLASSES - 1
FEATURE_DIM = 1024
