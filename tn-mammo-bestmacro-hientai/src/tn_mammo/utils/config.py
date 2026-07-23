from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from tn_mammo.constants import VIEW_ORDER


LOCKED_TEST_TOKENS = (
    "tn_locked_test132",
    "vindr_locked_test992",
    "locked_test",
    "test132",
    "test992",
)


def load_yaml_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    data = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Configuration root must be a mapping."
        )

    validate_config(data)

    return data


def _walk_values(
    value: Any,
) -> list[str]:
    values: list[str] = []

    if isinstance(value, Mapping):
        for item in value.values():
            values.extend(
                _walk_values(item)
            )
    elif isinstance(value, (list, tuple)):
        for item in value:
            values.extend(
                _walk_values(item)
            )
    elif isinstance(value, str):
        values.append(value)

    return values


def validate_config(
    config: Mapping[str, Any],
) -> None:
    view_order = tuple(
        config.get(
            "data",
            {},
        ).get(
            "view_order",
            (),
        )
    )

    if view_order != VIEW_ORDER:
        raise ValueError(
            "View order must be exactly "
            f"{VIEW_ORDER}; received "
            f"{view_order}."
        )

    train_and_valid = {
        "train": config.get(
            "data",
            {},
        ).get("train"),
        "validation": config.get(
            "data",
            {},
        ).get("validation"),
    }

    for value in _walk_values(
        train_and_valid
    ):
        lower = value.lower()

        if any(
            token in lower
            for token in LOCKED_TEST_TOKENS
        ):
            raise ValueError(
                "Locked-test path found in "
                f"training config: {value}"
            )
