from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    "/mnt/hcmus/breast_vn/code/new_implement"
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from tn_mammo.training import (  # noqa: E402
    run_with_failure_record,
)
from tn_mammo.utils import (  # noqa: E402
    load_yaml_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "train",
        ],
        default="train",
    )

    parser.add_argument(
        "--smoke-train-batches",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--smoke-valid-batches",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    config = load_yaml_config(
        args.config
    )

    if args.mode == "smoke":
        max_epochs_override = 1
        max_train_batches = (
            args.smoke_train_batches
        )
        max_valid_batches = (
            args.smoke_valid_batches
        )
    else:
        max_epochs_override = None
        max_train_batches = None
        max_valid_batches = None

    print(
        "[ENTRYPOINT]",
        json.dumps({
            "config": str(
                Path(args.config).resolve()
            ),
            "output_dir": str(
                Path(
                    args.output_dir
                ).resolve()
            ),
            "mode": args.mode,
            "max_epochs_override": (
                max_epochs_override
            ),
            "max_train_batches": (
                max_train_batches
            ),
            "max_valid_batches": (
                max_valid_batches
            ),
            "test_evaluated": False,
        }),
        flush=True,
    )

    run_with_failure_record(
        config=config,
        output_dir=args.output_dir,
        max_epochs_override=(
            max_epochs_override
        ),
        max_train_batches=(
            max_train_batches
        ),
        max_valid_batches=(
            max_valid_batches
        ),
        require_cuda=True,
    )


if __name__ == "__main__":
    main()
