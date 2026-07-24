# -*- coding: utf-8 -*-
"""CLI train model E1 từ config YAML.

Cách dùng:
    python train.py --config config.yaml --output-dir outputs/run1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tn_mammo.training.engine import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_training(config, Path(args.output_dir))


if __name__ == "__main__":
    main()
