# -*- coding: utf-8 -*-
"""CLI đánh giá checkpoint trên một manifest.

Cách dùng:
    python evaluate.py --checkpoint checkpoint/best_model.pt --manifest manifest.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tn_mammo.inference import run_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    run_eval(Path(args.checkpoint), Path(args.manifest))


if __name__ == "__main__":
    main()
