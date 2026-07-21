#!/usr/bin/env python3
"""Environment audit for TN-Mammo Task 1.

This script is intentionally read-only with respect to datasets and checkpoints.
It records package, GPU, Git and configuration state before any training begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_IMPORTS = (
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "PIL",
    "sklearn",
    "yaml",
    "matplotlib",
    "pytest",
)
OPTIONAL_IMPORTS = ("pydicom", "torchmetrics")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
        return {
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # audit must report rather than terminate the user shell
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def import_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_IMPORTS + OPTIONAL_IMPORTS:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "available"))
        except Exception as exc:
            versions[name] = f"UNAVAILABLE: {exc!r}"
            if name in REQUIRED_IMPORTS:
                missing.append(name)
    return versions, missing


def git_state(root: Path) -> dict[str, Any]:
    top = run_text(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
    if top["returncode"] != 0:
        return {"is_git_repo": False, "details": top}
    repo = Path(top["stdout"])
    return {
        "is_git_repo": True,
        "root": str(repo),
        "commit": run_text(["git", "-C", str(repo), "rev-parse", "HEAD"])["stdout"],
        "status_porcelain": run_text(["git", "-C", str(repo), "status", "--porcelain"])["stdout"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, action="append", default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions, missing = import_versions()

    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": str(torch.version.cuda),
            "device_count_visible": int(torch.cuda.device_count()),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        cuda = {"available": False, "error": repr(exc)}

    config_hashes = {}
    for path in args.config:
        if path.is_file():
            config_hashes[str(path.resolve())] = sha256_file(path)

    payload = {
        "document_contract": "TNM-CS-T1-2026-01",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(args.root.resolve()),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CONDA_DEFAULT_ENV",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "packages": versions,
        "missing_required_packages": missing,
        "cuda": cuda,
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git": git_state(args.root),
        "config_sha256": config_hashes,
        "implementation_references": {
            "Supervisor-Skills": "https://github.com/HKUSTDial/Supervisor-Skills",
            "coral-pytorch": "https://github.com/Raschka-research-group/coral-pytorch",
            "torchvision-densenet": "https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py",
            "timm": "https://github.com/huggingface/pytorch-image-models",
            "NYU-BIRADS": "https://github.com/nyukat/BIRADS_classifier",
            "NYU-four-view": "https://github.com/nyukat/breast_cancer_classifier",
            "classification-uncertainty": "https://github.com/dougbrion/pytorch-classification-uncertainty",
            "temperature-scaling": "https://github.com/gpleiss/temperature_scaling",
            "torchmetrics": "https://github.com/Lightning-AI/torchmetrics",
            "awesome-breast-imaging": "https://github.com/batmanlab/awesome-breast-imaging-resources",
            "mammographic-density-classification": "https://github.com/lich0031/Mammographic_Density_Classification",
            "two-views-classifier": "https://github.com/dpetrini/two-views-classifier",
            "mv-swin-t": "https://github.com/prithuls/mv-swin-t",
            "multiview-mamo": "https://github.com/levi3001/multiview-mamo",
            "coral-cnn": "https://github.com/Raschka-research-group/coral-cnn",
            "manifold-mixup": "https://github.com/vikasverma1077/manifold_mixup"
        },
        "policy": {
            "physical_gpu": 0,
            "selection_metric": "TN validation macro-F1",
            "locked_test_enabled": False,
            "view_order": ["L_CC", "L_MLO", "R_CC", "R_MLO"],
            "label_map": {"A": 0, "B": 1, "C": 2, "D": 3},
        },
    }
    payload["status"] = "PASS" if not missing else "FAIL_MISSING_DEPENDENCIES"

    output = args.output_dir / "input_audit.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))
    if missing:
        raise RuntimeError(f"Missing required packages: {missing}")


if __name__ == "__main__":
    main()
