"""Check CUDA/Kedro/PyTorch readiness for offline neural training."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    if not shutil.which(command[0]):
        return {"available": False, "error": f"{command[0]} not found"}
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def collect_env() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }
    try:
        import torch

        payload["torch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # pragma: no cover - environment probe
        payload["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        import kedro

        payload["kedro"] = {"available": True, "version": kedro.__version__}
    except Exception as exc:  # pragma: no cover - environment probe
        payload["kedro"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero if torch CUDA is unavailable")
    args = parser.parse_args()
    payload = collect_env()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_cuda and not payload.get("torch", {}).get("cuda_available"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
