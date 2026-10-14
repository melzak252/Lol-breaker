#!/usr/bin/env python3
"""Precompute and validate causal champion-pool features across the canonical sequence bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.champion_pool_features import (
    CHAMPION_FEATURE_NAMES,
    DIFF_FEATURE_NAMES,
    extract_champion_pool_features,
    verify_champion_features_invariants,
)


def resolve_research_root(explicit: Path | None = None) -> Path:
    """Resolve research root path following AGENTS.md conventions."""
    if explicit is not None:
        return explicit.resolve()
    env_val = os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    if env_val:
        return Path(env_val).resolve()
    root_file = PROJECT_ROOT / "data" / "research_root.txt"
    if root_file.is_file():
        text = root_file.read_text(encoding="utf-8").strip()
        if text:
            return Path(text).resolve()
    raise FileNotFoundError("Could not resolve research root via --research-root, ENSEMBLE_RESEARCH_ROOT, or data/research_root.txt")


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def prepare_champion_pool_features(
    bank_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract, validate, and persist champion-pool features."""
    t0 = time.time()
    bank_dir = bank_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    champions_path = bank_dir / "champions.npy"
    mask_path = bank_dir / "mask.npy"
    meta_path = bank_dir / "metadata.parquet"
    arrays_path = bank_dir / "arrays.npz"

    for p in (champions_path, mask_path, meta_path, arrays_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required bank file: {p}")

    print(f"Loading canonical sequence bank from {bank_dir}...")
    champions = np.load(champions_path)
    mask = np.load(mask_path)
    meta = pd.read_parquet(meta_path)
    n_series = len(meta)

    if champions.shape[0] != n_series or mask.shape[0] != n_series:
        raise ValueError(
            f"Shape mismatch: metadata has {n_series} rows, champions has {champions.shape[0]}, mask has {mask.shape[0]}"
        )

    print(f"Extracting champion-pool features for {n_series} series...")
    side_features, diff_features = extract_champion_pool_features(champions, mask)

    print("Verifying mathematical invariants and side-swap symmetry...")
    invariants = verify_champion_features_invariants(side_features, diff_features, champions, mask)
    print(f"Invariants check passed: max anti-symmetry error = {invariants['max_anti_symmetry_error']}")

    meta_sha256 = compute_sha256(meta_path)
    champions_sha256 = compute_sha256(champions_path)
    mask_sha256 = compute_sha256(mask_path)
    arrays_sha256 = compute_sha256(arrays_path)

    out_file = output_dir / "features.npz"
    print(f"Saving precomputed features to {out_file}...")
    np.savez_compressed(
        out_file,
        side_features=side_features,
        diff_features=diff_features,
        metadata_sha256=meta_sha256,
        feature_names=np.array(CHAMPION_FEATURE_NAMES),
        diff_feature_names=np.array(DIFF_FEATURE_NAMES),
    )
    out_sha256 = compute_sha256(out_file)

    elapsed = time.time() - t0
    receipt: dict[str, Any] = {
        "status": "COMPLETED",
        "elapsed_seconds": round(elapsed, 3),
        "n_series": n_series,
        "n_features_per_side": side_features.shape[-1],
        "n_diff_features": diff_features.shape[-1],
        "invariants": invariants,
        "source_hashes": {
            str(champions_path): champions_sha256,
            str(mask_path): mask_sha256,
            str(meta_path): meta_sha256,
            str(arrays_path): arrays_sha256,
        },
        "output_file": str(out_file),
        "output_sha256": out_sha256,
    }

    receipt_path = output_dir / "receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Features successfully precomputed and verified in {elapsed:.2f}s!")
    print(f"Receipt written to {receipt_path}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, help="Explicit path to release_corrected_bank")
    parser.add_argument("--research-root", type=Path, help="Explicit research root directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "04_feature" / "champion_pool",
        help="Target output directory (default: data/04_feature/champion_pool)",
    )
    args = parser.parse_args()

    if args.bank_dir is not None:
        bank_dir = args.bank_dir
    else:
        root = resolve_research_root(args.research_root)
        bank_dir = root / "a0-phase-walkforward-20260914" / "data" / "04_feature" / "release_corrected_bank"

    receipt = prepare_champion_pool_features(bank_dir, args.output_dir)
    print(json.dumps({"status": receipt["status"], "output_sha256": receipt["output_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
