"""Precompute and save 8-dimensional continuous Team Archetype Vectors across pro matches."""

from __future__ import annotations

from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

research_root = Path((project_root / "data/research_root.txt").read_text().strip())

import numpy as np
import pandas as pd
from src.models.team_archetype import compute_team_archetype_vectors
from scripts.a1.train_a1 import load_research_bank


def main():
    print("=== Precomputing 8-Dimensional Team Archetype Vectors ===")
    meta_df, _, players, _, _, _ = load_research_bank(research_root)
    bank_dir = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    with np.load(bank_dir / "arrays.npz") as z:
        w20 = z["w20"]

    print(f"Loaded {len(meta_df)} matches from release-corrected bank.")
    t0 = time.time()
    arch_a, arch_b = compute_team_archetype_vectors(w20, players)
    print(f"Computed archetype vectors in {time.time() - t0:.2f}s.")

    # Verifications
    assert arch_a.shape == (len(meta_df), 8)
    assert arch_b.shape == (len(meta_df), 8)
    assert np.isfinite(arch_a).all(), "Found non-finite values in arch_a"
    assert np.isfinite(arch_b).all(), "Found non-finite values in arch_b"

    out_dir = project_root / "data/04_feature/team_archetypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "team_archetypes.npz"
    np.savez_compressed(out_file, archetype_a=arch_a, archetype_b=arch_b)
    print(f"Saved team archetype vectors to {out_file}")


if __name__ == "__main__":
    main()
