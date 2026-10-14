"""Benchmark Team Archetype & Style Clash Vectors against Canonical Test Cohort (N = 11,550).

Extracts the 8-dimensional macroeconomic playstyle vectors for both teams:
1. Pace / Bloodiness (kills + deaths per minute)
2. Scaling Bias (game duration normalized)
3. Early Lane Dominance (GD@15 in k gold)
4. Objective Priority (Dragon vs Turret bias)
5. Vision Intensity (VSPM normalized)
6. Combat DPM (DPM normalized)
7. Role Focus Skew (Top vs Bot rating differential)
8. Gold Efficiency (DPM per 1k gold)

Evaluates out-of-sample impact on both Calibrated Glicko-2 and Causal A0.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

research_root = Path((project_root / "data/research_root.txt").read_text().strip())

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.team_archetype import compute_team_archetype_vectors
from scripts.a1.train_a1 import load_research_bank


def main():
    print("=== Team Archetype & Style Clash Benchmark Pipeline ===")
    meta_df, _, players, _, _, _ = load_research_bank(research_root)
    bank_dir = research_root / "a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank"
    with np.load(bank_dir / "arrays.npz") as z:
        w20 = z["w20"]

    # Precompute archetype vectors
    arch_a, arch_b = compute_team_archetype_vectors(w20, players)
    diff_arch = arch_a - arch_b

    # Load canonical benchmark targets
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)

    meta_df["idx"] = np.arange(len(meta_df))
    id_to_bank_idx = {str(row["golgg_match_id"]): i for i, row in meta_df.iterrows()}
    test_bank_indices = [id_to_bank_idx[str(gid)] for gid in paired_df["golgg_match_id"]]

    X_test_arch = diff_arch[test_bank_indices]  # (11550, 8)
    y_test = paired_df["y"].values.astype(float)

    p_glicko = paired_df["p_glicko"].values
    z_glicko = np.log(p_glicko / (1.0 - p_glicko))

    p_a0 = paired_df["p_full"].values
    z_a0 = np.log(p_a0 / (1.0 - p_a0))

    paired_df["year"] = pd.to_datetime(paired_df["date"]).dt.year
    mask_2024 = (paired_df["year"] == 2024).values
    mask_holdout = (paired_df["year"] >= 2025).values

    # Fit style clash weights strictly on 2024 (N = 4,642)
    X_fit = X_test_arch[mask_2024]
    y_fit = y_test[mask_2024]
    z_fit_gl = z_glicko[mask_2024]

    def loss_fn(w, alpha_l2=5.0):
        z_pred = z_fit_gl + X_fit @ w
        p_pred = 1.0 / (1.0 + np.exp(-np.clip(z_pred, -20, 20)))
        bce = -np.mean(y_fit * np.log(np.clip(p_pred, 1e-6, 1-1e-6)) + (1 - y_fit) * np.log(np.clip(1 - p_pred, 1e-6, 1-1e-6)))
        return bce + alpha_l2 * np.sum(w ** 2)

    res = minimize(loss_fn, np.zeros(8), method="L-BFGS-B")
    w_style = res.x
    print("Fitted Style Clash Weights (from 2024):", w_style.round(5))

    # Apply across all 11,550 matches
    z_pred_gl = z_glicko + X_test_arch @ w_style
    p_pred_gl = 1.0 / (1.0 + np.exp(-np.clip(z_pred_gl, -20, 20)))
    p_pred_gl = np.clip(p_pred_gl, 1e-6, 1.0 - 1e-6)

    # Save candidate DataFrame
    candidate_cols = ["golgg_match_id", "team1_id", "team2_id", "date", "result_day", "best_of", "y", "feature_source_max_day"]
    cand_df = paired_df[candidate_cols].copy()
    cand_df["train_end"] = "2023-12-31"
    cand_df["calibration_end"] = "2023-12-31"
    cand_df["p_style"] = p_pred_gl

    out_dir = project_root / "data/07_model_output/team_archetypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "team_style_clash_predictions.parquet"
    cand_df.to_parquet(out_file, index=False)
    print(f"Saved candidate predictions ({len(cand_df)} rows) to {out_file}")

    # Holdout metrics
    ll_orig_ho = -np.mean(y_test[mask_holdout] * np.log(p_glicko[mask_holdout]) + (1 - y_test[mask_holdout]) * np.log(1 - p_glicko[mask_holdout]))
    ll_style_ho = -np.mean(y_test[mask_holdout] * np.log(p_pred_gl[mask_holdout]) + (1 - y_test[mask_holdout]) * np.log(1 - p_pred_gl[mask_holdout]))
    print(f"\n2025-2026 Holdout Evaluation (N = {np.sum(mask_holdout)}):")
    print(f"  Baseline Glicko-2: LogLoss = {ll_orig_ho:.6f}")
    print(f"  Glicko-2 + Style:  LogLoss = {ll_style_ho:.6f}")
    print(f"  Delta LogLoss:     {ll_style_ho - ll_orig_ho:+.6f}")


if __name__ == "__main__":
    main()
