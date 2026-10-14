"""Evaluate Option A: Hierarchical Coverage-Gated Routing on full locked cohort."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.model_benchmark import compute_benchmark_summary


def load_and_align_data() -> pd.DataFrame:
    research_root = Path("data/research_root.txt").read_text().strip()
    paired_path = Path(research_root) / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    l79_path = Path("data/artifacts/corrected-historical-reruns-20260908/architecture-original-protocol/predictions.csv")

    paired = pd.read_parquet(paired_path)
    l79 = pd.read_csv(l79_path)

    paired["golgg_match_id"] = paired["golgg_match_id"].astype(str)
    paired["team1_id"] = paired["team1_id"].astype(str)
    paired["team2_id"] = paired["team2_id"].astype(str)

    l79["golgg_match_id"] = l79["golgg_match_id"].astype(str)
    l79["team1_id"] = l79["team1_id"].astype(str)
    l79["team2_id"] = l79["team2_id"].astype(str)

    merged = paired.merge(
        l79[["golgg_match_id", "team1_id", "team2_id", "linear79"]],
        on="golgg_match_id",
        how="left",
        suffixes=("", "_l79"),
    )

    is_rev = (merged["team1_id"] == merged["team2_id_l79"]) & (merged["team2_id"] == merged["team1_id_l79"])
    is_same = (merged["team1_id"] == merged["team1_id_l79"]) & (merged["team2_id"] == merged["team2_id_l79"])

    merged["p_l79"] = np.nan
    merged.loc[is_same, "p_l79"] = merged.loc[is_same, "linear79"]
    merged.loc[is_rev, "p_l79"] = 1.0 - merged.loc[is_rev, "linear79"]

    return merged


def run_sweep(
    merged: pd.DataFrame,
    secondary_col: str,
    space: str,  # "prob" or "logit"
    weights: list[float],
    n_bootstraps: int = 5000,
    random_seed: int = 20260915,
) -> pd.DataFrame:
    y = merged["y"].to_numpy(int)
    p_a0 = merged["p_full"].to_numpy(float)
    dates = merged["date"]
    p_sec = merged[secondary_col].to_numpy(float)

    has_sec = ~np.isnan(p_sec)
    n_total = len(y)
    n_has_sec = int(has_sec.sum())

    # Precalculate baseline A0 loss
    ll_a0 = -(y * np.log(p_a0) + (1 - y) * np.log(1.0 - p_a0))

    logit_a0 = logit(np.clip(p_a0, 1e-15, 1.0 - 1e-15))
    logit_sec = logit(np.clip(p_sec, 1e-15, 1.0 - 1e-15))

    results = []

    for w in weights:
        p_blend = p_a0.copy()

        if w == 1.0:
            # Pure A0
            pass
        else:
            if space == "prob":
                p_blend[has_sec] = w * p_a0[has_sec] + (1.0 - w) * p_sec[has_sec]
            elif space == "logit":
                blend_logits = w * logit_a0[has_sec] + (1.0 - w) * logit_sec[has_sec]
                p_blend[has_sec] = expit(blend_logits)
            else:
                raise ValueError(f"Unknown space: {space}")

        # Compute metrics on full cohort
        summary = compute_benchmark_summary(y, p_blend, probability_epsilon=None)
        ll_vec = -(y * np.log(p_blend) + (1 - y) * np.log(1.0 - p_blend))
        tail25 = int((ll_vec >= 2.5).sum())

        if w == 1.0:
            delta_obs = 0.0
            ci_lower = 0.0
            ci_upper = 0.0
            p_nonneg = 1.0
        else:
            delta_df = pd.DataFrame({"date": dates, "delta": ll_vec - ll_a0})
            delta_obs, ci_lower, ci_upper, samples = monthly_block_bootstrap_delta(
                delta_df,
                "delta",
                n_bootstraps=n_bootstraps,
                random_seed=random_seed,
            )
            p_nonneg = float(np.mean(samples >= 0.0))

        results.append({
            "model_combo": f"A0 + {secondary_col.replace('p_', '')}",
            "space": space,
            "w_a0": w,
            "w_secondary": round(1.0 - w, 2),
            "n_total": n_total,
            "n_covered": n_has_sec,
            "coverage_pct": round(100.0 * n_has_sec / n_total, 2),
            "log_loss": summary.log_loss,
            "brier": summary.brier_score,
            "ece": summary.ece,
            "calibration_slope": summary.calibration_slope,
            "tail_2_5": tail25,
            "delta_ll": delta_obs,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "p_delta_nonneg": p_nonneg,
            "significant_improvement": bool(ci_upper < 0.0),
        })

    return pd.DataFrame(results)


def main():
    merged = load_and_align_data()
    weights = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]

    sweeps = [
        ("p_l79", "prob"),
        ("p_l79", "logit"),
        ("p_039", "prob"),
        ("p_039", "logit"),
    ]

    all_dfs = []
    for sec_col, space in sweeps:
        print(f"Running sweep for {sec_col} in {space}-space...")
        df = run_sweep(merged, sec_col, space, weights)
        all_dfs.append(df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    out_path = Path("reports/benchmarks_2026/hierarchical_routing_results.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(out_path)
    final_df.to_csv(out_path.with_suffix(".csv"), index=False)

    print("\n--- RESULTS SUMMARY ---")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
