"""Perform TOST (Two One-Sided Tests) for statistical equivalence.

Evaluates:
1. LR vs MLP on Full Sports Cohort (N = 11,550)
2. LR vs MLP on Common Market Cohort (N = 3,229)
3. MLP 20-seed Ensemble vs LR on Full Sports Cohort (N = 11,550)
4. LR vs Market CLOSE on Common Market Cohort (N = 3,229)
5. MLP vs Market CLOSE on Common Market Cohort (N = 3,229)

Across practical equivalence margins: epsilon in [0.002, 0.003, 0.005, 0.010].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.bootstrap import monthly_block_bootstrap_delta

PANEL_PATH = ROOT / "data/08_reporting/thesis_lr_corrections_20260924/paired_predictions.parquet"
SEEDS_DIR = ROOT / "data/08_reporting/mlp_20seeds"
OUT_CSV = ROOT / "data/08_reporting/tost_results.csv"

def compute_tost(delta, date, epsilons=[0.002, 0.003, 0.004, 0.005, 0.010], n_bootstraps=10000, seed=20260926):
    """Run monthly block bootstrap and compute TOST p-values and CIs."""
    df_work = pd.DataFrame({"date": date, "delta": delta})
    observed_mean = float(delta.mean())
    
    # 90% CI is standard for TOST at alpha = 0.05
    # (equivalent to two one-sided alpha = 0.05 tests)
    # We also compute 95% CI
    months = pd.to_datetime(df_work["date"]).dt.to_period("M").astype(str).unique()
    month_stats = df_work.groupby(pd.to_datetime(df_work["date"]).dt.to_period("M").astype(str))["delta"].agg(["sum", "size"])
    delta_sums = month_stats["sum"].to_numpy(dtype=float)
    counts = month_stats["size"].to_numpy(dtype=float)
    
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstraps, dtype=float)
    for i in range(n_bootstraps):
        idx = rng.integers(0, len(month_stats), size=len(month_stats))
        samples[i] = delta_sums[idx].sum() / counts[idx].sum()
        
    ci_90 = [float(np.percentile(samples, 5.0)), float(np.percentile(samples, 95.0))]
    ci_95 = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
    
    results = {
        "observed_delta": observed_mean,
        "ci_90_lower": ci_90[0],
        "ci_90_upper": ci_90[1],
        "ci_95_lower": ci_95[0],
        "ci_95_upper": ci_95[1],
    }
    
    for eps in epsilons:
        # Test 1: H01: delta <= -eps  => p1 = P(sample <= -eps)
        p1 = float(np.mean(samples <= -eps))
        # Test 2: H02: delta >= +eps  => p2 = P(sample >= +eps)
        p2 = float(np.mean(samples >= +eps))
        # TOST p-value is max(p1, p2)
        p_tost = max(p1, p2)
        equivalent_90 = bool(ci_90[0] > -eps and ci_90[1] < eps)
        equivalent_95 = bool(ci_95[0] > -eps and ci_95[1] < eps)
        
        results[f"p_tost_eps_{eps}"] = p_tost
        results[f"equiv_90_eps_{eps}"] = equivalent_90
        results[f"equiv_95_eps_{eps}"] = equivalent_95
        
    return results

def main():
    print("Loading data...")
    panel = pd.read_parquet(PANEL_PATH)
    y = panel["y"].to_numpy(int)
    
    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))
    
    loss_lr = ll(panel["p_lr_ts_standard_seed42"].to_numpy())
    loss_mlp = ll(panel["p_mlp_ts_standard_seed42"].to_numpy())
    
    # 20-seed MLP ensemble
    preds_ens = np.zeros(len(panel), dtype=float)
    files = list(SEEDS_DIR.glob("full_seed*.parquet"))
    print(f"Found {len(files)} seeds in {SEEDS_DIR}")
    for f in files:
        preds_ens += pd.read_parquet(f)["p_calibrated"].to_numpy()
    preds_ens /= len(files)
    loss_mlp_ens = ll(preds_ens)
    
    # Market subset
    mkt_mask = panel[["market_open", "market_close"]].notna().all(axis=1).to_numpy()
    loss_close = ll(panel["market_close"].to_numpy())
    
    comparisons = [
        ("LR vs MLP (Sports, N=11550)", loss_lr - loss_mlp, panel["date"], np.ones(len(panel), dtype=bool)),
        ("MLP 20-seed Ensemble vs LR (Sports, N=11550)", loss_mlp_ens - loss_lr, panel["date"], np.ones(len(panel), dtype=bool)),
        ("LR vs MLP (Market, N=3229)", loss_lr[mkt_mask] - loss_mlp[mkt_mask], panel.loc[mkt_mask, "date"], np.ones(mkt_mask.sum(), dtype=bool)),
        ("LR vs Market CLOSE (Market, N=3229)", loss_lr[mkt_mask] - loss_close[mkt_mask], panel.loc[mkt_mask, "date"], np.ones(mkt_mask.sum(), dtype=bool)),
        ("MLP vs Market CLOSE (Market, N=3229)", loss_mlp[mkt_mask] - loss_close[mkt_mask], panel.loc[mkt_mask, "date"], np.ones(mkt_mask.sum(), dtype=bool)),
        ("MLP 20-seed Ensemble vs Market CLOSE (Market, N=3229)", loss_mlp_ens[mkt_mask] - loss_close[mkt_mask], panel.loc[mkt_mask, "date"], np.ones(mkt_mask.sum(), dtype=bool)),
    ]
    
    rows = []
    print("\nRunning TOST evaluations (10,000 bootstrap resamples per test)...")
    for name, delta, date_series, _ in comparisons:
        print(f"  Testing {name}...")
        res = compute_tost(delta, date_series, epsilons=[0.002, 0.003, 0.004, 0.005, 0.010])
        res["comparison"] = name
        rows.append(res)
        
    df_res = pd.DataFrame(rows)
    cols = ["comparison", "observed_delta", "ci_90_lower", "ci_90_upper", "ci_95_lower", "ci_95_upper",
            "p_tost_eps_0.002", "equiv_90_eps_0.002",
            "p_tost_eps_0.003", "equiv_90_eps_0.003",
            "p_tost_eps_0.005", "equiv_90_eps_0.005",
            "p_tost_eps_0.01", "equiv_90_eps_0.01"]
    
    print("\n" + "="*110)
    print("TOST EQUIVALENCE TESTING RESULTS")
    print("="*110)
    for r in rows:
        print(f"\n>>> Comparison: {r['comparison']}")
        print(f"    Observed Delta: {r['observed_delta']:+.6f}")
        print(f"    90% CI: [{r['ci_90_lower']:+.6f}; {r['ci_90_upper']:+.6f}]")
        print(f"    95% CI: [{r['ci_95_lower']:+.6f}; {r['ci_95_upper']:+.6f}]")
        print("    Equivalence tests:")
        for eps in [0.002, 0.003, 0.004, 0.005, 0.010]:
            p = r[f'p_tost_eps_{eps}']
            eq90 = r[f'equiv_90_eps_{eps}']
            eq95 = r[f'equiv_95_eps_{eps}']
            status = "EQUIVALENT (p < 0.05)" if p < 0.05 else "NOT EQUIVALENT"
            print(f"      Margin eps = {eps:.3f} (+/- {eps*1000:.1f}x10^-3): p_TOST = {p:.4f} -> {status} (90% CI inside: {eq90}, 95% CI inside: {eq95})")

    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full results to {OUT_CSV}")

if __name__ == "__main__":
    main()
