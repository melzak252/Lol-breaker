"""Test market probability accuracy using different devigging methods on OPEN and CLOSE odds.

Methods tested:
1. Multiplicative (Proportional): p_i = (1/O_i) / (1/O_a + 1/O_b)
2. Additive: p_i = (1/O_i) - (overround - 1) / 2
3. Power (Logarithmic): solves (1/O_a)^k + (1/O_b)^k = 1
4. Shin (1992): structural insider trading parameter z model
5. Unnormalized / Basic: p_i = 1 / O_i (clipped to < 1.0)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.devig_comparison import (
    devig_multiplicative,
    devig_additive,
    devig_power,
    devig_shin,
)
from src.analysis.metrics import binary_log_loss_vector

PANEL_PATH = ROOT / "data/08_reporting/thesis_lr_corrections_20260924/paired_predictions.parquet"
OUT_CSV = ROOT / "data/08_reporting/devig_methods_shootout_results.csv"

def compute_metrics(y_true, probs):
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    ll_vec = binary_log_loss_vector(y_true, p)
    log_loss = float(np.mean(ll_vec))
    brier = float(np.mean((p - y_true) ** 2))
    
    # Blowouts (LL >= 2.5, corresponding to p <= 0.082 on the winning side)
    blowouts = int(np.sum(ll_vec >= 2.5))
    
    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(y_true, p))
    except Exception:
        auc = 0.5
        
    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(p, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            prop_true = np.mean(y_true[mask])
            prop_pred = np.mean(p[mask])
            ece += np.sum(mask) * abs(prop_true - prop_pred)
    ece /= len(y_true)
    
    return {
        "logloss": log_loss,
        "brier": brier,
        "auc": auc,
        "ece_pp": ece * 100.0,
        "blowouts_ge_2_5": blowouts,
        "max_loss": float(np.max(ll_vec)),
    }

def run_devig_on_horizon(df, horizon_prefix):
    home_col = f"avg_{horizon_prefix}_home" if horizon_prefix == "open" else "avg_odds_home"
    away_col = f"avg_{horizon_prefix}_away" if horizon_prefix == "open" else "avg_odds_away"
    
    y = df["y"].to_numpy(int)
    swapped = df["market_alignment"] == "swapped"
    
    methods = {
        "Multiplicative (Proportional)": devig_multiplicative,
        "Additive": devig_additive,
        "Power (Logarithmic)": devig_power,
        "Shin (1992)": devig_shin,
    }
    
    results = {}
    for name, func in methods.items():
        probs = []
        for idx, row in df.iterrows():
            oa = float(row[home_col])
            ob = float(row[away_col])
            is_swapped = bool(swapped.loc[idx])
            try:
                pa, pb = func(oa, ob)
                # If swapped: Team 1 is away (ob), so p(Team 1) = pb
                # If same: Team 1 is home (oa), so p(Team 1) = pa
                p_t1 = pb if is_swapped else pa
                probs.append(p_t1)
            except Exception as e:
                probs.append(0.5)
        
        probs = np.array(probs, dtype=float)
        results[name] = compute_metrics(y, probs)
        
    return results

def main():
    print("Loading data...")
    df = pd.read_parquet(PANEL_PATH)
    
    # Common market subset with valid market_open and market_close
    common = df[df[["market_open", "market_close"]].notna().all(axis=1)].copy()
    print(f"Loaded {len(common)} matches with verified market odds.")
    
    rows = []
    for horizon in ["open", "close"]:
        print(f"\nEvaluating devig methods for {horizon.upper()} odds...")
        res = run_devig_on_horizon(common, horizon)
        for method_name, metrics in res.items():
            row = {"horizon": horizon.upper(), "method": method_name, **metrics}
            rows.append(row)
            
    res_df = pd.DataFrame(rows)
    print("\n" + "="*95)
    print("MARKET PROBABILITY ACCURACY ACROSS DEVIGGING METHODS (N = 3,229 matches)")
    print("="*95)
    print(res_df.to_string(index=False))
    
    # Also compare with LR and MLP on this exact same subset
    y = common["y"].to_numpy(int)
    lr_p = common["p_lr_ts_standard_seed42"].to_numpy()
    mlp_p = common["p_mlp_ts_standard_seed42"].to_numpy()
    
    lr_metrics = compute_metrics(y, lr_p)
    mlp_metrics = compute_metrics(y, mlp_p)
    
    print("\n" + "-"*95)
    print("COMPARISON WITH OUR SPORTS METAMODELS (N = 3,229 matches):")
    print("-"*95)
    print(f"LR ElasticNet: LogLoss = {lr_metrics['logloss']:.6f}, Brier = {lr_metrics['brier']:.6f}, AUC = {lr_metrics['auc']:.4f}, ECE = {lr_metrics['ece_pp']:.2f} pp, Blowouts = {lr_metrics['blowouts_ge_2_5']}")
    print(f"MLP (8 ReLU):  LogLoss = {mlp_metrics['logloss']:.6f}, Brier = {mlp_metrics['brier']:.6f}, AUC = {mlp_metrics['auc']:.4f}, ECE = {mlp_metrics['ece_pp']:.2f} pp, Blowouts = {mlp_metrics['blowouts_ge_2_5']}")
    
    res_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full devigging benchmark to {OUT_CSV}")

if __name__ == "__main__":
    main()
