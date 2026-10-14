"""Evaluate MLP ablations (no_context, no_binomial) across multiple random seeds."""
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.thesis_audit import add_series_features, feature_variants
from src.analysis.thesis_models import replay_monthly

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [7, 42, 2026]

def run_single_ablation(data, features, family, params, seed, name):
    t0 = time.time()
    effective_params = {**params, "random_state": seed}
    preds, _ = replay_monthly(
        data, features, family, effective_params,
        start="2024-01-01", calibration_start="2024-01-01"
    )
    dt = time.time() - t0
    return name, seed, preds, dt

def main():
    feature_path = "/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a/data/04_feature/thesis_audit_20260921/features.parquet"
    selection_path = "/home/melzak/.codex/.chatgpt-projects/g-p-6944b69bdee08191acfcfeb7a38f372a/data/08_reporting/thesis_audit_20260921/corrected_replay_resumed/selection.json"
    panel_path = ROOT / "data/08_reporting/thesis_lr_corrections_20260924/paired_predictions.parquet"

    print("Loading data...")
    data = add_series_features(pd.read_parquet(feature_path))
    panel = pd.read_parquet(panel_path)
    y = panel["y"].to_numpy(int)
    date = panel["date"]

    sel = json.load(open(selection_path))
    mlp_params = sel["selected"]["mlp"]
    variants = feature_variants()

    features_w20 = variants["w20"]
    features_no_context = variants["player_only"]
    features_no_bon = [f for f in features_w20 if not f.endswith("_binom_series")]

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    jobs = []
    # For seed 42 no_context and no_bon, we already have them in panel:
    # panel['p_supp_primary_no_context'], panel['p_supp_primary_no_binomial']
    # But for seeds 7 and 2026 we need to fit them.
    for seed in [7, 2026]:
        jobs.append((data, features_no_context, "mlp", mlp_params, seed, "no_context"))
        jobs.append((data, features_no_bon, "mlp", mlp_params, seed, "no_bon"))

    results = {}
    # Populate seed 42 from panel
    results[("no_context", 42)] = panel["p_supp_primary_no_context"].to_numpy()
    results[("no_bon", 42)] = panel["p_supp_primary_no_binomial"].to_numpy()

    print(f"Running {len(jobs)} ablation fits across seeds {SEEDS} (workers=2)...")
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_single_ablation, *job) for job in jobs]
        for f in as_completed(futures):
            name, seed, preds, dt = f.result()
            print(f"  [DONE] {name} seed={seed} in {dt:.1f}s")
            results[(name, seed)] = preds["p_calibrated"].to_numpy()

    print("\n" + "="*80)
    print("ANALYSIS OF MLP ABLATIONS ACROSS SEEDS")
    print("="*80)

    rows = []
    for seed in SEEDS:
        full_col = f"p_mlp_original_seed{seed}"
        p_full = panel[full_col].to_numpy()
        loss_full = ll(p_full)
        mean_ll_full = loss_full.mean()

        for name in ["no_context", "no_bon"]:
            p_abl = results[(name, seed)]
            loss_abl = ll(p_abl)
            mean_ll_abl = loss_abl.mean()
            delta = loss_abl - loss_full
            mean_delta = delta.mean()

            df_work = pd.DataFrame({"date": date, "delta": delta})
            _, ci_lo, ci_hi, sam = monthly_block_bootstrap_delta(
                df_work, "delta", n_bootstraps=5000, random_seed=20260921
            )
            frac_pos = float(np.mean(sam >= 0))
            ci_width = ci_hi - ci_lo

            rows.append({
                "ablation": name,
                "seed": seed,
                "ll_full": mean_ll_full,
                "ll_abl": mean_ll_abl,
                "delta": mean_delta,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "ci_width": ci_width,
                "fraction_positive": frac_pos
            })

    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

    out_csv = ROOT / "data/08_reporting/mlp_seed_ablations.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

if __name__ == "__main__":
    main()
