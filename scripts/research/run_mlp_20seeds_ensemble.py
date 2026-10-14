"""Run 20-seed ensemble analysis for MLP (Full vs No Context)."""
import json
import multiprocessing
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.bootstrap import monthly_block_bootstrap_delta
from src.analysis.thesis_audit import add_series_features, feature_variants
from src.analysis.thesis_models import replay_monthly

OUT_DIR = ROOT / "data/08_reporting/mlp_20seeds"

def get_research_root():
    import os
    env_root = os.environ.get("ENSEMBLE_RESEARCH_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    root_txt = ROOT / "data/research_root.txt"
    if root_txt.exists():
        p = Path(root_txt.read_text().strip())
        if p.exists():
            return p
    return ROOT

RESEARCH_ROOT = get_research_root()
FEATURE_PATH = RESEARCH_ROOT / "data/04_feature/thesis_audit_20260921/features.parquet"
SELECTION_PATH = RESEARCH_ROOT / "data/08_reporting/thesis_audit_20260921/corrected_replay_resumed/selection.json"
PANEL_PATH = ROOT / "data/08_reporting/thesis_lr_corrections_20260924/paired_predictions.parquet"
# Generate 20 seeds: anchor 3 known seeds + 17 random reproducible seeds
rng = np.random.default_rng(20260925)
RANDOM_SEEDS = [7, 42, 2026] + sorted(int(s) for s in rng.integers(100, 99999, size=17))

def run_job(job_spec):
    name, seed, variant_type = job_spec
    target_parquet = OUT_DIR / f"{name}_seed{seed}.parquet"
    if target_parquet.exists():
        return name, seed, variant_type, 0.0, True

    data = add_series_features(pd.read_parquet(FEATURE_PATH))
    sel = json.load(open(SELECTION_PATH))
    mlp_params = {**sel["selected"]["mlp"], "random_state": seed}
    variants = feature_variants()
    features = variants["w20"] if variant_type == "full" else variants["player_only"]

    t0 = time.time()
    preds, _ = replay_monthly(
        data, features, "mlp", mlp_params,
        start="2024-01-01", calibration_start="2024-01-01"
    )
    dt = time.time() - t0
    preds[[preds.columns[0], "p_calibrated"]].to_parquet(target_parquet, index=False)
    return name, seed, variant_type, dt, False

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    y = panel["y"].to_numpy(int)
    date = panel["date"]

    # Pre-populate known seeds 7, 42, 2026 if already available
    for s in [7, 42, 2026]:
        f_p = OUT_DIR / f"full_seed{s}.parquet"
        if not f_p.exists():
            col = f"p_mlp_original_seed{s}"
            panel[[panel.columns[0], col]].rename(columns={col: "p_calibrated"}).to_parquet(f_p, index=False)

    nc_42 = OUT_DIR / "no_context_seed42.parquet"
    if not nc_42.exists():
        panel[[panel.columns[0], "p_supp_primary_no_context"]].rename(columns={"p_supp_primary_no_context": "p_calibrated"}).to_parquet(nc_42, index=False)

    # Check already computed seeds in data/08_reporting/mlp_seed_ablations.csv
    prev_abl = ROOT / "data/08_reporting/mlp_seed_ablations.csv"

    jobs = []
    for seed in RANDOM_SEEDS:
        for vtype in ["full", "no_context"]:
            target_p = OUT_DIR / f"{vtype}_seed{seed}.parquet"
            if not target_p.exists():
                jobs.append((vtype, seed, vtype))

    print(f"Total seeds: {len(RANDOM_SEEDS)}: {RANDOM_SEEDS}")
    print(f"Remaining jobs to compute: {len(jobs)}")

    if jobs:
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_job, j) for j in jobs]
            for f in as_completed(futures):
                name, seed, vtype, dt, reused = f.result()
                status = "REUSED" if reused else f"{dt:.1f}s"
                print(f"  [{status}] {name} seed={seed}", flush=True)

    print("\nAll 20 seeds computed! Evaluating ensembles and individual metrics...")

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    # Load all 20 predictions
    preds_full = {}
    preds_no_context = {}
    indiv_rows = []

    for seed in RANDOM_SEEDS:
        p_f = pd.read_parquet(OUT_DIR / f"full_seed{seed}.parquet")["p_calibrated"].to_numpy()
        p_nc = pd.read_parquet(OUT_DIR / f"no_context_seed{seed}.parquet")["p_calibrated"].to_numpy()
        preds_full[seed] = p_f
        preds_no_context[seed] = p_nc

        l_f = ll(p_f).mean()
        l_nc = ll(p_nc).mean()
        delta = l_nc - l_f
        df_work = pd.DataFrame({"date": date, "delta": ll(p_nc) - ll(p_f)})
        _, lo, hi, sam = monthly_block_bootstrap_delta(df_work, "delta", n_bootstraps=2000, random_seed=seed)

        indiv_rows.append({
            "seed": seed, "ll_full": l_f, "ll_no_context": l_nc,
            "delta": delta, "ci_lower": lo, "ci_upper": hi,
            "fraction_positive": float(np.mean(sam >= 0))
        })

    indiv_df = pd.DataFrame(indiv_rows)
    print("\n--- INDIVIDUAL SEED RESULTS (20 SEEDS) ---")
    print(indiv_df.to_string(index=False))
    indiv_df.to_csv(OUT_DIR / "individual_20seeds.csv", index=False)

    # Cumulative ensemble analysis M = 1, 2, 3, 5, 10, 15, 20
    ens_rows = []
    cum_full = np.zeros(len(y), dtype=float)
    cum_nc = np.zeros(len(y), dtype=float)

    for i, seed in enumerate(RANDOM_SEEDS, 1):
        cum_full += preds_full[seed]
        cum_nc += preds_no_context[seed]

        if i in [1, 2, 3, 5, 10, 15, 20]:
            ens_f = cum_full / i
            ens_nc = cum_nc / i
            l_f = ll(ens_f).mean()
            l_nc = ll(ens_nc).mean()
            delta = l_nc - l_f
            df_work = pd.DataFrame({"date": date, "delta": ll(ens_nc) - ll(ens_f)})
            _, lo, hi, sam = monthly_block_bootstrap_delta(df_work, "delta", n_bootstraps=5000, random_seed=20260925)

            ens_rows.append({
                "ensemble_size_M": i,
                "ll_full": l_f,
                "ll_no_context": l_nc,
                "delta": delta,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": hi - lo,
                "fraction_positive": float(np.mean(sam >= 0))
            })

    ens_df = pd.DataFrame(ens_rows)
    print("\n" + "="*80)
    print("--- CUMULATIVE ENSEMBLE RESULTS (M = 1, 2, 3, 5, 10, 15, 20) ---")
    print("="*80)
    print(ens_df.to_string(index=False))
    ens_df.to_csv(OUT_DIR / "cumulative_ensemble.csv", index=False)
    print(f"\nSaved all results to {OUT_DIR}")

if __name__ == "__main__":
    main()
