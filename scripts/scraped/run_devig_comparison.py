"""Benchmark suite comparing devigging methods across scraped production matches.

Compares:
1. Multiplicative (Proportional)
2. Power Method (Logarithmic)
3. Additive Method
4. Basic (Raw Implied, Normalized)
5. Shin (1992) Method

Across three market horizons:
- OPEN (Earliest available quotes)
- 24h Before Match ([14h, 36h])
- CLOSE (Latest quotes < 2h before kickoff)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.analysis.devig_comparison import (
    devig_multiplicative,
    devig_power,
    devig_additive,
    devig_shin,
)

db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@timescaledb:5432/betting",
)
engine = create_engine(db_url)


def main():
    print("=== Extracting Scraped Production Data for Devig Comparison ===")
    with engine.connect() as conn:
        matches = pd.read_sql(
            text("""
            SELECT id as canonical_match_id, team_a_name, team_b_name, winner_side,
                   start_time_normalized, league, best_of
            FROM canonical_matches
            WHERE status IN ('finished', 'completed') AND winner_side IN ('team_a', 'team_b')
            ORDER BY start_time_normalized ASC
            """),
            conn,
        )

        odds = pd.read_sql(
            text("""
            SELECT os.canonical_match_id, b.name as bookmaker, os.odds_a, os.odds_b, os.scraped_at
            FROM odds_snapshots os
            JOIN bookmakers b ON b.id = os.bookmaker_id
            WHERE os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1.0 AND os.odds_b > 1.0
            ORDER BY os.scraped_at ASC
            """),
            conn,
        )

    print(f"Loaded {len(matches)} matches, {len(odds)} odds snapshots.")

    df = odds.merge(matches[["canonical_match_id", "start_time_normalized"]], on="canonical_match_id", how="inner")
    df["start_dt"] = pd.to_datetime(df["start_time_normalized"], format="mixed", utc=True)
    df["scraped_dt"] = pd.to_datetime(df["scraped_at"], format="mixed", utc=True)
    df = df[df["scraped_dt"] < df["start_dt"]].copy()
    df["hours_before"] = (df["start_dt"] - df["scraped_dt"]).dt.total_seconds() / 3600.0

    print("Devigging every snapshot via Multiplicative, Power, Additive, and Shin...")
    p_mult_list = []
    p_power_list = []
    p_add_list = []
    p_shin_list = []

    for _, r in df.iterrows():
        oa = float(r["odds_a"])
        ob = float(r["odds_b"])

        # 1. Multiplicative
        pm_a, _ = devig_multiplicative(oa, ob)
        p_mult_list.append(pm_a)

        # 2. Power
        pp_a, _ = devig_power(oa, ob)
        p_power_list.append(pp_a)

        # 3. Additive
        pa_a, _ = devig_additive(oa, ob)
        p_add_list.append(pa_a)

        # 4. Shin
        try:
            ps_a, _ = devig_shin(oa, ob)
        except Exception:
            ps_a = pm_a
        p_shin_list.append(ps_a)

    df["p_mult"] = p_mult_list
    df["p_power"] = p_power_list
    df["p_add"] = p_add_list
    df["p_shin"] = p_shin_list

    # Metric evaluation helper
    def evaluate_probabilities(p_series, y_series, name="Method"):
        p = np.clip(p_series.values.astype(float), 1e-6, 1.0 - 1e-6)
        y = y_series.values.astype(float)
        ll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        brier = np.mean((p - y) ** 2)
        acc = np.mean((p >= 0.5) == (y == 1.0))
        p_win = np.where(y == 1.0, p, 1.0 - p)
        blowouts = int(np.sum(-np.log(p_win) >= 2.5))
        # Calibration slope via linear regression on log-odds
        from scipy.stats import linregress
        log_odds = np.log(p / (1.0 - p))
        slope = linregress(log_odds, y).slope if len(y) > 10 else 1.0
        return {
            "method": name,
            "log_loss": round(float(ll), 4),
            "brier": round(float(brier), 4),
            "accuracy_pct": round(float(acc * 100), 1),
            "cal_slope": round(float(slope), 3),
            "blowouts": blowouts,
        }

    # -------------------------------------------------------------------------
    # HORIZON 1: Market OPEN Consensus
    # -------------------------------------------------------------------------
    open_df = df.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).first().reset_index()
    open_cons = open_df.groupby("canonical_match_id").agg(
        p_mult=("p_mult", "mean"),
        p_power=("p_power", "mean"),
        p_add=("p_add", "mean"),
        p_shin=("p_shin", "mean"),
        hours_before=("hours_before", "mean"),
    ).reset_index()

    cohort_open = matches.merge(open_cons, on="canonical_match_id", how="inner")
    y_open = np.where(cohort_open["winner_side"] == "team_a", 1.0, 0.0)
    y_open_s = pd.Series(y_open)

    # -------------------------------------------------------------------------
    # HORIZON 2: 24h Before Kickoff ([14h, 36h])
    # -------------------------------------------------------------------------
    df_24h = df[(df["hours_before"] >= 14.0) & (df["hours_before"] <= 36.0)].copy()
    h24_cons = df_24h.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    h24_cons = h24_cons.groupby("canonical_match_id").agg(
        p_mult=("p_mult", "mean"),
        p_power=("p_power", "mean"),
        p_add=("p_add", "mean"),
        p_shin=("p_shin", "mean"),
        hours_before=("hours_before", "mean"),
    ).reset_index()

    cohort_24h = matches.merge(h24_cons, on="canonical_match_id", how="inner")
    y_24h = np.where(cohort_24h["winner_side"] == "team_a", 1.0, 0.0)
    y_24h_s = pd.Series(y_24h)

    # -------------------------------------------------------------------------
    # HORIZON 3: Market CLOSE Consensus (<= 2h)
    # -------------------------------------------------------------------------
    df_close = df[df["hours_before"] <= 2.0].copy()
    close_cons = df_close.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    close_cons = close_cons.groupby("canonical_match_id").agg(
        p_mult=("p_mult", "mean"),
        p_power=("p_power", "mean"),
        p_add=("p_add", "mean"),
        p_shin=("p_shin", "mean"),
        hours_before=("hours_before", "mean"),
    ).reset_index()

    cohort_close = matches.merge(close_cons, on="canonical_match_id", how="inner")
    y_close = np.where(cohort_close["winner_side"] == "team_a", 1.0, 0.0)
    y_close_s = pd.Series(y_close)

    print(f"\nCohorts: OPEN N={len(cohort_open)}, 24h N={len(cohort_24h)}, CLOSE N={len(cohort_close)}")

    methods = [
        ("Multiplicative (Proportional)", "p_mult"),
        ("Power Method (Logarithmic)", "p_power"),
        ("Additive Method", "p_add"),
        ("Shin (1992) Method", "p_shin"),
    ]

    results = {}

    for horizon_name, c_df, y_s in [("OPEN (Early Quotes)", cohort_open, y_open_s),
                                     ("24h Window (14h-36h)", cohort_24h, y_24h_s),
                                     ("CLOSE (< 2h Kickoff)", cohort_close, y_close_s)]:
        print("\n" + "=" * 95)
        print(f"DEVIGGING METHOD COMPARISON: {horizon_name} (N = {len(c_df)} Matches)")
        print("=" * 95)
        print(f"{'Devigging Method':<30} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'CalSlope':<8} | {'Blowouts (>=2.5)':<16}")
        print("-" * 95)
        horizon_results = []
        for label, col in methods:
            m = evaluate_probabilities(c_df[col], y_s, name=label)
            horizon_results.append(m)
            print(f"{label:<30} | {m['log_loss']:.4f}   | {m['brier']:.4f}   | {m['accuracy_pct']:.1f}%    | {m['cal_slope']:.3f}    | {m['blowouts']:<16}")
        print("=" * 95)
        results[horizon_name] = horizon_results

    out_dir = Path("/app/data/08_reporting") if Path("/app").exists() else project_root / "data/08_reporting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "devig_methods_comparison_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved devigging comparison results to {out_file}")


if __name__ == "__main__":
    main()
