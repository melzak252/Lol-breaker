"""Exact, Side-Aligned, Zero-Leakage Benchmark on Scraped Pro Matches.

Resolves the 44.7% side-inversion bug by matching canonical team identities to
GOL.GG team1 / team2 identities from paired.parquet.

Evaluates:
1. Multiplicative Market Consensus (OPEN, 24h, CLOSE)
2. Causal A0 (Zero-Leakage from paired.parquet)
3. Consolidated A1 (A0 + Macro + Matchup + Tier Parity)
4. Calibrated Glicko-2
5. Calibrated Elo
6. EXP-039 (Thesis ElasticNet)
7. Shrunk Hybrids (at OPEN and at CLOSE)
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

research_root = Path((project_root / "data/research_root.txt").read_text().strip())

from src.analysis.devig_comparison import devig_multiplicative


def main():
    # 1. Load canonical benchmark ground truth (paired.parquet)
    paired_path = research_root / "rating-foundation-20260915/data/08_reporting/paired.parquet"
    paired_df = pd.read_parquet(paired_path)
    paired_df["golgg_match_id"] = paired_df["golgg_match_id"].astype(str)

    # 2. Load Consolidated A1 predictions
    a1_path = project_root / "data/07_model_output/upgrades/a0_tier_calibrated_p0_p2.parquet"
    a1_df = pd.read_parquet(a1_path)
    a1_df["golgg_match_id"] = a1_df["golgg_match_id"].astype(str)
    paired_df = paired_df.merge(a1_df[["golgg_match_id", "p_tier_cal"]], on="golgg_match_id", how="left")

    # 3. Connect to database and load mapped canonical matches and odds
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@192.168.1.17:5432/betting",
    )
    engine = create_engine(db_url)
    with engine.connect() as conn:
        scraped_matches = pd.read_sql(
            text("""
            SELECT cm.id as canonical_match_id, gmm.golgg_match_id, cm.team_a_name, cm.team_b_name,
                   cm.winner_side, cm.start_time_normalized, cm.league, cm.best_of
            FROM canonical_matches cm
            JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
            WHERE cm.status IN ('finished', 'completed') AND cm.winner_side IN ('team_a', 'team_b')
            ORDER BY cm.start_time_normalized ASC
            """),
            conn,
        )

        odds = pd.read_sql(
            text("""
            SELECT os.canonical_match_id, b.name as bookmaker, os.raw_team_a, os.raw_team_b, os.odds_a, os.odds_b, os.scraped_at
            FROM odds_snapshots os
            JOIN bookmakers b ON b.id = os.bookmaker_id
            WHERE os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1.0 AND os.odds_b > 1.0
            ORDER BY os.scraped_at ASC
            """),
            conn,
        )

    scraped_matches["golgg_match_id"] = scraped_matches["golgg_match_id"].astype(str)

    # 4. Determine exact side alignment between canonical_matches (team_a) and paired_df (team1)
    # Merge on golgg_match_id
    cohort = scraped_matches.merge(paired_df, on="golgg_match_id", how="inner")
    print(f"Matched {len(cohort)} scraped fixtures directly with canonical benchmark!")

    # Check side alignment:
    # In paired_df: y == 1 means team1 won, y == 0 means team2 won.
    # In canonical_matches: winner_side is 'team_a' or 'team_b'.
    # If winner_side == 'team_a' and y == 1, then team_a IS team1.
    # If winner_side == 'team_a' and y == 0, then team_a IS team2 (swapped!).
    cohort["is_team_a_team1"] = (
        ((cohort["winner_side"] == "team_a") & (cohort["y"] == 1)) |
        ((cohort["winner_side"] == "team_b") & (cohort["y"] == 0))
    )

    n_agrees = cohort["is_team_a_team1"].sum()
    n_swapped = (~cohort["is_team_a_team1"]).sum()
    print(f"Side alignment breakdown: {n_agrees} direct ({n_agrees/len(cohort)*100:.1f}%), {n_swapped} inverted ({n_swapped/len(cohort)*100:.1f}%).")

    # 5. Process odds snapshots with proper side alignment
    df_odds = odds.merge(
        cohort[["canonical_match_id", "start_time_normalized", "is_team_a_team1"]],
        on="canonical_match_id",
        how="inner",
    )
    df_odds["start_dt"] = pd.to_datetime(df_odds["start_time_normalized"], format="mixed", utc=True)
    df_odds["scraped_dt"] = pd.to_datetime(df_odds["scraped_at"], format="mixed", utc=True)
    df_odds = df_odds[df_odds["scraped_dt"] < df_odds["start_dt"]].copy()
    df_odds["hours_before"] = (df_odds["start_dt"] - df_odds["scraped_dt"]).dt.total_seconds() / 3600.0

    # Devigging using Multiplicative: q_i / (q_a + q_b)
    inv_a = 1.0 / df_odds["odds_a"]
    inv_b = 1.0 / df_odds["odds_b"]
    tot = inv_a + inv_b
    prop_p_a = inv_a / tot  # probability that raw_team_a wins

    # Align probability to team1:
    # If team_a IS team1: p_team1 = prop_p_a
    # If team_a IS team2 (swapped): p_team1 = 1.0 - prop_p_a
    df_odds["p_mkt_team1"] = np.where(df_odds["is_team_a_team1"], prop_p_a, 1.0 - prop_p_a)

    # -------------------------------------------------------------------------
    # HORIZON 1: Market OPEN Consensus (Earliest snapshot per bookmaker)
    # -------------------------------------------------------------------------
    first_per_book = df_odds.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).first().reset_index()
    open_cons = first_per_book.groupby("canonical_match_id").agg(
        open_mkt_p=("p_mkt_team1", "mean"),
        open_hours_before=("hours_before", "mean"),
    ).reset_index()

    # -------------------------------------------------------------------------
    # HORIZON 2: 24h Before Kickoff ([14h, 36h])
    # -------------------------------------------------------------------------
    df_24h = df_odds[(df_odds["hours_before"] >= 14.0) & (df_odds["hours_before"] <= 36.0)].copy()
    if len(df_24h) > 0:
        h24_per_book = df_24h.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
        h24_cons = h24_per_book.groupby("canonical_match_id").agg(
            h24_mkt_p=("p_mkt_team1", "mean"),
            h24_hours_before=("hours_before", "mean"),
        ).reset_index()
    else:
        h24_cons = pd.DataFrame(columns=["canonical_match_id", "h24_mkt_p", "h24_hours_before"])

    # -------------------------------------------------------------------------
    # HORIZON 3: Market CLOSE Consensus (<= 2h)
    # -------------------------------------------------------------------------
    df_close = df_odds[df_odds["hours_before"] <= 2.0].copy()
    if len(df_close) == 0:
        df_close = df_odds.copy()
    close_per_book = df_close.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    close_cons = close_per_book.groupby("canonical_match_id").agg(
        close_mkt_p=("p_mkt_team1", "mean"),
        close_hours_before=("hours_before", "mean"),
    ).reset_index()

    # Merge horizons into cohort
    eval_cohort = cohort.merge(open_cons, on="canonical_match_id", how="inner")
    eval_cohort = eval_cohort.merge(close_cons, on="canonical_match_id", how="left")
    eval_cohort = eval_cohort.merge(h24_cons, on="canonical_match_id", how="left")

    print(f"\nFinal Evaluated Cohort with Open Odds: N = {len(eval_cohort)} matches.")
    print(f"Matches with 24h odds: N = {eval_cohort['h24_mkt_p'].notna().sum()}")
    print(f"Matches with CLOSE odds: N = {eval_cohort['close_mkt_p'].notna().sum()}")

    # Ground truth: y == 1 means team1 won, y == 0 means team2 won
    y_test = eval_cohort["y"].values.astype(float)

    # Models predicting team1 win:
    p_a0 = eval_cohort["p_full"].values
    p_a1 = eval_cohort["p_tier_cal"].values
    p_gl = eval_cohort["p_glicko"].values
    p_elo = eval_cohort["p_elo"].values
    p_039 = eval_cohort["p_039"].values

    p_open = eval_cohort["open_mkt_p"].values
    p_close = eval_cohort["close_mkt_p"].fillna(eval_cohort["open_mkt_p"]).values

    # Shrunk Hybrids (50/50 in logit space)
    def blend(pm, pk, a=0.50):
        zm = np.log(np.clip(pm, 1e-6, 1-1e-6) / (1 - np.clip(pm, 1e-6, 1-1e-6)))
        zk = np.log(np.clip(pk, 1e-6, 1-1e-6) / (1 - np.clip(pk, 1e-6, 1-1e-6)))
        return 1.0 / (1.0 + np.exp(-((1 - a) * zm + a * zk)))

    p_hyb_open = blend(p_a1, p_open, 0.50)
    p_hyb_close = blend(p_a1, p_close, 0.50)

    # Metric calculator
    def m(p_vec, y_vec, name):
        mask = np.isfinite(p_vec) & np.isfinite(y_vec) & (p_vec > 0) & (p_vec < 1)
        p_c = np.clip(p_vec[mask], 1e-6, 1.0 - 1e-6)
        y_c = y_vec[mask]
        ll = -np.mean(y_c * np.log(p_c) + (1.0 - y_c) * np.log(1.0 - p_c))
        brier = np.mean((p_c - y_c) ** 2)
        acc = np.mean((p_c >= 0.5) == (y_c == 1.0)) * 100
        p_win = np.where(y_c == 1.0, p_c, 1.0 - p_c)
        blowouts = int(np.sum(-np.log(p_win) >= 2.5))
        return len(y_c), ll, brier, acc, blowouts

    models = [
        ("Shrunk Hybrid A1 (with Market CLOSE)", p_hyb_close),
        ("Shrunk Hybrid A1 (with Market OPEN)", p_hyb_open),
        ("Multiplicative Market CLOSE (Consensus)", p_close),
        ("Multiplicative Market OPEN (Consensus)", p_open),
        ("Consolidated A1 (Tier Parity)", p_a1),
        ("Causal A0 (Zero-Leakage)", p_a0),
        ("Calibrated Glicko-2", p_gl),
        ("EXP-039 (Thesis ElasticNet)", p_039),
        ("Calibrated Elo", p_elo),
    ]

    print("\n" + "=" * 110)
    print(f"CORRECTLY ALIGNED, ZERO-LEAKAGE BENCHMARK ON SCRAPED PRO MATCHES (N = {len(eval_cohort)})")
    print("Market consensus computed as ARITHMETIC MEAN across sportsbooks (Multiplicative)")
    print("=" * 110)
    print(f"{'Model Architecture':<42} | {'N Valid':<8} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'Blowouts (>=2.5)':<16}")
    print("-" * 110)
    for name, p_vec in models:
        n, ll, br, ac, bl = m(p_vec, y_test, name)
        print(f"{name:<42} | {n:<8} | {ll:.4f}   | {br:.4f}   | {ac:.1f}%    | {bl:<16}")
    print("=" * 110)

    # 24h Window evaluation
    has_24h = eval_cohort["h24_mkt_p"].notna().values
    y_24h = y_test[has_24h]
    p_24h_mkt = eval_cohort.loc[has_24h, "h24_mkt_p"].values
    p_24h_a0 = p_a0[has_24h]
    p_24h_a1 = p_a1[has_24h]
    p_24h_hyb = blend(p_24h_a1, p_24h_mkt, 0.50)
    h24_timing_str = f"Mean {eval_cohort['h24_hours_before'].dropna().mean():.1f}h prior"

    print("\n" + "=" * 110)
    print(f"24-HOUR BEFORE KICKOFF HORIZON BENCHMARK (N = {len(y_24h)} Matches with [14h, 36h] odds)")
    print("=" * 110)
    print(f"{'Model / Market (24h Window)':<42} | {'Advance Timing':<18} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'Blowouts (>=2.5)':<16}")
    print("-" * 110)
    for name, p_vec in [
        ("Multiplicative Market 24h (Consensus)", p_24h_mkt),
        ("Shrunk Hybrid A1 (24h Execution)", p_24h_hyb),
        ("Consolidated A1 (24h matches)", p_24h_a1),
        ("Causal A0 (24h matches)", p_24h_a0),
    ]:
        n, ll, br, ac, bl = m(p_vec, y_24h, name)
        print(f"{name:<42} | {h24_timing_str:<18} | {ll:.4f}   | {br:.4f}   | {ac:.1f}%    | {bl:<16}")
    print("=" * 110 + "\n")

if __name__ == "__main__":
    main()
