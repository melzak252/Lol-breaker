"""Comprehensive Horizon & Average Market Consensus Evaluation on Scraped Production Matches.

Computes:
1. 24h Before Match Market Consensus (Arithmetic Average across bookmakers)
2. Market OPEN Consensus (Arithmetic Average across bookmakers)
3. Market CLOSE Consensus (Arithmetic Average across bookmakers, < 2h prior)
4. Base Sports Model
5. Consolidated A1 Model
6. Shrunk Hybrid Model (at 24h, at OPEN, and at CLOSE)

Uses arithmetic average/consensus across sportsbooks, strictly avoids median,
and computes exact LogLoss, Brier, Accuracy, and Tail Blowouts (LL >= 2.5).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.analysis.shin_devig import shin_implied_probabilities

db_url = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://betting:13cd8fbc0b149a22b7748262e25006fa6d502baaa8383de12efa20b00d5589dc@timescaledb:5432/betting",
)
engine = create_engine(db_url)


def main():
    print("=== Extracting Scraped Production Data Across Horizons ===")
    with engine.connect() as conn:
        # 1. Finished canonical matches with verified winner
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

        # 2. Latest predictions for each match using ROW_NUMBER()
        preds = pd.read_sql(
            text("""
            WITH ranked_preds AS (
                SELECT cp.canonical_match_id, cp.model_name, cp.model_version, cp.prob_a, cp.prob_b,
                       ROW_NUMBER() OVER (PARTITION BY cp.canonical_match_id ORDER BY cp.predicted_at DESC) as rn
                FROM canonical_predictions cp
                WHERE (
                    cp.model_name IN ('Hybrid-Bayesian-Shrunk-A0-Market', 'Hybrid-Operational-Market')
                    OR cp.model_version LIKE '%a0.50%'
                )
            )
            SELECT canonical_match_id, model_name, model_version, prob_a, prob_b
            FROM ranked_preds
            WHERE rn = 1
            """),
            conn,
        )

        # 3. All pre-match odds snapshots across licensed sportsbooks
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

    print(f"Loaded {len(matches)} matches, {len(preds)} predictions, {len(odds)} odds snapshots.")

    # Merge matches with predictions
    merged_matches = matches.merge(preds, on="canonical_match_id", how="inner")
    print(f"Matches with valid prediction: N = {len(merged_matches)}")

    # Prepare odds snapshots with timestamps and hours_before
    df_odds = odds.merge(matches[["canonical_match_id", "start_time_normalized"]], on="canonical_match_id", how="inner")
    df_odds["start_dt"] = pd.to_datetime(df_odds["start_time_normalized"], format="mixed", utc=True)
    df_odds["scraped_dt"] = pd.to_datetime(df_odds["scraped_at"], format="mixed", utc=True)

    # Strictly pre-match filter
    df_odds = df_odds[df_odds["scraped_dt"] < df_odds["start_dt"]].copy()
    df_odds["hours_before"] = (df_odds["start_dt"] - df_odds["scraped_dt"]).dt.total_seconds() / 3600.0

    # Devigging using both Shin (1992) and Proportional
    shin_a_list = []
    prop_a_list = []
    for _, r in df_odds.iterrows():
        oa = float(r["odds_a"])
        ob = float(r["odds_b"])
        # Proportional
        inv_a = 1.0 / oa
        inv_b = 1.0 / ob
        tot = inv_a + inv_b
        prop_a = inv_a / tot
        prop_a_list.append(prop_a)

        # Shin
        try:
            spa, spb = shin_implied_probabilities(oa, ob)
        except Exception:
            spa = prop_a
        shin_a_list.append(spa)

    df_odds["prop_p_a"] = prop_a_list
    df_odds["shin_p_a"] = shin_a_list

    # -------------------------------------------------------------------------
    # HORIZON 1: Market OPEN Consensus (Earliest available snapshot per bookmaker)
    # Average across all bookmakers
    # -------------------------------------------------------------------------
    first_per_book = df_odds.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).first().reset_index()
    open_consensus = first_per_book.groupby("canonical_match_id").agg(
        open_mkt_prop_a=("prop_p_a", "mean"),
        open_mkt_shin_a=("shin_p_a", "mean"),
        open_hours_before=("hours_before", "mean"),
        open_books_count=("bookmaker", "count"),
    ).reset_index()

    # -------------------------------------------------------------------------
    # HORIZON 2: 24h Before Kickoff (Snapshots within [14h, 36h])
    # Average across all bookmakers in that window
    # -------------------------------------------------------------------------
    df_24h = df_odds[(df_odds["hours_before"] >= 14.0) & (df_odds["hours_before"] <= 36.0)].copy()
    if len(df_24h) > 0:
        h24_per_book = df_24h.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
        h24_consensus = h24_per_book.groupby("canonical_match_id").agg(
            h24_mkt_prop_a=("prop_p_a", "mean"),
            h24_mkt_shin_a=("shin_p_a", "mean"),
            h24_hours_before=("hours_before", "mean"),
            h24_books_count=("bookmaker", "count"),
        ).reset_index()
    else:
        h24_consensus = pd.DataFrame(columns=["canonical_match_id", "h24_mkt_prop_a", "h24_mkt_shin_a", "h24_hours_before", "h24_books_count"])

    # -------------------------------------------------------------------------
    # HORIZON 3: Market CLOSE Consensus (Latest snapshot within 2h of kickoff)
    # Average across all bookmakers
    # -------------------------------------------------------------------------
    df_close = df_odds[df_odds["hours_before"] <= 2.0].copy()
    if len(df_close) == 0:
        df_close = df_odds.copy()
    close_per_book = df_close.sort_values("scraped_dt").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    close_consensus = close_per_book.groupby("canonical_match_id").agg(
        close_mkt_prop_a=("prop_p_a", "mean"),
        close_mkt_shin_a=("shin_p_a", "mean"),
        close_hours_before=("hours_before", "mean"),
        close_books_count=("bookmaker", "count"),
    ).reset_index()

    # Assemble master cohort
    cohort = merged_matches.merge(open_consensus, on="canonical_match_id", how="inner")
    cohort = cohort.merge(close_consensus, on="canonical_match_id", how="left")
    cohort = cohort.merge(h24_consensus, on="canonical_match_id", how="left")

    print(f"\nFinal Analyzed Cohort with Open Odds: N = {len(cohort)} matches.")
    print(f"Matches with 24h odds ([14h, 36h]):    N = {cohort['h24_mkt_prop_a'].notna().sum()}")
    print(f"Matches with CLOSE odds (<= 2h):      N = {cohort['close_mkt_prop_a'].notna().sum()}")

    # Ground truth: 1.0 if team_a won, 0.0 if team_b won
    y_true = np.where(cohort["winner_side"] == "team_a", 1.0, 0.0)

    # Base predictions
    p_base_model = np.clip(cohort["prob_a"].values.astype(float), 1e-6, 1.0 - 1e-6)

    # Metric helper function
    def calc_metrics(p, y):
        mask_valid = np.isfinite(p) & np.isfinite(y)
        p_c = np.clip(p[mask_valid], 1e-6, 1.0 - 1e-6)
        y_c = y[mask_valid]
        ll = -np.mean(y_c * np.log(p_c) + (1.0 - y_c) * np.log(1.0 - p_c))
        brier = np.mean((p_c - y_c) ** 2)
        acc = np.mean((p_c >= 0.5) == (y_c == 1.0))
        p_win = np.where(y_c == 1.0, p_c, 1.0 - p_c)
        blowouts = int(np.sum(-np.log(p_win) >= 2.5))
        return {
            "n": int(len(y_c)),
            "log_loss": round(float(ll), 4),
            "brier": round(float(brier), 4),
            "accuracy_pct": round(float(acc * 100), 1),
            "blowouts": blowouts,
        }

    # Market Predictions (using Arithmetic Mean Consensus)
    p_open_shin = cohort["open_mkt_shin_a"].values
    p_open_prop = cohort["open_mkt_prop_a"].values

    p_close_shin = cohort["close_mkt_shin_a"].fillna(cohort["open_mkt_shin_a"]).values
    p_close_prop = cohort["close_mkt_prop_a"].fillna(cohort["open_mkt_prop_a"]).values

    # 24h cohort
    has_24h = cohort["h24_mkt_shin_a"].notna().values
    p_24h_shin = cohort.loc[has_24h, "h24_mkt_shin_a"].values
    p_24h_prop = cohort.loc[has_24h, "h24_mkt_prop_a"].values
    y_24h = y_true[has_24h]
    p_model_24h = p_base_model[has_24h]

    # Shrunk Hybrids: 50/50 logit blend
    def blend_logits(p_m, p_k, alpha=0.50):
        zm = np.log(p_m / (1.0 - p_m))
        zk = np.log(p_k / (1.0 - p_k))
        zh = (1.0 - alpha) * zm + alpha * zk
        return 1.0 / (1.0 + np.exp(-zh))

    p_hyb_open = blend_logits(p_base_model, p_open_shin, alpha=0.50)
    p_hyb_close = blend_logits(p_base_model, p_close_shin, alpha=0.50)
    p_hyb_24h = blend_logits(p_model_24h, p_24h_shin, alpha=0.50)

    # Compute all metric dictionaries
    m_base = calc_metrics(p_base_model, y_true)
    m_open_shin = calc_metrics(p_open_shin, y_true)
    m_open_prop = calc_metrics(p_open_prop, y_true)
    m_close_shin = calc_metrics(p_close_shin, y_true)
    m_close_prop = calc_metrics(p_close_prop, y_true)

    m_hyb_open = calc_metrics(p_hyb_open, y_true)
    m_hyb_close = calc_metrics(p_hyb_close, y_true)

    m_24h_mkt_shin = calc_metrics(p_24h_shin, y_24h)
    m_24h_mkt_prop = calc_metrics(p_24h_prop, y_24h)
    m_24h_model = calc_metrics(p_model_24h, y_24h)
    m_24h_hyb = calc_metrics(p_hyb_24h, y_24h)

    open_timing_str = f"Mean {cohort['open_hours_before'].mean():.1f}h prior"
    close_timing_str = f"Mean {cohort['close_hours_before'].dropna().mean():.1f}h prior"
    h24_timing_str = f"Mean {cohort['h24_hours_before'].dropna().mean():.1f}h prior" if len(y_24h) > 0 else "N/A"

    print("\n" + "=" * 110)
    print(f"ACCURACY BENCHMARK ON FULL SCRAPED PRODUCTION COHORT (N = {len(cohort)} Matches)")
    print("Market consensus computed as ARITHMETIC MEAN across all bookmakers (NOT MEDIAN)")
    print("=" * 110)
    print(f"{'Model / Market Horizon':<42} | {'Advance Timing':<18} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'Blowouts (>=2.5)':<16}")
    print("-" * 110)
    print(f"{'Bookmaker OPEN (Shin Average)':<42} | {open_timing_str:<18} | {m_open_shin['log_loss']:.4f}   | {m_open_shin['brier']:.4f}   | {m_open_shin['accuracy_pct']:.1f}%    | {m_open_shin['blowouts']:<16}")
    print(f"{'Bookmaker OPEN (Proportional Average)':<42} | {open_timing_str:<18} | {m_open_prop['log_loss']:.4f}   | {m_open_prop['brier']:.4f}   | {m_open_prop['accuracy_pct']:.1f}%    | {m_open_prop['blowouts']:<16}")
    print(f"{'Bookmaker CLOSE (Shin Average)':<42} | {close_timing_str:<18} | {m_close_shin['log_loss']:.4f}   | {m_close_shin['brier']:.4f}   | {m_close_shin['accuracy_pct']:.1f}%    | {m_close_shin['blowouts']:<16}")
    print(f"{'Bookmaker CLOSE (Proportional Average)':<42} | {close_timing_str:<18} | {m_close_prop['log_loss']:.4f}   | {m_close_prop['brier']:.4f}   | {m_close_prop['accuracy_pct']:.1f}%    | {m_close_prop['blowouts']:<16}")
    print(f"{'Base Sports Model (A0)':<42} | {'Pre-Match Baseline':<18} | {m_base['log_loss']:.4f}   | {m_base['brier']:.4f}   | {m_base['accuracy_pct']:.1f}%    | {m_base['blowouts']:<16}")
    print(f"{'Shrunk Hybrid (with Market OPEN)':<42} | {'Executed at OPEN':<18} | {m_hyb_open['log_loss']:.4f}   | {m_hyb_open['brier']:.4f}   | {m_hyb_open['accuracy_pct']:.1f}%    | {m_hyb_open['blowouts']:<16}")
    print(f"{'Shrunk Hybrid (with Market CLOSE)':<42} | {'Executed at CLOSE':<18} | {m_hyb_close['log_loss']:.4f}   | {m_hyb_close['brier']:.4f}   | {m_hyb_close['accuracy_pct']:.1f}%    | {m_hyb_close['blowouts']:<16}")
    print("=" * 110)

    print(f"\n" + "=" * 110)
    print(f"24-HOUR BEFORE KICKOFF HORIZON BENCHMARK (N = {len(y_24h)} Matches with [14h, 36h] odds)")
    print("=" * 110)
    print(f"{'Model / Market (24h Window)':<42} | {'Advance Timing':<18} | {'LogLoss':<8} | {'Brier':<8} | {'Accuracy':<8} | {'Blowouts (>=2.5)':<16}")
    print("-" * 110)
    print(f"{'Bookmaker 24h (Shin Average)':<42} | {h24_timing_str:<18} | {m_24h_mkt_shin['log_loss']:.4f}   | {m_24h_mkt_shin['brier']:.4f}   | {m_24h_mkt_shin['accuracy_pct']:.1f}%    | {m_24h_mkt_shin['blowouts']:<16}")
    print(f"{'Bookmaker 24h (Proportional Average)':<42} | {h24_timing_str:<18} | {m_24h_mkt_prop['log_loss']:.4f}   | {m_24h_mkt_prop['brier']:.4f}   | {m_24h_mkt_prop['accuracy_pct']:.1f}%    | {m_24h_mkt_prop['blowouts']:<16}")
    print(f"{'Base Sports Model (24h matches)':<42} | {'Pre-Match Baseline':<18} | {m_24h_model['log_loss']:.4f}   | {m_24h_model['brier']:.4f}   | {m_24h_model['accuracy_pct']:.1f}%    | {m_24h_model['blowouts']:<16}")
    print(f"{'Shrunk Hybrid (24h)':<42} | {'Executed ~24h prior':<18} | {m_24h_hyb['log_loss']:.4f}   | {m_24h_hyb['brier']:.4f}   | {m_24h_hyb['accuracy_pct']:.1f}%    | {m_24h_hyb['blowouts']:<16}")
    print("=" * 110 + "\n")

    results_data = {
        "full_cohort": {
            "n_matches": len(cohort),
            "open_shin": m_open_shin,
            "open_prop": m_open_prop,
            "close_shin": m_close_shin,
            "close_prop": m_close_prop,
            "base_model": m_base,
            "shrunk_hybrid_open": m_hyb_open,
            "shrunk_hybrid_close": m_hyb_close,
        },
        "h24_cohort": {
            "n_matches": len(y_24h),
            "market_24h_shin": m_24h_mkt_shin,
            "market_24h_prop": m_24h_mkt_prop,
            "model_24h": m_24h_model,
            "shrunk_hybrid_24h": m_24h_hyb,
        }
    }
    out_dir = Path("/app/data/08_reporting") if Path("/app").exists() else Path("data/08_reporting")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "scraped_horizons_consensus_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Saved horizon evaluation output to {out_file}")


if __name__ == "__main__":
    main()
