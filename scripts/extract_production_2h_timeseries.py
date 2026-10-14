"""Extract 2-hour decision time-series quote trajectories from production database."""
from __future__ import annotations
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.analysis.betting_guards_a_to_d import devig_shin

DB_URL = os.getenv("PROD_DATABASE_URL", "postgresql://betting:betting_local_password@localhost:5432/betting")


def get_lead_horizon(lead_hours: float) -> str | None:
    """Map lead hours before match start into operational decision buckets."""
    if lead_hours < 0:
        return None  # Post-start quote
    if lead_hours <= 0.5:
        return "T-30m"
    if lead_hours <= 1.0:
        return "T-1h"
    if lead_hours <= 2.5:
        return "T-2h"
    if lead_hours <= 6.5:
        return "T-6h"
    if lead_hours <= 12.5:
        return "T-12h"
    if lead_hours <= 24.5:
        return "T-24h"
    if lead_hours <= 48.5:
        return "T-48h"
    return "T-pre48h"


def extract_production_trajectory(engine=None, output_path: str = "data/04_feature/production_2h_odds_trajectory.parquet") -> pd.DataFrame:
    if engine is None:
        engine = create_engine(DB_URL)

    query = text("""
        SELECT 
            o.id as snapshot_id,
            o.canonical_match_id,
            o.bookmaker_id,
            b.name as bookmaker_name,
            o.scraped_at,
            o.odds_a,
            o.odds_b,
            cm.team_a_name,
            cm.team_b_name,
            cm.start_time_normalized,
            cm.league,
            cm.status as match_status,
            cm.winner_side
        FROM odds_snapshots o
        JOIN bookmakers b ON b.id = o.bookmaker_id
        JOIN canonical_matches cm ON cm.id = o.canonical_match_id
        WHERE o.market_type = 'match_winner'
          AND (o.is_live = 0 OR o.is_live IS NULL)
          AND o.odds_a > 1.0 
          AND o.odds_b > 1.0
          AND cm.start_time_normalized IS NOT NULL
        ORDER BY o.canonical_match_id, o.scraped_at ASC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError("No matching odds_snapshots found in production database.")

    # Calculate exact lead hours
    scraped_dt = pd.to_datetime(df["scraped_at"], utc=True)
    start_dt = pd.to_datetime(df["start_time_normalized"], utc=True)
    df["scraped_dt"] = scraped_dt
    df["start_dt"] = start_dt
    df["lead_hours"] = (start_dt - scraped_dt).dt.total_seconds() / 3600.0

    # Filter pre-match quotes only
    df = df[df["lead_hours"] >= 0].copy()
    df["horizon"] = df["lead_hours"].apply(get_lead_horizon)
    df = df[df["horizon"].notna()].copy()

    # For each (canonical_match_id, bookmaker_id, horizon), pick the latest quote
    df_sorted = df.sort_values(by=["canonical_match_id", "bookmaker_id", "horizon", "scraped_dt"])
    dedup = df_sorted.groupby(["canonical_match_id", "bookmaker_id", "horizon"]).last().reset_index()

    # Calculate fair probabilities via Shin
    p_shin_list = []
    for _, row in dedup.iterrows():
        oa, ob = float(row["odds_a"]), float(row["odds_b"])
        try:
            psa, _ = devig_shin(oa, ob)
        except Exception:
            psa = (1.0 / oa) / (1.0 / oa + 1.0 / ob)
        p_shin_list.append(psa)

    dedup["p_shin_a"] = p_shin_list

    # Join model predictions if available
    # Check canonical_predictions in DB
    with engine.connect() as conn:
        preds = pd.read_sql(text("""
            SELECT canonical_match_id, prob_a as p_model_a, model_name, model_version
            FROM canonical_predictions
            WHERE prediction_status = 'active'
        """), conn)

    if not preds.empty:
        preds_dedup = preds.drop_duplicates(subset=["canonical_match_id"])
        dedup = pd.merge(dedup, preds_dedup[["canonical_match_id", "p_model_a"]], on="canonical_match_id", how="left")
    else:
        dedup["p_model_a"] = np.nan

    # Export
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    dedup.to_parquet(out_p, index=False)
    print(f"Exported {len(dedup)} trajectory rows to {output_path}")
    return dedup


if __name__ == "__main__":
    extract_production_trajectory()
