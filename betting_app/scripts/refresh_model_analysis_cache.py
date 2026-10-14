"""Refresh cached payloads used by the Model Analysis page.

The `/horizon` frontend page should be fast and read-only during normal use.
This script runs the expensive Model Analysis calculations once and writes JSON
payloads under `/app/data/model_analysis_cache/`.  The API endpoints then
serve those files by default; pass `refresh=true` only from this script/manual
maintenance runs.

Usage:
  docker exec -w /app ensemblelegends-betting-scheduler \
    python -m betting_app.scripts.refresh_model_analysis_cache
"""

from __future__ import annotations

import argparse
from datetime import datetime, UTC

from betting_app.api.routers.timing import horizon_accuracy, model_clv_by_horizon
from betting_app.core.db import get_session


def refresh(days_back: int, min_matches_per_bin: int, max_odds_age_hours: float, tax_rate: float, min_ev: float) -> dict:
    started = datetime.now(UTC)
    db = get_session()
    try:
        accuracy = horizon_accuracy(
            min_matches_per_bin=min_matches_per_bin,
            max_days_back=days_back,
            refresh=True,
            db=db,
        )
        clv = model_clv_by_horizon(
            max_days_back=days_back,
            max_odds_age_hours=max_odds_age_hours,
            tax_rate=tax_rate,
            min_ev=min_ev,
            refresh=True,
            db=db,
        )
    finally:
        db.close()

    finished = datetime.now(UTC)
    return {
        "success": True,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_s": (finished - started).total_seconds(),
        "horizon_bins": len(accuracy.get("bins", [])),
        "clv_bins": len(clv.get("bins", [])),
        "params": {
            "days_back": days_back,
            "min_matches_per_bin": min_matches_per_bin,
            "max_odds_age_hours": max_odds_age_hours,
            "tax_rate": tax_rate,
            "min_ev": min_ev,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Model Analysis JSON cache")
    parser.add_argument("--days-back", type=int, default=90)
    parser.add_argument("--min-matches-per-bin", type=int, default=10)
    parser.add_argument("--max-odds-age-hours", type=float, default=4.0)
    parser.add_argument("--tax-rate", type=float, default=0.12)
    parser.add_argument("--min-ev", type=float, default=0.0)
    args = parser.parse_args()

    result = refresh(
        days_back=args.days_back,
        min_matches_per_bin=args.min_matches_per_bin,
        max_odds_age_hours=args.max_odds_age_hours,
        tax_rate=args.tax_rate,
        min_ev=args.min_ev,
    )
    print(result)


if __name__ == "__main__":
    main()
