"""Bootstrap analysis API — serve cached horizon block bootstrap results."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from betting_app.api.deps import query_df, get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bootstrap", tags=["bootstrap"])

# Path to cached bootstrap results (shared volume between api + scheduler)
_DEFAULT_BOOTSTRAP_DIR = Path(__file__).resolve().parents[3] / "data" / "horizon_block_bootstrap"
BOOTSTRAP_DIR = Path(os.getenv("BOOTSTRAP_DIR", str(_DEFAULT_BOOTSTRAP_DIR)))
RESULTS_CSV = BOOTSTRAP_DIR / "horizon_block_bootstrap_results.csv"
SAMPLES_CSV = BOOTSTRAP_DIR / "horizon_block_bootstrap_samples.csv"
MONTHLY_CSV = BOOTSTRAP_DIR / "horizon_monthly_observed_differences.csv"
PLOT_PNG = BOOTSTRAP_DIR / "horizon_block_bootstrap_ci.png"

def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file and return a list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Cached bootstrap file not found: {path}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _float(val: str) -> float | None:
    """Parse a float from a CSV cell, returning None if empty/invalid/inf."""
    if val == "" or val is None:
        return None
    try:
        f = float(val)
        if f == float('inf') or f == float('-inf'):
            return None
        return f
    except (ValueError, TypeError):
        return None


@router.get("/horizon")
def get_horizon_bootstrap(db=Depends(get_db)):
    """Return the latest cached horizon block bootstrap results.

    Results compare Thesis and Hybrid models vs Bookmaker across 6 time
    horizons using monthly block bootstrap (10,000 resamples). The cached
    analysis is match-oriented: one model × match × horizon observation.
    """
    # ── Read cached CSV ──────────────────────────────────────────────
    try:
        raw_rows = _read_csv(RESULTS_CSV)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Bootstrap results not available. Run the horizon_bootstrap scheduler task first.",
        )

    bins = []
    for row in raw_rows:
        bins.append({
            "model_label": row.get("model_label", ""),
            "model_name": row.get("model_name", ""),
            "comparison": row.get("comparison", ""),
            "label": row.get("label", ""),
            "hours_start": _float(row.get("hours_start", "0")),
            "hours_end": _float(row.get("hours_end", "")) if row.get("hours_end", "") else None,
            "sample_size": _float(row.get("sample_size", "0")),
            "n_blocks": _float(row.get("n_blocks", "0")),
            "model_logloss": _float(row.get("model_logloss", "")),
            "benchmark_logloss": _float(row.get("benchmark_logloss", "")),
            "observed_difference": _float(row.get("observed_difference", "")),
            "ci_low": _float(row.get("ci_low", "")),
            "ci_high": _float(row.get("ci_high", "")),
            "p_one_sided": _float(row.get("p_one_sided", "")),
            "significant_05": row.get("significant_05", "False").lower() == "true",
        })

    # ── Read monthly data if available ───────────────────────────────
    monthly = []
    try:
        monthly_rows = _read_csv(MONTHLY_CSV)
        for row in monthly_rows:
            monthly.append({
                "month": row.get("month", ""),
                "n_snapshots": _float(row.get("n_snapshots", "0")),
                "n_matches": _float(row.get("n_matches", "0")),
                "model_logloss": _float(row.get("model_logloss", "")),
                "bookmaker_logloss": _float(row.get("bookmaker_logloss", "")),
                "mean_difference": _float(row.get("mean_difference", "")),
                "horizon_bin": row.get("horizon_bin", ""),
                "model_label": row.get("model_label", ""),
            })
    except FileNotFoundError:
        pass

    # ── Check if a cached plot PNG is available ──────────────────────
    plot_available = PLOT_PNG.exists()

    # ── Get last update time from file mtime ─────────────────────────
    last_updated = None
    try:
        mtime = RESULTS_CSV.stat().st_mtime
        last_updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        pass

    # ── Get DB info for context ──────────────────────────────────────
    match_stats = {}
    try:
        stats = query_df(db, """
            SELECT
                COUNT(*) FILTER (WHERE status = 'upcoming') AS upcoming,
                COUNT(*) FILTER (WHERE status = 'finished') AS finished,
                COUNT(*) FILTER (WHERE status = 'expired') AS expired
            FROM canonical_matches
        """)
        if stats:
            row = stats[0]
            match_stats = {
                "upcoming": row.get("upcoming") if isinstance(row.get("upcoming"), (int, float)) else None,
                "finished": row.get("finished") if isinstance(row.get("finished"), (int, float)) else None,
                "expired": row.get("expired") if isinstance(row.get("expired"), (int, float)) else None,
            }
    except Exception as e:
        logger.warning("Failed to query canonical_matches stats: %s", e)
        match_stats = {"upcoming": None, "finished": None, "expired": None}

    return {
        "metadata": {
            "aggregation_level": "model_match_horizon",
            "sample_size_definition": "Number of unique matches in the model/horizon bin after collapsing snapshots.",
            "bootstrap_unit": "Monthly blocks of match-horizon observations",
            "snapshot_role": "Snapshots are used only to estimate the market probability within a match/horizon before collapsing to one observation.",
        },
        "bins": bins,
        "monthly": monthly,
        "last_updated": last_updated,
        "plot_available": plot_available,
        "match_stats": match_stats,
    }


@router.post("/horizon/refresh")
def refresh_horizon_bootstrap(db=Depends(get_db)):
    """Trigger a refresh of the horizon bootstrap analysis.

    Returns a status message indicating the refresh has been queued.
    The actual computation runs via the scheduler task (horizon_bootstrap).
    To run immediately, use POST /scheduler/trigger/horizon_bootstrap.
    """
    return {
        "status": "ok",
        "message": "Use POST /scheduler/trigger/horizon_bootstrap to run the bootstrap immediately, "
                    "or wait for the daily scheduled task.",
    }
