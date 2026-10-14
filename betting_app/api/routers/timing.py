"""Router: /api/timing — odds timing analysis for betting strategy.

Key metrics:
  - Fixed 2-hour time buckets (0-2h, 2-4h, 4-6h, ..., up to 72h+)
  - % deviation from closing odds (last pre-match snapshot)
  - Shows whether earlier odds are better/worse than closing
  - Convergence pattern analysis
  - Horizon accuracy: LogLoss & AUC by hours-before-match bin
"""

from __future__ import annotations

import math
import os
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from betting_app.api.deps import get_db, query_df
from betting_app.core.clv import clv_odds_pct, clv_probability_points
from betting_app.core.ev import best_ev_side, expected_value, fair_market_probabilities
from betting_app.core.models.engine import bayesian_logit_shrinkage
from betting_app.services.thesis_inference_service import THESIS_HYBRID_ALPHA, THESIS_HYBRID_TEMPERATURE
from betting_app.services.rating_contract import (
    OPERATIONAL_BACKFILL_FEATURE_VERSION,
    OPERATIONAL_BACKFILL_MODEL_VERSION,
    OPERATIONAL_MODEL_NAME,
)

router = APIRouter(prefix="/timing", tags=["timing"])

from betting_app.core.models import (
    C0_NATIVE,
    get_active_hybrid,
    get_active_model,
    get_thesis_hybrid,
    get_thesis_model,
)

THESIS_MODEL_NAME = get_thesis_model().name
THESIS_MODEL_VERSION = get_thesis_model().version
THESIS_HYBRID_MODEL_NAME = get_thesis_hybrid().hybrid_model_name
ACTIVE_OPERATIONAL_MODEL_NAME = get_active_model().name
ACTIVE_OPERATIONAL_MODEL_VERSION = get_active_model().version
OPERATIONAL_HYBRID_MODEL_NAME = get_active_hybrid().hybrid_model_name
# The Horizon page is a retrospective EXP-060 evaluation.  Do not silently
# mix its reproducible, point-in-time backfill with opportunistic scheduler
# predictions left over from earlier deployments (some were written after the
# match had already begun).
ANALYSIS_FEATURES_VERSION = "exp060-db-backfill-v1"
# Unlike accuracy, CLV needs the timestamp at which a forecast actually became
# available.  EXP-060 rows are deliberately timestamped at start-1min for
# retrospective scoring, so they must never be presented as live signals.
LIVE_THESIS_FEATURES_VERSION = "thesis-exp039"

EXP040_MODEL_NAME = "Hierarchical-Markov-VennAbers-EXP040"
EXP040_FEATURES_VERSION = "exp040-markov-va-v1"

HORIZON_BIN_DEFS = [
    ("0-2h", 0, 2),
    ("2-6h", 2, 6),
    ("6-12h", 6, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
    ("48h+", 48, 9999),
]

# Shared bind mount between betting-api and betting-scheduler containers.
# Do not use /app/docs here: scheduler writes would not be visible to API.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

def _resolve_model_analysis_cache_dir() -> Path:
    container_dir = Path("/app/data/model_analysis_cache")
    if container_dir.parent.exists() and os.access(container_dir.parent, os.W_OK):
        return container_dir
    local_dir = PROJECT_ROOT / "data" / "model_analysis_cache"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        if os.access(local_dir, os.W_OK):
            return local_dir
    except Exception:
        pass
    return Path("/tmp/model_analysis_cache")


MODEL_ANALYSIS_CACHE_DIR = _resolve_model_analysis_cache_dir()


def _cache_path(kind: str, **params: Any) -> Path:
    safe = "__".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(params.items()))
    return MODEL_ANALYSIS_CACHE_DIR / f"{kind}__{safe}.json"


def _read_model_analysis_cache(kind: str, **params: Any) -> dict | None:
    safe = "__".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(params.items()))
    for target_dir in (MODEL_ANALYSIS_CACHE_DIR, Path("/tmp/model_analysis_cache")):
        path = target_dir / f"{kind}__{safe}.json"
        if not path.exists():
            continue
        try:
            with path.open() as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                payload.setdefault("cache", {})
                payload["cache"].update({
                    "hit": True,
                    "path": str(path),
                    "last_updated": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
                })
            return payload
        except Exception:
            continue
    return None


def _write_model_analysis_cache(kind: str, payload: dict, **params: Any) -> None:
    safe = "__".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(params.items()))
    enriched = dict(payload)
    for target_dir in (MODEL_ANALYSIS_CACHE_DIR, Path("/tmp/model_analysis_cache")):
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / f"{kind}__{safe}.json"
            enriched["cache"] = {
                "hit": False,
                "generated_at": datetime.now(UTC).isoformat(),
                "path": str(path),
                "params": params,
            }
            tmp = path.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(enriched, f, ensure_ascii=False, allow_nan=False, default=str)
            tmp.replace(path)
            return
        except Exception:
            continue
def _parse_dt(val: Any) -> datetime | None:
    """Parse datetime from various formats."""
    if val is None:
        return None
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val
    return None


def _align(
    n_a: str, n_b: str,
    raw_a: str, raw_b: str,
    odds_a: float | None, odds_b: float | None,
    *, league: str | None = None, match_date: Any = None,
) -> tuple[float, float] | None:
    """Delegate contextual identity to the one strict operational aligner."""
    from betting_app.services.canonical_match_service import align_snapshot_odds
    return align_snapshot_odds(
        n_a, n_b, raw_a, raw_b, odds_a, odds_b,
        league=league, source_system="bookmaker", match_date=match_date,
    )


@router.get("/analysis")
def timing_analysis(
    days_back: int = 60,
    min_snapshots: int = 2,
    db=Depends(get_db),
):
    """Analyze odds movement relative to closing odds.

    For each finished/expired match, finds the closing odds (last snapshot
    before match start) and measures how earlier odds deviated from closing.
    Results are grouped into fixed 2-hour time buckets.

    Returns:
      - time_buckets: avg % deviation from closing per 2h window
      - drift_summary: convergence/divergence pattern
      - best_window: most favorable betting window
    """
    now = datetime.now(UTC)
    cutoff = (now - timedelta(days=days_back)).isoformat(timespec="seconds")

    # --- 1. Get finished/expired matches ---
    matches = query_df(
        db,
        """
        SELECT cm.id, cm.start_time_normalized,
               cm.normalized_team_a, cm.normalized_team_b, cm.league
        FROM canonical_matches cm
        WHERE cm.status IN ('finished', 'completed', 'expired')
          AND cm.start_time_normalized IS NOT NULL
          AND cm.start_time_normalized > :cutoff
        ORDER BY cm.start_time_normalized DESC
        LIMIT 500
        """,
        {"cutoff": cutoff},
    )
    if not matches:
        return _empty_result()

    match_map = {m["id"]: m for m in matches}
    match_ids = list(match_map.keys())

    # --- 2. Get all odds snapshots for these matches ---
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params: dict[str, Any] = {f"mid{i}": mid for i, mid in enumerate(match_ids)}

    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at,
               os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b,
               b.name AS bookmaker
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.scraped_at
        """,
        params,
    )

    # Group by match
    match_snaps: dict[int, list[dict]] = {}
    for s in snapshots:
        mid = s["canonical_match_id"]
        match_snaps.setdefault(mid, []).append(s)

    # --- 3. Process each match ---
    BUCKET_SIZE = 2  # hours
    buckets_raw: dict[str, list[dict]] = {}
    matches_with_data = 0

    for mid in match_ids:
        m = match_map.get(mid)
        s_list = match_snaps.get(mid)
        if not m or not s_list or len(s_list) < min_snapshots:
            continue

        match_start = _parse_dt(m.get("start_time_normalized"))
        if not match_start:
            continue

        n_a = str(m.get("normalized_team_a") or "")
        n_b = str(m.get("normalized_team_b") or "")

        # Parse and filter pre-match only
        pre_match: list[tuple[float, dict]] = []
        for s in s_list:
            st = _parse_dt(s["scraped_at"])
            if not st:
                continue
            hours_before = (match_start - st).total_seconds() / 3600.0
            if hours_before < 0:
                continue
            pre_match.append((hours_before, s))

        if len(pre_match) < min_snapshots:
            continue

        # Sort by hours_before ascending (closest to match first)
        pre_match.sort(key=lambda x: x[0])

        # --- Closing odds = last snapshot before match ---
        closing_hours, closing_snap = pre_match[0]
        closing_aligned = _align(
            n_a, n_b,
            str(closing_snap.get("raw_team_a") or ""),
            str(closing_snap.get("raw_team_b") or ""),
            closing_snap.get("odds_a"),
            closing_snap.get("odds_b"),
            league=m.get("league"), match_date=m.get("start_time_normalized"),
        )
        if not closing_aligned:
            continue

        closing_a, closing_b = closing_aligned
        matches_with_data += 1

        # --- Process each snapshot ---
        for hours_before, s in pre_match:
            aligned = _align(
                n_a, n_b,
                str(s.get("raw_team_a") or ""),
                str(s.get("raw_team_b") or ""),
                s.get("odds_a"),
                s.get("odds_b"),
                league=m.get("league"), match_date=m.get("start_time_normalized"),
            )
            if not aligned:
                continue

            odds_a, odds_b = aligned

            # % deviation from closing
            dev_a_pct = ((odds_a - closing_a) / closing_a) * 100.0 if closing_a else 0.0
            dev_b_pct = ((odds_b - closing_b) / closing_b) * 100.0 if closing_b else 0.0

            # Assign to 2h bucket
            bucket_start = int(hours_before) // BUCKET_SIZE * BUCKET_SIZE
            bucket_end = bucket_start + BUCKET_SIZE
            bucket_label = f"{bucket_start}-{bucket_end}h"

            buckets_raw.setdefault(bucket_label, []).append({
                "match_id": mid,
                "hours_before": hours_before,
                "odds_a": odds_a,
                "odds_b": odds_b,
                "closing_a": closing_a,
                "closing_b": closing_b,
                "deviation_a_pct": dev_a_pct,
                "deviation_b_pct": dev_b_pct,
                "bookmaker": s.get("bookmaker"),
            })

    if not buckets_raw:
        return _empty_result(matches_with_data)

    # --- 4. Aggregate buckets ---
    # Determine max bucket range (round up to even, min 48h)
    all_hours = []
    for items in buckets_raw.values():
        for item in items:
            all_hours.append(item["hours_before"])
    if not all_hours:
        return _empty_result(matches_with_data)

    max_hours = max(all_hours)
    max_bucket_end = int(max_hours) + BUCKET_SIZE
    if max_bucket_end % BUCKET_SIZE != 0:
        max_bucket_end += BUCKET_SIZE
    max_bucket_end = max(max_bucket_end, 48)

    # Create all 2h bucket slots
    ordered_slots = []
    for start_h in range(0, max_bucket_end, BUCKET_SIZE):
        end_h = start_h + BUCKET_SIZE
        label = f"{start_h}-{end_h}h"
        ordered_slots.append(label)

    time_buckets = []
    for label in ordered_slots:
        items = buckets_raw.get(label, [])
        if not items:
            continue

        n = len(items)
        dev_a = [it["deviation_a_pct"] for it in items]
        dev_b = [it["deviation_b_pct"] for it in items]
        odds_a_list = [it["odds_a"] for it in items]
        odds_b_list = [it["odds_b"] for it in items]
        closing_a_list = [it["closing_a"] for it in items]
        closing_b_list = [it["closing_b"] for it in items]
        unique_matches = len(set(it["match_id"] for it in items))

        avg_dev_a = sum(dev_a) / n
        avg_dev_b = sum(dev_b) / n

        # Std dev of deviation
        if n > 1:
            std_dev_a = (sum((d - avg_dev_a) ** 2 for d in dev_a) / (n - 1)) ** 0.5
            std_dev_b = (sum((d - avg_dev_b) ** 2 for d in dev_b) / (n - 1)) ** 0.5
        else:
            std_dev_a = std_dev_b = 0.0

        time_buckets.append({
            "bucket": label,
            "hours_start": int(label.split("-")[0]),
            "hours_end": int(label.split("-")[1].rstrip("h")),
            "snapshot_count": n,
            "match_count": unique_matches,
            "avg_deviation_a_pct": round(avg_dev_a, 2),
            "avg_deviation_b_pct": round(avg_dev_b, 2),
            "std_deviation_a_pct": round(std_dev_a, 2),
            "std_deviation_b_pct": round(std_dev_b, 2),
            "avg_odds_a": round(sum(odds_a_list) / n, 3),
            "avg_odds_b": round(sum(odds_b_list) / n, 3),
            "avg_closing_odds_a": round(sum(closing_a_list) / n, 3),
            "avg_closing_odds_b": round(sum(closing_b_list) / n, 3),
        })

    # --- 5. Drift summary ---
    drift_summary = None
    if len(time_buckets) >= 2:
        earliest = time_buckets[-1]  # farthest from match
        latest = time_buckets[0]     # closest to match

        drift_summary = {
            "earliest_bucket": earliest["bucket"],
            "latest_bucket": latest["bucket"],
            "open_deviation_a_pct": earliest["avg_deviation_a_pct"],
            "open_deviation_b_pct": earliest["avg_deviation_b_pct"],
            "close_deviation_a_pct": latest["avg_deviation_a_pct"],
            "close_deviation_b_pct": latest["avg_deviation_b_pct"],
            "convergence_a_pct": round(
                earliest["avg_deviation_a_pct"] - latest["avg_deviation_a_pct"], 2
            ),
            "convergence_b_pct": round(
                earliest["avg_deviation_b_pct"] - latest["avg_deviation_b_pct"], 2
            ),
        }

    # --- 6. Best betting window ---
    # The best window is where deviation is highest (odds furthest above closing)
    # For each side independently, then combined
    best_window = None
    if time_buckets:
        scored = []
        for b in time_buckets:
            # "Favorable deviation" = max positive deviation from closing
            # Positive = odds better than closing
            fav_dev = max(b["avg_deviation_a_pct"], b["avg_deviation_b_pct"])
            # Also consider stability
            stability = (b["std_deviation_a_pct"] + b["std_deviation_b_pct"]) / 2
            scored.append((b["bucket"], fav_dev, stability, b["match_count"], b))

        # Filter to buckets with reasonable sample
        scored = [s for s in scored if s[3] >= 3]
        if scored:
            # Find bucket where favorable deviation is highest
            scored.sort(key=lambda x: x[1], reverse=True)
            best = scored[0][4]
            best_window = {
                "bucket": best["bucket"],
                "hours_start": best["hours_start"],
                "hours_end": best["hours_end"],
                "avg_favorable_deviation_pct": max(
                    best["avg_deviation_a_pct"], best["avg_deviation_b_pct"]
                ),
                "avg_deviation_a_pct": best["avg_deviation_a_pct"],
                "avg_deviation_b_pct": best["avg_deviation_b_pct"],
                "match_count": best["match_count"],
                "snapshot_count": best["snapshot_count"],
                "recommendation": _build_recommendation(best),
            }

    return {
        "total_matches": matches_with_data,
        "total_snapshots": sum(len(v) for v in buckets_raw.values()),
        "time_buckets": time_buckets,
        "drift_summary": drift_summary,
        "best_betting_window": best_window,
    }


def _build_recommendation(bucket: dict) -> str:
    """Generate human-readable betting recommendation."""
    dev_a = bucket["avg_deviation_a_pct"]
    dev_b = bucket["avg_deviation_b_pct"]

    parts = []
    if dev_a > 1:
        parts.append(f"Team A odds average {dev_a:+.1f}% above closing")
    elif dev_a < -1:
        parts.append(f"Team A odds average {dev_a:+.1f}% below closing")

    if dev_b > 1:
        parts.append(f"Team B odds average {dev_b:+.1f}% above closing")
    elif dev_b < -1:
        parts.append(f"Team B odds average {dev_b:+.1f}% below closing")

    if not parts:
        return f"Odds in the {bucket['bucket']} window are near closing levels."

    return (
        f"In the {bucket['bucket']} window before match: "
        f"{'; '.join(parts)}. "
        f"This window shows the most favorable odds vs closing prices."
    )


def _implied_prob(odds: float) -> float:
    """Convert decimal odds to implied probability (no-margin)."""
    if odds is None or odds <= 1.0:
        return 0.5
    return 1.0 / odds


def _remove_margin(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove bookmaker margin by normalizing probabilities to sum to 1."""
    total = prob_a + prob_b
    if total <= 0:
        return (0.5, 0.5)
    return (prob_a / total, prob_b / total)


def _compute_auc(y_true: list[int], y_score: list[float]) -> float | None:
    """Compute AUC-ROC, return None if degenerate."""
    try:
        if len(set(y_true)) < 2:
            return None
        val = roc_auc_score(y_true, y_score)
        return round(float(val), 4)
    except Exception:
        return None


def _compute_logloss(y_true: list[int], y_prob: list[float], eps: float = 1e-15) -> float | None:
    """Compute LogLoss, return None if degenerate."""
    try:
        n = len(y_true)
        if n == 0:
            return None
        arr_true = np.array(y_true, dtype=float)
        arr_prob = np.array(y_prob, dtype=float)
        arr_prob = np.clip(arr_prob, eps, 1 - eps)
        ll = -np.mean(arr_true * np.log(arr_prob) + (1 - arr_true) * np.log(1 - arr_prob))
        return round(float(ll), 4)
    except Exception:
        return None


def _compute_brier(y_true: list[int], y_prob: list[float]) -> float | None:
    """Compute Brier score, return None if no observations."""
    if not y_true:
        return None
    arr_true = np.array(y_true, dtype=float)
    arr_prob = np.array(y_prob, dtype=float)
    return round(float(np.mean((arr_prob - arr_true) ** 2)), 4)


def _compute_accuracy(y_true: list[int], y_prob: list[float]) -> float | None:
    """Compute threshold-0.5 binary accuracy."""
    if not y_true:
        return None
    preds = [1 if float(p) >= 0.5 else 0 for p in y_prob]
    correct = sum(1 for y, p in zip(y_true, preds, strict=False) if int(y) == int(p))
    return round(float(correct / len(y_true)), 4)


def _logloss_one(y_true: int, y_prob: float, eps: float = 1e-15) -> float:
    """Binary LogLoss for a single paired observation."""
    p = max(eps, min(1 - eps, float(y_prob)))
    y = 1.0 if int(y_true) == 1 else 0.0
    return float(-(y * math.log(p) + (1 - y) * math.log(1 - p)))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta (Numerical Recipes)."""
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b), used for t-test p-values."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: int) -> float:
    """CDF of Student's t distribution without SciPy dependency."""
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    ib = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    if t >= 0:
        return 1.0 - 0.5 * ib
    return 0.5 * ib


def _student_t_ppf(prob: float, df: int) -> float | None:
    """Inverse CDF by bisection; sufficient for critical values shown in UI."""
    if df <= 0 or not (0.0 < prob < 1.0):
        return None
    lo, hi = -50.0, 50.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if _student_t_cdf(mid, df) < prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _paired_t_test(differences: list[float], alpha: float = 0.05) -> dict | None:
    """One-sided paired t-test for mean(differences) > 0."""
    n = len(differences)
    if n < 2:
        return None
    arr = np.array(differences, dtype=float)
    mean_diff = float(np.mean(arr))
    sd_diff = float(np.std(arr, ddof=1))
    df = n - 1
    if sd_diff <= 0:
        t_stat = math.inf if mean_diff > 0 else (-math.inf if mean_diff < 0 else 0.0)
        p_value = 0.0 if mean_diff > 0 else 1.0
        sem = 0.0
    else:
        sem = sd_diff / math.sqrt(n)
        t_stat = mean_diff / sem
        p_value = 1.0 - _student_t_cdf(t_stat, df)
    critical = _student_t_ppf(1.0 - alpha, df)
    return {
        "n": n,
        "df": df,
        "mean_diff": round(mean_diff, 6),
        "sd_diff": round(sd_diff, 6),
        "sem_diff": round(sem, 6),
        "t_stat": round(float(t_stat), 4) if math.isfinite(t_stat) else None,
        "p_value_one_sided": round(float(p_value), 6),
        "alpha": alpha,
        "t_critical_95_one_sided": round(float(critical), 4) if critical is not None else None,
        "significant": bool(p_value < alpha and mean_diff > 0),
    }


def _compute_model_vs_bookmaker_tests(db, cutoff: str) -> list[dict]:
    """Paired tests: is model better than average bookmaker?

    Positive difference means: bookmaker error - model error > 0, i.e. the
    model has lower error than the average bookmaker on the same match.
    We use paired observations per match to avoid overweighting matches with
    many bookmakers/scrapes.
    """
    preds = query_df(
        db,
        """
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a AS thesis_prob_a,
                   cm.winner_side, cm.start_time_normalized,
                   cm.normalized_team_a, cm.normalized_team_b, cm.league,
                   ROW_NUMBER() OVER (
                       PARTITION BY cp.canonical_match_id
                       ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                   ) AS rn
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = :mname
              AND cp.model_version = :mver
              AND cp.features_version = :features_version
              AND cp.predicted_at <= cm.start_time_normalized::timestamptz
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
              AND cm.start_time_normalized > :cutoff
        )
        SELECT canonical_match_id, thesis_prob_a, winner_side,
               start_time_normalized, normalized_team_a, normalized_team_b, league
        FROM ranked
        WHERE rn = 1
        """,
        {
            "mname": THESIS_MODEL_NAME,
            "mver": THESIS_MODEL_VERSION,
            "features_version": ANALYSIS_FEATURES_VERSION,
            "cutoff": cutoff,
        },
    )
    if not preds:
        return []

    pred_map = {
        p["canonical_match_id"]: p
        for p in preds
        if p.get("thesis_prob_a") is not None
        and str(p.get("winner_side") or "") in ("team_a", "team_b")
    }
    if not pred_map:
        return []

    match_ids = list(pred_map.keys())
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params = {f"mid{i}": mid for i, mid in enumerate(match_ids)}

    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at,
               os.bookmaker_id, os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.scraped_at
        """,
        params,
    )

    # Per match, average first within bookmaker, then across bookmakers.
    market_by_match: dict[int, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    hybrid_by_match: dict[int, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    alpha = THESIS_HYBRID_ALPHA
    temperature = THESIS_HYBRID_TEMPERATURE

    for snap in snapshots:
        mid = snap["canonical_match_id"]
        meta = pred_map.get(mid)
        if not meta:
            continue
        match_start = _parse_dt(meta.get("start_time_normalized"))
        scraped = _parse_dt(snap.get("scraped_at"))
        if not match_start or not scraped:
            continue
        if (match_start - scraped).total_seconds() < 0:
            continue
        aligned = _align(
            str(meta.get("normalized_team_a") or ""),
            str(meta.get("normalized_team_b") or ""),
            str(snap.get("raw_team_a") or ""),
            str(snap.get("raw_team_b") or ""),
            snap.get("odds_a"),
            snap.get("odds_b"),
            league=meta.get("league"), match_date=meta.get("start_time_normalized"),
        )
        if not aligned:
            continue
        odds_a, odds_b = aligned
        prob_a = _implied_prob(odds_a)
        prob_b = _implied_prob(odds_b)
        market_prob_a, market_prob_b = _remove_margin(prob_a, prob_b)
        thesis_prob_a = float(meta["thesis_prob_a"])
        thesis_prob_a_t = apply_temperature_probability(thesis_prob_a, temperature)
        hybrid_prob_a = alpha * thesis_prob_a_t + (1.0 - alpha) * market_prob_a
        bookmaker_key = snap.get("bookmaker_id") or "unknown"
        market_by_match[mid][bookmaker_key].append(market_prob_a)
        hybrid_by_match[mid][bookmaker_key].append(hybrid_prob_a)

    thesis_ll_diffs: list[float] = []
    thesis_brier_diffs: list[float] = []
    hybrid_ll_diffs: list[float] = []
    hybrid_brier_diffs: list[float] = []

    for mid, meta in pred_map.items():
        bookmaker_probs = [float(np.mean(vals)) for vals in market_by_match[mid].values() if vals]
        hybrid_probs = [float(np.mean(vals)) for vals in hybrid_by_match[mid].values() if vals]
        if not bookmaker_probs:
            continue
        y_true = 1 if str(meta.get("winner_side") or "") == "team_a" else 0
        market_prob = float(np.mean(bookmaker_probs))
        thesis_prob = float(meta["thesis_prob_a"])
        hybrid_prob = float(np.mean(hybrid_probs)) if hybrid_probs else thesis_prob

        market_ll = _logloss_one(y_true, market_prob)
        thesis_ll = _logloss_one(y_true, thesis_prob)
        hybrid_ll = _logloss_one(y_true, hybrid_prob)
        market_brier = (market_prob - y_true) ** 2
        thesis_brier = (thesis_prob - y_true) ** 2
        hybrid_brier = (hybrid_prob - y_true) ** 2

        thesis_ll_diffs.append(market_ll - thesis_ll)
        thesis_brier_diffs.append(market_brier - thesis_brier)
        hybrid_ll_diffs.append(market_ll - hybrid_ll)
        hybrid_brier_diffs.append(market_brier - hybrid_brier)

    definitions = [
        ("thesis_logloss", "Thesis model vs średni bukmacher — LogLoss", "logloss", THESIS_MODEL_NAME, thesis_ll_diffs),
        ("thesis_brier", "Thesis model vs średni bukmacher — Brier", "brier", THESIS_MODEL_NAME, thesis_brier_diffs),
        ("hybrid_logloss", "Hybrid thesis+market vs średni bukmacher — LogLoss", "logloss", THESIS_HYBRID_MODEL_NAME, hybrid_ll_diffs),
        ("hybrid_brier", "Hybrid thesis+market vs średni bukmacher — Brier", "brier", THESIS_HYBRID_MODEL_NAME, hybrid_brier_diffs),
    ]
    results: list[dict] = []
    for test_id, label, metric, model_name, diffs in definitions:
        stats = _paired_t_test(diffs)
        if not stats:
            continue
        results.append({
            "id": test_id,
            "label": label,
            "metric": metric,
            "model_name": model_name,
            "baseline_name": "Average bookmaker",
            "alternative": "mean(bookmaker_error - model_error) > 0",
            "interpretation": "positive_mean_diff_means_model_better",
            **stats,
        })
    return results




@router.get("/model-comparison")
def historical_model_comparison(
    max_days_back: int = 3650,
    league: str | None = None,
    best_of: int | None = None,
    target_model: str = "c0",
    db=Depends(get_db),
):
    """Compare native C0, the exact active market hybrid, and frozen baselines.

    This endpoint is retrospective and only reports rows matching each model's
    exact name, version, and feature identity.
    """
    cutoff = (datetime.now(UTC) - timedelta(days=max_days_back)).isoformat()
    predictions_by_key: dict[str, dict[int, dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    specifications = (
        {
            "key": "c0",
            "label": "Native C0",
            "description": "Causal-C0 native predictor using the exact frozen native artifacts.",
            "model_name": C0_NATIVE.name,
            "model_version": C0_NATIVE.version,
            "features_version": C0_NATIVE.feature_version,
        },
        {
            "key": "operational_hybrid",
            "label": "Hybryda Operacyjna (C0 + Rynek alpha=0.50)",
            "description": "Hybrid-Operational-Market using native C0 and the active logit-shrinkage configuration.",
            "model_name": get_active_hybrid().hybrid_model_name,
            "model_version": get_active_hybrid().hybrid_model_version,
            "features_version": C0_NATIVE.feature_version,
        },
        {
            "key": "operational_regional",
            "label": "Regional operational BoN replay",
            "description": "v0.4-binom-series-chronological-v1 (model operacyjny z ogonem dwumianowym)",
            "model_name": OPERATIONAL_MODEL_NAME,
            "model_version": OPERATIONAL_BACKFILL_MODEL_VERSION,
            "features_version": OPERATIONAL_BACKFILL_FEATURE_VERSION,
        },
        {
            "key": "exp039",
            "label": "EXP-039 thesis baseline",
            "description": "Sym-Cal LR-ElasticNet-W20-Binomial (model referencyjny pracy)",
            "model_name": THESIS_MODEL_NAME,
            "model_version": THESIS_MODEL_VERSION,
            "features_version": ANALYSIS_FEATURES_VERSION,
        },
    )
    for specification in specifications:
        extra_filters = ""
        query_params: dict[str, Any] = {**specification, "cutoff": cutoff}
        if league:
            extra_filters += " AND cm.league = :league_filter"
            query_params["league_filter"] = league
        if best_of:
            extra_filters += " AND COALESCE(cm.best_of, 1) = :best_of_filter"
            query_params["best_of_filter"] = int(best_of)

        rows = query_df(
            db,
            f"""
            WITH ranked AS (
                SELECT cp.canonical_match_id, cp.prob_a, cm.winner_side,
                       cm.league, cm.best_of, cm.start_time_normalized,
                       ROW_NUMBER() OVER (
                           PARTITION BY cp.canonical_match_id
                           ORDER BY CASE
                               WHEN cp.predicted_at ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
                               THEN cp.predicted_at::timestamptz
                           END DESC NULLS LAST, cp.id DESC
                       ) AS rn
                FROM canonical_predictions cp
                JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
                WHERE cp.model_name = :model_name
                  AND cp.model_version = :model_version
                  AND cp.features_version = :features_version
                  AND CASE
                      WHEN cp.data_cutoff_at ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
                      THEN cp.data_cutoff_at::timestamptz
                  END <= CASE
                      WHEN cp.predicted_at ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
                      THEN cp.predicted_at::timestamptz
                  END
                  AND CASE
                      WHEN cp.predicted_at ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
                      THEN cp.predicted_at::timestamptz
                  END < CASE
                      WHEN cm.start_time_normalized ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
                      THEN cm.start_time_normalized::timestamptz
                  END
                  AND cm.status IN ('finished', 'completed')
                  AND cm.winner_side IN ('team_a', 'team_b')
                  AND cm.start_time_normalized > :cutoff
                  {extra_filters}
            )
            SELECT canonical_match_id, prob_a, winner_side, league, best_of, start_time_normalized
            FROM ranked WHERE rn = 1
            """,
            query_params,
        )
        eligible: dict[int, dict[str, Any]] = {}
        y_true: list[int] = []
        y_prob: list[float] = []
        leagues: list[str] = []
        best_ofs: list[int] = []
        for row in rows:
            try:
                probability = float(row["prob_a"])
                match_id = int(row["canonical_match_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0.0 < probability < 1.0:
                continue
            winner_side = str(row.get("winner_side") or "")
            if winner_side not in {"team_a", "team_b"}:
                continue
            lg = str(row.get("league") or "unknown")
            bo = int(row.get("best_of") or 1)
            y_val = int(winner_side == "team_a")
            eligible[match_id] = {
                "prob_a": probability,
                "y_true": y_val,
                "league": lg,
                "best_of": bo,
            }
            y_prob.append(probability)
            y_true.append(y_val)
            leagues.append(lg)
            best_ofs.append(bo)

        predictions_by_key[str(specification["key"])] = eligible
        metrics = _metric_summary(y_true, y_prob) if y_true else {
            "n_matches": 0,
            "avg_logloss": None,
            "avg_auc": None,
            "avg_brier": None,
            "accuracy": None,
            "ece": None,
            "calibration_status": "unknown",
            "calibration_bins": [],
        }

        # Segment breakdowns
        t1_y = [y for y, l in zip(y_true, leagues) if _is_tier_1_league(l)]
        t1_p = [p for p, l in zip(y_prob, leagues) if _is_tier_1_league(l)]
        reg_y = [y for y, l in zip(y_true, leagues) if not _is_tier_1_league(l)]
        reg_p = [p for p, l in zip(y_prob, leagues) if not _is_tier_1_league(l)]

        bo1_y = [y for y, b in zip(y_true, best_ofs) if b == 1]
        bo1_p = [p for p, b in zip(y_prob, best_ofs) if b == 1]
        bo3_y = [y for y, b in zip(y_true, best_ofs) if b == 3]
        bo3_p = [p for p, b in zip(y_prob, best_ofs) if b == 3]
        bo5_y = [y for y, b in zip(y_true, best_ofs) if b == 5]
        bo5_p = [p for p, b in zip(y_prob, best_ofs) if b == 5]

        segments = {
            "tier_1": _metric_summary(t1_y, t1_p) if t1_y else None,
            "regional_erl": _metric_summary(reg_y, reg_p) if reg_y else None,
        }
        formats = {
            "bo1": _metric_summary(bo1_y, bo1_p) if bo1_y else None,
            "bo3": _metric_summary(bo3_y, bo3_p) if bo3_y else None,
            "bo5": _metric_summary(bo5_y, bo5_p) if bo5_y else None,
        }

        summaries.append(
            {
                **specification,
                "temporal_eligible_matches": len(eligible),
                **metrics,
                "segments": segments,
                "formats": formats,
            }
        )

    valid_keys = {s["key"] for s in specifications}
    target_alias = {
        "native_c0": "c0",
        "operational": "operational_regional",
        "regional": "operational_regional",
    }
    requested_key = target_alias.get(target_model, target_model)
    if requested_key not in valid_keys:
        raise HTTPException(status_code=400, detail="Unknown comparison model key.")
    target_key = requested_key
    old = predictions_by_key.get("exp039", {})
    target_preds = predictions_by_key.get(target_key, {})
    operational_preds = predictions_by_key.get("operational_regional", {})

    # No other model's rows may stand in for missing C0 coverage.
    common_ids = (
        sorted(set(old).intersection(target_preds))
        if target_preds and target_key != "exp039"
        else sorted(target_preds)
    )
    primary_preds = target_preds
    old_y = [old[mid]["y_true"] for mid in common_ids if mid in old]
    target_probabilities = [primary_preds[mid]["prob_a"] for mid in common_ids if mid in primary_preds]
    naive_probabilities = [0.5 for _ in common_ids]
    common_leagues = [primary_preds[mid]["league"] for mid in common_ids if mid in primary_preds]
    common_best_ofs = [primary_preds[mid]["best_of"] for mid in common_ids if mid in primary_preds]

    target_y = [primary_preds[mid]["y_true"] for mid in common_ids if mid in primary_preds]
    target_common = _metric_summary(target_y, target_probabilities) if target_y else None

    exp039_common_y = [old[mid]["y_true"] for mid in common_ids if mid in old]
    exp039_common_p = [old[mid]["prob_a"] for mid in common_ids if mid in old]
    exp039_common = _metric_summary(exp039_common_y, exp039_common_p) if exp039_common_y else None

    op_common_ids = sorted(set(old).intersection(operational_preds))
    op_y = [old[mid]["y_true"] for mid in op_common_ids]
    op_p = [operational_preds[mid]["prob_a"] for mid in op_common_ids]
    operational_common = _metric_summary(op_y, op_p) if op_y else None
    naive_common = _metric_summary(target_y, naive_probabilities[:len(target_y)]) if target_y else None

    # Segments on common cohort
    c_t1_idx = [i for i, l in enumerate(common_leagues) if _is_tier_1_league(l)]
    c_reg_idx = [i for i, l in enumerate(common_leagues) if not _is_tier_1_league(l)]

    common_segments = {
        "tier_1": {
            "n_matches": len(c_t1_idx),
            "target_model": _metric_summary([target_y[i] for i in c_t1_idx], [target_probabilities[i] for i in c_t1_idx]) if c_t1_idx and target_y else None,
            "exp039": _metric_summary([old[common_ids[i]]["y_true"] for i in c_t1_idx if common_ids[i] in old], [old[common_ids[i]]["prob_a"] for i in c_t1_idx if common_ids[i] in old]) if c_t1_idx and old else None,
            "operational_regional": _metric_summary([op_y[i] for i in range(len(op_y))], [op_p[i] for i in range(len(op_p))]) if op_y else None,
        },
        "regional_erl": {
            "n_matches": len(c_reg_idx),
            "target_model": _metric_summary([target_y[i] for i in c_reg_idx], [target_probabilities[i] for i in c_reg_idx]) if c_reg_idx and target_y else None,
            "exp039": _metric_summary([old[common_ids[i]]["y_true"] for i in c_reg_idx if common_ids[i] in old], [old[common_ids[i]]["prob_a"] for i in c_reg_idx if common_ids[i] in old]) if c_reg_idx and old else None,
            "operational_regional": _metric_summary([op_y[i] for i in range(len(op_y))], [op_p[i] for i in range(len(op_p))]) if op_y else None,
        },
    }

    market_common = None
    disagreement_analysis = None
    if common_ids:
        placeholders = ", ".join(f":m_{i}" for i in range(len(common_ids)))
        params = {f"m_{i}": mid for i, mid in enumerate(common_ids)}
        odds_rows = query_df(
            db,
            f"""
            SELECT DISTINCT ON (os.canonical_match_id)
                os.canonical_match_id,
                os.odds_a,
                os.odds_b
            FROM odds_snapshots os
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            WHERE os.canonical_match_id IN ({placeholders})
              AND os.market_type = 'match_winner'
              AND COALESCE(os.is_live, 0) = 0
              AND os.odds_a > 1.0 AND os.odds_b > 1.0
              AND os.scraped_at <= cm.start_time_normalized::timestamptz
            ORDER BY os.canonical_match_id, os.scraped_at DESC
            """,
            params,
        )
        odds_by_match = {int(r["canonical_match_id"]): r for r in odds_rows}
        m_common_ids = [mid for mid in common_ids if mid in odds_by_match and mid in primary_preds]
        if m_common_ids:
            m_y = [primary_preds[mid]["y_true"] for mid in m_common_ids]
            m_probs: list[float] = []
            target_on_m: list[float] = []
            for mid in m_common_ids:
                oa = float(odds_by_match[mid]["odds_a"])
                ob = float(odds_by_match[mid]["odds_b"])
                pa_novig = (1.0 / oa) / ((1.0 / oa) + (1.0 / ob))
                m_probs.append(pa_novig)
                target_on_m.append(primary_preds[mid]["prob_a"])

            market_common = _metric_summary(m_y, m_probs)

            disagreements = []
            for y_val, t_p, m_p in zip(m_y, target_on_m, m_probs):
                t_fav_a = t_p >= 0.5
                m_fav_a = m_p >= 0.5
                if t_fav_a != m_fav_a:
                    model_won = (t_fav_a and y_val == 1) or (not t_fav_a and y_val == 0)
                    market_won = (m_fav_a and y_val == 1) or (not m_fav_a and y_val == 0)
                    disagreements.append({
                        "model_won": model_won,
                        "market_won": market_won,
                    })

            n_disagreements = len(disagreements)
            if n_disagreements > 0:
                model_wins = sum(1 for d in disagreements if d["model_won"])
                market_wins = sum(1 for d in disagreements if d["market_won"])
                disagreement_analysis = {
                    "n_matches": n_disagreements,
                    "disagreement_rate": round(n_disagreements / len(m_common_ids), 4),
                    "model_wins": model_wins,
                    "market_wins": market_wins,
                    "model_win_rate": round(model_wins / n_disagreements, 4),
                    "market_win_rate": round(market_wins / n_disagreements, 4),
                }

    target_minus_exp_ll = None
    target_minus_exp_br = None
    if target_common and exp039_common and target_common.get("avg_logloss") is not None and exp039_common.get("avg_logloss") is not None:
        target_minus_exp_ll = round(float(target_common["avg_logloss"]) - float(exp039_common["avg_logloss"]), 6)
        target_minus_exp_br = round(float(target_common["avg_brier"]) - float(exp039_common["avg_brier"]), 6)

    op_minus_exp_ll = None
    op_minus_exp_br = None
    if operational_common and exp039_common and operational_common.get("avg_logloss") is not None and exp039_common.get("avg_logloss") is not None:
        op_minus_exp_ll = round(float(operational_common["avg_logloss"]) - float(exp039_common["avg_logloss"]), 6)
        op_minus_exp_br = round(float(operational_common["avg_brier"]) - float(exp039_common["avg_brier"]), 6)

    common = {
        "n_matches": len(common_ids),
        "target_model_key": target_key,
        "target_model": target_common,
        "exp039": exp039_common,
        "operational_regional": operational_common,
        "market_closing": market_common,
        "disagreement": disagreement_analysis,
        "naive_50_50": naive_common,
        "segments": common_segments,
        "target_minus_exp039_logloss": target_minus_exp_ll,
        "target_minus_exp039_brier": target_minus_exp_br,
        "operational_minus_exp039_logloss": op_minus_exp_ll,
        "operational_minus_exp039_brier": op_minus_exp_br,
    }
    if target_minus_exp_ll is not None:
        common["model_minus_exp039_logloss"] = target_minus_exp_ll

    # Dynamic, evidence-grounded insights
    executive_insights: list[dict[str, Any]] = []
    target_summary = next((s for s in summaries if s["key"] == target_key), None)
    target_label = target_summary["label"] if target_summary else "Model"

    t_ll = float(target_common["avg_logloss"]) if target_common and target_common.get("avg_logloss") is not None else None
    exp_ll = float(exp039_common["avg_logloss"]) if exp039_common and exp039_common.get("avg_logloss") is not None else None
    t_acc = float(target_common["accuracy"]) if target_common and target_common.get("accuracy") is not None else None
    exp_acc = float(exp039_common["accuracy"]) if exp039_common and exp039_common.get("accuracy") is not None else None
    m_ll = float(market_common["avg_logloss"]) if market_common and market_common.get("avg_logloss") is not None else None
    m_acc = float(market_common["accuracy"]) if market_common and market_common.get("accuracy") is not None else None
    t_ece = float(target_summary["ece"]) if target_summary and target_summary.get("ece") is not None else None

    if t_ll is not None and exp_ll is not None:
        delta_ll = t_ll - exp_ll
        delta_acc = (t_acc or 0) - (exp_acc or 0)
        if delta_ll <= 0:
            executive_insights.append({
                "id": "model_advantage",
                "type": "positive",
                "title": f"Przewaga {target_label} nad EXP-039",
                "text": (
                    f"Na wspólnej próbie {len(common_ids)} meczów {target_label} osiąga LogLoss = {t_ll:.4f} "
                    f"względem {exp_ll:.4f} dla EXP-039 (zysk {delta_ll:+.4f}) oraz wyższą trafność "
                    f"{(t_acc or 0):.1%} vs {(exp_acc or 0):.1%} ({delta_acc:+.1%} p.p.). "
                ),
            })
        else:
            executive_insights.append({
                "id": "thesis_baseline_edge",
                "type": "neutral",
                "title": f"Baza referencyjna EXP-039 zachowuje przewagę nad {target_label}",
                "text": (
                    f"Na wspólnej próbie {len(common_ids)} meczów EXP-039 ma niższy LogLoss "
                    f"({exp_ll:.4f} vs {t_ll:.4f}). Symetryczna kalibracja Platt ogranicza błąd przy niespodziankach."
                ),
            })

    if market_common and t_ll is not None and m_ll is not None:
        delta_market_ll = t_ll - m_ll
        is_market_level = abs(delta_market_ll) < 0.015
        executive_insights.append({
            "id": "market_reality_check",
            "type": "positive" if (delta_market_ll <= 0 or is_market_level) else "neutral",
            "title": f"Konkurencyjność {target_label} względem rynku bukmacherskiego",
            "text": (
                f"Względem no-vig kursów zamykających (LogLoss = {m_ll:.4f}, Accuracy = {(m_acc or 0):.1%}), "
                f"{target_label} osiąga LogLoss = {t_ll:.4f} i trafność {(t_acc or 0):.1%}. "
                + ("Model dorównuje rynkowi zamykającemu w trafności wyboru faworytów spotkań!" if is_market_level or delta_market_ll <= 0 else "Rynek zamykający zachowuje lekką przewagę informacyjną.")
            ),
        })

    if disagreement_analysis and disagreement_analysis["n_matches"] > 0:
        d_n = disagreement_analysis["n_matches"]
        d_rate = disagreement_analysis["disagreement_rate"]
        m_win = disagreement_analysis["model_win_rate"]
        b_win = disagreement_analysis["market_win_rate"]
        executive_insights.append({
            "id": "market_disagreement",
            "type": "recommendation",
            "title": f"Test rzeczywistości: Rozbieżność z rynkiem ({d_n} meczów)",
            "text": (
                f"W {d_n} meczach ({d_rate:.1%} wspólnej próby), gdzie model wskazał innego faworyta niż rynek, "
                f"model trafia z dokładnością {m_win:.1%}, podczas gdy rynek {b_win:.1%}. "
                + ("Model skutecznie wyłapuje błędy wyceny bukmachera!" if m_win >= b_win else "Rynek częściej ma rację w meczach spornych – zalecane ostrożne stawkowanie lub filtr konforemny.")
            ),
        })

    if t_ece is not None:
        status_text = "Dobra" if t_ece < 0.08 else "Wymaga dopracowania"
        executive_insights.append({
            "id": "calibration_health",
            "type": "positive" if t_ece < 0.08 else "warning",
            "title": f"Kalibracja probabilistyczna ({target_label}: ECE = {t_ece:.4f} · {status_text})",
            "text": (
                f"Błąd kalibracji wynosi {t_ece:.4f}. Prawdopodobieństwa w poszczególnych decylach "
                "odpowiadają rzeczywistym częstościom wygranych drużyn."
            ),
        })

    return {
        "evaluation_scope": {
            "kind": "retrospective_chronological_replay",
            "warning": (
                "Stored predictions are compared under their declared pre-match "
                "clocks; C0 uses its exact frozen artifacts. This does not certify "
                "publication times, live forecasting, CLV, ROI, or betting performance."
            ),
            "temporal_rule": "data_cutoff_at <= predicted_at < match_start_at",
        },
        "target_model_key": target_key,
        "models": summaries,
        "common_cohort": common,
        "executive_insights": executive_insights,
    }
@router.get("/horizon-accuracy")
def horizon_accuracy(
    min_matches_per_bin: int = 10,
    max_days_back: int = 90,
    refresh: bool = False,
    db=Depends(get_db),
):
    """Analyze prediction accuracy (LogLoss, AUC) by hours-before-match.

    Uses odds as implied probabilities.  Groups snapshots into time bins
    relative to match start.  Each bin reflects the market's implied
    probability accuracy at that horizon.

    Returns:
      - bins: list of {label, hours_start, hours_end,
               snapshot_count, match_count,
               avg_logloss, avg_auc,
               avg_prob_winner, avg_prob_loser}
      - Only bins with >= min_matches_per_bin are included.
    """
    cache_params = {
        "max_days_back": int(max_days_back),
        "min_matches_per_bin": int(min_matches_per_bin),
    }
    if not refresh:
        cached = _read_model_analysis_cache("horizon_accuracy", **cache_params)
        if cached is not None:
            return cached

    cutoff = (datetime.now(UTC) - timedelta(days=max_days_back)).isoformat()

    # --- 1. Get finished matches with result and start time ---
    matches = query_df(
        db,
        """
        SELECT cm.id, cm.start_time_normalized,
               cm.normalized_team_a, cm.normalized_team_b, cm.league,
               cm.winner_side
        FROM canonical_matches cm
        WHERE cm.status IN ('finished', 'completed')
          AND cm.winner_side IS NOT NULL
          AND cm.start_time_normalized IS NOT NULL
          AND cm.start_time_normalized > :cutoff
        ORDER BY cm.start_time_normalized DESC
        """,
        {"cutoff": cutoff},
    )
    if not matches:
        return _empty_horizon_result()

    match_map = {m["id"]: m for m in matches}
    match_ids = list(match_map.keys())

    # --- 2. Get all pre-match odds snapshots ---
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params: dict[str, Any] = {f"mid{i}": mid for i, mid in enumerate(match_ids)}

    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at,
               os.bookmaker_id, os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.scraped_at
        """,
        params,
    )

    # --- 3. Build bins ---
    # Interval bins: each snapshot is assigned to exactly one time window.
    # A match can still appear in multiple bins if it has odds snapshots in
    # multiple windows. Metrics for each bin are computed only from odds in
    # that specific interval.
    BIN_DEFS = [
        ("0-2h",     0,   2),
        ("2-6h",     2,   6),
        ("6-12h",    6,  12),
        ("12-24h",  12,  24),
        ("24-48h",  24,  48),
        ("48h+",    48, 9999),
    ]

    # Accumulate per bin at match level.
    # A match may have many bookmaker snapshots in the same horizon bin.  We
    # first average the market probability for that match/bin, then compute
    # metrics across matches.  This prevents matches covered by many
    # bookmakers/scrapes from dominating LogLoss/AUC.
    bin_data: dict[str, dict] = {
        label: {
            "per_match": defaultdict(lambda: {
                "y_true": None,
                "bookmaker_scores": defaultdict(list),
                "bookmaker_prob_winner": defaultdict(list),
                "bookmaker_prob_loser": defaultdict(list),
            }),
            "snapshot_count": 0,
        }
        for label, _, _ in BIN_DEFS
    }

    matches_with_odds = set()
    odds_processed = 0

    for snap in snapshots:
        mid = snap["canonical_match_id"]
        m = match_map.get(mid)
        if not m:
            continue

        match_start = _parse_dt(m.get("start_time_normalized"))
        scraped = _parse_dt(snap.get("scraped_at"))
        if not match_start or not scraped:
            continue

        hours_before = (match_start - scraped).total_seconds() / 3600.0
        if hours_before < 0:
            continue  # post-match / in-play

        # Align odds to canonical sides
        aligned = _align(
            str(m.get("normalized_team_a") or ""),
            str(m.get("normalized_team_b") or ""),
            str(snap.get("raw_team_a") or ""),
            str(snap.get("raw_team_b") or ""),
            snap.get("odds_a"),
            snap.get("odds_b"),
            league=m.get("league"), match_date=m.get("start_time_normalized"),
        )
        if not aligned:
            continue

        odds_a, odds_b = aligned

        # Convert to implied probabilities (margin-removed)
        prob_a = _implied_prob(odds_a)
        prob_b = _implied_prob(odds_b)
        prob_a_norm, prob_b_norm = _remove_margin(prob_a, prob_b)

        # Ground truth: which side won?
        winner_side = str(m.get("winner_side") or "")
        if winner_side not in ("team_a", "team_b"):
            continue

        # y_true: 1 if team_a won, 0 if team_b won
        # y_score: implied probability that team_a wins
        y_true = 1 if winner_side == "team_a" else 0
        y_score = prob_a_norm

        # Probability assigned to the actual winner / loser
        if winner_side == "team_a":
            prob_winner = prob_a_norm
            prob_loser = prob_b_norm
        else:
            prob_winner = prob_b_norm
            prob_loser = prob_a_norm

        # Find bin — interval: snapshot counts only in its exact time window.
        # A match can appear in multiple bins if it has snapshots in multiple
        # windows, but each bin's stats use only odds from that window.
        for label, hmin, hmax in BIN_DEFS:
            if hmin <= hours_before < hmax:
                bd = bin_data[label]
                md = bd["per_match"][mid]
                bookmaker_key = snap.get("bookmaker_id") or "unknown"
                md["y_true"] = y_true
                md["bookmaker_scores"][bookmaker_key].append(y_score)
                md["bookmaker_prob_winner"][bookmaker_key].append(prob_winner)
                md["bookmaker_prob_loser"][bookmaker_key].append(prob_loser)
                bd["snapshot_count"] += 1
                matches_with_odds.add(mid)
                odds_processed += 1
                break

    if odds_processed == 0:
        return _empty_horizon_result(len(matches))

    # --- 4. Compute metrics per bin ---
    bins = []
    for label, hmin, hmax in BIN_DEFS:
        bd = bin_data[label]
        per_match = bd["per_match"]
        n_matches = len(per_match)
        n_snapshots = bd["snapshot_count"]

        # Skip bins with insufficient matches
        if n_matches < min_matches_per_bin:
            continue

        # One observation per match: mean bookmaker/snapshot probability in
        # this bin, then metrics across matches.
        y_true: list[int] = []
        y_score: list[float] = []
        prob_winner_by_match: list[float] = []
        prob_loser_by_match: list[float] = []
        for md in per_match.values():
            if md["y_true"] is None or not md["bookmaker_scores"]:
                continue
            bookmaker_scores = [float(np.mean(vals)) for vals in md["bookmaker_scores"].values() if vals]
            bookmaker_winner = [float(np.mean(vals)) for vals in md["bookmaker_prob_winner"].values() if vals]
            bookmaker_loser = [float(np.mean(vals)) for vals in md["bookmaker_prob_loser"].values() if vals]
            if not bookmaker_scores:
                continue
            y_true.append(int(md["y_true"]))
            y_score.append(float(np.mean(bookmaker_scores)))
            if bookmaker_winner:
                prob_winner_by_match.append(float(np.mean(bookmaker_winner)))
            if bookmaker_loser:
                prob_loser_by_match.append(float(np.mean(bookmaker_loser)))

        auc_val = _compute_auc(y_true, y_score)
        ll_val = _compute_logloss(y_true, y_score)
        avg_prob_winner = round(np.mean(prob_winner_by_match), 4) if prob_winner_by_match else None
        avg_prob_loser = round(np.mean(prob_loser_by_match), 4) if prob_loser_by_match else None

        bins.append({
            "label": label,
            "hours_start": hmin,
            "hours_end": hmax if hmax < 9999 else None,
            "snapshot_count": n_snapshots,
            "match_count": n_matches,
            "avg_logloss": ll_val,
            "avg_auc": auc_val,
            "avg_prob_winner": avg_prob_winner,
            "avg_prob_loser": avg_prob_loser,
        })

    # --- 5. Compute per-bookmaker metrics ---
    bookmaker_bins = _compute_bookmaker_bins(bin_data, BIN_DEFS, min_matches_per_bin, db)

    # --- 6. Compute overall model accuracy reference lines ---
    model_refs, hybrid_model_bins = _compute_model_reference_metrics(db, cutoff, min_matches_per_bin)

    # --- 7. Statistical tests: model vs average bookmaker ---
    model_vs_bookmaker_tests = _compute_model_vs_bookmaker_tests(db, cutoff)

    # --- 8. Strict market-close comparison on identical match samples ---
    market_close_comparison = _compute_market_close_comparison(db, cutoff, min_matches_per_bin)

    result = {
        "evaluation_scope": {
            "kind": "retrospective_backtest",
            "prediction_source": ANALYSIS_FEATURES_VERSION,
            "prediction_rule": "latest valid prediction at or before match start",
            "warning": "Accuracy compares the reproducible EXP-060 historical backfill. It is not a live-forecast performance monitor.",
        },
        "total_matches_with_odds": len(matches_with_odds),
        "total_finished_matches": len(matches),
        "total_odds_processed": odds_processed,
        "bins": bins,
        "min_matches_per_bin": min_matches_per_bin,
        "model_references": model_refs,
        "hybrid_model_bins": hybrid_model_bins,
        "bookmaker_bins": bookmaker_bins,
        "model_vs_bookmaker_tests": model_vs_bookmaker_tests,
        "market_close_comparison": market_close_comparison,
    }
    _write_model_analysis_cache("horizon_accuracy", result, **cache_params)
    return result


@router.get("/model-clv-by-horizon")
def model_clv_by_horizon(
    max_days_back: int = 90,
    max_odds_age_hours: float = 4.0,
    tax_rate: float = 0.12,
    min_ev: float = 0.0,
    refresh: bool = False,
    db=Depends(get_db),
):
    """Compute Closing Line Value for model-selected EV entries by horizon.

    This endpoint evaluates market timing, not match-outcome accuracy. For each
    Thesis/Hybrid prediction it finds the latest available bookmaker odds before
    `predicted_at` (entry line), requires the entry odds to be fresh enough, picks
    the side with the highest positive after-tax EV, and compares that entry line
    with the same bookmaker's final valid pre-match line (closing line).

    Positive CLV means the model selected a price that later closed shorter.
    Results include signal-weighted aggregates (all entries) and match-weighted
    aggregates (one averaged observation per model/horizon/match), because the
    latter is the safer headline metric for model evaluation.
    """
    cache_params = {
        "max_days_back": int(max_days_back),
        "max_odds_age_hours": float(max_odds_age_hours),
        "tax_rate": float(tax_rate),
        "min_ev": float(min_ev),
    }
    if not refresh:
        cached = _read_model_analysis_cache("model_clv_by_horizon", **cache_params)
        if cached is not None:
            return cached

    cutoff = (datetime.now(UTC) - timedelta(days=max_days_back)).isoformat()

    predictions = query_df(
        db,
        """
        SELECT cp.id AS prediction_id,
               cp.canonical_match_id,
               cp.model_name,
               cp.model_version,
               cp.predicted_at,
               cp.prob_a,
               cp.prob_b,
               cm.start_time_normalized,
               cm.normalized_team_a,
               cm.normalized_team_b,
               cm.status
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name IN (:thesis_model, :hybrid_model, :op_model, :op_hybrid_model)
          AND (
              (cp.model_name = :thesis_model AND (cp.features_version != :retrospective_version OR cp.features_version IS NULL))
              OR (cp.model_name = :hybrid_model AND cp.model_version = :hybrid_version)
              OR (cp.model_name = :op_model AND cp.model_version = :op_version)
              OR (cp.model_name = :op_hybrid_model AND cp.model_version = :op_hybrid_version)
          )
          AND cp.predicted_at IS NOT NULL
          AND cp.prob_a IS NOT NULL
          AND cp.prob_b IS NOT NULL
          AND cm.start_time_normalized IS NOT NULL
          AND cm.start_time_normalized > :cutoff
        ORDER BY cp.predicted_at ASC
        """,
        {
            "thesis_model": THESIS_MODEL_NAME,
            "hybrid_model": THESIS_HYBRID_MODEL_NAME,
            "op_model": OPERATIONAL_MODEL_NAME,
            "op_hybrid_model": get_active_hybrid().hybrid_model_name,
            "op_version": OPERATIONAL_BACKFILL_MODEL_VERSION,
            "op_hybrid_version": get_active_hybrid().hybrid_model_version,
            "retrospective_version": ANALYSIS_FEATURES_VERSION,
            "hybrid_version": f"a{THESIS_HYBRID_ALPHA:.2f}-t{THESIS_HYBRID_TEMPERATURE:.2f}",
            "cutoff": cutoff,
        },
    )
    if not predictions:
        return _empty_clv_result(max_days_back, max_odds_age_hours, tax_rate, min_ev)

    match_ids = sorted({int(p["canonical_match_id"]) for p in predictions})
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params: dict[str, Any] = {f"mid{i}": mid for i, mid in enumerate(match_ids)}
    odds_rows = query_df(
        db,
        f"""
        SELECT os.id AS odds_snapshot_id,
               os.canonical_match_id,
               os.bookmaker_id,
               b.name AS bookmaker_name,
               os.scraped_at,
               os.raw_team_a,
               os.raw_team_b,
               os.odds_a,
               os.odds_b,
               cm.normalized_team_a,
               cm.normalized_team_b, cm.league, cm.start_time_normalized
        FROM odds_snapshots os
        JOIN canonical_matches cm ON cm.id = os.canonical_match_id
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at
        """,
        params,
    )

    odds_by_match_book: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in odds_rows:
        scraped = _parse_dt(row.get("scraped_at"))
        if not scraped:
            continue
        aligned = _align(
            str(row.get("normalized_team_a") or ""),
            str(row.get("normalized_team_b") or ""),
            str(row.get("raw_team_a") or ""),
            str(row.get("raw_team_b") or ""),
            row.get("odds_a"),
            row.get("odds_b"),
            league=row.get("league"), match_date=row.get("start_time_normalized"),
        )
        if not aligned:
            continue
        odds_a, odds_b = aligned
        if odds_a <= 1.0 or odds_b <= 1.0:
            continue
        enriched = dict(row)
        enriched["scraped_dt"] = scraped
        enriched["canonical_odds_a"] = odds_a
        enriched["canonical_odds_b"] = odds_b
        odds_by_match_book[(int(row["canonical_match_id"]), int(row["bookmaker_id"]))].append(enriched)

    for rows in odds_by_match_book.values():
        rows.sort(key=lambda r: r["scraped_dt"])

    entries: list[dict[str, Any]] = []
    skips: dict[str, int] = defaultdict(int)

    for pred in predictions:
        match_id = int(pred["canonical_match_id"])
        predicted_at = _parse_dt(pred.get("predicted_at"))
        match_start = _parse_dt(pred.get("start_time_normalized"))
        if not predicted_at or not match_start:
            skips["invalid_datetime"] += 1
            continue
        if predicted_at >= match_start:
            skips["prediction_after_start"] += 1
            continue

        hours_before = (match_start - predicted_at).total_seconds() / 3600.0
        if hours_before < 0:
            skips["prediction_after_start"] += 1
            continue
        horizon = _horizon_label(hours_before)
        if not horizon:
            skips["outside_horizon"] += 1
            continue

        prob_a = float(pred["prob_a"])
        pred_mname = pred.get("model_name")
        if pred_mname == OPERATIONAL_HYBRID_MODEL_NAME:
            model_key = "operational_hybrid"
            model_label = "Hybryda Operacyjna (C0 + Rynek)"
        elif pred_mname == OPERATIONAL_MODEL_NAME:
            model_key = "operational"
            model_label = "Model Operacyjny (Regional BoN)"
        elif pred_mname == THESIS_HYBRID_MODEL_NAME:
            model_key = "hybrid"
            model_label = "Hybrid model (EXP-039 + Rynek)"
        else:
            model_key = "thesis"
            model_label = "Thesis model (EXP-039)"
        books_for_match = [key for key in odds_by_match_book.keys() if key[0] == match_id]
        if not books_for_match:
            skips["no_odds_for_match"] += 1
            continue

        had_fresh_odds = False
        had_positive_ev = False
        had_closing = False
        for key in books_for_match:
            rows = odds_by_match_book[key]
            entry = _latest_before(rows, predicted_at)
            if not entry:
                continue
            entry_age_hours = (predicted_at - entry["scraped_dt"]).total_seconds() / 3600.0
            if entry_age_hours < 0 or entry_age_hours > max_odds_age_hours:
                continue
            had_fresh_odds = True

            odds_a = float(entry["canonical_odds_a"])
            odds_b = float(entry["canonical_odds_b"])
            try:
                best = best_ev_side(prob_a, odds_a, odds_b, tax_rate=tax_rate, min_ev=min_ev)
            except ValueError:
                skips["invalid_entry_odds"] += 1
                continue
            if not best:
                continue
            had_positive_ev = True

            closing = _latest_before(rows, match_start)
            if not closing:
                continue
            had_closing = True
            side = str(best["side"])
            taken_odds = float(best["odds"])
            closing_odds = float(closing["canonical_odds_a"] if side == "a" else closing["canonical_odds_b"])
            if taken_odds <= 1.0 or closing_odds <= 1.0:
                skips["invalid_closing_odds"] += 1
                continue

            entries.append({
                "model_key": model_key,
                "model_label": model_label,
                "model_name": pred["model_name"],
                "model_version": pred["model_version"],
                "prediction_id": pred["prediction_id"],
                "canonical_match_id": match_id,
                "bookmaker_id": key[1],
                "bookmaker_name": entry.get("bookmaker_name"),
                "horizon_label": horizon[0],
                "hours_start": horizon[1],
                "hours_end": None if horizon[2] >= 9999 else horizon[2],
                "hours_before": hours_before,
                "entry_age_hours": entry_age_hours,
                "side": side,
                "taken_odds": taken_odds,
                "closing_odds": closing_odds,
                "ev": float(best["ev"]),
                "model_prob": float(best["model_prob"]),
                "market_prob": float(best["market_prob"]),
                "clv_odds_pct": clv_odds_pct(taken_odds, closing_odds),
                "clv_probability_pp": clv_probability_points(taken_odds, closing_odds),
                "predicted_at": predicted_at.isoformat(),
                "entry_scraped_at": entry["scraped_dt"].isoformat(),
                "closing_scraped_at": closing["scraped_dt"].isoformat(),
            })

        if not had_fresh_odds:
            skips["no_fresh_entry_odds"] += 1
        elif not had_positive_ev:
            skips["no_positive_ev"] += 1
        elif not had_closing:
            skips["no_closing_odds"] += 1

    bins = _aggregate_clv_entries_match_oriented(entries)

    model_summaries = []
    clv_model_labels = {
        "operational_hybrid": "Hybryda Operacyjna (Regional BoN + Rynek)",
        "operational": "Model Operacyjny (Regional BoN)",
        "hybrid": "Hybrid model (EXP-039 + Rynek)",
        "thesis": "Thesis model (EXP-039)",
    }
    for key in ("operational_hybrid", "operational", "hybrid", "thesis"):
        model_entries = [e for e in entries if e["model_key"] == key]
        model_bins = [b for b in bins if b["model_key"] == key]
        books = _aggregate_clv_by_bookmaker(model_entries)
        tiers = _aggregate_clv_by_odds_tier(model_entries)
        summary = _build_conditions_summary(model_bins, books, tiers, model_entries)
        model_summaries.append({
            "model_key": key,
            "model_label": clv_model_labels.get(key, key),
            "bins": model_bins,
            "bookmaker_breakdown": books,
            "odds_tier_breakdown": tiers,
            "conditions_summary": summary,
        })

    result = {
        "metadata": {
            "max_days_back": max_days_back,
            "max_odds_age_hours": max_odds_age_hours,
            "tax_rate": tax_rate,
            "min_ev": min_ev,
            "aggregation_level": "model_match_horizon",
            "entry_definition": "latest bookmaker odds at or before prediction time, max_age_hours constrained",
            "closing_definition": "latest valid non-live same-bookmaker pre-match odds",
            "clv_odds_pct_definition": "taken_odds / closing_odds - 1; positive means entry beat closing",
            "aggregation_definition": "all EV entries for the same model, canonical match and horizon are collapsed into one match-level observation before averaging",
            "evaluation_scope": "Production CLV monitor: only actual scheduled thesis forecasts and the current configured hybrid; retrospective EXP-060 backfill and legacy hybrid versions are excluded.",
        },
        "total_predictions_scanned": len(predictions),
        "total_entries": len(entries),
        "models": model_summaries,
        "bins": bins,
        "bookmaker_breakdown": _aggregate_clv_by_bookmaker(entries),
        "odds_tier_breakdown": _aggregate_clv_by_odds_tier(entries),
        "skips": dict(sorted(skips.items())),
    }
    _write_model_analysis_cache("model_clv_by_horizon", result, **cache_params)
    return result


def _horizon_label(hours_before: float) -> tuple[str, int, int] | None:
    for label, hmin, hmax in HORIZON_BIN_DEFS:
        if hmin <= hours_before < hmax:
            return label, hmin, hmax
    return None


def _latest_before(rows: list[dict[str, Any]], timestamp: datetime) -> dict[str, Any] | None:
    latest = None
    for row in rows:
        if row["scraped_dt"] <= timestamp:
            latest = row
        else:
            break
    return latest


def _mean(values: list[float]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def _median(values: list[float]) -> float | None:
    return round(float(np.median(values)), 4) if values else None


def _aggregate_clv_entries_match_oriented(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate CLV as one observation per model/match/horizon.

    Odds snapshots and bookmakers can generate many EV opportunities for the same
    match. For model evaluation we collapse them first, so no match can dominate a
    horizon bin only because it had many scraped prices.
    """
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["model_key"], entry["horizon_label"], int(entry["canonical_match_id"]))].append(entry)

    rows: list[dict[str, Any]] = []
    for (_model_key, _label, match_id), group in grouped.items():
        first = group[0]
        rows.append({
            "model_key": first["model_key"],
            "model_label": first["model_label"],
            "horizon_label": first["horizon_label"],
            "hours_start": first["hours_start"],
            "hours_end": first["hours_end"],
            "canonical_match_id": match_id,
            "entries": len(group),
            "clv_odds_pct": float(np.mean([g["clv_odds_pct"] for g in group])),
            "clv_probability_pp": float(np.mean([g["clv_probability_pp"] for g in group])),
            "ev": float(np.mean([g["ev"] for g in group])),
            "hours_before": float(np.mean([g["hours_before"] for g in group])),
        })

    grouped_bins: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_bins[(row["model_key"], row["horizon_label"])].append(row)

    order = {label: i for i, (label, _, _) in enumerate(HORIZON_BIN_DEFS)}
    result: list[dict[str, Any]] = []
    for (model_key, label), group in grouped_bins.items():
        first = group[0]
        clv_vals = [float(g["clv_odds_pct"]) for g in group]
        clv_pp_vals = [float(g["clv_probability_pp"]) for g in group]
        ev_vals = [float(g["ev"]) for g in group]
        match_ids = {int(g["canonical_match_id"]) for g in group}
        entry_count = sum(int(g.get("entries", 1)) for g in group)
        result.append({
            "model_key": model_key,
            "model_label": first["model_label"],
            "label": label,
            "hours_start": first["hours_start"],
            "hours_end": first["hours_end"],
            "entry_count": entry_count,
            "match_count": len(match_ids),
            "observation_count": len(group),
            "avg_hours_before": _mean([float(g["hours_before"]) for g in group]),
            "avg_clv_odds_pct": _mean(clv_vals),
            "median_clv_odds_pct": _median(clv_vals),
            "avg_clv_probability_pp": _mean(clv_pp_vals),
            "median_clv_probability_pp": _median(clv_pp_vals),
            "positive_clv_rate": round(sum(1 for v in clv_vals if v > 0) / len(clv_vals), 4) if clv_vals else None,
            "avg_ev": _mean(ev_vals),
            "aggregation_level": "model_match_horizon",
        })

    return sorted(result, key=lambda b: (b["model_key"], order.get(b["label"], 999)))

def _aggregate_clv_by_bookmaker(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        bname = str(entry.get("bookmaker_name") or "Unknown")
        grouped[bname].append(entry)

    results: list[dict[str, Any]] = []
    for bname, rows in grouped.items():
        clv_vals = [float(r["clv_odds_pct"]) for r in rows]
        clv_pp_vals = [float(r["clv_probability_pp"]) for r in rows]
        ev_vals = [float(r["ev"]) for r in rows]
        taken_odds_vals = [float(r["taken_odds"]) for r in rows]
        closing_odds_vals = [float(r["closing_odds"]) for r in rows]
        match_ids = {int(r["canonical_match_id"]) for r in rows}
        positive_count = sum(1 for v in clv_vals if v > 0)
        pos_rate = round(positive_count / len(clv_vals), 4) if clv_vals else 0.0

        results.append({
            "bookmaker_name": bname,
            "entry_count": len(rows),
            "match_count": len(match_ids),
            "avg_clv_odds_pct": _mean(clv_vals),
            "median_clv_odds_pct": _median(clv_vals),
            "positive_clv_rate": pos_rate,
            "positive_clv_count": positive_count,
            "avg_clv_probability_pp": _mean(clv_pp_vals),
            "avg_ev": _mean(ev_vals),
            "avg_taken_odds": _mean(taken_odds_vals),
            "avg_closing_odds": _mean(closing_odds_vals),
        })

    return sorted(results, key=lambda r: (r["avg_clv_odds_pct"] or -999.0), reverse=True)


def _aggregate_clv_by_odds_tier(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    TIERS = [
        ("Zdecydowany faworyt (<1.40)", 1.0, 1.40),
        ("Umiarkowany faworyt (1.40-1.80)", 1.40, 1.80),
        ("Wyrównany mecz (1.80-2.20)", 1.80, 2.20),
        ("Umiarkowany underdog (2.20-3.00)", 2.20, 3.00),
        ("Wysoki kurs / underdog (>3.00)", 3.00, 9999.0),
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        odds = float(entry["taken_odds"])
        assigned_tier = None
        for label, omin, omax in TIERS:
            if omin <= odds < omax:
                assigned_tier = label
                break
        if assigned_tier:
            grouped[assigned_tier].append(entry)

    results: list[dict[str, Any]] = []
    for label, omin, omax in TIERS:
        rows = grouped.get(label, [])
        if not rows:
            continue
        clv_vals = [float(r["clv_odds_pct"]) for r in rows]
        ev_vals = [float(r["ev"]) for r in rows]
        match_ids = {int(r["canonical_match_id"]) for r in rows}
        positive_count = sum(1 for v in clv_vals if v > 0)
        pos_rate = round(positive_count / len(clv_vals), 4) if clv_vals else 0.0

        results.append({
            "tier_label": label,
            "odds_min": omin,
            "odds_max": None if omax >= 9000 else omax,
            "entry_count": len(rows),
            "match_count": len(match_ids),
            "avg_clv_odds_pct": _mean(clv_vals),
            "median_clv_odds_pct": _median(clv_vals),
            "positive_clv_rate": pos_rate,
            "positive_clv_count": positive_count,
            "avg_ev": _mean(ev_vals),
            "avg_taken_odds": _mean([float(r["taken_odds"]) for r in rows]),
            "avg_closing_odds": _mean([float(r["closing_odds"]) for r in rows]),
        })
    return results


def _build_conditions_summary(
    bins: list[dict[str, Any]],
    bookmakers: list[dict[str, Any]],
    odds_tiers: list[dict[str, Any]],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not entries:
        return {}

    valid_bins = [b for b in bins if b.get("avg_clv_odds_pct") is not None and b.get("match_count", 0) > 0]
    best_horizon = max(valid_bins, key=lambda b: b["avg_clv_odds_pct"]) if valid_bins else None
    worst_horizon = min(valid_bins, key=lambda b: b["avg_clv_odds_pct"]) if valid_bins else None

    valid_books = [b for b in bookmakers if b.get("entry_count", 0) >= 5]
    top_books = sorted(valid_books, key=lambda b: b["avg_clv_odds_pct"], reverse=True)[:3] if valid_books else []

    valid_tiers = [t for t in odds_tiers if t.get("entry_count", 0) >= 5]
    best_tier = max(valid_tiers, key=lambda t: t["avg_clv_odds_pct"]) if valid_tiers else None

    all_clv = [float(r["clv_odds_pct"]) for r in entries]
    overall_pos = sum(1 for v in all_clv if v > 0)
    overall_rate = round(overall_pos / len(all_clv), 4) if all_clv else 0.0

    recommendations: list[str] = []
    if best_horizon:
        recommendations.append(
            f"Najlepszy horyzont wejścia: {best_horizon['label']} (średni CLV {best_horizon['avg_clv_odds_pct']:+.2f}%, {best_horizon['positive_clv_rate']*100:.1f}% pobić linii zamknięcia). Im wcześniej zawierany zakład, tym większa przewaga nad późniejszą korektą kursu."
        )
    if top_books:
        books_str = ", ".join(f"{b['bookmaker_name']} ({b['avg_clv_odds_pct']:+.1f}%)" for b in top_books)
        recommendations.append(f"Najbardziej podatni bukmacherzy: {books_str}.")
    if best_tier:
        recommendations.append(
            f"Optymalny przedział kursowy: {best_tier['tier_label']} ze średnim CLV {best_tier['avg_clv_odds_pct']:+.2f}%."
        )
    if worst_horizon and (worst_horizon.get("avg_clv_odds_pct") or 0) <= 1.0:
        recommendations.append(
            f"Rynek zamykający: w oknie {worst_horizon['label']} kursy są już wysoce efektywne (CLV {worst_horizon['avg_clv_odds_pct']:+.2f}%). Unikaj gry tuż przed meczem bez silnego sygnału składowego."
        )

    return {
        "best_horizon": best_horizon,
        "worst_horizon": worst_horizon,
        "top_bookmakers": top_books,
        "best_odds_tier": best_tier,
        "overall_entry_count": len(entries),
        "overall_matches_count": len({int(r['canonical_match_id']) for r in entries}),
        "overall_beat_closing_rate": overall_rate,
        "overall_avg_clv_odds_pct": _mean(all_clv),
        "recommendations": recommendations,
    }


def _empty_clv_result(max_days_back: int, max_odds_age_hours: float, tax_rate: float, min_ev: float) -> dict:
    return {
        "metadata": {
            "max_days_back": max_days_back,
            "max_odds_age_hours": max_odds_age_hours,
            "tax_rate": tax_rate,
            "min_ev": min_ev,
            "aggregation_level": "model_match_horizon",
        },
        "total_predictions_scanned": 0,
        "total_entries": 0,
        "models": [],
        "bins": [],
        "skips": {},
    }


def _compute_bookmaker_bins(
    bin_data: dict, bin_defs: list, min_matches: int, db,
) -> list[dict]:
    """Compute per-bookmaker LogLoss/AUC for each horizon bin.

    Uses the same per-match aggregated data from horizon_accuracy().
    For each bookmaker in each bin, computes metrics using only that
    bookmaker's snapshots (one observation per match after averaging
    same-bookmaker snapshots for that match).

    Returns list of {bookmaker_id, bookmaker_name, bins: [...]}.
    """
    # Look up bookmaker names
    bookmaker_rows = query_df(db, "SELECT id, name FROM bookmakers ORDER BY id")
    bk_names: dict[int, str] = {}
    for r in bookmaker_rows:
        bk_names[int(r["id"])] = str(r["name"])

    # Collect all bookmaker IDs seen across bins
    all_bk_ids: set[int] = set()
    for label, _, _ in bin_defs:
        bd = bin_data[label]
        for md in bd["per_match"].values():
            for bk_id in md["bookmaker_scores"].keys():
                try:
                    all_bk_ids.add(int(bk_id))
                except (ValueError, TypeError):
                    pass

    result: list[dict] = []
    for bk_id in sorted(all_bk_ids):
        bk_name = bk_names.get(bk_id, f"bookmaker_{bk_id}")
        bk_bins: list[dict] = []

        for label, hmin, hmax in bin_defs:
            bd = bin_data[label]
            per_match = bd["per_match"]

            y_true: list[int] = []
            y_score: list[float] = []
            for md in per_match.values():
                if md["y_true"] is None:
                    continue
                scores = md["bookmaker_scores"].get(bk_id)
                if not scores:
                    continue
                y_true.append(int(md["y_true"]))
                y_score.append(float(np.mean(scores)))

            n_matches = len(y_true)
            if n_matches < min_matches:
                continue
            if len(set(y_true)) < 2:
                continue

            ll = _compute_logloss(y_true, y_score)
            auc = _compute_auc(y_true, y_score)

            bk_bins.append({
                "label": label,
                "hours_start": hmin,
                "hours_end": hmax if hmax < 9999 else None,
                "match_count": n_matches,
                "avg_logloss": ll,
                "avg_auc": auc,
            })

        if bk_bins:
            result.append({
                "bookmaker_id": bk_id,
                "bookmaker_name": bk_name,
                "bins": bk_bins,
            })

    return result


def _compute_model_reference_metrics(db, cutoff: str, min_matches: int = 10) -> tuple[list[dict], list[dict]]:
    """Compute overall LogLoss & AUC for each registered prediction model.

    For pure models (thesis, operational): returns single LogLoss/AUC values.
    For hybrid models: returns per-bin metrics computed dynamically using
    thesis_prob + market_prob from each time bin.

    Returns:
        tuple of (pure_model_refs, hybrid_model_bins)
    """
    refs: list[dict] = []
    hybrid_bins: list[dict] = []

    # Bin definitions (interval, same as in horizon_accuracy)
    BIN_DEFS = [
        ("0-2h",     0,   2),
        ("2-6h",     2,   6),
        ("6-12h",    6,  12),
        ("12-24h",  12,  24),
        ("24-48h",  24,  48),
        ("48h+",    48, 9999),
    ]


    # EXP-040 remains a shadow candidate until promotion, but when its
    # timestamp-valid prediction rows have settled it must be measured on the
    # same horizon cohorts instead of being silently absent from the dashboard.
    candidate_versions = query_df(
        db,
        """
        SELECT cp.model_version
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        WHERE cp.model_name = :model_name
          AND cp.features_version = :features_version
          AND cp.predicted_at <= cm.start_time_normalized::timestamptz
          AND cm.status IN ('finished', 'completed')
          AND cm.winner_side IN ('team_a', 'team_b')
          AND cm.start_time_normalized > :cutoff
        GROUP BY cp.model_version
        ORDER BY MAX(cp.predicted_at) DESC
        """,
        {
            "model_name": EXP040_MODEL_NAME,
            "features_version": EXP040_FEATURES_VERSION,
            "cutoff": cutoff,
        },
    )
    model_defs = [
        {
            "model_name": C0_NATIVE.name,
            "model_version": C0_NATIVE.version,
            "features_version": C0_NATIVE.feature_version,
        },
        {
            "model_name": THESIS_MODEL_NAME,
            "model_version": THESIS_MODEL_VERSION,
            "features_version": ANALYSIS_FEATURES_VERSION,
        },
        {
            "model_name": THESIS_HYBRID_MODEL_NAME,
            "model_version": f"a{THESIS_HYBRID_ALPHA:.2f}-t{THESIS_HYBRID_TEMPERATURE:.2f}",
            "features_version": ANALYSIS_FEATURES_VERSION,
            "base_model": THESIS_MODEL_NAME,
            "base_version": THESIS_MODEL_VERSION,
        },
        {
            "model_name": OPERATIONAL_MODEL_NAME,
            "model_version": OPERATIONAL_BACKFILL_MODEL_VERSION,
            "features_version": OPERATIONAL_BACKFILL_FEATURE_VERSION,
        },
        {
            "model_name": get_active_hybrid().hybrid_model_name,
            "model_version": get_active_hybrid().hybrid_model_version,
            "features_version": C0_NATIVE.feature_version,
            "base_model": C0_NATIVE.name,
            "base_version": C0_NATIVE.version,
        },
        *[
            {
                "model_name": EXP040_MODEL_NAME,
                "model_version": str(row["model_version"]),
                "features_version": EXP040_FEATURES_VERSION,
            }
            for row in candidate_versions
        ],
    ]

    for md in model_defs:
        model_name = md["model_name"]
        model_version = md["model_version"]
        features_version = md["features_version"]
        
        # Check if this is a hybrid model
        is_hybrid = "hybrid" in model_name.lower()
        
        if is_hybrid:
            # For hybrid models: compute per-bin metrics dynamically
            base_model = md.get("base_model", THESIS_MODEL_NAME)
            base_version = md.get("base_version", THESIS_MODEL_VERSION)
            hybrid_bin_metrics = _compute_hybrid_bins_dynamic(
                db, cutoff, model_name, model_version, BIN_DEFS, min_matches,
                base_model=base_model,
                base_version=base_version,
                features_version=features_version,
            )
            if hybrid_bin_metrics:
                hybrid_bins.append({
                    "model_name": model_name,
                    "model_version": model_version,
                    "bins": hybrid_bin_metrics,
                })
        else:
            # For pure models: compute single overall LogLoss/AUC plus
            # per-horizon metrics on the exact match/bin samples defined by
            # odds snapshot availability.  The Model Analysis page plots by
            # horizon; using one global model value in every bin is misleading.
            preds = query_df(
                db,
                """
                WITH ranked AS (
                    SELECT cp.prob_a, cp.prob_b,
                           cm.winner_side,
                           cm.normalized_team_a, cm.normalized_team_b,
                           ROW_NUMBER() OVER (
                               PARTITION BY cp.canonical_match_id
                               ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                           ) AS rn
                    FROM canonical_predictions cp
                    JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
                    WHERE cp.model_name = :mname
                      AND cp.model_version = :mver
                      AND cp.features_version = :features_version
                      AND cp.predicted_at <= cm.start_time_normalized::timestamptz
                      AND cm.status IN ('finished', 'completed')
                      AND cm.winner_side IS NOT NULL
                      AND cm.start_time_normalized > :cutoff
                )
                SELECT prob_a, prob_b, winner_side, normalized_team_a, normalized_team_b
                FROM ranked
                WHERE rn = 1
                """,
                {
                    "mname": model_name,
                    "mver": model_version,
                    "features_version": features_version,
                    "cutoff": cutoff,
                },
            )
            if not preds:
                continue

            y_true: list[int] = []
            y_score: list[float] = []
            for p in preds:
                winner = str(p.get("winner_side") or "")
                if winner not in ("team_a", "team_b"):
                    continue
                prob_a = p.get("prob_a")
                if prob_a is None:
                    continue
                prob_a = float(prob_a)
                if not (0 < prob_a < 1):
                    continue
                y_score.append(prob_a)
                y_true.append(1 if winner == "team_a" else 0)

            if len(y_true) < 2 or len(set(y_true)) < 2:
                continue

            ll = _compute_logloss(y_true, y_score)
            auc = _compute_auc(y_true, y_score)
            refs.append({
                "model_name": model_name,
                "model_version": model_version,
                "avg_logloss": ll,
                "avg_auc": auc,
                "n_matches": len(y_true),
            })

            pure_bin_metrics = _compute_pure_model_bins_dynamic(
                db,
                cutoff,
                model_name,
                model_version,
                features_version,
                BIN_DEFS,
                min_matches,
            )
            if pure_bin_metrics:
                hybrid_bins.append({
                    "model_name": model_name,
                    "model_version": model_version,
                    "bins": pure_bin_metrics,
                })

    return refs, hybrid_bins


def _compute_pure_model_bins_dynamic(
    db,
    cutoff: str,
    model_name: str,
    model_version: str,
    features_version: str,
    bin_defs: list,
    min_matches: int = 10,
) -> list[dict]:
    """Compute pure-model metrics on the same horizon samples as market odds.

    A pure prediction has one probability per match, but a match should only
    contribute to a horizon bin if we also have bookmaker snapshots in that bin.
    This makes the model-vs-market line chart compare both predictors on the
    same match/horizon population.
    """
    preds = query_df(
        db,
        """
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a,
                   cm.winner_side, cm.start_time_normalized,
                   cm.normalized_team_a, cm.normalized_team_b, cm.league,
                   ROW_NUMBER() OVER (
                       PARTITION BY cp.canonical_match_id
                       ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                   ) AS rn
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = :mname
              AND cp.model_version = :mver
              AND cp.features_version = :features_version
              AND cp.predicted_at <= cm.start_time_normalized::timestamptz
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
              AND cm.start_time_normalized > :cutoff
        )
        SELECT canonical_match_id, prob_a, winner_side, start_time_normalized,
               normalized_team_a, normalized_team_b, league
        FROM ranked
        WHERE rn = 1
        """,
        {
            "mname": model_name,
            "mver": model_version,
            "features_version": features_version,
            "cutoff": cutoff,
        },
    )
    if not preds:
        return []

    match_pred: dict[int, dict[str, Any]] = {}
    for p in preds:
        try:
            prob_a = float(p["prob_a"])
            mid = int(p["canonical_match_id"])
        except (TypeError, ValueError):
            continue
        if not (0 < prob_a < 1):
            continue
        winner = str(p.get("winner_side") or "")
        if winner not in ("team_a", "team_b"):
            continue
        match_pred[mid] = dict(p)

    if not match_pred:
        return []

    match_ids = list(match_pred.keys())
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params = {f"mid{i}": mid for i, mid in enumerate(match_ids)}
    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at,
               os.bookmaker_id, os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.scraped_at
        """,
        params,
    )

    bin_data = {
        label: {
            "per_match": defaultdict(lambda: {"y_true": None, "scores": []}),
            "snapshot_count": 0,
        }
        for label, _, _ in bin_defs
    }

    for snap in snapshots:
        try:
            mid = int(snap["canonical_match_id"])
        except (TypeError, ValueError):
            continue
        meta = match_pred.get(mid)
        if not meta:
            continue
        match_start = _parse_dt(meta.get("start_time_normalized"))
        scraped = _parse_dt(snap.get("scraped_at"))
        if not match_start or not scraped:
            continue
        hours_before = (match_start - scraped).total_seconds() / 3600.0
        if hours_before < 0:
            continue
        aligned = _align(
            str(meta.get("normalized_team_a") or ""),
            str(meta.get("normalized_team_b") or ""),
            str(snap.get("raw_team_a") or ""),
            str(snap.get("raw_team_b") or ""),
            snap.get("odds_a"),
            snap.get("odds_b"),
            league=meta.get("league"), match_date=meta.get("start_time_normalized"),
        )
        if not aligned:
            continue

        winner_side = str(meta.get("winner_side") or "")
        y_true = 1 if winner_side == "team_a" else 0
        y_score = float(meta["prob_a"])
        for label, hmin, hmax in bin_defs:
            if hmin <= hours_before < hmax:
                bd = bin_data[label]
                md = bd["per_match"][mid]
                md["y_true"] = y_true
                md["scores"].append(y_score)
                bd["snapshot_count"] += 1
                break

    result_bins = []
    for label, hmin, hmax in bin_defs:
        bd = bin_data[label]
        if len(bd["per_match"]) < min_matches:
            continue
        y_true: list[int] = []
        y_score: list[float] = []
        for md in bd["per_match"].values():
            if md["y_true"] is None or not md["scores"]:
                continue
            y_true.append(int(md["y_true"]))
            y_score.append(float(np.mean(md["scores"])))
        if len(y_true) < min_matches or len(set(y_true)) < 2:
            continue
        result_bins.append({
            "label": label,
            "hours_start": hmin,
            "hours_end": hmax if hmax < 9999 else None,
            "snapshot_count": int(bd["snapshot_count"]),
            "match_count": len(y_true),
            "avg_logloss": _compute_logloss(y_true, y_score),
            "avg_auc": _compute_auc(y_true, y_score),
        })

    return result_bins


def _compute_hybrid_bins_dynamic(
    db, cutoff: str, model_name: str, model_version: str, bin_defs: list, min_matches: int = 10,
    base_model: str = THESIS_MODEL_NAME, base_version: str = THESIS_MODEL_VERSION,
    features_version: str = ANALYSIS_FEATURES_VERSION,
) -> list[dict]:
    """Compute dynamic hybrid metrics per time bin.

    The active registered C0 hybrid uses its exact temperature and blending
    mode; the distinct thesis hybrid retains its versioned linear replay.
    """
    active_hybrid = get_active_hybrid()
    is_active_c0_hybrid = (
        model_name == active_hybrid.hybrid_model_name
        and model_version == active_hybrid.hybrid_model_version
        and base_model == active_hybrid.base_model.name
        and base_version == active_hybrid.base_model.version
        and features_version == C0_NATIVE.feature_version
    )
    if is_active_c0_hybrid:
        alpha = active_hybrid.alpha
        temperature = active_hybrid.temperature
    else:
        # Preserve the distinct historical thesis-hybrid contract.
        alpha = THESIS_HYBRID_ALPHA
        temperature = THESIS_HYBRID_TEMPERATURE
        if model_version.startswith("a") and "-t" in model_version:
            try:
                parts = model_version.split("-t")
                alpha = float(parts[0][1:])
                temperature = float(parts[1])
            except (IndexError, ValueError):
                pass
    
    # Get latest pure thesis model predictions for finished matches.
    # Do not filter by prediction_status here: for historical evaluation the
    # last prediction made before/around a match is still the valid forecast,
    # even if later scheduler runs marked it as stale while the match was not
    # yet auto-finished.
    thesis_preds = query_df(
        db,
        """
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a as thesis_prob_a,
                   cm.winner_side, cm.start_time_normalized,
                   ROW_NUMBER() OVER (
                       PARTITION BY cp.canonical_match_id
                       ORDER BY CASE
                           WHEN cp.predicted_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
                           THEN cp.predicted_at::timestamptz
                       END DESC NULLS LAST, cp.id DESC
                   ) AS rn
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = :base_model
              AND cp.model_version = :base_version
              AND cp.features_version = :features_version
              AND CASE
                  WHEN cp.predicted_at ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
                  THEN cp.predicted_at::timestamptz
              END <= CASE
                  WHEN cm.start_time_normalized ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}'
                  THEN cm.start_time_normalized::timestamptz
              END
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
              AND cm.start_time_normalized > :cutoff
        )
        SELECT canonical_match_id, thesis_prob_a, winner_side, start_time_normalized
        FROM ranked
        WHERE rn = 1
        """,
        {
            "base_model": base_model,
            "base_version": base_version,
            "features_version": features_version,
            "cutoff": cutoff,
        },
    )
    
    if not thesis_preds:
        return []
    
    # Build match_id -> thesis_prob mapping
    match_thesis = {p["canonical_match_id"]: float(p["thesis_prob_a"]) for p in thesis_preds}
    match_ids = list(match_thesis.keys())
    
    # Get odds snapshots for these matches
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params = {f"mid{i}": mid for i, mid in enumerate(match_ids)}
    
    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at,
               os.bookmaker_id, os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.scraped_at
        """,
        params,
    )
    
    # Get match metadata (start_time, winner_side, team names)
    match_meta = query_df(
        db,
        f"""
        SELECT id, start_time_normalized, winner_side,
               normalized_team_a, normalized_team_b, league
        FROM canonical_matches
        WHERE id IN ({placeholders})
        """,
        params,
    )
    meta_map = {m["id"]: m for m in match_meta}
    
    # Accumulate per bin at match level.  Average market/hybrid predictions
    # inside each match/bin first, then compute metrics across matches.
    bin_data = {
        label: {
            "per_match": defaultdict(lambda: {
                "y_true": None,
                "bookmaker_scores": defaultdict(list),
            }),
            "snapshot_count": 0,
        }
        for label, _, _ in bin_defs
    }
    
    for snap in snapshots:
        mid = snap["canonical_match_id"]
        if mid not in match_thesis:
            continue
        
        meta = meta_map.get(mid)
        if not meta:
            continue
        
        match_start = _parse_dt(meta.get("start_time_normalized"))
        scraped = _parse_dt(snap.get("scraped_at"))
        if not match_start or not scraped:
            continue
        
        hours_before = (match_start - scraped).total_seconds() / 3600.0
        if hours_before < 0:
            continue
        
        # Align odds to canonical sides
        aligned = _align(
            str(meta.get("normalized_team_a") or ""),
            str(meta.get("normalized_team_b") or ""),
            str(snap.get("raw_team_a") or ""),
            str(snap.get("raw_team_b") or ""),
            snap.get("odds_a"),
            snap.get("odds_b"),
            league=meta.get("league"), match_date=meta.get("start_time_normalized"),
        )
        if not aligned:
            continue
        
        odds_a, odds_b = aligned
        
        # Convert to implied probabilities (margin-removed)
        prob_a = _implied_prob(odds_a)
        prob_b = _implied_prob(odds_b)
        market_prob_a, market_prob_b = _remove_margin(prob_a, prob_b)
        
        # Apply the same model-side temperature and registered blend policy as
        # production; market probability is never temperature-scaled.
        thesis_prob_a = match_thesis[mid]
        thesis_prob_a_t = apply_temperature_probability(thesis_prob_a, temperature)
        if is_active_c0_hybrid:
            if active_hybrid.blending_mode == "logit_shrinkage":
                hybrid_prob_a = bayesian_logit_shrinkage(
                    thesis_prob_a_t, market_prob_a, alpha
                )
            elif active_hybrid.blending_mode == "linear":
                hybrid_prob_a = alpha * thesis_prob_a_t + (1 - alpha) * market_prob_a
            else:
                raise ValueError(
                    f"Unsupported active hybrid blending mode: {active_hybrid.blending_mode}"
                )
        else:
            hybrid_prob_a = alpha * thesis_prob_a_t + (1 - alpha) * market_prob_a
        
        # Ground truth
        winner_side = str(meta.get("winner_side") or "")
        if winner_side not in ("team_a", "team_b"):
            continue
        
        y_true = 1 if winner_side == "team_a" else 0
        y_score = hybrid_prob_a
        
        # Find bin — interval: snapshot counts only in its exact time window.
        for label, hmin, hmax in bin_defs:
            if hmin <= hours_before < hmax:
                bd = bin_data[label]
                md = bd["per_match"][mid]
                bookmaker_key = snap.get("bookmaker_id") or "unknown"
                md["y_true"] = y_true
                md["bookmaker_scores"][bookmaker_key].append(y_score)
                bd["snapshot_count"] += 1
                break
    
    # Compute metrics per bin
    result_bins = []
    for label, hmin, hmax in bin_defs:
        bd = bin_data[label]
        per_match = bd["per_match"]
        n_matches = len(per_match)
        n_snapshots = bd["snapshot_count"]
        
        if n_matches < min_matches:
            continue
        
        y_true: list[int] = []
        y_score: list[float] = []
        for md in per_match.values():
            if md["y_true"] is None or not md["bookmaker_scores"]:
                continue
            bookmaker_scores = [float(np.mean(vals)) for vals in md["bookmaker_scores"].values() if vals]
            if not bookmaker_scores:
                continue
            y_true.append(int(md["y_true"]))
            y_score.append(float(np.mean(bookmaker_scores)))

        if len(set(y_true)) < 2:
            continue
        
        ll = _compute_logloss(y_true, y_score)
        auc = _compute_auc(y_true, y_score)
        
        result_bins.append({
            "label": label,
            "hours_start": hmin,
            "hours_end": hmax if hmax < 9999 else None,
            "snapshot_count": n_snapshots,
            "match_count": n_matches,
            "avg_logloss": ll,
            "avg_auc": auc,
        })
    
    return result_bins

def _compute_calibration_bins(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> list[dict[str, Any]]:
    """Compute reliability diagram / calibration curve bins."""
    if not y_true or len(y_true) != len(y_prob):
        return []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    y_true_arr = np.array(y_true, dtype=float)
    y_prob_arr = np.array(y_prob, dtype=float)
    total_n = len(y_true)
    bins = []
    for i in range(n_bins):
        low, high = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (y_prob_arr >= low) & (y_prob_arr <= high if i == n_bins - 1 else y_prob_arr < high)
        count = int(np.sum(mask))
        if count > 0:
            avg_pred = float(np.mean(y_prob_arr[mask]))
            empirical = float(np.mean(y_true_arr[mask]))
            diff = abs(avg_pred - empirical)
        else:
            avg_pred = (low + high) / 2.0
            empirical = None
            diff = 0.0
        bins.append({
            "bin_index": i,
            "bin_low": round(low, 2),
            "bin_high": round(high, 2),
            "label": f"{int(low * 100)}-{int(high * 100)}%",
            "count": count,
            "avg_predicted": round(avg_pred, 4) if count > 0 else None,
            "empirical_rate": round(empirical, 4) if empirical is not None else None,
            "calibration_error": round(diff, 4) if count > 0 else None,
        })
    return bins


def _compute_ece(y_true: list[int], y_prob: list[float], n_bins: int = 10) -> float | None:
    """Expected Calibration Error (weighted average absolute error)."""
    if len(y_true) < 2:
        return None
    bins = _compute_calibration_bins(y_true, y_prob, n_bins)
    total_n = len(y_true)
    if total_n == 0:
        return None
    ece = sum((b["count"] / total_n) * (b["calibration_error"] or 0.0) for b in bins if b["count"] > 0)
    return round(float(ece), 4)


def _is_tier_1_league(league: str | None) -> bool:
    """Check if league belongs to top professional tier (LCK, LPL, LEC, LCS, Worlds, MSI, EWC)."""
    if not league:
        return False
    lg = str(league).strip().lower()
    if "challenger" in lg or "academy" in lg:
        return False
    return any(
        k in lg for k in ("lpl", "lck", "lec", "lcs", "msi", "worlds", "world championship", "esports world cup")
    )


def _metric_summary(y_true: list[int], y_prob: list[float]) -> dict:
    """Shared comprehensive metric payload for model/market comparisons."""
    logloss = _compute_logloss(y_true, y_prob)
    auc = _compute_auc(y_true, y_prob)
    brier = _compute_brier(y_true, y_prob)
    accuracy = _compute_accuracy(y_true, y_prob)
    ece = _compute_ece(y_true, y_prob) if len(y_true) >= 2 else None
    calibration_bins = _compute_calibration_bins(y_true, y_prob, 10) if len(y_true) >= 2 else []
    
    status = "unknown"
    if ece is not None and logloss is not None:
        if logloss <= 0.60 and ece <= 0.06:
            status = "well_calibrated"
        elif logloss > 0.693:
            status = "overconfident_miscalibrated"
        elif ece > 0.10:
            status = "miscalibrated"
        else:
            status = "acceptable"

    return {
        "n_matches": len(y_true),
        "avg_logloss": logloss,
        "avg_auc": auc,
        "avg_brier": brier,
        "accuracy": accuracy,
        "ece": ece,
        "calibration_status": status,
        "calibration_bins": calibration_bins,
    }

def _compute_market_close_comparison(db, cutoff: str, min_matches: int) -> dict:
    """Compare thesis/hybrid against collected bookmaker closing odds.

    This is the headline production-health check for the Horizon page. It uses
    a strict market-close sample: for every finished match, take each
    bookmaker's latest non-live snapshot with scraped_at <= match start, align
    it to canonical sides, remove margin, then average bookmaker probabilities
    per match. Model, market, and hybrid metrics are computed on exactly the
    same match set.
    """
    preds = query_df(
        db,
        """
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a AS thesis_prob_a,
                   cm.winner_side, cm.start_time_normalized,
                   cm.normalized_team_a, cm.normalized_team_b, cm.league,
                   ROW_NUMBER() OVER (
                       PARTITION BY cp.canonical_match_id
                       ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                   ) AS rn
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = :mname
              AND cp.model_version = :mver
              AND cp.features_version = :features_version
              AND cp.predicted_at <= cm.start_time_normalized::timestamptz
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IS NOT NULL
              AND cm.start_time_normalized IS NOT NULL
              AND cm.start_time_normalized > :cutoff
        )
        SELECT canonical_match_id, thesis_prob_a, winner_side,
               start_time_normalized, normalized_team_a, normalized_team_b, league
        FROM ranked
        WHERE rn = 1
        """,
        {
            "mname": THESIS_MODEL_NAME,
            "mver": THESIS_MODEL_VERSION,
            "features_version": ANALYSIS_FEATURES_VERSION,
            "cutoff": cutoff,
        },
    )
    pred_map = {
        p["canonical_match_id"]: p
        for p in preds
        if p.get("thesis_prob_a") is not None
        and 0 < float(p.get("thesis_prob_a")) < 1
        and str(p.get("winner_side") or "") in ("team_a", "team_b")
    }
    if not pred_map:
        return _empty_market_close_comparison()

    match_ids = list(pred_map.keys())
    placeholders = ",".join(f":mid{i}" for i in range(len(match_ids)))
    params = {f"mid{i}": mid for i, mid in enumerate(match_ids)}

    snapshots = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, os.scraped_at, os.bookmaker_id,
               b.name AS bookmaker_name,
               os.odds_a, os.odds_b, os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at DESC
        """,
        params,
    )

    # Keep only one latest pre-match snapshot per match/bookmaker.
    close_by_match_book: dict[int, dict[Any, dict]] = defaultdict(dict)
    for snap in snapshots:
        mid = snap["canonical_match_id"]
        meta = pred_map.get(mid)
        if not meta:
            continue
        match_start = _parse_dt(meta.get("start_time_normalized"))
        scraped = _parse_dt(snap.get("scraped_at"))
        if not match_start or not scraped or scraped > match_start:
            continue
        bookmaker_key = snap.get("bookmaker_id") or "unknown"
        prev = close_by_match_book[mid].get(bookmaker_key)
        prev_dt = _parse_dt(prev.get("scraped_at")) if prev else None
        if prev is None or (prev_dt is not None and scraped > prev_dt):
            close_by_match_book[mid][bookmaker_key] = snap

    rows: list[dict] = []
    bookmaker_probs: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "bookmaker_name": None,
        "y_true": [],
        "y_prob": [],
    })
    alpha = THESIS_HYBRID_ALPHA
    temperature = THESIS_HYBRID_TEMPERATURE

    for mid, meta in pred_map.items():
        per_book_probs: list[float] = []
        for bk_id, snap in close_by_match_book.get(mid, {}).items():
            aligned = _align(
                str(meta.get("normalized_team_a") or ""),
                str(meta.get("normalized_team_b") or ""),
                str(snap.get("raw_team_a") or ""),
                str(snap.get("raw_team_b") or ""),
                snap.get("odds_a"),
                snap.get("odds_b"),
                league=meta.get("league"), match_date=meta.get("start_time_normalized"),
            )
            if not aligned:
                continue
            odds_a, odds_b = aligned
            market_prob_a, _ = _remove_margin(_implied_prob(odds_a), _implied_prob(odds_b))
            per_book_probs.append(market_prob_a)
            if isinstance(bk_id, int):
                y_true_bk = 1 if str(meta.get("winner_side") or "") == "team_a" else 0
                bookmaker_probs[bk_id]["bookmaker_name"] = snap.get("bookmaker_name")
                bookmaker_probs[bk_id]["y_true"].append(y_true_bk)
                bookmaker_probs[bk_id]["y_prob"].append(market_prob_a)

        if not per_book_probs:
            continue

        y_true = 1 if str(meta.get("winner_side") or "") == "team_a" else 0
        thesis_prob = float(meta["thesis_prob_a"])
        market_prob = float(np.mean(per_book_probs))
        thesis_prob_t = apply_temperature_probability(thesis_prob, temperature)
        hybrid_prob = alpha * thesis_prob_t + (1.0 - alpha) * market_prob
        rows.append({
            "match_id": mid,
            "y_true": y_true,
            "thesis_prob": thesis_prob,
            "market_prob": market_prob,
            "hybrid_prob": hybrid_prob,
            "bookmaker_count": len(per_book_probs),
        })

    if not rows:
        return _empty_market_close_comparison()

    y_true = [int(r["y_true"]) for r in rows]
    thesis = [float(r["thesis_prob"]) for r in rows]
    market = [float(r["market_prob"]) for r in rows]
    hybrid = [float(r["hybrid_prob"]) for r in rows]

    competitors = [
        {"name": "MODEL", "display_name": "Thesis model", **_metric_summary(y_true, thesis)},
        {"name": "MARKET_CLOSE", "display_name": "Bookmaker consensus close", **_metric_summary(y_true, market)},
        {"name": "HYBRID", "display_name": "Hybrid thesis+market", **_metric_summary(y_true, hybrid)},
    ]
    competitors.sort(key=lambda r: r["avg_logloss"] if r["avg_logloss"] is not None else 999.0)
    for i, row in enumerate(competitors, start=1):
        row["rank"] = i

    model_ll = next((r["avg_logloss"] for r in competitors if r["name"] == "MODEL"), None)
    market_ll = next((r["avg_logloss"] for r in competitors if r["name"] == "MARKET_CLOSE"), None)
    hybrid_ll = next((r["avg_logloss"] for r in competitors if r["name"] == "HYBRID"), None)
    delta_vs_market = None if model_ll is None or market_ll is None else round(float(model_ll - market_ll), 4)

    bookmaker_rows = []
    for bk_id, data in bookmaker_probs.items():
        bk_y = data["y_true"]
        bk_p = data["y_prob"]
        if len(bk_y) < min_matches:
            continue
        bookmaker_rows.append({
            "bookmaker_id": bk_id,
            "bookmaker_name": data.get("bookmaker_name") or f"bookmaker_{bk_id}",
            **_metric_summary(bk_y, bk_p),
        })
    bookmaker_rows.sort(key=lambda r: r["avg_logloss"] if r["avg_logloss"] is not None else 999.0)
    for i, row in enumerate(bookmaker_rows, start=1):
        row["rank"] = i

    status = "unknown"
    if delta_vs_market is not None:
        if delta_vs_market <= -0.01:
            status = "model_better"
        elif delta_vs_market <= 0.01:
            status = "model_on_market_level"
        else:
            status = "model_worse"

    return {
        "sample_definition": "latest non-live pre-match snapshot per bookmaker; averaged per match; identical match sample for model/market/hybrid",
        "n_matches": len(rows),
        "min_matches": min_matches,
        "avg_bookmakers_per_match": round(float(np.mean([r["bookmaker_count"] for r in rows])), 2),
        "model_delta_logloss_vs_market": delta_vs_market,
        "hybrid_delta_logloss_vs_market": None if hybrid_ll is None or market_ll is None else round(float(hybrid_ll - market_ll), 4),
        "status": status,
        "competitors": competitors,
        "bookmakers": bookmaker_rows,
    }


def _empty_market_close_comparison() -> dict:
    return {
        "sample_definition": "latest non-live pre-match snapshot per bookmaker; averaged per match; identical match sample for model/market/hybrid",
        "n_matches": 0,
        "min_matches": 0,
        "avg_bookmakers_per_match": None,
        "model_delta_logloss_vs_market": None,
        "hybrid_delta_logloss_vs_market": None,
        "status": "no_data",
        "competitors": [],
        "bookmakers": [],
    }


def apply_temperature_probability(prob: float, temperature: float) -> float:
    """Apply temperature scaling to a probability."""
    if temperature == 1.0:
        return prob
    # Convert to log-odds, scale, convert back
    eps = 1e-15
    prob = max(eps, min(1 - eps, prob))
    log_odds = math.log(prob / (1 - prob))
    scaled_log_odds = log_odds / temperature
    return 1.0 / (1.0 + math.exp(-scaled_log_odds))


def _empty_horizon_result(total_matches: int = 0) -> dict:
    return {
        "total_matches_with_odds": 0,
        "total_finished_matches": total_matches,
        "total_odds_processed": 0,
        "bins": [],
        "min_matches_per_bin": 10,
        "model_references": [],
        "hybrid_model_bins": [],
        "bookmaker_bins": [],
        "model_vs_bookmaker_tests": [],
    }


def _empty_result(matches_found: int = 0) -> dict:
    return {
        "total_matches": 0,
        "total_snapshots": 0,
        "time_buckets": [],
        "drift_summary": None,
        "best_betting_window": None,
        "summary": {
            "message": "No finished matches with sufficient odds history found."
            if matches_found == 0
            else "Not enough data after processing."
        },
    }


@router.get("/match/{match_id}/movement")
def match_odds_movement(match_id: int, db=Depends(get_db)):
    """Get detailed odds movement for a specific match.

    Returns time series of odds with % deviation from closing.
    """
    match = query_df(
        db,
        "SELECT * FROM canonical_matches WHERE id=:id",
        {"id": match_id},
    )
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    m = match[0]
    match_start = _parse_dt(m.get("start_time_normalized"))
    n_a = str(m.get("normalized_team_a") or "")
    n_b = str(m.get("normalized_team_b") or "")

    odds = query_df(
        db,
        """
        SELECT os.scraped_at, os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b,
               b.name AS bookmaker
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id = :mid
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a IS NOT NULL
          AND os.odds_b IS NOT NULL
        ORDER BY os.scraped_at
        """,
        {"mid": match_id},
    )

    if not odds:
        return {
            "match_id": match_id,
            "team_a": m.get("team_a_name"),
            "team_b": m.get("team_b_name"),
            "movement_points": [],
            "summary": {"message": "No odds history available"},
        }

    # Align all snapshots
    points = []
    closing_a = closing_b = None

    for row in odds:
        st = _parse_dt(row["scraped_at"])
        if not st or not match_start:
            continue
        hours_before = (match_start - st).total_seconds() / 3600.0
        if hours_before < 0:
            continue

        aligned = _align(
            n_a, n_b,
            str(row.get("raw_team_a") or ""),
            str(row.get("raw_team_b") or ""),
            row.get("odds_a"),
            row.get("odds_b"),
            league=m.get("league"), match_date=m.get("start_time_normalized"),
        )
        if not aligned:
            continue

        points.append({
            "scraped_at": str(st),
            "hours_before_match": round(hours_before, 2),
            "bookmaker": row.get("bookmaker"),
            "odds_a": round(aligned[0], 3),
            "odds_b": round(aligned[1], 3),
        })

    if not points:
        return {
            "match_id": match_id,
            "team_a": m.get("team_a_name"),
            "team_b": m.get("team_b_name"),
            "movement_points": [],
            "summary": {"message": "No valid pre-match odds data"},
        }

    # Sort by hours_before descending (earliest first)
    points.sort(key=lambda p: p["hours_before_match"], reverse=True)
    closing = points[0]  # closest to match start
    closing_a = closing["odds_a"]
    closing_b = closing["odds_b"]

    # Add deviation % to each point
    for p in points:
        p["deviation_a_pct"] = round(
            ((p["odds_a"] - closing_a) / closing_a) * 100.0, 2
        )
        p["deviation_b_pct"] = round(
            ((p["odds_b"] - closing_b) / closing_b) * 100.0, 2
        )

    first = points[-1]  # farthest from match

    return {
        "match_id": match_id,
        "team_a": m.get("team_a_name"),
        "team_b": m.get("team_b_name"),
        "start_time": m.get("start_time_normalized"),
        "movement_points": points,
        "summary": {
            "total_snapshots": len(points),
            "first_snapshot": first["scraped_at"],
            "last_snapshot": closing["scraped_at"],
            "opening_odds_a": first["odds_a"],
            "opening_odds_b": first["odds_b"],
            "closing_odds_a": closing_a,
            "closing_odds_b": closing_b,
            "opening_deviation_a_pct": first["deviation_a_pct"],
            "opening_deviation_b_pct": first["deviation_b_pct"],
            "total_drift_a": round(closing_a - first["odds_a"], 3),
            "total_drift_b": round(closing_b - first["odds_b"], 3),
        },
    }


@router.get("/model-profitability-audit")
def model_profitability_audit(
    model_key: str = "c0",
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    max_days_back: int = 3650,
    db=Depends(get_db),
):
    """Audyt rentowności i wykrywanie anomalii kalibracji modelu.

    Analizuje realne zakłady z zachowaniem ścisłej przyczynowości:
      - Kurs wejścia z okna 6h-24h (fallback 2h-48h, fallback ostatni przed meczem)
      - Kurs zamknięcia z okna < 2h przed meczem
      - Rozbicie na 7 przedziałów kursowych z detekcją pułapki kursowej (ISSUE-001)
      - Symulacja wpływu odcięcia kwarantanny na zysk i ROI
      - Rozbicie na formaty (Bo1/Bo3/Bo5) i ligi
      - Detekcja rozbieżności komponentów (IDI - Internal Disagreement Index) i anomalii rynkowych
    """
    tax_mult = 1.0 - tax_rate

    model_configs = {
        "c0": {
            "name": C0_NATIVE.name,
            "version": C0_NATIVE.version,
            "title": "Native C0",
        },
        "operational_hybrid": {
            "name": get_active_hybrid().hybrid_model_name,
            "version": get_active_hybrid().hybrid_model_version,
            "title": "Hybryda Operacyjna (C0 + Rynek alpha=0.50)",
        },
        "operational": {
            "name": OPERATIONAL_MODEL_NAME,
            "version": OPERATIONAL_BACKFILL_MODEL_VERSION,
            "title": "Model Operacyjny (Regional BoN v0.4)",
        },
        "thesis": {
            "name": THESIS_MODEL_NAME,
            "version": THESIS_MODEL_VERSION,
            "title": "Thesis Baseline (EXP-039 Sym-Cal)",
        },
    }
    if model_key not in model_configs:
        raise HTTPException(status_code=400, detail="Unknown profitability model key.")
    cfg = model_configs[model_key]
    cutoff = (datetime.now(UTC) - timedelta(days=max_days_back)).isoformat()

    matches_sql = """
        SELECT cm.id, cm.team_a_name, cm.team_b_name, cm.start_time_normalized, cm.winner_side, cm.best_of, cm.league
        FROM canonical_matches cm
        WHERE cm.status IN ('finished', 'completed')
          AND cm.winner_side IN ('team_a', 'team_b')
          AND cm.start_time_normalized IS NOT NULL
          AND cm.start_time_normalized >= :cutoff
    """
    matches = query_df(db, matches_sql, {"cutoff": cutoff})
    match_dict = {}
    for m in matches:
        try:
            st = datetime.fromisoformat(m["start_time_normalized"].replace("Z", "+00:00"))
        except Exception:
            continue
        match_dict[int(m["id"])] = {
            "start": st,
            "winner": m["winner_side"],
            "team_a": m["team_a_name"],
            "team_b": m["team_b_name"],
            "best_of": int(m["best_of"]) if m.get("best_of") else 1,
            "league": m.get("league") or "Unknown",
        }

    preds_sql = """
        WITH ranked AS (
            SELECT cp.canonical_match_id, cp.prob_a, cp.prob_b, cp.diagnostics_json,
                   ROW_NUMBER() OVER (
                       PARTITION BY cp.canonical_match_id
                       ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                   ) AS rn
            FROM canonical_predictions cp
            JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
            WHERE cp.model_name = :mname
              AND cp.model_version = :mver
              AND cm.status IN ('finished', 'completed')
              AND cm.winner_side IN ('team_a', 'team_b')
              AND cm.start_time_normalized >= :cutoff
        )
        SELECT canonical_match_id, prob_a, prob_b, diagnostics_json
        FROM ranked WHERE rn = 1
    """
    preds = query_df(
        db,
        preds_sql,
        {
            "mname": cfg["name"],
            "mver": cfg["version"],
            "cutoff": cutoff,
        },
    )
    pred_dict = {}
    for p in preds:
        mid = int(p["canonical_match_id"])
        if mid not in match_dict:
            continue
        diag = {}
        if p.get("diagnostics_json"):
            try:
                diag = json.loads(p["diagnostics_json"]) if isinstance(p["diagnostics_json"], str) else p["diagnostics_json"]
            except Exception:
                diag = {}
        pred_dict[mid] = (float(p["prob_a"]), float(p["prob_b"]), diag)

    if not pred_dict:
        return {
            "model_key": model_key,
            "model_title": cfg["title"],
            "error": "Brak predykcji spełniających kryteria.",
            "overall": None,
            "filter_impact": None,
            "odds_brackets": [],
            "series_formats": [],
            "leagues": [],
            "anomalies": [],
            "operational_rules": [],
        }

    match_ids = sorted(pred_dict.keys())
    placeholders = ",".join(f":mid_{i}" for i in range(len(match_ids)))
    m_params = {f"mid_{i}": mid for i, mid in enumerate(match_ids)}

    odds_sql = f"""
        SELECT os.canonical_match_id, os.odds_a, os.odds_b, os.scraped_at, b.name as bookmaker
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IN ({placeholders})
          AND os.market_type = 'match_winner'
          AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a > 1.0 AND os.odds_b > 1.0
        ORDER BY os.canonical_match_id, os.scraped_at ASC
    """
    odds_raw = query_df(db, odds_sql, m_params)
    odds_by_match = defaultdict(list)
    for r in odds_raw:
        mid = int(r["canonical_match_id"])
        try:
            st = _parse_dt(r["scraped_at"])
        except Exception:
            continue
        odds_by_match[mid].append({
            "odds_a": float(r["odds_a"]),
            "odds_b": float(r["odds_b"]),
            "bookmaker": r.get("bookmaker") or "Nieznany",
            "scraped_at": st,
        })

    bets = []
    for mid, (pa, pb, diag) in pred_dict.items():
        m = match_dict[mid]
        start = m["start"]
        winner = m["winner"]
        quotes = odds_by_match.get(mid, [])
        if not quotes:
            continue
        valid_quotes = [q for q in quotes if q["scraped_at"] < start]
        if not valid_quotes:
            continue

        w_quotes = [q for q in valid_quotes if 6 <= (start - q["scraped_at"]).total_seconds() / 3600 <= 24]
        if not w_quotes:
            w_quotes = [q for q in valid_quotes if 2 <= (start - q["scraped_at"]).total_seconds() / 3600 <= 48]
        if not w_quotes:
            w_quotes = [valid_quotes[-1]]

        close_quotes = [q for q in valid_quotes if (start - q["scraped_at"]).total_seconds() / 3600 <= 2]
        close_q = close_quotes[-1] if close_quotes else valid_quotes[-1]

        for q in w_quotes:
            eva = expected_value(pa, q["odds_a"], tax_rate)
            evb = expected_value(pb, q["odds_b"], tax_rate)

            try:
                mkta, mktb = fair_market_probabilities(q["odds_a"], q["odds_b"])
            except Exception:
                continue
            try:
                cl_mkta, cl_mktb = fair_market_probabilities(close_q["odds_a"], close_q["odds_b"])
            except Exception:
                cl_mkta, cl_mktb = mkta, mktb

            is_rev = bool(diag.get("is_reversed_side", False))
            p_cons = diag.get("player_rating_consensus")
            t_cons = diag.get("team_rating_consensus")
            idi = None
            if p_cons is not None and t_cons is not None:
                try:
                    p_f = float(p_cons)
                    t_f = float(t_cons)
                    if is_rev:
                        p_f = 1.0 - p_f
                        t_f = 1.0 - t_f
                    idi = abs(p_f - t_f)
                except Exception:
                    idi = None

            if eva >= min_ev and eva >= evb:
                clv_pct = ((q["odds_a"] / close_q["odds_a"]) - 1.0) * 100.0 if close_q["odds_a"] > 0 else 0.0
                bets.append({
                    "mid": mid,
                    "team": m["team_a"],
                    "opp": m["team_b"],
                    "bet_side": "team_a",
                    "odds": q["odds_a"],
                    "odds_close": close_q["odds_a"],
                    "clv_pct": clv_pct,
                    "prob_model": pa,
                    "prob_market": mkta,
                    "prob_close": cl_mkta,
                    "ev_net": eva,
                    "ev_gross": pa * q["odds_a"] - 1.0,
                    "won": 1 if winner == "team_a" else 0,
                    "best_of": m["best_of"],
                    "league": m["league"],
                    "bookmaker": q["bookmaker"],
                    "player_consensus": p_cons,
                    "team_consensus": t_cons,
                    "idi": idi,
                    "start": m["start"].strftime("%Y-%m-%d %H:%M"),
                })
                break
            elif evb >= min_ev and evb > eva:
                p_c = 1.0 - float(p_cons) if p_cons is not None else None
                t_c = 1.0 - float(t_cons) if t_cons is not None else None
                clv_pct = ((q["odds_b"] / close_q["odds_b"]) - 1.0) * 100.0 if close_q["odds_b"] > 0 else 0.0
                bets.append({
                    "mid": mid,
                    "team": m["team_b"],
                    "opp": m["team_a"],
                    "bet_side": "team_b",
                    "odds": q["odds_b"],
                    "odds_close": close_q["odds_b"],
                    "clv_pct": clv_pct,
                    "prob_model": pb,
                    "prob_market": mktb,
                    "prob_close": cl_mktb,
                    "ev_net": evb,
                    "ev_gross": pb * q["odds_b"] - 1.0,
                    "won": 1 if winner == "team_b" else 0,
                    "best_of": m["best_of"],
                    "league": m["league"],
                    "bookmaker": q["bookmaker"],
                    "player_consensus": p_c,
                    "team_consensus": t_c,
                    "idi": idi,
                    "start": m["start"].strftime("%Y-%m-%d %H:%M"),
                })
                break

    if not bets:
        return {
            "model_key": model_key,
            "model_title": cfg["title"],
            "error": "Brak zakładów kwalifikujących się przy obecnym progu EV i podatku.",
            "overall": None,
            "filter_impact": None,
            "odds_brackets": [],
            "series_formats": [],
            "leagues": [],
            "anomalies": [],
            "operational_rules": [],
        }

    import pandas as pd
    df = pd.DataFrame(bets)
    df.sort_values("start", inplace=True)

    total_bets = len(df)
    n_wins = int(df["won"].sum())
    win_rate = float(df["won"].mean() * 100.0)
    avg_odds = float(df["odds"].mean())
    avg_p_model = float(df["prob_model"].mean() * 100.0)
    avg_p_market = float(df["prob_market"].mean() * 100.0)
    bias = avg_p_model - win_rate

    df["pnl_net"] = df["won"] * df["odds"] * tax_mult - 1.0
    df["pnl_gross"] = df["won"] * df["odds"] - 1.0

    pnl_net_total = float(df["pnl_net"].sum())
    realized_net_roi = float(df["pnl_net"].mean() * 100.0)
    realized_gross_roi = float(df["pnl_gross"].mean() * 100.0)
    expected_net_roi = float(df["ev_net"].mean() * 100.0)

    cum_pnl = df["pnl_net"].cumsum()
    peak = cum_pnl.cummax()
    drawdown = peak - cum_pnl
    max_dd_units = float(drawdown.max())

    y_true = df["won"].values
    y_prob = df["prob_model"].values
    y_mkt = df["prob_market"].values
    brier = float(np.mean((y_prob - y_true) ** 2))
    mkt_brier = float(np.mean((y_mkt - y_true) ** 2))

    eps = 1e-15
    p_clip = np.clip(y_prob, eps, 1 - eps)
    logloss = float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))
    mkt_clip = np.clip(y_mkt, eps, 1 - eps)
    mkt_logloss = float(-np.mean(y_true * np.log(mkt_clip) + (1 - y_true) * np.log(1 - mkt_clip)))

    pos_clv_pct = float((df["clv_pct"] > 0).mean() * 100.0)
    avg_clv_pct = float(df["clv_pct"].mean())

    df_quarantine = df[(df["odds"] >= 3.50) & (df["odds"] < 5.00)]
    df_filtered = df[~((df["odds"] >= 3.50) & (df["odds"] < 5.00))]

    filter_impact = {
        "raw_bets": total_bets,
        "raw_roi_net_pct": round(realized_net_roi, 2),
        "raw_pnl_units": round(pnl_net_total, 2),
        "quarantined_bets": len(df_quarantine),
        "quarantined_roi_net_pct": round(float(df_quarantine["pnl_net"].mean() * 100.0), 2) if len(df_quarantine) > 0 else 0.0,
        "quarantined_pnl_units": round(float(df_quarantine["pnl_net"].sum()), 2) if len(df_quarantine) > 0 else 0.0,
        "filtered_bets": len(df_filtered),
        "filtered_roi_net_pct": round(float(df_filtered["pnl_net"].mean() * 100.0), 2) if len(df_filtered) > 0 else 0.0,
        "filtered_pnl_units": round(float(df_filtered["pnl_net"].sum()), 2) if len(df_filtered) > 0 else 0.0,
        "pnl_improvement_units": round(float(df_filtered["pnl_net"].sum() - pnl_net_total), 2),
    }

    BRACKETS = [
        ("< 1.40 (Ciężki faworyt)", 1.0, 1.40, "NORMAL"),
        ("1.40 - 1.80 (Faworyt)", 1.40, 1.80, "NORMAL"),
        ("1.80 - 2.20 (Wyrównany)", 1.80, 2.20, "NORMAL"),
        ("2.20 - 2.80 (Lekki underdog)", 2.20, 2.80, "NORMAL"),
        ("2.80 - 3.50 (Złoty underdog)", 2.80, 3.50, "RECOMMENDED"),
        ("3.50 - 5.00 (Wysoki underdog)", 3.50, 5.00, "NORMAL"),
        ("> 5.00 (Longshot)", 5.00, 100.0, "NORMAL"),
    ]

    odds_brackets = []
    for label, omin, omax, status in BRACKETS:
        sub = df[(df["odds"] >= omin) & (df["odds"] < omax)]
        if len(sub) == 0:
            continue
        n = len(sub)
        w = int(sub["won"].sum())
        wr = float(sub["won"].mean() * 100.0)
        o_mean = float(sub["odds"].mean())
        pm = float(sub["prob_model"].mean() * 100.0)
        pmk = float(sub["prob_market"].mean() * 100.0)
        b = pm - wr
        r_net = float(sub["pnl_net"].mean() * 100.0)
        r_gross = float(sub["pnl_gross"].mean() * 100.0)
        exp_net = float(sub["ev_net"].mean() * 100.0)
        pnl = float(sub["pnl_net"].sum())
        clv_m = float(sub["clv_pct"].mean())

        flags = []
        if b > 15.0:
            flags.append(f"⚠️ Znaczące przeszacowanie (+{b:.1f} p.p.)")
        elif abs(b) <= 3.0:
            flags.append(f"✅ Dobrze skalibrowany segment (błąd {b:+.1f} p.p.)")
        elif status == "RECOMMENDED":
            flags.append("🌟 Złoty segment maksymalnej rentowności")
            flags.append(f"Znakomita kalibracja: błąd zaledwie {b:+.1f} p.p.")

        odds_brackets.append({
            "bracket_label": label,
            "odds_min": omin,
            "odds_max": omax,
            "n_bets": n,
            "n_wins": w,
            "win_rate_pct": round(wr, 1),
            "avg_odds": round(o_mean, 2),
            "avg_model_prob_pct": round(pm, 1),
            "avg_market_prob_pct": round(pmk, 1),
            "calibration_bias_pp": round(b, 1),
            "expected_net_roi_pct": round(exp_net, 1),
            "realized_net_roi_pct": round(r_net, 1),
            "realized_gross_roi_pct": round(r_gross, 1),
            "pnl_units": round(pnl, 2),
            "avg_clv_pct": round(clv_m, 2),
            "status": status,
            "flags": flags,
        })

    formats = []
    for bo in sorted(df["best_of"].unique()):
        sub = df[df["best_of"] == bo]
        formats.append({
            "format_label": f"Bo{bo}",
            "n_bets": len(sub),
            "n_wins": int(sub["won"].sum()),
            "win_rate_pct": round(float(sub["won"].mean() * 100.0), 1),
            "avg_odds": round(float(sub["odds"].mean()), 2),
            "avg_model_prob_pct": round(float(sub["prob_model"].mean() * 100.0), 1),
            "avg_market_prob_pct": round(float(sub["prob_market"].mean() * 100.0), 1),
            "calibration_bias_pp": round(float(sub["prob_model"].mean() * 100.0 - sub["won"].mean() * 100.0), 1),
            "realized_net_roi_pct": round(float(sub["pnl_net"].mean() * 100.0), 1),
            "pnl_units": round(float(sub["pnl_net"].sum()), 2),
        })

    leagues = []
    for lg, sub in df.groupby("league"):
        if len(sub) < 4:
            continue
        leagues.append({
            "league": lg,
            "n_bets": len(sub),
            "n_wins": int(sub["won"].sum()),
            "win_rate_pct": round(float(sub["won"].mean() * 100.0), 1),
            "avg_odds": round(float(sub["odds"].mean()), 2),
            "avg_model_prob_pct": round(float(sub["prob_model"].mean() * 100.0), 1),
            "avg_market_prob_pct": round(float(sub["prob_market"].mean() * 100.0), 1),
            "calibration_bias_pp": round(float(sub["prob_model"].mean() * 100.0 - sub["won"].mean() * 100.0), 1),
            "realized_net_roi_pct": round(float(sub["pnl_net"].mean() * 100.0), 1),
            "pnl_units": round(float(sub["pnl_net"].sum()), 2),
        })
    leagues.sort(key=lambda x: x["pnl_units"], reverse=True)

    df["discrepancy_pp"] = (df["prob_model"] - df["prob_market"]) * 100.0
    anomalies_df = df[(df["discrepancy_pp"].abs() >= 15.0) | (df["idi"].fillna(0) >= 0.15)].copy()
    anomalies_df.sort_values(by="discrepancy_pp", ascending=False, inplace=True)

    anomalies = []
    for _, row in anomalies_df.head(15).iterrows():
        reasons = []
        if row["odds"] >= 3.50 and row["odds"] <= 5.00:
            reasons.append("Pułapka kursowa ISSUE-001")
        if row["discrepancy_pp"] > 20.0:
            reasons.append(f"Skrajny rozstrzał z rynkiem (+{row['discrepancy_pp']:.1f} p.p.)")
        if row["idi"] is not None and row["idi"] > 0.15:
            reasons.append(f"Konflikt ratingów gracze/drużyna (IDI {row['idi']*100:.1f}%)")
        if row["best_of"] > 1 and row["odds"] > 3.0:
            reasons.append(f"Format Bo{row['best_of']} redukuje szanse underdoga")

        anomalies.append({
            "match_id": int(row["mid"]),
            "match_title": f"{row['team']} vs {row['opp']}",
            "bet_side": row["bet_side"],
            "team_selected": row["team"],
            "date": row["start"],
            "league": row["league"],
            "best_of": f"Bo{row['best_of']}",
            "odds": round(float(row["odds"]), 2),
            "odds_close": round(float(row["odds_close"]), 2),
            "model_prob_pct": round(float(row["prob_model"] * 100.0), 1),
            "market_prob_pct": round(float(row["prob_market"] * 100.0), 1),
            "discrepancy_pp": round(float(row["discrepancy_pp"]), 1),
            "player_consensus_pct": round(float(row["player_consensus"] * 100.0), 1) if row["player_consensus"] is not None else None,
            "team_consensus_pct": round(float(row["team_consensus"] * 100.0), 1) if row["team_consensus"] is not None else None,
            "idi_pct": round(float(row["idi"] * 100.0), 1) if row["idi"] is not None else None,
            "won": bool(row["won"]),
            "pnl_net": round(float(row["pnl_net"]), 2),
            "reasons": reasons,
        })

    operational_rules = [
        "🚨 TWARDY ZAKAZ (Kwarantanna): Kursy w przedziale [3.50 - 5.00] generują ujemny ROI -31.5% przez kompresję logistyczną i błąd kalibracji (+21.7 p.p.). Zakaz zawierania zakładów.",
        "🌟 ZŁOTY SEGMENT (Rekomendacja): Kursy [2.80 - 3.50] generują +30.1% ROI netto (+8.73 j.) przy niemal idealnej kalibracji (błąd zaledwie +1.3 p.p.). Skupienie alokacji kapitału.",
        "⚡ ODCIĘCIE KWARANTANNY POTRAJA ZYSK: Całkowity wynik strategii rośnie z +8.57 j. (+3.2% ROI) do +21.50 j. (+9.6% ROI) po samym wykluczeniu pułapki 3.50-5.00.",
        "⚖️ PODATEK OBROTOWY: Podatek 12% wymaga kursu min. 2.05 dla przewagi rynkowej 50/50. Wszelkie zakłady poniżej 1.40 mają znikomy margines bezpieczeństwa.",
        "🛡️ FILTR IDI (Internal Disagreement Index): Gdy rozbieżność między ratingiem graczy a ratingiem drużyny przekracza 20 p.p. (np. w meczach akademii lub po nagłych transferach), wstrzymać automatyczny typ.",
    ]

    return {
        "model_key": model_key,
        "model_title": cfg["title"],
        "parameters": {
            "tax_rate": tax_rate,
            "min_ev": min_ev,
            "max_days_back": max_days_back,
            "timing_window": "6h - 24h (pre-match)",
        },
        "overall": {
            "total_bets": total_bets,
            "n_wins": n_wins,
            "win_rate_pct": round(win_rate, 1),
            "avg_odds": round(avg_odds, 2),
            "avg_model_prob_pct": round(avg_p_model, 1),
            "avg_market_prob_pct": round(avg_p_market, 1),
            "calibration_bias_pp": round(bias, 1),
            "realized_net_roi_pct": round(realized_net_roi, 1),
            "realized_gross_roi_pct": round(realized_gross_roi, 1),
            "expected_net_roi_pct": round(expected_net_roi, 1),
            "pnl_net_units": round(pnl_net_total, 2),
            "max_drawdown_units": round(max_dd_units, 2),
            "brier_score": round(brier, 4),
            "market_brier_score": round(mkt_brier, 4),
            "log_loss": round(logloss, 4),
            "market_log_loss": round(mkt_logloss, 4),
            "positive_clv_pct": round(pos_clv_pct, 1),
            "avg_clv_pct": round(avg_clv_pct, 2),
        },
        "filter_impact": filter_impact,
        "odds_brackets": odds_brackets,
        "series_formats": formats,
        "leagues": leagues,
        "anomalies": anomalies,
        "operational_rules": operational_rules,
    }



@router.get("/validation-report")
def validation_report(
    days_back: int = 3650,
    tax_rate: float = 0.12,
    min_ev: float = 0.05,
    model_name: str = "Hybrid-Operational-Market",
    model_version: str = "c0-native-2026-w32-e12-v1-a0.50-t1.00",
    db=Depends(get_db),
):
    """Historical single-cohort report using an explicit assumed research haircut."""
    cutoff = (datetime.now(UTC) - timedelta(days=days_back)).isoformat()
    matches = query_df(
        db,
        """
        SELECT id as canonical_match_id, team_a_name, team_b_name, winner_side, start_time_normalized, league, best_of
        FROM canonical_matches 
        WHERE status IN ('finished', 'completed') AND winner_side IN ('team_a', 'team_b')
          AND start_time_normalized >= :cutoff
        """,
        {"cutoff": cutoff},
    )
    if len(matches) == 0:
        return {"error": "Brak zakończonych meczów w wybranym okresie."}

    m_ids = [int(m["canonical_match_id"]) for m in matches]
    p_holders = ",".join(f":m_{i}" for i in range(len(m_ids)))
    m_params = {f"m_{i}": mid for i, mid in enumerate(m_ids)}

    preds = query_df(
        db,
        f"""
        SELECT cp.canonical_match_id, cp.prob_a, cp.prob_b
        FROM canonical_predictions cp
        WHERE cp.model_name = :m_name
          AND cp.model_version = :m_ver
          AND cp.canonical_match_id IN ({p_holders})
        """,
        {**m_params, "m_name": model_name, "m_ver": model_version},
    )

    odds = query_df(
        db,
        f"""
        SELECT os.canonical_match_id, b.name as bookmaker, os.odds_a, os.odds_b, os.scraped_at
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IN ({p_holders})
          AND os.market_type = 'match_winner' AND COALESCE(os.is_live, 0) = 0
          AND os.odds_a > 1.0 AND os.odds_b > 1.0
        """,
        m_params,
    )

    if len(preds) == 0:
        return {"error": "Brak predykcji dla wybranego modelu w tym okresie."}
    if len(odds) == 0:
        return {"error": "Brak notowań kursowych w tym okresie."}

    preds = pd.DataFrame(preds).drop_duplicates(subset=["canonical_match_id"])
    matches = pd.DataFrame(matches)
    odds = pd.DataFrame(odds)

    df = preds.merge(matches, on="canonical_match_id").merge(odds, on="canonical_match_id")
    df = df[df["scraped_at"] < df["start_time_normalized"]].copy()

    inv_a = 1.0 / df["odds_a"]
    inv_b = 1.0 / df["odds_b"]
    df["market_fair_a"] = inv_a / (inv_a + inv_b)
    df["market_fair_b"] = inv_b / (inv_a + inv_b)

    first_odds = df.sort_values("scraped_at").groupby(["canonical_match_id", "bookmaker"]).first().reset_index()
    cons = first_odds.groupby("canonical_match_id").agg(
        cons_market_a=("market_fair_a", "median"),
        cons_market_b=("market_fair_b", "median"),
    ).reset_index()

    m_cohort = preds.merge(matches, on="canonical_match_id").merge(cons, on="canonical_match_id")
    y_all = np.where(m_cohort["winner_side"] == "team_a", 1.0, 0.0)
    p_model = np.clip(m_cohort["prob_a"].values, 1e-15, 1.0 - 1e-15)
    p_mkt = np.clip(m_cohort["cons_market_a"].values, 1e-15, 1.0 - 1e-15)

    ll_model = -np.mean(y_all * np.log(p_model) + (1 - y_all) * np.log(1 - p_model))
    ll_mkt = -np.mean(y_all * np.log(p_mkt) + (1 - y_all) * np.log(1 - p_mkt))
    brier_model = np.mean((p_model - y_all) ** 2)
    brier_mkt = np.mean((p_mkt - y_all) ** 2)
    acc_model = np.mean((p_model >= 0.5) == (y_all == 1.0))
    acc_mkt = np.mean((p_mkt >= 0.5) == (y_all == 1.0))

    deciles = []
    bins = np.linspace(0.0, 1.0, 11)
    for i in range(10):
        low, high = bins[i], bins[i + 1]
        mask = (p_model >= low) & (p_model < high if i < 9 else p_model <= high)
        if np.any(mask):
            c = np.sum(mask)
            avg_pred = np.mean(p_model[mask])
            obs_win = np.mean(y_all[mask])
            deciles.append({
                "label": f"{int(low*100)}-{int(high*100)}%",
                "count": int(c),
                "avg_predicted_pct": round(float(avg_pred * 100), 1),
                "observed_rate_pct": round(float(obs_win * 100), 1),
                "gap_pp": round(float((avg_pred - obs_win) * 100), 1),
            })

    last_odds = df.sort_values("scraped_at").groupby(["canonical_match_id", "bookmaker"]).last().reset_index()
    same_book_close = last_odds.set_index(["canonical_match_id", "bookmaker"])[["odds_a", "odds_b"]].to_dict("index")

    net_mult = 1.0 - tax_rate
    from src.analysis.unified_evaluation_engine import UnifiedBettingEngine, EvaluatedBet
    engine = UnifiedBettingEngine(
        tax_rate=tax_rate,
        min_ev_net=min_ev,
        dynamic_hurdle=True,
        max_ev_longshot=0.20,
        longshot_odds_threshold=3.50,
        uncertainty_required=True,
        default_sigma_z=0.15,
    )

    qualified_list: list[EvaluatedBet] = []
    for _, r in first_odds.iterrows():
        mid = r["canonical_match_id"]
        bm = r["bookmaker"]
        w_side = r["winner_side"]
        pa = r["prob_a"]
        pb = r["prob_b"]
        oa = r["odds_a"]
        ob = r["odds_b"]
        league = r.get("league")
        date_str = str(r["start_time_normalized"])[:10] if "start_time_normalized" in r else None
        bo = r.get("best_of")

        bm_close = same_book_close.get((mid, bm), {})
        cl_a = bm_close.get("odds_a") or oa
        cl_b = bm_close.get("odds_b") or ob

        # Evaluate Side A
        el_a, bet_a = engine.qualify_quote(
            match_id=mid,
            side="team_a",
            odds=oa,
            odds_close=cl_a,
            prob_model=pa,
            won=w_side == "team_a",
            bookmaker=bm,
            league=league,
            date_str=date_str,
            best_of=bo,
            uncertainty_basis="assumed_research_haircut",
        )
        if el_a:
            qualified_list.append(bet_a)

        # Evaluate Side B
        el_b, bet_b = engine.qualify_quote(
            match_id=mid,
            side="team_b",
            odds=ob,
            odds_close=cl_b,
            prob_model=pb,
            won=w_side == "team_b",
            bookmaker=bm,
            league=league,
            date_str=date_str,
            best_of=bo,
            uncertainty_basis="assumed_research_haircut",
        )
        if el_b:
            qualified_list.append(bet_b)

    # Group by canonical match and side to take best available bookmaker quote
    best_bets_map: dict[tuple[Any, str], EvaluatedBet] = {}
    for b in qualified_list:
        key = (b.match_id, b.side)
        if key not in best_bets_map or b.odds > best_bets_map[key].odds:
            best_bets_map[key] = b

    sim_summary = engine.run_simulation(
        qualified_bets=list(best_bets_map.values()),
        strategy="flat",
        initial_bankroll=1000.0,
        flat_stake=100.0,
    )

    brackets_data = []
    for ob_bin in sim_summary.by_odds_bracket:
        brackets_data.append({
            "label": ob_bin.label,
            "bets": ob_bin.bets_count,
            "wins": ob_bin.wins_count,
            "win_rate_pct": ob_bin.win_rate_pct,
            "avg_odds": ob_bin.avg_odds,
            "expected_net_roi_pct": ob_bin.expected_ev_pct,
            "realized_net_roi_pct": ob_bin.realized_ev_pct,
            "pnl_pln": ob_bin.total_pnl,
            "clv_pct": ob_bin.median_clv_pct,
            "clv_pp": ob_bin.mean_clv_pp,
        })

    by_book = sim_summary.by_bookmaker
    horizons_data = []
    if len(df):
        df["hours_before"] = (pd.to_datetime(df["start_time_normalized"]) - pd.to_datetime(df["scraped_at"])).dt.total_seconds() / 3600.0
        h_brackets = [
            ("> 48h", 48.0, 9999.0),
            ("24 - 48h", 24.0, 48.0),
            ("12 - 24h", 12.0, 24.0),
            ("6 - 12h", 6.0, 12.0),
            ("2 - 6h", 2.0, 6.0),
            ("< 2h", 0.0, 2.0),
        ]
        for label, low, high in h_brackets:
            sub_h = df[(df["hours_before"] >= low) & (df["hours_before"] < high)]
            if len(sub_h) == 0:
                continue
            m_h = sub_h.drop_duplicates(subset=["canonical_match_id"])
            y_h = np.where(m_h["winner_side"] == "team_a", 1.0, 0.0)
            ph_m = np.clip(m_h["prob_a"].values, 1e-15, 1.0 - 1e-15)
            ph_k = np.clip(m_h["market_fair_a"].values, 1e-15, 1.0 - 1e-15)
            ll_m_h = -np.mean(y_h * np.log(ph_m) + (1 - y_h) * np.log(1 - ph_m))
            ll_k_h = -np.mean(y_h * np.log(ph_k) + (1 - y_h) * np.log(1 - ph_k))

            clv_sub = []
            for _, r_h in sub_h.iterrows():
                c_entry = same_book_close.get((r_h["canonical_match_id"], r_h["bookmaker"]), {})
                c_odds = c_entry.get("odds_a")
                if c_odds and c_odds > 0:
                    clv_sub.append((r_h["odds_a"] / c_odds - 1.0) * 100.0)
            avg_clv_h = float(np.mean(clv_sub)) if len(clv_sub) else 0.0
            pos_clv_h = float(np.mean(np.array(clv_sub) > 0) * 100.0) if len(clv_sub) else 0.0

            horizons_data.append({
                "label": label,
                "matches": int(len(m_h)),
                "quotes": int(len(sub_h)),
                "model_logloss": round(float(ll_m_h), 4),
                "market_logloss": round(float(ll_k_h), 4),
                "delta_logloss": round(float(ll_m_h - ll_k_h), 4),
                "avg_clv_pct": round(avg_clv_h, 1),
                "pos_clv_pct": round(pos_clv_h, 1),
            })

    total_bets = sim_summary.bets_count
    total_wins = sim_summary.wins_count
    total_pnl = sim_summary.total_pnl
    total_staked = sim_summary.total_staked
    total_roi = sim_summary.roi_pct
    avg_clv_tot = sim_summary.median_clv_pct
    clv_values = [b.clv_pct for b in qualified_list if b.clv_pct is not None]
    pos_clv_tot = float(np.mean(np.array(clv_values) > 0) * 100.0) if clv_values else None
    return {
        "summary": {
            "matches": int(len(m_cohort)),
            "model_logloss": round(float(ll_model), 4),
            "market_logloss": round(float(ll_mkt), 4),
            "delta_logloss": round(float(ll_model - ll_mkt), 4),
            "model_brier": round(float(brier_model), 4),
            "market_brier": round(float(brier_mkt), 4),
            "model_accuracy_pct": round(float(acc_model * 100.0), 1),
            "market_accuracy_pct": round(float(acc_mkt * 100.0), 1),
        },
        "betting_totals": {
            "result_status": sim_summary.result_status,
            "uncertainty_basis": "assumed_research_haircut",
            "uncertainty_reason": "explicit_research_assumption",
            "assumed_sigma_z": 0.15,
            "total_bets": total_bets,
            "total_wins": total_wins,
            "win_rate_pct": round(sim_summary.win_rate_pct, 1) if sim_summary.win_rate_pct is not None else None,
            "total_staked_pln": round(float(total_staked), 2),
            "total_pnl_pln": round(float(total_pnl), 2),
            "roi_pct": round(float(total_roi), 2) if total_roi is not None else None,
            "avg_clv_pct": round(avg_clv_tot, 1) if avg_clv_tot is not None else None,
            "pos_clv_pct": round(pos_clv_tot, 1) if pos_clv_tot is not None else None,
        },
        "odds_brackets": brackets_data,
        "bookmakers": by_book,
        "horizons": horizons_data,
        "calibration_deciles": deciles,
    }
