"""Database loaders for historical model-vs-bookmaker backtests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from betting_app.core.db import query_df
from betting_app.ml.backtesting.types import HistoricalPrediction, MatchLabel, OddsQuote


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if value != value:  # pandas/numpy NaN
            return None
    except TypeError:
        pass
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

def load_finished_match_labels(
    *,
    days_back: int | None = None,
    session: Session | None = None,
) -> list[MatchLabel]:
    """Load finished canonical matches with a known winner side."""

    where = ["status IN ('finished', 'completed')", "winner_side IN ('team_a', 'team_b', 'a', 'b')"]
    params: dict[str, Any] = {}
    if days_back is not None:
        where.append("REPLACE(start_time_normalized, 'T', ' ') >= REPLACE(:min_start, 'T', ' ')")
        params["min_start"] = (datetime.now(UTC) - timedelta(days=int(days_back))).isoformat(timespec="seconds")

    try:
        df = query_df(
            f"""
            SELECT id AS canonical_match_id,
                   winner_side,
                   start_time_normalized,
                   league,
                   result_recorded_at
            FROM canonical_matches
            WHERE {' AND '.join(where)}
            """,
            params,
            session=session,
        )
    except Exception:
        df = query_df(
            f"""
            SELECT id AS canonical_match_id,
                   winner_side,
                   start_time_normalized,
                   league
            FROM canonical_matches
            WHERE {' AND '.join(where)}
            """,
            params,
            session=session,
        )
    labels: list[MatchLabel] = []
    for row in df.to_dict("records"):
        raw_side = row["winner_side"]
        side = "a" if raw_side == "team_a" else "b" if raw_side == "team_b" else raw_side
        labels.append(
            MatchLabel(
                canonical_match_id=int(row["canonical_match_id"]),
                winner_side=side,
                start_time=_parse_dt(row.get("start_time_normalized")),
                league=row.get("league"),
                result_available_at=_parse_dt(row.get("result_recorded_at")) if "result_recorded_at" in row else None,
            )
        )
    return labels


def load_predictions(
    *,
    model_name: str | None = None,
    model_version: str | None = None,
    only_active: bool = True,
    latest_per_match: bool = True,
    session: Session | None = None,
) -> list[HistoricalPrediction]:
    """Load canonical predictions for one model/version or all models."""

    where = ["prob_a IS NOT NULL", "prob_b IS NOT NULL"]
    params: dict[str, Any] = {}
    if model_name:
        where.append("model_name = :model_name")
        params["model_name"] = model_name
    if model_version:
        where.append("model_version = :model_version")
        params["model_version"] = model_version
    if only_active:
        where.append("COALESCE(prediction_status, 'active') = 'active'")

    df = query_df(
        f"""
        SELECT id AS prediction_id,
               canonical_match_id,
               model_name,
               model_version,
               prob_a,
               prob_b,
               predicted_at,
               data_cutoff_at
        FROM canonical_predictions
        WHERE {' AND '.join(where)}
        ORDER BY predicted_at, canonical_match_id
        """,
        params,
        session=session,
    )
    predictions = [
        HistoricalPrediction(
            canonical_match_id=int(row["canonical_match_id"]),
            model_name=str(row["model_name"]),
            model_version=str(row["model_version"]),
            prob_a=float(row["prob_a"]),
            prob_b=float(row["prob_b"]),
            predicted_at=_parse_dt(row.get("predicted_at")),
            data_cutoff_at=_parse_dt(row.get("data_cutoff_at")),
            prediction_id=int(row["prediction_id"]) if row.get("prediction_id") is not None else None,
        )
        for row in df.to_dict("records")
    ]
    if not latest_per_match:
        return predictions

    latest: dict[tuple[int, str, str], HistoricalPrediction] = {}
    for prediction in predictions:
        key = (prediction.canonical_match_id, prediction.model_name, prediction.model_version)
        current = latest.get(key)
        if current is None:
            latest[key] = prediction
            continue
        current_ts = current.predicted_at or datetime.min
        next_ts = prediction.predicted_at or datetime.min
        if next_ts >= current_ts:
            latest[key] = prediction
    return sorted(latest.values(), key=lambda p: (p.predicted_at is None, p.predicted_at, p.canonical_match_id))


def load_odds_quotes(
    *,
    canonical_match_ids: set[int] | None = None,
    session: Session | None = None,
) -> list[OddsQuote]:
    """Load collected odds snapshots already mapped to canonical matches."""

    params: dict[str, Any] = {}
    match_filter = ""
    if canonical_match_ids:
        placeholders: list[str] = []
        for idx, match_id in enumerate(sorted(canonical_match_ids)):
            key = f"match_id_{idx}"
            placeholders.append(f":{key}")
            params[key] = int(match_id)
        match_filter = f"AND os.canonical_match_id IN ({', '.join(placeholders)})"

    df = query_df(
        f"""
        SELECT os.id AS odds_snapshot_id,
               os.canonical_match_id,
               os.bookmaker_id,
               COALESCE(b.name, 'unknown') AS bookmaker_name,
               os.odds_a,
               os.odds_b,
               os.scraped_at,
               os.offer_url
        FROM odds_snapshots os
        LEFT JOIN bookmakers b ON b.id = os.bookmaker_id
        WHERE os.canonical_match_id IS NOT NULL
          AND os.odds_a > 1.0
          AND os.odds_b > 1.0
          {match_filter}
        ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at
        """,
        params,
        session=session,
    )
    return [
        OddsQuote(
            canonical_match_id=int(row["canonical_match_id"]),
            bookmaker_id=int(row["bookmaker_id"]),
            bookmaker_name=str(row["bookmaker_name"]),
            odds_a=float(row["odds_a"]),
            odds_b=float(row["odds_b"]),
            scraped_at=_parse_dt(row.get("scraped_at")) or datetime.min,
            odds_snapshot_id=int(row["odds_snapshot_id"]) if row.get("odds_snapshot_id") is not None else None,
            offer_url=row.get("offer_url"),
        )
        for row in df.to_dict("records")
    ]
