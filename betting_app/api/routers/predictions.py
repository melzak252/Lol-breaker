"""Router: /api/predictions — EV+ signals."""

from __future__ import annotations

from datetime import UTC, datetime
from betting_app.services.prematch_quote_service import load_prematch_quotes, parse_aware_utc, signal_is_actionable
from betting_app.services.market_service import expected_value
from fastapi import APIRouter, Depends, Query

from betting_app.api.deps import get_db, query_df
from betting_app.api.schemas import EVSignal, EVSignalResponse
from betting_app.services.market_service import kelly_fraction, none_or_float
from betting_app.core.models import get_active_hybrid, list_registered_models
from betting_app.services.bet_qualification_service import (
    DEFAULT_BASE_MIN_EV,
    DEFAULT_MAX_EV_NET,
    is_bet_eligible,
    model_requires_uncertainty,
    prediction_safety_diagnostics,
    qualification_tier,
)

DEFAULT_HYBRID_MODEL = get_active_hybrid()
DEFAULT_HYBRID_MODEL_NAME = DEFAULT_HYBRID_MODEL.hybrid_model_name
DEFAULT_HYBRID_MODEL_VERSION = DEFAULT_HYBRID_MODEL.hybrid_model_version
from betting_app.ml.model_lifecycle import RETIRED_PUBLIC_MODEL_NAME

router = APIRouter(prefix="/predictions", tags=["predictions"])

TAX_RATE = 0.12


@router.get("", response_model=EVSignalResponse)
def list_predictions(
    min_ev: float = Query(0.0),
    max_ev: float | None = Query(None),
    min_books: int = Query(1),
    model_name: str | None = Query(DEFAULT_HYBRID_MODEL_NAME),
    model_version: str | None = Query(DEFAULT_HYBRID_MODEL_VERSION),
    limit: int = Query(50, le=200),
    db=Depends(get_db),
):
    # Build dynamic WHERE clause for optional model filters
    where_clauses = [
        "mes.status='new'",
        "mes.side IN ('a', 'b')",
        "cp.model_name <> :retired_model_name",
        "cp.prediction_status='active'",
    ]
    params = {
        "min_ev": min_ev,
        "lim": limit,
        "retired_model_name": RETIRED_PUBLIC_MODEL_NAME,
    }

    if model_name:
        where_clauses.append("cp.model_name = :model_name")
        params["model_name"] = model_name
    if model_version:
        where_clauses.append("cp.model_version = :model_version")
        params["model_version"] = model_version

    where_sql = " AND ".join(where_clauses)

    rows = query_df(
        db,
        f"""
        SELECT
            mes.*,
            cp.model_name, cp.model_version,
            cp.prob_a, cp.prob_b, cp.diagnostics_json,
            cp.id AS prediction_id, cp.canonical_match_id AS prediction_match_id,
            cp.predicted_at, cp.data_cutoff_at, cp.prediction_status,
            cm.team_a_name, cm.team_b_name,
            cm.league, cm.start_time_normalized, cm.best_of,
            b.name AS bookmaker_name
        FROM model_ev_signals mes
        JOIN canonical_predictions cp ON cp.id=mes.canonical_prediction_id
        JOIN canonical_matches cm ON cm.id=mes.canonical_match_id
        JOIN bookmakers b ON b.id=mes.bookmaker_id
        WHERE {where_sql}
        ORDER BY ev DESC
        """,
        params,
    )
    now = datetime.now(UTC)
    contexts = load_prematch_quotes(sorted({int(r["canonical_match_id"]) for r in rows}), decision_at=now, session=db)

    signals: list[EVSignal] = []
    for row in rows:
        current = contexts[int(row["canonical_match_id"])]
        evaluated = parse_aware_utc(row.get("evaluated_at"))
        if evaluated is None:
            continue
        evaluation = load_prematch_quotes([current.canonical_match_id], decision_at=evaluated, session=db)[current.canonical_match_id]
        prediction = dict(row, id=row["prediction_id"], canonical_match_id=row["prediction_match_id"])
        if not signal_is_actionable(row, prediction, evaluation_context=evaluation, current_context=current, session=db)[0]:
            continue
        if len(current.quotes) < min_books:
            continue
        odds = none_or_float(row.get("odds"))
        try:
            safety = prediction_safety_diagnostics(
                row.get("diagnostics_json"),
                row["prob_a"],
                row["prob_b"],
                uncertainty_required=model_requires_uncertainty(row["model_name"]),
            )
            side = row["side"]
            prob = row[f"prob_{side}"]
            conservative = safety.get(f"p_low_{side}")
            eligible, _, decision = is_bet_eligible(
                prob_model=prob,
                prob_conservative=conservative,
                odds=odds,
                uncertainty_required=safety["uncertainty_required"],
                uncertainty_basis=safety["uncertainty_basis"],
                uncertainty_reason=safety["uncertainty_reason"],
                prob_market_novig=none_or_float(row.get("market_prob")),
                tax_rate=TAX_RATE,
                min_ev_net=max(DEFAULT_BASE_MIN_EV, min_ev),
                rating_disagreement=safety.get("rating_disagreement"),
                competition_tier=qualification_tier(
                    row.get("league"), row.get("start_time_normalized")
                ),
                max_ev_net=max_ev,
                best_of=int(row["best_of"]) if row.get("best_of") is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not eligible:
            continue
        prob_for_ev = decision["prob_for_ev"]
        ev = expected_value(prob_for_ev, odds, TAX_RATE)
        signals.append(
            EVSignal(
                canonical_match_id=row["canonical_match_id"],
                odds_snapshot_id=row["odds_snapshot_id"],
                quote_scraped_at=next(q.scraped_at.isoformat() for q in current.quotes if q.odds_snapshot_id == row["odds_snapshot_id"]),
                evaluated_at=evaluated.isoformat(),
                match=f"{row.get('team_a_name','?')} vs {row.get('team_b_name','?')}",
                league=row.get("league"),
                start_time_normalized=row.get("start_time_normalized"),
                model_name=row.get("model_name", "?"),
                model_version=row.get("model_version", "?"),
                side=row["side"],
                odds=odds,
                bookmaker=row.get("bookmaker_name", "?"),
                model_prob=prob,
                prob_conservative=decision["prob_conservative"],
                prob_for_ev=prob_for_ev,
                uncertainty_basis=decision["uncertainty_basis"],
                uncertainty_reason=decision["uncertainty_reason"],
                market_prob=none_or_float(row.get("market_prob")),
                ev=ev,
                kelly=kelly_fraction(prob_for_ev, odds, TAX_RATE),
                offer_url=next(q.offer_url for q in current.quotes if q.odds_snapshot_id == row["odds_snapshot_id"]),
            )
        )

    signals.sort(key=lambda signal: (-signal.ev, signal.canonical_match_id, signal.odds_snapshot_id))
    signals = signals[:limit]
    return EVSignalResponse(total=len(signals), signals=signals)


@router.get("/models")
def get_ev_models():
    """List models available for EV predictions."""
    return {
        "active_hybrid_model": get_active_hybrid().hybrid_model_name,
        "models": list_registered_models(),
    }
