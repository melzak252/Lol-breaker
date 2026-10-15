"""Router: /api/matches — upcoming match board and detail.

Uses SQLAlchemy text() with named parameters (:style).
Compatible with SQLite and PostgreSQL.
"""

from __future__ import annotations
from collections import defaultdict
from datetime import UTC, datetime, timedelta
import json
import math
import os
import re
import secrets
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from fastapi import APIRouter, Depends, Header, HTTPException

from betting_app.api.deps import get_db, query_df, query_one
from betting_app.services.canonical_match_service import canonical_team_key
from betting_app.services.current_roster_service import clean_player_name
from betting_app.api.schemas import (
    BettingRecommendation,
    ParlayRecommendationsResponse,
    AliasBlockRequest,
    AliasCreateRequest,
    AliasCreateResponse,
    AliasDeleteRequest,
    BookmakerOddsRow,
    GolggTeamsResponse,
    MatchBestOfUpdate,
    MatchBoardItem,
    MatchBoardResponse,
    MatchDetailResponse,
    MatchResultItem,
    MatchResultsResponse,
    MatchupSimulationRequest,
    MatchupSimulationResponse,
    PredictResponse,
    PredictionHistoryPoint,
    PredictionRow,
    RosterInfo,
    RosterPlayer,
    OddsHistoryPoint,
    TeamComparisonInfo,
    TeamMappingInfo,
    TeamRecentStats,
    UnmappedMatchItem,
    UnmappedMatchesResponse,
    GolggMatchCandidate,
    GolggMatchCandidatesResponse,
    MatchMappingRequest,
    MappingCheckResponse,
    MappingReviewDecisionRequest,
    MappingReviewDecisionResponse,
    MappingReviewItem,
    MappingReviewResponse,
    MatchRosterOverrideRequest,
    MatchRosterOverrideResponse,
    RosterVerificationRequest,
    RosterStatusResponse,
    SimulateFearlessRequest,
    SimulateFearlessResponse,
)
from betting_app.services.parlay_service import find_parlay_recommendations
from betting_app.services.roster_verification_service import RosterVerificationService
from betting_app.services.canonical_match_service import align_snapshot_odds, competition_family
from betting_app.core.ev import fair_market_probabilities
from betting_app.services.market_service import (
    enrich_arbitrage,
    expected_value,
    kelly_fraction,
    none_or_float,
    safe_json_get,
)
from betting_app.ml.calibration.conformal_contract import conformal_bounds_for_side
from betting_app.services.bet_qualification_service import (
    qualify_prediction_sides,
)

from betting_app.services.mapping_service import native_golgg_team_id, suggest_mapping
from betting_app.services.current_roster_service import lock_current_roster_teams, upsert_current_roster
from betting_app.services.thesis_inference_service import _load_roster_overrides
from betting_app.core.models import (
    PredictionEngine,
    get_active_hybrid,
    get_active_model,
    get_active_prediction_db_params,
    list_registered_models,
)
from betting_app.services.upcoming_inference_service import (
    generate_hybrid_predictions,
    predict_operational_match,
    register_operational_model,
    validate_native_serving_inputs,
    upsert_upcoming_features,
)
from betting_app.core.models.engine import blend_market_probability
from betting_app.services.prematch_quote_service import (
    load_prematch_quotes, parse_aware_utc, latest_predictions,
    prediction_time_is_actionable, stored_market_context_is_current,
    hybrid_prediction_contract_is_current,
)

router = APIRouter(prefix="/matches", tags=["matches"])


def _identity_review_token() -> str | None:
    environment_token = os.getenv("IDENTITY_REVIEW_TOKEN")
    if environment_token:
        return environment_token
    token_file = os.getenv("IDENTITY_REVIEW_TOKEN_FILE")
    if not token_file:
        return None
    try:
        token = Path(token_file).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Identity review token file is unreadable",
        ) from exc
    return token or None



TAX_RATE = 0.12
DEFAULT_MAX_ODDS_AGE_HOURS = 24.0
DEFAULT_MODEL_NAME = get_active_model().name
DEFAULT_MODEL_VERSION = get_active_model().version
HYBRID_MODEL_NAME = get_active_hybrid().hybrid_model_name
HYBRID_MODEL_VERSION = get_active_hybrid().hybrid_model_version


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _raw_diagnostics(value: Any) -> dict[str, Any] | None:
    """Expose only retained diagnostic objects, preserving historical absence."""
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _saved_context_valid(db, saved, current) -> bool:
    saved = _json_object(saved)
    at = parse_aware_utc(saved.get("decision_at"))
    if at is None:
        return False
    evaluation = load_prematch_quotes(
        [current.canonical_match_id], decision_at=at, session=db,
    )[current.canonical_match_id]
    return stored_market_context_is_current(
        saved, evaluation_context=evaluation, current_context=current,
    )[0]


def _serving_predictions(db, rows, context) -> dict[str, Any]:
    """Use an intact hybrid or the exact active pure base/current q, once."""
    result: dict[str, Any] = {}
    hybrid = get_active_hybrid()
    for row in latest_predictions(rows):
        name, version = row.get("model_name"), row.get("model_version")
        pa, pb = _probability(row.get("prob_a")), _probability(row.get("prob_b"))
        if pa is None or pb is None or not math.isclose(pa+pb, 1., abs_tol=1e-5):
            continue
        if not prediction_time_is_actionable(
            row, decision_at=context.decision_at, start_at=context.start_at,
            model_name=name, model_version=version,
        )[0]:
            continue
        if (name, version) == (hybrid.base_model.name, hybrid.base_model.version):
            result.update(model_prob_a=none_or_float(row.get("prob_a")),
                          model_prob_b=none_or_float(row.get("prob_b")),
                          model_diagnostics=row.get("diagnostics_json"))
        elif (name, version) == (hybrid.hybrid_model_name, hybrid.hybrid_model_version):
            diag = _json_object(row.get("diagnostics_json"))
            saved = _json_object(diag.get("market_context"))
            if hybrid_prediction_contract_is_current(row, session=db)[0] and _saved_context_valid(db, saved, context):
                result.update(hybrid_prob_a=none_or_float(row.get("prob_a")),
                              hybrid_prob_b=none_or_float(row.get("prob_b")),
                              hybrid_diagnostics=row.get("diagnostics_json"))
    if result.get("hybrid_prob_a") is None and result.get("model_prob_a") is not None and context.prob_a is not None:
        p = blend_market_probability(
            result["model_prob_a"], context.prob_a, alpha=hybrid.alpha,
            temperature=hybrid.temperature, blending_mode=hybrid.blending_mode,
        )
        result.update(hybrid_prob_a=p, hybrid_prob_b=1-p,
                      hybrid_diagnostics=result.get("model_diagnostics"))
    return result


def _finite_float(value: Any) -> float | None:
    try:
        number = none_or_float(value)
    except OverflowError:
        return None
    if number is None or not math.isfinite(number):
        return None
    return number


def _probability(value):
    if isinstance(value, bool):
        return None
    value = _finite_float(value)
    return value if value is not None and 0 <= value <= 1 else None


def _latest_display_states(rows, decision_at):
    latest = {}
    unknown = set()
    for row in rows:
        book = row["bookmaker_id"]
        clock = parse_aware_utc(row.get("scraped_at"))
        if clock is None:
            unknown.add(book)
            continue
        if clock > decision_at:
            continue
        key = (clock, int(row["odds_snapshot_id"]))
        if book not in latest or key > latest[book][0]:
            latest[book] = (key, row)
    return [row for book, (_, row) in latest.items() if book not in unknown]


def _pick_snapshot(rows: list[dict[str, Any]], odds_mode: str, *, start_at=None) -> dict[str, Any] | None:
    if any(parse_aware_utc(row.get("scraped_at")) is None for row in rows):
        return None
    ordered = sorted(
        ((parse_aware_utc(r.get("scraped_at")), int(r.get("odds_snapshot_id", r.get("id", 0))), r)
         for r in rows if parse_aware_utc(r.get("scraped_at")) is not None),
        key=lambda item: item[:2],
    )
    if start_at is not None:
        ordered = [item for item in ordered if item[0] < start_at]
    if not ordered:
        return None
    if odds_mode == "open":
        return ordered[0][2]
    if odds_mode == "mid":
        if start_at is None:
            return None
        midpoint = ordered[0][0] + (start_at - ordered[0][0]) / 2
        ordered = [item for item in ordered if item[0] <= midpoint]
    return ordered[-1][2]


def _align(row: dict, n_a: str, n_b: str) -> tuple[float, float] | None:
    return align_snapshot_odds(
        n_a, n_b, str(row.get("raw_team_a") or ""), str(row.get("raw_team_b") or ""),
        row.get("odds_a"), row.get("odds_b"), league=row.get("league"),
        source_system="bookmaker", match_date=row.get("start_time_normalized"),
    )


def _historical_consensus(rows, origin, n_a, n_b):
    latest = {}
    unknown = {row["bookmaker_id"] for row in rows if parse_aware_utc(row.get("scraped_at")) is None}
    for row in rows:
        clock = parse_aware_utc(row.get("scraped_at"))
        if clock is None or clock > origin:
            continue
        key = (clock, int(row.get("odds_snapshot_id", row.get("id", 0))))
        book = row["bookmaker_id"]
        if book not in latest or key > latest[book][0]:
            latest[book] = (key, row)
    pairs = []
    for book, (_, row) in latest.items():
        if book in unknown:
            continue
        aligned = _align(row, n_a, n_b)
        if row.get("is_live") == 0 and aligned is not None:
            pairs.append(fair_market_probabilities(*aligned)[0])
    return sum(pairs) / len(pairs) if pairs else None


def _historical_prediction(rows, *, origin, start_at, model_name, model_version):
    valid = []
    for row in rows:
        issued, cutoff = parse_aware_utc(row.get("predicted_at")), parse_aware_utc(row.get("data_cutoff_at"))
        if (row.get("model_name"), row.get("model_version")) != (model_name, model_version):
            continue
        if issued is not None and cutoff is not None and start_at is not None and cutoff <= issued <= origin < start_at:
            valid.append((issued, int(row["id"]), row))
    return max(valid, key=lambda item: item[:2])[2] if valid else None


def _historical_blend_contract(db, name, version):
    artifacts = query_df(db, "SELECT * FROM model_artifacts WHERE model_name=:name AND model_version=:version",
                         {"name": name, "version": version})
    contracts = []
    for artifact in artifacts:
        params = _json_object(artifact.get("model_params_json"))
        schema = _json_object(artifact.get("feature_schema_json"))
        declared = schema.get("base_model")
        if isinstance(declared, str) and "/" in declared:
            base_name, base_version = declared.rsplit("/", 1)
            if params.get("base_model_name") not in (None, base_name) or params.get("base_model_version") not in (None, base_version):
                return None
            params.setdefault("base_model_name", base_name)
            params.setdefault("base_model_version", base_version)
        contracts.append(params)
    active = get_active_hybrid()
    if (name, version) == (active.hybrid_model_name, active.hybrid_model_version):
        contracts.append(dict(base_model_name=active.base_model.name, base_model_version=active.base_model.version,
                              alpha=active.alpha, temperature=active.temperature, blending_mode=active.blending_mode))
    required = ("base_model_name", "base_model_version", "alpha", "temperature", "blending_mode")
    if not contracts or any(any(k not in c for k in required) for c in contracts):
        return None
    first = {k: contracts[0][k] for k in required}
    if any(any(c[k] != first[k] for k in required) for c in contracts):
        return None
    try:
        blend_market_probability(.5, .5, alpha=first["alpha"], temperature=first["temperature"],
                                 blending_mode=first["blending_mode"])
    except (TypeError, ValueError):
        return None
    return first


def _display_matches_quote(row, quote, aligned):
    """An operational row ID cannot authorize values replaced by an upsert."""
    return (
        quote is not None
        and row.get("canonical_match_id") == quote.canonical_match_id
        and row.get("bookmaker_id") == quote.bookmaker_id
        and row.get("bookmaker") == quote.bookmaker
        and row.get("market_type", "match_winner") == "match_winner"
        and row.get("is_live") == 0
        and row.get("raw_team_a") == quote.raw_team_a
        and row.get("raw_team_b") == quote.raw_team_b
        and parse_aware_utc(row.get("scraped_at")) == quote.scraped_at
        and aligned == (quote.odds_a, quote.odds_b)
        and row.get("source_url") == quote.source_url
    )


# ── GET /matches ────────────────────────────────────────────────────────────


@router.get("", response_model=MatchBoardResponse)
def list_matches(
    min_books: int = 1,
    days_ahead: int = 14,
    tax_rate: float = TAX_RATE,
    stale_hours: float = 6,
    max_odds_age_hours: float = DEFAULT_MAX_ODDS_AGE_HOURS,
    bookmaker: str | None = None,
    db=Depends(get_db),
):
    now = datetime.now(UTC)
    max_dt = (now + timedelta(days=days_ahead)).isoformat(timespec="seconds")
    now_iso = now.isoformat(timespec="seconds")
    stale_cutoff = now - timedelta(hours=stale_hours)
    odds_cutoff = now - timedelta(hours=max_odds_age_hours)

    seen_cte = """
        WITH seen_matches AS (
            SELECT DISTINCT canonical_match_id
            FROM upcoming_matches
            WHERE canonical_match_id IS NOT NULL
              AND (last_seen_at IS NULL OR last_seen_at > :stale_cutoff)
        ),
        """
    seen_join = "JOIN seen_matches sm ON sm.canonical_match_id=cm.id"
    odds = query_df(
        db,
        f"""
        {seen_cte}
        latest AS (
            SELECT os.* FROM odds_snapshots os WHERE market_type='match_winner'
        )
        SELECT cm.id AS canonical_match_id,
               cm.team_a_name, cm.team_b_name,
               cm.normalized_team_a, cm.normalized_team_b,
               cm.start_time_normalized, cm.league, cm.best_of,
               b.name AS bookmaker, l.id AS odds_snapshot_id, l.bookmaker_id, l.is_live,
               l.raw_team_a, l.raw_team_b,
               l.odds_a, l.odds_b,
               l.scraped_at, l.source_url,
               (
                   SELECT offer_url FROM upcoming_matches
                   WHERE canonical_match_id=l.canonical_match_id AND bookmaker_id=l.bookmaker_id
                   LIMIT 1
               ) AS offer_url
        FROM latest l
        JOIN canonical_matches cm ON cm.id=l.canonical_match_id
        {seen_join}
        JOIN bookmakers b ON b.id=l.bookmaker_id
        WHERE cm.start_time_normalized IS NOT NULL
          AND cm.status = 'upcoming'
          AND REPLACE(cm.start_time_normalized, 'T', ' ') > REPLACE(:now, 'T', ' ')
          AND REPLACE(cm.start_time_normalized, 'T', ' ') <= REPLACE(:max_dt, 'T', ' ')
        ORDER BY cm.start_time_normalized, cm.id
        """,
        {"now": now_iso, "max_dt": max_dt, "stale_cutoff": stale_cutoff, "odds_cutoff": odds_cutoff},
    )

    if not odds:
        return MatchBoardResponse(total=0, matches=[])

    # Align authoritative participant names, not legacy normalized cache keys.
    for row in odds:
        aligned = _align(row, str(row.get("team_a_name") or ""), str(row.get("team_b_name") or ""))
        row["_odds_a"], row["_odds_b"] = aligned if aligned is not None else (None, None)

    # Group by match
    groups: dict[int, list[dict]] = {}
    for row in odds:
        groups.setdefault(row["canonical_match_id"], []).append(row)

    # Load predictions
    preds = query_df(
        db,
        """
        SELECT p.* FROM canonical_predictions p
        WHERE prediction_status='active'
          AND ((model_name=:hn AND model_version=:hv) OR (model_name=:sn AND model_version=:sv))
        """,
        get_active_prediction_db_params(),
    )
    contexts = load_prematch_quotes(list(groups), decision_at=now, session=db)
    pred_map = {mid: _serving_predictions(db, [p for p in preds if p["canonical_match_id"] == mid], context)
                for mid, context in contexts.items()}

    prop_counts: dict[int, int] = {}
    try:
        prop_rows = query_df(
            db,
            """
            SELECT canonical_match_id, COUNT(*) AS cnt
            FROM prop_odds_snapshots
            WHERE canonical_match_id IS NOT NULL
            GROUP BY canonical_match_id
            """,
        )
        for r in prop_rows:
            if r.get("canonical_match_id") is not None:
                prop_counts[int(r["canonical_match_id"])] = int(r["cnt"])
    except Exception:
        pass

    items: list[MatchBoardItem] = []
    for mid, group in groups.items():
        group = _latest_display_states(group, now)
        group = [
            g
            for g in group
            if g.get("is_live") == 0 and now-parse_aware_utc(g["scraped_at"]) <= timedelta(hours=max_odds_age_hours) and all(
                g.get(key) is not None and math.isfinite(g[key]) and g[key] > 1.0
                for key in ("_odds_a", "_odds_b")
            )
        ]
        if not group:
            continue
        books = len({g["bookmaker"] for g in group})
        context = contexts[mid]
        quotes = {quote.odds_snapshot_id: quote for quote in context.quotes}
        sources_unchanged = all(any(
            g["odds_snapshot_id"] == quote.odds_snapshot_id
            and _display_matches_quote(g, quote, (g["_odds_a"], g["_odds_b"]))
            for g in group) for quote in context.quotes)

        # Filter by specific bookmaker if requested
        if bookmaker:
            bookmaker_group = [g for g in group if g["bookmaker"] == bookmaker]
            if not bookmaker_group:
                continue
            group = bookmaker_group
            books = 1  # only one bookmaker in view

        if books < min_books:
            continue
        actionable = [g for g in group if sources_unchanged and _display_matches_quote(
            g, quotes.get(g["odds_snapshot_id"]), (g["_odds_a"], g["_odds_b"]))]
        action_a = max(actionable, key=lambda g: g["_odds_a"]) if actionable else None
        action_b = max(actionable, key=lambda g: g["_odds_b"]) if actionable else None

        best_a = max(group, key=lambda g: g["_odds_a"])
        best_b = max(group, key=lambda g: g["_odds_b"])
        avg_a = sum(g["_odds_a"] for g in actionable) / len(actionable) if actionable else None
        avg_b = sum(g["_odds_b"] for g in actionable) / len(actionable) if actionable else None

        record = {
            "best_odds_a": action_a["_odds_a"] if action_a else None,
            "best_bookmaker_a": action_a["bookmaker"] if action_a else None,
            "best_offer_url_a": action_a.get("offer_url") if action_a else None,
            "avg_odds_a": avg_a,
            "best_odds_b": action_b["_odds_b"] if action_b else None,
            "best_bookmaker_b": action_b["bookmaker"] if action_b else None,
            "best_offer_url_b": action_b.get("offer_url") if action_b else None,
            "avg_odds_b": avg_b,
        }
        action_record = dict(record, best_odds_a=action_a["_odds_a"] if action_a else None,
                             best_odds_b=action_b["_odds_b"] if action_b else None)
        enrich_arbitrage(action_record, tax_rate=tax_rate)
        record.update({key: value for key, value in action_record.items() if key.startswith("arb_")})

        team_a_name = group[0].get("team_a_name")
        team_b_name = group[0].get("team_b_name")
        team_a_golgg, team_a_conf, team_a_source = (
            suggest_mapping(str(team_a_name)) if team_a_name else (None, 0.0, None)
        )
        team_b_golgg, team_b_conf, team_b_source = (
            suggest_mapping(str(team_b_name)) if team_b_name else (None, 0.0, None)
        )
        has_unmapped_teams = not team_a_golgg or not team_b_golgg
        # Match confidence = min of both team mapping confidences (0-1 scale)
        match_confidence = (
            min(team_a_conf, team_b_conf) if team_a_golgg and team_b_golgg else 0.0
        )

        # Do not surface stale model probabilities/EV when either side is not
        # mapped to a GOL.GG team. Those predictions may have been computed
        # before mapping changed and would be misleading.
        p = {} if has_unmapped_teams else pred_map.get(mid, {})
        market_a = context.prob_a
        prob_a = p.get("hybrid_prob_a") if actionable else None
        prob_b = p.get("hybrid_prob_b") if actionable else None

        decisions = qualify_prediction_sides(
            prob_a=prob_a,
            prob_b=prob_b,
            diagnostics=p.get("hybrid_diagnostics") or p.get("model_diagnostics"),
            model_name=HYBRID_MODEL_NAME,
            odds_a=action_record["best_odds_a"],
            odds_b=action_record["best_odds_b"],
            market_prob_a=market_a,
            tax_rate=tax_rate,
            league=group[0].get("league"),
            match_date=group[0].get("start_time_normalized"),
            max_ev_net=None,
        )
        hybrid_ev_a = decisions["a"][2].get("ev_net")
        hybrid_ev_b = decisions["b"][2].get("ev_net")
        fusion_symaug_ev_a = (
            expected_value(
                float(p["fusion_symaug_prob_a"]), float(record["best_odds_a"]), tax_rate
            )
            if p.get("fusion_symaug_prob_a") is not None
            else None
        )
        fusion_symaug_ev_b = (
            expected_value(
                float(p["fusion_symaug_prob_b"]), float(record["best_odds_b"]), tax_rate
            )
            if p.get("fusion_symaug_prob_b") is not None
            else None
        )
        rec_side = None
        rec_team = None
        rec_bm = None
        rec_odds = None
        rec_ev = None
        if not has_unmapped_teams:
            val_a = decisions["a"][0]
            val_b = decisions["b"][0]
            if val_a or val_b:
                if val_a and (not val_b or hybrid_ev_a >= hybrid_ev_b):
                    rec_side = "a"
                    rec_team = team_a_name
                    rec_bm = action_a["bookmaker"]
                    rec_odds = action_a["_odds_a"]
                    rec_ev = hybrid_ev_a
                else:
                    rec_side = "b"
                    rec_team = team_b_name
                    rec_bm = action_b["bookmaker"]
                    rec_odds = action_b["_odds_b"]
                    rec_ev = hybrid_ev_b

        # Only a direct model prediction with persisted Venn-Abers bounds can
        # be marked conformal. Hybrid probabilities mix a market quote after
        # calibration and therefore have no retained coverage guarantee.
        bounds_a = conformal_bounds_for_side(p.get("model_diagnostics"), "a")
        bounds_b = conformal_bounds_for_side(p.get("model_diagnostics"), "b")
        p_low_a = bounds_a[0] if bounds_a is not None else None
        p_low_b = bounds_b[0] if bounds_b is not None else None
        conf_ev_a = (
            expected_value(p_low_a, action_a["_odds_a"], tax_rate)
            if p_low_a is not None and action_a is not None else None
        )
        conf_ev_b = (
            expected_value(p_low_b, action_b["_odds_b"], tax_rate)
            if p_low_b is not None and action_b is not None else None
        )
        is_conf_a = bool(
            conf_ev_a is not None
            and bounds_a is not None
            and bounds_a[1] - bounds_a[0] <= 0.08
            and conf_ev_a > 0.0
        )
        is_conf_b = bool(
            conf_ev_b is not None
            and bounds_b is not None
            and bounds_b[1] - bounds_b[0] <= 0.08
            and conf_ev_b > 0.0
        )
        # A board has no account balance. Sizing belongs to the financial
        # ledger, where available capital and overlapping reservations exist.
        stake_a = None
        stake_b = None
        items.append(MatchBoardItem(
            canonical_match_id=mid,
            match=f"{team_a_name or '?'} vs {team_b_name or '?'}",
            league=group[0].get("league"),
            start_time_normalized=group[0].get("start_time_normalized"),
            best_of=group[0].get("best_of"),
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_golgg_name=team_a_golgg,
            team_b_golgg_name=team_b_golgg,
            team_a_mapping_source=team_a_source,
            team_b_mapping_source=team_b_source,
            has_unmapped_teams=has_unmapped_teams,
            match_confidence=round(match_confidence, 4) if match_confidence < 1.0 else None,
            bookmaker_count=books,
            diagnostic_best_odds_a=best_a["_odds_a"],
            diagnostic_best_odds_b=best_b["_odds_b"],
            quote_status="actionable" if actionable else "diagnostic",
            best_odds_a=record["best_odds_a"],
            best_bookmaker_a=record["best_bookmaker_a"],
            best_offer_url_a=record["best_offer_url_a"],
            avg_odds_a=record["avg_odds_a"],
            best_odds_b=record["best_odds_b"],
            best_bookmaker_b=record["best_bookmaker_b"],
            best_offer_url_b=record["best_offer_url_b"],
            avg_odds_b=record["avg_odds_b"],
            arb_no_tax=record.get("arb_no_tax", False),
            arb_after_tax=record.get("arb_after_tax", False),
            arb_margin_no_tax=record.get("arb_margin_no_tax"),
            arb_margin_after_tax=record.get("arb_margin_after_tax"),
            model_prob_a=p.get("model_prob_a"),
            model_prob_b=p.get("model_prob_b"),
            hybrid_prob_a=prob_a,
            hybrid_prob_b=prob_b,
            hybrid_ev_a=hybrid_ev_a,
            hybrid_ev_b=hybrid_ev_b,
            fusion_symaug_prob_a=p.get("fusion_symaug_prob_a"),
            fusion_symaug_prob_b=p.get("fusion_symaug_prob_b"),
            fusion_symaug_ev_a=fusion_symaug_ev_a,
            fusion_symaug_ev_b=fusion_symaug_ev_b,
            conformal_prob_low_a=round(p_low_a, 4) if p_low_a is not None else None,
            conformal_prob_low_b=round(p_low_b, 4) if p_low_b is not None else None,
            conformal_ev_a=conf_ev_a,
            conformal_ev_b=conf_ev_b,
            is_conformal_value_a=is_conf_a,
            is_conformal_value_b=is_conf_b,
            recommended_stake_a=stake_a,
            recommended_stake_b=stake_b,
            recommended_side=rec_side,
            recommended_team=rec_team,
            recommended_bookmaker=rec_bm,
            recommended_odds=rec_odds,
            recommended_ev=rec_ev,
            has_props=mid in prop_counts,
            prop_lines_count=prop_counts.get(mid, 0),
            last_scraped_at=max(parse_aware_utc(g["scraped_at"]) for g in group).isoformat(),
        ))

    items.sort(key=lambda x: (x.start_time_normalized or "", x.canonical_match_id))
    return MatchBoardResponse(total=len(items), matches=items)


# ── GET /matches/results ────────────────────────────────────────────────────


@router.get("/results", response_model=MatchResultsResponse)
def list_results(
    days_back: int = 30,
    model_name: str = HYBRID_MODEL_NAME,
    model_version: str = HYBRID_MODEL_VERSION,
    odds_mode: str = "close",
    db=Depends(get_db),
):
    """Return finished matches with auditable EV results.

    Unlike the legacy implementation, this does not read precomputed
    model_ev_signals.  It recomputes EV from the requested model/version and a
    deterministic bookmaker quote selection: open/mid/close/latest.
    """
    odds_mode = odds_mode.lower().strip()
    if odds_mode not in {"open", "mid", "close", "latest"}:
        raise HTTPException(status_code=400, detail="odds_mode must be one of: open, mid, close, latest")

    now = datetime.now(UTC)
    min_dt = (now - timedelta(days=days_back)).isoformat(timespec="seconds")

    use_hybrid = model_name.startswith("Hybrid")
    contract = _historical_blend_contract(db, model_name, model_version) if use_hybrid else None
    prediction_model_name = contract["base_model_name"] if contract else model_name
    prediction_model_version = contract["base_model_version"] if contract else model_version

    match_rows = query_df(
        db,
        """
        SELECT cm.id AS canonical_match_id,
               cm.team_a_name, cm.team_b_name,
               cm.league, cm.start_time_normalized,
               cm.best_of, cm.status,
               cm.winner_name, cm.loser_name, cm.winner_side,
               cm.result_source, cm.result_recorded_at,
               gm.team1_score, gm.team2_score
        FROM canonical_matches cm
        LEFT JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        LEFT JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        WHERE cm.status = 'finished'
          AND REPLACE(cm.start_time_normalized, 'T', ' ') >= REPLACE(:min_dt, 'T', ' ')
        ORDER BY cm.start_time_normalized DESC, cm.id DESC
        """,
        {"min_dt": min_dt},
    )

    if not match_rows:
        return MatchResultsResponse(
            total=0,
            days_back=days_back,
            model_name=model_name,
            model_version=model_version,
            odds_mode=odds_mode,
            results=[],
        )

    match_ids = [int(r["canonical_match_id"]) for r in match_rows]
    odds_rows: list[dict[str, Any]] = []
    # Avoid relying on array parameter support across drivers.
    for start in range(0, len(match_ids), 500):
        chunk = match_ids[start : start + 500]
        placeholders = ",".join(f":id_{i}" for i in range(len(chunk)))
        params = {f"id_{i}": mid for i, mid in enumerate(chunk)}
        odds_rows.extend(query_df(
            db,
            f"""
            SELECT os.id AS odds_snapshot_id, os.canonical_match_id, os.bookmaker_id,
                   b.name AS bookmaker, os.raw_team_a, os.raw_team_b,
                   os.odds_a, os.odds_b, os.scraped_at, os.offer_url, os.is_live,
                   cm.league, cm.start_time_normalized
            FROM odds_snapshots os
            JOIN bookmakers b ON b.id = os.bookmaker_id
            JOIN canonical_matches cm ON cm.id = os.canonical_match_id
            WHERE os.canonical_match_id IN ({placeholders})
              AND os.market_type = 'match_winner'
            ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at
            """,
            params,
        ))

    odds_by_match_bookmaker: dict[tuple[int, int], list[dict[str, Any]]] = {}
    odds_by_match: dict[int, list[list[dict[str, Any]]]] = {}
    for row in odds_rows:
        key = (int(row["canonical_match_id"]), int(row["bookmaker_id"]))
        odds_by_match_bookmaker.setdefault(key, []).append(row)
    for (match_id, _bookmaker_id), snapshots in odds_by_match_bookmaker.items():
        odds_by_match.setdefault(match_id, []).append(snapshots)

    items: list[MatchResultItem] = []
    for r in match_rows:
        match_id = int(r["canonical_match_id"])
        start_at = parse_aware_utc(r.get("start_time_normalized"))
        base_rows = query_df(db, "SELECT * FROM canonical_predictions WHERE canonical_match_id=:id AND model_name=:name AND model_version=:version",
                            {"id": match_id, "name": prediction_model_name, "version": prediction_model_version})
        hybrid_rows = query_df(db, "SELECT * FROM canonical_predictions WHERE canonical_match_id=:id AND model_name=:name AND model_version=:version",
                              {"id": match_id, "name": model_name, "version": model_version}) if use_hybrid else []
        unavailable_reason = "blend_contract_missing_or_conflicting" if use_hybrid and contract is None else "pre_quote_base_unavailable"
        all_states = [row for group in odds_by_match.get(match_id, []) for row in group]

        bookmaker_ev_details: dict[str, Any] = {}
        bookmakers_with_ev: list[str] = []
        best_ev_a: float | None = None
        best_ev_b: float | None = None
        best_odds_a: float | None = None
        best_odds_b: float | None = None
        best_model_prob_a: float | None = None
        best_model_prob_b: float | None = None
        best_kelly_a: float | None = None
        best_kelly_b: float | None = None

        for snapshots in odds_by_match.get(match_id, []):
            selected = _pick_snapshot(snapshots, odds_mode, start_at=start_at) if start_at else None
            if selected is None or selected.get("is_live") != 0:
                continue
            aligned = _align(selected, str(r.get("team_a_name") or ""), str(r.get("team_b_name") or ""))
            if aligned is None:
                continue
            odds_a, odds_b = aligned
            odds_a = none_or_float(odds_a)
            odds_b = none_or_float(odds_b)
            if odds_a is None or odds_b is None or odds_a <= 1.0 or odds_b <= 1.0:
                continue
            origin = parse_aware_utc(selected.get("scraped_at"))
            market_prob_a = _historical_consensus(all_states, origin, str(r.get("team_a_name") or ""), str(r.get("team_b_name") or ""))
            market_prob_b = 1 - market_prob_a if market_prob_a is not None else None
            base = _historical_prediction(base_rows, origin=origin, start_at=start_at,
                                         model_name=prediction_model_name, model_version=prediction_model_version)
            lineage_ok = True
            historical_hybrid = _historical_prediction(hybrid_rows, origin=origin, start_at=start_at,
                                                       model_name=model_name, model_version=model_version)
            if historical_hybrid and contract:
                diag = _json_object(historical_hybrid.get("diagnostics_json"))
                for key in ("base_model_name", "base_model_version", "alpha", "temperature", "blending_mode"):
                    if key in diag and diag[key] != contract[key]:
                        lineage_ok = False
                if diag.get("base_prediction_id") is not None:
                    linked = next((p for p in base_rows if p["id"] == diag["base_prediction_id"]), None)
                    base = _historical_prediction([linked] if linked else [], origin=parse_aware_utc(historical_hybrid["predicted_at"]), start_at=start_at,
                                                  model_name=prediction_model_name, model_version=prediction_model_version)
                    lineage_ok = lineage_ok and base is not None
                if not lineage_ok:
                    unavailable_reason = "historical_base_lineage_conflicting"
            base_prob_a = _probability(base.get("prob_a")) if base else None
            model_prob_a = None
            if lineage_ok and base_prob_a is not None and (not use_hybrid or contract is not None):
                if use_hybrid and market_prob_a is not None:
                    model_prob_a = blend_market_probability(
                        base_prob_a, market_prob_a, alpha=contract["alpha"],
                        temperature=contract["temperature"], blending_mode=contract["blending_mode"])
                elif not use_hybrid:
                    model_prob_a = base_prob_a
            model_prob_b = 1-model_prob_a if model_prob_a is not None else None
            score_diagnostics = _raw_diagnostics(base.get("diagnostics_json")) if base else None
            if score_diagnostics is not None and use_hybrid and contract and market_prob_a is not None:
                score_diagnostics = dict(score_diagnostics)
                for side, market in (("a", market_prob_a), ("b", 1-market_prob_a)):
                    low = _probability(score_diagnostics.get(f"p_low_{side}"))
                    if low is not None:
                        score_diagnostics[f"p_low_{side}"] = blend_market_probability(
                            low, market, alpha=contract["alpha"],
                            temperature=contract["temperature"], blending_mode=contract["blending_mode"],
                        )
            decisions = qualify_prediction_sides(
                prob_a=model_prob_a, prob_b=model_prob_b,
                diagnostics=score_diagnostics, model_name=model_name,
                odds_a=odds_a, odds_b=odds_b, market_prob_a=market_prob_a,
                league=r.get("league"), match_date=r.get("start_time_normalized"),
                best_of=r.get("best_of"),
            )
            if model_prob_a is not None:
                unavailable_reason = None
            if best_ev_a is None and (best_odds_a is None or odds_a > best_odds_a):
                best_odds_a = odds_a
            if best_ev_b is None and (best_odds_b is None or odds_b > best_odds_b):
                best_odds_b = odds_b

            ev_a = decisions["a"][2].get("ev_net")
            ev_b = decisions["b"][2].get("ev_net")
            kelly_a = kelly_fraction(decisions["a"][2]["prob_for_ev"], odds_a, TAX_RATE) if decisions["a"][0] else None
            kelly_b = kelly_fraction(decisions["b"][2]["prob_for_ev"], odds_b, TAX_RATE) if decisions["b"][0] else None
            bk_name = str(selected.get("bookmaker") or "?")
            if (ev_a is not None and ev_a > 0) or (ev_b is not None and ev_b > 0):
                bookmakers_with_ev.append(bk_name)
            if ev_a is not None and (best_ev_a is None or ev_a > best_ev_a):
                best_ev_a = ev_a
                best_odds_a = odds_a
                best_model_prob_a = model_prob_a
                best_kelly_a = kelly_a
            if ev_b is not None and (best_ev_b is None or ev_b > best_ev_b):
                best_ev_b = ev_b
                best_odds_b = odds_b
                best_model_prob_b = model_prob_b
                best_kelly_b = kelly_b
            bookmaker_ev_details[bk_name] = {
                "side_a": {
                    "ev": ev_a,
                    "odds": odds_a,
                    "model_prob": model_prob_a,
                    "market_prob": market_prob_a,
                    "kelly": kelly_a,
                    "odds_snapshot_id": int(selected["odds_snapshot_id"]),
                    "scraped_at": origin.isoformat(),
                },
                "side_b": {
                    "ev": ev_b,
                    "odds": odds_b,
                    "model_prob": model_prob_b,
                    "market_prob": market_prob_b,
                    "kelly": kelly_b,
                    "odds_snapshot_id": int(selected["odds_snapshot_id"]),
                    "scraped_at": origin.isoformat(),
                },
            }

        items.append(MatchResultItem(
            canonical_match_id=r["canonical_match_id"],
            team_a_name=r.get("team_a_name"),
            team_b_name=r.get("team_b_name"),
            league=r.get("league"),
            start_time_normalized=r.get("start_time_normalized"),
            best_of=r.get("best_of"),
            status=r.get("status"),
            winner_name=r.get("winner_name"),
            loser_name=r.get("loser_name"),
            winner_side=r.get("winner_side"),
            team_a_score=r.get("team1_score"),
            team_b_score=r.get("team2_score"),
            result_source=r.get("result_source"),
            result_recorded_at=str(r["result_recorded_at"]) if r.get("result_recorded_at") else None,
            best_ev_a=best_ev_a,
            best_ev_b=best_ev_b,
            best_odds_a=best_odds_a,
            best_odds_b=best_odds_b,
            model_prob_a=best_model_prob_a,
            model_prob_b=best_model_prob_b,
            kelly_a=best_kelly_a,
            kelly_b=best_kelly_b,
            bookmakers_with_ev=sorted(set(bookmakers_with_ev)),
            bookmaker_ev_details=bookmaker_ev_details,
            model_name=model_name,
            model_version=model_version,
            odds_mode=odds_mode,
            probability_basis="historical_quote_recomputation",
            probability_unavailable_reason=unavailable_reason if best_model_prob_a is None and best_model_prob_b is None else None,
        ))

    return MatchResultsResponse(
        total=len(items),
        days_back=days_back,
        model_name=model_name,
        model_version=model_version,
        odds_mode=odds_mode,
        results=items,
    )


# ── GET /matches/unmapped ───────────────────────────────────────────────────


@router.get("/unmapped", response_model=UnmappedMatchesResponse)
def list_unmapped_matches(
    status: str = "expired",
    limit: int = 100,
    db=Depends(get_db),
):
    """Return matches that are not yet mapped to GOL.GG."""
    rows = query_df(
        db,
        """
        SELECT cm.id AS canonical_match_id,
               cm.team_a_name, cm.team_b_name,
               cm.league, cm.start_time_normalized,
               cm.status,
               (
                   SELECT array_agg(DISTINCT b.name)
                   FROM odds_snapshots os
                   JOIN bookmakers b ON b.id = os.bookmaker_id
                   WHERE os.canonical_match_id = cm.id
               ) as bookmakers
        FROM canonical_matches cm
        LEFT JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        WHERE gmm.canonical_match_id IS NULL
          AND cm.status = :status
        ORDER BY cm.start_time_normalized DESC
        LIMIT :limit
        """,
        {"status": status, "limit": limit},
    )

    items = []
    for r in rows:
        # PostgreSQL array_agg returns a string like "{sts,betclic}" or a list
        bks = r.get("bookmakers")
        if isinstance(bks, str):
            bks = [b.strip() for b in bks.strip("{}").split(",") if b.strip()]
        elif bks is None:
            bks = []

        mapping_context = {
            "source_system": "bookmaker",
            "league": r.get("league"),
            "match_date": r.get("start_time_normalized"),
        }
        team_a_golgg, team_a_confidence, team_a_source = suggest_mapping(
            r["team_a_name"],
            **mapping_context,
        )
        team_b_golgg, team_b_confidence, team_b_source = suggest_mapping(
            r["team_b_name"],
            **mapping_context,
        )
        
        items.append(UnmappedMatchItem(
            canonical_match_id=r["canonical_match_id"],
            team_a_name=r["team_a_name"],
            team_b_name=r["team_b_name"],
            team_a_mapping={
                "canonical_name": r["team_a_name"],
                "golgg_name": team_a_golgg,
                "confidence": team_a_confidence,
                "source": team_a_source,
            },
            team_b_mapping={
                "canonical_name": r["team_b_name"],
                "golgg_name": team_b_golgg,
                "confidence": team_b_confidence,
                "source": team_b_source,
            },
            league=r.get("league"),
            start_time_normalized=r.get("start_time_normalized"),
            status=r["status"],
            bookmakers=bks
        ))
    
    return UnmappedMatchesResponse(total=len(items), matches=items)


# ── GET/POST /matches/mapping-review ───────────────────────────────────────


@router.get("/mapping-review", response_model=MappingReviewResponse)
def list_mapping_review_items(limit: int = 100, db=Depends(get_db)):
    """Return existing result links that fail the conservative identity gate."""
    rows = query_df(
        db,
        """
        SELECT gmm.id AS mapping_id, gmm.canonical_match_id, gmm.golgg_match_id,
               gmm.confidence, gmm.mapped_by,
               cm.team_a_name AS canonical_team_a,
               cm.team_b_name AS canonical_team_b,
               cm.start_time_normalized, cm.league,
               gm.team1_name AS golgg_team_a, gm.team2_name AS golgg_team_b,
               gm.date AS golgg_date, gm.tournament_name AS golgg_competition,
               (SELECT COUNT(*) FROM canonical_predictions p
                WHERE p.canonical_match_id = cm.id) AS prediction_count,
               (SELECT COUNT(*) FROM upcoming_match_features f
                WHERE f.canonical_match_id = cm.id) AS feature_count,
               (SELECT COUNT(*) FROM model_ev_signals s
                WHERE s.canonical_match_id = cm.id) AS signal_count,
               (SELECT COUNT(*) FROM bets b
                WHERE b.canonical_match_id = cm.id) AS bet_count
        FROM golgg_match_mappings gmm
        JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
        JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        ORDER BY cm.start_time_normalized DESC, cm.id DESC
        """,
    )
    items: list[MappingReviewItem] = []
    for row in rows:
        canonical_date = str(row.get("start_time_normalized") or "")[:10] or None
        golgg_date = str(row.get("golgg_date") or "")[:10] or None
        canonical_family = competition_family(row.get("league"))
        golgg_family = competition_family(row.get("golgg_competition"))
        confidence = float(row.get("confidence") or 0.0)
        reasons: list[str] = []
        if confidence < 0.95:
            reasons.append("confidence_below_0_95")
        if not canonical_date or not golgg_date:
            reasons.append("missing_date")
        elif canonical_date != golgg_date:
            reasons.append("date_mismatch")
        if canonical_family is None or golgg_family is None:
            reasons.append("unknown_competition_family")
        elif canonical_family != golgg_family:
            reasons.append("competition_conflict")
        if not reasons:
            continue
        items.append(
            MappingReviewItem(
                canonical_match_id=int(row["canonical_match_id"]),
                mapping_id=int(row["mapping_id"]),
                golgg_match_id=str(row["golgg_match_id"]),
                confidence=confidence,
                mapped_by=row.get("mapped_by"),
                canonical_team_a=row["canonical_team_a"],
                canonical_team_b=row["canonical_team_b"],
                canonical_date=canonical_date,
                canonical_competition=row.get("league"),
                golgg_team_a=row.get("golgg_team_a"),
                golgg_team_b=row.get("golgg_team_b"),
                golgg_date=golgg_date,
                golgg_competition=row.get("golgg_competition"),
                reasons=reasons,
                prediction_count=int(row.get("prediction_count") or 0),
                feature_count=int(row.get("feature_count") or 0),
                signal_count=int(row.get("signal_count") or 0),
                bet_count=int(row.get("bet_count") or 0),
            )
        )
        if len(items) >= limit:
            break
    return MappingReviewResponse(total=len(items), items=items)


@router.post("/mapping-review/decision", response_model=MappingReviewDecisionResponse)
def decide_mapping_review(
    body: MappingReviewDecisionRequest,
    review_token: str | None = Header(default=None, alias="X-Identity-Review-Token"),
    db=Depends(get_db),
):
    """Apply one operator-reviewed result-link decision atomically."""
    expected_token = _identity_review_token()
    if not expected_token:
        raise HTTPException(status_code=503, detail="Identity review mutations are disabled")
    if review_token is None or not secrets.compare_digest(review_token, expected_token):
        raise HTTPException(status_code=401, detail="Invalid identity review token")
    lock_suffix = " FOR UPDATE OF cm"
    current = db.execute(
        text(
            """
            SELECT gmm.id AS mapping_id, gmm.golgg_match_id,
                   cm.team_a_name, cm.team_b_name, cm.start_time_normalized,
                   cm.league,
                   (SELECT COUNT(*) FROM bets b
                    WHERE b.canonical_match_id = cm.id) AS bet_count
            FROM canonical_matches cm
            LEFT JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
            WHERE cm.id = :canonical_match_id
            """
            + lock_suffix
        ),
        {"canonical_match_id": body.canonical_match_id},
    ).mappings().first()
    if current is None:
        raise HTTPException(status_code=404, detail="Canonical match not found")
    old_id = str(current["golgg_match_id"]) if current["golgg_match_id"] is not None else None
    new_id = body.new_golgg_match_id.strip() if body.new_golgg_match_id else None
    if body.decision == "replace" and not new_id:
        raise HTTPException(status_code=400, detail="replace requires new_golgg_match_id")
    if body.decision != "replace" and new_id:
        raise HTTPException(status_code=400, detail="new_golgg_match_id is only valid for replace")
    if body.decision in {"replace", "invalidate"} and int(current["bet_count"] or 0) > 0:
        raise HTTPException(
            status_code=409,
            detail="Mappings with recorded bets require a separate settlement review",
        )
    if body.decision == "replace":
        exists = db.execute(
            text("SELECT 1 FROM golgg_matches WHERE match_id = :match_id"),
            {"match_id": new_id},
        ).first()
        if exists is None:
            raise HTTPException(status_code=404, detail="Replacement GOL.GG match not found")

    evidence = json.dumps(dict(current), default=str, sort_keys=True)
    try:
        if body.decision == "retain":
            if old_id is None:
                raise HTTPException(status_code=409, detail="No mapping exists to retain")
            db.execute(
                text(
                    """
                    UPDATE golgg_match_mappings
                    SET confidence = 1.0, mapped_by = 'manual-reviewed-v1',
                        mapped_at = CURRENT_TIMESTAMP
                    WHERE canonical_match_id = :canonical_match_id
                    """
                ),
                {"canonical_match_id": body.canonical_match_id},
            )
            new_id = old_id
        elif body.decision == "replace":
            if old_id is None:
                db.execute(
                    text(
                        """
                        INSERT INTO golgg_match_mappings (
                            canonical_match_id, golgg_match_id, confidence,
                            mapped_by, mapped_at
                        ) VALUES (
                            :canonical_match_id, :golgg_match_id, 1.0,
                            'manual-reviewed-v1', CURRENT_TIMESTAMP
                        )
                        """
                    ),
                    {"canonical_match_id": body.canonical_match_id, "golgg_match_id": new_id},
                )
            else:
                db.execute(
                    text(
                        """
                        UPDATE golgg_match_mappings
                        SET golgg_match_id = :golgg_match_id, confidence = 1.0,
                            mapped_by = 'manual-reviewed-v1',
                            mapped_at = CURRENT_TIMESTAMP
                        WHERE canonical_match_id = :canonical_match_id
                        """
                    ),
                    {"canonical_match_id": body.canonical_match_id, "golgg_match_id": new_id},
                )
        else:
            db.execute(
                text("DELETE FROM golgg_match_mappings WHERE canonical_match_id = :canonical_match_id"),
                {"canonical_match_id": body.canonical_match_id},
            )
            new_id = None

        if body.decision in {"replace", "invalidate"}:
            db.execute(
                text(
                    """
                    UPDATE canonical_matches
                    SET status = 'expired', winner_name = NULL, loser_name = NULL,
                        winner_normalized = NULL, winner_side = NULL,
                        result_source = NULL, result_source_match_id = NULL,
                        result_recorded_at = NULL
                    WHERE id = :canonical_match_id
                    """
                ),
                {"canonical_match_id": body.canonical_match_id},
            )
        decision_id = db.execute(
            text(
                """
                INSERT INTO mapping_review_decisions (
                    canonical_match_id, old_golgg_match_id, new_golgg_match_id,
                    decision, reason, operator, evidence_json, decided_at
                ) VALUES (
                    :canonical_match_id, :old_golgg_match_id, :new_golgg_match_id,
                    :decision, :reason, :operator, :evidence_json, CURRENT_TIMESTAMP
                )
                RETURNING id
                """
            ),
            {
                "canonical_match_id": body.canonical_match_id,
                "old_golgg_match_id": old_id,
                "new_golgg_match_id": new_id,
                "decision": body.decision,
                "reason": body.reason.strip(),
                "operator": body.operator.strip(),
                "evidence_json": evidence,
            },
        ).scalar_one()
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Replacement mapping conflicts with an existing reviewed link",
        ) from exc
    return MappingReviewDecisionResponse(
        decision_id=int(decision_id),
        canonical_match_id=body.canonical_match_id,
        decision=body.decision,
        old_golgg_match_id=old_id,
        new_golgg_match_id=new_id,
    )


# ── GET /matches/{id}/mapping-candidates ────────────────────────────────────


@router.get("/mapping-check/{golgg_id}", response_model=MappingCheckResponse)
def check_golgg_mapping(golgg_id: str, db=Depends(get_db)):
    """Check if a GOL.GG match ID is already mapped to any canonical match."""
    row = query_one(
        db,
        """
        SELECT cm.id, cm.team_a_name, cm.team_b_name, cm.start_time_normalized
        FROM golgg_match_mappings gmm
        JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
        WHERE gmm.golgg_match_id = :g_id
        """,
        {"g_id": golgg_id},
    )
    if row:
        return MappingCheckResponse(
            is_mapped=True,
            canonical_match_id=row["id"],
            team_a=row["team_a_name"],
            team_b=row["team_b_name"],
            start_time=row["start_time_normalized"],
        )
    return MappingCheckResponse(is_mapped=False)


@router.get("/{match_id}/mapping-candidates", response_model=GolggMatchCandidatesResponse)
def list_mapping_candidates(
    match_id: int,
    days_window: int = 3,
    db=Depends(get_db),
):
    """Suggest GOL.GG matches for a specific canonical match based on date and names."""
    meta = query_one(db, "SELECT start_time_normalized, team_a_name, team_b_name FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")

    start_time = meta["start_time_normalized"]
    if not start_time:
        return GolggMatchCandidatesResponse(candidates=[])

    # Search window
    dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    min_date = (dt - timedelta(days=days_window)).date().isoformat()
    max_date = (dt + timedelta(days=days_window)).date().isoformat()

    # Get all GOL.GG matches in window
    candidates = query_df(
        db,
        """
        SELECT match_id, team1_name, team2_name, date, team1_win, team2_win
        FROM golgg_matches
        WHERE date >= :min_date AND date <= :max_date
        ORDER BY date DESC
        """,
        {"min_date": min_date, "max_date": max_date},
    )

    # Simple scoring/filtering could be added here, but for now return all in window
    items = [GolggMatchCandidate(**c) for c in candidates]
    return GolggMatchCandidatesResponse(candidates=items)


# ── POST /matches/map ───────────────────────────────────────────────────────


@router.post("/map")
def map_match_manually(body: MatchMappingRequest, db=Depends(get_db)):
    """Create a manual mapping between a canonical match and a GOL.GG match."""
    # 1. Fetch GOL.GG match info to sync best_of
    golgg_match = query_one(
        db,
        "SELECT best_of FROM golgg_matches WHERE match_id = :g_id",
        {"g_id": body.golgg_match_id}
    )
    best_of = golgg_match.get("best_of") if golgg_match else None

    # 2. Insert mapping
    db.execute(
        text("""
            INSERT INTO golgg_match_mappings (canonical_match_id, golgg_match_id)
            VALUES (:c_id, :g_id)
            ON CONFLICT (canonical_match_id) DO UPDATE SET golgg_match_id = EXCLUDED.golgg_match_id
        """),
        {"c_id": body.canonical_match_id, "g_id": body.golgg_match_id},
    )

    # 3. Update status and best_of
    db.execute(
        text("""
            UPDATE canonical_matches 
            SET status = 'finished',
                best_of = COALESCE(:best_of, best_of)
            WHERE id = :id AND status IN ('expired', 'upcoming')
        """),
        {"id": body.canonical_match_id, "best_of": best_of},
    )

    db.commit()
    return {"ok": True}


# ── POST /matches/alias — create team alias mapping ──────────────────────────


@router.post("/alias", response_model=AliasCreateResponse)
def create_alias(body: AliasCreateRequest, db=Depends(get_db)):
    """Create a context-scoped manual team alias."""
    from betting_app.services.mapping_service import upsert_alias, normalize_team_name
    from betting_app.services.team_alias_service import is_short_alias

    normalized = normalize_team_name(body.raw_name)
    if not normalized:
        raise HTTPException(status_code=400, detail="Normalized name is empty")
    if is_short_alias(body.raw_name) and not (
        body.league_pattern or body.valid_from or body.valid_to
    ):
        raise HTTPException(
            status_code=400,
            detail="Short aliases require a league or validity scope",
        )

    scoped = bool(
        body.source_system
        or body.league_pattern
        or body.valid_from
        or body.valid_to
    )
    alias_id = upsert_alias(
        body.raw_name,
        body.golgg_team_name,
        source="manual",
        confirmed=True,
        source_system=body.source_system,
        league_pattern=body.league_pattern,
        valid_from=body.valid_from,
        valid_to=body.valid_to,
        notes="Created through the operator mapping workflow",
    )

    return AliasCreateResponse(
        id=alias_id,
        normalized_name=normalized,
        alias=body.golgg_team_name,
        source="manual-scoped" if scoped else "manual",
    )


# ── GET /matches/golgg-teams — list/search GolGG teams ──────────────────────


@router.get("/golgg-teams", response_model=GolggTeamsResponse)
def list_golgg_teams(q: str = "", limit: int = 50, db=Depends(get_db)):
    """Return GolGG team names, optionally filtered by query string."""
    from betting_app.services.mapping_service import load_golgg_team_candidates

    all_teams = load_golgg_team_candidates()
    if q:
        q_lower = q.lower()
        filtered = [t for t in all_teams if q_lower in t.lower()]
    else:
        filtered = all_teams

    return GolggTeamsResponse(teams=filtered[:limit])


# ── DELETE /matches/alias — remove team alias mapping ───────────────────────


@router.delete("/alias")
def delete_alias_endpoint(body: AliasDeleteRequest, db=Depends(get_db)):
    """Delete a manual team alias mapping for the given raw_name."""
    from betting_app.services.mapping_service import delete_alias

    deleted = delete_alias(body.raw_name, source="manual")
    if not deleted:
        raise HTTPException(status_code=404, detail="No manual alias found for that team")
    return {"ok": True, "deleted": True}

# ── GET /matches/rosters/status & POST /matches/rosters/verify ───────────────


@router.get("/rosters/status", response_model=RosterStatusResponse)
def get_rosters_status(db=Depends(get_db)):
    """Return the verification status of rosters for all upcoming matches."""
    from betting_app.core.matching import normalize_team_name

    try:
        matches = query_df(
            db,
            """
            SELECT id, team_a_name, team_b_name
            FROM canonical_matches
            WHERE status = 'upcoming'
            ORDER BY start_time_normalized ASC
            """,
        )
    except Exception:
        matches = []
    if not matches:
        return RosterStatusResponse(
            upcoming_matches_count=0,
            total_teams=0,
            teams_with_roster=0,
            teams_missing_roster=0,
            source_counts={},
            teams_missing=[],
            teams_verified=[],
        )

    unique_teams = sorted({
        name for col in ("team_a_name", "team_b_name")
        for name in [m.get(col) for m in matches]
        if name and str(name).strip()
    })

    try:
        roster_rows = query_df(
            db,
            """
            SELECT normalized_team_name, team_name, role, source, updated_at
            FROM team_current_roster_players
            """,
        )
    except Exception:
        roster_rows = []

    by_norm_team: dict[str, list[dict]] = defaultdict(list)
    for r in roster_rows:
        by_norm_team[str(r.get("normalized_team_name") or "")].append(r)

    teams_with_roster = 0
    teams_missing: list[str] = []
    source_counts: dict[str, int] = defaultdict(int)
    teams_verified: list[dict[str, Any]] = []

    for t in unique_teams:
        norm = normalize_team_name(t)
        p_list = by_norm_team.get(norm, [])
        roles = {str(p.get("role") or "").upper() for p in p_list}
        if len(roles.intersection({"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"})) == 5:
            teams_with_roster += 1
            first = p_list[0]
            src = str(first.get("source") or "auto")
            source_counts[src] += 1
            teams_verified.append({
                "team_name": t,
                "normalized_team_name": norm,
                "source": src,
                "roles_count": len(roles),
                "updated_at": str(first.get("updated_at") or ""),
            })
        else:
            teams_missing.append(t)

    return RosterStatusResponse(
        upcoming_matches_count=len(matches),
        total_teams=len(unique_teams),
        teams_with_roster=teams_with_roster,
        teams_missing_roster=len(teams_missing),
        source_counts=dict(source_counts),
        teams_missing=teams_missing,
        teams_verified=teams_verified,
    )


@router.post("/rosters/verify")
def verify_rosters_endpoint(body: RosterVerificationRequest = RosterVerificationRequest(), db=Depends(get_db)):
    """Trigger online roster verification and sync against LoL Fandom & Liquipedia."""
    service = RosterVerificationService()
    summary = service.verify_and_sync_rosters(
        team_names=body.team_names,
        session=db,
        source=body.source,
        force=body.force,
        dry_run=body.dry_run,
        limit=body.limit,
    )
    return summary

# ── GET /matches/active-teams — list current active teams with rosters ────────


@router.get("/active-teams")
def get_active_teams(db=Depends(get_db)):
    """Return teams with a known GOL.GG team row and five exact current players."""
    from betting_app.services.rating_contract import OPERATIONAL_RATINGS_VERSION

    teams = query_df(
        db,
        "SELECT id, team_id, team_name FROM golgg_teams ORDER BY team_name",
    )
    roster_rows = query_df(
        db,
        """
        SELECT team_id, team_name, player_id
        FROM team_current_roster_players
        WHERE player_id IS NOT NULL AND TRIM(player_id) <> ''
        """,
    )
    rating_rows = query_df(
        db,
        """
        SELECT entity_name, rating_value, games_played, last_match_at
        FROM entity_ratings
        WHERE ratings_version = :version AND entity_type = 'team' AND rating_system = 'gl'
        """,
        {"version": OPERATIONAL_RATINGS_VERSION},
    )
    ratings_by_name = {
        str(row.get("entity_name") or "").strip().casefold(): row
        for row in rating_rows if row.get("entity_name")
    }
    ready_teams = []
    for team in teams:
        team_name = str(team.get("team_name") or "").strip()
        external_id = str(team.get("team_id") or "").strip()
        players = [
            row for row in roster_rows
            if (
                str(row.get("team_id") or "").strip() == external_id and external_id
            ) or (
                str(row.get("team_name") or "").strip().casefold() == team_name.casefold()
            )
        ]
        player_ids = {
            str(row.get("player_id") or "").strip()
            for row in players if str(row.get("player_id") or "").strip()
        }
        if len(players) != 5 or len(player_ids) != 5 or not team.get("id") or not team_name:
            continue
        rating = ratings_by_name.get(team_name.casefold())
        ready_teams.append(
            {
                "team_row_id": int(team["id"]),
                "native_team_id": native_golgg_team_id(int(team["id"]), team_name=team_name),
                "name": team_name,
                "rating": none_or_float(rating.get("rating_value")) if rating else None,
                "games": int(rating.get("games_played") or 0) if rating else None,
                "last_active": str(rating.get("last_match_at") or "")[:10] if rating else None,
            }
        )
    ready_teams.sort(
        key=lambda row: (
            row["rating"] is None,
            -(row["rating"] or 0.0),
            row["name"].casefold(),
        )
    )
    return {"teams": ready_teams}


# ── POST /matches/matchup — custom team vs team simulation ──────────────────


@router.post("/matchup", response_model=MatchupSimulationResponse)
def simulate_matchup(body: MatchupSimulationRequest, db=Depends(get_db)):
    """Simulate a hypothetical head-to-head matchup between any two teams."""
    from betting_app.services.upcoming_inference_service import (
        DEFAULT_FEATURE_VERSION,
        DEFAULT_RATINGS_VERSION,
        DEFAULT_W20_VERSION,
        _native_c0_active,
        _normalized_best_of,
        build_features_for_match,
    )

    best_of = _normalized_best_of(body.best_of)
    resolved_native_ids = []
    for side, team_name, row_id, supplied_native_id in (
        ("a", body.team_a_name, body.team_a_team_row_id, body.native_team_a_id),
        ("b", body.team_b_name, body.team_b_team_row_id, body.native_team_b_id),
    ):
        native_id = supplied_native_id
        if row_id is not None:
            try:
                resolved_id = native_golgg_team_id(row_id, team_name=team_name)
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
            if native_id is not None and resolved_id is not None and str(native_id) != resolved_id:
                raise HTTPException(
                    status_code=422,
                    detail=f"Explicit native team ID for side {side} conflicts with its database team row.",
                )
            native_id = native_id if native_id is not None else resolved_id
        resolved_native_ids.append(native_id)
    decision_clock = datetime.now(UTC)
    synthetic_match = {
        "id": 0,
        "team_a_name": body.team_a_name,
        "team_b_name": body.team_b_name,
        "team_a_golgg_id": body.team_a_team_row_id,
        "team_b_golgg_id": body.team_b_team_row_id,
        "native_team_a_id": resolved_native_ids[0],
        "native_team_b_id": resolved_native_ids[1],
        "league": body.league or "Custom Matchup",
        "_c0_decision_at": decision_clock.isoformat(),
        "start_time_normalized": (decision_clock + timedelta(hours=1)).isoformat(),
        "best_of": best_of,
    }

    feature_result = build_features_for_match(
        synthetic_match,
        feature_version=DEFAULT_FEATURE_VERSION,
        ratings_version=DEFAULT_RATINGS_VERSION,
        w20_version=DEFAULT_W20_VERSION,
        min_mapping_confidence=0.50,
        team_a_roster_override=body.team_a_roster_override,
        team_b_roster_override=body.team_b_roster_override,
        persist=False,
    )
    if _native_c0_active(get_active_model()) and feature_result.get("status") != "ready_player":
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Native C0 matchup requires exact team IDs and five unique player IDs per side.",
                "missing": feature_result.get("missing", []),
            },
        )

    features = feature_result.get("features") or {}
    if "canonical" not in features:
        features["canonical"] = synthetic_match
    elif features["canonical"].get("best_of") is None:
        features["canonical"]["best_of"] = best_of
    pred_res = PredictionEngine.predict_from_features(features)
    series_prob_a = pred_res.prob_a
    map_prob_a = pred_res.map_prob_a
    map_prob_b = pred_res.map_prob_b
    c0_request = features.get("c0_request") or {}
    components = {
        "model_name": pred_res.model_name,
        "model_version": pred_res.model_version,
        "feature_version": pred_res.feature_version,
        "best_of": pred_res.best_of,
        "market_features_used": False,
        "side_symmetric": True,
        "epistemic_sigma_z": pred_res.epistemic_sigma_z,
        "p_low_a": pred_res.p_low_a,
        "p_low_b": pred_res.p_low_b,
        "c0_input_mode": c0_request.get("mode"),
        "c0_organization_identity_missing": c0_request.get("mode") == "no_organization",
        **pred_res.diagnostics,
    }
    # Extract rosters and comparison info
    team_comparison = None
    recent_stats_a = None
    recent_stats_b = None
    roster_a_info = None
    roster_b_info = None

    ratings = features.get("ratings")
    mapping = features.get("mapping")
    if isinstance(ratings, dict):
        team_a_info = TeamMappingInfo(
            canonical_name=body.team_a_name,
            golgg_name=mapping.get("team_a_golgg_name") if isinstance(mapping, dict) else None,
            confidence=_finite_float(mapping.get("team_a_confidence")) if isinstance(mapping, dict) else None,
            source=mapping.get("team_a_source") if isinstance(mapping, dict) else None,
        )
        team_b_info = TeamMappingInfo(
            canonical_name=body.team_b_name,
            golgg_name=mapping.get("team_b_golgg_name") if isinstance(mapping, dict) else None,
            confidence=_finite_float(mapping.get("team_b_confidence")) if isinstance(mapping, dict) else None,
            source=mapping.get("team_b_source") if isinstance(mapping, dict) else None,
        )
        team_a_ratings = ratings.get("team_a") or {}
        team_b_ratings = ratings.get("team_b") or {}
        team_comparison = TeamComparisonInfo(
            team_a=team_a_info,
            team_b=team_b_info,
            team_a_rating=_finite_float(team_a_ratings.get("gl", {}).get("rating_value")),
            team_b_rating=_finite_float(team_b_ratings.get("gl", {}).get("rating_value")),
            rating_system="Glicko",
            team_a_elo=_finite_float(team_a_ratings.get("elo", {}).get("rating_value")),
            team_b_elo=_finite_float(team_b_ratings.get("elo", {}).get("rating_value")),
            team_a_glicko=_finite_float(team_a_ratings.get("gl", {}).get("rating_value")),
            team_b_glicko=_finite_float(team_b_ratings.get("gl", {}).get("rating_value")),
            team_a_glicko_rd=_finite_float(team_a_ratings.get("gl", {}).get("rd")),
            team_b_glicko_rd=_finite_float(team_b_ratings.get("gl", {}).get("rd")),
            rating_probabilities=features.get("ratings", {}).get("probabilities"),
        )

    w20 = features.get("w20")
    if isinstance(w20, dict):
        def make_stats(raw: Any) -> TeamRecentStats | None:
            if not isinstance(raw, dict):
                return None
            return TeamRecentStats(
                team_name=raw.get("team_name"),
                matches_count=int(raw.get("matches_count", 0)) if raw.get("matches_count") is not None else None,
                games_count=int(raw.get("games_count", 0)) if raw.get("games_count") is not None else None,
                win_rate=_finite_float(raw.get("win_rate")),
                avg_kills=_finite_float(raw.get("avg_kills")),
                avg_deaths=_finite_float(raw.get("avg_deaths")),
                avg_gd15=_finite_float(raw.get("avg_gd15")),
                avg_dragons=_finite_float(raw.get("avg_dragons")),
                avg_nashors=_finite_float(raw.get("avg_nashors")),
                avg_towers=_finite_float(raw.get("avg_towers")),
                avg_game_duration=_finite_float(raw.get("avg_game_duration")),
                last_match_at=raw.get("last_match_at"),
            )
        recent_stats_a = make_stats(w20.get("team_a"))
        recent_stats_b = make_stats(w20.get("team_b"))

    player_ratings = features.get("player_ratings")
    if isinstance(player_ratings, dict):
        def build_roster(side_key: str, name: str) -> RosterInfo:
            roster = player_ratings.get(f"{side_key}_roster") or {}
            roster_players = roster.get("players") or []
            side_dict = player_ratings.get(side_key) or {}
            gl_dict = side_dict.get("gl") or {}
            ratings_by_player_id = {
                str(player.get("player_id") or player.get("normalized_entity_name")): player
                for player in gl_dict.get("players") or []
                if player.get("player_id") or player.get("normalized_entity_name")
            }
            raw_players = roster_players or gl_dict.get("players") or []
            p_objs = []
            for player in raw_players:
                player_id = str(player.get("player_id") or player.get("normalized_entity_name") or "")
                rating = ratings_by_player_id.get(player_id, {})
                rating_value = rating.get("rating_value")
                if rating_value is None:
                    rating_value = player.get("rating_value")
                rating_rd = rating.get("rd")
                if rating_rd is None:
                    rating_rd = player.get("rd")
                games_played = rating.get("games_played")
                if games_played is None:
                    games_played = player.get("games_played")
                p_objs.append(
                    RosterPlayer(
                        player_id=player_id,
                        player_name=player.get("player_name")
                        or player.get("entity_name")
                        or rating.get("player_name")
                        or rating.get("entity_name"),
                        role=player.get("role") or rating.get("role"),
                        glicko_rating=_finite_float(rating_value),
                        glicko_rd=_finite_float(rating_rd),
                        games_played=int(games_played) if games_played is not None else None,
                    )
                )
            return RosterInfo(
                team_name=roster.get("team_name") or name,
                source_match_id=roster.get("source_match_id"),
                source_date=roster.get("source_match_date"),
                source_tournament=roster.get("source_tournament"),
                roster_source=roster.get("source"),
                avg_glicko=_finite_float(gl_dict.get("avg_rating_value")),
                avg_glicko_rd=_finite_float(gl_dict.get("avg_rd")),
                players_with_rating=(
                    int(gl_dict["players_with_rating"])
                    if gl_dict.get("players_with_rating") is not None
                    else None
                ),
                players=p_objs,
            )

        roster_a_info = build_roster("team_a", body.team_a_name)
        roster_b_info = build_roster("team_b", body.team_b_name)

    return MatchupSimulationResponse(
        team_a_name=body.team_a_name,
        team_b_name=body.team_b_name,
        best_of=best_of,
        map_prob_a=round(map_prob_a, 4) if map_prob_a is not None else None,
        map_prob_b=round(map_prob_b, 4) if map_prob_b is not None else None,
        series_prob_a=round(series_prob_a, 4),
        series_prob_b=round(1.0 - series_prob_a, 4),
        model_name=pred_res.model_name,
        model_version=pred_res.model_version,
        roster_a=roster_a_info,
        roster_b=roster_b_info,
        recent_stats_a=recent_stats_a,
        recent_stats_b=recent_stats_b,
        team_comparison=team_comparison,
        components=components,
    )


# ── POST /matches/alias/block — mark a team as blocked/unmapped ─────────────


@router.post("/alias/block")
def block_alias_endpoint(body: AliasBlockRequest, db=Depends(get_db)):
    """Mark a team name as blocked/unmapped."""
    from betting_app.services.mapping_service import block_alias

    block_alias(body.raw_name)
    return {"ok": True, "blocked": True}


# ── DELETE /matches/alias/block — unblock a team mapping marker ──────────────


@router.delete("/alias/block")
def unblock_alias_endpoint(body: AliasBlockRequest, db=Depends(get_db)):
    """Remove a blocked/unmapped marker for a team name."""
    from betting_app.services.mapping_service import unblock_alias

    unblocked = unblock_alias(body.raw_name)
    if not unblocked:
        raise HTTPException(status_code=404, detail="No block found for that team")
    return {"ok": True, "unblocked": True}


def _build_match_recommendation(
    team_a_name: str | None,
    team_b_name: str | None,
    has_unmapped_teams: bool,
    odds_rows: list[BookmakerOddsRow],
    hybrid_prob_a: float | None,
    hybrid_prob_b: float | None,
    pure_prob_a: float | None,
    pure_prob_b: float | None,
    diagnostics: Any = None,
    hybrid_diagnostics: Any = None,
    league: str | None = None,
    match_date: Any = None,
    best_of: int | None = None,
) -> BettingRecommendation:
    """Build a structured betting recommendation based on the latest model system."""
    name_a = team_a_name or "Drużyna A"
    name_b = team_b_name or "Drużyna B"

    odds_rows = [row for row in odds_rows if row.quote_status == "actionable"]
    if has_unmapped_teams:
        return BettingRecommendation(
            has_value=False,
            verdict="unmapped",
            verdict_label="Wymagane mapowanie drużyn",
            summary="Przynajmniej jedna drużyna nie jest jeszcze powiązana z bazą GOL.GG. Przypisz alias, aby system mógł wyliczyć ratingi i wygenerować rekomendację.",
            reasons=[
                "Brak jednoznacznego powiązania nazwy bukmacherskiej ze statystykami GOL.GG."
            ],
        )

    valid_odds_a = [
        o
        for o in odds_rows
        if o.canonical_odds_a
        and math.isfinite(o.canonical_odds_a)
        and o.canonical_odds_a > 1.0
    ]
    valid_odds_b = [
        o
        for o in odds_rows
        if o.canonical_odds_b
        and math.isfinite(o.canonical_odds_b)
        and o.canonical_odds_b > 1.0
    ]
    if not valid_odds_a or not valid_odds_b:
        return BettingRecommendation(
            has_value=False,
            verdict="no_odds",
            verdict_label="Brak aktualnych kursów",
            summary="Brak aktywnych kursów bukmacherskich dla tego spotkania. Rekomendacja pojawi się automatycznie po pobraniu oferty.",
            reasons=[
                "Brak jawnie przedmeczowych kwotowań nie starszych niż 300 sekund."
            ],
        )

    best_row_a = max(valid_odds_a, key=lambda o: o.canonical_odds_a or 0.0)
    best_row_b = max(valid_odds_b, key=lambda o: o.canonical_odds_b or 0.0)
    best_odds_a = float(best_row_a.canonical_odds_a or 1.0)
    best_odds_b = float(best_row_b.canonical_odds_b or 1.0)

    market_pairs = [
        fair_market_probabilities(o.canonical_odds_a, o.canonical_odds_b)
        for o in odds_rows
        if o in valid_odds_a and o in valid_odds_b
    ]
    m_prob_a = (
        sum(pair[0] for pair in market_pairs) / len(market_pairs)
        if market_pairs
        else None
    )
    m_prob_b = 1.0 - m_prob_a if m_prob_a is not None else None

    eff_prob_a = hybrid_prob_a if hybrid_prob_a is not None else pure_prob_a
    eff_prob_b = hybrid_prob_b if hybrid_prob_b is not None else pure_prob_b

    if eff_prob_a is None or eff_prob_b is None:
        return BettingRecommendation(
            has_value=False,
            verdict="no_prediction",
            verdict_label="Oczekiwanie na predykcję",
            summary="Brak wyliczonej predykcji dla tego meczu w systemie operacyjnym.",
            reasons=[
                "Model nie wygenerował jeszcze prawdopodobieństw dla tego zestawienia."
            ],
        )

    bounds_a = conformal_bounds_for_side(diagnostics, "a")
    bounds_b = conformal_bounds_for_side(diagnostics, "b")
    p_low_a = bounds_a[0] if bounds_a is not None else None
    p_low_b = bounds_b[0] if bounds_b is not None else None

    uses_hybrid = hybrid_prob_a is not None
    decisions = qualify_prediction_sides(
        prob_a=eff_prob_a,
        prob_b=eff_prob_b,
        diagnostics=hybrid_diagnostics if uses_hybrid else diagnostics,
        model_name=HYBRID_MODEL_NAME if uses_hybrid else DEFAULT_MODEL_NAME,
        odds_a=best_odds_a,
        odds_b=best_odds_b,
        market_prob_a=m_prob_a,
        league=league,
        match_date=match_date,
        best_of=best_of,
    )
    score_a = decisions["a"][2].get("prob_for_ev")
    score_b = decisions["b"][2].get("prob_for_ev")
    min_odds_a = (
        round(1.0 / (score_a * (1.0 - TAX_RATE)), 2) if score_a else None
    )
    min_odds_b = (
        round(1.0 / (score_b * (1.0 - TAX_RATE)), 2) if score_b else None
    )
    ev_a = decisions["a"][2].get("ev_net")
    ev_b = decisions["b"][2].get("ev_net")
    pure_ev_a = (
        expected_value(pure_prob_a, best_odds_a, TAX_RATE)
        if pure_prob_a is not None
        else None
    )
    pure_ev_b = (
        expected_value(pure_prob_b, best_odds_b, TAX_RATE)
        if pure_prob_b is not None
        else None
    )

    conf_ev_a = (
        expected_value(p_low_a, best_odds_a, TAX_RATE) if p_low_a is not None else None
    )
    conf_ev_b = (
        expected_value(p_low_b, best_odds_b, TAX_RATE) if p_low_b is not None else None
    )

    has_val_a = decisions["a"][0]
    has_val_b = decisions["b"][0]

    if not has_val_a and not has_val_b:
        return BettingRecommendation(
            has_value=False,
            verdict="no_bet",
            verdict_label="Brak zakwalifikowanego typu",
            model_prob=pure_prob_a,
            hybrid_prob=hybrid_prob_a,
            market_prob=m_prob_a,
            summary="Żadna strona nie spełnia wspólnej polityki EV, niepewności i bezpieczeństwa.",
            reasons=[
                f"{name_a}: {decisions['a'][1]}",
                f"{name_b}: {decisions['b'][1]}",
            ],
            threshold_info="Dodatnie średnie EV nie wystarcza do rekomendacji.",
        )

    pick_side = "a" if has_val_a and (not has_val_b or ev_a >= ev_b) else "b"
    team = name_a if pick_side == "a" else name_b
    opp = name_b if pick_side == "a" else name_a
    bm_row = best_row_a if pick_side == "a" else best_row_b
    odds_val = best_odds_a if pick_side == "a" else best_odds_b
    ev_val = ev_a if pick_side == "a" else ev_b
    pure_ev_val = pure_ev_a if pick_side == "a" else pure_ev_b
    h_prob = eff_prob_a if pick_side == "a" else eff_prob_b
    p_prob = pure_prob_a if pick_side == "a" else pure_prob_b
    m_prob = m_prob_a if pick_side == "a" else m_prob_b
    min_odds = min_odds_a if pick_side == "a" else min_odds_b
    p_low = p_low_a if pick_side == "a" else p_low_b
    c_ev = conf_ev_a if pick_side == "a" else conf_ev_b

    bounds = bounds_a if pick_side == "a" else bounds_b
    is_conf = bool(bounds and (bounds[1] - bounds[0] <= 0.08) and c_ev and c_ev > 0)

    qualification = decisions[pick_side][2]
    score = qualification["prob_for_ev"]
    basis = qualification["uncertainty_basis"]
    half_k = kelly_fraction(score, odds_val, TAX_RATE) / 2.0
    quarter_k = (half_k / 2.0) if half_k else None
    edge_pts = (
        (h_prob - m_prob) * 100.0 if h_prob is not None and m_prob is not None else None
    )
    thesis_disag = qualification.get("rating_disagreement")
    if thesis_disag is None and isinstance(diagnostics, dict):
        thesis_disag = diagnostics.get("rating_disagreement")
    thesis_guard_ok = (thesis_disag <= 0.10) if thesis_disag is not None else True
    logit_disag = qualification.get("logit_disagreement")
    guard_b_ok = (logit_disag <= 0.85) if logit_disag is not None else True
    raw_clv_val = round((odds_val * m_prob) - 1.0, 4) if (odds_val and m_prob) else None


    reasons = [
        f"Kurs {odds_val:.2f} w {bm_row.bookmaker}: EV {ev_val:+.1%} po podatku 12%; ocena użyta do kwalifikacji {score:.1%}, podstawa niepewności: {basis} ({qualification['uncertainty_reason']}), bez gwarancji pokrycia.",
        f"Model hybrydowy szacuje prawdopodobieństwo wygranej na {h_prob:.1%}, podczas gdy rynek bez marży implikuje {m_prob:.1%}"
        + (
            f" ({edge_pts:+.1f} p.p. przewagi nad rynkiem)."
            if edge_pts is not None
            else "."
        ),
    ]
    if p_prob is not None and pure_ev_val is not None:
        reasons.append(
            f"Zapisany model sportowy ocenia szanse na {p_prob:.1%} (EV średniej: {pure_ev_val:+.1%}); nie jest to pomiar dolnej granicy niepewności."
        )
    if min_odds is not None:
        reasons.append(
            f"Kurs {odds_val:.2f} przewyższa wymagany próg opłacalności {min_odds:.2f} (kurs poniżej tej wartości byłby nieopłacalny)."
        )
    if quarter_k is not None and quarter_k > 0:
        reasons.append(
            f"Ułamek Kelly'ego wyliczony z oceny kwalifikacyjnej ({basis}): {quarter_k:.2%}; to nie gwarancja ochrony kapitału ani kwota uwzględniająca otwarte zakłady.",
        )

    summary = f"Rekomendacja: typ na {team} po kursie {odds_val:.2f} w {bm_row.bookmaker} ({ev_val:+.1%} EV po podatku 12%)."

    return BettingRecommendation(
        has_value=True,
        verdict="value_bet",
        verdict_label="Wartościowy typ (Value Bet)",
        side=pick_side,
        recommended_team=team,
        opponent_team=opp,
        bookmaker=bm_row.bookmaker,
        best_odds=odds_val,
        offer_url=bm_row.offer_url or bm_row.source_url,
        model_prob=round(p_prob, 4) if p_prob is not None else None,
        hybrid_prob=round(h_prob, 4) if h_prob is not None else None,
        market_prob=round(m_prob, 4) if m_prob is not None else None,
        ev=round(ev_val, 4) if ev_val is not None else None,
        pure_model_ev=round(pure_ev_val, 4) if pure_ev_val is not None else None,
        edge_percentage_points=round(edge_pts, 1) if edge_pts is not None else None,
        min_odds_required=min_odds,
        half_kelly=round(half_k, 4) if half_k is not None else None,
        quarter_kelly=round(quarter_k, 4) if quarter_k is not None else None,
        suggested_stake_pct=round(quarter_k, 4) if quarter_k is not None else None,
        conformal_prob_low=round(p_low, 4) if p_low is not None else None,
        conformal_ev=round(c_ev, 4) if c_ev is not None else None,
        is_conformal_safe=is_conf,
        thesis_disagreement=round(thesis_disag, 4) if thesis_disag is not None else None,
        thesis_guard_passed=thesis_guard_ok,
        guard_b_passed=guard_b_ok,
        logit_disagreement=round(logit_disag, 4) if logit_disag is not None else None,
        raw_clv=raw_clv_val,
        summary=summary,
        reasons=reasons,
        threshold_info=f"Próg opłacalności kursu: ≥ {min_odds:.2f} | Najlepsza oferta: {odds_val:.2f} ({bm_row.bookmaker})",
    )


# ── GET /matches/recommendations/parlays (IDEA-019) ─────────────────────────


@router.get("/recommendations/parlays", response_model=ParlayRecommendationsResponse)
def get_parlay_recommendations(
    bookmaker: str | None = None,
    max_odds: float = 2.20,
    min_prob: float = 0.50,
    min_ev: float = 0.0,
    limit: int = 5,
    db=Depends(get_db),
) -> ParlayRecommendationsResponse:
    """Recommend tax-amortized 2-leg favorite accumulator bets (AKO / Dubel)."""
    return find_parlay_recommendations(
        db=db,
        bookmaker=bookmaker,
        max_odds_per_leg=max_odds,
        min_prob_per_leg=min_prob,
        min_parlay_ev=min_ev,
        limit=limit,
    )


def _load_historical_match_rosters(
    db: Any,
    match_id: int,
    team_a_name: str | None,
    team_b_name: str | None,
) -> tuple[RosterInfo | None, RosterInfo | None]:
    mapping = query_df(
        db,
        "SELECT golgg_match_id FROM golgg_match_mappings WHERE canonical_match_id = :mid LIMIT 1",
        {"mid": match_id},
    )
    if not mapping or not mapping[0].get("golgg_match_id"):
        return None, None

    golgg_match_id = str(mapping[0]["golgg_match_id"])
    game_players = query_df(
        db,
        """
        SELECT game_id, team_name, role, player_id, player_name, champion_name
        FROM golgg_game_players
        WHERE match_id = :mid
        ORDER BY CAST(game_id AS INTEGER) ASC, role ASC
        """,
        {"mid": golgg_match_id},
    )
    if not game_players:
        return None, None

    first_game_id = game_players[0].get("game_id")
    lineup_rows = [r for r in game_players if r.get("game_id") == first_game_id]
    if not lineup_rows:
        lineup_rows = game_players[:10]

    key_a = canonical_team_key(team_a_name) if team_a_name else ""
    key_b = canonical_team_key(team_b_name) if team_b_name else ""

    distinct_teams = list(dict.fromkeys(r.get("team_name") for r in lineup_rows if r.get("team_name")))
    team_a_actual = distinct_teams[0] if len(distinct_teams) > 0 else (team_a_name or "")
    team_b_actual = distinct_teams[1] if len(distinct_teams) > 1 else (team_b_name or "")

    if len(distinct_teams) >= 2:
        k0 = canonical_team_key(distinct_teams[0])
        k1 = canonical_team_key(distinct_teams[1])
        if k0 == key_b or k1 == key_a:
            team_a_actual, team_b_actual = distinct_teams[1], distinct_teams[0]

    all_player_ids = [str(r.get("player_id")) for r in lineup_rows if r.get("player_id")]
    ratings_by_pid: dict[str, dict[str, Any]] = {}
    if all_player_ids:
        placeholders = ", ".join(f":p{i}" for i in range(len(all_player_ids)))
        params = {f"p{i}": pid for i, pid in enumerate(all_player_ids)}
        rating_rows = query_df(
            db,
            f"""
            SELECT normalized_entity_name, rating_system, rating_value, rd, games_played
            FROM entity_ratings
            WHERE entity_type = 'player'
              AND normalized_entity_name IN ({placeholders})
              AND ratings_version = 'ratings-v2'
            """,
            params,
        )
        for rr in rating_rows:
            p_id = str(rr.get("normalized_entity_name"))
            if p_id not in ratings_by_pid:
                ratings_by_pid[p_id] = {}
            ratings_by_pid[p_id][str(rr.get("rating_system"))] = rr

    players_a: list[RosterPlayer] = []
    players_b: list[RosterPlayer] = []

    for r in lineup_rows:
        tname = r.get("team_name")
        p_id = str(r.get("player_id") or "")
        p_ratings = ratings_by_pid.get(p_id, {})
        gl = p_ratings.get("gl", {})
        elo = p_ratings.get("elo", {})
        ts = p_ratings.get("ts", {})
        gp = none_or_float(gl.get("games_played") or elo.get("games_played"))
        rp = RosterPlayer(
            player_id=p_id or None,
            player_name=clean_player_name(r.get("player_name") or p_id),
            role=r.get("role"),
            champion_name=r.get("champion_name"),
            elo_rating=_finite_float(elo.get("rating_value")),
            glicko_rating=_finite_float(gl.get("rating_value")),
            glicko_rd=_finite_float(gl.get("rd")),
            trueskill_rating=_finite_float(ts.get("rating_value")),
            rating_uncertainty=_finite_float(ts.get("sigma")),
            games_played=int(gp) if gp is not None else None,
        )
        if tname == team_a_actual:
            players_a.append(rp)
        else:
            players_b.append(rp)

    def make_roster_info(p_list: list[RosterPlayer], t_name: Any) -> RosterInfo:
        gl_values = [p.glicko_rating for p in p_list if p.glicko_rating is not None]
        elo_values = [p.elo_rating for p in p_list if p.elo_rating is not None]
        rd_values = [p.glicko_rd for p in p_list if p.glicko_rd is not None]
        return RosterInfo(
            team_name=t_name,
            source_match_id=golgg_match_id,
            source_tournament="GOL.GG match history",
            roster_source="golgg_match",
            avg_elo=sum(elo_values) / len(elo_values) if elo_values else None,
            avg_glicko=sum(gl_values) / len(gl_values) if gl_values else None,
            avg_glicko_rd=sum(rd_values) / len(rd_values) if rd_values else None,
            players_with_rating=len(gl_values),
            players=p_list,
        )

    roster_a_out = make_roster_info(players_a, team_a_name) if players_a else None
    roster_b_out = make_roster_info(players_b, team_b_name) if players_b else None
    return roster_a_out, roster_b_out

# ── GET /matches/{id} ───────────────────────────────────────────────────────


@router.get("/{match_id}", response_model=MatchDetailResponse)
def match_detail(
    match_id: int,
    stale_hours: float = 72,
    max_odds_age_hours: float = DEFAULT_MAX_ODDS_AGE_HOURS,
    db=Depends(get_db),
):
    meta = query_df(db, "SELECT * FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    m = meta[0]

    live_team_a = suggest_mapping(str(m.get("team_a_name"))) if m.get("team_a_name") else (None, 0.0, None)
    live_team_b = suggest_mapping(str(m.get("team_b_name"))) if m.get("team_b_name") else (None, 0.0, None)
    has_unmapped_teams = not live_team_a[0] or not live_team_b[0]

    now = datetime.now(UTC)
    stale_cutoff = now - timedelta(hours=stale_hours)
    odds_cutoff = now - timedelta(hours=max_odds_age_hours)

    context = load_prematch_quotes([match_id], decision_at=now, session=db)[match_id]
    odds = query_df(db, """
        SELECT os.*, os.id AS odds_snapshot_id, b.name AS bookmaker
        FROM odds_snapshots os JOIN bookmakers b ON b.id=os.bookmaker_id
        WHERE canonical_match_id=:mid AND market_type='match_winner'
        """, {"mid": match_id})
    # Select the current book state before filtering its display age: never
    # revive an older observation when the newest is unavailable.
    odds = [row for row in _latest_display_states(odds, now)
            if parse_aware_utc(row.get("scraped_at")) >= odds_cutoff]
    raw_preds = query_df(db, """
        SELECT * FROM canonical_predictions WHERE canonical_match_id=:mid
        AND prediction_status='active'
        AND ((model_name=:sn AND model_version=:sv) OR (model_name=:hn AND model_version=:hv))
        """, {"mid": match_id, **get_active_prediction_db_params()})
    admitted = {quote.odds_snapshot_id: quote for quote in context.quotes}
    sources_unchanged = all(any(
        row["odds_snapshot_id"] == quote.odds_snapshot_id and _display_matches_quote(
            row, quote, _align(dict(row, league=m.get("league"), start_time_normalized=m.get("start_time_normalized")),
                              str(m.get("team_a_name") or ""), str(m.get("team_b_name") or "")))
        for row in odds) for quote in context.quotes)
    if not sources_unchanged:
        admitted = {}
    serving = {} if has_unmapped_teams or not sources_unchanged else _serving_predictions(db, raw_preds, context)
    hybrid_prob_a, hybrid_prob_b = serving.get("hybrid_prob_a"), serving.get("hybrid_prob_b")
    hybrid_diagnostics = serving.get("hybrid_diagnostics")
    odds_rows: list[BookmakerOddsRow] = []
    for row in odds:
        row.update(league=m.get("league"), start_time_normalized=m.get("start_time_normalized"))
        aligned = _align(row, str(m.get("team_a_name") or ""), str(m.get("team_b_name") or ""))
        oa, ob = aligned if aligned is not None else (None, None)
        quote = admitted.get(row["odds_snapshot_id"])
        if not _display_matches_quote(row, quote, aligned):
            quote = None
        if quote is not None:
            oa, ob = quote.odds_a, quote.odds_b
        clock = parse_aware_utc(row.get("scraped_at"))
        status = "actionable" if quote else ("diagnostic" if aligned is not None and row.get("is_live") == 0 else "unavailable")
        pa = hybrid_prob_a if quote else None
        pb = hybrid_prob_b if quote else None
        qualified = qualify_prediction_sides(
            prob_a=pa, prob_b=pb, diagnostics=hybrid_diagnostics, model_name=HYBRID_MODEL_NAME,
            odds_a=quote.odds_a if quote else None, odds_b=quote.odds_b if quote else None,
            market_prob_a=context.prob_a, league=m.get("league"), match_date=m.get("start_time_normalized"))
        odds_rows.append(BookmakerOddsRow(
            bookmaker=row["bookmaker"], raw_team_a=row.get("raw_team_a"), raw_team_b=row.get("raw_team_b"),
            canonical_odds_a=oa, canonical_odds_b=ob, odds_snapshot_id=row["odds_snapshot_id"],
            scraped_at=clock.isoformat() if clock else None, quote_scraped_at=clock.isoformat() if clock else None,
            quote_status=status, source_url=row.get("source_url"), offer_url=row.get("offer_url"),
            ev_a=qualified["a"][2].get("ev_net"),
            ev_b=qualified["b"][2].get("ev_net"),
            kelly_a=kelly_fraction(qualified["a"][2]["prob_for_ev"], oa, TAX_RATE) if qualified["a"][0] else None,
            kelly_b=kelly_fraction(qualified["b"][2]["prob_for_ev"], ob, TAX_RATE) if qualified["b"][0] else None))
    preds = [p for p in latest_predictions(raw_preds) if prediction_time_is_actionable(
        p, decision_at=now, start_at=context.start_at, model_name=p["model_name"], model_version=p["model_version"])[0]]
    best_a = max((q.odds_a for q in admitted.values()), default=None)
    best_b = max((q.odds_b for q in admitted.values()), default=None)
    pred_rows = []
    for p in preds:
        is_hybrid = p["model_name"] == HYBRID_MODEL_NAME
        pa, pb = _probability(p.get("prob_a")), _probability(p.get("prob_b"))
        serving_pa = serving.get("hybrid_prob_a" if is_hybrid else "model_prob_a")
        serving_pb = 1-serving_pa if serving_pa is not None else None
        decisions = qualify_prediction_sides(
            prob_a=serving_pa, prob_b=serving_pb, diagnostics=p.get("diagnostics_json"),
            model_name=p["model_name"], odds_a=best_a, odds_b=best_b,
            market_prob_a=context.prob_a, league=m.get("league"),
            match_date=m.get("start_time_normalized"), best_of=m.get("best_of"),
        )
        clock = parse_aware_utc(p.get("predicted_at"))
        pred_rows.append(PredictionRow(
            model_name=p["model_name"], model_version=p["model_version"], prob_a=pa, prob_b=pb,
            predicted_at=clock.isoformat() if clock else None,
            raw_diagnostics=_raw_diagnostics(p.get("diagnostics_json")),
            ev_a=decisions["a"][2].get("ev_net"),
            ev_b=decisions["b"][2].get("ev_net"),
            kelly_a=kelly_fraction(decisions["a"][2]["prob_for_ev"], best_a, TAX_RATE) if decisions["a"][0] else None,
            kelly_b=kelly_fraction(decisions["b"][2]["prob_for_ev"], best_b, TAX_RATE) if decisions["b"][0] else None))

    # Rosters from features_json
    roster_a: RosterInfo | None = None
    roster_b: RosterInfo | None = None
    roster_a_is_manual = False
    roster_b_is_manual = False
    recent_stats_a: TeamRecentStats | None = None
    recent_stats_b: TeamRecentStats | None = None
    team_comparison: TeamComparisonInfo | None = None

    feat = query_df(
        db,
        """
        SELECT features_json FROM upcoming_match_features
        WHERE canonical_match_id=:mid ORDER BY id DESC LIMIT 1
        """,
        {"mid": match_id},
    )
    manual_rosters = _load_roster_overrides(match_id)
    roster_a_is_manual = "a" in manual_rosters
    roster_b_is_manual = "b" in manual_rosters
    if feat:
        f = feat[0].get("features_json")
        if isinstance(f, str):
            import json
            try:
                f = json.loads(f)
            except Exception:
                f = {}

        # Extract team mapping info
        mapping = safe_json_get(f, ["mapping"])
        if isinstance(mapping, dict):
            team_a_conf = none_or_float(mapping.get("team_a_confidence"))
            team_a_golgg = mapping.get("team_a_golgg_name")
            team_b_conf = none_or_float(mapping.get("team_b_confidence"))
            team_b_golgg = mapping.get("team_b_golgg_name")

            live_a = live_team_a
            live_b = live_team_b

            # Always trust live mapping status over stored features_json. Stored
            # features may contain stale fuzzy mappings produced before aliases
            # changed or fuzzy auto-mapping was disabled.
            if live_a[0]:
                team_a_golgg, team_a_conf, team_a_source = live_a
            elif live_a[2] == "blocked":
                team_a_golgg, team_a_conf = None, None
                team_a_source = "blocked"
            else:
                team_a_golgg, team_a_conf, team_a_source = None, None, None

            if live_b[0]:
                team_b_golgg, team_b_conf, team_b_source = live_b
            elif live_b[2] == "blocked":
                team_b_golgg, team_b_conf = None, None
                team_b_source = "blocked"
            else:
                team_b_golgg, team_b_conf, team_b_source = None, None, None

            team_a_info = TeamMappingInfo(
                canonical_name=m.get("team_a_name"),
                golgg_name=team_a_golgg,
                confidence=team_a_conf,
                source=team_a_source,
            )
            team_b_info = TeamMappingInfo(
                canonical_name=m.get("team_b_name"),
                golgg_name=team_b_golgg,
                confidence=team_b_conf,
                source=team_b_source,
            )

            # Get team ratings for comparison
            ratings = safe_json_get(f, ["ratings"])
            team_a_rating = None
            team_b_rating = None
            rating_system = None
            team_a_elo = team_b_elo = None
            team_a_glicko = team_b_glicko = None
            team_a_glicko_rd = team_b_glicko_rd = None
            team_a_games_played = team_b_games_played = None
            rating_probabilities: dict[str, float] = {}

            if isinstance(ratings, dict):
                team_a_ratings = ratings.get("team_a", {})
                team_b_ratings = ratings.get("team_b", {})
                probs = ratings.get("probabilities", {})
                if isinstance(probs, dict):
                    rating_probabilities = {
                        str(k): parsed
                        for k, v in probs.items()
                        if (parsed := _finite_float(v)) is not None
                    }

                if isinstance(team_a_ratings, dict) and isinstance(team_b_ratings, dict):
                    team_a_elo = _finite_float(safe_json_get(team_a_ratings, ["elo", "rating_value"]))
                    team_b_elo = _finite_float(safe_json_get(team_b_ratings, ["elo", "rating_value"]))
                    team_a_glicko = _finite_float(safe_json_get(team_a_ratings, ["gl", "rating_value"]))
                    team_b_glicko = _finite_float(safe_json_get(team_b_ratings, ["gl", "rating_value"]))
                    team_a_glicko_rd = _finite_float(safe_json_get(team_a_ratings, ["gl", "rd"]))
                    team_b_glicko_rd = _finite_float(safe_json_get(team_b_ratings, ["gl", "rd"]))
                    gp_a = none_or_float(safe_json_get(team_a_ratings, ["gl", "games_played"]))
                    gp_b = none_or_float(safe_json_get(team_b_ratings, ["gl", "games_played"]))
                    team_a_games_played = int(gp_a) if gp_a is not None else None
                    team_b_games_played = int(gp_b) if gp_b is not None else None

                # Prefer Glicko rating system
                if "gl" in team_a_ratings and "gl" in team_b_ratings:
                    team_a_rating = _finite_float(team_a_ratings["gl"].get("rating_value"))
                    team_b_rating = _finite_float(team_b_ratings["gl"].get("rating_value"))
                    rating_system = "Glicko"
                elif "elo" in team_a_ratings and "elo" in team_b_ratings:
                    team_a_rating = _finite_float(team_a_ratings["elo"].get("rating_value"))
                    team_b_rating = _finite_float(team_b_ratings["elo"].get("rating_value"))
                    rating_system = "Elo"

            # Don't show ratings for unmapped/blocked teams
            if team_a_golgg is None:
                team_a_rating = None
            if team_b_golgg is None:
                team_b_rating = None

            team_comparison = TeamComparisonInfo(
                team_a=team_a_info,
                team_b=team_b_info,
                team_a_rating=team_a_rating,
                team_b_rating=team_b_rating,
                rating_system=rating_system,
                team_a_elo=team_a_elo,
                team_b_elo=team_b_elo,
                team_a_glicko=team_a_glicko,
                team_b_glicko=team_b_glicko,
                team_a_glicko_rd=team_a_glicko_rd,
                team_b_glicko_rd=team_b_glicko_rd,
                team_a_games_played=team_a_games_played,
                team_b_games_played=team_b_games_played,
                rating_probabilities=rating_probabilities,
            )

        def build_recent_stats(raw: Any) -> TeamRecentStats | None:
            if not isinstance(raw, dict):
                return None
            games = none_or_float(raw.get("games_count"))
            matches = none_or_float(raw.get("matches_count"))
            return TeamRecentStats(
                team_name=raw.get("team_name"),
                matches_count=int(matches) if matches is not None else None,
                games_count=int(games) if games is not None else None,
                win_rate=_finite_float(raw.get("win_rate")),
                avg_kills=_finite_float(raw.get("avg_kills")),
                avg_deaths=_finite_float(raw.get("avg_deaths")),
                avg_gd15=_finite_float(raw.get("avg_gd15")),
                avg_dragons=_finite_float(raw.get("avg_dragons")),
                avg_nashors=_finite_float(raw.get("avg_nashors")),
                avg_towers=_finite_float(raw.get("avg_towers")),
                avg_game_duration=_finite_float(raw.get("avg_game_duration")),
                last_match_at=raw.get("last_match_at"),
            )

        w20 = safe_json_get(f, ["w20"])
        if isinstance(w20, dict):
            recent_stats_a = build_recent_stats(w20.get("team_a"))
            recent_stats_b = build_recent_stats(w20.get("team_b"))

        player_ratings = safe_json_get(f, ["player_ratings"])
        roster_source = safe_json_get(player_ratings, ["roster_source"]) if isinstance(player_ratings, dict) else None

        def players_by_id(side_key: str, system: str) -> dict[str, dict[str, Any]]:
            raw_players = safe_json_get(player_ratings, [side_key, system, "players"])
            out: dict[str, dict[str, Any]] = {}
            if isinstance(raw_players, list):
                for player in raw_players:
                    if not isinstance(player, dict):
                        continue
                    candidates = [
                        player.get("normalized_entity_name"),
                        player.get("player_id"),
                        player.get("entity_name"),
                        player.get("player_name"),
                    ]
                    for c in candidates:
                        if not c:
                            continue
                        str_c = str(c)
                        clean_c = clean_player_name(str_c)
                        out[str_c] = player
                        if clean_c:
                            out[clean_c] = player
                            out[clean_c.lower()] = player
                            squashed = re.sub(r"[^a-zA-Z0-9]", "", clean_c).lower()
                            if squashed:
                                out[squashed] = player
            return out

        for side_key, side_label, out in [
            ("team_a_roster", m.get("team_a_name", "Team A"), "a"),
            ("team_b_roster", m.get("team_b_name", "Team B"), "b"),
        ]:
            players: list[RosterPlayer] = []
            manual_roster = manual_rosters.get(out)
            raw_roster = manual_roster or (safe_json_get(player_ratings, [side_key]) if isinstance(player_ratings, dict) else None)
            rating_side_key = "team_a" if side_key.startswith("team_a") else "team_b"
            gl_by_id = players_by_id(rating_side_key, "gl")
            elo_by_id = players_by_id(rating_side_key, "elo")
            ts_by_id = players_by_id(rating_side_key, "ts")
            gl_summary = safe_json_get(player_ratings, [rating_side_key, "gl"]) if isinstance(player_ratings, dict) else {}
            elo_summary = safe_json_get(player_ratings, [rating_side_key, "elo"]) if isinstance(player_ratings, dict) else {}

            if isinstance(raw_roster, dict) and isinstance(raw_roster.get("players"), list):
                for pl in raw_roster["players"]:
                    if not isinstance(pl, dict):
                        continue
                    pid = str(pl.get("player_id") or pl.get("normalized_entity_name") or pl.get("player_name") or "")
                    keys = [
                        pl.get("player_id"),
                        pl.get("player_name"),
                        pl.get("normalized_entity_name"),
                    ]
                    gl: dict[str, Any] = {}
                    elo: dict[str, Any] = {}
                    ts: dict[str, Any] = {}
                    for k in keys:
                        if not k:
                            continue
                        str_k = str(k)
                        clean_k = clean_player_name(str_k)
                        squashed_k = re.sub(r"[^a-zA-Z0-9]", "", clean_k).lower()
                        for test_k in (str_k, clean_k, clean_k.lower(), squashed_k):
                            if not gl and test_k in gl_by_id:
                                gl = gl_by_id[test_k]
                            if not elo and test_k in elo_by_id:
                                elo = elo_by_id[test_k]
                            if not ts and test_k in ts_by_id:
                                ts = ts_by_id[test_k]

                    gp = none_or_float(gl.get("games_played") or elo.get("games_played"))
                    canonical_ign = gl.get("entity_name") or elo.get("entity_name") or ts.get("entity_name")
                    ign_candidate = str(pl.get("player_id")) if pl.get("player_id") and not str(pl.get("player_id")).isdigit() else None
                    display_player_name = clean_player_name(canonical_ign or ign_candidate or pl.get("player_name") or pid)
                    resolved_pid = str(gl.get("player_id") or elo.get("player_id") or pid)

                    players.append(RosterPlayer(
                        player_id=resolved_pid or None,
                        player_name=display_player_name,
                        role=pl.get("role"),
                        champion_name=pl.get("champion_name"),
                        elo_rating=_finite_float(elo.get("rating_value")),
                        glicko_rating=_finite_float(gl.get("rating_value")),
                        glicko_rd=_finite_float(gl.get("rd")),
                        trueskill_rating=_finite_float(ts.get("rating_value")),
                        rating_uncertainty=_finite_float(ts.get("sigma")),
                        games_played=int(gp) if gp is not None else None,
                    ))
            ri = RosterInfo(
                team_name=(raw_roster.get("team_name") if isinstance(raw_roster, dict) else None) or side_label,
                source_match_id=str(raw_roster.get("source_match_id")) if isinstance(raw_roster, dict) and raw_roster.get("source_match_id") else None,
                source_date=str(raw_roster.get("source_match_date")) if isinstance(raw_roster, dict) and raw_roster.get("source_match_date") else None,
                source_tournament=str(raw_roster.get("source_tournament")) if isinstance(raw_roster, dict) and raw_roster.get("source_tournament") else None,
                roster_source="manual_override" if manual_roster else (str(roster_source) if roster_source else None),
                avg_elo=_finite_float(safe_json_get(elo_summary, ["avg_rating_value"])),
                avg_glicko=_finite_float(safe_json_get(gl_summary, ["avg_rating_value"])),
                avg_glicko_rd=_finite_float(safe_json_get(gl_summary, ["avg_rd"])),
                players_with_rating=int(none_or_float(safe_json_get(gl_summary, ["players_with_rating"])) or 0) if isinstance(gl_summary, dict) else None,
                players=players,
            )
            if out == "a":
                roster_a = ri
            else:
                roster_b = ri

    # A confirmed manual roster is valid even when feature generation is not
    # available (for example when a duplicated canonical match was repaired
    # after kickoff). Do not hide the user's confirmed lineup merely because
    # there is no current upcoming_match_features row to enrich it with.
    def manual_roster_info(side: str, fallback_team_name: Any) -> RosterInfo | None:
        raw_roster = manual_rosters.get(side)
        if not isinstance(raw_roster, dict):
            return None
        raw_players = raw_roster.get("players")
        if not isinstance(raw_players, list):
            raw_players = []
        return RosterInfo(
            team_name=raw_roster.get("team_name") or fallback_team_name,
            source_match_id=str(raw_roster.get("source_match_id")) if raw_roster.get("source_match_id") else None,
            source_date=str(raw_roster.get("source_match_date")) if raw_roster.get("source_match_date") else None,
            source_tournament=str(raw_roster.get("source_tournament")) if raw_roster.get("source_tournament") else None,
            roster_source="manual_override",
            players=[
                RosterPlayer(
                    player_id=str(player.get("player_id")) if player.get("player_id") else None,
                    player_name=player.get("player_name"),
                    role=player.get("role"),
                )
                for player in raw_players if isinstance(player, dict)
            ],
        )

    if roster_a is None and roster_a_is_manual:
        roster_a = manual_roster_info("a", m.get("team_a_name"))
    if roster_b is None and roster_b_is_manual:
        roster_b = manual_roster_info("b", m.get("team_b_name"))
    if roster_a is None and roster_b is None and not has_unmapped_teams:
        hist_a, hist_b = _load_historical_match_rosters(db, match_id, m.get("team_a_name"), m.get("team_b_name"))
        if hist_a is not None or hist_b is not None:
            roster_a = hist_a
            roster_b = hist_b
    operational_pred = next(
        (p for p in preds if p.get("model_name") == DEFAULT_MODEL_NAME), None
    )
    pure_prob_a = (
        none_or_float(operational_pred.get("prob_a")) if operational_pred else None
    )
    pure_prob_b = (
        none_or_float(operational_pred.get("prob_b")) if operational_pred else None
    )
    op_diagnostics = (
        operational_pred.get("diagnostics_json") if operational_pred else None
    )

    rec = _build_match_recommendation(
        team_a_name=m.get("team_a_name"),
        team_b_name=m.get("team_b_name"),
        has_unmapped_teams=has_unmapped_teams,
        odds_rows=odds_rows,
        hybrid_prob_a=hybrid_prob_a,
        hybrid_prob_b=hybrid_prob_b,
        pure_prob_a=pure_prob_a,
        pure_prob_b=pure_prob_b,
        diagnostics=op_diagnostics,
        hybrid_diagnostics=hybrid_diagnostics,
        league=m.get("league"),
        match_date=m.get("start_time_normalized"),
        best_of=m.get("best_of"),
    )

    return MatchDetailResponse(
        canonical_match_id=match_id,
        team_a_name=m.get("team_a_name"),
        team_b_name=m.get("team_b_name"),
        league=m.get("league"),
        start_time_normalized=m.get("start_time_normalized"),
        status=m.get("status"),
        best_of=m.get("best_of"),
        odds=odds_rows,
        predictions=pred_rows,
        roster_a=roster_a,
        roster_b=roster_b,
        roster_a_is_manual=roster_a_is_manual,
        roster_b_is_manual=roster_b_is_manual,
        recent_stats_a=recent_stats_a,
        recent_stats_b=recent_stats_b,
        team_comparison=team_comparison,
        recommendation=rec,
    )


# ── GET /matches/{id}/odds-history ──────────────────────────────────────────


@router.get("/{match_id}/odds-history", response_model=list[OddsHistoryPoint])
def odds_history(match_id: int, db=Depends(get_db)):
    meta = query_df(db, "SELECT * FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")

    rows = query_df(
        db,
        """
        SELECT b.name AS bookmaker,
               os.scraped_at,
               os.odds_a, os.odds_b,
               os.raw_team_a, os.raw_team_b
        FROM odds_snapshots os
        JOIN bookmakers b ON b.id=os.bookmaker_id
        WHERE os.canonical_match_id=:mid AND os.market_type='match_winner'
          AND COALESCE(os.is_live,0)=0
          AND os.odds_a IS NOT NULL AND os.odds_b IS NOT NULL
        ORDER BY os.scraped_at
        """,
        {"mid": match_id},
    )

    n_a = str(meta[0].get("team_a_name") or "")
    n_b = str(meta[0].get("team_b_name") or "")

    history: list[OddsHistoryPoint] = []
    for row in rows:
        row.update(league=meta[0].get("league"), start_time_normalized=meta[0].get("start_time_normalized"))
        aligned = _align(row, n_a, n_b)
        history.append(OddsHistoryPoint(
            bookmaker=row["bookmaker"],
            scraped_at=parse_aware_utc(row.get("scraped_at")).isoformat() if parse_aware_utc(row.get("scraped_at")) else str(row.get("scraped_at") or ""),
            odds_a=row.get("odds_a"),
            odds_b=row.get("odds_b"),
            canonical_odds_a=aligned[0] if aligned is not None else None,
            canonical_odds_b=aligned[1] if aligned is not None else None,
        ))
    return history


# ── GET /matches/{id}/market-comparison ─────────────────────────────────────


@router.get("/{match_id}/market-comparison")
def match_market_comparison(
    match_id: int,
    horizon_hours: float = 1.0,
    tolerance_hours: float = 0.5,
    db=Depends(get_db),
):
    """Compare Pinnacle consensus line with Polish bookmakers and model EV around a horizon."""
    from betting_app.services.oddspapi_service import compare_match_market

    comparison = compare_match_market(
        canonical_match_id=match_id,
        session=db,
        horizon_hours=horizon_hours,
        tolerance_hours=tolerance_hours,
    )
    if not comparison:
        raise HTTPException(status_code=404, detail="Match not found or start time unavailable")
    return comparison


# ── GET /matches/{id}/prediction-history ────────────────────────────────────


@router.get("/{match_id}/prediction-history", response_model=list[PredictionHistoryPoint])
def prediction_history(match_id: int, db=Depends(get_db)):
    """Diagnostic stored forecasts against latest known states at issuance, never future quotes."""
    meta = query_df(db, "SELECT * FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    match = meta[0]
    preds = query_df(db, "SELECT * FROM canonical_predictions WHERE canonical_match_id=:mid",
                     {"mid": match_id})
    states = query_df(db, """
        SELECT os.*, os.id AS odds_snapshot_id, b.name AS bookmaker
        FROM odds_snapshots os JOIN bookmakers b ON b.id=os.bookmaker_id
        WHERE canonical_match_id=:mid AND market_type='match_winner'
        """, {"mid": match_id})
    for state in states:
        state.update(league=match.get("league"), start_time_normalized=match.get("start_time_normalized"))
    result = []
    for prediction in sorted(preds, key=lambda p: (parse_aware_utc(p.get("predicted_at")) or datetime.min.replace(tzinfo=UTC), p["id"])):
        clock = parse_aware_utc(prediction.get("predicted_at"))
        known = _latest_display_states(states, clock) if clock else []
        pairs = [_align(row, str(match.get("team_a_name") or ""), str(match.get("team_b_name") or ""))
                 for row in known if row.get("is_live") == 0]
        pairs = [pair for pair in pairs if pair is not None]
        avg_a = sum(pair[0] for pair in pairs)/len(pairs) if pairs else None
        avg_b = sum(pair[1] for pair in pairs)/len(pairs) if pairs else None
        q = _historical_consensus(states, clock, str(match.get("team_a_name") or ""), str(match.get("team_b_name") or "")) if clock else None
        pa, pb = none_or_float(prediction.get("prob_a")), none_or_float(prediction.get("prob_b"))
        decisions = qualify_prediction_sides(
            prob_a=pa, prob_b=pb, diagnostics=prediction.get("diagnostics_json"),
            model_name=prediction["model_name"], odds_a=avg_a, odds_b=avg_b,
            market_prob_a=q, league=match.get("league"),
            match_date=match.get("start_time_normalized"), best_of=match.get("best_of"),
        )
        result.append(PredictionHistoryPoint(
            timestamp=clock.isoformat() if clock else str(prediction.get("predicted_at") or ""),
            model_name=prediction["model_name"], model_version=prediction["model_version"],
            prob_a=pa, prob_b=pb, avg_odds_a=avg_a, avg_odds_b=avg_b,
            market_prob_a=q, market_prob_b=1-q if q is not None else None,
            raw_diagnostics=_raw_diagnostics(prediction.get("diagnostics_json")),
            ev_a=decisions["a"][2].get("ev_net"),
            ev_b=decisions["b"][2].get("ev_net")))
    return result


# ── PATCH /matches/{id} — update best_of ────────────────────────────────────


@router.patch("/{match_id}", response_model=MatchBestOfUpdate)
def update_match_best_of(match_id: int, body: MatchBestOfUpdate, db=Depends(get_db)):
    meta = query_df(db, "SELECT id FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    db.execute(
        text("UPDATE canonical_matches SET best_of = :best_of WHERE id = :id"),
        {"best_of": body.best_of, "id": match_id},
    )
    db.commit()
    return body


# ── PUT /matches/{id}/roster — manually confirm an upcoming roster ─────────


def _resolve_roster_player(
    db,
    player_id: str | None,
    player_name: str,
    role: str | None = None,
    expected_team: str | None = None,
) -> dict[str, str]:
    """Resolve a UI player entry to a stable GOL.GG player id.

    Keeping the id is essential: the rating pipeline is keyed by
    ``normalized_entity_name``/player id, while names can change in feeds.
    """
    if player_id:
        row = query_one(
            db,
            """
            SELECT player_id, player_name
            FROM golgg_game_players
            WHERE CAST(player_id AS TEXT)=:player_id
            ORDER BY match_id DESC, game_id DESC
            LIMIT 1
            """,
            {"player_id": str(player_id)},
        )
        if row:
            return {"player_id": str(row["player_id"]), "player_name": str(row.get("player_name") or player_name)}

    candidates = query_df(
        db,
        """
        SELECT DISTINCT ON (player_id) player_id, player_name, role, team_name
        FROM golgg_game_players
        WHERE LOWER(player_name)=LOWER(:player_name)
          AND (:role IS NULL OR UPPER(COALESCE(role, ''))=UPPER(:role))
        ORDER BY player_id,
            CASE WHEN LOWER(COALESCE(team_name, ''))=LOWER(COALESCE(:expected_team, '')) THEN 0 ELSE 1 END,
            match_id DESC, game_id DESC
        """,
        {"player_name": player_name.strip(), "role": str(role or "").strip() or None, "expected_team": expected_team},
    )
    if not candidates:
        squashed = re.sub(r"[^a-zA-Z0-9]", "", clean_player_name(player_name)).lower()
        if squashed:
            candidates = query_df(
                db,
                """
                SELECT DISTINCT ON (player_id) player_id, player_name, role, team_name
                FROM golgg_game_players
                WHERE REPLACE(LOWER(player_name), ' ', '') = :squashed
                  AND (:role IS NULL OR UPPER(COALESCE(role, ''))=UPPER(:role))
                ORDER BY player_id,
                    CASE WHEN LOWER(COALESCE(team_name, ''))=LOWER(COALESCE(:expected_team, '')) THEN 0 ELSE 1 END,
                    match_id DESC, game_id DESC
                """,
                {"squashed": squashed, "role": str(role or "").strip() or None, "expected_team": expected_team},
            )
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail=f"Nie znaleziono zawodnika GOL.GG: {player_name}. Wybierz istniejącą nazwę lub popraw ID.",
        )
    if len(candidates) > 1:
        options = "; ".join(
            f"{row.get('player_name')} (ID {row.get('player_id')}, {row.get('team_name') or 'brak drużyny'})"
            for row in candidates[:5]
        )
        raise HTTPException(
            status_code=422,
            detail=f"Niejednoznaczny zawodnik GOL.GG: {player_name}. Wybierz go z wyszukiwarki. Kandydaci: {options}",
        )
    row = candidates[0]
    return {"player_id": str(row["player_id"]), "player_name": str(row.get("player_name") or player_name)}


@router.get("/{match_id}/roster/players")
def search_roster_players(
    match_id: int,
    query: str,
    team_side: str,
    role: str | None = None,
    db=Depends(get_db),
):
    """Search and explicitly select a stable GOL.GG player ID for a manual roster."""
    if team_side not in {"a", "b"}:
        raise HTTPException(status_code=422, detail="team_side must be 'a' or 'b'")
    query = query.strip()
    if len(query) < 2:
        return {"players": []}
    feature = query_one(
        db,
        "SELECT team_a_golgg_name, team_b_golgg_name FROM upcoming_match_features WHERE canonical_match_id=:match_id",
        {"match_id": match_id},
    ) or {}
    expected_team = feature.get("team_a_golgg_name") if team_side == "a" else feature.get("team_b_golgg_name")
    rows = query_df(
        db,
        """
        SELECT DISTINCT ON (player_id) player_id, player_name, role, team_name
        FROM golgg_game_players
        WHERE player_id IS NOT NULL AND player_name IS NOT NULL
          AND LOWER(player_name) LIKE LOWER(:pattern)
          AND (:role IS NULL OR UPPER(COALESCE(role, ''))=UPPER(:role))
        ORDER BY player_id,
            CASE WHEN LOWER(COALESCE(team_name, ''))=LOWER(COALESCE(:expected_team, '')) THEN 0 ELSE 1 END,
            match_id DESC, game_id DESC
        LIMIT 12
        """,
        {"pattern": f"%{query}%", "role": str(role or "").strip() or None, "expected_team": expected_team},
    )
    return {"players": [
        {
            "player_id": str(row["player_id"]), "player_name": str(row.get("player_name") or ""),
            "role": row.get("role"), "team_name": row.get("team_name"),
            "is_expected_team": bool(expected_team and str(row.get("team_name") or "").lower() == str(expected_team).lower()),
        }
        for row in rows
    ]}


@router.put("/{match_id}/roster", response_model=MatchRosterOverrideResponse)
def update_match_roster(match_id: int, body: MatchRosterOverrideRequest, db=Depends(get_db)):
    """Store a confirmed five-player roster for manual prediction.

    The override survives normal feature refreshes and is consumed by both the
    single-match Predict button and the scheduled thesis prediction pipeline.
    It does not fabricate player ratings: each input is resolved to an actual
    GOL.GG player id and uses that player's existing historical rating.
    """
    meta = query_one(
        db,
        "SELECT id, team_a_name, team_b_name, status FROM canonical_matches WHERE id=:id FOR UPDATE",
        {"id": match_id},
    )
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    if str(meta.get("status") or "") in {"finished", "completed"}:
        raise HTTPException(status_code=409, detail="Nie można zmienić składu zakończonego meczu")

    roles = [str(p.role or "").upper() for p in body.players]
    if any(role and role not in {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"} for role in roles):
        raise HTTPException(status_code=422, detail="Dozwolone role: TOP, JUNGLE, MID, ADC, SUPPORT")
    nonempty_roles = [role for role in roles if role]
    if len(nonempty_roles) != len(set(nonempty_roles)):
        raise HTTPException(status_code=422, detail="Każda rola może wystąpić tylko raz")

    feature = query_one(
        db,
        "SELECT team_a_golgg_name, team_b_golgg_name FROM upcoming_match_features WHERE canonical_match_id=:match_id",
        {"match_id": match_id},
    ) or {}
    expected_golgg_team = feature.get("team_a_golgg_name") if body.team_side == "a" else feature.get("team_b_golgg_name")
    resolved_players: list[dict[str, str | None]] = []
    seen_ids: set[str] = set()
    for player in body.players:
        resolved = _resolve_roster_player(
            db,
            player.player_id,
            player.player_name,
            player.role,
            expected_golgg_team,
        )
        if resolved["player_id"] in seen_ids:
            raise HTTPException(status_code=422, detail="Zawodnik nie może wystąpić dwa razy w tym samym składzie")
        seen_ids.add(resolved["player_id"])
        resolved_players.append({
            "player_id": resolved["player_id"],
            "player_name": resolved["player_name"],
            "role": str(player.role or "").upper() or None,
        })

    team_name = meta.get("team_a_name") if body.team_side == "a" else meta.get("team_b_name")
    now_iso = datetime.now(UTC).isoformat()
    roster = {
        "team_name": team_name,
        "source": "manual",
        "updated_at": now_iso,
        "source_match_id": None,
        "source_match_date": now_iso,
        "source_tournament": "manual confirmation",
        "players": resolved_players,
    }
    # Keep the canonical override guard and the durable team state in one
    # transaction, including a team's first current-roster insertion.
    roster_team_name = str(expected_golgg_team or team_name)
    lock_current_roster_teams(db, [roster_team_name])
    db.execute(
        text(
            """
            INSERT INTO match_roster_overrides(canonical_match_id, team_side, roster_json, updated_at)
            VALUES (:match_id, :team_side, :roster_json, :updated_at)
            ON CONFLICT (canonical_match_id, team_side)
            DO UPDATE SET roster_json=EXCLUDED.roster_json, updated_at=EXCLUDED.updated_at
            """
        ),
        {
            "match_id": match_id,
            "team_side": body.team_side,
            "roster_json": json.dumps(roster),
            "updated_at": now_iso,
        },
    )
    # Promote a confirmed lineup to the team's durable current roster as well.
    # The per-match override remains the hard guarantee for this exact match;
    # the team roster is the automatic fallback for subsequent fixtures.
    upsert_current_roster(
        db,
        team_name=roster_team_name,
        players=resolved_players,
        source="manual",
        source_match_date=now_iso,
    )
    db.commit()

    return MatchRosterOverrideResponse(
        canonical_match_id=match_id,
        team_side=body.team_side,
        roster=RosterInfo(
            team_name=str(team_name) if team_name else None,
            source_date=now_iso,
            source_tournament="manual confirmation",
            roster_source="manual_override",
            players=[RosterPlayer(**player) for player in resolved_players],
        ),
        message="Skład zapisany jako ręcznie potwierdzony.",
    )


@router.delete("/{match_id}/roster/{team_side}")
def delete_match_roster_override(match_id: int, team_side: str, db=Depends(get_db)):
    """Remove only the match-specific override.

    The selected side then uses the durable current team roster, which may be
    an automatic GOL.GG roster or a newer manual team confirmation.
    """
    if team_side not in {"a", "b"}:
        raise HTTPException(status_code=422, detail="team_side must be 'a' or 'b'")
    db.execute(text("SELECT id FROM canonical_matches WHERE id=:id FOR UPDATE"), {"id": match_id})
    result = db.execute(
        text("DELETE FROM match_roster_overrides WHERE canonical_match_id=:match_id AND team_side=:team_side"),
        {"match_id": match_id, "team_side": team_side},
    )
    db.commit()
    if not result.rowcount:
        raise HTTPException(status_code=404, detail="Brak ręcznego składu do przywrócenia")
    return {"ok": True, "message": "Usunięto override tego meczu; używany jest aktualny skład drużyny."}


# ── POST /matches/{id}/predict — run prediction for single match ────────────


@router.post("/{match_id}/predict", response_model=PredictResponse)
def predict_match(match_id: int, db=Depends(get_db)):
    """Build and store one regional operational prediction for a match.

    The native C0 model is the interactive prediction path. Frozen research
    models remain stored for retrospective, cohort-matched comparison only.
    """
    meta = query_df(db, "SELECT * FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    match = meta[0]
    team_a = match.get("team_a_name")
    team_b = match.get("team_b_name")
    if not team_a or not team_b:
        return PredictResponse(
            status="error",
            message="Match missing team names — cannot predict",
        )

    roster_overrides = _load_roster_overrides(match_id)
    active_hybrid = get_active_hybrid()
    base_model = active_hybrid.base_model
    try:
        prediction = predict_operational_match(
            match,
            model_name=base_model.name,
            model_version=base_model.version,
            feature_version=base_model.feature_version,
            ratings_version=base_model.ratings_version,
            w20_version=base_model.w20_version,
            team_a_roster_override=roster_overrides.get("a"),
            team_b_roster_override=roster_overrides.get("b"),
            persist=False,
            validate_current_inputs=True,
        )
    except ValueError as error:
        return PredictResponse(status="error", message=str(error))

    now_iso = datetime.now(UTC).isoformat()
    try:
        artifact_id = register_operational_model(
            model_name=base_model.name, model_version=base_model.version,
            feature_version=base_model.feature_version, ratings_version=base_model.ratings_version,
            runtime_identity=prediction.get("diagnostics", {}).get("model_identity"),
        )
        try:
            validate_native_serving_inputs(prediction["features"], session=db, lock=True)
        except ValueError as error:
            db.rollback()
            return PredictResponse(status="error", message=f"Prediction skipped: {error}")
        features = prediction["features"]
        upsert_upcoming_features(
            session=db, canonical_match_id=match_id,
            feature_version=base_model.feature_version, ratings_version=base_model.ratings_version,
            data_cutoff_at=None,
            team_a_golgg_name=features.get("mapping", {}).get("team_a_golgg_name"),
            team_b_golgg_name=features.get("mapping", {}).get("team_b_golgg_name"),
            feature_status="ready_player", missing_reason=None, features=features,
        )
        db.execute(
            text("""
            UPDATE canonical_predictions
            SET prediction_status = 'stale'
            WHERE canonical_match_id = :mid AND model_name = :mname AND model_version = :mver
            """),
            {"mid": match_id, "mname": base_model.name, "mver": base_model.version},
        )
        db.execute(
            text("""
            INSERT INTO canonical_predictions(
                canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json,
                prediction_status
            ) VALUES (:mid, :artifact_id, :mname, :mver, :predicted_at, :prob_a, :prob_b, :fver, :rver, :cutoff, :diag, 'active')
            """),
            {
                "mid": match_id,
                "artifact_id": artifact_id,
                "cutoff": prediction["data_cutoff_at"],
                "mname": base_model.name,
                "mver": base_model.version,
                "predicted_at": now_iso,
                "prob_a": prediction["prob_a"],
                "prob_b": prediction["prob_b"],
                "fver": base_model.feature_version,
                "rver": base_model.ratings_version,
                "diag": json.dumps(prediction.get("diagnostics") or {}, ensure_ascii=False, sort_keys=True),
            },
        )
        db.commit()
    except Exception as save_err:
        db.rollback()
        import logging
        logging.getLogger(__name__).warning("Failed to persist base prediction for match %s: %s", match_id, save_err)
        raise HTTPException(status_code=503, detail="Failed to persist prediction") from save_err

    generate_hybrid_predictions(
        canonical_match_id=match_id,
        base_model_name=base_model.name, base_model_version=base_model.version,
        hybrid_model_name=active_hybrid.hybrid_model_name,
        hybrid_model_version=active_hybrid.hybrid_model_version,
        alpha=active_hybrid.alpha, temperature=active_hybrid.temperature,
        blending_mode=active_hybrid.blending_mode,
    )
    hybrid_pred = query_df(
        db,
        """
        SELECT prob_a, prob_b
        FROM canonical_predictions
        WHERE canonical_match_id=:mid AND prediction_status='active'
          AND model_name=:hn AND model_version=:hv
        ORDER BY predicted_at DESC
        LIMIT 1
        """,
        {"mid": match_id, "hn": active_hybrid.hybrid_model_name, "hv": active_hybrid.hybrid_model_version},
    )
    hybrid_prob_a = none_or_float(hybrid_pred[0].get("prob_a")) if hybrid_pred else None
    hybrid_prob_b = none_or_float(hybrid_pred[0].get("prob_b")) if hybrid_pred else None
    return PredictResponse(
        status="ok",
        message=f"Predicted {team_a} vs {team_b}: {prediction['prob_a']:.1%} / {prediction['prob_b']:.1%}",
        prob_a=prediction["prob_a"],
        prob_b=prediction["prob_b"],
        hybrid_prob_a=hybrid_prob_a,
        hybrid_prob_b=hybrid_prob_b,
        model_name=prediction["model_name"],
        model_version=prediction["model_version"],
        diagnostics=prediction["diagnostics"],
    )

@router.post("/{match_id}/simulate-fearless", response_model=SimulateFearlessResponse)
def simulate_fearless_match(
    match_id: int,
    body: SimulateFearlessRequest = SimulateFearlessRequest(),
    db=Depends(get_db),
):
    """Simulate a Hard Fearless Draft series (Bo3 or Bo5) for a match.

    Couples native C0 token masking with champion pool entropy depletion
    and loser side-selection dynamics.
    """
    meta = query_df(db, "SELECT * FROM canonical_matches WHERE id=:id", {"id": match_id})
    if not meta:
        raise HTTPException(status_code=404, detail="Match not found")
    match = meta[0]
    team_a = match.get("team_a_name") or "Team A"
    team_b = match.get("team_b_name") or "Team B"

    # Resolve rosters from overrides, historical match data, or fallback
    roster_overrides = _load_roster_overrides(match_id)
    roster_a_info, roster_b_info = _load_historical_match_rosters(db, match_id, team_a, team_b)

    roster_a_players: list[str] = []
    roster_b_players: list[str] = []

    if roster_overrides.get("a"):
        roster_a_players = [str(p) for p in roster_overrides["a"].values()]
    elif roster_a_info and roster_a_info.players:
        roster_a_players = [p.player_id or p.player_name for p in roster_a_info.players if (p.player_id or p.player_name)]

    if roster_overrides.get("b"):
        roster_b_players = [str(p) for p in roster_overrides["b"].values()]
    elif roster_b_info and roster_b_info.players:
        roster_b_players = [p.player_id or p.player_name for p in roster_b_info.players if (p.player_id or p.player_name)]

    # Ensure 5 players per team
    from src.features.champion_pool_entropy import ROLES
    while len(roster_a_players) < 5:
        roster_a_players.append(f"{team_a}_{ROLES[len(roster_a_players)]}")
    while len(roster_b_players) < 5:
        roster_b_players.append(f"{team_b}_{ROLES[len(roster_b_players)]}")

    # Check for existing prediction to get baseline map probability
    base_map_prob = None
    try:
        pred_row = query_df(
            db,
            "SELECT prob_a FROM canonical_predictions WHERE canonical_match_id=:mid ORDER BY id DESC LIMIT 1",
            {"mid": match_id},
        )
        if pred_row and pred_row[0].get("prob_a") is not None:
            base_map_prob = float(pred_row[0]["prob_a"])
    except Exception:
        base_map_prob = None
    from src.models.fearless_simulator import simulate_fearless_series

    result = simulate_fearless_series(
        team_a=roster_a_players,
        team_b=roster_b_players,
        best_of=body.best_of,
        n_simulations=body.n_simulations,
        seed=body.seed,
        base_map_prob=base_map_prob,
    )

    res_dict = result.to_dict()
    return SimulateFearlessResponse(
        canonical_match_id=match_id,
        team_a_name=team_a,
        team_b_name=team_b,
        best_of=res_dict["best_of"],
        n_simulations=res_dict["n_simulations"],
        series_prob_a=res_dict["series_prob_a"],
        series_prob_b=res_dict["series_prob_b"],
        scorelines=res_dict["scorelines"],
        scoreline_pmf=res_dict["scoreline_pmf"],
        map_win_probabilities=res_dict["map_win_probabilities"],
        map_reach_probabilities=res_dict["map_reach_probabilities"],
        mean_unique_champions=res_dict["mean_unique_champions"],
        champions_drafted_distribution=res_dict["champions_drafted_distribution"],
    )


@router.get("/models/active")
def get_active_model_info():
    """Return the active operational model and hybrid configuration."""
    active = get_active_model()
    hybrid = get_active_hybrid()
    return {
        "status": "ok",
        "active_operational_model": active.identifier(),
        "active_hybrid_model": hybrid.identifier(),
        "operational_model": {
            "name": active.name,
            "display_name": active.effective_display_name,
            "short_code": active.effective_short_code,
            "version": active.version,
            "family": active.family.value if hasattr(active.family, "value") else str(active.family),
            "feature_version": active.feature_version,
            "ratings_version": active.ratings_version,
            "target": active.target.value if hasattr(active.target, "value") else str(active.target),
            "has_epistemic_uncertainty": active.has_epistemic_uncertainty,
        },
        "hybrid_model": {
            "name": hybrid.hybrid_model_name,
            "display_name": hybrid.display_name,
            "short_code": hybrid.short_code,
            "version": hybrid.hybrid_model_version,
            "alpha": hybrid.alpha,
            "temperature": hybrid.temperature,
            "blending_mode": hybrid.blending_mode,
        },
    }


@router.get("/models/registered")
def get_registered_models():
    """List all registered candidate, baseline, and operational models."""
    return {
        "status": "ok",
        "active_operational_model": get_active_model().identifier(),
        "active_hybrid_model": get_active_hybrid().identifier(),
        "models": list_registered_models(),
    }
