"""Service for storing, querying, and analyzing proposition odds history across bookmakers (IDEA-018)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from betting_app.core.db import get_session
from betting_app.models.bookmaker import Bookmaker
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import PropOddsSnapshot
from betting_app.models.prediction import ModelPropPrediction
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.kill_distribution_model import KillDistributionModel
from betting_app.ml.props.team_spread_model import TeamKillsSpreadModel
from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine, compute_novig_two_way
from betting_app.services.canonical_match_service import resolve_canonical_match
from betting_app.services.odds_service import get_or_create_bookmaker


class PropMatchNotFoundError(ValueError):
    """The requested canonical match does not exist."""


def save_prop_snapshots(
    parsed_props: ParsedMatchProps,
    *,
    canonical_match_id: int | None = None,
    league: str | None = None,
    match_start_time: str | None = None,
    source_url: str | None = None,
    scraped_at: datetime | str | None = None,
    db_session: Session | None = None,
) -> list[int]:
    """Persist normalized proposition market snapshots into prop_odds_snapshots."""
    if not parsed_props.lines:
        return []
    actual_source_url = source_url or getattr(parsed_props, "source_url", None)
    bookmaker_name = parsed_props.bookmaker.lower().strip()
    if bookmaker_name == "fortuna":
        bookmaker_name = "efortuna"

    own_session = db_session is None
    sess = db_session if db_session is not None else get_session()

    bookmaker_id: int | None = None
    b_row = sess.execute(
        select(Bookmaker.id).where(Bookmaker.name == bookmaker_name)
    ).scalar_one_or_none()
    if b_row is not None:
        bookmaker_id = int(b_row)
    else:
        bookmaker_id = get_or_create_bookmaker(bookmaker_name)
    if canonical_match_id is None:
        canonical_match_id = resolve_canonical_match(
            raw_team_a=parsed_props.raw_team_a,
            raw_team_b=parsed_props.raw_team_b,
            league=league,
            match_start_time=match_start_time,
            source_system="bookmaker_props",
        )

    # Normalize timestamp
    if scraped_at is None:
        dt_scraped = datetime.now(UTC)
    elif isinstance(scraped_at, str):
        try:
            dt_scraped = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
        except ValueError:
            dt_scraped = datetime.now(UTC)
    else:
        dt_scraped = scraped_at

    inserted_ids: list[int] = []
    try:
        for line in parsed_props.lines:
            odds_o = line.odds_over if line.odds_over is not None else line.odds_cover_a
            odds_u = line.odds_under if line.odds_under is not None else line.odds_cover_b

            # Compute no-vig probabilities if not already present
            p_over = line.novig_prob_over if line.novig_prob_over is not None else line.novig_prob_cover_a
            p_under = line.novig_prob_under if line.novig_prob_under is not None else line.novig_prob_cover_b
            margin = line.margin

            if (p_over is None or p_under is None or margin is None) and odds_o and odds_u:
                p_over, p_under, margin = compute_novig_two_way(odds_o, odds_u)
            payload_str = json.dumps(line.outcomes_payload) if line.outcomes_payload else None

            snap = PropOddsSnapshot(
                bookmaker_id=bookmaker_id,
                canonical_match_id=canonical_match_id,
                market_type=line.market_type,
                map_number=parsed_props.map_number,
                line=float(line.line),
                target_team=line.team or line.handicap_team,
                raw_team_a=parsed_props.raw_team_a,
                raw_team_b=parsed_props.raw_team_b,
                odds_over=odds_o,
                odds_under=odds_u,
                prob_over_novig=p_over,
                prob_under_novig=p_under,
                margin=margin,
                raw_market_name=line.raw_market_name,
                scraped_at=dt_scraped,
                source_url=actual_source_url,
                outcomes_payload=payload_str,
            )
            sess.add(snap)
            sess.flush()
            inserted_ids.append(snap.id)

        if own_session:
            sess.commit()
    except Exception:
        if own_session:
            sess.rollback()
        raise
    finally:
        if own_session:
            sess.close()

    return inserted_ids


def get_prop_odds_timeline(
    canonical_match_id: int,
    *,
    market_type: str | None = None,
    line: float | None = None,
    map_number: int | None = None,
    target_team: str | None = None,
    db_session: Session | None = None,
) -> list[dict[str, Any]]:
    """Return chronological time-series of prop odds snapshots for a canonical match."""
    own_session = db_session is None
    sess = db_session if db_session is not None else get_session()

    try:
        stmt = (
            select(PropOddsSnapshot, Bookmaker.name.label("bookmaker_name"))
            .join(Bookmaker, Bookmaker.id == PropOddsSnapshot.bookmaker_id)
            .where(PropOddsSnapshot.canonical_match_id == canonical_match_id)
        )

        if market_type:
            stmt = stmt.where(PropOddsSnapshot.market_type == market_type)
        if line is not None:
            stmt = stmt.where(PropOddsSnapshot.line == float(line))
        if map_number is not None:
            stmt = stmt.where(PropOddsSnapshot.map_number == int(map_number))
        if target_team:
            stmt = stmt.where(PropOddsSnapshot.target_team == target_team)

        stmt = stmt.order_by(PropOddsSnapshot.scraped_at.asc(), PropOddsSnapshot.id.asc())

        results = sess.execute(stmt).all()
        timeline: list[dict[str, Any]] = []
        for snap, b_name in results:
            timeline.append({
                "id": snap.id,
                "bookmaker": b_name,
                "canonical_match_id": snap.canonical_match_id,
                "market_type": snap.market_type,
                "map_number": snap.map_number,
                "line": snap.line,
                "target_team": snap.target_team,
                "raw_team_a": snap.raw_team_a,
                "raw_team_b": snap.raw_team_b,
                "odds_over": snap.odds_over,
                "odds_under": snap.odds_under,
                "prob_over_novig": snap.prob_over_novig,
                "prob_under_novig": snap.prob_under_novig,
                "margin": snap.margin,
                "raw_market_name": snap.raw_market_name,
                "scraped_at": snap.scraped_at.isoformat() if snap.scraped_at else None,
            })
        return timeline
    finally:
        if own_session:
            sess.close()


def get_latest_prop_odds(
    canonical_match_id: int,
    *,
    market_type: str | None = None,
    map_number: int | None = 1,
    db_session: Session | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent snapshot per bookmaker and line for an upcoming match."""
    timeline = get_prop_odds_timeline(
        canonical_match_id,
        market_type=market_type,
        map_number=map_number,
        db_session=db_session,
    )

    # Group by (bookmaker, market_type, map_number, line, target_team) and take the latest
    latest_dict: dict[tuple[str, str, int, float, str | None], dict[str, Any]] = {}
    for item in timeline:
        key = (
            item["bookmaker"],
            item["market_type"],
            item["map_number"],
            item["line"],
            item["target_team"],
        )
        latest_dict[key] = item

    return list(latest_dict.values())


def evaluate_match_props(
    canonical_match_id: int,
    *,
    tracker: ChronologicalPaceTracker | None = None,
    map_number: int = 1,
    tax_rate: float = 0.12,
    persist_prediction: bool = True,
    db_session: Session | None = None,
) -> dict[str, Any]:
    own_session = db_session is None
    sess = db_session if db_session is not None else get_session()
    try:
        match_row = sess.execute(
            select(CanonicalMatch).where(CanonicalMatch.id == canonical_match_id)
        ).scalar_one_or_none()

        if match_row is None:
            raise PropMatchNotFoundError(f"Canonical match id={canonical_match_id} not found")

        team_a = match_row.team_a_name
        team_b = match_row.team_b_name
        league = match_row.league

        # Fetch latest prop odds across all bookmakers
        latest_lines = get_latest_prop_odds(
            canonical_match_id,
            map_number=map_number,
            db_session=sess,
        )

        # Run statistical models
        ts_model = TeamKillsSpreadModel()
        kd_model = KillDistributionModel()

        if tracker is not None:
            spread_pred = ts_model.predict_from_tracker(tracker, team_a, team_b, league_name=league, tax_rate=tax_rate)
            tot_pred = kd_model.predict_from_tracker(tracker, team_a, team_b, league=league)
            recent_form_a = tracker.get_team_recent_form(team_a, n=5)
            recent_form_b = tracker.get_team_recent_form(team_b, n=5)
            league_ctx = tracker.get_league_context(league)
        else:
            # Fallback to zero-tracker priors
            dummy_tracker = ChronologicalPaceTracker()
            spread_pred = ts_model.predict_from_tracker(dummy_tracker, team_a, team_b, league_name=league, tax_rate=tax_rate)
            tot_pred = kd_model.predict_from_tracker(dummy_tracker, team_a, team_b, league=league)
            recent_form_a = dummy_tracker.get_team_recent_form(team_a, n=5)
            recent_form_b = dummy_tracker.get_team_recent_form(team_b, n=5)
            league_ctx = dummy_tracker.get_league_context(league)

        evaluated_lines: list[dict[str, Any]] = []
        signals: list[dict[str, Any]] = []

        for line_data in latest_lines:
            m_type = line_data["market_type"]
            val_line = float(line_data["line"])
            bookmaker = line_data["bookmaker"]
            target_t = line_data["target_team"]
            odds_o = line_data["odds_over"]
            odds_u = line_data["odds_under"]

            prob_over_model: float | None = None
            prob_under_model: float | None = None

            if m_type == "total_kills":
                ou_pred = tot_pred.get_line(val_line)
                if ou_pred:
                    prob_over_model = ou_pred.prob_over
                    prob_under_model = ou_pred.prob_under
            elif m_type == "team_kills":
                norm_target = (target_t or "").strip().lower()
                if norm_target == team_a.strip().lower():
                    ou_pred = spread_pred.team_a_lines.get(val_line)
                else:
                    ou_pred = spread_pred.team_b_lines.get(val_line)
                if ou_pred:
                    prob_over_model = ou_pred.prob_over
                    prob_under_model = ou_pred.prob_under
            elif m_type == "handicap_kills":
                hp_pred = spread_pred.get_handicap_line(val_line)
                if hp_pred:
                    prob_over_model = hp_pred.prob_cover_a
                    prob_under_model = hp_pred.prob_cover_b

            # Compute Net EV if model probabilities exist
            ev_over_net = None
            ev_under_net = None
            if prob_over_model and odds_o and odds_o > 1.0:
                ev_over_net = round((prob_over_model * odds_o * (1.0 - tax_rate)) - 1.0, 4)
            if prob_under_model and odds_u and odds_u > 1.0:
                ev_under_net = round((prob_under_model * odds_u * (1.0 - tax_rate)) - 1.0, 4)

            eval_item = {
                **line_data,
                "model_prob_over": round(prob_over_model, 4) if prob_over_model else None,
                "model_prob_under": round(prob_under_model, 4) if prob_under_model else None,
                "ev_over_net": ev_over_net,
                "ev_under_net": ev_under_net,
            }
            evaluated_lines.append(eval_item)

            # Signal triggers if Net EV > 0
            if ev_over_net and ev_over_net > 0.0:
                signals.append({
                    "bookmaker": bookmaker,
                    "market_type": m_type,
                    "selection": f"OVER {val_line}" if m_type != "handicap_kills" else f"{team_a} ({val_line:+4.1f})",
                    "odds": odds_o,
                    "model_prob": prob_over_model,
                    "fair_odds": round(1.0 / prob_over_model, 2),
                    "net_ev_tax12": round(ev_over_net * 100, 1),
                    "target_team": target_t,
                })
            if ev_under_net and ev_under_net > 0.0:
                signals.append({
                    "bookmaker": bookmaker,
                    "market_type": m_type,
                    "selection": f"UNDER {val_line}" if m_type != "handicap_kills" else f"{team_b} ({-val_line:+4.1f})",
                    "odds": odds_u,
                    "model_prob": prob_under_model,
                    "fair_odds": round(1.0 / prob_under_model, 2),
                    "net_ev_tax12": round(ev_under_net * 100, 1),
                    "target_team": target_t,
                })

        # Build structured distribution lines for the model's voice
        distribution_lines = []
        for line_val in [20.5, 22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5]:
            ou_p = tot_pred.get_line(line_val)
            if ou_p:
                distribution_lines.append({
                    "line": line_val,
                    "prob_over": round(ou_p.prob_over, 4),
                    "prob_under": round(ou_p.prob_under, 4),
                    "fair_odds_over": round(1.0 / ou_p.prob_over, 2) if ou_p.prob_over > 0 else None,
                    "fair_odds_under": round(1.0 / ou_p.prob_under, 2) if ou_p.prob_under > 0 else None,
                })

        distribution_data = {
            "total_lines": distribution_lines,
            "density_curve": tot_pred.density_curve,
            "quantiles": {
                "p10": tot_pred.quantiles.get("p10"),
                "p25": tot_pred.quantiles.get("p25"),
                "p50": tot_pred.quantiles.get("p50"),
                "p75": tot_pred.quantiles.get("p75"),
                "p90": tot_pred.quantiles.get("p90"),
            },
        }

        saved_prediction_id: int | None = None
        if persist_prediction:
            prop_pred = ModelPropPrediction(
                canonical_match_id=canonical_match_id,
                map_number=map_number,
                model_name="kill_spread_distribution",
                model_version="prop-v1.0",
                expected_total_kills=round(float(spread_pred.spread_features.expected_total_kills), 2),
                mu_team_a=round(float(spread_pred.mu_team_a), 2),
                mu_team_b=round(float(spread_pred.mu_team_b), 2),
                expected_spread=round(float(spread_pred.mu_team_a - spread_pred.mu_team_b), 2),
                league_avg_kills=round(float(league_ctx.avg_kills), 2) if league_ctx.avg_kills else None,
                pace_category=league_ctx.pace_category,
                distribution_json=json.dumps(distribution_data),
                signals_json=json.dumps(signals),
            )
            sess.add(prop_pred)
            sess.flush()
            saved_prediction_id = prop_pred.id
            sess.commit()

        return {
            "canonical_match_id": canonical_match_id,
            "saved_prediction_id": saved_prediction_id,
            "team_a": team_a,
            "team_b": team_b,
            "league": league,
            "map_number": map_number,
            "model_expectations": {
                "expected_total_kills": spread_pred.spread_features.expected_total_kills,
                "mu_team_a": spread_pred.mu_team_a,
                "mu_team_b": spread_pred.mu_team_b,
                "expected_spread_a_minus_b": round(spread_pred.mu_team_a - spread_pred.mu_team_b, 2),
                "league_avg_kills": league_ctx.avg_kills,
                "league_pace_category": league_ctx.pace_category,
                "distribution": distribution_data,
            },
            "evaluated_lines": evaluated_lines,
            "signals": sorted(signals, key=lambda s: s["net_ev_tax12"], reverse=True),
        }
    finally:
        if own_session:
            sess.close()


def get_saved_prop_prediction(
    canonical_match_id: int,
    map_number: int = 1,
    db_session: Session | None = None,
) -> dict[str, Any] | None:
    """Return the most recent saved model prop prediction for a canonical match and map."""
    own_session = db_session is None
    sess = db_session if db_session is not None else get_session()

    try:
        stmt = (
            select(ModelPropPrediction)
            .where(
                ModelPropPrediction.canonical_match_id == canonical_match_id,
                ModelPropPrediction.map_number == map_number,
            )
            .order_by(desc(ModelPropPrediction.predicted_at), desc(ModelPropPrediction.id))
            .limit(1)
        )
        row = sess.execute(stmt).scalar_one_or_none()
        if row is None:
            return None

        dist = json.loads(row.distribution_json) if row.distribution_json else {}
        sigs = json.loads(row.signals_json) if row.signals_json else []

        return {
            "id": row.id,
            "canonical_match_id": row.canonical_match_id,
            "map_number": row.map_number,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "predicted_at": row.predicted_at.isoformat() if row.predicted_at else None,
            "expected_total_kills": row.expected_total_kills,
            "mu_team_a": row.mu_team_a,
            "mu_team_b": row.mu_team_b,
            "expected_spread": row.expected_spread,
            "league_avg_kills": row.league_avg_kills,
            "pace_category": row.pace_category,
            "distribution": dist,
            "signals": sigs,
        }
    finally:
        if own_session:
            sess.close()
