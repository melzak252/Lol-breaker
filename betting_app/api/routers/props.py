"""FastAPI router for proposition markets (kill totals, team kills, kill handicaps) - IDEA-018."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from betting_app.api.deps import get_db
from betting_app.api.schemas import (
    IngestPropOddsRequest,
    MatchPropAnalysisResponse,
    PropOddsLatestResponse,
    PropOddsSnapshotItem,
    PropOddsTimelineResponse,
)
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.scrapers.props_parser import ParsedMatchProps, ParsedPropLine
from betting_app.services.prop_odds_service import (
    PropMatchNotFoundError,
    evaluate_match_props,
    get_latest_prop_odds,
    get_prop_odds_timeline,
    save_prop_snapshots,
)

router = APIRouter(prefix="/matches/{canonical_match_id}/props", tags=["props"])

# Module-level cached tracker instance to avoid recomputing on every request
_CACHED_TRACKER: ChronologicalPaceTracker | None = None


def get_cached_pace_tracker() -> ChronologicalPaceTracker:
    """Prime shared pace history; an empty query uses priors, SQL errors propagate."""
    global _CACHED_TRACKER
    if _CACHED_TRACKER is None:
        from sqlalchemy import text
        from betting_app.core.db import get_session

        sess = get_session()
        try:
            # Use the session connection so the history belongs to the same
            # transaction/database selected by the caller.
            games = sess.execute(
                text("""
                    SELECT team1_name, team2_name, tournament_name,
                           (team1_stats_json::jsonb->>'kills')::int AS kills_t1,
                           (team2_stats_json::jsonb->>'kills')::int AS kills_t2,
                           game_duration, team1_win
                    FROM golgg_games
                    WHERE game_duration IS NOT NULL
                      AND date >= '2024-01-01'
                      AND team1_stats_json IS NOT NULL
                      AND team2_stats_json IS NOT NULL
                    ORDER BY date ASC, game_id ASC
                """)
            )
            tracker = ChronologicalPaceTracker(window_size=20, prior_weight=5.0)
            for row in games:
                tracker.update_game(
                    team_1=row.team1_name,
                    team_2=row.team2_name,
                    kills_1=row.kills_t1,
                    kills_2=row.kills_t2,
                    duration_minutes=float(row.game_duration) / 60.0 if row.game_duration else 32.0,
                    winner_team_1=bool(row.team1_win == 1),
                )
            _CACHED_TRACKER = tracker
        finally:
            sess.close()
    return _CACHED_TRACKER


@router.get("/timeline", response_model=PropOddsTimelineResponse)
def get_props_timeline(
    canonical_match_id: int,
    market_type: str | None = Query(None, description="Filter by market: total_kills, team_kills, handicap_kills, duration"),
    line: float | None = Query(None, description="Filter by specific line, e.g. 26.5"),
    map_number: int | None = Query(1, description="Map number (1, 2, 3...)"),
    target_team: str | None = Query(None, description="Target team for team-specific markets"),
    db: Session = Depends(get_db),
) -> PropOddsTimelineResponse:
    """Return historical time-series of odds movement for a match proposition market."""
    timeline = get_prop_odds_timeline(
        canonical_match_id,
        market_type=market_type,
        line=line,
        map_number=map_number,
        target_team=target_team,
        db_session=db,
    )
    return PropOddsTimelineResponse(
        canonical_match_id=canonical_match_id,
        market_type=market_type,
        line=line,
        total_snapshots=len(timeline),
        timeline=[PropOddsSnapshotItem(**item) for item in timeline],
    )


@router.get("/latest", response_model=PropOddsLatestResponse)
def get_props_latest(
    canonical_match_id: int,
    market_type: str | None = Query(None, description="Optional market type filter"),
    map_number: int = Query(1, description="Map number (1, 2, 3...)"),
    db: Session = Depends(get_db),
) -> PropOddsLatestResponse:
    """Return current cross-bookmaker proposition odds for an upcoming match."""
    latest_lines = get_latest_prop_odds(
        canonical_match_id,
        market_type=market_type,
        map_number=map_number,
        db_session=db,
    )
    return PropOddsLatestResponse(
        canonical_match_id=canonical_match_id,
        map_number=map_number,
        total_lines=len(latest_lines),
        lines=[PropOddsSnapshotItem(**item) for item in latest_lines],
    )


@router.get("/analysis", response_model=MatchPropAnalysisResponse)
def get_props_analysis(
    canonical_match_id: int,
    map_number: int = Query(1, description="Map number (default 1)"),
    tax_rate: float = Query(0.12, ge=0.0, lt=1.0, description="Betting tax rate (default 0.12 for Polish regulated market)"),
    persist: bool = Query(True, description="Whether to persist generated prediction in model_prop_predictions"),
    db: Session = Depends(get_db),
) -> MatchPropAnalysisResponse:
    """Evaluate current bookmaker prop lines against statistical pace and spread models to detect positive Net EV."""
    tracker = get_cached_pace_tracker()
    try:
        analysis = evaluate_match_props(
            canonical_match_id,
            tracker=tracker,
            map_number=map_number,
            tax_rate=tax_rate,
            persist_prediction=persist,
            db_session=db,
        )
        return MatchPropAnalysisResponse(**analysis)
    except PropMatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/predict", response_model=MatchPropAnalysisResponse)
def predict_props_and_save(
    canonical_match_id: int,
    map_number: int = Query(1, description="Map number (default 1)"),
    tax_rate: float = Query(0.12, ge=0.0, lt=1.0, description="Betting tax rate (default 0.12)"),
    db: Session = Depends(get_db),
) -> MatchPropAnalysisResponse:
    """Explicitly generate and persist new statistical model prop predictions in database."""
    tracker = get_cached_pace_tracker()
    try:
        analysis = evaluate_match_props(
            canonical_match_id,
            tracker=tracker,
            map_number=map_number,
            tax_rate=tax_rate,
            persist_prediction=True,
            db_session=db,
        )
        return MatchPropAnalysisResponse(**analysis)
    except PropMatchNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/ingest")
def ingest_props(
    canonical_match_id: int,
    req: IngestPropOddsRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Manually or automated ingestion of parsed bookmaker proposition lines for a match."""
    parsed_lines = [
        ParsedPropLine(
            market_type=item.get("market_type", "total_kills"),
            line=float(item.get("line", 0.0)),
            team=item.get("team") or item.get("target_team"),
            odds_over=item.get("odds_over"),
            odds_under=item.get("odds_under"),
            odds_cover_a=item.get("odds_cover_a"),
            odds_cover_b=item.get("odds_cover_b"),
            novig_prob_over=item.get("prob_over_novig"),
            novig_prob_under=item.get("prob_under_novig"),
            margin=item.get("margin"),
            raw_market_name=item.get("raw_market_name"),
        )
        for item in req.lines
    ]
    parsed_props = ParsedMatchProps(
        bookmaker=req.bookmaker,
        raw_team_a=req.raw_team_a,
        raw_team_b=req.raw_team_b,
        map_number=req.map_number,
        lines=parsed_lines,
    )
    inserted_ids = save_prop_snapshots(
        parsed_props,
        canonical_match_id=canonical_match_id,
        league=req.league,
        match_start_time=req.match_start_time,
        source_url=req.source_url,
        db_session=db,
    )
    return {
        "status": "success",
        "canonical_match_id": canonical_match_id,
        "bookmaker": req.bookmaker,
        "inserted_count": len(inserted_ids),
        "snapshot_ids": inserted_ids,
    }
