"""FastAPI router for player search, profiles, rating trajectories, and comparisons."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from betting_app.api.deps import get_db
from betting_app.api.schemas import (
    PlayerComparisonResponse,
    PlayerProfileDetail,
    PlayerSearchItem,
    RatingTimelinePoint,
)
from betting_app.services.player_comparison_service import (
    compare_players,
    get_player_profile,
    get_player_rating_history,
    search_players,
)

router = APIRouter(prefix="/players", tags=["players"])


@router.get("/search", response_model=list[PlayerSearchItem])
def search_players_endpoint(
    query: str = Query(default="", min_length=1, max_length=100),
    limit: int = Query(default=15, ge=1, le=50),
    db: Session = Depends(get_db),
) -> list[PlayerSearchItem]:
    """Search professional players by nickname or ID."""
    return search_players(db, query=query, limit=limit)


@router.get("/compare", response_model=PlayerComparisonResponse)
def compare_players_endpoint(
    player_a: str = Query(description="Player A ID or nickname"),
    player_b: str = Query(description="Player B ID or nickname"),
    rating_system: str = Query(default="unified", description="Rating system for probability weighting"),
    db: Session = Depends(get_db),
) -> PlayerComparisonResponse:
    """Compare two players: multi-system ratings, head-to-head records, timeline, and model verdict."""
    clean_a = player_a.strip()
    clean_b = player_b.strip()
    if not clean_a or not clean_b:
        raise HTTPException(status_code=400, detail="Both player_a and player_b parameters are required")

    # If parameters might be player names rather than IDs, resolve to ID if needed
    id_a = _resolve_to_player_id(db, clean_a)
    id_b = _resolve_to_player_id(db, clean_b)

    result = compare_players(db, player_a_id=id_a, player_b_id=id_b, selected_system=rating_system)
    if not result:
        raise HTTPException(status_code=404, detail=f"One or both players not found: '{player_a}', '{player_b}'")
    return result


@router.get("/{player_id}", response_model=PlayerProfileDetail)
def get_player_profile_endpoint(
    player_id: str,
    db: Session = Depends(get_db),
) -> PlayerProfileDetail:
    """Get full rating profile and career stats for a player."""
    resolved_id = _resolve_to_player_id(db, player_id)
    profile = get_player_profile(db, resolved_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")
    return profile


@router.get("/{player_id}/history", response_model=list[RatingTimelinePoint])
def get_player_history_endpoint(
    player_id: str,
    limit: int = Query(default=250, ge=10, le=2000),
    db: Session = Depends(get_db),
) -> list[RatingTimelinePoint]:
    """Get chronological rating trajectory for a player across all rating systems."""
    resolved_id = _resolve_to_player_id(db, player_id)
    return get_player_rating_history(resolved_id, max_points=limit)


def _resolve_to_player_id(db: Session, identifier: str) -> str:
    """Resolve a nickname or player_id to the canonical normalized player_id."""
    clean = identifier.strip()
    # Check if identifier is already a numeric player_id
    if clean.isdigit():
        return clean

    # Search by exact name in entity_ratings
    from betting_app.api.deps import query_one

    row = query_one(
        db,
        """
        SELECT normalized_entity_name AS player_id
        FROM entity_ratings
        WHERE entity_type = 'player'
          AND LOWER(entity_name) = LOWER(:name)
        ORDER BY games_played DESC
        LIMIT 1
        """,
        {"name": clean},
    )
    if row and row.get("player_id"):
        return str(row["player_id"])

    # Search in golgg_game_players
    row_gp = query_one(
        db,
        """
        SELECT player_id
        FROM golgg_game_players
        WHERE LOWER(player_name) = LOWER(:name) AND player_id IS NOT NULL
        LIMIT 1
        """,
        {"name": clean},
    )
    if row_gp and row_gp.get("player_id"):
        return str(row_gp["player_id"])

    return clean
