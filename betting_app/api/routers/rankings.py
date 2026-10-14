"""Read-only team and player rating leaderboards."""

from __future__ import annotations

import calendar
import json
import math
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, Query

from betting_app.api.deps import get_db, query_df, query_one
from betting_app.api.schemas import RankingEntry, RankingsResponse
from betting_app.services.rating_contract import OPERATIONAL_RATINGS_VERSION

router = APIRouter(tags=["rankings"])

RatingSystem = Literal["unified", "elo", "gl", "ts", "os", "pl", "tm"]
EntityType = Literal["team", "player"]
SquadScope = Literal["major", "regional_academy", "regional", "development", "all", "main"]


def _tier_sql_expr(db, col: str = "state_json") -> str:
    return f"({col}::json->>'tier')"


def _offset_sql_expr(db, col: str = "ro.state_json") -> str:
    return f"COALESCE(({col}::json->>'offset')::float, 0.0)"


def _loc_sigma_sql_expr(db, col: str = "ro.state_json") -> str:
    return f"SQRT(GREATEST(0.0, COALESCE(({col}::json->>'location_variance')::float, 0.0)))"


def _subtract_months(value: date, months: int) -> date:
    target_index = value.year * 12 + value.month - 1 - months
    year, zero_based_month = divmod(target_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _load_regional_metadata(
    db,
    *,
    ratings_version: str,
    entity_type: EntityType,
    normalized_names: list[str],
) -> dict[str, dict[str, object]]:
    if not normalized_names:
        return {}
    params: dict[str, object] = {
        "ratings_version": ratings_version,
        "entity_type": entity_type,
        "gl": "gl",
    }
    placeholders: list[str] = []
    for index, name in enumerate(normalized_names):
        key = f"name_{index}"
        params[key] = name
        placeholders.append(f":{key}")
    states = query_df(
        db,
        f"""
        SELECT normalized_entity_name, state_json
        FROM entity_ratings
        WHERE ratings_version = :ratings_version
          AND entity_type = :entity_type
          AND rating_system = :gl
          AND normalized_entity_name IN ({", ".join(placeholders)})
        """,
        params,
    )
    result: dict[str, dict[str, object]] = {}
    for row in states:
        try:
            state = json.loads(str(row.get("state_json") or "{}"))
            variance = float(state.get("location_variance"))
            offset = float(state.get("offset"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        result[str(row["normalized_entity_name"])] = {
            "region_family": state.get("family"),
            "region_tier": state.get("tier"),
            "regional_offset": offset,
            "regional_uncertainty": math.sqrt(variance),
        }
    return result


@router.get("/rankings", response_model=RankingsResponse)
def get_rankings(
    entity_type: EntityType = "team",
    rating_system: RatingSystem = "unified",
    search: str | None = Query(default=None, max_length=100),
    min_games: int = Query(default=1, ge=0, le=10_000),
    active_within_months: int = Query(default=6, ge=0, le=120),
    squad_scope: SquadScope = "major",
    limit: int = Query(default=100, ge=1, le=500),
    db=Depends(get_db),
) -> RankingsResponse:
    """Return the regional operational leaderboard when its snapshot exists."""
    run = query_one(
        db,
        """
        SELECT ratings_version, data_cutoff_at
        FROM rating_runs
        WHERE status = 'completed'
        ORDER BY
            CASE WHEN ratings_version = :operational_version THEN 0 ELSE 1 END,
            finished_at DESC NULLS LAST,
            id DESC
        LIMIT 1
        """,
        {"operational_version": OPERATIONAL_RATINGS_VERSION},
    )
    if run is None:
        return RankingsResponse(
            entity_type=entity_type,
            rating_system=rating_system,
            total=0,
        )

    cutoff_date = date.fromisoformat(str(run["data_cutoff_at"])[:10])
    active_since = _subtract_months(cutoff_date, active_within_months) if active_within_months else None

    ratings_version = str(run["ratings_version"])
    available_rows = query_df(
        db,
        """
        SELECT DISTINCT rating_system
        FROM entity_ratings
        WHERE ratings_version = :ratings_version
          AND entity_type = :entity_type
          AND rating_value IS NOT NULL
        ORDER BY rating_system
        """,
        {"ratings_version": ratings_version, "entity_type": entity_type},
    )
    available_rating_systems = [str(row["rating_system"]) for row in available_rows]
    params: dict[str, object] = {
        "ratings_version": ratings_version,
        "entity_type": entity_type,
        "rating_system": rating_system,
        "min_games": min_games,
        "limit": limit,
    }
    cohort_filter = ""
    if active_since is not None:
        cohort_filter += " AND last_match_at IS NOT NULL AND SUBSTR(last_match_at, 1, 10) >= :active_since"
        params["active_since"] = active_since.isoformat()
    has_gl_system = "gl" in available_rating_systems
    tier_expr = _tier_sql_expr(db)
    offset_expr = _offset_sql_expr(db, "ro.state_json")
    loc_sigma_expr = _loc_sigma_sql_expr(db, "ro.state_json")
    development_label = """
        (
            LOWER(
                CASE
                    WHEN entity_type = 'team' THEN entity_name
                    ELSE COALESCE(team_name, '')
                END
            ) LIKE '%academy%'
            OR LOWER(
                CASE
                    WHEN entity_type = 'team' THEN entity_name
                    ELSE COALESCE(team_name, '')
                END
            ) LIKE '%challenger%'
            OR LOWER(
                CASE
                    WHEN entity_type = 'team' THEN entity_name
                    ELSE COALESCE(team_name, '')
                END
            ) LIKE '%youth%'
            OR LOWER(
                CASE
                    WHEN entity_type = 'team' THEN entity_name
                    ELSE COALESCE(team_name, '')
                END
            ) LIKE '%junior%'
            OR LOWER(
                CASE
                    WHEN entity_type = 'team' THEN entity_name
                    ELSE COALESCE(team_name, '')
                END
            ) LIKE '%development%'
        )
    """

    if squad_scope in ("major", "main"):
        if has_gl_system:
            cohort_filter += f"""
                AND normalized_entity_name IN (
                    SELECT normalized_entity_name
                    FROM entity_ratings
                    WHERE ratings_version = :ratings_version
                      AND entity_type = :entity_type
                      AND rating_system = 'gl'
                      AND {tier_expr} = 'major'
                )
            """
        else:
            cohort_filter += f" AND NOT {development_label}"
    elif squad_scope == "regional_academy":
        if has_gl_system:
            cohort_filter += f"""
                AND (
                    normalized_entity_name IN (
                        SELECT normalized_entity_name
                        FROM entity_ratings
                        WHERE ratings_version = :ratings_version
                          AND entity_type = :entity_type
                          AND rating_system = 'gl'
                          AND {tier_expr} IN ('regional', 'development')
                    )
                    OR {development_label}
                )
            """
        else:
            cohort_filter += f" AND {development_label}"
    elif squad_scope == "regional":
        if has_gl_system:
            cohort_filter += f"""
                AND normalized_entity_name IN (
                    SELECT normalized_entity_name
                    FROM entity_ratings
                    WHERE ratings_version = :ratings_version
                      AND entity_type = :entity_type
                      AND rating_system = 'gl'
                      AND {tier_expr} = 'regional'
                )
                AND NOT {development_label}
            """
        else:
            cohort_filter += f" AND NOT {development_label}"
    elif squad_scope == "development":
        if has_gl_system:
            cohort_filter += f"""
                AND (
                    normalized_entity_name IN (
                        SELECT normalized_entity_name
                        FROM entity_ratings
                        WHERE ratings_version = :ratings_version
                          AND entity_type = :entity_type
                          AND rating_system = 'gl'
                          AND {tier_expr} = 'development'
                    )
                    OR {development_label}
                )
            """
        else:
            cohort_filter += f" AND {development_label}"
    normalized_search = (search or "").strip().lower()
    search_filter = ""
    if normalized_search:
        search_filter = "WHERE LOWER(entity_name) LIKE :search"
        params["search"] = f"%{normalized_search}%"

    if rating_system == "unified":
        params["system_count"] = len(available_rating_systems)
        if has_gl_system:
            if len(available_rating_systems) > 1:
                params["w_gl"] = 0.80
                params["w_other"] = 0.20 / (len(available_rating_systems) - 1)
            else:
                params["w_gl"] = 1.00
                params["w_other"] = 0.00
            leaderboard_cte = f"""
                WITH regional_offsets AS (
                    SELECT
                        normalized_entity_name AS gl_normalized_name,
                        state_json
                    FROM (
                        SELECT
                            normalized_entity_name,
                            state_json,
                            ROW_NUMBER() OVER (
                                PARTITION BY normalized_entity_name
                                ORDER BY last_match_at DESC, id DESC
                            ) AS rn
                        FROM entity_ratings
                        WHERE ratings_version = :ratings_version
                          AND entity_type = :entity_type
                          AND rating_system = 'gl'
                          AND rating_value IS NOT NULL
                    ) ro_sub
                    WHERE rn = 1
                ),
                calibrated_entities AS (
                    SELECT
                        e.entity_type,
                        e.entity_name,
                        e.normalized_entity_name,
                        e.team_name,
                        e.role,
                        e.rating_system,
                        e.games_played,
                        e.last_match_at,
                        e.snapshot_at,
                        CASE
                            WHEN e.rating_system = 'gl' THEN (e.rating_value - 1.0 * COALESCE(e.rd, 350.0))
                            WHEN e.rating_system = 'elo' THEN (e.rating_value + {offset_expr} - 1.0 * {loc_sigma_expr})
                            WHEN e.rating_system IN ('ts', 'os') THEN (e.rating_value + ({offset_expr} - 1.0 * {loc_sigma_expr}) * (8.333 / 400.0) - 1.0 * COALESCE(e.sigma, 8.333))
                            WHEN e.rating_system IN ('pl', 'tm') THEN (e.rating_value + ({offset_expr} - 1.0 * {loc_sigma_expr}) * (18.75 / 400.0) - 1.0 * COALESCE(e.sigma, 18.75))
                            ELSE e.rating_value
                        END AS effective_score
                    FROM (
                        SELECT *,
                            ROW_NUMBER() OVER (
                                PARTITION BY rating_system, normalized_entity_name
                                ORDER BY last_match_at DESC, id DESC
                            ) AS entity_rn
                        FROM entity_ratings
                        WHERE ratings_version = :ratings_version
                          AND entity_type = :entity_type
                          AND rating_value IS NOT NULL
                          AND games_played >= :min_games
                          {cohort_filter}
                    ) e
                    LEFT JOIN regional_offsets ro
                           ON e.normalized_entity_name = ro.gl_normalized_name
                    WHERE e.entity_rn = 1
                ),
                system_positions AS (
                    SELECT
                        entity_type,
                        entity_name,
                        normalized_entity_name,
                        team_name,
                        role,
                        rating_system,
                        games_played,
                        last_match_at,
                        snapshot_at,
                        RANK() OVER (
                            PARTITION BY rating_system
                            ORDER BY effective_score DESC, games_played DESC, normalized_entity_name ASC
                        ) AS system_rank,
                        COUNT(*) OVER (PARTITION BY rating_system) AS system_total
                    FROM calibrated_entities
                ),
                consensus AS (
                    SELECT
                        entity_type,
                        MAX(entity_name) AS entity_name,
                        normalized_entity_name,
                        MAX(team_name) AS team_name,
                        MAX(role) AS role,
                        'unified' AS rating_system,
                        SUM(
                            (
                                CASE
                                    WHEN system_total = 1 THEN 100.0
                                    ELSE 100.0 * (system_total - system_rank) / (system_total - 1)
                                END
                            ) * (
                                CASE
                                    WHEN rating_system = 'gl' THEN :w_gl
                                    ELSE :w_other
                                END
                            )
                        ) AS rating_value,
                        NULL AS rd,
                        NULL AS sigma,
                        MIN(games_played) AS games_played,
                        MAX(last_match_at) AS last_match_at,
                        MAX(snapshot_at) AS snapshot_at,
                        NULL AS state_json,
                        COUNT(DISTINCT rating_system) AS system_count
                    FROM system_positions
                    GROUP BY entity_type, normalized_entity_name
                    HAVING COUNT(DISTINCT rating_system) = :system_count
                ),
                ranked AS (
                    SELECT
                        ROW_NUMBER() OVER (
                            ORDER BY rating_value DESC, games_played DESC, normalized_entity_name ASC
                        ) AS rank,
                        *
                    FROM consensus
                )
            """
        else:
            leaderboard_cte = f"""
                WITH system_positions AS (
                    SELECT
                        entity_type,
                        entity_name,
                        normalized_entity_name,
                        team_name,
                        role,
                        rating_system,
                        games_played,
                        last_match_at,
                        snapshot_at,
                        RANK() OVER (
                            PARTITION BY rating_system
                            ORDER BY rating_value DESC, games_played DESC, normalized_entity_name ASC
                        ) AS system_rank,
                        COUNT(*) OVER (PARTITION BY rating_system) AS system_total
                    FROM entity_ratings
                    WHERE ratings_version = :ratings_version
                      AND entity_type = :entity_type
                      AND rating_value IS NOT NULL
                      AND games_played >= :min_games
                      {cohort_filter}
                ),
                consensus AS (
                    SELECT
                        entity_type,
                        MAX(entity_name) AS entity_name,
                        normalized_entity_name,
                        MAX(team_name) AS team_name,
                        MAX(role) AS role,
                        'unified' AS rating_system,
                        AVG(
                            CASE
                                WHEN system_total = 1 THEN 100.0
                                ELSE 100.0 * (system_total - system_rank) / (system_total - 1)
                            END
                        ) AS rating_value,
                        NULL AS rd,
                        NULL AS sigma,
                        MIN(games_played) AS games_played,
                        MAX(last_match_at) AS last_match_at,
                        MAX(snapshot_at) AS snapshot_at,
                        NULL AS state_json,
                        COUNT(DISTINCT rating_system) AS system_count
                    FROM system_positions
                    GROUP BY entity_type, normalized_entity_name
                    HAVING COUNT(DISTINCT rating_system) = :system_count
                ),
                ranked AS (
                    SELECT
                        ROW_NUMBER() OVER (
                            ORDER BY rating_value DESC, games_played DESC, normalized_entity_name ASC
                        ) AS rank,
                        *
                    FROM consensus
                )
            """
    else:
        leaderboard_cte = f"""
            WITH ranked AS (
                SELECT
                    ROW_NUMBER() OVER (
                        ORDER BY rating_value DESC, games_played DESC, normalized_entity_name ASC
                    ) AS rank,
                    entity_type,
                    entity_name,
                    normalized_entity_name,
                    team_name,
                    role,
                    rating_system,
                    rating_value,
                    rd,
                    sigma,
                    games_played,
                    last_match_at,
                    snapshot_at,
                    state_json,
                    1 AS system_count
                FROM entity_ratings
                WHERE ratings_version = :ratings_version
                  AND entity_type = :entity_type
                  AND rating_system = :rating_system
                  AND rating_value IS NOT NULL
                  AND games_played >= :min_games
                  {cohort_filter}
            )
        """

    count = query_one(
        db,
        f"{leaderboard_cte} SELECT COUNT(*) AS total FROM ranked {search_filter}",
        params,
    )
    rows = query_df(
        db,
        f"""
        {leaderboard_cte}
        SELECT *
        FROM ranked
        {search_filter}
        ORDER BY rank
        LIMIT :limit
        """,
        params,
    )
    snapshot_at = next((str(row["snapshot_at"]) for row in rows if row.get("snapshot_at")), None)
    region_by_entity = _load_regional_metadata(
        db,
        ratings_version=ratings_version,
        entity_type=entity_type,
        normalized_names=[
            str(row["normalized_entity_name"]) for row in rows
        ],
    )
    rankings: list[RankingEntry] = []
    for row in rows:
        record = dict(row)
        record.pop("state_json", None)
        record.update(
            region_by_entity.get(
                str(record["normalized_entity_name"]), {}
            )
        )
        rankings.append(RankingEntry(**record))

    return RankingsResponse(
        entity_type=entity_type,
        rating_system=rating_system,
        ratings_version=ratings_version,
        data_cutoff_at=run.get("data_cutoff_at"),
        active_since=active_since.isoformat() if active_since else None,
        squad_scope=squad_scope,
        snapshot_at=snapshot_at,
        total=int(count["total"] if count else 0),
        available_rating_systems=available_rating_systems,
        rankings=rankings,
    )
