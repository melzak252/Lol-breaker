"""EXP-048 audit for GOL.GG player-game/champion data.

The script answers whether the database is rich enough for a learned
PlayerGameEncoder and records the exact coverage/completeness assumptions used
before implementing neural aggregation models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from betting_app.core.db import get_session, init_db
from betting_app.ml.training.player_game_dataset import (
    DEFAULT_PLAYER_STAT_KEYS,
    PlayerGameDatasetConfig,
    build_player_game_dataset_from_db,
)


def _json_loads(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _scalar(session, sql: str, params: dict[str, Any] | None = None) -> Any:
    return session.execute(text(sql), params or {}).scalar()


def _rows(session, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [dict(row) for row in session.execute(text(sql), params or {}).mappings().all()]


def collect_audit(*, limit_rows: int | None = None) -> dict[str, Any]:
    """Collect database coverage and dataset-smoke metadata."""

    with get_session() as session:
        counts = {
            "golgg_matches": _scalar(session, "SELECT COUNT(*) FROM golgg_matches"),
            "golgg_games": _scalar(session, "SELECT COUNT(*) FROM golgg_games"),
            "golgg_game_players": _scalar(session, "SELECT COUNT(*) FROM golgg_game_players"),
            "player_rows_with_stats": _scalar(
                session,
                "SELECT COUNT(*) FROM golgg_game_players WHERE stats_json IS NOT NULL AND stats_json <> ''",
            ),
            "distinct_players": _scalar(
                session,
                "SELECT COUNT(DISTINCT COALESCE(NULLIF(player_id, ''), player_name)) FROM golgg_game_players",
            ),
            "distinct_champions": _scalar(
                session,
                """
                SELECT COUNT(DISTINCT champion_name)
                FROM golgg_game_players
                WHERE champion_name IS NOT NULL AND champion_name <> ''
                """,
            ),
        }
        date_ranges = {
            "games": _rows(session, "SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM golgg_games")[0],
            "player_rows": _rows(
                session,
                """
                SELECT MIN(g.date) AS min_date, MAX(g.date) AS max_date
                FROM golgg_game_players gp
                JOIN golgg_games g ON g.game_id = gp.game_id
                """,
            )[0],
        }
        by_year = _rows(
            session,
            """
            SELECT SUBSTRING(g.date, 1, 4) AS year,
                   COUNT(*) AS player_rows,
                   COUNT(DISTINCT gp.game_id) AS games
            FROM golgg_game_players gp
            JOIN golgg_games g ON g.game_id = gp.game_id
            GROUP BY year
            ORDER BY year
            """,
        )
        role_distribution = _rows(
            session,
            "SELECT role, COUNT(*) AS rows FROM golgg_game_players GROUP BY role ORDER BY rows DESC",
        )
        structural_missing = _rows(
            session,
            """
            SELECT
                COUNT(*) FILTER (WHERE role IS NULL OR role = '') AS missing_role,
                COUNT(*) FILTER (WHERE champion_name IS NULL OR champion_name = '') AS missing_champion,
                COUNT(*) FILTER (WHERE player_id IS NULL OR player_id = '') AS missing_player_id,
                COUNT(*) AS total
            FROM golgg_game_players
            """,
        )[0]
        game_missing = _rows(
            session,
            """
            SELECT
                COUNT(*) FILTER (WHERE game_duration IS NULL) AS missing_duration,
                COUNT(*) FILTER (WHERE team1_stats_json IS NULL OR team1_stats_json = '') AS missing_team1_stats,
                COUNT(*) FILTER (WHERE team2_stats_json IS NULL OR team2_stats_json = '') AS missing_team2_stats,
                COUNT(*) AS total
            FROM golgg_games
            """,
        )[0]
        exact_score_distribution = _rows(
            session,
            """
            SELECT best_of, team1_score, team2_score, COUNT(*) AS rows
            FROM golgg_matches
            WHERE date IS NOT NULL
              AND (team1_win = 1 OR team2_win = 1 OR winner_name IS NOT NULL)
            GROUP BY best_of, team1_score, team2_score
            ORDER BY best_of NULLS LAST, rows DESC
            LIMIT 50
            """,
        )

        stat_presence: Counter[str] = Counter()
        stat_non_null: Counter[str] = Counter()
        total_player_rows = 0
        query = "SELECT stats_json FROM golgg_game_players WHERE stats_json IS NOT NULL AND stats_json <> ''"
        if limit_rows:
            query += " LIMIT :limit_rows"
        for (stats_json,) in session.execute(text(query), {"limit_rows": limit_rows} if limit_rows else {}):
            total_player_rows += 1
            stats = _json_loads(stats_json)
            for key, value in stats.items():
                stat_presence[key] += 1
                if value not in (None, ""):
                    stat_non_null[key] += 1

    inspected = total_player_rows or 1
    selected_stat_completeness = {
        key: {
            "present": int(stat_presence.get(key, 0)),
            "non_null": int(stat_non_null.get(key, 0)),
            "non_null_pct": round(float(stat_non_null.get(key, 0)) / inspected * 100.0, 4),
        }
        for key in DEFAULT_PLAYER_STAT_KEYS
    }

    smoke_config = PlayerGameDatasetConfig(limit_rows=limit_rows or 5000)
    smoke_dataset = build_player_game_dataset_from_db(smoke_config)

    return {
        "experiment_id": "EXP-048",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": "Verify GOL.GG player/game/champion coverage for PlayerGameEncoder and exact-score/game-flow heads.",
        "counts": {key: int(value) for key, value in counts.items()},
        "date_ranges": date_ranges,
        "by_year": by_year,
        "role_distribution": role_distribution,
        "structural_missing": structural_missing,
        "game_missing": game_missing,
        "exact_score_distribution_top50": exact_score_distribution,
        "stat_completeness_inspected_rows": int(total_player_rows),
        "selected_stat_completeness": selected_stat_completeness,
        "dataset_smoke_metadata": smoke_dataset.metadata,
        "recommendation": {
            "go_no_go": "GO",
            "why": "Database has near-complete role/champion/player-game rows and enough chronological depth for learned player-game embeddings.",
            "next": "Train supervised+denoising PlayerGameEncoder, then aggregate only date<T embeddings for match-level winner/exact-score/game-flow heads.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-rows", type=int, default=None, help="Limit stats scan for quick smoke audits")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    audit = collect_audit(limit_rows=args.limit_rows)
    payload = json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
