"""Import GOL.GG JSON cache into normalized SQLite tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betting_app.core.config import PROJECT_ROOT
from betting_app.core.db import transaction
from betting_app.core.matching import normalize_team_name, similarity
from src.utils import golgg_schema


DEFAULT_GOLGG_MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"


def load_golgg_json(path: Path = DEFAULT_GOLGG_MATCHES_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the local GOL.GG JSON list.

    The current source file is a large JSON array. We keep this import explicit
    and one-shot; normal app reads should use SQLite after this step.
    """

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload if isinstance(payload, list) else payload.get("matches", []) if isinstance(payload, dict) else []
    rows = [row for row in rows if isinstance(row, dict)]
    return rows[:limit] if limit else rows


def import_golgg_matches(
    path: Path = DEFAULT_GOLGG_MATCHES_PATH,
    *,
    limit: int | None = None,
    batch_size: int = 500,
) -> dict[str, int]:
    """Import GOL.GG matches, games and game-player appearances into SQLite."""

    matches = load_golgg_json(path, limit=limit)
    stats = {"matches": 0, "games": 0, "players": 0, "teams": 0}
    for start in range(0, len(matches), max(1, batch_size)):
        batch = matches[start : start + batch_size]
        batch_stats = import_golgg_batch(batch)
        for key, value in batch_stats.items():
            stats[key] += value
    return stats


def import_golgg_batch(matches: list[dict[str, Any]]) -> dict[str, int]:
    """Import one batch in a single transaction."""

    stats = {"matches": 0, "games": 0, "players": 0, "teams": 0}
    with transaction() as connection:
        for match in matches:
            match_id = str(match.get("match_id") or "").strip()
            if not match_id:
                continue
            connection.execute(
                """
                INSERT INTO golgg_matches(
                    match_id, date, tournament_name, patch, team1_name, team2_name,
                    team1_id, team2_id, team1_score, team2_score, team1_win, team2_win,
                    draw, games_played, best_of, winner_name, loser_name, source_link,
                    raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id) DO UPDATE SET
                    date = excluded.date,
                    tournament_name = excluded.tournament_name,
                    patch = excluded.patch,
                    team1_name = CASE
                        WHEN excluded.team1_id IS NOT NULL AND excluded.team1_id != '' THEN excluded.team1_name
                        WHEN golgg_matches.team1_id IS NOT NULL AND golgg_matches.team1_id != '' THEN golgg_matches.team1_name
                        ELSE COALESCE(NULLIF(excluded.team1_name, ''), golgg_matches.team1_name)
                    END,
                    team2_name = CASE
                        WHEN excluded.team2_id IS NOT NULL AND excluded.team2_id != '' THEN excluded.team2_name
                        WHEN golgg_matches.team2_id IS NOT NULL AND golgg_matches.team2_id != '' THEN golgg_matches.team2_name
                        ELSE COALESCE(NULLIF(excluded.team2_name, ''), golgg_matches.team2_name)
                    END,
                    team1_id = CASE
                        WHEN excluded.team1_id IS NOT NULL AND excluded.team1_id != '' THEN excluded.team1_id
                        ELSE golgg_matches.team1_id
                    END,
                    team2_id = CASE
                        WHEN excluded.team2_id IS NOT NULL AND excluded.team2_id != '' THEN excluded.team2_id
                        ELSE golgg_matches.team2_id
                    END,
                    team1_score = excluded.team1_score,
                    team2_score = excluded.team2_score,
                    team1_win = excluded.team1_win,
                    team2_win = excluded.team2_win,
                    draw = excluded.draw,
                    games_played = excluded.games_played,
                    best_of = excluded.best_of,
                    winner_name = excluded.winner_name,
                    loser_name = excluded.loser_name,
                    source_link = excluded.source_link,
                    raw_json = excluded.raw_json
                """,
                match_row(match, match_id),
            )
            stats["matches"] += 1
            stats["teams"] += upsert_golgg_team(connection, match.get("sname_t1"), match.get("date"))
            stats["teams"] += upsert_golgg_team(connection, match.get("sname_t2"), match.get("date"))

            for game in match.get("games") or []:
                if not isinstance(game, dict):
                    continue
                game_id = str(game.get("game_id") or "").strip()
                if not game_id:
                    continue
                connection.execute(
                    """
                    INSERT INTO golgg_games(
                        game_id, match_id, date, tournament_name, patch, team1_name, team2_name,
                        team1_id, team2_id, team1_win, team2_win, draw, team1_side, team2_side,
                        game_duration, team1_stats_json, team2_stats_json, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(game_id) DO UPDATE SET
                        match_id = excluded.match_id,
                        date = excluded.date,
                        tournament_name = excluded.tournament_name,
                        patch = excluded.patch,
                        team1_name = excluded.team1_name,
                        team2_name = excluded.team2_name,
                        team1_id = excluded.team1_id,
                        team2_id = excluded.team2_id,
                        team1_win = excluded.team1_win,
                        team2_win = excluded.team2_win,
                        draw = excluded.draw,
                        team1_side = excluded.team1_side,
                        team2_side = excluded.team2_side,
                        game_duration = excluded.game_duration,
                        team1_stats_json = excluded.team1_stats_json,
                        team2_stats_json = excluded.team2_stats_json,
                        raw_json = excluded.raw_json
                    """,
                    game_row(game, match_id),
                )
                stats["games"] += 1
                stats["teams"] += upsert_golgg_team(connection, game.get("t1_name"), game.get("date"))
                stats["teams"] += upsert_golgg_team(connection, game.get("t2_name"), game.get("date"))
                stats["players"] += upsert_game_players(connection, game, match_id, game_id)

            games = [g for g in (match.get("games") or []) if isinstance(g, dict) and g.get("game_id")]
            if games:
                g0 = games[0]
                g_t1_id = str(g0.get("team1_id") or g0.get("t1_id") or "").strip()
                g_t2_id = str(g0.get("team2_id") or g0.get("t2_id") or "").strip()
                g_t1_name = str(g0.get("team1_name") or g0.get("t1_name") or "").strip()
                g_t2_name = str(g0.get("team2_name") or g0.get("t2_name") or "").strip()
                if g_t1_id and g_t2_id:
                    m_t1 = str(match.get("sname_t1") or match.get("team1_name") or "")
                    m_t2 = str(match.get("sname_t2") or match.get("team2_name") or "")
                    s_same = similarity(m_t1, g_t1_name) + similarity(m_t2, g_t2_name)
                    s_swap = similarity(m_t1, g_t2_name) + similarity(m_t2, g_t1_name)
                    if s_same >= s_swap:
                        res_t1_name, res_t1_id = g_t1_name, g_t1_id
                        res_t2_name, res_t2_id = g_t2_name, g_t2_id
                    else:
                        res_t1_name, res_t1_id = g_t2_name, g_t2_id
                        res_t2_name, res_t2_id = g_t1_name, g_t1_id
                    connection.execute(
                        """
                        UPDATE golgg_matches
                        SET team1_name = :t1_name, team1_id = :t1_id,
                            team2_name = :t2_name, team2_id = :t2_id
                        WHERE match_id = :mid
                          AND (team1_id IS NULL OR team1_id = '' OR team2_id IS NULL OR team2_id = '')
                        """,
                        {
                            "mid": match_id,
                            "t1_name": res_t1_name,
                            "t1_id": res_t1_id,
                            "t2_name": res_t2_name,
                            "t2_id": res_t2_id,
                        },
                    )
    return stats


def match_row(match: dict[str, Any], match_id: str) -> tuple[Any, ...]:
    """Serialize one match row."""

    match_games = golgg_schema.games(match)
    games_played = len(match_games) if match_games else as_int(match.get("games_played"))

    return (
        match_id,
        match.get("date"),
        golgg_schema.match_tournament(match),
        match.get("patch"),
        golgg_schema.team1_name(match),
        golgg_schema.team2_name(match),
        as_text(golgg_schema.team1_id(match)),
        as_text(golgg_schema.team2_id(match)),
        as_int(golgg_schema.score1(match)),
        as_int(golgg_schema.score2(match)),
        as_bool(golgg_schema.score1(match) > golgg_schema.score2(match)),
        as_bool(golgg_schema.score2(match) > golgg_schema.score1(match)),
        as_bool(match.get("draw")),
        games_played,
        as_int(golgg_schema.best_of(match)),
        match.get("won"),
        match.get("lost"),
        match.get("link"),
        json_dumps(compact_match_raw(match)),
    )


def game_row(game: dict[str, Any], match_id: str) -> tuple[Any, ...]:
    """Serialize one game row."""

    return (
        str(game.get("game_id")),
        match_id,
        game.get("date"),
        game.get("tournament_name"),
        game.get("patch"),
        game.get("t1_name"),
        game.get("t2_name"),
        as_text(game.get("t1_id")),
        as_text(game.get("t2_id")),
        as_bool(game.get("t1_win")),
        as_bool(game.get("t2_win")),
        as_bool(game.get("draw")),
        game.get("t1_side"),
        game.get("t2_side"),
        as_int(game.get("game_duration")),
        json_dumps(game.get("t1_stats") or {}),
        json_dumps(game.get("t2_stats") or {}),
        json_dumps(compact_game_raw(game)),
    )


def upsert_game_players(connection, game: dict[str, Any], match_id: str, game_id: str) -> int:
    """Import player appearances for one game."""

    inserted = 0
    for side, team_id_key, team_name_key, players_key in (
        ("t1", "t1_id", "t1_name", "t1_players"),
        ("t2", "t2_id", "t2_name", "t2_players"),
    ):
        players = game.get(players_key) or {}
        if not isinstance(players, dict):
            continue
        for role, player in players.items():
            if not isinstance(player, dict):
                continue
            connection.execute(
                """
                INSERT INTO golgg_game_players(
                    game_id, match_id, team_id, team_name, side, role, player_id,
                    player_name, champion_id, champion_name, champion_image,
                    stats_json, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id, side, role) DO UPDATE SET
                    match_id = excluded.match_id,
                    team_id = excluded.team_id,
                    team_name = excluded.team_name,
                    player_id = excluded.player_id,
                    player_name = excluded.player_name,
                    champion_id = excluded.champion_id,
                    champion_name = excluded.champion_name,
                    champion_image = excluded.champion_image,
                    stats_json = excluded.stats_json,
                    raw_json = excluded.raw_json
                """,
                (
                    game_id,
                    match_id,
                    as_text(game.get(team_id_key)),
                    game.get(team_name_key),
                    side,
                    str(role),
                    as_text(player.get("player_id") or player.get("id")),
                    player.get("player_name") or player.get("name"),
                    as_text(player.get("champion_id")),
                    player.get("champion_name"),
                    player.get("champion_image"),
                    json_dumps(player.get("stats") or {}),
                    json_dumps(compact_player_raw(player)),
                ),
            )
            inserted += 1
    return inserted


def upsert_golgg_team(connection, team_name: Any, last_seen_at: Any = None) -> int:
    """Upsert team name into existing mapping table."""

    if not isinstance(team_name, str) or not team_name.strip():
        return 0
    connection.execute(
        """
        INSERT INTO golgg_teams(team_name, normalized_name)
        VALUES (?, ?)
        ON CONFLICT(team_name) DO UPDATE SET
            normalized_name = excluded.normalized_name
        """,
        (team_name.strip(), normalize_team_name(team_name)),
    )
    return 1


def as_bool(value: Any) -> int | None:
    """Convert Python bool-ish values to SQLite int."""

    if value is None:
        return None
    return int(bool(value))


def first_present(row: dict[str, Any], *keys: str) -> Any:
    """Return the first non-null/non-empty value without treating 0 as missing."""

    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        return value
    return None


def as_int(value: Any) -> int | None:
    """Convert value to int if possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_text(value: Any) -> str | None:
    """Convert non-empty values to text."""

    if value is None:
        return None
    text = str(value)
    return text if text else None


def json_dumps(value: Any) -> str:
    """Serialize JSON payload compactly."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_match_raw(match: dict[str, Any]) -> dict[str, Any]:
    """Keep match-level raw metadata without duplicating nested games."""

    return {key: value for key, value in match.items() if key != "games"}


def compact_game_raw(game: dict[str, Any]) -> dict[str, Any]:
    """Keep game-level raw metadata without duplicating player maps/stats."""

    omitted = {"t1_players", "t2_players", "t1_stats", "t2_stats"}
    return {key: value for key, value in game.items() if key not in omitted}


def compact_player_raw(player: dict[str, Any]) -> dict[str, Any]:
    """Keep player metadata without duplicating stats_json."""

    return {key: value for key, value in player.items() if key != "stats"}
