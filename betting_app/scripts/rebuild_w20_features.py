"""Rebuild leakage-safe W20 team rolling context from GOL.GG SQLite.

This mirrors the rolling-context logic used by the final metamodel experiments:
for each team we maintain a deque of the last N completed games and average:

- win_rate,
- kills/deaths,
- GD@15,
- DPM/VSPM,
- towers/nashors/gold/duration,
- dragons as an extra operational field.

The script stores the *latest* rolling state per team in `team_rolling_features`.
That is the context snapshot needed for upcoming-match inference after the local
GOL.GG cache has been refreshed and imported.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from tqdm import tqdm

from betting_app.core.db import connect, init_db, transaction
from betting_app.core.matching import normalize_team_name


DEFAULT_WINDOW_SIZE = 20
DEFAULT_FEATURE_VERSION = "w20-latest"

DEFAULT_STATS = {
    "win_rate": 0.5,
    "kills": 12.0,
    "deaths": 12.0,
    "gd15": 0.0,
    "dpm": 1800.0,
    "vspm": 7.0,
    "towers": 5.0,
    "dragons": 2.0,
    "nashors": 0.5,
    "gold": 55000.0,
    "duration": 1800.0,
}


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--limit-matches", type=int, default=None, help="Smoke-test only first N chronological matches.")
    args = parser.parse_args()

    init_db()
    stats = rebuild_w20_features(
        window_size=args.window_size,
        feature_version=args.feature_version,
        limit_matches=args.limit_matches,
    )
    print(
        "Rebuilt W20 features:",
        f"version={args.feature_version}",
        f"window={args.window_size}",
        f"matches={stats['matches']}",
        f"games={stats['games']}",
        f"teams={stats['teams']}",
        f"cutoff={stats['data_cutoff_at']}",
    )


def rebuild_w20_features(
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    feature_version: str = DEFAULT_FEATURE_VERSION,
    limit_matches: int | None = None,
) -> dict[str, Any]:
    """Compute latest W20 context per team and persist it."""

    team_history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=window_size))
    team_names: dict[str, str] = {}
    team_last_match: dict[str, str] = {}
    team_match_ids: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=window_size))
    matches = load_matches(limit_matches=limit_matches)

    # Bulk-load all games and player stats upfront (3 queries instead of ~176K)
    games_by_match = load_all_games_grouped()
    player_stats_by_game_side = load_all_player_stats_grouped()

    games_processed = 0

    for match in tqdm(matches, desc=f"Rebuilding W{window_size}"):
        match_id = str(match["match_id"])
        match_date = str(match["date"] or "")
        team1_id = str(match["team1_id"])
        team2_id = str(match["team2_id"])
        team_names[team1_id] = str(match["team1_name"] or team1_id)
        team_names[team2_id] = str(match["team2_name"] or team2_id)
        team_last_match[team1_id] = match_date
        team_last_match[team2_id] = match_date

        games = games_by_match.get(match_id, [])
        for game in games:
            update_team_history(team_history, team_match_ids, team1_id, match_id, game, player_stats_by_game_side)
            update_team_history(team_history, team_match_ids, team2_id, match_id, game, player_stats_by_game_side)
            games_processed += 1

    data_cutoff = max((str(match["date"]) for match in matches if match["date"]), default=datetime.now(UTC).date().isoformat())
    rows = []
    for team_id, history in team_history.items():
        team_name = team_names.get(team_id, team_id)
        averaged = average_history(history)
        rows.append(w20_row(
            feature_version=feature_version,
            team_name=team_name,
            window_size=window_size,
            data_cutoff_at=data_cutoff,
            matches_count=len(set(team_match_ids[team_id])),
            games_count=len(history),
            features=averaged,
            team_id=team_id,
            last_match_at=team_last_match.get(team_id),
        ))

    persist_rows(rows)
    return {"matches": len(matches), "games": games_processed, "teams": len(rows), "data_cutoff_at": data_cutoff}


def load_matches(limit_matches: int | None = None) -> list[dict[str, Any]]:
    """Load matches chronologically from SQLite."""

    query = """
        SELECT match_id, date, team1_id, team2_id, team1_name, team2_name
        FROM golgg_matches
        WHERE date IS NOT NULL
        ORDER BY date ASC, CAST(match_id AS INTEGER) ASC
    """
    if limit_matches:
        query += f" LIMIT {int(limit_matches)}"
    with connect() as connection:
        return [dict(row) for row in connection.execute(query).fetchall()]


def load_all_games_grouped() -> dict[str, list[dict[str, Any]]]:
    """Bulk-load ALL games from SQLite, grouped by match_id.

    Replaces per-match load_games_for_match() to avoid N+1 queries.
    """

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT game_id, match_id, team1_id, team2_id, team1_name, team2_name,
                   team1_win, team2_win, team1_stats_json, team2_stats_json,
                   game_duration
            FROM golgg_games
            ORDER BY CAST(game_id AS INTEGER) ASC
            """
        ).fetchall()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["match_id"])].append(dict(row))
    return grouped


def load_all_player_stats_grouped() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Bulk-load ALL player stats from SQLite, grouped by (game_id, side).

    Replaces per-game-side load_player_stats() to avoid N+1 queries.
    """

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT game_id, side, stats_json
            FROM golgg_game_players
            ORDER BY CASE role WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3 WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 9 END
            """
        ).fetchall()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["game_id"]), str(row["side"]))
        grouped[key].append(json_loads(row["stats_json"]))
    return grouped


def update_team_history(
    team_history: dict[str, deque[dict[str, float]]],
    team_match_ids: dict[str, deque[str]],
    team_id: str,
    match_id: str,
    game: dict[str, Any],
    player_stats_by_game_side: dict[tuple[str, str], list[dict[str, Any]]],
) -> None:
    """Append one completed game to a team's rolling history."""

    is_team_1 = str(game.get("team1_id") or "") == str(team_id)
    stats_key = "team1_stats_json" if is_team_1 else "team2_stats_json"
    win_key = "team1_win" if is_team_1 else "team2_win"
    side = "t1" if is_team_1 else "t2"
    team_stats = json_loads(game.get(stats_key))
    player_stats = player_stats_by_game_side.get((str(game["game_id"]), side), [])
    row = {
        "win": float(bool(game.get(win_key))),
        "kills": sum(safe_player_stat(player, "kills") for player in player_stats),
        "deaths": sum(safe_player_stat(player, "deaths") for player in player_stats),
        "gd15": sum(safe_player_stat(player, "gd@15") for player in player_stats),
        "dpm": sum(safe_player_stat(player, "dpm") for player in player_stats),
        "vspm": sum(safe_player_stat(player, "vspm") for player in player_stats),
        "towers": safe_team_stat(team_stats, "towers"),
        "dragons": safe_team_stat(team_stats, "dragons"),
        "nashors": safe_team_stat(team_stats, "nashors"),
        "gold": safe_team_stat(team_stats, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    team_history[team_id].append(row)
    team_match_ids[team_id].append(match_id)


def average_history(history: deque[dict[str, float]]) -> dict[str, float]:
    """Average one team's W20 history with neutral fallback."""

    if not history:
        return dict(DEFAULT_STATS)
    rows = list(history)
    return {
        "win_rate": mean(row["win"] for row in rows),
        "kills": mean(row["kills"] for row in rows),
        "deaths": mean(row["deaths"] for row in rows),
        "gd15": mean(row["gd15"] for row in rows),
        "dpm": mean(row["dpm"] for row in rows),
        "vspm": mean(row["vspm"] for row in rows),
        "towers": mean(row["towers"] for row in rows),
        "dragons": mean(row["dragons"] for row in rows),
        "nashors": mean(row["nashors"] for row in rows),
        "gold": mean(row["gold"] for row in rows),
        "duration": mean(row["duration"] for row in rows),
    }


def w20_row(
    *,
    feature_version: str,
    team_name: str,
    window_size: int,
    data_cutoff_at: str,
    matches_count: int,
    games_count: int,
    features: dict[str, float],
    team_id: str,
    last_match_at: str | None,
) -> tuple[Any, ...]:
    """Serialize a W20 feature row."""

    payload = dict(features)
    payload.update({"team_id": team_id, "last_match_at": last_match_at})
    return (
        feature_version,
        team_name,
        normalize_team_name(team_name),
        window_size,
        data_cutoff_at,
        matches_count,
        games_count,
        features["win_rate"],
        features["kills"],
        features["deaths"],
        features["gd15"],
        features["dpm"],
        features["vspm"],
        features["gold"],
        features["towers"],
        features["dragons"],
        features["nashors"],
        features["duration"],
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def persist_rows(rows: list[tuple[Any, ...]]) -> None:
    """Persist W20 rows."""

    with transaction() as connection:
        connection.executemany(
            """
            INSERT INTO team_rolling_features(
                feature_version, team_name, normalized_team_name, window_size,
                data_cutoff_at, matches_count, games_count, win_rate, avg_kills,
                avg_deaths, avg_gd15, avg_dpm, avg_vspm, avg_gold, avg_towers,
                avg_dragons, avg_nashors, avg_game_duration, features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_version, normalized_team_name, window_size)
            DO UPDATE SET
                team_name = excluded.team_name,
                data_cutoff_at = excluded.data_cutoff_at,
                matches_count = excluded.matches_count,
                games_count = excluded.games_count,
                win_rate = excluded.win_rate,
                avg_kills = excluded.avg_kills,
                avg_deaths = excluded.avg_deaths,
                avg_gd15 = excluded.avg_gd15,
                avg_dpm = excluded.avg_dpm,
                avg_vspm = excluded.avg_vspm,
                avg_gold = excluded.avg_gold,
                avg_towers = excluded.avg_towers,
                avg_dragons = excluded.avg_dragons,
                avg_nashors = excluded.avg_nashors,
                avg_game_duration = excluded.avg_game_duration,
                features_json = excluded.features_json
            """,
            rows,
        )


def json_loads(value: Any) -> dict[str, Any]:
    """Parse a JSON object safely."""

    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def safe_player_stat(player_stats: dict[str, Any], key: str) -> float:
    """Read player stat with zero fallback."""

    return float(player_stats.get(key) or 0.0)


def safe_team_stat(team_stats: dict[str, Any], key: str) -> float:
    """Read team stat with zero fallback."""

    return float(team_stats.get(key) or 0.0)


def mean(values) -> float:
    """Small dependency-free mean."""

    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


if __name__ == "__main__":
    main()
