#!/usr/bin/env python3
"""Build and index historical rating trajectories for all players.

This script replays GOL.GG match history chronologically through RatingManager
and persists player rating trajectories into a local SQLite database:
`data/player_rating_history.sqlite`.

The resulting database enables sub-millisecond querying of rating evolution
over time for any player.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path
import sqlite3
import sys
import time

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import create_engine, text
from betting_app.core.db import database_url
from betting_app.scripts.rebuild_ratings import RATING_SYSTEM_PARAMS
from src.ratings.manager import RatingManager


def build_player_rating_history(
    output_path: Path | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    output_path = output_path or (PROJECT_ROOT / "data" / "player_rating_history.sqlite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db_url = database_url()
    if verbose:
        print(f"Connecting to database...")
    engine = create_engine(db_url)

    t0 = time.time()
    with engine.connect() as conn:
        limit_clause = f"LIMIT {int(limit)}" if limit else ""
        if verbose:
            print("Loading matches, games, and rosters in bulk...")
        matches_rows = conn.execute(text(f"""
            SELECT match_id, date, team1_id, team2_id, team1_name, team2_name
            FROM golgg_matches
            WHERE COALESCE(draw, 0) = 0 AND date IS NOT NULL
            ORDER BY date ASC, match_id ASC
            {limit_clause}
        """)).fetchall()

        games_rows = conn.execute(text("""
            SELECT game_id, match_id, team1_win, team2_win
            FROM golgg_games
            ORDER BY game_id ASC
        """)).fetchall()

        players_rows = conn.execute(text("""
            SELECT match_id, side, player_id, player_name
            FROM golgg_game_players
            WHERE player_id IS NOT NULL
        """)).fetchall()

    if verbose:
        print(
            f"Loaded {len(matches_rows)} matches, {len(games_rows)} games, "
            f"{len(players_rows)} player rosters in {time.time() - t0:.2f}s"
        )

    # Index games by match_id
    games_by_match: dict[str, list[int]] = defaultdict(list)
    for gid, mid, t1w, t2w in games_rows:
        if t1w is not None and t2w is not None:
            games_by_match[str(mid)].append(1 if t1w else 0)

    # Index rosters by match_id and side
    players_by_match: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"t1": [], "t2": []})
    player_display_names: dict[str, str] = {}
    for mid, side, pid, pname in players_rows:
        sid = str(side).lower()
        mid_str = str(mid)
        pid_str = str(pid)
        if sid in ("t1", "t2") and pid_str not in players_by_match[mid_str][sid]:
            players_by_match[mid_str][sid].append(pid_str)
        if pid_str and pname and pid_str not in player_display_names:
            player_display_names[pid_str] = str(pname)

    # Temporary SQLite setup
    temp_db_path = output_path.with_suffix(".tmp")
    if temp_db_path.exists():
        temp_db_path.unlink()

    sqlite_conn = sqlite3.connect(str(temp_db_path))
    sqlite_conn.execute("PRAGMA synchronous = OFF")
    sqlite_conn.execute("PRAGMA journal_mode = MEMORY")
    sqlite_conn.execute("""
        CREATE TABLE player_rating_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            date TEXT NOT NULL,
            match_id TEXT NOT NULL,
            team_name TEXT,
            games_count INTEGER NOT NULL,
            elo REAL NOT NULL,
            gl REAL NOT NULL,
            gl_rd REAL NOT NULL,
            ts_mu REAL NOT NULL,
            os_mu REAL NOT NULL,
            pl_mu REAL NOT NULL,
            tm_mu REAL NOT NULL
        )
    """)

    manager = RatingManager(RATING_SYSTEM_PARAMS)
    player_games_count: dict[str, int] = defaultdict(int)

    insert_batch: list[tuple] = []
    total_inserted = 0
    t_replay_start = time.time()

    for idx, (mid, mdate_str, t1_id, t2_id, t1_name, t2_name) in enumerate(matches_rows):
        mid_str = str(mid)
        scores = games_by_match.get(mid_str)
        if not scores:
            continue
        rosters = players_by_match.get(mid_str)
        if not rosters or not rosters["t1"] or not rosters["t2"]:
            continue
        p1 = rosters["t1"]
        p2 = rosters["t2"]

        try:
            mdate = date.fromisoformat(str(mdate_str)[:10])
        except Exception:
            continue

        team1_key = str(t1_id or t1_name or "team1")
        team2_key = str(t2_id or t2_name or "team2")

        manager.update_before_match(team1_key, team2_key, p1, p2, mdate)
        for s1 in scores:
            manager.update_after_game(team1_key, team2_key, p1, p2, s1, 1 - s1)
        manager.update_after_match(team1_key, team2_key, p1, p2, scores)

        # Record history for participants
        n_games = len(scores)
        for pid in p1:
            player_games_count[pid] += n_games
            name = player_display_names.get(pid, pid)
            insert_batch.append((
                pid,
                name,
                str(mdate_str)[:10],
                mid_str,
                str(t1_name or ""),
                player_games_count[pid],
                round(float(manager.systems["elo"].get_player_rating(pid)), 2),
                round(float(manager.systems["gl"].get_player_rating(pid).rating), 2),
                round(float(manager.systems["gl"].get_player_rating(pid).rd), 2),
                round(float(manager.systems["ts"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["os"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["pl"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["tm"].get_player_rating(pid).mu), 2),
            ))

        for pid in p2:
            player_games_count[pid] += n_games
            name = player_display_names.get(pid, pid)
            insert_batch.append((
                pid,
                name,
                str(mdate_str)[:10],
                mid_str,
                str(t2_name or ""),
                player_games_count[pid],
                round(float(manager.systems["elo"].get_player_rating(pid)), 2),
                round(float(manager.systems["gl"].get_player_rating(pid).rating), 2),
                round(float(manager.systems["gl"].get_player_rating(pid).rd), 2),
                round(float(manager.systems["ts"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["os"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["pl"].get_player_rating(pid).mu), 2),
                round(float(manager.systems["tm"].get_player_rating(pid).mu), 2),
            ))

        if len(insert_batch) >= 10000:
            sqlite_conn.executemany("""
                INSERT INTO player_rating_history (
                    player_id, player_name, date, match_id, team_name, games_count,
                    elo, gl, gl_rd, ts_mu, os_mu, pl_mu, tm_mu
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_batch)
            sqlite_conn.commit()
            total_inserted += len(insert_batch)
            insert_batch.clear()

        if verbose and (idx + 1) % 5000 == 0:
            elapsed = time.time() - t_replay_start
            rate = (idx + 1) / elapsed
            print(f"Replayed {idx + 1}/{len(matches_rows)} matches ({rate:.1f} matches/s)...")

    if insert_batch:
        sqlite_conn.executemany("""
            INSERT INTO player_rating_history (
                player_id, player_name, date, match_id, team_name, games_count,
                elo, gl, gl_rd, ts_mu, os_mu, pl_mu, tm_mu
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_batch)
        sqlite_conn.commit()
        total_inserted += len(insert_batch)

    if verbose:
        print(f"Creating database indexes on {total_inserted} rows...")
    sqlite_conn.execute("CREATE INDEX idx_prh_player_id ON player_rating_history(player_id, date)")
    sqlite_conn.execute("CREATE INDEX idx_prh_player_name ON player_rating_history(player_name, date)")
    sqlite_conn.execute("CREATE INDEX idx_prh_date ON player_rating_history(date)")
    sqlite_conn.commit()
    sqlite_conn.close()

    temp_db_path.replace(output_path)
    if verbose:
        total_time = time.time() - t0
        print(f"Done! Persisted {total_inserted} history records into {output_path} in {total_time:.1f}s.")

    return total_inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Process first N matches only")
    args = parser.parse_args()
    build_player_rating_history(limit=args.limit)
