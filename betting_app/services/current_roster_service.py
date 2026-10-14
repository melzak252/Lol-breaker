"""Durable current team rosters, shared by GOL.GG ingestion and manual edits."""

from collections.abc import Iterable
import hashlib
import html
import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from betting_app.core.matching import normalize_team_name
from betting_app.core.db import get_session


ROLE_ORDER = {"TOP": 1, "JUNGLE": 2, "MID": 3, "ADC": 4, "SUPPORT": 5}


def lock_current_roster_teams(db: Any, team_names: Iterable[str | None]) -> None:
    """Serialize complete team roster states, including absent rows, until commit.

    Multi-team writers must pass their complete team set before their first
    mutation. Publication takes its canonical row first, then this ordered set.
    SQLite remains available only for single-threaded unit fixtures; it provides
    no advisory-lock or concurrent-publication guarantee.
    """
    normalized_names = sorted({
        normalize_team_name(str(name))
        for name in team_names
        if name is not None and str(name).strip()
    } - {""})
    if not normalized_names or db.get_bind().dialect.name == "sqlite":
        return
    for normalized in normalized_names:
        material = f"team_current_roster_players:{normalized}".encode("utf-8")
        key = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def delete_current_roster(db: Any, team_name: str) -> int:
    """Retract retained current rows under the same insertion/publication lock."""
    lock_current_roster_teams(db, [team_name])
    result = db.execute(
        text("DELETE FROM team_current_roster_players WHERE normalized_team_name=:normalized"),
        {"normalized": normalize_team_name(team_name)},
    )
    return result.rowcount


def load_current_roster(
    team_name: str | None,
    *,
    session: Any = None,
    lock: bool = False,
) -> dict[str, Any] | None:
    """Read the retained current rows, including their actual update clocks."""
    if not team_name:
        return None
    if session is None:
        with get_session() as db:
            return load_current_roster(team_name, session=db, lock=lock)
    if lock:
        lock_current_roster_teams(session, [team_name])
    rows = session.execute(
        text(
            """
            SELECT team_id, team_name, player_id, player_name, role, source,
                   source_match_id, source_game_id, source_match_date, updated_at
            FROM team_current_roster_players
            WHERE normalized_team_name = :normalized
            ORDER BY CASE role
                WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3
                WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 9 END
            """ + (" FOR SHARE" if lock else "")
        ),
        {"normalized": normalize_team_name(team_name)},
    ).mappings().all()
    if len(rows) != 5 or {str(row["role"]).upper() for row in rows} != set(ROLE_ORDER):
        return None
    first = rows[0]
    team_ids = {
        str(row["team_id"]).strip() for row in rows if row["team_id"] not in (None, "")
    }
    clocks = [row["updated_at"] for row in rows if row["updated_at"] is not None]
    latest_update = max(clocks) if clocks else None
    return {
        "basis": "current_stored_roster",
        "team_id": next(iter(team_ids)) if len(team_ids) == 1 else None,
        "team_name": first["team_name"] or team_name,
        "source": first["source"],
        "source_match_id": first["source_match_id"],
        "source_game_id": first["source_game_id"],
        "source_match_date": first["source_match_date"],
        "source_tournament": "manual confirmation" if first["source"] == "manual" else "GOL.GG current roster",
        "updated_at": latest_update.isoformat() if isinstance(latest_update, datetime) else latest_update,
        "players": [
            {
                "player_id": str(row["player_id"] or ""),
                "player_name": row["player_name"],
                "role": row["role"],
                "source": row["source"],
                "source_match_id": row["source_match_id"],
                "source_game_id": row["source_game_id"],
                "source_match_date": row["source_match_date"],
                "updated_at": (
                    row["updated_at"].isoformat() if isinstance(row["updated_at"], datetime)
                    else row["updated_at"]
                ),
            }
            for row in rows
        ],
    }


def clean_player_name(text_val: Any) -> str:
    """Clean HTML entities and whitespace from player names and handles."""
    if not text_val:
        return ""
    val = str(text_val)
    # Double unescape handles cases like &amp;nbsp; or nested entities
    val = html.unescape(html.unescape(val))
    val = val.replace("\xa0", " ").replace("&nbsp;", " ")
    val = re.sub(r"\s+", " ", val).strip()
    return val


def resolve_golgg_player_id(
    db: Any,
    player_name: str,
    expected_team: str | None = None,
) -> tuple[str, str] | None:
    """Resolve a player handle or name to a numeric GOL.GG player_id if available in database."""
    cleaned = clean_player_name(player_name)
    if not cleaned:
        return None
    try:
        # If it's already a numeric GOL.GG ID, verify it exists and return canonical name
        if cleaned.isdigit():
            row = db.execute(
                text(
                    """
                    SELECT player_id, player_name FROM golgg_game_players
                    WHERE player_id = :pid AND player_name IS NOT NULL AND TRIM(player_name) != ''
                    ORDER BY match_id DESC, game_id DESC LIMIT 1
                    """
                ),
                {"pid": cleaned},
            ).mappings().first()
            if row:
                return str(row["player_id"]), str(row["player_name"])
            return cleaned, cleaned

        squashed = re.sub(r"[^a-zA-Z0-9]", "", cleaned).lower()
        # Search 1: exact case-insensitive match on player_name
        rows = db.execute(
            text(
                """
                SELECT player_id, player_name, team_name FROM golgg_game_players
                WHERE LOWER(player_name) = :pname AND player_id IS NOT NULL AND TRIM(player_id) != ''
                ORDER BY
                    CASE WHEN :expected_team IS NOT NULL AND LOWER(COALESCE(team_name, '')) = LOWER(:expected_team) THEN 0 ELSE 1 END,
                    match_id DESC, game_id DESC
                LIMIT 5
                """
            ),
            {"pname": cleaned.lower(), "expected_team": (expected_team or "").lower()},
        ).mappings().fetchall()

        if rows:
            return str(rows[0]["player_id"]), str(rows[0]["player_name"])

        # Search 2: alphanumeric-squashed match (e.g. 'Blind Walker' vs 'BlindWalker')
        if squashed:
            rows = db.execute(
                text(
                    """
                    SELECT player_id, player_name, team_name FROM golgg_game_players
                    WHERE REPLACE(LOWER(player_name), ' ', '') = :squashed
                      AND player_id IS NOT NULL AND TRIM(player_id) != ''
                    ORDER BY
                        CASE WHEN :expected_team IS NOT NULL AND LOWER(COALESCE(team_name, '')) = LOWER(:expected_team) THEN 0 ELSE 1 END,
                        match_id DESC, game_id DESC
                    LIMIT 5
                    """
                ),
                {"squashed": squashed, "expected_team": (expected_team or "").lower()},
            ).mappings().fetchall()
            if rows:
                return str(rows[0]["player_id"]), str(rows[0]["player_name"])
    except Exception:
        if db.get_bind().dialect.name != "sqlite":
            # A PostgreSQL lookup failure must abort the owning transaction,
            # never silently release its canonical/team-state publication locks.
            raise
        if hasattr(db, "rollback"):
            try:
                db.rollback()
            except Exception:
                pass
        return None

    return None
def upsert_current_roster(
    db: Any,
    *,
    team_name: str,
    players: list[dict[str, Any]],
    source: str,
    source_match_id: str | None = None,
    source_game_id: str | None = None,
    source_match_date: str | None = None,
    team_id: str | None = None,
    force: bool = False,
) -> bool:
    """Replace a complete role roster if its source is newer than stored data (or force=True).

    A manual confirmation is timestamped at confirmation time, so delayed
    historical imports cannot undo it.  A genuinely later GOL.GG match does
    replace it, which is exactly the desired automatic behaviour.
    """
    normalized = normalize_team_name(team_name)
    if not normalized:
        return False
    normalized_players = []
    for player in players:
        role = str(player.get("role") or "").upper()
        if role not in ROLE_ORDER:
            continue
        raw_pid = clean_player_name(player.get("player_id"))
        raw_pname = clean_player_name(player.get("player_name"))
        # Determine lookup name and handle
        lookup_name = raw_pid if not raw_pid.isdigit() else raw_pname
        resolved = resolve_golgg_player_id(db, lookup_name or raw_pid, expected_team=team_name)
        if resolved:
            pid, pname = resolved
        else:
            pid = raw_pid or raw_pname
            pname = raw_pname or raw_pid
        if pid:
            normalized_players.append({
                "player_id": str(pid),
                "player_name": pname,
                "role": role,
            })
    if len(normalized_players) != 5 or len({p["role"] for p in normalized_players}) != 5:
        return False
    lock_current_roster_teams(db, [team_name])
    stamp = source_match_date or datetime.now(UTC).isoformat()
    current = db.execute(
        text(
            """
            SELECT source_match_date FROM team_current_roster_players
            WHERE normalized_team_name=:normalized
            ORDER BY source_match_date DESC NULLS LAST LIMIT 1
            """
        ),
        {"normalized": normalized},
    ).mappings().first()
    if not force and current and current.get("source_match_date") and str(current["source_match_date"]) > stamp:
        return False

    now = datetime.now(UTC).isoformat()
    for player in normalized_players:
        db.execute(
            text(
                """
                INSERT INTO team_current_roster_players(
                    team_id, team_name, normalized_team_name, player_id, player_name,
                    role, source, source_match_id, source_game_id, source_match_date, updated_at
                ) VALUES (
                    :team_id, :team_name, :normalized, :player_id, :player_name,
                    :role, :source, :source_match_id, :source_game_id, :source_match_date, :updated_at
                )
                ON CONFLICT (normalized_team_name, role) DO UPDATE SET
                    team_id=EXCLUDED.team_id, team_name=EXCLUDED.team_name,
                    player_id=EXCLUDED.player_id, player_name=EXCLUDED.player_name,
                    source=EXCLUDED.source, source_match_id=EXCLUDED.source_match_id,
                    source_game_id=EXCLUDED.source_game_id, source_match_date=EXCLUDED.source_match_date,
                    updated_at=EXCLUDED.updated_at
                """
            ),
            {**player, "team_id": team_id, "team_name": team_name, "normalized": normalized,
             "source": source, "source_match_id": source_match_id, "source_game_id": source_game_id,
             "source_match_date": stamp, "updated_at": now},
        )
    return True
