"""Player-game dataset utilities for learned LoL embeddings.

The EXP-048/EXP-049+ modelling direction starts with an encoder that turns one
completed player game into a compact representation.  Pre-match models can then
aggregate only embeddings from games that happened before the predicted match.

This module intentionally builds *completed player-game* rows, not direct
pre-match rows.  Leakage control happens at the next aggregation layer: for a
match at time ``T`` use only player-game rows with ``date < T``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import text

from betting_app.core.db import get_session


ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")

# High-availability, semantically broad per-player statistics from GOL.GG.
# Keys are kept exactly as they occur in ``golgg_game_players.stats_json``.
DEFAULT_PLAYER_STAT_KEYS: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "cs",
    "csm",
    "golds",
    "gpm",
    "gold%",
    "total_damage_to_champion",
    "dpm",
    "dmg%",
    "kp%",
    "wards_placed",
    "wards_destroyed",
    "control_wards_purchased",
    "vision_score",
    "vspm",
    "gd@15",
    "csd@15",
    "xpd@15",
    "lvld@15",
    "damage_dealt_to_turrets",
    "total_heal",
    "damage_self_mitigated",
    "double_kills",
    "triple_kills",
    "quadra_kills",
    "penta_kills",
)

TEAM_STAT_KEYS: tuple[str, ...] = ("kills", "towers", "dragons", "nashors", "gold")


@dataclass(frozen=True)
class PlayerGameDatasetConfig:
    """Configuration for EXP-048 player-game extraction.

    Args:
        min_date: First game date included.
        max_date: Optional final game date included.
        limit_rows: Optional player-row limit for smoke tests.
        require_core_stats: Drop rows missing all core K/D/A/CS/gold/damage
            fields.  Missing secondary stats are preserved as NaN for imputers.
        stat_keys: Player stat keys converted into numeric ``stat_*`` columns.
    """

    min_date: str = "2013-01-01"
    max_date: str | None = None
    limit_rows: int | None = None
    require_core_stats: bool = True
    stat_keys: tuple[str, ...] = DEFAULT_PLAYER_STAT_KEYS


@dataclass(frozen=True)
class PlayerGameDataset:
    """In-memory player-game dataset.

    ``frame`` contains one row per player-game.  ``feature_names`` are numeric
    columns suitable for a supervised/autoencoder PlayerGameEncoder.  Categorical
    identifiers (role/player/champion/team) are left as metadata for embedding
    tables in neural models.
    """

    frame: pd.DataFrame
    feature_names: list[str]
    categorical_names: list[str]
    target_names: list[str]
    metadata: dict[str, Any]


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


def _safe_float(value: Any, *, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if not math.isnan(converted) else default


def _patch_parts(patch: Any) -> tuple[float, float]:
    raw = str(patch or "")
    pieces = raw.split(".", maxsplit=1)
    major = _safe_float(pieces[0], default=math.nan) if pieces and pieces[0] else math.nan
    minor = _safe_float(pieces[1], default=math.nan) if len(pieces) > 1 else math.nan
    return major, minor


def _role_index(role: Any) -> float:
    try:
        return float(ROLES.index(str(role).upper()))
    except ValueError:
        return math.nan


def _side_blue(side_name: Any) -> float:
    side = str(side_name or "").strip().lower()
    if side == "blue":
        return 1.0
    if side == "red":
        return 0.0
    return math.nan


def _team_context(raw: pd.Series, side: str) -> tuple[dict[str, float], dict[str, float], float, float]:
    if side == "t1":
        team_stats = _json_loads(raw.get("team1_stats_json"))
        opp_stats = _json_loads(raw.get("team2_stats_json"))
        game_win = _safe_float(raw.get("game_team1_win"))
        blue = _side_blue(raw.get("team1_side"))
    else:
        team_stats = _json_loads(raw.get("team2_stats_json"))
        opp_stats = _json_loads(raw.get("team1_stats_json"))
        game_win = _safe_float(raw.get("game_team2_win"))
        blue = _side_blue(raw.get("team2_side"))
    return team_stats, opp_stats, game_win, blue


def load_player_game_rows_from_db(config: PlayerGameDatasetConfig | None = None) -> pd.DataFrame:
    """Load raw joined GOL.GG player-game rows from the active database."""

    cfg = config or PlayerGameDatasetConfig()
    where = ["g.date IS NOT NULL", "g.date >= :min_date", "gp.stats_json IS NOT NULL", "gp.stats_json <> ''"]
    params: dict[str, Any] = {"min_date": cfg.min_date}
    if cfg.max_date:
        where.append("g.date <= :max_date")
        params["max_date"] = cfg.max_date

    query = f"""
        SELECT
            g.game_id,
            g.match_id,
            g.date,
            g.tournament_name,
            g.patch,
            g.team1_id,
            g.team2_id,
            g.team1_name,
            g.team2_name,
            g.team1_win AS game_team1_win,
            g.team2_win AS game_team2_win,
            g.team1_side,
            g.team2_side,
            g.game_duration,
            g.team1_stats_json,
            g.team2_stats_json,
            m.team1_score,
            m.team2_score,
            m.team1_win AS match_team1_win,
            m.team2_win AS match_team2_win,
            m.best_of,
            gp.team_id,
            gp.team_name,
            gp.side,
            gp.role,
            gp.player_id,
            gp.player_name,
            gp.champion_id,
            gp.champion_name,
            gp.stats_json
        FROM golgg_game_players gp
        JOIN golgg_games g ON g.game_id = gp.game_id
        JOIN golgg_matches m ON m.match_id = gp.match_id
        WHERE {' AND '.join(where)}
        ORDER BY g.date ASC, g.game_id ASC,
            CASE gp.side WHEN 't1' THEN 1 WHEN 't2' THEN 2 ELSE 9 END,
            CASE gp.role WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3 WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 9 END
    """
    if cfg.limit_rows:
        query += " LIMIT :limit_rows"
        params["limit_rows"] = int(cfg.limit_rows)

    with get_session() as session:
        return pd.read_sql(text(query), session.connection(), params=params)


def build_player_game_dataset(
    raw_rows: pd.DataFrame,
    config: PlayerGameDatasetConfig | None = None,
) -> PlayerGameDataset:
    """Convert joined GOL.GG rows into encoder-ready player-game examples."""

    cfg = config or PlayerGameDatasetConfig()
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {"missing_core_stats": 0, "missing_side": 0}

    core_keys = {"kills", "deaths", "assists", "cs", "golds", "total_damage_to_champion"}

    for _, raw in raw_rows.iterrows():
        side = str(raw.get("side") or "").lower()
        if side not in {"t1", "t2"}:
            skipped["missing_side"] += 1
            continue

        player_stats = _json_loads(raw.get("stats_json"))
        if cfg.require_core_stats and all(player_stats.get(key) in (None, "") for key in core_keys):
            skipped["missing_core_stats"] += 1
            continue

        team_stats, opp_stats, game_win, blue = _team_context(raw, side)
        patch_major, patch_minor = _patch_parts(raw.get("patch"))
        team_id = str(raw.get("team_id") or "")
        match_team1_id = str(raw.get("team1_id") or "")
        match_team2_id = str(raw.get("team2_id") or "")
        if team_id and team_id == match_team1_id:
            match_win = _safe_float(raw.get("match_team1_win"))
        elif team_id and team_id == match_team2_id:
            match_win = _safe_float(raw.get("match_team2_win"))
        else:
            match_win = math.nan

        row: dict[str, Any] = {
            "game_id": str(raw.get("game_id")),
            "match_id": str(raw.get("match_id")),
            "date": raw.get("date"),
            "tournament_name": raw.get("tournament_name"),
            "patch": raw.get("patch"),
            "team_id": team_id or None,
            "team_name": raw.get("team_name"),
            "opponent_team_id": match_team2_id if team_id == match_team1_id else match_team1_id,
            "side": side,
            "role": raw.get("role"),
            "player_id": str(raw.get("player_id") or raw.get("player_name") or ""),
            "player_name": raw.get("player_name"),
            "champion_id": str(raw.get("champion_id") or raw.get("champion_name") or ""),
            "champion_name": raw.get("champion_name"),
            "role_index": _role_index(raw.get("role")),
            "side_blue": blue,
            "patch_major": patch_major,
            "patch_minor": patch_minor,
            "game_duration_seconds": _safe_float(raw.get("game_duration")),
            "game_win": game_win,
            "match_win": match_win,
            "team1_score": _safe_float(raw.get("team1_score")),
            "team2_score": _safe_float(raw.get("team2_score")),
            "best_of": _safe_float(raw.get("best_of")),
        }

        for key in cfg.stat_keys:
            row[f"stat_{key}"] = _safe_float(player_stats.get(key))
        for key in TEAM_STAT_KEYS:
            row[f"team_{key}"] = _safe_float(team_stats.get(key))
            row[f"opp_team_{key}"] = _safe_float(opp_stats.get(key))
        row["team_gold_diff"] = row["team_gold"] - row["opp_team_gold"]
        row["team_kill_diff"] = row["team_kills"] - row["opp_team_kills"]

        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values(["date", "game_id", "side", "role_index"]).reset_index(drop=True)

    feature_names = [
        "role_index",
        "side_blue",
        "patch_major",
        "patch_minor",
        "game_duration_seconds",
        "game_win",
        *[f"stat_{key}" for key in cfg.stat_keys],
        *[f"team_{key}" for key in TEAM_STAT_KEYS],
        *[f"opp_team_{key}" for key in TEAM_STAT_KEYS],
        "team_gold_diff",
        "team_kill_diff",
    ]
    categorical_names = ["player_id", "team_id", "opponent_team_id", "role", "champion_id"]
    target_names = [
        "match_win",
        "team1_score",
        "team2_score",
        "best_of",
        "stat_kills",
        "stat_deaths",
        "stat_assists",
        "team_kills",
        "game_duration_seconds",
    ]
    metadata = {
        "experiment_id": "EXP-048",
        "purpose": "Audit/build one-row-per-player-game data for supervised/autoencoder PlayerGameEncoder.",
        "leakage_note": "Rows describe completed games; pre-match aggregation must filter to date < predicted match date.",
        "config": asdict(cfg),
        "raw_rows": int(len(raw_rows)),
        "rows": int(len(frame)),
        "feature_count": len(feature_names),
        "categorical_count": len(categorical_names),
        "target_count": len(target_names),
        "skipped": {key: int(value) for key, value in skipped.items() if value},
    }
    if not frame.empty:
        metadata.update(
            {
                "date_min": frame["date"].min().isoformat() if pd.notna(frame["date"].min()) else None,
                "date_max": frame["date"].max().isoformat() if pd.notna(frame["date"].max()) else None,
                "distinct_players": int(frame["player_id"].nunique()),
                "distinct_champions": int(frame["champion_id"].nunique()),
                "distinct_games": int(frame["game_id"].nunique()),
                "distinct_matches": int(frame["match_id"].nunique()),
            }
        )
    return PlayerGameDataset(
        frame=frame,
        feature_names=feature_names,
        categorical_names=categorical_names,
        target_names=target_names,
        metadata=metadata,
    )


def build_player_game_dataset_from_db(config: PlayerGameDatasetConfig | None = None) -> PlayerGameDataset:
    """Load and build player-game dataset from the active database."""

    cfg = config or PlayerGameDatasetConfig()
    return build_player_game_dataset(load_player_game_rows_from_db(cfg), cfg)


def iter_player_game_rows(dataset: PlayerGameDataset) -> Iterable[dict[str, Any]]:
    """Yield JSON-serializable player-game rows."""

    for row in dataset.frame.replace({np.nan: None}).to_dict(orient="records"):
        if hasattr(row.get("date"), "isoformat"):
            row["date"] = row["date"].isoformat()
        yield row
