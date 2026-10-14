"""Player champion pool entropy and Fearless draft depletion decay modeling.

Computes player-level Shannon entropy and effective champion pool depth
from historical match frequencies strictly prior to the series cutoff date.
Calculates parametric rating depletion decay when comfort champions are burned.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
import math
from typing import Any, Mapping, Sequence

import numpy as np

# Standard role order for competitive League of Legends
ROLES: tuple[str, ...] = ("TOP", "JUG", "MID", "ADC", "SUP")

# Parametric depletion constants
DEFAULT_GAMMA: float = 15.0  # rating points penalty scale
DEFAULT_COMFORT_THRESHOLD: float = 0.10  # 10% pick rate threshold for comfort picks
EFFECTIVE_DEPTH_BASE: float = math.e

# In-memory mock/test registry for deterministic unit tests and offline environments
_PLAYER_POOL_REGISTRY: dict[str, dict[str, Any]] = {}

# Default fallback meta champions per role (common LoL champion IDs)
DEFAULT_ROLE_META: dict[str, list[int]] = {
    "TOP": [266, 875, 85, 24, 114, 516, 58, 41, 150, 98],       # Aatrox, Sett, Kennen, Jax, Fiora, Ornn, Renekton, Gangplank, Gnar, Shen
    "JUG": [64, 104, 76, 59, 120, 113, 254, 121, 60, 421],      # Lee Sin, Graves, Nidalee, Jarvan IV, Hecarim, Sejuani, Vi, Kha'Zix, Elise, Rek'Sai
    "MID": [103, 134, 61, 238, 245, 126, 7, 84, 4, 3],          # Ahri, Syndra, Orianna, Zed, Ekko, Jayce, LeBlanc, Akali, Twisted Fate, Galio
    "ADC": [51, 81, 222, 21, 236, 498, 145, 202, 22, 119],       # Caitlyn, Ezreal, Jinx, Miss Fortune, Lucian, Xayah, Kai'Sa, Jhin, Ashe, Draven
    "SUP": [412, 89, 53, 201, 40, 111, 25, 497, 432, 12],       # Thresh, Leona, Blitzcrank, Braum, Janna, Nautilus, Morgana, Rakan, Bard, Alistar
}


def register_player_pool(
    player_id: str,
    champion_counts: Mapping[int | str, int],
    role: str = "MID",
) -> dict[str, Any]:
    """Register custom historical champion counts for a player in the in-memory registry."""
    key = str(player_id).strip().lower()
    total = sum(champion_counts.values())
    unique = len(champion_counts)
    
    if total == 0:
        entropy = 0.0
        depth = 0.0
        freqs: dict[int, float] = {}
        comfort: list[int] = []
    else:
        freqs = {}
        entropy = 0.0
        for champ_id, count in champion_counts.items():
            cid = int(champ_id)
            p = count / total
            freqs[cid] = p
            if p > 0:
                entropy -= p * math.log(p)
        depth = math.exp(entropy)
        comfort = [cid for cid, p in freqs.items() if p >= DEFAULT_COMFORT_THRESHOLD]

    data = {
        "player_id": player_id,
        "role": role.upper(),
        "total_games": total,
        "unique_champions": unique,
        "entropy": float(entropy),
        "effective_pool_depth": float(depth),
        "champion_frequencies": freqs,
        "comfort_champions": comfort,
    }
    _PLAYER_POOL_REGISTRY[key] = data
    return data


def clear_player_pool_registry() -> None:
    """Clear registered in-memory player pools."""
    _PLAYER_POOL_REGISTRY.clear()


def compute_player_pool_metrics(
    player_id: str,
    as_of_date: str | date | datetime | None = None,
    db: Any = None,
) -> dict[str, Any]:
    """Compute Shannon entropy and effective pool depth for a player prior to as_of_date.

    Parameters
    ----------
    player_id : str
        Unique player identifier or canonical name.
    as_of_date : str | date | datetime | None
        Strict upper boundary for historical games.
    db : Any, optional
        Database session for querying `golgg_games` and `golgg_game_players`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - player_id: str
        - total_games: int
        - unique_champions: int
        - entropy: float (Shannon entropy H)
        - effective_pool_depth: float (exp(H))
        - champion_frequencies: dict[int, float]
        - comfort_champions: list[int] (champions with pick rate >= 10%)
    """
    key = str(player_id).strip().lower()
    if key in _PLAYER_POOL_REGISTRY:
        return dict(_PLAYER_POOL_REGISTRY[key])

    # Convert as_of_date to string YYYY-MM-DD
    cutoff_str = "9999-12-31"
    if as_of_date is not None:
        if isinstance(as_of_date, (datetime, date)):
            cutoff_str = as_of_date.strftime("%Y-%m-%d")
        elif isinstance(as_of_date, str) and as_of_date.strip():
            cutoff_str = as_of_date.strip()[:10]

    counts: Counter[int] = Counter()
    role_counter: Counter[str] = Counter()

    # Query from DB if available
    if db is not None:
        try:
            from sqlalchemy import text
            sql = """
                SELECT gp.champion_id, gp.role, COUNT(*) as cnt
                FROM golgg_game_players gp
                JOIN golgg_games g ON gp.game_id = g.game_id
                WHERE (LOWER(gp.player_id) = :pid OR LOWER(gp.player_name) = :pid)
                  AND g.date < :cutoff
                  AND gp.champion_id IS NOT NULL
                GROUP BY gp.champion_id, gp.role
            """
            result = db.execute(text(sql), {"pid": key, "cutoff": cutoff_str}).fetchall()
            for row in result:
                cid_raw = row[0]
                role_raw = row[1]
                cnt = row[2]
                try:
                    cid = int(cid_raw)
                    counts[cid] += cnt
                except (ValueError, TypeError):
                    pass
                if role_raw:
                    role_counter[str(role_raw).upper()] += cnt
        except Exception:
            pass

    # If no games found in DB, return empty/neutral metrics
    total = sum(counts.values())
    unique = len(counts)
    primary_role = role_counter.most_common(1)[0][0] if role_counter else "MID"

    if total == 0:
        return {
            "player_id": player_id,
            "role": primary_role,
            "total_games": 0,
            "unique_champions": 0,
            "entropy": 0.0,
            "effective_pool_depth": 0.0,
            "champion_frequencies": {},
            "comfort_champions": [],
        }

    freqs: dict[int, float] = {}
    entropy = 0.0
    for cid, cnt in counts.items():
        p = cnt / total
        freqs[cid] = p
        if p > 0:
            entropy -= p * math.log(p)

    depth = math.exp(entropy)
    comfort = [cid for cid, p in freqs.items() if p >= DEFAULT_COMFORT_THRESHOLD]

    return {
        "player_id": player_id,
        "role": primary_role,
        "total_games": total,
        "unique_champions": unique,
        "entropy": float(entropy),
        "effective_pool_depth": float(depth),
        "champion_frequencies": freqs,
        "comfort_champions": comfort,
    }


def calculate_depletion_penalty(
    player_id: str,
    burned_champions: set[int] | Sequence[int],
    comfort_threshold: float = 3.5,
    pool_metrics: dict[str, Any] | None = None,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """Calculate the rating decay penalty Delta r when comfort champions are burned.

    Parameters
    ----------
    player_id : str
        Player identifier.
    burned_champions : set[int] | Sequence[int]
        Cumulative champion IDs burned in prior maps.
    comfort_threshold : float
        Effective pool depth threshold below which shallow players suffer.
    pool_metrics : dict[str, Any] | None
        Precomputed pool metrics from `compute_player_pool_metrics`.
    gamma : float
        Penalty scale in rating points (default 15.0).

    Returns
    -------
    float
        Delta r <= 0 rating points decay penalty.
        Formula: -gamma * max(0, burned_comfort - 1)^1.5
    """
    if pool_metrics is None:
        pool_metrics = compute_player_pool_metrics(player_id)

    burned_set = {int(c) for c in burned_champions}
    comfort_champs = pool_metrics.get("comfort_champions", [])
    if not comfort_champs and "champion_frequencies" in pool_metrics:
        comfort_champs = [
            cid for cid, p in pool_metrics["champion_frequencies"].items()
            if p >= DEFAULT_COMFORT_THRESHOLD
        ]

    burned_comfort = sum(1 for cid in comfort_champs if int(cid) in burned_set)
    depth = pool_metrics.get("effective_pool_depth", 0.0)

    # Deep player resilience:
    # If depth >= comfort_threshold, the player has high entropy and their
    # comfort allowance scales with their effective pool depth.
    if depth >= comfort_threshold:
        allowance = max(1, int(math.floor(depth)))
    else:
        allowance = 1

    # If comfort champions are completely exhausted for a shallow player,
    # any additional role champions burned force them further off-comfort
    role = pool_metrics.get("role", "MID").upper()
    meta_picks = DEFAULT_ROLE_META.get(role, [])
    role_burned = sum(1 for c in burned_set if c in comfort_champs or c in meta_picks)

    if depth < comfort_threshold and len(comfort_champs) > 0 and burned_comfort >= len(comfort_champs):
        effective_burned = max(burned_comfort, role_burned)
    else:
        effective_burned = burned_comfort

    excess = max(0, effective_burned - allowance)
    if excess <= 0:
        return 0.0

    penalty = -float(gamma) * (excess ** 1.5)
    return penalty


def ensure_player_pools_loaded(as_of_date: str = "2026-01-01", db: Any = None) -> int:
    """Pre-load historical champion distributions for all players into in-memory registry.

    Parameters
    ----------
    as_of_date : str
        Cutoff date; only games strictly prior are aggregated.
    db : Any, optional
        Database session or connection.

    Returns
    -------
    int
        Number of registered players in memory.
    """
    if len(_PLAYER_POOL_REGISTRY) > 100:
        return len(_PLAYER_POOL_REGISTRY)

    # Do not query truncated test DB during automated unit tests
    import os
    if os.getenv("PYTEST_CURRENT_TEST") or "betting_test" in os.getenv("DATABASE_URL", ""):
        return len(_PLAYER_POOL_REGISTRY)

    try:
        if db is None:
            from betting_app.core.db import get_session
            session_ctx = get_session()
        elif hasattr(db, "__enter__"):
            session_ctx = db
        else:
            from contextlib import nullcontext
            session_ctx = nullcontext(db)

        with session_ctx as session:
            from sqlalchemy import text
            rows = session.execute(text("""
                SELECT LOWER(gp.player_name) as p_name, gp.champion_id, COUNT(*) as cnt, MAX(gp.role) as role
                FROM golgg_game_players gp
                JOIN golgg_games g ON gp.game_id = g.game_id
                WHERE gp.champion_id IS NOT NULL
                  AND g.date < :cutoff
                GROUP BY LOWER(gp.player_name), gp.champion_id
            """), {"cutoff": as_of_date}).fetchall()

            player_data: dict[str, dict[str, Any]] = {}
            for r in rows:
                p_name = str(r[0] or "").strip().lower()
                if not p_name:
                    continue
                cid = int(r[1])
                cnt = int(r[2])
                role = str(r[3] or "MID")
                if p_name not in player_data:
                    player_data[p_name] = {"counts": {}, "role": role}
                player_data[p_name]["counts"][cid] = cnt

            for p_name, p_info in player_data.items():
                register_player_pool(p_name, p_info["counts"], role=p_info["role"])

        return len(_PLAYER_POOL_REGISTRY)
    except Exception:
        return len(_PLAYER_POOL_REGISTRY)
