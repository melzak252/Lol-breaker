"""Historical Player Statistics Feature Extraction & Prior Fitting.

Extracts empirical priors for in-game player statistics (kills, deaths, assists,
creep score, objectives, pentakills) from historical GOL.GG records and fits
hierarchical prior distributions with robust fallbacks.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Standard competitive League of Legends roles
ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")

ROLE_ALIAS_MAP: dict[str, str] = {
    "TOP": "TOP",
    "JUG": "JUNGLE",
    "JUNGLE": "JUNGLE",
    "MID": "MID",
    "MIDDLE": "MID",
    "ADC": "ADC",
    "BOT": "ADC",
    "BOTTOM": "ADC",
    "SUP": "SUPPORT",
    "SUPPORT": "SUPPORT",
}


def normalize_role(role: str) -> str:
    """Normalize role string to canonical canonical standard."""
    clean = str(role or "").strip().upper()
    return ROLE_ALIAS_MAP.get(clean, clean)


# Default empirical role kill shares: Dirichlet concentration alpha (sum = 100.0)
DEFAULT_ROLE_KILL_SHARES: dict[str, float] = {
    "ADC": 33.0,
    "MID": 28.0,
    "JUNGLE": 18.0,
    "TOP": 16.0,
    "SUPPORT": 5.0,
}

# Default empirical role death shares: Dirichlet concentration alpha (sum = 100.0)
DEFAULT_ROLE_DEATH_SHARES: dict[str, float] = {
    "SUPPORT": 26.0,
    "TOP": 22.0,
    "JUNGLE": 20.0,
    "MID": 17.0,
    "ADC": 15.0,
}

# Default assist multipliers and role distribution
DEFAULT_ROLE_ASSIST_FACTORS: dict[str, Any] = {
    "multiplier": 2.15,
    "weights": {
        "SUPPORT": 0.35,
        "JUNGLE": 0.25,
        "MID": 0.18,
        "TOP": 0.12,
        "ADC": 0.10,
    },
}

# Default winner kill share: Beta(alpha_w, beta_w) mean ~ 70%
DEFAULT_WINNER_KILL_SHARE: dict[str, float] = {
    "alpha": 14.0,
    "beta": 6.0,
}

# Default objective and pentakill rates
DEFAULT_OBJECTIVE_RATES: dict[str, Any] = {
    "dragon_rate_per_minute": 0.145,
    "dragon_winner_share": 0.68,
    "baron_rate_per_minute": 0.042,
    "baron_winner_share": 0.82,
    "pentakill_rate_per_game": 0.016,
    "pentakill_role_distribution": {
        "ADC": 0.62,
        "MID": 0.28,
        "TOP": 0.06,
        "JUNGLE": 0.03,
        "SUPPORT": 0.01,
    },
}

# Default CS per minute rates by role
DEFAULT_ROLE_CSM: dict[str, float] = {
    "ADC": 9.1,
    "MID": 8.5,
    "TOP": 7.8,
    "JUNGLE": 5.5,
    "SUPPORT": 1.2,
}


def get_default_priors() -> dict[str, Any]:
    """Return static empirical reference priors when DB is unreachable or empty."""
    return {
        "role_kill_shares": dict(DEFAULT_ROLE_KILL_SHARES),
        "role_death_shares": dict(DEFAULT_ROLE_DEATH_SHARES),
        "role_assist_factors": {
            "multiplier": DEFAULT_ROLE_ASSIST_FACTORS["multiplier"],
            "weights": dict(DEFAULT_ROLE_ASSIST_FACTORS["weights"]),
        },
        "winner_kill_share": dict(DEFAULT_WINNER_KILL_SHARE),
        "objective_rates": {
            "dragon_rate_per_minute": DEFAULT_OBJECTIVE_RATES["dragon_rate_per_minute"],
            "dragon_winner_share": DEFAULT_OBJECTIVE_RATES["dragon_winner_share"],
            "baron_rate_per_minute": DEFAULT_OBJECTIVE_RATES["baron_rate_per_minute"],
            "baron_winner_share": DEFAULT_OBJECTIVE_RATES["baron_winner_share"],
            "pentakill_rate_per_game": DEFAULT_OBJECTIVE_RATES["pentakill_rate_per_game"],
            "pentakill_role_distribution": dict(DEFAULT_OBJECTIVE_RATES["pentakill_role_distribution"]),
        },
        "role_csm": dict(DEFAULT_ROLE_CSM),
        "player_profiles": {},
        "as_of_date": "1900-01-01",
        "sample_games": 0,
        "sample_player_records": 0,
    }


def extract_player_stat_priors(
    as_of_date: str = "2026-01-01",
    min_games: int = 10,
    db: Any = None,
) -> dict[str, Any]:
    """Extract empirical priors for in-game stats strictly prior to as_of_date.

    Parameters
    ----------
    as_of_date : str
        Cutoff date (YYYY-MM-DD); only games strictly prior (< as_of_date) are used.
    min_games : int
        Minimum games required to compute an individual player adjustment profile.
    db : Any, optional
        Database session or connection. If None, retrieves session via betting_app.core.db.

    Returns
    -------
    dict[str, Any]
        Dictionary of fitted priors including role Dirichlet alphas, player profiles,
        winner kill share Beta parameters, and objective rates.
    """
    priors = get_default_priors()
    priors["as_of_date"] = as_of_date

    # Attempt to query database
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
            # 1. Query role-level statistics
            role_sql = text("""
                SELECT 
                    gp.role,
                    COUNT(*) as count,
                    AVG((gp.stats_json::json->>'kills')::float) as avg_kills,
                    AVG((gp.stats_json::json->>'deaths')::float) as avg_deaths,
                    AVG((gp.stats_json::json->>'assists')::float) as avg_assists,
                    AVG((gp.stats_json::json->>'csm')::float) as avg_csm
                FROM golgg_game_players gp
                JOIN golgg_games g ON gp.game_id = g.game_id
                WHERE g.date < :as_of_date
                  AND gp.stats_json IS NOT NULL
                  AND gp.stats_json <> ''
                GROUP BY gp.role
            """)
            conn = session.connection() if hasattr(session, "connection") else session
            role_df = pd.read_sql(role_sql, conn, params={"as_of_date": as_of_date})

            if role_df.empty:
                logger.warning("No historical role rows found prior to %s; using default priors.", as_of_date)
                return priors

            # Calculate total games and player records
            total_records = int(role_df["count"].sum())
            priors["sample_player_records"] = total_records
            priors["sample_games"] = total_records // 10

            # Map raw role names to canonical roles
            role_metrics: dict[str, dict[str, float]] = {}
            for _, r in role_df.iterrows():
                c_role = normalize_role(str(r["role"]))
                if c_role not in ROLES:
                    continue
                role_metrics[c_role] = {
                    "avg_kills": float(r["avg_kills"] or 0.0),
                    "avg_deaths": float(r["avg_deaths"] or 0.0),
                    "avg_assists": float(r["avg_assists"] or 0.0),
                    "avg_csm": float(r["avg_csm"] or 0.0),
                }

            # Normalize role kill shares (Dirichlet total concentration 100.0)
            tot_kills = sum(m["avg_kills"] for m in role_metrics.values())
            if tot_kills > 0:
                priors["role_kill_shares"] = {
                    role: round(100.0 * (role_metrics.get(role, {}).get("avg_kills", 1.0) / tot_kills), 2)
                    for role in ROLES
                }

            # Normalize role death shares (Dirichlet total concentration 100.0)
            tot_deaths = sum(m["avg_deaths"] for m in role_metrics.values())
            if tot_deaths > 0:
                priors["role_death_shares"] = {
                    role: round(100.0 * (role_metrics.get(role, {}).get("avg_deaths", 1.0) / tot_deaths), 2)
                    for role in ROLES
                }

            # Assist multiplier and weights
            tot_assists = sum(m["avg_assists"] for m in role_metrics.values())
            if tot_kills > 0 and tot_assists > 0:
                mult = round(tot_assists / tot_kills, 2)
                priors["role_assist_factors"]["multiplier"] = mult
                priors["role_assist_factors"]["weights"] = {
                    role: round(role_metrics.get(role, {}).get("avg_assists", 1.0) / tot_assists, 4)
                    for role in ROLES
                }

            # Role CSM
            priors["role_csm"] = {
                role: round(role_metrics.get(role, {}).get("avg_csm", DEFAULT_ROLE_CSM[role]), 2)
                for role in ROLES
            }

            # 2. Query individual player profiles
            player_sql = text("""
                SELECT 
                    gp.player_id,
                    gp.player_name,
                    gp.role,
                    COUNT(*) as games_count,
                    AVG((gp.stats_json::json->>'kills')::float) as avg_kills,
                    AVG((gp.stats_json::json->>'deaths')::float) as avg_deaths,
                    AVG((gp.stats_json::json->>'assists')::float) as avg_assists,
                    AVG((gp.stats_json::json->>'csm')::float) as avg_csm
                FROM golgg_game_players gp
                JOIN golgg_games g ON gp.game_id = g.game_id
                WHERE g.date < :as_of_date
                  AND gp.stats_json IS NOT NULL
                  AND gp.stats_json <> ''
                GROUP BY gp.player_id, gp.player_name, gp.role
                HAVING COUNT(*) >= :min_games
            """)
            player_df = pd.read_sql(player_sql, conn, params={"as_of_date": as_of_date, "min_games": min_games})

            # Calculate baseline role kill fractions (e.g. 0.33, 0.28, ...)
            baseline_ks = {
                role: priors["role_kill_shares"][role] / 100.0 for role in ROLES
            }
            baseline_deaths = {
                role: role_metrics.get(role, {}).get("avg_deaths", 2.5) for role in ROLES
            }

            profiles: dict[str, dict[str, Any]] = {}
            for _, r in player_df.iterrows():
                p_id = str(r["player_id"]).strip()
                p_name = str(r["player_name"]).strip()
                c_role = normalize_role(str(r["role"]))
                if not p_id:
                    continue

                avg_k = float(r["avg_kills"] or 0.0)
                avg_d = float(r["avg_deaths"] or 0.0)
                avg_csm = float(r["avg_csm"] or 0.0)
                g_count = int(r["games_count"])

                # Kill share delta: compare player's estimated kill share to role baseline
                base_k = role_metrics.get(c_role, {}).get("avg_kills", 2.5)
                # Player kill share adjustment: delta in percentage points
                if tot_kills > 0:
                    player_ks = avg_k / tot_kills
                    delta_ks = player_ks - baseline_ks.get(c_role, 0.2)
                else:
                    delta_ks = 0.0

                # Death rate multiplier: player deaths vs role avg deaths
                role_d = baseline_deaths.get(c_role, 2.5)
                death_multiplier = avg_d / max(0.5, role_d)

                profile_entry = {
                    "player_id": p_id,
                    "player_name": p_name,
                    "role": c_role,
                    "games_count": g_count,
                    "delta_ks": float(np.clip(delta_ks, -0.15, 0.15)),
                    "death_multiplier": float(np.clip(death_multiplier, 0.4, 2.5)),
                    "csm": float(np.clip(avg_csm, 0.5, 14.0)),
                }

                # Store by player_id and lowercase player_name for fast lookup (retain primary role with max games)
                existing_id = profiles.get(p_id)
                if existing_id is None or g_count > existing_id.get("games_count", 0):
                    profiles[p_id] = profile_entry
                if p_name:
                    p_name_lower = p_name.lower()
                    existing_name = profiles.get(p_name_lower)
                    if existing_name is None or g_count > existing_name.get("games_count", 0):
                        profiles[p_name_lower] = profile_entry
            priors["player_profiles"] = profiles

    except Exception as exc:
        logger.warning("Could not extract player stat priors from DB (%s); using default priors.", exc)

    return priors
