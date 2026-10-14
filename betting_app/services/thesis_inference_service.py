"""Inference service for the thesis model: Sym-Cal LR-ElasticNet-W20-Binomial.

Loads the serialized pipeline and Platt calibrator, builds features for
upcoming matches using the same 46-feature schema as training, applies
order symmetry and Platt calibration, and stores predictions in
canonical_predictions.

Usage:
    from betting_app.services.thesis_inference_service import predict_upcoming_with_thesis_model
    results = predict_upcoming_with_thesis_model()
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import trueskill
import openskill.models

from betting_app.core.db import query_df, transaction
from betting_app.core.matching import normalize_team_name
from betting_app.core.ev import fair_market_probabilities
from betting_app.services.canonical_match_service import align_snapshot_odds
from betting_app.services.mapping_service import golgg_name_from_id
from betting_app.services.upcoming_inference_service import (
    RATING_SYSTEMS,
    apply_temperature_probability,
    load_canonical_matches,
    load_last_roster,
    load_player_ratings,
    load_team_ratings,
    load_w20,
    rating_probabilities,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THESIS_MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
THESIS_MODEL_VERSION = "exp-039"
THESIS_HYBRID_MODEL_NAME = "Hybrid-Thesis-Market"
# EXP-061 showed that EXP-039 adds signal to bookmaker no-vig probabilities
# when blended conservatively.  Keep market as the dominant prior, but allow
# enough model weight to improve LogLoss on open/mid/close odds without letting
# the model dominate. Formula: alpha * temperature(thesis_model) +
# (1-alpha) * market.
THESIS_HYBRID_ALPHA = 0.35
THESIS_HYBRID_TEMPERATURE = 0.80
EPSILON = 0.001

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTEFACT_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"
METADATA_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_metadata.json"

# Feature definitions (must match training exactly)
OPTUNA_BASE_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
    "player_elo_min1",
    "player_elo_min2",
    "player_gl_max1",
    "player_gl_max2",
    "player_gl_rd_avg1",
    "player_gl_rd_avg2",
    "player_ts_sigma_avg1",
    "player_ts_sigma_avg2",
    "player_os_sigma_avg1",
    "player_os_sigma_avg2",
    "player_pl_sigma_avg1",
    "player_pl_sigma_avg2",
    "player_tm_sigma_avg1",
    "player_tm_sigma_avg2",
]

ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate",
    "t2_rolling_win_rate",
    "t1_rolling_kills",
    "t2_rolling_kills",
    "t1_rolling_deaths",
    "t2_rolling_deaths",
    "t1_rolling_gd15",
    "t2_rolling_gd15",
    "t1_rolling_dpm",
    "t2_rolling_dpm",
    "t1_rolling_vspm",
    "t2_rolling_vspm",
    "t1_rolling_towers",
    "t2_rolling_towers",
    "t1_rolling_nashors",
    "t2_rolling_nashors",
    "t1_rolling_gold",
    "t2_rolling_gold",
    "t1_rolling_duration",
    "t2_rolling_duration",
]

RANK_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]

BINOMIAL_FEATURES = [f"{f}_binom_series" for f in RANK_PROB_FEATURES]

ALL_FEATURES = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + BINOMIAL_FEATURES

# Default values for missing rolling features
_DEFAULT_ROLLING = {
    "win_rate": 0.5,
    "kills": 12.0,
    "deaths": 12.0,
    "gd15": 0.0,
    "dpm": 1800.0,
    "vspm": 7.0,
    "towers": 5.0,
    "nashors": 0.5,
    "gold": 55000.0,
    "duration": 1800.0,
}


# ---------------------------------------------------------------------------
# Rating Systems Initialization (matching training parameters)
# ---------------------------------------------------------------------------
_ts_model = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.16, tau=0.25, draw_probability=0.0)
_os_model = openskill.models.PlackettLuce(mu=25.0, sigma=3.5)
_pl_model = openskill.models.PlackettLuce(mu=25.0, sigma=8.333, beta=18.75, tau=0.05)
_tm_model = openskill.models.ThurstoneMostellerFull(mu=25.0, sigma=8.333, beta=18.75, tau=0.05)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_pipeline = None
_calibrator = None


def _load_model():
    """Lazy-load the pipeline and calibrator."""
    global _pipeline, _calibrator
    if _pipeline is None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(f"Pipeline not found: {PIPELINE_PATH}")
        _pipeline = joblib.load(PIPELINE_PATH)
    if _calibrator is None:
        if not CALIBRATOR_PATH.exists():
            raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")
        _calibrator = joblib.load(CALIBRATOR_PATH)
    return _pipeline, _calibrator


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------
def _rating_prob(rating_a: float | None, rating_b: float | None, system: str) -> float:
    """Compute probability from rating difference."""
    if rating_a is None or rating_b is None:
        return 0.5
    diff = float(rating_a) - float(rating_b)
    if system in {"elo", "gl"}:
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))
    elif system == "os":
        return _sigmoid(diff / 5.0)
    else:
        return _sigmoid(diff / 8.333)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def _glicko_g(rd: float) -> float:
    q = math.log(10) / 400
    return 1 / math.sqrt(1 + 3 * (q**2) * (rd**2) / (math.pi**2))


def _glicko_expected_score(r1: float, rd1: float, r2: float, rd2: float) -> float:
    combined_rd = math.sqrt(rd1**2 + rd2**2)
    g_factor = _glicko_g(combined_rd)
    exponent = -g_factor * (r1 - r2) / 400
    return 1 / (1 + 10**exponent)


def _logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit for Platt calibration input."""
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _series_probability(map_prob: np.ndarray, best_of: int) -> np.ndarray:
    """Convert map-win probability to series probability."""
    prob = np.clip(map_prob.astype(float), EPSILON, 1.0 - EPSILON)
    if best_of == 1:
        return prob
    needed = best_of // 2 + 1
    series_prob = np.zeros_like(prob)
    for wins in range(needed, best_of + 1):
        series_prob += comb(best_of, wins) * np.power(prob, wins) * np.power(1.0 - prob, best_of - wins)
    return np.clip(series_prob, EPSILON, 1.0 - EPSILON)


def _load_roster_overrides(canonical_match_id: int) -> dict[str, dict[str, Any]]:
    """Load operator-confirmed rosters, without failing prediction on old DBs."""
    try:
        rows = query_df(
            """
            SELECT team_side, roster_json
            FROM match_roster_overrides
            WHERE canonical_match_id = ?
            """,
            (canonical_match_id,),
        )
    except Exception:
        # Allows local/dev databases that have not yet run the migration to
        # retain normal automatic prediction behaviour.
        return {}

    overrides: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        side = str(row.get("team_side") or "")
        raw = row.get("roster_json")
        try:
            roster = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            continue
        if side in {"a", "b"} and isinstance(roster, dict):
            players = roster.get("players")
            if isinstance(players, list) and len(players) == 5:
                overrides[side] = roster
    return overrides


def build_thesis_features_for_match(
    team_a_name: str,
    team_b_name: str,
    *,
    team_a_golgg_id: int | None = None,
    team_b_golgg_id: int | None = None,
    league: str | None = None,
    match_date: str | None = None,
    ratings_version: str = "latest-full",
    w20_version: str = "w20-latest",
    best_of: int = 1,
    team_a_roster_override: dict[str, Any] | None = None,
    team_b_roster_override: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build the 46-feature vector for one upcoming match.

    Uses GOL.GG team IDs directly when available (bypassing suggest_mapping)
    to avoid team alignment issues. Falls back to name-based suggest_mapping
    if IDs are not provided.

    Returns (feature_vector, diagnostics) where feature_vector is None if
    required data is missing.
    """
    diagnostics: dict[str, Any] = {
        "team_a": team_a_name,
        "team_b": team_b_name,
        "team_a_golgg_id": team_a_golgg_id,
        "team_b_golgg_id": team_b_golgg_id,
        "ratings_version": ratings_version,
        "w20_version": w20_version,
        "best_of": best_of,
        "missing": [],
    }

    # Resolve GOL.GG team names: prefer IDs over name-based suggest_mapping
    team_a_golgg = golgg_name_from_id(team_a_golgg_id)
    team_b_golgg = golgg_name_from_id(team_b_golgg_id)

    # If we have both IDs, check for side consistency and swap if necessary
    if team_a_golgg and team_b_golgg:
        g_norm_a = normalize_team_name(team_a_golgg)
        g_norm_b = normalize_team_name(team_b_golgg)
        c_norm_a = normalize_team_name(team_a_name)
        c_norm_b = normalize_team_name(team_b_name)

        # Check if GOL.GG IDs are swapped relative to canonical names
        # (e.g. GOL.GG ID A matches Canonical Name B and vice versa)
        is_a_at_b = (g_norm_a == c_norm_b or g_norm_a in c_norm_b or c_norm_b in g_norm_a)
        is_b_at_a = (g_norm_b == c_norm_a or g_norm_b in c_norm_a or c_norm_a in g_norm_b)

        if is_a_at_b and is_b_at_a:
            # Swapped! Fix it so team_a_golgg corresponds to team_a_name
            team_a_golgg, team_b_golgg = team_b_golgg, team_a_golgg
            diagnostics["side_swap_fixed"] = True
        elif is_a_at_b or is_b_at_a:
            # Partial mismatch or only one side matches the "wrong" team
            # This is riskier, but let's log it
            diagnostics["side_consistency_warning"] = f"Partial swap detected: A->B={is_a_at_b}, B->A={is_b_at_a}"

    diagnostics["golgg_resolved"] = {
        "from_ids": team_a_golgg is not None and team_b_golgg is not None,
        "team_a_golgg_name": team_a_golgg,
        "team_b_golgg_name": team_b_golgg,
    }

    if not team_a_golgg or not team_b_golgg:
        from betting_app.services.mapping_service import suggest_mapping

        # If we have one ID, we keep it and only map the missing one
        if not team_a_golgg:
            team_a_golgg, team_a_conf, team_a_source = suggest_mapping(
                team_a_name,
                source_system="bookmaker",
                league=league,
                match_date=match_date,
            )
        else:
            team_a_conf = 1.0
            team_a_source = "golgg_id"

        if not team_b_golgg:
            team_b_golgg, team_b_conf, team_b_source = suggest_mapping(
                team_b_name,
                source_system="bookmaker",
                league=league,
                match_date=match_date,
            )
        else:
            team_b_conf = 1.0
            team_b_source = "golgg_id"

        diagnostics["mapping_fallback"] = {
            "team_a_golgg_name": team_a_golgg,
            "team_b_golgg_name": team_b_golgg,
            "team_a_confidence": team_a_conf,
            "team_b_confidence": team_b_conf,
            "team_a_source": team_a_source,
            "team_b_source": team_b_source,
        }
    else:
        team_a_conf = team_b_conf = 1.0
        team_a_source = team_b_source = "golgg_id"

    if not team_a_golgg:
        diagnostics["missing"].append(f"team_a_mapping:{team_a_name}:{team_a_conf:.3f}")
    if not team_b_golgg:
        diagnostics["missing"].append(f"team_b_mapping:{team_b_name}:{team_b_conf:.3f}")
    if not team_a_golgg or not team_b_golgg:
        diagnostics["skip_reason"] = "unmapped_team"
        return None, diagnostics

    # A manually confirmed roster is intentionally preferred over the normal
    # last-observed GOL.GG fallback.  The latter is useful automatically, but
    # is often stale around substitutions and academy/main-team collisions.
    roster_a = team_a_roster_override or load_last_roster(team_a_golgg)
    roster_b = team_b_roster_override or load_last_roster(team_b_golgg)
    diagnostics["manual_roster_override"] = {
        "team_a": team_a_roster_override is not None,
        "team_b": team_b_roster_override is not None,
    }

    if not roster_a or len(roster_a.get("players", [])) < 5:
        diagnostics["missing"].append("team_a_roster")
    if not roster_b or len(roster_b.get("players", [])) < 5:
        diagnostics["missing"].append("team_b_roster")

    if not roster_a or not roster_b:
        diagnostics["skip_reason"] = "missing_roster"
        return None, diagnostics

    player_ids_a = [p["player_id"] for p in roster_a["players"]]
    player_ids_b = [p["player_id"] for p in roster_b["players"]]

    # Load player ratings
    all_player_ids = list(set(player_ids_a + player_ids_b))
    player_ratings = load_player_ratings(all_player_ids, ratings_version)

    # Load W20 rolling features
    w20_a = load_w20(team_a_golgg, w20_version)
    w20_b = load_w20(team_b_golgg, w20_version)
    if not w20_a:
        diagnostics["missing"].append("team_a_w20")
    if not w20_b:
        diagnostics["missing"].append("team_b_w20")

    # Build feature dict
    features: dict[str, float] = {}

    # Helper to get player rating with defaults (matching RatingManager)
    def get_p_rating(p_id, system):
        p_data = player_ratings.get(p_id, {}).get(system, {})
        if system == "elo":
            return p_data.get("rating_value", 1500.0)
        elif system == "gl":
            return p_data.get("rating_value", 1500.0), p_data.get("rd", 350.0)
        elif system == "ts":
            return p_data.get("rating_value", 25.0), p_data.get("sigma", 8.333)
        elif system == "os":
            return p_data.get("rating_value", 25.0), p_data.get("sigma", 3.5)
        else:  # pl, tm
            return p_data.get("rating_value", 25.0), p_data.get("sigma", 8.333)

    # 1. Rating probabilities (6 features)
    # Elo
    elo_a = [get_p_rating(p, "elo") for p in player_ids_a]
    elo_b = [get_p_rating(p, "elo") for p in player_ids_b]
    avg_elo_a = sum(elo_a) / len(elo_a)
    avg_elo_b = sum(elo_b) / len(elo_b)
    features["player_elo"] = 1.0 / (1.0 + 10 ** (-(avg_elo_a - avg_elo_b) / 400.0))

    # Glicko
    gl_a = [get_p_rating(p, "gl") for p in player_ids_a]
    gl_b = [get_p_rating(p, "gl") for p in player_ids_b]
    avg_gl_r_a = sum(r for r, rd in gl_a) / len(gl_a)
    avg_gl_rd_a = math.sqrt(sum(rd**2 for r, rd in gl_a) / len(gl_a))
    avg_gl_r_b = sum(r for r, rd in gl_b) / len(gl_b)
    avg_gl_rd_b = math.sqrt(sum(rd**2 for r, rd in gl_b) / len(gl_b))
    features["player_gl"] = _glicko_expected_score(avg_gl_r_a, avg_gl_rd_a, avg_gl_r_b, avg_gl_rd_b)

    # TrueSkill
    ts_a = [trueskill.Rating(mu, sigma) for mu, sigma in [get_p_rating(p, "ts") for p in player_ids_a]]
    ts_b = [trueskill.Rating(mu, sigma) for mu, sigma in [get_p_rating(p, "ts") for p in player_ids_b]]

    def ts_win_prob(team1, team2):
        mu_a = sum(r.mu for r in team1) / len(team1)
        mu_b = sum(r.mu for r in team2) / len(team2)
        sigma2_a = sum(r.sigma**2 for r in team1) / len(team1)
        sigma2_b = sum(r.sigma**2 for r in team2) / len(team2)
        delta_mu = mu_a - mu_b
        denom = math.sqrt(sigma2_a + sigma2_b + 2 * (_ts_model.beta**2))
        return _ts_model.cdf(delta_mu / denom)

    features["player_ts"] = ts_win_prob(ts_a, ts_b)

    # OpenSkill (PlackettLuce)
    os_a = [_os_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "os") for p in player_ids_a]]
    os_b = [_os_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "os") for p in player_ids_b]]
    features["player_os"] = _os_model.predict_win([os_a, os_b])[0]

    # PlackettLuce
    pl_a = [_pl_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "pl") for p in player_ids_a]]
    pl_b = [_pl_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "pl") for p in player_ids_b]]
    features["player_pl"] = _pl_model.predict_win([pl_a, pl_b])[0]

    # Thurstone
    tm_a = [_tm_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "tm") for p in player_ids_a]]
    tm_b = [_tm_model.rating(mu, sigma) for mu, sigma in [get_p_rating(p, "tm") for p in player_ids_b]]
    features["player_tm"] = _tm_model.predict_win([tm_a, tm_b])[0]

    # 2. Uncertainty features (14 features)
    # player_elo_min1/min2: min of player elos in each team
    features["player_elo_min1"] = min(elo_a)
    features["player_elo_min2"] = min(elo_b)

    # player_gl_max1/max2: max of player glicko ratings in each team
    features["player_gl_max1"] = max(r for r, rd in gl_a)
    features["player_gl_max2"] = max(r for r, rd in gl_b)

    # player_gl_rd_avg1/avg2: average RD
    features["player_gl_rd_avg1"] = sum(rd for r, rd in gl_a) / len(gl_a)
    features["player_gl_rd_avg2"] = sum(rd for r, rd in gl_b) / len(gl_b)

    # sigma_avg for ts, os, pl, tm
    features["player_ts_sigma_avg1"] = sum(r.sigma for r in ts_a) / len(ts_a)
    features["player_ts_sigma_avg2"] = sum(r.sigma for r in ts_b) / len(ts_b)
    features["player_os_sigma_avg1"] = sum(r.sigma for r in os_a) / len(os_a)
    features["player_os_sigma_avg2"] = sum(r.sigma for r in os_b) / len(os_b)
    features["player_pl_sigma_avg1"] = sum(r.sigma for r in pl_a) / len(pl_a)
    features["player_pl_sigma_avg2"] = sum(r.sigma for r in pl_b) / len(pl_b)
    features["player_tm_sigma_avg1"] = sum(r.sigma for r in tm_a) / len(tm_a)
    features["player_tm_sigma_avg2"] = sum(r.sigma for r in tm_b) / len(tm_b)

    # 3. Rolling features (20 features)
    for stat, default in _DEFAULT_ROLLING.items():
        val_a = w20_a.get(f"avg_{stat}") if w20_a and f"avg_{stat}" in w20_a else default
        val_b = w20_b.get(f"avg_{stat}") if w20_b and f"avg_{stat}" in w20_b else default
        if stat == "win_rate":
            val_a = w20_a.get("win_rate", default) if w20_a else default
            val_b = w20_b.get("win_rate", default) if w20_b else default
        features[f"t1_rolling_{stat}"] = float(val_a) if val_a is not None else default
        features[f"t2_rolling_{stat}"] = float(val_b) if val_b is not None else default

    # 4. Binomial features (6 features)
    rating_probs_array = np.array([features[f"player_{s}"] for s in RATING_SYSTEMS])
    series_probs = _series_probability(rating_probs_array, best_of)
    for i, system in enumerate(RATING_SYSTEMS):
        features[f"player_{system}_binom_series"] = float(series_probs[i])

    # Build feature vector in correct order
    feature_vector = np.array([[features.get(f, 0.0) for f in ALL_FEATURES]])

    diagnostics["features_built"] = len(ALL_FEATURES)
    diagnostics["missing_count"] = len(diagnostics["missing"])

    # Consistency check: verify GOL.GG team names (from IDs) match canonical names
    if team_a_golgg and team_b_golgg:
        g_norm_a = normalize_team_name(team_a_golgg)
        g_norm_b = normalize_team_name(team_b_golgg)
        c_norm_a = normalize_team_name(team_a_name)
        c_norm_b = normalize_team_name(team_b_name)

        a_matches = g_norm_a == c_norm_a or g_norm_a in c_norm_a or c_norm_a in g_norm_a
        b_matches = g_norm_b == c_norm_b or g_norm_b in c_norm_b or c_norm_b in g_norm_b
        crossed = not a_matches and not b_matches and (g_norm_a == c_norm_b or g_norm_b == c_norm_a)

        diagnostics["side_consistency"] = {
            "team_a_ok": a_matches,
            "team_b_ok": b_matches,
            "cross_swapped": crossed,
        }
        if crossed:
            diagnostics["missing"].append(f"side_swap_detected:golgg_ids appear swapped vs canonical team order")

    return feature_vector, diagnostics


def _swap_feature_vector(vec: np.ndarray) -> np.ndarray:
    """Swap team orientation in feature vector.

    For rating probs: p -> 1-p
    For rolling features: swap t1 <-> t2
    For binomial: p -> 1-p
    """
    swapped = vec.copy()

    # Rating probs (indices 0-5): p -> 1-p
    for i in range(6):
        swapped[0, i] = 1.0 - vec[0, i]

    # Uncertainty features (indices 6-19): swap team-specific values
    # player_elo_min1/min2: swap min/max
    swapped[0, 6] = vec[0, 7]  # min1 <-> min2
    swapped[0, 7] = vec[0, 6]
    # player_gl_max1/max2: swap
    swapped[0, 8] = vec[0, 9]
    swapped[0, 9] = vec[0, 8]
    # player_gl_rd_avg1/avg2: swap
    swapped[0, 10] = vec[0, 11]
    swapped[0, 11] = vec[0, 10]
    # sigma_avg1/avg2 for ts, os, pl, tm: swap
    for offset in range(4):
        base = 12 + offset * 2
        swapped[0, base] = vec[0, base + 1]
        swapped[0, base + 1] = vec[0, base]

    # Rolling features (indices 20-39): swap t1 <-> t2
    for i in range(10):
        t1_idx = 20 + i * 2
        t2_idx = 20 + i * 2 + 1
        swapped[0, t1_idx] = vec[0, t2_idx]
        swapped[0, t2_idx] = vec[0, t1_idx]

    # Binomial features (indices 40-45): p -> 1-p
    for i in range(6):
        swapped[0, 40 + i] = 1.0 - vec[0, 40 + i]

    return swapped


def _symmetrize(original: float, swapped: float) -> float:
    """Apply order symmetry: average original and (1-swapped)."""
    return 0.5 * (original + (1.0 - swapped))


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------
def predict_upcoming_with_thesis_model(
    *,
    ratings_version: str = "latest-full",
    w20_version: str = "w20-latest",
    include_past: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Generate predictions for upcoming matches using the thesis model.

    Returns list of prediction results with diagnostics.
    """
    pipeline, calibrator = _load_model()

    # Shared loader aligns raw bookmaker offer order to canonical A/B before
    # returning GOL.GG IDs. Bookmakers routinely reverse team ordering.
    matches = load_canonical_matches(include_past=include_past, limit=limit)

    if not matches:
        return []

    # Register model artifact
    model_artifact_id = _register_thesis_model()

    results: list[dict[str, Any]] = []

    with transaction() as conn:
        # Mark old predictions as stale so only the latest per match stays active
        conn.execute(
            """
            UPDATE canonical_predictions
            SET prediction_status = 'stale'
            WHERE prediction_status = 'active' 
              AND model_name = ? AND model_version = ?
            """,
            (THESIS_MODEL_NAME, THESIS_MODEL_VERSION),
        )

        for match in matches:
            match_id = int(match["id"])
            team_a = str(match["team_a_name"])
            team_b = str(match["team_b_name"])
            best_of = int(match["best_of"]) if match["best_of"] is not None else 1
            # GOL.GG IDs from upcoming_matches (may be None)
            golgg_a = int(match["team_a_golgg_id"]) if match.get("team_a_golgg_id") is not None else None
            golgg_b = int(match["team_b_golgg_id"]) if match.get("team_b_golgg_id") is not None else None

            # Build features using GOL.GG IDs (bypasses suggest_mapping)
            roster_overrides = _load_roster_overrides(match_id)
            feature_vec, diagnostics = build_thesis_features_for_match(
                team_a, team_b,
                team_a_golgg_id=golgg_a,
                team_b_golgg_id=golgg_b,
                league=str(match.get("league") or ""),
                match_date=str(match.get("start_time_normalized") or ""),
                ratings_version=ratings_version,
                w20_version=w20_version,
                best_of=best_of,
                team_a_roster_override=roster_overrides.get("a"),
                team_b_roster_override=roster_overrides.get("b"),
            )

            if feature_vec is None:
                diagnostics["error"] = "Could not build features"
                continue

            # Original prediction
            original_prob = float(np.clip(pipeline.predict_proba(feature_vec)[0, 1], EPSILON, 1.0 - EPSILON))

            # Swapped prediction
            swapped_vec = _swap_feature_vector(feature_vec)
            swapped_prob = float(np.clip(pipeline.predict_proba(swapped_vec)[0, 1], EPSILON, 1.0 - EPSILON))

            # Order symmetry
            sym_prob = _symmetrize(original_prob, swapped_prob)

            # Platt calibration
            calibrated_prob = float(np.clip(
                calibrator.predict_proba(_logit(np.array([sym_prob])))[0, 1],
                EPSILON, 1.0 - EPSILON,
            ))

            diagnostics["original_prob"] = original_prob
            diagnostics["swapped_prob"] = swapped_prob
            diagnostics["symmetric_prob"] = sym_prob
            diagnostics["calibrated_prob"] = calibrated_prob

            # Store prediction
            predicted_at = datetime.now(UTC).replace(microsecond=0).isoformat()
            conn.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    model_artifact_id,
                    THESIS_MODEL_NAME,
                    THESIS_MODEL_VERSION,
                    predicted_at,
                    calibrated_prob,
                    1.0 - calibrated_prob,
                    "thesis-exp039",
                    ratings_version,
                    None,
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )

            # Fetch the inserted prediction_id
            row = conn.execute(
                """
                SELECT id FROM canonical_predictions
                WHERE canonical_match_id = ? AND model_name = ? AND model_version = ?
                ORDER BY predicted_at DESC LIMIT 1
                """,
                (match_id, THESIS_MODEL_NAME, THESIS_MODEL_VERSION),
            ).fetchone()

            results.append({
                "prediction_id": row["id"] if row else None,
                "canonical_match_id": match_id,
                "match": f"{team_a} vs {team_b}",
                "prob_a": calibrated_prob,
                "prob_b": 1.0 - calibrated_prob,
                "diagnostics": diagnostics,
            })

    return results


def _register_thesis_model() -> int:
    """Register the thesis model in model_artifacts."""
    feature_schema = {
        "features": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "source": "thesis EXP-039 Sym-Cal LR-ElasticNet-W20-Binomial",
        "training_matches": 13289,
        "training_logloss": 0.5850,
        "training_auc": 0.7555,
    }
    params = {
        "C": 0.03297234640536737,
        "penalty": "elasticnet",
        "l1_ratio": 0.9439657999531195,
        "solver": "saga",
        "max_iter": 5000,
        "platt_C": 1.0,
        "order_symmetry": True,
    }

    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, feature_schema_json, model_params_json, status
            ) VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(model_name, model_version) DO UPDATE SET
                feature_schema_json = excluded.feature_schema_json,
                model_params_json = excluded.model_params_json,
                status = 'active'
            """,
            (
                THESIS_MODEL_NAME,
                THESIS_MODEL_VERSION,
                json.dumps(feature_schema, ensure_ascii=False, sort_keys=True),
                json.dumps(params, ensure_ascii=False, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT id FROM model_artifacts WHERE model_name = ? AND model_version = ?",
            (THESIS_MODEL_NAME, THESIS_MODEL_VERSION),
        ).fetchone()
        return int(row["id"])


# ---------------------------------------------------------------------------
# Thesis Hybrid (thesis model + market odds)
# ---------------------------------------------------------------------------

def _register_thesis_hybrid_model(
    *, alpha: float, temperature: float, version: str
) -> int:
    """Register the thesis hybrid model in model_artifacts."""
    feature_schema = {
        "base_model": f"{THESIS_MODEL_NAME}/{THESIS_MODEL_VERSION}",
        "market_signal": "average no-vig probability from latest bookmaker odds",
        "formula": "alpha * temperature(thesis_model_probability) + (1-alpha) * market_probability",
        "historical_reference": "thesis EXP-039 hybrid with market",
    }
    params = {"alpha": alpha, "temperature": temperature}
    with transaction() as conn:
        conn.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, feature_schema_json, model_params_json, status
            ) VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(model_name, model_version) DO UPDATE SET
                feature_schema_json = excluded.feature_schema_json,
                model_params_json = excluded.model_params_json,
                status = 'active'
            """,
            (
                THESIS_HYBRID_MODEL_NAME,
                version,
                json.dumps(feature_schema, ensure_ascii=False, sort_keys=True),
                json.dumps(params, ensure_ascii=False, sort_keys=True),
            ),
        )
        row = conn.execute(
            "SELECT id FROM model_artifacts WHERE model_name = ? AND model_version = ?",
            (THESIS_HYBRID_MODEL_NAME, version),
        ).fetchone()
        return int(row["id"])


def generate_thesis_hybrid_predictions(
    *,
    alpha: float = THESIS_HYBRID_ALPHA,
    temperature: float = THESIS_HYBRID_TEMPERATURE,
    hybrid_model_name: str = THESIS_HYBRID_MODEL_NAME,
    hybrid_model_version: str | None = None,
) -> list[dict[str, Any]]:
    """Blend thesis model probabilities with average no-vig bookmaker market.

    Formula mirrors the existing hybrid approach:

    ``p_hybrid = alpha * temperature(thesis_prob, T) + (1-alpha) * p_market``.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if hybrid_model_version is None:
        hybrid_model_version = f"a{alpha:.2f}-t{temperature:.2f}"

    model_artifact_id = _register_thesis_hybrid_model(
        alpha=alpha, temperature=temperature, version=hybrid_model_version
    )

    # Get latest thesis predictions + odds
    rows = query_df(
        """
        WITH latest_predictions AS (
            SELECT p.*
            FROM canonical_predictions p
            JOIN (
                SELECT canonical_match_id, model_name, model_version, MAX(predicted_at) AS predicted_at
                FROM canonical_predictions
                WHERE prediction_status = 'active' AND model_name = ? AND model_version = ?
                GROUP BY canonical_match_id, model_name, model_version
            ) lp ON lp.canonical_match_id = p.canonical_match_id
                AND lp.model_name = p.model_name
                AND lp.model_version = p.model_version
                AND lp.predicted_at = p.predicted_at
        ), latest_odds AS (
            SELECT os.*
            FROM odds_snapshots os
            JOIN (
                SELECT canonical_match_id, bookmaker_id, MAX(scraped_at) AS scraped_at
                FROM odds_snapshots
                WHERE market_type = 'match_winner' AND COALESCE(is_live, 0) = 0
                GROUP BY canonical_match_id, bookmaker_id
            ) lo ON lo.canonical_match_id = os.canonical_match_id
                 AND lo.bookmaker_id = os.bookmaker_id
                 AND lo.scraped_at = os.scraped_at
        )
        SELECT lp.id AS base_prediction_id, lp.canonical_match_id, lp.prob_a AS model_prob_a,
               lp.features_version, lp.ratings_version, lp.data_cutoff_at,
               cm.normalized_team_a, cm.normalized_team_b, cm.league, cm.start_time_normalized,
               os.raw_team_a, os.raw_team_b, os.odds_a, os.odds_b
        FROM latest_predictions lp
        JOIN canonical_matches cm ON cm.id = lp.canonical_match_id
        JOIN latest_odds os ON os.canonical_match_id = lp.canonical_match_id
        """,
        (THESIS_MODEL_NAME, THESIS_MODEL_VERSION),
    )

    if rows.empty:
        return []

    results: list[dict[str, Any]] = []
    canonical_match_ids = [int(mid) for mid in rows["canonical_match_id"].dropna().unique().tolist()]
    with transaction() as connection:
        # Mark old hybrid predictions as stale for matches that are about to be
        # regenerated, regardless of hybrid version.  Otherwise changing alpha
        # leaves multiple active Hybrid-Thesis-Market versions for the same
        # upcoming match, which is confusing for generic prediction endpoints.
        if canonical_match_ids:
            placeholders = ",".join("?" for _ in canonical_match_ids)
            connection.execute(
                f"""
                UPDATE canonical_predictions
                SET prediction_status = 'stale'
                WHERE prediction_status = 'active'
                  AND model_name = ?
                  AND canonical_match_id IN ({placeholders})
                """,
                (hybrid_model_name, *canonical_match_ids),
            )

        for canonical_match_id, group in rows.groupby("canonical_match_id"):
            market_probs: list[float] = []
            first = group.iloc[0].to_dict()

            for row in group.to_dict("records"):
                aligned = align_snapshot_odds(
                    str(row.get("normalized_team_a") or ""),
                    str(row.get("normalized_team_b") or ""),
                    str(row.get("raw_team_a") or ""),
                    str(row.get("raw_team_b") or ""),
                    row.get("odds_a"),
                    row.get("odds_b"),
                    league=row.get("league"), source_system="bookmaker",
                    match_date=row.get("start_time_normalized"),
                )
                if aligned is None:
                    continue
                odds_a, odds_b = aligned
                if odds_a is None or odds_b is None or float(odds_a) <= 1.0 or float(odds_b) <= 1.0:
                    logger.warning(
                        "Skipping invalid market odds for canonical_match_id=%s odds=(%s,%s)",
                        canonical_match_id,
                        odds_a,
                        odds_b,
                    )
                    continue
                try:
                    market_a, _ = fair_market_probabilities(float(odds_a), float(odds_b))
                except ValueError as exc:
                    logger.warning(
                        "Skipping invalid market odds for canonical_match_id=%s odds=(%s,%s): %s",
                        canonical_match_id,
                        odds_a,
                        odds_b,
                        exc,
                    )
                    continue
                market_probs.append(market_a)

            if not market_probs:
                continue

            model_prob = float(first["model_prob_a"])
            model_t = apply_temperature_probability(model_prob, temperature)
            market_prob = sum(market_probs) / len(market_probs)
            hybrid_prob = max(0.001, min(0.999, alpha * model_t + (1.0 - alpha) * market_prob))

            diagnostics = {
                "base_model_name": THESIS_MODEL_NAME,
                "base_model_version": THESIS_MODEL_VERSION,
                "base_prediction_id": int(first["base_prediction_id"]),
                "alpha": alpha,
                "temperature": temperature,
                "model_prob_a": model_prob,
                "model_prob_a_temperature": model_t,
                "market_prob_a_avg_no_vig": market_prob,
                "bookmakers_used": len(market_probs),
                "formula": "alpha * temp(thesis_model) + (1-alpha) * average_no_vig_market",
            }

            connection.execute(
                """
                INSERT INTO canonical_predictions(
                    canonical_match_id, model_artifact_id, model_name, model_version, predicted_at,
                    prob_a, prob_b, features_version, ratings_version, data_cutoff_at, diagnostics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(canonical_match_id),
                    model_artifact_id,
                    hybrid_model_name,
                    hybrid_model_version,
                    datetime.now(UTC).isoformat(),
                    hybrid_prob,
                    1.0 - hybrid_prob,
                    first.get("features_version"),
                    first.get("ratings_version"),
                    first.get("data_cutoff_at"),
                    json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
                ),
            )

            row = connection.execute(
                "SELECT id FROM canonical_predictions WHERE canonical_match_id = ? AND model_name = ? AND model_version = ? ORDER BY predicted_at DESC LIMIT 1",
                (int(canonical_match_id), hybrid_model_name, hybrid_model_version),
            ).fetchone()

            results.append({
                "prediction_id": int(row["id"]),
                "canonical_match_id": int(canonical_match_id),
                "prob_a": hybrid_prob,
                "prob_b": 1.0 - hybrid_prob,
                "diagnostics": diagnostics,
            })

    return results
