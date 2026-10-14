#!/usr/bin/env python3
"""Re-predict finished matches using exact GOL.GG team names, bypassing suggest_mapping().

This script queries finished canonical matches that have golgg_match_mappings,
uses the GOL.GG team1_name/team2_name directly (the order GOL.GG considers canonical),
builds the 46-feature vector for the Sym-Cal LR-ElasticNet-W20-Binomial model,
and stores predictions in canonical_predictions.

Usage:
    DATABASE_URL=postgresql://betting:betting@localhost:5432/betting \\
    python betting_app/scripts/re_predict_finished_golgg.py

    # Or if DATABASE_URL is already exported:
    python betting_app/scripts/re_predict_finished_golgg.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Force DATABASE_URL before importing project modules
_DEFAULT_PG_URL = "postgresql://betting:betting@localhost:5432/betting"
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = _DEFAULT_PG_URL

from betting_app.core.db import query_df, transaction
from betting_app.core.matching import normalize_team_name
from betting_app.services.upcoming_inference_service import (
    RATING_SYSTEMS,
    load_team_ratings,
    load_w20,
)

# ---------------------------------------------------------------------------
# Constants (mirrored from thesis_inference_service)
# ---------------------------------------------------------------------------
THESIS_MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
THESIS_MODEL_VERSION = "exp-039-golgg-corrected"
THESIS_HYBRID_MODEL_NAME = "Hybrid-Thesis-Market"
THESIS_HYBRID_VERSION = "a0.50-t0.80-golgg"
THESIS_HYBRID_ALPHA = 0.50
THESIS_HYBRID_TEMPERATURE = 0.80
EPSILON = 0.001

ARTEFACT_DIR = PROJECT_ROOT / "betting_app" / "models"
PIPELINE_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"
METADATA_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_metadata.json"

# Feature definitions
OPTUNA_BASE_FEATURES = [
    "player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm",
    "player_elo_min1", "player_elo_min2",
    "player_gl_max1", "player_gl_max2",
    "player_gl_rd_avg1", "player_gl_rd_avg2",
    "player_ts_sigma_avg1", "player_ts_sigma_avg2",
    "player_os_sigma_avg1", "player_os_sigma_avg2",
    "player_pl_sigma_avg1", "player_pl_sigma_avg2",
    "player_tm_sigma_avg1", "player_tm_sigma_avg2",
]

ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate", "t2_rolling_win_rate",
    "t1_rolling_kills", "t2_rolling_kills",
    "t1_rolling_deaths", "t2_rolling_deaths",
    "t1_rolling_gd15", "t2_rolling_gd15",
    "t1_rolling_dpm", "t2_rolling_dpm",
    "t1_rolling_vspm", "t2_rolling_vspm",
    "t1_rolling_towers", "t2_rolling_towers",
    "t1_rolling_nashors", "t2_rolling_nashors",
    "t1_rolling_gold", "t2_rolling_gold",
    "t1_rolling_duration", "t2_rolling_duration",
]

BINOMIAL_FEATURES = [f"player_{s}_binom_series" for s in ["elo", "gl", "ts", "os", "pl", "tm"]]
ALL_FEATURES = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + BINOMIAL_FEATURES

_DEFAULT_ROLLING = {
    "win_rate": 0.5, "kills": 12.0, "deaths": 12.0, "gd15": 0.0,
    "dpm": 1800.0, "vspm": 7.0, "towers": 5.0, "nashors": 0.5,
    "gold": 55000.0, "duration": 1800.0,
}


# ---------------------------------------------------------------------------
# Helpers (mirrored from thesis_inference_service)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def _rating_prob(rating_a: float | None, rating_b: float | None, system: str) -> float:
    if rating_a is None or rating_b is None:
        return 0.5
    diff = float(rating_a) - float(rating_b)
    if system in {"elo", "gl"}:
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))
    elif system == "os":
        return _sigmoid(diff / 5.0)
    else:
        return _sigmoid(diff / 8.333)


def _series_probability(map_prob: np.ndarray, best_of: int) -> np.ndarray:
    prob = np.clip(map_prob.astype(float), EPSILON, 1.0 - EPSILON)
    if best_of <= 1:
        return prob
    needed = best_of // 2 + 1
    series_prob = np.zeros_like(prob)
    for wins in range(needed, best_of + 1):
        series_prob += comb(best_of, wins) * np.power(prob, wins) * np.power(1.0 - prob, best_of - wins)
    return np.clip(series_prob, EPSILON, 1.0 - EPSILON)


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _swap_feature_vector(vec: np.ndarray) -> np.ndarray:
    swapped = vec.copy()
    # Rating probs (indices 0-5): p -> 1-p
    for i in range(6):
        swapped[0, i] = 1.0 - vec[0, i]
    # Uncertainty min1/min2 (indices 6-7)
    swapped[0, 6] = vec[0, 7]
    swapped[0, 7] = vec[0, 6]
    # max1/max2 (indices 8-9)
    swapped[0, 8] = vec[0, 9]
    swapped[0, 9] = vec[0, 8]
    # rd_avg1/avg2 (indices 10-11)
    swapped[0, 10] = vec[0, 11]
    swapped[0, 11] = vec[0, 10]
    # sigma_avg for ts, os, pl, tm (indices 12-19)
    for offset in range(4):
        base = 12 + offset * 2
        swapped[0, base] = vec[0, base + 1]
        swapped[0, base + 1] = vec[0, base]
    # Rolling (indices 20-39): swap t1<->t2
    for i in range(10):
        t1_idx = 20 + i * 2
        t2_idx = 20 + i * 2 + 1
        swapped[0, t1_idx] = vec[0, t2_idx]
        swapped[0, t2_idx] = vec[0, t1_idx]
    # Binomial (indices 40-45): p -> 1-p
    for i in range(6):
        swapped[0, 40 + i] = 1.0 - vec[0, 40 + i]
    return swapped


def _symmetrize(original: float, swapped: float) -> float:
    return 0.5 * (original + (1.0 - swapped))


# ---------------------------------------------------------------------------
# Feature building using GOL.GG team names directly (NO suggest_mapping)
# ---------------------------------------------------------------------------

def build_features_with_golgg_names(
    team_a_golgg: str,
    team_b_golgg: str,
    *,
    ratings_version: str = "latest-full",
    w20_version: str = "w20-latest",
    best_of: int = 1,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build the 46-feature vector using exact GOL.GG team names.

    This bypasses suggest_mapping() entirely — it uses GOL.GG names directly
    to look up ratings and W20 features.
    """
    diagnostics: dict[str, Any] = {
        "team_a_golgg": team_a_golgg,
        "team_b_golgg": team_b_golgg,
        "ratings_version": ratings_version,
        "w20_version": w20_version,
        "best_of": best_of,
        "missing": [],
    }

    # Load team ratings using exact GOL.GG names
    ratings_a = load_team_ratings(team_a_golgg, ratings_version)
    ratings_b = load_team_ratings(team_b_golgg, ratings_version)
    for system in RATING_SYSTEMS:
        if system not in ratings_a:
            diagnostics["missing"].append(f"team_a_rating:{system}")
        if system not in ratings_b:
            diagnostics["missing"].append(f"team_b_rating:{system}")

    # Load W20 rolling features
    w20_a = load_w20(team_a_golgg, w20_version)
    w20_b = load_w20(team_b_golgg, w20_version)
    if not w20_a:
        diagnostics["missing"].append("team_a_w20")
    if not w20_b:
        diagnostics["missing"].append("team_b_w20")

    features: dict[str, float] = {}

    # 1. Rating probabilities (6 features)
    for system in RATING_SYSTEMS:
        r_a = ratings_a.get(system, {}).get("rating_value")
        r_b = ratings_b.get(system, {}).get("rating_value")
        features[f"player_{system}"] = _rating_prob(r_a, r_b, system)

    # 2. Uncertainty features
    # Elo RD: min1/min2
    elo_rd_a = ratings_a.get("elo", {}).get("rd")
    elo_rd_b = ratings_b.get("elo", {}).get("rd")
    if elo_rd_a is not None and elo_rd_b is not None:
        features["player_elo_min1"] = min(float(elo_rd_a), float(elo_rd_b))
        features["player_elo_min2"] = max(float(elo_rd_a), float(elo_rd_b))
    else:
        features["player_elo_min1"] = 50.0
        features["player_elo_min2"] = 50.0

    # Glicko RD: max1/max2
    gl_rd_a = ratings_a.get("gl", {}).get("rd")
    gl_rd_b = ratings_b.get("gl", {}).get("rd")
    if gl_rd_a is not None and gl_rd_b is not None:
        features["player_gl_max1"] = max(float(gl_rd_a), float(gl_rd_b))
        features["player_gl_max2"] = min(float(gl_rd_a), float(gl_rd_b))
    else:
        features["player_gl_max1"] = 0.1
        features["player_gl_max2"] = 0.1

    features["player_gl_rd_avg1"] = float(gl_rd_a) if gl_rd_a is not None else 0.1
    features["player_gl_rd_avg2"] = float(gl_rd_b) if gl_rd_b is not None else 0.1

    # Sigma averages for ts, os, pl, tm
    for system in ["ts", "os", "pl", "tm"]:
        sigma_a = ratings_a.get(system, {}).get("sigma")
        sigma_b = ratings_b.get(system, {}).get("sigma")
        features[f"player_{system}_sigma_avg1"] = float(sigma_a) if sigma_a is not None else 0.06
        features[f"player_{system}_sigma_avg2"] = float(sigma_b) if sigma_b is not None else 0.06

    # 3. Rolling features (20 features)
    for stat, default in _DEFAULT_ROLLING.items():
        if stat == "win_rate":
            val_a = w20_a.get("win_rate", default) if w20_a else default
            val_b = w20_b.get("win_rate", default) if w20_b else default
        else:
            val_a = w20_a.get(f"avg_{stat}", default) if w20_a and f"avg_{stat}" in w20_a else default
            val_b = w20_b.get(f"avg_{stat}", default) if w20_b and f"avg_{stat}" in w20_b else default
        features[f"t1_rolling_{stat}"] = float(val_a) if val_a is not None else default
        features[f"t2_rolling_{stat}"] = float(val_b) if val_b is not None else default

    # 4. Binomial features (6 features)
    rating_probs = np.array([features[f"player_{s}"] for s in ["elo", "gl", "ts", "os", "pl", "tm"]])
    series_probs = _series_probability(rating_probs, best_of)
    for i, system in enumerate(["elo", "gl", "ts", "os", "pl", "tm"]):
        features[f"player_{system}_binom_series"] = float(series_probs[i])

    # Build vector in correct order
    feature_vector = np.array([[features.get(f, 0.0) for f in ALL_FEATURES]])

    diagnostics["features_built"] = len(ALL_FEATURES)
    diagnostics["missing_count"] = len(diagnostics["missing"])
    diagnostics["feature_vector"] = feature_vector.tolist()

    return feature_vector, diagnostics


# ---------------------------------------------------------------------------
# Main re-prediction logic
# ---------------------------------------------------------------------------

def load_finished_matches_with_golgg() -> list[dict[str, Any]]:
    """Load all finished canonical matches that have GOL.GG mappings."""
    sql = """
        SELECT cm.id AS canonical_match_id,
               cm.team_a_name, cm.team_b_name, cm.winner_name AS canonical_winner,
               cm.best_of, cm.start_time_normalized, cm.league,
               gm.match_id AS golgg_match_id,
               gm.team1_name AS golgg_team1, gm.team2_name AS golgg_team2,
               gm.winner_name AS golgg_winner,
               gm.date AS golgg_date
        FROM canonical_matches cm
        JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        WHERE cm.status = 'finished'
        ORDER BY gm.date ASC, cm.id ASC
    """
    df = query_df(sql)
    return df.to_dict("records") if not df.empty else []


def predict_finished_with_golgg() -> list[dict[str, Any]]:
    """Re-predict all finished matches using exact GOL.GG team names.

    Steps:
    1. Load finished matches with GOL.GG data
    2. Use golgg team1/team2 as the canonical order
    3. Build features directly (bypassing suggest_mapping)
    4. Run model with order symmetry + Platt calibration
    5. Store predictions in canonical_predictions
    """
    # Load pipeline and calibrator
    print(f"[*] Loading model pipeline from {PIPELINE_PATH}")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline not found: {PIPELINE_PATH}")
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")

    pipeline = joblib.load(PIPELINE_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    print(f"[+] Model loaded: pipeline={type(pipeline).__name__}, calibrator={type(calibrator).__name__}")

    matches = load_finished_matches_with_golgg()
    print(f"[+] Loaded {len(matches)} finished matches with GOL.GG data")

    if not matches:
        return []

    # Register the model artifact for this corrected run
    model_artifact_id = _register_artifact()

    results: list[dict[str, Any]] = []

    for match in matches:
        cm_id = int(match["canonical_match_id"])
        golgg_team1 = str(match["golgg_team1"])
        golgg_team2 = str(match["golgg_team2"])
        actual_winner = str(match["golgg_winner"] or match["canonical_winner"] or "")
        best_of = int(match["best_of"]) if match["best_of"] is not None else 1

        # Optional: skip if no best_of info (default to 1)
        print(f"  [{cm_id}] {golgg_team1} vs {golgg_team2} (Bo{best_of}) ... ", end="")

        # Build features using GOL.GG names directly
        feature_vec, diagnostics = build_features_with_golgg_names(
            golgg_team1, golgg_team2,
            best_of=best_of,
        )

        if feature_vec is None:
            print(f"SKIP - features failed: {diagnostics.get('skip_reason', 'unknown')}")
            results.append({
                "canonical_match_id": cm_id,
                "match": f"{golgg_team1} vs {golgg_team2}",
                "status": "skipped",
                "diagnostics": diagnostics,
            })
            continue

        # Original prediction
        original_prob = float(np.clip(
            pipeline.predict_proba(feature_vec)[0, 1],
            EPSILON, 1.0 - EPSILON,
        ))

        # Swapped prediction
        swapped_vec = _swap_feature_vector(feature_vec)
        swapped_prob = float(np.clip(
            pipeline.predict_proba(swapped_vec)[0, 1],
            EPSILON, 1.0 - EPSILON,
        ))

        # Order symmetry
        sym_prob = _symmetrize(original_prob, swapped_prob)

        # Platt calibration
        calibrated_prob = float(np.clip(
            calibrator.predict_proba(_logit(np.array([sym_prob])))[0, 1],
            EPSILON, 1.0 - EPSILON,
        ))

        # prob_a = P(golgg_team1 wins)
        prob_a = calibrated_prob
        prob_b = 1.0 - calibrated_prob

        diagnostics["original_prob"] = original_prob
        diagnostics["swapped_prob"] = swapped_prob
        diagnostics["symmetric_prob"] = sym_prob
        diagnostics["calibrated_prob"] = calibrated_prob
        diagnostics["golgg_team1"] = golgg_team1
        diagnostics["golgg_team2"] = golgg_team2
        diagnostics["golgg_winner"] = actual_winner
        diagnostics["model_prediction"] = {
            "predicted_winner": golgg_team1 if prob_a >= 0.5 else golgg_team2,
            "prob_a": prob_a,
            "prob_b": prob_b,
        }

        # Determine correctness if we know the actual winner
        if actual_winner:
            predicted_winner = golgg_team1 if prob_a >= 0.5 else golgg_team2
            correct = (normalize_team_name(predicted_winner) == normalize_team_name(actual_winner))
            diagnostics["correct_prediction"] = correct
        else:
            diagnostics["correct_prediction"] = None

        # Store prediction
        predicted_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        try:
            with transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO canonical_predictions(
                        canonical_match_id, model_artifact_id, model_name, model_version,
                        predicted_at, prob_a, prob_b,
                        features_version, ratings_version, data_cutoff_at, diagnostics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cm_id,
                        model_artifact_id,
                        THESIS_MODEL_NAME,
                        THESIS_MODEL_VERSION,
                        predicted_at,
                        prob_a,
                        prob_b,
                        "thesis-exp039-golgg",
                        "latest-full",
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
                    (cm_id, THESIS_MODEL_NAME, THESIS_MODEL_VERSION),
                ).fetchone()
                pred_id = int(row["id"]) if row else None

            correct_str = "✓" if diagnostics.get("correct_prediction") else "✗" if diagnostics.get("correct_prediction") is False else "?"
            print(f"DONE (prob_a={prob_a:.4f}, prob_b={prob_b:.4f}, correct={correct_str}) pred_id={pred_id}")

            results.append({
                "prediction_id": pred_id,
                "canonical_match_id": cm_id,
                "match": f"{golgg_team1} vs {golgg_team2}",
                "prob_a": prob_a,
                "prob_b": prob_b,
                "status": "predicted",
                "correct": diagnostics.get("correct_prediction"),
                "actual_winner": actual_winner,
                "diagnostics": diagnostics,
            })

        except Exception as e:
            print(f"DB ERROR: {e}")
            results.append({
                "canonical_match_id": cm_id,
                "match": f"{golgg_team1} vs {golgg_team2}",
                "status": "db_error",
                "error": str(e),
            })
            continue

    return results


def _register_artifact() -> int:
    """Register the golgg-corrected model version in model_artifacts."""
    feature_schema = {
        "features": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "source": "exp-039-golgg-corrected: using exact GOL.GG team names, no fuzzy mapping",
        "parent_model": "Sym-Cal LR-ElasticNet-W20-Binomial/exp-039",
        "note": "Re-prediction on finished matches with correct team alignment from GOL.GG",
    }
    params = {
        "order_symmetry": True,
        "platt_scaled": True,
        "team_source": "golgg_direct",
        "no_fuzzy_mapping": True,
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


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute LogLoss and accuracy from prediction results."""
    predicted = [r for r in results if r.get("correct") is not None]
    if not predicted:
        return {"n": 0, "note": "no predictions with known outcomes"}

    n = len(predicted)
    correct_count = sum(1 for r in predicted if r.get("correct"))
    accuracy = correct_count / n

    # LogLoss
    logloss_sum = 0.0
    for r in predicted:
        prob = r.get("prob_a", 0.5)
        # Actual: 1 if golgg_team1 == actual_winner
        # We stored prob_a = P(golgg_team1 wins)
        # r["correct"] is True if predicted_winner == actual_winner
        # But for LogLoss we need: y_true = 1 if golgg_team1 actually won
        # Let's figure it out from the diagnostics
        diag = r.get("diagnostics", {})
        golgg_team1 = diag.get("golgg_team1", "")
        golgg_winner = diag.get("golgg_winner", "")
        y_true = 1.0 if normalize_team_name(golgg_team1) == normalize_team_name(golgg_winner) else 0.0
        prob_clipped = max(EPSILON, min(1.0 - EPSILON, prob))
        logloss_sum += -(y_true * math.log(prob_clipped) + (1 - y_true) * math.log(1 - prob_clipped))

    logloss = logloss_sum / n

    return {
        "n": n,
        "accuracy": accuracy,
        "correct": correct_count,
        "incorrect": n - correct_count,
        "logloss": logloss,
    }


def compare_with_previous() -> list[dict[str, Any]]:
    """Compare golgg-corrected predictions with the original exp-039 predictions."""
    sql = """
    WITH original AS (
        SELECT cp.canonical_match_id, cp.prob_a AS orig_prob_a,
               cp.diagnostics_json AS orig_diag
        FROM canonical_predictions cp
        WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
          AND cp.model_version = 'exp-039'
          AND cp.prediction_status = 'active'
          AND cp.canonical_match_id IN (
              SELECT cm.id FROM canonical_matches cm WHERE cm.status = 'finished'
          )
    ), corrected AS (
        SELECT cp.canonical_match_id, cp.prob_a AS corr_prob_a,
               cp.diagnostics_json AS corr_diag
        FROM canonical_predictions cp
        WHERE cp.model_name = 'Sym-Cal LR-ElasticNet-W20-Binomial'
          AND cp.model_version = 'exp-039-golgg-corrected'
          AND cp.prediction_status = 'active'
    )
    SELECT o.canonical_match_id,
           o.orig_prob_a, c.corr_prob_a,
           o.orig_diag, c.corr_diag
    FROM original o
    JOIN corrected c ON c.canonical_match_id = o.canonical_match_id
    ORDER BY ABS(o.orig_prob_a - c.corr_prob_a) DESC
    """
    df = query_df(sql)
    if df.empty:
        return []
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Re-prediction of Finished Matches Using GOL.GG Team Names")
    print("  Model: Sym-Cal LR-ElasticNet-W20-Binomial")
    print("  Version: exp-039-golgg-corrected")
    print("=" * 70)
    print()

    # Step 1: Run predictions
    print("[Step 1] Running predictions with GOL.GG-corrected team alignment...")
    print()
    results = predict_finished_with_golgg()

    predicted = [r for r in results if r["status"] == "predicted"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "db_error"]

    print()
    print(f"[Summary] Total: {len(results)}, Predicted: {len(predicted)}, "
          f"Skipped: {len(skipped)}, Errors: {len(errors)}")

    # Step 2: Metrics
    print()
    print("[Step 2] Computing metrics...")
    metrics = compute_metrics(predicted)
    print(f"  N={metrics['n']}, Accuracy={metrics.get('accuracy', 'N/A'):.4f}, "
          f"LogLoss={metrics.get('logloss', 'N/A'):.4f}")
    if "correct" in metrics:
        print(f"  Correct: {metrics['correct']}, Incorrect: {metrics['incorrect']}")

    # Step 3: Compare with original exp-039 predictions
    print()
    print("[Step 3] Comparing with original exp-039 predictions...")
    comparisons = compare_with_previous()
    if comparisons:
        print(f"  Found {len(comparisons)} matches with both predictions")
        # Show biggest differences
        print()
        print("  Top 10 biggest probability shifts (orig vs corrected):")
        print(f"  {'Match ID':>8}  {'Orig Prob A':>12}  {'Corr Prob A':>12}  {'Delta':>10}")
        print("  " + "-" * 48)
        for comp in comparisons[:10]:
            delta = abs(comp["orig_prob_a"] - comp["corr_prob_a"])
            print(f"  {comp['canonical_match_id']:>8}  {comp['orig_prob_a']:>12.4f}  "
                  f"{comp['corr_prob_a']:>12.4f}  {delta:>10.4f}")
    else:
        print("  No comparisons available (original predictions may not exist for finished matches)")
        print("  (The original pipeline only predicts 'upcoming' matches)")

    # Step 4: Detailed results
    print()
    print("[Step 4] Detailed results:")
    print()
    print(f"  {'ID':>6}  {'Team A (GOL.GG)':<30} {'Team B (GOL.GG)':<30} {'P(A)':>8} {'Winner':<20} {'✓/✗':>4}")
    print("  " + "-" * 104)
    for r in predicted:
        diag = r.get("diagnostics", {})
        team_a = diag.get("golgg_team1", "?")
        team_b = diag.get("golgg_team2", "?")
        winner = diag.get("golgg_winner", "?")
        correct_mark = "✓" if r.get("correct") else "✗" if r.get("correct") is False else "?"
        print(f"  {r['canonical_match_id']:>6}  {team_a:<30} {team_b:<30} "
              f"{r['prob_a']:>8.4f} {winner:<20} {correct_mark:>4}")

    print()
    print("=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
