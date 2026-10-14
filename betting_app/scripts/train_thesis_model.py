"""Train and serialize the final thesis model: Sym-Cal LR-ElasticNet-W20-Binomial.

Reproduces the EXP-039 pipeline exactly as in the thesis:
  - Loads rating predictions + odds data (from golgg_y_predicts.csv, odds.csv)
  - Computes W20 rolling team features (from golgg_matches.json)
  - Adds binomial series-adjusted ranking features
  - Walk-forward evaluation with order symmetry (swap_orientation + symmetrize)
  - Platt calibration on logit-transformed OOF predictions (C=1.0, lbfgs)
  - Final Pipeline trained on ALL data
  - Saves Pipeline + Platt calibrator as joblib artefacts

Usage:
    python -m betting_app.scripts.train_thesis_model
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import deque
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.golgg_schema import team1_id, team2_id, games as golgg_games
from src.models.team_order import swap_orientation, symmetrize_binary_probabilities

MODEL_NAME = "Sym-Cal LR-ElasticNet-W20-Binomial"
MODEL_VERSION = "exp-039"
ARTEFACT_DIR = PROJECT_ROOT / "docs" / "assets" / "final_symmetric_calibrated_market_comparison"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_pipeline.joblib"
CALIBRATOR_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_calibrator.joblib"
METADATA_PATH = ARTEFACT_DIR / "sym_cal_lr_elasticnet_w20_binomial_metadata.json"

DATA_DIR = PROJECT_ROOT / "data"
TARGET = "y_true"
CONTEXT_WINDOW = 20
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
EPSILON = 0.001

# ---------------------------------------------------------------------------
# Feature definitions (from 06i_best_metamodel_config_search.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Rolling feature helpers (from 06i)
# ---------------------------------------------------------------------------
_DEFAULT_TEAM_STATS = {
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


def _safe_stat(player: dict, key: str) -> float:
    return float(player.get("stats", {}).get(key, 0.0) or 0.0)


def _safe_team_stat(game: dict, stats_key: str, key: str) -> float:
    return float((game.get(stats_key, {}) or {}).get(key, 0.0) or 0.0)


def _average_history(history: deque | None) -> dict[str, float]:
    if not history:
        return dict(_DEFAULT_TEAM_STATS)
    rows = list(history)
    return {
        "win_rate": float(np.mean([r["win"] for r in rows])),
        "kills": float(np.mean([r["kills"] for r in rows])),
        "deaths": float(np.mean([r["deaths"] for r in rows])),
        "gd15": float(np.mean([r["gd15"] for r in rows])),
        "dpm": float(np.mean([r["dpm"] for r in rows])),
        "vspm": float(np.mean([r["vspm"] for r in rows])),
        "towers": float(np.mean([r["towers"] for r in rows])),
        "nashors": float(np.mean([r["nashors"] for r in rows])),
        "gold": float(np.mean([r["gold"] for r in rows])),
        "duration": float(np.mean([r["duration"] for r in rows])),
    }


def _update_team_history(
    team_history: dict[str, deque],
    team_id: str,
    game: dict,
    window_size: int,
) -> None:
    is_team_1 = str(game.get("t1_id")) == str(team_id)
    win = bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))
    players_key = "t1_players" if is_team_1 else "t2_players"
    stats_key = "t1_stats" if is_team_1 else "t2_stats"
    players = game.get(players_key, {}) or {}
    game_stats = {
        "win": float(win),
        "kills": sum(_safe_stat(p, "kills") for p in players.values()),
        "deaths": sum(_safe_stat(p, "deaths") for p in players.values()),
        "dpm": sum(_safe_stat(p, "dpm") for p in players.values()),
        "vspm": sum(_safe_stat(p, "vspm") for p in players.values()),
        "gd15": sum(_safe_stat(p, "gd@15") for p in players.values()),
        "towers": _safe_team_stat(game, stats_key, "towers"),
        "nashors": _safe_team_stat(game, stats_key, "nashors"),
        "gold": _safe_team_stat(game, stats_key, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    if team_id not in team_history:
        team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)


def generate_rolling_features(window_size: int) -> pd.DataFrame:
    """Build leakage-safe W20 rolling context features from golgg_matches.json."""
    import json

    matches_path = DATA_DIR / "golgg_matches.json"
    with open(matches_path, "r", encoding="utf-8") as f:
        matches = json.load(f)
    matches.sort(key=lambda m: m["date"])

    team_history: dict[str, deque] = {}
    rows: list[dict] = []
    for match in tqdm(matches, desc=f"Rolling window {window_size}"):
        match_id = str(match["match_id"])
        t1 = team1_id(match)
        t2 = team2_id(match)
        t1_stats = _average_history(team_history.get(t1))
        t2_stats = _average_history(team_history.get(t2))
        row: dict = {"golgg_match_id": match_id, "context_window": window_size}
        for stat, val in t1_stats.items():
            row[f"t1_rolling_{stat}"] = val
        for stat, val in t2_stats.items():
            row[f"t2_rolling_{stat}"] = val
        rows.append(row)
        for game in golgg_games(match):
            _update_team_history(team_history, t1, game, window_size)
            _update_team_history(team_history, t2, game, window_size)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Binomial features
# ---------------------------------------------------------------------------
def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    """Convert map-win probabilities to best-of-series probabilities."""
    prob = np.clip(map_probability.astype(float), 0.001, 0.999)
    best_of_int = best_of.astype(int)
    result = prob.copy()
    for n_maps in (3, 5):
        needed = n_maps // 2 + 1
        series_prob = np.zeros_like(prob)
        for wins in range(needed, n_maps + 1):
            series_prob += (
                comb(n_maps, wins)
                * np.power(prob, wins)
                * np.power(1.0 - prob, n_maps - wins)
            )
        result = np.where(best_of_int == n_maps, series_prob, result)
    return np.clip(result, 0.001, 0.999)


def add_binomial_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add binomial series-adjusted ranking features."""
    enriched = data.copy()
    generated: list[str] = []
    best_of = enriched["BoN"].fillna(1).astype(int).to_numpy()
    for feature in RANK_PROB_FEATURES:
        col = f"{feature}_binom_series"
        enriched[col] = series_probability(enriched[feature].to_numpy(dtype=float), best_of)
        generated.append(col)
    return enriched, generated


# ---------------------------------------------------------------------------
# Platt calibration helpers (from src/models/calibration.py)
# ---------------------------------------------------------------------------
def logit(probability: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """Return clipped logit values as a two-dimensional array."""
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def build_logistic_regression() -> Pipeline:
    """Build the tuned ElasticNet logistic-regression pipeline (from 06ab)."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.03297234640536737,
                    penalty="elasticnet",
                    l1_ratio=0.9439657999531195,
                    solver="saga",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_base_data() -> pd.DataFrame:
    """Load rating predictions restricted to odds-mapped matches (2020+)."""
    predictions = pd.read_csv(DATA_DIR / "golgg_y_predicts.csv")
    odds = pd.read_csv(DATA_DIR / "odds.csv", usecols=["golgg_match_id"])
    predictions["golgg_match_id"] = predictions["golgg_match_id"].astype(str)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    data = predictions.merge(odds.drop_duplicates(), on="golgg_match_id", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"] >= pd.Timestamp("2020-01-01")].copy()
    return data.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print(f"Training {MODEL_NAME} ({MODEL_VERSION})")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading base data...")
    base = load_base_data()
    print(f"       Base rows: {len(base)}")

    print("[2/6] Generating rolling features...")
    rolling = generate_rolling_features(CONTEXT_WINDOW)
    print(f"       Rolling rows: {len(rolling)}")

    data = base.merge(rolling, on="golgg_match_id", how="inner")
    data = data.sort_values("date").reset_index(drop=True)

    print("[3/6] Adding binomial features...")
    data, binomial_features = add_binomial_features(data)
    all_features = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + binomial_features
    print(f"       Total features: {len(all_features)}")
    print(f"       Binomial features: {binomial_features}")

    clean = data.dropna(subset=all_features + [TARGET]).copy()
    clean = clean[clean["date"] >= pd.Timestamp("2020-01-01")].copy()
    clean = clean.reset_index(drop=True)
    print(f"       Clean rows: {len(clean)}")

    # 4. Walk-forward with order symmetry to generate OOF predictions
    print("[4/6] Walk-forward with order symmetry (OOF predictions)...")
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    print(f"       Initial train: {len(train_df)}, Test pool: {len(test_pool)}")

    oof_probs: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []

    for start in tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc="Walk-forward"):
        chunk = test_pool.iloc[start: start + UPDATE_INTERVAL].copy()

        # Train on expanding window
        model = build_logistic_regression()
        model.fit(train_df[all_features], train_df[TARGET].astype(int))

        # Predict on chunk — original orientation
        original_prob = np.clip(model.predict_proba(chunk[all_features])[:, 1], EPSILON, 1.0 - EPSILON)

        # Swap orientation for all test rows and predict
        swapped_chunk = swap_orientation(
            chunk,
            all_features,
            RANK_PROB_FEATURES,
            np.ones(len(chunk), dtype=bool),  # swap ALL rows
        )
        swapped_side_prob = np.clip(
            model.predict_proba(swapped_chunk[all_features])[:, 1],
            EPSILON, 1.0 - EPSILON,
        )

        # Order symmetry: average original and converted-back swapped
        p_sym = symmetrize_binary_probabilities(original_prob, swapped_side_prob)

        oof_probs.append(p_sym)
        oof_true.append(chunk[TARGET].astype(int).to_numpy())

        # Expand training window
        train_df = pd.concat([train_df, chunk], ignore_index=True)

    oof_probs_all = np.concatenate(oof_probs)
    oof_true_all = np.concatenate(oof_true)
    print(f"       OOF predictions: {len(oof_probs_all)}")

    # 5. Fit Platt calibrator on OOF predictions (logit-transformed)
    print("[5/6] Fitting Platt calibrator on OOF predictions...")
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    platt.fit(logit(oof_probs_all), oof_true_all)

    # Evaluate OOF with Platt calibration
    oof_calibrated = np.clip(
        platt.predict_proba(logit(oof_probs_all))[:, 1],
        EPSILON, 1.0 - EPSILON,
    )
    oof_ll = float(log_loss(oof_true_all, oof_calibrated))
    oof_auc = float(roc_auc_score(oof_true_all, oof_calibrated))
    print(f"       OOF Walk-Forward metrics: LogLoss={oof_ll:.4f}, AUC={oof_auc:.4f}")

    # 6. Train final pipeline on ALL data
    print("[6/6] Training final pipeline on ALL data...")
    pipeline = build_logistic_regression()
    X = clean[all_features]
    y = clean[TARGET].astype(int)
    pipeline.fit(X, y)
    print(f"       Pipeline trained on {len(clean)} matches.")

    # Evaluate final model on all data (for reference)
    full_original = np.clip(pipeline.predict_proba(X)[:, 1], EPSILON, 1.0 - EPSILON)
    full_swapped = swap_orientation(
        clean, all_features, RANK_PROB_FEATURES, np.ones(len(clean), dtype=bool),
    )
    full_swapped_prob = np.clip(
        pipeline.predict_proba(full_swapped[all_features])[:, 1],
        EPSILON, 1.0 - EPSILON,
    )
    full_sym = symmetrize_binary_probabilities(full_original, full_swapped_prob)
    full_calibrated = np.clip(
        platt.predict_proba(logit(full_sym))[:, 1],
        EPSILON, 1.0 - EPSILON,
    )

    n_all = len(clean)
    ll = float(log_loss(y, full_calibrated))
    auc_val = float(roc_auc_score(y, full_calibrated))
    print(f"\n       Final metrics (n={n_all}): LogLoss={ll:.4f}, AUC={auc_val:.4f}")

    # 7. Save artefacts
    print("\nSaving artefacts...")
    import joblib
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(platt, CALIBRATOR_PATH)
    print(f"       Pipeline: {PIPELINE_PATH}")
    print(f"       Calibrator: {CALIBRATOR_PATH}")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "training_date": pd.Timestamp.now().isoformat(),
        "n_matches": n_all,
        "n_features": len(all_features),
        "features": all_features,
        "rank_prob_features": RANK_PROB_FEATURES,
        "metrics": {
            "log_loss": ll,
            "auc": auc_val,
            "oof_log_loss": oof_ll,
            "oof_auc": oof_auc,
        },
        "hyperparams": {
            "C": 0.03297234640536737,
            "penalty": "elasticnet",
            "l1_ratio": 0.9439657999531195,
            "solver": "saga",
            "max_iter": 5000,
        },
        "platt_hyperparams": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        "pipeline_path": str(PIPELINE_PATH),
        "calibrator_path": str(CALIBRATOR_PATH),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"       Metadata: {METADATA_PATH}")

    print("\nDone! Model is ready for inference.")
    print("-" * 60)


if __name__ == "__main__":
    main()
