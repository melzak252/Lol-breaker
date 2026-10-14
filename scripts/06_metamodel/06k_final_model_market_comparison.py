"""Compare final metamodel variants with bookmaker market probabilities.

This script is intentionally limited to the thesis scope: rating baseline,
player-only metamodel, metamodel with historical context, and bookmaker opening
and closing probabilities. It does not run hybrid blending, EV betting, staking,
or bankroll simulations.
"""

from __future__ import annotations

import sys
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
BEST_CONFIG_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "final_model_market_comparison"
TARGET = "y_true"


def load_best_config_module() -> ModuleType:
    """Load the 06i best-config script as a Python module.

    Returns:
        Imported module object exposing feature constants and helper functions.

    Raises:
        ImportError: If the module cannot be imported from the expected path.
    """

    spec = importlib.util.spec_from_file_location("metamodel_best_config", BEST_CONFIG_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {BEST_CONFIG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def normalize_team_name(name: object) -> str:
    """Normalize team names for conservative market-side alignment.

    Args:
        name: Raw team name.

    Returns:
        Lowercase alphanumeric team label.
    """

    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def no_vig_probability(team1_odds: pd.Series, team2_odds: pd.Series) -> pd.Series:
    """Convert two-way decimal odds into no-vig team-1 probability.

    Args:
        team1_odds: Decimal odds for side stored as team 1 in ``odds.csv``.
        team2_odds: Decimal odds for side stored as team 2 in ``odds.csv``.

    Returns:
        Margin-normalized probability for the market team-1 side.
    """

    p1 = 1.0 / pd.to_numeric(team1_odds, errors="coerce")
    p2 = 1.0 / pd.to_numeric(team2_odds, errors="coerce")
    return p1 / (p1 + p2)


def load_market_probabilities() -> pd.DataFrame:
    """Load bookmaker probabilities and side labels from ``odds.csv``.

    Returns:
        DataFrame with raw market team sides and no-vig open/close
        probabilities.
    """

    odds = pd.read_csv(PROJECT_ROOT / "data" / "odds.csv")
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    odds["market_open_raw"] = no_vig_probability(odds["avg_open_home"], odds["avg_open_away"])
    odds["market_close_raw"] = no_vig_probability(odds["avg_odds_home"], odds["avg_odds_away"])
    return odds[
        [
            "golgg_match_id",
            "golgg_team1",
            "golgg_team2",
            "market_open_raw",
            "market_close_raw",
        ]
    ]


def align_market_to_model_side(data: pd.DataFrame) -> pd.DataFrame:
    """Align market probabilities to the model's team-1 side.

    Args:
        data: Prediction frame merged with market rows.

    Returns:
        Frame with aligned ``market_open`` and ``market_close`` columns. Rows
        with ambiguous side alignment keep missing aligned probabilities.
    """

    output = data.copy()
    model_t1 = output["team1_name"].map(normalize_team_name)
    model_t2 = output["team2_name"].map(normalize_team_name)
    market_t1 = output["golgg_team1"].map(normalize_team_name)
    market_t2 = output["golgg_team2"].map(normalize_team_name)

    same = (model_t1 == market_t1) & (model_t2 == market_t2)
    swapped = (model_t1 == market_t2) & (model_t2 == market_t1)
    output["market_side_alignment"] = np.select([same, swapped], ["same", "swapped"], default="unknown")
    output["market_open"] = np.select(
        [same, swapped],
        [output["market_open_raw"], 1.0 - output["market_open_raw"]],
        default=np.nan,
    )
    output["market_close"] = np.select(
        [same, swapped],
        [output["market_close_raw"], 1.0 - output["market_close_raw"]],
        default=np.nan,
    )
    return output


def walk_forward_prediction_frame(
    data: pd.DataFrame,
    features: list[str],
    probability_column: str,
    update_interval: int = 1000,
    mask_rate: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate walk-forward predictions while preserving match identifiers.

    Args:
        data: Chronologically sorted modelling frame.
        features: Feature columns used by LightGBM.
        probability_column: Name of output probability column.
        update_interval: Number of future matches predicted per refit.
        mask_rate: Probability of masking training feature values.
        random_seed: Random seed for masking.

    Returns:
        DataFrame with ``golgg_match_id``, date, target and probability column.
    """

    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    rng = np.random.default_rng(random_seed)
    params = load_best_config_module().optuna_params()
    rows: list[pd.DataFrame] = []

    for start in tqdm(range(0, len(test_pool), update_interval), desc=probability_column):
        test_chunk = test_pool.iloc[start : start + update_interval].copy()
        x_train = train_df[features].copy()
        if mask_rate > 0:
            x_train = x_train.mask(rng.random(x_train.shape) < mask_rate)
        model = LGBMClassifier(**params)
        model.fit(x_train, train_df[TARGET].astype(int))
        chunk_output = test_chunk[
            ["golgg_match_id", "date", TARGET, "team1_name", "team2_name"]
        ].copy()
        chunk_output[probability_column] = model.predict_proba(test_chunk[features])[:, 1]
        rows.append(chunk_output)
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    return pd.concat(rows, ignore_index=True)


def evaluate_probability(data: pd.DataFrame, model: str, column: str) -> dict[str, object]:
    """Evaluate one probability column.

    Args:
        data: DataFrame containing target and probability column.
        model: Human-readable model name.
        column: Probability column name.

    Returns:
        Dictionary with standard thesis metrics.
    """

    subset = data[[TARGET, column]].dropna().copy()
    y_true = subset[TARGET].astype(int).to_numpy()
    y_prob = np.clip(subset[column].to_numpy(), 0.001, 0.999)
    return {
        "model": model,
        "probability_column": column,
        "sample_size": int(len(subset)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": calculate_ece(y_true, y_prob),
    }


def main() -> None:
    """Run final model-vs-market comparison for thesis scope."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    best_config = load_best_config_module()
    base = best_config.load_base_data()
    rolling = best_config.generate_rolling_features(best_config.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")

    player_only = walk_forward_prediction_frame(
        data=data,
        features=best_config.OPTUNA_BASE_FEATURES,
        probability_column="prob_player_only_metamodel",
        update_interval=1000,
        mask_rate=0.0,
    )
    full_context = walk_forward_prediction_frame(
        data=data,
        features=best_config.OPTUNA_BASE_FEATURES + best_config.ROLLING_FULL_FEATURES,
        probability_column="prob_full_context_metamodel",
        update_interval=1000,
        mask_rate=0.1,
    )

    comparison = player_only.merge(
        full_context[["golgg_match_id", "prob_full_context_metamodel"]],
        on="golgg_match_id",
        how="inner",
    )
    market = load_market_probabilities()
    comparison = comparison.merge(market, on="golgg_match_id", how="left")
    comparison = align_market_to_model_side(comparison)
    comparison["player_glicko2"] = data.set_index("golgg_match_id").loc[
        comparison["golgg_match_id"], "player_gl"
    ].to_numpy()

    common = comparison.dropna(
        subset=[
            "player_glicko2",
            "prob_player_only_metamodel",
            "prob_full_context_metamodel",
            "market_open",
            "market_close",
        ]
    ).copy()

    metrics = pd.DataFrame(
        [
            evaluate_probability(common, "Player Glicko-2", "player_glicko2"),
            evaluate_probability(common, "Metamodel player-only", "prob_player_only_metamodel"),
            evaluate_probability(common, "Metamodel + context W50", "prob_full_context_metamodel"),
            evaluate_probability(common, "Market Open", "market_open"),
            evaluate_probability(common, "Market Close", "market_close"),
        ]
    ).sort_values("logloss")

    comparison.to_csv(ASSETS_DIR / "final_model_market_predictions.csv", index=False)
    common.to_csv(ASSETS_DIR / "final_model_market_common_sample.csv", index=False)
    metrics.to_csv(ASSETS_DIR / "final_model_market_metrics.csv", index=False)

    print("\n=== FINAL MODEL VS MARKET COMPARISON ===")
    print("Common sample size:", len(common))
    print("Market side alignment:")
    print(comparison["market_side_alignment"].value_counts(dropna=False).to_string())
    print(metrics.to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
