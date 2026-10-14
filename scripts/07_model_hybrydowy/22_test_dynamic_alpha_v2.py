"""Test dynamic alpha v2 variants for the EnsembleLegends case study.

This script is intentionally experimental. It compares bounded sigmoid dynamic
alpha variants updated after every match using LogLoss advantage, optional
mean reversion, and optional disagreement weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "docs" / "assets" / "hybrid_point7" / "hybrid_model_input_predictions.csv"
OUTPUT_DIR = ROOT / "docs" / "assets" / "hybrid_point7"

TAX_FACTOR = 0.88
EV_THRESHOLD = 0.05
START_BANKROLL = 100.0
FIXED_STAKE = 5.0
MIN_STAKE = 5.0
ALPHA_MIN = 0.30
ALPHA_MAX = 0.70
TEMPERATURE = 0.60


@dataclass(frozen=True)
class Period:
    """Evaluation period definition."""

    name: str
    start: str
    end: str


@dataclass(frozen=True)
class DynamicConfig:
    """Configuration for a dynamic alpha update rule."""

    name: str
    k_factor: float
    rho: float
    lambda_disagreement: float


PERIODS = [
    Period("validation_2021_2023", "2021-01-01", "2023-12-31"),
    Period("test_2024", "2024-01-01", "2024-12-31"),
    Period("forward_2025", "2025-01-01", "2025-12-31"),
]


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Return the logistic sigmoid of the input values.

    Args:
        values: Scalar or NumPy array.

    Returns:
        Sigmoid-transformed value or array.
    """

    return 1.0 / (1.0 + np.exp(-values))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities.

    Args:
        probabilities: Probability vector.
        temperature: Temperature value. Values below 1 sharpen probabilities.

    Returns:
        Temperature-scaled probabilities.
    """

    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    return np.clip(sigmoid(logits / temperature), 1e-6, 1 - 1e-6)


def binary_log_loss_value(target: int, probability: float) -> float:
    """Calculate single-observation binary log loss.

    Args:
        target: Binary outcome.
        probability: Probability of outcome 1.

    Returns:
        LogLoss value.
    """

    p = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return -(target * np.log(p) + (1 - target) * np.log(1 - p))


def load_dataset() -> pd.DataFrame:
    """Load and prepare the hybrid model input dataset.

    Returns:
        Prepared DataFrame sorted chronologically.
    """

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    required = [
        "date",
        "t1_win",
        "prob_model",
        "prob_market_open",
        "best_open_t1",
        "best_open_t2",
        "best_close_t1",
        "best_close_t2",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    df["team1_win"] = df["t1_win"].astype(int)
    df["prob_model_t"] = apply_temperature(df["prob_model"].to_numpy(), TEMPERATURE)
    return df


def dynamic_probability(df: pd.DataFrame, config: DynamicConfig) -> pd.DataFrame:
    """Generate dynamic alpha probabilities for one configuration.

    The update uses all matches chronologically. Alpha used for a match is based
    only on previous matches; after observing the outcome, theta is updated.

    Args:
        df: Prepared input data.
        config: Dynamic alpha configuration.

    Returns:
        DataFrame with alpha and probability columns.
    """

    theta = 0.0
    alpha_values: list[float] = []
    probabilities: list[float] = []
    score_values: list[float] = []

    for row in df.itertuples(index=False):
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * float(sigmoid(theta))
        p_model = float(row.prob_model_t)
        p_market = float(row.prob_market_open)
        p_dynamic = alpha * p_model + (1 - alpha) * p_market

        target = int(row.team1_win)
        loss_model = binary_log_loss_value(target, p_model)
        loss_market = binary_log_loss_value(target, p_market)
        disagreement = abs(p_model - p_market)
        weight = 1.0 + config.lambda_disagreement * disagreement
        score = np.clip(loss_market - loss_model, -1.0, 1.0) * weight

        alpha_values.append(alpha)
        probabilities.append(float(np.clip(p_dynamic, 1e-6, 1 - 1e-6)))
        score_values.append(float(score))

        theta = config.rho * theta + config.k_factor * float(score)

    output = df[["date", "team1_win", "prob_model_t", "prob_market_open"]].copy()
    output["config"] = config.name
    output["alpha"] = alpha_values
    output["probability"] = probabilities
    output["update_score"] = score_values
    return output


def static_probability(df: pd.DataFrame, alpha: float, name: str) -> pd.DataFrame:
    """Generate static alpha probabilities.

    Args:
        df: Prepared input data.
        alpha: Static alpha value.
        name: Variant name.

    Returns:
        DataFrame with static alpha predictions.
    """

    probability = alpha * df["prob_model_t"] + (1 - alpha) * df["prob_market_open"]
    output = df[["date", "team1_win", "prob_model_t", "prob_market_open"]].copy()
    output["config"] = name
    output["alpha"] = alpha
    output["probability"] = probability.clip(1e-6, 1 - 1e-6)
    output["update_score"] = np.nan
    return output


def choose_bet(row: pd.Series, probability: float) -> tuple[str | None, float, float]:
    """Choose the best EV side for a fixed-stake bet.

    Args:
        row: Match row with best opening odds.
        probability: Probability of team 1 win.

    Returns:
        Tuple of selected side, execution odds, and EV. Side is None if no bet.
    """

    ev_team1 = probability * float(row["best_open_t1"]) * TAX_FACTOR - 1.0
    ev_team2 = (1.0 - probability) * float(row["best_open_t2"]) * TAX_FACTOR - 1.0
    if ev_team1 <= EV_THRESHOLD and ev_team2 <= EV_THRESHOLD:
        return None, np.nan, max(ev_team1, ev_team2)
    if ev_team1 >= ev_team2:
        return "team1", float(row["best_open_t1"]), ev_team1
    return "team2", float(row["best_open_t2"]), ev_team2


def simulate_period(df: pd.DataFrame, predictions: pd.DataFrame, period: Period) -> dict[str, float | str]:
    """Evaluate one prediction variant in one period.

    Args:
        df: Original prepared input data.
        predictions: Prediction output for one configuration.
        period: Evaluation period.

    Returns:
        Summary metric dictionary.
    """

    merged = df.reset_index().merge(
        predictions.reset_index()[["index", "config", "alpha", "probability"]],
        on="index",
        how="inner",
    )
    mask = (merged["date"] >= pd.Timestamp(period.start)) & (merged["date"] <= pd.Timestamp(period.end))
    sample = merged.loc[mask].copy()
    if sample.empty:
        raise ValueError(f"Empty evaluation sample for {period.name}")

    bankroll = START_BANKROLL
    peak = START_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    profit = 0.0
    bets = 0
    wins = 0
    clv_values: list[float] = []

    for _, row in sample.iterrows():
        if bankroll < MIN_STAKE:
            continue
        side, odds, _ = choose_bet(row, float(row["probability"]))
        if side is None:
            continue
        stake = min(FIXED_STAKE, bankroll)
        is_win = (side == "team1" and int(row["team1_win"]) == 1) or (
            side == "team2" and int(row["team1_win"]) == 0
        )
        pnl = stake * (odds * TAX_FACTOR - 1.0) if is_win else -stake
        bankroll += pnl
        total_staked += stake
        profit += pnl
        bets += 1
        wins += int(is_win)
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak * 100.0 if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

        close_odds = float(row["best_close_t1"] if side == "team1" else row["best_close_t2"])
        if close_odds > 0:
            clv_values.append((odds - close_odds) / close_odds * 100.0)

    probability = sample["probability"].to_numpy()
    target = sample["team1_win"].to_numpy()
    alpha_series = sample["alpha"]
    yield_pct = profit / total_staked * 100.0 if total_staked > 0 else 0.0
    return {
        "config": str(sample["config"].iloc[0]),
        "period": period.name,
        "matches": float(len(sample)),
        "auc": roc_auc_score(target, probability),
        "logloss": log_loss(target, probability),
        "final_bankroll": bankroll,
        "yield_pct": yield_pct,
        "max_drawdown_pct": max_drawdown,
        "bets": float(bets),
        "win_rate_pct": wins / bets * 100.0 if bets else 0.0,
        "avg_clv_pct": float(np.mean(clv_values)) if clv_values else 0.0,
        "alpha_mean": float(alpha_series.mean()),
        "alpha_min": float(alpha_series.min()),
        "alpha_max": float(alpha_series.max()),
        "alpha_last": float(alpha_series.iloc[-1]),
    }


def selector_score(summary: pd.DataFrame) -> pd.Series:
    """Calculate model-selection score for validation period.

    Args:
        summary: Summary DataFrame.

    Returns:
        Selector score where higher is better.
    """

    return summary["yield_pct"] - 0.10 * summary["max_drawdown_pct"]


def build_dynamic_configs() -> list[DynamicConfig]:
    """Build dynamic alpha v2 configuration grid.

    Returns:
        List of configurations to evaluate.
    """

    configs: list[DynamicConfig] = []
    for rho in [1.0, 0.9995, 0.999, 0.995, 0.99]:
        for lamb in [0.0, 1.0, 2.0, 5.0, 10.0]:
            for k_factor in [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]:
                name = f"Dyn v2 rho={rho:g} lambda={lamb:g} K={k_factor:g}"
                configs.append(DynamicConfig(name, k_factor, rho, lamb))
    return configs


def evaluate_predictions(df: pd.DataFrame, predictions_list: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Evaluate prediction variants over all periods.

    Args:
        df: Prepared input data.
        predictions_list: Iterable of prediction DataFrames.

    Returns:
        Combined summary DataFrame.
    """

    rows: list[dict[str, float | str]] = []
    for predictions in predictions_list:
        for period in PERIODS:
            rows.append(simulate_period(df, predictions, period))
    return pd.DataFrame(rows)


def main() -> None:
    """Run dynamic alpha v2 experiment and save outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    dynamic_predictions = [dynamic_probability(df, config) for config in build_dynamic_configs()]
    dynamic_summary = evaluate_predictions(df, dynamic_predictions)
    validation = dynamic_summary[dynamic_summary["period"] == "validation_2021_2023"].copy()
    validation["selector_score"] = selector_score(validation)
    best_name = str(validation.sort_values("selector_score", ascending=False).iloc[0]["config"])

    selected_dynamic_predictions = [pred for pred in dynamic_predictions if pred["config"].iloc[0] == best_name][0]
    benchmark_predictions = [
        static_probability(df, 0.30, "Static alpha 0.30"),
        static_probability(df, 0.48, "Static alpha 0.48"),
        static_probability(df, 0.62, "Static alpha 0.62"),
    ]
    selected_summary = evaluate_predictions(df, [selected_dynamic_predictions] + benchmark_predictions)

    dynamic_summary.to_csv(OUTPUT_DIR / "dynamic_alpha_v2_grid_summary.csv", index=False)
    selected_summary.to_csv(OUTPUT_DIR / "dynamic_alpha_v2_selected_comparison.csv", index=False)
    selected_dynamic_predictions.to_csv(OUTPUT_DIR / "dynamic_alpha_v2_selected_predictions.csv", index=False)

    print("Best dynamic v2 config:", best_name)
    print(selected_summary.to_string(index=False))


if __name__ == "__main__":
    main()
