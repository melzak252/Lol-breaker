"""Compare dynamic-alpha update objectives for the case-study hybrid model.

The script tests whether dynamic alpha should be updated by predictive quality
(LogLoss) or by betting-oriented signals (unit profit / CLV). It uses the
already generated hybrid input artifact and writes comparison tables/figures
for Chapter 6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "docs" / "assets" / "hybrid_point7" / "hybrid_model_input_predictions.csv"
OUTPUT_DIR = ROOT / "docs" / "assets" / "hybrid_point7"

TAX_FACTOR = 0.88
EV_THRESHOLD = 0.05
ALPHA_MIN = 0.30
ALPHA_MAX = 0.70
TEMPERATURE = 0.60
START_BANKROLL = 100.0
FIXED_STAKE = 5.0
MIN_STAKE = 5.0


@dataclass(frozen=True)
class DynamicObjective:
    """Configuration of an online dynamic-alpha update objective."""

    name: str
    score_name: str
    k_values: tuple[float, ...]


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Compute a numerically stable logistic sigmoid."""

    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def apply_temperature(probability: pd.Series | np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""

    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p))
    return sigmoid(logits / temperature)


def no_vig_probability(odds_1: pd.Series, odds_2: pd.Series) -> pd.Series:
    """Convert two decimal odds columns into no-vig probability for side 1."""

    inv_1 = 1.0 / odds_1.astype(float)
    inv_2 = 1.0 / odds_2.astype(float)
    return inv_1 / (inv_1 + inv_2)


def expected_value(probability: float, odds: float) -> float:
    """Return expected value after Polish 12% turnover tax."""

    return probability * odds * TAX_FACTOR - 1.0


def choose_unit_bet(probability: float, row: pd.Series) -> tuple[int, float, float] | None:
    """Choose side by EV and return side, odds, and EV for a unit-stake bet.

    Returns:
        ``None`` if neither side clears the EV threshold. Otherwise tuple
        ``(side, selected_odds, selected_ev)`` where side is 1 or 2.
    """

    odds_1 = float(row["best_open_t1"])
    odds_2 = float(row["best_open_t2"])
    ev_1 = expected_value(probability, odds_1)
    ev_2 = expected_value(1.0 - probability, odds_2)
    if ev_1 <= EV_THRESHOLD and ev_2 <= EV_THRESHOLD:
        return None
    if ev_1 >= ev_2:
        return 1, odds_1, ev_1
    return 2, odds_2, ev_2


def unit_profit(probability: float, row: pd.Series) -> float:
    """Return unit profit for a probability expert on one match."""

    bet = choose_unit_bet(probability, row)
    if bet is None:
        return 0.0
    side, odds, _ = bet
    won = (side == 1 and int(row["y_true"]) == 1) or (side == 2 and int(row["y_true"]) == 0)
    return odds * TAX_FACTOR - 1.0 if won else -1.0


def unit_clv(probability: float, row: pd.Series) -> float:
    """Return CLV for a probability expert on one match, or 0 if no bet."""

    bet = choose_unit_bet(probability, row)
    if bet is None:
        return 0.0
    side, entry_odds, _ = bet
    close_odds = float(row["best_close_t1"] if side == 1 else row["best_close_t2"])
    if not np.isfinite(close_odds) or close_odds <= 1.0:
        return 0.0
    return entry_odds / close_odds - 1.0


def update_score(row: pd.Series, objective: str) -> float:
    """Compute the model-vs-market update score for one match."""

    p_model = float(row["prob_model_t"])
    p_market = float(row["prob_market_open"])
    y = int(row["y_true"])
    if objective == "logloss":
        model_loss = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
        market_loss = -(y * np.log(p_market) + (1 - y) * np.log(1 - p_market))
        return float(np.clip(market_loss - model_loss, -1.0, 1.0))
    if objective == "unit_profit":
        score = unit_profit(p_model, row) - unit_profit(p_market, row)
        return float(np.clip(score, -2.0, 2.0))
    if objective == "clv":
        score = unit_clv(p_model, row) - unit_clv(p_market, row)
        return float(np.clip(score, -0.50, 0.50))
    raise ValueError(f"Unknown objective: {objective}")


def run_dynamic_alpha(data: pd.DataFrame, objective: DynamicObjective, k_value: float) -> pd.DataFrame:
    """Run bounded sigmoid dynamic alpha for one objective and K."""

    theta = 0.0
    records: list[dict[str, float | int | str]] = []
    for _, row in data.iterrows():
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * float(sigmoid(theta))
        probability = alpha * float(row["prob_model_t"]) + (1.0 - alpha) * float(row["prob_market_open"])
        records.append(
            {
                "date": row["date"],
                "golgg_match_id": int(row["golgg_match_id"]),
                "y_true": int(row["y_true"]),
                "alpha": alpha,
                "probability": probability,
                "objective": objective.name,
                "k_value": k_value,
            }
        )
        theta += k_value * update_score(row, objective.score_name)
    return pd.DataFrame(records)


def simulate_fixed_stake(predictions: pd.DataFrame, data: pd.DataFrame) -> dict[str, float]:
    """Simulate fixed-stake betting for a prediction series."""

    sim = predictions.merge(
        data[
            [
                "golgg_match_id",
                "best_open_t1",
                "best_open_t2",
                "best_close_t1",
                "best_close_t2",
            ]
        ],
        on="golgg_match_id",
        how="left",
    )
    bankroll = START_BANKROLL
    peak = START_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    profit = 0.0
    bets = 0
    wins = 0
    clv_values: list[float] = []
    for _, row in sim.iterrows():
        if bankroll < MIN_STAKE:
            break
        bet = choose_unit_bet(float(row["probability"]), row)
        if bet is None:
            peak = max(peak, bankroll)
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak * 100.0)
            continue
        side, odds, _ = bet
        stake = min(FIXED_STAKE, bankroll)
        won = (side == 1 and int(row["y_true"]) == 1) or (side == 2 and int(row["y_true"]) == 0)
        result = stake * (odds * TAX_FACTOR - 1.0) if won else -stake
        bankroll += result
        profit += result
        total_staked += stake
        bets += 1
        wins += int(won)
        close_odds = float(row["best_close_t1"] if side == 1 else row["best_close_t2"])
        if np.isfinite(close_odds) and close_odds > 1:
            clv_values.append(odds / close_odds - 1.0)
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak * 100.0)
    return {
        "final_bankroll": bankroll,
        "profit": profit,
        "total_staked": total_staked,
        "yield_pct": profit / total_staked * 100.0 if total_staked else 0.0,
        "max_drawdown_pct": max_drawdown,
        "bets": bets,
        "win_rate_pct": wins / bets * 100.0 if bets else 0.0,
        "avg_clv_pct": float(np.mean(clv_values) * 100.0) if clv_values else 0.0,
    }


def probability_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Calculate probabilistic metrics for a prediction series."""

    y = predictions["y_true"].astype(int)
    p = predictions["probability"].clip(1e-6, 1 - 1e-6)
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def evaluate_period(predictions: pd.DataFrame, data: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    """Evaluate predictions on a date interval."""

    mask = (predictions["date"] >= pd.Timestamp(start)) & (predictions["date"] <= pd.Timestamp(end))
    period_predictions = predictions.loc[mask].copy()
    metrics = probability_metrics(period_predictions)
    metrics.update(simulate_fixed_stake(period_predictions, data))
    metrics["matches"] = float(len(period_predictions))
    return metrics


def build_static_predictions(data: pd.DataFrame, alpha: float, label: str) -> pd.DataFrame:
    """Build static-alpha prediction frame."""

    probability = alpha * data["prob_model_t"] + (1.0 - alpha) * data["prob_market_open"]
    return pd.DataFrame(
        {
            "date": data["date"],
            "golgg_match_id": data["golgg_match_id"],
            "y_true": data["y_true"],
            "alpha": alpha,
            "probability": probability,
            "objective": label,
            "k_value": np.nan,
        }
    )


def select_best_k(
    data: pd.DataFrame,
    objective: DynamicObjective,
    selector: Callable[[dict[str, float]], float],
) -> tuple[float, pd.DataFrame, dict[str, float]]:
    """Select K on 2021-2023 using a selector function."""

    best_k = objective.k_values[0]
    best_predictions = pd.DataFrame()
    best_metrics: dict[str, float] = {}
    best_score = -np.inf
    for k_value in objective.k_values:
        predictions = run_dynamic_alpha(data, objective, k_value)
        metrics = evaluate_period(predictions, data, "2021-01-01", "2023-12-31")
        score = selector(metrics)
        if score > best_score:
            best_score = score
            best_k = k_value
            best_predictions = predictions
            best_metrics = metrics
    return best_k, best_predictions, best_metrics


def main() -> None:
    """Run comparison and write artifacts."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    data = data.dropna(
        subset=[
            "prob_model",
            "prob_market_open",
            "best_open_t1",
            "best_open_t2",
            "best_close_t1",
            "best_close_t2",
            "y_true",
        ]
    ).copy()
    data["prob_model_t"] = apply_temperature(data["prob_model"], TEMPERATURE)

    objectives = [
        DynamicObjective("Dynamic alpha — LogLoss score", "logloss", (0.0025, 0.005, 0.01, 0.02, 0.05)),
        DynamicObjective("Dynamic alpha — unit profit score", "unit_profit", (0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)),
        DynamicObjective("Dynamic alpha — CLV score", "clv", (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)),
    ]

    rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    selector = lambda metrics: metrics["yield_pct"] - 0.10 * metrics["max_drawdown_pct"]
    for objective in objectives:
        best_k, predictions, validation_metrics = select_best_k(data, objective, selector)
        prediction_frames.append(predictions)
        for period_name, start, end in [
            ("validation_2021_2023", "2021-01-01", "2023-12-31"),
            ("test_2024", "2024-01-01", "2024-12-31"),
            ("forward_2025", "2025-01-01", "2025-12-31"),
        ]:
            metrics = evaluate_period(predictions, data, start, end)
            rows.append(
                {
                    "approach": objective.name,
                    "period": period_name,
                    "selected_k": best_k,
                    **metrics,
                }
            )

    static_candidates = [
        ("Static alpha 0.48", 0.48),
        ("Static alpha 0.62", 0.62),
        ("Static alpha 0.30", 0.30),
    ]
    for label, alpha in static_candidates:
        predictions = build_static_predictions(data, alpha, label)
        prediction_frames.append(predictions)
        for period_name, start, end in [
            ("validation_2021_2023", "2021-01-01", "2023-12-31"),
            ("test_2024", "2024-01-01", "2024-12-31"),
            ("forward_2025", "2025-01-01", "2025-12-31"),
        ]:
            metrics = evaluate_period(predictions, data, start, end)
            rows.append({"approach": label, "period": period_name, "selected_k": np.nan, **metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "dynamic_alpha_objective_comparison.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "dynamic_alpha_objective_predictions.csv", index=False
    )

    plot_data = summary[summary["period"].isin(["test_2024", "forward_2025"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=170)
    for ax, metric, title in [
        (axes[0], "yield_pct", "Yield: dynamic alpha objectives"),
        (axes[1], "max_drawdown_pct", "MaxDD: dynamic alpha objectives"),
    ]:
        pivot = plot_data.pivot(index="approach", columns="period", values=metric)
        pivot = pivot.sort_values("test_2024", ascending=False)
        pivot.plot(kind="barh", ax=ax, width=0.75, color=["#2563EB", "#F97316"])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("%")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(title="Period", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dynamic_alpha_objective_comparison.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
