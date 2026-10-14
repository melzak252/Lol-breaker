"""Validate temperature choice on 2020-2023 and test it on 2024+.

This experiment separates parameter selection from the final evaluation period.
The goal is to justify the temperature used in the case-study candidate without
choosing it directly on the 2024+ result table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HYBRID_DIR = PROJECT_ROOT / "docs" / "assets" / "hybrid_point7"
INPUT_PATH = HYBRID_DIR / "hybrid_model_input_predictions.csv"

OUTPUT_RESULTS = HYBRID_DIR / "hybrid_temperature_history_validation.csv"
OUTPUT_SELECTION = HYBRID_DIR / "hybrid_temperature_history_selection.csv"
OUTPUT_FIGURE = HYBRID_DIR / "hybrid_temperature_history_validation.png"

VALIDATION_START = "2020-01-01"
VALIDATION_END = "2023-12-31"
TEST_START = "2024-01-01"
# Static alpha family used in the case-study narrative.
ALPHAS = [0.30, 0.48, 0.62]
TEMPERATURES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00, 1.10, 1.20]
TAX_RATE = 0.12
EV_THRESHOLD = 0.05
FIXED_STAKE = 10.0
INITIAL_BANKROLL = 100.0


@dataclass(frozen=True)
class Period:
    """Definition of a validation or test period.

    Attributes:
        name: Human-readable period name.
        start: Inclusive start timestamp.
        end: Optional inclusive end timestamp.
    """

    name: str
    start: str
    end: str | None = None


def logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits with clipping.

    Args:
        probability: Probability values.

    Returns:
        Logit values.
    """

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities.

    Args:
        value: Logit values.

    Returns:
        Probability values.
    """

    return 1 / (1 + np.exp(-value))


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to binary probabilities.

    Args:
        probability: Raw probability values.
        temperature: Temperature value.

    Returns:
        Temperature-scaled probabilities.
    """

    return sigmoid(logit(probability) / temperature)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Compute equal-width expected calibration error.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probabilities for class 1.
        bins: Number of probability bins.

    Returns:
        ECE value.
    """

    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def choose_bet(row: pd.Series, probability: float) -> tuple[str | None, float, float]:
    """Choose a fixed-stake bet using best opening odds and EV threshold.

    Args:
        row: Match row.
        probability: Team 1 win probability.

    Returns:
        Tuple with selected side, selected odds and selected EV.
    """

    odds_t1 = row["best_open_t1"]
    odds_t2 = row["best_open_t2"]
    if pd.isna(odds_t1) or pd.isna(odds_t2):
        return None, 0.0, 0.0

    ev_t1 = probability * odds_t1 * (1 - TAX_RATE) - 1
    ev_t2 = (1 - probability) * odds_t2 * (1 - TAX_RATE) - 1
    if ev_t1 > ev_t2 and ev_t1 > EV_THRESHOLD:
        return "t1", float(odds_t1), float(ev_t1)
    if ev_t2 > EV_THRESHOLD:
        return "t2", float(odds_t2), float(ev_t2)
    return None, 0.0, 0.0


def simulate_unit_stake(data: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    """Simulate idealized fixed-stake betting without slippage.

    Args:
        data: Chronological match data.
        probability: Team 1 probabilities aligned to ``data``.

    Returns:
        Summary metrics.
    """

    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_drawdown = 0.0
    total_profit = 0.0
    total_staked = 0.0
    bets = 0
    wins = 0

    for idx, (_, row) in enumerate(data.iterrows()):
        side, odds, _ = choose_bet(row, float(probability[idx]))
        if side is None or bankroll < FIXED_STAKE:
            continue

        is_win = bool(row["y_true"] == 1) if side == "t1" else bool(row["y_true"] == 0)
        if is_win:
            profit = FIXED_STAKE * (odds * (1 - TAX_RATE) - 1)
            wins += 1
        else:
            profit = -FIXED_STAKE
        bankroll += profit
        total_profit += profit
        total_staked += FIXED_STAKE
        bets += 1
        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak * 100)

    return {
        "final_bankroll": bankroll,
        "roi_pct": (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL * 100,
        "unit_yield_pct": total_profit / total_staked * 100 if total_staked else 0.0,
        "max_drawdown_pct": max_drawdown,
        "bets": bets,
        "win_rate_pct": wins / bets * 100 if bets else 0.0,
    }


def period_slice(data: pd.DataFrame, period: Period) -> pd.DataFrame:
    """Return rows inside a named period.

    Args:
        data: Input match data.
        period: Period definition.

    Returns:
        Filtered data frame.
    """

    mask = data["date"] >= pd.Timestamp(period.start)
    if period.end is not None:
        mask &= data["date"] <= pd.Timestamp(period.end)
    return data[mask].copy().reset_index(drop=True)


def evaluate_period(data: pd.DataFrame, period: Period) -> list[dict[str, float | int | str]]:
    """Evaluate all alpha-temperature pairs in one period.

    Args:
        data: Full input data.
        period: Period definition.

    Returns:
        Result rows.
    """

    subset = period_slice(data, period)
    y_true = subset["y_true"].to_numpy(dtype=int)
    model = subset["prob_model"].to_numpy(dtype=float)
    market = subset["prob_market_open"].to_numpy(dtype=float)
    rows: list[dict[str, float | int | str]] = []

    for alpha in ALPHAS:
        for temperature in TEMPERATURES:
            model_t = apply_temperature(model, temperature)
            probability = alpha * model_t + (1 - alpha) * market
            betting = simulate_unit_stake(subset, probability)
            rows.append(
                {
                    "period": period.name,
                    "alpha": alpha,
                    "temperature": temperature,
                    "n": len(subset),
                    "auc": roc_auc_score(y_true, probability),
                    "logloss": log_loss(y_true, probability, labels=[0, 1]),
                    "brier": float(np.mean((probability - y_true) ** 2)),
                    "ece": expected_calibration_error(y_true, probability),
                    **betting,
                }
            )
    return rows


def select_temperatures(results: pd.DataFrame) -> pd.DataFrame:
    """Select best validation temperatures and attach test-period metrics.

    Args:
        results: All result rows.

    Returns:
        Selection summary.
    """

    validation = results[results["period"] == "validation_2020_2023"]
    test = results[results["period"] == "test_2024_plus"]
    rows = []
    for alpha in ALPHAS:
        alpha_validation = validation[validation["alpha"] == alpha]
        for objective, ascending in [("logloss", True), ("ece", True), ("unit_yield_pct", False)]:
            selected = alpha_validation.sort_values(objective, ascending=ascending).iloc[0]
            matched = test[
                (test["alpha"] == alpha)
                & (test["temperature"] == selected["temperature"])
            ].iloc[0]
            rows.append(
                {
                    "alpha": alpha,
                    "selection_objective": objective,
                    "selected_temperature": selected["temperature"],
                    "validation_logloss": selected["logloss"],
                    "validation_ece": selected["ece"],
                    "validation_unit_yield_pct": selected["unit_yield_pct"],
                    "validation_bets": selected["bets"],
                    "test_logloss": matched["logloss"],
                    "test_ece": matched["ece"],
                    "test_unit_yield_pct": matched["unit_yield_pct"],
                    "test_max_drawdown_pct": matched["max_drawdown_pct"],
                    "test_bets": matched["bets"],
                    "test_final_bankroll": matched["final_bankroll"],
                }
            )
    return pd.DataFrame(rows)


def save_plot(results: pd.DataFrame) -> None:
    """Save validation/test temperature curves.

    Args:
        results: All result rows.
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("logloss", "LogLoss ↓"),
        ("ece", "ECE ↓"),
        ("unit_yield_pct", "Unit Yield % ↑"),
        ("max_drawdown_pct", "MaxDD % ↓"),
    ]
    colors = {"validation_2020_2023": "#2563EB", "test_2024_plus": "#DC2626"}
    linestyles = {0.30: "-", 0.48: "--", 0.62: ":"}

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for period_name, period_group in results.groupby("period"):
            for alpha, group in period_group.groupby("alpha"):
                label = f"{period_name.replace('_', ' ')} | α={alpha:.2f}"
                ax.plot(
                    group["temperature"],
                    group[metric],
                    label=label,
                    color=colors[period_name],
                    linestyle=linestyles[alpha],
                    linewidth=2.0,
                )
        ax.axvline(0.60, color="#111827", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Temperature T")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, fontsize=8)
    fig.suptitle("Temperature selected on 2020-2023 vs evaluated on 2024+", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(OUTPUT_FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the temperature validation experiment."""

    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    periods = [
        Period("validation_2020_2023", VALIDATION_START, VALIDATION_END),
        Period("test_2024_plus", TEST_START),
    ]
    rows = []
    for period in periods:
        rows.extend(evaluate_period(data, period))
    results = pd.DataFrame(rows)
    selection = select_temperatures(results)

    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_RESULTS, index=False)
    selection.to_csv(OUTPUT_SELECTION, index=False)
    save_plot(results)

    print(f"Saved: {OUTPUT_RESULTS}")
    print(f"Saved: {OUTPUT_SELECTION}")
    print(f"Saved: {OUTPUT_FIGURE}")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
