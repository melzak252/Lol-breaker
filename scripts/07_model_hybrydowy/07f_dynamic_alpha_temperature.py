"""Test Elo-like dynamic alpha and temperature for the hybrid model.

The experiment reuses Chapter 7 operational predictions and updates two
parameters online after every resolved match:

* alpha: trust assigned to the statistical metamodel versus market average,
* temperature: sharpness of metamodel probabilities before blending.

The update is intentionally simple and auditable. It is not trained on the
future; each match is predicted with the state available before the result, and
only then the state is updated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.probability_metrics import calculate_ece
ASSET_DIR = PROJECT_ROOT / "docs" / "assets" / "hybrid_point7"
INPUT_PATH = ASSET_DIR / "hybrid_model_input_predictions.csv"

INITIAL_ALPHA = 0.50
INITIAL_TEMPERATURE = 1.00
ALPHA_GRID = [0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
TEMP_GRID = [0.0, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
ALPHA_BOUNDS = (0.0, 1.0)
TEMP_BOUNDS = (0.50, 2.50)


@dataclass(frozen=True)
class SimulationConfig:
    """Financial simulation configuration."""

    initial_bankroll: float = 100.0
    fixed_stake: float = 10.0
    min_stake: float = 2.0
    max_stake: float = 100.0
    tax_rate: float = 0.12
    slippage: float = 0.01
    ev_threshold: float = 0.05
    kelly_fraction: float = 0.25


def configure_style() -> None:
    """Configure consistent chart style."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def logit(probability: np.ndarray | float) -> np.ndarray | float:
    """Convert probability to logit with clipping."""
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    """Convert logit to probability."""
    return 1 / (1 + np.exp(-value))


def apply_temperature(probability: float, temperature: float) -> float:
    """Apply temperature scaling to one probability."""
    return float(sigmoid(logit(probability) / temperature))


def blend_probability(model_probability: float, market_probability: float, alpha: float) -> float:
    """Blend model and market probabilities linearly."""
    return float(alpha * model_probability + (1 - alpha) * market_probability)


def binary_loss(y_true: int, probability: float) -> float:
    """Return binary log-loss for one observation."""
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return -np.log(probability if y_true == 1 else 1 - probability)


def evaluate_probability(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Evaluate probability metrics."""
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "logloss": float(log_loss(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": calculate_ece(y_true, probability),
    }


def dynamic_alpha_temperature(
    df: pd.DataFrame,
    alpha_k: float,
    temperature_k: float,
    initial_alpha: float = INITIAL_ALPHA,
    initial_temperature: float = INITIAL_TEMPERATURE,
) -> pd.DataFrame:
    """Generate online predictions with Elo-like alpha and temperature updates.

    Args:
        df: Chronological dataframe with market/model probabilities and labels.
        alpha_k: Learning rate for alpha update.
        temperature_k: Learning rate for log-temperature update.
        initial_alpha: Initial trust in metamodel.
        initial_temperature: Initial temperature.

    Returns:
        Dataframe with prediction, alpha and temperature before each match.
    """
    alpha = initial_alpha
    log_temperature = float(np.log(initial_temperature))
    rows: list[dict[str, float | str | int]] = []

    for _, row in df.iterrows():
        y_true = int(row["y_true"])
        market_probability = float(row["prob_market_open"])
        raw_model_probability = float(row["prob_model"])
        temperature = float(np.exp(log_temperature))
        model_probability = apply_temperature(raw_model_probability, temperature)
        hybrid_probability = blend_probability(model_probability, market_probability, alpha)

        model_loss = binary_loss(y_true, model_probability)
        market_loss = binary_loss(y_true, market_probability)
        hybrid_loss = binary_loss(y_true, hybrid_probability)
        confidence = abs(model_probability - 0.5) * 2

        rows.append(
            {
                "date": row["date"],
                "golgg_match_id": int(row["golgg_match_id"]),
                "y_true": y_true,
                "alpha": alpha,
                "temperature": temperature,
                "probability": hybrid_probability,
                "model_probability_temperature": model_probability,
                "market_probability": market_probability,
                "model_loss": model_loss,
                "market_loss": market_loss,
                "hybrid_loss": hybrid_loss,
            }
        )

        alpha = float(np.clip(alpha + alpha_k * (market_loss - model_loss), *ALPHA_BOUNDS))
        temp_signal = confidence * (model_loss - np.log(2.0))
        log_temperature = float(np.log(np.clip(np.exp(log_temperature + temperature_k * temp_signal), *TEMP_BOUNDS)))

    return pd.DataFrame(rows)


def select_bet(row: pd.Series, probability: float, config: SimulationConfig) -> tuple[str | None, float, bool, float]:
    """Select a bet using EV threshold and best opening odds."""
    net_multiplier = 1 - config.tax_rate
    ev_t1 = row["best_open_t1"] * net_multiplier * probability - 1
    ev_t2 = row["best_open_t2"] * net_multiplier * (1 - probability) - 1
    if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
        return "t1", float(row["best_open_t1"]), int(row["y_true"] == 1) == 1, float(ev_t1)
    if ev_t2 > config.ev_threshold:
        return "t2", float(row["best_open_t2"]), int(row["y_true"] == 0) == 1, float(ev_t2)
    return None, 0.0, False, 0.0


def dynamic_alpha_temperature_bet_settlement(
    df: pd.DataFrame,
    alpha_k: float,
    temperature_k: float,
    config: SimulationConfig,
    initial_alpha: float = INITIAL_ALPHA,
    initial_temperature: float = INITIAL_TEMPERATURE,
) -> pd.DataFrame:
    """Generate online predictions updated only after placed bets settle.

    The update follows the user's Elo-like idea: alpha and temperature are not
    changed after every match, but only after the strategy actually places a bet
    and the bet outcome is known. The state used for a match is always the state
    available before that match, so the procedure remains chronological.

    Args:
        df: Chronological dataframe with market/model probabilities and labels.
        alpha_k: Learning rate for alpha update.
        temperature_k: Learning rate for log-temperature update.
        config: Financial simulation configuration used for bet selection.
        initial_alpha: Initial trust in metamodel.
        initial_temperature: Initial temperature.

    Returns:
        Dataframe with pre-match alpha, temperature, probability and bet state.
    """
    alpha = initial_alpha
    log_temperature = float(np.log(initial_temperature))
    rows: list[dict[str, float | str | int | bool | None]] = []

    for _, row in df.iterrows():
        y_true = int(row["y_true"])
        market_probability = float(row["prob_market_open"])
        raw_model_probability = float(row["prob_model"])
        temperature = float(np.exp(log_temperature))
        model_probability = apply_temperature(raw_model_probability, temperature)
        hybrid_probability = blend_probability(model_probability, market_probability, alpha)
        side, _, is_win, _ = select_bet(row, hybrid_probability, config)

        rows.append(
            {
                "date": row["date"],
                "golgg_match_id": int(row["golgg_match_id"]),
                "y_true": y_true,
                "alpha": alpha,
                "temperature": temperature,
                "probability": hybrid_probability,
                "model_probability_temperature": model_probability,
                "market_probability": market_probability,
                "bet_side": side,
                "bet_win": bool(is_win) if side is not None else None,
            }
        )

        if side is None:
            continue

        outcome = 1 if is_win else 0
        model_side_probability = model_probability if side == "t1" else 1 - model_probability
        market_side_probability = market_probability if side == "t1" else 1 - market_probability
        model_loss = binary_loss(outcome, model_side_probability)
        market_loss = binary_loss(outcome, market_side_probability)
        confidence = abs(model_side_probability - 0.5) * 2

        alpha = float(np.clip(alpha + alpha_k * (market_loss - model_loss), *ALPHA_BOUNDS))
        temp_signal = confidence * (model_loss - np.log(2.0))
        log_temperature = float(np.log(np.clip(np.exp(log_temperature + temperature_k * temp_signal), *TEMP_BOUNDS)))

    return pd.DataFrame(rows)


def calculate_kelly_stake(bankroll: float, odds: float, probability: float, config: SimulationConfig) -> float:
    """Calculate fractional Kelly stake with min/max clipping."""
    execution_odds = max(1.01, odds * (1 - config.slippage))
    b_value = execution_odds * (1 - config.tax_rate) - 1
    if b_value <= 0:
        return 0.0
    full_kelly = ((b_value * probability) - (1 - probability)) / b_value
    if full_kelly <= 0:
        return 0.0
    stake = bankroll * config.kelly_fraction * full_kelly
    return float(min(max(stake, config.min_stake), config.max_stake))


def simulate_finance(
    df: pd.DataFrame,
    probability: np.ndarray,
    staking: str,
    config: SimulationConfig,
) -> dict[str, float]:
    """Run financial simulation for one probability vector."""
    bankroll = config.initial_bankroll
    total_profit = 0.0
    total_staked = 0.0
    bets = 0
    wins = 0
    history = [bankroll]
    clv_values: list[float] = []

    for (_, row), prob_t1 in zip(df.iterrows(), probability):
        side, raw_odds, is_win, _ = select_bet(row, float(prob_t1), config)
        if side is None:
            history.append(bankroll)
            continue

        side_probability = float(prob_t1) if side == "t1" else float(1 - prob_t1)
        if staking == "fixed_percent":
            stake = min(max(bankroll * 0.02, config.min_stake), config.max_stake)
        elif staking == "kelly_025":
            stake = calculate_kelly_stake(bankroll, raw_odds, side_probability, config)
        else:
            stake = config.fixed_stake

        if stake <= 0 or bankroll < stake:
            history.append(bankroll)
            continue

        close_odds = row["best_close_t1"] if side == "t1" else row["best_close_t2"]
        if close_odds > 1:
            clv_values.append((raw_odds - close_odds) / close_odds * 100)

        execution_odds = max(1.01, raw_odds * (1 - config.slippage))
        profit = stake * execution_odds * (1 - config.tax_rate) - stake if is_win else -stake
        bankroll += profit
        total_profit += profit
        total_staked += stake
        bets += 1
        wins += int(is_win)
        history.append(bankroll)

    history_array = np.array(history)
    peaks = np.maximum.accumulate(history_array)
    drawdowns = (peaks - history_array) / (peaks + 1e-9)
    return {
        "final_bankroll": float(bankroll),
        "roi_pct": float((bankroll - config.initial_bankroll) / config.initial_bankroll * 100),
        "yield_pct": float(total_profit / total_staked * 100 if total_staked > 0 else 0.0),
        "max_drawdown_pct": float(np.max(drawdowns) * 100),
        "bets": float(bets),
        "win_rate_pct": float(wins / bets * 100 if bets > 0 else 0.0),
        "avg_stake": float(total_staked / bets if bets > 0 else 0.0),
        "avg_clv_pct": float(np.mean(clv_values) if clv_values else 0.0),
    }


def run_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run dynamic alpha/temperature grid search."""
    result_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    config = SimulationConfig()

    for alpha_k in ALPHA_GRID:
        for temperature_k in TEMP_GRID:
            predictions = dynamic_alpha_temperature(df, alpha_k, temperature_k)
            merged = pd.concat(
                [df.reset_index(drop=True), predictions[["alpha", "temperature", "probability"]]],
                axis=1,
            )
            for scope_name, scope_mask in {
                "2021+": merged["date"] >= pd.Timestamp("2021-01-01"),
                "2024+": merged["date"] >= pd.Timestamp("2024-01-01"),
            }.items():
                scoped = merged[scope_mask].copy()
                y_true = scoped["y_true"].to_numpy()
                probability = scoped["probability"].to_numpy()
                metrics = evaluate_probability(y_true, probability)
                fixed_percent = simulate_finance(scoped, probability, "fixed_percent", config)
                kelly = simulate_finance(scoped, probability, "kelly_025", config)
                result_rows.append(
                    {
                        "scope": scope_name,
                        "alpha_k": alpha_k,
                        "temperature_k": temperature_k,
                        "final_alpha": float(scoped["alpha"].iloc[-1]),
                        "mean_alpha": float(scoped["alpha"].mean()),
                        "final_temperature": float(scoped["temperature"].iloc[-1]),
                        "mean_temperature": float(scoped["temperature"].mean()),
                        **metrics,
                        "fixed_percent_final_bankroll": fixed_percent["final_bankroll"],
                        "fixed_percent_roi_pct": fixed_percent["roi_pct"],
                        "fixed_percent_yield_pct": fixed_percent["yield_pct"],
                        "fixed_percent_maxdd_pct": fixed_percent["max_drawdown_pct"],
                        "fixed_percent_bets": fixed_percent["bets"],
                        "kelly025_final_bankroll": kelly["final_bankroll"],
                        "kelly025_roi_pct": kelly["roi_pct"],
                        "kelly025_yield_pct": kelly["yield_pct"],
                        "kelly025_maxdd_pct": kelly["max_drawdown_pct"],
                        "kelly025_bets": kelly["bets"],
                    }
                )
            predictions["alpha_k"] = alpha_k
            predictions["temperature_k"] = temperature_k
            prediction_frames.append(predictions)

    return pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True)


def run_bet_settlement_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run K grid where alpha and temperature update after placed bets only."""
    result_rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    config = SimulationConfig()

    for alpha_k in ALPHA_GRID:
        for temperature_k in TEMP_GRID:
            predictions = dynamic_alpha_temperature_bet_settlement(df, alpha_k, temperature_k, config)
            merged = pd.concat(
                [df.reset_index(drop=True), predictions[["alpha", "temperature", "probability", "bet_side", "bet_win"]]],
                axis=1,
            )
            for scope_name, scope_mask in {
                "2021+": merged["date"] >= pd.Timestamp("2021-01-01"),
                "2024+": merged["date"] >= pd.Timestamp("2024-01-01"),
            }.items():
                scoped = merged[scope_mask].copy()
                y_true = scoped["y_true"].to_numpy()
                probability = scoped["probability"].to_numpy()
                metrics = evaluate_probability(y_true, probability)
                fixed_percent = simulate_finance(scoped, probability, "fixed_percent", config)
                kelly = simulate_finance(scoped, probability, "kelly_025", config)
                placed_bets = scoped["bet_side"].notna()
                result_rows.append(
                    {
                        "scope": scope_name,
                        "alpha_k": alpha_k,
                        "temperature_k": temperature_k,
                        "final_alpha": float(scoped["alpha"].iloc[-1]),
                        "mean_alpha": float(scoped["alpha"].mean()),
                        "final_temperature": float(scoped["temperature"].iloc[-1]),
                        "mean_temperature": float(scoped["temperature"].mean()),
                        "online_update_bets": float(placed_bets.sum()),
                        **metrics,
                        "fixed_percent_final_bankroll": fixed_percent["final_bankroll"],
                        "fixed_percent_roi_pct": fixed_percent["roi_pct"],
                        "fixed_percent_yield_pct": fixed_percent["yield_pct"],
                        "fixed_percent_maxdd_pct": fixed_percent["max_drawdown_pct"],
                        "fixed_percent_bets": fixed_percent["bets"],
                        "kelly025_final_bankroll": kelly["final_bankroll"],
                        "kelly025_roi_pct": kelly["roi_pct"],
                        "kelly025_yield_pct": kelly["yield_pct"],
                        "kelly025_maxdd_pct": kelly["max_drawdown_pct"],
                        "kelly025_bets": kelly["bets"],
                    }
                )
            predictions["alpha_k"] = alpha_k
            predictions["temperature_k"] = temperature_k
            prediction_frames.append(predictions)

    return pd.DataFrame(result_rows), pd.concat(prediction_frames, ignore_index=True)


def save_heatmap(results: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    """Save 2024+ grid heatmap."""
    data = results[results["scope"] == "2024+"].copy()
    pivot = data.pivot(index="temperature_k", columns="alpha_k", values=metric).sort_index().sort_index(axis=1)
    fig_width = max(11.0, 0.9 * len(pivot.columns) + 3.0)
    fig_height = max(6.5, 0.55 * len(pivot.index) + 2.0)
    plt.figure(figsize=(fig_width, fig_height))
    fmt = ".3f" if metric == "logloss" else ".0f"
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        annot_kws={"fontsize": 7},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"shrink": 0.85},
    )
    ax.set_title(title, weight="bold")
    ax.set_xlabel("K alpha")
    ax.set_ylabel("K temperature")
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{value:g}" for value in pivot.index], rotation=0)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_state_plot(predictions: pd.DataFrame, best_row: pd.Series) -> None:
    """Save alpha and temperature trajectories for the best 2024+ LogLoss config."""
    data = predictions[
        (predictions["alpha_k"] == best_row["alpha_k"])
        & (predictions["temperature_k"] == best_row["temperature_k"])
    ].copy()
    data["date"] = pd.to_datetime(data["date"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.lineplot(data=data, x="date", y="alpha", ax=axes[0], color="steelblue")
    axes[0].set_title("Dynamic alpha — aktualizacja typu Elo", weight="bold")
    axes[0].set_ylabel("Alpha")
    sns.lineplot(data=data, x="date", y="temperature", ax=axes[1], color="darkorange")
    axes[1].set_title("Dynamic temperature — ostrość prawdopodobieństw", weight="bold")
    axes[1].set_ylabel("Temperature")
    axes[1].set_xlabel("Data")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "hybrid_dynamic_alpha_temperature_states.png", bbox_inches="tight")
    plt.close()


def save_summary_plot(results: pd.DataFrame) -> None:
    """Save readable comparison plot for top 2024+ dynamic configs."""
    data = results[results["scope"] == "2024+"].copy().sort_values("logloss").head(10)
    data["variant"] = data.apply(
        lambda row: f"aK={row['alpha_k']}, tK={row['temperature_k']}", axis=1
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    sns.barplot(data=data, x="logloss", y="variant", hue="variant", legend=False, ax=axes[0], palette="mako")
    axes[0].set_title("Top dynamic configs — LogLoss 2024+", weight="bold")
    axes[0].set_xlabel("LogLoss")
    axes[0].set_ylabel("")
    sns.barplot(
        data=data,
        x="kelly025_final_bankroll",
        y="variant",
        hue="variant",
        legend=False,
        ax=axes[1],
        palette="crest",
    )
    axes[1].set_title("Ten sam top — Kelly 0.25 final bankroll", weight="bold")
    axes[1].set_xlabel("Final bankroll")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "hybrid_dynamic_alpha_temperature_top2024.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run dynamic alpha-temperature experiment."""
    configure_style()
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    results, predictions = run_grid(df)
    bet_results, bet_predictions = run_bet_settlement_grid(df)

    results.to_csv(ASSET_DIR / "hybrid_dynamic_alpha_temperature_grid.csv", index=False)
    predictions.to_csv(ASSET_DIR / "hybrid_dynamic_alpha_temperature_predictions.csv", index=False)
    bet_results.to_csv(ASSET_DIR / "hybrid_dynamic_alpha_temperature_bet_k_grid.csv", index=False)
    bet_predictions.to_csv(ASSET_DIR / "hybrid_dynamic_alpha_temperature_bet_k_predictions.csv", index=False)

    best_logloss = results[results["scope"] == "2024+"].sort_values("logloss").iloc[0]
    best_bankroll = results[results["scope"] == "2024+"].sort_values(
        "kelly025_final_bankroll", ascending=False
    ).iloc[0]
    save_heatmap(
        results,
        "logloss",
        ASSET_DIR / "hybrid_dynamic_alpha_temperature_logloss_heatmap.png",
        "Dynamic alpha + temperature — LogLoss 2024+",
    )
    save_heatmap(
        results,
        "kelly025_final_bankroll",
        ASSET_DIR / "hybrid_dynamic_alpha_temperature_kelly_heatmap.png",
        "Dynamic alpha + temperature — Kelly 0.25 final bankroll 2024+",
    )
    save_state_plot(predictions, best_logloss)
    save_summary_plot(results)

    save_heatmap(
        bet_results,
        "logloss",
        ASSET_DIR / "hybrid_dynamic_alpha_temperature_bet_k_logloss_heatmap.png",
        "Bet-settled dynamic alpha + temperature — LogLoss 2024+",
    )
    save_heatmap(
        bet_results,
        "kelly025_final_bankroll",
        ASSET_DIR / "hybrid_dynamic_alpha_temperature_bet_k_kelly_heatmap.png",
        "Bet-settled dynamic alpha + temperature — Kelly 0.25 final bankroll 2024+",
    )

    print("Best 2024+ LogLoss config:")
    print(best_logloss.to_string())
    print("\nBest 2024+ Kelly 0.25 bankroll config:")
    print(best_bankroll.to_string())

    best_bet_logloss = bet_results[bet_results["scope"] == "2024+"].sort_values("logloss").iloc[0]
    best_bet_bankroll = bet_results[bet_results["scope"] == "2024+"].sort_values(
        "kelly025_final_bankroll", ascending=False
    ).iloc[0]
    print("\nBest 2024+ bet-settled LogLoss config:")
    print(best_bet_logloss.to_string())
    print("\nBest 2024+ bet-settled Kelly 0.25 bankroll config:")
    print(best_bet_bankroll.to_string())


if __name__ == "__main__":
    main()
