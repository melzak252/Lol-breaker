"""Simulate fixed-percentage staking for Chapter 7 hybrid candidates.

The experiment reuses already generated hybrid predictions and tests a simple
bankroll rule: stake a fixed percentage of current bankroll, but never less
than a specified minimum stake. This is closer to practical staking than the
previous fixed-10 diagnostic, while still being simpler than Kelly sizing.
"""

from __future__ import annotations

import sys
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

INITIAL_BANKROLL = 100.0
STAKE_FRACTION = 0.02
MIN_STAKE = 2.0
MAX_STAKE = 100.0
TAX_RATE = 0.12
SLIPPAGE = 0.01
EV_THRESHOLD = 0.05


def logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits with clipping.

    Args:
        probability: Probability array.

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
    """Apply temperature scaling.

    Args:
        probability: Probability values.
        temperature: Positive temperature parameter.

    Returns:
        Temperature-scaled probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return sigmoid(logit(probability) / temperature)


def blend_linear(model_prob: np.ndarray, market_prob: np.ndarray, alpha: float) -> np.ndarray:
    """Linearly blend model and market probabilities.

    Args:
        model_prob: Model probability for team 1.
        market_prob: Market probability for team 1.
        alpha: Model weight.

    Returns:
        Hybrid probability for team 1.
    """
    return alpha * model_prob + (1 - alpha) * market_prob


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate predictive probability metrics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with predictive metrics.
    """
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "ECE": calculate_ece(y_true, y_prob),
    }


def simulate_fixed_percent(
    df: pd.DataFrame,
    probability: np.ndarray,
    variant: str,
    scope: str,
    initial_bankroll: float = INITIAL_BANKROLL,
    stake_fraction: float = STAKE_FRACTION,
    min_stake: float = MIN_STAKE,
    max_stake: float = MAX_STAKE,
    tax_rate: float = TAX_RATE,
    slippage: float = SLIPPAGE,
    ev_threshold: float = EV_THRESHOLD,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Run fixed-percent staking simulation with minimum stake.

    Args:
        df: Chronological dataframe with odds and labels.
        probability: Team 1 win probabilities aligned with ``df``.
        variant: Variant name.
        scope: Scope label, e.g. 2024+.
        initial_bankroll: Starting bankroll.
        stake_fraction: Fraction of current bankroll staked per qualifying bet.
        min_stake: Minimum allowed stake.
        max_stake: Maximum allowed stake.
        tax_rate: Turnover tax rate.
        slippage: Execution slippage on decimal odds.
        ev_threshold: Minimum EV threshold required to bet.

    Returns:
        Summary dictionary and per-match bankroll history dataframe.
    """
    bankroll = initial_bankroll
    total_profit = 0.0
    total_staked = 0.0
    bets = 0
    wins = 0
    skipped_bankroll = 0
    net_multiplier = 1 - tax_rate
    history_rows = []

    for (_, row), p_t1 in zip(df.iterrows(), probability):
        ev_t1 = row["best_open_t1"] * net_multiplier * p_t1 - 1
        ev_t2 = row["best_open_t2"] * net_multiplier * (1 - p_t1) - 1
        placed = False
        profit = 0.0
        stake = 0.0
        side = "none"

        if ev_t1 > ev_t2 and ev_t1 > ev_threshold:
            raw_odds = row["best_open_t1"]
            is_win = int(row["y_true"] == 1)
            selected_ev = ev_t1
            side = "t1"
        elif ev_t2 > ev_threshold:
            raw_odds = row["best_open_t2"]
            is_win = int(row["y_true"] == 0)
            selected_ev = ev_t2
            side = "t2"
        else:
            raw_odds = np.nan
            is_win = 0
            selected_ev = np.nan

        if side != "none":
            stake = min(max(bankroll * stake_fraction, min_stake), max_stake)
            if bankroll >= stake:
                execution_odds = max(1.01, raw_odds * (1 - slippage))
                profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
                bankroll += profit
                total_profit += profit
                total_staked += stake
                bets += 1
                wins += is_win
                placed = True
            else:
                skipped_bankroll += 1

        history_rows.append(
            {
                "Scope": scope,
                "Variant": variant,
                "date": row["date"],
                "golgg_match_id": row.get("golgg_match_id"),
                "bankroll": bankroll,
                "placed": placed,
                "stake": stake if placed else 0.0,
                "profit": profit,
                "side": side,
                "selected_ev": selected_ev,
            }
        )

    history = pd.DataFrame(history_rows)
    bankroll_values = np.concatenate([[initial_bankroll], history["bankroll"].to_numpy()])
    peaks = np.maximum.accumulate(bankroll_values)
    drawdowns = (peaks - bankroll_values) / (peaks + 1e-9)

    summary = {
        "Scope": scope,
        "Variant": variant,
        "InitialBankroll": initial_bankroll,
        "FinalBankroll": bankroll,
        "Profit": bankroll - initial_bankroll,
        "ROI": (bankroll - initial_bankroll) / initial_bankroll * 100,
        "Yield": total_profit / total_staked * 100 if total_staked > 0 else 0.0,
        "MaxDD": float(np.max(drawdowns) * 100),
        "Bets": bets,
        "WinRate": wins / bets * 100 if bets > 0 else 0.0,
        "TotalStaked": total_staked,
        "AvgStake": total_staked / bets if bets > 0 else 0.0,
        "MaxStake": max_stake,
        "SkippedBankroll": skipped_bankroll,
    }
    return summary, history


def build_variants(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build probability variants to test.

    Args:
        df: Input dataframe.

    Returns:
        Mapping from variant name to probability array.
    """
    market = df["prob_market_open"].to_numpy()
    model = df["prob_model"].to_numpy()
    model_t06 = apply_temperature(model, 0.60)
    model_t07 = apply_temperature(model, 0.70)
    return {
        "Market Avg Open": market,
        "Metamodel T=1.00": model,
        "Metamodel T=0.60": model_t06,
        "Hybrid a=0.48 T=1.00": blend_linear(model, market, 0.48),
        "Hybrid a=0.48 T=0.70": blend_linear(model_t07, market, 0.48),
        "Hybrid a=0.48 T=0.60": blend_linear(model_t06, market, 0.48),
        "Hybrid a=0.62 T=0.80": blend_linear(apply_temperature(model, 0.80), market, 0.62),
    }


def save_bankroll_plot(history: pd.DataFrame, scope: str) -> None:
    """Save bankroll-over-time plot for fixed-percent simulation.

    Args:
        history: Bankroll history table.
        scope: Scope filter.
    """
    sns.set_theme(style="whitegrid", context="talk")
    plot_df = history[history["Scope"] == scope].copy()
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.lineplot(data=plot_df, x="date", y="bankroll", hue="Variant", ax=ax, linewidth=2)
    ax.axhline(INITIAL_BANKROLL, color="black", linestyle="--", alpha=0.5)
    ax.set_title(
        f"Fixed-percent staking: {scope}, stake=clip(2% bankroll, 2 PLN, 100 PLN)",
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll [PLN]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    output_name = f"hybrid_fixed_percent_bankroll_{scope.replace('+', 'plus')}.png"
    fig.savefig(ASSET_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run fixed-percent staking sensitivity test."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.dropna(subset=["prob_model", "prob_market_open", "best_open_t1", "best_open_t2", "y_true"])
    df = df.sort_values("date").reset_index(drop=True)

    scopes = {
        "2021+": df,
        "2024+": df[df["date"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True),
    }

    summary_rows = []
    history_frames = []
    for scope_name, scope_df in scopes.items():
        variants = build_variants(scope_df)
        y_true = scope_df["y_true"].to_numpy()
        for variant_name, probability in variants.items():
            metrics = evaluate_metrics(y_true, probability)
            summary, history = simulate_fixed_percent(scope_df, probability, variant_name, scope_name)
            summary_rows.append({**summary, **metrics})
            history_frames.append(history)

    summary_df = pd.DataFrame(summary_rows)
    history_df = pd.concat(history_frames, ignore_index=True)
    summary_df.to_csv(ASSET_DIR / "hybrid_fixed_percent_staking_summary.csv", index=False)
    history_df.to_csv(ASSET_DIR / "hybrid_fixed_percent_staking_history.csv", index=False)

    save_bankroll_plot(history_df, "2024+")
    save_bankroll_plot(history_df, "2021+")

    print("=== Fixed-percent staking summary: 2024+ ===")
    cols = ["Variant", "FinalBankroll", "ROI", "Yield", "MaxDD", "Bets", "AvgStake", "WinRate", "LogLoss"]
    print(summary_df[summary_df["Scope"] == "2024+"][cols].sort_values("FinalBankroll", ascending=False).to_string(index=False))
    print("\nSaved fixed-percent staking artefacts.")


if __name__ == "__main__":
    main()
