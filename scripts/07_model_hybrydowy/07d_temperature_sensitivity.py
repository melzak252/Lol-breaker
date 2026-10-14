"""Test temperature scaling for metamodel and hybrid probabilities.

This script is intentionally lightweight: it reuses already generated Chapter 7
predictions from ``docs/assets/hybrid_point7/hybrid_model_input_predictions.csv``
and does not retrain LightGBM. The goal is to check whether probability
sharpening/flattening improves statistical or financial behaviour.
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

TEMPERATURES = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
ALPHAS = [0.00, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]
PLOT_VARIANTS = [
    "Metamodel",
    "Hybrid alpha=0.25",
    "Hybrid alpha=0.50",
    "Hybrid alpha=0.75",
]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for temperature-sensitivity financial diagnostics."""

    initial_bankroll: float = 100.0
    min_stake: float = 2.0
    max_stake: float = 100.0
    tax_rate: float = 0.12
    slippage: float = 0.01
    ev_threshold: float = 0.05
    kelly_fraction: float = 0.25


def logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits with numerical clipping.

    Args:
        probability: Probability array.

    Returns:
        Logit-transformed values.
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
    """Apply temperature scaling to probability values.

    Args:
        probability: Probability array.
        temperature: Positive temperature. Values below 1 sharpen probabilities,
            values above 1 flatten them toward 0.5.

    Returns:
        Temperature-scaled probabilities.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return sigmoid(logit(probability) / temperature)


def blend_linear(model_prob: np.ndarray, market_prob: np.ndarray, alpha: float) -> np.ndarray:
    """Blend model and market probabilities linearly.

    Args:
        model_prob: Model probabilities for team 1.
        market_prob: Market probabilities for team 1.
        alpha: Weight assigned to model probabilities.

    Returns:
        Hybrid probability array.
    """
    return alpha * model_prob + (1 - alpha) * market_prob


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate probability metrics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Dictionary with AUC, LogLoss, Brier and ECE.
    """
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "LogLoss": log_loss(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "ECE": calculate_ece(y_true, y_prob),
    }


def simulate_fixed_stake(
    df: pd.DataFrame,
    probability: np.ndarray,
    initial_bankroll: float = 100.0,
    stake: float = 10.0,
    tax_rate: float = 0.12,
    slippage: float = 0.01,
    ev_threshold: float = 0.05,
) -> dict[str, float]:
    """Run fixed-stake betting simulation on best opening odds.

    Args:
        df: Chronological match dataframe.
        probability: Team 1 win probabilities aligned with ``df``.
        initial_bankroll: Starting bankroll.
        stake: Fixed stake per bet.
        tax_rate: Turnover tax rate.
        slippage: Odds slippage applied to execution odds.
        ev_threshold: Minimum EV required to place a bet.

    Returns:
        Financial summary dictionary.
    """
    bankroll = initial_bankroll
    total_profit = 0.0
    total_staked = 0.0
    bets = 0
    wins = 0
    history = [bankroll]
    net_multiplier = 1 - tax_rate

    for (_, row), p_t1 in zip(df.iterrows(), probability):
        ev_t1 = row["best_open_t1"] * net_multiplier * p_t1 - 1
        ev_t2 = row["best_open_t2"] * net_multiplier * (1 - p_t1) - 1

        if ev_t1 > ev_t2 and ev_t1 > ev_threshold:
            raw_odds = row["best_open_t1"]
            is_win = int(row["y_true"] == 1)
        elif ev_t2 > ev_threshold:
            raw_odds = row["best_open_t2"]
            is_win = int(row["y_true"] == 0)
        else:
            history.append(bankroll)
            continue

        if bankroll < stake:
            history.append(bankroll)
            continue

        execution_odds = max(1.01, raw_odds * (1 - slippage))
        profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
        bankroll += profit
        total_profit += profit
        total_staked += stake
        bets += 1
        wins += is_win
        history.append(bankroll)

    history_array = np.array(history)
    peaks = np.maximum.accumulate(history_array)
    drawdowns = (peaks - history_array) / (peaks + 1e-9)

    return {
        "FinalBankroll": bankroll,
        "ROI": (bankroll - initial_bankroll) / initial_bankroll * 100,
        "Yield": total_profit / total_staked * 100 if total_staked > 0 else 0.0,
        "MaxDD": float(np.max(drawdowns) * 100),
        "Bets": bets,
        "WinRate": wins / bets * 100 if bets > 0 else 0.0,
        "TotalStaked": total_staked,
    }


def calculate_kelly_stake(bankroll: float, odds: float, probability: float, config: SimulationConfig) -> float:
    """Calculate fractional Kelly stake after tax and slippage.

    Args:
        bankroll: Current bankroll.
        odds: Raw decimal odds at opening.
        probability: Model probability for the selected side.
        config: Financial simulation configuration.

    Returns:
        Stake clipped to configured min/max constraints.
    """
    execution_odds = max(1.01, odds * (1 - config.slippage))
    b_value = execution_odds * (1 - config.tax_rate) - 1
    if b_value <= 0:
        return 0.0
    full_kelly = ((b_value * probability) - (1 - probability)) / b_value
    if full_kelly <= 0:
        return 0.0
    stake = bankroll * config.kelly_fraction * full_kelly
    return float(min(max(stake, config.min_stake), config.max_stake))


def simulate_unit_and_kelly(
    df: pd.DataFrame,
    probability: np.ndarray,
    config: SimulationConfig | None = None,
) -> dict[str, float]:
    """Run unit-stake and Kelly diagnostics for temperature variants.

    The unit-stake part avoids the visual artefact of fixed-stake bankruptcy:
    every qualifying bet contributes one stake, so Yield and Bets describe signal
    quality rather than path-dependent capital survival.

    Args:
        df: Chronological match dataframe.
        probability: Team 1 win probabilities aligned with ``df``.
        config: Optional simulation configuration.

    Returns:
        Dictionary with unit-stake and Kelly metrics.
    """
    config = config or SimulationConfig()
    bankroll = config.initial_bankroll
    history = [bankroll]
    unit_profit = 0.0
    unit_staked = 0.0
    kelly_profit = 0.0
    kelly_staked = 0.0
    bets = 0
    wins = 0
    net_multiplier = 1 - config.tax_rate

    for (_, row), p_t1 in zip(df.iterrows(), probability):
        ev_t1 = row["best_open_t1"] * net_multiplier * p_t1 - 1
        ev_t2 = row["best_open_t2"] * net_multiplier * (1 - p_t1) - 1

        if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
            raw_odds = float(row["best_open_t1"])
            side_probability = float(p_t1)
            is_win = int(row["y_true"] == 1)
        elif ev_t2 > config.ev_threshold:
            raw_odds = float(row["best_open_t2"])
            side_probability = float(1 - p_t1)
            is_win = int(row["y_true"] == 0)
        else:
            history.append(bankroll)
            continue

        execution_odds = max(1.01, raw_odds * (1 - config.slippage))
        unit_result = execution_odds * net_multiplier - 1 if is_win else -1
        unit_profit += unit_result
        unit_staked += 1

        stake = calculate_kelly_stake(bankroll, raw_odds, side_probability, config)
        if stake > 0 and bankroll >= stake:
            profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
            bankroll += profit
            kelly_profit += profit
            kelly_staked += stake
        history.append(bankroll)
        bets += 1
        wins += is_win

    history_array = np.array(history)
    peaks = np.maximum.accumulate(history_array)
    drawdowns = (peaks - history_array) / (peaks + 1e-9)
    return {
        "UnitYield": unit_profit / unit_staked * 100 if unit_staked > 0 else 0.0,
        "UnitProfit": unit_profit,
        "KellyFinalBankroll": bankroll,
        "KellyYield": kelly_profit / kelly_staked * 100 if kelly_staked > 0 else 0.0,
        "KellyMaxDD": float(np.max(drawdowns) * 100),
        "Bets": bets,
        "WinRate": wins / bets * 100 if bets > 0 else 0.0,
    }


def save_temperature_plot(results: pd.DataFrame) -> None:
    """Save temperature sensitivity plot.

    Args:
        results: Temperature experiment result table.
    """
    sns.set_theme(style="whitegrid", context="talk")
    subset = results[results["Scope"] == "2024+"]
    plot_df = subset[subset["Variant"].isin(PLOT_VARIANTS)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ["LogLoss", "ECE", "ROI", "Yield"]
    for ax, metric in zip(axes.flatten(), metrics):
        sns.lineplot(
            data=plot_df,
            x="Temperature",
            y=metric,
            hue="Variant",
            marker="o",
            ax=ax,
        )
        ax.axvline(1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(metric)
        ax.set_xlabel("Temperature")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(ASSET_DIR / "hybrid_temperature_sensitivity_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_readable_temperature_plots(results: pd.DataFrame) -> None:
    """Save cleaner temperature plots without path-dependent ROI artefacts.

    Args:
        results: Temperature experiment result table.
    """
    sns.set_theme(style="whitegrid", context="talk")
    subset = results[results["Scope"] == "2024+"].copy()
    plot_df = subset[subset["Variant"].isin(PLOT_VARIANTS)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metric_specs = [
        ("LogLoss", "LogLoss — niżej lepiej"),
        ("ECE", "ECE — niżej lepiej"),
        ("UnitYield", "Unit Yield przy EV > 5%"),
        ("Bets", "Liczba zakładów po filtrze EV"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), metric_specs):
        sns.lineplot(data=plot_df, x="Temperature", y=metric, hue="Variant", marker="o", ax=ax)
        ax.axvline(1.0, color="black", linestyle="--", alpha=0.45)
        ax.axvspan(0.60, 0.80, color="green", alpha=0.08, label="T=0.60–0.80")
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Temperature")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flatten():
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Temperature scaling — jakość prawdopodobieństw i filtr EV 2024+", weight="bold")
    fig.text(
        0.5,
        0.02,
        "Unit Yield nie zależy od ścieżki bankrolla, dlatego lepiej pokazuje sygnał niż surowy ROI fixed-stake.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(ASSET_DIR / "hybrid_temperature_sensitivity_readable_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.lineplot(data=plot_df, x="Temperature", y="KellyFinalBankroll", hue="Variant", marker="o", ax=axes[0])
    axes[0].axvline(1.0, color="black", linestyle="--", alpha=0.45)
    axes[0].set_title("Kelly 0.25 final bankroll", weight="bold")
    axes[0].set_xlabel("Temperature")
    axes[0].grid(alpha=0.25)
    sns.lineplot(data=plot_df, x="Temperature", y="KellyMaxDD", hue="Variant", marker="o", ax=axes[1])
    axes[1].axvline(1.0, color="black", linestyle="--", alpha=0.45)
    axes[1].set_title("Kelly 0.25 MaxDD", weight="bold")
    axes[1].set_xlabel("Temperature")
    axes[1].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Temperature scaling — diagnostyka bankrolla 2024+", weight="bold")
    fig.text(
        0.5,
        0.03,
        "Bankroll jest metryką ścieżkową: należy interpretować go razem z MaxDD i liczbą zakładów.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(ASSET_DIR / "hybrid_temperature_kelly_diagnostics_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_alpha(variant: str) -> float | None:
    """Extract alpha value from hybrid variant name.

    Args:
        variant: Variant name, for example ``Hybrid alpha=0.50``.

    Returns:
        Parsed alpha value or ``None`` for non-hybrid variants.
    """
    prefix = "Hybrid alpha="
    if not variant.startswith(prefix):
        return None
    return float(variant.replace(prefix, ""))


def normalize_metric(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Normalize a metric to a 0-1 score.

    Args:
        values: Metric values.
        higher_is_better: Whether larger raw values should receive larger scores.

    Returns:
        Normalized score where 1 means best in the grid.
    """
    min_value = values.min()
    max_value = values.max()
    if np.isclose(max_value, min_value):
        return pd.Series(np.ones(len(values)), index=values.index)
    normalized = (values - min_value) / (max_value - min_value)
    if higher_is_better:
        return normalized
    return 1 - normalized


def save_alpha_temperature_heatmaps(results: pd.DataFrame) -> None:
    """Save heatmaps for fixed alpha and temperature combinations.

    Args:
        results: Temperature experiment result table.
    """
    sns.set_theme(style="white", context="talk")
    subset = results[(results["Scope"] == "2024+") & results["Variant"].str.startswith("Hybrid alpha=")].copy()
    subset["Alpha"] = subset["Variant"].map(parse_alpha)
    subset = subset.dropna(subset=["Alpha"])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    metric_specs = [
        ("LogLoss", "LogLoss — niżej lepiej", "viridis_r", ".3f"),
        ("ECE", "ECE — niżej lepiej", "mako_r", ".3f"),
        ("UnitYield", "Unit Yield — wyżej lepiej", "YlGn", ".1f"),
        ("KellyMaxDD", "MaxDD Kelly 0.25 — niżej lepiej", "rocket_r", ".1f"),
    ]
    for ax, (metric, title, cmap, fmt) in zip(axes.flatten(), metric_specs):
        pivot = subset.pivot(index="Alpha", columns="Temperature", values=metric).sort_index(ascending=False)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 7},
        )
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Temperature T")
        ax.set_ylabel("Alpha α")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Stałe wartości α i T — najważniejsze metryki 2024+", weight="bold")
    fig.text(
        0.5,
        0.02,
        "Heatmapy pokazują kompromis: nie szukamy jednej komórki, tylko regionu dobrego jednocześnie probabilistycznie i finansowo.",
        ha="center",
        fontsize=11,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(ASSET_DIR / "hybrid_alpha_temperature_metric_heatmaps_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    scored = subset.copy()
    scored["score_logloss"] = normalize_metric(scored["LogLoss"], higher_is_better=False)
    scored["score_ece"] = normalize_metric(scored["ECE"], higher_is_better=False)
    scored["score_yield"] = normalize_metric(scored["UnitYield"], higher_is_better=True)
    scored["score_drawdown"] = normalize_metric(scored["KellyMaxDD"], higher_is_better=False)
    scored["RegionScore"] = (
        0.35 * scored["score_logloss"]
        + 0.20 * scored["score_ece"]
        + 0.30 * scored["score_yield"]
        + 0.15 * scored["score_drawdown"]
    )
    scored.to_csv(ASSET_DIR / "hybrid_alpha_temperature_region_score_2024.csv", index=False)

    pivot = scored.pivot(index="Alpha", columns="Temperature", values="RegionScore").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Composite region score", "shrink": 0.85},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Nałożenie metryk: region kompromisu α/T 2024+", weight="bold")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Alpha α")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.text(
        0.5,
        0.02,
        "Score = 35% LogLoss + 20% ECE + 30% Unit Yield + 15% MaxDD. To heurystyka diagnostyczna, nie funkcja celu produkcyjnego.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ASSET_DIR / "hybrid_alpha_temperature_region_score_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    roi_pivot = subset.pivot(index="Alpha", columns="Temperature", values="ROI").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        roi_pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "ROI %", "shrink": 0.85},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("ROI fixed-stake dla stałych wartości α i T 2024+", weight="bold")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Alpha α")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.text(
        0.5,
        0.02,
        "ROI jest metryką ścieżkową i zależy od liczby oraz kolejności zakładów; należy czytać ją razem z Unit Yield, MaxDD i Bets.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ASSET_DIR / "hybrid_alpha_temperature_roi_heatmap_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run temperature scaling sensitivity tests."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.dropna(subset=["prob_model", "prob_market_open", "best_open_t1", "best_open_t2", "y_true"])
    df = df.sort_values("date").reset_index(drop=True)

    scopes = {
        "2021+": df,
        "2024+": df[df["date"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True),
    }

    rows = []
    for scope_name, scope_df in scopes.items():
        y_true = scope_df["y_true"].to_numpy()
        market_prob = scope_df["prob_market_open"].to_numpy()
        model_prob_raw = scope_df["prob_model"].to_numpy()

        for temperature in TEMPERATURES:
            model_prob = apply_temperature(model_prob_raw, temperature)
            variants = {"Market Open": market_prob, "Metamodel": model_prob}
            for alpha in ALPHAS:
                variants[f"Hybrid alpha={alpha:.2f}"] = blend_linear(model_prob, market_prob, alpha)

            for variant_name, prob in variants.items():
                metric_row = evaluate_metrics(y_true, prob)
                financial_row = simulate_fixed_stake(scope_df, prob)
                robust_financial_row = simulate_unit_and_kelly(scope_df, prob)
                rows.append(
                    {
                        "Scope": scope_name,
                        "Variant": variant_name,
                        "Temperature": temperature,
                        **metric_row,
                        **financial_row,
                        **robust_financial_row,
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(ASSET_DIR / "hybrid_temperature_sensitivity.csv", index=False)
    save_temperature_plot(results)
    save_readable_temperature_plots(results)
    save_alpha_temperature_heatmaps(results)

    print("\n=== TEMPERATURE SENSITIVITY: BEST BY SCOPE/VARIANT ===")
    for (scope, variant), group in results.groupby(["Scope", "Variant"]):
        best_logloss = group.sort_values("LogLoss").iloc[0]
        best_roi = group.sort_values("ROI", ascending=False).iloc[0]
        print(
            f"{scope} | {variant}: "
            f"best LogLoss T={best_logloss['Temperature']:.2f} "
            f"LL={best_logloss['LogLoss']:.5f} ROI={best_logloss['ROI']:.2f}% | "
            f"best ROI T={best_roi['Temperature']:.2f} "
            f"ROI={best_roi['ROI']:.2f}% LL={best_roi['LogLoss']:.5f}"
        )

    print(f"\nSaved: {ASSET_DIR / 'hybrid_temperature_sensitivity.csv'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_temperature_sensitivity_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_temperature_sensitivity_readable_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_temperature_kelly_diagnostics_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_alpha_temperature_metric_heatmaps_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_alpha_temperature_region_score_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_alpha_temperature_roi_heatmap_2024.png'}")


if __name__ == "__main__":
    main()
