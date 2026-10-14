"""Evaluate entropy-based adaptive temperature scaling for hybrid betting.

The experiment compares static temperature scaling with a simple sample-wise
temperature rule inspired by entropy-based adaptive temperature scaling. The
rule is intentionally interpretable: probability entropy controls whether the
metamodel prediction is sharpened or flattened before blending with the market.
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

ALPHAS = [0.40, 0.50, 0.60]
BASE_TEMPERATURES = [0.50, 0.60, 0.70, 0.80, 1.00]
ENTROPY_STRENGTHS = [-1.00, -0.50, 0.00, 0.50, 1.00]
TEMPERATURE_BOUNDS = (0.50, 2.00)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for unit and Kelly-style financial diagnostics."""

    initial_bankroll: float = 100.0
    min_stake: float = 2.0
    max_stake: float = 100.0
    tax_rate: float = 0.12
    slippage: float = 0.01
    ev_threshold: float = 0.05
    kelly_fraction: float = 0.25


def logit(probability: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits with clipping.

    Args:
        probability: Probability vector.

    Returns:
        Logit-transformed vector.
    """
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities.

    Args:
        values: Logit vector.

    Returns:
        Probability vector.
    """
    return 1 / (1 + np.exp(-values))


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Calculate normalized binary entropy in the [0, 1] range.

    Args:
        probability: Probability vector.

    Returns:
        Normalized entropy. Values near 1 mean near-even predictions.
    """
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped))
    return entropy / np.log(2)


def entropy_temperature(
    probability: np.ndarray,
    base_temperature: float,
    entropy_strength: float,
) -> np.ndarray:
    """Create sample-wise temperatures from prediction entropy.

    Positive strength flattens high-entropy examples and sharpens low-entropy
    examples. Negative strength tests the opposite direction.

    Args:
        probability: Raw metamodel probability vector.
        base_temperature: Baseline temperature level.
        entropy_strength: Strength of entropy-dependent adjustment.

    Returns:
        Sample-wise temperature vector clipped to safe bounds.
    """
    entropy = binary_entropy(probability)
    temperature = base_temperature * np.exp(entropy_strength * (entropy - 0.5))
    return np.clip(temperature, *TEMPERATURE_BOUNDS)


def apply_temperature(probability: np.ndarray, temperature: np.ndarray | float) -> np.ndarray:
    """Apply scalar or sample-wise temperature scaling.

    Args:
        probability: Raw probability vector.
        temperature: Scalar or vector of temperatures.

    Returns:
        Temperature-scaled probabilities.
    """
    return sigmoid(logit(probability) / temperature)


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate probabilistic metrics.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Metric dictionary.
    """
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "LogLoss": float(log_loss(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": calculate_ece(y_true, y_prob),
    }


def calculate_kelly_stake(bankroll: float, odds: float, probability: float, config: SimulationConfig) -> float:
    """Calculate fractional Kelly stake.

    Args:
        bankroll: Current bankroll.
        odds: Decimal odds before slippage.
        probability: Probability of selected side.
        config: Financial simulation configuration.

    Returns:
        Stake size after min/max constraints.
    """
    execution_odds = max(1.01, odds * (1 - config.slippage))
    net_odds = execution_odds * (1 - config.tax_rate)
    edge_fraction = (probability * net_odds - 1) / max(net_odds - 1, 1e-9)
    stake = bankroll * max(edge_fraction, 0.0) * config.kelly_fraction
    if stake <= 0:
        return 0.0
    return min(max(stake, config.min_stake), config.max_stake, bankroll)


def simulate_financials(df: pd.DataFrame, probability: np.ndarray, config: SimulationConfig) -> dict[str, float]:
    """Simulate unit-yield and Kelly diagnostics.

    Args:
        df: Match dataframe with odds and labels.
        probability: Probability for team 1.
        config: Financial simulation configuration.

    Returns:
        Financial diagnostics dictionary.
    """
    bankroll = config.initial_bankroll
    history = [bankroll]
    unit_profit = 0.0
    unit_staked = 0.0
    bets = 0
    wins = 0
    net_multiplier = 1 - config.tax_rate

    for idx, row in df.reset_index(drop=True).iterrows():
        prob_t1 = float(probability[idx])
        prob_t2 = 1 - prob_t1
        odds_t1 = float(row["best_open_t1"])
        odds_t2 = float(row["best_open_t2"])
        ev_t1 = prob_t1 * odds_t1 * net_multiplier - 1
        ev_t2 = prob_t2 * odds_t2 * net_multiplier - 1

        if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
            side_probability = prob_t1
            raw_odds = odds_t1
            is_win = int(row["y_true"]) == 1
        elif ev_t2 > config.ev_threshold:
            side_probability = prob_t2
            raw_odds = odds_t2
            is_win = int(row["y_true"]) == 0
        else:
            history.append(bankroll)
            continue

        execution_odds = max(1.01, raw_odds * (1 - config.slippage))
        unit_profit += execution_odds * net_multiplier - 1 if is_win else -1
        unit_staked += 1

        stake = calculate_kelly_stake(bankroll, raw_odds, side_probability, config)
        if stake > 0 and bankroll >= stake:
            profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
            bankroll += profit
        history.append(bankroll)
        bets += 1
        wins += int(is_win)

    history_array = np.array(history)
    peaks = np.maximum.accumulate(history_array)
    drawdowns = (peaks - history_array) / (peaks + 1e-9)
    return {
        "UnitYield": unit_profit / unit_staked * 100 if unit_staked > 0 else 0.0,
        "KellyFinalBankroll": bankroll,
        "KellyMaxDD": float(np.max(drawdowns) * 100),
        "Bets": bets,
        "WinRate": wins / bets * 100 if bets > 0 else 0.0,
    }


def build_predictions(df: pd.DataFrame, alpha: float, base_temperature: float, entropy_strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Build entropy-temperature hybrid predictions.

    Args:
        df: Match dataframe.
        alpha: Hybrid model weight.
        base_temperature: Baseline temperature.
        entropy_strength: Entropy adjustment strength.

    Returns:
        Tuple of hybrid probabilities and sample-wise temperatures.
    """
    raw_model = df["prob_model"].to_numpy()
    market = df["prob_market_open"].to_numpy()
    temperatures = entropy_temperature(raw_model, base_temperature, entropy_strength)
    model_scaled = apply_temperature(raw_model, temperatures)
    hybrid = alpha * model_scaled + (1 - alpha) * market
    return hybrid, temperatures


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run entropy-temperature grid for 2021+ and 2024+ scopes.

    Args:
        df: Full input dataframe.

    Returns:
        Result dataframe.
    """
    config = SimulationConfig()
    rows: list[dict[str, float | str]] = []
    scopes = {
        "2021+": df,
        "2024+": df[df["date"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True),
    }
    for scope_name, scope_df in scopes.items():
        y_true = scope_df["y_true"].to_numpy()
        for alpha in ALPHAS:
            for base_temperature in BASE_TEMPERATURES:
                for entropy_strength in ENTROPY_STRENGTHS:
                    probability, temperatures = build_predictions(scope_df, alpha, base_temperature, entropy_strength)
                    rows.append(
                        {
                            "Scope": scope_name,
                            "Alpha": alpha,
                            "BaseTemperature": base_temperature,
                            "EntropyStrength": entropy_strength,
                            "MeanTemperature": float(np.mean(temperatures)),
                            "MedianTemperature": float(np.median(temperatures)),
                            **evaluate_metrics(y_true, probability),
                            **simulate_financials(scope_df, probability, config),
                        }
                    )
    return pd.DataFrame(rows)


def save_plots(results: pd.DataFrame) -> None:
    """Save plots for entropy-based temperature experiment.

    Args:
        results: Result dataframe.
    """
    sns.set_theme(style="whitegrid", context="talk")
    subset = results[(results["Scope"] == "2024+") & (results["Alpha"] == 0.50)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title, cmap, fmt in [
        (axes[0], "LogLoss", "LogLoss — niżej lepiej", "viridis_r", ".3f"),
        (axes[1], "UnitYield", "Unit Yield — wyżej lepiej", "YlGn", ".1f"),
    ]:
        pivot = subset.pivot(index="EntropyStrength", columns="BaseTemperature", values=metric).sort_index(ascending=False)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.85},
            annot_kws={"fontsize": 8},
        )
        ax.set_title(title, weight="bold")
        ax.set_xlabel("Base temperature")
        ax.set_ylabel("Entropy strength")
    fig.suptitle("Entropy-based temperature scaling — α=0.50, 2024+", weight="bold")
    fig.text(
        0.5,
        0.02,
        "Positive strength: high-entropy cases are flattened, low-entropy cases sharpened. Negative strength tests the reverse.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(ASSET_DIR / "hybrid_entropy_temperature_heatmap_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    best = results[results["Scope"] == "2024+"].sort_values("LogLoss").head(15).copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    labels = [
        f"α={row.Alpha:.2f}\nT0={row.BaseTemperature:.2f}\nγ={row.EntropyStrength:.2f}"
        for row in best.itertuples(index=False)
    ]
    sns.barplot(data=best, x=labels, y="LogLoss", hue="UnitYield", palette="viridis", dodge=False, ax=ax)
    ax.set_title("Najlepsze warianty entropy-based T według LogLoss 2024+", weight="bold")
    ax.set_xlabel("Konfiguracja")
    ax.set_ylabel("LogLoss")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Unit Yield %", loc="upper right")
    plt.tight_layout()
    fig.savefig(ASSET_DIR / "hybrid_entropy_temperature_top_configs_2024.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(results: pd.DataFrame) -> None:
    """Write generated appendix snippet.

    Args:
        results: Result dataframe.
    """
    output_path = PROJECT_ROOT / "docs" / "whitepaper" / "appendix" / "generated" / "07_entropy_temperature_autogenerated.md"
    best_logloss = results[results["Scope"] == "2024+"].sort_values("LogLoss").head(8)
    best_yield = results[results["Scope"] == "2024+"].sort_values("UnitYield", ascending=False).head(8)

    lines = [
        "---",
        "type: generated-experiment",
        "tags: [whitepaper, hybrid-model, entropy-temperature, calibration]",
        "project: inzynierka",
        "status: autogenerated",
        "---",
        "",
        "# Entropy-based temperature scaling — autogenerated",
        "",
        "> [!abstract]",
        "> Eksperyment sprawdza sample-wise temperature scaling, w którym temperatura zależy od entropii predykcji metamodelu.",
        "",
        "## Najlepsze warianty 2024+ według LogLoss",
        "",
        best_logloss.to_markdown(index=False),
        "",
        "## Najlepsze warianty 2024+ według Unit Yield",
        "",
        best_yield.to_markdown(index=False),
        "",
        "## Artefakty",
        "",
        "- `docs/assets/hybrid_point7/hybrid_entropy_temperature_results.csv`",
        "- `docs/assets/hybrid_point7/hybrid_entropy_temperature_heatmap_2024.png`",
        "- `docs/assets/hybrid_point7/hybrid_entropy_temperature_top_configs_2024.png`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run entropy-based temperature experiment."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.dropna(subset=["prob_model", "prob_market_open", "best_open_t1", "best_open_t2", "y_true"])
    df = df.sort_values("date").reset_index(drop=True)

    results = run_experiment(df)
    results.to_csv(ASSET_DIR / "hybrid_entropy_temperature_results.csv", index=False)
    save_plots(results)
    write_markdown(results)

    print("\n=== ENTROPY TEMPERATURE: BEST 2024+ BY LOGLOSS ===")
    cols = [
        "Alpha",
        "BaseTemperature",
        "EntropyStrength",
        "MeanTemperature",
        "AUC",
        "LogLoss",
        "ECE",
        "UnitYield",
        "KellyMaxDD",
        "Bets",
    ]
    print(results[results["Scope"] == "2024+"].sort_values("LogLoss")[cols].head(10).to_string(index=False))
    print(f"Saved: {ASSET_DIR / 'hybrid_entropy_temperature_results.csv'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_entropy_temperature_heatmap_2024.png'}")
    print(f"Saved: {ASSET_DIR / 'hybrid_entropy_temperature_top_configs_2024.png'}")


if __name__ == "__main__":
    main()
