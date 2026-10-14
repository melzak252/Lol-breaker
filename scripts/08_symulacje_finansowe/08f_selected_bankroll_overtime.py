"""Plot bankroll over time for selected ROI/MaxDD trade-off variants.

The script uses the dense financial setup based on the symmetrized and Platt-
calibrated W20-Binomial probability. It focuses on a small set of configurations
chosen from the ROI--MaxDD frontier rather than on every tested alpha/T/Kelly
combination.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.thesis_style import apply_thesis_style, clean_axis, palette


OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_point8"
DENSE_GRID_SCRIPT = Path(__file__).with_name("08e_dense_alpha_kelly_temperature_grid.py")


def load_dense_grid_module() -> object:
    """Load dense-grid helper module from its numeric filename."""

    spec = importlib.util.spec_from_file_location("dense_financial_grid", DENSE_GRID_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load dense-grid script from {DENSE_GRID_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dense_grid = load_dense_grid_module()


@dataclass(frozen=True)
class SelectedVariant:
    """Selected financial configuration from the ROI--MaxDD frontier."""

    label: str
    alpha: float
    temperature: float
    kelly: float


SELECTED_VARIANTS = [
    SelectedVariant("Konserwatywny: a=0.30, T=0.60, K=0.10", 0.30, 0.60, 0.10),
    SelectedVariant("Sweet spot: a=0.30, T=0.60, K=0.15", 0.30, 0.60, 0.15),
    SelectedVariant("Agresywniejszy: a=0.25, T=0.50, K=0.20", 0.25, 0.50, 0.20),
    SelectedVariant("Limit 20% DD: a=0.25, T=0.50, K=0.25", 0.25, 0.50, 0.25),
    SelectedVariant("Maks ROI <25% DD: a=0.45, T=0.40, K=0.25", 0.45, 0.40, 0.25),
]


def simulate_bankroll_history(data: pd.DataFrame, variant: SelectedVariant) -> pd.DataFrame:
    """Simulate bankroll history for one selected configuration.

    Args:
        data: Chronologically sorted 2024+ financial dataset.
        variant: Selected alpha, temperature and Kelly configuration.

    Returns:
        Per-match bankroll history with bet and drawdown diagnostics.
    """

    y_true = data["y_true"].astype(int).to_numpy()
    market = data["market_open"].to_numpy(dtype=float)
    model = data["prob_sym_platt"].to_numpy(dtype=float)
    odds_t1 = data["best_open_t1"].to_numpy(dtype=float)
    odds_t2 = data["best_open_t2"].to_numpy(dtype=float)

    model_temp = dense_grid.apply_temperature(model, variant.temperature)
    probabilities = variant.alpha * model_temp + (1.0 - variant.alpha) * market

    bankroll = dense_grid.INITIAL_BANKROLL
    peak = dense_grid.INITIAL_BANKROLL
    total_staked = 0.0
    total_profit = 0.0
    bets = 0
    wins = 0
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for idx, probability_t1 in enumerate(probabilities):
        probability_t2 = 1.0 - float(probability_t1)
        ev_t1 = float(probability_t1) * odds_t1[idx] * (1.0 - dense_grid.TAX_RATE) - 1.0
        ev_t2 = probability_t2 * odds_t2[idx] * (1.0 - dense_grid.TAX_RATE) - 1.0

        side = "none"
        stake = 0.0
        profit = 0.0
        is_win = False

        if ev_t1 > ev_t2 and ev_t1 > dense_grid.EV_THRESHOLD:
            side = "t1"
            selected_probability = float(probability_t1)
            raw_odds = odds_t1[idx]
            is_win = bool(y_true[idx] == 1)
        elif ev_t2 > dense_grid.EV_THRESHOLD:
            side = "t2"
            selected_probability = probability_t2
            raw_odds = odds_t2[idx]
            is_win = bool(y_true[idx] == 0)
        else:
            selected_probability = 0.0
            raw_odds = 0.0

        if side != "none" and bankroll > 0:
            execution_odds = raw_odds * (1.0 - dense_grid.SLIPPAGE)
            net_decimal = execution_odds * (1.0 - dense_grid.TAX_RATE)
            b_value = net_decimal - 1.0
            if b_value > 0:
                full_kelly = ((b_value * selected_probability) - (1.0 - selected_probability)) / b_value
                if full_kelly > 0:
                    stake = bankroll * full_kelly * variant.kelly
                    stake = min(max(stake, dense_grid.MIN_STAKE), dense_grid.MAX_STAKE)
                    if stake <= bankroll:
                        if is_win:
                            profit = stake * (net_decimal - 1.0)
                            wins += 1
                        else:
                            profit = -stake
                        bankroll += profit
                        total_staked += stake
                        total_profit += profit
                        bets += 1
                    else:
                        stake = 0.0
                        profit = 0.0
                        side = "none"

        peak = max(peak, bankroll)
        drawdown_pct = ((peak - bankroll) / peak * 100.0) if peak > 0 else 0.0
        rows.append(
            {
                "date": data.iloc[idx]["date"],
                "golgg_match_id": data.iloc[idx]["golgg_match_id"],
                "variant": variant.label,
                "alpha": variant.alpha,
                "temperature": variant.temperature,
                "kelly": variant.kelly,
                "bankroll": bankroll,
                "peak": peak,
                "drawdown_pct": drawdown_pct,
                "side": side,
                "stake": stake,
                "profit": profit,
                "bets": bets,
                "wins": wins,
                "total_staked": total_staked,
                "total_profit": total_profit,
            }
        )

    return pd.DataFrame(rows)


def summarize_history(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize final bankroll, ROI, yield and MaxDD for each variant."""

    rows = []
    for variant, group in history.groupby("variant", sort=False):
        last = group.iloc[-1]
        rows.append(
            {
                "variant": variant,
                "alpha": float(last["alpha"]),
                "temperature": float(last["temperature"]),
                "kelly": float(last["kelly"]),
                "final_bankroll": float(last["bankroll"]),
                "roi_pct": (float(last["bankroll"]) / dense_grid.INITIAL_BANKROLL - 1.0) * 100.0,
                "yield_pct": (
                    float(last["total_profit"]) / float(last["total_staked"]) * 100.0
                    if float(last["total_staked"]) > 0
                    else 0.0
                ),
                "max_drawdown_pct": float(group["drawdown_pct"].max()),
                "bets": int(last["bets"]),
                "win_rate_pct": int(last["wins"]) / int(last["bets"]) * 100.0 if int(last["bets"]) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_bankroll(history: pd.DataFrame) -> None:
    """Save linear and logarithmic bankroll-over-time plots."""

    apply_thesis_style(context="paper")
    colors = palette(len(SELECTED_VARIANTS))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for color, (variant, group) in zip(colors, history.groupby("variant", sort=False), strict=False):
        ax.plot(group["date"], group["bankroll"], label=variant, color=color, linewidth=2.0)
    ax.set_title("Bankroll w czasie dla wybranych wariantów ROI--MaxDD")
    ax.set_xlabel("Data")
    ax.set_ylabel("Bankroll")
    ax.legend(fontsize=8, loc="upper left")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_selected_bankroll_overtime_linear.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for color, (variant, group) in zip(colors, history.groupby("variant", sort=False), strict=False):
        ax.plot(group["date"], group["bankroll"], label=variant, color=color, linewidth=2.0)
    ax.set_title("Bankroll w czasie dla wybranych wariantów ROI--MaxDD (skala log)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Bankroll, skala logarytmiczna")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_selected_bankroll_overtime_log.png")
    plt.close(fig)


def main() -> None:
    """Generate bankroll histories and plots for selected configurations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = dense_grid.load_dataset().sort_values("date").reset_index(drop=True)

    histories = [simulate_bankroll_history(data, variant) for variant in SELECTED_VARIANTS]
    history = pd.concat(histories, ignore_index=True)
    summary = summarize_history(history)

    history.to_csv(OUTPUT_DIR / "financial_selected_bankroll_overtime.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "financial_selected_bankroll_overtime_summary.csv", index=False)
    plot_bankroll(history)

    print("Selected bankroll-over-time summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
