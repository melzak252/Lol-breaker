"""Run a dense alpha × Kelly × temperature financial grid.

This diagnostic extends the coarse financial backtest by scanning alpha in
[0, 1], Kelly fraction in [0, 1], and temperature in [0, 2]. Temperature equal
to zero is implemented as a near-deterministic limiting case of temperature
scaling.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.thesis_style import apply_thesis_style, clean_axis, palette


FINANCIAL_SUITE = PROJECT_ROOT / "scripts" / "08_symulacje_finansowe" / "08_financial_validation_suite.py"
CALIBRATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "calibration_symmetry_diagnostic"
    / "calibration_symmetry_predictions.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_point8"

ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
KELLY_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TEMPERATURE_GRID = np.round(np.concatenate([[0.0], np.arange(0.1, 2.0001, 0.1)]), 2)
INITIAL_BANKROLL = 100.0
TAX_RATE = 0.12
SLIPPAGE = 0.01
EV_THRESHOLD = 0.05
MIN_STAKE = 2.0
MAX_STAKE = 100.0


def load_financial_suite() -> object:
    """Load current financial validation helper module."""

    spec = importlib.util.spec_from_file_location("financial_suite", FINANCIAL_SUITE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load financial suite from {FINANCIAL_SUITE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_dataset() -> pd.DataFrame:
    """Load 2024+ common sample with symmetrized Platt predictions."""

    suite = load_financial_suite()
    data = suite.load_dataset()
    data["golgg_match_id"] = data["golgg_match_id"].astype(str)

    calibrated = pd.read_csv(CALIBRATION_PATH)
    calibrated = calibrated[
        (calibrated["base_variant"] == "Order-symmetrized prediction")
        & (calibrated["calibration"] == "platt_expanding")
    ].copy()
    calibrated["golgg_match_id"] = calibrated["golgg_match_id"].astype(str)
    calibrated = calibrated[["golgg_match_id", "y_prob"]].rename(columns={"y_prob": "prob_sym_platt"})

    merged = data.merge(calibrated, on="golgg_match_id", how="inner")
    return merged[merged["date"] >= pd.Timestamp("2024-01-01")].copy().reset_index(drop=True)


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probabilities.

    Args:
        probability: Base probabilities.
        temperature: Temperature value. A value of zero is treated as the
            deterministic limiting case.

    Returns:
        Temperature-scaled probabilities.
    """

    clipped = np.clip(probability, 0.001, 0.999)
    if temperature <= 0.0:
        return np.where(clipped >= 0.5, 0.999, 0.001)
    logits = np.log(clipped / (1.0 - clipped))
    scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
    return np.clip(scaled, 0.001, 0.999)


def simulate_arrays(
    probabilities: np.ndarray,
    kelly_fraction: float,
    y_true: np.ndarray,
    odds_t1: np.ndarray,
    odds_t2: np.ndarray,
    close_t1: np.ndarray,
    close_t2: np.ndarray,
) -> dict[str, float | int]:
    """Simulate one Kelly strategy using NumPy arrays."""

    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    total_profit = 0.0
    bets = 0
    wins = 0
    clv_values: list[float] = []

    if kelly_fraction <= 0.0:
        return {
            "final_bankroll": bankroll,
            "roi_pct": 0.0,
            "yield_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_staked": 0.0,
            "bets": 0,
            "win_rate_pct": 0.0,
            "avg_clv_pct": 0.0,
        }

    for idx, prob_t1 in enumerate(probabilities):
        prob_t2 = 1.0 - prob_t1
        ev_t1 = prob_t1 * odds_t1[idx] * (1.0 - TAX_RATE) - 1.0
        ev_t2 = prob_t2 * odds_t2[idx] * (1.0 - TAX_RATE) - 1.0
        if ev_t1 > ev_t2 and ev_t1 > EV_THRESHOLD:
            side = 1
            selected_prob = prob_t1
            raw_odds = odds_t1[idx]
            close_odds = close_t1[idx]
            is_win = y_true[idx] == 1
        elif ev_t2 > EV_THRESHOLD:
            side = 2
            selected_prob = prob_t2
            raw_odds = odds_t2[idx]
            close_odds = close_t2[idx]
            is_win = y_true[idx] == 0
        else:
            side = 0

        if side == 0 or bankroll <= 0:
            peak = max(peak, bankroll)
            continue

        execution_odds = raw_odds * (1.0 - SLIPPAGE)
        net_decimal = execution_odds * (1.0 - TAX_RATE)
        b_value = net_decimal - 1.0
        if b_value <= 0:
            continue
        full_kelly = ((b_value * selected_prob) - (1.0 - selected_prob)) / b_value
        if full_kelly <= 0:
            continue
        stake = bankroll * full_kelly * kelly_fraction
        stake = min(max(stake, MIN_STAKE), MAX_STAKE)
        if stake > bankroll:
            continue

        if is_win:
            profit = stake * (net_decimal - 1.0)
            wins += 1
        else:
            profit = -stake
        bankroll += profit
        total_staked += stake
        total_profit += profit
        bets += 1
        if close_odds > 0:
            clv_values.append((raw_odds - close_odds) / close_odds)

        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)

    return {
        "final_bankroll": bankroll,
        "roi_pct": (bankroll / INITIAL_BANKROLL - 1.0) * 100.0,
        "yield_pct": (total_profit / total_staked * 100.0) if total_staked else 0.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "total_staked": total_staked,
        "bets": bets,
        "win_rate_pct": (wins / bets * 100.0) if bets else 0.0,
        "avg_clv_pct": (float(np.mean(clv_values)) * 100.0) if clv_values else 0.0,
    }


def run_grid(data: pd.DataFrame) -> pd.DataFrame:
    """Run dense alpha × Kelly × temperature grid."""

    y_true = data["y_true"].astype(int).to_numpy()
    market = data["market_open"].to_numpy(dtype=float)
    model = data["prob_sym_platt"].to_numpy(dtype=float)
    odds_t1 = data["best_open_t1"].to_numpy(dtype=float)
    odds_t2 = data["best_open_t2"].to_numpy(dtype=float)
    close_t1 = data["best_close_t1"].fillna(0.0).to_numpy(dtype=float)
    close_t2 = data["best_close_t2"].fillna(0.0).to_numpy(dtype=float)

    rows: list[dict[str, float | int]] = []
    for temperature in TEMPERATURE_GRID:
        model_temp = apply_temperature(model, float(temperature))
        for alpha in ALPHA_GRID:
            hybrid_prob = alpha * model_temp + (1.0 - alpha) * market
            clipped_prob = np.clip(hybrid_prob, 1e-6, 1.0 - 1e-6)
            logloss_value = log_loss(y_true, clipped_prob)
            for kelly in KELLY_GRID:
                result = simulate_arrays(
                    clipped_prob,
                    float(kelly),
                    y_true,
                    odds_t1,
                    odds_t2,
                    close_t1,
                    close_t2,
                )
                result.update(
                    {
                        "alpha": float(alpha),
                        "kelly": float(kelly),
                        "temperature": float(temperature),
                        "logloss": logloss_value,
                    }
                )
                rows.append(result)
    summary = pd.DataFrame(rows)
    summary["roi_to_maxdd"] = summary["roi_pct"] / summary["max_drawdown_pct"].clip(lower=1e-6)
    return summary


def save_roi_curves(summary: pd.DataFrame) -> None:
    """Save dense ROI sensitivity curves."""

    apply_thesis_style(context="paper")

    alpha_plot = summary[(summary["kelly"] == 0.05) & (summary["temperature"].isin([0.5, 0.8, 1.0, 1.5, 2.0]))]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.lineplot(data=alpha_plot, x="alpha", y="roi_pct", hue="temperature", marker="o", palette=palette(5), ax=ax)
    ax.set_title("ROI względem alfa (Kelly=0.05)")
    ax.set_xlabel("Alfa modelu")
    ax.set_ylabel("ROI [%]")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_dense_roi_by_alpha.png")
    plt.close(fig)

    kelly_plot = summary[(summary["temperature"] == 0.8) & (summary["alpha"].isin([0.2, 0.4, 0.6, 0.8, 1.0]))]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.lineplot(data=kelly_plot, x="kelly", y="roi_pct", hue="alpha", marker="o", palette=palette(5), ax=ax)
    ax.set_title("ROI względem współczynnika Kelly'ego (T=0.80)")
    ax.set_xlabel("Współczynnik Kelly'ego")
    ax.set_ylabel("ROI [%]")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_dense_roi_by_kelly.png")
    plt.close(fig)

    temperature_plot = summary[(summary["kelly"] == 0.05) & (summary["alpha"].isin([0.2, 0.4, 0.6, 0.8, 1.0]))]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.lineplot(
        data=temperature_plot,
        x="temperature",
        y="roi_pct",
        hue="alpha",
        marker="o",
        palette=palette(5),
        ax=ax,
    )
    ax.set_title("ROI względem temperatury T (Kelly=0.05)")
    ax.set_xlabel("Temperatura T")
    ax.set_ylabel("ROI [%]")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_dense_roi_by_temperature.png")
    plt.close(fig)


def main() -> None:
    """Run dense grid and save results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    summary = run_grid(data)
    summary.to_csv(OUTPUT_DIR / "financial_dense_alpha_kelly_temperature_grid.csv", index=False)
    save_roi_curves(summary)

    top_roi = summary.sort_values("roi_pct", ascending=False).head(20)
    top_risk = summary[(summary["bets"] >= 100) & (summary["max_drawdown_pct"] <= 25)].sort_values(
        "roi_to_maxdd", ascending=False
    ).head(20)
    top_roi.to_csv(OUTPUT_DIR / "financial_dense_top_roi.csv", index=False)
    top_risk.to_csv(OUTPUT_DIR / "financial_dense_top_risk_controlled.csv", index=False)

    print("Top ROI configurations:")
    print(
        top_roi[
            ["alpha", "temperature", "kelly", "roi_pct", "yield_pct", "max_drawdown_pct", "bets", "logloss"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nBest risk-controlled configurations (MaxDD <= 25%):")
    print(
        top_risk[
            ["alpha", "temperature", "kelly", "roi_pct", "yield_pct", "max_drawdown_pct", "bets", "logloss"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved dense grid outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
