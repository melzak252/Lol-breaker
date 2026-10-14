"""Generate static alpha/temperature candidate artifacts for case study chapter 6."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
ALPHA_CANDIDATES = [0.00, 0.30, 0.48, 0.62, 1.00]
TEMPERATURE_CANDIDATES = [0.60, 0.80, 1.00]


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to probability values.

    Args:
        probabilities: Probability vector.
        temperature: Temperature value.

    Returns:
        Temperature-scaled probability vector.
    """

    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    scaled = 1.0 / (1.0 + np.exp(-logits / temperature))
    return np.clip(scaled, 1e-6, 1 - 1e-6)


def simulate_fixed_stake(df: pd.DataFrame, probabilities: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    """Simulate a fixed stake teaser strategy.

    Args:
        df: Match-level data with opening odds and outcomes.
        probabilities: Probability of team 1 winning.

    Returns:
        Dictionary with bankroll and betting metrics.
    """

    bankroll = START_BANKROLL
    peak = START_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    profit = 0.0
    bets = 0
    wins = 0

    history: list[dict[str, float | str]] = []
    for row, probability in zip(df.itertuples(index=False), probabilities):
        if bankroll < FIXED_STAKE:
            history.append({"date": row.date, "bankroll": bankroll})
            continue
        ev_team1 = probability * float(row.best_open_t1) * TAX_FACTOR - 1.0
        ev_team2 = (1.0 - probability) * float(row.best_open_t2) * TAX_FACTOR - 1.0
        if max(ev_team1, ev_team2) <= EV_THRESHOLD:
            history.append({"date": row.date, "bankroll": bankroll})
            continue

        if ev_team1 >= ev_team2:
            odds = float(row.best_open_t1)
            won = int(row.t1_win) == 1
        else:
            odds = float(row.best_open_t2)
            won = int(row.t1_win) == 0

        pnl = FIXED_STAKE * (odds * TAX_FACTOR - 1.0) if won else -FIXED_STAKE
        bankroll += pnl
        total_staked += FIXED_STAKE
        profit += pnl
        bets += 1
        wins += int(won)
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak * 100.0)
        history.append({"date": row.date, "bankroll": bankroll})

    return {
        "final_bankroll": bankroll,
        "yield_pct": profit / total_staked * 100.0 if total_staked else 0.0,
        "maxdd_pct": max_drawdown,
        "bets": float(bets),
        "win_rate_pct": wins / bets * 100.0 if bets else 0.0,
    }, pd.DataFrame(history)


def load_2024_dataset() -> pd.DataFrame:
    """Load operational 2024+ hybrid input data.

    Returns:
        Prepared 2024+ DataFrame.
    """

    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    required = ["prob_model", "prob_market_open", "best_open_t1", "best_open_t2", "t1_win"]
    df = df.dropna(subset=required)
    return df[df["date"] >= pd.Timestamp("2024-01-01")].copy()


def build_candidate_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate fixed alpha and temperature candidates.

    Args:
        df: Prepared 2024+ data.

    Returns:
        Candidate grid DataFrame.
    """

    rows: list[dict[str, float | str]] = []
    target = df["t1_win"].astype(int).to_numpy()
    for alpha in ALPHA_CANDIDATES:
        for temperature in TEMPERATURE_CANDIDATES:
            if alpha == 0.0 and temperature != 1.0:
                continue
            model_probability = apply_temperature(df["prob_model"].to_numpy(), temperature)
            probability = (
                df["prob_market_open"].to_numpy()
                if alpha == 0.0
                else alpha * model_probability + (1 - alpha) * df["prob_market_open"].to_numpy()
            )
            metrics, _ = simulate_fixed_stake(df, probability)
            rows.append(
                {
                    "alpha": alpha,
                    "temperature": temperature,
                    "variant": f"alpha={alpha:.2f} T={temperature:.2f}",
                    "auc": roc_auc_score(target, probability),
                    "logloss": log_loss(target, probability),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def save_summary_plot(selected: pd.DataFrame) -> None:
    """Save a two-panel candidate summary plot.

    Args:
        selected: Selected candidate DataFrame.
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#6B7280", "#60A5FA", "#10B981", "#F59E0B", "#EF4444"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    axes[0].bar(selected["name"], selected["logloss"], color=colors)
    axes[0].set_title("Jakość probabilistyczna kandydatów", pad=18, weight="bold")
    axes[0].set_ylabel("LogLoss (niżej lepiej)")
    axes[0].set_ylim(selected["logloss"].min() - 0.005, selected["logloss"].max() + 0.022)
    axes[0].tick_params(axis="x", rotation=18)
    for index, value in enumerate(selected["logloss"]):
        axes[0].text(index, value + 0.002, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(selected["name"], selected["yield_pct"], color=colors)
    axes[1].set_title("Fixed stake teaser: Yield kandydatów", pad=18, weight="bold")
    axes[1].set_ylabel("Yield (%)")
    axes[1].set_ylim(
        min(0.0, float(selected["yield_pct"].min()) - 10.0),
        float(selected["yield_pct"].max()) + 28.0,
    )
    axes[1].tick_params(axis="x", rotation=18)
    for index, row in enumerate(selected.itertuples(index=False)):
        offset = 3.0 if row.yield_pct >= 0 else -8.0
        va = "bottom" if row.yield_pct >= 0 else "top"
        axes[1].text(
            index,
            row.yield_pct + offset,
            f"{row.yield_pct:.1f}%\n{int(row.bets)} bets",
            ha="center",
            va=va,
            fontsize=9,
        )

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.2, w_pad=2.4)
    fig.savefig(OUTPUT_DIR / "hybrid_static_candidate_temperature_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_bankroll_plot(df: pd.DataFrame, selected: pd.DataFrame) -> None:
    """Save bankroll over time for selected candidates.

    Args:
        df: Prepared 2024+ data.
        selected: Selected candidate metadata.
    """

    label_map = {
        (0.00, 1.00): "Market Avg Open",
        (0.30, 0.60): "Hybrid Defensive α=0.30 T=0.60",
        (0.48, 0.60): "Hybrid Financial α=0.48 T=0.60",
        (0.62, 0.80): "Hybrid Probabilistic α=0.62 T=0.80",
        (1.00, 1.00): "Model Only",
    }
    colors = {
        "Market Avg Open": "#6B7280",
        "Hybrid Defensive α=0.30 T=0.60": "#60A5FA",
        "Hybrid Financial α=0.48 T=0.60": "#10B981",
        "Hybrid Probabilistic α=0.62 T=0.80": "#F59E0B",
        "Model Only": "#EF4444",
    }
    histories: list[pd.DataFrame] = []
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axis = plt.subplots(figsize=(12, 6))
    for row in selected.itertuples(index=False):
        model_probability = apply_temperature(df["prob_model"].to_numpy(), float(row.temperature))
        probability = (
            df["prob_market_open"].to_numpy()
            if float(row.alpha) == 0.0
            else float(row.alpha) * model_probability + (1 - float(row.alpha)) * df["prob_market_open"].to_numpy()
        )
        _, history = simulate_fixed_stake(df, probability)
        label = label_map[(float(row.alpha), float(row.temperature))]
        history["variant"] = label
        histories.append(history)
        axis.plot(pd.to_datetime(history["date"]), history["bankroll"], label=label, linewidth=2.2, color=colors[label])

    pd.concat(histories, ignore_index=True).to_csv(OUTPUT_DIR / "hybrid_static_candidate_bankroll_history_2024.csv", index=False)
    axis.set_title("Bankroll teaser dla stałych kandydatów alpha/T")
    axis.set_ylabel("Bankroll")
    axis.set_xlabel("Data")
    axis.legend(loc="upper left", frameon=True, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hybrid_static_candidate_bankroll_2024.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate candidate CSV files and summary figure."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_2024_dataset()
    grid = build_candidate_grid(df)
    grid.to_csv(OUTPUT_DIR / "hybrid_static_candidate_temperature_grid_2024.csv", index=False)

    names = {
        (0.00, 1.00): "Market Avg Open",
        (0.30, 0.60): "Hybrid Defensive\nα=0.30 T=0.60",
        (0.48, 0.60): "Hybrid Financial\nα=0.48 T=0.60",
        (0.62, 0.80): "Hybrid Probabilistic\nα=0.62 T=0.80",
        (1.00, 1.00): "Model Only",
    }
    selected = grid[grid.apply(lambda row: (row["alpha"], row["temperature"]) in names, axis=1)].copy()
    selected["name"] = selected.apply(lambda row: names[(row["alpha"], row["temperature"])], axis=1)
    selected.to_csv(OUTPUT_DIR / "hybrid_static_candidate_selected_2024.csv", index=False)
    save_summary_plot(selected)
    save_bankroll_plot(df, selected)
    printable = selected[["name", "alpha", "temperature", "auc", "logloss", "final_bankroll", "yield_pct", "maxdd_pct", "bets"]].copy()
    printable["name"] = printable["name"].str.replace("α", "alpha", regex=False)
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
