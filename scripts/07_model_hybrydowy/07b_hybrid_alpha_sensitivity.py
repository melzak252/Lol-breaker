import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score


BOOKMAKERS = ["betclic", "betfan", "efortuna", "lv_bet", "sts", "superbet"]


@dataclass(frozen=True)
class FixedStakeConfig:
    """Configuration for fixed-stake bankroll simulations."""

    initial_bankroll: float = 100.0
    fixed_stake: float = 10.0
    tax_rate: float = 0.12
    slippage: float = 0.01
    ev_threshold: float = 0.05


def add_best_close_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Add best closing odds and corresponding bookmaker names.

    Args:
        df: Match-level dataframe with bookmaker closing odds columns.

    Returns:
        Dataframe enriched with best closing odds for both teams.
    """
    result = df.copy()
    odds1_cols = [f"odds1_{bookmaker}_close" for bookmaker in BOOKMAKERS]
    odds2_cols = [f"odds2_{bookmaker}_close" for bookmaker in BOOKMAKERS]

    result["max_close_t1"] = result[odds1_cols].max(axis=1)
    result["max_close_t2"] = result[odds2_cols].max(axis=1)
    result["best_bookie_t1_close"] = (
        result[odds1_cols]
        .idxmax(axis=1)
        .str.replace("odds1_", "", regex=False)
        .str.replace("_close", "", regex=False)
        .str.upper()
    )
    result["best_bookie_t2_close"] = (
        result[odds2_cols]
        .idxmax(axis=1)
        .str.replace("odds2_", "", regex=False)
        .str.replace("_close", "", regex=False)
        .str.upper()
    )
    return result


def simulate_fixed_stake(
    df: pd.DataFrame,
    probability_column: str,
    config: FixedStakeConfig,
) -> tuple[dict[str, float | int], list[float]]:
    """Simulate fixed-stake EV betting for one probability column.

    Args:
        df: Chronologically sorted dataframe.
        probability_column: Column containing T1 win probability.
        config: Fixed-stake simulation parameters.

    Returns:
        Summary metrics and full bankroll history aligned to dataframe rows.
    """
    bankroll = config.initial_bankroll
    bankroll_history = [bankroll]
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    bets = 0
    net_multiplier = 1 - config.tax_rate

    for _, row in df.iterrows():
        p_t1 = row[probability_column]
        if pd.isnull(p_t1):
            bankroll_history.append(bankroll)
            continue

        ev_t1 = row["max_close_t1"] * net_multiplier * p_t1 - 1
        ev_t2 = row["max_close_t2"] * net_multiplier * (1 - p_t1) - 1

        if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
            raw_odds = row["max_close_t1"]
            is_win = int(row["y_true"] == 1)
        elif ev_t2 > config.ev_threshold:
            raw_odds = row["max_close_t2"]
            is_win = int(row["y_true"] == 0)
        else:
            bankroll_history.append(bankroll)
            continue

        stake = config.fixed_stake
        if bankroll < stake:
            bankroll_history.append(bankroll)
            continue

        execution_odds = max(1.01, raw_odds * (1 - config.slippage))
        profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
        bankroll += profit
        total_staked += stake
        total_profit += profit
        wins += is_win
        bets += 1
        bankroll_history.append(bankroll)

    bankroll_array = np.array(bankroll_history)
    peaks = np.maximum.accumulate(bankroll_array)
    drawdowns = (peaks - bankroll_array) / (peaks + 1e-9)

    summary = {
        "Final Bankroll": bankroll,
        "Total Profit": total_profit,
        "ROI (%)": (bankroll - config.initial_bankroll) / config.initial_bankroll * 100,
        "Yield (%)": total_profit / total_staked * 100 if total_staked > 0 else 0.0,
        "Max Drawdown (%)": float(np.max(drawdowns) * 100),
        "Min Bankroll": float(np.min(bankroll_array)),
        "Total Staked": total_staked,
        "Bets": bets,
        "Win Rate (%)": wins / bets * 100 if bets > 0 else 0.0,
    }
    return summary, bankroll_history


def main() -> None:
    """Run close-odds alpha sensitivity analysis for hybrid blending in 2024+."""
    print("Running 07b_hybrid_alpha_sensitivity.py...")
    os.makedirs("docs/assets", exist_ok=True)

    df = pd.read_csv("data/golgg_final_hybrid_results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = add_best_close_odds(df)
    df["prob_avg_close"] = 1 / df["avg_odds_home"] / (
        (1 / df["avg_odds_home"]) + (1 / df["avg_odds_away"])
    )
    df = df[df["date"] >= datetime(2024, 1, 1)].sort_values("date").reset_index(drop=True)
    df = df.dropna(
        subset=[
            "metamodel_lgbm_calibrated",
            "prob_avg_close",
            "y_true",
            "max_close_t1",
            "max_close_t2",
        ]
    ).copy()

    config = FixedStakeConfig()
    alpha_rows = []
    histories: dict[float, list[float]] = {}

    for alpha in np.round(np.linspace(0.0, 1.0, 21), 2):
        column = f"hybrid_alpha_{alpha:.2f}"
        df[column] = (
            alpha * df["metamodel_lgbm_calibrated"]
            + (1 - alpha) * df["prob_avg_close"]
        ).clip(0.01, 0.99)

        y_true = df["y_true"].to_numpy()
        y_prob = df[column].to_numpy()
        summary, history = simulate_fixed_stake(df, column, config)
        histories[float(alpha)] = history
        alpha_rows.append(
            {
                "Alpha": float(alpha),
                "AUC": roc_auc_score(y_true, y_prob),
                "LogLoss": log_loss(y_true, y_prob),
                **summary,
            }
        )

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv("docs/assets/hybrid_alpha_sensitivity_2024_close_fixed_stake.csv", index=False)

    best_roi = alpha_df.loc[alpha_df["ROI (%)"].idxmax()]
    best_yield = alpha_df.loc[alpha_df["Yield (%)"].idxmax()]
    best_logloss = alpha_df.loc[alpha_df["LogLoss"].idxmin()]
    best_auc = alpha_df.loc[alpha_df["AUC"].idxmax()]
    best_dd = alpha_df.loc[alpha_df["Max Drawdown (%)"].idxmin()]

    print("\n=== ALPHA SENSITIVITY 2024+ CLOSE ODDS FIXED STAKE ===")
    print(alpha_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== BEST ALPHAS ===")
    print(f"Best ROI alpha:     {best_roi['Alpha']:.2f} | ROI={best_roi['ROI (%)']:.2f}% | DD={best_roi['Max Drawdown (%)']:.2f}%")
    print(f"Best Yield alpha:   {best_yield['Alpha']:.2f} | Yield={best_yield['Yield (%)']:.2f}% | Bets={int(best_yield['Bets'])}")
    print(f"Best LogLoss alpha: {best_logloss['Alpha']:.2f} | LogLoss={best_logloss['LogLoss']:.4f} | ROI={best_logloss['ROI (%)']:.2f}%")
    print(f"Best AUC alpha:     {best_auc['Alpha']:.2f} | AUC={best_auc['AUC']:.4f} | ROI={best_auc['ROI (%)']:.2f}%")
    print(f"Best DD alpha:      {best_dd['Alpha']:.2f} | DD={best_dd['Max Drawdown (%)']:.2f}% | ROI={best_dd['ROI (%)']:.2f}%")

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.lineplot(data=alpha_df, x="Alpha", y="ROI (%)", marker="o", ax=axes[0, 0], color="green")
    axes[0, 0].set_title("Alpha vs ROI (Close Odds, Fixed Stake, 2024+)")
    axes[0, 0].axvline(best_roi["Alpha"], color="green", linestyle="--", alpha=0.6)

    sns.lineplot(data=alpha_df, x="Alpha", y="Yield (%)", marker="o", ax=axes[0, 1], color="blue")
    axes[0, 1].set_title("Alpha vs Yield")
    axes[0, 1].axvline(best_yield["Alpha"], color="blue", linestyle="--", alpha=0.6)

    sns.lineplot(data=alpha_df, x="Alpha", y="Max Drawdown (%)", marker="o", ax=axes[1, 0], color="red")
    axes[1, 0].set_title("Alpha vs Max Drawdown")
    axes[1, 0].axvline(best_dd["Alpha"], color="red", linestyle="--", alpha=0.6)

    ax_auc = axes[1, 1]
    ax_loss = ax_auc.twinx()
    sns.lineplot(data=alpha_df, x="Alpha", y="AUC", marker="o", ax=ax_auc, color="purple", label="AUC")
    sns.lineplot(data=alpha_df, x="Alpha", y="LogLoss", marker="s", ax=ax_loss, color="orange", label="LogLoss")
    ax_auc.set_title("Alpha vs Statistical Metrics")
    ax_auc.axvline(best_auc["Alpha"], color="purple", linestyle="--", alpha=0.4)
    ax_loss.axvline(best_logloss["Alpha"], color="orange", linestyle="--", alpha=0.4)
    ax_auc.legend(loc="upper left")
    ax_loss.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("docs/assets/hybrid_alpha_sensitivity_2024_close.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        alpha_df["LogLoss"],
        alpha_df["ROI (%)"],
        c=alpha_df["Max Drawdown (%)"],
        s=80 + alpha_df["Bets"] / 8,
        cmap="coolwarm",
        alpha=0.85,
    )
    for _, row in alpha_df.iterrows():
        plt.text(row["LogLoss"], row["ROI (%)"], f"α={row['Alpha']:.2f}", fontsize=8)
    plt.colorbar(scatter, label="Max Drawdown (%)")
    plt.title("Statistical Quality vs Financial Return (Close Odds, Fixed Stake, 2024+)")
    plt.xlabel("LogLoss (lower is better)")
    plt.ylabel("ROI (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/hybrid_alpha_roi_vs_logloss_2024_close.png", dpi=300, bbox_inches="tight")

    selected_alphas = sorted(
        {
            0.0,
            0.5,
            1.0,
            float(best_roi["Alpha"]),
            float(best_logloss["Alpha"]),
            float(best_dd["Alpha"]),
        }
    )

    plt.figure(figsize=(12, 7))
    for alpha in selected_alphas:
        row = alpha_df[alpha_df["Alpha"] == alpha].iloc[0]
        label = f"α={alpha:.2f} | ROI={row['ROI (%)']:.0f}% | DD={row['Max Drawdown (%)']:.0f}%"
        plt.plot(df["date"], histories[alpha][1:], label=label, linewidth=2 if alpha == float(best_roi["Alpha"]) else 1.5)
    plt.title("Bankroll Over Time by Alpha (Close Odds, Fixed Stake 10$, 2024+)")
    plt.xlabel("Date")
    plt.ylabel("Bankroll ($)")
    plt.axhline(config.initial_bankroll, color="black", linestyle="--", alpha=0.6)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/hybrid_alpha_bankroll_fixed_stake_2024_close.png", dpi=300, bbox_inches="tight")

    print("\nSaved alpha sensitivity artifacts to docs/assets/.")
    print("07b_hybrid_alpha_sensitivity.py completed successfully.")


if __name__ == "__main__":
    main()
