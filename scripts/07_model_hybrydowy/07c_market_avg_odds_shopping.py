import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BOOKMAKERS = ["betclic", "betfan", "efortuna", "lv_bet", "sts", "superbet"]


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for Market Avg + max odds-shopping simulations."""

    initial_bankroll: float = 100.0
    fixed_stake: float = 10.0
    tax_rate: float = 0.12
    slippage: float = 0.0
    ev_threshold: float = 0.05
    allow_negative_bankroll: bool = False


def implied_prob(o1: pd.Series, o2: pd.Series) -> pd.Series:
    """Calculate margin-normalized implied probability for team 1.

    Args:
        o1: Decimal odds for team 1.
        o2: Decimal odds for team 2.

    Returns:
        Margin-normalized probability of team 1 winning.
    """
    return (1 / o1) / ((1 / o1) + (1 / o2))


def prepare_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add market probabilities and best odds columns.

    Args:
        df: Raw odds dataframe.

    Returns:
        Enriched dataframe with open/close probabilities and max odds.
    """
    result = df.copy()
    result["date"] = pd.to_datetime(result["odds_date"])
    result["prob_avg_open"] = implied_prob(result["avg_open_home"], result["avg_open_away"])
    result["prob_avg_close"] = implied_prob(result["avg_odds_home"], result["avg_odds_away"])

    for stage in ["open", "close"]:
        odds1_cols = [f"odds1_{bookmaker}_{stage}" for bookmaker in BOOKMAKERS]
        odds2_cols = [f"odds2_{bookmaker}_{stage}" for bookmaker in BOOKMAKERS]
        result[f"max_{stage}_t1"] = result[odds1_cols].max(axis=1)
        result[f"max_{stage}_t2"] = result[odds2_cols].max(axis=1)
        odds1_for_idx = result[odds1_cols].fillna(-np.inf)
        odds2_for_idx = result[odds2_cols].fillna(-np.inf)
        result[f"best_{stage}_bookie_t1"] = (
            odds1_for_idx
            .idxmax(axis=1)
            .str.replace("odds1_", "", regex=False)
            .str.replace(f"_{stage}", "", regex=False)
            .str.upper()
        )
        result[f"best_{stage}_bookie_t2"] = (
            odds2_for_idx
            .idxmax(axis=1)
            .str.replace("odds2_", "", regex=False)
            .str.replace(f"_{stage}", "", regex=False)
            .str.upper()
        )
        result.loc[result[f"max_{stage}_t1"].isna(), f"best_{stage}_bookie_t1"] = np.nan
        result.loc[result[f"max_{stage}_t2"].isna(), f"best_{stage}_bookie_t2"] = np.nan

    return result


def simulate_market_strategy(
    df: pd.DataFrame,
    probability_stage: str,
    execution_stage: str,
    config: SimulationConfig,
) -> tuple[dict[str, float | int | str], pd.DataFrame, list[float]]:
    """Simulate Market Avg probability with max odds shopping.

    Args:
        df: Chronologically sorted dataframe.
        probability_stage: ``open`` or ``close`` for Market Avg probability.
        execution_stage: ``open`` or ``close`` for max odds execution.
        config: Simulation configuration.

    Returns:
        Summary metrics, bet log, and bankroll history.
    """
    prob_col = f"prob_avg_{probability_stage}"
    max_t1_col = f"max_{execution_stage}_t1"
    max_t2_col = f"max_{execution_stage}_t2"
    book_t1_col = f"best_{execution_stage}_bookie_t1"
    book_t2_col = f"best_{execution_stage}_bookie_t2"

    bankroll = config.initial_bankroll
    bankroll_history = [bankroll]
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    bet_rows = []
    net_multiplier = 1 - config.tax_rate

    for _, row in df.iterrows():
        if pd.isnull(row[prob_col]) or pd.isnull(row[max_t1_col]) or pd.isnull(row[max_t2_col]):
            bankroll_history.append(bankroll)
            continue

        p_t1 = row[prob_col]
        ev_t1 = row[max_t1_col] * net_multiplier * p_t1 - 1
        ev_t2 = row[max_t2_col] * net_multiplier * (1 - p_t1) - 1

        if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
            side = "T1"
            raw_odds = row[max_t1_col]
            bookmaker = row[book_t1_col]
            probability = p_t1
            is_win = int(row["t1_win"] == 1)
            edge = ev_t1
        elif ev_t2 > config.ev_threshold:
            side = "T2"
            raw_odds = row[max_t2_col]
            bookmaker = row[book_t2_col]
            probability = 1 - p_t1
            is_win = int(row["t1_win"] == 0)
            edge = ev_t2
        else:
            bankroll_history.append(bankroll)
            continue

        stake = config.fixed_stake
        if not config.allow_negative_bankroll and bankroll < stake:
            bankroll_history.append(bankroll)
            continue

        execution_odds = max(1.01, raw_odds * (1 - config.slippage))
        profit = stake * execution_odds * net_multiplier - stake if is_win else -stake
        bankroll += profit
        total_staked += stake
        total_profit += profit
        wins += is_win

        bet_rows.append(
            {
                "date": row["date"],
                "tournament": row["tournament"],
                "side": side,
                "bookmaker": bookmaker,
                "probability": probability,
                "raw_odds": raw_odds,
                "execution_odds": execution_odds,
                "edge": edge,
                "win": is_win,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            }
        )
        bankroll_history.append(bankroll)

    bets_df = pd.DataFrame(bet_rows)
    bankroll_array = np.array(bankroll_history)
    peaks = np.maximum.accumulate(bankroll_array)
    drawdowns = (peaks - bankroll_array) / (peaks + 1e-9)
    total_bets = len(bets_df)

    summary = {
        "Probability Stage": probability_stage,
        "Execution Stage": execution_stage,
        "Slippage (%)": config.slippage * 100,
        "Final Bankroll": bankroll,
        "Total Profit": total_profit,
        "ROI (%)": (bankroll - config.initial_bankroll) / config.initial_bankroll * 100,
        "Yield (%)": total_profit / total_staked * 100 if total_staked > 0 else 0.0,
        "Max Drawdown (%)": float(np.max(drawdowns) * 100),
        "Min Bankroll": float(np.min(bankroll_array)),
        "Total Staked": total_staked,
        "Bets": total_bets,
        "Win Rate (%)": wins / total_bets * 100 if total_bets > 0 else 0.0,
        "Mean Edge (%)": bets_df["edge"].mean() * 100 if total_bets > 0 else 0.0,
        "Mean Odds": bets_df["raw_odds"].mean() if total_bets > 0 else 0.0,
    }

    return summary, bets_df, bankroll_history


def main() -> None:
    """Run Market Avg + max odds-shopping simulations for 2024+."""
    print("Running 07c_market_avg_odds_shopping.py...")
    os.makedirs("docs/assets", exist_ok=True)

    df = pd.read_csv("data/odds.csv")
    df = prepare_market_data(df)
    df = df[df["date"] >= datetime(2024, 1, 1)].sort_values("date").reset_index(drop=True)
    df = df[df["t1_win"] | df["t2_win"]].copy()
    df["t1_win"] = df["t1_win"].astype(int)

    scenarios = [
        ("open", "open", 0.00, "Opening probability + opening max odds"),
        ("open", "open", 0.01, "Opening probability + opening max odds, 1% slippage"),
        ("close", "close", 0.00, "Closing probability + closing max odds"),
        ("close", "close", 0.01, "Closing probability + closing max odds, 1% slippage"),
        ("close", "open", 0.00, "Closing probability + opening max odds (ex-post benchmark)"),
    ]

    summaries = []
    all_bets = []
    histories = {}

    for prob_stage, exec_stage, slippage, label in scenarios:
        config = SimulationConfig(slippage=slippage)
        summary, bets_df, history = simulate_market_strategy(df, prob_stage, exec_stage, config)
        summary["Scenario"] = label
        summaries.append(summary)
        bets_df["Scenario"] = label
        all_bets.append(bets_df)
        histories[label] = history

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[
        [
            "Scenario",
            "Probability Stage",
            "Execution Stage",
            "Slippage (%)",
            "Final Bankroll",
            "ROI (%)",
            "Yield (%)",
            "Max Drawdown (%)",
            "Min Bankroll",
            "Bets",
            "Win Rate (%)",
            "Mean Edge (%)",
            "Mean Odds",
            "Total Staked",
            "Total Profit",
        ]
    ]
    summary_df.to_csv("docs/assets/market_avg_odds_shopping_2024_summary.csv", index=False)
    pd.concat(all_bets, ignore_index=True).to_csv("docs/assets/market_avg_odds_shopping_2024_bets.csv", index=False)

    print("\n=== MARKET AVG + MAX ODDS SHOPPING (2024+, EV > 5%, fixed stake 10$) ===")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.2f}"))

    plt.figure(figsize=(12, 7))
    for label, history in histories.items():
        plt.plot(df["date"], history[1:], label=label)
    plt.title("Market Avg + Max Odds Shopping Bankroll (2024+, EV > 5%)")
    plt.xlabel("Date")
    plt.ylabel("Bankroll ($)")
    plt.axhline(100, color="black", linestyle="--", alpha=0.6)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/assets/market_avg_odds_shopping_2024_bankroll.png", dpi=300, bbox_inches="tight")

    print("\nSaved results to docs/assets/market_avg_odds_shopping_2024_*.csv/png")
    print("07c_market_avg_odds_shopping.py completed successfully.")


if __name__ == "__main__":
    main()
