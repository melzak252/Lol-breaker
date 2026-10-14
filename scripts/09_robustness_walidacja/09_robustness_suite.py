"""Robustness and stress-test suite for Chapter 9.

The suite reuses the leakage-safe hybrid input predictions generated for
Chapter 7 and tests whether the financial conclusions from Chapter 8 survive
changes in EV threshold, slippage, stake sizing, execution price assumptions,
and market segments. Opening odds are the only operational execution prices;
closing odds are used only for CLV diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "docs" / "assets" / "hybrid_point7" / "hybrid_model_input_predictions.csv"
OUTPUT_DIR = ROOT / "docs" / "assets" / "robustness_point9"

INITIAL_BANKROLL = 100.0
TAX_RATE = 0.12
BASE_SLIPPAGE = 0.01
BASE_EV_THRESHOLD = 0.05
BASE_MIN_STAKE = 2.0
BASE_MAX_STAKE = 100.0
RANDOM_SEED = 42

TIER1_KEYWORDS = (
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "MSI",
    "World Championship",
    "Worlds",
    "First Stand",
    "LTA",
    "LCP",
)


@dataclass(frozen=True)
class Candidate:
    """Probability candidate used in robustness tests.

    Attributes:
        name: Display name.
        source: Candidate family: market, metamodel, or hybrid.
        alpha: Hybrid alpha. Ignored for market/metamodel.
        temperature: Temperature applied to metamodel probability.
    """

    name: str
    source: str
    alpha: float | None = None
    temperature: float = 1.0


@dataclass(frozen=True)
class SimulationConfig:
    """Financial simulation configuration.

    Attributes:
        ev_threshold: Minimum EV required to enter a bet.
        slippage: Execution slippage applied to selected odds.
        min_stake: Minimum stake.
        max_stake: Maximum stake.
        staking_kind: fixed, percent, or kelly.
        staking_value: Fixed stake, bankroll fraction, or Kelly multiplier.
        execution_mode: best_open, avg_open, or sts_open.
        ev_after_slippage: Whether EV filter uses slippage-adjusted odds.
    """

    ev_threshold: float = BASE_EV_THRESHOLD
    slippage: float = BASE_SLIPPAGE
    min_stake: float = BASE_MIN_STAKE
    max_stake: float = BASE_MAX_STAKE
    staking_kind: str = "kelly"
    staking_value: float = 0.25
    execution_mode: str = "best_open"
    ev_after_slippage: bool = False


def configure_style() -> None:
    """Configure plotting style for thesis figures."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
        }
    )


def safe_logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    """Return a clipped logit transformation."""
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Return sigmoid values."""
    return 1 / (1 + np.exp(-value))


def apply_temperature(probability: pd.Series | np.ndarray, temperature: float) -> np.ndarray:
    """Apply binary temperature scaling."""
    return sigmoid(safe_logit(probability) / temperature)


def candidate_probability(data: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    """Build probability vector for a candidate."""
    if candidate.source == "market":
        return data["prob_market_open"].to_numpy(dtype=float)

    model_prob = apply_temperature(data["prob_model"], candidate.temperature)
    if candidate.source == "metamodel":
        return model_prob
    if candidate.alpha is None:
        raise ValueError("Hybrid candidate requires alpha.")
    market_prob = data["prob_market_open"].to_numpy(dtype=float)
    return candidate.alpha * model_prob + (1 - candidate.alpha) * market_prob


def is_tier1(tournament: str) -> bool:
    """Classify a tournament as Tier 1 using keyword heuristics."""
    text = str(tournament)
    return any(keyword in text for keyword in TIER1_KEYWORDS)


def add_segments(data: pd.DataFrame) -> pd.DataFrame:
    """Add segment labels used in robustness reporting."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["tier_segment"] = np.where(frame["tournament"].apply(is_tier1), "Tier 1", "Regional / ERL")
    frame["bon_segment"] = "Bo" + frame["BoN"].astype(int).astype(str)
    frame["market_favorite_prob"] = np.maximum(frame["prob_market_open"], 1 - frame["prob_market_open"])
    frame["favorite_side"] = np.where(frame["prob_market_open"] >= 0.5, "t1", "t2")
    frame["odds_bucket"] = pd.cut(
        np.maximum(frame["avg_open_home"], frame["avg_open_away"]),
        bins=[1.0, 1.5, 2.25, 4.0, 100.0],
        labels=["low", "near_even", "underdog", "longshot"],
        include_lowest=True,
    ).astype(str)
    return frame


def load_data() -> pd.DataFrame:
    """Load hybrid input data and add segments."""
    data = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    data = data.sort_values("date").reset_index(drop=True)
    return add_segments(data)


def get_execution_odds(row: pd.Series, mode: str) -> tuple[float, float]:
    """Return execution odds for team 1 and team 2 under a mode."""
    if mode == "best_open":
        return float(row["best_open_t1"]), float(row["best_open_t2"])
    if mode == "avg_open":
        return float(row["avg_open_home"]), float(row["avg_open_away"])
    if mode == "sts_open":
        return float(row.get("odds1_sts_open", np.nan)), float(row.get("odds2_sts_open", np.nan))
    raise ValueError(f"Unknown execution mode: {mode}")


def choose_bet(
    row: pd.Series,
    probability: float,
    config: SimulationConfig,
) -> tuple[str | None, float, float, float]:
    """Choose a bet if the configured EV threshold is exceeded."""
    odds_t1, odds_t2 = get_execution_odds(row, config.execution_mode)
    if pd.isna(odds_t1) or pd.isna(odds_t2) or odds_t1 <= 1 or odds_t2 <= 1:
        return None, 0.0, 0.0, 0.0

    ev_odds_t1 = odds_t1 * (1 - config.slippage) if config.ev_after_slippage else odds_t1
    ev_odds_t2 = odds_t2 * (1 - config.slippage) if config.ev_after_slippage else odds_t2
    prob_t1 = float(probability)
    prob_t2 = 1 - prob_t1
    ev_t1 = prob_t1 * ev_odds_t1 * (1 - TAX_RATE) - 1
    ev_t2 = prob_t2 * ev_odds_t2 * (1 - TAX_RATE) - 1

    if ev_t1 > ev_t2 and ev_t1 > config.ev_threshold:
        return "t1", prob_t1, odds_t1, float(ev_t1)
    if ev_t2 > config.ev_threshold:
        return "t2", prob_t2, odds_t2, float(ev_t2)
    return None, 0.0, 0.0, 0.0


def calculate_stake(
    bankroll: float,
    probability: float,
    raw_odds: float,
    config: SimulationConfig,
) -> float:
    """Calculate stake according to the simulation config."""
    if bankroll <= 0:
        return 0.0
    if config.staking_kind == "fixed":
        stake = config.staking_value
    elif config.staking_kind == "percent":
        stake = bankroll * config.staking_value
        stake = min(max(stake, config.min_stake), config.max_stake)
    elif config.staking_kind == "kelly":
        execution_odds = raw_odds * (1 - config.slippage)
        net_decimal = execution_odds * (1 - TAX_RATE)
        b = net_decimal - 1
        if b <= 0:
            return 0.0
        full_kelly = ((b * probability) - (1 - probability)) / b
        if full_kelly <= 0:
            return 0.0
        stake = bankroll * full_kelly * config.staking_value
        stake = min(max(stake, config.min_stake), config.max_stake)
    else:
        raise ValueError(f"Unknown staking kind: {config.staking_kind}")
    return float(stake) if stake <= bankroll else 0.0


def simulate(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    candidate_name: str,
    config: SimulationConfig,
    label: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Simulate a candidate and return summary plus bet-level rows."""
    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    total_profit = 0.0
    bets = 0
    wins = 0
    skipped_due_bankroll = 0
    clv_values: list[float] = []
    stake_values: list[float] = []
    rows: list[dict[str, float | int | str]] = []

    for idx, (_, row) in enumerate(data.iterrows()):
        side, selected_prob, raw_odds, selected_ev = choose_bet(row, probabilities[idx], config)
        if side is None:
            continue
        stake = calculate_stake(bankroll, selected_prob, raw_odds, config)
        if stake <= 0:
            skipped_due_bankroll += 1
            continue

        execution_odds = raw_odds * (1 - config.slippage)
        is_win = (side == "t1" and row["y_true"] == 1) or (side == "t2" and row["y_true"] == 0)
        profit = stake * (execution_odds * (1 - TAX_RATE) - 1) if is_win else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)
        bets += 1
        wins += int(is_win)
        total_staked += stake
        total_profit += profit
        stake_values.append(stake)

        close_odds = row["best_close_t1"] if side == "t1" else row["best_close_t2"]
        clv = np.nan
        if pd.notna(close_odds) and close_odds > 0:
            clv = (raw_odds - float(close_odds)) / float(close_odds)
            clv_values.append(clv)

        rows.append(
            {
                "label": label,
                "candidate": candidate_name,
                "date": row["date"],
                "year": row["year"],
                "tournament": row["tournament"],
                "BoN": row["BoN"],
                "bon_segment": row["bon_segment"],
                "tier_segment": row["tier_segment"],
                "odds_bucket": row["odds_bucket"],
                "side": side,
                "is_favorite_bet": side == row["favorite_side"],
                "probability": selected_prob,
                "raw_odds": raw_odds,
                "selected_ev": selected_ev,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
                "is_win": int(is_win),
                "clv": clv,
            }
        )

    summary: dict[str, float | int | str] = {
        "label": label,
        "candidate": candidate_name,
        "final_bankroll": bankroll,
        "roi_pct": (bankroll / INITIAL_BANKROLL - 1) * 100,
        "yield_pct": (total_profit / total_staked * 100) if total_staked > 0 else 0.0,
        "max_drawdown_pct": max_drawdown * 100,
        "total_staked": total_staked,
        "bets": bets,
        "win_rate_pct": wins / bets * 100 if bets else 0.0,
        "avg_stake": float(np.mean(stake_values)) if stake_values else 0.0,
        "median_stake": float(np.median(stake_values)) if stake_values else 0.0,
        "cap_hit_pct": float(np.mean(np.isclose(stake_values, config.max_stake)) * 100) if stake_values else 0.0,
        "min_bankroll": float(min([INITIAL_BANKROLL] + [r["bankroll"] for r in rows])) if rows else INITIAL_BANKROLL,
        "skipped_due_bankroll": skipped_due_bankroll,
        "avg_clv_pct": float(np.nanmean(clv_values) * 100) if clv_values else np.nan,
        "median_clv_pct": float(np.nanmedian(clv_values) * 100) if clv_values else np.nan,
        "beating_close_pct": float(np.mean(np.asarray(clv_values) > 0) * 100) if clv_values else np.nan,
    }
    return summary, pd.DataFrame(rows)


def base_candidates() -> list[Candidate]:
    """Return candidate list used in Chapter 9."""
    return [
        Candidate("Market Avg Open", "market"),
        Candidate("Metamodel T=1.00", "metamodel", temperature=1.0),
        Candidate("Hybrid a=0.62 T=0.80", "hybrid", alpha=0.62, temperature=0.80),
        Candidate("Hybrid a=0.48 T=0.70", "hybrid", alpha=0.48, temperature=0.70),
        Candidate("Hybrid a=0.48 T=0.60", "hybrid", alpha=0.48, temperature=0.60),
    ]


def run_sensitivity(
    data: pd.DataFrame,
    candidate: Candidate,
    values: list[float | str],
    param_name: str,
    make_config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a one-dimensional sensitivity experiment."""
    summaries = []
    bets = []
    probabilities = candidate_probability(data, candidate)
    for value in values:
        label = f"{param_name}={value}"
        summary, bet_rows = simulate(data, probabilities, candidate.name, make_config(value), label)
        summary[param_name] = value
        summaries.append(summary)
        bets.append(bet_rows)
    return pd.DataFrame(summaries), pd.concat(bets, ignore_index=True) if bets else pd.DataFrame()


def summarize_segments(bets: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize bet-level performance by a segment."""
    if bets.empty:
        return pd.DataFrame()
    grouped = bets.groupby(group_column, dropna=False)
    return grouped.agg(
        bets=("profit", "size"),
        profit=("profit", "sum"),
        total_staked=("stake", "sum"),
        win_rate_pct=("is_win", lambda x: float(x.mean() * 100)),
        avg_odds=("raw_odds", "mean"),
        avg_ev=("selected_ev", "mean"),
        avg_clv_pct=("clv", lambda x: float(np.nanmean(x) * 100)),
        median_clv_pct=("clv", lambda x: float(np.nanmedian(x) * 100)),
    ).reset_index().assign(
        yield_pct=lambda frame: frame["profit"] / frame["total_staked"] * 100,
    )


def bootstrap_ci(bets: pd.DataFrame, n_bootstrap: int = 500) -> pd.DataFrame:
    """Bootstrap confidence intervals for selected bet-level metrics."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for candidate, group in bets.groupby("candidate"):
        if group.empty:
            continue
        profits = group["profit"].to_numpy(dtype=float)
        stakes = group["stake"].to_numpy(dtype=float)
        clv = group["clv"].dropna().to_numpy(dtype=float)
        boot_yield = []
        boot_roi = []
        boot_clv = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(group), len(group))
            profit_sum = profits[idx].sum()
            stake_sum = stakes[idx].sum()
            boot_yield.append(profit_sum / stake_sum * 100 if stake_sum > 0 else 0.0)
            boot_roi.append(profit_sum / INITIAL_BANKROLL * 100)
            if len(clv) > 0:
                clv_idx = rng.integers(0, len(clv), len(clv))
                boot_clv.append(np.mean(clv[clv_idx]) * 100)
        rows.append(
            {
                "candidate": candidate,
                "yield_ci_low": float(np.percentile(boot_yield, 2.5)),
                "yield_ci_high": float(np.percentile(boot_yield, 97.5)),
                "roi_ci_low": float(np.percentile(boot_roi, 2.5)),
                "roi_ci_high": float(np.percentile(boot_roi, 97.5)),
                "clv_ci_low": float(np.percentile(boot_clv, 2.5)) if boot_clv else np.nan,
                "clv_ci_high": float(np.percentile(boot_clv, 97.5)) if boot_clv else np.nan,
                "bets": len(group),
            }
        )
    return pd.DataFrame(rows)


def save_line_plot(data: pd.DataFrame, x: str, y: str, hue: str, path: Path, title: str) -> None:
    """Save a simple line plot for sensitivity outputs."""
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=data, x=x, y=y, hue=hue, marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_bar_plot(data: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    """Save a horizontal bar plot."""
    plot_data = data.sort_values(y, ascending=True)
    plt.figure(figsize=(11, max(5, 0.45 * len(plot_data))))
    sns.barplot(data=plot_data, x=y, y=x, hue=x, palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel(y)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run robustness suite and save outputs."""
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    modern = data[data["date"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True)
    candidates = base_candidates()
    primary = Candidate("Hybrid a=0.48 T=0.60", "hybrid", alpha=0.48, temperature=0.60)
    defensive = Candidate("Hybrid a=0.62 T=0.80", "hybrid", alpha=0.62, temperature=0.80)

    base_config = SimulationConfig(staking_kind="kelly", staking_value=0.25)

    # Candidate leaderboard under the base robustness policy.
    summaries = []
    base_bets = []
    for candidate in candidates:
        summary, bets = simulate(
            modern,
            candidate_probability(modern, candidate),
            candidate.name,
            base_config,
            "base_kelly_025_2024",
        )
        summaries.append(summary)
        base_bets.append(bets)
    leaderboard = pd.DataFrame(summaries)
    base_bets_frame = pd.concat(base_bets, ignore_index=True)
    leaderboard.to_csv(OUTPUT_DIR / "robustness_base_leaderboard_2024.csv", index=False)
    base_bets_frame.to_csv(OUTPUT_DIR / "robustness_base_bets_2024.csv", index=False)

    # Sensitivities for primary candidate.
    ev_summary, _ = run_sensitivity(
        modern,
        primary,
        [0.0, 0.025, 0.05, 0.075, 0.10],
        "ev_threshold",
        lambda value: SimulationConfig(ev_threshold=float(value), staking_kind="kelly", staking_value=0.25),
    )
    ev_summary.to_csv(OUTPUT_DIR / "robustness_ev_threshold.csv", index=False)
    save_line_plot(ev_summary, "ev_threshold", "yield_pct", "candidate", OUTPUT_DIR / "robustness_ev_threshold_yield.png", "EV threshold sensitivity — Yield")

    slippage_rows = []
    for after_slippage in [False, True]:
        summary, _ = run_sensitivity(
            modern,
            primary,
            [0.0, 0.01, 0.02, 0.03],
            "slippage",
            lambda value, after=after_slippage: SimulationConfig(
                slippage=float(value),
                ev_after_slippage=after,
                staking_kind="kelly",
                staking_value=0.25,
            ),
        )
        summary["ev_filter"] = "after_slippage" if after_slippage else "before_slippage"
        slippage_rows.append(summary)
    slippage_summary = pd.concat(slippage_rows, ignore_index=True)
    slippage_summary.to_csv(OUTPUT_DIR / "robustness_slippage.csv", index=False)
    save_line_plot(slippage_summary, "slippage", "yield_pct", "ev_filter", OUTPUT_DIR / "robustness_slippage_yield.png", "Slippage sensitivity — Yield")

    kelly_summary, _ = run_sensitivity(
        modern,
        primary,
        [0.05, 0.10, 0.25, 0.50, 1.00],
        "kelly_multiplier",
        lambda value: SimulationConfig(staking_kind="kelly", staking_value=float(value)),
    )
    kelly_summary.to_csv(OUTPUT_DIR / "robustness_kelly_multiplier.csv", index=False)
    save_line_plot(kelly_summary, "kelly_multiplier", "max_drawdown_pct", "candidate", OUTPUT_DIR / "robustness_kelly_drawdown.png", "Kelly multiplier sensitivity — MaxDD")

    cap_summary, _ = run_sensitivity(
        modern,
        primary,
        [25, 50, 100, 250],
        "max_stake",
        lambda value: SimulationConfig(max_stake=float(value), staking_kind="kelly", staking_value=0.25),
    )
    cap_summary.to_csv(OUTPUT_DIR / "robustness_stake_cap.csv", index=False)
    save_line_plot(cap_summary, "max_stake", "final_bankroll", "candidate", OUTPUT_DIR / "robustness_stake_cap_bankroll.png", "Stake cap sensitivity — final bankroll")

    min_stake_summary, _ = run_sensitivity(
        modern,
        primary,
        [0, 2, 5],
        "min_stake",
        lambda value: SimulationConfig(min_stake=float(value), staking_kind="kelly", staking_value=0.25),
    )
    min_stake_summary.to_csv(OUTPUT_DIR / "robustness_min_stake.csv", index=False)

    execution_summary, execution_bets = run_sensitivity(
        modern,
        primary,
        ["best_open", "avg_open", "sts_open"],
        "execution_mode",
        lambda value: SimulationConfig(execution_mode=str(value), staking_kind="kelly", staking_value=0.25),
    )
    execution_summary.to_csv(OUTPUT_DIR / "robustness_execution_price.csv", index=False)
    execution_bets.to_csv(OUTPUT_DIR / "robustness_execution_bets.csv", index=False)
    save_bar_plot(execution_summary, "execution_mode", "yield_pct", OUTPUT_DIR / "robustness_execution_yield.png", "Execution price sensitivity — Yield")

    flat_rows = []
    for unit in [1.0, 2.0, 10.0]:
        summary, _ = simulate(
            modern,
            candidate_probability(modern, primary),
            primary.name,
            SimulationConfig(staking_kind="fixed", staking_value=unit),
            f"flat_{unit}",
        )
        summary["flat_unit"] = unit
        flat_rows.append(summary)
    pd.DataFrame(flat_rows).to_csv(OUTPUT_DIR / "robustness_conservative_flat.csv", index=False)

    # Segmentation and CLV under base Kelly 0.25 for primary candidate.
    primary_base_summary, primary_bets = simulate(
        modern,
        candidate_probability(modern, primary),
        primary.name,
        base_config,
        "primary_base_kelly_025_2024",
    )
    primary_bets.to_csv(OUTPUT_DIR / "robustness_primary_bets_2024.csv", index=False)
    segment_frames = []
    for column in ["year", "bon_segment", "tier_segment", "odds_bucket", "is_favorite_bet", "tournament"]:
        frame = summarize_segments(primary_bets, column)
        frame.insert(0, "segment_type", column)
        frame = frame.rename(columns={column: "segment"})
        segment_frames.append(frame)
        frame.to_csv(OUTPUT_DIR / f"robustness_segment_{column}.csv", index=False)
    segments = pd.concat(segment_frames, ignore_index=True)
    segments.to_csv(OUTPUT_DIR / "robustness_segments_all.csv", index=False)

    clv = primary_bets.assign(
        clv_positive=lambda frame: frame["clv"] > 0,
        result=lambda frame: np.where(frame["is_win"] == 1, "winner", "loser"),
    )
    clv_summary = clv.groupby("result").agg(
        bets=("profit", "size"),
        mean_clv_pct=("clv", lambda x: float(np.nanmean(x) * 100)),
        median_clv_pct=("clv", lambda x: float(np.nanmedian(x) * 100)),
        beating_close_pct=("clv_positive", lambda x: float(x.mean() * 100)),
    ).reset_index()
    overall_clv = pd.DataFrame(
        [
            {
                "result": "all",
                "bets": len(clv),
                "mean_clv_pct": float(np.nanmean(clv["clv"]) * 100),
                "median_clv_pct": float(np.nanmedian(clv["clv"]) * 100),
                "beating_close_pct": float((clv["clv"] > 0).mean() * 100),
            }
        ]
    )
    clv_full = pd.concat([overall_clv, clv_summary], ignore_index=True)
    clv_full.to_csv(OUTPUT_DIR / "robustness_clv_summary.csv", index=False)

    bootstrap_ci(base_bets_frame).to_csv(OUTPUT_DIR / "robustness_bootstrap_ci.csv", index=False)

    # LogLoss bootstrap for candidates on 2024+ matches.
    rng = np.random.default_rng(RANDOM_SEED)
    logloss_rows = []
    y_true = modern["y_true"].to_numpy(dtype=int)
    for candidate in candidates:
        probs = candidate_probability(modern, candidate)
        boot = []
        for _ in range(500):
            idx = rng.integers(0, len(y_true), len(y_true))
            boot.append(log_loss(y_true[idx], probs[idx], labels=[0, 1]))
        logloss_rows.append(
            {
                "candidate": candidate.name,
                "logloss": log_loss(y_true, probs, labels=[0, 1]),
                "logloss_ci_low": float(np.percentile(boot, 2.5)),
                "logloss_ci_high": float(np.percentile(boot, 97.5)),
            }
        )
    pd.DataFrame(logloss_rows).to_csv(OUTPUT_DIR / "robustness_logloss_bootstrap_ci.csv", index=False)

    # Summary markdown for quick inspection.
    with (OUTPUT_DIR / "robustness_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write("Chapter 9 robustness suite completed.\n\n")
        handle.write("Base leaderboard 2024+ Kelly 0.25:\n")
        handle.write(leaderboard.sort_values("final_bankroll", ascending=False).to_string(index=False))
        handle.write("\n\nPrimary candidate base summary:\n")
        handle.write(str(primary_base_summary))


if __name__ == "__main__":
    main()
