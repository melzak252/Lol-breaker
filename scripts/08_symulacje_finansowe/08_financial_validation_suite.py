"""Financial validation suite for the current LR-W20-Binomial model.

This script replaces the legacy hybrid-input dependency with the current final
common sample produced in Chapter 7. It evaluates the final logistic model,
the opening market benchmark and simple model-market hybrids under several
staking policies.

The output is diagnostic only: it tests historical betting-style simulations,
but the thesis narrative remains focused on probabilistic prediction quality.
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

from src.visualization.thesis_style import apply_thesis_style, clean_axis, colors_for


INPUT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "assets"
    / "final_w20_binomial_market_comparison"
    / "final_w20_binomial_market_common_sample.csv"
)
ODDS_PATH = PROJECT_ROOT / "data" / "odds.csv"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "financial_point8"

BOOKMAKERS = ["betclic", "betfan", "efortuna", "lv_bet", "sts", "superbet"]
MODEL_PROBABILITY = "prob_lr_elasticnet_w20_binomial"

INITIAL_BANKROLL = 100.0
TAX_RATE = 0.12
SLIPPAGE = 0.01
EV_THRESHOLD = 0.05
MIN_STAKE = 2.0
MAX_STAKE = 100.0
FIXED_STAKE = 10.0


@dataclass(frozen=True)
class Candidate:
    """Probability candidate evaluated in the financial suite.

    Attributes:
        name: Human-readable candidate name.
        source: Candidate source: market, model, or hybrid.
        alpha: Model weight for hybrid candidates.
        temperature: Temperature applied to model probabilities before mixing.
    """

    name: str
    source: str
    alpha: float | None = None
    temperature: float = 1.0


@dataclass(frozen=True)
class StakingPolicy:
    """Staking policy configuration."""

    name: str
    kind: str
    fixed_stake: float | None = None
    fraction: float | None = None
    min_stake: float = MIN_STAKE
    max_stake: float = MAX_STAKE


def safe_logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    """Calculate a clipped logit transform."""

    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Calculate the sigmoid function."""

    return 1.0 / (1.0 + np.exp(-value))


def apply_temperature(probability: pd.Series | np.ndarray, temperature: float) -> np.ndarray:
    """Apply binary temperature scaling to a probability vector."""

    return sigmoid(safe_logit(probability) / temperature)


def load_dataset() -> pd.DataFrame:
    """Load final model sample and attach executable bookmaker odds.

    Returns:
        Chronologically sorted final common sample with best open/close odds
        aligned to GOL.GG Team 1 and Team 2.
    """

    final_sample = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    odds = pd.read_csv(ODDS_PATH)
    final_sample["golgg_match_id"] = final_sample["golgg_match_id"].astype(str)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)

    odds_columns = ["golgg_match_id", "tournament"]
    for bookmaker in BOOKMAKERS:
        odds_columns.extend(
            [
                f"odds1_{bookmaker}_open",
                f"odds2_{bookmaker}_open",
                f"odds1_{bookmaker}_close",
                f"odds2_{bookmaker}_close",
            ]
        )

    data = final_sample.merge(odds[odds_columns], on="golgg_match_id", how="inner")
    for side in ("t1", "t2"):
        data[f"best_open_{side}"] = np.nan
        data[f"best_close_{side}"] = np.nan

    open_t1_values: list[float] = []
    open_t2_values: list[float] = []
    close_t1_values: list[float] = []
    close_t2_values: list[float] = []

    for _, row in data.iterrows():
        if row["market_side_alignment"] == "swapped":
            open_t1_cols = [f"odds2_{bookmaker}_open" for bookmaker in BOOKMAKERS]
            open_t2_cols = [f"odds1_{bookmaker}_open" for bookmaker in BOOKMAKERS]
            close_t1_cols = [f"odds2_{bookmaker}_close" for bookmaker in BOOKMAKERS]
            close_t2_cols = [f"odds1_{bookmaker}_close" for bookmaker in BOOKMAKERS]
        else:
            open_t1_cols = [f"odds1_{bookmaker}_open" for bookmaker in BOOKMAKERS]
            open_t2_cols = [f"odds2_{bookmaker}_open" for bookmaker in BOOKMAKERS]
            close_t1_cols = [f"odds1_{bookmaker}_close" for bookmaker in BOOKMAKERS]
            close_t2_cols = [f"odds2_{bookmaker}_close" for bookmaker in BOOKMAKERS]

        open_t1_values.append(float(pd.to_numeric(row[open_t1_cols], errors="coerce").max()))
        open_t2_values.append(float(pd.to_numeric(row[open_t2_cols], errors="coerce").max()))
        close_t1_values.append(float(pd.to_numeric(row[close_t1_cols], errors="coerce").max()))
        close_t2_values.append(float(pd.to_numeric(row[close_t2_cols], errors="coerce").max()))

    data["best_open_t1"] = open_t1_values
    data["best_open_t2"] = open_t2_values
    data["best_close_t1"] = close_t1_values
    data["best_close_t2"] = close_t2_values

    required = [
        "y_true",
        MODEL_PROBABILITY,
        "market_open",
        "market_close",
        "best_open_t1",
        "best_open_t2",
    ]
    return data.dropna(subset=required).sort_values("date").reset_index(drop=True)


def candidate_probability(data: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    """Build Team-1 probability for a candidate."""

    market = data["market_open"].to_numpy(dtype=float)
    model = apply_temperature(data[MODEL_PROBABILITY], candidate.temperature)

    if candidate.source == "market":
        return market
    if candidate.source == "model":
        return model
    if candidate.source == "hybrid":
        if candidate.alpha is None:
            raise ValueError("Hybrid candidate requires alpha.")
        return candidate.alpha * model + (1.0 - candidate.alpha) * market
    raise ValueError(f"Unknown candidate source: {candidate.source}")


def choose_bet(row: pd.Series, probability: float) -> tuple[str | None, float, float, float]:
    """Choose the side with positive expected value, if any."""

    prob_t1 = float(probability)
    prob_t2 = 1.0 - prob_t1
    odds_t1 = float(row["best_open_t1"])
    odds_t2 = float(row["best_open_t2"])
    ev_t1 = prob_t1 * odds_t1 * (1.0 - TAX_RATE) - 1.0
    ev_t2 = prob_t2 * odds_t2 * (1.0 - TAX_RATE) - 1.0

    if ev_t1 > ev_t2 and ev_t1 > EV_THRESHOLD:
        return "t1", prob_t1, odds_t1, ev_t1
    if ev_t2 > EV_THRESHOLD:
        return "t2", prob_t2, odds_t2, ev_t2
    return None, 0.0, 0.0, 0.0


def calculate_stake(
    bankroll: float,
    policy: StakingPolicy,
    probability: float,
    raw_odds: float,
) -> float:
    """Calculate stake for the selected policy."""

    if bankroll <= 0:
        return 0.0
    if policy.kind == "fixed":
        stake = float(policy.fixed_stake or 0.0)
    elif policy.kind == "percent":
        stake = bankroll * float(policy.fraction or 0.0)
        stake = min(max(stake, policy.min_stake), policy.max_stake)
    elif policy.kind == "kelly":
        execution_odds = raw_odds * (1.0 - SLIPPAGE)
        net_decimal = execution_odds * (1.0 - TAX_RATE)
        b = net_decimal - 1.0
        if b <= 0:
            return 0.0
        full_kelly = ((b * probability) - (1.0 - probability)) / b
        if full_kelly <= 0:
            return 0.0
        stake = bankroll * full_kelly * float(policy.fraction or 0.0)
        stake = min(max(stake, policy.min_stake), policy.max_stake)
    else:
        raise ValueError(f"Unknown staking policy: {policy.kind}")
    return float(stake if stake <= bankroll else 0.0)


def simulate_strategy(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    candidate_name: str,
    policy: StakingPolicy,
    scope_name: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Simulate one candidate and staking policy."""

    bankroll = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_drawdown = 0.0
    total_staked = 0.0
    total_profit = 0.0
    bets = 0
    wins = 0
    clv_values: list[float] = []
    rows: list[dict[str, float | int | str]] = []

    for index, (_, row) in enumerate(data.iterrows()):
        side, selected_prob, raw_odds, selected_ev = choose_bet(row, probabilities[index])
        profit = 0.0
        stake = 0.0
        is_win = False

        if side is not None:
            stake = calculate_stake(bankroll, policy, selected_prob, raw_odds)
            if stake > 0:
                execution_odds = raw_odds * (1.0 - SLIPPAGE)
                if side == "t1":
                    is_win = bool(row["y_true"] == 1)
                    close_odds = row.get("best_close_t1", np.nan)
                else:
                    is_win = bool(row["y_true"] == 0)
                    close_odds = row.get("best_close_t2", np.nan)

                if is_win:
                    profit = stake * (execution_odds * (1.0 - TAX_RATE) - 1.0)
                    wins += 1
                else:
                    profit = -stake
                bankroll += profit
                total_staked += stake
                total_profit += profit
                bets += 1

                if pd.notna(close_odds) and close_odds > 0:
                    clv_values.append((raw_odds - float(close_odds)) / float(close_odds))

        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)
        rows.append(
            {
                "scope": scope_name,
                "candidate": candidate_name,
                "staking_policy": policy.name,
                "date": row["date"],
                "match_index": index,
                "bankroll": bankroll,
                "stake": stake,
                "profit": profit,
                "selected_ev": selected_ev,
                "bet_placed": int(stake > 0),
                "win": int(is_win),
            }
        )

    summary = {
        "scope": scope_name,
        "candidate": candidate_name,
        "staking_policy": policy.name,
        "final_bankroll": bankroll,
        "profit": total_profit,
        "roi_pct": (bankroll / INITIAL_BANKROLL - 1.0) * 100.0,
        "yield_pct": (total_profit / total_staked * 100.0) if total_staked else 0.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "total_staked": total_staked,
        "bets": bets,
        "win_rate_pct": (wins / bets * 100.0) if bets else 0.0,
        "avg_stake": (total_staked / bets) if bets else 0.0,
        "avg_clv_pct": (np.mean(clv_values) * 100.0) if clv_values else 0.0,
    }
    return summary, pd.DataFrame(rows)


def evaluate_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate probabilistic metrics for one candidate."""

    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    return {
        "auc": roc_auc_score(y_true, clipped),
        "logloss": log_loss(y_true, clipped),
        "brier": brier_score_loss(y_true, clipped),
        "ece": calculate_ece(y_true, clipped),
    }


def save_top_bankroll_plot(history: pd.DataFrame, summary: pd.DataFrame, scope: str) -> None:
    """Save a bankroll plot for the best fixed-percent configurations."""

    fixed_percent = summary[
        (summary["scope"] == scope)
        & (summary["staking_policy"] == "Fixed percent 2% min2 max100")
    ].sort_values("final_bankroll", ascending=False)
    selected = fixed_percent.head(5)[["candidate", "staking_policy"]]
    plot_data = history.merge(selected, on=["candidate", "staking_policy"])
    plot_data = plot_data[plot_data["scope"] == scope].copy()
    if plot_data.empty:
        return

    apply_thesis_style(context="paper")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.lineplot(
        data=plot_data,
        x="date",
        y="bankroll",
        hue="candidate",
        palette=colors_for(plot_data["candidate"].unique()),
        linewidth=2.0,
        ax=ax,
    )
    ax.set_title(f"Symulacja finansowa — bankroll ({scope})")
    ax.set_xlabel("Data")
    ax.set_ylabel("Bankroll")
    clean_axis(ax)
    fig.tight_layout()
    filename = f"financial_bankroll_{scope.lower().replace('+', 'plus')}.png"
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_kelly_sensitivity(summary: pd.DataFrame) -> None:
    """Save Kelly multiplier sensitivity plot for hybrid candidates."""

    data = summary[
        summary["staking_policy"].str.startswith("Kelly")
        & summary["candidate"].str.contains("Hybrid")
        & (summary["scope"] == "2024+")
    ].copy()
    if data.empty:
        return
    data["kelly_multiplier"] = data["staking_policy"].str.extract(r"Kelly ([0-9.]+)").astype(float)

    apply_thesis_style(context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.lineplot(data=data, x="kelly_multiplier", y="roi_pct", hue="candidate", marker="o", ax=axes[0])
    axes[0].set_title("Kelly multiplier vs ROI — 2024+")
    axes[0].set_xlabel("Kelly multiplier")
    axes[0].set_ylabel("ROI [%]")
    clean_axis(axes[0])
    sns.lineplot(
        data=data,
        x="kelly_multiplier",
        y="max_drawdown_pct",
        hue="candidate",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Kelly multiplier vs Max Drawdown — 2024+")
    axes[1].set_xlabel("Kelly multiplier")
    axes[1].set_ylabel("Max Drawdown [%]")
    clean_axis(axes[1])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_kelly_sensitivity_2024.png", bbox_inches="tight")
    plt.close(fig)


def save_drawdown_frontier(summary: pd.DataFrame) -> None:
    """Save a risk-return scatter plot for the 2024+ financial grid."""

    data = summary[(summary["scope"] == "2024+") & (summary["bets"] >= 25)].copy()
    if data.empty:
        return

    data["policy_family"] = data["staking_policy"].str.extract(r"^(Fixed|Kelly)")
    apply_thesis_style(context="paper")
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    sns.scatterplot(
        data=data,
        x="max_drawdown_pct",
        y="roi_pct",
        hue="policy_family",
        size="bets",
        sizes=(30, 180),
        alpha=0.75,
        ax=ax,
    )
    ax.axvline(25, color="#2F3640", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(35, color="#2F3640", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_title("Kompromis ROI--MaxDD dla siatki alpha i stakingu — 2024+")
    ax.set_xlabel("Max Drawdown [%] (niżej = bezpieczniej)")
    ax.set_ylabel("ROI [%]")
    clean_axis(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "financial_roi_drawdown_frontier_2024.png", bbox_inches="tight")
    plt.close(fig)


def build_candidates() -> list[Candidate]:
    """Build a dense alpha grid for model-market hybrid candidates."""

    candidates = [
        Candidate("Market Open", "market"),
        Candidate("LR-ElasticNet-W20-Binomial", "model", temperature=1.0),
        Candidate("LR-ElasticNet T=0.80", "model", temperature=0.8),
        Candidate("LR-ElasticNet T=0.90", "model", temperature=0.9),
        Candidate("LR-ElasticNet T=1.10", "model", temperature=1.1),
        Candidate("LR-ElasticNet T=1.20", "model", temperature=1.2),
    ]

    for temperature in (0.8, 0.9, 1.0):
        for alpha in np.round(np.arange(0.1, 1.0, 0.1), 2):
            if temperature == 1.0:
                name = f"Hybrid a={alpha:.2f}"
            else:
                name = f"Hybrid a={alpha:.2f} T={temperature:.2f}"
            candidates.append(
                Candidate(
                    name=name,
                    source="hybrid",
                    alpha=float(alpha),
                    temperature=float(temperature),
                )
            )
    return candidates


def build_staking_policies() -> list[StakingPolicy]:
    """Build conservative and aggressive staking policies for drawdown analysis."""

    return [
        StakingPolicy("Fixed stake 2", "fixed", fixed_stake=2.0),
        StakingPolicy("Fixed stake 5", "fixed", fixed_stake=5.0),
        StakingPolicy("Fixed stake 10", "fixed", fixed_stake=FIXED_STAKE),
        StakingPolicy(
            "Fixed percent 0.5% min2 max100",
            "percent",
            fraction=0.005,
        ),
        StakingPolicy(
            "Fixed percent 1% min2 max100",
            "percent",
            fraction=0.01,
        ),
        StakingPolicy("Fixed percent 2% min2 max100", "percent", fraction=0.02),
        StakingPolicy(
            "Kelly 0.02 min2 max100",
            "kelly",
            fraction=0.02,
        ),
        StakingPolicy(
            "Kelly 0.05 min2 max100",
            "kelly",
            fraction=0.05,
        ),
        StakingPolicy("Kelly 0.10 min2 max100", "kelly", fraction=0.10),
        StakingPolicy("Kelly 0.25 min2 max100", "kelly", fraction=0.25),
        StakingPolicy("Kelly 0.50 min2 max100", "kelly", fraction=0.50),
    ]


def main() -> None:
    """Run current-model financial validation experiments."""

    apply_thesis_style(context="paper")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()

    candidates = build_candidates()
    policies = build_staking_policies()
    scopes = {
        "2021+": data[data["date"] >= pd.Timestamp("2021-01-01")].copy(),
        "2024+": data[data["date"] >= pd.Timestamp("2024-01-01")].copy(),
    }

    summaries: list[dict[str, float | int | str]] = []
    histories: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []

    for scope_name, scope_data in scopes.items():
        y_true = scope_data["y_true"].to_numpy(dtype=int)
        for candidate in candidates:
            probabilities = candidate_probability(scope_data, candidate)
            metric_row = {
                "scope": scope_name,
                "candidate": candidate.name,
                **evaluate_probability_metrics(y_true, probabilities),
                "n_matches": len(scope_data),
            }
            metric_rows.append(metric_row)
            for policy in policies:
                summary, history = simulate_strategy(
                    scope_data,
                    probabilities,
                    candidate.name,
                    policy,
                    scope_name,
                )
                summary.update(metric_row)
                summaries.append(summary)
                histories.append(history)

    summary_df = pd.DataFrame(summaries)
    summary_df["roi_to_maxdd"] = summary_df["roi_pct"] / summary_df[
        "max_drawdown_pct"
    ].clip(lower=1e-6)
    history_df = pd.concat(histories, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    summary_df.to_csv(OUTPUT_DIR / "financial_validation_summary.csv", index=False)
    history_df.to_csv(OUTPUT_DIR / "financial_validation_bankroll_history.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "financial_validation_probability_metrics.csv", index=False)

    save_top_bankroll_plot(history_df, summary_df, "2021+")
    save_top_bankroll_plot(history_df, summary_df, "2024+")
    save_kelly_sensitivity(summary_df)
    save_drawdown_frontier(summary_df)

    top_2024 = summary_df[summary_df["scope"] == "2024+"].sort_values(
        "final_bankroll", ascending=False
    ).head(12)
    print("\nTop 2024+ financial configurations:")
    print(
        top_2024[
            [
                "candidate",
                "staking_policy",
                "final_bankroll",
                "roi_pct",
                "yield_pct",
                "max_drawdown_pct",
                "bets",
                "avg_stake",
                "logloss",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    risk_controlled = summary_df[
        (summary_df["scope"] == "2024+")
        & (summary_df["roi_pct"] > 0)
        & (summary_df["bets"] >= 100)
        & (summary_df["max_drawdown_pct"] <= 25)
    ].sort_values("roi_to_maxdd", ascending=False)
    if not risk_controlled.empty:
        print("\nBest 2024+ configurations with MaxDD <= 25%:")
        print(
            risk_controlled[
                [
                    "candidate",
                    "staking_policy",
                    "final_bankroll",
                    "roi_pct",
                    "yield_pct",
                    "max_drawdown_pct",
                    "roi_to_maxdd",
                    "bets",
                    "logloss",
                ]
            ]
            .head(12)
            .to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
