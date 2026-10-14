"""Chronological evaluation of Adaptive Horizon Alpha on the thesis common sample.

Compares:
1. Pure Model (metamodel_lgbm_prob / thesis stacker)
2. Pure Market Open (wisdom of crowd at opening)
3. Pure Market Close (wisdom of crowd at closing)
4. Static Hybrid (alpha = 0.50)
5. Static Hybrid (alpha = 0.35)
6. Adaptive Horizon Hybrid:
   - When evaluating opening line (>24h): alpha_open = 0.55
   - When evaluating closing line (<2h): alpha_close = 0.25
   - Measures LogLoss, Brier, ECE, Accuracy
   - Measures CLV (Closing Line Value %) and flat-stake ROI (12% tax, min EV 5%)
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)


def calc_ece(y_t: np.ndarray, p_arr: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for l, r in zip(edges[:-1], edges[1:]):
        mask = (p_arr >= l) & (p_arr <= r if r == 1.0 else p_arr < r)
        if np.any(mask):
            ece += (mask.sum() / len(y_t)) * abs(y_t[mask].mean() - p_arr[mask].mean())
    return float(ece)


def eval_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = clip(p)
    return {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "ece": calc_ece(y, p),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def clv_and_betting_simulation(
    df: pd.DataFrame,
    prob_col: str,
    *,
    net_mult: float = 0.88,
    min_ev: float = 0.05,
) -> dict[str, Any]:
    """Simulate opening bets and measure Closing Line Value (CLV)."""
    clv_list: list[float] = []
    bets: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        p = row[prob_col]
        ev1 = (row["max_open_t1"] * net_mult * p) - 1.0
        ev2 = (row["max_open_t2"] * net_mult * (1.0 - p)) - 1.0

        p_close = row["prob_market_close"]
        fair_close_1 = 1.0 / p_close
        fair_close_2 = 1.0 / (1.0 - p_close)

        y = row["y_true"]

        if ev1 >= min_ev:
            clv = (row["max_open_t1"] / fair_close_1) - 1.0
            clv_list.append(clv)
            ret = net_mult * row["max_open_t1"] - 1.0 if y == 1 else -1.0
            bets.append({"ret": ret, "win": int(y == 1), "ev": ev1})
        elif ev2 >= min_ev:
            clv = (row["max_open_t2"] / fair_close_2) - 1.0
            clv_list.append(clv)
            ret = net_mult * row["max_open_t2"] - 1.0 if y == 0 else -1.0
            bets.append({"ret": ret, "win": int(y == 0), "ev": ev2})

    n_bets = len(bets)
    if n_bets == 0:
        return {"n_bets": 0, "roi": 0.0, "avg_clv_pct": 0.0, "beat_close_rate": 0.0, "win_rate": 0.0}

    total_profit = sum(b["ret"] for b in bets)
    roi = total_profit / n_bets
    avg_clv = float(np.mean(clv_list) * 100.0)
    beat_close = float(np.mean([1 if x > 0 else 0 for x in clv_list]) * 100.0)
    win_rate = float(np.mean([b["win"] for b in bets]) * 100.0)

    return {
        "n_bets": n_bets,
        "roi_pct": roi * 100.0,
        "avg_clv_pct": avg_clv,
        "beat_close_rate_pct": beat_close,
        "win_rate_pct": win_rate,
        "total_profit_u": total_profit,
    }


def main():
    print("Loading data...")
    df_meta = pd.read_csv("/home/melzak/dev/inzynierka/data/golgg_stacking_results.csv")
    df_odds = pd.read_csv("/home/melzak/dev/inzynierka/data/odds.csv")

    df_meta["golgg_match_id"] = df_meta["golgg_match_id"].astype(str)
    df_odds["golgg_match_id"] = df_odds["golgg_match_id"].astype(str)

    # Market no-vig probabilities
    df_odds["prob_market_open"] = (1.0 / df_odds["avg_open_home"]) / (
        (1.0 / df_odds["avg_open_home"]) + (1.0 / df_odds["avg_open_away"])
    )
    df_odds["prob_market_close"] = (1.0 / df_odds["avg_odds_home"]) / (
        (1.0 / df_odds["avg_odds_home"]) + (1.0 / df_odds["avg_odds_away"])
    )

    bookies = ["sts", "superbet", "betclic", "efortuna", "lv_bet", "betfan"]
    open_t1_cols = [f"odds1_{b}_open" for b in bookies if f"odds1_{b}_open" in df_odds.columns]
    open_t2_cols = [f"odds2_{b}_open" for b in bookies if f"odds2_{b}_open" in df_odds.columns]
    df_odds["max_open_t1"] = df_odds[open_t1_cols].max(axis=1)
    df_odds["max_open_t2"] = df_odds[open_t2_cols].max(axis=1)

    df = pd.merge(
        df_meta[["golgg_match_id", "metamodel_lgbm_prob", "y_true", "date"]],
        df_odds[["golgg_match_id", "prob_market_open", "prob_market_close", "max_open_t1", "max_open_t2", "avg_odds_home", "avg_odds_away"]],
        on="golgg_match_id",
    )
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])

    # Test Cohort 2024+
    test_df = df[df["date"] >= "2024-01-01"].sort_values("date").reset_index(drop=True)
    print(f"Test cohort (2024+): N = {len(test_df):,} matches")

    # Construct hybrid probabilities
    # 1. Static alphas
    test_df["prob_hybrid_static_50_open"] = 0.50 * test_df["metamodel_lgbm_prob"] + 0.50 * test_df["prob_market_open"]
    test_df["prob_hybrid_static_35_open"] = 0.35 * test_df["metamodel_lgbm_prob"] + 0.65 * test_df["prob_market_open"]
    test_df["prob_hybrid_static_50_close"] = 0.50 * test_df["metamodel_lgbm_prob"] + 0.50 * test_df["prob_market_close"]
    test_df["prob_hybrid_static_35_close"] = 0.35 * test_df["metamodel_lgbm_prob"] + 0.65 * test_df["prob_market_close"]

    # 2. Adaptive Horizon Alpha
    # Open: higher alpha (0.55) -> model has more edge when bookies open early lines
    # Close: lower alpha (0.25) -> closing line is sharp, model weight should shrink
    test_df["prob_hybrid_adaptive_open"] = 0.55 * test_df["metamodel_lgbm_prob"] + 0.45 * test_df["prob_market_open"]
    test_df["prob_hybrid_adaptive_close"] = 0.25 * test_df["metamodel_lgbm_prob"] + 0.75 * test_df["prob_market_close"]

    y = test_df["y_true"].to_numpy(dtype=int)

    forecast_models = {
        "Pure Model": test_df["metamodel_lgbm_prob"].to_numpy(),
        "Pure Market Open": test_df["prob_market_open"].to_numpy(),
        "Pure Market Close": test_df["prob_market_close"].to_numpy(),
        "Static Hybrid (a=0.50) @ Open": test_df["prob_hybrid_static_50_open"].to_numpy(),
        "Static Hybrid (a=0.35) @ Open": test_df["prob_hybrid_static_35_open"].to_numpy(),
        "Adaptive Hybrid (a=0.55) @ Open": test_df["prob_hybrid_adaptive_open"].to_numpy(),
        "Static Hybrid (a=0.50) @ Close": test_df["prob_hybrid_static_50_close"].to_numpy(),
        "Static Hybrid (a=0.35) @ Close": test_df["prob_hybrid_static_35_close"].to_numpy(),
        "Adaptive Hybrid (a=0.25) @ Close": test_df["prob_hybrid_adaptive_close"].to_numpy(),
    }

    f_rows = []
    for name, p in forecast_models.items():
        m = eval_metrics(y, p)
        f_rows.append({"model": name, **m})
    f_df = pd.DataFrame(f_rows)

    # Financial / CLV Simulation on Opening Lines
    strategies = {
        "Pure Model": "metamodel_lgbm_prob",
        "Static Hybrid (a=0.50)": "prob_hybrid_static_50_open",
        "Static Hybrid (a=0.35)": "prob_hybrid_static_35_open",
        "Adaptive Hybrid (a=0.55)": "prob_hybrid_adaptive_open",
    }

    s_rows = []
    for name, col in strategies.items():
        res = clv_and_betting_simulation(test_df, col, net_mult=0.88, min_ev=0.05)
        s_rows.append({"strategy": name, **res})
    s_df = pd.DataFrame(s_rows)

    print("\n" + "=" * 80)
    print("PROBABILISTIC ACCURACY ON COMMON COHORT (2024-2026)")
    print("=" * 80)
    print(f_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("CLV AND EV BETTING SIMULATION (Tax 12%, Min EV +5%)")
    print("=" * 80)
    print(s_df.to_string(index=False))


if __name__ == "__main__":
    main()
