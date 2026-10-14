"""Comprehensive benchmark evaluating the 4 architectural rebuild paradigms on the 622-match cohort:

1. Baseline EXP-039 (frozen)
2. Temperature Scaling (logit flattening)
3. Venn-Abers Conformal Calibration (finite-sample calibrated multiprobability bounds)
4. Hierarchical Markov Series Simulation (dynamic rotating side selection for Bo3/Bo5)
5. Market Residual Learning (predicting unpriced market error y - P_market)
6. Conformal Risk Control (P_low lower-bound gating under 12% turnover tax)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from betting_app.ml.calibration.candidate_calibration import (
    TemperatureScalingCalibrator,
    brier_score_decomposition,
    expected_calibration_error,
)
from betting_app.ml.calibration.venn_abers import (
    ConformalRiskGater,
    VennAbersCalibrator,
)
from betting_app.ml.models.markov_series import MarkovSeriesSimulator
from betting_app.ml.models.market_residual import MarketResidualModel, ResidualEdgeDetector

logger = logging.getLogger(__name__)

COHORT_PATH = Path("reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv")


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def simulate_fixed_stake_betting(
    y_true: np.ndarray,
    p_action: np.ndarray,
    odds_a: np.ndarray,
    odds_b: np.ndarray,
    min_edge: float = 0.04,
    tax_rate: float = 0.12,
    p_lower_a: np.ndarray | None = None,
    use_conformal_gate: bool = False,
) -> dict[str, Any]:
    bets = 0
    wins = 0
    profit_gross = 0.0
    profit_net = 0.0

    for i in range(len(y_true)):
        oa = odds_a[i]
        ob = odds_b[i]
        y = y_true[i]
        pm = p_action[i]

        edge_a = pm - (1.0 / oa)
        edge_b = (1.0 - pm) - (1.0 / ob)

        # Conformal gate condition: even lower bound must yield positive EV
        if use_conformal_gate and p_lower_a is not None:
            pla = p_lower_a[i]
            plb = 1.0 - pm  # symmetrical bound
            ev_low_a = pla * (oa * (1.0 - tax_rate)) - 1.0
            ev_low_b = plb * (ob * (1.0 - tax_rate)) - 1.0
            if ev_low_a <= 0.0:
                edge_a = -1.0
            if ev_low_b <= 0.0:
                edge_b = -1.0

        if edge_a > min_edge and edge_a > edge_b:
            bets += 1
            if y == 1:
                wins += 1
                profit_gross += oa - 1.0
                profit_net += oa * (1.0 - tax_rate) - 1.0
            else:
                profit_gross -= 1.0
                profit_net -= 1.0
        elif edge_b > min_edge and edge_b > edge_a:
            bets += 1
            if y == 0:
                wins += 1
                profit_gross += ob - 1.0
                profit_net += ob * (1.0 - tax_rate) - 1.0
            else:
                profit_gross -= 1.0
                profit_net -= 1.0

    return {
        "bets": bets,
        "wins": wins,
        "winrate": round(wins / bets * 100, 2) if bets > 0 else 0.0,
        "roi_gross": round(profit_gross / bets * 100, 2) if bets > 0 else 0.0,
        "roi_net_tax12": round(profit_net / bets * 100, 2) if bets > 0 else 0.0,
    }


def run_full_rebuild_benchmark() -> dict[str, Any]:
    df = pd.read_csv(COHORT_PATH)
    n = len(df)
    y_true = (df["winner_side"] == "team_a").astype(int).values
    p_exp039 = df["exp039_parity_v2_prob_team_a"].values
    p_close = df["market_close_p_a_novig"].values
    p_open = df["market_open_p_a_novig"].values

    odds_close_a = 1.0 / np.clip(df["market_close_p_a_raw"].values, 0.01, 0.99)
    odds_close_b = 1.0 / np.clip(
        1.0 + df["market_close_avg_margin"].values - df["market_close_p_a_raw"].values,
        0.01,
        2.0,
    )

    # 1. Baseline EXP-039
    b_ll = log_loss(y_true, p_exp039)
    b_brier = brier_score(y_true, p_exp039)
    b_ece = expected_calibration_error(y_true, p_exp039, n_bins=10)
    b_bets = simulate_fixed_stake_betting(y_true, p_exp039, odds_close_a, odds_close_b, min_edge=0.04)

    # 2. Temperature Scaling
    z_exp039 = logit(np.clip(p_exp039, 1e-4, 1.0 - 1e-4))
    ts_cal = TemperatureScalingCalibrator()
    ts_cal.fit(z_exp039, y_true)
    p_ts = ts_cal.transform(z_exp039)
    ts_ll = log_loss(y_true, p_ts)
    ts_brier = brier_score(y_true, p_ts)
    ts_ece = expected_calibration_error(y_true, p_ts, n_bins=10)
    ts_bets = simulate_fixed_stake_betting(y_true, p_ts, odds_close_a, odds_close_b, min_edge=0.04)

    # 3. Venn-Abers Conformal Calibration (5-fold out-of-fold)
    n_splits = 5
    split_size = n // n_splits
    p_va = np.zeros(n)
    p0_va = np.zeros(n)
    p1_va = np.zeros(n)

    for fold in range(n_splits):
        val_idx = np.arange(fold * split_size, (fold + 1) * split_size if fold < n_splits - 1 else n)
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        va = VennAbersCalibrator()
        # Train on hybrid scores (0.35 model + 0.65 market close)
        score_train = 0.35 * p_exp039[train_idx] + 0.65 * p_close[train_idx]
        score_val = 0.35 * p_exp039[val_idx] + 0.65 * p_close[val_idx]

        va.fit(score_train, y_true[train_idx])
        p_pt, p_low, p_up = va.predict_intervals(score_val)
        p_va[val_idx] = p_pt
        p0_va[val_idx] = p_low
        p1_va[val_idx] = p_up

    va_ll = log_loss(y_true, p_va)
    va_brier = brier_score(y_true, p_va)
    va_ece = expected_calibration_error(y_true, p_va, n_bins=10)
    va_bets = simulate_fixed_stake_betting(y_true, p_va, odds_close_a, odds_close_b, min_edge=0.04)

    # 4. Conformal Risk Control (using p0_va as lower bound gate)
    crc_bets = simulate_fixed_stake_betting(
        y_true,
        p_va,
        odds_close_a,
        odds_close_b,
        min_edge=0.04,
        p_lower_a=p0_va,
        use_conformal_gate=True,
    )

    # 5. Hierarchical Markov Series Simulator (Bo3 with rotating side selection)
    markov_sim = MarkovSeriesSimulator()
    best_of_arr = df["best_of"].fillna(3).astype(int).values
    p_markov = np.zeros(n)
    for i in range(n):
        bo = best_of_arr[i]
        # Neutral game win probability approximated from model
        p_markov[i] = markov_sim.predict_series_proba(
            p_neutral_a=p_exp039[i],
            team_a_has_game1_priority=True,  # seed priority
            best_of=bo,
            blue_side_bonus=0.22,
        )
    # Calibrate markov series with Venn-Abers
    p_markov_va = np.zeros(n)
    for fold in range(n_splits):
        val_idx = np.arange(fold * split_size, (fold + 1) * split_size if fold < n_splits - 1 else n)
        train_idx = np.setdiff1d(np.arange(n), val_idx)
        va = VennAbersCalibrator()
        va.fit(p_markov[train_idx], y_true[train_idx])
        p_markov_va[val_idx] = va.predict_proba(p_markov[val_idx])[:, 1]

    markov_ll = log_loss(y_true, p_markov_va)
    markov_brier = brier_score(y_true, p_markov_va)
    markov_ece = expected_calibration_error(y_true, p_markov_va, n_bins=10)

    # 6. Market Residual Learning Model
    # Feature diff: model - market
    X_diff = (p_exp039 - p_close).reshape(-1, 1)
    res_model = MarketResidualModel(alpha=50.0, max_residual=0.10)
    p_res = np.zeros(n)

    for fold in range(n_splits):
        val_idx = np.arange(fold * split_size, (fold + 1) * split_size if fold < n_splits - 1 else n)
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        res_model.fit(X_diff[train_idx], y_true[train_idx], p_market=p_close[train_idx])
        p_res[val_idx] = res_model.predict_proba(X_diff[val_idx], p_market=p_close[val_idx])

    mr_ll = log_loss(y_true, p_res)
    mr_brier = brier_score(y_true, p_res)
    mr_ece = expected_calibration_error(y_true, p_res, n_bins=10)
    mr_bets = simulate_fixed_stake_betting(y_true, p_res, odds_close_a, odds_close_b, min_edge=0.04)

    return {
        "cohort_size": n,
        "exp039_baseline": {"log_loss": b_ll, "brier": b_brier, "ece": b_ece, "bets": b_bets},
        "temperature_scaling": {"log_loss": ts_ll, "brier": ts_brier, "ece": ts_ece, "bets": ts_bets},
        "venn_abers": {"log_loss": va_ll, "brier": va_brier, "ece": va_ece, "bets": va_bets},
        "markov_series": {"log_loss": markov_ll, "brier": markov_brier, "ece": markov_ece},
        "market_residual": {"log_loss": mr_ll, "brier": mr_brier, "ece": mr_ece, "bets": mr_bets},
        "conformal_risk_gater": {"bets": crc_bets},
    }


def main() -> None:
    res = run_full_rebuild_benchmark()
    print("\n" + "=" * 80)
    print("      UNIFIED ARCHITECTURAL REBUILD BENCHMARK (622 MATCHES)      ")
    print("=" * 80)

    print("\n1. CALIBRATION ERROR (ECE) AND LOSS COMPARISON:")
    print(f"{'Architecture':<32} | {'LogLoss':<10} | {'Brier':<10} | {'ECE (Calib Error)':<18}")
    print("-" * 75)
    print(f"{'0. EXP-039 Frozen Baseline':<32} | {res['exp039_baseline']['log_loss']:<10.4f} | {res['exp039_baseline']['brier']:<10.4f} | {res['exp039_baseline']['ece']:<18.4f}")
    print(f"{'1. Temperature Scaling':<32} | {res['temperature_scaling']['log_loss']:<10.4f} | {res['temperature_scaling']['brier']:<10.4f} | {res['temperature_scaling']['ece']:<18.4f}")
    print(f"{'2. Markov Bo3/Bo5 Simulator':<32} | {res['markov_series']['log_loss']:<10.4f} | {res['markov_series']['brier']:<10.4f} | {res['markov_series']['ece']:<18.4f}")
    print(f"{'3. Market Residual Model':<32} | {res['market_residual']['log_loss']:<10.4f} | {res['market_residual']['brier']:<10.4f} | {res['market_residual']['ece']:<18.4f}")
    print(f"{'4. Venn-Abers Conformal Predictor':<32} | {res['venn_abers']['log_loss']:<10.4f} | {res['venn_abers']['brier']:<10.4f} | {res['venn_abers']['ece']:<18.4f}")

    print("\n2. BETTING PERFORMANCE UNDER POLISH 12% TURNOVER TAX (Edge > 4%):")
    print(f"{'Strategy / Gater':<32} | {'Bets':<6} | {'WinRate':<8} | {'ROI Gross':<10} | {'ROI Net (12% Tax)':<18}")
    print("-" * 80)
    b0 = res['exp039_baseline']['bets']
    b_mr = res['market_residual']['bets']
    b_va = res['venn_abers']['bets']
    b_crc = res['conformal_risk_gater']['bets']
    print(f"{'0. EXP-039 Baseline (Unfiltered)':<32} | {b0['bets']:<6d} | {b0['winrate']:<7.1f}% | {b0['roi_gross']:<9.2f}% | {b0['roi_net_tax12']:<17.2f}%")
    print(f"{'1. Market Residual Model':<32} | {b_mr['bets']:<6d} | {b_mr['winrate']:<7.1f}% | {b_mr['roi_gross']:<9.2f}% | {b_mr['roi_net_tax12']:<17.2f}%")
    print(f"{'2. Venn-Abers Point Estimate':<32} | {b_va['bets']:<6d} | {b_va['winrate']:<7.1f}% | {b_va['roi_gross']:<9.2f}% | {b_va['roi_net_tax12']:<17.2f}%")
    print(f"{'3. Conformal Risk Gate (P_low)':<32} | {b_crc['bets']:<6d} | {b_crc['winrate']:<7.1f}% | {b_crc['roi_gross']:<9.2f}% | {b_crc['roi_net_tax12']:<17.2f}%")


if __name__ == "__main__":
    main()
