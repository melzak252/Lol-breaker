"""EXP-040 Candidate training, calibration, and benchmarking pipeline.

This script benchmarks next-generation candidate improvements (temperature scaling,
beta calibration, uncertainty gating, and candidate feature interactions) against
the frozen EXP-039 baseline on the 622-match historical market cohort.

Never overwrites frozen exp-039 artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from betting_app.ml.calibration.candidate_calibration import (
    BetaCalibrator,
    TemperatureScalingCalibrator,
    UncertaintyGatedCalibrator,
    brier_score_decomposition,
    expected_calibration_error,
)
from betting_app.ml.features.candidate_features import (
    assemble_symmetric_candidate_features,
    compute_patch_decay_weights,
    compute_roster_continuity,
    compute_side_advantage,
)

logger = logging.getLogger(__name__)

DEFAULT_COHORT_PATH = Path("reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv")
DEFAULT_ODDSPAPI_PATH = Path("data/oddspapi_lol_2026_model_audit/selected_pre_match_quotes.csv")


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(int) == y_true))


def simulate_betting(
    y_true: np.ndarray,
    p_model: np.ndarray,
    odds_a: np.ndarray,
    odds_b: np.ndarray,
    min_edge: float = 0.05,
    tax_rate: float = 0.12,
) -> dict[str, Any]:
    bets = 0
    wins = 0
    profit_gross = 0.0
    profit_net = 0.0

    for i in range(len(y_true)):
        pm = p_model[i]
        oa = odds_a[i]
        ob = odds_b[i]
        y = y_true[i]

        edge_a = pm - (1.0 / oa)
        edge_b = (1.0 - pm) - (1.0 / ob)

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


def run_benchmark(
    cohort_path: Path = DEFAULT_COHORT_PATH,
    oddspapi_path: Path = DEFAULT_ODDSPAPI_PATH,
) -> dict[str, Any]:
    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    df = pd.read_csv(cohort_path)
    y_true = (df["winner_side"] == "team_a").astype(int).values
    p_exp039 = df["exp039_parity_v2_prob_team_a"].values
    p_mkt_open = df["market_open_p_a_novig"].values
    p_mkt_close = df["market_close_p_a_novig"].values

    odds_close_a = 1.0 / np.clip(df["market_close_p_a_raw"].values, 0.01, 0.99)
    odds_close_b = 1.0 / np.clip(
        1.0 + df["market_close_avg_margin"].values - df["market_close_p_a_raw"].values,
        0.01,
        2.0,
    )

    odds_open_a = 1.0 / np.clip(df["market_open_p_a_raw"].values, 0.01, 0.99)
    odds_open_b = 1.0 / np.clip(
        1.0 + df["market_open_avg_margin"].values - df["market_open_p_a_raw"].values,
        0.01,
        2.0,
    )

    # 1. Evaluate baseline EXP-039
    exp039_metrics = {
        "log_loss": binary_log_loss(y_true, p_exp039),
        "brier": brier_score(y_true, p_exp039),
        "accuracy": accuracy_score(y_true, p_exp039),
        "ece": expected_calibration_error(y_true, p_exp039, n_bins=10),
        "brier_decomp": brier_score_decomposition(y_true, p_exp039),
        "betting_close_edge_05": simulate_betting(y_true, p_exp039, odds_close_a, odds_close_b, min_edge=0.05),
        "betting_open_edge_05": simulate_betting(y_true, p_exp039, odds_open_a, odds_open_b, min_edge=0.05),
    }

    # 2. Fit nested Out-of-Fold Temperature Scaling
    # To prevent leakage, we use K-fold chronological blocks to fit T
    z_exp039 = logit(np.clip(p_exp039, 1e-4, 1.0 - 1e-4))
    n = len(df)
    n_splits = 5
    oof_p_temp = np.zeros(n)
    temperatures = []

    split_size = n // n_splits
    for fold in range(n_splits):
        val_idx = np.arange(fold * split_size, (fold + 1) * split_size if fold < n_splits - 1 else n)
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        temp_cal = TemperatureScalingCalibrator()
        temp_cal.fit(z_exp039[train_idx], y_true[train_idx])
        oof_p_temp[val_idx] = temp_cal.transform(z_exp039[val_idx])
        temperatures.append(temp_cal.temperature_)

    avg_temperature = float(np.mean(temperatures))

    exp040_temp_metrics = {
        "avg_temperature": avg_temperature,
        "log_loss": binary_log_loss(y_true, oof_p_temp),
        "brier": brier_score(y_true, oof_p_temp),
        "accuracy": accuracy_score(y_true, oof_p_temp),
        "ece": expected_calibration_error(y_true, oof_p_temp, n_bins=10),
        "brier_decomp": brier_score_decomposition(y_true, oof_p_temp),
        "betting_close_edge_05": simulate_betting(y_true, oof_p_temp, odds_close_a, odds_close_b, min_edge=0.05),
        "betting_open_edge_05": simulate_betting(y_true, oof_p_temp, odds_open_a, odds_open_b, min_edge=0.05),
    }

    # 3. Fit Beta Calibration (Kull et al., 2017)
    oof_p_beta = np.zeros(n)
    for fold in range(n_splits):
        val_idx = np.arange(fold * split_size, (fold + 1) * split_size if fold < n_splits - 1 else n)
        train_idx = np.setdiff1d(np.arange(n), val_idx)

        beta_cal = BetaCalibrator()
        beta_cal.fit(p_exp039[train_idx], y_true[train_idx])
        oof_p_beta[val_idx] = beta_cal.transform(p_exp039[val_idx])

    exp040_beta_metrics = {
        "log_loss": binary_log_loss(y_true, oof_p_beta),
        "brier": brier_score(y_true, oof_p_beta),
        "accuracy": accuracy_score(y_true, oof_p_beta),
        "ece": expected_calibration_error(y_true, oof_p_beta, n_bins=10),
        "betting_close_edge_05": simulate_betting(y_true, oof_p_beta, odds_close_a, odds_close_b, min_edge=0.05),
        "betting_open_edge_05": simulate_betting(y_true, oof_p_beta, odds_open_a, odds_open_b, min_edge=0.05),
    }

    # 4. Uncertainty & Market Discrepancy Gating
    # Shrink extreme probabilities toward market consensus when model diverges excessively
    gated_cal = UncertaintyGatedCalibrator(discrepancy_threshold=0.18, max_shrinkage=0.45)
    nominal_sigma = np.full_like(oof_p_temp, 1.8)
    p_gated_close = gated_cal.calibrate(oof_p_temp, nominal_sigma, nominal_sigma, p_market=p_mkt_close)
    p_gated_open = gated_cal.calibrate(oof_p_temp, nominal_sigma, nominal_sigma, p_market=p_mkt_open)

    exp040_gated_metrics = {
        "log_loss_close_context": binary_log_loss(y_true, p_gated_close),
        "brier_close_context": brier_score(y_true, p_gated_close),
        "ece_close_context": expected_calibration_error(y_true, p_gated_close, n_bins=10),
        "betting_close_edge_05": simulate_betting(y_true, p_gated_close, odds_close_a, odds_close_b, min_edge=0.05),
        "betting_open_edge_05": simulate_betting(y_true, p_gated_open, odds_open_a, odds_open_b, min_edge=0.05),
    }

    # 5. Hybrid candidate: 0.25 Calibrated Model + 0.75 Market Reference
    p_hybrid_close = 0.25 * oof_p_temp + 0.75 * p_mkt_close
    p_hybrid_open = 0.35 * oof_p_temp + 0.65 * p_mkt_open

    hybrid_candidate_metrics = {
        "log_loss_close": binary_log_loss(y_true, p_hybrid_close),
        "brier_close": brier_score(y_true, p_hybrid_close),
        "accuracy_close": accuracy_score(y_true, p_hybrid_close),
        "ece_close": expected_calibration_error(y_true, p_hybrid_close, n_bins=10),
        "betting_close_edge_04": simulate_betting(y_true, p_hybrid_close, odds_close_a, odds_close_b, min_edge=0.04),
        "betting_open_edge_04": simulate_betting(y_true, p_hybrid_open, odds_open_a, odds_open_b, min_edge=0.04),
    }

    report = {
        "cohort_size": n,
        "baseline_exp039": exp039_metrics,
        "candidate_temperature_scaling": exp040_temp_metrics,
        "candidate_beta_calibration": exp040_beta_metrics,
        "candidate_uncertainty_gated": exp040_gated_metrics,
        "candidate_hybrid": hybrid_candidate_metrics,
    }

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_benchmark()
    print("\n=======================================================")
    print("        EXP-040 CANDIDATE vs EXP-039 BENCHMARK        ")
    print("=======================================================")
    print(f"Cohort size: {report['cohort_size']} matches\n")

    b = report["baseline_exp039"]
    t = report["candidate_temperature_scaling"]
    beta = report["candidate_beta_calibration"]
    g = report["candidate_uncertainty_gated"]
    h = report["candidate_hybrid"]

    print("1. PROBABILISTIC QUALITY (Lower is better)")
    print(f"{'Metric':<18} | {'EXP-039 Baseline':<18} | {'EXP-040 TempScaled':<20} | {'EXP-040 BetaCal':<18} | {'EXP-040 Hybrid':<18}")
    print("-" * 105)
    print(f"{'Log Loss':<18} | {b['log_loss']:<18.4f} | {t['log_loss']:<20.4f} | {beta['log_loss']:<18.4f} | {h['log_loss_close']:<18.4f}")
    print(f"{'Brier Score':<18} | {b['brier']:<18.4f} | {t['brier']:<20.4f} | {beta['brier']:<18.4f} | {h['brier_close']:<18.4f}")
    print(f"{'ECE (Calib Error)':<18} | {b['ece']:<18.4f} | {t['ece']:<20.4f} | {beta['ece']:<18.4f} | {h['ece_close']:<18.4f}")

    print("\n2. CALIBRATION METRICS")
    print(f"Average Fitted Temperature T: {t['avg_temperature']:.3f} (T > 1.0 proves model overconfidence was flattened)")
    print(f"Reliability (Miscalibration Penalty): EXP-039={b['brier_decomp']['reliability']:.5f} -> TempScaled={t['brier_decomp']['reliability']:.5f}")

    print("\n3. BETTING REALITY UNDER POLISH 12% TAX (Edge > 5%)")
    print(f"{'Model/Strategy':<25} | {'Bets':<5} | {'WinRate':<8} | {'ROI Gross':<10} | {'ROI Net (12% Tax)':<18}")
    print("-" * 75)
    bc = b["betting_close_edge_05"]
    tc = t["betting_close_edge_05"]
    gc = g["betting_close_edge_05"]
    hc = h["betting_close_edge_04"]
    print(f"{'EXP-039 Baseline (Close)':<25} | {bc['bets']:<5d} | {bc['winrate']:<7.1f}% | {bc['roi_gross']:<9.2f}% | {bc['roi_net_tax12']:<17.2f}%")
    print(f"{'EXP-040 TempScaled (Close)':<25} | {tc['bets']:<5d} | {tc['winrate']:<7.1f}% | {tc['roi_gross']:<9.2f}% | {tc['roi_net_tax12']:<17.2f}%")
    print(f"{'EXP-040 UncertaintyGated':<25} | {gc['bets']:<5d} | {gc['winrate']:<7.1f}% | {gc['roi_gross']:<9.2f}% | {gc['roi_net_tax12']:<17.2f}%")
    print(f"{'EXP-040 Hybrid (Edge>4%)':<25} | {hc['bets']:<5d} | {hc['winrate']:<7.1f}% | {hc['roi_gross']:<9.2f}% | {hc['roi_net_tax12']:<17.2f}%")


if __name__ == "__main__":
    main()
