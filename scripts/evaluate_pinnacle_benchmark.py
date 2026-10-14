#!/usr/bin/env python3
"""Evaluate models and hybrid predictors against the Pinnacle sharp market benchmark.

This script performs a rigorous, auditable evaluation on matches where pre-match
Pinnacle odds are available. It calculates proper scoring rules, paired bootstrap
significance tests, disagreement matrices, and betting simulations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate models against Pinnacle sharp market quotes."
    )
    parser.add_argument(
        "--quotes",
        type=Path,
        default=PROJECT_ROOT / "data/oddspapi_lol_2026_model_audit/selected_pre_match_quotes.csv",
        help="Path to selected pre-match quotes CSV",
    )
    parser.add_argument(
        "--common",
        type=Path,
        default=PROJECT_ROOT / "reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv",
        help="Path to exp039_market_common.csv with features and predictions",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "reports/exp081_siamese_focal_uncertainty_gating/pinnacle_comparison.json",
        help="Path to save output JSON summary",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=5000,
        help="Number of bootstrap replicates for paired difference testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()

def calculate_ece(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error with uniform binning."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for i in range(n_bins):
        mask = (p_pred >= bins[i]) & (p_pred < bins[i + 1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(p_pred[mask])
            ece += (np.sum(mask) / total) * abs(bin_acc - bin_conf)
    return float(ece)


def main() -> int:
    args = parse_args()

    quotes_path = args.quotes
    if not quotes_path.exists():
        fallback = PROJECT_ROOT.parent / "inzynierka" / "data/oddspapi_lol_2026_model_audit/selected_pre_match_quotes.csv"
        if fallback.exists():
            quotes_path = fallback
        else:
            print(f"Error: quotes file not found: {args.quotes}", file=sys.stderr)
            return 1

    common_path = args.common
    if not common_path.exists():
        fallback = PROJECT_ROOT.parent / "inzynierka" / "reports/exp039_db_market_backtest_v3_corrected/exp039_market_common.csv"
        if fallback.exists():
            common_path = fallback
        else:
            print(f"Error: common file not found: {args.common}", file=sys.stderr)
            return 1

    print(f"Loading quotes from: {quotes_path}")
    df_quotes = pd.read_csv(quotes_path)
    pin_quotes = df_quotes[df_quotes["bookmaker"] == "pinnacle"].copy()
    print(f"Found {len(pin_quotes)} Pinnacle pre-match quotes.")

    print(f"Loading common dataset from: {common_path}")
    df_common = pd.read_csv(common_path)
    df = pin_quotes.merge(
        df_common,
        on="canonical_match_id",
        how="inner",
        suffixes=("_pin", "_common"),
    )
    n = len(df)
    print(f"Merged unified cohort size: {n} matches.")
    if n == 0:
        print("Error: 0 matches merged. Check canonical_match_id alignment.", file=sys.stderr)
        return 1

    y_true = df["winner_a"].values.astype(int)

    # Probabilities
    p_pin_novig = df["market_prob_a_novig"].values
    p_pin_raw = 1.0 / df["odds_a"].values
    p_pin_raw_norm = p_pin_raw / (p_pin_raw + 1.0 / df["odds_b"].values)

    p_mkt_open = df["market_open_p_a_novig"].values
    p_mkt_mid = df["market_mid_p_a_novig"].values
    p_mkt_close = df["market_close_p_a_novig"].values

    p_exp039_sym = df["exp039_symmetric_prob_team_a"].values
    p_exp039_par = df["exp039_parity_v2_prob_team_a"].values

    # Hybrids
    z_mod = logit(np.clip(p_exp039_sym, 1e-4, 1.0 - 1e-4))
    p_mod_t08 = expit(z_mod / 0.80)
    p_hybrid_lin_open = 0.35 * p_mod_t08 + 0.65 * p_mkt_open

    z_open = logit(np.clip(p_mkt_open, 1e-4, 1.0 - 1e-4))
    p_hybrid_bayes_open_35 = expit(0.35 * z_mod + 0.65 * z_open)
    p_hybrid_bayes_open_50 = expit(0.50 * z_mod + 0.50 * z_open)

    z_pin = logit(np.clip(p_pin_novig, 1e-4, 1.0 - 1e-4))
    p_hybrid_pin_lin = 0.35 * p_mod_t08 + 0.65 * p_pin_novig
    p_hybrid_pin_bayes_35 = expit(0.35 * z_mod + 0.65 * z_pin)
    p_hybrid_pin_bayes_50 = expit(0.50 * z_mod + 0.50 * z_pin)

    predictors: dict[str, np.ndarray] = {
        "Pinnacle No-Vig Fair": p_pin_novig,
        "Pinnacle Normalized Implied": p_pin_raw_norm,
        "Market Consensus Open (PL)": p_mkt_open,
        "Market Consensus Mid (PL)": p_mkt_mid,
        "Market Consensus Close (PL)": p_mkt_close,
        "EXP-039 Symmetric (Pure Model)": p_exp039_sym,
        "EXP-039 Parity v2 (Thesis Model)": p_exp039_par,
        "Operational Hybrid (Model + Market Open)": p_hybrid_lin_open,
        "Bayesian Hybrid a0.35 (Model + Market Open)": p_hybrid_bayes_open_35,
        "Bayesian Hybrid a0.50 (Model + Market Open)": p_hybrid_bayes_open_50,
        "Hybrid Linear (Model + Pinnacle No-Vig)": p_hybrid_pin_lin,
        "Bayesian Hybrid a0.35 (Model + Pinnacle No-Vig)": p_hybrid_pin_bayes_35,
        "Bayesian Hybrid a0.50 (Model + Pinnacle No-Vig)": p_hybrid_pin_bayes_50,
    }

    # Margins
    pin_margin = float(np.mean(df["market_margin"].values))
    open_margin = float(np.mean(df["market_open_avg_margin"].values))
    close_margin = float(np.mean(df["market_close_avg_margin"].values))

    # Coin flip mask
    coin_mask = (p_pin_novig >= 0.45) & (p_pin_novig <= 0.55)
    n_coin = int(np.sum(coin_mask))

    metrics_list = []
    for name, p in predictors.items():
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        ll = float(log_loss(y_true, p_c))
        brier = float(brier_score_loss(y_true, p_c))
        auc = float(roc_auc_score(y_true, p_c))
        acc = float(accuracy_score(y_true, (p_c >= 0.5).astype(int)))
        ece = calculate_ece(y_true, p_c)
        ll_coin = (
            float(log_loss(y_true[coin_mask], p_c[coin_mask]))
            if n_coin > 0
            else None
        )

        metrics_list.append(
            {
                "predictor": name,
                "log_loss": round(ll, 6),
                "brier": round(brier, 6),
                "roc_auc": round(auc, 6),
                "accuracy": round(acc, 6),
                "ece": round(ece, 6),
                "coin_flip_log_loss": round(ll_coin, 6) if ll_coin is not None else None,
            }
        )

    # Disagreement analysis
    fav_pin = (p_pin_novig >= 0.5).astype(int)
    fav_mod = (p_exp039_sym >= 0.5).astype(int)
    disagree_mask = fav_pin != fav_mod
    n_disagree = int(np.sum(disagree_mask))
    acc_pin_dis = (
        float(accuracy_score(y_true[disagree_mask], fav_pin[disagree_mask]))
        if n_disagree > 0
        else 0.0
    )
    acc_mod_dis = (
        float(accuracy_score(y_true[disagree_mask], fav_mod[disagree_mask]))
        if n_disagree > 0
        else 0.0
    )
    ll_pin_dis = (
        float(log_loss(y_true[disagree_mask], p_pin_novig[disagree_mask]))
        if n_disagree > 0
        else 0.0
    )
    ll_mod_dis = (
        float(log_loss(y_true[disagree_mask], p_exp039_sym[disagree_mask]))
        if n_disagree > 0
        else 0.0
    )

    # Paired Bootstrap
    np.random.seed(args.seed)
    B = args.bootstrap_reps
    idx = np.random.randint(0, n, size=(B, n))

    ll_pin_i = -(y_true * np.log(p_pin_novig) + (1 - y_true) * np.log(1 - p_pin_novig))
    ll_mod_i = -(y_true * np.log(p_exp039_sym) + (1 - y_true) * np.log(1 - p_exp039_sym))
    ll_close_i = -(y_true * np.log(p_mkt_close) + (1 - y_true) * np.log(1 - p_mkt_close))

    diff_mod_pin = ll_mod_i - ll_pin_i
    diff_close_pin = ll_close_i - ll_pin_i

    boot_mod_pin = np.mean(diff_mod_pin[idx], axis=1)
    boot_close_pin = np.mean(diff_close_pin[idx], axis=1)

    bootstrap_results = {
        "model_minus_pinnacle": {
            "mean_delta_log_loss": round(float(np.mean(diff_mod_pin)), 6),
            "ci_95_low": round(float(np.percentile(boot_mod_pin, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_mod_pin, 97.5)), 6),
            "p_value": round(float(np.mean(boot_mod_pin <= 0)), 6),
        },
        "pl_close_minus_pinnacle": {
            "mean_delta_log_loss": round(float(np.mean(diff_close_pin)), 6),
            "ci_95_low": round(float(np.percentile(boot_close_pin, 2.5)), 6),
            "ci_95_high": round(float(np.percentile(boot_close_pin, 97.5)), 6),
            "p_value": round(float(np.mean(boot_close_pin <= 0)), 6),
        },
    }

    # Best of breakdown
    best_of_results = {}
    for bo in [1, 3, 5]:
        sub = df[df["best_of"] == bo]
        if len(sub) > 0:
            y_sub = sub["winner_a"].values.astype(int)
            p_pin_sub = sub["market_prob_a_novig"].values
            p_mod_sub = sub["exp039_symmetric_prob_team_a"].values
            p_pl_sub = sub["market_close_p_a_novig"].values

            best_of_results[f"Bo{bo}"] = {
                "n": len(sub),
                "pinnacle_log_loss": round(float(log_loss(y_sub, p_pin_sub)), 4),
                "pinnacle_accuracy": round(
                    float(accuracy_score(y_sub, (p_pin_sub >= 0.5).astype(int))), 4
                ),
                "model_log_loss": round(float(log_loss(y_sub, p_mod_sub)), 4),
                "model_accuracy": round(
                    float(accuracy_score(y_sub, (p_mod_sub >= 0.5).astype(int))), 4
                ),
                "pl_close_log_loss": round(float(log_loss(y_sub, p_pl_sub)), 4),
            }

    # Betting simulation against Pinnacle odds
    odds_a = df["odds_a"].values
    odds_b = df["odds_b"].values
    ev_a = p_exp039_sym * odds_a - 1.0
    ev_b = (1.0 - p_exp039_sym) * odds_b - 1.0

    betting_results = {}
    for ev_thresh in [0.0, 0.03, 0.05, 0.08]:
        profits = []
        for i in range(n):
            if ev_a[i] >= ev_thresh and ev_a[i] > ev_b[i]:
                pnl = (odds_a[i] - 1.0) if y_true[i] == 1 else -1.0
                profits.append(pnl)
            elif ev_b[i] >= ev_thresh and ev_b[i] > ev_a[i]:
                pnl = (odds_b[i] - 1.0) if y_true[i] == 0 else -1.0
                profits.append(pnl)
        if profits:
            p_arr = np.array(profits)
            betting_results[f"ev_ge_{int(ev_thresh*100)}pct"] = {
                "n_bets": len(profits),
                "win_rate": round(float(np.mean(p_arr > 0)), 4),
                "roi": round(float(np.sum(p_arr) / len(profits)), 4),
                "total_pnl": round(float(np.sum(p_arr)), 2),
            }

    output_payload = {
        "cohort": {
            "n_matches": n,
            "date_min": str(df["match_start_at"].min())[:10],
            "date_max": str(df["match_start_at"].max())[:10],
            "pinnacle_mean_vig": round(pin_margin, 4),
            "polish_open_mean_vig": round(open_margin, 4),
            "polish_close_mean_vig": round(close_margin, 4),
        },
        "metrics": metrics_list,
        "disagreement_analysis": {
            "n_disagreements": n_disagree,
            "disagreement_pct": round(n_disagree / n, 4),
            "pinnacle_accuracy_on_disagreement": round(acc_pin_dis, 4),
            "model_accuracy_on_disagreement": round(acc_mod_dis, 4),
            "pinnacle_log_loss_on_disagreement": round(ll_pin_dis, 4),
            "model_log_loss_on_disagreement": round(ll_mod_dis, 4),
        },
        "paired_bootstrap": bootstrap_results,
        "best_of_breakdown": best_of_results,
        "betting_against_pinnacle": betting_results,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\nSaved structured comparison to: {args.output_json}")

    # Pretty print summary table
    df_print = pd.DataFrame(metrics_list)
    print("\n--- SUMMARY METRICS TABLE ---")
    print(df_print.to_string(index=False))

    print("\n--- PAIRED BOOTSTRAP (5000 repl.) ---")
    print(
        f"Delta LogLoss (Model - Pinnacle): +{bootstrap_results['model_minus_pinnacle']['mean_delta_log_loss']:.4f} "
        f"[95% CI: {bootstrap_results['model_minus_pinnacle']['ci_95_low']:.4f}, {bootstrap_results['model_minus_pinnacle']['ci_95_high']:.4f}], "
        f"p-val = {bootstrap_results['model_minus_pinnacle']['p_value']:.4f}"
    )
    print(
        f"Delta LogLoss (PL Close - Pinnacle): +{bootstrap_results['pl_close_minus_pinnacle']['mean_delta_log_loss']:.4f} "
        f"[95% CI: {bootstrap_results['pl_close_minus_pinnacle']['ci_95_low']:.4f}, {bootstrap_results['pl_close_minus_pinnacle']['ci_95_high']:.4f}], "
        f"p-val = {bootstrap_results['pl_close_minus_pinnacle']['p_value']:.4f}"
    )

    print("\n--- DISAGREEMENTS (N={}) ---".format(n_disagree))
    print(
        f"Pinnacle Acc: {acc_pin_dis*100:.1f}% (LL: {ll_pin_dis:.4f}) | "
        f"Model Acc: {acc_mod_dis*100:.1f}% (LL: {ll_mod_dis:.4f})"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
