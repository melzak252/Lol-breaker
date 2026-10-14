"""Expanding-window walk-forward cross-validation for Total Kills Distribution (IDEA-018)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import text
import statsmodels.api as sm
import statsmodels.formula.api as smf

from betting_app.core.db import get_session
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.kill_distribution_model import KillDistributionModel


def generate_quarterly_folds() -> list[tuple[str, str, str]]:
    """Define expanding-window walk-forward boundaries from 2024 to 2026."""
    return [
        ("Fold 1 (2024-Q1)", "2024-01-01", "2024-04-01"),
        ("Fold 2 (2024-Q2)", "2024-04-01", "2024-07-01"),
        ("Fold 3 (2024-Q3)", "2024-07-01", "2024-10-01"),
        ("Fold 4 (2024-Q4)", "2024-10-01", "2025-01-01"),
        ("Fold 5 (2025-Q1)", "2025-01-01", "2025-04-01"),
        ("Fold 6 (2025-Q2)", "2025-04-01", "2025-07-01"),
        ("Fold 7 (2025-Q3)", "2025-07-01", "2025-10-01"),
        ("Fold 8 (2025-Q4)", "2025-10-01", "2026-01-01"),
        ("Fold 9 (2026-YTD)", "2026-01-01", "2026-09-06"),
    ]


def compute_discrete_crps(actual_kills: np.ndarray, mus: np.ndarray, alphas: np.ndarray, max_k: int = 80) -> float:
    """Compute mean CRPS for Negative Binomial distribution predictions."""
    k_range = np.arange(0, max_k + 1)
    crps_values = []
    for y_true, mu, alpha in zip(actual_kills, mus, alphas):
        n = 1.0 / alpha
        p = 1.0 / (1.0 + alpha * mu)
        cdf = stats.nbinom.cdf(k_range, n, p)
        heaviside = (k_range >= y_true).astype(float)
        crps_values.append(np.sum((cdf - heaviside) ** 2))
    return float(np.mean(crps_values))


def compute_nll(actual_kills: np.ndarray, mus: np.ndarray, alphas: np.ndarray) -> float:
    """Compute mean Negative Log-Likelihood of actual discrete kill counts."""
    n_params = 1.0 / alphas
    p_params = 1.0 / (1.0 + alphas * mus)
    log_pmfs = stats.nbinom.logpmf(actual_kills, n_params, p_params)
    return float(-np.mean(log_pmfs))


def compute_quantile_coverage(actual_kills: np.ndarray, mus: np.ndarray, alphas: np.ndarray) -> dict[str, float]:
    """Check empirical coverage of prediction intervals."""
    n_params = 1.0 / alphas
    p_params = 1.0 / (1.0 + alphas * mus)

    q10 = stats.nbinom.ppf(0.10, n_params, p_params)
    q25 = stats.nbinom.ppf(0.25, n_params, p_params)
    q75 = stats.nbinom.ppf(0.75, n_params, p_params)
    q90 = stats.nbinom.ppf(0.90, n_params, p_params)

    cov_50 = float(np.mean((actual_kills >= q25) & (actual_kills <= q75)))
    cov_80 = float(np.mean((actual_kills >= q10) & (actual_kills <= q90)))
    return {"coverage_p25_p75": round(cov_50, 4), "coverage_p10_p90": round(cov_80, 4)}


def run_kill_distribution_walk_forward(
    output_path: Path | str | None = "reports/kill_distribution_walk_forward_results.json",
    min_games_history: int = 5,
) -> dict[str, Any]:
    """Run full expanding-window walk-forward validation for total kills distribution."""
    sess = get_session()
    try:
        games_df = pd.read_sql(text("""
            SELECT 
                game_id, 
                match_id, 
                date, 
                tournament_name,
                team1_name,
                team2_name,
                game_duration,
                team1_win,
                (team1_stats_json::jsonb->>'kills')::int AS kills_t1,
                (team2_stats_json::jsonb->>'kills')::int AS kills_t2
            FROM golgg_games
            WHERE game_duration IS NOT NULL 
              AND team1_stats_json IS NOT NULL 
              AND team2_stats_json IS NOT NULL
              AND date IS NOT NULL
              AND date >= '2022-01-01'
            ORDER BY date ASC, game_id ASC
        """), sess.bind)
    finally:
        sess.close()

    games_df["total_kills"] = games_df["kills_t1"] + games_df["kills_t2"]
    games_df["duration_minutes"] = games_df["game_duration"] / 60.0

    tracker = ChronologicalPaceTracker(window_size=20, prior_weight=5.0)

    records = []
    for row in games_df.itertuples(index=False):
        features = tracker.get_matchup_spread_features(row.team1_name, row.team2_name, row.tournament_name)
        pace_a = tracker.get_team_pace(row.team1_name, row.tournament_name)
        pace_b = tracker.get_team_pace(row.team2_name, row.tournament_name)

        records.append({
            "game_id": row.game_id,
            "date": row.date,
            "tournament_name": row.tournament_name,
            "team1_name": row.team1_name,
            "team2_name": row.team2_name,
            "sample_1": pace_a.sample_games,
            "sample_2": pace_b.sample_games,
            "elo_diff": features.elo_diff,
            "abs_elo_diff": features.abs_elo_diff,
            "prob_fav": max(features.prob_win_a, features.prob_win_b),
            "exp_total_kills": features.expected_total_kills,
            "total_kills": row.total_kills,
            "duration_minutes": row.duration_minutes,
        })

        tracker.update_game(
            team_1=row.team1_name,
            team_2=row.team2_name,
            kills_1=row.kills_t1,
            kills_2=row.kills_t2,
            duration_minutes=row.duration_minutes,
            winner_team_1=bool(row.team1_win == 1),
        )

    all_data = pd.DataFrame(records)

    folds = generate_quarterly_folds()
    fold_summaries = []
    all_test_preds = []

    for fold_name, test_start, test_end in folds:
        train_data = all_data[all_data["date"] < test_start].copy()
        test_data = all_data[(all_data["date"] >= test_start) & (all_data["date"] < test_end)].copy()

        if len(test_data) == 0 or len(train_data) < 1000:
            continue

        valid_test = test_data[(test_data["sample_1"] >= min_games_history) & (test_data["sample_2"] >= min_games_history)].copy()
        if len(valid_test) == 0:
            continue

        # 1. Fit Naive Distribution
        train_mean = float(train_data["total_kills"].mean())
        train_var = float(train_data["total_kills"].var())
        alpha_naive = max((train_var - train_mean) / (train_mean ** 2), 0.01)

        # 2. Fit Pace-Only Model
        m_pace = smf.glm(
            "total_kills ~ exp_total_kills",
            data=train_data,
            family=sm.families.NegativeBinomial(alpha=0.08),
        ).fit()

        # 3. Fit Pace + Spread + Interaction Model
        m_spread = smf.glm(
            "total_kills ~ exp_total_kills * abs_elo_diff",
            data=train_data,
            family=sm.families.NegativeBinomial(alpha=0.08),
        ).fit()

        # Predict out-of-fold on test set
        y_test = valid_test["total_kills"].values
        mu_naive = np.full(len(valid_test), train_mean)
        alpha_naive_vec = np.full(len(valid_test), alpha_naive)

        mu_pace = m_pace.predict(valid_test).values
        alpha_pace_vec = np.full(len(valid_test), 0.08)

        mu_spread = m_spread.predict(valid_test).values
        # Dynamic dispersion (stomps have wider dispersion)
        alpha_spread_vec = 0.056 + 0.00008 * valid_test["abs_elo_diff"].values

        # Compute CRPS
        crps_naive = compute_discrete_crps(y_test, mu_naive, alpha_naive_vec)
        crps_pace = compute_discrete_crps(y_test, mu_pace, alpha_pace_vec)
        crps_spread = compute_discrete_crps(y_test, mu_spread, alpha_spread_vec)

        # Compute NLL
        nll_naive = compute_nll(y_test, mu_naive, alpha_naive_vec)
        nll_pace = compute_nll(y_test, mu_pace, alpha_pace_vec)
        nll_spread = compute_nll(y_test, mu_spread, alpha_spread_vec)

        # Compute MAE
        mae_naive = float(np.mean(np.abs(y_test - mu_naive)))
        mae_pace = float(np.mean(np.abs(y_test - mu_pace)))
        mae_spread = float(np.mean(np.abs(y_test - mu_spread)))

        # Quantile Calibration
        cov_naive = compute_quantile_coverage(y_test, mu_naive, alpha_naive_vec)
        cov_pace = compute_quantile_coverage(y_test, mu_pace, alpha_pace_vec)
        cov_spread = compute_quantile_coverage(y_test, mu_spread, alpha_spread_vec)

        # Over/Under 28.5 line evaluation
        p_over_naive = 1.0 - stats.nbinom.cdf(28, 1.0 / alpha_naive_vec, 1.0 / (1.0 + alpha_naive_vec * mu_naive))
        p_over_pace = 1.0 - stats.nbinom.cdf(28, 1.0 / alpha_pace_vec, 1.0 / (1.0 + alpha_pace_vec * mu_pace))
        p_over_spread = 1.0 - stats.nbinom.cdf(28, 1.0 / alpha_spread_vec, 1.0 / (1.0 + alpha_spread_vec * mu_spread))
        y_over_28_5 = (y_test > 28.5).astype(int)

        logloss_naive_28 = float(log_loss(y_over_28_5, p_over_naive))
        logloss_pace_28 = float(log_loss(y_over_28_5, p_over_pace))
        logloss_spread_28 = float(log_loss(y_over_28_5, p_over_spread))

        fold_summaries.append({
            "fold": fold_name,
            "period": f"{test_start} to {test_end}",
            "train_matches": len(train_data),
            "test_matches": len(valid_test),
            "crps": {
                "naive": round(crps_naive, 4),
                "pace_only": round(crps_pace, 4),
                "pace_and_spread": round(crps_spread, 4),
                "delta_vs_pace": round(crps_spread - crps_pace, 4),
            },
            "nll": {
                "naive": round(nll_naive, 4),
                "pace_only": round(nll_pace, 4),
                "pace_and_spread": round(nll_spread, 4),
            },
            "mae": {
                "naive": round(mae_naive, 3),
                "pace_only": round(mae_pace, 3),
                "pace_and_spread": round(mae_spread, 3),
            },
            "calibration_p10_p90": {
                "target": 0.80,
                "naive": cov_naive["coverage_p10_p90"],
                "pace_only": cov_pace["coverage_p10_p90"],
                "pace_and_spread": cov_spread["coverage_p10_p90"],
            },
            "over_28_5_logloss": {
                "naive": round(logloss_naive_28, 4),
                "pace_only": round(logloss_pace_28, 4),
                "pace_and_spread": round(logloss_spread_28, 4),
                "delta_vs_pace": round(logloss_spread_28 - logloss_pace_28, 4),
            },
        })

        valid_test["mu_spread"] = mu_spread
        valid_test["alpha_spread"] = alpha_spread_vec
        valid_test["crps_spread"] = [compute_discrete_crps(np.array([y]), np.array([m]), np.array([a])) for y, m, a in zip(y_test, mu_spread, alpha_spread_vec)]
        all_test_preds.append(valid_test)

    combined = pd.concat(all_test_preds, ignore_index=True)
    all_y = combined["total_kills"].values
    all_mu = combined["mu_spread"].values
    all_alpha = combined["alpha_spread"].values

    total_crps = compute_discrete_crps(all_y, all_mu, all_alpha)
    total_nll = compute_nll(all_y, all_mu, all_alpha)
    total_mae = float(np.mean(np.abs(all_y - all_mu)))
    total_cov = compute_quantile_coverage(all_y, all_mu, all_alpha)

    summary = {
        "title": "Rozkład Liczby Zabójstw: Walidacja Krocząca Walk-Forward (IDEA-018)",
        "total_test_matches": len(combined),
        "total_folds": len(fold_summaries),
        "overall_distribution_metrics": {
            "model": "Negative Binomial (Pace * Spread + Dynamic Heteroscedastic Dispersion)",
            "crps": round(total_crps, 4),
            "nll": round(total_nll, 4),
            "mae": round(total_mae, 3),
            "calibration_p10_p90": total_cov["coverage_p10_p90"],
            "calibration_p25_p75": total_cov["coverage_p25_p75"],
        },
        "folds": fold_summaries,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = run_kill_distribution_walk_forward()
    print(json.dumps(res, indent=2))
