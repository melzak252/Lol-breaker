"""Chronological out-of-time backtest and benchmark for KillDistributionModel (IDEA-018)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import text

from betting_app.core.db import get_session
from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.kill_distribution_model import KillDistributionModel


def run_kill_distribution_backtest(
    train_end_date: str = "2024-01-01",
    min_games_history: int = 5,
    tax_rate: float = 0.12,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute leakage-safe chronological evaluation of the Negative Binomial model."""
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
    model = KillDistributionModel()

    records = []
    for row in games_df.itertuples(index=False):
        t1 = row.team1_name
        t2 = row.team2_name
        d = row.date
        league = row.tournament_name
        actual_kills = row.total_kills
        duration = row.duration_minutes

        # Pre-match parameter calculation
        matchup_pace = tracker.get_matchup_expected_pace(t1, t2, league)
        sample_a = matchup_pace["team_a_pace"].sample_games
        sample_b = matchup_pace["team_b_pace"].sample_games
        exp_pace = matchup_pace["expected_total_kills"]

        params = model.estimate_params(exp_pace)

        records.append({
            "game_id": row.game_id,
            "date": d,
            "tournament_name": league,
            "league_key": matchup_pace["league_key"],
            "team_a": t1,
            "team_b": t2,
            "sample_a": sample_a,
            "sample_b": sample_b,
            "expected_pace": exp_pace,
            "pred_mu": params.mu,
            "pred_std": params.std,
            "actual_kills": actual_kills,
            "actual_duration": duration,
        })

        # Strictly post-game update
        tracker.update_post_game(t1, row.kills_t1, row.kills_t2, duration)
        tracker.update_post_game(t2, row.kills_t2, row.kills_t1, duration)

    eval_df = pd.DataFrame(records)

    # Split into train (2022-2023) and out-of-time test (2024-2026)
    train_mask = eval_df["date"] < train_end_date
    test_mask = eval_df["date"] >= train_end_date

    # Filter to matches where both teams have sufficient history
    valid_history = (eval_df["sample_a"] >= min_games_history) & (eval_df["sample_b"] >= min_games_history)
    test_df = eval_df[test_mask & valid_history].copy()

    # Evaluate across common prop lines
    test_mu = test_df["pred_mu"].values
    test_actual = test_df["actual_kills"].values
    n_param = 1.0 / model.alpha
    p_params = 1.0 / (1.0 + model.alpha * test_mu)

    lines_to_eval = [22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5]
    line_metrics = {}

    for line in lines_to_eval:
        floor_l = int(line)
        prob_under = stats.nbinom.cdf(floor_l, n_param, p_params)
        prob_over = 1.0 - prob_under

        y_actual_over = (test_actual > line).astype(int)
        naive_rate = float(np.mean(y_actual_over))
        naive_probs = np.full_like(y_actual_over, naive_rate, dtype=float)

        brier_model = float(brier_score_loss(y_actual_over, prob_over))
        brier_naive = float(brier_score_loss(y_actual_over, naive_probs))

        logloss_model = float(log_loss(y_actual_over, prob_over))
        logloss_naive = float(log_loss(y_actual_over, naive_probs))

        line_metrics[str(line)] = {
            "actual_over_rate": round(naive_rate, 4),
            "model_brier": round(brier_model, 4),
            "naive_brier": round(brier_naive, 4),
            "delta_brier": round(brier_model - brier_naive, 4),
            "model_logloss": round(logloss_model, 4),
            "naive_logloss": round(logloss_naive, 4),
            "delta_logloss": round(logloss_model - logloss_naive, 4),
        }

    # Value bet simulation at 1.85 / 1.85 lines with 12% tax
    odds = 1.85
    eff_odds = odds * (1.0 - tax_rate)
    break_even_p = 1.0 / eff_odds  # ~61.43%

    # Simulate against synthetic line = median of league
    league_medians = eval_df[train_mask].groupby("league_key")["actual_kills"].median().to_dict()
    test_df["bookmaker_line"] = test_df["league_key"].map(league_medians).fillna(28.0) + 0.5

    p_over_sim = []
    for _, r in test_df.iterrows():
        l_val = int(r["bookmaker_line"])
        p_u = stats.nbinom.cdf(l_val, n_param, 1.0 / (1.0 + model.alpha * r["pred_mu"]))
        p_over_sim.append(1.0 - p_u)

    test_df["prob_over"] = p_over_sim
    test_df["prob_under"] = 1.0 - np.array(p_over_sim)
    test_df["actual_over"] = (test_df["actual_kills"] > test_df["bookmaker_line"]).astype(int)

    sim_results = {}
    for edge in [0.0, 0.03, 0.05, 0.08, 0.10]:
        cutoff = break_even_p + edge
        over_bets = test_df[test_df["prob_over"] >= cutoff]
        under_bets = test_df[test_df["prob_under"] >= cutoff]

        n_bets = len(over_bets) + len(under_bets)
        if n_bets > 0:
            hits = int(over_bets["actual_over"].sum() + (1 - under_bets["actual_over"]).sum())
            profit = round((hits * eff_odds) - n_bets, 2)
            hit_rate = round(hits / n_bets, 4)
            roi = round((profit / n_bets) * 100.0, 2)
        else:
            hits, profit, hit_rate, roi = 0, 0.0, 0.0, 0.0

        sim_results[f"edge_{int(edge*100)}pct"] = {
            "threshold_prob": round(cutoff, 4),
            "bets": n_bets,
            "hits": hits,
            "hit_rate": hit_rate,
            "break_even_rate": round(break_even_p, 4),
            "profit_units": profit,
            "roi_pct": roi,
        }

    summary = {
        "model": "KillDistributionModel (NegativeBinomial GLM)",
        "train_period": f"2022-01-01 to {train_end_date}",
        "test_period": f"{train_end_date} to 2026-09-05",
        "train_games_count": int(train_mask.sum()),
        "test_games_evaluated": len(test_df),
        "min_games_history": min_games_history,
        "parameters": {
            "intercept": model.intercept,
            "pace_coef": model.pace_coef,
            "alpha": model.alpha,
        },
        "line_evaluation": line_metrics,
        "synthetic_market_simulation_tax12": sim_results,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    rep = run_kill_distribution_backtest(output_path="reports/kill_distribution_model_metrics.json")
    print(json.dumps(rep, indent=2))
