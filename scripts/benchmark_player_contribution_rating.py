"""Benchmark bounded player contribution rating updates vs standard Glicko-2.

Protocol:
1. Replay GOL.GG match history chronologically (daily periods).
2. Compute pre-match win probability for each game using team average player rating.
3. Compare:
   - Baseline: standard Glicko-2 team-outcome update (all 5 players get identical actual-expected score).
   - Candidate (Bounded Zero-Sum Player Contribution):
     - Compute player in-game performance score P_i using role-normalized metrics (KDA, KP%, DPM share, Gold share).
     - Calculate team-relative contribution delta: d_i = clip(k * (P_i - P_team_mean), -max_delta, max_delta)
     - Enforce exact zero-sum on team: sum(d_i) = 0.
     - Effective score for player i: s_i = clip(S_team + d_i, 0.0, 1.0)
     - Update Glicko-2 using s_i instead of S_team.
4. Evaluate on common out-of-time test split (2024+):
   - Sample size N
   - LogLoss
   - Brier score
   - ROC AUC
   - ECE
   - Paired LogLoss delta & t-test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# Add worktree src to path
SRC_ROOT = Path("/tmp/inzynierka-operational-model-cutover")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.ratings.glicko2_core import (
    Glicko2Observation,
    Glicko2State,
    expected_score,
    update,
)

ROLES = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


def clip_prob(p: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, float(p)))


def compute_performance_score(stats: dict[str, Any], role: str) -> float:
    """Compute a robust, normalized player performance score from available boxscore stats."""
    if not stats:
        return 0.5

    def get_num(key: str, default: float = 0.0) -> float:
        val = stats.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    kills = get_num("kills")
    deaths = get_num("deaths")
    assists = get_num("assists")
    kda = (kills + assists) / max(1.0, deaths)

    kp = get_num("kp%", 50.0) / 100.0 if "kp%" in stats else get_num("kp", 0.5)
    dmg_share = get_num("dmg%", 20.0) / 100.0 if "dmg%" in stats else get_num("dmg_share", 0.20)
    gold_share = get_num("gold%", 20.0) / 100.0 if "gold%" in stats else get_num("gold_share", 0.20)
    dpm = get_num("dpm", 400.0)
    vspm = get_num("vspm", 1.5)

    # Role weights
    if role == "SUPPORT":
        score = 0.35 * (kp / 0.70) + 0.35 * (vspm / 2.5) + 0.20 * (kda / 3.5) + 0.10 * (dpm / 250.0)
    elif role == "JUNGLE":
        score = 0.35 * (kp / 0.65) + 0.25 * (kda / 3.5) + 0.20 * (vspm / 1.8) + 0.20 * (dmg_share / 0.18)
    elif role in ("MID", "ADC"):
        score = 0.40 * (dmg_share / 0.26) + 0.25 * (dpm / 550.0) + 0.20 * (kda / 4.0) + 0.15 * (gold_share / 0.23)
    else:  # TOP
        score = 0.35 * (dmg_share / 0.22) + 0.30 * (dpm / 480.0) + 0.20 * (kda / 3.0) + 0.15 * (kp / 0.55)

    return float(np.clip(score, 0.1, 2.0))


@dataclass
class PlayerStatRecord:
    player_id: str
    role: str
    perf_score: float


def run_simulation(
    matches_file: Path,
    *,
    test_start_date: date,
    max_delta: float = 0.15,
    delta_scale: float = 0.20,
) -> dict[str, Any]:
    print(f"Loading matches from {matches_file}...")
    with open(matches_file, "r", encoding="utf-8") as f:
        matches = json.load(f)

    # Sort strictly chronologically by date
    matches = [m for m in matches if m.get("date") and m.get("games")]
    matches.sort(key=lambda m: (m["date"], str(m.get("match_id", ""))))

    # Two engine states: standard and contribution-adjusted
    std_ratings: dict[str, Glicko2State] = defaultdict(Glicko2State)
    adj_ratings: dict[str, Glicko2State] = defaultdict(Glicko2State)

    y_true: list[int] = []
    std_probs: list[float] = []
    adj_probs: list[float] = []
    test_dates: list[str] = []

    n_games_evaluated = 0

    for m in matches:
        match_date = date.fromisoformat(m["date"])
        is_test = match_date >= test_start_date

        for g in m.get("games", []):
            t1_players_dict = g.get("t1_players") or {}
            t2_players_dict = g.get("t2_players") or {}

            if len(t1_players_dict) != 5 or len(t2_players_dict) != 5:
                continue

            # Check for valid player IDs
            t1_ids = [p["player_id"] for p in t1_players_dict.values() if p and p.get("player_id")]
            t2_ids = [p["player_id"] for p in t2_players_dict.values() if p and p.get("player_id")]
            if len(t1_ids) != 5 or len(t2_ids) != 5:
                continue

            t1_win = g.get("t1_win")
            if t1_win is None:
                continue
            s_team1 = 1.0 if t1_win else 0.0
            s_team2 = 1.0 - s_team1

            # 1. Pre-game prediction using team-average ratings
            # Standard
            std_r1 = np.mean([std_ratings[pid].rating for pid in t1_ids])
            std_rd1 = math.sqrt(np.mean([std_ratings[pid].rd ** 2 for pid in t1_ids]))
            std_r2 = np.mean([std_ratings[pid].rating for pid in t2_ids])
            std_rd2 = math.sqrt(np.mean([std_ratings[pid].rd ** 2 for pid in t2_ids]))
            p_std = expected_score(std_r1, std_rd1, std_r2, std_rd2)

            # Adjusted
            adj_r1 = np.mean([adj_ratings[pid].rating for pid in t1_ids])
            adj_rd1 = math.sqrt(np.mean([adj_ratings[pid].rd ** 2 for pid in t1_ids]))
            adj_r2 = np.mean([adj_ratings[pid].rating for pid in t2_ids])
            adj_rd2 = math.sqrt(np.mean([adj_ratings[pid].rd ** 2 for pid in t2_ids]))
            p_adj = expected_score(adj_r1, adj_rd1, adj_r2, adj_rd2)

            if is_test:
                y_true.append(int(t1_win))
                std_probs.append(clip_prob(p_std))
                adj_probs.append(clip_prob(p_adj))
                test_dates.append(m["date"])
                n_games_evaluated += 1

            # 2. Extract in-game performance scores
            p_scores1 = []
            for role, pdata in t1_players_dict.items():
                stats = pdata.get("stats") or {}
                p_scores1.append(compute_performance_score(stats, role))

            p_scores2 = []
            for role, pdata in t2_players_dict.items():
                stats = pdata.get("stats") or {}
                p_scores2.append(compute_performance_score(stats, role))

            # Team 1 zero-sum allocation
            mean1 = float(np.mean(p_scores1))
            deltas1 = [delta_scale * (s - mean1) for s in p_scores1]
            deltas1 = [max(-max_delta, min(max_delta, d)) for d in deltas1]
            # Zero-sum mean centering
            mean_delta1 = float(np.mean(deltas1))
            deltas1 = [d - mean_delta1 for d in deltas1]

            # Team 2 zero-sum allocation
            mean2 = float(np.mean(p_scores2))
            deltas2 = [delta_scale * (s - mean2) for s in p_scores2]
            deltas2 = [max(-max_delta, min(max_delta, d)) for d in deltas2]
            mean_delta2 = float(np.mean(deltas2))
            deltas2 = [d - mean_delta2 for d in deltas2]

            # 3. Post-game ratings update
            # --- Standard Update ---
            for pid in t1_ids:
                obs = [Glicko2Observation(opponent_rating=std_r2, opponent_rd=std_rd2, score=s_team1)]
                std_ratings[pid] = update(std_ratings[pid], obs)
            for pid in t2_ids:
                obs = [Glicko2Observation(opponent_rating=std_r1, opponent_rd=std_rd1, score=s_team2)]
                std_ratings[pid] = update(std_ratings[pid], obs)

            # --- Adjusted Contribution Update ---
            for pid, delta in zip(t1_ids, deltas1):
                # Adjusted score bounded to [0, 1]
                effective_score = float(np.clip(s_team1 + delta, 0.0, 1.0))
                obs = [Glicko2Observation(opponent_rating=adj_r2, opponent_rd=adj_rd2, score=effective_score)]
                adj_ratings[pid] = update(adj_ratings[pid], obs)

            for pid, delta in zip(t2_ids, deltas2):
                effective_score = float(np.clip(s_team2 + delta, 0.0, 1.0))
                obs = [Glicko2Observation(opponent_rating=adj_r1, opponent_rd=adj_rd1, score=effective_score)]
                adj_ratings[pid] = update(adj_ratings[pid], obs)

    print(f"Simulation completed. Evaluated {n_games_evaluated:,} games in test period ({test_start_date}+).")

    y = np.array(y_true, dtype=int)
    p_std_arr = np.array(std_probs)
    p_adj_arr = np.array(adj_probs)

    def calc_ece(y_t: np.ndarray, p_arr: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for l, r in zip(edges[:-1], edges[1:]):
            mask = (p_arr >= l) & (p_arr <= r if r == 1.0 else p_arr < r)
            if np.any(mask):
                ece += (mask.sum() / len(y_t)) * abs(y_t[mask].mean() - p_arr[mask].mean())
        return float(ece)

    ll_std = float(log_loss(y, p_std_arr))
    ll_adj = float(log_loss(y, p_adj_arr))
    brier_std = float(brier_score_loss(y, p_std_arr))
    brier_adj = float(brier_score_loss(y, p_adj_arr))
    auc_std = float(roc_auc_score(y, p_std_arr))
    auc_adj = float(roc_auc_score(y, p_adj_arr))
    ece_std = calc_ece(y, p_std_arr)
    ece_adj = calc_ece(y, p_adj_arr)
    acc_std = float(accuracy_score(y, p_std_arr >= 0.5))
    acc_adj = float(accuracy_score(y, p_adj_arr >= 0.5))

    # Paired delta test
    deltas = -(y * np.log(p_adj_arr) + (1 - y) * np.log(1 - p_adj_arr)) - (
        -(y * np.log(p_std_arr) + (1 - y) * np.log(1 - p_std_arr))
    )
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    t_stat = mean_delta / (std_delta / math.sqrt(len(deltas))) if std_delta > 0 else 0.0

    return {
        "n_games": len(y),
        "standard_glicko": {
            "logloss": ll_std,
            "brier": brier_std,
            "auc": auc_std,
            "ece": ece_std,
            "accuracy": acc_std,
        },
        "contribution_glicko": {
            "logloss": ll_adj,
            "brier": brier_adj,
            "auc": auc_adj,
            "ece": ece_adj,
            "accuracy": acc_adj,
            "max_delta": max_delta,
            "delta_scale": delta_scale,
        },
        "comparison": {
            "delta_logloss": mean_delta,
            "delta_brier": brier_adj - brier_std,
            "t_statistic": t_stat,
            "candidate_better": bool(mean_delta < 0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="/home/melzak/dev/inzynierka/data/golgg_matches.json")
    parser.add_argument("--test-start", default="2024-01-01")
    parser.add_argument("--max-delta", type=float, default=0.15)
    parser.add_argument("--delta-scale", type=float, default=0.20)
    args = parser.parse_args()

    res = run_simulation(
        Path(args.data_file),
        test_start_date=date.fromisoformat(args.test_start),
        max_delta=args.max_delta,
        delta_scale=args.delta_scale,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (Test Split 2024+)")
    print("=" * 60)
    print(f"Sample size: {res['n_games']:,} maps")
    print(f"{'Metric':<15} {'Standard Glicko':<18} {'Contribution Glicko':<18} {'Delta'}")
    print("-" * 60)
    print(
        f"{'LogLoss':<15} {res['standard_glicko']['logloss']:<18.6f} {res['contribution_glicko']['logloss']:<18.6f} {res['comparison']['delta_logloss']:+.6f}"
    )
    print(
        f"{'Brier':<15} {res['standard_glicko']['brier']:<18.6f} {res['contribution_glicko']['brier']:<18.6f} {res['comparison']['delta_brier']:+.6f}"
    )
    print(
        f"{'ROC AUC':<15} {res['standard_glicko']['auc']:<18.6f} {res['contribution_glicko']['auc']:<18.6f} {res['contribution_glicko']['auc'] - res['standard_glicko']['auc']:+.6f}"
    )
    print(
        f"{'ECE':<15} {res['standard_glicko']['ece']:<18.6f} {res['contribution_glicko']['ece']:<18.6f} {res['contribution_glicko']['ece'] - res['standard_glicko']['ece']:+.6f}"
    )
    print(
        f"{'Accuracy':<15} {res['standard_glicko']['accuracy']:<18.6f} {res['contribution_glicko']['accuracy']:<18.6f} {res['contribution_glicko']['accuracy'] - res['standard_glicko']['accuracy']:+.6f}"
    )
    print("-" * 60)
    print(f"Paired LogLoss t-stat: {res['comparison']['t_statistic']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
