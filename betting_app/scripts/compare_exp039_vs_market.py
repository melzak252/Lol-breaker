"""EXP-052b: Compare EXP-039 model predictions (from DB) vs bookmaker odds.

Extracts EXP-039 predictions from the ``canonical_predictions`` table, joins with
``canonical_matches`` for golgg_match_id and team names, joins with ``data/odds.csv``,
computes no-vig bookmaker probabilities, and reports model vs market metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db, get_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--odds-file", default=None)
    parser.add_argument("--model-name", default="Sym-Cal LR-ElasticNet-W20-Binomial")
    parser.add_argument("--model-version", default="exp-039")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"log_loss": None, "brier": None, "auc": None, "accuracy": None, "n": 0}
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) >= 2 else None
    except ValueError:
        auc = None
    return {
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true, (y_prob > 0.5).astype(int))),
        "n": int(len(y_true)),
    }


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > boundaries[i]) & (y_prob <= boundaries[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_prob) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def main() -> None:
    args = parse_args()
    init_db()

    # ── Extract EXP-039 predictions from DB ──
    session = get_session()
    query = text("""
        SELECT cp.canonical_match_id,
               cp.prob_a,
               cp.prob_b,
               cm.team_a_name,
               cm.team_b_name,
               gmm.golgg_match_id,
               gm.team1_name,
               gm.team2_name,
               gm.date,
               gm.team1_win
        FROM canonical_predictions cp
        JOIN canonical_matches cm ON cm.id = cp.canonical_match_id
        JOIN golgg_match_mappings gmm ON gmm.canonical_match_id = cm.id
        JOIN golgg_matches gm ON gm.match_id = gmm.golgg_match_id
        WHERE cp.model_name = :model_name
          AND cp.model_version = :model_version
    """)
    df = pd.read_sql(query, session.bind, params={"model_name": args.model_name, "model_version": args.model_version})
    session.close()
    print(f"EXP-039 predictions from DB: {len(df)} rows")

    df["golgg_match_id"] = df["golgg_match_id"].astype(str)
    df["y_true"] = df["team1_win"].astype(int)

    # Align prob_a (canonical team_a) to golgg team1
    # If team_a_name matches golgg team1, prob_a is for team1
    # If team_a_name matches golgg team2, 1-prob_a is for team1
    from betting_app.core.matching import normalize_team_name
    df["norm_a"] = df["team_a_name"].apply(normalize_team_name)
    df["norm_b"] = df["team_b_name"].apply(normalize_team_name)
    df["norm_g1"] = df["team1_name"].apply(normalize_team_name)
    df["norm_g2"] = df["team2_name"].apply(normalize_team_name)
    df["a_is_team1"] = df["norm_a"] == df["norm_g1"]
    df["a_is_team2"] = df["norm_a"] == df["norm_g2"]
    df = df[df["a_is_team1"] | df["a_is_team2"]].copy()
    df["p_model"] = np.where(df["a_is_team1"], df["prob_a"].astype(float), 1.0 - df["prob_a"].astype(float))
    print(f"After team alignment: {len(df)} rows")

    # ── Load odds ──
    odds_path = args.odds_file or str(Path(args.data_dir) / "odds.csv")
    odds = pd.read_csv(odds_path)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    print(f"Odds data: {len(odds)} rows")

    # ── Join ──
    merged = df.merge(odds, on="golgg_match_id", how="inner", suffixes=("_model", "_odds"))
    print(f"Joined (model ∩ odds): {len(merged)} matches")

    # ── Team alignment (model already aligned to golgg team1, now align odds to golgg team1) ──
    merged["teams_match"] = (merged["team1_name"] == merged["golgg_team1"]) & (merged["team2_name"] == merged["golgg_team2"])
    merged["teams_swapped"] = (merged["team1_name"] == merged["golgg_team2"]) & (merged["team2_name"] == merged["golgg_team1"])
    merged["teams_neither"] = ~merged["teams_match"] & ~merged["teams_swapped"]
    print(f"Team alignment: match={merged['teams_match'].sum()}, swapped={merged['teams_swapped'].sum()}, neither={merged['teams_neither'].sum()}")
    merged = merged[~merged["teams_neither"]].copy()

    # p_model is already aligned to golgg team1; no need to flip
    merged["p_model_aligned"] = merged["p_model"]

    # ── Filter valid odds ──
    merged = merged[merged["avg_odds_home"].notna() & merged["avg_odds_away"].notna()].copy()
    print(f"With valid avg odds: {len(merged)} matches")

    # ── Bookmaker probabilities (aligned to model team1) ──
    merged["bm_p_team1_raw"] = np.where(
        merged["teams_match"],
        1.0 / merged["avg_odds_home"],
        np.where(merged["teams_swapped"], 1.0 / merged["avg_odds_away"], np.nan),
    )
    merged["bm_p_team2_raw"] = np.where(
        merged["teams_match"],
        1.0 / merged["avg_odds_away"],
        np.where(merged["teams_swapped"], 1.0 / merged["avg_odds_home"], np.nan),
    )
    merged = merged[merged["bm_p_team1_raw"].notna()].copy()
    print(f"After team alignment filter: {len(merged)} matches")

    merged["bm_margin"] = merged["bm_p_team1_raw"] + merged["bm_p_team2_raw"] - 1
    print(f"Average bookmaker margin: {merged['bm_margin'].mean():.4f}")

    merged["bm_p_team1_novig"] = merged["bm_p_team1_raw"] / (merged["bm_p_team1_raw"] + merged["bm_p_team2_raw"])

    # ── Metrics ──
    y_true = merged["y_true"].astype(int).to_numpy()
    p_model = merged["p_model_aligned"].to_numpy()
    p_book_novig = merged["bm_p_team1_novig"].to_numpy()
    p_book_raw = merged["bm_p_team1_raw"].to_numpy()

    print("\n" + "=" * 70)
    print(f"EXP-039 vs Bookmaker on {len(merged)} common matches")
    print("=" * 70)

    model_metrics = _compute_metrics(y_true, p_model)
    book_novig = _compute_metrics(y_true, p_book_novig)
    book_raw = _compute_metrics(y_true, p_book_raw)

    model_metrics["ece"] = _ece(y_true, p_model)
    book_novig["ece"] = _ece(y_true, p_book_novig)
    book_raw["ece"] = _ece(y_true, p_book_raw)

    print(f"\n{'Model':<30} {'LogLoss':>8} {'Brier':>8} {'AUC':>8} {'Accuracy':>8} {'ECE':>8} {'N':>6}")
    print("-" * 80)
    for name, m in [
        ("EXP-039 (DB predictions)", model_metrics),
        ("Bookmaker (no-vig)", book_novig),
        ("Bookmaker (raw implied)", book_raw),
    ]:
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A"
        print(f"{name:<30} {m['log_loss']:>8.4f} {m['brier']:>8.4f} {auc_str:>8} {m['accuracy']:>8.4f} {m['ece']:>8.4f} {m['n']:>6}")

    # ── Betting simulation ──
    print("\n" + "=" * 70)
    print("BETTING SIMULATION (using EXP-039 predictions)")
    print("=" * 70)

    odds_team1 = np.where(merged["teams_match"], merged["avg_odds_home"], merged["avg_odds_away"])
    odds_team2 = np.where(merged["teams_match"], merged["avg_odds_away"], merged["avg_odds_home"])

    print(f"\n{'Edge>':>8} {'Bets':>6} {'Wins':>6} {'Profit':>8} {'ROI':>8} {'WinRate':>8}")
    print("-" * 50)
    for min_edge in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        profit = 0.0
        bets = 0
        wins = 0
        for i in range(len(y_true)):
            p = p_model[i]
            o1 = odds_team1[i]
            o2 = odds_team2[i]
            if np.isnan(o1) or np.isnan(o2) or o1 <= 1 or o2 <= 1:
                continue
            imp1 = 1.0 / o1
            imp2 = 1.0 / o2
            edge1 = p - imp1
            edge2 = (1 - p) - imp2
            if edge1 > min_edge:
                bets += 1
                if y_true[i] == 1:
                    profit += o1 - 1
                    wins += 1
                else:
                    profit -= 1
            elif edge2 > min_edge:
                bets += 1
                if y_true[i] == 0:
                    profit += o2 - 1
                    wins += 1
                else:
                    profit -= 1
        roi = profit / bets * 100 if bets > 0 else 0
        win_rate = wins / bets * 100 if bets > 0 else 0
        print(f"{min_edge:>7.2f} {bets:>6d} {wins:>6d} {profit:>8.1f} {roi:>7.1f}% {win_rate:>7.1f}%")

    # ── Save JSON ──
    payload: dict[str, Any] = {
        "experiment_id": "EXP-052b",
        "model_name": args.model_name,
        "model_version": args.model_version,
        "db_prediction_count": int(len(df)),
        "odds_count": int(len(odds)),
        "common_count": int(len(merged)),
        "avg_bookmaker_margin": float(merged["bm_margin"].mean()),
        "metrics": {
            "exp039_model": model_metrics,
            "bookmaker_novig": book_novig,
            "bookmaker_raw_implied": book_raw,
        },
    }
    output = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()