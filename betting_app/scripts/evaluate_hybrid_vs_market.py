"""EXP-052: Evaluate EXP-051 hybrid model OOF predictions vs bookmaker odds.

Builds the EXP-051 hybrid dataset (embedding + W20 features), runs walk-forward
training with ``collect_oof=True`` to collect per-match calibrated OOF probabilities,
joins with ``data/odds.csv`` on ``golgg_match_id``, computes no-vig bookmaker
probabilities, and reports model vs market metrics (LogLoss, Brier, AUC, Accuracy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db
from betting_app.ml.training.hybrid_embedding_w20_dataset import (
    HybridEmbeddingW20Config,
    build_hybrid_embedding_w20_dataset,
    load_legacy_w20_features,
)
from betting_app.ml.training.player_embedding_match_dataset import (
    PlayerEmbeddingMatchDatasetConfig,
    build_match_dataset_from_embeddings,
    encode_player_game_embeddings,
)
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.strength_dataset import load_golgg_match_results
from betting_app.ml.training.strength_model import StrengthModelConfig, train_strength_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-artifact", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--odds-file", default=None, help="Path to odds CSV (default: <data-dir>/odds.csv)")
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--limit-player-rows", type=int, default=None)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=250)
    parser.add_argument("--min-prior-player-games", type=int, default=50)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--market-common-only", action="store_true")
    parser.add_argument("--no-team-features", action="store_true")
    parser.add_argument("--no-days-features", action="store_true")
    parser.add_argument("--allow-target-disagreement", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--initial-train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--step-size", type=int, default=2000)
    parser.add_argument("--min-fold-train-size", type=int, default=3000)
    parser.add_argument("--logistic-c", type=float, default=0.03297234640536737)
    parser.add_argument("--l1-ratio", type=float, default=0.9439657999531195)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--fusion-file", default=None, help="Path to fusion_predictions_all.json (default: <data-dir>/fusion_predictions_all.json)")
    parser.add_argument("--save-oof", default=None, help="Save OOF predictions to this CSV path")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    """Compute LogLoss, Brier, AUC, Accuracy."""
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
    """Expected Calibration Error."""
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

    # ── Build hybrid dataset (same as train_hybrid_embedding_w20_model.py) ──
    player_dataset = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_player_rows)
    )
    embeddings = encode_player_game_embeddings(
        player_dataset,
        encoder_artifact=Path(args.encoder_artifact),
        device=args.device,
        batch_size=args.embedding_batch_size,
    )
    matches = load_golgg_match_results(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_matches)
    embedding_dataset = build_match_dataset_from_embeddings(
        matches,
        embeddings,
        PlayerEmbeddingMatchDatasetConfig(
            history_size=args.history_size,
            min_prior_player_games=args.min_prior_player_games,
        ),
    )

    hybrid_config = HybridEmbeddingW20Config(
        data_dir=args.data_dir,
        rolling_window=args.rolling_window,
        min_date=args.min_date,
        max_date=args.max_date,
        market_common_only=args.market_common_only,
        include_team_features=not args.no_team_features,
        include_days_features=not args.no_days_features,
        require_target_agreement=not args.allow_target_disagreement,
    )
    legacy_features, legacy_feature_names = load_legacy_w20_features(hybrid_config)
    hybrid_dataset = build_hybrid_embedding_w20_dataset(
        embedding_dataset,
        legacy_features,
        legacy_feature_names,
        hybrid_config,
    )

    # ── Train with collect_oof=True ──
    model_config = StrengthModelConfig(
        model_name="HybridEmbedding-W20-LR",
        model_version="exp-051",
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_fold_train_size=args.min_fold_train_size,
        logistic_c=args.logistic_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        use_order_augmentation=False,
        calibrate=True,
        collect_oof=True,
    )
    training = train_strength_model(hybrid_dataset, model_config)

    if training.oof_frame is None or training.oof_frame.empty:
        raise RuntimeError("No OOF predictions collected — check collect_oof config")

    oof = training.oof_frame.copy()
    oof["match_id"] = oof["match_id"].astype(str)
    print(f"OOF predictions: {len(oof)} matches")

    if args.save_oof:
        oof.to_csv(args.save_oof, index=False)
        print(f"OOF predictions saved to {args.save_oof}")

    # ── Load odds ──
    odds_path = args.odds_file or str(Path(args.data_dir) / "odds.csv")
    odds = pd.read_csv(odds_path)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    print(f"Odds data: {len(odds)} rows")

    # ── Join OOF with odds ──
    merged = oof.merge(odds, left_on="match_id", right_on="golgg_match_id", how="inner", suffixes=("_oof", "_odds"))
    print(f"Joined (OOF ∩ odds): {len(merged)} matches")

    # ── Team alignment check ──
    if "team1_name" in merged.columns and "golgg_team1" in merged.columns:
        merged["teams_match"] = (merged["team1_name"] == merged["golgg_team1"]) & (merged["team2_name"] == merged["golgg_team2"])
        merged["teams_swapped"] = (merged["team1_name"] == merged["golgg_team2"]) & (merged["team2_name"] == merged["golgg_team1"])
        merged["teams_neither"] = ~merged["teams_match"] & ~merged["teams_swapped"]
        print(f"Team alignment: match={merged['teams_match'].sum()}, swapped={merged['teams_swapped'].sum()}, neither={merged['teams_neither'].sum()}")
        # Drop ambiguous rows
        merged = merged[~merged["teams_neither"]].copy()
    else:
        merged["teams_match"] = True
        merged["teams_swapped"] = False

    # ── Filter to rows with valid avg odds ──
    merged = merged[merged["avg_odds_home"].notna() & merged["avg_odds_away"].notna()].copy()
    print(f"With valid avg odds: {len(merged)} matches")

    # ── Compute bookmaker implied probabilities (aligned to model team1) ──
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

    # Bookmaker margin
    merged["bm_margin"] = merged["bm_p_team1_raw"] + merged["bm_p_team2_raw"] - 1
    print(f"Average bookmaker margin: {merged['bm_margin'].mean():.4f}")

    # No-vig probabilities
    merged["bm_p_team1_novig"] = merged["bm_p_team1_raw"] / (merged["bm_p_team1_raw"] + merged["bm_p_team2_raw"])

    # ── Prepare arrays ──
    y_true = merged["target"].astype(int).to_numpy()
    p_model_cal = merged["oof_prob_calibrated"].to_numpy()
    p_model_raw = merged["oof_prob_raw"].to_numpy()
    p_book_novig = merged["bm_p_team1_novig"].to_numpy()
    p_book_raw = merged["bm_p_team1_raw"].to_numpy()

    # ── Metrics ──
    print("\n" + "=" * 70)
    print(f"EXP-052: Model vs Market on {len(merged)} common matches")
    print("=" * 70)

    model_cal = _compute_metrics(y_true, p_model_cal)
    model_raw = _compute_metrics(y_true, p_model_raw)
    book_novig = _compute_metrics(y_true, p_book_novig)
    book_raw = _compute_metrics(y_true, p_book_raw)

    model_cal["ece"] = _ece(y_true, p_model_cal)
    model_raw["ece"] = _ece(y_true, p_model_raw)
    book_novig["ece"] = _ece(y_true, p_book_novig)
    book_raw["ece"] = _ece(y_true, p_book_raw)

    print(f"\n{'Model':<30} {'LogLoss':>8} {'Brier':>8} {'AUC':>8} {'Accuracy':>8} {'ECE':>8} {'N':>6}")
    print("-" * 80)
    all_metrics: list[tuple[str, dict]] = [
        ("EXP-051 calibrated OOF", model_cal),
        ("EXP-051 raw OOF", model_raw),
        ("Bookmaker (no-vig)", book_novig),
        ("Bookmaker (raw implied)", book_raw),
    ]

    # ── Load fusion predictions and player_elo for same-subset comparison ──
    fusion_path = args.fusion_file or str(Path(args.data_dir) / "fusion_predictions_all.json")
    fusion_data: dict[str, Any] = {}
    if Path(fusion_path).exists():
        with open(fusion_path, "r", encoding="utf-8") as fh:
            fusion_data = json.load(fh)
        print(f"\nFusion predictions loaded: {len(fusion_data)} entries")
        merged["fusion_v2"] = merged["match_id"].map(lambda x: fusion_data.get(x, {}).get("fusion_v2", np.nan))
        merged["fusion_v2_sym"] = merged["match_id"].map(lambda x: fusion_data.get(x, {}).get("fusion_v2_sym", np.nan))
        merged["fusion_v2_archsym"] = merged["match_id"].map(lambda x: fusion_data.get(x, {}).get("fusion_v2_archsym", np.nan))
        merged["player_elo"] = merged["match_id"].map(lambda x: fusion_data.get(x, {}).get("player_elo", np.nan))

        for name, col in [
            ("Fusion v2", "fusion_v2"),
            ("Fusion v2+SymAug", "fusion_v2_sym"),
            ("Fusion v2+ArchSym", "fusion_v2_archsym"),
            ("Player Elo", "player_elo"),
        ]:
            mask = merged[col].notna()
            if mask.sum() > 0:
                y_sub = y_true[mask.to_numpy()]
                p_sub = merged.loc[mask, col].to_numpy()
                m = _compute_metrics(y_sub, p_sub)
                m["ece"] = _ece(y_sub, p_sub)
                all_metrics.append((f"{name} ({mask.sum()})", m))
    else:
        print(f"\nFusion predictions file not found: {fusion_path}")

    for name, m in all_metrics:
        auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A"
        print(f"{name:<30} {m['log_loss']:>8.4f} {m['brier']:>8.4f} {auc_str:>8} {m['accuracy']:>8.4f} {m['ece']:>8.4f} {m['n']:>6}")

    # ── Betting simulation ──
    print("\n" + "=" * 70)
    print("BETTING SIMULATION (using calibrated OOF probabilities)")
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
            p_model = p_model_cal[i]
            o1 = odds_team1[i]
            o2 = odds_team2[i]
            if np.isnan(o1) or np.isnan(o2) or o1 <= 1 or o2 <= 1:
                continue
            imp1 = 1.0 / o1
            imp2 = 1.0 / o2
            edge1 = p_model - imp1
            edge2 = (1 - p_model) - imp2
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

    # ── Save JSON output ──
    payload: dict[str, Any] = {
        "experiment_id": "EXP-052",
        "experiment_ids": ["EXP-049", "EXP-050", "EXP-051", "EXP-052"],
        "encoder_artifact": str(Path(args.encoder_artifact)),
        "model_config": {
            "model_name": model_config.model_name,
            "model_version": model_config.model_version,
            "initial_train_size": model_config.initial_train_size,
            "test_size": model_config.test_size,
            "step_size": model_config.step_size,
            "logistic_c": model_config.logistic_c,
            "l1_ratio": model_config.l1_ratio,
            "calibrate": model_config.calibrate,
            "collect_oof": model_config.collect_oof,
        },
        "oof_count": int(len(oof)),
        "odds_count": int(len(odds)),
        "common_count": int(len(merged)),
        "avg_bookmaker_margin": float(merged["bm_margin"].mean()),
        "metrics": {
            "model_calibrated": model_cal,
            "model_raw": model_raw,
            "bookmaker_novig": book_novig,
            "bookmaker_raw_implied": book_raw,
            **{name: m for name, m in all_metrics[4:]},
        },
        "hybrid_dataset_metadata": hybrid_dataset.metadata,
        "training_metrics": training.metrics,
    }
    output = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()