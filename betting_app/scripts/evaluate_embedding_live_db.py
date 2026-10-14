"""Evaluate player-embedding match models against live DB market odds.

This is a DB-native follow-up to EXP-049/050/051.  It rebuilds leakage-safe
player embedding match features from the active PostgreSQL database, trains
chronological OOF models, and compares their OOF probabilities to bookmaker
no-vig probabilities on canonical matches mapped back to GOL.GG.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db, query_df
from betting_app.core.matching import normalize_team_name
from betting_app.ml.training.player_embedding_match_dataset import (
    PlayerEmbeddingMatchDatasetConfig,
    build_match_dataset_from_embeddings,
    encode_player_game_embeddings,
)
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.strength_dataset import StrengthDataset, StrengthDatasetConfig, build_strength_dataset_from_db, load_golgg_match_results
from betting_app.ml.training.strength_model import StrengthModelConfig, train_strength_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-artifact", required=True)
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--limit-player-rows", type=int, default=None)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=250)
    parser.add_argument("--min-prior-player-games", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--initial-train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--step-size", type=int, default=3000)
    parser.add_argument("--min-fold-train-size", type=int, default=3000)
    parser.add_argument("--logistic-c", type=float, default=0.05)
    parser.add_argument("--l1-ratio", type=float, default=0.50)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)) if len(y) else None,
    }


def _train_oof(dataset: StrengthDataset, *, name: str, args: argparse.Namespace) -> Any:
    cfg = StrengthModelConfig(
        model_name=name,
        model_version="live-db-oof",
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_fold_train_size=args.min_fold_train_size,
        logistic_c=args.logistic_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        use_order_augmentation=True,
        calibrate=True,
        collect_oof=True,
    )
    return train_strength_model(dataset, cfg)


def _build_db_hybrid(embedding_ds: Any, strength_ds: StrengthDataset) -> StrengthDataset:
    emb = embedding_ds.frame.copy()
    st = strength_ds.frame.copy()
    emb["match_id"] = emb["match_id"].astype(str)
    st["match_id"] = st["match_id"].astype(str)
    merged = emb.merge(st, on="match_id", how="inner", suffixes=("", "_strength"))
    if "target_strength" in merged.columns:
        merged = merged[merged["target"].astype(int) == merged["target_strength"].astype(int)].copy()
    # Prefer embedding-side metadata columns.
    drop_cols = [c for c in merged.columns if c.endswith("_strength") and c not in {"date_strength"}]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    strength_features = [f for f in strength_ds.feature_names if f in merged.columns]
    feature_names = list(embedding_ds.feature_names) + [f for f in strength_features if f not in embedding_ds.feature_names]
    metadata = {
        "experiment_id": "EXP-051-live-db-native",
        "purpose": "PlayerGameEncoder embedding features plus DB-native leakage-safe strength features.",
        "note": "This is not byte-identical to historical EXP-051 because legacy golgg_y_predicts.csv W20/rating features are not available for current July rows.",
        "embedding_metadata": embedding_ds.metadata,
        "strength_metadata": strength_ds.metadata,
        "rows": int(len(merged)),
        "feature_count": int(len(feature_names)),
    }
    if not merged.empty:
        metadata["date_min"] = pd.to_datetime(merged["date"], utc=True, errors="coerce").min().isoformat()
        metadata["date_max"] = pd.to_datetime(merged["date"], utc=True, errors="coerce").max().isoformat()
    return StrengthDataset(frame=merged, feature_names=feature_names, metadata=metadata)


def _load_market_alignment() -> pd.DataFrame:
    sql = """
    WITH latest_odds AS (
      SELECT DISTINCT ON (os.canonical_match_id, os.bookmaker_id)
        os.canonical_match_id,
        os.bookmaker_id,
        os.odds_a,
        os.odds_b,
        os.scraped_at
      FROM odds_snapshots os
      JOIN canonical_matches cm ON cm.id = os.canonical_match_id
      WHERE os.odds_a > 1 AND os.odds_b > 1
        AND os.scraped_at <= cm.start_time_normalized::timestamptz + interval '30 minutes'
      ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at DESC
    ), market AS (
      SELECT
        canonical_match_id,
        COUNT(*) AS books,
        AVG((1.0/odds_a) / ((1.0/odds_a) + (1.0/odds_b))) AS market_prob_a_novig,
        AVG((1.0/odds_a) + (1.0/odds_b) - 1.0) AS avg_margin
      FROM latest_odds
      GROUP BY canonical_match_id
    ), latest_pred AS (
      SELECT DISTINCT ON (canonical_match_id, model_name, model_version)
        canonical_match_id, model_name, model_version, prob_a, prob_b, predicted_at
      FROM canonical_predictions
      WHERE (model_name, model_version) IN (
        ('Hybrid-Thesis-Market', 'a0.05-t0.80'),
        ('Hybrid-Thesis-Market', 'a0.50-t0.80'),
        ('Sym-Cal LR-ElasticNet-W20-Binomial', 'exp-039')
      )
      ORDER BY canonical_match_id, model_name, model_version, predicted_at DESC
    )
    SELECT
      gmm.golgg_match_id::text AS match_id,
      gm.team1_name AS golgg_team1,
      gm.team2_name AS golgg_team2,
      cm.id AS canonical_match_id,
      cm.team_a_name,
      cm.team_b_name,
      cm.normalized_team_a,
      cm.normalized_team_b,
      cm.start_time_normalized::timestamptz AS start_at,
      cm.winner_side,
      m.books,
      m.market_prob_a_novig,
      m.avg_margin,
      MAX(lp.prob_a) FILTER (WHERE lp.model_name='Hybrid-Thesis-Market' AND lp.model_version='a0.05-t0.80') AS hybrid_new_prob_a,
      MAX(lp.prob_a) FILTER (WHERE lp.model_name='Hybrid-Thesis-Market' AND lp.model_version='a0.50-t0.80') AS hybrid_old_prob_a,
      MAX(lp.prob_a) FILTER (WHERE lp.model_name='Sym-Cal LR-ElasticNet-W20-Binomial') AS exp039_prob_a
    FROM golgg_match_mappings gmm
    JOIN golgg_matches gm ON gm.match_id::text = gmm.golgg_match_id::text
    JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
    LEFT JOIN market m ON m.canonical_match_id = cm.id
    LEFT JOIN latest_pred lp ON lp.canonical_match_id = cm.id
    WHERE cm.status='finished' AND cm.winner_side IN ('team_a','team_b','A','B')
    GROUP BY gmm.golgg_match_id, gm.team1_name, gm.team2_name, cm.id, cm.team_a_name, cm.team_b_name,
             cm.normalized_team_a, cm.normalized_team_b, cm.start_time_normalized, cm.winner_side,
             m.books, m.market_prob_a_novig, m.avg_margin
    """
    return query_df(sql)


def _align_prob_to_golgg_team1(row: pd.Series, prob_a_col: str) -> float:
    p = row.get(prob_a_col)
    if pd.isna(p):
        return np.nan
    g1 = normalize_team_name(str(row.get("golgg_team1") or ""))
    ca = normalize_team_name(str(row.get("team_a_name") or row.get("normalized_team_a") or ""))
    cb = normalize_team_name(str(row.get("team_b_name") or row.get("normalized_team_b") or ""))
    if g1 and ca and g1 == ca:
        return float(p)
    if g1 and cb and g1 == cb:
        return float(1.0 - float(p))
    return np.nan


def _market_compare(oof: pd.DataFrame, label: str) -> dict[str, Any]:
    market = _load_market_alignment()
    merged = oof.copy()
    merged["match_id"] = merged["match_id"].astype(str)
    merged = merged.merge(market, on="match_id", how="inner")
    for src in ["market_prob_a_novig", "hybrid_new_prob_a", "hybrid_old_prob_a", "exp039_prob_a"]:
        merged[f"{src}_team1"] = merged.apply(lambda r: _align_prob_to_golgg_team1(r, src), axis=1)
    out: dict[str, Any] = {
        "label": label,
        "common_mapped_rows": int(len(merged)),
        "date_min": str(pd.to_datetime(merged["date"], utc=True, errors="coerce").min()) if len(merged) else None,
        "date_max": str(pd.to_datetime(merged["date"], utc=True, errors="coerce").max()) if len(merged) else None,
        "metrics": {},
    }
    y = merged["target"].astype(int).to_numpy()
    for name, col in [
        (f"{label}_oof_calibrated", "oof_prob_calibrated"),
        (f"{label}_oof_raw", "oof_prob_raw"),
        ("market_no_vig", "market_prob_a_novig_team1"),
        ("hybrid_new_a0.05", "hybrid_new_prob_a_team1"),
        ("hybrid_old_a0.50", "hybrid_old_prob_a_team1"),
        ("exp039", "exp039_prob_a_team1"),
    ]:
        mask = merged[col].notna()
        if int(mask.sum()) == 0:
            continue
        out["metrics"][name] = _metrics(y[mask.to_numpy()], merged.loc[mask, col].to_numpy(dtype=float))
    return out


def main() -> None:
    args = parse_args()
    init_db()

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
    embedding_ds = build_match_dataset_from_embeddings(
        matches,
        embeddings,
        PlayerEmbeddingMatchDatasetConfig(history_size=args.history_size, min_prior_player_games=args.min_prior_player_games),
    )
    strength_ds = build_strength_dataset_from_db(
        StrengthDatasetConfig(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_matches)
    )
    hybrid_ds = _build_db_hybrid(embedding_ds, strength_ds)

    embedding_training = _train_oof(embedding_ds, name="PlayerEmbedding-Match-LR", args=args)
    hybrid_training = _train_oof(hybrid_ds, name="Embedding-Strength-Hybrid-LR", args=args)
    if embedding_training.oof_frame is None or hybrid_training.oof_frame is None:
        raise RuntimeError("OOF predictions were not collected")

    payload = {
        "experiment_id": "EXP-054",
        "description": "Recovered-DB evaluation of player embedding model and DB-native embedding+strength hybrid.",
        "encoder_artifact": str(Path(args.encoder_artifact)),
        "args": vars(args),
        "player_dataset_metadata": player_dataset.metadata,
        "embedding_match_dataset_metadata": embedding_ds.metadata,
        "strength_dataset_metadata": strength_ds.metadata,
        "hybrid_dataset_metadata": hybrid_ds.metadata,
        "embedding_model": {
            "config": asdict(embedding_training.metrics["config"]) if not isinstance(embedding_training.metrics.get("config"), dict) else embedding_training.metrics["config"],
            "metrics": embedding_training.metrics,
            "folds": [asdict(f) for f in embedding_training.folds],
            "market_comparison": _market_compare(embedding_training.oof_frame, "embedding"),
        },
        "embedding_strength_hybrid": {
            "metrics": hybrid_training.metrics,
            "folds": [asdict(f) for f in hybrid_training.folds],
            "market_comparison": _market_compare(hybrid_training.oof_frame, "embedding_strength_hybrid"),
        },
    }
    output = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
