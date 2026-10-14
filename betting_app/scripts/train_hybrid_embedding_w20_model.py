"""Train EXP-051 hybrid model: EXP-050 embeddings + EXP-039/045 W20 features."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from betting_app.core.db import init_db
from betting_app.ml.training.hybrid_embedding_w20_dataset import (
    HybridEmbeddingW20Config,
    build_hybrid_embedding_w20_dataset,
    load_legacy_w20_features,
    save_hybrid_embedding_w20_dataset,
)
from betting_app.ml.training.player_embedding_match_dataset import (
    PlayerEmbeddingMatchDatasetConfig,
    build_match_dataset_from_embeddings,
    encode_player_game_embeddings,
    save_match_embedding_dataset,
)
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.strength_dataset import load_golgg_match_results
from betting_app.ml.training.strength_model import StrengthModelConfig, save_strength_artifacts, train_strength_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-artifact", required=True, help="Path to PlayerGameEncoder artifact directory")
    parser.add_argument("--data-dir", default="data", help="Directory containing golgg_y_predicts.csv, odds.csv, golgg_matches.json")
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
    parser.add_argument("--use-order-augmentation", action="store_true", help="Off by default: generic swap is unsafe for legacy t1/t2 probability features")
    parser.add_argument("--model-name", default="HybridEmbedding-W20-LR")
    parser.add_argument("--model-version", default="exp-051")
    parser.add_argument("--artifact-root", default="betting_app/models/ml")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


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

    model_config = StrengthModelConfig(
        model_name=args.model_name,
        model_version=args.model_version,
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_fold_train_size=args.min_fold_train_size,
        logistic_c=args.logistic_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        use_order_augmentation=args.use_order_augmentation,
        calibrate=True,
    )
    training = train_strength_model(hybrid_dataset, model_config)
    artifact_path: str | None = None
    if not args.no_save:
        artifact_path = str(save_strength_artifacts(dataset=hybrid_dataset, training=training, config=model_config, artifact_root=args.artifact_root))
        save_match_embedding_dataset(embedding_dataset, Path(artifact_path) / "match_embedding_dataset")
        save_hybrid_embedding_w20_dataset(hybrid_dataset, Path(artifact_path) / "hybrid_embedding_w20_dataset")

    payload: dict[str, Any] = {
        "experiment_ids": ["EXP-049", "EXP-050", "EXP-051"],
        "encoder_artifact": str(Path(args.encoder_artifact)),
        "player_dataset_metadata": player_dataset.metadata,
        "embedding_match_dataset_metadata": embedding_dataset.metadata,
        "hybrid_dataset_metadata": hybrid_dataset.metadata,
        "model_config": asdict(model_config),
        "metrics": training.metrics,
        "folds": [asdict(fold) for fold in training.folds],
        "artifact_path": artifact_path,
    }
    output = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
