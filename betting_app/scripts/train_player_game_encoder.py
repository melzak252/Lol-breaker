"""Train EXP-049 PlayerGameEncoder.

This is the CLI fallback for environments where Kedro is not installed.  The
same pure training functions are used by the Kedro pipeline nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from betting_app.core.db import init_db
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.player_game_encoder import PlayerGameEncoderConfig, train_player_game_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default="2013-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke-test row limit; omit for full training")
    parser.add_argument("--keep-sparse-rows", action="store_true", help="Do not drop rows missing core K/D/A/CS/gold/damage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--model-version", default="exp-049")
    parser.add_argument("--artifact-root", type=Path, default=Path("betting_app/models/ml"))
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    init_db()
    dataset = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(
            min_date=args.min_date,
            max_date=args.max_date,
            limit_rows=args.limit_rows,
            require_core_stats=not args.keep_sparse_rows,
        )
    )
    result = train_player_game_encoder(
        dataset,
        PlayerGameEncoderConfig(
            model_version=args.model_version,
            epochs=args.epochs,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            device=args.device,
        ),
        artifact_root=args.artifact_root,
    )
    payload = {
        "artifact_path": str(result.artifact_path),
        "metadata": result.metadata,
        "history_tail": result.history[-5:],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
