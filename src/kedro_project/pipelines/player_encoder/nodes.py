"""Kedro nodes for EXP-048/EXP-049 player encoder work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from betting_app.core.db import init_db
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.player_game_encoder import PlayerGameEncoderConfig, train_player_game_encoder
from betting_app.scripts.audit_player_game_data import collect_audit


def run_player_game_audit(parameters: dict[str, Any]) -> dict[str, Any]:
    init_db()
    audit_cfg = parameters.get("audit", {})
    audit = collect_audit(limit_rows=audit_cfg.get("limit_rows"))
    output_path = audit_cfg.get("json_output")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def train_player_encoder(parameters: dict[str, Any]) -> dict[str, Any]:
    init_db()
    dataset_cfg = parameters.get("dataset", {})
    train_cfg = parameters.get("training", {})
    artifact_root = parameters.get("artifact_root", "betting_app/models/ml")
    dataset = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(
            min_date=dataset_cfg.get("min_date", "2013-01-01"),
            max_date=dataset_cfg.get("max_date"),
            limit_rows=dataset_cfg.get("limit_rows"),
            require_core_stats=dataset_cfg.get("require_core_stats", True),
        )
    )
    result = train_player_game_encoder(
        dataset,
        PlayerGameEncoderConfig(
            model_version=train_cfg.get("model_version", "exp-049"),
            embedding_dim=train_cfg.get("embedding_dim", 16),
            hidden_dim=train_cfg.get("hidden_dim", 128),
            latent_dim=train_cfg.get("latent_dim", 64),
            dropout=train_cfg.get("dropout", 0.15),
            batch_size=train_cfg.get("batch_size", 2048),
            epochs=train_cfg.get("epochs", 20),
            learning_rate=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
            validation_fraction=train_cfg.get("validation_fraction", 0.15),
            random_state=train_cfg.get("random_state", 42),
            device=train_cfg.get("device", "auto"),
            num_workers=train_cfg.get("num_workers", 0),
        ),
        artifact_root=artifact_root,
    )
    return {
        "artifact_path": str(result.artifact_path),
        "metadata": result.metadata,
        "history_tail": result.history[-5:],
    }
