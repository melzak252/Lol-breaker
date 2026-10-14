"""Pipeline definition for EXP-048 audit and EXP-049 encoder training."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from kedro_project.pipelines.player_encoder.nodes import run_player_game_audit, train_player_encoder


def create_pipeline(**_: object) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_player_game_audit,
                inputs="params:exp048_player_encoder",
                outputs="player_game_audit",
                name="run_player_game_audit",
            ),
            node(
                func=train_player_encoder,
                inputs="params:exp048_player_encoder",
                outputs="player_encoder_training_result",
                name="train_player_encoder",
            ),
        ]
    )
