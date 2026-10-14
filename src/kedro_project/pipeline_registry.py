"""Project pipeline registry."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from kedro_project.pipelines import evaluation, weekly_retrain
from kedro_project.pipelines.player_encoder.pipeline import create_pipeline as create_player_encoder_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    weekly = weekly_retrain.create_pipeline()
    historical_evaluation = evaluation.create_pipeline()
    player_encoder = create_player_encoder_pipeline()
    return {
        "__default__": weekly + historical_evaluation + player_encoder,
        "weekly_retrain": weekly,
        "evaluation": historical_evaluation,
        "player_encoder": player_encoder,
        "exp048_player_encoder": player_encoder,
    }
