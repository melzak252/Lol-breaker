from __future__ import annotations

import json

import pandas as pd

from betting_app.ml.training.player_game_dataset import build_player_game_dataset
from betting_app.ml.training.player_game_encoder import (
    PlayerGameEncoderConfig,
    build_vocabularies,
    encode_categoricals,
    prepare_numeric_matrix,
)
from betting_app.tests.test_player_game_dataset import _raw_row


def test_player_game_encoder_config_defaults_are_gpu_ready() -> None:
    cfg = PlayerGameEncoderConfig()

    assert cfg.model_name == "PlayerGameEncoder"
    assert cfg.model_version == "exp-049"
    assert cfg.device == "auto"
    assert cfg.batch_size >= 1024


def test_prepare_numeric_matrix_imputes_and_scales_without_torch() -> None:
    dataset = build_player_game_dataset(
        pd.DataFrame(
            [
                _raw_row(player_id="p1", champion_id="103"),
                _raw_row(
                    game_id="g2",
                    player_id="p2",
                    champion_id="84",
                    stats_json=json.dumps({"kills": 1, "deaths": 3, "assists": 4, "cs": 210, "golds": 11000}),
                ),
            ]
        )
    )

    matrix, stats = prepare_numeric_matrix(dataset)

    assert matrix.shape == (2, len(dataset.feature_names))
    assert not pd.isna(matrix).any()
    assert stats["feature_names"] == dataset.feature_names


def test_build_vocabularies_and_encode_categoricals_reserve_zero_for_unknown() -> None:
    dataset = build_player_game_dataset(pd.DataFrame([_raw_row(player_id="p1", champion_id="103")]))
    vocabularies = build_vocabularies(dataset)
    encoded = encode_categoricals(dataset.frame, vocabularies)

    assert vocabularies["player_id"] == {"p1": 1}
    assert encoded["player_id"].tolist() == [1]
    mutated = dataset.frame.copy()
    mutated.loc[0, "player_id"] = "unknown"
    assert encode_categoricals(mutated, vocabularies)["player_id"].tolist() == [0]
