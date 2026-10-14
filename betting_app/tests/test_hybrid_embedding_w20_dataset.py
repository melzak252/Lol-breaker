from __future__ import annotations

import math

import numpy as np
import pandas as pd

from betting_app.ml.training.hybrid_embedding_w20_dataset import (
    HybridEmbeddingW20Config,
    add_binomial_features,
    build_hybrid_embedding_w20_dataset,
    series_probability,
)
from betting_app.ml.training.player_embedding_match_dataset import PlayerEmbeddingMatchDataset


def test_series_probability_handles_best_of_values() -> None:
    probabilities = series_probability(np.array([0.5, 0.6, 0.6]), np.array([1, 3, 5]))

    assert probabilities[0] == 0.5
    assert math.isclose(probabilities[1], 0.648, rel_tol=1e-9)
    assert math.isclose(probabilities[2], 0.68256, rel_tol=1e-9)


def test_add_binomial_features_uses_bon_column() -> None:
    frame = pd.DataFrame({"player_elo": [0.6], "BoN": [3]})

    out, names = add_binomial_features(frame, ["player_elo", "missing"])

    assert names == ["player_elo_binom_series"]
    assert math.isclose(out.loc[0, "player_elo_binom_series"], 0.648, rel_tol=1e-9)


def test_build_hybrid_embedding_w20_dataset_merges_and_filters_target_disagreements() -> None:
    embedding_frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2024-01-01",
                "target": 1,
                "team1_embedding_count": 10.0,
                "team2_embedding_count": 12.0,
                "embedding_count_diff": -2.0,
                "embedding_mean_diff_0": 0.3,
            },
            {
                "match_id": "m2",
                "date": "2024-01-02",
                "target": 0,
                "team1_embedding_count": 11.0,
                "team2_embedding_count": 9.0,
                "embedding_count_diff": 2.0,
                "embedding_mean_diff_0": -0.2,
            },
        ]
    )
    embedding_dataset = PlayerEmbeddingMatchDataset(
        frame=embedding_frame,
        feature_names=["team1_embedding_count", "team2_embedding_count", "embedding_count_diff", "embedding_mean_diff_0"],
        metadata={"experiment_id": "EXP-050"},
    )
    legacy = pd.DataFrame(
        [
            {"golgg_match_id": "m1", "date": "2024-01-01", "y_true": 1, "player_elo": 0.65, "t1_rolling_win_rate": 0.7},
            {"golgg_match_id": "m2", "date": "2024-01-02", "y_true": 1, "player_elo": 0.40, "t1_rolling_win_rate": 0.4},
            {"golgg_match_id": "m3", "date": "2024-01-03", "y_true": 0, "player_elo": 0.50, "t1_rolling_win_rate": 0.5},
        ]
    )

    dataset = build_hybrid_embedding_w20_dataset(
        embedding_dataset,
        legacy,
        ["player_elo", "t1_rolling_win_rate"],
        HybridEmbeddingW20Config(min_date="2024-01-01", require_target_agreement=True),
    )

    assert len(dataset.frame) == 1
    assert dataset.frame.loc[0, "match_id"] == "m1"
    assert dataset.feature_names == [
        "team1_embedding_count",
        "team2_embedding_count",
        "embedding_count_diff",
        "embedding_mean_diff_0",
        "player_elo",
        "t1_rolling_win_rate",
    ]
    assert dataset.metadata["skipped"] == {"target_disagreement": 1}
