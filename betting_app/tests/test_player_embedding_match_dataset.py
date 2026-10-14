from __future__ import annotations

import pandas as pd

from betting_app.ml.training.player_embedding_match_dataset import (
    PlayerEmbeddingMatchDatasetConfig,
    build_match_dataset_from_embeddings,
)


def test_match_embedding_dataset_uses_only_prior_dates() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 1, "team2_win": 0},
            {"match_id": "m2", "date": "2024-01-02", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 0, "team2_win": 1},
            {"match_id": "m3", "date": "2024-01-02", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 1, "team2_win": 0},
            {"match_id": "m4", "date": "2024-01-03", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 1, "team2_win": 0},
        ]
    )
    embeddings = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "game_id": "g1", "team_id": "A", "team_name": "A", "embedding_0": 1.0, "embedding_1": 2.0},
            {"match_id": "m1", "date": "2024-01-01", "game_id": "g1", "team_id": "B", "team_name": "B", "embedding_0": 10.0, "embedding_1": 20.0},
            {"match_id": "m2", "date": "2024-01-02", "game_id": "g2", "team_id": "A", "team_name": "A", "embedding_0": 3.0, "embedding_1": 4.0},
            {"match_id": "m2", "date": "2024-01-02", "game_id": "g2", "team_id": "B", "team_name": "B", "embedding_0": 30.0, "embedding_1": 40.0},
            {"match_id": "m3", "date": "2024-01-02", "game_id": "g3", "team_id": "A", "team_name": "A", "embedding_0": 5.0, "embedding_1": 6.0},
            {"match_id": "m3", "date": "2024-01-02", "game_id": "g3", "team_id": "B", "team_name": "B", "embedding_0": 50.0, "embedding_1": 60.0},
        ]
    )
    dataset = build_match_dataset_from_embeddings(
        matches,
        embeddings,
        PlayerEmbeddingMatchDatasetConfig(history_size=10, min_prior_player_games=1, include_std_features=False),
    )

    assert dataset.metadata["rows"] == 3
    by_match = dataset.frame.set_index("match_id")
    assert by_match.loc["m2", "team1_embedding_mean_0"] == 1.0
    assert by_match.loc["m3", "team1_embedding_mean_0"] == 1.0
    assert by_match.loc["m4", "team1_embedding_mean_0"] == 3.0
    assert by_match.loc["m4", "team2_embedding_mean_0"] == 30.0
    assert dataset.metadata["skipped"] == {"min_prior_player_games": 1}


def test_match_embedding_dataset_respects_min_prior_count() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 1, "team2_win": 0},
            {"match_id": "m2", "date": "2024-01-02", "team1_id": "A", "team2_id": "B", "team1_name": "A", "team2_name": "B", "team1_win": 1, "team2_win": 0},
        ]
    )
    embeddings = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "team_id": "A", "team_name": "A", "embedding_0": 1.0},
            {"match_id": "m1", "date": "2024-01-01", "team_id": "B", "team_name": "B", "embedding_0": 2.0},
        ]
    )
    dataset = build_match_dataset_from_embeddings(
        matches,
        embeddings,
        PlayerEmbeddingMatchDatasetConfig(history_size=10, min_prior_player_games=2),
    )

    assert dataset.frame.empty
    assert dataset.metadata["skipped"] == {"min_prior_player_games": 2}
