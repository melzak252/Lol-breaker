import math

import pandas as pd

from betting_app.ml.training.strength_dataset import StrengthDatasetConfig, build_strength_dataset
from betting_app.ml.training.strength_model import StrengthModelConfig, swap_feature_frame, train_strength_model


def _match(
    idx: int,
    date: str,
    team1: str,
    team2: str,
    team1_score: int,
    team2_score: int,
    patch: str = "14.1",
) -> dict[str, object]:
    team1_won = team1_score > team2_score
    return {
        "match_id": str(idx),
        "date": date,
        "tournament_name": "Test League",
        "patch": patch,
        "team1_name": team1,
        "team2_name": team2,
        "team1_id": idx * 10 + 1,
        "team2_id": idx * 10 + 2,
        "team1_score": team1_score,
        "team2_score": team2_score,
        "team1_win": int(team1_won),
        "team2_win": int(not team1_won),
        "draw": 0,
        "games_played": team1_score + team2_score,
        "best_of": 3,
        "winner_name": team1 if team1_won else team2,
        "loser_name": team2 if team1_won else team1,
    }


def test_strength_dataset_uses_only_prior_matches_for_features() -> None:
    raw = pd.DataFrame(
        [
            _match(1, "2024-01-01", "Alpha", "Beta", 2, 0),
            _match(2, "2024-01-08", "Alpha", "Beta", 0, 2),
        ]
    )

    dataset = build_strength_dataset(raw, StrengthDatasetConfig(min_prior_matches=0))

    assert len(dataset.frame) == 2
    first = dataset.frame.iloc[0]
    second = dataset.frame.iloc[1]
    assert first["team1_prior_matches"] == 0.0
    assert first["team2_prior_matches"] == 0.0
    assert math.isnan(first["team1_win_rate_w5"])
    assert second["team1_prior_matches"] == 1.0
    assert second["team2_prior_matches"] == 1.0
    assert second["team1_win_rate_w5"] == 1.0
    assert second["team2_win_rate_w5"] == 0.0
    assert dataset.metadata["anti_leakage"].startswith("Features are computed before updating")


def test_strength_dataset_updates_state_for_rows_skipped_by_min_prior_matches() -> None:
    raw = pd.DataFrame(
        [
            _match(1, "2024-01-01", "Alpha", "Beta", 2, 0),
            _match(2, "2024-01-08", "Alpha", "Beta", 2, 1),
        ]
    )

    dataset = build_strength_dataset(raw, StrengthDatasetConfig(min_prior_matches=1))

    assert len(dataset.frame) == 1
    row = dataset.frame.iloc[0]
    assert row["match_id"] == "2"
    assert row["team1_prior_matches"] == 1.0
    assert row["team2_prior_matches"] == 1.0
    assert dataset.metadata["skipped"] == {"min_prior_matches": 1}


def test_swap_feature_frame_is_symmetric() -> None:
    features = pd.DataFrame(
        [
            {
                "team1_elo": 1600.0,
                "team2_elo": 1500.0,
                "elo_diff": 100.0,
                "team1_prior_matches": 20.0,
                "team2_prior_matches": 12.0,
                "prior_matches_diff": 8.0,
                "team1_win_rate_w5": 0.8,
                "team2_win_rate_w5": 0.4,
                "win_rate_diff_w5": 0.4,
                "elo_expected_team1": 0.64,
                "h2h_team1_win_rate": 0.75,
            }
        ]
    )

    swapped = swap_feature_frame(features)
    assert swapped.loc[0, "team1_elo"] == 1500.0
    assert swapped.loc[0, "team2_elo"] == 1600.0
    assert swapped.loc[0, "elo_diff"] == -100.0
    assert swapped.loc[0, "win_rate_diff_w5"] == -0.4
    assert swapped.loc[0, "elo_expected_team1"] == 0.36
    assert swapped.loc[0, "h2h_team1_win_rate"] == 0.25
    pd.testing.assert_frame_equal(swap_feature_frame(swapped), features)


def test_train_strength_model_smoke_walk_forward() -> None:
    raw = pd.DataFrame(
        [
            _match(
                idx=i,
                date=f"2024-01-{(i - 1) % 28 + 1:02d}",
                team1="Alpha" if i % 2 else "Beta",
                team2="Beta" if i % 2 else "Alpha",
                team1_score=2 if i % 3 else 1,
                team2_score=1 if i % 3 else 2,
            )
            for i in range(1, 90)
        ]
    )
    dataset = build_strength_dataset(raw, StrengthDatasetConfig(min_prior_matches=1))

    result = train_strength_model(
        dataset,
        StrengthModelConfig(
            initial_train_size=30,
            test_size=20,
            step_size=20,
            min_fold_train_size=20,
            calibrate=False,
        ),
    )

    assert result.metrics["rows"] == len(dataset.frame)
    assert result.metrics["feature_count"] == len(dataset.feature_names)
    assert result.metrics["fold_count"] >= 1
    assert result.metrics["oof_count"] > 0
