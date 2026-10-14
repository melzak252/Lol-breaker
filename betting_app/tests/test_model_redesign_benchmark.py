import numpy as np
import pandas as pd

from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks


def test_fit_stopping_selection_calibration_and_test_dates_are_disjoint():
    dates = pd.Series(
        [
            "2019-12-31",
            "2022-12-31",
            "2023-01-01",
            "2023-06-30",
            "2023-07-01",
            "2023-09-30",
            "2023-10-01",
            "2023-12-31",
            "2024-01-01",
            "2024-12-31",
            "2025-01-01",
        ]
    )
    blocks = temporal_blocks(dates, 2024)
    expected = {
        "train": [1],
        "stop": [2, 3],
        "select": [4, 5],
        "calibration": [6, 7],
        "test": [8, 9],
    }
    for name, positions in expected.items():
        assert np.flatnonzero(blocks[name]).tolist() == positions
    assert np.stack(list(blocks.values())).sum(axis=0).max() == 1


def test_semester_calibration_excludes_selection_and_future_outcomes():
    dates = pd.Series(
        [
            "2022-12-31",
            "2023-03-31",
            "2023-04-01",
            "2023-06-30",
            "2023-07-01",
            "2023-12-31",
            "2024-01-01",
        ]
    )
    blocks = temporal_blocks(dates, 2024, protocol="semester-calibration")
    expected = {
        "train": [0],
        "stop": [1],
        "select": [2, 3],
        "calibration": [4, 5],
        "test": [6],
    }
    for name, positions in expected.items():
        assert np.flatnonzero(blocks[name]).tolist() == positions
    assert np.stack(list(blocks.values())).sum(axis=0).max() == 1


def test_earlier_training_start_keeps_calibration_and_test_outcomes_separate():
    dates = pd.Series(
        [
            "2017-12-31",
            "2018-01-01",
            "2018-12-31",
            "2019-01-01",
            "2019-04-01",
            "2019-07-01",
            "2019-12-31",
            "2020-01-01",
            "2020-12-31",
            "2021-01-01",
        ]
    )
    blocks = temporal_blocks(
        dates, 2020, protocol="semester-calibration", train_start="2018-01-01"
    )
    expected = {
        "train": [1, 2],
        "stop": [3],
        "select": [4],
        "calibration": [5, 6],
        "test": [7, 8],
    }
    for name, positions in expected.items():
        assert np.flatnonzero(blocks[name]).tolist() == positions
    assert np.stack(list(blocks.values())).sum(axis=0).max() == 1


def test_replay_context_alignment_preserves_legacy_orientation():
    legacy = pd.DataFrame(
        [
            dict(
                golgg_match_id="1",
                date="2024-01-01",
                team1_id="a",
                team2_id="b",
                best_of=3,
                y_true=1,
            )
        ]
    )
    replay = pd.DataFrame(
        [
            dict(
                golgg_match_id="1",
                date="2024-01-01",
                team1_id="b",
                team2_id="a",
                best_of=3,
                y_true=0,
                d_strength=-2.0,
                c_history=10.0,
                score_a=0,
                score_b=2,
                roster_min_prior_series=9,
                map_prob_team=0.2,
                map_prob_player=0.3,
            )
        ]
    )
    frame, audit = align_legacy_context(legacy, replay)
    assert frame.loc[0, "d_replay_strength"] == 2.0
    assert frame.loc[0, "c_replay_history"] == 10.0
    assert frame.loc[0, "score_a"] == 2 and frame.loc[0, "score_b"] == 0
    assert audit["reversed_rows"] == 1


def test_corrected_date_or_target_is_excluded_with_reason_not_forced_join():
    legacy = pd.DataFrame(
        [
            dict(
                golgg_match_id="1",
                date="2024-01-01",
                team1_id="a",
                team2_id="b",
                best_of=3,
                y_true=1,
            )
        ]
    )
    replay = pd.DataFrame(
        [
            dict(
                golgg_match_id="1",
                date="2024-01-02",
                team1_id="a",
                team2_id="b",
                best_of=3,
                y_true=1,
                d_strength=2.0,
                c_history=10.0,
                score_a=2,
                score_b=0,
                roster_min_prior_series=9,
                map_prob_team=0.8,
                map_prob_player=0.7,
            )
        ]
    )
    frame, audit = align_legacy_context(legacy, replay)
    assert frame.empty
    assert audit["date_mismatch"] == 1
