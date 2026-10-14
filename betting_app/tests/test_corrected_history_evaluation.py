"""Offline corrected-history contracts; no database or network required."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from betting_app.tests.test_siamese_research_dataset import match
from scripts.corrected_history_evaluation import (
    build_exp039_features,
    chronological_masks,
    replay_ratings,
    replay,
    validate_series,
    verify_replay,
)
from scripts.build_siamese_research_dataset import build_dataset


def test_same_day_results_cannot_change_any_same_day_rating_or_w20_snapshot():
    history = [match(1, "2023-01-01"), match(2, "2023-01-02"), match(3, "2023-01-02"), match(4, "2023-01-03")]
    original = replay_ratings(history)
    changed = deepcopy(history)
    changed[1] = match(2, "2023-01-02", winner=False, kills=100)
    other = replay_ratings(changed)
    features = [c for c in original if c != "y_true"]
    pd.testing.assert_frame_equal(original.iloc[:3][features], other.iloc[:3][features])
    assert original.iloc[3].team_elo != other.iloc[3].team_elo
    assert original.iloc[1].days_since_last_1 == original.iloc[2].days_since_last_1 == 1
    first, _ = build_dataset(original, history, day_batched_ratings=True)
    second, _ = build_dataset(other, changed, day_batched_ratings=True)
    cols = [c for c in first if "rolling" in c]
    pd.testing.assert_frame_equal(first.iloc[:2][cols], second.iloc[:2][cols])
    assert first.golgg_match_id.tolist() == ["2", "3", "4"]


def test_map_side_changes_do_not_change_series_ratings():
    original = [match(1, "2023-01-01"), match(2, "2023-01-09")]
    swapped = deepcopy(original)
    game = swapped[0]["games"][0]
    for field in ("id", "win", "players", "stats"):
        game[f"t1_{field}"], game[f"t2_{field}"] = game[f"t2_{field}"], game[f"t1_{field}"]
    pd.testing.assert_frame_equal(replay_ratings(original), replay_ratings(swapped))


def test_actual_inactivity_increases_glicko_deviation():
    short = replay_ratings([match(1, "2023-01-01"), match(2, "2023-01-02")])
    long = replay_ratings([match(1, "2023-01-01"), match(2, "2023-04-01")])
    assert long.iloc[1].team_gl_rd1 > short.iloc[1].team_gl_rd1
    assert long.iloc[1].days_since_last_1 == 90


@pytest.mark.parametrize("fault", ["identity", "roster", "winner", "score"])
def test_invalid_results_fail_before_rating_updates(fault):
    row = match(1, "2023-01-01")
    game = row["games"][0]
    if fault == "identity":
        game["t2_id"] = game["t1_id"]
    elif fault == "roster":
        game["t2_players"] = deepcopy(game["t1_players"])
    elif fault == "winner":
        game["t2_win"] = game["t1_win"]
    else:
        row["t1_score"] = 0
    with pytest.raises(ValueError):
        validate_series(row)


def test_fold_boundaries_never_share_fit_calibration_or_test_days():
    dates = pd.Series(["2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2025-01-01"])
    fit, calibration, test = chronological_masks(dates, 2024)
    assert fit.tolist() == [True, False, False, False, False]
    assert calibration.tolist() == [False, True, True, False, False]
    assert test.tolist() == [False, False, False, True, False]


def test_feature_contracts_have_exact_46_and_79_dimensions_and_swap_involution():
    from scripts.train_and_tune_siamese_series import build_training_features
    from src.models.team_order import swap_orientation
    raw = [match(1, "2023-01-01"), match(2, "2023-01-02")]
    frame, _ = build_dataset(replay_ratings(raw), raw, day_batched_ratings=True)
    x39, names, rank = build_exp039_features(frame)
    x81, canonical = build_training_features(frame)
    assert len(names) == 46 and x81.shape == (1, 79) and len(canonical) == 79
    flipped = swap_orientation(x39, names, rank, np.ones(len(frame), dtype=bool))
    restored = swap_orientation(flipped, names, rank, np.ones(len(frame), dtype=bool))
    np.testing.assert_allclose(restored[names], x39[names])
    assert flipped.player_gl.iloc[0] == pytest.approx(1 - x39.player_gl.iloc[0])


def test_legacy_or_tampered_replay_is_not_accepted(tmp_path):
    (tmp_path / "audit.json").write_text('{"ratings_version": "legacy"}')
    with pytest.raises(ValueError, match="corrected replay"):
        verify_replay(tmp_path)


def test_replay_refuses_overwrite_and_verifies_source_and_output_hashes(tmp_path):
    import json
    source = tmp_path / "fresh_history.json"
    source.write_text(json.dumps([match(1, "2023-01-01"), match(2, "2023-01-02")]))
    output = tmp_path / "replay"
    replay(source, output)
    audit = verify_replay(output)
    assert audit["source"]["raw_series"] == audit["state_transition_series"] == 2
    assert audit["frozen_artifacts_before"] == audit["frozen_artifacts_after"]
    original = (output / "ratings.csv").read_bytes()
    with pytest.raises(FileExistsError):
        replay(source, output)
    assert (output / "ratings.csv").read_bytes() == original
    (output / "ratings.csv").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_replay(output)


def test_tied_bo2_updates_ratings_and_w20_without_binary_scoring_row():
    tied = match(1, "2023-01-01")
    tied.update(best_of=2, t1_score=1, t2_score=1, t1_win=False, t2_win=False, draw=True)
    tied["games"].append(match(2, "2023-01-01", winner=False, kills=30)["games"][0])
    following = match(3, "2023-01-02")
    ratings = replay_ratings([tied, following])
    assert ratings.golgg_match_id.tolist() == ["3"]
    assert ratings.iloc[0].team_gl_rd1 < 350
    assert ratings.iloc[0].days_since_last_1 == 1
    snapshots, _ = build_dataset(ratings, [tied, following], day_batched_ratings=True)
    assert snapshots.iloc[0].history_games_1 == 2
    assert snapshots.iloc[0].t1_rolling_win_rate == .5
    assert snapshots.iloc[0].t1_rolling_kills == 20


def test_substitute_receives_only_their_actual_map_update():
    series = match(1, "2023-01-01")
    series.update(best_of=3, t1_score=2)
    second_map = match(2, "2023-01-01")["games"][0]
    second_map["t1_players"]["0"]["player_id"] = "substitute"
    series["games"].append(second_map)
    following = match(3, "2023-01-02")
    following["games"][0]["t1_players"]["0"]["player_id"] = "substitute"
    with_sub = replay_ratings([series, following]).iloc[1]
    no_sub_series = deepcopy(series)
    no_sub_series["games"][1]["t1_players"]["0"]["player_id"] = "0"
    without_sub = replay_ratings([no_sub_series, following]).iloc[1]
    assert with_sub.player_elo_min1 > without_sub.player_elo_min1
    assert with_sub.player_gl_rd_avg1 < without_sub.player_gl_rd_avg1
    assert with_sub.team_elo == pytest.approx(without_sub.team_elo)


def test_collection_quarantine_remains_visible_and_pinned_in_replay(tmp_path):
    import json

    source = tmp_path / "fresh_history.json"
    source.write_text(json.dumps([match(1, "2023-01-01"), match(2, "2023-01-02")]))
    manifest = source.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "status": "incomplete",
        "counts": {"discovered_matches": 3, "clean_matches": 2,
                   "unresolved_matches": 1, "pending_matches": 0, "discovery_failures": 0},
    }))
    source.with_suffix(".quarantine.json").write_text(json.dumps([
        {"match_id": "3", "status": "unresolved", "reasons": ["source has no result"]},
    ]))
    output = tmp_path / "replay"
    replay(source, output)
    audit = verify_replay(output)
    assert audit["collection"]["counts"]["unresolved_matches"] == 1
    assert audit["collection"]["complete"] is False
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="collection provenance changed"):
        verify_replay(output)


def test_later_map_substitute_receives_prior_series_experience():
    series = match(1, "2023-01-01")
    series.update(best_of=3, t1_score=2)
    second_map = match(2, "2023-01-01")["games"][0]
    second_map["t1_players"]["0"]["player_id"] = "substitute"
    series["games"].append(second_map)
    following = match(3, "2023-01-02")
    following["games"][0]["t1_players"]["0"]["player_id"] = "substitute"
    ratings = replay_ratings([series, following])
    snapshots, _ = build_dataset(ratings, [series, following], day_batched_ratings=True)
    assert snapshots.iloc[0].roster_min_prior_series == 1


def test_weighted_bo5_series_with_initial_wins_is_valid_and_completed_in_two_played_wins():
    row = match(1, "2023-01-01")
    second = deepcopy(row["games"][0])
    second["game_id"] = "2"
    row["games"].append(second)
    row.update(best_of=5, t1_score=2, t2_score=0, t1_win=True, t2_win=False, initial_wins={"a": 1})
    scores = validate_series(row)
    assert scores == [1, 1]


def test_extra_map_after_weighted_bo5_is_rejected():
    row = match(1, "2023-01-01")
    for gid in ("2", "3"):
        g = deepcopy(row["games"][0])
        g["game_id"] = gid
        row["games"].append(g)
    row.update(best_of=5, t1_score=2, t2_score=0, t1_win=True, t2_win=False, initial_wins={"a": 1})
    with pytest.raises(ValueError, match="extra map after completed series"):
        validate_series(row)


@pytest.mark.parametrize("bad_adv", [{"a": -1}, {"a": 3}, {"a": 1, "b": 1}])
def test_invalid_initial_wins_are_rejected(bad_adv):
    row = match(1, "2023-01-01")
    second = deepcopy(row["games"][0])
    second["game_id"] = "2"
    row["games"].append(second)
    row.update(best_of=5, t1_score=2, t2_score=0, t1_win=True, t2_win=False, initial_wins=bad_adv)
    with pytest.raises(ValueError, match="invalid initial map advantage"):
        validate_series(row)


def test_chronological_fold_excludes_late_completed_labels_and_keeps_calibration_disjoint():
    from scripts.corrected_history_evaluation import chronological_fold_masks
    frame = pd.DataFrame({
        "date": ["2018-01-01", "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01"],
        "result_day": ["2018-01-01", "2019-01-01", "2019-01-01", "2020-01-01", "2020-01-01"],
    })
    fit, calibration = chronological_fold_masks(
        frame, origin="2020-01-01T00:00:00Z",
        calibration_start="2019-01-01", fit_start="2018-01-01",
    )
    assert fit.tolist() == [True, False, False, False, False]
    assert calibration.tolist() == [False, False, True, False, False]


def test_new_chronological_replay_is_immutable_and_rejects_changed_snapshots(tmp_path):
    import json
    from scripts.corrected_history_evaluation import chronological_replay, verify_chronological_replay
    source, output = tmp_path / "source.json", tmp_path / "replay"
    source.write_text(json.dumps([match(1, "2018-01-01"), match(2, "2019-01-01")]))
    chronological_replay(source, output)
    verified = verify_chronological_replay(output)
    assert verified["included_rows"] == 1
    with pytest.raises(FileExistsError):
        chronological_replay(source, output)
    with (output / "snapshots.csv").open("a") as handle:
        handle.write("\nmodified")
    with pytest.raises(ValueError, match="checksum"):
        verify_chronological_replay(output)


def test_partial_history_provenance_is_preserved_and_cannot_be_rebound(tmp_path):
    import json
    from scripts.corrected_history_evaluation import chronological_replay, sha256, verify_chronological_replay
    source, provenance = tmp_path / "partial.json", tmp_path / "projection.json"
    source.write_text(json.dumps([match(1, "2018-01-01"), match(2, "2019-01-01")]))
    evidence = {
        "qualification": "retrospective_incomplete_history_diagnostic_only",
        "projection_sha256": sha256(source), "counts": {"existing_quarantined_prior_series": 265},
    }
    provenance.write_text(json.dumps(evidence))
    output = tmp_path / "replay"
    chronological_replay(source, output, history_provenance=provenance)
    audit = verify_chronological_replay(output)
    assert audit["history_provenance"]["evidence"] == evidence
    assert audit["collection_completeness_certified"] is False
    evidence["counts"]["existing_quarantined_prior_series"] = 0
    provenance.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="provenance"):
        verify_chronological_replay(output)
