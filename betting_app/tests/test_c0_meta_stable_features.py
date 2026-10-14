"""Fixed 180-day player support, unchanged 30-day meta, and causal activity."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest

from src.features.c0_meta_features import build_meta_features
from src.features.c0_meta_stable_features import build_stable_meta_features


META_NAMES = ("meta_stats", "meta_champions", "meta_support", "meta_patch")
PLAYER_NAMES = ("player_stats", "player_roles", "player_champions", "player_valid", "player_activity")
CUTOFF = date(2025, 2, 1)


def observation(series="past", player="p", age=12, value=2.0, **changes):
    played = CUTOFF - timedelta(days=age)
    record = dict(series_id=series, played_day=played.isoformat(),
                  release_day=(played + timedelta(days=1)).isoformat(), competition="LCK",
                  patch="15.1", player_id=player, role="top", champion_id="A",
                  values12=[value] * 12, observed12=[True] * 12)
    record.update(changes)
    return record


def target(row="target", **changes):
    record = dict(row_id=row, cutoff_day=CUTOFF.isoformat(), competition="LCK",
                  roster_a=["p", None, None, None, None], roster_b=[None] * 5)
    record.update(changes)
    return record


def build(records, targets=None, *, native_valid=None, native_roles=None, champion_ids=("A", "B")):
    rows = [target()] if targets is None else targets
    valid = np.zeros((len(rows), 2, 5), dtype=np.bool_) if native_valid is None else native_valid
    roles = np.zeros((len(rows), 2, 5, 5), dtype=np.float32) if native_roles is None else native_roles
    return build_stable_meta_features(records, rows, champion_ids=champion_ids,
                                      native_valid=valid, native_roles=roles)


def test_long_native_unknown_and_known_without_representation_are_distinct():
    records = [observation("long", "long", age=90, role="mid"),
               observation("old", "old", age=300)]
    rows = [target(roster_a=["long", "native", "never", "old", None])]
    valid = np.zeros((1, 2, 5), dtype=np.bool_)
    valid[0, 0, 1] = True
    roles = np.zeros((1, 2, 5, 5), dtype=np.float32)
    roles[0, 0, 1, 1] = .75  # Unknown native-role mass remains absent.
    roles[0, 0, 2, 4] = 1  # Unsupported caller roles must not invent support.
    result = build(records, rows, native_valid=valid, native_roles=roles)
    np.testing.assert_array_equal(result.arrays["player_valid"][0, 0], [True, True, False, False, False])
    activity = result.arrays["player_activity"][0, 0]
    np.testing.assert_allclose(activity[0], [0, np.log1p(1), np.log1p(90), 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(activity[1], [0, 0, 0, 0, 0, 0, 1, 1])
    np.testing.assert_array_equal(activity[2], np.zeros(8))
    np.testing.assert_allclose(activity[3], [0, 0, np.log1p(300), 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(activity[4], np.zeros(8))
    np.testing.assert_array_equal(result.arrays["player_roles"][0, 0, 0], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(result.arrays["player_roles"][0, 0, 1], [0, .75, 0, 0, 0])
    assert not result.arrays["player_roles"][0, 0, 2:].any()
    assert not result.arrays["player_stats"][0, 0, 1:].any()
    assert not result.arrays["player_champions"][0, 0, 1:].any()
    assert not result.arrays["meta_stats"].any()
    assert result.diagnostics[0]["long_player_source_indices"][0] == [[0], [], [], [], []]
    prior = result.diagnostics[0]["prior_activity"][0]
    assert prior[3]["latest_played_day"] == records[1]["played_day"]
    assert prior[3]["max_effective_release_day"] == records[1]["release_day"]
    assert prior[2]["latest_played_day"] is None


def test_exact_180_day_boundaries_and_30_day_meta_match_original_full_catalog():
    records = [observation("lower180", age=180), observation("outside180", age=181),
               observation("lower30", age=30, value=4), observation("outside30", age=31, value=8),
               observation("played-equal", age=0, release_day="2025-01-31"),
               observation("release-equal", age=10, release_day=CUTOFF.isoformat())]
    rows = [target(), target("next", cutoff_day="2025-02-02")]
    result = build(records, rows)
    original = build_meta_features(records, rows, champion_ids=("A", "B"))
    assert isinstance(result.sources, dict)
    for source_index in (0, 2, 3, 4, 5):
        assert result.sources[source_index] == original.sources[source_index]
    for name in META_NAMES:
        np.testing.assert_array_equal(result.arrays[name], original.arrays[name])
    for i, diagnostic in enumerate(original.diagnostics):
        for key, value in diagnostic.items():
            assert result.diagnostics[i][key] == value
    assert result.diagnostics[0]["long_player_source_indices"][0][0] == [0, 2, 3]
    assert result.diagnostics[1]["long_player_source_indices"][0][0] == [2, 3, 4, 5]
    np.testing.assert_allclose(result.arrays["player_activity"][0, 0, 0, :2], [np.log1p(1), np.log1p(3)])
    np.testing.assert_allclose(result.arrays["player_stats"][0, 0, 0, :12], 14 / 3)
    assert result.diagnostics[0]["long_window_start_day"] == "2024-08-05"


def test_late_release_does_not_rejuvenate_played_time_or_last_played():
    records = [observation("old-late", age=400, release_day="2025-01-31"),
               observation("newer-play", age=200),
               observation("long-late", age=100, player="q", release_day="2025-01-31"),
               observation("hidden", age=1, release_day=CUTOFF.isoformat())]
    rows = [target(roster_a=["p", "q", None, None, None])]
    result = build(records, rows)
    assert not result.arrays["player_valid"][0, 0, 0]
    assert result.arrays["player_valid"][0, 0, 1]
    np.testing.assert_allclose(result.arrays["player_activity"][0, 0, :2, 2], np.log1p([200, 100]))
    prior = result.diagnostics[0]["prior_activity"][0][0]
    assert prior["latest_played_day"] == records[1]["played_day"]
    assert prior["max_effective_release_day"] == "2025-01-31"
    assert prior["latest_played_source_index"] == 1
    assert prior["max_effective_release_source_index"] == 0
    assert prior["prior_source_count"] == 2
    assert result.diagnostics[0]["long_player_source_indices"][0][:2] == [[], [2]]


def test_long_roles_unknown_mass_champion_absence_and_oov_preserve_original_meanings():
    records = [observation("known", age=80, role="MID"),
               observation("oov", age=90, role=None, champion_id="Z", values12=[6] * 12),
               observation("absent", age=100, role="not-a-role", champion_id=None,
                           values12=[np.nan] * 12, observed12=[False] * 12)]
    valid = np.ones((1, 2, 5), dtype=np.bool_)
    roles = np.zeros((1, 2, 5, 5), dtype=np.float32)
    roles[..., 4] = 1
    result = build(records, native_valid=valid, native_roles=roles)
    np.testing.assert_allclose(result.arrays["player_roles"][0, 0, 0], [0, 0, 1 / 3, 0, 0])
    np.testing.assert_array_equal(result.arrays["player_champions"][0, 0, 0], [.5, .5, 0])
    summary = result.arrays["player_stats"][0, 0, 0]
    np.testing.assert_allclose(summary[:12], 4)
    np.testing.assert_allclose(summary[12:24], 2)
    np.testing.assert_allclose(summary[24:36], 2 / 3)
    assert result.arrays["player_activity"][0, 0, 0, 6] == 1
    assert not result.arrays["meta_support"].any()


def test_source_permutation_future_and_target_fields_do_not_change_features():
    records = [observation("b", age=80, value=6), observation("a", age=10, value=2),
               observation("ancient", age=800)]
    rows = [target("later", cutoff_day="2025-02-03"), target("earlier"), target("earlier")]
    baseline = build(records, rows)
    permuted = build(list(reversed(records)), rows)
    future = observation("future", age=0, value=999, champion_id="future", role="support")
    hidden = observation("unreleased", age=5, release_day="2025-03-01", value=-999)
    changed_rows = [dict(row, patch="99.1", actual_roles=["support"] * 10,
                         draft=["future"] * 10, outcome=1) for row in rows]
    changed = build(records + [future, hidden], changed_rows)
    # The extra record remains future relative to every tested cutoff.
    future_only = build(records + [hidden], changed_rows)
    assert baseline.row_ids == ("later", "earlier", "earlier")
    for name in PLAYER_NAMES + META_NAMES:
        np.testing.assert_array_equal(baseline.arrays[name], permuted.arrays[name])
        np.testing.assert_array_equal(baseline.arrays[name], future_only.arrays[name])
        np.testing.assert_array_equal(baseline.arrays[name][1], changed.arrays[name][1])
    for index, row in enumerate(rows):
        alone = build(records, [row])
        for name in PLAYER_NAMES + META_NAMES:
            np.testing.assert_array_equal(baseline.arrays[name][index], alone.arrays[name][0])
    members = permuted.diagnostics[0]["long_player_source_indices"][0][0]
    assert {permuted.sources[i]["series_id"] for i in members} == {"a", "b"}


def test_player_order_native_alignment_and_side_swap_do_not_change_meta():
    records = [observation("p", age=80, role="mid"), observation("q", "q", age=10, value=8)]
    ab = target(roster_a=["q", "p", "native", "substitute", None], roster_b=["p", None, None, None, None])
    ba = target(roster_a=ab["roster_b"], roster_b=ab["roster_a"])
    valid = np.zeros((2, 2, 5), dtype=np.bool_)
    valid[0, 0, 2] = valid[1, 1, 2] = True
    roles = np.zeros((2, 2, 5, 5), dtype=np.float32)
    roles[0, 0, 2, 1] = roles[1, 1, 2, 1] = .5
    result = build(records, [ab, ba], native_valid=valid, native_roles=roles)
    for name in PLAYER_NAMES:
        np.testing.assert_array_equal(result.arrays[name][0, [1, 0]], result.arrays[name][1])
    for name in META_NAMES:
        np.testing.assert_array_equal(result.arrays[name][0], result.arrays[name][1])
    assert result.diagnostics[0]["long_player_source_indices"][0][:2] == [[1], [0]]


def test_single_pass_inputs_nonmutation_empty_and_original_batch_shapes():
    records = [observation(age=80)]
    rows = [target()]
    records_before, rows_before = deepcopy(records), deepcopy(rows)
    valid = np.zeros((1, 2, 5), dtype=np.bool_)
    roles = np.zeros((1, 2, 5, 5), dtype=np.float32)
    valid_before, roles_before = valid.copy(), roles.copy()
    result = build_stable_meta_features(iter(records), iter(rows), champion_ids=("A", "B"),
                                        native_valid=valid, native_roles=roles)
    assert records == records_before and rows == rows_before
    np.testing.assert_array_equal(valid, valid_before)
    np.testing.assert_array_equal(roles, roles_before)
    assert set(result.arrays) == set(PLAYER_NAMES + META_NAMES)
    assert result.arrays["player_activity"].shape == (1, 2, 5, 8)
    for name, array in result.arrays.items():
        assert array.dtype == (np.bool_ if name == "player_valid" else np.float32)
        assert np.isfinite(array).all()
    empty = build([])
    assert all(not array.any() for array in empty.arrays.values())
    no_targets = build(records, [])
    assert no_targets.row_ids == ()
    assert all(array.shape[0] == 0 for array in no_targets.arrays.values())
    assert no_targets.sources == {}


@pytest.mark.parametrize("valid,roles", [
    (np.zeros((1, 2, 4), dtype=np.bool_), np.zeros((1, 2, 5, 5), dtype=np.float32)),
    (np.zeros((1, 2, 5), dtype=np.float32), np.zeros((1, 2, 5, 5), dtype=np.float32)),
    (np.zeros((1, 2, 5), dtype=np.bool_), np.zeros((1, 2, 5, 4), dtype=np.float32)),
    (np.zeros((1, 2, 5), dtype=np.bool_), np.full((1, 2, 5, 5), np.nan, dtype=np.float32)),
    (np.zeros((1, 2, 5), dtype=np.bool_), np.full((1, 2, 5, 5), -.1, dtype=np.float32)),
    (np.zeros((1, 2, 5), dtype=np.bool_), np.ones((1, 2, 5, 5), dtype=np.float32)),
])
def test_invalid_native_shapes_types_and_role_frequencies_are_rejected(valid, roles):
    with pytest.raises((TypeError, ValueError)):
        build([], native_valid=valid, native_roles=roles)


@pytest.mark.parametrize("champions", [("A", "A"), (None, "A"), (0, "A")])
def test_invalid_fixed_vocabulary_is_rejected(champions):
    with pytest.raises(ValueError):
        build([], champion_ids=champions)


def test_invalid_stats_and_rosters_are_rejected_even_outside_meta_window():
    with pytest.raises(ValueError):
        build([observation(age=80, values12=[np.nan] * 12)])
    with pytest.raises(ValueError):
        build([], [target(roster_a=["p"])])


def test_old_activity_only_history_is_streamed_without_stat_normalization_or_dense_catalog():
    records = [
        dict(series_id="ancient", player_id="p", played_day="2013-08-17", release_day="2013-08-18"),
        dict(series_id="older", player_id="p", played_day="2014-08-17", release_day="2014-08-18"),
        dict(series_id="irrelevant", player_id="elsewhere", played_day="2015-08-17", release_day="2015-08-18"),
        observation("current"),
    ]
    result = build(iter(records))
    assert set(result.sources) == {3}
    prior = result.diagnostics[0]["prior_activity"][0][0]
    assert prior["prior_source_count"] == 3
    assert prior["latest_played_source_index"] == 3
    old_only = build(records[:3])
    assert set(old_only.sources) == {1}
    assert old_only.sources[1]["source_index"] == 1
    assert "source_sha256" not in old_only.sources[1]
    assert old_only.arrays["player_activity"][0, 0, 0, 3] == 1
    assert not old_only.arrays["player_valid"].any()


def test_missing_player_id_does_not_share_unknown_identity_history():
    result = build([observation("missing", player=None), observation("blank", player="")])
    assert not result.arrays["player_activity"].any()
    assert not result.arrays["player_valid"].any()
    assert not result.arrays["player_stats"].any()
    assert result.arrays["meta_support"][0, 0, 0, 0] == 2


def test_whole_series_release_is_strictly_prior_for_every_map():
    records = [observation("same-series", age=90, release_day="2025-02-01"),
               observation("same-series", age=80, release_day="2025-02-01")]
    result = build(records, [target(), target("after", cutoff_day="2025-02-02")])
    assert not result.arrays["player_activity"][0].any()
    assert result.diagnostics[0]["long_player_source_indices"][0][0] == []
    assert result.diagnostics[1]["long_player_source_indices"][0][0] == [0, 1]
    np.testing.assert_allclose(result.arrays["player_activity"][1, 0, 0, 2], np.log1p(81))
