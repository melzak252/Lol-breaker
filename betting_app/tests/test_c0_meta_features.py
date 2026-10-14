"""Consumer contracts for causal, role-aware thirty-played-day features."""
from __future__ import annotations

import numpy as np
import pytest

from src.features.c0_meta_features import ROLES, STATS, SUMMARY_WIDTH, build_meta_features


ARRAY_NAMES = (
    "player_stats", "player_roles", "player_champions", "player_valid",
    "meta_stats", "meta_champions", "meta_support", "meta_patch",
)


def observation(series="past", player="p", value=2.0, **changes):
    record = dict(series_id=series, played_day="2025-01-20", release_day="2025-01-21",
                  competition="LCK", patch="15.1", player_id=player, role="top",
                  champion_id="A", values12=[value] * 12, observed12=[True] * 12)
    record.update(changes)
    return record


def target(row="target", **changes):
    record = dict(row_id=row, cutoff_day="2025-02-01", competition="LCK",
                  roster_a=["p", None, None, None, None], roster_b=[None] * 5)
    record.update(changes)
    return record


def build(records, targets=None, champion_ids=("A", "B")):
    return build_meta_features(records, [target()] if targets is None else targets,
                               champion_ids=champion_ids)


def test_contract_shapes_finite_dtypes_and_retained_original_row_order():
    result = build([observation()], [target("late"), target("early", cutoff_day="2025-01-01")])
    assert ROLES == ("top", "jungle", "mid", "adc", "support")
    assert STATS == ("kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm",
                     "gd@15", "csd@15", "xpd@15", "kp%", "dmg%")
    assert SUMMARY_WIDTH == 40
    assert result.row_ids == ("late", "early")
    assert set(result.arrays) == set(ARRAY_NAMES)
    widths = dict(player_stats=40, player_roles=5, player_champions=3,
                  meta_stats=40, meta_champions=3, meta_support=2, meta_patch=2)
    for name in ARRAY_NAMES:
        array = result.arrays[name]
        assert array.shape == ((2, 2, 5) if name == "player_valid" else (2, 2, 5, widths[name]))
        assert array.dtype == (np.bool_ if name == "player_valid" else np.float32)
        assert np.isfinite(array).all()
    assert not result.arrays["player_valid"][1].any()
    assert len(result.diagnostics) == 2
    assert result.diagnostics[1]["source_indices"] == []


def test_future_perturbations_target_draft_and_actual_role_never_change_inputs():
    prior = observation()
    future = observation("future", value=999, played_day="2025-02-01", release_day="2025-02-02",
                         role="support", champion_id="future-champion", patch="99.99")
    hidden = observation("unreleased", value=-999, release_day="2025-02-01")
    original = build([prior])
    changed = build([prior, future, hidden], [target(patch="99.99", actual_roles=["support"] * 10,
                                                      draft=["future-champion"] * 10)])
    for name in ARRAY_NAMES:
        np.testing.assert_array_equal(original.arrays[name], changed.arrays[name])
    assert changed.arrays["player_champions"].shape[-1] == 3
    assert changed.diagnostics[0]["target_patch"] == "99.99"
    assert changed.diagnostics[0]["source_indices"] == [0]


def test_played_window_and_independent_release_boundaries_with_late_arrival():
    records = [
        observation("lower", played_day="2025-01-02", release_day="2025-01-03"),
        observation("too-old", played_day="2025-01-01", release_day="2025-01-31"),
        observation("release-equal", played_day="2025-01-25", release_day="2025-02-01"),
        observation("played-equal", played_day="2025-02-01", release_day="2025-01-31"),
        observation("late", played_day="2025-01-05", release_day="2025-01-31"),
    ]
    result = build(records, [target(), target("next", cutoff_day="2025-02-02")])
    assert result.diagnostics[0]["source_indices"] == [0, 4]
    assert result.diagnostics[1]["source_indices"] == [2, 3, 4]
    first = result.diagnostics[0]
    assert first["max_played_day"] == "2025-01-05"
    assert first["max_release_day"] == "2025-01-31"
    assert first["max_effective_release_day"] == "2025-01-31"
    assert first["window_start_day"] == "2025-01-02"
    assert {result.sources[index]["series_id"] for index in first["source_indices"]} == {"lower", "late"}
    assert np.isclose(result.arrays["player_stats"][0, 0, 0, 36], np.log1p(2))


def test_absent_champion_is_not_unknown_observed_champion():
    records = [observation("known"), observation("outside", champion_id="Z"),
               observation("explicit-unknown", champion_id=0), observation("absent", champion_id=None),
               observation("blank", champion_id="")]
    result = build(records)
    np.testing.assert_allclose(result.arrays["player_champions"][0, 0, 0], [2 / 3, 1 / 3, 0])
    np.testing.assert_allclose(result.arrays["meta_champions"][0, 0, 0], [2 / 3, 1 / 3, 0])
    assert np.isclose(result.arrays["player_stats"][0, 0, 0, 37], 3 / 5)
    assert result.arrays["player_valid"][0, 0, 0]


def test_masked_summary_preserves_missingness_and_unknown_role_mass():
    missing = [False] * 12
    missing[0] = True
    records = [observation("known", value=2),
               observation("unknown-role", role=None, values12=[6] + [np.nan] * 11, observed12=missing),
               observation("no-stats", role="not-a-source-role", values12=[np.nan] * 12,
                           observed12=[False] * 12, patch=None, champion_id=None)]
    result = build(records)
    summary = result.arrays["player_stats"][0, 0, 0]
    np.testing.assert_allclose(summary[:12], [4] + [2] * 11)
    np.testing.assert_allclose(summary[12:24], [2] + [0] * 11)
    np.testing.assert_allclose(summary[24:36], [2 / 3] + [1 / 3] * 11)
    np.testing.assert_allclose(summary[37:], [2 / 3, 1 / 3, 2 / 3])
    np.testing.assert_allclose(result.arrays["player_roles"][0, 0, 0], [1 / 3, 0, 0, 0, 0])
    assert result.arrays["meta_support"][0, 0, 0, 0] == 1
    assert result.diagnostics[0]["unknown_role_source_indices"] == [1, 2]


def test_local_pooling_uses_first_and_second_moments_and_fixed_hundred_prior():
    records = [observation("local", value=10, champion_id="B", patch="15.2"),
               observation("other", player="q", value=0, competition="LEC", patch="15.1")]
    result = build(records)
    global_summary = result.arrays["meta_stats"][0, 0, 0]
    local_summary = result.arrays["meta_stats"][0, 1, 0]
    np.testing.assert_allclose(global_summary[:12], 5)
    np.testing.assert_allclose(global_summary[12:24], 5)
    weight = 1 / 101
    mean = weight * 10 + (1 - weight) * 5
    second_moment = weight * 100 + (1 - weight) * 50
    np.testing.assert_allclose(local_summary[:12], mean)
    np.testing.assert_allclose(local_summary[12:24], np.sqrt(second_moment - mean**2), rtol=1e-6)
    np.testing.assert_allclose(result.arrays["meta_support"][0, :, 0], [[2, 1], [1, weight]])
    np.testing.assert_allclose(result.arrays["meta_champions"][0, 1, 0],
                               [0, (1 - weight) / 2, weight + (1 - weight) / 2])
    assert np.isclose(local_summary[36], np.log1p(1))
    np.testing.assert_array_equal(result.arrays["meta_patch"][0, :, 0], [[2, .5], [1, 1]])
    assert result.diagnostics[0]["global_role_source_indices"][0] == [0, 1]
    assert result.diagnostics[0]["local_role_source_indices"][0] == [0]


def test_pooling_missing_stats_does_not_treat_absence_as_observed_zero():
    records = [observation("local", values12=[np.nan] * 12, observed12=[False] * 12,
                           champion_id=None),
               observation("global", value=7, competition="LEC")]
    result = build(records)
    np.testing.assert_allclose(result.arrays["meta_stats"][0, 1, 0, :12], 7)
    np.testing.assert_allclose(result.arrays["meta_stats"][0, 1, 0, 12:24], 0)
    np.testing.assert_allclose(result.arrays["meta_stats"][0, 1, 0, 24:36], (100 / 101) * .5)
    np.testing.assert_allclose(result.arrays["meta_champions"][0, 1, 0], [0, 1, 0])
    assert result.arrays["meta_support"][0, 1, 0, 0] == 1


def test_roster_player_identity_order_substitutes_and_side_swap():
    records = [observation("p-map", player="p", role="mid", value=1),
               observation("q-map", player="q", role="jungle", value=8)]
    ab = target(roster_a=["q", "p", None, "substitute", None], roster_b=["p", None, None, None, None])
    ba = target(roster_a=ab["roster_b"], roster_b=ab["roster_a"])
    result = build(records, [ab, ba])
    for name in ("player_stats", "player_roles", "player_champions", "player_valid"):
        np.testing.assert_array_equal(result.arrays[name][0, [1, 0]], result.arrays[name][1])
    assert result.arrays["player_stats"][0, 0, 0, 0] == 8
    assert result.arrays["player_roles"][0, 0, 0, 1] == 1
    assert result.arrays["player_roles"][0, 0, 1, 2] == 1
    assert not result.arrays["player_valid"][0, 0, 3]
    assert result.diagnostics[0]["player_source_indices"][0][:2] == [[1], [0]]
    for name in ("meta_stats", "meta_champions", "meta_support", "meta_patch"):
        np.testing.assert_array_equal(result.arrays[name][0], result.arrays[name][1])


def test_sparse_empty_and_no_local_context_have_explicit_support():
    empty = build([])
    for array in empty.arrays.values():
        assert not array.any()
    assert empty.diagnostics[0]["max_effective_release_day"] is None
    assert empty.diagnostics[0]["patch_coverage"]["known_observations"] == 0
    sparse = build([observation(player="elsewhere", role="adc", competition="LEC")])
    assert not sparse.arrays["player_valid"].any()
    np.testing.assert_array_equal(sparse.arrays["meta_stats"][0, 0, 3, :36],
                                  sparse.arrays["meta_stats"][0, 1, 3, :36])
    assert sparse.arrays["meta_support"][0, 1, 3, 0] == 0
    assert sparse.arrays["meta_support"][0, 1, 3, 1] == 0
    assert sparse.arrays["meta_stats"][0, 1, 3, 36] == 0
    assert sparse.arrays["meta_patch"][0, 1, 3].sum() == 0
    no_rows = build([observation()], [])
    assert no_rows.row_ids == ()
    assert all(array.shape[0] == 0 for array in no_rows.arrays.values())


def test_real_patch_labels_are_only_explicit_source_strings_never_calendar_inference():
    records = [observation("unknown-none", patch=None), observation("unknown-text", patch="unknown"),
               observation("unknown-blank", patch=""), observation("unknown-number", patch=15.1),
               observation("label-a", patch="15.1"), observation("label-b", patch="15.2")]
    result = build(records, [target(patch="99.1")])
    assert np.isclose(result.arrays["player_stats"][0, 0, 0, 39], 2 / 6)
    np.testing.assert_allclose(result.arrays["meta_patch"][0, 0, 0], [2, .5])
    coverage = result.diagnostics[0]["patch_coverage"]
    assert coverage == dict(observations=6, known_observations=2, unknown_observations=4,
                            known_fraction=2 / 6, labels={"15.1": 1, "15.2": 1})
    assert [result.sources[index]["patch"] for index in result.diagnostics[0]["source_indices"]] == [None] * 4 + ["15.1", "15.2"]
    assert result.diagnostics[0]["patch_transition"] == {
        "latest_played_day": "2025-01-20", "latest_labels": ["15.1", "15.2"],
        "target_patch_seen": False, "target_differs_from_latest": True,
    }


def test_batch_is_deterministic_for_unsorted_targets_and_observation_order():
    records = [observation("b", value=6), observation("a", value=2, release_day="2025-01-30")]
    targets = [target("later", cutoff_day="2025-02-03"), target("earlier")]
    result = build(records, targets)
    reversed_records = build(list(reversed(records)), targets)
    for name in ARRAY_NAMES:
        np.testing.assert_array_equal(result.arrays[name], reversed_records.arrays[name])
    for index, row in enumerate(targets):
        alone = build(records, [row])
        for name in ARRAY_NAMES:
            np.testing.assert_array_equal(result.arrays[name][index], alone.arrays[name][0])


def test_missing_player_ids_never_share_a_synthetic_unknown_player_history():
    result = build([observation(player=None), observation(player="", role=None)])
    assert not result.arrays["player_valid"].any()
    assert not result.arrays["player_stats"].any()
    assert result.arrays["meta_support"][0, 0, 0, 0] == 1


@pytest.mark.parametrize("champion_ids", [("A", "A"), (0, "A"), (None, "A")])
def test_fixed_champion_vocabulary_rejects_ambiguous_bins(champion_ids):
    with pytest.raises(ValueError):
        build([], champion_ids=champion_ids)


def test_observed_nonfinite_value_and_wrong_stat_width_are_rejected():
    with pytest.raises(ValueError):
        build([observation(values12=[np.nan] * 12)])
    with pytest.raises(ValueError):
        build([observation(values12=[1], observed12=[True])])


def test_utc_daily_timestamps_and_uppercase_native_roles_are_supported():
    import pandas as pd

    record = observation(role="MID", played_day=pd.Timestamp("2025-01-20"),
                         release_day=pd.Timestamp("2025-01-21T23:00:00Z"))
    result = build([record], [target(cutoff_day=pd.Timestamp("2025-02-01T02:00:00+02:00"))])
    assert result.arrays["player_roles"][0, 0, 0, 2] == 1
    assert result.arrays["meta_support"][0, 0, 2, 0] == 1
    assert result.diagnostics[0]["cutoff_day"] == "2025-02-01"
