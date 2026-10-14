"""Historical token joins must follow identities, not row or role order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.prematch_history_bank import align_history


def banks():
    a = [f"a{i}" for i in range(5)]
    b = [f"b{i}" for i in range(5)]
    fresh = pd.DataFrame([{"match_id": "7", "date": "2025-01-04", "release_day": "2025-01-05", "best_of": 3, "y": 1, "team_a": "A", "team_b": "B", "roster_a": a, "roster_b": b, "max_history_release_day": "2025-01-03"}])
    old = pd.DataFrame([{"golgg_match_id": "7", "date": "2025-01-04", "result_day": "2025-01-05", "best_of": 3, "y": 0, "team1_id": "B", "team2_id": "A", "roster_a": b[::-1], "roster_b": a[::-1], "feature_source_max_day": "2025-01-03"}])
    history = np.arange(10, dtype=np.float32).reshape(1, 2, 5, 1, 1)
    mask = np.ones((1, 2, 5, 1), dtype=bool)
    return fresh, old, history, mask


def test_history_follows_player_identity_across_side_and_role_permutations():
    fresh, old, history, mask = banks()
    mask[0, 0, [1, 4], 0] = False
    aligned, visible, _ = align_history(fresh, old, history, mask)
    np.testing.assert_array_equal(aligned[..., 0, 0], [[[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]]])
    np.testing.assert_array_equal(visible[..., 0], [[[True] * 5, [False, True, True, False, True]]])


@pytest.mark.parametrize("mutation", ["roster", "release", "same_day_history", "label"])
def test_history_join_rejects_incompatible_or_unavailable_records(mutation):
    fresh, old, history, mask = banks()
    if mutation == "roster":
        old.at[0, "roster_a"] = ["unknown", "b3", "b2", "b1", "b0"]
    elif mutation == "release":
        old.at[0, "result_day"] = "2025-01-06"
    elif mutation == "same_day_history":
        old.at[0, "feature_source_max_day"] = "2025-01-04"
    else:
        old.at[0, "y"] = 1
    with pytest.raises(ValueError):
        align_history(fresh, old, history, mask)


def test_row_permutations_preserve_the_match_join():
    fresh, old, history, mask = banks()
    fresh = pd.concat([fresh.assign(match_id="8"), fresh.assign(match_id="7")], ignore_index=True)
    old = pd.concat([old.assign(golgg_match_id="7"), old.assign(golgg_match_id="8")], ignore_index=True)
    history = np.concatenate([history + 100, history], axis=0)
    mask = np.concatenate([mask, mask], axis=0)


    aligned, _, _ = align_history(fresh, old, history, mask)

    np.testing.assert_array_equal(aligned[0, ..., 0, 0], [[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]])
    np.testing.assert_array_equal(aligned[1, ..., 0, 0], [[109, 108, 107, 106, 105], [104, 103, 102, 101, 100]])


@pytest.mark.parametrize("duplicate_side", ["fresh", "a0"])
def test_duplicate_match_ids_are_rejected(duplicate_side):
    fresh, old, history, mask = banks()
    fresh = pd.concat([fresh, fresh], ignore_index=True)
    old = pd.concat([old, old], ignore_index=True)
    fresh.loc[1, "match_id"] = "8"
    old.loc[1, "golgg_match_id"] = "8"
    history = np.concatenate([history, history])
    mask = np.concatenate([mask, mask])
    if duplicate_side == "fresh":
        fresh.loc[1, "match_id"] = "7"
    else:
        old.loc[1, "golgg_match_id"] = "7"
    with pytest.raises(ValueError):
        align_history(fresh, old, history, mask)


def test_missing_release_bounds_are_legal_only_for_empty_histories():
    fresh, old, history, mask = banks()
    fresh.at[0, "max_history_release_day"] = None
    old.at[0, "feature_source_max_day"] = None
    _, visible, _ = align_history(fresh, old, history, np.zeros_like(mask))
    assert not visible.any()
    with pytest.raises(ValueError):
        align_history(fresh, old, history, mask)


def test_finiteness_validation_covers_the_final_partial_chunk():
    fresh, old, history, mask = banks()
    ids = [str(i) for i in range(129)]
    fresh = pd.concat([fresh] * len(ids), ignore_index=True).assign(match_id=ids)
    old = pd.concat([old] * len(ids), ignore_index=True).assign(golgg_match_id=ids)
    history = np.repeat(history, len(ids), axis=0)
    mask = np.repeat(mask, len(ids), axis=0)
    history[-1, 0, 0, 0, 0] = np.nan
    mask[-1, 0, 0, 0] = False
    _, visible, _ = align_history(fresh, old, history, mask)
    assert visible.sum() == mask.size - 1
    mask[-1, 0, 0, 0] = True
    with pytest.raises(ValueError):
        align_history(fresh, old, history, mask)


def test_empty_player_histories_allow_prior_global_state_watermarks():
    fresh, old, history, mask = banks()
    mask[:] = False
    _, visible, _ = align_history(fresh, old, history, mask)
    assert not visible.any()
    old.at[0, "feature_source_max_day"] = None
    _, visible, _ = align_history(fresh, old, history, mask)
    assert not visible.any()
    old.at[0, "feature_source_max_day"] = "2025-01-04"
    with pytest.raises(ValueError):
        align_history(fresh, old, history, mask)
