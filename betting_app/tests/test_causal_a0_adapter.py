"""Native tensor boundaries; real frozen-artifact parity is exercised by the local pilot."""
from copy import deepcopy
import json

import numpy as np
import pytest

from betting_app.core.models.causal_a0 import CausalA0Adapter, validate_snapshot
from src.analysis.ev_asof_replay import AsOfSnapshot, _RAW_SHAPES, _TEMPORAL_SHAPES


def cold_snapshot():
    target = {
        "origin_id": "local-cold", "best_of": 3, "cutoff_day": "2026-01-02",
        "feature_source_max_day": None, "historical_scenario": True,
        "information_status": "historical_scenario",
        "roster_a": ["a3", "a1", "a4", "a5", "a2"],
        "roster_b": ["b1", "b2", "b3", "b4", "b5"],
    }
    for side in ("a", "b"):
        order = sorted(target[f"roster_{side}"])
        target[f"tensor_player_order_{side}"] = order
        target[f"raw_tensor_player_order_{side}"] = order.copy()
        target[f"temporal_tensor_player_order_{side}"] = order.copy()
    return AsOfSnapshot(
        target,
        {key: np.zeros(shape, dtype=np.float32) for key, shape in _RAW_SHAPES.items()},
        {key: np.zeros(shape, dtype=np.int64 if key == "champions" else np.float32)
         for key, shape in _TEMPORAL_SHAPES.items()},
        True, None,
    )


def test_cold_known_roster_is_valid_without_organization_history():
    assert validate_snapshot(cold_snapshot(), year=2026) == 3


def test_serialized_context_roster_fields_preserve_sorted_tensor_slots():
    snapshot = cold_snapshot()
    for side in ("a", "b"):
        for field in (
            f"roster_{side}", f"tensor_player_order_{side}",
            f"raw_tensor_player_order_{side}", f"temporal_tensor_player_order_{side}",
        ):
            snapshot.target[field] = json.dumps(snapshot.target[field])
    assert validate_snapshot(snapshot, year=2026) == 3


def test_same_day_or_future_history_is_not_an_asof_snapshot():
    snapshot = cold_snapshot()
    snapshot.target["feature_source_max_day"] = "2026-01-02"
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_player_slot_mismatch_is_rejected_before_native_inference():
    snapshot = cold_snapshot()
    snapshot.target["temporal_tensor_player_order_a"].reverse()
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_future_annual_recipe_cannot_score_an_earlier_year():
    snapshot = cold_snapshot()
    snapshot.target["cutoff_day"] = "2025-12-31"
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)

def test_fixed_2026_contract_accepts_only_frozen_forward_cutoffs_without_mutation():
    from betting_app.core.models.causal_a0 import (
        FIXED_2026_CUTOFF_END,
        FIXED_2026_CUTOFF_START,
        validate_fixed_2026_snapshot,
    )

    snapshot = cold_snapshot()
    snapshot.target.update(
        cutoff_day=FIXED_2026_CUTOFF_START,
        model_year=2026,
        decision_at=None,
        effective_decision_at="2026-11-01T10:00:00+00:00",
        decision_time_source="scenario",
        scenario_decision_at="2026-11-01T10:00:00+00:00",
        feature_available_at="2026-11-01T10:00:00+00:00",
    )
    before = deepcopy(snapshot)
    assert validate_fixed_2026_snapshot(snapshot) == 3
    assert snapshot.target == before.target
    assert all(np.array_equal(snapshot.raw[key], before.raw[key]) for key in snapshot.raw)
    assert all(np.array_equal(snapshot.hist[key], before.hist[key]) for key in snapshot.hist)

    snapshot.target.update(
        cutoff_day="2029-06-30",
        model_year=2029,
        decision_at=None,
        effective_decision_at="2029-06-30T10:00:00+00:00",
        scenario_decision_at="2029-06-30T10:00:00+00:00",
        feature_available_at="2029-06-30T10:00:00+00:00",
    )
    assert validate_fixed_2026_snapshot(snapshot) == 3
    snapshot.target.update(
        cutoff_day=FIXED_2026_CUTOFF_END,
        decision_at=None,
        effective_decision_at="2029-07-01T10:00:00+00:00",
        scenario_decision_at="2029-07-01T10:00:00+00:00",
        feature_available_at="2029-07-01T10:00:00+00:00",
    )
    with pytest.raises(ValueError, match="outside the fixed 2026 serving contract"):
        validate_fixed_2026_snapshot(snapshot)


def test_fixed_future_contract_does_not_weaken_annual_wrong_year_rejection():
    from betting_app.core.models.causal_a0 import validate_fixed_2026_snapshot

    snapshot = cold_snapshot()
    snapshot.target.update(
        cutoff_day="2027-01-02",
        decision_at=None,
        effective_decision_at="2027-01-02T10:00:00+00:00",
        decision_time_source="scenario",
        scenario_decision_at="2027-01-02T10:00:00+00:00",
        feature_available_at="2027-01-02T10:00:00+00:00",
    )
    with pytest.raises(ValueError, match="cutoff year differs"):
        validate_snapshot(snapshot, year=2026)
    assert validate_fixed_2026_snapshot(snapshot) == 3


def test_fixed_future_contract_rejects_history_released_after_cutoff():
    from betting_app.core.models.causal_a0 import validate_fixed_2026_snapshot

    snapshot = cold_snapshot()
    snapshot.target.update(
        cutoff_day="2027-01-02",
        decision_at=None,
        effective_decision_at="2027-01-02T10:00:00+00:00",
        decision_time_source="scenario",
        scenario_decision_at="2027-01-02T10:00:00+00:00",
        feature_available_at="2027-01-02T10:00:00+00:00",
        feature_source_max_day="2027-01-02",
    )
    snapshot.raw["players"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="strictly before cutoff_day"):
        validate_fixed_2026_snapshot(snapshot)

@pytest.mark.parametrize("damage", ["shape", "nan", "mask", "champion", "overlap", "bo"])
def test_invalid_native_inputs_fail_closed(damage):
    snapshot = deepcopy(cold_snapshot())
    if damage == "shape":
        snapshot.raw["players"] = np.zeros((2, 4, 21))
    elif damage == "nan":
        snapshot.raw["players"][0, 0, 0] = np.nan
    elif damage == "mask":
        snapshot.hist["mask"][0, 0, 0] = 2
    elif damage == "champion":
        snapshot.hist["champions"] = np.full((2, 5, 16), -1, dtype=np.int64)
    elif damage == "overlap":
        snapshot.target["roster_b"][0] = "a1"
    else:
        snapshot.target["best_of"] = 2
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_rating_context_also_requires_a_release_day():
    snapshot = cold_snapshot()
    snapshot.raw["players"][0, 0, 0] = 1
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_decision_at_start_is_not_prematch():
    snapshot = cold_snapshot()
    snapshot.target.update(
        decision_at="2026-01-02T12:00:00+00:00",
        effective_decision_at="2026-01-02T12:00:00+00:00",
        start_at="2026-01-02T12:00:00+00:00",
        decision_time_source="actual",
    )
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def scenario_snapshot():
    snapshot = cold_snapshot()
    snapshot.target.update(
        decision_time_source="scenario",
        effective_decision_at="2026-01-02T10:00:00+00:00",
        scenario_decision_at="2026-01-02T10:00:00+00:00",
        scenario_start_at="2026-01-02T12:00:00+00:00",
        feature_available_at="2026-01-02T10:00:00+00:00",
    )
    return snapshot


def test_scenario_feature_availability_must_not_follow_effective_cutoff():
    snapshot = scenario_snapshot()
    assert validate_snapshot(snapshot, year=2026) == 3
    snapshot.target["feature_available_at"] = "2026-01-02T10:01:00+00:00"
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_scenario_decision_cannot_follow_declared_start():
    snapshot = scenario_snapshot()
    snapshot.target["scenario_start_at"] = "2026-01-02T09:00:00+00:00"
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_declared_effective_clock_binds_daily_cutoff_even_without_source_label():
    snapshot = cold_snapshot()
    snapshot.target["effective_decision_at"] = "2026-01-01T10:00:00+00:00"
    with pytest.raises(ValueError):
        validate_snapshot(snapshot, year=2026)


def test_changed_manifest_is_rejected_before_artifact_loading(tmp_path):
    manifest = tmp_path / "a0-phase-walkforward-20260914/data/06_models/a0_manifest_causal.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    adapter = CausalA0Adapter(2026, research_root=tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _ = adapter.identity
