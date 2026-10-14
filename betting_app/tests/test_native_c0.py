from __future__ import annotations

from copy import deepcopy
from datetime import timezone

import numpy as np
import pytest

from betting_app.core.models import native_c0
from betting_app.core.models.native_c0 import (
    NativeC0Adapter,
    encode_snapshot,
)


DECISION = "2026-10-02T12:00:00+00:00"
ROSTER_A = ["a1", "a2", "a3", "a4", "a5"]
ROSTER_B = ["b1", "b2", "b3", "b4", "b5"]


def _adapter() -> NativeC0Adapter:
    return NativeC0Adapter.__new__(NativeC0Adapter)


def _request(best_of: int = 3, *, decision_at: str = DECISION, mode: str = "full") -> dict:
    return {
        "team1_id": "team-a",
        "team2_id": "team-b",
        "roster_a": list(reversed(ROSTER_A)),
        "roster_b": list(reversed(ROSTER_B)),
        "best_of": best_of,
        "decision_at": decision_at,
        "competition_context": "Worlds",
        "start_at": "2026-10-02T14:00:00Z",
        "mode": mode,
    }


def _snapshot() -> dict:
    raw = {
        "players": np.arange(2 * 5 * 21, dtype=np.float32).reshape(2, 5, 21),
        "core": np.arange(2 * 5, dtype=np.float32).reshape(2, 5),
        "team": np.arange(2 * 11, dtype=np.float32).reshape(2, 11),
        "w20": np.arange(2 * 10, dtype=np.float32).reshape(2, 10),
        "gates": np.arange(2 * 2, dtype=np.float32).reshape(2, 2),
    }
    hist = {
        "history": np.arange(2 * 5 * 16 * 33, dtype=np.float16).reshape(2, 5, 16, 33),
        "mask": np.arange(2 * 5 * 16, dtype=np.uint8).reshape(2, 5, 16) % 2,
        "champions": np.arange(2 * 5 * 16, dtype=np.uint16).reshape(2, 5, 16),
    }
    target = {
        "team1_id": "team-a",
        "team2_id": "team-b",
        "best_of": 3,
        "decision_at": DECISION,
        "effective_decision_at": DECISION,
        "start_at": "2026-10-02T14:00:00Z",
        "competition_context": "Worlds",
        "tournament_name": "Worlds",
        "roster_a": list(reversed(ROSTER_A)),
        "roster_b": list(reversed(ROSTER_B)),
        "cutoff_day": "2026-10-02",
        "feature_source_max_day": "2026-10-01",
        "feature_available": True,
    }
    for side, roster in (("a", ROSTER_A), ("b", ROSTER_B)):
        for key in (
            f"tensor_player_order_{side}",
            f"raw_tensor_player_order_{side}",
            f"temporal_tensor_player_order_{side}",
        ):
            target[key] = list(roster)
    return {"target": target, "raw": raw, "hist": hist}


@pytest.mark.parametrize("best_of", [1, 3, 5])
def test_native_request_accepts_supported_series_formats_and_manual_ids(best_of: int) -> None:
    request = _adapter()._validate_request(_request(best_of, mode="no_w20"))

    assert request["best_of"] == best_of
    assert request["team1_id"] == "team-a"
    assert request["roster_a"] == sorted(ROSTER_A)
    assert request["decision_at"].tzinfo == timezone.utc
    assert request["mode"] == "no_w20"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"decision_at": "2026-10-02T12:00:00"}, "timezone"),
        ({"decision_at": "2025-12-31T23:59:00Z"}, "pre-2026-01-01"),
        ({"best_of": 7}, "best_of"),
        ({"roster_a": ["a1", "a1", "a3", "a4", "a5"]}, "unique"),
    ],
)
def test_native_request_rejects_invalid_or_leaking_inputs(overrides: dict, message: str) -> None:
    request = _request()
    request.update(overrides)

    with pytest.raises(ValueError, match=message):
        _adapter()._validate_request(request)


def test_exact_native_snapshot_wire_round_trip_preserves_tensor_dtype_and_values() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    encoded = encode_snapshot(_snapshot())
    target, raw, hist = adapter._decode_snapshot(encoded, request)

    assert target["feature_source_max_day"] == "2026-10-01"
    for family, expected in (("raw", _snapshot()["raw"]), ("hist", _snapshot()["hist"])):
        actual = raw if family == "raw" else hist
        for key, array in expected.items():
            assert actual[key].dtype == array.dtype
            np.testing.assert_array_equal(actual[key], array)


def test_native_snapshot_rejects_history_on_decision_day() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    snapshot = _snapshot()
    snapshot["target"]["feature_source_max_day"] = "2026-10-02"

    with pytest.raises(ValueError, match="decision-day boundary"):
        adapter._decode_snapshot(encode_snapshot(snapshot), request)


def test_native_snapshot_requires_every_exact_sorted_player_slot_order() -> None:
    adapter = _adapter()
    request = adapter._validate_request(_request())
    snapshot = _snapshot()
    del snapshot["target"]["temporal_tensor_player_order_a"]

    with pytest.raises(ValueError, match="omits exact native player-slot order"):
        adapter._decode_snapshot(encode_snapshot(snapshot), request)


def test_native_snapshot_rejects_tensor_dtype_substitution() -> None:
    snapshot = _snapshot()
    snapshot["raw"]["players"] = snapshot["raw"]["players"].astype(np.float64)

    with pytest.raises(ValueError, match="requires shape"):
        encode_snapshot(snapshot)


def test_raw_history_defaults_to_canonical_source_and_allows_explicit_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_c0, "PROJECT_ROOT", tmp_path)
    canonical = tmp_path / native_c0._CANONICAL_RAW_HISTORY_RELATIVE
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical raw source")
    config = {"raw_history_path": native_c0._CANONICAL_RAW_HISTORY_RELATIVE}

    default_path, default_kind, default_is_canonical = native_c0._resolve_raw_history_source(config, None)
    assert default_path == canonical
    assert default_kind == native_c0._CANONICAL_RAW_HISTORY_KIND
    assert default_is_canonical is True

    override = tmp_path / "approved-alternative.json"
    override.write_bytes(b"explicit alternative")
    override_path, override_kind, override_is_canonical = native_c0._resolve_raw_history_source(
        config, str(override),
    )
    assert override_path == override
    assert override_kind == "explicit_raw_history_override"
    assert override_is_canonical is False


def test_missing_explicit_raw_history_fails_without_automatic_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(native_c0, "PROJECT_ROOT", tmp_path)
    canonical = tmp_path / native_c0._CANONICAL_RAW_HISTORY_RELATIVE
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical raw source")
    config = {"raw_history_path": native_c0._CANONICAL_RAW_HISTORY_RELATIVE}

    with pytest.raises(FileNotFoundError):
        native_c0._resolve_raw_history_source(config, str(tmp_path / "missing-override.json"))


@pytest.fixture(scope="module")
def genuine_native_client():
    from betting_app.services.c0_inference import C0InferenceClient

    client = C0InferenceClient()
    try:
        yield client
    finally:
        client.close()


def test_genuine_native_diagnostics_bind_exact_request_without_changing_probability(
    genuine_native_client,
    monkeypatch,
) -> None:
    monkeypatch.delenv("C0_RAW_HISTORY_PATH", raising=False)
    monkeypatch.delenv("C0_RELEASE_MAP_PATH", raising=False)
    request = _request()
    original = deepcopy(request)
    ordinary = genuine_native_client.predict_request(request)
    evidence = genuine_native_client.predict_evidence(request)
    diagnostics = ordinary.diagnostics

    assert request == original
    assert evidence.prediction.prob_a == pytest.approx(ordinary.prob_a, abs=1e-12)
    assert ordinary.prob_b == pytest.approx(1.0 - ordinary.prob_a, abs=1e-12)
    assert ordinary.map_prob_a is None
    assert ordinary.map_prob_b is None
    assert ordinary.epistemic_sigma_z is None
    assert ordinary.p_low_a is None
    assert ordinary.p_low_b is None
    assert diagnostics["probability_target"] == "series_win_already_projected"
    assert diagnostics["uncertainty_available"] is False
    assert diagnostics["information_status_basis"] == "declared_input_provenance_not_authenticated"
    provenance = diagnostics["feature_provenance"]
    assert provenance["source_kind"] == "canonical_frozen_golgg_history"
    assert provenance["raw_history_sha256"] == native_c0._CANONICAL_RAW_HISTORY_SHA256
    assert provenance["dataset_freshness"]["raw_history_max_source_game_day"] == "2026-09-08"
    assert provenance["dataset_freshness"]["raw_history_max_effective_release_day"] == "2026-09-08"
    assert provenance["dataset_freshness"]["feature_history_release_cutoff_day"] == "2026-09-08"
    assert provenance["dataset_freshness"]["query_decision_day_exclusive_cutoff"] == "2026-10-02"
    assert diagnostics["input_hashes"] == evidence.prediction.diagnostics["input_hashes"]
    assert diagnostics["c0_request"] == {
        **original,
        "roster_a": sorted(ROSTER_A),
        "roster_b": sorted(ROSTER_B),
        "start_at": "2026-10-02T14:00:00+00:00",
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"roster_a": ["a1", "a2", "a3", "a4", "replacement"]}, "roster_a differs"),
        ({"decision_at": "2026-10-02T12:01:00+00:00"}, "decision clocks differ"),
        ({"start_at": "2026-10-02T15:00:00+00:00"}, "start_at differs"),
        ({"competition_context": "MSI"}, "competition context differs"),
    ],
    ids=["roster", "decision_clock", "start_clock", "competition_context"],
)
def test_native_snapshot_withholds_changed_request_instead_of_relabeling_tensors(
    overrides: dict,
    message: str,
) -> None:
    adapter = _adapter()
    snapshot = encode_snapshot(_snapshot())
    original = deepcopy(snapshot)
    request = {**_request(), **overrides}

    with pytest.raises(ValueError, match=message):
        adapter._decode_snapshot(snapshot, adapter._validate_request(request))

    assert snapshot == original
