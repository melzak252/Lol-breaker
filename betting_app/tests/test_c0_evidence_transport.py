"""Evidence transport regressions; no external source access."""
from datetime import datetime, timezone

import pytest

from betting_app.core.models.native_c0 import NativeC0Adapter
from betting_app.services.c0_worker import _parse_request


def test_worker_accepts_bounded_evidence_operation():
    assert _parse_request(b'{"id":1,"op":"predict_evidence"}\n')["op"] == "predict_evidence"
    with pytest.raises(ValueError, match="exceeds"):
        _parse_request(b"x" * (16 * 1024 * 1024 + 1))


@pytest.fixture(scope="module")
def evidence_client():
    from betting_app.services.c0_inference import C0InferenceClient
    client = C0InferenceClient()
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("best_of", [1, 3, 5])
@pytest.mark.parametrize("mode", ["full", "no_w20", "no_organization"])
def test_genuine_child_evidence_replays_exact_arrays_and_ordinary_probability(best_of, mode, evidence_client):
    from betting_app.tests.test_native_c0 import _request
    request = _request(best_of, mode=mode)
    client = evidence_client
    ordinary = client.predict_request(request)
    evidence = client.predict_evidence(request)
    assert evidence.prediction.prob_a == pytest.approx(ordinary.prob_a, abs=1e-12)
    assert evidence.generated_at <= evidence.parent_received_at
    assert evidence.generated_at > datetime.fromisoformat(request["decision_at"])
    validator = object.__new__(NativeC0Adapter)
    _, raw, hist = validator._decode_snapshot(
        evidence.encoded_snapshot, validator._validate_request(request),
    )
    replay = client.predict_evidence(
        request, snapshot={"target": evidence.encoded_snapshot["target"], "raw": raw, "hist": hist},
    )
    assert replay.prediction.prob_a == pytest.approx(ordinary.prob_a, abs=1e-12)
    assert replay.native_history_provenance["raw_history_sha256"] is None
    assert replay.native_history_provenance["release_map_sha256"] is None
    assert replay.native_history_provenance["raw_feature_replay_source"] is None
    for side in ("a", "b"):
        for prefix in ("tensor", "raw_tensor", "temporal_tensor"):
            assert evidence.encoded_snapshot["target"][f"{prefix}_player_order_{side}"] == sorted(request[f"roster_{side}"])


def test_real_evidence_worker_rotation_preserves_prediction_and_closes_old_child():
    from betting_app.services.c0_inference import C0InferenceClient
    from betting_app.tests.test_native_c0 import _request
    client = C0InferenceClient()
    client.max_worker_requests = 1
    try:
        first = client.predict_evidence(_request())
        old_child = client._process
        second = client.predict_evidence(_request())
        assert old_child is not None and old_child.poll() is not None
        assert client._process is not old_child
        assert second.prediction.prob_a == pytest.approx(first.prediction.prob_a, abs=1e-12)
        assert first.parent_received_at <= second.generated_at <= second.parent_received_at
    finally:
        client.close()
