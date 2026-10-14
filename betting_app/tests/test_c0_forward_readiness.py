"""Deterministic local C0 evidence contracts; no collection or native fitting."""
import base64
from datetime import datetime, timezone
import hashlib
import json

import pytest

from betting_app.services.c0_shadow_readiness import (
    canonical_bytes, persist_receipt, evaluate_c0_readiness,
    resolve_c0_roster, validate_dependency_receipt, validate_source_adjudication,
    snapshot_tensor_records,
)

UTC = timezone.utc

def test_receipt_content_addressed_idempotent_and_no_replacement(tmp_path):
    receipt = {"schema": "native_c0_dependency_v1", "evidence_class": "unverified"}
    first = persist_receipt(receipt, tmp_path)
    assert persist_receipt(receipt, tmp_path) == first
    path = tmp_path / (first + ".json")
    assert path.read_bytes() == canonical_bytes(receipt)
    assert path.stat().st_mode & 0o777 == 0o600
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="different"):
        persist_receipt(receipt, tmp_path)


def test_deadline_expired_never_rolls_and_flags_do_not_authorize():
    protocol = {"schema": "c0_market_study_v1", "protocol_id": "native-c0-market-study-v1"}
    before = evaluate_c0_readiness(protocol, now=datetime(2026, 10, 6, tzinfo=UTC),
                                   authorization={"authenticated": True, "g0": "PASS"})
    assert before["status"] == "BLOCKED_EXTERNAL_PREREQUISITES"
    after = evaluate_c0_readiness(protocol, now=datetime(2026, 10, 31, tzinfo=UTC))
    assert after["status"] == "EXPIRED_UNSTARTED"
    assert after["activation_deadline"] == "2026-10-31T00:00:00+00:00"
    assert after["real_fit_count"] == 0


def test_dependency_declared_hashes_are_not_builder_execution_proof():
    receipt = {"schema": "native_c0_dependency_v1", "snapshot_sha256": "a" * 64,
               "tensor_sha256": {}, "builder": {"execution_receipt_refs": []},
               "coverage_through_exclusive": "2026-11-02T00:00:00+00:00"}
    reasons = validate_dependency_receipt(receipt, protocol={}, encoded_snapshot={}, sources=[])
    assert "native_history_coverage_unverified" in reasons


def test_self_authenticated_source_cannot_pass_adjudication():
    receipt = {"schema": "native_c0_source_adjudication_v1", "decision": "authenticated",
               "approved_locator": "https://example.com/fixtures", "source_id": "official"}
    reasons = validate_source_adjudication(receipt, protocol={}, captures=[])
    assert reasons


def test_prior_known_exact_context_and_withdrawal_blocks_fallback():
    cutoff = datetime(2026, 10, 6, 12, tzinfo=UTC)
    prior = {"team_id": "1", "match_id": "old", "competition_context": "LCK",
             "competitive_unit_id": "first", "roster": [1, 2, 3, 4, 5],
             "published_at": "2026-10-05T12:00:00+00:00", "status": "released"}
    result = resolve_c0_roster(team_id="1", match_id="new", cutoff_at=cutoff,
                              competition_context="LCK", competitive_unit_id="first",
                              announcements=[prior])
    assert result["status"] == "prior_known"
    assert result["announcement"]["roster"] == [1, 2, 3, 4, 5]
    withdrawal = {**prior, "match_id": "new", "status": "withdrawn",
                  "published_at": "2026-10-06T11:59:00+00:00"}
    blocked = resolve_c0_roster(team_id="1", match_id="new", cutoff_at=cutoff,
                               competition_context="LCK", competitive_unit_id="first",
                               announcements=[prior, withdrawal])
    assert blocked["announcement"] is None
    assert blocked["missing_reason"] == "latest_announcement_retracted"
    wrong = resolve_c0_roster(team_id="1", match_id="new", cutoff_at=cutoff,
                             competition_context="LEC", competitive_unit_id="first",
                             announcements=[prior])
    assert wrong["announcement"] is None


def test_tensor_bytes_dtype_order_and_hash_preserved():
    shapes = {"raw": {"players": ((2,5,21), "float32",4), "core": ((2,5),"float32",4),
                       "team": ((2,11),"float32",4), "w20": ((2,10),"float32",4),
                       "gates": ((2,2),"float32",4)},
              "hist": {"history": ((2,5,16,33),"float16",2), "mask": ((2,5,16),"uint8",1),
                       "champions": ((2,5,16),"uint16",2)}}
    snapshot = {"target": {"roster_a": [1,2,3,4,5], "roster_b": [6,7,8,9,10]}, "raw": {}, "hist": {}}
    for side in ("a", "b"):
        for prefix in ("tensor", "raw_tensor", "temporal_tensor"):
            snapshot["target"][prefix + "_player_order_" + side] = snapshot["target"]["roster_" + side]
    for family, entries in shapes.items():
        for name, (shape, dtype, size) in entries.items():
            count = size
            for dimension in shape:
                count *= dimension
            snapshot[family][name] = {"dtype": dtype, "shape": list(shape),
                                      "data_base64": base64.b64encode(bytes(count)).decode()}
    records, hashes = snapshot_tensor_records(snapshot)
    assert len(hashes) == 8
    assert records["hist"]["champions"]["dtype"] == "uint16"
    assert "data_base64" not in records["raw"]["players"]
    snapshot["target"]["raw_tensor_player_order_a"] = [5,4,3,2,1]
    with pytest.raises(ValueError, match="order"):
        snapshot_tensor_records(snapshot)


@pytest.fixture
def evidence_session():
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    url = os.environ.get("TEST_DATABASE_URL")
    if not url or "_test" not in url:
        pytest.skip("requires explicitly isolated TEST_DATABASE_URL")
    engine = create_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            yield session
        transaction.rollback()
    engine.dispose()


def test_genuine_child_recording_and_exact_replay_are_unadmitted(evidence_session, tmp_path, monkeypatch):
    from uuid import uuid4
    from betting_app.services.c0_inference import C0InferenceClient
    from betting_app.services import c0_shadow_readiness as service
    from betting_app.services.forecast_evidence import append_forecast_evidence, load_forecast_evidence

    # Redirect only the content-addressed receipt destination, not native artifacts.
    original_persist = service.persist_receipt
    monkeypatch.setattr(service, "persist_receipt", lambda receipt, _: original_persist(receipt, tmp_path / "receipts"))
    request = {
        "team1_id": "fixture-a", "team2_id": "fixture-b",
        "roster_a": ["a1", "a2", "a3", "a4", "a5"],
        "roster_b": ["b1", "b2", "b3", "b4", "b5"],
        "decision_at": "2026-09-08T12:00:00+00:00",
        "start_at": "2026-09-08T14:00:00+00:00",
        "competition_context": "LCK", "best_of": 3, "mode": "full",
    }
    schedule = {
        "protocol_id": service.PROTOCOL_ID, "provider_id": "fixture",
        "provider_fixture_id": "fixture-1", "canonical_fixture_id": "fixture-1",
        "origin_at": request["decision_at"], "run_id": "local-" + uuid4().hex,
        "scheduled_start_at": request["start_at"], "known_start_bound": request["start_at"],
    }
    key = "c0-" + uuid4().hex
    client = C0InferenceClient()
    try:
        row = service.record_c0_snapshot(
            evidence_session, client, request, evidence_key=key,
            sources=[], input_source_ids=[], schedule=schedule, quotes=[],
            information_status="historical_scenario",
        )
        payload = load_forecast_evidence(evidence_session, key)
        encoded = service._encoded_from_envelope(payload)
        _, hashes = service.snapshot_tensor_records(encoded)
        assert len(hashes) == 8
        assert payload["input"]["dependency_receipt"]["tensor_sha256"] == hashes
        assert payload["input"]["admission_reasons"] == ["native_history_coverage_unverified"]
        assert payload["information_status"] == "historical_scenario"
        assert datetime.fromisoformat(payload["forecast"]["generated_at"]) > datetime.fromisoformat(request["decision_at"])
        assert datetime.fromisoformat(payload["forecast"]["generated_at"]) <= datetime.fromisoformat(payload["forecast"]["parent_received_at"]) <= row.recorded_at
        from betting_app.services.forecast_evidence import _forecast_link_reasons
        candidate = {
            "protocol_id": service.PROTOCOL_ID, "provider_id": "fixture", "fixture_id": "fixture-1",
            "fixture": {"canonical_fixture_id": "fixture-1", "team_ids": ["fixture-a", "fixture-b"],
                        "best_of": 3, "competition_context": "LCK"},
            "origin_at": datetime.fromisoformat(request["decision_at"]),
            "scheduled_start_at": datetime.fromisoformat(request["start_at"]),
            "known_start_bound": datetime.fromisoformat(request["start_at"]),
            "rosters": {"fixture-a": {"roster": request["roster_a"]},
                        "fixture-b": {"roster": request["roster_b"]}},
            "source_captures": [],
        }
        assert "golgg_match_id" not in payload["input"]["target"]
        link_reasons = _forecast_link_reasons(payload, candidate, recorded_at=row.recorded_at)
        assert "forecast_envelope_fixture_mismatch" not in link_reasons
        assert "forecast_envelope_time_invalid" in link_reasons
        repeated = append_forecast_evidence(evidence_session, evidence_key=key, payload=payload)
        assert repeated.id == row.id
        from betting_app.core.models.native_c0 import NativeC0Adapter
        decoder = object.__new__(NativeC0Adapter)
        target, raw, hist = decoder._decode_snapshot(encoded, decoder._validate_request(request))
        replay = client.predict_evidence(request, snapshot={"target": target, "raw": raw, "hist": hist})
        assert replay.prediction.prob_a == pytest.approx(payload["forecast"]["prob_a"], abs=1e-12)
        assert replay.native_history_provenance.get("raw_history_sha256") is None
        assert replay.native_history_provenance.get("release_map_sha256") is None
        assert replay.native_history_provenance.get("authenticated_coverage_through_exclusive") is None
        rollback_key = "c0-rollback-" + uuid4().hex
        with pytest.raises(RuntimeError), evidence_session.begin_nested():
            append_forecast_evidence(evidence_session, evidence_key=rollback_key, payload=payload)
            raise RuntimeError("cancel decision")
        assert load_forecast_evidence(evidence_session, rollback_key) is None
    finally:
        client.close()


def test_phase_export_keeps_no_forecast_primary_and_seals_before_labels(evidence_session, tmp_path):
    from uuid import uuid4
    from betting_app.services.c0_shadow_readiness import export_c0_study_bundle, PROTOCOL_ID
    from betting_app.services.forecast_evidence import append_opportunity_event
    fixture_id = "missing-" + uuid4().hex
    payload = {
        "status": "rejected", "rejection_reasons": ["quote_unavailable", "roster_missing"],
        "clocks": {}, "model_availability": {"available": False},
        "feature_availability": {"available": False}, "inference_eligible": False,
        "forward_admitted": False, "information_status": "local_fixture",
        "source_authentication_status": "local_fixture", "source_capture_ids": [],
        "canonical_fixture_id": fixture_id,
    }
    event = append_opportunity_event(
        evidence_session, event_key="event-" + uuid4().hex, protocol_id=PROTOCOL_ID,
        provider_id="fixture", fixture_id=fixture_id,
        origin_at=datetime(2026, 11, 1, 12, tzinfo=UTC),
        event_type="primary_decision", primary_decision=True, parent_event_key=None,
        payload=payload,
    )
    protocol = {"schema": "c0_market_study_v1", "protocol_id": PROTOCOL_ID}
    output = tmp_path / "TRAIN"
    manifest = export_c0_study_bundle(
        evidence_session, protocol=protocol, phase="TRAIN", authorization=None,
        body_store=tmp_path / "bodies", output_dir=output,
    )
    rows = [json.loads(line) for line in (output / "opportunities.jsonl").read_text().splitlines()]
    selected = next(row for row in rows if row["event_key"] == event.event_key)
    assert selected["forecast_evidence_key"] is None
    assert selected["payload"]["rejection_reasons"] == payload["rejection_reasons"]
    assert selected["canonical_match_id"] == fixture_id
    assert (output / "labels.jsonl").read_bytes() == b""
    membership = (output / "membership.json").read_bytes()
    assert hashlib.sha256(membership).hexdigest() == manifest["membership_sha256"]
    for filename, receipt in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == receipt["sha256"]
    with pytest.raises(FileExistsError):
        export_c0_study_bundle(
            evidence_session, protocol=protocol, phase="TRAIN", authorization=None,
            body_store=tmp_path / "bodies", output_dir=output,
        )


@pytest.mark.parametrize("withdrawn", [False, True])
def test_actual_captured_prior_route_precedes_callback_and_withdrawal_blocks(evidence_session, tmp_path, withdrawn):
    from datetime import timedelta
    from uuid import uuid4
    from betting_app.tests.test_local_opportunity_orchestration import (
        _boundary_inputs, _capture_json, _roster_payload,
    )
    from betting_app.services.forecast_evidence import load_source_capture, orchestrate_local_opportunity_batch
    from betting_app.services.c0_shadow_readiness import PROTOCOL_ID
    now = datetime.now(UTC)
    origin = now - timedelta(seconds=5)
    fixture_id, old_discovery = _boundary_inputs(evidence_session, tmp_path, origin)
    original = load_source_capture(evidence_session, old_discovery, body_store=tmp_path)
    fixture = json.loads(original["body_bytes"])["fixtures"][0]
    fixture["competition_context"] = "LCK"
    fixture["competitive_unit_ids"] = {"team-a": "first-a", "team-b": "first-b"}
    roster_seen = origin - timedelta(minutes=6)
    roster = _roster_payload("prior-other-fixture", roster_seen)
    for item in roster["announcements"]:
        item["competition_context"] = "LCK"
        item["competitive_unit_id"] = fixture["competitive_unit_ids"][item["team_id"]]
        item["status"] = "released"
    if withdrawn:
        roster["announcements"].append({
            **roster["announcements"][0], "match_id": fixture_id,
            "published_at": (origin - timedelta(seconds=10)).isoformat(),
            "status": "withdrawn", "roster": [],
        })
        roster_seen = origin - timedelta(seconds=9)
    roster_id = "c0-roster-" + uuid4().hex
    _capture_json(evidence_session, tmp_path, roster_id, "roster", roster, roster_seen)
    fixture["roster_capture_ids"] = [roster_id]
    discovery_id = "c0-discovery-" + uuid4().hex
    _capture_json(evidence_session, tmp_path, discovery_id, "discovery",
                  {"fixtures": [fixture]}, origin - timedelta(minutes=30))
    calls = []
    result = orchestrate_local_opportunity_batch(
        evidence_session, protocol_id=PROTOCOL_ID, provider_id="fixture",
        batch_id="batch-" + uuid4().hex, discovery_capture_ids=[discovery_id],
        body_store=tmp_path, as_of=now, max_fixtures=1,
        protocol={"schema": "c0_market_study_v1", "protocol_id": PROTOCOL_ID},
        predictor=lambda candidate: calls.append(candidate),
    )
    if withdrawn:
        assert not calls
        assert "latest_roster_withdrawn" in result["fixtures"][0]["rejection_reasons"]
    else:
        assert len(calls) == 1
        assert calls[0]["roster_results"]["team-a"]["status"] == "prior_known"
        assert calls[0]["c0_request"]["roster_a"] == ["a1", "a2", "a3", "a4", "a5"]
        assert calls[0]["c0_request"]["mode"] == "full"
        assert calls[0]["c0_request"]["decision_at"] == origin.isoformat()
        assert not result["fixtures"][0]["forward_admitted"]


def test_announced_exact_fixture_does_not_require_prior_unit_fallback():
    announced = {
        "team_id": "team-a", "match_id": "exact-fixture", "competition_context": "LCK",
        "roster": ["a1", "a2", "a3", "a4", "a5"], "status": "announced",
        "published_at": "2026-10-06T10:00:00+00:00",
    }
    route = resolve_c0_roster(
        team_id="team-a", match_id="exact-fixture",
        cutoff_at=datetime(2026, 10, 6, 12, tzinfo=UTC),
        competition_context="LCK", competitive_unit_id=None, announcements=[announced],
    )
    assert route["status"] == "announced"
    assert route["announcement"]["roster"] == announced["roster"]
    missing = resolve_c0_roster(
        team_id="team-a", match_id="another-fixture",
        cutoff_at=datetime(2026, 10, 6, 12, tzinfo=UTC),
        competition_context="LCK", competitive_unit_id=None, announcements=[announced],
    )
    assert missing["announcement"] is None


@pytest.mark.parametrize("publication", ["bad", "2026-10-06T11:59:00", None])
def test_latest_capture_with_unverified_publication_never_revives_old_roster(publication):
    old = {"team_id": "1", "match_id": "old", "competition_context": "LCK",
           "competitive_unit_id": "first", "roster": [1, 2, 3, 4, 5],
           "published_at": "2026-10-05T12:00:00+00:00", "status": "released"}
    latest = {**old, "match_id": "new", "published_at": publication,
              "response_received_at": "2026-10-06T11:59:01+00:00"}
    route = resolve_c0_roster(team_id="1", match_id="new",
        cutoff_at=datetime(2026, 10, 6, 12, tzinfo=UTC),
        competition_context="LCK", competitive_unit_id="first", announcements=[old, latest])
    assert route["announcement"] is None
    assert route["missing_reason"] == "roster_publication_unverified"
    latest["status"] = "withdrawn"
    route = resolve_c0_roster(team_id="1", match_id="new",
        cutoff_at=datetime(2026, 10, 6, 12, tzinfo=UTC),
        competition_context="LCK", competitive_unit_id="first", announcements=[old, latest])
    assert route["announcement"] is None
    assert route["missing_reason"] == "latest_announcement_retracted"
