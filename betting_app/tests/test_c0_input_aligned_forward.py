"""Retained-only readiness inspection; no source access or configured database."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.analysis.c0_input_aligned_forward import inspect_forward_readiness

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json"
CONTRACTS = ROOT / "data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/source_contracts.json"
SOURCE_IDS = {
    "oddspapi:v4:fixtures", "oddspapi:v4:pinnacle:match_winner",
    "bookmaker:totalbet:match_winner", "bookmaker:lebull:match_winner",
    "golgg:completed-history", "liquipedia:context",
    "official:unit-and-announcement", "official:actual-start-and-settlement",
}


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bound_protocol(tmp_path):
    evidence = tmp_path / "retained"
    evidence.mkdir()
    forward = tmp_path / "forward.json"
    write_json(forward, {
        "schema": "c0_market_study_v1", "protocol_id": "native-c0-market-study-v1",
        "source_contracts": json.loads(CONTRACTS.read_text())["sources"],
        "evidence_root": str(evidence), "trusted_evidence_refs": [],
    })
    protocol = tmp_path / "protocol.json"
    value = {
        "schema": "c0_input_aligned_study_v1", "run_id": "unit_readiness",
        "forward_protocol_path": str(forward),
        "forward_protocol_sha256": hashlib.sha256(forward.read_bytes()).hexdigest(),
        "activation_matrix_path": str(MATRIX),
        "activation_matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "source_contracts_path": str(CONTRACTS),
        "source_contracts_sha256": hashlib.sha256(CONTRACTS.read_bytes()).hexdigest(),
    }
    write_json(protocol, value)
    return protocol, value, forward, evidence


def test_missing_authority_is_resolved_per_source_without_approving_candidate_urls(bound_protocol, tmp_path):
    protocol, _, _, _ = bound_protocol
    result = inspect_forward_readiness(protocol, output_dir=tmp_path / "report", now=datetime(2026, 10, 6, tzinfo=UTC))
    assert result["status"] == "BLOCKED_EXTERNAL_PREREQUISITES"
    assert result["g0_status"] == "G0_BLOCKED"
    assert result["external_requests"] == result["real_fit_count"] == 0
    assert {row["source_id"] for row in result["sources"]} == SOURCE_IDS
    for source in result["sources"]:
        assert source["effective_authentication"] == "unverified"
        assert source["acquisition_approved"] is False
        assert {"provider_permission", "retention", "entitlement", "quota", "approved_locator", "participant_binding", "retained_capture"} <= set(source["missing_records"])
        assert source["source_contract_sha256"]
    pinnacle = next(row for row in result["sources"] if row["source_id"] == "oddspapi:v4:pinnacle:match_winner")
    assert pinnacle["candidate_locator"] == "https://api.oddspapi.io/v4/odds"
    assert pinnacle["approved_locator"] is None
    stored = json.loads((tmp_path / "report" / "readiness.json").read_text())
    assert stored == result
    assert result["local_mechanics"]["status"] == "EXERCISED_LOCAL_SOFTWARE_ONLY"
    assert result["local_mechanics"]["external_requests"] == 0
    assert result["local_mechanics"]["passed"]
    cases = {row["case"]: row for row in result["local_mechanics"]["quote_cases"]}
    assert cases["valid"]["admitted"] == 1
    assert all(cases[name]["admitted"] == 0 for name in ("withdrawn", "live", "unknown_live", "invalid", "clock_unknown", "start_unknown", "started"))
    assert result["local_mechanics"]["absence_accounting"]["preserved_count"] == 3


def test_original_deadline_expires_without_rolling_or_changing_start(bound_protocol, tmp_path):
    protocol, _, _, _ = bound_protocol
    result = inspect_forward_readiness(protocol, output_dir=tmp_path / "expired", now=datetime(2026, 10, 31, tzinfo=UTC))
    assert result["status"] == "EXPIRED_UNSTARTED"
    assert result["activation_deadline"] == "2026-10-31T00:00:00+00:00"
    assert result["start_at"] == "2026-11-01T00:00:00+00:00"
    assert result["real_fit_count"] == 0


def test_synthetic_authenticated_flags_and_broad_approval_do_not_grant_permission(bound_protocol, tmp_path):
    protocol, value, forward, evidence = bound_protocol
    candidate = evidence / "authentication.json"
    digest = write_json(candidate, [{
        "schema": "native_c0_source_adjudication_v1", "source_id": "oddspapi:v4:fixtures",
        "protocol_id": "native-c0-market-study-v1", "decision": "authenticated",
        "evidence_class": "synthetic", "permission_evidence_refs": [],
    }])
    value["approved_for_collection"] = True
    value["user_authorization"] = "orchestrate do everything"
    value["forward_evidence"] = {"authentication_ref": {"uri": "authentication.json", "sha256": digest}}
    write_json(protocol, value)
    result = inspect_forward_readiness(protocol, output_dir=tmp_path / "synthetic", now=datetime(2026, 10, 6, tzinfo=UTC))
    source = next(row for row in result["sources"] if row["source_id"] == "oddspapi:v4:fixtures")
    assert not source["acquisition_approved"]
    assert source["effective_authentication"] == "unverified"
    assert source["adjudications"][0]["reasons"]
    assert result["g0_status"] == "G0_BLOCKED"


def test_invalid_retained_evidence_hash_is_explicit_denial_not_silent_absence(bound_protocol, tmp_path):
    protocol, value, _, evidence = bound_protocol
    write_json(evidence / "authorization.json", {"approved": True})
    value["forward_evidence"] = {"authorization_ref": {"uri": "authorization.json", "sha256": "0" * 64}}
    write_json(protocol, value)
    result = inspect_forward_readiness(protocol, output_dir=tmp_path / "invalid", now=datetime(2026, 10, 6, tzinfo=UTC))
    assert result["status"] == "BLOCKED_EXTERNAL_PREREQUISITES"
    assert any(row["reason"] == "retained_evidence_invalid" for row in result["evidence_errors"])
    assert not any(row["acquisition_approved"] for row in result["sources"])


@pytest.mark.parametrize("evidence_class", ["authentic", "synthetic"])
def test_pinned_permission_scope_is_source_bound_and_never_substitutes_capture_or_g0(bound_protocol, tmp_path, evidence_class):
    protocol, value, forward, retained = bound_protocol
    original = json.loads(forward.read_text())
    source_id = "oddspapi:v4:fixtures"
    locator = "https://api.oddspapi.io/v4/fixtures"
    original["source_contracts"][0]["approved_locator"] = locator
    proof = {
        "source_id": source_id, "approved_locator": locator,
        "evidence_class": evidence_class,
        "scope": {"request": True, "use": True, "retention": False, "entitlement": True, "quota": True},
        "effective_from": "2026-10-01T00:00:00+00:00",
        "effective_until": "2026-11-01T00:00:00+00:00",
    }
    digest = write_json(retained / "permission.json", proof)
    original["trusted_evidence_refs"] = [{"uri": "permission.json", "sha256": digest, "kind": "source_permission"}]
    value["forward_protocol_sha256"] = write_json(forward, original)
    write_json(protocol, value)
    result = inspect_forward_readiness(protocol, output_dir=tmp_path / "partial", now=datetime(2026, 10, 6, tzinfo=UTC))
    source = next(row for row in result["sources"] if row["source_id"] == source_id)
    assert source["record_resolution"]["provider_permission"] is (evidence_class == "authentic")
    assert source["record_resolution"]["entitlement"] is (evidence_class == "authentic")
    assert "retention" in source["missing_records"]
    assert "retained_capture" in source["missing_records"]
    assert "participant_binding" in source["missing_records"]
    assert not source["acquisition_approved"]
    other = next(row for row in result["sources"] if row["source_id"] == "oddspapi:v4:pinnacle:match_winner")
    assert not other["record_resolution"]["provider_permission"]
    assert result["g0_status"] == "G0_BLOCKED"


def test_bound_file_mutation_and_overwrite_are_refused(bound_protocol, tmp_path):
    protocol, _, forward, _ = bound_protocol
    output = tmp_path / "immutable"
    inspect_forward_readiness(protocol, output_dir=output, now=datetime(2026, 10, 6, tzinfo=UTC))
    original = (output / "readiness.json").read_bytes()
    with pytest.raises(FileExistsError):
        inspect_forward_readiness(protocol, output_dir=output, now=datetime(2026, 10, 7, tzinfo=UTC))
    assert (output / "readiness.json").read_bytes() == original
    forward.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        inspect_forward_readiness(protocol, output_dir=tmp_path / "mutated", now=datetime(2026, 10, 6, tzinfo=UTC))
    assert not (tmp_path / "mutated").exists()
