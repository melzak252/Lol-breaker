"""Retained-only native C0 forward evidence. No collection, training or deployment.

A prediction proves local compatibility, not authority, source completeness or G0.
All authority references must resolve to retained bytes pinned by the frozen protocol.
"""
from __future__ import annotations

import base64
import binascii
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence, TYPE_CHECKING
from urllib.parse import urlsplit

from sqlalchemy import select

from betting_app.models.prediction import ForecastEvidenceSnapshot

if TYPE_CHECKING:
    from betting_app.services.c0_inference import C0InferenceClient

UTC = timezone.utc
PROTOCOL_ID = "native-c0-market-study-v1"
NATIVE_NAME = "Causal-C0"
NATIVE_VERSION = "c0-native-2026-w32-e12-v1"
DEADLINE = datetime(2026, 10, 31, tzinfo=UTC)
SHAPES = {
    "raw.players": ("float32", (2, 5, 21), 4),
    "raw.core": ("float32", (2, 5), 4),
    "raw.team": ("float32", (2, 11), 4),
    "raw.w20": ("float32", (2, 10), 4),
    "raw.gates": ("float32", (2, 2), 4),
    "hist.history": ("float16", (2, 5, 16, 33), 2),
    "hist.mask": ("uint8", (2, 5, 16), 1),
    "hist.champions": ("uint16", (2, 5, 16), 2),
}
DEPENDENCY_KEYS = {
    "schema", "protocol_id", "snapshot_sha256", "tensor_sha256", "builder",
    "raw_history", "release_map", "feature_cutoff_at", "feature_source_max_day",
    "coverage_through_exclusive", "source_capture_refs", "recorded_at", "evidence_class",
}
ADJUDICATION_KEYS = {
    "schema", "protocol_id", "source_id", "source_contract_sha256", "approved_locator",
    "permission_evidence_refs", "capture_refs", "schema_identity_evidence_refs",
    "adjudicator", "adjudicated_at", "decision", "evidence_class",
}


def _json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _clock(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must be timezone-aware")
    return value.astimezone(UTC)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _atomic_bytes(path: Path, content: bytes, *, idempotent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".c0-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not idempotent or path.read_bytes() != content:
                raise ValueError(f"existing artifact has different bytes or cannot be overwritten: {path}")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def persist_receipt(receipt: Mapping[str, Any], receipt_dir: Path) -> str:
    content = canonical_bytes(receipt)
    digest = hashlib.sha256(content).hexdigest()
    _atomic_bytes(Path(receipt_dir) / (digest + ".json"), content, idempotent=True)
    return digest


def snapshot_tensor_records(encoded_snapshot: Mapping[str, Any]) -> tuple[dict, dict]:
    if set(encoded_snapshot) != {"target", "raw", "hist"}:
        raise ValueError("native snapshot requires target/raw/hist")
    target = encoded_snapshot["target"]
    for side in ("a", "b"):
        roster = target.get("roster_" + side)
        if not isinstance(roster, list) or len(roster) != 5 or len(set(map(str, roster))) != 5:
            raise ValueError("native snapshot requires five distinct players")
        for prefix in ("tensor", "raw_tensor", "temporal_tensor"):
            if target.get(prefix + "_player_order_" + side) != roster:
                raise ValueError("native tensor player order mismatch")
    if set(map(str, target["roster_a"])) & set(map(str, target["roster_b"])):
        raise ValueError("native snapshot sides share players")
    records = {"raw": {}, "hist": {}}
    hashes = {}
    for family in records:
        expected = {key.split(".")[1] for key in SHAPES if key.startswith(family + ".")}
        if not isinstance(encoded_snapshot[family], Mapping) or set(encoded_snapshot[family]) != expected:
            raise ValueError("native snapshot has unexpected tensor keys")
    for key, (dtype, shape, width) in SHAPES.items():
        family, name = key.split(".")
        value = encoded_snapshot[family][name]
        if set(value) != {"dtype", "shape", "data_base64"} or value["dtype"] != dtype or value["shape"] != list(shape):
            raise ValueError("native tensor dtype/shape mismatch")
        try:
            content = base64.b64decode(value["data_base64"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("native tensor bytes invalid") from exc
        if len(content) != math.prod(shape) * width or base64.b64encode(content).decode("ascii") != value["data_base64"]:
            raise ValueError("native tensor byte length/encoding mismatch")
        digest = hashlib.sha256(content).hexdigest()
        records[family][name] = {"dtype": dtype, "shape": list(shape),
                                 "content_base64": value["data_base64"], "sha256": digest}
        hashes[key] = digest
    return records, hashes


def _contracts(protocol: Mapping[str, Any]) -> dict[str, Mapping]:
    values = protocol.get("source_contracts", {})
    if isinstance(values, Mapping):
        return dict(values)
    if isinstance(values, list):
        return {item["source_id"]: item for item in values if isinstance(item, Mapping) and "source_id" in item}
    return {}


def _retained_blob(ref: Any, protocol: Mapping[str, Any], *, trusted_kind: str | None = None) -> bytes:
    if not isinstance(ref, Mapping) or not _digest(ref.get("sha256")):
        raise ValueError("retained evidence reference requires URI/SHA")
    uri = ref.get("uri", ref.get("evidence_uri"))
    if not isinstance(uri, str):
        raise ValueError("retained evidence URI missing")
    parts = urlsplit(uri)
    if parts.scheme not in {"", "file", "local", "sha256"} or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("only credential-free retained local evidence can be read")
    root_value = protocol.get("evidence_root")
    if not root_value:
        raise ValueError("retained evidence root missing")
    root = Path(root_value).resolve()
    if parts.scheme == "sha256":
        path = root / (ref["sha256"] + ".json")
    else:
        path = Path(parts.path if parts.scheme == "file" else (parts.netloc + parts.path if parts.scheme == "local" else uri))
        path = path if path.is_absolute() else root / path
    path = path.resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("retained reference escapes root or is absent")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != ref["sha256"]:
        raise ValueError("retained reference hash mismatch")
    if trusted_kind is not None:
        pinned = protocol.get("trusted_evidence_refs", [])
        if not any(item.get("sha256") == ref["sha256"] and item.get("uri", item.get("evidence_uri")) == uri
                   and item.get("kind") == trusted_kind for item in pinned if isinstance(item, Mapping)):
            raise ValueError("authority reference is not pinned by protocol")
    return body


def _retained_ref(ref: Any, protocol: Mapping[str, Any], *, trusted_kind: str | None = None) -> dict:
    body = _retained_blob(ref, protocol, trusted_kind=trusted_kind)
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("retained proof must be a JSON object")
    return value


def validate_source_adjudication(receipt: Mapping[str, Any], *, protocol: Mapping[str, Any], captures: Sequence[Mapping]) -> list[str]:
    """Derive effective authentication; never change the original capture enum."""
    try:
        if set(receipt) != ADJUDICATION_KEYS or receipt["schema"] != "native_c0_source_adjudication_v1" or receipt["protocol_id"] != PROTOCOL_ID:
            raise ValueError("adjudication schema")
        contract = _contracts(protocol).get(receipt["source_id"])
        if not isinstance(contract, Mapping) or receipt["source_contract_sha256"] != _sha(contract):
            raise ValueError("source contract mismatch")
        locator = receipt["approved_locator"]
        url = urlsplit(locator)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("unsafe locator")
        if (url.hostname in {"example.com", "example.org", "example.net", "localhost"}
                or url.hostname.endswith((".example", ".test", ".invalid", ".localhost"))
                or url.hostname.startswith("example.")):
            raise ValueError("fixture locator")
        approved = contract.get("approved_locators", [contract.get("approved_locator")])
        if locator not in approved or receipt["decision"] != "authenticated" or receipt["evidence_class"] != "authentic" or not receipt["adjudicator"]:
            raise ValueError("locator/decision")
        adjudicated = _clock(receipt["adjudicated_at"])
        capture_map = {item["capture_id"]: item for item in captures}
        refs = receipt["capture_refs"]
        if not refs or not receipt["permission_evidence_refs"] or not receipt["schema_identity_evidence_refs"]:
            raise ValueError("proof absent")
        permissions = [_retained_ref(ref, protocol, trusted_kind="source_permission") for ref in receipt["permission_evidence_refs"]]
        identities = [_retained_ref(ref, protocol, trusted_kind="source_identity") for ref in receipt["schema_identity_evidence_refs"]]
        for ref in refs:
            capture = capture_map.get(ref.get("capture_id"))
            if capture is None or capture.get("sha256") != ref.get("sha256") or capture.get("source_class") == "local_fixture":
                raise ValueError("capture reference invalid")
            body = base64.b64decode(capture["content_base64"], validate=True)
            if hashlib.sha256(body).hexdigest() != ref["sha256"]:
                raise ValueError("capture body mismatch")
            request_metadata = capture.get("request_metadata")
            if request_metadata is None:
                request_metadata = json.loads(capture.get("request_metadata_json", "{}"))
            actual_locator = request_metadata.get("url", capture.get("approved_locator"))
            if actual_locator is None and request_metadata.get("origin") and request_metadata.get("path"):
                actual_locator = request_metadata["origin"] + request_metadata["path"]
            # Sanitized immutable request metadata may retain an origin/path pair.
            if isinstance(actual_locator, Mapping):
                actual_locator = actual_locator.get("origin", "") + actual_locator.get("path", "")
            if actual_locator != locator:
                raise ValueError("capture locator mismatch")
            started, received, recorded = (_clock(capture[key]) for key in ("request_started_at", "response_received_at", "recorded_at"))
            if not started <= received <= recorded <= adjudicated:
                raise ValueError("capture clock chain")
            if capture.get("observed_at") is not None and not started <= _clock(capture["observed_at"]) <= recorded:
                raise ValueError("capture observation clock chain")
            if capture.get("first_observed_at") is not None and _clock(capture["first_observed_at"]) > received:
                raise ValueError("first observation follows retained response")
            if capture.get("source_id") != receipt["source_id"]:
                raise ValueError("capture stable source ID mismatch")
            if not any(proof.get("source_id") == receipt["source_id"] and proof.get("approved_locator") == locator
                       and all(proof.get("scope", {}).get(scope) is True for scope in ("request", "use", "retention", "entitlement", "quota"))
                       and _clock(proof["effective_from"]) <= started < _clock(proof["effective_until"])
                       for proof in permissions):
                raise ValueError("permission scope/dates")
            if not any(proof.get("source_id") == receipt["source_id"] and proof.get("source_contract_sha256") == receipt["source_contract_sha256"]
                       and proof.get("schema_confirmed") is True and proof.get("stable_source_id_confirmed") is True
                       and proof.get("participant_bindings_confirmed") is True and ref["sha256"] in proof.get("capture_sha256", [])
                       for proof in identities):
                raise ValueError("identity/schema proof")
    except (ValueError, TypeError, KeyError, OSError, binascii.Error):
        return ["source_authentication_unverified"]
    return []


def validate_dependency_receipt(receipt: Mapping[str, Any] | None, *, protocol: Mapping[str, Any], encoded_snapshot: Mapping, sources: Sequence[Mapping]) -> list[str]:
    try:
        if not isinstance(receipt, Mapping) or set(receipt) != DEPENDENCY_KEYS or receipt["schema"] != "native_c0_dependency_v1" or receipt["protocol_id"] != PROTOCOL_ID:
            raise ValueError("dependency schema")
        _, hashes = snapshot_tensor_records(encoded_snapshot)
        if receipt["snapshot_sha256"] != _sha(encoded_snapshot) or receipt["tensor_sha256"] != hashes:
            raise ValueError("dependency snapshot mismatch")
        target = encoded_snapshot["target"]
        cutoff = datetime.combine(date.fromisoformat(target["cutoff_day"]), datetime.min.time(), UTC)
        if _clock(receipt["feature_cutoff_at"]) != cutoff or receipt["feature_source_max_day"] != target.get("feature_source_max_day"):
            raise ValueError("dependency cutoff mismatch")
        if not receipt["feature_source_max_day"] or date.fromisoformat(receipt["feature_source_max_day"]) >= cutoff.date():
            raise ValueError("future feature history")
        if _clock(receipt["coverage_through_exclusive"]) < cutoff or receipt["evidence_class"] != "authentic":
            raise ValueError("coverage unknown")
        builder = receipt["builder"]
        if set(builder) != {"id", "code_sha256", "replay_sha256", "execution_receipt_refs"} or not builder["id"] or not _digest(builder["code_sha256"]) or not _digest(builder["replay_sha256"]):
            raise ValueError("builder unknown")
        executions = [_retained_ref(ref, protocol, trusted_kind="native_builder_execution") for ref in builder["execution_receipt_refs"]]
        if not executions:
            raise ValueError("no actual builder proof")
        for field in ("raw_history", "release_map"):
            item = receipt[field]
            if set(item) != {"sha256", "evidence_refs"} or not _digest(item["sha256"]) or not item["evidence_refs"]:
                raise ValueError("history/release unknown")
            for ref in item["evidence_refs"]:
                proof = _retained_ref(ref, protocol, trusted_kind="native_history")
                if proof.get("sha256") != item["sha256"] or _clock(proof["coverage_through_exclusive"]) < cutoff:
                    raise ValueError("history proof mismatch")
                content_ref = proof["content_ref"]
                if content_ref.get("sha256") != item["sha256"]:
                    raise ValueError("history bytes not retained")
                _retained_blob(content_ref, protocol)
        captures = {item.get("capture_id", item.get("source_id")): item for item in sources}
        if not receipt["source_capture_refs"]:
            raise ValueError("history capture references absent")
        for ref in receipt["source_capture_refs"]:
            actual = captures.get(ref["capture_id"])
            if actual is None or actual.get("sha256") != ref["sha256"]:
                raise ValueError("history capture mismatch")
            body = base64.b64decode(actual["content_base64"], validate=True)
            if hashlib.sha256(body).hexdigest() != ref["sha256"]:
                raise ValueError("history capture bytes mismatch")
            decision = _clock(encoded_snapshot["target"]["effective_decision_at"])
            if any(_clock(actual[field]) > decision for field in
                   ("response_received_at", "first_observed_at", "observed_at") if actual.get(field)):
                raise ValueError("history capture unavailable at decision")
        if not any(proof.get("schema") == "native_c0_builder_execution_v1" and proof.get("protocol_id") == PROTOCOL_ID
                   and proof.get("snapshot_sha256") == receipt["snapshot_sha256"] and proof.get("tensor_sha256") == hashes
                   and proof.get("builder_id") == builder["id"] and proof.get("code_sha256") == builder["code_sha256"]
                   and proof.get("replay_sha256") == builder["replay_sha256"] and proof.get("raw_history_sha256") == receipt["raw_history"]["sha256"]
                   and proof.get("release_map_sha256") == receipt["release_map"]["sha256"]
                   and proof.get("source_capture_refs") == receipt["source_capture_refs"]
                   and proof.get("feature_cutoff_at") == receipt["feature_cutoff_at"]
                   and proof.get("feature_source_max_day") == receipt["feature_source_max_day"]
                   and proof.get("coverage_through_exclusive") == receipt["coverage_through_exclusive"]
                   and _clock(proof["executed_at"]) <= _clock(receipt["recorded_at"])
                   for proof in executions):
            raise ValueError("builder execution mismatch")
    except (ValueError, TypeError, KeyError, OSError, binascii.Error):
        return ["native_history_coverage_unverified"]
    return []


def evaluate_c0_readiness(protocol: Mapping[str, Any], *, now: datetime, authorization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    clock = _clock(now)
    reasons = []
    if protocol.get("schema") != "c0_market_study_v1" or protocol.get("protocol_id") != PROTOCOL_ID:
        reasons.append("protocol_identity_invalid")
    try:
        if not isinstance(authorization, Mapping) or authorization.get("schema") != "native_c0_authorization_v1" or authorization.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("authority absent")
        recorded = _clock(authorization["recorded_at"])
        if not recorded < DEADLINE or recorded > clock:
            raise ValueError("activation deadline")
        approvals = [_retained_ref(ref, protocol, trusted_kind="acquisition_authority") for ref in authorization["approval_evidence_refs"]]
        compatibility = _retained_ref(authorization["native_compatibility_receipt_ref"], protocol, trusted_kind="native_compatibility")
        if not approvals or not any(proof.get("protocol_id") == PROTOCOL_ID and proof.get("activation_deadline") == DEADLINE.isoformat()
                                   and proof.get("approved") is True and _clock(proof["approved_at"]) <= recorded for proof in approvals):
            raise ValueError("authority proof")
        if compatibility.get("model_name") != NATIVE_NAME or compatibility.get("model_version") != NATIVE_VERSION or compatibility.get("mode") != "full" or compatibility.get("status") != "PASS":
            raise ValueError("native compatibility")
        native_inventory_hash = protocol.get("native_inventory_sha256")
        if native_inventory_hash is None and isinstance(protocol.get("native_inventory"), Mapping):
            native_inventory_hash = _sha(protocol["native_inventory"])
        if not _digest(native_inventory_hash) or compatibility.get("native_inventory_sha256") != native_inventory_hash:
            raise ValueError("native compatibility must bind frozen inventory")
        complete_recipe = compatibility["complete_recipe"]
        compatibility_recorded = _clock(compatibility["recorded_at"])
        if (compatibility_recorded > recorded
                or any(_clock(complete_recipe[key]) >= compatibility_recorded
                       for key in ("fit_cutoff_at", "calibration_cutoff_at"))):
            raise ValueError("complete native recipe availability unverified")
        contracts = _contracts(protocol)
        if len(contracts) != 8 or authorization["source_contract_sha256"] != {key: _sha(value) for key, value in contracts.items()}:
            raise ValueError("source contracts")
        if not authorization.get("adjudication_sha256") or any(authorization.get(key) != "PASS" for key in ("native_status", "source_status", "g0_status")):
            raise ValueError("source/G0 status")
        verified_adjudications = protocol.get("authentication_receipts", [])
        captures = protocol.get("captures", [])
        hashes = {_sha(receipt) for receipt in verified_adjudications
                  if _clock(receipt["adjudicated_at"]) <= recorded
                  and not validate_source_adjudication(receipt, protocol=protocol, captures=captures)}
        if set(authorization["adjudication_sha256"]) != hashes or len({receipt.get("source_id") for receipt in verified_adjudications if _sha(receipt) in hashes}) != 8:
            raise ValueError("source authentication")
        g0 = _retained_ref(authorization["g0_receipt_ref"], protocol, trusted_kind="native_g0")
        if (g0.get("protocol_id") != PROTOCOL_ID or g0.get("status") != "PASS"
                or g0.get("source_contract_sha256") != authorization["source_contract_sha256"]
                or g0.get("native_inventory_sha256") != native_inventory_hash
                or not _clock(g0["recorded_at"]) <= recorded
                or not g0.get("native_dependency_receipt_ref")):
            raise ValueError("G0 proof")
        native_dependency = _retained_ref(g0["native_dependency_receipt_ref"], protocol, trusted_kind="native_dependency")
        native_snapshot = _retained_ref(g0["native_snapshot_ref"], protocol, trusted_kind="native_snapshot")
        if validate_dependency_receipt(native_dependency, protocol=protocol, encoded_snapshot=native_snapshot, sources=captures):
            raise ValueError("native G0 lacks checked actual construction/coverage")
    except (ValueError, TypeError, KeyError, OSError):
        reasons.extend(["acquisition_authority_unverified", "source_authentication_unverified", "native_history_coverage_unverified", "G0_BLOCKED"])
    status = "READY_AUTHORIZED" if not reasons else ("EXPIRED_UNSTARTED" if clock >= DEADLINE else "BLOCKED_EXTERNAL_PREREQUISITES")
    return {"schema": "native_c0_readiness_v1", "protocol_id": PROTOCOL_ID, "status": status,
            "activation_deadline": DEADLINE.isoformat(), "start_at": "2026-11-01T00:00:00+00:00",
            "evaluated_at": clock.isoformat(), "reasons": sorted(set(reasons)), "real_fit_count": 0,
            "local_compatibility": "UNVERIFIED" if authorization is None else "RETAINED_PROOF_REQUIRED"}


def resolve_c0_roster(*, team_id: Any, match_id: Any, cutoff_at: datetime,
                      competition_context: Any, competitive_unit_id: Any,
                      announcements: Sequence[Mapping]) -> dict:
    """Latest applicable exact competitive-unit evidence; never target-result lookup."""
    cutoff = _clock(cutoff_at)
    candidates = []
    for item in announcements:
        exact_fixture = str(item.get("match_id")) == str(match_id)
        if (str(item.get("team_id")) != str(team_id) or not competition_context
                or item.get("competition_context") != competition_context
                or (not exact_fixture and (not competitive_unit_id or item.get("competitive_unit_id") != competitive_unit_id))
                or (exact_fixture and competitive_unit_id and item.get("competitive_unit_id") != competitive_unit_id)):
            continue
        try:
            available_clocks = [_clock(item[key]) for key in
                                ("response_received_at", "first_observed_at", "observed_at") if item.get(key)]
            published = item.get("published_at")
            if published is not None:
                try:
                    available_clocks.append(_clock(published))
                except (ValueError, TypeError):
                    item = {**item, "publication_clock_invalid": True}
            available = max(available_clocks)
            if available <= cutoff:
                candidates.append((available, item))
        except (ValueError, TypeError):
            return {"announcement": None, "status": "blocked", "missing_reason": "roster_evidence_clock_unverified",
                    "ambiguity_evidence": [dict(item)]}
    exact = [(at, item) for at, item in candidates if str(item.get("match_id")) == str(match_id)]
    applicable = exact or candidates
    if not applicable:
        return {"announcement": None, "status": "missing", "missing_reason": "roster_evidence_missing", "ambiguity_evidence": []}
    latest = max(at for at, _ in applicable)
    selected = [item for at, item in applicable if at == latest]
    if any(str(item.get("status", "")).lower() in {"withdrawn", "retracted", "cancelled"} for item in selected):
        reason = "latest_announcement_retracted"
    elif any(item.get("publication_clock_invalid") or
             (item.get("published_at") is None and item.get("effective_authentication_status") != "authenticated")
             for item in selected):
        reason = "roster_publication_unverified"
    elif any(not isinstance(item.get("roster"), list) or len(item["roster"]) != 5 or len(set(map(str, item["roster"]))) != 5 for item in selected):
        reason = "announced_roster_incomplete"
    elif len({tuple(sorted(map(str, item["roster"]))) for item in selected}) != 1:
        reason = "conflicting_latest_announcements"
    elif not exact and any(str(item.get("status", "")).lower() not in {"released", "confirmed", "announced", "active"} for item in selected):
        reason = "roster_evidence_missing"
    else:
        return {"announcement": dict(selected[0]), "status": "announced" if exact else "prior_known", "missing_reason": None, "ambiguity_evidence": []}
    return {"announcement": None, "status": "blocked", "missing_reason": reason, "ambiguity_evidence": list(selected)}


def record_c0_snapshot(session, client: C0InferenceClient, request: Mapping[str, Any], *,
                       snapshot: Mapping[str, Any] | None = None, evidence_key: str, sources,
                       input_source_ids, schedule, quotes, information_status: str,
                       dependency_receipt: Mapping[str, Any] | None = None,
                       authentication_receipts: Sequence[Mapping[str, Any]] = ()) -> ForecastEvidenceSnapshot:
    from betting_app.services.forecast_evidence import append_forecast_evidence, load_forecast_evidence
    if information_status not in {"local_fixture", "historical_scenario", "historical_reconstruction", "recorded"}:
        raise ValueError("unsupported generic information_status")
    normalized_sources = []
    for source in sources:
        item = {key: value for key, value in source.items() if key not in {"body_bytes", "body"}}
        if item.get("capture_id") is not None:
            item["provider_source_id"] = item.get("source_id")
            item["source_id"] = item["capture_id"]
        normalized_sources.append(_json(item))
    input_ids = list(input_source_ids)
    if (len(input_ids) != len(set(input_ids))
            or not set(input_ids).issubset({item["source_id"] for item in normalized_sources})):
        raise ValueError("input source IDs must cite retained consumed inputs")
    quote_ids = {item.get("capture_id") for item in quotes}
    if set(input_ids) & quote_ids:
        raise ValueError("quotes are not native sports forecast inputs")
    existing = load_forecast_evidence(session, evidence_key)
    if existing is not None:
        # A fresh native completion cannot masquerade as an idempotent old inference.
        raise ValueError("evidence_key already records a prediction; replay append_forecast_evidence instead")
    result = client.predict_evidence(request, snapshot=snapshot, canonical_match_id=schedule.get("canonical_match_id"))
    encoded = _json(result.encoded_snapshot)
    tensors, hashes = snapshot_tensor_records(encoded)
    prediction = result.prediction
    generated, received = _clock(result.generated_at), _clock(result.parent_received_at)
    if generated > received or received > datetime.now(UTC):
        raise ValueError("native completion/receive clock order invalid")
    if prediction.model_name != NATIVE_NAME or prediction.model_version != NATIVE_VERSION:
        raise ValueError("native C0 identity mismatch")
    diagnostics = _json(prediction.diagnostics)
    expected = diagnostics.get("input_hashes", {})
    for key, digest in hashes.items():
        if expected.get(key, expected.get(key.split(".")[1])) != digest:
            raise ValueError("native prediction tensor hash mismatch")
    target = encoded["target"]
    mode = request.get("mode", "full")
    if diagnostics.get("mode") != mode or prediction.best_of != target.get("best_of"):
        raise ValueError("native mode/best_of mismatch")
    provenance = _json(result.native_history_provenance)
    provenance["authenticated_coverage_through_exclusive"] = None
    cutoff = datetime.combine(date.fromisoformat(target["cutoff_day"]), datetime.min.time(), UTC)
    max_day = target.get("feature_source_max_day")
    history_status = "FROZEN_HISTORY_STALE_FOR_CUTOFF"
    freshness = provenance.get("dataset_freshness", {})
    inventory_max_day = freshness.get("raw_history_max_effective_release_day", max_day)
    if inventory_max_day and cutoff.date() <= date.fromisoformat(inventory_max_day) + timedelta(days=1):
        history_status = "FROZEN_HISTORY_CAUSAL_ONLY"
    provenance["history_status"] = history_status
    provenance["feature_cutoff_at"] = cutoff.isoformat()
    native_status = diagnostics.get("information_status", provenance.get("status"))
    protocol = schedule.get("protocol", {"protocol_id": schedule.get("protocol_id")})
    if dependency_receipt is None:
        replay = provenance.get("raw_feature_replay_source", {})
        dependency_receipt = {
            "schema": "native_c0_dependency_v1", "protocol_id": schedule.get("protocol_id", PROTOCOL_ID),
            "snapshot_sha256": _sha(encoded), "tensor_sha256": hashes,
            "builder": {"id": None, "code_sha256": None, "replay_sha256": replay.get("sha256") if snapshot is None else None, "execution_receipt_refs": []},
            "raw_history": {"sha256": provenance.get("raw_history_sha256") if snapshot is None else None, "evidence_refs": []},
            "release_map": {"sha256": provenance.get("release_map_sha256") if snapshot is None else None, "evidence_refs": []},
            "feature_cutoff_at": cutoff.isoformat(), "feature_source_max_day": max_day,
            "coverage_through_exclusive": None, "source_capture_refs": [],
            "recorded_at": datetime.now(UTC).isoformat(), "evidence_class": "unverified",
        }
    if set(dependency_receipt) != DEPENDENCY_KEYS or dependency_receipt.get("snapshot_sha256") != _sha(encoded) or dependency_receipt.get("tensor_sha256") != hashes:
        raise ValueError("dependency receipt does not bind exact native snapshot")
    run_id = protocol.get("run_id", schedule.get("run_id"))
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("C0 evidence requires a safe protocol run_id")
    receipt_dir = Path("data/05_model_input/c0_market_study") / run_id / "evidence/receipts"
    dependency_hash = persist_receipt(dependency_receipt, receipt_dir)
    authentication_hashes = []
    for receipt in authentication_receipts:
        if set(receipt) != ADJUDICATION_KEYS or receipt.get("schema") != "native_c0_source_adjudication_v1":
            raise ValueError("source adjudication schema invalid")
        authentication_hashes.append(persist_receipt(receipt, receipt_dir))
    reasons = validate_dependency_receipt(dependency_receipt, protocol=protocol, encoded_snapshot=encoded, sources=sources)
    if not reasons:
        provenance["authenticated_coverage_through_exclusive"] = dependency_receipt["coverage_through_exclusive"]
    elif snapshot is not None:
        provenance["history_status"] = "NATIVE_HISTORY_UNVERIFIED"
    payload = {
        "schema_version": 1, "information_status": information_status,
        "model": {"model_name": NATIVE_NAME, "model_version": NATIVE_VERSION,
                  "feature_version": prediction.feature_version, "mode": mode,
                  "c0_serving_contract": "native-c0-forward-v1",
                  "native_identity": diagnostics.get("model_identity", {})},
        "input": {"target": target, **tensors, "source_ids": list(input_source_ids),
                  "native_c0_history": provenance, "native_information_status": native_status,
                  "encoded_snapshot_sha256": _sha(encoded), "dependency_receipt": _json(dependency_receipt),
                  "dependency_receipt_sha256": dependency_hash,
                  "authentication_receipts": _json(authentication_receipts),
                  "authentication_receipt_sha256": authentication_hashes,
                  "admission_reasons": reasons},
        "forecast": {"prob_a": prediction.prob_a, "prob_b": prediction.prob_b,
                     "best_of": prediction.best_of, "mode": mode,
                     "model_name": NATIVE_NAME, "model_version": NATIVE_VERSION,
                     "feature_version": prediction.feature_version,
                     "generated_at": generated.isoformat(), "parent_received_at": received.isoformat(),
                     "simulated_origin_at": _clock(request.get("decision_at", target["effective_decision_at"])).isoformat(),
                     "availability_evidence": "actual_native_completion_not_authenticated"},
        "roster": {key: value for key, value in target.items() if "roster" in key or "player_order" in key or key in {"team1_id", "team2_id", "competition_context"}},
        "schedule": _json({key: value for key, value in schedule.items() if key != "protocol"}),
        "quotes": _json(quotes), "sources": normalized_sources,
    }
    return append_forecast_evidence(session, evidence_key=evidence_key, payload=payload)


def _encoded_from_envelope(envelope: Mapping) -> dict:
    value = envelope["input"]
    return {"target": value["target"], **{family: {key: {"dtype": item["dtype"], "shape": item["shape"], "data_base64": item["content_base64"]}
                                                   for key, item in value[family].items()} for family in ("raw", "hist")}}


def c0_forecast_link_reasons(envelope: Mapping, candidate: Mapping, *, recorded_at: datetime) -> list[str]:
    reasons = []
    try:
        target, forecast, schedule = envelope["input"]["target"], envelope["forecast"], envelope["schedule"]
        fixture = candidate["fixture"]
        teams = [str(item) for item in fixture.get("team_ids", [])]
        if (len(teams) != 2 or [str(target.get("team1_id")), str(target.get("team2_id"))] != teams
                or schedule.get("protocol_id") != PROTOCOL_ID or candidate["protocol_id"] != PROTOCOL_ID
                or schedule.get("provider_id") != candidate["provider_id"]
                or str(schedule.get("provider_fixture_id")) != str(candidate["fixture_id"])
                or str(schedule.get("canonical_fixture_id")) != str(fixture.get("canonical_fixture_id"))
                or forecast.get("best_of") != fixture.get("best_of") or target.get("best_of") != fixture.get("best_of")
                or target.get("competition_context") != fixture.get("competition_context")):
            reasons.append("forecast_envelope_fixture_mismatch")
        if (envelope["model"].get("model_name") != NATIVE_NAME
                or envelope["model"].get("model_version") != NATIVE_VERSION
                or envelope["model"].get("c0_serving_contract") != "native-c0-forward-v1"
                or envelope["model"].get("mode") != "full"
                or forecast.get("mode") != "full"):
            reasons.append("forecast_envelope_model_mismatch")
        origin = _clock(candidate["origin_at"])
        if any(_clock(item) != origin for item in (forecast["simulated_origin_at"], target["effective_decision_at"], schedule["origin_at"])):
            reasons.append("forecast_envelope_origin_mismatch")
        if any(_clock(schedule[key]) != _clock(candidate[key]) for key in ("scheduled_start_at", "known_start_bound")):
            reasons.append("forecast_envelope_schedule_mismatch")
        generated, received, recorded = _clock(forecast["generated_at"]), _clock(forecast["parent_received_at"]), _clock(recorded_at)
        if not origin <= generated <= received <= recorded <= origin + timedelta(seconds=60) or recorded >= _clock(candidate["known_start_bound"]) or recorded > datetime.now(UTC):
            reasons.append("forecast_envelope_time_invalid")
        p, q = forecast["prob_a"], forecast["prob_b"]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in (p, q)) or not math.isclose(p + q, 1, abs_tol=1e-8):
            reasons.append("forecast_envelope_probability_invalid")
        snapshot_tensor_records(_encoded_from_envelope(envelope))
        for team, side in zip(teams, ("a", "b")):
            roster = (candidate["rosters"].get(team) or {}).get("roster", [])
            if len(roster) != 5 or set(map(str, roster)) != set(map(str, target.get("roster_" + side, []))):
                reasons.append("forecast_envelope_roster_mismatch")
        expected = {item["capture_id"]: item for item in candidate["source_captures"]}
        actual = {item["source_id"]: item for item in envelope["sources"]}
        if set(expected) != set(actual):
            reasons.append("forecast_envelope_capture_mismatch")
        else:
            for key, capture in expected.items():
                source = actual[key]
                if any(source.get(field) != capture.get(field) for field in ("sha256", "evidence_uri", "source_class", "source_authentication_status")):
                    reasons.append("forecast_envelope_capture_mismatch")
                for field in ("request_started_at", "response_received_at", "first_observed_at", "observed_at", "retrieved_at", "published_at", "source_effective_at"):
                    a, b = source.get(field), capture.get(field)
                    if (a is None) != (b is None) or (a is not None and _clock(a) != _clock(b)):
                        reasons.append("forecast_envelope_capture_mismatch")
                if key in envelope["input"]["source_ids"] and max(_clock(capture[field]) for field in ("response_received_at", "first_observed_at", "observed_at") if capture.get(field)) > origin:
                    reasons.append("forecast_input_unavailable_at_cutoff")
            from betting_app.services.forecast_evidence import _decorate_provenance_records
            proof_context = candidate.get("protocol") or {}
            proof_captures = {item["capture_id"]: item for item in proof_context.get("captures", [])}
            proof_captures.update({key: {**capture, "content_base64": actual[key]["content_base64"]}
                                   for key, capture in expected.items()})
            announcements = []
            for key, capture in expected.items():
                content = base64.b64decode(actual[key]["content_base64"], validate=True)
                if hashlib.sha256(content).hexdigest() != capture["sha256"]:
                    raise ValueError("linked roster capture bytes mismatch")
                try:
                    body = json.loads(content)
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(body, dict) or "announcements" not in body:
                    continue
                records = _decorate_provenance_records(body["announcements"], capture)
                authenticated = any(
                    any(ref.get("capture_id") == key for ref in receipt.get("capture_refs", []))
                    and not validate_source_adjudication(receipt, protocol=proof_context,
                        captures=list(proof_captures.values()))
                    for receipt in [*proof_context.get("authentication_receipts", []),
                                    *envelope["input"].get("authentication_receipts", [])])
                for item in records:
                    item["effective_authentication_status"] = "authenticated" if authenticated else "unverified"
                    for clock in ("response_received_at", "first_observed_at"):
                        item[clock] = capture.get(clock)
                announcements.extend(records)
            for team, side in zip(teams, ("a", "b")):
                selected = resolve_c0_roster(
                    team_id=team, match_id=candidate["fixture_id"], cutoff_at=origin,
                    competition_context=candidate["fixture"].get("competition_context"),
                    competitive_unit_id=(candidate["fixture"].get("competitive_unit_ids") or {}).get(team),
                    announcements=announcements)
                announcement = selected.get("announcement") or {}
                if (selected.get("status") not in {"announced", "prior_known"}
                        or announcement.get("source_capture_id") not in envelope["input"]["source_ids"]
                        or set(map(str, announcement.get("roster", []))) !=
                           set(map(str, target.get("roster_" + side, [])))):
                    reasons.append("forecast_envelope_roster_source_unverified")
    except (ValueError, TypeError, KeyError):
        reasons.append("forecast_envelope_native_evidence_invalid")
    return sorted(set(reasons))


def export_c0_study_bundle(session, *, protocol, phase, authorization, body_store, output_dir) -> dict:
    """Seal membership before reading phase labels; copy immutable retained evidence."""
    from betting_app.services.forecast_evidence import export_opportunity_ledger, load_forecast_evidence, load_opportunity_events, load_source_capture
    if protocol.get("schema") != "c0_market_study_v1" or protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("C0 protocol required")
    phase = phase.upper()
    calendar = {"TRAIN": ("2026-11-01", "2027-11-01", "2027-11-15"),
                "CAL": ("2027-12-01", "2028-06-01", "2028-06-15"),
                "DEVELOPMENT": ("2028-07-01", "2029-07-01", "2029-07-15")}
    if phase not in calendar:
        raise ValueError("unknown phase")
    start, end, lock = [datetime.fromisoformat(value).replace(tzinfo=UTC) for value in calendar[phase]]
    destination = Path(output_dir)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    rows = load_opportunity_events(session, protocol_id=PROTOCOL_ID)
    primaries = {row.fixture_id: row for row in rows if row.primary_decision and row.origin_at is not None and start <= _clock(row.origin_at) < end}
    selected = [row for row in rows if row.fixture_id in primaries or
                (row.origin_at is not None and start <= _clock(row.origin_at) < end) or
                (row.origin_at is None and start <= _clock(row.recorded_at) < end)]
    # Reuse the ledger serializer, then retain phase-scoped rows without changing evidence.
    staging = destination / ".all-opportunities.jsonl"
    export_opportunity_ledger(session, protocol_id=PROTOCOL_ID, output_path=staging, max_rows=max(1, len(rows)))
    all_serialized = [json.loads(line) for line in staging.read_bytes().splitlines()]
    staging.unlink()
    selected_keys = {row.event_key for row in selected}
    opportunities = [item for item in all_serialized if item["event_key"] in selected_keys]
    for item in opportunities:
        item["canonical_match_id"] = item["payload"].get("canonical_fixture_id")
        item["nominal_cutoff_at"] = item["origin_at"]
    files, retained_bodies, retained_receipts = {}, {}, {}
    def emit(name, values):
        content = b"".join(canonical_bytes(value) + b"\n" for value in values)
        _atomic_bytes(destination / name, content)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "row_count": len(values)}
    emit("opportunities.jsonl", opportunities)
    membership = {"schema": "c0_phase_membership_v1", "protocol_id": PROTOCOL_ID, "phase": phase,
                  "opportunity_sha256": files["opportunities.jsonl"]["sha256"],
                  "event_keys": sorted(selected_keys), "primary_fixture_ids": sorted(primaries)}
    _atomic_bytes(destination / "membership.json", canonical_bytes(membership))
    files["membership.json"] = {"sha256": _sha(membership)}
    envelopes, captures, dependencies, authentication, bindings = [], [], [], [], []
    capture_ids = sorted({capture for row in selected for capture in row.payload["source_capture_ids"]})
    envelope_keys = sorted({row.forecast_evidence_key for row in selected if row.forecast_evidence_key})
    for key in envelope_keys:
        payload = load_forecast_evidence(session, key)
        row = session.execute(select(ForecastEvidenceSnapshot).where(ForecastEvidenceSnapshot.evidence_key == key)).scalar_one()
        envelopes.append({"evidence_key": key, "content_sha256": row.content_sha256, "recorded_at": _clock(row.recorded_at).isoformat(), "payload": payload})
        dependencies.append(payload["input"]["dependency_receipt"])
        authentication.extend(payload["input"].get("authentication_receipts", []))
    decoded_bodies = []
    capture_ids = sorted(set(capture_ids) | {
        capture for row in selected for capture in row.payload.get("outcome_capture_ids", [])
    } | set(protocol.get("phase_label_capture_ids", {}).get(phase, []))
      | {capture["capture_id"] for capture in protocol.get("captures", [])}
      | {ref["capture_id"] for dependency in dependencies for ref in dependency.get("source_capture_refs", [])})
    authentication.extend(protocol.get("authentication_receipts", []))
    capture_ids = sorted(set(capture_ids) | {
        ref["capture_id"] for receipt in authentication for ref in receipt.get("capture_refs", [])})
    for capture_id in capture_ids:
        capture = load_source_capture(session, capture_id, body_store=Path(body_store))
        if capture is None:
            raise ValueError("linked capture missing")
        body = capture.pop("body_bytes", capture.pop("body", None))
        if body is None and capture.get("content_base64") is None:
            captures.append(_json(capture))
            continue
        if body is None:
            body = base64.b64decode(capture["content_base64"], validate=True)
        capture["content_base64"] = base64.b64encode(body).decode("ascii")
        digest = hashlib.sha256(body).hexdigest()
        relative = "bodies/" + digest + ".bin"
        _atomic_bytes(destination / relative, body, idempotent=True)
        retained_bodies[digest] = relative
        captures.append(_json(capture))
        try:
            decoded = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(decoded, dict):
            decoded_bodies.append((capture, decoded))
            for binding in [*decoded.get("bindings", []), *decoded.get("participant_bindings", [])]:
                if not isinstance(binding, dict):
                    raise ValueError("binding must be explicit object")
                bindings.append({**binding, "capture_id": capture_id, "source_sha256": digest})
    unique_auth = {_sha(item): item for item in authentication}
    for capture in captures:
        passed = [digest for digest, receipt in unique_auth.items() if any(ref.get("capture_id") == capture["capture_id"] for ref in receipt.get("capture_refs", []))
                  and not validate_source_adjudication(receipt, protocol=protocol, captures=captures)]
        capture["effective_authentication_status"] = "authenticated" if passed else "unverified"
        capture["authentication_receipt_sha256"] = sorted(passed)
    for values in (dependencies, list(unique_auth.values())):
        for receipt in values:
            digest = persist_receipt(receipt, destination / "receipts")
            retained_receipts[digest] = "receipts/" + digest + ".json"
    emit("envelopes.jsonl", envelopes)
    emit("captures.jsonl", captures)
    emit("bindings.jsonl", bindings)
    emit("dependencies.jsonl", list({_sha(item): item for item in dependencies}.values()))
    emit("authentication.jsonl", list(unique_auth.values()))
    # Labels are accessed only after the membership seal has been durably written.
    labels = []
    canonical_ids = {str(row.payload.get("canonical_fixture_id")) for row in primaries.values()}
    for capture, decoded in decoded_bodies:
        for label in decoded.get("labels", []):
            if not isinstance(label, dict):
                raise ValueError("label must be explicit object")
            if str(label.get("canonical_fixture_id", label.get("canonical_match_id"))) not in canonical_ids:
                continue
            labels.append({**label, "capture_id": capture["capture_id"], "source_sha256": capture["sha256"],
                           "source_id": capture.get("source_id"),
                           "published_at": label.get("published_at", capture.get("published_at")),
                           "response_received_at": capture.get("response_received_at"),
                           "first_observed_at": capture.get("first_observed_at"),
                           "observed_at": capture.get("observed_at"),
                           "recorded_at": capture.get("recorded_at"), "label_lock_at": lock.isoformat()})
    emit("labels.jsonl", labels)
    authority = _json(authorization) if authorization is not None else {}
    _atomic_bytes(destination / "authorization.json", canonical_bytes(authority))
    files["authorization.json"] = {"sha256": _sha(authority)}
    def retain_references(value):
        if isinstance(value, Mapping):
            if _digest(value.get("sha256")) and isinstance(value.get("uri", value.get("evidence_uri")), str):
                content = _retained_blob(value, protocol)
                try:
                    proof = json.loads(content)
                except (ValueError, UnicodeDecodeError):
                    proof = None
                if isinstance(proof, Mapping):
                    if canonical_bytes(proof) != content:
                        raise ValueError("retained receipt must use canonical UTF8 JSON")
                    relative = "receipts/" + value["sha256"] + ".json"
                    retained_receipts[value["sha256"]] = relative
                else:
                    relative = "bodies/" + value["sha256"] + ".bin"
                    retained_bodies[value["sha256"]] = relative
                _atomic_bytes(destination / relative, content, idempotent=True)
                retain_references(proof)
            else:
                for child in value.values():
                    retain_references(child)
        elif isinstance(value, list):
            for child in value:
                retain_references(child)
    for value in [authority, *dependencies, *unique_auth.values()]:
        retain_references(value)
    manifest = {"schema": "c0_market_study_bundle_v1", "protocol_id": PROTOCOL_ID,
                "protocol_sha256": _sha(protocol), "phase": phase, "files": files,
                "retained_bodies": retained_bodies, "retained_receipts": retained_receipts,
                "membership_sha256": _sha(membership)}
    _atomic_bytes(destination / "manifest.json", canonical_bytes(manifest))
    return manifest
