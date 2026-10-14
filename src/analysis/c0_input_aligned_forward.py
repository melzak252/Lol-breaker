"""Source-bound C0 readiness inspection. Retained files only; never acquires data.

The local mechanical exercise is synthetic software evidence, not source authority,
G0 admission, forecast generation or a replacement capture/ledger implementation.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from betting_app.services import c0_shadow_readiness as evidence

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[2]
FORWARD_PROTOCOL = ROOT / "data/05_model_input/c0_market_study/c0_market_contracts_20261006_01/protocol.json"
ACTIVATION_MATRIX = ROOT / "data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json"
SOURCE_CONTRACTS = ROOT / "data/05_model_input/betting_ev_reliability/forward_protocol_20260930_234022/source_contracts.json"
SOURCE_IDS = (
    "oddspapi:v4:fixtures", "oddspapi:v4:pinnacle:match_winner",
    "bookmaker:totalbet:match_winner", "bookmaker:lebull:match_winner",
    "golgg:completed-history", "liquipedia:context",
    "official:unit-and-announcement", "official:actual-start-and-settlement",
)


def _bound_json(path: Path, expected: str | None, inputs: list) -> dict:
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"bound input hash mismatch: {path}")
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError(f"bound input must be a JSON object: {path}")
    inputs.append({"path": str(path.resolve()), "sha256": digest})
    return value


def _path(value: str | None, default: Path, protocol_path: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    # Frozen repository paths remain repository relative; private bundles may
    # bind paths relative to their own protocol directory.
    return ROOT / path if str(path).startswith("data/") else protocol_path.parent / path


def _local_mechanics() -> dict:
    """Observe existing local admission APIs against explicit synthetic inputs."""
    import pandas as pd
    from betting_app.services import prematch_quote_service as quotes
    from betting_app.services.forecast_evidence import _canonical_event_payload
    from betting_app.services import team_alias_service as aliases

    cutoff = datetime(2026, 10, 6, 12, tzinfo=UTC)
    fixture = dict(id=1, normalized_team_a="Alpha", normalized_team_b="Beta",
                   league="LCK", status="scheduled",
                   start_time_normalized=(cutoff + timedelta(hours=1)).isoformat())
    prior = dict(id=1, canonical_match_id=1, bookmaker_id=1, bookmaker="Local",
                 raw_team_a="Alpha", raw_team_b="Beta", odds_a=3.0, odds_b=2.0,
                 is_live=0, scraped_at=cutoff-timedelta(seconds=100),
                 source_url=None, offer_url=None)
    cases = [("valid", {}, {}, True), ("withdrawn", {"odds_a": None, "odds_b": None}, {}, False),
             ("live", {"is_live": 1}, {}, False), ("unknown_live", {"is_live": None}, {}, False),
             ("invalid", {"odds_a": "bad"}, {}, False), ("clock_unknown", {"scraped_at": None}, {}, False),
             ("start_unknown", {}, {"start_time_normalized": None}, False),
             ("started", {}, {"start_time_normalized": cutoff.isoformat()}, False)]
    quote_cases = []
    for name, state_changes, fixture_changes, should_admit in cases:
        latest = {**prior, "id": 2, "scraped_at": cutoff, **state_changes}
        def local_query(sql, params, **kwargs):
            return pd.DataFrame([{**fixture, **fixture_changes}] if "FROM canonical_matches" in sql else [prior, latest])
        with (patch.object(quotes, "query_df", local_query),
              patch.object(aliases, "_resolve_durable_identity", return_value=None),
              patch.object(aliases, "_team_alias_columns", return_value=set())):
            context = quotes.load_prematch_quotes([1], decision_at=cutoff)[1]
        quote_cases.append({"case": name, "admitted": len(context.quotes),
                            "reasons": [item.reason for item in context.rejections],
                            "passed": bool(context.quotes) == should_admit})

    prediction = dict(model_name="Causal-C0", model_version=evidence.NATIVE_VERSION,
                      prediction_status="active", data_cutoff_at=cutoff-timedelta(seconds=20),
                      predicted_at=cutoff-timedelta(seconds=10))
    clock_cases = []
    for name, change, start, expected in (
        ("causal", {}, cutoff+timedelta(hours=1), True),
        ("future_issuance", {"predicted_at": cutoff+timedelta(seconds=1)}, cutoff+timedelta(hours=1), False),
        ("future_input", {"data_cutoff_at": cutoff+timedelta(seconds=1)}, cutoff+timedelta(hours=1), False),
        ("unknown_issuance", {"predicted_at": None}, cutoff+timedelta(hours=1), False),
        ("started", {}, cutoff, False),
    ):
        allowed, reason = quotes.prediction_time_is_actionable(
            {**prediction, **change}, decision_at=cutoff, start_at=start,
            model_name="Causal-C0", model_version=evidence.NATIVE_VERSION)
        clock_cases.append({"case": name, "actionable": allowed, "reason": reason, "passed": allowed == expected})

    released = dict(team_id="1", match_id="old", competition_context="LCK", competitive_unit_id="first",
                    roster=["1", "2", "3", "4", "5"], status="released",
                    published_at=(cutoff-timedelta(days=1)).isoformat(),
                    observed_at=(cutoff-timedelta(days=1)).isoformat())
    roster_cases = []
    for name, announcements, expected in (
        ("prior_known", [released], "prior_known"),
        ("withdrawn", [released, {**released, "status": "withdrawn", "observed_at": cutoff.isoformat(), "published_at": cutoff.isoformat()}], "blocked"),
        ("unknown_clock", [{**released, "observed_at": None, "published_at": None}], "blocked"),
        ("wrong_context", [{**released, "competition_context": "LCK-CL"}], "missing"),
    ):
        route = evidence.resolve_c0_roster(team_id="1", match_id="future", cutoff_at=cutoff,
            competition_context="LCK", competitive_unit_id="first", announcements=announcements)
        roster_cases.append({"case": name, "status": route["status"], "reason": route["missing_reason"],
                             "passed": route["status"] == expected})

    absences = []
    for reason in ("roster_evidence_missing", "latest_announcement_retracted", "latest_quote_invalid"):
        payload = dict(status="rejected", rejection_reasons=[reason], clocks={"origin_at": cutoff.isoformat()},
                       model_availability={"available": False, "reason": reason},
                       feature_availability={"available": False, "reason": reason},
                       inference_eligible=False, forward_admitted=False, information_status="local_fixture",
                       source_authentication_status="local_fixture", source_capture_ids=[])
        serialized, digest = _canonical_event_payload(payload)
        retained = json.loads(serialized)
        absences.append({"payload": retained, "sha256": digest, "forecast": None,
                         "passed": retained == payload and not retained["forward_admitted"]})
    additional = [
        {"scope": "immutable DB no-forecast accounting and C0 roster callback routing", "status": "NOT_EXECUTED",
         "requires": "Explicit disposable TEST_DATABASE_URL containing _test; never configured DB",
         "command": ".venv/bin/python -m pytest -q betting_app/tests/test_opportunity_ledger.py::test_discovery_preserves_unknown_fixture_and_every_rejection_without_forecast betting_app/tests/test_c0_forward_readiness.py::test_actual_captured_prior_route_precedes_callback_and_withdrawal_blocks betting_app/tests/test_c0_forward_readiness.py::test_phase_export_keeps_no_forecast_primary_and_seals_before_labels"},
    ]
    return {"status": "EXERCISED_LOCAL_SOFTWARE_ONLY", "evidence_class": "synthetic",
            "external_requests": 0, "database_requests": 0,
            "identity_scope": "Real strict alignment with an explicitly empty synthetic alias inventory; no user database lookup",
            "quote_cases": quote_cases, "clock_cases": clock_cases, "roster_cases": roster_cases,
            "absence_accounting": {"scope": "existing immutable payload validator only; no database ledger claim",
                                   "preserved_count": len(absences), "events": absences},
            "passed": all(row["passed"] for rows in (quote_cases, clock_cases, roster_cases, absences) for row in rows),
            "additional_checks": additional}


def inspect_forward_readiness(protocol_path, *, output_dir, now=None) -> dict:
    """Resolve eight source authorities and save immutable, hash-bound findings.

    Optional forward_evidence contains exact local authorization_ref,
    authentication_ref and captures_ref, each URI/SHA bound. The original C0
    protocol's trusted_evidence_refs remain the authority; a new study cannot
    promote its own synthetic receipts or a broad orchestration request.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"readiness output already exists: {destination}")
    clock = evidence._clock(datetime.now(UTC) if now is None else now)
    input_path = Path(protocol_path).resolve()
    inputs = []
    study = _bound_json(input_path, None, inputs)
    forward = _bound_json(_path(study.get("forward_protocol_path"), FORWARD_PROTOCOL, input_path),
                          study.get("forward_protocol_sha256"), inputs)
    matrix = _bound_json(_path(study.get("activation_matrix_path"), ACTIVATION_MATRIX, input_path),
                         study.get("activation_matrix_sha256"), inputs)
    original = _bound_json(_path(study.get("source_contracts_path"), SOURCE_CONTRACTS, input_path),
                           study.get("source_contracts_sha256"), inputs)
    contract_map = evidence._contracts(forward)
    originals = {item["source_id"]: item for item in original["sources"]}
    requirements = {item["source_id"]: item for item in matrix["sources"]}
    if any(set(mapping) != set(SOURCE_IDS) for mapping in (contract_map, originals, requirements)):
        raise ValueError("forward source identities must preserve exactly the original eight sources")
    if matrix["frozen_dates"]["readiness_deadline"] != evidence.DEADLINE.isoformat() or matrix["frozen_dates"]["train_start"] != "2026-11-01T00:00:00+00:00":
        raise ValueError("original forward dates may not roll")

    errors, retained_refs = [], []
    requested = study.get("forward_evidence", {})
    if not isinstance(requested, Mapping):
        raise ValueError("forward_evidence must contain scoped retained references")
    def retained(name, default):
        ref = requested.get(name + "_ref")
        if ref is None:
            return default
        try:
            body = evidence._retained_blob(ref, forward)
            value = json.loads(body)
            if not isinstance(value, type(default)):
                raise ValueError("retained evidence type invalid")
            retained_refs.append({"role": name, "uri": ref.get("uri", ref.get("evidence_uri")), "sha256": ref["sha256"]})
            return value
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append({"role": name, "reason": "retained_evidence_invalid", "detail": str(exc)})
            return default
    authorization = retained("authorization", {}) or None
    receipts = retained("authentication", forward.get("authentication_receipts", []))
    captures = retained("captures", forward.get("captures", []))
    for role, values in (("authentication", receipts), ("captures", captures)):
        if not isinstance(values, list):
            errors.append({"role": role, "reason": "retained_evidence_invalid", "detail": "expected a retained record list"})
    receipts = receipts if isinstance(receipts, list) else []
    captures = captures if isinstance(captures, list) else []
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or receipt.get("source_id") not in SOURCE_IDS:
            errors.append({"role": "authentication", "reason": "retained_evidence_invalid", "detail": "missing or unregistered source identity"})
    checked_forward = {**forward, "authentication_receipts": receipts, "captures": captures}
    # Invalid evidence must never produce a positive readiness decision.
    try:
        readiness = evidence.evaluate_c0_readiness(checked_forward, now=clock, authorization=authorization)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        errors.append({"role": "readiness", "reason": "retained_evidence_invalid", "detail": str(exc)})
        readiness = evidence.evaluate_c0_readiness(forward, now=clock)
    authority_records = {source_id: [] for source_id in SOURCE_IDS}
    for ref in forward.get("trusted_evidence_refs", []):
        if not isinstance(ref, Mapping) or ref.get("kind") != "source_permission":
            continue
        try:
            proof = evidence._retained_ref(ref, forward, trusted_kind="source_permission")
            source_id = proof.get("source_id")
            if source_id not in authority_records:
                raise ValueError("permission record has an unregistered source identity")
            contract = contract_map[source_id]
            locator = proof.get("approved_locator")
            scope = proof.get("scope", {})
            usable = (proof.get("evidence_class") == "authentic" and isinstance(scope, Mapping)
                      and locator in contract.get("approved_locators", [contract.get("approved_locator")])
                      and isinstance(locator, str) and locator.startswith("https://")
                      and evidence._clock(proof["effective_from"]) <= clock < evidence._clock(proof["effective_until"]))
            authority_records[source_id].append({
                "uri": ref.get("uri", ref.get("evidence_uri")), "sha256": ref["sha256"],
                "evidence_class": proof.get("evidence_class"), "approved_locator": locator,
                "scope": scope, "effective_from": proof.get("effective_from"),
                "effective_until": proof.get("effective_until"),
                "effective_authority": usable,
                "reason": None if usable else "authority_class_locator_or_applicable_dates_unverified",
            })
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append({"role": "permission", "reason": "retained_evidence_invalid", "detail": str(exc)})
    acquisition_approval = False
    approval_records = []
    if isinstance(authorization, Mapping):
        try:
            recorded = evidence._clock(authorization["recorded_at"])
            if (authorization.get("schema") != "native_c0_authorization_v1"
                    or authorization.get("protocol_id") != evidence.PROTOCOL_ID
                    or not recorded < evidence.DEADLINE or recorded > clock
                    or authorization.get("source_contract_sha256") != {key: evidence._sha(value) for key, value in contract_map.items()}):
                raise ValueError("acquisition authorization identity/deadline/source binding")
            for ref in authorization["approval_evidence_refs"]:
                proof = evidence._retained_ref(ref, forward, trusted_kind="acquisition_authority")
                approved = (proof.get("evidence_class") == "authentic"
                            and proof.get("protocol_id") == evidence.PROTOCOL_ID
                            and proof.get("activation_deadline") == evidence.DEADLINE.isoformat()
                            and proof.get("approved") is True and evidence._clock(proof["approved_at"]) <= recorded)
                approval_records.append({"sha256": ref["sha256"], "uri": ref.get("uri", ref.get("evidence_uri")),
                                         "approved": approved, "source_contract_sha256": authorization["source_contract_sha256"]})
            acquisition_approval = any(row["approved"] for row in approval_records)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append({"role": "acquisition_approval", "reason": "retained_evidence_invalid", "detail": str(exc)})

    sources = []
    for source_id in SOURCE_IDS:
        contract, requirement = contract_map[source_id], requirements[source_id]
        adjudications, authentic = [], []
        for receipt in receipts:
            if not isinstance(receipt, Mapping) or receipt.get("source_id") != source_id:
                continue
            reasons = evidence.validate_source_adjudication(receipt, protocol=checked_forward, captures=captures)
            if not reasons:
                if evidence._clock(receipt["adjudicated_at"]) > clock:
                    reasons = ["adjudication_not_available_at_inspection"]
            authority_refs = []
            permission_refs = receipt.get("permission_evidence_refs", [])
            if not isinstance(permission_refs, list):
                reasons = sorted(set([*reasons, "provider_authority_record_unverified"]))
                errors.append({"source_id": source_id, "role": "permission", "reason": "retained_evidence_invalid", "detail": "permission references must be a list"})
                permission_refs = []
            for ref in permission_refs:
                try:
                    proof = evidence._retained_ref(ref, forward, trusted_kind="source_permission")
                    if proof.get("evidence_class") != "authentic":
                        raise ValueError("permission record must be actual authentic authority, not a candidate or synthetic receipt")
                    if (proof.get("source_id") != source_id or proof.get("approved_locator") != receipt.get("approved_locator")):
                        raise ValueError("permission record source/locator binding mismatch")
                    authority_refs.append({"uri": ref.get("uri", ref.get("evidence_uri")), "sha256": ref["sha256"],
                                           "scope": proof.get("scope"), "effective_from": proof.get("effective_from"),
                                           "effective_until": proof.get("effective_until")})
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    reasons = sorted(set([*reasons, "provider_authority_record_unverified"]))
                    errors.append({"source_id": source_id, "role": "permission", "reason": "retained_evidence_invalid", "detail": str(exc)})
            row = {"sha256": evidence._sha(receipt), "reasons": reasons, "authority_refs": authority_refs,
                   "evidence_class": receipt.get("evidence_class"), "capture_refs": receipt.get("capture_refs", [])}
            adjudications.append(row)
            if not reasons and authority_refs:
                authentic.append(receipt)
        authorities = authority_records[source_id]
        effective = [row for row in authorities if row["effective_authority"]]
        scope_records = {
            "provider_permission": any(row["scope"].get("request") is True and row["scope"].get("use") is True for row in effective),
            "retention": any(row["scope"].get("retention") is True for row in effective),
            "entitlement": any(row["scope"].get("entitlement") is True for row in effective),
            "quota": any(row["scope"].get("quota") is True for row in effective),
            "approved_locator": bool(effective),
            "participant_binding": bool(authentic),
            "retained_capture": bool(authentic),
        }
        missing = [key for key, supported in scope_records.items() if not supported]
        approved = acquisition_approval and all(scope_records.values())
        if not approved:
            missing.append("sealed_acquisition_approval")
        locator = requirement.get("locator", {})
        sources.append({"source_id": source_id, "purpose": requirement["purpose"],
            "source_contract_sha256": evidence._sha(contract), "original_source_contract_sha256": evidence._sha(originals[source_id]),
            "activation_matrix_entry_sha256": evidence._sha(requirement),
            "candidate_locator": locator.get("configured_candidate", locator.get("configured_scraper", locator.get("configured_client"))),
            "approved_locator": effective[0]["approved_locator"] if effective else None,
            "effective_authentication": "authenticated" if authentic else "unverified",
            "acquisition_approved": approved, "probe_authorized": False,
            "requests_this_inspection": 0, "missing_records": missing,
            "authority_records": authorities, "record_resolution": scope_records,
            "requirements": {"authority": requirement["authorization_retention_entitlement"], "quota": requirement["quota"],
                             "locator": locator, "binding_and_capture": requirement["binding_and_capture_unmet"]},
            "adjudications": adjudications})
    if errors or any(source["missing_records"] for source in sources):
        readiness["status"] = "EXPIRED_UNSTARTED" if clock >= evidence.DEADLINE else "BLOCKED_EXTERNAL_PREREQUISITES"
        readiness["reasons"] = sorted(set([*readiness["reasons"], "source_authentication_unverified", "G0_BLOCKED"]))
        # Acquisition approval is independently reported, not inferred from G0.
        # This inspector never authorizes a probe, even when authority is retained.
    mechanics = _local_mechanics()
    result = {**readiness, "schema": "c0_input_aligned_forward_readiness_v1", "run_id": study.get("run_id"),
              "g0_status": "PASS" if readiness["status"] == "READY_AUTHORIZED" else "G0_BLOCKED",
              "external_requests": 0, "acquisition_attempts": 0, "real_fit_count": 0,
              "evidence_inputs": inputs, "retained_evidence_refs": retained_refs, "evidence_errors": errors,
              "acquisition_approval_records": approval_records,
              "sources": sources, "local_mechanics": mechanics,
              "local_mechanics_sha256": evidence._sha(mechanics),
              "cross_source_prerequisites": matrix["cross_source_activation_prerequisites"],
              "common_capture_prerequisites": matrix["common_capture_prerequisites"],
              "probe_limits_not_authorization": matrix["probe_limits"],
              "boundary": "No source/network/DB acquisition. Prior E1–E4 evidence not repeated. Synthetic mechanics do not grant permission, prove authentic construction or admit G0."}
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    for index, source in enumerate(sources, 1):
        evidence._atomic_bytes(destination / "sources" / f"{index:02d}.json", evidence.canonical_bytes(source))
    evidence._atomic_bytes(destination / "mechanics.json", evidence.canonical_bytes(mechanics))
    evidence._atomic_bytes(destination / "readiness.json", evidence.canonical_bytes(result))
    return result
