"""Append-only persistence for local forecast evidence envelopes."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from src.analysis.quote_provenance import extract_winner_quote
from src.analysis.roster_provenance import resolve_roster_evidence


from betting_app.models.prediction import (
    ForecastEvidenceSnapshot,
    OpportunityEvent,
    SourceCapture,
)

_INFORMATION_STATUSES = {
    "local_fixture",
    "historical_scenario",
    "historical_reconstruction",
    "recorded",
}
_ENVELOPE_OBJECTS = ("model", "input", "forecast", "roster", "schedule")

def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _canonical_payload(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        # Ensure there are no unserializable objects and hash exactly the stored UTF-8 bytes.
        encoded = text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("payload must contain finite JSON values") from exc
    _validate_payload(payload)
    return text, hashlib.sha256(encoded).hexdigest()


def _is_aware_clock(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if payload.get("information_status") not in _INFORMATION_STATUSES:
        raise ValueError("unsupported information_status")
    for field in _ENVELOPE_OBJECTS:
        if not isinstance(payload.get(field), dict):
            raise ValueError(f"{field} must be an object")
    if not isinstance(payload.get("quotes"), list) or not isinstance(payload.get("sources"), list):
        raise ValueError("quotes and sources must be lists")

    forecast = payload["forecast"]
    if not _is_aware_clock(forecast.get("generated_at")):
        raise ValueError("forecast.generated_at must be timezone-aware")
    sources: dict[str, dict[str, Any]] = {}
    for source in payload["sources"]:
        if not isinstance(source, dict):
            raise ValueError("each source must be an object")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise ValueError("source_id values must be unique nonempty strings")
        content = source.get("content_base64")
        digest = source.get("sha256")
        if not isinstance(content, str) or not isinstance(digest, str):
            raise ValueError("sources require content_base64 and sha256")
        try:
            exact_bytes = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("source content_base64 is invalid") from exc
        if base64.b64encode(exact_bytes).decode("ascii") != content:
            raise ValueError("source content_base64 must use canonical encoding")
        if hashlib.sha256(exact_bytes).hexdigest() != digest:
            raise ValueError("source sha256 does not match exact content bytes")
        if not isinstance(source.get("evidence_uri"), str) or not source["evidence_uri"]:
            raise ValueError("source evidence_uri must be a nonempty string")
        if not _is_aware_clock(source.get("retrieved_at")):
            raise ValueError("source retrieved_at must be timezone-aware")
        sources[source_id] = source

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError("payload object keys must be strings")
                if key.endswith("_at") and nested is not None and not _is_aware_clock(nested):
                    raise ValueError(f"{key} must be a timezone-aware clock")
                if key == "source_ids":
                    if not isinstance(nested, list) or any(
                        not isinstance(source_id, str) or source_id not in sources
                        for source_id in nested
                    ):
                        raise ValueError("source_ids must refer to sources in this envelope")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("payload numbers must be finite")

    visit(payload)


def append_forecast_evidence(
    session: Session, *, evidence_key: str, payload: dict[str, Any]
) -> ForecastEvidenceSnapshot:
    """Append evidence within the caller's transaction, idempotent by key and bytes."""
    if not isinstance(evidence_key, str) or not evidence_key.strip():
        raise ValueError("evidence_key must be a nonempty string")
    if len(evidence_key) > 200:
        raise ValueError("evidence_key exceeds 200 characters")
    payload_json, content_sha256 = _canonical_payload(payload)
    table = ForecastEvidenceSnapshot.__table__
    statement = (
        insert(table)
        .values(
            evidence_key=evidence_key,
            content_sha256=content_sha256,
            payload_json=payload_json,
        )
        .on_conflict_do_nothing(index_elements=[table.c.evidence_key])
    )
    session.execute(statement)
    row = session.execute(
        select(ForecastEvidenceSnapshot).where(
            ForecastEvidenceSnapshot.evidence_key == evidence_key
        )
    ).scalar_one()
    if row.content_sha256 != content_sha256 or row.payload_json != payload_json:
        raise ValueError("evidence_key already exists with different payload bytes")
    return row


def load_forecast_evidence(session: Session, evidence_key: str) -> dict[str, Any] | None:
    """Load an independent JSON value after verifying stored bytes and evidence links."""
    row = session.execute(
        select(ForecastEvidenceSnapshot).where(
            ForecastEvidenceSnapshot.evidence_key == evidence_key
        )
    ).scalar_one_or_none()
    if row is None:
        return None
    encoded = row.payload_json.encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != row.content_sha256:
        raise ValueError("stored forecast evidence content hash mismatch")
    try:
        payload = json.loads(row.payload_json, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("stored forecast evidence is not valid JSON") from exc
    canonical, digest = _canonical_payload(payload)
    if canonical != row.payload_json or digest != row.content_sha256:
        raise ValueError("stored forecast evidence is not canonical or valid")
    return payload

_SENSITIVE_METADATA_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|token|secret|password|credential|session|bearer)",
    re.IGNORECASE,
)
_UTC = timezone.utc


def _aware_datetime(value: Any, field: str, *, nullable: bool = True) -> datetime | None:
    if value is None and nullable:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
        except ValueError as exc:
            raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return value.astimezone(_UTC)


def _safe_url_parts(value: str) -> dict[str, Any]:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https", "fixture"} or not parts.netloc:
        return {}
    origin = f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}"
    safe_params = {
        key: val
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not _SENSITIVE_METADATA_KEY.search(key)
    }
    return {"origin": origin, "path": parts.path or "/", "params": safe_params}


def _sanitize_metadata_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for name, nested in value.items():
            if not isinstance(name, str):
                continue
            if _SENSITIVE_METADATA_KEY.search(name):
                continue
            if name.casefold() in {"url", "request_url", "full_url"} and isinstance(nested, str):
                sanitized[name] = _sanitize_metadata_value(_safe_url_parts(nested))
            else:
                sanitized[name] = _sanitize_metadata_value(nested, key=name)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_metadata_value(item, key=key) for item in value]
    if isinstance(value, str) and value.startswith(("http://", "https://", "fixture://")):
        safe = _safe_url_parts(value)
        if not safe:
            return None
        if key.casefold() == "origin":
            return safe["origin"]
        query = urlencode(safe["params"])
        return safe["origin"] + safe["path"] + (f"?{query}" if query else "")
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    raise ValueError("metadata must contain JSON-compatible values")


def _canonical_metadata(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    sanitized = _sanitize_metadata_value(value)
    try:
        return json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{field} must contain finite JSON values") from exc


def _safe_evidence_uri(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence_uri must be a nonempty credential-free locator")
    parts = urlsplit(value)
    if parts.scheme not in {"fixture", "http", "https", "file", "sha256"}:
        raise ValueError("evidence_uri must use an explicit fixture/http/https/file/sha256 scheme")
    if not parts.netloc and not parts.path:
        raise ValueError("evidence_uri must identify a source")
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path or "/", "", ""))


def _body_path(body_store: Path, digest: str) -> Path:
    return body_store / digest[:2] / f"{digest}.bin"


def _verify_body_path(path: Path, expected_digest: str) -> bytes:
    try:
        body = path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("captured source body file is missing") from exc
    if hashlib.sha256(body).hexdigest() != expected_digest:
        raise ValueError("captured source body file hash mismatch")
    return body


def _persist_exact_body(body_store: Path, body: bytes, digest: str) -> str:
    root = Path(body_store)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    directory = root / digest[:2]
    directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(directory, 0o700)
    target = _body_path(root, digest)
    if target.is_symlink():
        raise ValueError("content-addressed source body path must not be a symlink")
    if target.exists():
        if _verify_body_path(target, digest) != body:
            raise ValueError("content-addressed source body collision")
        return f"sha256://{digest}"

    descriptor, temporary_name = tempfile.mkstemp(prefix=".capture-", dir=directory)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _verify_body_path(target, digest) != body:
                raise ValueError("content-addressed source body collision")
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return f"sha256://{digest}"


def _stored_capture_values(row: SourceCapture) -> dict[str, Any]:
    return {
        "source_id": row.source_id,
        "evidence_uri": row.evidence_uri,
        "sha256": row.sha256,
        "content_base64": row.content_base64,
        "body_uri": row.body_uri,
        "body_representation": row.body_representation,
        "content_encoding": row.content_encoding,
        "character_encoding": row.character_encoding,
        "request_started_at": row.request_started_at,
        "response_received_at": row.response_received_at,
        "retrieved_at": row.retrieved_at,
        "first_observed_at": row.first_observed_at,
        "observed_at": row.observed_at,
        "published_at": row.published_at,
        "source_effective_at": row.source_effective_at,
        "source_revision": row.source_revision,
        "http_status": row.http_status,
        "source_class": row.source_class,
        "availability_basis": row.availability_basis,
        "attempt_status": row.attempt_status,
        "failure_reason": row.failure_reason,
        "request_metadata_json": row.request_metadata_json,
        "response_metadata_json": row.response_metadata_json,
        "source_authentication_status": row.source_authentication_status,
    }


def _lock_evidence_identity(session: Session, *identity: str | None) -> None:
    material = json.dumps(identity, separators=(",", ":")).encode("utf-8")
    key = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)
    session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})


def capture_source_attempt(
    session: Session,
    *,
    capture_id: str,
    source_id: str,
    evidence_uri: str,
    raw_body: bytes | None,
    body_store: Path,
    request_started_at: datetime | None,
    response_received_at: datetime | None,
    first_observed_at: datetime | None,
    published_at: datetime | None = None,
    source_effective_at: datetime | None = None,
    source_revision: str | None = None,
    http_status: int | None = None,
    source_class: str,
    availability_basis: str,
    body_representation: str = "http_body",
    content_encoding: str | None = None,
    character_encoding: str | None = None,
    request_metadata: Mapping[str, Any],
    response_metadata: Mapping[str, Any],
    failure_reason: str | None = None,
    observed_at: datetime | None = None,
) -> SourceCapture:
    """Persist one exact response attempt; bytes are durable and verified before its row."""
    for field, value, limit in (
        ("capture_id", capture_id, 200),
        ("source_id", source_id, 200),
        ("source_class", source_class, 50),
        ("availability_basis", availability_basis, 100),
        ("body_representation", body_representation, 40),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{field} must be a nonempty string of at most {limit} characters")
    if body_representation not in {"http_body", "rendered_dom"}:
        raise ValueError("body_representation must be http_body or rendered_dom")
    if raw_body is not None and not isinstance(raw_body, bytes):
        raise ValueError("raw_body must be exact bytes or None")
    if http_status is not None and (isinstance(http_status, bool) or not isinstance(http_status, int) or not 100 <= http_status <= 599):
        raise ValueError("http_status must be an HTTP status or None")

    request_started = _aware_datetime(request_started_at, "request_started_at")
    response_received = _aware_datetime(response_received_at, "response_received_at")
    first_observed = _aware_datetime(first_observed_at, "first_observed_at")
    received_observed = _aware_datetime(observed_at, "observed_at")
    if received_observed is None:
        received_observed = response_received
    if request_started is not None and response_received is not None and request_started > response_received:
        raise ValueError("request_started_at cannot follow response_received_at clock")
    if request_started is not None and received_observed is not None and request_started > received_observed:
        raise ValueError("request_started_at cannot follow observed_at clock")
    if response_received is not None and received_observed is not None and response_received > received_observed:
        raise ValueError("response_received_at cannot follow observed_at clock")
    if first_observed is not None and received_observed is not None and first_observed > received_observed:
        raise ValueError("first_observed_at cannot follow observed_at clock")

    digest: str | None = None
    encoded_body: str | None = None
    body_uri: str | None = None
    if raw_body is not None:
        digest = hashlib.sha256(raw_body).hexdigest()
        encoded_body = base64.b64encode(raw_body).decode("ascii")
        body_uri = _persist_exact_body(Path(body_store), raw_body, digest)
        if hashlib.sha256(_verify_body_path(_body_path(Path(body_store), digest), digest)).hexdigest() != digest:
            raise ValueError("durable source body failed post-write hash verification")
        revision_identity = (
            SourceCapture.source_revision == source_revision
            if source_revision is not None
            else SourceCapture.source_revision.is_(None)
        )
        _lock_evidence_identity(session, "source_revision", source_id, digest, source_revision)
        prior_first_observed = session.execute(
            select(SourceCapture.first_observed_at)
            .where(
                SourceCapture.source_id == source_id,
                SourceCapture.sha256 == digest,
                revision_identity,
                SourceCapture.first_observed_at.is_not(None),
            )
            .order_by(SourceCapture.first_observed_at)
            .limit(1)
        ).scalar_one_or_none()
        if prior_first_observed is not None:
            if first_observed is not None and first_observed < prior_first_observed:
                raise ValueError("first_observed_at cannot precede the earliest persisted exact-revision observation")
            first_observed = prior_first_observed
        else:
            if first_observed is None:
                first_observed = response_received or received_observed
            if (
                first_observed is not None
                and response_received is not None
                and first_observed < response_received
            ):
                raise ValueError(
                    "first_observed_at cannot predate the first response without a persisted exact-revision observation"
                )

    safe_uri = _safe_evidence_uri(evidence_uri)
    request_json = _canonical_metadata(request_metadata, "request_metadata")
    response_json = _canonical_metadata(response_metadata, "response_metadata")
    failure = failure_reason.strip() if isinstance(failure_reason, str) and failure_reason.strip() else None
    content_type = ""
    try:
        content_type = str(json.loads(response_json).get("content_type", "")).casefold()
    except (TypeError, ValueError):
        pass
    if failure is None and http_status is not None and not 200 <= http_status < 300:
        failure = "http_error"
    if failure is None and raw_body is None:
        failure = "body_absent"
    if failure is None and raw_body == b"":
        failure = "empty_body"
    if failure is None and raw_body is not None and "json" in content_type:
        try:
            json.loads(raw_body, parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
            failure = "malformed_json"
    authentication_status = "local_fixture" if source_class == "local_fixture" else "unverified"
    values = {
        "source_id": source_id,
        "evidence_uri": safe_uri,
        "sha256": digest,
        "content_base64": encoded_body,
        "body_uri": body_uri,
        "body_representation": body_representation,
        "content_encoding": content_encoding,
        "character_encoding": character_encoding,
        "request_started_at": request_started,
        "response_received_at": response_received,
        "retrieved_at": response_received,
        "first_observed_at": first_observed,
        "observed_at": received_observed,
        "published_at": _aware_datetime(published_at, "published_at"),
        "source_effective_at": _aware_datetime(source_effective_at, "source_effective_at"),
        "source_revision": source_revision,
        "http_status": http_status,
        "source_class": source_class,
        "availability_basis": availability_basis,
        "attempt_status": "failed" if failure else "captured",
        "failure_reason": failure,
        "request_metadata_json": request_json,
        "response_metadata_json": response_json,
        "source_authentication_status": authentication_status,
    }
    table = SourceCapture.__table__
    session.execute(
        insert(table)
        .values(capture_id=capture_id, **values)
        .on_conflict_do_nothing(index_elements=[table.c.capture_id])
    )
    row = session.execute(
        select(SourceCapture).where(SourceCapture.capture_id == capture_id)
    ).scalar_one()
    if any(getattr(row, field) != value for field, value in values.items()):
        raise ValueError("capture_id already exists with conflicting attempt metadata or body bytes")
    return row


def load_source_capture(
    session: Session, capture_id: str, *, body_store: Path
) -> dict[str, Any] | None:
    """Load capture metadata and return exact body bytes after independent disk/base64 hash checks."""
    row = session.execute(
        select(SourceCapture).where(SourceCapture.capture_id == capture_id)
    ).scalar_one_or_none()
    if row is None:
        return None
    body: bytes | None = None
    if row.sha256 is not None:
        if row.body_uri != f"sha256://{row.sha256}" or row.content_base64 is None:
            raise ValueError("stored source capture body reference is inconsistent")
        body = _verify_body_path(_body_path(Path(body_store), row.sha256), row.sha256)
        try:
            base64_body = base64.b64decode(row.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("stored source capture base64 is invalid") from exc
        if base64.b64encode(base64_body).decode("ascii") != row.content_base64 or base64_body != body:
            raise ValueError("stored source capture base64 does not match exact body bytes")
    elif row.content_base64 is not None or row.body_uri is not None:
        raise ValueError("stored source capture has a body reference without a hash")
    try:
        request_metadata = json.loads(row.request_metadata_json, parse_constant=_reject_json_constant)
        response_metadata = json.loads(row.response_metadata_json, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("stored source capture metadata is invalid JSON") from exc
    if _canonical_metadata(request_metadata, "request_metadata") != row.request_metadata_json:
        raise ValueError("stored source capture request metadata is not canonical")
    if _canonical_metadata(response_metadata, "response_metadata") != row.response_metadata_json:
        raise ValueError("stored source capture response metadata is not canonical")
    result = {field: getattr(row, field) for field in (
        "id", "capture_id", "source_id", "evidence_uri", "sha256", "content_base64",
        "body_uri", "body_representation", "content_encoding", "character_encoding",
        "request_started_at", "response_received_at", "retrieved_at", "first_observed_at",
        "observed_at", "published_at", "source_effective_at", "source_revision",
        "http_status", "source_class", "availability_basis", "attempt_status",
        "failure_reason", "source_authentication_status", "recorded_at",
    )}
    result["request_metadata"] = request_metadata
    result["response_metadata"] = response_metadata
    result["body_bytes"] = body
    return result


def audit_source_capture_store(
    session: Session, *, body_store: Path
) -> dict[str, list[str]]:
    """Identify orphan, missing, and corrupt files without repairing immutable evidence."""
    root = Path(body_store)
    rows = session.execute(select(SourceCapture)).scalars().all()
    referenced: set[Path] = set()
    missing: list[str] = []
    corrupt: list[str] = []
    for row in rows:
        if row.sha256 is None:
            continue
        path = _body_path(root, row.sha256)
        referenced.add(path)
        if not path.is_file():
            missing.append(row.capture_id)
        else:
            try:
                _verify_body_path(path, row.sha256)
            except ValueError:
                corrupt.append(row.capture_id)
    files = set(root.glob("*/*.bin")) if root.exists() else set()
    if root.exists():
        files.update(root.glob("*/.capture-*"))
    orphans = sorted(path.relative_to(root).as_posix() for path in files - referenced)
    return {
        "orphan_body_uris": orphans,
        "missing_body_capture_ids": sorted(missing),
        "corrupt_body_capture_ids": sorted(corrupt),
    }


_OPPORTUNITY_EVENT_TYPES = {
    "discovery",
    "schedule_revision",
    "origin_pending",
    "primary_decision",
    "correction",
    "outcome_release",
}
_OPPORTUNITY_REQUIRED_PAYLOAD_FIELDS = {
    "status",
    "rejection_reasons",
    "clocks",
    "model_availability",
    "feature_availability",
    "inference_eligible",
    "forward_admitted",
    "information_status",
    "source_authentication_status",
    "source_capture_ids",
}


def _canonical_event_payload(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("opportunity payload must be an object")
    missing = _OPPORTUNITY_REQUIRED_PAYLOAD_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"opportunity payload is missing required fields: {sorted(missing)}")
    if not isinstance(payload["status"], str) or not payload["status"]:
        raise ValueError("opportunity payload status must be a nonempty string")
    reasons = payload["rejection_reasons"]
    if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
        raise ValueError("rejection_reasons must be a list of nonempty strings")
    for field in ("clocks", "model_availability", "feature_availability"):
        if not isinstance(payload[field], dict):
            raise ValueError(f"opportunity payload {field} must be an object")
    for field in ("inference_eligible", "forward_admitted"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"opportunity payload {field} must be a boolean")
    if payload["information_status"] not in _INFORMATION_STATUSES:
        raise ValueError("unsupported opportunity information_status")
    if payload["source_authentication_status"] not in {
        "local_fixture", "unverified", "authenticated",
    }:
        raise ValueError("unsupported source_authentication_status")
    capture_ids = payload["source_capture_ids"]
    if not isinstance(capture_ids, list) or any(not isinstance(item, str) or not item for item in capture_ids):
        raise ValueError("source_capture_ids must be a list of nonempty strings")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("opportunity object keys must be strings")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("opportunity payload numbers must be finite")

    visit(payload)
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("opportunity payload must contain finite JSON values") from exc
    return encoded.decode("utf-8"), hashlib.sha256(encoded).hexdigest()


def append_opportunity_event(
    session: Session,
    *,
    event_key: str,
    protocol_id: str,
    provider_id: str,
    fixture_id: str,
    origin_at: datetime | None,
    event_type: str,
    primary_decision: bool,
    parent_event_key: str | None,
    payload: dict[str, Any],
    forecast_evidence_key: str | None = None,
) -> OpportunityEvent:
    """Append a canonical, reason-complete event in the caller's transaction."""
    for field, value, limit in (
        ("event_key", event_key, 200),
        ("protocol_id", protocol_id, 120),
        ("provider_id", provider_id, 120),
        ("fixture_id", fixture_id, 200),
    ):
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            raise ValueError(f"{field} must be a nonempty string of at most {limit} characters")
    if event_type not in _OPPORTUNITY_EVENT_TYPES:
        raise ValueError(f"unsupported opportunity event_type: {event_type!r}")
    if not isinstance(primary_decision, bool):
        raise ValueError("primary_decision must be a boolean")
    origin = _aware_datetime(origin_at, "origin_at")
    if primary_decision and origin is None:
        raise ValueError("primary_decision requires a non-null origin_at")
    if event_type in {"schedule_revision", "correction"} and parent_event_key is None:
        raise ValueError("schedule revision and correction events require an existing parent")
    if parent_event_key is not None:
        parent = session.execute(
            select(OpportunityEvent).where(OpportunityEvent.event_key == parent_event_key)
        ).scalar_one_or_none()
        if parent is None:
            raise ValueError("parent_event_key must refer to an existing immutable event")
        if (parent.protocol_id, parent.provider_id, parent.fixture_id) != (
            protocol_id, provider_id, fixture_id
        ):
            raise ValueError("parent event must have the same protocol/provider/fixture identity")
    if forecast_evidence_key is not None:
        stored_forecast = load_forecast_evidence(session, forecast_evidence_key)
        if stored_forecast is None:
            raise ValueError("forecast_evidence_key must refer to an existing genuine forecast envelope")

    payload_json, payload_sha256 = _canonical_event_payload(payload)
    if payload["forward_admitted"] and (
        not payload["inference_eligible"]
        or payload["source_authentication_status"] != "authenticated"
        or payload["status"] != "forecasted"
        or forecast_evidence_key is None
        or not payload["source_capture_ids"]
    ):
        raise ValueError(
            "forward_admitted requires eligible inputs, authenticated sources, and an existing forecast envelope"
        )
    capture_ids = payload["source_capture_ids"]
    source_rows = []
    if capture_ids:
        source_rows = session.execute(
            select(SourceCapture).where(SourceCapture.capture_id.in_(capture_ids))
        ).scalars().all()
        if {row.capture_id for row in source_rows} != set(capture_ids):
            raise ValueError("opportunity source_capture_ids must refer to stored immutable captures")
        statuses = {row.source_authentication_status for row in source_rows}
        linked_authentication = (
            "unverified"
            if "unverified" in statuses
            else "local_fixture"
            if "local_fixture" in statuses
            else "authenticated"
            if statuses == {"authenticated"}
            else "unverified"
        )
        if protocol_id == "native-c0-market-study-v1" and payload.get("c0_protocol") is not None:
            from betting_app.services.c0_shadow_readiness import validate_source_adjudication
            c0_protocol = payload.get("c0_protocol", {})
            captures = [
                {"capture_id": row.capture_id, "recorded_at": row.recorded_at,
                 "request_metadata": json.loads(row.request_metadata_json),
                 **_stored_capture_values(row)} for row in source_rows
            ]
            checked = {
                ref.get("capture_id")
                for receipt in c0_protocol.get("authentication_receipts", [])
                if not validate_source_adjudication(receipt, protocol=c0_protocol, captures=captures)
                for ref in receipt.get("capture_refs", [])
            }
            linked_authentication = (
                "authenticated" if set(capture_ids).issubset(checked)
                else "local_fixture" if any(row.source_class == "local_fixture" for row in source_rows)
                else "unverified"
            )
        if payload["source_authentication_status"] != linked_authentication:
            raise ValueError("opportunity source authentication must match linked capture provenance")
    if forecast_evidence_key is not None:
        details = payload
        roster_results = details.get("roster_results", {})
        linked_candidate = {
            "protocol_id": protocol_id,
            "provider_id": provider_id,
            "fixture_id": fixture_id,
            "fixture": details,
            "origin_at": origin,
            "scheduled_start_at": _aware_datetime(payload["clocks"].get("scheduled_start_at"), "scheduled_start_at"),
            "known_start_bound": _aware_datetime(payload["clocks"].get("known_start_bound"), "known_start_bound"),
            "rosters": {key: value.get("announcement") for key, value in roster_results.items()},
            "source_captures": [
                {"capture_id": row.capture_id, **_stored_capture_values(row)} for row in source_rows
            ],
            "protocol": payload.get("c0_protocol"),
            "authorization": payload.get("c0_authorization"),
        }
        recorded_at = session.execute(
            select(ForecastEvidenceSnapshot.recorded_at).where(
                ForecastEvidenceSnapshot.evidence_key == forecast_evidence_key
            )
        ).scalar_one()
        link_reasons = _forecast_link_reasons(stored_forecast, linked_candidate, recorded_at=recorded_at)
        if link_reasons:
            raise ValueError("forecast envelope does not bind this opportunity: " + ", ".join(link_reasons))
        if protocol_id == "native-c0-market-study-v1" and payload["forward_admitted"]:
            from betting_app.services.c0_shadow_readiness import (
                _encoded_from_envelope, evaluate_c0_readiness, validate_dependency_receipt,
            )
            evaluation = _aware_datetime(payload["clocks"].get("decision_at"), "decision_at", nullable=False)
            if (evaluation < _aware_datetime(recorded_at, "recorded_at", nullable=False)
                    or evaluation > origin + timedelta(seconds=60)
                    or evaluation >= linked_candidate["known_start_bound"]
                    or evaluation > datetime.now(UTC)):
                raise ValueError("C0 evaluation must follow recording within the actual prematch completion window")
            checked_readiness = evaluate_c0_readiness(
                payload.get("c0_protocol", {}), now=evaluation,
                authorization=payload.get("c0_authorization"),
            )
            dependency_reasons = validate_dependency_receipt(
                stored_forecast["input"].get("dependency_receipt"),
                protocol=payload.get("c0_protocol", {}),
                encoded_snapshot=_encoded_from_envelope(stored_forecast),
                sources=stored_forecast["sources"],
            )
            if checked_readiness["status"] != "READY_AUTHORIZED" or dependency_reasons:
                raise ValueError("C0 forward admission requires checked authority/native source evidence")
    values = {
        "protocol_id": protocol_id,
        "provider_id": provider_id,
        "fixture_id": fixture_id,
        "origin_at": origin,
        "event_type": event_type,
        "primary_decision": primary_decision,
        "parent_event_key": parent_event_key,
        "payload_json": payload_json,
        "payload_sha256": payload_sha256,
        "forecast_evidence_key": forecast_evidence_key,
    }
    table = OpportunityEvent.__table__
    session.execute(
        insert(table)
        .values(event_key=event_key, **values)
        .on_conflict_do_nothing(index_elements=[table.c.event_key])
    )
    row = session.execute(
        select(OpportunityEvent).where(OpportunityEvent.event_key == event_key)
    ).scalar_one()
    if any(getattr(row, field) != value for field, value in values.items()):
        raise ValueError("event_key already exists with conflicting opportunity bytes or metadata")
    return row


def load_opportunity_events(
    session: Session,
    *,
    protocol_id: str,
    provider_id: str | None = None,
    fixture_id: str | None = None,
) -> list[OpportunityEvent]:
    """Return verified immutable events in actual DB recording order."""
    statement = select(OpportunityEvent).where(OpportunityEvent.protocol_id == protocol_id)
    if provider_id is not None:
        statement = statement.where(OpportunityEvent.provider_id == provider_id)
    if fixture_id is not None:
        statement = statement.where(OpportunityEvent.fixture_id == fixture_id)
    statement = statement.order_by(OpportunityEvent.recorded_at, OpportunityEvent.id)
    rows = session.execute(statement).scalars().all()
    for row in rows:
        canonical, digest = _canonical_event_payload(row.payload)
        if canonical != row.payload_json or digest != row.payload_sha256:
            raise ValueError("stored opportunity event payload is not canonical or hash-verified")
    return list(rows)


def _iso_clock(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _capture_availability(capture: Mapping[str, Any]) -> datetime | None:
    actual = [
        capture[name] for name in ("response_received_at", "first_observed_at", "observed_at")
        if capture.get(name) is not None
    ]
    if not actual:
        return None
    declared = [
        capture[name] for name in ("published_at", "source_effective_at")
        if capture.get(name) is not None
    ]
    return max(actual + declared)


def _read_capture_json(
    session: Session, capture_id: str, body_store: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = load_source_capture(session, capture_id, body_store=body_store)
    if capture is None:
        raise ValueError(f"referenced source capture does not exist: {capture_id}")
    if capture["attempt_status"] != "captured":
        raise ValueError(f"referenced source capture failed: {capture_id}")
    if capture["body_bytes"] is None:
        raise ValueError(f"referenced source capture has no body: {capture_id}")
    try:
        value = json.loads(capture["body_bytes"], parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"referenced source capture is not valid JSON: {capture_id}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"referenced source capture JSON root must be an object: {capture_id}")
    return capture, value


def _event_key(batch_id: str, fixture_id: str, event_type: str, identity: str) -> str:
    encoded = "\0".join((batch_id, fixture_id, event_type, identity)).encode("utf-8")
    return f"plan09:{hashlib.sha256(encoded).hexdigest()}"


def _append_batch_event(
    session: Session,
    *,
    protocol_id: str,
    provider_id: str,
    batch_id: str,
    fixture_id: str,
    origin_at: datetime | None,
    event_type: str,
    primary_decision: bool,
    parent_event_key: str | None,
    status: str,
    rejection_reasons: Sequence[str],
    clocks: dict[str, Any],
    model_availability: dict[str, Any],
    feature_availability: dict[str, Any],
    information_status: str,
    source_authentication_status: str,
    source_capture_ids: Sequence[str],
    details: Mapping[str, Any],
    inference_eligible: bool = False,
    forward_admitted: bool = False,
    forecast_evidence_key: str | None = None,
    identity: str = "",
) -> OpportunityEvent:
    payload = {
        "status": status,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "clocks": clocks,
        "model_availability": model_availability,
        "feature_availability": feature_availability,
        "inference_eligible": inference_eligible,
        "forward_admitted": forward_admitted,
        "information_status": information_status,
        "source_authentication_status": source_authentication_status,
        "source_capture_ids": list(dict.fromkeys(source_capture_ids)),
        **dict(details),
    }
    return append_opportunity_event(
        session,
        event_key=_event_key(batch_id, fixture_id, event_type, identity),
        protocol_id=protocol_id,
        provider_id=provider_id,
        fixture_id=fixture_id,
        origin_at=origin_at,
        event_type=event_type,
        primary_decision=primary_decision,
        parent_event_key=parent_event_key,
        payload=payload,
        forecast_evidence_key=forecast_evidence_key,
    )


def _capture_source_auth_status(captures: Sequence[Mapping[str, Any]]) -> str:
    statuses = {capture.get("source_authentication_status") for capture in captures}
    if "unverified" in statuses:
        return "unverified"
    if "local_fixture" in statuses:
        return "local_fixture"
    return "authenticated" if statuses == {"authenticated"} else "unverified"


def _decorate_provenance_records(
    records: Any, capture: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("captured roster evidence arrays must contain objects")
    availability = _capture_availability(capture)
    if availability is None:
        raise ValueError("captured provenance has no actual availability clock")
    observation = capture.get("observed_at") or capture.get("response_received_at") or capture.get("first_observed_at")
    decorated = []
    for item in records:
        record = dict(item)
        record["observed_at"] = _iso_clock(observation)
        if capture.get("evidence_uri"):
            record["evidence_uri"] = capture["evidence_uri"]
        record["captured_available_at"] = _iso_clock(availability)
        record["source_capture_id"] = capture["capture_id"]
        record["source_capture_sha256"] = capture["sha256"]
        decorated.append(record)
    return decorated




def _forecast_link_reasons(
    envelope: Mapping[str, Any], candidate: Mapping[str, Any], *, recorded_at: datetime
) -> list[str]:
    """Bind a stored forecast to this exact fixture, origin, roster and capture set."""
    if candidate.get("protocol_id") == "native-c0-market-study-v1":
        from betting_app.services.c0_shadow_readiness import c0_forecast_link_reasons
        return c0_forecast_link_reasons(envelope, candidate, recorded_at=recorded_at)
    reasons = []
    fixture = candidate["fixture"]
    forecast = envelope["forecast"]
    target = envelope["input"].get("target", {})
    schedule = envelope["schedule"]
    teams = [str(value) for value in fixture.get("team_ids", [])]
    if (
        len(teams) != 2
        or str(target.get("golgg_match_id")) != str(fixture.get("canonical_fixture_id"))
        or [str(target.get("team1_id")), str(target.get("team2_id"))] != teams
        or forecast.get("best_of") != fixture.get("best_of")
        or target.get("best_of") != fixture.get("best_of")
        or schedule.get("protocol_id") != candidate["protocol_id"]
        or schedule.get("provider_id") != candidate["provider_id"]
        or str(schedule.get("provider_fixture_id")) != str(candidate["fixture_id"])
        or str(schedule.get("canonical_fixture_id")) != str(fixture.get("canonical_fixture_id"))
    ):
        reasons.append("forecast_envelope_fixture_mismatch")
    origin = candidate["origin_at"]
    if origin is None or any(
        _aware_datetime(value, "forecast_origin") != origin
        for value in (
            forecast.get("simulated_origin_at"),
            target.get("effective_decision_at"),
            schedule.get("origin_at"),
        )
    ):
        reasons.append("forecast_envelope_origin_mismatch")
    if (
        _aware_datetime(schedule.get("scheduled_start_at"), "scheduled_start_at") != candidate["scheduled_start_at"]
        or _aware_datetime(schedule.get("known_start_bound"), "known_start_bound") != candidate["known_start_bound"]
    ):
        reasons.append("forecast_envelope_schedule_mismatch")
    generated = _aware_datetime(forecast["generated_at"], "generated_at", nullable=False)
    bound = candidate["known_start_bound"]
    if (
        origin is None or bound is None or generated < origin
        or generated > origin + timedelta(seconds=60)
        or generated >= bound or generated > datetime.now(_UTC)
        or recorded_at < generated or recorded_at > origin + timedelta(seconds=60)
        or recorded_at >= bound
    ):
        reasons.append("forecast_envelope_time_invalid")
    probabilities = (forecast.get("prob_a"), forecast.get("prob_b"))
    if (
        any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in probabilities)
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8)
    ):
        reasons.append("forecast_envelope_probability_invalid")
    for team, side in zip(teams, ("a", "b")):
        announcement = candidate["rosters"].get(team) or {}
        players = {str(value) for value in announcement.get("roster", [])}
        if len(players) != 5 or {str(value) for value in target.get(f"roster_{side}", [])} != players:
            reasons.append("forecast_envelope_roster_mismatch")
    expected_sources = {item["capture_id"]: item for item in candidate["source_captures"]}
    actual_sources = {item["source_id"]: item for item in envelope["sources"]}
    if set(actual_sources) != set(expected_sources):
        reasons.append("forecast_envelope_capture_mismatch")
    else:
        for capture_id, expected in expected_sources.items():
            actual = actual_sources[capture_id]
            metadata_match = all(
                actual.get(field) == expected[field]
                for field in ("sha256", "evidence_uri", "source_class", "source_authentication_status")
            )
            clocks_match = all(
                _aware_datetime(actual.get(field), field) == expected[field]
                for field in (
                    "request_started_at", "response_received_at", "retrieved_at",
                    "first_observed_at", "observed_at", "published_at", "source_effective_at",
                )
            )
            if not metadata_match or not clocks_match:
                reasons.append("forecast_envelope_capture_mismatch")
                break
    return sorted(set(reasons))


def orchestrate_local_opportunity_batch(
    session: Session,
    *,
    protocol_id: str,
    provider_id: str,
    batch_id: str,
    discovery_capture_ids: Sequence[str],
    body_store: Path,
    as_of: datetime,
    max_fixtures: int,
    predictor: Callable[[dict[str, Any]], str | None] | None = None,
    protocol: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a bounded captured fixture universe and append complete opportunity events.

    No network, operational prediction table, or speculative forecast is produced.
    `predictor`, when supplied, must return an already-persisted genuine forecast
    evidence key; only that immutable envelope may be linked to an opportunity.
    """
    as_of_clock = _aware_datetime(as_of, "as_of", nullable=False)
    is_c0 = protocol_id == "native-c0-market-study-v1"
    if protocol is not None and protocol.get("protocol_id") != protocol_id:
        raise ValueError("protocol context must match protocol_id")
    if is_c0 and (protocol is None or protocol.get("schema") != "c0_market_study_v1"):
        raise ValueError("native C0 orchestration requires its frozen protocol")
    if protocol is not None and not is_c0:
        raise ValueError("explicit protocol dispatch supports only native C0")
    if isinstance(max_fixtures, bool) or not isinstance(max_fixtures, int) or max_fixtures <= 0:
        raise ValueError("max_fixtures must be a positive integer")
    if not isinstance(discovery_capture_ids, Sequence) or isinstance(discovery_capture_ids, (str, bytes)):
        raise ValueError("discovery_capture_ids must be a bounded sequence")
    if not discovery_capture_ids:
        raise ValueError("at least one discovery capture is required")
    if not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 120:
        raise ValueError("batch_id must be a nonempty string of at most 120 characters")
    if not isinstance(protocol_id, str) or not protocol_id.strip() or len(protocol_id) > 120:
        raise ValueError("protocol_id must be a nonempty string of at most 120 characters")
    if not isinstance(provider_id, str) or not provider_id.strip() or len(provider_id) > 120:
        raise ValueError("provider_id must be a nonempty string of at most 120 characters")

    feed_entries: list[tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]] = []
    for discovery_capture_id in discovery_capture_ids:
        capture, body = _read_capture_json(session, discovery_capture_id, Path(body_store))
        fixtures = body.get("fixtures")
        if not isinstance(fixtures, list) or any(not isinstance(item, dict) for item in fixtures):
            raise ValueError("discovery capture requires a fixtures array of objects")
        for index, fixture in enumerate(fixtures):
            feed_entries.append((capture, body, index, fixture))
    if len(feed_entries) > max_fixtures:
        raise ValueError(
            f"discovery fixture count {len(feed_entries)} exceeds max_fixtures={max_fixtures}"
        )


    output_fixtures: list[dict[str, Any]] = []
    accounted_fixture_ids: list[str] = []
    pending_fixture_ids: list[str] = []
    primary_decision_fixture_ids: list[str] = []
    batch_events: list[OpportunityEvent] = []

    for discovery_capture, _discovery_body, row_index, fixture in feed_entries:
        raw_provider_id = fixture.get("provider_fixture_id")
        provider_fixture_id = (
            str(raw_provider_id).strip()
            if isinstance(raw_provider_id, (str, int)) and not isinstance(raw_provider_id, bool)
            and str(raw_provider_id).strip()
            else None
        )
        fixture_id = provider_fixture_id or (
            f"unknown:{discovery_capture['capture_id']}:{row_index}"
        )
        accounted_fixture_ids.append(fixture_id)
        linked_captures: dict[str, dict[str, Any]] = {
            discovery_capture["capture_id"]: discovery_capture
        }
        capture_ids: list[str] = [discovery_capture["capture_id"]]
        reasons: list[str] = []
        if provider_fixture_id is None:
            reasons.append("unknown_provider_id")
        if not fixture.get("canonical_fixture_id"):
            reasons.append("canonical_identity_unresolved")

        discovered_at = _capture_availability(discovery_capture)
        team_ids = fixture.get("team_ids")
        teams_valid = (
            isinstance(team_ids, list)
            and len(team_ids) == 2
            and all(isinstance(team, (str, int)) and str(team).strip() for team in team_ids)
            and str(team_ids[0]) != str(team_ids[1])
        )
        if not teams_valid:
            reasons.append("canonical_identity_unresolved")
            team_ids = []
        if fixture.get("best_of") not in (1, 3, 5):
            reasons.append("unsupported_best_of")

        discovery_event = _append_batch_event(
            session,
            protocol_id=protocol_id,
            provider_id=provider_id,
            batch_id=batch_id,
            fixture_id=fixture_id,
            origin_at=None,
            event_type="discovery",
            primary_decision=False,
            parent_event_key=None,
            status="discovered",
            rejection_reasons=reasons,
            clocks={
                "discovered_at": _iso_clock(discovered_at),
                "request_started_at": _iso_clock(discovery_capture["request_started_at"]),
                "response_received_at": _iso_clock(discovery_capture["response_received_at"]),
                "first_observed_at": _iso_clock(discovery_capture["first_observed_at"]),
                "observed_at": _iso_clock(discovery_capture["observed_at"]),
                "origin_at": None,
            },
            model_availability={"available": False, "reason": "not_attempted"},
            feature_availability={"available": False, "reason": "not_assessed"},
            information_status="local_fixture" if discovery_capture["source_class"] == "local_fixture" else "recorded",
            source_authentication_status=discovery_capture["source_authentication_status"],
            source_capture_ids=capture_ids,
            details={
                "provider_fixture_id": provider_fixture_id,
                "canonical_fixture_id": fixture.get("canonical_fixture_id"),
                "team_ids": team_ids,
                "best_of": fixture.get("best_of"),
            },
            identity=f"{discovery_capture['capture_id']}:{row_index}",
        )
        batch_events.append(discovery_event)

        schedule_versions: list[dict[str, Any]] = []
        schedule_ids = fixture.get("schedule_capture_ids", [])
        if not isinstance(schedule_ids, list) or any(not isinstance(item, str) for item in schedule_ids):
            reasons.append("schedule_unavailable")
            schedule_ids = []
        for revision_index, schedule_id in enumerate(schedule_ids):
            try:
                schedule_capture, schedule_body = _read_capture_json(
                    session, schedule_id, Path(body_store)
                )
                linked_captures[schedule_id] = schedule_capture
                capture_ids.append(schedule_id)
                start_at = _aware_datetime(
                    schedule_body.get("scheduled_start_at"), "scheduled_start_at", nullable=False
                )
                availability_at = _capture_availability(schedule_capture)
                if availability_at is None:
                    raise ValueError("schedule capture has no actual availability clock")
                schedule_versions.append({
                    "capture": schedule_capture,
                    "capture_id": schedule_id,
                    "scheduled_start_at": start_at,
                    "available_at": availability_at,
                    "source_status": str(schedule_body.get("status") or "unknown"),
                    "revision": schedule_body.get("revision"),
                    "revision_index": revision_index,
                })
            except (TypeError, ValueError):
                reasons.append("schedule_unavailable")
                continue
        schedule_versions.sort(key=lambda item: (item["available_at"], item["capture_id"]))
        for schedule in schedule_versions:
            nominal_origin = schedule["scheduled_start_at"] - timedelta(hours=1)
            schedule_event = _append_batch_event(
                session,
                protocol_id=protocol_id,
                provider_id=provider_id,
                batch_id=batch_id,
                fixture_id=fixture_id,
                origin_at=nominal_origin,
                event_type="schedule_revision",
                primary_decision=False,
                parent_event_key=discovery_event.event_key,
                status="schedule_revision",
                rejection_reasons=[],
                clocks={
                    "discovered_at": _iso_clock(discovered_at),
                    "scheduled_start_at": _iso_clock(schedule["scheduled_start_at"]),
                    "nominal_origin_at": _iso_clock(nominal_origin),
                    "response_received_at": _iso_clock(schedule["capture"]["response_received_at"]),
                    "first_observed_at": _iso_clock(schedule["capture"]["first_observed_at"]),
                    "observed_at": _iso_clock(schedule["capture"]["observed_at"]),
                    "schedule_available_at": _iso_clock(schedule["available_at"]),
                },
                model_availability={"available": False, "reason": "not_attempted"},
                feature_availability={"available": False, "reason": "not_assessed"},
                information_status="local_fixture" if schedule["capture"]["source_class"] == "local_fixture" else "recorded",
                source_authentication_status=schedule["capture"]["source_authentication_status"],
                source_capture_ids=[discovery_capture["capture_id"], schedule["capture_id"]],
                details={
                    "schedule_revision": schedule["revision"],
                    "source_status": schedule["source_status"],
                },
                identity=f"{discovery_event.event_key}:{schedule['capture_id']}",
            )
            batch_events.append(schedule_event)

        due_schedule: dict[str, Any] | None = None
        for schedule in schedule_versions:
            if schedule["available_at"] > as_of_clock:
                continue
            if due_schedule is None:
                due_schedule = schedule
                continue
            due_origin = due_schedule["scheduled_start_at"] - timedelta(hours=1)
            if schedule["available_at"] < due_origin:
                due_schedule = schedule
            else:
                break

        _lock_evidence_identity(session, "opportunity_primary", protocol_id, provider_id, fixture_id)
        existing_primary = session.execute(
            select(OpportunityEvent).where(
                OpportunityEvent.protocol_id == protocol_id,
                OpportunityEvent.provider_id == provider_id,
                OpportunityEvent.fixture_id == fixture_id,
                OpportunityEvent.primary_decision.is_(True),
            )
        ).scalar_one_or_none()
        if existing_primary is not None:
            existing_payload = existing_primary.payload
            primary_decision_fixture_ids.append(fixture_id)
            output_fixtures.append({
                "fixture_id": fixture_id,
                "provider_fixture_id": provider_fixture_id,
                "primary_origin_at": _iso_clock(existing_primary.origin_at),
                "primary_status": existing_payload.get("status"),
                "inference_eligible": existing_payload.get("inference_eligible", False),
                "forward_admitted": existing_payload.get("forward_admitted", False),
                "rejection_reasons": existing_payload.get("rejection_reasons", []),
                "source_capture_ids": existing_payload.get("source_capture_ids", []),
                "forecast_evidence_key": existing_primary.forecast_evidence_key,
            })
            continue

        if due_schedule is None:
            nominal_origin = None
            primary_status = "pending"
            reasons.append("schedule_unavailable")
            primary_reasons = list(reasons)
            source_authentication = _capture_source_auth_status(list(linked_captures.values()))
            pending_event = _append_batch_event(
                session,
                protocol_id=protocol_id,
                provider_id=provider_id,
                batch_id=batch_id,
                fixture_id=fixture_id,
                origin_at=None,
                event_type="origin_pending",
                primary_decision=False,
                parent_event_key=discovery_event.event_key,
                status=primary_status,
                rejection_reasons=primary_reasons,
                clocks={"discovered_at": _iso_clock(discovered_at), "origin_at": None, "as_of": _iso_clock(as_of_clock)},
                model_availability={"available": False, "reason": "not_attempted"},
                feature_availability={"available": False, "reason": "not_assessed"},
                information_status="local_fixture" if source_authentication == "local_fixture" else "recorded",
                source_authentication_status=source_authentication,
                source_capture_ids=capture_ids,
                details={"provider_fixture_id": provider_fixture_id, "schedule_revision": None},
                identity=f"{discovery_event.event_key}:no-due-schedule",
            )
            batch_events.append(pending_event)
            pending_fixture_ids.append(fixture_id)
            output_fixtures.append({
                "fixture_id": fixture_id,
                "provider_fixture_id": provider_fixture_id,
                "primary_origin_at": None,
                "primary_status": primary_status,
                "inference_eligible": False,
                "forward_admitted": False,
                "rejection_reasons": sorted(set(primary_reasons)),
                "source_capture_ids": list(dict.fromkeys(capture_ids)),
                "forecast_evidence_key": None,
            })
            continue

        nominal_origin = due_schedule["scheduled_start_at"] - timedelta(hours=1)
        if as_of_clock < nominal_origin:
            primary_status = "pending"
            pending_reasons = list(reasons)
            source_authentication = _capture_source_auth_status(list(linked_captures.values()))
            pending_event = _append_batch_event(
                session,
                protocol_id=protocol_id,
                provider_id=provider_id,
                batch_id=batch_id,
                fixture_id=fixture_id,
                origin_at=nominal_origin,
                event_type="origin_pending",
                primary_decision=False,
                parent_event_key=discovery_event.event_key,
                status=primary_status,
                rejection_reasons=pending_reasons,
                clocks={
                    "discovered_at": _iso_clock(discovered_at),
                    "scheduled_start_at": _iso_clock(due_schedule["scheduled_start_at"]),
                    "origin_at": _iso_clock(nominal_origin),
                    "as_of": _iso_clock(as_of_clock),
                },
                model_availability={"available": False, "reason": "not_attempted"},
                feature_availability={"available": False, "reason": "not_assessed"},
                information_status="local_fixture" if source_authentication == "local_fixture" else "recorded",
                source_authentication_status=source_authentication,
                source_capture_ids=capture_ids,
                details={
                    "provider_fixture_id": provider_fixture_id,
                    "schedule_revision": due_schedule["revision"],
                },
                identity=f"{discovery_event.event_key}:{due_schedule['capture_id']}",
            )
            batch_events.append(pending_event)
            pending_fixture_ids.append(fixture_id)
            output_fixtures.append({
                "fixture_id": fixture_id,
                "provider_fixture_id": provider_fixture_id,
                "primary_origin_at": _iso_clock(nominal_origin),
                "primary_status": primary_status,
                "inference_eligible": False,
                "forward_admitted": False,
                "rejection_reasons": sorted(set(pending_reasons)),
                "source_capture_ids": list(dict.fromkeys(capture_ids)),
                "forecast_evidence_key": None,
            })
            continue

        schedule_available_at = due_schedule["available_at"]
        late_discovery = (
            discovered_at is None
            or discovered_at >= nominal_origin
            or schedule_available_at >= nominal_origin
        )
        if late_discovery:
            reasons.append("late_discovery")
        if due_schedule["source_status"].casefold() not in {
            "scheduled", "upcoming", "rescheduled",
        }:
            reasons.append("schedule_unavailable")

        rosters: dict[str, dict[str, Any] | None] = {}
        roster_results: dict[str, Any] = {}
        roster_capture_ids = fixture.get("roster_capture_ids", [])
        if not isinstance(roster_capture_ids, list) or any(not isinstance(item, str) for item in roster_capture_ids):
            roster_capture_ids = []
            reasons.append("roster_missing")
        roster_announcements: list[dict[str, Any]] = []
        roster_links: list[dict[str, Any]] = []
        roster_proof_captures = {}
        if is_c0:
            proof_capture_ids = {ref["capture_id"]
                for receipt in protocol.get("authentication_receipts", [])
                if any(ref.get("capture_id") in roster_capture_ids for ref in receipt.get("capture_refs", []))
                for ref in receipt.get("capture_refs", [])}
            for proof_capture_id in proof_capture_ids:
                try:
                    capture = load_source_capture(session, proof_capture_id, body_store=Path(body_store))
                    if capture is not None:
                        body = capture.pop("body_bytes")
                        capture["content_base64"] = base64.b64encode(body).decode("ascii")
                        roster_proof_captures[proof_capture_id] = capture
                except (TypeError, ValueError):
                    reasons.append("roster_authentication_capture_unavailable")
        for roster_capture_id in roster_capture_ids:
            try:
                roster_capture, roster_body = _read_capture_json(
                    session, roster_capture_id, Path(body_store)
                )
                linked_captures[roster_capture_id] = roster_capture
                capture_ids.append(roster_capture_id)
                decorated_announcements = _decorate_provenance_records(roster_body.get("announcements", []), roster_capture)
                if is_c0:
                    from betting_app.services.c0_shadow_readiness import validate_source_adjudication
                    checked = any(
                        any(ref.get("capture_id") == roster_capture_id for ref in receipt.get("capture_refs", []))
                        and not validate_source_adjudication(
                            receipt, protocol=protocol, captures=list(roster_proof_captures.values())
                        )
                        for receipt in protocol.get("authentication_receipts", [])
                    )
                    for announcement in decorated_announcements:
                        announcement["effective_authentication_status"] = "authenticated" if checked else "unverified"
                        for clock in ("response_received_at", "first_observed_at"):
                            announcement[clock] = _iso_clock(roster_capture.get(clock))
                roster_announcements.extend(decorated_announcements)
                roster_links.extend(
                    _decorate_provenance_records(roster_body.get("identity_links", []), roster_capture)
                )
            except (TypeError, ValueError):
                reasons.append("roster_missing")

        if teams_valid:
            start_at = due_schedule["scheduled_start_at"]
            for team_id in team_ids:
                try:
                    if is_c0:
                        from betting_app.services.c0_shadow_readiness import resolve_c0_roster
                        resolution = resolve_c0_roster(
                            team_id=team_id, match_id=fixture_id, cutoff_at=nominal_origin,
                            competition_context=fixture.get("competition_context"),
                            competitive_unit_id=(fixture.get("competitive_unit_ids", {}).get(str(team_id))
                                                 if isinstance(fixture.get("competitive_unit_ids"), dict) else None),
                            announcements=roster_announcements,
                        )
                    else:
                        resolution = resolve_roster_evidence(
                            team_id=team_id,
                            match_id=fixture_id,
                            cutoff_at=nominal_origin,
                            start_at=start_at,
                            identity_links=roster_links,
                            announcements=roster_announcements,
                        )
                except (TypeError, ValueError):
                    resolution = {
                        "announcement": None,
                        "status": "ambiguous",
                        "missing_reason": "roster_evidence_invalid",
                        "ambiguity_evidence": [],
                    }
                roster_results[str(team_id)] = resolution
                announcement = resolution.get("announcement")
                rosters[str(team_id)] = announcement
                if announcement is None:
                    missing_reason = resolution.get("missing_reason")
                    if missing_reason == "latest_announcement_retracted":
                        reasons.append("latest_roster_withdrawn")
                    elif missing_reason in {
                        "conflicting_latest_announcements",
                        "announced_roster_incomplete",
                        "announced_roster_duplicate_players",
                    }:
                        reasons.append("latest_roster_incomplete_or_conflicted")
                    else:
                        reasons.append("roster_missing")
        else:
            reasons.append("roster_missing")
        if teams_valid and all(rosters.get(str(team_id)) for team_id in team_ids):
            roster_a = rosters[str(team_ids[0])]["roster"]
            roster_b = rosters[str(team_ids[1])]["roster"]
            if len(set(roster_a) & set(roster_b)):
                reasons.append("latest_roster_incomplete_or_conflicted")

        quote_refs = fixture.get("quote_capture_refs", [])
        if not isinstance(quote_refs, list):
            quote_refs = []
            reasons.append("quote_unavailable")
        quote_evidence: list[dict[str, Any]] = []
        valid_bookmakers: dict[str, str] = {}
        for quote_ref in quote_refs:
            if not isinstance(quote_ref, dict):
                reasons.append("quote_unavailable")
                continue
            quote_capture_id = quote_ref.get("capture_id")
            bookmaker_slug = quote_ref.get("bookmaker_slug")
            try:
                quote_capture, quote_body = _read_capture_json(
                    session, str(quote_capture_id), Path(body_store)
                )
                linked_captures[str(quote_capture_id)] = quote_capture
                capture_ids.append(str(quote_capture_id))
            except (TypeError, ValueError):
                reasons.append("quote_unavailable")
                continue
            binding_capture_id = quote_ref.get("binding_capture_id")
            participant_bindings: list[dict[str, Any]] = []
            if binding_capture_id:
                try:
                    binding_capture, binding_body = _read_capture_json(
                        session, str(binding_capture_id), Path(body_store)
                    )
                    linked_captures[str(binding_capture_id)] = binding_capture
                    capture_ids.append(str(binding_capture_id))
                    binding_available = _capture_availability(binding_capture)
                    if binding_available is None or binding_available > nominal_origin:
                        raise ValueError("participant binding was unavailable at the origin")
                    participant_bindings = _decorate_provenance_records(
                        binding_body.get("participant_bindings", []), binding_capture
                    )
                except (TypeError, ValueError):
                    participant_bindings = []
            quote_fixture = quote_body.get("fixture")
            quote_context = dict(quote_fixture) if isinstance(quote_fixture, dict) else {}
            malformed_context = quote_fixture is not None and not isinstance(quote_fixture, dict)
            quote_context.setdefault("fixtureId", quote_body.get("fixtureId"))
            quote_context.setdefault("start_at", _iso_clock(due_schedule["scheduled_start_at"]))
            if teams_valid:
                quote_context.setdefault("participant1Id", str(team_ids[0]))
                quote_context.setdefault("participant2Id", str(team_ids[1]))
            try:
                if malformed_context:
                    raise ValueError("quote fixture context must be an object")
                parsed_quote = extract_winner_quote(
                    quote_body,
                    str(bookmaker_slug or ""),
                    cutoff_at=nominal_origin,
                    fixture=quote_context,
                    participant_bindings=participant_bindings,
                )
            except (TypeError, ValueError):
                parsed_quote = None
            quote_receipt = quote_capture.get("response_received_at")
            quote_age_seconds = (
                (nominal_origin - quote_receipt).total_seconds()
                if quote_receipt is not None
                else None
            )
            quote_available = _capture_availability(quote_capture)
            fresh = (
                quote_age_seconds is not None and 0 <= quote_age_seconds <= 300
                and quote_available is not None and quote_available <= nominal_origin
            )
            if not fresh:
                reasons.append("stale_quote")
            if parsed_quote is None:
                reasons.append("quote_unavailable")
                quote_evidence.append({
                    "capture_id": quote_capture_id,
                    "sha256": quote_capture["sha256"],
                    "bookmaker_slug": bookmaker_slug,
                    "underlying_bookmaker_id": quote_ref.get("underlying_bookmaker_id"),
                    "response_received_at": _iso_clock(quote_receipt),
                    "quote_age_seconds": quote_age_seconds,
                    "parse_status": "unavailable",
                })
                continue
            if parsed_quote["quote_admission_status"] != "admitted_with_bookmaker_evidence":
                if not participant_bindings:
                    reasons.append("participant_binding_absent")
                elif parsed_quote["binding_validation_reasons"]:
                    reasons.append("participant_binding_invalid")
                else:
                    reasons.append("quote_scope_unverified")
            states_active = (
                parsed_quote["state_1"]["status"] == "active"
                and parsed_quote["state_2"]["status"] == "active"
            )
            both_prices = parsed_quote["odds_1"] is not None and parsed_quote["odds_2"] is not None
            if not states_active or not both_prices:
                reasons.append("latest_quote_invalid")
            underlying_bookmaker_id = quote_ref.get("underlying_bookmaker_id")
            if (
                fresh
                and states_active
                and both_prices
                and parsed_quote["quote_admission_status"] == "admitted_with_bookmaker_evidence"
                and isinstance(underlying_bookmaker_id, str)
                and underlying_bookmaker_id.strip()
            ):
                bookmaker = str(bookmaker_slug).strip().casefold()
                identity = underlying_bookmaker_id.strip()
                if bookmaker in valid_bookmakers and valid_bookmakers[bookmaker] != identity:
                    reasons.append("bookmaker_identity_conflict")
                else:
                    valid_bookmakers[bookmaker] = identity
            quote_evidence.append({
                "capture_id": quote_capture_id,
                "sha256": quote_capture["sha256"],
                "bookmaker_slug": bookmaker_slug,
                "underlying_bookmaker_id": underlying_bookmaker_id,
                "response_received_at": _iso_clock(quote_receipt),
                "first_observed_at": _iso_clock(quote_capture["first_observed_at"]),
                "observed_at": _iso_clock(quote_capture["observed_at"]),
                "quote_age_seconds": quote_age_seconds,
                "fresh": fresh,
                "odds_1": parsed_quote["odds_1"],
                "odds_2": parsed_quote["odds_2"],
                "quote_at": parsed_quote["quote_at"],
                "quote_admission_status": parsed_quote["quote_admission_status"],
                "binding_validation_reasons": parsed_quote["binding_validation_reasons"],
                "outcome_fixture_participant_indexes": parsed_quote["outcome_fixture_participant_indexes"],
                "state_1": parsed_quote["state_1"]["status"],
                "state_2": parsed_quote["state_2"]["status"],
            })
        if not quote_refs:
            reasons.append("quote_unavailable")
        if len(set(valid_bookmakers.values())) < 2:
            reasons.append("insufficient_distinct_books")

        actual_start_at: datetime | None = None
        actual_start_capture_id = fixture.get("actual_start_capture_id")
        if actual_start_capture_id:
            try:
                actual_capture, actual_body = _read_capture_json(
                    session, str(actual_start_capture_id), Path(body_store)
                )
                if not is_c0:
                    linked_captures[str(actual_start_capture_id)] = actual_capture
                    capture_ids.append(str(actual_start_capture_id))
                actual_start_at = _aware_datetime(
                    actual_body.get("actual_start_at"), "actual_start_at", nullable=False
                )
                if actual_start_at <= as_of_clock:
                    reasons.append("actual_start_reached")
            except (TypeError, ValueError):
                actual_start_at = None

        feature_payload = fixture.get("feature_payload")
        c0_request = None
        if is_c0 and teams_valid and all(rosters.get(str(team)) for team in team_ids):
            original_request = fixture.get("c0_request", {})
            if not isinstance(original_request, dict):
                original_request = {}
                reasons.append("native_request_invalid")
            if original_request.get("mode", "full") != "full":
                reasons.append("native_primary_mode_invalid")
            if fixture.get("competition_context"):
                c0_request = {
                    **original_request,
                    "team1_id": str(team_ids[0]), "team2_id": str(team_ids[1]),
                    "roster_a": sorted(map(str, rosters[str(team_ids[0])]["roster"])),
                    "roster_b": sorted(map(str, rosters[str(team_ids[1])]["roster"])),
                    "best_of": fixture.get("best_of"), "mode": "full",
                    "competition_context": fixture["competition_context"],
                    "decision_at": nominal_origin.isoformat(),
                    "start_at": due_schedule["scheduled_start_at"].isoformat(),
                }
            feature_payload = {"c0_request": c0_request} if c0_request is not None else None
        feature_available = isinstance(feature_payload, dict) and bool(feature_payload)
        if not feature_available:
            reasons.append("feature_unavailable")
        source_authentication_status = _capture_source_auth_status(list(linked_captures.values()))
        if is_c0:
            from betting_app.services.c0_shadow_readiness import validate_source_adjudication
            adjudications = protocol.get("authentication_receipts", [])
            checked_capture_ids = {
                ref.get("capture_id")
                for receipt in adjudications
                if not validate_source_adjudication(receipt, protocol=protocol, captures=list(linked_captures.values()))
                for ref in receipt.get("capture_refs", [])
            }
            source_authentication_status = (
                "authenticated" if set(capture_ids).issubset(checked_capture_ids)
                else "local_fixture" if any(item["source_class"] == "local_fixture" for item in linked_captures.values())
                else "unverified"
            )
        if source_authentication_status != "authenticated":
            reasons.append(
                "local_fixture_not_authenticated"
                if source_authentication_status == "local_fixture"
                else "source_authentication_unverified"
            )

        if as_of_clock > nominal_origin + timedelta(seconds=60):
            reasons.append("origin_deadline_missed")
        known_starts = [
            schedule["scheduled_start_at"] for schedule in schedule_versions
            if schedule["available_at"] <= as_of_clock
        ]
        if actual_start_at is not None:
            known_starts.append(actual_start_at)
        known_start_bound = min(known_starts)
        if as_of_clock >= known_start_bound:
            reasons.append("known_start_reached")
        is_late = "late_discovery" in reasons
        is_started = "actual_start_reached" in reasons
        hard_reasons = [
            reason
            for reason in set(reasons)
            if reason not in {
                "local_fixture_not_authenticated",
                "source_authentication_unverified",
            }
        ]
        inference_eligible = not hard_reasons
        forecast_evidence_key: str | None = None
        model_availability = {"available": False, "reason": "not_attempted"}
        if is_late:
            primary_status = "late_discovery"
        elif is_started:
            primary_status = "started"
        elif hard_reasons:
            primary_status = "rejected"
        elif predictor is None:
            reasons.append("model_unavailable")
            primary_status = "eligible_without_forecast"
        else:
            candidate = {
                "protocol_id": protocol_id,
                "provider_id": provider_id,
                "fixture": fixture,
                "fixture_id": fixture_id,
                "origin_at": nominal_origin,
                "scheduled_start_at": due_schedule["scheduled_start_at"],
                "known_start_bound": known_start_bound,
                "rosters": rosters,
                "roster_results": roster_results,
                "quotes": quote_evidence,
                "source_captures": [linked_captures[key] for key in capture_ids if key in linked_captures],
                "feature_payload": feature_payload,
                **({"c0_request": c0_request, "protocol": protocol,
                    "authorization": authorization} if is_c0 else {}),
            }
            try:
                with session.begin_nested() as inference_transaction:
                    prediction_result = predictor(candidate)
                    envelope = (
                        load_forecast_evidence(session, prediction_result)
                        if isinstance(prediction_result, str) else None
                    )
                    if envelope is None:
                        link_reasons = ["forecast_envelope_missing"]
                    else:
                        recorded_at = session.execute(
                            select(ForecastEvidenceSnapshot.recorded_at).where(
                                ForecastEvidenceSnapshot.evidence_key == prediction_result
                            )
                        ).scalar_one()
                        link_reasons = _forecast_link_reasons(envelope, candidate, recorded_at=recorded_at)
                    if link_reasons:
                        inference_transaction.rollback()
            except Exception as exc:
                reasons.append("model_failure")
                primary_status = "model_failure"
                model_availability = {
                    "available": False,
                    "reason": "inference_error",
                    "error_type": type(exc).__name__,
                }
            else:
                model_availability = {"available": True, "reason": "inference_attempted"}
                if link_reasons:
                    reasons.extend(link_reasons)
                    primary_status = "eligible_without_forecast"
                else:
                    forecast_evidence_key = prediction_result
                    primary_status = "forecasted"
        if predictor is None:
            model_availability = {"available": False, "reason": "model_unavailable"}
        decision_at = datetime.now(_UTC)
        if inference_eligible and (
            decision_at > nominal_origin + timedelta(seconds=60)
            or decision_at >= known_start_bound
        ):
            if decision_at > nominal_origin + timedelta(seconds=60):
                reasons.append("origin_deadline_missed")
            if decision_at >= known_start_bound:
                reasons.append("known_start_reached")
            if actual_start_at is not None and decision_at >= actual_start_at:
                reasons.append("actual_start_reached")
            inference_eligible = False
            forecast_evidence_key = None
            primary_status = "rejected"
        forward_admitted = False
        if is_c0 and forecast_evidence_key is not None and inference_eligible:
            from betting_app.services.c0_shadow_readiness import (
                _encoded_from_envelope, evaluate_c0_readiness, validate_dependency_receipt,
            )
            readiness = evaluate_c0_readiness(protocol, now=decision_at, authorization=authorization)
            dependency_reasons = validate_dependency_receipt(
                envelope["input"].get("dependency_receipt"), protocol=protocol,
                encoded_snapshot=_encoded_from_envelope(envelope),
                sources=envelope["sources"],
            )
            reasons.extend(readiness["reasons"])
            reasons.extend(dependency_reasons)
            # Actual start/settlement evidence may be released after this primary
            # decision; phase admission certifies it without a new decision.
            forward_admitted = (
                readiness["status"] == "READY_AUTHORIZED"
                and not dependency_reasons
                and source_authentication_status == "authenticated"
            )
        if not forward_admitted:
            reasons.append("forward_admission_not_authorized")
        feature_availability = {
            "available": feature_available,
            "reason": "captured_native_feature_payload" if feature_available else "feature_unavailable",
        }
        information_status = (
            "local_fixture"
            if any(capture["source_class"] == "local_fixture" for capture in linked_captures.values())
            else "recorded"
        )
        decision_clocks = {
            "discovered_at": _iso_clock(discovered_at),
            "schedule_available_at": _iso_clock(schedule_available_at),
            "scheduled_start_at": _iso_clock(due_schedule["scheduled_start_at"]),
            "origin_at": _iso_clock(nominal_origin),
            "evaluation_at": _iso_clock(as_of_clock),
            "decision_at": _iso_clock(decision_at),
            "actual_start_at": _iso_clock(actual_start_at),
            "known_start_bound": _iso_clock(known_start_bound),
            "source_response_received_at": {
                key: _iso_clock(value["response_received_at"])
                for key, value in linked_captures.items()
            },
            "source_first_observed_at": {
                key: _iso_clock(value["first_observed_at"])
                for key, value in linked_captures.items()
            },
        }
        primary_event = _append_batch_event(
            session,
            protocol_id=protocol_id,
            provider_id=provider_id,
            batch_id=batch_id,
            fixture_id=fixture_id,
            origin_at=nominal_origin,
            event_type="primary_decision",
            primary_decision=True,
            parent_event_key=discovery_event.event_key,
            status=primary_status,
            rejection_reasons=reasons,
            clocks=decision_clocks,
            model_availability=model_availability,
            feature_availability=feature_availability,
            information_status=information_status,
            source_authentication_status=source_authentication_status,
            source_capture_ids=list(dict.fromkeys(capture_ids)),
            inference_eligible=inference_eligible,
            forward_admitted=forward_admitted,
            details={
                "provider_fixture_id": provider_fixture_id,
                "canonical_fixture_id": fixture.get("canonical_fixture_id"),
                "team_ids": team_ids,
                **({"competition_context": fixture.get("competition_context"),
                    "competitive_unit_ids": fixture.get("competitive_unit_ids"),
                    "c0_authorization": authorization,
                    "c0_protocol": protocol,
                    "outcome_capture_ids": list(dict.fromkeys(
                        ([str(actual_start_capture_id)] if actual_start_capture_id else [])
                        + fixture.get("outcome_capture_ids", [])
                    ))} if is_c0 else {}),
                "best_of": fixture.get("best_of"),
                "schedule_revision": due_schedule["revision"],
                "schedule_capture_id": due_schedule["capture_id"],
                "roster_results": roster_results,
                "quote_evidence": quote_evidence,
                "input_eligibility": "eligible_for_local_inference" if not hard_reasons else "rejected",
                "replacement_fixture_id": fixture.get("replacement_fixture_id"),
            },
            forecast_evidence_key=forecast_evidence_key,
            identity=due_schedule["capture_id"],
        )
        batch_events.append(primary_event)
        primary_decision_fixture_ids.append(fixture_id)
        output_fixtures.append({
            "fixture_id": fixture_id,
            "provider_fixture_id": provider_fixture_id,
            "primary_origin_at": _iso_clock(nominal_origin),
            "primary_status": primary_status,
            "inference_eligible": inference_eligible,
            "forward_admitted": forward_admitted,
            "rejection_reasons": sorted(set(reasons)),
            "source_capture_ids": list(dict.fromkeys(capture_ids)),
            "forecast_evidence_key": forecast_evidence_key,
        })

    unique_accounted = list(dict.fromkeys(accounted_fixture_ids))
    output_fixtures = list({row["fixture_id"]: row for row in output_fixtures}.values())
    all_events = load_opportunity_events(
        session, protocol_id=protocol_id, provider_id=provider_id
    )
    counts = {
        "discovered": len(unique_accounted),
        "pending": len(set(pending_fixture_ids)),
        "primary_decisions": len(set(primary_decision_fixture_ids)),
        "with_forecast": sum(row["forecast_evidence_key"] is not None for row in output_fixtures),
        "without_forecast": sum(row["forecast_evidence_key"] is None for row in output_fixtures),
    }
    serialized_events = [
        {
            "event_key": event_row.event_key,
            "protocol_id": event_row.protocol_id,
            "provider_id": event_row.provider_id,
            "fixture_id": event_row.fixture_id,
            "origin_at": _iso_clock(event_row.origin_at),
            "event_type": event_row.event_type,
            "primary_decision": event_row.primary_decision,
            "parent_event_key": event_row.parent_event_key,
            "payload": event_row.payload,
            "payload_sha256": event_row.payload_sha256,
            "forecast_evidence_key": event_row.forecast_evidence_key,
            "recorded_at": _iso_clock(event_row.recorded_at),
        }
        for event_row in all_events
    ]
    return {
        "batch_id": batch_id,
        "protocol_id": protocol_id,
        "provider_id": provider_id,
        "as_of": _iso_clock(as_of_clock),
        "input_fixture_count": len(feed_entries),
        "accounted_fixture_ids": unique_accounted,
        "pending_fixture_ids": list(dict.fromkeys(pending_fixture_ids)),
        "primary_decision_fixture_ids": list(dict.fromkeys(primary_decision_fixture_ids)),
        "events": serialized_events,
        "fixtures": output_fixtures,
        "counts": counts,
    }


def export_opportunity_ledger(
    session: Session,
    *,
    protocol_id: str,
    output_path: Path,
    max_rows: int,
) -> dict[str, Any]:
    """Export every immutable event atomically as hash-receipted canonical JSONL."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    rows = load_opportunity_events(session, protocol_id=protocol_id)
    if len(rows) > max_rows:
        raise ValueError(f"opportunity event count {len(rows)} exceeds max_rows={max_rows}")
    serialized = []
    fixture_ids: list[str] = []
    for row in rows:
        fixture_ids.append(row.fixture_id)
        serialized.append({
            "id": row.id,
            "event_key": row.event_key,
            "protocol_id": row.protocol_id,
            "provider_id": row.provider_id,
            "fixture_id": row.fixture_id,
            "origin_at": _iso_clock(row.origin_at),
            "event_type": row.event_type,
            "primary_decision": row.primary_decision,
            "parent_event_key": row.parent_event_key,
            "payload": row.payload,
            "payload_sha256": row.payload_sha256,
            "forecast_evidence_key": row.forecast_evidence_key,
            "recorded_at": _iso_clock(row.recorded_at),
        })
    output = b"".join(
        json.dumps(
            item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8") + b"\n"
        for item in serialized
    )
    digest = hashlib.sha256(output).hexdigest()
    destination = Path(output_path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".opportunity-export-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "protocol_id": protocol_id,
        "row_count": len(rows),
        "output_path": str(destination),
        "sha256": digest,
        "fixture_ids": list(dict.fromkeys(fixture_ids)),
        "missing_forecast_count": sum(row.forecast_evidence_key is None for row in rows),
    }
