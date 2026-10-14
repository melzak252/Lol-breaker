"""Exact-byte local source capture contracts against an isolated PostgreSQL DB."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from betting_app.services.forecast_evidence import capture_source_attempt, load_source_capture


UTC = timezone.utc
BASE = datetime(2026, 10, 2, 3, 0, tzinfo=UTC)


@pytest.fixture
def session():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url or "_test" not in url:
        pytest.skip("requires an explicitly isolated TEST_DATABASE_URL")
    engine = create_engine(url)
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as current:
            yield current
        transaction.rollback()
    engine.dispose()


def _capture(session, body_store, *, capture_id=None, raw_body=b'{"x":1}', **changes):
    values = {
        "capture_id": capture_id or str(uuid4()),
        "source_id": "fixture-source",
        "evidence_uri": "fixture://local/source/fixture-source",
        "raw_body": raw_body,
        "body_store": body_store,
        "request_started_at": BASE,
        "response_received_at": BASE + timedelta(seconds=1),
        "first_observed_at": BASE + timedelta(seconds=1),
        "published_at": None,
        "source_effective_at": None,
        "source_revision": None,
        "http_status": 200,
        "source_class": "local_fixture",
        "availability_basis": "local_fixture_bytes",
        "body_representation": "http_body",
        "content_encoding": "identity",
        "character_encoding": "utf-8",
        "request_metadata": {
            "method": "GET",
            "url": "https://api.example/v4/fixtures?apiKey=do-not-retain&league=lol",
            "query": {"apiKey": "do-not-retain", "league": "lol"},
            "headers": {"Cookie": "do-not-retain", "Accept": "application/json"},
        },
        "response_metadata": {
            "content_type": "application/json",
            "content_encoding": "identity",
            "response_id": "fixture-response-1",
            "headers": {"Set-Cookie": "do-not-retain", "Content-Type": "application/json"},
        },
    }
    values.update(changes)
    return capture_source_attempt(session, **values)


def test_capture_keeps_exact_bytes_and_distinct_attempts(tmp_path, session):
    compact = b'{"fixture":"x","price":1.8}'
    spaced = b'{ "fixture" : "x", "price" : 1.8 }'
    first_id, duplicate_id, distinct_id = (str(uuid4()) for _ in range(3))

    first = _capture(session, tmp_path, capture_id=first_id, raw_body=compact)
    same = _capture(session, tmp_path, capture_id=first_id, raw_body=compact)
    later_repeat = BASE + timedelta(hours=2)
    repeated_attempt = _capture(
        session, tmp_path, capture_id=duplicate_id, raw_body=compact,
        request_started_at=later_repeat - timedelta(seconds=1),
        response_received_at=later_repeat,
        first_observed_at=BASE + timedelta(seconds=1),
        observed_at=later_repeat,
    )
    whitespace_variant = _capture(session, tmp_path, capture_id=distinct_id, raw_body=spaced)

    loaded = load_source_capture(session, first_id, body_store=tmp_path)
    assert first.id == same.id
    assert first.id != repeated_attempt.id
    assert first.sha256 == repeated_attempt.sha256 == hashlib.sha256(compact).hexdigest()
    assert first.body_uri == repeated_attempt.body_uri
    assert first.capture_id != repeated_attempt.capture_id
    assert repeated_attempt.response_received_at > repeated_attempt.first_observed_at
    assert whitespace_variant.sha256 == hashlib.sha256(spaced).hexdigest()
    assert whitespace_variant.sha256 != first.sha256
    assert loaded["body_bytes"] == compact
    assert loaded["content_base64"] == first.content_base64
    assert loaded["request_started_at"] == BASE
    assert loaded["response_received_at"] == BASE + timedelta(seconds=1)
    assert loaded["first_observed_at"] == BASE + timedelta(seconds=1)
    assert loaded["published_at"] is None and loaded["source_effective_at"] is None
    assert loaded["source_authentication_status"] == "local_fixture"
    parsed = json.loads(loaded["body_bytes"])
    parsed["price"] = 99
    assert load_source_capture(session, first_id, body_store=tmp_path)["body_bytes"] == compact
    safe_metadata = json.dumps({"request": loaded["request_metadata"], "response": loaded["response_metadata"]})
    assert "do-not-retain" not in safe_metadata
    with pytest.raises(ValueError, match="capture_id"):
        _capture(session, tmp_path, capture_id=first_id, raw_body=b'{"fixture":"changed"}')


def test_capture_records_malformed_http_empty_and_no_response_failures(tmp_path, session):
    malformed_id, denied_id, empty_id, timeout_id = (str(uuid4()) for _ in range(4))
    malformed = _capture(
        session, tmp_path, capture_id=malformed_id, raw_body=b'{not-json',
        response_metadata={"content_type": "application/json"},
    )
    denied = _capture(
        session, tmp_path, capture_id=denied_id, raw_body=b'{"error":"denied"}',
        http_status=403,
    )
    empty = _capture(session, tmp_path, capture_id=empty_id, raw_body=b"")
    timeout = _capture(
        session, tmp_path, capture_id=timeout_id, raw_body=None,
        response_received_at=None, first_observed_at=None, http_status=None,
        failure_reason="timeout",
    )

    assert (malformed.attempt_status, malformed.failure_reason) == ("failed", "malformed_json")
    assert malformed.sha256 == hashlib.sha256(b"{not-json").hexdigest()
    assert denied.failure_reason == "http_error"
    assert (empty.attempt_status, empty.failure_reason) == ("failed", "empty_body")
    assert empty.sha256 == hashlib.sha256(b"").hexdigest()
    assert (timeout.http_status, timeout.response_received_at, timeout.first_observed_at) == (None, None, None)
    assert timeout.attempt_status == "failed" and timeout.failure_reason == "timeout"
    assert load_source_capture(session, timeout_id, body_store=tmp_path)["body_bytes"] is None


def test_capture_rejects_unverifiable_backdated_first_observation(tmp_path, session):
    with pytest.raises(ValueError, match="first_observed_at cannot predate the first response"):
        _capture(
            session,
            tmp_path,
            raw_body=b'{"new-exact-revision":true}',
            request_started_at=BASE,
            response_received_at=BASE + timedelta(seconds=1),
            first_observed_at=BASE,
        )


def test_capture_rejects_clock_contradictions_and_database_mutation(tmp_path, session):
    with pytest.raises(ValueError, match="clock"):
        _capture(
            session, tmp_path, request_started_at=BASE + timedelta(seconds=2),
            response_received_at=BASE + timedelta(seconds=1),
            first_observed_at=BASE + timedelta(seconds=1),
        )
    row = _capture(session, tmp_path)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(
            text("UPDATE source_captures SET failure_reason = 'rewritten' WHERE capture_id = :key"),
            {"key": row.capture_id},
        )
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(
            text("DELETE FROM source_captures WHERE capture_id = :key"),
            {"key": row.capture_id},
        )
    assert load_source_capture(session, row.capture_id, body_store=tmp_path)["body_bytes"] == b'{"x":1}'


def test_capture_rejects_timeout_observed_before_request(tmp_path, session):
    with pytest.raises(ValueError, match="request_started_at"):
        _capture(
            session, tmp_path, raw_body=None, response_received_at=None,
            first_observed_at=None, observed_at=BASE - timedelta(seconds=1),
            http_status=None, failure_reason="transport_error",
        )


def test_capture_sanitizes_fixture_locator_credentials(tmp_path, session):
    row = _capture(
        session, tmp_path,
        evidence_uri="fixture://reader:private-password@local/path?apiKey=private-key#private-token",
        request_metadata={"origin": "fixture://reader:private-password@local/path?apiKey=private-key"},
    )
    loaded = load_source_capture(session, row.capture_id, body_store=tmp_path)
    metadata = json.dumps({"uri": loaded["evidence_uri"], "request": loaded["request_metadata"]})
    assert "private-" not in metadata
    assert loaded["evidence_uri"] == "fixture://local/path"


def test_capture_audit_retains_crash_staging_orphans(tmp_path, session):
    from betting_app.services.forecast_evidence import audit_source_capture_store
    row = _capture(session, tmp_path)
    staging = tmp_path / row.sha256[:2] / ".capture-interrupted"
    staging.write_bytes(b"partial response before process exit")
    audit = audit_source_capture_store(session, body_store=tmp_path)
    assert staging.relative_to(tmp_path).as_posix() in audit["orphan_body_uris"]
    assert staging.read_bytes() == b"partial response before process exit"
