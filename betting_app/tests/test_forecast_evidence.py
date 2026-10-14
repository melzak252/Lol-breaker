"""Consumer-visible immutability against an explicitly selected PostgreSQL test DB."""
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from betting_app.services.forecast_evidence import append_forecast_evidence, load_forecast_evidence


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


def evidence():
    raw = b'{"fixture":"local-1","status":"local fixture, not authentic publication"}'
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1, "information_status": "local_fixture",
        "model": {"name": "test-only-not-a-model", "version": "fixture", "source_ids": ["source-1"]},
        "input": {"snapshot": {"cutoff_day": "2026-01-02"}, "source_ids": ["source-1"]},
        "forecast": {"prob_a": 0.6, "prob_b": 0.4, "generated_at": now, "simulated_origin_at": "2026-01-02T10:00:00+00:00"},
        "roster": {"a": ["a1", "a2", "a3", "a4", "a5"], "b": ["b1", "b2", "b3", "b4", "b5"], "source_ids": ["source-1"]},
        "schedule": {"scheduled_start_at": "2026-01-02T12:00:00+00:00", "actual_start_at": None, "source_ids": ["source-1"]},
        "quotes": [{"fixture_id": "local-1", "bookmaker": "fixture", "market_id": "winner", "outcomes": [{"side": "A", "outcome_id": "a", "odds": "1.80"}, {"side": "B", "outcome_id": "b", "odds": "2.10"}], "source_ids": ["source-1"]}],
        "sources": [{"source_id": "source-1", "evidence_uri": "fixture://local-1", "sha256": hashlib.sha256(raw).hexdigest(), "content_base64": base64.b64encode(raw).decode(), "published_at": None, "observed_at": None, "retrieved_at": now}],
    }


def test_append_round_trip_is_idempotent_and_keeps_real_and_scenario_clocks(session):
    # A long-lived transaction must not backdate an actual evidence receipt.
    transaction_started = session.execute(text("SELECT CURRENT_TIMESTAMP")).scalar_one()
    before_append = session.execute(text("SELECT clock_timestamp()")).scalar_one()
    key, payload = str(uuid4()), evidence()
    first = append_forecast_evidence(session, evidence_key=key, payload=payload)
    repeated = append_forecast_evidence(session, evidence_key=key, payload=deepcopy(payload))
    assert first.id == repeated.id
    assert first.recorded_at.tzinfo is not None
    assert load_forecast_evidence(session, key) == payload
    assert transaction_started <= before_append <= first.recorded_at
    payload["quotes"][0]["outcomes"][0]["odds"] = "99.0"
    assert load_forecast_evidence(session, key)["quotes"][0]["outcomes"][0]["odds"] == "1.80"
    with pytest.raises(ValueError):
        append_forecast_evidence(session, evidence_key=key, payload=payload)


@pytest.mark.parametrize("statement", [
    "UPDATE forecast_evidence_snapshots SET payload_json = '{}' WHERE evidence_key = :key",
    "DELETE FROM forecast_evidence_snapshots WHERE evidence_key = :key",
])
def test_database_rejects_in_place_changes(session, statement):
    key, payload = str(uuid4()), evidence()
    append_forecast_evidence(session, evidence_key=key, payload=payload)
    with pytest.raises(DBAPIError), session.begin_nested():
        session.execute(text(statement), {"key": key})
    assert load_forecast_evidence(session, key) == payload


@pytest.mark.parametrize("damage", ["source_hash", "dangling_link", "naive_clock", "nonfinite"])
def test_corrupt_or_ambiguous_evidence_is_not_written(session, damage):
    payload = evidence()
    if damage == "source_hash":
        payload["sources"][0]["content_base64"] = base64.b64encode(b"other bytes").decode()
    elif damage == "dangling_link":
        payload["schedule"]["source_ids"] = ["missing"]
    elif damage == "naive_clock":
        payload["sources"][0]["retrieved_at"] = "2026-09-30T12:00:00"
    else:
        payload["forecast"]["prob_a"] = float("nan")
    key = str(uuid4())
    with pytest.raises(ValueError):
        append_forecast_evidence(session, evidence_key=key, payload=payload)
    assert load_forecast_evidence(session, key) is None


def test_append_participates_in_callers_transaction(session):
    key = str(uuid4())
    with pytest.raises(RuntimeError), session.begin_nested():
        append_forecast_evidence(session, evidence_key=key, payload=evidence())
        raise RuntimeError("cancel local decision")
    assert load_forecast_evidence(session, key) is None
