"""Offline feed-to-provenance-to-opportunity ledger path; all responses are local bytes."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from betting_app.services.forecast_evidence import (
    capture_source_attempt,
    export_opportunity_ledger,
    orchestrate_local_opportunity_batch,
)

UTC = timezone.utc


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


def _capture_json(session, store: Path, capture_id: str, source_id: str, value, received_at):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return capture_source_attempt(
        session,
        capture_id=capture_id,
        source_id=source_id,
        evidence_uri=f"fixture://{source_id}/{capture_id}",
        raw_body=raw,
        body_store=store,
        request_started_at=received_at - timedelta(seconds=1),
        response_received_at=received_at,
        first_observed_at=received_at,
        observed_at=received_at,
        published_at=None,
        source_effective_at=None,
        source_revision=None,
        http_status=200,
        source_class="local_fixture",
        availability_basis="local_fixture_bytes",
        body_representation="http_body",
        content_encoding="identity",
        character_encoding="utf-8",
        request_metadata={"method": "GET", "origin": "fixture://local", "path": "/fixture"},
        response_metadata={"content_type": "application/json", "content_encoding": "identity"},
    )


def _binding_rows(fixture_id: str, bookmaker: str, observed_at: datetime):
    common = {
        "fixture_id": fixture_id,
        "bookmaker": bookmaker,
        "market_id": "181",
        "bookmaker_fixture_id": f"{bookmaker}-{fixture_id}",
        "bookmaker_market_id": f"{bookmaker}-winner",
        "published_at": (observed_at - timedelta(seconds=30)).isoformat(),
        "evidence_uri": f"fixture://bindings/{bookmaker}/{fixture_id}",
    }
    return [
        {
            **common,
            "outcome_id": "181",
            "participant_id": "team-a",
            "participant_name": "Team A",
            "bookmaker_outcome_id": f"{bookmaker}-home",
            "bookmaker_participant_id": f"{bookmaker}-team-a",
        },
        {
            **common,
            "outcome_id": "182",
            "participant_id": "team-b",
            "participant_name": "Team B",
            "bookmaker_outcome_id": f"{bookmaker}-away",
            "bookmaker_participant_id": f"{bookmaker}-team-b",
        },
    ]


def _quote_payload(fixture_id: str, bookmaker: str, start_at: datetime, state_at: datetime,
                   *, invalid_latest=False):
    states_a = [{"createdAt": state_at.isoformat(), "price": 1.8, "active": True}]
    states_b = [{"createdAt": state_at.isoformat(), "price": 2.1, "active": True}]
    if invalid_latest:
        latest_at = state_at + timedelta(seconds=1)
        states_a.append({"createdAt": latest_at.isoformat(), "price": 1.8, "active": False})
        states_b.append({"createdAt": latest_at.isoformat(), "price": 2.1, "active": False})
    return {
        "fixtureId": fixture_id,
        "fixture": {
            "fixtureId": fixture_id,
            "startTime": start_at.isoformat(),
            "participant1Id": "team-a",
            "participant1Name": "Team A",
            "participant2Id": "team-b",
            "participant2Name": "Team B",
        },
        "bookmakers": {
            bookmaker: {
                "markets": {
                    "181": {
                        "outcomes": {
                            "181": {"players": {"0": states_a}},
                            "182": {"players": {"0": states_b}},
                        }
                    }
                }
            }
        },
    }


def _roster_payload(fixture_id: str, observed_at: datetime, *, withdrawn=False):
    announced_at = (observed_at - timedelta(minutes=2)).isoformat()
    team_a = {
        "match_id": fixture_id,
        "team_id": "team-a",
        "roster": ["a1", "a2", "a3", "a4", "a5"],
        "published_at": announced_at,
        "evidence_uri": f"fixture://roster/{fixture_id}/team-a",
        "status": "announced",
    }
    if withdrawn:
        team_a = {
            **team_a,
            "published_at": (observed_at - timedelta(seconds=1)).isoformat(),
            "status": "retracted",
            "roster": [],
        }
    team_b = {
        "match_id": fixture_id,
        "team_id": "team-b",
        "roster": ["b1", "b2", "b3", "b4", "b5"],
        "published_at": announced_at,
        "evidence_uri": f"fixture://roster/{fixture_id}/team-b",
        "status": "announced",
    }
    return {"identity_links": [], "announcements": [team_a, team_b]}


def test_local_batch_parses_every_fixture_and_exports_complete_primary_and_pending_ledger(tmp_path, session):
    now = datetime.now(UTC).replace(microsecond=0)
    store = tmp_path / "exact-source-bytes"
    batch_id = str(uuid4())
    feed_seen = now - timedelta(minutes=30)
    schedule_seen = now - timedelta(minutes=20)
    roster_seen = now - timedelta(minutes=6)
    feed_rows = []
    expected = {}
    source_capture_ids = []

    def add_capture(capture_id, source_id, value, seen):
        row = _capture_json(session, store, capture_id, source_id, value, seen)
        source_capture_ids.append(row.capture_id)
        return row.capture_id

    def add_fixture(name, *, start_at, origin_at=None, canonical=True, provider_id=True,
                    roster_mode="valid", books=2, binding=True, binding_invalid=False,
                    quote_state="active", quote_received_at=None, schedule_revisions=(),
                    actual_start_at=None, feature=True):
        fixture_id = f"provider-{name}" if provider_id else None
        stable_fixture_id = fixture_id or f"unknown-{name}"
        origin = origin_at or (start_at - timedelta(hours=1))
        schedule_ids = []
        versions = schedule_revisions or [(start_at, schedule_seen)]
        for revision, (scheduled_start, observed) in enumerate(versions):
            schedule_id = f"schedule:{name}:{revision}"
            schedule_ids.append(add_capture(
                schedule_id, "schedule-fixture",
                {"scheduled_start_at": scheduled_start.isoformat(), "status": "scheduled", "revision": revision},
                observed,
            ))
        roster_ids = []
        if roster_mode is not None:
            roster_id = f"roster:{name}"
            roster_ids.append(add_capture(
                roster_id, "roster-fixture",
                _roster_payload(stable_fixture_id, roster_seen, withdrawn=roster_mode == "withdrawn"),
                roster_seen,
            ))
        quote_refs = []
        quote_count = books
        for book_index in range(quote_count):
            slug = "pinnacle" if book_index == 0 else "totalbet"
            underlying_id = f"book-{slug}"
            quote_id = f"quote:{name}:{slug}"
            receipt = quote_received_at or (origin - timedelta(seconds=30))
            state_at = receipt - timedelta(seconds=1)
            quote_capture_id = add_capture(
                quote_id, f"quote-{slug}",
                _quote_payload(
                    stable_fixture_id, slug, start_at, state_at,
                    invalid_latest=quote_state == "invalid_latest",
                ),
                receipt,
            )
            binding_id = None
            if binding:
                binding_id = f"binding:{name}:{slug}"
                binding_rows = _binding_rows(stable_fixture_id, slug, receipt)
                if binding_invalid:
                    binding_rows[0]["bookmaker"] = "different-bookmaker"
                add_capture(
                    binding_id, f"binding-{slug}",
                    {"participant_bindings": binding_rows}, receipt,
                )
            quote_refs.append({
                "capture_id": quote_capture_id,
                "bookmaker_slug": slug,
                "underlying_bookmaker_id": underlying_id,
                "binding_capture_id": binding_id,
            })
        actual_start_capture_id = None
        if actual_start_at is not None:
            actual_start_capture_id = add_capture(
                f"actual-start:{name}", "actual-start-fixture",
                {"actual_start_at": actual_start_at.isoformat()}, actual_start_at,
            )
        row = {
            "provider_fixture_id": fixture_id,
            "canonical_fixture_id": f"canonical-{name}" if canonical else None,
            "team_ids": ["team-a", "team-b"],
            "best_of": 3,
            "schedule_capture_ids": schedule_ids,
            "roster_capture_ids": roster_ids,
            "quote_capture_refs": quote_refs,
            "actual_start_capture_id": actual_start_capture_id,
            "replacement_fixture_id": None,
            "feature_payload": (
                feature if isinstance(feature, dict)
                else {"native_fixture_input": {"valid": True}} if feature else None
            ),
        }
        feed_rows.append(row)
        expected[stable_fixture_id] = {"provider_fixture_id": fixture_id, "origin": origin}
        return stable_fixture_id

    standard_origin = now - timedelta(seconds=10)
    standard_start = standard_origin + timedelta(hours=1)
    eligible_id = add_fixture("eligible", start_at=standard_start)
    unknown_id = add_fixture("unknown", start_at=standard_start, provider_id=False, canonical=False)
    withdrawn_id = add_fixture("withdrawn", start_at=standard_start, roster_mode="withdrawn")
    binding_id = add_fixture("unbound", start_at=standard_start, binding=False)
    invalid_binding_id = add_fixture("invalid-binding", start_at=standard_start, binding_invalid=True)
    invalid_id = add_fixture("invalid", start_at=standard_start, quote_state="invalid_latest")
    one_book_id = add_fixture("one-book", start_at=standard_start, books=1)
    stale_id = add_fixture(
        "stale", start_at=standard_start,
        quote_received_at=standard_origin - timedelta(seconds=301),
    )
    late_id = add_fixture(
        "late", start_at=now + timedelta(minutes=20),
    )
    before_trigger_id = add_fixture(
        "reschedule-before", start_at=now + timedelta(hours=2),
        schedule_revisions=[
            (now + timedelta(hours=2), now - timedelta(minutes=20)),
            (now + timedelta(hours=3), now - timedelta(minutes=1)),
        ],
    )
    after_trigger_id = add_fixture(
        "reschedule-after", start_at=now + timedelta(minutes=50),
        schedule_revisions=[
            (now + timedelta(minutes=50), now - timedelta(minutes=30)),
            (now + timedelta(hours=2), now - timedelta(minutes=1)),
        ],
    )
    started_id = add_fixture(
        "started", start_at=standard_start,
        actual_start_at=now - timedelta(minutes=1),
    )
    model_failure_id = add_fixture(
        "model-failure", start_at=standard_start,
        feature={"native_fixture_input": {"valid": False}},
    )
    pending_id = add_fixture("pending", start_at=now + timedelta(hours=2), roster_mode=None, books=0)

    discovery_id = add_capture(
        "discovery:batch", "provider-feed",
        {"fixtures": feed_rows}, feed_seen,
    )

    callback_fixture_ids = []

    def local_predictor(candidate):
        callback_fixture_ids.append(candidate["fixture_id"])
        # A deliberately unusable local feature payload exercises error capture without a fake forecast.
        fixture = candidate["fixture"]
        if not fixture["feature_payload"]["native_fixture_input"]["valid"]:
            raise RuntimeError("local model fixture failure")
        return None
    result = orchestrate_local_opportunity_batch(
        session,
        protocol_id="forward-test-v1",
        provider_id="fixture-provider",
        batch_id=batch_id,
        discovery_capture_ids=[discovery_id],
        body_store=store,
        as_of=now,
        max_fixtures=20,
        predictor=local_predictor,
    )

    assert result["input_fixture_count"] == len(feed_rows) == 14
    known_ids = {key for key, value in expected.items() if value["provider_fixture_id"] is not None}
    assert known_ids <= set(result["accounted_fixture_ids"])
    assert len(result["accounted_fixture_ids"]) == len(expected)
    unknown_rows = [row for row in result["fixtures"] if row["provider_fixture_id"] is None]
    assert len(unknown_rows) == 1
    unknown_id = unknown_rows[0]["fixture_id"]
    assert len(result["accounted_fixture_ids"]) == len(set(result["accounted_fixture_ids"]))
    report = {row["fixture_id"]: row for row in result["fixtures"]}
    assert report[eligible_id]["primary_status"] == "eligible_without_forecast"
    assert eligible_id in callback_fixture_ids
    assert model_failure_id in callback_fixture_ids
    assert report[eligible_id]["inference_eligible"] is True
    assert report[eligible_id]["forward_admitted"] is False
    assert "forward_admission_not_authorized" in report[eligible_id]["rejection_reasons"]
    assert report[eligible_id]["forecast_evidence_key"] is None
    assert report[unknown_id]["primary_status"] == "rejected"
    assert "unknown_provider_id" in report[unknown_id]["rejection_reasons"]
    assert "latest_roster_withdrawn" in report[withdrawn_id]["rejection_reasons"]
    assert "participant_binding_absent" in report[binding_id]["rejection_reasons"]
    assert "participant_binding_invalid" in report[invalid_binding_id]["rejection_reasons"]
    assert "latest_quote_invalid" in report[invalid_id]["rejection_reasons"]
    assert "insufficient_distinct_books" in report[one_book_id]["rejection_reasons"]
    assert "stale_quote" in report[stale_id]["rejection_reasons"]
    assert report[late_id]["primary_status"] == "late_discovery"
    assert report[before_trigger_id]["primary_status"] == "pending"
    assert report[after_trigger_id]["primary_origin_at"] == expected[after_trigger_id]["origin"].isoformat()
    assert "actual_start_reached" in report[started_id]["rejection_reasons"]
    assert report[model_failure_id]["primary_status"] == "model_failure"
    assert "model_failure" in report[model_failure_id]["rejection_reasons"]
    assert report[model_failure_id]["inference_eligible"] is True
    assert report[model_failure_id]["forward_admitted"] is False
    assert all(row["forward_admitted"] is False for row in report.values())
    assert report[pending_id]["primary_status"] == "pending"
    assert set(result["pending_fixture_ids"]) == {before_trigger_id, pending_id}
    assert result["counts"]["with_forecast"] == 0
    assert all(
        row["forecast_evidence_key"] is None
        for row in result["fixtures"]
    )
    for fixture_id, row in report.items():
        assert row["source_capture_ids"]
        assert row["fixture_id"] == fixture_id

    exported = export_opportunity_ledger(
        session,
        protocol_id="forward-test-v1",
        output_path=tmp_path / "opportunities.jsonl",
        max_rows=500,
    )
    lines = (tmp_path / "opportunities.jsonl").read_bytes().splitlines()
    assert exported["row_count"] == len(lines)
    assert exported["sha256"] == hashlib.sha256(b"\n".join(lines) + b"\n").hexdigest()
    exported_events = [json.loads(line) for line in lines]
    assert {event["fixture_id"] for event in exported_events} == known_ids | {unknown_id}
    assert any(event["event_type"] == "schedule_revision" for event in exported_events)
    assert all(event["forecast_evidence_key"] is None for event in exported_events)


def test_local_orchestrator_enforces_fixture_bound_before_ledger_writes(tmp_path, session):
    now = datetime.now(UTC).replace(microsecond=0)
    store = tmp_path / "exact-source-bytes"
    discovery = _capture_json(
        session, store, "discovery:bounded", "provider-feed",
        {"fixtures": [
            {"provider_fixture_id": "one", "schedule_capture_ids": []},
            {"provider_fixture_id": "two", "schedule_capture_ids": []},
        ]}, now,
    )
    with pytest.raises(ValueError, match="max_fixtures must be a positive integer"):
        orchestrate_local_opportunity_batch(
            session, protocol_id="forward-test-v1", provider_id="fixture-provider",
            batch_id=str(uuid4()), discovery_capture_ids=[discovery.capture_id],
            body_store=store, as_of=now, max_fixtures=0,
        )
    with pytest.raises(ValueError, match="exceeds max_fixtures=1"):
        orchestrate_local_opportunity_batch(
            session, protocol_id="forward-test-v1", provider_id="fixture-provider",
            batch_id=str(uuid4()), discovery_capture_ids=[discovery.capture_id],
            body_store=store, as_of=now, max_fixtures=1,
        )
    assert session.execute(text("SELECT COUNT(*) FROM opportunity_events")).scalar_one() == 0


def test_local_orchestrator_rejects_malformed_capture_before_ledger_writes(tmp_path, session):
    now = datetime.now(UTC).replace(microsecond=0)
    store = tmp_path / "exact-source-bytes"
    malformed = capture_source_attempt(
        session,
        capture_id="discovery:malformed",
        source_id="provider-feed",
        evidence_uri="fixture://provider-feed/discovery-malformed",
        raw_body=b'{"fixtures": [',
        body_store=store,
        request_started_at=now - timedelta(seconds=1),
        response_received_at=now,
        first_observed_at=now,
        observed_at=now,
        published_at=None,
        source_effective_at=None,
        source_revision=None,
        http_status=200,
        source_class="local_fixture",
        availability_basis="local_fixture_bytes",
        body_representation="http_body",
        content_encoding="identity",
        character_encoding="utf-8",
        request_metadata={"method": "GET", "path": "/fixtures"},
        response_metadata={"content_type": "application/json"},
    )
    with pytest.raises(ValueError, match="capture failed"):
        orchestrate_local_opportunity_batch(
            session, protocol_id="forward-test-v1", provider_id="fixture-provider",
            batch_id=str(uuid4()), discovery_capture_ids=[malformed.capture_id],
            body_store=store, as_of=now, max_fixtures=1,
        )
    assert session.execute(text("SELECT COUNT(*) FROM opportunity_events")).scalar_one() == 0


def _boundary_inputs(session, store, origin, *, case="valid", duplicate=False):
    """Actual stored local bytes; one fixture with independently captured inputs."""
    fixture_id = f"boundary-{uuid4()}"
    capture_ids = {}

    def capture(kind, value, seen):
        capture_id = f"{kind}:{fixture_id}"
        if case == f"unknown-{kind}-clock":
            raw = json.dumps(value, sort_keys=True).encode()
            row = capture_source_attempt(
                session, capture_id=capture_id, source_id=kind,
                evidence_uri=f"fixture://boundary/{capture_id}", raw_body=raw,
                body_store=store, request_started_at=None, response_received_at=None,
                first_observed_at=None, observed_at=None,
                published_at=origin - timedelta(hours=1), source_class="local_fixture",
                availability_basis="local_fixture_bytes", request_metadata={},
                response_metadata={"content_type": "application/json"},
            )
        else:
            row = _capture_json(session, store, capture_id, kind, value, seen)
        capture_ids[kind] = row.capture_id
        return row.capture_id

    start = origin + timedelta(hours=1)
    schedule = capture(
        "schedule", {"scheduled_start_at": start.isoformat(), "status": "scheduled"},
        origin - timedelta(minutes=20),
    )
    roster_seen = origin - timedelta(minutes=6)
    roster_body = _roster_payload(fixture_id, roster_seen)
    if case == "unknown-roster-clock":
        for announcement in roster_body["announcements"]:
            announcement["observed_at"] = roster_seen.isoformat()
            announcement["captured_available_at"] = roster_seen.isoformat()
    roster = capture("roster", roster_body, roster_seen)
    quote_refs = []
    for book in ("pinnacle", "totalbet"):
        seen = origin - timedelta(seconds=30)
        quote_body = _quote_payload(fixture_id, book, start, seen - timedelta(seconds=1))
        if case == "malformed-quote" and book == "pinnacle":
            quote_body["fixture"] = 42
        quote = capture(f"quote-{book}", quote_body, seen)
        binding = capture(
            f"binding-{book}", {"participant_bindings": _binding_rows(fixture_id, book, seen)},
            seen,
        )
        quote_refs.append({
            "capture_id": quote, "bookmaker_slug": book,
            "underlying_bookmaker_id": book, "binding_capture_id": binding,
        })
    if case == "duplicate-book":
        quote_refs = [quote_refs[0], {**quote_refs[0], "underlying_bookmaker_id": "second-feed-not-second-book"}]
    fixture = {
        "provider_fixture_id": fixture_id, "canonical_fixture_id": f"canonical-{fixture_id}",
        "team_ids": [] if case == "missing-teams" else ["team-a", "team-b"],
        "best_of": 3, "schedule_capture_ids": [schedule],
        "roster_capture_ids": [roster], "quote_capture_refs": quote_refs,
        "feature_payload": {"native_fixture_input": {"valid": True}},
    }
    discovery = capture(
        "discovery", {"fixtures": [fixture, fixture] if duplicate else [fixture]},
        origin - timedelta(minutes=30),
    )
    return fixture_id, discovery


@pytest.mark.parametrize("case,reason", [
    ("missing-teams", "canonical_identity_unresolved"),
    ("malformed-quote", "quote_unavailable"),
    ("unknown-discovery-clock", "late_discovery"),
    ("unknown-roster-clock", "roster_missing"),
    ("duplicate-book", "insufficient_distinct_books"),
])
def test_rejected_capture_boundaries_preserve_fixture_rows(tmp_path, session, case, reason):
    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(
        session, tmp_path, now - timedelta(seconds=10), case=case,
    )
    result = orchestrate_local_opportunity_batch(
        session, protocol_id="boundary-tests", provider_id="fixture-provider",
        batch_id=str(uuid4()), discovery_capture_ids=[discovery], body_store=tmp_path,
        as_of=now, max_fixtures=2,
    )
    assert result["accounted_fixture_ids"] == [fixture_id]
    row = result["fixtures"][0]
    assert reason in row["rejection_reasons"]
    assert not row["inference_eligible"]
    assert row["forecast_evidence_key"] is None
    assert sum(event["primary_decision"] for event in result["events"]) == 1


def test_overlapping_discovery_rows_keep_one_fixture_primary(tmp_path, session):
    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(
        session, tmp_path, now - timedelta(seconds=10), duplicate=True,
    )
    result = orchestrate_local_opportunity_batch(
        session, protocol_id="boundary-tests", provider_id="fixture-provider",
        batch_id=str(uuid4()), discovery_capture_ids=[discovery], body_store=tmp_path,
        as_of=now, max_fixtures=2,
    )
    assert result["input_fixture_count"] == 2
    assert result["accounted_fixture_ids"] == [fixture_id]
    assert result["counts"]["discovered"] == result["counts"]["primary_decisions"] == 1
    assert len(result["fixtures"]) == 1
    assert sum(event["event_type"] == "discovery" for event in result["events"]) == 2
    assert sum(event["primary_decision"] for event in result["events"]) == 1


def test_missed_origin_deadline_does_not_invoke_model(tmp_path, session):
    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(session, tmp_path, now - timedelta(seconds=61))
    calls = []
    result = orchestrate_local_opportunity_batch(
        session, protocol_id="boundary-tests", provider_id="fixture-provider",
        batch_id=str(uuid4()), discovery_capture_ids=[discovery], body_store=tmp_path,
        as_of=now, max_fixtures=1, predictor=lambda candidate: calls.append(candidate),
    )
    assert calls == []
    assert result["accounted_fixture_ids"] == [fixture_id]
    assert "origin_deadline_missed" in result["fixtures"][0]["rejection_reasons"]


@pytest.mark.parametrize("failure", ["existing_foreign", "new_foreign", "database_failure"])
def test_forecast_link_rejection_preserves_ledger_and_rolls_back_callback(tmp_path, session, failure):
    from betting_app.services.forecast_evidence import append_forecast_evidence, load_forecast_evidence
    from betting_app.tests.test_forecast_evidence import evidence

    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(session, tmp_path, now - timedelta(seconds=10))
    foreign_key = str(uuid4())
    if failure == "existing_foreign":
        append_forecast_evidence(session, evidence_key=foreign_key, payload=evidence())

    def predictor(candidate):
        if failure != "existing_foreign":
            append_forecast_evidence(session, evidence_key=foreign_key, payload=evidence())
        if failure == "database_failure":
            session.execute(text("SELECT 1 / 0"))
        return foreign_key

    result = orchestrate_local_opportunity_batch(
        session, protocol_id="boundary-tests", provider_id="fixture-provider",
        batch_id=str(uuid4()), discovery_capture_ids=[discovery], body_store=tmp_path,
        as_of=now, max_fixtures=1, predictor=predictor,
    )
    row = result["fixtures"][0]
    assert result["accounted_fixture_ids"] == [fixture_id]
    assert row["inference_eligible"] and not row["forward_admitted"]
    assert row["forecast_evidence_key"] is None
    reason = "model_failure" if failure == "database_failure" else "forecast_envelope_fixture_mismatch"
    assert reason in row["rejection_reasons"]
    assert (load_forecast_evidence(session, foreign_key) is not None) == (failure == "existing_foreign")
    assert session.execute(text("SELECT COUNT(*) FROM opportunity_events WHERE primary_decision")).scalar_one() == 1


@pytest.mark.parametrize("schedule_state", ["future", "missing"])
def test_later_discovery_cannot_unfreeze_an_existing_primary(tmp_path, session, schedule_state):
    from betting_app.services.forecast_evidence import load_source_capture

    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(session, tmp_path, now - timedelta(seconds=10))
    arguments = dict(
        session=session, protocol_id="frozen-origin-tests", provider_id="fixture-provider",
        body_store=tmp_path, as_of=now, max_fixtures=1,
    )
    first = orchestrate_local_opportunity_batch(
        **arguments, batch_id=str(uuid4()), discovery_capture_ids=[discovery],
    )
    fixture = json.loads(load_source_capture(session, discovery, body_store=tmp_path)["body_bytes"])["fixtures"][0]
    fixture["schedule_capture_ids"] = []
    if schedule_state == "future":
        revised = _capture_json(
            session, tmp_path, f"revised:{fixture_id}", "schedule",
            {"scheduled_start_at": (now + timedelta(hours=3)).isoformat(), "status": "rescheduled"}, now,
        )
        fixture["schedule_capture_ids"] = [revised.capture_id]
    later_feed = _capture_json(
        session, tmp_path, f"later-feed:{fixture_id}", "discovery", {"fixtures": [fixture]}, now,
    )
    later = orchestrate_local_opportunity_batch(
        **arguments, batch_id=str(uuid4()), discovery_capture_ids=[later_feed.capture_id],
    )
    assert later["fixtures"][0]["primary_origin_at"] == first["fixtures"][0]["primary_origin_at"]
    assert later["fixtures"][0]["primary_status"] == first["fixtures"][0]["primary_status"]
    assert later["counts"]["primary_decisions"] == 1
    assert later["pending_fixture_ids"] == []
    assert session.execute(text("SELECT COUNT(*) FROM opportunity_events WHERE primary_decision")).scalar_one() == 1


def test_model_completion_after_deadline_is_not_an_eligible_decision(tmp_path, session, monkeypatch):
    from betting_app.services import forecast_evidence

    now = datetime.now(UTC)
    fixture_id, discovery = _boundary_inputs(session, tmp_path, now - timedelta(seconds=10))
    current = [now]

    class ClockMeta(type):
        def __instancecheck__(cls, value):
            return isinstance(value, datetime)

    class Clock(datetime, metaclass=ClockMeta):
        @classmethod
        def now(cls, tz=None):
            return current[0]

    monkeypatch.setattr(forecast_evidence, "datetime", Clock)

    def delayed_model(candidate):
        current[0] += timedelta(seconds=61)
        return None

    result = forecast_evidence.orchestrate_local_opportunity_batch(
        session, protocol_id="completion-deadline", provider_id="fixture-provider",
        batch_id=str(uuid4()), discovery_capture_ids=[discovery], body_store=tmp_path,
        as_of=now, max_fixtures=1, predictor=delayed_model,
    )
    row = result["fixtures"][0]
    assert row["fixture_id"] == fixture_id
    assert not row["inference_eligible"]
    assert "origin_deadline_missed" in row["rejection_reasons"]
    assert row["forecast_evidence_key"] is None
