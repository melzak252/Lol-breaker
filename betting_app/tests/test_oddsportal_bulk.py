"""Offline end-to-end bulk acquisition and interruption durability contracts."""
import asyncio
import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import httpx
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from src.utils import oddsportal_collection as c
from src.utils.oddsportal_bulk import download_inventory

FIXTURES = Path(__file__).parent / "fixtures/oddsportal"
SHELL = json.loads((FIXTURES / "current_event_shell.json").read_text())
HISTORY = json.loads((FIXTURES / "current_price_history.json").read_text())
IDS = ["ne8lT6bs", "AbCd1234", "EfGh5678", "IjKl9012"]


def encrypted(value):
    iv = bytes(range(16))
    key = hashlib.pbkdf2_hmac("sha256", c.PASSWORD.encode(), c.SALT.encode(), 1000, dklen=32)
    ciphertext = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(json.dumps(value).encode(), 16))
    return base64.b64encode(base64.b64encode(ciphertext) + b":" + iv.hex().encode()).decode()


def shell_for(event_id):
    shell = copy.deepcopy(SHELL)
    shell["d"]["eventData"]["id"] = event_id
    shell["d"]["eventBody"].pop("tabs", None)
    return shell


def odds_for(event_id, *, missing_clock=False):
    market = {
        "bettingTypeId": 3, "scopeId": 2, "handicapTypeId": 0,
        "handicapValue": 0, "mixedParameterId": 0, "mixedParameterName": None,
        "isBack": True, "outcomeId": list(HISTORY["d"]["history"]["back"]),
        "odds": {"165": [1.4, 2.75]}, "openingOdd": {"165": [1.55, 2.29]},
        "openingChangeTime": {"165": [1765628964, 1765628964]},
        "changeTime": {"165": [1765698443, 1765698443]}, "act": {"165": False},
    }
    if missing_clock:
        market.pop("openingChangeTime")
        market.pop("changeTime")
    return {"d": {"encodeventId": event_id, "bt": 3, "sc": 2,
                  "oddsdata": {"back": {"E-3-2-0-0-0": market}}}}


def inventory_file(tmp_path, ids=IDS, *, complete=False):
    path = tmp_path / "inventory.json"
    rows = [{"event_id": event_id, "url": f"{c.BASE_URL}/event-{event_id}/",
             "home_team": "Old home", "away_team": "Old away",
             "source_occurrences": [{"kind": "legacy", "row": index}],
             "source_season": {"id": "2026", "name": "2026"},
             "legacy_row": {"url": f"{c.BASE_URL}/event-{event_id}/", "odds1_sts_open": "9.9"},
             "odds1_sts_open": 9.9, "avg_open_home": 9.9}
            for index, event_id in enumerate(ids)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "inventory-metadata.json").write_text(json.dumps({
        "inventory_complete": complete, "inventory_coverage_ledger_path": "retained-coverage.json",
        "inventory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unique_events": len(rows), "max_matches": None,
    }), encoding="utf-8")
    return path


def run_for(tmp_path, *, concurrency=1, with_history=False, with_markets=False):
    return c.CollectionRun.create("bulk", tmp_path / "data", mode="archive", delay_seconds=0,
                                  concurrency=concurrency, with_history=with_history,
                                  with_markets=with_markets)


def transport_for(calls, *, bad_event=None, denied_history=False, interrupt_event=None,
                  empty_event=None, missing_clock_event=None, activity=None, gzip_shell=False):
    async def handler(request):
        calls.append(str(request.url))
        path = request.url.path
        if path == "/proxy/api/home-data":
            return httpx.Response(200, text=encrypted({"geoIPcode": "PL"}))
        for event_id in IDS:
            if path == f"/proxy/ajax-event-data-shell/{event_id}/0/":
                if event_id == interrupt_event:
                    raise asyncio.CancelledError("simulated process interruption")
                if activity is not None:
                    activity["active"] += 1
                    activity["maximum"] = max(activity["maximum"], activity["active"])
                    await asyncio.sleep(0.005)
                    activity["active"] -= 1
                if gzip_shell:
                    return httpx.Response(200, content=gzip.compress(encrypted(shell_for(event_id)).encode()),
                                          headers={"content-type": "application/json", "content-encoding": "gzip"})
                return httpx.Response(200, text=encrypted(shell_for(event_id)))
            if path == f"/proxy/match-event/1-36-{event_id}-3-2-yjd10.dat":
                payload = odds_for(event_id, missing_clock=event_id == missing_clock_event)
                if event_id == bad_event:
                    payload["d"].pop("oddsdata")
                if event_id == empty_event:
                    market = payload["d"]["oddsdata"]["back"]["E-3-2-0-0-0"]
                    market["odds"] = {}
                    market["openingOdd"] = {}
                return httpx.Response(200, text=encrypted(payload))
            if path == f"/proxy/match-event-history/36-{event_id}-3-2-0-165/":
                if denied_history:
                    return httpx.Response(429, content=b"Rate limited")
                return httpx.Response(200, text=encrypted(HISTORY))
            if path == f"/proxy/ajax-event-nav/{event_id}/0/":
                return httpx.Response(200, text=encrypted({"s": 1, "d": {"nav": {"3": {"2": ["165"]}}}}))
        raise AssertionError(f"Unexpected source route: {request.url}")
    return httpx.MockTransport(handler)


def settled_rows(run):
    return [json.loads(line) for line in (run.derived_dir / "settled-records.jsonl").read_text().splitlines()]


def test_every_input_event_is_attempted_failure_does_not_suppress_later_events(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:3], complete=False)
        source_bytes = inventory.read_bytes()
        run = run_for(tmp_path, with_history=True, with_markets=True)
        calls = []
        async with httpx.AsyncClient(transport=transport_for(
            calls, bad_event=IDS[1], missing_clock_event=IDS[2])) as client:
            manifest = await download_inventory(client, run, inventory, checkpoint_every=1)
        records = json.loads(run.json_path.read_text())
        assert [row["event_id"] for row in records] == IDS[:3]
        assert [row["odds_status"] for row in records] == ["available", "failed", "available"]
        assert all(row["odds_reason"] for row in records)
        assert manifest["total_expected_ids"] == manifest["attempted_events"] == manifest["completed_events"] == 3
        assert manifest["not_requested_events"] == 0
        assert manifest["inventory_complete"] is False
        assert manifest["acquisition_scope"] == "all_events_in_validated_inventory"
        assert manifest["whole_history_claimed"] is False
        assert manifest["inventory_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert Path(manifest["inventory_snapshot_path"]).read_bytes() == source_bytes
        assert {row["event_id"] for row in settled_rows(run)} == set(IDS[:3])
        for event_id in IDS[:3]:
            assert sum(f"/ajax-event-data-shell/{event_id}/" in url for url in calls) == 1
        assert records[0]["price_history"]["165"]["status"] == "available"
        assert records[0]["market_navigation"]["status"] == "available"
        assert records[2]["odds1_sts_open_at"] is None
        assert records[2]["odds1_sts_close_at"] is None
        assert records[2]["source_occurrences"] == [{"kind": "legacy", "row": 2}]
        assert records[2]["source_season"] == {"id": "2026", "name": "2026"}
        progress = json.loads((run.reporting_dir / "progress.json").read_text())
        assert progress["completed"] == 3 and progress["pending"] == 0
        assert progress["status_counts"] == {"available": 2, "not_available": 0, "failed": 1, "not_requested": 0}
    asyncio.run(scenario())


def test_history_denial_keeps_obtained_prices_and_exposes_every_remaining_id(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path)
        run = run_for(tmp_path, with_history=True)
        calls = []
        async with httpx.AsyncClient(transport=transport_for(calls, denied_history=True)) as client:
            manifest = await download_inventory(client, run, inventory, checkpoint_every=20)
        records = json.loads(run.json_path.read_text())
        assert records[0]["odds_status"] == "available"
        assert records[0]["odds1_sts_open"] == 1.55
        assert records[0]["price_history"]["165"]["capture"]["http_status"] == 429
        assert len(records) == len(IDS) == len(settled_rows(run))
        assert [row["odds_status"] for row in records[1:]] == ["not_requested"] * 3
        assert all("access_denied" in row["odds_reason"] for row in records[1:])
        assert all(row["odds1_sts_open"] is None for row in records[1:])
        assert not any(f"/ajax-event-data-shell/{event_id}/" in url for event_id in IDS[1:] for url in calls)
        assert manifest["attempted_events"] == 1
        assert manifest["completed_events"] == 4 and manifest["not_requested_events"] == 3
        assert manifest["stop_reason"] == "access_denied"
        progress = json.loads((run.reporting_dir / "progress.json").read_text())
        assert progress["pending"] == 0 and progress["completed"] == 4
    asyncio.run(scenario())


def test_interruption_flushes_successful_rows_and_updates_pending_progress(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:3])
        run = run_for(tmp_path)
        calls = []
        async with httpx.AsyncClient(transport=transport_for(calls, interrupt_event=IDS[1])) as client:
            with pytest.raises(asyncio.CancelledError):
                await download_inventory(client, run, inventory, checkpoint_every=100)
        rows = settled_rows(run)
        assert len(rows) == 1 and rows[0]["event_id"] == IDS[0]
        assert rows[0]["odds1_sts_open"] == 1.55
        assert not run.json_path.exists() and not run.csv_path.exists()
        progress = json.loads((run.reporting_dir / "progress.json").read_text())
        assert progress["status"] == "INTERRUPTED"
        assert progress["completed"] == 1 and progress["pending"] == 2
        assert progress["status_counts"]["available"] == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("invalid", [
    [{"event_id": IDS[0], "url": f"{c.BASE_URL}/event-{IDS[0]}/"}] * 2,
    [{"event_id": IDS[0], "url": f"https://example.org/event-{IDS[0]}/"}],
    [{"event_id": IDS[0], "url": f"{c.BASE_URL}/event-{IDS[1]}/"}],
    {"records": []}, [None],
])
def test_invalid_inventory_is_rejected_before_any_source_request(tmp_path, invalid):
    async def scenario():
        inventory = tmp_path / "invalid.json"
        inventory.write_text(json.dumps(invalid), encoding="utf-8")
        run = run_for(tmp_path)
        calls = []
        async with httpx.AsyncClient(transport=transport_for(calls)) as client:
            with pytest.raises(ValueError):
                await download_inventory(client, run, inventory)
        assert calls == [] and run.captures == []
    asyncio.run(scenario())


@pytest.mark.parametrize("gzip_shell", [False, True])
def test_malformed_moneyline_retains_actual_shell_metadata_and_capture_not_old_prices(tmp_path, gzip_shell):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:1])
        run = run_for(tmp_path)
        calls = []
        async with httpx.AsyncClient(transport=transport_for(calls, bad_event=IDS[0], gzip_shell=gzip_shell)) as client:
            await download_inventory(client, run, inventory)
        row = json.loads(run.json_path.read_text())[0]
        assert row["odds_status"] == "failed" and row["odds_reason"]
        assert row["home_team"] == "Sentinels" and row["away_team"] == "Cloud9"
        assert row["date"] == datetime.fromtimestamp(SHELL["d"]["eventBody"]["startDate"], timezone.utc).isoformat()
        assert row["homeResult"] == "1" and row["awayResult"] == "2"
        assert row["event_metadata_status"] == "parsed_current_shell"
        assert row["odds1_sts_open"] is None and row["avg_open_home"] is None
        assert "openingsOdds" not in row and "closingOdds" not in row
        assert row["legacy_row"]["odds1_sts_open"] == "9.9"
        assert row["source_occurrences"] == [{"kind": "legacy", "row": 0}]
        assert json.loads(Path(row["shell_decoded_payload_path"]).read_text()) == shell_for(IDS[0])
        body = Path(row["shell_response_path"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == row["shell_response_sha256"]
        assert row["shell_capture_started_at"] and row["shell_capture_completed_at"]
        assert len(row["event_acquisition_captures"]) == 2
        assert len(calls) == 3  # No metadata rescue/fallback HTTP request.
    asyncio.run(scenario())


def test_no_prices_is_not_available_with_explicit_reason_and_complete_inventory(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:1], complete=True)
        run = run_for(tmp_path)
        async with httpx.AsyncClient(transport=transport_for([], empty_event=IDS[0])) as client:
            manifest = await download_inventory(client, run, inventory)
        row = json.loads(run.json_path.read_text())[0]
        assert row["odds_status"] == "not_available" and row["odds_reason"]
        assert row["odds1_sts_open"] is None
        assert manifest["inventory_complete"] is True
        assert manifest["inventory_coverage_ledger_path"] == "retained-coverage.json"
    asyncio.run(scenario())


def test_concurrent_event_tasks_are_bounded_without_dropping_ids(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path)
        run = run_for(tmp_path, concurrency=2)
        activity = {"active": 0, "maximum": 0}
        async with httpx.AsyncClient(transport=transport_for([], activity=activity)) as client:
            manifest = await download_inventory(client, run, inventory, checkpoint_every=2)
        assert activity["maximum"] == 2
        assert manifest["completed_events"] == len(IDS)
        assert len(settled_rows(run)) == len(IDS)
    asyncio.run(scenario())


def test_unbound_completeness_metadata_cannot_claim_complete_source_inventory(tmp_path):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:1], complete=True)
        metadata_path = tmp_path / "inventory-metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["inventory_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        run = run_for(tmp_path)
        async with httpx.AsyncClient(transport=transport_for([])) as client:
            manifest = await download_inventory(client, run, inventory)
        assert manifest["inventory_complete"] is None
        assert manifest["inventory_completeness_status"] == "unverified_metadata_binding"
        assert manifest["whole_history_claimed"] is False
    asyncio.run(scenario())


def test_unexpected_worker_error_interrupts_without_losing_settled_prices(tmp_path, monkeypatch):
    async def scenario():
        inventory = inventory_file(tmp_path, IDS[:3])
        run = run_for(tmp_path)
        original_fetch = c.fetch_detailed_odds

        async def broken_fetch(client, collection, event_id):
            if event_id == IDS[1]:
                raise RuntimeError("unexpected worker defect")
            return await original_fetch(client, collection, event_id)

        monkeypatch.setattr(c, "fetch_detailed_odds", broken_fetch)
        async with httpx.AsyncClient(transport=transport_for([])) as client:
            with pytest.raises(RuntimeError):
                await download_inventory(client, run, inventory)
        rows = settled_rows(run)
        assert [row["event_id"] for row in rows] == IDS[:1]
        assert rows[0]["odds_status"] == "available"
        assert rows[0]["odds1_sts_open"] == 1.55
        progress = json.loads((run.reporting_dir / "progress.json").read_text())
        assert progress["status"] == "INTERRUPTED"
        assert progress["completed"] == 1 and progress["pending"] == 2

    asyncio.run(scenario())
