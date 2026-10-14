"""Offline public-proxy transport, provenance, repair and safe-output regressions."""
import asyncio
import base64
import copy
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from src.utils import oddsportal_collection as collector

FIXTURES = Path(__file__).parent / "fixtures/oddsportal"
SHELL = json.loads((FIXTURES / "current_event_shell.json").read_text())
HISTORY = json.loads((FIXTURES / "current_price_history.json").read_text())
EVENT_ID = SHELL["d"]["eventData"]["id"]


@pytest.fixture
def payload():
    # Controlled prices/source epochs exercise the current HTTP protocol with
    # independently encrypted entities; they are not claimed as live observations.
    return {"d": {"encodeventId": EVENT_ID, "bt": 3, "sc": 2, "oddsdata": {"back": {
        "E-3-2-0-0-0": {
            "bettingTypeId": 3, "scopeId": 2, "handicapTypeId": 0,
            "handicapValue": 0, "mixedParameterId": 0, "mixedParameterName": None,
            "isBack": True, "outcomeId": list(HISTORY["d"]["history"]["back"]),
            "odds": {"165": [1.4, 2.75]}, "openingOdd": {"165": [1.55, 2.29]},
            "openingChangeTime": {"165": [1765628964, 1765628964]},
            "changeTime": {"165": [1765698443, 1765698443]}, "act": {"165": False},
            "history": {"165": [{"opaque": "retained"}]}, "volume": {"165": [12, 15]},
        }
    }}}}


def encrypted(payload):
    iv = bytes(range(16))
    key = hashlib.pbkdf2_hmac("sha256", collector.PASSWORD.encode(), collector.SALT.encode(), 1000, dklen=32)
    ciphertext = AES.new(key, AES.MODE_CBC, iv).encrypt(pad(json.dumps(payload).encode(), 16))
    return base64.b64encode(base64.b64encode(ciphertext) + b":" + iv.hex().encode()).decode()


def proxy_transport(payload, *, shell=SHELL, history=HISTORY, geo="PL", calls=None):
    """Only the observed public routes return fixtures; unexpected paths fail."""
    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        path = request.url.path
        if path == "/proxy/api/home-data":
            return httpx.Response(200, text=encrypted({"geoIPcode": geo}))
        if path == f"/proxy/ajax-event-data-shell/{EVENT_ID}/0/":
            return httpx.Response(200, text=encrypted(shell))
        if path == f"/proxy/match-event/1-36-{EVENT_ID}-3-2-yjd10.dat":
            assert request.url.params["geo"] == geo
            assert request.url.params["lang"] == "pl"
            return httpx.Response(200, text=encrypted(payload))
        if path == f"/proxy/match-event-history/36-{EVENT_ID}-3-2-0-165/":
            assert request.url.params["geo"] == geo
            return httpx.Response(200, text=encrypted(history))
        if path == f"/proxy/ajax-event-nav/{EVENT_ID}/0/":
            return httpx.Response(200, text=encrypted({"s": 1, "d": {"nav": {"3": {"2": ["165"]}}}}))
        if path == f"/event-{EVENT_ID}/":
            return httpx.Response(200, text="<html><title>Match page</title></html>")
        raise AssertionError(f"Unexpected request route: {request.url}")
    return httpx.MockTransport(handler)


def new_run(tmp_path, mode="refresh", with_history=False, with_markets=False):
    return collector.CollectionRun.create("test-run", tmp_path / "data", mode=mode, delay_seconds=0, with_history=with_history, with_markets=with_markets)


def source_csv(tmp_path, extra=False, duplicate=False):
    path = tmp_path / "legacy.csv"
    text = f"url,home_team,away_team,odds1_sts_open,odds2_sts_open\nhttps://www.oddsportal.com/event-{EVENT_ID}/,Hanwha Life,T1,1.70,2.10\n"
    if extra:
        text += "https://www.oddsportal.com/event-Other123/,Alpha,Beta,1.80,2.20\n"
    if duplicate:
        text += text.splitlines()[1] + "\n"
    path.write_text(text)
    return path


def test_public_shell_details_csv_preserve_source_order_datetimes_and_raw_entities(tmp_path, payload):
    async def scenario():
        run = new_run(tmp_path)
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, calls=calls)) as client:
            result = await collector.fetch_detailed_odds(client, run, EVENT_ID)
        collector.write_outputs(run, [result])
        assert result["home_team"] == "Sentinels"
        assert result["away_team"] == "Cloud9"
        assert result["openingsOdds"]["165"] == [1.55, 2.29]
        assert result["closingOdds"]["165"] == [1.4, 2.75]
        assert result["source_bookmaker_active"]["165"] is False
        assert result["odds1_sts_open_at"] == "2025-12-13T12:29:24+00:00"
        assert result["odds2_sts_close_at"] == "2025-12-14T07:47:23+00:00"
        assert result["odds_capture_completed_at"] != result["odds2_sts_close_at"]
        assert len(calls) == 3
        captures = [json.loads(line) for line in (run.raw_dir / "captures.jsonl").read_text().splitlines()]
        for capture in captures:
            body = Path(capture["body_path"]).read_bytes()
            assert capture["body_sha256"] == hashlib.sha256(body).hexdigest()
            assert datetime.fromisoformat(capture["capture_started_at"]) <= datetime.fromisoformat(capture["capture_completed_at"])
        assert Path(result["odds_response_path"]).read_bytes() == encrypted(payload).encode()
        assert json.loads(Path(result["shell_decoded_payload_path"]).read_text()) == SHELL
        decoded = json.loads(Path(result["odds_decoded_payload_path"]).read_text())
        assert decoded["d"]["oddsdata"]["back"]["E-3-2-0-0-0"]["history"] == {"165": [{"opaque": "retained"}]}
        with run.csv_path.open(newline="") as stream:
            saved = next(csv.DictReader(stream))
        assert saved["odds1_sts_open"] == "1.55"
        assert saved["odds2_sts_open"] == "2.29"
        assert saved["odds1_sts_open_at"] == result["odds1_sts_open_at"]
        assert saved["quote_timing_semantics"] == "provider_open_and_last_not_certified_close"
    asyncio.run(scenario())


def test_exact_market_identity_survives_current_http_decode(tmp_path, payload):
    async def scenario():
        altered = copy.deepcopy(payload)
        market = altered["d"]["oddsdata"]["back"]["E-3-2-0-0-0"]
        wrong = copy.deepcopy(market)
        wrong["odds"]["165"] = [9.0, 1.01]
        altered["d"]["oddsdata"]["back"] = {"E-3-2-0-1-0": wrong, "E-3-2-0-0-0": market}
        async with httpx.AsyncClient(transport=proxy_transport(altered)) as client:
            result = await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert result["closingOdds"]["165"] == [1.4, 2.75]
    asyncio.run(scenario())


def test_wrong_shell_event_stops_before_market_request(tmp_path, payload):
    async def scenario():
        shell = copy.deepcopy(SHELL)
        shell["d"]["eventData"]["id"] = "Other123"
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, shell=shell, calls=calls)) as client:
            with pytest.raises(ValueError, match="shell identity"):
                await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert len(calls) == 2
    asyncio.run(scenario())


def test_wrong_market_event_cannot_be_accepted_after_correct_shell(tmp_path, payload):
    async def scenario():
        payload["d"]["encodeventId"] = "Other123"
        async with httpx.AsyncClient(transport=proxy_transport(payload)) as client:
            with pytest.raises(ValueError, match="event identity"):
                await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
    asyncio.run(scenario())


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_denial_stops_entire_refresh_without_retry_or_legacy_recertification(tmp_path, status):
    async def scenario():
        calls = []
        source = source_csv(tmp_path, extra=True)
        original = source.read_bytes()
        run = new_run(tmp_path)
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(status, content=b"Access denied")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.refresh_csv(client, run, source)
        assert len(calls) == 1
        assert manifest["status"] == "FAILED"
        assert manifest["stop_reason"] == "access_denied"
        assert manifest["failed_rows"] == 1
        assert manifest["not_requested_rows"] == 1
        assert source.read_bytes() == original
        records = json.loads(run.json_path.read_text())
        assert records[0]["odds_status"] == "failed"
        assert records[0]["odds1_sts_open"] is None
        assert records[0]["legacy_row"]["odds1_sts_open"] == "1.70"
        assert records[1]["odds_status"] == "not_requested_access_denied"
        assert records[1]["odds1_sts_open"] is None
        capture = json.loads((run.raw_dir / "captures.jsonl").read_text().splitlines()[0])
        assert capture["http_status"] == status
        assert Path(capture["body_path"]).read_bytes() == b"Access denied"
    asyncio.run(scenario())


def test_refresh_keeps_original_separate_stages_and_participant_discrepancy(tmp_path, payload):
    async def scenario():
        source = source_csv(tmp_path, extra=True)
        original = source.read_bytes()
        run = new_run(tmp_path)
        async with httpx.AsyncClient(transport=proxy_transport(payload)) as client:
            manifest = await collector.refresh_csv(client, run, source, max_matches=1)
        records = json.loads(run.json_path.read_text())
        assert source.read_bytes() == original
        assert records[0]["legacy_identity_certified"] is False
        assert records[0]["home_team"] == "Sentinels"
        assert records[0]["away_team"] == "Cloud9"
        assert records[0]["participant_discrepancy"]["prior"] == ["Hanwha Life", "T1"]
        assert records[0]["odds1_sts_open"] == 1.55
        assert records[1]["odds_status"] == "legacy_not_refreshed"
        assert records[1]["quote_timing_semantics"] == "unknown_legacy_export"
        assert manifest["selected_for_refresh"] == 1
        assert run.raw_dir == tmp_path / "data/01_raw/oddsportal/test-run"
        assert run.json_path == tmp_path / "data/02_intermediate/oddsportal/test-run/matches.json"
        assert run.manifest_path == tmp_path / "data/08_reporting/oddsportal/test-run/run.json"
        assert json.loads(run.manifest_path.read_text())["source_sha256"] == hashlib.sha256(original).hexdigest()
        assert next(run.raw_dir.glob("input-*.csv")).read_bytes() == original
        with pytest.raises(ValueError, match="new run"):
            new_run(tmp_path)
    asyncio.run(scenario())


def test_reversed_legacy_participants_are_reported_never_silently_swapped(tmp_path, payload):
    async def scenario():
        source = source_csv(tmp_path)
        source.write_text(source.read_text().replace("Hanwha Life,T1", "Cloud9,Sentinels"))
        run = new_run(tmp_path)
        async with httpx.AsyncClient(transport=proxy_transport(payload)) as client:
            await collector.refresh_csv(client, run, source)
        row = json.loads(run.json_path.read_text())[0]
        assert row["participant_order_comparison"] == "reversed"
        assert row["home_team"] == "Sentinels"
        assert row["odds1_sts_open"] == 1.55
        assert row["odds2_sts_open"] == 2.29
        assert row["legacy_row"]["home_team"] == "Cloud9"
        assert row["legacy_identity_certified"] is False
    asyncio.run(scenario())


def test_malformed_payload_is_failure_not_empty_success(tmp_path):
    async def scenario():
        source = source_csv(tmp_path)
        run = new_run(tmp_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))) as client:
            manifest = await collector.refresh_csv(client, run, source)
        assert manifest["status"] == "FAILED"
        assert manifest["failed_rows"] == 1
        row = json.loads(run.json_path.read_text())[0]
        assert row["odds_status"] == "failed"
        assert row["odds1_sts_open"] is None
        assert manifest["captures"] == 1
    asyncio.run(scenario())


def test_duplicate_headers_and_conflicting_participants_are_rejected(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text(f"url,avg_open_home,avg_open_home\n/event-{EVENT_ID}/,1.55,2.29\n")
    with pytest.raises(ValueError, match="duplicate"):
        collector.read_legacy_csv(path)
    prior = {"event_id": EVENT_ID, "home_team": "Sentinels", "away_team": "Cloud9"}
    with pytest.raises(ValueError, match="Conflicting participants"):
        collector.merge_matches([prior], [{**prior, "home_team": "Cloud9", "away_team": "Sentinels"}])


def test_duplicate_refresh_rows_fetch_exact_event_once(tmp_path, payload):
    async def scenario():
        source = source_csv(tmp_path, duplicate=True)
        calls = []
        run = new_run(tmp_path)
        async with httpx.AsyncClient(transport=proxy_transport(payload, calls=calls)) as client:
            manifest = await collector.refresh_csv(client, run, source)
        assert len(calls) == 3
        assert manifest["rows"] == 2
        assert manifest["refreshed_rows"] == 2
        records = json.loads(run.json_path.read_text())
        assert records[0]["home_team"] == records[1]["home_team"] == "Sentinels"
    asyncio.run(scenario())


def test_missing_provider_clock_remains_null(tmp_path, payload):
    async def scenario():
        payload["d"]["oddsdata"]["back"]["E-3-2-0-0-0"].pop("changeTime")
        async with httpx.AsyncClient(transport=proxy_transport(payload)) as client:
            result = await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert result["odds1_sts_close_at"] is None
        assert result["closingOdds"]["165"] == [1.4, 2.75]
    asyncio.run(scenario())


def test_single_event_probe_fetches_only_page_bootstrap_shell_and_details(tmp_path, payload):
    async def scenario():
        calls = []
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=proxy_transport(payload, calls=calls)) as client:
            manifest = await collector.scrape(client, run, event_url=f"https://www.oddsportal.com/event-{EVENT_ID}/")
        assert len(calls) == 4
        assert not any("results" in url for url in calls)
        assert manifest["captures"] == 4
        row = json.loads(run.json_path.read_text())[0]
        assert row["event_id"] == EVENT_ID
        assert row["home_team"] == "Sentinels"
        assert row["date"] is not None
        assert row["event_metadata_status"] == "parsed_current_shell"
    asyncio.run(scenario())


def test_bootstrap_geo_is_used_without_forced_polish_override(tmp_path, payload):
    async def scenario():
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, geo="BG", calls=calls)) as client:
            result = await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert result["source_geo"] == "BG"
        assert "geo=BG" in calls[-1]
    asyncio.run(scenario())


def test_missing_bootstrap_geo_prevents_shell_and_price_requests(tmp_path, payload):
    async def scenario():
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, geo=None, calls=calls)) as client:
            with pytest.raises(ValueError, match="geoIPcode"):
                await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert len(calls) == 1
    asyncio.run(scenario())


def test_optional_history_retains_returned_rows_without_claiming_complete_timeline(tmp_path, payload):
    async def scenario():
        run = new_run(tmp_path, with_history=True)
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, calls=calls)) as client:
            result = await collector.fetch_detailed_odds(client, run, EVENT_ID)
        assert len(calls) == 4
        history = result["price_history"]["165"]
        assert history["status"] == "available"
        assert [row["outcome_index"] for row in history["rows"]] == [0, 1]
        assert [row["odds"] for row in history["rows"]] == [4.3, 1.18]
        assert all(row["source_change_at"] is not None for row in history["rows"])
        assert "not_certified_complete_or_terminal" in result["history_timing_semantics"]
        assert json.loads(Path(history["history_decoded_payload_path"]).read_text()) == HISTORY
    asyncio.run(scenario())


def test_transport_retry_is_bounded(tmp_path):
    async def scenario():
        calls = []
        def handler(request):
            calls.append(request.url)
            raise httpx.ConnectError("offline", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(httpx.ConnectError):
                await collector.safe_request(client, new_run(tmp_path), "https://www.oddsportal.com/proxy/api/home-data", attempts=2)
        assert len(calls) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize("run_id", ["../escape", "", "x/y", ".", ".."])
def test_run_identity_cannot_escape_artifact_roots(tmp_path, run_id):
    with pytest.raises(ValueError):
        collector.CollectionRun.create(run_id, tmp_path, mode="archive", delay_seconds=0)


def test_html_protocol_mismatch_is_retained_and_stops_remaining_requests(tmp_path):
    async def scenario():
        source = source_csv(tmp_path, extra=True)
        run = new_run(tmp_path)
        calls = []
        def handler(request):
            calls.append(request.url)
            return httpx.Response(200, text="<html><title>OddsPortal</title></html>", headers={"content-type": "text/html"})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.refresh_csv(client, run, source)
        assert len(calls) == 1
        assert manifest["status"] == "FAILED"
        assert manifest["stop_reason"] == "provider_protocol_changed"
        assert "HTML is not an encrypted" in manifest["failures"][0]["error"]
        assert json.loads(run.json_path.read_text())[1]["odds_status"] == "not_requested_provider_protocol_changed"
        assert manifest["captures"] == 1
    asyncio.run(scenario())


def test_redirect_to_unrelated_sport_is_not_followed_or_accepted(tmp_path):
    async def scenario():
        calls = []
        def handler(request):
            calls.append(request.url)
            return httpx.Response(302, headers={"location": "https://www.oddsportal.com/pl/results/football/"}, content=b"moved")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, new_run(tmp_path, mode="archive"))
        assert len(calls) == 1
        assert str(calls[0]) == "https://www.oddsportal.com/pl/results/esports/"
        assert manifest["status"] == "FAILED"
        assert manifest["rows"] == 0
        assert manifest["captures"] == 1
    asyncio.run(scenario())


def test_refresh_cli_writes_manifest_and_respects_bound(tmp_path, payload, monkeypatch):
    source = source_csv(tmp_path, extra=True)
    original_client = httpx.AsyncClient
    transport = proxy_transport(payload)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))
    result = collector.refresh_main([
        "--run-id", "cli-run", "--input-csv", str(source), "--max-matches", "1",
        "--delay-seconds", "0", "--output-dir", str(tmp_path / "data"),
    ])
    assert result == 0
    manifest = json.loads((tmp_path / "data/08_reporting/oddsportal/cli-run/run.json").read_text())
    assert manifest["rows"] == 2
    assert manifest["selected_for_refresh"] == 1
    assert manifest["concurrency"] == 1
    assert manifest["source_authenticated"] is False


def test_rainbet_and_unknown_provider_payload_are_retained(tmp_path, payload):
    async def scenario():
        market = payload["d"]["oddsdata"]["back"]["E-3-2-0-0-0"]
        market["odds"]["1193"] = [1.6, 2.4]
        market["odds"]["9999"] = [1.7, 2.3]
        async with httpx.AsyncClient(transport=proxy_transport(payload)) as client:
            result = await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert result["odds1_rainbet_close"] == 1.6
        assert result["odds2_rainbet_close"] == 2.4
        retained = json.loads(Path(result["odds_decoded_payload_path"]).read_text())
        assert retained["d"]["oddsdata"]["back"]["E-3-2-0-0-0"]["odds"]["9999"] == [1.7, 2.3]
    asyncio.run(scenario())


def test_published_request_key_cannot_redirect_or_escape_proxy_path(tmp_path, payload):
    async def scenario():
        shell = copy.deepcopy(SHELL)
        shell["d"]["eventData"]["xhashf"] = "%2e%2e%2fsecret"
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, shell=shell, calls=calls)) as client:
            with pytest.raises(ValueError, match="prematch request key"):
                await collector.fetch_detailed_odds(client, new_run(tmp_path), EVENT_ID)
        assert len(calls) == 2
    asyncio.run(scenario())


def test_history_denial_retains_obtained_prices_and_stops_next_event(tmp_path, payload):
    async def scenario():
        calls = []
        base = proxy_transport(payload)
        async def handler(request):
            calls.append(str(request.url))
            if "/proxy/match-event-history/" in request.url.path:
                return httpx.Response(429, content=b"Rate limited")
            return await base.handle_async_request(request)
        run = new_run(tmp_path, with_history=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.refresh_csv(client, run, source_csv(tmp_path, extra=True))
        assert len(calls) == 4
        assert manifest["status"] == "FAILED"
        assert manifest["stop_reason"] == "access_denied"
        records = json.loads(run.json_path.read_text())
        assert records[0]["odds1_sts_open"] == 1.55
        assert records[0]["price_history"]["165"]["status"] == "failed"
        assert records[0]["price_history"]["165"]["capture"]["http_status"] == 429
        assert records[1]["odds_status"] == "not_requested_access_denied"
    asyncio.run(scenario())


def test_malformed_history_retains_capture_and_failure_instead_of_empty_success(tmp_path, payload):
    async def scenario():
        bad_history = copy.deepcopy(HISTORY)
        back = bad_history["d"]["history"]["back"]
        back["unexpected-outcome"] = back.pop(next(iter(back)))
        run = new_run(tmp_path, with_history=True)
        async with httpx.AsyncClient(transport=proxy_transport(payload, history=bad_history)) as client:
            result = await collector.fetch_detailed_odds(client, run, EVENT_ID)
        entry = result["price_history"]["165"]
        assert entry["status"] == "failed"
        assert "outcomes" in entry["error"]
        assert Path(entry["history_response_path"]).read_bytes() == encrypted(bad_history).encode()
        assert run.failures[0]["stage"] == "price_history"
    asyncio.run(scenario())


def archive_html(*, malformed_row=False, pages=1):
    props = json.loads((FIXTURES / "current_archive_props.json").read_text())
    initial = props["initial"]
    row = initial["rows"][0]
    original_id = row["encodeEventId"]
    row["encodeEventId"] = EVENT_ID
    row["url"] = row["url"].replace(original_id, EVENT_ID)
    initial["rows"] = [row] + ([{"encodeEventId": "bad"}] if malformed_row else [])
    initial["seasonTabs"] = initial["seasonTabs"][:1]
    initial["pagination"]["pageCount"] = pages
    initial["pagination"]["activePage"] = 1
    flight = "9d:" + json.dumps(["$", "$Larchive", None, props]) + "\n"
    midpoint = len(flight) // 2
    return "<html>" + "".join(
        "<script>self.__next_f.push(" + json.dumps([1, chunk]) + ")</script>"
        for chunk in (flight[:midpoint], flight[midpoint:])
    ) + "</html>"


def test_current_archive_ssr_rows_reach_current_shell_transport_with_failed_row_receipt(tmp_path, payload):
    async def scenario():
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        base = proxy_transport(payload)
        async def handler(request):
            if str(request.url) == archive_url:
                return httpx.Response(200, text=archive_html(malformed_row=True))
            return await base.handle_async_request(request)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, archive_url)
        assert manifest["status"] == "PARTIAL"
        assert manifest["available_rows"] == 1
        assert manifest["failed_archive_rows"] == 1
        failure = next(item for item in manifest["failures"] if item["stage"] == "archive_row")
        assert failure["source_row"] == {"encodeEventId": "bad"}
        assert Path(failure["archive_capture"]["body_path"]).read_bytes() == archive_html(malformed_row=True).encode()
        row = json.loads(run.json_path.read_text())[0]
        assert row["event_id"] == EVENT_ID
        assert row["home_team"] == "Sentinels"
        assert row["odds1_sts_open"] == 1.55
        assert "source_archive_row" in row
    asyncio.run(scenario())


def test_bounded_archive_stops_without_requesting_other_pages_or_seasons(tmp_path, payload):
    async def scenario():
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        base = proxy_transport(payload)
        calls = []
        async def handler(request):
            calls.append(str(request.url))
            if str(request.url) == archive_url:
                return httpx.Response(200, text=archive_html(pages=2))
            return await base.handle_async_request(request)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, archive_url, max_matches=1)
        assert manifest["status"] == "COMPLETED"
        assert len(calls) == 4
        assert manifest["rows"] == 1
    asyncio.run(scenario())


def test_provider_soft_404_is_explicit_failure_with_retained_html(tmp_path):
    async def scenario():
        run = new_run(tmp_path, mode="archive")
        body = b"<html><h1>404 Offside - page not found</h1></html>"
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))) as client:
            manifest = await collector.scrape(client, run, "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/")
        assert manifest["status"] == "FAILED"
        assert manifest["rows"] == 0
        capture = json.loads((run.raw_dir / "captures.jsonl").read_text().splitlines()[0])
        assert Path(capture["body_path"]).read_bytes() == body
        assert manifest["failures"]
    asyncio.run(scenario())


def test_with_markets_retains_navigation_without_duplicate_moneyline_request(tmp_path, payload):
    async def scenario():
        calls = []
        async with httpx.AsyncClient(transport=proxy_transport(payload, calls=calls)) as client:
            result = await collector.fetch_detailed_odds(client, new_run(tmp_path, with_markets=True), EVENT_ID)
        assert len(calls) == 4
        assert result["market_navigation"]["status"] == "available"
        entry = result["additional_markets"]["3-2"]
        assert entry["status"] == "retained_default_moneyline"
        assert entry["odds_decoded_payload_path"] == result["odds_decoded_payload_path"]
        assert json.loads(Path(result["market_navigation"]["navigation_decoded_payload_path"]).read_text())["d"]["nav"] == {"3": {"2": ["165"]}}
    asyncio.run(scenario())


def test_other_advertised_markets_are_raw_only_and_bound_to_exact_request(tmp_path, payload):
    async def scenario():
        other = copy.deepcopy(payload)
        other["d"]["bt"] = 10
        other["d"]["opaqueUnsupportedMarket"] = {"history": ["unaltered"]}
        base = proxy_transport(payload)
        calls = []
        async def handler(request):
            calls.append(str(request.url))
            if request.url.path == f"/proxy/ajax-event-nav/{EVENT_ID}/0/":
                return httpx.Response(200, text=encrypted({"s": 1, "d": {"nav": {"3": {"2": ["165"]}, "10": {"2": ["165"]}}}}))
            if request.url.path == f"/proxy/match-event/1-36-{EVENT_ID}-10-2-yjd10.dat":
                return httpx.Response(200, text=encrypted(other))
            return await base.handle_async_request(request)
        run = new_run(tmp_path, with_markets=True)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await collector.fetch_detailed_odds(client, run, EVENT_ID)
        assert len(calls) == 5
        entry = result["additional_markets"]["10-2"]
        assert entry["status"] == "retained"
        assert entry["interpretation"] == "raw_advertised_market_not_normalized"
        assert json.loads(Path(entry["market_decoded_payload_path"]).read_text()) == other
        assert result["odds1_sts_open"] == 1.55
    asyncio.run(scenario())


@pytest.mark.parametrize("wrong_page_count", [False, True])
def test_native_archive_pagination_reuses_ssr_and_checks_advertised_page_identity(tmp_path, payload, wrong_page_count):
    async def scenario():
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        html = archive_html(pages=2)
        initial = collector.parse_current_archive(html, archive_url)["source_initial"]
        page2 = {"s": 1, "d": {
            "total": 2, "onePage": 1, "page": 2, "rows": initial["rows"],
            "pagination": {**initial["pagination"], "activePage": 2, "pageCount": 3 if wrong_page_count else 2},
        }}
        expected_page2 = collector.BASE_URL + "/proxy" + httpx.URL(initial["pagination"]["url"]).path.rstrip("/") + "/page/2/"
        base = proxy_transport(payload)
        calls = []
        async def handler(request):
            calls.append(str(request.url))
            if str(request.url) == archive_url:
                return httpx.Response(200, text=html)
            if str(request.url) == expected_page2:
                return httpx.Response(200, text=encrypted(page2))
            return await base.handle_async_request(request)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, archive_url)
        assert calls.count(archive_url) == 1
        assert calls.count(expected_page2) == 1
        assert len(calls) == 5
        assert manifest["rows"] == 1
        assert manifest["status"] == ("PARTIAL" if wrong_page_count else "COMPLETED")
        if wrong_page_count:
            assert "page count changed" in manifest["failures"][0]["error"]
    asyncio.run(scenario())


def test_catalogue_reconstruction_records_published_slug_and_validates_target_ssr(tmp_path, payload):
    async def scenario():
        href = "/pl/esports//league-of-legends-lcs/results/"
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        base = proxy_transport(payload)
        calls = []
        async def handler(request):
            calls.append(str(request.url))
            if str(request.url) == "https://www.oddsportal.com/pl/results/esports/":
                return httpx.Response(200, text=f'<html><a href="{href}">LCS</a></html>')
            if str(request.url) == archive_url:
                return httpx.Response(200, text=archive_html())
            return await base.handle_async_request(request)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, max_matches=1)
        assert manifest["status"] == "COMPLETED"
        assert len(calls) == 5
        assert not any("/esports//" in url for url in calls)
        receipt = manifest["discovery"][0]
        assert receipt["advertised_href"] == href
        assert receipt["archive_url"] == archive_url
        assert receipt["locator_semantics"] == "published_catalogue_slug_under_observed_canonical_game_root"
    asyncio.run(scenario())


@pytest.mark.parametrize("year", [2015, 2016])
def test_advertised_old_empty_shell_is_named_source_absence_not_fake_dataset(tmp_path, year):
    async def scenario():
        props = json.loads((FIXTURES / f"current_archive_empty_{year}.json").read_text())
        initial = props["initial"]
        active_url = initial["activeSeason"]["url"]
        initial["seasonTabs"] = [season for season in initial["seasonTabs"] if season["href"] == active_url]
        flight = "9d:" + json.dumps(["$", "$Larchive", None, props]) + "\n"
        html = "<html><script>self.__next_f.push(" + json.dumps([1, flight]) + ")</script></html>"
        calls = []
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200, text=html)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, collector.BASE_URL + active_url)
        assert len(calls) == 1
        assert manifest["status"] == "SOURCE_ABSENT"
        assert manifest["rows"] == 0
        assert manifest["available_rows"] == 0
        assert manifest["failures"] == []
        absence = manifest["source_absences"][0]
        assert absence["reason"] == "archive_season_no_source_rows"
        assert absence["observed_rows"] == []
        assert absence["observed_pagination"] == []
        assert absence["capture"]["http_status"] == 200
        assert json.loads(run.json_path.read_text()) == []
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["refresh", "archive"])
def test_current_metadata_does_not_export_contradictory_legacy_identity_aliases(tmp_path, payload, mode):
    async def scenario():
        run = new_run(tmp_path, mode=mode)
        base = proxy_transport(payload)
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        async def handler(request):
            if str(request.url) == archive_url:
                return httpx.Response(200, text=archive_html())
            return await base.handle_async_request(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            if mode == "archive":
                await collector.scrape(client, run, archive_url, max_matches=1)
            else:
                source = tmp_path / "legacy.csv"
                source.write_text(
                    "url,home_team,away_team,name,date-start-base,home-winner,away-winner,encodeEventId\n"
                    + f"https://www.oddsportal.com/event-{EVENT_ID}/,Old home,Old away,Old home - Old away,1,win,lost,{EVENT_ID}\n" * 2
                )
                await collector.refresh_csv(client, run, source)
        records = json.loads(run.json_path.read_text())
        for row in records:
            assert (row["home_team"], row["away_team"]) == ("Sentinels", "Cloud9")
            assert row["winner"] == "away"
            assert row["date"] == datetime.fromtimestamp(SHELL["d"]["eventBody"]["startDate"], tz=collector.timezone.utc).isoformat()
            assert not {"name", "date-start-base", "home-winner", "away-winner", "encodeEventId"} & row.keys()
        with run.csv_path.open() as stream:
            exported = list(csv.DictReader(stream))
        assert not {"name", "date-start-base", "home-winner", "away-winner", "encodeEventId"} & exported[0].keys()
    asyncio.run(scenario())


def test_unavailable_archive_prices_do_not_restore_superseded_identity_fields(tmp_path, payload):
    async def scenario():
        payload["d"]["encodeventId"] = "Other123"
        archive_url = "https://www.oddsportal.com/pl/esports/league-of-legends/league-of-legends-lcs/results/"
        base = proxy_transport(payload)
        async def handler(request):
            if str(request.url) == archive_url:
                return httpx.Response(200, text=archive_html())
            return await base.handle_async_request(request)
        run = new_run(tmp_path, mode="archive")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            manifest = await collector.scrape(client, run, archive_url, max_matches=1)
        row = json.loads(run.json_path.read_text())[0]
        assert manifest["status"] == "FAILED"
        assert row["odds_status"] == "failed"
        assert row["event_id"] == EVENT_ID
        assert not {"name", "date-start-base", "home-winner", "away-winner", "encodeEventId"} & row.keys()
        assert row["source_archive_row"]["encodeEventId"] == EVENT_ID
    asyncio.run(scenario())
