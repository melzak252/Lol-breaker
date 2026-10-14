"""Observable behavior of the shared Liquipedia request budget and cache."""

import gzip
import io
import json
import multiprocessing
import os
from datetime import UTC, datetime
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from typing import Any
from unittest.mock import Mock
import urllib.error

import pytest

from betting_app.services.liquipedia_service import LiquipediaClient
from betting_app.services.liquipedia_bracket_service import LiquipediaBracketService
from betting_app.services import liquipedia_transport
from betting_app.services.liquipedia_transport import LiquipediaRequestError


class Response:
    def __init__(self, data: Any = None, *, body: bytes | None = None, headers: dict[str, str] | None = None):
        self.data = body if body is not None else json.dumps(data).encode()
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.data

    def info(self):
        return self.headers


class Clock:
    def __init__(self, now: float):
        self.now = now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def set_clock(monkeypatch, now: float = 100.0) -> Clock:
    clock = Clock(now)
    monkeypatch.setattr(liquipedia_transport, "time", clock)
    return clock


def _query_from_process(api_url: str, cache_dir: str, start, ready, results) -> None:
    os.environ["LIQUIPEDIA_CACHE_DIR"] = cache_dir
    ready.put(True)
    start.wait(10)
    client = LiquipediaClient(api_url=api_url)
    try:
        results.put(("ok", client._query({"action": "parse", "page": "T1", "prop": "text", "format": "json"})))
    except Exception as error:  # pragma: no cover - reported to the parent process
        results.put(("error", type(error).__name__, str(error)))


def test_repeated_roster_request_across_clients_uses_shared_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"parse": {"text": {"*": "same roster"}}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    params = {"action": "parse", "page": "T1", "prop": "text", "format": "json"}
    first = LiquipediaClient()._query(params)
    second = LiquipediaClient()._query(params)
    assert first == second == {"parse": {"text": {"*": "same roster"}}}
    assert network.call_count == 1


def test_roster_and_bracket_clients_share_parse_request_spacing(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"parse": {"text": {"*": "roster"}}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    LiquipediaClient()._query({"action": "parse", "page": "T1", "prop": "text", "format": "json"})
    data, error = LiquipediaBracketService(cache_dir=tmp_path / "brackets").fetch_liquipedia_page("LCK/Playoffs")
    assert network.call_count == 1
    assert data is None and error


def test_different_queries_observe_two_second_host_spacing(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(side_effect=lambda *_a, **_kw: Response({"query": {}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    liquipedia_transport.query(url, {"action": "opensearch", "search": "one"}, "research-agent")

    clock.now = 101.5
    with pytest.raises(LiquipediaRequestError) as deferred:
        liquipedia_transport.query(url, {"action": "opensearch", "search": "two"}, "research-agent")
    assert deferred.value.kind == "deferred"
    assert deferred.value.retry_at == 102.0
    assert network.call_count == 1

    clock.now = deferred.value.retry_at
    liquipedia_transport.query(url, {"action": "opensearch", "search": "two"}, "research-agent")
    assert network.call_count == 2


def test_explicit_default_port_uses_same_hostname_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    liquipedia_transport.query(
        "https://liquipedia.example/api.php", {"action": "opensearch", "search": "one"}, "research-agent"
    )
    clock.now = 101

    with pytest.raises(LiquipediaRequestError) as deferred:
        liquipedia_transport.query(
            "https://liquipedia.example:443/api.php",
            {"action": "opensearch", "search": "two"},
            "research-agent",
        )
    assert deferred.value.kind == "deferred"
    assert deferred.value.retry_at == 102
    assert network.call_count == 1


def test_parse_and_expandtemplates_share_thirty_second_spacing(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    liquipedia_transport.query(url, {"action": "parse", "page": "one"}, "research-agent")

    clock.now = 129.5
    with pytest.raises(LiquipediaRequestError) as deferred:
        liquipedia_transport.query(url, {"action": "expandtemplates", "text": "two"}, "research-agent")
    assert deferred.value.kind == "deferred"
    assert deferred.value.retry_at == 130.0
    assert network.call_count == 1

    clock.now = deferred.value.retry_at
    liquipedia_transport.query(url, {"action": "expandtemplates", "text": "two"}, "research-agent")
    assert network.call_count == 2


def test_explicit_collector_waits_for_spacing_but_default_clients_fail_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    liquipedia_transport.query(url, {"action": "parse", "page": "one"}, "research-agent")
    clock.now = 100.5

    with pytest.raises(LiquipediaRequestError) as deferred:
        liquipedia_transport.query(url, {"action": "parse", "page": "two"}, "research-agent")
    assert deferred.value.kind == "deferred"
    data = liquipedia_transport.query(
        url, {"action": "parse", "page": "two"}, "research-agent", wait_for_spacing=True
    )
    assert data == {"ok": True}
    assert clock.now == 130.0
    assert network.call_count == 2


def test_cache_expires_at_configured_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(side_effect=lambda *_a, **_kw: Response({"result": network.call_count}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    params = {"action": "opensearch", "search": "cache-expiry"}

    assert liquipedia_transport.query(url, params, "research-agent", cache_ttl_seconds=5) == {"result": 1}
    clock.now = 104.999
    assert liquipedia_transport.query(url, params, "research-agent", cache_ttl_seconds=5) == {"result": 1}
    clock.now = 105.0
    assert liquipedia_transport.query(url, params, "research-agent", cache_ttl_seconds=5) == {"result": 2}
    assert network.call_count == 2


def test_corrupt_host_state_fails_closed_without_network_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    liquipedia_transport.query(url, {"action": "opensearch", "search": "prime"}, "research-agent")
    next(tmp_path.glob("host-*.json")).write_text("{corrupt", encoding="utf-8")

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(url, {"action": "opensearch", "search": "different"}, "research-agent")
    assert error.value.kind == "state"
    assert network.call_count == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("last_request_at", float("nan")),
        ("last_parse_at", float("inf")),
        ("cooldown_until", True),
        ("last_request_at", -1),
        ("missing_required_key", None),
    ],
)
def test_invalid_host_budget_state_values_fail_closed(tmp_path, monkeypatch, field, value):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    liquipedia_transport.query(url, {"action": "opensearch", "search": "prime"}, "research-agent")
    state_path = next(tmp_path.glob("host-*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if field == "missing_required_key":
        state.pop("last_parse_at")
    else:
        state[field] = value
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(url, {"action": "opensearch", "search": "different"}, "research-agent")
    assert error.value.kind == "state"
    assert network.call_count == 1


def test_corrupt_query_cache_fails_closed_without_network_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    params = {"action": "opensearch", "search": "corrupt-cache"}
    liquipedia_transport.query(url, params, "research-agent")
    next(tmp_path.glob("query-*.json")).write_text("not-json", encoding="utf-8")

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(url, params, "research-agent")
    assert error.value.kind == "state"
    assert network.call_count == 1


@pytest.mark.parametrize(
    "corruption",
    ["nan_expiry", "bool_expiry", "bool_retry_at", "infinite_retry_at", "expired_missing_data"],
)
def test_corrupt_cache_timestamps_fail_closed_even_after_expiry(tmp_path, monkeypatch, corruption):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=lambda *_a, **_kw: Response({"ok": True}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    params = {"action": "opensearch", "search": "corrupt-timestamp"}
    liquipedia_transport.query(url, params, "research-agent")
    cache_path = next(tmp_path.glob("query-*.json"))
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    if corruption == "nan_expiry":
        entry["expires_at"] = float("nan")
    elif corruption == "bool_expiry":
        entry["expires_at"] = True
    elif corruption == "expired_missing_data":
        entry["expires_at"] = 0
        entry.pop("data")
    else:
        entry["success"] = False
        entry.pop("data")
        entry["expires_at"] = time.time() + 30
        entry["error"] = {
            "message": "cached error",
            "kind": "transport",
            "retry_at": True if corruption == "bool_retry_at" else float("inf"),
            "status_code": None,
        }
    cache_path.write_text(json.dumps(entry), encoding="utf-8")

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(url, params, "research-agent")
    assert error.value.kind == "state"
    assert network.call_count == 1


@pytest.mark.parametrize(("retry_after", "expected_delay"), [(None, 3600), ("120", 3600), ("5400", 5400), ("inf", 3600)])
def test_http_429_uses_at_least_one_hour_and_honors_seconds_header(
    tmp_path, monkeypatch, retry_after, expected_delay
):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch, 1000.0)
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    failure = urllib.error.HTTPError("https://liquipedia.example/api.php", 429, "Too Many Requests", headers, io.BytesIO())
    network = Mock(side_effect=failure)
    monkeypatch.setattr("urllib.request.urlopen", network)

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(
            "https://liquipedia.example/api.php", {"action": "opensearch", "search": "rate-limit"}, "research-agent"
        )
    assert error.value.status_code == 429
    assert error.value.retry_at == clock.now + expected_delay
    assert error.value.kind == "throttled"
    assert network.call_count == 1


def test_http_429_parses_retry_after_http_date(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch, 1000.0)
    retry_at = clock.now + 7200
    retry_header = format_datetime(datetime.fromtimestamp(retry_at, tz=UTC), usegmt=True)
    failure = urllib.error.HTTPError(
        "https://liquipedia.example/api.php", 429, "Too Many Requests", {"Retry-After": retry_header}, io.BytesIO()
    )
    network = Mock(side_effect=failure)
    monkeypatch.setattr("urllib.request.urlopen", network)

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(
            "https://liquipedia.example/api.php", {"action": "parse", "page": "date-limit"}, "research-agent"
        )
    assert error.value.retry_at == retry_at
    assert network.call_count == 1


def test_http_403_sets_twenty_four_hour_host_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch, 2000.0)
    failure = urllib.error.HTTPError(
        "https://liquipedia.example/api.php", 403, "Forbidden", {}, io.BytesIO()
    )
    network = Mock(side_effect=failure)
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"

    with pytest.raises(LiquipediaRequestError) as error:
        liquipedia_transport.query(url, {"action": "parse", "page": "blocked"}, "research-agent")
    assert error.value.retry_at == clock.now + 86400
    assert error.value.status_code == 403
    with pytest.raises(LiquipediaRequestError) as cooldown:
        liquipedia_transport.query(
            url, {"action": "opensearch", "search": "other"}, "research-agent", wait_for_spacing=True
        )
    assert cooldown.value.kind == "throttled"
    assert cooldown.value.retry_at == error.value.retry_at
    assert clock.now == 2000.0
    assert network.call_count == 1


def test_api_missingtitle_error_is_typed_and_negative_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    network = Mock(return_value=Response({"error": {"code": "missingtitle", "info": "No such page"}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    url = "https://liquipedia.example/api.php"
    params = {"action": "parse", "page": "missing"}

    with pytest.raises(LiquipediaRequestError) as first:
        liquipedia_transport.query(url, params, "research-agent")
    with pytest.raises(LiquipediaRequestError) as second:
        liquipedia_transport.query(url, params, "research-agent")
    assert first.value.kind == second.value.kind == "missing_page"
    assert "missingtitle" in str(second.value)
    assert network.call_count == 1

    clock.now += liquipedia_transport.NEGATIVE_CACHE_TTL_SECONDS
    with pytest.raises(LiquipediaRequestError):
        liquipedia_transport.query(url, params, "research-agent")
    assert network.call_count == 2


def test_gzip_opensearch_list_payload_and_identifying_headers_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    payload = ["T1", ["T1/Results", "T1"], ["https://liquipedia.net/leagueoflegends/T1"], []]
    network = Mock(
        return_value=Response(
            body=gzip.compress(json.dumps(payload).encode()), headers={"Content-Encoding": "gzip"}
        )
    )
    monkeypatch.setattr("urllib.request.urlopen", network)

    client = LiquipediaClient(api_url="https://liquipedia.example/api.php")
    assert client.search_team_page("T1") == "T1"
    request = network.call_args.args[0]
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["user-agent"] == client.user_agent
    assert headers["api-user-agent"] == client.user_agent
    assert headers["accept-encoding"] == "gzip"


def test_transport_failure_does_not_trigger_roster_title_search_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(side_effect=urllib.error.URLError("offline"))
    monkeypatch.setattr("urllib.request.urlopen", network)
    client = LiquipediaClient(api_url="https://liquipedia.example/api.php")

    assert client.fetch_active_roster("Missing Team") == []
    assert client.last_error is not None
    assert client.last_error.kind == "transport"
    assert network.call_count == 1


def test_missingtitle_api_error_allows_only_explicit_roster_title_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch)
    roster_html = (
        "<table><tr><th>ID</th><th>Name</th><th>Position</th></tr>"
        "<tr><td>Top</td><td>Top Name</td><td>Top</td></tr>"
        "<tr><td>Jungle</td><td>Jungle Name</td><td>Jungle</td></tr>"
        "<tr><td>Mid</td><td>Mid Name</td><td>Mid</td></tr>"
        "<tr><td>ADC</td><td>ADC Name</td><td>ADC</td></tr>"
        "<tr><td>Support</td><td>Support Name</td><td>Support</td></tr></table>"
    )
    network = Mock(
        side_effect=[
            Response({"error": {"code": "missingtitle", "info": "No such page"}}),
            Response(["Original Team", ["Resolved Team"], [], []]),
            Response({"parse": {"text": {"*": roster_html}}}),
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", network)
    client = LiquipediaClient(api_url="https://liquipedia.example/api.php", wait_for_spacing=True)

    roster = client.fetch_active_roster("Original Team")
    assert [player.role for player in roster] == ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
    assert network.call_count == 3
    assert clock.now == 130.0


@pytest.mark.parametrize(("api_code", "error_kind"), [("badvalue", "api"), ("ratelimited", "throttled"), ("maxlag", "throttled")])
def test_non_missing_api_errors_do_not_trigger_title_search(tmp_path, monkeypatch, api_code, error_kind):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    network = Mock(return_value=Response({"error": {"code": api_code, "info": "Request rejected"}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    client = LiquipediaClient(api_url="https://liquipedia.example/api.php")

    assert client.fetch_active_roster("Unknown Team") == []
    assert client.last_error is not None
    assert client.last_error.kind == error_kind
    assert network.call_count == 1


@pytest.mark.parametrize("api_code", ["ratelimited", "maxlag"])
def test_api_throttle_error_json_sets_host_cooldown(tmp_path, monkeypatch, api_code):
    monkeypatch.setenv("LIQUIPEDIA_CACHE_DIR", str(tmp_path))
    clock = set_clock(monkeypatch, 500.0)
    network = Mock(return_value=Response({"error": {"code": api_code, "info": "Request deferred"}}))
    monkeypatch.setattr("urllib.request.urlopen", network)
    client = LiquipediaClient(api_url="https://liquipedia.example/api.php")

    assert client.fetch_active_roster("Unknown Team") == []
    assert client.last_error is not None and client.last_error.kind == "throttled"
    assert client.last_error.retry_at == 500.0 + liquipedia_transport.NEGATIVE_CACHE_TTL_SECONDS
    assert client._query({"action": "opensearch", "search": "different"}) is None
    assert client.last_error is not None and client.last_error.kind == "throttled"
    assert clock.now == 500.0
    assert network.call_count == 1


def test_concurrent_process_clients_single_flight_one_network_request(tmp_path):
    class CountingHandler(BaseHTTPRequestHandler):
        request_count = 0
        request_lock = threading.Lock()

        def do_GET(self):
            with self.request_lock:
                type(self).request_count += 1
            time.sleep(0.1)
            body = json.dumps({"parse": {"text": {"*": "one shared response"}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), CountingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    ready = context.Queue()
    results = context.Queue()
    api_url = f"http://127.0.0.1:{server.server_port}/api.php"
    processes = [
        context.Process(target=_query_from_process, args=(api_url, str(tmp_path), start, ready, results))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert [ready.get(timeout=10) for _ in processes] == [True, True]
        start.set()
        responses = [results.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        assert responses == [("ok", {"parse": {"text": {"*": "one shared response"}}})] * 2
        assert CountingHandler.request_count == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
