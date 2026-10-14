"""Manual OddsPortal acquisition with separate raw, derived and reporting artifacts.

Provider opening/last-change clocks are not certified market-close clocks. This
collector neither authenticates a source for G0 nor registers an application feed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urljoin, urlsplit

import httpx
import parsel

from src.utils.oddsportal_data import (
    decrypt_server_response, event_id_from_url, parse_archive_row, parse_detailed_odds,
)
from src.utils.oddsportal_current import _provider_path, parse_current_archive, parse_event_shell, parse_price_history

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.oddsportal.com"
TOURNAMENTS_URL = f"{BASE_URL}/pl/results/esports/"
GAME_ROOT = f"{BASE_URL}/pl/esports/league-of-legends/"
PASSWORD = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k"
SALT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"
BOOKIES = {"163": "eFortuna", "165": "STS", "502": "LV BET", "539": "Betclic", "572": "BETFAN", "591": "Superbet", "819": "Fuksiarz", "1193": "Rainbet"}
TIMING_SEMANTICS = "provider_open_and_last_not_certified_close"
BASE_COLS = ["date", "home_team", "away_team", "homeResult", "awayResult", "winner", "avg_odds_home", "avg_odds_away", "avg_open_home", "avg_open_away", "tournament"]
BOOKIE_COLS = [
    column
    for name in sorted(value.lower().replace(" ", "_") for value in BOOKIES.values())
    for column in (
        f"odds1_{name}_close", f"odds1_{name}_open", f"odds2_{name}_close", f"odds2_{name}_open",
        f"odds1_{name}_open_at", f"odds2_{name}_open_at", f"odds1_{name}_close_at", f"odds2_{name}_close_at",
    )
]
FINAL_CSV_COLS = BASE_COLS + BOOKIE_COLS + [
    "url", "event_id", "scraper_schema", "odds_status", "odds_error", "market_key",
    "quote_timing_semantics", "source_outcome_ids", "source_bookmaker_active",
    "source_open_change_at", "source_last_change_at", "odds_capture_started_at",
    "odds_capture_completed_at", "odds_response_sha256", "odds_response_path",
    "odds_http_status", "odds_decoded_payload_path", "odds_decoded_payload_sha256",
    "archive_response_sha256", "event_page_response_sha256", "event_metadata_status",
    "legacy_identity_certified",
]
DENIED_STATUSES = {401, 403, 429}
SUPERSEDED_IDENTITY_ALIASES = {"name", "date-start-base", "home-winner", "away-winner", "encodeEventId"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "www.oddsportal.com"
            or parsed.username is not None or parsed.password is not None or parsed.port is not None):
        raise ValueError("Expected an HTTPS www.oddsportal.com source URL without credentials or port")
    return url


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _atomic_text(path: Path, content: str) -> None:
    # Only paths inside a newly created, exclusively owned run are passed here.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _retain_bytes(path: Path, body: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != body:
            raise ValueError(f"Existing response artifact is corrupt: {path}")


class AcquisitionStopped(RuntimeError):
    """No further request may start after source access has been denied."""


class SourceProtocolChanged(ValueError):
    """The retained source is not the historical odds/archive protocol."""


@dataclass
class CollectionRun:
    run_id: str
    data_root: Path
    mode: str
    delay_seconds: float = 1.0
    concurrency: int = 1
    with_history: bool = False
    with_markets: bool = False
    raw_dir: Path = field(init=False)
    derived_dir: Path = field(init=False)
    reporting_dir: Path = field(init=False)
    started_at: str = field(default_factory=utc_now)
    failures: list[dict] = field(default_factory=list)
    captures: list[dict] = field(default_factory=list)
    discovery: list[dict] = field(default_factory=list)
    source_absences: list[dict] = field(default_factory=list)
    stop_reason: str | None = None
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _pace_lock: asyncio.Lock = field(init=False, repr=False)
    _last_request: float | None = field(default=None, init=False, repr=False)
    _bootstrap: dict | None = field(default=None, init=False, repr=False)
    _bootstrap_lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.run_id):
            raise ValueError("run-id must be a nonempty safe path component")
        if self.mode not in {"archive", "refresh"}:
            raise ValueError("Unsupported acquisition mode")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int) or self.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if isinstance(self.delay_seconds, bool) or not isinstance(self.delay_seconds, (float, int)) or not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("delay-seconds must be finite and nonnegative")
        self.data_root = Path(self.data_root).resolve()
        self.raw_dir = self.data_root / "01_raw/oddsportal" / self.run_id
        self.derived_dir = self.data_root / "02_intermediate/oddsportal" / self.run_id
        self.reporting_dir = self.data_root / "08_reporting/oddsportal" / self.run_id
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._pace_lock = asyncio.Lock()
        self._bootstrap_lock = asyncio.Lock()

    @classmethod
    def create(cls, run_id: str, data_root: Path = PROJECT_ROOT / "data", *, mode: str, delay_seconds: float = 1.0, concurrency: int = 1, with_history: bool = False, with_markets: bool = False) -> CollectionRun:
        run = cls(run_id, data_root, mode, delay_seconds=delay_seconds, concurrency=concurrency, with_history=with_history, with_markets=with_markets)
        directories = (run.raw_dir, run.derived_dir, run.reporting_dir)
        if any(path.exists() or path.is_symlink() for path in directories):
            raise ValueError("Use a new run-id/output root; existing acquisition artifacts are never overwritten")
        for path in directories:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.mkdir(exist_ok=False)
        _atomic_json(run.manifest_path, run._manifest_base("RUNNING"))
        return run

    @property
    def json_path(self) -> Path:
        return self.derived_dir / "matches.json"

    @property
    def csv_path(self) -> Path:
        return self.derived_dir / "matches.csv"

    @property
    def manifest_path(self) -> Path:
        return self.reporting_dir / "run.json"

    def _manifest_base(self, status: str) -> dict:
        return {
            "schema": "oddsportal_acquisition_v2", "run_id": self.run_id, "mode": self.mode,
            "status": status, "started_at": self.started_at,
            "raw_dir": str(self.raw_dir), "derived_dir": str(self.derived_dir),
            "reporting_dir": str(self.reporting_dir), "delay_seconds": self.delay_seconds,
            "concurrency": self.concurrency, "bookmakers": BOOKIES,
            "with_history": self.with_history, "bootstrap": self._bootstrap,
            "with_markets": self.with_markets, "discovery": self.discovery,
            "quote_timing_semantics": TIMING_SEMANTICS, "source_authenticated": False,
            "source_activation_changed": False,
        }

    def finish(self, records: list[dict], **metadata) -> dict:
        write_outputs(self, records)
        available = sum(row.get("odds_status") == "available" for row in records)
        failed = sum(row.get("odds_status") == "failed" for row in records)
        status = "FAILED" if self.stop_reason or (self.failures and not available) else "PARTIAL" if self.failures else "COMPLETED"
        if status == "COMPLETED" and self.source_absences:
            status = "SOURCE_ABSENT" if not records else "PARTIAL_SOURCE_COVERAGE"
        artifacts = []
        for path in sorted((*self.raw_dir.iterdir(), *self.derived_dir.iterdir())):
            if path.is_file():
                artifacts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
        manifest = {
            **self._manifest_base(status), **metadata, "completed_at": utc_now(), "rows": len(records),
            "available_rows": available, "failed_rows": failed,
            "failed_archive_rows": sum(item.get("stage") == "archive_row" for item in self.failures),
            "failure_count": len(self.failures),
            "legacy_not_refreshed_rows": sum(row.get("odds_status") == "legacy_not_refreshed" for row in records),
            "not_requested_rows": sum(str(row.get("odds_status", "")).startswith("not_requested") for row in records),
            "captures": len(self.captures), "stop_reason": self.stop_reason,
            "failures": self.failures, "artifacts": artifacts,
            "source_absences": self.source_absences,
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest

    async def pace(self) -> None:
        async with self._pace_lock:
            if self.stop_reason:
                raise AcquisitionStopped(self.stop_reason)
            if self._last_request is not None:
                remaining = self.delay_seconds - (time.monotonic() - self._last_request)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            if self.stop_reason:
                raise AcquisitionStopped(self.stop_reason)
            self._last_request = time.monotonic()


def capture_response(run: CollectionRun, response: httpx.Response, started_at: str, completed_at: str) -> dict:
    """Retain HTTP-decoded entity bytes before decrypting or parsing anything."""
    body = response.content
    digest = hashlib.sha256(body).hexdigest()
    raw_path = run.raw_dir / f"{digest}.body"
    _retain_bytes(raw_path, body)
    record = {
        "url": str(response.request.url), "http_status": response.status_code,
        "capture_started_at": started_at, "capture_completed_at": completed_at,
        "body_sha256": digest, "body_path": str(raw_path), "body_bytes": len(body),
        "body_semantics": "httpx_decoded_http_entity_before_oddsportal_decryption",
        "response_headers": {name: response.headers[name] for name in (
            "content-type", "content-encoding", "etag", "last-modified", "retry-after", "location"
        ) if name in response.headers},
    }
    with (run.raw_dir / "captures.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    run.captures.append(record)
    return record


async def safe_request(client: httpx.AsyncClient, run: CollectionRun, url: str, *, attempts: int = 3) -> httpx.Response:
    source_url(url)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError("attempts must be an integer between 1 and 3")
    for attempt in range(attempts):
        try:
            async with run._semaphore:
                await run.pace()
                started_at = utc_now()
                response = await client.get(url, timeout=30.0, follow_redirects=False)
                completed_at = utc_now()
                response.extensions["oddsportal_capture"] = capture_response(run, response, started_at, completed_at)
                if response.status_code in DENIED_STATUSES:
                    run.stop_reason = "access_denied"
                response.raise_for_status()
                if any(marker in response.content.lower() for marker in (
                    b"<title>just a moment", b"<title>access denied", b"<title>attention required!",
                )):
                    run.stop_reason = "access_denied"
                    raise ValueError("Source returned an access challenge; retained without bypass")
                return response
        except (httpx.RemoteProtocolError, httpx.TimeoutException, httpx.ConnectError):
            if attempt + 1 == attempts:
                raise
            # No status retries, especially not 429. Transport retries remain paced.
            await asyncio.sleep(run.delay_seconds * (attempt + 1))
    raise AssertionError("Unreachable request attempts")


async def _fetch_archive_response(client: httpx.AsyncClient, run: CollectionRun, url: str) -> httpx.Response:
    """Follow only published locale redirects for the same LoL archive identity."""
    original = _provider_path(url)
    seen = set()
    for _ in range(6):
        if url in seen:
            raise ValueError("Archive locale redirect loop")
        seen.add(url)
        try:
            return await safe_request(client, run, url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.response.headers.get("location")
            if not location:
                raise ValueError("Archive redirect has no published location") from exc
            target = source_url(urljoin(url, location))
            if (len(original) != 4 or original[:2] != ("esports", "league-of-legends")
                    or original[3] != "results" or _provider_path(target) != original):
                raise ValueError("Archive redirect changes its requested LoL identity") from exc
            url = target
    raise ValueError("Archive locale redirect limit exceeded")


def _decode_payload(run: CollectionRun, response: httpx.Response) -> tuple[dict, Path, str]:
    if response.headers.get("content-type", "").lower().startswith("text/html") or response.content.lstrip().startswith(b"<"):
        run.stop_reason = "provider_protocol_changed"
        raise SourceProtocolChanged("HTML is not an encrypted odds/archive payload; provider protocol has changed")
    payload = json.loads(decrypt_server_response(response.text, PASSWORD, SALT))
    if not isinstance(payload, dict):
        raise ValueError("Expected an object in decrypted source payload")
    body = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    path = run.derived_dir / f"payload-{digest}.json"
    _retain_bytes(path, body)
    return payload, path, digest


def _response_metadata(prefix: str, response: httpx.Response, decoded_path: Path, decoded_sha: str) -> dict:
    capture = response.extensions["oddsportal_capture"]
    return {
        f"{prefix}_capture_started_at": capture["capture_started_at"],
        f"{prefix}_capture_completed_at": capture["capture_completed_at"],
        f"{prefix}_response_sha256": capture["body_sha256"], f"{prefix}_response_path": capture["body_path"],
        f"{prefix}_http_status": capture["http_status"], f"{prefix}_decoded_payload_path": str(decoded_path),
        f"{prefix}_decoded_payload_sha256": decoded_sha,
    }


async def load_bootstrap(client: httpx.AsyncClient, run: CollectionRun) -> dict:
    async with run._bootstrap_lock:
        if run._bootstrap is None:
            try:
                response = await safe_request(client, run, f"{BASE_URL}/proxy/api/home-data")
                payload, path, digest = _decode_payload(run, response)
                geo = payload.get("geoIPcode")
                if not isinstance(geo, str) or not re.fullmatch(r"[A-Z]{2}", geo):
                    raise ValueError("Current bootstrap lacks a supported actual geoIPcode; no geography override")
                run._bootstrap = {"geoIPcode": geo, **_response_metadata("bootstrap", response, path, digest)}
            except ValueError:
                run.stop_reason = run.stop_reason or "source_integrity_failure"
                raise
        return run._bootstrap


async def fetch_detailed_odds(client: httpx.AsyncClient, run: CollectionRun, event_id: str) -> dict:
    if event_id_from_url(f"/event-{event_id}/") != event_id:
        raise ValueError("Invalid requested event identity")
    bootstrap = await load_bootstrap(client, run)
    shell_response = await safe_request(client, run, f"{BASE_URL}/proxy/ajax-event-data-shell/{event_id}/0/")
    shell_payload, shell_path, shell_sha = _decode_payload(run, shell_response)
    shell = parse_event_shell(shell_payload, event_id)
    if shell["request_bases"]["requestBasePreMatch"] != "/match-event/":
        raise SourceProtocolChanged("Unsupported published prematch request base")
    raw_key = shell["source_event_data"].get("xhashf")
    if not isinstance(raw_key, str):
        raise ValueError("Event shell lacks its published prematch request key")
    key = unquote(raw_key, errors="strict")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key):
        raise ValueError("Unsupported published prematch request key")
    query = urlencode({"geo": bootstrap["geoIPcode"], "lang": "pl"})
    url = f"{BASE_URL}/proxy/match-event/{shell['version_id']}-{shell['sport_id']}-{event_id}-3-2-{key}.dat?{query}"
    response = await safe_request(client, run, url)
    payload, decoded_path, decoded_sha = _decode_payload(run, response)
    result = {
        **shell, **parse_detailed_odds(payload, event_id, BOOKIES),
        **_response_metadata("shell", shell_response, shell_path, shell_sha),
        **_response_metadata("odds", response, decoded_path, decoded_sha),
        "tournament": shell["source_event_data"].get("tournamentName"),
        "event_metadata_status": "parsed_current_shell", "source_geo": bootstrap["geoIPcode"],
    }
    if run.with_history:
        market = payload["d"]["oddsdata"]["back"][result["market_key"]]
        providers = sorted(set(market.get("odds", {})) | set(market.get("openingOdd", {})))
        result["price_history"] = {}
        result["history_timing_semantics"] = "provider_returned_change_rows_not_certified_complete_or_terminal"
        for provider in providers:
            if run.stop_reason:
                result["price_history"][provider] = {"status": f"not_requested_{run.stop_reason}"}
                continue
            history_record = {}
            try:
                if shell["request_bases"]["requestBaseOddsHistory"] != "/match-event-history/" or not re.fullmatch(r"[0-9]+", provider):
                    raise ValueError("Unsupported published history request base or provider ID")
                history_url = f"{BASE_URL}/proxy/match-event-history/{shell['sport_id']}-{event_id}-3-2-0-{provider}/?{query}"
                history_response = await safe_request(client, run, history_url)
                history_payload, history_path, history_sha = _decode_payload(run, history_response)
                history_record.update(_response_metadata("history", history_response, history_path, history_sha))
                rows = parse_price_history(history_payload, result["source_outcome_ids"], provider)
                result["price_history"][provider] = {**history_record, "status": "available", "rows": rows}
            except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    history_record["capture"] = exc.response.extensions.get("oddsportal_capture")
                result["price_history"][provider] = {**history_record, "status": "failed", "error": str(exc)}
                _failure(run, "price_history", exc, event_id=event_id, bookmaker_id=provider)
    if run.with_markets and not run.stop_reason:
        await _collect_additional_markets(client, run, shell, key, query, result)
    elif run.with_markets:
        result["market_navigation"] = {"status": f"not_requested_{run.stop_reason}"}
    return result


async def _collect_additional_markets(client: httpx.AsyncClient, run: CollectionRun, shell: dict,
                                      key: str, query: str, result: dict) -> None:
    event_id = shell["event_id"]
    nav_record = {}
    try:
        response = await safe_request(client, run, f"{BASE_URL}/proxy/ajax-event-nav/{event_id}/0/")
        nav, nav_path, nav_sha = _decode_payload(run, response)
        nav_record.update(_response_metadata("navigation", response, nav_path, nav_sha))
        if type(nav.get("s")) is not int or nav["s"] != 1 or not isinstance(nav.get("d"), dict) or not isinstance(nav["d"].get("nav"), dict):
            raise ValueError("Malformed current event market navigation")
        advertised = nav["d"]["nav"]
        result["additional_markets"] = {}
        for betting_type, scopes in advertised.items():
            if not isinstance(betting_type, str) or not re.fullmatch(r"[1-9][0-9]*", betting_type) or not isinstance(scopes, dict):
                raise ValueError("Unsupported advertised market type/scopes")
            for scope, providers in scopes.items():
                if not isinstance(scope, str) or not re.fullmatch(r"[1-9][0-9]*", scope) or not isinstance(providers, list):
                    raise ValueError("Unsupported advertised market scope/providers")
                market_id = f"{betting_type}-{scope}"
                if market_id == "3-2":
                    result["additional_markets"][market_id] = {
                        "status": "retained_default_moneyline", "advertised_provider_ids": providers,
                        "odds_decoded_payload_path": result["odds_decoded_payload_path"],
                        "odds_decoded_payload_sha256": result["odds_decoded_payload_sha256"],
                    }
                    continue
                if run.stop_reason:
                    result["additional_markets"][market_id] = {"status": f"not_requested_{run.stop_reason}"}
                    continue
                market_record = {"advertised_provider_ids": providers, "interpretation": "raw_advertised_market_not_normalized"}
                try:
                    url = f"{BASE_URL}/proxy/match-event/{shell['version_id']}-{shell['sport_id']}-{event_id}-{betting_type}-{scope}-{key}.dat?{query}"
                    market_response = await safe_request(client, run, url)
                    payload, path, digest = _decode_payload(run, market_response)
                    market_record.update(_response_metadata("market", market_response, path, digest))
                    data = payload.get("d")
                    if (not isinstance(data, dict) or data.get("encodeventId") != event_id
                            or type(data.get("bt")) is not int or data["bt"] != int(betting_type)
                            or type(data.get("sc")) is not int or data["sc"] != int(scope)):
                        raise ValueError("Additional market response identity does not match advertised request")
                    market_record["status"] = "retained"
                except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
                    if isinstance(exc, httpx.HTTPStatusError):
                        market_record["capture"] = exc.response.extensions.get("oddsportal_capture")
                    market_record.update(status="failed", error=str(exc))
                    _failure(run, "additional_market", exc, event_id=event_id, market=market_id)
                result["additional_markets"][market_id] = market_record
        nav_record["status"] = "available"
    except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            nav_record["capture"] = exc.response.extensions.get("oddsportal_capture")
        nav_record.update(status="failed", error=str(exc))
        _failure(run, "market_navigation", exc, event_id=event_id)
    result["market_navigation"] = nav_record


def merge_matches(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = {}
    for row in (*existing, *incoming):
        event_id = row["event_id"]
        if event_id in merged and any(merged[event_id].get(key) != row.get(key) for key in ("home_team", "away_team")):
            raise ValueError(f"Conflicting participants for event {event_id}")
        merged[event_id] = row
    return list(merged.values())


def write_outputs(run: CollectionRun, records: list[dict]) -> None:
    # JSON retains full source rows, opaque payload links and legacy values. CSV
    # retains unknown input/derived columns as well, rather than silently dropping them.
    columns = list(dict.fromkeys([*FINAL_CSV_COLS, *(key for row in records for key in row if key not in {"legacy_row", "source_archive_row", "event_page_metadata"})]))
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=run.derived_dir, prefix=".matches.csv.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (list, dict)) else value for key, value in record.items() if key in columns})
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, run.csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(run.json_path, records)


def _apply_details(record: dict, details: dict) -> None:
    prior = record.get("legacy_row") or record.get("source_archive_row")
    if isinstance(prior, dict):
        previous = [record.get("home_team"), record.get("away_team")]
        current = [details.get("home_team"), details.get("away_team")]
        record["prior_participant_order"] = previous
        record["participant_order_comparison"] = "direct" if previous == current else "reversed" if previous == current[::-1] else "different_or_unknown"
        if previous != current:
            record["participant_discrepancy"] = {"prior": previous, "current_source": current, "prices_follow": "current_source_home_away"}
    record.update(details)
    # Original source aliases remain in the nested raw row, not beside fresh metadata.
    for key in SUPERSEDED_IDENTITY_ALIASES:
        record.pop(key, None)
    record["odds_status"] = "available" if details["openingsOdds"] or details["closingOdds"] else "not_available"
    for key, prefix in (("closingOdds", "avg_odds"), ("openingsOdds", "avg_open")):
        pairs = list(details[key].values())
        record[f"{prefix}_home"] = sum(pair[0] for pair in pairs) / len(pairs) if pairs else None
        record[f"{prefix}_away"] = sum(pair[1] for pair in pairs) / len(pairs) if pairs else None


def _failure(run: CollectionRun, stage: str, error: Exception, **metadata) -> None:
    run.failures.append({"stage": stage, "error": str(error), "error_type": type(error).__name__, **metadata})


async def _collect_details(client: httpx.AsyncClient, run: CollectionRun, records: list[dict]) -> None:
    # Bounded batches prevent a denial from leaving thousands of queued requests.
    for offset in range(0, len(records), run.concurrency):
        batch = records[offset:offset + run.concurrency]
        results = await asyncio.gather(*(fetch_detailed_odds(client, run, row["event_id"]) for row in batch), return_exceptions=True)
        for row, details in zip(batch, results):
            if isinstance(details, AcquisitionStopped):
                row.update(odds_status=f"not_requested_{run.stop_reason or 'access_denied'}", odds_error=str(details))
            elif isinstance(details, (ValueError, httpx.HTTPError)):
                if isinstance(details, SourceProtocolChanged):
                    run.stop_reason = run.stop_reason or "provider_protocol_changed"
                row.update(odds_status="failed", odds_error=str(details))
                _failure(run, "detailed_odds", details, event_id=row["event_id"])
            elif isinstance(details, BaseException):
                raise details
            else:
                _apply_details(row, details)
        if run.stop_reason:
            for row in records[offset + len(batch):]:
                row.update(odds_status=f"not_requested_{run.stop_reason}", odds_error=f"Acquisition stopped: {run.stop_reason}; no request attempted")
            break


def read_legacy_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ValueError("Missing or duplicate CSV column names; resolve source ambiguity first")
        if "url" not in headers:
            raise ValueError("Input CSV must retain OddsPortal event URLs")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("CSV row width does not match the declared header")
        return rows


def needs_refresh(row: dict) -> bool:
    # Legacy rows with clocks still lack current parser/raw-capture provenance.
    if row.get("scraper_schema") not in {"oddsportal_v2", "oddsportal_repair_v2"} or not row.get("odds_response_sha256"):
        return True
    has_price = False
    for column in BOOKIE_COLS:
        if column.endswith("_at") or row.get(column) in (None, ""):
            continue
        try:
            price = float(row[column])
        except (ValueError, TypeError):
            return True
        if not math.isfinite(price) or price <= 1 or not row.get(f"{column}_at"):
            return True
        has_price = True
    return not has_price


def _clear_legacy_odds(record: dict) -> None:
    for key in list(record):
        if key in SUPERSEDED_IDENTITY_ALIASES:
            record.pop(key)
            continue
        if (key.startswith(("odds", "avg_odds", "avg_open", "source_", "shell_", "history_", "event_page_"))
                or key in {"openingsOdds", "closingOdds", "market_key", "quote_timing_semantics", "price_history", "request_bases", "participant_discrepancy", "participant_order_comparison", "prior_participant_order", "additional_markets", "market_navigation"}):
            record.pop(key)
    record.update({key: None for key in BOOKIE_COLS})
    record.update({key: None for key in ("avg_odds_home", "avg_odds_away", "avg_open_home", "avg_open_away")})
    record["quote_timing_semantics"] = "unavailable_refresh"


async def refresh_csv(client: httpx.AsyncClient, run: CollectionRun, input_csv: Path, *, max_matches: int | None = None) -> dict:
    _validate_max_matches(max_matches)
    source_bytes = input_csv.read_bytes()
    records = []
    selected_indices = []
    grouped: dict[str, list[int]] = {}
    try:
        source_snapshot = run.raw_dir / f"input-{hashlib.sha256(source_bytes).hexdigest()}.csv"
        _retain_bytes(source_snapshot, source_bytes)
        source_rows = read_legacy_csv(source_snapshot)
        records = [{**row, "legacy_row": dict(row), "legacy_identity_certified": False,
                    "scraper_schema": "oddsportal_repair_v2", "odds_status": "legacy_not_refreshed",
                    "quote_timing_semantics": "unknown_legacy_export"} for row in source_rows]
        selected_indices = [index for index, row in enumerate(source_rows) if needs_refresh(row)]
        if max_matches is not None:
            selected_indices = selected_indices[:max_matches]
        for index in selected_indices:
            record = records[index]
            _clear_legacy_odds(record)
            try:
                # Permit only evidenced root-relative provider paths or the validated HTTPS host.
                url = source_url(urljoin(BASE_URL, record.get("url", "")))
                event_id = event_id_from_url(url)
                record["event_id"] = event_id
                record["url"] = url
                record["odds_status"] = "pending_refresh"
                grouped.setdefault(event_id, []).append(index)
            except ValueError as exc:
                record.update(odds_status="failed", odds_error=str(exc))
                _failure(run, "refresh_identity", exc, row=index)
        requests = []
        valid_groups = {}
        for event_id, indices in grouped.items():
            first = records[indices[0]]
            if any(any(records[index].get(key) != first.get(key) for key in ("home_team", "away_team")) for index in indices[1:]):
                error = ValueError(f"Conflicting participants for event {event_id}")
                for index in indices:
                    records[index].update(odds_status="failed", odds_error=str(error))
                    _failure(run, "refresh_identity", error, row=index, event_id=event_id)
                continue
            valid_groups[event_id] = indices
            requests.append(first)
        await _collect_details(client, run, requests)
        for event_id, indices in valid_groups.items():
            first = records[indices[0]]
            for index in indices[1:]:
                duplicate = records[index]
                duplicate.update({
                    key: value for key, value in first.items()
                    if key.startswith(("odds", "avg_odds", "avg_open", "source_", "shell_", "history_", "participant_", "prior_participant_"))
                    or key in {"openingsOdds", "closingOdds", "market_key", "quote_timing_semantics",
                               "home_team", "away_team", "date", "homeResult", "awayResult", "results",
                               "winner", "version_id", "sport_id", "tournament", "event_metadata_status",
                               "request_bases", "price_history", "additional_markets", "market_navigation", "isLive", "realLive", "isPostponed",
                               "isStarted", "isFinished", "isFinishedGracePeriod"}
                })
    except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
        _failure(run, "refresh", exc)
    return run.finish(records, source_path=str(input_csv.resolve()), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                      selected_for_refresh=len(selected_indices), distinct_selected_events=len(grouped),
                      refreshed_rows=sum(row.get("odds_status") in {"available", "not_available"} for row in records),
                      legacy_identity_certified=False)


async def get_tournament_links(client: httpx.AsyncClient, run: CollectionRun) -> list[str]:
    response = await safe_request(client, run, TOURNAMENTS_URL)
    links = parsel.Selector(text=response.text).xpath('//a[contains(@href, "league-of-legends")]/@href').getall()
    tournaments = {}
    for link in links:
        published = source_url(urljoin(BASE_URL, link))
        path = urlsplit(published).path
        match = re.fullmatch(r"/(?:[a-z]{2}/)?esports//(league-of-legends-[a-z0-9-]+)/results/", path)
        if match:
            url = GAME_ROOT + match.group(1) + "/results/"
            semantics = "published_catalogue_slug_under_observed_canonical_game_root"
        elif re.fullmatch(r"/(?:[a-z]{2}/)?esports/league-of-legends/league-of-legends-[a-z0-9-]+/results/", path):
            url = published
            semantics = "published_complete_archive_locator"
        else:
            continue
        if url not in tournaments:
            tournaments[url] = {
                "source_catalogue_url": TOURNAMENTS_URL, "advertised_href": link,
                "archive_url": url, "locator_semantics": semantics, "canonical_game_context": GAME_ROOT,
                "catalogue_capture": response.extensions["oddsportal_capture"],
                "requires_archive_source_identity_validation": True,
            }
    if not tournaments:
        raise ValueError("No exact LoL catalogue archives found; source HTML may have changed")
    run.discovery.extend(tournaments.values())
    return list(tournaments)


async def process_archive_page(client: httpx.AsyncClient, run: CollectionRun, archive_url: str,
                               existing: dict[str, dict], *, response: httpx.Response | None = None,
                               initial: dict | None = None, max_matches: int | None = None) -> list[dict]:
    url = source_url(archive_url)
    response = response if response is not None else await _fetch_archive_response(client, run, url)
    data = initial if initial is not None else parse_current_archive(response.text, url)["source_initial"]
    body = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    decoded_path = run.derived_dir / f"archive-{digest}.json"
    _retain_bytes(decoded_path, body)
    bases = []
    seen = dict(existing)
    for index, row in enumerate(data["rows"]):
        try:
            if not isinstance(row, dict) or row.get("sport-id") != 36:
                raise ValueError("Archive row must identify current esports sport-id=36")
            base = parse_archive_row(row)
            for key in SUPERSEDED_IDENTITY_ALIASES:
                base.pop(key, None)
            if base["event_id"] in seen:
                prior = seen[base["event_id"]]
                prior_identity = parse_archive_row(prior["source_archive_row"]) if "source_archive_row" in prior else prior
                merge_matches([prior_identity], [base])
                continue
            if max_matches is not None and len(bases) >= max_matches:
                break
            seen[base["event_id"]] = base
            base.update(scraper_schema="oddsportal_v2", source_archive_row=row,
                        archive_response_sha256=response.extensions["oddsportal_capture"]["body_sha256"],
                        archive_capture=response.extensions["oddsportal_capture"],
                        archive_decoded_payload_path=str(decoded_path), archive_decoded_payload_sha256=digest)
            bases.append(base)
        except ValueError as exc:
            _failure(run, "archive_row", exc, page_url=url, row_index=index, source_row=row,
                     archive_capture=response.extensions["oddsportal_capture"])
    await _collect_details(client, run, bases)
    return bases


async def _fetch_archive_page(client: httpx.AsyncClient, run: CollectionRun, initial: dict, page: int) -> tuple[httpx.Response, dict]:
    base = source_url(initial["pagination"]["url"])
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if parsed.query or parsed.fragment or not re.fullmatch(r"/(?:[a-z]{2}/)?ajax-sport-country-tournament-archive_/36/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/1/0", path):
        raise SourceProtocolChanged("Unsupported published current archive pagination locator")
    advertised_ajax = initial["activeSeason"].get("ajaxUrl")
    if not isinstance(advertised_ajax, str) or urlsplit(urljoin(BASE_URL, advertised_ajax)).path.rstrip("/") != path:
        raise ValueError("Archive pagination locator conflicts with advertised active season")
    response = await safe_request(client, run, f"{BASE_URL}/proxy{path}/page/{page}/")
    payload, _, _ = _decode_payload(run, response)
    data = payload.get("d")
    if type(payload.get("s")) is not int or payload["s"] != 1 or not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError("Current archive page response lacks successful source rows")
    pagination = data.get("pagination")
    if (type(data.get("page")) is not int or data["page"] != page or not isinstance(pagination, dict)
            or type(pagination.get("activePage")) is not int or pagination["activePage"] != page
            or type(pagination.get("pageCount")) is not int or pagination["pageCount"] != initial["pagination"]["pageCount"]
            or source_url(pagination.get("url", "")).rstrip("/") != base.rstrip("/")):
        raise ValueError("Current archive page identity/page count changed")
    total, size = data.get("total"), data.get("onePage")
    if (type(total) is not int or total < 0 or type(size) is not int or size <= 0
            or max(1, (total + size - 1) // size) != pagination["pageCount"]
            or len(data["rows"]) != min(size, max(0, total - (page - 1) * size))):
        raise ValueError("Current archive page total/size/row count is inconsistent")
    for row in data["rows"]:
        if not isinstance(row, dict):
            continue
        if row.get("tournament-id") is not None and row["tournament-id"] != initial["activeSeason"].get("id"):
            raise ValueError("Current archive page belongs to another tournament season")
        template = row.get("superTemplate")
        if isinstance(template, dict) and template.get("url") != "league-of-legends":
            raise ValueError("Current archive page belongs to another esport")
    return response, {**data, "activeSeason": initial["activeSeason"], "seasonTabs": initial["seasonTabs"], "sport": "esports"}


async def process_tournament(client: httpx.AsyncClient, run: CollectionRun, tournament_url: str,
                             existing: dict[str, dict], *, max_matches: int | None = None, checkpoint=None) -> list[dict]:
    response = await _fetch_archive_response(client, run, source_url(tournament_url))
    first_parsed = parse_current_archive(response.text, tournament_url)
    first = first_parsed["source_initial"]
    seasons = sorted(first["seasonTabs"], key=lambda season: 0 if isinstance(season, dict) and season.get("href") == first["activeSeason"]["url"] else 1)
    if not seasons:
        raise SourceProtocolChanged("Current archive contains no advertised season references")
    seen_seasons = set()
    matches = []
    for season in seasons:
        if not isinstance(season, dict) or not season.get("href"):
            raise ValueError("Invalid current tournament season metadata")
        season_url = source_url(urljoin(BASE_URL, season["href"]))
        if season_url in seen_seasons:
            continue
        seen_seasons.add(season_url)
        is_initial = season["href"] == first["activeSeason"]["url"]
        season_response = response if is_initial else await _fetch_archive_response(client, run, season_url)
        season_parsed = first_parsed if is_initial else parse_current_archive(season_response.text, season_url)
        season_data = season_parsed["source_initial"]
        remaining = None if max_matches is None else max_matches - len(matches)
        if remaining is not None and remaining <= 0:
            return matches
        if season_parsed["archive_status"] in {"no_source_rows", "empty_page"}:
            await process_archive_page(client, run, tournament_url if is_initial else season_url,
                                       existing, response=season_response, initial=season_data)
            run.source_absences.append({
                "reason": f"archive_season_{season_parsed['archive_status']}",
                "season": season, "url": season_url,
                "capture": season_response.extensions["oddsportal_capture"],
                "observed_rows": season_data["rows"], "observed_pagination": season_data["pagination"],
            })
            continue
        page_count = season_data["pagination"]["pageCount"]
        for page in range(1, page_count + 1):
            remaining = None if max_matches is None else max_matches - len(matches)
            if remaining is not None and remaining <= 0:
                return matches
            if page == 1:
                page_response, page_data = season_response, season_data
                page_url = tournament_url if is_initial else season_url
            else:
                page_response, page_data = await _fetch_archive_page(client, run, season_data, page)
                page_url = str(page_response.request.url)
            page_matches = await process_archive_page(client, run, page_url, existing,
                                                      response=page_response, initial=page_data, max_matches=remaining)
            for match in page_matches:
                match["source_season"] = season
                existing[match["event_id"]] = match
            matches = merge_matches(matches, page_matches)
            if checkpoint is not None:
                checkpoint(page_matches)
            if run.stop_reason or (max_matches is not None and len(matches) >= max_matches):
                return matches
    return matches


async def _single_event(client: httpx.AsyncClient, run: CollectionRun, url: str) -> dict:
    event_id = event_id_from_url(source_url(url))
    # Retain the public page independently; authoritative home/away order and
    # source start/results come from the strictly matched current event shell.
    record = {"url": url, "event_id": event_id, "scraper_schema": "oddsportal_v2", "date": None,
              "home_team": None, "away_team": None, "winner": None, "event_metadata_status": "retained_uninterpreted"}
    try:
        response = await safe_request(client, run, url)
        selector = parsel.Selector(text=response.text)
        record["event_page_metadata"] = {
            "title": selector.xpath("//title/text()").get(),
            "components": [{"tag": node.root.tag, "attributes": dict(node.root.attrib)}
                           for node in selector.xpath("//*[contains(name(), '-component')]")],
        }
        record["event_page_capture"] = response.extensions["oddsportal_capture"]
        record["event_page_response_sha256"] = record["event_page_capture"]["body_sha256"]
        await _collect_details(client, run, [record])
    except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
        record.update(odds_status="failed", odds_error=str(exc))
        _failure(run, "event_page", exc, event_id=event_id)
    return record


async def scrape(client: httpx.AsyncClient, run: CollectionRun, archive_url: str | None = None, *,
                 event_url: str | None = None, max_matches: int | None = None) -> dict:
    _validate_max_matches(max_matches)
    if event_url and archive_url:
        raise ValueError("Specify either an archive URL or --event-url, not both")
    records = []
    existing = {}
    def checkpoint(incoming):
        nonlocal records
        records = merge_matches(records, incoming)
        write_outputs(run, records)
    try:
        if event_url:
            records = [await _single_event(client, run, event_url)]
        else:
            tournaments = [source_url(archive_url)] if archive_url else await get_tournament_links(client, run)
            for url in tournaments:
                remaining = None if max_matches is None else max_matches - len(records)
                if remaining is not None and remaining <= 0:
                    break
                await process_tournament(client, run, url, existing, max_matches=remaining, checkpoint=checkpoint)
                if run.stop_reason:
                    break
    except (ValueError, httpx.HTTPError, AcquisitionStopped) as exc:
        _failure(run, "archive", exc, url=archive_url or event_url or TOURNAMENTS_URL)
    return run.finish(records, source_url=archive_url or event_url or TOURNAMENTS_URL, max_matches=max_matches)


def _validate_max_matches(max_matches: int | None) -> None:
    if max_matches is not None and (isinstance(max_matches, bool) or not isinstance(max_matches, int) or max_matches <= 0):
        raise ValueError("max-matches must be a positive integer")


def _parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="New safe run identity; existing run directories are refused")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data", help="Data root containing stage01/stage02/stage08; default: project data/")
    parser.add_argument("--max-matches", type=int, help="Bound selected CSV rows/archive events; other legacy rows remain explicitly uncertified")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Minimum interval between HTTP request starts (default: 1 second)")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum simultaneous requests (default: 1)")
    parser.add_argument("--with-history", action="store_true", help="Retain provider-returned per-book changes; never certify a complete timeline")
    parser.add_argument("--with-markets", action="store_true", help="Retain exact advertised additional market payloads without unsupported interpretation")
    if mode == "refresh":
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--input-csv", type=Path, help="Older export; never overwritten")
        source.add_argument("--inventory-json", type=Path, help="Download every exact event in a complete inventory")
        parser.add_argument("--checkpoint-every", type=int, default=100, help="Fsync settled events and progress after this many events")
    else:
        parser.add_argument("archive_url", nargs="?", help="One explicit OddsPortal results URL; omitted: discover LoL archives")
        parser.add_argument("--event-url", help="Retain one event page and full-event details without archive discovery")
        parser.add_argument("--inventory-only", action="store_true", help="Exhaust advertised leagues/seasons/pages without detailed odds requests")
        parser.add_argument("--known-events-csv", type=Path, help="Union known historical event URLs with the full archive inventory")
    return parser


async def _main(mode: str, argv: list[str] | None = None) -> int:
    parser = _parser(mode)
    args = parser.parse_args(argv)
    try:
        _validate_max_matches(args.max_matches)
        if mode == "refresh":
            if args.input_csv:
                read_legacy_csv(args.input_csv)
            else:
                from src.utils.oddsportal_bulk import _load_inventory
                _load_inventory(args.inventory_json.read_bytes())
                if args.max_matches is not None:
                    raise ValueError("--inventory-json downloads every event; --max-matches is not allowed")
            if args.checkpoint_every < 1:
                raise ValueError("--checkpoint-every must be positive")
        else:
            if args.archive_url and args.event_url:
                raise ValueError("Specify either an archive URL or --event-url, not both")
            if args.event_url and (args.inventory_only or args.known_events_csv):
                raise ValueError("Event-page acquisition cannot be combined with archive inventory")
            if args.known_events_csv:
                read_legacy_csv(args.known_events_csv)
                if args.max_matches is not None and not args.inventory_only:
                    raise ValueError("Known-event union requires exhaustive or --inventory-only acquisition")
            if args.archive_url:
                source_url(args.archive_url)
            if args.event_url:
                event_id_from_url(source_url(args.event_url))
        run = CollectionRun.create(args.run_id, args.output_dir, mode=mode, delay_seconds=args.delay_seconds, concurrency=args.concurrency, with_history=args.with_history, with_markets=args.with_markets)
        full_archive = mode == "archive" and not args.event_url and args.max_matches is None and not args.inventory_only
        inventory_run = CollectionRun.create(args.run_id + "_inventory", args.output_dir, mode="archive", delay_seconds=args.delay_seconds, concurrency=args.concurrency) if full_archive else None
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=30.0, headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}) as client:
        if mode == "refresh":
            if args.input_csv:
                manifest = await refresh_csv(client, run, args.input_csv, max_matches=args.max_matches)
            else:
                from src.utils.oddsportal_bulk import download_inventory
                manifest = await download_inventory(client, run, args.inventory_json, checkpoint_every=args.checkpoint_every)
        elif args.inventory_only or full_archive:
            from src.utils.oddsportal_inventory import collect_inventory
            inventory = await collect_inventory(client, inventory_run or run, archive_url=args.archive_url,
                                                known_events_csv=args.known_events_csv, max_matches=args.max_matches)
            if args.inventory_only:
                manifest = inventory
            elif inventory_run.stop_reason:
                run.stop_reason = inventory_run.stop_reason
                records = json.loads(inventory_run.json_path.read_bytes())
                for record in records:
                    record.update(odds_status="not_requested", odds_reason=f"inventory_stopped_{run.stop_reason}")
                manifest = run.finish(records, inventory_source_path=str(inventory_run.json_path),
                                      inventory_complete=False, inventory_stop_reason=run.stop_reason)
            else:
                from src.utils.oddsportal_bulk import download_inventory
                # Phase changes do not reset the source's global minimum request interval.
                run._last_request = inventory_run._last_request
                manifest = await download_inventory(client, run, inventory_run.json_path)
        else:
            manifest = await scrape(client, run, args.archive_url, event_url=args.event_url, max_matches=args.max_matches)
    print(f"{manifest['status']}: {manifest['rows']} rows, {manifest['available_rows']} with prices; {run.manifest_path}")
    return 0 if manifest["status"] == "COMPLETED" else 1


def archive_main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main("archive", argv))


def refresh_main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main("refresh", argv))
