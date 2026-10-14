"""Exhaust published LoL archive inventory without acquiring or certifying odds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from src.utils import oddsportal_collection as c
from src.utils.oddsportal_current import _archive_props, _provider_path

_METADATA_FIELDS = ("url", "home_team", "away_team", "date", "tournament", "homeResult", "awayResult", "winner")
_RECOVERABLE = (ValueError, httpx.HTTPError, c.AcquisitionStopped)


def _season_key(url: str) -> tuple[str, ...]:
    c.source_url(url)
    parsed = urlsplit(url)
    parts = _provider_path(url)
    if (parsed.query or parsed.fragment or len(parts) != 4
            or parts[:2] != ("esports", "league-of-legends") or parts[3] != "results"):
        raise ValueError("Advertised season URL must identify an exact LoL results archive")
    return parts


def _capture(run: c.CollectionRun, url: str) -> dict | None:
    # A failed HTTP request still retains its body before raising for status.
    return next((item for item in reversed(run.captures) if item["url"] == url), None)


def _fail(run: c.CollectionRun, ledger: dict, stage: str, error: Exception, **metadata) -> None:
    if isinstance(error, c.SourceProtocolChanged) and not run.stop_reason:
        run.stop_reason = "provider_protocol_changed"
    c._failure(run, stage, error, **metadata)
    ledger.setdefault("failures", []).append(run.failures[-1])
    ledger["status"] = "failed"


class _Events:
    """Exact-ID union with append-only source occurrences, not fuzzy deduplication."""

    def __init__(self, run: c.CollectionRun, max_matches: int | None):
        self.run = run
        self.limit = max_matches
        self.records: dict[str, dict] = {}
        self.variant_keys: dict[str, set[str]] = {}
        self.progress_path = run.derived_dir / "inventory-progress.jsonl"
        self.stream = self.progress_path.open("x", encoding="utf-8")
        self.occurrences = 0
        self.omitted_occurrences = 0

    @property
    def bounded(self) -> bool:
        return self.limit is not None and len(self.records) >= self.limit

    def add(self, metadata: dict, occurrence: dict) -> bool:
        event_id = metadata["event_id"]
        record = self.records.get(event_id)
        if record is None and self.bounded:
            self.omitted_occurrences += 1
            return False
        if record is None:
            record = {**metadata, "scraper_schema": "oddsportal_inventory_v1",
                      "metadata_variants": [], "metadata_conflicts": {},
                      "canonical_metadata_source": occurrence["kind"], "legacy_identity_certified": False}
            c._clear_legacy_odds(record)
            record.update(source_occurrences=[], odds_status="not_requested_inventory",
                          quote_timing_semantics="unavailable_inventory")
            self.records[event_id] = record
            self.variant_keys[event_id] = set()
        elif occurrence["kind"] == "archive" and record["canonical_metadata_source"] != "archive":
            record.update(metadata)
            record["canonical_metadata_source"] = "archive"
        record["source_occurrences"].append(occurrence)
        if occurrence["kind"] == "archive":
            if "source_archive_row" not in record:
                record["source_archive_row"] = occurrence["source_row"]
                record.update({key: occurrence[key] for key in (
                    "archive_capture", "archive_response_sha256", "archive_decoded_payload_path",
                    "archive_decoded_payload_sha256", "source_season")})
        elif "legacy_row" not in record:
            record["legacy_row"] = occurrence["source_row"]
        variant = {"kind": occurrence["kind"], "metadata": metadata}
        signature = json.dumps(variant, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if signature not in self.variant_keys[event_id]:
            self.variant_keys[event_id].add(signature)
            record["metadata_variants"].append(variant)
        self.occurrences += 1
        self.stream.write(json.dumps({"event_id": event_id, "occurrence": occurrence, "metadata": metadata},
                                     ensure_ascii=False, allow_nan=False) + "\n")
        return True

    def checkpoint(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.checkpoint()
        self.stream.close()

    def finalize(self) -> list[dict]:
        for record in self.records.values():
            for field in _METADATA_FIELDS:
                values = []
                for variant in record["metadata_variants"]:
                    value = variant["metadata"].get(field)
                    if value not in values:
                        values.append(value)
                if len(values) > 1:
                    record["metadata_conflicts"][field] = values
            record["event_metadata_status"] = "validated_archive" if record["canonical_metadata_source"] == "archive" else "unknown_legacy"
            record["date_status"] = "validated_archive_utc" if record["canonical_metadata_source"] == "archive" and record.get("date") else "unknown"
        return list(self.records.values())


def _legacy_inventory(run: c.CollectionRun, events: _Events, path: Path, ledger: dict) -> dict:
    source_bytes = path.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    snapshot = run.raw_dir / f"input-{digest}.csv"
    c._retain_bytes(snapshot, source_bytes)
    receipt = {"source_path": str(path.resolve()), "snapshot_path": str(snapshot), "sha256": digest,
               "bytes": len(source_bytes), "row_count": 0, "valid_rows": 0, "invalid_rows": 0,
               "legacy_identity_certified": False, "status": "attempted"}
    ledger["known_events"] = receipt
    try:
        rows = c.read_legacy_csv(snapshot)
    except ValueError as exc:
        _fail(run, receipt, "inventory_legacy_csv", exc, snapshot_path=str(snapshot), source_sha256=digest)
        return receipt
    receipt["row_count"] = len(rows)
    for index, row in enumerate(rows):
        try:
            event_id = c.event_id_from_url(row.get("url"))
            url = c.source_url(urljoin(c.BASE_URL, row["url"]))
            for field in ("event_id", "encodeEventId"):
                if row.get(field) and row[field] != event_id:
                    raise ValueError(f"Legacy {field} conflicts with exact URL event identity")
            metadata = {"event_id": event_id, "url": url, "date": None,
                        **{key: row.get(key) or None for key in ("home_team", "away_team", "tournament")},
                        "homeResult": None, "awayResult": None, "winner": None}
            events.add(metadata, {"kind": "legacy_csv", "row_index": index, "source_row": row,
                                  "snapshot_path": str(snapshot), "source_sha256": digest,
                                  "legacy_identity_certified": False, "legacy_date_certified": False})
            receipt["valid_rows"] += 1
        except ValueError as exc:
            receipt["invalid_rows"] += 1
            c._failure(run, "inventory_legacy_row", exc, row_index=index, source_row=row,
                       snapshot_path=str(snapshot), source_sha256=digest)
            receipt.setdefault("failures", []).append(run.failures[-1])
    receipt["status"] = "completed" if not receipt["invalid_rows"] else "partial"
    events.checkpoint()
    return receipt


def _archive_rows(run: c.CollectionRun, events: _Events, season: dict, page: dict,
                  response: httpx.Response, data: dict) -> None:
    body = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    decoded_path = run.derived_dir / f"archive-{digest}.json"
    c._retain_bytes(decoded_path, body)
    capture = response.extensions["oddsportal_capture"]
    page.update(capture=capture, row_count=len(data["rows"]), valid_rows=0, invalid_rows=0,
                added_occurrences=0, decoded_payload_path=str(decoded_path), decoded_payload_sha256=digest)
    for index, row in enumerate(data["rows"]):
        try:
            if not isinstance(row, dict) or type(row.get("sport-id")) is not int or row["sport-id"] != 36:
                raise ValueError("Archive row must identify current esports sport-id=36")
            for field in ("tournament-id", "tournament_id"):
                if field in row and row[field] != data["activeSeason"].get("id"):
                    raise ValueError("Archive row belongs to another tournament season")
            if row.get("tournament-url", _season_key(season["url"])[2]) != _season_key(season["url"])[2]:
                raise ValueError("Archive row belongs to another league archive")
            metadata = c.parse_archive_row(row)
            for field in c.SUPERSEDED_IDENTITY_ALIASES:
                metadata.pop(field, None)
            added = events.add(metadata, {"kind": "archive", "page_url": str(response.request.url),
                "season_url": season["url"], "page": page["page"], "row_index": index,
                "source_row": row, "source_season": data["activeSeason"],
                "archive_capture": capture, "archive_response_sha256": capture["body_sha256"],
                "archive_decoded_payload_path": str(decoded_path), "archive_decoded_payload_sha256": digest})
            page["valid_rows"] += 1
            page["added_occurrences"] += int(added)
        except ValueError as exc:
            page["invalid_rows"] += 1
            c._failure(run, "archive_row", exc, page_url=str(response.request.url), season_url=season["url"],
                       row_index=index, source_row=row, archive_capture=capture,
                       archive_decoded_payload_path=str(decoded_path), archive_decoded_payload_sha256=digest)
            page.setdefault("failures", []).append(run.failures[-1])
    page["status"] = "completed" if not page["invalid_rows"] else "partial"
    page["completed"] = not page["invalid_rows"]
    events.checkpoint()


async def collect_inventory(client: httpx.AsyncClient, run: c.CollectionRun, *, archive_url=None,
                            known_events_csv=None, max_matches=None) -> dict:
    """Union every published league/season/page with all valid historical CSV IDs.

    HTTP denial, challenge and explicit protocol stops remain global. Ordinary
    malformed/404 seasons and pages are retained locally and do not erase later
    independent sources. A missing source is not certified as complete coverage.
    """
    c._validate_max_matches(max_matches)
    ledger_path = run.derived_dir / "inventory-coverage.json"
    metadata_path = run.derived_dir / "inventory-metadata.json"
    ledger = {"schema": "oddsportal_inventory_coverage_v1", "catalogue": {
        "url": archive_url or c.TOURNAMENTS_URL, "attempted": False, "status": "not_attempted"},
        "leagues": [], "seasons": [], "source_absences": [], "max_matches": max_matches}
    events = _Events(run, max_matches)
    seasons: dict[tuple[str, ...], dict] = {}
    known_events = None

    def register(url: str, league: dict, advertisement: dict | None = None, capture=None) -> dict:
        key = _season_key(url)
        if key not in seasons:
            seasons[key] = {"url": url, "attempted": False, "completed": False,
                            "status": "not_attempted", "advertisements": [], "pages": []}
            ledger["seasons"].append(seasons[key])
        season = seasons[key]
        if season["url"] not in league["season_urls"]:
            league["season_urls"].append(season["url"])
        if advertisement is not None:
            season["advertisements"].append({"source_league_url": league["url"],
                                            "source_season": advertisement, "capture": capture})
        return season

    def advertise(parsed: dict, league: dict, capture: dict) -> list[str]:
        references = []
        for index, tab in enumerate(parsed["seasons"]):
            try:
                if not isinstance(tab, dict) or not isinstance(tab.get("href"), str) or not tab["href"]:
                    raise ValueError("Invalid current tournament season advertisement")
                advertised = register(c.source_url(urljoin(c.BASE_URL, tab["href"])), league, tab, capture)
                if advertised["url"] not in references:
                    references.append(advertised["url"])
            except ValueError as exc:
                c._failure(run, "inventory_season_advertisement", exc, league_url=league["url"],
                           advertisement_index=index, source_season=tab, capture=capture)
                league.setdefault("failures", []).append(run.failures[-1])
        return references

    async def collect_season(season: dict, league: dict) -> None:
        season.update(attempted=True, status="attempted")
        try:
            response = await c._fetch_archive_response(client, run, season["url"])
            season["capture"] = response.extensions["oddsportal_capture"]
            try:
                parsed = c.parse_current_archive(response.text, season["url"])
            except ValueError:
                # Reject mismatched rows, but independently visit explicit LoL links.
                # Stale current aliases can still advertise valid historical seasons.
                initial = _archive_props(response.text)["initial"]
                if initial.get("sport") == "esports" and isinstance(initial.get("seasonTabs"), list):
                    season["advertised_season_urls"] = advertise(
                        {"seasons": initial["seasonTabs"]}, league, season["capture"])
                raise
            data = parsed["source_initial"]
            season["active_season"] = parsed["active_season"]
            season["source_context"] = parsed["source_context"]
            season["advertised_season_urls"] = advertise(parsed, league, season["capture"])
            if not isinstance(parsed["active_season"].get("url"), str):
                raise ValueError("Archive requires explicit active-season source identity")
            if parsed["archive_status"] == "no_source_rows":
                absence = {"reason": "archive_season_no_source_rows", "url": season["url"],
                           "season": parsed["active_season"], "capture": season["capture"],
                           "observed_rows": data["rows"], "observed_pagination": data["pagination"]}
                run.source_absences.append(absence)
                ledger["source_absences"].append(absence)
                season.update(status="source_absent", completed=True, row_count=0, advertised_page_count=None)
                return
            if parsed["pagination"]["activePage"] != 1:
                raise ValueError("Initial archive response must be advertised page one")
            page_count = parsed["pagination"]["pageCount"]
            season["advertised_page_count"] = page_count
            season["pages"] = [{"page": page, "attempted": False, "completed": False,
                                "status": "not_attempted"} for page in range(1, page_count + 1)]
            if parsed["archive_status"] == "empty_page":
                absence = {"reason": "archive_season_empty_page", "url": season["url"],
                           "season": parsed["active_season"], "capture": season["capture"],
                           "observed_rows": data["rows"], "observed_pagination": data["pagination"]}
                run.source_absences.append(absence)
                ledger["source_absences"].append(absence)
            for page in season["pages"]:
                if run.stop_reason or events.bounded:
                    break
                page.update(attempted=True, status="attempted")
                capture_count = len(run.captures)
                attempt_capture = response.extensions["oddsportal_capture"] if page["page"] == 1 else None
                try:
                    if page["page"] == 1:
                        page_response, page_data = response, data
                    else:
                        page_response, page_data = await c._fetch_archive_page(client, run, data, page["page"])
                    page["url"] = str(page_response.request.url)
                    _archive_rows(run, events, season, page, page_response, page_data)
                except _RECOVERABLE as exc:
                    # Pagination locators/identity are checked by the shared helper.
                    capture = run.captures[-1] if len(run.captures) > capture_count else attempt_capture
                    _fail(run, page, "inventory_archive_page", exc, season_url=season["url"],
                          page=page["page"], capture=capture)
            season["row_count"] = sum(page.get("row_count", 0) for page in season["pages"])
            season["completed"] = all(page["completed"] for page in season["pages"])
            season["status"] = "completed" if season["completed"] else "partial"
            if parsed["archive_status"] == "empty_page" and season["completed"]:
                season["status"] = "source_absent"
        except _RECOVERABLE as exc:
            _fail(run, season, "inventory_archive_season", exc, season_url=season["url"],
                  capture=season.get("capture") or _capture(run, season["url"]))

    try:
        if known_events_csv is not None:
            try:
                known_events = _legacy_inventory(run, events, Path(known_events_csv), ledger)
            except (ValueError, OSError) as exc:
                _fail(run, ledger, "inventory_legacy_csv", exc, source_path=str(known_events_csv))
        catalogue = ledger["catalogue"]
        urls = []
        if archive_url is not None:
            try:
                url = c.source_url(archive_url)
                _season_key(url)
                urls = [url]
                catalogue.update(status="explicit_archive", attempted=False)
            except ValueError as exc:
                _fail(run, catalogue, "inventory_archive_locator", exc, source_url=archive_url)
        elif not events.bounded:
            catalogue.update(attempted=True, status="attempted")
            try:
                urls = await c.get_tournament_links(client, run)
                catalogue.update(status="completed", capture=_capture(run, c.TOURNAMENTS_URL))
            except _RECOVERABLE as exc:
                _fail(run, catalogue, "inventory_catalogue", exc, capture=_capture(run, c.TOURNAMENTS_URL))
        for url in urls:
            ledger["leagues"].append({"url": url, "attempted": False, "completed": False,
                                     "status": "not_attempted", "season_urls": []})
        for league in ledger["leagues"]:
            if run.stop_reason or events.bounded:
                break
            league.update(attempted=True, status="attempted")
            try:
                season = register(league["url"], league)
                if not season["attempted"]:
                    # The league's SSR request is also season page one; never refetch it.
                    await collect_season(season, league)
                else:
                    for url_ref in season.get("advertised_season_urls", []):
                        register(url_ref, league)
                # New advertisements may add renamed seasons while walking this list.
                offset = 0
                while offset < len(league["season_urls"]) and not run.stop_reason and not events.bounded:
                    pending = seasons[_season_key(league["season_urls"][offset])]
                    offset += 1
                    if not pending["attempted"]:
                        await collect_season(pending, league)
                league["completed"] = all(seasons[_season_key(ref)]["completed"] for ref in league["season_urls"])
                league["status"] = "completed" if league["completed"] and not league.get("failures") else "partial"
            except _RECOVERABLE as exc:
                _fail(run, league, "inventory_archive_league", exc, league_url=league["url"],
                      capture=_capture(run, league["url"]))
    finally:
        events.close()
    complete = (max_matches is None and not run.stop_reason and not run.failures and not run.source_absences
                and all(league["completed"] for league in ledger["leagues"])
                and ledger["catalogue"]["status"] in {"completed", "explicit_archive"})
    records = events.finalize()
    inventory_digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, allow_nan=False)
    for chunk in encoder.iterencode(records):
        inventory_digest.update(chunk.encode("utf-8"))
    inventory_digest.update(b"\n")
    ledger.update(inventory_complete=complete, stop_reason=run.stop_reason, unique_events=len(records),
                  retained_occurrences=events.occurrences, omitted_occurrences=events.omitted_occurrences,
                  failures=run.failures)
    c._atomic_json(ledger_path, ledger)
    metadata = {"inventory_complete": complete, "inventory_sha256": inventory_digest.hexdigest(),
                "inventory_coverage_ledger_path": str(ledger_path),
                "inventory_progress_path": str(events.progress_path), "inventory_metadata_path": str(metadata_path),
                "known_events": known_events, "max_matches": max_matches, "unique_events": len(records),
                "retained_occurrences": events.occurrences, "omitted_occurrences": events.omitted_occurrences,
                "inventory_only": True, "legacy_identity_certified": False,
                "source_url": archive_url or c.TOURNAMENTS_URL}
    c._atomic_json(metadata_path, metadata)
    return run.finish(records, **metadata)
