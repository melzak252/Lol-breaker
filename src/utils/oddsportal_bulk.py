"""Download every validated inventoried event with append-only settlement evidence.

This is an acquisition of an explicit input inventory, not certification of the
provider's whole history. Transport, source validation and normalization remain
owned by oddsportal_collection; recovery only reparses already retained entities.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from src.utils import oddsportal_collection as c

_STATUSES = ("available", "not_available", "failed", "not_requested")


def _reject_nonfinite(value: str):
    raise ValueError(f"Nonfinite JSON value in inventory: {value}")


def _load_inventory(body: bytes) -> list[dict]:
    try:
        records = json.loads(body, parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid inventory JSON: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError("Inventory JSON must be an array of canonical event records")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Inventory record {index} must be an object")
        event_id, url = record.get("event_id"), record.get("url")
        if not isinstance(event_id, str) or not isinstance(url, str):
            raise ValueError(f"Inventory record {index} requires string event_id and url")
        c.source_url(url)
        if c.event_id_from_url(url) != event_id:
            raise ValueError(f"Inventory record {index} event ID disagrees with its source URL")
        if event_id in seen:
            raise ValueError(f"Duplicate inventory event ID: {event_id}")
        seen.add(event_id)
    return records


def _inventory_metadata(run: c.CollectionRun, path: Path, digest: str) -> dict:
    """Only a byte-bound inventory sidecar can establish discovery completeness."""
    result = {"inventory_complete": None, "inventory_completeness_status": "metadata_not_provided",
              "inventory_coverage_ledger_path": None}
    metadata_path = path.parent / "inventory-metadata.json"
    if not metadata_path.is_file():
        return result
    body = metadata_path.read_bytes()
    metadata_digest = hashlib.sha256(body).hexdigest()
    snapshot = run.raw_dir / f"inventory-metadata-{metadata_digest}.json"
    c._retain_bytes(snapshot, body)
    result.update(inventory_metadata_source_path=str(metadata_path.resolve()),
                  inventory_metadata_snapshot_path=str(snapshot),
                  inventory_metadata_sha256=metadata_digest)
    try:
        metadata = json.loads(body, parse_constant=_reject_nonfinite)
    except (ValueError, UnicodeError):
        result["inventory_completeness_status"] = "invalid_metadata"
        return result
    if not isinstance(metadata, dict) or metadata.get("inventory_sha256") != digest:
        result["inventory_completeness_status"] = "unverified_metadata_binding"
        return result
    completeness = metadata.get("inventory_complete")
    if type(completeness) is not bool:
        result["inventory_completeness_status"] = "metadata_has_no_completeness_decision"
        return result
    result.update(inventory_complete=completeness,
                  inventory_completeness_status="bound_inventory_metadata",
                  inventory_coverage_ledger_path=metadata.get("inventory_coverage_ledger_path"))
    return result


def _prepare_record(record: dict) -> None:
    original = dict(record)
    # The exact inventory and source originals are evidence, not current quotes.
    # _clear_legacy_odds also clears source_*; keep discovery provenance explicitly.
    provenance = {key: record[key] for key in ("source_archive_row", "source_occurrences", "source_season") if key in record}
    c._clear_legacy_odds(record)
    record.update(provenance)
    record.update(inventory_record=original, scraper_schema="oddsportal_v2",
                  odds_status="pending", odds_attempted=False,
                  quote_timing_semantics="unavailable_new_acquisition")


def _event_captures(captures: list[dict], event_id: str) -> list[dict]:
    shell_path = f"/proxy/ajax-event-data-shell/{event_id}/0/"
    nav_path = f"/proxy/ajax-event-nav/{event_id}/0/"
    market_path = re.compile(rf"/proxy/match-event/[1-9][0-9]*-36-{re.escape(event_id)}-[1-9][0-9]*-[1-9][0-9]*-[A-Za-z0-9_-]+\.dat\Z")
    history_prefix = f"/proxy/match-event-history/36-{event_id}-"
    return [capture for capture in captures
            if (path := urlsplit(capture["url"]).path) in (shell_path, nav_path)
            or market_path.fullmatch(path) or path.startswith(history_prefix)]


def _retain_valid_shell(run: c.CollectionRun, record: dict, captures: list[dict]) -> None:
    """Recover validated metadata from the exact retained shell, never HTTP fallback."""
    record["event_acquisition_captures"] = captures
    shell_path = f"/proxy/ajax-event-data-shell/{record['event_id']}/0/"
    for capture in reversed(captures):
        if urlsplit(capture["url"]).path != shell_path or capture["http_status"] != 200:
            continue
        body = Path(capture["body_path"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != capture["body_sha256"]:
            raise ValueError("Retained shell entity hash does not match its capture")
        response = httpx.Response(capture["http_status"], content=body,
                                  headers={key: value for key, value in capture["response_headers"].items()
                                           if key.lower() != "content-encoding"},
                                  request=httpx.Request("GET", capture["url"]))
        response.extensions["oddsportal_capture"] = capture
        try:
            payload, decoded_path, decoded_sha = c._decode_payload(run, response)
            shell = c.parse_event_shell(payload, record["event_id"])
        except ValueError as exc:
            record["shell_metadata_error"] = str(exc)
            return
        record.update(shell)
        record.update(c._response_metadata("shell", response, decoded_path, decoded_sha))
        record.update(tournament=shell["source_event_data"].get("tournamentName"),
                      event_metadata_status="parsed_current_shell")
        if run._bootstrap is not None:
            record["source_geo"] = run._bootstrap["geoIPcode"]
        return


async def download_inventory(client: httpx.AsyncClient, run: c.CollectionRun,
                             inventory_json: Path, *, checkpoint_every: int = 100) -> dict:
    """Settle every unique canonical input event; persist progress without full rewrites.

    Unexpected interruption is propagated after cancelling in-flight workers and
    fsyncing settled JSONL/progress. Pending events remain visible in that receipt;
    this function does not implement live resume or additional retry behavior.
    """
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    inventory_json = Path(inventory_json)
    body = inventory_json.read_bytes()
    # Full preflight precedes task creation, bootstrap reservation and any requests.
    records = _load_inventory(body)
    digest = hashlib.sha256(body).hexdigest()
    snapshot = run.raw_dir / f"inventory-{digest}.json"
    c._retain_bytes(snapshot, body)
    metadata = {
        "inventory_source_path": str(inventory_json.resolve()), "inventory_snapshot_path": str(snapshot),
        "inventory_sha256": digest, "total_expected_ids": len(records),
        "acquisition_scope": "all_events_in_validated_inventory", "whole_history_claimed": False,
        **_inventory_metadata(run, inventory_json, digest),
    }
    for record in records:
        _prepare_record(record)
    settled_path = run.derived_dir / "settled-records.jsonl"
    progress_path = run.reporting_dir / "progress.json"
    metadata.update(settled_records_path=str(settled_path), progress_path=str(progress_path))
    counts = dict.fromkeys(_STATUSES, 0)
    completed = attempted = next_index = 0
    tasks = []

    with settled_path.open("x", encoding="utf-8", newline="\n") as stream:
        def checkpoint(status: str) -> None:
            stream.flush()
            os.fsync(stream.fileno())
            # Deliberately excludes captures/raw-artifact inventories and row arrays.
            c._atomic_json(progress_path, {
                "schema": "oddsportal_bulk_progress_v1", "run_id": run.run_id,
                "status": status, "updated_at": c.utc_now(),
                "inventory_sha256": digest, "total_expected_ids": len(records),
                "completed": completed, "pending": len(records) - completed,
                "attempted": attempted, "not_requested": counts["not_requested"],
                "status_counts": dict(counts), "stop_reason": run.stop_reason,
                "settled_records_path": str(settled_path),
            })

        def settle(record: dict) -> None:
            nonlocal completed
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            counts[record["odds_status"]] += 1
            completed += 1
            if completed % checkpoint_every == 0:
                checkpoint("STOPPING" if run.stop_reason else "RUNNING")

        async def worker() -> None:
            nonlocal next_index, attempted
            while next_index < len(records) and not run.stop_reason:
                record = records[next_index]
                next_index += 1
                capture_start = len(run.captures)
                record["odds_attempted"] = True
                attempted += 1
                try:
                    details = await c.fetch_detailed_odds(client, run, record["event_id"])
                    c._apply_details(record, details)
                    record["odds_reason"] = (
                        "provider_open_or_last_moneyline_returned" if record["odds_status"] == "available"
                        else "provider_moneyline_has_no_supported_bookmaker_prices")
                except (ValueError, httpx.HTTPError, c.AcquisitionStopped) as exc:
                    captures = _event_captures(run.captures[capture_start:], record["event_id"])
                    _retain_valid_shell(run, record, captures)
                    if isinstance(exc, c.SourceProtocolChanged):
                        run.stop_reason = run.stop_reason or "provider_protocol_changed"
                    if isinstance(exc, c.AcquisitionStopped) and not captures:
                        record["odds_attempted"] = False
                        attempted -= 1
                        record.update(odds_status="not_requested",
                                      odds_reason=f"acquisition_stopped_{run.stop_reason or 'source_stopped'}",
                                      odds_error=str(exc))
                    else:
                        record.update(odds_status="failed", odds_reason=f"{type(exc).__name__}: {exc}",
                                      odds_error=str(exc))
                        c._failure(run, "detailed_odds", exc, event_id=record["event_id"])
                settle(record)

        checkpoint("RUNNING")
        try:
            # At most concurrency event coroutines exist; safe_request keeps its
            # shared request semaphore and global pace/stop authority underneath.
            tasks = [asyncio.create_task(worker()) for _ in range(min(run.concurrency, len(records)))]
            await asyncio.gather(*tasks)
            if run.stop_reason:
                for record in records[next_index:]:
                    record.update(odds_status="not_requested",
                                  odds_reason=f"acquisition_stopped_{run.stop_reason}",
                                  odds_error=f"Acquisition stopped: {run.stop_reason}; no request attempted")
                    settle(record)
            checkpoint("STOPPED" if run.stop_reason else "COMPLETED")
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            checkpoint("INTERRUPTED")
            raise
    return run.finish(records, **metadata, attempted_events=attempted, completed_events=completed,
                      not_requested_events=counts["not_requested"], acquisition_status_counts=dict(counts))
