"""Parse retained current OddsPortal shells and history without transport or inference.

History is the provider's returned change list, not a certified complete chronology.
The retained sample contains initial quotes and omits the terminal prices supplied
separately by the market response. Neither source clock is a capture/publication
clock, and H2H locator participant order does not establish market side order.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from src.utils.oddsportal_data import (
    _EVENT_ID,
    _mapping,
    _number,
    _utc_epoch,
    event_id_from_url,
)

__all__ = ["parse_event_shell", "parse_price_history", "parse_current_archive"]

_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?\Z")
_STATE_FIELDS = (
    "isLive", "realLive", "isPostponed", "isStarted", "isFinished",
    "isFinishedGracePeriod",
)
_REQUEST_BASE_FIELDS = (
    "requestBasePreMatch", "requestBaseOddsHistory", "updateScoreRequestBase",
)


def _success_data(payload: Mapping) -> Mapping:
    payload = _mapping(payload, "payload")
    if type(payload.get("s")) is not int or payload["s"] != 1:
        raise ValueError("OddsPortal response does not report success s=1")
    return _mapping(payload.get("d"), "payload.d")


def _score(value: Any) -> float | None:
    try:
        score = _number(value, "result score")
    except ValueError:
        return None
    return score if score >= 0 else None


def parse_event_shell(payload: Mapping, expected_event_id: str) -> dict[str, Any]:
    """Validate the current event shell and retain source identity/state/results.

    Only an explicit finished flag plus finite nonnegative unequal scores supports
    a winner. Missing/scheduled/tied results remain unknown, not fabricated draws.
    Source dictionaries and request bases are retained as metadata, not authority
    to perform requests or certify their availability time.
    """
    if not isinstance(expected_event_id, str) or not _EVENT_ID.fullmatch(expected_event_id):
        raise ValueError("Invalid expected event identity")
    data = _success_data(payload)
    event = _mapping(data.get("eventData"), "eventData")
    body = _mapping(data.get("eventBody"), "eventBody")
    if event.get("id") != expected_event_id:
        raise ValueError("Event shell identity does not match requested event")
    if type(event.get("sportId")) is not int or event["sportId"] != 36:
        raise ValueError("Event shell must identify esports sportId=36")
    version = event.get("versionId")
    if type(version) is not int or version <= 0:
        raise ValueError("Event shell versionId must be a positive integer")
    home, away = event.get("home"), event.get("away")
    if any(not isinstance(name, str) or not name.strip() for name in (home, away)):
        raise ValueError("Event shell must identify both participants")
    if " ".join(home.split()).casefold() == " ".join(away.split()).casefold():
        raise ValueError("Event shell must identify distinct participants")

    # No tab/locator is invented when absent; a supplied prematch URL must agree.
    if "tabs" in body:
        tabs = _mapping(body["tabs"], "eventBody.tabs")
        if "eventDetail" in tabs:
            detail = _mapping(tabs["eventDetail"], "tabs.eventDetail")
            if "prematch" in detail:
                prematch = _mapping(detail["prematch"], "tabs.eventDetail.prematch")
                if "url" in prematch and event_id_from_url(prematch["url"]) != expected_event_id:
                    raise ValueError("Prematch tab event identity does not match requested event")

    home_result, away_result = body.get("homeResult"), body.get("awayResult")
    winner = None
    if event.get("isFinished") is True:
        home_score, away_score = _score(home_result), _score(away_result)
        if home_score is not None and away_score is not None and home_score != away_score:
            winner = "home" if home_score > away_score else "away"
    result = {
        "event_id": expected_event_id,
        "home_team": home,
        "away_team": away,
        "date": _utc_epoch(body.get("startDate"), "eventBody.startDate"),
        "homeResult": home_result,
        "awayResult": away_result,
        "results": {"home": home_result, "away": away_result},
        "winner": winner,
        "version_id": version,
        "sport_id": event["sportId"],
        "source_event_data": dict(event),
        "source_event_body": dict(body),
        "request_bases": {field: data.get(field) for field in _REQUEST_BASE_FIELDS},
    }
    result.update({field: event.get(field) for field in _STATE_FIELDS})
    return result


def parse_price_history(
    payload: Mapping, expected_outcome_ids: Sequence[str], bookmaker_id: str,
) -> list[dict[str, Any]]:
    """Normalize the exact requested back history in evidenced outcome order.

    Require two distinct expected outcomes and exactly the requested bookmaker.
    Never suppress unsupported/foreign/malformed rows or treat missing history
    as an empty successful observation. Row ordering and raw values are retained;
    no row is labelled a terminal close or a complete opening-to-close series.
    """
    if (not isinstance(expected_outcome_ids, (list, tuple))
            or len(expected_outcome_ids) != 2
            or any(not isinstance(value, str) or not value.strip() for value in expected_outcome_ids)
            or expected_outcome_ids[0] == expected_outcome_ids[1]):
        raise ValueError("History requires two distinct expected source outcome IDs")
    if (not isinstance(bookmaker_id, str) or not bookmaker_id
            or bookmaker_id != bookmaker_id.strip()):
        raise ValueError("History requires an exact nonempty bookmaker ID string")
    data = _success_data(payload)
    history = _mapping(data.get("history"), "history")
    back = _mapping(history.get("back"), "history.back")
    if set(back) != set(expected_outcome_ids):
        raise ValueError("History outcomes do not exactly match requested market outcomes")
    if "lay" in history and history["lay"] != []:
        raise ValueError("Unsupported nonempty or malformed lay history")
    result = []
    for outcome_index, outcome_id in enumerate(expected_outcome_ids):
        providers = _mapping(back[outcome_id], f"history.back.{outcome_id}")
        if set(providers) != {bookmaker_id}:
            raise ValueError("History provider does not exactly match requested bookmaker")
        rows = providers[bookmaker_id]
        if not isinstance(rows, list) or not rows:
            raise ValueError("History provider must supply a nonempty source row array")
        for row_index, row in enumerate(rows):
            label = f"history.back.{outcome_id}.{bookmaker_id}[{row_index}]"
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{label} must be [decimal-string odds, volume, epoch]")
            raw_odds, raw_volume, epoch = row
            if not isinstance(raw_odds, str) or not _DECIMAL.fullmatch(raw_odds):
                raise ValueError(f"{label} odds must be a decimal string")
            odds = _number(raw_odds, f"{label}.odds")
            if odds <= 1:
                raise ValueError(f"{label} decimal odds must be greater than one")
            if isinstance(raw_volume, bool) or not isinstance(raw_volume, (int, float)):
                raise ValueError(f"{label} volume must be finite nonnegative numeric data")
            volume = _number(raw_volume, f"{label}.volume")
            if volume < 0:
                raise ValueError(f"{label} volume must be nonnegative")
            if epoch is None:
                raise ValueError(f"{label} must include its source change epoch")
            changed_at = _utc_epoch(epoch, f"{label}.epoch")
            result.append({
                "outcome_id": outcome_id,
                "outcome_index": outcome_index,
                "bookmaker_id": bookmaker_id,
                "odds": odds,
                "volume": volume,
                "source_epoch": epoch,
                "source_change_at": changed_at,
                "source_row_index": row_index,
                "source_row": list(row),
                "quote_timing_semantics": "provider_history_change_not_certified_complete",
            })
    return result


def _provider_path(url: str) -> tuple[str, ...]:
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("Archive context URL must be a nonempty exact string")
    try:
        parsed = urlsplit(url)
        if parsed.netloc:
            if (parsed.scheme not in ("", "http", "https")
                    or parsed.hostname not in ("oddsportal.com", "www.oddsportal.com")
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port is not None):
                raise ValueError("Archive context URL must be on the OddsPortal host")
        elif parsed.scheme or not parsed.path.startswith("/") or url.startswith("//"):
            raise ValueError("Archive context URL must be root-relative or on-host")
    except ValueError as exc:
        raise ValueError(f"Invalid archive context URL: {exc}") from exc
    parts = tuple(part for part in parsed.path.split("/") if part)
    if parts and re.fullmatch(r"[a-z]{2}", parts[0]):
        parts = parts[1:]
    return parts


def _archive_props(html: str) -> Mapping:
    if not isinstance(html, str) or not html:
        raise ValueError("Archive HTML must be a nonempty string")
    decoder = json.JSONDecoder()
    chunks = []
    for script in re.finditer(r"<script\b[^>]*>(.*?)</script\s*>", html, re.DOTALL | re.IGNORECASE):
        content = script.group(1)
        for call in re.finditer(r"self\.__next_f\.push\(\s*", content):
            try:
                argument, end = decoder.raw_decode(content, call.end())
            except json.JSONDecodeError as exc:
                raise ValueError("Malformed archive Flight JSON argument") from exc
            if not content[end:].lstrip().startswith(")"):
                raise ValueError("Malformed archive Flight call")
            if not isinstance(argument, list) or not argument:
                raise ValueError("Malformed archive Flight chunk")
            if type(argument[0]) is int and argument[0] == 1:
                if len(argument) != 2 or not isinstance(argument[1], str):
                    raise ValueError("Archive Flight data chunk must contain a string")
                chunks.append(argument[1])
    candidates = []
    for line in "".join(chunks).splitlines():
        _, separator, record = line.partition(":")
        if not separator:
            continue
        try:
            value = json.loads(record)
        except json.JSONDecodeError:
            # Flight also contains non-JSON module/import/string records.
            continue
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                initial = item.get("initial")
                if (item.get("sport") == "esports" and isinstance(initial, dict)
                        and all(key in initial for key in ("rows", "pagination", "activeSeason", "seasonTabs"))):
                    candidates.append(item)
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
    if len(candidates) != 1:
        raise ValueError("Archive Flight must contain exactly one esports archive props object")
    return candidates[0]


def parse_current_archive(html: str, expected_url: str) -> dict[str, Any]:
    """Read an observed Next Flight league archive without evaluating JavaScript.

    Preserve every raw event row and composite SSR price map. Individual malformed
    events are left for the collector's strict archive-row parser and retained
    failure accounting; contradictory supplied sport/game/tournament context
    invalidates the page itself. These SSR prices do not establish detailed quote
    orientation, opening/last prices or source clocks.
    Observed empty-season shells with rows=[] and pagination=[] are explicit
    no_source_rows absences, not successful historical quote collections. A valid
    paginated empty page remains distinct; raw initial metadata is never repaired.
    """
    expected = _provider_path(expected_url)
    if (len(expected) != 4 or expected[:2] != ("esports", "league-of-legends")
            or expected[3] != "results"):
        raise ValueError("Expected archive URL must identify a League of Legends tournament results page")
    props = _archive_props(html)
    initial = _mapping(props["initial"], "archive.initial")
    if "sport" in initial and initial["sport"] != "esports":
        raise ValueError("Archive initial sport does not match requested esports source")
    rows = initial["rows"]
    seasons = initial["seasonTabs"]
    if not isinstance(rows, list) or not isinstance(seasons, list):
        raise ValueError("Archive rows and seasonTabs must be arrays")
    raw_pagination = initial["pagination"]
    no_source_rows = not rows and isinstance(raw_pagination, list) and not raw_pagination
    pagination = None
    if no_source_rows:
        if initial.get("sport") != "esports":
            raise ValueError("Empty source archive requires explicit esports identity")
    else:
        pagination = _mapping(raw_pagination, "archive.pagination")
        page_count, active_page = pagination.get("pageCount"), pagination.get("activePage")
        if (type(page_count) is not int or page_count <= 0
                or type(active_page) is not int or not 1 <= active_page <= page_count):
            raise ValueError("Archive pagination requires positive pageCount and in-range activePage")
    active = _mapping(initial["activeSeason"], "archive.activeSeason")
    if no_source_rows and "url" not in active:
        raise ValueError("Empty source archive requires an explicit matching active-season URL")
    if "url" in active and _provider_path(active["url"]) != expected:
        raise ValueError("Archive active season URL does not match requested tournament")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if "sport-id" in row and (type(row["sport-id"]) is not int or row["sport-id"] != 36):
            raise ValueError("Archive row sport identity does not match esports")
        if "sport-url-name" in row and row["sport-url-name"] != "esports":
            raise ValueError("Archive row sport URL does not match esports")
        if "country-name" in row and row["country-name"] != "League of Legends":
            raise ValueError("Archive row game does not match League of Legends")
        if "tournament-url" in row and row["tournament-url"] != expected[2]:
            raise ValueError("Archive row tournament does not match requested league")
        if "superTemplate" in row:
            game = _mapping(row["superTemplate"], "archive row superTemplate")
            if "url" in game and game["url"] != expected[1]:
                raise ValueError("Archive row game URL does not match requested game")
        if "breadcrumbs" in row:
            breadcrumbs = _mapping(row["breadcrumbs"], "archive row breadcrumbs")
            for key, parts in (("sport", expected[:1]), ("country", expected[:2]), ("tournament", expected[:3])):
                if key in breadcrumbs:
                    crumb = _mapping(breadcrumbs[key], f"archive breadcrumbs.{key}")
                    if "url" in crumb and _provider_path(crumb["url"]) != parts:
                        raise ValueError(f"Archive {key} breadcrumb does not match requested context")
    return {
        "rows": rows,
        "seasons": seasons,
        "archive_status": "no_source_rows" if no_source_rows else ("rows_available" if rows else "empty_page"),
        "pagination": None if pagination is None else dict(pagination),
        "active_season": dict(active),
        "source_initial": dict(initial),
        "source_context": {key: value for key, value in props.items() if key != "initial"},
        "expected_url": expected_url,
    }
