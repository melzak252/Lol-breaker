"""Decode retained OddsPortal payloads without inventing identity or quote timing.

Provider opening/last price-change clocks are not collection clocks. The legacy
``closingOdds``/``*_close`` names remain export-compatible, not certified closes.
"""
from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import math
import re
import zlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlsplit

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

__all__ = [
    "decrypt_server_response",
    "parse_detailed_odds",
    "parse_archive_row",
    "event_id_from_url",
]

_BASE_URL = "https://www.oddsportal.com"
_MARKET_KEY = "E-3-2-0-0-0"
_EVENT_ID = re.compile(r"[A-Za-z0-9]{8}\Z")
_PATH_EVENT_ID = re.compile(r"-([A-Za-z0-9]{8})/?\Z")


def decrypt_server_response(enc: str, password: str, salt: str) -> str:
    """Strictly decode base64(ciphertext-base64:IV-hex), AES-CBC and PKCS7.

    The provider optionally gzips the plaintext before encryption. Malformed
    transport, padding, compression or UTF-8 raises ValueError, never empty JSON.
    """
    if not all(isinstance(value, str) for value in (enc, password, salt)):
        raise ValueError("Encrypted response, password and salt must be strings")
    try:
        envelope = base64.b64decode(enc, validate=True).decode("ascii")
        parts = envelope.split(":")
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{32}", parts[1]):
            raise ValueError("Invalid encrypted response envelope or IV")
        ciphertext = base64.b64decode(parts[0], validate=True)
        if not ciphertext or len(ciphertext) % AES.block_size:
            raise ValueError("Invalid encrypted response ciphertext length")
        key = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 1000, dklen=32
        )
        padded = AES.new(key, AES.MODE_CBC, bytes.fromhex(parts[1])).decrypt(ciphertext)
        plaintext = unpad(padded, AES.block_size, style="pkcs7")
        if plaintext.startswith(b"\x1f\x8b"):
            plaintext = gzip.decompress(plaintext)
        return plaintext.decode("utf-8")
    except (ValueError, binascii.Error, UnicodeError, OSError, EOFError, zlib.error) as exc:
        raise ValueError(f"Invalid encrypted OddsPortal response: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite numeric data") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite numeric data")
    return number


def _utc_epoch(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a numeric source epoch")
    epoch = _number(value, label)
    try:
        return datetime.fromtimestamp(epoch, timezone.utc).isoformat()
    except (ValueError, OverflowError, OSError) as exc:
        raise ValueError(f"{label} is not a representable UTC epoch") from exc


def _pair(value: Any, label: str) -> list:
    # Current PHP serialization can emit explicit array indices in either key order.
    # Arbitrary outcome/name-keyed objects still provide no supported orientation.
    if isinstance(value, Mapping) and set(value) == {"0", "1"}:
        return [value["0"], value["1"]]
    if isinstance(value, list) and len(value) == 2:
        return value
    raise ValueError(f"{label} must be a two-outcome [home, away] array or explicit 0/1 object")


def _price_map(value: Any, label: str) -> dict[str, list[float]]:
    result = {}
    for bid, pair in _mapping(value, label).items():
        if not isinstance(bid, str) or not bid:
            raise ValueError(f"{label} has an invalid bookmaker ID")
        prices = [_number(price, f"{label}.{bid}") for price in _pair(pair, f"{label}.{bid}")]
        if any(price <= 1 for price in prices):
            raise ValueError(f"{label}.{bid} decimal odds must be greater than one")
        result[bid] = prices
    return result


def parse_detailed_odds(payload: Mapping, event_id: str, bookies: Mapping[str, str]) -> dict[str, Any]:
    """Parse the exact two-way, full-event back market in provider side order.

    Missing prices stay missing. Source clocks and activation are retained as
    provider metadata, with unknown values left null rather than certified.
    """
    if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
        raise ValueError("Invalid requested event identity")
    data = _mapping(_mapping(payload, "payload").get("d"), "payload.d")
    if data.get("encodeventId") != event_id:
        raise ValueError("Payload event identity does not match requested event")
    for field, expected in (("bt", 3), ("sc", 2)):
        if type(data.get(field)) is not int or data[field] != expected:
            raise ValueError(f"Unsupported event market metadata: {field}")
    back = _mapping(_mapping(data.get("oddsdata"), "oddsdata").get("back"), "back")
    market = _mapping(back.get(_MARKET_KEY), f"market {_MARKET_KEY}")
    for field, expected in (("bettingTypeId", 3), ("scopeId", 2), ("handicapTypeId", 0), ("mixedParameterId", 0)):
        if type(market.get(field)) is not int or market[field] != expected:
            raise ValueError(f"Unsupported market metadata: {field}")
    if _number(market.get("handicapValue"), "handicapValue") != 0:
        raise ValueError("Unsupported market handicapValue")
    if market.get("mixedParameterName") is not None or market.get("isBack") is not True:
        raise ValueError("Unsupported mixed market or non-back market")
    outcomes = _pair(market.get("outcomeId"), "outcomeId")
    if any(not isinstance(value, str) or not value.strip() for value in outcomes) or outcomes[0] == outcomes[1]:
        raise ValueError("Market must identify two distinct source outcomes")

    opening = _price_map(market.get("openingOdd", {}), "openingOdd")
    last = _price_map(market.get("odds", {}), "odds")
    opening_clocks = _mapping({} if market.get("openingChangeTime") is None else market["openingChangeTime"], "openingChangeTime")
    last_clocks = _mapping({} if market.get("changeTime") is None else market["changeTime"], "changeTime")
    active = _mapping({} if market.get("act") is None else market["act"], "act")
    result: dict[str, Any] = {
        "openingsOdds": {},
        "closingOdds": {},
        "event_id": event_id,
        "market_key": _MARKET_KEY,
        "source_outcome_ids": list(outcomes),
        "source_open_change_at": {},
        "source_last_change_at": {},
        "source_bookmaker_active": {},
        "quote_timing_semantics": "provider_open_and_last_not_certified_close",
    }
    for bid, name in _mapping(bookies, "bookies").items():
        if not isinstance(bid, str) or not isinstance(name, str) or not name.strip():
            raise ValueError("Bookmaker mapping must contain string IDs and nonempty names")
        normalized = name.lower().replace(" ", "_")
        for prices, destination, suffix in ((opening, "openingsOdds", "open"), (last, "closingOdds", "close")):
            if bid in prices:
                result[destination][bid] = prices[bid]
                result[f"odds1_{normalized}_{suffix}"] = prices[bid][0]
                result[f"odds2_{normalized}_{suffix}"] = prices[bid][1]
        if not any(bid in mapping for mapping in (opening, last, opening_clocks, last_clocks, active)):
            continue
        for clocks, destination, suffix in ((opening_clocks, "source_open_change_at", "open"), (last_clocks, "source_last_change_at", "close")):
            raw = clocks.get(bid)
            pair = [None, None] if raw is None else _pair(raw, f"{destination}.{bid}")
            times = [_utc_epoch(value, f"{destination}.{bid}") for value in pair]
            result[destination][bid] = times
            result[f"odds1_{normalized}_{suffix}_at"] = times[0]
            result[f"odds2_{normalized}_{suffix}_at"] = times[1]
        status = active.get(bid)
        result["source_bookmaker_active"][bid] = status if isinstance(status, bool) else None
    return result


def event_id_from_url(url: str) -> str:
    """Extract an exact event ID from an OddsPortal path or evidenced #ID form.

    Root-relative provider URLs are accepted for archive rows. Off-host URLs,
    absent IDs and contradictory path/fragment IDs raise ValueError.
    """
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("Missing or invalid event URL")
    try:
        parsed = urlsplit(url)
        if parsed.netloc:
            if (parsed.scheme not in ("", "http", "https")
                    or parsed.hostname not in ("oddsportal.com", "www.oddsportal.com")
                    or parsed.username is not None or parsed.password is not None
                    or parsed.port is not None):
                raise ValueError("Event URL must be on the OddsPortal host")
        elif parsed.scheme or not parsed.path.startswith("/") or url.startswith("//"):
            raise ValueError("Event URL must be an OddsPortal URL or root-relative path")
    except ValueError as exc:
        raise ValueError(f"Invalid event URL: {exc}") from exc
    # H2H paths identify two participants; only their fragment identifies an event.
    path_match = None if "/esports/h2h/" in parsed.path else _PATH_EVENT_ID.search(parsed.path)
    path_id = path_match.group(1) if path_match else None
    fragment_id = parsed.fragment if _EVENT_ID.fullmatch(parsed.fragment) else None
    if parsed.fragment and fragment_id is None:
        raise ValueError("Unsupported event URL fragment")
    if path_id and fragment_id and path_id != fragment_id:
        raise ValueError("Conflicting event identities in URL")
    event_id = path_id or fragment_id
    if event_id is None:
        raise ValueError("Event URL has no encoded event identity")
    return event_id


def parse_archive_row(row: Mapping) -> dict[str, Any]:
    """Retain archive fixture identity and results with an aware UTC start.

    Only explicit winner flags establish a winner/draw; missing or unplayed
    results are unknown, even if displayed scores are equal.
    """
    row = _mapping(row, "archive row")
    event_id = row.get("encodeEventId")
    if not isinstance(event_id, str) or not _EVENT_ID.fullmatch(event_id):
        raise ValueError("Archive row is missing a valid encoded event identity")
    if event_id_from_url(row.get("url")) != event_id:
        raise ValueError("Archive URL event identity does not match encodeEventId")
    home, away = row.get("home-name"), row.get("away-name")
    if any(not isinstance(name, str) or not name.strip() for name in (home, away)):
        raise ValueError("Archive row is missing participants")
    if " ".join(home.split()).casefold() == " ".join(away.split()).casefold():
        raise ValueError("Archive row must identify distinct participants")
    home_flag, away_flag = row.get("home-winner"), row.get("away-winner")
    if (home_flag == "win" and away_flag in ("win", "draw")) or (away_flag == "win" and home_flag == "draw"):
        raise ValueError("Archive winner flags conflict")
    winner = None
    if home_flag == "win":
        winner = "home"
    elif away_flag == "win":
        winner = "away"
    elif home_flag == "draw" and away_flag == "draw":
        winner = "draw"
    return {
        "date": _utc_epoch(row.get("date-start-base"), "date-start-base"),
        "date-start-base": row.get("date-start-base"),
        "tournament": row.get("tournament-name"),
        "home_team": home,
        "away_team": away,
        "homeResult": row.get("homeResult"),
        "awayResult": row.get("awayResult"),
        "home-winner": home_flag,
        "away-winner": away_flag,
        "winner": winner,
        "name": f"{home} - {away}",
        "url": urljoin(_BASE_URL, row["url"]),
        "encodeEventId": event_id,
        "event_id": event_id,
    }
