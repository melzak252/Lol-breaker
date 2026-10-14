"""Persistent, host-wide request budgeting and response caching for Liquipedia."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_CACHE_DIR = Path("data/cache/liquipedia")
DEFAULT_CACHE_TTL_SECONDS = 6 * 60 * 60
NEGATIVE_CACHE_TTL_SECONDS = 5 * 60
REQUEST_INTERVAL_SECONDS = 2
PARSE_INTERVAL_SECONDS = 30
FORBIDDEN_COOLDOWN_SECONDS = 24 * 60 * 60
RATE_LIMIT_COOLDOWN_SECONDS = 60 * 60
_API_THROTTLE_ERROR_CODES = frozenset({"ratelimited", "maxlag"})
_MISSING_PAGE_ERROR_CODES = frozenset({"missingtitle", "invalidtitle", "missingpage", "nosuchpage"})
_STATE_VERSION = 1


class LiquipediaRequestError(Exception):
    """A failed, deferred, or untrusted Liquipedia request."""

    def __init__(
        self,
        message: str,
        *,
        retry_at: float | None = None,
        kind: str = "transport",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_at = retry_at
        self.kind = kind
        self.status_code = status_code


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = path.open("a+b")
    except OSError as exc:
        raise LiquipediaRequestError(f"Could not open Liquipedia host lock: {exc}", kind="state") from exc
    locked = False
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            locked = True
        except OSError as exc:
            raise LiquipediaRequestError(f"Could not acquire Liquipedia host lock: {exc}", kind="state") from exc
        yield
    finally:
        try:
            if locked:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise LiquipediaRequestError(f"Could not release Liquipedia host lock: {exc}", kind="state") from exc
        finally:
            lock_file.close()


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        _atomic_json_write(path, state)
    except Exception as exc:
        raise LiquipediaRequestError(f"Could not persist Liquipedia host budget state: {exc}", kind="state") from exc


def _load_json(path: Path, *, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:
        raise LiquipediaRequestError(
            f"Liquipedia {label} is corrupt or unreadable: {path}: {exc}", kind="state"
        ) from exc


def _cache_directory() -> Path:
    configured = os.environ.get("LIQUIPEDIA_CACHE_DIR")
    return Path(configured) if configured else DEFAULT_CACHE_DIR


def _host_identity(api_url: str) -> str:
    parsed = urllib.parse.urlsplit(api_url)
    if not parsed.hostname:
        raise LiquipediaRequestError(f"Invalid Liquipedia API URL: {api_url!r}", kind="request")
    return parsed.hostname.lower()


def _canonical_query_key(api_url: str, params: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            {"api_url": api_url, "params": params},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise LiquipediaRequestError(f"Liquipedia query parameters are not serializable: {exc}", kind="request") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _valid_timestamp(value: Any) -> bool:
    if value is None:
        return True
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "version": _STATE_VERSION,
            "last_request_at": None,
            "last_parse_at": None,
            "cooldown_until": None,
            "cooldown_message": None,
        }
    state = _load_json(path, label="host budget state")
    required = {"version", "last_request_at", "last_parse_at", "cooldown_until", "cooldown_message"}
    if (
        not isinstance(state, dict)
        or not required.issubset(state)
        or type(state.get("version")) is not int
        or state.get("version") != _STATE_VERSION
        or any(
            not _valid_timestamp(state.get(field))
            for field in ("last_request_at", "last_parse_at", "cooldown_until")
        )
        or (state.get("cooldown_message") is not None and not isinstance(state.get("cooldown_message"), str))
    ):
        raise LiquipediaRequestError(f"Liquipedia host budget state has an invalid schema: {path}", kind="state")
    return state


def _read_cache(path: Path, now: float) -> tuple[bool, Any]:
    if not path.exists():
        return False, None
    entry = _load_json(path, label="response cache")
    if (
        not isinstance(entry, dict)
        or type(entry.get("version")) is not int
        or entry.get("version") != _STATE_VERSION
        or not _valid_timestamp(entry.get("expires_at"))
        or entry.get("expires_at") is None
        or not isinstance(entry.get("success"), bool)
    ):
        raise LiquipediaRequestError(f"Liquipedia response cache has an invalid schema: {path}", kind="state")
    if entry["success"]:
        if not isinstance(entry.get("data"), (dict, list)):
            raise LiquipediaRequestError(f"Liquipedia response cache has invalid data: {path}", kind="state")
        if now >= entry["expires_at"]:
            return False, None
        return True, entry["data"]
    error = entry.get("error")
    if (
        not isinstance(error, dict)
        or not isinstance(error.get("message"), str)
        or not isinstance(error.get("kind", "transport"), str)
        or (error.get("status_code") is not None and type(error.get("status_code")) is not int)
        or not _valid_timestamp(error.get("retry_at"))
    ):
        raise LiquipediaRequestError(f"Liquipedia negative cache has invalid data: {path}", kind="state")
    if now >= entry["expires_at"]:
        return False, None
    raise LiquipediaRequestError(
        error["message"],
        retry_at=error.get("retry_at"),
        kind=error.get("kind", "transport"),
        status_code=error.get("status_code"),
    )



def _cache_error(path: Path, error: LiquipediaRequestError, *, now: float) -> None:
    retry_at = error.retry_at
    try:
        _atomic_json_write(
            path,
            {
                "version": _STATE_VERSION,
                "expires_at": now + NEGATIVE_CACHE_TTL_SECONDS,
                "success": False,
                "error": {
                    "message": str(error),
                    "retry_at": retry_at,
                    "kind": error.kind,
                    "status_code": error.status_code,
                },
            },
        )
    except Exception as exc:
        raise LiquipediaRequestError(f"Could not persist Liquipedia error cache: {exc}", kind="state") from exc


def _retry_after_seconds(value: str | None, now: float) -> float:
    if value:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except (ValueError, OverflowError):
            pass
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            delay = retry_at.timestamp() - now
            if math.isfinite(delay):
                return max(0.0, delay)
        except (TypeError, ValueError, OverflowError):
            pass
    return 0.0


def _response_url(api_url: str, params: dict[str, Any]) -> str:
    separator = "&" if "?" in api_url else "?"
    return f"{api_url}{separator}{urllib.parse.urlencode(params, doseq=True)}"


def query(
    api_url: str,
    params: dict[str, Any],
    user_agent: str,
    timeout: float = 15,
    cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    *,
    wait_for_spacing: bool = False,
) -> dict[str, Any] | list[Any]:
    """Query Liquipedia through a shared file-backed cache and per-host budget.

    Requests are serialized under one OS-level lock per host. In fail-fast mode,
    a request that would violate spacing raises ``LiquipediaRequestError`` with
    ``retry_at`` rather than sleeping. Explicit collectors may opt into waiting
    for spacing of at most 30 seconds; errors and cooldowns are never waited out.
    """
    host = _host_identity(api_url)
    directory = _cache_directory()
    host_key = hashlib.sha256(host.encode("utf-8")).hexdigest()
    cache_key = _canonical_query_key(api_url, params)
    state_path = directory / f"host-{host_key}.json"
    lock_path = directory / f"host-{host_key}.lock"
    cache_path = directory / f"query-{cache_key}.json"

    with _exclusive_lock(lock_path):
        now = time.time()
        state = _read_state(state_path)
        cache_hit, cached_data = _read_cache(cache_path, now)
        if cache_hit:
            return cached_data

        cooldown_until = state.get("cooldown_until")
        if cooldown_until is not None and now < cooldown_until:
            raise LiquipediaRequestError(
                state.get("cooldown_message") or f"Liquipedia host is cooling down until {cooldown_until:.3f}",
                retry_at=float(cooldown_until),
                kind="throttled",
            )

        is_parse_request = params.get("action") in {"parse", "expandtemplates"}
        while True:
            now = time.time()
            waits = []
            last_request = state.get("last_request_at")
            if last_request is not None:
                waits.append(float(last_request) + REQUEST_INTERVAL_SECONDS - now)
            last_parse = state.get("last_parse_at")
            if is_parse_request and last_parse is not None:
                waits.append(float(last_parse) + PARSE_INTERVAL_SECONDS - now)
            wait_seconds = max(waits, default=0.0)
            if wait_seconds <= 0:
                break
            retry_at = now + wait_seconds
            if not wait_for_spacing or wait_seconds > PARSE_INTERVAL_SECONDS:
                raise LiquipediaRequestError(
                    f"Liquipedia request deferred by host spacing until {retry_at:.3f}",
                    retry_at=retry_at,
                    kind="deferred",
                )
            time.sleep(wait_seconds)

        # Persist the reservation before network I/O so a process crash cannot
        # make a second process believe the host was never just contacted.
        now = time.time()
        state["last_request_at"] = now
        if is_parse_request:
            state["last_parse_at"] = now
        try:
            _atomic_json_write(state_path, state)
        except Exception as exc:
            raise LiquipediaRequestError(f"Could not persist Liquipedia host budget state: {exc}", kind="state") from exc

        request = urllib.request.Request(
            _response_url(api_url, params),
            headers={
                "User-Agent": user_agent,
                "Api-User-Agent": user_agent,
                "Accept-Encoding": "gzip",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                content_encoding = response.info().get("Content-Encoding", "")
                if content_encoding.lower() == "gzip":
                    import gzip

                    raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, (dict, list)):
                raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")
            if isinstance(data, dict) and "error" in data:
                api_error = data["error"]
                if isinstance(api_error, dict):
                    description = api_error.get("info") or api_error.get("code") or api_error
                    code = str(api_error.get("code") or "").lower()
                else:
                    description = api_error
                    code = ""
                now = time.time()
                message = f"Liquipedia API error{f' ({code})' if code else ''}: {description}"
                if code in _API_THROTTLE_ERROR_CODES:
                    retry_at = now + NEGATIVE_CACHE_TTL_SECONDS
                    state["cooldown_until"] = retry_at
                    state["cooldown_message"] = message
                    state["version"] = _STATE_VERSION
                    _save_state(state_path, state)
                    error_kind = "throttled"
                elif code in _MISSING_PAGE_ERROR_CODES:
                    retry_at = now + NEGATIVE_CACHE_TTL_SECONDS
                    error_kind = "missing_page"
                else:
                    retry_at = now + NEGATIVE_CACHE_TTL_SECONDS
                    error_kind = "api"
                error = LiquipediaRequestError(message, retry_at=retry_at, kind=error_kind)
                _cache_error(cache_path, error, now=now)
                raise error
        except urllib.error.HTTPError as exc:
            now = time.time()
            if exc.code == 403:
                retry_at = now + FORBIDDEN_COOLDOWN_SECONDS
                message = f"Liquipedia API HTTP 403: {exc.reason}; host cooldown is 24 hours"
                state["cooldown_until"] = retry_at
                state["cooldown_message"] = message
                state["version"] = _STATE_VERSION
                _save_state(state_path, state)
                error = LiquipediaRequestError(message, retry_at=retry_at, kind="throttled", status_code=403)
            elif exc.code == 429:
                retry_after = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None, now)
                retry_at = now + max(RATE_LIMIT_COOLDOWN_SECONDS, retry_after)
                message = f"Liquipedia API HTTP 429: {exc.reason}; retry after {retry_at:.3f}"
                state["cooldown_until"] = retry_at
                state["cooldown_message"] = message
                state["version"] = _STATE_VERSION
                _save_state(state_path, state)
                error = LiquipediaRequestError(message, retry_at=retry_at, kind="throttled", status_code=429)
            else:
                error = LiquipediaRequestError(
                    f"Liquipedia API HTTP {exc.code}: {exc.reason}",
                    retry_at=now + NEGATIVE_CACHE_TTL_SECONDS,
                    kind="transport",
                    status_code=exc.code,
                )
            _cache_error(cache_path, error, now=now)
            raise error from exc
        except LiquipediaRequestError:
            raise
        except Exception as exc:
            now = time.time()
            error = LiquipediaRequestError(
                f"Liquipedia request failed: {exc}",
                retry_at=now + NEGATIVE_CACHE_TTL_SECONDS,
                kind="transport",
            )
            _cache_error(cache_path, error, now=now)
            raise error from exc

        try:
            _atomic_json_write(
                cache_path,
                {
                    "version": _STATE_VERSION,
                    "expires_at": time.time() + cache_ttl_seconds,
                    "success": True,
                    "data": data,
                },
            )
        except Exception as exc:
            raise LiquipediaRequestError(f"Could not persist Liquipedia response cache: {exc}", kind="state") from exc
        return data
