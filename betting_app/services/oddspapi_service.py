"""OddsPapi service with budget enforcement, fixture synchronization, and horizon market comparisons."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from betting_app.ml.calibration.conformal_contract import conformal_bounds_for_side


from betting_app.core.db import get_session
from betting_app.core.matching import normalize_team_name
from betting_app.models.match import CanonicalMatch
from betting_app.models.odds import OddsSnapshot
from betting_app.models.oddspapi import OddspapiFixtureMapping, OddspapiRequestLog
from betting_app.models.prediction import CanonicalPrediction, SourceCapture
from betting_app.services.forecast_evidence import capture_source_attempt
from betting_app.services.odds_service import get_or_create_bookmaker
from betting_app.services.canonical_match_service import align_snapshot_odds
from src.analysis.quote_provenance import extract_winner_quote

logger = logging.getLogger(__name__)

ODDSPAPI_API_BASE = "https://api.oddspapi.io/v4"
DEFAULT_MONTHLY_LIMIT = 250
DEFAULT_DAILY_LIMIT = 8
LOL_SPORT_ID = 18

PRIORITY_LEAGUES: set[str] = {
    "lck",
    "lpl",
    "lec",
    "lcs",
    "worlds",
    "msi",
    "lfl",
    "lck challengers",
    "lck cl",
}


class OddsPapiError(RuntimeError):
    """Base exception for OddsPapi service failures."""


class OddsPapiBudgetExhaustedError(OddsPapiError):
    """Raised when monthly or daily request allowance has been exhausted."""


class OddsPapiHTTPError(OddsPapiError):
    """Raised on remote HTTP errors with status and response diagnostics."""

    def __init__(self, path: str, status: int, body: str) -> None:
        self.path = path
        self.status = status
        self.body = body
        self.retry_after_seconds: float | None = None
        try:
            retry_after = json.loads(body).get("error", {}).get("retryAfter")
            if retry_after:
                self.retry_after_seconds = float(str(retry_after).split()[0])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
        super().__init__(f"OddsPapi {path} returned HTTP {status}: {body[:300]}")


def parse_utc(value: str | datetime) -> datetime:
    """Parse string or datetime to timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


class OddsPapiBudgetGuard:
    """Enforces strict daily and monthly quota boundaries backed by oddspapi_request_logs."""

    def __init__(
        self,
        monthly_limit: int = DEFAULT_MONTHLY_LIMIT,
        daily_limit: int = DEFAULT_DAILY_LIMIT,
        monthly_window_days: int = 30,
        daily_window_hours: int = 24,
    ) -> None:
        self.monthly_limit = int(os.getenv("ODDSPAPI_MONTHLY_LIMIT", monthly_limit))
        self.daily_limit = int(os.getenv("ODDSPAPI_DAILY_LIMIT", daily_limit))
        self.monthly_window_days = monthly_window_days
        self.daily_window_hours = daily_window_hours

    def check_budget(self, session: Session) -> tuple[bool, str, dict[str, int]]:
        """Verify request capacity against persisted audit logs."""
        now = datetime.now(UTC)
        monthly_since = now - timedelta(days=self.monthly_window_days)
        daily_since = now - timedelta(hours=self.daily_window_hours)

        monthly_used = (
            session.scalar(
                select(func.count(OddspapiRequestLog.id)).where(
                    OddspapiRequestLog.created_at >= monthly_since
                )
            )
            or 0
        )

        daily_used = (
            session.scalar(
                select(func.count(OddspapiRequestLog.id)).where(
                    OddspapiRequestLog.created_at >= daily_since
                )
            )
            or 0
        )

        stats = {
            "monthly_used": int(monthly_used),
            "monthly_limit": self.monthly_limit,
            "monthly_remaining": max(0, self.monthly_limit - int(monthly_used)),
            "daily_used": int(daily_used),
            "daily_limit": self.daily_limit,
            "daily_remaining": max(0, self.daily_limit - int(daily_used)),
        }

        if monthly_used >= self.monthly_limit:
            return (
                False,
                f"OddsPapi monthly quota reached ({monthly_used}/{self.monthly_limit})",
                stats,
            )

        if daily_used >= self.daily_limit:
            return (
                False,
                f"OddsPapi daily cap reached ({daily_used}/{self.daily_limit})",
                stats,
            )

        return True, "OK", stats

    def record_request(
        self,
        session: Session,
        endpoint: str,
        fixture_id: str | None,
        status_code: int,
        response_time_ms: int | None = None,
        *,
        commit: bool = True,
    ) -> OddspapiRequestLog:
        """Record an API request, optionally within a caller-owned transaction."""
        log_entry = OddspapiRequestLog(
            endpoint=endpoint,
            fixture_id=fixture_id,
            status_code=status_code,
            response_time_ms=response_time_ms,
            created_at=datetime.now(UTC),
        )
        session.add(log_entry)
        if commit:
            session.commit()
        return log_entry


class OddsPapiClient:
    """HTTP client for OddsPapi with budget tracking and diagnostic error reporting."""

    def __init__(
        self,
        api_key: str | None = None,
        budget_guard: OddsPapiBudgetGuard | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("ODDSPAPI_API_KEY", "")
        self.budget_guard = budget_guard or OddsPapiBudgetGuard()
        self.timeout_seconds = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        session: Session | None = None,
        fixture_id: str | None = None,
    ) -> Any:
        """Perform a metered GET request against OddsPapi."""
        if not self.is_configured():
            raise OddsPapiError("OddsPapi API key is not configured (set ODDSPAPI_API_KEY).")

        sess = session or get_session()
        own_session = session is None
        try:
            allowed, reason, stats = self.budget_guard.check_budget(sess)
            if not allowed:
                raise OddsPapiBudgetExhaustedError(reason)

            query_params = dict(params or {})
            query_params["apiKey"] = self.api_key

            url = f"{ODDSPAPI_API_BASE}{path}?{urlencode(query_params)}"
            request = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "LoL-Breaker OddsPapi Client/1.0",
                },
            )

            start_t = time.perf_counter()
            status_code = 0
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    status_code = response.status
                    body = response.read().decode("utf-8")
                    duration_ms = int((time.perf_counter() - start_t) * 1000)
                    self.budget_guard.record_request(
                        sess, path, fixture_id, status_code, duration_ms
                    )
                    return json.loads(body)
            except HTTPError as error:
                status_code = error.code
                duration_ms = int((time.perf_counter() - start_t) * 1000)
                body = error.read().decode("utf-8", errors="replace")
                self.budget_guard.record_request(
                    sess, path, fixture_id, status_code, duration_ms
                )
                raise OddsPapiHTTPError(path, status_code, body) from error
            except URLError as error:
                raise OddsPapiError(f"OddsPapi connection error: {error.reason}") from error
        finally:
            if own_session:
                sess.close()
    def get_captured(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        session: Session,
        capture_id: str,
        source_id: str,
        evidence_uri: str,
        body_store: Path,
        source_class: str = "unverified",
        availability_basis: str = "client_response_received",
        source_revision: str | None = None,
        published_at: datetime | None = None,
        source_effective_at: datetime | None = None,
        fixture_id: str | None = None,
    ) -> tuple[Any | None, SourceCapture]:
        """Capture exact response bytes before JSON parsing in a caller-owned transaction."""
        if not self.is_configured():
            raise OddsPapiError("OddsPapi API key is not configured (set ODDSPAPI_API_KEY).")
        allowed, reason, _stats = self.budget_guard.check_budget(session)
        if not allowed:
            raise OddsPapiBudgetExhaustedError(reason)

        query_params = dict(params or {})
        query_params["apiKey"] = self.api_key
        url = f"{ODDSPAPI_API_BASE}{path}?{urlencode(query_params)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "LoL-Breaker OddsPapi Client/1.0",
            },
        )
        request_started_at = datetime.now(UTC)
        start_t = time.perf_counter()

        def retain_attempt(
            raw_body: bytes | None,
            *,
            status_code: int | None,
            headers: Any,
            response_received_at: datetime | None,
            failure_reason: str | None = None,
        ) -> SourceCapture:
            header_values = {
                str(key).casefold(): str(value)
                for key, value in (headers.items() if headers is not None else [])
            }
            return capture_source_attempt(
                session,
                capture_id=capture_id,
                source_id=source_id,
                evidence_uri=evidence_uri,
                raw_body=raw_body,
                body_store=body_store,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                first_observed_at=response_received_at,
                observed_at=datetime.now(UTC),
                published_at=published_at,
                source_effective_at=source_effective_at,
                source_revision=source_revision,
                http_status=status_code,
                source_class=source_class,
                availability_basis=availability_basis,
                body_representation="http_body",
                content_encoding=header_values.get("content-encoding"),
                character_encoding="utf-8",
                request_metadata={
                    "method": "GET",
                    "url": url,
                    "query": query_params,
                    "headers": {"accept": "application/json"},
                },
                response_metadata={
                    "status": status_code,
                    "content_type": header_values.get("content-type"),
                    "content_encoding": header_values.get("content-encoding"),
                    "response_id": header_values.get("x-request-id")
                    or header_values.get("request-id"),
                },
                failure_reason=failure_reason,
            )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                status_code = response.status
                raw_body = response.read()
                response_received_at = datetime.now(UTC)
                duration_ms = int((time.perf_counter() - start_t) * 1000)
                self.budget_guard.record_request(
                    session,
                    path,
                    fixture_id,
                    status_code,
                    duration_ms,
                    commit=False,
                )
                capture = retain_attempt(
                    raw_body,
                    status_code=status_code,
                    headers=response.headers,
                    response_received_at=response_received_at,
                )
        except HTTPError as error:
            raw_body = error.read()
            response_received_at = datetime.now(UTC)
            duration_ms = int((time.perf_counter() - start_t) * 1000)
            self.budget_guard.record_request(
                session,
                path,
                fixture_id,
                error.code,
                duration_ms,
                commit=False,
            )
            capture = retain_attempt(
                raw_body,
                status_code=error.code,
                headers=error.headers,
                response_received_at=response_received_at,
            )
            return None, capture
        except URLError:
            duration_ms = int((time.perf_counter() - start_t) * 1000)
            self.budget_guard.record_request(
                session,
                path,
                fixture_id,
                0,
                duration_ms,
                commit=False,
            )
            capture = retain_attempt(
                None,
                status_code=None,
                headers=None,
                response_received_at=None,
                failure_reason="transport_error",
            )
            return None, capture

        if capture.attempt_status != "captured" or raw_body is None:
            return None, capture
        try:
            return json.loads(raw_body.decode("utf-8")), capture
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, capture




def sync_oddspapi_fixtures(
    session: Session | None = None,
    days_ahead: int = 7,
    client: OddsPapiClient | None = None,
) -> dict[str, Any]:
    """Fetch upcoming LoL fixtures and match them to canonical_matches.

    Costs exactly 1 request to /v4/fixtures.
    """
    api_client = client or OddsPapiClient()
    if not api_client.is_configured():
        return {"status": "skipped", "reason": "ODDSPAPI_API_KEY not configured"}

    sess = session or get_session()
    own_session = session is None
    try:
        now = datetime.now(UTC)
        to_date = now + timedelta(days=days_ahead)

        payload = api_client.get(
            "/fixtures",
            {
                "sportId": LOL_SPORT_ID,
                "from": now.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
            },
            session=sess,
        )

        if not isinstance(payload, list):
            return {"status": "error", "error": f"Unexpected payload type {type(payload).__name__}"}

        # Load upcoming canonical matches in the corresponding window
        canonical_matches = sess.scalars(
            select(CanonicalMatch).where(
                CanonicalMatch.status.in_(("upcoming", "active", "pending"))
            )
        ).all()

        canonical_by_pair: dict[frozenset[str], list[CanonicalMatch]] = {}
        for cm in canonical_matches:
            if not cm.team_a_name or not cm.team_b_name:
                continue
            norm_a = normalize_team_name(cm.team_a_name)
            norm_b = normalize_team_name(cm.team_b_name)
            pair = frozenset((norm_a, norm_b))
            canonical_by_pair.setdefault(pair, []).append(cm)

        synced_count = 0
        mapped_count = 0

        for fixture in payload:
            fixture_id = str(fixture.get("fixtureId") or "")
            if not fixture_id:
                continue

            raw_team_1 = str(fixture.get("participant1Name") or "")
            raw_team_2 = str(fixture.get("participant2Name") or "")
            if not raw_team_1 or not raw_team_2:
                continue

            norm_1 = normalize_team_name(raw_team_1)
            norm_2 = normalize_team_name(raw_team_2)
            pair = frozenset((norm_1, norm_2))

            start_time_str = fixture.get("startTime")
            fixture_start = parse_utc(start_time_str) if start_time_str else None

            # Attempt to resolve canonical match
            canonical_match_id: int | None = None
            team_1_is_a: int | None = None

            candidates = canonical_by_pair.get(pair, [])
            if candidates and fixture_start:
                valid_candidates = []
                for candidate in candidates:
                    if not candidate.start_time_normalized:
                        continue
                    c_start = parse_utc(candidate.start_time_normalized)
                    if abs((c_start - fixture_start).total_seconds()) <= 43200:  # 12h
                        valid_candidates.append((abs((c_start - fixture_start).total_seconds()), candidate))

                if len(valid_candidates) == 1:
                    matched_cm = valid_candidates[0][1]
                    canonical_match_id = matched_cm.id
                    norm_a = normalize_team_name(matched_cm.team_a_name)
                    team_1_is_a = 1 if norm_1 == norm_a else 0
                    mapped_count += 1

            existing = sess.scalar(
                select(OddspapiFixtureMapping).where(
                    OddspapiFixtureMapping.fixture_id == fixture_id
                )
            )

            if existing:
                if canonical_match_id is not None:
                    existing.canonical_match_id = canonical_match_id
                    existing.provider_team_1_is_a = team_1_is_a
                existing.has_odds = 1 if fixture.get("hasOdds") else 0
                existing.last_synced_at = now
            else:
                mapping = OddspapiFixtureMapping(
                    fixture_id=fixture_id,
                    canonical_match_id=canonical_match_id,
                    sport_id=LOL_SPORT_ID,
                    league=str(fixture.get("tournamentName") or fixture.get("leagueName") or ""),
                    provider_team_1=raw_team_1,
                    provider_team_2=raw_team_2,
                    provider_team_1_is_a=team_1_is_a,
                    start_time=fixture_start,
                    has_odds=1 if fixture.get("hasOdds") else 0,
                    last_synced_at=now,
                )
                sess.add(mapping)

            synced_count += 1

        sess.commit()
        return {
            "status": "success",
            "fixtures_synced": synced_count,
            "mapped_to_canonical": mapped_count,
        }
    except OddsPapiBudgetExhaustedError as err:
        logger.warning(f"OddsPapi fixture sync blocked by budget: {err}")
        return {"status": "budget_exhausted", "error": str(err)}
    finally:
        if own_session:
            sess.close()


def _observed_live_state(payload: dict[str, Any], extracted: dict[str, Any]) -> int | None:
    """Read explicit source state; a pre-start request alone is not prematch proof."""
    sources = [payload, payload.get("fixture"), extracted.get("source_market_evidence")]
    books = payload.get("bookmakerOdds") or payload.get("bookmakers") or {}
    if isinstance(books, dict):
        sources.append(books.get("pinnacle"))
    states: list[int] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("isLive", "islive", "is_live", "live", "inPlay", "isInPlay", "inplay"):
            if key in source:
                value = source[key]
                if isinstance(value, bool) or type(value) is int and value in (0, 1):
                    states.append(int(value))
                else:
                    return None
    if extracted.get("explicit_scope_status") == "rejected_explicit_live_scope":
        return 1
    return max(states) if states else None


def build_oddspapi_observed_snapshot(
    payload: dict[str, Any], extracted: dict[str, Any], *,
    mapping: OddspapiFixtureMapping, canonical_match_id: int,
    bookmaker_id: int, observed_at: datetime, participant_bindings: list[dict[str, Any]],
) -> OddsSnapshot:
    """Retain an identifiable observed state, including unavailable prices."""
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("observed_at must be timezone-aware")
    observed_at = observed_at.astimezone(UTC)
    admitted = extracted.get("quote_admission_status") == "admitted_with_bookmaker_evidence"
    active_pair = admitted and all(
        (extracted.get(key) or {}).get("status") == "active"
        for key in ("state_1", "state_2")
    )
    raw_a = extracted.get("participant_1_name") if admitted else mapping.provider_team_1
    raw_b = extracted.get("participant_2_name") if admitted else mapping.provider_team_2
    return OddsSnapshot(
        bookmaker_id=bookmaker_id, canonical_match_id=canonical_match_id,
        market_type="match_winner", raw_team_a=raw_a, raw_team_b=raw_b,
        odds_a=extracted.get("odds_1") if active_pair else None,
        odds_b=extracted.get("odds_2") if active_pair else None,
        is_live=_observed_live_state(payload, extracted), scraped_at=observed_at,
        source_url=f"oddspapi://v4/odds/{mapping.fixture_id}",
        raw_payload=json.dumps({
            "provider_payload": payload, "quote_provenance": extracted,
            "response_observed_at": observed_at.isoformat(),
            "participant_bindings": participant_bindings,
        }, ensure_ascii=False, default=str),
    )


def fetch_pinnacle_horizon_odds(
    session: Session | None = None,
    target_horizon_hours: float = 6.0,
    tolerance_hours: float = 1.0,
    max_requests: int = 4,
    client: OddsPapiClient | None = None,
    *,
    participant_bindings: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Fetch Pinnacle odds for matches in the target horizon window (e.g. T−6h).

    Only queries matches with an existing OddspapiFixtureMapping that lack a recent observation.
    ``saved`` counts retained observations, split into ``admitted`` (binding evidence
    accepted) and ``quarantined`` (prices withheld). Admission does not authenticate
    the evidence source or certify execution; inactive admitted states also have no prices.
    """
    binding_rows = list(participant_bindings)
    def evidence_scope_id(value: Any) -> str | None:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None
        text = str(value).strip()
        if not text or text.casefold() in {"nan", "none", "null", "<na>", "nat"}:
            return None
        return text
    api_client = client or OddsPapiClient()
    if not api_client.is_configured():
        return {"status": "skipped", "reason": "ODDSPAPI_API_KEY not configured"}

    sess = session or get_session()
    own_session = session is None
    try:
        now = datetime.now(UTC)
        window_start = now + timedelta(hours=target_horizon_hours - tolerance_hours)
        window_end = now + timedelta(hours=target_horizon_hours + tolerance_hours)

        pinnacle_id = get_or_create_bookmaker("pinnacle")

        # Find upcoming matches mapped to OddsPapi
        mappings = sess.scalars(
            select(OddspapiFixtureMapping).where(
                OddspapiFixtureMapping.canonical_match_id.is_not(None),
                OddspapiFixtureMapping.start_time >= window_start,
                OddspapiFixtureMapping.start_time <= window_end,
                OddspapiFixtureMapping.has_odds == 1,
            )
        ).all()

        if not mappings:
            return {
                "status": "success", "fetched": 0, "saved": 0, "candidates": 0,
                "admitted": 0, "quarantined": 0,
            }

        # Filter out matches with any Pinnacle observation in the last 2 hours.
        eligible: list[OddspapiFixtureMapping] = []
        for mapping in mappings:
            cm_id = mapping.canonical_match_id
            assert cm_id is not None
            recent_quote = sess.scalar(
                select(OddsSnapshot.id).where(
                    OddsSnapshot.canonical_match_id == cm_id,
                    OddsSnapshot.bookmaker_id == pinnacle_id,
                    OddsSnapshot.scraped_at >= (now - timedelta(hours=2)).isoformat(),
                )
            )
            if not recent_quote:
                eligible.append(mapping)

        # Sort eligible matches: priority leagues first, then earliest start
        def league_priority(m: OddspapiFixtureMapping) -> tuple[int, datetime]:
            league_str = (m.league or "").lower()
            is_priority = 0 if any(p in league_str for p in PRIORITY_LEAGUES) else 1
            start = m.start_time or now
            return is_priority, start

        eligible.sort(key=league_priority)
        to_fetch = eligible[:max_requests]

        fetched_count = 0
        saved_count = 0
        admitted_count = 0
        quarantined_count = 0

        for mapping in to_fetch:
            cm_id = mapping.canonical_match_id
            assert cm_id is not None

            # Verify budget before each individual call
            allowed, reason, _ = api_client.budget_guard.check_budget(sess)
            if not allowed:
                logger.warning(f"Stopping horizon odds fetch: {reason}")
                break

            try:
                payload = api_client.get(
                    "/odds",
                    {"fixtureId": mapping.fixture_id},
                    session=sess,
                    fixture_id=mapping.fixture_id,
                )
            except OddsPapiHTTPError as err:
                logger.error(f"Failed to fetch odds for fixture {mapping.fixture_id}: {err}")
                continue
            except OddsPapiBudgetExhaustedError:
                break
            response_observed_at = datetime.now(UTC)

            fetched_count += 1
            cm = sess.get(CanonicalMatch, cm_id)
            if not cm:
                continue

            fixture_context: dict[str, Any] = {
                "fixtureId": mapping.fixture_id,
                "participant1Name": mapping.provider_team_1,
                "participant2Name": mapping.provider_team_2,
            }
            if (
                mapping.start_time is not None
                and mapping.start_time.tzinfo is not None
                and mapping.start_time.utcoffset() is not None
            ):
                fixture_context["start_at"] = mapping.start_time
            fixture_binding_rows = [
                row
                for row in binding_rows
                if isinstance(row, dict)
                and evidence_scope_id(row.get("fixture_id")) == mapping.fixture_id
                and evidence_scope_id(row.get("bookmaker")) == "pinnacle"
                and evidence_scope_id(row.get("market_id")) == "181"
            ]
            extracted = extract_winner_quote(
                payload,
                bookmaker_slug="pinnacle",
                cutoff_at=response_observed_at,
                fixture=fixture_context,
                participant_bindings=fixture_binding_rows,
            )
            # Absence is not a withdrawal; an observed identifiable market is.
            if extracted is None:
                continue
            if extracted.get("source_fixture_id_matches_context") is False:
                continue
            snapshot = build_oddspapi_observed_snapshot(
                payload, extracted, mapping=mapping, canonical_match_id=cm.id,
                bookmaker_id=pinnacle_id, observed_at=response_observed_at,
                participant_bindings=fixture_binding_rows,
            )
            existing = sess.scalars(select(OddsSnapshot).where(
                OddsSnapshot.canonical_match_id == cm.id,
                OddsSnapshot.bookmaker_id == pinnacle_id,
                OddsSnapshot.market_type == "match_winner",
                OddsSnapshot.scraped_at == response_observed_at,
            )).first()
            if existing is None:
                sess.add(snapshot)
            else:
                for key in ("raw_team_a", "raw_team_b", "odds_a", "odds_b", "is_live", "raw_payload", "source_url"):
                    setattr(existing, key, getattr(snapshot, key))
            saved_count += 1
            if extracted["quote_admission_status"] == "admitted_with_bookmaker_evidence":
                admitted_count += 1
            else:
                quarantined_count += 1

        sess.commit()
        return {
            "status": "success",
            "candidates": len(eligible),
            "fetched": fetched_count,
            "saved": saved_count,
            "admitted": admitted_count,
            "quarantined": quarantined_count,
        }
    finally:
        if own_session:
            sess.close()


def compare_match_market(
    canonical_match_id: int,
    session: Session | None = None,
    horizon_hours: float = 6.0,
    tolerance_hours: float = 1.5,
) -> dict[str, Any] | None:
    """Retrospective strictly aligned quote comparison, not a current recommendation."""
    sess = session or get_session()
    own_session = session is None
    try:
        cm = sess.get(CanonicalMatch, canonical_match_id)
        if not cm or not cm.start_time_normalized:
            return None

        match_start = parse_utc(cm.start_time_normalized)
        target_time = match_start - timedelta(hours=horizon_hours)

        # Load snapshots for this match
        from betting_app.models.bookmaker import Bookmaker

        snapshots = (
            sess.execute(
                select(OddsSnapshot, Bookmaker.name)
                .join(Bookmaker, Bookmaker.id == OddsSnapshot.bookmaker_id)
                .where(
                    OddsSnapshot.canonical_match_id == canonical_match_id,
                    OddsSnapshot.market_type == "match_winner",
                )
            )
            .all()
        )

        # Parse before selecting state: persisted offsets cannot be sorted lexically.
        latest_by_bookmaker: dict[str, Any] = {}
        clocks_by_bookmaker: dict[str, tuple[datetime, int]] = {}
        unknown_clock_books: set[str] = set()
        for snap, book_name in snapshots:
            try:
                clock = datetime.fromisoformat(snap.scraped_at.replace("Z", "+00:00")) if isinstance(snap.scraped_at, str) else snap.scraped_at
                if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() is None:
                    unknown_clock_books.add(book_name)
                    continue
                clock = clock.astimezone(UTC)
            except (ValueError, TypeError, OverflowError):
                unknown_clock_books.add(book_name)
                continue
            if clock > target_time:
                continue
            rank = (clock, snap.id)
            if book_name not in clocks_by_bookmaker or rank > clocks_by_bookmaker[book_name]:
                clocks_by_bookmaker[book_name] = rank
                latest_by_bookmaker[book_name] = snap
        for name in unknown_clock_books:
            latest_by_bookmaker.pop(name, None)

        def aligned_prices(snap: OddsSnapshot) -> tuple[float, float] | None:
            if snap.is_live != 0:
                return None
            return align_snapshot_odds(
                cm.team_a_name, cm.team_b_name, snap.raw_team_a, snap.raw_team_b,
                snap.odds_a, snap.odds_b, league=cm.league,
                source_system="bookmaker", match_date=cm.start_time_normalized,
            )

        pinnacle_snap = latest_by_bookmaker.get("pinnacle")
        pinnacle_prices = aligned_prices(pinnacle_snap) if pinnacle_snap else None
        pinnacle_prob_a: float | None = None
        pinnacle_margin: float | None = None
        if pinnacle_prices:
            inv_a, inv_b = (1.0 / price for price in pinnacle_prices)
            pinnacle_margin = inv_a + inv_b - 1.0
            pinnacle_prob_a = inv_a / (inv_a + inv_b)

        # Load latest model prediction
        pred = sess.scalars(
            select(CanonicalPrediction)
            .where(CanonicalPrediction.canonical_match_id == canonical_match_id)
            .order_by(CanonicalPrediction.predicted_at.desc())
        ).first()

        model_prob_a = float(pred.prob_a) if pred and pred.prob_a is not None else None
        conformal_a = (
            conformal_bounds_for_side(pred.diagnostics_json, "a")
            if pred is not None
            else None
        )
        conformal_b = (
            conformal_bounds_for_side(pred.diagnostics_json, "b")
            if pred is not None
            else None
        )

        bookmakers_comparison: list[dict[str, Any]] = []
        for name, snap in sorted(latest_by_bookmaker.items()):
            prices = aligned_prices(snap)
            odds_a, odds_b = prices if prices else (None, None)
            if prices:
                inv_a, inv_b = (1.0 / price for price in prices)
                margin = inv_a + inv_b - 1.0
                novig_a = inv_a / (inv_a + inv_b)
            else:
                margin = novig_a = None

            delta_pin = round(novig_a - pinnacle_prob_a, 4) if novig_a is not None and pinnacle_prob_a is not None else None

            # Calculate EV with 12% turnover tax for Polish bookmakers
            ev_a = None
            ev_b = None
            ev_conf_a = None
            ev_conf_b = None
            is_conf_a = False
            is_conf_b = False

            if model_prob_a is not None and prices is not None:
                ev_a = round(model_prob_a * (odds_a * 0.88) - 1.0, 4)
                ev_b = round((1.0 - model_prob_a) * (odds_b * 0.88) - 1.0, 4)

                if (
                    conformal_a is not None
                    and conformal_b is not None
                    and conformal_a[1] - conformal_a[0] <= 0.08
                    and conformal_b[1] - conformal_b[0] <= 0.08
                ):
                    ev_conf_a = round(
                        conformal_a[0] * (odds_a * 0.88) - 1.0, 4
                    )
                    ev_conf_b = round(
                        conformal_b[0] * (odds_b * 0.88) - 1.0, 4
                    )
                    is_conf_a = ev_conf_a > 0.0
                    is_conf_b = ev_conf_b > 0.0

            bookmakers_comparison.append(
                {
                    "bookmaker": name,
                    "odds_a": odds_a,
                    "odds_b": odds_b,
                    "novig_prob_a": round(novig_a, 4) if novig_a is not None else None,
                    "margin": round(margin, 4) if margin is not None else None,
                    "scraped_at": snap.scraped_at,
                    "delta_to_pinnacle": delta_pin,
                    "ev_team_a_tax12": ev_a,
                    "ev_team_b_tax12": ev_b,
                    "ev_conformal_low_a": ev_conf_a,
                    "ev_conformal_low_b": ev_conf_b,
                    "is_conformal_value_a": is_conf_a,
                    "is_conformal_value_b": is_conf_b,
                }
            )

        return {
            "canonical_match_id": canonical_match_id,
            "team_a": cm.team_a_name,
            "team_b": cm.team_b_name,
            "league": cm.league,
            "match_start_at": cm.start_time_normalized,
            "target_horizon_hours": horizon_hours,
            "model_prob_a": model_prob_a,
            "pinnacle_novig_prob_a": round(pinnacle_prob_a, 4) if pinnacle_prob_a is not None else None,
            "pinnacle_margin": round(pinnacle_margin, 4) if pinnacle_margin is not None else None,
            "bookmakers": bookmakers_comparison,
        }
    finally:
        if own_session:
            sess.close()
