"""Confirmed lineup ingestion service and substitution detection.

Checks whether confirmed 5-player rosters are available for upcoming matches
(typically released 30-45 minutes before match start) and detects player
substitutions against baseline/historical rosters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import logging
from typing import Any, Sequence

from betting_app.services.canonical_match_service import parse_iso
from betting_app.core.matching import normalize_team_name

logger = logging.getLogger(__name__)

ROLE_ORDER = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
DEFAULT_LINEUP_WINDOW_MINUTES = 45
DEFAULT_MAX_POST_START_MINUTES = 15


def normalize_player_id(val: Any) -> str:
    """Extract and normalize player identifier from string or dict."""
    if val is None:
        return ""
    if isinstance(val, dict):
        raw = val.get("player_id") or val.get("player_name") or val.get("name") or ""
        return str(raw).strip()
    return str(val).strip()


def detect_lineup_substitution(
    previous_roster: Sequence[Any],
    confirmed_roster: Sequence[Any],
) -> tuple[bool, list[str]]:
    """Detect changes and player substitutions between previous and confirmed rosters.

    Args:
        previous_roster: Starting roster (list of player IDs/names or dicts) known prior to confirmation.
        confirmed_roster: Confirmed starting roster (list of player IDs/names or dicts).

    Returns:
        tuple[bool, list[str]]:
            - bool: True if at least one substitution was detected, False otherwise.
            - list[str]: List of substituted player IDs (the incoming substitute players).
    """
    if not confirmed_roster:
        return False, []

    prev_ids = [normalize_player_id(p) for p in previous_roster if normalize_player_id(p)]
    conf_ids = [normalize_player_id(p) for p in confirmed_roster if normalize_player_id(p)]

    prev_set = set(prev_ids)
    # Incoming substitutes: players present in confirmed_roster that were NOT in previous_roster
    subs = [pid for pid in conf_ids if pid not in prev_set]

    has_substitution = len(subs) > 0
    return has_substitution, subs


def is_within_lineup_window(
    match_start_at: str | datetime | None,
    current_time: str | datetime | None = None,
    *,
    window_minutes: int = DEFAULT_LINEUP_WINDOW_MINUTES,
    max_post_start_minutes: int = DEFAULT_MAX_POST_START_MINUTES,
) -> tuple[bool, str]:
    """Check whether current_time is within the valid pre-match lineup window [start - 45m, start].

    Returns:
        tuple[bool, str]: (is_valid_window, reason_code)
    """
    if match_start_at is None:
        return False, "missing_start_time"

    if isinstance(match_start_at, str):
        start_dt = parse_iso(match_start_at)
    else:
        start_dt = match_start_at

    if start_dt is None:
        return False, "unparseable_start_time"

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=UTC)

    if current_time is None:
        now_dt = datetime.now(UTC)
    elif isinstance(current_time, str):
        now_dt = parse_iso(current_time) or datetime.now(UTC)
    else:
        now_dt = current_time

    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=UTC)

    diff_seconds = (start_dt - now_dt).total_seconds()
    diff_minutes = diff_seconds / 60.0

    # > window_minutes before start (e.g. > 45 minutes, or > 2 hours) -> too early
    if diff_minutes > window_minutes:
        return False, f"outside_window_too_early:{diff_minutes:.1f}m_until_start"

    # Match has started longer than allowed post-start grace -> past match
    if diff_minutes < -max_post_start_minutes:
        return False, f"outside_window_past_match:{-diff_minutes:.1f}m_since_start"

    return True, "valid_window"


def extract_roster_from_dict(
    data: dict[str, Any],
    keys: Sequence[str],
) -> list[Any] | None:
    """Extract a 5-player roster from candidate keys in a match dictionary."""
    for key in keys:
        val = data.get(key)
        if isinstance(val, list) and len(val) >= 5:
            return val
        if isinstance(val, dict) and "players" in val and isinstance(val["players"], list) and len(val["players"]) >= 5:
            return val["players"]
    return None


def check_confirmed_lineup(
    match_id: int | str | None,
    match_start_at: str | datetime | None = None,
    canonical_match: dict[str, Any] | None = None,
    *,
    current_time: str | datetime | None = None,
    window_minutes: int = DEFAULT_LINEUP_WINDOW_MINUTES,
    max_post_start_minutes: int = DEFAULT_MAX_POST_START_MINUTES,
    session: Any = None,
) -> dict[str, Any]:
    """Check whether confirmed 5-player rosters are available for a match.

    Checks:
    1. Valid pre-match timing window (e.g., [start - 45m, start]). Ignores if too early (> 2h).
    2. Availability of confirmed 5-player rosters for team A and/or team B.
    3. Detection of substitutions against baseline/previous rosters if present.

    Args:
        match_id: Canonical or external match ID.
        match_start_at: Scheduled match start time (ISO string or datetime).
        canonical_match: Optional dictionary containing match details, rosters, or overrides.
        current_time: Optional datetime or ISO string for simulation/testing.
        window_minutes: Minutes before match start when confirmed lineups open (default 45).
        max_post_start_minutes: Minutes after match start to accept confirmation (default 15).
        session: Optional SQLAlchemy DB session for roster table lookup.

    Returns:
        dict[str, Any]: Detailed status dictionary.
    """
    match_data = canonical_match or {}
    eff_start = match_start_at or match_data.get("start_time_normalized") or match_data.get("start_at") or match_data.get("start_time")
    eff_match_id = match_id if match_id is not None else match_data.get("id") or match_data.get("canonical_match_id")

    # 1. Verify timing window
    window_valid, window_reason = is_within_lineup_window(
        eff_start,
        current_time=current_time,
        window_minutes=window_minutes,
        max_post_start_minutes=max_post_start_minutes,
    )

    if not window_valid:
        status = "outside_window" if "too_early" in window_reason else "past_match" if "past_match" in window_reason else "invalid_time"
        return {
            "is_confirmed": False,
            "window_valid": False,
            "status": status,
            "match_id": eff_match_id,
            "reason": f"Timing check failed: {window_reason}",
            "team_a_roster": None,
            "team_b_roster": None,
            "substitutions_detected": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
        }

    # 2. Check if explicitly marked unconfirmed in match_data
    if match_data.get("is_lineup_confirmed") is False:
        return {
            "is_confirmed": False,
            "window_valid": True,
            "status": "unconfirmed",
            "match_id": eff_match_id,
            "reason": "Lineup explicitly flagged as unconfirmed",
            "team_a_roster": None,
            "team_b_roster": None,
            "substitutions_detected": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
        }

    # 3. Extract confirmed rosters from canonical_match
    team_a_roster = extract_roster_from_dict(
        match_data,
        ["confirmed_roster_a", "team_a_confirmed_lineup", "team_a_lineup", "lineup_a", "confirmed_lineup_a", "team_a_roster"],
    )
    team_b_roster = extract_roster_from_dict(
        match_data,
        ["confirmed_roster_b", "team_b_confirmed_lineup", "team_b_lineup", "lineup_b", "confirmed_lineup_b", "team_b_roster"],
    )

    # Check confirmed_rosters sub-dict
    if isinstance(match_data.get("confirmed_rosters"), dict):
        conf_dict = match_data["confirmed_rosters"]
        if not team_a_roster and "team_a" in conf_dict:
            team_a_roster = conf_dict["team_a"]
        if not team_b_roster and "team_b" in conf_dict:
            team_b_roster = conf_dict["team_b"]

    # Fallback: single confirmed_roster key if provided for single team context
    if not team_a_roster and not team_b_roster:
        single_roster = extract_roster_from_dict(match_data, ["confirmed_roster", "confirmed_lineup", "lineup", "players"])
        if single_roster:
            team_a_roster = single_roster

    # Check 5-player availability
    has_a = bool(team_a_roster and len(team_a_roster) >= 5)
    has_b = bool(team_b_roster and len(team_b_roster) >= 5)

    # If canonical match specifies both teams, both should be available; if single team context, 1 is sufficient
    needs_both = bool(match_data.get("team_a_name") and match_data.get("team_b_name"))
    if needs_both:
        is_available = has_a and has_b
    else:
        is_available = has_a or has_b

    if not is_available:
        return {
            "is_confirmed": False,
            "window_valid": True,
            "status": "incomplete_roster",
            "match_id": eff_match_id,
            "reason": "Confirmed 5-player rosters not available",
            "team_a_roster": team_a_roster,
            "team_b_roster": team_b_roster,
            "substitutions_detected": False,
            "substituted_players": {"team_a": [], "team_b": []},
            "substituted_player_ids": [],
        }

    # 4. Check for substitutions against baseline rosters if provided
    prev_a = extract_roster_from_dict(
        match_data,
        ["previous_roster_a", "team_a_previous_roster", "prev_roster_a", "baseline_roster_a"],
    ) or []
    prev_b = extract_roster_from_dict(
        match_data,
        ["previous_roster_b", "team_b_previous_roster", "prev_roster_b", "baseline_roster_b"],
    ) or []

    has_sub_a, subs_a = detect_lineup_substitution(prev_a, team_a_roster or [])
    has_sub_b, subs_b = detect_lineup_substitution(prev_b, team_b_roster or [])

    substitutions_detected = has_sub_a or has_sub_b
    all_substituted_ids = subs_a + subs_b

    return {
        "is_confirmed": True,
        "window_valid": True,
        "status": "confirmed",
        "match_id": eff_match_id,
        "team_a_roster": team_a_roster,
        "team_b_roster": team_b_roster,
        "substitutions_detected": substitutions_detected,
        "substitutions": {
            "team_a": {"has_sub": has_sub_a, "substituted_players": subs_a},
            "team_b": {"has_sub": has_sub_b, "substituted_players": subs_b},
        },
        "substituted_player_ids": all_substituted_ids,
        "reason": None,
    }


class ConfirmedLineupScraper:
    """Service to ingest and monitor confirmed starting lineups."""

    def __init__(
        self,
        window_minutes: int = DEFAULT_LINEUP_WINDOW_MINUTES,
        max_post_start_minutes: int = DEFAULT_MAX_POST_START_MINUTES,
    ) -> None:
        self.window_minutes = window_minutes
        self.max_post_start_minutes = max_post_start_minutes

    def check_match(
        self,
        match: dict[str, Any],
        current_time: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Check lineup status for a canonical match."""
        return check_confirmed_lineup(
            match_id=match.get("id"),
            match_start_at=match.get("start_time_normalized"),
            canonical_match=match,
            current_time=current_time,
            window_minutes=self.window_minutes,
            max_post_start_minutes=self.max_post_start_minutes,
        )

    def detect_substitutions(
        self,
        previous_roster: Sequence[Any],
        confirmed_roster: Sequence[Any],
    ) -> tuple[bool, list[str]]:
        """Detect substitutions between baseline and confirmed rosters."""
        return detect_lineup_substitution(previous_roster, confirmed_roster)
