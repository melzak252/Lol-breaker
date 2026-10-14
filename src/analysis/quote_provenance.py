"""Lossless as-of extraction of OddsPapi match-winner quotes.

Provider outcome IDs identify provider conventions, not bookmaker-specific
participant mappings. This extractor preserves identity evidence but never
certifies execution; callers need independently attributable bookmaker binding.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Any, Iterable


WINNER_MARKET_ID = "181"
OUTCOME_TEAM_1 = "181"
OUTCOME_TEAM_2 = "182"


def _aware_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _raw_timestamp(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _price(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 1.0 else None


def _fixture_time(fixture: dict[str, Any] | None) -> datetime | None:
    if not fixture:
        return None
    starts: list[datetime] = []
    for key in ("start_at", "startTime", "start_time", "trueStartTime", "true_start_time"):
        if key in fixture and fixture[key] is not None:
            parsed = _aware_utc(fixture[key])
            if parsed is None:
                raise ValueError(f"fixture {key} must be a timezone-aware timestamp")
            starts.append(parsed)
    return min(starts) if starts else None


def _observations(outcome: Any, cutoff: datetime | None, start: datetime | None) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    players = outcome.get("players") if isinstance(outcome, dict) else None
    if isinstance(players, dict):
        for player_key, player_rows in players.items():
            rows = player_rows if isinstance(player_rows, list) else [player_rows]
            for raw_state in rows:
                if not isinstance(raw_state, dict):
                    continue
                raw_created_at = raw_state.get("createdAt")
                source_created_at = _aware_utc(raw_created_at)
                if source_created_at is None:
                    status = "missing_or_naive_timestamp"
                elif cutoff is not None and source_created_at > cutoff:
                    status = "after_cutoff"
                elif start is not None and source_created_at >= start:
                    status = "at_or_after_fixture_start"
                else:
                    status = "eligible_for_latest"
                observations.append(
                    {
                        "player_key": str(player_key),
                        "created_at": _raw_timestamp(raw_created_at),
                        # createdAt orders provider states; it is not a local receipt clock.
                        "observed_at": None,
                        "status": status,
                        "raw_state": raw_state,
                    }
                )

    eligible = [item for item in observations if item["status"] == "eligible_for_latest"]
    if not observations:
        return {
            "status": "no_source_states",
            "source_created_at": None,
            "observed_at": None,
            "observations": observations,
            "latest_states": [],
        }
    if any(item["status"] == "missing_or_naive_timestamp" for item in observations):
        return {
            "status": "unorderable_state_present",
            "source_created_at": None,
            "observed_at": None,
            "observations": observations,
            "latest_states": [],
        }
    if not eligible:
        return {
            "status": "no_state_available_by_cutoff",
            "source_created_at": None,
            "observed_at": None,
            "observations": observations,
            "latest_states": [],
        }

    latest_time = max(_aware_utc(item["created_at"]) for item in eligible)
    latest = [item for item in eligible if _aware_utc(item["created_at"]) == latest_time]
    for item in latest:
        item["status"] = "latest_source_state"
    first_state = latest[0]["raw_state"]
    first_signature = {key: value for key, value in first_state.items() if key != "createdAt"}
    has_conflict = any(
        {key: value for key, value in item["raw_state"].items() if key != "createdAt"} != first_signature
        for item in latest[1:]
    )
    if has_conflict:
        status = "equal_time_conflict"
    else:
        raw_state = first_state
        active = raw_state.get("active")
        if active is not True:
            status = "inactive" if active is False else "active_state_unknown"
        elif _price(raw_state.get("price")) is None:
            status = "invalid_price"
        else:
            status = "active"
    return {
        "status": status,
        "source_created_at": latest_time.isoformat(),
        "observed_at": None,
        "observations": observations,
        "latest_states": latest,
    }


def _identity_fields(outcome: Any, latest_states: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    containers: list[tuple[str, Any]] = []
    if isinstance(outcome, dict):
        containers.append(("outcome", outcome))
    containers.extend(("snapshot", item["raw_state"]) for item in latest_states)

    id_keys = ("participantId", "teamId", "competitorId", "participant_id", "team_id")
    name_keys = ("participantName", "teamName", "competitorName", "participant_name", "team_name")
    for source, container in containers:
        if not isinstance(container, dict):
            continue
        for nested_key in ("participant", "team", "competitor"):
            nested = container.get(nested_key)
            if isinstance(nested, dict):
                container = {**container, **nested}
        participant_id = next((container[key] for key in id_keys if container.get(key) is not None), None)
        participant_name = next((container[key] for key in name_keys if container.get(key) is not None), None)
        if participant_id is not None or participant_name is not None:
            records.append(
                {
                    "source": source,
                    "participant_id": str(participant_id) if participant_id is not None else None,
                    "participant_name": str(participant_name) if participant_name is not None else None,
                }
            )

    ids = {item["participant_id"] for item in records if item["participant_id"] is not None}
    names = {item["participant_name"] for item in records if item["participant_name"] is not None}
    return {
        "participant_id": next(iter(ids)) if len(ids) == 1 else None,
        "participant_name": next(iter(names)) if len(names) == 1 else None,
        "evidence": records,
        "conflict": len(ids) > 1 or len(names) > 1,
        "present": bool(records),
    }


def _fixture_identity(fixture: dict[str, Any] | None, index: int) -> tuple[str | None, str | None]:
    if not fixture:
        return None, None
    id_keys = (f"participant{index}Id", f"participant_{index}_id", f"team{index}Id", f"provider_team_{index}_id")
    name_keys = (f"participant{index}Name", f"participant_{index}_name", f"team{index}Name", f"provider_team_{index}")
    participant_id = next((fixture[key] for key in id_keys if fixture.get(key) is not None), None)
    participant_name = next((fixture[key] for key in name_keys if fixture.get(key) is not None), None)
    return (
        str(participant_id) if participant_id is not None else None,
        str(participant_name) if participant_name is not None else None,
    )


def _normalized_name(value: str | None) -> str | None:
    return " ".join(value.casefold().split()) if value is not None else None


def _participant_binding_status(
    identities: tuple[dict[str, Any], dict[str, Any]], fixture: dict[str, Any] | None
) -> str:
    first, second = identities
    if first["conflict"] or second["conflict"]:
        return "conflicting_source_participant_fields"
    if not first["present"] and not second["present"]:
        return "unverified_provider_outcome_convention"
    if not first["present"] or not second["present"]:
        return "partial_source_participant_fields_unverified"

    expected = (_fixture_identity(fixture, 1), _fixture_identity(fixture, 2))
    source_names = (_normalized_name(first["participant_name"]), _normalized_name(second["participant_name"]))
    expected_names = (_normalized_name(expected[0][1]), _normalized_name(expected[1][1]))
    if all(source_names) and all(expected_names):
        if source_names == expected_names:
            return "source_participants_match_fixture_order_unverified"
        if source_names == tuple(reversed(expected_names)):
            return "source_participants_match_fixture_reverse_unverified"
        return "source_participants_mismatch_fixture"

    source_ids = (first["participant_id"], second["participant_id"])
    expected_ids = (expected[0][0], expected[1][0])
    if all(source_ids) and all(expected_ids):
        if source_ids == expected_ids:
            return "source_participants_match_fixture_order_unverified"
        if source_ids == tuple(reversed(expected_ids)):
            return "source_participants_match_fixture_reverse_unverified"
        return "source_participants_mismatch_fixture"
    return "explicit_source_participants_unverified"


def _identifier(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def _validate_participant_bindings(
    participant_bindings: Iterable[dict[str, Any]],
    *,
    source_fixture_id: Any,
    bookmaker: str,
    market_id: str,
    fixture: dict[str, Any] | None,
    cutoff: datetime | None,
    start: datetime | None,
    identities: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, int], list[str]]:
    records = list(participant_bindings or ())
    source_id = str(source_fixture_id) if source_fixture_id is not None else None
    fixture_id = _identifier(fixture, "fixtureId") if fixture is not None else None
    if source_id is not None and fixture_id is not None and source_id != fixture_id:
        return "source_fixture_mismatch", records, {}, ["source_fixture_id_mismatch"]
    if not records:
        return "unverified_provider_outcome_convention", [], {}, ["no_binding_evidence_supplied"]
    if cutoff is None:
        return "rejected_binding_evidence", records, {}, ["binding_cutoff_missing"]
    if start is not None and cutoff >= start:
        return "rejected_binding_evidence", records, {}, ["cutoff_not_before_fixture_start"]
    if source_id is None or fixture_id is None:
        return "rejected_binding_evidence", records, {}, ["binding_fixture_id_unavailable"]

    expected = {index: _fixture_identity(fixture, index) for index in (1, 2)}
    if any(not participant_id and not participant_name for participant_id, participant_name in expected.values()):
        return "rejected_binding_evidence", records, {}, ["fixture_participant_identity_incomplete"]

    reasons: list[str] = []
    indexes: dict[str, int] = {}
    seen_bookmaker_outcomes: set[str] = set()
    seen_bookmaker_participants: set[str] = set()
    seen_provider_participants: set[str] = set()
    bookmaker_fixture_ids: set[str] = set()
    bookmaker_market_ids: set[str] = set()

    for record in records:
        if not isinstance(record, dict):
            reasons.append("binding_record_not_an_object")
            continue
        if (
            _identifier(record, "fixture_id") != source_id
            or _identifier(record, "bookmaker") != bookmaker
            or _identifier(record, "market_id") != market_id
        ):
            reasons.append("binding_record_wrong_scope")
            continue

        outcome_id = _identifier(record, "outcome_id")
        participant_id = _identifier(record, "participant_id")
        participant_name = _identifier(record, "participant_name")
        bookmaker_fixture_id = _identifier(record, "bookmaker_fixture_id")
        bookmaker_market_id = _identifier(record, "bookmaker_market_id")
        bookmaker_outcome_id = _identifier(record, "bookmaker_outcome_id")
        bookmaker_participant_id = _identifier(record, "bookmaker_participant_id")
        evidence_uri = _identifier(record, "evidence_uri")
        published_at = _aware_utc(record.get("published_at"))
        observed_at = _aware_utc(record.get("observed_at"))
        if outcome_id not in (OUTCOME_TEAM_1, OUTCOME_TEAM_2):
            reasons.append("binding_outcome_id_invalid")
            continue
        if not all((participant_id, bookmaker_fixture_id, bookmaker_market_id,
                    bookmaker_outcome_id, bookmaker_participant_id, evidence_uri)):
            reasons.append("binding_identifiers_or_evidence_uri_missing")
            continue
        if published_at is None or observed_at is None or published_at > observed_at:
            reasons.append("binding_timestamps_missing_naive_or_inconsistent")
            continue
        if published_at > cutoff or observed_at > cutoff or (
            start is not None and (published_at >= start or observed_at >= start)
        ):
            reasons.append("binding_evidence_not_available_by_cutoff")
            continue

        participant_indexes = []
        for index, (expected_id, expected_name) in expected.items():
            if expected_id is not None:
                matches = participant_id == expected_id
            else:
                matches = (
                    participant_name is not None
                    and _normalized_name(participant_name) == _normalized_name(expected_name)
                )
            if matches and (
                participant_name is None
                or expected_name is None
                or _normalized_name(participant_name) == _normalized_name(expected_name)
            ):
                participant_indexes.append(index)
        if len(participant_indexes) != 1:
            reasons.append("binding_participant_mismatch_or_ambiguous_fixture")
            continue

        side_index = 1 if outcome_id == OUTCOME_TEAM_1 else 2
        identity = identities[side_index - 1]
        if identity["conflict"]:
            reasons.append("payload_participant_identity_conflict")
            continue
        source_participant_id = identity["participant_id"]
        source_participant_name = identity["participant_name"]
        if source_participant_id is not None and source_participant_id != participant_id:
            reasons.append("binding_conflicts_with_payload_participant_id")
            continue
        expected_name = expected[participant_indexes[0]][1]
        identity_name = participant_name or expected_name
        if (
            source_participant_name is not None
            and identity_name is not None
            and _normalized_name(source_participant_name) != _normalized_name(identity_name)
        ):
            reasons.append("binding_conflicts_with_payload_participant_name")
            continue
        if outcome_id in indexes:
            reasons.append("duplicate_provider_outcome_binding")
            continue

        indexes[outcome_id] = participant_indexes[0]
        seen_provider_participants.add(participant_id)
        seen_bookmaker_outcomes.add(bookmaker_outcome_id)
        seen_bookmaker_participants.add(bookmaker_participant_id)
        bookmaker_fixture_ids.add(bookmaker_fixture_id)
        bookmaker_market_ids.add(bookmaker_market_id)

    if reasons:
        return "rejected_binding_evidence", records, indexes, sorted(set(reasons))
    if (
        set(indexes) != {OUTCOME_TEAM_1, OUTCOME_TEAM_2}
        or set(indexes.values()) != {1, 2}
        or len(seen_provider_participants) != 2
        or len(seen_bookmaker_outcomes) != 2
        or len(seen_bookmaker_participants) != 2
        or len(bookmaker_fixture_ids) != 1
        or len(bookmaker_market_ids) != 1
    ):
        return "rejected_binding_evidence", records, {}, ["binding_set_incomplete_or_conflicting"]
    return "admitted_with_bookmaker_evidence", records, indexes, []




def _explicit_scope_status(*sources: Any) -> str | None:
    live_keys = {"live", "islive", "inplay", "isinplay", "liveonly"}
    map_keys = {
        "map", "mapid", "mapname", "mapnumber", "mapno", "mapindex", "maponly", "ismap",
        "gamenumber", "gameno", "gameindex",
    }
    scope_text_keys = {
        "scope", "scopetype", "marketscope", "period", "segment",
        "marketname", "markettype", "type", "status",
    }
    scope_container_keys = {"scope", "scopetype", "marketscope", "period", "segment"}
    live_found = False
    map_found = False

    def inspect(raw_key: Any, value: Any, *, in_scope: bool = False) -> None:
        nonlocal live_found, map_found
        key = re.sub(r"[^a-z]", "", str(raw_key).casefold())
        if key in live_keys and (
            value is True
            or isinstance(value, (int, float)) and not isinstance(value, bool) and value == 1
            or isinstance(value, str)
            and re.sub(r"[^a-z0-9]", "", value.casefold()) in ("true", "yes", "1", "live", "inplay")
        ):
            live_found = True
        if key in map_keys and value is not None and value is not False and value != "":
            map_found = True
        if (key in scope_text_keys or in_scope and key == "name") and isinstance(value, str):
            normalized = re.sub(r"[_-]+", " ", value.casefold())
            if re.search(r"\b(?:live|in play)\b", normalized):
                live_found = True
            if (
                re.search(r"\b(?:map|game)\s*(?:#|no\.?\s*)?\d+\b", normalized)
                or re.search(r"\bmap\s+(?:winner|moneyline)\b|\b(?:winner|moneyline)\s+map\b", normalized)
                or normalized.strip() in {"map", "map winner", "map moneyline"}
            ):
                map_found = True
        if key in scope_container_keys and isinstance(value, dict):
            for nested_key, nested_value in value.items():
                inspect(nested_key, nested_value, in_scope=True)

    for source in sources:
        if isinstance(source, dict):
            for key, value in source.items():
                inspect(key, value)
    if live_found:
        return "rejected_explicit_live_scope"
    if map_found:
        return "rejected_map_only_scope"
    return None


def extract_winner_quote(
    payload: dict[str, Any],
    bookmaker_slug: str,
    *,
    cutoff_at: datetime | None = None,
    fixture: dict[str, Any] | None = None,
    participant_bindings: Iterable[dict[str, Any]] = (),
) -> dict[str, Any] | None:
    """Return source-ordered winner-market prices and their complete state evidence.

    A quote side is usable only when its latest source state at or before the
    cutoff (and strictly before fixture start, when supplied) is explicitly
    active, timestamped, conflict-free, and has a finite decimal price above 1.
    All source states remain in the returned evidence, including inactive,
    malformed, and future updates.
    Independent participant_bindings can admit an evidenced source mapping, but
    never certify execution.
    """
    if cutoff_at is not None and _aware_utc(cutoff_at) is None:
        raise ValueError("cutoff_at must be a timezone-aware timestamp")
    cutoff = _aware_utc(cutoff_at) if cutoff_at is not None else None
    start = _fixture_time(fixture)

    books = payload.get("bookmakerOdds") or payload.get("bookmakers") or {}
    bookmaker = books.get(bookmaker_slug) if isinstance(books, dict) else None
    if not isinstance(bookmaker, dict):
        return None
    markets = bookmaker.get("markets") or {}
    market = markets.get(WINNER_MARKET_ID) if isinstance(markets, dict) else None
    if not isinstance(market, dict):
        return None
    outcomes = market.get("outcomes") or {}
    if not isinstance(outcomes, dict):
        outcomes = {}

    source_outcome_1 = outcomes.get(OUTCOME_TEAM_1)
    source_outcome_2 = outcomes.get(OUTCOME_TEAM_2)
    state_1 = _observations(source_outcome_1, cutoff, start)
    state_2 = _observations(source_outcome_2, cutoff, start)
    payload_scope = {key: value for key, value in payload.items() if key not in ("bookmakerOdds", "bookmakers")}
    fixture_scope = payload.get("fixture")
    bookmaker_scope = {key: value for key, value in bookmaker.items() if key != "markets"}
    market_scope = {key: value for key, value in market.items() if key != "outcomes"}
    if "name" in market_scope:
        market_scope.setdefault("marketName", market_scope.pop("name"))
    outcome_scope_1 = {key: value for key, value in source_outcome_1.items() if key != "players"} if isinstance(source_outcome_1, dict) else {}
    outcome_scope_2 = {key: value for key, value in source_outcome_2.items() if key != "players"} if isinstance(source_outcome_2, dict) else {}
    latest_state_scopes = [
        item["raw_state"]
        for state in (state_1, state_2)
        for item in state["latest_states"]
    ]
    explicit_scope_status = _explicit_scope_status(
        fixture_scope,
        payload_scope,
        bookmaker_scope,
        market_scope,
        outcome_scope_1,
        outcome_scope_2,
        *latest_state_scopes,
    )
    identities = (
        _identity_fields(source_outcome_1, state_1["latest_states"]),
        _identity_fields(source_outcome_2, state_2["latest_states"]),
    )
    quote_admission_status, binding_records, outcome_indexes, binding_reasons = _validate_participant_bindings(
        participant_bindings,
        source_fixture_id=payload.get("fixtureId"),
        bookmaker=bookmaker_slug,
        market_id=WINNER_MARKET_ID,
        fixture=fixture,
        cutoff=cutoff,
        start=start,
        identities=identities,
    )
    if explicit_scope_status is not None:
        quote_admission_status = explicit_scope_status
        outcome_indexes = {}

    def usable_price(state: dict[str, Any]) -> float | None:
        if explicit_scope_status is not None or state["status"] != "active" or not state["latest_states"]:
            return None
        return _price(state["latest_states"][0]["raw_state"].get("price"))

    odds_1 = usable_price(state_1)
    odds_2 = usable_price(state_2)
    quote_at_1 = state_1["latest_states"][0]["created_at"] if state_1["latest_states"] else None
    quote_at_2 = state_2["latest_states"][0]["created_at"] if state_2["latest_states"] else None
    quote_at = None
    if odds_1 is not None and odds_2 is not None:
        time_1 = _aware_utc(quote_at_1)
        time_2 = _aware_utc(quote_at_2)
        if time_1 is not None and time_2 is not None:
            quote_at = max(time_1, time_2).isoformat()

    limitations: list[str] = []
    if explicit_scope_status is not None:
        limitations.append(explicit_scope_status)
    elif quote_admission_status == "unverified_provider_outcome_convention":
        limitations.append("no_independent_bookmaker_participant_binding_evidence")
    elif quote_admission_status != "admitted_with_bookmaker_evidence":
        limitations.append("bookmaker_participant_binding_evidence_rejected")
        limitations.extend(binding_reasons)
    if cutoff is None:
        limitations.append("cutoff_not_supplied")
    if payload.get("fixtureId") is None:
        limitations.append("source_fixture_id_missing")
    if odds_1 is None or odds_2 is None:
        limitations.append("complete_two_sided_active_quote_unavailable")
    for side, state in (("outcome_1", state_1), ("outcome_2", state_2)):
        if state["status"] != "active":
            limitations.append(f"{side}_latest_state_{state['status']}")
    limitations.append("quote_admission_does_not_certify_execution")

    participant_binding_status = (
        quote_admission_status
        if quote_admission_status != "unverified_provider_outcome_convention"
        else _participant_binding_status(identities, fixture)
    )
    fixture_identity_1 = _fixture_identity(fixture, 1)
    fixture_identity_2 = _fixture_identity(fixture, 2)
    source_fixture_id = payload.get("fixtureId")
    context_fixture_id = _identifier(fixture, "fixtureId") if fixture is not None else None
    source_fixture_id_matches_context = (
        str(source_fixture_id) == context_fixture_id
        if source_fixture_id is not None and context_fixture_id is not None
        else None
    )
    binding_by_outcome = {
        _identifier(record, "outcome_id"): record
        for record in binding_records
        if isinstance(record, dict)
    } if quote_admission_status == "admitted_with_bookmaker_evidence" else {}
    binding_1 = binding_by_outcome.get(OUTCOME_TEAM_1, {})
    binding_2 = binding_by_outcome.get(OUTCOME_TEAM_2, {})
    return {
        "odds_1": odds_1,
        "odds_2": odds_2,
        "quote_at_1": quote_at_1,
        "quote_at_2": quote_at_2,
        "quote_at": quote_at,
        "cutoff_at": cutoff.isoformat() if cutoff is not None else None,
        "fixture_start_at": start.isoformat() if start is not None else None,
        "fixture_participant_1_id": fixture_identity_1[0],
        "fixture_participant_1_name": fixture_identity_1[1],
        "fixture_participant_2_id": fixture_identity_2[0],
        "fixture_participant_2_name": fixture_identity_2[1],
        "market_id": WINNER_MARKET_ID,
        "outcome_id_1": OUTCOME_TEAM_1 if source_outcome_1 is not None else None,
        "outcome_id_2": OUTCOME_TEAM_2 if source_outcome_2 is not None else None,
        "source_fixture_id": payload.get("fixtureId"),
        "bookmaker": bookmaker_slug,
        "source_market_evidence": market,
        "source_outcome_evidence_1": source_outcome_1,
        "source_outcome_evidence_2": source_outcome_2,
        "participant_binding_status": participant_binding_status,
        "quote_admission_status": quote_admission_status,
        "explicit_scope_status": explicit_scope_status,
        "source_fixture_id_matches_context": source_fixture_id_matches_context,
        "participant_binding_evidence": binding_records,
        "binding_validation_reasons": binding_reasons,
        "outcome_fixture_participant_indexes": outcome_indexes,
        "participant_1_id": binding_1.get("participant_id") or identities[0]["participant_id"],
        "participant_1_name": binding_1.get("participant_name") or identities[0]["participant_name"],
        "participant_2_id": binding_2.get("participant_id") or identities[1]["participant_id"],
        "participant_2_name": binding_2.get("participant_name") or identities[1]["participant_name"],
        "participant_evidence_1": identities[0]["evidence"],
        "participant_evidence_2": identities[1]["evidence"],
        "state_1": state_1,
        "state_2": state_2,
        "execution_eligibility": "not_certified",
        "provenance_limitations": limitations,
    }
