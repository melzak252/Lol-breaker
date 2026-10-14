"""Pure temporal evidence resolution for team lineage and announced rosters.

This module deliberately does not infer identity from names or consult mutable
current-roster tables. Its inputs are immutable, timestamped evidence records.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _identifier(value: Any, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a non-empty exact ID")
    result = str(value).strip()
    if not result or result.casefold() in {"nan", "none", "null", "<na>", "nat"}:
        raise ValueError(f"{field} must be a non-empty exact ID")
    return result


def _clock(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    else:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _json_key(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _load_records(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"evidence file field {key!r} must be a list of records")
    return [dict(item) for item in value]


def load_roster_evidence(path: str | Path | None) -> dict[str, Any]:
    """Load immutable JSON evidence and return its SHA-256 keyed by path.

    The hash identifies local bytes; it is not a publication certification.
    """
    if path is None:
        return {"identity_links": [], "announcements": [], "input_hashes": {}}
    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid roster evidence JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("roster evidence JSON must contain an object")
    links = _load_records(payload.get("identity_links", []), "identity_links")
    announcements = _load_records(payload.get("announcements", []), "announcements")
    return {
        "identity_links": links,
        "announcements": announcements,
        "input_hashes": {str(source): hashlib.sha256(raw).hexdigest()},
    }


def _prepare_identity_links(
    identity_links: Iterable[Mapping[str, Any]], cutoff: datetime, start: datetime
) -> list[dict[str, Any]]:
    prepared = []
    for item in identity_links:
        if not isinstance(item, Mapping):
            raise ValueError("identity link must be an object")
        record = dict(item)
        source = _identifier(record.get("source_team_id"), "source_team_id")
        target = _identifier(record.get("target_team_id"), "target_team_id")
        published = _clock(record.get("published_at"), "identity published_at")
        observed = _clock(record.get("observed_at"), "identity observed_at")
        captured_available = _clock(record.get("captured_available_at", record["observed_at"]),
                                    "identity captured_available_at")
        if not record.get("evidence_uri"):
            raise ValueError("identity link requires evidence_uri")
        valid_from = _clock(record.get("valid_from"), "identity valid_from")
        valid_to = None if record.get("valid_to") is None else _clock(record["valid_to"], "identity valid_to")
        if valid_to is not None and valid_to <= valid_from:
            raise ValueError("identity validity interval must have valid_to after valid_from")
        if (published <= cutoff and observed <= cutoff and captured_available <= cutoff
                and valid_from <= start and (valid_to is None or start < valid_to)):
            prepared.append({
                "record": record, "source": source, "target": target,
                "published": published, "observed": observed,
            })
    return prepared


def _lineage(
    team: str, links: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any]], str | None, list[dict[str, Any]]]:
    # Links point from a predecessor (source) to its explicitly evidenced
    # successor (target). Conflicting parents and cycles are not resolved by
    # source ordering or input order.
    parents: dict[str, set[str]] = {}
    successors: dict[str, set[str]] = {}
    records_by_edge: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in links:
        source, target = item["source"], item["target"]
        parents.setdefault(target, set()).add(source)
        successors.setdefault(source, set()).add(target)
        records = records_by_edge.setdefault((source, target), [])
        if _json_key(item["record"]) not in {_json_key(record) for record in records}:
            records.append(item["record"])
    conflicted = {target for target, values in parents.items() if len(values) > 1}
    if team in conflicted:
        evidence = [item["record"] for item in links if item["target"] == team]
        return [team], [], "conflicting_identity_lineage", sorted(evidence, key=_json_key)

    history = [team]
    used: list[dict[str, Any]] = []
    seen = {team}
    current = team
    while current in parents:
        predecessor = next(iter(parents[current]))
        if len(successors[predecessor]) != 1:
            evidence = [item["record"] for item in links if item["source"] == predecessor]
            return [team], [], "conflicting_identity_successors", sorted(used + evidence, key=_json_key)
        edge_records = records_by_edge[(predecessor, current)]
        used.extend(edge_records)
        if predecessor in seen:
            return [team], [], "cyclic_identity_lineage", sorted(used, key=_json_key)
        if predecessor in conflicted:
            evidence = [item["record"] for item in links if item["target"] == predecessor]
            return [team], [], "conflicting_identity_lineage", sorted(used + evidence, key=_json_key)
        seen.add(predecessor)
        history.append(predecessor)
        current = predecessor
    # A record pointing outward from the requested current ID is a reversed
    # purported lineage, not evidence to merge the IDs.
    reversed_records = [item["record"] for item in links if item["source"] == team and item["target"] != team]
    if reversed_records:
        return [team], [], "reversed_identity_lineage", sorted(reversed_records, key=_json_key)
    return history, used, None, []


def _prepare_announcements(
    announcements: Iterable[Mapping[str, Any]], match_id: str, team_id: str,
    cutoff: datetime, start: datetime,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    candidates = []
    for item in announcements:
        if not isinstance(item, Mapping):
            raise ValueError("roster announcement must be an object")
        record = dict(item)
        match = _identifier(record.get("match_id"), "announcement match_id")
        team = _identifier(record.get("team_id"), "announcement team_id")
        published = _clock(record.get("published_at"), "announcement published_at")
        observed = _clock(record.get("observed_at"), "announcement observed_at")
        captured_available = _clock(record.get("captured_available_at", record["observed_at"]),
                                    "announcement captured_available_at")
        if not record.get("evidence_uri"):
            raise ValueError("roster announcement requires evidence_uri")
        status = str(record.get("status", "announced")).strip().lower()
        if status not in {"announced", "retracted"}:
            raise ValueError(f"unsupported roster announcement status: {status!r}")
        roster = record.get("roster")
        if status == "announced" and not isinstance(roster, (list, tuple)):
            raise ValueError("roster announcement requires a player ID sequence")
        if status == "retracted" and not isinstance(roster, (list, tuple)):
            roster = []
        if (match != match_id or team != team_id or published > cutoff
                or observed > cutoff or captured_available > cutoff):
            continue
        candidates.append((published, observed, record, list(roster), status))

    if not candidates:
        return None, None, []
    latest_key = max((published, observed) for published, observed, *_ in candidates)
    latest = [entry for entry in candidates if entry[:2] == latest_key]
    evidence = [entry[2] for entry in latest]
    canonical = {_json_key(entry[2]) for entry in latest}
    if len(canonical) != 1:
        return None, "conflicting_latest_announcements", evidence
    _, _, record, roster, status = latest[0]
    if status == "retracted":
        return None, "latest_announcement_retracted", evidence
    if len(roster) != 5:
        return None, "announced_roster_incomplete", evidence
    try:
        normalized = [_identifier(player, "announced player ID") for player in roster]
    except ValueError:
        return None, "announced_roster_incomplete", evidence
    if len(set(normalized)) != 5:
        return None, "announced_roster_duplicate_players", evidence
    record["roster"] = normalized
    return record, None, []




def resolve_roster_evidence(
    *,
    team_id: Any,
    match_id: Any,
    cutoff_at: Any,
    start_at: Any,
    identity_links: Iterable[Mapping[str, Any]] = (),
    announcements: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve explicit team ancestry and a fully announced match roster.

    All clocks are mandatory and timezone-aware. Identity links must be
    published and observed by cutoff and valid for the match start. An
    announcement must be exact-match/exact-team and all publication, observation
    and captured-availability clocks must be no later than cutoff < start.
    """
    team = _identifier(team_id, "team_id")
    match = _identifier(match_id, "match_id")
    cutoff = _clock(cutoff_at, "cutoff_at")
    start = _clock(start_at, "start_at")
    if cutoff >= start:
        raise ValueError("cutoff_at must be strictly before start_at")

    prepared_links = _prepare_identity_links(identity_links, cutoff, start)
    history, identity_used, identity_error, identity_ambiguity = _lineage(team, prepared_links)
    announcement, announcement_error, announcement_ambiguity = _prepare_announcements(
        announcements, match, team, cutoff, start
    )
    reason = identity_error or announcement_error
    return {
        "history_team_ids": history,
        "announcement": announcement,
        "identity_evidence": identity_used,
        "status": "ambiguous" if reason else "resolved",
        "missing_reason": reason,
        "ambiguity_evidence": identity_ambiguity or announcement_ambiguity,
    }
