"""Release-aware daily as-of reconstruction for the frozen EV experiment.

This module reconstructs a historical scenario, not certified announcement or
forecast-publication times.  It consumes the frozen GOL.GG JSON array through
its archived streaming reader and evaluates annual frozen models on the exact
same reconstructed snapshots.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SHA256 = "7bca9fc39bad77d3ecb84f4cd607a076861630dd14cb16467abb30f32db2b0cd"
BANK_RECEIPT_SHA256 = "bc4fd6375cbc3e1728808ad6e2f7b232b0dac09f69730d0db50c632d6e0aef1f"
BANK_SOURCE_SHA256 = "991c23cdc67a119a6a2be6b04cb60018ab0f3eeec979f34871f8213f9d6899ca"
BANK_RELEASE_SHA256 = "e54914ba79c5620a819a48a03bf877c373c6e53bd019219e43ffdd7db47c2620"
FEATURE_ARCHIVE_RELATIVE = Path("unified-ratings-20260911")
CAUSAL_ARCHIVE_RELATIVE = Path("a0-phase-walkforward-20260914")
BANK_RELATIVE = Path("a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank")
K5_MODEL_RELATIVE = Path("data/06_models/a0_uncertainty/a0_uncertainty_20260926_222155")
RAW_HISTORY_RELATIVE = Path("data/artifacts/golgg-database-recovery-20260909/matches.json")
FEATURE_MANIFEST_SHA256 = "d0de3217e13304d9b907b74baed7742fceea62f0d2289e476f25ced5c6b96597"
MODEL_YEARS = (2024, 2025, 2026)
REPLAY_SCHEMA_VERSION = 4
REPLAY_CONTRACT_VERSION = "a0_input_repair_asof_v4"
FORECASTERS = ("native_annual_a0", "seed_only", "block_bootstrap")
PRODUCTION_HORIZONS = ("OPEN", "48h", "24h", "CLOSE")
ASOF_COHORT = "production_asof_previous_roster"
REPLAY_RUN_ID = "a0_input_repair_20260930_074319"

# These identities are cross-checked against the complete archived manifest and
# the bytes that are actually imported.  In particular, a same-named local
# script is never accepted as the replay implementation.
FEATURE_SOURCE_SHA256 = {
    "code/scripts/local_history_stream.py": "e1827fabee6f769b6d85e029ec18a62979d4df22e34915d9cea68f395d506c2c",
    "code/scripts/temporal_player_features.py": "1d2dec76d44f6b6544217fb688116b1cfc849fda118bbf57621c2af1b044424c",
    "code/scripts/roster_optional_features.py": "47947f0a3c9fdd17407a5da0ff769cffdce391b7f34fc8cf8d90443e07bbcd6d",
    "code/scripts/prospective_sports_features.py": "d7073842effb75b4a0b37f2264655f51827b147ba5108045ac376c3a84ea899f",
}
FEATURE_AUDIT_SHA256 = "5d1a70fa8d6ed70901348e8c71aef06b944ea956419610980b541addcfd12826"
NATIVE_SOURCE_SHA256 = {
    "code/causal_predictor.py": "6bc40d8c5993aaea1fe464af5934812516dbe9adf545d0832640ec4d3b2cf140",
    "code/a0_predictor.py": "4c8de9095d00be8a117057410b301de88af02670b075d36b1879be5f1f35407f",
    "code/causal_nn.py": "afabdd9169c9eb8df95b74f3c225fd37c158e989cf3a1ea41801d41ad97a3d3c",
    "code/causal_calibrate.py": "fdb526e652f1a1316810f1d4e30b090d1791b92f859794c628995b5e95528a26",
}

_RAW_SHAPES = {
    "players": (2, 5, 21),
    "core": (2, 5),
    "team": (2, 11),
    "w20": (2, 10),
    "gates": (2, 2),
}
_TEMPORAL_SHAPES = {
    "history": (2, 5, 16, 33),
    "mask": (2, 5, 16),
    "champions": (2, 5, 16),
}


@dataclass(frozen=True, slots=True)
class AsOfSnapshot:
    """One row-aligned forecast snapshot with scoring-only labels kept separate."""

    target: dict[str, Any]
    raw: dict[str, np.ndarray] | None
    hist: dict[str, np.ndarray] | None
    available: bool
    missing_reason: str | None


@dataclass(slots=True)
class _ReplayArrays:
    targets: pd.DataFrame
    raw: dict[str, np.ndarray]
    hist: dict[str, np.ndarray]
    validation_receipt: dict[str, Any]
    history_audit: dict[str, Any]
    release_audit: dict[str, Any]
    feature_source_hashes: dict[str, str]
    label_release_days: dict[str, date]
    roster_evidence_input_hashes: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read frozen JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}: {path}")
    return actual


def _project_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return (PROJECT_ROOT / default).resolve()
    supplied = Path(value).expanduser()
    return (PROJECT_ROOT / supplied).resolve() if not supplied.is_absolute() else supplied.resolve()


def _normalize_id(value: Any, label: str) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Invalid {label}: {value!r}")
    if isinstance(value, (int, np.integer)):
        result = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"Invalid {label}: {value!r}")
        result = str(int(value))
    else:
        result = str(value).strip()
    if not result or result.lower() in {"none", "null", "nan", "<na>"}:
        return None
    return result


def _as_day(value: Any, label: str) -> date:
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            return value.tz_convert("UTC").date()
        return value.date()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid {label}: {value!r}") from error
    raise ValueError(f"Invalid {label}: {value!r}")


def _timestamp(value: Any, label: str) -> pd.Timestamp | pd.NaT:
    if value is None or value is pd.NA or value is pd.NaT:
        return pd.NaT
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if pd.isna(result):
        return pd.NaT
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"Timezone-aware timestamp required in {label}: {value!r}")
    return result.tz_convert("UTC")


def _source_series_order(match_id: str) -> tuple[int, int | str]:
    # Mirrors the archived _order: numeric GOL.GG IDs sort numerically, then
    # nonnumeric fixture IDs sort lexically. This is the bank's stable tie-break.
    return (0, int(match_id)) if match_id.isascii() and match_id.isdecimal() else (1, match_id)


def _order_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, tuple) and len(value) == 2 and value[0] in (0, 1):
        return (int(value[0]), value[1])
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return (0, int(value))
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return (0, int(value))
    return (1, str(value))


def _event_roster(event: Mapping[str, Any]) -> list[str | None]:
    roster = event.get("roster")
    if not isinstance(roster, (list, tuple)):
        raise ValueError("roster event must contain a player sequence")
    if len(roster) > 5:
        raise ValueError("roster event cannot contain more than five player slots")
    normalized = [_normalize_id(value, "roster player ID") for value in roster]
    known = [value for value in normalized if value is not None]
    if len(known) != len(set(known)):
        raise ValueError("roster event contains duplicate known player IDs")
    return [*normalized, *([None] * (5 - len(normalized)))]


def _canonical_competition_context(value: Any) -> str | None:
    normalized = _normalize_id(value, "competition context")
    return normalized.strip() if normalized is not None and normalized.strip() else None


def _target_competition_context(metadata: Mapping[str, Any], match_id: str) -> str | None:
    tournament = _canonical_competition_context(metadata.get("tournament_name"))
    supplied = _canonical_competition_context(metadata.get("competition_context"))
    if tournament and supplied and tournament != supplied:
        raise ValueError(f"Target tournament_name and competition_context disagree for {match_id}")
    return tournament or supplied

def _select_prior_event_for_teams(
    events: Iterable[Mapping[str, Any]],
    team_ids: Iterable[str],
    competition_context: str | None,
    cutoff_day: Any,
) -> Mapping[str, Any] | None:
    teams = set(team_ids)
    if not teams or competition_context is None:
        return None
    cutoff = _as_day(cutoff_day, "cutoff day")
    winner: Mapping[str, Any] | None = None
    winner_key: tuple[Any, ...] | None = None
    for event in events:
        if _normalize_id(event.get("team_id"), "roster event team ID") not in teams:
            continue
        if _canonical_competition_context(event.get("competition_context")) != competition_context:
            continue
        release = _as_day(event.get("effective_release_day"), "effective release day")
        if release >= cutoff:
            continue
        key = (
            release,
            _order_key(event.get("source_series_order", 0)),
            int(event.get("map_index", 0)),
            str(event.get("source_match_id", "")),
        )
        if winner_key is None or key > winner_key:
            winner, winner_key = event, key
    return winner


def _select_prior_event(
    events: Iterable[Mapping[str, Any]],
    team_id: Any,
    competition_context: str | None,
    cutoff_day: Any,
) -> Mapping[str, Any] | None:
    team = _normalize_id(team_id, "team ID")
    if team is None:
        return None
    return _select_prior_event_for_teams(events, [team], competition_context, cutoff_day)


def select_prior_roster(
    roster_events: Iterable[Mapping[str, Any]] | Mapping[str, Iterable[Mapping[str, Any]]],
    team_id: Any,
    competition_context: str,
    cutoff_day: Any,
) -> list[str | None]:
    """Select the latest strictly prior roster within an exact competition context."""
    context = _canonical_competition_context(competition_context)
    if context is None:
        raise ValueError("competition_context must be a nonempty exact context")
    team = _normalize_id(team_id, "team ID")
    if team is None:
        return [None] * 5
    candidates = roster_events.get(team, ()) if isinstance(roster_events, Mapping) else roster_events
    selected = _select_prior_event(candidates, team, context, cutoff_day)
    return [None] * 5 if selected is None else _event_roster(selected)

def _roster_evidence_input(value: Any) -> dict[str, Any]:
    from src.analysis.roster_provenance import load_roster_evidence

    if value is None:
        return load_roster_evidence(None)
    if isinstance(value, (str, Path)):
        return load_roster_evidence(_project_path(value, Path(value)))
    if not isinstance(value, Mapping):
        raise ValueError("roster_evidence must be a local path or loaded evidence mapping")
    identity_links = value.get("identity_links", ())
    announcements = value.get("announcements", ())
    input_hashes = value.get("input_hashes", {})
    if not isinstance(identity_links, (list, tuple)) or not isinstance(announcements, (list, tuple)):
        raise ValueError("Loaded roster evidence must contain identity_links and announcements sequences")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("Loaded roster evidence input_hashes must be a mapping")
    return {
        "identity_links": list(identity_links),
        "announcements": list(announcements),
        "input_hashes": dict(input_hashes),
    }


def _roster_source(
    event: Mapping[str, Any] | None,
    *,
    source_type: str = "unresolved",
    history_team_ids: Sequence[str] = (),
    identity_evidence: Sequence[Mapping[str, Any]] = (),
    ambiguity_evidence: Sequence[Mapping[str, Any]] = (),
    missing_reason: str | None = None,
) -> dict[str, Any]:
    if event is None:
        return {
            "source_type": source_type,
            "match_id": None,
            "team_id": None,
            "effective_release_day": None,
            "original_map_day": None,
            "map_index": None,
            "source_series_order": None,
            "release_source": None,
            "published_at": None,
            "observed_at": None,
            "evidence_uri": None,
            "roster": [None] * 5,
            "known_slots": 0,
            "history_team_ids": list(history_team_ids),
            "identity_evidence": list(identity_evidence),
            "ambiguity_evidence": list(ambiguity_evidence),
            "missing_reason": missing_reason,
        }
    roster = _event_roster(event)
    return {
        "source_type": source_type,
        "match_id": event.get("match_id", event.get("source_match_id")),
        "team_id": event.get("team_id"),
        "effective_release_day": (
            _as_day(event["effective_release_day"], "effective release day").isoformat()
            if event.get("effective_release_day") is not None else None
        ),
        "original_map_day": event.get("original_map_day"),
        "map_index": int(event.get("map_index", 0)) if event.get("map_index") is not None else None,
        "source_series_order": event.get("source_series_order"),
        "release_source": event.get("release_source"),
        "published_at": event.get("published_at"),
        "observed_at": event.get("observed_at"),
        "evidence_uri": event.get("evidence_uri"),
        "roster": roster,
        "known_slots": sum(player is not None for player in roster),
        "history_team_ids": list(history_team_ids),
        "identity_evidence": list(identity_evidence),
        "missing_reason": missing_reason,
        "ambiguity_evidence": list(ambiguity_evidence),
    }


def _resolve_roster_side(
    *,
    team_id: str | None,
    match_id: str | None,
    cutoff_at: pd.Timestamp | pd.NaT,
    start_at: pd.Timestamp | pd.NaT,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if team_id is None or pd.isna(cutoff_at):
        return {
            "history_team_ids": [team_id] if team_id is not None else [],
            "announcement": None,
            "identity_evidence": [],
            "ambiguity_evidence": [],
            "status": "ambiguous",
            "missing_reason": "team_identity_or_decision_cutoff_unavailable",
        }
    if not pd.isna(start_at) and cutoff_at >= start_at:
        return {
            "history_team_ids": [team_id],
            "announcement": None,
            "identity_evidence": [],
            "ambiguity_evidence": [],
            "status": "ambiguous",
            "missing_reason": "decision_cutoff_not_before_match_start",
        }
    # Historical exact-ID lookup remains usable without a match-start clock.
    # Temporal links and announcements need a real start time for validity and
    # cutoff-before-start admission; do not substitute a calendar date.
    if pd.isna(start_at):
        return {
            "history_team_ids": [team_id],
            "announcement": None,
            "identity_evidence": [],
            "ambiguity_evidence": [],
            "status": "resolved",
            "missing_reason": None,
        }
    from src.analysis.roster_provenance import resolve_roster_evidence

    return resolve_roster_evidence(
        team_id=team_id,
        match_id=match_id,
        cutoff_at=cutoff_at,
        start_at=start_at,
        identity_links=evidence["identity_links"],
        announcements=evidence["announcements"],
    )


def _target_forecast_eligibility(
    target: Mapping[str, Any],
    cutoff_day: date | None,
    native_release_day: Any = None,
) -> dict[str, Any]:
    """Gate daily forecasts on independent native and explicit label releases.

    Same-day releases remain forecast-eligible and excluded from daily history.
    Played/result dates are not publication evidence; unknown stays unknown.
    """
    if native_release_day is None or pd.isna(native_release_day):
        native_release_day = target.get("target_native_effective_release_day")
    native_day = (
        None if native_release_day is None or pd.isna(native_release_day)
        else _as_day(native_release_day, "native target effective release day")
    )
    label_day = None
    for column in (
        "effective_label_release_day", "label_release_day", "result_release_day",
        "target_effective_label_release_day",
    ):
        value = target.get(column)
        if value is not None and not pd.isna(value):
            label_day = _as_day(value, "target effective label release day")
            break
    forbidden = cutoff_day is not None and any(
        day is not None and day < cutoff_day for day in (native_day, label_day)
    )
    return {
        "forecast_eligible": not forbidden,
        "forecast_absence": "target_already_released_before_cutoff" if forbidden else None,
        "target_native_effective_release_day": native_day.isoformat() if native_day is not None else None,
        "target_effective_label_release_day": label_day.isoformat() if label_day is not None else None,
    }


def _target_native_release_inventory(
    roster_events: Iterable[Mapping[str, Any]],
    supplied: Mapping[str, Any] | None = None,
) -> dict[str, date]:
    releases: dict[str, date] = {}
    if supplied is not None:
        for identifier, value in supplied.items():
            identifier = _normalize_id(identifier, "native target series ID")
            if identifier is None:
                raise ValueError("Missing native target series ID")
            releases[identifier] = _as_day(value, "native target effective release day")
        return releases
    for event in roster_events:
        identifier = _normalize_id(event.get("source_match_id"), "source match ID")
        if identifier is None:
            continue
        day = _as_day(event["effective_release_day"], "effective release day")
        if identifier in releases and releases[identifier] != day:
            raise ValueError(f"Inconsistent roster/history release for series {identifier}")
        releases[identifier] = day
    return releases



def build_asof_targets(
    opportunities: pd.DataFrame,
    metadata: pd.DataFrame,
    roster_events: Iterable[Mapping[str, Any]],
    *,
    roster_evidence: Any = None,
    context_provenance: Mapping[str, str] | None = None,
    target_native_release_days: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build leakage-safe target rows and explicitly sourced roster scenarios."""
    required = {"row_id", "golgg_match_id"}
    missing = sorted(required.difference(opportunities.columns))
    if missing:
        raise ValueError(f"opportunities missing required fields: {missing}")
    if opportunities["row_id"].isna().any() or opportunities["row_id"].astype(str).duplicated().any():
        raise ValueError("opportunity row_id values must be present and unique")
    meta = _metadata_index(metadata)
    events = list(roster_events)
    by_team = _index_roster_events(events)
    native_releases = _target_native_release_inventory(events, target_native_release_days)
    effective_at = _effective_decision_times(opportunities)
    evidence = _roster_evidence_input(roster_evidence)
    rows: list[dict[str, Any]] = []
    for position, (_, source_row) in enumerate(opportunities.iterrows()):
        record = source_row.to_dict()
        match_id = _normalize_id(record.get("golgg_match_id"), "opportunity match ID")
        target_meta = meta.get(match_id or "", {})
        context = _target_competition_context(target_meta, match_id or "")
        for column, value in target_meta.items():
            if column in {"team1_id", "team2_id", "best_of", "y", "date", "result_day", "result_release_day", "result_release_source", "effective_label_release_day", "label_release_day", "tournament_name", "competition_context"}:
                record[column] = value
        record["tournament_name"] = context
        record["competition_context"] = context
        decision = effective_at.iloc[position]
        cutoff = None if pd.isna(decision) else decision.tz_convert("UTC").date()
        decision_at = pd.NaT if pd.isna(decision) else _timestamp(decision, "effective decision time")
        record.update(_target_forecast_eligibility(record, cutoff, native_releases.get(match_id)))
        start_value = source_row.get("start_at", pd.NaT)
        start_at = _timestamp(start_value, "target start time")
        team_a = _normalize_id(record.get("team1_id"), "team1_id")
        team_b = _normalize_id(record.get("team2_id"), "team2_id")
        if not record["forecast_eligible"]:
            side_a = {
                "status": "unavailable",
                "missing_reason": record["forecast_absence"],
            }
            side_b = dict(side_a)
        else:
            side_a = _resolve_roster_side(
                team_id=team_a, match_id=match_id, cutoff_at=decision_at,
                start_at=start_at, evidence=evidence,
            )
            side_b = _resolve_roster_side(
                team_id=team_b, match_id=match_id, cutoff_at=decision_at,
                start_at=start_at, evidence=evidence,
            )

        sides = []
        for team_id, resolution in ((team_a, side_a), (team_b, side_b)):
            if resolution.get("status") == "resolved":
                history_team_ids = [
                    normalized for value in resolution.get("history_team_ids", ())
                    if (normalized := _normalize_id(value, "temporal history team ID")) is not None
                ]
            else:
                history_team_ids = [team_id] if team_id is not None else []
            # The pure resolver returns the original ID even without lineage
            # evidence. Never infer a predecessor from an ambiguous resolution.
            if team_id is not None and team_id not in history_team_ids:
                history_team_ids.insert(0, team_id)
            event = _select_prior_event_for_teams(
                (item for candidate in history_team_ids for item in by_team.get(candidate, ())),
                history_team_ids,
                context,
                cutoff,
            ) if record["forecast_eligible"] and cutoff is not None and context is not None else None
            announcement = resolution.get("announcement")
            if announcement is not None:
                if not isinstance(announcement, Mapping):
                    raise ValueError("Resolved roster announcement must be an object")
                selected_event = dict(announcement)
                selected_event.setdefault("team_id", team_id)
                source_type = "announced"
            else:
                selected_event = event
                source_type = "prior_observed_scenario" if event is not None else "unresolved"
            side_missing_reason = resolution.get("missing_reason")
            if context is None and announcement is None and resolution.get("status") == "resolved":
                side_missing_reason = "target_competition_context_unavailable"
                resolution["missing_reason"] = side_missing_reason
            elif event is None and announcement is None and cutoff is not None and resolution.get("status") == "resolved":
                side_missing_reason = "no_prior_roster_in_target_competition_context"
                resolution["missing_reason"] = side_missing_reason
            roster = [None] * 5 if selected_event is None else _event_roster(selected_event)
            source = _roster_source(
                selected_event,
                source_type=source_type,
                history_team_ids=history_team_ids,
                identity_evidence=resolution.get("identity_evidence", ()),
                ambiguity_evidence=resolution.get("ambiguity_evidence", ()),
                missing_reason=side_missing_reason,
            )
            sides.append({
                "roster": roster,
                "source": source,
                "source_type": source_type,
                "resolution": resolution,
                "event": selected_event,
                "history_team_ids": history_team_ids,
            })
        roster_a, roster_b = sides[0]["roster"], sides[1]["roster"]
        overlap = sorted(set(player for player in roster_a if player is not None).intersection(
            player for player in roster_b if player is not None
        ))
        ambiguity = any(side["resolution"].get("status") != "resolved" for side in sides)
        if (
            ambiguity or overlap or not _valid_roster(roster_a) or not _valid_roster(roster_b)
        ):
            status = "unresolved"
        elif any(side["source_type"] == "announced" for side in sides):
            status = "announced"
        else:
            status = "prior_observed_scenario"
        missing_reasons = [
            side["resolution"].get("missing_reason")
            for side in sides if side["resolution"].get("missing_reason")
        ]
        roster_missing_reason = (
            "overlapping_prior_rosters" if overlap
            else str(missing_reasons[0]) if missing_reasons
            else "incomplete_or_unknown_roster_identity" if status == "unresolved"
            else None
        )
        assumption = record.get("scenario_timestamp_assumption")
        if assumption is not None and pd.isna(assumption):
            assumption = None
        decision_time_source = "actual" if "decision_at" in source_row and not pd.isna(source_row["decision_at"]) else (
            "scenario" if cutoff is not None else None
        )
        record.update({
            "row_order": position,
            "effective_decision_at": decision,
            "decision_time_source": decision_time_source,
            "cutoff_day": cutoff.isoformat() if cutoff is not None else None,
            "model_year": int(cutoff.year) if cutoff is not None else None,
            "roster_a": roster_a,
            "roster_b": roster_b,
            "observed_roster_a": roster_a,
            "observed_roster_b": roster_b,
            "roster_status": status,
            "roster_missing_reason": roster_missing_reason,
            "roster_evidence_status_a": sides[0]["resolution"].get("status"),
            "roster_evidence_status_b": sides[1]["resolution"].get("status"),
            "roster_evidence_missing_reason_a": sides[0]["resolution"].get("missing_reason"),
            "roster_evidence_missing_reason_b": sides[1]["resolution"].get("missing_reason"),
            "roster_identity_evidence_a": sides[0]["resolution"].get("identity_evidence", []),
            "roster_identity_evidence_b": sides[1]["resolution"].get("identity_evidence", []),
            "roster_ambiguity_evidence_a": sides[0]["resolution"].get("ambiguity_evidence", []),
            "roster_ambiguity_evidence_b": sides[1]["resolution"].get("ambiguity_evidence", []),
            "roster_announcement_a": sides[0]["resolution"].get("announcement"),
            "roster_announcement_b": sides[1]["resolution"].get("announcement"),
            "roster_context_provenance": (
                (context_provenance.get(match_id, "target_metadata_exact")
                 if context_provenance is not None else "target_metadata_exact")
                if context is not None
                else "missing_target_competition_context"
            ),
            "roster_evidence_input_hashes": evidence["input_hashes"],
            "overlapping_prior_player_ids": overlap,
            "draft_status": "unknown",
            "roster_source_a": sides[0]["source"],
            "roster_source_b": sides[1]["source"],
            "target_metadata_matched": bool(target_meta),
            "replay_contract": REPLAY_CONTRACT_VERSION,
            "scenario_timestamp_assumption": assumption,
        })
        record.pop("draft", None)
        record.pop("draft_a", None)
        record.pop("draft_b", None)
        rows.append(record)
    return pd.DataFrame(rows).reset_index(drop=True)




def _raw_release_day(raw_match: Mapping[str, Any], release_days: Mapping[str, Any]) -> tuple[date, str, str]:
    match_id = _normalize_id(raw_match.get("match_id"), "source match ID")
    if match_id is None:
        raise ValueError("historical series has no usable match_id")
    games = raw_match.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError(f"historical series {match_id} has no completed maps")
    days: list[date] = []
    for index, game in enumerate(games, start=1):
        if not isinstance(game, Mapping) or "date" not in game:
            raise ValueError(f"historical series {match_id} map {index} has no source date")
        days.append(_as_day(game["date"], f"historical map {match_id}/{index} date"))
    latest = max(days)
    has_override = match_id in release_days
    release_value = release_days.get(match_id, latest.isoformat())
    try:
        override_day = date.fromisoformat(str(release_value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid corrected release day for series {match_id}: {release_value!r}") from error
    effective = max(latest, override_day)
    provenance = "corrected_release_map_override" if has_override else "latest_source_game_day_fallback"
    effective_reason = "override_day" if has_override and override_day >= latest else "source_game_day_floor"
    return effective, provenance, effective_reason


def _player_id_from_payload(player: Any, context: str) -> str | None:
    if not isinstance(player, Mapping):
        raise ValueError(f"Invalid player payload in {context}")
    return _normalize_id(player.get("player_id", player.get("id")), f"player ID in {context}")


def _events_for_raw_match(
    raw_match: Mapping[str, Any], release_days: Mapping[str, Any]
) -> list[dict[str, Any]]:
    match_id = _normalize_id(raw_match.get("match_id"), "source match ID")
    if match_id is None:
        raise ValueError("historical series has no usable match_id")
    release_day, release_source, release_reason = _raw_release_day(raw_match, release_days)
    order = _source_series_order(match_id)
    games = raw_match["games"]
    events: list[dict[str, Any]] = []
    for map_index, game in enumerate(games, start=1):
        map_day = _as_day(game["date"], f"historical map {match_id}/{map_index} date")
        for side in ("t1", "t2"):
            team_id = _normalize_id(game.get(f"{side}_id"), f"map team ID {match_id}/{map_index}/{side}")
            if team_id is None:
                raise ValueError(f"Missing map team ID for {match_id}/{map_index}/{side}")
            payload = game.get(f"{side}_players")
            if not isinstance(payload, Mapping):
                raise ValueError(f"Missing player mapping for {match_id}/{map_index}/{side}")
            roster = [
                _player_id_from_payload(player, f"{match_id}/{map_index}/{side}/{role}")
                for role, player in payload.items()
            ]
            known = [player for player in roster if player is not None]
            if len(known) != len(set(known)):
                raise ValueError(f"Duplicate player IDs on map {match_id}/{map_index}/{side}")
            if len(roster) > 5:
                raise ValueError(f"More than five player slots on map {match_id}/{map_index}/{side}")
            roster = [*roster, *([None] * (5 - len(roster)))]
            events.append({
                "team_id": team_id,
                "tournament_name": _canonical_competition_context(raw_match.get("tournament_name")),
                "competition_context": _canonical_competition_context(raw_match.get("tournament_name")),
                "roster": roster,
                "map_index": map_index,
                "source_series_order": order,
                "source_match_id": match_id,
                "original_map_day": map_day.isoformat(),
                "effective_release_day": release_day.isoformat(),
                "release_source": release_source,
                "effective_release_reason": release_reason,
            })
    return events


def build_roster_events(
    raw_matches: Iterable[Mapping[str, Any]], release_days: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build team-scoped last-map roster events from streamed raw matches."""
    if not isinstance(release_days, Mapping):
        raise ValueError("release_days must be an ID-to-calendar-day mapping")
    output: list[dict[str, Any]] = []
    for match in raw_matches:
        if not isinstance(match, Mapping):
            raise ValueError("historical match entries must be objects")
        output.extend(_events_for_raw_match(match, release_days))
    return output


def _effective_decision_times(opportunities: pd.DataFrame) -> pd.Series:
    from src.analysis.ev_approach_data import effective_times

    if "decision_at" not in opportunities and "scenario_decision_at" not in opportunities:
        raise KeyError("opportunities require decision_at or scenario_decision_at")
    return effective_times(opportunities, "decision_at")


def _index_roster_events(
    roster_events: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    by_team: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in roster_events:
        team = _normalize_id(event.get("team_id"), "roster event team ID")
        if team is not None:
            by_team[team].append(event)
    for team_events in by_team.values():
        team_events.sort(key=lambda event: (
            _as_day(event["effective_release_day"], "effective release day"),
            _order_key(event.get("source_series_order", 0)),
            int(event.get("map_index", 0)),
            str(event.get("source_match_id", "")),
        ))
    return by_team


def _metadata_index(metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "golgg_match_id" not in metadata:
        raise ValueError("target metadata requires golgg_match_id")
    allowed = (
        "team1_id", "team2_id", "best_of", "y", "date", "result_day",
        "result_release_day", "result_release_source", "effective_label_release_day",
        "label_release_day", "tournament_name", "competition_context",
    )
    source = metadata.loc[:, [column for column in ("golgg_match_id", *allowed) if column in metadata]].copy()
    source["golgg_match_id"] = source["golgg_match_id"].map(lambda value: _normalize_id(value, "metadata match ID"))
    if source["golgg_match_id"].isna().any() or source["golgg_match_id"].duplicated().any():
        raise ValueError("metadata has missing or duplicate golgg_match_id")
    return {
        str(row["golgg_match_id"]): {column: row[column] for column in source.columns if column != "golgg_match_id"}
        for row in source.to_dict("records")
    }


def _valid_roster(roster: Sequence[str | None]) -> bool:
    return len(roster) == 5 and all(player is not None for player in roster) and len(set(roster)) == 5



def _resolve_research_root(value: str | Path | None) -> Path:
    from src.analysis.ev_approach_data import _resolve_research_root as resolve

    return resolve(value)




def _verify_protocol(path: Path) -> dict[str, Any]:
    actual = _verify_hash(path, PROTOCOL_SHA256, "frozen experiment protocol")
    protocol = _read_json(path)
    if protocol.get("run_id") != "ev_all_approaches_20260927_220902":
        raise ValueError("Frozen experiment protocol run_id mismatch")
    protocol["_verified_sha256"] = actual
    return protocol


def _verify_corrected_bank(research_root: Path, protocol: Mapping[str, Any], raw_path: Path) -> tuple[Path, dict[str, str], dict[str, Any], dict[str, str]]:
    bank_relative = Path(str(protocol["inputs"]["corrected_bank_relative"]))
    bank = research_root / bank_relative
    receipt_path = bank / "completed.json"
    _verify_hash(receipt_path, BANK_RECEIPT_SHA256, "corrected bank completion receipt")
    receipt = _read_json(receipt_path)
    if receipt.get("status") not in {"COMPLETE", "PASS"}:
        raise ValueError("Corrected A0 bank is not complete")
    if receipt.get("source_sha256") != BANK_SOURCE_SHA256:
        raise ValueError("Corrected bank pins an unexpected raw history source")
    raw_hash = _verify_hash(raw_path, BANK_SOURCE_SHA256, "frozen raw history")
    inventory = receipt.get("outputs_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Corrected bank receipt has no output hash inventory")
    verified: dict[str, str] = {"corrected_bank/completed.json": BANK_RECEIPT_SHA256, "raw_history": raw_hash}
    for filename, expected_value in inventory.items():
        expected = expected_value.get("sha256") if isinstance(expected_value, dict) else expected_value
        if not isinstance(expected, str):
            raise ValueError(f"Corrected bank receipt has malformed hash for {filename}")
        path = bank / filename
        verified[f"corrected_bank/{filename}"] = _verify_hash(path, expected, f"corrected bank {filename}")
    if verified.get("corrected_bank/release_days.json") != BANK_RELEASE_SHA256:
        raise ValueError("Corrected bank release map differs from its frozen digest")
    release_path = bank / "release_days.json"
    release_days = _read_json(release_path)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in release_days.items()):
        raise ValueError("Corrected release map must contain string IDs and ISO date strings")
    for match_id, value in release_days.items():
        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid corrected release day for {match_id}: {value!r}") from error
    release_member = receipt.get("release_member")
    release_member_sha = receipt.get("release_member_sha256")
    if not isinstance(release_member, str) or not isinstance(release_member_sha, str):
        raise ValueError("Corrected bank receipt lacks R8 release-map provenance")
    release_archive = research_root / "full-phase-validation-20260910/full-phase-validation-evidence.tar.gz"
    if not release_archive.is_file():
        raise FileNotFoundError(f"Missing archived R8 release source: {release_archive}")
    try:
        with tarfile.open(release_archive, "r:gz") as archive:
            source_member = archive.extractfile(release_member)
            if source_member is None:
                raise FileNotFoundError(f"Missing R8 release member {release_member} in {release_archive}")
            payload = source_member.read()
    except (tarfile.TarError, OSError) as error:
        raise ValueError(f"Unable to verify R8 release provenance in {release_archive}") from error
    member_hash = hashlib.sha256(payload).hexdigest()
    if member_hash != release_member_sha:
        raise ValueError("R8 release member digest differs from corrected bank receipt")
    verified["r8_release_source_member"] = member_hash
    return bank, release_days, receipt, verified


def _verify_manifest_archive(archive_root: Path, required: Mapping[str, str]) -> dict[str, str]:
    manifest_path = archive_root / "manifest.json"
    _verify_hash(manifest_path, FEATURE_MANIFEST_SHA256, "frozen feature source manifest")
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Archived source manifest has no file inventory: {manifest_path}")
    verified: dict[str, str] = {str(manifest_path): _sha256(manifest_path)}
    for relative, expected_pinned in required.items():
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise ValueError(f"Archived manifest does not include {relative}")
        expected = str(entry.get("sha256", ""))
        if expected != expected_pinned:
            raise ValueError(f"Archived manifest identity changed for {relative}")
        path = archive_root / relative
        verified[str(path)] = _verify_hash(path, expected_pinned, f"archived source {relative}")
    return verified


def _verify_imported_module(module: Any, expected_path: Path, expected_hash: str, label: str) -> str:
    origin = getattr(module, "__file__", None)
    if not origin or Path(origin).resolve() != expected_path.resolve():
        raise ImportError(f"Refusing non-archived {label}: expected {expected_path}, got {origin}")
    return _verify_hash(expected_path, expected_hash, f"imported {label}")


def _activate_feature_backend(research_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    archive_root = research_root / FEATURE_ARCHIVE_RELATIVE
    expected = dict(FEATURE_SOURCE_SHA256)
    expected["data/04_feature/temporal-architectures-v1/audit.json"] = FEATURE_AUDIT_SHA256
    verified = _verify_manifest_archive(archive_root, expected)
    audit_path = archive_root / "data/04_feature/temporal-architectures-v1/audit.json"
    audit = _read_json(audit_path)
    vocabulary = audit.get("champion_vocabulary")
    if not isinstance(vocabulary, dict) or vocabulary.get("") != 0:
        raise ValueError("Archived champion vocabulary is absent or incompatible")
    normalized_vocabulary: dict[str, int] = {}
    for champion, code in vocabulary.items():
        if isinstance(code, bool) or not isinstance(code, int) or code < 0 or code > np.iinfo(np.uint16).max:
            raise ValueError("Archived champion vocabulary contains an invalid code")
        normalized_vocabulary[str(champion)] = int(code)
    if len(set(normalized_vocabulary.values())) != len(normalized_vocabulary):
        raise ValueError("Archived champion vocabulary codes are not unique")

    # The uncertainty loader first activates and validates the compatible model
    # namespace. Its receipt also pins the exact nonlinear-history backend.
    from src.models import a0_uncertainty

    backend, backend_hashes = a0_uncertainty._activate_archive_backend(research_root)
    scripts_package = importlib.import_module("scripts")
    archive_scripts = str((archive_root / "code/scripts").resolve())
    paths = list(getattr(scripts_package, "__path__", []))
    scripts_package.__path__ = [archive_scripts, *(path for path in paths if Path(path).resolve() != Path(archive_scripts))]
    importlib.invalidate_caches()
    before = set(sys.modules)
    local_history = importlib.import_module("scripts.local_history_stream")
    temporal = importlib.import_module("scripts.temporal_player_features")
    roster_optional = importlib.import_module("scripts.roster_optional_features")
    prospective = importlib.import_module("scripts.prospective_sports_features")
    imports = {
        "local_history_stream": (local_history, archive_root / "code/scripts/local_history_stream.py", FEATURE_SOURCE_SHA256["code/scripts/local_history_stream.py"]),
        "temporal_player_features": (temporal, archive_root / "code/scripts/temporal_player_features.py", FEATURE_SOURCE_SHA256["code/scripts/temporal_player_features.py"]),
        "roster_optional_features": (roster_optional, archive_root / "code/scripts/roster_optional_features.py", FEATURE_SOURCE_SHA256["code/scripts/roster_optional_features.py"]),
        "prospective_sports_features": (prospective, archive_root / "code/scripts/prospective_sports_features.py", FEATURE_SOURCE_SHA256["code/scripts/prospective_sports_features.py"]),
    }
    for label, (module, path, digest) in imports.items():
        verified[f"import:{label}"] = _verify_imported_module(module, path, digest, label)
    # Verify every newly imported archive script through its manifest entry.
    files = _read_json(archive_root / "manifest.json")["files"]
    for name in set(sys.modules).difference(before):
        if not name.startswith("scripts."):
            continue
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to((archive_root / "code").resolve()):
            continue
        relative = path.relative_to(archive_root).as_posix()
        entry = files.get(relative)
        if not isinstance(entry, dict):
            raise ImportError(f"Archived dependency is not covered by its manifest: {name} -> {path}")
        verified[str(path)] = _verify_hash(path, str(entry.get("sha256", "")), f"archived dependency {name}")

    # Several structural helpers live in src.* rather than scripts.*. They may
    # resolve to the workspace only when byte-identical to the archived source.
    for relative in ("code/src/models/symmetric_series.py", "code/src/utils/golgg_schema.py"):
        archived = archive_root / relative
        manifest_record = files.get(relative)
        if not isinstance(manifest_record, dict):
            raise ValueError(f"Archived source manifest omits {relative}")
        expected_hash = str(manifest_record.get("sha256", ""))
        _verify_hash(archived, expected_hash, f"archived helper {relative}")
        module_name = "src.models.symmetric_series" if "symmetric_series" in relative else "src.utils.golgg_schema"
        module = importlib.import_module(module_name)
        origin = Path(module.__file__).resolve()
        if origin != archived.resolve():
            _verify_hash(origin, expected_hash, f"workspace-equivalent helper {module_name}")
        verified[f"helper:{module_name}"] = expected_hash
    return {
        "iter_matches": local_history.iter_matches,
        "collect_history": temporal.collect_history,
        "TemporalState": temporal.TemporalState,
        "RosterFeatureState": roster_optional.RosterFeatureState,
        "champion_vocabulary": normalized_vocabulary,
        "model_backend": backend,
        "model_backend_hashes": backend_hashes,
    }, verified


def _verify_native_sources(research_root: Path) -> dict[str, str]:
    archive_root = research_root / CAUSAL_ARCHIVE_RELATIVE
    verified: dict[str, str] = {}
    for relative, expected in NATIVE_SOURCE_SHA256.items():
        verified[str(archive_root / relative)] = _verify_hash(archive_root / relative, expected, f"native source {relative}")
    return verified


def _activate_native_predictor(research_root: Path) -> Any:
    code_root = (research_root / CAUSAL_ARCHIVE_RELATIVE / "code").resolve()
    expected = {
        Path(relative).stem: code_root / Path(relative).name
        for relative in NATIVE_SOURCE_SHA256
    }
    code_string = str(code_root)
    if code_string not in sys.path:
        sys.path.insert(0, code_string)
    importlib.invalidate_caches()
    for name, path in expected.items():
        loaded = sys.modules.get(name)
        if loaded is not None:
            origin = getattr(loaded, "__file__", None)
            if not origin or Path(origin).resolve() != path.resolve():
                raise ImportError(f"Refusing shadowed native A0 module {name}: {origin}")
    modules: dict[str, Any] = {}
    for name, path in expected.items():
        module = importlib.import_module(name)
        _verify_imported_module(
            module,
            path,
            NATIVE_SOURCE_SHA256[f"code/{name}.py"],
            name,
        )
        modules[name] = module
    predictor = modules["causal_predictor"]
    if not issubclass(predictor.CausalA0Predictor, modules["a0_predictor"].A0Predictor):
        raise TypeError("Archived CausalA0Predictor does not inherit the archived A0 inference contract")
    return predictor



def _corrected_series_day(item: Any, release_days: Mapping[str, Any]) -> date:
    if not getattr(item, "games", None):
        raise ValueError(f"Historical series {getattr(item, 'identifier', None)!r} has no games")
    source_latest = max(game.day for game in item.games)
    identifier = _normalize_id(item.identifier, "historical series ID")
    if identifier is None:
        raise ValueError("Historical series has no usable identifier")
    release_value = release_days.get(identifier, source_latest.isoformat())
    try:
        return max(source_latest, date.fromisoformat(str(release_value)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid release override for series {item.identifier}: {release_value!r}") from error


def _stream_history(
    raw_history_path: Path,
    release_days: Mapping[str, Any],
    backend: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, date]]:
    events: list[dict[str, Any]] = []
    stream = backend["iter_matches"](raw_history_path)

    def capture() -> Iterator[Mapping[str, Any]]:
        for match in stream:
            events.extend(_events_for_raw_match(match, release_days))
            yield match

    # The archived collect_history owns the JSON traversal. Only compact
    # normalized series/maps/features and team roster events survive each yield.
    series, metadata, extra, audit = backend["collect_history"](capture())
    effective: dict[str, date] = {}
    delayed: list[dict[str, Any]] = []
    corrected: list[Any] = []
    for item in series:
        day = _corrected_series_day(item, release_days)
        identifier = _normalize_id(item.identifier, "historical series ID")
        if identifier is None:
            raise ValueError("Historical series has no usable identifier")
        effective[identifier] = day
        original = max(game.day for game in item.games)
        if day != original:
            delayed.append({"source_match_id": identifier, "original_source_latest_day": original.isoformat(), "effective_release_day": day.isoformat()})
            games = list(item.games)
            games[-1] = replace(games[-1], day=day)
            item = replace(item, games=games)
        corrected.append(item)
    corrected.sort(key=lambda item: (item.games[-1].day, item.order))
    event_ids = {str(event["source_match_id"]) for event in events}
    series_ids = {str(item.identifier) for item in corrected}
    if event_ids != series_ids:
        missing_events = sorted(series_ids - event_ids)[:5]
        missing_series = sorted(event_ids - series_ids)[:5]
        raise ValueError(f"Roster/history series identity mismatch: missing_events={missing_events}, missing_series={missing_series}")
    for event in events:
        sid = str(event["source_match_id"])
        if _as_day(event["effective_release_day"], "effective release day") != effective[sid]:
            raise ValueError(f"Roster event release precedence differs from corrected series {sid}")
    release_audit = {
        "series_count": len(corrected),
        "map_count": int(sum(len(item.games) for item in corrected)),
        "roster_event_count": len(events),
        "delayed_release_count": len(delayed),
        "delayed_release_examples": delayed[:100],
        "release_map_override_count": sum(event["release_source"] == "corrected_release_map_override" for event in events),
        "source_game_day_fallback_event_count": sum(event["release_source"] == "latest_source_game_day_fallback" for event in events),
        "effective_release_precedence": "max(max(game.day), date.fromisoformat(release_days.get(series_id, max(game.day).isoformat())))",
        "order": "effective_release_day then archived numeric-ID/lexical series.order; all maps in a series released together",
    }
    return corrected, metadata, extra, events, audit, release_audit, effective


def _read_bank_validation_data(bank: Path, bank_receipt: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    metadata_path = bank / "metadata.parquet"
    original_path = bank / "original_metadata.jsonl"
    metadata = pd.read_parquet(metadata_path)
    original = pd.read_json(
        original_path,
        lines=True,
        convert_dates=False,
        dtype={"golgg_match_id": str, "team1_id": str, "team2_id": str},
    )
    ids = metadata["golgg_match_id"].astype(str).tolist()
    if ids != original["golgg_match_id"].astype(str).tolist():
        raise ValueError("Corrected-bank target order differs from original validation metadata")
    for field in ("date", "team1_id", "team2_id", "best_of"):
        if field in metadata and field in original:
            if not metadata[field].astype(str).reset_index(drop=True).equals(original[field].astype(str).reset_index(drop=True)):
                raise ValueError(f"Corrected-bank and original target metadata disagree in {field}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(bank / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    temporal = {
        key: np.load(bank / f"{key}.npy", mmap_mode="r")
        for key in ("history", "mask", "champions")
    }
    if any(len(value) != len(metadata) for value in (*arrays.values(), *temporal.values())):
        raise ValueError("Corrected-bank oracle arrays are not aligned to target metadata")
    return original, arrays, temporal


def _select_oracle_controls(original: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"golgg_match_id", "date", "team1_id", "team2_id", "best_of", "roster_a", "roster_b"}
    missing = sorted(required.difference(original.columns))
    if missing:
        raise ValueError(f"Original bank metadata lacks oracle identity fields: {missing}")
    work = original.copy()
    work["_oracle_index"] = np.arange(len(work), dtype=np.int64)
    work["_year"] = pd.to_datetime(work["date"], errors="raise").dt.year
    work["_id"] = work["golgg_match_id"].astype(str)
    work = work.sort_values(["_year", "date", "_id"], kind="stable")
    chosen: list[dict[str, Any]] = []
    # Fixed, date-ordered controls across available years and best-of formats;
    # actual target rosters are confined to this compatibility oracle.
    for (year, best_of), group in work.groupby(["_year", "best_of"], sort=True):
        if int(year) not in MODEL_YEARS:
            continue
        row = group.iloc[0]
        roster_a = [_normalize_id(value, "oracle roster A ID") for value in row["roster_a"]]
        roster_b = [_normalize_id(value, "oracle roster B ID") for value in row["roster_b"]]
        if not _valid_roster(roster_a) or not _valid_roster(roster_b):
            raise ValueError(f"Invalid archived actual roster in bank oracle row {row['_id']}")
        chosen.append({
            "row_id": f"bank_oracle:{row['_id']}",
            "golgg_match_id": row["_id"],
            "cutoff_day": _as_day(row["date"], "oracle cutoff date"),
            "team1_id": _normalize_id(row["team1_id"], "oracle team A"),
            "team2_id": _normalize_id(row["team2_id"], "oracle team B"),
            "best_of": int(best_of),
            "roster_a": roster_a,
            "roster_b": roster_b,
            "bank_index": int(row["_oracle_index"]),
        })
    if not chosen:
        raise ValueError("No deterministic canonical-date rows are available for bank compatibility controls")
    return chosen


def _feature_values(
    state: Any,
    target: Mapping[str, Any],
    query_rosters: tuple[Sequence[str], Sequence[str]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    team_a = str(target["team1_id"])
    team_b = str(target["team2_id"])
    cutoff = _as_day(target["cutoff_day"], "snapshot cutoff day")
    best_of_value = target.get("best_of")
    if isinstance(best_of_value, bool) or best_of_value is None or pd.isna(best_of_value):
        raise ValueError("best_of is missing from an otherwise feature-available query")
    best_of = int(best_of_value)
    if best_of not in (1, 3, 5):
        raise ValueError(f"Unsupported best_of for A0 inference: {best_of}")
    if len(query_rosters) != 2:
        raise ValueError("A feature query requires exactly two player rosters")
    canonical_rosters: list[list[str]] = []
    for side, roster in zip(("A", "B"), query_rosters):
        if not isinstance(roster, (list, tuple)) or len(roster) != 5:
            raise ValueError(f"Feature query requires five identified players on side {side}")
        normalized = [_normalize_id(player, f"feature query player ID on side {side}") for player in roster]
        if any(player is None for player in normalized) or len(set(normalized)) != 5:
            raise ValueError(f"Feature query requires five distinct identified players on side {side}")
        canonical_rosters.append(sorted(normalized))
    tensor_order_a, tensor_order_b = canonical_rosters
    raw_pair, pair_proof = state.base.pair(
        team_a, tensor_order_a, team_b, tensor_order_b, cutoff, best_of
    )
    history_rows = [
        [state.player(player, cutoff) for player in roster]
        for roster in canonical_rosters
    ]
    raw = {key: np.asarray(value, dtype=np.float32) for key, value in raw_pair.items()}
    hist = {
        key: np.asarray([[slot[key] for slot in side] for side in history_rows], dtype=dtype)
        for key, dtype in (("history", np.float16), ("mask", np.uint8), ("champions", np.uint16))
    }
    proof = {
        "teams": pair_proof,
        "feature_source_max_day": state.base.base.last_day.isoformat() if state.base.base.last_day else None,
        "cutoff_day": cutoff.isoformat(),
        "query_roster_a": list(tensor_order_a),
        "query_roster_b": list(tensor_order_b),
        "tensor_player_order_a": list(tensor_order_a),
        "tensor_player_order_b": list(tensor_order_b),
        "raw_tensor_player_order_a": list(tensor_order_a),
        "raw_tensor_player_order_b": list(tensor_order_b),
        "temporal_tensor_player_order_a": list(tensor_order_a),
        "temporal_tensor_player_order_b": list(tensor_order_b),
    }
    return raw, hist, proof


def _logical_state_marker(state: Any) -> tuple[Any, Any, int, int]:
    base = state.base.base
    return (
        base.last_day,
        base.last_order,
        len(state.histories),
        len(base.manager.systems["gl"].player_ratings),
    )



def _oracle_compare(
    state: Any,
    control: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    temporal: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    before = _logical_state_marker(state)
    raw, hist, proof = _feature_values(
        state,
        {**control, "cutoff_day": control["cutoff_day"].isoformat()},
        (list(control["roster_a"]), list(control["roster_b"])),
    )
    if before != _logical_state_marker(state):
        raise ValueError("Bank compatibility query mutated logical chronological history")
    errors: dict[str, float] = {}
    for key, generated in raw.items():
        if key not in arrays:
            raise ValueError(f"Corrected-bank raw oracle omits {key}")
        expected = np.asarray(arrays[key][int(control["bank_index"])], dtype=np.float32)
        value = np.asarray(generated, dtype=np.float32)
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.array_equal(value, expected):
            mismatch = float(np.max(np.abs(value.astype(np.float64) - expected.astype(np.float64)))) if value.shape == expected.shape else None
            raise ValueError(f"Corrected-bank raw compatibility mismatch for {key}/{control['golgg_match_id']}: {mismatch}")
        errors[f"{key}_max_abs_error"] = 0.0
    for key, generated in hist.items():
        expected_dtype = {"history": np.float16, "mask": np.uint8, "champions": np.uint16}[key]
        expected = np.asarray(temporal[key][int(control["bank_index"])], dtype=expected_dtype)
        value = np.asarray(generated, dtype=expected_dtype)
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.array_equal(value, expected):
            mismatch = float(np.max(np.abs(value.astype(np.float64) - expected.astype(np.float64)))) if value.shape == expected.shape else None
            raise ValueError(f"Corrected-bank temporal compatibility mismatch for {key}/{control['golgg_match_id']}: {mismatch}")
        errors[f"{key}_max_abs_error"] = 0.0
    return {
        "row_id": control["row_id"],
        "golgg_match_id": control["golgg_match_id"],
        "cutoff_day": control["cutoff_day"].isoformat(),
        "bank_index": int(control["bank_index"]),
        "expected_dtypes": {"raw": "float32", "history": "float16", "mask": "uint8", "champions": "uint16"},
        "all_feature_keys_exact": True,
        "max_abs_error": errors,
        "feature_source_max_day": proof["feature_source_max_day"],
    }


def _initialize_arrays(size: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = {
        key: np.full((size, *shape), np.nan, dtype=np.float32)
        for key, shape in _RAW_SHAPES.items()
    }
    hist: dict[str, np.ndarray] = {
        "history": np.full((size, *_TEMPORAL_SHAPES["history"]), np.nan, dtype=np.float16),
        "mask": np.zeros((size, *_TEMPORAL_SHAPES["mask"]), dtype=np.uint8),
        "champions": np.zeros((size, *_TEMPORAL_SHAPES["champions"]), dtype=np.uint16),
    }
    return raw, hist


def _reason_for_target(target: Mapping[str, Any]) -> str | None:
    if target.get("forecast_absence") == "target_already_released_before_cutoff":
        return "target_already_released_before_cutoff"
    if target.get("cutoff_day") is None or target.get("model_year") is None:
        return "unknown_decision_cutoff"
    if target.get("team1_id") is None or target.get("team2_id") is None:
        return "missing_team_identity"
    if target.get("team1_id") == target.get("team2_id"):
        return "identical_team_identity"
    overlap = target.get("overlapping_prior_player_ids") or []
    if overlap:
        return "overlapping_prior_rosters"
    roster_status = target.get("roster_status")
    if roster_status not in {"prior_observed_scenario", "announced"}:
        return target.get("roster_missing_reason") or "unresolved_roster_identity"
    if not _valid_roster(target.get("roster_a", ())) or not _valid_roster(target.get("roster_b", ())):
        return "incomplete_or_unknown_roster_identity"
    best_of = target.get("best_of")
    if best_of is None or pd.isna(best_of):
        return "missing_best_of"
    if int(best_of) not in (1, 3, 5):
        return "unsupported_best_of"
    return None


def _replay_targets(
    targets: pd.DataFrame,
    series: list[Any],
    history_metadata: dict[str, Any],
    extra: dict[str, Any],
    roster_events: list[dict[str, Any]],
    backend: Mapping[str, Any],
    validation_controls: Sequence[Mapping[str, Any]] = (),
    bank_arrays: Mapping[str, np.ndarray] | None = None,
    bank_temporal: Mapping[str, np.ndarray] | None = None,
    target_native_release_days: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    state = backend["TemporalState"]()
    state.vocabulary = dict(backend["champion_vocabulary"])
    if state.vocabulary != backend["champion_vocabulary"]:
        raise ValueError("Unable to freeze archived champion vocabulary")
    release_by_id = {str(event["source_match_id"]): _as_day(event["effective_release_day"], "event release") for event in roster_events}
    if len(release_by_id) != len({str(item.identifier) for item in series}):
        raise ValueError("Roster event source IDs are not a unique series inventory")
    native_releases = _target_native_release_inventory(roster_events, target_native_release_days)

    raw_arrays, hist_arrays = _initialize_arrays(len(targets))
    rows = [row.to_dict() for _, row in targets.iterrows()]
    by_day: dict[date, list[tuple[str, int, Mapping[str, Any]]]] = defaultdict(list)
    for index, target in enumerate(rows):
        cutoff_value = target.get("cutoff_day")
        cutoff = None if cutoff_value is None else _as_day(cutoff_value, "target cutoff day")
        identifier = _normalize_id(target.get("golgg_match_id"), "target match ID")
        target.update(_target_forecast_eligibility(target, cutoff, native_releases.get(identifier)))
        if cutoff is None:
            continue
        by_day[cutoff].append(("forecast", index, target))
    for control_index, control in enumerate(validation_controls):
        by_day[_as_day(control["cutoff_day"], "oracle cutoff day")].append(("oracle", control_index, control))
    query_days = sorted(by_day)
    position = 0
    consumed = 0
    oracle_results: list[dict[str, Any]] = []
    available_count = 0
    unavailable_counts = defaultdict(int)
    for cutoff in query_days:
        while position < len(series) and series[position].games[-1].day < cutoff:
            item = series[position]
            identifier = str(item.identifier)
            if release_by_id.get(identifier) != item.games[-1].day:
                raise ValueError(f"History transition order is inconsistent with corrected release for {identifier}")
            for game_extra in extra.get(item.identifier, []):
                for _, _, champion in game_extra.get("players", {}).values():
                    if champion and champion not in state.vocabulary:
                        raise ValueError(f"Historical champion is missing from frozen vocabulary: {champion}")
            state.consume(item, history_metadata.pop(item.identifier), extra.pop(item.identifier))
            position += 1
            consumed += 1
        if state.base.base.last_day is not None and state.base.base.last_day >= cutoff:
            raise ValueError("Temporal state consumed same-day or future history")
        for kind, index, target in sorted(by_day[cutoff], key=lambda item: (0 if item[0] == "oracle" else 1, item[1])):
            if kind == "oracle":
                if bank_arrays is None or bank_temporal is None:
                    raise ValueError("Bank oracle controls require archived comparison arrays")
                oracle_results.append(_oracle_compare(state, target, bank_arrays, bank_temporal))
                continue
            source_max = state.base.base.last_day.isoformat() if state.base.base.last_day else None
            rows[index]["feature_source_max_day"] = source_max
            reason = _reason_for_target(target)
            if reason:
                rows[index]["feature_available"] = False
                rows[index]["missing_reason"] = (
                    reason if reason == "target_already_released_before_cutoff"
                    else target.get("roster_missing_reason") or reason
                )
                rows[index]["team_feature_provenance"] = {
                    "status": "unavailable",
                    "roster_status": target.get("roster_status"),
                    "roster_source_a": target.get("roster_source_a"),
                    "roster_source_b": target.get("roster_source_b"),
                    "overlapping_prior_player_ids": target.get("overlapping_prior_player_ids", []),
                }
                continue
            before = _logical_state_marker(state)
            feature_target = dict(target)
            feature_target["cutoff_day"] = cutoff.isoformat()
            raw, hist, proof = _feature_values(
                state, feature_target, (target["roster_a"], target["roster_b"])
            )
            if before != _logical_state_marker(state):
                raise ValueError("As-of feature query mutated logical chronological history")
            source_max = proof["feature_source_max_day"]
            if source_max is not None and _as_day(source_max, "feature source max day") >= cutoff:
                raise ValueError("As-of feature source reaches the cutoff day")
            for key, value in raw.items():
                expected_shape = _RAW_SHAPES[key]
                array = np.asarray(value, dtype=np.float32)
                if array.shape != expected_shape or not np.isfinite(array).all():
                    raise ValueError(f"Invalid reconstructed {key} shape or values for {target.get('row_id')}")
                raw_arrays[key][index] = array
            for key, value in hist.items():
                array = np.asarray(value, dtype=hist_arrays[key].dtype)
                if array.shape != _TEMPORAL_SHAPES[key]:
                    raise ValueError(f"Invalid reconstructed {key} shape for {target.get('row_id')}")
                hist_arrays[key][index] = array
            rows[index]["feature_available"] = True
            rows[index]["missing_reason"] = None
            rows[index]["feature_source_max_day"] = source_max
            rows[index]["team_feature_provenance"] = proof["teams"]
            rows[index]["snapshot_query_roster_a"] = proof["query_roster_a"]
            rows[index]["snapshot_query_roster_b"] = proof["query_roster_b"]
            rows[index]["tensor_player_order_a"] = proof["tensor_player_order_a"]
            rows[index]["tensor_player_order_b"] = proof["tensor_player_order_b"]
            rows[index]["raw_tensor_player_order_a"] = proof["raw_tensor_player_order_a"]
            rows[index]["raw_tensor_player_order_b"] = proof["raw_tensor_player_order_b"]
            rows[index]["temporal_tensor_player_order_a"] = proof["temporal_tensor_player_order_a"]
            rows[index]["temporal_tensor_player_order_b"] = proof["temporal_tensor_player_order_b"]
            available_count += 1
    unavailable_counts.clear()
    for target in rows:
        target.setdefault("feature_available", False)
        if not target["feature_available"]:
            target.setdefault("missing_reason", _reason_for_target(target) or "snapshot_unavailable")
            unavailable_counts[target["missing_reason"]] += 1
    leftovers = len(history_metadata)
    roster_scenario_counts = {
        scenario: sum(target.get("roster_status") == scenario for target in rows)
        for scenario in ("prior_observed_scenario", "announced", "unresolved")
    }
    overlapping_count = sum(bool(target.get("overlapping_prior_player_ids")) for target in rows)
    receipt = {
        "query_days": len(query_days),
        "forecast_rows": len(rows),
        "feature_available_rows": available_count,
        "feature_unavailable_rows": len(rows) - available_count,
        "unavailable_reasons": dict(sorted(unavailable_counts.items())),
        "history_series_total": len(series),
        "history_series_consumed": consumed,
        "history_series_not_yet_released_at_last_cutoff": leftovers,
        "final_state_release_day": state.base.base.last_day.isoformat() if state.base.base.last_day else None,
        "roster_scenario_counts": roster_scenario_counts,
        "overlapping_roster_rows": overlapping_count,
        "oracle_control_count": len(oracle_results),
        "logical_history_read_only_queries": True,
    }
    if len(oracle_results) != len(validation_controls):
        raise ValueError("Not all deterministic corrected-bank oracle controls were queried")
    if any(not item["all_feature_keys_exact"] for item in oracle_results):
        raise ValueError("Corrected bank compatibility oracle did not pass")
    receipt["oracle_controls"] = oracle_results
    return raw_arrays, hist_arrays, rows, receipt


def _build_replay_arrays(
    opportunities: pd.DataFrame,
    metadata: pd.DataFrame,
    research_root: Path,
    raw_history_path: Path,
    release_days: Mapping[str, Any],
    *,
    validation_controls: Sequence[Mapping[str, Any]] = (),
    bank_arrays: Mapping[str, np.ndarray] | None = None,
    bank_temporal: Mapping[str, np.ndarray] | None = None,
    feature_state_factory: Any = None,
    roster_evidence: Any = None,
) -> _ReplayArrays:
    backend, feature_hashes = _activate_feature_backend(research_root)
    if feature_state_factory is not None:
        backend = dict(backend)
        backend["TemporalState"] = feature_state_factory
    series, history_metadata, extra, roster_events, history_audit, release_audit, label_release_days = _stream_history(
        raw_history_path, release_days, backend
    )
    target_metadata = metadata.copy()
    supplied_contexts = [
        _target_competition_context(row, str(row.get("golgg_match_id", "")))
        for row in target_metadata.to_dict("records")
    ]
    history_tournaments = {
        str(match_id): _canonical_competition_context(value.get("tournament_name"))
        for match_id, value in history_metadata.items()
        if isinstance(value, Mapping)
    }
    context_sources: dict[str, str] = {}
    if "golgg_match_id" in target_metadata:
        normalized_ids = target_metadata["golgg_match_id"].map(
            lambda value: _normalize_id(value, "metadata match ID")
        )
        contexts = []
        for supplied, match_id in zip(supplied_contexts, normalized_ids):
            archived = history_tournaments.get(match_id or "")
            if archived and supplied and archived != supplied:
                raise ValueError(f"Archived and supplied competition contexts disagree for {match_id}")
            contexts.append(archived or supplied)
            if archived:
                context_sources[match_id] = "archived_target_context_scenario"
        target_metadata["tournament_name"] = contexts
        target_metadata["competition_context"] = target_metadata["tournament_name"]
    evidence = _roster_evidence_input(roster_evidence)
    targets = build_asof_targets(
        opportunities, target_metadata, roster_events, roster_evidence=evidence,
        context_provenance=context_sources,
        target_native_release_days=label_release_days,
    )
    series.sort(key=lambda item: (_corrected_series_day(item, release_days), item.order))
    raw, hist, records, replay_receipt = _replay_targets(
        targets, series, history_metadata, extra, roster_events, backend,
        validation_controls=validation_controls, bank_arrays=bank_arrays,
        bank_temporal=bank_temporal,
        target_native_release_days=label_release_days,
    )
    targets = pd.DataFrame(records)
    release_audit.update({
        "history_audit": history_audit,
        "roster_availability": {
            scenario: int(targets["roster_status"].eq(scenario).sum())
            for scenario in ("prior_observed_scenario", "announced", "unresolved")
        },
        "overlapping_roster_rows": int(targets["overlapping_prior_player_ids"].map(bool).sum()),
        "roster_evidence_input_hashes": evidence["input_hashes"],
    })
    return _ReplayArrays(
        targets, raw, hist, replay_receipt, history_audit, release_audit,
        feature_hashes, label_release_days, evidence["input_hashes"],
    )

def reconstruct_asof_snapshots(
    opportunities: pd.DataFrame,
    metadata: pd.DataFrame,
    raw_history: str | Path | Iterable[Mapping[str, Any]],
    release_days: Mapping[str, Any],
    feature_state_factory: Any = None,
    *,
    roster_evidence: Any = None,
) -> Iterator[AsOfSnapshot]:
    """Replay released history and yield one immutable row-aligned snapshot."""
    if isinstance(raw_history, (str, Path)):
        raw_path = Path(raw_history).expanduser().resolve()
        from src.analysis.ev_approach_data import _resolve_research_root

        research_root = _resolve_research_root(None)
    else:
        # Iterable raw sources are useful for bounded pure API consumers. The
        # archived collector still owns normalization; this convenience path
        # uses a one-shot stream without ever materializing the raw history.
        from src.analysis.ev_approach_data import _resolve_research_root

        research_root = _resolve_research_root(None)
        raw_path = None
    if raw_path is None:
        backend, _ = _activate_feature_backend(research_root)
        event_list: list[dict[str, Any]] = []
        def capture_iter() -> Iterator[Mapping[str, Any]]:
            for match in raw_history:  # type: ignore[union-attr]
                event_list.extend(_events_for_raw_match(match, release_days))
                yield match
        series, history_metadata, extra, history_audit = backend["collect_history"](capture_iter())
        effective = {str(item.identifier): _corrected_series_day(item, release_days) for item in series}
        corrected = []
        for item in series:
            release = effective[str(item.identifier)]
            games = list(item.games)
            games[-1] = replace(games[-1], day=release)
            corrected.append(replace(item, games=games))
        corrected.sort(key=lambda item: (item.games[-1].day, item.order))
        events = event_list
        targets = build_asof_targets(
            opportunities, metadata, events, roster_evidence=roster_evidence,
            target_native_release_days=effective,
        )
        raw, hist, records, _ = _replay_targets(
            targets, corrected, history_metadata, extra, events, backend,
            target_native_release_days=effective,
        )
        del history_audit
        for index, target in enumerate(records):
            yield AsOfSnapshot(
                target, {key: value[index] for key, value in raw.items()},
                {key: value[index] for key, value in hist.items()},
                bool(target.get("feature_available")), target.get("missing_reason"),
            )
        return

    protocol_path = PROJECT_ROOT / "data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json"
    protocol = _verify_protocol(protocol_path)
    _, frozen_release_days, _, _ = _verify_corrected_bank(research_root, protocol, raw_path)
    supplied_release_days = {
        str(key): _as_day(value, f"supplied release day for {key}").isoformat()
        for key, value in release_days.items()
    }
    if supplied_release_days != frozen_release_days:
        raise ValueError("As-of path replay release days differ from the verified corrected-bank map")
    result = _build_replay_arrays(
        opportunities, metadata, research_root, raw_path, frozen_release_days,
        feature_state_factory=feature_state_factory,
        roster_evidence=roster_evidence,
    )
    for index, target in result.targets.iterrows():
        yield AsOfSnapshot(
            target.to_dict(), {key: value[index] for key, value in result.raw.items()},
            {key: value[index] for key, value in result.hist.items()},
            bool(target.get("feature_available")), target.get("missing_reason"),
        )


def _check_k5_bundle(
    model_dir: Path,
    research_root: Path,
    digest_cache: dict[Path, str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not model_dir.is_dir():
        return None, {"status": "unavailable", "missing_reason": "missing_annual_model_folder", "model_dir": str(model_dir)}
    receipt_path = model_dir / "receipt.json"
    model_path = model_dir / "inference_model.joblib"
    absent = [str(path.name) for path in (receipt_path, model_path) if not path.is_file()]
    if absent:
        return None, {"status": "unavailable", "missing_reason": "missing_annual_model_artifact:" + ",".join(absent), "model_dir": str(model_dir)}
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "COMPLETE":
        raise ValueError(f"Saved K5 receipt is not COMPLETE: {receipt_path}")
    year = int(model_dir.name)
    variant = model_dir.parent.name
    if int(receipt.get("year", -1)) != year or receipt.get("variant") != variant:
        raise ValueError(f"Saved K5 receipt identity mismatch: {model_dir}")
    train_ids = receipt.get("train_ids")
    calibration_ids = receipt.get("cal_ids")
    max_train_release = receipt.get("max_train_release")
    max_cal_release = receipt.get("max_cal_release")
    if (
        not isinstance(train_ids, list) or not train_ids
        or not isinstance(calibration_ids, list) or not calibration_ids
        or not max_train_release or not max_cal_release
    ):
        raise ValueError(f"Saved K5 receipt omits training/calibration identities or release maxima: {receipt_path}")
    if (
        _as_day(max_train_release, "K5 max train release") >= date(year - 1, 1, 1)
        or _as_day(max_cal_release, "K5 max CAL release") >= date(year, 1, 1)
    ):
        raise ValueError(f"Saved K5 fit includes labels released outside its annual schedule: {receipt_path}")
    input_hashes = receipt.get("input_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        raise ValueError(f"Saved K5 receipt omits pinned training inputs: {receipt_path}")
    verified_inputs: dict[str, str] = {}
    for stored_path, expected in input_hashes.items():
        marker = "/research_inputs/"
        if marker not in str(stored_path):
            raise ValueError(f"Unrecognized saved K5 training input path: {stored_path}")
        relative = str(stored_path).split(marker, 1)[1]
        source = (research_root / relative).resolve()
        digest = digest_cache.get(source)
        if digest is None:
            digest = _verify_hash(source, str(expected), f"K5 training input {relative}")
            digest_cache[source] = digest
        elif digest != expected:
            raise ValueError(f"K5 training input hash mismatch: {source}")
        verified_inputs[relative] = digest
    output_hashes = receipt.get("outputs_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError(f"Saved K5 receipt omits output hash inventory: {receipt_path}")
    expected_model_hash = output_hashes.get("inference_model.joblib")
    if not expected_model_hash:
        raise ValueError(f"Saved K5 receipt does not pin inference_model.joblib: {receipt_path}")
    verified_outputs = {
        "inference_model.joblib": _verify_hash(model_path, str(expected_model_hash), f"K5 inference bundle {year}/{variant}")
    }
    for name, expected in output_hashes.items():
        if name in {"inference_model.joblib", "prediction_parquet"}:
            continue
        path = model_dir / str(name)
        if not path.is_file():
            return None, {
                "status": "unavailable", "missing_reason": f"missing_annual_model_artifact:{name}",
                "model_dir": str(model_dir), "receipt_sha256": _sha256(receipt_path),
                "verified_input_count": len(verified_inputs),
            }
        verified_outputs[str(name)] = _verify_hash(path, str(expected), f"K5 checkpoint {year}/{variant}/{name}")
    for member in receipt.get("members", []):
        if not isinstance(member, Mapping):
            continue
        checkpoints = member.get("checkpoints", {})
        if not isinstance(checkpoints, Mapping):
            continue
        for old_path, expected in checkpoints.items():
            name = Path(str(old_path)).name
            path = model_dir / name
            if not path.is_file():
                return None, {
                    "status": "unavailable", "missing_reason": f"missing_annual_model_artifact:{name}",
                    "model_dir": str(model_dir), "receipt_sha256": _sha256(receipt_path),
                    "verified_input_count": len(verified_inputs),
                }
            actual = _verify_hash(path, str(expected), f"K5 member checkpoint {year}/{variant}/{name}")
            verified_outputs[name] = actual
    summary = {
        "status": "verified",
        "year": year,
        "variant": variant,
        "model_dir": str(model_dir),
        "receipt_sha256": _sha256(receipt_path),
        "inference_model_sha256": verified_outputs["inference_model.joblib"],
        "verified_input_count": len(verified_inputs),
        "verified_output_count": len(verified_outputs),
        "input_hashes": verified_inputs,
        "output_hashes": verified_outputs,
        "max_train_release": max_train_release,
        "max_cal_release": max_cal_release,
        "train_ids": train_ids,
        "calibration_ids": calibration_ids,
        "train_ids_count": len(train_ids),
        "cal_ids_count": len(calibration_ids),
    }
    return receipt, summary


def _native_model_folder(research_root: Path, year: int) -> Path:
    return research_root / CAUSAL_ARCHIVE_RELATIVE / "data/06_models/causal_a0" / str(year)


def _max_release_for_ids(ids: Sequence[Any], release_days: Mapping[str, date], label: str) -> str:
    normalized = [_normalize_id(value, f"{label} match ID") for value in ids]
    if not normalized or any(value is None for value in normalized):
        raise ValueError(f"{label} contains missing match IDs")
    missing = sorted(set(normalized).difference(release_days))
    if missing:
        raise ValueError(f"{label} IDs are absent from the corrected history: {missing[:5]}")
    return max(release_days[value] for value in normalized if value is not None).isoformat()


def _empty_features(raw: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {key: np.zeros_like(value) for key, value in raw.items()},
        {key: np.zeros_like(value) for key, value in hist.items()},
    )


def _swap_features(raw: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {key: np.asarray(value)[:, ::-1].copy() for key, value in raw.items()},
        {key: np.asarray(value)[:, ::-1].copy() for key, value in hist.items()},
    )


def _input_digest(raw: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in {**raw, **hist}.items()
    }


def _batch_features(
    raw_arrays: Mapping[str, np.ndarray],
    hist_arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        {key: np.asarray(value[indices]).copy() for key, value in raw_arrays.items()},
        {key: np.asarray(value[indices]).copy() for key, value in hist_arrays.items()},
    )


def _native_smoke(model: Any, raw: dict[str, np.ndarray], hist: dict[str, np.ndarray], best_of: np.ndarray) -> dict[str, Any]:
    before = _input_digest(raw, hist)
    normal = np.asarray(model.predict(raw, hist, best_of), dtype=float)
    swapped_raw, swapped_hist = _swap_features(raw, hist)
    swapped = np.asarray(model.predict(swapped_raw, swapped_hist, best_of), dtype=float)
    cold_raw, cold_hist = _empty_features(raw, hist)
    cold = np.asarray(model.predict(cold_raw, cold_hist, best_of), dtype=float)
    after = _input_digest(raw, hist)
    if before != after:
        raise ValueError("Native A0 inference mutated its input snapshots")
    if normal.shape != (len(best_of), 3) or swapped.shape != normal.shape or cold.shape != normal.shape:
        raise ValueError("Native A0 smoke returned an unexpected optional-mode shape")
    for name, values in (("normal", normal), ("swapped", swapped), ("empty-history", cold)):
        if not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError(f"Native A0 smoke returned invalid {name} probabilities")
    swap_error = float(np.max(np.abs(normal + swapped - 1.0)))
    cold_error = float(np.max(np.abs(cold - 0.5)))
    if swap_error > 2e-6 or cold_error > 2e-6:
        raise ValueError(f"Native A0 symmetry/cold smoke failed: swap={swap_error}, cold={cold_error}")
    return {
        "rows": int(len(best_of)),
        "swap_error_by_mode": [float(np.max(np.abs(normal[:, mode] + swapped[:, mode] - 1.0))) for mode in range(3)],
        "empty_history_probability_error_by_mode": [float(np.max(np.abs(cold[:, mode] - 0.5))) for mode in range(3)],
        "optional_modes": ["full", "no_w20", "no_organization"],
        "input_mutation": False,
    }


def _k5_smoke(model: Any, raw: dict[str, np.ndarray], hist: dict[str, np.ndarray], best_of: np.ndarray, summarize_members: Any) -> dict[str, Any]:
    before = _input_digest(raw, hist)
    checks: dict[str, Any] = {"rows": int(len(best_of)), "optional_modes": [], "swap_error_by_mode": {}, "input_mutation": False}
    for mode in ("full", "no_w20", "no_organization"):
        normal = np.asarray(model.predict_members(raw, hist, best_of, mode=mode), dtype=float)
        reverse_raw, reverse_hist = _swap_features(raw, hist)
        reverse = np.asarray(model.predict_members(reverse_raw, reverse_hist, best_of, mode=mode), dtype=float)
        if (
            normal.shape != (len(best_of), 5)
            or reverse.shape != normal.shape
            or not np.isfinite(normal).all()
            or not np.isfinite(reverse).all()
            or ((normal <= 0) | (normal >= 1)).any()
            or ((reverse <= 0) | (reverse >= 1)).any()
        ):
            raise ValueError(f"Saved K5 {mode} smoke returned invalid member predictions")
        error = float(np.max(np.abs(normal + reverse - 1.0)))
        if error > 2e-6:
            raise ValueError(f"Saved K5 swap smoke failed for {mode}: {error}")
        checks["optional_modes"].append(mode)
        checks["swap_error_by_mode"][mode] = error
        if mode == "full":
            cold_raw, cold_hist = _empty_features(raw, hist)
            cold = np.asarray(model.predict_members(cold_raw, cold_hist, best_of, mode="full"), dtype=float)
            if (
                cold.shape != normal.shape
                or not np.isfinite(cold).all()
                or ((cold <= 0) | (cold >= 1)).any()
            ):
                raise ValueError("Saved K5 empty-history smoke returned invalid member predictions")
            cold_error = float(np.max(np.abs(cold - 0.5)))
            if cold_error > 2e-6:
                raise ValueError(f"Saved K5 empty-history smoke failed: {cold_error}")
            checks["empty_history_probability_error"] = cold_error
            cold_summary = summarize_members(cold)
            variance_error = float(np.max(np.abs(cold_summary["total_variance"] - 0.25)))
            if not np.isfinite(variance_error) or variance_error > 2e-6:
                raise ValueError(f"Saved K5 empty-history variance smoke failed: {variance_error}")
            checks["empty_history_total_variance_error"] = variance_error
    after = _input_digest(raw, hist)
    if before != after:
        raise ValueError("Saved K5 inference mutated its input snapshots")
    return checks


def _base_forecast_row(target: Mapping[str, Any], forecaster: str) -> dict[str, Any]:
    year = target.get("model_year")
    scenario_fit = None if year is None or pd.isna(year) else f"{int(year):04d}-01-01T00:00:00+00:00"
    return {
        "replay_contract": REPLAY_CONTRACT_VERSION,
        "row_id": target.get("row_id"),
        "golgg_match_id": target.get("golgg_match_id"),
        "cohort": target.get("cohort"),
        "horizon": target.get("horizon"),
        "date": target.get("date"),
        "decision_at": target.get("decision_at", pd.NaT),
        "effective_decision_at": target.get("effective_decision_at", pd.NaT),
        "start_at": target.get("start_at", pd.NaT),
        "scenario_start_at": target.get("scenario_start_at", pd.NaT),
        "label_available_at": target.get("label_available_at", pd.NaT),
        "scenario_decision_at": target.get("scenario_decision_at", pd.NaT),
        "scenario_label_available_at": target.get("scenario_label_available_at", pd.NaT),
        "y": target.get("y"),
        "best_of": target.get("best_of"),
        "forecaster": forecaster,
        "fit_origin": pd.NaT,
        "scenario_fit_origin": scenario_fit,
        "fit_origin_source": "annual_model_schedule_scenario" if scenario_fit else None,
        "p": np.nan,
        "status": "unavailable",
        "missing_reason": target.get("missing_reason"),
        "forecast_eligible": target.get("forecast_eligible"),
        "forecast_absence": target.get("forecast_absence"),
        "target_native_effective_release_day": target.get("target_native_effective_release_day"),
        "target_effective_label_release_day": target.get("target_effective_label_release_day"),
        "score_kind": "probability",
        "model_year": year,
        "cutoff_day": target.get("cutoff_day"),
        "decision_time_source": target.get("decision_time_source"),
        "roster_status": target.get("roster_status"),
        "roster_missing_reason": target.get("roster_missing_reason"),
        "roster_evidence_status_a": target.get("roster_evidence_status_a"),
        "roster_evidence_status_b": target.get("roster_evidence_status_b"),
        "roster_evidence_missing_reason_a": target.get("roster_evidence_missing_reason_a"),
        "roster_evidence_missing_reason_b": target.get("roster_evidence_missing_reason_b"),
        "roster_a": target.get("roster_a"),
        "roster_b": target.get("roster_b"),
        "observed_roster_a": target.get("observed_roster_a"),
        "observed_roster_b": target.get("observed_roster_b"),
        "roster_source_a": target.get("roster_source_a"),
        "roster_source_b": target.get("roster_source_b"),
        "roster_identity_evidence_a": target.get("roster_identity_evidence_a"),
        "roster_identity_evidence_b": target.get("roster_identity_evidence_b"),
        "roster_ambiguity_evidence_a": target.get("roster_ambiguity_evidence_a"),
        "roster_ambiguity_evidence_b": target.get("roster_ambiguity_evidence_b"),
        "roster_evidence_input_hashes": target.get("roster_evidence_input_hashes"),
        "roster_announcement_a": target.get("roster_announcement_a"),
        "roster_announcement_b": target.get("roster_announcement_b"),
        "overlapping_prior_player_ids": target.get("overlapping_prior_player_ids"),
        "snapshot_query_roster_a": target.get("snapshot_query_roster_a"),
        "snapshot_query_roster_b": target.get("snapshot_query_roster_b"),
        "tensor_player_order_a": target.get("tensor_player_order_a"),
        "tensor_player_order_b": target.get("tensor_player_order_b"),
        "raw_tensor_player_order_a": target.get("raw_tensor_player_order_a"),
        "raw_tensor_player_order_b": target.get("raw_tensor_player_order_b"),
        "temporal_tensor_player_order_a": target.get("temporal_tensor_player_order_a"),
        "temporal_tensor_player_order_b": target.get("temporal_tensor_player_order_b"),
        "feature_source_max_day": target.get("feature_source_max_day"),
        "scenario_timestamp_assumption": target.get("scenario_timestamp_assumption"),
        "snapshot_index": target.get("row_order"),
    }


def _infer_predictions(
    arrays: _ReplayArrays,
    research_root: Path,
    *,
    digest_cache: dict[Path, str],
    batch_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from src.models.a0_uncertainty import load_model, summarize_members

    targets = arrays.targets
    forecasts: list[dict[str, Any]] = []
    row_map: dict[tuple[int, str], dict[str, Any]] = {}
    for index, target in targets.iterrows():
        for forecaster in FORECASTERS:
            row = _base_forecast_row(target, forecaster)
            reason = target.get("missing_reason")
            if not bool(target.get("feature_available", False)):
                row["missing_reason"] = reason or "snapshot_unavailable"
            row_map[(int(index), forecaster)] = row
    for index, target in targets.iterrows():
        year = target.get("model_year")
        if pd.isna(year) or int(year) in MODEL_YEARS or not bool(target.get("feature_available", False)):
            continue
        for forecaster in FORECASTERS:
            row_map[(int(index), forecaster)]["missing_reason"] = f"unsupported_cutoff_model_year:{int(year)}"
    model_receipts: dict[str, Any] = {}
    smoke_receipts: dict[str, Any] = {}
    native_sources = _verify_native_sources(research_root)
    native_module = _activate_native_predictor(research_root)

    for year in MODEL_YEARS:
        year_indices = np.flatnonzero(
            pd.to_numeric(targets["model_year"], errors="coerce").fillna(-1).to_numpy(dtype=int) == year
        )
        feature_indices = np.asarray([
            index for index in year_indices if bool(targets.iloc[index].get("feature_available", False))
        ], dtype=np.int64)
        if not len(year_indices):
            continue
        bo = pd.to_numeric(targets.iloc[feature_indices]["best_of"], errors="coerce").to_numpy(dtype=np.int64) if len(feature_indices) else np.empty(0, dtype=np.int64)
        origin = pd.Timestamp(year=year, month=1, day=1, tz="UTC")

        native_dir = _native_model_folder(research_root, year)
        native_receipt = native_dir / "calibration.json"
        native_required = [
            native_receipt,
            native_dir / "nn_20260910.joblib", native_dir / "nn_20260911.joblib", native_dir / "nn_20260912.joblib",
            native_dir / "other" / f"{year}_ratings_raw_20260910.joblib",
            native_dir / "other" / f"{year}_full_mlp_20260910.joblib",
            native_dir / "other" / f"{year}_full_mlp_20260911.joblib",
            native_dir / "other" / f"{year}_full_mlp_20260912.joblib",
            native_dir / "mixture_full.joblib", native_dir / "mixture_no_w20.joblib", native_dir / "mixture_no_organization.joblib",
        ]
        missing_native = [path.name for path in native_required if not path.is_file()]
        if missing_native:
            for index in year_indices:
                row = row_map[(int(index), "native_annual_a0")]
                if bool(targets.iloc[index].get("feature_available", False)):
                    row["missing_reason"] = "missing_annual_model_artifact:" + ",".join(missing_native)
            model_receipts[f"native_annual_a0/{year}"] = {
                "status": "unavailable", "missing_reason": "missing_annual_model_artifact:" + ",".join(missing_native),
            }
        elif len(feature_indices):
            from src.models.a0_uncertainty import _sha256 as uncertainty_sha256

            predictor = native_module.CausalA0Predictor(year)
            native_calibration = _read_json(native_receipt)
            train_ids = native_calibration.get("train_ids")
            annual_calibration_ids = native_calibration.get("annual_cal_ids")
            mixture_calibration_ids = native_calibration.get("mixture_cal_ids")
            if (
                not isinstance(train_ids, list) or not train_ids
                or not isinstance(annual_calibration_ids, list) or not annual_calibration_ids
                or not isinstance(mixture_calibration_ids, list) or not mixture_calibration_ids
            ):
                raise ValueError(f"Native annual receipt omits training or calibration identities: {native_receipt}")
            calibration_ids = list(dict.fromkeys([*annual_calibration_ids, *mixture_calibration_ids]))
            if set(train_ids).intersection(calibration_ids):
                raise ValueError(f"Native annual train/calibration identities overlap: {native_receipt}")
            max_train_release = _max_release_for_ids(train_ids, arrays.label_release_days, "native train")
            max_cal_release = _max_release_for_ids(calibration_ids, arrays.label_release_days, "native calibration")
            if (
                _as_day(max_train_release, "native max train release") >= date(year - 1, 1, 1)
                or _as_day(max_cal_release, "native max CAL release") >= date(year, 1, 1)
            ):
                raise ValueError(f"Native annual fit includes labels released outside its schedule: {native_receipt}")
            native_artifact_hashes = dict(predictor.hashes)
            model_receipts[f"native_annual_a0/{year}"] = {
                "status": "verified", "year": year, "model_dir": str(native_dir),
                "calibration_sha256": uncertainty_sha256(native_receipt),
                "source_hashes": native_sources,
                "artifact_hashes": native_artifact_hashes,
                "max_train_release": max_train_release,
                "max_cal_release": max_cal_release,
                "label_release_source": "corrected release-map override or latest source game day",
                "train_ids": train_ids,
                "calibration_ids": calibration_ids,
                "annual_calibration_ids": annual_calibration_ids,
                "mixture_calibration_ids": mixture_calibration_ids,
                "train_ids_count": len(train_ids),
                "cal_ids_count": len(calibration_ids),
            }
            for start in range(0, len(feature_indices), batch_size):
                batch_indices = feature_indices[start : start + batch_size]
                raw, hist = _batch_features(arrays.raw, arrays.hist, batch_indices)
                result = np.asarray(predictor.predict(raw, hist, targets.iloc[batch_indices]["best_of"].to_numpy(dtype=int)), dtype=float)
                if result.shape != (len(batch_indices), 3) or not np.isfinite(result).all() or ((result <= 0) | (result >= 1)).any():
                    raise ValueError(f"Native A0 returned invalid output for year {year}")
                if start == 0:
                    smoke_raw, smoke_hist = _batch_features(arrays.raw, arrays.hist, batch_indices[: min(len(batch_indices), 3)])
                    smoke_receipts[f"native_annual_a0/{year}"] = _native_smoke(
                        predictor, smoke_raw, smoke_hist,
                        targets.iloc[batch_indices[: min(len(batch_indices), 3)]]["best_of"].to_numpy(dtype=int),
                    )
                for local, index in enumerate(batch_indices):
                    row = row_map[(int(index), "native_annual_a0")]
                    row["p"] = float(result[local, 0])
                    row["p_no_w20"] = float(result[local, 1])
                    row["p_no_organization"] = float(result[local, 2])
                    row["status"] = "available"
                    row["missing_reason"] = None
                    row["scenario_fit_origin"] = origin.isoformat()
            del predictor
            gc.collect()

        for variant in ("seed_only", "block_bootstrap"):
            forecaster = variant
            model_dir = PROJECT_ROOT / K5_MODEL_RELATIVE / variant / str(year)
            receipt, model_receipt = _check_k5_bundle(model_dir, research_root, digest_cache)
            model_receipts[f"{variant}/{year}"] = model_receipt
            if receipt is None:
                if model_receipt.get("status") == "unavailable":
                    for index in year_indices:
                        row = row_map[(int(index), forecaster)]
                        if bool(targets.iloc[index].get("feature_available", False)):
                            row["missing_reason"] = model_receipt.get("missing_reason", "missing_annual_model_artifact")
                continue
            if not len(feature_indices):
                continue
            model = load_model(model_dir, research_root)
            for start in range(0, len(feature_indices), batch_size):
                batch_indices = feature_indices[start : start + batch_size]
                raw, hist = _batch_features(arrays.raw, arrays.hist, batch_indices)
                best_of = targets.iloc[batch_indices]["best_of"].to_numpy(dtype=int)
                members = np.asarray(model.predict_members(raw, hist, best_of, mode="full"), dtype=float)
                no_w20 = np.asarray(model.predict_members(raw, hist, best_of, mode="no_w20"), dtype=float)
                no_org = np.asarray(model.predict_members(raw, hist, best_of, mode="no_organization"), dtype=float)
                if members.shape != (len(batch_indices), 5) or not np.isfinite(members).all() or ((members <= 0) | (members >= 1)).any():
                    raise ValueError(f"Saved K5 model returned invalid predictions for {variant}/{year}")
                if (
                    no_w20.shape != members.shape
                    or no_org.shape != members.shape
                    or not np.isfinite(no_w20).all()
                    or not np.isfinite(no_org).all()
                    or ((no_w20 <= 0) | (no_w20 >= 1)).any()
                    or ((no_org <= 0) | (no_org >= 1)).any()
                ):
                    raise ValueError(f"Saved K5 optional modes returned invalid predictions for {variant}/{year}")
                summary = summarize_members(members)
                summary_probabilities = {
                    key: np.asarray(summary[key], dtype=float)
                    for key in ("p", "p_q10", "p_q90")
                }
                summary_uncertainties = {
                    key: np.asarray(summary[key], dtype=float)
                    for key in ("p_std", "sigma_z")
                }
                if (
                    any(values.shape != (len(batch_indices),) for values in (*summary_probabilities.values(), *summary_uncertainties.values()))
                    or any(not np.isfinite(values).all() for values in (*summary_probabilities.values(), *summary_uncertainties.values()))
                    or any(((values <= 0) | (values >= 1)).any() for values in summary_probabilities.values())
                    or any((values < 0).any() for values in summary_uncertainties.values())
                    or (summary_probabilities["p_q10"] > summary_probabilities["p"]).any()
                    or (summary_probabilities["p"] > summary_probabilities["p_q90"]).any()
                ):
                    raise ValueError(f"Saved K5 member summary returned invalid probabilities for {variant}/{year}")
                if start == 0:
                    smoke_raw, smoke_hist = _batch_features(arrays.raw, arrays.hist, batch_indices[: min(len(batch_indices), 3)])
                    smoke_best_of = targets.iloc[batch_indices[: min(len(batch_indices), 3)]]["best_of"].to_numpy(dtype=int)
                    smoke_receipts[f"{variant}/{year}"] = _k5_smoke(model, smoke_raw, smoke_hist, smoke_best_of, summarize_members)
                for local, index in enumerate(batch_indices):
                    row = row_map[(int(index), forecaster)]
                    row["p"] = float(summary["p"][local])
                    row["p_std"] = float(summary["p_std"][local])
                    row["sigma_z"] = float(summary["sigma_z"][local])
                    row["p_q10"] = float(summary["p_q10"][local])
                    row["p_q90"] = float(summary["p_q90"][local])
                    row["p_no_w20"] = float(no_w20[local].mean())
                    row["p_no_organization"] = float(no_org[local].mean())
                    for member_index in range(5):
                        row[f"p_member_{member_index}"] = float(members[local, member_index])
                    row["status"] = "available"
                    row["missing_reason"] = None
                    row["scenario_fit_origin"] = origin.isoformat()
            del model
            gc.collect()
    forecasts = [row_map[key] for key in sorted(row_map, key=lambda key: (key[0], FORECASTERS.index(key[1])))]
    frame = pd.DataFrame(forecasts)
    frame["score_kind"] = "probability"
    receipt = {
        "annual_schedule": {
            str(year): {
                "selection": "UTC decision cutoff year; never target start year or nearest-year fallback",
                "scenario_model_origin": f"{year:04d}-01-01T00:00:00+00:00",
                "native": model_receipts.get(f"native_annual_a0/{year}", {"status": "not_needed"}),
                "seed_only": model_receipts.get(f"seed_only/{year}", {"status": "not_needed"}),
                "block_bootstrap": model_receipts.get(f"block_bootstrap/{year}", {"status": "not_needed"}),
            }
            for year in MODEL_YEARS
        },
        "smoke_checks": smoke_receipts,
        "model_receipts": model_receipts,
        "native_source_hashes": native_sources,
        "batch_size": batch_size,
        "predictions_do_not_join_cached_test_rows": True,
    }
    return frame, receipt


def run_asof_replay(
    opportunities: pd.DataFrame,
    metadata: pd.DataFrame,
    research_root: str | Path | None = None,
    raw_history_path: str | Path | None = None,
    *,
    protocol_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    feature_state_factory: Any = None,
    roster_evidence: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct release-aware snapshots and run all three annual forecasters."""
    root = _resolve_research_root(research_root)
    protocol = _verify_protocol(_project_path(protocol_path, Path("data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json")))
    raw_path = _project_path(raw_history_path, Path(protocol["inputs"]["raw_history"]))
    bank, releases, bank_receipt, hashes = _verify_corrected_bank(root, protocol, raw_path)
    original, bank_arrays, bank_temporal = _read_bank_validation_data(bank, bank_receipt)
    controls = _select_oracle_controls(original)
    production = opportunities.copy()
    production = production.loc[
        production.get("cohort", pd.Series(index=production.index, dtype="string")).astype("string").eq("production")
        & production.get("horizon", pd.Series(index=production.index, dtype="string")).astype("string").isin(PRODUCTION_HORIZONS)
    ].reset_index(drop=True)
    if production.empty:
        raise ValueError("No production OPEN/48h/24h/CLOSE opportunity rows are available")
    production["cohort"] = ASOF_COHORT
    result = _build_replay_arrays(
        production, metadata, root, raw_path, releases,
        validation_controls=controls, bank_arrays=bank_arrays, bank_temporal=bank_temporal,
        feature_state_factory=feature_state_factory, roster_evidence=roster_evidence,
    )
    # Raw parser intermediates and oracle bank arrays are released before any
    # annual full-mixture or K5 bundles are loaded.
    del original, bank_arrays, bank_temporal
    gc.collect()
    predictions, model_receipt = _infer_predictions(result, root, digest_cache={})
    artifact_created_at_utc = datetime.now(timezone.utc).isoformat()
    predictions["artifact_created_at_utc"] = artifact_created_at_utc
    input_hashes = {
        **hashes,
        "protocol": protocol["_verified_sha256"],
        "asof_replay_source": _sha256(Path(__file__).resolve()),
    }
    receipt = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_contract": REPLAY_CONTRACT_VERSION,
        "run_id": REPLAY_RUN_ID,
        "artifact_created_at_utc": artifact_created_at_utc,
        "status": "COMPLETE_DEVELOPMENT_SCENARIO",
        "scope": "daily-release roster scenario with explicit temporal identity/announcement evidence; not certified historical forecast issuance",
        "input_hashes": input_hashes,
        "roster_evidence_input_hashes": result.roster_evidence_input_hashes,
        "feature_source_hashes": result.feature_source_hashes,
        "source_preparation_receipt": None,
        "release_precedence": "max(max(game.day), date.fromisoformat(releases.get(series_id, max(game.day).isoformat())))",
        "release_map_is_optional_override": True,
        "history": result.release_audit,
        "snapshot_replay": result.validation_receipt,
        "bank_compatibility_oracle": {
            "status": "PASS_EXACT_CAST_VALUES",
            "rows": len(result.validation_receipt["oracle_controls"]),
            "comparisons": result.validation_receipt["oracle_controls"],
            "actual_rosters_used_only_for_validation": True,
        },
        "forecasts": model_receipt,
        "prediction_rows": len(predictions),
        "prediction_status_counts": predictions.groupby(["forecaster", "status"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "roster_availability_counts": result.release_audit["roster_availability"],
        "unknown_actual_timestamps_remain_missing": True,
        "scenario_timestamp_assumption": "decision_at is the historical scenario cutoff; prior-observed lineups are explicitly labeled scenarios, announcements require timestamped evidence available before cutoff and match start; artifact_created_at_utc is actual replay creation time",
        "artifact_schema": {
            "snapshots_npz": "row-aligned raw_{players,core,team,w20,gates} float32; temporal_history float16; temporal_mask uint8; temporal_champions uint16",
            "snapshot_ledger_csv": "schema v4; independent native/explicit-label release clocks and forecast eligibility; exact tournament context, archived target-context provenance, original observed rosters and evidence provenance remain separate from lexical tensor player order; unavailable rows and reasons retained",
            "predictions_csv": "schema v4; row keys, decision/scenario timestamps, independent target-release clocks and forecast absence, exact tournament context and provenance, roster evidence/clocks, actual tensor order, model outputs, artifact_created_at_utc; unavailable rows retained",
        },
        "cohort": ASOF_COHORT,
    }
    return predictions, receipt


def _ensure_fresh_output(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise FileExistsError(f"Output directory must be fresh and empty: {path}")


def _csv_safe_frame(frame: pd.DataFrame, json_columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in json_columns:
        if column in result:
            result[column] = result[column].map(
                lambda value: json.dumps(_json_safe(value), sort_keys=True, allow_nan=False)
                if value is not None and value is not pd.NA and not (isinstance(value, float) and np.isnan(value))
                else ""
            )
    return result


def _save_snapshot_npz(path: Path, targets: pd.DataFrame, raw: Mapping[str, np.ndarray], hist: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {
        "row_id": targets["row_id"].astype(str).to_numpy(dtype=f"U{max(1, targets['row_id'].astype(str).str.len().max())}"),
        "golgg_match_id": targets["golgg_match_id"].astype(str).to_numpy(dtype=f"U{max(1, targets['golgg_match_id'].astype(str).str.len().max())}"),
        "cutoff_day": targets["cutoff_day"].fillna("").astype(str).to_numpy(dtype="U10"),
        **{f"raw_{key}": np.asarray(value) for key, value in raw.items()},
        **{f"temporal_{key}": np.asarray(value) for key, value in hist.items()},
    }
    np.savez_compressed(path, **payload)
    return _sha256(path)


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    from src.analysis.ev_approach_data import prepare_datasets

    protocol_path = _project_path(args.protocol, Path("data/05_model_input/ev_all_approaches/ev_all_approaches_20260927_220902/protocol.json"))
    protocol = _verify_protocol(protocol_path)
    research_root = _resolve_research_root(args.research_root)
    output_dir = _project_path(args.output_dir, Path(args.output_dir))
    _ensure_fresh_output(output_dir)
    raw_path = _project_path(args.raw_history, Path(protocol["inputs"]["raw_history"]))
    prepared = prepare_datasets(
        research_root=research_root,
        panel_path=args.panel_path,
        quote_path=args.quote_path,
        uncertainty_dir=args.uncertainty_dir,
    )
    canonical = prepared["canonical"]
    metadata_columns = [
        column for column in ("golgg_match_id", "team1_id", "team2_id", "best_of", "y", "date", "result_day")
        if column in canonical
    ]
    metadata = canonical.loc[:, metadata_columns].copy()
    opportunities = prepared["opportunities"].copy()
    production = opportunities.loc[
        opportunities["cohort"].astype("string").eq("production")
        & opportunities["horizon"].astype("string").isin(PRODUCTION_HORIZONS)
    ].reset_index(drop=True)
    if production.empty:
        raise ValueError("No production OPEN/48h/24h/CLOSE opportunities are available")
    production["cohort"] = ASOF_COHORT
    bank, releases, bank_receipt, source_hashes = _verify_corrected_bank(research_root, protocol, raw_path)
    original, bank_arrays, bank_temporal = _read_bank_validation_data(bank, bank_receipt)
    controls = _select_oracle_controls(original)
    arrays = _build_replay_arrays(
        production, metadata, research_root, raw_path, releases,
        validation_controls=controls, bank_arrays=bank_arrays, bank_temporal=bank_temporal,
        roster_evidence=getattr(args, "roster_evidence", None),
    )
    del original, bank_arrays, bank_temporal
    gc.collect()
    predictions, model_receipt = _infer_predictions(arrays, research_root, digest_cache={})
    artifact_created_at_utc = datetime.now(timezone.utc).isoformat()
    predictions["artifact_created_at_utc"] = artifact_created_at_utc

    snapshots_path = _project_path(args.snapshots_output, output_dir / "stage04_asof_snapshots.npz")
    ledger_path = _project_path(args.ledger_output, output_dir / "stage04_asof_roster_ledger.csv")
    predictions_path = _project_path(args.predictions_output, output_dir / "stage07_asof_predictions.csv")
    receipt_path = _project_path(args.receipt_output, output_dir / "asof_replay_receipt.json")
    for path in (snapshots_path, ledger_path, predictions_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing as-of artifact: {path}")
    for path in (snapshots_path, ledger_path, predictions_path, receipt_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    snapshot_hash = _save_snapshot_npz(snapshots_path, arrays.targets, arrays.raw, arrays.hist)
    ledger_columns = [
        "replay_contract", "row_order", "row_id", "golgg_match_id", "cohort", "horizon", "date", "decision_at",
        "scenario_decision_at", "start_at", "scenario_start_at", "label_available_at",
        "scenario_label_available_at", "effective_decision_at", "decision_time_source",
        "cutoff_day", "model_year", "best_of", "team1_id", "team2_id",
        "forecast_eligible", "forecast_absence",
        "target_native_effective_release_day", "target_effective_label_release_day",
        "roster_a", "roster_b", "observed_roster_a", "observed_roster_b", "roster_status",
        "roster_missing_reason", "roster_evidence_status_a", "roster_evidence_status_b",
        "roster_evidence_missing_reason_a", "roster_evidence_missing_reason_b",
        "roster_identity_evidence_a", "roster_identity_evidence_b",
        "roster_announcement_a", "roster_announcement_b", "roster_evidence_input_hashes",
        "roster_ambiguity_evidence_a", "roster_ambiguity_evidence_b",
        "overlapping_prior_player_ids", "roster_source_a", "roster_source_b",
        "team_feature_provenance", "snapshot_query_roster_a", "snapshot_query_roster_b",
        "tensor_player_order_a", "tensor_player_order_b",
        "raw_tensor_player_order_a", "raw_tensor_player_order_b",
        "temporal_tensor_player_order_a", "temporal_tensor_player_order_b",
        "draft_status", "feature_available", "missing_reason", "feature_source_max_day",
        "scenario_timestamp_assumption",
    ]
    ledger_columns = [column for column in ledger_columns if column in arrays.targets]
    ledger = _csv_safe_frame(
        arrays.targets.loc[:, ledger_columns],
        (
            "roster_a", "roster_b", "observed_roster_a", "observed_roster_b",
            "roster_identity_evidence_a", "roster_identity_evidence_b",
            "roster_announcement_a", "roster_announcement_b", "roster_evidence_input_hashes",
            "roster_ambiguity_evidence_a", "roster_ambiguity_evidence_b",
            "overlapping_prior_player_ids", "roster_source_a", "roster_source_b",
            "team_feature_provenance", "snapshot_query_roster_a", "snapshot_query_roster_b",
            "tensor_player_order_a", "tensor_player_order_b",
            "raw_tensor_player_order_a", "raw_tensor_player_order_b",
            "temporal_tensor_player_order_a", "temporal_tensor_player_order_b",
        ),
    )
    _csv_safe_frame(
        predictions,
        (
            "roster_a", "roster_b", "observed_roster_a", "observed_roster_b",
            "roster_source_a", "roster_source_b",
            "roster_identity_evidence_a", "roster_identity_evidence_b",
            "roster_announcement_a", "roster_announcement_b",
            "roster_evidence_input_hashes",
            "roster_ambiguity_evidence_a", "roster_ambiguity_evidence_b",
            "overlapping_prior_player_ids",
            "snapshot_query_roster_a", "snapshot_query_roster_b",
            "tensor_player_order_a", "tensor_player_order_b",
            "raw_tensor_player_order_a", "raw_tensor_player_order_b",
            "temporal_tensor_player_order_a", "temporal_tensor_player_order_b",
        ),
    ).to_csv(predictions_path, index=False)
    ledger.to_csv(ledger_path, index=False)
    receipt = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_contract": REPLAY_CONTRACT_VERSION,
        "run_id": REPLAY_RUN_ID,
        "artifact_created_at_utc": artifact_created_at_utc,
        "status": "COMPLETE_DEVELOPMENT_SCENARIO",
        "evidence_limit": protocol.get("evidence_limit"),
        "scope": "daily-release roster scenario with explicit temporal identity/announcement evidence; not certified historical forecast issuance",
        "protocol": {"path": str(protocol_path), "sha256": protocol["_verified_sha256"]},
        "source_preparation_receipt": prepared["receipt"],
        "input_hashes": {
            **source_hashes,
            "protocol": protocol["_verified_sha256"],
            "asof_replay_source": _sha256(Path(__file__).resolve()),
            "asof_contract_tests": _sha256(PROJECT_ROOT / "betting_app/tests/test_ev_asof_replay.py"),
        },
        "roster_evidence_input_hashes": arrays.roster_evidence_input_hashes,
        "feature_source_hashes": arrays.feature_source_hashes,
        "history": arrays.release_audit,
        "snapshot_replay": arrays.validation_receipt,
        "bank_compatibility_oracle": {
            "status": "PASS_EXACT_CAST_VALUES",
            "rows": len(arrays.validation_receipt["oracle_controls"]),
            "comparisons": arrays.validation_receipt["oracle_controls"],
            "actual_rosters_used_only_for_validation": True,
        },
        "forecasts": model_receipt,
        "output_hashes": {
            "stage04_asof_snapshots.npz": snapshot_hash,
            "stage04_asof_roster_ledger.csv": _sha256(ledger_path),
            "stage07_asof_predictions.csv": _sha256(predictions_path),
        },
        "output_paths": {
            "snapshots": str(snapshots_path),
            "roster_ledger": str(ledger_path),
            "predictions": str(predictions_path),
        },
        "prediction_rows": len(predictions),
        "prediction_status_counts": predictions.groupby(["forecaster", "status"], dropna=False).size().rename("rows").reset_index().to_dict("records"),
        "roster_availability_counts": arrays.release_audit["roster_availability"],
        "unknown_actual_timestamps_remain_missing": True,
        "scenario_timestamp_assumption": "decision_at is the historical scenario cutoff; prior-observed lineups are explicitly labeled scenarios, announcements require timestamped evidence available before cutoff and match start; artifact_created_at_utc is actual replay creation time",
        "artifact_schema": {
            "snapshots_npz": "row-aligned row_id/golgg_match_id/cutoff_day; raw_{players,core,team,w20,gates} float32; temporal_history float16; temporal_mask uint8; temporal_champions uint16",
            "snapshot_ledger_csv": "schema v4; independent native/explicit-label release clocks and forecast eligibility; exact tournament context, archived target-context provenance, original observed rosters and evidence provenance remain separate from lexical tensor player order; unavailable rows and reasons retained",
            "predictions_csv": "schema v4; row keys, decision/scenario timestamps, independent target-release clocks and forecast absence, exact tournament context and provenance, roster evidence/clocks, actual tensor order, model outputs, artifact_created_at_utc; unavailable rows retained",
        },
        "cohort": ASOF_COHORT,
    }
    _write_json(receipt_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--raw-history", type=Path)
    parser.add_argument("--roster-evidence", type=Path)
    parser.add_argument("--panel-path", type=Path)
    parser.add_argument("--quote-path", type=Path)
    parser.add_argument("--uncertainty-dir", type=Path)
    parser.add_argument("--snapshots-output", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)
    receipt = _run_cli(args)
    print(json.dumps({
        "status": receipt["status"],
        "prediction_rows": receipt["prediction_rows"],
        "output_paths": receipt["output_paths"],
        "receipt": str(_project_path(args.receipt_output, _project_path(args.output_dir, Path(args.output_dir)) / "asof_replay_receipt.json")),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
