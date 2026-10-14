"""Offline, slot-indexed tournament scenarios and fail-closed source coverage.

Recipes are deliberately not a prose compiler or point-in-time certification.
The catalog distinguishes source-backed topology from generic scenario mechanics;
both retain explicit unresolved historical rules. No results feed instantiation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "conf/base/tournament_formats.json"
DEFAULT_BUNDLE = ROOT / "data/artifacts/leaguepedia-tournament-rules-20260908"
FAMILIES = ("lck", "lpl", "lec", "lcs", "worlds", "msi", "eu_masters")


def _load_catalog() -> dict[str, Any]:
    catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))
    if catalog.get("version") != 1:
        raise ValueError("Unsupported tournament catalog version")
    ids = [profile["id"] for profile in catalog["profiles"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate tournament profile IDs")
    return catalog


def list_profiles() -> list[dict[str, Any]]:
    """Return metadata, excluding executable templates and historical outcomes."""
    return [{key: value for key, value in profile.items() if key != "spec"}
            for profile in _load_catalog()["profiles"]]


def instantiate_profile(profile_id: str, teams: Sequence[str]) -> dict[str, Any]:
    """Fill zero-based ordered team slots; return an independent engine spec.

    Team IDs are opaque exact strings. Caller order supplies declared initial
    seeds/group allocation; it is never derived from observed historical ranks.
    This is a scenario API, not authorization to label a forecast historical-PIT.
    """
    catalog = _load_catalog()
    profile = next((row for row in catalog["profiles"] if row["id"] == profile_id), None)
    if profile is None:
        raise ValueError(f"Unknown tournament profile: {profile_id}")
    if isinstance(teams, (str, bytes)) or not isinstance(teams, Sequence):
        raise ValueError("teams must be an ordered sequence of team IDs")
    if len(teams) != profile["team_count"]:
        raise ValueError(f"Profile {profile_id} requires {profile['team_count']} team slots")
    if any(not isinstance(team, str) or not team.strip() for team in teams):
        raise ValueError("Each team ID must be a nonempty string")
    if len(set(teams)) != len(teams):
        raise ValueError("Team IDs must be unique")

    def fill(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"slot"}:
                index = value["slot"]
                if type(index) is not int or not 0 <= index < len(teams):
                    raise ValueError(f"Invalid team slot in profile {profile_id}: {index}")
                return teams[index]
            return {key: fill(item) for key, item in value.items()}
        if isinstance(value, list):
            return [fill(item) for item in value]
        return value

    spec = fill(profile["spec"])
    phase_sources = {row["phase_id"]: row["source"] for row in catalog["phase_rules"]}
    spec["metadata"] = {
        "profile_id": profile_id,
        "profile_status": profile["status"],
        "structure_source_verified": profile["structure_source_verified"],
        "version_scope": profile["version_scope"],
        "assumptions": profile["assumptions"],
        "sources": [phase_sources[phase_id] for phase_id in profile["source_phase_ids"]],
        "historical_publication_certified": False,
        "historically_ready": False,
        "observed_results_used": False,
    }
    return spec


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_catalog(bundle: str | Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    """Audit every source phase offline, without using standings as simulated seeds.

    Bracket and standings evidence is title-level: counts do not assert a phase
    join or machine-compiled rules. Current revisions fail historical readiness
    even where their topology supports an executable recipe.
    """
    bundle = Path(bundle)
    catalog = _load_catalog()
    events = _jsonl(bundle / "events.jsonl")
    event_ids = [event["phase_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate historical phase IDs in bundle")
    rules = {row["phase_id"]: row for row in catalog["phase_rules"]}
    profile_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in catalog["profiles"]:
        for phase_id in profile["source_phase_ids"]:
            profile_map[phase_id].append(profile)
    brackets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(bundle / "bracket_sources.jsonl"):
        brackets[row["title"]].append(row)
    standings = Counter(row["event_title"] for row in _jsonl(bundle / "standings.jsonl"))
    research = {family: json.loads((bundle / "research" / f"{family}.json").read_text(encoding="utf-8"))
                for family in FAMILIES}
    phases = []
    for event in events:
        phase_id = event["phase_id"]
        rule = rules.get(phase_id)
        profiles = profile_map.get(phase_id, [])
        source = event.get("source", {})
        changed = rule is not None and source != rule["source"]
        gaps = ["Historical pre-event publication/capture is not certified by this retrieved revision.",
                "Pre-event participant IDs, initial seed/group assignments and model/features require separate point-in-time evidence.",
                "Side choice, draft and duration rules are not simulated by supplied series probabilities."]
        if rule is None:
            gaps.append("Phase has no manually reviewed catalog classification; no executable historical recipe authorized.")
        else:
            gaps.extend(rule["additional_gaps"])
        if changed:
            gaps.append("Source provenance changed since catalog review; recipe linkage must be re-reviewed.")
        if not profiles:
            gaps.append("No executable profile configured; historical prose and observed brackets have not been compiled.")
        for profile in profiles:
            gaps.extend(profile["assumptions"])
        status = ("cancelled_no_enacted_format" if rule and rule["mechanic"] == "cancelled"
                  else "source_revision_changed" if changed
                  else "unreviewed_phase" if rule is None
                  else "historical_publication_unverified")
        evidence = brackets.get(event["title"], [])
        phases.append({
            "phase_id": phase_id, "family": event["family"], "year": event["year"],
            "stage": event["stage"], "title": event["title"],
            "start_date": event.get("start_date"), "end_date": event.get("end_date"),
            "event_status": event.get("status"), "participant_count": event.get("participant_count"),
            "mechanic": rule["mechanic"] if rule else "unreviewed",
            "source": source, "status": status, "historically_ready": False,
            "source_claims_historical_publication_certified": event.get("historical_publication_certified", False),
            "configured": bool(profiles) and not changed,
            "profile_ids": [profile["id"] for profile in profiles] if not changed else [],
            "source_backed_profile_ids": [p["id"] for p in profiles if p["structure_source_verified"]] if not changed else [],
            "generic_profile_ids": [p["id"] for p in profiles if not p["structure_source_verified"]] if not changed else [],
            "format_summary": event.get("format_summary"),
            "rule_evidence": {key: event.get(key) for key in (
                "series_formats", "advancement_and_seeding", "draw_constraints", "side_selection_and_draft")},
            "bracket_evidence": {"scope": "event_title_not_phase_join", "record_count": len(evidence),
                                 "templates": sorted({row["template"] for row in evidence}),
                                 "revision_ids": sorted({row["revid"] for row in evidence if row.get("revid") is not None})},
            "standings_evidence": {"scope": "event_title_not_phase_join", "row_count": standings[event["title"]],
                                   "used_for_recipe_seeding": False},
            "readiness_gaps": list(dict.fromkeys(gaps)),
        })
    families = {}
    for family in sorted(set(FAMILIES) | {row["family"] for row in phases}):
        rows = [row for row in phases if row["family"] == family]
        configured = sum(row["configured"] for row in rows)
        families[family] = {
            "phase_count": len(rows), "years": sorted({row["year"] for row in rows}),
            "configured_phase_count": configured, "unconfigured_phase_count": len(rows) - configured,
            "source_backed_phase_count": sum(bool(row["source_backed_profile_ids"]) for row in rows),
            "generic_only_phase_count": sum(bool(row["generic_profile_ids"]) and not row["source_backed_profile_ids"] for row in rows),
            "historically_ready_count": 0, "historically_unresolved_count": len(rows),
            "mechanics": dict(sorted(Counter(row["mechanic"] for row in rows).items())),
            "research_gaps": research.get(family, {}).get("gaps", []),
            "additional_sources": research.get(family, {}).get("additional_sources", []),
        }
    return {
        "version": 1, "bundle": str(bundle), "network_used": False,
        "phase_count": len(phases), "catalog_expected_phase_count": catalog["phase_count"],
        "catalogued_phases_missing_from_bundle": sorted(set(rules) - set(event_ids)),
        "profile_count": len(catalog["profiles"]), "historically_ready_count": 0,
        "configured_phase_count": sum(row["configured"] for row in phases),
        "unconfigured_phase_count": sum(not row["configured"] for row in phases),
        "historically_unresolved_count": len(phases),
        "policy": catalog["source_policy"],
        "coverage_note": "Configured means a mechanics scenario exists, not that the historical phase is fully compiled. Source-backed means topology evidence, not all draw/tie/side/draft rules or historical publication verification.",
        "families": families,
        "profiles": [{key: value for key, value in profile.items() if key != "spec"}
                     for profile in catalog["profiles"]],
        "phases": phases,
    }
