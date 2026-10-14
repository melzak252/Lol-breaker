#!/usr/bin/env python3
"""Prepare an exhaustive local reconstruction ledger, then explicitly bind research folds.

The prepare command reads archived sources only. Bind is a separate, opt-in all-pair
export/inference operation requiring both --benchmark-dir and --history. Labels are
written by a third command; neither bind nor forecast execution reads outcomes.json.
No captured-forecast or whole-edition boundary certification is inferred here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.tournament_catalog import DEFAULT_BUNDLE, DEFAULT_CATALOG, instantiate_profile

VERSION = "tournament-historical-cohort-v1"
DEFAULT_READINESS = ROOT / "data/artifacts/tournament-validation-roadmap-20260910/source-readiness-v3-r1/readiness_manifest.json"
ARTIFACT_ROOT = ROOT / "data/artifacts/tournament-validation-roadmap-20260910"
# Reviewed source-backed deterministic rank-to-slot contracts, not a generic
# interpretation of every profile's hypothetical input order.
PRIOR_RANK_PROFILES = {
    "lck-2020-playoff-stepladder", "lck-road-to-msi",
    "lpl-2020-staggered-playoffs", "lpl-2020-spring-mid-season-cup",
    "lpl-2021-2024-koth-double-elimination",
}
MODEL_FILES = {
    "exp039": ("exp039", "exp039.joblib"),
    "linear79": ("linear79", "linear79.joblib"),
    "exp081_weak": ("exp081", "exp081.json"),
    "exp081_strong": ("exp081", "exp081_strong.json"),
    "exp081_strong_ensemble": ("exp081", "exp081_strong_ensemble.json"),
    "glicko": ("glicko", "glicko.json"),
    "glicko_format": ("glicko_format", "glicko_format.json"),
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def fresh_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("research outputs must remain under the new roadmap artifact root")
    path.mkdir(parents=True, exist_ok=False)
    return path


def canonical(name, aliases):
    value = aliases.get(name, aliases.get(name.strip().casefold(), {}))
    if value.get("resolved") is True and value.get("canonical_title"):
        return value["canonical_title"].strip().casefold()
    # Exact provider spellings are identity evidence, not fuzzy transliteration.
    return name.strip().casefold()


def explicit_aliases(aliases):
    """Expand only display/title spellings explicitly emitted by resolved templates."""
    result, candidates = dict(aliases), defaultdict(list)
    for name, row in aliases.items():
        if row.get("resolved") is True and row.get("canonical_title"):
            candidates[name].append(row)
            for key in ("display_name", "canonical_title"):
                if row.get(key):
                    candidates[row[key]].append(row)
    for name, rows in candidates.items():
        if len({row["canonical_title"].casefold() for row in rows}) != 1:
            continue
        if name not in result:
            result[name] = {**rows[0], "expanded_from_explicit_template_field": name}
        if result[name].get("resolved") is True:
            result.setdefault(name.strip().casefold(), result[name])
    return result


def identity_at(name, cutoff_day, records, aliases):
    wanted = canonical(name, aliases)
    candidates = []
    for row in records:
        if not row.get("date") or row["date"] >= cutoff_day:
            continue
        for side in ("1", "2"):
            if row.get("team" + side + "_id") is not None and canonical(row.get("team" + side, ""), aliases) == wanted:
                if not str(row["team" + side + "_id"]).strip():
                    continue
                candidates.append((row["date"], str(row["team" + side + "_id"]), row, side))
    if not candidates:
        raise ValueError(f"no_strictly_prior_exact_provider_identity:{name}")
    latest = max(item[0] for item in candidates)
    current = [item for item in candidates if item[0] == latest]
    if len({item[1] for item in current}) != 1:
        raise ValueError(f"conflicting_latest_provider_identity:{name}:{latest}")
    day, team_id, row, side = current[0]
    return team_id, {"name": name, "canonical_name": wanted, "team_id": team_id,
                     "provider_name": row["team" + side], "provider_side": side,
                     "golgg_match_id": row["golgg_match_id"], "source_day": day,
                     "method": "exact_archived_alias_or_name_latest_strictly_prior_provider_record",
                     "alias_evidence": aliases.get(name), "provider_alias_evidence": aliases.get(row["team" + side]),
                     "uses_scores_winners_or_placing": False}


def prior_seed_rows(target, prior_events, standings, count):
    candidates = []
    for event in prior_events:
        if (event["phase_id"] == target["phase_id"] or event["title"] == target["title"]
                or event.get("status") != "completed" or not event.get("end_date")
                or event["end_date"] >= target["start_date"]):
            continue
        tables = defaultdict(list)
        for row in standings:
            if row.get("event_title") == event["title"]:
                tables[row.get("table_index")].append(row)
        for rows in tables.values():
            ordered = []
            for rank in range(1, count + 1):
                matches = [row for row in rows if str(row.get("placement")) == str(rank)]
                if len(matches) != 1:
                    break
                ordered.append(matches[0])
            if len(ordered) == count and len({row["team"] for row in ordered}) == count:
                candidates.append((ordered, event))
    if len(candidates) != 1:
        raise ValueError(f"prior_rank_seed_table_not_unique_or_complete:{len(candidates)}")
    return candidates[0]


def source_boundary(event, edition):
    start, end = date.fromisoformat(event["start_date"]), date.fromisoformat(event["end_date"])
    if end < start:
        raise ValueError("phase_end_precedes_start")
    return {"edition_start_at": edition.get("edition_start_at"),
            "edition_start_status": ("independently_evidenced_source_calendar_day" if edition.get("edition_start_at")
                                     else "unverified_whole_edition_boundary"),
            "phase_start_at": start.isoformat() + "T00:00:00Z",
            "phase_start_forecast_cutoff": (start - timedelta(days=1)).isoformat() + "T00:00:00Z",
            "phase_start_source_date": start.isoformat(),
            "phase_start_precision": "source_calendar_day_retrospectively_interpreted_as_UTC_midnight",
            "target_boundary_rule": "archived_phase_end_calendar_day_start; excludes_final_day_forecasts",
            "terminal_target_start_at": end.isoformat() + "T00:00:00Z"}


def page_content(phase):
    source = phase["source_event"]["source"]
    payload = load(phase["archived_identity_evidence"]["path"])
    pages = [page for page in payload["query"]["pages"] if page["pageid"] == source["pageid"]]
    revisions = [rev for page in pages for rev in page.get("revisions", []) if rev["revid"] == source["revid"]]
    if len(revisions) != 1:
        raise ValueError("archived_phase_revision_not_unique")
    return revisions[0]["slots"]["main"]["content"]


def linked_prior_events(phase, phases):
    identity = phase["archived_identity_evidence"]["identity"]
    links = set(phase["archived_identity_evidence"]["format_links"])
    output = []
    for other in phases:
        event = other["source_event"]
        evidence = other["archived_identity_evidence"]
        names = {event["title"], evidence["identity"].get("CM_StandardName")}
        reciprocal = phase["source_event"]["title"] in evidence["format_links"]
        if (names & links or reciprocal) and event["family"] == phase["source_event"]["family"]:
            if event["year"] == phase["source_event"]["year"] and identity.get("split") == evidence["identity"].get("split"):
                output.append(event)
    return output


def replay_elimination(spec, records):
    """Deterministic source join, not simulated or outcome-selected seed construction."""
    used, completed, rankings = set(), [], {}
    outputs, source_days = {}, {}

    def resolve(ref):
        if isinstance(ref, str):
            return ref
        if "stage" in ref:
            return rankings[ref["stage"]][ref["rank"] - 1]
        if "seeded" in ref:
            return sorted((resolve(item) for item in ref["seeded"]), key=spec["teams"].index)[ref["rank"] - 1]
        if "winner" in ref or "loser" in ref:
            kind = "winner" if "winner" in ref else "loser"
            return outputs[ref[kind]][0 if kind == "winner" else 1]
        raise ValueError(f"unsupported_source_slot:{ref}")

    def play(stage, node_id, a, b, best_of, dependencies=()):
        matches = [row for row in records if row["wiki_series_id"] not in used
                   and {row["team_a"], row["team_b"]} == {a, b} and row["best_of"] == best_of]
        if not matches:
            raise ValueError(f"no_legal_archived_node_series:{stage['id']}:{node_id}:{a}:{b}")
        matches.sort(key=lambda row: (row["source_date"], row.get("scheduled_at_utc", "")))
        if len(matches) > 1 and (matches[0]["source_date"], matches[0].get("scheduled_at_utc")) == (matches[1]["source_date"], matches[1].get("scheduled_at_utc")):
            raise ValueError("ambiguous_repeated_match_schedule")
        row = matches[0]
        if any(source_days[dependency] > row["source_date"] for dependency in dependencies):
            raise ValueError("observed_node_precedes_its_completed_ancestor")
        sa, sb = row["score_a"], row["score_b"]
        if row["team_a"] != a:
            sa, sb = sb, sa
        if max(sa, sb) != best_of // 2 + 1 or min(sa, sb) not in range(best_of // 2 + 1):
            raise ValueError("illegal_archived_series_score")
        winner, loser = (a, b) if sa > sb else (b, a)
        used.add(row["wiki_series_id"])
        completed.append({"id": row["wiki_series_id"], "stage": stage["id"], "group": "all", "round": node_id,
                          "team_a": a, "team_b": b, "best_of": best_of, "weight": 1,
                          "winner": winner, "loser": loser, "score_a": sa, "score_b": sb,
                          "source_date": row["source_date"]})
        source_days[node_id] = row["source_date"]
        return winner, loser

    for stage in spec["stages"]:
        outputs = {}
        if stage["kind"] == "gauntlet":
            entrants = [resolve(ref) for ref in stage["entrants"]]
            winner, eliminated = entrants[-1], []
            for number, challenger in enumerate(reversed(entrants[:-1]), 1):
                winner, loser = play(stage, number, challenger, winner, stage["best_of"], [number - 1] if number > 1 else [])
                eliminated.append(loser)
            rankings[stage["id"]] = [winner, *reversed(eliminated)]
        elif stage["kind"] == "graph":
            def dependencies(value):
                if isinstance(value, dict):
                    found = [value[k] for k in ("winner", "loser") if k in value]
                    return found + [item for child in value.values() if isinstance(child, list)
                                    for entry in child for item in dependencies(entry)]
                return []
            for node in stage["matches"]:
                outputs[node["id"]] = play(stage, node["id"], resolve(node["a"]), resolve(node["b"]),
                    node.get("best_of", stage["best_of"]), dependencies(node["a"]) + dependencies(node["b"]))
            rankings[stage["id"]] = [resolve(ref) for ref in stage["ranking"]]
        else:
            raise ValueError(f"non_elimination_source_recipe:{stage['kind']}")
    if used != {row["wiki_series_id"] for row in records}:
        raise ValueError(f"unconsumed_archived_phase_series:{len(records) - len(used)}")
    completed.sort(key=lambda row: (row["source_date"], row["id"]))
    return completed, rankings


def normalized_records(phase_id, reconciliation, team_ids, aliases):
    records, evidence = [], []
    for joined in reconciliation:
        if joined["phase_id"] != phase_id:
            continue
        row = joined["raw_source"]
        day = joined["event_day"]["conservative_event_day"]
        if row["status"] != "completed" or row.get("validation_errors") not in ("", "[]", None) or not day:
            raise ValueError(f"incomplete_or_invalid_source_series:{row['wiki_series_id']}")
        a, b = team_ids.get(canonical(row["team1"], aliases)), team_ids.get(canonical(row["team2"], aliases))
        if not a or not b:
            raise ValueError(f"source_series_identity_outside_declared_seeds:{row['team1']}:{row['team2']}")
        records.append({"wiki_series_id": row["wiki_series_id"], "team_a": a, "team_b": b,
                        "score_a": int(row["team1_score"]), "score_b": int(row["team2_score"]),
                        "best_of": int(row["best_of"]), "source_date": day,
                        "scheduled_at_utc": row.get("scheduled_at_utc")})
        evidence.append(joined)
    if not records:
        raise ValueError("no_joined_archived_series")
    return records, evidence


def joint_categories(teams, count):
    return [json.dumps(list(items), separators=(",", ":")) for items in combinations(sorted(teams), count)]


def declared_targets(spec, completed, rankings, boundary, placement_rows, aliases, team_ids):
    """Declare support from graph/seed contracts; historical values leave separately."""
    targets, labels, observables = [], {}, []
    start = boundary["terminal_target_start_at"]
    teams, stage = spec["teams"], spec["stages"][-1]

    def add(identifier, kind, source, categories, observed=None, **extra):
        targets.append({"target_id": identifier, "target_kind": kind, "target_family": identifier.split(":")[0],
                        "source": source, "categories": categories, "target_start_at": start, **extra})
        if observed is not None:
            labels[identifier] = observed

    if "champion" in spec:
        champion = rankings[spec["champion"]["stage"]][0]
        add("champion", "champion", "champion_prob", teams, champion)
        first = stage["ranking"][:2]
        if len(first) == 2 and "winner" in first[0] and first[1] == {"loser": first[0]["winner"]}:
            node_id = first[0]["winner"]
            spec["final"] = {"stage": stage["id"], "group": "all", "round": node_id}
            finalists = rankings[stage["id"]][:2]
            add("joint_finalists", "joint", "joint_final_prob", joint_categories(teams, 2),
                json.dumps(sorted(finalists), separators=(",", ":")))
    if stage.get("advance"):
        advanced = rankings[stage["id"]][:stage["advance"]]
        for team in teams:
            add(f"qualification:{team}", "binary", "advance_prob", ["0", "1"], str(int(team in advanced)),
                stage=stage["id"], team=team)
        add("joint_qualification", "joint", "joint_advance_prob", joint_categories(teams, stage["advance"]),
            json.dumps(sorted(advanced), separators=(",", ":")), stage=stage["id"])

    # Exact official full-phase placement tables are evidence for scoring bands,
    # not a reason to import final finish order into seeds or choose a bracket.
    official = {}
    for row in placement_rows:
        team = team_ids.get(canonical(row["team"], aliases))
        if team:
            official.setdefault(team, set()).add(str(row["placement"]).replace("–", "-"))
    placement_reason = "official_full_entrant_bands_not_supported_by_phase_ranks"
    if set(official) == set(teams) and all(len(value) == 1 for value in official.values()) and len(rankings[stage["id"]]) == len(teams):
        ordered = [next(iter(official[team])) for team in rankings[stage["id"]]]
        bands, valid = [], True
        for index, label in enumerate(ordered, 1):
            if not re.fullmatch(r"\d+(?:-\d+)?", label):
                valid = False
                break
            bounds = list(map(int, label.split("-")))
            if index not in range(bounds[0], bounds[-1] + 1):
                valid = False
                break
            if not bands or bands[-1]["id"] != label:
                bands.append({"id": label, "ranks": []})
            bands[-1]["ranks"].append(index)
        if valid and all(band["ranks"] == list(range(int(band["id"].split("-")[0]), int(band["id"].split("-")[-1]) + 1)) for band in bands):
            # LPL 2020 ranks 5/6 have an additional combined-season rule; the
            # graph's arbitrary branch rank is NOT this official placement.
            if spec["id"] not in {"lpl-2020-staggered-playoffs", "lpl-2020-spring-mid-season-cup"}:
                placement_reason = None
                for team in teams:
                    add(f"official_placement:{team}", "placement", "stage_rank_prob", [band["id"] for band in bands],
                        next(iter(official[team])), stage=stage["id"], group="all", team=team, bands=bands)
    declared_formats = [node.get("best_of", part["best_of"]) for part in spec["stages"]
                        for node in part.get("matches", [])]
    declared_formats.extend(part["best_of"] for part in spec["stages"] if part["kind"] == "gauntlet"
                            for _ in range(len(part["entrants"]) - 1))
    maximum_series, maximum_maps = len(declared_formats), sum(declared_formats)
    for kind, maximum in (("future_series_count", maximum_series), ("future_map_count", maximum_maps),
                          ("repeat_encounter_count", maximum_series), ("upset_count", maximum_series)):
        categories = [str(number) for number in range(maximum + 1)]
        observable = {"id": kind, "kind": kind, "categories": categories}
        if kind == "upset_count":
            observable.update(reference_origin_id="post-draw", reference_model="exp039", threshold=0.5)
        observables.append(observable)
        add(kind, "count", "observable_prob", categories, observable_id=kind)
    node_id = stage["matches"][-1]["id"]
    node = {"stage": stage["id"], "group": "all", "round": node_id, "match": 1}
    node_result = next(row for row in completed if row["stage"] == stage["id"] and row["round"] == node_id)
    matchup_id = "terminal_node_matchup"
    categories = joint_categories(teams, 2) + ["absent"]
    observables.append({"id": matchup_id, "kind": "node_matchup", "node": node,
                        "categories": categories, "absent_category": "absent"})
    add(matchup_id, "categorical", "observable_prob", categories,
        json.dumps(sorted([node_result["team_a"], node_result["team_b"]]), separators=(",", ":")), observable_id=matchup_id)
    score_id, participants = "terminal_node_seed1_seed2_score", teams[:2]
    wins = stage["matches"][-1].get("best_of", stage["best_of"]) // 2 + 1
    categories = [f"{wins}-{loser}" for loser in range(wins)] + [f"{loser}-{wins}" for loser in range(wins)] + ["absent"]
    observables.append({"id": score_id, "kind": "node_score", "node": node, "participants": participants,
                        "categories": categories, "absent_category": "absent"})
    observed = "absent"
    if set(participants) == {node_result["team_a"], node_result["team_b"]}:
        scores = {node_result["team_a"]: node_result["score_a"], node_result["team_b"]: node_result["score_b"]}
        observed = f"{scores[participants[0]]}-{scores[participants[1]]}"
    add(score_id, "categorical", "observable_prob", categories, observed, observable_id=score_id)
    return targets, observables, labels, placement_reason


def profile_blockers(profile):
    reasons = []
    kinds = {stage["kind"] for stage in profile["spec"]["stages"]}
    if not profile["structure_source_verified"]:
        reasons.append("profile_is_generic_scenario_not_archived_rule_graph")
    if "swiss" in kinds:
        reasons.append("swiss_pool_draw_rematch_and_residual_tie_rules_not_fully_source_supported")
    if "round_robin" in kinds:
        reasons.append("round_robin_residual_tiebreak_or_linked_draw_policy_not_historically_reconstructed")
    if profile["id"] not in PRIOR_RANK_PROFILES:
        reasons.append("source_seed_or_group_rank_to_graph_slot_contract_not_complete")
    return reasons


def prepare(output_dir, *, readiness_path=DEFAULT_READINESS, bundle=DEFAULT_BUNDLE):
    from scripts.audit_tournament_replay_readiness import build_readiness_manifest
    output, bundle, readiness_path = fresh_output(output_dir), Path(bundle).resolve(), Path(readiness_path).resolve()
    old = load(readiness_path)
    # Reconcile the explicit standard-name aliases discovered by this admission
    # pass; preserve the supplied inventory and every raw table without mutation.
    readiness = build_readiness_manifest(bundle)
    if {p["phase_id"] for p in old["phases"]} != {p["phase_id"] for p in readiness["phases"]}:
        raise ValueError("declared_readiness_phase_inventory_changed")
    write_new(output / "readiness_manifest.json", readiness)
    catalog = load(DEFAULT_CATALOG)
    profiles = {profile["id"]: profile for profile in catalog["profiles"]}
    aliases = explicit_aliases(load(bundle / "team_aliases.json"))
    standings, placements = jsonl(bundle / "standings.jsonl"), jsonl(bundle / "placements.jsonl")
    providers = jsonl(bundle / "golgg_series_index.jsonl")
    groups, brackets = jsonl(bundle / "source_group_definitions.jsonl"), jsonl(bundle / "bracket_sources.jsonl")
    global_paths = {readiness_path, DEFAULT_CATALOG.resolve(), Path(__file__).resolve(),
                    *[bundle / name for name in ("events.jsonl", "standings.jsonl", "placements.jsonl", "team_aliases.json",
                       "golgg_series_index.jsonl", "golgg_source_inventory.json", "source_group_definitions.jsonl", "bracket_sources.jsonl")]}
    global_paths.update(Path(item["path"]).resolve() for item in readiness["inputs"] if item["exists"])
    for row in standings + placements:
        if row.get("raw_path"):
            global_paths.add(bundle / "raw" / Path(row["raw_path"]).name)
    reference_hashes = {str(path): digest(path) for path in sorted(global_paths)}
    coverage = []
    for phase in readiness["phases"]:
        event, phase_id = phase["source_event"], phase["phase_id"]
        edition = next(item for item in readiness["editions"] if phase_id in item["phase_ids"])
        selected = [profiles[key] for key in phase["profile_ids"]]
        group_rows = [row for row in groups if row["title"].removeprefix("Data:") == event["title"]]
        bracket_rows = [row for row in brackets if row["title"] == event["title"]]
        prior_events = linked_prior_events(phase, readiness["phases"])
        source_scan = {"format_excerpt": phase["archived_identity_evidence"]["format_excerpt"],
                       "archived_phase_source": phase["archived_identity_evidence"],
                       "prior_phase_candidates": prior_events, "archived_group_definitions": group_rows,
                       "archived_bracket_definitions": bracket_rows,
                       "bracket_named_slot_fields": [key for row in bracket_rows for key in row["fields"] if re.search(r"_team_[12]$", key)],
                       "profile_rule_gaps": phase["catalog_gaps"]}
        entry = {"phase_id": phase_id, "tournament_id": phase["tournament_id"], "title": event["title"],
                 "family": event["family"], "year": event["year"], "phase": event["stage"],
                 "format": event.get("format_summary"), "event_status": event["status"],
                 "profile_ids": phase["profile_ids"], "source_ready": False, "captured_forecast": False,
                 "historical_availability_certified": False, "reasons": [], "source_scan": source_scan,
                 "profile_hashes": {p["id"]: object_digest(p) for p in selected},
                 "source_event_sha256": object_digest(event), "assumptions": {p["id"]: p["assumptions"] for p in selected}}
        reasons = entry["reasons"]
        if event["status"] != "completed":
            reasons.append("phase_not_completed_labels_pending_or_cancelled")
        if not selected:
            reasons.append("no_executable_source_profile")
        if not event.get("start_date") or not event.get("end_date"):
            reasons.append("announced_phase_calendar_boundary_missing")
        for profile in selected:
            reasons.extend(profile_blockers(profile))
        profile = selected[0] if len(selected) == 1 else None
        if len(selected) > 1:
            reasons.append("multiple_profile_scope_contracts_require_disambiguation")
        if profile and profile["id"] in PRIOR_RANK_PROFILES and event.get("start_date"):
            try:
                rows, prior = prior_seed_rows(event, prior_events, standings, profile["team_count"])
                forecast_day = (date.fromisoformat(event["start_date"]) - timedelta(days=1)).isoformat()
                if prior["end_date"] >= forecast_day:
                    raise ValueError("seed_source_not_completed_before_pre_phase_forecast_day")
                seed_ids, identities = [], []
                for row in rows:
                    team_id, proof = identity_at(row["team"], forecast_day, providers, aliases)
                    seed_ids.append(team_id)
                    identities.append(proof)
                if len(set(seed_ids)) != len(seed_ids):
                    raise ValueError("canonical_provider_seed_collision")
                entry["seed_evidence"] = {"source_phase": prior, "prior_standings": rows,
                                           "team_ids_in_seed_order": seed_ids, "identity_evidence": identities}
            except (ValueError, KeyError) as error:
                reasons.append(str(error))
        else:
            reasons.append("announced_complete_canonical_seed_slots_not_compiled")
        reasons[:] = list(dict.fromkeys(reasons))
        if not reasons:
            try:
                spec = instantiate_profile(profile["id"], entry["seed_evidence"]["team_ids_in_seed_order"])
                names = {canonical(row["team"], aliases): team for row, team in zip(rows, spec["teams"], strict=True)}
                records, evidence = normalized_records(phase_id, readiness["series_reconciliation"], names, aliases)
                completed, rankings = replay_elimination(spec, records)
                boundary = source_boundary(event, edition)
                source_date = re.findall(r"(?m)^\|sdate\s*=\s*(\d{4}-\d{2}-\d{2})", page_content(phase))
                if len(source_date) != 1 or source_date[0] != event["start_date"]:
                    raise ValueError("phase_start_not_independently_supported_by_archived_infobox")
                targets, observables, labels, placement_reason = declared_targets(
                    spec, completed, rankings, boundary,
                    [row for row in placements if row["title"] == event["title"]], aliases, names)
                if not targets or boundary["terminal_target_start_at"] <= boundary["phase_start_at"]:
                    raise ValueError("no_target_strictly_after_declared_phase_origin")
                source = {"version": 1, "event": {"tournament_id": entry["tournament_id"], "phase_id": phase_id,
                           "family": event["family"], "tier": "Major" if event["family"] != "eu_masters" else "Regional",
                           "format": "+".join(stage["kind"] for stage in spec["stages"]), "forecast_scope": "phase_start",
                           "phase_stages": [stage["id"] for stage in spec["stages"]], **boundary},
                          "pre_draw_cutoff": (date.fromisoformat(event["start_date"]) - timedelta(days=2)).isoformat() + "T00:00:00Z",
                          "post_draw_cutoff": boundary["phase_start_forecast_cutoff"], "spec": spec, "seeds": spec["teams"],
                          "draws": [], "completed": completed, "targets": targets, "observables": observables,
                          "evidence_files": list(reference_hashes), "evidence_hashes": reference_hashes,
                          "reconstruction_policy": {"source_dates": "normalized conservative days from readiness series_reconciliation",
                              "availability": "completed series assumed available only on next UTC day; not publication certification",
                              "rosters": "strictly prior observed roster proxy is required at bind time",
                              "pre_draw_cutoff": "previous UTC day required by compiler schema; no forecast emitted at this cutoff",
                              "edition_boundary": "unknown retained as null; not phase-as-edition", "outcome_labels_in_forecast_inputs": False}}
                relative = f"phases/{phase_id}"
                write_new(output / relative / "source.json", source)
                write_new(output / relative / "outcomes.json", {"version": VERSION, "phase_id": phase_id,
                          "tournament_id": entry["tournament_id"], "labels": labels, "completed": completed,
                          "rankings": rankings, "source_series": evidence,
                          "upset_labels": "derived only by labels command from fixed earlier exp039 reference table",
                          "forecast_input": False})
                entry.update(source_ready=True, source_path=f"{relative}/source.json", outcomes_path=f"{relative}/outcomes.json",
                             source_sha256=digest(output / relative / "source.json"), outcomes_sha256=digest(output / relative / "outcomes.json"),
                             observations={"series": len(completed), "legal_full_prefix": True, "completed_ancestor_join": True},
                             target_support={"families": dict(Counter(target["target_kind"] for target in targets)),
                                             "placement_exclusion": placement_reason,
                                             "champion_exclusion": None if "champion" in spec else "qualification_routes_have_no_championship_final"},
                             boundary=boundary)
            except (ValueError, KeyError, TypeError) as error:
                reasons.append(f"source_reconstruction_failed:{error}")
        entry["scopes"] = {scope: {"eligible": entry["source_ready"] if scope in ("phase_start", "remaining_phase") else False,
                            "reasons": reasons if not entry["source_ready"] else ([] if scope in ("phase_start", "remaining_phase")
                                else ["whole_edition_boundary_and_complete_transition_graph_unverified"])}
                           for scope in ("phase_start", "remaining_phase", "full_tournament", "remaining_tournament")}
        entry["modes"] = {"pre_draw": {"eligible": False, "reason": "no_source_supported_unknown_random_draw"},
                          **{mode: {"eligible": entry["source_ready"], "reasons": reasons} for mode in ("post_draw", "daily_rollforward", "post_round")}}
        coverage.append(entry)
    if len(coverage) != len({entry["phase_id"] for entry in coverage}) or len(coverage) != len(old["phases"]):
        raise AssertionError("phase_coverage_lost_or_duplicated")
    coverage_by_id = {entry["phase_id"]: entry for entry in coverage}
    linked_editions = []
    for edition in readiness["editions"]:
        if len(edition["phase_ids"]) < 2 or not any(coverage_by_id[key]["source_ready"] for key in edition["phase_ids"]):
            continue
        members = []
        for phase_id in edition["phase_ids"]:
            phase = next(row for row in readiness["phases"] if row["phase_id"] == phase_id)
            days = sorted({row["event_day"]["conservative_event_day"] for row in readiness["series_reconciliation"]
                           if row.get("phase_id") == phase_id and row.get("event_day", {}).get("conservative_event_day")})
            members.append({"phase_id": phase_id, "title": phase["source_event"]["title"],
                "announced_phase_start_source_date": phase["source_event"].get("start_date"),
                "announced_phase_end_source_date": phase["source_event"].get("end_date"),
                "retrospective_checkpoint_days": [(date.fromisoformat(day) + timedelta(days=1)).isoformat() for day in days],
                "source_ready_phase_forecast": coverage_by_id[phase_id]["source_ready"],
                "phase_exclusions": coverage_by_id[phase_id]["reasons"],
                "source_identity_evidence": phase["archived_identity_evidence"]})
        linked_editions.append({"tournament_id": edition["tournament_id"], "edition_start_at": edition["edition_start_at"],
            "phases": members, "qualification": "source_linked_calendar_evidence_not_a_runnable_whole_edition",
            "checkpoint_precision": "completed_source_day_available_next_UTC_day_retrospective_policy",
            "whole_edition_exclusions": ["independent_whole_edition_boundary_unverified",
                                       "complete_phase_rules_and_transition_draw_contract_unverified"]})
    write_new(output / "source_linked_editions.json", linked_editions)
    summary = {"phases": len(coverage), "source_ready_phases": sum(entry["source_ready"] for entry in coverage),
               "source_ready_editions": len({entry["tournament_id"] for entry in coverage if entry["source_ready"]}),
               "captured_forecasts": 0, "certified_whole_edition_boundaries": 0,
               "by_family": {family: {"inventory": sum(entry["family"] == family for entry in coverage),
                            "source_ready": sum(entry["family"] == family and entry["source_ready"] for entry in coverage)}
                             for family in sorted({entry["family"] for entry in coverage})},
               "exclusion_reasons": dict(Counter(reason for entry in coverage for reason in entry["reasons"]))}
    manifest = {"version": VERSION, "qualification": "retrospective_reconstruction_not_captured",
                "summary": summary, "phases": coverage, "reference_hashes": reference_hashes,
                "readiness_path": "readiness_manifest.json", "readiness_sha256": digest(output / "readiness_manifest.json"),
                "source_linked_editions_path": "source_linked_editions.json",
                "source_linked_editions_sha256": digest(output / "source_linked_editions.json"),
                "limitations": ["No minimum observed kickoff is an edition boundary.",
                    "Archived results enter only completed prefixes and separate labels, never seed slots.",
                    "Group memberships exist but do not resolve residual ties, human choices, or downstream draws.",
                    "No source-ready Swiss or full RR-to-playoff edition is asserted where source rules remain scenario assumptions."]}
    write_new(output / "coverage_manifest.json", manifest)
    return summary


def verify_references(coverage):
    for path, expected in coverage["reference_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"referenced_source_content_changed:{path}")


def bind(cohort_dir, output_dir, *, benchmark_dir, history):
    from scripts.export_prospective_features import iter_matches
    from scripts.prepare_daily_tournament_evaluation import compile_daily_tournament_evaluation
    from scripts.prospective_sports_features import ChronologicalFeatureState, _prepare_history
    from scripts.tournament_evaluation import _validate_manifest
    from src.models.tournament_formats import required_best_ofs
    from src.models.tournament_prediction import (
        prepare_pair_bundle_from_state, predict_model_pairs, fit_score_distributions, golgg_score_rows,
    )
    cohort, benchmark, history = Path(cohort_dir).resolve(), Path(benchmark_dir).resolve(), Path(history).resolve()
    coverage = load(cohort / "coverage_manifest.json")
    verify_references(coverage)
    if not (benchmark / "models").is_dir() or not history.is_file():
        raise ValueError("explicit benchmark models directory and historical source file required")
    output = fresh_output(output_dir)
    history_hash = digest(history)
    code_hashes = {str(ROOT / name): digest(ROOT / name) for name in (
        "scripts/prepare_tournament_historical_manifest.py", "src/models/tournament_prediction.py",
        "scripts/prospective_sports_features.py", "scripts/prepare_daily_tournament_evaluation.py",
    )}
    events, failures, artifact_hashes, cache, target_exclusions = [], [], {}, {}, []
    phases, requests, matrices = {}, [], {}
    for phase in coverage["phases"]:
        if not phase["source_ready"]:
            continue
        source_path = cohort / phase["source_path"]
        if digest(source_path) != phase["source_sha256"]:
            raise ValueError("immutable_prepared_source_changed")
        compiled = output / "compiled" / phase["phase_id"]
        compile_daily_tournament_evaluation(source_path, compiled)
        event = load(compiled / "event_manifest.json")["event"]
        spec = load(compiled / "spec.json")
        event["spec"] = str(compiled / "spec.json")
        phases[phase["phase_id"]] = (phase, spec, source_path)
        for origin in event["origins"]:
            if origin.get("state"):
                origin["state"] = str(compiled / origin["state"])
            if origin["forecast_mode"] == "daily_rollforward":
                origin["axis"] = "axis3"
                origin["reference_origin_id"] = "post-draw"
        requests.extend((cutoff, phase["phase_id"]) for cutoff in sorted({row["cutoff"] for row in event["origins"]}))
        events.append(event)
    _validate_manifest({"version": "tournament-evaluation-v3", "qualification": "retrospective_reconstruction",
        "models": [{"id": name, "kind": "series_table"} for name in MODEL_FILES], "events": events})
    requests.sort()
    state, position = ChronologicalFeatureState(), 0
    if requests:
        last_cutoff = date.fromisoformat(requests[-1][0][:10])
        series, source_audit = _prepare_history(iter_matches(history), last_cutoff)
        score_rows = list(golgg_score_rows(iter_matches(history), last_cutoff))
        identifiers = {item.order: item.identifier for item in series}
    else:
        series, score_rows, identifiers, source_audit = [], [], {}, {}
    for cutoff, phase_id in requests:
        phase, spec, source_path = phases[phase_id]
        formats = required_best_ofs(spec)
        year = int(phase["boundary"]["phase_start_at"][:4])
        at = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        while position < len(series) and series[position].games[-1].day < at.date():
            state.consume(series[position])
            position += 1
        directory = output / "matrices" / phase_id / cutoff[:10]
        directory.mkdir(parents=True)
        tables, exclusions = {}, {}
        try:
            missing = sorted(set(spec["teams"]) - state.latest_rosters.keys())
            if missing:
                raise ValueError(f"no prior observed roster for {missing}")
            rosters = {team: state.latest_rosters[team][1] for team in spec["teams"]}
            proxy = {team: {"date": state.latest_rosters[team][0][0].isoformat(),
                           "match_id": identifiers[state.latest_rosters[team][0][1]],
                           "game_index": state.latest_rosters[team][0][2]} for team in spec["teams"]}
            pairs = prepare_pair_bundle_from_state(state, rosters, cutoff=cutoff, best_ofs=formats,
                input_hashes={str(history): history_hash, str(source_path): phase["source_sha256"]},
                roster_provenance={"policy": "previous_observed_roster_proxy", "observations": proxy},
                previous_roster_observations=proxy)
            write_new(directory / "pairs.json", pairs)
        except (ValueError, KeyError, TypeError) as error:
            pairs = None
            exclusions = {name: f"all_pair_export_failed:{error}" for name in MODEL_FILES}
        for name, (kind, filename) in MODEL_FILES.items():
            artifact_path = benchmark / "models" / str(year) / filename
            if not artifact_path.is_file():
                exclusions[name] = f"annual_benchmark_artifact_missing:{artifact_path}"
                continue
            artifact_hashes[str(artifact_path)] = digest(artifact_path)
            if pairs is None:
                continue
            try:
                key = (year, name)
                if key not in cache:
                    if artifact_path.suffix == ".joblib":
                        import joblib
                        cache[key] = joblib.load(artifact_path)
                    else:
                        cache[key] = load(artifact_path)
                probabilities, provenance = predict_model_pairs(cache[key], pairs, model_kind=kind,
                    artifact_sha256=artifact_hashes[str(artifact_path)], teams=spec["teams"], best_ofs=formats,
                    cutoff=cutoff, mode="chronological")
                provenance.update(model_id=name, model_artifact_sha256=artifact_hashes[str(artifact_path)])
                table_path = directory / f"{name}.json"
                write_new(table_path, {"probability_unit": "series_win", "rows": [
                    {"team_a": a, "team_b": b, "best_of": bo, "p": value}
                    for (a, b, bo), value in probabilities.items()], "provenance": provenance})
                tables[name] = str(table_path)
            except (ValueError, KeyError, TypeError) as error:
                exclusions[name] = f"annual_model_contract_or_inference_failed:{error}"
        distributions, score_provenance = fit_score_distributions(score_rows, cutoff=at.date(), best_ofs=formats)
        score_path = directory / "score_model.json"
        write_new(score_path, {"distributions": distributions, "provenance": score_provenance})
        matrices[cutoff, phase_id] = {"tables": tables, "model_exclusions": exclusions, "score_model": str(score_path)}
        failures.extend({"phase_id": phase_id, "cutoff": cutoff, "model_id": name, "reason": reason}
                        for name, reason in exclusions.items())
        print(json.dumps({"phase_id": phase_id, "cutoff": cutoff, "models": len(tables),
                          "consumed_series": position}), flush=True)
    for event in events:
        for origin in event["origins"]:
            origin.update(deepcopy(matrices[origin["cutoff"], event["phase_id"]]))
        reference = next(origin for origin in event["origins"] if origin["origin_id"] == "post-draw")
        if "exp039" not in reference["tables"]:
            reason = "declared_fixed_exp039_upset_reference_unavailable_at_phase_start"
            target_exclusions.append({"phase_id": event["phase_id"], "target_id": "upset_count", "reason": reason})
            for origin in event["origins"]:
                origin["observables"] = [row for row in origin["observables"] if row["id"] != "upset_count"]
                origin["targets"] = [row for row in origin["targets"] if row["target_id"] != "upset_count"]
    verify_references(coverage)
    if digest(history) != history_hash or any(digest(path) != value for path, value in {**artifact_hashes, **code_hashes}.items()):
        raise ValueError("history_benchmark_or_binding_code_changed")
    manifest = {"version": "tournament-evaluation-v3", "qualification": "retrospective_reconstruction",
                "models": [{"id": name, "kind": "series_table"} for name in MODEL_FILES], "events": events,
                "evidence_files": [str(cohort / "coverage_manifest.json"), str(history), *artifact_hashes,
                                   *coverage["reference_hashes"], *code_hashes,
                                   *[str(cohort / phase["source_path"]) for phase in coverage["phases"] if phase["source_ready"]]],
                "evidence_hashes": {str(cohort / "coverage_manifest.json"): digest(cohort / "coverage_manifest.json"),
                                    str(history): history_hash, **artifact_hashes, **coverage["reference_hashes"], **code_hashes},
                "exclusions": [{"phase_id": phase["phase_id"], "reasons": phase["reasons"]}
                               for phase in coverage["phases"] if not phase["source_ready"]],
                "model_failures": failures, "target_exclusions": target_exclusions}
    write_new(output / "evaluator_manifest.json", manifest)
    write_new(output / "binding_summary.json", {"events": len(events), "origins": sum(len(event["origins"]) for event in events),
        "model_failures": failures, "outcomes_read": False, "unique_phase_cutoffs": len(requests),
        "history_series_consumed_once": position, "source_admission": source_audit, "code_sha256": code_hashes})
    return {"events": len(events), "model_failures": len(failures), "manifest": str(output / "evaluator_manifest.json")}


def labels(cohort_dir, bound_dir, output_path):
    cohort, bound = Path(cohort_dir).resolve(), Path(bound_dir).resolve()
    output_path = Path(output_path).resolve()
    if not output_path.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError("outcomes must remain under new roadmap artifact root")
    coverage, manifest = load(cohort / "coverage_manifest.json"), load(bound / "evaluator_manifest.json")
    entries = {phase["phase_id"]: phase for phase in coverage["phases"]}
    outcomes, exclusions = [], []
    for event in manifest["events"]:
        entry = entries[event["phase_id"]]
        outcome_path = cohort / entry["outcomes_path"]
        if digest(outcome_path) != entry["outcomes_sha256"]:
            raise ValueError("scoring_source_outcomes_changed")
        historical = load(outcome_path)
        initial = next(origin for origin in event["origins"] if origin["origin_id"] == "post-draw")
        reference = load(initial["tables"]["exp039"]) if "exp039" in initial.get("tables", {}) else None
        reference_pairs = {(row["team_a"], row["team_b"], row["best_of"]): row["p"] for row in reference["rows"]} if reference else {}
        for origin in event["origins"]:
            known = {row["id"] for row in historical["completed"] if row["source_date"] < origin["cutoff"][:10]}
            future = [row for row in historical["completed"] if row["id"] not in known]
            observed = dict(historical["labels"])
            observed.update(future_series_count=str(len(future)), future_map_count=str(sum(row["score_a"] + row["score_b"] for row in future)))
            seen, repeats = Counter(), 0
            for row in sorted(historical["completed"], key=lambda item: (item["source_date"], item["id"])):
                pair = tuple(sorted([row["team_a"], row["team_b"]]))
                if row["id"] not in known and seen[pair]:
                    repeats += 1
                seen[pair] += 1
            observed["repeat_encounter_count"] = str(repeats)
            if reference:
                upsets = 0
                for row in future:
                    key = (row["winner"], row["loser"], row["best_of"])
                    p = reference_pairs[key] if key in reference_pairs else 1 - reference_pairs[key[1], key[0], key[2]]
                    upsets += p < 0.5
                observed["upset_count"] = str(upsets)
            for target in origin["targets"]:
                identifier = target["target_id"]
                if identifier not in observed:
                    exclusions.append({"phase_id": event["phase_id"], "origin_id": origin["origin_id"], "target_id": identifier,
                                       "reason": "fixed_earlier_exp039_reference_missing"})
                    continue
                outcomes.append({"tournament_id": event["tournament_id"], "phase_id": event["phase_id"],
                                 "origin_id": origin["origin_id"], "target_id": identifier, "observed": observed[identifier]})
    write_new(output_path, outcomes)
    write_new(output_path.with_suffix(".metadata.json"), {
        "version": VERSION, "qualification": "retrospective_reconstruction",
        "outcomes_sha256": digest(output_path), "exclusions": exclusions, "forecast_input": False,
        "fixed_upset_reference": "exp039 table at the same phase's post-draw origin",
    })
    return {"outcomes": len(outcomes), "exclusions": len(exclusions), "path": str(output_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("prepare")
    source.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    source.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    source.add_argument("--output-dir", type=Path, required=True)
    binding = commands.add_parser("bind")
    binding.add_argument("--cohort-dir", type=Path, required=True)
    binding.add_argument("--benchmark-dir", type=Path, required=True)
    binding.add_argument("--history", type=Path, required=True)
    binding.add_argument("--output-dir", type=Path, required=True)
    scoring = commands.add_parser("labels")
    scoring.add_argument("--cohort-dir", type=Path, required=True)
    scoring.add_argument("--bound-dir", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_dir, readiness_path=args.readiness, bundle=args.bundle)
    elif args.command == "bind":
        result = bind(args.cohort_dir, args.output_dir, benchmark_dir=args.benchmark_dir, history=args.history)
    else:
        result = labels(args.cohort_dir, args.bound_dir, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
