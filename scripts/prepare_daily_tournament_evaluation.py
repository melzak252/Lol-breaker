#!/usr/bin/env python3
"""Compile source-backed daily tournament-evaluation artifacts without modelling.

Input schema (JSON object)::

    {
      "version": 1,
      "event": {
        "tournament_id": "unique edition", "phase_id": "unique phase",
        "edition_start_at": "UTC timestamp", "family": "...", "tier": "...",
        "format": "...", "forecast_scope": "full_tournament|phase_start|"
                          "remaining_tournament|remaining_phase",
        "phase_start_at": "UTC timestamp (required for phase scopes)",
        "phase_stages": ["spec stage IDs belonging to this phase"]
      },
      "pre_draw_cutoff": "YYYY-MM-DDT00:00:00Z",
      "post_draw_cutoff": "YYYY-MM-DDT00:00:00Z",
      "spec": {"version": 1, "teams": ["declared entrants", "..."], ...},
      "seeds": ["every declared entrant once, in declared seed order"],
      "draws": [{"id": "source ID", "source_date": "YYYY-MM-DD", "stage": "...",
                 "group": "...", "round": 1, "pairs": [["A", "B"], ...]}],
      "completed": [{"id": "source ID", "source_date": "YYYY-MM-DD",
                     "stage": "...", "group": "...", "round": 1,
                     "team_a": "A", "team_b": "B", "best_of": 3, "weight": 1,
                     "winner": "A", "loser": "B", "score_a": 2, "score_b": 0}],
      "targets": [{"target_id": "...", "target_kind": "...", "source": "...",
                   "target_start_at": "UTC timestamp", ...}]
    }

``draws`` may be empty only for deterministic seeded/graph specifications.

``completed`` must be ordered by ``(source_date, id)``.  Source calendar days
are retrospective reconstruction evidence, not availability timestamps.  The
compiler never reads labels or invokes a model.

A new output directory is atomically published and contains::

    event_manifest.json       normalized evaluator event fragment
    origins.json              chronological pre_draw/post_draw/daily/post_round schedule
    states/YYYY-MM-DD.json    daily_rollforward state at each UTC midnight
    outcomes.template.json    separate, deliberately unlabeled outcome join template
    source_hashes.json        SHA-256 digests of supplied source sections
    readiness_ledger.json     inclusion/exclusion evidence by origin
    spec.json                 normalized declared simulator specification

Daily state rows carry ``source_date`` and no fabricated completion or
availability timestamp.  At cutoff ``D 00:00 UTC`` they include only records
whose source calendar day is strictly before ``D``.  ``pre_draw`` is emitted
only for a spec-declared uniform unknown draw and has no published pairing
state.  Deterministic seeded/graph specifications omit ``pre_draw`` and emit a
state-free ``post_draw`` with ``draw_knowledge.status="deterministic_spec"``.

With ``--model-config`` supply a runner v3 manifest containing explicit models
and per-origin ``pairs``, ``tables``, ``score_model`` or ``model_exclusions``.
Bindings are joined on edition and phase IDs, exact UTC cutoff, and start-versus-
current matrix role (axis1 versus axes2/3). Multiple bindings for a key must agree.
The configured daily forecast axis is preserved at each cutoff, including
axis3 frozen-versus-refreshed comparisons; graph checkpoints retain axis3.
Missing bindings are rejected, never replaced with a start matrix. The compiler
reads configuration only (not model artifacts
or outcomes), rebases its paths, and writes ``evaluator_manifest.json`` validated
by the runner. Thus frozen and refreshed input policies remain explicit.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


VERSION = "daily-tournament-evaluation-preparation-v2"
QUALIFICATION = "retrospective_reconstruction"
MODES = {"pre_draw", "post_draw", "daily_rollforward", "post_round"}
SCOPES = {"full_tournament", "phase_start", "remaining_tournament", "remaining_phase"}
TRACE_FIELDS = ("stage", "group", "round", "team_a", "team_b", "best_of", "weight",
                "winner", "loser", "score_a", "score_b")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"input must be readable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("phase input must be a JSON object")
    return payload


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or "T" not in value:
        raise ValueError(f"{field} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be a valid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _midnight(value: Any, field: str) -> datetime:
    parsed = _utc(value, field)
    if parsed.time() != time.min:
        raise ValueError(f"{field} must be exactly 00:00:00 UTC")
    return parsed


def _source_day(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO UTC calendar day")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO UTC calendar day") from error


def _iso_midnight(day: date) -> str:
    return datetime.combine(day, time.min, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _required_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires nonempty {key}")
    return value


def _validate_source_rows(rows: Any, name: str, required: tuple[str, ...]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be an explicit list")
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    order: list[tuple[date, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{index}] must be an object")
        item = dict(row)
        identifier = _required_string(item, "id", f"{name}[{index}]")
        if identifier in identifiers:
            raise ValueError(f"duplicate {name} source id: {identifier}")
        identifiers.add(identifier)
        day = _source_day(item.get("source_date"), f"{name}[{index}].source_date")
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"{name}[{index}] missing required source facts: {missing}")
        result.append(item)
        order.append((day, identifier))
    if order != sorted(order):
        raise ValueError(f"{name} source records must be sorted by source_date then id")
    return result


def _validate_input(payload: dict[str, Any]) -> tuple[dict[str, Any], datetime, datetime, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if payload.get("version") != 1:
        raise ValueError("phase input requires version=1")
    event = payload.get("event")
    if not isinstance(event, dict):
        raise ValueError("phase input requires event object")
    for key in ("tournament_id", "phase_id", "family", "tier", "format"):
        _required_string(event, key, "event")
    if event.get("forecast_scope") not in SCOPES:
        raise ValueError("event.forecast_scope must be explicit and recognized")
    edition_start = (_utc(event["edition_start_at"], "event.edition_start_at")
                     if event.get("edition_start_at") is not None else None)
    if edition_start is None and (
            event.get("edition_start_status") != "unverified_whole_edition_boundary"
            or event["forecast_scope"] not in {"phase_start", "remaining_phase"}):
        raise ValueError("unknown edition start is permitted only for explicitly unverified phase-scoped forecasts")
    if event.get("forecast_scope") in {"phase_start", "remaining_phase"}:
        phase_start = _utc(event.get("phase_start_at"), "event.phase_start_at")
        if edition_start is not None and phase_start < edition_start:
            raise ValueError("phase_start_at cannot precede edition_start_at")
    pre_cutoff = _midnight(payload.get("pre_draw_cutoff"), "pre_draw_cutoff")
    post_cutoff = _midnight(payload.get("post_draw_cutoff"), "post_draw_cutoff")
    if pre_cutoff >= post_cutoff:
        raise ValueError("pre_draw_cutoff must be earlier than post_draw_cutoff")

    spec = payload.get("spec")
    if not isinstance(spec, dict) or spec.get("version") != 1:
        raise ValueError("spec requires version=1")
    from src.models.tournament_formats import validate_tournament_spec
    validate_tournament_spec(spec)
    if event.get("phase_stages") is not None:
        phase_stages = event["phase_stages"]
        known_stages = {stage["id"] for stage in spec["stages"]}
        if (not isinstance(phase_stages, list) or not phase_stages
                or any(not isinstance(stage, str) for stage in phase_stages)
                or len(set(phase_stages)) != len(phase_stages)
                or not set(phase_stages) <= known_stages):
            raise ValueError("phase_stages must identify distinct declared spec stages")
    teams = spec.get("teams")
    if (not isinstance(teams, list) or not teams or any(not isinstance(team, str) or not team for team in teams)
            or len(set(teams)) != len(teams)):
        raise ValueError("spec.teams must be unique declared entrant IDs")
    seeds = payload.get("seeds")
    if not isinstance(seeds, list) or any(not isinstance(team, str) for team in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be an ordered list of unique declared entrant IDs")
    if seeds != teams:
        raise ValueError("predeclared entrants/seeds contradict spec.teams")

    draws = _validate_source_rows(payload.get("draws"), "draws", ("stage", "group", "round", "pairs"))
    completed = _validate_source_rows(payload.get("completed"), "completed", TRACE_FIELDS)
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("targets must be a nonempty explicit list")
    target_ids: set[str] = set()
    normalized_targets: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"targets[{index}] must be an object")
        item = dict(target)
        target_id = _required_string(item, "target_id", f"targets[{index}]")
        if target_id in target_ids:
            raise ValueError(f"duplicate target_id: {target_id}")
        target_ids.add(target_id)
        _required_string(item, "target_kind", f"targets[{index}]")
        _required_string(item, "source", f"targets[{index}]")
        _utc(item.get("target_start_at"), f"targets[{index}].target_start_at")
        if any(key in item for key in ("outcome", "label", "result")):
            raise ValueError("forecast targets must not contain outcome labels")
        normalized_targets.append(item)

    _validate_completed(completed, teams)
    _validate_draws(draws, spec, event, completed, post_cutoff)
    return event, pre_cutoff, post_cutoff, draws, completed, normalized_targets


def _validate_draws(draws: list[dict[str, Any]], spec: dict[str, Any], event: dict[str, Any],
                    completed: list[dict[str, Any]], post_cutoff: datetime) -> None:
    teams = spec["teams"]
    stages = {stage["id"]: stage for stage in spec["stages"]}
    known_blocks: set[tuple[Any, Any, Any]] = set()
    for draw in draws:
        key = draw["stage"], draw["group"], draw["round"]
        if key in known_blocks:
            raise ValueError("contradictory predeclared pairing facts for one round/block")
        known_blocks.add(key)
        pairs = draw["pairs"]
        if (draw["stage"] not in stages or not isinstance(draw["group"], str) or not draw["group"]
                or isinstance(draw["round"], bool) or not isinstance(draw["round"], (str, int))
                or not isinstance(pairs, list) or not pairs):
            raise ValueError("draws require a complete stage/group/round pairing block")
        participants: list[str] = []
        for pair in pairs:
            if (not isinstance(pair, list) or len(pair) != 2 or any(team not in teams for team in pair)
                    or pair[0] == pair[1]):
                raise ValueError("draw pairing contradicts declared entrants")
            participants.extend(pair)
        if len(set(participants)) != len(participants):
            raise ValueError("draw pairing duplicates a declared entrant")
    selected = set(event.get("phase_stages", stages))
    visible = [draw for draw in draws if _source_day(draw["source_date"], "draw.source_date") < post_cutoff.date()]
    prior = [row for row in completed if _source_day(row["source_date"], "source_date") < post_cutoff.date()]
    _, entrants = _source_facts(spec, prior, visible)
    for stage_id in selected:
        stage = stages[stage_id]
        if stage.get("draw", {}).get("policy") != "uniform":
            continue
        groups = entrants.get(stage_id)
        if groups is None:
            if event["forecast_scope"] in {"phase_start", "remaining_phase"}:
                raise ValueError("phase draw requires completed ancestor entrant evidence")
            continue
        first = [draw for draw in visible if draw["stage"] == stage_id and str(draw["round"]) == "1"]
        if not first:
            raise ValueError("post_draw requires source-backed published first-round pairings before its cutoff")
        actual = [team for draw in first for pair in draw["pairs"] for team in pair]
        expected = [team for group in groups.values() for team in group]
        bye_count = (1 << (len(expected) - 1).bit_length()) - len(expected)
        if stage["kind"] in {"single_elimination", "double_elimination"} and bye_count:
            byes = ([expected[stage["entrants"].index(ref)] for ref in stage["byes"]]
                    if "byes" in stage else expected[:bye_count])
            expected = [team for team in expected if team not in byes]
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise ValueError("first-round pairings contradict phase entrants/seeds")


def _validate_completed(completed: list[dict[str, Any]], teams: list[str]) -> None:
    identities: set[tuple[Any, Any, Any, frozenset[str]]] = set()
    for row in completed:
        if (not isinstance(row["stage"], str) or not isinstance(row["group"], str) or not row["group"]
                or isinstance(row["round"], bool) or not isinstance(row["round"], (str, int))):
            raise ValueError("completed source requires graph node identity")
        a, b, winner, loser = row["team_a"], row["team_b"], row["winner"], row["loser"]
        if not all(isinstance(team, str) and team in teams for team in (a, b, winner, loser)) or a == b or {winner, loser} != {a, b}:
            raise ValueError("completed source contradicts declared entrants or sides")
        if any(isinstance(row[key], bool) or not isinstance(row[key], int) for key in ("best_of", "weight", "score_a", "score_b")):
            raise ValueError("completed source requires integer best_of, weight and score")
        needed = row["best_of"] // 2 + 1
        if row["best_of"] < 1 or row["best_of"] % 2 != 1 or row["weight"] < 1 or min(row["score_a"], row["score_b"]) < 0 or max(row["score_a"], row["score_b"]) != needed or min(row["score_a"], row["score_b"]) >= needed or ((row["score_a"] > row["score_b"]) != (winner == a)):
            raise ValueError("completed source requires a legal actual series score")
        identity = row["stage"], row["group"], row["round"], frozenset((a, b))
        if identity in identities:
            raise ValueError("duplicate completed series graph identity")
        identities.add(identity)


def _draw_policy(spec: dict[str, Any]) -> dict[str, Any]:
    policies = []
    for stage in spec.get("stages", []):
        if isinstance(stage, dict) and "draw" in stage:
            policies.append({"stage": stage.get("id"), "draw": stage["draw"]})
    return {"policy": "spec_declared", "declared_draws": policies}

def _has_uniform_unknown_draw(spec: dict[str, Any]) -> bool:
    return any(isinstance(stage, dict) and isinstance(stage.get("draw"), dict)
               and stage["draw"].get("policy") == "uniform"
               for stage in spec.get("stages", []))


def _state(cutoff: datetime, draws: list[dict[str, Any]], completed: list[dict[str, Any]]) -> dict[str, Any]:
    day = cutoff.date()
    def visible(row: dict[str, Any]) -> bool:
        return _source_day(row["source_date"], "source_date") < day
    # IDs stay in the evidence ledger; state rows have only the simulator trace
    # identity/facts plus a source calendar day under daily_rollforward.
    state_completed = [{key: row[key] for key in TRACE_FIELDS} | {"source_date": row["source_date"]}
                       for row in completed if visible(row)]
    state_pairings = [{key: row[key] for key in ("stage", "group", "round", "pairs", "source_date")}
                      for row in draws if visible(row)]
    return {"version": 1, "cutoff": _iso_midnight(day), "temporal_policy": "daily_rollforward",
            "completed": state_completed, "pairings": state_pairings, "orders": [], "group_orders": []}


def _eligible_targets(targets: list[dict[str, Any]], cutoff: datetime) -> list[dict[str, Any]]:
    eligible = [dict(target) for target in targets if _utc(target["target_start_at"], "target_start_at") > cutoff]
    if not eligible:
        return []
    return eligible


def _source_facts(spec: dict[str, Any], completed: list[dict[str, Any]],
                  draws: list[dict[str, Any]]) -> tuple[list, dict]:
    """Replay facts with engine semantics; never sample missing matches or ties."""
    from src.models.tournament_formats import _Checkpoint, _Run, _groups

    class Unresolved(Exception):
        pass

    class FactRandom:
        def shuffle(self, values):
            raise Unresolved

    records, entrants = [], {}

    class FactRun(_Run):
        def resolve(self, ref):
            if not isinstance(ref, str) and (
                    ref["stage"] not in self.rankings or self.stage_uncertain.get(ref["stage"])):
                raise Unresolved
            return super().resolve(ref)

        def match(self, a, b, round_number, group="all", best_of=None, weight=1):
            if a is not None and b is not None and (
                    self.stage["id"], group, round_number, frozenset((a, b))) not in self.checkpoint.completed:
                raise Unresolved
            return super().match(a, b, round_number, group, best_of, weight)

        def draw(self, teams, draw, source_groups, no_rematches=False, records=None, round_number=1):
            if (self.stage["id"], "all", round_number) not in self.checkpoint.pairings:
                raise Unresolved
            return super().draw(teams, draw, source_groups, no_rematches, records, round_number)

        def graph(self):
            # The engine's graph resolver is nested. Match its literal, rank,
            # winner/loser, seeded-pool and explicit-null supply forms exactly.
            outputs = {}
            seed_order = self.index if self.stage.get("seed_order") == "tournament" else self.seed_order
            unknown = object()

            def resolve(slot):
                if slot is None:
                    return None, None
                if isinstance(slot, dict) and "seeded" in slot:
                    pool = [resolve(source) for source in slot["seeded"]]
                    if any(team is unknown for team, _ in pool):
                        return unknown, None
                    ordered = sorted(pool, key=lambda item: seed_order[item[0]])
                    times = [available for _, available in pool if available is not None]
                    return ordered[slot["rank"] - 1][0], max(times, default=None)
                if isinstance(slot, dict) and set(slot) in ({"winner"}, {"loser"}):
                    outcome, node = next(iter(slot.items()))
                    return outputs.get(node, {}).get(outcome, (unknown, None))
                try:
                    team = self.resolve(slot)
                except Unresolved:
                    return unknown, None
                return team, self.team_available.get(team)

            by_node = {}
            nodes = {node["id"] for node in self.stage["matches"]}
            for row in completed:
                if row["stage"] == self.stage["id"]:
                    if row["round"] not in nodes or row["group"] != "all":
                        raise ValueError("completed graph record names an undeclared node or group")
                    by_node.setdefault(row["round"], []).append(row)
            for node in self.stage["matches"]:
                supplies = [resolve(node["a"]), resolve(node["b"])]
                sides = [team for team, _ in supplies]
                expected = frozenset(sides) if all(team is not unknown and team is not None for team in sides) else None
                rows = by_node.get(node["id"], [])
                records.append(((self.stage["id"], "all", node["id"]), rows, expected))
                if rows:
                    if len(rows) != 1 or expected is None:
                        raise ValueError("completed graph node requires one record and completed ancestor evidence")
                    row = rows[0]
                    if (frozenset((row["team_a"], row["team_b"])) != expected
                            or row["best_of"] != node.get("best_of", self.stage["best_of"]) or row["weight"] != 1):
                        raise ValueError("completed graph record contradicts declared sides or ancestor path")
                    if any(available is not None and available > _source_day(row["source_date"], "source_date")
                           for _, available in supplies):
                        raise ValueError("completed graph ancestor source day follows its descendant")
                    winner, loser = self.match(*sides, node["id"], best_of=node.get("best_of"))
                    available = _source_day(row["source_date"], "source_date")
                    outputs[node["id"]] = {"winner": (winner, available), "loser": (loser, available)}
                elif None in sides and unknown not in sides:
                    outputs[node["id"]] = {"winner": next(supply for supply in supplies if supply[0] is not None)}
                else:
                    self.future_series += 1
            ranking = [resolve(slot)[0] for slot in self.stage["ranking"]]
            if unknown in ranking:
                raise Unresolved
            return ranking

    last_day = max((_source_day(row["source_date"], "source_date") for row in [*completed, *draws]),
                   default=date(1970, 1, 1))
    cutoff = datetime.combine(last_day + timedelta(days=1), time.min, tzinfo=timezone.utc)
    run = FactRun(spec, {}, {}, FactRandom(), False, validate_state=True)
    for stage in spec["stages"]:
        stage_completed = [row for row in completed if row["stage"] == stage["id"]]
        stage_draws = [row for row in draws if row["stage"] == stage["id"]]
        run.checkpoint = _Checkpoint(_state(cutoff, stage_draws, stage_completed), spec)
        run.stage = stage
        run.used_completed, run.used_pairings, run.used_orders, run.used_group_orders = set(), set(), set(), set()
        run.spec = {**spec, "stages": [stage]}
        if stage.get("carry_from") and run.stage_uncertain.get(stage["carry_from"], True):
            if stage_completed or stage_draws:
                raise ValueError("completed carryover stage requires completed ancestor evidence")
            run.stage_uncertain[stage["id"]] = True
            continue
        try:
            entrants[stage["id"]] = {name: [run.resolve(ref) for ref in refs]
                                     for name, refs in _groups(stage).items()}
            run.run()
        except Unresolved:
            run.stage_uncertain[stage["id"]] = True
            if stage_completed and stage["id"] not in entrants:
                raise ValueError("completed graph node requires completed ancestor evidence")
            if stage_draws and stage["id"] not in entrants:
                raise ValueError("published draw requires completed ancestor entrant evidence")
    return records, entrants


def _graph_node_records(spec: dict[str, Any], completed: list[dict[str, Any]],
                        draws: list[dict[str, Any]]) -> list:
    return _source_facts(spec, completed, draws)[0]


def _round_checkpoints(spec: dict[str, Any], draws: list[dict[str, Any]],
                       completed: list[dict[str, Any]]) -> list[tuple[tuple[Any, Any, Any], datetime]]:
    by_block: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in completed:
        by_block.setdefault((row["stage"], row["group"], row["round"]), []).append(row)
    checkpoints: dict[tuple[Any, Any, Any], datetime] = {}
    draw_pairs = {(draw["stage"], draw["group"], draw["round"]): {frozenset(pair) for pair in draw["pairs"]}
                  for draw in draws}
    for block, rows in by_block.items():
        if block not in draw_pairs:
            continue
        completed_pairs = {frozenset((row["team_a"], row["team_b"])) for row in rows}
        if completed_pairs == draw_pairs[block]:
            cutoff_day = max(_source_day(row["source_date"], "source_date") for row in rows) + timedelta(days=1)
            checkpoints[block] = datetime.combine(cutoff_day, time.min, tzinfo=timezone.utc)
    for block, rows, expected_pair in _graph_node_records(spec, completed, draws):
        if (len(rows) == 1 and frozenset((rows[0]["team_a"], rows[0]["team_b"])) == expected_pair):
            cutoff_day = _source_day(rows[0]["source_date"], "source_date") + timedelta(days=1)
            checkpoints[block] = datetime.combine(cutoff_day, time.min, tzinfo=timezone.utc)
    return sorted(checkpoints.items(), key=lambda item: (item[1], tuple(map(str, item[0]))))


def _origin(origin_id: str, mode: str, axis: str, cutoff: datetime, targets: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    if mode not in MODES:
        raise AssertionError(f"unsupported compiler mode: {mode}")
    if not targets:
        raise ValueError(f"unresolved target starts before or at {mode} cutoff")
    return {"origin_id": origin_id, "forecast_mode": mode, "axis": axis, "cutoff": _iso_midnight(cutoff.date()),
            "forecast_scope": extra.pop("forecast_scope"), "targets": targets, **extra}


def _schedule(event: dict[str, Any], pre_cutoff: datetime, post_cutoff: datetime, draws: list[dict[str, Any]],
              completed: list[dict[str, Any]], targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    scope = event["forecast_scope"]
    policy = _draw_policy(event["spec"])
    selected_stages = set(event.get("phase_stages", [stage["id"] for stage in event["spec"]["stages"]]))
    phase_spec = {**event["spec"], "stages": [stage for stage in event["spec"]["stages"] if stage["id"] in selected_stages]}
    has_pre_draw = _has_uniform_unknown_draw(phase_spec)
    remaining_scope = {"full_tournament": "remaining_tournament", "phase_start": "remaining_phase"}.get(scope, scope)
    schedule = []
    if has_pre_draw:
        schedule.append(_origin("pre-draw", "pre_draw", "axis1", pre_cutoff,
                                _eligible_targets(targets, pre_cutoff), forecast_scope=scope,
                                draw_knowledge={"status": "unknown", "simulation_policy": policy}))
    states: dict[str, dict[str, Any]] = {}

    def add_state(cutoff: datetime) -> str:
        name = f"states/{cutoff.date().isoformat()}.json"
        states[name] = _state(cutoff, draws, completed)
        return name
    if has_pre_draw and scope == "phase_start":
        prior = _state(pre_cutoff, draws, completed)
        if prior["completed"] or prior["pairings"]:
            schedule[0]["state"] = add_state(pre_cutoff)

    post_extra = {
        "draw_knowledge": ({"status": "published_first_round", "simulation_policy": policy}
                           if has_pre_draw else {"status": "deterministic_spec"}),
    }
    if has_pre_draw or scope == "phase_start" and (_state(post_cutoff, draws, completed)["completed"]):
        post_extra["state"] = add_state(post_cutoff)
    schedule.append(_origin("post-draw", "post_draw", "axis2" if has_pre_draw else "axis1",
                            post_cutoff, _eligible_targets(targets, post_cutoff), forecast_scope=scope,
                            **post_extra))
    last_start = max(_utc(target["target_start_at"], "target_start_at") for target in targets)
    current = post_cutoff
    while any(_utc(target["target_start_at"], "target_start_at") > current for target in targets):
        state_name = add_state(current)
        schedule.append(_origin(f"daily-{current.date().isoformat()}", "daily_rollforward", "axis2", current,
                                _eligible_targets(targets, current), forecast_scope=remaining_scope, state=state_name,
                                state_temporal_policy="daily_rollforward"))
        current += timedelta(days=1)
        if current > last_start + timedelta(days=1):
            raise AssertionError("daily schedule exceeded final target")
    for block, cutoff in _round_checkpoints(event["spec"], draws, completed):
        checkpoint_targets = _eligible_targets(targets, cutoff)
        if checkpoint_targets:
            state_name = add_state(cutoff)
            stage, group, round_number = block
            if cutoff < post_cutoff or stage not in selected_stages:
                continue
            schedule.append(_origin(f"post-round-{stage}-{group}-{round_number}-{cutoff.date().isoformat()}",
                                    "post_round", "axis3", cutoff, checkpoint_targets,
                                    forecast_scope=remaining_scope, state=state_name,
                                    checkpoint={"stage": stage, "group": group, "round": round_number},
                                    state_temporal_policy="daily_rollforward"))
    reference = next(origin["origin_id"] for origin in schedule if origin["axis"] == "axis1")
    for origin in schedule:
        origin["phase_stages"] = sorted(selected_stages)
        if event.get("observables"):
            origin["observables"] = event["observables"]
        if origin["axis"] == "axis3":
            origin["reference_origin_id"] = reference
    indexed = list(enumerate(schedule))
    schedule = [origin for _, origin in sorted(indexed, key=lambda item: (item[1]["cutoff"], item[0]))]
    if len({origin["origin_id"] for origin in schedule}) != len(schedule):
        raise ValueError("generated duplicate origin ID")
    return schedule, states


def _post_round_readiness(spec: dict[str, Any], draws: list[dict[str, Any]],
                          completed: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed_pairs: dict[tuple[Any, Any, Any], set[frozenset[str]]] = {}
    for row in completed:
        block = row["stage"], row["group"], row["round"]
        completed_pairs.setdefault(block, set()).add(frozenset((row["team_a"], row["team_b"])))
    readiness = []
    seen = set()
    for draw in draws:
        block = draw["stage"], draw["group"], draw["round"]
        expected = {frozenset(pair) for pair in draw["pairs"]}
        observed = completed_pairs.get(block, set())
        status = "eligible" if observed == expected else "excluded"
        reason = ("every source-backed pairing in the block has a completed source record"
                  if status == "eligible" else
                  "round/block remains incomplete under source-backed pairing facts")
        readiness.append({"block": {"stage": block[0], "group": block[1], "round": block[2]},
                          "status": status, "reason": reason})
        seen.add(block)
    for block, rows, expected_pair in _graph_node_records(spec, completed, draws):
        if block in seen:
            continue
        status = ("eligible" if len(rows) == 1
                  and frozenset((rows[0]["team_a"], rows[0]["team_b"])) == expected_pair else "excluded")
        reason = ("declared graph node and completed ancestor evidence prove its participants"
                  if status == "eligible" else
                  "graph node lacks a completed source record or resolved ancestor evidence")
        readiness.append({"block": {"stage": block[0], "group": block[1], "round": block[2]},
                          "status": status, "reason": reason})
    checkpoints = dict(_round_checkpoints(spec, draws, completed))
    for entry in readiness:
        block = entry["block"]
        cutoff = checkpoints.get((block["stage"], block["group"], block["round"]))
        if cutoff is not None:
            entry["cutoff"] = _iso_midnight(cutoff.date())
            entry["unresolved_target_ids"] = [target["target_id"] for target in _eligible_targets(targets, cutoff)]
            if not entry["unresolved_target_ids"]:
                entry.update(status="excluded", reason="completed checkpoint has no future unresolved targets")
    return readiness


def _ledger(schedule: list[dict[str, Any]], spec: dict[str, Any], draws: list[dict[str, Any]],
            completed: list[dict[str, Any]], targets: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for origin in schedule:
        cutoff = _midnight(origin["cutoff"], "origin.cutoff")
        date_cutoff = cutoff.date()
        included_completed = [row["id"] for row in completed if _source_day(row["source_date"], "source_date") < date_cutoff]
        excluded_completed = [row["id"] for row in completed if _source_day(row["source_date"], "source_date") >= date_cutoff]
        included_draws = [row["id"] for row in draws if _source_day(row["source_date"], "source_date") < date_cutoff]
        excluded_draws = [row["id"] for row in draws if _source_day(row["source_date"], "source_date") >= date_cutoff]
        entries.append({"origin_id": origin["origin_id"], "forecast_mode": origin["forecast_mode"], "cutoff": origin["cutoff"],
                        "qualification": QUALIFICATION, "completed_included": included_completed,
                        "completed_excluded": excluded_completed, "draws_included": included_draws,
                        "draws_excluded": excluded_draws,
                        "unresolved_target_ids": [target["target_id"] for target in origin["targets"]],
                        "reason": "source calendar day strictly before cutoff UTC day"})
    return {"version": VERSION, "qualification": QUALIFICATION, "entries": entries,
            "pre_draw_readiness": (
                {"status": "eligible", "reason": "spec declares a uniform unknown draw policy"}
                if any(origin["forecast_mode"] == "pre_draw" for origin in schedule) else
                {"status": "not_applicable", "reason": "spec does not declare a uniform unknown draw policy"}),
            "post_round_readiness": _post_round_readiness(spec, draws, completed, targets),
            "limitations": ["retrospective reconstruction; source calendar days are not certified publication timestamps",
                            "no model execution or outcome-label read occurred"]}


def _write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _merge_evidence_pins(pinned: dict[Path, str], supplied: Any, base: Path) -> None:
    if not isinstance(supplied, dict):
        raise ValueError("evidence_hashes must map local paths to SHA-256 pins")
    for value, expected in supplied.items():
        if (not isinstance(value, str) or not value or not isinstance(expected, str)
                or len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected)):
            raise ValueError("evidence_hashes requires local paths and lowercase SHA-256 pins")
        path = (base / value).resolve()
        if path in pinned and pinned[path] != expected:
            raise ValueError("conflicting evidence hash pins for one resolved path")
        try:
            actual = _file_digest(path)
        except OSError as error:
            raise ValueError(f"pinned evidence is missing or unreadable: {path}") from error
        if actual != expected:
            raise ValueError(f"pinned evidence hash changed: {path}")
        pinned[path] = expected


def _evaluator_manifest(config_path: Path, output: Path, event: dict[str, Any],
                        spec: dict[str, Any], states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from scripts.tournament_evaluation import VERSION as RUNNER_VERSION
    from scripts.tournament_evaluation import _path, _validate_manifest, _validate_mode_state

    config = _load(config_path)
    if config.get("version") != RUNNER_VERSION or config.get("qualification") != QUALIFICATION:
        raise ValueError("model configuration requires retrospective runner v3 qualification")
    events = [item for item in config.get("events", [])
              if (item["tournament_id"], item["phase_id"]) == (event["tournament_id"], event["phase_id"])]
    if len(events) != 1:
        raise ValueError("model configuration must identify this edition and phase exactly once")

    def rebase(value: str) -> str:
        return os.path.relpath(_path(config_path.parent, value), output)

    configuration_pins: dict[Path, str] = {}
    _merge_evidence_pins(configuration_pins, config.get("evidence_hashes", {}), config_path.parent)

    def binding_key(origin: dict[str, Any]) -> tuple[datetime, bool]:
        if origin.get("axis") not in {"axis1", "axis2", "axis3"}:
            raise ValueError("model configuration origins require an explicit runner axis")
        return _utc(origin["cutoff"], "model origin cutoff"), origin["axis"] == "axis1"

    bindings = {}
    binding_origins: dict[tuple[datetime, bool], list[str]] = {}
    daily_axes: dict[tuple[datetime, bool], str] = {}
    for origin in events[0]["origins"]:
        identity = binding_key(origin)
        if origin.get("forecast_mode") == "daily_rollforward":
            if identity in daily_axes and daily_axes[identity] != origin["axis"]:
                raise ValueError("model configuration has conflicting daily comparison axes")
            daily_axes[identity] = origin["axis"]
        inputs = {}
        for key in ("pairs", "tables"):
            if key in origin:
                if not isinstance(origin[key], dict):
                    raise ValueError(f"model origin {key} must map model IDs to paths")
                inputs[key] = {name: rebase(value) for name, value in origin[key].items()}
        if "score_model" in origin:
            inputs["score_model"] = rebase(origin["score_model"])
        if "model_exclusions" in origin:
            inputs["model_exclusions"] = origin["model_exclusions"]
        if identity in bindings and bindings[identity] != inputs:
            raise ValueError("model configuration has conflicting bindings for the same cutoff and matrix role")
        bindings[identity] = inputs
        binding_origins.setdefault(identity, []).append(origin["origin_id"])
    models = [dict(model) for model in config.get("models", [])]
    for model in models:
        if "artifact" in model:
            model["artifact"] = rebase(model["artifact"])
    configured_event = {**event, "origins": []}
    for origin in event["origins"]:
        key = binding_key(origin)
        if key not in bindings:
            raise ValueError(f"model configuration lacks explicit inputs at {origin['cutoff']} for its matrix role")
        configured = {**origin, **bindings[key]}
        if origin["forecast_mode"] == "daily_rollforward":
            if key not in daily_axes:
                raise ValueError(f"model configuration lacks an explicit daily comparison axis at {origin['cutoff']}")
            configured["axis"] = daily_axes[key]
            if configured["axis"] == "axis3":
                configured["reference_origin_id"] = next(
                    item["origin_id"] for item in event["origins"] if item["axis"] == "axis1")
        for model in models:
            name = model["id"]
            if name in configured.get("model_exclusions", {}):
                continue
            input_key = "tables" if model["kind"] == "series_table" else "pairs"
            if name not in configured.get(input_key, {}):
                raise ValueError(f"model configuration lacks {input_key} or exclusion for {name}")
        _validate_mode_state(configured, spec, states.get(origin.get("state")), QUALIFICATION)
        configured_event["origins"].append(configured)
    manifest = {
        "version": RUNNER_VERSION, "qualification": QUALIFICATION,
        "models": models, "events": [configured_event],
        "evidence_files": [rebase(str(config_path)), "source_hashes.json", "readiness_ledger.json",
                           *[rebase(value) for value in config.get("evidence_files", [])]],
        "evidence_hashes": {os.path.relpath(path, output): value for path, value in configuration_pins.items()},
        "exclusions": config.get("exclusions", []),
        "model_configuration": {
            "source": rebase(str(config_path)), "binding_rule": "exact_tournament_phase_UTC_cutoff_and_start_or_current_matrix",
            "origin_bindings": {origin["origin_id"]: binding_origins[binding_key(origin)]
                                for origin in event["origins"]},
        },
    }
    _validate_manifest(manifest)
    return manifest


def compile_daily_tournament_evaluation(input_path: str | Path, output_dir: str | Path,
                                       *, model_config: str | Path | None = None) -> dict[str, Any]:
    """Compile one new immutable local preparation directory from explicit JSON."""
    source = Path(input_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a new immutable artifact directory")
    payload = _load(source)
    event, pre_cutoff, post_cutoff, draws, completed, targets = _validate_input(payload)
    # Keep a single declared spec copy local to the new artifact; never read it
    # from outcomes or synthesize entrants, seeds, or pairing facts.
    event = dict(event)
    event["spec"] = payload["spec"]
    event["observables"] = payload.get("observables", [])
    schedule, states = _schedule(event, pre_cutoff, post_cutoff, draws, completed, targets)
    manifest_event = {key: event[key] for key in ("tournament_id", "phase_id", "edition_start_at", "family", "tier", "format")}
    for key in ("phase_start_at", "phase_stages", "edition_start_status", "phase_start_precision"):
        if key in event:
            manifest_event[key] = event[key]
    manifest_event["spec"] = "spec.json"
    manifest_event["origins"] = schedule
    manifest = {"version": VERSION, "qualification": QUALIFICATION, "event": manifest_event,
                "protocol": {"daily_state_policy": "daily_rollforward", "daily_cutoff_rule": "source_date < cutoff UTC calendar day",
                             "forecast_phase": "outcome_blind", "prospective_capture": "strict_timestamped only"}}
    outcome_template = {"version": VERSION, "qualification": QUALIFICATION,
                        "labels": [{"target_id": target["target_id"], "target_kind": target["target_kind"],
                                    "target_start_at": target["target_start_at"], "outcome": None,
                                    "tournament_id": event["tournament_id"], "phase_id": event["phase_id"]}
                                   for target in targets],
                        "instruction": "Populate only after immutable forecasts are persisted; not a compiler input."}
    source_hashes = {"version": VERSION, "input_file_sha256": _file_digest(source),
                     "sections": {key: _digest(payload[key]) for key in ("event", "spec", "seeds", "draws", "completed", "targets")},
                     "compiler_file_sha256": _file_digest(Path(__file__).resolve())}
    evidence_paths = payload.get("evidence_files", [])
    if not isinstance(evidence_paths, list) or any(not isinstance(path, str) or not path for path in evidence_paths):
        raise ValueError("evidence_files must be an explicit list of local paths")
    evidence_paths = sorted({(source.parent / path).resolve() for path in evidence_paths})
    source_hashes["evidence_files"] = {str(path): _file_digest(path) for path in evidence_paths}
    ledger = _ledger(schedule, payload["spec"], draws, completed, targets)
    evaluator = None
    if model_config is not None:
        config_path = Path(model_config).resolve()
        source_hashes["model_config_sha256"] = _file_digest(config_path)
        evaluator = _evaluator_manifest(config_path, output, manifest_event, payload["spec"], states)
        pinned = {source: source_hashes["input_file_sha256"],
                  **{Path(path): value for path, value in source_hashes["evidence_files"].items()}}
        pinned[config_path] = source_hashes["model_config_sha256"]
        _merge_evidence_pins(pinned, evaluator["evidence_hashes"], output)
        evaluator["evidence_hashes"] = {os.path.relpath(path, output): value for path, value in pinned.items()}

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.pending-", dir=output.parent))
    try:
        (staging / "states").mkdir()
        _write_new(staging / "spec.json", payload["spec"])
        _write_new(staging / "event_manifest.json", manifest)
        _write_new(staging / "origins.json", {"version": VERSION, "qualification": QUALIFICATION, "origins": schedule})
        _write_new(staging / "outcomes.template.json", outcome_template)
        _write_new(staging / "source_hashes.json", source_hashes)
        _write_new(staging / "readiness_ledger.json", ledger)
        for relative, state in states.items():
            _write_new(staging / relative, state)
        if evaluator is not None:
            _write_new(staging / "evaluator_manifest.json", evaluator)
            if _file_digest(config_path) != source_hashes["model_config_sha256"]:
                raise ValueError("model configuration changed during compilation")
            _merge_evidence_pins({}, evaluator["evidence_hashes"], output)
        if _file_digest(source) != source_hashes["input_file_sha256"]:
            raise ValueError("source input changed during compilation")
        if any(_file_digest(Path(path)) != value for path, value in source_hashes["evidence_files"].items()):
            raise ValueError("referenced evidence changed during compilation")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"status": "compiled", "output_dir": str(output), "qualification": QUALIFICATION,
            "origins": len(schedule), "states": len(states), "source_sha256": source_hashes["input_file_sha256"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="explicit source-backed phase input JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="new immutable local artifact directory")
    parser.add_argument("--model-config", type=Path, help="runner v3 manifest with explicit model bindings at each cutoff")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(compile_daily_tournament_evaluation(args.input, args.output_dir,
                                                           model_config=args.model_config), indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
