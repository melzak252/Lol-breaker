"""Source-bound retrospective native tournament scenarios, separate from labels.

Archive retrieval is not publication evidence. Opening fixtures may condition a
retrospective scenario; later draws and results enter only observed-label replay.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re

from src.models.tournament_formats import (
    _Run, _perfect_matching, validate_tournament_spec,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/artifacts/leaguepedia-tournament-rules-20260908"
RAW = ROOT / "data/artifacts/golgg-database-recovery-20260909/matches.json"
REFERENCE = ROOT / "data/07_model_output/a0_context_reliability/context_reliability_20261004_02/benchmark_reference_v2.parquet"
CASES = ("worlds-2023", "worlds-2024", "msi-2023", "msi-2024", "worlds-2025", "msi-2025", "msi-2026", "lck-2026-road-to-msi")
MODELS = ("native_c0_unchanged", "residual_posterior_mean", "residual_independent_matched_marginals", "residual_coherent", "origin_eligible_annual_a0", "prior_calibrated_glicko_format", "uniform")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _iter_raw_series(path):
    """Frame a JSON array in bounded chunks; decode only one series at a time.

    The stdlib decoder preserves JSON types (including integer IDs and boolean
    winners). No optional parser dependency or full-source text buffer is needed.
    """
    opened = ended = in_string = escaped = after_comma = False
    depth = 0
    parts = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), ""):
            start = 0
            for pos, char in enumerate(chunk):
                if ended:
                    if not char.isspace():
                        raise ValueError("extra data after raw series array")
                    continue
                if not opened:
                    if char.isspace():
                        start = pos + 1
                        continue
                    if char != "[":
                        raise ValueError("raw series source must be a JSON array")
                    opened = True
                    start = pos + 1
                    continue
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char in "[{":
                    depth += 1
                elif char in ",]" and depth == 0:
                    parts.append(chunk[start:pos])
                    text = "".join(parts)
                    parts.clear()
                    if char == "," or text.strip():
                        yield json.loads(text)
                    elif after_comma:
                        raise ValueError("trailing comma in raw series array")
                    after_comma = char == ","
                    ended = char == "]"
                    start = pos + 1
                elif char in "]}":
                    depth -= 1
            if opened and not ended:
                parts.append(chunk[start:])
    if not ended:
        raise ValueError("unterminated raw series array")


def _read_raw_cohort(path):
    """Retain only cohort evidence, leaving missing/malformed values untouched.

    Role order and nonstandard role names matter to the actual-roster fallback.
    All maps remain available for release boundaries and stopped series scores.
    """
    def fields(value, names):
        return {key: item for key, item in value.items() if key in names} if isinstance(value, dict) else value

    result = []
    for series in _iter_raw_series(path):
        match = fields(series, ("match_id", "date", "tournament_name", "games"))
        if isinstance(match, dict) and isinstance(match.get("games"), list):
            games = []
            for original in match["games"]:
                game = fields(original, ("date", "t1_id", "t2_id", "t1_name", "t2_name",
                                         "t1_win", "t2_win", "t1_players", "t2_players"))
                if isinstance(game, dict):
                    for side in ("t1", "t2"):
                        key = f"{side}_players"
                        if isinstance(game.get(key), dict):
                            game[key] = {role: fields(player, ("player_id",))
                                         for role, player in game[key].items()}
                games.append(game)
            match["games"] = games
        result.append(match)
    return result


def _utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff/publication must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _ref(stage, rank):
    return {"stage": stage, "group": "all", "rank": rank}


def _bands(stage, ranges):
    return [{"label": str(lo) if lo == hi else f"{lo}-{hi}",
             "placements": [_ref(stage, rank) for rank in range(lo, hi + 1)]}
            for lo, hi in ranges]


def _source_contract(edition, contract):
    if not isinstance(contract, dict) or contract.get("edition_id") != edition:
        raise ValueError("edition-specific rule source required")
    path = Path(contract.get("source_path", ""))
    digest = contract.get("source_sha256", {}).get("rules")
    if not path.is_file() or _hash(path) != digest:
        raise ValueError("missing or changed edition rule evidence")
    if "raw_source_path" in contract:
        raw_path = Path(contract["raw_source_path"])
        if not raw_path.is_file() or _hash(raw_path) != contract["source_sha256"].get("raw_rules"):
            raise ValueError("missing or changed underlying raw rule evidence")
    if contract.get("opening_seed_basis") != "prior_source":
        raise ValueError("opening seeds require prior source identity, never future results")
    if contract.get("draw_basis") not in {
        "retrospective_randomized_scenario", "retrospective_known_opening_draw_scenario", "source_fixed"
    }:
        raise ValueError("unknown draw provenance")
    if edition.startswith("worlds-"):
        year = int(edition.split("-")[-1])
        swiss = contract.get("swiss", {})
        if (swiss.get("no_rematches") is not (year >= 2024)
                or swiss.get("wins_to_advance") != 3 or swiss.get("losses_to_eliminate") != 3):
            raise ValueError("contradictory edition-specific Swiss rules")
        if year >= 2024 and swiss.get("rematch_fallback") != "waive_only_if_no_feasible_draw":
            raise ValueError("Worlds rematch waiver must require an infeasible draw")
        knockout = contract.get("knockout", {})
        if not all(knockout.get(key) is True for key in ("three_zero_opposite_halves", "three_zero_vs_three_two")):
            raise ValueError("missing official Worlds quarterfinal restrictions")
    elif edition.startswith("msi-"):
        msi = contract.get("msi", {})
        if msi.get("upper_round_two_loser_routing") != "cross_half" or msi.get("final_reset") != "single_final":
            raise ValueError("MSI requires cross-half lower drops and one grand final")


def build_research_event_spec(edition_id, teams, *, initial_pairings=None, source_contract=None):
    """Build from entrants and opening evidence only, never future opponents."""
    _source_contract(edition_id, source_contract)
    teams = list(teams)
    if len(set(teams)) != len(teams) or any(not isinstance(t, str) or not t for t in teams):
        raise ValueError("entrants require distinct exact native identities")
    if initial_pairings is not None:
        initial_pairings = [list(pair) for pair in initial_pairings]
        if (any(len(pair) != 2 for pair in initial_pairings)
                or sorted(t for pair in initial_pairings for t in pair) != sorted(teams)):
            raise ValueError("opening fixtures must cover entrants exactly once")
        if source_contract["draw_basis"] not in {"retrospective_known_opening_draw_scenario", "source_fixed"}:
            raise ValueError("fixed opening pairs require an explicit known-draw scenario")
    metadata = {"edition_id": edition_id, "historical_publication_certified": bool(source_contract.get("authenticated_prestart")),
                "observed_results_used": False, "draw_basis": source_contract["draw_basis"]}
    if edition_id.startswith("worlds-"):
        if len(teams) != 16:
            raise ValueError("Worlds Swiss needs sixteen entrants")
        swiss = {"id": "swiss", "kind": "swiss", "entrants": teams, "best_of": 1,
                 "decider_best_of": 3, "wins_to_advance": 3, "losses_to_eliminate": 3,
                 "advance": 8, "no_rematches": source_contract["swiss"]["no_rematches"],
                 "tiebreakers": ["random"]}
        if "rematch_fallback" in source_contract["swiss"]:
            swiss["rematch_fallback"] = source_contract["swiss"]["rematch_fallback"]
        if initial_pairings is not None:
            swiss["initial_pairings"] = initial_pairings
        stages = [swiss, {"id": "knockout", "kind": "single_elimination", "best_of": 5,
                         "entrants": [_ref("swiss", rank) for rank in range(1, 9)],
                         "swiss_quarterfinal_from": "swiss"}]
        bands = _bands("knockout", [(1, 1), (2, 2), (3, 4), (5, 8)])
        bands += _bands("swiss", [(9, 11), (12, 14), (15, 16)])
        spec = {"version": 1, "id": edition_id, "teams": teams, "stages": stages,
                "champion": _ref("knockout", 1), "placement_bands": bands, "metadata": metadata}
    elif edition_id.startswith("msi-"):
        if len(teams) != 8 or initial_pairings is None:
            raise ValueError("MSI requires eight entrants and source-bound opening bracket slots")
        slots = [team for pair in initial_pairings for team in pair]
        spec = {"version": 1, "id": edition_id, "teams": teams,
                "stages": [{"id": "bracket", "kind": "double_elimination", "entrants": slots,
                            "best_of": 5, "seeding": "ordered", "final_reset": "single_final"}],
                "champion": _ref("bracket", 1),
                "placement_bands": _bands("bracket", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 8)]),
                "metadata": metadata}
    elif edition_id == "lck-2026-road-to-msi":
        spec = deepcopy(source_contract["phase_spec"])
        if spec["teams"] != teams:
            raise ValueError("LCK entrants disagree with prior phase source")
        spec["id"] = edition_id
        spec.pop("champion", None)
        spec.pop("final", None)
        spec["metadata"] = metadata
        ranges = source_contract.get("official_placement_ranges")
        if ranges:
            spec["placement_bands"] = _bands("qualifier", ranges)
        else:
            spec.pop("placement_bands", None)
    else:
        raise ValueError("unsupported research edition")
    validate_tournament_spec(spec)
    return spec


def validate_research_event(event):
    """Reject malformed identities, outcome-conditioned inputs and false authenticity."""
    edition = event["edition_id"]
    contract = event["source_provenance"]
    _source_contract(edition, contract)
    cutoff = _utc(event["cutoff"])
    if cutoff.year != event["year"]:
        raise ValueError("event year differs from actual UTC cutoff year")
    teams = event["teams"]
    if event["spec"]["teams"] != teams or set(event["rosters"]) != set(teams):
        raise ValueError("native entrant/roster/spec membership differs")
    if set(event.get("roster_evidence", {})) != set(teams):
        raise ValueError("roster evidence must cover exactly the frozen entrants")
    label_path = Path(event.get("labels_path", ""))
    if not event.get("labels_path") or any(
            field in event and label_path.resolve() == Path(event[field]).resolve()
            for field in ("input_path", "source_path")):
        raise ValueError("observed labels must have their own separate file")
    all_players = set()
    conditional = False
    for team in teams:
        roster = event["rosters"][team]
        if (not isinstance(roster, list) or len(roster) != 5 or len(set(roster)) != 5
                or any(not isinstance(p, str) or not p for p in roster) or all_players.intersection(roster)):
            raise ValueError("each entrant needs five distinct disjoint real player identities")
        all_players.update(roster)
        evidence = event["roster_evidence"][team]
        if not evidence.get("source_match_id") or not evidence.get("source_sha256"):
            raise ValueError("missing roster source evidence")
        basis = evidence.get("basis")
        if evidence.get("context_basis") != basis:
            raise ValueError("roster context basis differs from declared basis")
        if evidence.get("source_context") not in contract.get("roster_context_bindings", {}):
            raise ValueError("roster source context lacks edition-bound native evidence")
        if not contract["roster_context_bindings"][evidence["source_context"]]:
            raise ValueError("empty roster context proof set")
        if basis == "first_event_actual_roster":
            if datetime.fromisoformat(evidence["source_day"]).date() < cutoff.date():
                raise ValueError("first-event actual roster predates event cutoff")
            conditional = True
            if evidence.get("authenticated_prestart"):
                raise ValueError("actual first-event roster is not prestart publication evidence")
        elif basis == "strictly_prior_observed_same_context":
            if datetime.fromisoformat(evidence["source_day"]).date() >= cutoff.date():
                raise ValueError("prior roster must be released strictly before cutoff day")
        else:
            raise ValueError("unknown roster basis")
    if conditional and contract.get("authenticated_prestart"):
        raise ValueError("conditional actual rosters cannot authenticate an event as prestart")
    aggregate_basis = "first_event_actual_roster" if conditional else "strictly_prior_observed_same_context"
    if event.get("roster_basis") != aggregate_basis:
        raise ValueError("aggregate roster basis contradicts per-team evidence")
    if contract.get("authenticated_prestart") or contract["draw_basis"] == "source_fixed":
        for field in ("published_at", "available_at"):
            if field not in contract or _utc(contract[field]) >= cutoff:
                raise ValueError("authenticated draw needs strictly prior publication and availability evidence")
        proof = contract.get("publication_evidence")
        if not isinstance(proof, dict) or not Path(proof.get("path", "")).is_file() or _hash(proof["path"]) != proof.get("sha256"):
            raise ValueError("authenticated draw requires hash-bound publication proof")
    for field in ("future_pairings", "later_draws", "observed_results"):
        if field in contract or field in event:
            raise ValueError("future results/draws must remain separate from forecast inputs")
    validate_tournament_spec(event["spec"])
    stages = event["spec"]["stages"]
    opening = stages[0].get("initial_pairings") if edition.startswith("worlds-") else None
    if edition.startswith("msi-"):
        slots = stages[0]["entrants"]
        opening = [slots[i:i + 2] for i in range(0, len(slots), 2)]
    expected = build_research_event_spec(edition, teams, initial_pairings=opening, source_contract=contract)
    actual = {k: v for k, v in event["spec"].items() if k != "metadata"}
    if actual != {k: v for k, v in expected.items() if k != "metadata"}:
        raise ValueError("declared tournament mechanics differ from edition-bound source contract")
    if contract.get("opening_pairings") is not None and contract["opening_pairings"] != opening:
        raise ValueError("declared opening fixtures differ from frozen source bindings")


class _ObservedRun(_Run):
    """Replay observed outcomes through the existing format engine, not a new bracket."""
    def __init__(self, spec, matches):
        super().__init__(spec, {}, {}, random.Random(0), True)
        self.matches = matches
        self.remaining = defaultdict(deque)
        for index, row in enumerate(matches):
            self.remaining[frozenset((row["team_a"], row["team_b"]))].append((index, row))
        self.consumed = set()

    def match(self, a, b, round_number, group="all", best_of=None, weight=1):
        if a is None or b is None:
            return (b, None) if a is None else (a, None)
        queue = self.remaining[frozenset((a, b))]
        if not queue:
            raise ValueError(f"observed result missing for declared route {self.stage['id']}:{round_number}:{a}/{b}")
        index, row = queue.popleft()
        bo = best_of or self.stage["best_of"]
        if row["best_of"] != bo or (row.get("stage") is not None and row["stage"] != self.stage["id"]):
            raise ValueError("observed format/stage contradicts declared route")
        day = str(row["date"])[:10]
        if any(self.team_available.get(team, day) > day for team in (a, b)):
            raise ValueError("observed route result precedes an ancestor result")
        self.team_available[a] = self.team_available[b] = day
        if row.get("round") is not None and str(row["round"]) != str(round_number):
            raise ValueError("observed node contradicts declared route")
        scores = {row["team_a"]: row["wins_a"], row["team_b"]: row["wins_b"]}
        winner, loser = (a, b) if scores[a] > scores[b] else (b, a)
        record = {"stage": self.stage["id"], "group": group, "round": round_number,
                  "team_a": a, "team_b": b, "best_of": bo, "winner": winner, "loser": loser,
                  "score_a": scores[a], "score_b": scores[b], "weight": weight}
        self.log.append(record)
        self.trace.append(record)
        self.series += 1
        self.played.add(frozenset((a, b)))
        self.consumed.add(index)
        return winner, loser

    def draw(self, teams, draw, source_groups, no_rematches=False, records=None, round_number=1):
        rows = [r for i, r in enumerate(self.matches) if i not in self.consumed
                and r.get("stage") == self.stage["id"] and str(r.get("round")) == str(round_number)]
        if not rows:
            raise ValueError("observed draw is missing its source-bound round identity")
        pairs = [(r["team_a"], r["team_b"]) for r in rows]
        if sorted(t for p in pairs for t in p) != sorted(teams):
            raise ValueError("observed draw does not cover its active field")
        if records and any(records[a] != records[b] for a, b in pairs):
            raise ValueError("observed Swiss draw crosses record brackets")
        if no_rematches and any(frozenset(p) in self.played for p in pairs):
            if getattr(self, "_label_rematch_fallback", self.stage.get("rematch_fallback")) != "waive_only_if_no_feasible_draw":
                raise ValueError("observed draw repeats a forbidden pair")
            for pair in pairs:
                bucket = [t for t in teams if records[t] == records[pair[0]]]
                try:
                    _perfect_matching(bucket, lambda a, b: frozenset((a, b)) not in self.played,
                                      lambda a, b: 0, random.Random(0))
                except ValueError:
                    continue
                if frozenset(pair) in self.played:
                    raise ValueError("observed rematch waiver had a feasible nonrematch alternative")
        return pairs

    def swiss(self, teams, source_groups):
        # The forecast uses the declared waiver policy. Label replay uses the
        # actual source draws and independently checks whether any waiver was legal.
        stage = self.stage
        self._label_rematch_fallback = stage.get("rematch_fallback")
        replay = dict(stage)
        replay.pop("rematch_fallback", None)
        self.stage = replay
        try:
            return super().swiss(teams, source_groups)
        finally:
            self.stage = stage

    def first_pairs(self, teams, source_groups=None, tie=False):
        if not tie and "swiss_quarterfinal_from" in self.stage:
            rows = [r for i, r in enumerate(self.matches) if i not in self.consumed
                    and r.get("stage") == self.stage["id"] and str(r.get("round")) == "1"]
            pairs = [(r["team_a"], r["team_b"]) for r in rows]
            if len(pairs) != 4 or sorted(t for p in pairs for t in p) != sorted(teams):
                raise ValueError("observed quarterfinal draw must cover all eight qualifiers")
            records = self.records[self.stage["swiss_quarterfinal_from"]]
            halves = set()
            for index, pair in enumerate(pairs):
                unbeaten = [t for t in pair if records[t] == {"wins": 3, "losses": 0}]
                if unbeaten:
                    if len(unbeaten) != 1:
                        raise ValueError("observed quarterfinal pairs two unbeaten entrants")
                    opponent = next(t for t in pair if t not in unbeaten)
                    if records[opponent] != {"wins": 3, "losses": 2}:
                        raise ValueError("observed quarterfinal contradicts 3-0 versus 3-2 rule")
                    halves.add(index // 2)
            if halves != {0, 1}:
                raise ValueError("observed unbeaten quarterfinals must occupy opposite halves")
            return pairs
        return super().first_pairs(teams, source_groups, tie)


def build_observed_labels(event_id, edition_id, matches, *, spec=None):
    """Admit every observed stopped series against an already frozen input spec."""
    if spec is None:
        raise ValueError("labels require an independently frozen tournament spec")
    validate_tournament_spec(spec)
    rows = [dict(row) for row in matches]
    ids = set()
    for row in rows:
        if row["team_a"] == row["team_b"] or not {row["team_a"], row["team_b"]} <= set(spec["teams"]):
            raise ValueError("observed match has unknown/duplicate entrants")
        bo, a, b = row["best_of"], row["wins_a"], row["wins_b"]
        if (bo not in (1, 3, 5) or isinstance(a, bool) or isinstance(b, bool)
                or int(a) != a or int(b) != b or min(a, b) < 0
                or max(a, b) != bo // 2 + 1 or min(a, b) >= bo // 2 + 1):
            raise ValueError("observed match is not a legal stopped score")
        if "native_scores" in row and (row["native_scores"].get(row["team_a"]) != a
                                      or row["native_scores"].get(row["team_b"]) != b):
            raise ValueError(f"source/native stopped-score conflict: {row['match_id']}")
        if not row.get("match_id") or row["match_id"] in ids:
            raise ValueError("observed match identities must be unique")
        datetime.fromisoformat(str(row["date"])[:10])
        ids.add(row["match_id"])
    replay = _ObservedRun(spec, rows).run()
    if len(replay.consumed) != len(rows):
        raise ValueError("observed matches do not exhaust exactly the declared tournament route")
    champion_ref = spec.get("champion")
    champion = replay.resolve(champion_ref) if champion_ref else None
    qualifiers = {}
    for stage in spec["stages"]:
        if "advance" in stage:
            qualifiers[stage["id"]] = replay.rankings[stage["id"]]["all"][:stage["advance"]]
    placements = {}
    for band in spec.get("placement_bands", []):
        limits = [int(v) for v in band["label"].split("-")]
        for ref in band["placements"]:
            placements[replay.resolve(ref)] = [limits[0], limits[-1]]
    swiss_records = None
    for stage in spec["stages"]:
        if stage["kind"] == "swiss":
            swiss_records = {team: f"{r['wins']}-{r['losses']}" for team, r in replay.records[stage["id"]].items()}
    return {"event_id": event_id, "edition_id": edition_id, "champion": champion,
            "finalists": list(replay.finalists) if champion is not None and replay.finalists else None,
            "qualifiers": qualifiers, "swiss_records": swiss_records,
            "placement_bands": placements or None, "matches": rows,
            "route_admission": "ALL_OBSERVED_MATCHES_REPLAYED_THROUGH_FROZEN_FORMAT_ENGINE"}


def _wiki_text(rule):
    bundle = _json(ROOT / rule["source"]["raw_path"])
    pages = bundle["query"]["pages"]
    if isinstance(pages, dict):
        pages = pages.values()
    page = next((p for p in pages if p.get("title") == rule["title"]), None)
    if page is None:
        raise ValueError("edition rule raw bundle lacks its named source page")
    revision = next((r for r in page["revisions"] if r["revid"] == rule["source"]["revid"]), None)
    if revision is None:
        raise ValueError("edition raw rule revision differs")
    return revision["slots"]["main"]["content"]


def _bracket_fields(text, kind):
    """Extract one balanced template; nested templates and inline ends are legal."""
    starts = list(re.finditer(r"\{\{\s*Bracket\s*\|\s*" + kind + r"\b", text, re.I))
    if len(starts) != 1:
        raise ValueError("raw source lacks one unambiguous edition bracket template")
    start = starts[0]
    depth, pos, field_start = 1, start.end(), start.end()
    fields = {}
    while pos < len(text):
        token = text[pos:pos + 2]
        if token == "{{":
            depth += 1
            pos += 2
        elif token == "}}":
            depth -= 1
            if depth == 0:
                part = text[field_start:pos]
                if "=" in part:
                    key, value = part.split("=", 1)
                    key = key.strip().lower()
                    if key in fields:
                        raise ValueError("duplicate bracket field")
                    fields[key] = value.strip()
                return fields
            pos += 2
        elif text[pos] == "|" and depth == 1:
            part = text[field_start:pos]
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip().lower()
                if key in fields:
                    raise ValueError("duplicate bracket field")
                fields[key] = value.strip()
            field_start = pos + 1
            pos += 1
        else:
            pos += 1
    raise ValueError("unterminated edition bracket template")


def _knockout_slots(text):
    fields = _bracket_fields(text, "8SE")
    if fields.get("cargo", "").lower() != "yes":
        raise ValueError("Worlds bracket is not Cargo-bound")
    slots = {}
    # Named rounds and their numbered source nodes are the archived 8SE
    # Cargo convention. Overrides replace individual nodes, not round order.
    for round_number, title, count in ((1, "Quarterfinals", 4), (2, "Semifinals", 2), (3, "Finals", 1)):
        if fields.get(f"r{round_number}_title") != title:
            raise ValueError("Worlds bracket lacks its explicit source round title")
        for number in range(1, count + 1):
            slot = f"R{round_number}M{number}"
            value = fields.get(f"{slot.lower()}_cargomatch", f"{title}_{number}")
            if not re.fullmatch(re.escape(title) + r"_\d+", value):
                raise ValueError("invalid Worlds Cargo node")
            slots[slot] = value
    if len(set(slots.values())) != 7:
        raise ValueError("Worlds Cargo nodes must be unique")
    return slots


def _opening_slots(text, family):
    if family == "worlds":
        nodes = _knockout_slots(text)
        return [nodes[f"R1M{i}"] for i in range(1, 5)]
    fields = _bracket_fields(text, "8DE")
    slots = [fields.get(f"r1m{i}_cargomatch") for i in range(1, 5)]
    if any(not value for value in slots) or len(set(slots)) != 4:
        raise ValueError("raw bracket lacks one unambiguous opening slot")
    return slots


def _opening_constraints(text, family, year, team_names, aliases, opening):
    """Bind printed regions/pools; validate only the known opening fixture."""
    rosters = re.findall(r"\{\{TeamRoster\s*\|\s*team\s*=\s*([^|\n]+)\s*\n\|\s*region\s*=\s*([^|\n]+)", text, re.I)
    regions = {}
    for team, source_name in team_names.items():
        candidates = {region.strip().upper() for name, region in rosters
                      if _alias_names(name.strip(), aliases).intersection(_alias_names(source_name, aliases))}
        if len(candidates) != 1:
            raise ValueError(f"printed prior entrant region unavailable/ambiguous: {source_name}")
        regions[team] = next(iter(candidates))
    pools = {}
    if family == "worlds":
        starts = list(re.finditer(r"\{\{PoolStart\s*\|\s*text\s*=\s*Pool\s*(\d+)\s*\}\}", text, re.I))
        for i, start in enumerate(starts):
            section = text[start.end(): starts[i + 1].start() if i + 1 < len(starts) else text.index("== Results ==")]
            names = re.findall(r"\{\{TeamRoster\s*\|\s*team\s*=\s*([^|\n]+)", section, re.I)
            for team, source_name in team_names.items():
                if any(_alias_names(name.strip(), aliases).intersection(_alias_names(source_name, aliases)) for name in names):
                    pools[team] = int(start.group(1))
        expected = [5, 6, 5] if year == 2025 else [4, 4, 4, 4]
        if [sum(pool == i for pool in pools.values()) for i in range(1, len(expected) + 1)] != expected:
            raise ValueError("printed Worlds pool membership differs from edition rules")
        legal = {(1, 3), (2, 2)} if year == 2025 else {(1, 4), (2, 3)}
        if any(tuple(sorted((pools[a], pools[b]))) not in legal or regions[a] == regions[b] for a, b in opening):
            raise ValueError("known Worlds opening fixture violates printed pool/regional rules")
    else:
        table = text.split("Qualified for Bracket Stage", 1)[1].split("Qualified for Play-In", 1)[0]
        entries = re.findall(r"\|(?:\s*\d+|\s*Q)\s*\|\|\s*([1-4])\s*\|\|\|?\s*\{\{[Tt]eam\|([^}]+)\}\}", table)
        for pool, name in entries:
            for team, source_name in team_names.items():
                if _alias_names(name.strip(), aliases).intersection(_alias_names(source_name, aliases)):
                    pools[team] = int(pool)
        if set(pools) != set(team_names):
            raise ValueError("printed MSI opening pools do not cover all entrants")
        legal = {(1, 3), (2, 3)} if year == 2023 else {(1, 4), (2, 3)}
        if any(tuple(sorted((pools[a], pools[b]))) not in legal or regions[a] == regions[b] for a, b in opening):
            raise ValueError("known MSI opening fixture violates printed pool/regional rules")
        if year >= 2024:
            for half in (opening[:2], opening[2:]):
                if len({regions[t] for pair in half for t in pair}) != 4:
                    raise ValueError("known MSI half contains two teams from one region")
        else:
            if any(sum(pools[t] == 1 for pair in half for t in pair) != 1 for half in (opening[:2], opening[2:])):
                raise ValueError("MSI2023 tier-one teams must occupy opposite halves")
    return {"entrant_regions": regions, "entrant_pools": pools,
            "opening_rules_checked": True}


def _source_rows(frame):
    counters = defaultdict(int)
    result = {}
    for row in frame.sort_values("source_ordinal").to_dict("records"):
        section = row["section_path"]
        counters[section] += 1
        result[f"{section}_{counters[section]}"] = row
    return result


def _write_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _roster(game, side):
    """Reject malformed source rosters at the event admission boundary."""
    players = game.get(f"{side}_players", {})
    if not isinstance(players, dict):
        raise ValueError("raw first-map roster players must be an object")
    if any(not isinstance(player, dict) for player in players.values()):
        raise ValueError("raw first-map roster player entries must be objects")
    roster = [str(players[role]["player_id"]) for role in ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")
              if role in players and players[role].get("player_id") is not None]
    if len(roster) != 5:
        # Archive roles are sometimes abbreviated; preserve only actual players,
        # never insert a missing slot or fabricate an identity.
        roster = [str(p["player_id"]) for p in players.values() if p.get("player_id") is not None]
    if len(roster) != 5 or len(set(roster)) != 5:
        raise ValueError("raw first-map roster is not exactly five unique players")
    return roster


def _eligibility(year, research_root, cutoff, artifact_contracts):
    result = {}
    for model in MODELS:
        reason = None
        available = True
        if model.startswith("residual_") and year < 2026:
            available, reason = False, "RESIDUAL_EXTENSION_NOT_ORIGIN_ELIGIBLE_BEFORE_2026_CAL2025_SELECTION"
        elif model == "origin_eligible_annual_a0":
            from src.analysis.ev_asof_replay import _native_model_folder
            available = _native_model_folder(research_root, year).is_dir()
            reason = None if available else f"ANNUAL_A0_{year}_ARTIFACT_UNAVAILABLE"
        elif model == "native_c0_unchanged" and year == 2023:
            available, reason = False, "NATIVE_C0_2023_CONTROL_REQUIRES_SEPARATE_EARLIER_ONLY_PREPARATION"
            control = (artifact_contracts or {}).get("2023:c0")
            if control is not None and control.get("available") is not False:
                from src.models.tournament_native import native_model_eligibility
                cell = native_model_eligibility({"year": year, "cutoff": cutoff}, "c0", control)
                available, reason = cell["available"], cell["reason"]
        result[model] = {"available": available, "reason": reason,
                         "artifact_verification": "REQUIRED_BY_RESEARCH_NATIVE_PROVIDER" if model not in {"uniform", "prior_calibrated_glicko_format"} else "NOT_APPLICABLE"}
    return result


def _alias_names(name, aliases):
    entry = aliases.get(name, {})
    return {str(value).strip().casefold() for value in
            (name, entry.get("canonical_title"), entry.get("display_name"), entry.get("template_target_title"))
            if value}


def _prior_identity(name, aliases, raw, cutoff, raw_sha256):
    names = _alias_names(name, aliases)
    candidates = []
    for match in raw:
        if not (str(match.get("date", ""))[:4] == cutoff[:4] and match["date"] < cutoff[:10]) or not match.get("games"):
            continue
        game = match["games"][0]
        for side in ("t1", "t2"):
            if names.intersection(_alias_names(game.get(f"{side}_name", ""), aliases)):
                candidates.append((match["date"], int(match["match_id"]), str(game[f"{side}_id"])))
    if not candidates:
        raise ValueError(f"exact strictly prior native identity unavailable: {name}")
    latest = max(candidates)
    return latest[2], {"source_name": name, "native_id": latest[2], "source_day": latest[0],
                       "source_match_id": str(latest[1]), "source_sha256": raw_sha256}


def _lookup_roster(team, cutoff, reference, raw, contexts, actual_match_ids,
                   reference_sha256, raw_sha256):
    """Only released edition-bound priors, otherwise mapped first-event actual."""
    day = cutoff[:10]
    prior = reference[
        (reference.team1_id.astype(str).eq(team) | reference.team2_id.astype(str).eq(team))
        & reference.date.astype(str).lt(day) & reference.result_day.astype(str).lt(day)
        & reference.date.astype(str).str.startswith(cutoff[:4])
        & reference.tournament_name.isin(contexts)]
    choices = []
    for row in prior.to_dict("records"):
        side = "a" if str(row["team1_id"]) == team else "b"
        choices.append((str(row["date"])[:10], int(row["golgg_match_id"]),
                        [str(p) for p in row[f"roster_{side}"]],
                        str(row["tournament_name"]), reference_sha256))
    team_raw = [m for m in raw if m.get("games") and team in
                (str(m["games"][0]["t1_id"]), str(m["games"][0]["t2_id"]))]
    for match in team_raw:
        if (match["tournament_name"] in contexts and match["date"][:4] == cutoff[:4]
                and match["date"] < day
                and max(str(g.get("date", match["date"]))[:10] for g in match["games"]) < day):
            game = match["games"][0]
            side = "t1" if str(game["t1_id"]) == team else "t2"
            choices.append((match["date"], int(match["match_id"]), _roster(game, side),
                            match["tournament_name"], raw_sha256))
    if choices:
        source_day, match_id, roster, context, digest = max(choices, key=lambda c: (c[0], c[1]))
        basis = "strictly_prior_observed_same_context"
    else:
        actual = [m for m in team_raw if str(m["match_id"]) in actual_match_ids
                  and m["date"] >= day and m["tournament_name"] in contexts]
        if not actual:
            raise ValueError(f"NATIVE_FIRST_EVENT_ROSTER_ABSENT: {team}")
        match = min(actual, key=lambda m: (m["date"], int(m["match_id"])))
        game = match["games"][0]
        roster = _roster(game, "t1" if str(game["t1_id"]) == team else "t2")
        source_day, match_id, context, digest = match["date"], match["match_id"], match["tournament_name"], raw_sha256
        basis = "first_event_actual_roster"
    return roster, {"basis": basis, "context_basis": basis, "source_context": context,
                    "source_day": source_day, "source_match_id": str(match_id),
                    "source_sha256": digest, "authenticated_prestart": False}


def _raw_stopped_scores(match):
    scores = defaultdict(int)
    for game in match["games"]:
        if game.get("t1_win") is True:
            scores[str(game["t1_id"])] += 1
        elif game.get("t2_win") is True:
            scores[str(game["t2_id"])] += 1
        else:
            raise ValueError(f"raw map winner unavailable: {match['match_id']}")
        scores.setdefault(str(game["t1_id"]), 0)
        scores.setdefault(str(game["t2_id"]), 0)
    return scores


def prepare_research_cohort(output_dir, research_root=None, reference_path=None, artifact_contracts=None):
    """Freeze exactly eight source-bound scenarios and a complete missingness matrix.

    All inputs are exclusively written before observed labels are constructed.
    Existing catalogue coverage (234 phases) is retained, not reauthenticated.
    """
    import pandas as pd
    from src.models.a0_target_experiment import resolve_research_root
    research_root = resolve_research_root(research_root)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"cohort destination already exists: {output_dir}")
    reference_path = Path(reference_path or REFERENCE).resolve()
    sources = [RAW, SOURCE / "series_results.csv", SOURCE / "golgg_mapping.jsonl",
               SOURCE / "golgg_series_index.jsonl", SOURCE / "team_aliases.json", reference_path,
               SOURCE / "research/worlds.json", SOURCE / "research/msi.json", SOURCE / "catalog.json"]
    source_sha = {str(p): _hash(p) for p in sources}
    raw = _read_raw_cohort(RAW)
    raw_by_id = {str(m["match_id"]): m for m in raw}
    index = {str(row["golgg_match_id"]): row for row in
             (json.loads(line) for line in (SOURCE / "golgg_series_index.jsonl").read_text().splitlines())}
    mappings = {row["wiki_series_id"]: row for row in
                (json.loads(line) for line in (SOURCE / "golgg_mapping.jsonl").read_text().splitlines())}
    frame = pd.read_csv(SOURCE / "series_results.csv")
    reference = pd.read_parquet(reference_path)
    rules = {family: _json(SOURCE / f"research/{family}.json")["events"] for family in ("worlds", "msi")}
    aliases = _json(SOURCE / "team_aliases.json")
    events, exclusions, pending = {}, [], {}
    for edition in CASES:
        family = edition.split("-")[0]
        year = 2026 if family == "lck" else int(edition.split("-")[-1])
        cutoff = None
        try:
            if family == "lck":
                source_path = ROOT / "data/artifacts/tournament-daily-evaluation-20260909/lck-road-to-msi-source.json"
                phase = _json(source_path)
                lck_rule_path = SOURCE / "research/lck.json"
                lck_rule = next(r for r in _json(lck_rule_path)["events"]
                                if r["year"] == 2026 and r["title"] == "LCK/2026 Season/Road to MSI")
                lck_text = _wiki_text(lck_rule)
                printed_places = re.findall(r"\{\{TournamentResults/Line\|([^{}]+)\}\}", lck_text)
                lck_place_rows = []
                for line in printed_places:
                    fields = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
                    if "place" in fields and "team" in fields:
                        place = [int(value.strip()) for value in fields["place"].split("-")]
                        lck_place_rows.append({"band": [place[0], place[-1]], "team_name": fields["team"].strip()})
                if sorted(row["band"] for row in lck_place_rows) != [[rank, rank] for rank in range(1, 7)]:
                    raise ValueError("LCK official placement categories lack explicit source evidence")
                raw_rule_path = ROOT / lck_rule["source"]["raw_path"]
                for evidence_path in [source_path, lck_rule_path, raw_rule_path] + [ROOT / p for p in phase["source_evidence"]]:
                    source_sha[str(evidence_path)] = _hash(evidence_path)
                cutoff = phase["post_draw_cutoff"]
                title = "LCK 2026 Road to MSI"
                contract = {"edition_id": edition, "source_path": str(source_path),
                            "source_sha256": {"rules": _hash(source_path)}, "phase_spec": phase["spec"],
                            "authenticated_prestart": False, "opening_seed_basis": "prior_source",
                            "draw_basis": "retrospective_known_opening_draw_scenario",
                            "qualification_target": "TWO_MSI_BERTHS_NOT_A_CHAMPIONSHIP"}
                contract.update({"raw_source_path": str(raw_rule_path),
                                 "source_sha256": {"rules": _hash(source_path), "raw_rules": _hash(raw_rule_path),
                                                   "processed_lck_rules": _hash(lck_rule_path)},
                                 "processed_rule_source_path": str(lck_rule_path),
                                 "rule_revision": lck_rule["source"]})
                contract["official_placement_ranges"] = [row["band"] for row in sorted(lck_place_rows, key=lambda row: row["band"])]
                contract["official_placement_basis"] = "NAMED_RAW_PAGE_TOURNAMENTRESULTS_LINE_EXPLICIT_PLACE_FIELDS"
                teams = list(phase["spec"]["teams"])
                spec = build_research_event_spec(edition, teams, source_contract=contract)
                rows = []
                for completed in phase["completed"]:
                    rows.append({"team_a": completed["team_a"], "team_b": completed["team_b"],
                                 "best_of": completed["best_of"], "wins_a": completed["score_a"],
                                 "wins_b": completed["score_b"], "date": completed["source_date"],
                                 "match_id": completed["id"], "stage": "qualifier", "round": completed["round"]})
                    candidate = [m for m in raw if m.get("games") and m["date"] == completed["source_date"]
                                 and {str(m["games"][0]["t1_id"]), str(m["games"][0]["t2_id"])}
                                 == {completed["team_a"], completed["team_b"]}]
                    if len(candidate) != 1:
                        raise ValueError("LCK observed route lacks an unambiguous native raw source series")
                    rows[-1]["native_scores"] = dict(_raw_stopped_scores(candidate[0]))
                    rows[-1]["native_match_id"] = str(candidate[0]["match_id"])
                team_names = {}
                for m in sorted(raw, key=lambda row: (str(row.get("date", "")), int(row["match_id"]))):
                    if m.get("games") and m["date"][:4] == str(year) and m["date"] < cutoff[:10]:
                        for side in ("t1", "t2"):
                            game = m["games"][0]
                            native_id = str(game[f"{side}_id"])
                            if native_id in teams:
                                team_names[native_id] = game[f"{side}_name"]
                contract["identity_evidence"] = {}
                for team in teams:
                    prior_id, proof = _prior_identity(team_names[team], aliases, raw, cutoff, source_sha[str(RAW)])
                    if prior_id != team:
                        raise ValueError("LCK phase entrant contradicts exact prior identity")
                    contract["identity_evidence"][team] = proof
                roster_contexts = {"LCK 2026 Rounds 1-2"}
                contract["roster_context_bindings"] = {
                    "LCK 2026 Rounds 1-2": [{"basis": "explicit_preregistered_phase_prior",
                                             "source_path": str(source_path), "sha256": _hash(source_path)}]}
                actual_match_ids = {row["native_match_id"] for row in rows}
            else:
                phase_kind = "swiss" if family == "worlds" else "playoffs"
                rule = next(r for r in rules[family] if r["year"] == year and r["stage"] == phase_kind)
                title = rule["title"]
                # Preserve the benchmark's pre-start-day origin rather than
                # admitting an additional calendar day of roster/history data.
                cutoff = (pd.Timestamp(rule["start_date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
                text = _wiki_text(rule)
                path = SOURCE / f"research/{family}.json"
                raw_rule_path = ROOT / rule["source"]["raw_path"]
                source_sha[str(raw_rule_path)] = _hash(raw_rule_path)
                contract = {"edition_id": edition, "source_path": str(path),
                            "source_sha256": {"rules": _hash(path), "raw_rules": _hash(raw_rule_path)},
                            "raw_source_path": str(raw_rule_path), "rule_revision": rule["source"],
                            "rule_contract": {"advancement_and_seeding": rule["advancement_and_seeding"],
                                              "draw_constraints": rule["draw_constraints"]},
                            "authenticated_prestart": False, "opening_seed_basis": "prior_source",
                            "draw_basis": "retrospective_known_opening_draw_scenario"}
                selected = frame[(frame.event_title == title) & (frame.status == "completed")]
                if family == "msi":
                    selected = selected[(selected.schedule_part == 2)]
                    if len(selected) != 14:
                        raise ValueError(f"MSI bracket requires exactly fourteen source series; got {len(selected)}")
                    contract["msi"] = {"upper_round_two_loser_routing": "cross_half", "final_reset": "single_final",
                                       "regional_opening_constraint": "same_region_forbidden_upper_round_one_only" if year == 2023 else "same_region_opposite_halves"}
                else:
                    if len(selected) != 40:
                        raise ValueError(f"Worlds requires exactly forty source series; got {len(selected)}")
                    contract["swiss"] = {"no_rematches": year >= 2024, "wins_to_advance": 3, "losses_to_eliminate": 3}
                    if year >= 2024:
                        contract["swiss"]["rematch_fallback"] = "waive_only_if_no_feasible_draw"
                    contract["knockout"] = {"three_zero_opposite_halves": True, "three_zero_vs_three_two": True}
                    contract["opening_pool_sizes"] = [5, 6, 5] if year == 2025 else [4, 4, 4, 4]
                    contract["opening_pool_pairing"] = "1_vs_3_and_2_vs_2" if year == 2025 else "1_vs_4_and_2_vs_3"
                source_rows = _source_rows(selected)
                bound = {}
                team_names = {}
                identity_evidence = {}
                for key, source_row in source_rows.items():
                    mapping = mappings.get(source_row["wiki_series_id"])
                    match = None
                    if mapping is not None and mapping["status"] == "matched":
                        native_series = index[str(mapping["golgg_match_id"])]
                        ids = [str(native_series["team1_id"]), str(native_series["team2_id"])]
                        if mapping.get("side_swapped"):
                            ids.reverse()
                        match = raw_by_id[str(mapping["golgg_match_id"])]
                        match_id, day = str(mapping["golgg_match_id"]), str(mapping["golgg_date"])
                    else:
                        # The older archive compiler stops before 2026. Exact
                        # prior aliases repair identity coverage, not outcomes.
                        identities = [_prior_identity(source_row[f"team{side}"], aliases, raw, cutoff, source_sha[str(RAW)])
                                      for side in (1, 2)]
                        ids = [value[0] for value in identities]
                        for native, proof in identities:
                            identity_evidence[native] = proof
                        day = str(source_row["scheduled_at_utc"])[:10]
                        candidates = [m for m in raw if m.get("games") and m["date"] == day
                                      and {str(m["games"][0]["t1_id"]), str(m["games"][0]["t2_id"])} == set(ids)]
                        if len(candidates) > 1:
                            raise ValueError("ambiguous exact native series identity")
                        match = candidates[0] if candidates else None
                        match_id = str(match["match_id"]) if match else f"wiki:{source_row['wiki_series_id']}"
                    for side, native in enumerate(ids, 1):
                        team_names.setdefault(native, source_row[f"team{side}"])
                    bound[key] = {"team_a": ids[0], "team_b": ids[1], "best_of": int(source_row["best_of"]),
                                  "wins_a": int(source_row["team1_score"]), "wins_b": int(source_row["team2_score"]),
                                  "date": day, "match_id": match_id,
                                  "wiki_series_id": source_row["wiki_series_id"],
                                  "native_match_source_available": match is not None}
                    if match is not None:
                        bound[key]["native_scores"] = dict(_raw_stopped_scores(match))
                if family == "msi":
                    slot_keys = _opening_slots(text, "msi")
                    opening = [[bound[key]["team_a"], bound[key]["team_b"]] for key in slot_keys]
                else:
                    opening_keys = [key for key in bound if key.startswith("Round 1_")]
                    opening = [[bound[key]["team_a"], bound[key]["team_b"]] for key in opening_keys]
                teams = [t for pair in opening for t in pair]
                # Source aliases bind exact printed identities. Later opponents
                # are not consulted when constructing entrants or bracket slots.
                team_names = {t: team_names[t] for t in teams}
                contract["opening_source_bindings"] = [key for key in slot_keys] if family == "msi" else opening_keys
                for team in teams:
                    prior_id, proof = _prior_identity(team_names[team], aliases, raw, cutoff, source_sha[str(RAW)])
                    if prior_id != team:
                        raise ValueError(f"opening native ID contradicts exact prior identity: {team_names[team]}")
                    identity_evidence[team] = proof
                contract["identity_evidence"] = {t: identity_evidence[t] for t in teams}
                contract.update(_opening_constraints(text, family, year, team_names, aliases, opening))
                contract["opening_pairings"] = opening
                spec = build_research_event_spec(edition, teams, initial_pairings=opening, source_contract=contract)
                rows = list(bound.values())
                if family == "worlds":
                    nodes = _knockout_slots(text)
                    knockout_keys = {key for key in bound if not key.startswith("Round ")}
                    if set(nodes.values()) != knockout_keys:
                        raise ValueError("Worlds source Cargo nodes do not cover exactly four QF/two SF/one final")
                    # Observed draw/node table is label-only, never part of spec.
                    order = {key: i for i, key in enumerate(nodes.values())}
                    ordered_keys = sorted(bound, key=lambda k: (1, order[k]) if k in order else
                                          (0, list(bound).index(k)))
                    rows = []
                    for key in ordered_keys:
                        row = bound[key]
                        if key.startswith("Round "):
                            row.update(stage="swiss", round=int(key.split("_")[0].split()[-1]))
                        else:
                            row.update(stage="knockout", round={"Quarterfinals": 1, "Semifinals": 2, "Finals": 3}[key.split("_")[0]])
                            row["cargo_slot"] = next(slot for slot, node in nodes.items() if node == key)
                            row["cargo_node"] = key
                            row["cargo_source"] = {"path": str(raw_rule_path),
                                                   "sha256": source_sha[str(raw_rule_path)],
                                                   "basis": "BALANCED_8SE_EXPLICIT_OVERRIDES_AND_NAMED_ROUND_NUMBERED_CARGO_NODES"}
                        rows.append(row)
                edition_titles = {r["title"] for r in rules[family] if r["year"] == year}
                context_proofs = defaultdict(list)
                for mapping in mappings.values():
                    if mapping.get("event_title") in edition_titles and mapping.get("status") == "matched":
                        match = raw_by_id.get(str(mapping["golgg_match_id"]))
                        if match is None or match["tournament_name"] != mapping["golgg_tournament"]:
                            raise ValueError("native context mapping contradicts raw source")
                        context_proofs[match["tournament_name"]].append({
                            "wiki_title": mapping["event_title"], "wiki_series_id": mapping["wiki_series_id"],
                            "source_match_id": str(match["match_id"]), "source_sha256": source_sha[str(RAW)],
                            "mapping_sha256": source_sha[str(SOURCE / "golgg_mapping.jsonl")]})
                # Editions after the mapping compiler cutoff use the exact
                # wiki-date/native-identity join already bound above, not aliases.
                for row in rows:
                    match = raw_by_id.get(str(row["match_id"]))
                    if match is not None:
                        context_proofs[match["tournament_name"]].append({
                            "wiki_title": title, "wiki_series_id": row["wiki_series_id"],
                            "source_match_id": str(match["match_id"]), "source_sha256": source_sha[str(RAW)],
                            "basis": "exact_wiki_date_and_native_identity_join"})
                roster_contexts = set(context_proofs)
                contract["roster_context_bindings"] = dict(context_proofs)
                actual_match_ids = {str(row["match_id"]) for row in rows if row["native_match_source_available"]}
            contract["team_source_names"] = team_names
            contract["team_source_aliases"] = {t: sorted(_alias_names(name, aliases)) for t, name in team_names.items()}
            rosters, evidence = {}, {}
            first_appearances = {}
            for row in sorted(rows, key=lambda row: (row["date"], str(row["match_id"]))):
                for team in (row["team_a"], row["team_b"]):
                    first_appearances.setdefault(team, str(row.get("native_match_id", row["match_id"])))
            for team in teams:
                roster, proof = _lookup_roster(
                    team, cutoff, reference, raw, roster_contexts,
                    actual_match_ids.intersection({first_appearances[team]}),
                    source_sha[str(reference_path)], source_sha[str(RAW)])
                rosters[team] = roster
                evidence[team] = proof
            event = {"id": edition, "edition_id": edition, "year": year, "cutoff": cutoff,
                     "event_title": title, "scope": "retrospective_conditional_scenario", "teams": teams,
                     "rosters": rosters, "roster_basis": "first_event_actual_roster" if any(e["basis"] == "first_event_actual_roster" for e in evidence.values()) else "strictly_prior_observed_same_context",
                     "roster_evidence": evidence, "spec": spec,
                     "labels_path": str(output_dir / "labels" / f"{edition}.json"),
                     "input_path": str(output_dir / "inputs" / f"{edition}.json"),
                     "source_path": str(output_dir / "sources" / f"{edition}.json"),
                     "model_eligibility": _eligibility(year, research_root, cutoff, artifact_contracts), "source_provenance": contract}
            validate_research_event(event)
            events[edition] = event
            pending[edition] = rows
            for model, cell in event["model_eligibility"].items():
                if not cell["available"]:
                    exclusions.append({"event_id": edition, "edition_id": edition, "model": model, "reason": cell["reason"]})
        except (ValueError, KeyError, StopIteration, FileNotFoundError) as exc:
            reason = f"SOURCE_BOUND_EVENT_PREREQUISITE_UNAVAILABLE: {type(exc).__name__}: {exc}"
            events[edition] = {"id": edition, "edition_id": edition, "year": year, "cutoff": cutoff,
                               "available": False, "reason": reason, "spec": None,
                               "model_eligibility": {m: {"available": False, "reason": reason} for m in MODELS}}
            exclusions.extend({"event_id": edition, "edition_id": edition, "model": m, "reason": reason} for m in MODELS)
    # Input freeze comes first for every admitted case. Labels are deliberately
    # impossible to load from input JSON, and later bracket draws live only here.
    for edition, event in events.items():
        if edition in pending:
            _write_exclusive(event["input_path"], event)
            _write_exclusive(event["source_path"], event["source_provenance"])
            event["input_sha256"] = _hash(event["input_path"])
            event["source_snapshot_sha256"] = _hash(event["source_path"])
    for edition, rows in pending.items():
        event = events[edition]
        try:
            labels = build_observed_labels(edition, edition, rows, spec=event["spec"])
            if edition == "lck-2026-road-to-msi":
                source_contract = event["source_provenance"]
                source_text = _wiki_text({"title": "LCK/2026 Season/Road to MSI",
                                          "source": source_contract["rule_revision"]})
                source_placements = {}
                for line in re.findall(r"\{\{TournamentResults/Line\|([^{}]+)\}\}", source_text):
                    fields = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
                    if "place" not in fields or "team" not in fields:
                        continue
                    native_id, _ = _prior_identity(fields["team"].strip(), aliases, raw,
                                                   event["cutoff"], source_sha[str(RAW)])
                    place = [int(value.strip()) for value in fields["place"].split("-")]
                    source_placements[native_id] = [place[0], place[-1]]
                if source_placements != labels["placement_bands"]:
                    raise ValueError("LCK graph route differs from explicitly printed official placements")
                labels["official_placement_source"] = {"path": source_contract["raw_source_path"],
                                                       "sha256": source_contract["source_sha256"]["raw_rules"],
                                                       "basis": source_contract["official_placement_basis"]}
            _write_exclusive(event["labels_path"], labels)
            event["labels_sha256"] = _hash(event["labels_path"])
            event["labels_available"] = True
        except ValueError as exc:
            event["labels_available"] = False
            event["labels_unavailable_reason"] = f"OBSERVED_ROUTE_ADMISSION_FAILED: {exc}"
            exclusions.append({"event_id": edition, "edition_id": edition, "model": "all_scoring_targets", "reason": event["labels_unavailable_reason"]})
    catalogue = _json(SOURCE / "catalog.json")
    manifest = {"schema": "native_c0_tournament_research_cohort_v1", "events": events,
                "exclusions": exclusions, "source_sha256": source_sha,
                "source_gaps": [{"event_id": edition, "edition_id": edition,
                                 "reason": "NATIVE_RAW_OBSERVED_SERIES_ABSENT_SOURCE_CSV_LABEL_RETAINED",
                                 "wiki_series_id": row["wiki_series_id"], "match_id": row["match_id"]}
                                for edition, rows in pending.items() for row in rows
                                if row.get("native_match_source_available") is False],
                "catalogue_coverage": {"source_path": str(SOURCE / "catalog.json"),
                                       "sha256": source_sha[str(SOURCE / "catalog.json")],
                                       "coverage_history_preserved": True,
                                       "source_catalogue": catalogue},
                "policy": "Eight retrospective scenarios, not authenticated historical prestart forecasts; all model checkpoints require separate provider verification."}
    _write_exclusive(output_dir / "manifest.json", manifest)
    return manifest
