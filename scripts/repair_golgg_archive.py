#!/usr/bin/env python3
"""Repair corroborated legacy header reversals offline; retain unresolved history.

Never changes a game, reconstructs missing statistics, or overwrites the input.
The output is not a replacement for complete, source-archived recollection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_prospective_features import _hash_file, iter_matches
from scripts.prospective_sports_features import (
    _compact_series, _flag, _game_days, _history_helpers, _identity, _integer,
    _order, _series_team_ids,
)
from src.utils import golgg_schema as schema

VERSION = "golgg-local-header-repair-v1"
SCORE_KEYS = (("score_1", "t1_score"), ("score_2", "t2_score"))


def repair_record(raw: dict, helpers) -> tuple[dict, dict | None, list[str]]:
    """Change only grounded header fields; failed checks retain the entire row."""
    try:
        identifier = _identity(raw.get("match_id"), "match ID")
        days = _game_days(raw, date.fromisoformat(raw["date"]))
        teams = _series_team_ids(raw)
        scores = [0, 0]
        names = {team: set() for team in teams}
        for index, team in enumerate(teams, 1):
            for key in (f"sname_t{index}", f"name_{index}", f"t{index}_name"):
                name = schema._normalize_team_name(raw.get(key))
                if name:
                    names[team].add(name)
        for game in raw["games"]:
            for side in (1, 2):
                team = str(game[f"t{side}_id"])
                scores[teams.index(team)] += _flag(game.get(f"t{side}_win"), "game winner")
                name = schema._normalize_team_name(game.get(f"t{side}_name"))
                if name:
                    names[team].add(name)
        declared = []
        for aliases in SCORE_KEYS:
            values = {_integer(raw[key], key) for key in aliases if raw.get(key) is not None}
            if len(values) > 1:
                raise ValueError("conflicting raw score aliases")
            declared.append(next(iter(values), None))
        changes = {}
        for side, aliases in enumerate(SCORE_KEYS):
            for key in aliases:
                if raw.get(key) is not None and _integer(raw[key], key) != scores[side]:
                    changes[key] = {"before": raw[key], "after": scores[side]}
            key = f"t{side + 1}_win"
            expected = scores[side] > scores[1 - side]
            if raw.get(key) is not None and _flag(raw[key], key) != expected:
                changes[key] = {"before": raw[key], "after": expected}
        candidate = raw
        evidence = None
        if changes:
            if not any(raw.get(key) is not None for key in ("tid_1", "t1_id", "tid_2", "t2_id")):
                raise ValueError("ungrounded header orientation; refusing directional repair")
            if None in declared or sorted(declared) != sorted(scores):
                raise ValueError("header score totals disagree with maps; missing games are not repaired")
            if scores[0] == scores[1]:
                raise ValueError("draw result contradiction requires source adjudication")
            winner_index = int(scores[1] > scores[0])
            winner, loser = teams[winner_index], teams[1 - winner_index]
            won = schema._normalize_team_name(raw.get("won"))
            if not won or won not in names[winner] or won in names[loser]:
                raise ValueError("winner text does not uniquely corroborate game results")
            candidate = {**raw, **{key: change["after"] for key, change in changes.items()}}
            evidence = {
                "changes": changes, "winner_id": winner, "won": raw["won"],
                "game_ids": [game["game_id"] for game in raw["games"]],
                "score_by_team_id": dict(zip(teams, scores, strict=True)),
            }
        compact = _compact_series(candidate, _order(identifier), helpers, set(), days)
        issues = []
        if any(game.history1 is None or game.history2 is None for game in compact.games):
            issues.append("missing_required_w20_stats")
        return candidate, evidence, issues
    except (KeyError, TypeError, ValueError) as exc:
        # A rejected repair is visible in the ledger, never a skipped source row.
        return raw, None, [str(exc)]


def repair_archive(source: Path, output: Path) -> dict:
    """Write a separate inventory-preserving candidate and per-record audit trail."""
    source, output = Path(source).resolve(), Path(output).resolve()
    source_hash = _hash_file(source)
    output.mkdir(parents=True, exist_ok=False)
    helpers = _history_helpers()
    counts = Counter()
    issues_count = Counter()
    seen_series = set()
    seen_games = set()
    temporary = output / "matches.incomplete.json"
    with temporary.open("x", encoding="utf-8") as records, (output / "changes.jsonl").open("x", encoding="utf-8") as changes, (output / "unresolved.jsonl").open("x", encoding="utf-8") as unresolved:
        records.write("[\n")
        for row_number, raw in enumerate(iter_matches(source), 1):
            if not isinstance(raw, dict):
                raise ValueError(f"source row {row_number} is not a match object")
            identifier = _identity(raw.get("match_id"), "match ID")
            if identifier in seen_series:
                raise ValueError(f"duplicate source match ID: {identifier}")
            seen_series.add(identifier)
            fixed, evidence, issues = repair_record(raw, helpers)
            # Structural defects inside a series remain ledgered; duplicates
            # across series are an archive-level error, not separate matches.
            games = raw.get("games")
            local_games = {
                str(game["game_id"]) for game in (games if isinstance(games, list) else [])
                if isinstance(game, dict) and game.get("game_id") is not None
            }
            if local_games & seen_games:
                raise ValueError(f"game IDs shared across source series: {identifier}")
            seen_games.update(local_games)
            entry = {"source_row": row_number, "match_id": identifier}
            if evidence:
                changes.write(json.dumps({**entry, **evidence}, allow_nan=False) + "\n")
                counts["repaired_series"] += 1
                counts["changed_fields"] += len(evidence["changes"])
            if issues:
                unresolved.write(json.dumps({**entry, "issues": issues}, allow_nan=False) + "\n")
                counts["unresolved_series"] += 1
                issues_count.update(issues)
            records.write((",\n" if row_number > 1 else "") + json.dumps(fixed, ensure_ascii=False, allow_nan=False))
            counts["series"] += 1
        records.write("\n]\n")
    if _hash_file(source) != source_hash:
        raise ValueError("source archive changed during repair; candidate not finalized")
    temporary.rename(output / "matches.json")
    report = {
        "schema_version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked_unresolved_history" if counts["unresolved_series"] else "complete_local_metadata_repair",
        "source": str(source), "source_sha256": source_hash,
        "output_sha256": _hash_file(output / "matches.json"),
        **{key: counts[key] for key in ("series", "repaired_series", "changed_fields", "unresolved_series")},
        "unresolved_reasons": dict(issues_count),
        "games_changed": 0,
        "missing_measurements_imputed": 0,
        "model_artifacts_changed": False,
        "files": {name: _hash_file(output / name) for name in ("changes.jsonl", "unresolved.jsonl")},
        "code_sha256": {name: _hash_file(ROOT / name) for name in (
            "scripts/repair_golgg_archive.py", "scripts/export_prospective_features.py",
            "scripts/prospective_sports_features.py", "src/utils/golgg_schema.py",
        )},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    args = parser.parse_args()
    print(json.dumps(repair_archive(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
