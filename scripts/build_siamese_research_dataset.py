#!/usr/bin/env python3
"""Reconstruct complete W20 snapshots for a retrospective architecture diagnostic.

This is NOT a point-in-time production dataset: legacy rating provenance and
roster announcement times are unavailable. Missing values are never imputed.
All exclusions are counted. Raw sources and frozen model artifacts stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import importlib.util
from itertools import groupby
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.competition_tiers import classify_competition
from src.models.symmetric_series import build_feature_mapping
from src.utils import golgg_schema as schema


def _history_helpers():
    # Reuse the historical W20 aggregation definition, but never its missing-data
    # fallbacks or same-day sequential replay. Validate each game before calling.
    path = ROOT / "scripts/06_metamodel/06i_best_metamodel_config_search.py"
    spec = importlib.util.spec_from_file_location("research_w20_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _complete_game(game: dict[str, Any], team: str) -> bool:
    if team not in (str(game.get("t1_id")), str(game.get("t2_id"))):
        return False
    side = "t1" if str(game.get("t1_id")) == team else "t2"
    players = game.get(f"{side}_players") or {}
    if (
        len(players) != 5
        or game.get("t1_win") not in (True, False)
        or game.get("t2_win") not in (True, False)
    ):
        return False
    if bool(game["t1_win"]) == bool(game["t2_win"]):
        return False
    values = [game.get("game_duration")]
    values.extend(
        (game.get(f"{side}_stats") or {}).get(key)
        for key in ("towers", "nashors", "gold")
    )
    values.extend(
        (p.get("stats") or {}).get(key)
        for p in players.values()
        for key in ("kills", "deaths", "dpm", "vspm", "gd@15")
    )
    try:
        return (
            all(v is not None and np.isfinite(float(v)) for v in values)
            and float(values[0]) > 0
        )
    except (TypeError, ValueError):
        return False


def build_dataset(
    ratings: pd.DataFrame, matches: list[dict[str, Any]], *, day_batched_ratings: bool = False
) -> tuple[pd.DataFrame, dict]:
    """Align one completed series per ID; build W20 from strictly earlier dates."""
    required = {"golgg_match_id", "date", "team1_id", "team2_id", "BoN", "y_true"}
    if missing := required.difference(ratings):
        raise ValueError(f"missing rating metadata: {sorted(missing)}")
    ratings = ratings.copy()
    for name in ("golgg_match_id", "team1_id", "team2_id"):
        if ratings[name].isna().any():
            raise ValueError(f"missing identity: {name}")
        ratings[name] = ratings[name].astype(str)
    ids = [str(m["match_id"]) for m in matches]
    if ratings.golgg_match_id.duplicated().any() or len(set(ids)) != len(ids):
        raise ValueError("duplicate match IDs in source")
    raw_by_id = dict(zip(ids, matches, strict=True))
    rated = ratings.set_index("golgg_match_id").to_dict("index")
    helpers = _history_helpers()
    ordered = sorted(matches, key=lambda m: (m["date"], str(m["match_id"])))
    # A missing game remains in the last-20 window as None, rather than silently
    # turning W20 into 'last 20 games with convenient data'.
    histories = defaultdict(lambda: deque(maxlen=20))
    player_counts = Counter()
    excluded = Counter()
    output = []
    for day, grouped in groupby(ordered, key=lambda m: m["date"]):
        period = list(grouped)
        effective_date = pd.Timestamp(day).date()
        teams = Counter(
            t for m in period for t in (schema.team1_id(m), schema.team2_id(m))
        )
        players = Counter(
            p for m in period for p in (*schema.players1(m), *schema.players2(m))
        )
        for m in period:
            mid = str(m["match_id"])
            if mid not in rated:
                continue
            row = dict(rated[mid])
            t1, t2 = schema.team1_id(m), schema.team2_id(m)
            roster = (*schema.players1(m), *schema.players2(m))
            game_rows = schema.games(m)
            bo = schema.best_of(m)
            valid_games = game_rows and all(
                {str(g.get("t1_id")), str(g.get("t2_id"))} == {t1, t2}
                and g.get("t1_win") in (True, False)
                and g.get("t2_win") in (True, False)
                and bool(g.get("t1_win")) != bool(g.get("t2_win"))
                and not g.get("draw")
                for g in game_rows
            )
            s1 = schema.score1(m) if valid_games else 0
            s2 = schema.score2(m) if valid_games else 0
            if (
                bo not in (1, 3, 5)
                or not valid_games
                or max(s1, s2) != bo // 2 + 1
                or min(s1, s2) > bo // 2
            ):
                excluded["invalid_or_incomplete_series"] += 1
                continue
            if (
                row["team1_id"] != t1
                or row["team2_id"] != t2
                or str(row["date"]) != str(day)
                or row["BoN"] != bo
                or row["y_true"] != int(s1 > s2)
            ):
                excluded["rating_metadata_mismatch"] += 1
                continue
            if len(set(roster)) != 10:
                excluded["incomplete_roster"] += 1
                continue
            if not day_batched_ratings and (
                any(teams[t] > 1 for t in (t1, t2))
                or any(players[p] > 1 for p in roster)
            ):
                excluded["same_day_participant"] += 1
                continue
            if any(
                not histories[t] or any(g is None for g in histories[t])
                for t in (t1, t2)
            ):
                excluded["incomplete_w20_history"] += 1
                continue
            for side, team in ((1, t1), (2, t2)):
                row.update(
                    {
                        f"t{side}_rolling_{key}": value
                        for key, value in helpers.average_history(
                            histories[team]
                        ).items()
                    }
                )
                row[f"history_games_{side}"] = len(histories[team])
            row.update(
                golgg_match_id=mid,
                best_of=bo,
                tournament=schema.match_tournament(m),
                competition_tier=classify_competition(
                    schema.match_tournament(m), effective_date
                ).tier.value,
                roster_min_prior_series=min(player_counts[p] for p in roster),
            )
            try:
                build_feature_mapping(row, best_of=bo)
            except ValueError:
                excluded["invalid_rating_features"] += 1
                continue
            output.append(row)
        # Freeze every prediction in a day before consuming any result that day.
        for m in period:
            participants = set()
            for game in schema.games(m):
                for side in ("t1_players", "t2_players"):
                    participants.update(schema._players_from_game_side(game, side))
                for team in (schema.team1_id(m), schema.team2_id(m)):
                    if _complete_game(game, team):
                        helpers.update_team_history(histories, team, game, 20)
                    else:
                        histories[team].append(None)
            player_counts.update(participants)
    excluded["no_raw_match"] = sum(mid not in raw_by_id for mid in rated)
    frame = pd.DataFrame(output)
    audit = {
        "rating_rows": len(ratings),
        "raw_series": len(matches),
        "included_rows": len(frame),
        "exclusions": dict(excluded),
        "window_games": 20,
        "same_day_policy": (
            "predict before entire day; verified day-batched ratings"
            if day_batched_ratings else
            "predict before entire day; exclude repeated team/player dates from legacy ratings"
        ),
        "missing_policy": "exclude if any required statistic in last 20 games is unavailable; no zero fill",
        "temporal_status": "retrospective_diagnostic_only",
        "promotion_blockers": [
            *([] if day_batched_ratings else [
                "legacy rating source/version and pre-update computation unverifiable"
            ]),
            "rosters observed from games, not timestamped pre-match announcements",
            "source availability, prediction and quote timestamps unavailable",
            "historical cohort previously inspected; not an untouched holdout",
        ],
    }
    if len(frame):
        audit["date_min"] = frame.date.min()
        audit["date_max"] = frame.date.max()
        audit["modern_2024_rows"] = int((frame.date >= "2024-01-01").sum())
        audit["series_formats"] = frame.best_of.value_counts().sort_index().to_dict()
    assert len(frame) + sum(excluded.values()) == len(ratings)
    return frame, audit


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratings", type=Path, default=ROOT / "data/golgg_y_predicts.csv"
    )
    parser.add_argument(
        "--matches", type=Path, default=ROOT / "data/golgg_matches.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ratings = pd.read_csv(
        args.ratings, dtype={"golgg_match_id": str, "team1_id": str, "team2_id": str}
    )
    with args.matches.open() as stream:
        matches = json.load(stream)
    frame, audit = build_dataset(ratings, matches)
    audit["source_sha256"] = {
        str(p): sha256(p) for p in (args.ratings, args.matches, Path(__file__))
    }
    frame.to_csv(args.output_dir / "snapshots.csv", index=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(audit, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
