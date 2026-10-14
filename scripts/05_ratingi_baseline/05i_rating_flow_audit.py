"""Audit temporal competition flow and Player Glicko-2 performance (EXP-075).

This runner is deliberately read-only with respect to rating artefacts. It joins the
existing weight-one Player Glicko-2 probabilities by match ID, reconstructs domestic
competition affiliations using only earlier calendar dates, and fits small diagnostic
calibrators on 2021-2023 for descriptive evaluation on 2024+ data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.competition_tiers import (
    CompetitionIdentity,
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.ratings.glicko import GlickoRating
from src.utils.golgg_schema import (
    game_score_for_match_team1,
    games,
    match_tournament,
    players1,
    players2,
    team1_id,
    team1_name,
    team2_id,
    team2_name,
)

MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
RATINGS_PATH = PROJECT_ROOT / "data" / "golgg_y_predicts.csv"
ODDS_PATH = PROJECT_ROOT / "data" / "odds.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "experiments" / "exp075_rating_flow_audit"
)
SELECTION_START = date(2021, 1, 1)
DIAGNOSTIC_START = date(2024, 1, 1)
PROBABILITY_EPSILON = 0.001
RANDOM_SEED = 75
UNKNOWN_FAMILY = "unknown"
LOWER_TIERS = {
    CompetitionTier.MINOR_TOP_LEVEL,
    CompetitionTier.REGIONAL,
    CompetitionTier.DEVELOPMENT,
}
KNOWN_AFFILIATION_TIERS = tuple(
    tier
    for tier in CompetitionTier
    if tier not in {CompetitionTier.INTERNATIONAL, CompetitionTier.UNKNOWN}
)
COHORTS = (
    "overall",
    "major_major",
    "regional_regional",
    "development",
    "known_cross_league",
    "major_vs_lower",
)
CONTEXTS = (
    "major_vs_lower",
    "known_cross_league",
    "development",
    "major_major",
    "regional_regional",
    "other",
)


@dataclass(frozen=True)
class MatchRecord:
    match_id: str
    match_date: date
    tournament: str
    competition: CompetitionIdentity
    team_1: str
    team_2: str
    team_1_name: str
    team_2_name: str
    players_1: tuple[str, ...]
    players_2: tuple[str, ...]
    player_names: tuple[tuple[str, str], ...]
    target: int
    game_results: tuple[int, ...]


@dataclass(frozen=True)
class DomesticAffiliation:
    family: str
    tier: CompetitionTier
    source_date: date
    source_tournament: str
    source_team: str
    matched_rule: str


@dataclass(frozen=True)
class RatingPrediction:
    match_id: str
    match_date: date
    target: int
    probability: float
    team_1: str | None
    team_2: str | None


def parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    parser.add_argument("--rating-predictions", type=Path, default=RATINGS_PATH)
    parser.add_argument(
        "--odds",
        type=Path,
        default=ODDS_PATH,
        help="Optional mapping source; only golgg_match_id is read, never odds values.",
    )
    parser.add_argument(
        "--no-odds",
        action="store_true",
        help="Do not load an odds mapping file.",
    )
    parser.add_argument(
        "--diagnostic-require-odds",
        action="store_true",
        help="Restrict diagnostic fitting/evaluation to match IDs present in --odds.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--selection-start", type=parse_iso_date, default=SELECTION_START)
    parser.add_argument("--holdout-start", type=parse_iso_date, default=DIAGNOSTIC_START)
    parser.add_argument("--ridge-c", type=float, default=1.0)
    parser.add_argument("--probability-epsilon", type=float, default=PROBABILITY_EPSILON)
    parser.add_argument("--active-days", type=int, default=60)
    parser.add_argument("--top-players", type=int, default=100)
    parser.add_argument("--low-evidence-games", type=int, default=10)
    args = parser.parse_args()
    if args.selection_start >= args.holdout_start:
        parser.error("--selection-start must be earlier than --holdout-start")
    if args.ridge_c <= 0:
        parser.error("--ridge-c must be positive")
    if not 0 < args.probability_epsilon < 0.5:
        parser.error("--probability-epsilon must be in (0, 0.5)")
    if args.active_days < 0:
        parser.error("--active-days must be non-negative")
    if args.top_players < 1:
        parser.error("--top-players must be positive")
    if args.low_evidence_games < 0:
        parser.error("--low-evidence-games must be non-negative")
    if args.no_odds and args.diagnostic_require_odds:
        parser.error("--diagnostic-require-odds cannot be combined with --no-odds")
    return args


def canonical_match_id(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty match ID")
    if text.isdigit():
        return str(int(text))
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"invalid match ID {value!r}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"non-integral match ID {value!r}")
    return str(int(number))


def _complete_counts(counts: Counter[str], keys: Sequence[str]) -> dict[str, int]:
    """Return a stable count mapping that retains explicit zeroes."""
    return {key: int(counts[key]) for key in keys}


def _player_names(match: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    names: dict[str, str] = {}
    for game in games(match):
        for side_key in ("t1_players", "t2_players"):
            payload = game.get(side_key)
            if not isinstance(payload, dict):
                continue
            for player in payload.values():
                if not isinstance(player, dict):
                    continue
                raw_id = player.get("player_id") or player.get("id")
                raw_name = player.get("player_name") or player.get("name")
                if raw_id is not None and raw_name:
                    names[str(raw_id)] = str(raw_name)
    return tuple(sorted(names.items()))


def load_matches(path: Path) -> tuple[list[MatchRecord], dict[str, int]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("matches input must contain a JSON list")

    records: list[MatchRecord] = []
    counts = Counter(raw_matches=len(payload))
    seen_ids: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            counts["invalid_non_object"] += 1
            continue
        if raw.get("draw"):
            counts["excluded_draw"] += 1
            continue
        match_id = canonical_match_id(raw.get("match_id"))
        if match_id in seen_ids:
            raise ValueError(f"duplicate match ID in matches input: {match_id}")
        seen_ids.add(match_id)
        match_date = date.fromisoformat(str(raw["date"]))
        tournament = str(match_tournament(raw) or "").strip()
        identity = classify_competition(tournament, match_date)
        results = tuple(
            game_score_for_match_team1(raw, game) for game in games(raw)
        )
        if results:
            target = int(sum(results) > len(results) / 2)
        elif raw.get("t1_win") is not None:
            target = int(bool(raw.get("t1_win")))
        else:
            counts["excluded_missing_target"] += 1
            continue
        records.append(
            MatchRecord(
                match_id=match_id,
                match_date=match_date,
                tournament=tournament,
                competition=identity,
                team_1=str(team1_id(raw)),
                team_2=str(team2_id(raw)),
                team_1_name=str(team1_name(raw) or team1_id(raw)),
                team_2_name=str(team2_name(raw) or team2_id(raw)),
                players_1=tuple(str(value) for value in players1(raw)),
                players_2=tuple(str(value) for value in players2(raw)),
                player_names=_player_names(raw),
                target=target,
                game_results=results,
            )
        )
    records.sort(key=lambda record: (record.match_date, int(record.match_id)))
    counts["non_draw_records"] = len(records)
    return records, _complete_counts(
        counts,
        (
            "raw_matches",
            "invalid_non_object",
            "excluded_draw",
            "excluded_missing_target",
            "non_draw_records",
        ),
    )


def _optional_int(value: str | None, field: str, match_id: str) -> int | None:
    if value is None or not value.strip():
        return None
    number = float(value)
    if not number.is_integer() or int(number) not in (0, 1):
        raise ValueError(f"{field} is not binary for rating match {match_id}")
    return int(number)


def _optional_team_id(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return canonical_match_id(value)


def load_rating_predictions(
    path: Path,
) -> tuple[dict[str, RatingPrediction], dict[str, int]]:
    predictions: dict[str, RatingPrediction] = {}
    counts = Counter()
    required = {"golgg_match_id", "date", "y_true", "player_gl"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"rating predictions missing columns: {sorted(missing)}")
        for row in reader:
            counts["rows"] += 1
            match_id = canonical_match_id(row["golgg_match_id"])
            if match_id in predictions:
                raise ValueError(f"duplicate rating prediction match ID: {match_id}")
            target = _optional_int(row.get("y_true"), "y_true", match_id)
            probability_text = (row.get("player_gl") or "").strip()
            if target is None or not probability_text:
                counts["excluded_missing_target_or_probability"] += 1
                continue
            probability = float(probability_text)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"Player Glicko probability outside [0, 1] for match {match_id}"
                )
            predictions[match_id] = RatingPrediction(
                match_id=match_id,
                match_date=date.fromisoformat(row["date"]),
                target=target,
                probability=probability,
                team_1=_optional_team_id(row.get("team1_id")),
                team_2=_optional_team_id(row.get("team2_id")),
            )
    counts["eligible_rows"] = len(predictions)
    return predictions, _complete_counts(
        counts,
        ("rows", "excluded_missing_target_or_probability", "eligible_rows"),
    )


def load_odds_match_ids(path: Path | None) -> tuple[set[str] | None, dict[str, object]]:
    if path is None:
        return None, {"status": "disabled", "rows": 0, "unique_match_ids": 0}
    if not path.exists():
        return None, {"status": "not_found", "rows": 0, "unique_match_ids": 0}
    identifiers: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "golgg_match_id" not in (reader.fieldnames or ()):
            raise ValueError("odds mapping missing golgg_match_id")
        for row in reader:
            rows += 1
            raw_id = (row.get("golgg_match_id") or "").strip()
            if raw_id:
                identifiers.add(canonical_match_id(raw_id))
    return identifiers, {
        "status": "loaded",
        "rows": rows,
        "unique_match_ids": len(identifiers),
    }


def _affiliation_fields(
    prefix: str, affiliation: DomesticAffiliation | None
) -> dict[str, object]:
    if affiliation is None:
        return {
            f"{prefix}_family": UNKNOWN_FAMILY,
            f"{prefix}_tier": CompetitionTier.UNKNOWN.value,
            f"{prefix}_affiliation_date": None,
            f"{prefix}_affiliation_tournament": None,
            f"{prefix}_affiliation_team": None,
        }
    return {
        f"{prefix}_family": affiliation.family,
        f"{prefix}_tier": affiliation.tier.value,
        f"{prefix}_affiliation_date": affiliation.source_date.isoformat(),
        f"{prefix}_affiliation_tournament": affiliation.source_tournament,
        f"{prefix}_affiliation_team": affiliation.source_team,
    }


def _is_known(affiliation: DomesticAffiliation | None) -> bool:
    return affiliation is not None and affiliation.family != UNKNOWN_FAMILY


def _cohort_flags(
    competition: CompetitionIdentity,
    affiliation_1: DomesticAffiliation | None,
    affiliation_2: DomesticAffiliation | None,
) -> dict[str, bool]:
    tier_1 = affiliation_1.tier if affiliation_1 else CompetitionTier.UNKNOWN
    tier_2 = affiliation_2.tier if affiliation_2 else CompetitionTier.UNKNOWN
    return {
        "cohort_overall": True,
        "cohort_major_major": tier_1 is CompetitionTier.MAJOR
        and tier_2 is CompetitionTier.MAJOR,
        "cohort_regional_regional": tier_1 is CompetitionTier.REGIONAL
        and tier_2 is CompetitionTier.REGIONAL,
        "cohort_development": competition.tier is CompetitionTier.DEVELOPMENT
        or tier_1 is CompetitionTier.DEVELOPMENT
        or tier_2 is CompetitionTier.DEVELOPMENT,
        "cohort_known_cross_league": competition.scope is CompetitionScope.CROSS_LEAGUE
        and _is_known(affiliation_1)
        and _is_known(affiliation_2),
        "cohort_major_vs_lower": (
            tier_1 is CompetitionTier.MAJOR and tier_2 in LOWER_TIERS
        )
        or (tier_2 is CompetitionTier.MAJOR and tier_1 in LOWER_TIERS),
    }


def _exclusive_context(row: dict[str, object]) -> str:
    for context in CONTEXTS[:-1]:
        if bool(row[f"cohort_{context}"]):
            return context
    return "other"


def reconstruct_temporal_flow(
    records: Sequence[MatchRecord],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, DomesticAffiliation],
    dict[str, DomesticAffiliation],
    dict[str, object],
]:
    team_domestic: dict[str, DomesticAffiliation] = {}
    player_domestic: dict[str, DomesticAffiliation] = {}
    audit_rows: list[dict[str, object]] = []
    edge_events: list[dict[str, object]] = []
    counts = Counter()

    for match_date, date_group in groupby(records, key=lambda item: item.match_date):
        day = list(date_group)
        team_candidates: defaultdict[str, list[DomesticAffiliation]] = defaultdict(list)
        player_candidates: defaultdict[str, list[DomesticAffiliation]] = defaultdict(list)

        for record in day:
            affiliation_1 = team_domestic.get(record.team_1)
            affiliation_2 = team_domestic.get(record.team_2)
            row: dict[str, object] = {
                "match_id": record.match_id,
                "date": record.match_date.isoformat(),
                "tournament": record.tournament,
                "competition_family": record.competition.family,
                "competition_tier": record.competition.tier.value,
                "competition_scope": record.competition.scope.value,
                "matched_rule": record.competition.matched_rule,
                "team_1_id": record.team_1,
                "team_1_name": record.team_1_name,
                "team_2_id": record.team_2,
                "team_2_name": record.team_2_name,
                "target": record.target,
            }
            row.update(_affiliation_fields("team_1", affiliation_1))
            row.update(_affiliation_fields("team_2", affiliation_2))
            row.update(_cohort_flags(record.competition, affiliation_1, affiliation_2))
            row["diagnostic_context"] = _exclusive_context(row)
            audit_rows.append(row)

            if record.competition.scope is CompetitionScope.CROSS_LEAGUE:
                counts["cross_league_matches"] += 1
                if not _is_known(affiliation_1) or not _is_known(affiliation_2):
                    counts["bridge_excluded_unknown_affiliation"] += 1
                elif affiliation_1.family == affiliation_2.family:
                    counts["bridge_excluded_same_family"] += 1
                else:
                    edge_events.append(
                        {
                            "edge_type": "direct_bridge",
                            "source_family": affiliation_1.family,
                            "source_tier": affiliation_1.tier.value,
                            "target_family": affiliation_2.family,
                            "target_tier": affiliation_2.tier.value,
                            "date": record.match_date.isoformat(),
                            "match_id": record.match_id,
                            "tournament": record.tournament,
                            "player_id": None,
                            "source_team": record.team_1,
                            "target_team": record.team_2,
                        }
                    )
                    counts["direct_bridge_events"] += 1

            if (
                record.competition.scope is CompetitionScope.DOMESTIC
                and record.competition.family != UNKNOWN_FAMILY
                and record.competition.tier is not CompetitionTier.UNKNOWN
            ):
                for team_id, player_ids in (
                    (record.team_1, record.players_1),
                    (record.team_2, record.players_2),
                ):
                    candidate = DomesticAffiliation(
                        family=record.competition.family,
                        tier=record.competition.tier,
                        source_date=match_date,
                        source_tournament=record.tournament,
                        source_team=team_id,
                        matched_rule=record.competition.matched_rule,
                    )
                    if team_id:
                        team_candidates[team_id].append(candidate)
                    for player_id in player_ids:
                        player_candidates[player_id].append(candidate)

        for team_id, candidates in sorted(team_candidates.items()):
            destinations = {(item.family, item.tier) for item in candidates}
            if len(destinations) != 1:
                counts["ambiguous_same_day_team_affiliations"] += 1
                continue
            team_domestic[team_id] = sorted(
                candidates, key=lambda item: (item.source_tournament, item.source_team)
            )[0]

        for player_id, candidates in sorted(player_candidates.items()):
            destinations = {(item.family, item.tier) for item in candidates}
            if len(destinations) != 1:
                counts["ambiguous_same_day_player_affiliations"] += 1
                continue
            destination = sorted(
                candidates, key=lambda item: (item.source_tournament, item.source_team)
            )[0]
            previous = player_domestic.get(player_id)
            if previous is not None and previous.family != destination.family:
                edge_events.append(
                    {
                        "edge_type": "player_transfer",
                        "source_family": previous.family,
                        "source_tier": previous.tier.value,
                        "target_family": destination.family,
                        "target_tier": destination.tier.value,
                        "date": match_date.isoformat(),
                        "match_id": None,
                        "tournament": destination.source_tournament,
                        "player_id": player_id,
                        "source_team": previous.source_team,
                        "target_team": destination.source_team,
                    }
                )
                counts["player_transfer_events"] += 1
            player_domestic[player_id] = destination

    counts["teams_with_domestic_affiliation"] = len(team_domestic)
    counts["players_with_domestic_affiliation"] = len(player_domestic)
    count_summary = _complete_counts(
        counts,
        (
            "cross_league_matches",
            "bridge_excluded_unknown_affiliation",
            "bridge_excluded_same_family",
            "direct_bridge_events",
            "ambiguous_same_day_team_affiliations",
            "ambiguous_same_day_player_affiliations",
            "player_transfer_events",
            "teams_with_domestic_affiliation",
            "players_with_domestic_affiliation",
        ),
    )
    return audit_rows, edge_events, team_domestic, player_domestic, count_summary


def align_predictions(
    records: Sequence[MatchRecord],
    audit_rows: Sequence[dict[str, object]],
    ratings: dict[str, RatingPrediction],
    odds_ids: set[str] | None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    record_by_id = {record.match_id: record for record in records}
    audit_by_id = {str(row["match_id"]): row for row in audit_rows}
    counts = Counter()
    joined: list[dict[str, object]] = []
    for match_id, prediction in sorted(ratings.items(), key=lambda item: int(item[0])):
        record = record_by_id.get(match_id)
        if record is None:
            counts["rating_rows_without_match"] += 1
            continue
        if prediction.match_date != record.match_date:
            raise RuntimeError(
                f"rating/match date mismatch for {match_id}: "
                f"{prediction.match_date} != {record.match_date}"
            )
        if prediction.target != record.target:
            raise RuntimeError(
                f"rating/match target mismatch for {match_id}: "
                f"{prediction.target} != {record.target}"
            )
        if prediction.team_1 is not None and prediction.team_1 != record.team_1:
            raise RuntimeError(f"rating/match team-1 mismatch for {match_id}")
        if prediction.team_2 is not None and prediction.team_2 != record.team_2:
            raise RuntimeError(f"rating/match team-2 mismatch for {match_id}")
        row = dict(audit_by_id[match_id])
        row["player_glicko_probability"] = prediction.probability
        row["odds_mapped"] = odds_ids is not None and match_id in odds_ids
        joined.append(row)
    matched_ids = {str(row["match_id"]) for row in joined}
    counts["aligned_predictions"] = len(joined)
    counts["matches_without_rating_prediction"] = sum(
        record.match_id not in matched_ids for record in records
    )
    counts["aligned_odds_mapped"] = sum(bool(row["odds_mapped"]) for row in joined)
    count_summary = _complete_counts(
        counts,
        (
            "rating_rows_without_match",
            "aligned_predictions",
            "matches_without_rating_prediction",
            "aligned_odds_mapped",
        ),
    )
    return (
        sorted(joined, key=lambda row: (str(row["date"]), int(str(row["match_id"])))),
        count_summary,
    )


def probability_metrics(
    rows: Sequence[dict[str, object]],
    probability_key: str,
    epsilon: float = PROBABILITY_EPSILON,
) -> dict[str, object]:
    if not rows:
        return {
            "n": 0,
            "date_min": None,
            "date_max": None,
            "positives": 0,
            "negatives": 0,
            "event_rate": None,
            "mean_probability": None,
            "calibration_gap": None,
            "absolute_calibration_gap": None,
            "log_loss": None,
            "brier": None,
            "auc": None,
        }
    labels = np.asarray([int(row["target"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row[probability_key]) for row in rows], dtype=float)
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    event_rate = float(np.mean(labels))
    mean_probability = float(np.mean(probabilities))
    positives = int(np.sum(labels))
    auc = float(roc_auc_score(labels, probabilities)) if 0 < positives < len(labels) else None
    return {
        "n": len(rows),
        "date_min": min(str(row["date"]) for row in rows),
        "date_max": max(str(row["date"]) for row in rows),
        "positives": positives,
        "negatives": len(rows) - positives,
        "event_rate": event_rate,
        "mean_probability": mean_probability,
        "calibration_gap": mean_probability - event_rate,
        "absolute_calibration_gap": abs(mean_probability - event_rate),
        "log_loss": float(
            np.mean(
                -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
            )
        ),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "auc": auc,
    }


def cohort_metrics(
    rows: Sequence[dict[str, object]], epsilon: float = PROBABILITY_EPSILON
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for cohort in COHORTS:
        selected = [row for row in rows if bool(row[f"cohort_{cohort}"])]
        result.append(
            {
                "cohort": cohort,
                **probability_metrics(selected, "player_glicko_probability", epsilon),
            }
        )
    return result
def major_side_calibration(
    rows: Sequence[dict[str, object]],
    probability_key: str,
    epsilon: float = PROBABILITY_EPSILON,
) -> dict[str, object]:
    oriented: list[dict[str, object]] = []
    for row in rows:
        if not bool(row["cohort_major_vs_lower"]):
            continue
        major_is_team_1 = row["team_1_tier"] == CompetitionTier.MAJOR.value
        probability = float(row[probability_key])
        oriented.append(
            {
                "date": row["date"],
                "target": int(row["target"]) if major_is_team_1 else 1 - int(row["target"]),
                "major_win_probability": probability if major_is_team_1 else 1.0 - probability,
            }
        )
    metrics = probability_metrics(oriented, "major_win_probability", epsilon)
    return {
        **metrics,
        "orientation": "probability and outcome are both oriented to the major-tier team",
    }




def _known_tiers(row: dict[str, object]) -> bool:
    return (
        row["team_1_tier"] != CompetitionTier.UNKNOWN.value
        and row["team_2_tier"] != CompetitionTier.UNKNOWN.value
    )


def diagnostic_sample(
    rows: Sequence[dict[str, object]],
    selection_start: date,
    holdout_start: date,
    require_odds: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    selection: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    counts = Counter()
    for row in rows:
        row_date = date.fromisoformat(str(row["date"]))
        if row_date < selection_start:
            counts["excluded_before_selection_start"] += 1
            continue
        if not _known_tiers(row):
            counts["excluded_unknown_team_affiliation"] += 1
            continue
        if require_odds and not bool(row["odds_mapped"]):
            counts["excluded_not_odds_mapped"] += 1
            continue
        if row_date < holdout_start:
            selection.append(row)
        else:
            holdout.append(row)
    counts["selection_eligible"] = len(selection)
    counts["holdout_eligible"] = len(holdout)
    return selection, holdout, _complete_counts(
        counts,
        (
            "excluded_before_selection_start",
            "excluded_unknown_team_affiliation",
            "excluded_not_odds_mapped",
            "selection_eligible",
            "holdout_eligible",
        ),
    )


def _logit(probability: float, epsilon: float) -> float:
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def _tier_difference(row: dict[str, object], tier: CompetitionTier) -> float:
    return float(row["team_1_tier"] == tier.value) - float(
        row["team_2_tier"] == tier.value
    )


def model_matrix(
    rows: Sequence[dict[str, object]], model_id: str, epsilon: float
) -> tuple[np.ndarray, list[str]]:
    tier_names = [f"tier_difference::{tier.value}" for tier in KNOWN_AFFILIATION_TIERS]
    if model_id == "baseline_logit_calibration":
        feature_names = ["baseline_logit"]
        values = [[_logit(float(row["player_glicko_probability"]), epsilon)] for row in rows]
    elif model_id == "tier_offset":
        feature_names = ["baseline_logit", *tier_names]
        values = [
            [
                _logit(float(row["player_glicko_probability"]), epsilon),
                *[_tier_difference(row, tier) for tier in KNOWN_AFFILIATION_TIERS],
            ]
            for row in rows
        ]
    elif model_id == "tier_offset_context_slope":
        feature_names = [
            *[f"baseline_logit_x_context::{context}" for context in CONTEXTS],
            *tier_names,
        ]
        values = []
        for row in rows:
            baseline_logit = _logit(float(row["player_glicko_probability"]), epsilon)
            context = str(row["diagnostic_context"])
            values.append(
                [
                    *[
                        baseline_logit if context == candidate else 0.0
                        for candidate in CONTEXTS
                    ],
                    *[_tier_difference(row, tier) for tier in KNOWN_AFFILIATION_TIERS],
                ]
            )
    else:
        raise ValueError(f"unknown diagnostic model: {model_id}")
    return np.asarray(values, dtype=float), feature_names


def _with_probabilities(
    rows: Sequence[dict[str, object]], probabilities: Iterable[float]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row, probability in zip(rows, probabilities, strict=True):
        output.append({**row, "diagnostic_probability": float(probability)})
    return output


def _append_model_metric_rows(
    result_rows: list[dict[str, object]],
    *,
    model: str,
    split: str,
    fitted_on: str,
    ridge_c: float | None,
    symmetry_contract: str,
    feature_names: Sequence[str],
    coefficients: dict[str, float],
    evaluated: Sequence[dict[str, object]],
    probability_key: str,
    epsilon: float,
) -> None:
    for cohort in COHORTS:
        selected = [row for row in evaluated if bool(row[f"cohort_{cohort}"])]
        result_rows.append(
            {
                "model": model,
                "split": split,
                "cohort": cohort,
                "fitted_on": fitted_on,
                "ridge_c": ridge_c,
                "fit_intercept": False,
                "symmetry_contract": symmetry_contract,
                "feature_names_json": json.dumps(list(feature_names)),
                "coefficients_json": json.dumps(coefficients, sort_keys=True),
                **probability_metrics(selected, probability_key, epsilon),
            }
        )


def fit_diagnostic_models(
    selection: Sequence[dict[str, object]],
    holdout: Sequence[dict[str, object]],
    ridge_c: float,
    epsilon: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, np.ndarray],
]:
    if not selection:
        raise ValueError("no eligible 2021-2023 selection observations")
    if not holdout:
        raise ValueError("no eligible 2024+ diagnostic observations")
    labels = np.asarray([int(row["target"]) for row in selection], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("diagnostic selection data must contain both target classes")

    control_id = "player_glicko_external_weight_one"
    control_coefficients = {"baseline_logit": 1.0}
    result_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {
        control_id: {
            "fitted": False,
            "fit_intercept": False,
            "coefficients": control_coefficients,
        }
    }
    holdout_probabilities: dict[str, np.ndarray] = {
        control_id: np.asarray(
            [float(row["player_glicko_probability"]) for row in holdout],
            dtype=float,
        )
    }
    for split_name, split_rows in (
        ("selection_2021_2023", selection),
        ("diagnostic_2024_plus", holdout),
    ):
        _append_model_metric_rows(
            result_rows,
            model=control_id,
            split=split_name,
            fitted_on="external_control_not_fitted",
            ridge_c=None,
            symmetry_contract="side swap negates baseline logit",
            feature_names=["baseline_logit"],
            coefficients=control_coefficients,
            evaluated=split_rows,
            probability_key="player_glicko_probability",
            epsilon=epsilon,
        )

    for model_id in (
        "baseline_logit_calibration",
        "tier_offset",
        "tier_offset_context_slope",
    ):
        x_train, feature_names = model_matrix(selection, model_id, epsilon)
        x_holdout, holdout_names = model_matrix(holdout, model_id, epsilon)
        if feature_names != holdout_names:
            raise RuntimeError("diagnostic feature alignment changed between splits")
        estimator = LogisticRegression(
            C=ridge_c,
            solver="lbfgs",
            fit_intercept=False,
            max_iter=5000,
            tol=1e-10,
            random_state=RANDOM_SEED,
        )
        estimator.fit(x_train, labels)
        coefficients = {
            name: float(value)
            for name, value in zip(feature_names, estimator.coef_[0], strict=True)
        }
        fitted_models[model_id] = {
            "fitted": True,
            "fit_intercept": False,
            "ridge_c": ridge_c,
            "iterations": int(estimator.n_iter_[0]),
            "coefficients": coefficients,
        }
        for split_name, split_rows, matrix in (
            ("selection_2021_2023", selection, x_train),
            ("diagnostic_2024_plus", holdout, x_holdout),
        ):
            probabilities = estimator.predict_proba(matrix)[:, 1]
            if split_name == "diagnostic_2024_plus":
                holdout_probabilities[model_id] = probabilities
            evaluated = _with_probabilities(split_rows, probabilities)
            _append_model_metric_rows(
                result_rows,
                model=model_id,
                split=split_name,
                fitted_on="selection_2021_2023",
                ridge_c=ridge_c,
                symmetry_contract=(
                    "all inputs negate under side swap; invariant contexts only multiply baseline logit"
                ),
                feature_names=feature_names,
                coefficients=coefficients,
                evaluated=evaluated,
                probability_key="diagnostic_probability",
                epsilon=epsilon,
            )
    return result_rows, fitted_models, holdout_probabilities




def _bernoulli_log_losses(
    labels: np.ndarray, probabilities: np.ndarray, epsilon: float
) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))


def _bootstrap_block_key(row: dict[str, object]) -> str:
    row_date = str(row["date"])
    if row["competition_scope"] == CompetitionScope.CROSS_LEAGUE.value:
        return f"event::{row['competition_family']}::{row_date[:4]}"
    return f"month::{row_date[:7]}"


def paired_holdout_comparisons(
    holdout: Sequence[dict[str, object]],
    probabilities_by_model: dict[str, np.ndarray],
    epsilon: float,
    *,
    repetitions: int = 5000,
) -> list[dict[str, object]]:
    control_id = "player_glicko_external_weight_one"
    control_probabilities = probabilities_by_model[control_id]
    labels = np.asarray([int(row["target"]) for row in holdout], dtype=int)
    if len(control_probabilities) != len(holdout):
        raise RuntimeError("control probability alignment changed")
    control_losses = _bernoulli_log_losses(labels, control_probabilities, epsilon)
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for model_id in sorted(probabilities_by_model):
        if model_id == control_id:
            continue
        candidate_probabilities = probabilities_by_model[model_id]
        if len(candidate_probabilities) != len(holdout):
            raise RuntimeError(f"{model_id} probability alignment changed")
        candidate_losses = _bernoulli_log_losses(
            labels, candidate_probabilities, epsilon
        )
        paired_deltas = candidate_losses - control_losses
        for cohort in COHORTS:
            indices = [
                index
                for index, row in enumerate(holdout)
                if bool(row[f"cohort_{cohort}"])
            ]
            if not indices:
                continue
            block_members: defaultdict[str, list[int]] = defaultdict(list)
            for index in indices:
                block_members[_bootstrap_block_key(holdout[index])].append(index)
            blocks = sorted(block_members)
            block_delta_sums = np.asarray(
                [
                    float(np.sum(paired_deltas[block_members[block]]))
                    for block in blocks
                ],
                dtype=float,
            )
            block_sizes = np.asarray(
                [len(block_members[block]) for block in blocks],
                dtype=float,
            )
            if len(blocks) == 1:
                ci_low = ci_high = float(np.mean(paired_deltas[indices]))
            else:
                draws = rng.integers(
                    0, len(blocks), size=(repetitions, len(blocks))
                )
                sampled_deltas = np.sum(block_delta_sums[draws], axis=1) / np.sum(
                    block_sizes[draws], axis=1
                )
                ci_low, ci_high = (
                    float(value)
                    for value in np.quantile(sampled_deltas, (0.025, 0.975))
                )
            rows.append(
                {
                    "model": model_id,
                    "control": control_id,
                    "split": "diagnostic_2024_plus",
                    "cohort": cohort,
                    "n": len(indices),
                    "bootstrap_unit": (
                        "competition family-year for cross-league events; calendar month otherwise"
                    ),
                    "bootstrap_blocks": len(blocks),
                    "bootstrap_repetitions": repetitions,
                    "random_seed": RANDOM_SEED,
                    "control_log_loss": float(np.mean(control_losses[indices])),
                    "candidate_log_loss": float(np.mean(candidate_losses[indices])),
                    "log_loss_delta_vs_control": float(
                        np.mean(paired_deltas[indices])
                    ),
                    "paired_block_bootstrap_ci_low": ci_low,
                    "paired_block_bootstrap_ci_high": ci_high,
                }
            )
    return rows


def correction_assessment(
    comparison_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    overall = [
        row for row in comparison_rows if row["cohort"] == "overall"
    ]
    best = min(
        overall,
        key=lambda row: (
            float(row["candidate_log_loss"]),
            str(row["model"]),
        ),
    )
    delta = float(best["log_loss_delta_vs_control"])
    ci_high = float(best["paired_block_bootstrap_ci_high"])
    uncertainty_supported = ci_high < 0.0
    return {
        "external_control": best["control"],
        "best_diagnostic_model": best["model"],
        "holdout_log_loss_delta_vs_control": delta,
        "paired_block_bootstrap_ci": [
            float(best["paired_block_bootstrap_ci_low"]),
            ci_high,
        ],
        "bootstrap_blocks": int(best["bootstrap_blocks"]),
        "descriptively_justified": delta < 0.0,
        "uncertainty_supported": uncertainty_supported,
        "recommendation": (
            f"Prototype {best['model']} as a new rating candidate; do not replace the "
            "baseline until prospective validation."
            if uncertainty_supported
            else "Retain the external weight-one Player Glicko baseline; the fitted "
            "correction lacks block-bootstrap support."
        ),
        "decision_rule": (
            "lower 2024+ LogLoss with the paired block-bootstrap 95% interval "
            "entirely below zero"
        ),
        "inference_limit": (
            "The 2024+ cohort is temporally untouched by this fit but is not a virgin "
            "repository holdout; uncertainty is descriptive, not a promotion test."
        ),
    }


def aggregate_coverage(
    records: Sequence[MatchRecord],
    aligned_ids: set[str] | None = None,
    odds_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    groups: defaultdict[tuple[str, str, str, str, str], list[MatchRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.tournament,
            record.competition.family,
            record.competition.tier.value,
            record.competition.scope.value,
            record.competition.matched_rule,
        )
        groups[key].append(record)
    rows: list[dict[str, object]] = []
    for key, matches_for_competition in sorted(groups.items()):
        tournament, family, tier, scope, matched_rule = key
        teams = {
            team
            for record in matches_for_competition
            for team in (record.team_1, record.team_2)
            if team
        }
        players = {
            player
            for record in matches_for_competition
            for player in (*record.players_1, *record.players_2)
        }
        rows.append(
            {
                "tournament": tournament,
                "family": family,
                "tier": tier,
                "scope": scope,
                "matched_rule": matched_rule,
                "matches": len(matches_for_competition),
                "date_min": min(record.match_date for record in matches_for_competition).isoformat(),
                "date_max": max(record.match_date for record in matches_for_competition).isoformat(),
                "unique_teams": len(teams),
                "unique_players": len(players),
                "aligned_player_glicko_predictions": sum(
                    record.match_id in aligned_ids for record in matches_for_competition
                )
                if aligned_ids is not None
                else 0,
                "odds_mapped_matches": sum(
                    record.match_id in odds_ids for record in matches_for_competition
                )
                if odds_ids is not None
                else 0,
            }
        )
    return rows


def aggregate_edges(edge_events: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: defaultdict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for event in edge_events:
        key = (
            str(event["edge_type"]),
            str(event["source_family"]),
            str(event["source_tier"]),
            str(event["target_family"]),
            str(event["target_tier"]),
        )
        groups[key].append(event)
    rows: list[dict[str, object]] = []
    for key, events in sorted(groups.items()):
        edge_type, source_family, source_tier, target_family, target_tier = key
        rows.append(
            {
                "edge_type": edge_type,
                "source_family": source_family,
                "source_tier": source_tier,
                "target_family": target_family,
                "target_tier": target_tier,
                "events": len(events),
                "date_min": min(str(event["date"]) for event in events),
                "date_max": max(str(event["date"]) for event in events),
                "unique_matches": len({event["match_id"] for event in events if event["match_id"]}),
                "unique_players": len({event["player_id"] for event in events if event["player_id"]}),
                "unique_team_pairs": len(
                    {
                        (event["source_team"], event["target_team"])
                        for event in events
                    }
                ),
                "match_ids": "|".join(
                    sorted(
                        {str(event["match_id"]) for event in events if event["match_id"]},
                        key=int,
                    )
                ),
                "player_ids": "|".join(
                    sorted({str(event["player_id"]) for event in events if event["player_id"]})
                ),
            }
        )
    return rows


def _component_membership(
    adjacency: dict[str, set[str]], prefix: str
) -> tuple[dict[str, str], dict[str, int]]:
    component_by_family: dict[str, str] = {}
    component_size_by_family: dict[str, int] = {}
    component_number = 0
    for start in sorted(adjacency):
        if start in component_by_family:
            continue
        component_number += 1
        pending = [start]
        members: set[str] = set()
        while pending:
            family = pending.pop()
            if family in members:
                continue
            members.add(family)
            pending.extend(sorted(adjacency[family] - members, reverse=True))
        component_id = f"{prefix}{component_number:03d}"
        for family in members:
            component_by_family[family] = component_id
            component_size_by_family[family] = len(members)
    return component_by_family, component_size_by_family


def build_flow_nodes(
    records: Sequence[MatchRecord], edge_events: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    observations: defaultdict[str, list[tuple[CompetitionTier, MatchRecord]]] = defaultdict(list)
    for record in records:
        if (
            record.competition.scope is CompetitionScope.DOMESTIC
            and record.competition.family != UNKNOWN_FAMILY
        ):
            observations[record.competition.family].append((record.competition.tier, record))

    direct_events = [
        event for event in edge_events if event["edge_type"] == "direct_bridge"
    ]
    direct_adjacency = {family: set() for family in observations}
    all_flow_adjacency = {family: set() for family in observations}
    for event in edge_events:
        source = str(event["source_family"])
        target = str(event["target_family"])
        if source not in all_flow_adjacency or target not in all_flow_adjacency:
            continue
        all_flow_adjacency[source].add(target)
        all_flow_adjacency[target].add(source)
        if event["edge_type"] == "direct_bridge":
            direct_adjacency[source].add(target)
            direct_adjacency[target].add(source)

    direct_component, direct_component_size = _component_membership(
        direct_adjacency, "D"
    )
    all_flow_component, all_flow_component_size = _component_membership(
        all_flow_adjacency, "A"
    )
    as_of = max(record.match_date for record in records)
    rows: list[dict[str, object]] = []
    for family, values in sorted(observations.items()):
        tier_counts = Counter(tier.value for tier, _ in values)
        primary_tier = min(tier_counts, key=lambda value: (-tier_counts[value], value))
        family_records = [record for _, record in values]
        direct = [
            event
            for event in direct_events
            if family in (event["source_family"], event["target_family"])
        ]
        related_flow = [
            event
            for event in edge_events
            if family in (event["source_family"], event["target_family"])
        ]
        last_direct_bridge = (
            max(date.fromisoformat(str(event["date"])) for event in direct)
            if direct
            else None
        )
        last_flow = (
            max(date.fromisoformat(str(event["date"])) for event in related_flow)
            if related_flow
            else None
        )
        transfers_in = [
            event
            for event in edge_events
            if event["edge_type"] == "player_transfer" and event["target_family"] == family
        ]
        transfers_out = [
            event
            for event in edge_events
            if event["edge_type"] == "player_transfer" and event["source_family"] == family
        ]
        rows.append(
            {
                "family": family,
                "primary_tier": primary_tier,
                "tier_history": "|".join(sorted(tier_counts)),
                "direct_bridge_component_id": direct_component[family],
                "direct_bridge_component_size": direct_component_size[family],
                "direct_bridge_partners": len(direct_adjacency[family]),
                "all_flow_component_id": all_flow_component[family],
                "all_flow_component_size": all_flow_component_size[family],
                "all_flow_partners": len(all_flow_adjacency[family]),
                "domestic_matches": len(family_records),
                "date_min": min(record.match_date for record in family_records).isoformat(),
                "date_max": max(record.match_date for record in family_records).isoformat(),
                "unique_teams": len(
                    {
                        team
                        for record in family_records
                        for team in (record.team_1, record.team_2)
                        if team
                    }
                ),
                "unique_players": len(
                    {
                        player
                        for record in family_records
                        for player in (*record.players_1, *record.players_2)
                    }
                ),
                "direct_bridge_events": len(direct),
                "last_direct_bridge_date": (
                    last_direct_bridge.isoformat() if last_direct_bridge else None
                ),
                "days_since_last_direct_bridge": (
                    (as_of - last_direct_bridge).days if last_direct_bridge else None
                ),
                "last_flow_date": last_flow.isoformat() if last_flow else None,
                "days_since_last_flow": (
                    (as_of - last_flow).days if last_flow else None
                ),
                "transfer_arrivals": len(transfers_in),
                "transfer_departures": len(transfers_out),
            }
        )
    return rows


def replay_player_glicko(
    records: Sequence[MatchRecord], as_of: date
) -> tuple[GlickoRating, dict[str, object]]:
    glicko = GlickoRating(tau=0.5)
    player_names: dict[str, str] = {}
    player_last_appearance: dict[str, date] = {}
    player_game_counts: Counter[str] = Counter()
    counts = Counter()

    for match_date, date_group in groupby(records, key=lambda item: item.match_date):
        pending: list[MatchRecord] = []
        for record in date_group:
            player_names.update(record.player_names)
            for player_id in (*record.players_1, *record.players_2):
                player_last_appearance[player_id] = match_date
            if (
                len(record.players_1) != 5
                or len(record.players_2) != 5
                or len(set(record.players_1)) != 5
                or len(set(record.players_2)) != 5
                or not record.game_results
            ):
                counts["excluded_incomplete_or_ambiguous_roster"] += 1
                continue
            glicko.update_rd_before_match(
                record.team_1,
                record.team_2,
                list(record.players_1),
                list(record.players_2),
                match_date,
            )
            pending.append(record)
        for record in pending:
            counts["eligible_matches"] += 1
            counts["eligible_games"] += len(record.game_results)
            for player_id in (*record.players_1, *record.players_2):
                player_game_counts[player_id] += len(record.game_results)
            for score_1 in record.game_results:
                glicko.update_player(
                    list(record.players_1), list(record.players_2), score_1, 1 - score_1
                )
                glicko.update_team(record.team_1, record.team_2, score_1, 1 - score_1)

    for player_id, rating in glicko.player_ratings.items():
        glicko.apply_time_decay(rating, player_last_appearance.get(player_id), as_of)
    counts["players_observed"] = len(player_last_appearance)
    replay_counts = _complete_counts(
        counts,
        (
            "excluded_incomplete_or_ambiguous_roster",
            "eligible_matches",
            "eligible_games",
            "players_observed",
        ),
    )
    return glicko, {
        "counts": replay_counts,
        "player_names": player_names,
        "player_last_appearance": player_last_appearance,
        "player_game_counts": player_game_counts,
    }


def active_player_rows(
    glicko: GlickoRating,
    replay: dict[str, object],
    player_domestic: dict[str, DomesticAffiliation],
    as_of: date,
    active_days: int,
    low_evidence_games: int,
) -> list[dict[str, object]]:
    player_names = replay["player_names"]
    player_last_appearance = replay["player_last_appearance"]
    player_game_counts = replay["player_game_counts"]
    if not isinstance(player_names, dict) or not isinstance(player_last_appearance, dict):
        raise TypeError("invalid replay metadata")
    cutoff = as_of - timedelta(days=active_days)
    rows: list[dict[str, object]] = []
    for player_id, last_appearance in sorted(player_last_appearance.items()):
        if not isinstance(last_appearance, date) or last_appearance < cutoff:
            continue
        games_played = int(player_game_counts[player_id])
        if games_played == 0:
            continue
        rating = glicko.get_player_rating(player_id)
        affiliation = player_domestic.get(player_id)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_names.get(player_id, player_id),
                "glicko_rating": float(rating.rating),
                "glicko_rd": float(rating.rd),
                "glicko_lower_bound": float(rating.rating - 2.0 * rating.rd),
                "games_played": games_played,
                "low_evidence": games_played < low_evidence_games,
                "last_appearance": last_appearance.isoformat(),
                "days_since_last_appearance": (as_of - last_appearance).days,
                "domestic_family": affiliation.family if affiliation else UNKNOWN_FAMILY,
                "domestic_tier": affiliation.tier.value if affiliation else CompetitionTier.UNKNOWN.value,
                "affiliation_source_date": affiliation.source_date.isoformat() if affiliation else None,
                "affiliation_source_tournament": affiliation.source_tournament if affiliation else None,
                "affiliation_source_team": affiliation.source_team if affiliation else None,
            }
        )
    rows.sort(key=lambda row: (-float(row["glicko_rating"]), str(row["player_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rating_rank"] = rank
    return rows


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def player_distribution_rows(
    players: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: defaultdict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for player in players:
        tier = str(player["domestic_tier"])
        family = str(player["domestic_family"])
        groups[("overall", "all", "all")].append(player)
        groups[("tier", tier, "all")].append(player)
        groups[("family", "all", family)].append(player)
        groups[("tier_family", tier, family)].append(player)
    rows: list[dict[str, object]] = []
    for (group_type, tier, family), members in sorted(groups.items()):
        row: dict[str, object] = {
            "group_type": group_type,
            "tier": tier,
            "family": family,
            "active_players": len(members),
            "low_evidence_players": sum(bool(member["low_evidence"]) for member in members),
            "unknown_affiliation_players": sum(
                member["domestic_family"] == UNKNOWN_FAMILY for member in members
            ),
            "date_min_last_appearance": min(str(member["last_appearance"]) for member in members),
            "date_max_last_appearance": max(str(member["last_appearance"]) for member in members),
        }
        for field in ("glicko_rating", "glicko_rd", "glicko_lower_bound"):
            values = [float(member[field]) for member in members]
            for label, q in (
                ("min", 0.0),
                ("p10", 0.10),
                ("p25", 0.25),
                ("median", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("max", 1.0),
            ):
                row[f"{field}_{label}"] = _quantile(values, q)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def main() -> None:
    args = parse_args()
    records, match_counts = load_matches(args.matches)
    if not records:
        raise ValueError("no eligible non-draw matches")
    ratings, rating_counts = load_rating_predictions(args.rating_predictions)
    odds_path = None if args.no_odds else args.odds
    odds_ids, odds_counts = load_odds_match_ids(odds_path)
    if args.diagnostic_require_odds and odds_ids is None:
        raise FileNotFoundError("odds mapping is required for diagnostics but was not loaded")

    audit_rows, edge_events, team_domestic, player_domestic, temporal_counts = reconstruct_temporal_flow(records)
    joined, alignment_counts = align_predictions(records, audit_rows, ratings, odds_ids)
    coverage = aggregate_coverage(
        records,
        aligned_ids={str(row["match_id"]) for row in joined},
        odds_ids=odds_ids,
    )
    unknown_competitions = [
        row
        for row in coverage
        if row["tier"] == CompetitionTier.UNKNOWN.value
        or row["scope"] == CompetitionScope.UNKNOWN.value
        or row["family"] == UNKNOWN_FAMILY
    ]
    edges = aggregate_edges(edge_events)
    nodes = build_flow_nodes(records, edge_events)
    cohorts = cohort_metrics(joined, args.probability_epsilon)
    major_calibration = major_side_calibration(
        joined, "player_glicko_probability", args.probability_epsilon
    )
    bridge_rows = [
        row for row in joined if bool(row["cohort_known_cross_league"])
    ]

    selection, holdout, diagnostic_counts = diagnostic_sample(
        joined,
        args.selection_start,
        args.holdout_start,
        args.diagnostic_require_odds,
    )
    model_results, fitted_models, holdout_probabilities = fit_diagnostic_models(
        selection, holdout, args.ridge_c, args.probability_epsilon
    )
    model_comparisons = paired_holdout_comparisons(
        holdout,
        holdout_probabilities,
        args.probability_epsilon,
    )
    correction = correction_assessment(model_comparisons)

    as_of = max(record.match_date for record in records)
    glicko, replay = replay_player_glicko(records, as_of)
    active_players = active_player_rows(
        glicko,
        replay,
        player_domestic,
        as_of,
        args.active_days,
        args.low_evidence_games,
    )
    top_players = active_players[: args.top_players]
    distributions = player_distribution_rows(active_players)
    active_cutoff = as_of - timedelta(days=args.active_days)
    active_players_excluded_zero_rated_games = sum(
        last_appearance >= active_cutoff
        and int(replay["player_game_counts"][player_id]) == 0
        for player_id, last_appearance in replay["player_last_appearance"].items()
    )
    direct_bridge_component_ids = {
        str(row["direct_bridge_component_id"]) for row in nodes
    }
    all_flow_component_ids = {
        str(row["all_flow_component_id"]) for row in nodes
    }
    isolated_flow_nodes = sum(
        int(row["direct_bridge_partners"]) == 0 for row in nodes
    )
    isolated_all_flow_nodes = sum(
        int(row["all_flow_partners"]) == 0 for row in nodes
    )
    stale_flow_nodes = sum(
        row["days_since_last_direct_bridge"] is None
        or int(row["days_since_last_direct_bridge"]) > 365
        for row in nodes
    )
    stale_all_flow_nodes = sum(
        row["days_since_last_flow"] is None
        or int(row["days_since_last_flow"]) > 365
        for row in nodes
    )

    coverage_fields = (
        "tournament", "family", "tier", "scope", "matched_rule", "matches",
        "date_min", "date_max", "unique_teams", "unique_players",
        "aligned_player_glicko_predictions", "odds_mapped_matches",
    )
    node_fields = (
        "family", "primary_tier", "tier_history", "direct_bridge_component_id",
        "direct_bridge_component_size", "direct_bridge_partners",
        "all_flow_component_id", "all_flow_component_size", "all_flow_partners",
        "domestic_matches", "date_min", "date_max", "unique_teams",
        "unique_players", "direct_bridge_events", "last_direct_bridge_date",
        "days_since_last_direct_bridge", "last_flow_date", "days_since_last_flow",
        "transfer_arrivals", "transfer_departures",
    )
    edge_fields = (
        "edge_type", "source_family", "source_tier", "target_family", "target_tier",
        "events", "date_min", "date_max", "unique_matches", "unique_players",
        "unique_team_pairs", "match_ids", "player_ids",
    )
    prediction_fields = (
        "match_id", "date", "tournament", "competition_family", "competition_tier",
        "competition_scope", "matched_rule", "team_1_id", "team_1_name", "team_1_family",
        "team_1_tier", "team_1_affiliation_date", "team_1_affiliation_tournament",
        "team_1_affiliation_team", "team_2_id", "team_2_name", "team_2_family",
        "team_2_tier", "team_2_affiliation_date", "team_2_affiliation_tournament",
        "team_2_affiliation_team", "target", "player_glicko_probability", "odds_mapped",
        "cohort_overall", "cohort_major_major", "cohort_regional_regional",
        "cohort_development", "cohort_known_cross_league", "cohort_major_vs_lower",
        "diagnostic_context",
    )
    model_fields = (
        "model", "split", "cohort", "fitted_on", "ridge_c", "fit_intercept",
        "symmetry_contract", "feature_names_json", "coefficients_json", "n",
        "date_min", "date_max", "positives", "negatives", "event_rate",
        "mean_probability", "calibration_gap", "absolute_calibration_gap",
        "log_loss", "brier", "auc",
    )
    comparison_fields = (
        "model", "control", "split", "cohort", "n", "bootstrap_unit",
        "bootstrap_blocks", "bootstrap_repetitions", "random_seed",
        "control_log_loss", "candidate_log_loss", "log_loss_delta_vs_control",
        "paired_block_bootstrap_ci_low", "paired_block_bootstrap_ci_high",
    )
    player_fields = (
        "rating_rank", "player_id", "player_name", "glicko_rating", "glicko_rd",
        "glicko_lower_bound", "games_played", "low_evidence", "last_appearance",
        "days_since_last_appearance", "domestic_family", "domestic_tier",
        "affiliation_source_date", "affiliation_source_tournament", "affiliation_source_team",
    )
    distribution_fields = (
        "group_type", "tier", "family", "active_players", "low_evidence_players",
        "unknown_affiliation_players", "date_min_last_appearance", "date_max_last_appearance",
        *tuple(
            f"{field}_{quantile}"
            for field in ("glicko_rating", "glicko_rd", "glicko_lower_bound")
            for quantile in ("min", "p10", "p25", "median", "p75", "p90", "max")
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "competition_coverage.csv", coverage, coverage_fields)
    write_csv(args.output_dir / "unknown_competitions.csv", unknown_competitions, coverage_fields)
    write_csv(args.output_dir / "flow_nodes.csv", nodes, node_fields)
    write_csv(args.output_dir / "flow_edges.csv", edges, edge_fields)
    write_csv(args.output_dir / "bridge_predictions.csv", bridge_rows, prediction_fields)
    write_csv(args.output_dir / "diagnostic_model_results.csv", model_results, model_fields)
    write_csv(
        args.output_dir / "diagnostic_model_comparisons.csv",
        model_comparisons,
        comparison_fields,
    )
    write_csv(
        args.output_dir / "player_rating_distribution.csv",
        distributions,
        distribution_fields,
    )
    write_csv(args.output_dir / "current_top_players.csv", top_players, player_fields)

    top_100 = active_players[:100]
    summary = {
        "experiment": "EXP-075",
        "status": "completed",
        "question": (
            "How does rating evidence move between competition ecosystems, where is Player Glicko "
            "miscalibrated by ecosystem context, and is a symmetric correction justified?"
        ),
        "inputs": {
            "matches": str(args.matches),
            "rating_predictions": str(args.rating_predictions),
            "odds_mapping": str(odds_path) if odds_path is not None else None,
            "odds_columns_consumed": ["golgg_match_id"] if odds_ids is not None else [],
            "closing_odds_used_as_model_input": False,
        },
        "dates": {
            "match_min": min(record.match_date for record in records).isoformat(),
            "match_max": as_of.isoformat(),
            "selection_start_inclusive": args.selection_start.isoformat(),
            "selection_end_inclusive": (args.holdout_start - timedelta(days=1)).isoformat(),
            "diagnostic_start_inclusive": args.holdout_start.isoformat(),
            "active_player_cutoff_inclusive": (as_of - timedelta(days=args.active_days)).isoformat(),
        },
        "counts": {
            "matches": match_counts,
            "rating_predictions": rating_counts,
            "odds_mapping": odds_counts,
            "alignment": alignment_counts,
            "temporal_flow": temporal_counts,
            "diagnostic_eligibility": diagnostic_counts,
            "competition_names": len(coverage),
            "unknown_competition_names": len(unknown_competitions),
            "unknown_competition_matches": sum(
                int(row["matches"]) for row in unknown_competitions
            ),
            "flow_nodes": len(nodes),
            "flow_edges": len(edges),
            "bridge_predictions": len(bridge_rows),
            "direct_bridge_components": len(direct_bridge_component_ids),
            "all_flow_components": len(all_flow_component_ids),
            "isolated_flow_nodes": isolated_flow_nodes,
            "isolated_all_flow_nodes": isolated_all_flow_nodes,
            "flow_nodes_without_bridge_in_last_365_days": stale_flow_nodes,
            "flow_nodes_without_any_flow_in_last_365_days": stale_all_flow_nodes,
            "active_players": len(active_players),
            "current_top_players_written": len(top_players),
            "active_low_evidence_players": sum(bool(row["low_evidence"]) for row in active_players),
            "active_unknown_affiliation_players": sum(
                row["domestic_family"] == UNKNOWN_FAMILY for row in active_players
            ),
            "glicko_replay": replay["counts"],
            "active_players_excluded_zero_rated_games": (
                active_players_excluded_zero_rated_games
            ),
        },
        "same_day_contract": (
            "Every match on a date observes team/player domestic affiliations from strictly earlier "
            "dates; all domestic affiliation updates are applied only after all observations on that date."
        ),
        "affiliation_contract": (
            "Only known domestic competitions update last domestic family; international, cross-league, "
            "and unknown competitions never overwrite it. Conflicting same-day domestic destinations do not update state."
        ),
        "cohort_metrics": cohorts,
        "major_vs_lower_calibration": major_calibration,
        "flow_connectivity": {
            "direct_bridge_definition": (
                "undirected connectivity from cross-league matches only"
            ),
            "direct_bridge_components": len(direct_bridge_component_ids),
            "direct_bridge_isolated_nodes": isolated_flow_nodes,
            "nodes_without_direct_bridge_in_last_365_days": stale_flow_nodes,
            "all_flow_definition": (
                "undirected connectivity from direct cross-league matches plus "
                "inferred player transfers"
            ),
            "all_flow_components": len(all_flow_component_ids),
            "all_flow_isolated_nodes": isolated_all_flow_nodes,
            "nodes_without_any_flow_in_last_365_days": stale_all_flow_nodes,
        },
        "diagnostic_design": {
            "selection_only_fit": True,
            "selection_window": [
                args.selection_start.isoformat(),
                (args.holdout_start - timedelta(days=1)).isoformat(),
            ],
            "untouched_diagnostic_window": [args.holdout_start.isoformat(), as_of.isoformat()],
            "odds_mapping_required": args.diagnostic_require_odds,
            "ridge_c": args.ridge_c,
            "probability_epsilon": args.probability_epsilon,
            "fit_intercept": False,
            "side_symmetry": (
                "Tier indicators are side-A minus side-B; context indicators are invariant and only "
                "multiply the signed baseline logit. Therefore every model input vector negates under side swap."
            ),
            "external_control": (
                "Stored Player Glicko probability with weight-one baseline logit; not fitted or replayed for prediction."
            ),
            "models": fitted_models,
            "correction_assessment": correction,
            "holdout_comparisons": model_comparisons,
        },
        "active_player_audit": {
            "active_days": args.active_days,
            "low_evidence_games_threshold": args.low_evidence_games,
            "top_players_requested": args.top_players,
            "ranking": "descending final Player Glicko rating; player ID breaks ties",
            "rating_state": (
                "Fresh in-memory replay of the established GlickoRating(tau=0.5); five-distinct-player "
                "matches only, date-batched RD updates, all active-player RDs decayed to match_max."
            ),
            "top_100_tier_composition": _counter_dict(
                str(row["domestic_tier"]) for row in top_100
            ),
            "top_100_family_composition": _counter_dict(
                str(row["domestic_family"]) for row in top_100
            ),
            "top_100_low_evidence_players": sum(bool(row["low_evidence"]) for row in top_100),
            "top_100_unknown_affiliation_players": sum(
                row["domestic_family"] == UNKNOWN_FAMILY for row in top_100
            ),
        },
        "artefacts": [
            "summary.json",
            "competition_coverage.csv",
            "unknown_competitions.csv",
            "flow_nodes.csv",
            "flow_edges.csv",
            "bridge_predictions.csv",
            "diagnostic_model_results.csv",
            "diagnostic_model_comparisons.csv",
            "player_rating_distribution.csv",
            "current_top_players.csv",
        ],
        "limitations": [
            "Tournament classification is rule-based; unmatched names remain explicit unknowns and are never treated as regional evidence.",
            "Calendar dates, not exact start timestamps, are available; same-day observations are therefore batched and no within-day ordering is inferred for affiliation state.",
            "Historical match rosters have no independent available_at timestamp and are treated as observations attached to the match date.",
            "A direct bridge requires both teams to have a known domestic affiliation from an earlier date; early-history and newly formed teams are excluded.",
            "Player transfers are inferred from later domestic match appearances, not contracts; same-day conflicting destinations are left unresolved.",
            "Cohorts use strictly prior domestic affiliations, so first observed domestic appearances are unknown rather than backfilled from the current event.",
            "The fixed 2021-2023 ridge diagnostics are descriptive context corrections, not a replacement rating system; they do not update or mutate Glicko state.",
            "The 2024+ period is temporally untouched by this fit but may have been inspected by earlier repository experiments, so it is not a virgin final holdout.",
            "AUC is null for empty or single-class cohorts; calibration gap is mean predicted probability minus observed event rate.",
            "The optional odds file contributes match-ID membership only. No opening or closing price is a feature, target, weight, or calibration input.",
            "Current player ratings inherit the established baseline's cross-ecosystem update mechanics; affiliation summaries diagnose that exposure rather than correcting it.",
            "Installed glicko2==2.1.0 is not treated as a verified reference implementation; EXP-075 diagnoses its stored output rather than endorsing its equations.",
            "The current wrapper truncates aggregate team rating and RD to integers and updates series game-by-game rather than as one simultaneous Glicko rating period.",
            "Active means last observed match appearance within the configured window, not a verified current contract or announced roster.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
