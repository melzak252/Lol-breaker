#!/usr/bin/env python3
"""Evaluate the fixed family-calibrated Player Glicko-2 successor.

The runner is deliberately read-only with respect to databases and rating/model
artifacts. It replays JSON matches in complete calendar-date periods and joins
the resulting pre-match probabilities to the frozen ``player_gl`` baseline by
match ID, date, and label. The 2024+ comparison is diagnostic because those
outcomes have already been inspected; this runner never makes a promotion
claim.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import groupby
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.competition_tiers import (  # noqa: E402
    CompetitionIdentity,
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)
from src.ratings.family_calibrated_glicko2 import (  # noqa: E402
    FamilyCalibratedGlicko2,
    RatingEvent,
)
from src.utils.golgg_schema import (  # noqa: E402
    game_score_for_match_team1,
    normalized_match,
)

MATCHES_PATH = PROJECT_ROOT / "data" / "golgg_matches.json"
BASELINE_PATH = PROJECT_ROOT / "data" / "golgg_y_predicts.csv"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "experiments"
    / "exp075_calibrated_glicko2_evaluation"
)
CANDIDATE_SYSTEM = "gl2f"
CANDIDATE_VERSION = "player-glicko2-family-v1"
LEGACY_SYSTEM = "gl"
DIAGNOSTIC_START = date(2024, 1, 1)
BOOTSTRAP_REPETITIONS = 5_000
RANDOM_SEED = 75
ECE_BINS = 10
PROBABILITY_EPSILON = 1e-15
UNKNOWN = "unknown"

COHORTS = (
    "overall",
    "major_major",
    "regional_regional",
    "development_involved",
    "known_cross_league",
    "major_vs_minor",
    "major_vs_regional",
)

COHORT_DEFINITIONS = {
    "overall": "all aligned 2024+ rows",
    "major_major": "both strictly-prior domestic team tiers are major",
    "regional_regional": "both strictly-prior domestic team tiers are regional",
    "development_involved": (
        "the event is development or either strictly-prior domestic team tier is development"
    ),
    "known_cross_league": (
        "cross-league event with known strictly-prior domestic affiliations on both sides"
    ),
    "major_vs_minor": (
        "strictly-prior domestic team tiers are major versus minor_top_level"
    ),
    "major_vs_regional": (
        "strictly-prior domestic team tiers are major versus regional"
    ),
}

PREDICTION_FIELDS = (
    "golgg_match_id",
    "date",
    "candidate_system",
    "candidate_version",
    "legacy_system",
    "y_true",
    "candidate_probability",
    "legacy_probability",
    "candidate_log_loss",
    "legacy_log_loss",
    "paired_log_loss_delta",
    "tournament",
    "competition_family",
    "competition_tier",
    "competition_scope",
    "matched_rule",
    "team_1_id",
    "team_2_id",
    "team_1_family",
    "team_1_tier",
    "team_1_affiliation_date",
    "team_2_family",
    "team_2_tier",
    "team_2_affiliation_date",
    "bootstrap_block",
    *tuple(f"cohort_{cohort}" for cohort in COHORTS),
)


@dataclass(frozen=True, slots=True)
class MatchRecord:
    match_id: str
    match_date: date
    tournament: str
    competition: CompetitionIdentity
    team_1_id: str
    team_2_id: str
    players_1: tuple[str, ...]
    players_2: tuple[str, ...]
    scores: tuple[int, ...]
    y_true: int


@dataclass(frozen=True, slots=True)
class BaselinePrediction:
    match_id: str
    match_date: date
    probability: float
    y_true: int


@dataclass(frozen=True, slots=True)
class DomesticAffiliation:
    family: str
    tier: str
    source_date: date
    source_match_id: str


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    match_id: str
    probability: float
    affiliation_1: DomesticAffiliation | None
    affiliation_2: DomesticAffiliation | None


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    match: MatchRecord
    candidate: CandidatePrediction
    legacy_probability: float
    cohorts: frozenset[str]

    @property
    def candidate_log_loss(self) -> float:
        return bernoulli_log_loss(self.match.y_true, self.candidate.probability)

    @property
    def legacy_log_loss(self) -> float:
        return bernoulli_log_loss(self.match.y_true, self.legacy_probability)

    @property
    def paired_log_loss_delta(self) -> float:
        return self.candidate_log_loss - self.legacy_log_loss

    def to_csv_row(self) -> dict[str, object]:
        affiliation_1 = self.candidate.affiliation_1
        affiliation_2 = self.candidate.affiliation_2
        row: dict[str, object] = {
            "golgg_match_id": self.match.match_id,
            "date": self.match.match_date.isoformat(),
            "candidate_system": CANDIDATE_SYSTEM,
            "candidate_version": CANDIDATE_VERSION,
            "legacy_system": LEGACY_SYSTEM,
            "y_true": self.match.y_true,
            "candidate_probability": self.candidate.probability,
            "legacy_probability": self.legacy_probability,
            "candidate_log_loss": self.candidate_log_loss,
            "legacy_log_loss": self.legacy_log_loss,
            "paired_log_loss_delta": self.paired_log_loss_delta,
            "tournament": self.match.tournament,
            "competition_family": self.match.competition.family,
            "competition_tier": self.match.competition.tier.value,
            "competition_scope": self.match.competition.scope.value,
            "matched_rule": self.match.competition.matched_rule,
            "team_1_id": self.match.team_1_id,
            "team_2_id": self.match.team_2_id,
            "team_1_family": affiliation_1.family if affiliation_1 else UNKNOWN,
            "team_1_tier": affiliation_1.tier if affiliation_1 else UNKNOWN,
            "team_1_affiliation_date": (
                affiliation_1.source_date.isoformat() if affiliation_1 else ""
            ),
            "team_2_family": affiliation_2.family if affiliation_2 else UNKNOWN,
            "team_2_tier": affiliation_2.tier if affiliation_2 else UNKNOWN,
            "team_2_affiliation_date": (
                affiliation_2.source_date.isoformat() if affiliation_2 else ""
            ),
            "bootstrap_block": bootstrap_block_key(self),
        }
        row.update(
            {f"cohort_{cohort}": int(cohort in self.cohorts) for cohort in COHORTS}
        )
        return row


def _natural_id_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value, value)


def canonical_match_id(value: object) -> str:
    if value is None:
        raise ValueError("empty match ID")
    match_id = str(value).strip()
    if not match_id:
        raise ValueError("empty match ID")
    return match_id


def _calendar_date(value: object, *, field: str) -> date:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error


def _normalized_players(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    players = {
        str(value).strip()
        for value in values
        if value is not None and str(value).strip()
    }
    return tuple(sorted(players, key=_natural_id_key))


def _scores(raw: dict[str, Any], normalized: dict[str, Any]) -> tuple[int, ...]:
    raw_games = normalized["games"]
    if raw_games:
        scores: list[int] = []
        for game in raw_games:
            if bool(game.get("draw")):
                return ()
            team_1_win = game.get("t1_win")
            team_2_win = game.get("t2_win")
            if team_1_win is None and team_2_win is None:
                return ()
            if team_1_win is None:
                team_1_win = not bool(team_2_win)
            if team_2_win is None:
                team_2_win = not bool(team_1_win)
            if bool(team_1_win) == bool(team_2_win):
                return ()
            compatible_game = dict(game)
            compatible_game["t1_win"] = bool(team_1_win)
            compatible_game["t2_win"] = bool(team_2_win)
            scores.append(game_score_for_match_team1(raw, compatible_game))
        return tuple(scores)

    score_1 = int(normalized["score_1"])
    score_2 = int(normalized["score_2"])
    if score_1 != score_2:
        return (1,) * score_1 + (0,) * score_2
    if bool(raw.get("draw")):
        return ()
    team_1_win = raw.get("t1_win")
    team_2_win = raw.get("t2_win")
    if team_1_win is None and team_2_win is None:
        return ()
    if team_1_win is None:
        team_1_win = not bool(team_2_win)
    if team_2_win is None:
        team_2_win = not bool(team_1_win)
    if bool(team_1_win) == bool(team_2_win):
        return ()
    return (int(bool(team_1_win)),)


def load_matches(path: Path) -> tuple[list[MatchRecord], dict[str, int]]:
    """Load eligible match history through the canonical GOL.GG schema helpers."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("matches input must contain a JSON list")

    counts: Counter[str] = Counter(raw_matches=len(payload))
    seen_ids: set[str] = set()
    records: list[MatchRecord] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"match at JSON index {index} is not an object")
        normalized = normalized_match(raw)
        match_id = canonical_match_id(normalized["match_id"])
        if match_id in seen_ids:
            raise ValueError(f"duplicate match ID in matches input: {match_id}")
        seen_ids.add(match_id)
        match_date = _calendar_date(normalized["date"], field=f"date for match {match_id}")
        tournament = str(normalized["tournament"] or "").strip()
        competition = classify_competition(tournament, match_date)
        if (
            competition.family == UNKNOWN
            or competition.tier is CompetitionTier.UNKNOWN
            or competition.scope is CompetitionScope.UNKNOWN
        ):
            raise ValueError(
                "unknown competition classification for "
                f"match {match_id}: {tournament!r} ({competition.matched_rule})"
            )

        team_1_id = str(normalized["tid_1"] or "").strip()
        team_2_id = str(normalized["tid_2"] or "").strip()
        if not team_1_id or not team_2_id or team_1_id == team_2_id:
            counts["excluded_invalid_identity"] += 1
            continue

        scores = _scores(raw, normalized)
        if not scores or sum(scores) * 2 == len(scores):
            counts["excluded_draw_or_missing_outcome"] += 1
            continue
        players_1 = _normalized_players(normalized["players_1"])
        players_2 = _normalized_players(normalized["players_2"])
        if not players_1 or not players_2:
            counts["excluded_missing_roster"] += 1
            continue
        if set(players_1) & set(players_2):
            counts["excluded_invalid_identity"] += 1
            continue

        records.append(
            MatchRecord(
                match_id=match_id,
                match_date=match_date,
                tournament=tournament,
                competition=competition,
                team_1_id=team_1_id,
                team_2_id=team_2_id,
                players_1=players_1,
                players_2=players_2,
                scores=scores,
                y_true=int(sum(scores) * 2 > len(scores)),
            )
        )

    records.sort(
        key=lambda record: (record.match_date, _natural_id_key(record.match_id))
    )
    counts["eligible_matches"] = len(records)
    return records, {
        key: int(counts[key])
        for key in (
            "raw_matches",
            "excluded_draw_or_missing_outcome",
            "excluded_missing_roster",
            "excluded_invalid_identity",
            "eligible_matches",
        )
    }


def _binary_label(value: object, *, match_id: str) -> int:
    text = str(value).strip()
    if text not in {"0", "1", "0.0", "1.0"}:
        raise ValueError(f"baseline y_true for match {match_id} is not binary")
    return int(float(text))


def load_baseline(path: Path) -> dict[str, BaselinePrediction]:
    """Load only the four declared baseline columns; no odds source is consulted."""

    required = {"golgg_match_id", "date", "player_gl", "y_true"}
    predictions: dict[str, BaselinePrediction] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "baseline CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        for row_number, row in enumerate(reader, start=2):
            match_id = canonical_match_id(row["golgg_match_id"])
            if match_id in predictions:
                raise ValueError(f"duplicate match ID in baseline CSV: {match_id}")
            try:
                probability = float(row["player_gl"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid player_gl for match {match_id} on CSV row {row_number}"
                ) from error
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"player_gl for match {match_id} is outside [0, 1]")
            predictions[match_id] = BaselinePrediction(
                match_id=match_id,
                match_date=_calendar_date(
                    row["date"], field=f"baseline date for match {match_id}"
                ),
                probability=probability,
                y_true=_binary_label(row["y_true"], match_id=match_id),
            )
    if not predictions:
        raise ValueError("baseline CSV contains no prediction rows")
    return predictions


def _event_affiliations(
    match: MatchRecord,
    prior: Mapping[str, DomesticAffiliation],
) -> tuple[DomesticAffiliation | None, DomesticAffiliation | None]:
    affiliation_1 = prior.get(match.team_1_id)
    affiliation_2 = prior.get(match.team_2_id)
    if (
        match.competition.scope is CompetitionScope.CROSS_LEAGUE
        and (affiliation_1 is None or affiliation_2 is None)
    ):
        # A partially known bridge would apply a one-sided location offset. Passing
        # unknown/unknown keeps this event strictly player-only as predeclared.
        return None, None
    return affiliation_1, affiliation_2


def _apply_domestic_affiliations(
    prior: dict[str, DomesticAffiliation],
    period: Sequence[MatchRecord],
) -> None:
    candidates: defaultdict[str, list[DomesticAffiliation]] = defaultdict(list)
    for match in period:
        if match.competition.scope is not CompetitionScope.DOMESTIC:
            continue
        for team_id in (match.team_1_id, match.team_2_id):
            candidates[team_id].append(
                DomesticAffiliation(
                    family=match.competition.family,
                    tier=match.competition.tier.value,
                    source_date=match.match_date,
                    source_match_id=match.match_id,
                )
            )
    for team_id, values in sorted(candidates.items()):
        destinations = {(item.family, item.tier) for item in values}
        if len(destinations) != 1:
            continue
        prior[team_id] = min(
            values, key=lambda item: _natural_id_key(item.source_match_id)
        )


def replay_candidate(
    records: Sequence[MatchRecord],
) -> dict[str, CandidatePrediction]:
    """Replay one fixed candidate in frozen, complete calendar-date periods."""

    engine = FamilyCalibratedGlicko2()
    domestic_affiliations: dict[str, DomesticAffiliation] = {}
    predictions: dict[str, CandidatePrediction] = {}
    for _, grouped in groupby(records, key=lambda record: record.match_date):
        period = tuple(grouped)
        events: list[RatingEvent] = []
        snapshots: dict[
            str, tuple[DomesticAffiliation | None, DomesticAffiliation | None]
        ] = {}
        for match in period:
            affiliation_1, affiliation_2 = _event_affiliations(
                match, domestic_affiliations
            )
            snapshots[match.match_id] = (affiliation_1, affiliation_2)
            events.append(
                RatingEvent(
                    event_id=match.match_id,
                    event_date=match.match_date,
                    team_a_id=match.team_1_id,
                    team_b_id=match.team_2_id,
                    players_a=match.players_1,
                    players_b=match.players_2,
                    family_a=affiliation_1.family if affiliation_1 else UNKNOWN,
                    family_b=affiliation_2.family if affiliation_2 else UNKNOWN,
                    tier_a=affiliation_1.tier if affiliation_1 else UNKNOWN,
                    tier_b=affiliation_2.tier if affiliation_2 else UNKNOWN,
                    scores=match.scores,
                )
            )
        period_probabilities = engine.process_period(events)
        for match in period:
            affiliation_1, affiliation_2 = snapshots[match.match_id]
            predictions[match.match_id] = CandidatePrediction(
                match_id=match.match_id,
                probability=float(period_probabilities[match.match_id]),
                affiliation_1=affiliation_1,
                affiliation_2=affiliation_2,
            )
        # This is intentionally after every prediction and rating update on the
        # date. Thus source_date is always strictly earlier for the next period.
        _apply_domestic_affiliations(domestic_affiliations, period)
    return predictions


def _cohort_memberships(
    match: MatchRecord, candidate: CandidatePrediction
) -> frozenset[str]:
    affiliation_1 = candidate.affiliation_1
    affiliation_2 = candidate.affiliation_2
    tier_1 = affiliation_1.tier if affiliation_1 else UNKNOWN
    tier_2 = affiliation_2.tier if affiliation_2 else UNKNOWN
    memberships = {"overall"}
    if tier_1 == tier_2 == CompetitionTier.MAJOR.value:
        memberships.add("major_major")
    if tier_1 == tier_2 == CompetitionTier.REGIONAL.value:
        memberships.add("regional_regional")
    if (
        match.competition.tier is CompetitionTier.DEVELOPMENT
        or CompetitionTier.DEVELOPMENT.value in {tier_1, tier_2}
    ):
        memberships.add("development_involved")
    if (
        match.competition.scope is CompetitionScope.CROSS_LEAGUE
        and affiliation_1 is not None
        and affiliation_2 is not None
    ):
        memberships.add("known_cross_league")
    if {tier_1, tier_2} == {
        CompetitionTier.MAJOR.value,
        CompetitionTier.MINOR_TOP_LEVEL.value,
    }:
        memberships.add("major_vs_minor")
    if {tier_1, tier_2} == {
        CompetitionTier.MAJOR.value,
        CompetitionTier.REGIONAL.value,
    }:
        memberships.add("major_vs_regional")
    return frozenset(memberships)


def align_evaluation_rows(
    records: Sequence[MatchRecord],
    candidates: Mapping[str, CandidatePrediction],
    baseline: Mapping[str, BaselinePrediction],
    *,
    diagnostic_start: date = DIAGNOSTIC_START,
) -> tuple[list[EvaluationRow], dict[str, int]]:
    """Validate the keyed join and return identical candidate/control rows."""

    record_by_id = {record.match_id: record for record in records}
    player_day_appearances = Counter(
        (record.match_date, player_id)
        for record in records
        for player_id in (*record.players_1, *record.players_2)
    )

    aligned_all = 0
    aligned_before_diagnostic = 0
    diagnostic_before_same_day_exclusion = 0
    excluded_repeated_player_same_date = 0
    rows: list[EvaluationRow] = []
    for record in records:
        legacy = baseline.get(record.match_id)
        if legacy is None:
            continue
        aligned_all += 1
        if legacy.match_date != record.match_date:
            raise ValueError(
                f"baseline/match date mismatch for {record.match_id}: "
                f"{legacy.match_date} != {record.match_date}"
            )
        if legacy.y_true != record.y_true:
            raise ValueError(
                f"baseline/match label mismatch for {record.match_id}: "
                f"{legacy.y_true} != {record.y_true}"
            )
        candidate = candidates.get(record.match_id)
        if candidate is None:
            raise RuntimeError(f"candidate prediction missing for {record.match_id}")
        if record.match_date < diagnostic_start:
            aligned_before_diagnostic += 1
            continue
        diagnostic_before_same_day_exclusion += 1
        if any(
            player_day_appearances[(record.match_date, player_id)] > 1
            for player_id in (*record.players_1, *record.players_2)
        ):
            excluded_repeated_player_same_date += 1
            continue
        rows.append(
            EvaluationRow(
                match=record,
                candidate=candidate,
                legacy_probability=legacy.probability,
                cohorts=_cohort_memberships(record, candidate),
            )
        )
    if not rows:
        raise ValueError(
            f"no aligned evaluation rows on or after {diagnostic_start.isoformat()}"
        )
    return rows, {
        "baseline_rows": len(baseline),
        "baseline_without_eligible_match": len(set(baseline) - set(record_by_id)),
        "candidate_predictions": len(candidates),
        "aligned_rows_all_dates": aligned_all,
        "aligned_rows_before_diagnostic": aligned_before_diagnostic,
        "diagnostic_rows_before_same_day_exclusion": (
            diagnostic_before_same_day_exclusion
        ),
        "diagnostic_rows_excluded_repeated_player_same_date": (
            excluded_repeated_player_same_date
        ),
        "diagnostic_rows_2024_plus": len(rows),
        "eligible_matches_without_baseline": len(records) - aligned_all,
    }


def bernoulli_log_loss(y_true: int, probability: float) -> float:
    clipped = min(max(float(probability), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    return -math.log(clipped if y_true else 1.0 - clipped)


def _auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _ece(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    total = len(labels)
    weighted_error = 0.0
    for bin_index in range(ECE_BINS):
        members = [
            index
            for index, probability in enumerate(probabilities)
            if min(int(probability * ECE_BINS), ECE_BINS - 1) == bin_index
        ]
        if not members:
            continue
        confidence = math.fsum(probabilities[index] for index in members) / len(members)
        accuracy = math.fsum(labels[index] for index in members) / len(members)
        weighted_error += len(members) * abs(accuracy - confidence) / total
    return weighted_error


def probability_metrics(
    rows: Sequence[EvaluationRow], *, candidate: bool
) -> dict[str, object]:
    if not rows:
        return {
            "n": 0,
            "date_min": None,
            "date_max": None,
            "log_loss": None,
            "brier": None,
            "auc": None,
            "ece": None,
        }
    labels = [row.match.y_true for row in rows]
    probabilities = [
        row.candidate.probability if candidate else row.legacy_probability
        for row in rows
    ]
    return {
        "n": len(rows),
        "date_min": min(row.match.match_date for row in rows).isoformat(),
        "date_max": max(row.match.match_date for row in rows).isoformat(),
        "log_loss": math.fsum(
            bernoulli_log_loss(label, probability)
            for label, probability in zip(labels, probabilities, strict=True)
        )
        / len(rows),
        "brier": math.fsum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities, strict=True)
        )
        / len(rows),
        "auc": _auc(labels, probabilities),
        "ece": _ece(labels, probabilities),
    }


def bootstrap_block_key(row: EvaluationRow | Mapping[str, object]) -> str:
    """Return the fixed paired-resampling block for one evaluation row."""

    if isinstance(row, EvaluationRow):
        row_date = row.match.match_date
        scope = row.match.competition.scope.value
        family_1 = row.candidate.affiliation_1.family if row.candidate.affiliation_1 else UNKNOWN
        family_2 = row.candidate.affiliation_2.family if row.candidate.affiliation_2 else UNKNOWN
    else:
        row_date = _calendar_date(row["date"], field="bootstrap row date")
        scope = str(row["competition_scope"])
        family_1 = str(row.get("team_1_family") or UNKNOWN)
        family_2 = str(row.get("team_2_family") or UNKNOWN)
    if scope == CompetitionScope.CROSS_LEAGUE.value:
        family_pair = "|".join(sorted((family_1, family_2)))
        return f"family-pair:{family_pair}/year:{row_date.year}"
    return f"month:{row_date:%Y-%m}"


def build_bootstrap_blocks(
    rows: Sequence[EvaluationRow],
) -> dict[str, tuple[str, ...]]:
    members: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        members[bootstrap_block_key(row)].append(row.match.match_id)
    return {
        key: tuple(sorted(values, key=_natural_id_key))
        for key, values in sorted(members.items())
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_block_bootstrap(
    rows: Sequence[EvaluationRow],
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    if repetitions < 5_000:
        raise ValueError("paired block bootstrap requires at least 5000 repetitions")
    if not rows:
        return {
            "n": 0,
            "unit": (
                "calendar month for domestic rows; competition-family-pair/year for cross rows"
            ),
            "blocks": 0,
            "block_keys": [],
            "repetitions": repetitions,
            "seed": seed,
            "paired_log_loss_delta": None,
            "ci_95_two_sided_low": None,
            "ci_95_two_sided_high": None,
            "upper_one_sided_95": None,
        }

    indexed_blocks: defaultdict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        indexed_blocks[bootstrap_block_key(row)].append(row)
    block_keys = sorted(indexed_blocks)
    block_delta_sums = [
        math.fsum(row.paired_log_loss_delta for row in indexed_blocks[key])
        for key in block_keys
    ]
    block_sizes = [len(indexed_blocks[key]) for key in block_keys]
    point_delta = math.fsum(block_delta_sums) / math.fsum(block_sizes)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        sampled = [rng.randrange(len(block_keys)) for _ in block_keys]
        draws.append(
            math.fsum(block_delta_sums[index] for index in sampled)
            / math.fsum(block_sizes[index] for index in sampled)
        )
    return {
        "n": len(rows),
        "unit": (
            "calendar month for domestic rows; competition-family-pair/year for cross rows"
        ),
        "blocks": len(block_keys),
        "block_keys": block_keys,
        "repetitions": repetitions,
        "seed": seed,
        "paired_log_loss_delta": point_delta,
        "ci_95_two_sided_low": _quantile(draws, 0.025),
        "ci_95_two_sided_high": _quantile(draws, 0.975),
        "upper_one_sided_95": _quantile(draws, 0.95),
    }


def _cohort_results(rows: Sequence[EvaluationRow]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for cohort in COHORTS:
        selected = [row for row in rows if cohort in row.cohorts]
        candidate_metrics = probability_metrics(selected, candidate=True)
        legacy_metrics = probability_metrics(selected, candidate=False)
        results[cohort] = {
            "definition": COHORT_DEFINITIONS[cohort],
            "candidate": candidate_metrics,
            "legacy": legacy_metrics,
            "paired_log_loss_delta": (
                None
                if not selected
                else float(candidate_metrics["log_loss"])
                - float(legacy_metrics["log_loss"])
            ),
            "bootstrap": paired_block_bootstrap(selected),
        }
    return results


def _gate_evidence(
    rows: Sequence[EvaluationRow], cohort_results: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    overall_bootstrap = cohort_results["overall"]["bootstrap"]
    bridge_bootstrap = cohort_results["known_cross_league"]["bootstrap"]
    if not isinstance(overall_bootstrap, Mapping) or not isinstance(
        bridge_bootstrap, Mapping
    ):
        raise TypeError("invalid bootstrap result")
    bridge_rows = [row for row in rows if "known_cross_league" in row.cohorts]
    bridge_families = sorted(
        {
            affiliation.family
            for row in bridge_rows
            for affiliation in (
                row.candidate.affiliation_1,
                row.candidate.affiliation_2,
            )
            if affiliation is not None
        }
    )
    bridge_seasons = sorted({row.match.match_date.year for row in bridge_rows})
    bridge_evidence = {
        "matches": len(bridge_rows),
        "family_pair_year_blocks": int(bridge_bootstrap["blocks"]),
        "domestic_families": bridge_families,
        "seasons": bridge_seasons,
    }
    bridge_evidence_sufficient = (
        len(bridge_rows) >= 200
        and int(bridge_bootstrap["blocks"]) >= 12
        and len(bridge_families) >= 4
        and len(bridge_seasons) >= 2
    )
    overall_upper = float(overall_bootstrap["upper_one_sided_95"])
    bridge_upper_value = bridge_bootstrap["ci_95_two_sided_high"]
    bridge_upper = (
        None if bridge_upper_value is None else float(bridge_upper_value)
    )
    return {
        "evidence_scope": "reused_2024_plus_diagnostic_only",
        "overall_noninferiority": {
            "criterion": (
                "upper one-sided 95% paired block-bootstrap bound for candidate minus legacy "
                "LogLoss is below +0.002"
            ),
            "margin": 0.002,
            "diagnostic_upper_bound": overall_upper,
            "diagnostic_condition_met": overall_upper < 0.002,
            "promotion_evidence": False,
        },
        "bridge_improvement": {
            "criterion": (
                "upper two-sided 95% paired family-pair/year bootstrap bound for candidate "
                "minus legacy LogLoss is below zero"
            ),
            "minimum_evidence": {
                "matches": 200,
                "family_pair_year_blocks": 12,
                "domestic_families": 4,
                "seasons": 2,
            },
            "observed_evidence": bridge_evidence,
            "evidence_sufficient": bridge_evidence_sufficient,
            "diagnostic_upper_bound": bridge_upper,
            "diagnostic_direction_met": (
                None if bridge_upper is None else bridge_upper < 0.0
            ),
            "diagnostic_condition_met": (
                None
                if bridge_upper is None or not bridge_evidence_sufficient
                else bridge_upper < 0.0
            ),
            "promotion_evidence": False,
        },
        "prospective_promotion": {
            "available": False,
            "decision": None,
            "reason": (
                "The 2024+ outcomes are an already-reused diagnostic cohort. This run does not "
                "implement a frozen, blinded prospective protocol, so it cannot authorize an "
                "operational-default replacement."
            ),
        },
    }


def evaluate(
    matches_path: Path,
    baseline_path: Path,
) -> tuple[dict[str, object], list[EvaluationRow]]:
    records, match_counts = load_matches(matches_path)
    if not records:
        raise ValueError("matches input has no eligible matches")
    baseline = load_baseline(baseline_path)
    candidate_predictions = replay_candidate(records)
    rows, alignment_counts = align_evaluation_rows(
        records, candidate_predictions, baseline
    )
    cohort_results = _cohort_results(rows)
    cross_rows = [
        row
        for row in rows
        if row.match.competition.scope is CompetitionScope.CROSS_LEAGUE
    ]
    cross_player_only = sum(
        row.candidate.affiliation_1 is None
        and row.candidate.affiliation_2 is None
        for row in cross_rows
    )
    summary: dict[str, object] = {
        "experiment": "EXP-075-successor-fixed-evaluation",
        "status": "completed_diagnostic",
        "systems": {
            "candidate": {
                "system": CANDIDATE_SYSTEM,
                "version": CANDIDATE_VERSION,
                "engine": "src.ratings.family_calibrated_glicko2.FamilyCalibratedGlicko2",
                "parameters": {
                    "tau": 0.5,
                    "convergence_tolerance": 1e-6,
                    "rating_period_days": 30.0,
                    "initial_family_deviation": 150.0,
                    "initial_tier_deviation": 100.0,
                    "bridge_process_deviation": 1.0,
                },
            },
            "legacy": {
                "system": LEGACY_SYSTEM,
                "probability_column": "player_gl",
            },
        },
        "inputs": {
            "matches": str(matches_path),
            "baseline": str(baseline_path),
            "baseline_columns_consumed": [
                "golgg_match_id",
                "date",
                "player_gl",
                "y_true",
            ],
            "odds_files_read": False,
            "odds_values_used": False,
            "database_read_or_modified": False,
            "model_or_rating_artifacts_read_or_modified": False,
        },
        "diagnostic_window": {
            "start_inclusive": DIAGNOSTIC_START.isoformat(),
            "date_min": min(row.match.match_date for row in rows).isoformat(),
            "date_max": max(row.match.match_date for row in rows).isoformat(),
            "prospective": False,
            "interpretation": (
                "2024+ is already-reused diagnostic evidence and cannot promote this candidate."
            ),
        },
        "counts": {
            "matches": match_counts,
            "alignment": alignment_counts,
            "unknown_competitions": 0,
            "diagnostic_cross_league_rows": len(cross_rows),
            "diagnostic_cross_rows_player_only_missing_prior_affiliation": cross_player_only,
        },
        "integrity": {
            "alignment_key": "golgg_match_id",
            "date_and_label_equality_required": True,
            "calendar_date_batched_replay": True,
            "affiliations_require_source_date_strictly_before_match_date": True,
            "same_day_affiliation_updates_deferred_until_period_complete": True,
            "legacy_same_day_sequential_exclusion": (
                "exclude every diagnostic row when any participant appears in another "
                "eligible match on the same calendar date"
            ),
            "unknown_competition_policy": "fail_closed",
            "cross_missing_prior_affiliation_policy": "unknown/unknown; player-only",
        },
        "metrics": cohort_results,
        "gates": _gate_evidence(rows, cohort_results),
        "operational_default_replacement": {
            "available": False,
            "recommendation": (
                "Do not replace an operational default from this run; retain the server snapshot "
                "as a candidate until prospective promotion evidence is available."
            ),
        },
        "artifacts": ["summary.json", "predictions.csv"],
    }
    return summary, rows


def _render_predictions(rows: Sequence[EvaluationRow]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=PREDICTION_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(row.to_csv_row() for row in rows)
    return buffer.getvalue()


def write_artifacts(
    output_dir: Path,
    summary: Mapping[str, object],
    rows: Sequence[EvaluationRow],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.csv").write_text(
        _render_predictions(rows), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation(
    matches_path: Path,
    baseline_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    matches_resolved = matches_path.resolve()
    baseline_resolved = baseline_path.resolve()
    targets = {
        (output_dir / "summary.json").resolve(),
        (output_dir / "predictions.csv").resolve(),
    }
    if matches_resolved in targets or baseline_resolved in targets:
        raise ValueError("output artifacts must not overwrite an input")
    summary, rows = evaluate(matches_path, baseline_path)
    write_artifacts(output_dir, summary, rows)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_evaluation(args.matches, args.baseline, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
