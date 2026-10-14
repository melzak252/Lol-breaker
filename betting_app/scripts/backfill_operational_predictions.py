"""Replay the regional operational model chronologically for completed matches.

This is a retrospective evaluation dataset, not a live prediction record. For
each complete calendar date, every target prediction is emitted from state that
contains only earlier dates; the date is updated only after all its predictions
are materialized. It never reads bookmaker odds as model input.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from datetime import UTC, date, datetime, time, timedelta
from itertools import groupby
from typing import Any, Sequence

from betting_app.core.db import connect, transaction
from betting_app.core.matching import normalize_team_name, similarity
from betting_app.scripts import rebuild_calibrated_ratings as calibrated
from betting_app.scripts import rebuild_regional_ratings as regional
from betting_app.scripts import rebuild_w20_features as w20
from betting_app.services.rating_contract import (
    OPERATIONAL_BACKFILL_FEATURE_VERSION,
    OPERATIONAL_BACKFILL_MODEL_VERSION,
    OPERATIONAL_MODEL_NAME,
)
from betting_app.services.upcoming_inference_service import (
    _normalized_best_of,
    predict_probability_from_features,
    series_probability,
    w20_probability,
)
from src.ratings.competition_adjustment import CompetitionAdjustment, adjust_probability
from src.ratings.manager import RatingManager

BACKFILL_RATINGS_VERSION = "ratings-v2-chronological-v1"
PREDICTION_MODE = "chronological-regional-backfill-v1"


def _parse_start(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_targets() -> dict[str, dict[str, Any]]:
    """Return one auditable canonical target per linked completed GOL.GG match."""
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT gmm.golgg_match_id, cm.id AS canonical_match_id,
                   cm.start_time_normalized, cm.best_of, cm.winner_side,
                   cm.team_a_name, cm.team_b_name
            FROM golgg_match_mappings gmm
            JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
            WHERE cm.status IN ('finished', 'completed')
              AND cm.winner_side IN ('team_a', 'team_b')
              AND cm.start_time_normalized IS NOT NULL
            ORDER BY gmm.golgg_match_id, cm.id
            """
        ).fetchall()
    targets: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row["golgg_match_id"])
        if key in targets:
            raise ValueError(f"multiple canonical matches map to GOL.GG match {key}")
        targets[key] = dict(row)
    return targets

def _is_reversed_mapping(
    canonical_a: str, canonical_b: str, golgg_a: str, golgg_b: str
) -> bool:
    """Check whether canonical match sides are reversed relative to GOL.GG order."""
    norm_ca = normalize_team_name(canonical_a)
    norm_cb = normalize_team_name(canonical_b)
    norm_ga = normalize_team_name(golgg_a)
    norm_gb = normalize_team_name(golgg_b)

    if norm_ca == norm_ga or norm_cb == norm_gb:
        return False
    if norm_ca == norm_gb or norm_cb == norm_ga:
        return True

    sim_direct = similarity(canonical_a, golgg_a) + similarity(canonical_b, golgg_b)
    sim_reversed = similarity(canonical_a, golgg_b) + similarity(canonical_b, golgg_a)
    return sim_reversed > sim_direct


def _event_affiliations(
    period: Sequence[calibrated.LoadedMatch],
    metadata: calibrated.RebuildMetadata,
) -> tuple[list[calibrated.RatingEvent], dict[str, tuple[str, str]]]:
    """Build the frozen-prior regional events and affiliation lookup for one date."""
    team_candidates = calibrated._domestic_team_candidates(period)
    events: list[calibrated.RatingEvent] = []
    affiliations: dict[str, tuple[str, str]] = {}
    for match in period:
        if match.competition.scope is calibrated.CompetitionScope.DOMESTIC:
            affiliation_a = calibrated._single_affiliation(team_candidates.get(match.team_a_id, ()))
            affiliation_b = calibrated._single_affiliation(team_candidates.get(match.team_b_id, ()))
        else:
            affiliation_a = metadata.team_affiliations.get(match.team_a_id)
            affiliation_b = metadata.team_affiliations.get(match.team_b_id)
        family_a = affiliation_a.family if affiliation_a else calibrated.UNKNOWN_AFFILIATION
        family_b = affiliation_b.family if affiliation_b else calibrated.UNKNOWN_AFFILIATION
        tier_a = affiliation_a.tier if affiliation_a else calibrated.CompetitionTier.UNKNOWN.value
        tier_b = affiliation_b.tier if affiliation_b else calibrated.CompetitionTier.UNKNOWN.value
        affiliations[match.event_id] = (family_a, tier_a, family_b, tier_b)
        events.append(
            calibrated.RatingEvent(
                event_id=match.event_id,
                event_date=match.event_date,
                team_a_id=match.team_a_id,
                team_b_id=match.team_b_id,
                players_a=tuple(player.player_id for player in match.players_a),
                players_b=tuple(player.player_id for player in match.players_b),
                family_a=family_a,
                family_b=family_b,
                tier_a=tier_a,
                tier_b=tier_b,
                scores=match.scores,
            )
        )
    return events, affiliations


def _raw_probabilities(
    manager: RatingManager,
    match: calibrated.LoadedMatch,
    adjustment: CompetitionAdjustment,
) -> tuple[dict[str, float], dict[str, float]]:
    players_a = [player.player_id for player in match.players_a]
    players_b = [player.player_id for player in match.players_b]
    team_probabilities: dict[str, float] = {}
    player_probabilities: dict[str, float] = {}
    for system_name in regional.RAW_SYSTEMS:
        system = manager.systems[system_name]
        team_probabilities[system_name] = adjust_probability(
            float(system.predict_team_win_prob(match.team_a_id, match.team_b_id)), adjustment
        )
        player_probabilities[system_name] = adjust_probability(
            float(system.predict_player_win_prob(players_a, players_b)), adjustment
        )
    return team_probabilities, player_probabilities


def _update_raw_manager(manager: RatingManager, match: calibrated.LoadedMatch) -> None:
    players_a = [player.player_id for player in match.players_a]
    players_b = [player.player_id for player in match.players_b]
    for score_a in match.scores:
        for system_name in regional.RAW_SYSTEMS:
            system = manager.systems[system_name]
            system.update_team(match.team_a_id, match.team_b_id, score_a, 1 - score_a)
            system.update_player(players_a, players_b, score_a, 1 - score_a)


def _w20_features(history: deque[dict[str, float]]) -> dict[str, float]:
    averaged = w20.average_history(history)
    return {
        "win_rate": averaged["win_rate"],
        "avg_kills": averaged["kills"],
        "avg_deaths": averaged["deaths"],
        "avg_gd15": averaged["gd15"],
        "avg_dpm": averaged["dpm"],
        "avg_vspm": averaged["vspm"],
        "avg_towers": averaged["towers"],
        "avg_dragons": averaged["dragons"],
        "avg_nashors": averaged["nashors"],
        "avg_gold": averaged["gold"],
        "avg_game_duration": averaged["duration"],
    }


def _register_backfill_artifact(*, apply: bool) -> int | None:
    if not apply:
        return None
    feature_schema = json.dumps(
        {
            "prediction_mode": PREDICTION_MODE,
            "ratings": list(regional.PUBLIC_SYSTEMS),
            "ratings_version": BACKFILL_RATINGS_VERSION,
            "features_version": OPERATIONAL_BACKFILL_FEATURE_VERSION,
            "formula": "map_p = 0.70 * player_rating_consensus + 0.20 * team_rating_consensus + 0.10 * w20_probability; match_p = binomial_tail(map_p, best_of); no market input",
            "temporal_rule": "state contains complete calendar dates strictly before the target date",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    params = json.dumps(
        {
            "player_rating_weight": 0.70,
            "team_rating_weight": 0.20,
            "w20_weight": 0.10,
            "series_projection": "binomial_tail",
            "supported_best_of": [1, 3, 5, 7],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO model_artifacts(
                model_name, model_version, feature_schema_json, model_params_json, status
            ) VALUES (?, ?, ?, ?, 'active')
            ON CONFLICT(model_name, model_version) DO NOTHING
            """,
            (OPERATIONAL_MODEL_NAME, OPERATIONAL_BACKFILL_MODEL_VERSION, feature_schema, params),
        )
        row = connection.execute(
            """
            SELECT id, feature_schema_json, model_params_json, status
            FROM model_artifacts WHERE model_name = ? AND model_version = ?
            """,
            (OPERATIONAL_MODEL_NAME, OPERATIONAL_BACKFILL_MODEL_VERSION),
        ).fetchone()
    if row is None or row["status"] != "active":
        raise RuntimeError("operational historical artifact was not persisted")
    if row["feature_schema_json"] != feature_schema or row["model_params_json"] != params:
        raise ValueError("existing operational historical artifact has a different contract")
    return int(row["id"])


def backfill_operational_predictions(*, apply: bool, limit: int | None = None) -> dict[str, int]:
    """Build one leakage-safe operational prediction per eligible finished match."""
    targets = _canonical_targets()
    matches = calibrated.load_matches()
    game_rows = w20.load_all_games_grouped()
    player_stats = w20.load_all_player_stats_grouped()
    engine = calibrated.FamilyCalibratedGlicko2()
    metadata = calibrated.RebuildMetadata()
    raw_manager = RatingManager(regional.raw.RATING_SYSTEM_PARAMS)
    history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=20))
    match_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=20))
    prepared: list[dict[str, Any]] = []
    skipped_timestamp = 0

    for _, grouped in groupby(matches, key=lambda item: item.event_date):
        period = tuple(grouped)
        events, affiliations = _event_affiliations(period, metadata)
        raw_by_event: dict[str, tuple[dict[str, float], dict[str, float], CompetitionAdjustment]] = {}
        for match in period:
            family_a, tier_a, family_b, tier_b = affiliations[match.event_id]
            location = engine.get_location_difference(family_a, tier_a, family_b, tier_b)
            adjustment = CompetitionAdjustment(mean=float(location.mean), variance=float(location.variance))
            team_probs, player_probs = _raw_probabilities(raw_manager, match, adjustment)
            raw_by_event[match.event_id] = (team_probs, player_probs, adjustment)

        regional_probs = engine.process_period(events)
        for match in period:
            target = targets.get(match.event_id)
            if target is None:
                continue
            start_at = _parse_start(target["start_time_normalized"])
            if start_at is None:
                continue
            cutoff_at = datetime.combine(match.event_date - timedelta(days=1), time.max, tzinfo=UTC)
            predicted_at = start_at - timedelta(minutes=1)
            if cutoff_at > predicted_at:
                skipped_timestamp += 1
                continue
            team_probs, player_probs, adjustment = raw_by_event[match.event_id]
            regional_probability = float(regional_probs[match.event_id])
            team_probs["gl"] = regional_probability
            player_probs["gl"] = regional_probability
            team_consensus = sum(team_probs.values()) / len(team_probs)
            player_consensus = sum(player_probs.values()) / len(player_probs)
            w20_a = _w20_features(history[match.team_a_id])
            w20_b = _w20_features(history[match.team_b_id])
            feature_payload = {
                "ratings": {"probabilities": {"consensus": team_consensus, **team_probs}},
                "player_ratings": {"probabilities": {"consensus": player_consensus, **player_probs}},
                "w20": {"probability": w20_probability(w20_a, w20_b)},
            }
            map_probability, components = predict_probability_from_features(feature_payload)
            best_of = _normalized_best_of(target.get("best_of"))
            probability = series_probability(map_probability, best_of)
            is_reversed = _is_reversed_mapping(
                str(target.get("team_a_name") or ""),
                str(target.get("team_b_name") or ""),
                match.team_a_name,
                match.team_b_name,
            )
            canonical_prob_a = 1.0 - probability if is_reversed else probability
            prepared.append(
                {
                    "canonical_match_id": int(target["canonical_match_id"]),
                    "predicted_at": predicted_at.isoformat(),
                    "data_cutoff_at": cutoff_at.isoformat(),
                    "prob_a": canonical_prob_a,
                    "diagnostics_json": json.dumps(
                        {
                            **components,
                            "prediction_mode": PREDICTION_MODE,
                            "golgg_match_id": match.event_id,
                            "event_date": match.event_date.isoformat(),
                            "best_of": best_of,
                            "is_reversed_side": is_reversed,
                            "map_win_probability": 1.0 - map_probability if is_reversed else map_probability,
                            "series_win_probability": canonical_prob_a,
                            "regional_adjustment": {"mean": adjustment.mean, "variance": adjustment.variance},
                            "team_probabilities": team_probs,
                            "player_probabilities": player_probs,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            if limit is not None and len(prepared) >= limit:
                break
        if limit is not None and len(prepared) >= limit:
            break

        team_candidates = calibrated._domestic_team_candidates(period)
        calibrated._apply_period_metadata(metadata, period, team_candidates)
        for match in period:
            _update_raw_manager(raw_manager, match)
            for game in game_rows.get(match.event_id, []):
                w20.update_team_history(history, match_history, match.team_a_id, match.event_id, game, player_stats)
                w20.update_team_history(history, match_history, match.team_b_id, match.event_id, game, player_stats)

    artifact_id = _register_backfill_artifact(apply=apply)
    if apply:
        assert artifact_id is not None
        with transaction() as connection:
            connection.execute(
                """
                UPDATE canonical_predictions SET prediction_status = 'stale'
                WHERE model_name = ? AND model_version = ? AND prediction_status = 'active'
                """,
                (OPERATIONAL_MODEL_NAME, OPERATIONAL_BACKFILL_MODEL_VERSION),
            )
            for row in prepared:
                connection.execute(
                    """
                    INSERT INTO canonical_predictions(
                        canonical_match_id, model_artifact_id, model_name, model_version,
                        predicted_at, prob_a, prob_b, prediction_status, features_version,
                        ratings_version, data_cutoff_at, diagnostics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                    """,
                    (
                        row["canonical_match_id"],
                        artifact_id,
                        OPERATIONAL_MODEL_NAME,
                        OPERATIONAL_BACKFILL_MODEL_VERSION,
                        row["predicted_at"],
                        row["prob_a"],
                        1.0 - row["prob_a"],
                        OPERATIONAL_BACKFILL_FEATURE_VERSION,
                        BACKFILL_RATINGS_VERSION,
                        row["data_cutoff_at"],
                        row["diagnostics_json"],
                    ),
                )
    return {
        "targets": len(targets),
        "prepared": len(prepared),
        "skipped_timestamp": skipped_timestamp,
        "applied": int(apply),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist the rebuilt comparison cohort.")
    parser.add_argument("--limit", type=int, default=None, help="Predict only the first N eligible matches.")
    args = parser.parse_args(argv)
    result = backfill_operational_predictions(apply=args.apply, limit=args.limit)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
