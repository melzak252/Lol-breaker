#!/usr/bin/env python3
"""Materialize the immutable ``ratings-v2`` regional multi-rating snapshot.

The snapshot shares one chronological match cohort across all six public systems:
Elo, family-calibrated Glicko-2 (``gl``), TrueSkill, OpenSkill,
Plackett-Luce, and Thurstone-Mosteller.  The ``gl`` state is the sole
regional posterior; inference projects that posterior onto the five raw
systems exactly once for cross-family predictions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from typing import Any, Sequence

import trueskill
from betting_app.scripts import rebuild_calibrated_ratings as calibrated
from betting_app.scripts import rebuild_ratings as raw
from src.models.competition_tiers import CompetitionTier
from src.ratings.manager import RatingManager

ROOKIE_MAJOR_GAMES_THRESHOLD = 20
ROOKIE_MAJOR_ELO_CAP = 1750.0
ROOKIE_MAJOR_TS_MU_CAP = 28.0
ROOKIE_MAJOR_TS_SIGMA_FLOOR = 3.83
ROOKIE_MAJOR_OS_MU_CAP = 28.0
ROOKIE_MAJOR_OS_SIGMA_FLOOR = 3.83
ROOKIE_MAJOR_PL_MU_CAP = 33.0
ROOKIE_MAJOR_PL_SIGMA_FLOOR = 5.0
ROOKIE_MAJOR_TM_MU_CAP = 31.0
ROOKIE_MAJOR_TM_SIGMA_FLOOR = 3.95
RATINGS_VERSION = "ratings-v2"
DEFAULT_SOURCE = "regional-ratings-v2"
REGIONAL_ENGINE = "family-calibrated-glicko2-v1"
RAW_SYSTEMS = ("elo", "ts", "os", "pl", "tm")
PUBLIC_SYSTEMS = ("elo", "gl", "ts", "os", "pl", "tm")


def _raw_rows(
    *,
    manager: RatingManager,
    engine: calibrated.FamilyCalibratedGlicko2,
    metadata: calibrated.RebuildMetadata,
    run_id: int,
    version: str,
    snapshot_at: str,
) -> list[tuple[Any, ...]]:
    """Serialize raw systems for exactly the regional Glicko entity cohort."""
    rows: list[tuple[Any, ...]] = []
    player_ids = set(engine.player_ids)
    eligible_team_ids = {
        team_id
        for team_id, roster in metadata.team_rosters.items()
        if any(player_id in player_ids for player_id in roster)
    }
    for system_name in RAW_SYSTEMS:
        system = manager.systems[system_name]
        team_rows: dict[str, tuple[Any, ...]] = {}
        team_priorities: dict[str, tuple[str, str]] = {}
        for team_id in sorted(eligible_team_ids):
            rating = system.team_ratings.get(team_id)
            if rating is None:
                raise RuntimeError(
                    f"ratings-v2 {system_name} is missing team {team_id!r}"
                )
            team_name = metadata.team_names[team_id]
            normalized_team_name = calibrated.normalize_team_name(team_name)
            priority = (metadata.team_last_activity.get(team_id, ""), team_id)
            if priority <= team_priorities.get(normalized_team_name, ("", "")):
                continue
            team_priorities[normalized_team_name] = priority
            team_rows[normalized_team_name] = raw.entity_rating_row(
                run_id,
                version,
                snapshot_at,
                "team",
                team_name,
                normalized_team_name,
                team_name,
                None,
                system_name,
                rating,
                metadata.team_games[team_id],
                metadata.team_last_activity.get(team_id),
                {"team_id": team_id},
            )
        rows.extend(team_rows[key] for key in sorted(team_rows))

        for player_id in sorted(player_ids):
            rating = system.player_ratings.get(player_id)
            if rating is None:
                raise RuntimeError(
                    f"ratings-v2 {system_name} is missing player {player_id!r}"
                )
            team_id = metadata.player_team_ids.get(player_id)
            team_name = metadata.team_names.get(team_id, team_id) if team_id else None
            rows.append(
                raw.entity_rating_row(
                    run_id,
                    version,
                    snapshot_at,
                    "player",
                    metadata.player_names[player_id],
                    player_id,
                    team_name,
                    metadata.player_roles.get(player_id),
                    system_name,
                    rating,
                    metadata.player_games[player_id],
                    metadata.player_last_activity.get(player_id),
                    {"player_id": player_id},
                )
            )
    return rows


def _replay_raw_systems(
    matches: Sequence[calibrated.LoadedMatch],
) -> RatingManager:
    """Replay the same complete-date ordered matches used by regional Glicko,
    calibrating rookie transitions into major competition tiers."""
    manager = RatingManager(raw.RATING_SYSTEM_PARAMS)
    major_games_count: dict[str, int] = defaultdict(int)
    for match in matches:
        is_major = match.competition.tier in (
            CompetitionTier.MAJOR,
            CompetitionTier.INTERNATIONAL,
        )
        players_a = [player.player_id for player in match.players_a]
        players_b = [player.player_id for player in match.players_b]
        if is_major:
            for p in players_a + players_b:
                if major_games_count[p] < ROOKIE_MAJOR_GAMES_THRESHOLD:
                    elo = manager.systems["elo"].get_player_rating(p)
                    if elo > ROOKIE_MAJOR_ELO_CAP:
                        manager.systems["elo"].player_ratings[p] = ROOKIE_MAJOR_ELO_CAP
                    ts_r = manager.systems["ts"].get_player_rating(p)
                    if ts_r.mu > ROOKIE_MAJOR_TS_MU_CAP:
                        manager.systems["ts"].player_ratings[p] = trueskill.Rating(
                            mu=ROOKIE_MAJOR_TS_MU_CAP,
                            sigma=max(ts_r.sigma, ROOKIE_MAJOR_TS_SIGMA_FLOOR),
                        )
                    os_r = manager.systems["os"].get_player_rating(p)
                    if os_r.mu > ROOKIE_MAJOR_OS_MU_CAP:
                        manager.systems["os"].player_ratings[p] = (
                            manager.systems["os"].model.rating(
                                mu=ROOKIE_MAJOR_OS_MU_CAP,
                                sigma=max(os_r.sigma, ROOKIE_MAJOR_OS_SIGMA_FLOOR),
                            )
                        )
                    pl_r = manager.systems["pl"].get_player_rating(p)
                    if pl_r.mu > ROOKIE_MAJOR_PL_MU_CAP:
                        manager.systems["pl"].player_ratings[p] = (
                            manager.systems["pl"].pl_model.rating(
                                mu=ROOKIE_MAJOR_PL_MU_CAP,
                                sigma=max(pl_r.sigma, ROOKIE_MAJOR_PL_SIGMA_FLOOR),
                            )
                        )
                    tm_r = manager.systems["tm"].get_player_rating(p)
                    if tm_r.mu > ROOKIE_MAJOR_TM_MU_CAP:
                        manager.systems["tm"].player_ratings[p] = (
                            manager.systems["tm"].tm_model.rating(
                                mu=ROOKIE_MAJOR_TM_MU_CAP,
                                sigma=max(tm_r.sigma, ROOKIE_MAJOR_TM_SIGMA_FLOOR),
                            )
                        )
            for p in players_a + players_b:
                major_games_count[p] += len(match.scores)

        for score_a in match.scores:
            for system_name in RAW_SYSTEMS:
                system = manager.systems[system_name]
                system.update_team(
                    match.team_a_id,
                    match.team_b_id,
                    score_a,
                    1 - score_a,
                )
                system.update_player(
                    players_a,
                    players_b,
                    score_a,
                    1 - score_a,
                )
    return manager


def _systems_payload(
    *,
    engine: calibrated.FamilyCalibratedGlicko2,
    metadata: calibrated.RebuildMetadata,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": RATINGS_VERSION,
        "public_systems": list(PUBLIC_SYSTEMS),
        "raw_systems": {
            system: raw.RATING_SYSTEM_PARAMS[system] for system in RAW_SYSTEMS
        },
        "gl": {
            "engine": REGIONAL_ENGINE,
            "parameters": calibrated.SYSTEM_PARAMETERS,
            "state": engine.to_state(),
            "metadata": metadata.to_state(),
            "checkpoint": checkpoint,
        },
    }


def _validate_rows(rows: Sequence[tuple[Any, ...]]) -> None:
    systems = {str(row[8]) for row in rows}
    if systems != set(PUBLIC_SYSTEMS):
        raise RuntimeError(
            "ratings-v2 materialization has an invalid public system set: "
            f"{sorted(systems)!r}"
        )
    expected = Counter(
        (str(row[3]), str(row[5]))
        for row in rows
        if row[8] == "gl"
    )
    if not expected or any(count != 1 for count in expected.values()):
        raise RuntimeError("ratings-v2 regional Glicko cohort contains duplicate identities")
    for system in RAW_SYSTEMS:
        current = Counter(
            (str(row[3]), str(row[5]))
            for row in rows
            if row[8] == system
        )
        if current != expected:
            raise RuntimeError(
                f"ratings-v2 {system} rows do not exactly match the regional Glicko cohort"
            )


def rebuild_regional_ratings(
    *,
    version: str = RATINGS_VERSION,
    source: str = DEFAULT_SOURCE,
    until_date: str | None = None,
) -> dict[str, Any]:
    """Fully replay and atomically replace a regional multi-rating snapshot.

    Partial and incremental rebuilds are intentionally unsupported: every public
    rating system must have the same cohort and regional posterior.
    """
    if version != RATINGS_VERSION:
        raise ValueError(
            f"regional rebuild requires immutable version {RATINGS_VERSION!r}, got {version!r}"
        )
    if until_date is not None:
        date.fromisoformat(until_date)

    start = calibrated._start_run(version, source)
    try:
        matches = calibrated.load_matches(until_date=until_date)
        if not matches:
            raise ValueError("regional ratings rebuild found no eligible matches")
        engine = calibrated.FamilyCalibratedGlicko2()
        metadata = calibrated.RebuildMetadata()
        checkpoint = calibrated.process_matches(engine, metadata, matches)
        if checkpoint is None:
            raise RuntimeError("regional ratings rebuild did not produce a cutoff checkpoint")
        raw_manager = _replay_raw_systems(matches)
        cutoff = max(match.event_date for match in matches).isoformat()
        rows = calibrated.materialize_entity_rows(
            engine=engine,
            metadata=metadata,
            run_id=start.run_id,
            version=version,
            snapshot_at=cutoff,
            rating_system="gl",
            competition_calibration=REGIONAL_ENGINE,
        )
        rows.extend(
            _raw_rows(
                manager=raw_manager,
                engine=engine,
                metadata=metadata,
                run_id=start.run_id,
                version=version,
                snapshot_at=cutoff,
            )
        )
        _validate_rows(rows)
        calibrated._commit_snapshot(
            start=start,
            version=version,
            source=source,
            cutoff=cutoff,
            rows=rows,
            systems_json=json.dumps(
                _systems_payload(
                    engine=engine, metadata=metadata, checkpoint=checkpoint
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
            metadata=metadata,
            players_processed=len(tuple(engine.player_ids)),
        )
    except Exception as error:
        calibrated._record_failure(start, error)
        raise

    return {
        "version": version,
        "mode": "full",
        "matches": len(matches),
        "games": sum(len(match.scores) for match in matches),
        "players": len(tuple(engine.player_ids)),
        "entities": len({(str(row[3]), str(row[5])) for row in rows}),
        "rows": len(rows),
        "data_cutoff_at": cutoff,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", dest="until_date", help="Include complete dates through YYYY-MM-DD.")
    parser.add_argument("--ratings-version", default=RATINGS_VERSION)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    stats = rebuild_regional_ratings(
        version=args.ratings_version,
        source=args.source,
        until_date=args.until_date,
    )
    print(
        "Rebuilt regional ratings:",
        f"version={stats['version']}",
        f"matches={stats['matches']}",
        f"games={stats['games']}",
        f"players={stats['players']}",
        f"rows={stats['rows']}",
        f"cutoff={stats['data_cutoff_at']}",
    )


if __name__ == "__main__":
    main()
