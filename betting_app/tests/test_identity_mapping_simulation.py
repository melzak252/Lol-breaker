from __future__ import annotations

import pandas as pd

from scripts import analyze_identity_mapping as simulation


def _empty_aliases() -> pd.DataFrame:
    return pd.DataFrame(columns=["id", "normalized_name"])


def test_competition_family_normalizes_collected_source_labels() -> None:
    assert simulation.competition_family("Riot LoL / LCK") == "lck"
    assert simulation.competition_family("KeSPA LoL / LCK CL") == "lck_cl"
    assert simulation.competition_family("CD 2026 Split 2") == "circuito_desafiante"
    assert simulation.competition_family("Esports Wolrd Cup SA & LATAM CQ") == "ewc"
    assert simulation.competition_family("Lastlap / LES") == "les"


def test_conflicting_equal_scope_aliases_are_order_independent_and_ambiguous() -> None:
    rows = pd.DataFrame(
        [
            {
                "id": 1,
                "normalized_name": "keyd stars",
                "alias": "Keyd Stars",
                "source": "manual-a",
                "source_system": None,
                "league_pattern": None,
                "tournament_pattern": None,
                "valid_from": None,
                "valid_to": None,
                "is_active": 1,
                "is_blocked": 0,
            },
            {
                "id": 2,
                "normalized_name": "keyd stars",
                "alias": "Vivo Keyd Stars",
                "source": "manual-b",
                "source_system": None,
                "league_pattern": None,
                "tournament_pattern": None,
                "valid_from": None,
                "valid_to": None,
                "is_active": 1,
                "is_blocked": 0,
            },
        ]
    )

    outcomes = []
    for aliases in (rows, rows.iloc[::-1].reset_index(drop=True)):
        resolution = simulation.SnapshotAliasResolver(aliases).resolve(
            "Keyd Stars",
            source_system="bookmaker",
            competition="CBLOL",
            tournament=None,
            on_date="2026-09-03",
        )
        outcomes.append((resolution.status, resolution.reason))

    assert outcomes == [
        ("ambiguous", "conflicting_applicable_targets"),
        ("ambiguous", "conflicting_applicable_targets"),
    ]


def test_nip_alias_is_lpl_scoped() -> None:
    resolver = simulation.SnapshotAliasResolver(_empty_aliases())

    lpl = resolver.resolve(
        "NiP",
        source_system="bookmaker",
        competition="LPL",
        tournament=None,
        on_date="2026-09-05",
    )
    nlc = resolver.resolve(
        "NiP",
        source_system="bookmaker",
        competition="NLC",
        tournament=None,
        on_date="2026-09-05",
    )

    assert lpl.key == simulation.semantic_key("Ninjas in Pyjamas")
    assert lpl.source == "proposed_scoped"
    assert nlc.key != lpl.key


def test_exact_date_gate_is_safer_than_one_day_gate() -> None:
    rows = pd.DataFrame(
        [
            {
                "canonical_match_id": 10,
                "confidence": 1.0,
                "date_distance_days": 1,
                "competition_compatible": True,
                "canonical_competition_family": "lpl",
                "golgg_competition_family": "lpl",
            }
        ]
    )

    result = simulation.simulate_containment(rows).iloc[0]

    assert bool(result["containment_one_day_accept"]) is True
    assert bool(result["containment_exact_date_accept"]) is False
    assert bool(result["competition_accept"]) is False


def test_identity_replay_does_not_choose_a_next_day_candidate() -> None:
    frames = {
        "canonical": pd.DataFrame(
            [
                {
                    "id": 20,
                    "team_a_name": "T1",
                    "team_b_name": "Gen.G",
                    "start_time_normalized": "2026-09-04T08:00:00+00:00",
                    "league": "LCK",
                }
            ]
        ),
        "golgg": pd.DataFrame(
            [
                {
                    "match_id": 100,
                    "date": "2026-09-03",
                    "tournament_name": "LCK 2026",
                    "team1_name": "T1",
                    "team2_name": "Gen.G",
                }
            ]
        ),
        "mappings": pd.DataFrame(
            [
                {
                    "id": 1,
                    "canonical_match_id": 20,
                    "golgg_match_id": 100,
                }
            ]
        ),
    }

    decisions, summary = simulation.simulate_identity_replay(
        frames,
        simulation.SnapshotAliasResolver(_empty_aliases()),
    )

    assert decisions.iloc[0]["outcome"] == "review_no_candidate"
    assert summary.auto_accept == 0


def test_upcoming_duplicate_replay_finds_reversed_nip_jd_fixture() -> None:
    frames = {
        "canonical": pd.DataFrame(
            [
                {
                    "id": 145826,
                    "team_a_name": "Ninjas in Pyjamas",
                    "team_b_name": "JD Gaming",
                    "start_time_normalized": "2026-09-05T09:00:00+00:00",
                    "league": "LPL",
                    "status": "upcoming",
                },
                {
                    "id": 145830,
                    "team_a_name": "JD",
                    "team_b_name": "NiP",
                    "start_time_normalized": "2026-09-05T09:00:00Z",
                    "league": "LPL",
                    "status": "upcoming",
                },
            ]
        )
    }

    result = simulation.simulate_upcoming_duplicates(
        frames,
        simulation.SnapshotAliasResolver(_empty_aliases()),
    )

    assert result[["left_canonical_match_id", "right_canonical_match_id"]].to_dict("records") == [
        {"left_canonical_match_id": 145826, "right_canonical_match_id": 145830}
    ]
    assert result.iloc[0]["start_difference_minutes"] == 0.0
