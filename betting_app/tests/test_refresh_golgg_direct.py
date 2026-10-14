from datetime import datetime

from betting_app.core.db import query_df, transaction
from betting_app.scripts.refresh_golgg_direct import (
    auto_map_new_matches,
    result_mapping_review_reason,
    select_recent_existing_matches,
)




def test_select_recent_existing_matches_picks_only_known_recent_matches() -> None:
    matches = [
        {"match_id": "old", "date": "2026-05-01"},
        {"match_id": "recent", "date": "2026-07-04"},
        {"match_id": "new", "date": "2026-07-04"},
    ]

    selected = select_recent_existing_matches(
        matches,
        {"old", "recent"},
        days=45,
        now=datetime(2026, 7, 23),
    )
    assert [match["match_id"] for match in selected] == ["recent"]


def test_result_mapping_containment_requires_exact_date_and_competition() -> None:
    common = {
        "identity_confidence": 1.0,
        "canonical_date": "2026-09-03",
        "golgg_date": "2026-09-03",
        "canonical_competition": "TJ Sports LoL / LPL",
        "golgg_competition": "LPL 2026",
    }

    assert result_mapping_review_reason(**common) is None
    assert result_mapping_review_reason(
        **{**common, "canonical_date": "2026-09-04"}
    ) == "date_mismatch"
    assert result_mapping_review_reason(
        **{**common, "golgg_competition": "LCK 2026"}
    ) == "competition_mismatch"
    assert result_mapping_review_reason(
        **{**common, "identity_confidence": 0.949}
    ) == "identity_confidence"


def test_auto_map_routes_next_day_result_to_review(client) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO canonical_matches(
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b,
                start_time_normalized, league, status, match_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                10,
                "t1-geng-20260904",
                "T1",
                "Gen.G",
                "t1",
                "gen g",
                "2026-09-04T08:00:00+00:00",
                "LCK",
                "expired",
                1.0,
            ),
        )
        connection.execute(
            """
            INSERT INTO golgg_matches(
                match_id, date, tournament_name, team1_name, team2_name
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("100", "2026-09-03", "LCK 2026", "T1", "Gen.G"),
        )

    result = auto_map_new_matches(["100"], candidate_statuses=["expired"])

    assert result["mapped"] == 0
    assert result["review"] == 1
    assert result["review_reasons"] == {"no_exact_date_candidate": 1}
    assert query_df(
        "SELECT canonical_match_id FROM golgg_match_mappings"
    ).empty


def test_auto_map_accepts_one_exact_identity_date_competition_candidate(client) -> None:
    with transaction() as connection:
        connection.execute(
            """
            INSERT INTO canonical_matches(
                id, canonical_key, team_a_name, team_b_name,
                normalized_team_a, normalized_team_b,
                start_time_normalized, league, status, match_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                11,
                "nip-jd-20260905",
                "Ninjas in Pyjamas",
                "JD Gaming",
                "ninjas in pyjamas",
                "jd gaming",
                "2026-09-05T09:00:00+00:00",
                "TJ Sports LoL / LPL",
                "expired",
                1.0,
            ),
        )
        connection.execute(
            """
            INSERT INTO golgg_matches(
                match_id, date, tournament_name, team1_name, team2_name
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "101",
                "2026-09-05",
                "LPL 2026",
                "JD Gaming",
                "Ninjas in Pyjamas",
            ),
        )

    result = auto_map_new_matches(["101"], candidate_statuses=["expired"])
    mapping = query_df(
        """
        SELECT canonical_match_id, confidence, mapped_by
        FROM golgg_match_mappings
        WHERE golgg_match_id = ?
        """,
        ("101",),
    ).iloc[0]

    assert result["mapped"] == 1
    assert result["review"] == 0
    assert int(mapping["canonical_match_id"]) == 11
    assert float(mapping["confidence"]) == 1.0
    assert mapping["mapped_by"] == "auto-contained-v1"
