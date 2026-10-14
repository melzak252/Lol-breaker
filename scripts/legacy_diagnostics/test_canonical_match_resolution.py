import pytest
from betting_app.services.canonical_match_service import (
    canonical_match_score,
    infer_best_of,
    normalize_start_time,
)
from betting_app.scrapers.efortuna_parser import _infer_best_from_map_line




def _candidate(status: str, start_time: str) -> dict[str, str]:
    return {
        "normalized_team_a": "bilibili gaming",
        "normalized_team_b": "hanwha life esports",
        "start_time_normalized": start_time,
        "league": "MSI",
        "status": status,
    }


def test_identical_expired_match_with_different_start_time_is_not_boosted() -> None:
    score = canonical_match_score(
        "bilibili gaming",
        "hanwha life esports",
        "2026-07-12T07:30:00+00:00",
        "MSI",
        _candidate("expired", "2026-07-11T08:00:00+00:00"),
    )

    assert score < 0.78


def test_identical_upcoming_match_keeps_unstable_label_boost() -> None:
    score = canonical_match_score(
        "bilibili gaming",
        "hanwha life esports",
        "2026-07-12T07:30:00+00:00",
        "MSI",
        _candidate("upcoming", "2026-07-12T08:00:00+00:00"),
    )

    assert score >= 0.85


def test_identical_upcoming_match_days_apart_is_not_boosted() -> None:
    score = canonical_match_score(
        "bilibili gaming",
        "hanwha life esports",
        "2026-07-15T07:30:00+00:00",
        "MSI",
        _candidate("upcoming", "2026-07-12T07:30:00+00:00"),
    )

    assert score < 0.78


def test_exact_team_names_do_not_merge_across_competition_families() -> None:
    score = canonical_match_score(
        "t1",
        "gen g",
        "2026-09-05T09:00:00+00:00",
        "LCK",
        {
            "normalized_team_a": "t1",
            "normalized_team_b": "gen g",
            "start_time_normalized": "2026-09-05T09:00:00+00:00",
            "league": "LCK Challengers",
            "status": "upcoming",
        },
    )

    assert score == 0.0


def test_polish_month_label_without_year_is_normalized() -> None:
    from datetime import datetime, timezone
    clock = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    assert normalize_start_time("Śr 15. Lip, 11:20", now_utc=clock) == "2026-07-15T11:20:00+00:00"


@pytest.mark.parametrize(
    ("league", "expected_best_of"),
    [
        ("LCK", 3),
        ("LPL", 3),
        ("LEC", 3),
        ("LPLOL", 1),
        ("Inygon / LPLOL", 1),
        ("NACL", 1),
        ("NA Challengers League", 1),
        ("LCK Playoffs", 5),
        ("LEC Finals", 5),
        ("LPL Bracket", 5),
        ("LCK Road to MSI", 5),
        ("Regional Qualifiers", 5),
        ("Unknown Tournament", 1),
        (None, 1),
    ],
)
def test_infer_best_of_formats(league: str | None, expected_best_of: int) -> None:
    assert infer_best_of(league) == expected_best_of


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("2.5", 3),
        ("-2.5", 3),
        ("- 2.5", 3),
        ("+ 2.5", 3),
        ("4.5", 5),
        ("- 4.5", 5),
        ("1.5", 1),
        ("- 1.5", 1),
        ("invalid", None),
    ],
)
def test_efortuna_map_totals_parsing_with_whitespace(line: str, expected: int | None) -> None:
    assert _infer_best_from_map_line(line) == expected
