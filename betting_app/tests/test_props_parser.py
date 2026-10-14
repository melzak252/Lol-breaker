"""Unit tests for props_parser (STS, Betclic, Superbet) using hermetic fixtures (IDEA-018)."""

from __future__ import annotations

import math
import pytest

from betting_app.scrapers.props_parser import (
    compute_novig_two_way,
    extract_line_number,
    parse_betclic_map_props,
    parse_sts_map_props,
    parse_superbet_map_props,
)


def test_compute_novig_two_way() -> None:
    # 1.85 / 1.85 typical line with margin
    p1, p2, margin = compute_novig_two_way(1.85, 1.85)
    assert math.isclose(p1, 0.50, abs_tol=1e-3)
    assert math.isclose(p2, 0.50, abs_tol=1e-3)
    assert math.isclose(p1 + p2, 1.0, abs_tol=1e-3)
    # Margin = 1/1.85 + 1/1.85 - 1 = 2/1.85 - 1 = 1.0811 - 1 = ~0.0811 (8.11%)
    assert math.isclose(margin, 0.0811, abs_tol=1e-3)

    # Asymmetric line: 1.50 vs 2.60
    p1_asym, p2_asym, margin_asym = compute_novig_two_way(1.50, 2.60)
    assert p1_asym > p2_asym
    assert math.isclose(p1_asym + p2_asym, 1.0, abs_tol=1e-3)


def test_extract_line_number() -> None:
    assert extract_line_number("1. mapa - suma zabójstw (26.5)") == 26.5
    assert extract_line_number("Powyżej 28,5") == 28.5
    assert extract_line_number("Handicap -4.5") == -4.5
    assert extract_line_number("+6.5 killi") == 6.5
    assert extract_line_number("Bez numeru") is None


def test_parse_sts_map_props() -> None:
    mock_sts_payload = {
        "team_a": "G2 Esports",
        "team_b": "Fnatic",
        "map_number": 1,
        "markets": [
            {
                "name": "1. mapa - suma zabójstw (28.5)",
                "outcomes": [
                    {"name": "Powyżej 28.5", "odds": 1.90},
                    {"name": "Poniżej 28.5", "odds": 1.80},
                ],
            },
            {
                "name": "1. mapa - liczba zabójstw drużyny 1 (15.5)",
                "outcomes": [
                    {"name": "Powyżej 15.5", "odds": 1.85},
                    {"name": "Poniżej 15.5", "odds": 1.85},
                ],
            },
            {
                "name": "1. mapa - handicap zabójstw (-4.5)",
                "outcomes": [
                    {"name": "G2 Esports (-4.5)", "odds": 2.10},
                    {"name": "Fnatic (+4.5)", "odds": 1.65},
                ],
            },
            {
                "name": "1. mapa - czas trwania (32.5)",
                "outcomes": [
                    {"name": "Powyżej 32.5", "odds": 1.95},
                    {"name": "Poniżej 32.5", "odds": 1.75},
                ],
            },
        ],
    }

    result = parse_sts_map_props(mock_sts_payload)
    assert result.bookmaker == "STS"
    assert result.raw_team_a == "G2 Esports"
    assert result.raw_team_b == "Fnatic"
    assert len(result.lines) == 4

    # Total kills check
    totals = result.get_total_kills_lines()
    assert len(totals) == 1
    assert totals[0].line == 28.5
    assert totals[0].odds_over == 1.90
    assert totals[0].odds_under == 1.80
    assert totals[0].novig_prob_over is not None

    # Team kills check
    team_lines = result.get_team_kills_lines("G2 Esports")
    assert len(team_lines) == 1
    assert team_lines[0].line == 15.5
    assert team_lines[0].team == "G2 Esports"

    # Handicap check
    hc_lines = result.get_handicap_lines()
    assert len(hc_lines) == 1
    assert hc_lines[0].line == -4.5
    assert hc_lines[0].odds_cover_a == 2.10
    assert hc_lines[0].odds_cover_b == 1.65

    # Duration check
    dur_lines = result.get_duration_lines()
    assert len(dur_lines) == 1
    assert dur_lines[0].line == 32.5


def test_parse_betclic_map_props() -> None:
    mock_betclic_payload = {
        "raw_team_a": "T1",
        "raw_team_b": "Gen.G",
        "map_number": 1,
        "market_list": [
            {
                "title": "Łączna liczba zabójstw",
                "line": 26.5,
                "choices": [
                    {"name": "Więcej niż 26.5", "odds": 1.78},
                    {"name": "Mniej niż 26.5", "odds": 1.92},
                ],
            },
            {
                "title": "Handicap zabójstw (z dogrywką)",
                "line": -2.5,
                "choices": [
                    {"name": "T1 (-2.5)", "odds": 1.90},
                    {"name": "Gen.G (+2.5)", "odds": 1.80},
                ],
            },
        ],
    }

    result = parse_betclic_map_props(mock_betclic_payload)
    assert result.bookmaker == "Betclic"
    assert len(result.lines) == 2

    totals = result.get_total_kills_lines()
    assert len(totals) == 1
    assert totals[0].line == 26.5
    assert totals[0].odds_over == 1.78
    assert totals[0].odds_under == 1.92

    hc = result.get_handicap_lines()
    assert len(hc) == 1
    assert hc[0].line == -2.5
    assert hc[0].odds_cover_a == 1.90


def test_parse_superbet_map_props() -> None:
    mock_superbet_payload = {
        "match_name_a": "Bilibili Gaming",
        "match_name_b": "Top Esports",
        "map_index": 1,
        "events": [
            {
                "name": "Suma zabójstw - Mapa 1",
                "line": 29.5,
                "selections": [
                    {"name": "Powyżej 29.5", "price": 1.85},
                    {"name": "Poniżej 29.5", "price": 1.85},
                ],
            }
        ],
    }

    result = parse_superbet_map_props(mock_superbet_payload)
    assert result.bookmaker == "Superbet"
    assert len(result.lines) == 1
    line = result.lines[0]
    assert line.market_type == "total_kills"
    assert line.line == 29.5
    assert line.odds_over == 1.85
    assert line.odds_under == 1.85
    assert math.isclose(line.novig_prob_over, 0.50, abs_tol=1e-2)
