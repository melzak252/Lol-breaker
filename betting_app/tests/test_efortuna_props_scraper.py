"""Tests for eFortuna LoL proposition scraper."""

import pytest

from betting_app.scrapers.efortuna_props_scraper import (
    EFortunaPropsScraper,
    classify_efortuna_market,
)


def test_classify_efortuna_market():
    mtype, map_num = classify_efortuna_market("1. mapa - liczba zabójstw (26.5)")
    assert mtype == "total_kills"
    assert map_num == 1

    mtype, map_num = classify_efortuna_market("2. mapa - handicap zabójstw (-4.5)")
    assert mtype == "handicap_kills"
    assert map_num == 2

    mtype, map_num = classify_efortuna_market("1. mapa - czas trwania meczu (31.5)")
    assert mtype == "duration"
    assert map_num == 1

    assert classify_efortuna_market("1. mapa - 1. zabójstwo")[0] == "first_blood"
    assert classify_efortuna_market("1. mapa - 1. smok")[0] == "first_dragon"
    assert classify_efortuna_market("1. mapa - 1. baron")[0] == "first_baron"


def test_efortuna_props_scraper_parse():
    scraper = EFortunaPropsScraper()

    payload = {
        "team_a": "Karmine Corp",
        "team_b": "Team Vitality",
        "tables": [
            {
                "name": "1. mapa - liczba zabójstw (27.5)",
                "bets": [
                    {"name": "+ 27.5", "odds": "1.82"},
                    {"name": "- 27.5", "odds": "1.98"},
                ],
            },
            {
                "name": "1. mapa - 1. zabójstwo",
                "bets": [
                    {"name": "Karmine Corp", "odds": "1.92"},
                    {"name": "Team Vitality", "odds": "1.88"},
                ],
            },
            {
                "name": "2. mapa - czas trwania w minutach (32.5)",
                "bets": [
                    {"name": "Więcej niż 32.5", "odds": "2.05"},
                    {"name": "Mniej niż 32.5", "odds": "1.70"},
                ],
            },
        ],
    }

    results = scraper.parse_event_props(payload)
    assert len(results) == 2  # Map 1 and Map 2

    # Map 1
    m1 = next(r for r in results if r.map_number == 1)
    assert m1.bookmaker == "efortuna"
    assert m1.raw_team_a == "Karmine Corp"
    assert m1.raw_team_b == "Team Vitality"

    tk = m1.get_total_kills_lines()
    assert len(tk) == 1
    assert tk[0].line == 27.5
    assert tk[0].odds_over == 1.82
    assert tk[0].odds_under == 1.98
    assert tk[0].novig_prob_over > 0.5

    fb = m1.get_first_blood_lines()
    assert len(fb) == 1
    assert fb[0].odds_cover_a == 1.92
    assert fb[0].odds_cover_b == 1.88

    # Map 2
    m2 = next(r for r in results if r.map_number == 2)
    dur = m2.get_duration_lines()
    assert len(dur) == 1
    assert dur[0].line == 32.5
    assert dur[0].odds_over == 2.05
