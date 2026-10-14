"""Tests for Betclic LoL proposition scraper."""

import pytest

from betting_app.scrapers.betclic_props_scraper import (
    BetclicPropsScraper,
    classify_betclic_market,
)


def test_classify_betclic_market():
    mtype, map_num = classify_betclic_market("Liczba zabójstw na 1. mapie (26.5)")
    assert mtype == "total_kills"
    assert map_num == 1

    mtype, map_num = classify_betclic_market("Handicap zabójstw na mapie 2 (-4.5)")
    assert mtype == "handicap_kills"
    assert map_num == 2

    mtype, map_num = classify_betclic_market("Czas trwania 1. mapy (w minutach)")
    assert mtype == "duration"
    assert map_num == 1

    assert classify_betclic_market("1. mapa - Pierwsza krew")[0] == "first_blood"
    assert classify_betclic_market("1. mapa - Pierwszy smok")[0] == "first_dragon"
    assert classify_betclic_market("1. mapa - Pierwszy baron")[0] == "first_baron"


def test_betclic_props_scraper_parse():
    scraper = BetclicPropsScraper()

    payload = {
        "team_a": "G2 Esports",
        "team_b": "Movistar KOI",
        "markets": [
            {
                "name": "Liczba zabójstw na 1. mapie (28.5)",
                "selections": [
                    {"name": "+ 28.5", "price": "1.80"},
                    {"name": "- 28.5", "price": "1.92"},
                ],
            },
            {
                "name": "1. mapa - Pierwszy smok",
                "selections": [
                    {"name": "G2 Esports", "price": "1.75"},
                    {"name": "Movistar KOI", "price": "1.98"},
                ],
            },
            {
                "name": "Czas gry na 2. mapie (31.5)",
                "selections": [
                    {"name": "Powyżej 31.5", "price": "1.95"},
                    {"name": "Poniżej 31.5", "price": "1.78"},
                ],
            },
        ],
    }

    results = scraper.parse_event_props(payload)
    assert len(results) == 2

    # Map 1
    m1 = next(r for r in results if r.map_number == 1)
    assert m1.bookmaker == "betclic"
    assert m1.raw_team_a == "G2 Esports"
    assert m1.raw_team_b == "Movistar KOI"

    tk = m1.get_total_kills_lines()
    assert len(tk) == 1
    assert tk[0].line == 28.5
    assert tk[0].odds_over == 1.80
    assert tk[0].odds_under == 1.92
    assert tk[0].novig_prob_over > 0.5

    fd = m1.get_first_dragon_lines()
    assert len(fd) == 1
    assert fd[0].odds_cover_a == 1.75
    assert fd[0].odds_cover_b == 1.98

    # Map 2
    m2 = next(r for r in results if r.map_number == 2)
    dur = m2.get_duration_lines()
    assert len(dur) == 1
    assert dur[0].line == 31.5
    assert dur[0].odds_over == 1.95
