"""Tests for Superbet LoL proposition scraper."""

import pytest

from betting_app.scrapers.superbet_props_scraper import (
    SuperbetPropsScraper,
    classify_superbet_market,
)


def test_classify_superbet_market():
    mtype, map_num = classify_superbet_market("Liczba zabójstw na mapie 1 (27.5)")
    assert mtype == "total_kills"
    assert map_num == 1

    mtype, map_num = classify_superbet_market("Handicap zabójstw na mapie 2 (-4.5)")
    assert mtype == "handicap_kills"
    assert map_num == 2

    mtype, map_num = classify_superbet_market("Czas trwania mapy 1 (32.5 minuty)")
    assert mtype == "duration"
    assert map_num == 1

    assert classify_superbet_market("Mapa 1 - 1. krew")[0] == "first_blood"
    assert classify_superbet_market("Mapa 1 - 1. smok")[0] == "first_dragon"
    assert classify_superbet_market("Mapa 1 - 1. baron")[0] == "first_baron"
    assert classify_superbet_market("Mapa 1 - 1. wieża")[0] == "first_tower"


def test_superbet_props_scraper_parse():
    scraper = SuperbetPropsScraper()

    payload = {
        "matchName": "Team BDS · MAD Lions KOI",
        "markets": [
            {
                "marketName": "Liczba zabójstw na mapie 1 (26.5)",
                "choices": [
                    {"choiceName": "+ 26.5", "price": "1.78"},
                    {"choiceName": "- 26.5", "price": "1.95"},
                ],
            },
            {
                "marketName": "Mapa 1 - 1. wieża",
                "choices": [
                    {"choiceName": "Team BDS", "price": "1.82"},
                    {"choiceName": "MAD Lions KOI", "price": "1.92"},
                ],
            },
            {
                "marketName": "Czas trwania mapy 2 (30.5)",
                "choices": [
                    {"choiceName": "Powyżej 30.5", "price": "1.88"},
                    {"choiceName": "Poniżej 30.5", "price": "1.85"},
                ],
            },
        ],
    }

    results = scraper.parse_event_props(payload)
    assert len(results) == 2

    # Map 1
    m1 = next(r for r in results if r.map_number == 1)
    assert m1.bookmaker == "superbet"
    assert m1.raw_team_a == "Team BDS"
    assert m1.raw_team_b == "MAD Lions KOI"

    tk = m1.get_total_kills_lines()
    assert len(tk) == 1
    assert tk[0].line == 26.5
    assert tk[0].odds_over == 1.78
    assert tk[0].odds_under == 1.95

    ft = m1.get_first_tower_lines()
    assert len(ft) == 1
    assert ft[0].odds_cover_a == 1.82
    assert ft[0].odds_cover_b == 1.92

    # Map 2
    m2 = next(r for r in results if r.map_number == 2)
    dur = m2.get_duration_lines()
    assert len(dur) == 1
    assert dur[0].line == 30.5
    assert dur[0].odds_over == 1.88
