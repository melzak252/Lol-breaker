"""Tests for STS LoL proposition scraper."""

import pytest

from betting_app.scrapers.sts_props_scraper import STSPropsScraper, classify_sts_market


def test_classify_sts_market():
    # Total kills
    mtype, map_num = classify_sts_market("1. mapa - suma zabójstw (26.5)")
    assert mtype == "total_kills"
    assert map_num == 1

    # Map 2 total kills
    mtype, map_num = classify_sts_market("2. mapa - suma zabójstw (28.5)")
    assert mtype == "total_kills"
    assert map_num == 2

    # Kill handicap
    mtype, map_num = classify_sts_market("1. mapa - handicap zabójstw (-4.5)")
    assert mtype == "handicap_kills"
    assert map_num == 1

    # Duration
    mtype, map_num = classify_sts_market("1. mapa - czas trwania meczu w minutach")
    assert mtype == "duration"
    assert map_num == 1

    # First objectives
    assert classify_sts_market("1. mapa - 1. krew")[0] == "first_blood"
    assert classify_sts_market("1. mapa - 1. smok")[0] == "first_dragon"
    assert classify_sts_market("1. mapa - 1. baron")[0] == "first_baron"
    assert classify_sts_market("1. mapa - 1. wieża")[0] == "first_tower"


def test_sts_props_scraper_parse_fixture():
    scraper = STSPropsScraper()

    # Realistic STS SBK fixture payload
    fixture_data = {
        "H": {"N": "Fnatic"},
        "A": {"N": "G2 Esports"},
        "a": {"offer_1": True},
    }
    offers_data = {
        "offer_1": {
            "m": {
                "m1": {
                    "l": {
                        "l1": {
                            "n": "1. mapa - suma zabójstw (27.5)",
                            "o": {
                                "o1": {"n": "Powyżej 27.5", "O": "1.85"},
                                "o2": {"n": "Poniżej 27.5", "O": "1.95"},
                            },
                        }
                    }
                },
                "m2": {
                    "l": {
                        "l2": {
                            "n": "1. mapa - 1. krew",
                            "o": {
                                "o3": {"n": "Fnatic", "O": "1.90"},
                                "o4": {"n": "G2 Esports", "O": "1.85"},
                            },
                        }
                    }
                },
                "m3": {
                    "l": {
                        "l3": {
                            "n": "1. mapa - czas trwania w minutach (32.5)",
                            "o": {
                                "o5": {"n": "Więcej niż 32.5", "O": "2.10"},
                                "o6": {"n": "Mniej niż 32.5", "O": "1.68"},
                            },
                        }
                    }
                },
            }
        }
    }

    parsed = scraper.parse_event_props(
        payload={"T": {"t1": {"FX": {"f1": fixture_data}}}, "O": offers_data}
    )

    assert len(parsed) == 1
    m_props = parsed[0]
    assert m_props.bookmaker == "sts"
    assert m_props.raw_team_a == "Fnatic"
    assert m_props.raw_team_b == "G2 Esports"
    assert m_props.map_number == 1

    # Check total kills
    tk_lines = m_props.get_total_kills_lines()
    assert len(tk_lines) == 1
    assert tk_lines[0].line == 27.5
    assert tk_lines[0].odds_over == 1.85
    assert tk_lines[0].odds_under == 1.95
    assert tk_lines[0].novig_prob_over > 0.5

    # Check first blood
    fb_lines = m_props.get_first_blood_lines()
    assert len(fb_lines) == 1
    assert fb_lines[0].odds_cover_a == 1.90
    assert fb_lines[0].odds_cover_b == 1.85

    # Check duration
    dur_lines = m_props.get_duration_lines()
    assert len(dur_lines) == 1
    assert dur_lines[0].line == 32.5
    assert dur_lines[0].odds_over == 2.10
    assert dur_lines[0].odds_under == 1.68
