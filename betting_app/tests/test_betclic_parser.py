from betting_app.scrapers.betclic_parser import parse_event_link


def test_parse_event_link_does_not_fold_team_digit_into_t1_odds():
    offer = parse_event_link(
        "LCK +51 zakł. T1 10:00 - KT Rolster T11,25KT Rolster3,40",
        "https://www.betclic.pl/league-of-legends-slol/lck-c23480/t1-kt-rolster-m1166394890924032",
    )

    assert offer is not None
    assert offer.raw_team_a == "T1"
    assert offer.raw_team_b == "KT Rolster"
    assert offer.odds_a == 1.25
    assert offer.odds_b == 3.40


def test_parse_event_link_keeps_t1_in_gen_g_offer():
    offer = parse_event_link(
        "LCK +51 zakł. Gen.G 08:00 - T1 Gen.G1,62T12,10",
        "https://www.betclic.pl/league-of-legends-slol/lck-c23480/gen-g-t1-m1166394889637888",
    )

    assert offer is not None
    assert offer.raw_team_a == "Gen.G"
    assert offer.raw_team_b == "T1"
    assert offer.odds_a == 1.62
    assert offer.odds_b == 2.10


def test_parse_event_link_does_not_fold_team_digit_into_cloud9_odds():
    offer = parse_event_link(
        "LCS +51 zakł. Team Liquid 20:00 - Cloud9 Team Liquid1,70Cloud91,98",
        "https://www.betclic.pl/league-of-legends-slol/lcs-c25466/team-liquid-cloud9-m1167117933359210",
    )

    assert offer is not None
    assert offer.raw_team_a == "Team Liquid"
    assert offer.raw_team_b == "Cloud9"
    assert offer.odds_a == 1.70
    assert offer.odds_b == 1.98


def test_parse_event_link_keeps_legitimate_double_digit_odd():
    offer = parse_event_link(
        "LCK +51 zakł. Gen.G 10:00 - Underdog Gen.G11,25Underdog1,02",
        "https://www.betclic.pl/league-of-legends-slol/lck-c23480/gen-g-underdog-m1",
    )

    assert offer is not None
    assert offer.raw_team_a == "Gen.G"
    assert offer.raw_team_b == "Underdog"
    assert offer.odds_a == 11.25
    assert offer.odds_b == 1.02
