from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from src.models.competition_tiers import (
    CompetitionIdentity,
    CompetitionScope,
    CompetitionTier,
    classify_competition,
)


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("LCK 2025 Rounds 1-2", "LCK"),
        ("LPL Summer Playoffs 2024", "LPL"),
        ("LEC 2026 Versus Season", "LEC"),
        ("LCS Championship 2024", "LCS"),
        ("LTA North 2025 Split 2", "LTA"),
        ("LTA South 2025 Split 3", "LTA"),
    ],
)
def test_current_major_leagues(name: str, family: str) -> None:
    identity = classify_competition(name)

    assert identity.family == family
    assert identity.tier is CompetitionTier.MAJOR
    assert identity.scope is CompetitionScope.DOMESTIC
    assert identity.matched_rule.startswith("major.")


@pytest.mark.parametrize(
    ("name", "family", "matched_rule"),
    [
        ("EU LCS Summer 2018", "LEC", "major.lec_historical_alias"),
        ("European Championship Spring 2020", "LEC", "major.lec_historical_alias"),
        ("NA LCS Spring 2018", "LCS", "major.lcs_historical_alias"),
        ("Champions Korea Spring 2015", "LCK", "major.lck_historical_alias"),
        ("OGN Champions Spring 2014", "LCK", "major.lck_historical_alias"),
    ],
)
def test_historical_major_aliases_have_canonical_families(
    name: str,
    family: str,
    matched_rule: str,
) -> None:
    assert classify_competition(name) == CompetitionIdentity(
        family=family,
        tier=CompetitionTier.MAJOR,
        scope=CompetitionScope.DOMESTIC,
        matched_rule=matched_rule,
    )


@pytest.mark.parametrize(
    ("name", "family", "matched_rule"),
    [
        ("World Championship Play-In 2022", "Worlds", "international.worlds"),
        ("Worlds 2025 Main Event", "Worlds", "international.worlds"),
        ("2025 Mid-Season Invitational", "MSI", "international.msi"),
        ("MSI 2024", "MSI", "international.msi"),
        ("First Stand 2025", "First Stand", "international.first_stand"),
        ("2026 First Stand", "First Stand", "international.first_stand"),
        (
            "Esports World Cup 2024",
            "Esports World Cup",
            "international.esports_world_cup",
        ),
    ],
)
def test_global_events_are_cross_league(
    name: str,
    family: str,
    matched_rule: str,
) -> None:
    assert classify_competition(name) == CompetitionIdentity(
        family=family,
        tier=CompetitionTier.INTERNATIONAL,
        scope=CompetitionScope.CROSS_LEAGUE,
        matched_rule=matched_rule,
    )


@pytest.mark.parametrize(
    ("name", "family"),
    [
        ("LCP 2025 Mid Season", "LCP"),
        ("PCS Summer Playoffs 2024", "PCS"),
        ("VCS 2026 Spring", "VCS"),
        ("CBLOL Split 2 2024", "CBLOL"),
        ("LJL 2025 Ignite", "LJL"),
        ("LLA Closing 2024", "LLA"),
        ("LCO Split 1 2024", "LCO"),
    ],
)
def test_minor_top_level_leagues(name: str, family: str) -> None:
    identity = classify_competition(name)

    assert identity.family == family
    assert identity.tier is CompetitionTier.MINOR_TOP_LEVEL
    assert identity.scope is CompetitionScope.DOMESTIC
    assert identity.matched_rule == f"minor_top_level.{family.casefold()}"


@pytest.mark.parametrize(
    ("name", "family", "matched_rule"),
    [
        ("LFL 2025 Spring Split", "LFL", "regional.lfl"),
        ("Prime League 1st Div 2025 Spring Split", "Prime League", "regional.prime_league"),
        ("Ultraliga Summer Playoffs 2024", "Ultraliga", "regional.ultraliga"),
        ("EBL 2025 Season Finals", "EBL", "regional.ebl"),
        ("LVP Summer Playoffs 2021", "SuperLiga", "regional.superliga_historical_lvp"),
        ("LVP UKLC Spring 2019", "NLC", "regional.nlc_historical_uklc"),
        ("MCR LoL 2019", "Hitpoint Masters", "regional.hitpoint_historical_mcr"),
        ("LES 2026 Final Four", "LES", "regional.les"),
        ("REL Season 2", "REL", "regional.romanian_esports_league"),
        (
            "Trinity Force Puchar Polski",
            "Ultraliga",
            "regional.ultraliga_historical_polish_cup",
        ),
    ],
)
def test_emea_regional_leagues(
    name: str,
    family: str,
    matched_rule: str,
) -> None:
    assert classify_competition(name) == CompetitionIdentity(
        family=family,
        tier=CompetitionTier.REGIONAL,
        scope=CompetitionScope.DOMESTIC,
        matched_rule=matched_rule,
    )


def test_emea_masters_is_regional_but_cross_league() -> None:
    assert classify_competition("EMEA Masters 2026 Spring Main Event") == CompetitionIdentity(
        family="EMEA Masters",
        tier=CompetitionTier.REGIONAL,
        scope=CompetitionScope.CROSS_LEAGUE,
        matched_rule="cross_league.emea_masters",
    )
    assert classify_competition("EU Masters Summer 2022").scope is CompetitionScope.CROSS_LEAGUE
    iberian = classify_competition("Iberian Cup 2020")
    assert iberian.family == "Iberian Cup"
    assert iberian.tier is CompetitionTier.REGIONAL
    assert iberian.scope is CompetitionScope.CROSS_LEAGUE


@pytest.mark.parametrize(
    ("name", "family", "matched_rule"),
    [
        ("LCK CL Spring Playoffs 2024", "LCK CL", "development.lck_cl"),
        ("LCK Challengers League 2025", "LCK CL", "development.lck_cl"),
        ("LDL Summer 2024", "LDL", "development.ldl"),
        ("NACL 2026 Spring", "NACL", "development.nacl"),
        ("NA Academy Summer 2022", "NA Academy", "development.na_academy"),
        ("CBLOL Academy Split 2 2024", "CBLOL Academy", "development.cblol_academy"),
        ("LFL Div2 2025 Winter", "LFL Div2", "development.lfl_div2"),
        ("Challenge France 2017", "LFL Div2", "development.lfl_div2"),
        ("CK Summer 2020", "LCK CL", "development.lck_cl_historical_ck"),
        (
            "EU CS Spring 2017",
            "EU Challenger Series",
            "development.eu_challenger_series",
        ),
        (
            "NA CS Summer 2017",
            "NA Challenger Series",
            "development.na_challenger_series",
        ),
        (
            "Hitpoint Challengers 2021 Spring",
            "Hitpoint Challengers",
            "development.hitpoint_challengers",
        ),
        (
            "Prime League 2nd Div Summer 2024",
            "Prime League Second Division",
            "development.prime_league_second_division",
        ),
        (
            "LVP SL 2nd Div Spring 2024",
            "SuperLiga Second Division",
            "development.superliga_second_division",
        ),
        ("TCL Div2 Winter 2024", "TCL Div2", "development.tcl_div2"),
    ],
)
def test_development_rules_precede_broad_parent_leagues(
    name: str,
    family: str,
    matched_rule: str,
) -> None:
    assert classify_competition(name) == CompetitionIdentity(
        family=family,
        tier=CompetitionTier.DEVELOPMENT,
        scope=CompetitionScope.DOMESTIC,
        matched_rule=matched_rule,
    )


def test_domestic_road_to_msi_is_not_mistaken_for_msi() -> None:
    assert classify_competition("LCK 2025 Road to MSI") == CompetitionIdentity(
        family="LCK",
        tier=CompetitionTier.MAJOR,
        scope=CompetitionScope.DOMESTIC,
        matched_rule="major.lck",
    )


def test_tcl_tier_transition_is_date_aware() -> None:
    historical = classify_competition("TCL Summer", date(2022, 7, 1))
    regional = classify_competition("TCL Spring", date(2024, 3, 1))

    assert historical == CompetitionIdentity(
        family="TCL",
        tier=CompetitionTier.MINOR_TOP_LEVEL,
        scope=CompetitionScope.DOMESTIC,
        matched_rule="minor_top_level.tcl.pre_2023",
    )
    assert regional == CompetitionIdentity(
        family="TCL",
        tier=CompetitionTier.REGIONAL,
        scope=CompetitionScope.DOMESTIC,
        matched_rule="regional.tcl.from_2023",
    )


def test_tcl_uses_tournament_year_when_match_date_is_missing() -> None:
    assert classify_competition("TCL Winter 2022").tier is CompetitionTier.MINOR_TOP_LEVEL
    assert classify_competition("TCL Spring 2024").tier is CompetitionTier.REGIONAL
    assert classify_competition("TCL Spring").matched_rule == "regional.tcl.undated"


def test_other_cross_regional_event_does_not_look_domestic() -> None:
    identity = classify_competition("LCK-LPL-LMS-VCS Rift Rivals 2019")

    assert identity.family == "Rift Rivals"
    assert identity.tier is CompetitionTier.INTERNATIONAL
    assert identity.scope is CompetitionScope.CROSS_LEAGUE
    assert identity.matched_rule == "international.rift_rivals"


@pytest.mark.parametrize("name", [None, "", "   ", float("nan"), "<NA>"])
def test_missing_values_are_explicitly_unknown(name: object) -> None:
    assert classify_competition(name) == CompetitionIdentity(
        family="unknown",
        tier=CompetitionTier.UNKNOWN,
        scope=CompetitionScope.UNKNOWN,
        matched_rule="unknown.missing",
    )


@pytest.mark.parametrize("name", ["Community Cup 2026", 12345, object()])
def test_unmatched_values_are_not_coerced_to_regional(name: object) -> None:
    identity = classify_competition(name)

    assert identity.family == "unknown"
    assert identity.tier is CompetitionTier.UNKNOWN
    assert identity.scope is CompetitionScope.UNKNOWN
    assert identity.matched_rule == "unknown.no_match"


def test_public_enum_values_match_the_serialization_contract() -> None:
    assert [tier.value for tier in CompetitionTier] == [
        "international",
        "major",
        "minor_top_level",
        "regional",
        "development",
        "unknown",
    ]
    assert [scope.value for scope in CompetitionScope] == [
        "domestic",
        "cross_league",
        "unknown",
    ]


def test_identity_is_frozen_and_classification_is_deterministic() -> None:
    first = classify_competition("  EMEA—Masters 2025 Spring  ")
    second = classify_competition("EMEA Masters 2025 Spring")

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.family = "changed"  # type: ignore[misc]
