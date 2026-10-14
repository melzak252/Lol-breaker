"""Auditable tournament-name taxonomy for League of Legends competitions.

The source datasets contain free-form tournament names rather than a stable
competition identifier.  This module deliberately uses ordered, named rules so
that every classification can be traced back to a specific decision.  More
specific development and cross-league rules precede broad league-family rules.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Pattern


class CompetitionTier(str, Enum):
    """Competitive level represented by a tournament."""

    INTERNATIONAL = "international"
    MAJOR = "major"
    MINOR_TOP_LEVEL = "minor_top_level"
    REGIONAL = "regional"
    DEVELOPMENT = "development"
    UNKNOWN = "unknown"


class CompetitionScope(str, Enum):
    """Whether results belong to one domestic ecosystem or connect leagues."""

    DOMESTIC = "domestic"
    CROSS_LEAGUE = "cross_league"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CompetitionIdentity:
    """Canonical and auditable identity assigned to a tournament name."""

    family: str
    tier: CompetitionTier
    scope: CompetitionScope
    matched_rule: str


@dataclass(frozen=True)
class _Rule:
    pattern: Pattern[str]
    identity: CompetitionIdentity
    not_before: date | None = None
    before: date | None = None

    def matches(self, normalized_name: str, effective_date: date | None) -> bool:
        if self.not_before is not None:
            if effective_date is None or effective_date < self.not_before:
                return False
        if self.before is not None:
            if effective_date is None or effective_date >= self.before:
                return False
        return self.pattern.search(normalized_name) is not None


def _identity(
    family: str,
    tier: CompetitionTier,
    scope: CompetitionScope,
    matched_rule: str,
) -> CompetitionIdentity:
    return CompetitionIdentity(
        family=family,
        tier=tier,
        scope=scope,
        matched_rule=matched_rule,
    )


def _rule(
    pattern: str,
    family: str,
    tier: CompetitionTier,
    scope: CompetitionScope = CompetitionScope.DOMESTIC,
    matched_rule: str = "",
    *,
    not_before: date | None = None,
    before: date | None = None,
) -> _Rule:
    return _Rule(
        pattern=re.compile(pattern),
        identity=_identity(family, tier, scope, matched_rule),
        not_before=not_before,
        before=before,
    )


# Names are normalized to lower-case ASCII words before these expressions run.
# Ordering is part of the public audit semantics: lower-tier and cross-league
# exceptions must win before broad tokens such as ``lck``, ``lpl``, or ``lcs``.
_RULES: tuple[_Rule, ...] = (
    # Academy, challenger, and second-division competitions.
    _rule(
        r"\blck (?:cl|challengers?(?: league)?)\b",
        "LCK CL",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lck_cl",
    ),
    _rule(
        r"(?:^|\s)(?:ldl|league of legends development league)(?:\s|$)",
        "LDL",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.ldl",
    ),
    _rule(
        r"\b(?:nacl|na challengers?(?: league)?|north american challengers?(?: league)?|americas challengers?)\b",
        "NACL",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.nacl",
    ),
    _rule(
        r"\blcs proving grounds\b",
        "LCS Proving Grounds",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lcs_proving_grounds",
    ),
    _rule(
        r"\bcblol academy\b",
        "CBLOL Academy",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.cblol_academy",
    ),
    _rule(
        r"\bljl academy\b",
        "LJL Academy",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.ljl_academy",
    ),
    _rule(
        r"\b(?:na|north american) academy\b",
        "NA Academy",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.na_academy",
    ),
    _rule(
        r"\bturkey academy\b",
        "Turkey Academy",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.turkey_academy",
    ),
    _rule(
        r"\b(?:circuito desafiante|brcc)\b|^cd (?:20\d{2} )?split\b",
        "Circuito Desafiante",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.circuito_desafiante",
    ),
    _rule(
        r"^lrn\b",
        "LRN",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lrn",
    ),
    _rule(
        r"^lrs\b",
        "LRS",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lrs",
    ),
    _rule(
        r"^ck (?:spring|summer)\b",
        "LCK CL",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lck_cl_historical_ck",
    ),
    _rule(
        r"^challenge france\b|^lfl div(?:ision)? ?2\b",
        "LFL Div2",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.lfl_div2",
    ),
    _rule(
        r"^eu cs\b",
        "EU Challenger Series",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.eu_challenger_series",
    ),
    _rule(
        r"^na cs\b",
        "NA Challenger Series",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.na_challenger_series",
    ),
    _rule(
        r"^hitpoint challengers?\b",
        "Hitpoint Challengers",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.hitpoint_challengers",
    ),
    _rule(
        r"^prime league 2nd div(?:ision)?\b",
        "Prime League Second Division",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.prime_league_second_division",
    ),
    _rule(
        r"^(?:lvp sl|superliga) 2nd div(?:ision)?\b|\blvp2\b",
        "SuperLiga Second Division",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.superliga_second_division",
    ),
    _rule(
        r"^tcl div(?:ision)? ?2\b",
        "TCL Div2",
        CompetitionTier.DEVELOPMENT,
        matched_rule="development.tcl_div2",
    ),
    # Cross-regional tournaments. EMEA/EU Masters connects domestic ERLs but is
    # not a global international event, so tier and scope intentionally differ.
    _rule(
        r"\b(?:emea|eu) masters?\b",
        "EMEA Masters",
        CompetitionTier.REGIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "cross_league.emea_masters",
    ),
    _rule(
        r"\biberian cup\b",
        "Iberian Cup",
        CompetitionTier.REGIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "cross_league.iberian_cup",
    ),
    _rule(
        r"\bamericas cup\b",
        "Americas Cup",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.americas_cup",
    ),
    _rule(
        r"\brift rivals\b",
        "Rift Rivals",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.rift_rivals",
    ),
    _rule(
        r"\bbattle of the atlantic\b",
        "Battle of the Atlantic",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.battle_of_the_atlantic",
    ),
    _rule(
        r"^mid season cup\b",
        "Mid-Season Cup",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.mid_season_cup",
    ),
    _rule(
        r"^mid season showdown\b",
        "Mid-Season Showdown",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.mid_season_showdown",
    ),
    _rule(
        r"\b(?:international wildcard|iwci|iwc qualifier|iwc desafio|iwc turkey)\b",
        "International Wildcard",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.wildcard",
    ),
    # Global international events. Anchoring MSI prevents domestic names such
    # as "LCK Road to MSI" from being reclassified as international.
    _rule(
        r"^(?:(?:season \d+|20\d{2}) )?(?:(?:league of legends )?world championship|worlds|mistrzostwa swiata)\b",
        "Worlds",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.worlds",
    ),
    _rule(
        r"^(?:20\d{2} )?(?:msi|mid season invitational)\b",
        "MSI",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.msi",
    ),
    _rule(
        r"^(?:20\d{2} )?first stand\b",
        "First Stand",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.first_stand",
    ),
    _rule(
        r"^(?:20\d{2} )?(?:esports world cup|ewc)\b",
        "Esports World Cup",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.esports_world_cup",
    ),
    _rule(
        r"\b(?:intel extreme masters|iem)\b",
        "Intel Extreme Masters",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.iem",
    ),
    _rule(
        r"\ball star(?: event)?\b",
        "All-Star",
        CompetitionTier.INTERNATIONAL,
        CompetitionScope.CROSS_LEAGUE,
        "international.all_star",
    ),
    # Historical and current major ecosystems. Explicit aliases precede the
    # shorter abbreviations so their audit trail records the historical name.
    _rule(
        r"\bkespa cup\b|^champions (?:spring|summer|winter)\b",
        "LCK",
        CompetitionTier.MAJOR,
        matched_rule="major.lck_domestic_cup",
    ),
    _rule(
        r"\bdemacia cup\b",
        "LPL",
        CompetitionTier.MAJOR,
        matched_rule="major.lpl_domestic_cup",
    ),
    _rule(
        r"^eu regional qualifiers?\b",
        "LEC",
        CompetitionTier.MAJOR,
        matched_rule="major.lec_historical_qualifier",
    ),
    _rule(
        r"^na regional qualifiers?\b",
        "LCS",
        CompetitionTier.MAJOR,
        matched_rule="major.lcs_historical_qualifier",
    ),
    _rule(
        (
            r"\b(?:eu lcs|european (?:league of legends )?championship series|"
            r"european championship|league of legends emea championship)\b"
        ),
        "LEC",
        CompetitionTier.MAJOR,
        matched_rule="major.lec_historical_alias",
    ),
    _rule(
        r"\b(?:na lcs|north americ(?:a|an) (?:league of legends )?championship series)\b",
        "LCS",
        CompetitionTier.MAJOR,
        matched_rule="major.lcs_historical_alias",
    ),
    _rule(
        r"\b(?:ogn champions|champions korea|korea champions|league of legends champions korea)\b",
        "LCK",
        CompetitionTier.MAJOR,
        matched_rule="major.lck_historical_alias",
    ),
    _rule(
        r"\b(?:league of legends pro league|lol pro league|tencent lol pro league|tj sports lol lpl)\b",
        "LPL",
        CompetitionTier.MAJOR,
        matched_rule="major.lpl_long_name",
    ),
    _rule(
        r"\b(?:league of legends championship series|league championship series)\b",
        "LCS",
        CompetitionTier.MAJOR,
        matched_rule="major.lcs_long_name",
    ),
    _rule(
        r"\blec\b",
        "LEC",
        CompetitionTier.MAJOR,
        matched_rule="major.lec",
    ),
    _rule(
        r"\blck\b",
        "LCK",
        CompetitionTier.MAJOR,
        matched_rule="major.lck",
    ),
    _rule(
        r"\blpl\b",
        "LPL",
        CompetitionTier.MAJOR,
        matched_rule="major.lpl",
    ),
    _rule(
        r"\blcs\b",
        "LCS",
        CompetitionTier.MAJOR,
        matched_rule="major.lcs",
    ),
    _rule(
        r"\blta(?: north| south)?\b|\bleague of the americas\b",
        "LTA",
        CompetitionTier.MAJOR,
        matched_rule="major.lta",
    ),
    # Top-level leagues outside the four long-standing major ecosystems.
    _rule(
        r"\blcp\b|\bleague of legends championship pacific\b",
        "LCP",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lcp",
    ),
    _rule(
        r"\bpcs\b|\bpacific championship series\b",
        "PCS",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.pcs",
    ),
    _rule(
        r"\bvcs\b|\bvietnam championship series\b",
        "VCS",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.vcs",
    ),
    _rule(
        r"\bcblol\b|\bcampeonato brasileiro de league of legends\b",
        "CBLOL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.cblol",
    ),
    _rule(
        r"\bljl\b|\b(?:league of legends )?japan league\b",
        "LJL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.ljl",
    ),
    _rule(
        r"\blla\b|\bliga latinoamerica\b",
        "LLA",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lla",
    ),
    _rule(
        r"\blco\b|\bleague of legends circuit oceania\b",
        "LCO",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lco",
    ),
    _rule(
        r"\blms\b|\bleague of legends masters series\b",
        "LMS",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lms",
    ),
    _rule(
        r"\bopl\b|\boceanic pro league\b",
        "OPL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.opl",
    ),
    _rule(
        r"\blcl\b|\bleague of legends continental league\b",
        "LCL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lcl",
    ),
    _rule(
        r"\b(?:cls|lln)\b",
        "LLA",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lla_historical_alias",
    ),
    _rule(
        r"\b(?:lst|sea tour)\b",
        "LST",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.lst",
    ),
    _rule(
        r"\bgpl\b|\bgarena premier league\b",
        "GPL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.gpl",
    ),
    # TCL was an independent top-level league through 2022 and joined the EMEA
    # regional-league system in 2023. The tournament year is used when an
    # explicit match date is unavailable.
    _rule(
        r"\btcl\b|\bturkish championship league\b",
        "TCL",
        CompetitionTier.MINOR_TOP_LEVEL,
        matched_rule="minor_top_level.tcl.pre_2023",
        before=date(2023, 1, 1),
    ),
    _rule(
        r"\btcl\b|\bturkish championship league\b",
        "TCL",
        CompetitionTier.REGIONAL,
        matched_rule="regional.tcl.from_2023",
        not_before=date(2023, 1, 1),
    ),
    _rule(
        r"\btcl\b|\bturkish championship league\b",
        "TCL",
        CompetitionTier.REGIONAL,
        matched_rule="regional.tcl.undated",
    ),
    # EMEA regional leagues (ERLs).
    _rule(
        r"^lvp uklc\b",
        "NLC",
        CompetitionTier.REGIONAL,
        matched_rule="regional.nlc_historical_uklc",
    ),
    _rule(
        r"^lvp (?:spring|summer|winter)\b",
        "SuperLiga",
        CompetitionTier.REGIONAL,
        matched_rule="regional.superliga_historical_lvp",
    ),
    _rule(
        r"^mcr lol\b",
        "Hitpoint Masters",
        CompetitionTier.REGIONAL,
        matched_rule="regional.hitpoint_historical_mcr",
    ),
    _rule(
        r"^rel season\b",
        "REL",
        CompetitionTier.REGIONAL,
        matched_rule="regional.romanian_esports_league",
    ),
    _rule(
        r"^trinity force puchar polski\b",
        "Ultraliga",
        CompetitionTier.REGIONAL,
        matched_rule="regional.ultraliga_historical_polish_cup",
    ),
    _rule(r"\blfl\b|\bla ligue francaise\b", "LFL", CompetitionTier.REGIONAL, matched_rule="regional.lfl"),
    _rule(r"\bprime league\b", "Prime League", CompetitionTier.REGIONAL, matched_rule="regional.prime_league"),
    _rule(r"\bultraliga\b", "Ultraliga", CompetitionTier.REGIONAL, matched_rule="regional.ultraliga"),
    _rule(r"\bebl\b|\besports balkan league\b", "EBL", CompetitionTier.REGIONAL, matched_rule="regional.ebl"),
    _rule(
        r"\bnlc\b|\bnorthern league of legends championship\b",
        "NLC",
        CompetitionTier.REGIONAL,
        matched_rule="regional.nlc",
    ),
    _rule(
        r"\b(?:lvp )?(?:sl|slo|superliga)\b|\bsuperliga orange\b|^sl (?:20\d{2} )?(?:spring|summer|winter)\b",
        "SuperLiga",
        CompetitionTier.REGIONAL,
        matched_rule="regional.superliga",
    ),
    _rule(r"\blpol\b|\blplol\b|\bliga portuguesa\b", "LPLOL", CompetitionTier.REGIONAL, matched_rule="regional.lpol"),
    _rule(
        r"\bhitpoint (?:masters|winter|legends)\b",
        "Hitpoint Masters",
        CompetitionTier.REGIONAL,
        matched_rule="regional.hitpoint_masters",
    ),
    _rule(
        r"\b(?:greek legends league|gll|hll)\b",
        "HLL",
        CompetitionTier.REGIONAL,
        matched_rule="regional.hll",
    ),
    _rule(
        r"\barabian league\b",
        "Arabian League",
        CompetitionTier.REGIONAL,
        matched_rule="regional.arabian_league",
    ),
    _rule(
        r"\bbaltic masters\b",
        "Baltic Masters",
        CompetitionTier.REGIONAL,
        matched_rule="regional.baltic_masters",
    ),
    _rule(
        r"\b(?:pg nationals|lit (?:20\d{2} )?(?:spring|summer|winter))\b",
        "LIT",
        CompetitionTier.REGIONAL,
        matched_rule="regional.lit",
    ),
    _rule(
        r"\b(?:elite series|(?:transip )?road of legends|belgian league|dutch league)\b",
        "Road of Legends",
        CompetitionTier.REGIONAL,
        matched_rule="regional.road_of_legends",
    ),
    _rule(
        r"\brift legends\b",
        "Rift Legends",
        CompetitionTier.REGIONAL,
        matched_rule="regional.rift_legends",
    ),
    _rule(
        r"^les (?:20\d{2} )?(?:kick off|spring|summer|winter|final four|finals?)\b",
        "LES",
        CompetitionTier.REGIONAL,
        matched_rule="regional.les",
    ),
)

_UNKNOWN_MISSING = _identity(
    "unknown",
    CompetitionTier.UNKNOWN,
    CompetitionScope.UNKNOWN,
    "unknown.missing",
)
_UNKNOWN_UNSTRINGIFIABLE = _identity(
    "unknown",
    CompetitionTier.UNKNOWN,
    CompetitionScope.UNKNOWN,
    "unknown.unstringifiable",
)
_UNKNOWN_NO_MATCH = _identity(
    "unknown",
    CompetitionTier.UNKNOWN,
    CompetitionScope.UNKNOWN,
    "unknown.no_match",
)
_MISSING_NAMES = frozenset({"", "na", "nan", "nat", "none", "null"})
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_name(name: object) -> str | None:
    if name is None:
        return None
    text = name.decode("utf-8") if isinstance(name, bytes) else str(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    normalized = _NON_WORD_RE.sub(" ", text.casefold()).strip()
    return None if normalized in _MISSING_NAMES else normalized


def _effective_date(normalized_name: str, match_date: date | None) -> date | None:
    if isinstance(match_date, datetime):
        return match_date.date()
    if match_date is not None:
        return match_date
    year_match = _YEAR_RE.search(normalized_name)
    if year_match is None:
        return None
    return date(int(year_match.group(1)), 1, 1)


def classify_competition(name: object, match_date: date | None = None) -> CompetitionIdentity:
    """Classify a free-form tournament name using the first matching rule.

    Args:
        name: Raw tournament value from a source dataset.
        match_date: Match date used for time-dependent league identities. When
            omitted, an explicit four-digit tournament year is used if present.

    Returns:
        A frozen identity containing canonical family, tier, scope, and the
        stable identifier of the rule that matched. Missing and unmatched names
        remain explicitly unknown.
    """

    try:
        normalized_name = _normalize_name(name)
    except Exception:
        return _UNKNOWN_UNSTRINGIFIABLE
    if normalized_name is None:
        return _UNKNOWN_MISSING

    effective_date = _effective_date(normalized_name, match_date)
    for rule in _RULES:
        if rule.matches(normalized_name, effective_date):
            return rule.identity
    return _UNKNOWN_NO_MATCH


__all__ = [
    "CompetitionIdentity",
    "CompetitionScope",
    "CompetitionTier",
    "classify_competition",
]
