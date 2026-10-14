"""Regional Competition Family Scaling and Location Resolution.

Implements the verified REGION_PLAN architecture:
1. Domestic rating points do not scale 1:1 across regional borders.
2. Applies a domestic rating discount factor (gamma = 0.70) to inter-regional matchups.
3. Automatically maps teams to their domestic competition family (LCK, LPL, LEC, LCS, Minor, ERL).
"""

from __future__ import annotations

import math
from typing import Mapping

DEFAULT_REGIONAL_GAMMA: float = 0.70

# Default regional strength offsets (relative to LEC anchor = 0.0)
DEFAULT_REGIONAL_OFFSETS: dict[str, float] = {
    "LCK": 0.463,           # Korea major region
    "LPL": 0.355,           # China major region
    "LCS": 0.013,           # North America
    "LEC": 0.000,           # Europe anchor
    "Minor Tier 1": -0.456, # PCS, VCS, CBLOL, LJL
    "Regional / ERL": -0.550, # EMEA Regional Leagues, NACL, LCK CL
    "Other": 0.000,
}

LEAGUE_TO_REGION: dict[str, str] = {
    "lck": "LCK", "korea": "LCK",
    "lpl": "LPL", "china": "LPL",
    "lcs": "LCS", "north_america": "LCS", "na": "LCS",
    "lec": "LEC", "europe": "LEC", "eu": "LEC",
    "cblol": "Minor Tier 1", "pcs": "Minor Tier 1", "vcs": "Minor Tier 1", "ljl": "Minor Tier 1",
    "lfl": "Regional / ERL", "superliga": "Regional / ERL", "prime": "Regional / ERL",
    "nacl": "Regional / ERL", "lck_cl": "Regional / ERL", "lrn": "Regional / ERL",
    "lrs": "Regional / ERL", "tcl": "Regional / ERL", "lit": "Regional / ERL",
}

# Known global roster associations for top international contenders
KNOWN_TEAM_REGIONS: dict[str, str] = {
    # LEC
    "g2": "LEC", "g2 esports": "LEC", "fnatic": "LEC", "karmine corp": "LEC", "movistar koi": "LEC",
    "mad lions koi": "LEC", "team vitality": "LEC", "team bds": "LEC", "rogue": "LEC", "giantx": "LEC",
    "sk gaming": "LEC",
    # LCK
    "gen.g": "LCK", "geng": "LCK", "t1": "LCK", "hanwha life": "LCK", "hanwha life esports": "LCK",
    "dplus kia": "LCK", "dplus": "LCK", "kt rolster": "LCK", "kt": "LCK", "kwangdong freecs": "LCK",
    "fearx": "LCK", "drx": "LCK", "ns redforce": "LCK", "brion": "LCK",
    # LPL
    "bilibili gaming": "LPL", "blg": "LPL", "top esports": "LPL", "tes": "LPL", "jd gaming": "LPL",
    "jdg": "LPL", "weibo gaming": "LPL", "wbg": "LPL", "edward gaming": "LPL", "edg": "LPL",
    "funplus phoenix": "LPL", "fpx": "LPL", "lng esports": "LPL", "lng": "LPL", "anyone's legend": "LPL",
    "ninjas in pyjamas": "LPL", "nip": "LPL", "invictus gaming": "LPL", "ig": "LPL", "royal never give up": "LPL",
    # LCS
    "team liquid": "LCS", "liquid": "LCS", "cloud9": "LCS", "c9": "LCS", "flyquest": "LCS",
    "dignitas": "LCS", "100 thieves": "LCS", "shopify rebellion": "LCS", "immortals": "LCS",
    # Minor Tier 1
    "pain gaming": "Minor Tier 1", "loud": "Minor Tier 1", "red canids": "Minor Tier 1",
    "gam esports": "Minor Tier 1", "psg talon": "Minor Tier 1", "fukuoka softbank hawks": "Minor Tier 1",
    "shg": "Minor Tier 1", "detonatioN focusme": "Minor Tier 1", "dfm": "Minor Tier 1",
    "ctbc flying oyster": "Minor Tier 1", "cfo": "Minor Tier 1",
}


def resolve_team_region(team_name: str | None, league: str | None = None) -> str:
    """Resolve a team's canonical domestic competition region."""
    if not team_name:
        return "Other"
    norm = team_name.lower().strip()
    if norm in KNOWN_TEAM_REGIONS:
        return KNOWN_TEAM_REGIONS[norm]

    # Partial prefix/suffix matches for common abbreviations
    for k, reg in KNOWN_TEAM_REGIONS.items():
        if len(k) >= 3 and (norm.startswith(k) or norm.endswith(k) or f" {k} " in f" {norm} "):
            return reg

    if league:
        l_norm = league.lower().replace(" ", "_").replace("-", "_").strip()
        for k, reg in LEAGUE_TO_REGION.items():
            if k in l_norm:
                return reg

    return "Other"


def compute_regional_logit_offset(
    team_a: str | None,
    team_b: str | None,
    league: str | None = None,
    *,
    regional_offsets: Mapping[str, float] | None = None,
    gamma: float = DEFAULT_REGIONAL_GAMMA,
) -> float:
    """Compute directional logit offset z_offset = (offset_a - offset_b) * gamma."""
    if not team_a or not team_b:
        return 0.0

    reg_a = resolve_team_region(team_a, league)
    reg_b = resolve_team_region(team_b, league)

    # If same region or unresolvable, no inter-regional scaling applies
    if reg_a == reg_b or reg_a == "Other" or reg_b == "Other":
        return 0.0

    offsets = regional_offsets or DEFAULT_REGIONAL_OFFSETS
    off_a = offsets.get(reg_a, 0.0)
    off_b = offsets.get(reg_b, 0.0)

    offset_diff = (off_a - off_b) * gamma
    return float(offset_diff)
