"""Fixed-graph and scenario tournament simulations powered by native C0."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import text
from betting_app.core.db import get_session
from betting_app.core.matching import normalize_team_name
from betting_app.services.canonical_match_service import canonical_team_key
from betting_app.services import c0_inference, mapping_service


@dataclass
class BracketMatchNode:
    id: str  # e.g. "UB_R1_M1", "UB_Final", "Grand_Final"
    name: str  # e.g. "Upper Round 1 - Match 1"
    round_name: str  # e.g. "Upper Round 1", "Lower Final"
    bracket_section: str  # "upper", "lower", "final"
    best_of: int = 5
    team1: str | None = None  # Team name or None if TBD
    team2: str | None = None
    winner: str | None = None  # Confirmed winner if match already played
    score1: int | None = None
    score2: int | None = None
    next_match_winner_id: str | None = None  # Where does the winner advance
    next_match_winner_slot: int = 1  # Slot 1 or 2 in target match
    next_match_loser_id: str | None = None  # Where does the loser drop (Double Elim)
    next_match_loser_slot: int = 2


@dataclass
class TournamentBracket:
    id: str
    name: str
    region: str
    format: str  # "double_elimination", "single_elimination"
    matches: dict[str, BracketMatchNode]
    teams: list[str]


def get_lck_2026_playoffs_bracket() -> TournamentBracket:
    """Construct the curated 6-team LCK playoff graph with embedded played results.

    This live-state scenario is not a certified pre-event bracket or rules archive:
      Upper Round 1 (Quarterfinals):
        - UB_R1_M1: KT Rolster (3) vs Dplus (0) -> KT won, Dplus dropped to Lower R1
        - UB_R1_M2: T1 (3) vs BNK FearX (2) -> T1 won, BNK FearX dropped to Lower R1
      Upper Round 2 (Semifinals):
        - UB_R2_M1: Gen.G (3) vs KT Rolster (0) -> Gen.G won to UB Final, KT dropped to Lower R2
        - UB_R2_M2: Hanwha Life (3) vs T1 (2) -> Hanwha won to UB Final, T1 dropped to Lower R3
      Lower Round 1:
        - LB_R1: Dplus (3) vs BNK FearX (2) -> Dplus won, BNK FearX eliminated
      Lower Round 2 (Sept 4):
        - LB_R2: KT Rolster vs Dplus -> Winner to Lower R3 vs T1, Loser eliminated
      Upper Final (Sept 5):
        - UB_Final: Gen.G vs Hanwha Life -> Winner to Grand Final, Loser to Lower Final
      Lower Round 3 / Semifinal (Sept 6):
        - LB_R3: T1 vs Winner(KT vs Dplus) -> Winner to Lower Final, Loser eliminated
      Lower Final (Sept 12):
        - LB_Final: Loser(Gen.G vs Hanwha) vs Winner(LB_R3) -> Winner to Grand Final, Loser 3rd
      Grand Final (Sept 13):
        - Grand_Final: Winner(Gen.G vs Hanwha) vs Winner(Lower Final)
    """
    teams = ["Gen.G", "Hanwha Life Esports", "T1", "KT Rolster", "Dplus", "BNK FearX"]

    matches: dict[str, BracketMatchNode] = {
        # Upper Round 1
        "UB_R1_M1": BracketMatchNode(
            id="UB_R1_M1",
            name="Upper Quarterfinal 1",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="KT Rolster",
            team2="Dplus",
            winner="KT Rolster",
            score1=3,
            score2=0,
            next_match_winner_id="UB_R2_M1",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1",
            next_match_loser_slot=1,
        ),
        "UB_R1_M2": BracketMatchNode(
            id="UB_R1_M2",
            name="Upper Quarterfinal 2",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="T1",
            team2="BNK FearX",
            winner="T1",
            score1=3,
            score2=2,
            next_match_winner_id="UB_R2_M2",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1",
            next_match_loser_slot=2,
        ),
        # Upper Round 2 (Semifinals)
        "UB_R2_M1": BracketMatchNode(
            id="UB_R2_M1",
            name="Upper Semifinal 1",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="Gen.G",
            team2="KT Rolster",
            winner="Gen.G",
            score1=3,
            score2=0,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_R2",
            next_match_loser_slot=1,
        ),
        "UB_R2_M2": BracketMatchNode(
            id="UB_R2_M2",
            name="Upper Semifinal 2",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="Hanwha Life Esports",
            team2="T1",
            winner="Hanwha Life Esports",
            score1=3,
            score2=2,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R3",
            next_match_loser_slot=1,
        ),
        # Upper Final
        "UB_Final": BracketMatchNode(
            id="UB_Final",
            name="Upper Bracket Final",
            round_name="Upper Final",
            bracket_section="upper",
            best_of=5,
            team1="Gen.G",
            team2="Hanwha Life Esports",
            winner=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_Final",
            next_match_loser_slot=1,
        ),
        # Lower Round 1
        "LB_R1": BracketMatchNode(
            id="LB_R1",
            name="Lower Round 1",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1="Dplus",
            team2="BNK FearX",
            winner="Dplus",
            score1=3,
            score2=2,
            next_match_winner_id="LB_R2",
            next_match_winner_slot=2,
        ),
        # Lower Round 2
        "LB_R2": BracketMatchNode(
            id="LB_R2",
            name="Lower Round 2 (Worlds Decider)",
            round_name="Lower Round 2",
            bracket_section="lower",
            best_of=5,
            team1="KT Rolster",
            team2="Dplus",
            winner=None,
            next_match_winner_id="LB_R3",
            next_match_winner_slot=2,
        ),
        # Lower Round 3 / Semifinal
        "LB_R3": BracketMatchNode(
            id="LB_R3",
            name="Lower Bracket Semifinal",
            round_name="Lower Round 3",
            bracket_section="lower",
            best_of=5,
            team1="T1",
            team2=None,
            next_match_winner_id="LB_Final",
            next_match_winner_slot=2,
        ),
        # Lower Final
        "LB_Final": BracketMatchNode(
            id="LB_Final",
            name="Lower Bracket Final",
            round_name="Lower Final",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=2,
        ),
        # Grand Final
        "Grand_Final": BracketMatchNode(
            id="Grand_Final",
            name="Grand Final",
            round_name="Grand Final",
            bracket_section="final",
            best_of=5,
            team1=None,
            team2=None,
        ),
    }

    return TournamentBracket(
        id="lck_2026_playoffs",
        name="LCK 2026 Season - Playoffs",
        region="LCK",
        format="double_elimination",
        matches=matches,
        teams=teams,
    )


def get_lec_2026_summer_playoffs_bracket() -> TournamentBracket:
    """Construct the curated 6-team LEC 2026 Summer fixed playoff graph."""
    teams = ["Karmine Corp", "GIANTX", "G2 Esports", "Team Vitality", "Natus Vincere", "Movistar KOI"]

    matches: dict[str, BracketMatchNode] = {
        # Upper Semifinal 1
        "UB_SF1": BracketMatchNode(
            id="UB_SF1",
            name="Upper Semifinal 1",
            round_name="Upper Semifinals",
            bracket_section="upper",
            best_of=5,
            team1="Karmine Corp",
            team2="GIANTX",
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_R1_M1",
            next_match_loser_slot=2,
        ),
        # Upper Semifinal 2
        "UB_SF2": BracketMatchNode(
            id="UB_SF2",
            name="Upper Semifinal 2",
            round_name="Upper Semifinals",
            bracket_section="upper",
            best_of=5,
            team1="G2 Esports",
            team2="Team Vitality",
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1_M2",
            next_match_loser_slot=2,
        ),
        # Lower Round 1 - Match 1
        "LB_R1_M1": BracketMatchNode(
            id="LB_R1_M1",
            name="Lower Round 1 - Match 1",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1="Natus Vincere",
            team2=None,  # Loser of KC vs GX
            next_match_winner_id="LB_SF",
            next_match_winner_slot=1,
        ),
        # Lower Round 1 - Match 2
        "LB_R1_M2": BracketMatchNode(
            id="LB_R1_M2",
            name="Lower Round 1 - Match 2",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1="Movistar KOI",
            team2=None,  # Loser of G2 vs VIT
            next_match_winner_id="LB_SF",
            next_match_winner_slot=2,
        ),
        # Upper Final
        "UB_Final": BracketMatchNode(
            id="UB_Final",
            name="Upper Bracket Final",
            round_name="Upper Final",
            bracket_section="upper",
            best_of=5,
            team1=None,
            team2=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_Final",
            next_match_loser_slot=1,
        ),
        # Lower Semifinal
        "LB_SF": BracketMatchNode(
            id="LB_SF",
            name="Lower Bracket Semifinal",
            round_name="Lower Semifinal",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            next_match_winner_id="LB_Final",
            next_match_winner_slot=2,
        ),
        # Lower Final
        "LB_Final": BracketMatchNode(
            id="LB_Final",
            name="Lower Bracket Final",
            round_name="Lower Final",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=2,
        ),
        # Grand Final
        "Grand_Final": BracketMatchNode(
            id="Grand_Final",
            name="LEC Grand Final (Nice)",
            round_name="Grand Final",
            bracket_section="final",
            best_of=5,
            team1=None,
            team2=None,
        ),
    }

    return TournamentBracket(
        id="lec_2026_summer_playoffs",
        name="LEC 2026 Summer - Playoffs",
        region="LEC",
        format="double_elimination",
        matches=matches,
        teams=teams,
    )


def get_lpl_2026_split3_playoffs_bracket() -> TournamentBracket:
    """Construct the LPL 2026 Split 3 Playoff (Grand Finals) double-elimination bracket tree."""
    teams = [
        "Bilibili Gaming",
        "Anyone's Legend",
        "Team WE",
        "JD Gaming",
        "LGD Gaming",
        "Top Esports",
        "Invictus Gaming",
        "Ninjas in Pyjamas",
    ]

    matches: dict[str, BracketMatchNode] = {
        # Upper Round 1
        "UB_R1_M1": BracketMatchNode(
            id="UB_R1_M1",
            name="Upper Round 1 - Match 1",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="Top Esports",
            team2="LGD Gaming",
            winner=None,
            next_match_winner_id="UB_R2_M2",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1_M1",
            next_match_loser_slot=2,
        ),
        "UB_R1_M2": BracketMatchNode(
            id="UB_R1_M2",
            name="Upper Round 1 - Match 2",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="JD Gaming",
            team2="Team WE",
            winner=None,
            next_match_winner_id="UB_R2_M1",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1_M2",
            next_match_loser_slot=2,
        ),
        # Upper Round 2 (Upper Semifinals)
        "UB_R2_M1": BracketMatchNode(
            id="UB_R2_M1",
            name="Upper Semifinal 1",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="Bilibili Gaming",
            team2=None,
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R2_M1",
            next_match_loser_slot=2,
        ),
        "UB_R2_M2": BracketMatchNode(
            id="UB_R2_M2",
            name="Upper Semifinal 2",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="Anyone's Legend",
            team2=None,
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_R2_M2",
            next_match_loser_slot=2,
        ),
        # Lower Round 1
        "LB_R1_M1": BracketMatchNode(
            id="LB_R1_M1",
            name="Lower Round 1 - Match 1",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1="Invictus Gaming",
            team2=None,
            winner=None,
            next_match_winner_id="LB_R2_M1",
            next_match_winner_slot=1,
        ),
        "LB_R1_M2": BracketMatchNode(
            id="LB_R1_M2",
            name="Lower Round 1 - Match 2",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1="Ninjas in Pyjamas",
            team2=None,
            winner=None,
            next_match_winner_id="LB_R2_M2",
            next_match_winner_slot=1,
        ),
        # Lower Round 2
        "LB_R2_M1": BracketMatchNode(
            id="LB_R2_M1",
            name="Lower Round 2 - Match 1",
            round_name="Lower Round 2",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_R3",
            next_match_winner_slot=1,
        ),
        "LB_R2_M2": BracketMatchNode(
            id="LB_R2_M2",
            name="Lower Round 2 - Match 2",
            round_name="Lower Round 2",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_R3",
            next_match_winner_slot=2,
        ),
        # Lower Round 3 (Lower Semifinal)
        "LB_R3": BracketMatchNode(
            id="LB_R3",
            name="Lower Bracket Semifinal",
            round_name="Lower Round 3",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_Final",
            next_match_winner_slot=2,
        ),
        # Upper Final
        "UB_Final": BracketMatchNode(
            id="UB_Final",
            name="Upper Bracket Final",
            round_name="Upper Final",
            bracket_section="upper",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_Final",
            next_match_loser_slot=1,
        ),
        # Lower Final
        "LB_Final": BracketMatchNode(
            id="LB_Final",
            name="Lower Bracket Final",
            round_name="Lower Final",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=2,
        ),
        # Grand Final
        "Grand_Final": BracketMatchNode(
            id="Grand_Final",
            name="LPL Grand Final (Shanghai)",
            round_name="Grand Final",
            bracket_section="final",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
        ),
    }

    return TournamentBracket(
        id="lpl_2026_split3_playoffs",
        name="LPL 2026 Split 3 - Playoffs",
        region="LPL",
        format="double_elimination",
        matches=matches,
        teams=teams,
    )


def get_lcs_2026_championship_bracket() -> TournamentBracket:
    """Construct the curated 6-team LCS 2026 Summer Playoff fixed graph."""
    teams = ["Team Liquid", "LYON", "Cloud9", "FlyQuest", "Sentinels", "Shopify Rebellion"]

    matches: dict[str, BracketMatchNode] = {
        # Upper Quarterfinal 1
        "UB_R1_M1": BracketMatchNode(
            id="UB_R1_M1",
            name="Upper Quarterfinal 1",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="Cloud9",
            team2="Shopify Rebellion",
            winner=None,
            next_match_winner_id="UB_R2_M1",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1",
            next_match_loser_slot=1,
        ),
        # Upper Quarterfinal 2
        "UB_R1_M2": BracketMatchNode(
            id="UB_R1_M2",
            name="Upper Quarterfinal 2",
            round_name="Upper Round 1",
            bracket_section="upper",
            best_of=5,
            team1="Sentinels",
            team2="FlyQuest",
            winner=None,
            next_match_winner_id="UB_R2_M2",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R1",
            next_match_loser_slot=2,
        ),
        # Upper Semifinal 1
        "UB_R2_M1": BracketMatchNode(
            id="UB_R2_M1",
            name="Upper Semifinal 1",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="LYON",
            team2=None,
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_R2",
            next_match_loser_slot=1,
        ),
        # Upper Semifinal 2
        "UB_R2_M2": BracketMatchNode(
            id="UB_R2_M2",
            name="Upper Semifinal 2",
            round_name="Upper Round 2",
            bracket_section="upper",
            best_of=5,
            team1="Team Liquid",
            team2=None,
            winner=None,
            next_match_winner_id="UB_Final",
            next_match_winner_slot=2,
            next_match_loser_id="LB_R3",
            next_match_loser_slot=1,
        ),
        # Lower Round 1
        "LB_R1": BracketMatchNode(
            id="LB_R1",
            name="Lower Round 1",
            round_name="Lower Round 1",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_R2",
            next_match_winner_slot=2,
        ),
        # Lower Round 2
        "LB_R2": BracketMatchNode(
            id="LB_R2",
            name="Lower Round 2",
            round_name="Lower Round 2",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_R3",
            next_match_winner_slot=2,
        ),
        # Upper Final
        "UB_Final": BracketMatchNode(
            id="UB_Final",
            name="Upper Bracket Final",
            round_name="Upper Final",
            bracket_section="upper",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=1,
            next_match_loser_id="LB_Final",
            next_match_loser_slot=1,
        ),
        # Lower Semifinal
        "LB_R3": BracketMatchNode(
            id="LB_R3",
            name="Lower Bracket Semifinal",
            round_name="Lower Round 3",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="LB_Final",
            next_match_winner_slot=2,
        ),
        # Lower Final
        "LB_Final": BracketMatchNode(
            id="LB_Final",
            name="Lower Bracket Final",
            round_name="Lower Final",
            bracket_section="lower",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
            next_match_winner_id="Grand_Final",
            next_match_winner_slot=2,
        ),
        # Grand Final
        "Grand_Final": BracketMatchNode(
            id="Grand_Final",
            name="LCS Grand Final",
            round_name="Grand Final",
            bracket_section="final",
            best_of=5,
            team1=None,
            team2=None,
            winner=None,
        ),
    }

    return TournamentBracket(
        id="lcs_2026_championship",
        name="LCS 2026 Summer - Playoffs",
        region="LCS",
        format="double_elimination",
        matches=matches,
        teams=teams,
    )

SUPPORTED_BRACKETS = {
    "lck_2026_playoffs": get_lck_2026_playoffs_bracket,
    "lec_2026_summer_playoffs": get_lec_2026_summer_playoffs_bracket,
    "lpl_2026_split3_playoffs": get_lpl_2026_split3_playoffs_bracket,
    "lcs_2026_championship": get_lcs_2026_championship_bracket,
}

TOURNAMENT_PROBABILITY_VERSION = "c0-native-2026-w32-e12-v1"


class TournamentSimulator:
    """Monte Carlo fixed-graph simulator using direct native C0 series outputs."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        team_ids: Mapping[str, str] | None = None,
        team_rosters: Mapping[str, Sequence[str]] | None = None,
        decision_at: datetime | None = None,
        mode: str = "full",
        competition_context: str | None = None,
    ):
        self._rng = random.Random(seed)
        self._seed = seed
        self.team_ids = dict(team_ids or {})
        self.team_rosters = {key: tuple(value) for key, value in (team_rosters or {}).items()}
        now = datetime.now(timezone.utc)
        self.decision_at = decision_at or now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.decision_at.tzinfo is None or self.decision_at.utcoffset() is None:
            raise ValueError("decision_at must be timezone-aware.")
        if mode not in {"full", "no_w20", "no_organization"}:
            raise ValueError("Unsupported C0 inference mode.")
        self.mode = mode
        self.competition_context = competition_context
        self._prob_cache: dict[tuple[Any, ...], float] = {}
        self._native_diagnostics: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._identity_cache: dict[str, tuple[str, tuple[str, ...]]] = {}
        self._organization_identity_available: dict[str, bool] = {}
        self._organization_identity_source: dict[str, str] = {}
        self._roster_basis: dict[str, str] = {}
        self._team_keys: dict[str, str] = {}

    @staticmethod
    def _validate_best_of(best_of: int) -> None:
        if type(best_of) is not int or best_of not in {1, 3, 5}:
            raise ValueError("Native C0 supports best_of values 1, 3, and 5.")

    @staticmethod
    def _validate_simulations(n_simulations: int) -> None:
        if type(n_simulations) is not int or n_simulations <= 0:
            raise ValueError("n_simulations must be a positive integer.")

    def _team_key(self, team: str) -> str:
        key = self._team_keys.get(team)
        if key is None:
            key = canonical_team_key(team)
            self._team_keys[team] = key
        return key

    def _team_identity(self, team: str) -> tuple[str, tuple[str, ...]]:
        cached = self._identity_cache.get(team)
        if cached is not None:
            return cached
        key = self._team_key(team)
        team_id = self.team_ids.get(team)
        roster = self.team_rosters.get(team)
        if (team_id is None) != (roster is None):
            raise ValueError(f"Native C0 unavailable: {team} requires team ID and roster IDs as a verified pair.")
        if team_id is None:
            with get_session() as session:
                rows = session.execute(
                    text(
                        """
                        SELECT player_id, role
                        FROM team_current_roster_players
                        WHERE normalized_team_name = :normalized
                        ORDER BY CASE UPPER(role)
                            WHEN 'TOP' THEN 1 WHEN 'JUNGLE' THEN 2 WHEN 'MID' THEN 3
                            WHEN 'ADC' THEN 4 WHEN 'SUPPORT' THEN 5 ELSE 6
                        END
                        """
                    ),
                    {"normalized": normalize_team_name(team)},
                ).mappings().all()
            if not rows:
                raise ValueError(f"Native C0 unavailable: no exact current roster for {team}.")
            player_ids = [str(row["player_id"]) for row in rows if row.get("player_id")]
            roles = {str(row["role"]).upper() for row in rows if row.get("role")}
            if (
                len(player_ids) != 5 or len(set(player_ids)) != 5
                or roles != {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}
            ):
                raise ValueError(f"Native C0 unavailable: {team} lacks a complete five-role roster.")
            native_team_id = mapping_service.native_golgg_team_id(None, team_name=team)
            if native_team_id is None:
                team_id = f"tournament-target:{key}"
                self._organization_identity_available[team] = False
                self._organization_identity_source[team] = "unobserved"
            else:
                team_id = str(native_team_id)
                self._organization_identity_available[team] = True
                self._organization_identity_source[team] = "golgg_native_team_id"
            roster = player_ids
            self._roster_basis[team] = "current_stored_roster"
        else:
            self._organization_identity_available[team] = True
            self._organization_identity_source[team] = "explicit_native_team_id"
            self._roster_basis[team] = "explicit_scenario_roster"
        values = tuple(str(player_id) for player_id in roster)
        if not key or not team_id or len(values) != 5 or len(set(values)) != 5 or any(not value for value in values):
            raise ValueError(f"Native C0 unavailable: {team} lacks exact canonical/team/player IDs.")
        identity = (str(team_id), values)
        self._identity_cache[team] = identity
        return identity

    def estimate_matchup_probability(self, team1: str, team2: str, best_of: int = 5) -> float:
        """Return C0's already-series probability without map-to-series projection."""
        self._validate_best_of(best_of)
        if self._team_key(team1) == self._team_key(team2):
            raise ValueError("A match requires two distinct named teams.")
        team1_id, roster1 = self._team_identity(team1)
        team2_id, roster2 = self._team_identity(team2)
        prediction_mode = (
            self.mode
            if self._organization_identity_available[team1] and self._organization_identity_available[team2]
            else "no_organization"
        )
        origin = self.decision_at.isoformat()
        cache_key = (
            team1_id, roster1, team2_id, roster2, best_of, origin,
            self.competition_context, prediction_mode, TOURNAMENT_PROBABILITY_VERSION,
        )
        if cache_key not in self._prob_cache:
            prediction = c0_inference.predict_c0({
                "canonical": {"id": None},
                "c0_request": {
                    "team1_id": team1_id,
                    "team2_id": team2_id,
                    "roster_a": list(roster1),
                    "roster_b": list(roster2),
                    "best_of": best_of,
                    "decision_at": origin,
                    "competition_context": self.competition_context,
                    "start_at": None,
                    "mode": prediction_mode,
                },
            })
            probability = float(prediction.prob_a)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Native C0 returned an invalid series probability.")
            self._prob_cache[cache_key] = probability
            self._native_diagnostics[cache_key] = {
                "team1_id": team1_id,
                "team2_id": team2_id,
                "organization_identity_available": {
                    team1: self._organization_identity_available[team1],
                    team2: self._organization_identity_available[team2],
                },
                "mode": prediction_mode,
                "best_of": best_of,
                "roster_provenance": {
                    "basis": {team1: self._roster_basis[team1], team2: self._roster_basis[team2]},
                    "submitted_roster_a": list(roster1),
                    "submitted_roster_b": list(roster2),
                    "published_at": None,
                    "authenticated_prior_announcement": False,
                },
                "diagnostics": dict(getattr(prediction, "diagnostics", {}) or {}),
            }
        return self._prob_cache[cache_key]


    def estimate_score_distribution(self, team1: str, team2: str, best_of: int = 5) -> dict[str, float]:
        raise ValueError("Native C0 does not provide map-level probabilities for scoreline projection.")

    def _live_probability(self, team1: str, team2: str, best_of: int, score1: int, score2: int) -> float:
        raise ValueError("Native C0 tournament simulation does not support conditioning on partial live scores.")

    def _provenance(self, teams: Sequence[str]) -> dict[str, Any]:
        identities = {team: self._team_identity(team)[0] for team in teams}
        return {
            "prediction_source": "native_c0",
            "probability_model": "Causal-C0",
            "probability_model_version": TOURNAMENT_PROBABILITY_VERSION,
            "probability_unit": "series",
            "side_policy": "ordered_native_team_ids",
            "team_ids": identities,
            "team_rosters": {team: list(self._team_identity(team)[1]) for team in teams},
            "roster_basis": {team: self._roster_basis[team] for team in teams},
            "roster_publication_status": "unknown",
            "authenticated_prior_announcement": False,
            "decision_at": self.decision_at.isoformat(),
            "mode": self.mode,
            "competition_context": self.competition_context,
            "organization_identity_available": {
                team: self._organization_identity_available[team]
                for team in teams
            },
            "organization_identity_source": {
                team: self._organization_identity_source[team]
                for team in teams
            },
            "native_prediction_provenance": list(self._native_diagnostics.values()),
            "eligibility_live": 0,
            "point_in_time_certified": False,
            "seed": self._seed,
            "strength_uncertainty": "not_modelled",
        }

    def _validate_bracket(
        self, bracket: TournamentBracket, manual: dict[str, str],
    ) -> tuple[list[str], str, dict[str, tuple[int, int]]]:
        """Validate the graph before RNG use; missing slots are not implicit byes."""
        matches = bracket.matches
        if bracket.format not in {"single_elimination", "double_elimination"}:
            raise ValueError("Only fixed single/double-elimination graphs are supported.")
        keys = [self._team_key(team) for team in bracket.teams]
        if not matches or len(keys) < 2 or any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("A bracket requires matches and distinct named participants.")
        if set(manual) - set(matches):
            raise ValueError("Manual overrides reference an unknown match.")
        dependencies: dict[str, set[str]] = {match_id: set() for match_id in matches}
        incoming: dict[tuple[str, int], str] = {}
        for match_id, node in matches.items():
            if node.id != match_id:
                raise ValueError(f"Match key and node id disagree: {match_id}.")
            self._validate_best_of(node.best_of)
            for target, slot in (
                (node.next_match_winner_id, node.next_match_winner_slot),
                (node.next_match_loser_id, node.next_match_loser_slot),
            ):
                if target is None:
                    continue
                if target not in matches or type(slot) is not int or slot not in {1, 2}:
                    raise ValueError(f"Invalid advancement target or slot from {match_id}.")
                if (target, slot) in incoming:
                    raise ValueError(f"Multiple entrants feed {target} slot {slot}.")
                incoming[target, slot] = match_id
                dependencies[target].add(match_id)
            if any(team is not None and team not in bracket.teams for team in (node.team1, node.team2, node.winner)):
                raise ValueError(f"Unknown participant in {match_id}.")
            if node.team1 is not None and node.team1 == node.team2:
                raise ValueError(f"A team cannot play itself in {match_id}.")
            if node.winner is not None and node.winner not in (node.team1, node.team2):
                raise ValueError(f"Confirmed winner is not a participant in {match_id}.")
            if match_id in manual:
                if manual[match_id] not in bracket.teams:
                    raise ValueError(f"Unknown manual winner in {match_id}.")
                if node.winner is not None and manual[match_id] != node.winner:
                    raise ValueError(f"Cannot override a confirmed result in {match_id}.")
            if (node.score1 is None) != (node.score2 is None):
                raise ValueError(f"Both scores must be supplied together in {match_id}.")
            if node.score1 is not None:
                needed = node.best_of // 2 + 1
                if any(type(score) is not int or not 0 <= score <= needed for score in (node.score1, node.score2)):
                    raise ValueError(f"Invalid score in {match_id}.")
                if not node.team1 or not node.team2:
                    raise ValueError(f"A scored match requires known participants: {match_id}.")
                terminal1, terminal2 = node.score1 == needed, node.score2 == needed
                if terminal1 and terminal2:
                    raise ValueError(f"Both teams cannot win {match_id}.")
                score_winner = node.team1 if terminal1 else node.team2 if terminal2 else None
                if score_winner != node.winner:
                    raise ValueError(f"Score and confirmed winner disagree in {match_id}.")
        seed_teams = []
        for match_id, node in matches.items():
            for slot, team in ((1, node.team1), (2, node.team2)):
                if (match_id, slot) not in incoming:
                    if team is None:
                        raise ValueError(f"Missing participant in {match_id}; implicit byes are unsupported.")
                    seed_teams.append(team)
        if sorted(seed_teams) != sorted(bracket.teams):
            raise ValueError("Each participant must enter the graph exactly once as a seed.")
        order: list[str] = []
        visited: set[str] = set()
        while len(order) < len(matches):
            ready = [match_id for match_id, deps in dependencies.items() if match_id not in visited and deps <= visited]
            if not ready:
                raise ValueError("Bracket contains an advancement cycle.")
            order.extend(ready)
            visited.update(ready)
        finals = [match_id for match_id, node in matches.items() if node.next_match_winner_id is None]
        if len(finals) != 1 or matches[finals[0]].next_match_loser_id is not None:
            raise ValueError("A fixed bracket requires one terminal championship without a reset.")
        final_id = finals[0]
        distances: dict[str, int] = {}
        elimination_groups: dict[int, list[str]] = {}
        for match_id in reversed(order):
            node = matches[match_id]
            distance = 0 if match_id == final_id else 1 + distances[node.next_match_winner_id]
            distances[match_id] = distance
            if node.next_match_loser_id is None:
                elimination_groups.setdefault(distance, []).append(match_id)
        bands: dict[str, tuple[int, int]] = {}
        rank = 2
        for distance in sorted(elimination_groups):
            group = elimination_groups[distance]
            for match_id in group:
                bands[match_id] = (rank, rank + len(group) - 1)
            rank += len(group)
        if rank != len(bracket.teams) + 1:
            raise ValueError("The graph must eliminate every non-champion exactly once.")
        return order, final_id, bands
    def simulate(
        self,
        bracket: TournamentBracket,
        n_simulations: int = 10000,
        manual_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Simulate the remaining fixed graph, preserving confirmed results.

        Overrides force only actual participants and never rewrite played history.
        A forced future winner requires compatible upstream outcomes; contradictory
        scenarios are rejected, not repaired by inserting an eliminated team.
        Unfinished live series are withheld: C0 has no conditional remaining-map law.
        """
        self._validate_simulations(n_simulations)
        manual = manual_overrides or {}
        execution_order, final_id, placement_bands = self._validate_bracket(bracket, manual)
        counts = {team: {cutoff: 0 for cutoff in (1, 2, 3, 4)} for team in bracket.teams}
        ambiguous = {team: set() for team in bracket.teams}
        joint_final_counts: dict[tuple[str, str], int] = defaultdict(int)
        live_probabilities = {
            match_id: self._live_probability(node.team1, node.team2, node.best_of, node.score1, node.score2)
            for match_id, node in bracket.matches.items()
            if node.winner is None and match_id not in manual
            and node.score1 is not None and (node.score1 or node.score2)
        }
        for _ in range(n_simulations):
            state = {match_id: [node.team1, node.team2] for match_id, node in bracket.matches.items()}
            eliminated: set[str] = set()
            for match_id in execution_order:
                node = bracket.matches[match_id]
                team1, team2 = state[match_id]
                if not team1 or not team2 or team1 == team2 or team1 in eliminated or team2 in eliminated:
                    raise ValueError(f"Invalid or eliminated participants in {match_id}.")
                winner = node.winner or manual.get(match_id)
                if winner is not None and winner not in (team1, team2):
                    raise ValueError(f"Winner is not a resolved participant in {match_id}.")
                if winner is None:
                    probability = (
                        live_probabilities[match_id]
                        if match_id in live_probabilities
                        else self.estimate_matchup_probability(team1, team2, node.best_of)
                    )
                    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                        raise ValueError(f"Invalid series probability in {match_id}.")
                    winner = team1 if self._rng.random() < probability else team2
                loser = team2 if winner == team1 else team1
                for target_id, slot, entrant in (
                    (node.next_match_winner_id, node.next_match_winner_slot, winner),
                    (node.next_match_loser_id, node.next_match_loser_slot, loser),
                ):
                    if target_id is not None:
                        existing = state[target_id][slot - 1]
                        if existing is not None and existing != entrant:
                            raise ValueError(f"Resolved entrant contradicts supplied participant in {target_id}.")
                        state[target_id][slot - 1] = entrant
                if node.next_match_loser_id is None:
                    eliminated.add(loser)
                    lower, upper = placement_bands[match_id]
                    for cutoff in (1, 2, 3, 4):
                        if upper <= cutoff:
                            counts[loser][cutoff] += 1
                        elif lower <= cutoff:
                            ambiguous[loser].add(cutoff)
                if match_id == final_id:
                    for cutoff in (1, 2, 3, 4):
                        counts[winner][cutoff] += 1
                    if team1 and team2:
                        pair = tuple(sorted([team1, team2]))
                        joint_final_counts[pair] += 1
            if len(eliminated) != len(bracket.teams) - 1:
                raise ValueError("Simulation did not eliminate every non-champion.")

        results = []
        for team in bracket.teams:
            result: dict[str, Any] = {"team": team}
            for cutoff, key in ((1, "champion_prob"), (2, "top2_prob"), (3, "top3_prob"), (4, "top4_prob")):
                result[key] = None if cutoff in ambiguous[team] else counts[team][cutoff] / n_simulations
            results.append(result)
        results.sort(key=lambda row: row["champion_prob"], reverse=True)
        top_finalists = [
            {"pair": f"{p[0]} vs {p[1]}", "prob": round(cnt / n_simulations, 4)}
            for p, cnt in sorted(joint_final_counts.items(), key=lambda x: -x[1])[:5]
        ]
        return {
            "tournament_id": bracket.id,
            "tournament_name": bracket.name,
            "simulations": n_simulations,
            "standings": results,
            "joint_finalists": top_finalists,
            "probability_model": "Causal-C0",
            "probability_model_version": TOURNAMENT_PROBABILITY_VERSION,
            "provenance": self._provenance(bracket.teams),
            "simulation_scope": {
                "mode": "conditional_current_bracket",
                "pre_event_forecast": False,
                "rules_verified": False,
                "format": "fixed_graph_no_reseeding_no_reset_no_implicit_byes",
                "placement_policy": "elimination_depth_bands_null_when_cutoff_splits_tie",
                "manual_overrides": dict(manual),
            },
            "bracket": {
                match_id: {
                    "id": node.id, "name": node.name, "round_name": node.round_name,
                    "bracket_section": node.bracket_section, "best_of": node.best_of,
                    "team1": node.team1, "team2": node.team2, "winner": node.winner,
                    "score1": node.score1, "score2": node.score2,
                }
                for match_id, node in bracket.matches.items()
            },
        }

@dataclass(frozen=True)
class WorldsTeam:
    """A manually configured Worlds participant and its Swiss draw metadata."""

    name: str
    region: str
    pool: int | None = None


class WorldsSimulator(TournamentSimulator):
    """Unverified 15+4 participant scenario, not an implementation certified against Worlds rules."""

    def simulate_series_winner(
        self,
        team1: str,
        team2: str,
        best_of: int,
        region1: str | None = None,
        region2: str | None = None,
    ) -> str:
        """Simulate one series and return its winner."""
        del region1, region2
        probability = self.estimate_matchup_probability(team1, team2, best_of=best_of)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Invalid series probability.")
        return team1 if self._rng.random() < probability else team2

    def simulate_play_in(self, teams: Sequence[WorldsTeam]) -> str:
        """Simulate the configured four-team Bo5 Play-In, with no grand-final reset."""
        if len(teams) != 4 or len({self._team_key(team.name) for team in teams}) != 4:
            raise ValueError("Play-In requires four distinct participants.")
        bracket = list(teams)
        self._rng.shuffle(bracket)

        upper_winner_1 = self.simulate_series_winner(bracket[0].name, bracket[1].name, best_of=5)
        upper_loser_1 = bracket[1].name if upper_winner_1 == bracket[0].name else bracket[0].name
        upper_winner_2 = self.simulate_series_winner(bracket[2].name, bracket[3].name, best_of=5)
        upper_loser_2 = bracket[3].name if upper_winner_2 == bracket[2].name else bracket[2].name

        lower_round_winner = self.simulate_series_winner(upper_loser_1, upper_loser_2, best_of=5)
        upper_final_winner = self.simulate_series_winner(upper_winner_1, upper_winner_2, best_of=5)
        upper_final_loser = upper_winner_2 if upper_final_winner == upper_winner_1 else upper_winner_1
        lower_final_winner = self.simulate_series_winner(
            lower_round_winner,
            upper_final_loser,
            best_of=5,
        )
        return self.simulate_series_winner(upper_final_winner, lower_final_winner, best_of=5)

    def simulate_swiss_round(
        self,
        pool: Sequence[str],
        best_of: int,
        played_pairs: set[frozenset[str]],
    ) -> tuple[list[str], list[str]]:
        """Uniformly sample a legal perfect matching; never silently allow a rematch.

        Count completions rather than using a capped shuffle retry or a biased
        first-feasible search. Record buckets in this scenario contain at most 16 teams.
        """
        self._validate_best_of(best_of)
        names = list(pool)
        keys = [self._team_key(team) for team in names]
        if len(names) > 16 or len(names) % 2 or len(set(keys)) != len(keys) or any(not key for key in keys):
            raise ValueError("Swiss buckets require an even number of distinct teams, at most 16.")
        allowed = {
            (first, second): frozenset((names[first], names[second])) not in played_pairs
            for first in range(len(names)) for second in range(first + 1, len(names))
        }

        @lru_cache(maxsize=None)
        def completions(mask: int) -> int:
            if not mask:
                return 1
            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            rest = mask ^ first_bit
            return sum(
                completions(rest ^ (1 << second))
                for second in range(first + 1, len(names))
                if rest & (1 << second) and allowed[first, second]
            )

        mask = (1 << len(names)) - 1
        if not completions(mask):
            raise ValueError("No legal no-rematch Swiss pairing exists in this record bucket.")
        pairs: list[tuple[str, str]] = []
        while mask:
            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            rest = mask ^ first_bit
            draw = self._rng.randrange(completions(mask))
            for second in range(first + 1, len(names)):
                if not rest & (1 << second) or not allowed[first, second]:
                    continue
                count = completions(rest ^ (1 << second))
                if draw < count:
                    pairs.append((names[first], names[second]))
                    mask = rest ^ (1 << second)
                    break
                draw -= count
        winners: list[str] = []
        losers: list[str] = []
        for team1, team2 in pairs:
            winner = self.simulate_series_winner(team1, team2, best_of)
            loser = team2 if winner == team1 else team1
            played_pairs.add(frozenset((team1, team2)))
            winners.append(winner)
            losers.append(loser)
        return winners, losers

    @staticmethod
    def validate_participants(
        direct_teams: Sequence[WorldsTeam],
        play_in_teams: Sequence[WorldsTeam],
        play_in_winner_pool: int,
    ) -> None:
        """Validate the fixed 15 direct + 4 Play-In participant contract."""
        if len(direct_teams) != 15:
            raise ValueError("Worlds requires exactly 15 direct Swiss participants.")
        if len(play_in_teams) != 4:
            raise ValueError("Worlds requires exactly 4 Play-In participants.")
        if type(play_in_winner_pool) is not int or play_in_winner_pool not in {1, 2, 3, 4}:
            raise ValueError("The Play-In qualifier must be assigned to Swiss pool 1, 2, 3, or 4.")

        all_teams = [*direct_teams, *play_in_teams]
        team_keys = [canonical_team_key(team.name) for team in all_teams]
        if any(not key for key in team_keys):
            raise ValueError("Every Worlds participant requires a team name.")
        if len(set(team_keys)) != len(team_keys):
            raise ValueError("A team cannot occupy more than one Worlds slot.")
        if any(not team.region.strip() for team in all_teams):
            raise ValueError("Every Worlds participant requires a region.")
        if any(type(team.pool) is not int or team.pool not in {1, 2, 3, 4} for team in direct_teams):
            raise ValueError("Every direct Swiss participant requires pool 1, 2, 3, or 4.")

        pool_counts = {pool: sum(team.pool == pool for team in direct_teams) for pool in range(1, 5)}
        if pool_counts[play_in_winner_pool] != 3 or any(
            count != 4 for pool, count in pool_counts.items() if pool != play_in_winner_pool
        ):
            raise ValueError(
                "Direct Swiss slots must fill three pools with four teams and the Play-In pool with three."
            )

    def simulate_worlds(
        self,
        direct_teams: Sequence[WorldsTeam],
        play_in_teams: Sequence[WorldsTeam],
        play_in_winner_pool: int,
        n_simulations: int = 5000,
    ) -> dict[str, Any]:
        """Simulate a manually configured Worlds from Play-In through the Bo5 final."""
        self._validate_simulations(n_simulations)
        self.validate_participants(direct_teams, play_in_teams, play_in_winner_pool)

        all_teams = [*direct_teams, *play_in_teams]
        direct_names = [team.name for team in direct_teams]
        qualifier_counts = {team.name: 0 for team in all_teams}
        swiss_advance_counts = {team.name: 0 for team in all_teams}
        knockout_top4_counts = {team.name: 0 for team in all_teams}
        knockout_top2_counts = {team.name: 0 for team in all_teams}
        champion_counts = {team.name: 0 for team in all_teams}
        metadata = {team.name: team for team in all_teams}

        for direct_team in direct_teams:
            qualifier_counts[direct_team.name] = n_simulations

        for _ in range(n_simulations):
            play_in_qualifier = self.simulate_play_in(play_in_teams)
            qualifier_counts[play_in_qualifier] += 1
            qualifier_metadata = metadata[play_in_qualifier]
            participant_teams = [
                *direct_teams,
                WorldsTeam(
                    name=play_in_qualifier,
                    region=qualifier_metadata.region,
                    pool=play_in_winner_pool,
                ),
            ]
            records = {team.name: [0, 0] for team in participant_teams}
            played_pairs: set[frozenset[str]] = set()
            advanced_teams: list[str] = []
            next_buckets: dict[tuple[int, int], list[str]] = {}

            first_round_pools = {
                pool: [team.name for team in participant_teams if team.pool == pool]
                for pool in range(1, 5)
            }
            for higher_pool, lower_pool in ((1, 4), (2, 3)):
                self._rng.shuffle(first_round_pools[higher_pool])
                self._rng.shuffle(first_round_pools[lower_pool])
                for team1, team2 in zip(
                    first_round_pools[higher_pool],
                    first_round_pools[lower_pool],
                    strict=True,
                ):
                    winner = self.simulate_series_winner(team1, team2, best_of=1)
                    loser = team2 if winner == team1 else team1
                    played_pairs.add(frozenset((team1, team2)))
                    records[winner][0] += 1
                    records[loser][1] += 1
                    next_buckets.setdefault((1, 0), []).append(winner)
                    next_buckets.setdefault((0, 1), []).append(loser)

            buckets = next_buckets
            for _round_number in range(2, 6):
                next_buckets = {}
                for (wins, losses), pool in buckets.items():
                    best_of = 3 if wins == 2 or losses == 2 else 1
                    winners, losers = self.simulate_swiss_round(pool, best_of, played_pairs)
                    for winner in winners:
                        records[winner][0] += 1
                        new_wins, new_losses = records[winner]
                        if new_wins == 3:
                            advanced_teams.append(winner)
                        else:
                            next_buckets.setdefault((new_wins, new_losses), []).append(winner)
                    for loser in losers:
                        records[loser][1] += 1
                        new_wins, new_losses = records[loser]
                        if new_losses < 3:
                            next_buckets.setdefault((new_wins, new_losses), []).append(loser)
                buckets = next_buckets
                if len(advanced_teams) == 8:
                    break

            if len(advanced_teams) != 8 or buckets or any(
                wins != 3 and losses != 3 for wins, losses in records.values()
            ):
                raise ValueError("Swiss stage did not resolve exactly eight qualifiers and eight eliminations.")

            for team in advanced_teams:
                swiss_advance_counts[team] += 1

            self._rng.shuffle(advanced_teams)
            quarterfinal_winners = [
                self.simulate_series_winner(advanced_teams[index], advanced_teams[index + 1], best_of=5)
                for index in range(0, 8, 2)
            ]
            for team in quarterfinal_winners:
                knockout_top4_counts[team] += 1

            semifinal_winners = [
                self.simulate_series_winner(
                    quarterfinal_winners[index],
                    quarterfinal_winners[index + 1],
                    best_of=5,
                )
                for index in range(0, 4, 2)
            ]
            for team in semifinal_winners:
                knockout_top2_counts[team] += 1
            champion_counts[
                self.simulate_series_winner(semifinal_winners[0], semifinal_winners[1], best_of=5)
            ] += 1

        standings = []
        for team in all_teams:
            standings.append(
                {
                    "team": team.name,
                    "region": team.region,
                    "stage": "direct_swiss" if team.name in direct_names else "play_in",
                    "pool": team.pool if team.name in direct_names else None,
                    "play_in_qualifier_prob": qualifier_counts[team.name] / n_simulations,
                    "champion_prob": champion_counts[team.name] / n_simulations,
                    "top2_prob": knockout_top2_counts[team.name] / n_simulations,
                    "top4_prob": knockout_top4_counts[team.name] / n_simulations,
                    "top8_swiss_prob": swiss_advance_counts[team.name] / n_simulations,
                }
            )

        standings.sort(key=lambda standing: standing["champion_prob"], reverse=True)
        return {
            "tournament_id": "worlds_2026",
            "tournament_name": "League of Legends World Championship 2026",
            "format": "play_in_double_elimination_bo5_swiss_and_knockout",
            "simulations": n_simulations,
            "provenance": self._provenance([team.name for team in all_teams]),
            "simulation_scope": {
                "mode": "unverified_user_configured_scenario",
                "rules_verified": False,
                "pre_event_forecast": False,
                "play_in": "four_teams_bo5_double_elimination_no_reset",
                "swiss": "three_wins_or_losses_bo3_at_advancement_or_elimination",
                "first_round_draw": "pool1_vs_pool4_pool2_vs_pool3_no_region_constraint",
                "later_draw": "uniform_legal_no_rematch_matching_within_record_bucket",
                "knockout_draw": "unseeded_random_no_record_constraint",
                "limitations": [
                    "No locally verified official 2026 rules or participant qualification evidence.",
                    "Region restrictions and Swiss-record knockout seeding are not implemented.",
                    "No live results, dynamic reseeding, bracket reset, or strength updates.",
                ],
            },
            "teams": [team.name for team in all_teams],
            "direct_teams": [
                {"team": team.name, "region": team.region, "pool": team.pool} for team in direct_teams
            ],
            "play_in_teams": [{"team": team.name, "region": team.region} for team in play_in_teams],
            "play_in_winner_pool": play_in_winner_pool,
            "standings": standings,
        }
