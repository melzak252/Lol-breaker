"""Advanced Multi-Engine Tournament Simulation Framework for League of Legends.

Integrates:
1. Riot Sequential Swiss Stage Draw Engine (``src.models.tournament_swiss_lottery``):
   Official ball-by-ball sequential draw lottery with sliding-slot mechanics, pool-by-pool
   descending resolution, regional separation in Round 1, and no-rematch backtracking.
2. Hard Fearless Series Simulation Engine (``src.models.fearless_series_engine``):
   Sequential map-by-map simulation under official Hard Fearless Draft rules with Blue/Red side
   choice dynamics and champion pool depletion comfort penalties.
3. In-Path Tournament Bayesian Momentum Filter (``src.models.tournament_bayesian_momentum``):
   State-space Bayesian filter tracking in-tournament team momentum, confidence shifts, and
   meta-adaptation drift across sequential rounds with martingale preservation.
4. Exact Subtree Hybrid Knockout DP (``src.models.tournament_exact``):
   Analytical dynamic programming integration for single-elimination knockout brackets,
   yielding exact Rao-Blackwellized tournament distributions over sampled Swiss qualification paths.
5. Epistemic Temperature Softening and Glicko Anchor Shrinkage:
   Blends unadjusted/model base probabilities with Glicko-2 ratings via shrinkage lambda and
   temperature softening on logit odds while preserving exact side anti-symmetry.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import random
from typing import Any

import numpy as np

from src.models.tournament_swiss_lottery import SwissMatchup, simulate_swiss_round_draws
from src.models.fearless_series_engine import (
    FearlessSeriesResult,
    simulate_fearless_series_maps,
    series_prob_to_map_prob,
)
from src.models.tournament_bayesian_momentum import (
    InPathTournamentBayesianFilter,
    TeamMomentumState,
)
from src.models.tournament_exact import _knockout_dp
from src.ratings.glicko2_core import (
    Glicko2State,
    expected_score,
    series_win_probability,
)

# Optional bracket definitions from betting_app if available
try:
    from betting_app.services.tournament_service import (
        BracketMatchNode,
        TournamentBracket,
        get_lck_2026_playoffs_bracket,
        get_lcs_2026_championship_bracket,
        get_lec_2026_summer_playoffs_bracket,
        get_lpl_2026_split3_playoffs_bracket,
    )
except ImportError:
    TournamentBracket = Any  # type: ignore[misc, assignment]
    BracketMatchNode = Any  # type: ignore[misc, assignment]
    get_lck_2026_playoffs_bracket = None  # type: ignore[assignment]
    get_lcs_2026_championship_bracket = None  # type: ignore[assignment]
    get_lec_2026_summer_playoffs_bracket = None  # type: ignore[assignment]
    get_lpl_2026_split3_playoffs_bracket = None  # type: ignore[assignment]


__all__ = [
    "AdvancedTournamentSimulator",
    "TournamentSimulationResult",
    "StandingsList",
    "JointFinalistsList",
    "SeriesLengthDistribution",
    "RegionDistribution",
    "blend_probability",
    "DEFAULT_WORLDS_16_TEAMS",
    "DEFAULT_WORLDS_REGIONS",
    "DEFAULT_WORLDS_RATINGS",
    "DEFAULT_WORLDS_ROSTERS",
]

# Standard 16 participants for Worlds Swiss stage
DEFAULT_WORLDS_16_TEAMS: tuple[str, ...] = (
    "Gen.G",
    "Anyone's Legend",
    "Team Liquid",
    "G2 Esports",
    "Team Secret Whales",
    "Los Grandes",
    "Hanwha Life Esports",
    "Bilibili Gaming",
    "LYON",
    "Movistar KOI",
    "T1",
    "Top Esports",
    "CTBC Flying Oyster",
    "Dplus KIA",
    "Invictus Gaming",
    "Karmine Corp",
)

# Regional affiliations for Worlds participants
DEFAULT_WORLDS_REGIONS: dict[str, str] = {
    "Gen.G": "LCK",
    "Hanwha Life Esports": "LCK",
    "T1": "LCK",
    "Dplus KIA": "LCK",
    "Anyone's Legend": "LPL",
    "Bilibili Gaming": "LPL",
    "Top Esports": "LPL",
    "Invictus Gaming": "LPL",
    "G2 Esports": "LEC",
    "Movistar KOI": "LEC",
    "Karmine Corp": "LEC",
    "Team Liquid": "LCS",
    "LYON": "LCS",
    "Cloud9": "LCS",
    "Team Secret Whales": "LCP",
    "CTBC Flying Oyster": "LCP",
    "Los Grandes": "CBLOL",
}

# Competitive baseline ratings (Elo / Glicko scale)
DEFAULT_WORLDS_RATINGS: dict[str, float] = {
    "Gen.G": 1740.0,
    "Bilibili Gaming": 1720.0,
    "T1": 1690.0,
    "Hanwha Life Esports": 1680.0,
    "Top Esports": 1660.0,
    "Anyone's Legend": 1620.0,
    "Dplus KIA": 1610.0,
    "G2 Esports": 1580.0,
    "LGD Gaming": 1560.0,
    "Team Liquid": 1520.0,
    "LYON": 1510.0,
    "Karmine Corp": 1500.0,
    "Cloud9": 1490.0,
    "Team Secret Whales": 1480.0,
    "CTBC Flying Oyster": 1460.0,
    "Los Grandes": 1420.0,
}

# Baseline 5-man rosters [TOP, JUG, MID, ADC, SUP]
DEFAULT_WORLDS_ROSTERS: dict[str, list[str]] = {
    "Gen.G": ["Kiin", "Canyon", "Chovy", "Ruler", "Duro"],
    "Bilibili Gaming": ["Bin", "XUN", "Knight", "Viper", "ON"],
    "T1": ["Doran", "Oner", "Faker", "Peyz", "Keria"],
    "Hanwha Life Esports": ["Zeus", "Kanavi", "Zeka", "Gumayusi", "Delight"],
    "Top Esports": ["ZUIAN", "Tian", "Creme", "JackeyLove", "Zhuo"],
    "Anyone's Legend": ["Breathe", "Tarzan", "Shanks", "Hope", "Kael"],
    "Dplus KIA": ["Siwoo", "Lucid", "ShowMaker", "Smash", "Career"],
    "G2 Esports": ["BrokenBlade", "SkewMond", "Caps", "Hans Sama", "Labrov"],
    "Team Liquid": ["Morgan", "Josedeodo", "Quid", "Yeon", "CoreJJ"],
    "LYON": ["Dhokla", "Inspired", "Saint", "Berserker", "Isles"],
    "Karmine Corp": ["Canna", "Yike", "Kyeahoo", "Caliste", "Busio"],
    "Invictus Gaming": ["TheShy", "Wei", "RooKie", "JiaQi", "meiko"],
    "Cloud9": ["Thanatos", "Blaber", "Loki", "Tactical", "Vulcan"],
    "Movistar KOI": ["Myrwn", "Elyoya", "jojopyun", "Supa", "Alvaro"],
    "Team Secret Whales": ["Pun", "Hizto", "Dire", "Eddie", "Bie"],
    "CTBC Flying Oyster": ["Rest", "Shad0w", "Pout", "Doggo", "Kino"],
    "Los Grandes": ["Zest", "Curse", "Feisty", "Duduhh", "Ackerman"],
    "MVK Esports": ["Hasmed", "Yushin", "Kuro", "Slayder", "Palette"],
    "FURIA": ["Guigo", "Tatu", "Tutsz", "Ayu", "Jojo"],
}


def blend_probability(
    base_p: float,
    glicko_p: float,
    shrinkage_lambda: float = 0.20,
    temperature: float = 1.05,
    eps: float = 1e-6,
) -> float:
    """Blend base series probability with Glicko anchor under epistemic softening.

    Formulation:
        logit_blended = (1 - lambda) * logit(base_p) + lambda * logit(glicko_p)
        logit_soft = logit_blended / temperature
        p_eff = sigmoid(logit_soft)

    Preserves exact side anti-symmetry:
        blend_probability(p, q, lam, temp) + blend_probability(1-p, 1-q, lam, temp) == 1.0.

    Parameters
    ----------
    base_p : float
        Unadjusted series win probability for Team A.
    glicko_p : float
        Glicko-2 anchor win probability for Team A.
    shrinkage_lambda : float
        Shrinkage parameter in [0.0, 1.0] blending towards the Glicko anchor.
    temperature : float
        Epistemic softening factor >= 1.0 scaling logits towards uncertainty (0.5).
    eps : float
        Clamping epsilon for numerical stability.

    Returns
    -------
    float
        Blended, softened win probability for Team A in [eps, 1.0 - eps].
    """
    p1 = min(max(float(base_p), eps), 1.0 - eps)
    p2 = min(max(float(glicko_p), eps), 1.0 - eps)

    z_base = math.log(p1 / (1.0 - p1))
    z_glicko = math.log(p2 / (1.0 - p2))

    lam = float(np.clip(shrinkage_lambda, 0.0, 1.0))
    z_blended = (1.0 - lam) * z_base + lam * z_glicko

    temp = max(float(temperature), 1e-4)
    z_soft = z_blended / temp

    p_eff = 1.0 / (1.0 + math.exp(-z_soft))
    return float(np.clip(p_eff, eps, 1.0 - eps))


def _simulate_standard_series_maps(
    team_a: str,
    team_b: str,
    best_of: int,
    base_series_prob: float,
    rng: random.Random,
) -> tuple[str, str, int, int, str]:
    """Simulate a standard non-fearless series via sequential map Bernoulli trials.

    Returns
    -------
    tuple[str, str, int, int, str]
        (winner, loser, score_winner, score_loser, scoreline)
    """
    if best_of == 1:
        win_a = rng.random() < base_series_prob
        winner = team_a if win_a else team_b
        loser = team_b if win_a else team_a
        return winner, loser, 1, 0, "1-0"

    target_wins = (best_of // 2) + 1
    p_map = series_prob_to_map_prob(base_series_prob, best_of)

    score_a = 0
    score_b = 0
    while score_a < target_wins and score_b < target_wins:
        if rng.random() < p_map:
            score_a += 1
        else:
            score_b += 1

    winner = team_a if score_a == target_wins else team_b
    loser = team_b if winner == team_a else team_a
    sw = max(score_a, score_b)
    sl = min(score_a, score_b)
    return winner, loser, sw, sl, f"{sw}-{sl}"


def _sample_fallback_swiss_draws(
    active_records: dict[str, tuple[int, int]],
    past_matchups: set[frozenset[str]],
    round_num: int,
    team_regions: dict[str, str] | None,
    rng: random.Random,
) -> list[SwissMatchup]:
    """Fallback Swiss pairings when Riot ball-by-ball sequential draw is disabled."""
    active = {t: rec for t, rec in active_records.items() if rec[0] < 3 and rec[1] < 3}
    if not active:
        return []

    pools: dict[tuple[int, int], list[str]] = {}
    for team, rec in active.items():
        pools.setdefault(rec, []).append(team)

    sorted_records = sorted(pools.keys(), key=lambda r: (-r[0], r[1]))
    matchups: list[SwissMatchup] = []

    for record in sorted_records:
        pool_teams = list(pools[record])
        rng.shuffle(pool_teams)
        pool_name = f"{record[0]}-{record[1]}"
        best_of = 3 if (record[0] >= 2 or record[1] >= 2) else 1

        def find_legal_pairs(remaining: list[str]) -> list[tuple[str, str]] | None:
            if not remaining:
                return []
            first = remaining[0]
            candidates = []
            for cand in remaining[1:]:
                if round_num == 1 and team_regions is not None:
                    if team_regions.get(first) == team_regions.get(cand):
                        continue
                elif round_num > 1:
                    if frozenset({first, cand}) in past_matchups:
                        continue
                candidates.append(cand)

            rng.shuffle(candidates)
            for cand in candidates:
                sub = [t for t in remaining[1:] if t != cand]
                res = find_legal_pairs(sub)
                if res is not None:
                    return [(first, cand)] + res
            return None

        paired = find_legal_pairs(pool_teams)
        if paired is None:
            # Fallback waiver: minimal constraint pairing
            paired = [(pool_teams[i], pool_teams[i + 1]) for i in range(0, len(pool_teams), 2)]

        for ta, tb in paired:
            matchups.append(
                SwissMatchup(
                    team_a=ta,
                    team_b=tb,
                    pool=pool_name,
                    round_num=round_num,
                    best_of=best_of,
                )
            )

    return matchups


class StandingsList(list):
    """Subclass of list supporting dual list iteration and team-name dictionary access."""

    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        super().__init__(items)
        self._by_team: dict[str, dict[str, Any]] = {
            item["team"]: item for item in items if "team" in item
        }

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._by_team[item]
        return super().__getitem__(item)

    def get(self, team: str, default: Any = None) -> Any:
        return self._by_team.get(team, default)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return dict(self._by_team)


class JointFinalistsList(list):
    """Subclass of list supporting dual list iteration and pair-tuple / pair-string lookup."""

    def __init__(self, items: Sequence[dict[str, Any]]) -> None:
        super().__init__(items)
        self._by_key: dict[Any, float] = {}
        for item in items:
            prob = float(item.get("prob", 0.0))
            if "pair" in item:
                self._by_key[item["pair"]] = prob
            if "teams" in item:
                t1, t2 = item["teams"]
                self._by_key[(t1, t2)] = prob
                self._by_key[(t2, t1)] = prob
                self._by_key[frozenset({t1, t2})] = prob

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (str, tuple, frozenset)):
            return self._by_key[item]
        return super().__getitem__(item)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._by_key.get(key, default)


class SeriesLengthDistribution(dict):
    """Dictionary supporting nested Bo3/Bo5 queries and flat scoreline queries."""

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, int):
                if key in (2, 3):
                    return self.get("Bo3", {})
                elif key in (4, 5):
                    return self.get("Bo5", {})
            raise


class RegionDistribution(dict):
    """Dictionary supporting integer and string key lookup for distinct region counts."""

    def __getitem__(self, key: Any) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, str) and key.isdigit():
                return super().__getitem__(int(key))
            elif isinstance(key, int):
                return super().__getitem__(str(key))
            raise


@dataclass
class TournamentSimulationResult:
    """Consolidated outcome distributions from tournament Monte Carlo simulation.

    Attributes
    ----------
    standings : StandingsList
        List of standings records sorted by champion_prob descending:
        each row contains {'team', 'champion_prob', 'top2_prob', 'top3_prob', 'top4_prob'}.
    joint_finalists : JointFinalistsList
        Top joint final pairings with their probabilities:
        each row contains {'pair', 'teams', 'prob'}.
    swiss_qualifiers_prob : dict[str, float] | None
        Probability of each team advancing to Knockouts (Worlds only).
    region_distribution : dict[int, float] | None
        Probability distribution over 2, 3, 4, 5 distinct regions in Top 8 (Worlds only).
    series_length_distribution : SeriesLengthDistribution
        Bo3 and Bo5 scoreline distributions ('3-0', '3-1', '3-2', '2-0', '2-1').
    simulations : int
        Number of complete tournament rollouts executed.
    seed : int
        Deterministic random seed used for simulation.
    tournament : str
        Tournament identifier or region name.
    parameters : dict[str, Any]
        Configuration switches and hyperparameter settings used in the simulation.
    """

    standings: StandingsList
    joint_finalists: JointFinalistsList
    swiss_qualifiers_prob: dict[str, float] | None
    region_distribution: RegionDistribution | None
    series_length_distribution: SeriesLengthDistribution
    simulations: int
    seed: int
    tournament: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "standings": list(self.standings),
            "joint_finalists": list(self.joint_finalists),
            "swiss_qualifiers_prob": self.swiss_qualifiers_prob,
            "region_distribution": dict(self.region_distribution) if self.region_distribution else None,
            "series_length_distribution": dict(self.series_length_distribution),
            "simulations": self.simulations,
            "seed": self.seed,
            "tournament": self.tournament,
            "parameters": self.parameters,
        }


class AdvancedTournamentSimulator:
    """Comprehensive tournament simulator combining Swiss Lottery, Fearless Draft,

    Bayesian Momentum, and Hybrid Knockout DP.
    """

    def __init__(
        self,
        use_fearless: bool = True,
        use_bayesian_momentum: bool = True,
        use_hybrid_knockout: bool = True,
        use_riot_lottery: bool = True,
        shrinkage_lambda: float = 0.20,
        temperature: float = 1.05,
        seed: int | None = None,
        *,
        learning_rate: float = 0.12,
        adaptation_volatility: float = 0.04,
        max_abs_logit_offset: float = 0.85,
        map_margin_weight: float = 0.03,
        default_ratings: Mapping[str, float] | None = None,
        default_rosters: Mapping[str, list[str]] | None = None,
        default_regions: Mapping[str, str] | None = None,
        default_glicko_rds: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize simulator configuration and hyperparameters.

        Parameters
        ----------
        use_fearless : bool, default=True
            Simulate sequential maps under Hard Fearless Draft rules with side choice
            and comfort pool depletion penalties.
        use_bayesian_momentum : bool, default=True
            Track in-path Bayesian momentum and meta adaptation drift across series.
        use_hybrid_knockout : bool, default=True
            Use exact subtree dynamic programming (_knockout_dp) for single-elimination
            knockout brackets rather than Monte Carlo.
        use_riot_lottery : bool, default=True
            Use Riot's official ball-by-ball sequential draw lottery with sliding slots
            for Worlds Swiss Stage.
        shrinkage_lambda : float, default=0.20
            Logit shrinkage parameter blending model base probability with Glicko-2 anchor.
        temperature : float, default=1.05
            Epistemic temperature softening factor on logit odds (T >= 1.0).
        seed : int | None, optional
            Deterministic random seed for full reproducibility.
        learning_rate : float, default=0.12
            Bayesian momentum innovation learning rate (eta).
        adaptation_volatility : float, default=0.04
            Bayesian momentum meta adaptation volatility standard deviation.
        max_abs_logit_offset : float, default=0.85
            Clamping limit on accumulated team logit momentum offsets.
        map_margin_weight : float, default=0.03
            Bonus offset per net map differential in Bayesian momentum.
        default_ratings : Mapping[str, float] | None, optional
            Fallback baseline ratings per team.
        default_rosters : Mapping[str, list[str]] | None, optional
            Fallback 5-player rosters per team.
        default_regions : Mapping[str, str] | None, optional
            Fallback regional identifiers per team.
        default_glicko_rds : Mapping[str, float] | None, optional
            Fallback Glicko rating deviations per team.
        """
        self.use_fearless = bool(use_fearless)
        self.use_bayesian_momentum = bool(use_bayesian_momentum)
        self.use_hybrid_knockout = bool(use_hybrid_knockout)
        self.use_riot_lottery = bool(use_riot_lottery)
        self.shrinkage_lambda = float(np.clip(shrinkage_lambda, 0.0, 1.0))
        self.temperature = max(float(temperature), 1e-4)
        self.seed = int(seed) if seed is not None else None

        self.learning_rate = float(learning_rate)
        self.adaptation_volatility = float(adaptation_volatility)
        self.max_abs_logit_offset = float(max_abs_logit_offset)
        self.map_margin_weight = float(map_margin_weight)

        self.default_ratings = dict(default_ratings) if default_ratings else {}
        self.default_rosters = dict(default_rosters) if default_rosters else {}
        self.default_regions = dict(default_regions) if default_regions else {}
        self.default_glicko_rds = dict(default_glicko_rds) if default_glicko_rds else {}

    def get_matchup_probability(
        self,
        team_a: str,
        team_b: str,
        best_of: int = 1,
        *,
        base_probabilities: Mapping[tuple[str, str, int], float]
        | Mapping[tuple[str, str], float]
        | None = None,
        team_ratings: Mapping[str, float] | None = None,
        glicko_states: Mapping[str, Glicko2State] | Mapping[str, tuple[float, float]] | None = None,
        shrinkage_lambda: float | None = None,
        temperature: float | None = None,
    ) -> float:
        """Compute effective series win probability with Glicko shrinkage and softening."""
        s_lambda = self.shrinkage_lambda if shrinkage_lambda is None else shrinkage_lambda
        temp = self.temperature if temperature is None else temperature

        # 1. Base probability
        base_p: float | None = None
        if base_probabilities is not None:
            if (team_a, team_b, best_of) in base_probabilities:
                base_p = float(base_probabilities[(team_a, team_b, best_of)])
            elif (team_b, team_a, best_of) in base_probabilities:
                base_p = 1.0 - float(base_probabilities[(team_b, team_a, best_of)])
            elif (team_a, team_b) in base_probabilities:
                p_map = float(base_probabilities[(team_a, team_b)])
                base_p = series_win_probability(p_map, best_of) if best_of > 1 else p_map
            elif (team_b, team_a) in base_probabilities:
                p_map = 1.0 - float(base_probabilities[(team_b, team_a)])
                base_p = series_win_probability(p_map, best_of) if best_of > 1 else p_map

        if base_p is None:
            ratings = dict(self.default_ratings)
            if team_ratings:
                ratings.update(team_ratings)
            ra = ratings.get(team_a, DEFAULT_WORLDS_RATINGS.get(team_a, 1500.0))
            rb = ratings.get(team_b, DEFAULT_WORLDS_RATINGS.get(team_b, 1500.0))
            p_map = float(1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0)))
            base_p = series_win_probability(p_map, best_of) if best_of > 1 else p_map

        # 2. Glicko anchor
        ra_g, rda_g = 1500.0, 100.0
        rb_g, rdb_g = 1500.0, 100.0

        if glicko_states and team_a in glicko_states:
            va = glicko_states[team_a]
            ra_g, rda_g = (va.rating, va.rd) if isinstance(va, Glicko2State) else va
        elif team_ratings and team_a in team_ratings:
            ra_g = team_ratings[team_a]
            rda_g = self.default_glicko_rds.get(team_a, 100.0)
        else:
            ra_g = self.default_ratings.get(team_a, DEFAULT_WORLDS_RATINGS.get(team_a, 1500.0))
            rda_g = self.default_glicko_rds.get(team_a, 100.0)

        if glicko_states and team_b in glicko_states:
            vb = glicko_states[team_b]
            rb_g, rdb_g = (vb.rating, vb.rd) if isinstance(vb, Glicko2State) else vb
        elif team_ratings and team_b in team_ratings:
            rb_g = team_ratings[team_b]
            rdb_g = self.default_glicko_rds.get(team_b, 100.0)
        else:
            rb_g = self.default_ratings.get(team_b, DEFAULT_WORLDS_RATINGS.get(team_b, 1500.0))
            rdb_g = self.default_glicko_rds.get(team_b, 100.0)

        p_g_map = expected_score(ra_g, rda_g, rb_g, rdb_g)
        glicko_p = series_win_probability(p_g_map, best_of) if best_of > 1 else p_g_map

        # 3. Shrinkage and temperature softening
        return blend_probability(base_p, glicko_p, s_lambda, temp)

    def simulate_worlds(
        self,
        teams: Sequence[str] | None = None,
        team_regions: Mapping[str, str] | None = None,
        team_ratings: Mapping[str, float] | None = None,
        team_rosters: Mapping[str, list[str]] | None = None,
        glicko_states: Mapping[str, Glicko2State] | Mapping[str, tuple[float, float]] | None = None,
        base_probabilities: Mapping[tuple[str, str, int], float]
        | Mapping[tuple[str, str], float]
        | None = None,
        initial_pairings: Sequence[tuple[str, str]] | None = None,
        n_simulations: int = 1000,
        seed: int | None = None,
        *,
        use_fearless: bool | None = None,
        use_bayesian_momentum: bool | None = None,
        use_hybrid_knockout: bool | None = None,
        use_riot_lottery: bool | None = None,
        shrinkage_lambda: float | None = None,
        temperature: float | None = None,
    ) -> TournamentSimulationResult:
        """Execute full 16-team League of Legends World Championship simulation.

        Combines:
        1. 16-team Swiss Stage: 5 sequential rounds using Riot lottery (or fallback),
           Bo1 non-decider games, Bo3 qualification/elimination series, and side choice.
        2. In-Path Bayesian Momentum Filter tracking team trajectory across Swiss.
        3. Quarterfinal pairings respecting 3-0 vs 3-2 seeding in opposite bracket halves.
        4. Knockout Stage: evaluated via exact subtree DP (Hybrid) or Monte Carlo.

        Returns
        -------
        TournamentSimulationResult
            Contains standings, joint finalists, swiss qualifier probabilities,
            distinct region distribution, and series length distributions.
        """
        fearless = self.use_fearless if use_fearless is None else bool(use_fearless)
        bayesian = self.use_bayesian_momentum if use_bayesian_momentum is None else bool(use_bayesian_momentum)
        hybrid = self.use_hybrid_knockout if use_hybrid_knockout is None else bool(use_hybrid_knockout)
        riot_lottery = self.use_riot_lottery if use_riot_lottery is None else bool(use_riot_lottery)
        s_lambda = self.shrinkage_lambda if shrinkage_lambda is None else float(shrinkage_lambda)
        temp = self.temperature if temperature is None else float(temperature)

        sim_seed = self.seed if seed is None else int(seed)
        if sim_seed is None:
            sim_seed = random.randint(0, 2**31 - 1)

        team_list = list(DEFAULT_WORLDS_16_TEAMS) if teams is None else list(teams)
        if len(team_list) != 16:
            raise ValueError(f"Worlds simulation requires exactly 16 teams, got {len(team_list)}")

        regions = dict(DEFAULT_WORLDS_REGIONS)
        if self.default_regions:
            regions.update(self.default_regions)
        if team_regions:
            regions.update(team_regions)

        ratings = dict(DEFAULT_WORLDS_RATINGS)
        if self.default_ratings:
            ratings.update(self.default_ratings)
        if team_ratings:
            ratings.update(team_ratings)

        rosters = dict(DEFAULT_WORLDS_ROSTERS)
        if self.default_rosters:
            rosters.update(self.default_rosters)
        if team_rosters:
            rosters.update(team_rosters)

        # Precompute neutral Bo5 base probability matrix for hybrid DP
        n_teams = len(team_list)
        base_bo5_matrix = np.full((n_teams, n_teams), 0.5, dtype=float)
        team_idx = {t: i for i, t in enumerate(team_list)}
        for i, ta in enumerate(team_list):
            for j, tb in enumerate(team_list):
                if i != j:
                    base_bo5_matrix[i, j] = self.get_matchup_probability(
                        ta,
                        tb,
                        best_of=5,
                        base_probabilities=base_probabilities,
                        team_ratings=ratings,
                        glicko_states=glicko_states,
                        shrinkage_lambda=s_lambda,
                        temperature=temp,
                    )

        # Accumulators
        champ_acc: dict[str, float] = {t: 0.0 for t in team_list}
        top2_acc: dict[str, float] = {t: 0.0 for t in team_list}
        top4_acc: dict[str, float] = {t: 0.0 for t in team_list}
        qual_acc: dict[str, float] = {t: 0.0 for t in team_list}
        region_acc: Counter[int] = Counter()
        joint_final_acc: Counter[tuple[str, str]] = Counter()
        bo3_counts: Counter[str] = Counter()
        bo5_counts: Counter[str] = Counter()

        master_rng = random.Random(sim_seed)

        def run_single_series(
            team_a: str,
            team_b: str,
            best_of: int,
            p_match: float,
            rng: random.Random,
        ) -> tuple[str, str, int, int, str]:
            if fearless:
                r_a = rosters.get(team_a)
                r_b = rosters.get(team_b)
                s_res = simulate_fearless_series_maps(
                    team_a=team_a,
                    team_b=team_b,
                    roster_a=r_a,
                    roster_b=r_b,
                    best_of=best_of,
                    base_series_prob=p_match,
                    rng=rng,
                )
                winner = s_res.winner
                loser = s_res.loser
                sa = s_res.score_winner if winner == team_a else s_res.score_loser
                sb = s_res.score_loser if winner == team_a else s_res.score_winner
                sline = f"{s_res.score_winner}-{s_res.score_loser}"
            else:
                winner, loser, sa, sb, sline = _simulate_standard_series_maps(
                    team_a=team_a,
                    team_b=team_b,
                    best_of=best_of,
                    base_series_prob=p_match,
                    rng=rng,
                )
            return winner, loser, sa, sb, sline

        for _ in range(n_simulations):
            sim_rng = random.Random(master_rng.getrandbits(32))
            momentum_filter = (
                InPathTournamentBayesianFilter(
                    learning_rate=self.learning_rate,
                    adaptation_volatility=self.adaptation_volatility,
                    max_abs_logit_offset=self.max_abs_logit_offset,
                    map_margin_weight=self.map_margin_weight,
                    rng=sim_rng,
                )
                if bayesian
                else None
            )

            active_records = {t: (0, 0) for t in team_list}
            past_matchups: set[frozenset[str]] = set()

            # 1. Swiss Stage (Rounds 1–5)
            for round_num in range(1, 6):
                if round_num == 1 and initial_pairings is not None:
                    round_matches = [
                        SwissMatchup(team_a=p[0], team_b=p[1], pool="0-0", round_num=1, best_of=1)
                        for p in initial_pairings
                    ]
                elif riot_lottery:
                    round_matches = simulate_swiss_round_draws(
                        active_records=active_records,
                        past_matchups=past_matchups,
                        round_num=round_num,
                        team_regions=regions,
                        rng=sim_rng,
                    )
                else:
                    round_matches = _sample_fallback_swiss_draws(
                        active_records=active_records,
                        past_matchups=past_matchups,
                        round_num=round_num,
                        team_regions=regions,
                        rng=sim_rng,
                    )

                for m in round_matches:
                    p_eff = self.get_matchup_probability(
                        m.team_a,
                        m.team_b,
                        best_of=m.best_of,
                        base_probabilities=base_probabilities,
                        team_ratings=ratings,
                        glicko_states=glicko_states,
                        shrinkage_lambda=s_lambda,
                        temperature=temp,
                    )
                    p_match = (
                        momentum_filter.get_adjusted_probability(p_eff, m.team_a, m.team_b)
                        if momentum_filter
                        else p_eff
                    )

                    winner, loser, sa, sb, sline = run_single_series(
                        m.team_a, m.team_b, m.best_of, p_match, sim_rng
                    )

                    if momentum_filter:
                        momentum_filter.update_after_series(
                            m.team_a, m.team_b, winner, sa, sb, pre_match_prob_a=p_match
                        )

                    if m.best_of == 3:
                        bo3_counts[sline] += 1

                    w_rec = active_records[winner]
                    l_rec = active_records[loser]
                    active_records[winner] = (w_rec[0] + 1, w_rec[1])
                    active_records[loser] = (l_rec[0], l_rec[1] + 1)
                    past_matchups.add(frozenset({m.team_a, m.team_b}))

            qualifiers = [t for t, rec in active_records.items() if rec[0] == 3]
            for t in qualifiers:
                qual_acc[t] += 1.0

            distinct_reg = len({regions.get(t, "Unknown") for t in qualifiers})
            region_acc[distinct_reg] += 1

            # 2. Quarterfinal Draw
            unbeaten = [t for t in qualifiers if active_records[t] == (3, 0)]
            three_two = [t for t in qualifiers if active_records[t] == (3, 2)]

            sim_rng.shuffle(unbeaten)
            sim_rng.shuffle(three_two)
            opponents = three_two[:2]
            remaining = [t for t in qualifiers if t not in unbeaten + opponents]
            sim_rng.shuffle(remaining)

            pairs = [
                (unbeaten[0], opponents[0]),
                (remaining[0], remaining[1]),
                (unbeaten[1], opponents[1]),
                (remaining[2], remaining[3]),
            ]
            if sim_rng.randrange(2):
                pairs[0], pairs[1] = pairs[1], pairs[0]
            if sim_rng.randrange(2):
                pairs[2], pairs[3] = pairs[3], pairs[2]

            # 3. Knockout Stage Resolution
            if hybrid:
                stage = {"id": "knockout", "best_of": 5, "entrants": qualifiers}
                spec = {
                    "teams": qualifiers,
                    "champion": {"stage": "knockout", "group": "all", "rank": 1},
                    "stages": [stage],
                }
                n_q = len(qualifiers)
                prob_mat = np.full((n_q, n_q), 0.5, dtype=float)
                for i, ti in enumerate(qualifiers):
                    for j, tj in enumerate(qualifiers):
                        if i != j:
                            p_base_bo5 = base_bo5_matrix[team_idx[ti], team_idx[tj]]
                            prob_mat[i, j] = (
                                momentum_filter.get_adjusted_probability(p_base_bo5, ti, tj)
                                if momentum_filter
                                else p_base_bo5
                            )

                dp_res = _knockout_dp(spec, stage, qualifiers, pairs, {5: prob_mat})

                for t, prob in dp_res["champion_prob"].items():
                    champ_acc[t] += prob
                for t, prob in dp_res["final_prob"].items():
                    top2_acc[t] += prob
                for t in qualifiers:
                    ranks = dp_res["stage_rank_prob"]["knockout"]["all"][t]
                    top4_acc[t] += sum(ranks[:4])

                for pair_json, prob in dp_res["joint_final_prob"].items():
                    pair_tuple = tuple(sorted(json.loads(pair_json)))
                    joint_final_acc[pair_tuple] += prob

                # Sample Bo5 scorelines for series length distribution
                qf_w = []
                for p in pairs:
                    p_eff = prob_mat[qualifiers.index(p[0]), qualifiers.index(p[1])]
                    w, _, _, _, sline = run_single_series(p[0], p[1], 5, p_eff, sim_rng)
                    qf_w.append(w)
                    bo5_counts[sline] += 1

                sf_w = []
                for i in (0, 2):
                    t1, t2 = qf_w[i], qf_w[i + 1]
                    p_eff = prob_mat[qualifiers.index(t1), qualifiers.index(t2)]
                    w, _, _, _, sline = run_single_series(t1, t2, 5, p_eff, sim_rng)
                    sf_w.append(w)
                    bo5_counts[sline] += 1

                t1, t2 = sf_w[0], sf_w[1]
                p_eff = prob_mat[qualifiers.index(t1), qualifiers.index(t2)]
                _, _, _, _, sline = run_single_series(t1, t2, 5, p_eff, sim_rng)
                bo5_counts[sline] += 1

            else:
                # Full Monte Carlo Knockout Bracket
                qf_w = []
                for p in pairs:
                    p_eff = base_bo5_matrix[team_idx[p[0]], team_idx[p[1]]]
                    p_m = (
                        momentum_filter.get_adjusted_probability(p_eff, p[0], p[1])
                        if momentum_filter
                        else p_eff
                    )
                    w, l, sa, sb, sline = run_single_series(p[0], p[1], 5, p_m, sim_rng)
                    if momentum_filter:
                        momentum_filter.update_after_series(p[0], p[1], w, sa, sb, pre_match_prob_a=p_m)
                    qf_w.append(w)
                    bo5_counts[sline] += 1

                sf_w = []
                for i in (0, 2):
                    t1, t2 = qf_w[i], qf_w[i + 1]
                    p_eff = base_bo5_matrix[team_idx[t1], team_idx[t2]]
                    p_m = (
                        momentum_filter.get_adjusted_probability(p_eff, t1, t2)
                        if momentum_filter
                        else p_eff
                    )
                    w, l, sa, sb, sline = run_single_series(t1, t2, 5, p_m, sim_rng)
                    if momentum_filter:
                        momentum_filter.update_after_series(t1, t2, w, sa, sb, pre_match_prob_a=p_m)
                    sf_w.append(w)
                    bo5_counts[sline] += 1

                t1, t2 = sf_w[0], sf_w[1]
                p_eff = base_bo5_matrix[team_idx[t1], team_idx[t2]]
                p_m = (
                    momentum_filter.get_adjusted_probability(p_eff, t1, t2)
                    if momentum_filter
                    else p_eff
                )
                champ, runner_up, sa, sb, sline = run_single_series(t1, t2, 5, p_m, sim_rng)
                if momentum_filter:
                    momentum_filter.update_after_series(t1, t2, champ, sa, sb, pre_match_prob_a=p_m)
                bo5_counts[sline] += 1

                champ_acc[champ] += 1.0
                top2_acc[champ] += 1.0
                top2_acc[runner_up] += 1.0
                for t in qf_w:
                    top4_acc[t] += 1.0

                final_pair = tuple(sorted([t1, t2]))
                joint_final_acc[final_pair] += 1.0

        # Post-process distributions
        inv_n = 1.0 / n_simulations
        tot_c = math.fsum(champ_acc[t] * inv_n for t in team_list)
        c_scale = 1.0 / tot_c if tot_c > 0 else 1.0
        tot_t2 = math.fsum(top2_acc[t] * inv_n for t in team_list)
        t2_scale = 2.0 / tot_t2 if tot_t2 > 0 else 1.0

        standings_rows = []
        for t in team_list:
            standings_rows.append({
                "team": t,
                "champion_prob": (champ_acc[t] * inv_n) * c_scale,
                "top2_prob": (top2_acc[t] * inv_n) * t2_scale,
                "top3_prob": None,  # Single-elimination does not contest 3rd place
                "top4_prob": top4_acc[t] * inv_n,
            })
        standings_rows.sort(
            key=lambda x: (-x["champion_prob"], -x["top2_prob"], -x["top4_prob"])
        )
        standings = StandingsList(standings_rows)

        joint_finalists_rows = []
        for pair, cnt in sorted(joint_final_acc.items(), key=lambda x: -x[1])[:10]:
            joint_finalists_rows.append({
                "pair": f"{pair[0]} vs {pair[1]}",
                "teams": pair,
                "prob": round(cnt * inv_n, 6),
            })
        joint_finalists = JointFinalistsList(joint_finalists_rows)

        swiss_qualifiers_prob = {
            t: round(qual_acc[t] * inv_n, 6)
            for t in sorted(team_list, key=lambda t: -qual_acc[t])
        }

        region_dict: dict[int, float] = {
            k: round(region_acc.get(k, 0) * inv_n, 6) for k in (2, 3, 4, 5)
        }
        for k in sorted(region_acc):
            if k not in region_dict:
                region_dict[k] = round(region_acc[k] * inv_n, 6)
        region_distribution = RegionDistribution(region_dict)

        total_bo3 = sum(bo3_counts.values()) or 1
        total_bo5 = sum(bo5_counts.values()) or 1

        bo3_dist = {
            "2-0": round(bo3_counts.get("2-0", 0) / total_bo3, 6),
            "2-1": round(bo3_counts.get("2-1", 0) / total_bo3, 6),
        }
        bo5_dist = {
            "3-0": round(bo5_counts.get("3-0", 0) / total_bo5, 6),
            "3-1": round(bo5_counts.get("3-1", 0) / total_bo5, 6),
            "3-2": round(bo5_counts.get("3-2", 0) / total_bo5, 6),
        }

        series_length_distribution = SeriesLengthDistribution({
            "Bo3": bo3_dist,
            "Bo5": bo5_dist,
            "2-0": bo3_dist["2-0"],
            "2-1": bo3_dist["2-1"],
            "3-0": bo5_dist["3-0"],
            "3-1": bo5_dist["3-1"],
            "3-2": bo5_dist["3-2"],
        })

        return TournamentSimulationResult(
            standings=standings,
            joint_finalists=joint_finalists,
            swiss_qualifiers_prob=swiss_qualifiers_prob,
            region_distribution=region_distribution,
            series_length_distribution=series_length_distribution,
            simulations=n_simulations,
            seed=sim_seed,
            tournament="worlds",
            parameters={
                "use_fearless": fearless,
                "use_bayesian_momentum": bayesian,
                "use_hybrid_knockout": hybrid,
                "use_riot_lottery": riot_lottery,
                "shrinkage_lambda": s_lambda,
                "temperature": temp,
            },
        )

    def simulate_bracket(
        self,
        bracket_or_region: str | TournamentBracket | dict[str, Any] = "LEC",
        n_simulations: int = 1000,
        seed: int | None = None,
        teams: Sequence[str] | None = None,
        team_ratings: Mapping[str, float] | None = None,
        team_rosters: Mapping[str, list[str]] | None = None,
        glicko_states: Mapping[str, Glicko2State] | Mapping[str, tuple[float, float]] | None = None,
        base_probabilities: Mapping[tuple[str, str, int], float]
        | Mapping[tuple[str, str], float]
        | None = None,
        manual_overrides: Mapping[str, str] | None = None,
        reset_played: bool = False,
        *,
        use_fearless: bool | None = None,
        use_bayesian_momentum: bool | None = None,
        shrinkage_lambda: float | None = None,
        temperature: float | None = None,
    ) -> TournamentSimulationResult:
        """Simulate a regional playoff bracket (LEC, LCS, LPL, LCK, or custom TournamentBracket).

        Supports double-elimination, stepladder, and single-elimination graphs.
        Respects confirmed historical results unless reset_played=True.

        Returns
        -------
        TournamentSimulationResult
            Contains standings with top1 (champion), top2, top3, top4 probabilities,
            joint finalists, and Bo3/Bo5 scoreline distributions.
        """
        fearless = self.use_fearless if use_fearless is None else bool(use_fearless)
        bayesian = self.use_bayesian_momentum if use_bayesian_momentum is None else bool(use_bayesian_momentum)
        s_lambda = self.shrinkage_lambda if shrinkage_lambda is None else float(shrinkage_lambda)
        temp = self.temperature if temperature is None else float(temperature)

        sim_seed = self.seed if seed is None else int(seed)
        if sim_seed is None:
            sim_seed = random.randint(0, 2**31 - 1)

        # 1. Resolve TournamentBracket
        bracket: TournamentBracket
        bracket_name = "custom"
        if isinstance(bracket_or_region, str):
            key = bracket_or_region.strip().upper()
            if "LEC" in key:
                if get_lec_2026_summer_playoffs_bracket is None:
                    raise RuntimeError("betting_app.services.tournament_service is required for LEC bracket")
                bracket = get_lec_2026_summer_playoffs_bracket()
                bracket_name = "LEC"
            elif "LCK" in key:
                if get_lck_2026_playoffs_bracket is None:
                    raise RuntimeError("betting_app.services.tournament_service is required for LCK bracket")
                bracket = get_lck_2026_playoffs_bracket()
                bracket_name = "LCK"
            elif "LPL" in key:
                if get_lpl_2026_split3_playoffs_bracket is None:
                    raise RuntimeError("betting_app.services.tournament_service is required for LPL bracket")
                bracket = get_lpl_2026_split3_playoffs_bracket()
                bracket_name = "LPL"
            elif "LCS" in key:
                if get_lcs_2026_championship_bracket is None:
                    raise RuntimeError("betting_app.services.tournament_service is required for LCS bracket")
                bracket = get_lcs_2026_championship_bracket()
                bracket_name = "LCS"
            else:
                raise ValueError(
                    f"Unsupported bracket region: '{bracket_or_region}'. Supported: LEC, LCK, LPL, LCS"
                )
        elif hasattr(bracket_or_region, "matches") and hasattr(bracket_or_region, "teams"):
            bracket = bracket_or_region
            bracket_name = getattr(bracket, "id", "custom")
        else:
            raise TypeError(
                f"bracket_or_region must be a string or TournamentBracket, got {type(bracket_or_region)}"
            )

        team_list = list(bracket.teams) if teams is None else list(teams)
        ratings = dict(self.default_ratings)
        if team_ratings:
            ratings.update(team_ratings)

        rosters = dict(self.default_rosters)
        if team_rosters:
            rosters.update(team_rosters)

        manual = dict(manual_overrides) if manual_overrides else {}

        # 2. Extract topological order, final match, and placement bands
        matches = bracket.matches
        dependencies: dict[str, set[str]] = {m_id: set() for m_id in matches}
        incoming: dict[tuple[str, int], str] = {}
        for m_id, node in matches.items():
            if node.next_match_winner_id:
                incoming[(node.next_match_winner_id, node.next_match_winner_slot)] = m_id
                dependencies[node.next_match_winner_id].add(m_id)
            if node.next_match_loser_id:
                incoming[(node.next_match_loser_id, node.next_match_loser_slot)] = m_id
                dependencies[node.next_match_loser_id].add(m_id)

        execution_order: list[str] = []
        visited: set[str] = set()
        while len(execution_order) < len(matches):
            ready = [
                m_id
                for m_id, deps in dependencies.items()
                if m_id not in visited and deps <= visited
            ]
            if not ready:
                raise ValueError("Bracket contains an advancement cycle.")
            execution_order.extend(ready)
            visited.update(ready)

        finals = [m_id for m_id, node in matches.items() if node.next_match_winner_id is None]
        if not finals:
            raise ValueError("Bracket requires a terminal championship match")
        final_id = finals[0]

        distances: dict[str, int] = {}
        elimination_groups: dict[int, list[str]] = {}
        for m_id in reversed(execution_order):
            node = matches[m_id]
            d = 0 if m_id == final_id else 1 + distances[node.next_match_winner_id]
            distances[m_id] = d
            if node.next_match_loser_id is None:
                elimination_groups.setdefault(d, []).append(m_id)

        placement_bands: dict[str, tuple[int, int]] = {}
        current_rank = 2
        for dist in sorted(elimination_groups):
            grp = elimination_groups[dist]
            for m_id in grp:
                placement_bands[m_id] = (current_rank, current_rank + len(grp) - 1)
            current_rank += len(grp)

        counts = {t: {cutoff: 0 for cutoff in (1, 2, 3, 4)} for t in team_list}
        ambiguous = {t: set() for t in team_list}
        joint_final_counts: Counter[tuple[str, str]] = Counter()
        bo3_counts: Counter[str] = Counter()
        bo5_counts: Counter[str] = Counter()

        master_rng = random.Random(sim_seed)

        for _ in range(n_simulations):
            sim_rng = random.Random(master_rng.getrandbits(32))
            momentum_filter = (
                InPathTournamentBayesianFilter(
                    learning_rate=self.learning_rate,
                    adaptation_volatility=self.adaptation_volatility,
                    max_abs_logit_offset=self.max_abs_logit_offset,
                    map_margin_weight=self.map_margin_weight,
                    rng=sim_rng,
                )
                if bayesian
                else None
            )

            state: dict[str, list[str | None]] = {
                m_id: [node.team1, node.team2] for m_id, node in matches.items()
            }

            if reset_played:
                # Reset all non-seed slots
                for m_id in matches:
                    for slot in (1, 2):
                        if (m_id, slot) in incoming:
                            state[m_id][slot - 1] = None

            eliminated: set[str] = set()

            for m_id in execution_order:
                node = matches[m_id]
                t1, t2 = state[m_id]
                if t1 is None or t2 is None or t1 == t2:
                    raise RuntimeError(f"Unresolved or colliding participants in {m_id}: ({t1}, {t2})")

                winner = None if reset_played else (node.winner or manual.get(m_id))
                loser: str | None = None
                sa, sb = 0, 0
                sline = "3-0"

                if winner is not None:
                    loser = t2 if winner == t1 else t1
                    needed = (node.best_of // 2) + 1
                    sa = needed if winner == t1 else 0
                    sb = 0 if winner == t1 else needed
                    sline = f"{needed}-0"
                else:
                    p_eff = self.get_matchup_probability(
                        t1,
                        t2,
                        best_of=node.best_of,
                        base_probabilities=base_probabilities,
                        team_ratings=ratings,
                        glicko_states=glicko_states,
                        shrinkage_lambda=s_lambda,
                        temperature=temp,
                    )
                    p_match = (
                        momentum_filter.get_adjusted_probability(p_eff, t1, t2)
                        if momentum_filter
                        else p_eff
                    )

                    if fearless:
                        r_a = rosters.get(t1)
                        r_b = rosters.get(t2)
                        s_res = simulate_fearless_series_maps(
                            team_a=t1,
                            team_b=t2,
                            roster_a=r_a,
                            roster_b=r_b,
                            best_of=node.best_of,
                            base_series_prob=p_match,
                            rng=sim_rng,
                        )
                        winner = s_res.winner
                        loser = s_res.loser
                        sa = s_res.score_winner if winner == t1 else s_res.score_loser
                        sb = s_res.score_loser if winner == t1 else s_res.score_winner
                        sline = f"{s_res.score_winner}-{s_res.score_loser}"
                    else:
                        winner, loser, sa, sb, sline = _simulate_standard_series_maps(
                            team_a=t1,
                            team_b=t2,
                            best_of=node.best_of,
                            base_series_prob=p_match,
                            rng=sim_rng,
                        )

                    if momentum_filter:
                        momentum_filter.update_after_series(t1, t2, winner, sa, sb, pre_match_prob_a=p_match)

                if node.best_of == 3:
                    bo3_counts[sline] += 1
                elif node.best_of == 5:
                    bo5_counts[sline] += 1

                if node.next_match_winner_id:
                    state[node.next_match_winner_id][node.next_match_winner_slot - 1] = winner
                if node.next_match_loser_id:
                    state[node.next_match_loser_id][node.next_match_loser_slot - 1] = loser

                if node.next_match_loser_id is None and loser is not None:
                    eliminated.add(loser)
                    lower, upper = placement_bands[m_id]
                    for cutoff in (1, 2, 3, 4):
                        if upper <= cutoff:
                            counts[loser][cutoff] += 1
                        elif lower <= cutoff:
                            ambiguous[loser].add(cutoff)

                if m_id == final_id:
                    for cutoff in (1, 2, 3, 4):
                        counts[winner][cutoff] += 1
                    joint_pair = tuple(sorted([t1, t2]))
                    joint_final_counts[joint_pair] += 1

        inv_n = 1.0 / n_simulations
        tot_c = math.fsum(counts[t][1] * inv_n for t in team_list)
        c_scale = 1.0 / tot_c if tot_c > 0 else 1.0
        tot_t2 = math.fsum(counts[t][2] * inv_n for t in team_list)
        t2_scale = 2.0 / tot_t2 if tot_t2 > 0 else 1.0
        standings_rows = []
        for t in team_list:
            top3 = counts[t][3] * inv_n if 3 not in ambiguous[t] else None
            standings_rows.append({
                "team": t,
                "champion_prob": (counts[t][1] * inv_n) * c_scale,
                "top2_prob": (counts[t][2] * inv_n) * t2_scale,
                "top3_prob": top3,
                "top4_prob": counts[t][4] * inv_n,
            })
        standings_rows.sort(
            key=lambda x: (-x["champion_prob"], -x["top2_prob"], -x["top4_prob"])
        )
        standings = StandingsList(standings_rows)

        joint_finalists_rows = []
        for pair, cnt in sorted(joint_final_counts.items(), key=lambda x: -x[1])[:10]:
            joint_finalists_rows.append({
                "pair": f"{pair[0]} vs {pair[1]}",
                "teams": pair,
                "prob": round(cnt * inv_n, 6),
            })
        joint_finalists = JointFinalistsList(joint_finalists_rows)

        total_bo3 = sum(bo3_counts.values()) or 1
        total_bo5 = sum(bo5_counts.values()) or 1

        bo3_dist = {
            "2-0": round(bo3_counts.get("2-0", 0) / total_bo3, 6),
            "2-1": round(bo3_counts.get("2-1", 0) / total_bo3, 6),
        }
        bo5_dist = {
            "3-0": round(bo5_counts.get("3-0", 0) / total_bo5, 6),
            "3-1": round(bo5_counts.get("3-1", 0) / total_bo5, 6),
            "3-2": round(bo5_counts.get("3-2", 0) / total_bo5, 6),
        }

        series_length_distribution = SeriesLengthDistribution({
            "Bo3": bo3_dist,
            "Bo5": bo5_dist,
            "2-0": bo3_dist["2-0"],
            "2-1": bo3_dist["2-1"],
            "3-0": bo5_dist["3-0"],
            "3-1": bo5_dist["3-1"],
            "3-2": bo5_dist["3-2"],
        })

        return TournamentSimulationResult(
            standings=standings,
            joint_finalists=joint_finalists,
            swiss_qualifiers_prob=None,
            region_distribution=None,
            series_length_distribution=series_length_distribution,
            simulations=n_simulations,
            seed=sim_seed,
            tournament=bracket_name,
            parameters={
                "use_fearless": fearless,
                "use_bayesian_momentum": bayesian,
                "shrinkage_lambda": s_lambda,
                "temperature": temp,
            },
        )
