"""Data contracts and schemas for in-game proposition prediction models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PaceFeatures:
    """Pre-match pace and tempo indicators for a team or matchup."""

    team_name: str
    sample_games: int
    avg_kills: float
    avg_deaths: float
    avg_duration_minutes: float
    ckpm: float  # Combined Kills Per Minute
    first_blood_rate: float | None = None
    first_tower_rate: float | None = None
    first_dragon_rate: float | None = None


@dataclass(frozen=True)
class MatchupSpreadFeatures:
    """Pre-match spread, rating, and pace indicators between two opponents."""

    team_a: str
    team_b: str
    rating_a: float
    rating_b: float
    elo_diff: float  # rating_a - rating_b
    abs_elo_diff: float  # abs(rating_a - rating_b)
    prob_win_a: float
    prob_win_b: float
    exp_pace_kills_a: float
    exp_pace_kills_b: float
    expected_total_kills: float
    expected_duration_minutes: float


@dataclass(frozen=True)
class HandicapLinePrediction:
    """Analytical forecast and market comparison for a kill handicap line (e.g. Team A -5.5)."""

    handicap_line: float  # Line for Team A (e.g. -5.5 or +5.5)
    prob_cover_a: float  # P(kills_a - kills_b > handicap_line)
    prob_cover_b: float  # P(kills_b - kills_a > -handicap_line)
    fair_odds_a: float
    fair_odds_b: float
    market_odds_a: float | None = None
    market_odds_b: float | None = None
    ev_a_gross: float | None = None
    ev_b_gross: float | None = None
    ev_a_net_tax12: float | None = None
    ev_b_net_tax12: float | None = None


@dataclass(frozen=True)
class TeamKillsSpreadPrediction:
    """Joint prediction for team totals, total kills, and kill handicap spread."""

    team_a: str
    team_b: str
    league: str | None
    spread_features: MatchupSpreadFeatures
    mu_team_a: float
    mu_team_b: float
    mu_total: float
    alpha: float
    team_a_lines: dict[float, OverUnderLinePrediction] = field(default_factory=dict)
    team_b_lines: dict[float, OverUnderLinePrediction] = field(default_factory=dict)
    handicap_lines: dict[float, HandicapLinePrediction] = field(default_factory=dict)
    total_lines: dict[float, OverUnderLinePrediction] = field(default_factory=dict)
    difference_pmf: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_handicap_line(self, line: float) -> HandicapLinePrediction | None:
        return self.handicap_lines.get(float(line))

@dataclass(frozen=True)
class KillDistributionParams:
    """Parameters defining a Negative Binomial distribution for map kills."""

    mu: float  # Mean expected kills E[Y]
    alpha: float  # Dispersion parameter where Var[Y] = mu + alpha * mu^2
    variance: float  # Var[Y]
    std: float  # sqrt(Var[Y])
    n_param: float  # r = 1 / alpha for scipy.stats.nbinom
    p_param: float  # p = 1 / (1 + alpha * mu) for scipy.stats.nbinom


@dataclass(frozen=True)
class OverUnderLinePrediction:
    """Predicted probabilities and fair prices for one Over/Under line."""

    line: float  # e.g. 26.5
    prob_over: float  # P(Kills > line)
    prob_under: float  # P(Kills < line)
    fair_odds_over: float  # 1 / prob_over
    fair_odds_under: float  # 1 / prob_under
    # Expected values if market odds are supplied
    market_odds_over: float | None = None
    market_odds_under: float | None = None
    ev_over_gross: float | None = None
    ev_under_gross: float | None = None
    ev_over_net_tax12: float | None = None
    ev_under_net_tax12: float | None = None

@dataclass(frozen=True)
class RecentGameSummary:
    """Summary of a team's single past match."""

    opponent: str
    kills_for: int
    kills_against: int
    total_kills: int
    duration_minutes: float
    won: bool
    date: str | None = None


@dataclass(frozen=True)
class TeamRecentForm:
    """Recent form and kill trend for a team over past matches."""

    team_name: str
    sample_size: int
    avg_kills: float
    avg_deaths: float
    avg_total_kills: float
    avg_duration_minutes: float
    min_kills: int = 0
    max_kills: int = 0
    std_kills: float = 0.0
    kill_brackets: dict[str, int] = field(default_factory=dict)  # e.g. {"<10": 0, "10-14": 1, "15-19": 3, "20-24": 1, "25+": 0}
    recent_games: list[RecentGameSummary] = field(default_factory=list)
@dataclass(frozen=True)
class LeaguePaceContext:
    """League benchmark pace metrics and historical kill distribution."""

    league_name: str | None
    league_key: str
    avg_kills: float
    avg_duration_minutes: float
    ckpm: float
    pace_category: str  # e.g. "kontrolowana (wolne makro)", "dynamiczna (wysokie tempo)"
    p25_kills: float
    median_kills: float
    p75_kills: float


@dataclass(frozen=True)
class MatchPropContext:
    """Pre-match historical and league context accompanying proposition predictions."""

    team_a_form: TeamRecentForm
    team_b_form: TeamRecentForm
    league_context: LeaguePaceContext
    expected_pace_kills: float
    delta_vs_league_avg: float
    summary_narrative: str


@dataclass
class KillDistributionPrediction:
    """Full probability distribution forecast for map total kills."""

    team_a: str
    team_b: str
    league: str | None
    params: KillDistributionParams
    lines: dict[float, OverUnderLinePrediction] = field(default_factory=dict)
    density_curve: list[dict[str, float]] = field(default_factory=list)  # [{'kills': 25, 'prob': 0.045}, ...]
    quantiles: dict[str, int] = field(default_factory=dict)  # {'p10': 18, 'p25': 23, 'p50': 29, 'p75': 35, 'p90': 42}
    context: MatchPropContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mu(self) -> float:
        return self.params.mu

    @property
    def std(self) -> float:
        return self.params.std

    def get_line(self, line: float) -> OverUnderLinePrediction | None:
        return self.lines.get(float(line))
