"""Probabilistic Game Duration Model using Gamma GLM for League of Legends (IDEA-018)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any
import numpy as np
from scipy import stats

from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker

STANDARD_DURATION_LINES: tuple[float, ...] = (28.5, 30.5, 32.5, 34.5)


@dataclass(frozen=True)
class DurationDistributionParams:
    """Continuous distribution parameters for match duration in minutes."""

    mean: float
    std: float
    shape_k: float
    scale_theta: float
    p10: float
    p25: float
    median: float
    p75: float
    p90: float


@dataclass(frozen=True)
class DurationLinePrediction:
    """Valuation of an Over/Under game duration line."""

    line: float
    prob_over: float
    prob_under: float
    fair_odds_over: float
    fair_odds_under: float
    market_odds_over: float | None = None
    market_odds_under: float | None = None
    ev_over_gross: float | None = None
    ev_under_gross: float | None = None
    ev_over_net_tax12: float | None = None
    ev_under_net_tax12: float | None = None


@dataclass(frozen=True)
class DurationPrediction:
    """Complete duration prediction output for a match."""

    team_a: str
    team_b: str
    expected_duration_minutes: float
    params: DurationDistributionParams
    lines: dict[float, DurationLinePrediction] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_line(self, line: float) -> DurationLinePrediction | None:
        return self.lines.get(float(line))


class GameDurationModel:
    """Predicts continuous duration distribution and Over/Under lines via Gamma distribution."""

    def __init__(
        self,
        default_std: float = 4.8,
        spread_coef: float = 0.0075,
        min_duration: float = 20.0,
        max_duration: float = 55.0,
    ):
        self.default_std = default_std
        self.spread_coef = spread_coef
        self.min_duration = min_duration
        self.max_duration = max_duration

    def estimate_duration_mean(
        self,
        expected_pace_duration: float,
        elo_diff: float,
    ) -> float:
        """Estimate expected duration. Elo disparity accelerates match completion."""
        abs_diff = abs(float(elo_diff))
        # Larger Elo gap -> faster victory for the favorite
        adjusted_mean = expected_pace_duration - (self.spread_coef * abs_diff)
        return float(np.clip(adjusted_mean, self.min_duration, self.max_duration))

    def estimate_params(
        self,
        expected_pace_duration: float,
        elo_diff: float,
        std: float | None = None,
    ) -> DurationDistributionParams:
        """Fit Gamma distribution parameters (k, theta) from target mean and std."""
        mu = self.estimate_duration_mean(expected_pace_duration, elo_diff)
        sigma = std if std is not None else self.default_std

        # In Gamma distribution: mean = k * theta, var = k * theta^2 = sigma^2
        # theta = sigma^2 / mu, k = mu / theta = mu^2 / sigma^2
        scale_theta = (sigma**2) / mu
        shape_k = (mu**2) / (sigma**2)

        # Compute quantiles
        p10 = float(stats.gamma.ppf(0.10, a=shape_k, scale=scale_theta))
        p25 = float(stats.gamma.ppf(0.25, a=shape_k, scale=scale_theta))
        median = float(stats.gamma.ppf(0.50, a=shape_k, scale=scale_theta))
        p75 = float(stats.gamma.ppf(0.75, a=shape_k, scale=scale_theta))
        p90 = float(stats.gamma.ppf(0.90, a=shape_k, scale=scale_theta))

        return DurationDistributionParams(
            mean=round(mu, 2),
            std=round(sigma, 2),
            shape_k=round(shape_k, 4),
            scale_theta=round(scale_theta, 4),
            p10=round(p10, 2),
            p25=round(p25, 2),
            median=round(median, 2),
            p75=round(p75, 2),
            p90=round(p90, 2),
        )

    def compute_duration_line(
        self,
        params: DurationDistributionParams,
        line: float,
        market_odds_over: float | None = None,
        market_odds_under: float | None = None,
        tax_rate: float = 0.12,
    ) -> DurationLinePrediction:
        """Compute exact probabilities and EV for a single duration line."""
        prob_under = float(stats.gamma.cdf(line, a=params.shape_k, scale=params.scale_theta))
        prob_under = min(max(prob_under, 1e-6), 1.0 - 1e-6)
        prob_over = 1.0 - prob_under

        fair_odds_over = round(1.0 / prob_over, 4)
        fair_odds_under = round(1.0 / prob_under, 4)

        ev_over_gross, ev_over_net = None, None
        if market_odds_over and market_odds_over > 1.0:
            ev_over_gross = round((prob_over * market_odds_over) - 1.0, 4)
            ev_over_net = round((prob_over * market_odds_over * (1.0 - tax_rate)) - 1.0, 4)

        ev_under_gross, ev_under_net = None, None
        if market_odds_under and market_odds_under > 1.0:
            ev_under_gross = round((prob_under * market_odds_under) - 1.0, 4)
            ev_under_net = round((prob_under * market_odds_under * (1.0 - tax_rate)) - 1.0, 4)

        return DurationLinePrediction(
            line=float(line),
            prob_over=round(prob_over, 4),
            prob_under=round(prob_under, 4),
            fair_odds_over=fair_odds_over,
            fair_odds_under=fair_odds_under,
            market_odds_over=market_odds_over,
            market_odds_under=market_odds_under,
            ev_over_gross=ev_over_gross,
            ev_under_gross=ev_under_gross,
            ev_over_net_tax12=ev_over_net,
            ev_under_net_tax12=ev_under_net,
        )

    def predict_duration(
        self,
        team_a: str,
        team_b: str,
        expected_pace_duration: float,
        elo_diff: float,
        target_lines: tuple[float, ...] = STANDARD_DURATION_LINES,
        market_lines: dict[float, tuple[float, float]] | None = None,
        tax_rate: float = 0.12,
    ) -> DurationPrediction:
        """Generate complete duration forecast and priced lines."""
        params = self.estimate_params(expected_pace_duration, elo_diff)
        lines_dict: dict[float, DurationLinePrediction] = {}

        for line in target_lines:
            m_over, m_under = None, None
            if market_lines and line in market_lines:
                m_over, m_under = market_lines[line]
            lines_dict[line] = self.compute_duration_line(
                params, line, market_odds_over=m_over, market_odds_under=m_under, tax_rate=tax_rate
            )

        return DurationPrediction(
            team_a=team_a,
            team_b=team_b,
            expected_duration_minutes=params.mean,
            params=params,
            lines=lines_dict,
            metadata={
                "elo_diff": elo_diff,
                "expected_pace_duration": expected_pace_duration,
                "model": "Gamma-Duration-Pace-Spread",
            },
        )

    def predict_from_tracker(
        self,
        tracker: ChronologicalPaceTracker,
        team_a: str,
        team_b: str,
        league_name: str | None = None,
        market_lines: dict[float, tuple[float, float]] | None = None,
        tax_rate: float = 0.12,
    ) -> DurationPrediction:
        """Convenience method using tracker matchup features."""
        features = tracker.get_matchup_spread_features(team_a, team_b, league_name=league_name)
        return self.predict_duration(
            team_a=team_a,
            team_b=team_b,
            expected_pace_duration=features.expected_duration_minutes,
            elo_diff=features.elo_diff,
            market_lines=market_lines,
            tax_rate=tax_rate,
        )
