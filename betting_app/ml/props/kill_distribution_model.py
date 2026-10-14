"""Negative Binomial distribution model for predicting professional LoL map kills (IDEA-018)."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy import stats

from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.schemas import (
    KillDistributionParams,
    KillDistributionPrediction,
    MatchPropContext,
    OverUnderLinePrediction,
)

# Calibrated defaults from out-of-time training on 38,804 professional games
# ln(mu) = INTERCEPT + PACE_COEF * pace + SPREAD_COEF * spread + INTERACTION * (pace * spread)
DEFAULT_INTERCEPT = 2.4883
DEFAULT_PACE_COEF = 0.02994
DEFAULT_SPREAD_COEF = 0.00130
DEFAULT_INTERACTION_COEF = -0.0000458
DEFAULT_BASE_ALPHA = 0.0560  # Base dispersion for close matches
DEFAULT_SPREAD_ALPHA_COEF = 0.000080  # Fatter tails for stomps

# Common proposition lines offered by sportsbooks
STANDARD_PROP_LINES = (18.5, 20.5, 22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5, 36.5, 38.5, 40.5)


class KillDistributionModel:
    """Statistical model for predicting discrete kill probability distributions."""

    def __init__(
        self,
        intercept: float = DEFAULT_INTERCEPT,
        pace_coef: float = DEFAULT_PACE_COEF,
        spread_coef: float = DEFAULT_SPREAD_COEF,
        interaction_coef: float = DEFAULT_INTERACTION_COEF,
        base_alpha: float = DEFAULT_BASE_ALPHA,
        spread_alpha_coef: float = DEFAULT_SPREAD_ALPHA_COEF,
        alpha: float | None = None,
    ) -> None:
        self.intercept = float(intercept)
        self.pace_coef = float(pace_coef)
        self.spread_coef = float(spread_coef)
        self.interaction_coef = float(interaction_coef)
        chosen_alpha = alpha if alpha is not None else base_alpha
        self.base_alpha = max(float(chosen_alpha), 1e-4)
        self.spread_alpha_coef = float(spread_alpha_coef)
        self.alpha = self.base_alpha

    def estimate_params(
        self,
        expected_pace_kills: float,
        team_spread: float = 0.0,
    ) -> KillDistributionParams:
        """Estimate mean mu and dispersion parameters incorporating pace and team spread."""
        pace = float(expected_pace_kills)
        spread = float(team_spread)

        # Log-link GLM with pace, spread, and interaction:
        # ln(mu) = intercept + pace_coef * pace + spread_coef * spread + interaction * (pace * spread)
        log_mu = (
            self.intercept
            + self.pace_coef * pace
            + self.spread_coef * spread
            + self.interaction_coef * (pace * spread)
        )
        mu = max(math.exp(log_mu), 1.0)

        # Heteroscedastic dispersion: stomps have wider relative dispersion (fatter tails)
        alpha = max(self.base_alpha + self.spread_alpha_coef * spread, 0.01)

        # Variance for Negative Binomial: Var[Y] = mu + alpha * mu^2
        variance = mu + alpha * (mu**2)
        std = math.sqrt(variance)

        # Scipy nbinom parameterization:
        # n = 1 / alpha (number of failures)
        # p = 1 / (1 + alpha * mu) (success probability)
        # E[Y] = n * (1 - p) / p = mu
        # Var[Y] = n * (1 - p) / p^2 = mu + alpha * mu^2
        n_param = 1.0 / alpha
        p_param = 1.0 / (1.0 + alpha * mu)

        return KillDistributionParams(
            mu=round(mu, 3),
            alpha=round(alpha, 5),
            variance=round(variance, 3),
            std=round(std, 3),
            n_param=round(n_param, 5),
            p_param=round(p_param, 5),
        )

    def compute_quantiles(self, params: KillDistributionParams) -> dict[str, int]:
        """Compute key percentiles of the kill distribution (P10, P25, Median, P75, P90)."""
        n, p = params.n_param, params.p_param
        return {
            "p10": int(stats.nbinom.ppf(0.10, n, p)),
            "p25": int(stats.nbinom.ppf(0.25, n, p)),
            "p50": int(stats.nbinom.ppf(0.50, n, p)),
            "p75": int(stats.nbinom.ppf(0.75, n, p)),
            "p90": int(stats.nbinom.ppf(0.90, n, p)),
        }

    def compute_discrete_crps(
        self,
        params: KillDistributionParams,
        actual_kills: int | float,
        max_k: int = 80,
    ) -> float:
        """Compute Continuous Ranked Probability Score (CRPS) against an observed kill count."""
        k_vals = np.arange(0, max_k + 1)
        cdf = stats.nbinom.cdf(k_vals, params.n_param, params.p_param)
        heaviside = (k_vals >= actual_kills).astype(float)
        return float(np.sum((cdf - heaviside) ** 2))

    def compute_line_probabilities(
        self,
        params: KillDistributionParams,
        line: float,
        market_odds_over: float | None = None,
        market_odds_under: float | None = None,
        tax_rate: float = 0.12,
    ) -> OverUnderLinePrediction:
        """Compute analytical Over/Under probabilities and EV for a specific half-line."""
        floor_line = int(math.floor(line))

        prob_under = float(stats.nbinom.cdf(floor_line, params.n_param, params.p_param))
        prob_under = min(max(prob_under, 1e-6), 1.0 - 1e-6)
        prob_over = 1.0 - prob_under

        fair_odds_over = round(1.0 / prob_over, 4)
        fair_odds_under = round(1.0 / prob_under, 4)

        # EV calculation if bookmaker odds are supplied
        ev_over_gross = None
        ev_over_net = None
        if market_odds_over is not None and market_odds_over > 1.0:
            ev_over_gross = round((prob_over * market_odds_over) - 1.0, 4)
            eff_odds = market_odds_over * (1.0 - tax_rate)
            ev_over_net = round((prob_over * eff_odds) - 1.0, 4)

        ev_under_gross = None
        ev_under_net = None
        if market_odds_under is not None and market_odds_under > 1.0:
            ev_under_gross = round((prob_under * market_odds_under) - 1.0, 4)
            eff_odds = market_odds_under * (1.0 - tax_rate)
            ev_under_net = round((prob_under * eff_odds) - 1.0, 4)

        return OverUnderLinePrediction(
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

    def generate_density_curve(
        self,
        params: KillDistributionParams,
        min_kills: int = 5,
        max_kills: int = 65,
    ) -> list[dict[str, float]]:
        """Generate PMF probability points for charting and visualization."""
        curve = []
        for k in range(min_kills, max_kills + 1):
            p = float(stats.nbinom.pmf(k, params.n_param, params.p_param))
            curve.append({"kills": k, "prob": round(p, 5)})
        return curve

    def predict_distribution(
        self,
        team_a: str,
        team_b: str,
        league: str | None,
        expected_pace_kills: float,
        team_spread: float = 0.0,
        target_lines: tuple[float, ...] = STANDARD_PROP_LINES,
        market_lines: dict[float, tuple[float, float]] | None = None,
        tax_rate: float = 0.12,
        context: MatchPropContext | None = None,
    ) -> KillDistributionPrediction:
        """Generate full distribution forecast, quantiles, and line analysis for a matchup."""
        params = self.estimate_params(expected_pace_kills, team_spread=team_spread)
        quantiles = self.compute_quantiles(params)

        lines_pred: dict[float, OverUnderLinePrediction] = {}
        for line in target_lines:
            odds_over, odds_under = None, None
            if market_lines and line in market_lines:
                odds_over, odds_under = market_lines[line]

            pred = self.compute_line_probabilities(
                params=params,
                line=line,
                market_odds_over=odds_over,
                market_odds_under=odds_under,
                tax_rate=tax_rate,
            )
            lines_pred[line] = pred

        density_curve = self.generate_density_curve(params)

        return KillDistributionPrediction(
            team_a=team_a,
            team_b=team_b,
            league=league,
            params=params,
            lines=lines_pred,
            density_curve=density_curve,
            quantiles=quantiles,
            context=context,
            metadata={
                "model_family": "NegativeBinomialGLM (Pace + Spread + Dynamic Dispersion)",
                "intercept": self.intercept,
                "pace_coef": self.pace_coef,
                "spread_coef": self.spread_coef,
                "interaction_coef": self.interaction_coef,
                "base_alpha": self.base_alpha,
                "expected_pace_kills": expected_pace_kills,
                "team_spread": team_spread,
            },
        )

    def predict_from_tracker(
        self,
        tracker: ChronologicalPaceTracker,
        team_a: str,
        team_b: str,
        league: str | None = None,
        market_lines: dict[float, tuple[float, float]] | None = None,
    ) -> KillDistributionPrediction:
        """Convenience method: extract pace and spread from tracker and predict distribution."""
        spread_features = tracker.get_matchup_spread_features(team_a, team_b, league)
        prop_context = tracker.get_matchup_prop_context(team_a, team_b, league)

        prediction = self.predict_distribution(
            team_a=team_a,
            team_b=team_b,
            league=league,
            expected_pace_kills=spread_features.expected_total_kills,
            team_spread=spread_features.abs_elo_diff,
            market_lines=market_lines,
            context=prop_context,
        )
        prediction.metadata["team_a_sample"] = tracker.get_team_pace(team_a, league).sample_games
        prediction.metadata["team_b_sample"] = tracker.get_team_pace(team_b, league).sample_games
        prediction.metadata["elo_diff"] = spread_features.elo_diff
        prediction.metadata["prob_win_a"] = spread_features.prob_win_a
        return prediction
