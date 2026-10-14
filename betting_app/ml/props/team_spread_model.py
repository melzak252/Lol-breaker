"""Team kills and handicap spread prediction model using rating delta and discrete convolution (IDEA-018)."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy import stats

from betting_app.ml.props.feature_extractor import ChronologicalPaceTracker
from betting_app.ml.props.schemas import (
    HandicapLinePrediction,
    KillDistributionParams,
    MatchupSpreadFeatures,
    OverUnderLinePrediction,
    TeamKillsSpreadPrediction,
)

# Calibrated coefficients from out-of-time modeling on 38,804 professional games:
# ln(mu_team) = INTERCEPT + PACE_COEF * exp_pace_kills + SPREAD_COEF * elo_diff
DEFAULT_TEAM_INTERCEPT = 1.6935
DEFAULT_TEAM_PACE_COEF = 0.0672
DEFAULT_TEAM_SPREAD_COEF = 0.00050
DEFAULT_TEAM_ALPHA = 0.10  # Individual team dispersion (Var = mu + alpha * mu^2)

# Standard handicap and individual team lines
STANDARD_HANDICAP_LINES = (-10.5, -8.5, -6.5, -5.5, -4.5, -2.5, 2.5, 4.5, 5.5, 6.5, 8.5, 10.5)
STANDARD_TEAM_LINES = (8.5, 10.5, 12.5, 14.5, 16.5, 18.5, 20.5)
STANDARD_TOTAL_LINES = (22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5)


class TeamKillsSpreadModel:
    """Joint statistical model for individual team kills and kill handicap spreads."""

    def __init__(
        self,
        intercept: float = DEFAULT_TEAM_INTERCEPT,
        pace_coef: float = DEFAULT_TEAM_PACE_COEF,
        spread_coef: float = DEFAULT_TEAM_SPREAD_COEF,
        alpha: float = DEFAULT_TEAM_ALPHA,
    ) -> None:
        self.intercept = float(intercept)
        self.pace_coef = float(pace_coef)
        self.spread_coef = float(spread_coef)
        self.alpha = max(float(alpha), 1e-5)

    def estimate_team_mus(
        self,
        exp_pace_a: float,
        exp_pace_b: float,
        elo_diff: float,
    ) -> tuple[float, float]:
        """Estimate mean expected kills for both teams with strict skew-symmetry."""
        # ln(mu_A) = intercept + pace_coef * exp_pace_a + spread_coef * (elo_a - elo_b)
        log_mu_a = self.intercept + self.pace_coef * float(exp_pace_a) + self.spread_coef * float(elo_diff)
        # ln(mu_B) = intercept + pace_coef * exp_pace_b + spread_coef * (elo_b - elo_a)
        log_mu_b = self.intercept + self.pace_coef * float(exp_pace_b) - self.spread_coef * float(elo_diff)

        mu_a = max(math.exp(log_mu_a), 1.0)
        mu_b = max(math.exp(log_mu_b), 1.0)
        return round(mu_a, 3), round(mu_b, 3)

    def compute_difference_pmf(
        self,
        mu_a: float,
        mu_b: float,
        max_kills: int = 60,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact discrete difference distribution D = Kills_A - Kills_B via convolution."""
        n_a, p_a = 1.0 / self.alpha, 1.0 / (1.0 + self.alpha * mu_a)
        n_b, p_b = 1.0 / self.alpha, 1.0 / (1.0 + self.alpha * mu_b)

        k_vals = np.arange(0, max_kills + 1)
        pmf_a = stats.nbinom.pmf(k_vals, n_a, p_a)
        pmf_b = stats.nbinom.pmf(k_vals, n_b, p_b)

        # Discrete convolution of Kills_A and -Kills_B
        diff_pmf = np.convolve(pmf_a, pmf_b[::-1])
        diff_values = np.arange(-max_kills, max_kills + 1)
        return diff_values, diff_pmf

    def compute_handicap_line(
        self,
        diff_values: np.ndarray,
        diff_pmf: np.ndarray,
        handicap_line: float,
        market_odds_a: float | None = None,
        market_odds_b: float | None = None,
        tax_rate: float = 0.12,
    ) -> HandicapLinePrediction:
        """Compute probability and EV of Team A covering a handicap line H."""
        # Team A covers handicap H if (Kills_A - Kills_B) > H
        mask_cover_a = diff_values > handicap_line
        prob_cover_a = float(diff_pmf[mask_cover_a].sum())
        prob_cover_a = min(max(prob_cover_a, 1e-6), 1.0 - 1e-6)
        prob_cover_b = 1.0 - prob_cover_a

        fair_odds_a = round(1.0 / prob_cover_a, 4)
        fair_odds_b = round(1.0 / prob_cover_b, 4)

        # Expected Value calculations
        ev_a_gross, ev_a_net = None, None
        if market_odds_a and market_odds_a > 1.0:
            ev_a_gross = round((prob_cover_a * market_odds_a) - 1.0, 4)
            ev_a_net = round((prob_cover_a * market_odds_a * (1.0 - tax_rate)) - 1.0, 4)

        ev_b_gross, ev_b_net = None, None
        if market_odds_b and market_odds_b > 1.0:
            ev_b_gross = round((prob_cover_b * market_odds_b) - 1.0, 4)
            ev_b_net = round((prob_cover_b * market_odds_b * (1.0 - tax_rate)) - 1.0, 4)

        return HandicapLinePrediction(
            handicap_line=float(handicap_line),
            prob_cover_a=round(prob_cover_a, 4),
            prob_cover_b=round(prob_cover_b, 4),
            fair_odds_a=fair_odds_a,
            fair_odds_b=fair_odds_b,
            market_odds_a=market_odds_a,
            market_odds_b=market_odds_b,
            ev_a_gross=ev_a_gross,
            ev_b_gross=ev_b_gross,
            ev_a_net_tax12=ev_a_net,
            ev_b_net_tax12=ev_b_net,
        )

    def compute_team_total_line(
        self,
        mu: float,
        line: float,
        market_odds_over: float | None = None,
        market_odds_under: float | None = None,
        tax_rate: float = 0.12,
    ) -> OverUnderLinePrediction:
        """Compute analytical Over/Under probabilities for a single team's total kills."""
        floor_line = int(math.floor(line))
        n_param = 1.0 / self.alpha
        p_param = 1.0 / (1.0 + self.alpha * mu)

        prob_under = float(stats.nbinom.cdf(floor_line, n_param, p_param))
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

    def predict_matchup(
        self,
        spread_features: MatchupSpreadFeatures,
        league: str | None = None,
        target_handicaps: tuple[float, ...] = STANDARD_HANDICAP_LINES,
        target_team_lines: tuple[float, ...] = STANDARD_TEAM_LINES,
        target_total_lines: tuple[float, ...] = STANDARD_TOTAL_LINES,
        tax_rate: float = 0.12,
    ) -> TeamKillsSpreadPrediction:
        """Generate comprehensive joint forecast for team totals, total kills, and spread."""
        mu_a, mu_b = self.estimate_team_mus(
            exp_pace_a=spread_features.exp_pace_kills_a,
            exp_pace_b=spread_features.exp_pace_kills_b,
            elo_diff=spread_features.elo_diff,
        )
        mu_total = round(mu_a + mu_b, 3)

        # Difference PMF
        diff_values, diff_pmf = self.compute_difference_pmf(mu_a, mu_b)

        # Handicap line predictions
        handicap_preds: dict[float, HandicapLinePrediction] = {}
        for h in target_handicaps:
            handicap_preds[h] = self.compute_handicap_line(diff_values, diff_pmf, h, tax_rate=tax_rate)

        # Single team totals
        team_a_lines: dict[float, OverUnderLinePrediction] = {}
        team_b_lines: dict[float, OverUnderLinePrediction] = {}
        for line in target_team_lines:
            team_a_lines[line] = self.compute_team_total_line(mu_a, line, tax_rate=tax_rate)
            team_b_lines[line] = self.compute_team_total_line(mu_b, line, tax_rate=tax_rate)

        # Match total lines (using combined alpha ~0.08)
        total_lines: dict[float, OverUnderLinePrediction] = {}
        total_alpha = 0.08
        n_tot = 1.0 / total_alpha
        p_tot = 1.0 / (1.0 + total_alpha * mu_total)
        for line in target_total_lines:
            floor_l = int(math.floor(line))
            p_under = float(stats.nbinom.cdf(floor_l, n_tot, p_tot))
            p_under = min(max(p_under, 1e-6), 1.0 - 1e-6)
            p_over = 1.0 - p_under
            total_lines[line] = OverUnderLinePrediction(
                line=float(line),
                prob_over=round(p_over, 4),
                prob_under=round(p_under, 4),
                fair_odds_over=round(1.0 / p_over, 4),
                fair_odds_under=round(1.0 / p_under, 4),
            )

        # PMF snippet for charting (-25 to +25)
        curve = []
        for d, p in zip(diff_values, diff_pmf):
            if -25 <= d <= 25:
                curve.append({"diff": int(d), "prob": round(float(p), 5)})

        return TeamKillsSpreadPrediction(
            team_a=spread_features.team_a,
            team_b=spread_features.team_b,
            league=league,
            spread_features=spread_features,
            mu_team_a=mu_a,
            mu_team_b=mu_b,
            mu_total=mu_total,
            alpha=self.alpha,
            team_a_lines=team_a_lines,
            team_b_lines=team_b_lines,
            handicap_lines=handicap_preds,
            total_lines=total_lines,
            difference_pmf=curve,
            metadata={
                "model_family": "TeamKillsSpreadModel (NegativeBinomial + Convolution)",
                "intercept": self.intercept,
                "pace_coef": self.pace_coef,
                "spread_coef": self.spread_coef,
                "alpha": self.alpha,
            },
        )

    def predict_from_tracker(
        self,
        tracker: ChronologicalPaceTracker,
        team_a: str,
        team_b: str,
        league_name: str | None = None,
        tax_rate: float = 0.12,
    ) -> TeamKillsSpreadPrediction:
        """Extract pre-match features from tracker and predict joint distribution."""
        spread_features = tracker.get_matchup_spread_features(team_a, team_b, league_name)
        return self.predict_matchup(spread_features, league=league_name, tax_rate=tax_rate)
