"""Canonical Unified Evaluation & Staking Engine for LoL Betting.

Single Source of Truth (SSOT) for:
1. Candidate bet qualification via `bet_qualification_service.is_bet_eligible`
2. Model-based EV, historical heuristic adjustment and realized returns (PLN and %)
3. Portfolio simulations: Flat staking, 1/4 Kelly, and fixed fraction
4. Multi-dimensional diagnostic slicing (EV bins, odds brackets, bookmakers)
5. Historical expected-versus-realized diagnostics (not forward calibration proof)

Guarantees:
- Callers with identical declared evidence and policies receive identical numbers.
- Applies the shared qualification gate before portfolio selection.
- Handles both production database records and historical research files (e.g. best_available_odds).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from betting_app.services.bet_qualification_service import (
    DEFAULT_BASE_MIN_EV,
    DEFAULT_COINFLIP_MAX_ODDS,
    DEFAULT_COINFLIP_MIN_ODDS,
    DEFAULT_HURDLE_ODDS_CUTOFF,
    DEFAULT_HURDLE_PENALTY_SCALE,
    DEFAULT_LONGSHOT_ODDS_THRESHOLD,
    DEFAULT_MAX_BO1_MARKET_GAP,
    DEFAULT_MAX_COINFLIP_MARKET_GAP,
    DEFAULT_MAX_EV_LONGSHOT,
    DEFAULT_MAX_EV_NET,
    DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
    DEFAULT_MAX_RATING_DISAGREEMENT,
    DEFAULT_MAX_TIER1_MARKET_GAP,
    is_bet_eligible,
    qualification_tier,
)


@dataclass(frozen=True)
class EvaluatedBet:
    """A candidate quote, including rejection and declared uncertainty evidence."""

    match_id: str | int
    side: str  # 'team_a' | 'team_b'
    bookmaker: str
    odds: float
    odds_close: float | None
    prob_model: float
    prob_conservative: float | None
    ev_net: float | None  # Expected net EV; absent when required evidence is missing
    ev_calibrated: float | None  # Historical heuristic adjustment, not certified EV
    prob_for_ev: float | None
    uncertainty_basis: str
    uncertainty_reason: str
    won: bool
    stake: float
    pnl: float  # Net profit/loss in PLN after tax
    clv_pct: float | None = None  # (odds / odds_close - 1) * 100
    clv_pp: float | None = None  # (1/close - 1/open) * 100
    competition_tier: str | None = None
    best_of: int | None = None
    quarantine: bool = False
    quarantine_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationBin:
    """Per-bet expected vs realized calibration metrics for a specific slice/bin."""

    label: str
    bets_count: int
    wins_count: int
    win_rate_pct: float
    avg_odds: float
    avg_stake: float
    total_staked: float
    total_pnl: float
    expected_ev_pct: float  # Mean model expected EV per bet (%)
    expected_profit_per_bet: float  # PLN expected per bet
    realized_ev_pct: float  # Mean realized net return per bet (%) = (total_pnl / total_staked) * 100
    realized_profit_per_bet: float  # PLN realized per bet = total_pnl / bets_count
    calibration_gap_pct: float  # realized_ev_pct - expected_ev_pct
    calibration_gap_pln: float  # realized_profit_per_bet - expected_profit_per_bet
    ev_realization_rate_pct: float  # (realized_ev_pct / expected_ev_pct) * 100
    median_clv_pct: float | None = None
    mean_clv_pp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationSummary:
    """Complete summary of a betting portfolio run."""

    strategy: str  # 'flat' | 'quarter_kelly' | 'fixed_fraction'
    initial_bankroll: float
    final_bankroll: float
    total_staked: float
    total_pnl: float
    roi_pct: float | None  # Undefined without turnover
    bets_count: int
    wins_count: int
    win_rate_pct: float | None
    avg_odds: float | None
    avg_stake: float | None
    expected_ev_pct: float | None  # Mean historically adjusted EV per bet (%)
    expected_profit_per_bet: float | None
    realized_ev_pct: float | None
    realized_profit_per_bet: float | None
    calibration_gap_pln: float | None
    max_drawdown_pct: float
    max_drawdown_pln: float
    median_clv_pct: float | None
    mean_clv_pp: float | None
    raw_ev_pct: float | None = None  # Mean raw uncalibrated EV per bet (%)
    result_status: str = "computed"
    bins: list[CalibrationBin] = field(default_factory=list)
    by_odds_bracket: list[CalibrationBin] = field(default_factory=list)
    by_bookmaker: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_p_low(
    prob: float,
    kappa: float = 0.75,
    sigma_z: float = 0.15,
) -> float:
    """Apply a logit-space research haircut, not a measured uncertainty bound.

    logit(p_low) = logit(p) - kappa * sigma_z
    """
    p_clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    logit_val = math.log(p_clipped / (1.0 - p_clipped))
    z_low = logit_val - (kappa * sigma_z)
    return float(1.0 / (1.0 + math.exp(-z_low)))


def compute_ensemble_p_low(
    prob: float, sigma_z: float, kappa: float = 0.75, mode: str = "logit"
) -> float:
    """Side-specific conservative probability from ensemble dispersion."""
    if not math.isfinite(prob) or not 0.0 < prob < 1.0:
        raise ValueError("prob must be finite and in (0, 1)")
    if not math.isfinite(sigma_z) or sigma_z < 0.0:
        raise ValueError("dispersion must be finite and nonnegative")
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("kappa must be finite and nonnegative")
    p = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    if mode == "logit":
        z = math.log(p / (1.0 - p)) - kappa * sigma_z
        result = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
    elif mode == "linear":
        result = p - kappa * sigma_z
    else:
        raise ValueError(f"unknown uncertainty mode: {mode}")
    return min(prob, max(1e-6 if prob >= 1e-6 else prob, result))


def compute_quantile_p_low(prob_mean: float, p_q10: float) -> float:
    """Conservative side probability; use 1 - q90 for the opposite side."""
    if not all(math.isfinite(v) and 0.0 < v < 1.0 for v in (prob_mean, p_q10)):
        raise ValueError("probabilities must be finite and in (0, 1)")
    return min(prob_mean, p_q10)


def compute_shrunk_p_low(
    prob_shrunk: float,
    prob_model: float,
    prob_market: float,
    odds: float,
    kappa: float = 0.75,
    sigma_base: float = 0.12,
    beta_disagreement: float = 0.25,
    odds_ref: float = 1.50,
) -> float:
    """Apply a historical hybrid research haircut, not a measured bound.

    The fixed heuristic depends on model-market logit disagreement and odds:
       sigma_z = (sigma_base + beta * |z_mod - z_mkt|) * sqrt(max(1.0, odds / odds_ref))
       logit(p_low) = logit(p_shrunk) - kappa * sigma_z
    """
    p_s = float(np.clip(prob_shrunk, 1e-6, 1.0 - 1e-6))
    p_mod = float(np.clip(prob_model, 1e-6, 1.0 - 1e-6))
    p_mkt = float(np.clip(prob_market, 1e-6, 1.0 - 1e-6))

    z_shrunk = math.log(p_s / (1.0 - p_s))
    z_mod = math.log(p_mod / (1.0 - p_mod))
    z_mkt = math.log(p_mkt / (1.0 - p_mkt))

    diff_z = abs(z_mod - z_mkt)
    odds_factor = math.sqrt(max(1.0, float(odds) / odds_ref))
    sigma_z = (sigma_base + beta_disagreement * diff_z) * odds_factor

    z_low = z_shrunk - (kappa * sigma_z)
    p_low = 1.0 / (1.0 + math.exp(-z_low))
    return float(min(p_s, p_low))

def calculate_quarter_kelly_stake(
    bankroll: float,
    prob_for_ev: float,
    odds: float,
    tax_rate: float = 0.12,
    cap_fraction: float = 0.025,
    fraction: float = 0.25,
    min_bankroll: float = 10.0,
) -> float:
    """Compute 1/4 Kelly stake respecting bankroll cap and tax."""
    if bankroll <= min_bankroll:
        return 0.0
    net_multiplier = odds * (1.0 - tax_rate)
    b_net = net_multiplier - 1.0
    if b_net <= 0.0 or prob_for_ev <= 0.0 or prob_for_ev >= 1.0:
        return 0.0
    q = 1.0 - prob_for_ev
    f_star = (prob_for_ev * b_net - q) / b_net
    if f_star <= 0.0:
        return 0.0
    stake_fraction = min(cap_fraction, fraction * f_star)
    return float(bankroll * stake_fraction)


class UnifiedBettingEngine:
    """Reproducible declared-evidence qualification and historical simulation."""

    def __init__(
        self,
        tax_rate: float = 0.12,
        min_ev_net: float = DEFAULT_BASE_MIN_EV,
        dynamic_hurdle: bool = True,
        hurdle_odds_cutoff: float = DEFAULT_HURDLE_ODDS_CUTOFF,
        hurdle_penalty_scale: float = DEFAULT_HURDLE_PENALTY_SCALE,
        max_ev_longshot: float | None = DEFAULT_MAX_EV_LONGSHOT,
        longshot_odds_threshold: float = DEFAULT_LONGSHOT_ODDS_THRESHOLD,
        max_ev_net: float | None = DEFAULT_MAX_EV_NET,
        max_negative_clv_drift: float | None = DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
        max_bo1_market_gap: float | None = DEFAULT_MAX_BO1_MARKET_GAP,
        bo1_max_odds: float | None = 3.00,
        max_tier1_market_gap: float | None = DEFAULT_MAX_TIER1_MARKET_GAP,
        max_coinflip_market_gap: float | None = DEFAULT_MAX_COINFLIP_MARKET_GAP,
        coinflip_min_odds: float = DEFAULT_COINFLIP_MIN_ODDS,
        coinflip_max_odds: float = DEFAULT_COINFLIP_MAX_ODDS,
        uncertainty_required: bool = True,
        kappa: float = 0.75,
        default_sigma_z: float = 0.15,
    ) -> None:
        self.tax_rate = tax_rate
        self.min_ev_net = min_ev_net
        self.dynamic_hurdle = dynamic_hurdle
        self.hurdle_odds_cutoff = hurdle_odds_cutoff
        self.hurdle_penalty_scale = hurdle_penalty_scale
        self.max_ev_longshot = max_ev_longshot
        self.longshot_odds_threshold = longshot_odds_threshold
        self.max_ev_net = max_ev_net
        self.max_negative_clv_drift = max_negative_clv_drift
        self.max_bo1_market_gap = max_bo1_market_gap
        self.bo1_max_odds = bo1_max_odds
        self.max_tier1_market_gap = max_tier1_market_gap
        self.max_coinflip_market_gap = max_coinflip_market_gap
        self.coinflip_min_odds = coinflip_min_odds
        self.coinflip_max_odds = coinflip_max_odds
        self.uncertainty_required = uncertainty_required
        self.kappa = kappa
        self.default_sigma_z = default_sigma_z

    def qualify_quote(
        self,
        match_id: str | int,
        side: str,
        odds: float,
        prob_model: float,
        won: bool,
        prob_conservative: float | None = None,
        prob_market_novig: float | None = None,
        prob_market_close_novig: float | None = None,
        odds_close: float | None = None,
        bookmaker: str = "unknown",
        league: str | None = None,
        date_str: str | None = None,
        best_of: int | None = None,
        rating_disagreement: float | None = None,
        flat_stake: float = 100.0,
        prob_standalone_model: float | None = None,
        uncertainty_basis: str | None = None,
        uncertainty_reason: str | None = None,
    ) -> tuple[bool, EvaluatedBet]:
        """Qualify a single candidate betting quote through the canonical gate."""
        if prob_conservative is None and uncertainty_basis == "assumed_research_haircut":
            if prob_standalone_model is not None and prob_market_novig is not None:
                prob_conservative = compute_shrunk_p_low(
                    prob_shrunk=prob_model,
                    prob_model=prob_standalone_model,
                    prob_market=prob_market_novig,
                    odds=odds,
                    kappa=self.kappa,
                )
            else:
                prob_conservative = compute_p_low(
                    prob_model, kappa=self.kappa, sigma_z=self.default_sigma_z
                )

        tier = qualification_tier(league, date_str) if league and date_str else None

        eligible, reason, diag = is_bet_eligible(
            prob_model=prob_model,
            prob_conservative=prob_conservative,
            odds=odds,
            prob_market_novig=prob_market_novig,
            tax_rate=self.tax_rate,
            min_ev_net=self.min_ev_net,
            dynamic_hurdle=self.dynamic_hurdle,
            hurdle_odds_cutoff=self.hurdle_odds_cutoff,
            hurdle_penalty_scale=self.hurdle_penalty_scale,
            rating_disagreement=rating_disagreement,
            competition_tier=tier,
            max_tier1_market_gap=self.max_tier1_market_gap,
            max_ev_net=self.max_ev_net,
            best_of=best_of,
            max_ev_longshot=self.max_ev_longshot,
            longshot_odds_threshold=self.longshot_odds_threshold,
            max_bo1_market_gap=self.max_bo1_market_gap,
            bo1_max_odds=self.bo1_max_odds,
            prob_market_close_novig=prob_market_close_novig,
            max_negative_clv_drift=self.max_negative_clv_drift,
            max_coinflip_market_gap=self.max_coinflip_market_gap,
            coinflip_min_odds=self.coinflip_min_odds,
            coinflip_max_odds=self.coinflip_max_odds,
            uncertainty_required=self.uncertainty_required,
            uncertainty_basis=uncertainty_basis,
            uncertainty_reason=uncertainty_reason,
        )

        ev_net = diag["ev_net"]
        net_mult = 1.0 - self.tax_rate
        pnl = flat_stake * (odds * net_mult - 1.0) if won else -flat_stake
        clv_pct = ((odds / odds_close) - 1.0) * 100.0 if odds_close and odds_close > 0 else None
        clv_pp = ((1.0 / odds_close) - (1.0 / odds)) * 100.0 if odds_close and odds and odds_close > 0 and odds > 0 else None
        # Match entropy calculation (aleatoric randomness in [0, 1])
        # H = -(p*log2(p) + (1-p)*log2(1-p)). Max = 1.0 at p=0.50, low on favorites
        p_c = max(1e-6, min(1.0 - 1e-6, float(prob_model)))
        entropy_val = -(p_c * math.log2(p_c) + (1.0 - p_c) * math.log2(1.0 - p_c))
        entropy_val = max(0.0, min(1.0, float(entropy_val)))

        # Historical in-sample heuristic; neither a measured probability bound
        # nor evidence of forward EV calibration. Preserve its fixed coefficients.
        beta_shrink = 0.73
        odds_dampener = 1.0 + 0.15 * max(0.0, odds - 1.0)
        entropy_haircut = 1.0 - 0.15 * entropy_val
        ev_calibrated = None
        if ev_net is not None:
            ev_calibrated = (float(ev_net) * beta_shrink / odds_dampener) * entropy_haircut
            ev_calibrated = max(0.0, min(0.25, ev_calibrated))
        bet = EvaluatedBet(
            match_id=match_id,
            side=side,
            bookmaker=bookmaker,
            odds=odds,
            odds_close=odds_close,
            prob_model=prob_model,
            prob_conservative=diag["prob_conservative"],
            prob_for_ev=diag["prob_for_ev"],
            uncertainty_basis=diag["uncertainty_basis"],
            uncertainty_reason=diag["uncertainty_reason"],
            ev_net=round(float(ev_net), 4) if ev_net is not None else None,
            ev_calibrated=(
                round(float(ev_calibrated), 4) if ev_calibrated is not None else None
            ),
            won=won,
            stake=flat_stake,
            pnl=round(float(pnl), 2),
            clv_pct=round(float(clv_pct), 2) if clv_pct is not None else None,
            clv_pp=round(float(clv_pp), 2) if clv_pp is not None else None,
            competition_tier=tier,
            best_of=best_of,
            quarantine=diag.get("quarantine", False),
            quarantine_reason=diag.get("quarantine_reason", reason if not eligible else None),
        )
        return eligible, bet

    def run_simulation(
        self,
        qualified_bets: Sequence[EvaluatedBet],
        strategy: str = "flat",
        initial_bankroll: float = 1000.0,
        flat_stake: float = 100.0,
        cap_fraction: float = 0.025,
        quarter_kelly_fraction: float = 0.25,
        ev_bins: Sequence[tuple[str, float, float]] | None = None,
    ) -> SimulationSummary:
        """Run bankroll simulation and compute full per-bet calibration metrics."""
        if not qualified_bets:
            return SimulationSummary(
                strategy=strategy,
                initial_bankroll=initial_bankroll,
                final_bankroll=initial_bankroll,
                total_staked=0.0,
                total_pnl=0.0,
                roi_pct=None,
                bets_count=0,
                wins_count=0,
                win_rate_pct=None,
                avg_odds=None,
                avg_stake=None,
                expected_ev_pct=None,
                raw_ev_pct=None,
                expected_profit_per_bet=None,
                realized_ev_pct=None,
                realized_profit_per_bet=None,
                calibration_gap_pln=None,
                max_drawdown_pct=0.0,
                max_drawdown_pln=0.0,
                median_clv_pct=None,
                mean_clv_pp=None,
                result_status="no_eligible_bets",
            )

        bankroll = float(initial_bankroll)
        trajectory = [bankroll]
        staked_amounts: list[float] = []
        pnl_amounts: list[float] = []
        net_mult = 1.0 - self.tax_rate

        evaluated_records: list[dict[str, Any]] = []

        for bet in qualified_bets:
            p_for_ev = bet.prob_for_ev
            if strategy == "quarter_kelly":
                stk = calculate_quarter_kelly_stake(
                    bankroll=bankroll,
                    prob_for_ev=p_for_ev,
                    odds=bet.odds,
                    tax_rate=self.tax_rate,
                    cap_fraction=cap_fraction,
                    fraction=quarter_kelly_fraction,
                )
            elif strategy == "fixed_fraction":
                stk = bankroll * cap_fraction
            else:  # flat
                stk = float(flat_stake)

            pnl = stk * (bet.odds * net_mult - 1.0) if bet.won else -stk
            bankroll += pnl
            trajectory.append(bankroll)
            staked_amounts.append(stk)
            pnl_amounts.append(pnl)

            rec = bet.to_dict()
            rec["sim_stake"] = stk
            rec["sim_pnl"] = pnl
            evaluated_records.append(rec)

        df_res = pd.DataFrame(evaluated_records)

        # Peak and drawdown
        traj_series = pd.Series(trajectory)
        peak_series = traj_series.cummax()
        dd_series = (peak_series - traj_series) / peak_series
        max_dd_pct = float(dd_series.max() * 100.0)
        max_dd_pln = float((peak_series - traj_series).max())

        total_staked = float(sum(staked_amounts))
        total_pnl = float(sum(pnl_amounts))
        final_bankroll = float(bankroll)
        n_bets = len(qualified_bets)
        wins = int(df_res["won"].sum())
        win_rate = float(wins / n_bets * 100.0)
        avg_odds = float(df_res["odds"].mean())
        avg_stake = float(np.mean(staked_amounts))

        # Per-bet expected vs realized calibration
        raw_ev_pct = float(df_res["ev_net"].mean() * 100.0)
        exp_ev_pct = float(df_res["ev_calibrated"].mean() * 100.0)
        exp_profit_per_bet = float(avg_stake * (exp_ev_pct / 100.0))
        real_profit_per_bet = float(total_pnl / n_bets)
        real_ev_pct = float(total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0
        roi_pct = real_ev_pct
        calib_gap_pln = float(real_profit_per_bet - exp_profit_per_bet)
        clv_valid = df_res["clv_pct"].dropna()
        med_clv = float(clv_valid.median()) if len(clv_valid) else None
        clv_pp_valid = df_res["clv_pp"].dropna()
        mean_clv_pp = float(clv_pp_valid.mean()) if len(clv_pp_valid) else None

        # 1. Historical expected-versus-realized bins of heuristically adjusted EV
        if ev_bins is None:
            ev_bins = [
                ("0% - 5% EV", 0.00, 0.05),
                ("5% - 10% EV", 0.05, 0.10),
                ("10% - 15% EV", 0.10, 0.15),
                ("15% - 20% EV", 0.15, 0.20),
                ("> 20% EV", 0.20, 99.00),
            ]
        calib_bins: list[CalibrationBin] = []
        for lbl, low, high in ev_bins:
            sub = df_res[(df_res["ev_calibrated"] >= low) & (df_res["ev_calibrated"] < high)]
            if sub.empty:
                continue
            sub_n = len(sub)
            sub_wins = int(sub["won"].sum())
            sub_stk = float(sub["sim_stake"].sum())
            sub_pnl = float(sub["sim_pnl"].sum())
            sub_avg_stk = float(sub["sim_stake"].mean())
            sub_exp_ev = float(sub["ev_calibrated"].mean() * 100.0)
            sub_exp_prof = float(sub_avg_stk * (sub_exp_ev / 100.0))
            sub_real_prof = float(sub_pnl / sub_n)
            sub_real_ev = float(sub_pnl / sub_stk * 100.0) if sub_stk > 0 else 0.0
            realization = float(sub_real_ev / sub_exp_ev * 100.0) if sub_exp_ev != 0 else 0.0
            s_clv = sub["clv_pct"].dropna()
            s_clv_pp = sub["clv_pp"].dropna()

            calib_bins.append(
                CalibrationBin(
                    label=lbl,
                    bets_count=sub_n,
                    wins_count=sub_wins,
                    win_rate_pct=round(float(sub_wins / sub_n * 100.0), 1),
                    avg_odds=round(float(sub["odds"].mean()), 2),
                    avg_stake=round(sub_avg_stk, 2),
                    total_staked=round(sub_stk, 2),
                    total_pnl=round(sub_pnl, 2),
                    expected_ev_pct=round(sub_exp_ev, 2),
                    expected_profit_per_bet=round(sub_exp_prof, 2),
                    realized_ev_pct=round(sub_real_ev, 2),
                    realized_profit_per_bet=round(sub_real_prof, 2),
                    calibration_gap_pct=round(sub_real_ev - sub_exp_ev, 2),
                    calibration_gap_pln=round(sub_real_prof - sub_exp_prof, 2),
                    ev_realization_rate_pct=round(realization, 1),
                    median_clv_pct=round(float(s_clv.median()), 2) if len(s_clv) else None,
                    mean_clv_pp=round(float(s_clv_pp.mean()), 2) if len(s_clv_pp) else None,
                )
            )

        # 2. Breakdown by odds bracket
        odds_brackets = [
            ("< 1.80", 0.0, 1.80),
            ("1.80 - 2.50", 1.80, 2.50),
            ("2.50 - 3.50", 2.50, 3.50),
            ("3.50 - 5.00", 3.50, 5.00),
            ("> 5.00", 5.00, 999.0),
        ]
        odds_bins: list[CalibrationBin] = []
        for lbl, low, high in odds_brackets:
            sub = df_res[(df_res["odds"] >= low) & (df_res["odds"] < high)]
            if sub.empty:
                continue
            sub_n = len(sub)
            sub_wins = int(sub["won"].sum())
            sub_stk = float(sub["sim_stake"].sum())
            sub_pnl = float(sub["sim_pnl"].sum())
            sub_avg_stk = float(sub["sim_stake"].mean())
            sub_exp_ev = float(sub["ev_net"].mean() * 100.0)
            sub_exp_prof = float(sub_avg_stk * (sub_exp_ev / 100.0))
            sub_real_prof = float(sub_pnl / sub_n)
            sub_real_ev = float(sub_pnl / sub_stk * 100.0) if sub_stk > 0 else 0.0
            realization = float(sub_real_ev / sub_exp_ev * 100.0) if sub_exp_ev != 0 else 0.0
            s_clv = sub["clv_pct"].dropna()
            s_clv_pp = sub["clv_pp"].dropna()

            odds_bins.append(
                CalibrationBin(
                    label=lbl,
                    bets_count=sub_n,
                    wins_count=sub_wins,
                    win_rate_pct=round(float(sub_wins / sub_n * 100.0), 1),
                    avg_odds=round(float(sub["odds"].mean()), 2),
                    avg_stake=round(sub_avg_stk, 2),
                    total_staked=round(sub_stk, 2),
                    total_pnl=round(sub_pnl, 2),
                    expected_ev_pct=round(sub_exp_ev, 2),
                    expected_profit_per_bet=round(sub_exp_prof, 2),
                    realized_ev_pct=round(sub_real_ev, 2),
                    realized_profit_per_bet=round(sub_real_prof, 2),
                    calibration_gap_pct=round(sub_real_ev - sub_exp_ev, 2),
                    calibration_gap_pln=round(sub_real_prof - sub_exp_prof, 2),
                    ev_realization_rate_pct=round(realization, 1),
                    median_clv_pct=round(float(s_clv.median()), 2) if len(s_clv) else None,
                    mean_clv_pp=round(float(s_clv_pp.mean()), 2) if len(s_clv_pp) else None,
                )
            )

        # 3. By bookmaker
        by_bm: list[dict[str, Any]] = []
        if "bookmaker" in df_res.columns:
            for bm_name, sub in df_res.groupby("bookmaker"):
                b_n = len(sub)
                b_wins = int(sub["won"].sum())
                b_pnl = float(sub["sim_pnl"].sum())
                b_stk = float(sub["sim_stake"].sum())
                s_clv = sub["clv_pct"].dropna()
                s_clv_pp = sub["clv_pp"].dropna()
                by_bm.append({
                    "bookmaker": str(bm_name),
                    "bets": b_n,
                    "wins": b_wins,
                    "win_rate_pct": round(float(b_wins / b_n * 100.0), 1),
                    "avg_odds": round(float(sub["odds"].mean()), 2),
                    "pnl": round(b_pnl, 2),
                    "roi_pct": round(float(b_pnl / b_stk * 100.0), 1) if b_stk > 0 else 0.0,
                    "clv": round(float(s_clv.median()), 1) if len(s_clv) else 0.0,
                    "clv_pp": round(float(s_clv_pp.mean()), 2) if len(s_clv_pp) else 0.0,
                })
            by_bm.sort(key=lambda x: x["pnl"], reverse=True)

        return SimulationSummary(
            strategy=strategy,
            initial_bankroll=round(initial_bankroll, 2),
            final_bankroll=round(final_bankroll, 2),
            total_staked=round(total_staked, 2),
            total_pnl=round(total_pnl, 2),
            roi_pct=round(roi_pct, 2),
            bets_count=n_bets,
            wins_count=wins,
            win_rate_pct=round(win_rate, 2),
            avg_odds=round(avg_odds, 2),
            avg_stake=round(avg_stake, 2),
            expected_ev_pct=round(exp_ev_pct, 2),
            raw_ev_pct=round(raw_ev_pct, 2),
            expected_profit_per_bet=round(exp_profit_per_bet, 2),
            realized_ev_pct=round(real_ev_pct, 2),
            realized_profit_per_bet=round(real_profit_per_bet, 2),
            calibration_gap_pln=round(calib_gap_pln, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_pln=round(max_dd_pln, 2),
            median_clv_pct=round(med_clv, 2) if med_clv is not None else None,
            mean_clv_pp=round(mean_clv_pp, 2) if mean_clv_pp is not None else None,
            bins=calib_bins,
            by_odds_bracket=odds_bins,
            by_bookmaker=by_bm,
        )
