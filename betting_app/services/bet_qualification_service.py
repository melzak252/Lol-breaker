"""Capital risk qualification and safety gating for betting recommendations."""

from __future__ import annotations

import json
import math
from typing import Any

from betting_app.core.models.registry import get_active_hybrid, get_model

DEFAULT_BASE_MIN_EV: float = 0.05
DEFAULT_HURDLE_ODDS_CUTOFF: float = 2.20
DEFAULT_HURDLE_PENALTY_SCALE: float = 0.08
DEFAULT_MAX_RATING_DISAGREEMENT: float = 0.15
DEFAULT_MAX_TIER1_MARKET_GAP: float = 0.08
DEFAULT_MAX_EV_NET: float = 0.25
DEFAULT_MAX_EV_LONGSHOT: float = 0.25
DEFAULT_LONGSHOT_ODDS_THRESHOLD: float = 3.50
DEFAULT_MAX_BO1_MARKET_GAP: float = 0.12
DEFAULT_MAX_NEGATIVE_CLV_DRIFT: float = -0.015
DEFAULT_MAX_COINFLIP_MARKET_GAP: float | None = None
DEFAULT_COINFLIP_MIN_ODDS: float = 1.80
DEFAULT_COINFLIP_MAX_ODDS: float = 2.50
DEFAULT_BO1_MAX_ODDS: float | None = None
DEFAULT_MAX_LOGIT_DISAGREEMENT: float | None = 0.85
DEFAULT_WASSERSTEIN_EPSILON: float = 0.025
def model_requires_uncertainty(model_name: str) -> bool:
    spec = get_model(model_name)
    if spec is not None:
        return spec.has_uncertainty
    hybrid = get_active_hybrid()
    return model_name == hybrid.hybrid_model_name and hybrid.base_model.has_uncertainty


def prediction_safety_diagnostics(
    raw: Any,
    prob_a: float,
    prob_b: float,
    *,
    uncertainty_required: bool,
) -> dict[str, Any]:
    """Validate persisted decision inputs; old one-sided diagnostics cannot qualify."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("malformed prediction diagnostics JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("prediction diagnostics must be an object")
    if not (
        math.isfinite(prob_a)
        and math.isfinite(prob_b)
        and 0.0 < prob_a < 1.0
        and 0.0 < prob_b < 1.0
        and math.isclose(prob_a + prob_b, 1.0, abs_tol=1e-5)
    ):
        raise ValueError("prediction means must be finite and complementary")
    marker = raw.get("uncertainty_required", False)
    if not isinstance(marker, bool):
        raise ValueError("uncertainty_required must be a boolean")
    required = (
        uncertainty_required
        or marker
        or model_requires_uncertainty(str(raw.get("base_model_name", "")))
    )
    required = required or any(
        raw.get(key) is not None
        for key in (
            "p_low_a",
            "p_low_b",
            "epistemic_sigma_z",
            "prob_risk_adjusted_p_low",
        )
    )
    diag = dict(raw)
    diag["uncertainty_required"] = required
    for key, maximum in (("p_low_a", prob_a), ("p_low_b", prob_b)):
        value = diag.get(key)
        if value is not None and (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= maximum
        ):
            raise ValueError(
                f"{key} must be finite and between zero and its side mean"
            )
    basis = diag.get("uncertainty_basis")
    if basis is None:
        basis = (
            "measured_bounds"
            if any(diag.get(key) is not None for key in ("p_low_a", "p_low_b"))
            else "missing_bounds" if required else "mean"
        )
    if basis not in {
        "mean", "measured_bounds", "assumed_research_haircut", "missing_bounds"
    }:
        raise ValueError("unknown uncertainty_basis")
    diag["uncertainty_basis"] = basis
    diag["uncertainty_reason"] = diag.get("uncertainty_reason") or {
        "mean": "uncertainty_not_requested",
        "measured_bounds": "provided_lower_bound",
        "assumed_research_haircut": "explicit_research_assumption",
        "missing_bounds": "missing_uncertainty",
    }[basis]
    if required and any(diag.get(key) is None for key in ("p_low_a", "p_low_b")):
        diag["uncertainty_basis"] = "missing_bounds"
        diag["uncertainty_reason"] = "missing_uncertainty"
        return diag
    sigma = diag.get("epistemic_sigma_z")
    if sigma is not None and (
        not isinstance(sigma, (int, float))
        or not math.isfinite(sigma)
        or sigma < 0.0
    ):
        raise ValueError("epistemic_sigma_z must be finite and nonnegative")
    disagreement = diag.get("rating_disagreement")
    if required or disagreement is not None:
        if (
            not isinstance(disagreement, (int, float))
            or not math.isfinite(disagreement)
            or not 0.0 <= disagreement <= 1.0
        ):
            raise ValueError("rating_disagreement must be finite and in [0, 1]")
    return diag


def dynamic_min_ev_hurdle(
    odds: float,
    base_min_ev: float = DEFAULT_BASE_MIN_EV,
    odds_cutoff: float = DEFAULT_HURDLE_ODDS_CUTOFF,
    penalty_scale: float = DEFAULT_HURDLE_PENALTY_SCALE,
) -> float:
    """Apply an explicit extra EV margin to low-odds bets (a policy, not a guarantee).

    The penalty has not been shown to eliminate selection bias or favorite losses:
        EV_min(odds) = base_min_ev + max(0, (odds_cutoff - odds) / odds_cutoff) * penalty_scale
    """
    if not all(
        math.isfinite(value)
        for value in (odds, base_min_ev, odds_cutoff, penalty_scale)
    ):
        raise ValueError("EV hurdle inputs must be finite")
    if odds <= 1.0 or odds_cutoff <= 1.0 or base_min_ev < 0.0 or penalty_scale < 0.0:
        raise ValueError(
            "EV hurdle requires odds/cutoff > 1 and nonnegative thresholds"
        )
    if odds >= odds_cutoff:
        return base_min_ev
    penalty = ((odds_cutoff - odds) / odds_cutoff) * penalty_scale
    return base_min_ev + penalty


def is_bet_eligible(
    prob_model: float,
    odds: float,
    prob_market_novig: float | None = None,
    tax_rate: float = 0.12,
    min_ev_net: float = DEFAULT_BASE_MIN_EV,
    dynamic_hurdle: bool = True,
    hurdle_odds_cutoff: float = DEFAULT_HURDLE_ODDS_CUTOFF,
    hurdle_penalty_scale: float = DEFAULT_HURDLE_PENALTY_SCALE,
    rating_disagreement: float | None = None,
    max_rating_disagreement: float | None = DEFAULT_MAX_RATING_DISAGREEMENT,
    competition_tier: str | None = None,
    max_tier1_market_gap: float | None = DEFAULT_MAX_TIER1_MARKET_GAP,
    max_ev_net: float | None = None,
    tier1_only: bool = False,
    best_of: int | None = None,
    max_ev_longshot: float | None = DEFAULT_MAX_EV_LONGSHOT,
    longshot_odds_threshold: float = DEFAULT_LONGSHOT_ODDS_THRESHOLD,
    max_bo1_market_gap: float | None = DEFAULT_MAX_BO1_MARKET_GAP,
    bo1_max_odds: float | None = DEFAULT_BO1_MAX_ODDS,
    prob_market_close_novig: float | None = None,
    max_negative_clv_drift: float | None = DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
    max_coinflip_market_gap: float | None = DEFAULT_MAX_COINFLIP_MARKET_GAP,
    coinflip_min_odds: float = DEFAULT_COINFLIP_MIN_ODDS,
    coinflip_max_odds: float = DEFAULT_COINFLIP_MAX_ODDS,
    max_logit_disagreement: float | None = DEFAULT_MAX_LOGIT_DISAGREEMENT,
    universal_rating_guard: bool = False,
    max_odds: float | None = None,
    wasserstein_epsilon: float | None = None,
    *,
    prob_conservative: float | None = None,
    uncertainty_required: bool = False,
    uncertainty_basis: str | None = None,
    uncertainty_reason: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Determine whether a proposed bet satisfies mathematical EV and safety boundaries.

    Parameters
    ----------
    prob_model : float
        Calibrated model probability of winning (in (0, 1)).
        Retained for display and mean-market divergence guards.
    prob_conservative : float | None
        Side-specific conservative score used for EV; never the other side's complement.
    uncertainty_required : bool
        Reject missing/invalid conservative scores rather than falling back to the mean.
    uncertainty_basis : str | None
        Evidence identity: mean, measured_bounds, assumed_research_haircut, or
        missing_bounds. A provided lower score defaults to measured_bounds.
    uncertainty_reason : str | None
        Persisted evidence explanation, retained when the declared score is valid.
    odds : float
        Decimal bookmaker odds (> 1.0).
    prob_market_novig : float | None, optional
        Consensus no-vig fair market probability, if available.
    tax_rate : float, default=0.12
        Mandatory turnover tax fraction (e.g. 0.12 for Poland).
    min_ev_net : float, default=0.05
        Base minimum net expected value required for qualification (+5%).
    dynamic_hurdle : bool, default=True
        Whether to require an additional EV margin on low odds.
    hurdle_odds_cutoff : float, default=2.20
        Odds cutoff below which the dynamic EV penalty applies.
    hurdle_penalty_scale : float, default=0.08
        Maximum additional EV required as odds approach 0.

    Returns
    -------
    tuple[bool, str, dict[str, Any]]
        (is_eligible, reason_code, diagnostics)
    """
    if not 0.0 < prob_model < 1.0:
        raise ValueError(f"prob_model must be in (0, 1), got {prob_model}")
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError(f"odds must be > 1.0 and finite, got {odds}")
    if not 0.0 <= tax_rate < 1.0:
        raise ValueError(f"tax_rate must be in [0, 1), got {tax_rate}")
    if prob_market_novig is not None and not 0.0 < prob_market_novig < 1.0:
        raise ValueError(
            f"prob_market_novig must be in (0, 1), got {prob_market_novig}"
        )
    if prob_market_close_novig is not None and not 0.0 < prob_market_close_novig < 1.0:
        raise ValueError(
            f"prob_market_close_novig must be in (0, 1), got {prob_market_close_novig}"
        )
    if best_of is not None and (not isinstance(best_of, int) or isinstance(best_of, bool) or best_of <= 0):
        raise ValueError("best_of must be a positive integer")
    for name, value in (
        ("min_ev_net", min_ev_net),
        ("hurdle_penalty_scale", hurdle_penalty_scale),
        ("rating_disagreement", rating_disagreement),
        ("max_rating_disagreement", max_rating_disagreement),
        ("max_tier1_market_gap", max_tier1_market_gap),
        ("max_ev_longshot", max_ev_longshot),
        ("max_bo1_market_gap", max_bo1_market_gap),
    ):
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if wasserstein_epsilon is not None and (
        not isinstance(wasserstein_epsilon, (int, float))
        or not math.isfinite(wasserstein_epsilon)
        or not 0.0 <= wasserstein_epsilon < 1.0
    ):
        raise ValueError("wasserstein_epsilon must be finite and in [0, 1)")
    if max_negative_clv_drift is not None and (
        not isinstance(max_negative_clv_drift, (int, float)) or not math.isfinite(max_negative_clv_drift)
    ):
        raise ValueError("max_negative_clv_drift must be finite")
    if not math.isfinite(longshot_odds_threshold) or longshot_odds_threshold <= 1.0:
        raise ValueError("longshot_odds_threshold must be finite and > 1.0")
    if max_ev_net is not None and (
        not isinstance(max_ev_net, (int, float)) or not math.isfinite(max_ev_net) or max_ev_net <= 0
    ):
        raise ValueError("max_ev_net must be finite and positive")
    if not math.isfinite(hurdle_odds_cutoff) or hurdle_odds_cutoff <= 1.0:
        raise ValueError("hurdle_odds_cutoff must be finite and > 1.0")
    if not isinstance(uncertainty_required, bool):
        raise ValueError("uncertainty_required must be a boolean")
    if uncertainty_basis is None:
        uncertainty_basis = (
            "measured_bounds" if prob_conservative is not None
            else "missing_bounds" if uncertainty_required else "mean"
        )
    if uncertainty_basis not in {
        "mean", "measured_bounds", "assumed_research_haircut", "missing_bounds"
    }:
        raise ValueError("unknown uncertainty_basis")
    evidence = {
        "prob_model": prob_model,
        "prob_conservative": prob_conservative,
        "prob_for_ev": None,
        "ev_net": None,
        "uncertainty_required": uncertainty_required,
        "uncertainty_basis": uncertainty_basis,
        "uncertainty_reason": uncertainty_reason or {
            "mean": "uncertainty_not_requested",
            "measured_bounds": "provided_lower_bound",
            "assumed_research_haircut": "explicit_research_assumption",
            "missing_bounds": "missing_uncertainty",
        }[uncertainty_basis],
    }
    if uncertainty_basis == "missing_bounds" or (
        prob_conservative is None
        and (uncertainty_required or uncertainty_basis != "mean")
    ):
        evidence.update(
            prob_conservative=None,
            uncertainty_basis="missing_bounds",
            uncertainty_reason="missing_uncertainty",
            quarantine=True,
            quarantine_reason="missing_uncertainty",
        )
        return False, "missing_uncertainty", evidence
    if prob_conservative is None:
        prob_for_ev = prob_model
    elif (
        uncertainty_basis == "mean"
        or not isinstance(prob_conservative, (int, float))
        or not math.isfinite(prob_conservative)
        or not 0.0 <= prob_conservative <= prob_model
    ):
        evidence.update(
            uncertainty_reason="invalid_uncertainty",
            quarantine=True,
            quarantine_reason="invalid_uncertainty",
        )
        return False, "invalid_uncertainty", evidence
    else:
        prob_for_ev = prob_conservative

    return_factor = odds * (1.0 - tax_rate)
    ev_net = (prob_for_ev * return_factor) - 1.0

    if dynamic_hurdle:
        effective_min_ev = dynamic_min_ev_hurdle(
            odds=odds,
            base_min_ev=min_ev_net,
            odds_cutoff=hurdle_odds_cutoff,
            penalty_scale=hurdle_penalty_scale,
        )
    else:
        effective_min_ev = min_ev_net

    diag: dict[str, Any] = {
        **evidence,
        "prob_for_ev": prob_for_ev,
        "odds": odds,
        "tax_rate": tax_rate,
        "prob_market_novig": prob_market_novig,
        "ev_net": round(ev_net, 4),
        "min_ev_net": min_ev_net,
        "effective_min_ev": round(effective_min_ev, 4),
        "dynamic_hurdle": dynamic_hurdle,
        "quarantine": False,
        "max_ev_net": max_ev_net,
        "best_of": best_of,
        "max_ev_longshot": max_ev_longshot,
        "longshot_odds_threshold": longshot_odds_threshold,
        "max_bo1_market_gap": max_bo1_market_gap,
        "prob_market_close_novig": prob_market_close_novig,
        "max_negative_clv_drift": max_negative_clv_drift,
    }
    if prob_market_novig is not None:
        diag["market_gap"] = round(prob_model - prob_market_novig, 4)
    if prob_market_close_novig is not None and prob_market_novig is not None:
        diag["clv_drift"] = round(prob_market_close_novig - prob_market_novig, 4)
    if rating_disagreement is not None:
        diag["rating_disagreement"] = round(rating_disagreement, 4)
    if competition_tier is not None:
        diag["competition_tier"] = competition_tier
    if ev_net < effective_min_ev:
        return False, "insufficient_ev", diag

    if wasserstein_epsilon is not None and wasserstein_epsilon > 0.0:
        prob_wasserstein = max(0.0, prob_for_ev - float(wasserstein_epsilon))
        ev_wasserstein = (prob_wasserstein * return_factor) - 1.0
        diag["wasserstein_epsilon"] = round(float(wasserstein_epsilon), 4)
        diag["ev_wasserstein"] = round(ev_wasserstein, 4)
        if ev_wasserstein <= 0.0:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "wasserstein_dro_ambiguous_edge"
            return False, "wasserstein_dro_ambiguous_edge", diag
    if max_ev_net is not None and ev_net > max_ev_net:
        diag["quarantine"] = True
        diag["quarantine_reason"] = "extreme_ev_overconfidence"
        return False, "extreme_ev_overconfidence", diag
    # Physical odds ceiling (variance elimination)
    if max_odds is not None and odds > max_odds:
        diag["quarantine"] = True
        diag["quarantine_reason"] = "extreme_odds_variance_cap"
        return False, "extreme_odds_variance_cap", diag

    # Rule A: EV Ceiling on Longshots
    if (
        odds > longshot_odds_threshold
        and max_ev_longshot is not None
        and ev_net > max_ev_longshot
    ):
        diag["quarantine"] = True
        diag["quarantine_reason"] = "extreme_ev_longshot_cap"
        return False, "extreme_ev_longshot_cap", diag

    # Rule B: Bo1 Discrepancy Quarantine
    if best_of == 1 and bo1_max_odds is not None and odds > bo1_max_odds:
        diag["quarantine"] = True
        diag["quarantine_reason"] = "bo1_longshot_variance_cap"
        return False, "bo1_longshot_variance_cap", diag

    if (
        best_of == 1
        and prob_market_novig is not None
        and max_bo1_market_gap is not None
    ):
        if abs(prob_model - prob_market_novig) >= max_bo1_market_gap:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "bo1_extreme_market_divergence"
            return False, "bo1_extreme_market_divergence", diag

    # Rule C: Negative CLV Drift Quarantine
    if (
        prob_market_close_novig is not None
        and prob_market_novig is not None
        and max_negative_clv_drift is not None
    ):
        clv_drift = prob_market_close_novig - prob_market_novig
        if clv_drift <= max_negative_clv_drift:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "negative_market_drift_clv"
            return False, "negative_market_drift_clv", diag

    # ISSUE-003 / Thesis Gating: Reject bets with high internal rating/model disagreement
    if (
        (universal_rating_guard or odds <= hurdle_odds_cutoff)
        and rating_disagreement is not None
        and max_rating_disagreement is not None
        and rating_disagreement > max_rating_disagreement
    ):
        diag["quarantine"] = True
        diag["quarantine_reason"] = "severe_rating_disagreement"
        return False, "severe_rating_disagreement", diag
    # ISSUE-003 Gating: Reject sharp Tier-1 divergence without market confirmation
    if (
        competition_tier in ("Tier-1 International", "Tier-1 Domestic")
        and prob_market_novig is not None
        and max_tier1_market_gap is not None
    ):
        market_gap = prob_model - prob_market_novig
        if market_gap > max_tier1_market_gap:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "tier1_sharp_market_divergence"
            return False, "tier1_sharp_market_divergence", diag
    # Rule D (CALIBRATION-001): Reject coin-flip overconfidence trap in 1.80-2.50 bracket
    if (
        coinflip_min_odds <= odds <= coinflip_max_odds
        and prob_market_novig is not None
        and max_coinflip_market_gap is not None
    ):
        market_gap = abs(prob_model - prob_market_novig)
        if market_gap >= max_coinflip_market_gap:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "coinflip_market_divergence_trap"
            return False, "coinflip_market_divergence_trap", diag

    # Guard B: Dynamic Logit Disagreement Gate (Tail-sensitive protection against Optimizer's Curse)
    if (
        prob_market_novig is not None
        and max_logit_disagreement is not None
        and 0.0 < prob_model < 1.0
        and 0.0 < prob_market_novig < 1.0
    ):
        from src.analysis.betting_guards_a_to_d import dynamic_logit_disagreement
        logit_gap = dynamic_logit_disagreement(prob_model, prob_market_novig)
        diag["logit_disagreement"] = round(logit_gap, 4)
        if logit_gap > max_logit_disagreement:
            diag["quarantine"] = True
            diag["quarantine_reason"] = "dynamic_logit_market_divergence"
            return False, "dynamic_logit_market_divergence", diag
    # Tier-1 Gating: Reject regional/secondary leagues without referee verification when requested
    if tier1_only and competition_tier not in ("Tier-1 International", "Tier-1 Domestic"):
        diag["quarantine"] = True
        diag["quarantine_reason"] = "non_tier1_unverified_competition"
        return False, "non_tier1_unverified_competition", diag
    return True, "eligible", diag


def qualification_tier(league: str | None, match_date: Any = None) -> str:
    """Map competition identity to the established safety-policy vocabulary."""
    from datetime import date
    from src.models.competition_tiers import CompetitionTier, classify_competition

    effective_date = None
    if match_date is not None and not (isinstance(match_date, float) and math.isnan(match_date)):
        s = str(match_date).strip()[:10]
        if s and s.lower() not in ("nan", "none"):
            try:
                effective_date = date.fromisoformat(s)
            except ValueError:
                effective_date = None
    tier = classify_competition(league, effective_date).tier
    return {
        CompetitionTier.INTERNATIONAL: "Tier-1 International",
        CompetitionTier.MAJOR: "Tier-1 Domestic",
    }.get(tier, tier.value)


def qualify_prediction_sides(
    *,
    prob_a: float,
    prob_b: float,
    diagnostics: Any,
    model_name: str,
    odds_a: float,
    odds_b: float,
    market_prob_a: float,
    league: str | None = None,
    match_date: Any = None,
    tax_rate: float = 0.12,
    min_ev_net: float = DEFAULT_BASE_MIN_EV,
    max_ev_net: float | None = None,
    best_of: int | None = None,
    market_prob_close_a: float | None = None,
    max_ev_longshot: float | None = DEFAULT_MAX_EV_LONGSHOT,
    longshot_odds_threshold: float = DEFAULT_LONGSHOT_ODDS_THRESHOLD,
    max_bo1_market_gap: float | None = DEFAULT_MAX_BO1_MARKET_GAP,
    max_negative_clv_drift: float | None = DEFAULT_MAX_NEGATIVE_CLV_DRIFT,
) -> dict[str, tuple[bool, str, dict[str, Any]]]:
    """Shared read-side gate; malformed persisted predictions never recommend."""

    try:
        diag = prediction_safety_diagnostics(
            diagnostics,
            prob_a,
            prob_b,
            uncertainty_required=model_requires_uncertainty(model_name),
        )
        tier_name = qualification_tier(league, match_date)
        effective_best_of = best_of if best_of is not None else diag.get("best_of")
        if isinstance(effective_best_of, str):
            try:
                effective_best_of = int(effective_best_of)
            except ValueError:
                effective_best_of = None
        return {
            side: is_bet_eligible(
                prob_model=prob,
                odds=odds,
                prob_market_novig=market,
                tax_rate=tax_rate,
                min_ev_net=min_ev_net,
                rating_disagreement=diag.get("rating_disagreement"),
                competition_tier=tier_name,
                max_ev_net=max_ev_net,
                best_of=effective_best_of,
                max_ev_longshot=max_ev_longshot,
                longshot_odds_threshold=longshot_odds_threshold,
                max_bo1_market_gap=max_bo1_market_gap,
                prob_market_close_novig=close_market,
                max_negative_clv_drift=max_negative_clv_drift,
                prob_conservative=diag.get(f"p_low_{side}"),
                uncertainty_required=diag["uncertainty_required"],
                uncertainty_basis=diag["uncertainty_basis"],
                uncertainty_reason=diag["uncertainty_reason"],
            )
            for side, prob, odds, market, close_market in (
                ("a", prob_a, odds_a, market_prob_a, market_prob_close_a),
                (
                    "b",
                    prob_b,
                    odds_b,
                    1.0 - market_prob_a,
                    (1.0 - market_prob_close_a) if market_prob_close_a is not None else None,
                ),
            )
        }
    except (TypeError, ValueError):
        return {side: (False, "invalid_prediction_safety", {}) for side in ("a", "b")}
