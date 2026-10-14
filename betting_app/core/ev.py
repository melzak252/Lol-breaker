"""Expected value and market probability calculations."""

from src.analysis.devig_comparison import devig_multiplicative, devig_shin

def implied_probability(decimal_odds: float) -> float:
    """Return raw implied probability from decimal odds."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal_odds


def fair_market_probabilities(odds_a: float, odds_b: float, method: str = "shin") -> tuple[float, float]:
    """Remove two-way bookmaker margin using Shin (1992) or Multiplicative devigging.

    Parameters
    ----------
    odds_a, odds_b : float
        Decimal odds (> 1.0).
    method : {"shin", "multiplicative", "power", "additive"}, default="shin"
        Method for stripping bookmaker margin. "shin" eliminates empirical
        favorite-longshot bias.
    """
    if method == "multiplicative":
        return devig_multiplicative(odds_a, odds_b)
    elif method in ("shin", "additive"):
        return devig_shin(odds_a, odds_b)
    elif method == "power":
        from src.analysis.devig_comparison import devig_power
        return devig_power(odds_a, odds_b)
    raise ValueError(f"Unknown devigging method: {method}")
def expected_value(model_prob: float, decimal_odds: float, tax_rate: float = 0.12) -> float:
    """Calculate expected value with Polish betting tax convention."""

    if not 0 <= model_prob <= 1:
        raise ValueError("model_prob must be in [0, 1]")
    if not 0 <= tax_rate < 1:
        raise ValueError("tax_rate must be in [0, 1)")
    return model_prob * decimal_odds * (1.0 - tax_rate) - 1.0


def best_ev_side(prob_a: float, odds_a: float, odds_b: float, tax_rate: float, min_ev: float) -> dict[str, float | str] | None:
    """Select the side with the highest EV if it crosses the threshold."""

    prob_b = 1.0 - prob_a
    ev_a = expected_value(prob_a, odds_a, tax_rate)
    ev_b = expected_value(prob_b, odds_b, tax_rate)
    market_a, market_b = fair_market_probabilities(odds_a, odds_b)

    if ev_a >= ev_b and ev_a > min_ev:
        return {"side": "a", "odds": odds_a, "model_prob": prob_a, "market_prob": market_a, "ev": ev_a}
    if ev_b > min_ev:
        return {"side": "b", "odds": odds_b, "model_prob": prob_b, "market_prob": market_b, "ev": ev_b}
    return None


def qualify_optimal_synergy_bet(
    prob_c0: float,
    prob_thesis: float,
    prob_market_novig: float,
    odds: float,
    tax_rate: float = 0.12,
    alpha: float = 0.50,
    min_ev: float = 0.05,
    max_ev: float = 0.25,
    max_odds: float = 4.00,
    max_disagreement: float = 0.10,
    max_logit_gap: float = 0.85,
    wasserstein_epsilon: float = 0.025,
    eps: float = 1e-6,
) -> tuple[bool, str, dict[str, float]]:
    """Production qualification for the Optimal Synergy + Wasserstein DRO Strategy.

    1. Rejects if odds > max_odds (physical tail variance elimination).
    2. Vetoes if |prob_c0 - prob_thesis| > max_disagreement (Thesis Guard <= 10%).
    3. Vetoes if |logit(prob_c0) - logit(prob_market)| > max_logit_gap (Guard B <= 0.85).
    4. Shrinks prob_c0 with prob_market using Bayesian logit shrinkage (alpha=0.50).
    5. Enforces Wasserstein DRO ambiguity ball (prob_hybrid - wasserstein_epsilon) * (1 - tax) * odds - 1 > 0.
    6. Qualifies if net post-tax EV is in [min_ev, max_ev].
    """
    import math

    if odds > max_odds:
        return False, "odds_exceeds_max_threshold", {"odds": odds, "max_odds": max_odds}

    delta_models = abs(prob_c0 - prob_thesis)
    if delta_models > max_disagreement:
        return False, "severe_thesis_disagreement", {"delta_models": delta_models, "max_disagreement": max_disagreement}

    p_c0_c = max(eps, min(1.0 - eps, prob_c0))
    p_mkt_c = max(eps, min(1.0 - eps, prob_market_novig))
    logit_c0 = math.log(p_c0_c / (1.0 - p_c0_c))
    logit_mkt = math.log(p_mkt_c / (1.0 - p_mkt_c))

    logit_gap = abs(logit_c0 - logit_mkt)
    if logit_gap > max_logit_gap:
        return False, "dynamic_logit_market_divergence", {"logit_gap": logit_gap, "max_logit_gap": max_logit_gap}

    logit_blend = alpha * logit_c0 + (1.0 - alpha) * logit_mkt
    prob_hybrid = 1.0 / (1.0 + math.exp(-logit_blend))

    ev = expected_value(prob_hybrid, odds, tax_rate)
    prob_dro = max(0.0, prob_hybrid - max(0.0, float(wasserstein_epsilon)))
    ev_dro = expected_value(prob_dro, odds, tax_rate)
    clv_raw = (odds * prob_market_novig) - 1.0
    clv_net = (odds * (1.0 - tax_rate) * prob_market_novig) - 1.0

    diag = {
        "prob_c0": prob_c0,
        "prob_thesis": prob_thesis,
        "prob_market": prob_market_novig,
        "prob_hybrid": prob_hybrid,
        "prob_dro": prob_dro,
        "wasserstein_epsilon": float(wasserstein_epsilon),
        "delta_models": delta_models,
        "logit_gap": logit_gap,
        "odds": odds,
        "ev": ev,
        "ev_dro": ev_dro,
        "clv_raw": clv_raw,
        "clv_net": clv_net,
    }

    if ev < min_ev:
        return False, "insufficient_ev", diag
    if wasserstein_epsilon > 0.0 and ev_dro <= 0.0:
        return False, "wasserstein_dro_ambiguous_edge", diag
    if ev > max_ev:
        return False, "extreme_ev_overconfidence", diag

    return True, "eligible", diag
