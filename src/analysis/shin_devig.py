"""2-way Shin (1992) devigging implementation."""

import math
try:
    from scipy.optimize import brentq
except ImportError:
    brentq = None


def shin_implied_probabilities(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Calculate true fair probabilities by stripping bookmaker margin and longshot bias.

    Hyun Song Shin (1992): 'Prices of State Contingent Claims with Insider Traders,
    and the Favorite-Longshot Bias', Economic Journal, 102(411), 426-435.

    Solves for the insider proportion parameter z in the Shin normalization equation:
        q_i = 1 / odds_i,  S = sum(q_i) > 1
        pi_i(z) = (sqrt(z^2 + 4*(1 - z)*(q_i^2 / S)) - z) / (2*(1 - z))
        sum(pi_i(z)) = 1
    """
    if odds_a <= 1.0 or odds_b <= 1.0 or not math.isfinite(odds_a) or not math.isfinite(odds_b):
        raise ValueError(f"Odds must be > 1.0 and finite: {odds_a}, {odds_b}")

    q_a = 1.0 / odds_a
    q_b = 1.0 / odds_b
    s = q_a + q_b

    # If bookmaker margin is zero or negative (arbitrage/fair odds)
    if s <= 1.0001:
        tot = q_a + q_b
        return q_a / tot, q_b / tot

    def objective(z: float) -> float:
        denom = 2.0 * (1.0 - z)
        if denom <= 1e-12:
            return 1.0
        pi_a = (math.sqrt(z * z + 4.0 * (1.0 - z) * (q_a * q_a / s)) - z) / denom
        pi_b = (math.sqrt(z * z + 4.0 * (1.0 - z) * (q_b * q_b / s)) - z) / denom
        return (pi_a + pi_b) - 1.0

    # Solve for insider parameter z in [0, 0.999]
    if brentq is not None:
        try:
            z_star = brentq(objective, 0.0, 0.9999, xtol=1e-10)
        except Exception:
            z_star = 0.0
    else:
        # Bisection fallback
        lo, hi = 0.0, 0.9999
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if objective(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        z_star = 0.5 * (lo + hi)

    denom = 2.0 * (1.0 - z_star)
    if denom <= 1e-12:
        tot = q_a + q_b
        return q_a / tot, q_b / tot

    pi_a = (math.sqrt(z_star * z_star + 4.0 * (1.0 - z_star) * (q_a * q_a / s)) - z_star) / denom
    pi_b = (math.sqrt(z_star * z_star + 4.0 * (1.0 - z_star) * (q_b * q_b / s)) - z_star) / denom

    tot_pi = pi_a + pi_b
    return float(pi_a / tot_pi), float(pi_b / tot_pi)
