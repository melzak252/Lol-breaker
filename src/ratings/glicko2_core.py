"""Small, dependency-free implementation of the Glicko-2 update equations.

Ratings and rating deviations exposed by this module use the traditional
Glicko scale.  Volatility and all intermediate calculations use the Glicko-2
scale described in Mark Glickman's specification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

_SCALE = 173.7178
_DEFAULT_RATING = 1500.0


def _as_finite_float(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Glicko2State:
    """A player's rating, rating deviation, and volatility."""

    rating: float = _DEFAULT_RATING
    rd: float = 350.0
    volatility: float = 0.06

    def __post_init__(self) -> None:
        rating = _as_finite_float("rating", self.rating)
        rd = _as_finite_float("rd", self.rd)
        volatility = _as_finite_float("volatility", self.volatility)
        if rd < 0.0:
            raise ValueError("rd must be non-negative")
        if volatility <= 0.0:
            raise ValueError("volatility must be positive")
        object.__setattr__(self, "rating", rating)
        object.__setattr__(self, "rd", rd)
        object.__setattr__(self, "volatility", volatility)


@dataclass(frozen=True, slots=True)
class Glicko2Observation:
    """One result against an opponent, optionally power-weighted."""

    opponent_rating: float
    opponent_rd: float
    score: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        opponent_rating = _as_finite_float("opponent_rating", self.opponent_rating)
        opponent_rd = _as_finite_float("opponent_rd", self.opponent_rd)
        score = _as_finite_float("score", self.score)
        weight = _as_finite_float("weight", self.weight)
        if opponent_rd < 0.0:
            raise ValueError("opponent_rd must be non-negative")
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be between zero and one")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        object.__setattr__(self, "opponent_rating", opponent_rating)
        object.__setattr__(self, "opponent_rd", opponent_rd)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "weight", weight)


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _logistic(value: float) -> float:
    """Overflow-safe logistic function, including for infinite intermediates."""

    if value >= 0.0:
        tail = math.exp(-value)
        return 1.0 / (1.0 + tail)
    head = math.exp(value)
    return head / (1.0 + head)


def expected_score(
    rating: float,
    rd: float,
    opponent_rating: float,
    opponent_rd: float,
) -> float:
    """Return the symmetric win probability implied by two uncertain ratings.

    Both deviations enter the attenuation term.  Consequently changing sides
    produces the exact complementary probability, including when the players'
    deviations differ.
    """

    rating_value = _as_finite_float("rating", rating)
    rd_value = _as_finite_float("rd", rd)
    opponent_rating_value = _as_finite_float("opponent_rating", opponent_rating)
    opponent_rd_value = _as_finite_float("opponent_rd", opponent_rd)
    if rd_value < 0.0:
        raise ValueError("rd must be non-negative")
    if opponent_rd_value < 0.0:
        raise ValueError("opponent_rd must be non-negative")

    # This is algebraically ``g(hypot(rd, opponent_rd) / scale)`` times
    # the scaled rating difference.  Normalizing first prevents otherwise
    # finite public inputs near the binary64 limit from producing inf * 0.
    normalization = max(
        abs(rating_value),
        abs(opponent_rating_value),
        rd_value,
        opponent_rd_value,
        _SCALE,
    )
    normalized_denominator = math.sqrt(
        (_SCALE / normalization) ** 2
        + 3.0
        * (
            (rd_value / normalization) ** 2
            + (opponent_rd_value / normalization) ** 2
        )
        / (math.pi * math.pi)
    )
    log_odds = (
        rating_value / normalization - opponent_rating_value / normalization
    ) / normalized_denominator
    return _logistic(log_odds)

def series_win_probability(
    map_probability: float,
    best_of: int = 1,
) -> float:
    """Project a single-map win probability onto a series win probability.

    Uses exact independent map combinatorics:
        P_Bo1(p) = p
        P_Bo3(p) = p^2 * (3 - 2p)
        P_Bo5(p) = p^3 * (10 - 15p + 6p^2)

    For arbitrary odd N = 2k - 1, computes the binomial upper tail sum.
    Exact anti-symmetry P(p, N) + P(1-p, N) == 1.0 is preserved.
    """

    map_prob = _as_finite_float("map_probability", map_probability)
    if not (0.0 <= map_prob <= 1.0):
        raise ValueError("map_probability must be in [0, 1]")
    if (
        isinstance(best_of, bool)
        or not isinstance(best_of, int)
        or best_of <= 0
        or best_of % 2 == 0
    ):
        raise ValueError("best_of must be an odd positive integer")
    if best_of == 1:
        return map_prob
    if best_of == 3:
        return map_prob * map_prob * (3.0 - 2.0 * map_prob)
    if best_of == 5:
        p2 = map_prob * map_prob
        return map_prob * p2 * (10.0 - 15.0 * map_prob + 6.0 * p2)

    n = best_of
    k = (n + 1) // 2
    total = 0.0
    for i in range(k, n + 1):
        total += math.comb(n, i) * (map_prob**i) * ((1.0 - map_prob) ** (n - i))
    return total



def inflate(
    state: Glicko2State,
    periods: int,
    max_rd: float = 350.0,
) -> Glicko2State:
    """Increase rating deviation for a number of inactive rating periods."""

    if not isinstance(state, Glicko2State):
        raise TypeError("state must be a Glicko2State")
    if isinstance(periods, bool) or not isinstance(periods, int):
        raise TypeError("periods must be an integer")
    if periods < 0:
        raise ValueError("periods must be non-negative")
    maximum = _as_finite_float("max_rd", max_rd)
    if maximum < 0.0:
        raise ValueError("max_rd must be non-negative")

    inflated_phi = math.hypot(state.rd / _SCALE, state.volatility * math.sqrt(periods))
    inflated_rd = min(maximum, inflated_phi * _SCALE)
    if inflated_rd == state.rd:
        return state
    return Glicko2State(state.rating, inflated_rd, state.volatility)


def _volatility_objective(
    x: float,
    *,
    delta_squared: float,
    phi_squared: float,
    variance: float,
    a: float,
    tau_squared: float,
) -> float:
    exponential = math.exp(x)
    denominator = phi_squared + variance + exponential
    likelihood = (
        exponential * (delta_squared - phi_squared - variance - exponential)
        / (2.0 * denominator * denominator)
    )
    return likelihood - (x - a) / tau_squared


def _new_volatility(
    phi: float,
    volatility: float,
    variance: float,
    delta: float,
    tau: float,
    tolerance: float,
) -> float:
    """Solve step 5 of the official Glicko-2 algorithm."""

    phi_squared = phi * phi
    delta_squared = delta * delta
    a = 2.0 * math.log(volatility)
    tau_squared = tau * tau

    # For a tau this small the regularization pins the root to ``a`` more
    # closely than a binary64 value can represent.
    if tau_squared == 0.0 or a + tau_squared == a:
        return volatility

    def objective(x: float) -> float:
        return _volatility_objective(
            x,
            delta_squared=delta_squared,
            phi_squared=phi_squared,
            variance=variance,
            a=a,
            tau_squared=tau_squared,
        )

    lower = a
    if delta_squared > phi_squared + variance:
        upper = math.log(delta_squared - phi_squared - variance)
    else:
        k = 1
        upper = a - tau
        while objective(upper) < 0.0:
            k += 1
            upper = a - k * tau

    f_lower = objective(lower)
    f_upper = objective(upper)
    while abs(upper - lower) > tolerance:
        denominator = f_upper - f_lower
        if denominator == 0.0:
            break
        candidate = lower + (lower - upper) * f_lower / denominator
        if candidate == lower or candidate == upper:
            candidate = (lower + upper) / 2.0
            if candidate == lower or candidate == upper:
                break
        f_candidate = objective(candidate)
        if f_candidate * f_upper <= 0.0:
            lower = upper
            f_lower = f_upper
        else:
            f_lower *= 0.5
        upper = candidate
        f_upper = f_candidate

    return math.exp(lower / 2.0)


def update(
    state: Glicko2State,
    observations: Sequence[Glicko2Observation],
    tau: float = 0.5,
    convergence_tolerance: float = 1e-6,
) -> Glicko2State:
    """Apply a simultaneous, power-likelihood-weighted Glicko-2 update.

    Every observation is evaluated against the frozen input state.  A weight
    of ``n`` is therefore equivalent to observing the same result ``n``
    times.  Zero-weight observations carry no evidence and are discarded.
    """

    if not isinstance(state, Glicko2State):
        raise TypeError("state must be a Glicko2State")
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a list-like sequence")
    tau_value = _as_finite_float("tau", tau)
    tolerance = _as_finite_float("convergence_tolerance", convergence_tolerance)
    if tau_value <= 0.0:
        raise ValueError("tau must be positive")
    if tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive")

    evidence: list[Glicko2Observation] = []
    for observation in observations:
        if not isinstance(observation, Glicko2Observation):
            raise TypeError("every observation must be a Glicko2Observation")
        if observation.weight > 0.0:
            evidence.append(observation)
    if not evidence:
        return state

    # Sorting makes the floating-point reductions independent of caller order.
    evidence.sort(
        key=lambda item: (
            item.opponent_rating,
            item.opponent_rd,
            item.score,
            item.weight,
        )
    )

    mu = (state.rating - _DEFAULT_RATING) / _SCALE
    phi = state.rd / _SCALE
    information_terms: list[float] = []
    score_terms: list[float] = []
    for observation in evidence:
        opponent_mu = (observation.opponent_rating - _DEFAULT_RATING) / _SCALE
        opponent_phi = observation.opponent_rd / _SCALE
        attenuation = _g(opponent_phi)
        expectation = _logistic(attenuation * (mu - opponent_mu))
        information_terms.append(
            observation.weight
            * attenuation
            * attenuation
            * expectation
            * (1.0 - expectation)
        )
        score_terms.append(
            observation.weight
            * attenuation
            * (observation.score - expectation)
        )

    information = math.fsum(information_terms)
    score_sum = math.fsum(score_terms)
    if information <= 0.0 or not math.isfinite(information):
        raise ArithmeticError("observations do not yield finite rating information")
    variance = 1.0 / information
    delta = variance * score_sum
    new_volatility = _new_volatility(
        phi,
        state.volatility,
        variance,
        delta,
        tau_value,
        tolerance,
    )
    pre_rating_phi = math.hypot(phi, new_volatility)
    new_phi = 1.0 / math.sqrt(1.0 / (pre_rating_phi * pre_rating_phi) + 1.0 / variance)
    new_mu = mu + new_phi * new_phi * score_sum

    result = Glicko2State(
        _DEFAULT_RATING + _SCALE * new_mu,
        _SCALE * new_phi,
        new_volatility,
    )
    return result
