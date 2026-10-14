"""Exchangeable, positively correlated map outcomes for declared Bo1/3/5.

A series shares one latent map propensity drawn from Beta(p*kappa,
(1-p)*kappa), with rho=1/(kappa+1). This is residual shared uncertainty,
not causal momentum, a draft policy, or a confidence interval. Endpoint map
probabilities are the corresponding degenerate distributions. rho=0 is IID.

Fit assumptions: fit_map_slope uses earlier, completed series and fixed,
strictly pre-series team logits. Its IID stopped-score likelihood uses every
observed map count: longer series contain variable amounts of map information,
not equal-series winner weighting. It neither observes nor invents map order.
Freeze that slope before fit_correlation on a later, disjoint development block;
the latter fits one global rho to Bo3/5 terminal scores. Callers own temporal
partitioning. Neither fit is post-calibration inference about uncertainty.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, gammaln, logsumexp


def _finite(value: ArrayLike, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _format(best_of: ArrayLike) -> np.ndarray:
    result = _finite(best_of, "best_of")
    if not np.all(np.isin(result, (1, 3, 5))):
        raise ValueError("best_of must be declared as 1, 3, or 5")
    return result.astype(np.int64)


def _probability_inputs(p: ArrayLike, best_of: ArrayLike, rho: ArrayLike):
    p = _finite(p, "p")
    rho = _finite(rho, "rho")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p must be in [0, 1]")
    if np.any((rho < 0) | (rho >= 1)):
        raise ValueError("rho must be in [0, 1)")
    return np.broadcast_arrays(p, _format(best_of), rho)


def _scores(best_of: np.ndarray, wins_a: ArrayLike, wins_b: ArrayLike):
    best_of, a, b = np.broadcast_arrays(
        best_of, _finite(wins_a, "wins_a"), _finite(wins_b, "wins_b")
    )
    k = best_of // 2 + 1
    legal = ((a == k) & (b >= 0) & (b < k)) | ((b == k) & (a >= 0) & (a < k))
    if not np.all(legal & (a == np.floor(a)) & (b == np.floor(b))):
        raise ValueError("scores must be legal completed terminal scores for best_of")
    return best_of, a.astype(np.int64), b.astype(np.int64)


def _log_terminal(p, a, b, rho):
    """Log stopped mass using at most five rising-factor terms.

    This is betaln(alpha+a,beta+b)-betaln(alpha,beta), evaluated without
    subtracting huge near-equal beta logs as rho approaches zero. The scaled
    rising factors also avoid overflowing kappa for tiny positive rho.
    """
    result = gammaln(a + b) - gammaln(np.minimum(a, b) + 1) - gammaln(np.maximum(a, b))
    with np.errstate(divide="ignore"):
        log_rho = np.log(rho)
        log_scale = np.log1p(-rho)
        log_p = np.log(p)
        log_q = np.log1p(-p)
        for j in range(3):
            # The first success factor is exactly p, including degenerate p.
            if j == 0:
                success = log_p
            else:
                success = np.logaddexp(log_p + log_scale, np.log(j) + log_rho)
                success -= np.logaddexp(log_scale, np.log(j) + log_rho)
            failure = np.logaddexp(log_q + log_scale, np.log(float(j)) + log_rho)
            failure -= np.logaddexp(log_scale, np.log(a + j) + log_rho)
            result = result + np.where(j < a, success, 0.0)
            result = result + np.where(j < b, failure, 0.0)
    return result


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def terminal_score_probability(
    p: ArrayLike,
    best_of: ArrayLike,
    wins_a: ArrayLike,
    wins_b: ArrayLike,
    rho: ArrayLike = 0,
) -> float | np.ndarray:
    """Probability of a legal stopped score, with NumPy broadcasting.

    For k:l the path multiplicity is C(k+l-1,l), not C(k+l,k).
    All inputs are validated without clipping; impossible scores under p=0/1
    have probability zero. Scalar inputs produce a float.
    """
    p, best_of, rho = _probability_inputs(p, best_of, rho)
    best_of, a, b = _scores(best_of, wins_a, wins_b)
    p, a, b, rho = np.broadcast_arrays(p, a, b, rho)
    return _scalar_or_array(np.exp(_log_terminal(p, a, b, rho)))


def series_probability(
    p: ArrayLike, best_of: ArrayLike, rho: ArrayLike = 0
) -> float | np.ndarray:
    """P(A wins), summing legal terminal paths; declared format is required."""
    p, best_of, rho = _probability_inputs(p, best_of, rho)
    k = best_of // 2 + 1
    masses = []
    for losses in range(3):
        # Invalid lengths are excluded, not interpreted as played-map formats.
        b = np.full_like(k, losses)
        masses.append(np.where(losses < k, _log_terminal(p, k, b, rho), -np.inf))
    result = np.exp(logsumexp(np.stack(masses), axis=0))
    return _scalar_or_array(result)


def fit_map_slope(
    z: ArrayLike, best_of: ArrayLike, wins_a: ArrayLike, wins_b: ArrayLike
) -> float:
    """Fit a finite positive slope for sigmoid(slope*z) by IID score MLE.

    Stopped-score multiplicity is constant with respect to slope. Thus the
    derivative uses wins_a and wins_b as map counts, retaining the variable
    map information of each series. No intercept or first-map order is fitted.
    Empty/uninformative data raise ValueError; boundary or non-finite MLEs and
    numerical optimizer failures raise RuntimeError rather than a default slope.
    """
    z = _finite(z, "z")
    _, a, b = _scores(_format(best_of), wins_a, wins_b)
    z, a, b = np.broadcast_arrays(z, a, b)
    if z.size == 0 or not np.any(z != 0):
        raise ValueError("map slope needs nonempty, informative team logits")
    # Optimize in scaled coordinates to keep very large finite logits safe.
    scale = float(np.max(np.abs(z)))
    x = z / scale

    def derivative(t):
        return float(np.mean(x * (b * expit(t * x) - a * expit(-t * x))))

    if derivative(0.0) >= 0:
        raise RuntimeError("map slope MLE is not strictly positive")
    opposing = np.where(x > 0, x * b, -x * a)
    if not np.any(opposing > 0):
        raise RuntimeError("map slope has no finite MLE (complete separation)")
    upper = 1.0
    while derivative(upper) <= 0:
        upper *= 2
        if not np.isfinite(upper):
            raise RuntimeError("map slope optimizer could not bracket a finite MLE")
    root, result = brentq(
        derivative,
        0.0,
        upper,
        xtol=np.finfo(float).tiny,
        rtol=8 * np.finfo(float).eps,
        maxiter=1000,
        full_output=True,
        disp=False,
    )
    slope = root / scale
    if not result.converged or not np.isfinite(slope) or slope <= 0:
        raise RuntimeError("map slope optimizer failed to obtain a finite positive MLE")
    return float(slope)


def fit_correlation(
    p: ArrayLike, best_of: ArrayLike, wins_a: ArrayLike, wins_b: ArrayLike
) -> float:
    """Fit one global rho on later Bo3/5 stopped-score likelihood.

    p must come from the frozen earlier slope, not winner calibration. Bo1 and
    deterministic propensities carry no rho information. A bounded scalar fit
    competes with the exact nested rho=0 likelihood (no hyperparameter grid).
    No informative longer series raises ValueError. All-sweep data have only
    an unattained rho=1 optimum and raise RuntimeError; optimizer failure is
    likewise explicit. Returned rho is dependence, never a confidence bound.
    """
    p, best_of, _ = _probability_inputs(p, best_of, 0)
    best_of, a, b = _scores(best_of, wins_a, wins_b)
    p, best_of, a, b = np.broadcast_arrays(p, best_of, a, b)
    if np.any(((p == 0) & (a > 0)) | ((p == 1) & (b > 0))):
        raise ValueError("observed score has zero probability for deterministic p")
    informative = (best_of > 1) & (p > 0) & (p < 1)
    if not np.any(informative):
        raise ValueError("rho is unidentifiable without informative Bo3/Bo5 scores")
    p, a, b = p[informative], a[informative], b[informative]
    if np.all(np.minimum(a, b) == 0):
        raise RuntimeError("rho MLE approaches excluded boundary 1 for all-sweep data")

    def objective(rho):
        return -float(np.mean(_log_terminal(p, a, b, rho)))

    iid_loss = objective(0.0)
    result = minimize_scalar(
        objective,
        bounds=(0.0, np.nextafter(1.0, 0.0)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"rho optimizer failed: {result.message}")
    return 0.0 if iid_loss <= result.fun else float(result.x)
