"""Proper scoring rules must describe the full predicted distribution."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm, nbinom

from src.analysis.prop_distribution_evaluation import lognormal_mixture_crps, negative_binomial_mixture_crps


def test_lognormal_mixture_crps_matches_independent_cdf_integral():
    mus = np.log(np.array([[20., 35.]])); sigmas = np.array([[.2, .35]])
    y = 30.
    weights = np.array([.4, .6])
    cdf = lambda t: sum(w * lognorm.cdf(t, s, scale=np.exp(m)) for w, m, s in zip(weights, mus[0], sigmas[0]))
    oracle = quad(lambda t: cdf(t) ** 2, 0., y, epsabs=1e-8)[0] + quad(lambda t: (1-cdf(t)) ** 2, y, np.inf, epsabs=1e-8)[0]
    got = lognormal_mixture_crps(np.array([y]), mus, sigmas, weights)
    assert got[0] == pytest.approx(oracle, abs=1e-7)


def test_negative_binomial_crps_matches_independent_absolute_distance_expectation():
    mu = np.array([[4., 8.]]); alpha = np.array([[.2, .1]]); weights = np.array([.3, .7])
    k = np.arange(200)
    pmf = sum(w * nbinom.pmf(k, 1/a, 1/(1+a*m)) for w, m, a in zip(weights, mu[0], alpha[0]))
    oracle = np.sum(pmf * np.abs(k - 6)) - .5 * np.sum(pmf[:, None] * pmf[None, :] * np.abs(k[:, None]-k[None, :]))
    got, bound = negative_binomial_mixture_crps(np.array([6.]), mu, alpha, weights)
    assert got[0] == pytest.approx(oracle, abs=1e-7)
    assert 0. <= bound < 1e-6
