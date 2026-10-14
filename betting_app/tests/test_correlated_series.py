import numpy as np
import pytest

from src.models.correlated_series import (
    fit_correlation,
    fit_map_slope,
    series_probability,
    terminal_score_probability,
)


def test_stopped_scores_form_distribution_and_match_winner_probability():
    for bo in (1, 3, 5):
        k = bo // 2 + 1
        for rho in (0.0, 0.2, 0.8):
            p = np.array([0.2, 0.5, 0.8])
            win = sum(
                terminal_score_probability(p, bo, k, loss, rho) for loss in range(k)
            )
            loss = sum(
                terminal_score_probability(p, bo, wins, k, rho) for wins in range(k)
            )
            np.testing.assert_allclose(win + loss, 1.0, atol=1e-12)
            np.testing.assert_allclose(win, series_probability(p, bo, rho), atol=1e-12)
            np.testing.assert_allclose(
                win + series_probability(1 - p, bo, rho), 1.0, atol=1e-12
            )


def test_iid_and_strong_correlation_limits_preserve_mean():
    p = np.array([0.2, 0.5, 0.8])
    np.testing.assert_allclose(series_probability(p, 3, 0), 3 * p * p - 2 * p * p * p)
    np.testing.assert_allclose(
        series_probability(p, 5, 0), 10 * p**3 - 15 * p**4 + 6 * p**5
    )
    np.testing.assert_allclose(series_probability(p, 1, 0.6), p)
    np.testing.assert_allclose(series_probability(p, 5, 0.999999), p, atol=1e-6)


def test_invalid_formats_scores_and_correlation_are_errors():
    with pytest.raises(ValueError):
        series_probability(0.6, 2, 0.2)
    with pytest.raises(ValueError):
        series_probability(0.6, 3, 1.0)
    with pytest.raises(ValueError):
        terminal_score_probability(0.6, 3, 2, 2, 0.2)
    with pytest.raises(ValueError):
        series_probability(float("nan"), 3, 0.2)


def test_broadcasting_endpoints_and_tiny_rho_keep_degenerate_distributions():
    p = np.array([[0.0], [0.3], [1.0]])
    bo = np.array([1, 3, 5])
    iid = series_probability(p, bo)
    np.testing.assert_allclose(series_probability(p, bo, 1e-300), iid, atol=1e-14)
    np.testing.assert_allclose(iid[0], 0.0)
    np.testing.assert_allclose(iid[-1], 1.0)
    np.testing.assert_allclose(
        series_probability(p, bo, np.nextafter(1.0, 0.0)), p + np.zeros(3)
    )
    assert terminal_score_probability(0.0, 5, 3, 0, 0.8) == 0.0
    assert terminal_score_probability(1.0, 5, 3, 0, 0.8) == 1.0
    assert isinstance(series_probability(0.4, 3), float)


@pytest.mark.parametrize(
    "p,bo,a,b,rho",
    [
        (-0.1, 3, 2, 0, 0.0),
        (1.1, 3, 2, 0, 0.0),
        (0.5, 3.5, 2, 0, 0.0),
        (0.5, 3, 2, 0.5, 0.0),
        (0.5, 5, 2, 1, 0.0),
        (0.5, 3, 2, -1, 0.0),
        (0.5, 3, 2, 0, -0.1),
        (0.5, 3, 2, 0, np.inf),
    ],
)
def test_invalid_probability_or_nonterminal_score_is_not_clipped(p, bo, a, b, rho):
    with pytest.raises(ValueError):
        terminal_score_probability(p, bo, a, b, rho)


def test_map_slope_uses_all_stopped_map_counts_not_equal_series_winners():
    # Five A maps and two B maps: the IID MLE odds are 5:2, not 3:1.
    slope = fit_map_slope(np.ones(4), [1, 1, 3, 3], [1, 1, 2, 1], [0, 0, 0, 2])
    np.testing.assert_allclose(slope, np.log(2.5), rtol=1e-10)
    mirrored = fit_map_slope(-np.ones(4), [1, 1, 3, 3], [0, 0, 0, 2], [1, 1, 2, 1])
    np.testing.assert_allclose(mirrored, slope, rtol=1e-10)


def test_fitted_iid_score_distribution_is_a_nested_negative_control():
    # Exact IID Bo5 score frequencies at p=.5, without Monte Carlo noise.
    a = np.repeat([3, 3, 3, 0, 1, 2], [2, 3, 3, 2, 3, 3])
    b = np.repeat([0, 1, 2, 3, 3, 3], [2, 3, 3, 2, 3, 3])
    assert fit_correlation(0.5, 5, a, b) < 1e-6
    # A greater-than-IID proportion of split scores puts the MLE at exact zero.
    assert fit_correlation(0.5, 3, [2, 1], [1, 2]) == 0.0


def test_correlation_recovers_score_dependence_even_with_balanced_winners():
    # At p=.5 in Bo3, sweep probability is (1+rho)/2: 14/20 implies rho=.4.
    a = np.repeat([2, 0, 2, 1], [7, 7, 3, 3])
    b = np.repeat([0, 2, 1, 2], [7, 7, 3, 3])
    rho = fit_correlation(0.5, 3, a, b)
    np.testing.assert_allclose(rho, 0.4, atol=1e-7)
    np.testing.assert_allclose(series_probability(0.5, 3, rho), 0.5)


def test_unidentified_and_boundary_fits_raise_instead_of_defaulting():
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_correlation(0.5, 1, [1, 0], [0, 1])
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_correlation([0.0, 1.0], 3, [0, 2], [2, 0])
    with pytest.raises(ValueError, match="zero probability"):
        fit_correlation(0.0, 3, 2, 0)
    with pytest.raises(RuntimeError, match="boundary"):
        fit_correlation(0.5, 3, [2, 0], [0, 2])
    with pytest.raises(ValueError, match="informative"):
        fit_map_slope([0.0, 0.0], 1, [1, 0], [0, 1])
    with pytest.raises(RuntimeError, match="positive"):
        fit_map_slope([1.0, 1.0], 1, [0, 0], [1, 1])
    with pytest.raises(RuntimeError, match="separation"):
        fit_map_slope([1.0, -1.0], 1, [1, 0], [0, 1])
