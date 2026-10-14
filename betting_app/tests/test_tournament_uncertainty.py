"""Temporal, posterior and marginal contracts for sports-only residual strength."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tournament_uncertainty import fit_team_strength_posterior


def _evidence():
    return pd.DataFrame({
        "golgg_match_id": range(24), "date": pd.date_range("2024-01-01", periods=24),
        "team1_id": ["A"] * 24, "team2_id": ["B"] * 24,
        "y_true": [1] * 18 + [0] * 6, "sports": [0.5] * 24,
    })


def test_posterior_learns_shrunk_strength_and_preserves_opponent_covariance():
    evidence = _evidence()
    weak = fit_team_strength_posterior(evidence, "sports", cutoff="2024-02-01", prior_sd=1.0)
    strong = fit_team_strength_posterior(evidence, "sports", cutoff="2024-02-01", prior_sd=0.1)
    assert 0 < weak.mean[0] - weak.mean[1] < np.log(3)
    assert abs(strong.mean[0]) < abs(weak.mean[0])
    assert weak.covariance[0, 1] > 0
    assert np.linalg.eigvalsh(weak.covariance).min() > 0
    assert weak.provenance["fit_rows"] == 24


def test_cutoff_excludes_same_day_and_future_outcomes():
    evidence = _evidence()
    cutoff = "2024-01-15"
    before = fit_team_strength_posterior(evidence, "sports", cutoff=cutoff, prior_sd=0.5)
    evidence.loc[evidence.date >= cutoff, "y_true"] = 1 - evidence.loc[evidence.date >= cutoff, "y_true"]
    after = fit_team_strength_posterior(evidence, "sports", cutoff=cutoff, prior_sd=0.5)
    np.testing.assert_array_equal(before.mean, after.mean)
    np.testing.assert_array_equal(before.covariance, after.covariance)
    assert before.provenance["fit_rows"] == 14


def test_marginals_are_symmetric_and_variance_is_separate_from_mean_correction():
    posterior = fit_team_strength_posterior(_evidence(), "sports", cutoff="2024-02-01", prior_sd=1.0)
    forward = posterior.predict(["A"], ["B"], [0.4])
    reverse = posterior.predict(["B"], ["A"], [0.6])
    np.testing.assert_allclose(forward + reverse, 1, atol=1e-14)
    mean_only = posterior.predict(["A"], ["B"], [0.4], integrate=False)
    assert forward[0] < mean_only[0]
    assert posterior.predict(["A", "A"], ["B", "B"], [0, 1]).tolist() == [0, 1]
    assert posterior.predict(["new1"], ["new2"], [0.5])[0] == pytest.approx(0.5)


def test_sampling_is_repeatable_shared_and_does_not_mutate_global_rng():
    posterior = fit_team_strength_posterior(_evidence(), "sports", cutoff="2024-02-01", prior_sd=1.0)
    before = np.random.get_state()
    draws = posterior.sample(["A", "B", "new"], 4000, 82)
    again = posterior.sample(["A", "B", "new"], 4000, 82)
    after = np.random.get_state()
    for team in draws:
        np.testing.assert_array_equal(draws[team], again[team])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert np.cov(draws["A"], draws["B"])[0, 1] > 0
    means = posterior.sample(["A", "B"], 9, 82, uncertainty_scale=0)
    np.testing.assert_array_equal(means["A"], np.full(9, posterior.mean[0]))


@pytest.mark.parametrize("fault", ["duplicate", "self", "nan", "endpoint", "nonbinary", "missing_team", "empty", "prior"])
def test_unsafe_fit_evidence_is_rejected(fault):
    frame = _evidence()
    prior = 0.5
    if fault == "duplicate": frame.loc[1, "golgg_match_id"] = 0
    elif fault == "self": frame.loc[0, "team2_id"] = "A"
    elif fault == "nan": frame.loc[0, "sports"] = np.nan
    elif fault == "endpoint": frame.loc[0, "sports"] = 0
    elif fault == "nonbinary":
        frame["y_true"] = frame.y_true.astype(float)
        frame.loc[0, "y_true"] = 0.5
    elif fault == "missing_team": frame.loc[0, "team1_id"] = None
    elif fault == "empty": frame = frame.iloc[:0]
    elif fault == "prior": prior = 0
    with pytest.raises(ValueError):
        fit_team_strength_posterior(frame, "sports", cutoff="2024-02-01", prior_sd=prior)


@pytest.mark.parametrize("kwargs", [{"simulations": 0}, {"simulations": True}, {"seed": -1}, {"uncertainty_scale": -1}, {"uncertainty_scale": np.nan}])
def test_sampling_rejects_invalid_controls(kwargs):
    posterior = fit_team_strength_posterior(_evidence(), "sports", cutoff="2024-02-01", prior_sd=1.0)
    arguments = {"simulations": 10, "seed": 82, **kwargs}
    with pytest.raises(ValueError):
        posterior.sample(["A", "B"], **arguments)


def test_prediction_rejects_misaligned_or_self_paired_inputs():
    posterior = fit_team_strength_posterior(_evidence(), "sports", cutoff="2024-02-01", prior_sd=1.0)
    for first, second, probability in [(["A"], ["B", "C"], [0.5]), (["A"], ["A"], [0.5]), (["A"], ["B"], [np.nan])]:
        with pytest.raises(ValueError):
            posterior.predict(first, second, probability)
