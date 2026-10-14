"""Consumer-visible distribution boundaries for the standalone sports model."""
import numpy as np
import pytest
from scipy.stats import nbinom

from src.models.prop_distribution import DistributionPrediction, distribution_nll, fit_distribution_model


def prediction():
    return DistributionPrediction(
        kills_mu=np.array([20., 30., 40.]), kills_alpha=np.array([.1, .1, .1]),
        duration_log_mu=np.log(np.array([25., 30., 35.])), duration_log_sigma=np.array([.2, .2, .2]),
        length_probs=np.array([[1., 0., 0.], [.4, .6, 0.], [.2, .3, .5]]),
        best_of=np.array([1, 3, 5]),
    )


def test_count_tail_includes_integer_boundary_and_zero_count():
    pred = prediction()
    expected = nbinom.sf(20, 10., 10. / 30.)
    assert pred.kills_over(20.5)[0] == pytest.approx(expected)
    assert pred.kills_over(-.5)[0] == 1.
    assert np.all(pred.kills_over(24.5) >= pred.kills_over(30.5))


def test_length_distribution_preserves_legal_support():
    pred = prediction()
    assert np.allclose(pred.maps_over(2.5), [0., .6, 1.])
    assert np.allclose(pred.maps_over(3.5), [0., 0., .8])
    assert pred.duration_over(30.)[1] == pytest.approx(.5)
    assert np.all(pred.duration_over(28.) >= pred.duration_over(34.))


def test_missing_task_label_does_not_become_zero_loss_observation():
    pred = prediction()
    losses = distribution_nll(pred, np.array([0., np.nan, 40.]), np.array([25., 30., np.nan]), np.array([1, 3, 5]))
    assert np.isfinite(losses['kills'][0])
    assert np.isnan(losses['kills'][1])
    assert np.isnan(losses['duration'][2])
    assert losses['length'][0] == 0.
    assert losses['length'][1] == pytest.approx(-np.log(.6))


@pytest.mark.parametrize(
    "kills,duration,maps",
    [
        ([.5, 30., 40.], [25., 30., 35.], [1., 3., 5.]),
        ([-1., 30., 40.], [25., 30., 35.], [1., 3., 5.]),
        ([20., 30., 40.], [0., 30., 35.], [1., 3., 5.]),
        ([20., 30., 40.], [25., 30., 35.], [2., 3., 5.]),
        ([20., 30., 40.], [25., 30., 35.], [1., 4., 5.]),
        ([np.inf, 30., 40.], [25., 30., 35.], [1., 3., 5.]),
    ],
)
def test_illegal_observed_labels_raise_instead_of_becoming_missing(kills, duration, maps):
    with pytest.raises(ValueError):
        distribution_nll(prediction(), kills, duration, maps)


def test_zero_probability_length_event_has_infinite_proper_loss():
    pred = prediction()
    pred.length_probs[1] = [1., 0., 0.]
    losses = distribution_nll(pred, [np.nan] * 3, [np.nan] * 3, [1., 3., 5.])
    assert np.isposinf(losses["length"][1])


def test_prediction_rejects_mass_outside_best_of_support():
    pred = prediction()
    with pytest.raises(ValueError, match="legal"):
        DistributionPrediction(
            pred.kills_mu, pred.kills_alpha, pred.duration_log_mu, pred.duration_log_sigma,
            np.array([[1., 0., 0.], [.4, .5, .1], [.2, .3, .5]]), pred.best_of,
        )


def test_duration_tail_is_stable_without_exponentiating_logmean():
    pred = prediction()
    pred.duration_log_mu[:] = 1000.
    assert np.array_equal(pred.duration_over([-1., 0., 1.]), [1., 1., 1.])
    assert np.array_equal(pred.duration_over(np.inf), np.zeros(3))


def test_mixture_nll_is_exact_density_mixture_not_average_parameters():
    from src.models.prop_distribution import DistributionMixture
    first, second = prediction(), prediction()
    second.kills_mu = np.array([60., 70., 80.])
    second.duration_log_mu = np.log(np.array([45., 50., 55.]))
    second.length_probs[1] = [.8, .2, 0.]
    mixture = DistributionMixture([first, second], weights=np.array([1., 3.]))
    labels = ([20., np.nan, 40.], [25., 30., np.nan], [1., 3., 5.])
    left, right = distribution_nll(first, *labels), distribution_nll(second, *labels)
    actual = distribution_nll(mixture, *labels)
    for task in ("kills", "duration", "length"):
        observed = ~np.isnan(left[task])
        expected = np.full(len(left[task]), np.nan)
        expected[observed] = -np.logaddexp(np.log(.25) - left[task][observed], np.log(.75) - right[task][observed])
        assert np.allclose(actual[task], expected, equal_nan=True)
    for method in ("kills_over", "duration_over", "maps_over"):
        assert np.allclose(getattr(mixture, method)(2.5),
                           .25 * getattr(first, method)(2.5) + .75 * getattr(second, method)(2.5))


def _fit_pooled():
    X = np.array([[np.nan, 1.], [1., np.nan], [2., 3.], [3., 4.], [4., 5.], [5., 6.]])
    bo = np.array([1, 3, 3, 5, 5, 5])
    kills = np.array([0., 20., np.nan, 30., 40., 50.])
    duration = np.array([20., 25., 30., np.nan, 35., 40.])
    maps = np.array([1., 2., 3., 3., 4., 5.])
    return fit_distribution_model(
        "pooled", X, bo, kills, duration, maps,
        X_cal=np.full_like(X, np.nan), best_of_cal=bo,
        kills_cal=np.array([0., 40., 50., 60., 70., 80.]),
        duration_cal=np.array([25., 30., 35., 40., 45., 50.]), maps_cal=maps,
    )


def test_pooled_means_are_train_only_and_cold_rows_are_valid():
    model = _fit_pooled()
    pred = model.predict(np.full((3, 2), np.nan), [1, 3, 5])
    assert np.allclose(pred.kills_mu, np.mean([0., 20., 30., 40., 50.]))
    assert np.allclose(pred.duration_log_mu, np.mean(np.log([20., 25., 30., 35., 40.])))
    assert np.all(pred.kills_alpha > 0)
    assert np.all(pred.duration_log_sigma > 0)
    assert model.fit_receipt["calibration"]["partition"] == "CAL"
    assert model.fit_receipt["calibration"]["means_frozen"] is True
    assert model.fit_receipt["task_support"]["kills"]["train"] == 5


def test_native_bundle_joblib_roundtrip_preserves_predictions(tmp_path):
    import joblib
    model = _fit_pooled()
    destination = tmp_path / "distribution.joblib"
    joblib.dump(model, destination)
    restored = joblib.load(destination)
    X = np.full((3, 2), np.nan)
    before, after = model.predict(X, [1, 3, 5]), restored.predict(X, [1, 3, 5])
    for field in ("kills_mu", "kills_alpha", "duration_log_mu", "duration_log_sigma", "length_probs", "best_of"):
        assert np.array_equal(getattr(before, field), getattr(after, field))
    assert restored.fit_receipt == model.fit_receipt
