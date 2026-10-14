import numpy as np
import pytest
from scipy.special import expit

from src.models.redesign_series import SeriesEstimator, fit_candidates


@pytest.fixture(scope="module")
def candidates():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(480, 3))
    x[:, 2] = rng.integers(0, 2, len(x))
    y = rng.binomial(1, expit(1.2 * x[:, 0] + 0.5 * x[:, 1] * (1 + x[:, 2])))
    x.setflags(write=False)
    y.setflags(write=False)
    return fit_candidates(
        x[:240],
        y[:240],
        x[240:320],
        y[240:320],
        x[320:400],
        y[320:400],
        x[400:],
        y[400:],
        n_odd=2,
        feature_names=["d_team_glicko_logit", "d_form_w20", "c_bo3"],
        seed=83,
        max_rounds=60,
    )


def test_fitted_candidates_preserve_raw_swap_and_neutral_evidence(candidates):
    x = np.array([[1.7, -0.3, 0.0], [-0.8, 0.4, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    swapped = x.copy()
    swapped[:, :2] *= -1
    for model in candidates.values():
        p = model.predict(x)
        assert np.all((p > 0) & (p < 1))
        np.testing.assert_allclose(p + model.predict(swapped), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p[2:], 0.5, rtol=0, atol=1e-12)
        raw = model.raw_probability(x)
        np.testing.assert_allclose(
            raw + model.raw_probability(swapped), 1.0, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(raw[2:], 0.5, rtol=0, atol=1e-12)


def test_fixed_ridge_does_not_gain_undeclared_context_interactions(candidates):
    model = candidates["fixed_ridge"]
    original = np.array([[1.4, -0.8, 0.0], [-1.4, 0.8, 0.0]])
    changed_context = original.copy()
    changed_context[:, 2] = 1.0
    np.testing.assert_array_equal(
        model.predict(original), model.predict(changed_context)
    )


def test_saved_candidate_reproduces_probabilities(candidates, tmp_path):
    x = np.array([[1.4, -0.8, 1.0], [-1.4, 0.8, 1.0]])
    for name, model in candidates.items():
        path = tmp_path / f"{name}.joblib"
        model.save(path)
        np.testing.assert_array_equal(
            SeriesEstimator.load(path).predict(x), model.predict(x)
        )
        np.testing.assert_array_equal(
            SeriesEstimator.load(path).raw_probability(x), model.raw_probability(x)
        )
        with pytest.raises(FileExistsError):
            model.save(path)


def test_candidate_rejects_invalid_features_instead_of_silent_defaults(candidates):
    for model in candidates.values():
        with pytest.raises(ValueError):
            model.predict(np.array([[np.nan, 0.0, 1.0]]))
        with pytest.raises(ValueError):
            model.predict(np.array([[0.0, 1.0]]))
        with pytest.raises(ValueError):
            model.predict(np.array([[0.0, np.inf, 1.0]]))
        with pytest.raises(ValueError):
            model.predict(np.array([0.0, 1.0, 1.0]))
        with pytest.raises(ValueError):
            model.predict(np.array([["0", "1", "1"]]))


@pytest.mark.parametrize(
    "invalid", ["nonbinary", "label_shape", "single_class", "empty_stage", "bad_schema"]
)
def test_training_rejects_malformed_partitions(invalid):
    x = np.array(
        [[1.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [0.2, -0.1, 1.0], [-0.3, 0.2, 0.0]]
    )
    y = np.array([1.0, 0.0, 1.0, 0.0])
    stop_x, stop_y, train_y = x, y, y
    names = ["d_team_glicko_logit", "d_form_w20", "c_bo3"]
    if invalid == "nonbinary":
        stop_y = np.array([1.0, 0.0, 2.0, 0.0])
    elif invalid == "label_shape":
        stop_y = y[:, None]
    elif invalid == "single_class":
        train_y = np.ones(4)
    elif invalid == "empty_stage":
        stop_x, stop_y = x[:0], y[:0]
    else:
        names = ["c_bo3", "d_form_w20", "d_team_glicko_logit"]
    with pytest.raises(ValueError):
        fit_candidates(
            x,
            train_y,
            stop_x,
            stop_y,
            x,
            y,
            x,
            y,
            n_odd=2,
            feature_names=names,
            max_rounds=1,
        )
