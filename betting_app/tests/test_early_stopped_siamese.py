"""Observable contracts for the experimental STOP-selected Siamese ensemble."""

import joblib
import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from scripts.early_stopped_siamese import fit_early_stopped_ensemble
from scripts.train_and_tune_siamese_series import NumPySiameseMLP, fit_platt_scaling


def synthetic_split():
    rng = np.random.RandomState(71)
    x = rng.normal(size=(48, 3)) + np.array([2.0, -1.0, 0.5])
    y = (x[:, 0] - 2.0 * x[:, 1] > 3.0).astype(int)
    return x[:32], y[:32], x[32:], y[32:]


def test_raw_predictions_are_swap_antisymmetric_reproducible_and_picklable(tmp_path):
    data = synthetic_split()
    scorer, _ = fit_early_stopped_ensemble(
        *data, gamma=1.5, seed=19, max_epochs=2, batch_size=16
    )
    repeated, _ = fit_early_stopped_ensemble(
        *data, gamma=1.5, seed=19, max_epochs=2, batch_size=16
    )
    probes = np.vstack([data[2], np.zeros((1, 3))])
    raw = scorer.logit(probes)
    np.testing.assert_array_equal(scorer.logit(-probes), -raw)
    assert raw[-1] == 0.0
    np.testing.assert_array_equal(repeated.logit(probes), raw)
    artifact = tmp_path / "ensemble.joblib"
    joblib.dump(scorer, artifact)
    np.testing.assert_array_equal(joblib.load(artifact).logit(probes), raw)


def test_patience_restores_every_member_from_best_calibrated_ensemble_epoch(
    monkeypatch,
):
    # A controlled optimizer trajectory uses the real network forward pass:
    # epoch one learns the correct feature; epoch two switches to its negation.
    # Thus an inferior *last* ensemble cannot accidentally pass this regression.
    def step_towards_then_away(self, grads, lr):
        strength = 1.0 + abs(float(self.W1[0, 0]))
        self.t += 1
        for parameter in self.params:
            parameter.fill(0.0)
        self.W1[0 if self.t == 1 else 1, 0] = strength
        self.W2[0, 0] = 1.0
        self.W3[0, 0] = 1.0

    monkeypatch.setattr(NumPySiameseMLP, "step_adam", step_towards_then_away)
    x = np.array([[-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0], [2.0, -2.0]])
    y = np.array([0, 0, 1, 1])
    kwargs = {"gamma": 0.0, "seed": 31, "members": 5, "patience": 1, "batch_size": 4}
    first_epoch, _ = fit_early_stopped_ensemble(x, y, x, y, max_epochs=1, **kwargs)
    restored, audit = fit_early_stopped_ensemble(x, y, x, y, max_epochs=5, **kwargs)
    assert audit["best_epoch"] == 1
    assert audit["epochs_ran"] == 2
    assert audit["stopped_early"]
    raw = restored.logit(x)
    np.testing.assert_array_equal(raw, first_epoch.logit(x))
    eps = np.finfo(float).eps
    forecast_logit = logit(np.clip(expit(raw), eps, 1.0 - eps))
    slope = fit_platt_scaling(forecast_logit, y)
    expected_loss = log_loss(y, expit(slope * forecast_logit), labels=[0, 1])
    assert audit["best_stop_log_loss"] == pytest.approx(expected_loss)
    assert audit["history"][1]["stop_log_loss"] > expected_loss
    # The selected STOP slope must not leak into inference returned for final CAL.
    assert not np.allclose(raw, slope * forecast_logit)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("x_train", np.ones(32)),
        ("x_train", np.full((32, 3), np.nan)),
        ("x_stop", np.empty((0, 3))),
        ("x_stop", np.ones((16, 4))),
        ("x_stop", np.full((16, 3), np.inf)),
        ("y_train", np.full(32, 0.5)),
        ("y_stop", np.ones(15)),
        ("y_stop", np.ones((8, 2))),
        ("y_stop", np.full(16, np.nan)),
    ],
)
def test_invalid_training_or_stop_data_is_rejected(field, replacement):
    inputs = dict(zip(("x_train", "y_train", "x_stop", "y_stop"), synthetic_split()))
    inputs[field] = replacement
    with pytest.raises(ValueError):
        fit_early_stopped_ensemble(**inputs, gamma=0.0, seed=7, max_epochs=1)


@pytest.mark.parametrize(
    "option,value",
    [
        ("gamma", -1.0),
        ("lr", np.nan),
        ("weight_decay", -0.1),
        ("members", 0),
        ("max_epochs", 1.5),
        ("patience", 0),
        ("batch_size", True),
        ("seed", -1),
    ],
)
def test_invalid_optimization_bounds_are_rejected(option, value):
    options = {"gamma": 0.0, "seed": 7, "max_epochs": 1}
    options[option] = value
    with pytest.raises((TypeError, ValueError)):
        fit_early_stopped_ensemble(*synthetic_split(), **options)


def test_inference_rejects_wrong_dimensions_and_nonfinite_values():
    scorer, _ = fit_early_stopped_ensemble(
        *synthetic_split(), gamma=0.0, seed=7, members=1, max_epochs=1
    )
    for invalid in (np.ones((2, 4)), np.full((2, 3), np.inf), np.ones(3)):
        with pytest.raises(ValueError):
            scorer.logit(invalid)
