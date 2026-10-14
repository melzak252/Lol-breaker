import numpy as np
import pytest
from scipy.special import expit

from scripts.experiment_market_distillation import train_student
from scripts.oddsfree_distillation import fit_auxiliary_student


def test_zero_auxiliary_weight_matches_existing_trainer_on_all_rows():
    rng = np.random.default_rng(89)
    x = rng.normal(size=(90, 4))
    y = rng.binomial(1, expit(x @ np.array([0.8, -0.5, 0.3, 0.1])))
    teacher = np.full(len(y), np.nan)
    teacher[::3] = expit(-x[::3, 0])
    expected, _ = train_student(x, y, c=0.4)
    actual, _ = fit_auxiliary_student(x, y, teacher, weight=0.0, c=0.4)
    probe = rng.normal(size=(20, 4))
    np.testing.assert_allclose(
        expit(probe @ actual), expit(probe @ expected), rtol=0, atol=1e-10
    )


def test_uncovered_outcomes_change_predictions_instead_of_being_dropped():
    x = np.ones((8, 1))
    teacher = np.array([0.1, 0.1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    wins_uncovered = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    losses_uncovered = np.zeros(8)
    wins_coef, _ = fit_auxiliary_student(x, wins_uncovered, teacher, weight=0.3, c=10)
    losses_coef, _ = fit_auxiliary_student(x, losses_uncovered, teacher, weight=0.3, c=10)
    assert expit(wins_coef[0]) > 0.55
    assert expit(losses_coef[0]) < 0.15


def test_auxiliary_mean_matches_repeated_soft_likelihood_without_reweighting_outcomes():
    x = np.array([[1.0, 0.5], [-0.7, 1.2], [0.3, -0.8], [1.4, 0.2], [-1.0, 0.7], [0.6, -1.1]])
    y = np.array([1, 0, 0, 1, 1, 0])
    teacher = np.array([0.15, np.nan, np.nan, 0.25, np.nan, np.nan])
    covered = np.isfinite(teacher)
    # weight * n_all / n_covered = 2: each covered soft likelihood occurs
    # twice in the equivalent summed objective, while each outcome occurs once.
    repeated_x = np.concatenate((x, x[covered], x[covered]))
    repeated_targets = np.concatenate((y, teacher[covered], teacher[covered]))
    expected, _ = train_student(repeated_x, repeated_targets, c=0.7)
    actual, _ = fit_auxiliary_student(x, y, teacher, weight=2 / 3, c=0.7)
    probe = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, -1.2]])
    np.testing.assert_allclose(
        expit(probe @ actual), expit(probe @ expected), rtol=0, atol=2e-7
    )


@pytest.mark.parametrize("swap_subset", [False, True])
def test_reversing_sides_preserves_complementary_series_predictions(swap_subset):
    rng = np.random.default_rng(890)
    x = rng.normal(size=(70, 3))
    y = rng.binomial(1, expit(x[:, 0] - x[:, 1]))
    teacher = expit(0.7 * x[:, 0] + 0.2 * x[:, 2])
    teacher[::4] = np.nan
    original, _ = fit_auxiliary_student(x, y, teacher, weight=0.5, c=0.5)
    swap = np.arange(len(y)) % 2 == 0 if swap_subset else np.ones(len(y), dtype=bool)
    swapped_x, swapped_y, swapped_teacher = x.copy(), y.copy(), teacher.copy()
    swapped_x[swap] *= -1
    swapped_y[swap] = 1 - swapped_y[swap]
    swapped_teacher[swap] = 1 - swapped_teacher[swap]
    reversed_coef, _ = fit_auxiliary_student(
        swapped_x, swapped_y, swapped_teacher, weight=0.5, c=0.5
    )
    probe = rng.normal(size=(20, 3))
    np.testing.assert_allclose(
        expit(-probe @ reversed_coef), 1 - expit(probe @ original), rtol=0, atol=2e-8
    )
    np.testing.assert_allclose(
        expit(-probe @ original), 1 - expit(probe @ original), rtol=0, atol=2e-16
    )
    assert expit(np.zeros(3) @ original) == 0.5


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("y", np.array([0.0, np.nan, 1.0])),
        ("y", np.array([0.0, 0.5, 1.0])),
        ("teacher", np.array([0.2, np.inf, np.nan])),
        ("teacher", np.array([-0.01, 0.5, np.nan])),
        ("teacher", np.array([0.2, 1.01, np.nan])),
    ],
)
def test_invalid_outcomes_and_soft_targets_are_rejected(field, invalid):
    inputs = {
        "x": np.array([[1.0], [-0.5], [0.2]]),
        "y": np.array([1, 0, 1]),
        "teacher": np.array([0.8, 0.2, np.nan]),
        "weight": 0.3,
    }
    inputs[field] = invalid
    with pytest.raises(ValueError):
        fit_auxiliary_student(**inputs)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("x", np.array([1.0, 2.0, 3.0])),
        ("x", np.array([[1.0], [np.inf], [0.0]])),
        ("x", np.empty((3, 0))),
        ("y", np.array([[1], [0], [1]])),
        ("teacher", np.array([0.2, np.nan])),
        ("weight", -0.1),
        ("weight", np.nan),
        ("c", 0.0),
        ("c", np.inf),
    ],
)
def test_malformed_training_inputs_are_rejected(field, invalid):
    inputs = {
        "x": np.array([[1.0], [-0.5], [0.2]]),
        "y": np.array([1, 0, 1]),
        "teacher": np.array([0.8, 0.2, np.nan]),
        "weight": 0.3,
        "c": 0.1,
    }
    inputs[field] = invalid
    with pytest.raises(ValueError):
        fit_auxiliary_student(**inputs)


def test_empty_training_cohort_is_rejected():
    with pytest.raises(ValueError):
        fit_auxiliary_student(np.empty((0, 2)), np.array([]), np.array([]), weight=0)


def test_missing_teacher_is_allowed_only_when_no_auxiliary_loss_is_requested():
    x = np.array([[1.0], [0.8], [-0.7]])
    y = np.array([1, 1, 0])
    teacher = np.full(3, np.nan)
    expected, _ = train_student(x, y, c=1.0)
    actual, _ = fit_auxiliary_student(x, y, teacher, weight=0, c=1.0)
    np.testing.assert_allclose(expit(x @ actual), expit(x @ expected), rtol=0, atol=1e-10)
    with pytest.raises(ValueError):
        fit_auxiliary_student(x, y, teacher, weight=0.3)
