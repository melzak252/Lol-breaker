import json

import numpy as np
import pytest

from src.models.a0_sports_correction import AnchoredSportsCorrection


def test_identity_and_neutral_predictions_are_exact():
    p_a0 = np.array([0.2, 0.5, 0.8])
    model = AnchoredSportsCorrection(include_specialist=True, penalty=0.01)
    assert np.array_equal(model.predict_proba(p_a0, np.array([0.3, 0.5, 0.7]), [1, 3, 5]), p_a0)

    neutral = AnchoredSportsCorrection(True, 0.01).fit(
        np.full(20, 0.5), np.full(20, 0.5), np.full(20, 3), np.tile([0, 1], 10)
    )
    assert np.array_equal(
        neutral.predict_proba(np.full(20, 0.5), np.full(20, 0.5), np.full(20, 3)), np.full(20, 0.5)
    )


def test_correctly_calibrated_nonneutral_anchor_is_an_optimum():
    p = np.full(100, 0.8)
    y = np.r_[np.ones(80), np.zeros(20)]
    model = AnchoredSportsCorrection(False, 0.01).fit(p, None, np.ones(100), y)
    predicted = model.predict_proba(p, None, np.ones(100))
    np.testing.assert_allclose(predicted, p, rtol=0, atol=1e-12)
    actual = np.mean(-y * np.log(predicted) - (1 - y) * np.log1p(-predicted))
    actual += 0.005 * np.square(model.coefficients).sum()
    assert model.fit_diagnostics["objective"] == pytest.approx(actual, abs=1e-12)


def test_side_swap_and_outcome_flip_preserve_fitted_coefficients():
    p_a0 = np.array([0.25, 0.4, 0.65, 0.8] * 12)
    p_specialist = np.array([0.15, 0.55, 0.72, 0.9] * 12)
    best_of = np.array([1, 3, 5, 3] * 12)
    y = np.array([0, 1, 1, 0] * 12)
    original = AnchoredSportsCorrection(True, 0.001).fit(p_a0, p_specialist, best_of, y)
    swapped = AnchoredSportsCorrection(True, 0.001).fit(1 - p_a0, 1 - p_specialist, best_of, 1 - y)

    assert np.allclose(
        original.predict_proba(p_a0, p_specialist, best_of),
        1 - swapped.predict_proba(1 - p_a0, 1 - p_specialist, best_of), atol=1e-10,
    )
    assert np.allclose(original.coefficients, swapped.coefficients, atol=1e-8)


def test_information_bearing_specialist_improves_synthetic_conditional_loss():
    p_a0 = np.full(200, 0.5)
    p_specialist = np.tile(np.array([0.25, 0.75]), 100)
    y = (p_specialist > 0.5).astype(int)
    best_of = np.full(200, 3)
    candidate = AnchoredSportsCorrection(True, 1e-4).fit(p_a0, p_specialist, best_of, y)
    control = AnchoredSportsCorrection(False, 1e-4).fit(p_a0, None, best_of, y)

    def loss(probability):
        return -np.mean(y * np.log(probability) + (1 - y) * np.log1p(-probability))

    assert loss(candidate.predict_proba(p_a0, p_specialist, best_of)) < loss(control.predict_proba(p_a0, None, best_of))
    assert candidate.coefficients[1, 2] > 0


def test_bo_without_fit_support_remains_identity():
    p_a0 = np.array([0.25, 0.75])
    model = AnchoredSportsCorrection(False, 0.01).fit(p_a0, None, [1, 1], [0, 1])
    assert np.array_equal(model.predict_proba(p_a0, None, [5, 5]), p_a0)


def test_artifact_roundtrip_preserves_predictions_and_diagnostics():
    p_a0 = np.array([0.2, 0.4, 0.6, 0.8] * 8)
    p_specialist = np.array([0.3, 0.7, 0.8, 0.2] * 8)
    best_of = np.array([1, 3, 5, 3] * 8)
    y = np.array([0, 1, 1, 0] * 8)
    fitted = AnchoredSportsCorrection(True, 0.01).fit(p_a0, p_specialist, best_of, y)
    restored = AnchoredSportsCorrection.from_dict(json.loads(json.dumps(fitted.to_dict())))

    assert np.array_equal(
        restored.predict_proba(p_a0, p_specialist, best_of), fitted.predict_proba(p_a0, p_specialist, best_of)
    )
    assert restored.fit_diagnostics == fitted.fit_diagnostics


def test_invalid_inputs_and_incompatible_artifacts_fail_visibly():
    model = AnchoredSportsCorrection(True, 0.01)
    with pytest.raises(ValueError):
        model.predict_proba([0.0], [0.5], [3])
    with pytest.raises(ValueError):
        model.predict_proba([0.5], [0.5], [7])
    with pytest.raises(ValueError):
        model.predict_proba([0.5, 0.6], [0.5], [3, 3])
    with pytest.raises(ValueError):
        model.predict_proba([0.5], None, [3])
    state = AnchoredSportsCorrection(False, 0.01).to_dict()
    state["coefficients"] = [[0.0, 0.0]]
    with pytest.raises(ValueError):
        AnchoredSportsCorrection.from_dict(state)
