import json

import numpy as np
import pytest

from src.models.a0_probability_connector import ProbabilityConnector


def test_known_mixture_optimum_and_identity_boundary():
    p = np.full(100, .8)
    specialist = np.full(100, .2)
    model = ProbabilityConnector(0.).fit(p, specialist, np.r_[np.ones(60), np.zeros(40)])
    assert model.alpha == pytest.approx(1 / 3, abs=1e-12)
    np.testing.assert_allclose(model.predict_proba(p, specialist), .6, rtol=0., atol=1e-12)
    identity = ProbabilityConnector(.01).fit(p, specialist, np.r_[np.ones(80), np.zeros(20)])
    assert identity.alpha == 0.
    np.testing.assert_array_equal(identity.predict_proba(p, specialist), p)


def test_extreme_specialist_endpoint_keeps_native_probability_precision():
    with np.errstate(divide="raise", invalid="raise"):
        model = ProbabilityConnector(0.).fit([.8], [1e-20], [0])
    assert model.alpha == 1.
    np.testing.assert_array_equal(model.predict_proba([.8], [1e-20]), [1e-20])


def test_symmetry_neutral_and_json_roundtrip():
    a = np.array([.2, .4, .6, .8] * 10)
    b = np.array([.1, .2, .9, .7] * 10)
    y = np.array([0, 0, 1, 1] * 10)
    model = ProbabilityConnector(.01).fit(a, b, y)
    swapped = ProbabilityConnector(.01).fit(1 - a, 1 - b, 1 - y)
    assert model.alpha == pytest.approx(swapped.alpha, abs=1e-12)
    np.testing.assert_allclose(model.predict_proba(a, b) + model.predict_proba(1 - a, 1 - b), 1., atol=1e-15)
    np.testing.assert_array_equal(model.predict_proba([.5], [.5]), [.5])
    saved = ProbabilityConnector.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_array_equal(saved.predict_proba(a, b), model.predict_proba(a, b))
    with pytest.raises(ValueError):
        saved.predict_proba([0.], [.5])
    state = saved.to_dict()
    state['alpha'] = 1.01
    with pytest.raises(ValueError):
        ProbabilityConnector.from_dict(state)


def test_c0_anchored_connector_has_explicit_identity_and_roundtrips():
    from src.models.a0_probability_connector import C0ProbabilityConnector

    c0 = np.array([.2, .4, .6, .8])
    specialist = np.array([.8, .6, .4, .2])
    model = C0ProbabilityConnector(.01, anchor_identity="native12__full__full")
    model.fit(c0, specialist, [1, 1, 0, 0])
    assert model.anchor_identity == "native12__full__full"
    assert model.predict_proba(c0, specialist)[0] > c0[0]
    saved = C0ProbabilityConnector.from_dict(json.loads(json.dumps(model.to_dict())))
    assert saved.anchor_identity == "native12__full__full"
    np.testing.assert_array_equal(saved.predict_proba(c0, specialist), model.predict_proba(c0, specialist))
    wrong = model.to_dict()
    wrong["anchor_identity"] = "annual_a0"
    with pytest.raises(ValueError, match="anchor"):
        C0ProbabilityConnector.from_dict(wrong)


def test_nonnegative_slope_control_preserves_neutrality_and_identity():
    from src.models.a0_probability_connector import C0SlopeCalibration

    p = np.array([.2, .4, .6, .8])
    neutral = C0SlopeCalibration(.01).fit([.5] * 4, [0, 0, 1, 1])
    assert neutral.slope == 1.
    np.testing.assert_array_equal(neutral.predict_proba([.5]), [.5])
    model = C0SlopeCalibration(0.).fit(np.full(100, .8), np.r_[np.ones(60), np.zeros(40)])
    assert model.slope == pytest.approx(np.log(1.5) / np.log(4), abs=1e-12)
    np.testing.assert_allclose(model.predict_proba([.8, .2]), [.6, .4], atol=1e-12, rtol=0)
