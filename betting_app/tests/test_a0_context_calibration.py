import json

import numpy as np
import pytest

from src.models.a0_context_calibration import ContextSlopeCalibration


CONTEXT_NAMES = (
    "low_statistics",
    "novel_roster_support",
    "high_sports_disagreement",
)


def _contexts(rows=12):
    return np.asarray([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(rows)], dtype=float)


def test_pooled_calibration_is_complement_symmetric_and_neutral():
    p = np.asarray([0.12, 0.28, 0.43, 0.5, 0.57, 0.72, 0.88, 0.31])
    x = _contexts(len(p))
    y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0])
    model = ContextSlopeCalibration("pooled", 10.0).fit(p, y, x)

    q = model.predict_proba(p, x)
    q_complement = model.predict_proba(1 - p, x)
    np.testing.assert_allclose(q_complement, 1 - q, atol=1e-14, rtol=0)
    assert model.predict_proba([0.5], [[1, 0, 1]])[0] == 0.5


def test_fit_enforces_each_of_eight_context_slopes_with_inclusive_bounds():
    p = np.tile([0.08, 0.2, 0.4, 0.6, 0.8, 0.92], 8)
    x = np.repeat(_contexts(8), 6, axis=0)
    y = (np.arange(len(p)) % 3 == 0).astype(float)
    model = ContextSlopeCalibration("independent", 1.0).fit(p, y, x)

    slopes = []
    for state in _contexts(8):
        q = model.predict_proba([0.2, 0.8], np.repeat(state[None, :], 2, axis=0))
        slopes.append((np.log(q[1] / (1 - q[1])) - np.log(q[0] / (1 - q[0]))) / (np.log(0.8 / 0.2) - np.log(0.2 / 0.8)))
    assert np.all(np.asarray(slopes) >= 0.05 - 1e-8)
    assert np.all(np.asarray(slopes) <= 20.0 + 1e-8)


def test_global_negative_label_oracle_reaches_lower_slope_bound_and_roundtrips():
    p = np.full(20, 0.9)
    y = np.zeros(20)
    model = ContextSlopeCalibration("global", 1.0).fit(p, y, np.full((20, 3), np.nan))
    artifact = model.to_dict()
    assert artifact["parameters"]["a0"] == pytest.approx(0.05, abs=1e-7)
    restored = ContextSlopeCalibration.from_dict(json.loads(json.dumps(artifact)))
    assert restored.to_dict() == artifact


def test_unknown_context_has_no_effect_and_is_recorded_without_mutation():
    p = np.asarray([0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.1, 0.9])
    y = np.asarray([0, 1, 0, 1, 1, 0, 0, 1])
    contexts = _contexts(len(p))
    contexts[0, 1] = np.nan
    original = contexts.copy()
    model = ContextSlopeCalibration("pooled", 10.0).fit(p, y, contexts)
    changed = contexts.copy()
    changed[0, 1] = 0.0

    np.testing.assert_array_equal(contexts, original)
    assert model.predict_proba(p, contexts)[0] == pytest.approx(model.predict_proba(p, changed)[0])
    assert model.to_dict()["fit_diagnostics"]["unknown_context_counts"]["novel_roster_support"] == 1


def test_controls_respond_to_their_penalty_centers_and_global_ignores_context():
    p = np.tile([0.2, 0.8], 32)
    x = np.repeat(_contexts(8), 8, axis=0)
    y = np.tile([0, 1, 1, 0, 0, 1, 1, 0], 8)
    pooled = ContextSlopeCalibration("pooled", 10.0).fit(p, y, x)
    independent = ContextSlopeCalibration("independent", 10.0).fit(p, y, x)
    global_model = ContextSlopeCalibration("global", 10.0).fit(p, y, np.full((len(p), 3), np.nan))

    pooled_q = pooled.predict_proba(p, x)
    independent_q = independent.predict_proba(p, x)
    assert np.max(np.abs(pooled_q - independent_q)) > 1e-5
    context_variants = np.tile([0, 0, 0], (len(p), 1)).astype(float)
    alternate = context_variants.copy()
    alternate[:, :] = _contexts(8)[np.arange(len(p)) % 8]
    np.testing.assert_array_equal(
        global_model.predict_proba(p, context_variants),
        global_model.predict_proba(p, alternate),
    )


def test_penalty_changes_context_arm_fit_toward_its_declared_center():
    p = np.tile([0.1, 0.2, 0.7, 0.9], 20)
    x = np.tile(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float), (20, 1))
    y = np.tile([0, 0, 0, 1], 20)
    weak = ContextSlopeCalibration("pooled", 1.0).fit(p, y, x)
    strong = ContextSlopeCalibration("pooled", 100.0).fit(p, y, x)
    assert abs(strong.to_dict()["parameters"]["a_low_statistics"] - strong.to_dict()["parameters"]["a0"]) < abs(weak.to_dict()["parameters"]["a_low_statistics"] - weak.to_dict()["parameters"]["a0"])


def test_fit_is_deterministic_and_artifact_roundtrip_preserves_predictions():
    p = np.asarray([0.13, 0.25, 0.37, 0.62, 0.74, 0.87] * 4)
    x = np.tile(_contexts(6), (4, 1))
    y = (np.arange(len(p)) % 4 < 2).astype(float)
    first = ContextSlopeCalibration("pooled", 10.0).fit(p, y, x)
    second = ContextSlopeCalibration("pooled", 10.0).fit(p, y, x)
    restored = ContextSlopeCalibration.from_dict(json.loads(json.dumps(first.to_dict())))

    assert first.to_dict() == second.to_dict()
    np.testing.assert_array_equal(restored.predict_proba(p, x), first.predict_proba(p, x))
    artifact = first.to_dict()
    assert artifact["anchor_identity"] == "native12__full__full"
    assert tuple(artifact["context_names"]) == CONTEXT_NAMES


def test_invalid_inputs_extreme_probabilities_and_malformed_artifacts_reject():
    model = ContextSlopeCalibration("pooled", 10.0)
    x = _contexts(4)
    with pytest.raises(ValueError):
        model.fit([0.0, 0.2, 0.8, 1.0], [0, 0, 1, 1], x)
    with pytest.raises(ValueError):
        model.fit([0.2, np.nan, 0.8, 0.5], [0, 0, 1, 1], x)
    with pytest.raises(ValueError):
        model.fit([0.2, 0.3, 0.8, 0.9], [0, 2, 1, 0], x)
    with pytest.raises(ValueError):
        model.fit([0.2, 0.3, 0.8, 0.9], [0, 0, 1, 1], np.full((4, 3), 2.0))
    fitted = ContextSlopeCalibration("global", 1.0).fit(
        [1e-12, 1e-8, 1 - 1e-6, 0.9], [0, 0, 1, 1], np.full((4, 3), np.nan)
    )
    q = fitted.predict_proba([1e-12, 1e-8, 1 - 1e-6, 0.9], np.full((4, 3), np.nan))
    assert np.all(np.isfinite(q)) and np.all((q > 0) & (q < 1))
    with pytest.raises(ValueError):
        fitted.predict_proba([0.0], np.full((1, 3), np.nan))
    saturating = fitted.to_dict()
    saturating["parameters"] = {"a0": 20.0}
    saturated_model = ContextSlopeCalibration.from_dict(saturating)
    with pytest.raises(ValueError):
        saturated_model.predict_proba([1e-300], np.full((1, 3), np.nan))
    malformed = fitted.to_dict()
    malformed["anchor_identity"] = "annual_a0"
    with pytest.raises(ValueError):
        ContextSlopeCalibration.from_dict(malformed)
