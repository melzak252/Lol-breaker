"""Fixed same-feature EV-loss experiment: algebra, local fits and sealed heads.

These tests intentionally lead the new API. No global optimality claim is made
for sigmoid return-MSE: its numerical oracle is a same-center local solve.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logit

from src.models.c0_betting_price_head import price_features
from src.models.c0_ev_loss_head import (
    C0EVLossHead,
    ev_loss_objective,
    fit_ev_loss_head,
    return_mse,
    select_ev_loss_head,
)
from src.models import ev_approach_models


OBJECTIVES = ("likelihood", "return_mse")


def population():
    rng = np.random.default_rng(20261006)
    m, z, e = rng.normal(size=(3, 64))
    m[:4] = [-2.8, -1.8, 1.8, 2.8]  # Both sides of the fixed .8 knot.
    frame = pd.DataFrame({
        "row_id": np.arange(64).astype(str), "q": expit(m), "p_C0": expit(z),
        "odds_a": 1.1 + np.exp(-e / 3), "odds_b": 1.1 + np.exp(e / 3),
    })
    frame["y"] = np.arange(64) % 2
    frame.attrs["c0_bindings"] = {
        "scope": "historical_development_only", "protocol_sha256": "synthetic",
        "train_membership_sha256": "synthetic-train", "code_sha256": "synthetic-code",
    }
    return frame


def independent_x(frame, recipe):
    m = logit(np.clip(frame.q.to_numpy(float), 1e-6, 1 - 1e-6))
    knot = np.log(4.)
    columns = [np.sign(m) * np.minimum(abs(m), knot),
               np.sign(m) * np.maximum(abs(m) - knot, 0.),
               np.log(frame.odds_b.to_numpy(float)) - np.log(frame.odds_a.to_numpy(float))]
    if recipe == "H-price":
        columns.append(logit(np.clip(frame.p_C0.to_numpy(float), 1e-6, 1 - 1e-6)))
    return np.column_stack(columns)


def independent_loss(theta, x, y, oa, ob, center, regularization, objective):
    eta = x @ theta
    p = expit(eta)
    if objective == "likelihood":
        raw = np.mean(np.logaddexp(0., eta) - y * eta)
    else:
        # Derive from both coherent side errors, not the production weight helper.
        error_a = .88 * oa * (p - y)
        error_b = .88 * ob * ((1 - p) - (1 - y))
        raw = np.mean((error_a ** 2 + error_b ** 2) / 2)
    return raw + regularization / 2 * np.sum((theta - center) ** 2)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("recipe", ["M-price", "H-price"])
def test_exact_objective_and_finite_difference_gradient(recipe, objective):
    frame = population()
    x = independent_x(frame, recipe)
    np.testing.assert_allclose(price_features(frame, recipe), x, atol=1e-14)
    center = np.array([1., 1., 0.] + ([0.] if recipe == "H-price" else []))
    theta = np.array([.7, 1.2, -.35] + ([.2] if recipe == "H-price" else []))
    args = (x, frame.y.to_numpy(float), frame.odds_a.to_numpy(float),
            frame.odds_b.to_numpy(float), center, .1, objective)
    value, gradient = ev_loss_objective(theta, *args)
    oracle = lambda t: independent_loss(t, *args)
    assert value == pytest.approx(oracle(theta), rel=1e-13, abs=1e-13)
    step = 1e-6
    numerical = [(oracle(theta + step * axis) - oracle(theta - step * axis)) / (2 * step)
                 for axis in np.eye(len(theta))]
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)


def test_return_mse_is_both_sides_taxed_not_best_side_or_normalized_weights():
    frame = pd.DataFrame({"y": [1., 0., 1.], "odds_a": [2., 3., 6.],
                          "odds_b": [4., 1.5, 2.]})
    p = np.array([.25, .6, .8])
    expected = .5 * ((.88 * frame.odds_a.to_numpy() * (p - frame.y)) ** 2
                     + (.88 * frame.odds_b.to_numpy() * ((1 - p) - (1 - frame.y))) ** 2)
    np.testing.assert_allclose(return_mse(frame, p), expected, atol=1e-14)
    x = np.ones((3, 1))
    theta, center = np.array([.4]), np.array([1.])
    args = (x, frame.y.to_numpy(), frame.odds_a.to_numpy(), frame.odds_b.to_numpy(), center, .1)
    base, grad = ev_loss_objective(theta, *args, "return_mse")
    scaled_args = (x, args[1], args[2] * 2, args[3] * 2, center, .1)
    scaled, scaled_grad = ev_loss_objective(theta, *scaled_args, "return_mse")
    penalty = .05 * np.sum((theta - center) ** 2)
    assert scaled - penalty == pytest.approx(4 * (base - penalty))
    np.testing.assert_allclose(scaled_grad - .1 * (theta - center),
                               4 * (grad - .1 * (theta - center)))


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("recipe", ["M-price", "H-price"])
def test_fit_uses_same_features_and_local_oracle_and_is_deterministic(recipe, objective):
    frame = population()
    head = fit_ev_loss_head(recipe, objective, frame, regularization=.1)
    repeat = fit_ev_loss_head(recipe, objective, frame, regularization=.1)
    assert head.coefficients == repeat.coefficients
    assert head.objective == objective
    center = np.array([1., 1., 0.] + ([0.] if recipe == "H-price" else []))
    bounds = [(0, None), (0, None), (None, None)] + ([(0, None)] if recipe == "H-price" else [])
    x = independent_x(frame, recipe)
    args = (x, frame.y.to_numpy(float), frame.odds_a.to_numpy(float),
            frame.odds_b.to_numpy(float), center, .1, objective)
    oracle = minimize(lambda t: independent_loss(t, *args), center, method="SLSQP",
                      bounds=bounds, options={"ftol": 1e-12, "maxiter": 2000})
    assert oracle.success
    theta = np.asarray(head.coefficients)
    assert independent_loss(theta, *args) == pytest.approx(oracle.fun, abs=2e-8)
    _, gradient = ev_loss_objective(theta, *args)
    projected = gradient.copy()
    for i, (lower, _) in enumerate(bounds):
        if lower is not None and theta[i] < 1e-7:
            projected[i] = min(projected[i], 0.)
    assert np.max(abs(projected)) < 2e-5
    np.testing.assert_allclose(head.predict_details(frame)["logit"], x @ theta, atol=1e-13)
    swapped = frame.assign(q=1-frame.q, p_C0=1-frame.p_C0,
                           odds_a=frame.odds_b, odds_b=frame.odds_a, y=1-frame.y)
    np.testing.assert_allclose(head.predict(frame) + head.predict(swapped), 1., atol=2e-14)
    np.testing.assert_allclose(return_mse(frame, head.predict(frame)),
                               return_mse(swapped, head.predict(swapped)), atol=2e-14)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_market_only_support_and_native_absence_are_recipe_specific(objective):
    frame = population().drop(columns="p_C0")
    assert np.isfinite(fit_ev_loss_head("M-price", objective, frame, regularization=.1).predict(frame)).all()
    with pytest.raises(ValueError):
        fit_ev_loss_head("H-price", objective, frame, regularization=.1)
    for broken in (frame.iloc[:0], frame.assign(y=0), frame.assign(q=.5)):
        with pytest.raises(ValueError, match="INSUFFICIENT_FIT_SUPPORT"):
            fit_ev_loss_head("M-price", objective, broken, regularization=.1)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("column,value", [
    ("q", True), ("p_C0", np.bool_(False)), ("q", -.01), ("p_C0", 1.01),
    ("q", np.nan), ("y", True), ("y", .5), ("y", np.inf),
    ("odds_a", True), ("odds_b", 1.), ("odds_a", np.inf), ("odds_b", np.nan),
    ("odds_a", 1e200),
])
def test_invalid_inputs_and_overflowing_derived_weights_rejected(objective, column, value):
    frame = population()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        fit_ev_loss_head("H-price", objective, frame, regularization=.1)


@pytest.mark.parametrize("probability", [[True, .4], [np.nan, .4], [np.inf, .4], [-.1, .4], [1.1, .4]])
def test_return_mse_rejects_invalid_probabilities(probability):
    frame = pd.DataFrame({"y": [0, 1], "odds_a": [2., 3.], "odds_b": [3., 2.]})
    with pytest.raises(ValueError):
        return_mse(frame, probability)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_endpoint_inputs_record_input_and_output_clipping_separately(objective):
    frame = population()
    head = fit_ev_loss_head("H-price", objective, frame, regularization=.1)
    endpoints = frame.iloc[:2].copy()
    endpoints["q"], endpoints["p_C0"] = [0., 1.], [1., 0.]
    details = head.predict_details(endpoints)
    np.testing.assert_array_equal(details["input_clipped"], [True, True])
    np.testing.assert_allclose(details["logit"], independent_x(endpoints, "H-price") @ head.coefficients)
    np.testing.assert_array_equal(details["probability"], np.clip(details["raw_probability"], 1e-6, 1-1e-6))
    np.testing.assert_array_equal(details["clipped"], details["probability"] != details["raw_probability"])
    np.testing.assert_allclose(return_mse(endpoints, [0., 1.]),
                               .5 * .88**2 * (endpoints.odds_a**2 + endpoints.odds_b**2)
                               * (np.array([0., 1.]) - endpoints.y)**2)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_json_roundtrip_exact_predictions_and_contract_tamper_rejection(objective):
    frame = population()
    head = fit_ev_loss_head("H-price", objective, frame, regularization=.1)
    artifact = json.loads(json.dumps(head.to_dict(), allow_nan=False))
    restored = C0EVLossHead.from_dict(artifact)
    assert restored.to_dict() == artifact
    assert artifact["tax_rate"] == .12
    assert artifact["bindings"] == frame.attrs["c0_bindings"]
    np.testing.assert_array_equal(head.predict(frame), restored.predict(frame))
    for key, value in [("objective", "profit"), ("tax_rate", 0.), ("epsilon", .01),
                       ("native_identity", "EXP081"), ("mode", "no_history"),
                       ("transform", "linear"), ("center", [0., 0., 0., 0.]),
                       ("bounds", [[None, None]] * 4), ("coefficients", [0.]),
                       ("regularization", True)]:
        broken = copy.deepcopy(artifact)
        broken[key] = value
        with pytest.raises(ValueError):
            C0EVLossHead.from_dict(broken)
    broken = copy.deepcopy(artifact)
    broken["optimizer"]["success"] = False
    with pytest.raises(ValueError):
        C0EVLossHead.from_dict(broken)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_cal_uses_identical_two_sided_mse_and_returns_existing_candidate_no_refit(objective, monkeypatch):
    train, cal = population().iloc[:40], population().iloc[40:]
    candidates = [fit_ev_loss_head("H-price", objective, train, regularization=r) for r in (.01, .1, 1.)]
    before = [copy.deepcopy(h.to_dict()) for h in candidates]
    def forbidden(*args, **kwargs):
        pytest.fail("CAL must not fit, refit, or invoke an optimizer")
    monkeypatch.setattr(ev_approach_models, "minimize", forbidden)
    selected, scores = select_ev_loss_head(candidates, cal)
    expected = [float(np.mean(.5 * .88**2 * (cal.odds_a.to_numpy()**2 + cal.odds_b.to_numpy()**2)
                             * (h.predict(cal) - cal.y.to_numpy())**2)) for h in candidates]
    best = min(expected)
    chosen = max((h for h, score in zip(candidates, expected) if score <= best + 1e-12),
                 key=lambda h: h.regularization)
    assert selected is chosen
    assert scores["lambda"] == chosen.regularization
    assert scores["return_mse"] == pytest.approx(expected[candidates.index(chosen)])
    np.testing.assert_allclose([r["return_mse"] for r in scores["candidates"]], expected)
    assert [h.to_dict() for h in candidates] == before
    for bad_cal in (cal.iloc[:0], cal.assign(y=0)):
        with pytest.raises(ValueError, match="INSUFFICIENT_FIT_SUPPORT"):
            select_ev_loss_head(candidates, bad_cal)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_exact_cal_tie_favors_largest_lambda_even_if_order_reversed(objective):
    frame = population()
    candidates = [fit_ev_loss_head("M-price", objective, frame, regularization=r) for r in (1., .01, .1)]
    cal = frame.iloc[:4].assign(q=.5, odds_a=2., odds_b=2.)
    chosen, scores = select_ev_loss_head(candidates, cal)
    assert chosen is candidates[0]
    assert scores["return_mse"] == pytest.approx(.88**2)
    for wrong in (candidates[:2], [candidates[0]] * 3):
        with pytest.raises(ValueError):
            select_ev_loss_head(wrong, cal)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_optimizer_failure_propagates_without_retry_or_usable_head(objective, monkeypatch):
    attempts = []
    def fail_solver(fun, initial, **kwargs):
        attempts.append(np.asarray(initial).copy())
        return SimpleNamespace(success=False, x=np.asarray(initial), fun=fun(initial),
                               status=2, message="forced line-search failure", nit=1, nfev=1)
    monkeypatch.setattr(ev_approach_models, "minimize", fail_solver)
    with pytest.raises(RuntimeError, match="forced line-search failure"):
        fit_ev_loss_head("H-price", objective, population(), regularization=.1)
    assert len(attempts) == 1
    np.testing.assert_array_equal(attempts[0], [1., 1., 0., 0.])


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_training_objective_uses_raw_not_scoring_clipped_probability(objective):
    theta, center = np.array([40.]), np.array([0.])
    x = np.array([[1.], [-1.]])
    y = np.array([0., 1.])
    oa, ob = np.array([2., 3.]), np.array([4., 2.])
    value, gradient = ev_loss_objective(theta, x, y, oa, ob, center, 0., objective)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    expected = independent_loss(theta, x, y, oa, ob, center, 0., objective)
    assert value == pytest.approx(expected, abs=1e-14)
    clipped = np.clip(expit(x @ theta), 1e-6, 1-1e-6)
    if objective == "likelihood":
        scoring = np.mean(-y*np.log(clipped) - (1-y)*np.log1p(-clipped))
    else:
        scoring = np.mean(.5*.88**2*(oa**2+ob**2)*(clipped-y)**2)
    assert abs(value - scoring) > 1e-6


@pytest.mark.parametrize("column,value", [
    ("odds_a", True), ("odds_b", 1.), ("odds_a", np.inf),
    ("odds_b", 1e200), ("y", True), ("y", .5),
])
def test_return_mse_validates_prices_labels_and_derived_weights(column, value):
    frame = pd.DataFrame({"y": [0, 1], "odds_a": [2., 3.], "odds_b": [3., 2.]})
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        return_mse(frame, [.2, .7])


@pytest.mark.parametrize("regularization", [True, -.1, np.nan, np.inf])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_invalid_regularization_is_not_a_fit_budget(regularization, objective):
    with pytest.raises(ValueError):
        fit_ev_loss_head("M-price", objective, population(), regularization=regularization)


def test_cal_tolerance_is_absolute_and_mixed_objectives_cannot_be_selected(monkeypatch):
    frame = population()
    candidates = [fit_ev_loss_head("M-price", "return_mse", frame, regularization=r)
                  for r in (.01, .1, 1.)]
    cal = frame.iloc[:4].assign(q=.5, odds_a=2., odds_b=2.)
    # Both classes, p=.5 has zero first derivative of mean squared error:
    # a 2e-7 perturbation increases CAL return-MSE by only 1.24e-13.
    def prediction(head, rows):
        offset = 2e-7 if head.regularization == 1. else 0.
        return np.full(len(rows), .5 + offset)
    monkeypatch.setattr(C0EVLossHead, "predict", prediction)
    chosen, _ = select_ev_loss_head(candidates, cal)
    assert chosen is candidates[-1]
    mixed = candidates[:2] + [
        fit_ev_loss_head("M-price", "likelihood", frame, regularization=1.)]
    with pytest.raises(ValueError):
        select_ev_loss_head(mixed, cal)
