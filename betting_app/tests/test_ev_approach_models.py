from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import ev_approach_models as models


METHODS = (
    "sports_cal",
    "market_cal",
    "market_shape",
    "joint_logloss",
    "joint_brier",
    "return_mse",
    "market_residual",
    "class_weighted",
    "class_weighted_corrected",
    "focal",
    "focal_corrected",
    "label_smoothing",
    "label_smoothing_corrected",
    "ranking",
    "ranking_calibrated",
    "close_target",
    "clv_target",
    "direct_profit",
    "kelly_utility",
)


QUOTE_REQUIRED = (
    "joint_logloss",
    "joint_brier",
    "return_mse",
    "market_residual",
    "close_target",
    "clv_target",
    "direct_profit",
    "kelly_utility",
)


PROBABILITY_METHODS = tuple(method for method in METHODS if method != "ranking")
CORRECTION_METHODS = (
    "class_weighted_corrected",
    "focal_corrected",
    "label_smoothing_corrected",
)


def opportunity_frame(n: int = 240) -> pd.DataFrame:
    """Deterministic, side-oriented training rows with real quote-like inputs."""
    index = np.arange(n)
    p_s = 0.32 + 0.36 * ((index % 5) / 4)
    q = 0.30 + 0.40 * ((index % 7) / 6)
    y = ((index * 7 + index // 5) % 11 < np.round(p_s * 11)).astype(int)
    odds_a = 1.65 + (index % 9) * 0.31
    odds_b = 1.72 + (index % 8) * 0.34
    close_q = 0.28 + 0.44 * ((index % 6) / 5)
    close_odds_a = 1.70 + (index % 7) * 0.33
    close_odds_b = 1.75 + (index % 6) * 0.36
    return pd.DataFrame({
        "p_s": p_s,
        "q": q,
        "y": y,
        "odds_a": odds_a,
        "odds_b": odds_b,
        "close_q": close_q,
        "close_odds_a": close_odds_a,
        "close_odds_b": close_odds_b,
        "best_of": np.where(index % 2, 3, 5),
        "favorite_side": np.where(q >= 0.5, "A", "B"),
    })


def _fit(method: str, train: pd.DataFrame):
    return models.fit_head(method, train, regularization=0.1, tax_rate=0.12)


def _predict(fitted, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(fitted.predict(frame), dtype=float)


def _swap_sides(frame: pd.DataFrame) -> pd.DataFrame:
    swapped = frame.copy()
    for column in ("p_s", "q", "close_q"):
        swapped[column] = 1.0 - swapped[column]
    swapped["y"] = 1 - swapped["y"]
    for left, right in (("odds_a", "odds_b"), ("close_odds_a", "close_odds_b")):
        swapped[left], swapped[right] = frame[right].to_numpy(), frame[left].to_numpy()
    swapped["favorite_side"] = np.where(swapped["q"] >= 0.5, "A", "B")
    return swapped


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_heads_are_side_complementary(method):
    train = opportunity_frame()
    probe = train.iloc[:32].copy()
    fitted = _fit(method, train)
    swapped_fit = _fit(method, _swap_sides(train))

    predicted = _predict(fitted, probe)
    opposite = _predict(swapped_fit, _swap_sides(probe))

    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 0) & (predicted <= 1))
    np.testing.assert_allclose(predicted + opposite, 1.0, atol=2e-5, rtol=0)


def test_raw_ranking_score_is_not_mislabeled_as_calibrated_probability():
    fitted = _fit("ranking", opportunity_frame())
    score_kind = fitted.metadata.get("score_kind")
    assert score_kind == "ranking_score"
    assert np.isfinite(_predict(fitted, opportunity_frame().iloc[:20])).all()


@pytest.mark.parametrize("method", METHODS)
def test_prediction_does_not_read_held_out_outcomes_or_future_close(method):
    frame = opportunity_frame()
    train, held_out = frame.iloc[:200].copy(), frame.iloc[200:].copy()
    fitted = _fit(method, train)
    baseline = _predict(fitted, held_out)

    changed = held_out.copy()
    changed["y"] = 1 - changed["y"]
    changed["close_q"] = 1 - changed["close_q"]
    changed["close_odds_a"] = changed["close_odds_a"] + 4.0
    changed["close_odds_b"] = changed["close_odds_b"] + 4.0
    after_mutation = _predict(fitted, changed)

    np.testing.assert_allclose(baseline, after_mutation, atol=0, rtol=0)


def test_native_and_market_controls_keep_their_distinct_information_identity():
    frame = opportunity_frame()
    probe = frame.iloc[:48].copy()
    altered_market = probe.copy()
    altered_market["q"] = 1.0 - altered_market["q"]
    altered_market["odds_a"], altered_market["odds_b"] = (
        probe["odds_b"].to_numpy(), probe["odds_a"].to_numpy())
    sports = _fit("sports_cal", frame)
    np.testing.assert_allclose(_predict(sports, probe), _predict(sports, altered_market), atol=0, rtol=0)

    altered_sports = probe.copy()
    altered_sports["p_s"] = 1.0 - altered_sports["p_s"]
    market = _fit("market_cal", frame)
    np.testing.assert_allclose(_predict(market, probe), _predict(market, altered_sports), atol=0, rtol=0)


def test_market_shape_is_market_only_and_uses_outer_spline_without_sports_input():
    train = opportunity_frame().drop(columns="p_s")
    fitted = _fit("market_shape", train)
    probe = pd.DataFrame({"q": [0.5, 0.8, 0.95]})
    prediction = _predict(fitted, probe)

    assert np.isfinite(prediction).all()
    assert fitted.metadata["transformations"]["features"] == [
        "sign(z_q)*min(abs(z_q), logit(0.8))",
        "sign(z_q)*max(abs(z_q)-logit(0.8), 0)",
    ]
    assert models._features(probe, "market_shape")[0][2, 1] > 0.0


@pytest.mark.parametrize("method", ["market_cal", "market_shape"])
def test_market_calibration_heads_fit_and_predict_from_q_without_sports_input(method):
    train = opportunity_frame().drop(columns="p_s")
    fitted = _fit(method, train)

    assert np.isfinite(_predict(fitted, pd.DataFrame({"q": [0.4, 0.6]}))).all()
    assert fitted.metadata["required_probability_inputs"] == ["q"]
    assert fitted.metadata["native_sports_signal_used"] is False

@pytest.mark.parametrize(("method", "column"), [("sports_cal", "p_s"), ("market_cal", "q")])
def test_calibration_regularization_shrinks_slope_toward_identity(method, column):
    frame = opportunity_frame()
    signal = frame[column].to_numpy()
    frame["y"] = (signal > 0.5).astype(int)

    unregularized = models.fit_head(method, frame, regularization=0.0)
    regularized = models.fit_head(method, frame, regularization=1.0)

    assert abs(regularized.coefficients[0] - 1.0) < abs(unregularized.coefficients[0] - 1.0)


@pytest.mark.parametrize("method", [
    "class_weighted", "focal", "label_smoothing", "clv_target",
])
def test_probability_validation_loss_scores_the_emitted_prediction(method):
    frame = opportunity_frame()
    fitted = _fit(method, frame)
    prediction = np.clip(_predict(fitted, frame), 1e-12, 1.0 - 1e-12)
    y = frame["y"].to_numpy(dtype=float)
    expected = np.mean(-(y * np.log(prediction) + (1.0 - y) * np.log1p(-prediction)))

    assert fitted.validation_loss(frame) == pytest.approx(expected)


def test_ranking_calibration_excludes_ranker_labels_released_at_or_after_first_cal_decision():
    frame = opportunity_frame(20)
    frame["row_id"] = [f"row-{index}" for index in range(len(frame))]
    frame["decision_at"] = pd.NaT
    frame["label_available_at"] = pd.NaT
    frame["scenario_decision_at"] = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    frame["scenario_label_available_at"] = (
        frame["scenario_decision_at"] + pd.Timedelta(hours=12)
    )
    frame.loc[15, "scenario_label_available_at"] = (
        frame.loc[16, "scenario_decision_at"] + pd.Timedelta(hours=1)
    )

    fitted = _fit("ranking_calibrated", frame)

    assert "row-15" not in fitted.metadata["fit_partition"]["fit_ids"]
    assert fitted.metadata["fit_partition"]["ordering"].startswith("actual/scenario")


@pytest.mark.parametrize("method", QUOTE_REQUIRED)
def test_quote_dependent_targets_reject_invalid_market_quotes(method):
    train = opportunity_frame()
    train.loc[train.index[0], "odds_a"] = 0.0

    with pytest.raises((ValueError, KeyError), match="(?i)quote|odds|price|invalid|positive"):
        _fit(method, train)


@pytest.mark.parametrize("method", CORRECTION_METHODS)
def test_corrected_loss_heads_return_finite_coherent_probabilities(method):
    train = opportunity_frame()
    fitted = _fit(method, train)
    predicted = _predict(fitted, train.iloc[-32:])

    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 0) & (predicted <= 1))


def test_return_mse_recovers_conditional_probability_when_odds_are_uninformative():
    # At each sports-signal value the outcome rate is known exactly in this
    # finite population; independent quote variation only changes return-loss
    # weights and must not move its unrestricted conditional-probability oracle.
    n_per_group = 1200
    p_values = np.repeat([0.30, 0.70], n_per_group)
    y = np.tile(np.r_[np.ones(360), np.zeros(840)], 2).astype(int)
    y[n_per_group:] = np.tile(np.r_[np.ones(840), np.zeros(360)], 1).astype(int)
    order = np.random.default_rng(20260928).permutation(len(y))
    frame = pd.DataFrame({
        "p_s": p_values[order],
        "q": np.full(len(y), 0.5),
        "y": y[order],
        "odds_a": np.random.default_rng(11).uniform(1.5, 5.0, len(y)),
        "odds_b": np.random.default_rng(12).uniform(1.5, 5.0, len(y)),
        "best_of": 3,
    })
    fitted = models.fit_head("return_mse", frame, regularization=0.0, tax_rate=0.12)
    probe = pd.DataFrame({
        "p_s": [0.30, 0.70], "q": [0.5, 0.5], "odds_a": [2.0, 4.0],
        "odds_b": [3.0, 2.0], "best_of": [3, 3],
    })
    predicted = _predict(fitted, probe)

    np.testing.assert_allclose(predicted, [0.30, 0.70], atol=0.08, rtol=0)
