"""Learned specialist connections must remain causal and side symmetric."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.prematch_stack import SpecialistStack, stack_training_indices


def test_stack_learns_a_useful_expert_without_breaking_side_complements(tmp_path):
    rng = np.random.default_rng(765)
    signal = rng.normal(size=160)
    y = (signal > 0).astype(int)
    p = np.column_stack((1 / (1 + np.exp(-signal)), np.full(160, .5), 1 / (1 + np.exp(signal))))
    context = rng.uniform(size=(160, 4))
    model = SpecialistStack(C=.1).fit(p, context, y)
    result = model.predict_proba(p, context)
    assert np.mean((result >= .5) == y) > .9
    np.testing.assert_allclose(result + model.predict_proba(1 - p, context), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.predict_proba(np.full((2, 3), .5), context[:2]), .5, atol=0, rtol=0)
    path = tmp_path / "stack.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(p, context), result)


def test_stack_fits_only_available_out_of_time_forecasts():
    frame = pd.DataFrame({
        "match_id": ["a", "b", "c", "d"],
        "date": ["2023-10-01", "2023-10-02", "2023-10-03", "2024-01-01"],
        "release_day": ["2023-10-01", "2024-01-01", "2023-12-31", "2024-01-01"],
        "model_origin": ["2023-10-01"] * 3 + ["2024-01-01"],
        "training_cutoff": ["2023-04-01"] * 3 + ["2023-07-01"],
    })
    np.testing.assert_array_equal(np.flatnonzero(stack_training_indices(frame, "2024-01-01")), [0, 2])
    frame.loc[0, "training_cutoff"] = "2023-10-02"
    with pytest.raises(ValueError):
        stack_training_indices(frame, "2024-01-01")


def test_stack_learns_context_conditional_expert_usefulness_and_normalizes_train_context():
    rng = np.random.default_rng(812)
    context = rng.normal(size=(400, 4))
    context[:, 0] = np.tile([0.0, 1.0], 200)
    signal = rng.normal(size=400) * 2
    y = np.where(context[:, 0] > 0, signal > 0, signal < 0).astype(int)
    experts = np.column_stack((
        1 / (1 + np.exp(-signal)),
        np.full(len(signal), .5),
        1 / (1 + np.exp(signal)),
    ))

    model = SpecialistStack().fit(experts, context, y)

    prediction = model.predict_proba(experts, context)
    assert np.mean((prediction >= .5) == y) > .8
    unrelated_experts = np.concatenate((experts, experts[:1]))
    unrelated_context = np.concatenate((context, context[:1] + 1e6))
    np.testing.assert_allclose(
        model.predict_proba(unrelated_experts, unrelated_context)[:len(experts)],
        prediction, atol=1e-12, rtol=0,
    )


def test_stack_context_uses_symmetric_player_aggregates_and_expert_disagreement():
    from src.models.prematch_features import PLAYER_FEATURE_NAMES
    from src.models.prematch_stack import stack_context

    players = np.zeros((2, 2, 5, len(PLAYER_FEATURE_NAMES)), dtype=float)
    support = PLAYER_FEATURE_NAMES.index("log_appearance_count")
    inactivity = PLAYER_FEATURE_NAMES.index("log_days_since_last_appearance")
    rating_sd = PLAYER_FEATURE_NAMES.index("slow_rating_sd")
    players[0, :, :, support] = np.arange(10).reshape(2, 5)
    players[0, :, :, inactivity] = np.arange(10).reshape(2, 5)
    players[0, :, :, rating_sd] = 2
    players[1, :, :, support] = 4
    players[1, :, :, inactivity] = 3
    players[1, :, :, rating_sd] = 6
    batch = {"players": players}
    probs = np.array([[.1, .5, .9], [.2, .2, .2]])

    result = stack_context(batch, probs)

    np.testing.assert_allclose(result, [[0, 9, 2, np.std([.1, .5, .9])], [4, 3, 6, 0]], atol=1e-15)
    swapped_batch = {"players": players[:, ::-1].copy()}
    np.testing.assert_array_equal(stack_context(swapped_batch, probs), result)


def test_stack_rejects_invalid_expert_probabilities_and_nonfinite_player_context():
    from src.models.prematch_features import PLAYER_FEATURE_NAMES
    from src.models.prematch_stack import stack_context

    p = np.full((3, 3), .5)
    context = np.zeros((3, 4))
    y = np.array([0, 1, 0])
    for invalid in (np.nan, np.inf, -.01, 1.01):
        bad = p.copy()
        bad[0, 1] = invalid
        with pytest.raises(ValueError):
            SpecialistStack().fit(bad, context, y)
    batch = {"players": np.zeros((3, 2, 5, len(PLAYER_FEATURE_NAMES)))}
    batch["players"][0, 0, 0, PLAYER_FEATURE_NAMES.index("slow_rating_sd")] = np.nan
    with pytest.raises(ValueError):
        stack_context(batch, p)


def test_stack_context_requires_player_fields_and_chronology_rejects_duplicates():
    from src.models.prematch_stack import stack_context

    with pytest.raises(ValueError):
        stack_context({}, np.full((1, 3), .5))
    frame = pd.DataFrame({
        "match_id": ["a", "a"],
        "date": ["2023-10-01", "2023-10-02"],
        "release_day": ["2023-09-30", "2023-10-01"],
        "model_origin": ["2023-10-01", "2023-10-01"],
        "training_cutoff": ["2023-04-01", "2023-04-01"],
    })
    with pytest.raises(ValueError):
        stack_training_indices(frame, "2024-01-01")


def test_extrapolated_connector_probabilities_obey_the_shared_probability_policy():
    signal = np.tile([-1., 1.], 80)
    gate = np.repeat([-1., 1.], 80)
    experts = np.column_stack((1 / (1 + np.exp(-signal)), np.full(160, .5), np.full(160, .5)))
    context = np.column_stack((gate, np.zeros((160, 3))))
    model = SpecialistStack().fit(experts, context, (signal * gate > 0).astype(int))
    extreme_context = np.array([[1e6, 0., 0., 0.], [-1e6, 0., 0., 0.]])
    probability = model.predict_proba(np.tile([.9, .5, .5], (2, 1)), extreme_context)
    assert ((probability >= 1e-6) & (probability <= 1 - 1e-6)).all()


def test_debutants_have_finite_symmetric_context_without_invented_inactivity():
    from src.models.prematch_features import PLAYER_FEATURE_NAMES
    from src.models.prematch_stack import stack_context

    players = np.zeros((2, 2, 5, len(PLAYER_FEATURE_NAMES)))
    support = PLAYER_FEATURE_NAMES.index("log_appearance_count")
    inactivity = PLAYER_FEATURE_NAMES.index("log_days_since_last_appearance")
    players[..., PLAYER_FEATURE_NAMES.index("slow_rating_sd")] = 2
    players[0, ..., support] = 1
    players[0, ..., inactivity] = np.log1p(5)
    players[0, 0, 0, support] = 0
    players[0, 0, 0, inactivity] = np.nan
    players[1, ..., inactivity] = np.nan
    experts = np.array([[.7, .6, .8], [.5, .5, .5]])
    context = stack_context({"players": players}, experts)
    np.testing.assert_allclose(context[:, :3], [[0, np.log1p(5), 2], [0, 0, 2]])
    swapped_context = stack_context({"players": players[:, ::-1]}, 1 - experts)
    np.testing.assert_allclose(swapped_context, context, atol=1e-15, rtol=0)
    model = SpecialistStack().fit(experts, context, [1, 0])
    probability = model.predict_proba(experts, context)
    assert probability[1] == .5
    np.testing.assert_allclose(probability + model.predict_proba(1 - experts, swapped_context), 1, atol=1e-12, rtol=0)
    players[0, 0, 0, support] = 1
    with pytest.raises(ValueError):
        stack_context({"players": players}, experts)
