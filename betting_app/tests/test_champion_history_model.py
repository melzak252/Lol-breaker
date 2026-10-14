import numpy as np
import pytest

from src.models.champion_history_model import HistoryPrediction


def test_legal_terminal_scores_determine_winner_length_and_map_existence():
    scores = np.array([[.7, 0, 0, .3, 0, 0], [.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]])
    original = scores.copy()
    result = HistoryPrediction(score_probs=scores, best_of=np.array([1, 3, 5]))
    np.testing.assert_allclose(result.p_win, [.7, .5, .45], atol=1e-14)
    np.testing.assert_allclose(result.length_probs, [[1, 0, 0], [.3, .7, 0], [.4, .3, .3]], atol=1e-14)
    np.testing.assert_allclose(result.play_probs, [[1, 0, 0, 0, 0], [1, 1, .7, 0, 0], [1, 1, 1, .6, .3]], atol=1e-14)
    np.testing.assert_array_equal(scores, original)


def test_illegal_terminal_score_mass_is_rejected_not_renormalized():
    with pytest.raises(ValueError):
        HistoryPrediction(score_probs=np.array([[.5, .1, 0, .4, 0, 0]]), best_of=np.array([1]))


def test_swapping_terminal_score_orientation_preserves_length_but_complements_winner():
    scores = np.array([[.11, .29, 0, .17, .43, 0]])
    a = HistoryPrediction(score_probs=scores, best_of=np.array([3]))
    b = HistoryPrediction(score_probs=scores[:, [3, 4, 5, 0, 1, 2]], best_of=np.array([3]))
    np.testing.assert_allclose(a.p_win + b.p_win, 1, atol=1e-14)
    np.testing.assert_allclose(a.length_probs, b.length_probs, atol=1e-14)
    np.testing.assert_allclose(a.play_probs, b.play_probs, atol=1e-14)


def test_map_forecasts_are_conditional_and_score_only_does_not_invent_props():
    scores = np.array([[.5, 0, 0, .5, 0, 0]])
    score_only = HistoryPrediction(scores, np.array([1]))
    with pytest.raises(RuntimeError, match="score-only"):
        score_only.map_prediction(1)
    forecast = HistoryPrediction(
        scores, np.array([1]), kills_mu=np.full((1, 5), 30.),
        kills_alpha=np.full((1, 5), .2), duration_log_mu=np.full((1, 5), 3.5),
        duration_log_sigma=np.full((1, 5), .25),
    )
    assert forecast.play_probs[0, 5 - 1] == 0
    assert forecast.map_prediction(5).kills_mu[0] == 30.
    for invalid in (0, 6, 1.5, True):
        with pytest.raises(ValueError):
            forecast.map_prediction(invalid)


def test_invalid_probabilities_and_partial_prop_parameters_are_rejected():
    for scores in ([[np.nan, 0, 0, .5, 0, 0]], [[-.1, 0, 0, 1.1, 0, 0]],
                   [[.2, 0, 0, .2, 0, 0]]):
        with pytest.raises(ValueError):
            HistoryPrediction(np.array(scores), np.array([1]))
    with pytest.raises(ValueError):
        HistoryPrediction(np.array([[.5, 0, 0, .5, 0, 0]]), np.array([1]),
                          kills_mu=np.ones((1, 5)))


@pytest.fixture(scope="module")
def fitted_history_pair():
    import pandas as pd
    from src.models.champion_history_model import fit_history_model

    rng = np.random.default_rng(82)
    shape = (6, 2, 5, 16)
    arrays = {
        "history": rng.normal(size=shape + (2,)).astype(np.float32),
        "champions": rng.integers(0, 4, shape, dtype=np.int64),
        "patches": rng.integers(0, 4, shape, dtype=np.int64),
        "competitions": rng.integers(0, 4, shape, dtype=np.int64),
        "roles": rng.integers(0, 6, shape, dtype=np.int64),
        "mask": np.ones(shape, dtype=bool),
        "players": rng.normal(size=(6, 2, 5, 2)).astype(np.float32),
        "role_probs": np.full((6, 2, 5, 6), 1 / 6, dtype=np.float32),
        "context": np.ones((6, 2), dtype=np.float32),
        "best_of": np.array([1, 3, 5, 1, 3, 5]),
    }
    arrays["mask"][0] = False
    arrays["history"][0] = np.nan
    arrays["context"][4:] = 1000.
    series = pd.DataFrame({"score_index": [0, 1, 5, 3, 4, 2]})
    maps = pd.DataFrame({"row_index": [0, 1, 1, 2, 2, 2],
                         "map_index": [1, 1, 2, 1, 2, 3],
                         "kills": [0, 30, 70, 18, 42, 100],
                         "duration_minutes": [22., 25., 49., 33., 41., 61.]})
    common = dict(config={}, seed=20261007, multitask=True)
    visible = fit_history_model(arrays, series, maps, np.arange(4),
                                champion_enabled=True, **common)
    hidden = fit_history_model(arrays, series, maps, np.arange(4),
                               champion_enabled=False, **common)
    return arrays, visible, hidden


def test_prediction_is_swap_equivariant_roster_invariant_and_nonmutating(fitted_history_pair):
    arrays, model, _ = fitted_history_pair
    before = {key: value.copy() for key, value in arrays.items()}
    original = model.predict(arrays)
    swapped = {key: value[:, ::-1].copy() if key in
               ("history", "champions", "patches", "competitions", "roles", "mask", "players", "role_probs")
               else value.copy() for key, value in arrays.items()}
    swapped_result = model.predict(swapped)
    np.testing.assert_allclose(original.score_probs[:, [3, 4, 5, 0, 1, 2]],
                               swapped_result.score_probs, atol=2e-7)
    np.testing.assert_allclose(original.kills_mu, swapped_result.kills_mu, rtol=2e-6)
    permuted = {key: value[:, :, [4, 2, 0, 3, 1]].copy() if key in
                ("history", "champions", "patches", "competitions", "roles", "mask", "players", "role_probs")
                else value.copy() for key, value in arrays.items()}
    np.testing.assert_allclose(original.score_probs, model.predict(permuted).score_probs,
                               atol=2e-7)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_hidden_champion_forecasts_ignore_all_identity_codes_and_have_matching_capacity(fitted_history_pair):
    arrays, visible, hidden = fitted_history_pair
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["champions"][:] = 999999
    first, second = hidden.predict(arrays), hidden.predict(changed)
    np.testing.assert_array_equal(first.score_probs, second.score_probs)
    np.testing.assert_array_equal(first.kills_mu, second.kills_mu)
    assert visible.fit_receipt["parameter_count"] == hidden.fit_receipt["parameter_count"]
    assert hidden.fit_receipt["epochs_completed"] == 12
    assert hidden.fit_receipt["optimizer_steps"] == 12
    assert hidden.fit_receipt["train_series_count"] == 4
    assert np.isfinite(visible.predict(changed).score_probs).all()
    assert visible.fit_receipt["initial_state_sha256"] == hidden.fit_receipt["initial_state_sha256"]
    assert visible.fit_receipt["epoch_order_sha256"] == hidden.fit_receipt["epoch_order_sha256"]


def test_preprocessing_excludes_held_out_rows_and_masked_history_is_inert(fitted_history_pair):
    arrays, model, _ = fitted_history_pair
    np.testing.assert_array_equal(model.preprocessing.mean["context"], [1., 1.])
    changed = {key: value.copy() for key, value in arrays.items()}
    changed["history"][0] = 1e6
    for key in ("champions", "patches", "competitions", "roles"):
        changed[key][0] = 1
    before = model.predict(arrays, np.array([0]))
    after = model.predict(changed, np.array([0]))
    np.testing.assert_array_equal(before.score_probs, after.score_probs)
    np.testing.assert_array_equal(before.kills_mu, after.kills_mu)
    empty = model.predict(arrays, np.array([], dtype=np.int64))
    assert empty.score_probs.shape == (0, 6)
    assert empty.map_prediction(1).kills_mu.shape == (0,)


def test_unknown_identity_has_the_same_forecast_as_explicit_unk(fitted_history_pair):
    arrays, model, _ = fitted_history_pair
    unknown = {key: value.copy() for key, value in arrays.items()}
    explicit = {key: value.copy() for key, value in arrays.items()}
    for key in ("champions", "patches", "competitions"):
        unknown[key][:] = 999999
        explicit[key][:] = 1
    np.testing.assert_array_equal(model.predict(unknown).score_probs,
                                  model.predict(explicit).score_probs)
    invalid = {key: value.copy() for key, value in arrays.items()}
    invalid["champions"][1, 0, 0, 0] = -1
    with pytest.raises(ValueError):
        model.predict(invalid)


def test_ensemble_keeps_seed_density_components_and_survives_joblib(fitted_history_pair, tmp_path):
    import joblib
    from src.models.champion_history_model import HistoryEnsemble
    from src.models.prop_distribution import DistributionMixture, distribution_nll

    arrays, visible, hidden = fitted_history_pair
    ensemble = HistoryEnsemble([visible, hidden, visible], kill_scale=1.5, duration_scale=1.2)
    prediction = ensemble.predict(arrays)
    mixture = prediction.map_prediction(1)
    assert isinstance(mixture, DistributionMixture)
    assert len(mixture.components) == 3
    expected = (2 * visible.predict(arrays).score_probs + hidden.predict(arrays).score_probs) / 3
    np.testing.assert_allclose(prediction.score_probs, expected, atol=1e-14)
    np.testing.assert_allclose(mixture.components[0].kills_alpha,
                               visible.predict(arrays).kills_alpha[:, 0] * 1.5)
    assert np.isfinite(distribution_nll(mixture, np.full(6, 1200.),
                                      np.full(6, 200.), np.full(6, np.nan))["kills"]).all()
    path = tmp_path / "bundle.joblib"
    joblib.dump(ensemble, path)
    restored = joblib.load(path).predict(arrays)
    np.testing.assert_array_equal(restored.score_probs, prediction.score_probs)
    np.testing.assert_array_equal(restored.map_prediction(1).kills_over(30), mixture.kills_over(30))


def test_calibration_fits_only_three_bounded_scalars_and_keeps_fitted_networks(fitted_history_pair):
    import pandas as pd
    import torch
    from src.models.champion_history_model import calibrate_history_ensemble

    arrays, visible, hidden = fitted_history_pair
    snapshots = [dict((key, value.clone()) for key, value in model.network.state_dict().items())
                 for model in (visible, hidden)]
    series = pd.DataFrame({"score_index": [0, 1, 5, 3, 4, 2]})
    maps = pd.DataFrame({"row_index": [4, 4, 5], "map_index": [1, 2, 1],
                         "kills": [0., 1700., 30.],
                         "duration_minutes": [10., 220., 30.]})
    ensemble, receipt = calibrate_history_ensemble(
        [visible, hidden, visible], arrays, series, maps, np.array([4, 5]))
    assert receipt["scalar_count"] == 3
    for name in ("score_temperature", "kill_scale", "duration_scale"):
        assert .25 <= getattr(ensemble, name) <= 4.
        assert np.isfinite(receipt[name]["nll"])
    assert receipt["kill_scale"]["observed_series"] == 2
    assert receipt["kill_scale"]["observed_maps"] == 3
    for model, snapshot in zip((visible, hidden), snapshots):
        for key, value in model.network.state_dict().items():
            assert torch.equal(value, snapshot[key])
    prediction = ensemble.predict(arrays, np.array([4, 5]))
    np.testing.assert_allclose(prediction.score_probs,
                               np.mean([part.score_probs for part in prediction.components], axis=0),
                               atol=1e-14)
