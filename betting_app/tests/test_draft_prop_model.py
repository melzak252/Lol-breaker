"""Consumer-visible symmetry and exact-mixture likelihood invariants."""
import joblib
import pytest
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import nbinom, lognorm

from src.models.draft_prop_model import fit_draft_prop_model
from src.models.prop_distribution import DistributionMixture, fit_distribution_model, distribution_nll


def test_reflected_prediction_preserves_full_law_under_team_swap():
    n = 48
    a, b = np.arange(n) % 4, (np.arange(n) * 3 + 1) % 4
    frame = pd.DataFrame(dict(x_signal=np.linspace(-1., 1., n),
                              d_champion_a_0=a, d_champion_b_0=b,
                              kills=20. + 2. * (a + b),
                              duration_minutes=25. + a + b,
                              best_of=np.where(np.arange(n) % 2, 3, 5),
                              maps_played=np.where(np.arange(n) % 2, 2., 3.)))
    train, cal = frame.iloc[:32].copy(), frame.iloc[32:].copy()
    length = fit_distribution_model('pooled', train[['x_signal']].to_numpy(),
                                    train.best_of.to_numpy(), train.kills.to_numpy(),
                                    train.duration_minutes.to_numpy(), train.maps_played.to_numpy(),
                                    X_cal=cal[['x_signal']].to_numpy(),
                                    best_of_cal=cal.best_of.to_numpy(), kills_cal=cal.kills.to_numpy(),
                                    duration_cal=cal.duration_minutes.to_numpy(), maps_cal=cal.maps_played.to_numpy())
    model = fit_draft_prop_model(train, cal,
                                feature_columns=['x_signal', 'd_champion_a_0', 'd_champion_b_0'],
                                categorical_features=['d_champion_a_0', 'd_champion_b_0'],
                                side_swap={'d_champion_a_0': 'd_champion_b_0',
                                           'd_champion_b_0': 'd_champion_a_0'},
                                length_reference=length, reference_features=['x_signal'],
                                config={'n_estimators': 10, 'num_leaves': 3,
                                        'min_child_samples': 2, 'n_jobs': 1}, seed=7)
    before = cal.copy(deep=True)
    pred = model.predict(cal)
    swapped = cal.copy(deep=True)
    swapped[['d_champion_a_0', 'd_champion_b_0']] = cal[['d_champion_b_0', 'd_champion_a_0']].to_numpy()
    other = model.predict(swapped)
    for line in [15.5, 25.5, 40.5]:
        np.testing.assert_allclose(pred.kills_over(line), other.kills_over(line), rtol=0, atol=1e-14)
        np.testing.assert_allclose(pred.duration_over(line), other.duration_over(line), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(pred.weights, [.5, .5])
    for i, component in enumerate(pred.components):
        np.testing.assert_array_equal(component.kills_mu, other.components[1 - i].kills_mu)
        np.testing.assert_array_equal(component.duration_log_mu, other.components[1 - i].duration_log_mu)
        np.testing.assert_array_equal(component.length_probs, other.components[1 - i].length_probs)
    pd.testing.assert_frame_equal(cal, before)
    scores = distribution_nll(pred, cal.kills.to_numpy(), cal.duration_minutes.to_numpy(), cal.maps_played.to_numpy())
    kill_logs = np.stack([nbinom.logpmf(cal.kills, 1/c.kills_alpha,
                                       1/(1+c.kills_alpha*c.kills_mu)) for c in pred.components])
    duration_logs = np.stack([lognorm.logpdf(cal.duration_minutes, c.duration_log_sigma,
                                             scale=np.exp(c.duration_log_mu)) for c in pred.components])
    np.testing.assert_allclose(scores['kills'], -logsumexp(kill_logs + np.log(pred.weights)[:, None], axis=0), atol=1e-12)
    np.testing.assert_allclose(scores['duration'], -logsumexp(duration_logs + np.log(pred.weights)[:, None], axis=0), atol=1e-12)
    length_indices = (cal.maps_played.to_numpy() - (cal.best_of.to_numpy() // 2 + 1)).astype(int)
    length_logs = np.stack([np.log(c.length_probs[np.arange(len(cal)), length_indices])
                            for c in pred.components])
    np.testing.assert_allclose(scores['length'],
                               -logsumexp(length_logs + np.log(pred.weights)[:, None], axis=0),
                               atol=1e-12)


@pytest.fixture(scope='module')
def fitted_case():
    n = 96
    idx = np.arange(n)
    frame = pd.DataFrame({
        'x_signal': np.sin(idx / 7.),
        'p_global_kills_mean': np.where(idx < 8, np.nan, 26.),
        'd_champion_a_0': idx % 5,
        'd_champion_b_0': (idx * 3 + 2) % 5,
        'kills': 15. + (idx % 5) * 2 + (idx * 3 + 2) % 5 + idx % 7,
        'duration_minutes': 23. + idx % 5 + 3. * np.cos(idx / 9.),
        'best_of': np.where(idx % 2, 3, 5),
        'maps_played': np.where(idx % 2, 2. + (idx % 3 == 0), 3. + idx % 3),
    })
    train, cal = frame.iloc[:72].copy(), frame.iloc[72:].copy()
    # A genuine distribution model provides a nonuniform legal length PMF.
    length = fit_distribution_model(
        'pooled', train[['x_signal']].to_numpy(), train.best_of.to_numpy(),
        train.kills.to_numpy(), train.duration_minutes.to_numpy(), train.maps_played.to_numpy(),
        X_cal=cal[['x_signal']].to_numpy(), best_of_cal=cal.best_of.to_numpy(),
        kills_cal=cal.kills.to_numpy(), duration_cal=cal.duration_minutes.to_numpy(),
        maps_cal=cal.maps_played.to_numpy())
    kwargs = {
        'feature_columns': ['x_signal', 'p_global_kills_mean', 'd_champion_a_0', 'd_champion_b_0'],
        'categorical_features': ['d_champion_a_0', 'd_champion_b_0'],
        'side_swap': {'d_champion_a_0': 'd_champion_b_0', 'd_champion_b_0': 'd_champion_a_0'},
        'length_reference': length, 'reference_features': ['x_signal'],
        'config': {'n_estimators': 10, 'num_leaves': 3, 'min_child_samples': 2, 'n_jobs': 1},
        'seed': 17,
    }
    train_before, cal_before = train.copy(deep=True), cal.copy(deep=True)
    model = fit_draft_prop_model(train, cal, **kwargs)
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(cal, cal_before)
    return train, cal, kwargs, model


def test_cal_labels_cannot_change_train_mean_heads(fitted_case):
    train, cal, kwargs, model = fitted_case
    changed = cal.copy(deep=True)
    changed['kills'] += 40.
    changed['duration_minutes'] *= 2.
    refitted = fit_draft_prop_model(train, changed, **kwargs)
    before, after = model.predict(cal), refitted.predict(cal)
    for left, right in zip(before.components, after.components):
        np.testing.assert_array_equal(left.kills_mu, right.kills_mu)
        np.testing.assert_array_equal(left.duration_log_mu, right.duration_log_mu)
        np.testing.assert_array_equal(left.length_probs, right.length_probs)
    assert not np.isclose(model.duration_sigma, refitted.duration_sigma)


def test_unknown_champions_and_cold_history_have_legal_noninvented_laws(fitted_case):
    _, cal, kwargs, model = fitted_case
    cold = cal.iloc[:3].copy()
    cold['x_signal'] = np.nan
    cold['p_global_kills_mean'] = np.nan
    cold[['d_champion_a_0', 'd_champion_b_0']] = -1
    before = cold.copy(deep=True)
    prediction = model.predict(cold)
    np.testing.assert_array_equal(prediction.weights, [.5, .5])
    reference = kwargs['length_reference'].predict(cold[['x_signal']].to_numpy(), cold.best_of.to_numpy())
    for component in prediction.components:
        assert np.isfinite(component.kills_mu).all() and (component.kills_mu > 0).all()
        assert np.isfinite(component.duration_log_mu).all()
        assert (component.kills_alpha > 0).all() and (component.duration_log_sigma > 0).all()
        np.testing.assert_array_equal(component.length_probs, reference.length_probs)
        np.testing.assert_allclose(component.length_probs.sum(axis=1), 1.)
        legal = np.arange(3)[None, :] <= (cold.best_of.to_numpy() // 2)[:, None]
        assert (component.length_probs[~legal] == 0).all()
    pd.testing.assert_frame_equal(cold, before)




def test_cal_dispersion_minimizes_exact_mixture_proper_nll(fitted_case):
    _, cal, _, model = fitted_case
    prediction = model.predict(cal)
    fitted = distribution_nll(prediction, cal.kills, cal.duration_minutes, cal.maps_played)
    mus = np.stack([c.kills_mu for c in prediction.components])
    logmus = np.stack([c.duration_log_mu for c in prediction.components])
    for alpha in (1e-6, .001, .01, .1, 1., 100.):
        logs = nbinom.logpmf(cal.kills.to_numpy()[None, :], 1. / alpha, 1. / (1. + alpha * mus))
        candidate = -(logsumexp(logs, axis=0) - np.log(2.)).mean()
        assert fitted['kills'].mean() <= candidate + 1e-8
    for sigma in (1e-4, .01, .1, .5, 1., 5.):
        logs = lognorm.logpdf(cal.duration_minutes.to_numpy()[None, :], sigma, scale=np.exp(logmus))
        candidate = -(logsumexp(logs, axis=0) - np.log(2.)).mean()
        assert fitted['duration'].mean() <= candidate + 1e-8




def test_saved_model_replay_preserves_complete_law(fitted_case, tmp_path):
    _, cal, _, model = fitted_case
    path = tmp_path / 'draft_model.joblib'
    joblib.dump(model, path)
    restored = joblib.load(path)
    before, after = model.predict(cal), restored.predict(cal)
    np.testing.assert_array_equal(before.weights, after.weights)
    for left, right in zip(before.components, after.components):
        for field in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma', 'length_probs', 'best_of'):
            np.testing.assert_array_equal(getattr(left, field), getattr(right, field))


@pytest.mark.parametrize('column,value', [
    ('x_signal', np.inf), ('d_champion_a_0', .5),
    ('d_champion_a_0', -2), ('d_champion_a_0', 2 ** 31), ('d_champion_a_0', np.nan),
])
def test_prediction_rejects_invalid_numeric_and_category_contracts(fitted_case, column, value):
    _, cal, _, model = fitted_case
    bad = cal.copy()
    bad[column] = value
    with pytest.raises(ValueError):
        model.predict(bad)


def test_prediction_rejects_missing_columns_but_accepts_empty_rows(fitted_case):
    _, cal, _, model = fitted_case
    with pytest.raises(ValueError, match='missing'):
        model.predict(cal.drop(columns=['d_champion_b_0']))
    empty = model.predict(cal.iloc[:0])
    assert len(empty.best_of) == 0
    assert all(c.length_probs.shape == (0, 3) for c in empty.components)


@pytest.mark.parametrize('override', [
    {'side_swap': {'d_champion_a_0': 'd_champion_b_0'}},
    {'side_swap': {'d_champion_a_0': 'd_champion_a_0', 'd_champion_b_0': 'd_champion_b_0'}},
    {'categorical_features': []},
    {'reference_features': ['d_champion_a_0']},
    {'feature_columns': ['x_signal', 'x_signal']},
    {'config': {'objective': 'regression'}},
    {'config': {'deterministic': False}},
    {'config': {'n_estimators': 0}},
    {'config': {'n_jobs': None}},
    {'config': {'learning_rate': np.inf}},
])
def test_fit_rejects_invalid_schema_mapping_and_configuration_before_fitting(fitted_case, override):
    train, cal, kwargs, _ = fitted_case
    with pytest.raises(ValueError):
        fit_draft_prop_model(train, cal, **(kwargs | override))


@pytest.mark.parametrize('column,value', [
    ('kills', -1.), ('kills', 3.5), ('kills', np.inf),
    ('duration_minutes', 0.), ('duration_minutes', np.inf), ('maps_played', 6.),
])
def test_fit_rejects_invalid_cal_labels_before_fitting(fitted_case, column, value):
    train, cal, kwargs, _ = fitted_case
    bad = cal.copy()
    bad[column] = value
    with pytest.raises(ValueError):
        fit_draft_prop_model(train, bad, **kwargs)


def test_missing_task_labels_are_masked_and_no_support_is_rejected(fitted_case):
    train, cal, kwargs, _ = fitted_case
    partial_train, partial_cal = train.copy(), cal.copy()
    partial_train.loc[partial_train.index[:3], 'kills'] = np.nan
    partial_cal.loc[partial_cal.index[:2], 'duration_minutes'] = np.nan
    model = fit_draft_prop_model(partial_train, partial_cal, **kwargs)
    observed = partial_cal['duration_minutes'].notna().to_numpy()
    prediction = model.predict(partial_cal)
    y = partial_cal.loc[observed, 'duration_minutes'].to_numpy()
    logmus = np.stack([c.duration_log_mu[observed] for c in prediction.components])
    fitted = -logsumexp(lognorm.logpdf(y[None, :], model.duration_sigma,
                                     scale=np.exp(logmus)), axis=0).mean() + np.log(2.)
    for sigma in (.01, .1, 1.):
        alternative = -logsumexp(lognorm.logpdf(y[None, :], sigma,
                                               scale=np.exp(logmus)), axis=0).mean() + np.log(2.)
        assert fitted <= alternative + 1e-8
    unsupported = cal.copy()
    unsupported['kills'] = np.nan
    with pytest.raises(ValueError, match='support'):
        fit_draft_prop_model(train, unsupported, **kwargs)


@pytest.mark.parametrize('oracle', [False, True])
def test_prediction_never_reads_hidden_target_outcomes(fitted_case, oracle):
    train, cal, kwargs, model = fitted_case
    if not oracle:
        model = fit_draft_prop_model(train, cal, **(kwargs | {
            'feature_columns': ['x_signal', 'p_global_kills_mean'],
            'categorical_features': [], 'side_swap': {},
        }))
    original = model.predict(cal)
    changed = cal.copy(deep=True)
    changed['kills'] = 999.
    changed['duration_minutes'] = 999.
    changed['maps_played'] = 999.
    hidden = model.predict(changed)
    for left, right in zip(original.components, hidden.components):
        for name in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma', 'length_probs'):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
