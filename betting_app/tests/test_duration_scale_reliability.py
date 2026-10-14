"""P1 consumer contracts: fixture mathematics only, no source/DB or real fits."""
import copy
import hashlib
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm

from src.models.prop_distribution import DistributionModel, DistributionPrediction
from src.models.prop_distribution_runner import SelectedDistributionModel
from src.models.duration_scale_reliability import (
    DurationScaleModel, ScalePreprocessor, accept_fit_result,
    cal_scale_objective, load_duration_scale, load_selected_tree,
    mean_preserving_prediction, train_reference_sigma, train_scale_objective,
)
from src.analysis.duration_scale_reliability import partition_duration_rows


COUNT = 'x_pace_player_duration_minutes_count_mean_level'
GAP = 'x_pace_player_duration_minutes_mean_gap'
MISSING = 'x_pace_player_missing_duration_minutes_level'


def test_source_oracle_promotes_stored_values_before_aggregation():
    from scripts.verify_duration_scale_reliability import summaries

    players = np.array([[[[1e8], [1], [-1e8], [1], [1]],
                         [[0], [0], [0], [0], [0]]]], dtype=np.float32)
    teams = np.array([[[1e8], [1]]], dtype=np.float32)
    result = summaries(players, teams, [0], [0])
    np.testing.assert_array_equal(result['x_player_mean_0_level'], [.3])
    np.testing.assert_array_equal(result['x_player_mean_0_gap'], [.6])
    np.testing.assert_array_equal(result['x_team_0_level'], [50000000.5])


def scale_frame():
    return pd.DataFrame({COUNT: [0., 1., 3., 7., 15., 4.],
                         GAP: [1., 2., np.nan, 4., 8., 3.],
                         MISSING: [5., 0., 1., 2., 0., .5]})


def base_prediction():
    return DistributionPrediction(np.array([20., 30., 40.]), np.full(3, .1),
                                  np.log([25., 30., 35.]), np.full(3, .2),
                                  np.array([[1., 0., 0.], [.4, .6, 0.], [.2, .3, .5]]),
                                  np.array([1, 3, 5]))


class FixtureRegressor:
    """Serialized, untrained deterministic law, not a patched loader response."""
    def __init__(self, logarithmic=False):
        self.logarithmic = logarithmic

    def predict(self, values):
        result = 25. + np.asarray(values)[:, 0]
        return np.log(result) if self.logarithmic else result


def frozen_tree(tmp_path):
    tree = DistributionModel('tree', {'num_leaves': 15}, 20261007, 1)
    tree.kills_model = FixtureRegressor()
    tree.duration_model = FixtureRegressor(logarithmic=True)
    tree.duration_sigma = .2
    tree.kills_alpha = .1
    selected = SelectedDistributionModel({'duration': 'tree_leaves_15'},
                                         {'tree_leaves_15': tree}, ['x_fixture'])
    path = tmp_path / 'old_selected.joblib'
    joblib.dump(selected, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_selected_tree(path, expected_sha256=digest,
                                selected_id='tree_leaves_15', feature_columns=['x_fixture'])
    return loaded, path


def test_matched_global_and_conditional_laws_preserve_arithmetic_mean_not_median():
    base = base_prediction()
    snapshot = copy.deepcopy(base)
    expected_mean = np.exp(base.duration_log_mu + .5 * base.duration_log_sigma ** 2)
    for log_scale in (np.full(3, np.log(2.)), np.array([-.4, .3, .9])):
        actual = mean_preserving_prediction(base, log_scale)
        sigma = np.clip(.2 * np.exp(log_scale), 1e-4, 5.)
        mu = np.log(expected_mean) - .5 * sigma ** 2
        np.testing.assert_allclose(actual.duration_log_sigma, sigma, rtol=1e-14)
        np.testing.assert_allclose(actual.duration_log_mu, mu, rtol=1e-14)
        np.testing.assert_allclose(np.exp(actual.duration_log_mu + .5 * sigma ** 2),
                                   expected_mean, rtol=1e-10, atol=0.)
        np.testing.assert_allclose(actual.duration_over(31.5),
                                   lognorm.sf(31.5, s=sigma, scale=np.exp(mu)), atol=1e-12)
        assert not np.allclose(actual.duration_over(31.5), base.duration_over(31.5))
        assert not np.allclose(np.exp(actual.duration_log_mu), np.exp(base.duration_log_mu))
        for field in ('kills_mu', 'kills_alpha', 'length_probs', 'best_of'):
            np.testing.assert_array_equal(getattr(actual, field), getattr(base, field))
    for field in vars(base):
        np.testing.assert_array_equal(getattr(base, field), getattr(snapshot, field))


def test_scale_support_clips_parameters_without_clipping_observations():
    actual = mean_preserving_prediction(base_prediction(), [-100., 100., 0.])
    np.testing.assert_array_equal(actual.duration_log_sigma, [1e-4, 5., .2])
    assert np.isfinite(actual.duration_over(1000.)).all()


@pytest.mark.parametrize('log_scale', [[np.nan, 0., 0.], [np.inf, 0., 0.], [0., 0.], [[0., 0., 0.]]])
def test_scale_rejects_nonfinite_or_misaligned_offsets(log_scale):
    with pytest.raises(ValueError):
        mean_preserving_prediction(base_prediction(), log_scale)


def test_train_normalization_imputes_only_gap_and_retains_missing_player_count():
    train = scale_frame()
    original = train.copy(deep=True)
    pre = ScalePreprocessor.fit(train)
    imputed_gap = np.array([1., 2., 3., 4., 8., 3.])
    values = np.column_stack([np.log1p(train[COUNT]), imputed_gap])
    expected = np.column_stack([np.clip((values - values.mean(0)) / values.std(0), -3., 3.),
                                train[MISSING]])
    assert pre.gap_median == 3.
    np.testing.assert_allclose(pre.mean, values.mean(0))
    np.testing.assert_allclose(pre.scale, values.std(0))
    np.testing.assert_allclose(pre.transform(train), expected)
    pd.testing.assert_frame_equal(train, original)
    cal = pd.DataFrame({COUNT: [0., 16.], GAP: [np.nan, 10000.], MISSING: [5., 0.]})
    stats = (pre.mean.copy(), pre.scale.copy(), pre.gap_median)
    transformed = pre.transform(cal)
    assert transformed[0, 1] == pytest.approx((3. - values[:, 1].mean()) / values[:, 1].std())
    assert transformed[1, 1] == 3.
    np.testing.assert_array_equal(transformed[:, 2], [5., 0.])
    np.testing.assert_array_equal(pre.mean, stats[0])
    np.testing.assert_array_equal(pre.scale, stats[1])
    assert pre.gap_median == stats[2]


@pytest.mark.parametrize('field,value', [(COUNT, -1.), (COUNT, np.nan), (COUNT, np.inf),
                                        (GAP, np.inf), (GAP, -1.),
                                        (MISSING, np.nan), (MISSING, -1.), (MISSING, 6.)])
def test_invalid_support_or_missingness_does_not_become_cold_history(field, value):
    frame = scale_frame()
    frame.loc[0, field] = value
    with pytest.raises(ValueError):
        ScalePreprocessor.fit(frame)


@pytest.mark.parametrize('kind', ['constant_count', 'constant_missing', 'all_missing_gap', 'rank_deficient'])
def test_unidentifiable_train_design_hard_stops_without_dropping_covariates(kind):
    frame = scale_frame()
    if kind == 'constant_count':
        frame[COUNT] = 16.
    elif kind == 'constant_missing':
        frame[MISSING] = 0.
    elif kind == 'all_missing_gap':
        frame[GAP] = np.nan
    else:
        frame[GAP] = np.log1p(frame[COUNT])
        frame[MISSING] = np.linspace(0., 5., len(frame))
    with pytest.raises(ValueError, match='(?i)variance|rank|observed'):
        ScalePreprocessor.fit(frame)


def test_train_reference_is_analytic_and_never_uses_saved_cal_sigma():
    mu = np.log([20., 25., 30.])
    y = np.exp(mu + np.array([.1, -.2, .3]))
    assert train_reference_sigma(mu, y) == pytest.approx(np.sqrt(.14 / 3.))
    with pytest.raises(ValueError):
        train_reference_sigma(mu, np.exp(mu))  # zero residual scale is not positive support


@pytest.mark.parametrize('y', [[0., 25., 30.], [-1., 25., 30.], [np.nan, 25., 30.], [np.inf, 25., 30.]])
def test_head_objective_requires_finite_positive_observed_duration(y):
    with pytest.raises(ValueError):
        train_reference_sigma(np.log([20., 25., 30.]), y)


def independent_loss(theta, z, mu0, y, anchor_sigma, penalty, conditional):
    offsets = theta[0] + (z @ theta[1:] if conditional else z)
    sigma = np.clip(anchor_sigma * np.exp(offsets), 1e-4, 5.)
    log_mean = mu0 + .5 * anchor_sigma ** 2
    mu = log_mean - .5 * sigma ** 2
    nll = -lognorm.logpdf(y, s=sigma, scale=np.exp(mu)).mean()
    return nll + penalty


def test_train_objective_and_gradient_match_independent_density_and_finite_difference():
    z = np.array([[-1., .2, 0.], [.4, -.3, 1.], [.8, .6, .5], [-.2, -.5, 2.]])
    mu0 = np.log([22., 28., 35., 30.])
    y = np.array([25., 29., 40., 24.])
    theta = np.array([.1, -.2, .05, .1])
    sigma_t = .18
    expected = independent_loss(theta, z, mu0, y, sigma_t, .1 * np.dot(theta[1:], theta[1:]), True)
    loss, gradient = train_scale_objective(theta, z, mu0, y, sigma_t)
    assert loss == pytest.approx(expected, abs=1e-12)
    differences = []
    for index in range(4):
        step = np.eye(4)[index] * 1e-6
        plus, minus = theta + step, theta - step
        differences.append((independent_loss(plus, z, mu0, y, sigma_t, .1 * np.dot(plus[1:], plus[1:]), True)
                            - independent_loss(minus, z, mu0, y, sigma_t, .1 * np.dot(minus[1:], minus[1:]), True)) / 2e-6)
    np.testing.assert_allclose(gradient, differences, rtol=1e-6, atol=1e-7)


def test_cal_scalar_uses_old_cal_anchor_and_frozen_slopes_not_train_intercept():
    z = np.array([[-1., 0., 5.], [1., .3, 0.]])
    slopes = np.array([.1, -.2, .03])
    mu0, y, c, sigma0 = np.log([25., 30.]), np.array([23., 35.]), .15, .2
    offsets = z @ slopes
    expected = independent_loss(np.array([c]), offsets, mu0, y, sigma0, .001 * c ** 2, False)
    loss, derivative = cal_scale_objective(c, z, mu0, y, sigma0, slopes)
    assert loss == pytest.approx(expected, abs=1e-12)
    plus = independent_loss(np.array([c + 1e-6]), offsets, mu0, y, sigma0, .001 * (c + 1e-6) ** 2, False)
    minus = independent_loss(np.array([c - 1e-6]), offsets, mu0, y, sigma0, .001 * (c - 1e-6) ** 2, False)
    assert derivative == pytest.approx((plus - minus) / 2e-6, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize('stage,x', [('TRAIN_HEAD', [-1., .1, .2, .3]),
                                   ('PG_CAL', [np.log(4.)]), ('PC_CAL', [np.log(.25)])])
def test_successful_bounded_fit_records_boundary_without_budget_restart(stage, x):
    result = SimpleNamespace(success=True, x=np.array(x), fun=3., jac=np.zeros(len(x)),
                             nit=12, nfev=16, message='converged')
    receipt = accept_fit_result(result, stage=stage)
    assert receipt['status'] == 'success'
    assert receipt['boundary_hit'] is True
    assert receipt['iterations'] == 12
    assert receipt['estimation_operations'] == 1
    assert receipt['projected_gradient_inf'] <= 1e-5
    np.testing.assert_array_equal(receipt['parameters'], x)


@pytest.mark.parametrize('changes', [{'success': False}, {'fun': np.nan}, {'x': [1.01, 0., 0., 0.]},
                                    {'nit': 501}, {'x': [np.nan, 0., 0., 0.]},
                                    {'jac': [0., .01, 0., 0.]}])
def test_numerical_failure_stops_recipe_instead_of_accepting_fallback(changes):
    fields = dict(success=True, x=np.zeros(4), fun=3., jac=np.zeros(4),
                  nit=12, nfev=16, message='fixture')
    fields.update(changes)
    with pytest.raises(ValueError, match='(?i)fit|bound|finite|converg|iteration'):
        accept_fit_result(SimpleNamespace(**fields), stage='TRAIN_HEAD')


def test_selected_tree_load_extracts_raw_location_and_global_cal_sigma_without_mutation(tmp_path):
    tree, path = frozen_tree(tmp_path)
    before = path.read_bytes()
    frame = pd.DataFrame({'x_fixture': [0., 5.], 'best_of': [1, 1]})
    pred = tree.predict(frame)
    np.testing.assert_allclose(pred.duration_log_mu, np.log([25., 30.]))
    np.testing.assert_array_equal(pred.duration_log_sigma, [.2, .2])
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='(?i)hash|trust|pin'):
        load_selected_tree(path, expected_sha256='0' * 64,
                           selected_id='tree_leaves_15', feature_columns=['x_fixture'])
    with pytest.raises(ValueError, match='(?i)select|identity|tree'):
        load_selected_tree(path, expected_sha256=hashlib.sha256(before).hexdigest(),
                           selected_id='tree_leaves_31', feature_columns=['x_fixture'])


def test_own_saved_model_preserves_means_swaps_and_frozen_state(tmp_path):
    tree, old_path = frozen_tree(tmp_path)
    pre = ScalePreprocessor.fit(scale_frame())
    model = DurationScaleModel(tree=tree, preprocessor=pre, slopes=np.array([.1, -.1, .02]),
                               c_global=.2, c_conditional=-.1, train_reference_sigma=.18,
                               train_intercept=.9, fit_receipt={'estimation_operations': 4})
    frame = scale_frame().iloc[:2].copy()
    frame['x_fixture'], frame['best_of'] = [0., 5.], [1, 1]
    frame['team_a'], frame['team_b'] = ['a', 'c'], ['b', 'd']
    swapped = frame.copy()
    swapped['team_a'], swapped['team_b'] = frame['team_b'], frame['team_a']
    original_bytes = old_path.read_bytes()
    destination = tmp_path / 'p1.joblib'
    model.save(destination)
    saved_bytes = destination.read_bytes()
    restored = load_duration_scale(destination, expected_sha256=hashlib.sha256(saved_bytes).hexdigest())
    base = tree.predict(frame)
    expected_mean = np.exp(base.duration_log_mu + .5 * .2 ** 2)
    for arm, offset in [('P0', np.zeros(2)), ('PG', np.full(2, .2)),
                        ('PC', -.1 + pre.transform(frame) @ model.slopes)]:
        before, after, reversed_pred = model.predict(frame, arm=arm), restored.predict(frame, arm=arm), restored.predict(swapped, arm=arm)
        np.testing.assert_allclose(np.exp(after.duration_log_mu + .5 * after.duration_log_sigma ** 2), expected_mean, rtol=1e-10)
        np.testing.assert_allclose(after.duration_log_sigma, np.clip(.2 * np.exp(offset), 1e-4, 5.))
        np.testing.assert_array_equal(before.duration_over(31.5), after.duration_over(31.5))
        np.testing.assert_allclose(after.duration_over(31.5), reversed_pred.duration_over(31.5), atol=1e-8, rtol=0.)
    assert restored.train_intercept == .9  # retained provenance, never added to PC
    assert restored.fit_receipt == {'estimation_operations': 4}
    assert destination.read_bytes() == saved_bytes
    assert old_path.read_bytes() == original_bytes
    with pytest.raises(ValueError):
        restored.predict(frame, arm='BEST_ASSESS')
    with pytest.raises(FileExistsError):
        model.save(destination)


def chronology_frame():
    return pd.DataFrame({'match_id': ['t', 'c', 'a', 'late'],
                         'date': ['2025-06-30', '2025-07-01', '2026-01-08', '2025-06-29'],
                         'release_day': ['2025-06-30', '2025-07-02', '2026-01-09', '2025-07-01'],
                         'max_history_release_day': ['2025-06-29', None, '2026-01-07', '2025-06-28'],
                         'max_pace_release_day': ['2025-06-29', None, '2026-01-07', '2025-06-28']})


def test_partition_origin_keeps_late_releases_out_of_closed_train_and_same_day_withheld():
    frame = chronology_frame()
    masks = partition_duration_rows(frame)
    assert frame.loc[masks['TRAIN'], 'match_id'].tolist() == ['t']
    assert frame.loc[masks['CAL'], 'match_id'].tolist() == ['c']
    assert frame.loc[masks['ASSESS'], 'match_id'].tolist() == ['a']
    assert not any(mask.iloc[3] for mask in masks.values())


@pytest.mark.parametrize('field,value', [('max_history_release_day', '2025-06-30'),
                                        ('max_pace_release_day', '2025-07-01'),
                                        ('release_day', '2025-06-29')])
def test_origin_boundary_rejects_same_day_future_history_or_reversed_label_release(field, value):
    frame = chronology_frame()
    frame.loc[0, field] = value
    with pytest.raises(ValueError, match='(?i)release|prior|origin|history'):
        partition_duration_rows(frame)
