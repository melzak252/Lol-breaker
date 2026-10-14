import numpy as np
import pandas as pd
import pytest

from src.analysis.thesis_models import replay_monthly, positive_slope_calibration, select_development_configuration


def fixture_frame():
    dates = pd.date_range('2019-01-01', '2024-03-01', freq='7D')
    return pd.DataFrame({'golgg_match_id': np.arange(len(dates)).astype(str), 'date': dates,
                         'result_day': dates, 'player_elo': np.where(np.arange(len(dates)) % 2, .7, .3),
                         'y_true': np.arange(len(dates)) % 2})


def test_calibration_preserves_symmetry_and_positive_slope():
    p = np.linspace(.02, .98, 200)
    y = (np.arange(200) % 3 == 0).astype(int)
    slope = positive_slope_calibration(p, y)
    assert slope > 0
    from scipy.special import expit, logit
    np.testing.assert_allclose(expit(slope * logit(p)) + expit(slope * logit(1-p)), 1, atol=1e-14)


def test_replay_strict_release_cutoffs_and_future_mutation_invariance():
    data = fixture_frame()
    data.loc[0, 'result_day'] = pd.Timestamp('2024-02-15')
    pred, receipts = replay_monthly(data, ['player_elo'], start='2024-01-01', min_calibration_samples=2)
    jan = receipts[0]
    assert '0' not in jan['train_ids']
    assert pd.Timestamp(jan['train_max_release']) < pd.Timestamp(jan['origin'])
    data.loc[data.date >= '2024-02-01', 'y_true'] = 1-data.loc[data.date >= '2024-02-01', 'y_true']
    changed, _ = replay_monthly(data, ['player_elo'], start='2024-01-01', min_calibration_samples=2)
    np.testing.assert_allclose(pred.loc[pred.origin == '2024-01-01', 'p_calibrated'], changed.loc[changed.origin == '2024-01-01', 'p_calibrated'])
    assert all(pd.Timestamp(r['cal_max_release']) < pd.Timestamp(r['origin']) for r in receipts if r['cal_max_release'])


def test_replay_swapped_input_gives_complementary_probability():
    data = fixture_frame()
    pred, _ = replay_monthly(data, ['player_elo'], start='2024-01-01')
    opposite = data.copy()
    opposite['player_elo'] = 1-opposite.player_elo
    opposite['y_true'] = 1-opposite.y_true
    swapped, _ = replay_monthly(opposite, ['player_elo'], start='2024-01-01')
    np.testing.assert_allclose(pred.p_calibrated + swapped.p_calibrated, 1, atol=1e-6)


def test_selection_ignores_all_post_development_rows():
    data = fixture_frame()
    grid = {'logistic': [{'C': .1}, {'C': 1.0}]}
    before = select_development_configuration(data, ['player_elo'], candidates=grid)
    data.loc[data.date >= '2024-01-01', ['player_elo', 'y_true']] = [np.nan, 99]
    after = select_development_configuration(data, ['player_elo'], candidates=grid)
    assert before == after


def test_invalid_duplicate_ids_and_unclassified_orientation_fail():
    data = fixture_frame()
    with pytest.raises(ValueError, match='duplicate'):
        replay_monthly(pd.concat([data,data.iloc[:1]]), ['player_elo'])
    data['mystery_feature'] = 1
    with pytest.raises(ValueError, match='orientation'):
        replay_monthly(data, ['mystery_feature'])


@pytest.mark.parametrize('family,params', [
    ('logistic', {}), ('extra_trees', {'n_estimators': 5}),
    ('hist_gradient_boosting', {'max_iter': 5, 'min_samples_leaf': 5}),
    ('lightgbm', {'n_estimators': 5, 'min_child_samples': 5}),
    ('mlp', {'max_iter': 5}),
])
def test_all_families_fit_and_emit_valid_probabilities(family, params):
    data = fixture_frame()
    result, receipts = replay_monthly(data, ['player_elo'], family, params,
                                      start='2024-01-01', end='2024-02-01')
    assert result.p_calibrated.between(0,1).all()
    assert receipts[0]['family'] == family
    assert receipts[0]['cal_rows'] == 0


def test_2023_selected_on_development_rows_never_enter_calibration():
    data = fixture_frame()
    result, receipts = replay_monthly(data, ['player_elo'], start='2023-01-01', min_calibration_samples=2)
    jan = next(r for r in receipts if r['origin'] == '2024-01-01')
    assert jan['cal_rows'] == 0
    assert jan['calibration_slope'] == 1
    assert not jan['calibrator_available']


def test_unreleased_validation_labels_cannot_influence_selection():
    data = fixture_frame()
    withheld = data.index[(data.date >= '2023-01-01') & (data.date < '2024-01-01')][0]
    data.loc[withheld, 'result_day'] = pd.Timestamp('2024-01-01')
    grid = {'logistic': [{'C': .1}]}
    before = select_development_configuration(data, ['player_elo'], candidates=grid)
    data.loc[withheld, 'y_true'] = 1-data.loc[withheld, 'y_true']
    after = select_development_configuration(data, ['player_elo'], candidates=grid)
    assert before == after
    assert str(withheld) not in before['validation_ids']
