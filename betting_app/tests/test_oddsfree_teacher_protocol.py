import numpy as np
import pandas as pd

from scripts.experiment_oddsfree_distillation import build_teachers, teacher_masks


def test_chronological_teacher_never_learns_target_year_outcomes(tmp_path):
    records = []
    xrow = [-2., -1., -.5, .5, 1., 2.]
    outcomes = [0, 0, 1, 0, 1, 1]
    for year in range(2020, 2025):
        for month in (6, 12):
            for index, (feature, outcome) in enumerate(zip(xrow, outcomes)):
                records.append(dict(golgg_match_id=f'{year}-{month}-{index}', date=f'{year}-{month:02d}-01',
                                    y_true=outcome, feature=feature, odds_a=1.9, odds_b=1.9, market=.5))
    frame = pd.DataFrame(records)
    matrix = frame[['feature']].to_numpy()
    original_dir, changed_dir = tmp_path / 'original', tmp_path / 'changed'
    original_dir.mkdir()
    changed_dir.mkdir()
    original, _ = build_teachers(frame, matrix, ['strength'], original_dir)
    changed = frame.copy()
    changed.loc[changed.date.str.startswith('2022'), 'y_true'] = 1 - changed.loc[changed.date.str.startswith('2022'), 'y_true']
    perturbed, _ = build_teachers(changed, matrix, ['strength'], changed_dir)
    earlier = frame.date.str[:4].isin(['2021', '2022'])
    later = frame.date.str.startswith('2023')
    np.testing.assert_array_equal(original['hybrid'][earlier], perturbed['hybrid'][earlier])
    assert np.max(np.abs(original['hybrid'][later] - perturbed['hybrid'][later])) > .01
    assert np.isnan(original['market'][frame.date.str.startswith('2020')]).all()


def test_teacher_fit_and_calibration_boundaries_exclude_prediction_year():
    dates = pd.Series(['2019-12-31', '2020-06-30', '2020-07-01', '2020-12-31', '2021-01-01', '2021-12-31', '2022-01-01'])
    masks = teacher_masks(dates, 2021)
    assert dates[masks['train']].tolist() == ['2020-06-30']
    assert dates[masks['calibration']].tolist() == ['2020-07-01', '2020-12-31']
    assert dates[masks['prediction']].tolist() == ['2021-01-01', '2021-12-31']
    assert np.stack(list(masks.values())).sum(axis=0).max() == 1
