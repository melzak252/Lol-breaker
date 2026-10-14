import numpy as np
from src.analysis.thesis_metrics import _metrics


def test_endpoints_keep_brier_exact_and_logloss_finite():
    scores, bins = _metrics(np.array([0., 1., 1., 0.]), np.array([0., 1., 0., 1.]))
    assert scores['brier'] == .5
    assert np.isfinite(scores['logloss'])
    assert scores['ece'] == .5
    assert sum(row['n'] for row in bins) == 4
    assert [row['bin'] for row in bins] == [0, 9]


def test_equal_probability_and_single_class_diagnostics_are_explicit():
    scores, bins = _metrics(np.ones(4), np.full(4, .5))
    assert np.isnan(scores['auc'])
    assert np.isnan(scores['calibration_slope'])
    assert scores['brier'] == .25
    assert np.isclose(scores['logloss'], np.log(2))
    assert bins[0]['n'] == 4
