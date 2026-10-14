"""Tiny consumer oracles; no fit or research data is needed."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, lognorm

from src.analysis.champion_history_evaluation import (
    evaluate_prediction, evaluate_first_map_baseline, paired_bootstrap, paired_comparison,
)
from src.models.prop_distribution import DistributionPrediction, DistributionMixture


class TinyPrediction:
    def __init__(self, scores, bo, stats=True):
        self.score_probs = np.asarray(scores, float)
        self.best_of = np.asarray(bo, int)
        self.p_win = self.score_probs[:, :3].sum(axis=1)
        self.length_probs = self.score_probs[:, :3] + self.score_probs[:, 3:]
        grid = (self.best_of // 2 + 1)[:, None] + np.arange(3)
        self.play_probs = np.column_stack([
            (self.length_probs * (grid >= m)).sum(axis=1) for m in range(1, 6)
        ])
        self.stats = stats

    def map_prediction(self, m):
        if not self.stats:
            raise NotImplementedError('score-only predictions have no stat head')
        n = len(self.best_of)
        return DistributionPrediction(np.full(n, 4. * m), np.full(n, .2),
                                      np.full(n, np.log(10. * m)), np.full(n, .2),
                                      self.length_probs, self.best_of)


def tiny_frames():
    series = pd.DataFrame({'row_index': [17, 29], 'match_id': ['a', 'b'],
                           'date': ['2026-01-02', '2026-02-03'], 'best_of': [3, 5],
                           'score_index': [1., 3.], 'winner': [1., 0.]})
    maps = pd.DataFrame({'row_index': [17, 17, 17, 29], 'game_id': ['a1', 'a2', 'a3', 'b1'],
                         'map_index': [1, 2, 3, 1], 'kills': [4., 8., np.nan, 3.],
                         'duration_minutes': [10., np.nan, 30., 12.]})
    return series, maps


def test_coherent_score_winner_length_rps_and_unplayed_map_brier_oracles():
    series, maps = tiny_frames()
    prediction = TinyPrediction([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]], [3, 5])
    metrics, rows, map_rows, profiles = evaluate_prediction(prediction, series, maps)
    assert rows.winner_nll.tolist() == pytest.approx([-np.log(.5), -np.log(.55)])
    assert rows.score_nll.tolist() == pytest.approx([-np.log(.3), -np.log(.3)])
    assert rows.length_nll.tolist() == pytest.approx([-np.log(.7), -np.log(.4)])
    assert rows.length_rps.tolist() == pytest.approx([.3**2, (.4-1)**2 + (.7-1)**2])
    # Recorded b1 is not evidence of a one-map BO5; the terminal score is 0:3.
    assert rows.map_play_brier.tolist() == pytest.approx([.3**2/3, (.6**2+.3**2)/5])
    assert metrics['map_play']['n'] == 8
    assert len(map_rows) == 4
    assert metrics['kills']['absent_labels'] == 1
    assert metrics['duration']['absent_labels'] == 1
    assert set(profiles.loc[profiles.dimension == 'patch', 'value']) == {'UNKNOWN'}


def test_all_maps_use_their_own_conditional_distribution_and_series_means():
    series, maps = tiny_frames()
    p = TinyPrediction([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]], [3, 5])
    metrics, rows, scored, _ = evaluate_prediction(p, series, maps)
    assert scored.kills_mean.tolist() == [4., 8., 12., 4.]
    expected = -nbinom.logpmf(8, 5, 1/(1+.2*8))
    assert scored.loc[1, 'kills_nll'] == pytest.approx(expected)
    assert rows.loc[0, 'kills_nll'] == pytest.approx(scored.loc[:2, 'kills_nll'].mean())
    assert metrics['kills']['series_weighted']['nll'] == pytest.approx(rows.kills_nll.mean())
    assert metrics['kills']['map_weighted']['nll'] == pytest.approx(scored.kills_nll.mean())
    assert np.isnan(scored.loc[2, 'kills_nll'])


def test_central_intervals_and_mean_mae_match_independent_marginal_quantiles():
    series, maps = tiny_frames()
    p = TinyPrediction([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]], [3, 5])
    _, _, scored, _ = evaluate_prediction(p, series, maps)
    assert scored.loc[0, 'kills_q25'] == nbinom.ppf(.25, 5, 1/(1+.2*4))
    assert scored.loc[0, 'kills_q75'] == nbinom.ppf(.75, 5, 1/(1+.2*4))
    assert scored.loc[0, 'duration_q10'] == pytest.approx(lognorm.ppf(.1, .2, scale=10))
    assert scored.loc[0, 'duration_q90'] == pytest.approx(lognorm.ppf(.9, .2, scale=10))
    assert scored.loc[0, 'duration_mae'] == pytest.approx(abs(10*np.exp(.02)-10))
    assert scored.loc[0, 'duration_coverage_80'] == 1.


def test_score_only_and_missing_labels_remain_explicit_not_zero():
    series, maps = tiny_frames()
    series.loc[1, ['winner', 'score_index']] = np.nan
    p = TinyPrediction([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]], [3, 5], stats=False)
    metrics, rows, scored, _ = evaluate_prediction(p, series, maps)
    assert metrics['kills']['status'] == 'unavailable_score_only'
    assert metrics['kills']['n'] == 0
    assert scored.kills_nll.isna().all()
    assert rows.loc[1, ['winner_nll', 'score_nll', 'length_nll', 'map_play_brier']].isna().all()
    assert scored.loc[0, 'kills_absence_reason'] == 'unavailable_score_only'
    assert metrics['winner']['absent_labels'] == 1


def test_bo1_length_is_excluded_but_winner_score_and_map_play_are_scored():
    series = pd.DataFrame({'row_index': [5], 'match_id': ['one'], 'date': ['2026-01-01'],
                           'best_of': [1], 'winner': [0], 'score_index': [3]})
    maps = pd.DataFrame(columns=['row_index', 'map_index', 'kills', 'duration_minutes'])
    metrics, rows, _, _ = evaluate_prediction(TinyPrediction([[.7, 0, 0, .3, 0, 0]], [1]), series, maps)
    assert np.isnan(rows.loc[0, 'length_nll'])
    assert metrics['length']['excluded_bo1'] == 1
    assert rows.loc[0, 'winner_nll'] == pytest.approx(-np.log(.3))
    assert rows.loc[0, 'map_play_brier'] == 0


def test_first_map_baseline_does_not_copy_predictions_onto_later_maps():
    series, maps = tiny_frames()
    pred = TinyPrediction([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]], [3, 5]).map_prediction(1)
    metrics, rows, scored, _ = evaluate_first_map_baseline(pred, series, maps)
    assert scored.loc[scored.map_index > 1, 'kills_nll'].isna().all()
    assert set(scored.loc[scored.map_index > 1, 'kills_absence_reason']) == {'unsupported_later_map_baseline'}
    assert metrics['kills']['unsupported_later_maps'] == 2
    assert rows.loc[0, 'kills_nll'] == pytest.approx(scored.loc[0, 'kills_nll'])
    assert metrics['length']['nll'] == pytest.approx((-np.log(.7)-np.log(.4))/2)
    assert metrics['map_play']['n'] == 8
    assert rows.map_play_brier.tolist() == pytest.approx([.3**2/3, (.6**2+.3**2)/5])


def test_paired_month_bootstrap_preserves_all_months_and_whole_series_means():
    candidate = pd.DataFrame({'row_index': [1, 2, 3, 4], 'match_id': list('abcd'),
                              'date': ['2026-01-01', '2026-01-02', '2026-02-01', '2026-03-01'],
                              'kills_nll': [2., 4., 10., np.nan]})
    control = candidate.copy()
    control['kills_nll'] = [1., 1., 6., np.nan]
    result = paired_comparison(candidate, control, 'kills', draws=5000, seed=7)
    assert result['months'] == ['2026-01', '2026-02', '2026-03']
    assert result['blocks'] == 3
    assert result['n'] == 3
    assert result['match_weighted_delta'] == pytest.approx(8/3)
    assert result['equal_month_delta'] == pytest.approx(3.)
    rng = np.random.default_rng(7)
    sampled = rng.integers(0, 3, size=(5000, 3))
    sums, counts = np.array([4., 4., 0.]), np.array([2., 1., 0.])
    denominator = counts[sampled].sum(axis=1)
    oracle = sums[sampled].sum(axis=1)[denominator > 0] / denominator[denominator > 0]
    tail = (1-result['confidence'])/2
    assert result['ci'] == pytest.approx(np.quantile(oracle, [tail, 1-tail]))
    assert result['undefined_draws'] == int((denominator == 0).sum())


def test_bootstrap_rejects_unpaired_population_and_applies_family_multiplicity():
    a = pd.DataFrame({'row_index': [1, 2], 'match_id': ['a', 'b'],
                       'date': ['2026-01-01', '2026-02-01'], 'winner_nll': [1., 2.]})
    with pytest.raises(ValueError, match='identical'):
        paired_comparison(a, a.iloc[::-1], 'winner', draws=20)
    result = paired_bootstrap(a, a, ['winner_nll'], draws=20, family_confidence=.95, family_size=12)
    assert result['confidence'] == pytest.approx(1-.05/12)
    assert result['metrics']['winner_nll']['ci'] == [0., 0.]


def test_exact_history_ensemble_is_not_mistaken_for_score_only_or_parameter_average():
    from src.models.champion_history_model import HistoryPrediction
    series, maps = tiny_frames()
    scores = np.array([[.2, .3, 0, .1, .4, 0], [.1, .15, .2, .3, .15, .1]])
    components = tuple(HistoryPrediction(
        scores, np.array([3, 5]), kills_mu=np.full((2, 5), mu),
        kills_alpha=np.full((2, 5), .2), duration_log_mu=np.full((2, 5), np.log(20)),
        duration_log_sigma=np.full((2, 5), .2),
    ) for mu in [2., 8., 14.])
    mixture = HistoryPrediction(scores, np.array([3, 5]), components=components)
    metrics, _, scored, _ = evaluate_prediction(mixture, series, maps)
    oracle_density = sum(nbinom.pmf(4, 5, 1/(1+.2*mu)) for mu in [2., 8., 14.])/3
    assert metrics['kills']['status'] == 'available'
    assert scored.loc[0, 'kills_nll'] == pytest.approx(-np.log(oracle_density))
    assert scored.loc[0, 'kills_nll'] != pytest.approx(-nbinom.logpmf(4, 5, 1/(1+.2*8)))


def test_auc_calibration_ties_and_impossible_events_are_not_epsilon_clipped():
    series = pd.DataFrame({'row_index': [1, 2, 3], 'match_id': ['a', 'b', 'c'],
                           'date': ['2026-01-01']*3, 'best_of': [1]*3,
                           'winner': [1, 0, 1], 'score_index': [0, 3, 0]})
    maps = pd.DataFrame(columns=['row_index', 'map_index', 'kills', 'duration_minutes'])
    prediction = TinyPrediction([[.5, 0, 0, .5, 0, 0], [.5, 0, 0, .5, 0, 0],
                                 [0, 0, 0, 1, 0, 0]], [1]*3)
    metrics, rows, _, _ = evaluate_prediction(prediction, series, maps, calibration_bins=2)
    # Positive probabilities 0 and .5 against the single negative .5: one tie.
    assert metrics['winner']['auc'] == .25
    assert metrics['winner']['calibration_ece'] == pytest.approx(1/3)
    assert np.isinf(rows.loc[2, 'winner_nll'])
    assert np.isinf(rows.loc[2, 'score_nll'])


def test_bootstrap_rejects_iid_map_rows_and_reports_totally_missing_task():
    a = pd.DataFrame({'row_index': [1, 2], 'match_id': ['a', 'b'],
                       'date': ['2026-01-01', '2026-02-01'], 'kills_nll': [np.nan, np.nan]})
    result = paired_comparison(a, a, 'kills', draws=10)
    assert result['n'] == 0
    assert result['undefined_draws'] == 10
    assert result['ci'] is None
    assert result['status'] == 'no_paired_labels'
    repeated = pd.concat([a.iloc[:1], a.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='whole series'):
        paired_comparison(repeated, repeated, 'kills', draws=10)
