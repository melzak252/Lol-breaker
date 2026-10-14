"""Contracts for the single-origin data-focus experiment, without archive access."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.a0_data_focus import drop_history_tokens, partition_indices, market_tier_weights


def test_history_corruption_cannot_reveal_padding_or_propagate_hidden_nan():
    history = torch.tensor([[[[[2., 3.], [float('nan'), 9.], [7., 8.]]]]])
    original = torch.tensor([[[[True, False, True]]]])
    inputs = {'history': history, 'mask': original, 'players': torch.ones(1, 2, 5, 4)}
    keep = torch.tensor([[[[False, True, True]]]])
    result = drop_history_tokens(inputs, keep)
    assert torch.equal(result['mask'], torch.tensor([[[[False, False, True]]]]))
    assert torch.equal(result['history'], torch.tensor([[[[[0., 0.], [0., 0.], [7., 8.]]]]]))
    assert torch.equal(original, torch.tensor([[[[True, False, True]]]]))
    assert history[0, 0, 0, 0, 0] == 2
    assert torch.isnan(history[0, 0, 0, 1, 0])
    assert torch.equal(result['players'], inputs['players'])


def test_corruption_commutes_with_side_swap_and_supports_empty_histories():
    history = torch.arange(2 * 5 * 16 * 3, dtype=torch.float32).reshape(1, 2, 5, 16, 3)
    mask = torch.ones((1, 2, 5, 16), dtype=torch.bool)
    keep = (history[..., 0].to(torch.int64) % 7) < 3
    result = drop_history_tokens({'history': history, 'mask': mask}, keep)
    swapped = drop_history_tokens({'history': history.flip(1), 'mask': mask.flip(1)}, keep.flip(1))
    torch.testing.assert_close(swapped['history'], result['history'].flip(1))
    assert torch.equal(swapped['mask'], result['mask'].flip(1))
    empty = drop_history_tokens({'history': history, 'mask': mask}, torch.zeros_like(mask))
    assert not empty['mask'].any()
    assert torch.equal(empty['history'], torch.zeros_like(history))


def test_corruption_rejects_broadcasting_across_players():
    inputs = {'history': torch.ones(2, 2, 5, 16, 33), 'mask': torch.ones(2, 2, 5, 16, dtype=torch.bool)}
    with pytest.raises(ValueError, match='shape'):
        drop_history_tokens(inputs, torch.ones(2, 2, 1, 16, dtype=torch.bool))


def test_partition_respects_release_and_boundary_dates():
    frame = pd.DataFrame({
        'date': ['2025-06-29', '2025-06-30', '2025-07-01', '2025-12-31', '2026-01-01'],
        'result_day': ['2025-06-30', '2025-07-01', '2025-07-02', '2026-01-01', '2026-01-02'],
        'feature_source_max_day': ['2025-06-28', '2025-06-29', '2025-06-30', '2025-12-30', '2025-12-31'],
    })
    split = partition_indices(frame)
    assert split['train'].tolist() == [0]
    assert split['cal'].tolist() == [2]
    assert split['validation'].tolist() == [4]
    frame.loc[4, 'feature_source_max_day'] = '2026-01-01'
    with pytest.raises(ValueError, match='feature'):
        partition_indices(frame)


def test_missing_cutoff_requires_verified_source_free_cold_start():
    frame = pd.DataFrame({
        'date': ['2025-06-29'], 'result_day': ['2025-06-29'],
        'feature_source_max_day': [None], 'unknown_players': [10],
        'min_player_series': [0], 'new_organization': [True],
    })
    with pytest.raises(ValueError, match='feature'):
        partition_indices(frame)
    history = {'mask': np.zeros((1, 2, 5, 16), dtype=np.uint8)}
    raw = {'gates': np.zeros((1, 2, 2))}
    assert partition_indices(frame, history=history, raw=raw)['train'].tolist() == [0]
    history['mask'][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match='feature'):
        partition_indices(frame, history=history, raw=raw)
    history['mask'][:] = 0
    raw['gates'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='feature'):
        partition_indices(frame, history=history, raw=raw)


def test_market_weighting_uses_train_membership_and_keeps_uncovered_tiers():
    frame = pd.DataFrame({'golgg_match_id': list('abcdef'), 'tier': ['major', 'major', 'regional', 'development', 'international', 'major']})
    train = np.arange(4)
    weights, receipt = market_tier_weights(frame, train, {'a', 'c', 'e', 'f'})
    expected = np.array([1., 1., 2., .5])
    expected /= expected.mean()
    np.testing.assert_allclose(weights, expected)
    assert receipt['covered_rows'] == 2
    repeated, _ = market_tier_weights(frame, train, {'a', 'c'})
    np.testing.assert_array_equal(weights, repeated)
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1.)


def test_market_weighting_rejects_absent_training_market_support():
    frame = pd.DataFrame({'golgg_match_id': ['old', 'future'], 'tier': ['major', 'regional']})
    with pytest.raises(ValueError, match='market'):
        market_tier_weights(frame, np.array([0]), {'future'})


def test_reporting_does_not_hide_extreme_errors_with_legacy_clipping():
    from src.analysis.a0_data_focus import _summary

    frame = pd.DataFrame({'y': [1, 0, 1, 0], 'p': [1e-6, .4, .9, .2]})
    summary = _summary(frame, 'p')
    assert summary['log_loss'] == pytest.approx(-np.log(1e-6 * .6 * .9 * .8) / 4)
    assert summary['tail_ge_2_5'] == 1


def test_market_join_preserves_orientation_despite_reordered_fixtures():
    from src.analysis.a0_data_focus import _market_view

    joined = pd.DataFrame({
        'golgg_match_id': ['1', '2'], 'team1_id': ['A', 'C'], 'team2_id': ['B', 'D'],
        'y': [1, 0], 'best_of': [3, 5], 'p_a0_full': [.3, .6],
    })
    panel = pd.DataFrame({
        'golgg_match_id': ['2', '1'], 'team1_id': ['C', 'A'], 'team2_id': ['D', 'B'],
        'y': [0., 1.], 'best_of': [5., 3.], 'p_a0': [.6, .3],
        'q_consensus': [.7, .2], 'fixture_id': [200, 100], 'book_count': [3, 2],
    })
    result = _market_view(joined, panel).set_index('golgg_match_id')
    assert result.q_consensus.to_dict() == {'2': .7, '1': .2}
    assert result.loc['1', 'team1_id'] == 'A'
    panel.loc[0, ['team1_id', 'team2_id']] = ['D', 'C']
    with pytest.raises(ValueError, match='identity'):
        _market_view(joined, panel)


def test_completed_evaluation_is_never_rewritten(tmp_path, monkeypatch):
    from src.analysis import a0_data_focus as analysis

    monkeypatch.setattr(analysis, 'PROJECT_ROOT', tmp_path)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text('{"run_id": "frozen"}')
    report = tmp_path / 'data/08_reporting/a0_data_focus/frozen'
    report.mkdir(parents=True)
    completed = report / 'evaluation.json'
    completed.write_text('{"status": "preserve"}')
    metrics = report / 'metrics.csv'
    metrics.write_text('preserve existing evidence\n')
    with pytest.raises(FileExistsError, match='immutable'):
        analysis.evaluate_experiment(protocol, research_root=tmp_path)
    assert completed.read_text() == '{"status": "preserve"}'
    assert metrics.read_text() == 'preserve existing evidence\n'
