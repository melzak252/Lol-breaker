"""Independent whole-block mathematics; no fits, source requests or database."""
import hashlib

import numpy as np
import pytest

from src.analysis.sports_reliability_common import paired_block_contrast


def expanded_draws(differences, labels, universe, *, mask=None, draws=29, seed=7):
    """Materialize sampled rows, rather than reuse production block sums."""
    differences = np.asarray(differences, dtype=float)
    labels = np.asarray(labels)
    selected = np.ones(len(labels), dtype=bool) if mask is None else np.asarray(mask)
    indices = np.random.default_rng(seed).integers(
        0, len(universe), size=(draws, len(universe)), dtype=np.int64)
    weighted = np.full(draws, np.nan)
    equal = np.full(draws, np.nan)
    for draw_id, sampled in enumerate(indices):
        sampled_rows = []
        occupied_means = []
        for block_index in sampled:
            values = differences[selected & (labels == universe[block_index])]
            sampled_rows.extend(values.tolist())
            if len(values):
                occupied_means.append(sum(values.tolist()) / len(values))
        if sampled_rows:
            weighted[draw_id] = sum(sampled_rows) / len(sampled_rows)
            equal[draw_id] = sum(occupied_means) / len(occupied_means)
    digest = hashlib.sha256(indices.tobytes(order='C')).hexdigest()
    return weighted, equal, digest


def test_whole_block_sampling_matches_expanded_rows_and_not_equal_block_weights():
    differences = np.array([0., 0., 0., 12.])
    labels = np.array(['Jan', 'Jan', 'Jan', 'Feb'])
    universe = ['Jan', 'Feb']
    weighted, equal, digest = expanded_draws(differences, labels, universe)
    result = paired_block_contrast(
        differences, labels, universe=universe, confidence=.8,
        min_rows=1, min_blocks=1, draws=29, seed=7)
    assert result['status'] == 'ESTIMABLE'
    assert result['reason'] is None
    assert result['point'] == 3.
    assert result['equal_block']['point'] == 6.
    assert result['n_rows'] == 4
    assert result['n_blocks'] == result['n_universe_blocks'] == 2
    assert result['undefined_draws'] == result['nonfinite_selected_rows'] == 0
    assert result['bootstrap_index_sha256'] == digest
    assert result['block_summary'] == [
        {'block': 'Jan', 'count': 3, 'sum': 0., 'mean_or_null': 0.},
        {'block': 'Feb', 'count': 1, 'sum': 12., 'mean_or_null': 12.},
    ]
    assert [result['lower'], result['upper']] == pytest.approx(
        np.quantile(weighted, [.1, .9], method='linear'))
    assert [result['equal_block']['lower'], result['equal_block']['upper']] == pytest.approx(
        np.quantile(equal, [.1, .9], method='linear'))
    assert result['confidence'] == .8
    assert result['one_sided'] is False
    assert (result['draws'], result['seed']) == (29, 7)
    assert (result['min_rows'], result['min_blocks']) == (1, 1)


def test_zero_subgroup_month_and_absent_month_remain_in_universe_and_null_intervals():
    differences = np.array([2., 4., 99.])
    labels = np.array(['Jan', 'Jan', 'Feb'])
    mask = np.array([True, True, False])
    universe = ['Jan', 'Feb', 'Mar']
    weighted, equal, digest = expanded_draws(
        differences, labels, universe, mask=mask, draws=64, seed=11)
    expected_undefined = int(np.isnan(weighted).sum())
    assert expected_undefined > 0
    result = paired_block_contrast(
        differences, labels, mask=mask, universe=universe,
        min_rows=1, min_blocks=1, draws=64, seed=11)
    assert result['status'] == 'NOT_ESTIMABLE'
    assert result['reason']
    assert result['undefined_draws'] == expected_undefined == int(np.isnan(equal).sum())
    assert result['bootstrap_index_sha256'] == digest
    assert (result['n_rows'], result['n_blocks'], result['n_universe_blocks']) == (2, 1, 3)
    assert result['point'] == result['equal_block']['point'] == 3.
    assert result['lower'] is result['upper'] is None
    assert result['equal_block']['lower'] is result['equal_block']['upper'] is None
    assert result['block_summary'][1:] == [
        {'block': 'Feb', 'count': 0, 'sum': 0., 'mean_or_null': None},
        {'block': 'Mar', 'count': 0, 'sum': 0., 'mean_or_null': None},
    ]


def test_default_universe_includes_unselected_rows_and_is_permutation_invariant():
    differences = np.array([2., -4., 8., 16., 32.])
    labels = np.array(['Mar', 'Jan', 'Feb', 'Jan', 'Mar'])
    mask = np.array([False, True, True, True, False])
    permutation = np.array([4, 2, 0, 3, 1])
    options = dict(min_rows=1, min_blocks=1, draws=64, seed=11)
    original = paired_block_contrast(differences, labels, mask=mask, **options)
    permuted = paired_block_contrast(
        differences[permutation], labels[permutation], mask=mask[permutation], **options)
    explicit = paired_block_contrast(
        differences, labels, mask=mask, universe=['Feb', 'Jan', 'Mar'], **options)
    assert original == permuted == explicit
    assert original['n_universe_blocks'] == 3
    assert original['n_blocks'] == 2
    assert original['block_summary'][-1] == {
        'block': 'Mar', 'count': 0, 'sum': 0., 'mean_or_null': None}


def test_one_sided_upper_uses_confidence_not_two_sided_tail_and_keeps_lower_null():
    differences = [0., 0., 0., 12.]
    labels = ['Jan', 'Jan', 'Jan', 'Feb']
    universe = ['Jan', 'Feb']
    weighted, equal, _ = expanded_draws(differences, labels, universe, draws=101)
    options = dict(universe=universe, confidence=.6,
                   min_rows=1, min_blocks=1, draws=101, seed=7)
    two = paired_block_contrast(differences, labels, **options)
    one = paired_block_contrast(differences, labels, one_sided=True, **options)
    assert one['status'] == two['status'] == 'ESTIMABLE'
    assert one['lower'] is one['equal_block']['lower'] is None
    assert one['upper'] == pytest.approx(np.quantile(weighted, .6, method='linear'))
    assert one['equal_block']['upper'] == pytest.approx(
        np.quantile(equal, .6, method='linear'))
    assert [two['lower'], two['upper']] == pytest.approx(
        np.quantile(weighted, [.2, .8], method='linear'))
    assert one['bootstrap_index_sha256'] == two['bootstrap_index_sha256']
    assert one['one_sided'] is True


def test_reversing_signed_contrast_reverses_points_and_two_sided_endpoints():
    differences = np.array([-4., 2., 8., 16.])
    labels = ['Jan', 'Jan', 'Feb', 'Mar']
    options = dict(universe=['Jan', 'Feb', 'Mar'], confidence=.8,
                   min_rows=1, min_blocks=1, draws=101, seed=7)
    forward = paired_block_contrast(differences, labels, **options)
    reverse = paired_block_contrast(-differences, labels, **options)
    for positive, negative in ((forward, reverse),
                               (forward['equal_block'], reverse['equal_block'])):
        assert negative['point'] == pytest.approx(-positive['point'])
        assert negative['lower'] == pytest.approx(-positive['upper'])
        assert negative['upper'] == pytest.approx(-positive['lower'])
    assert forward['bootstrap_index_sha256'] == reverse['bootstrap_index_sha256']


@pytest.mark.parametrize('min_rows,min_blocks', [(5, 1), (1, 3)])
def test_insufficient_series_or_occupied_blocks_cannot_be_estimable(min_rows, min_blocks):
    result = paired_block_contrast(
        [-10., -10., -10., -10.], ['Jan', 'Jan', 'Jan', 'Feb'],
        universe=['Jan', 'Feb'], min_rows=min_rows, min_blocks=min_blocks,
        draws=29, seed=7)
    assert result['status'] == 'INSUFFICIENT_SUPPORT'
    assert result['reason']
    assert result['point'] == result['equal_block']['point'] == -10.
    assert result['undefined_draws'] == 0
    assert (result['n_rows'], result['n_blocks']) == (4, 2)


def test_empty_selection_retains_population_blocks_and_reports_all_draws_undefined():
    result = paired_block_contrast(
        [1., 2.], ['Jan', 'Feb'], mask=[False, False],
        min_rows=1, min_blocks=1, draws=29, seed=7)
    assert result['status'] == 'NOT_ESTIMABLE'
    assert result['reason']
    assert (result['n_rows'], result['n_blocks'], result['n_universe_blocks']) == (0, 0, 2)
    assert result['undefined_draws'] == 29
    assert result['point'] is result['lower'] is result['upper'] is None
    assert all(result['equal_block'][key] is None for key in ('point', 'lower', 'upper'))


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_selected_nonfinite_loss_is_reported_not_dropped_or_clipped(invalid):
    result = paired_block_contrast(
        [2., invalid], ['Jan', 'Feb'], min_rows=1, min_blocks=1, draws=29, seed=7)
    assert result['status'] == 'NOT_ESTIMABLE'
    assert result['reason']
    assert result['n_rows'] == 2
    assert result['nonfinite_selected_rows'] == 1
    assert result['point'] is result['lower'] is result['upper'] is None
    assert all(result['equal_block'][key] is None for key in ('point', 'lower', 'upper'))


def test_unselected_nonfinite_loss_does_not_poison_a_supported_selected_contrast():
    differences = np.array([2., 4., np.inf])
    labels = np.array(['Jan', 'Feb', 'Feb'])
    mask = np.array([True, True, False])
    snapshots = tuple(value.copy() for value in (differences, labels, mask))
    result = paired_block_contrast(
        differences, labels, mask=mask, min_rows=2, min_blocks=2, draws=29, seed=7)
    assert result['status'] == 'ESTIMABLE'
    assert result['nonfinite_selected_rows'] == 0
    assert result['n_rows'] == 2
    assert result['point'] == result['equal_block']['point'] == 3.
    for original, snapshot in zip((differences, labels, mask), snapshots):
        np.testing.assert_array_equal(original, snapshot)
