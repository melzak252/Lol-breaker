"""Frozen advancement gates must not promote a zero-crossing or degraded law."""
import copy
import pytest

from src.models.draft_prop_runner import stage2_gate


def _evidence():
    metrics = {}
    for candidate, crps, coverage in [('existing_selected', 5., .80),
                                      ('matched_no_draft', 4.9, .79),
                                      ('actual_composition_oracle', 4.8, .80)]:
        metrics[candidate] = {task: {'n': 3103, 'crps': crps, 'central_80_coverage': coverage}
                              for task in ('kills', 'duration')}
    comparisons = {f'{task}_oracle_minus_{reference}': {'n': 3103, 'ci': [-.01, -.001]}
                   for task in ('kills', 'duration')
                   for reference in ('matched_no_draft', 'existing_selected')}
    return metrics, comparisons


@pytest.mark.parametrize('upper', [0., .001, float('nan')])
def test_one_nonnegative_or_unknown_upper_bound_stops_all_later_modeling(upper):
    metrics, comparisons = _evidence()
    comparisons['duration_oracle_minus_matched_no_draft']['ci'][1] = upper
    gate = stage2_gate(metrics, comparisons)
    assert gate['status'] == 'FAIL'
    assert gate['next_stage'] == 'NOT_RUN_BY_FROZEN_GATE'


def test_nll_gain_does_not_override_worse_crps_or_interval_coverage():
    metrics, comparisons = _evidence()
    metrics['actual_composition_oracle']['kills']['crps'] = 5.1
    metrics['actual_composition_oracle']['duration']['central_80_coverage'] = .75
    gate = stage2_gate(metrics, comparisons)
    assert gate['status'] == 'FAIL'
    failures = [row for row in gate['checks'] if not row['pass']]
    assert any(row['guard'] == 'CRPS_nonincrease' for row in failures)
    assert any(row['guard'] == '80pct_coverage_deviation_noninferiority' for row in failures)


def test_all_declared_improvements_advance_without_adopting_or_mutating_evidence():
    metrics, comparisons = _evidence()
    original = copy.deepcopy((metrics, comparisons))
    gate = stage2_gate(metrics, comparisons)
    assert gate['status'] == 'PASS'
    assert gate['next_stage'] == 'AUTHORIZED_BY_FROZEN_GATE'
    assert (metrics, comparisons) == original
