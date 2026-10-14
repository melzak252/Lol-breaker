"""Proper scores, complete outcomes, and event rather than entrant weighting."""
from copy import deepcopy
from math import log

import numpy as np
import pandas as pd
import pytest

from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from scripts.score_tournament_forecasts import score_event, summarize_scores


def _event():
    bracket = TournamentBracket(
        id='event', name='Event', region='test', format='single_elimination',
        teams=['A', 'B'], matches={'final': BracketMatchNode(
            id='final', name='Final', round_name='Final', bracket_section='final',
            team1='A', team2='B', best_of=3)},
    )
    forecast = {'champion_prob': {'A': 0.75, 'B': 0.25},
                'final_prob': {'A': 1.0, 'B': 1.0},
                'node_reach_prob': {'final': {'A': 1.0, 'B': 1.0}},
                'championship_node_id': 'final', 'probability_unit': 'series_win'}
    results = {'tournament_id': 'event', 'matches': {'final': {
        'team1': 'A', 'team2': 'B', 'winner': 'A', 'score1': 2, 'score2': 1}}}
    return bracket, forecast, results


def test_champion_scores_use_categorical_not_binary_log_loss():
    scores, calibration = score_event(*_event())
    assert scores['champion_log_loss'] == pytest.approx(-log(0.75))
    assert scores['champion_brier'] == pytest.approx(0.125)
    assert scores['final_brier'] == 0
    assert scores['node_reach_brier'] == 0
    assert len(calibration) == 6


@pytest.mark.parametrize('fault', ['mass', 'missing_team', 'nan', 'missing_outcome',
                                  'wrong_winner', 'wrong_score', 'wrong_final'])
def test_invalid_forecast_or_outcome_is_rejected_instead_of_dropped(fault):
    bracket, forecast, results = _event()
    if fault == 'mass':
        forecast['champion_prob']['A'] = 0.9
    elif fault == 'missing_team':
        del forecast['champion_prob']['B']
    elif fault == 'nan':
        forecast['champion_prob']['A'] = np.nan
    elif fault == 'missing_outcome':
        results['matches']['final']['winner'] = None
    elif fault == 'wrong_winner':
        results['matches']['final']['winner'] = 'C'
    elif fault == 'wrong_score':
        results['matches']['final']['score1'] = 1
    else:
        forecast['final_prob'] = {'A': 0.5, 'B': 0.5}
    with pytest.raises(ValueError):
        score_event(bracket, forecast, results)


def test_zero_probability_of_observed_champion_has_infinite_log_score():
    bracket, forecast, results = _event()
    forecast['champion_prob'] = {'A': 0, 'B': 1}
    scores, _ = score_event(bracket, forecast, results)
    assert scores['champion_log_loss'] == float('inf')
    assert scores['champion_brier'] == 2


def test_event_bootstrap_and_mean_do_not_weight_large_tournaments_more():
    scores = pd.DataFrame([
        {'tournament_id': 'small', 'model': 'sports', 'entrant_count': 2,
         'champion_brier': 0.1, 'champion_log_loss': 0.2},
        {'tournament_id': 'large', 'model': 'sports', 'entrant_count': 32,
         'champion_brier': 0.9, 'champion_log_loss': 1.8},
    ])
    summary, _ = summarize_scores(scores, replicates=5000, seed=82)
    brier = summary[(summary.model == 'sports') & (summary.metric == 'champion_brier')].iloc[0]
    assert brier['mean'] == pytest.approx(0.5)
    assert brier['ci_low'] == pytest.approx(0.1)
    assert brier['ci_high'] == pytest.approx(0.9)
    assert brier['events'] == 2


def test_models_must_cover_identical_events_for_paired_comparison():
    frame = pd.DataFrame([
        {'tournament_id': 'one', 'model': 'a', 'champion_brier': 0.2},
        {'tournament_id': 'two', 'model': 'b', 'champion_brier': 0.1},
    ])
    with pytest.raises(ValueError, match='same events'):
        summarize_scores(frame)


def test_supplied_placement_distribution_is_scored_and_must_match_champion():
    bracket, forecast, results = _event()
    forecast['placement_prob'] = {'A': {'1': 0.75, '2': 0.25},
                                  'B': {'1': 0.25, '2': 0.75}}
    results['placements'] = {'A': 1, 'B': 2}
    scores, _ = score_event(bracket, forecast, results)
    assert scores['placement_rps'] == pytest.approx(0.0625)
    assert scores['placement_log_loss'] == pytest.approx(-log(0.75))
    bad = deepcopy(forecast)
    bad['placement_prob']['A'] = {'1': 0.5, '2': 0.5}
    with pytest.raises(ValueError, match='placement'):
        score_event(bracket, bad, results)


def test_later_opponents_cannot_override_the_supplied_graph():
    bracket, forecast, results = _event()
    bracket.teams += ['C', 'D']
    bracket.matches['final'].team1 = None
    bracket.matches['final'].team2 = None
    for node_id, first, second, slot in [('left', 'A', 'B', 1), ('right', 'C', 'D', 2)]:
        bracket.matches[node_id] = BracketMatchNode(
            id=node_id, name=node_id, round_name='Semifinal', bracket_section='upper',
            team1=first, team2=second, best_of=3,
            next_match_winner_id='final', next_match_winner_slot=slot)
        results['matches'][node_id] = {'team1': first, 'team2': second,
                                      'winner': first, 'score1': 2, 'score2': 0}
    # The archived final claims B, but the supplied graph routes C from right.
    with pytest.raises(ValueError, match='participants disagree'):
        score_event(bracket, forecast, results)


def test_finalist_probability_mass_cannot_cross_semifinal_slots():
    bracket, forecast, results = _event()
    bracket.teams += ['C', 'D']
    bracket.matches['final'].team1 = None
    bracket.matches['final'].team2 = None
    for node_id, first, second, slot in [('left', 'A', 'B', 1), ('right', 'C', 'D', 2)]:
        bracket.matches[node_id] = BracketMatchNode(
            id=node_id, name=node_id, round_name='Semifinal', bracket_section='upper',
            team1=first, team2=second, best_of=3,
            next_match_winner_id='final', next_match_winner_slot=slot)
        results['matches'][node_id] = {'team1': first, 'team2': second,
                                      'winner': first, 'score1': 2, 'score2': 0}
        forecast['node_reach_prob'][node_id] = {
            team: float(team in (first, second)) for team in bracket.teams}
    results['matches']['final']['team2'] = 'C'
    forecast['final_prob'] = {'A': 0.9, 'B': 0.9, 'C': 0.1, 'D': 0.1}
    forecast['node_reach_prob']['final'] = forecast['final_prob'].copy()
    forecast['champion_prob'] = {'A': 0.45, 'B': 0.45, 'C': 0.05, 'D': 0.05}
    with pytest.raises(ValueError, match='rout'):
        score_event(bracket, forecast, results)


def test_optional_placements_use_only_explicitly_available_common_events():
    bracket, forecast, results = _event()
    unavailable, _ = score_event(bracket, forecast, results)
    forecast['placement_prob'] = {'A': {'1': 0.75, '2': 0.25}, 'B': {'1': 0.25, '2': 0.75}}
    results['placements'] = {'A': 1, 'B': 2}
    available, _ = score_event(bracket, forecast, results)
    frame = pd.DataFrame([
        {**values, 'tournament_id': event, 'model': model}
        for event, values in (('ordinary', unavailable), ('untied', available))
        for model in ('a', 'b')
    ])
    summary, _ = summarize_scores(frame)
    champion = summary[(summary.model == 'a') & (summary.metric == 'champion_brier')].iloc[0]
    placement = summary[(summary.model == 'a') & (summary.metric == 'placement_brier')].iloc[0]
    assert champion['events'] == 2
    assert placement['events'] == 1
    assert placement['input_events'] == 2
    assert placement['excluded_unavailable_events'] == 1
    assert placement['mean'] == pytest.approx(0.125)
    assert pd.isna(placement['ci_low'])
    # One model cannot selectively omit an otherwise scoreable event.
    frame.loc[(frame.model == 'b') & (frame.tournament_id == 'untied'), 'placement_brier'] = np.nan
    with pytest.raises(ValueError, match='incomplete'):
        summarize_scores(frame)


def _ledger_row(*, model='exp081', tournament='edition', origin='start',
                target='champion', kind='champion', probabilities=None, **extra):
    return {
        'tournament_id': tournament, 'origin_id': origin, 'target_id': target,
        'model': model, 'phase_id': 'playoff', 'axis': 'axis1',
        'edition_start_at': '2026-01-10T12:00:00Z',
        'cutoff': '2026-01-10T10:00:00Z', 'target_kind': kind,
        'probabilities': probabilities if probabilities is not None else {'A': .75, 'B': .25},
        'resolved': False, 'format': 'single_elimination', 'family': 'test',
        'tier': 'Major', 'best_of': 3, 'forecast_mode': 'pre_draw',
        'observation_day_end': None, **extra,
    }


def _ledger_outcome(row, observed):
    return {key: row[key] for key in ('tournament_id', 'phase_id', 'origin_id', 'target_id')} | {
        'observed': observed}


def _evaluate(rows, outcomes, **kwargs):
    from scripts.score_tournament_forecasts import evaluate_ledger
    return evaluate_ledger(rows, outcomes, bootstrap=20, **kwargs)


def _summary(result, metric, **filters):
    return next(row for row in result['summaries']
                if row['metric'] == metric and all(row[key] == value for key, value in filters.items()))


def test_ledger_proper_scores_preserve_actual_finalists_and_joint_targets():
    champion = _ledger_row()
    finalist = _ledger_row(target='finalist:B', kind='binary', target_family='finalist',
                          probabilities={'no': .2, 'yes': .8}, positive_category='yes')
    qualification = _ledger_row(target='qualify:B:worlds', kind='binary',
                               target_family='qualification', probabilities={'no': .7, 'yes': .3})
    joint = _ledger_row(target='finalist_pair', kind='joint',
                       probabilities={'A|B': .6, 'A|C': .4})
    rows = [champion, finalist, qualification, joint]
    result = _evaluate(rows, [_ledger_outcome(row, observed)
                              for row, observed in zip(rows, ['A', 'yes', 'no', 'A|B'])])
    assert _summary(result, 'brier', target_kind='champion')['mean'] == pytest.approx(.125)
    assert _summary(result, 'log_loss', target_kind='champion')['mean'] == pytest.approx(-log(.75))
    assert _summary(result, 'brier', target_family='finalist')['mean'] == pytest.approx(.04)
    assert _summary(result, 'brier', target_family='qualification')['mean'] == pytest.approx(.09)
    assert _summary(result, 'brier', target_kind='joint')['mean'] == pytest.approx(.32)
    assert _summary(result, 'log_loss', target_kind='joint')['mean'] == pytest.approx(-log(.6))


def test_ledger_placement_rps_preserves_official_ties_and_declared_order():
    rows = [_ledger_row(target=f'placement:{team}', kind='placement',
                        probabilities={'3-4': .5, '1': .2, '2': .3},
                        categories=['1', '2', '3-4']) for team in ('A', 'B')]
    result = _evaluate(rows, [_ledger_outcome(row, '3-4') for row in rows])
    assert _summary(result, 'rps')['mean'] == pytest.approx((.2**2 + .5**2) / 2)
    assert _summary(result, 'brier')['mean'] == pytest.approx(.38)
    assert _summary(result, 'log_loss')['mean'] == pytest.approx(-log(.5))
    assert result['coverage']['observed_targets'] == 2
    assert {row['observed'] for row in result['target_scores']} == {'3-4'}


@pytest.mark.parametrize('fault', ['duplicate_forecast', 'duplicate_outcome',
                                  'extra_outcome', 'missing_model_target', 'resolved_mismatch',
                                  'cutoff_mismatch', 'category_mismatch', 'phase_mismatch'])
def test_ledger_rejects_nonunique_or_mismatched_target_cohorts(fault):
    rows = [_ledger_row(model=model) for model in ('a', 'b')]
    outcomes = [_ledger_outcome(rows[0], 'A')]
    if fault == 'duplicate_forecast':
        rows.append(deepcopy(rows[0]))
    elif fault == 'duplicate_outcome':
        outcomes.append(deepcopy(outcomes[0]))
    elif fault == 'extra_outcome':
        outcomes.append({**outcomes[0], 'target_id': 'missing'})
    elif fault == 'missing_model_target':
        rows.append(_ledger_row(model='a', target='other'))
    elif fault == 'resolved_mismatch':
        rows[1]['resolved'] = True
    elif fault == 'cutoff_mismatch':
        rows[1]['cutoff'] = '2026-01-10T09:00:00Z'
    elif fault == 'category_mismatch':
        rows[1]['probabilities'] = {'A': .75, 'C': .25}
    else:
        rows[1]['phase_id'] = 'other'
    with pytest.raises(ValueError):
        _evaluate(rows, outcomes)


@pytest.mark.parametrize('probabilities', [
    {'A': .5}, {'A': np.nan, 'B': .25}, {'A': np.inf, 'B': 0},
    {'A': -.1, 'B': 1.1}, {'A': .5, 'B': .4}, {'A': True, 'B': False},
])
def test_ledger_invalid_probabilities_are_errors_not_successful_subset(probabilities):
    row = _ledger_row(probabilities=probabilities)
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, 'B')])


def test_ledger_pending_outcomes_are_not_forecast_failures():
    row = _ledger_row()
    result = _evaluate([row], [])
    assert result['coverage']['pending_targets'] == 1
    assert result['coverage']['forecast_failed_rows'] == 0
    assert result['target_scores'][0]['status'] == 'pending_outcome'
    assert result['verdict']['status'] == 'inconclusive'
    assert result['summaries'][0]['mean'] is None
    failed = {**row, 'probabilities': None, 'forecast_status': 'failed',
              'failure_reason': 'missing timestamped feature state'}
    failure = _evaluate([failed], [])
    assert failure['coverage']['forecast_failed_rows'] == 1
    assert failure['target_scores'][0]['status'] == 'forecast_failed'


def test_ledger_failure_blocks_comparison_without_dropping_the_target():
    rows = [_ledger_row(model=model, target=target)
            for model in ('a', 'b') for target in ('one', 'two')]
    rows[-1].update(forecast_status='failed', failure_reason='unavailable feature',
                    probabilities=None)
    outcomes = [_ledger_outcome(row, 'A') for row in rows[:2]]
    result = _evaluate(rows, outcomes)
    for summary in result['summaries']:
        assert summary['counts']['eligible_targets'] == 2
        assert summary['counts']['scored_targets'] == 1
        assert summary['mean'] is None
    assert all(row['delta'] is None for row in result['comparisons'])


def test_ledger_resolved_targets_are_excluded_for_every_model():
    rows = [_ledger_row(model=model, target=target, resolved=target == 'decided')
            for model in ('a', 'b') for target in ('future', 'decided')]
    outcomes = [_ledger_outcome(row, 'A') for row in rows[:2]]
    result = _evaluate(rows, outcomes)
    assert result['coverage']['resolved_targets'] == 1
    for summary in result['summaries']:
        assert summary['counts']['eligible_targets'] == 1
        assert summary['counts']['resolved_targets'] == 1
        assert summary['counts']['scored_targets'] == 1


def test_ledger_zero_probability_is_infinite_but_mc_zero_is_unresolved():
    import json
    exact = _ledger_row(probabilities={'A': 0., 'B': 1.}, probability_source='exact')
    result = _evaluate([exact], [_ledger_outcome(exact, 'A')])
    assert _summary(result, 'log_loss')['mean'] == 'Infinity'
    assert _summary(result, 'log_loss')['counts']['infinite_targets'] == 1
    assert _summary(result, 'brier')['mean'] == 2.
    json.dumps(result, allow_nan=False)
    mc = {**exact, 'probability_source': 'monte_carlo', 'mc_samples': 100,
          'mc_hits': {'A': 0, 'B': 100}, 'mc_precision_resolved': True}
    unresolved = _evaluate([mc], [_ledger_outcome(mc, 'A')])
    assert _summary(unresolved, 'log_loss')['mean'] is None
    assert _summary(unresolved, 'log_loss')['counts']['unresolved_targets'] == 1
    assert _summary(unresolved, 'brier')['mean'] == 2.
    json.dumps(unresolved, allow_nan=False)


def test_ledger_whole_edition_bootstrap_and_origin_weighting():
    rows, outcomes = [], []
    for edition, month, origins, probability in [('small', 1, 1, .9), ('large', 2, 4, .1)]:
        for origin in range(origins):
            for model, p in [('a', probability), ('b', .5)]:
                row = _ledger_row(model=model, tournament=edition, origin=str(origin),
                                  edition_start_at=f'2026-{month:02d}-10T12:00:00Z',
                                  cutoff=f'2026-{month:02d}-10T10:00:00Z',
                                  probabilities={'A': p, 'B': 1-p})
                rows.append(row)
                if model == 'a':
                    outcomes.append(_ledger_outcome(row, 'A'))
    result = _evaluate(rows, outcomes)
    summary = _summary(result, 'brier', model='a')
    assert summary['mean'] == pytest.approx((.02 + 1.62) / 2)
    assert summary['counts']['editions'] == 2
    assert summary['counts']['origins'] == 5
    assert result['protocol']['bootstrap']['blocks'] == [
        {'month': '2026-01', 'editions': ['small']},
        {'month': '2026-02', 'editions': ['large']}]
    comparison = next(row for row in result['comparisons'] if row['metric'] == 'brier')
    assert comparison['delta'] == pytest.approx(.32)
    assert comparison['inference_status'] == 'inconclusive_sparse_blocks'
    assert comparison['ci_low'] is None


def test_series_accuracy_ties_auc_and_brier_decomposition_are_observable():
    rows, outcomes = [], []
    for index, (probability, observed) in enumerate([(.5, 'yes'), (.8, 'yes'), (.2, 'no')]):
        row = _ledger_row(target=f'series:{index}', kind='series', axis='axis2',
                          probabilities={'no': 1-probability, 'yes': probability},
                          positive_category='yes')
        rows.append(row)
        outcomes.append(_ledger_outcome(row, observed))
    result = _evaluate(rows, outcomes)
    series = result['series_diagnostics'][0]
    assert series['accuracy'] == pytest.approx(5/6)
    assert series['auc'] == 1.
    assert series['brier'] == pytest.approx(.11)
    decomposition = series['brier_decomposition']
    assert decomposition['uncertainty'] - decomposition['resolution'] + decomposition['reliability'] + decomposition['within_bin_residual'] == pytest.approx(.11)
    calibration = result['calibration'][0]
    assert calibration['ece_10bin'] == pytest.approx(.3)
    assert calibration['mce_10bin'] == pytest.approx(.5)
    assert len(calibration['bins']) == 10
    assert calibration['regression']['status'] != 'estimated'


def test_diagnostics_preserve_composite_format_membership_and_missing_horizons():
    row = _ledger_row(format=['round_robin', 'single_elimination'], best_of='mixed',
                      forecast_scope='phase_start')
    result = _evaluate([row], [_ledger_outcome(row, 'A')])
    formats = {cell['value'] for cell in result['diagnostics'] if cell['dimension'] == 'format'}
    scopes = {cell['value'] for cell in result['diagnostics'] if cell['dimension'] == 'forecast_scope'}
    assert scopes == {'phase_start'}
    assert formats == {'round_robin', 'single_elimination'}
    horizon = [cell for cell in result['diagnostics'] if cell['dimension'] == 'horizon']
    assert horizon[0]['value'] == 'unavailable'
    assert result['verdict']['production_qualified'] is False


def test_ledger_axis_specific_models_and_joint_specific_cohorts_are_not_mismatches():
    rows = [_ledger_row(model=model) for model in ('exp081', 'fair_series', 'flat')]
    rows.extend(_ledger_row(model=model, origin='checkpoint', axis='axis3')
                for model in ('exp081:frozen', 'exp081:refreshed', 'fair_series', 'flat'))
    rows.extend(_ledger_row(model=model, target='final_pair', kind='joint',
                            probabilities={'A|B': .8, 'A|C': .2})
                for model in ('exp081', 'fair_series'))
    outcomes = [_ledger_outcome(rows[0], 'A'), _ledger_outcome(rows[3], 'A'),
                _ledger_outcome(rows[-1], 'A|B')]
    result = _evaluate(rows, outcomes)
    joint = [row for row in result['comparisons'] if row['target_kind'] == 'joint']
    assert {row['reference'] for row in joint} == {'fair_series'}
    rolling = [row for row in result['comparisons'] if row['axis'] == 'axis3']
    assert any(row['model'] == 'exp081:frozen' and row['reference'] == 'exp081:refreshed'
               for row in rolling)


def test_mc_precision_failure_blocks_all_scores_without_altering_denominators():
    row = _ledger_row(probability_source='monte_carlo', mc_samples=4,
                      mc_hits={'A': 3, 'B': 1}, mc_precision_resolved=False)
    result = _evaluate([row], [_ledger_outcome(row, 'A')])
    for summary in result['summaries']:
        assert summary['mean'] is None
        assert summary['counts']['eligible_targets'] == 1
        assert summary['counts']['unresolved_targets'] == 1
    assert result['calibration'][0]['included_targets'] == 0
    assert result['calibration'][0]['exclusions']['matched_cohort_unscorable_brier'] == 1


def test_both_infinite_models_never_produce_a_favorable_or_zero_paired_delta():
    rows = [_ledger_row(model=model, probabilities={'A': 0., 'B': 1.},
                        probability_source='exact') for model in ('a', 'b')]
    result = _evaluate(rows, [_ledger_outcome(rows[0], 'A')])
    comparison = next(row for row in result['comparisons'] if row['metric'] == 'log_loss')
    assert comparison['model_mean'] == comparison['reference_mean'] == 'Infinity'
    assert comparison['delta'] is None
    assert comparison['paired_scored_targets'] == 1
    assert comparison['paired_finite_targets'] == 0
    assert comparison['probability_non_improvement'] is None


def test_supported_calibration_and_identical_paired_month_blocks_have_known_values():
    from scripts.score_tournament_forecasts import evaluate_ledger
    rows, outcomes = [], []
    for month in range(1, 11):
        for index, (probability, observed) in enumerate(
                [(.25, 'yes'), (.25, 'no'), (.25, 'no'), (.25, 'no'),
                 (.75, 'yes'), (.75, 'yes'), (.75, 'yes'), (.75, 'no')]):
            for model in ('a', 'b'):
                row = _ledger_row(
                    model=model, tournament=f'edition-{month:02d}', target=f'series:{index}',
                    kind='series', axis='axis2',
                    edition_start_at=f'2026-{month:02d}-10T12:00:00Z',
                    cutoff=f'2026-{month:02d}-10T10:00:00Z',
                    probabilities={'no': 1-probability, 'yes': probability},
                    positive_category='yes')
                rows.append(row)
                if model == 'a':
                    outcomes.append(_ledger_outcome(row, observed))
    result = evaluate_ledger(rows, outcomes)
    calibration = result['calibration'][0]
    assert calibration['ece_10bin'] == pytest.approx(0.)
    assert calibration['regression']['status'] == 'estimated'
    assert calibration['regression']['intercept'] == pytest.approx(0., abs=1e-7)
    assert calibration['regression']['slope'] == pytest.approx(1., abs=1e-7)
    assert _summary(result, 'brier', model='a')['mean'] == pytest.approx(.1875)
    for comparison in result['comparisons']:
        assert comparison['delta'] == comparison['ci_low'] == comparison['ci_high'] == 0.
        assert comparison['probability_non_improvement'] == 1.
        assert comparison['paired_eligible_targets'] == 80
    assert result['verdict']['status'] == 'inconclusive'

def test_daily_rollforward_rows_are_reported_by_mode_and_horizon_without_pooling():
    daily = _ledger_row(model='daily', tournament='daily-edition', origin='daily-origin',
                        cutoff='2026-01-11T00:00:00Z', forecast_mode='daily_rollforward',
                        observation_day_end='2026-01-10', horizon='next_day')
    pre_draw = _ledger_row(model='pre', tournament='pre-edition', origin='pre-origin',
                          forecast_mode='pre_draw', observation_day_end=None,
                          horizon='next_day')
    result = _evaluate([daily, pre_draw],
                       [_ledger_outcome(daily, 'A'), _ledger_outcome(pre_draw, 'A')])
    breakdown = result['mode_horizon_breakdown']
    daily_cell = next(cell for cell in breakdown
                      if cell['forecast_mode'] == 'daily_rollforward'
                      and cell['horizon'] == 'next_day')
    pre_draw_cell = next(cell for cell in breakdown
                         if cell['forecast_mode'] == 'pre_draw'
                         and cell['horizon'] == 'next_day')
    assert daily_cell['summaries'][0]['counts']['forecast_rows'] == 1
    assert pre_draw_cell['summaries'][0]['counts']['forecast_rows'] == 1
    assert daily_cell['comparisons'] == pre_draw_cell['comparisons'] == []


@pytest.mark.parametrize('changes', [
    {'forecast_mode': 'daily_rollforward', 'cutoff': '2026-01-11T01:00:00Z',
     'observation_day_end': '2026-01-10'},
    {'forecast_mode': 'daily_rollforward', 'cutoff': '2026-01-11T00:00:00Z',
     'observation_day_end': '2026-01-11'},
    {'forecast_mode': 'pre_draw', 'observation_day_end': '2026-01-10'},
])
def test_ledger_rejects_invalid_mode_observation_day_semantics(changes):
    row = _ledger_row(**changes)
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, 'A')])


@pytest.mark.parametrize('field', ['forecast_mode', 'observation_day_end'])
def test_ledger_requires_forecast_mode_and_observation_day_end(field):
    row = _ledger_row()
    row.pop(field)
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, 'A')])


def test_phase_identity_prevents_collisions_without_inflating_edition_count():
    first = _ledger_row(phase_id='groups')
    second = _ledger_row(phase_id='playoffs')
    outcomes = [{**_ledger_outcome(row, 'A'), 'phase_id': row['phase_id']}
                for row in (first, second)]
    report = _evaluate([first, second], outcomes)
    assert len(report['target_scores']) == 2
    assert report['coverage']['editions'] == 1


def test_rare_positive_hits_do_not_certify_log_probability_precision():
    from scripts.score_tournament_forecasts import monte_carlo_precision
    report = monte_carlo_precision({'rare': .0001, 'common': .9999}, 200000,
                                   looks=5, cells=2)
    assert report['resolved'] is False


def test_phase_origins_balance_targets_before_edition_aggregation():
    rows = [_ledger_row(phase_id='groups', target=target, probabilities={'A': .9, 'B': .1})
            for target in ('first', 'second')]
    rows.append(_ledger_row(phase_id='playoffs', probabilities={'A': .1, 'B': .9},
                            cutoff='2026-01-12T10:00:00Z'))
    report = _evaluate(rows, [_ledger_outcome(row, 'A') for row in rows])
    summary = _summary(report, 'brier')
    assert summary['mean'] == pytest.approx((.02 + 1.62) / 2)
    assert summary['counts']['origins'] == report['coverage']['origins'] == 2
    assert summary['counts']['editions'] == 1
    assert len([row for row in report['origin_scores'] if row['metric'] == 'brier']) == 2


def test_generic_categories_and_count_rps_keep_complete_support_and_distinct_scores():
    count = _ledger_row(kind='count', target='future_maps', categories=['0', '1', '2'],
                        probabilities={'0': .2, '1': .3, '2': .5})
    categorical = _ledger_row(kind='categorical', target='node_score',
                              probabilities={'2-0': .2, '2-1': .3, 'absent': .5})
    result = _evaluate([count, categorical],
                       [_ledger_outcome(count, '1'), _ledger_outcome(categorical, '2-1')])
    assert _summary(result, 'count_rps')['mean'] == pytest.approx((.2**2 + .5**2)/2)
    for kind in ('count', 'categorical'):
        assert _summary(result, 'brier', target_kind=kind)['mean'] == pytest.approx(.78)
        assert _summary(result, 'log_loss', target_kind=kind)['mean'] == pytest.approx(-log(.3))
    assert all(row['metric'] != 'rps' for row in result['summaries'])
    assert {row['category'] for row in next(
        row for row in result['distribution_diagnostics'] if row['target_kind'] == 'count'
    )['category_frequencies']} == {'0', '1', '2'}


@pytest.mark.parametrize('categories', [None, ['0', '2', '1'], ['0', '1'], ['0', 'one', '2']])
def test_count_rejects_missing_non_numeric_or_unordered_support(categories):
    row = _ledger_row(kind='count', categories=categories,
                      probabilities={'0': .2, '1': .3, '2': .5})
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, '1')])


def test_predictive_sets_compare_actual_mass_and_use_fixed_ties_without_ordinal_team_pit():
    row = _ledger_row(probabilities={'C': .2, 'B': .4, 'A': .4})
    result = _evaluate([row], [_ledger_outcome(row, 'B')])
    diagnostic = result['distribution_diagnostics'][0]
    half = diagnostic['predictive_sets'][0]
    assert half['nominal_level'] == .5
    assert half['actual_included_mass'] == pytest.approx(.8)
    assert half['coverage'] == 1
    assert half['mean_set_size'] == 2
    assert diagnostic['predictive_set_rows'][0]['included_categories'] == ['A', 'B']
    assert half['coverage_minus_mass_interval']['ci_low'] is None
    assert diagnostic['pit']['rows'] == []
    assert diagnostic['editions'] == 1


def test_ordered_pit_is_seeded_order_independent_and_paired_without_phase_pseudoreplication():
    rows = [_ledger_row(model=model, phase_id=phase, kind='count',
                        categories=['0', '1', '2'], probabilities={'0': .2, '1': .3, '2': .5})
            for phase in ('groups', 'playoffs') for model in ('a', 'b')]
    outcomes = [_ledger_outcome(row, '1') for row in rows if row['model'] == 'a']
    result = _evaluate(rows, outcomes, seed=7)
    reversed_result = _evaluate(list(reversed(rows)), list(reversed(outcomes)), seed=7)
    first, second = result['distribution_diagnostics']
    pits = first['pit']['rows']
    assert pits == second['pit']['rows']
    assert pits == reversed_result['distribution_diagnostics'][0]['pit']['rows']
    assert all(.2 <= row['pit'] < .5 for row in pits)
    assert pits[0]['pit'] != pits[1]['pit']
    assert first['editions'] == 1 and first['origins'] == 2
    assert all(row['interval']['ci_low'] is None for row in first['pit']['bins'])


def test_mc_bounds_account_for_unsampled_support_structural_proof_and_planned_looks():
    from scripts.score_tournament_forecasts import monte_carlo_precision
    possible = monte_carlo_precision({'rare': 0., 'common': 1.}, 40000, cells=2, looks=5)
    interval = possible['intervals']['rare']
    assert interval['high'] == pytest.approx(1-(.05/(2*2*5))**(1/40000))
    assert interval['log_probability_half_width'] is None
    assert possible['unsampled_categories'] == ['rare']
    assert possible['log_resolved'] is False
    proven = monte_carlo_precision({'rare': 0., 'common': 1.}, 40000,
                                   structural_zeros=['rare'], cells=2, looks=5)
    assert proven['resolved'] is True
    rare_hits = monte_carlo_precision({'rare': .0001, 'common': .9999}, 200000, cells=2, looks=5)
    assert rare_hits['marginal_resolved'] is True
    assert rare_hits['log_resolved'] is False
    uncorrected = monte_carlo_precision({'a': .5, 'b': .5}, 40000)
    corrected = monte_carlo_precision({'a': .5, 'b': .5}, 40000, cells=20, looks=5)
    assert corrected['intervals']['a']['absolute_half_width'] > uncorrected['intervals']['a']['absolute_half_width']


def test_observed_structural_zero_is_infinite_only_with_explicit_evidence():
    row = _ledger_row(probabilities={'A': 0., 'B': 1.}, probability_source='monte_carlo',
                      mc_hits={'A': 0, 'B': 40000}, mc_samples=40000, mc_precision_resolved=True,
                      structural_zero_categories=['A'])
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, 'A')])
    row['structural_zero_evidence'] = {'A': 'Replay graph proves A eliminated before this origin'}
    result = _evaluate([row], [_ledger_outcome(row, 'A')])
    assert _summary(result, 'log_loss')['mean'] == 'Infinity'
    assert _summary(result, 'brier')['mean'] == 2.


def test_seed_precision_budget_uses_numerical_delta_interval_not_history_ci():
    from scripts.analyze_tournament_simulation_precision import _mc_summary
    runs = [{'seed': index, 'complete_cohort': {'contrasts': [{
        'forecast_mode': 'post_draw', 'forecast_scope': 'phase_start',
        'axis': 'axis1', 'target_family': 'qualification',
        'target_kind': 'binary', 'metric': 'log_loss', 'model': 'exp081', 'reference': 'fair_series',
        'locked_primary': True, 'delta': delta, 'ci_low': -100., 'ci_high': 100.}]}}
        for index, delta in enumerate((-.0201, -.02, -.0199))]
    report = _mc_summary(runs, 'complete_cohort')[0]
    assert 0 < report['mc_mean_delta_95_half_width'] < .001
    assert report['locked_primary_delta_ll_budget']['status'] == 'pass'
    assert report['predictive_ci'] is None
    runs[1]['complete_cohort']['contrasts'][0]['delta'] = None
    incomplete = _mc_summary(runs, 'complete_cohort')[0]
    assert incomplete['locked_primary_delta_ll_budget']['status'] == 'inconclusive_unscorable'
    assert incomplete['mc_mean_delta_95_half_width'] is None


def test_unknown_edition_boundary_preserves_phase_scores_without_imputing_month():
    rows = [_ledger_row(phase_id=phase, edition_start_at=None,
                        edition_start_status='unverified_whole_edition_boundary',
                        phase_start_at=started, forecast_scope='phase_start')
            for phase, started in (('groups', '2026-01-10T12:00:00Z'),
                                   ('playoffs', '2026-02-10T12:00:00Z'))]
    result = _evaluate(rows, [_ledger_outcome(row, 'A') for row in rows])
    assert _summary(result, 'brier')['mean'] == pytest.approx(.125)
    assert _summary(result, 'brier')['counts']['editions'] == 1
    assert _summary(result, 'brier')['counts']['origins'] == 2
    assert _summary(result, 'brier')['edition_start_min'] is None
    assert _summary(result, 'brier')['ci_low'] is None
    assert result['coverage']['occupied_months'] == 0
    assert result['coverage']['editions_with_unknown_start'] == 1
    assert result['protocol']['bootstrap']['blocks'] == []
    assert result['target_scores'][0]['edition_start_at'] is None
    assert result['calibration'][0]['bins'][7]['occupied_months'] == 0
    assert result['distribution_diagnostics'][0]['predictive_sets'][0]['coverage'] == 1


@pytest.mark.parametrize('changes', [
    {'forecast_scope': 'whole_edition'},
    {'edition_start_status': 'verified'},
    {'phase_start_at': None},
    {'edition_start_at': '2026-01-10T12:00:00Z'},
])
def test_unknown_edition_boundary_requires_explicit_phase_only_provenance(changes):
    row = _ledger_row(edition_start_at=None,
                      edition_start_status='unverified_whole_edition_boundary',
                      phase_start_at='2026-01-10T12:00:00Z', forecast_scope='phase_start')
    row.update(changes)
    with pytest.raises(ValueError):
        _evaluate([row], [_ledger_outcome(row, 'A')])


@pytest.mark.parametrize('kind', ['champion', 'series'])
def test_forecast_scopes_remain_separate_in_scores_calibration_and_distributions(kind):
    rows, outcomes = [], []
    for scope, phases in [('full_tournament', [('start', .9)]),
                          ('phase_start', [('groups', .75), ('playoffs', .25)])]:
        for phase, probability in phases:
            for model in ('exp081', 'fair_series'):
                p = probability if model == 'exp081' else .5
                row = _ledger_row(model=model, phase_id=phase, forecast_scope=scope,
                                  kind=kind, probabilities={'A': p, 'B': 1-p})
                rows.append(row)
                if model == 'exp081':
                    outcomes.append(_ledger_outcome(row, 'A'))
    result = _evaluate(rows, outcomes)
    factor = 1 if kind == 'series' else 2
    for scope, expected, origins in [('full_tournament', .01, 1), ('phase_start', .3125, 2)]:
        summary = _summary(result, 'brier', model='exp081', forecast_scope=scope)
        assert summary['mean'] == pytest.approx(factor * expected)
        assert summary['counts']['editions'] == 1
        assert summary['counts']['origins'] == origins
        comparison = next(row for row in result['comparisons']
                          if row['metric'] == 'brier' and row['forecast_scope'] == scope)
        assert comparison['delta'] == pytest.approx(factor * (expected - .25))
        calibration = next(row for row in result['calibration']
                           if row['model'] == 'exp081' and row['forecast_scope'] == scope)
        expected_ece = .1 if origins == 1 else .5 if kind == 'series' else .25
        assert calibration['ece_10bin'] == pytest.approx(expected_ece)
        diagnostic = next(row for row in result[
            'series_diagnostics' if kind == 'series' else 'distribution_diagnostics']
                          if row['model'] == 'exp081' and row['forecast_scope'] == scope)
        assert diagnostic['editions'] == 1
        assert diagnostic['origins'] == origins
    assert {row['forecast_scope'] for row in result['origin_scores']} == {
        'full_tournament', 'phase_start'}


def test_scope_specific_models_match_only_their_own_target_cohort():
    whole = _ledger_row(model='fair_series', forecast_scope='full_tournament')
    phase = _ledger_row(model='exp081', phase_id='playoffs', forecast_scope='phase_start')
    reference = phase | {'model': 'fair_series'}
    result = _evaluate([whole, phase, reference],
                       [_ledger_outcome(row, 'A') for row in (whole, phase)])
    assert {row['forecast_scope'] for row in result['comparisons']} == {'phase_start'}
    mismatched = phase | {'model': 'fair_series', 'forecast_scope': 'full_tournament'}
    with pytest.raises(ValueError):
        _evaluate([phase, mismatched], [_ledger_outcome(phase, 'A')])


@pytest.mark.parametrize('fault, expected_status', [
    ('failed', 'unavailable_unscorable'),
    ('mc_unresolved', 'unavailable_unscorable'),
    ('infinite', 'unavailable_infinite'),
    ('reference_infinite', 'unavailable_infinite'),
    ('missing_reference', 'unavailable_reference'),
])
def test_pilot_blocks_keep_adverse_targets_and_editions_in_the_full_denominator(fault, expected_status):
    from scripts.academic_cohort import pilot_blocks
    rows, outcomes = [], []
    for edition, month in [('first', '01'), ('second', '02')]:
        for target in ('finite', 'adverse'):
            for model in ('exp081', 'fair_series'):
                probability = .8 if model == 'exp081' else .5
                row = _ledger_row(model=model, tournament=edition, target=target,
                                  forecast_scope='phase_start',
                                  edition_start_at=f'2026-{month}-01T00:00:00Z',
                                  probabilities={'A': probability, 'B': 1-probability})
                affected_model = 'fair_series' if fault == 'reference_infinite' else 'exp081'
                if target == 'adverse' and model == affected_model:
                    if fault == 'failed':
                        row.update(forecast_status='failed', failure_reason='missing features', probabilities=None)
                    elif fault == 'mc_unresolved':
                        row.update(probability_source='monte_carlo', mc_samples=10,
                                   mc_hits={'A': 8, 'B': 2}, mc_precision_resolved=False)
                    elif fault in ('infinite', 'reference_infinite'):
                        row['probabilities'] = {'A': 0., 'B': 1.}
                rows.append(row)
                if model == 'exp081':
                    outcomes.append(_ledger_outcome(row, 'A'))
    scores = _evaluate(rows, outcomes)
    if fault == 'missing_reference':
        scores['target_scores'] = [row for row in scores['target_scores']
                                   if row['model'] != 'fair_series' or row['target_id'] != 'adverse']
    result = pilot_blocks(scores)[0]
    assert result['eligible_targets'] == 4
    assert result['matched_targets'] == (2 if fault == 'missing_reference' else 4)
    assert result['paired_finite_targets'] == 2
    assert result['candidate_edition_blocks'] == result['start_month_blocks'] == 2
    assert result['edition_balanced_log_loss_delta'] is None
    assert result['delta_status'] == expected_status
    assert result['edition_delta_sample_variance'] is None
    assert result['power_status'] == 'not_estimable_unscorable_or_infinite_cohort'
    if fault == 'infinite':
        assert result['model_mean_log_loss'] == 'Infinity'
    elif fault == 'reference_infinite':
        assert result['reference_mean_log_loss'] == 'Infinity'
    elif fault == 'missing_reference':
        assert result['reference_score_status_counts']['missing_reference'] == 2


def test_pilot_blocks_preserve_unavailable_reference_cohort_and_unknown_months():
    from scripts.academic_cohort import pilot_blocks
    rows = [_ledger_row(tournament=edition, forecast_scope='phase_start',
                        edition_start_at=None, edition_start_status='unverified_whole_edition_boundary',
                        phase_start_at=f'2026-{month}-01T00:00:00Z')
            for edition, month in [('first', '01'), ('second', '02')]]
    scores = _evaluate(rows, [_ledger_outcome(row, 'A') for row in rows])
    result = pilot_blocks(scores)[0]
    assert result['eligible_targets'] == 2
    assert result['matched_targets'] == result['start_month_blocks'] == 0
    assert result['unknown_edition_month_blocks'] == 2
    assert result['delta_status'] == 'unavailable_reference'
    assert result['edition_balanced_log_loss_delta'] is None
    assert all(row['edition_start_month'] is None for row in result['edition_values'])
    assert result['edition_delta_sample_variance'] is None


def test_pilot_blocks_balance_real_editions_separately_by_scope_without_month_imputation():
    from scripts.academic_cohort import pilot_blocks
    rows, outcomes = [], []
    for scope, phase in [('full_tournament', 'whole'), ('phase_start', 'groups'), ('phase_start', 'playoffs')]:
        for model in ('exp081', 'fair_series'):
            p = (.8 if scope == 'full_tournament' else .2) if model == 'exp081' else .5
            row = _ledger_row(model=model, phase_id=phase, forecast_scope=scope,
                              probabilities={'A': p, 'B': 1-p})
            rows.append(row)
            if model == 'exp081':
                outcomes.append(_ledger_outcome(row, 'A'))
    contrasts = pilot_blocks(_evaluate(rows, outcomes))
    assert len(contrasts) == 2
    for row in contrasts:
        assert row['candidate_edition_blocks'] == row['start_month_blocks'] == 1
        expected = -log(.8 if row['forecast_scope'] == 'full_tournament' else .2) + log(.5)
        assert row['edition_balanced_log_loss_delta'] == pytest.approx(expected)
    unknown = [row | {'edition_start_at': None, 'edition_start_status': 'unverified_whole_edition_boundary',
                     'phase_start_at': '2026-01-01T00:00:00Z' if row['phase_id'] == 'groups'
                                       else '2026-02-01T00:00:00Z'}
               for row in rows if row['forecast_scope'] == 'phase_start']
    result = pilot_blocks(_evaluate(unknown, [_ledger_outcome(row, 'A') for row in unknown
                                             if row['model'] == 'exp081']))[0]
    assert result['edition_balanced_log_loss_delta'] == pytest.approx(-log(.2) + log(.5))
    assert result['start_month_blocks'] == 0
    assert result['unknown_edition_month_blocks'] == 1
    assert result['edition_delta_sample_variance'] is None


def test_precision_contrasts_and_method_availability_do_not_cross_scopes():
    from scripts.analyze_tournament_simulation_precision import _available_methods, _contrasts, _mc_summary
    rows, outcomes = [], []
    for scope, p in [('full_tournament', .8), ('phase_start', .2)]:
        for model in ('exp081', 'fair_series'):
            probability = p if model == 'exp081' else .5
            row = _ledger_row(model=model, phase_id=scope, forecast_scope=scope,
                              kind='binary', target_family='qualification', forecast_mode='post_draw',
                              probabilities={'A': probability, 'B': 1-probability})
            rows.append(row)
            if model == 'exp081':
                outcomes.append(_ledger_outcome(row, 'A'))
    contrasts = _contrasts(_evaluate(rows, outcomes), 'exp081')
    primary = [row for row in contrasts if row['locked_primary']]
    assert len(primary) == 1 and primary[0]['forecast_scope'] == 'phase_start'
    for row in contrasts:
        assert {identity[1] for identity in row['matched_target_ids']} == {row['forecast_scope']}
    runs = [{'seed': seed, 'complete_cohort': {'contrasts': contrasts}} for seed in (1, 2, 3)]
    summaries = _mc_summary(runs, 'complete_cohort')
    assert len(summaries) == 4
    assert all(row['status'] == 'numerical_variation_only' for row in summaries)
    failed = [row | {'forecast_status': 'failed', 'failure_reason': 'missing features'}
              if row['model'] == 'exp081' and row['forecast_scope'] == 'phase_start' else row
              for row in rows]
    available, removed = _available_methods(failed)
    assert {(row['model'], row['forecast_scope']) for row in available} == {
        ('exp081', 'full_tournament'), ('fair_series', 'full_tournament'), ('fair_series', 'phase_start')}
    assert removed[0]['forecast_scope'] == 'phase_start'


def test_precision_cli_requires_explicit_manifest_and_outcomes_before_creating_output(tmp_path):
    from scripts.analyze_tournament_simulation_precision import main
    output = tmp_path / 'precision'
    for supplied in ([], ['--manifest', str(tmp_path / 'manifest.json')],
                     ['--outcomes', str(tmp_path / 'outcomes.json')]):
        with pytest.raises(SystemExit) as error:
            main(['--output-dir', str(output), *supplied])
        assert error.value.code == 2
        assert not output.exists()


def test_precision_analysis_rejects_legacy_manifest_without_rewriting_evidence(tmp_path):
    import json
    from scripts.analyze_tournament_simulation_precision import analyze
    manifest = tmp_path / 'manifest.json'
    payload = json.dumps({'version': 'tournament-evaluation-v2',
                          'qualification': 'retrospective_reconstruction', 'models': [{'id': 'exp081'}]})
    manifest.write_text(payload)
    output = tmp_path / 'precision'
    with pytest.raises(ValueError, match='tournament-evaluation-v3'):
        analyze(manifest, tmp_path / 'outcomes.json', output, seeds=[1, 2, 3],
                simulations=10, model='exp081', existing_scores=[])
    assert manifest.read_text() == payload
    assert not output.exists()


def test_precision_reuses_complete_seeds_without_mutating_forecasts(tmp_path):
    import json
    from scripts.analyze_tournament_simulation_precision import analyze
    from betting_app.tests.test_tournament_evaluation_runner import _manifest, _save
    from src.models.tournament_prediction import file_digest

    manifest_path, manifest = _manifest(tmp_path)
    manifest['qualification'] = 'retrospective_reconstruction'
    _save(manifest_path, manifest)
    outcomes = tmp_path / 'outcomes.json'
    artifact = tmp_path / 'model.json'
    _save(artifact, {'probability_unit': 'series_win', 'p': 1.0})
    table = json.loads((tmp_path / 'table.json').read_text())
    table['provenance'].update(training_cutoff='2026-01-01T00:00:00Z',
                               calibration_cutoff='2026-01-01T00:00:00Z',
                               model_id='sports', model_artifact_sha256=file_digest(artifact))
    _save(tmp_path / 'table.json', table)
    _save(outcomes, [{'tournament_id': 'event', 'phase_id': 'event-final',
                      'origin_id': 'start', 'target_id': 'champion', 'observed': 'A'}])
    options = dict(seeds=[101, 202, 303], simulations=50, model='sports', existing_scores=[])
    original = analyze(manifest_path, outcomes, tmp_path / 'original', **options)
    directories = [tmp_path / 'original' / f'seed-{seed}' for seed in options['seeds']]
    pinned = {path: file_digest(path) for directory in directories
              for path in (directory / 'protocol.json', directory / 'status.json', directory / 'forecasts.jsonl')}

    reused = analyze(manifest_path, outcomes, tmp_path / 'reused', existing_runs=directories, **options)
    assert reused['complete_cohort_mc'] == original['complete_cohort_mc']
    assert reused['available_method_mc'] == original['available_method_mc']
    assert {path: file_digest(path) for path in pinned} == pinned

    with pytest.raises(ValueError, match='simulation'):
        analyze(manifest_path, outcomes, tmp_path / 'wrong-budget', existing_runs=directories,
                **(options | {'simulations': 51}))
    assert not (tmp_path / 'wrong-budget').exists()

    with pytest.raises(ValueError, match='seed'):
        analyze(manifest_path, outcomes, tmp_path / 'missing-seed', existing_runs=directories[:2], **options)
    assert not (tmp_path / 'missing-seed').exists()
