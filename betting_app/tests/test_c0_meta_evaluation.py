"""Behavioral contract for saved, complete meta forecasts and paired scoring."""
import json

import numpy as np
import pandas as pd
import pytest

from src.analysis.c0_meta_evaluation import evaluate_meta, evaluate_predictions


def forecasts(months=(1, 2, 3)):
    rows = []
    for year in (2024, 2025, 2026):
        for month in months:
            for y in (0, 1):
                p_c0 = .4 if y == 0 else .6
                rows.append({
                    'golgg_match_id': str(len(rows)), 'team1_id': 'a', 'team2_id': 'b',
                    'date': f'{year}-{month:02d}-15', 'result_day': f'{year}-{month:02d}-15',
                    'feature_source_max_day': f'{year}-{month:02d}-14', 'best_of': 3, 'y': y,
                    'partition': {2024: 'train', 2025: 'cal', 2026: 'assessment'}[year],
                    'p_c0': p_c0,
                    'p_player': (.3 if y == 0 else .7) if year == 2026 else p_c0,
                    'p_meta': (.2 if y == 0 else .8) if year == 2026 else p_c0,
                    'tournament_name': 'Regional League', 'target_patch': None,
                })
    return pd.DataFrame(rows)


def test_numeric_confidence_is_grouped_into_fixed_c0_bands():
    frame = forecasts(months=(1, 2, 3, 4))
    assessment = frame.partition.eq('assessment')
    frame.loc[assessment, 'p_c0'] = [.52, .58, .62, .72, .77, .88, .92, .97]
    frame['confidence'] = np.maximum(frame.p_c0, 1-frame.p_c0)
    result = evaluate_predictions(frame, draws=32)
    profiles = result['sport_profiles']['confidence']
    assert {key: value['rows'] for key, value in profiles.items()} == {
        '[.5,.6)': 2, '[.6,.75)': 2, '[.75,.9)': 2, '[.9,1)': 2,
    }


def test_primary_proper_scores_are_unclipped_and_warmup_remains_descriptive():
    frame = forecasts()
    # A valid tiny probability must not be clipped by the benchmark defaults.
    frame.loc[frame.partition.eq('assessment') & frame.y.eq(0), 'p_meta'] = 1e-12
    result = evaluate_predictions(frame, draws=32)
    primary = result['primary']
    assert primary['rows'] == 6
    assert result['full_cohort']['rows'] == 18
    assert result['partition_counts'] == {'train': 6, 'cal': 6, 'assessment': 6}
    assert primary['metrics']['p_meta']['log_loss'] == pytest.approx(
        (-np.log1p(-1e-12) - np.log(.8)) / 2, abs=1e-14)
    assert primary['metrics']['p_meta']['brier_score'] == pytest.approx((1e-24 + .04) / 2)
    assert primary['contrasts']['meta_minus_c0']['log_loss_monthly']['delta'] == pytest.approx(
        primary['metrics']['p_meta']['log_loss'] + np.log(.6))
    assert primary['contrasts']['meta_minus_c0']['log_loss_monthly']['confidence_level'] == .975
    assert primary['contrasts']['meta_minus_c0']['brier_monthly']['confidence_level'] == .95
    assert result['warmup']['forecast_claim'] == 'descriptive_exact_c0_only'


def test_null_patch_is_retained_as_unknown_and_source_name_year_is_only_a_proxy():
    result = evaluate_predictions(forecasts(), draws=32)
    assert result['sport_profiles']['patch_known']['unknown']['rows'] == 6
    assert result['sport_profiles']['patch_phase']['unknown']['rows'] == 6
    sensitivity = result['primary']['source_name_year_proxy_sensitivity']
    assert sensitivity['certified_editions'] is False
    assert sensitivity['contrasts']['meta_minus_c0']['log_loss']['blocks'] == 1


@pytest.mark.parametrize('invalid', ['duplicate', 'missing_probability', 'bad_warmup', 'wrong_partition'])
def test_incomplete_or_inconsistent_forecast_population_is_rejected(invalid):
    frame = forecasts()
    if invalid == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif invalid == 'missing_probability':
        frame.loc[0, 'p_meta'] = np.nan
    elif invalid == 'bad_warmup':
        frame.loc[0, 'p_player'] += .01
    else:
        frame.loc[0, 'partition'] = 'assessment'
    with pytest.raises(ValueError):
        evaluate_predictions(frame, draws=32)


def test_primary_success_requires_both_controls_and_nonpositive_brier_deltas():
    frame = forecasts()
    result = evaluate_predictions(frame, draws=32)
    assert result['status'] == 'DEVELOPMENT_SIGNAL_RETAINED'
    assert result['decision']['primary_success'] is True
    assert result['decision']['model_promotion'] is False
    assessment = frame.partition.eq('assessment')
    frame.loc[assessment, 'p_player'] = frame.loc[assessment, 'p_meta']
    result = evaluate_predictions(frame, draws=32)
    assert result['status'] == 'NOT_ADOPTED'
    assert result['decision']['checks']['meta_vs_player_ll_upper_negative'] is False


def test_one_month_cannot_supply_a_successful_confidence_interval():
    result = evaluate_predictions(forecasts(months=(1,)), draws=32)
    assert result['primary']['contrasts']['meta_minus_c0']['log_loss_monthly']['ci_upper'] is None
    assert result['decision']['primary_success'] is False
    assert result['status'] == 'NOT_ADOPTED'


def test_development_predictions_do_not_replace_main_warmup_or_assessment():
    frame = forecasts()
    frame['p_meta_development'] = np.where(frame.y, .95, .05)
    result = evaluate_predictions(frame, draws=32)
    assert result['development_diagnostics']['train']['p_meta_development']['sample_size'] == 6
    assert result['development_diagnostics']['cal']['p_meta_development']['log_loss'] == pytest.approx(-np.log(.95))
    assert result['primary']['metrics']['p_meta']['log_loss'] == pytest.approx(-np.log(.8))


def test_file_evaluation_refuses_existing_output_directory_before_writing(tmp_path):
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({}))
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(FileExistsError):
        evaluate_meta(protocol, output)
    assert list(output.iterdir()) == []


def test_log_loss_improvement_with_worse_brier_cannot_pass_the_primary_gate():
    template = forecasts()
    template = template.loc[template.partition.eq('assessment')]
    rows = []
    for row in template.to_dict('records'):
        for base, candidate in ((.01, .2), (.8, .55), (.8, .55), (.8, .55)):
            entry = dict(row, golgg_match_id=str(len(rows)))
            entry['p_c0'] = entry['p_player'] = base if row['y'] else 1-base
            entry['p_meta'] = candidate if row['y'] else 1-candidate
            rows.append(entry)
    result = evaluate_predictions(pd.DataFrame(rows), draws=32)
    checks = result['decision']['checks']
    assert checks['meta_vs_c0_ll_upper_negative'] is True
    assert checks['meta_vs_player_ll_upper_negative'] is True
    assert checks['meta_vs_c0_brier_nonpositive'] is False
    assert checks['meta_vs_player_brier_nonpositive'] is False
    assert result['status'] == 'NOT_ADOPTED'


def test_excluded_boundary_labels_remain_in_warmup_but_not_fit_diagnostics():
    frame = forecasts()
    frame['effective_label_release_day'] = frame.result_day
    frame['fit_exclusion_reason'] = None
    frame.loc[0, 'partition'] = 'excluded_fit_labels'
    frame.loc[0, 'fit_exclusion_reason'] = 'released_on_fit_boundary'
    frame.loc[0, 'effective_label_release_day'] = '2025-01-01'
    result = evaluate_predictions(frame, draws=32)
    assert result['full_cohort']['rows'] == 18
    assert result['warmup']['rows'] == 12
    assert result['partition_counts']['excluded_fit_labels'] == 1
    frame.loc[0, 'effective_label_release_day'] = '2024-12-31'
    with pytest.raises(ValueError):
        evaluate_predictions(frame, draws=32)


@pytest.mark.parametrize('status', ['RUNNING', 'SMOKE_PASSED', 'COMPLETED'])
def test_formal_evaluation_rejects_uncompleted_or_unbound_execution(tmp_path, status):
    execution = tmp_path / 'execution.json'
    execution.write_text(json.dumps({
        'schema': 'c0_meta_execution_v1', 'status': status,
        'protocol_sha256': '0' * 64,
    }))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({
        'schema': 'c0_role_meta_v1', 'status': 'FROZEN_BEFORE_FITTING',
        'execution_path': str(execution),
        'evaluation': {'draws': 5000, 'seed': 20261007,
                       'primary_confidence': .975, 'probability_epsilon': None},
    }))
    output = tmp_path / 'report'
    with pytest.raises(ValueError):
        evaluate_meta(protocol, output)
    assert not output.exists()
