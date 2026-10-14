"""Observable run persistence, temporal boundaries and real conditional forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tournament_evaluation import run_evaluation


def _save(path, payload):
    path.write_text(json.dumps(payload))
    return path.name


def _manifest(tmp_path):
    spec = {'version': 1, 'id': 'fixture', 'teams': ['A', 'B'], 'stages': [
        {'id': 'final', 'kind': 'single_elimination', 'entrants': ['A', 'B'], 'best_of': 1,
         'draw': {'policy': 'uniform'}}
    ], 'champion': {'stage': 'final', 'group': 'all', 'rank': 1}}
    table = {'probability_unit': 'series_win', 'rows': [
        {'team_a': 'A', 'team_b': 'B', 'best_of': 1, 'p': 1.0}
    ], 'provenance': {'feature_history_max_at': '2026-01-01T00:00:00Z'}}
    manifest = {
        'version': 'tournament-evaluation-v3', 'qualification': 'synthetic_mechanics',
        'models': [{'id': 'sports', 'kind': 'series_table'}],
        'events': [{'tournament_id': 'event', 'phase_id': 'event-final',
                    'edition_start_at': '2026-01-02T12:00:00Z',
                    'family': 'fixture', 'tier': 'fixture', 'format': 'single_elimination',
                    'spec': _save(tmp_path / 'spec.json', spec),
                    'origins': [{'origin_id': 'start', 'axis': 'axis1',
                                 'forecast_scope': 'full_tournament',
                                 'cutoff': '2026-01-02T00:00:00Z',
                                 'tables': {'sports': _save(tmp_path / 'table.json', table)},
                                 'forecast_mode': 'pre_draw',
                                 'targets': [{'target_id': 'champion', 'target_kind': 'champion',
                                              'source': 'champion_prob',
                                              'target_start_at': '2026-01-02T12:00:00Z'}]}]}],
    }
    path = tmp_path / 'manifest.json'
    _save(path, manifest)
    return path, manifest


def test_runner_persists_real_predictions_and_resumes_without_duplicates(tmp_path):
    path, _ = _manifest(tmp_path)
    output = tmp_path / 'run'
    result = run_evaluation(path, output, simulations=100, seed=7)
    assert result['status'] == 'completed'
    rows = [json.loads(line) for line in (output / 'forecasts.jsonl').read_text().splitlines()]
    sports = next(row for row in rows if row['model'] == 'sports')
    assert sports['probabilities'] == {'A': 1.0, 'B': 0.0}
    assert next(row for row in rows if row['model'] == 'flat')['probabilities'] == {'A': 0.5, 'B': 0.5}
    before = (output / 'forecasts.jsonl').read_bytes()
    run_evaluation(path, output, simulations=100, seed=7, resume=True)
    assert (output / 'forecasts.jsonl').read_bytes() == before


def test_linked_phase_origins_keep_edition_identity_and_distinct_frozen_matrices(tmp_path):
    path, manifest = _manifest(tmp_path)
    spec = {
        'version': 1, 'id': 'linked-edition', 'teams': ['A', 'B'],
        'stages': [
            {'id': 'qualifier', 'kind': 'single_elimination', 'entrants': ['A', 'B'], 'best_of': 1},
            {'id': 'final', 'kind': 'single_elimination', 'best_of': 1,
             'entrants': [{'stage': 'qualifier', 'group': 'all', 'rank': rank} for rank in (1, 2)]},
        ],
        'champion': {'stage': 'final', 'group': 'all', 'rank': 1},
    }
    _save(tmp_path / 'spec.json', spec)
    first = manifest['events'][0]
    first.update(phase_id='qualifier', phase_start_at=first['edition_start_at'])
    start = first['origins'][0]
    start.update(forecast_mode='post_draw', draw_knowledge={'status': 'deterministic_spec'})
    start['targets'][0]['target_start_at'] = '2026-01-08T12:00:00Z'
    second = json.loads(json.dumps(first))
    second.update(phase_id='final', phase_start_at='2026-01-06T12:00:00Z')
    table = json.loads((tmp_path / 'table.json').read_text())
    table['rows'][0]['p'] = 0.0
    _save(tmp_path / 'phase-table.json', table)
    second_start = second['origins'][0]
    second_start.update(
        forecast_scope='phase_start', phase_stages=['final'], cutoff='2026-01-06T00:00:00Z',
        tables={'sports': 'phase-table.json'},
    )
    completed = {
        'stage': 'qualifier', 'group': 'all', 'round': 1, 'team_a': 'A', 'team_b': 'B',
        'best_of': 1, 'weight': 1, 'winner': 'A', 'loser': 'B',
        'score_a': 1, 'score_b': 0, 'source_date': '2026-01-03',
    }
    second_start['state'] = _save(tmp_path / 'phase-start-state.json', {
        'version': 1, 'cutoff': second_start['cutoff'], 'temporal_policy': 'daily_rollforward',
        'completed': [completed],
    })
    checkpoint = {
        **second_start, 'origin_id': 'checkpoint', 'axis': 'axis3',
        'reference_origin_id': 'start',
        'forecast_scope': 'remaining_phase', 'forecast_mode': 'daily_rollforward',
        'cutoff': '2026-01-07T00:00:00Z', 'tables': {'sports': 'table.json'},
        'state': _save(tmp_path / 'checkpoint-state.json', {
            'version': 1, 'cutoff': '2026-01-07T00:00:00Z',
            'temporal_policy': 'daily_rollforward', 'completed': [completed],
        }),
    }
    second['origins'].append(checkpoint)
    manifest['events'].append(second)
    _save(path, manifest)

    result = run_evaluation(path, tmp_path / 'run', simulations=50, seed=7)
    rows = [json.loads(line) for line in (tmp_path / 'run' / 'forecasts.jsonl').read_text().splitlines()]

    assert result['completed_origins'] == 3
    assert {row['tournament_id'] for row in rows} == {'event'}
    starts = {row['phase_id']: row for row in rows if row['origin_id'] == 'start' and row['model'] == 'sports'}
    assert starts['qualifier']['probabilities']['A'] == 1.0
    assert starts['final']['probabilities']['A'] == 0.0
    current = {row['model']: row for row in rows if row['origin_id'] == 'checkpoint'}
    assert current['sports:frozen']['probabilities']['A'] == 0.0
    assert current['sports:refreshed']['probabilities']['A'] == 1.0
    before = (tmp_path / 'run' / 'forecasts.jsonl').read_bytes()
    run_evaluation(path, tmp_path / 'run', simulations=50, seed=7, resume=True)
    assert (tmp_path / 'run' / 'forecasts.jsonl').read_bytes() == before


def test_resume_rejects_changed_source_even_when_manifest_unchanged(tmp_path):
    path, _ = _manifest(tmp_path)
    output = tmp_path / 'run'
    run_evaluation(path, output, simulations=100, seed=7)
    table = json.loads((tmp_path / 'table.json').read_text())
    table['rows'][0]['p'] = 0.0
    _save(tmp_path / 'table.json', table)
    with pytest.raises(ValueError, match='changed|differ|hash'):
        run_evaluation(path, output, simulations=100, seed=7, resume=True)


def test_forecast_builder_never_reads_outcome_file(tmp_path):
    path, manifest = _manifest(tmp_path)
    manifest['outcomes'] = 'not-yet-created.json'
    _save(path, manifest)
    result = run_evaluation(path, tmp_path / 'run', simulations=100)
    assert result['status'] == 'completed'
    assert not (tmp_path / 'not-yet-created.json').exists()


def test_post_start_origin_is_not_accepted_as_full_tournament_forecast(tmp_path):
    path, manifest = _manifest(tmp_path)
    manifest['events'][0]['origins'][0]['cutoff'] = '2026-01-03T00:00:00Z'
    _save(path, manifest)
    with pytest.raises(ValueError, match='before|precede'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_round_series_targets_require_published_pairing_before_cutoff(tmp_path):
    path, manifest = _manifest(tmp_path)
    origin = manifest['events'][0]['origins'][0]
    origin['targets'] = [{'target_id': 'match', 'target_kind': 'series',
                          'source': 'series_probability', 'team_a': 'A', 'team_b': 'B', 'best_of': 1,
                          'target_start_at': '2026-01-02T12:00:00Z',
                          'pairing_available_at': '2026-01-02T01:00:00Z'}]
    _save(path, manifest)
    with pytest.raises(ValueError, match='pairing'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_interruption_preserves_completed_origin_and_resume_finishes_remaining(tmp_path, monkeypatch):
    import scripts.tournament_evaluation as runner

    path, manifest = _manifest(tmp_path)
    second = json.loads(json.dumps(manifest['events'][0]))
    second['tournament_id'] = 'second-event'
    manifest['events'].append(second)
    _save(path, manifest)
    real_probabilities = runner._probabilities

    def interrupt(model, origin, event, *args):
        if event['tournament_id'] == 'second-event':
            raise RuntimeError('interrupted after first saved origin')
        return real_probabilities(model, origin, event, *args)

    monkeypatch.setattr(runner, '_probabilities', interrupt)
    output = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='interrupted'):
        runner.run_evaluation(path, output, simulations=100)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['completed_origins'] == 1
    rows = [json.loads(line) for line in (output / 'forecasts.jsonl').read_text().splitlines()]
    assert {row['tournament_id'] for row in rows} == {'event'}
    monkeypatch.setattr(runner, '_probabilities', real_probabilities)
    finished = runner.run_evaluation(path, output, simulations=100, resume=True)
    assert finished['completed_origins'] == 2
    rows = [json.loads(line) for line in (output / 'forecasts.jsonl').read_text().splitlines()]
    assert len({(row['tournament_id'], row['phase_id'], row['origin_id'], row['model'], row['target_id'])
                for row in rows}) == len(rows)


def test_changed_completed_forecast_ledger_cannot_be_scored(tmp_path):
    from scripts.tournament_evaluation import score_evaluation

    path, _ = _manifest(tmp_path)
    output = tmp_path / 'run'
    run_evaluation(path, output, simulations=100)
    with (output / 'forecasts.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='ledger changed'):
        score_evaluation(output, tmp_path / 'unread-outcomes.json', tmp_path / 'scores.json')


def test_resume_rejects_changed_saved_origin_not_just_flat_ledger(tmp_path):
    path, _ = _manifest(tmp_path)
    output = tmp_path / 'run'
    run_evaluation(path, output, simulations=100)
    origin = next((output / 'origins').glob('*/record.json'))
    content = json.loads(origin.read_text())
    content['forecasts'][0]['probabilities'] = {'A': 1.0, 'B': 0.0}
    origin.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='origin.*changed|origin.*hash'):
        run_evaluation(path, output, simulations=100, resume=True)


def test_prospective_capture_rejects_prior_day_reconstructed_state(tmp_path):
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    path, manifest = _manifest(tmp_path)
    manifest['qualification'] = 'prospective_capture'
    event = manifest['events'][0]
    event['edition_start_at'] = (now - timedelta(days=1)).isoformat()
    for field in ('rules_available_at', 'entrants_available_at', 'seeds_available_at'):
        event[field] = (now - timedelta(days=3)).isoformat()
    origin = event['origins'][0]
    origin['forecast_mode'] = 'daily_rollforward'
    origin['axis'] = 'axis2'
    origin['cutoff'] = (now - timedelta(minutes=1)).isoformat()
    origin['targets'][0]['target_start_at'] = (now + timedelta(days=1)).isoformat()
    origin['state'] = _save(tmp_path / 'state.json', {
        'version': 1, 'cutoff': origin['cutoff'], 'completed': [],
        'temporal_policy': 'prior_day_reconstruction',
    })
    table = json.loads((tmp_path / 'table.json').read_text())
    table['provenance'].update(model_id='sports', model_artifact_sha256='a' * 64,
                               training_cutoff=(now - timedelta(days=30)).isoformat(),
                               calibration_cutoff=(now - timedelta(days=3)).isoformat(),
                               created_at=(now - timedelta(minutes=2)).isoformat())
    _save(tmp_path / 'table.json', table)
    _save(path, manifest)
    with pytest.raises(ValueError, match='reconstructed|strict_timestamped'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_missing_refresh_is_explicit_failure_not_frozen_feature_fallback(tmp_path):
    path, manifest = _manifest(tmp_path)
    start = manifest['events'][0]['origins'][0]
    checkpoint = {
        'origin_id': 'checkpoint', 'axis': 'axis3', 'forecast_scope': 'remaining_tournament',
        'reference_origin_id': 'start',
        'forecast_mode': 'daily_rollforward', 'cutoff': '2026-01-03T00:00:00Z',
        'model_exclusions': {'sports': 'missing completed historical game objects'},
        'targets': [{**start['targets'][0], 'target_start_at': '2026-01-04T12:00:00Z'}],
        'state': _save(tmp_path / 'state.json', {
            'version': 1, 'cutoff': '2026-01-03T00:00:00Z',
            'temporal_policy': 'daily_rollforward', 'completed': [],
        }),
    }
    manifest['events'][0]['origins'].append(checkpoint)
    _save(path, manifest)
    output = tmp_path / 'run'
    result = run_evaluation(path, output, simulations=100)
    assert result['forecast_failures'] == 1
    rows = {row['model']: row for row in map(json.loads, (output / 'forecasts.jsonl').read_text().splitlines())
            if row['origin_id'] == 'checkpoint'}
    assert rows['sports:frozen']['probabilities']['A'] == 1.0
    assert rows['sports:refreshed']['forecast_status'] == 'failed'
    assert rows['sports:refreshed']['probabilities'] == {}
    assert rows['sports:refreshed']['failure_reason'] == 'missing completed historical game objects'
    assert rows['fair_series']['probabilities']['A'] < 1.0


def _completed_final(day, field='completed_date'):
    return {
        'stage': 'final', 'group': 'all', 'round': 1, 'team_a': 'A', 'team_b': 'B',
        'best_of': 1, 'winner': 'A', 'loser': 'B', 'score_a': 1, 'score_b': 0,
        'weight': 1, field: day,
    }


def test_runner_persists_every_declared_forecast_mode_with_provenance(tmp_path):
    for mode in ('pre_draw', 'post_draw', 'daily_rollforward', 'post_round'):
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        path, manifest = _manifest(mode_dir)
        origin = manifest['events'][0]['origins'][0]
        if mode == 'post_draw':
            origin['forecast_mode'] = mode
            origin['state'] = _save(mode_dir / 'state.json', {
                'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'strict_timestamped',
                'completed': [], 'pairings': [{'stage': 'final', 'group': 'all', 'round': 1,
                                                'pairs': [['A', 'B']],
                                                'published_at': '2026-01-01T00:00:00Z',
                                                'available_at': origin['cutoff']}],
            })
        elif mode == 'daily_rollforward':
            origin.update(forecast_mode=mode, axis='axis2', cutoff='2026-01-03T00:00:00Z')
            origin['targets'][0]['target_start_at'] = '2026-01-04T12:00:00Z'
            origin['state'] = _save(mode_dir / 'state.json', {
                'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'daily_rollforward',
                'completed': [],
            })
        elif mode == 'post_round':
            origin.update(forecast_mode=mode, axis='axis2', cutoff='2026-01-03T00:00:00Z',
                          checkpoint={'stage': 'final', 'group': 'all', 'round': 1})
            origin['targets'][0].update(resolved=True, target_start_at='2026-01-02T12:00:00Z')
            completed = _completed_final('unused')
            completed.pop('completed_date')
            completed.update(completed_at='2026-01-02T18:00:00Z', available_at='2026-01-02T20:00:00Z')
            origin['state'] = _save(mode_dir / 'state.json', {
                'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'strict_timestamped',
                'completed': [completed], 'pairings': [{
                    'stage': 'final', 'group': 'all', 'round': 1, 'pairs': [['A', 'B']],
                    'published_at': '2026-01-02T12:00:00Z', 'available_at': '2026-01-02T12:00:00Z',
                }],
            })
        _save(path, manifest)
        output = mode_dir / 'run'
        run_evaluation(path, output, simulations=100, seed=7)
        rows = [json.loads(line) for line in (output / 'forecasts.jsonl').read_text().splitlines()]
        assert {row['forecast_mode'] for row in rows} == {mode}
        expected_day = '2026-01-02' if mode == 'daily_rollforward' else None
        assert {row['observation_day_end'] for row in rows} == {expected_day}


def test_pre_draw_requires_declared_unknown_draw_policy(tmp_path):
    path, manifest = _manifest(tmp_path)
    spec = json.loads((tmp_path / 'spec.json').read_text())
    spec['stages'][0].pop('draw')
    _save(tmp_path / 'spec.json', spec)
    with pytest.raises(ValueError, match='unknown.*draw|draw.*policy'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


@pytest.mark.parametrize(
    ('cutoff', 'completed', 'message'),
    [
        ('2026-01-03T00:00:00Z', [_completed_final('2026-01-03', 'source_date')], 'strictly precede'),
        ('2026-01-03T01:00:00Z', [], 'UTC midnight'),
    ],
)
def test_daily_rollforward_rejects_same_day_and_non_midnight_state(tmp_path, cutoff, completed, message):
    path, manifest = _manifest(tmp_path)
    origin = manifest['events'][0]['origins'][0]
    origin.update(axis='axis2', forecast_mode='daily_rollforward', cutoff=cutoff)
    origin['targets'][0]['target_start_at'] = '2026-01-04T12:00:00Z'
    origin['state'] = _save(tmp_path / 'state.json', {
        'version': 1, 'cutoff': cutoff, 'temporal_policy': 'daily_rollforward', 'completed': completed,
    })
    _save(path, manifest)
    with pytest.raises(ValueError, match=message):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_daily_rollforward_rejects_completed_result_without_source_day(tmp_path):
    path, manifest = _manifest(tmp_path)
    origin = manifest['events'][0]['origins'][0]
    origin.update(axis='axis2', forecast_mode='daily_rollforward', cutoff='2026-01-03T00:00:00Z')
    origin['targets'][0]['target_start_at'] = '2026-01-04T12:00:00Z'
    completed = _completed_final('2026-01-02')
    completed.pop('completed_date')
    origin['state'] = _save(tmp_path / 'state.json', {
        'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'daily_rollforward',
        'completed': [completed],
    })
    _save(path, manifest)
    with pytest.raises(ValueError, match='source_date provenance'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_post_draw_accepts_declared_deterministic_spec_without_pairing_state(tmp_path):
    path, manifest = _manifest(tmp_path)
    spec = json.loads((tmp_path / 'spec.json').read_text())
    spec['stages'][0].pop('draw')
    _save(tmp_path / 'spec.json', spec)
    origin = manifest['events'][0]['origins'][0]
    origin.update(forecast_mode='post_draw', draw_knowledge={'status': 'deterministic_spec'})
    _save(path, manifest)
    result = run_evaluation(path, tmp_path / 'run', simulations=100)
    assert result['status'] == 'completed'


def test_runner_rejects_v1_manifest_after_forecast_provenance_cutover(tmp_path):
    path, manifest = _manifest(tmp_path)
    manifest['version'] = 'tournament-evaluation-v1'
    _save(path, manifest)
    with pytest.raises(ValueError, match='version'):
        run_evaluation(path, tmp_path / 'run', simulations=100)


def test_post_round_rejects_missing_checkpoint_and_partial_pairing_block(tmp_path):
    path, manifest = _manifest(tmp_path)
    origin = manifest['events'][0]['origins'][0]
    origin.update(forecast_mode='post_round', axis='axis2', cutoff='2026-01-03T00:00:00Z')
    origin['targets'][0].update(resolved=True, target_start_at='2026-01-02T12:00:00Z')
    completed = _completed_final('unused')
    completed.pop('completed_date')
    completed.update(completed_at='2026-01-02T18:00:00Z', available_at='2026-01-02T20:00:00Z')
    origin['state'] = _save(tmp_path / 'state.json', {
        'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'strict_timestamped',
        'completed': [completed],
    })
    _save(path, manifest)
    with pytest.raises(ValueError, match='explicit checkpoint'):
        run_evaluation(path, tmp_path / 'missing-checkpoint', simulations=100)

    origin['checkpoint'] = {'stage': 'final', 'group': 'all', 'round': 1}
    origin['state'] = _save(tmp_path / 'state.json', {
        'version': 1, 'cutoff': origin['cutoff'], 'temporal_policy': 'strict_timestamped',
        'completed': [completed, completed],
        'pairings': [{'stage': 'final', 'group': 'all', 'round': 1, 'pairs': [['A', 'B']],
                      'published_at': '2026-01-02T12:00:00Z', 'available_at': '2026-01-02T12:00:00Z'}],
    })
    _save(path, manifest)
    with pytest.raises(ValueError, match='partial or mismatched'):
        run_evaluation(path, tmp_path / 'partial-block', simulations=100)



def test_post_round_accepts_exact_completed_declared_graph_match():
    from scripts.tournament_evaluation import _validate_mode_state

    completed = _completed_final('unused')
    completed.pop('completed_date')
    completed.update(stage='main', round='r1')
    origin = {'forecast_mode': 'post_round', 'checkpoint': {'stage': 'main', 'group': 'all', 'round': 'r1'}}
    spec = {'stages': [{'id': 'main', 'kind': 'graph', 'matches': [{'id': 'r1'}]}]}
    _validate_mode_state(origin, spec, {'completed': [completed]}, 'synthetic_mechanics')

    origin['checkpoint']['round'] = 'unknown'
    with pytest.raises(ValueError, match='completed records|complete pairing block|declared graph match'):
        _validate_mode_state(origin, spec, {'completed': [completed]}, 'synthetic_mechanics')



def test_phase_replay_preserves_unknown_whole_edition_boundary(tmp_path):
    path, manifest = _manifest(tmp_path)
    event = manifest['events'][0]
    event.update(edition_start_at=None, edition_start_status='unverified_whole_edition_boundary',
                 phase_start_at='2026-01-02T12:00:00Z')
    origin = event['origins'][0]
    origin.update(forecast_scope='phase_start', phase_stages=['final'])
    _save(path, manifest)
    run_evaluation(path, tmp_path / 'phase', simulations=20)
    rows = list(map(json.loads, (tmp_path / 'phase' / 'forecasts.jsonl').read_text().splitlines()))
    assert all(row['edition_start_at'] is None for row in rows)
    origin['forecast_scope'] = 'full_tournament'
    _save(path, manifest)
    with pytest.raises(ValueError, match='independently evidenced'):
        run_evaluation(path, tmp_path / 'whole', simulations=20)


def test_start_phase_draw_retains_previous_phase_observations():
    from scripts.tournament_evaluation import _validate_mode_state
    origin = {'forecast_mode': 'pre_draw', 'phase_stages': ['final']}
    spec = {'stages': [{'id': 'group', 'kind': 'round_robin'},
                       {'id': 'final', 'draw': {'policy': 'uniform'}}]}
    state = {'completed': [{'stage': 'group'}]}
    _validate_mode_state(origin, spec, state, 'synthetic_mechanics')
    state['completed'].append({'stage': 'final'})
    with pytest.raises(ValueError, match='own phase'):
        _validate_mode_state(origin, spec, state, 'synthetic_mechanics')


def test_observable_ledger_keeps_oriented_score_and_count_categories(tmp_path):
    path, manifest = _manifest(tmp_path)
    origin = manifest['events'][0]['origins'][0]
    origin['observables'] = [
        {'id': 'score', 'kind': 'node_score', 'node': {'stage': 'final', 'round': 1, 'match': 1},
         'participants': ['A', 'B'], 'categories': ['1-0', '0-1', 'absent'], 'absent_category': 'absent'},
        {'id': 'maps', 'kind': 'future_map_count', 'categories': ['1']},
    ]
    origin['targets'] = [
        {'target_id': observable['id'], 'target_kind': kind, 'source': 'observable_prob',
         'observable_id': observable['id'], 'categories': observable['categories'],
         'target_start_at': '2026-01-02T12:00:00Z'}
        for observable, kind in zip(origin['observables'], ['categorical', 'count'])]
    _save(path, manifest)
    run_evaluation(path, tmp_path / 'observables', simulations=20)
    rows = list(map(json.loads, (tmp_path / 'observables' / 'forecasts.jsonl').read_text().splitlines()))
    sports = {row['target_id']: row for row in rows if row['model'] == 'sports'}
    assert sports['score']['probabilities'] == {'1-0': 1, '0-1': 0, 'absent': 0}
    assert sports['maps']['probabilities'] == {'1': 1}
    assert not any(row['model'] == 'flat' for row in rows)
    assert sports['score']['mc_precision_resolved'] is False
