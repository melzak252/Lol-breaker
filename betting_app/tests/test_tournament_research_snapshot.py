from dataclasses import asdict
from datetime import UTC, datetime, timedelta
import json

import pandas as pd
import pytest

from betting_app.services.current_roster_service import ROLE_ORDER
from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from scripts.tournament_research_snapshot import capture_tournament, replay_tournament
from src.models.symmetric_series import _PROBABILITY_FIELDS, _REQUIRED_BASE_FIELDS


def inputs(tmp_path):
    directory = tmp_path / 'inputs'
    directory.mkdir()
    bracket = TournamentBracket('fixture', 'Fixture only', 'fixture', 'single_elimination',
        {'final': BracketMatchNode('final', 'Final', 'final', 'final', 3, 'A', 'B')}, ['A', 'B'])
    (directory / 'bracket.json').write_text(json.dumps(asdict(bracket)))
    (directory / 'source.txt').write_text('Synthetic frozen inputs for contract verification, not a real forecast.')
    metadata = {
        'event_start_at': (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        'evidence_files': ['source.txt'], 'bracket_sources': ['source.txt'], 'feature_sources': ['source.txt'],
        'rosters': {team: {'players': {role: f'{team}-{role}' for role in ROLE_ORDER},
                            'source_files': ['source.txt']} for team in ['A', 'B']},
    }
    (directory / 'event.json').write_text(json.dumps(metadata))
    raw = {field: .5 if field in _PROBABILITY_FIELDS else 0. for field in _REQUIRED_BASE_FIELDS}
    raw.update(team_a='A', team_b='B', best_of=3,
               feature_history_max_at=(datetime.now(UTC)-timedelta(days=1)).isoformat())
    pd.DataFrame([raw]).to_csv(directory / 'pair_features.csv', index=False)
    return directory, metadata


def protocol(tmp_path):
    # Real numerical inference from local canonical artifacts, not mocked probabilities.
    from scripts.confirm_oddsfree_student import freeze_protocol
    from scripts.experiment_oddsfree_distillation import VERSION
    from scripts.train_and_tune_siamese_series import build_training_features
    from src.models.symmetric_series import _REQUIRED_BASE_FIELDS
    import hashlib
    import joblib
    import numpy as np

    source = tmp_path / 'models'
    source.mkdir()
    raw = {field: .5 if field in _PROBABILITY_FIELDS else 0. for field in _REQUIRED_BASE_FIELDS}
    _, names = build_training_features(pd.DataFrame([{**raw, 'best_of': 3}]))
    stages = {'train': {'first':'2020-01-01','last':'2024-12-31'},
              'stop': {'first':'2025-01-01','last':'2025-03-31'},
              'select': {'first':'2025-04-01','last':'2025-06-30'},
              'calibration': {'first':'2025-07-01','last':'2025-12-31'},
              'test': {'first':'2026-01-01','last':'2026-05-11'}}
    for stage in stages.values():
        stage['n'] = 6
    models = {}
    for name, weight in [('outcome', 0.), ('market_w10', .1)]:
        path = source / f'2026-{name}.joblib'
        joblib.dump({'version':VERSION,'status':'experimental_not_qualified','target':'series','model':name,
                     'year':2026,'feature_names':names,'scale':np.ones(len(names)), 'coef':np.zeros(len(names)),
                     'slope':1.,'weight':weight,'teacher':None if not weight else 'market',
                     'stages':stages,'requires_odds_at_inference':False,'limitations':['fixture only']},path)
        models[name]={'artifact':path.name,'artifact_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'weight':weight,'final_calibration_slope':1.}
    (source / '2026-fold.json').write_text(json.dumps({'year':2026,'stages':stages,'models':models}))
    (source / 'protocol.json').write_text(json.dumps({'experiment':'EXP-089','version':VERSION}))
    (source / 'summary.json').write_text(json.dumps({'rows':8860,'status':'completed_research_not_promoted'}))
    destination = tmp_path / 'protocol'
    freeze_protocol(source, destination)
    return destination


def test_snapshot_replay_reconstructs_complete_series_forecast(tmp_path):
    source, _ = inputs(tmp_path)
    locked = protocol(tmp_path)
    snapshot = tmp_path / 'snapshot'
    capture_tournament(locked, source, snapshot)
    first = replay_tournament(locked, snapshot, tmp_path / 'first', simulations=12000, seed=89)
    second = replay_tournament(locked, snapshot, tmp_path / 'second', simulations=12000, seed=89)
    assert first['models'] == second['models']
    assert first['snapshot_sha256'] == second['snapshot_sha256']
    assert first['qualification'] == 'prospective_local_capture_not_source_certified'
    for result in first['models'].values():
        assert abs(result['champion_prob']['A'] - .5) < .02
        assert result['final_prob'] == {'A': 1., 'B': 1.}
    with pytest.raises(FileExistsError):
        capture_tournament(locked, source, snapshot)


def test_snapshot_refuses_post_start_event_and_future_features(tmp_path):
    source, metadata = inputs(tmp_path)
    locked = protocol(tmp_path)
    metadata['event_start_at'] = (datetime.now(UTC)-timedelta(seconds=1)).isoformat()
    (source/'event.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        capture_tournament(locked, source, tmp_path/'late')
    metadata['event_start_at'] = (datetime.now(UTC)+timedelta(days=1)).isoformat()
    (source/'event.json').write_text(json.dumps(metadata))
    frame = pd.read_csv(source/'pair_features.csv')
    frame['feature_history_max_at'] = (datetime.now(UTC)+timedelta(hours=1)).isoformat()
    frame.to_csv(source/'pair_features.csv',index=False)
    with pytest.raises(ValueError):
        capture_tournament(locked, source, tmp_path/'future')


def test_snapshot_rejects_altered_archived_features(tmp_path):
    source, _ = inputs(tmp_path)
    locked = protocol(tmp_path)
    snapshot = tmp_path/'snapshot'
    capture_tournament(locked, source, snapshot)
    path = snapshot/'inputs'/'pair_features.csv'
    frame = pd.read_csv(path)
    frame['team_elo'] = .9
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError):
        replay_tournament(locked,snapshot,tmp_path/'tampered')


def test_snapshot_requires_roster_evidence_and_rejects_path_escape(tmp_path):
    source, metadata = inputs(tmp_path)
    locked = protocol(tmp_path)
    metadata['rosters']['A']['source_files'] = []
    (source/'event.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        capture_tournament(locked,source,tmp_path/'unproven')
    metadata['rosters']['A']['source_files'] = ['source.txt']
    metadata['evidence_files'] = ['../secret.txt']
    (source/'event.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        capture_tournament(locked,source,tmp_path/'escape')


def test_prestart_scoring_rejects_resealed_posthoc_champion_probabilities(tmp_path, monkeypatch):
    import hashlib
    import sys
    from scripts.score_tournament_forecasts import main

    source, _ = inputs(tmp_path)
    locked = protocol(tmp_path)
    snapshot, simulation = tmp_path/'snapshot', tmp_path/'simulation'
    capture_tournament(locked, source, snapshot)
    replay_tournament(locked, snapshot, simulation, simulations=2000, seed=89)
    results = tmp_path/'results.json'
    results.write_text(json.dumps({'tournament_id': 'fixture', 'matches': {'final': {
        'team1': 'A', 'team2': 'B', 'winner': 'A', 'score1': 2, 'score2': 1}}}))
    index = tmp_path/'index.json'
    index.write_text(json.dumps([{
        'snapshot': 'snapshot/snapshot.json', 'protocol': 'protocol',
        'forecast': 'simulation/simulation.json', 'bracket': 'snapshot/inputs/bracket.json',
        'results': 'results.json',
    }]))
    arguments = ['score_tournament_forecasts', '--bundle-manifest', str(index), '--model', 'outcome']
    monkeypatch.setattr(sys, 'argv', arguments + ['--output-dir', str(tmp_path/'valid-scores')])
    main()
    forecast_path = simulation/'simulation.json'
    altered = json.loads(forecast_path.read_text())
    altered['models']['outcome']['champion_prob'] = {'A': 1.0, 'B': 0.0}
    forecast_path.write_text(json.dumps(altered))
    (simulation/'simulation.sha256').write_text(hashlib.sha256(forecast_path.read_bytes()).hexdigest() + '\n')
    monkeypatch.setattr(sys, 'argv', arguments + ['--output-dir', str(tmp_path/'altered-scores')])
    with pytest.raises(ValueError, match='captur|replay'):
        main()
