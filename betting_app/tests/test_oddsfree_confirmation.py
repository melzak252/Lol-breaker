import hashlib
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts import confirm_oddsfree_student as confirmation
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS


@pytest.fixture
def source_run(tmp_path):
    """Small numerical artifacts, never repository research data or fitted evidence."""
    path = tmp_path / 'source'
    path.mkdir()
    stages = {
        'train': {'n': 20, 'first': '2022-01-01', 'last': '2023-12-01'},
        'stop': {'n': 5, 'first': '2024-01-01', 'last': '2024-03-01'},
        'select': {'n': 5, 'first': '2024-04-01', 'last': '2024-06-01'},
        'calibration': {'n': 5, 'first': '2024-07-01', 'last': '2024-12-01'},
    }
    row = {field: 0.5 for field in _REQUIRED_BASE_FIELDS}
    row['best_of'] = 3
    names = confirmation.build_training_features(pd.DataFrame([row]))[1]
    fold = {'year': 2026, 'stages': stages, 'models': {}}
    for name, weight in [('outcome', 0.0), ('market_w10', 0.1)]:
        coef = np.zeros(len(names))
        coef[names.index('team_elo_logit')] = 0.7 + weight
        artifact = {
            'target': 'series', 'requires_odds_at_inference': False, 'model': name,
            'year': 2026, 'weight': weight, 'teacher': None if name == 'outcome' else 'market',
            'feature_names': names, 'coef': coef, 'scale': np.ones(len(names)), 'slope': 1.1,
            'stages': stages,
        }
        filename = f'2026-{name}.joblib'
        joblib.dump(artifact, path / filename)
        fold['models'][name] = {
            'artifact': filename, 'artifact_sha256': hashlib.sha256((path / filename).read_bytes()).hexdigest(),
            'weight': weight, 'final_calibration_slope': 1.1,
        }
    (path / '2026-fold.json').write_text(json.dumps(fold))
    (path / 'protocol.json').write_text(json.dumps({'experiment': 'EXP-089'}))
    return path


def test_freeze_rejects_changed_training_feature_code(source_run, tmp_path, clock):
    source_protocol = {'experiment': 'EXP-089',
                       'source_sha256': {'src/models/symmetric_series.py': '0' * 64}}
    (source_run / 'protocol.json').write_text(json.dumps(source_protocol))
    with pytest.raises(ValueError):
        confirmation.freeze_protocol(source_run, tmp_path / 'changed_feature_contract')


@pytest.fixture
def clock(monkeypatch):
    class Clock(datetime):
        current = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz)

    monkeypatch.setattr(confirmation, 'datetime', Clock)
    return Clock


@pytest.fixture
def protocol(tmp_path, clock, source_run):
    path = tmp_path / 'protocol'
    confirmation.freeze_protocol(source_run, path)
    return path


@pytest.fixture
def inputs(tmp_path, clock):
    rows = []
    for i in range(3):
        row = {field: 0.5 for field in _REQUIRED_BASE_FIELDS}
        row.update(golgg_match_id=f'match-{i}', best_of=3, team_elo=0.55 + 0.1 * i)
        rows.append(row)
    raw = tmp_path / 'sports.csv'
    metadata = tmp_path / 'metadata.csv'
    pd.DataFrame(rows).to_csv(raw, index=False)
    pd.DataFrame({
        'golgg_match_id': [row['golgg_match_id'] for row in rows],
        'match_start_at': [(clock.current + timedelta(days=i + 1)).isoformat() for i in range(3)],
        'feature_history_max_at': [(clock.current - timedelta(hours=1)).isoformat()] * 3,
    }).to_csv(metadata, index=False)
    return raw, metadata


def test_oddsfree_predictions_are_series_probabilities_and_preserve_index(protocol, inputs):
    rows = pd.read_csv(inputs[0]).set_index('golgg_match_id')
    predicted = confirmation.predict_frozen(protocol, rows)
    assert predicted.index.equals(rows.index)
    source_p = rows['team_elo'].to_numpy()
    for name, coefficient in [('outcome', .7), ('market_w10', .8)]:
        expected = 1 / (1 + np.exp(-1.1 * coefficient * np.log(source_p / (1-source_p))))
        np.testing.assert_allclose(predicted['p__'+name], expected, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match='column|odds|unexpected'):
        confirmation.predict_frozen(protocol, rows.assign(odds_a=1.5))


@pytest.mark.parametrize('field,value', [
    ('match_start_at', '2026-09-09T11:59:59+00:00'),
    ('feature_history_max_at', '2026-09-09T12:00:01+00:00'),
    ('match_start_at', '2026-09-10T12:00:00'),
])
def test_capture_rejects_started_events_future_features_and_naive_times(protocol, inputs, tmp_path, field, value):
    metadata = pd.read_csv(inputs[1])
    metadata.loc[0, field] = value
    metadata.to_csv(inputs[1], index=False)
    with pytest.raises(ValueError):
        confirmation.capture_forecasts(protocol, *inputs, tmp_path / 'capture')


def test_capture_rechecks_start_after_inference(protocol, inputs, tmp_path, clock, monkeypatch):
    original = confirmation.predict_frozen

    def slow_prediction(*args):
        result = original(*args)
        clock.current += timedelta(days=1)
        return result

    monkeypatch.setattr(confirmation, 'predict_frozen', slow_prediction)
    with pytest.raises(ValueError, match='start|prospective'):
        confirmation.capture_forecasts(protocol, *inputs, tmp_path / 'capture')


@pytest.mark.parametrize('filename', [
    '2026-outcome.joblib', 'protocol.json', 'source/2026-fold.json',
    'inference_source/src/models/symmetric_series.py',
])
def test_protocol_tampering_prevents_prediction(protocol, inputs, filename):
    path = protocol / filename
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='hash|integrity'):
        confirmation.predict_frozen(protocol, pd.read_csv(inputs[0]))


def test_partial_settlement_counts_pending_without_rescoring_pending_rows(protocol, inputs, tmp_path, clock):
    forecast = tmp_path / 'capture'
    manifest = confirmation.capture_forecasts(protocol, *inputs, forecast)
    clock.current += timedelta(days=2)
    metadata = pd.read_csv(inputs[1])
    outcomes = tmp_path / 'outcomes.csv'
    pd.DataFrame({
        'golgg_match_id': ['match-0'],
        'match_start_at': [metadata.loc[0, 'match_start_at']],
        'y_true': [1],
        'settled_at': [(clock.current - timedelta(hours=1)).isoformat()],
    }).to_csv(outcomes, index=False)
    result = confirmation.score_confirmation(protocol, [forecast], outcomes, tmp_path / 'score')
    assert result['settled_count'] == 1
    assert result['pending_count'] == 2
    assert result['cohort_ids'] == ['match-0']
    assert result['metrics']['outcome']['n'] == 1
    assert result['paired']['status'] == 'unavailable_fewer_than_two_months'
    assert result['promotion'] is False
    predictions = pd.read_csv(forecast / 'predictions.csv')
    assert result['metrics']['outcome']['log_loss'] == pytest.approx(-np.log(predictions.loc[0, 'p__outcome']))
    with pytest.raises(FileExistsError):
        confirmation.capture_forecasts(protocol, *inputs, forecast)


@pytest.mark.parametrize('filename', ['raw_rows.csv', 'metadata.csv', 'predictions.csv'])
def test_scoring_rejects_changed_archived_bytes(protocol, inputs, tmp_path, clock, filename):
    forecast = tmp_path / 'capture'
    confirmation.capture_forecasts(protocol, *inputs, forecast)
    path = forecast / filename
    path.write_bytes(path.read_bytes() + b'\n')
    outcomes = tmp_path / 'outcomes.csv'
    pd.DataFrame(columns=['golgg_match_id', 'match_start_at', 'y_true', 'settled_at']).to_csv(outcomes, index=False)
    with pytest.raises(ValueError, match='hash|integrity'):
        confirmation.score_confirmation(protocol, [forecast], outcomes, tmp_path / 'score')


def test_scoring_recomputes_even_if_prediction_hash_is_resealed(protocol, inputs, tmp_path):
    forecast = tmp_path / 'capture'
    confirmation.capture_forecasts(protocol, *inputs, forecast)
    path = forecast / 'predictions.csv'
    predictions = pd.read_csv(path)
    predictions.loc[0, 'p__outcome'] = 0.123456789
    predictions.to_csv(path, index=False)
    manifest_path = forecast / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['predictions.csv'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    (forecast / 'manifest.sha256').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + '\n')
    outcomes = tmp_path / 'outcomes.csv'
    pd.DataFrame(columns=['golgg_match_id', 'match_start_at', 'y_true', 'settled_at']).to_csv(outcomes, index=False)
    with pytest.raises(ValueError, match='prediction|recomput'):
        confirmation.score_confirmation(protocol, [forecast], outcomes, tmp_path / 'score')


def test_empty_settlement_reports_unavailable_not_success(protocol, inputs, tmp_path):
    forecast = tmp_path / 'capture'
    confirmation.capture_forecasts(protocol, *inputs, forecast)
    outcomes = tmp_path / 'outcomes.csv'
    pd.DataFrame(columns=['golgg_match_id', 'match_start_at', 'y_true', 'settled_at']).to_csv(outcomes, index=False)
    result = confirmation.score_confirmation(protocol, [forecast], outcomes, tmp_path / 'score')
    assert result['settled_count'] == 0
    assert result['pending_count'] == 3
    assert result['paired']['log_loss']['mean'] is None


@pytest.mark.parametrize('problem', ['unknown_id', 'duplicate_id', 'future_settlement', 'before_start', 'changed_start', 'nonbinary'])
def test_invalid_settlements_reject_entire_score(protocol, inputs, tmp_path, clock, problem):
    forecast = tmp_path / 'capture'
    confirmation.capture_forecasts(protocol, *inputs, forecast)
    clock.current += timedelta(days=3)
    start = pd.read_csv(inputs[1]).loc[0, 'match_start_at']
    row = {
        'golgg_match_id': 'match-0', 'match_start_at': start, 'y_true': 1,
        'settled_at': (clock.current - timedelta(hours=1)).isoformat(),
    }
    if problem == 'unknown_id':
        row['golgg_match_id'] = 'unknown'
    elif problem == 'future_settlement':
        row['settled_at'] = (clock.current + timedelta(seconds=1)).isoformat()
    elif problem == 'before_start':
        row['settled_at'] = (pd.Timestamp(start) - timedelta(seconds=1)).isoformat()
    elif problem == 'changed_start':
        row['match_start_at'] = (pd.Timestamp(start) + timedelta(hours=1)).isoformat()
    elif problem == 'nonbinary':
        row['y_true'] = 0.5
    outcomes = tmp_path / 'outcomes.csv'
    pd.DataFrame([row, row] if problem == 'duplicate_id' else [row]).to_csv(outcomes, index=False)
    with pytest.raises(ValueError):
        confirmation.score_confirmation(protocol, [forecast], outcomes, tmp_path / 'score')


def test_missing_metadata_row_is_not_silently_filtered(protocol, inputs, tmp_path):
    metadata = pd.read_csv(inputs[1]).iloc[:2]
    metadata.to_csv(inputs[1], index=False)
    with pytest.raises(ValueError, match='every|match'):
        confirmation.capture_forecasts(protocol, *inputs, tmp_path / 'capture')


@pytest.mark.parametrize('field,value', [
    ('target', 'map'), ('requires_odds_at_inference', True), ('weight', 0.3),
    ('scale', np.array([np.inf])), ('slope', np.nan),
])
def test_invalid_model_contract_rejected_even_with_matching_source_hash(source_run, tmp_path, clock, field, value):
    path = source_run / '2026-market_w10.joblib'
    artifact = joblib.load(path)
    artifact[field] = value
    joblib.dump(artifact, path)
    fold_path = source_run / '2026-fold.json'
    fold = json.loads(fold_path.read_text())
    fold['models']['market_w10']['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    fold_path.write_text(json.dumps(fold))
    with pytest.raises(ValueError):
        confirmation.freeze_protocol(source_run, tmp_path / 'invalid_protocol')


def test_protocol_rejects_capture_before_its_real_lock(protocol, inputs, tmp_path, clock):
    clock.current -= timedelta(seconds=1)
    with pytest.raises(ValueError, match='future|lock'):
        confirmation.capture_forecasts(protocol, *inputs, tmp_path / 'capture')


def test_confirmation_accumulates_disjoint_monthly_batches(protocol, inputs, tmp_path, clock):
    first, second = tmp_path/'first', tmp_path/'second'
    confirmation.capture_forecasts(protocol, *inputs, first)
    first_metadata = pd.read_csv(inputs[1])
    raw = pd.read_csv(inputs[0])
    raw['golgg_match_id'] = raw['golgg_match_id'] + '-later'
    raw.to_csv(inputs[0],index=False)
    later_metadata = first_metadata.copy()
    later_metadata['golgg_match_id'] += '-later'
    later_metadata['match_start_at'] = (pd.to_datetime(later_metadata.match_start_at)+timedelta(days=36)).map(lambda t:t.isoformat())
    later_metadata.to_csv(inputs[1],index=False)
    confirmation.capture_forecasts(protocol, *inputs, second)
    clock.current += timedelta(days=45)
    outcomes = pd.concat([first_metadata, later_metadata],ignore_index=True).drop(columns='feature_history_max_at')
    outcomes['y_true'] = [0,1,1,1,0,1]
    outcomes['settled_at'] = (pd.to_datetime(outcomes.match_start_at)+timedelta(hours=2)).map(lambda t:t.isoformat())
    outcome_path = tmp_path/'outcomes.csv'
    outcomes.to_csv(outcome_path,index=False)
    result = confirmation.score_confirmation(protocol,[first,second],outcome_path,tmp_path/'score')
    assert result['settled_count'] == 6 and result['pending_count'] == 0
    assert result['paired']['months'] == 2
    assert result['paired']['log_loss']['resamples'] >= 5000
    with pytest.raises(ValueError):
        confirmation.score_confirmation(protocol,[first,first],outcome_path,tmp_path/'duplicated')
