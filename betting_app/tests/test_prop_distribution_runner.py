"""Frozen inference and supported task absences are consumer contracts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import prop_distribution_runner as runner


def frozen_fixture(tmp_path, monkeypatch, n=60):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    run_id = 'fixture'
    paths = runner._paths(run_id)
    for path in paths.values():
        path.mkdir(parents=True)
    i = np.arange(n)
    bo = np.array([1, 3, 5])[i % 3]
    frame = pd.DataFrame({'match_id': i.astype(str), 'date': (pd.Timestamp('2010-01-01') + pd.to_timedelta(i, unit='D')).strftime('%Y-%m-%d'), 'best_of': bo, 'kills': 20. + i % 15, 'duration_minutes': 25. + i % 12, 'maps_played': bo//2 + 1 + np.where(bo==1, 0, np.where(bo==3, (i//3)%2, (i//3)%3)), 'x_pace': (i%7).astype(float)})
    frame['release_day'] = frame['date']
    dataset_path = paths['04_feature'] / 'dataset.parquet'
    frame.to_parquet(dataset_path, index=False)
    protocol = {'fixed_lines': {'kills': [28.5], 'duration_minutes': [31.5], 'length': [2.5]}}
    protocol_path = paths['05_model_input'] / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol))
    binding = {'dataset_path': str(dataset_path.relative_to(tmp_path)), 'feature_columns': ['x_pace'], 'pins': {str(p.relative_to(tmp_path)): runner.sha256(p) for p in (protocol_path,dataset_path)}, 'protected_hashes': {}}
    (paths['05_model_input'] / 'binding.json').write_text(json.dumps(binding))
    return run_id, paths, frame, protocol_path


def test_saved_inference_rejects_changed_frozen_protocol(tmp_path, monkeypatch):
    run_id, paths, frame, protocol_path = frozen_fixture(tmp_path, monkeypatch)
    model = runner._fit('pooled', frame.iloc[:40], frame.iloc[40:], ['x_pace'], 20261007, {})
    selected = runner.SelectedDistributionModel({task: 'pooled' for task in runner.TASKS}, {'pooled': model}, ['x_pace'])
    model_path = paths['06_models'] / 'selected_model.joblib'
    joblib.dump(selected, model_path)
    (paths['08_reporting'] / 'decision.json').write_text(json.dumps({'output_hashes': {str(model_path.relative_to(tmp_path)): runner.sha256(model_path)}}))
    protocol = json.loads(protocol_path.read_text())
    protocol['fixed_lines']['kills'] = [36.5]
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='Frozen input/code changed'):
        runner.predict_saved_row(run_id, '0')


def test_real_four_family_smoke_accepts_missing_task_labels(tmp_path, monkeypatch):
    run_id, paths, frame, protocol_path = frozen_fixture(tmp_path, monkeypatch, n=3000)
    frame.loc[::19, 'kills'] = np.nan
    frame.loc[::23, 'duration_minutes'] = np.nan
    frame.loc[::29, 'maps_played'] = np.nan
    dataset_path = paths['04_feature'] / 'dataset.parquet'
    frame.to_parquet(dataset_path, index=False)
    binding_path = paths['05_model_input'] / 'binding.json'
    binding = json.loads(binding_path.read_text())
    binding['pins'][str(dataset_path.relative_to(tmp_path))] = runner.sha256(dataset_path)
    binding_path.write_text(json.dumps(binding))
    result = runner.run_smoke(run_id)
    assert result['source_partition'] == 'TRAIN_ONLY'
    assert result['assessment_rows_used'] == 0
    for family in result['families']:
        for task in runner.TASKS:
            assert family['fit_receipt']['task_support'][task]['cal'] < result['cal_rows']
