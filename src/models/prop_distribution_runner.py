"""Frozen historical experiment and saved-row inference for sports prop marginals."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analysis.prop_distribution_evaluation import paired_comparison, profile_scores, score_prediction
from src.features.prop_distribution_data import build_distribution_dataset
from src.models.prop_distribution import DistributionMixture, distribution_nll, fit_distribution_model

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('kills', 'duration', 'length')
CODE_PATHS = ('src/features/prop_distribution_data.py', 'src/models/prop_distribution.py', 'src/models/prop_distribution_runner.py', 'src/analysis/prop_distribution_evaluation.py', 'scripts/train_prop_distribution.py', 'src/models/prematch_features.py', 'scripts/export_prospective_features.py', 'src/analysis/bootstrap.py')


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n', encoding='utf-8')


def _paths(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Run ID must be a safe single path component.')
    return {stage: ROOT / f'data/{stage}/prop_distribution/{run_id}' for stage in ('04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting')}


def partition_rows(frame):
    dates, releases = frame['date'].astype(str), frame['release_day'].astype(str)
    return {'TRAIN': (dates < '2025-07-01') & (releases < '2025-07-01'), 'CAL': (dates >= '2025-07-01') & (dates < '2026-01-01') & (releases < '2026-01-01'), 'ASSESS': (dates >= '2026-01-01') & (dates < '2027-01-01')}


def freeze_run(run_id):
    paths = _paths(run_id)
    protocol_path = paths['05_model_input'] / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    binding_path = paths['05_model_input'] / 'binding.json'
    if binding_path.exists():
        raise FileExistsError('An existing freeze is immutable; use its binding or a new run ID.')
    receipt = build_distribution_dataset(ROOT / protocol['source'], ROOT / protocol['feature_dir'], paths['04_feature'])
    dataset_path = Path(receipt['dataset_path'])
    frame = pd.read_parquet(dataset_path)
    counts = {key: int(mask.sum()) for key, mask in partition_rows(frame).items()}
    for key, count in counts.items():
        if count != protocol['partitions'][key]['audited_rows']:
            raise ValueError(f'{key} support differs from the frozen audit: {count}')
    features = [c for c in frame if c.startswith('x_')]
    if not features or frame['match_id'].duplicated().any():
        raise ValueError('Features missing or duplicate target IDs.')
    pin_paths = [ROOT / p for p in CODE_PATHS] + [protocol_path, dataset_path, ROOT / protocol['source'], ROOT / protocol['feature_dir'] / 'metadata.parquet', ROOT / protocol['feature_dir'] / 'arrays.npz', ROOT / protocol['feature_dir'] / 'audit.json']
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in pin_paths}
    protected = [ROOT / 'conf/base/research_benchmark.json', ROOT / 'conf/base/c0_serving.json', ROOT / 'betting_app/core/models/registry.py']
    c0_root = ROOT / 'data/06_models/a0_data_focus/a0_data_focus_20260929_192443'
    protected += list((c0_root / 'control').glob('**/model.joblib*'))
    protected += list((c0_root / 'shared').glob('model.joblib*'))
    binding = {'run_id': run_id, 'frozen_at': datetime.now(timezone.utc).isoformat(), 'pins': pins, 'protected_hashes': {str(p.relative_to(ROOT)): sha256(p) for p in protected}, 'dataset_path': str(dataset_path.relative_to(ROOT)), 'feature_columns': features, 'partition_counts': counts, 'data_receipt': receipt, 'status': 'FROZEN_BEFORE_FORMAL_FITS'}
    _json(binding_path, binding)
    return binding


def _load_frozen(run_id):
    paths = _paths(run_id)
    binding = json.loads((paths['05_model_input'] / 'binding.json').read_text())
    for relative, expected in binding['pins'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Frozen input/code changed: {relative}')
    for relative, expected in binding['protected_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Protected C0/canonical input changed: {relative}')
    protocol = json.loads((paths['05_model_input'] / 'protocol.json').read_text())
    frame = pd.read_parquet(ROOT / binding['dataset_path'])
    return paths, protocol, binding, frame


def _fit(family, train, cal, features, seed, config):
    return fit_distribution_model(family, train[features].to_numpy(float), train['best_of'].to_numpy(int), train['kills'].to_numpy(float), train['duration_minutes'].to_numpy(float), train['maps_played'].to_numpy(float), X_cal=cal[features].to_numpy(float), best_of_cal=cal['best_of'].to_numpy(int), kills_cal=cal['kills'].to_numpy(float), duration_cal=cal['duration_minutes'].to_numpy(float), maps_cal=cal['maps_played'].to_numpy(float), seed=seed, config=config)


class SeedMixtureModel:
    """Averaging actual component distributions, never mean NB parameters."""
    def __init__(self, components):
        self.components = tuple(components)

    def predict(self, X, best_of):
        return DistributionMixture(tuple(model.predict(X, best_of) for model in self.components))


class SelectedDistributionModel:
    """Independent CAL-selected marginals; intentionally not a joint law."""
    def __init__(self, selected, models, feature_columns):
        self.selected = dict(selected)
        self.models = {key: models[key] for key in set(selected.values())}
        self.feature_columns = list(feature_columns)

    def predict(self, frame):
        X = frame[self.feature_columns].to_numpy(float)
        best_of = frame['best_of'].to_numpy(int)
        cached = {key: model.predict(X, best_of) for key, model in self.models.items()}
        return {task: cached[key] for task, key in self.selected.items()}


def _prediction_frame(candidate, pred, frame):
    components = pred.components if hasattr(pred, 'components') else [pred]
    weights = pred.weights if hasattr(pred, 'weights') else [1.]
    base = frame[['match_id', 'date', 'best_of', 'kills', 'duration_minutes', 'maps_played']].copy().reset_index(drop=True)
    records = []
    for index, (weight, component) in enumerate(zip(weights, components)):
        row = base.copy()
        row['candidate'] = candidate
        row['component'] = index
        row['weight'] = weight
        for name in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma'):
            row[name] = getattr(component, name)
        for length in range(3):
            row[f'length_p_{length}'] = component.length_probs[:, length]
        records.append(row)
    return pd.concat(records, ignore_index=True)


def run_smoke(run_id):
    """Exercise real TRAIN data only; no primary assessment peek or formal recipe."""
    paths, protocol, binding, frame = _load_frozen(run_id)
    train_all = frame.loc[partition_rows(frame)['TRAIN']]
    sample = train_all.sample(min(3000, len(train_all)), random_state=20261007).sort_values(['date', 'match_id'])
    train, cal = sample.iloc[:2200], sample.iloc[2200:]
    output = paths['08_reporting'] / 'software_smoke.json'
    if output.exists():
        raise FileExistsError('Smoke receipt already exists.')
    receipts = []
    for family, config in [('pooled', {}), ('glm', {'ridge': 1.}), ('tree', {'n_estimators': 10, 'num_leaves': 15, 'min_child_samples': 20, 'n_jobs': 4}), ('neural', {'hidden': 64, 'epochs': 1})]:
        model = _fit(family, train, cal, binding['feature_columns'], 20261007, config)
        pred = model.predict(cal[binding['feature_columns']].to_numpy(float), cal['best_of'].to_numpy(int))
        losses = distribution_nll(pred, cal['kills'].to_numpy(float), cal['duration_minutes'].to_numpy(float), cal['maps_played'].to_numpy(float))
        labels = {'kills': cal['kills'].to_numpy(float), 'duration': cal['duration_minutes'].to_numpy(float), 'length': cal['maps_played'].to_numpy(float)}
        for task, value in losses.items():
            if not np.isfinite(value[np.isfinite(labels[task])]).all():
                raise ValueError('Smoke produced nonfinite observed scores.')
        if not np.allclose(pred.length_probs.sum(axis=1), 1):
            raise ValueError('Smoke legal-length mass is invalid.')
        receipts.append({'family': family, 'fit_receipt': model.fit_receipt, 'finite_score_rows': len(cal)})
    receipt = {'status': 'SOFTWARE_ONLY', 'source_partition': 'TRAIN_ONLY', 'train_rows': len(train), 'cal_rows': len(cal), 'smoke_fits': 4, 'formal_fits': 0, 'assessment_rows_used': 0, 'families': receipts}
    _json(output, receipt)
    return receipt


def run_experiment(run_id):
    paths, protocol, binding, frame = _load_frozen(run_id)
    paths['06_models'].mkdir(parents=True, exist_ok=False)
    paths['07_model_output'].mkdir(parents=True, exist_ok=False)
    paths['08_reporting'].mkdir(parents=True, exist_ok=True)
    partitions = {key: frame.loc[mask].reset_index(drop=True) for key, mask in partition_rows(frame).items()}
    train, cal, assess = (partitions[key] for key in ('TRAIN', 'CAL', 'ASSESS'))
    features = binding['feature_columns']
    models, receipts = {}, []
    for recipe in protocol['candidate_recipes']:
        started = time.monotonic()
        family, name, config = recipe['family'], recipe['id'], recipe['config']
        print(f'FIT {name}', flush=True)
        if family != 'neural':
            model = _fit(family, train, cal, features, 20261007, config)
            models[name] = model
            joblib.dump(model, paths['06_models'] / f'{name}.joblib', compress=3)
            receipts.append({'candidate': name, 'elapsed_seconds': time.monotonic()-started, 'fit_receipt': model.fit_receipt})
        else:
            seeds12, seeds24 = [], []
            for seed in recipe['seeds']:
                print(f'FIT {name} seed={seed}', flush=True)
                model = _fit(family, train, cal, features, seed, config)
                if 12 not in model.checkpoints:
                    raise ValueError('Continuous neural trajectory lacks frozen epoch12 endpoint.')
                seeds12.append(model.checkpoints[12]); seeds24.append(model)
                joblib.dump(model, paths['06_models'] / f'{name}_seed_{seed}.joblib', compress=3)
                receipts.append({'candidate': name, 'seed': seed, 'elapsed_seconds': time.monotonic()-started, 'fit_receipt': model.fit_receipt})
            models[f'{name}_epoch_12'] = SeedMixtureModel(seeds12)
            models[f'{name}_epoch_24'] = SeedMixtureModel(seeds24)
        _json(paths['08_reporting'] / 'fit_receipts.json', receipts)
    # Candidate and matched GLM selection uses CAL, before any ASSESS predictions.
    cal_scores = {}
    cal_predictions = []
    for name, model in models.items():
        pred = model.predict(cal[features].to_numpy(float), cal['best_of'].to_numpy(int))
        losses = distribution_nll(pred, cal['kills'].to_numpy(float), cal['duration_minutes'].to_numpy(float), cal['maps_played'].to_numpy(float))
        cal_scores[name] = {task: float(np.nanmean(np.where(cal['best_of'].to_numpy() > 1, losses[task], np.nan) if task == 'length' else losses[task])) for task in TASKS}
        cal_predictions.append(_prediction_frame(name, pred, cal))
    selected = {task: min(models, key=lambda name: (cal_scores[name][task], name)) for task in TASKS}
    glm_selected = {task: min((name for name in models if name.startswith('glm_')), key=lambda name: (cal_scores[name][task], name)) for task in TASKS}
    selection = {'selected': selected, 'glm_controls': glm_selected, 'cal_scores': cal_scores, 'selected_at': datetime.now(timezone.utc).isoformat(), 'assessment_predictions_generated': False}
    _json(paths['06_models'] / 'selection.json', selection)
    pd.concat(cal_predictions, ignore_index=True).to_parquet(paths['07_model_output'] / 'cal_distribution_components.parquet', index=False)
    joblib.dump(SelectedDistributionModel(selected, models, features), paths['06_models'] / 'selected_model.joblib', compress=3)
    metrics, score_rows, predictions, line_rows, profiles = {}, {}, [], [], []
    for name, model in models.items():
        print(f'ASSESS {name}', flush=True)
        pred = model.predict(assess[features].to_numpy(float), assess['best_of'].to_numpy(int))
        metric, rows, lines = score_prediction(pred, assess, protocol['fixed_lines'])
        metrics[name], score_rows[name] = metric, rows
        predictions.append(_prediction_frame(name, pred, assess))
        line_rows.extend(dict(candidate=name, **line) for line in lines)
        profiles.extend(profile_scores(name, rows, assess))
        rows.to_parquet(paths['07_model_output'] / f'{name}_scores.parquet', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(paths['07_model_output'] / 'distribution_components.parquet', index=False)
    pd.DataFrame(line_rows).to_csv(paths['08_reporting'] / 'line_diagnostics.csv', index=False)
    pd.DataFrame(profiles).to_csv(paths['08_reporting'] / 'profiles.csv', index=False)
    decisions, comparisons = {}, {}
    confidence = protocol['uncertainty']['primary_confidence_level']
    for task in TASKS:
        winner = selected[task]
        comparisons[task] = {}
        secondary = 'rps' if task == 'length' else 'crps'
        pass_gate = True
        for control in ['pooled', glm_selected[task]]:
            comparison = paired_comparison(score_rows[winner], score_rows[control], task, confidence=confidence)
            comparison['candidate'] = winner; comparison['control'] = control
            comparisons[task][control] = comparison
            pass_gate &= comparison['ci'][1] < 0 and metrics[winner][task][secondary] <= metrics[control][task][secondary]
        decisions[task] = {'selected': winner, 'glm_control': glm_selected[task], 'status': 'DEVELOPMENT_CANDIDATE' if pass_gate else 'NO_ESTABLISHED_INCREMENTAL_GAIN', 'selected_metrics': metrics[winner][task]}
    _json(paths['08_reporting'] / 'metrics.json', metrics)
    _json(paths['08_reporting'] / 'comparisons.json', comparisons)
    protected_unchanged = {path: sha256(ROOT / path) == expected for path, expected in binding['protected_hashes'].items()}
    if not all(protected_unchanged.values()):
        raise ValueError('Protected C0/canonical manifest changed during experiment.')
    output_hashes = {str(p.relative_to(ROOT)): sha256(p) for stage in ('06_models', '07_model_output', '08_reporting') for p in sorted(paths[stage].glob('*')) if p.is_file()}
    decision = {'run_id': run_id, 'status': 'COMPLETED', 'scope': 'HISTORICAL_DEVELOPMENT_ONLY', 'tasks': decisions, 'formal_fits': len(receipts), 'continuous_neural_fits': sum('seed' in r for r in receipts), 'assessment_refits': 0, 'c0_fits': 0, 'partition_counts': {key: len(value) for key, value in partitions.items()}, 'candidates': len(models), 'protected_unchanged': protected_unchanged, 'output_hashes': output_hashes, 'environment': {package: importlib.metadata.version(package) for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'torch', 'joblib')}, 'python': platform.python_version(), 'threads': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}, 'completed_at': datetime.now(timezone.utc).isoformat(), 'limitations': protocol['limitations']}
    _json(paths['08_reporting'] / 'decision.json', decision)
    return decision


def predict_saved_row(run_id, match_id):
    paths, protocol, binding, frame = _load_frozen(run_id)
    selection_path = paths['06_models'] / 'selected_model.joblib'
    decision = json.loads((paths['08_reporting'] / 'decision.json').read_text())
    for path in (selection_path, ROOT / binding['dataset_path']):
        expected = decision['output_hashes'].get(str(path.relative_to(ROOT)), binding['pins'].get(str(path.relative_to(ROOT))))
        if expected is None or sha256(path) != expected:
            raise ValueError(f'Untrusted saved-row input: {path.name}')
    row = frame.loc[frame['match_id'].astype(str) == str(match_id)]
    if len(row) != 1:
        raise ValueError('Match ID must identify exactly one retained historical row.')
    model = joblib.load(selection_path)
    preds = model.predict(row)
    result = {'run_id': run_id, 'match_id': str(match_id), 'date': str(row.iloc[0]['date']), 'best_of': int(row.iloc[0]['best_of']), 'scenario': 'historical_actual_first_map_roster_daily_release', 'selected': model.selected, 'joint_distribution': False}
    for task, pred in preds.items():
        lines = protocol['fixed_lines']['duration_minutes' if task == 'duration' else task]
        fn = getattr(pred, {'kills': 'kills_over', 'duration': 'duration_over', 'length': 'maps_over'}[task])
        result[task] = {'p_over': {str(line): float(fn(line)[0]) for line in lines}}
    length_pred = preds['length']
    comps = length_pred.components if hasattr(length_pred, 'components') else [length_pred]
    weights = length_pred.weights if hasattr(length_pred, 'weights') else [1.]
    probs = sum(w * c.length_probs[0] for w, c in zip(weights, comps))
    needed = result['best_of'] // 2 + 1
    result['length']['pmf'] = {str(needed + i): float(probs[i]) for i in range(result['best_of'] - needed + 1)}
    return result
