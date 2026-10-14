"""Frozen LR/MLP correction experiment; never changes historical selection or ratings.

Reuses the thesis feature replay, monthly training, calibration and strict joins.
This runner does not tune candidates or define a competing metric implementation.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import platform
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from src.analysis import thesis_models
from src.analysis.thesis_audit import add_series_features, feature_variants, rating_stream
from src.analysis.thesis_evaluation import block_difference, join_prediction_streams
from src.analysis.thesis_metrics import _metrics
from src.analysis.metrics import binary_log_loss_vector

SEEDS = (7, 42, 2026)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def run_variant(job, output):
    """Instrument existing replay without changing fitting or calibration behavior."""
    directory = Path(output) / job['name']
    directory.mkdir(parents=True, exist_ok=False)
    data = add_series_features(pd.read_parquet(job['features_path']))
    columns = feature_variants()['w20']
    original_factory = thesis_models.make_model
    captured = {}
    coefficients, diagnostics = [], []
    start_wall, start_cpu = perf_counter(), process_time()
    previous = [start_wall, start_cpu]

    def factory(*args, **kwargs):
        pipeline = original_factory(*args, **kwargs)
        captured['pipeline'] = pipeline
        return pipeline

    def progress(receipt):
        pipeline = captured['pipeline']
        estimator = pipeline.named_steps['model']
        wall, cpu = perf_counter(), process_time()
        diagnostic = {
            'variant': job['name'], 'origin': receipt['origin'],
            'family': job['family'], 'train_rows': receipt['train_rows'],
            'test_rows': receipt['test_rows'],
            'step_wall_seconds': wall - previous[0],
            'step_cpu_seconds': cpu - previous[1],
            'iterations': int(np.max(estimator.n_iter_)),
            'calibration_slope': receipt['calibration_slope'],
        }
        previous[:] = [wall, cpu]
        if job['family'] == 'logistic':
            weights = estimator.coef_.ravel()
            scaler = pipeline.named_steps['scaler']
            diagnostic['parameter_count'] = int(estimator.coef_.size + estimator.intercept_.size)
            diagnostic['nonzero_coefficients'] = int(np.count_nonzero(np.abs(weights) > 1e-10))
            for index, feature in enumerate(columns):
                coefficients.append({
                    'variant': job['name'], 'origin': receipt['origin'], 'feature': feature,
                    'standardized_coefficient': float(weights[index]),
                    'scale': float(scaler.scale_[index]), 'mean': float(scaler.mean_[index]),
                    'intercept': float(estimator.intercept_[0]),
                })
        else:
            diagnostic['parameter_count'] = int(sum(x.size for x in estimator.coefs_ + estimator.intercepts_))
            diagnostic['training_loss'] = float(estimator.loss_)
            validation = getattr(estimator, 'best_validation_score_', None)
            diagnostic['best_validation_accuracy'] = float(validation) if validation is not None else None
        diagnostics.append(diagnostic)
        print(json.dumps({k: diagnostic[k] for k in ('variant', 'origin', 'step_cpu_seconds')}), flush=True)

    thesis_models.make_model = factory
    try:
        predictions, receipts = thesis_models.replay_monthly(
            data, columns, job['family'], job['params'], start='2024-01-01',
            calibration_start='2024-01-01', progress=progress,
        )
    finally:
        thesis_models.make_model = original_factory
    predictions.to_parquet(directory / 'predictions.parquet', index=False)
    save_json(directory / 'fit_receipts.json', receipts)
    pd.DataFrame(diagnostics).to_csv(directory / 'fit_diagnostics.csv', index=False)
    if coefficients:
        pd.DataFrame(coefficients).to_csv(directory / 'coefficients.csv', index=False)
    receipt = {
        'variant': job['name'], 'rows': len(predictions), 'folds': len(receipts),
        'wall_seconds': perf_counter() - start_wall, 'cpu_seconds': process_time() - start_cpu,
        'prediction_sha256': sha(directory / 'predictions.parquet'),
        'fit_receipt_sha256': sha(directory / 'fit_receipts.json'),
    }
    save_json(directory / 'COMPLETE.json', receipt)
    return receipt


def verify_streams(jobs, output, feature_path, archive):
    metadata = pd.read_parquet(feature_path).set_index('golgg_match_id')
    metadata.index = metadata.index.astype(str)
    old_receipts = json.loads((archive / 'corrected_replay_resumed/models/logistic/fit_receipts.json').read_text())
    by_origin = {r['origin']: r for r in old_receipts}
    streams, evidence = {}, []
    for job in jobs:
        directory = output / job['name']
        predictions = pd.read_parquet(directory / 'predictions.parquet')
        receipts = json.loads((directory / 'fit_receipts.json').read_text())
        if len(receipts) != 33 or len(predictions) != 11550:
            raise ValueError(f'Incomplete replay: {job["name"]}')
        for row in receipts:
            original = by_origin[row['origin']]
            for kind in ('train', 'test', 'cal'):
                if row[f'{kind}_ids'] != original[f'{kind}_ids']:
                    raise ValueError(f'Membership changed: {job["name"]} {kind} {row["origin"]}')
                if kind != 'test' and row[f'{kind}_ids']:
                    subset = metadata.loc[list(map(str, row[f'{kind}_ids']))]
                    origin = pd.Timestamp(row['origin'])
                    if not ((pd.to_datetime(subset.date) < origin) & (pd.to_datetime(subset.result_day) < origin)).all():
                        raise ValueError('Future TRAIN/CAL labels')
        raw = np.clip(.5 * (predictions.p_original + 1 - predictions.p_swapped), 1e-6, 1 - 1e-6)
        np.testing.assert_allclose(raw, predictions.p_raw, atol=1e-14, rtol=0)
        np.testing.assert_allclose(expit(predictions.calibration_slope * logit(raw)), predictions.p_calibrated, atol=1e-14, rtol=0)
        original_error = None
        if job['name'] in ('lr_original_seed42', 'mlp_original_seed42'):
            family = 'logistic' if job['family'] == 'logistic' else 'mlp'
            previous = pd.read_parquet(archive / f'corrected_replay_resumed/models/{family}/predictions.parquet')
            previous.golgg_match_id = previous.golgg_match_id.astype(str)
            current = predictions.assign(golgg_match_id=predictions.golgg_match_id.astype(str)).set_index('golgg_match_id')
            previous = previous.set_index('golgg_match_id').loc[current.index]
            original_error = float(np.max(np.abs(previous.p_calibrated - current.p_calibrated)))
            np.testing.assert_allclose(current.p_calibrated, previous.p_calibrated, atol=1e-12, rtol=0)
        streams[job['name']] = predictions
        streams[job['name'] + '_raw'] = predictions.assign(p_calibrated=predictions.p_raw)
        evidence.append({'variant': job['name'], 'folds': len(receipts), 'rows': len(predictions),
                         'memberships_equal_original': True, 'original_seed42_max_error': original_error})
    return streams, evidence


def development_evidence(feature_path, selection, output):
    """Reconstruct the historical 2023 comparison, not a new selection."""
    data = add_series_features(pd.read_parquet(feature_path))
    dates, released = pd.to_datetime(data.date), pd.to_datetime(data.result_day)
    earlier = data[(dates < '2024-01-01') & (released < '2024-01-01')].copy()
    earlier = thesis_models._prepare(
        earlier, feature_variants()['w20'], 'golgg_match_id', 'date', 'result_day', 'y_true',
    )
    train = earlier[(earlier.date < '2023-01-01') & (earlier.result_day < '2023-01-01')]
    validation = earlier[earlier.date >= '2023-01-01']
    if thesis_models._membership(train.golgg_match_id) != selection['train_ids_sha256']:
        raise ValueError('Development TRAIN membership drift')
    if thesis_models._membership(validation.golgg_match_id) != selection['validation_ids_sha256']:
        raise ValueError('Development validation membership drift')
    result = validation[['golgg_match_id', 'date', 'result_day', 'y_true']].copy()
    scores = {}
    for family in ('logistic', 'mlp'):
        original, opposite, messages = thesis_models._fit_predict(
            train, validation, feature_variants()['w20'], family,
            selection['selected'][family], 'y_true', 42,
        )
        p = np.clip(.5 * (original + 1 - opposite), 1e-6, 1 - 1e-6)
        result[f'p_{family}'] = p
        metric, _ = _metrics(validation.y_true.to_numpy(float), p)
        previous = min(r['logloss'] for r in selection['candidates'] if r['family'] == family)
        np.testing.assert_allclose(metric['logloss'], previous, atol=1e-12, rtol=0)
        scores[family] = {'metrics': metric, 'warnings': messages}
    delta = binary_log_loss_vector(result.y_true, result.p_logistic) - binary_log_loss_vector(result.y_true, result.p_mlp)
    evidence = {
        'status': 'HISTORICAL_SELECTION_RECONSTRUCTION_NOT_NEW_HOLDOUT',
        'train_rows': len(train), 'validation_rows': len(validation), 'scores': scores,
        'lr_minus_mlp_monthly': block_difference(delta, result.date.dt.strftime('%Y-%m')),
        'selection_unchanged': selection['primary_family'],
    }
    result.to_parquet(output / 'development_predictions.parquet', index=False)
    save_json(output / 'development_comparison.json', evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--run-name', default='thesis_lr_corrections_20260924')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('run-name must be one new directory name')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    research = args.research_root or Path((ROOT / 'data/research_root.txt').read_text().strip())
    archive = research / 'data/08_reporting/thesis_audit_20260921'
    original_features = research / 'data/04_feature/thesis_audit_20260921/features.parquet'
    selection_path = archive / 'corrected_replay_resumed/selection.json'
    original_panel = archive / 'evaluation_final/paired_predictions.parquet'
    bank = research / 'a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank'
    raw = ROOT / 'data/artifacts/golgg-database-recovery-20260909/matches.json'
    feature_output = ROOT / 'data/04_feature' / args.run_name
    prediction_output = ROOT / 'data/07_model_output' / args.run_name
    output = ROOT / 'data/08_reporting' / args.run_name
    for path in (feature_output, prediction_output, output):
        if path.exists():
            raise FileExistsError(path)
    selection = json.loads(selection_path.read_text())
    paths = [raw, original_features, selection_path, original_panel, bank / 'metadata.parquet',
             bank / 'release_days.json', Path(__file__), ROOT / 'src/analysis/thesis_corrections.py',
             *[ROOT / f'src/analysis/{name}.py' for name in ('thesis_models', 'thesis_features', 'thesis_audit', 'thesis_evaluation')],
             ROOT / 'src/ratings/trueskill_rating.py']
    paths.extend(archive / 'corrected_replay_resumed/models' / family / filename
                 for family in ('logistic', 'mlp')
                 for filename in ('predictions.parquet', 'fit_receipts.json'))
    sources = {str(p.resolve()): sha(p) for p in paths}
    jobs = []
    for version, feature_path in [('original', original_features), ('ts_standard', feature_output / 'features.parquet')]:
        for short, family in [('lr', 'logistic'), ('mlp', 'mlp')]:
            for seed in SEEDS:
                jobs.append({'name': f'{short}_{version}_seed{seed}', 'family': family,
                             'features_path': str(feature_path),
                             'params': {**selection['selected'][family], 'random_state': seed}})
        jobs.append({'name': f'mlp_{version}_no_early_stopping', 'family': 'mlp',
                     'features_path': str(feature_path),
                     'params': {**selection['selected']['mlp'], 'early_stopping': False}})
    output.mkdir(parents=True)
    prediction_output.mkdir(parents=True)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'FROZEN_BEFORE_NEW_FITS',
        'question': 'Does standard TrueSkill scaling materially alter thesis results, and is an LR-centered engineering narrative defensible?',
        'hypotheses': ['Only player_ts and its derived binomial feature change; rating state and cohort stay fixed.',
                       'Compare LR/MLP at three declared seeds; do not select a best seed.',
                       'Control the MLP accuracy-based early stop by disabling it, not by tuning to TEST.'],
        'sources': sources, 'python': platform.python_version(), 'jobs': jobs,
        'features': feature_variants()['w20'], 'workers': args.workers,
        'training_budget': {'variants': 14, 'monthly_fits': 462, 'seeds': list(SEEDS),
                            'hyperparameter_searches': 0, 'mlp_max_epochs': 300,
                            'lr_max_iterations': 5000},
        'train': 'All valid strictly earlier rows and released labels, monthly expanding origins since2024.',
        'calibration': 'Existing positive zero-intercept logit slope; only prior2024+ OOS predictions and released labels, min200.',
        'evaluation': 'Exactly11550 historical targets2024-2026; common OPEN/CLOSE3229. Historical development only.',
        'primary_metrics': ['logloss', 'brier'], 'uncertainty': 'Paired monthly5000 resamples; match-weighted and equal-month; tournament-year sensitivity.',
        'decision': 'Report every fixed variant. A correction is judged mathematically first; statistical gain is not required to call a formula standard.',
        'success': 'Exact baseline replay, unchanged membership, finite symmetric predictions, all variants scored on identical IDs, no source overwrite.',
        'lr_narrative': 'Engineering choice based on observed accuracy, stability, parameter count and measured compute, not a retroactive2023 winner.',
        'development_reconstruction': 'Two additional fixed fits on pre2023 TRAIN, raw-symmetric predictions on2023 released before2024; verify original selection scores, do not select anew.',
        'limitations': ['Historical data already inspected; no new protected holdout.',
                       'First-map known-roster scenario; no quote-time availability certification.',
                       'Seed repetitions share targets and are not independent test samples.',
                       'No hyperparameter or window reselection; no production deployment or thesis edit.'],
    }
    save_json(output / 'protocol.json', protocol)
    development_evidence(original_features, selection, output)
    from src.analysis.thesis_corrections import replay_trueskill_correction
    corrected_path = replay_trueskill_correction(
        raw, bank / 'metadata.parquet', bank / 'release_days.json', original_features, feature_output,
    )
    print(json.dumps({'corrected_features': str(corrected_path)}), flush=True)
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_variant, job, str(prediction_output)) for job in jobs]
        for future in as_completed(futures):
            receipt = future.result()
            completed.append(receipt)
            print(json.dumps({'completed_variant': receipt}), flush=True)
    streams, checks = verify_streams(jobs, prediction_output, original_features, archive)
    corrected = add_series_features(pd.read_parquet(corrected_path))
    ts = rating_stream(corrected, 'player_ts')
    ts.to_parquet(prediction_output / 'ts_standard.parquet', index=False)
    streams['ts_standard'] = ts
    streams['ts_standard_raw'] = ts.assign(p_calibrated=ts.p_raw)
    panel = join_prediction_streams(pd.read_parquet(original_panel), streams)
    if len(panel) != 11550 or int(panel[['market_open', 'market_close']].notna().all(axis=1).sum()) != 3229:
        raise ValueError('Cohort drift')
    panel.to_parquet(output / 'paired_predictions.parquet', index=False)
    pd.concat([pd.read_csv(prediction_output / job['name'] / 'fit_diagnostics.csv') for job in jobs]).to_csv(output / 'fit_diagnostics.csv', index=False)
    pd.concat([pd.read_csv(prediction_output / job['name'] / 'coefficients.csv') for job in jobs if job['family'] == 'logistic']).to_csv(output / 'coefficients.csv', index=False)
    if any(sha(path) != digest for path, digest in sources.items()):
        raise RuntimeError('Frozen source changed during execution')
    save_json(output / 'replay_verification.json', {'streams': checks, 'sources_unchanged': True,
                                                  'monthly_fits': sum(x['folds'] for x in checks)})
    save_json(output / 'COMPLETE.json', {
        'status': 'HISTORICAL_DEVELOPMENT_ONLY', 'cohorts': {'sports': 11550, 'market_common': 3229},
        'jobs': completed, 'panel_sha256': sha(output / 'paired_predictions.parquet'),
        'feature_output': str(feature_output.relative_to(ROOT)),
        'prediction_output': str(prediction_output.relative_to(ROOT)),
    })
    print(json.dumps({'complete': str(output)}), flush=True)


if __name__ == '__main__':
    main()
