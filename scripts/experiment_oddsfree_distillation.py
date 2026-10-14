#!/usr/bin/env python3
"""EXP-089: all-outcome training plus chronological auxiliary market supervision.

Research only. Students consume canonical79 sports features and predict SERIES
outcomes, never map probabilities. No database, scraping, or model promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting_app.core.models.engine import apply_temperature_scaling, bayesian_logit_shrinkage
from scripts.benchmark_model_redesign import align_legacy_context, temporal_blocks
from scripts.benchmark_siamese_architectures import attach_market
from scripts.build_siamese_research_dataset import sha256
from scripts.evaluate_oddsfree_distillation import evaluate
from scripts.experiment_market_distillation import train_student
from scripts.oddsfree_distillation import fit_auxiliary_student
from scripts.train_and_tune_siamese_series import build_training_features, fit_platt_scaling
from src.analysis.probability_metrics import binary_log_loss_vector
from src.models.symmetric_series import _REQUIRED_BASE_FIELDS

VERSION = 'exp089-all-outcomes-auxiliary-series-v1'
WEIGHTS = (0.1, 0.3, 0.5)
IDENTITIES = {'golgg_match_id': 'string', 'team1_id': 'string', 'team2_id': 'string'}
LIMITATIONS = [
    'Retrospective research on previously inspected matches; no untouched holdout or promotion.',
    'Legacy feature/rating provenance and roster availability timestamps are uncertified.',
    'Closing averages are historical teacher labels only, never student inference inputs; quote timestamps are unverified.',
    'The chronological hybrid teacher uses a newly fitted ridge79 sports model, NOT the frozen EXP081 artifact.',
    'Common auxiliary coverage requires an earlier-fitted teacher; first-year outcome rows are retained without teacher labels.',
    'Student outputs are SERIES probabilities. They cannot be re-expanded as independent map probabilities.',
    'Match-date snapshots do not establish pre-tournament information; full tournament replay requires a frozen start-time state and bracket contract.',
    'No qualifying, staking, profit, drawdown, or automatic application-model changes.',
]


def save_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def read_frame(path):
    return pd.read_csv(path, dtype=IDENTITIES, low_memory=False).sort_values(['date', 'golgg_match_id']).reset_index(drop=True)


def stages(dates, masks):
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f'empty chronological stage: {name}')
        result[name] = {'n': int(mask.sum()), 'first': str(dates[mask].min()), 'last': str(dates[mask].max())}
    return result


def teacher_masks(dates, year):
    """Fit before prior H2, calibrate prior H2, predict the next complete year."""
    dates = pd.to_datetime(dates)
    return {
        'train': ((dates >= '2020-01-01') & (dates < f'{year-1}-07-01')).to_numpy(),
        'calibration': ((dates >= f'{year-1}-07-01') & (dates < f'{year}-01-01')).to_numpy(),
        'prediction': ((dates >= f'{year}-01-01') & (dates < f'{year+1}-01-01')).to_numpy(),
    }


def build_teachers(full, x, names, output):
    y = full.y_true.to_numpy(float)
    sports = np.full(len(full), np.nan)
    folds = []
    for year in range(2021, 2025):
        masks = teacher_masks(full.date, year)
        record = {'prediction_year': year, 'stages': stages(full.date, masks)}
        train, cal, predict = (masks[n] for n in ('train', 'calibration', 'prediction'))
        assert not np.any(train & cal) and not np.any((train | cal) & predict)
        assert full.date[train | cal].max() < full.date[predict].min()
        scaler = StandardScaler(with_mean=False).fit(x[train])
        coef, optimization = train_student(scaler.transform(x[train]), y[train])
        slope = fit_platt_scaling(scaler.transform(x[cal]) @ coef, y[cal])
        sports[predict] = expit(slope * (scaler.transform(x[predict]) @ coef))
        artifact = {'version': VERSION, 'role': 'chronological_ridge_teacher', 'feature_names': names,
                    'scale': scaler.scale_, 'coef': coef, 'slope': slope, 'stages': record['stages']}
        path = output / f'teacher-{year}.joblib'
        joblib.dump(artifact, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(expit(restored['slope'] * ((x[predict] / restored['scale']) @ restored['coef'])), sports[predict], atol=1e-12, rtol=0)
        record.update(optimization=optimization, slope=slope, artifact=path.name, artifact_sha256=sha256(path))
        folds.append(record)
    market = full.market.to_numpy(float)
    prices = full[['odds_a', 'odds_b']].to_numpy(float)
    covered = np.isfinite(sports) & np.isfinite(market) & (np.sum(1 / prices, axis=1) >= 1)
    targets = {'market': np.where(covered, market, np.nan), 'hybrid': np.full(len(full), np.nan)}
    for position in np.flatnonzero(covered):
        targets['hybrid'][position] = bayesian_logit_shrinkage(apply_temperature_scaling(sports[position], .8), market[position], .5)
    target_rows = full[['golgg_match_id', 'date']].copy()
    target_rows['chronological_sports_teacher'] = sports
    target_rows['market_target'] = targets['market']
    target_rows['hybrid_target'] = targets['hybrid']
    target_rows['auxiliary_eligible'] = covered
    target_rows.to_csv(output / 'training_teacher_targets.csv', index=False)
    return targets, {'folds': folds, 'common_auxiliary_n': int(covered.sum()), 'alpha': .5, 'temperature': .8,
                     'blending_mode': 'logit_shrinkage', 'sports_teacher': 'new chronological canonical79 ridge',
                     'first_year_policy': '2020 outcome rows retained, no teacher supervision', 'underround_policy': 'exclude from auxiliary labels, retain for outcome training and descriptive evaluation'}


def predict_artifact(artifact, raw_sports_rows):
    """Odds-free inference from an explicitly experimental serialized SERIES model."""
    x, names = build_training_features(raw_sports_rows)
    if names != artifact['feature_names']:
        raise ValueError('serialized feature order mismatch')
    return expit(artifact['slope'] * ((x / artifact['scale']) @ artifact['coef']))


def run(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    input_paths = {
        'legacy': ROOT / 'data/artifacts/siamese-integrity-v1/snapshots.csv',
        'replay': ROOT / 'data/artifacts/model-redesign-v1/replay/snapshots.csv',
        'reference': ROOT / 'data/artifacts/model-redesign-v1/benchmark-semester/legacy79/predictions.csv',
        'odds': ROOT / 'data/odds.csv',
    }
    code_paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT) and path.suffix == '.py' and path.relative_to(ROOT).parts[0] in ('scripts', 'src', 'betting_app'):
                code_paths.add(path)
    code_hashes = {}
    for path in sorted(code_paths):
        relative = path.relative_to(ROOT)
        code_hashes[str(relative)] = sha256(path)
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        if sha256(destination) != code_hashes[str(relative)]:
            raise ValueError('source changed during snapshot')
    save_json(output / 'protocol.json', {
        'experiment': 'EXP-089', 'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
        'weights_fixed_before_run': list(WEIGHTS), 'selection_includes_weight_zero': True,
        'selection': 'provisional calibration on Q1; choose family weight on Q2 hard-outcome LogLoss; final calibration on disjoint H2',
        'outer_folds': [2024, 2025, 2026], 'training_start': '2020-01-01',
        'outcome_training': 'ALL rows, identical population/scaler/regularization for every arm',
        'teacher_supervision': 'same covered training rows for market and chronological-hybrid arms',
        'loss': 'mean_all BCE(y,p) + lambda*mean_covered BCE(q,p) + ||w||^2/(2*C*N_all)',
        'student': 'canonical79 no-intercept linear logit; positive outcome-only calibration',
        'limitations': LIMITATIONS, 'source_sha256': code_hashes,
        'input_sha256': {key: sha256(path) for key, path in input_paths.items()},
    })
    try:
        legacy, replay = read_frame(input_paths['legacy']), read_frame(input_paths['replay'])
        replay_audit = json.loads(input_paths['replay'].with_name('audit.json').read_text())
        if sha256(input_paths['replay']) != replay_audit['snapshots_sha256']:
            raise ValueError('replay artifact hash mismatch')
        full, alignment = align_legacy_context(legacy, replay)
        full = full.sort_values(['date', 'golgg_match_id']).reset_index(drop=True)
        full, market_audit = attach_market(full, pd.read_csv(input_paths['odds'], dtype={'golgg_match_id': 'string'}, low_memory=False))
        x, names = build_training_features(full)
        if x.shape[1] != 79 or not np.isfinite(x).all():
            raise ValueError('canonical79 contract failed')
        y = full.y_true.to_numpy(float)
        targets, teacher_audit = build_teachers(full, x, names, output)
        reference = read_frame(input_paths['reference'])
        predictions, folds = [], []
        raw_fields = sorted(_REQUIRED_BASE_FIELDS | {'best_of'})
        for year in (2024, 2025, 2026):
            masks = temporal_blocks(full.date, year, protocol='semester-calibration')
            stage_records = stages(full.date, masks)
            train, stop, select, cal, test = (masks[n] for n in ('train', 'stop', 'select', 'calibration', 'test'))
            assert np.stack(list(masks.values())).sum(axis=0).max() == 1
            for before, after in zip(('train', 'stop', 'select', 'calibration'), ('stop', 'select', 'calibration', 'test')):
                assert stage_records[before]['last'] < stage_records[after]['first']
            scaler = StandardScaler(with_mean=False).fit(x[train])
            scaled = {name: scaler.transform(x[mask]) for name, mask in masks.items()}
            rows = full.loc[test, ['golgg_match_id', 'date', 'y_true', 'best_of', 'competition_tier', 'tournament', 'roster_min_prior_series', 'odds_a', 'odds_b', 'market']].copy()
            rows['player_consensus'] = full.loc[test, [f'player_{s}' for s in ('elo', 'gl', 'ts', 'os', 'pl', 'tm')]].mean(axis=1)
            rows['team_consensus'] = full.loc[test, [f'team_{s}' for s in ('elo', 'gl', 'ts', 'os', 'pl', 'tm')]].mean(axis=1)
            fold = {'year': year, 'stages': stage_records, 'training_outcome_n': int(train.sum()),
                    'auxiliary_training_n': int(np.isfinite(targets['market'][train]).sum()), 'models': {}, 'selected': {}}
            configs = [('outcome', None, 0.)] + [(f'{family}_w{int(weight*100):02d}', family, weight) for family in ('market', 'hybrid') for weight in WEIGHTS]
            for name, family, weight in configs:
                q = np.full(int(train.sum()), np.nan) if family is None else targets[family][train]
                coef, optimization = fit_auxiliary_student(scaled['train'], y[train], q, weight)
                stop_slope = fit_platt_scaling(scaled['stop'] @ coef, y[stop])
                select_p = expit(stop_slope * (scaled['select'] @ coef))
                selection_ll = float(binary_log_loss_vector(y[select], select_p).mean())
                slope = fit_platt_scaling(scaled['calibration'] @ coef, y[cal])
                p = expit(slope * (scaled['test'] @ coef))
                if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
                    raise ValueError('student probabilities must remain finite and open')
                symmetry = float(np.max(abs(p + expit(slope * ((-scaled['test']) @ coef)) - 1)))
                if symmetry > 1e-6:
                    raise ValueError('student side symmetry failed')
                artifact = {'version': VERSION, 'status': 'experimental_not_qualified', 'target': 'series',
                            'model': name, 'year': year, 'feature_names': names, 'scale': scaler.scale_,
                            'coef': coef, 'slope': slope, 'weight': weight, 'teacher': family,
                            'stages': stage_records, 'requires_odds_at_inference': False, 'limitations': LIMITATIONS}
                path = output / f'{year}-{name}.joblib'
                joblib.dump(artifact, path)
                restored = joblib.load(path)
                # An actual serialized inference call, with odds, outcome and all evaluation metadata removed.
                inferred = predict_artifact(restored, full.loc[test, raw_fields])
                np.testing.assert_allclose(inferred, p, atol=1e-12, rtol=0)
                rows['p__' + name] = p
                fold['models'][name] = {'weight': weight, 'teacher': family, 'optimization': optimization,
                                        'provisional_stop_slope': stop_slope, 'selection_log_loss': selection_ll,
                                        'final_calibration_slope': slope, 'symmetry_max_error': symmetry,
                                        'artifact': path.name, 'artifact_sha256': sha256(path)}
            for family in ('market', 'hybrid'):
                options = ['outcome'] + [name for name, f, _ in configs if f == family]
                selected = min(options, key=lambda name: fold['models'][name]['selection_log_loss'])
                rows[f'p__{family}_selected'] = rows['p__' + selected]
                fold['selected'][family] = {'model': selected, 'weight': fold['models'][selected]['weight'],
                                            'artifact': fold['models'][selected]['artifact'],
                                            'selection_log_loss': fold['models'][selected]['selection_log_loss']}
            predictions.append(rows)
            folds.append(fold)
            save_json(output / f'{year}-fold.json', fold)
            print(f'Completed fold {year}: {fold["training_outcome_n"]} outcome rows; selected {fold["selected"]}', flush=True)
        prediction = pd.concat(predictions, ignore_index=True).sort_values(['date', 'golgg_match_id']).reset_index(drop=True)
        for column in ('golgg_match_id', 'date', 'y_true', 'best_of', 'competition_tier'):
            np.testing.assert_array_equal(prediction[column], reference[column], err_msg=column)
        for column in ('odds_a', 'odds_b', 'market'):
            np.testing.assert_allclose(prediction[column], reference[column], equal_nan=True, atol=1e-12, rtol=0)
        prediction['p__frozen_ridge'] = reference.fixed_ridge.to_numpy()
        prediction['p__stored_exp081_diagnostic'] = reference.stored_exp081.to_numpy()
        control_delta = float(np.mean(binary_log_loss_vector(prediction.y_true, prediction.p__outcome) - binary_log_loss_vector(prediction.y_true, prediction.p__frozen_ridge)))
        prediction.to_csv(output / 'predictions.csv', index=False)
        evaluation = evaluate(prediction, output)
        for path, expected in code_hashes.items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f'code changed during experiment: {path}')
        summary = {'experiment': 'EXP-089', 'version': VERSION, 'status': 'completed_research_not_promoted',
                   'created_at': datetime.now(timezone.utc).isoformat(), 'rows': len(prediction),
                   'market_rows': int(prediction.market.notna().sum()), 'alignment': alignment, 'market_join': market_audit,
                   'teacher': teacher_audit, 'folds': folds, 'control_delta_LL_vs_frozen_ridge': control_delta,
                   'evaluation': evaluation, 'limitations': LIMITATIONS + replay_audit['promotion_blockers'],
                   'environment': {'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib')}},
                   'runtime_seconds': time.monotonic()-started, 'command': sys.argv,
                   'source_sha256': code_hashes, 'predictions_sha256': sha256(output / 'predictions.csv')}
        save_json(output / 'summary.json', summary)
        print(json.dumps({'rows': len(prediction), 'control_delta_LL': control_delta, 'runtime_seconds': summary['runtime_seconds'], 'output': str(output)}, indent=2), flush=True)
    except Exception as exc:
        save_json(output / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc), 'runtime_seconds': time.monotonic()-started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
