"""Frozen Stage2 information gate; later draft/series stages require its PASS."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analysis.prop_distribution_evaluation import paired_comparison, profile_scores, score_prediction
from src.features.draft_prop_data import prepare_draft_data
from src.models.draft_prop_model import fit_draft_prop_model
from src.models.prop_distribution import DistributionMixture, DistributionPrediction, distribution_nll
from src.models.prop_distribution_runner import partition_rows as _standalone_partition_rows

ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = ('src/features/draft_prop_data.py', 'src/models/draft_prop_model.py',
              'src/models/draft_prop_runner.py', 'scripts/train_draft_prop_distribution.py')
VARIANTS = ('matched_no_draft', 'actual_composition_oracle')


def partition_rows(frame):
    """Use whole-series effective releases for both vocabulary and formal fits."""
    if 'draft_effective_release_day' not in frame:
        raise ValueError('Prepared whole-series release metadata is required.')
    clocks = pd.DataFrame({'date': frame['date'], 'release_day': frame['draft_effective_release_day']},
                          index=frame.index)
    return _standalone_partition_rows(clocks)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def _paths(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Run ID must be a safe single path component.')
    return {stage: ROOT / f'data/{stage}/draft_prop/{run_id}' for stage in
            ('04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting')}


def _read_protocol(run_id):
    paths = _paths(run_id)
    protocol = json.loads((paths['05_model_input'] / 'protocol.json').read_text())
    if protocol['run_id'] != run_id:
        raise ValueError('Protocol run identity differs.')
    return paths, protocol


def prepare_run(run_id):
    paths, protocol = _read_protocol(run_id)
    dataset_path = paths['04_feature'] / 'dataset.parquet'
    schema_path = paths['04_feature'] / 'schema.json'
    if dataset_path.exists() or schema_path.exists() or (paths['05_model_input'] / 'binding.json').exists():
        raise FileExistsError('Prepared/frozen inputs are immutable; do not overwrite.')
    if sha256(ROOT / protocol['source']) != protocol['source_sha256']:
        raise ValueError('Draft source differs from the audited immutable source.')
    base = pd.read_parquet(ROOT / protocol['base_dataset'])
    maps = pd.read_parquet(ROOT / protocol['map_drafts'])
    frame, schema, audit = prepare_draft_data(maps, base)
    rules = json.loads((ROOT / protocol['rules']).read_text())
    bindings = {r['competition_edition_phase']: r['draft_mode'] for r in rules['bindings']}
    frame['draft_rule_status'] = frame['tournament'].map(bindings).fillna('UNKNOWN')
    if not frame['match_id'].equals(base['match_id']) or frame['match_id'].duplicated().any():
        raise ValueError('Prepared target population/order differs from original standalone cohort.')
    counts = {}
    for key, mask in partition_rows(frame).items():
        valid = mask & frame['draft_valid'].astype(bool)
        counts[key] = {'original': int(mask.sum()), 'valid': int(valid.sum()), 'absent': int((mask & ~frame['draft_valid'].astype(bool)).sum())}
        expected = protocol['partitions'][key]
        wanted = expected.get('matched_valid_composition_rows', expected.get('rows'))
        if counts[key]['valid'] != wanted:
            raise ValueError(f'{key} valid support differs from frozen audit: {counts[key]}')
    if (~frame.loc[partition_rows(frame)['ASSESS'], 'draft_valid'].astype(bool)).any():
        raise ValueError('Assessment absences differ from pre-fit audited population.')
    paths['04_feature'].mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_path, index=False)
    _json(schema_path, schema)
    audit.update({'run_id': run_id, 'partition_counts': counts, 'original_target_rows': len(base),
                  'prepared_target_rows': len(frame), 'formal_fits': 0,
                  'rule_support_assessment': frame.loc[partition_rows(frame)['ASSESS'], 'draft_rule_status'].value_counts().to_dict()})
    _json(paths['08_reporting'] / 'feature_audit.json', audit)
    return audit


def freeze_run(run_id):
    paths, protocol = _read_protocol(run_id)
    binding_path = paths['05_model_input'] / 'binding.json'
    if binding_path.exists():
        raise FileExistsError('Existing freeze is immutable.')
    dataset_path = paths['04_feature'] / 'dataset.parquet'
    schema_path = paths['04_feature'] / 'schema.json'
    frame = pd.read_parquet(dataset_path)
    schema = json.loads(schema_path.read_text())
    original = json.loads((ROOT / 'data/05_model_input/prop_distribution/prop_distribution_20261007_01/binding.json').read_text())
    protected = json.loads((paths['08_reporting'] / 'initial_integrity.json').read_text())['protected_hashes']
    pins = dict(original['pins'])
    for relative, expected in {**pins, **original['protected_hashes'], **protected}.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Original/protected input changed before freeze: {relative}')
    owned = [ROOT / path for path in CODE_PATHS]
    owned += [paths['05_model_input'] / 'protocol.json', ROOT / protocol['rules'],
              dataset_path, schema_path, ROOT / protocol['map_drafts'],
              paths['08_reporting'] / 'source_audit.json', paths['08_reporting'] / 'nested_source_audit.json',
              paths['08_reporting'] / 'feature_audit.json', paths['08_reporting'] / 'initial_integrity.json']
    pins.update({str(path.relative_to(ROOT)): sha256(path) for path in owned})
    counts = {key: int((mask & frame['draft_valid'].astype(bool)).sum()) for key, mask in partition_rows(frame).items()}
    binding = {'run_id': run_id, 'frozen_at': datetime.now(timezone.utc).isoformat(),
               'status': 'FROZEN_BEFORE_FORMAL_FITS', 'pins': pins, 'protected_hashes': protected,
               'dataset_path': str(dataset_path.relative_to(ROOT)), 'schema_path': str(schema_path.relative_to(ROOT)),
               'partition_counts': counts, 'feature_counts': {'shared': len(schema['shared_features']), 'oracle': len(schema['oracle_features'])},
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    _json(binding_path, binding)
    return binding


def _load_frozen(run_id):
    paths, protocol = _read_protocol(run_id)
    binding = json.loads((paths['05_model_input'] / 'binding.json').read_text())
    if binding['run_id'] != run_id:
        raise ValueError('Binding run identity differs.')
    for relative, expected in {**binding['pins'], **binding['protected_hashes']}.items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Frozen/protected input or code changed: {relative}')
    frame = pd.read_parquet(ROOT / binding['dataset_path'])
    schema = json.loads((ROOT / binding['schema_path']).read_text())
    return paths, protocol, binding, frame, schema


def _reference(protocol, schema):
    selected = joblib.load(ROOT / protocol['existing_model'])
    if list(selected.feature_columns) != list(schema['base_features']):
        raise ValueError('Frozen standalone feature ordering differs from reused context.')
    return selected, selected.models[selected.selected['length']]


def _fit(variant, train, cal, schema, length_reference, config, seed):
    oracle = variant == 'actual_composition_oracle'
    if variant not in VARIANTS:
        raise ValueError('Unknown frozen variant.')
    return fit_draft_prop_model(train, cal,
                               feature_columns=schema['oracle_features'] if oracle else schema['shared_features'],
                               categorical_features=schema['categorical_features'] if oracle else [],
                               side_swap=schema['side_swap'] if oracle else {},
                               length_reference=length_reference, reference_features=schema['base_features'],
                               config=config, seed=seed)


def run_smoke(run_id):
    paths, protocol, binding, frame, schema = _load_frozen(run_id)
    output = paths['08_reporting'] / 'software_smoke.json'
    if output.exists():
        raise FileExistsError('Software smoke already recorded.')
    train_all = frame.loc[partition_rows(frame)['TRAIN'] & frame['draft_valid'].astype(bool)]
    sample = train_all.sample(min(2500, len(train_all)), random_state=protocol['stage2']['seed']).sort_values(['date', 'match_id'])
    train, cal = sample.iloc[:2000].copy(), sample.iloc[2000:].copy()
    _, length_reference = _reference(protocol, schema)
    config = {**protocol['stage2']['config'], 'n_estimators': 10}
    receipts = []
    for variant in VARIANTS:
        print(f'SOFTWARE_SMOKE {variant}', flush=True)
        model = _fit(variant, train, cal, schema, length_reference, config, protocol['stage2']['seed'])
        prediction = model.predict(cal)
        scores = distribution_nll(prediction, cal['kills'].to_numpy(float), cal['duration_minutes'].to_numpy(float), cal['maps_played'].to_numpy(float))
        for task, values in scores.items():
            if not np.isfinite(values).all():
                raise ValueError(f'Software smoke nonfinite {task} score.')
        receipts.append({'variant': variant, 'fit_receipt': model.fit_receipt, 'finite_score_rows': len(cal)})
    receipt = {'run_id': run_id, 'status': 'SOFTWARE_ONLY', 'source_partition': 'TRAIN_ONLY',
               'train_rows': len(train), 'cal_rows': len(cal), 'assessment_rows_used': 0,
               'formal_mean_fits': 0, 'formal_dispersion_fits': 0, 'software_mean_fits': 4,
               'software_dispersion_fits': 4, 'variants': receipts}
    _json(output, receipt)
    return receipt


def _components(pred):
    return (pred.components, pred.weights) if isinstance(pred, DistributionMixture) else ((pred,), np.array([1.]))


def _selected_prediction(selected, frame):
    """Compose independently selected genuine marginals without a joint-law claim."""
    per_task = selected.predict(frame)
    components = [_components(per_task[task]) for task in ('kills', 'duration', 'length')]
    result, weights = [], []
    for indices in itertools.product(*(range(len(item[0])) for item in components)):
        kill, duration, length = [item[0][index] for item, index in zip(components, indices)]
        result.append(DistributionPrediction(kill.kills_mu, kill.kills_alpha,
                                             duration.duration_log_mu, duration.duration_log_sigma,
                                             length.length_probs, length.best_of))
        weights.append(float(np.prod([item[1][index] for item, index in zip(components, indices)])))
    return DistributionMixture(result, np.asarray(weights))


def _prediction_frame(candidate, prediction, frame):
    components, weights = _components(prediction)
    base = frame[['match_id', 'date', 'best_of', 'kills', 'duration_minutes', 'maps_played']].copy().reset_index(drop=True)
    records = []
    for index, (component, weight) in enumerate(zip(components, weights)):
        row = base.copy()
        row['candidate'], row['component'], row['weight'] = candidate, index, float(weight)
        row['information_scope'] = 'ACTUAL_COMPOSITION_ORACLE' if candidate == 'actual_composition_oracle' else 'NO_DRAFT_SCENARIO'
        for name in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma'):
            row[name] = getattr(component, name)
        for j in range(3):
            row[f'length_p_{j}'] = component.length_probs[:, j]
        records.append(row)
    return pd.concat(records, ignore_index=True)


def stage2_gate(metrics, comparisons, coverage_tolerance=.02):
    """Predeclared joint task gate; zero-crossing bounds are not improvements."""
    checks = []
    for task in ('kills', 'duration'):
        oracle = metrics['actual_composition_oracle'][task]
        for reference in ('matched_no_draft', 'existing_selected'):
            contrast = comparisons[f'{task}_oracle_minus_{reference}']
            checks.append({'task': task, 'reference': reference, 'guard': 'upper_NLL_CI_below_zero',
                           'value': float(contrast['ci'][1]), 'pass': bool(np.isfinite(contrast['ci'][1]) and contrast['ci'][1] < 0)})
            crps_delta = oracle['crps'] - metrics[reference][task]['crps']
            checks.append({'task': task, 'reference': reference, 'guard': 'CRPS_nonincrease',
                           'value': float(crps_delta), 'pass': bool(np.isfinite(crps_delta) and crps_delta <= 0)})
            coverage_delta = abs(oracle['central_80_coverage'] - .8) - abs(metrics[reference][task]['central_80_coverage'] - .8)
            checks.append({'task': task, 'reference': reference, 'guard': '80pct_coverage_deviation_noninferiority',
                           'value': float(coverage_delta), 'limit': coverage_tolerance,
                           'pass': bool(np.isfinite(coverage_delta) and coverage_delta <= coverage_tolerance)})
    passed = all(check['pass'] for check in checks)
    return {'status': 'PASS' if passed else 'FAIL', 'checks': checks,
            'next_stage': 'AUTHORIZED_BY_FROZEN_GATE' if passed else 'NOT_RUN_BY_FROZEN_GATE'}


def _draft_profiles(candidate, rows, frame):
    records = profile_scores(candidate, rows, frame)
    work = rows.copy()
    for column in ('draft_rule_status', 'draft_patch', 'map_index'):
        work[column] = frame[column].to_numpy()
    count = frame['draft_player_champ_min_count'].to_numpy(float)
    work['player_champion_history_support'] = np.where(count == 0, 'at_least_one_zero_history', np.where(count < 10, 'all_positive_min_below10', 'all_min10plus'))
    for dimension in ('draft_rule_status', 'draft_patch', 'map_index', 'player_champion_history_support'):
        for value, group in work.groupby(dimension, dropna=False):
            for task in ('kills', 'duration'):
                ix = group.index
                y = frame.loc[ix, 'kills' if task == 'kills' else 'duration_minutes'].to_numpy(float)
                record = {'candidate': candidate, 'dimension': dimension, 'value': str(value), 'task': task,
                          'n': len(group), 'nll': float(group[f'{task}_nll'].mean()),
                          'crps': float(group[f'{task}_proper_secondary'].mean())}
                for lower, upper, label in ((25, 75, '50'), (10, 90, '80')):
                    lo, hi = group[f'{task}_q{lower}'].to_numpy(), group[f'{task}_q{upper}'].to_numpy()
                    record[f'central_{label}_coverage'] = float(np.mean((y >= lo) & (y <= hi)))
                    record[f'central_{label}_mean_width'] = float(np.mean(hi - lo))
                records.append(record)
    return records


def run_stage2(run_id):
    paths, protocol, binding, frame, schema = _load_frozen(run_id)
    if not (paths['08_reporting'] / 'software_smoke.json').exists():
        raise ValueError('Execute the declared TRAIN-only software smoke before formal fits.')
    paths['06_models'].mkdir(parents=True, exist_ok=False)
    paths['07_model_output'].mkdir(parents=True, exist_ok=False)
    partitions = {key: frame.loc[mask & frame['draft_valid'].astype(bool)].reset_index(drop=True)
                  for key, mask in partition_rows(frame).items()}
    train, cal, assess = (partitions[key] for key in ('TRAIN', 'CAL', 'ASSESS'))
    for key, part in partitions.items():
        if len(part) != binding['partition_counts'][key]:
            raise ValueError('Frozen partition membership/support changed.')
    selected, length_reference = _reference(protocol, schema)
    models, receipts = {}, []
    for variant in VARIANTS:
        started = time.monotonic()
        print(f'FORMAL_FIT {variant} TRAIN={len(train)} CAL={len(cal)}', flush=True)
        model = _fit(variant, train, cal, schema, length_reference, protocol['stage2']['config'], protocol['stage2']['seed'])
        model_path = paths['06_models'] / f'{variant}.joblib'
        joblib.dump(model, model_path, compress=3)
        models[variant] = model
        receipts.append({'variant': variant, 'elapsed_seconds': time.monotonic() - started,
                         'model_path': str(model_path.relative_to(ROOT)), 'fit_receipt': model.fit_receipt})
        _json(paths['08_reporting'] / 'fit_receipts.json', receipts)
    predictions = {'existing_selected': _selected_prediction(selected, assess)}
    predictions.update({key: model.predict(assess) for key, model in models.items()})
    metrics, row_scores, line_scores, profile_records, exports = {}, {}, [], [], []
    fixed_lines = protocol['stage2']['fixed_lines']
    for candidate, prediction in predictions.items():
        print(f'SCORE {candidate} ASSESS={len(assess)}', flush=True)
        summary, rows, lines = score_prediction(prediction, assess, fixed_lines)
        summary['length']['status'] = 'UNCHANGED_FROZEN_REFERENCE_NOT_A_NEW_HEAD'
        metrics[candidate], row_scores[candidate] = summary, rows
        line_scores.extend({'candidate': candidate, **line} for line in lines)
        profile_records.extend(_draft_profiles(candidate, rows, assess))
        exports.append(_prediction_frame(candidate, prediction, assess))
    comparisons = {}
    for task in ('kills', 'duration'):
        for reference in ('matched_no_draft', 'existing_selected'):
            comparisons[f'{task}_oracle_minus_{reference}'] = paired_comparison(
                row_scores['actual_composition_oracle'], row_scores[reference], task,
                confidence=protocol['stage2']['bootstrap']['per_contrast_confidence'],
                draws=protocol['stage2']['bootstrap']['draws'], seed=protocol['stage2']['bootstrap']['seed'])
    gate = stage2_gate(metrics, comparisons)
    _json(paths['08_reporting'] / 'metrics.json', metrics)
    _json(paths['08_reporting'] / 'comparisons.json', comparisons)
    _json(paths['08_reporting'] / 'stage2_gate.json', gate)
    pd.concat(exports, ignore_index=True).to_parquet(paths['07_model_output'] / 'distribution_components.parquet', index=False)
    pd.concat([rows.assign(candidate=key) for key, rows in row_scores.items()], ignore_index=True).to_parquet(paths['07_model_output'] / 'row_scores.parquet', index=False)
    pd.DataFrame(profile_records).to_parquet(paths['08_reporting'] / 'profiles.parquet', index=False)
    _json(paths['08_reporting'] / 'fixed_lines.json', line_scores)
    output_paths = [paths['06_models'] / f'{variant}.joblib' for variant in VARIANTS]
    output_paths += [paths['07_model_output'] / 'distribution_components.parquet', paths['07_model_output'] / 'row_scores.parquet']
    output_paths += [paths['08_reporting'] / name for name in ('fit_receipts.json', 'metrics.json', 'comparisons.json', 'stage2_gate.json', 'profiles.parquet', 'fixed_lines.json')]
    decision = {'run_id': run_id, 'status': 'STAGE2_COMPLETED' if gate['status'] == 'PASS' else 'COMPLETED',
                'decision': 'INSUFFICIENT_EVIDENCE' if gate['status'] == 'PASS' else 'NOT_ADOPTED',
                'reason': 'Advance to separately frozen Stage3 execution; no adoption from oracle information.' if gate['status'] == 'PASS' else 'Actual-composition diagnostic fails frozen joint improvement gate; no later fits/simulation.',
                'classification': 'ACTUAL_COMPOSITION_ORACLE_DEVELOPMENT_ONLY',
                'stage2_gate': gate['status'], 'stage3': 'PENDING_AUTHORIZED_GATE' if gate['status'] == 'PASS' else 'NOT_RUN_BY_FROZEN_GATE',
                'stage4': 'NOT_RUN_PENDING_STAGE3_GATE' if gate['status'] == 'PASS' else 'NOT_RUN_BY_FROZEN_GATE',
                'partition_counts': binding['partition_counts'], 'first_map_only': True,
                'formal_mean_regressors': 4, 'formal_dispersion_fits': 4,
                'new_length_fits': 0, 'C0_fits': 0, 'assessment_fits': 0, 'complex_draft_policy_fits': 0,
                'draft_simulations': 0, 'series_simulations': 0, 'production_writes': 0, 'deployment': False,
                'rule_support': assess['draft_rule_status'].value_counts().to_dict(),
                'standard_rule_support': int((assess['draft_rule_status'] == 'STANDARD').sum()),
                'market_evidence': {'status': 'NO_NEW_MARKET_SUPERIORITY_CLAIM',
                                    'prior_decision': 'data/08_reporting/prop_market_comparison/prop_market_comparison_20261007_01/decision.json',
                                    'kill_price_fixture_ids': ['83480', '83526'],
                                    'kill_price_fixtures_in_frozen_assessment': int(assess['match_id'].isin(['83480', '83526']).sum()),
                                    'duration_prices': 0, 'unchanged_length': 'previous negative Pinnacle comparison retained'},
                'output_hashes': {str(path.relative_to(ROOT)): sha256(path) for path in output_paths},
                'limitations': ['Actual compositions and first-map rosters are information-rich historical scenarios, not issued pre-match forecasts.',
                                'No ban or pick-order sequence labels; no verified STANDARD assessment phase.',
                                'First-map information gate does not evaluate later-map Fearless depletion or certify every possible composition representation.',
                                'All2026is development-exposed; intervals are not independent prospective evidence.']}
    _json(paths['08_reporting'] / 'decision.json', decision)
    return decision


def predict_saved_row(run_id, match_id, variant='actual_composition_oracle'):
    paths, protocol, binding, frame, schema = _load_frozen(run_id)
    if variant not in VARIANTS:
        raise ValueError('Unknown saved variant.')
    decision = json.loads((paths['08_reporting'] / 'decision.json').read_text())
    for relative, expected in decision['output_hashes'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Frozen output changed: {relative}')
    selected = frame.loc[frame['match_id'].astype(str) == str(match_id)]
    if len(selected) != 1:
        raise ValueError('Saved historical target must exist exactly once.')
    if variant == 'actual_composition_oracle' and not bool(selected.iloc[0]['draft_valid']):
        return {'match_id': str(match_id), 'variant': variant, 'available': False,
                'absence': str(selected.iloc[0]['draft_absence'])}
    model = joblib.load(paths['06_models'] / f'{variant}.joblib')
    pred = model.predict(selected)
    return {'match_id': str(match_id), 'variant': variant, 'available': True,
            'information_scope': 'ACTUAL_COMPOSITION_ORACLE_NOT_PREMATCH' if variant == 'actual_composition_oracle' else 'NO_DRAFT_HISTORICAL_SCENARIO',
            'map_index': 1, 'date': str(selected.iloc[0]['date']), 'best_of': int(pred.best_of[0]),
            'kills_over': {str(line): float(pred.kills_over(line)[0]) for line in protocol['stage2']['fixed_lines']['kills']},
            'duration_over_minutes': {str(line): float(pred.duration_over(line)[0]) for line in protocol['stage2']['fixed_lines']['duration_minutes']},
            'unchanged_length_probs': pred.components[0].length_probs[0].tolist(),
            'mixture_components': len(pred.components)}
