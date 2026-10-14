"""Bound, historical research execution; never a serving-model cutover."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = 'champion_history_20261007_01'
BASE = 'data/04_feature/prop_distribution/prop_distribution_20261007_01/dataset.parquet'
SOURCE = 'data/artifacts/golgg-database-recovery-20260909/matches.json'
RULES = 'data/05_model_input/draft_prop/draft_prop_20261007_01/rules.json'
C0_OUTPUT = 'data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet'
PROP_MODEL = 'data/06_models/prop_distribution/prop_distribution_20261007_01/selected_model.joblib'
CODE_PATHS = ('src/features/champion_history_data.py', 'src/models/champion_history_model.py',
              'src/analysis/champion_history_evaluation.py', 'src/models/champion_history_runner.py',
              'scripts/train_champion_history.py', 'src/models/prematch_features.py',
              'src/features/prop_distribution_data.py', 'src/models/prop_distribution.py',
              'src/analysis/prop_distribution_evaluation.py', 'src/models/tournament_scorelines.py',
              'src/models/prop_distribution_runner.py', 'scripts/export_prospective_features.py',
              'src/features/c0_meta_features.py', 'src/utils/golgg_schema.py')
VARIANTS = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
SEEDS = (20261007, 20261008, 20261009)
CONFIDENCE = 1 - .05 / 12


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _paths(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Unsafe run name')
    return {stage: ROOT / f'data/{stage}/champion_history/{run_id}'
            for stage in ('04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting')}


def partitions(series):
    day = series['date'].astype(str)
    release = series['effective_release_day'].astype(str)
    return {'TRAIN': np.flatnonzero((day < '2025-07-01') & (release < '2025-07-01')),
            'CAL': np.flatnonzero((day >= '2025-07-01') & (day < '2026-01-01') & (release < '2026-01-01')),
            'ASSESS': np.flatnonzero((day >= '2026-01-01') & (day < '2027-01-01'))}


def _integrity(paths):
    receipt = json.loads((paths['08_reporting'] / 'initial_integrity.json').read_text())
    mismatches = [p for p, h in receipt['protected_hashes'].items() if sha256(ROOT / p) != h]
    if mismatches:
        raise ValueError(f'Protected input changed: {mismatches}')
    return {'checked_at': _now(), 'status': 'PASS', 'protected_count': len(receipt['protected_hashes']),
            'protected_hashes': receipt['protected_hashes']}


def prepare_run(run_id=RUN_ID, research_root=None):
    from src.features.champion_history_data import prepare_history_data
    paths = _paths(run_id)
    _integrity(paths)
    if (paths['04_feature'] / 'arrays.npz').exists():
        raise FileExistsError('Prepared data already exists; do not overwrite an experiment')
    base = pd.read_parquet(ROOT / BASE)
    rules = json.loads((ROOT / RULES).read_text())
    series, maps, arrays, schema, audit = prepare_history_data(
        ROOT / SOURCE, base, history_length=16, rules=rules, research_root=research_root)
    if len(series) != len(base) or not np.array_equal(series.match_id.astype(str), base.match_id.astype(str)):
        raise ValueError('Canonical rows were lost or reordered')
    if not np.array_equal(series['row_index'], np.arange(len(series))):
        raise ValueError('Invalid canonical row indexing')
    for name, array in arrays.items():
        if len(array) != len(series):
            raise ValueError(f'Array {name} is not canonical-row aligned')
    paths['04_feature'].mkdir(parents=True, exist_ok=True)
    series.to_parquet(paths['04_feature'] / 'series.parquet', index=False)
    maps.to_parquet(paths['04_feature'] / 'maps.parquet', index=False)
    np.savez_compressed(paths['04_feature'] / 'arrays.npz', **arrays)
    _json(paths['04_feature'] / 'schema.json', schema)
    _json(paths['04_feature'] / 'audit.json', audit)
    counts = {}
    for name, indices in partitions(series).items():
        sm = maps.loc[maps.row_index.isin(indices)]
        counts[name] = {'series': len(indices), 'recorded_maps': len(sm),
                        'winner_labels': int(series.iloc[indices].winner.notna().sum()),
                        'score_labels': int(series.iloc[indices].score_index.notna().sum()),
                        'kills_labels': int(sm.kills.notna().sum()),
                        'duration_labels': int(sm.duration_minutes.notna().sum()),
                        'first_map_kills': int(sm.loc[sm.map_index == 1, 'kills'].notna().sum()),
                        'first_map_duration': int(sm.loc[sm.map_index == 1, 'duration_minutes'].notna().sum())}
    pins = {str(p.relative_to(ROOT)): sha256(p) for p in paths['04_feature'].iterdir() if p.is_file()}
    receipt = {'run_id': run_id, 'prepared_at': _now(), 'counts': counts, 'pins': pins,
               'source_sha256': sha256(ROOT / SOURCE), 'base_sha256': sha256(ROOT / BASE),
               'champions_are_historical_only': True, 'target_patch_is_profile_only': True,
               'later_maps_share_series_start_history': True, 'formal_fits': 0}
    _json(paths['04_feature'] / 'prepared.json', receipt)
    return receipt


def freeze_run(run_id=RUN_ID):
    from src.models.champion_history_model import DEFAULT_CONFIG
    paths = _paths(run_id)
    _integrity(paths)
    destination = paths['05_model_input'] / 'protocol.json'
    if destination.exists():
        raise FileExistsError('Protocol already frozen')
    prepared = json.loads((paths['04_feature'] / 'prepared.json').read_text())
    if [prepared['counts'][k]['series'] for k in ('TRAIN', 'CAL', 'ASSESS')] != [36205, 1328, 3103]:
        raise ValueError('Population differs from frozen split; review before fitting')
    primary = [('B', 'A', 'winner_nll'), ('D', 'C', 'winner_nll'),
               ('B', 'C0', 'winner_nll'), ('D', 'C0', 'winner_nll'),
               ('B', 'A', 'length_nll'), ('D', 'C', 'length_nll'),
               ('B', 'old_props', 'length_nll'), ('D', 'old_props', 'length_nll'),
               ('D', 'C', 'kills_nll'), ('D_first', 'old_props', 'kills_nll'),
               ('D', 'C', 'duration_nll'), ('D_first', 'old_props', 'duration_nll')]
    protocol = {'run_id': run_id, 'frozen_at': _now(), 'status': 'FROZEN_BEFORE_FORMAL_FITS',
        'hypothesis': 'Historical champion-performance-pace interactions improve pre-match outcomes or prop distributions',
        'scenario': 'actual_first_map_roster_conditioned_daily_release; not authenticated publication',
        'history_length': 16, 'config': dict(DEFAULT_CONFIG), 'variants': VARIANTS, 'seeds': SEEDS,
        'counts': prepared['counts'], 'formal_trajectory_budget': 12, 'calibration_scalar_budget': 8,
        'c0_fits': 0, 'assessment_fits': 0, 'epochs': 12,
        'score_temperature': {'bounds': [.25, 4], 'log_scalar_quadratic_penalty': .001},
        'dispersion_scales': {'bounds': [.25, 4], 'log_scalar_quadratic_penalty': .001,
                              'kills': 'multiply alpha, not mean', 'duration': 'multiply sigma, not logmean'},
        'primary_contrasts': primary, 'primary_family_size': 12,
        'confidence': CONFIDENCE, 'bootstrap_draws': 5000, 'bootstrap_seed': 20261007,
        'bootstrap_unit': 'paired whole-series scores, monthly blocks, all assessment months retained',
        'prop_series_score': 'mean proper score over available played maps within series; map-weighted also reported',
        'cal_selection': {'winner': 'min winner NLL among B,D', 'length': 'min length NLL among B,D',
                          'kills': 'D fixed against C and existing first-map reference',
                          'duration': 'D fixed against C and existing first-map reference'},
        'gates': {'required': 'Both primary NLL upper bounds strictly below zero; same evaluated labels',
                  'winner': 'Brier point nonincrease against matched control and C0',
                  'length': 'RPS point nonincrease against matched control and old props',
                  'kills': 'CRPS point nonincrease versus C all maps and old props first map',
                  'duration': 'CRPS point nonincrease versus C all maps and old props first map',
                  'coverage': {'50_min': .45, '80_min': .75,
                               'max_absolute_nominal_error_increase_vs_both_controls': .025},
                  'bundle': 'All four task gates pass; one common variant supplies coherent winner and length'},
        'reference_limits': {'old_props': 'first-map kills/duration only; length supports entire series',
                             'coherent_score': 'Existing fixed Markov beta .35 law preserves frozen C0 series margin; heuristic diagnostic, not new map outcome model'},
        'forbidden': ['target picks', 'target bans', 'target side', 'target realized roles',
                      'target performance or outcomes as features', 'uncertified target patch', 'future maps', 'odds'],
        'fearless': 'Exact evidenced competition/edition/phase only; UNKNOWN is not STANDARD; no draft simulation',
        'protected': 'C0/artifacts/canonical/serving/prior negatives/G0/10-of-10/forward dates unchanged',
        'assessment_status': 'DEVELOPMENT_ONLY; every historical year exposed',
        'no_automatic_retry_recipe': True, 'software_smoke': 'one TRAIN-only D trajectory, one epoch, 256 series; not formal fit'}
    if int(protocol['config']['epochs']) != 12:
        raise ValueError('Training duration differs from authorized frozen duration')
    _json(destination, protocol)
    pins = dict(prepared['pins'])
    for p in (*CODE_PATHS, BASE, SOURCE, RULES, C0_OUTPUT, PROP_MODEL):
        pins[p] = sha256(ROOT / p)
    audit = json.loads((paths['04_feature'] / 'audit.json').read_text())
    if audit['strength_provenance']['kind'] == 'corrected_symmetry_bank':
        bank = Path(audit['strength_provenance']['path'])
        for name in ('metadata.parquet', 'arrays.npz', 'audit.json'):
            p = bank / name
            locator = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            pins[locator] = sha256(p)
    from src.features.champion_history_data import _paths as feature_roots
    release_relative = Path('data/03_primary/ground_up_lol_20260927_run001/release_days.json')
    for root in feature_roots(None):
        p = root / release_relative
        if p.is_file():
            locator = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
            pins[locator] = sha256(p)
            break
    pins[str((paths['04_feature'] / 'prepared.json').relative_to(ROOT))] = sha256(paths['04_feature'] / 'prepared.json')
    binding = {'run_id': run_id, 'bound_at': _now(), 'protocol_sha256': sha256(destination), 'pins': pins, 'formal_fits': 0}
    _json(paths['05_model_input'] / 'binding.json', binding)
    return {'protocol': protocol, 'binding': binding}


def load_frozen(run_id=RUN_ID):
    paths = _paths(run_id)
    _integrity(paths)
    protocol = json.loads((paths['05_model_input'] / 'protocol.json').read_text())
    binding = json.loads((paths['05_model_input'] / 'binding.json').read_text())
    if sha256(paths['05_model_input'] / 'protocol.json') != binding['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    pins = dict(binding['pins'])
    continuation_path = paths['05_model_input'] / 'scoring_continuation_binding.json'
    if continuation_path.exists():
        continuation = json.loads(continuation_path.read_text())
        if continuation['frozen_binding_sha256'] != sha256(paths['05_model_input'] / 'binding.json'):
            raise ValueError('Original frozen binding differs from scoring continuation')
        changed = continuation['runner_change']
        runner_path = 'src/models/champion_history_runner.py'
        if changed['original_sha256'] != pins[runner_path] or sha256(ROOT / changed['original_snapshot']) != pins[runner_path]:
            raise ValueError('Original frozen runner source was not preserved')
        pins[runner_path] = changed['current_sha256']
        pins.update(continuation['additional_pins'])
    for p, h in pins.items():
        if sha256(ROOT / p) != h:
            raise ValueError(f'Frozen input/code changed: {p}')
    series = pd.read_parquet(paths['04_feature'] / 'series.parquet')
    maps = pd.read_parquet(paths['04_feature'] / 'maps.parquet')
    with np.load(paths['04_feature'] / 'arrays.npz') as saved:
        arrays = {k: saved[k] for k in saved.files}
    return paths, protocol, binding, series, maps, arrays


def _subset(series, maps, indices):
    sub = series.iloc[indices].copy().reset_index(drop=True)
    return sub, maps.loc[maps.row_index.isin(indices)].copy().reset_index(drop=True)


def run_smoke(run_id=RUN_ID):
    from src.models.champion_history_model import fit_history_model, HistoryEnsemble
    from src.analysis.champion_history_evaluation import evaluate_prediction
    paths, protocol, binding, series, maps, arrays = load_frozen(run_id)
    destination = paths['08_reporting'] / 'software_smoke.json'
    if destination.exists():
        raise FileExistsError('Smoke receipt already exists')
    train = partitions(series)['TRAIN']
    indices = train[series.iloc[train].date.astype(str).to_numpy() >= '2024-01-01'][:256]
    config = {**protocol['config'], 'epochs': 1, 'batch_size': 64}
    model = fit_history_model(arrays, series, maps, indices, config=config,
                              seed=SEEDS[0], champion_enabled=True, multitask=True, software_smoke=True)
    model_path = paths['08_reporting'] / 'software_smoke_model.joblib'
    joblib.dump(model, model_path)
    saved = joblib.load(model_path)
    a, b = model.predict(arrays, indices), saved.predict(arrays, indices)
    if not np.array_equal(a.score_probs, b.score_probs):
        raise ValueError('Saved smoke inference differs')
    ensemble = HistoryEnsemble((saved, saved, saved))
    prediction = ensemble.predict(arrays, indices)
    sm, mm = _subset(series, maps, indices)
    metrics, ss, ms, profiles = evaluate_prediction(prediction, sm, mm)
    receipt = {'run_id': run_id, 'completed_at': _now(), 'status': 'PASS',
               'TRAIN_only': True, 'formal_fits': 0, 'software_fit_count': 1,
               'series': len(sm), 'recorded_maps': len(mm), 'metrics': metrics,
               'fit_receipt': saved.fit_receipt, 'saved_replay_max_error': 0.,
               'three_component_smoke': 'same saved model repeated for software mixture exercise; NOT three independent fits'}
    _json(destination, receipt)
    return receipt


def _components(prediction):
    components = getattr(prediction, 'components', None)
    return tuple(components) if components else (prediction,)


def _save_prediction(prediction, series, maps, output, name):
    terminal = series[['row_index', 'match_id', 'date', 'best_of', 'winner', 'score_index']].copy()
    terminal['candidate'] = name
    terminal['p_win'] = prediction.p_win
    for i in range(6):
        terminal[f'score_p_{i}'] = prediction.score_probs[:, i]
    for i in range(3):
        terminal[f'length_p_{i}'] = prediction.length_probs[:, i]
    for i in range(5):
        terminal[f'map_{i+1}_play_probability'] = prediction.play_probs[:, i]
    terminal.to_parquet(output / f'{name}_terminal.parquet', index=False)
    components = _components(prediction)
    component_frames = []
    for c, comp in enumerate(components):
        base = series[['row_index', 'match_id', 'date', 'best_of']].copy()
        base['candidate'], base['component'], base['weight'] = name, c, 1 / len(components)
        for i in range(6):
            base[f'score_p_{i}'] = comp.score_probs[:, i]
        try:
            for m in range(1, 6):
                dp = comp.map_prediction(m)
                row = base.loc[base.best_of >= m].copy()
                supported = base.best_of.to_numpy() >= m
                row['map_index'] = m
                row['probability_played'] = prediction.play_probs[supported, m-1]
                row['conditional_on_played'] = True
                for field in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma'):
                    row[field] = getattr(dp, field)[supported]
                component_frames.append(row)
        except NotImplementedError:
            base['statistics_head_available'] = False
            component_frames.append(base)
    pd.concat(component_frames, ignore_index=True).to_parquet(output / f'{name}_components.parquet', index=False)


def _old_prediction(base):
    from src.models.prop_distribution import DistributionPrediction, DistributionMixture
    selected = joblib.load(ROOT / PROP_MODEL).predict(base)
    entries = []
    for task in ('kills', 'duration', 'length'):
        pred = selected[task]
        comps = pred.components if hasattr(pred, 'components') else (pred,)
        weights = pred.weights if hasattr(pred, 'weights') else np.ones(1)
        entries.append(tuple(zip(weights, comps)))
    components, weights = [], []
    for (wk, k), (wd, d), (wl, l) in itertools.product(*entries):
        components.append(DistributionPrediction(k.kills_mu, k.kills_alpha,
                          d.duration_log_mu, d.duration_log_sigma, l.length_probs, l.best_of))
        weights.append(wk * wd * wl)
    return DistributionMixture(components, np.asarray(weights))


def _reference_metadata():
    from src.features.champion_history_data import _paths as research_paths
    manifest = json.loads((ROOT / 'conf/base/research_benchmark.json').read_text())
    for root in research_paths(None):
        candidate = root / manifest['input']['path']
        if candidate.is_file():
            if sha256(candidate) != manifest['input']['sha256']:
                raise ValueError('Canonical native participant metadata changed')
            return candidate, pd.read_parquet(candidate)
    raise FileNotFoundError('Canonical native participant metadata is unavailable')


def _align_c0_reference(saved, metadata, series):
    saved = saved.assign(match_id=saved.golgg_match_id.astype(str)).set_index('match_id')
    metadata = metadata.assign(match_id=metadata.golgg_match_id.astype(str)).set_index('match_id')
    if saved.index.has_duplicates or metadata.index.has_duplicates:
        raise ValueError('C0 reference contains duplicate identity')
    ids = series.match_id.astype(str)
    saved, metadata = saved.loc[ids], metadata.loc[ids]
    if not np.array_equal(saved.y.to_numpy(float), metadata.y.to_numpy(float)):
        raise ValueError('C0 native label lineage differs from canonical metadata')
    if not np.array_equal(saved.best_of.to_numpy(int), series.best_of.to_numpy(int)):
        raise ValueError('C0 format differs')
    a, b = series.team_a.astype(str).to_numpy(), series.team_b.astype(str).to_numpy()
    one, two = metadata.team1_id.astype(str).to_numpy(), metadata.team2_id.astype(str).to_numpy()
    direct, reversed_ = (one == a) & (two == b), (one == b) & (two == a)
    if not np.all(direct ^ reversed_):
        raise ValueError('C0 participant identity does not bind to target orientation')
    y = np.where(reversed_, 1 - saved.y.to_numpy(float), saved.y.to_numpy(float))
    if not np.array_equal(y, series.winner.to_numpy(float)):
        raise ValueError('C0 winner orientation or labels differ')
    probability = saved['w32_e12__full'].to_numpy(float)
    return np.where(reversed_, 1 - probability, probability), reversed_


def _c0_prediction(series):
    from src.models.champion_history_model import HistoryPrediction
    from src.models.tournament_scorelines import ConditionalScorelineModel
    _, metadata = _reference_metadata()
    p, _ = _align_c0_reference(pd.read_parquet(ROOT / C0_OUTPUT), metadata, series)
    law = ConditionalScorelineModel(beta_momentum=.35)
    scores = np.zeros((len(series), 6))
    for i, (probability, bo) in enumerate(zip(p, series.best_of.to_numpy(int))):
        needed = bo // 2 + 1
        pmf = law.scoreline_pmf(float(probability), int(bo))
        for lost in range(needed):
            scores[i, lost] = pmf[(needed, lost)]
            scores[i, 3+lost] = pmf[(lost, needed)]
    prediction = HistoryPrediction(score_probs=scores, best_of=series.best_of.to_numpy(int))
    if not np.allclose(prediction.p_win, p, atol=1e-14, rtol=0):
        raise ValueError('Coherent reference changed C0 series probability')
    return prediction


def _paired(candidate, control, metric, *, confidence=CONFIDENCE):
    from src.analysis.champion_history_evaluation import paired_comparison
    result = paired_comparison(candidate, control, metric, confidence=confidence, draws=5000, seed=20261007)
    return {**result, 'status': 'SCORED' if result['status'] == 'available' else 'INSUFFICIENT_EVIDENCE',
            'match_weighted': {'delta': result['match_weighted_delta'], 'interval': result['ci']},
            'equal_month': {'delta': result['equal_month_delta'], 'interval': result['equal_month_ci']}}


def _passes(comparisons, key):
    item = comparisons[key]
    return item['status'] == 'SCORED' and item['match_weighted']['interval'][1] < 0


def _nonincrease(a, b, metric):
    aa, bb = a[metric].to_numpy(float), b[metric].to_numpy(float)
    valid = ~np.isnan(aa) & ~np.isnan(bb)
    return bool(valid.any() and np.mean(aa[valid]-bb[valid]) <= 0)


def _coverage_guard(candidate, control, task):
    reasons = []
    for level, minimum in ((50, .45), (80, .75)):
        metric = f'{task}_coverage_{level}'
        aa, bb = candidate[metric].to_numpy(float), control[metric].to_numpy(float)
        valid = ~np.isnan(aa) & ~np.isnan(bb)
        if not valid.any():
            reasons.append(f'{level}% coverage absent')
            continue
        ca, cb = float(aa[valid].mean()), float(bb[valid].mean())
        if ca < minimum or abs(ca-level/100) > abs(cb-level/100)+.025:
            reasons.append(f'{level}% coverage guard: candidate={ca}, control={cb}')
    return reasons


def run_experiment(run_id=RUN_ID):
    from src.models.champion_history_model import fit_history_model, calibrate_history_ensemble
    from src.analysis.champion_history_evaluation import evaluate_prediction, evaluate_first_map_baseline
    paths, protocol, binding, series, maps, arrays = load_frozen(run_id)
    if not (paths['08_reporting'] / 'software_smoke.json').exists():
        raise ValueError('TRAIN-only actual smoke is required before formal fits')
    for stage in ('06_models', '07_model_output'):
        paths[stage].mkdir(parents=True, exist_ok=True)
    execution_path = paths['08_reporting'] / 'execution.json'
    if execution_path.exists():
        raise FileExistsError('Formal execution already started; no automatic retry')
    split = partitions(series)
    execution = {'run_id': run_id, 'started_at': _now(), 'status': 'RUNNING', 'formal_trajectories': [],
                 'calibration_fits': [], 'c0_fits': 0, 'assessment_fits': 0}
    _json(execution_path, execution)
    models = {name: [] for name in VARIANTS}
    for seed in SEEDS:
        for name, (champion, multitask) in VARIANTS.items():
            record = {'variant': name, 'seed': seed, 'started_at': _now(), 'status': 'RUNNING'}
            execution['formal_trajectories'].append(record)
            _json(execution_path, execution)
            print(f'FIT {name} seed={seed} epochs=12', flush=True)
            model = fit_history_model(arrays, series, maps, split['TRAIN'], config=protocol['config'],
                                      seed=seed, champion_enabled=champion, multitask=multitask)
            destination = paths['06_models'] / f'{name}_seed_{seed}.joblib'
            joblib.dump(model, destination)
            models[name].append(model)
            record.update(status='COMPLETED', completed_at=_now(), model_path=str(destination.relative_to(ROOT)),
                          sha256=sha256(destination), fit_receipt=model.fit_receipt)
            _json(execution_path, execution)
    cal_series, cal_maps = _subset(series, maps, split['CAL'])
    cal_scores, ensembles = {}, {}
    for name in VARIANTS:
        ensemble, receipt = calibrate_history_ensemble(models[name], arrays, series, maps, split['CAL'])
        ensembles[name] = ensemble
        execution['calibration_fits'].append({'variant': name, 'receipt': receipt})
        joblib.dump(ensemble, paths['06_models'] / f'{name}_ensemble.joblib')
        pred = ensemble.predict(arrays, split['CAL'])
        metrics, scores, map_scores, profiles = evaluate_prediction(pred, cal_series, cal_maps)
        cal_scores[name] = scores
        _json(paths['08_reporting'] / f'{name}_cal_metrics.json', metrics)
        scores.to_parquet(paths['07_model_output'] / f'{name}_cal_series_scores.parquet', index=False)
        _save_prediction(pred, cal_series, cal_maps, paths['07_model_output'], f'{name}_cal')
        _json(execution_path, execution)
    selection = {task: min(('B','D'), key=lambda v: float(cal_scores[v][metric].mean()))
                 for task, metric in (('winner','winner_nll'), ('length','length_nll'))}
    selection.update(kills='D', duration='D')
    _json(paths['06_models'] / 'selection.json', {'selected': selection, 'selected_at': _now(),
          'assessment_generated': False, 'cal_scores': {v: {m: float(cal_scores[v][m].mean())
          for m in ('winner_nll','length_nll','kills_nll','duration_nll')} for v in VARIANTS}})
    return _score_completed(paths, protocol, series, maps, arrays, ensembles, selection, execution)


def _score_completed(paths, protocol, series, maps, arrays, ensembles, selection, execution):
    from src.analysis.champion_history_evaluation import evaluate_prediction, evaluate_first_map_baseline
    run_id = execution['run_id']
    split = partitions(series)
    execution_path = paths['08_reporting'] / 'execution.json'
    assess_series, assess_maps = _subset(series, maps, split['ASSESS'])
    scores, metrics_all, profile_frames = {}, {}, []
    for name, ensemble in ensembles.items():
        pred = ensemble.predict(arrays, split['ASSESS'])
        metrics, ss, ms, profiles = evaluate_prediction(pred, assess_series, assess_maps)
        scores[name], metrics_all[name] = ss, metrics
        ss.to_parquet(paths['07_model_output'] / f'{name}_series_scores.parquet', index=False)
        ms.to_parquet(paths['07_model_output'] / f'{name}_map_scores.parquet', index=False)
        profiles['candidate'] = name
        profile_frames.append(profiles)
        _save_prediction(pred, assess_series, assess_maps, paths['07_model_output'], name)
        if name == 'D':
            first_maps = assess_maps.loc[assess_maps.map_index == 1]
            fm, fs, _, _ = evaluate_prediction(pred, assess_series, first_maps)
            scores['D_first'], metrics_all['D_first'] = fs, fm
    c0 = _c0_prediction(assess_series)
    c0_metrics, c0_scores, _, c0_profiles = evaluate_prediction(c0, assess_series, assess_maps)
    scores['C0'], metrics_all['C0_coherent_reference'] = c0_scores, c0_metrics
    c0_profiles['candidate'] = 'C0_coherent_reference'
    profile_frames.append(c0_profiles)
    _save_prediction(c0, assess_series, assess_maps, paths['07_model_output'], 'C0_coherent_reference')
    base = pd.read_parquet(ROOT / BASE).iloc[split['ASSESS']].copy().reset_index(drop=True)
    old = _old_prediction(base)
    old_component_rows = []
    for component, (weight, marginal) in enumerate(zip(old.weights, old.components)):
        row = assess_series[['row_index', 'match_id', 'date', 'best_of']].copy()
        row['component'], row['weight'] = component, weight
        for field in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma'):
            row[field] = getattr(marginal, field)
        for column in range(3):
            row[f'length_p_{column}'] = marginal.length_probs[:, column]
        old_component_rows.append(row)
    pd.concat(old_component_rows, ignore_index=True).to_parquet(
        paths['07_model_output'] / 'old_props_components.parquet', index=False)
    old_metrics, old_scores, old_maps, old_profiles = evaluate_first_map_baseline(old, assess_series, assess_maps)
    scores['old_props'], metrics_all['old_props'] = old_scores, old_metrics
    old_scores.to_parquet(paths['07_model_output'] / 'old_props_series_scores.parquet', index=False)
    old_maps.to_parquet(paths['07_model_output'] / 'old_props_first_map_scores.parquet', index=False)
    c0_scores.to_parquet(paths['07_model_output'] / 'C0_series_scores.parquet', index=False)
    scores['D_first'].to_parquet(paths['07_model_output'] / 'D_first_series_scores.parquet', index=False)
    old_profiles['candidate'] = 'old_props'
    profile_frames.append(old_profiles)
    comparisons = {}
    for candidate, control, metric in protocol['primary_contrasts']:
        key = f'{candidate}-{control}:{metric}'
        comparisons[key] = _paired(scores[candidate], scores[control], metric)
    diagnostics = {}
    for candidate, control in (('C','A'), ('D','B')):
        for metric in ('winner_nll','length_nll'):
            diagnostics[f'{candidate}-{control}:{metric}'] = _paired(scores[candidate], scores[control], metric, confidence=.95)
    task_decisions = {}
    for task in ('winner','length'):
        candidate = selection[task]
        control = 'A' if candidate == 'B' else 'C'
        reference = 'C0' if task == 'winner' else 'old_props'
        required = [f'{candidate}-{control}:{task}_nll', f'{candidate}-{reference}:{task}_nll']
        guard_metric = 'winner_brier' if task == 'winner' else 'length_rps'
        guards = all(_nonincrease(scores[candidate], scores[c], guard_metric) for c in (control,reference))
        passed = all(_passes(comparisons, k) for k in required) and guards
        task_decisions[task] = {'selected_variant': candidate, 'decision': 'ADOPT' if passed else 'NOT_ADOPTED',
                                'primary_contrasts': required, 'point_guard_passed': guards}
    for task in ('kills','duration'):
        required = [f'D-C:{task}_nll', f'D_first-old_props:{task}_nll']
        crps = (_nonincrease(scores['D'], scores['C'], f'{task}_crps') and
                _nonincrease(scores['D_first'], scores['old_props'], f'{task}_crps'))
        coverage = _coverage_guard(scores['D'], scores['C'], task) + _coverage_guard(scores['D_first'], scores['old_props'], task)
        passed = all(_passes(comparisons,k) for k in required) and crps and not coverage
        task_decisions[task] = {'selected_variant': 'D', 'decision': 'ADOPT' if passed else 'NOT_ADOPTED',
                                'primary_contrasts': required, 'crps_guard_passed': crps, 'coverage_failures': coverage,
                                'later_map_existing_reference': 'UNAVAILABLE; all-map matched control comparison retained'}
    bundle = all(v['decision'] == 'ADOPT' for v in task_decisions.values()) and selection['winner'] == selection['length']
    _json(paths['08_reporting'] / 'metrics.json', metrics_all)
    _json(paths['08_reporting'] / 'comparisons.json', comparisons)
    _json(paths['08_reporting'] / 'multitask_diagnostics.json', diagnostics)
    pd.concat(profile_frames, ignore_index=True).to_csv(paths['08_reporting'] / 'profiles.csv', index=False)
    formal_count = len(execution['formal_trajectories'])
    seed_epochs = sum(r['fit_receipt']['epochs_completed'] for r in execution['formal_trajectories'])
    cal_count = sum(r['receipt']['scalar_count'] for r in execution['calibration_fits'])
    if formal_count != 12 or seed_epochs != 144 or cal_count != 8:
        raise ValueError('Actual fit accounting differs from the frozen experiment')
    execution.update(status='COMPLETED', completed_at=_now(), formal_fit_count=formal_count,
                     formal_seed_epochs=seed_epochs, ensemble_cal_scalar_fit_count=cal_count, software_fit_count=1)
    _json(execution_path, execution)
    _json(paths['08_reporting'] / 'final_integrity.json', _integrity(paths))
    decision = {'run_id': run_id, 'status': 'COMPLETED', 'decision': 'ADOPT' if bundle else 'NOT_ADOPTED',
                'tasks': task_decisions, 'overall_bundle_gate_passed': bundle,
                'evidence_status': 'DEVELOPMENT_ONLY', 'selection': selection,
                'formal_fits': formal_count, 'calibration_fits': cal_count, 'c0_fits': 0, 'assessment_fits': 0,
                'no_joint_kills_duration_claim': True, 'no_production_or_betting_adoption': True,
                'output_hashes': {str(p.relative_to(ROOT)): sha256(p) for stage in ('06_models','07_model_output')
                                  for p in sorted(paths[stage].iterdir()) if p.is_file()}}
    _json(paths['08_reporting'] / 'decision.json', decision)
    return decision


def score_saved_run(run_id=RUN_ID):
    """Complete an explicitly bound scoring repair without neural or CAL fits."""
    paths, protocol, _, series, maps, arrays = load_frozen(run_id)
    if not (paths['05_model_input'] / 'scoring_continuation_binding.json').exists():
        raise ValueError('An explicit scoring-only continuation binding is required')
    started = paths['08_reporting'] / 'scoring_continuation_start.json'
    if started.exists() or (paths['08_reporting'] / 'decision.json').exists():
        raise FileExistsError('Scoring continuation already started; no automatic retry')
    execution = json.loads((paths['08_reporting'] / 'execution.json').read_text())
    if (len(execution['formal_trajectories']) != 12 or
            any(r['status'] != 'COMPLETED' for r in execution['formal_trajectories']) or
            sum(r['receipt']['scalar_count'] for r in execution['calibration_fits']) != 8):
        raise ValueError('Completed frozen models and eight CAL scalars are required')
    for record in execution['formal_trajectories']:
        if sha256(ROOT / record['model_path']) != record['sha256']:
            raise ValueError('Saved trajectory changed after fitting')
    ensembles = {v: joblib.load(paths['06_models'] / f'{v}_ensemble.joblib') for v in VARIANTS}
    for record in execution['calibration_fits']:
        if ensembles[record['variant']].calibration_receipt != record['receipt']:
            raise ValueError('Saved CAL receipt differs from executed fitting')
    selection = json.loads((paths['06_models'] / 'selection.json').read_text())['selected']
    _json(started, {'started_at': _now(), 'new_neural_fits': 0, 'new_CAL_fits': 0,
                    'models': {str(p.relative_to(ROOT)): sha256(p)
                               for p in paths['06_models'].iterdir() if p.is_file()}})
    decision = _score_completed(paths, protocol, series, maps, arrays, ensembles, selection, execution)
    original = json.loads((paths['08_reporting'] / 'execution_interrupted.json').read_text())
    for old in original['formal_trajectories']:
        if old['status'] == 'COMPLETED':
            record = next(r for r in execution['formal_trajectories']
                          if r['variant'] == old['variant'] and r['seed'] == old['seed'])
            record.update(training_started_at=old['started_at'], training_completed_at=old['completed_at'],
                          restored_checkpoint_sha256=old['sha256'], checkpoint_restored_without_fit=True)
    accounting = {'formal_completed_trajectories': 12, 'formal_fit_attempts': 13,
                  'interrupted_fit_attempts': 1, 'restored_completed_checkpoints': 3,
                  'resumed_actual_fit_attempts': 9, 'interrupted_partial_seed_epochs': 'UNKNOWN',
                  'authorization_sha256': sha256(paths['05_model_input'] / 'resume_authorization.json'),
                  'original_execution_sha256': sha256(paths['08_reporting'] / 'execution_interrupted.json'),
                  'no_recipe_or_gate_change': True, 'scoring_repair_new_fits': 0}
    execution.update(accounting)
    decision.update(accounting)
    _json(paths['08_reporting'] / 'execution.json', execution)
    _json(paths['08_reporting'] / 'decision.json', decision)
    _json(paths['08_reporting'] / 'resumption_completed.json',
          {**accounting, 'status': 'COMPLETED', 'completed_at': _now()})
    return decision


def predict_saved(run_id, match_ids, variant='D'):
    paths, protocol, binding, series, maps, arrays = load_frozen(run_id)
    decision = json.loads((paths['08_reporting'] / 'decision.json').read_text())
    model_path = paths['06_models'] / f'{variant}_ensemble.joblib'
    expected = decision['output_hashes'][str(model_path.relative_to(ROOT))]
    if sha256(model_path) != expected:
        raise ValueError('Saved ensemble differs from executed binding')
    indices = []
    for match_id in match_ids:
        found = np.flatnonzero(series.match_id.astype(str).to_numpy() == str(match_id))
        if len(found) != 1:
            raise ValueError('ID must identify one canonical row')
        if not str(series.iloc[found[0]].date).startswith('2026-'):
            raise ValueError('This recipe is a 2026 historical research model, not earlier-origin inference')
        indices.append(int(found[0]))
    prediction = joblib.load(model_path).predict(arrays, np.asarray(indices))
    rows = []
    for i, index in enumerate(indices):
        row = {'match_id': str(series.iloc[index].match_id), 'best_of': int(prediction.best_of[i]),
               'winner_probability': float(prediction.p_win[i]), 'score_probabilities': prediction.score_probs[i],
               'length_probabilities': prediction.length_probs[i], 'map_play_probabilities': prediction.play_probs[i],
               'scenario': 'actual_first_map_roster_daily_release', 'joint_kills_duration': False,
               'map_distributions': []}
        for m in range(1, int(prediction.best_of[i])+1):
            try:
                marginal = prediction.map_prediction(m)
            except NotImplementedError:
                break
            row['map_distributions'].append({'map_index': m, 'probability_played': float(prediction.play_probs[i,m-1]),
                'conditional_on_played': True,
                'p_kills_over': {str(line): float(marginal.kills_over(line)[i]) for line in (20.5,30.5,40.5)},
                'p_duration_over_minutes': {str(line): float(marginal.duration_over(line)[i]) for line in (25,30,35,40)}})
        rows.append(row)
    return _clean({'run_id': run_id, 'variant': variant, 'predictions': rows})
