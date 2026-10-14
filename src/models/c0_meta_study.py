"""Frozen-base, role-aware thirty-day meta research experiment.

The source/training contracts of reused C0 artifacts stay intact. This study
binds its current forward implementation separately and replays every saved
native probability before fitting either correction. No database or network.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = 'rating-foundation-20260915/data/08_reporting/paired.parquet'
BANK = 'a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank'
ANNUAL_RUN = 'model_reliability_20261001_055321'
CONTROL_RUN = 'a0_data_focus_20260929_192443'
SEEDS = (20261007, 20261008, 20261009)
BASE_SEEDS = (20260910, 20260911, 20260912)
BRANCHES = ('player', 'meta')
SOURCE_FILES = (
    'src/features/c0_meta_features.py', 'src/features/c0_meta_source.py',
    'src/models/c0_meta.py', 'src/models/c0_meta_study.py',
    'src/analysis/c0_meta_evaluation.py', 'scripts/train_c0_meta.py',
    'scripts/run_model_benchmark.py', 'src/analysis/ev_asof_replay.py',
    'src/models/a0_data_focus.py', 'src/models/a0_uncertainty.py',
    'src/models/a0_native_refit.py', 'src/models/a0_reliability.py',
    'src/models/a0_target_experiment.py', 'src/models/a0_accelerator.py',
    'src/analysis/model_benchmark.py', 'src/analysis/research_benchmark.py',
    'src/analysis/production_market_benchmark.py', 'src/analysis/bootstrap.py',
    'betting_app/core/models/native_c0.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _qualified(path, root):
    path = Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        return 'project:' + path.relative_to(PROJECT_ROOT).as_posix()
    return 'research:' + path.relative_to(root).as_posix()


def _resolve(key, root):
    if key.startswith('project:'):
        return PROJECT_ROOT / key[8:]
    if key.startswith('research:'):
        return root / key[9:]
    path = Path(key)
    return path if path.is_absolute() else root / path


def _root(value=None):
    from src.models.a0_target_experiment import resolve_research_root
    return resolve_research_root(value)


def _paths(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be one path component')
    return {name: PROJECT_ROOT / 'data' / stage / 'c0_meta' / run_id for name, stage in (
        ('features', '04_feature'), ('input', '05_model_input'),
        ('models', '06_models'), ('output', '07_model_output'), ('report', '08_reporting'))}


def temporal_partitions(frame):
    """Keep every target; exclude only ineligible labels from closed fitting."""
    dates = pd.to_datetime(frame.date, errors='raise')
    released = pd.to_datetime(frame.get('effective_label_release_day', frame.result_day), errors='raise')
    if dates.isna().any() or released.isna().any() or (released < dates).any():
        raise ValueError('invalid target/release chronology')
    if not dates.dt.year.isin((2024, 2025, 2026)).all():
        raise ValueError('correction cohort must be 2024–2026')
    masks = {
        'train': dates.dt.year.eq(2024) & released.lt('2025-01-01'),
        'cal': dates.dt.year.eq(2025) & released.lt('2026-01-01'),
        'assessment': dates.dt.year.eq(2026),
    }
    masks['excluded_fit_labels'] = ~masks['train'] & ~masks['cal'] & ~masks['assessment']
    return {name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()}


@dataclass(frozen=True)
class MetaNormalizer:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    count_mean: float
    count_scale: float
    support_mean: float
    support_scale: float

    @classmethod
    def fit(cls, arrays, train):
        indices = np.asarray(train, dtype=np.int64)
        if not len(indices):
            raise ValueError('normalizer requires closed TRAIN rows')
        means, scales = [], []
        for stat in range(12):
            values = []
            for name in ('player_stats', 'meta_stats'):
                data = np.asarray(arrays[name])[indices]
                mask = data[..., 24 + stat] > 0
                if name == 'player_stats':
                    mask &= np.asarray(arrays['player_valid'])[indices]
                values.append(data[..., stat][mask].astype(np.float64))
            observed = np.concatenate(values)
            means.append(float(observed.mean()) if len(observed) else 0.)
            sd = float(observed.std()) if len(observed) else 0.
            scales.append(sd if sd > 1e-6 else 1.)
        counts = np.concatenate([np.asarray(arrays[name])[indices, ..., 36].ravel().astype(float)
                                 for name in ('player_stats', 'meta_stats')])
        support = np.log1p(np.asarray(arrays['meta_support'])[indices, ..., 0].ravel().astype(float))
        return cls(tuple(means), tuple(scales), float(counts.mean()), max(float(counts.std()), 1e-6),
                   float(support.mean()), max(float(support.std()), 1e-6))

    def to_dict(self):
        return {name: list(getattr(self, name)) if name in ('mean', 'scale') else getattr(self, name)
                for name in ('mean', 'scale', 'count_mean', 'count_scale', 'support_mean', 'support_scale')}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value['mean']), tuple(value['scale']), value['count_mean'], value['count_scale'],
                   value['support_mean'], value['support_scale'])

    def transform(self, arrays):
        result = dict(arrays)
        mean, scale = np.asarray(self.mean), np.asarray(self.scale)
        for name in ('player_stats', 'meta_stats'):
            value = np.asarray(arrays[name], dtype=np.float32).copy()
            known = value[..., 24:36] > 0
            value[..., :12] = np.where(known, (value[..., :12] - mean) / scale, 0)
            value[..., 12:24] = np.where(known, value[..., 12:24] / scale, 0)
            value[..., 36] = (value[..., 36] - self.count_mean) / self.count_scale
            if name == 'player_stats':
                value = np.where(np.asarray(arrays['player_valid'])[..., None], value, 0)
            result[name] = value.astype(np.float32, copy=False)
        support = np.asarray(arrays['meta_support'], dtype=np.float32).copy()
        support[..., 0] = (np.log1p(support[..., 0]) - self.support_mean) / self.support_scale
        result['meta_support'] = support
        patch = np.asarray(arrays['meta_patch'], dtype=np.float32).copy()
        patch[..., 0] = np.log1p(patch[..., 0])
        result['meta_patch'] = patch
        return result


def combine_seed_logits(base_probability, seed_logits):
    base = np.asarray(base_probability, dtype=float)
    logits = np.asarray(seed_logits, dtype=float)
    if base.ndim != 1 or logits.ndim != 2 or logits.shape[1] != len(base) or not len(logits):
        raise ValueError('seed logits must align with base probabilities')
    if not np.isfinite(logits).all() or not np.isfinite(base).all() or ((base <= 0) | (base >= 1)).any():
        raise ValueError('finite logits and strict-interior base probabilities required')
    return logit(base) + (logits - logit(base)[None]).mean(axis=0)


def fit_amplitude(base_probability, correction, labels):
    base, delta, y = map(lambda a: np.asarray(a, dtype=float), (base_probability, correction, labels))
    if base.ndim != 1 or not len(base) or delta.shape != base.shape or y.shape != base.shape:
        raise ValueError('CAL arrays must be nonempty aligned vectors')
    if not np.isfinite(base).all() or not np.isfinite(delta).all() or ((base <= 0) | (base >= 1)).any() or not np.isin(y, (0, 1)).all():
        raise ValueError('CAL requires valid probabilities, corrections and binary labels')
    z = logit(base)
    derivative = lambda a: float(np.mean((expit(z + a * delta) - y) * delta))
    left, right = derivative(0.), derivative(1.)
    amplitude = 0. if left >= 0 else 1. if right <= 0 else float(brentq(derivative, 0., 1., xtol=1e-14))
    objective = lambda a: float(np.mean(np.logaddexp(0., z + a * delta) - y * (z + a * delta)))
    return dict(amplitude=amplitude, cal_log_loss=objective(amplitude), identity_log_loss=objective(0.),
                derivative_at_zero=left, derivative_at_one=right, rows=len(y))


def _probabilities(base, delta, amplitude):
    # Identity and unsupported cold corrections preserve native probability bits.
    result = np.asarray(base, dtype=float).copy()
    if amplitude != 0:
        active = np.asarray(delta) != 0
        result[active] = expit(logit(result[active]) + amplitude * np.asarray(delta)[active])
    return result


def _origin_records():
    records = []
    for year in (2024, 2025, 2026):
        family, run = ('model_reliability', ANNUAL_RUN) if year != 2026 else ('a0_data_focus', CONTROL_RUN)
        suffix = f'/neural/{year}' if year != 2026 else ''
        records.append(dict(year=year,
            model_root=f'data/06_models/{family}/{run}{suffix}',
            completion_path=f'data/08_reporting/{family}/{run}{suffix}/completed.json',
            control_path=f'data/07_model_output/{family}/{run}{suffix}/control.parquet',
            source_snapshot_path=f'data/05_model_input/model_reliability/{ANNUAL_RUN}/neural_sources.zip' if year != 2026 else None))
    return records


def _audit_origins(root, metadata):
    """Original receipts, input hashes, released memberships and archive bytes."""
    from src.models.a0_reliability import annual_partitions
    from src.models.a0_native_refit import _contract_digest
    from betting_app.core.models.native_c0 import _COMPLETION_SHA256
    completion_pins = {
        2024: '446ad1bc23c12e39e1f43c6d80a411c6d5231c7f76b60ec9bbec2399cedc9e38',
        2025: 'ec8e68ed453e4fc708ef2a8674c43b8b9f366f191bd773eafc2a043bb0fa561c',
        2026: _COMPLETION_SHA256,
    }
    inputs, records = {}, _origin_records()
    for record in records:
        model_root = PROJECT_ROOT / record['model_root']
        execution_path = model_root / 'execution.json'
        completion_path = PROJECT_ROOT / record['completion_path']
        if sha(completion_path) != completion_pins[record['year']]:
            raise ValueError('original C0 completion receipt changed')
        execution = json.loads(execution_path.read_text())
        completion = json.loads(completion_path.read_text())
        contract = execution['contract']
        if completion.get('contract') != contract or completion.get('status') != 'TRAINING_COMPLETE_NOT_SCORED':
            raise ValueError(f'incomplete original control: {record["year"]}')
        split = annual_partitions(metadata, record['year'])
        expected = {name: _contract_digest(metadata.iloc[ix].golgg_match_id.astype(str).tolist()) for name, ix in split.items()}
        if expected != contract['split_ids_sha256']:
            raise ValueError('origin membership differs from original C0 training')
        for key, expected_sha in contract['input_sha256'].items():
            path = _resolve(key, root)
            normalized = _qualified(path, root)
            if normalized not in inputs and sha(path) != expected_sha:
                raise ValueError(f'original C0 input changed: {key}')
            if normalized in inputs and inputs[normalized] != expected_sha:
                raise ValueError('C0 original input contracts disagree')
            inputs[normalized] = expected_sha
        for key, expected_sha in contract['source_sha256'].items():
            if key.startswith('research:'):
                if sha(_resolve(key, root)) != expected_sha:
                    raise ValueError(f'original archive backend changed: {key}')
                inputs[key] = expected_sha
        snapshot = record['source_snapshot_path']
        if snapshot:
            with zipfile.ZipFile(PROJECT_ROOT / snapshot) as archive:
                for key, expected_sha in contract['source_sha256'].items():
                    if hashlib.sha256(archive.read(key.replace(':', '/'))).hexdigest() != expected_sha:
                        raise ValueError(f'original training snapshot mismatch: {key}')
            inputs['project:' + snapshot] = sha(PROJECT_ROOT / snapshot)
        selected = [model_root / 'shared/model.joblib', model_root / 'control/calibration/model.joblib',
                    *[model_root / 'control' / f'seed_{seed}/model.joblib' for seed in BASE_SEEDS]]
        for path in selected:
            receipt_path = path.with_suffix('.joblib.receipt.json')
            receipt = json.loads(receipt_path.read_text())
            if receipt['checkpoint_sha256'] != sha(path):
                raise ValueError(f'original checkpoint changed: {path}')
            for item in (path, receipt_path):
                original_pin = completion['artifacts'].get(item.relative_to(PROJECT_ROOT).as_posix())
                if original_pin != sha(item):
                    raise ValueError(f'original completed asset changed: {item}')
                inputs[_qualified(item, root)] = sha(item)
        for path in (execution_path, completion_path, PROJECT_ROOT / record['control_path']):
            if path != completion_path and completion['artifacts'].get(path.relative_to(PROJECT_ROOT).as_posix()) != sha(path):
                raise ValueError(f'original completion no longer binds asset: {path}')
            inputs[_qualified(path, root)] = sha(path)
        record['training_source_sha256'] = contract['source_sha256']
        record['train_max_release_day'] = str(metadata.iloc[split['train']].result_day.max())
        record['cal_max_release_day'] = str(metadata.iloc[split['cal']].result_day.max())
        record['retrospective_artifact_not_historical_issuance'] = True
    return records, inputs


def freeze_protocol(run_id, research_root=None):
    root, paths = _root(research_root), _paths(run_id)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('new run requires nonexistent stage directories')
    metadata = pd.read_parquet(root / BANK / 'metadata.parquet')
    metadata['golgg_match_id'] = metadata.golgg_match_id.astype(str)
    metadata['date'] = pd.to_datetime(metadata.date).dt.strftime('%Y-%m-%d')
    metadata['result_day'] = pd.to_datetime(metadata.result_day).dt.strftime('%Y-%m-%d')
    origins, inputs = _audit_origins(root, metadata)
    raw = PROJECT_ROOT / 'data/artifacts/golgg-database-recovery-20260909/matches.json'
    reader = root / 'unified-ratings-20260911/code/scripts/local_history_stream.py'
    from betting_app.core.models.native_c0 import _CANONICAL_RAW_HISTORY_SHA256
    if sha(raw) != _CANONICAL_RAW_HISTORY_SHA256:
        raise ValueError('frozen canonical raw source changed')
    for path in (raw, reader, root / CANONICAL, PROJECT_ROOT / 'conf/base/research_benchmark.json'):
        inputs[_qualified(path, root)] = sha(path)
    sources = {'project:' + relative: sha(PROJECT_ROOT / relative) for relative in SOURCE_FILES}
    sources.update({key: value for key, value in inputs.items() if key.startswith('research:') and key.endswith('.py')})
    paths['input'].mkdir(parents=True)
    snapshot = paths['input'] / 'sources.zip'
    with zipfile.ZipFile(snapshot, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sources:
            archive.write(_resolve(key, root), key.replace(':', '/'))
    relative = lambda path: path.relative_to(PROJECT_ROOT).as_posix()
    protocol = dict(schema='c0_role_meta_v1', run_id=run_id, status='FROZEN_BEFORE_FITTING',
        created_at_utc=datetime.now(timezone.utc).isoformat(), scope='EXPOSED_HISTORICAL_DEVELOPMENT',
        source_sha256=sources, input_sha256=inputs, source_snapshot_path=relative(snapshot), source_snapshot_sha256=sha(snapshot),
        origins=origins, canonical_path=CANONICAL, raw_path=relative(raw), reader_path=reader.relative_to(root).as_posix(),
        release_path=BANK + '/release_days.json', protocol_path=relative(paths['input'] / 'protocol.json'),
        features_path=relative(paths['features'] / 'features.npz'), frame_path=relative(paths['features'] / 'metadata.parquet'),
        diagnostics_path=relative(paths['features'] / 'provenance.pkl'), raw_source_path=relative(paths['features'] / 'raw_source.pkl'),
        prepared_receipt_path=relative(paths['features'] / 'prepared.json'), models_path=relative(paths['models']),
        predictions_path=relative(paths['output'] / 'predictions.parquet'), execution_path=relative(paths['report'] / 'execution.json'),
        training=dict(seeds=list(SEEDS), branches=list(BRANCHES), epochs=12, batch_size=128, threads=2,
                      lr=.001, weight_decay=.01, shuffle_closed_train=True, objective='binary_series_log_loss',
                      base_fits=0, neural_fits=6, scalar_calibration_fits=2, max_logit_correction=.5),
        features=dict(window_days=30, player_summary_window_days=30, competition_prior=100,
                      backbone='fixed eligible 2024 native12 three-seed hidden concatenation96',
                      champion_vocabulary='observed and series-released strictly before2024',
                      target_patch='evaluation only; no authenticated advance announcement',
                      target_roster='actual first-map five conditional daily scenario, not publication certification',
                      patches='explicit historical source labels: coverage/diversity/concentration, no date inference',
                      player_only='same topology/count, zero meta; not equal effective parameter capacity'),
        partitions=dict(train_year=2024, cal_year=2025, assessment_year=2026, expected_rows=11550,
                        expected_assessment_rows=3103, fitting_labels_strictly_before_next_year=True),
        calibration='one convex identity-bounded amplitude on mean seed logit correction per branch, CAL2025 only',
        evaluation=dict(draws=5000, seed=20261007, primary_confidence=.975, probability_epsilon=None,
                        two_primary_contrasts=['meta_minus_c0', 'meta_minus_player'],
                        primary_ll_upper_below=0., primary_brier_delta_at_most=0.,
                        source_proxy='tournament source name/year, not independently certified editions',
                        patch_transition='first7days since historical exact-competition observation of target explicit patch; evaluation only'),
        protected=dict(canonical_manifest_unchanged=True, original_artifacts_unchanged=True,
                       live_acquisition=False, deployment=False, betting_claim=False))
    _json(paths['input'] / 'protocol.json', protocol)
    print(json.dumps(dict(protocol=protocol['protocol_path'], sha256=sha(paths['input'] / 'protocol.json'))), flush=True)
    return paths['input'] / 'protocol.json'


def _verify_protocol(protocol_path, root):
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol['schema'] != 'c0_role_meta_v1' or protocol['features']['window_days'] != 30:
        raise ValueError('unsupported meta recipe')
    for inventory in ('source_sha256', 'input_sha256'):
        for key, expected in protocol[inventory].items():
            if sha(_resolve(key, root)) != expected:
                raise ValueError(f'frozen {inventory} changed: {key}')
    if sha(PROJECT_ROOT / protocol['source_snapshot_path']) != protocol['source_snapshot_sha256']:
        raise ValueError('frozen study source snapshot changed')
    return protocol


def _load_control(record, root):
    from src.models import a0_data_focus as native
    contract = json.loads((PROJECT_ROOT / record['model_root'] / 'execution.json').read_text())['contract']
    backend, hashes = native.uncertainty._activate_archive_backend(root)
    for key, value in hashes.items():
        if key.startswith('research:') and contract['source_sha256'].get(key) != value:
            raise ValueError('C0 original forward backend changed')
    model_root = PROJECT_ROOT / record['model_root']
    shared = native._load_checkpoint(model_root / 'shared/model.joblib', {**contract, 'component': 'shared'})
    models = [native._load_checkpoint(model_root / 'control' / f'seed_{seed}/model.joblib',
                {**contract, 'component': 'neural', 'variant': 'control', 'seed': seed}) for seed in BASE_SEEDS]
    calibration = native._load_checkpoint(model_root / 'control/calibration/model.joblib',
                                         {**contract, 'component': 'calibration', 'variant': 'control'})
    return backend, shared, models, calibration


def prepare_features(protocol_path, research_root=None):
    if os.environ.get('BETTING_APP_C0_WORKER') != '1':
        raise RuntimeError('native preparation requires an isolated C0 worker namespace')
    import torch
    from src.models import a0_data_focus as native
    from src.features.c0_meta_source import read_raw_meta_source
    from src.features.c0_meta_features import build_meta_features
    from src.analysis.ev_asof_replay import _canonical_competition_context
    from src.analysis.research_benchmark import IDENTITY
    root = _root(research_root)
    protocol = _verify_protocol(protocol_path, root)
    paths = _paths(protocol['run_id'])
    if paths['features'].exists():
        raise FileExistsError(paths['features'])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native.uncertainty._activate_archive_backend(root)
    data = native.uncertainty._load_training_data(root)
    canonical = pd.read_parquet(root / CANONICAL)
    canonical['golgg_match_id'] = canonical.golgg_match_id.astype(str)
    if len(canonical) != 11550 or canonical.golgg_match_id.duplicated().any():
        raise ValueError('canonical cohort changed')
    bank_indices = data['frame'].set_index('golgg_match_id').index.get_indexer(canonical.golgg_match_id)
    if (bank_indices < 0).any():
        raise ValueError('canonical target absent from corrected feature bank')
    frame = data['frame'].iloc[bank_indices].reset_index(drop=True).copy()
    for column in IDENTITY:
        if not np.array_equal(frame[column].astype(str), canonical[column].astype(str)):
            raise ValueError(f'canonical identity mismatch: {column}')
    controls, max_error, replayed = [], 0., 0
    backbone = None
    for record in protocol['origins']:
        backend, shared, models, calibration = _load_control(record, root)
        saved = pd.read_parquet(PROJECT_ROOT / record['control_path'])
        saved['golgg_match_id'] = saved.golgg_match_id.astype(str)
        expected_ids = frame.loc[frame.date.str[:4].eq(str(record['year'])), 'golgg_match_id']
        if saved.golgg_match_id.duplicated().any() or set(saved.golgg_match_id) != set(expected_ids):
            raise ValueError('annual saved control coverage differs')
        saved = saved.set_index('golgg_match_id').loc[expected_ids].reset_index()
        indices = data['frame'].set_index('golgg_match_id').index.get_indexer(saved.golgg_match_id)
        for column in IDENTITY:
            if not np.array_equal(saved[column].astype(str), data['frame'].iloc[indices][column].astype(str)):
                raise ValueError(f'saved annual control orientation mismatch: {column}')
        for begin in range(0, len(indices), 256):
            ix = indices[begin:begin + 256]
            prediction = native.predict_models(backend, shared, models, calibration,
                {k: v[ix] for k, v in data['raw'].items()}, {k: v[ix] for k, v in data['history'].items()},
                data['frame'].iloc[ix].best_of.to_numpy(dtype=int))
            for component, values in prediction.items():
                reference = saved.iloc[begin:begin + len(ix)][[f'p_{component}_{mode}' for mode in native.MODES]].to_numpy(float)
                error = float(np.max(np.abs(values - reference)))
                if error > 2e-6:
                    raise ValueError('native control replay failed before meta fitting')
                max_error = max(max_error, error)
                replayed += int(np.asarray(values).size)
        controls.append(saved)
        if record['year'] == 2024:
            backbone = models
        print(json.dumps(dict(stage='native_replay', year=record['year'], rows=len(saved), max_error=max_error)), flush=True)
    combined = pd.concat(controls).set_index('golgg_match_id').loc[frame.golgg_match_id]
    frame['p_c0'] = combined.p_full_full.to_numpy(float)
    bounds = {record['year']: (record['train_max_release_day'], record['cal_max_release_day']) for record in protocol['origins']}
    frame['base_train_end'] = [bounds[int(day[:4])][0] for day in frame.date]
    frame['base_calibration_end'] = [bounds[int(day[:4])][1] for day in frame.date]
    for name in ('tier', 'family', 'history_stat_coverage', 'csd_xpd_coverage'):
        if name in combined:
            frame[name] = combined[name].to_numpy()
    profile = np.empty((len(frame), 2, 5, 96), dtype=np.float32)
    bank_best_of = data['frame'].best_of.to_numpy(dtype=int)
    for seed_index, model in enumerate(backbone):
        model.network.eval()
        captured = []
        hook = model.network.auxiliary_head.register_forward_pre_hook(lambda module, arguments: captured.append(arguments[0].detach()))
        try:
            with torch.no_grad():
                for begin in range(0, len(bank_indices), 256):
                    ix = bank_indices[begin:begin + 256]
                    inputs, _ = model.batch(data['raw'], data['history'], None, ix,
                                           bank_best_of, True, True)
                    captured.clear()
                    model.network(inputs, return_aux=True)
                    if len(captured) != 1 or tuple(captured[0].shape) != (len(ix), 2, 5, 32):
                        raise ValueError('native player representation hook contract changed')
                    profile[begin:begin + len(ix), ..., seed_index * 32:(seed_index + 1) * 32] = captured[0].numpy()
        finally:
            hook.remove()
    source = read_raw_meta_source(PROJECT_ROOT / protocol['raw_path'], root / protocol['release_path'], root / protocol['reader_path'])
    inventory = source.series.set_index('golgg_match_id').loc[frame.golgg_match_id]
    native_release = inventory.effective_label_release_day.astype(str).to_numpy()
    stored_release = frame.result_day.astype(str).to_numpy()
    frame['effective_label_release_day'] = np.where(native_release > stored_release, native_release, stored_release)
    targets = [dict(row_id=row.golgg_match_id, cutoff_day=row.date,
                    competition=_canonical_competition_context(row.tournament_name),
                    roster_a=row.roster_a, roster_b=row.roster_b, patch=patch)
               for row, patch in zip(frame.itertuples(), inventory.patch)]
    batch = build_meta_features(source.observations, targets, champion_ids=source.champion_ids)
    if list(batch.row_ids) != frame.golgg_match_id.tolist():
        raise ValueError('meta feature target identity alignment changed')
    arrays = {**batch.arrays, 'profile': profile, 'base_probability': frame.p_c0.to_numpy(float), 'best_of': frame.best_of.to_numpy(np.int64)}
    frame['c0_feature_source_max_day'] = frame.feature_source_max_day
    frame['feature_source_max_day'] = [
        max((value for value in (original, item['max_effective_release_day']) if pd.notna(value)), default=None)
        for original, item in zip(frame.c0_feature_source_max_day, batch.diagnostics)
    ]
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite prepared features: {key}')
    split = temporal_partitions(frame)
    frame['partition'] = ''
    frame['exclusion_reason'] = ''
    for name, ix in split.items():
        frame.loc[ix, 'partition'] = name
    frame.loc[split['excluded_fit_labels'], 'exclusion_reason'] = 'label_not_released_before_closed_fit_origin'
    frame['target_patch'] = inventory.patch.to_numpy()
    frame['patch_source'] = inventory.patch_source.to_numpy()
    frame['local_support'] = arrays['meta_support'][:, 1, :, 0].sum(axis=1)
    frame['local_missingness'] = frame.local_support.eq(0)
    frame['support_missingness'] = ~arrays['player_valid'].all(axis=(1, 2))
    frame['both_cold'] = ~arrays['player_valid'].any(axis=(1, 2))
    frame['confidence'] = np.maximum(frame.p_c0, 1 - frame.p_c0)
    frame['unknown_player_count'] = frame.unknown_players
    frame['new_org'] = frame.new_organization
    frame['patch_phase'] = patch_phases(source.observations, targets)
    normalizer = MetaNormalizer.fit(arrays, split['train'])
    paths['features'].mkdir(parents=True)
    np.savez_compressed(paths['features'] / 'features.npz', **arrays)
    frame.to_parquet(paths['features'] / 'metadata.parquet', index=False)
    with (paths['features'] / 'provenance.pkl').open('xb') as stream:
        pickle.dump(dict(diagnostics=batch.diagnostics, sources=batch.sources), stream, protocol=5)
    with (paths['features'] / 'raw_source.pkl').open('xb') as stream:
        pickle.dump(source, stream, protocol=5)
    _json(paths['features'] / 'normalizer.json', normalizer.to_dict())
    receipt = dict(status='PREPARED', protocol_sha256=sha(protocol_path),
        features_sha256=sha(paths['features'] / 'features.npz'), frame_sha256=sha(paths['features'] / 'metadata.parquet'),
        diagnostics_sha256=sha(paths['features'] / 'provenance.pkl'), raw_source_sha256=sha(paths['features'] / 'raw_source.pkl'),
        normalizer_sha256=sha(paths['features'] / 'normalizer.json'), vocabulary=list(source.champion_ids),
        source_audit=source.audit, partition_counts={k: len(v) for k, v in split.items()},
        base_replay_max_error=max_error, base_replayed_probabilities=replayed,
        fixed_backbone_year=2024, base_fits=0, rows=len(frame))
    _json(paths['features'] / 'prepared.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def patch_phases(observations, targets):
    """Evaluation-only phase from released exact-competition patch evidence.

    Age is measured from the earliest original played day observed in released
    retained history, not a presumed publisher patch-release date. A new patch
    without earlier local evidence is explicitly unsupported, not a transition.
    """
    records = sorted(observations, key=lambda row: max(row['played_day'], row['release_day']))
    first_seen, known_competitions = {}, set()
    phases = ['unknown'] * len(targets)
    cursor = 0
    for index in sorted(range(len(targets)), key=lambda i: targets[i]['cutoff_day']):
        target = targets[index]
        cutoff = target['cutoff_day']
        while cursor < len(records) and max(records[cursor]['played_day'], records[cursor]['release_day']) < cutoff:
            row = records[cursor]
            cursor += 1
            if isinstance(row['patch'], str) and row['patch']:
                key = (row['competition'], row['patch'])
                first_seen[key] = min(first_seen.get(key, row['played_day']), row['played_day'])
                known_competitions.add(row['competition'])
        patch, competition = target.get('patch'), target.get('competition')
        if not isinstance(patch, str) or not patch:
            continue
        first = first_seen.get((competition, patch))
        if first is None:
            phases[index] = 'unseen_in_prior_context' if competition in known_competitions else 'no_prior_patch_context'
        else:
            age = (pd.Timestamp(cutoff) - pd.Timestamp(first)).days
            phases[index] = 'first7days' if 0 <= age < 7 else 'later'
    return phases


def _prepared(protocol, protocol_path):
    paths = _paths(protocol['run_id'])
    receipt = json.loads((paths['features'] / 'prepared.json').read_text())
    if receipt['status'] != 'PREPARED' or receipt['protocol_sha256'] != sha(protocol_path):
        raise ValueError('prepared data belongs to another protocol')
    for filename, key in (('features.npz', 'features_sha256'), ('metadata.parquet', 'frame_sha256'),
                          ('provenance.pkl', 'diagnostics_sha256'), ('raw_source.pkl', 'raw_source_sha256'),
                          ('normalizer.json', 'normalizer_sha256')):
        if sha(paths['features'] / filename) != receipt[key]:
            raise ValueError(f'prepared artifact changed: {filename}')
    with np.load(paths['features'] / 'features.npz', allow_pickle=False) as bank:
        arrays = {key: bank[key] for key in bank.files}
    frame = pd.read_parquet(paths['features'] / 'metadata.parquet')
    normalizer = MetaNormalizer.from_dict(json.loads((paths['features'] / 'normalizer.json').read_text()))
    return arrays, frame, normalizer, receipt


def _tensors(arrays):
    import torch
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def _seed_logits(model, tensors, indices):
    import torch
    result = np.empty(len(indices), dtype=float)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(indices), 256):
            ix = indices[begin:begin + 256]
            result[begin:begin + len(ix)] = model({k: v[ix] for k, v in tensors.items()}).numpy()
    return result


def run_experiment(protocol_path, research_root=None, smoke=False):
    import torch
    from src.models.c0_meta import C0MetaCorrection
    root = _root(research_root)
    protocol = _verify_protocol(protocol_path, root)
    raw_arrays, frame, normalizer, prepared = _prepared(protocol, protocol_path)
    paths = _paths(protocol['run_id'] + '_smoke' if smoke else protocol['run_id'])
    if any(paths[name].exists() for name in ('models', 'output', 'report')):
        raise FileExistsError('execution stages already exist; no implicit retry/overwrite')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arrays = normalizer.transform(raw_arrays)
    tensors = _tensors(arrays)
    split = temporal_partitions(frame)
    train, cal = split['train'], split['cal']
    if smoke:
        train, cal = train[:128], cal[:64]
    if not len(train) or not len(cal):
        raise ValueError('closed TRAIN/CAL lack support')
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
    all_indices = np.arange(len(frame)) if not smoke else np.concatenate((train, cal, split['assessment'][:16]))
    base = arrays['base_probability'][all_indices]
    seeds = SEEDS[:1] if smoke else SEEDS
    epochs = 1 if smoke else protocol['training']['epochs']
    paths['models'].mkdir(parents=True)
    histories, calibrations, corrected, model_hashes = {}, {}, {}, {}
    restored_models = {}
    for branch in BRANCHES:
        logits = []
        restored_models[branch] = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = C0MetaCorrection(len(prepared['vocabulary']) + 1, use_meta=branch == 'meta')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
            rng = np.random.default_rng(seed)
            losses = []
            for epoch in range(epochs):
                model.train()
                order = rng.permutation(train)
                total = 0.
                for begin in range(0, len(order), 128):
                    ix = order[begin:begin + 128]
                    prediction = model({k: v[ix] for k, v in tensors.items()})
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, y[ix])
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite correction TRAIN loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                    optimizer.step()
                    total += float(loss.detach()) * len(ix)
                losses.append(total / len(train))
                print(json.dumps(dict(stage='smoke' if smoke else 'fit', branch=branch, seed=seed, epoch=epoch + 1, train_loss=losses[-1])), flush=True)
            path = paths['models'] / branch / f'seed_{seed}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                vocabulary_size=model.vocabulary_size, use_meta=model.use_meta, seed=seed, epochs=epochs,
                protocol_sha256=sha(protocol_path), normalizer=normalizer.to_dict(), train_losses=losses), path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            restored = C0MetaCorrection(checkpoint['vocabulary_size'], checkpoint['use_meta'])
            restored.load_state_dict(checkpoint['state_dict'], strict=True)
            restored.eval()
            reference = _seed_logits(model, tensors, all_indices)
            replay = _seed_logits(restored, tensors, all_indices)
            np.testing.assert_array_equal(reference, replay)
            logits.append(replay)
            restored_models[branch].append(restored)
            model_hashes[path.relative_to(PROJECT_ROOT).as_posix()] = sha(path)
            histories[f'{branch}:{seed}'] = losses
        combined = combine_seed_logits(base, np.asarray(logits))
        delta = combined - logit(base)
        delta = np.where(arrays['player_valid'][all_indices].any(axis=(1, 2)), delta, 0.)
        cal_positions = np.flatnonzero(np.isin(all_indices, cal))
        calibrations[branch] = fit_amplitude(base[cal_positions], delta[cal_positions], frame.y.to_numpy()[all_indices[cal_positions]])
        corrected[branch] = _probabilities(base, delta, calibrations[branch]['amplitude'])
    structural = _verify_saved_inference(restored_models, calibrations, tensors, all_indices, corrected)
    output = frame.iloc[all_indices].copy().reset_index(drop=True)
    for branch in BRANCHES:
        output[f'p_{branch}_development'] = corrected[branch]
        output[f'p_{branch}'] = np.where(output.partition.eq('assessment'), corrected[branch], output.p_c0)
    assessment = output.partition.eq('assessment').to_numpy()
    head_train_end = str(frame.iloc[train].effective_label_release_day.max())
    head_cal_end = str(frame.iloc[cal].effective_label_release_day.max())
    base_train = output.base_train_end.to_numpy(str)
    base_cal = output.base_calibration_end.to_numpy(str)
    output['train_end'] = np.where(assessment & (base_train < head_train_end), head_train_end, base_train)
    output['calibration_end'] = np.where(assessment & (base_cal < head_cal_end), head_cal_end, base_cal)
    output['correction_train_end'] = np.where(assessment, head_train_end, None)
    output['correction_calibration_end'] = np.where(assessment, head_cal_end, None)
    output['forecast_kind'] = np.where(assessment, 'correction_assessment', 'unchanged_c0_warmup')
    paths['output'].mkdir(parents=True)
    output.to_parquet(paths['output'] / 'predictions.parquet', index=False)
    calibration_path = paths['models'] / 'calibration.json'
    _json(calibration_path, calibrations)
    model_hashes[calibration_path.relative_to(PROJECT_ROOT).as_posix()] = sha(calibration_path)
    bundle_path = paths['models'] / 'bundle.json'
    inference_sources = {key: protocol['source_sha256'][key] for key in (
        'project:src/models/c0_meta.py', 'project:src/models/c0_meta_study.py')}
    _json(bundle_path, dict(schema='c0_meta_bundle_v1', protocol_sha256=sha(protocol_path),
        run_id=protocol['run_id'], smoke=smoke, seeds=list(seeds), vocabulary=prepared['vocabulary'],
        origin_day='2026-01-01', features_sha256=prepared['features_sha256'], frame_sha256=prepared['frame_sha256'],
        inference_source_sha256=inference_sources,
        normalizer=normalizer.to_dict(), calibrations=calibrations, model_sha256=model_hashes))
    model_hashes[bundle_path.relative_to(PROJECT_ROOT).as_posix()] = sha(bundle_path)
    if not smoke:
        _verify_protocol(protocol_path, root)
    execution = dict(schema='c0_meta_execution_v1', status='SMOKE_PASSED' if smoke else 'COMPLETED',
        run_id=protocol['run_id'], scope='EXPOSED_HISTORICAL_DEVELOPMENT', protocol_sha256=sha(protocol_path),
        source_sha256=protocol['source_sha256'], features_sha256=prepared['features_sha256'], frame_sha256=prepared['frame_sha256'],
        prepared_receipt_sha256=sha(PROJECT_ROOT / protocol['prepared_receipt_path']),
        predictions_sha256=sha(paths['output'] / 'predictions.parquet'), models_sha256=model_hashes,
        structural=structural, fits=dict(neural=len(BRANCHES) * len(seeds), calibration=2, base=0),
        epochs=epochs, seeds=list(seeds), train_rows=len(train), cal_rows=len(cal),
        assessment_rows=int(output.partition.eq('assessment').sum()), calibrations=calibrations,
        train_losses=histories, parameter_count=sum(p.numel() for p in restored_models['meta'][0].parameters()),
        generated_at_utc=datetime.now(timezone.utc).isoformat(), genuine_future_forecasts=0,
        live_acquisition=False, deployment=False, protected_inputs_unchanged=True)
    _json(paths['report'] / 'execution.json', execution)
    print(json.dumps(execution), flush=True)
    return execution


def _ensemble(models, calibration, tensors):
    import torch
    base = tensors['base_probability'].numpy()
    if calibration['amplitude'] == 0:
        return base.copy()
    with torch.no_grad():
        logits = np.stack([model(tensors).numpy() for model in models])
        supported = tensors['player_valid'].any(dim=(1, 2)).numpy()
    delta = np.where(supported, combine_seed_logits(base, logits) - logit(base), 0.)
    return _probabilities(base, delta, calibration['amplitude'])


def _verify_saved_inference(models, calibrations, tensors, indices, expected):
    import torch
    sample = indices[:min(24, len(indices))]
    original = {k: v[sample].clone() for k, v in tensors.items()}
    reverse = {k: v.clone() for k, v in original.items()}
    for name in ('profile', 'player_stats', 'player_roles', 'player_champions', 'player_valid'):
        reverse[name] = reverse[name].flip(1)
    reverse['base_probability'] = 1 - reverse['base_probability']
    cold = {k: v.clone() for k, v in original.items()}
    cold['player_valid'].zero_()
    errors, replay_errors, cold_errors = [], [], []
    for branch in BRANCHES:
        p = _ensemble(models[branch], calibrations[branch], original)
        positions = np.searchsorted(indices, sample) if np.all(indices[:-1] <= indices[1:]) else np.array([np.flatnonzero(indices == i)[0] for i in sample])
        replay_errors.append(float(np.max(np.abs(p - expected[branch][positions]))))
        errors.append(float(np.max(np.abs(p + _ensemble(models[branch], calibrations[branch], reverse) - 1))))
        cold_errors.append(float(np.max(np.abs(_ensemble(models[branch], calibrations[branch], cold) - original['base_probability'].numpy()))))
    for key in original:
        np.testing.assert_array_equal(original[key].numpy(), tensors[key][sample].numpy())
    if max(errors + replay_errors) > 1e-7 or max(cold_errors) > 1e-15:
        raise ValueError('saved correction structural verification failed')
    return dict(passed=True, cases=len(sample), symmetry_max_error=max(errors),
                saved_replay_max_error=max(replay_errors), cold_start_max_error=max(cold_errors),
                input_nonmutation=True, no_second_best_of_projection=True)


def predict_saved(bundle_path, feature_path, indices):
    """Replay a saved correction only on its bound, origin-eligible feature pack."""
    import torch
    from src.models.c0_meta import C0MetaCorrection
    path = Path(bundle_path)
    bundle = json.loads(path.read_text())
    if bundle['origin_day'] != '2026-01-01':
        raise ValueError('saved correction has an unsupported fitting origin')
    metadata_path = Path(feature_path).with_name('metadata.parquet')
    if sha(feature_path) != bundle['features_sha256'] or sha(metadata_path) != bundle['frame_sha256']:
        raise ValueError('saved inference feature pack changed')
    metadata = pd.read_parquet(metadata_path, columns=['date'])
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or not len(selected) or (selected < 0).any() or (selected >= len(metadata)).any():
        raise ValueError('saved inference indices must identify retained feature rows')
    if not (metadata.iloc[selected].date.astype(str) >= bundle['origin_day']).all():
        raise ValueError('saved calibrated correction is ineligible before its origin')
    for key, expected in bundle['inference_source_sha256'].items():
        if sha(_resolve(key, None)) != expected:
            raise ValueError('saved correction forward source changed')
    for relative, expected in bundle['model_sha256'].items():
        if sha(PROJECT_ROOT / relative) != expected:
            raise ValueError('saved correction artifact changed')
    verified_paths = {(PROJECT_ROOT / relative).resolve() for relative in bundle['model_sha256']}
    with np.load(feature_path, allow_pickle=False) as bank:
        arrays = {key: bank[key][indices] for key in bank.files}
    normalizer = MetaNormalizer.from_dict(bundle['normalizer'])
    tensors = _tensors(normalizer.transform(arrays))
    result = {}
    for branch in BRANCHES:
        models = []
        for seed in bundle['seeds']:
            checkpoint_path = (path.parent / branch / f'seed_{seed}.pt').resolve()
            if checkpoint_path not in verified_paths:
                raise ValueError('saved checkpoint path is outside the verified artifact inventory')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if checkpoint['protocol_sha256'] != bundle['protocol_sha256']:
                raise ValueError('saved checkpoint belongs to another protocol')
            model = C0MetaCorrection(checkpoint['vocabulary_size'], checkpoint['use_meta'])
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()
            models.append(model)
        result[branch] = _ensemble(models, bundle['calibrations'][branch], tensors)
        if branch == 'meta':
            with torch.no_grad():
                result['role_meta_embeddings'] = np.stack([model.export_role_meta(tensors).numpy() for model in models])
    result['c0'] = arrays['base_probability']
    return result
