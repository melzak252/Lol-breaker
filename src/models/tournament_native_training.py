"""Exclusive preparation of the single authorized 2023 native W32/E12 control.

No archive or hash-pinned training module is modified. This module is invoked in
its own process because native pickle classes own the archive src.models namespace.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import zipfile
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source_path(key, root):
    return root / key.removeprefix('research:') if key.startswith('research:') else PROJECT_ROOT / key.removeprefix('project:')


def _verify_original_library(root):
    from betting_app.core.models.native_c0 import _read_pinned_json, _verify_hash
    completion_path = PROJECT_ROOT / 'data/08_reporting/model_reliability/model_reliability_20261001_055321/neural/2024/completed.json'
    completion, _ = _read_pinned_json(completion_path, '446ad1bc23c12e39e1f43c6d80a411c6d5231c7f76b60ec9bbec2399cedc9e38', 'original annual control library')
    sources = completion['contract']['source_sha256']
    for key, digest in sources.items():
        _verify_hash(_source_path(key, root), digest, f'original native training primitive {key}')
    return sources


def _verify_control_snapshot(model_root, contract):
    """Enforce the original pre-fit pins, independently of completion inventory."""
    from betting_app.core.models.native_c0 import _read_pinned_json, _verify_hash

    model_root = Path(model_root)
    protocol, _ = _read_pinned_json(model_root / 'protocol.json', contract['protocol_sha256'], '2023 pre-fit protocol')
    if protocol.get('schema') != 'tournament_native_2023_control_protocol_v1' or protocol.get('status') != 'FROZEN_BEFORE_FITTING':
        raise ValueError('invalid 2023 pre-fit protocol')
    for key in ('run_id', 'year', 'recipe', 'source_sha256', 'input_sha256', 'split_ids_sha256'):
        if protocol.get(key) != contract[key]:
            raise ValueError(f'2023 pre-fit protocol mismatch:{key}')
    relative = model_root.resolve().relative_to(PROJECT_ROOT)
    if not relative.is_relative_to(Path('data/06_models/tournament_native')):
        raise ValueError('2023 native control must use its exclusive tournament_native namespace')
    report = PROJECT_ROOT / str(relative).replace('data/06_models/', 'data/08_reporting/', 1)
    membership_path = report / 'partitions.parquet'
    if Path(protocol['membership_path']).resolve() != membership_path.resolve():
        raise ValueError('2023 pre-fit protocol membership path mismatch')
    _verify_hash(model_root / 'sources.zip', protocol['source_snapshot_sha256'], '2023 frozen source snapshot')
    _verify_hash(membership_path, protocol['membership_file_sha256'], '2023 frozen partition membership')


def _write_control_completion(paths, contract):
    from betting_app.core.models.native_c0 import _read_json
    from src.models.a0_native_refit import _sha256, _write_json_exclusive

    execution = _read_json(paths['models'] / 'execution.json', '2023 native execution')
    if execution['contract'] != contract:
        raise ValueError('2023 completion execution contract mismatch')
    _verify_control_snapshot(paths['models'], contract)
    completion = {'status': 'TRAINING_COMPLETE_NOT_SCORED', 'contract': contract,
                  'artifacts': {str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                                for directory in paths.values() for path in sorted(directory.rglob('*')) if path.is_file()}}
    _write_json_exclusive(paths['report'] / 'completed.json', completion)
    return completion


def load_control(model_root, root):
    from betting_app.core.models.native_c0 import _read_json, _verify_hash
    execution = _read_json(Path(model_root) / 'execution.json', '2023 native execution')
    contract = execution['contract']
    if contract['year'] != 2023 or contract['recipe'] != {'variant': 'control', 'seeds': [20260910, 20260911, 20260912], 'epochs': 12}:
        raise ValueError('2023 checkpoint is not the authorized native control recipe')
    _verify_control_snapshot(model_root, contract)
    for key, digest in contract['source_sha256'].items():
        _verify_hash(_source_path(key, root), digest, f'frozen 2023 control source {key}')
    from src.models import a0_data_focus as native
    backend, _ = native.uncertainty._activate_archive_backend(root)
    shared = native._load_checkpoint(Path(model_root) / 'shared/model.joblib', {**contract, 'component': 'shared'})
    models = [native._load_checkpoint(Path(model_root) / f'control/seed_{seed}/model.joblib',
                                     {**contract, 'component': 'neural', 'variant': 'control', 'seed': seed}) for seed in native.SEEDS]
    calibration = native._load_checkpoint(Path(model_root) / 'control/calibration/model.joblib',
                                         {**contract, 'component': 'calibration', 'variant': 'control'})
    return backend, shared, models, calibration


def prepare_2023_control(run_id, research_root=None, smoke=False):
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('run_id must be one nonempty path component')
    from src.analysis.ev_asof_replay import _resolve_research_root
    root = _resolve_research_root(research_root)
    sources = _verify_original_library(root)
    from src.models import a0_data_focus as native
    from src.models.a0_native_refit import (_fit_baseline_normalizer, _atomic_checkpoint, _contract_digest,
                                          _sha256, _versions, _write_json_exclusive, _torch_state_hash, _calculate_forecast)
    from src.models.a0_target_objectives import fit_beta_target_on_device
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    backend, hashes = native.uncertainty._activate_archive_backend(root)
    data = native.uncertainty._load_training_data(root)
    inputs = native._relative_input_hashes(data['input_hashes'], root)
    for key, digest in sources.items():
        if _sha256(_source_path(key, root)) != digest:
            raise ValueError('original native training library changed before freeze')
    for key in ('src/models/tournament_native_training.py', 'src/models/tournament_native.py'):
        sources[key] = _sha256(PROJECT_ROOT / key)
    frame = data['frame']
    dates, released = pd.to_datetime(frame.date), pd.to_datetime(frame.result_day)
    if dates.isna().any() or released.isna().any() or (released < dates).any() or frame.golgg_match_id.duplicated().any():
        raise ValueError('invalid native input identities or label-release chronology')
    split = {'train': np.flatnonzero(((dates < '2022-07-01') & (released < '2022-07-01')).to_numpy()),
             'cal': np.flatnonzero(((dates >= '2022-07-01') & (dates < '2023-01-01') & (released < '2023-01-01')).to_numpy()),
             'validation': np.flatnonzero(((dates >= '2023-01-01') & (dates < '2024-01-01')).to_numpy())}
    if any(len(split[key]) == 0 for key in ('train', 'cal')):
        raise ValueError('2023 native control requires nonempty closed TRAIN/CAL')
    paths = {name: PROJECT_ROOT / f'data/{stage}/tournament_native/{run_id}/2023'
             for name, stage in (('models', '06_models'), ('output', '07_model_output'), ('report', '08_reporting'))}
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f'exclusive native namespace already exists:{path}')
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    memberships = {key: _contract_digest(frame.iloc[ix].golgg_match_id.tolist()) for key, ix in split.items()}
    source_zip = paths['models'] / 'sources.zip'
    with zipfile.ZipFile(source_zip, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for key, digest in sources.items():
            path = _source_path(key, root)
            if _sha256(path) != digest:
                raise ValueError('native source changed during pre-fit snapshot')
            archive.write(path, key.replace(':', '/', 1))
    members = []
    for key, ix in split.items():
        rows = frame.iloc[ix][['golgg_match_id', 'date', 'result_day']].copy()
        rows['partition'] = key
        members.append(rows)
    pd.concat(members, ignore_index=True).to_parquet(paths['report'] / 'partitions.parquet', index=False)
    recipe = {'variant': 'control', 'seeds': list(native.SEEDS), 'epochs': 12}
    protocol = {'schema': 'tournament_native_2023_control_protocol_v1', 'status': 'FROZEN_BEFORE_FITTING',
                'run_id': run_id, 'year': 2023, 'source_sha256': sources, 'source_snapshot_sha256': _sha256(source_zip),
                'input_sha256': inputs, 'bank_receipt_sha256': data['bank_receipt_sha256'], 'recipe': recipe,
                'train_end_exclusive': '2022-07-01', 'cal_end_exclusive': '2023-01-01',
                'split_ids_sha256': memberships, 'membership_path': str(paths['report'] / 'partitions.parquet'),
                'membership_file_sha256': _sha256(paths['report'] / 'partitions.parquet'),
                'training': {'shared_rating': 'full TRAIN', 'shared_mlp': 'three seeds warm24/full12',
                             'objective': 'native terminal beta NLL plus0.1 masked auxiliary SmoothL1',
                             'batch_size': 128, 'threads': 2, 'shuffle': False, 'smoke_does_not_reduce_recipe': True},
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    protocol_path = paths['models'] / 'protocol.json'
    _write_json_exclusive(protocol_path, protocol)
    contract = {'schema': 'tournament_native_2023_control_origin_v1', 'run_id': run_id, 'year': 2023,
                'scope': 'EXPOSED_HISTORICAL_DEVELOPMENT', 'recipe': recipe, 'protocol_sha256': _sha256(protocol_path),
                'source_sha256': sources, 'input_sha256': inputs, 'versions': _versions(), 'split_ids_sha256': memberships}
    _write_json_exclusive(paths['models'] / 'execution.json', {'contract': contract})
    train, cal = split['train'], split['cal']
    labels, best_of = frame.y.to_numpy(float), frame.best_of.to_numpy(int)
    bo_index = (best_of - 1) // 2
    baseline, normalizer = _fit_baseline_normalizer(backend, data, train)
    rating, rating_details = native.uncertainty._fit_rating_model(backend, data['raw'], train, labels, bo_index, native.SEEDS[0])
    shared = {'baseline': baseline, 'normalizer': normalizer, 'rating': rating, 'mlps': []}
    shared_details = {'rating': rating_details, 'mlps': []}
    for seed in native.SEEDS:
        warm, warm_details = native.uncertainty._fit_training_component(backend, data['raw'], data['history'], train, labels, bo_index,
                                                                     seed, kind='mlp', full=False, epochs=24)
        full, full_details = native.uncertainty._fit_training_component(backend, data['raw'], data['history'], train, labels, bo_index,
                                                                     seed, kind='mlp', full=True, initial=warm, epochs=12)
        shared['mlps'].append(full)
        shared_details['mlps'].append({'seed': seed, 'warm': warm_details, 'full': full_details})
    _atomic_checkpoint(paths['models'] / 'shared/model.joblib', shared, {**contract, 'component': 'shared'}, shared_details)
    models = []
    for seed in native.SEEDS:
        model = backend['nonlinear_history'].NonlinearHistoryModel(baseline, normalizer, seed=seed, epochs=12)
        model.main_objective = 'terminal'
        initial = _torch_state_hash(model)
        started = time.monotonic()
        fit_beta_target_on_device(model, data['raw'], data['history'], None, train, bo_index, data['wins_a'], data['wins_b'],
                                  data['targets'], data['target_mask'], device='cpu', objective='terminal',
                                  progress=lambda epoch, main, aux, diagnostics: print(f'2023 control epoch={epoch} main={main:.6f}', file=sys.stderr, flush=True))
        _atomic_checkpoint(paths['models'] / f'control/seed_{seed}/model.joblib', model,
                           {**contract, 'component': 'neural', 'variant': 'control', 'seed': seed},
                           {'initial_state_sha256': initial, 'final_state_sha256': _torch_state_hash(model), 'epochs': 12,
                            'optimizer_updates': 12 * math.ceil(len(train) / 128), 'runtime_seconds': time.monotonic() - started})
        models.append(model)
    raw_cal, hist_cal = native._subset_inputs(data, cal)
    neural_z, rho, raw_p = _calculate_forecast(models, data, cal, backend)
    other_z = native._other_logits(shared, raw_cal, hist_cal, best_of[cal])
    calibration = native._calibrate(backend, neural_z, other_z, raw_cal['gates'], frame.iloc[cal], data['wins_a'][cal], data['wins_b'][cal])
    _atomic_checkpoint(paths['models'] / 'control/calibration/model.joblib', calibration,
                       {**contract, 'component': 'calibration', 'variant': 'control'}, {'beta': calibration['beta_records'], 'cal_rows': len(cal)})
    np.savez_compressed(paths['output'] / 'control_cal_logits.npz', indices=cal, neural_logits=neural_z, other_logits=other_z, rho=rho, raw_seed_probability=raw_p)
    for key, digest in sources.items():
        if _sha256(_source_path(key, root)) != digest:
            raise ValueError('native source changed during preparation')
    for key, digest in inputs.items():
        if _sha256(root / key) != digest:
            raise ValueError('native input changed during preparation')
    completion = _write_control_completion(paths, contract)
    completion_path = paths['report'] / 'completed.json'
    artifact_contract = {'model_kind': 'c0', 'model_year': 2023,
                         'train_end': str(frame.iloc[train].result_day.max())[:10] + 'T00:00:00Z',
                         'calibration_end': str(frame.iloc[cal].result_day.max())[:10] + 'T00:00:00Z',
                         'artifact_paths': {key: str(PROJECT_ROOT / key) for key in completion['artifacts']},
                         'artifact_sha256': completion['artifacts'], 'provenance': contract}
    return {'status': completion['status'], 'model_root': str(paths['models']), 'output_root': str(paths['output']), 'report_root': str(paths['report']),
            'protocol_path': str(protocol_path), 'protocol_sha256': _sha256(protocol_path), 'completion_path': str(completion_path),
            'completion_sha256': _sha256(completion_path), 'model_year': 2023, 'recipe': recipe, 'smoke_requested': bool(smoke),
            'artifact_contract': artifact_contract}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--research-root', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Retains the exact three-seed E12 recipe; does not produce a reduced control')
    args = parser.parse_args()
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        receipt = prepare_2023_control(args.run_id, args.research_root, args.smoke)
    print(json.dumps(receipt, allow_nan=False))
