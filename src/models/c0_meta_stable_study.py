"""Bounded 180-day player / unchanged 30-day meta follow-up; no live inputs."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pandas as pd
from scipy.special import logit

from src.models import c0_meta_study as old

PROJECT_ROOT = old.PROJECT_ROOT
PRIOR_RUN = 'c0_meta_20261007_094552_integrity_v2'
SCHEMA = 'c0_role_meta_v1'
PLAYER_DAYS = 180
PRIOR_SEAL_SHA256 = '69615ffa220f4e54810b47a57b606777d389e793e8bcc09377220da6a624f1c6'
EXTRA_SOURCES = ('src/features/c0_meta_stable_features.py', 'src/models/c0_meta_stable.py',
    'src/models/c0_meta_stable_study.py', 'scripts/train_c0_meta_stable.py',
    'src/analysis/metrics.py', 'src/analysis/thesis_metrics.py', 'src/models/competition_tiers.py',
    'betting_app/core/ev.py', 'src/analysis/devig_comparison.py')


def paths(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be one path component')
    return {name: PROJECT_ROOT / 'data' / stage / 'c0_meta_stable' / run_id for name, stage in (
        ('features', '04_feature'), ('input', '05_model_input'), ('models', '06_models'),
        ('output', '07_model_output'), ('report', '08_reporting'))}


class StableNormalizer:
    """Keep original meta scaling; fit long-player/activity scaling on TRAIN only."""
    def __init__(self, meta, mean, scale, count_mean, count_scale, activity_mean, activity_scale):
        self.meta = meta
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.count_mean, self.count_scale = count_mean, count_scale
        self.activity_mean, self.activity_scale = np.asarray(activity_mean), np.asarray(activity_scale)

    @classmethod
    def fit(cls, arrays, train, meta):
        train = np.asarray(train, dtype=int)
        if not len(train):
            raise ValueError('normalizer needs closed TRAIN support')
        stats = arrays['player_stats'][train]
        mean, scale = [], []
        for k in range(12):
            values = stats[..., k][stats[..., 24+k] > 0].astype(float)
            mean.append(float(values.mean()) if len(values) else 0.)
            scale.append(max(float(values.std()), 1e-6) if len(values) else 1.)
        counts = stats[..., 36].astype(float)
        activity = arrays['player_activity'][train]
        am, sd = [], []
        for k in range(3):
            mask = activity[..., 3] > 0 if k == 2 else arrays['player_valid'][train]
            values = activity[..., k][mask].astype(float)
            am.append(float(values.mean()) if len(values) else 0.)
            sd.append(max(float(values.std()), 1e-6) if len(values) else 1.)
        return cls(meta, mean, scale, float(counts.mean()), max(float(counts.std()), 1e-6), am, sd)

    def to_dict(self):
        return dict(meta=self.meta.to_dict(), mean=self.mean.tolist(), scale=self.scale.tolist(),
                    count_mean=self.count_mean, count_scale=self.count_scale,
                    activity_mean=self.activity_mean.tolist(), activity_scale=self.activity_scale.tolist())

    @classmethod
    def from_dict(cls, value):
        return cls(old.MetaNormalizer.from_dict(value['meta']), **{k: value[k] for k in (
            'mean', 'scale', 'count_mean', 'count_scale', 'activity_mean', 'activity_scale')})

    def transform(self, arrays):
        result = self.meta.transform(arrays)
        stats = arrays['player_stats'].copy()
        present = stats[..., 24:36] > 0
        stats[..., :12] = np.where(present, (stats[..., :12]-self.mean)/self.scale, 0.)
        stats[..., 12:24] = np.where(present, stats[..., 12:24]/self.scale, 0.)
        stats[..., 36] = (stats[..., 36]-self.count_mean)/self.count_scale
        result['player_stats'] = np.where(arrays['player_valid'][..., None], stats, 0.).astype(np.float32)
        activity = arrays['player_activity'].copy()
        activity[..., :3] = (activity[..., :3]-self.activity_mean)/self.activity_scale
        activity[..., 2] = np.where(activity[..., 3] > 0, activity[..., 2], 0.)
        result['player_activity'] = activity.astype(np.float32)
        return result


def verify_protocol(protocol_path, root):
    p = json.loads(Path(protocol_path).read_text())
    if p['schema'] != SCHEMA or p.get('feature_contract') != 'stable_player180_v1' or p['features']['player_summary_window_days'] != PLAYER_DAYS or p['features']['window_days'] != 30:
        raise ValueError('unsupported stable-profile recipe')
    expected_training = dict(seeds=list(old.SEEDS), branches=list(old.BRANCHES), epochs=12,
        batch_size=128, threads=2, lr=.001, weight_decay=.01, shuffle_closed_train=True,
        objective='binary_series_log_loss', base_fits=0, neural_fits=6,
        scalar_calibration_fits=2, max_logit_correction=.5)
    if p['training'] != expected_training:
        raise ValueError('stable-profile fixed training budget changed')
    for inventory in ('source_sha256', 'input_sha256'):
        for key, expected in p[inventory].items():
            if old.sha(old._resolve(key, root)) != expected:
                raise ValueError(f'frozen {inventory} changed: {key}')
    if old.sha(PROJECT_ROOT / p['source_snapshot_path']) != p['source_snapshot_sha256']:
        raise ValueError('stable source snapshot changed')
    return p


def _authenticate_prior_seal(seal_path,expected_sha,project_root,run_id,protocol_sha):
    if old.sha(seal_path) != expected_sha:
        raise ValueError('prior seal changed')
    seal=json.loads(Path(seal_path).read_text())
    if (seal['schema']!='c0_meta_sealed_verification_v1' or seal['run_id']!=run_id
            or seal['status']!='COMPLETED' or seal['decision']!='NOT_ADOPTED'
            or seal['protocol_sha256']!=protocol_sha):
        raise ValueError('prior seal does not authenticate the expected completed negative')
    bindings={}
    for relative,expected in seal['artifact_sha256'].items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('invalid prior sealed artifact path')
        # Documentation may advance. Historical research data/proofs may not.
        if relative.startswith('data/'):
            if old.sha(project_root/path)!=expected:
                raise ValueError(f'prior sealed artifact changed: {relative}')
            bindings['project:'+relative]=expected
    return bindings


def freeze_protocol(run_id, research_root=None):
    root, stage = old._root(research_root), paths(run_id)
    if any(p.exists() for p in stage.values()):
        raise FileExistsError('new run requires nonexistent stage directories')
    prior_paths = old._paths(PRIOR_RUN)
    prior_protocol = old._verify_protocol(prior_paths['input'] / 'protocol.json', root)
    prior_execution = json.loads((prior_paths['report'] / 'execution.json').read_text())
    if prior_execution['status'] != 'COMPLETED' or old.sha(prior_paths['input'] / 'protocol.json') != prior_execution['protocol_sha256']:
        raise ValueError('prior study is not a bound completed experiment')
    sealed_bindings=_authenticate_prior_seal(prior_paths['report']/'sealed_verification.json',
        PRIOR_SEAL_SHA256,PROJECT_ROOT,PRIOR_RUN,old.sha(prior_paths['input']/'protocol.json'))
    # Historical source closure is checked against its immutable capsule, not repinned to current CLI.
    with zipfile.ZipFile(PROJECT_ROOT / prior_protocol['source_snapshot_path']) as z:
        import hashlib
        for key, expected in prior_protocol['source_sha256'].items():
            if hashlib.sha256(z.read(key.replace(':', '/'))).hexdigest() != expected:
                raise ValueError('prior historical source capsule changed')
    metadata = pd.read_parquet(root / old.BANK / 'metadata.parquet')
    metadata['golgg_match_id'] = metadata.golgg_match_id.astype(str)
    for name in ('date', 'result_day'):
        metadata[name] = pd.to_datetime(metadata[name]).dt.strftime('%Y-%m-%d')
    origins, inputs = old._audit_origins(root, metadata)
    inputs.update(sealed_bindings)
    for key, expected in prior_protocol['input_sha256'].items():
        if old.sha(old._resolve(key, root)) != expected:
            raise ValueError(f'prior protected input changed: {key}')
        inputs[key] = expected
    prepared = json.loads((prior_paths['features'] / 'prepared.json').read_text())
    if (old.sha(prior_paths['features']/'prepared.json')!=prior_execution['prepared_receipt_sha256']
            or prepared['status']!='PREPARED'
            or prepared['protocol_sha256']!=prior_execution['protocol_sha256']
            or any(prepared[k]!=prior_execution[k] for k in ('features_sha256','frame_sha256'))):
        raise ValueError('prior preparation is not authenticated by its completion receipt')
    bindings = {'features.npz': 'features_sha256', 'metadata.parquet': 'frame_sha256',
                'normalizer.json': 'normalizer_sha256', 'prepared.json': None}
    for name, key in bindings.items():
        path = prior_paths['features'] / name
        digest = old.sha(path)
        if key and digest != prepared[key]:
            raise ValueError('authenticated prior features changed')
        inputs[old._qualified(path, root)] = digest
    for relative, expected in prior_execution['models_sha256'].items():
        path = PROJECT_ROOT / relative
        if old.sha(path) != expected:
            raise ValueError('prior meta model changed')
        inputs[old._qualified(path, root)] = expected
    prior_predictions = PROJECT_ROOT / prior_protocol['predictions_path']
    if old.sha(prior_predictions) != prior_execution['predictions_sha256']:
        raise ValueError('prior meta forecasts changed')
    inputs[old._qualified(prior_predictions, root)] = prior_execution['predictions_sha256']
    for name in ('sealed_verification.json', 'evaluation/metrics.json', 'evaluation/decision.json'):
        path = prior_paths['report'] / name
        inputs[old._qualified(path, root)] = old.sha(path)
    for path in (prior_paths['input'] / 'protocol.json', prior_paths['report'] / 'execution.json',
                 PROJECT_ROOT / prior_protocol['source_snapshot_path']):
        inputs[old._qualified(path, root)] = old.sha(path)
    sources = {key: old.sha(old._resolve(key, root)) for key in prior_protocol['source_sha256']}
    for name in EXTRA_SOURCES:
        sources['project:'+name] = old.sha(PROJECT_ROOT / name)
    for relative in ('nonlinear-history-20260912/code/scripts/roster_optional_features.py',
                     'nonlinear-history-20260912/code/scripts/temporal_player_features.py'):
        sources['research:'+relative]=old.sha(root/relative)
    stage['input'].mkdir(parents=True)
    snapshot = stage['input'] / 'sources.zip'
    with zipfile.ZipFile(snapshot, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for key in sources:
            z.write(old._resolve(key, root), key.replace(':', '/'))
    p = dict(prior_protocol)
    relative = lambda path: path.relative_to(PROJECT_ROOT).as_posix()
    p.update(schema=SCHEMA, run_id=run_id, created_at_utc=datetime.now(timezone.utc).isoformat(),
             origins=origins, source_sha256=sources, input_sha256=inputs,
             source_snapshot_path=relative(snapshot), source_snapshot_sha256=old.sha(snapshot),
             prior_run=PRIOR_RUN, prior_prepared_path=relative(prior_paths['features'] / 'prepared.json'),
             prior_features_path=relative(prior_paths['features'] / 'features.npz'),
             prior_frame_path=relative(prior_paths['features'] / 'metadata.parquet'),
             prior_normalizer_path=relative(prior_paths['features'] / 'normalizer.json'))
    p['feature_contract'] = 'stable_player180_v1'
    for key, path in dict(protocol=stage['input']/'protocol.json', features=stage['features']/'features.npz',
            frame=stage['features']/'metadata.parquet', diagnostics=stage['features']/'provenance.pkl',
            raw_source=stage['features']/'raw_source.pkl', prepared_receipt=stage['features']/'prepared.json',
            models=stage['models'], predictions=stage['output']/'predictions.parquet',
            execution=stage['report']/'execution.json').items():
        p[key+'_path'] = relative(path)
    p['features'] = {**prior_protocol['features'], 'player_summary_window_days': PLAYER_DAYS,
        'player_support': 'long180 OR native_history; raw identity alone is not usable representation',
        'activity': ['log_recent30_count','log_long180_count','log_inactivity_days','raw_history_known',
                     'recent_supported','long_supported','native_supported','identity_known'],
        'native_role_fallback': 'prior last16 known-role frequencies only when long180 absent',
        'meta_normalizer': 'unchanged original TRAIN2024 normalizer; long-player/activity scalers TRAIN-only',
        'parameter_change': 'only eight required player feature inputs: +128 weights, no hidden-width increase'}
    p['preparation'] = dict(base_refits=0, prior_authenticated_h_reused=True,
        original_prior_native_replayed_probabilities=prepared['base_replayed_probabilities'],
        original_prior_native_replay_max_error=prepared['base_replay_max_error'],
        raw_retention_first_cutoff='2013-08-17', vocabulary='original pre2024 fixed164, not parser retention cutoff')
    old._json(stage['input'] / 'protocol.json', p)
    print(json.dumps(dict(protocol=p['protocol_path'], sha256=old.sha(stage['input']/'protocol.json'))), flush=True)
    return stage['input']/'protocol.json'


def prepare_features(protocol_path, research_root=None):
    from src.features.c0_meta_source import read_raw_meta_source, RawMetaSource
    from src.features.c0_meta_stable_features import build_stable_meta_features
    from src.analysis.ev_asof_replay import _canonical_competition_context
    root = old._root(research_root)
    p = verify_protocol(protocol_path, root)
    stage = paths(p['run_id'])
    if stage['features'].exists():
        raise FileExistsError(stage['features'])
    with np.load(PROJECT_ROOT / p['prior_features_path'], allow_pickle=False) as bank:
        prior = {key: bank[key] for key in bank.files}
    frame = pd.read_parquet(PROJECT_ROOT / p['prior_frame_path'])
    metadata = pd.read_parquet(root / old.BANK / 'metadata.parquet')
    ix = metadata.golgg_match_id.astype(str).tolist()
    lookup = {value: k for k, value in enumerate(ix)}
    indices = np.array([lookup[str(value)] for value in frame.golgg_match_id])
    for name in ('roster_a', 'roster_b'):
        if not all(list(a) == list(b) for a,b in zip(frame[name], metadata.iloc[indices][name])):
            raise ValueError('native player axis changed')
    masks = np.asarray(np.load(root / old.BANK / 'mask.npy', mmap_mode='r')[indices], dtype=bool)
    native_valid = masks.any(axis=-1)
    history = np.load(root / old.BANK / 'history.npy', mmap_mode='r')
    native_roles = np.empty((*native_valid.shape, 5), dtype=np.float32)
    for begin in range(0, len(indices), 256):
        stop = min(begin+256, len(indices))
        role_values = history[indices[begin:stop], ..., 27:32]
        counts = masks[begin:stop].sum(axis=-1, keepdims=True)
        native_roles[begin:stop] = np.divide((role_values*masks[begin:stop,...,None]).sum(axis=-2),
            counts, out=np.zeros((*counts.shape[:-1],5),np.float32), where=counts>0)
    prepared_prior = json.loads((PROJECT_ROOT / p['prior_prepared_path']).read_text())
    source = read_raw_meta_source(PROJECT_ROOT / p['raw_path'], root / p['release_path'],
        root / p['reader_path'], first_cutoff=p['preparation']['raw_retention_first_cutoff'])
    audit = {**source.audit, 'vocabulary_rule': 'authenticated original pre2024 vocabulary',
             'vocabulary_size': len(prepared_prior['vocabulary']), 'vocabulary_binding': p['prior_prepared_path']}
    source = RawMetaSource(source.observations, tuple(prepared_prior['vocabulary']), source.series, audit)
    targets = [dict(row_id=r.golgg_match_id, cutoff_day=r.date,
        competition=_canonical_competition_context(r.tournament_name), roster_a=r.roster_a,
        roster_b=r.roster_b, patch=r.target_patch) for r in frame.itertuples()]
    batch = build_stable_meta_features(source.observations, targets, champion_ids=source.champion_ids,
                                      native_valid=native_valid, native_roles=native_roles)
    arrays = {**batch.arrays, 'profile': prior['profile'], 'base_probability': prior['base_probability'],
              'best_of': prior['best_of']}
    for name in ('meta_stats','meta_champions','meta_support','meta_patch'):
        np.testing.assert_array_equal(arrays[name], prior[name])
    frame['prior_recent_support_missingness'] = frame.support_missingness
    frame['prior_both_no_recent'] = frame.both_cold
    frame['recent_support_missingness'] = ~arrays['player_activity'][...,4].all(axis=(1,2))
    frame['long_support_missingness'] = ~arrays['player_activity'][...,5].all(axis=(1,2))
    frame['native_support_missingness'] = ~native_valid.all(axis=(1,2))
    frame['support_missingness'] = ~arrays['player_valid'].all(axis=(1,2))
    frame['both_cold'] = ~arrays['player_valid'].any(axis=(1,2))
    frame['unknown_player_count'] = (arrays['player_activity'][...,7] == 0).sum(axis=(1,2))
    frame['restored_player_count'] = (arrays['player_valid'] & (arrays['player_activity'][...,4] == 0)).sum(axis=(1,2))
    frame['native_only_count'] = (native_valid & (arrays['player_activity'][...,5] == 0)).sum(axis=(1,2))
    frame['inactive_player_count'] = ((arrays['player_activity'][...,3] > 0) &
                                     (arrays['player_activity'][...,2] > np.log1p(30))).sum(axis=(1,2))
    core = np.load(root / old.BANK / 'core.npy', mmap_mode='r')[indices]
    frame['new_roster_count'] = (core[..., 4] == 0).sum(axis=1)
    long_counts = np.rint(np.expm1(arrays['player_activity'][..., 1]))
    minimum = long_counts.min(axis=(1, 2))
    frame['long_min_support_band'] = np.select(
        [minimum == 0, minimum <= 10, minimum <= 50],
        ['0', '1-10', '11-50'], default='51+')
    # All extended evidence maxima, including old activity-only observations, remain strictly causal.
    for i, diagnostic in enumerate(batch.diagnostics):
        values = [frame.at[i,'feature_source_max_day'], diagnostic.get('max_effective_release_day')]
        for side in diagnostic['prior_activity']:
            for item in side:
                values.append(item.get('max_effective_release_day'))
        for sides in diagnostic['long_player_source_indices']:
            for members in sides:
                values.extend(batch.sources[k]['effective_release_day'] for k in members)
        frame.at[i,'feature_source_max_day'] = max((v for v in values if pd.notna(v)), default=None)
    split = old.temporal_partitions(frame)
    scaler = StableNormalizer.fit(arrays, split['train'], old.MetaNormalizer.from_dict(
        json.loads((PROJECT_ROOT / p['prior_normalizer_path']).read_text())))
    normalized = scaler.transform(arrays)
    prior_normalized = scaler.meta.transform(prior)
    for name in ('meta_stats','meta_support','meta_patch'):
        np.testing.assert_array_equal(normalized[name], prior_normalized[name])
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'nonfinite stable features: {name}')
    stage['features'].mkdir(parents=True)
    np.savez_compressed(stage['features']/'features.npz', **arrays)
    frame.to_parquet(stage['features']/'metadata.parquet', index=False)
    for name, value in (('provenance.pkl',dict(diagnostics=batch.diagnostics,sources=batch.sources)),
                        ('raw_source.pkl',source)):
        with (stage['features']/name).open('xb') as stream:
            pickle.dump(value,stream,protocol=5)
    old._json(stage['features']/'normalizer.json',scaler.to_dict())
    receipt = dict(status='PREPARED',protocol_sha256=old.sha(protocol_path),rows=len(frame),
        partition_counts={k:len(v) for k,v in split.items()},source_audit=audit,
        vocabulary=list(source.champion_ids),meta_raw_bitwise_unchanged=True,meta_normalized_bitwise_unchanged=True,
        base_fits=0,authenticated_prior_profile_reused=True,prior_native_replay= p['preparation'],
        support_by_partition={part:{k:int(frame.loc[ix,k].sum()) for k in ('restored_player_count','native_only_count',
            'inactive_player_count','unknown_player_count','both_cold','prior_both_no_recent')} for part,ix in split.items()})
    for name,key in (('features.npz','features_sha256'),('metadata.parquet','frame_sha256'),
                     ('provenance.pkl','diagnostics_sha256'),('raw_source.pkl','raw_source_sha256'),
                     ('normalizer.json','normalizer_sha256')):
        receipt[key]=old.sha(stage['features']/name)
    verify_protocol(protocol_path,root)
    old._json(stage['features']/'prepared.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


def prepared(p,protocol_path):
    stage=paths(p['run_id'])
    receipt=json.loads((stage['features']/'prepared.json').read_text())
    if receipt['status']!='PREPARED' or receipt['protocol_sha256']!=old.sha(protocol_path):
        raise ValueError('prepared stable data is not protocol-bound')
    for name,key in (('features.npz','features_sha256'),('metadata.parquet','frame_sha256'),
        ('normalizer.json','normalizer_sha256'),('provenance.pkl','diagnostics_sha256'),('raw_source.pkl','raw_source_sha256')):
        if old.sha(stage['features']/name)!=receipt[key]:
            raise ValueError(f'prepared artifact changed: {name}')
    with np.load(stage['features']/'features.npz',allow_pickle=False) as bank:
        arrays={key:bank[key] for key in bank.files}
    return arrays,pd.read_parquet(stage['features']/'metadata.parquet'),StableNormalizer.from_dict(
        json.loads((stage['features']/'normalizer.json').read_text())),receipt


def structural(models,calibrations,tensors,indices,expected):
    import torch
    sample=indices[-min(24,len(indices)):]
    original={k:v[sample].clone() for k,v in tensors.items()}
    reverse={k:v.clone() for k,v in original.items()}
    for key in ('profile','player_stats','player_roles','player_champions','player_valid','player_activity'):
        reverse[key]=reverse[key].flip(1)
    reverse['base_probability']=1-reverse['base_probability']
    cold={k:v.clone() for k,v in original.items()}
    cold['player_valid'].zero_();cold['player_activity'][...,3:].zero_()
    errors, replay, cold_errors=[],[],[]
    for branch in old.BRANCHES:
        prob=old._ensemble(models[branch],calibrations[branch],original)
        positions=np.array([np.flatnonzero(indices==i)[0] for i in sample])
        replay.append(float(np.max(np.abs(prob-expected[branch][positions]))))
        errors.append(float(np.max(np.abs(prob+old._ensemble(models[branch],calibrations[branch],reverse)-1))))
        cold_errors.append(float(np.max(np.abs(old._ensemble(models[branch],calibrations[branch],cold)-original['base_probability'].numpy()))))
    for key in original:
        np.testing.assert_array_equal(original[key].numpy(),tensors[key][sample].numpy())
    if max(errors+replay)>1e-7 or max(cold_errors)>1e-15:
        raise ValueError('saved stable structural checks failed')
    return dict(passed=True,cases=len(sample),symmetry_max_error=max(errors),saved_replay_max_error=max(replay),
                cold_start_max_error=max(cold_errors),input_nonmutation=True,no_second_best_of_projection=True)


def run_experiment(protocol_path,research_root=None,smoke=False):
    import torch
    from src.models.c0_meta_stable import StableC0MetaCorrection
    root=old._root(research_root);p=verify_protocol(protocol_path,root)
    raw,frame,normalizer,receipt=prepared(p,protocol_path)
    stage=paths(p['run_id']+'_smoke' if smoke else p['run_id'])
    if any(stage[k].exists() for k in ('models','output','report')):
        raise FileExistsError('execution artifacts exist; no implicit overwrite or retry')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    arrays=normalizer.transform(raw);tensors=old._tensors(arrays);split=old.temporal_partitions(frame)
    train,cal=split['train'],split['cal']
    if smoke:train,cal=train[:128],cal[:64]
    indices=np.arange(len(frame)) if not smoke else np.concatenate((train,cal,split['assessment'][:16]))
    seeds=old.SEEDS[:1] if smoke else old.SEEDS;epochs=1 if smoke else 12
    y=torch.tensor(frame.y.to_numpy(),dtype=torch.float64);base=arrays['base_probability'][indices]
    models,calibrations,corrected,hashes,histories={},{},{},{},{}
    stage['models'].mkdir(parents=True)
    for branch in old.BRANCHES:
        models[branch]=[];logits=[]
        for seed in seeds:
            torch.manual_seed(seed)
            model=StableC0MetaCorrection(len(receipt['vocabulary'])+1,branch=='meta')
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            rng=np.random.default_rng(seed);losses=[]
            for epoch in range(epochs):
                model.train();order=rng.permutation(train);total=0.
                for begin in range(0,len(order),128):
                    ix=order[begin:begin+128]
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model({k:v[ix] for k,v in tensors.items()}),y[ix])
                    if not torch.isfinite(loss):raise ValueError('nonfinite stable TRAIN loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                    total+=float(loss.detach())*len(ix)
                losses.append(total/len(train))
                print(json.dumps(dict(stage='smoke' if smoke else 'fit',branch=branch,seed=seed,epoch=epoch+1,train_loss=losses[-1])),flush=True)
            path=stage['models']/branch/f'seed_{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                vocabulary_size=model.vocabulary_size,use_meta=model.use_meta,seed=seed,epochs=epochs,
                protocol_sha256=old.sha(protocol_path),normalizer=normalizer.to_dict(),train_losses=losses),path)
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            restored=StableC0MetaCorrection(checkpoint['vocabulary_size'],checkpoint['use_meta'])
            restored.load_state_dict(checkpoint['state_dict'],strict=True);restored.eval()
            replay=old._seed_logits(restored,tensors,indices)
            np.testing.assert_array_equal(replay,old._seed_logits(model,tensors,indices))
            models[branch].append(restored);logits.append(replay)
            hashes[path.relative_to(PROJECT_ROOT).as_posix()]=old.sha(path);histories[f'{branch}:{seed}']=losses
        delta=old.combine_seed_logits(base,np.asarray(logits))-logit(base)
        delta=np.where(arrays['player_valid'][indices].any(axis=(1,2)),delta,0.)
        positions=np.flatnonzero(np.isin(indices,cal))
        calibrations[branch]=old.fit_amplitude(base[positions],delta[positions],frame.y.to_numpy()[indices[positions]])
        corrected[branch]=old._probabilities(base,delta,calibrations[branch]['amplitude'])
    proof=structural(models,calibrations,tensors,indices,corrected)
    output=frame.iloc[indices].copy().reset_index(drop=True);assessment=output.partition.eq('assessment').to_numpy()
    for branch in old.BRANCHES:
        output[f'p_{branch}_development']=corrected[branch]
        output[f'p_{branch}']=np.where(assessment,corrected[branch],output.p_c0)
    te=str(frame.iloc[train].effective_label_release_day.max());ce=str(frame.iloc[cal].effective_label_release_day.max())
    output['train_end']=np.where(assessment & (output.base_train_end.to_numpy(str)<te),te,output.base_train_end)
    output['calibration_end']=np.where(assessment & (output.base_calibration_end.to_numpy(str)<ce),ce,output.base_calibration_end)
    output['correction_train_end']=np.where(assessment,te,None);output['correction_calibration_end']=np.where(assessment,ce,None)
    output['forecast_kind']=np.where(assessment,'stable_correction_assessment','unchanged_c0_warmup')
    stage['output'].mkdir(parents=True);output.to_parquet(stage['output']/'predictions.parquet',index=False)
    calibration_path=stage['models']/'calibration.json';old._json(calibration_path,calibrations)
    hashes[calibration_path.relative_to(PROJECT_ROOT).as_posix()]=old.sha(calibration_path)
    sources={key:expected for key,expected in p['source_sha256'].items() if key.endswith('.py')}
    bundle_path=stage['models']/'bundle.json'
    old._json(bundle_path,dict(schema='c0_meta_stable_bundle_v1',protocol_sha256=old.sha(protocol_path),run_id=p['run_id'],
        smoke=smoke,seeds=list(seeds),vocabulary=receipt['vocabulary'],origin_day='2026-01-01',
        features_sha256=receipt['features_sha256'],frame_sha256=receipt['frame_sha256'],
        inference_source_sha256={k:v for k,v in sources.items() if k.startswith('project:') and k in (
            'project:src/models/c0_meta_stable.py','project:src/models/c0_meta_stable_study.py',
            'project:src/models/c0_meta.py','project:src/models/c0_meta_study.py')},
        normalizer=normalizer.to_dict(),calibrations=calibrations,model_sha256=hashes))
    hashes[bundle_path.relative_to(PROJECT_ROOT).as_posix()]=old.sha(bundle_path)
    verify_protocol(protocol_path,root)
    execution=dict(schema='c0_meta_execution_v1',feature_contract='stable_player180_v1',status='SMOKE_PASSED' if smoke else 'COMPLETED',
        run_id=p['run_id'],scope='EXPOSED_HISTORICAL_DEVELOPMENT',protocol_sha256=old.sha(protocol_path),
        source_sha256=p['source_sha256'],features_sha256=receipt['features_sha256'],frame_sha256=receipt['frame_sha256'],
        prepared_receipt_sha256=old.sha(PROJECT_ROOT/p['prepared_receipt_path']),predictions_sha256=old.sha(stage['output']/'predictions.parquet'),
        models_sha256=hashes,structural=proof,fits=dict(neural=2*len(seeds),calibration=2,base=0),epochs=epochs,
        seeds=list(seeds),train_rows=len(train),cal_rows=len(cal),assessment_rows=int(assessment.sum()),
        calibrations=calibrations,train_losses=histories,parameter_count=sum(q.numel() for q in models['meta'][0].parameters()),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),genuine_future_forecasts=0,
        live_acquisition=False,deployment=False,protected_inputs_unchanged=True)
    old._json(stage['report']/'execution.json',execution);print(json.dumps(execution),flush=True)
    return execution


def predict_saved(bundle_path,feature_path,indices):
    import torch
    from src.models.c0_meta_stable import StableC0MetaCorrection
    path=Path(bundle_path);bundle=json.loads(path.read_text());selected=np.asarray(indices,dtype=int)
    if bundle['schema']!='c0_meta_stable_bundle_v1' or bundle['origin_day']!='2026-01-01':
        raise ValueError('unsupported stable saved origin')
    metadata=Path(feature_path).with_name('metadata.parquet')
    if old.sha(feature_path)!=bundle['features_sha256'] or old.sha(metadata)!=bundle['frame_sha256']:
        raise ValueError('saved stable feature pack changed')
    dates=pd.read_parquet(metadata,columns=['date'])
    if selected.ndim!=1 or not len(selected) or (selected<0).any() or (selected>=len(dates)).any():
        raise ValueError('invalid saved indices')
    if not (dates.iloc[selected].date.astype(str)>=bundle['origin_day']).all():
        raise ValueError('saved stable correction ineligible before origin')
    for key,digest in bundle['inference_source_sha256'].items():
        if old.sha(old._resolve(key,None))!=digest:raise ValueError('saved stable source changed')
    verified={}
    for key,digest in bundle['model_sha256'].items():
        actual=(PROJECT_ROOT/key).resolve()
        if old.sha(actual)!=digest:raise ValueError('saved stable model changed')
        verified[actual]=digest
    with np.load(feature_path,allow_pickle=False) as bank:
        arrays={k:bank[k][selected] for k in bank.files}
    tensors=old._tensors(StableNormalizer.from_dict(bundle['normalizer']).transform(arrays));result={}
    for branch in old.BRANCHES:
        models=[]
        for seed in bundle['seeds']:
            model_path=(path.parent/branch/f'seed_{seed}.pt').resolve()
            if model_path not in verified:raise ValueError('saved stable checkpoint outside verified inventory')
            checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
            if checkpoint['protocol_sha256']!=bundle['protocol_sha256']:raise ValueError('checkpoint protocol changed')
            model=StableC0MetaCorrection(checkpoint['vocabulary_size'],checkpoint['use_meta'])
            model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();models.append(model)
        result[branch]=old._ensemble(models,bundle['calibrations'][branch],tensors)
        if branch=='meta':
            result['role_meta_embeddings']=np.stack([m.export_role_meta(tensors).numpy() for m in models])
    result['c0']=arrays['base_probability']
    return result


def evaluate_support_profiles(protocol_path,benchmark_dir,output_path,research_root=None):
    """Supplement the existing benchmark, never refit/redefine its main decision."""
    from src.analysis.c0_meta_evaluation import _monthly_contrasts, _metrics
    from src.analysis.research_benchmark import safe
    p=verify_protocol(protocol_path,old._root(research_root));stage=paths(p['run_id'])
    metrics_path=Path(benchmark_dir)/'metrics.json'
    benchmark=json.loads(metrics_path.read_text())
    execution_path=stage['report']/'execution.json'
    execution=json.loads(execution_path.read_text())
    expected=dict(run_id=p['run_id'],protocol_sha256=old.sha(protocol_path),
        execution_sha256=old.sha(execution_path),
        predictions_sha256=old.sha(stage['output']/'predictions.parquet'),
        complete_native_coverage_verified=True,source_bindings_verified=True)
    if benchmark['receipt'] != expected or execution['status'] != 'COMPLETED':
        raise ValueError('support profiles require a bound completed central benchmark')
    prepared(p,protocol_path)
    frame=pd.read_parquet(stage['output']/'predictions.parquet')
    primary=frame.loc[frame.partition.eq('assessment')]
    slices={}
    for key in ('restored_player_count','native_only_count','inactive_player_count','unknown_player_count',
                'recent_support_missingness','long_support_missingness','native_support_missingness',
                'support_missingness','new_roster_count','long_min_support_band','new_org','local_missingness'):
        slices[key]={}
        for label,group in primary.groupby(key,dropna=False):
            slices[key][str(label)]=dict(rows=len(group),metrics=_metrics(group),
                contrasts=_monthly_contrasts(group,5000,20261007),descriptive_only=True)
    result=safe(dict(scope='EXPOSED_HISTORICAL_SUPPORT_DIAGNOSTICS_ONLY',
        benchmark_sha256=old.sha(metrics_path),receipt=expected,stable_support_profiles=slices,
        new_roster_definition='zero native prior roster games; not organization novelty',
        sparse_definition='minimum180day maps among ten roster slots: 0/1-10/11-50/51+',
        no_additional_adoption_gate=True))
    old._json(output_path,result)
    print(json.dumps(result),flush=True)
    return result
