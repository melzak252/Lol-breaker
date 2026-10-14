"""Isolated P1 historical study orchestration; never modifies inherited artifacts."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import json
import re
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import lognorm
from src.models.duration_scale_reliability import (COLUMNS, ScalePreprocessor, DurationScaleModel,
    sha256, load_selected_tree, load_duration_scale, train_reference_sigma,
    train_scale_objective, cal_scale_objective, accept_fit_result)
from src.models.prop_distribution import distribution_nll
from src.analysis.prop_distribution_evaluation import lognormal_mixture_crps
from src.analysis.sports_reliability_common import paired_block_contrast, contrast_draw_arrays
from src.models.competition_tiers import classify_competition

SCENARIO = 'historical_actual_first_map_roster_daily_release'
MONTHS = [f'2026-{m:02d}' for m in range(1, 10)]
STAGES = ['04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting']

def _now(): return datetime.now(timezone.utc).isoformat()
def _json(path):
    with Path(path).open() as stream: return json.load(stream)
def _save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False)
def _ids(ids): return hashlib.sha256('\n'.join(map(str, ids)).encode()).hexdigest()
def _roots(run_id, repo_root):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id) or run_id in ('.', '..'): raise ValueError('unsafe run_id')
    return {s: Path(repo_root) / 'data' / s / 'duration_scale_reliability' / run_id for s in STAGES}
def _e(run_id, repo): return Path(repo) / 'data/08_reporting/sports_reliability_improvements' / run_id

def partition_duration_rows(frame):
    required = ['match_id', 'date', 'release_day', 'max_history_release_day', 'max_pace_release_day']
    if any(c not in frame for c in required): raise ValueError('missing origin/release lineage columns')
    if frame.match_id.isna().any() or frame.match_id.astype(str).duplicated().any(): raise ValueError('duplicate or missing match IDs')
    date = pd.to_datetime(frame.date, utc=True, errors='raise'); release = pd.to_datetime(frame.release_day, utc=True, errors='raise')
    if date.isna().any() or release.isna().any() or (release < date).any(): raise ValueError('invalid label release origin')
    for col in required[3:]:
        prior = pd.to_datetime(frame[col], utc=True, errors='raise')
        if (prior.notna() & (prior >= date)).any(): raise ValueError(f'{col} must be strictly prior to origin')
    t = pd.Timestamp('2025-07-01', tz='UTC'); c = pd.Timestamp('2026-01-01', tz='UTC'); end = pd.Timestamp('2027-01-01', tz='UTC')
    return {'TRAIN': (date < t) & (release < t), 'CAL': (date >= t) & (date < c) & (release < c), 'ASSESS': (date >= c) & (date < end)}

def _freeze_run_impl(run_id, *, repo_root, research_root):
    repo = Path(repo_root).resolve(); roots = _roots(run_id, repo); evidence = _e(run_id, repo)
    contract = _json(evidence / 'p1_contract.json'); authorization = _json(evidence / 'authorization_and_initial_pins.json')
    inputs = contract['resolved_inputs']; old = _json(repo / inputs['binding'])
    paths = set(old['pins']) | set(old.get('protected_hashes', {}))
    paths.update(authorization['protected_before'])
    paths.update(authorization['reviewed_design_hashes'])
    for key in ('protocol','binding','dataset','source','pace_state','pace_audit','selected_bundle','selected_tree','selection','decision','fit_receipts','old_assess_laws','old_cal_laws','old_tree_scores'): paths.add(inputs[key])
    for name in inputs['raw_feature_files']: paths.add(inputs['raw_feature_dir'] + '/' + name)
    paths.update(['src/models/duration_scale_reliability.py','src/analysis/duration_scale_reliability.py','scripts/train_duration_scale.py','scripts/verify_duration_scale_reliability.py','src/analysis/sports_reliability_common.py'])
    paths.update(['src/models/competition_tiers.py','src/analysis/model_reliability.py','conf/base/research_benchmark.json'])
    benchmark = _json(repo/'conf/base/research_benchmark.json')['input']
    taxonomy_path = repo/benchmark['path']
    if not taxonomy_path.exists(): taxonomy_path = Path(research_root).resolve()/benchmark['path']
    if sha256(taxonomy_path) != benchmark['sha256']: raise ValueError('benchmark taxonomy source hash mismatch')
    paths.update(str((evidence / name).relative_to(repo)) for name in ('p1_contract.json','common_contract.json','oracle_contract.json','prefit_design_resolutions.json','authorization_and_initial_pins.json'))
    pins = {p: sha256(repo / p) for p in sorted(paths)}
    for path, expected in {**authorization['protected_before'], **authorization['reviewed_design_hashes']}.items():
        if pins[path] != expected: raise ValueError(f'protected authorization pin changed: {path}')
    for name in ('dataset','source','selected_bundle','selected_tree'):
        if pins[inputs[name]] != inputs[name + '_sha256']: raise ValueError(f'{name} source hash differs from authorization')
    differences = {p: {'old': h, 'current': pins[p]} for p,h in old['pins'].items() if pins[p] != h}
    for path in differences:
        if not path.endswith('.py'): raise ValueError(f'old frozen artifact/data pin changed: {path}')
    roots['05_model_input'].mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=1, study='P1', run_id=run_id, created_at_utc=_now(), scenario=SCENARIO,
        research_root=str(Path(research_root).resolve()), evidence_root=str(evidence.relative_to(repo)),
        contract=contract, optimizer={'TRAIN': {'method':'L-BFGS-B','maxiter':500,'ftol':1e-12,'gtol':1e-8}, 'CAL': {'method':'bounded','maxiter':500,'xatol':1e-8}},
        operation_trace=['TRAIN_REFERENCE_SIGMA','TRAIN_HEAD','PG_CAL','PC_CAL'], primary_slots=[1,2], family_slots=10,
        month_universe=MONTHS, draws=10000, seed=20261009, old_source_discrepancies=differences,
        public_case_selection='first lexical match_id per BO/history/missing-gap state, maximum32',
        oracle_schema=_json(evidence / 'oracle_contract.json'), prefit_resolutions=_json(evidence / 'prefit_design_resolutions.json'),
        taxonomy_metadata_path=str(taxonomy_path),taxonomy_metadata_sha256=benchmark['sha256'],
        profiles={'major':'classify_competition(tournament, UTC origin date).tier.value==major; other explicit tiers nonmajor; missing tournament UNKNOWN', 'source_unseen':'benchmark unknown_players>0 unseen; ==0 seen; absent/null UNKNOWN; exact golgg_match_id mapping','proxy':'tournament source name within calendar year; UNKNOWN retained'})
    _save(roots['05_model_input'] / 'protocol.json', protocol)
    binding = dict(schema_version=1, run_id=run_id, created_at_utc=_now(), pins=pins, old_expected_pins=old['pins'], old_source_discrepancies=differences,
                   feature_columns=old['feature_columns'], dataset_path=inputs['dataset'], selected_bundle=inputs['selected_bundle'],
                   protocol_sha256=sha256(roots['05_model_input'] / 'protocol.json'), authorization=authorization)
    binding['external_pins'] = {str(taxonomy_path):benchmark['sha256']}
    _save(roots['05_model_input'] / 'binding.json', binding)
    _save(roots['05_model_input']/'freeze_receipt.json',dict(status='FROZEN',created_at_utc=_now(),estimation_operations=0,protocol_sha256=sha256(roots['05_model_input']/'protocol.json'),binding_sha256=sha256(roots['05_model_input']/'binding.json')))
    return {'status':'FROZEN','run_id':run_id,'old_source_discrepancies':differences,'estimation_operations':0}

def freeze_run(run_id, *, repo_root, research_root):
    roots = _roots(run_id, Path(repo_root).resolve())
    if roots['05_model_input'].exists(): raise FileExistsError(roots['05_model_input'])
    try:
        return _freeze_run_impl(run_id,repo_root=repo_root,research_root=research_root)
    except Exception as exc:
        roots['05_model_input'].mkdir(parents=True,exist_ok=True)
        failure = dict(status='INVALID_FREEZE',created_at_utc=_now(),reason=str(exc),estimation_operations=0,primary_slots={'1':'NOT_RUN','2':'NOT_RUN'})
        _save(roots['05_model_input']/'freeze_failure.json',failure)
        roots['08_reporting'].mkdir(parents=True,exist_ok=False)
        _save(roots['08_reporting']/'decision.json',dict(failure,status='NO_ADOPTION',branch='NO_FIT'))
        _save(roots['08_reporting']/'fit_receipts.json',dict(operation_trace=[],receipts=[],estimation_operations=0))
        evidence = _e(run_id,repo_root)
        oracle_path = evidence/'oracle_contract.json'
        _save(roots['08_reporting']/'oracle_inputs.json',dict(schema_version=1,study='P1',run_id=run_id,scenario=SCENARIO,branch_status='INVALID_FREEZE',created_at_utc=_now(),protocol_sha256=None,binding_sha256=None,oracle_contract_sha256=sha256(oracle_path) if oracle_path.exists() else None,artifact_descriptors=[dict(role='freeze_failure',repo_relative_path=str((roots['05_model_input']/'freeze_failure.json').relative_to(Path(repo_root).resolve())),sha256=sha256(roots['05_model_input']/'freeze_failure.json'),format='json',shape_or_row_count=None,dtype_or_columns=None,units=None,ordered_axis_names=[],ordered_id_axis_reference=None,absence_status=None,absence_reason=None)],population_ids={},operation_trace=[],missingness_ledger={'reason':'source closure could not be frozen','failure_reason':str(exc)},public_cases={'status':'NOT_RUN','reason':'INVALID_FREEZE','cases':[]},prefit_resolutions=_json(evidence/'prefit_design_resolutions.json') if (evidence/'prefit_design_resolutions.json').exists() else None))
        raise

def _validate(run_id, repo):
    roots = _roots(run_id, repo); binding = _json(roots['05_model_input'] / 'binding.json'); protocol = _json(roots['05_model_input'] / 'protocol.json')
    if sha256(roots['05_model_input'] / 'protocol.json') != binding['protocol_sha256']: raise ValueError('protocol hash mismatch')
    for path, digest in binding['pins'].items():
        if sha256(Path(repo) / path) != digest: raise ValueError(f'frozen pin mismatch: {path}')
    for path,digest in binding.get('external_pins',{}).items():
        if sha256(path) != digest: raise ValueError(f'frozen external source pin mismatch: {path}')
    return roots, binding, protocol

def prepare_run(run_id, *, repo_root, research_root):
    repo = Path(repo_root).resolve(); roots,binding,protocol = _validate(run_id, repo)
    roots['04_feature'].mkdir(parents=True, exist_ok=False)
    populations = {}
    try:
        frame = pd.read_parquet(repo / binding['dataset_path']); masks = partition_duration_rows(frame)
        counts = {p:int(m.sum()) for p,m in masks.items()}
        populations = {p:frame.loc[m,'match_id'].astype(str).tolist() for p,m in masks.items()}
        raw_closure = frame.copy()
        raw_closure['row_index'] = np.arange(len(frame))
        raw_closure['partition'] = 'EXCLUDED'
        for partition, mask in masks.items(): raw_closure.loc[mask,'partition'] = partition
        raw_closure['month'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m')
        raw_closure['history'] = np.select([frame[COLUMNS[0]].eq(16),frame[COLUMNS[0]].gt(0)&frame[COLUMNS[0]].lt(16),frame[COLUMNS[0]].eq(0)],['full','partial','cold'],default='UNKNOWN')
        raw_closure['gap_observed'] = frame[COLUMNS[1]].notna()
        raw_closure['source_match_id'] = frame.match_id
        raw_closure['target_origin_day'] = frame.date
        raw_closure['effective_release_day'] = frame.release_day
        raw_closure['competition_context_id'] = frame.tournament
        raw_closure['tournament_source_name'] = frame.tournament
        raw_closure['scenario'] = SCENARIO
        raw_closure['label_availability'] = frame.duration_minutes.notna()
        raw_closure['label_absence_reason'] = np.where(raw_closure.partition.eq('EXCLUDED'),'OUTSIDE_RELEASE_ELIGIBLE_PARTITIONS',None)
        for target, source in [('team_a_id','team_a'),('team_b_id','team_b'),('roster_a_ids','roster_a'),('roster_b_ids','roster_b')]: raw_closure[target] = frame[source]
        raw_closure['winner_label'] = frame['winner'] if 'winner' in frame else None
        raw_closure['terminal_score_label'] = frame.maps_played
        raw_closure['max_used_release_day'] = pd.concat([pd.to_datetime(frame.max_history_release_day,utc=True),pd.to_datetime(frame.max_pace_release_day,utc=True)],axis=1).max(axis=1)
        raw_closure.to_parquet(roots['04_feature']/'raw_closure.parquet',index=False)
        _save(roots['04_feature']/'raw_inventory.json',dict(created_at_utc=_now(),counts=counts,population_ids=populations,raw_closure_sha256=sha256(roots['04_feature']/'raw_closure.parquet'),canonical_source_sha256=binding['pins'][protocol['contract']['resolved_inputs']['source']],estimation_operations=0))
        if counts != {'TRAIN':36205,'CAL':1328,'ASSESS':3103}: raise ValueError(f'canonical population mismatch: {counts}')
        labels = frame.duration_minutes.to_numpy(float)
        selected = np.logical_or.reduce([m.to_numpy() for m in masks.values()])
        if not np.isfinite(labels[selected]).all() or np.any(labels[selected] <= 0): raise ValueError('positive finite duration labels required on complete canonical population')
        pre = ScalePreprocessor.fit(frame.loc[masks['TRAIN']]); z = pre.transform(frame)
        tree = load_selected_tree(repo / binding['selected_bundle'], expected_sha256=binding['pins'][binding['selected_bundle']], feature_columns=binding['feature_columns'])
        if not np.isclose(tree.model.duration_sigma, protocol['contract']['resolved_inputs']['old_sigma0'], rtol=0, atol=1e-14): raise ValueError('old CAL sigma identity mismatch')
        base = tree.predict(frame)
        metadata = pd.read_parquet(repo / protocol['contract']['resolved_inputs']['raw_feature_dir'] / 'metadata.parquet')
        if metadata.match_id.astype(str).duplicated().any():
            raise ValueError('raw metadata source identity duplicate')
        if not np.array_equal(frame.match_id.astype(str).to_numpy(),metadata.match_id.astype(str).to_numpy()):
            raise ValueError('dataset identity/order differs from pinned raw metadata')
        for column in ['date','release_day','best_of','team_a','team_b','tournament']:
            if not np.array_equal(frame[column].astype(str).to_numpy(),metadata[column].astype(str).to_numpy()):
                raise ValueError(f'raw metadata identity/origin mismatch: {column}')
        for column in ['roster_a','roster_b']:
            if any(list(a) != list(b) for a,b in zip(frame[column],metadata[column])):
                raise ValueError(f'raw metadata roster slot mismatch: {column}')
        parity = {}
        for partition, key in [('CAL','old_cal_laws'),('ASSESS','old_assess_laws')]:
            old_laws = pd.read_parquet(repo / protocol['contract']['resolved_inputs'][key])
            old_laws = old_laws.loc[old_laws.candidate.eq('tree_leaves_15')]
            expected_ids = frame.loc[masks[partition], 'match_id'].astype(str).to_numpy()
            if not np.array_equal(old_laws.match_id.astype(str).to_numpy(), expected_ids):
                raise ValueError(f'{partition} saved tree exact population/order mismatch')
            parity[partition] = dict(n_rows=len(old_laws),ordered_ids_sha256=_ids(expected_ids),old_laws_sha256=binding['pins'][protocol['contract']['resolved_inputs'][key]],fields={})
            for field in ['duration_log_mu','duration_log_sigma']:
                actual = getattr(base, field)[masks[partition].to_numpy()]
                parity[partition]['fields'][field] = dict(max_absolute_error=float(np.max(np.abs(actual-old_laws[field].to_numpy()))),atol=1e-10,rtol=1e-10)
                if not np.allclose(actual, old_laws[field].to_numpy(), atol=1e-10, rtol=1e-10):
                    raise ValueError(f'{partition} isolated P0 saved-law equivalence failed: {field}')
        assess_frame = frame.loc[masks['ASSESS']]
        old_scores = pd.read_parquet(repo/protocol['contract']['resolved_inputs']['old_tree_scores'])
        if not np.array_equal(old_scores.match_id.astype(str).to_numpy(),assess_frame.match_id.astype(str).to_numpy()):
            raise ValueError('ASSESS saved score exact population/order mismatch')
        assess_prediction = tree.predict(assess_frame)
        replay_scores = {
            'duration_nll':distribution_nll(assess_prediction,assess_frame.kills.to_numpy(),assess_frame.duration_minutes.to_numpy(),assess_frame.maps_played.to_numpy())['duration'],
            'duration_proper_secondary':lognormal_mixture_crps(assess_frame.duration_minutes.to_numpy(),assess_prediction.duration_log_mu[:,None],assess_prediction.duration_log_sigma[:,None],np.ones(1)),
            'duration_mean':np.exp(assess_prediction.duration_log_mu+.5*assess_prediction.duration_log_sigma**2)}
        parity['ASSESS']['scores'] = {}
        for field, actual in replay_scores.items():
            parity['ASSESS']['scores'][field] = dict(max_absolute_error=float(np.max(np.abs(actual-old_scores[field].to_numpy()))),atol=1e-10,rtol=1e-10)
            if not np.allclose(actual,old_scores[field].to_numpy(),atol=1e-10,rtol=1e-10):
                raise ValueError(f'ASSESS P0 old saved proper-score equivalence failed: {field}')
        _save(roots['04_feature']/'p0_parity.json',dict(status='PASS',created_at_utc=_now(),estimation_operations=0,partitions=parity,old_source_discrepancies=binding['old_source_discrepancies'],scope='isolated pinned raw saved-tree CAL/ASSESS laws; not old native runner source closure certification'))
        frame = frame.copy(); frame['row_index'] = np.arange(len(frame)); frame['partition'] = 'EXCLUDED'
        for p,m in masks.items(): frame.loc[m,'partition'] = p
        frame['mu0'] = base.duration_log_mu; frame['sigma0'] = base.duration_log_sigma
        for j in range(3): frame[f'z{j}'] = z[:,j]
        frame['month'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m')
        frame['history'] = np.select([frame[COLUMNS[0]].eq(16), frame[COLUMNS[0]].gt(0)&frame[COLUMNS[0]].lt(16),frame[COLUMNS[0]].eq(0)], ['full','partial','cold'], default='UNKNOWN')
        frame['major_state'] = [classify_competition(str(name),pd.Timestamp(day).date()).tier.value for name,day in zip(frame.tournament,frame.date)]
        frame['major_state'] = np.where(frame.tournament.isna() | frame.major_state.eq('unknown'),'UNKNOWN',np.where(frame.major_state.eq('major'),'major','nonmajor'))
        taxonomy = pd.read_parquet(protocol['taxonomy_metadata_path'],columns=['golgg_match_id','unknown_players'])
        taxonomy['match_id'] = taxonomy.golgg_match_id.astype(str)
        if taxonomy.match_id.duplicated().any(): raise ValueError('benchmark taxonomy duplicate canonical IDs')
        unknown = frame.match_id.astype(str).map(taxonomy.set_index('match_id').unknown_players)
        frame['source_unseen_state'] = np.select([unknown.gt(0),unknown.eq(0)],['unseen','seen'],default='UNKNOWN')
        frame['proxy_block'] = frame.tournament.fillna('UNKNOWN').astype(str) + '/' + pd.to_datetime(frame.date).dt.year.astype(str)
        frame['gap_observed'] = frame[COLUMNS[1]].notna()
        frame['label_availability'] = frame.duration_minutes.notna()
        frame['label_absence_reason'] = np.where(frame.partition.eq('EXCLUDED'), 'OUTSIDE_RELEASE_ELIGIBLE_PARTITIONS', None)
        frame.to_parquet(roots['04_feature'] / 'prepared.parquet', index=False)
        ledger = frame[['row_index','match_id','date','release_day','partition','best_of','tournament','team_a','team_b','roster_a','roster_b','duration_minutes','label_availability','label_absence_reason']].copy()
        ledger['source_match_id'] = ledger.match_id
        ledger['target_origin_day'] = ledger.date
        ledger['effective_release_day'] = ledger.release_day
        ledger['competition_context_id'] = ledger.tournament
        ledger['tournament_source_name'] = ledger.tournament
        for target, source in [('team_a_id','team_a'),('team_b_id','team_b'),('roster_a_ids','roster_a'),('roster_b_ids','roster_b')]:
            ledger[target] = ledger[source]
        ledger['winner_label'] = frame['winner'] if 'winner' in frame else None
        ledger['terminal_score_label'] = frame['maps_played']
        ledger['max_used_release_day'] = pd.concat([pd.to_datetime(frame.max_history_release_day,utc=True),pd.to_datetime(frame.max_pace_release_day,utc=True)],axis=1).max(axis=1)
        ledger['scenario'] = SCENARIO
        ledger.to_parquet(roots['04_feature']/'series_ledger.parquet',index=False)
        frame[['row_index','match_id','partition','month','best_of','history','major_state','source_unseen_state','proxy_block','gap_observed','label_availability','label_absence_reason']].to_parquet(roots['04_feature']/'memberships.parquet',index=False)
        rules = {'partition':protocol['contract']['chronology'],'month':'UTC origin year-month','best_of':'inherited canonical BO1/3/5; unknown not recoded','history':f'{COLUMNS[0]}==16 full; 0<count<16 partial; ==0 cold; otherwise UNKNOWN','gap_observed':f'finite observed {COLUMNS[1]}','major_state':protocol['profiles']['major'],'source_unseen_state':protocol['profiles']['source_unseen'],'proxy_block':protocol['profiles']['proxy'],'label_availability':'observed positive finite duration in release-eligible partition'}
        membership_receipt = {}
        for column, rule in rules.items():
            rule_bytes = json.dumps(rule,sort_keys=True).encode()
            membership_receipt[column] = dict(rule=rule,rule_sha256=hashlib.sha256(rule_bytes).hexdigest(),source_snapshot_sha256=binding['pins'][binding['dataset_path']],states={str(state):{'n_series':int(frame[column].eq(state).sum()),'month_counts':{month:int((frame[column].eq(state)&frame.month.eq(month)).sum()) for month in MONTHS}} for state in frame[column].dropna().unique()},unknown_count=int(frame[column].isna().sum()+frame[column].eq('UNKNOWN').sum()))
        _save(roots['04_feature']/'membership_rules.json',membership_receipt)
        np.savez_compressed(roots['04_feature']/'train_design.npz', match_id=np.asarray(frame.loc[masks['TRAIN'],'match_id'].astype(str),dtype=str), z=z[masks['TRAIN'].to_numpy()], mu0=base.duration_log_mu[masks['TRAIN'].to_numpy()], y=labels[masks['TRAIN'].to_numpy()], gap_observed=frame.loc[masks['TRAIN'],'gap_observed'].to_numpy())
        raw,_ = pre._raw(frame.loc[masks['TRAIN']], pre.gap_median)
        normalizer = dict(columns=COLUMNS, gap_median=pre.gap_median, mean=pre.mean.tolist(), scale=pre.scale.tolist(), ddof=0,
            train_ids_sha256=_ids(frame.loc[masks['TRAIN'],'match_id']), n_rows=counts['TRAIN'], counts=np.isfinite(raw).sum(0).tolist(),
            sums=raw.sum(0).tolist(), sums_of_squares=(raw**2).sum(0).tolist(), rank=4, raw_variance=raw.var(0).tolist(),
            observed_gap_count=int(frame.loc[masks['TRAIN'],COLUMNS[1]].notna().sum()), dtype='float64', clipping=[-3,3], missingness_unstandardized=True)
        observed_raw = raw.copy()
        observed_raw[:,1] = frame.loc[masks['TRAIN'],COLUMNS[1]].to_numpy(float)
        normalizer.update(observed_counts=np.isfinite(observed_raw).sum(0).tolist(),observed_sums=np.nansum(observed_raw,axis=0).tolist(),observed_sums_of_squares=np.nansum(observed_raw**2,axis=0).tolist(),zero_variance_handling='HARD_STOP',rounding='float64 throughout')
        _save(roots['04_feature'] / 'normalizer.json', normalizer)
        receipt = dict(status='PREPARED', created_at_utc=_now(), counts=counts, estimation_operations=0,
                       prepared_sha256=sha256(roots['04_feature'] / 'prepared.parquet'), normalizer_sha256=sha256(roots['04_feature'] / 'normalizer.json'),p0_parity_sha256=sha256(roots['04_feature']/'p0_parity.json'),
                       population_ids={p:frame.loc[m,'match_id'].astype(str).tolist() for p,m in masks.items()}, old_source_discrepancies=binding['old_source_discrepancies'])
    except Exception as exc:
        failed = {'status':'INVALID_PREPARE','created_at_utc':_now(),'estimation_operations':0,'reason':str(exc),'primary_slots':{'1':'NOT_RUN','2':'NOT_RUN'},'population_ids':populations}
        _save(roots['05_model_input'] / 'prepare_receipt.json', failed)
        roots['08_reporting'].mkdir(parents=True, exist_ok=False)
        _save(roots['08_reporting']/'decision.json',dict(status='NO_ADOPTION',branch='NO_FIT',reason=str(exc),estimation_operations=0,primary_slots={'1':'NOT_RUN','2':'NOT_RUN'},created_at_utc=_now()))
        _save(roots['08_reporting']/'fit_receipts.json',dict(operation_trace=[],receipts=[],estimation_operations=0))
        _oracle_index(run_id,repo,roots,binding,protocol,failed,[],'INVALID_PREPARE')
        raise
    _save(roots['05_model_input'] / 'prepare_receipt.json', receipt)
    return receipt

def _contrast_exports(delta, blocks, universe, result, destination):
    raw = contrast_draw_arrays(delta, blocks, universe=universe)
    np.savez_compressed(destination, delta=delta, blocks=np.asarray(blocks,dtype=str),
        universe=np.asarray(raw['universe'],dtype=str), indices=raw['draw_indices'],
        counts=raw['block_counts'],sums=raw['block_sums'],means=raw['block_means'],
        weighted_denominator=raw['weighted_denominator'],equal_denominator=raw['equal_denominator'],
        weighted_delta=raw['weighted_delta'],equal_delta=raw['equal_delta'],undefined=raw['undefined'])

def _metrics(laws):
    output = {}
    for arm, rows in laws.groupby('arm',sort=False):
        metric = {'n':len(rows),'nll':float(rows.nll.mean()),'crps':float(rows.crps.mean()),'mean_mae':float(np.abs(rows.fixed_mean-rows.duration_minutes).mean()),'mean_bias':float((rows.fixed_mean-rows.duration_minutes).mean()),'pit_mean':float(rows.pit.mean()),'pit_histogram':np.histogram(rows.pit,bins=np.linspace(0,1,11))[0].tolist()}
        for level,lo,hi in [(50,'q25','q75'),(80,'q10','q90'),(90,'q05','q95'),(95,'q025','q975')]:
            coverage = float(((rows.duration_minutes>=rows[lo])&(rows.duration_minutes<=rows[hi])).mean())
            metric[f'coverage{level}']=coverage; metric[f'coverage_error{level}']=abs(coverage-level/100); metric[f'width{level}']=float((rows[hi]-rows[lo]).mean())
        for line in [25.5,28.5,31.5,34.5,37.5]: metric[f'brier_{line}']=float(((rows[f'p_over_{line}']-(rows.duration_minutes>line))**2).mean())
        output[arm]=metric
    return output

def run_experiment(run_id, *, repo_root, research_root):
    repo=Path(repo_root).resolve(); roots,binding,protocol=_validate(run_id,repo)
    prepared=_json(roots['05_model_input']/'prepare_receipt.json')
    if prepared['status']!='PREPARED': raise ValueError('successful frozen prepare required')
    for name,key in [('prepared.parquet','prepared_sha256'),('normalizer.json','normalizer_sha256'),('p0_parity.json','p0_parity_sha256')]:
        if sha256(roots['04_feature']/name)!=prepared[key]: raise ValueError('prepared artifact pin mismatch')
    if _json(roots['04_feature']/'p0_parity.json')['status'] != 'PASS': raise ValueError('observed full old saved P0 parity required before fits')
    approval_path=roots['05_model_input']/'prefit_approval.json'
    approval=_json(approval_path)
    if approval.get('status')!='APPROVED' or approval.get('protocol_sha256')!=binding['protocol_sha256'] or approval.get('binding_sha256')!=sha256(roots['05_model_input']/'binding.json'): raise ValueError('parent scientific/oracle prefit approval not bound')
    if binding['old_source_discrepancies'] and approval.get('allow_isolated_saved_tree_replay') is not True: raise ValueError('parent provenance resolution required for old/current closure discrepancies')
    if binding['old_source_discrepancies']:
        if approval.get('approved_source_discrepancies') != binding['old_source_discrepancies'] or approval.get('noncandidate_source_only') is not True:
            raise ValueError('parent must explicitly qualify exact old/current noncandidate source discrepancies')
    for stage in ['06_models','07_model_output','08_reporting']: roots[stage].mkdir(parents=True,exist_ok=False)
    frame=pd.read_parquet(roots['04_feature']/'prepared.parquet'); norm=_json(roots['04_feature']/'normalizer.json')
    pre=ScalePreprocessor(norm['gap_median'],np.array(norm['mean']),np.array(norm['scale']))
    tree=load_selected_tree(repo/binding['selected_bundle'],expected_sha256=binding['pins'][binding['selected_bundle']],feature_columns=binding['feature_columns'])
    trace=[]; receipts=[]
    def estimate(kind, partition, function):
        row=dict(operation_id=kind,study='P1',kind=kind,arm={'TRAIN_REFERENCE_SIGMA':'TRAIN','TRAIN_HEAD':'TRAIN','PG_CAL':'PG','PC_CAL':'PC'}[kind],seed=None,partition=partition,authorized_ceiling=4,counter_before=len(trace),counter_after=len(trace)+1,started_at_utc=_now(),finished_at_utc=None,status='STARTED',parameters_sha256=None,input_ids_sha256=_ids(frame.loc[frame.partition.eq(partition),'match_id']),iterations=None,function_evaluations=None,failure_reason=None)
        trace.append(row)
        _save(roots['08_reporting']/f'operation_{len(trace):02d}_started.json',row)
        try:
            value,receipt=function(); row.update(status='SUCCESS',finished_at_utc=_now(),iterations=receipt.get('iterations',0),function_evaluations=receipt.get('nfev',0),parameters_sha256=hashlib.sha256(json.dumps(receipt,sort_keys=True,allow_nan=False).encode()).hexdigest()); receipts.append(receipt)
            _save(roots['08_reporting']/f'operation_{len(trace):02d}_finished.json',row)
            return value
        except Exception as exc:
            row.update(status='FAILED',finished_at_utc=_now(),failure_reason=str(exc)); _save(roots['08_reporting']/f'operation_{len(trace):02d}_finished.json',row); raise
    train=frame.loc[frame.partition.eq('TRAIN')]; cal=frame.loc[frame.partition.eq('CAL')]
    z_train=pre.transform(train); z_cal=pre.transform(cal)
    mu_train=train.mu0.to_numpy(); y_train=train.duration_minutes.to_numpy()
    mu_cal=cal.mu0.to_numpy(); y_cal=cal.duration_minutes.to_numpy()
    try:
        def sigma_operation():
            value=train_reference_sigma(train.mu0,train.duration_minutes)
            return value,dict(stage='TRAIN_REFERENCE_SIGMA',status='success',parameters=[value],estimation_operations=1,objective=None,iterations=0,nfev=0)
        sigma_t=estimate('TRAIN_REFERENCE_SIGMA','TRAIN',sigma_operation)
        def head():
            result=minimize(train_scale_objective,np.zeros(4),args=(z_train,mu_train,y_train,sigma_t),jac=True,method='L-BFGS-B',bounds=[(-1,1)]*4,options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
            return result.x,accept_fit_result(result,stage='TRAIN_HEAD')
        theta=estimate('TRAIN_HEAD','TRAIN',head)
        def scalar(stage,slopes):
            def function():
                result=minimize_scalar(lambda c: cal_scale_objective(c,z_cal,mu_cal,y_cal,tree.model.duration_sigma,slopes)[0],bounds=(np.log(.25),np.log(4)),method='bounded',options={'maxiter':500,'xatol':1e-8})
                loss,jac=cal_scale_objective(result.x,z_cal,mu_cal,y_cal,tree.model.duration_sigma,slopes)
                checked=SimpleNamespace(success=result.success,x=np.array([result.x]),fun=loss,jac=np.array([jac]),nit=result.nit,nfev=result.nfev,message=result.message)
                return float(result.x),accept_fit_result(checked,stage=stage)
            return function
        cg=estimate('PG_CAL','CAL',scalar('PG_CAL',np.zeros(3))); cc=estimate('PC_CAL','CAL',scalar('PC_CAL',theta[1:]))
        model=DurationScaleModel(tree=tree,preprocessor=pre,slopes=theta[1:],c_global=cg,c_conditional=cc,train_reference_sigma=sigma_t,train_intercept=float(theta[0]),fit_receipt={'estimation_operations':4,'receipts':receipts})
        model.save(roots['06_models']/'duration_scale.joblib')
        _save(roots['06_models']/'receipt.json',dict(created_at_utc=_now(),artifact_sha256=sha256(roots['06_models']/'duration_scale.joblib'),sigma_T=sigma_t,a=float(theta[0]),b=theta[1:].tolist(),c_G=cg,c_C=cc,receipts=receipts,estimation_operations=4))
        public_cases = []
        absent_states = []
        lexical = frame.loc[frame.partition.ne('EXCLUDED')].assign(_id=frame.loc[frame.partition.ne('EXCLUDED'),'match_id'].astype(str)).sort_values('_id')
        state_masks = {f'BO{bo}':lexical.best_of.eq(bo) for bo in [1,3,5]}
        state_masks.update({f'history_{h}':lexical.history.eq(h) for h in ['full','partial','cold']})
        state_masks.update({'gap_missing':lexical[COLUMNS[1]].isna(),'gap_observed':lexical[COLUMNS[1]].notna()})
        for state, mask in state_masks.items():
            if not mask.any():
                absent_states.append({'state':state,'status':'ABSENT_IN_SOURCE'})
                continue
            row = lexical.loc[mask].iloc[:1].drop(columns='_id')
            before = hashlib.sha256(np.r_[model.slopes,model.preprocessor.mean,model.preprocessor.scale,model.c_global,model.c_conditional,model.train_intercept].tobytes()).hexdigest()
            swapped = row.copy()
            for left,right in [('team_a','team_b'),('roster_a','roster_b')]:
                swapped[left],swapped[right] = row[right].copy(),row[left].copy()
            for arm in ['P0','PG','PC']:
                actual = model.predict(row,arm=arm); reverse = model.predict(swapped,arm=arm)
                mean = np.exp(actual.duration_log_mu + .5*actual.duration_log_sigma**2)
                if not np.allclose(actual.duration_over(31.5),reverse.duration_over(31.5),atol=1e-8,rtol=0):
                    raise ValueError('public side-swap CDF contract failed')
                if not np.allclose(mean,np.exp(row.mu0.to_numpy()+.5*row.sigma0.to_numpy()**2),rtol=1e-10,atol=0):
                    raise ValueError('public mean-preservation contract failed')
                after = hashlib.sha256(np.r_[model.slopes,model.preprocessor.mean,model.preprocessor.scale,model.c_global,model.c_conditional,model.train_intercept].tobytes()).hexdigest()
                if before != after: raise ValueError('public prediction mutated model scale state')
                public_cases.append(dict(case_id=f'{state}/{arm}',study='P1',arm=arm,seed_or_ensemble=None,match_id=str(row.match_id.iloc[0]),input_reference_and_sha256={'path':str((roots['04_feature']/'prepared.parquet').relative_to(repo)),'sha256':prepared['prepared_sha256']},target_origin=str(row.date.iloc[0]),scenario=SCENARIO,contract_state=state,expected_status='PREDICTED',expected_reason=None,public_result_reference={'duration_log_mu':float(actual.duration_log_mu[0]),'duration_log_sigma':float(actual.duration_log_sigma[0]),'predictive_mean':float(mean[0]),'p_over':{str(line):float(actual.duration_over(line)[0]) for line in [25.5,28.5,31.5,34.5,37.5]}},independent_expected_reference='independent verifier reconstruction from pinned row/tree/parameters',comparison_kind='singleton_and_swap',tolerances={'mean_rtol':1e-10,'cdf_atol':1e-8},state_hash_before=before,state_hash_after=after))
        _save(roots['07_model_output']/'public_cases.json',dict(selection_rule=protocol['public_case_selection'],cases=public_cases,absent_states=absent_states))
        outputs=[]
        retained=frame.loc[frame.partition.ne('EXCLUDED')]
        for arm in ['P0','PG','PC']:
            prediction=model.predict(retained,arm=arm)
            rows=retained[['match_id','date','best_of','tournament','partition','duration_minutes','mu0','sigma0','history','month','proxy_block','major_state','source_unseen_state']].copy()
            rows['gap_observed'] = retained.gap_observed
            rows['missing_player_count'] = retained[COLUMNS[2]]
            rows['source_present'] = retained.source_present
            rows['arm']=arm; rows['fixed_mean']=np.exp(rows.mu0+.5*rows.sigma0**2); rows['duration_log_mu']=prediction.duration_log_mu; rows['duration_log_sigma']=prediction.duration_log_sigma
            rows['predictive_mean'] = np.exp(prediction.duration_log_mu+.5*prediction.duration_log_sigma**2)
            if not np.allclose(rows.predictive_mean,rows.fixed_mean,rtol=1e-10,atol=0):
                raise ValueError('full-population predictive mean identity failed')
            rows['kills_mu'] = prediction.kills_mu
            rows['kills_alpha'] = prediction.kills_alpha
            for length in range(3): rows[f'length_p_{length}'] = prediction.length_probs[:,length]
            rows['pit']=lognorm.cdf(rows.duration_minutes,s=prediction.duration_log_sigma,scale=np.exp(prediction.duration_log_mu))
            rows['nll']=distribution_nll(prediction,retained.kills.to_numpy(),retained.duration_minutes.to_numpy(),retained.maps_played.to_numpy())['duration']
            rows['crps']=lognormal_mixture_crps(rows.duration_minutes.to_numpy(),prediction.duration_log_mu[:,None],prediction.duration_log_sigma[:,None],np.ones(1))
            for name,q in [('q025',.025),('q05',.05),('q10',.1),('q25',.25),('q50',.5),('q75',.75),('q90',.9),('q95',.95),('q975',.975)]: rows[name]=lognorm.ppf(q,s=prediction.duration_log_sigma,scale=np.exp(prediction.duration_log_mu))
            for line in [25.5,28.5,31.5,34.5,37.5]: rows[f'p_over_{line}']=prediction.duration_over(line)
            rows['eligible']=True; rows['absence_reason']=None; outputs.append(rows)
            if not np.isfinite(rows[['duration_log_mu','duration_log_sigma','pit','nll','crps']].to_numpy()).all():
                raise ValueError('nonfinite law or proper score on a retained row')
        laws=pd.concat(outputs,ignore_index=True); laws.to_parquet(roots['07_model_output']/'duration_predictions.parquet',index=False)
        assess=laws.loc[laws.partition.eq('ASSESS')]; metrics=_metrics(assess)
        comparisons={}; guards={}; byarm={a:assess.loc[assess.arm.eq(a)].reset_index(drop=True) for a in ['P0','PG','PC']}
        pc=byarm['PC']; blocks=pc.month.to_numpy(); cohort=[]
        contrast_inputs = pc[['match_id','month','proxy_block','best_of','history']].copy()
        contrast_inputs['month_code'] = contrast_inputs.month.map({month:i for i,month in enumerate(MONTHS)})
        for arm, rows in byarm.items():
            if not np.array_equal(rows.match_id.to_numpy(),pc.match_id.to_numpy()):
                raise ValueError('candidate/reference exact canonical population mismatch')
            contrast_inputs[f'{arm}_nll'] = rows.nll
            contrast_inputs[f'{arm}_crps'] = rows.crps
            contrast_inputs[f'{arm}_missing'] = ~np.isfinite(rows.nll)
            contrast_inputs[f'{arm}_covered80'] = (rows.duration_minutes>=rows.q10)&(rows.duration_minutes<=rows.q90)
            contrast_inputs[f'{arm}_covered50'] = (rows.duration_minutes>=rows.q25)&(rows.duration_minutes<=rows.q75)
        for ref in ['P0','PG']:
            contrast_inputs[f'PC_minus_{ref}_nll'] = contrast_inputs.PC_nll-contrast_inputs[f'{ref}_nll']
            contrast_inputs[f'PC_minus_{ref}_crps'] = contrast_inputs.PC_crps-contrast_inputs[f'{ref}_crps']
        contrast_inputs['eligible'] = True
        contrast_inputs['paired_absent'] = False
        contrast_inputs.to_parquet(roots['08_reporting']/'contrast_inputs.parquet',index=False)
        for slot,ref in [(1,'P0'),(2,'PG')]:
            delta=pc.nll.to_numpy()-byarm[ref].nll.to_numpy(); result=paired_block_contrast(delta,blocks,universe=MONTHS)
            result.update(slot=slot,candidate='PC',reference=ref,metric='duration_density_nll_minutes',passed=result['status']=='ESTIMABLE' and result['upper'] is not None and result['upper']<-.002 and result['equal_block']['point']<0)
            comparisons[f'PC-{ref}']=result; _contrast_exports(delta,blocks,MONTHS,result,roots['08_reporting']/f'contrast_PC_{ref}.npz')
            proxy=pc.proxy_block.to_numpy(); universe=sorted(set(proxy)); sensitivity=paired_block_contrast(delta,proxy,universe=universe,confidence=.95,min_blocks=1)
            comparisons[f'proxy_PC-{ref}']=sensitivity; _contrast_exports(delta,proxy,universe,sensitivity,roots['08_reporting']/f'proxy_PC_{ref}.npz')
            crps_delta=pc.crps.to_numpy()-byarm[ref].crps.to_numpy(); comparisons[f'crps_PC-{ref}']=paired_block_contrast(crps_delta,blocks,universe=MONTHS)
            _contrast_exports(crps_delta,blocks,MONTHS,comparisons[f'crps_PC-{ref}'],roots['08_reporting']/f'crps_PC_{ref}.npz')
            cohort.append(metrics['PC']['crps']<=metrics[ref]['crps'] and metrics['PC']['coverage_error50']-metrics[ref]['coverage_error50']<=.005)
        cohort.append(metrics['P0']['coverage_error80']-metrics['PC']['coverage_error80']>=.01 and metrics['PC']['coverage_error80']<=metrics['PG']['coverage_error80'])
        for group,mask in [('BO1',pc.best_of.eq(1)),('BO3',pc.best_of.eq(3)),('BO5',pc.best_of.eq(5)),('history_full',pc.history.eq('full')),('history_partial',pc.history.eq('partial'))]:
            for ref in ['P0','PG']:
                delta=pc.nll.to_numpy()-byarm[ref].nll.to_numpy(); result=paired_block_contrast(delta,blocks,mask=mask.to_numpy(),universe=MONTHS,one_sided=True,min_rows=100,min_blocks=4)
                sub=_metrics(pd.concat([pc.loc[mask],byarm[ref].loc[mask]])) if mask.any() else {}
                worsening=sub['PC']['coverage_error80']-sub[ref]['coverage_error80'] if sub else None
                result.update(candidate='PC',reference=ref,group=group,coverage_error80_worsening=worsening,passed=result['status']=='ESTIMABLE' and result['upper'] is not None and result['upper']<=.01 and worsening is not None and worsening<=.025)
                guards[f'{group}/{ref}']=result
                _contrast_exports(delta[mask],blocks[mask],MONTHS,result,roots['08_reporting']/f'guard_{group}_{ref}.npz')
        profiles={}
        for col in ['month','best_of','history','major_state','source_unseen_state','gap_observed','missing_player_count','source_present']:
            profiles[col]={str(value):_metrics(assess.loc[assess[col].eq(value)]) for value in assess[col].unique()}
        metrics={'ASSESS':metrics,'partitions':{p:_metrics(laws.loc[laws.partition.eq(p)]) for p in ['TRAIN','CAL']},'profiles':profiles,'cohort_guard_passed':all(cohort)}
        decision=dict(status='DEVELOPMENT_CANDIDATE' if all(comparisons[f'PC-{ref}']['passed'] for ref in ['P0','PG']) and all(g['passed'] for g in guards.values()) and all(cohort) else 'NO_ADOPTION',scenario=SCENARIO,production_adoption=False,reason='Fixed historical development acceptance gates; no deployment or financial certification',created_at_utc=_now())
        _save(roots['08_reporting']/'metrics.json',metrics); _save(roots['08_reporting']/'comparisons.json',comparisons); _save(roots['08_reporting']/'guards.json',guards); _save(roots['08_reporting']/'decision.json',decision)
    except Exception as exc:
        _save(roots['08_reporting']/'decision.json',dict(status='NO_ADOPTION',reason=str(exc),created_at_utc=_now(),primary_slots={'1':'NOT_RUN','2':'NOT_RUN'},retry_allowed=False))
        _save(roots['08_reporting']/'fit_receipts.json',dict(operation_trace=trace,receipts=receipts,estimation_operations=len(trace)))
        _oracle_index(run_id,repo,roots,binding,protocol,prepared,trace,'FAILED')
        raise
    _save(roots['08_reporting']/'fit_receipts.json',dict(operation_trace=trace,receipts=receipts,estimation_operations=len(trace)))
    _oracle_index(run_id,repo,roots,binding,protocol,prepared,trace,decision['status'])
    return decision

def _oracle_index(run_id,repo,roots,binding,protocol,prepared,trace,status):
    descriptors=[]
    public_path = roots['07_model_output']/'public_cases.json'
    public_cases = _json(public_path) if public_path.exists() else {'status':'NOT_RUN','reason':status,'selection_rule':protocol['public_case_selection'],'cases':[]}
    for stage,root in roots.items():
        if not root.exists(): continue
        for path in sorted(root.iterdir()):
            if path.name=='oracle_inputs.json' or not path.is_file(): continue
            columns=None; count=None
            if path.suffix=='.parquet':
                import pyarrow.parquet as pq
                metadata=pq.read_metadata(path); count=metadata.num_rows; columns=pq.read_schema(path).names
            elif path.suffix == '.npz':
                with np.load(path,allow_pickle=False) as arrays:
                    count = {}; columns = {}
                    for key in arrays.files:
                        array = arrays[key]
                        count[key] = list(array.shape)
                        columns[key] = str(array.dtype)
            descriptors.append(dict(role=path.stem,repo_relative_path=str(path.relative_to(repo)),sha256=sha256(path),format=path.suffix[1:],shape_or_row_count=count,dtype_or_columns=columns,units='minutes/density as column named',ordered_axis_names=['row'],ordered_id_axis_reference='04_feature/prepared.parquet:match_id',absence_status=None,absence_reason=None))
    represented = {d['role'] for d in descriptors}
    for role in ['raw_closure','raw_inventory','p0_parity','prepared','normalizer','series_ledger','memberships','membership_rules','train_design','duration_scale','receipt','duration_predictions','public_cases','metrics','comparisons','guards','decision','fit_receipts']:
        if role not in represented:
            descriptors.append(dict(role=role,repo_relative_path=None,sha256=None,format=None,shape_or_row_count=None,dtype_or_columns=None,units=None,ordered_axis_names=[],ordered_id_axis_reference=None,absence_status='NOT_RUN',absence_reason=status))
    for path, digest in binding['pins'].items():
        descriptors.append(dict(role='pinned_input',repo_relative_path=path,sha256=digest,format=Path(path).suffix[1:],shape_or_row_count=None,dtype_or_columns=None,units='inherited source units',ordered_axis_names=[],ordered_id_axis_reference=None,absence_status=None,absence_reason=None))
    for path,digest in binding.get('external_pins',{}).items():
        descriptors.append(dict(role='taxonomy_metadata',repo_relative_path=str(Path(path).relative_to(repo)) if Path(path).is_relative_to(repo) else None,external_absolute_path=path,sha256=digest,format=Path(path).suffix[1:],shape_or_row_count=None,dtype_or_columns=['golgg_match_id','unknown_players'],units='player count',ordered_axis_names=['canonical source row'],ordered_id_axis_reference='golgg_match_id',absence_status=None,absence_reason=None))
    _save(roots['08_reporting']/'oracle_inputs.json',dict(schema_version=1,study='P1',run_id=run_id,scenario=SCENARIO,branch_status=status,created_at_utc=_now(),protocol_sha256=sha256(roots['05_model_input']/'protocol.json'),binding_sha256=sha256(roots['05_model_input']/'binding.json'),oracle_contract_sha256=binding['pins'][protocol['evidence_root']+'/oracle_contract.json'],artifact_descriptors=descriptors,population_ids=prepared['population_ids'],operation_trace=trace,missingness_ledger={'all_canonical_rows_retained':status not in ['FAILED','INVALID_PREPARE'],'output_absence_is_hard_stop':True,'branch_reason':status},public_cases=public_cases,prefit_resolutions=protocol['prefit_resolutions']))

def predict_saved_row(run_id, match_id, *, repo_root, research_root):
    repo=Path(repo_root).resolve(); roots,binding,protocol=_validate(run_id,repo)
    prepared=_json(roots['05_model_input']/'prepare_receipt.json')
    if sha256(roots['04_feature']/'prepared.parquet')!=prepared['prepared_sha256']: raise ValueError('saved input origin hash mismatch')
    frame=pd.read_parquet(roots['04_feature']/'prepared.parquet'); rows=frame.loc[frame.match_id.astype(str).eq(str(match_id))]
    if len(rows)!=1 or rows.partition.iloc[0]=='EXCLUDED': raise ValueError('unique retained actual row required')
    partition_duration_rows(rows)
    receipt=_json(roots['06_models']/'receipt.json'); model=load_duration_scale(roots['06_models']/'duration_scale.joblib',expected_sha256=receipt['artifact_sha256'])
    arms={}
    for arm in ['P0','PG','PC']:
        pred=model.predict(rows,arm=arm); arms[arm]=dict(duration_log_mu=float(pred.duration_log_mu[0]),duration_log_sigma=float(pred.duration_log_sigma[0]),predictive_mean=float(np.exp(pred.duration_log_mu[0]+.5*pred.duration_log_sigma[0]**2)),p_over={str(line):float(pred.duration_over(line)[0]) for line in [25.5,28.5,31.5,34.5,37.5]})
    return dict(run_id=run_id,match_id=str(match_id),date=str(rows.date.iloc[0]),best_of=int(rows.best_of.iloc[0]),scenario=SCENARIO,artifact_sha256=receipt['artifact_sha256'],joint_distribution=False,arms=arms)
