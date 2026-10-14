"""Immutable exposed-development study changing only frozen C0 roster inputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy

from src.analysis import c0_betting_development as old
from src.analysis.c0_betting_development_evaluation import (
    evaluate_development, _fixed_selection, _probability, _value_summary)
from src.models.ev_approach_models import (
    c0_features, fit_c0_market_head, select_c0_market_head, _labels)
from src.models.c0_betting_price_head import price_features, fit_price_head

PROJECT_ROOT = old.PROJECT_ROOT
SCHEMA = 'c0_input_aligned_v1'
RECIPES = old.RECIPES
LAMBDAS = old.LAMBDAS
ORIGINAL = 'data/05_model_input/c0_betting_development/c0_betting_development_20261006_01/protocol.json'
MATRIX = 'data/08_reporting/a0_context_reliability/source_activation_20261004_01/matrix.json'
CODE = (*old.CODE, 'src/analysis/c0_input_aligned_study.py',
        'src/analysis/c0_input_aligned_data.py', 'src/analysis/c0_input_aligned_forward.py',
        'src/analysis/ev_asof_replay.py')
RETIREMENT = 'data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_01/integrity_retirement.json'
PREVIOUS_ORACLE = 'data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_01/independent/verify.py'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _stages(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_id):
        raise ValueError('Invalid run ID')
    return {number: PROJECT_ROOT/f'data/{name}/c0_input_aligned'/run_id
            for number, name in ((4,'04_feature'), (5,'05_model_input'),
                (6,'06_models'), (7,'07_model_output'), (8,'08_reporting'))}


def preflight_outputs(paths):
    """Inspect every destination before creating any parent or writing bytes."""
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f'Immutable output already exists: {path}')


def _reference(path):
    return {'root': 'repository', 'path': path, 'sha256': old._sha(PROJECT_ROOT/path)}


def freeze_input_aligned(run_id, research_root=None):
    stages = _stages(run_id)
    preflight_outputs([stages[5]/'protocol.json', stages[5]/'freeze_receipt.json',
                       stages[6], stages[7], stages[8]/'decision.json'])
    root = old._resolve_research_root(research_root)
    original = json.loads((PROJECT_ROOT/ORIGINAL).read_text())
    seal = json.loads((PROJECT_ROOT/ORIGINAL).with_name('freeze_receipt.json').read_text())
    if seal['protocol_file_sha256'] != old._sha(PROJECT_ROOT/ORIGINAL) or seal['protocol_sha256'] != old._canonical_sha(original):
        raise ValueError('Original protocol seal mismatch')
    for reference in original['sources'].values():
        if old._sha(old._source_path(reference, root)) != reference['sha256']:
            raise ValueError('Original historical input changed: '+reference['path'])
    for path, digest in original['preserved_forward_code_hashes'].items():
        if old._sha(PROJECT_ROOT/path) != digest:
            raise ValueError('Forward-pinned code changed: '+path)
    baseline_path = 'data/08_reporting/c0_input_aligned/c0_input_aligned_20261006_01/orchestration_baseline.json'
    baseline = json.loads((PROJECT_ROOT/baseline_path).read_text())
    for path, digest in baseline['files'].items():
        if old._sha(PROJECT_ROOT/path) != digest:
            raise ValueError('Immutable original changed: '+path)
    matrix = json.loads((PROJECT_ROOT/MATRIX).read_text())
    retirement = json.loads((PROJECT_ROOT/RETIREMENT).read_text())
    if (retirement.get('status') != 'RETIRED_BEFORE_FORMAL_FITS'
            or retirement.get('formal_head_fits') != 0 or retirement.get('base_fits') != 0
            or retirement.get('assessment_executed') is not False
            or retirement.get('hypothesis_changed') is not False
            or retirement.get('assessment_tuning') is not False):
        raise ValueError('Invalid zero-fit integrity retirement')
    protocol = deepcopy(original)
    protocol.update(schema=SCHEMA, run_id=run_id, frozen_at_actual_utc=_now(),
        hypothesis='Fit unchanged bounded heads on strictly prior-known exact-context roster inputs; all years development-exposed.',
        input_distribution='strict_prior_known_exact_context_five',
        expected_phase_denominators={'TRAIN':775,'CAL':370,'ASSESS':523},
        code_sha256={path:old._sha(PROJECT_ROOT/path) for path in CODE},
        immutable_originals=baseline['files'],
        forward_protocol_path=old.FORWARD, activation_matrix_path=MATRIX,
        source_contracts_path=matrix['source_contracts_path'])
    protocol['integrity_correction'] = {
        'retired_run_id':retirement['run_id'],
        'reason':'Prediction-before-update eligibility and exact common predictive coverage repaired before formal fits.',
        'hypothesis_changed':False, 'assessment_tuning':False,
        'original_market_complete_denominator':original['expected_population']['primary_common'],
        'prior_native_support':'Unknown at freeze; derived from corrected TRAIN/CAL inputs before fitting.',
        'coverage_gate_population':'Explicit primary forecast-eligible native-admitted valid p_C0/q/binary-label population, before head prediction.',
        'input_only_native_contrasts':'S/S-cal require valid matched native probabilities and labels, not q.'}
    protocol.pop('code_hashes', None)
    protocol['sources'].update(original_protocol=_reference(ORIGINAL),
        orchestration_baseline=_reference(baseline_path), activation_matrix=_reference(MATRIX),
        source_contracts=_reference(matrix['source_contracts_path']))
    protocol['sources'].update(integrity_retirement=_reference(RETIREMENT),
        previous_independent_oracle=_reference(PREVIOUS_ORACLE))
    protocol['sources']['raw_history'] = _reference(
        'data/artifacts/golgg-database-recovery-20260909/matches.json')
    for field in ('forward_protocol_path','activation_matrix_path','source_contracts_path'):
        protocol[field.removesuffix('_path')+'_sha256'] = old._sha(PROJECT_ROOT/protocol[field])
    # This subprocess only imports the owned repository modules; no native archive
    # import, generation, fitting, source request or DB access occurs at freeze.
    subprocess.run([sys.executable, '-c',
        'import src.analysis.c0_input_aligned_data; import src.analysis.c0_input_aligned_study; import src.analysis.c0_input_aligned_forward'],
        cwd=PROJECT_ROOT, check=True, timeout=60)
    old._save(stages[5]/'protocol.json', protocol)
    old._save(stages[5]/'freeze_receipt.json', {
        'protocol_file_sha256':old._sha(stages[5]/'protocol.json'),
        'protocol_sha256':old._canonical_sha(protocol), 'formal_head_fits':0,'base_fits':0})
    return stages[5]/'protocol.json'


def verify_protocol(protocol_path, research_root=None):
    path = Path(protocol_path).resolve()
    protocol = json.loads(path.read_text())
    seal = json.loads(path.with_name('freeze_receipt.json').read_text())
    if seal['protocol_file_sha256'] != old._sha(path) or seal['protocol_sha256'] != old._canonical_sha(protocol):
        raise ValueError('Input-aligned freeze seal mismatch')
    expected = {'schema':SCHEMA, 'recipes':list(RECIPES), 'lambdas':list(LAMBDAS),
        'formal_fit_budget':18,'base_fit_budget':0,'mode':'full',
        'native_identity':'Causal-C0:c0-native-2026-w32-e12-v1',
        'calendar':{key:list(value) for key,value in old.PHASES.items()},
        'max_quote_age_seconds':86400,'epsilon':1e-6}
    if any(protocol.get(key) != value for key,value in expected.items()):
        raise ValueError('Changed input-aligned study contract')
    original = json.loads((PROJECT_ROOT/ORIGINAL).read_text())
    for field in ('scope','statistics','selection','objective','tax','minimum_books',
                  'minimum_net_ev_exclusive','freshness_diagnostic_seconds',
                  'primary_candidate','primary_controls','centers','price_features','success'):
        if protocol.get(field) != original.get(field):
            raise ValueError('Changed original gate/policy: '+field)
    _stages(protocol['run_id'])
    root = old._resolve_research_root(research_root)
    for reference in protocol['sources'].values():
        if old._sha(old._source_path(reference,root)) != reference['sha256']:
            raise ValueError('Frozen source changed: '+reference['path'])
    for mapping in ('code_sha256','preserved_forward_code_hashes','immutable_originals'):
        for filename,digest in protocol[mapping].items():
            if old._sha(PROJECT_ROOT/filename) != digest:
                raise ValueError('Frozen code/original changed: '+filename)
    return protocol


def fit_membership(panel):
    """No assessment label is inspected while certifying TRAIN/CAL support."""
    required = {'row_id','phase','fit_eligible','market_eligible','native_admitted','financial_eligible'}
    if not required.issubset(panel) or panel.row_id.isna().any() or panel.row_id.duplicated().any():
        raise ValueError('Invalid panel membership')
    for column in ('fit_eligible','market_eligible','native_admitted','financial_eligible'):
        if not panel[column].map(lambda value:isinstance(value,(bool,np.bool_))).all():
            raise ValueError('Explicit membership booleans required: '+column)
    forecast_eligible = _forecast_eligibility(panel)
    eligible = panel.fit_eligible & panel.market_eligible & panel.native_admitted & forecast_eligible
    train = panel.loc[(panel.phase == 'TRAIN') & eligible].copy()
    cal = panel.loc[(panel.phase == 'CAL') & eligible].copy()
    status, reason = 'SUPPORTED', None
    try:
        for frame in (train,cal):
            y = _labels(frame)
            if not len(y) or len(np.unique(y)) != 2:
                raise ValueError('both_classes_required')
            for recipe in RECIPES:
                x = price_features(frame,recipe) if recipe.endswith('-price') else c0_features(frame,recipe)
                if frame is train and np.linalg.matrix_rank(x) != x.shape[1]:
                    raise ValueError('rank_deficient_'+recipe)
    except ValueError as error:
        status, reason = 'INSUFFICIENT_FIT_SUPPORT', str(error)
    receipt = {'TRAIN':train.row_id.tolist(),'CAL':cal.row_id.tolist(),
        'ASSESS':panel.loc[panel.phase=='ASSESS','row_id'].tolist(),
        'support_status':status,'support_reason':reason,
        'coverage':{'provider':len(panel),'financial_eligible':int(panel.financial_eligible.sum()),
            'market_only':int((panel.market_eligible & ~panel.native_admitted).sum()),
            'native_only':int((panel.native_admitted & ~panel.market_eligible).sum()),
            'native_absent':int((~panel.native_admitted).sum()),
            'phase_denominators':panel.groupby('phase').size().to_dict()}}
    return train,cal,receipt


def _forecast_eligibility(frame):
    """Missing flags are supported only for legacy synthetic unit fixtures."""
    eligible = frame.get('forecast_eligible', pd.Series(True, index=frame.index))
    if not eligible.map(lambda value:isinstance(value,(bool,np.bool_))).all():
        raise ValueError('forecast_eligible must contain explicit booleans')
    return eligible


class _ForecastEligibleHead:
    def __init__(self, head):
        self.head = head

    def predict_details(self, frame):
        if not _forecast_eligibility(frame).all():
            raise ValueError('target_already_released_before_cutoff')
        return self.head.predict_details(frame)


def _predictive_population(frame, *, market=True):
    primary = frame.get('primary_decision', pd.Series(True, index=frame.index))
    common = primary & _forecast_eligibility(frame) & frame.native_admitted
    for column in ('p_C0', 'y') + (('q',) if market else ()):
        common &= frame[column].map(_probability)
    return common & frame.y.isin([0,1])


def evaluate_input_aligned(panel, heads, draws=5000, seed=20261006):
    """Guard every recipe, retaining source inputs and the full mature cash universe."""
    _forecast_eligibility(panel)
    result = evaluate_development(panel,
        {recipe:_ForecastEligibleHead(head) for recipe,head in heads.items()}, draws, seed)
    by_id = panel.set_index('row_id')
    for row in result['rows']:
        row['forecast_eligible'] = bool(by_id.loc[row['row_id']].get('forecast_eligible', True))
        row['forecast_absence'] = by_id.loc[row['row_id']].get('forecast_absence', None)
        if not row['forecast_eligible']:
            row['absence'] = 'target_already_released_before_cutoff'
    for recipe in RECIPES:
        absences = [row['absence'] for row in result['rows']
                    if row['recipe']==recipe and row['absence'] is not None]
        result['metrics'][recipe]['absence_counts'] = pd.Series(absences, dtype=object).value_counts().to_dict()
    population_ids = panel.loc[_predictive_population(panel), 'row_id'].tolist()
    ledgers = {recipe:{row['row_id']:row for row in result['rows'] if row['recipe']==recipe}
               for recipe in RECIPES}
    result['common_predictive_coverage'] = {}
    for control, contrast in result['primary_contrasts'].items():
        candidate_count = sum(_probability(ledgers['H-price'][key]['probability']) for key in population_ids)
        control_count = sum(_probability(ledgers[control][key]['probability']) for key in population_ids)
        result['common_predictive_coverage'][control] = {
            'population_row_ids':population_ids, 'row_ids':contrast['row_ids'],
            'candidate_count':candidate_count, 'control_count':control_count}
        result['gates']['coverage_vs_'+control] = candidate_count >= control_count
    result['reasons'] = [name for name,passed in result['gates'].items() if not passed]
    result['decision'] = 'DEVELOPMENT_CANDIDATE_RETAINED' if not result['reasons'] else 'NOT_ADOPTED'
    result['estimands']['coverage_gate'] = 'explicit_pre_prediction_primary_native_market_label_common_population'
    return result


def _assessment(frame, heads, protocol):
    frame = frame.loc[frame.phase=='ASSESS'].copy().reset_index(drop=True)
    frame.attrs['monthly_universe'] = sorted(frame.month.unique())
    return frame,evaluate_input_aligned(frame,heads,draws=protocol['statistics']['bootstrap_draws'],
                                       seed=protocol['statistics']['seed'])


def _report(result):
    return {key:value for key,value in result.items() if key!='rows'}


def _input_contrast(frame, candidate, baseline, recipe, months, resamples):
    """Consumer-valid exact native pairs; only sports-only recipes omit q."""
    observations = []
    for _,row in frame.loc[_predictive_population(frame, market=recipe not in {'S','S-cal'})].iterrows():
        key = row.row_id
        p, q = candidate[key]['probability'], baseline[key]['probability']
        if not _probability(p) or not _probability(q):
            continue
        y = float(row.y)
        loss = float(-(xlogy(y,p)+xlogy(1-y,1-p)) + xlogy(y,q)+xlogy(1-y,1-q))
        observations.append((key,row.month,loss,float((p-y)**2-(q-y)**2)))
    ll = _value_summary([(month,loss) for _,month,loss,_ in observations], months,resamples,.95)
    brier = _value_summary([(month,loss) for _,month,_,loss in observations], months,resamples,.95)
    return {'candidate':'actual_lineup/'+recipe, 'control':'prior_known/'+recipe,
        'delta_LL':ll['point'], 'LL_interval':ll['interval'],
        'delta_Brier':brier['point'], 'Brier_interval':brier['interval'],
        'equal_month_delta_LL':ll['equal_month_point'], 'equal_month_LL_interval':ll['equal_month_interval'],
        'equal_month_delta_Brier':brier['equal_month_point'], 'equal_month_Brier_interval':brier['equal_month_interval'],
        'common_count':len(observations), 'row_ids':[key for key,_,_,_ in observations],
        'contributing_blocks':ll['contributing_blocks'], 'undefined_draws':ll['undefined_draws'],
        'equal_month_undefined_draws':ll['equal_month_undefined_draws'],
        'confidence_level':.95, 'status':ll['status'], 'LL':ll, 'Brier':brier}


def _input_diagnostic(prior, actual, heads, protocol, full_prior_result):
    indexed = actual.set_index('row_id')
    if not prior.row_id.is_unique or not actual.row_id.is_unique or set(prior.row_id)!=set(actual.row_id):
        raise ValueError('Input-only diagnostic opportunity identities changed')
    for column in ('phase','month','q','odds_a','odds_b','y','quotes','financial_eligible'):
        # Serialized nested quote arrays are compared canonically, not by ndarray equality.
        for _,row in prior.iterrows():
            if old._canonical_sha(row[column]) != old._canonical_sha(indexed.loc[row.row_id,column]):
                raise ValueError('Input-only diagnostic changed '+column)
    for column in ('forecast_eligible','forecast_absence','primary_decision'):
        if (column in prior) != (column in actual):
            raise ValueError('Input-only diagnostic changed '+column)
        if column in prior:
            for _,row in prior.iterrows():
                if old._canonical_sha(row[column]) != old._canonical_sha(indexed.loc[row.row_id,column]):
                    raise ValueError('Input-only diagnostic changed '+column)
    actual_valid = _predictive_population(indexed, market=False)
    common = _predictive_population(prior, market=False) & prior.row_id.map(actual_valid)
    a = prior.loc[common].copy()
    if 'primary_decision' not in a:
        a['primary_decision'] = True
    b = actual.set_index('row_id').loc[a.row_id].reset_index()
    months = sorted(prior.month.unique())
    a.attrs['monthly_universe'] = b.attrs['monthly_universe'] = months
    kwargs = dict(draws=protocol['statistics']['bootstrap_draws'],seed=protocol['statistics']['seed'])
    p_result = evaluate_input_aligned(a,heads,**kwargs)
    a_result = evaluate_input_aligned(b,heads,**kwargs)
    # Reuse locked contrast/bootstrap arithmetic, relabeling only the two input scenarios.
    resamples = np.random.default_rng(kwargs['seed']).integers(0,len(months),(kwargs['draws'],len(months)))
    p_ledgers = {recipe:{row['row_id']:row for row in p_result['rows'] if row['recipe']==recipe} for recipe in RECIPES}
    a_ledgers = {recipe:{row['row_id']:row for row in a_result['rows'] if row['recipe']==recipe} for recipe in RECIPES}
    contrasts = {}
    for recipe in RECIPES:
        contrasts[recipe] = _input_contrast(a,a_ledgers[recipe],p_ledgers[recipe],
                                            recipe,months,resamples)
    fixed = full_prior_result['own_selections']['S']
    _, full_actual = _assessment(actual,heads,protocol)
    full_ledgers = {recipe:{row['row_id']:row for row in full_actual['rows'] if row['recipe']==recipe} for recipe in RECIPES}
    return {'classification':'INPUT_ONLY_DIAGNOSTIC_NO_FIT_NO_SELECTION',
        'common_row_ids':a.row_id.tolist(),'prior_native_absent':int((~prior.native_admitted).sum()),
        'actual_native_absent':int((~actual.native_admitted).sum()),
        'prior':_report(p_result),'actual':_report(a_result),'paired_input_contrasts':contrasts,
        'fixed_prior_S':fixed,
        'actual_on_fixed_prior_S':{recipe:_fixed_selection(fixed,actual.set_index('row_id'),
            full_ledgers[recipe],months,resamples) for recipe in RECIPES},
        'own_policy_actual':_report(full_actual)}


def run_input_aligned(protocol_path, research_root=None):
    protocol = verify_protocol(protocol_path,research_root)
    stages = _stages(protocol['run_id'])
    preparation = json.loads((stages[5]/'preparation_receipt.json').read_text())
    if preparation.get('protocol_sha256') != old._canonical_sha(protocol):
        raise ValueError('Preparation protocol seal mismatch')
    if preparation.get('evidence_class') == 'SOFTWARE_SMOKE_NOT_FORMAL' or '_smoke_' in protocol['run_id']:
        raise ValueError('Smoke preparation cannot enter formal fitting')
    for filename,seal in preparation['outputs'].items():
        path = PROJECT_ROOT/filename
        digest = seal['sha256'] if isinstance(seal,dict) else seal
        if old._sha(path)!=digest or (isinstance(seal,dict) and path.stat().st_size!=seal['bytes']):
            raise ValueError('Preparation output changed: '+filename)
    outputs = [stages[6],stages[7],stages[5]/'fit_membership.json']
    outputs += [stages[8]/name for name in ('assessment.json','fresh300s_diagnostic.json',
        'actual_lineup_diagnostic.json','decision.json','execution_receipt.json')]
    preflight_outputs(outputs)
    required_seals = {str((stages[5]/name).relative_to(PROJECT_ROOT)) for name in (
        'prior_panel.parquet','actual_lineup_panel.parquet','fresh300s_prior_panel.parquet')}
    if not required_seals.issubset(preparation['outputs']):
        raise ValueError('Preparation does not seal every consumed panel')
    prior = pd.read_parquet(stages[5]/'prior_panel.parquet')
    actual = pd.read_parquet(stages[5]/'actual_lineup_panel.parquet')
    fresh = pd.read_parquet(stages[5]/'fresh300s_prior_panel.parquet')
    if len(prior)!=1668 or prior.groupby('phase').size().to_dict()!=protocol['expected_phase_denominators']:
        raise ValueError('Frozen provider phase denominators changed')
    if actual.row_id.tolist()!=prior.row_id.tolist() or fresh.row_id.tolist()!=prior.row_id.tolist():
        raise ValueError('Diagnostic opportunity identity/order changed')
    for frame in (prior,actual,fresh):
        if 'forecast_eligible' not in frame or 'forecast_absence' not in frame:
            raise ValueError('Real panels require explicit forecast eligibility and absence')
        _forecast_eligibility(frame)
    for diagnostic in (actual,fresh):
        if diagnostic.forecast_eligible.tolist()!=prior.forecast_eligible.tolist():
            raise ValueError('Diagnostic forecast eligibility changed')
    train,cal,membership = fit_membership(prior)
    bindings = {'protocol_sha256':old._canonical_sha(protocol),'protocol_file_sha256':old._sha(protocol_path),
        'train_row_ids_sha256':old._canonical_sha(train.row_id.tolist()),
        'cal_row_ids_sha256':old._canonical_sha(cal.row_id.tolist()),
        'sources':{key:value['sha256'] for key,value in protocol['sources'].items()},
        'code_sha256':protocol['code_sha256'],'preparation_sha256':old._sha(stages[5]/'preparation_receipt.json'),
        'actual_fit_started_at_utc':_now(),'scope':'exposed_input_aligned_development_only'}
    membership['bindings'] = bindings
    old._save(stages[5]/'fit_membership.json',membership)
    fit_count,attempts,heads,selections = 0,[],{},{}
    status = membership['support_status']
    if status=='SUPPORTED':
        train.attrs['c0_bindings']=bindings
        stages[6].mkdir(parents=True,exist_ok=False)
        try:
            for recipe in RECIPES:
                if recipe in {'S','M','Fixed50'}:
                    head=fit_c0_market_head(recipe,train,regularization=0.)
                    selection={'fixed':True,'lambda':0.}
                else:
                    candidates=[]
                    for regularization in LAMBDAS:
                        attempt={'recipe':recipe,'lambda':regularization,'started_at':_now()}
                        attempts.append(attempt)
                        fitter=fit_price_head if recipe.endswith('-price') else fit_c0_market_head
                        candidate=fitter(recipe,train,regularization=regularization)
                        fit_count+=1
                        attempt.update(success=True,optimizer=candidate.optimizer)
                        old._save(stages[6]/f'{recipe}_lambda_{regularization:g}.json',candidate.to_dict())
                        candidates.append(candidate)
                    head,selection=select_c0_market_head(candidates,cal)
                heads[recipe]=head
                selections[recipe]=selection
                old._save(stages[6]/f'{recipe}_selected.json',head.to_dict())
        except RuntimeError as error:
            attempts[-1].update(success=False,error=str(error))
            status='OPTIMIZER_FAILED'
        if status=='SUPPORTED':
            if fit_count!=18 or len(heads)!=9:
                raise RuntimeError('Wrong frozen fit/recipe budget')
            old._save(stages[6]/'selection.json',{'selection':selections,'formal_head_fits':fit_count,
                'base_fits':0,'bindings':bindings,'actual_selected_at_utc':_now()})
            assess,result=_assessment(prior,heads,protocol)
            fresh_assess,diagnostic=_assessment(fresh,heads,protocol)
            stages[7].mkdir(parents=True,exist_ok=False)
            pd.DataFrame(result['rows']).to_parquet(stages[7]/'assessment_predictions_and_ledger.parquet',index=False)
            pd.DataFrame(diagnostic['rows']).to_parquet(stages[7]/'fresh300s_diagnostic_predictions_and_ledger.parquet',index=False)
            old._save(stages[8]/'assessment.json',_report(result))
            old._save(stages[8]/'fresh300s_diagnostic.json',{'classification':'TRANSFER_DIAGNOSTIC_NO_FIT_NO_SELECTION','result':_report(diagnostic)})
            old._save(stages[8]/'actual_lineup_diagnostic.json',_input_diagnostic(assess,
                actual.loc[actual.phase=='ASSESS'].copy(),heads,protocol,result))
            status='COMPLETED'
    decision={'schema':SCHEMA,'run_id':protocol['run_id'],'status':status,
        'decision':result['decision'] if status=='COMPLETED' else status,
        'gates':result['gates'] if status=='COMPLETED' else None,
        'formal_head_fits':fit_count,'fit_attempt_count':len(attempts),'base_fits':0,
        'selected_recipe_count':len(heads) if status=='COMPLETED' else 0,
        'coverage':membership['coverage'],'bindings':bindings,'optimizer_attempts':attempts,
        'actual_completed_at_utc':_now(),'scope':'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY',
        'deployed':False,'canonical_models_changed':False,'live_quote_requests':0,
        'limitations':protocol['limitations']}
    old._save(stages[8]/'decision.json',decision)
    artifacts={str(path.relative_to(PROJECT_ROOT)):{'sha256':old._sha(path),'bytes':path.stat().st_size}
        for stage in stages.values() if stage.exists() for path in sorted(stage.iterdir()) if path.is_file()}
    old._save(stages[8]/'execution_receipt.json',{'decision':decision,'artifacts':artifacts,
        'preserved_forward_code_hashes':protocol['preserved_forward_code_hashes'],
        'immutable_originals':protocol['immutable_originals']})
    return decision


def prepare_input_aligned(protocol, *, research_root=None, smoke_limit=None):
    from src.analysis.c0_input_aligned_data import prepare_input_aligned as prepare
    return prepare(protocol,research_root=research_root,smoke_limit=smoke_limit)


def inspect_forward_readiness(protocol_path, *, output_dir, now=None):
    from src.analysis.c0_input_aligned_forward import inspect_forward_readiness as inspect
    return inspect(protocol_path,output_dir=output_dir,now=now)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='approach-comparison c0-input-aligned-study')
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('freeze','prepare','run','readiness'):
        actions.add_argument('--'+action,action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--research-root',type=Path)
    parser.add_argument('--smoke-limit',type=int)
    args=parser.parse_args(argv)
    if args.freeze:
        if not args.run_id or args.protocol or args.smoke_limit is not None:
            parser.error('freeze requires only run-id/research-root')
        result=freeze_input_aligned(args.run_id,args.research_root)
    else:
        if args.protocol is None or args.run_id or (args.smoke_limit is not None and not args.prepare):
            parser.error('protocol required; run-id freeze only; smoke-limit prepare only')
        protocol=verify_protocol(args.protocol,args.research_root)
        if args.prepare:
            result=prepare_input_aligned(protocol,research_root=args.research_root,smoke_limit=args.smoke_limit)
        elif args.run:
            result=run_input_aligned(args.protocol,args.research_root)
        else:
            result=inspect_forward_readiness(args.protocol,output_dir=_stages(protocol['run_id'])[8]/'forward_readiness')
    print(json.dumps(old._json_safe(result),sort_keys=True,default=str))
    return result
