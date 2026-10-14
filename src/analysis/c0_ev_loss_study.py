"""One sealed loss-only historical study; never native training or deployment.

Return-MSE composed with a sigmoid is nonconvex. A converged bounded local
optimizer is not a certificate of global optimality. Software smoke has its own
stage destinations and cannot write formal models or historical conclusions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.special import expit

from src.analysis import c0_betting_development as old
from src.analysis import c0_input_aligned_study as aligned
from src.analysis.c0_ev_loss_evaluation import evaluate_ev_loss
from src.models.c0_betting_price_head import price_features
from src.models.c0_ev_loss_head import C0EVLossHead, fit_ev_loss_head, select_ev_loss_head
from src.models.ev_approach_models import C0MarketHead, _labels

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT
DEFAULT_SOURCE_PROTOCOL = 'data/05_model_input/c0_input_aligned/c0_input_aligned_20261006_02/protocol.json'
SOURCE_PROTOCOL = DEFAULT_SOURCE_PROTOCOL
SCHEMA = 'c0_ev_loss_study_v1'
PLAN = 'TODO/10_c0_ev_loss_objective.md'
BASELINE = 'data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/orchestration_baseline.json'
PREFIT = 'data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/pre_fit_source_verification.json'
DISPATCHER = 'src/analysis/ev_approach_runner.py'
OLD_DISPATCHER = 'data/08_reporting/c0_ev_loss/c0_ev_loss_20261006_01/pre_extension_source/' + DISPATCHER
REFERENCES = ('S', 'S-cal', 'M', 'M-cal', 'M-shape', 'Fixed50', 'H')
ARMS = {'H-likelihood': 'H-price', 'H-return_mse': 'H-price',
        'M-likelihood': 'M-price', 'M-return_mse': 'M-price'}
LAMBDAS = (.01, .1, 1.)
CODE = tuple(dict.fromkeys((*aligned.CODE, 'src/models/ev_approach_models.py',
    'src/analysis/ev_approach_evaluation.py', 'src/analysis/ev_approach_policies.py',
    'src/analysis/devig_comparison.py',
    'src/analysis/c0_ev_loss_study.py', 'src/analysis/c0_ev_loss_evaluation.py',
    'src/models/c0_ev_loss_head.py', 'scripts/run_unified_betting_evaluation.py')))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return old._sha(Path(path))


def _load(path):
    return json.loads(Path(path).read_text())


def _stages(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', run_id):
        raise ValueError('Invalid immutable run ID')
    return {number: PROJECT_ROOT/'data'/directory/'c0_ev_loss'/run_id
            for number, directory in ((5, '05_model_input'), (6, '06_models'),
                                      (7, '07_model_output'), (8, '08_reporting'))}


def preflight_outputs(paths):
    """Check all destinations without creating any parent or writing bytes."""
    for path in paths:
        if Path(path).exists() or Path(path).is_symlink():
            raise FileExistsError(f'Immutable output already exists: {path}')


def _write_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content)


def _research_root():
    return old._resolve_research_root(None)


def _source_path(ref):
    if ref['root'] == 'repository':
        root = PROJECT_ROOT
    elif ref['root'] == 'code':
        root = CODE_ROOT
    elif ref['root'] == 'research':
        root = _research_root()
    else:
        raise ValueError('Unknown source root')
    path = (root/ref['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Source escapes declared root')
    return path


def _reference(path, *, root='repository'):
    ref = {'root': root, 'path': str(path)}
    actual = _source_path(ref)
    return {**ref, 'sha256': _sha(actual), 'bytes': actual.stat().st_size}


def _verify_ref(ref):
    path = _source_path(ref)
    if (_sha(path) != ref['sha256'] or
            ('bytes' in ref and path.stat().st_size != ref['bytes'])):
        raise ValueError('Frozen source hash/seal changed: ' + ref['path'])


def _artifact_refs(value):
    if isinstance(value, dict):
        return [dict(ref, path=ref.get('path', name), root=ref.get('root', 'repository'))
                for name, ref in value.items()]
    return [dict(ref, root=ref.get('root', 'repository')) for ref in value]


def _read_source():
    """Verify retained inputs, not the obsolete dispatcher code identity."""
    path = (PROJECT_ROOT/SOURCE_PROTOCOL).resolve()
    source = _load(path)
    seal = _load(path.with_name('freeze_receipt.json'))
    if (seal['protocol_file_sha256'] != _sha(path) or
            seal['protocol_sha256'] != old._canonical_sha(source)):
        raise ValueError('Previous protocol seal mismatch')
    preparation = _load(path.with_name('preparation_receipt.json'))
    if (preparation['protocol_sha256'] != old._canonical_sha(source) or
            preparation.get('evidence_class') != 'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY'):
        raise ValueError('Previous preparation seal/class mismatch')
    if (source.get('schema') != 'c0_input_aligned_v1' or source.get('mode') != 'full' or
            source.get('native_identity') != 'Causal-C0:c0-native-2026-w32-e12-v1' or
            source.get('tax') != .12 or source.get('epsilon') != 1e-6 or
            source.get('minimum_net_ev_exclusive') != .05):
        raise ValueError('Previous native/policy contract changed')
    previous_stages = {number: PROJECT_ROOT/'data'/directory/'c0_input_aligned'/source['run_id']
        for number, directory in ((5, '05_model_input'), (6, '06_models'), (8, '08_reporting'))}
    verification_path = previous_stages[8]/'sealed_verification.json'
    verification = _load(verification_path)
    strict = str(SOURCE_PROTOCOL) == DEFAULT_SOURCE_PROTOCOL
    if strict:
        if (source['expected_phase_denominators'] != {'TRAIN':775, 'CAL':370, 'ASSESS':523} or
                verification.get('prior_input_population') != {
                    'provider':1668, 'native':1351, 'roster_absences':312, 'known_result_prohibitions':5} or
                verification.get('verification', {}).get('independent', {}).get('status') != 'PASS'):
            raise ValueError('Previous population/independent PASS changed')
    elif verification.get('status') != 'PASS':
        raise ValueError('Injected source must have a PASS seal')
    refs = dict(source.get('sources', {}))
    for key, name in (('source_protocol', 'protocol.json'), ('source_freeze', 'freeze_receipt.json'),
                      ('preparation', 'preparation_receipt.json'), ('prior_panel', 'prior_panel.parquet'),
                      ('actual_lineup_panel', 'actual_lineup_panel.parquet'),
                      ('fresh300s_prior_panel', 'fresh300s_prior_panel.parquet')):
        filename = str((previous_stages[5]/name).relative_to(PROJECT_ROOT))
        refs[key] = _reference(filename)
        if name.endswith('.parquet') and filename not in preparation['outputs']:
            raise ValueError('Preparation does not seal consumed panel: ' + filename)
    refs['source_verification'] = _reference(str(verification_path.relative_to(PROJECT_ROOT)))
    for recipe in REFERENCES:
        refs['reference_'+recipe] = _reference(str((previous_stages[6]/f'{recipe}_selected.json').relative_to(PROJECT_ROOT)))
        head = C0MarketHead.from_dict(_load(_source_path(refs['reference_'+recipe])))
        if head.recipe_id != recipe:
            raise ValueError('Previous selected reference recipe changed')
    for ref in _artifact_refs(preparation['outputs']):
        _verify_ref(ref)
    for ref in _artifact_refs(verification['artifacts']):
        _verify_ref(ref)
    originals = dict(source.get('immutable_originals', {}))
    prior_artifacts = []
    if strict:
        baseline = _load(PROJECT_ROOT/BASELINE)
        prefit = _load(PROJECT_ROOT/PREFIT)
        if (len(baseline['originals']) != 17 or len(baseline['prior_run_artifacts']) != 71 or
                prefit.get('status') != 'PASS' or prefit.get('originals') != baseline['originals']):
            raise ValueError('Original17/prior71 baseline PASS seal changed')
        if _sha(CODE_ROOT/PLAN) != baseline['plan_sha256']:
            raise ValueError('Authorized plan bytes changed before freeze')
        originals = baseline['originals']
        prior_artifacts = _artifact_refs(baseline['prior_run_artifacts'])
        refs.update(orchestration_baseline=_reference(BASELINE), pre_fit_source_verification=_reference(PREFIT),
                    preserved_dispatcher=_reference(OLD_DISPATCHER))
        if refs['preserved_dispatcher']['sha256'] != source['code_sha256'][DISPATCHER]:
            raise ValueError('Previous dispatcher snapshot differs from previous sealed code')
        native_path = str((previous_stages[8]/'independent/receipt_corrected_02_native.json').relative_to(PROJECT_ROOT))
        refs['independent_native'] = _reference(native_path)
        native = _load(PROJECT_ROOT/native_path)
        if native.get('failures') != [] or native.get('checked_counts', {}).get('native', 0) <= 0:
            raise ValueError('Independent native replay did not pass')
    for ref in prior_artifacts:
        _verify_ref(ref)
    for filename, digest in originals.items():
        _verify_ref({'root':'repository', 'path':filename, 'sha256':digest})
    for filename, digest in source.get('preserved_forward_code_hashes', {}).items():
        _verify_ref({'root':'repository', 'path':filename, 'sha256':digest})
    for filename, digest in source.get('code_sha256', {}).items():
        if filename == DISPATCHER and strict:
            continue  # Authorized routing has a separately preserved old snapshot.
        _verify_ref({'root':'repository', 'path':filename, 'sha256':digest})
    for ref in refs.values():
        _verify_ref(ref)
    return source, refs, originals, prior_artifacts


def freeze_ev_loss(run_id):
    stages = _stages(run_id)
    preflight_outputs([stages[5], stages[6], stages[7], *[stages[8]/name for name in
        ('assessment.json', 'decision.json', 'execution_receipt.json',
         'fresh300s_diagnostic.json', 'actual_lineup_diagnostic.json')]])
    source, refs, originals, artifacts = _read_source()
    code_hashes = {filename:_sha(CODE_ROOT/filename) for filename in CODE}
    refs['authorized_plan'] = _reference(PLAN, root='code')
    protocol = {'schema':SCHEMA, 'run_id':run_id, 'frozen_at_actual_utc':_now(),
        'scope':'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY', 'evidence_class':'development',
        'native_identity':source['native_identity'], 'mode':'full',
        'source_run_id':source['run_id'], 'source_protocol_sha256':old._canonical_sha(source),
        'sources':refs, 'immutable_originals':originals, 'prior_run_artifacts':artifacts,
        'preserved_forward_code_hashes':source.get('preserved_forward_code_hashes', {}),
        'previous_code_sha256':source.get('code_sha256', {}), 'code_sha256':code_hashes,
        'calendar':deepcopy(source['calendar']),
        'expected_phase_denominators':source['expected_phase_denominators'],
        'expected_common_membership':{'TRAIN':636, 'CAL':299, 'ASSESS':382}
            if str(SOURCE_PROTOCOL) == DEFAULT_SOURCE_PROTOCOL else None,
        'expected_native_population':1351 if str(SOURCE_PROTOCOL) == DEFAULT_SOURCE_PROTOCOL else None,
        'arms':ARMS, 'reference_recipes':list(REFERENCES), 'lambdas':list(LAMBDAS),
        'formal_fit_budget':12, 'base_fit_budget':0, 'cal_refit_budget':0,
        'selection':'two_sided_mean_scoring_CAL_return_mse_largest_lambda_within_1e-12_no_refit',
        'tax':.12, 'net_multiplier':.88, 'epsilon':1e-6, 'minimum_net_ev_exclusive':.05,
        'centers':{'M-price':[1.,1.,0.], 'H-price':[1.,1.,0.,0.]},
        'bounds':{'M-price':[[0.,None],[0.,None],[None,None]],
                  'H-price':[[0.,None],[0.,None],[None,None],[0.,None]]},
        'price_features':deepcopy(source.get('price_features', {
            'transform':'odd_logit_fixed_knot_0.8_plus_signed_log_execution_price_ratio'})),
        'objectives':{'likelihood':'mean_raw_logit_bernoulli_NLL_plus_lambda/2_centered_squared_L2',
                      'return_mse':'mean_0.5*0.88^2*(odds_a^2+odds_b^2)*(raw_p-y)^2_plus_lambda/2_centered_squared_L2'},
        'optimizer':{'method':'L-BFGS-B', 'initialization':'declared_center', 'maxiter':2000,
                     'ftol':1e-12, 'gtol':1e-8, 'maxls':50, 'retries':0},
        'statistics':{**source['statistics'], 'risk_contrasts':3,
                      'risk_interval_level':1-.05/3, 'block':'calendar_month'},
        'policy':'best_aligned_price_per_side_raw_net_EV>0.05_tie_A_flat_unit_one_bet_fixture',
        'fit_clock':'actual_execution_not_nominal_historical_origin',
        'limitations':[*source.get('limitations', []),
            'Sigmoid return-MSE is nonconvex; convergence/local independent agreement is not global optimality.',
            'Native C0 is reused unchanged; all 2026 is development-exposed; no authentic issuance or execution.',
            'Actual-lineup and fresh300s are same-head no-fit/no-selection diagnostics.'],
        'deployment_authorized':False}
    _write_bytes(stages[5]/'plan_snapshot.md', (CODE_ROOT/PLAN).read_bytes())
    for filename in CODE:
        _write_bytes(stages[5]/'code_snapshot'/filename, (CODE_ROOT/filename).read_bytes())
    protocol['snapshots'] = {
        'plan':_reference(str((stages[5]/'plan_snapshot.md').relative_to(PROJECT_ROOT))),
        'code':{filename:_reference(str((stages[5]/'code_snapshot'/filename).relative_to(PROJECT_ROOT)))
                for filename in CODE}}
    protocol['authorization_plan_origin'] = dict(refs['authorized_plan'])
    # Subsequent execution-status prose may change the working TODO, never this plan.
    refs['authorized_plan'] = dict(protocol['snapshots']['plan'])
    old._save(stages[5]/'protocol.json', protocol)
    old._save(stages[5]/'freeze_receipt.json', {'protocol_file_sha256':_sha(stages[5]/'protocol.json'),
        'protocol_sha256':old._canonical_sha(protocol), 'formal_head_fits':0, 'synthetic_head_fits':0,
        'base_fits':0, 'actual_frozen_at_utc':_now()})
    return stages[5]/'protocol.json'


def _verify_protocol(protocol_path):
    path = Path(protocol_path).resolve()
    protocol, seal = _load(path), _load(path.with_name('freeze_receipt.json'))
    if (seal['protocol_file_sha256'] != _sha(path) or
            seal['protocol_sha256'] != old._canonical_sha(protocol)):
        raise ValueError('EV-loss protocol freeze seal mismatch')
    expected = {'schema':SCHEMA, 'formal_fit_budget':12, 'base_fit_budget':0,
        'cal_refit_budget':0, 'arms':ARMS, 'lambdas':list(LAMBDAS), 'reference_recipes':list(REFERENCES),
        'tax':.12, 'net_multiplier':.88, 'epsilon':1e-6, 'mode':'full',
        'native_identity':'Causal-C0:c0-native-2026-w32-e12-v1'}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError('Changed EV-loss objective/fit budget contract')
    stages = _stages(protocol['run_id'])
    if path != (stages[5]/'protocol.json').resolve():
        raise ValueError('Protocol is not in its bound stage05 destination')
    for ref in protocol['sources'].values():
        _verify_ref(ref)
    for ref in protocol['prior_run_artifacts']:
        _verify_ref(ref)
    for mapping in ('immutable_originals', 'preserved_forward_code_hashes'):
        for filename, digest in protocol[mapping].items():
            _verify_ref({'root':'repository', 'path':filename, 'sha256':digest})
    for filename, digest in protocol['code_sha256'].items():
        _verify_ref({'root':'code', 'path':filename, 'sha256':digest})
    _verify_ref(protocol['snapshots']['plan'])
    for filename, ref in protocol['snapshots']['code'].items():
        _verify_ref(ref)
        if ref['sha256'] != protocol['code_sha256'][filename]:
            raise ValueError('Code snapshot differs from frozen implementation')
    source = _load(_source_path(protocol['sources']['source_protocol']))
    if old._canonical_sha(source) != protocol['source_protocol_sha256']:
        raise ValueError('Previous protocol lineage changed')
    return protocol


def fit_membership(panel):
    """Common TRAIN/CAL support; assessment outcomes are never support inputs."""
    required = {'row_id','phase','fit_eligible','market_eligible','native_admitted','financial_eligible',
                'forecast_eligible'}
    if (not required.issubset(panel) or panel.row_id.isna().any() or
            panel.row_id.duplicated().any() or not panel.phase.isin(['TRAIN','CAL','ASSESS']).all()):
        raise ValueError('Invalid EV-loss panel membership')
    for name in required - {'row_id','phase'}:
        if not panel[name].map(lambda v:isinstance(v,(bool,np.bool_))).all():
            raise ValueError('Explicit membership booleans required: '+name)
    eligible = panel.fit_eligible & panel.market_eligible & panel.native_admitted & panel.forecast_eligible
    if 'primary_decision' in panel:
        if not panel.primary_decision.map(lambda v:isinstance(v,(bool,np.bool_))).all():
            raise ValueError('Explicit primary decision boolean required')
        eligible &= panel.primary_decision
    train = panel.loc[(panel.phase == 'TRAIN') & eligible].copy()
    cal = panel.loc[(panel.phase == 'CAL') & eligible].copy()
    status, reason = 'SUPPORTED', None
    try:
        for frame in (train, cal):
            labels = _labels(frame)
            if len(labels) == 0 or len(np.unique(labels)) != 2:
                raise ValueError('both_classes_required')
            for recipe in ('H-price', 'M-price'):
                x = price_features(frame, recipe)
                weights = .5*.88**2*(frame.odds_a.to_numpy(dtype=float)**2 + frame.odds_b.to_numpy(dtype=float)**2)
                if not np.isfinite(weights).all():
                    raise ValueError('nonfinite_return_weights')
                if frame is train and np.linalg.matrix_rank(x) != x.shape[1]:
                    raise ValueError('rank_deficient_'+recipe)
    except ValueError as error:
        status, reason = 'INSUFFICIENT_FIT_SUPPORT', str(error)
    return train, cal, {'TRAIN':train.row_id.tolist(), 'CAL':cal.row_id.tolist(),
        'ASSESS':panel.loc[panel.phase == 'ASSESS','row_id'].tolist(),
        'support_status':status, 'support_reason':reason,
        'coverage':{'provider':len(panel), 'financial_eligible':int(panel.financial_eligible.sum()),
            'phase_denominators':panel.groupby('phase').size().to_dict(),
            'native_absent':int((~panel.native_admitted).sum()),
            'market_only':int((panel.market_eligible & ~panel.native_admitted).sum()),
            'native_only':int((panel.native_admitted & ~panel.market_eligible).sum())}}


def _synthetic_panel():
    rng = np.random.default_rng(20261006)
    rows = []
    for phase in ('TRAIN','CAL','ASSESS'):
        start, end = (pd.Timestamp(day, tz='UTC') for day in old.PHASES[phase])
        for i in range(200):
            z, m = rng.normal(0, 1.8, 2)
            oa, ob = rng.uniform(1.4, 4.5, 2)
            origin = start + (end-start)*(i/200)
            rows.append({'row_id':f'synthetic-{phase}-{i}', 'phase':phase, 'month':origin.strftime('%Y-%m'),
                'evaluation_cutoff_at':origin.isoformat(),
                'p_C0':float(expit(z)), 'q':float(expit(m)), 'odds_a':float(oa), 'odds_b':float(ob),
                'y':i % 2, 'primary_decision':True, 'fit_eligible':True, 'market_eligible':True,
                'native_admitted':True, 'forecast_eligible':True, 'forecast_absence':None,
                'financial_eligible':True, 'source_class':'SYNTHETIC_SOFTWARE_ONLY',
                'quotes':[{'bookmaker':'synthetic', 'quote_id':f'{phase}-{i}',
                           'odds_a':float(oa), 'odds_b':float(ob)}]})
    return pd.DataFrame(rows)


def _assessment(frame):
    result = frame.loc[frame.phase == 'ASSESS'].copy().reset_index(drop=True)
    result.attrs['monthly_universe'] = sorted(result.month.unique())
    return result


def _report(result):
    return {key:({name:{k:v for k,v in arm.items() if k != 'rows'}
                        for name,arm in value.items()} if key == 'arms' else value)
            for key,value in result.items() if key != 'rows'}


def _destinations(stages):
    return [stages[6], stages[7], stages[5]/'fit_membership.json', *[
        stages[8]/name for name in ('assessment.json','fresh300s_diagnostic.json',
        'actual_lineup_diagnostic.json','decision.json','execution_receipt.json','fit_attempts.json')]]


def run_ev_loss(protocol_path, *, smoke=False):
    protocol = _verify_protocol(protocol_path)
    formal_stages = _stages(protocol['run_id'])
    if not smoke and ('_smoke' in protocol['run_id'] or protocol.get('evidence_class') != 'development'):
        raise ValueError('Software smoke cannot enter a formal destination')
    # A separate run suffix keeps smoke from even creating formal model/output dirs.
    stages = _stages(protocol['run_id'] + '_software_smoke') if smoke else formal_stages
    preflight_outputs(list(stages.values()) if smoke else _destinations(stages))
    if smoke:
        prior = _synthetic_panel()
        actual, fresh = prior.copy(deep=True), prior.copy(deep=True)
    else:
        prior = pd.read_parquet(_source_path(protocol['sources']['prior_panel']))
        actual = pd.read_parquet(_source_path(protocol['sources']['actual_lineup_panel']))
        fresh = pd.read_parquet(_source_path(protocol['sources']['fresh300s_prior_panel']))
        if prior.groupby('phase').size().to_dict() != protocol['expected_phase_denominators']:
            raise ValueError('Frozen provider phase denominators changed')
        for diagnostic in (actual, fresh):
            if diagnostic.row_id.tolist() != prior.row_id.tolist():
                raise ValueError('Diagnostic opportunity identity/order changed')
            for field in ('forecast_eligible','forecast_absence','primary_decision'):
                if field in prior and old._canonical_sha(diagnostic[field].tolist()) != old._canonical_sha(prior[field].tolist()):
                    raise ValueError('Diagnostic forecast eligibility changed: '+field)
    for frame in (prior, actual, fresh):
        if 'quotes' in frame:
            # Arrow's nested sequence decoder returns arrays; values are unchanged.
            frame['quotes'] = frame.quotes.map(
                lambda quotes:quotes.tolist() if isinstance(quotes, np.ndarray) else quotes)
    train, cal, membership = fit_membership(prior)
    expected = protocol.get('expected_common_membership') if not smoke else None
    if expected:
        if len(train) != expected['TRAIN'] or len(cal) != expected['CAL']:
            raise ValueError('Frozen common TRAIN/CAL membership changed')
        if int(prior.native_admitted.sum()) != protocol['expected_native_population']:
            raise ValueError('Frozen native population changed')
        if (int((~prior.forecast_eligible).sum()) != 5 or
                (prior.loc[~prior.forecast_eligible,'forecast_absence'] != 'target_already_released_before_cutoff').any()):
            raise ValueError('Known-result forecast prohibitions changed')
    bindings = {'protocol_sha256':old._canonical_sha(protocol), 'protocol_file_sha256':_sha(protocol_path),
        'sources':{key:ref['sha256'] for key,ref in protocol['sources'].items()},
        'code_sha256':protocol['code_sha256'], 'train_row_ids_sha256':old._canonical_sha(train.row_id.tolist()),
        'cal_row_ids_sha256':old._canonical_sha(cal.row_id.tolist()), 'actual_fit_started_at_utc':_now(),
        'scope':'SOFTWARE_ONLY' if smoke else 'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY',
        'nominal_calendar':protocol['calendar'], 'base_fits':0, 'cal_refits':0}
    membership['bindings'] = bindings
    old._save(stages[5]/'fit_membership.json', membership)
    if smoke:
        old._save(stages[5]/'software_receipt.json', {'classification':'SOFTWARE_ONLY',
            'rows':600, 'phase_rows':200, 'seed':20261006, 'protocol_sha256':bindings['protocol_sha256']})
        for name,frame in (('prior_panel',prior), ('actual_lineup_panel',actual), ('fresh300s_prior_panel',fresh)):
            frame.to_parquet(stages[5]/f'{name}.parquet', index=False)
    status, attempts, selections = membership['support_status'], [], {}
    completed_fits = 0
    if status == 'SUPPORTED':
        stages[6].mkdir(parents=True, exist_ok=False)
        train.attrs['c0_bindings'] = bindings
        # All twelve must converge before any family is selected or scored.
        for arm,recipe in ARMS.items():
            objective = arm.split('-',1)[1]
            for lam in LAMBDAS:
                attempt = {'arm':arm, 'recipe':recipe, 'objective':objective, 'lambda':lam,
                           'started_at_actual_utc':_now()}
                attempts.append(attempt)
                try:
                    head = fit_ev_loss_head(recipe, objective, train, regularization=lam)
                    if head.optimizer.get('success') is not True:
                        raise RuntimeError('Unconverged candidate')
                    old._save(stages[6]/f'{arm}_lambda_{lam:g}.json', head.to_dict())
                    completed_fits += 1
                    attempt.update(success=True, optimizer=head.optimizer)
                except (RuntimeError, ValueError) as error:
                    status = 'INVALID_FIT_INPUT' if isinstance(error, ValueError) else 'OPTIMIZER_FAILED'
                    attempt.update(success=False, error=str(error), error_type=type(error).__name__,
                                   failure_class=status, optimizer=getattr(error,'optimizer',None))
                finally:
                    attempt['completed_at_actual_utc'] = _now()
                if status != 'SUPPORTED':
                    old._save(stages[6]/f'{arm}_lambda_{lam:g}_failed.json', attempt)
                    break
            if status != 'SUPPORTED':
                break
    old._save(stages[8]/'fit_attempts.json', {'attempts':attempts,
        'fit_attempt_count':len(attempts), 'completed_head_fits':completed_fits})
    result = None
    stage_failure = None
    if status == 'SUPPORTED':
        try:
            if completed_fits != 12:
                raise RuntimeError('Wrong fixed twelve-fit budget')
            # No code/input drift is authorized during the fixed training stage.
            _verify_protocol(protocol_path)
            selected = {}
            for arm in ARMS:
                # CAL selection consumes retained candidate bytes, without refitting.
                loaded = [C0EVLossHead.from_dict(_load(stages[6]/f'{arm}_lambda_{lam:g}.json')) for lam in LAMBDAS]
                selected[arm], selections[arm] = select_ev_loss_head(loaded, cal)
            for arm, head in selected.items():
                old._save(stages[6]/f'{arm}_selected.json', head.to_dict())
            reference_heads = {}
            for recipe in REFERENCES:
                content = _source_path(protocol['sources']['reference_'+recipe]).read_bytes()
                target = stages[6]/'reference_heads'/f'{recipe}_selected.json'
                _write_bytes(target, content)
                reference_heads[recipe] = C0MarketHead.from_dict(_load(target))
            old._save(stages[6]/'selection.json', {'selection':selections, 'bindings':bindings,
                'criterion':'return_mse', 'formal_head_fits':0 if smoke else completed_fits,
                'synthetic_head_fits':completed_fits if smoke else 0, 'base_fits':0, 'cal_refits':0,
                'actual_selected_at_utc':_now()})
            arms = {arm:C0EVLossHead.from_dict(_load(stages[6]/f'{arm}_selected.json')) for arm in ARMS}
            assess, actual_assess, fresh_assess = map(_assessment, (prior,actual,fresh))
            if expected:
                common = assess.native_admitted & assess.market_eligible & assess.forecast_eligible
                if int(common.sum()) != expected['ASSESS']:
                    raise ValueError('Frozen common assessment population changed')
            kwargs = {'draws':protocol['statistics']['bootstrap_draws'], 'seed':protocol['statistics']['seed']}
            primary_result = evaluate_ev_loss(assess, arms, reference_heads, **kwargs)
            actual_result = evaluate_ev_loss(actual_assess, arms, reference_heads, **kwargs)
            fresh_result = evaluate_ev_loss(fresh_assess, arms, reference_heads, **kwargs)
            input_diagnostics = {}
            for objective in ('likelihood','return_mse'):
                heads = {**reference_heads, 'H-price':arms['H-'+objective], 'M-price':arms['M-'+objective]}
                input_diagnostics[objective] = aligned._input_diagnostic(
                    assess,actual_assess,heads,protocol,primary_result['arms'][objective])
            # Never publish a partial assessment when a diagnostic stage fails.
            stages[7].mkdir(parents=True, exist_ok=False)
            for filename,evaluation in (('assessment',primary_result), ('actual_lineup_diagnostic',actual_result),
                                        ('fresh300s_diagnostic',fresh_result)):
                evaluation['evidence_class'] = bindings['scope']
                ledger = pd.DataFrame(evaluation['rows'])
                ledger['evidence_class'] = bindings['scope']
                ledger.to_parquet(stages[7]/f'{filename}_predictions_and_ledger.parquet',index=False)
            old._save(stages[8]/'assessment.json', _report(primary_result))
            old._save(stages[8]/'fresh300s_diagnostic.json', {
                'evidence_class':bindings['scope'],
                'classification':'TRANSFER_DIAGNOSTIC_NO_FIT_NO_SELECTION','result':_report(fresh_result)})
            old._save(stages[8]/'actual_lineup_diagnostic.json', {
                'evidence_class':bindings['scope'],
                'classification':'INPUT_ONLY_DIAGNOSTIC_NO_FIT_NO_SELECTION',
                'result':_report(actual_result), 'paired_input_diagnostics':input_diagnostics})
            result, status = primary_result, 'COMPLETED'
        except (RuntimeError, ValueError, FloatingPointError, KeyError, TypeError) as error:
            status = 'STAGE_FAILED'
            stage_failure = {'error':str(error), 'error_type':type(error).__name__,
                'completed_at_actual_utc':_now(), 'retries':0, 'assessment_published':False}
    decision = {'schema':SCHEMA, 'run_id':protocol['run_id'], 'status':status,
        'decision':'SOFTWARE_ONLY' if smoke else (result['decision'] if result else status),
        'gates':result['gates'] if result else None, 'formal_head_fits':0 if smoke else completed_fits,
        'synthetic_head_fits':completed_fits if smoke else 0, 'fit_attempt_count':len(attempts),
        'base_fits':0, 'cal_refits':0, 'selected_arm_count':len(selections),
        'reference_head_count':len(REFERENCES) if status == 'COMPLETED' else 0,
        'real_fit_count':0 if smoke else completed_fits,
        'synthetic_fit_count':completed_fits if smoke else 0,
        'coverage':membership['coverage'], 'selection':selections, 'bindings':bindings,
        'optimizer_attempts':attempts, 'actual_completed_at_utc':_now(),
        'stage_failure':stage_failure,
        'evidence_class':'SOFTWARE_ONLY' if smoke else 'EXPOSED_HISTORICAL_DEVELOPMENT_ONLY',
        'deployed':False, 'canonical_models_changed':False, 'live_quote_requests':0,
        'external_acquisition':0, 'wagers_placed':0, 'genuine_future_forecasts':0,
        'limitations':protocol['limitations']}
    old._save(stages[8]/'decision.json', decision)
    artifacts = {str(path.relative_to(PROJECT_ROOT)):{'sha256':_sha(path),'bytes':path.stat().st_size}
        for stage in stages.values() if stage.exists() for path in sorted(stage.rglob('*')) if path.is_file()}
    old._save(stages[8]/'execution_receipt.json', {'decision':decision, 'artifacts':artifacts,
        'immutable_originals':protocol['immutable_originals'], 'prior_run_artifacts':protocol['prior_run_artifacts'],
        'preserved_forward_code_hashes':protocol['preserved_forward_code_hashes'],
        'code_sha256':protocol['code_sha256'], 'protocol_sha256':bindings['protocol_sha256']})
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(prog='approach-comparison c0-ev-loss-study')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--freeze', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args(argv)
    if args.freeze:
        if not args.run_id or args.protocol:
            parser.error('--freeze requires --run-id and forbids --protocol')
        result = {'protocol':str(freeze_ev_loss(args.run_id)), 'formal_head_fits':0, 'base_fits':0}
    else:
        if not args.protocol or args.run_id:
            parser.error('--run/--smoke require --protocol and forbid --run-id')
        result = run_ev_loss(args.protocol, smoke=args.smoke)
    print(json.dumps(old._json_safe(result), sort_keys=True, allow_nan=False))
    return result
