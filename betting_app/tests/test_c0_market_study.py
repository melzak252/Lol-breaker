"""Consumer-level regressions for the separately versioned native study."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.optimize import minimize
from src.models.ev_approach_models import (
    C0MarketHead, C0_RECIPES, C0_EPSILON, fit_c0_market_head, c0_features,
    _regularized_loss,
)
from src.analysis.ev_approach_evaluation import evaluate_c0_market_heads
from src.analysis.ev_approach_data import C0_PHASES, validate_c0_protocol


def population(n=240):
    rng = np.random.default_rng(20260929)
    z, m = rng.normal(size=(2, n))
    return pd.DataFrame({'row_id': [str(i) for i in range(n)], 'p_C0': expit(z),
        'q': expit(m), 'y': rng.binomial(1, expit(.3*z+.7*m)),
        'native_admitted': True, 'financial_eligible': True,
        'month': ['2028-07' if i < n//2 else '2028-08' for i in range(n)],
        'quotes': [[] for _ in range(n)]})


@pytest.mark.parametrize('recipe', ['S-cal','M-cal','M-shape','H'])
def test_independent_optimum_gradient_and_swap(recipe):
    frame = population()
    x = c0_features(frame, recipe)
    y = frame.y.to_numpy()
    center = np.array([0.,1.] if recipe == 'H' else [1.]*x.shape[1])
    theta = center + .1
    objective, grad = _regularized_loss(theta,x,y,center,.1,'logloss')
    def independent(t):
        eta = x @ t
        return np.mean(np.logaddexp(0,eta)-y*eta)+.05*np.sum((t-center)**2)
    assert objective == pytest.approx(independent(theta), abs=1e-14)
    finite = np.array([(independent(theta+np.eye(len(theta))[i]*1e-5)-independent(theta-np.eye(len(theta))[i]*1e-5))/2e-5 for i in range(len(theta))])
    np.testing.assert_allclose(grad,finite,atol=1e-9)
    bounds = [(.05,20)] if len(theta)==1 else [(0,None)]*2
    other = minimize(independent,center,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':2000})
    head = fit_c0_market_head(recipe,frame,regularization=.1)
    assert independent(np.array(head.coefficients)) == pytest.approx(other.fun,abs=1e-8)
    swapped = frame.assign(p_C0=1-frame.p_C0,q=1-frame.q,y=1-frame.y)
    np.testing.assert_allclose(head.predict(frame)+head.predict(swapped),1,atol=1e-14)
    replay = C0MarketHead.from_dict(head.to_dict())
    np.testing.assert_array_equal(head.predict(frame),replay.predict(frame))


def test_endpoints_and_recipe_specific_inputs():
    frame = pd.DataFrame({'p_C0':[0.,1.]})
    head = fit_c0_market_head('S',frame,regularization=0)
    np.testing.assert_allclose(head.predict(frame),[C0_EPSILON,1-C0_EPSILON],atol=1e-15)
    market = fit_c0_market_head('M',pd.DataFrame({'q':[.2,.8]}),regularization=0)
    np.testing.assert_allclose(market.predict(pd.DataFrame({'q':[.2,.8]})),[.2,.8])
    for invalid in (True, None, float('inf'), -0.1,1.1):
        with pytest.raises(ValueError):
            head.predict(pd.DataFrame({'p_C0':[invalid]}))


def test_rank_and_one_class_support():
    frame = population().assign(q=.5)
    with pytest.raises(ValueError,match='INSUFFICIENT_FIT_SUPPORT'):
        fit_c0_market_head('H',frame,regularization=.1)
    with pytest.raises(ValueError,match='INSUFFICIENT_FIT_SUPPORT'):
        fit_c0_market_head('S-cal',population().assign(y=0),regularization=.1)


def test_no_selection_and_one_month_never_pass():
    frame = population()
    heads = {r:fit_c0_market_head(r,frame,regularization=.1) for r in C0_RECIPES}
    result = evaluate_c0_market_heads(frame,heads,draws=50)
    assert result['decision'] == 'NO_BETTING_EDGE_CLAIM'
    assert result['metrics']['H']['ROI']['interval'] is None
    assert result['metrics']['H']['ROI']['equal_month_point'] is None
    one = evaluate_c0_market_heads(frame.assign(month='2028-07'),heads,draws=50)
    assert one['contrasts']['M-cal']['status'] == 'NOT_ESTIMABLE'


def test_wrong_native_mode_and_calendar_rejected():
    protocol = {'schema':'c0_market_study_v1','protocol_id':'native-c0-market-study-v1',
        'native_identity':'Causal-C0:c0-native-2026-w32-e12-v1','mode':'full',
        'calendar':{k:list(v) for k,v in C0_PHASES.items()},'epsilon':1e-6,'lambdas':[.01,.1,1.]}
    validate_c0_protocol(protocol)
    for field,value in [('mode','no_history'),('native_identity','EXP081')]:
        with pytest.raises(ValueError):
            validate_c0_protocol({**protocol,field:value})


def test_cli_synthetic_seals_and_zero_real_fits(tmp_path):
    from scripts.run_unified_betting_evaluation import run_c0_market_study
    import json
    frozen = tmp_path/'frozen'
    run_c0_market_study(['--freeze','--run-id','pytest-native-c0','--output-dir',str(frozen)])
    protocol = frozen/'protocol.json'
    run_c0_market_study(['--protocol',str(protocol),'--no-fit','--output-dir',str(tmp_path/'blocked')])
    blocked = json.loads((tmp_path/'blocked'/'receipt.json').read_text())
    assert blocked['status'] == 'G0_BLOCKED'
    assert blocked['real_fit_count'] == blocked['model_count'] == 0
    synthetic = tmp_path/'synthetic'
    run_c0_market_study(['--protocol',str(protocol),'--synthetic-smoke','--output-dir',str(synthetic)])
    receipt = json.loads((synthetic/'receipt.json').read_text())
    assert receipt['real_fit_count'] == 0
    assert receipt['synthetic_fit_count'] == 12
    assert receipt['row_count'] == 600
    assert receipt['status'] == 'SYNTHETIC_SOFTWARE_ONLY'
    assert len(list((synthetic/'train'/'candidates').glob('*.json'))) == 12
    assert len(list((synthetic/'select'/'selected').glob('*.json'))) == 7
    for phase,clock in [('train','2027-11-16'),('select','2028-06-16'),('assess','2029-07-16')]:
        stage = json.loads((synthetic/phase/'receipt.json').read_text())
        assert stage['stage_clock'].startswith(clock)
    assert receipt['failure_cases']['zero_selection']['interval'] is None
    with pytest.raises(FileExistsError):
        run_c0_market_study(['--protocol',str(protocol),'--synthetic-smoke','--output-dir',str(synthetic)])
    import hashlib
    from scripts.run_unified_betting_evaluation import _validate_c0_previous_stage
    from src.analysis.ev_approach_data import c0_canonical_bytes
    frozen_protocol = json.loads(protocol.read_text())
    previous_path = synthetic/'train'/'receipt.json'
    previous = json.loads(previous_path.read_text())
    omitted = json.loads(json.dumps(previous))
    omitted['artifacts'].pop('candidates/H-0.01.json')
    previous_path.write_bytes(c0_canonical_bytes(omitted) + b'\n')
    with pytest.raises(ValueError, match='artifact set'):
        _validate_c0_previous_stage(previous_path, expected_stage='train',
                                   protocol=frozen_protocol, synthetic=True)
    relative = 'candidates/S-cal-0.01.json'
    candidate_path = previous_path.parent/relative
    foreign = json.loads((previous_path.parent/'candidates/H-0.01.json').read_text())
    candidate_path.write_bytes(c0_canonical_bytes(foreign) + b'\n')
    previous['artifacts'][relative] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    previous_path.write_bytes(c0_canonical_bytes(previous) + b'\n')
    with pytest.raises(ValueError, match='family/lambda/TRAIN'):
        _validate_c0_previous_stage(previous_path, expected_stage='train',
                                   protocol=frozen_protocol, synthetic=True)


def test_corrupt_and_wrong_phase_bundle_is_input_error(tmp_path):
    import json
    import hashlib
    from src.analysis.ev_approach_data import c0_digest, prepare_c0_market_study_rows
    protocol = {'schema':'c0_market_study_v1','protocol_id':'native-c0-market-study-v1',
        'native_identity':'Causal-C0:c0-native-2026-w32-e12-v1','mode':'full',
        'calendar':{k:list(v) for k,v in C0_PHASES.items()},'epsilon':1e-6,'lambdas':[.01,.1,1.]}
    manifest = {'schema':'c0_market_study_bundle_v1','phase':'CAL',
        'protocol_id':protocol['protocol_id'],'protocol_sha256':c0_digest(protocol)}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Wrong phase'):
        prepare_c0_market_study_rows(tmp_path,protocol,phase='TRAIN')
    manifest['phase'] = 'TRAIN'
    manifest['files'] = {'opportunities.jsonl':{'sha256':'0'*64}}
    (tmp_path/'opportunities.jsonl').write_text('{}\\n')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='SHA-256'):
        prepare_c0_market_study_rows(tmp_path,protocol,phase='TRAIN')


def test_exact_lambda_tie_largest_and_no_refit():
    from src.models.ev_approach_models import select_c0_market_head
    frame = population()
    candidates = [C0MarketHead('S-cal',(1.,),value,{'success':True},{'membership':'train'})
                  for value in (.01,.1,1.)]
    selected, receipt = select_c0_market_head(candidates,frame)
    assert selected is candidates[-1]
    assert receipt['lambda'] == 1.
    assert selected.bindings == {'membership':'train'}


def test_actual_label_availability_and_late_release():
    from src.analysis.ev_approach_data import c0_label_available_at
    capture = {'response_received_at':'2027-11-10T00:00:00Z',
               'recorded_at':'2027-11-10T00:00:01Z'}
    assert c0_label_available_at({},capture,phase='TRAIN') == pd.Timestamp('2027-11-10T00:00:01Z')
    with pytest.raises(ValueError,match='phase lock'):
        c0_label_available_at({'available_at':'2027-11-15T00:00:00Z'},capture,phase='TRAIN')
    with pytest.raises(ValueError,match='precedes'):
        c0_label_available_at({'available_at':'2027-11-09T00:00:00Z'},capture,phase='TRAIN')


def test_net_overstatement_sign_decomposition_and_retained_denominator():
    frame = population(20).assign(p_C0=.6,q=.55)
    frame['quotes'] = [[{'bookmaker':'A','odds_snapshot_id':i,'odds_a':3.,'odds_b':1.1}]
                      for i in range(len(frame))]
    heads = {
        r:C0MarketHead(r,(.5,.5) if r == 'Fixed50' else (0.,1.) if r == 'H'
                      else (1.,1.) if r == 'M-shape' else (1.,),0.,{'success':True},{})
        for r in C0_RECIPES
    }
    report = evaluate_c0_market_heads(frame,heads,draws=100)
    selected = [r for r in report['rows'] if r['recipe'] == 'S' and r['selected']]
    assert len(selected) == len(frame)
    for index,row in enumerate(selected):
        y = frame.iloc[index].y
        assert row['overstatement'] == pytest.approx(.88*3*(.6-y))
        assert row['model_market_component']+row['market_noise_component'] == pytest.approx(row['overstatement'])
    assert report['metrics']['S']['financial_eligible_count'] == len(frame)
    assert report['metrics']['S']['profit_per_opportunity']['point'] == pytest.approx(
        np.mean([r['profit'] for r in selected]))


def test_unknown_cutoff_and_retained_participant_binding_preserve_absence(tmp_path):
    import base64
    import hashlib
    from src.analysis.ev_approach_data import c0_canonical_bytes, c0_digest, prepare_c0_market_study_rows
    protocol = {'schema':'c0_market_study_v1','protocol_id':'native-c0-market-study-v1',
        'native_identity':'Causal-C0:c0-native-2026-w32-e12-v1','mode':'full','code_hashes':{},
        'calendar':{k:list(v) for k,v in C0_PHASES.items()},'epsilon':1e-6,'lambdas':[.01,.1,1.]}
    original = {'fixture_id':'fixture','bookmaker':'pinnacle','participant_id':'A','market_id':'181'}
    body = c0_canonical_bytes({'participant_bindings':[original]})
    sha = hashlib.sha256(body).hexdigest()
    binding = {**original,'capture_id':'binding-capture','source_sha256':sha}
    values = {
        'opportunities.jsonl':[{'event_key':'pending','event_type':'discovery','primary_decision':False,
            'canonical_match_id':'fixture','nominal_cutoff_at':None,'payload':{'clocks':{}}}],
        'captures.jsonl':[{'capture_id':'binding-capture','sha256':sha,
                          'content_base64':base64.b64encode(body).decode(),'source_class':'local_fixture'}],
        'bindings.jsonl':[binding], 'envelopes.jsonl':[], 'dependencies.jsonl':[],
        'authentication.jsonl':[], 'labels.jsonl':[]}
    files = {}
    for name, rows in values.items():
        content = b''.join(c0_canonical_bytes(row)+b'\n' for row in rows)
        (tmp_path/name).write_bytes(content)
        files[name] = {'sha256':hashlib.sha256(content).hexdigest(),'row_count':len(rows)}
    membership = {'schema':'c0_phase_membership_v1','phase':'TRAIN','protocol_id':protocol['protocol_id'],
        'opportunity_sha256':files['opportunities.jsonl']['sha256'],'event_keys':['pending']}
    for name, value in [('membership.json',membership),('authorization.json',{})]:
        content = c0_canonical_bytes(value)
        (tmp_path/name).write_bytes(content)
        files[name] = {'sha256':hashlib.sha256(content).hexdigest()}
    manifest = {'schema':'c0_market_study_bundle_v1','phase':'TRAIN','protocol_id':protocol['protocol_id'],
        'protocol_sha256':c0_digest(protocol),'files':files,'membership_sha256':c0_digest(membership)}
    (tmp_path/'manifest.json').write_bytes(c0_canonical_bytes(manifest))
    frame, admission = prepare_c0_market_study_rows(tmp_path,protocol,phase='TRAIN')
    assert frame.row_id.tolist() == ['pending']
    assert pd.isna(frame.iloc[0].decision_at)
    assert 'cutoff_unknown' in frame.iloc[0].admission_reasons
    assert not frame.iloc[0].financial_eligible
    assert admission['status'] == 'G0_BLOCKED'
    binding['participant_id'] = 'B'
    content = c0_canonical_bytes(binding)+b'\n'
    (tmp_path/'bindings.jsonl').write_bytes(content)
    manifest['files']['bindings.jsonl']['sha256'] = hashlib.sha256(content).hexdigest()
    (tmp_path/'manifest.json').write_bytes(c0_canonical_bytes(manifest))
    with pytest.raises(ValueError,match='differs from retained source bytes'):
        prepare_c0_market_study_rows(tmp_path,protocol,phase='TRAIN')
