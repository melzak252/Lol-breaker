"""Pure retained-file/metadata regressions; never fit a candidate or scan sources."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.verify_champion_oov_reliability as oracle_module
from scripts.verify_champion_oov_reliability import (
    ARMS, SEEDS, Audit, Inputs, closure_checks, digest, execution_branch, historical_pin_checks,
    resolve_evidence, retained_operations, terminal_integrity_checks, trace_check,
    verify_token_positions,
)


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def inputs(root, branch, operations=(), execution_status=None, decision=None):
    def artifact(role, fallback=None):
        path = root / (role + '.json')
        if not path.exists():
            raise ValueError('Required artifact missing: ' + role)
        return path
    value = SimpleNamespace(repo=root, research=root, run_id='fixture',
                            index={'branch_status': branch, 'operation_trace': list(operations)},
                            descriptors=[], path=lambda p: root / p,
                            artifact=artifact, stage=lambda stage, name: name)
    if execution_status:
        put(root, 'operation_trace.json', {'status': execution_status, 'operation_trace': list(operations),
                                          'neural_fits': sum(o['operation'] == 'W1_NEURAL_FIT' for o in operations),
                                          'CAL_terminal_scalars': sum(o['operation'] == 'W1_TERMINAL_CAL' for o in operations),
                                          'seed_epochs': sum(o.get('fit_receipt', {}).get('epochs_completed', 0) for o in operations)})
    if decision:
        put(root, 'decision.json', decision)
    return value


def operation(n, arm='H', seed=SEEDS[0], kind='W1_NEURAL_FIT', status='COMPLETED', before=0, epochs=12):
    return dict(operation_id=f'W1:{n}', study='W1', operation=kind, kind=kind, arm=arm,
                seed=seed, partition='TRAIN' if kind == 'W1_NEURAL_FIT' else 'CAL',
                authorized_ceiling=9 if kind == 'W1_NEURAL_FIT' else 3,
                counter_before=before, counter_after=before + 1, started_at_utc='a',
                finished_at_utc='b', status=status, parameters_sha256='p', input_ids_sha256='i',
                iterations=12 if status == 'COMPLETED' else None, function_evaluations=1,
                failure_reason='failed' if status == 'FAILED' else None, epochs=12,
                **({'fit_receipt': {'epochs_completed': epochs}} if epochs is not None and kind == 'W1_NEURAL_FIT' else {}))


def test_float32_zero_reference_preserves_relative_error_diagnostics():
    audit = Audit()
    with np.errstate(divide='raise', invalid='raise'):
        audit.compare('float32', np.array([0, 2], dtype=np.float32),
                      np.array([0, 1], dtype=np.float32))
    check = audit.checks[0]
    assert check['maximum_absolute_difference'] == 1.0
    assert check['maximum_relative_difference'] == 1.0
    assert check['status'] == 'FAIL'
    assert check['mismatches'] == 1


def test_closed_zero_fit_is_dispatched_from_persisted_status(tmp_path):
    value = inputs(tmp_path, 'CLOSED_BY_SUPPORT', decision={'status': 'CLOSED_BY_SUPPORT', 'decision': 'NO_ADOPTION'})
    audit = Audit()
    assert execution_branch(value, audit) == 'ZERO_FIT'
    counts = trace_check(value, [], audit, 'ZERO_FIT')
    assert counts['neural'] == counts['seed_epochs'] == counts['planned_seed_epochs'] == 0
    assert not audit.failures


def test_later_failure_preserves_completed_prefix_and_unknown_failed_progress(tmp_path):
    done = operation(1)
    failed = operation(2, seed=SEEDS[1], status='FAILED', before=1, epochs=None)
    value = inputs(tmp_path, 'FIT_FAILED_WITHIN_BUDGET', [done, failed], 'FIT_FAILED_WITHIN_BUDGET',
                   {'status': 'FIT_FAILED_WITHIN_BUDGET', 'decision': 'NO_ADOPTION'})
    put(tmp_path, f'H_seed_{SEEDS[0]}_fit_receipt.json', done['fit_receipt'])
    audit = Audit()
    branch = execution_branch(value, audit)
    assert branch == 'PARTIAL_FAILED'
    counts = trace_check(value, [], audit, branch)
    assert counts['neural'] == 2
    assert counts['neural_completed'] == 1
    assert counts['seed_epochs'] == 12
    assert counts['planned_seed_epochs'] == 24
    assert counts['failed_attempt_seed_epochs'] == 'UNKNOWN'
    assert counts['seed_epoch_ceiling'] == 108
    assert retained_operations(value, audit)[0] == [('H', SEEDS[0])]
    assert not audit.failures
    put(tmp_path, f'H_seed_{SEEDS[0]}_fit_receipt.json', {'epochs_completed': 11})
    retained_operations(value, audit)
    assert any(c['check_id'].startswith('retained_receipt:') for c in audit.failures)


def test_full_completion_has_nine_observed_trajectories_and_three_scalars(tmp_path):
    operations = []
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            operations.append(operation(len(operations)+1, arm, seed, before=3*a+s))
        operations.append(operation(len(operations)+1, arm, None, 'W1_TERMINAL_CAL', before=a))
    value = inputs(tmp_path, 'NO_ADOPTION', operations, 'COMPLETED', {'status': 'NO_ADOPTION', 'decision': 'NO_ADOPTION'})
    audit = Audit()
    assert execution_branch(value, audit) == 'COMPLETED'
    counts = trace_check(value, [], audit, 'COMPLETED')
    assert (counts['neural'], counts['CAL_temperature'], counts['seed_epochs']) == (9, 3, 108)
    assert counts['failed_attempt_seed_epochs'] == 0
    assert not audit.failures


@pytest.mark.parametrize('decision', [
    {'status': 'DEVELOPMENT_CANDIDATE', 'decision': 'DEVELOPMENT_CANDIDATE'},
    {'status': 'NO_ADOPTION', 'decision': 'NO_ADOPTION'},
])
def test_failed_execution_rejects_success_terminal_receipt(tmp_path, decision):
    value = inputs(tmp_path, 'FIT_FAILED_WITHIN_BUDGET', [operation(1, status='FAILED', epochs=None)],
                   'FIT_FAILED_WITHIN_BUDGET', decision)
    audit = Audit()
    execution_branch(value, audit)
    assert any(c['check_id'] == 'terminal_execution_decision_coherent' for c in audit.failures)


def test_attempt_order_cannot_skip_first_seed_or_resume_after_failure(tmp_path):
    value = inputs(tmp_path, 'FIT_FAILED_WITHIN_BUDGET', [
        operation(1, seed=SEEDS[1], status='FAILED', epochs=None), operation(2, seed=SEEDS[0], before=1)],
        'FIT_FAILED_WITHIN_BUDGET')
    audit = Audit()
    trace_check(value, [], audit, 'PARTIAL_FAILED')
    assert any(c['check_id'] == 'trace_authorized_prefix' for c in audit.failures)
    assert any(c['check_id'] == 'trace_failed_terminal_prefix' for c in audit.failures)


def test_oracle_recovers_bound_custom_root_and_rejects_substitution(tmp_path):
    evidence = tmp_path / 'custom'
    contract = {'input_paths': {'example': 'pinned'}}
    put(evidence, 'w1_contract.json', contract)
    protocol = {'evidence_root': str(evidence.resolve()), 'w1_contract': contract}
    protocol_path = put(tmp_path, 'protocol.json', protocol)
    binding = {'protocol_sha256': digest(protocol_path), 'pins': {'custom/w1_contract.json': digest(evidence / 'w1_contract.json')}}
    put(tmp_path, 'binding.json', binding)
    value = inputs(tmp_path, 'NO_ADOPTION')
    value.index.update(protocol_sha256=digest(protocol_path), binding_sha256=digest(tmp_path / 'binding.json'))
    assert resolve_evidence(value) == evidence.resolve()
    substitute = tmp_path / 'substitute'
    put(substitute, 'w1_contract.json', contract)
    with pytest.raises(ValueError, match='evidence root'):
        resolve_evidence(value, substitute)
    put(evidence, 'w1_contract.json', {'input_paths': {'example': 'different'}})
    with pytest.raises(ValueError, match='contract'):
        resolve_evidence(value, evidence)


def test_historical_models_and_exports_must_match_old_authority_not_current_binding(tmp_path):
    paths = {'saved_d0_models': ['D_seed.joblib'], 'saved_d0_ensemble': 'D_ensemble.joblib',
             'saved_d0_cal_outputs': ['D_cal.parquet'], 'saved_d0_assessment_outputs': ['D_assess.parquet'],
             'saved_c0_assessment_outputs': ['C0.parquet'], 'old_scoring_binding': 'scoring.json', 'old_decision': 'old.json'}
    for name in ['D_seed.joblib', 'D_ensemble.joblib', 'D_cal.parquet', 'D_assess.parquet', 'C0.parquet']:
        (tmp_path / name).write_bytes(name.encode())
    models = {p: digest(tmp_path / p) for p in ['D_seed.joblib', 'D_ensemble.joblib']}
    exports = {p: digest(tmp_path / p) for p in ['D_cal.parquet', 'D_assess.parquet', 'C0.parquet']}
    put(tmp_path, 'scoring.json', {'additional_pins': models, 'models_before_scoring': models})
    put(tmp_path, 'old.json', {'output_hashes': exports})
    value = inputs(tmp_path, 'NO_ADOPTION')
    value.contract = {'input_paths': paths}
    audit = Audit()
    historical_pin_checks(value, audit)
    assert not audit.failures
    (tmp_path / 'D_seed.joblib').write_bytes(b'changed model')
    (tmp_path / 'D_cal.parquet').write_bytes(b'changed reference')
    historical_pin_checks(value, audit)
    assert any(c['check_id'].startswith('historical_D0_model:') for c in audit.failures)
    assert any(c['check_id'].startswith('historical_reference_export:') for c in audit.failures)


def test_failed_attempt_with_retained_progress_counts_observation_not_plan(tmp_path):
    failed=operation(1,status='FAILED',epochs=None)
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[failed],'FIT_FAILED_WITHIN_BUDGET',
                 {'status':'FIT_FAILED_WITHIN_BUDGET','decision':'NO_ADOPTION'})
    put(tmp_path,f'H_seed_{SEEDS[0]}_fit_receipt.json',{'epochs_completed':4})
    execution=json.loads((tmp_path/'operation_trace.json').read_text())
    execution['seed_epochs']=4
    put(tmp_path,'operation_trace.json',execution)
    audit=Audit()
    counts=trace_check(value,[],audit,'PARTIAL_FAILED')
    assert counts['seed_epochs']==counts['failed_attempt_seed_epochs']==4
    assert counts['planned_seed_epochs']==12
    assert counts['neural']==1 and counts['neural_completed']==0
    assert not audit.failures


def test_failure_cannot_retain_a_success_integrity_receipt(tmp_path):
    reason={'failure_phase':'FINAL_INTEGRITY','failure_reason':'protected byte changed'}
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[], 'FIT_FAILED_WITHIN_BUDGET',
                 {'status':'FIT_FAILED_WITHIN_BUDGET','decision':'NO_ADOPTION',**reason})
    execution=json.loads((tmp_path/'operation_trace.json').read_text())
    put(tmp_path,'operation_trace.json',{**execution,**reason})
    audit=Audit()
    terminal_integrity_checks(value,audit,{'protected_before':{}},'PARTIAL_FAILED')
    assert not audit.failures
    put(tmp_path,'final_integrity.json',{'protected_after':{}})
    terminal_integrity_checks(value,audit,{'protected_before':{}},'PARTIAL_FAILED')
    assert any(c['check_id']=='terminal_failed_integrity_has_no_success_receipt' for c in audit.failures)


def test_completed_integrity_receipt_cannot_claim_different_protected_pins(tmp_path):
    value=inputs(tmp_path,'NO_ADOPTION')
    put(tmp_path,'final_integrity.json',{'protected_after':{'protected':'different'}})
    audit=Audit()
    terminal_integrity_checks(value,audit,{'protected_before':{'protected':'frozen'}},'COMPLETED')
    assert any(c['check_id']=='terminal_final_integrity_receipt' for c in audit.failures)


def test_actual_saved_runner_rejects_substituted_root_before_loading_inputs(tmp_path):
    from src.analysis.champion_oov_reliability import predict_saved
    bound=tmp_path/'bound'
    alternate=tmp_path/'alternate'
    contract={'input_paths':{'old_prepared_root':'intentionally_absent'}}
    put(bound,'w1_contract.json',contract)
    put(alternate,'w1_contract.json',contract)
    stage=tmp_path/'data/05_model_input/champion_oov_reliability/fixture'
    protocol_path=put(stage,'protocol.json',{'evidence_root':str(bound.resolve()),'w1_contract':contract})
    put(stage,'binding.json',{'protocol_sha256':digest(protocol_path),'pins':{}})
    with pytest.raises(ValueError,match='evidence root'):
        predict_saved('fixture',['one'],repo_root=tmp_path,research_root=tmp_path,evidence_root=alternate)


def closure_outputs(root, primary_status='NOT_RUN_FIT_FAILED', guard_status='NOT_RUN'):
    comparisons={f'HC-{ref}:{group}:winner_nll':{'status':primary_status}
                 for ref,group in [('D0','natural_champion_oov'),('H','natural_champion_oov'),
                                   ('C','natural_champion_oov'),('C0','all')]}
    guards={f'HC-{ref}:{group}':{'status':guard_status}
            for group in ('BO3','BO5','major','player_source_unseen') for ref in ('D0','C0')}
    guards['HC-D0:seen_champion_history']={'status':guard_status}
    put(root,'comparisons.json',comparisons)
    put(root,'guards.json',guards)


def test_partial_failure_closure_keeps_all_reserved_slots_unrun_and_budget_consumed(tmp_path):
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[], 'FIT_FAILED_WITHIN_BUDGET',
                 {'status':'FIT_FAILED_WITHIN_BUDGET','decision':'NO_ADOPTION',
                  'neural_fits':2,'CAL_terminal_scalars':0,'seed_epochs':12})
    closure_outputs(tmp_path)
    audit=Audit()
    counts={'neural':2,'CAL_temperature':0,'seed_epochs':12}
    assert closure_checks(value,audit,True,counts)=={str(s):'NOT_RUN_FIT_FAILED' for s in (3,4,5,6)}
    assert not audit.failures
    comparisons=json.loads((tmp_path/'comparisons.json').read_text())
    next(iter(comparisons.values()))['point']=-0.1
    put(tmp_path,'comparisons.json',comparisons)
    closure_checks(value,audit,True,counts)
    assert any(c['check_id'].startswith('closure_primary_NOT_RUN:') for c in audit.failures)


def test_final_integrity_failure_preserves_computed_evidence_without_authorizing_it(tmp_path):
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[], 'FIT_FAILED_WITHIN_BUDGET',
                 {'status':'FIT_FAILED_WITHIN_BUDGET','decision':'NO_ADOPTION',
                  'neural_fits':9,'CAL_terminal_scalars':3,'seed_epochs':108})
    execution=json.loads((tmp_path/'operation_trace.json').read_text())
    put(tmp_path,'operation_trace.json',{**execution,'failure_phase':'FINAL_INTEGRITY'})
    closure_outputs(tmp_path,'PASS','PASS')
    before=(tmp_path/'comparisons.json').read_bytes()
    audit=Audit()
    counts={'neural':9,'CAL_temperature':3,'seed_epochs':108}
    assert closure_checks(value,audit,True,counts)=={str(s):'NOT_AUTHORIZING_INTEGRITY_FAILED' for s in (3,4,5,6)}
    assert (tmp_path/'comparisons.json').read_bytes()==before
    assert not audit.failures


def test_completed_scalar_receipt_is_verified_before_a_later_failure(tmp_path):
    scalar=operation(4,seed=None,kind='W1_TERMINAL_CAL')
    scalar['receipt']={'score_temperature':{'temperature':1.25}}
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[scalar])
    put(tmp_path,'H_temperature.json',scalar['receipt'])
    audit=Audit()
    assert retained_operations(value,audit)==([],['H'])
    assert not audit.failures
    put(tmp_path,'H_temperature.json',{'score_temperature':{'temperature':2}})
    retained_operations(value,audit)
    assert any(c['check_id']=='retained_scalar_receipt:H' for c in audit.failures)


@pytest.mark.parametrize('epochs',[-1,13,True,1.5])
def test_invalid_recorded_failed_progress_is_rejected_not_reclassified_unknown(tmp_path,epochs):
    failed=operation(1,status='FAILED',epochs=epochs)
    value=inputs(tmp_path,'FIT_FAILED_WITHIN_BUDGET',[failed],'FIT_FAILED_WITHIN_BUDGET')
    audit=Audit()
    trace_check(value,[],audit,'PARTIAL_FAILED')
    assert any(c['check_id'].startswith('trace_recorded_epoch_bounds:') for c in audit.failures)


def test_historical_model_pin_authorities_must_agree_even_if_bytes_match_one(tmp_path):
    paths={'old_scoring_binding':'scoring.json','old_decision':'old.json',
           'saved_d0_models':['D.joblib'],'saved_d0_ensemble':'D_ensemble.joblib',
           'saved_d0_cal_outputs':[],'saved_d0_assessment_outputs':[],'saved_c0_assessment_outputs':[]}
    (tmp_path/'D.joblib').write_bytes(b'frozen model')
    (tmp_path/'D_ensemble.joblib').write_bytes(b'frozen ensemble')
    pins={name:digest(tmp_path/name) for name in ('D.joblib','D_ensemble.joblib')}
    put(tmp_path,'scoring.json',{'additional_pins':pins,'models_before_scoring':{**pins,'D.joblib':'different'}})
    put(tmp_path,'old.json',{'output_hashes':{}})
    value=inputs(tmp_path,'NO_ADOPTION')
    value.contract={'input_paths':paths}
    audit=Audit()
    historical_pin_checks(value,audit)
    assert [c['check_id'] for c in audit.failures]==['historical_D0_model:D.joblib']


def test_stale_prefit_index_cannot_hide_a_persisted_attempt(tmp_path):
    value=inputs(tmp_path,'FROZEN_BEFORE_FIT')
    put(tmp_path,'execution.json',{'status':'RUNNING','operation_trace':[operation(1)]})
    audit=Audit()
    assert execution_branch(value,audit)=='PREPARE'
    assert any(c['check_id']=='prepare_no_execution_attempt' for c in audit.failures)


def test_zero_fit_index_cannot_hide_a_failed_execution(tmp_path):
    value=inputs(tmp_path,'CLOSED_BY_SUPPORT',[], 'FIT_FAILED_WITHIN_BUDGET',
                 {'status':'CLOSED_BY_SUPPORT','decision':'NO_ADOPTION'})
    put(tmp_path,'execution.json',{'status':'FIT_FAILED_WITHIN_BUDGET','operation_trace':[]})
    audit=Audit()
    execution_branch(value,audit)
    assert any(c['check_id']=='zero_fit_execution_matches_branch' for c in audit.failures)


@pytest.mark.parametrize('defect', [None, 'archive', 'current', 'other_source'])
def test_verifier_lineage_cannot_override_frozen_study_inputs(tmp_path, monkeypatch, defect):
    relative = 'scripts/verify_champion_oov_reliability.py'
    current = tmp_path / relative
    current.parent.mkdir()
    current.write_text('current verifier')
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    archive = evidence / 'frozen_verifier.py'
    archive.write_text('frozen verifier')
    other = tmp_path / 'model.py'
    other.write_text('frozen model')
    pins = {relative: digest(archive), 'model.py': digest(other)}
    put(tmp_path, 'binding.json', {'pins': pins})
    value = object.__new__(Inputs)
    value.repo = value.research = tmp_path
    value.run_id = 'run'
    value.index = {'protocol_sha256': 'bound'}
    value.descriptors = [{'role': 'binding', 'repo_relative_path': 'binding.json'}]
    value._verifier_archive = None
    lineage = {'schema': 'w1_verifier_lineage_v1', 'run_id': 'run',
               'allowed_source': relative, 'scientific_changes': False,
               'archived_source': str(archive.relative_to(tmp_path)),
               'frozen_sha256': pins[relative], 'current_sha256': digest(current)}
    if defect == 'archive':
        archive.write_text('tampered frozen verifier')
    elif defect == 'current':
        current.write_text('unapproved current verifier')
    elif defect == 'other_source':
        lineage['allowed_source'] = 'model.py'
    manifest = put(evidence, 'verifier_lineage.json', lineage)
    monkeypatch.setattr(oracle_module, '__file__', str(current))
    if defect:
        with pytest.raises(ValueError):
            value.set_verifier_lineage(manifest, evidence)
        assert value._verifier_archive is None
    else:
        value.set_verifier_lineage(manifest, evidence)
        assert digest(value.path(relative)) == pins[relative]
        other.write_text('tampered model')
        assert digest(value.path('model.py')) != pins['model.py']


@pytest.mark.parametrize('defect', [None, 'cross_batch_duplicate', 'wrong_event', 'missing_row'])
def test_streamed_token_audit_preserves_complete_permuted_membership(tmp_path, defect):
    events = np.arange(8).reshape(2, 2, 1, 2)
    positions = np.argwhere(np.ones(events.shape, bool))
    table = pd.DataFrame(positions, columns=['row_index', 'side', 'player_slot', 'history_slot'])
    table['event_index'] = np.arange(8)
    table = table.iloc[[7, 0, 5, 2, 1, 4, 3, 6]].reset_index(drop=True)
    if defect == 'cross_batch_duplicate':
        table.iloc[-1] = table.iloc[0]
    elif defect == 'wrong_event':
        table.loc[7, 'event_index'] = 99
    elif defect == 'missing_row':
        table = table.iloc[:-1]
    path = tmp_path / 'tokens.parquet'
    table.to_parquet(path, index=False, row_group_size=3)
    audit = Audit()
    verify_token_positions(path, events, np.ones(events.shape, bool), audit, batch_size=2)
    assert bool(audit.failures) == (defect is not None)
    checks = {check['check_id']: check for check in audit.checks}
    assert checks['token_rows_all_valid']['status'] == ('FAIL' if defect == 'missing_row' else 'PASS')
    assert checks['token_positions_unique']['status'] == ('FAIL' if defect == 'cross_batch_duplicate' else 'PASS')
    event_failures = [check for check in audit.failures if check['check_id'].startswith('token_event_positions:')]
    assert sum(check['mismatches'] for check in event_failures) == (1 if defect == 'wrong_event' else 0)
