"""Deterministic R1 error-boundary regressions; no fitting or native inference."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from src.analysis import lineup_law_reliability as runner
from betting_app.services import c0_inference as native


def source():
    return {'match_id': '1', 'date': '2026-04-01', 'tournament_name': 'League',
            'best_of': 3, 'games': [{'date': '2026-04-01', 't1_id': 'A', 't2_id': 'B',
            **{side + '_players': {str(i): {'player_id': prefix + str(i)} for i in range(5)}
               for side, prefix in (('t1', 'a'), ('t2', 'b'))}}]}


def test_contradictory_header_opponents_are_source_validation_failure():
    raw = source()
    raw.update(tid_1='A', tid_2='A')
    before = deepcopy(raw)
    with pytest.raises(runner.SourceValidationError, match='distinct'):
        runner._source_maps(raw, {})
    assert raw == before


@pytest.mark.parametrize('header', [{}, {'tid_1': 'A'}, {'tid_2': 'B'}, {'tid_1': 'B'}])
def test_compatible_missing_header_retains_schema_inference(header):
    raw = source()
    raw.update(header)
    events, _ = runner._source_maps(raw, {})
    assert {event['team_id'] for event in events} == {'A', 'B'}


def frozen_run(tmp_path, monkeypatch):
    repo, research, paths = runner._roots('integrity', tmp_path, tmp_path)
    for path in paths.values():
        path.mkdir(parents=True)
    pin = repo / 'input.json'
    pin.write_text('{}')
    protocol = {'run_id': 'integrity', 'scope': 'HISTORICAL_DEVELOPMENT_ONLY',
                'repo_root': str(repo), 'research_root': str(research),
                'hashes': {'oracle_contract.json': {'path': str(pin), 'sha256': runner._sha(pin)},
                           'raw_source': {'path': str(pin), 'sha256': runner._sha(pin)}},
                'import_origins': {}, 'protected_before': {}, 'population_ids': [],
                'prefit_resolutions': {}}
    runner._json(paths['05_model_input'] / 'protocol.json', protocol)
    runner._json(paths['05_model_input'] / 'binding.json', {})
    monkeypatch.setattr(runner, '_frozen', lambda *args, **kwargs: (repo, research, paths, protocol, {}))
    return paths, protocol


def files_under(tmp_path):
    return {str(path.relative_to(tmp_path)): path.read_bytes()
            for path in tmp_path.rglob('*') if path.is_file()}


def test_repeated_ready_prepare_refusal_changes_no_retained_bytes(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    runner._json(paths['08_reporting'] / 'prepare_started.json', {'formal_fits': 0})
    runner._json(paths['08_reporting'] / 'support.json', {'status': 'READY_FOR_TWO_TRAIN_FITS'})
    before = files_under(tmp_path)
    with pytest.raises(FileExistsError):
        runner.prepare_run('integrity', repo_root=tmp_path, research_root=tmp_path)
    assert files_under(tmp_path) == before


@pytest.mark.parametrize('error,kind', [(KeyboardInterrupt('stop'), 'INTERRUPTED'),
    (OSError('disk failure'), 'IO'), (RuntimeError('implementation defect'), 'SOFTWARE'),
    (ValueError('untyped implementation defect'), 'SOFTWARE')])
def test_owned_prepare_execution_failure_never_declares_source_invalid(tmp_path, monkeypatch, error, kind):
    paths, protocol = frozen_run(tmp_path, monkeypatch)
    protocol['hashes']['native_release_map'] = protocol['hashes']['raw_source']
    monkeypatch.setattr(runner, '_load', lambda path: (_ for _ in ()).throw(error)
                        if str(path) == protocol['hashes']['raw_source']['path'] else json.loads(path.read_text()))
    with pytest.raises(type(error)):
        runner.prepare_run('integrity', repo_root=tmp_path, research_root=tmp_path)
    support = json.loads((paths['08_reporting'] / 'support.json').read_text())
    stop = json.loads((paths['08_reporting'] / 'prepare_hard_stop.json').read_text())
    decision = json.loads((paths['08_reporting'] / 'decision.json').read_text())
    assert stop['failure_kind'] == kind
    assert stop['identity_release_validation_status'] == 'NOT_ESTABLISHED'
    assert stop['formal_fits'] == decision['formal_fits'] == 0
    assert decision['unused_TRAIN_slots'] == 2
    assert stop['status'] == support['status'] == decision['status'] == 'INCONCLUSIVE'
    assert support['counts']['identity_release_valid'] is None
    assert 'source_backed_failure' not in support


def test_prepare_owns_marker_even_when_marker_serialization_fails(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    safe = runner._safe

    def fail_marker(value):
        if isinstance(value, dict) and set(value) == {'started_at_utc', 'formal_fits'}:
            raise OSError('marker serialization failure')
        return safe(value)

    monkeypatch.setattr(runner, '_safe', fail_marker)
    with pytest.raises(OSError, match='marker serialization'):
        runner.prepare_run('integrity', repo_root=tmp_path, research_root=tmp_path)
    stop = runner._load(paths['08_reporting'] / 'prepare_hard_stop.json')
    assert stop['failure_kind'] == 'IO'
    assert stop['formal_fits'] == 0
    assert stop['identity_release_validation_status'] == 'NOT_ESTABLISHED'


def test_partial_attempt_failure_retains_consumed_slot_without_source_blame(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    runner._json(paths['08_reporting'] / 'conditional_attempt.json',
                 {'arm': 'conditional', 'status': 'ATTEMPTED', 'counter_before': 0, 'counter_after': 1})
    runner._failed_command('integrity', 'run', KeyboardInterrupt('fit interrupted'), tmp_path, tmp_path)
    decision = runner._load(paths['08_reporting'] / 'decision.json')
    stop = runner._load(paths['08_reporting'] / 'run_hard_stop.json')
    assert decision['formal_fits'] == 1
    assert decision['completed_fits'] == 0
    assert decision['unused_TRAIN_slots'] == 1
    assert decision['operation_trace'][0]['status'] == 'INTERRUPTED_OR_FAILED'
    assert stop['failure_kind'] == 'INTERRUPTED'
    assert stop['identity_release_validation_status'] == 'NOT_ESTABLISHED'


def test_identified_source_defect_alone_creates_identity_release_block(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    runner._json(paths['08_reporting'] / 'prepare_started.json', {})
    raw = source()
    raw.update(tid_1='A', tid_2='A')
    with pytest.raises(runner.SourceValidationError) as raised:
        runner._source_maps(raw, {})
    runner._failed_command('integrity', 'prepare', raised.value, tmp_path, tmp_path)
    support = runner._load(paths['08_reporting'] / 'support.json')
    stop = runner._load(paths['08_reporting'] / 'prepare_hard_stop.json')
    assert support['status'] == stop['status'] == 'BLOCKED_IDENTITY_RELEASE'
    assert support['counts']['identity_release_valid'] is False
    assert stop['failure_kind'] == 'SOURCE_VALIDATION'
    assert stop['identity_release_validation_status'] == 'INVALID'


def test_postfit_fatal_failure_preserves_actual_budget_and_validated_source(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    runner._json(paths['08_reporting'] / 'support.json',
                 {'status': 'READY_FOR_TWO_TRAIN_FITS', 'counts': {'identity_release_valid': True}})
    for index, arm in enumerate(('conditional', 'global')):
        runner._json(paths['08_reporting'] / (arm + '_result.json'),
                     {'arm': arm, 'status': 'COMPLETED', 'counter_before': index, 'counter_after': index + 1})
    runner._failed_command('integrity', 'run', native.C0InferenceError('corrupt evidence'), tmp_path, tmp_path)
    decision = runner._load(paths['08_reporting'] / 'decision.json')
    stop = runner._load(paths['08_reporting'] / 'run_hard_stop.json')
    assert decision['formal_fits'] == decision['completed_fits'] == 2
    assert decision['unused_TRAIN_slots'] == 0
    assert [item['counter_after'] for item in decision['operation_trace']] == [1, 2]
    assert stop['failure_kind'] == 'NATIVE_INTEGRITY'
    assert stop['identity_release_validation_status'] == 'VALID'
    assert decision['status'] == 'INCONCLUSIVE'


def test_late_owned_failure_is_retained_without_rewriting_earlier_decision(tmp_path, monkeypatch):
    paths, _ = frozen_run(tmp_path, monkeypatch)
    decision_path = paths['08_reporting'] / 'decision.json'
    runner._json(decision_path, {'status': 'NO_LINEUP_LAW_GAIN', 'formal_fits': 2})
    runner._json(paths['08_reporting'] / 'operation_trace.json',
                 [{'arm': arm, 'status': 'COMPLETED'} for arm in ('conditional', 'global')])
    runner._json(paths['08_reporting'] / 'support.json',
                 {'status': 'READY_FOR_TWO_TRAIN_FITS', 'counts': {'identity_release_valid': True}})
    before = decision_path.read_bytes()
    runner._failed_command('integrity', 'run', OSError('final index write failed'), tmp_path, tmp_path)
    stop = runner._load(paths['08_reporting'] / 'run_hard_stop.json')
    assert stop['failure_kind'] == 'IO'
    assert stop['formal_fits'] == stop['completed_fits'] == 2
    assert stop['unused_TRAIN_slots'] == 0
    assert decision_path.read_bytes() == before
    assert runner._load(paths['08_reporting'] / 'oracle_inputs.json')['branch_status'] == 'INCONCLUSIVE'


class FaultBoundaryClient(native.C0InferenceClient):
    """Exercise real identity/predict_evidence paths with no process or model."""
    def __init__(self, fault):
        self.fault = fault
        self._lock = threading.RLock()
        self._process = None
        self._identity = None
        self._last_received_at = None
        self._closed = False
        self._worker_requests = 0
        self._started_at = time.monotonic()
        self.request_timeout = 1
        self.max_worker_requests = 100
        self.max_worker_lifetime = 1000
        self.prediction_calls = 0
        self.close_calls = 0

    def _ensure_worker(self):
        if self.fault == 'identity':
            native._validate_identity({})
        elif self.fault == 'startup_timeout':
            raise native.C0WorkerTimeout('controlled startup deadline')
        if self._process is None:
            self._process = SimpleNamespace(pid=4321, poll=lambda: None)
            self._identity = {'fault_injection_only': True}

    def _exchange(self, frame, *, timeout):
        self._ensure_worker()
        self.prediction_calls += 1
        if self.fault == 'timeout':
            self._process = None  # Frozen client stops the process before raising.
            raise native.C0WorkerTimeout('controlled prediction deadline')
        return {}  # Real predict_evidence rejects this evidence schema.

    def close(self):
        self.close_calls += 1
        self._closed = True
        self._process = None


def native_inputs(tmp_path, monkeypatch, fault):
    paths, protocol = frozen_run(tmp_path, monkeypatch)
    protocol['hashes']['native_config'] = protocol['hashes']['raw_source']
    runner._json(paths['08_reporting'] / 'support.json', {'source_series': 1})
    runner._table(paths['08_reporting'] / 'origin_conflicts.parquet',
                  [{'match_id': '2', 'conflict_source_ids': [], 'native_consumed_prefix_length': 1}])
    events, _ = runner._source_maps(source(), {})
    earlier = source()
    earlier.update(match_id='10', date='2026-03-01')
    earlier['games'][0]['date'] = '2026-03-01'
    for side in ('t1', 't2'):
        for player in earlier['games'][0][side + '_players'].values():
            player['player_id'] = 'old-' + player['player_id']
    older_events, _ = runner._source_maps(earlier, {})
    events.extend(older_events)
    sides = []
    for side, team in (('A', 'A'), ('B', 'B')):
        state = runner.candidate_state(events, team_id=team, competition_context='League',
            cutoff_day='2026-04-20', target_match_id='2')
        sides.append({'match_id': '2', 'side': side, 'state': json.dumps(state)})
    series = [{'match_id': '2', 'partition': 'ASSESS', 'date': '2026-04-20',
        'team_a_id': 'A', 'team_b_id': 'B', 'best_of': 3, 'competition_context': 'League',
        'native_release_day': '2026-04-20', 'label_release_day': '2026-04-20'}]
    client = FaultBoundaryClient(fault)
    monkeypatch.setattr(native, 'C0InferenceClient', lambda *args, **kwargs: client)
    return paths, protocol, series, sides, client


@pytest.mark.parametrize('fault,predictions,sessions', [('identity', 0, 0), ('schema', 1, 1)])
def test_native_identity_and_evidence_schema_failures_are_fatal_and_retained(tmp_path, monkeypatch, fault, predictions, sessions):
    paths, protocol, series, sides, client = native_inputs(tmp_path, monkeypatch, fault)
    with pytest.raises(native.C0InferenceError):
        runner._capture_pairs(paths, protocol, series, sides)
    rows = pd.read_parquet(paths['07_model_output'] / 'native_pairs.parquet').to_dict('records')
    receipt = runner._load(paths['08_reporting'] / 'native_execution.json')
    assert rows[0]['status'] == 'NATIVE_PAIR_FATAL'
    assert rows[0]['failure_kind'] == 'NATIVE_INTEGRITY'
    assert receipt['native_worker_sessions'] == sessions
    assert receipt['native_request_count'] == predictions
    assert client.prediction_calls == predictions
    assert receipt['native_blocked_pairs'] == 3
    assert len(rows) == 1
    assert client._closed and receipt['worker_closed'] is True


def test_first_prediction_timeout_counts_established_worker_without_retry(tmp_path, monkeypatch):
    paths, protocol, series, sides, client = native_inputs(tmp_path, monkeypatch, 'timeout')
    _, statuses = runner._capture_pairs(paths, protocol, series, sides)
    receipt = runner._load(paths['08_reporting'] / 'native_execution.json')
    rows = pd.read_parquet(paths['07_model_output'] / 'native_pairs.parquet').to_dict('records')
    assert statuses == {'2': 'NATIVE_PAIR_UNAVAILABLE'}
    assert rows[0]['failure_kind'] == 'AVAILABILITY'
    assert receipt['native_worker_sessions'] == 1
    assert receipt['native_worker_recycles'] == 0
    assert receipt['worker_pids'] == [4321]
    assert receipt['native_request_count'] == client.prediction_calls == 1
    assert receipt['native_valid_pairs'] == 0
    assert receipt['native_blocked_pairs'] == 3
    assert receipt['state_advance_origin_count'] == 0
    assert receipt['state_advance_origin_attempt_count'] == 1
    assert receipt['state_advance_origin_unconfirmed_count'] == 1
    assert receipt['worker_sessions'][0]['closed_at'] is not None
    assert receipt['worker_startup_attempts'][0]['status'] == 'ESTABLISHED'
    assert client._closed


def test_startup_timeout_is_not_established_session_or_prediction(tmp_path, monkeypatch):
    paths, protocol, series, sides, client = native_inputs(tmp_path, monkeypatch, 'startup_timeout')
    runner._capture_pairs(paths, protocol, series, sides)
    receipt = runner._load(paths['08_reporting'] / 'native_execution.json')
    assert receipt['native_worker_sessions'] == receipt['native_request_count'] == 0
    assert receipt['native_worker_startup_failures'] == 1
    assert receipt['worker_startup_attempts'][0]['status'] == 'FAILED'
    assert client.prediction_calls == 0 and client._closed


def test_inherited_native_failure_is_never_requested_and_new_timeout_is_counted_separately(tmp_path, monkeypatch):
    paths, protocol, series, sides, client = native_inputs(tmp_path, monkeypatch, 'timeout')
    parent_pairs = [{'match_id': '2', 'candidate_a_index': 0, 'candidate_b_index': 0,
                     'status': 'NATIVE_PAIR_FATAL', 'failure_reason': 'receipt field bug',
                     'failure_kind': 'SOFTWARE', 'failure_type': 'AttributeError',
                     'native_requested': True, 'session_index': 0}]
    inherited = {'pairs': parent_pairs, 'blocked': [], 'execution': {'native_request_count': 1},
                 'no_retry_native_match_ids': ['2']}
    second = {**series[0], 'match_id': '3'}
    runner._table(paths['08_reporting'] / 'second_origin.parquet',
                  [{'match_id': '3', 'conflict_source_ids': [], 'native_consumed_prefix_length': 1}])
    origin = paths['08_reporting'] / 'origin_conflicts.parquet'
    pd.concat([pd.read_parquet(origin), pd.read_parquet(paths['08_reporting'] / 'second_origin.parquet')]).to_parquet(origin, index=False)
    _, statuses = runner._capture_pairs(paths, protocol, series + [second],
        sides + [{**side, 'match_id': '3', 'state': json.dumps({**json.loads(side['state']), 'target_match_id': '3'})}
                 for side in sides], inherited=inherited)
    rows = pd.read_parquet(paths['07_model_output'] / 'native_pairs.parquet').to_dict('records')
    receipt = runner._load(paths['08_reporting'] / 'native_execution.json')
    assert statuses['2'] == 'INHERITED_NATIVE_FAILURE_NO_RETRY'
    assert statuses['3'] == 'NATIVE_PAIR_UNAVAILABLE'
    assert client.prediction_calls == receipt['native_request_count'] == receipt['new_native_request_count'] == 1
    assert receipt['requests_by_series'] == {'3': 1}
    assert receipt['inherited_request_count'] == 1
    inherited_row = next(row for row in rows if row['match_id'] == '2')
    assert {key: inherited_row[key] for key in parent_pairs[0]} == parent_pairs[0]
    assert inherited_row['execution_scope'] == 'PARENT_INHERITED'
    assert client._closed



def recovery_parent(tmp_path):
    """Closed balanced TRAIN optimum stored analytically; no test fits."""
    import hashlib
    import numpy as np
    from src.models.lineup_law_reliability import lineup_objective, rows_to_arrays

    repo, research, paths = runner._roots('parent', tmp_path, tmp_path)
    for path in paths.values():
        path.mkdir(parents=True)
    runtime = {}
    pins = {}
    for relative in runner._RECOVERY_RUNTIME:
        current = repo / relative
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text('corrected runtime ' + relative)
        archived = repo / 'archive' / current.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_text('original runtime ' + relative)
        runtime[relative] = {'snapshot': str(archived.relative_to(repo)), 'sha256': runner._sha(archived)}
        pins[relative] = {'path': str(current), 'sha256': runner._sha(archived)}
    source_pin = repo / 'source.json'
    source_pin.write_text('{}')
    pins['raw_source'] = {'path': str(source_pin), 'sha256': runner._sha(source_pin)}
    for role in ('native_config', 'native_release_map', 'oracle_contract.json'):
        pins[role] = pins['raw_source']
    protocol = {'run_id': 'parent', 'repo_root': str(repo), 'research_root': str(research),
        'scope': 'HISTORICAL_DEVELOPMENT_ONLY', 'library_versions': {}, 'hashes': pins,
        'import_origins': {}, 'protected_before': {}, 'fit_budget': {**runner._ZERO_FITS, 'TRAIN': 2, 'restarts': 0},
        'population_ids': {'TRAIN': ['10', '11', '12'], 'CAL': ['20', '21', '22'], 'ASSESS': ['2', '3', '4']},
        'prefit_resolutions': {}, 'output_paths': {stage: str(path) for stage, path in paths.items()}}
    runner._json(paths['05_model_input'] / 'protocol.json', protocol)
    binding = {'protocol_sha256': runner._sha(paths['05_model_input'] / 'protocol.json'),
               'pins': pins, 'import_origins': {}, 'protected_before': {}}
    runner._json(paths['05_model_input'] / 'binding.json', binding)
    raw = source()
    events, _ = runner._source_maps(raw, {})
    older = deepcopy(raw)
    older['match_id'] = '9'
    older['date'] = older['games'][0]['date'] = '2026-03-01'
    for side in ('t1', 't2'):
        for player in older['games'][0][side + '_players'].values():
            player['player_id'] = 'old-' + player['player_id']
    older_events, _ = runner._source_maps(older, {})
    events.extend(older_events)
    series, opportunities = [], []
    for partition, ids in protocol['population_ids'].items():
        for index, mid in enumerate(ids):
            row = {'match_id': mid, 'partition': partition, 'date': '2026-04-20', 'month': '2026-04',
                'team_a_id': 'A', 'team_b_id': 'B', 'best_of': 3, 'competition_context': 'League',
                'native_release_day': '2026-04-20', 'label_release_day': '2026-04-20',
                'major': 'YES', 'player_source_unseen': 'NO', 'tournament_source_name': 'League',
                'history_state_a': 'partial', 'history_state_b': 'partial',
                'candidate_count_a': 2, 'candidate_count_b': 2,
                'incomplete_history_a': 0, 'incomplete_history_b': 0}
            series.append(row)
            for side in ('A', 'B'):
                state = runner.candidate_state(events, team_id=side, competition_context='League',
                    cutoff_day=row['date'], target_match_id=mid)
                opportunities.append({**row, 'side': side, 'label': runner.CLASS_ORDER[index],
                    'state': json.dumps(state), 'candidates': json.dumps(state['candidates']), 'absence': None,
                    'history_ids': ['1', '9'], 'incomplete_history_count': 0, 'history_state': 'partial',
                    'counts': [1., 1.], 'ages': [1., 1.], 'previous_available': True})
    runner._table(paths['04_feature'] / 'series.parquet', series)
    runner._table(paths['04_feature'] / 'opportunities.parquet', opportunities)
    runner._table(paths['08_reporting'] / 'origin_conflicts.parquet', [
        {'match_id': mid, 'conflict_source_ids': [], 'native_consumed_prefix_length': 1} for mid in ('2', '3', '4')])
    runner._json(paths['05_model_input'] / 'public_cases.json', [])
    runner._json(paths['05_model_input'] / 'public_case_contracts.json', {})
    prepared = {str(path): runner._sha(path) for path in (
        paths['04_feature'] / 'series.parquet', paths['04_feature'] / 'opportunities.parquet',
        paths['08_reporting'] / 'origin_conflicts.parquet', paths['05_model_input'] / 'public_cases.json',
        paths['05_model_input'] / 'public_case_contracts.json')}
    support = {'status': 'READY_FOR_TWO_TRAIN_FITS', 'protocol_sha256': binding['protocol_sha256'],
               'frozen_input_hashes': pins, 'prepared_hashes': prepared, 'counts': {'identity_release_valid': True},
               'source_series': 1}
    runner._json(paths['08_reporting'] / 'support.json', support)
    oracle = repo / 'prepare_oracle.json'
    runner._json(oracle, {'status': 'PASS_PREPARE_CONTRACT'})
    runner._json(paths['05_model_input'] / 'prefit_approval.json', {'status': 'APPROVED', 'run_id': 'parent',
        'protocol_sha256': binding['protocol_sha256'], 'binding_sha256': runner._sha(paths['05_model_input'] / 'binding.json'),
        'support_sha256': runner._sha(paths['08_reporting'] / 'support.json'), 'fit_budget': protocol['fit_budget'],
        'counts': support['counts'], 'independent_prepare_oracle': {'path': str(oracle.relative_to(repo)),
            'sha256': runner._sha(oracle), 'status': 'PASS_PREPARE_CONTRACT'}})
    train = [row for row in opportunities if row['partition'] == 'TRAIN']
    keys = [[row['match_id'], row['side']] for row in train]
    operations, laws = [], []
    for index, arm in enumerate(('conditional', 'global')):
        conditional = arm == 'conditional'
        theta = np.zeros(6 if conditional else 2)
        objective, gradient = lineup_objective(theta, *rows_to_arrays(train), conditional=conditional)
        artifact = {'accepted': True, 'success': True, 'conditional': conditional, 'theta': theta.tolist(),
            'iterations': 0, 'objective': objective, 'gradient': gradient.tolist(),
            'protocol_sha256': binding['protocol_sha256'], 'frozen_input_hashes': pins,
            'code_hashes': {}, 'training_keys': keys}
        path = paths['06_models'] / (arm + '.json')
        runner._json(path, artifact)
        operation = {'arm': arm, 'partition': 'TRAIN', 'status': 'COMPLETED',
            'counter_before': index, 'counter_after': index + 1, 'parameters_sha256': runner._sha(path),
            'input_ids_sha256': hashlib.sha256(json.dumps(keys).encode()).hexdigest()}
        operations.append(operation)
        laws.extend(runner._law_rows(opportunities, runner.LineupLaw(artifact), arm, runner._sha(path)))
    runner._table(paths['07_model_output'] / 'side_laws.parquet', laws)
    runner._table(paths['07_model_output'] / 'series_scores.parquet', runner._series_scores(series, laws))
    runner._json(paths['08_reporting'] / 'operation_trace.json', operations)
    runner._json(paths['08_reporting'] / 'cal_audit.json', {'class_support_valid': True})
    runner._json(paths['08_reporting'] / 'run_hard_stop.json', {'no_restart': True, 'failure_kind': 'SOFTWARE',
        'failure_type': 'AttributeError', 'failure_reason': "'C0EvidenceResult' object has no attribute 'received_at'",
        'identity_release_validation_status': 'VALID', 'formal_fits': 2, 'completed_fits': 2, 'status': 'INCONCLUSIVE'})
    runner._json(paths['08_reporting'] / 'decision.json', {'status': 'INCONCLUSIVE', 'operation_trace': operations})
    runner._table(paths['07_model_output'] / 'native_pairs.parquet', [{
        'match_id': mid, 'candidate_a_index': 0, 'candidate_b_index': 0, 'native_requested': True,
        'status': 'NATIVE_PAIR_FATAL' if mid == '3' else 'NATIVE_PAIR_UNAVAILABLE',
        'failure_type': 'AttributeError' if mid == '3' else 'C0WorkerTimeout',
        'failure_kind': 'SOFTWARE' if mid == '3' else 'AVAILABILITY',
        'failure_reason': 'receipt field bug' if mid == '3' else 'timeout', 'session_index': int(mid) - 2
    } for mid in ('2', '3')])
    runner._table(paths['07_model_output'] / 'native_blocked_pairs.parquet', [{
        'match_id': mid, 'candidate_a_index': a, 'candidate_b_index': b,
        'reason': 'NOT_REQUESTED_AFTER_NATIVE_PAIR_FAILURE', 'conflict_source_ids': [],
        'cutoff_day': '2026-04-20', 'native_requested': False
    } for mid in ('2', '3') for a, b in ((0, 1), (1, 0), (1, 1))])
    runner._json(paths['08_reporting'] / 'native_execution.json', {'native_request_count': 2,
        'native_valid_pairs': 0, 'raw_snapshots_persisted': 0, 'worker_closed': True})
    runner._oracle_index(repo, paths, protocol, 'INCONCLUSIVE', operations)
    authorization_path = repo / 'recovery_authorization.json'
    runner._json(authorization_path, {'schema': 'lineup_law_technical_recovery_authorization_v1',
        'run_id': 'recovery', 'parent_run_id': 'parent', 'inherited_TRAIN_fits': 2,
        'new_fit_budget': runner._ZERO_FITS, 'allowed_source_changes': list(runner._RECOVERY_RUNTIME),
        'old_runtime_snapshots': runtime, 'scientific_contract_changed': False,
        'extra_recipe_or_hyperparameter_choice': False, 'new_lineup_assessment_metrics_or_intervals_read': False,
        'parent_file_pins': {str(path.relative_to(repo)): runner._sha(path)
                            for directory in paths.values() for path in directory.rglob('*') if path.is_file()},
        'protected_before': {}, 'original_failed_requests': 2, 'no_retry_native_match_ids': ['2', '3']})
    return paths, authorization_path


def test_recovery_finishes_with_zero_fits_immutable_parent_and_saved_scores(tmp_path, monkeypatch):
    parent_paths, authorization = recovery_parent(tmp_path)
    parent_before = {str(path): path.read_bytes() for directory in parent_paths.values()
                     for path in directory.rglob('*') if path.is_file()}
    client = FaultBoundaryClient('timeout')
    monkeypatch.setattr(native, 'C0InferenceClient', lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, 'fit_lineup_law', lambda *args, **kwargs: pytest.fail('Recovery must never fit'))
    decision = runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                        repo_root=tmp_path, research_root=tmp_path)
    _, _, paths = runner._roots('recovery', tmp_path, tmp_path)
    assert decision['formal_fits'] == decision['completed_fits'] == decision['inherited_TRAIN_fits'] == 2
    assert decision['new_fit_counts'] == runner._ZERO_FITS
    assert runner._load(paths['08_reporting'] / 'new_operation_trace.json') == []
    assert all(Path(path).read_bytes() == content for path, content in parent_before.items())
    for stage, names in (('06_models', ('conditional.json', 'global.json')),
                         ('07_model_output', ('side_laws.parquet', 'series_scores.parquet')),
                         ('08_reporting', ('operation_trace.json',))):
        for name in names:
            assert (paths[stage] / name).read_bytes() == (parent_paths[stage] / name).read_bytes()
    execution = runner._load(paths['08_reporting'] / 'native_execution.json')
    assert execution['native_request_count'] == execution['new_native_request_count'] == client.prediction_calls == 1
    assert execution['requests_by_series'] == {'4': 1}
    assert execution['inherited_request_count'] == 2
    predictions = runner.predict_run('recovery', ['2', '3', '4'], repo_root=tmp_path, research_root=tmp_path)
    assert [row['bounds']['absence'] for row in predictions] == [
        'INHERITED_NATIVE_FAILURE_NO_RETRY', 'INHERITED_NATIVE_FAILURE_NO_RETRY', 'NATIVE_PAIR_UNAVAILABLE']
    assert all(row['bounds']['prob_a'] is None for row in predictions)
    assert all(row['side_laws'][0]['q'] is not None for row in predictions)
    before_repeat = files_under(tmp_path)
    with pytest.raises(FileExistsError):
        runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                  repo_root=tmp_path, research_root=tmp_path)
    with pytest.raises(ValueError, match='no TRAIN fit slots'):
        runner.run_study('recovery', repo_root=tmp_path, research_root=tmp_path)
    assert files_under(tmp_path) == before_repeat


def test_recovery_resumes_interrupted_run_and_reuses_existing_snapshots(tmp_path, monkeypatch):
    import hashlib
    parent_paths, authorization = recovery_parent(tmp_path)
    client = FaultBoundaryClient('timeout')
    monkeypatch.setattr(native, 'C0InferenceClient', lambda *args, **kwargs: client)
    monkeypatch.setattr(runner, 'fit_lineup_law', lambda *args, **kwargs: pytest.fail('Recovery must never fit'))
    _, _, paths = runner._roots('recovery', tmp_path, tmp_path)
    paths['07_model_output'].mkdir(parents=True)
    (paths['07_model_output'] / 'snapshots').mkdir()
    source_sha = runner._sha(tmp_path / 'source.json')
    for i in range(2):
        for j in range(2):
            snap_path = paths['07_model_output'] / 'snapshots' / (hashlib.sha256(b'4').hexdigest() + f'_{i}_{j}.json')
            snap_data = {
                'schema': 'lineup_law_native_capture_v1', 'match_id': '4', 'pair': [i, j],
                'request': {'roster_a': ['old-player1', 'player2', 'player3', 'player4', 'player5'],
                            'roster_b': ['old-player1', 'player2', 'player3', 'player4', 'player5'],
                            'competition_context': 'League', 'decision_at': '2026-04-20T00:00:00+00:00', 'mode': 'full'},
                'encoded_snapshot': {'raw': {}, 'hist': {}}, 'native_history_provenance': {'raw_history_sha256': source_sha,
                                                                      'release_map_sha256': source_sha,
                                                                      'history_series_consumed': 1,
                                                                      'feature_source_max_day': '2026-04-01'},
                'prediction': {'prob_a': 0.6, 'prob_b': 0.4, 'best_of': 3, 'model_name': 'Causal-C0',
                               'model_version': 'c0-native-2026-w32-e12-v1', 'feature_version': 'native-c0-history16-v1',
                               'map_prob_a': None, 'map_prob_b': None, 'epistemic_sigma_z': None, 'p_low_a': None, 'p_low_b': None,
                               'canonical_match_id': None, 'diagnostics': {'mode': 'full', 'probability_target': 'series_win_already_projected', 'input_hashes': {}}},
                'generated_at': '2026-04-20T00:00:01+00:00', 'received_at': '2026-04-20T00:00:02+00:00'
            }
            snap_path.write_text(json.dumps(snap_data))
    decision = runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                        repo_root=tmp_path, research_root=tmp_path)
    assert client.prediction_calls == 0
    assert decision['formal_fits'] == 2
    assert decision['status'] in ('NO_LINEUP_LAW_GAIN', 'LINEUP_LAW_DEVELOPMENT_SUPPORTED_NOT_WINNER_MODEL', 'INCONCLUSIVE')


@pytest.mark.parametrize('defect', ['model_bytes', 'prepared_bytes', 'archived_runtime', 'unapproved_science', 'budget'])
def test_recovery_rejects_provenance_defects_before_creating_namespace(tmp_path, defect):
    parent_paths, authorization = recovery_parent(tmp_path)
    if defect == 'model_bytes':
        (parent_paths['06_models'] / 'conditional.json').write_text('{}')
    elif defect == 'prepared_bytes':
        (parent_paths['04_feature'] / 'opportunities.parquet').write_bytes(b'changed')
    elif defect == 'archived_runtime':
        (tmp_path / 'archive/lineup_law_reliability.py').write_text('changed')
    elif defect == 'unapproved_science':
        (tmp_path / 'source.json').write_text('changed')
    else:
        value = runner._load(authorization)
        value['new_fit_budget']['TRAIN'] = 1
        authorization.write_text(json.dumps(value))
    before = files_under(tmp_path)
    with pytest.raises(ValueError):
        runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                  repo_root=tmp_path, research_root=tmp_path)
    assert files_under(tmp_path) == before
    assert not any(path.exists() for path in runner._roots('recovery', tmp_path, tmp_path)[2].values())


def test_fatal_recovery_hard_stop_retains_inherited_budget_without_touching_parent(tmp_path, monkeypatch):
    parent_paths, authorization = recovery_parent(tmp_path)
    before = {str(path): path.read_bytes() for directory in parent_paths.values()
              for path in directory.rglob('*') if path.is_file()}
    client = FaultBoundaryClient('schema')
    monkeypatch.setattr(native, 'C0InferenceClient', lambda *args, **kwargs: client)
    with pytest.raises(native.C0InferenceError):
        runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                  repo_root=tmp_path, research_root=tmp_path)
    paths = runner._roots('recovery', tmp_path, tmp_path)[2]
    stop = runner._load(paths['08_reporting'] / 'recover_evaluation_hard_stop.json')
    assert stop['formal_fits'] == stop['completed_fits'] == stop['inherited_TRAIN_fits'] == 2
    assert stop['new_fit_counts'] == runner._ZERO_FITS
    assert stop['no_restart'] is True
    assert stop['identity_release_validation_status'] == 'VALID'
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert client._closed


@pytest.mark.parametrize('defect', ['nonstationary_fit', 'unaccepted_fit', 'unapproved_fit', 'different_failure', 'retry_target'])
def test_recovery_rejects_invalid_semantics_even_when_artifact_hashes_are_reauthorized(tmp_path, defect):
    paths, authorization = recovery_parent(tmp_path)
    model_path = paths['06_models'] / 'conditional.json'
    if defect in ('nonstationary_fit', 'unaccepted_fit'):
        model = runner._load(model_path)
        if defect == 'nonstationary_fit':
            model['theta'][0] = .1
        else:
            model['accepted'] = False
        model_path.write_text(json.dumps(model))
        operations = runner._load(paths['08_reporting'] / 'operation_trace.json')
        operations[0]['parameters_sha256'] = runner._sha(model_path)
        (paths['08_reporting'] / 'operation_trace.json').write_text(json.dumps(operations))
        decision = runner._load(paths['08_reporting'] / 'decision.json')
        decision['operation_trace'] = operations
        (paths['08_reporting'] / 'decision.json').write_text(json.dumps(decision))
    elif defect == 'unapproved_fit':
        approval = runner._load(paths['05_model_input'] / 'prefit_approval.json')
        approval['status'] = 'REJECTED'
        (paths['05_model_input'] / 'prefit_approval.json').write_text(json.dumps(approval))
    elif defect == 'different_failure':
        stop = runner._load(paths['08_reporting'] / 'run_hard_stop.json')
        stop['failure_type'] = 'ValueError'
        (paths['08_reporting'] / 'run_hard_stop.json').write_text(json.dumps(stop))
    value = runner._load(authorization)
    value['parent_file_pins'] = {str(path.relative_to(tmp_path)): runner._sha(path)
                                for directory in paths.values() for path in directory.rglob('*') if path.is_file()}
    if defect == 'retry_target':
        value['no_retry_native_match_ids'] = ['3']
    authorization.write_text(json.dumps(value))
    before = files_under(tmp_path)
    with pytest.raises(ValueError):
        runner.recover_evaluation('recovery', parent_run_id='parent', authorization_path=authorization,
                                  repo_root=tmp_path, research_root=tmp_path)
    assert files_under(tmp_path) == before
