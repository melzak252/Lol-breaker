#!/usr/bin/env python3
"""Independent W1 raw-lineage, saved-weight and historical decision oracle.

Usage (each output MUST be a new directory)::
  python scripts/verify_champion_oov_reliability.py --mode prepare --run-id RUN --repo-root ROOT --research-root DATA --output FRESH_PATH
  python scripts/verify_champion_oov_reliability.py --mode completed --run-id RUN --repo-root ROOT --research-root DATA --output FRESH_PATH

No training, optimizer update, candidate feature builder or candidate forward is
used to construct expected values. Public predict/transform calls are actuals
under test only. The unchanged source parser is an explicitly shared dependency;
inherited opponent-strength features are pinned input constants, not newly
independently reconstructed ratings. All failures retain the frozen tolerances.

An optional --evidence-root must equal the resolved root frozen in protocol.json;
otherwise the oracle recovers that root. --mode completed audits the persisted
terminal branch: zero-fit closure, partial failed prefix, or complete execution.
PASS_PARTIAL_FAILURE_CLOSURE certifies retained evidence only, never adoption or
unavailable failed-fit numerical progress. Epoch totals count recorded progress;
the attempted twelve-epoch plans and fixed108 ceiling are reported separately.
"""
from __future__ import annotations

import argparse
from bisect import insort
from collections import Counter, defaultdict
import ast
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import unicodedata
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from scipy.special import logsumexp
from scipy.stats import rankdata

CHANNELS = ('champion', 'patch', 'competition')
ARRAY_KEYS = dict(zip(CHANNELS, ('champions', 'patches', 'competitions')))
SEEDS = (20261007, 20261008, 20261009)
ARMS = ('H', 'C', 'HC')
MONTHS = tuple(f'2026-{m:02d}' for m in range(1, 10))
NUMERIC = tuple(range(17)) + (18, 19)
NAMES = ('kills', 'deaths', 'assists', 'csm', 'gpm', 'dpm', 'vspm', 'gd@15', 'csd@15', 'xpd@15', 'kp%', 'dmg%', 'kills_for', 'kills_against', 'total_kills', 'duration_minutes', 'prior_opponent_strength_mean', 'prior_opponent_strength_sd', 'played_age_days', 'release_age_days')
TOLS = {'exact': (0., 0.), 'normalizer': (1e-10, 1e-10), 'phi': (2e-6, 0.), 'probability': (2e-6, 0.), 'logit': (2e-6, 2e-6), 'metric': (1e-10, 1e-10), 'replayed_score': (2e-5, 2e-5), 'gradient': (2e-6, 2e-5), 'objective': (1e-8, 0.)}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def held(channel, identity):
    return int.from_bytes(hashlib.sha256((channel + ':' + identity).encode()).digest(), 'big') % 5 == 0


class Audit:
    def __init__(self):
        self.checks = []
        self.failures = []

    def truth(self, name, condition, reason=None, scope='all', kind='exact'):
        entry = dict(check_id=name, kind=kind, scope=scope, input_reference=name,
                     independent_method='Independent source or contract assertion', expected_reference='frozen contract',
                     observed_reference=name, compared_values=1, tolerance_kind=kind,
                     maximum_absolute_difference=None, maximum_relative_difference=None,
                     missing_mask_mismatches=0, status='PASS' if condition else 'FAIL', reason=reason,
                     failure_examples=[])
        self.checks.append(entry)
        if not condition:
            self.failures.append(entry)

    def compare(self, name, actual, expected, kind='exact'):
        a, e = np.asarray(actual), np.asarray(expected)
        entry = dict(check_id=name, kind=kind, scope='every indexed value', input_reference=name,
                     independent_method='Independent arithmetic / raw-source alignment', expected_reference='oracle arrays',
                     observed_reference=name, compared_values=int(e.size), tolerance_kind=kind,
                     maximum_absolute_difference=None, maximum_relative_difference=None,
                     missing_mask_mismatches=0, status='FAIL', reason=None, failure_examples=[])
        if a.shape != e.shape:
            entry['reason'] = f'Shape mismatch: {a.shape} != {e.shape}'
        elif a.dtype.kind in 'biufc' and e.dtype.kind in 'biufc':
            atol, rtol = TOLS[kind]
            am, em = np.isnan(a), np.isnan(e)
            entry['missing_mask_mismatches'] = int(np.count_nonzero(am != em))
            finite = np.isfinite(a) & np.isfinite(e)
            delta = np.abs(a[finite].astype(float) - e[finite].astype(float))
            entry['maximum_absolute_difference'] = float(delta.max(initial=0))
            entry['maximum_relative_difference'] = float(np.max(delta / np.maximum(np.abs(e[finite].astype(float)), 1e-300), initial=0))
            matches = np.isclose(a, e, atol=atol, rtol=rtol, equal_nan=True)
            entry['status'] = 'PASS' if matches.all() else 'FAIL'
            entry['mismatches'] = int(np.count_nonzero(~matches))
            entry['failure_examples'] = [dict(index=i.tolist(), actual=clean(a[tuple(i)]), expected=clean(e[tuple(i)])) for i in np.argwhere(~matches)[:12]]
        else:
            matches = a == e
            entry['status'] = 'PASS' if matches.all() else 'FAIL'
            entry['mismatches'] = int(np.count_nonzero(~matches))
            entry['failure_examples'] = [dict(index=i.tolist(), actual=clean(a[tuple(i)]), expected=clean(e[tuple(i)])) for i in np.argwhere(~matches)[:12]]
        self.checks.append(entry)
        if entry['status'] == 'FAIL':
            self.failures.append(entry)


class Inputs:
    def __init__(self, repo, research, run_id, audit):
        self.repo, self.research, self.run_id, self.audit = repo, research, run_id, audit
        self._verifier_archive = None
        self.hashes = {}
        self.index_path = self.path(f'data/08_reporting/champion_oov_reliability/{run_id}/oracle_inputs.json')
        self.index = load_json(self.index_path)
        self.descriptors = self.index['artifact_descriptors']
        if isinstance(self.descriptors, dict):
            self.descriptors = [dict(v, role=k) for k, v in self.descriptors.items()]
        required_index = ('schema_version','study','run_id','scenario','branch_status','created_at_utc','protocol_sha256','binding_sha256','oracle_contract_sha256','artifact_descriptors','population_ids','operation_trace','missingness_ledger','public_cases','prefit_resolutions')
        audit.truth('index_schema', all(k in self.index for k in required_index))
        audit.truth('index_identity', self.index.get('study') == 'W1' and self.index.get('run_id') == run_id)

    def set_verifier_lineage(self, manifest_path, evidence):
        """Replay frozen verifier provenance without repinning any study input."""
        manifest_path = Path(manifest_path).resolve()
        if manifest_path != Path(evidence).resolve() / 'verifier_lineage.json':
            raise ValueError('Verifier lineage must use the bound evidence manifest')
        lineage = load_json(manifest_path)
        relative = 'scripts/verify_champion_oov_reliability.py'
        current = (self.repo / relative).resolve()
        archive = (self.repo / lineage['archived_source']).resolve()
        if (lineage.get('schema') != 'w1_verifier_lineage_v1'
                or lineage.get('run_id') != self.run_id
                or lineage.get('allowed_source') != relative
                or lineage.get('scientific_changes') is not False
                or not archive.is_relative_to(Path(evidence).resolve())
                or current != Path(__file__).resolve()
                or digest(current) != lineage.get('current_sha256')):
            raise ValueError('Verifier-only lineage scope or current source differs')
        if self.index['protocol_sha256'] is None:
            pins = load_json(self.artifact('stage0', self.stage('08_reporting', 'stage0.json')))['initial_pins']
        else:
            pins = load_json(self.artifact('binding', self.stage('05_model_input', 'binding.json')))['pins']
        if (pins.get(relative) != lineage.get('frozen_sha256')
                or digest(archive) != pins.get(relative)):
            raise ValueError('Archived verifier differs from the original frozen pin')
        self._verifier_archive = archive
        return dict(lineage, manifest_sha256=digest(manifest_path),
                    scope='Only the verifier source resolves to its exact archived bytes; all other input pins remain live-checked')

    def validate_descriptors(self):
        audit = self.audit
        for d in self.descriptors:
            required = ('role', 'repo_relative_path', 'sha256', 'format', 'shape_or_row_count', 'dtype_or_columns', 'units', 'ordered_axis_names', 'ordered_id_axis_reference', 'absence_status', 'absence_reason')
            audit.truth('descriptor_schema:' + d.get('role', '?'), all(k in d for k in required))
            if d.get('absence_status') in ('AVAILABLE', 'PRESENT', None) and d.get('repo_relative_path'):
                p = self.path(d['repo_relative_path'])
                value = digest(p)
                self.hashes[str(p)] = value
                audit.truth('descriptor_sha:' + d['role'], value == d['sha256'])

    def path(self, relative):
        p = Path(relative)
        resolved = p if p.is_absolute() else next((root / p for root in (self.repo, self.research) if (root / p).exists()), self.repo / p)
        if (self._verifier_archive is not None
                and resolved.resolve() == (self.repo / 'scripts/verify_champion_oov_reliability.py').resolve()):
            return self._verifier_archive
        return resolved

    def artifact(self, role, fallback=None):
        entries = [d for d in self.descriptors if d.get('role') == role]
        if len(entries) == 1 and entries[0].get('repo_relative_path'):
            return self.path(entries[0]['repo_relative_path'])
        if fallback is not None:
            p = self.path(fallback)
            if p.is_file():
                self.audit.truth('descriptor_present:' + role, any(d.get('repo_relative_path') == fallback for d in self.descriptors))
                return p
        raise ValueError('Required artifact missing or ambiguous: ' + role)

    def available(self, role, fallback):
        entries = [d for d in self.descriptors if d.get('role') == role and d.get('repo_relative_path')]
        path = self.path(entries[0]['repo_relative_path']) if len(entries) == 1 else self.path(fallback)
        if path.is_file():
            return self.artifact(role, fallback)
        return None

    def stage(self, stage, filename):
        return f'data/{stage}/champion_oov_reliability/{self.run_id}/{filename}'


def resolve_evidence(inputs, explicit=None):
    """Recover the single evidence authority before any source/model loads."""
    if inputs.index['protocol_sha256'] is None:
        root = Path(explicit) if explicit is not None else Path('data/08_reporting/sports_reliability_improvements') / inputs.run_id
        if not root.is_absolute():
            root = inputs.repo / root
        inputs.evidence = root.resolve()
        inputs.contract = load_json(inputs.evidence / 'w1_contract.json')
        return inputs.evidence
    protocol_path = inputs.artifact('protocol', inputs.stage('05_model_input', 'protocol.json'))
    binding_path = inputs.artifact('binding', inputs.stage('05_model_input', 'binding.json'))
    protocol, binding = load_json(protocol_path), load_json(binding_path)
    if (digest(protocol_path) != inputs.index['protocol_sha256']
            or digest(binding_path) != inputs.index['binding_sha256']
            or digest(protocol_path) != binding['protocol_sha256']):
        raise ValueError('frozen protocol/binding hash mismatch')
    root = Path(protocol['evidence_root'])
    if not root.is_absolute():
        raise ValueError('frozen evidence root must be absolute')
    root = root.resolve()
    supplied = Path(explicit) if explicit is not None else root
    if not supplied.is_absolute():
        supplied = inputs.repo / supplied
    if supplied.resolve() != root:
        raise ValueError('supplied evidence root differs from frozen evidence root')
    contract_path = root / 'w1_contract.json'
    contract = load_json(contract_path)
    pins = binding['pins']
    locator = str(contract_path.relative_to(inputs.repo)) if contract_path.is_relative_to(inputs.repo) else str(contract_path)
    if contract != protocol['w1_contract'] or pins.get(locator) != digest(contract_path):
        raise ValueError('evidence contract differs from frozen contract or pin')
    inputs.evidence, inputs.contract = root, protocol['w1_contract']
    return root


def historical_pin_checks(inputs, audit):
    paths = inputs.contract['input_paths']
    scoring = load_json(inputs.path(paths['old_scoring_binding']))
    decision = load_json(inputs.path(paths['old_decision']))
    for locator in paths['saved_d0_models'] + [paths['saved_d0_ensemble']]:
        actual = digest(inputs.path(locator))
        audit.truth('historical_D0_model:' + locator,
                    actual == scoring['additional_pins'].get(locator)
                    and actual == scoring['models_before_scoring'].get(locator))
    for key in ('saved_d0_cal_outputs', 'saved_d0_assessment_outputs', 'saved_c0_assessment_outputs'):
        for locator in paths[key]:
            audit.truth('historical_reference_export:' + locator,
                        digest(inputs.path(locator)) == decision['output_hashes'].get(locator))


def operation_trace(inputs):
    trace = inputs.index['operation_trace']
    if isinstance(trace, str):
        trace = load_json(inputs.path(trace))
    if isinstance(trace, dict):
        trace = trace.get('operation_trace', trace.get('operations', trace.get('trace', [])))
    return trace


def execution_branch(inputs, audit):
    branch = inputs.index['branch_status']
    execution_path=inputs.path(inputs.stage('08_reporting','execution.json'))
    if branch in ('SUPPORT_PASS_PENDING_FREEZE', 'FROZEN_BEFORE_FIT'):
        audit.truth('prepare_no_execution_attempt',not execution_path.exists())
        return 'PREPARE'
    decision = load_json(inputs.artifact('decision', inputs.stage('08_reporting', 'decision.json')))
    if branch in ('CLOSED_BY_SUPPORT', 'CLOSED_NO_FITS', 'HARD_STOP_BEFORE_FIT'):
        audit.truth('terminal_execution_decision_coherent', decision['status'] == branch and decision['decision'] == 'NO_ADOPTION')
        if execution_path.is_file():
            execution=load_json(inputs.artifact('operation_trace',inputs.stage('08_reporting','execution.json')))
            audit.truth('zero_fit_execution_matches_branch',execution['status']==branch and not execution['operation_trace'])
            audit.truth('execution_trace_matches_index',execution['operation_trace']==operation_trace(inputs))
        return 'ZERO_FIT'
    execution = load_json(inputs.artifact('operation_trace', inputs.stage('08_reporting', 'execution.json')))
    audit.truth('execution_trace_matches_index', execution['operation_trace'] == operation_trace(inputs))
    if execution['status'] == 'FIT_FAILED_WITHIN_BUDGET':
        audit.truth('terminal_execution_decision_coherent',
                    branch == execution['status'] == decision['status'] and decision['decision'] == 'NO_ADOPTION')
        return 'PARTIAL_FAILED'
    audit.truth('terminal_execution_decision_coherent',
                execution['status'] == 'COMPLETED' and branch == decision['status']
                and decision['status'] in ('DEVELOPMENT_CANDIDATE', 'NO_ADOPTION')
                and decision['decision'] == decision['status'])
    return 'COMPLETED'


def terminal_integrity_checks(inputs, audit, authorization, branch):
    path = inputs.path(inputs.stage('08_reporting','final_integrity.json'))
    if branch == 'COMPLETED':
        integrity = load_json(path)
        audit.truth('terminal_final_integrity_receipt',
                    integrity.get('protected_after') == authorization['protected_before'])
    elif branch == 'PARTIAL_FAILED':
        execution = load_json(inputs.artifact('operation_trace',inputs.stage('08_reporting','execution.json')))
        decision = load_json(inputs.artifact('decision',inputs.stage('08_reporting','decision.json')))
        audit.truth('terminal_failure_reason_recorded',
                    execution.get('failure_phase') in ('FIT_OR_EXPORT','FINAL_INTEGRITY')
                    and bool(execution.get('failure_reason'))
                    and execution.get('failure_phase') == decision.get('failure_phase')
                    and execution.get('failure_reason') == decision.get('failure_reason'))
        audit.truth('terminal_failed_integrity_has_no_success_receipt',not path.exists())


def retained_operations(inputs, audit):
    members, scalars = [], []
    for op in operation_trace(inputs):
        complete=op['status'] in ('COMPLETED','SUCCESS')
        arm, seed = op['arm'], op['seed']
        if not complete:
            scalar_path=inputs.path(inputs.stage('06_models',arm+'_temperature.json'))
            if op['status']!='FAILED' or op['operation']!='W1_TERMINAL_CAL' or not scalar_path.is_file():
                continue
        if op['operation'] == 'W1_NEURAL_FIT':
            prefix = f'{arm}_seed_{seed}'
            receipt = load_json(inputs.artifact(prefix + '_fit_receipt', inputs.stage('06_models', prefix + '_fit_receipt.json')))
            audit.truth('retained_receipt:' + prefix, receipt == op['fit_receipt'] and receipt['epochs_completed'] == 12)
            members.append((arm, seed))
        elif op['operation'] == 'W1_TERMINAL_CAL':
            receipt = load_json(inputs.artifact(arm + '_temperature', inputs.stage('06_models', arm + '_temperature.json')))
            if complete or op.get('receipt') is not None:
                audit.truth('retained_scalar_receipt:' + arm,receipt == op.get('receipt'))
            scalars.append(arm)
    return members, scalars


def failed_export_state_checks(inputs, arrays, series, support, vocab, stats, audit):
    """Audit standalone state/receipt files saved before an export failure."""
    train=np.flatnonzero(series.split.to_numpy()=='TRAIN')
    states_count=0
    for op in operation_trace(inputs):
        if op['status']!='FAILED':
            continue
        arm,seed=op['arm'],op['seed']
        if op['operation']=='W1_TERMINAL_CAL':
            path=inputs.available(arm+'_ensemble',inputs.stage('06_models',arm+'_ensemble.joblib'))
            if path is None:
                inputs.unavailable_comparisons.append(arm+'_ensemble')
                continue
            ensemble=joblib.load(path)
            temperature=float(ensemble.temperature)
            audit.truth('failed_export_scalar_temperature:'+arm,math.isfinite(temperature) and .25<=temperature<=4)
            audit.truth('failed_export_ensemble_members:'+arm,len(ensemble.models)==3)
            cal=np.flatnonzero(series.split.to_numpy()=='CAL')
            laws=[]
            for member,member_seed in zip(ensemble.models,SEEDS):
                prefix=f'{arm}_seed_{member_seed}'
                state=load_state(inputs.artifact(prefix+'_final_state',inputs.stage('06_models',prefix+'_final_state.pt')))
                for key,value in state.items():
                    audit.compare('failed_export_ensemble_state:'+prefix+':'+key,member.state_dict[key].cpu().numpy(),value.cpu().numpy())
                law=np.empty((len(cal),6))
                for begin in range(0,len(cal),256):
                    law[begin:begin+256]=replay(state,arm,arrays,cal[begin:begin+256],stats,support,vocab)[0]
                laws.append(temper(law,temperature))
            actual=ensemble.predict(arrays,cal,identity_state={'n_prior':support}).score_probs
            audit.compare('failed_export_ensemble_replay:'+arm,actual,np.mean(laws,axis=0),'probability')
            if inputs.available(arm+'_temperature',inputs.stage('06_models',arm+'_temperature.json')) is None:
                inputs.unavailable_comparisons.append(arm+'_scalar_objective_receipt')
            continue
        if op['operation']!='W1_NEURAL_FIT':
            continue
        prefix=f'{arm}_seed_{seed}'
        expected=None
        retained={}
        for suffix in ('initial_state','final_state'):
            path=inputs.available(prefix+'_'+suffix,inputs.stage('06_models',prefix+'_'+suffix+'.pt'))
            if path is None:
                inputs.unavailable_comparisons.append(prefix+'_'+suffix)
                continue
            if expected is None:
                expected=independent_initial_state(arrays,train,pd.read_parquet(inputs.artifact('maps')),seed,arm)
            state=load_state(path)
            retained[suffix]=state
            states_count+=int(suffix=='final_state')
            audit.truth('failed_export_state_keys:'+prefix+':'+suffix,set(state)==set(expected))
            for key,value in state.items():
                audit.truth('failed_export_state_shape:'+prefix+':'+suffix+':'+key,
                            key in expected and value.shape==expected[key].shape
                            and value.dtype==expected[key].dtype
                            and (not value.is_floating_point() or torch.isfinite(value).all().item()))
                if suffix=='initial_state' and key in expected:
                    audit.compare('failed_export_initial_stream:'+prefix+':'+key,value.cpu().numpy(),expected[key].cpu().numpy())
        receipt_path=inputs.available(prefix+'_fit_receipt',inputs.stage('06_models',prefix+'_fit_receipt.json'))
        optimizer_path=inputs.available(prefix+'_optimizer',inputs.stage('06_models',prefix+'_optimizer.pt'))
        if receipt_path is not None:
            receipt=load_json(receipt_path)
            audit.truth('failed_export_receipt:'+prefix,receipt==op.get('fit_receipt'))
            if receipt.get('epochs_completed')==12 and optimizer_path is not None and 'final_state' in retained:
                trajectory_check(inputs,arm,seed,train,audit)
            else:
                inputs.unavailable_comparisons.append(prefix+'_complete_optimizer_trajectory')
        else:
            inputs.unavailable_comparisons.append(prefix+'_fit_receipt')
        model_path=inputs.available(prefix,inputs.stage('06_models',prefix+'.joblib'))
        if model_path is not None and 'final_state' in retained:
            model=joblib.load(model_path)
            for key,value in retained['final_state'].items():
                audit.compare('failed_export_joblib_state:'+prefix+':'+key,model.network.state_dict()[key].cpu().numpy(),value.cpu().numpy())
        else:
            inputs.unavailable_comparisons.append(prefix+'_joblib_state_correspondence')
    return states_count


def series_order(frame, series, audit, name):
    audit.compare(name + ':ordered_ids', frame['match_id'].astype(str).to_numpy(), series['match_id'].astype(str).to_numpy())
    audit.truth(name + ':unique_ids', not frame['match_id'].astype(str).duplicated().any())
    for col in ('date', 'best_of'):
        if col in frame:
            audit.compare(name + ':' + col, frame[col].to_numpy(), series[col].to_numpy())


def raw_observations(inputs, canonical, source, overrides):
    """Bounded use of the explicitly permitted unchanged low-level parser.

    Parser receives all canonical release constraints but no production prepared
    data. Chunk-level parser sorting is not relied upon for global event order.
    """
    from src.features.champion_history_data import _read_source, iter_matches
    targets = {str(r['match_id']): r for r in canonical.to_dict('records')}
    all_match, all_game, all_observation = set(), set(), set()
    observations, series_meta, vocab_candidates = [], [], {c: [] for c in CHANNELS}
    source_labels, map_labels = {}, []
    with tempfile.TemporaryDirectory(prefix='w1_oracle_source_') as temp:
        chunk = []
        def consume(rows):
            path = Path(temp) / 'parser_chunk.json'
            chunk_observations = []
            path.write_text(json.dumps(rows, allow_nan=True))
            records, _ = _read_source(path, targets, overrides)
            for record in records:
                if record.match_id in all_match:
                    raise ValueError('Duplicate raw series: ' + record.match_id)
                all_match.add(record.match_id)
                series_meta.append((record.day.isoformat(), record.match_id, record.release.isoformat()))
                if record.match_id in targets:
                    score = record.score_index
                    if targets[record.match_id]['team_a'] != record.teams[0] and math.isfinite(score):
                        score = score+3 if score<3 else score-3
                    source_labels[record.match_id] = dict(score_index=score,winner=float(score<3) if math.isfinite(score) else math.nan,score_reason=record.score_reason)
                for event in record.maps:
                    if event.game_id in all_game:
                        raise ValueError('Duplicate raw game: ' + event.game_id)
                    all_game.add(event.game_id)
                    if record.match_id in targets:
                        map_labels.append(dict(match_id=record.match_id,game_id=event.game_id,map_index=event.map_index,
                                              kills=event.labels['kills'],duration_minutes=event.labels['duration_minutes'],
                                              kills_reason=event.labels['kills_reason'],duration_reason=event.labels['duration_reason']))
                    if event.history_eligible and record.day.isoformat() < '2025-07-01' and record.release.isoformat() < '2025-07-01':
                        for identity in sorted({v for v in event.champions if v is not None}):
                            vocab_candidates['champion'].append((record.day.isoformat(), record.match_id, event.map_index, identity))
                        for c, identity in (('patch', event.patch), ('competition', event.competition)):
                            if identity is not None:
                                vocab_candidates[c].append((record.day.isoformat(), record.match_id, event.map_index, identity))
                    for slot, player in enumerate(event.players):
                        if player is None:
                            continue
                        oid = event.game_id + ':' + player
                        if oid in all_observation:
                            raise ValueError('Duplicate source player-map: ' + oid)
                        all_observation.add(oid)
                        row = dict(observation_id=oid, match_id=record.match_id, game_id=event.game_id, player_id=player,
                                   date=record.day.isoformat(), played_day=date.fromordinal(event.played).isoformat(),
                                   release_day=record.release.isoformat(), history_eligible=bool(event.history_eligible),
                                   champion=event.champions[slot], patch=event.patch, competition=event.competition,
                                   champion_status=int(event.champion_status[slot]), role=int(event.roles[slot]),
                                   map_index=event.map_index, source_slot=slot)
                        row.update({name: float(event.numeric[slot, i]) for i, name in enumerate(NAMES[:16])})
                        chunk_observations.append(row)
            frame_chunk = pd.DataFrame(chunk_observations)
            for name in ('observation_id','match_id','game_id','player_id','date','played_day','release_day',*CHANNELS):
                frame_chunk[name] = frame_chunk[name].astype('string[pyarrow]')
            for name in NAMES[:16]:
                frame_chunk[name] = frame_chunk[name].astype(np.float32)
            observations.append(frame_chunk)
            path.unlink()
        for raw in iter_matches(source):
            chunk.append(raw)
            if len(chunk) == 128:
                consume(chunk)
                chunk.clear()
        if chunk:
            consume(chunk)
    key = lambda identity: (0, int(identity)) if identity.isascii() and identity.isdecimal() else (1, identity)
    ranks = {mid: rank for rank, (_, mid, _) in enumerate(sorted(series_meta, key=lambda r: (r[0], key(r[1]))))}
    frame = pd.concat(observations, ignore_index=True)
    frame['source_rank'] = frame['match_id'].map(ranks).astype('int64')
    vocabularies = {}
    for c in CHANNELS:
        v = {}
        for _, mid, mi, identity in sorted(vocab_candidates[c], key=lambda r: (r[0], key(r[1]), r[2], r[3])):
            v.setdefault(identity, len(v) + 2)
        vocabularies[c] = v
    return frame, vocabularies, source_labels, pd.DataFrame(map_labels)


def verify_token_positions(path, event_ix, mask, audit, *, batch_size=65536):
    """Check every token without retaining the multi-million-row position table."""
    columns = ['row_index', 'side', 'player_slot', 'history_slot', 'event_index']
    seen = np.zeros(event_ix.size, dtype=np.uint16)
    shape = np.asarray(event_ix.shape, dtype=np.int64)
    rows, unique = 0, True
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
        values = batch.to_pandas().to_numpy(np.int64)
        positions = values[:, :4]
        # Indexing first retains NumPy's existing out-of-range/negative-index semantics.
        actual = event_ix[tuple(positions.T)]
        flat = np.ravel_multi_index(tuple((positions % shape).T), event_ix.shape)
        signs = np.sum((positions < 0) * (1 << np.arange(4)), axis=1)
        bits = np.left_shift(np.uint16(1), signs.astype(np.uint16))
        keys = flat * 16 + signs
        unique &= len(np.unique(keys)) == len(keys) and not np.any(seen[flat] & bits)
        np.bitwise_or.at(seen, flat, bits)
        audit.compare(f'token_event_positions:{rows}', actual, values[:, 4])
        rows += len(values)
    audit.truth('token_rows_all_valid', rows == int(mask.sum()))
    audit.truth('token_positions_unique', bool(unique))


def verify_source(inputs, arrays, series, schema, audit, output):
    contract = inputs.contract
    paths = contract['input_paths']
    canonical = pd.read_parquet(inputs.path(paths['canonical_base']))
    canonical['match_id'] = canonical['match_id'].astype(str)
    series_order(canonical, series, audit, 'canonical')
    for column in ('release_day','team_a','team_b'):
        audit.compare('canonical_metadata:'+column,canonical[column].astype(str).to_numpy(),series[column].astype(str).to_numpy())
    for column in ('roster_a','roster_b'):
        audit.compare('canonical_roster:'+column,np.asarray(canonical[column].tolist()),np.asarray(series[column].tolist()))
    # The old helper accepts either the correction map or its recorded wrapper.
    from src.features.champion_history_data import _release_overrides
    overrides = _release_overrides((inputs.research, inputs.repo))
    raw, vocab, source_labels, raw_map_labels = raw_observations(inputs, series, inputs.path(paths['raw_source']), overrides)
    for field in ('winner','score_index'):
        expected = np.asarray([source_labels.get(mid,{}).get(field,math.nan) for mid in series.match_id])
        audit.compare('raw_supervised:'+field,series[field].to_numpy(float),expected)
    expected_reasons = [source_labels.get(mid,{}).get('score_reason','source_series_missing') for mid in series.match_id]
    audit.compare('raw_supervised:score_reason',series.score_reason.to_numpy(),expected_reasons)
    saved_maps = pd.read_parquet(inputs.artifact('maps'))
    raw_map_labels = raw_map_labels.sort_values(['match_id','map_index']).reset_index(drop=True)
    saved_maps = saved_maps.sort_values(['match_id','map_index']).reset_index(drop=True)
    audit.compare('map_universe',saved_maps[['match_id','game_id','map_index']].astype(str).to_numpy(),raw_map_labels[['match_id','game_id','map_index']].astype(str).to_numpy())
    for field in ('kills','duration_minutes'):
        audit.compare('raw_map_labels:'+field,saved_maps[field].to_numpy(float),raw_map_labels[field].to_numpy(float))
    for field in ('kills_reason','duration_reason'):
        if field in saved_maps:
            audit.compare('raw_map_reasons:'+field,saved_maps[field].to_numpy(),raw_map_labels[field].to_numpy())
    exported = pd.read_parquet(inputs.artifact('source_observations', inputs.stage('04_feature','source_observations.parquet')))
    audit.truth('source_observation_unique', not exported['observation_id'].duplicated().any())
    audit.truth('source_observation_all_ids', set(exported['observation_id']) == set(raw['observation_id']))
    if set(exported['observation_id']) != set(raw['observation_id']):
        raise ValueError('Source observation universe mismatch; no intersection repair permitted')
    exported = exported.set_index('observation_id').loc[raw['observation_id']].reset_index()
    for col in ('match_id','game_id','player_id','date','played_day','release_day','history_eligible','champion','patch','competition','champion_status','role','source_rank','map_index','source_slot'):
        audit.compare('source:' + col, exported[col].fillna('<MISSING>').to_numpy(), raw[col].fillna('<MISSING>').to_numpy())
    for name in NAMES[:16]:
        audit.compare('source_numeric:' + name, exported[name].to_numpy(float), raw[name].to_numpy(float))
    audit.truth('source_release_bounds', ((raw['release_day'] >= raw['date']) & (raw['release_day'] >= raw['played_day'])).all())
    for c in CHANNELS:
        audit.truth('closed_vocab:' + c, schema[c + '_vocabulary'] == vocab[c])
        values = raw[(raw.date < '2025-07-01') & (raw.release_day < '2025-07-01') & raw.history_eligible]
        support = values.groupby(c, dropna=True).size().to_dict()
        write_json(output / (c + '_TRAIN_source_support.json'), dict(unique_source_observations=support,
                   missing_source_observations=int(values[c].isna().sum()), missing_state_trained=bool(values[c].isna().any()),
                   held_identities=sorted(i for i in vocab[c] if held(c, i))))
    identity_schema = load_json(inputs.artifact('identity_schema', inputs.stage('04_feature','identity_schema.json')))
    tables = identity_schema['canonical_id_tables']
    support_npz = np.load(inputs.artifact('support_arrays', inputs.stage('04_feature','support_arrays.npz')), allow_pickle=False)
    event_ix = support_npz['event_index']
    audit.compare('event_index_mask', event_ix >= 0, arrays['mask'])
    audit.truth('padding_event_minus_one', (event_ix[~arrays['mask']] == -1).all())
    audit.truth('event_indices_unique', exported['event_index'].is_unique)
    event_lookup = exported.set_index('event_index')
    del exported, canonical, saved_maps, raw_map_labels, source_labels
    verify_token_positions(inputs.artifact('identity_tokens', inputs.stage('04_feature','identity_tokens.parquet')),
                           event_ix, arrays['mask'], audit)
    # Source observations contain each player-map only once. Causal counts use a
    # sorted release vector per identity, never duplicated window occurrences.
    release_vectors = {}
    valid_raw = raw[raw.history_eligible]
    for c in CHANNELS:
        release_vectors[c] = {identity: np.sort(g.release_day.to_numpy(str)) for identity, g in valid_raw.groupby(c)}
    oid_to_event = dict(zip(event_lookup.observation_id, event_lookup.index))
    groups = {k: np.zeros(len(series), bool) for k in ('natural_champion_oov','seen_champion_history','missing_champion_identity','cold_history','patch_oov','competition_oov')}
    counts = {c: {s: np.zeros(len(series), np.int64) for s in ('known','oov','missing')} for c in CHANNELS}
    support = {c: support_npz['n_prior_' + c] for c in CHANNELS}
    identity_indices = {c: support_npz[c + '_identity_index'] for c in CHANNELS}
    for c in CHANNELS:
        audit.truth('support_dtype:' + c, support[c].dtype == np.dtype('int64') and support[c].shape == arrays['mask'].shape and (support[c] >= 0).all())
        audit.truth('support_padding_zero:' + c, (support[c][~arrays['mask']] == 0).all())
    chronological_rows = sorted(range(len(series)), key=lambda i: (str(series.iloc[i]['date']), str(series.iloc[i]['match_id'])))
    # Reconstruct a release sweep once, retaining only last16 history events per
    # player. This is independent of candidate query-window state.
    releases = sorted(valid_raw[['release_day','played_day','source_rank','map_index','source_slot','player_id','observation_id']].itertuples(index=False, name=None))
    prior, rp = defaultdict(list), 0
    role_counts = defaultdict(lambda: np.zeros(6, np.int64))
    event_roles = event_lookup['role'].to_dict()
    for begin in range(0, len(chronological_rows), 256):
        ix = np.asarray(chronological_rows[begin:begin + 256], np.int64)
        shape = (len(ix), 2, 5, 16)
        expected_events = np.full(shape, -1, np.int64)
        expected_probs = np.zeros((len(ix), 2, 5, 6), np.float32)
        for j, i in enumerate(ix):
            row = series.iloc[i]
            day = str(row['date'])
            while rp < len(releases) and releases[rp][0] < day:
                release, played, rank, mi, slot, player, oid = releases[rp]
                insort(prior[player], (played, rank, mi, slot, oid))
                if len(prior[player]) > 16:
                    prior[player].pop(0)
                role_counts[player][event_roles[oid_to_event[oid]]] += 1
                rp += 1
            for side, roster_name in enumerate(('roster_a','roster_b')):
                for player_slot, player in enumerate(row[roster_name]):
                    player = None if pd.isna(player) else str(player)
                    entries = prior.get(player, ()) if player is not None else ()
                    expected_events[j,side,player_slot,16-len(entries):] = [oid_to_event[e[-1]] for e in entries]
                    rc = role_counts.get(player, np.zeros(6, np.int64))
                    expected_probs[j,side,player_slot] = rc / rc.sum() if rc.sum() else np.array([1,0,0,0,0,0])
        audit.compare(f'lineage_batch:{begin}', event_ix[ix], expected_events)
        audit.compare(f'role_probs_batch:{begin}', arrays['role_probs'][ix], expected_probs)
        mask = expected_events >= 0
        events = event_lookup.loc[expected_events[mask]]
        target_days = np.broadcast_to(series.iloc[ix].date.to_numpy(str)[:,None,None,None], shape)[mask]
        target_ids = np.broadcast_to(series.iloc[ix].match_id.to_numpy(str)[:,None,None,None], shape)[mask]
        audit.truth(f'exclusive_release_batch:{begin}', np.all(events.release_day.to_numpy(str) < target_days) and np.all(events.match_id.to_numpy(str) != target_ids))
        audit.compare(f'raw_stats_batch:{begin}', arrays['history'][ix][mask][:,:16], events[list(NAMES[:16])].to_numpy(np.float32))
        origins = np.asarray(target_days, dtype='datetime64[D]')
        ages = np.column_stack(((origins-events.played_day.to_numpy(dtype='datetime64[D]')).astype(np.int64), (origins-events.release_day.to_numpy(dtype='datetime64[D]')).astype(np.int64)))
        audit.compare(f'ages_batch:{begin}', arrays['history'][ix][mask][:,18:], ages)
        audit.compare(f'roles_batch:{begin}', arrays['roles'][ix][mask], events.role.to_numpy())
        if 'champion_status' in arrays:
            audit.compare(f'champion_status_batch:{begin}', arrays['champion_status'][ix][mask], events.champion_status.to_numpy())
        for c in CHANNELS:
            ids = events[c].fillna('<MISSING>').to_numpy(str)
            known = np.fromiter((vocab[c].get(identity, 1) if identity != '<MISSING>' else 0 for identity in ids), np.int64, len(ids))
            audit.compare(f'codes_batch:{c}:{begin}', arrays[ARRAY_KEYS[c]][ix][mask], known)
            table = tables[c]
            actual_identity_ix = identity_indices[c][ix][mask]
            if isinstance(table, dict):
                actual_ids = np.asarray([table[str(ii)] if ii else '<MISSING>' for ii in actual_identity_ix])
            else:
                actual_ids = np.asarray(['<MISSING>' if identity is None else identity for identity in table])[actual_identity_ix]
            audit.compare(f'canonical_batch:{c}:{begin}', actual_ids, ids)
            expected_count = np.zeros(len(ids), np.int64)
            for identity in np.unique(ids):
                selected = ids == identity
                if identity != '<MISSING>':
                    expected_count[selected] = np.searchsorted(release_vectors[c].get(identity, np.array([],str)), target_days[selected], side='left')
            audit.compare(f'n_prior_batch:{c}:{begin}', support[c][ix][mask], expected_count)
            full_codes = np.zeros(shape,np.int64)
            full_codes[mask] = known
            for state, predicate in (('missing',full_codes==0),('oov',full_codes==1),('known',full_codes>=2)):
                counts[c][state][ix] = (predicate & mask).sum(axis=(1,2,3))
    for c in CHANNELS:
        groups[c + '_oov' if c != 'champion' else 'natural_champion_oov'] = counts[c]['oov'] > 0
    groups['seen_champion_history'] = (counts['champion']['known'] > 0) & ~groups['natural_champion_oov']
    groups['missing_champion_identity'] = counts['champion']['missing'] > 0
    groups['cold_history'] = ~arrays['mask'].reshape(len(series), -1).any(1)
    stage0_series = pd.read_parquet(inputs.artifact('stage0_series', inputs.stage('04_feature','stage0_series.parquet')))
    series_order(stage0_series, series, audit, 'stage0_series')
    held_rows=np.zeros(len(series),bool)
    for c in CHANNELS:
        table=tables[c]
        flags=np.asarray([identity in vocab[c] and held(c,identity) for identity in table])
        held_tokens=flags[identity_indices[c]]&arrays['mask']
        held_rows|=held_tokens.any(axis=(1,2,3))
    audit.compare('held_identity_history_source',stage0_series.held_identity_history.to_numpy(bool),held_rows)
    for name, mask in groups.items():
        audit.compare('membership:' + name, stage0_series[name].to_numpy(bool), mask)
    for c in CHANNELS:
        for state, values in counts[c].items():
            audit.compare('window_count:' + c + ':' + state, stage0_series[f'{c}_{state}_tokens'].to_numpy(np.int64), values)
    original = np.load(inputs.path(paths['old_prepared_root'] + '/arrays.npz'), allow_pickle=False)
    for key in arrays:
        if key in original:
            pinned_values = original[key]
            for begin in range(0, len(series), 256):
                audit.compare(f'pinned_original_input:{key}:{begin}', arrays[key][begin:begin+256], pinned_values[begin:begin+256])
            del pinned_values
    for begin in range(0, len(series), 256):
        ix = np.arange(begin, min(begin+256,len(series)))
        mask = arrays['mask'][ix]
        ei = event_ix[ix][mask]
        audit.compare(f'inherited_opponent_constants:{begin}', arrays['history'][ix][mask][:,16:18], event_lookup.loc[ei,list(NAMES[16:18])].to_numpy(np.float32))
    inventory = load_json(inputs.artifact('stage0', inputs.stage('08_reporting','stage0.json')))
    assessment = series.split.to_numpy() == 'ASSESS'
    months = series.date.astype(str).str[:7].to_numpy()
    supported = True
    for name in ('natural_champion_oov','seen_champion_history'):
        per_month = {m: int(np.count_nonzero(assessment & groups[name] & (months == m))) for m in MONTHS}
        expected = dict(series_count=sum(per_month.values()), months=[m for m,n in per_month.items() if n], per_month_series_counts=per_month)
        reported = inventory['assessment'][name]
        audit.compare('stage0_count:' + name, reported['series_count'], expected['series_count'])
        audit.compare('stage0_months:' + name, reported['months'], expected['months'])
        for m, n in per_month.items():
            reported_months = reported['per_month']
            audit.compare('stage0_month_count:' + name + ':' + m, reported_months[m], n)
        supported &= expected['series_count'] >= 200 and len(expected['months']) >= 6
    expected_reasons=[]
    for group in ('natural_champion_oov','seen_champion_history'):
        selected=assessment&groups[group]
        if np.count_nonzero(selected)<200:
            expected_reasons.append(group+': fewer than 200 canonical assessment series')
        if len(set(months[selected]))<6:
            expected_reasons.append(group+': fewer than six nonzero assessment months')
    component = inventory['component_contract']
    independent_counts = normalizers(arrays, np.flatnonzero(series.split.to_numpy() == 'TRAIN'))['history']['count']
    available = tuple(schema['history_feature_names']) == NAMES and bool((independent_counts[list(NUMERIC)] > 0).all())
    audit.truth('component_available_source', component['available'] == available)
    audit.truth('component_masks_source', component['valid_masks'] == (not audit.failures))
    supported &= available and not audit.failures and component['phi_dimension'] == 36
    audit.truth('component_named_fields', tuple(schema['history_feature_names']) == NAMES and arrays['history'].shape[-1] == 20)
    if not available:
        expected_reasons.append('component fields unavailable or zero TRAIN numeric support')
    if not component['valid_masks']:
        expected_reasons.append('component/source validity mask contract failed')
    if component['phi_dimension']!=36:
        expected_reasons.append('phi_dimension must equal 36')
    train_ix=np.flatnonzero(series.split.to_numpy()=='TRAIN')
    inherited_sizes={c:max(2,int(arrays[ARRAY_KEYS[c]][train_ix].max())+1) for c in CHANNELS}
    capacity=all(all(code<inherited_sizes[c] for code in vocab[c].values()) for c in CHANNELS)
    compare_mapping(audit,'inherited_embedding_capacity',component,dict(inherited_embedding_sizes=inherited_sizes,embedding_capacity_compatible=capacity))
    if not capacity:
        expected_reasons.append('frozen vocabulary exceeds original TRAIN embedding capacity; hard-stop before fits')
    supported &= capacity
    compare_mapping(audit,'support_gate_exact_reasons',inventory['support_gate'],dict(status='CLOSED_NO_FITS' if expected_reasons else 'SUPPORT_PASS',reasons=expected_reasons,allowed_neural_fits=0 if expected_reasons else 9,allowed_CAL_terminal_scalars=0 if expected_reasons else 3))
    compare_mapping(audit,'component_fields',component,dict(numeric_fields=[NAMES[i] for i in NUMERIC],stat_masks=list(NAMES[:12]),roles=['TOP','JUNGLE','MID','ADC','SUPPORT'],TRAIN_observation_counts=independent_counts.tolist(),role_observations=int(np.count_nonzero(arrays['mask'][series.split.to_numpy()=='TRAIN']&(arrays['roles'][series.split.to_numpy()=='TRAIN']>0)))))
    for c in CHANNELS:
        tr = series.split.to_numpy()=='TRAIN'
        source_train = raw.history_eligible & (raw.date<'2025-07-01') & (raw.release_day<'2025-07-01')
        missing = dict(source_events=int(raw.loc[source_train,c].isna().sum()),target_rows=int(np.count_nonzero(tr & (counts[c]['missing']>0))),target_tokens=int(counts[c]['missing'][tr].sum()))
        missing['legacy_missing_embedding_support']='UNTRAINED_NO_OBSERVED_TRAIN_TOKENS' if missing['target_tokens']==0 else 'OBSERVED_TRAIN_TOKENS_NOT_RELIABILITY_CLAIM'
        compare_mapping(audit,'TRAIN_missingness:'+c,inventory['TRAIN_missing_identity'][c],missing)
        for split in ('TRAIN','CAL','ASSESS'):
            selected=series.split.to_numpy()==split
            expected_tokens={s:int(counts[c][s][selected].sum()) for s in ('known','oov','missing')}
            expected_tokens['observed']=sum(expected_tokens.values())
            compare_mapping(audit,f'token_totals:{c}:{split}',inventory['token_totals'][c][split],expected_tokens)
        table=pd.read_parquet(inputs.artifact(c+'_TRAIN_support',inputs.stage('04_feature',c+'_TRAIN_support.parquet')))
        table_ids=table.canonical_id.fillna('<MISSING>').astype(str).to_numpy()
        expected_counts=raw.loc[source_train,c].fillna('<MISSING>').value_counts().to_dict()
        audit.truth('TRAIN_support_all_identities:'+c,set(table_ids)==set(expected_counts))
        audit.compare('TRAIN_support_unique_events:'+c,table.unique_source_events.to_numpy(),[expected_counts.get(identity,-1) for identity in table_ids])
        for split,inventory_key in (('TRAIN','TRAIN'),('CAL','CAL'),('ASSESS','assessment')):
            for group,gmask in groups.items():
                if group in inventory[inventory_key]:
                    selected=(series.split.to_numpy()==split)&gmask
                    audit.compare(f'group_count:{split}:{group}',inventory[inventory_key][group]['series_count'],int(selected.sum()))
        del table
    taxonomy_checks(inputs,series,stage0_series,audit)
    identity_public_checks(raw,series,vocab,audit)
    return groups, support, supported, inventory, raw, vocab



def taxonomy_checks(inputs,series,rows,audit):
    """Read frozen taxonomy rule literals as data; independently evaluate them."""
    tree=ast.parse(inputs.path('src/models/competition_tiers.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_RULES')
    rules=[]
    for call in node.value.elts:
        pattern=ast.literal_eval(call.args[0])
        tier=call.args[2].attr.lower()
        bounds={}
        for kw in call.keywords:
            if kw.arg in ('before','not_before'):
                bounds[kw.arg]=date(*(ast.literal_eval(v) for v in kw.value.args))
        rules.append((re.compile(pattern),tier,bounds))
    tiers=[]
    for name,day in zip(series.tournament,series.date):
        text='' if name is None else str(name)
        normalized=unicodedata.normalize('NFKD',text).casefold()
        normalized=re.sub('[^a-z0-9]+',' ',''.join(c for c in normalized if not unicodedata.combining(c))).strip()
        target=date.fromisoformat(str(day))
        tier='unknown'
        if normalized not in ('','na','nan','nat','none','null'):
            for expression,value,bounds in rules:
                if target<bounds.get('not_before',date.min) or target>=bounds.get('before',date.max):
                    continue
                if expression.search(normalized):
                    tier=value
                    break
        tiers.append(tier)
    audit.compare('taxonomy_independent_tier',rows.profile_tier.to_numpy(),tiers)
    states=['UNKNOWN' if t=='unknown' else 'major' if t=='major' else 'nonmajor' for t in tiers]
    audit.compare('taxonomy_major_states',rows.major_state.to_numpy(),states)
    manifest=load_json(inputs.path('conf/base/research_benchmark.json'))
    metadata=pd.read_parquet(inputs.path(manifest['input']['path']))
    key='match_id' if 'match_id' in metadata else 'golgg_match_id'
    audit.truth('taxonomy_benchmark_ids_unique',not metadata[key].astype(str).duplicated().any())
    lookup=metadata.assign(_id=metadata[key].astype(str)).set_index('_id')
    unknown=pd.to_numeric(lookup.unknown_players,errors='coerce').reindex(series.match_id.astype(str)).to_numpy(float)
    expected=np.where(np.isnan(unknown),'UNKNOWN',np.where(unknown>0,'unseen','seen'))
    audit.compare('taxonomy_source_states',rows.player_source_state.to_numpy(),expected)
    audit.compare('taxonomy_source_unknown_counts',rows.benchmark_unknown_players.to_numpy(float),unknown)


def identity_public_checks(raw,series,vocab,audit):
    """Public actual API perturbations supplement the full source-state oracle."""
    from src.analysis.champion_oov_reliability import build_identity_state
    from src.models.champion_oov_reliability import held_identity,identity_residual_weights,held_channel_codes
    for c in CHANNELS:
        identities=sorted(str(v) for v in raw[c].dropna().unique())
        audit.compare('held_SHA_all_identities:'+c,[held_identity(c,i) for i in identities],[held(c,i) for i in identities])
        codes=np.asarray([vocab[c].get(i,1) for i in identities]+[0,0],np.int64)
        ids=np.asarray(identities+[None,None],object)
        mask=np.ones(len(codes),bool)
        mask[-1]=False
        counts=np.arange(len(codes),dtype=np.int64)*7
        for hold in (False,True):
            expected=np.asarray([n/(n+20) if valid and identity in vocab[c] and not (hold and held(c,identity)) else 0 for identity,n,valid in zip(ids,counts,mask)])
            actual=identity_residual_weights(c,ids,counts,mask,vocab[c],held_out=hold)
            audit.compare(f'public_residual_gates:{c}:{hold}',actual,expected,'normalizer')
        expected_codes=codes.copy()
        for i,identity in enumerate(ids):
            if mask[i] and identity in vocab[c] and held(c,identity):
                expected_codes[i]=1
        audit.compare('public_held_routing:'+c,held_channel_codes(c,codes,vocab[c],mask),expected_codes)
    sample=raw.sort_values(['release_day','observation_id']).iloc[:256].copy()
    targets=series.groupby('split',sort=True).head(1)[['match_id','date']].copy()
    def expected_state(observations):
        eligible=observations[observations.history_eligible]
        vv={}
        for c in CHANNELS:
            ids=sorted(str(v) for v in eligible.loc[(eligible.date<'2025-07-01')&(eligible.release_day<'2025-07-01'),c].dropna().unique())
            vv[c]={identity:n+2 for n,identity in enumerate(ids)}
        targets_expected={}
        for row in targets.itertuples(index=False):
            targets_expected[str(row.match_id)]={}
            for c in CHANNELS:
                counter=Counter(str(v) for v in eligible.loc[eligible.release_day<str(row.date),c].dropna())
                targets_expected[str(row.match_id)][c]=dict(counter)
        return dict(vocabularies=vv,targets=targets_expected)
    actual=build_identity_state(sample,targets,'2025-07-01')
    audit.truth('identity_public_raw_state',actual==expected_state(sample))
    audit.truth('identity_public_row_reorder',build_identity_state(sample.iloc[::-1],targets.iloc[::-1],'2025-07-01')==actual)
    future=sample.iloc[:1].copy()
    future['observation_id']='oracle_future_source'
    future['match_id']='oracle_future_match'
    future['date']=future['release_day']='2027-01-01'
    for c in CHANNELS:
        future[c]='oracle_future_'+c
    augmented=pd.concat((sample,future),ignore_index=True)
    audit.truth('identity_public_future_perturbation',build_identity_state(augmented,targets,'2025-07-01')==expected_state(augmented))
    for row in targets.itertuples(index=False):
        target_event=future.copy()
        target_event['observation_id']='oracle_target_'+str(row.match_id)
        target_event['match_id']=str(row.match_id)
        target_event['date']=target_event['release_day']=str(row.date)
        augmented=pd.concat((sample,target_event),ignore_index=True)
        result=build_identity_state(augmented,targets,'2025-07-01')
        audit.truth('identity_public_target_withheld:'+str(row.match_id),result==expected_state(augmented) and result['targets'][str(row.match_id)]==actual['targets'][str(row.match_id)])
    duplicate=pd.concat((sample,sample.iloc[:1]),ignore_index=True)
    try:
        build_identity_state(duplicate,targets,'2025-07-01')
    except ValueError:
        rejected=True
    else:
        rejected=False
    audit.truth('identity_public_duplicate_event_rejected',rejected)

def normalizers(arrays, train):
    stats = {}
    for key in ('history','players','context'):
        width = arrays[key].shape[-1]
        total, square, count = np.zeros(width), np.zeros(width), np.zeros(width,np.int64)
        for start in range(0,len(train),256):
            ix = train[start:start+256]
            raw = arrays[key][ix].astype(np.float64)
            valid = np.isfinite(raw)
            if key == 'history':
                valid &= arrays['mask'][ix][...,None]
            observed = np.where(valid,raw,0).reshape(-1,width)
            total += observed.sum(0)
            square += np.square(observed).sum(0)
            count += valid.reshape(-1,width).sum(0)
        mean = total / np.maximum(count,1)
        scale = np.sqrt(np.maximum(square/np.maximum(count,1)-mean**2,0))
        scale[scale < 1e-6] = 1
        stats[key] = dict(mean=mean, scale=scale, count=count, sum=total, sum_of_squares=square)
    return stats


def normalize(raw, stats, mask=None):
    valid = np.isfinite(raw) & (stats['count'] > 0)
    if mask is not None:
        valid &= mask[...,None]
    values = (raw-stats['mean']).astype(np.float32)
    values = (values/stats['scale']).astype(np.float32)
    values[~valid] = 0
    return values, valid


def phi_batch(arrays, ix, stats):
    normalized, valid = normalize(arrays['history'][ix],stats['history'],arrays['mask'][ix])
    roles = arrays['roles'][ix]
    role_bits = roles[...,None] == np.arange(1,6)
    phi = np.concatenate((normalized[...,NUMERIC],valid[...,:12],role_bits),-1).astype(np.float32)
    phi[~arrays['mask'][ix]] = 0
    return phi


def replay(state, arm, arrays, ix, stats, support, vocab):
    """Separately expressed functional tensor graph; never candidate forward."""
    def linear(v, key):
        return F.linear(v,state[key+'.weight'],state[key+'.bias'])
    with torch.inference_mode():
        mask = torch.from_numpy(arrays['mask'][ix])
        inputs = {}
        for key in ('history','players','context'):
            norm, observed = normalize(arrays[key][ix],stats[key],arrays['mask'][ix] if key=='history' else None)
            inputs[key] = torch.from_numpy(np.concatenate((norm,observed.astype(np.float32)),-1))
        phi = torch.from_numpy(phi_batch(arrays,ix,stats))
        embedding, status_codes, traces = {}, None, {}
        for c in CHANNELS:
            codes = np.array(arrays[ARRAY_KEYS[c]][ix],dtype=np.int64,copy=True)
            held_codes = np.array([code for identity,code in vocab[c].items() if held(c,identity)],np.int64)
            held_mask = np.isin(codes,held_codes) & arrays['mask'][ix]
            if arm in ('H','HC'):
                codes[held_mask] = 1
            residual = state[c+'.weight'][torch.from_numpy(codes)]
            if arm in ('H','D0'):
                embedding[c] = residual
            else:
                n = support[c][ix].astype(np.float32)
                gate = np.where(codes>=2,n/(n+20),0).astype(np.float32)
                gate[codes==0] = 1
                projection = linear(phi,'projectors.'+c)
                embedding[c] = (projection + residual*torch.from_numpy(gate)[...,None])*mask[...,None]
                traces[c] = dict(projection=projection.numpy(),gate=gate,embedding=embedding[c].numpy())
            if c == 'champion':
                status_codes = codes
        status = torch.from_numpy((np.minimum(status_codes,2)[...,None] == np.arange(3)).astype(np.float32))
        perf = torch.tanh(linear(inputs['history'],'performance'))
        interaction = perf*(1+torch.tanh(linear(embedding['champion'],'champion_performance')))
        role = F.embedding(torch.from_numpy(arrays['roles'][ix].astype(np.int64)),state['role.weight'])
        token = torch.tanh(linear(torch.cat((interaction,embedding['champion'],role,status,embedding['patch'],embedding['competition']),-1),'token') + state['age'])
        energy = linear(token,'attention').squeeze(-1)
        energy = energy.masked_fill(~mask,-torch.inf)
        present = mask.any(-1)
        energy = torch.where(present[...,None],energy,torch.zeros_like(energy))
        attention = torch.softmax(energy,-1)*mask
        summary = torch.sum(attention[...,None]*token,dim=-2)+(~present)[...,None]*state['cold']
        player = torch.cat((summary,inputs['players']),-1)
        role_prob = torch.from_numpy(arrays['role_probs'][ix].astype(np.float32))
        rosters = {}
        for name in ('strength','style'):
            p = torch.tanh(linear(player,'player_'+name+'.0'))
            role_sum = (role_prob[...,None]*p[...,None,:]).sum(-3)
            role_sum = role_sum/role_prob.sum(-2).clamp_min(1e-6)[...,None]
            pooled = torch.cat((p.mean(-2),role_sum.flatten(-2)),-1)
            rosters[name] = torch.tanh(linear(pooled,'roster_'+name+'.0'))
        strength,style = rosters['strength'],rosters['style']
        orientations = []
        for a,b in ((0,1),(1,0)):
            orientations.append(torch.cat((strength[:,a],strength[:,b],style[:,a],style[:,b],inputs['context']),-1))
        hidden = torch.tanh(linear(torch.stack(orientations,1),'score_head.0'))
        logits = linear(hidden,'score_head.2').reshape(-1,6)
        legal = torch.from_numpy(np.tile(np.arange(3),2)[None,:] <= (arrays['best_of'][ix]//2)[:,None])
        logits = logits.masked_fill(~legal,-torch.inf)
        law = torch.softmax(logits.double(),-1).numpy()
        return law,logits.numpy(),traces


def temper(pmf, temperature):
    if not math.isfinite(temperature) or not .25 <= temperature <= 4:
        raise ValueError('Saved temperature outside frozen bounds')
    with np.errstate(divide='ignore'):
        logs = np.log(pmf)/temperature
    return np.exp(logs-logsumexp(logs,axis=-1,keepdims=True))


def temperature_objective(members, labels, temperature):
    mixture = np.mean([temper(p,temperature) for p in members],axis=0)
    return float(-np.log(mixture[np.arange(len(labels)),labels]).mean()+.001*math.log(temperature)**2)


def scalar_check(receipt, members, labels, audit, arm):
    detail = receipt['score_temperature']
    t = float(receipt['temperature'])
    objective = temperature_objective(members,labels,t)
    audit.compare('temperature_objective:'+arm,detail['objective'],objective,'objective')
    audit.compare('temperature_penalty:'+arm,detail['penalty'],.001*math.log(t)**2,'normalizer')
    # Verify the independently written penalized objective in log-temperature
    # coordinates, matching the saved projected-gradient estimand.
    z = math.log(t)
    h = 1e-5*max(1,abs(z))
    lower,upper = math.log(.25),math.log(4.)
    f = lambda value: temperature_objective(members,labels,math.exp(value))
    if z-h >= lower and z+h <= upper:
        gradient = (f(z+h)-f(z-h))/(2*h)
        gradient_half = (f(z+h/2)-f(z-h/2))/h
        audit.compare('temperature_derivative_stability:'+arm,gradient,gradient_half,'gradient')
    elif z <= lower+h:
        gradient = (f(z+h)-objective)/h
    else:
        gradient = (objective-f(z-h))/h
    projected = 0 if (t==.25 and gradient>0) or (t==4 and gradient<0) else gradient
    audit.compare('temperature_log_gradient:'+arm,detail['raw_gradient_log_temperature'],gradient,'gradient')
    audit.compare('temperature_projected_gradient:'+arm,detail['projected_gradient'],projected,'gradient')
    audit.truth('temperature_stationary:'+arm,abs(projected)<=1e-5,dict(raw_gradient=gradient,projected_gradient=projected,temperature=t))
    audit.truth('temperature_optimizer:'+arm,detail['convergence_success'] is True and detail['maxiter']==300 and detail['xatol']==1e-7 and detail['nfev']<=300)
    for i,entry in enumerate(detail['candidate_evaluations']):
        audit.compare(f'temperature_candidate_objective:{arm}:{i}',entry['penalized_objective'],temperature_objective(members,labels,entry['temperature']),'objective')
    audit.truth('temperature_candidate_selection:'+arm,objective<=min(v['penalized_objective'] for v in detail['candidate_evaluations'])+1e-10)
    for i,entry in enumerate(detail['objective_evaluations']):
        audit.compare(f'temperature_objective_trace:{arm}:{i}',entry['penalized_objective'],temperature_objective(members,labels,entry['temperature']),'objective')
    # Zero verification solves: direct feasible derivatives check the saved
    # scalar without introducing any independently optimized replacement.
    return t


def probabilities(frame):
    families = [('score_p_'+str(i) for i in range(6)), ('p_'+str(i) for i in range(6)), ('score_'+str(i) for i in range(6)), ('score_prob_'+str(i) for i in range(6)), ('p'+str(i) for i in range(6))]
    for fields in families:
        fields = tuple(fields)
        if all(f in frame for f in fields):
            return frame[list(fields)].to_numpy(float)
    raise ValueError('Terminal artifact must export all six named PMF bins')


def score_laws(pmf, series):
    legal = np.tile(np.arange(3),2)[None,:] <= (series.best_of.to_numpy()//2)[:,None]
    if pmf.shape != legal.shape or not np.isfinite(pmf).all() or (pmf<0).any() or np.any(pmf[~legal]!=0):
        raise ValueError('Malformed law; no output masking permitted')
    if not np.allclose(pmf.sum(1),1,atol=1e-10,rtol=1e-10):
        raise ValueError('PMF is not normalized')
    winner = series.winner.to_numpy(float)
    labels = series.score_index.to_numpy(float)
    pwin = pmf[:,:3].sum(1)
    nll = np.full(len(series),np.nan)
    terminal = np.full(len(series),np.nan)
    brier = np.full(len(series),np.nan)
    wm = np.isfinite(winner)
    tm = np.isfinite(labels)
    if not np.isin(winner[wm],(0,1)).all() or not np.isin(labels[tm],range(6)).all():
        raise ValueError('Invalid supervised classes')
    prob = np.where(winner[wm]==1,pwin[wm],1-pwin[wm])
    if (prob<=0).any() or (pmf[np.flatnonzero(tm),labels[tm].astype(int)]<=0).any():
        raise ValueError('Zero true-label probability makes observed scores infinite')
    nll[wm] = -np.log(prob)
    brier[wm] = (pwin[wm]-winner[wm])**2
    terminal[tm] = -np.log(pmf[np.flatnonzero(tm),labels[tm].astype(int)])
    return dict(winner_nll=nll,winner_brier=brier,score_nll=terminal,p_win=pwin)


def law_metrics(pmf,series):
    scores=score_laws(pmf,series)
    y=series.winner.to_numpy(float)
    observed=np.isfinite(y)
    y,p=y[observed],scores['p_win'][observed]
    def mean(values):
        values=np.asarray(values,float)
        return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
    positives=int((y==1).sum())
    negatives=int((y==0).sum())
    auc=float((rankdata(p)[y==1].sum()-positives*(positives+1)/2)/(positives*negatives)) if positives and negatives else None
    bins=[]
    ece=0.
    bin_index=np.asarray([min(9,int(10*float(value))) for value in p],dtype=int)
    for b in range(10):
        chosen=bin_index==b
        n=int(chosen.sum())
        forecast,outcome=mean(p[chosen]),mean(y[chosen])
        if n:
            ece+=n/len(y)*abs(forecast-outcome)
        bins.append(dict(lower=b/10,upper=(b+1)/10,n=n,mean_prediction=forecast,observed_rate=outcome))
    winner=dict(n=len(y),absent_labels=int((~observed).sum()),nll=mean(scores['winner_nll']),brier=mean(scores['winner_brier']),auc=auc,
                auc_absence_reason=None if positives and negatives else 'requires_both_winner_classes',calibration_ece=ece if len(y) else None,calibration_bins=bins)
    terminal=series.score_index.to_numpy(float)
    tm=np.isfinite(terminal)
    score=dict(n=int(tm.sum()),absent_labels=int((~tm).sum()),nll=mean(scores['score_nll']))
    bo=series.best_of.to_numpy(int)
    length=series.maps_played.to_numpy(float) if 'maps_played' in series else np.full(len(series),np.nan)
    derived=bo//2+1+terminal%3
    if (np.isfinite(length)&tm&(length!=derived)).any():
        raise ValueError('Raw length class contradicts terminal class')
    length=np.where(np.isfinite(length),length,derived)
    known=np.isfinite(length)
    grid=bo[:,None]//2+1+np.arange(3)
    lp=pmf[:,:3]+pmf[:,3:]
    length_nll=np.full(len(series),np.nan)
    length_rps=np.full(len(series),np.nan)
    active=known&(bo>1)
    li=(length[active]-(bo[active]//2+1)).astype(int)
    length_nll[active]=-np.log(lp[np.flatnonzero(active),li])
    for j in np.flatnonzero(active):
        k=bo[j]//2+1
        length_rps[j]=sum((lp[j,:t+1].sum()-(length[j]<=k+t))**2 for t in range(k-1))
    play=np.stack([(lp*(grid>=m)).sum(1) for m in range(1,6)],axis=1)
    play_mask=known[:,None]&(np.arange(1,6)[None,:]<=bo[:,None])
    play_error=np.where(play_mask,(play-(np.arange(1,6)[None,:]<=length[:,None]))**2,np.nan)
    per_series=np.where(known,np.nansum(play_error,1)/bo,np.nan)
    metrics=dict(winner=winner,score=score,
                 length=dict(n=int(active.sum()),absent_labels=int((~known).sum()),excluded_bo1=int((bo==1).sum()),nll=mean(length_nll),rps=mean(length_rps)),
                 map_play=dict(n=int(play_mask.sum()),absent_series_labels=int((~known).sum()),brier=mean(play_error.ravel()),series_weighted_brier=mean(per_series),
                               by_map_index=[dict(map_index=m,n=int(play_mask[:,m-1].sum()),brier=mean(play_error[:,m-1])) for m in range(1,6)]))
    scores.update(length_nll=length_nll,length_rps=length_rps,map_play_brier=per_series)
    return metrics,scores


def bootstrap_expected(delta, labels, mask, universe, confidence=.995, one_sided=False,min_rows=200,min_blocks=6):
    selected = np.asarray(mask,bool)
    if not np.isfinite(delta[selected]).all():
        raise ValueError('Nonfinite selected contrast; do not hide with finite intersection')
    counts = np.array([np.count_nonzero(selected & (labels==b)) for b in universe],np.int64)
    sums = np.array([delta[selected & (labels==b)].sum(dtype=np.float64) for b in universe])
    means = np.divide(sums,counts,out=np.zeros(len(counts)),where=counts>0)
    matrix = np.random.default_rng(20261009).integers(0,len(universe),size=(10000,len(universe)),dtype=np.int64)
    denominators = counts[matrix].sum(1)
    equal_denominators = (counts[matrix]>0).sum(1)
    draws = np.divide(sums[matrix].sum(1),denominators,out=np.full(10000,np.nan),where=denominators>0)
    equal = np.divide(means[matrix].sum(1),equal_denominators,out=np.full(10000,np.nan),where=equal_denominators>0)
    undefined = int(np.count_nonzero(denominators==0))
    supported = int(np.count_nonzero(counts))
    status = 'NOT_ESTIMABLE' if undefined or not selected.any() else 'INSUFFICIENT_SUPPORT' if selected.sum()<min_rows or supported<min_blocks else 'ESTIMABLE'
    def bounds(values):
        if undefined or not selected.any():
            return None,None
        if one_sided:
            return None,float(np.quantile(values,confidence,method='linear'))
        return tuple(float(v) for v in np.quantile(values,[(1-confidence)/2,1-(1-confidence)/2],method='linear'))
    lo,hi = bounds(draws)
    elo,ehi = bounds(equal)
    result = dict(status=status,n_rows=int(selected.sum()),n_blocks=supported,n_universe_blocks=len(universe),
                  point=float(sums.sum()/counts.sum()) if counts.sum() else None,lower=lo,upper=hi,
                  equal_block=dict(point=float(means[counts>0].mean()) if supported else None,lower=elo,upper=ehi),
                  undefined_draws=undefined,nonfinite_selected_rows=0,confidence=confidence,one_sided=one_sided,
                  draws=10000,seed=20261009,bootstrap_index_sha256=hashlib.sha256(matrix.tobytes(order='C')).hexdigest(),
                  min_rows=min_rows,min_blocks=min_blocks,
                  block_summary=[dict(block=str(b),count=int(n),sum=float(s),mean_or_null=float(s/n) if n else None) for b,n,s in zip(universe,counts,sums)])
    return result,dict(matrix=matrix,weighted=draws,equal=equal,weighted_denominator=denominators,equal_denominator=equal_denominators,counts=counts,sums=sums)


def export_draw_check(inputs,audit,name,expected,delta,mask,candidate_scores,reference_scores,ids,labels):
    if name.startswith(('primary_slot_','proxy_slot_')):
        prefix='primary' if name.startswith('primary') else 'proxy'
        slot=int(name.rsplit('_',1)[1])
        reference={3:'D0',4:'H',5:'C',6:'C0'}[slot]
        group='all' if slot==6 else 'natural_champion_oov'
        stem=f'{prefix}_HC_{reference}_{group}_draws'
    elif name.startswith('guard_'):
        group,reference=name[len('guard_'):].rsplit('_',1)
        stem=f'guard_HC_{reference}_{group}_draws'
    else:
        stem=name+'_draws'
    actual=np.load(inputs.artifact(stem,inputs.stage('08_reporting',stem+'.npz')),allow_pickle=False)
    mapping={'matrix':'draw_indices','weighted':'weighted_delta','equal':'equal_delta',
             'weighted_denominator':'weighted_denominator','equal_denominator':'equal_denominator',
             'counts':'block_counts','sums':'block_sums'}
    for key,saved in mapping.items():
        audit.compare('draw_array:'+name+':'+key,actual[saved],expected[key],'metric' if key in ('weighted','equal','sums') else 'exact')
    audit.compare('draw_undefined:'+name,actual['undefined'],np.isnan(expected['weighted'])|np.isnan(expected['equal']))
    audit.compare('draw_selection:'+name,actual['selected'],mask)
    audit.compare('draw_per_series_delta:'+name,actual['differences'],delta,'metric')
    audit.truth('draw_output_missingness:'+name,not actual['candidate_missing'][mask].any() and not actual['reference_missing'][mask].any())
    audit.compare('draw_ids:'+name,actual['match_ids'],ids)
    audit.compare('draw_blocks:'+name,actual['block_labels'],labels)
    audit.compare('draw_candidate_scores:'+name,actual['candidate_scores'],candidate_scores,'metric')
    audit.compare('draw_reference_scores:'+name,actual['reference_scores'],reference_scores,'metric')
    audit.compare('draw_candidate_absence:'+name,actual['candidate_missing'],~np.isfinite(candidate_scores))
    audit.compare('draw_reference_absence:'+name,actual['reference_missing'],~np.isfinite(reference_scores))
    draw_counts=np.stack([np.bincount(row,minlength=len(expected['counts'])) for row in expected['matrix']])
    audit.compare('draw_multiplicities:'+name,actual['draw_counts'],draw_counts)
    expected_means=np.divide(expected['sums'],expected['counts'],out=np.full(len(expected['counts']),np.nan),where=expected['counts']>0)
    audit.compare('draw_block_means:'+name,actual['block_means'],expected_means,'metric')


def independent_initial_state(arrays,train,maps,seed,arm):
    """Primitive tensor initializer in frozen original constructor order.

    No candidate/reference network is instantiated or forwarded, and no
    optimizer exists. The old TRAIN baseline prop bias is an input convention.
    """
    nn=torch.nn
    state={}
    def module(key,value):
        for name,tensor in value.state_dict().items():
            state[key+'.'+name]=tensor.detach().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for c,width in zip(CHANNELS,(8,4,4)):
            n=max(2,int(arrays[ARRAY_KEYS[c]][train].max())+1)
            module(c,nn.Embedding(n,width))
        module('role',nn.Embedding(6,8))
        state['age']=torch.zeros(16,32)
        h,p,c=(arrays[k].shape[-1] for k in ('history','players','context'))
        module('performance',nn.Linear(2*h,32))
        module('champion_performance',nn.Linear(8,32))
        module('token',nn.Linear(32+16+3+8,32))
        module('attention',nn.Linear(32,1))
        state['cold']=torch.zeros(32)
        for key in ('player_strength.0','player_style.0'):
            module(key,nn.Linear(32+2*p,32))
        for key in ('roster_strength.0','roster_style.0'):
            module(key,nn.Linear(224,32))
        module('score_head.0',nn.Linear(128+2*c,32))
        module('score_head.2',nn.Linear(32,3))
        module('map_embedding',nn.Embedding(5,8))
        module('prop_head.0',nn.Linear(128+2*c+8,32))
        module('prop_head.2',nn.Linear(32,4))
        state['map_ids']=torch.arange(5)
        state['loser_scores']=torch.tensor([0,1,2,0,1,2])
        if arm!='H':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261009)
                for channel,width in zip(CHANNELS,(8,4,4)):
                    module('projectors.'+channel,nn.Linear(36,width))
    label_maps=maps[maps.row_index.isin(train)]
    km=label_maps.kills.to_numpy(float)
    dm=label_maps.duration_minutes.to_numpy(float)
    mean_kills=float(np.nanmean(km)) if np.isfinite(km).any() else 1.
    mean_log_duration=float(np.nanmean(np.log(dm))) if np.isfinite(dm).any() else 0.
    state['prop_head.2.bias']=torch.tensor([math.log(max(mean_kills,1e-6)),math.log(.1),mean_log_duration,math.log(.25)])
    return state


def compare_mapping(audit,name,actual,expected):
    for k,v in expected.items():
        if k not in actual:
            audit.truth(name+':'+k,False,'Required field absent')
        elif isinstance(v,dict):
            compare_mapping(audit,name+':'+k,actual[k],v)
        elif v is None:
            audit.truth(name+':'+k,actual[k] is None)
        elif isinstance(v,list) and v and isinstance(v[0],dict):
            audit.truth(name+':'+k+':length',len(actual[k])==len(v))
            for i,(aa,ee) in enumerate(zip(actual[k],v)):
                compare_mapping(audit,f'{name}:{k}:{i}',aa,ee)
        else:
            audit.compare(name+':'+k,actual[k],v,'metric' if isinstance(v,float) else 'exact')


def trace_check(inputs, train, audit, branch):
    trace = operation_trace(inputs)
    required = ('operation_id','study','kind','arm','seed','partition','authorized_ceiling','counter_before','counter_after','started_at_utc','finished_at_utc','status','parameters_sha256','input_ids_sha256','iterations','function_evaluations','failure_reason')
    audit.truth('trace_entry_schema', all(all(k in op for k in required) for op in trace))
    audit.truth('trace_operation_ids_unique', len({op['operation_id'] for op in trace}) == len(trace))
    neural = [op for op in trace if op.get('operation') == 'W1_NEURAL_FIT']
    scalar = [op for op in trace if op.get('operation') == 'W1_TERMINAL_CAL']
    expected = [(kind, arm, seed) for arm in ARMS for kind, seed in
                [('W1_NEURAL_FIT', s) for s in SEEDS] + [('W1_TERMINAL_CAL', None)]]
    audit.truth('trace_authorized_prefix', [(op.get('operation'), op['arm'], op['seed']) for op in trace] == expected[:len(trace)])
    audit.truth('trace_only_authorized_operations', len(neural) + len(scalar) == len(trace))
    audit.truth('trace_budget', len(neural) <= 9 and len(scalar) <= 3)
    for kind, ops, ceiling, partition in (('neural', neural, 9, 'TRAIN'), ('temperature', scalar, 3, 'CAL')):
        for n, op in enumerate(ops):
            audit.truth(f'trace_count_before:{kind}:{n}',
                        op['counter_before'] == n and op['counter_after'] == n+1
                        and op['authorized_ceiling'] == ceiling and op['partition'] == partition
                        and op['finished_at_utc'] is not None and op['started_at_utc'] <= op['finished_at_utc'])
    completed = lambda op: op['status'] in ('COMPLETED', 'SUCCESS')
    audit.truth('trace_terminal_operation_status', all(completed(op) or op['status'] == 'FAILED' for op in trace))
    audit.truth('trace_failed_terminal_prefix', all(completed(op) for op in trace[:-1]))
    observed_epochs = 0
    unknown = False
    failed_epochs = 0
    for op in neural:
        epochs = op.get('fit_receipt', {}).get('epochs_completed')
        receipt_name=f"{op['arm']}_seed_{op['seed']}_fit_receipt"
        receipt_path=inputs.path(inputs.stage('06_models',receipt_name+'.json'))
        if receipt_path.is_file():
            retained=load_json(inputs.artifact(receipt_name,inputs.stage('06_models',receipt_name+'.json')))
            if op.get('fit_receipt') is not None:
                audit.truth('trace_retained_receipt_matches:'+op['operation_id'],retained==op['fit_receipt'])
            epochs=retained.get('epochs_completed')
        recorded = isinstance(epochs, int) and not isinstance(epochs, bool) and 0 <= epochs <= 12
        audit.truth('trace_recorded_epoch_bounds:'+op['operation_id'],epochs is None or recorded)
        audit.truth('trace_planned_epochs:' + op['operation_id'], op.get('epochs') == 12)
        if completed(op):
            audit.truth('trace_completed_epochs:' + op['operation_id'], recorded and epochs == 12)
        elif not recorded:
            unknown = True
        if recorded:
            observed_epochs += epochs
            if not completed(op):
                failed_epochs += epochs
    if branch in ('ZERO_FIT', 'PREPARE'):
        audit.truth('closure_zero_operations', not trace)
    elif branch == 'COMPLETED':
        audit.truth('completed_exact_operations', len(neural) == 9 and len(scalar) == 3 and all(completed(op) for op in trace))
    counts = dict(neural=len(neural), CAL_temperature=len(scalar), seed_epochs=observed_epochs,
                  neural_completed=sum(completed(op) for op in neural),
                  CAL_temperature_completed=sum(completed(op) for op in scalar),
                  failed_attempt_seed_epochs='UNKNOWN' if unknown else failed_epochs,
                  seed_epochs_status='OBSERVED_LOWER_BOUND' if unknown else 'COMPLETE_RECORDED',
                  planned_seed_epochs=12*len(neural), seed_epoch_ceiling=108, ASSESS=0)
    if branch in ('COMPLETED', 'PARTIAL_FAILED'):
        execution = load_json(inputs.artifact('operation_trace', inputs.stage('08_reporting', 'execution.json')))
        for field, expected_value in (('neural_fits', len(neural)), ('CAL_terminal_scalars', len(scalar)), ('seed_epochs', observed_epochs)):
            audit.truth('execution_budget:' + field, execution[field] == expected_value)
    return counts


def trajectory_check(inputs,arm,seed,train,audit):
    receipt = load_json(inputs.artifact(f'{arm}_seed_{seed}_fit_receipt',inputs.stage('06_models',f'{arm}_seed_{seed}_fit_receipt.json')))
    epochs = receipt['epoch_losses']
    audit.truth(f'epochs:{arm}:{seed}',len(epochs)==12 and receipt['epochs_completed']==12)
    rng = np.random.default_rng(seed)
    expected_updates = math.ceil(len(train)/256)
    update_index = 0
    prop_updates=0
    updates = receipt['optimizer_update_trace']
    canonical=pd.read_parquet(inputs.artifact('series'))
    maps=pd.read_parquet(inputs.artifact('maps'))
    label_rows={'score':set(canonical.loc[canonical.score_index.notna(),'row_index'].astype(int))}
    for task,column in (('kills','kills'),('duration','duration_minutes')):
        label_rows[task]=set(maps.loc[maps[column].notna(),'row_index'].astype(int))
    for n,epoch in enumerate(epochs):
        order = rng.permutation(train)
        sha = hashlib.sha256(np.ascontiguousarray(order,dtype=np.int64).tobytes()).hexdigest()
        audit.truth(f'shuffle:{arm}:{seed}:{n}',receipt['epoch_order_sha256'][n]==sha)
        audit.truth(f'epoch_rows:{arm}:{seed}:{n}',epoch['series_seen']==len(train) and epoch['epoch']==n+1)
        for start in range(0,len(order),256):
            batch=order[start:start+256]
            entry=updates[update_index]
            audit.truth(f'update:{arm}:{seed}:{update_index}',entry['epoch']==n+1 and entry['step']==update_index+1 and entry['batch_series']==len(batch) and entry['indices_sha256']==hashlib.sha256(batch.tobytes()).hexdigest())
            audit.truth(f'update_objective_finite:{arm}:{seed}:{update_index}',math.isfinite(entry['objective']))
            for task,eligible in label_rows.items():
                expected_count=sum(int(i) in eligible for i in batch)
                audit.compare(f'update_observed_labels:{arm}:{seed}:{update_index}:{task}',entry['observed_series'][task],expected_count)
            prop_updates+=int(any(int(i) in label_rows['kills'] or int(i) in label_rows['duration'] for i in batch))
            update_index+=1
        for task in ('score','kills','duration'):
            expected_observed=sum(int(i) in label_rows[task] for i in train)
            audit.compare(f'epoch_observed_labels:{arm}:{seed}:{n}:{task}',epoch['observed_series'][task],expected_observed)
            loss=epoch['losses'][task]
            audit.truth(f'epoch_loss:{arm}:{seed}:{n}:{task}',loss is None if epoch['observed_series'][task]==0 else math.isfinite(loss))
    audit.compare(f'total_updates:{arm}:{seed}',len(updates),12*expected_updates)
    audit.compare(f'TRAIN_input_indices:{arm}:{seed}',receipt['train_indices_sha256'],hashlib.sha256(train.tobytes()).hexdigest())
    audit.truth(f'projector_rng:{arm}:{seed}',receipt['projector_seed']==(None if arm=='H' else 20261009))
    config = receipt['config']
    expected_config = dict(width=32,embedding_dim=8,epochs=12,batch_size=256,lr=.001,weight_decay=.0001,threads=4,kills_weight=.25,duration_weight=.25)
    compare_mapping(audit,f'trajectory_config:{arm}:{seed}',config,expected_config)
    optimizer_path = inputs.artifact(f'{arm}_seed_{seed}_optimizer',inputs.stage('06_models',f'{arm}_seed_{seed}_optimizer.pt'))
    optimizer = torch.load(optimizer_path,map_location='cpu',weights_only=False)
    audit.truth(f'optimizer_kind:{arm}:{seed}',receipt['optimizer']=='AdamW')
    for group in optimizer['param_groups']:
        audit.compare(f'optimizer_lr:{arm}:{seed}',group['lr'],.001)
        audit.compare(f'optimizer_decay:{arm}:{seed}',group['weight_decay'],.0001)
        audit.truth(f'optimizer_no_schedule:{arm}:{seed}',group.get('amsgrad',False) is False)
        audit.compare(f'optimizer_betas:{arm}:{seed}',group['betas'],(.9,.999))
        audit.compare(f'optimizer_epsilon:{arm}:{seed}',group['eps'],1e-8)
        audit.truth(f'optimizer_descent:{arm}:{seed}',group.get('maximize',False) is False)
    audit.truth(f'optimizer_state_nonempty:{arm}:{seed}',bool(optimizer['state']))
    final=load_state(inputs.artifact(f'{arm}_seed_{seed}_final_state',inputs.stage('06_models',f'{arm}_seed_{seed}_final_state.pt')))
    parameter_names=[name for name,value in final.items() if value.is_floating_point()]
    parameter_ids=[key for group in optimizer['param_groups'] for key in group['params']]
    audit.compare(f'optimizer_parameter_count:{arm}:{seed}',len(parameter_ids),len(parameter_names))
    parameter_map=dict(zip(parameter_ids,parameter_names))
    for key,state in optimizer['state'].items():
        if 'step' in state:
            expected_steps=prop_updates if parameter_map[key].startswith(('prop_head.','map_embedding.')) else 12*expected_updates
            audit.compare(f'optimizer_steps:{arm}:{seed}:{key}',float(state['step']),expected_steps)
        for name in ('exp_avg','exp_avg_sq'):
            audit.truth(f'optimizer_finite:{arm}:{seed}:{key}:{name}',name in state and torch.isfinite(state[name]).all().item())
            audit.compare(f'optimizer_moment_shape:{arm}:{seed}:{key}:{name}',state[name].shape,final[parameter_map[key]].shape)
    return receipt


def load_state(path):
    state = torch.load(path,map_location='cpu',weights_only=False)
    return state['state_dict'] if isinstance(state,dict) and 'state_dict' in state else state


def public_checks(model,state,arm,arrays,support,vocab,stats,series,audit):
    # Lexicographic first representatives, with coverage selected from source
    # state only (never labels, probabilities or prediction errors).
    order = sorted(range(len(series)),key=lambda i:str(series.iloc[i].match_id))
    cases = []
    predicates = {f'BO{b}':arrays['best_of']==b for b in (1,3,5)}
    predicates['masked_statistics']=(arrays['mask'][...,None]&~np.isfinite(arrays['history'][...,:12])).any(axis=(1,2,3,4))
    predicates['unknown_player']=series.unknown_player_count.to_numpy()>0
    predicates['cold'] = ~arrays['mask'].reshape(len(series),-1).any(1)
    for c in CHANNELS:
        codes=arrays[ARRAY_KEYS[c]]
        predicates[c+'_missing'] = ((codes==0)&arrays['mask']).reshape(len(series),-1).any(1)
        predicates[c+'_OOV'] = ((codes==1)&arrays['mask']).reshape(len(series),-1).any(1)
        predicates[c+'_seen'] = ((codes>=2)&arrays['mask']).reshape(len(series),-1).any(1)
        held_codes=[code for identity,code in vocab[c].items() if held(c,identity)]
        predicates[c+'_held'] = (np.isin(codes,held_codes)&arrays['mask']).reshape(len(series),-1).any(1)
    absent=[]
    for name,predicate in predicates.items():
        ix=next((i for i in order if predicate[i]),None)
        if ix is None:
            absent.append(name)
        elif ix not in cases:
            cases.append(ix)
    cases=np.array(cases[:32],np.int64)
    if not len(cases):
        raise ValueError('No public deterministic cases')
    local={k:np.array(v[cases],copy=True) for k,v in arrays.items()}
    local_support={c:np.array(support[c][cases],copy=True) for c in CHANNELS}
    ids={'n_prior':local_support}
    def fingerprint():
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in local.items()}
        hashes.update({c:hashlib.sha256(v.tobytes()).hexdigest() for c,v in local_support.items()})
        hashes['state']=hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for _,t in sorted(model.network.state_dict().items()))).hexdigest()
        hashes['normalizer']=joblib.hash(model.preprocessing)
        return hashes
    before=fingerprint()
    expected,expected_logits,_=replay(state,arm,local,np.arange(len(cases)),stats,local_support,vocab)
    actual=model.predict(local,identity_state=ids)
    audit.compare(f'public_batch:{arm}:{model.seed}',actual.score_probs,expected,'probability')
    audit.compare(f'public_batch_logits:{arm}:{model.seed}',actual.score_logits,expected_logits,'logit')
    for i in range(len(cases)):
        actual=model.predict(local,np.array([i]),identity_state=ids)
        audit.compare(f'public_singleton:{arm}:{model.seed}:{i}',actual.score_probs,expected[i:i+1],'probability')
        audit.compare(f'public_singleton_logits:{arm}:{model.seed}:{i}',actual.score_logits,expected_logits[i:i+1],'logit')
    swapped={k:v[:,::-1].copy() if v.ndim>=3 and v.shape[1]==2 else v.copy() for k,v in local.items()}
    swap_support={c:v[:,::-1].copy() for c,v in local_support.items()}
    actual=model.predict(swapped,identity_state={'n_prior':swap_support})
    expected_swap,swap_logits,_=replay(state,arm,swapped,np.arange(len(cases)),stats,swap_support,vocab)
    audit.compare(f'public_swap_replay:{arm}:{model.seed}',actual.score_probs,expected_swap,'probability')
    audit.compare(f'public_swap_logits:{arm}:{model.seed}',actual.score_logits,swap_logits,'logit')
    audit.compare(f'public_swap_reversal:{arm}:{model.seed}',actual.score_probs,expected[:,[3,4,5,0,1,2]],'probability')
    permutation=np.array([4,2,0,3,1])
    permuted={k:v[:,:,permutation].copy() if v.ndim>=4 and v.shape[1:3]==(2,5) else v.copy() for k,v in local.items()}
    perm_support={c:v[:,:,permutation].copy() for c,v in local_support.items()}
    actual=model.predict(permuted,identity_state={'n_prior':perm_support})
    audit.compare(f'public_player_bundle_permutation:{arm}:{model.seed}',actual.score_probs,expected,'probability')
    audit.compare(f'public_player_bundle_permutation_logits:{arm}:{model.seed}',actual.score_logits,expected_logits,'logit')
    cold={k:v.copy() for k,v in local.items()}
    cold['mask'][:]=False
    cold_expected,cold_logits,_=replay(state,arm,cold,np.arange(len(cases)),stats,local_support,vocab)
    cold_actual=model.predict(cold,identity_state=ids)
    audit.compare(f'public_cold_logits:{arm}:{model.seed}',cold_actual.score_logits,cold_logits,'logit')
    audit.compare(f'public_cold:{arm}:{model.seed}',cold_actual.score_probs,cold_expected,'probability')
    for key in ('champions','patches','competitions','roles'):
        cold[key][:]=0
    cold['history'][:]=np.nan
    cold_actual=model.predict(cold,identity_state=ids)
    audit.compare(f'public_padding_inert:{arm}:{model.seed}',cold_actual.score_probs,cold_expected,'probability')
    synthetic_states=[]
    for mode in ('missing','OOV','unknown_player','masked_stats','variable_counts'):
        fixture={k:v[:1].copy() for k,v in local.items()}
        fixture_support={c:v[:1].copy() for c,v in local_support.items()}
        if mode in ('missing','OOV'):
            fixture['mask'][:]=True
            for c in CHANNELS:
                fixture[ARRAY_KEYS[c]][:]=0 if mode=='missing' else 1
                fixture_support[c][:]=0 if mode=='missing' else 1000
        elif mode=='unknown_player':
            fixture['mask'][:]=False
            fixture['players'][:]=0
            fixture['role_probs'][:]=0
            fixture['role_probs'][...,0]=1
        elif mode=='masked_stats':
            fixture['history'][...,:12]=np.nan
        else:
            varied=np.arange(fixture['mask'].size,dtype=np.int64).reshape(fixture['mask'].shape)%137
            for c in CHANNELS:
                fixture_support[c]=np.where(fixture['mask']&(fixture[ARRAY_KEYS[c]]>=2),varied,0)
        law,logits,traces=replay(state,arm,fixture,np.array([0]),stats,fixture_support,vocab)
        actual=model.predict(fixture,identity_state={'n_prior':fixture_support})
        audit.compare(f'public_synthetic:{arm}:{model.seed}:{mode}',actual.score_probs,law,'probability')
        audit.compare(f'public_synthetic_logits:{arm}:{model.seed}:{mode}',actual.score_logits,logits,'logit')
        synthetic_states.append(mode)
    for bo in (0,2,7):
        invalid={k:v.copy() for k,v in local.items()}
        invalid['best_of'][:]=bo
        try:
            model.predict(invalid,identity_state=ids)
        except (ValueError,NotImplementedError):
            rejected=True
        else:
            rejected=False
        audit.truth(f'public_unknown_BO_rejected:{arm}:{model.seed}:{bo}',rejected)
    audit.truth(f'public_nonmutation:{arm}:{model.seed}',fingerprint()==before)
    return dict(case_indices=cases.tolist(),match_ids=series.iloc[cases].match_id.astype(str).tolist(),absent_in_source=absent,explicit_synthetic_contract_cases=synthetic_states)




def public_ensemble_checks(inputs,arm,arrays,support,stats,vocab,series,audit,selected):
    ensemble=joblib.load(inputs.artifact(arm+'_ensemble',inputs.stage('06_models',arm+'_ensemble.joblib')))
    selected=list(selected)
    ix=np.asarray(selected,np.int64)
    local={k:np.array(v[ix],copy=True) for k,v in arrays.items()}
    counts={c:np.array(support[c][ix],copy=True) for c in CHANNELS}
    pmfs=[replay(model.state_dict,arm,local,np.arange(len(ix)),stats,counts,vocab)[0] for model in ensemble.models]
    expected=np.mean([temper(p,ensemble.temperature) for p in pmfs],axis=0)
    before=joblib.hash(ensemble)
    actual=ensemble.predict(local,identity_state={'n_prior':counts})
    audit.compare('public_ensemble_batch:'+arm,actual.score_probs,expected,'probability')
    audit.compare('public_ensemble_length:'+arm,actual.length_probs,expected[:,:3]+expected[:,3:],'probability')
    for j in range(len(ix)):
        actual=ensemble.predict(local,np.array([j]),identity_state={'n_prior':counts})
        audit.compare(f'public_ensemble_singleton:{arm}:{j}',actual.score_probs,expected[j:j+1],'probability')
    swap={k:v[:,::-1].copy() if v.ndim>=3 and v.shape[1]==2 else v.copy() for k,v in local.items()}
    sc={c:v[:,::-1].copy() for c,v in counts.items()}
    actual=ensemble.predict(swap,identity_state={'n_prior':sc})
    audit.compare('public_ensemble_swap:'+arm,actual.score_probs,expected[:,[3,4,5,0,1,2]],'probability')
    audit.truth('public_ensemble_nonmutation:'+arm,joblib.hash(ensemble)==before)
    from src.analysis.champion_oov_reliability import predict_saved
    actual= predict_saved(inputs.run_id,series.iloc[ix].match_id.astype(str).tolist(),arm=arm,repo_root=inputs.repo,research_root=inputs.research,evidence_root=inputs.evidence)
    output_rows=actual['predictions'] if 'predictions' in actual else actual['rows']
    audit.compare('public_saved_ID_order:'+arm,[r['match_id'] for r in output_rows],series.iloc[ix].match_id.astype(str).to_numpy())
    audit.compare('public_saved_inference:'+arm,[r['terminal_probabilities'] for r in output_rows],expected,'probability')
    if arm==ARMS[0]:
        with tempfile.TemporaryDirectory(prefix='w1_oracle_root_rejection_') as directory:
            substitute=Path(directory)
            write_json(substitute/'w1_contract.json',inputs.contract)
            rejected=False
            try:
                predict_saved(inputs.run_id,series.iloc[ix].match_id.astype(str).tolist(),arm=arm,
                              repo_root=inputs.repo,research_root=inputs.research,evidence_root=substitute)
            except ValueError as error:
                rejected='evidence root' in str(error)
            audit.truth('public_saved_bound_root_substitution_rejected',rejected)


def model_checks(inputs,arrays,series,support,vocab,stats,audit,output,partial=False):
    train=np.flatnonzero(series.split.to_numpy()=='TRAIN')
    raw_laws={a:{} for a in ARMS}
    saved_laws={a:{} for a in ARMS}
    public=[]
    states_count=0
    members, scalars = retained_operations(inputs, audit)
    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) not in members:
                continue
            prefix=f'{arm}_seed_{seed}'
            state=load_state(inputs.artifact(prefix+'_final_state',inputs.stage('06_models',prefix+'_final_state.pt')))
            initial=load_state(inputs.artifact(prefix+'_initial_state',inputs.stage('06_models',prefix+'_initial_state.pt')))
            states_count+=1
            audit.truth('state_keys:'+prefix,set(state)==set(initial))
            for key in state:
                audit.truth('state_shape:'+prefix+':'+key,state[key].shape==initial[key].shape and state[key].dtype==initial[key].dtype and (not state[key].is_floating_point() or torch.isfinite(state[key]).all().item()))
            extra=sum(v.numel() for k,v in state.items() if k.startswith('projectors.'))
            audit.compare('added_parameters:'+prefix,extra,0 if arm=='H' else 592)
            for c,width in zip(CHANNELS,(8,4,4)):
                audit.compare('embedding_width:'+prefix+':'+c,state[c+'.weight'].shape[1],width)
                if arm!='H':
                    audit.compare('projector_shape:'+prefix+':'+c,state['projectors.'+c+'.weight'].shape,(width,36))
            trajectory_check(inputs,arm,seed,train,audit)
            expected_initial=independent_initial_state(arrays,train,pd.read_parquet(inputs.artifact('maps')),seed,arm)
            audit.truth('initial_backbone_keys:'+prefix,set(initial)==set(expected_initial))
            for key,value in expected_initial.items():
                audit.compare('initial_seed_stream:'+prefix+':'+key,initial[key].cpu().numpy(),value.cpu().numpy())
            model=joblib.load(inputs.artifact(prefix,inputs.stage('06_models',prefix+'.joblib')))
            for key in state:
                audit.compare('joblib_state:'+prefix+':'+key,model.network.state_dict()[key].cpu().numpy(),state[key].cpu().numpy())
            for key in stats:
                for field,attribute in (('mean','mean'),('scale','scale'),('count','counts')):
                    audit.compare('model_normalizer:'+prefix+':'+key+':'+field,getattr(model.preprocessing,attribute)[key],stats[key][field],'normalizer' if field!='count' else 'exact')
            for split in ('TRAIN','CAL','ASSESS'):
                ix=np.flatnonzero(series.split.to_numpy()==split)
                role = f'{arm}_{split}_seed_raw_terminal'
                fallback = inputs.stage('07_model_output', role + '.parquet')
                path = inputs.available(role, fallback) if partial else inputs.artifact(role, fallback)
                if partial and split == 'ASSESS' and path is None:
                    continue
                frame = pd.read_parquet(path) if path is not None else None
                if frame is not None:
                    frame=frame[frame['seed']==seed].reset_index(drop=True)
                ss=series.iloc[ix].reset_index(drop=True)
                if frame is not None:
                    series_order(frame,ss,audit,prefix+':'+split)
                    saved=probabilities(frame)
                else:
                    # Re-query retained weights, never fit; do not pretend an
                    # unavailable saved export was compared.
                    saved=model.predict(arrays,ix,identity_state={'n_prior':support}).score_probs
                    inputs.unavailable_comparisons.append(role + ':' + str(seed))
                expected=np.empty((len(ix),6))
                logits=np.empty((len(ix),6))
                for begin in range(0,len(ix),256):
                    block=ix[begin:begin+256]
                    law,logit,traces=replay(state,arm,arrays,block,stats,support,vocab)
                    expected[begin:begin+len(block)]=law
                    logits[begin:begin+len(block)]=logit
                    if 'composition_preprocessing' in model.__dict__:
                        actual_phi=model.composition_preprocessing.transform({k:v[block] for k,v in arrays.items()})
                        audit.compare(f'normalized_phi:{prefix}:{split}:{begin}',actual_phi,phi_batch(arrays,block,stats),'phi')
                audit.compare('full_raw_law:'+prefix+':'+split,saved,expected,'probability')
                if frame is not None:
                    logit_cols=[f'logit_{i}' for i in range(6)]
                    audit.truth('exported_logits:'+prefix+':'+split,all(c in frame for c in logit_cols))
                    if all(c in frame for c in logit_cols):
                        audit.compare('full_logits:'+prefix+':'+split,frame[logit_cols].to_numpy(),logits,'logit')
                score_laws(saved,ss)
                np.savez_compressed(output/f'{prefix}_{split}_replay.npz',pmf=expected,logits=logits,row_indices=ix)
                raw_laws[arm].setdefault(split,[]).append(expected)
                saved_laws[arm].setdefault(split,[]).append(saved)
            public.append(public_checks(model,state,arm,arrays,support,vocab,stats,series,audit))
            del model,state,initial
    ensembles={}
    for arm in ARMS:
        if arm not in scalars:
            continue
        receipt=load_json(inputs.artifact(arm+'_temperature',inputs.stage('06_models',arm+'_temperature.json')))
        cal=series[series.split=='CAL'].reset_index(drop=True)
        labels=cal.score_index.to_numpy(float)
        observed=np.isfinite(labels)
        cal_indices=np.flatnonzero(series.split.to_numpy()=='CAL')
        audit.compare('temperature_label_row_indices:'+arm,receipt['label_row_indices'],cal_indices[observed])
        audit.compare('temperature_label_classes:'+arm,receipt['label_score_indices'],labels[observed].astype(int))
        t=scalar_check(receipt,[p[observed] for p in saved_laws[arm]['CAL']],labels[observed].astype(int),audit,arm)
        ensembles[arm]={}
        for split in ('TRAIN','CAL','ASSESS'):
            ss=series[series.split==split].reset_index(drop=True)
            if split not in raw_laws[arm]:
                inputs.unavailable_comparisons.append(arm + ':' + split)
                continue
            expected=np.mean([temper(p,t) for p in raw_laws[arm][split]],axis=0)
            saved_expected=np.mean([temper(p,t) for p in saved_laws[arm][split]],axis=0)
            member_role=f'{arm}_{split}_seed_tempered_terminal'
            member_fallback=inputs.stage('07_model_output',member_role+'.parquet')
            member_path=inputs.available(member_role,member_fallback) if partial else inputs.artifact(member_role,member_fallback)
            if member_path is not None:
                members_frame=pd.read_parquet(member_path)
                for seed,p in zip(SEEDS,saved_laws[arm][split]):
                    mf=members_frame[members_frame.seed==seed].reset_index(drop=True)
                    series_order(mf,ss,audit,f'tempered:{arm}:{split}:{seed}')
                    audit.compare(f'tempered:{arm}:{split}:{seed}',probabilities(mf),temper(p,t),'metric')
            else:
                inputs.unavailable_comparisons.append(member_role)
            role=f'{arm}_{split}_terminal'
            fallback=inputs.stage('07_model_output',role+'.parquet')
            export_path=inputs.available(role,fallback) if partial else inputs.artifact(role,fallback)
            if export_path is None:
                inputs.unavailable_comparisons.append(role)
                continue
            exported=pd.read_parquet(export_path)
            series_order(exported,ss,audit,'ensemble:'+arm+':'+split)
            actual=probabilities(exported)
            audit.compare('ensemble_replay:'+arm+':'+split,actual,expected,'probability')
            audit.compare('ensemble_saved_law_arithmetic:'+arm+':'+split,actual,saved_expected,'metric')
            audit.compare('ensemble_winner:'+arm+':'+split,exported.p_win.to_numpy(float),actual[:,:3].sum(1),'metric')
            actual_scores=score_laws(actual,ss)
            expected_scores=score_laws(expected,ss)
            aggregate,expanded=law_metrics(actual,ss)
            metrics_path=inputs.available('metrics',inputs.stage('08_reporting','metrics.json')) if partial else inputs.artifact('metrics',inputs.stage('08_reporting','metrics.json'))
            if metrics_path is not None:
                reported_metrics=load_json(metrics_path)[arm][split]
                compare_mapping(audit,f'aggregate_metrics:{arm}:{split}',reported_metrics,aggregate)
            else:
                inputs.unavailable_comparisons.append('metrics:'+arm+':'+split)
            score_role=f'{arm}_{split}_series_scores'
            score_fallback=inputs.stage('07_model_output',score_role+'.parquet')
            score_path=inputs.available(score_role,score_fallback) if partial else inputs.artifact(score_role,score_fallback)
            if score_path is not None:
                score_frame=pd.read_parquet(score_path)
                series_order(score_frame,ss,audit,'scores:'+arm+':'+split)
                for field in ('length_nll','length_rps','map_play_brier'):
                    audit.compare(f'length_and_play_scores:{arm}:{split}:{field}',score_frame[field].to_numpy(float),expanded[field],'metric')
                for field in ('winner_nll','winner_brier','score_nll'):
                    audit.compare('saved_law_scores:'+arm+':'+split+':'+field,score_frame[field].to_numpy(float),actual_scores[field],'metric')
                    audit.compare('replayed_scores:'+arm+':'+split+':'+field,score_frame[field].to_numpy(float),expected_scores[field],'replayed_score')
            else:
                inputs.unavailable_comparisons.append(score_role)
            true_probability=np.where(ss.winner.to_numpy()==1,actual[:,:3].sum(1),actual[:,3:].sum(1))
            audit.truth('score_sensitivity:'+arm+':'+split,np.all(true_probability[np.isfinite(ss.winner.to_numpy())]>0),dict(minimum_true_probability=float(true_probability.min()),largest_LL_sensitivity=float((1/true_probability).max())))
            ensembles[arm][split]=actual_scores
        if not partial:
            public_ensemble_checks(inputs,arm,arrays,support,stats,vocab,series,audit,public[0]['case_indices'])
        else:
            ensemble=joblib.load(inputs.artifact(arm+'_ensemble',inputs.stage('06_models',arm+'_ensemble.joblib')))
            audit.compare('retained_ensemble_temperature:'+arm,ensemble.temperature,t)
            cal_ix=np.flatnonzero(series.split.to_numpy()=='CAL')
            actual=ensemble.predict(arrays,cal_ix,identity_state={'n_prior':support}).score_probs
            audit.compare('retained_ensemble_law:'+arm,actual,np.mean([temper(p,t) for p in saved_laws[arm]['CAL']],axis=0),'metric')
    if not partial:
        public_export_checks(inputs,arrays,series,support,stats,vocab,audit,output)
        d0_replay_checks(inputs,arrays,series,support,stats,vocab,audit,output)
    return ensembles,states_count,public


def references(inputs,series,audit):
    contract=inputs.contract
    paths=contract['input_paths']
    assess=series[series.split=='ASSESS'].reset_index(drop=True)
    result={}
    for arm,key in (('D0','saved_d0_assessment_outputs'),('C0','saved_c0_assessment_outputs')):
        score_path=next(p for p in paths[key] if p.endswith('series_scores.parquet'))
        frame=pd.read_parquet(inputs.path(score_path))
        series_order(frame,assess,audit,'old_reference:'+arm)
        result[arm]={}
        for name in ('winner_nll','winner_brier'):
            candidates=(name,'winner_ll' if name=='winner_nll' else 'brier')
            col=next((c for c in candidates if c in frame),None)
            if col is None:
                raise ValueError('Old reference missing '+name)
            result[arm][name]=frame[col].to_numpy(float)
        terminal_path=next(p for p in paths[key] if 'terminal.parquet' in p)
        terminal=pd.read_parquet(inputs.path(terminal_path))
        series_order(terminal,assess,audit,'old_terminal:'+arm)
        scores=score_laws(probabilities(terminal),assess)
        for name in ('winner_nll','winner_brier'):
            audit.compare('old_reference_score:'+arm+':'+name,result[arm][name],scores[name],'metric')
    manifest=load_json(inputs.path(paths['canonical_manifest_read_only']))
    metadata=pd.read_parquet(inputs.path(manifest['input']['path']))
    audit.truth('native_C0_metadata_pin',digest(inputs.path(manifest['input']['path']))==manifest['input']['sha256'])
    metadata=metadata.assign(_id=metadata.golgg_match_id.astype(str)).set_index('_id')
    native=pd.read_parquet(inputs.path(paths['native_c0_output']))
    native=native.assign(_id=native.golgg_match_id.astype(str)).set_index('_id')
    audit.truth('native_C0_unique_ids',metadata.index.is_unique and native.index.is_unique)
    ids=assess.match_id.astype(str).to_numpy()
    audit.truth('native_C0_all_assessment_present',all(mid in metadata.index and mid in native.index for mid in ids))
    m=metadata.loc[ids]
    n=native.loc[ids]
    direct=(m.team1_id.astype(str).to_numpy()==assess.team_a.astype(str).to_numpy())&(m.team2_id.astype(str).to_numpy()==assess.team_b.astype(str).to_numpy())
    reverse=(m.team1_id.astype(str).to_numpy()==assess.team_b.astype(str).to_numpy())&(m.team2_id.astype(str).to_numpy()==assess.team_a.astype(str).to_numpy())
    audit.truth('native_C0_orientation_only_participants',np.all(direct^reverse))
    audit.compare('native_C0_dates',pd.to_datetime(m.date).dt.strftime('%Y-%m-%d').to_numpy(),assess.date.astype(str).to_numpy())
    audit.compare('native_C0_best_of',n.best_of.to_numpy(int),assess.best_of.to_numpy(int))
    p=n[paths['native_c0_column']].to_numpy(float)
    p=np.where(reverse,1-p,p)
    winner=assess.winner.to_numpy(float)
    label_prob=np.where(winner==1,p,1-p)
    observed=np.isfinite(winner)
    audit.truth('native_C0_finite_legal',np.isfinite(p).all() and ((p>=0)&(p<=1)).all() and (label_prob[observed]>0).all())
    expected_nll=np.where(observed,-np.log(label_prob),np.nan)
    expected_brier=np.where(observed,(p-winner)**2,np.nan)
    audit.compare('native_C0_independent_winner_nll',result['C0']['winner_nll'],expected_nll,'metric')
    audit.compare('native_C0_independent_brier',result['C0']['winner_brier'],expected_brier,'metric')
    return result


def d0_replay_checks(inputs,arrays,series,support,stats,vocab,audit,output):
    contract=inputs.contract
    ensemble=joblib.load(inputs.path(contract['input_paths']['saved_d0_ensemble']))
    temperature=float(ensemble.score_temperature)
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Original D0 saved temperature invalid')
    def original_temper(p):
        with np.errstate(divide='ignore'):
            lp=np.log(p)/temperature
        return np.exp(lp-logsumexp(lp,axis=-1,keepdims=True))
    for split in ('TRAIN','CAL','ASSESS'):
        ix=np.flatnonzero(series.split.to_numpy()==split)
        ss=series.iloc[ix].reset_index(drop=True)
        actual_raw=pd.read_parquet(inputs.artifact(f'D0_{split}_seed_raw_terminal',inputs.stage('07_model_output',f'D0_{split}_seed_raw_terminal.parquet')))
        actual_tempered=pd.read_parquet(inputs.artifact(f'D0_{split}_seed_tempered_terminal',inputs.stage('07_model_output',f'D0_{split}_seed_tempered_terminal.parquet')))
        independent,saved=[],[]
        for model,seed in zip(ensemble.models,SEEDS):
            state=model.state_dict
            expected=np.empty((len(ix),6))
            for begin in range(0,len(ix),256):
                expected[begin:begin+256]=replay(state,'D0',arrays,ix[begin:begin+256],stats,support,vocab)[0]
            frame=actual_raw[actual_raw.seed==seed].reset_index(drop=True)
            series_order(frame,ss,audit,f'D0_raw_ids:{split}:{seed}')
            law=probabilities(frame)
            audit.compare(f'D0_saved_weight_replay:{split}:{seed}',law,expected,'probability')
            tempered=actual_tempered[actual_tempered.seed==seed].reset_index(drop=True)
            series_order(tempered,ss,audit,f'D0_tempered_ids:{split}:{seed}')
            audit.compare(f'D0_original_temperature:{split}:{seed}',probabilities(tempered),original_temper(law),'metric')
            independent.append(original_temper(expected))
            saved.append(original_temper(law))
        frame=pd.read_parquet(inputs.artifact(f'D0_{split}_terminal',inputs.stage('07_model_output',f'D0_{split}_terminal.parquet')))
        series_order(frame,ss,audit,'D0_ensemble_ids:'+split)
        audit.compare('D0_ensemble_independent:'+split,probabilities(frame),np.mean(independent,axis=0),'probability')
        audit.compare('D0_ensemble_saved_law:'+split,probabilities(frame),np.mean(saved,axis=0),'metric')
        np.savez_compressed(output/f'D0_{split}_replay.npz',pmf=np.mean(independent,axis=0),row_indices=ix)


def stage0_descriptives(inputs,series,groups,refs,audit,output):
    ss=series[series.split=='ASSESS'].reset_index(drop=True)
    assess=series.split.to_numpy()=='ASSESS'
    months=ss.date.astype(str).str[:7].to_numpy()
    actual=load_json(inputs.artifact('stage0_comparisons',inputs.stage('08_reporting','stage0_comparisons.json')))['comparisons']
    delta=refs['D0']['winner_nll']-refs['C0']['winner_nll']
    for group in ('natural_champion_oov','seen_champion_history'):
        mask=groups[group][assess]&np.isfinite(ss.winner.to_numpy(float))
        expected,draws=bootstrap_expected(delta,months,mask,MONTHS)
        compare_mapping(audit,'stage0_descriptive:'+group,actual[group],expected)
        export_draw_check(inputs,audit,'stage0_'+group,draws,delta,mask,refs['D0']['winner_nll'],refs['C0']['winner_nll'],ss.match_id.astype(str).to_numpy(),months)
        np.savez_compressed(output/('stage0_'+group+'.npz'),**draws,delta=delta,mask=mask)
    for arm in ('D0','C0'):
        frame=pd.read_parquet(inputs.artifact(arm+'_ASSESS_series_scores',inputs.stage('07_model_output',arm+'_ASSESS_series_scores.parquet')))
        series_order(frame,ss,audit,'stage0_reference:'+arm)
        for metric in ('winner_nll','winner_brier'):
            audit.compare('stage0_reference_score:'+arm+':'+metric,frame[metric].to_numpy(float),refs[arm][metric],'metric')


def public_export_checks(inputs,arrays,series,support,stats,vocab,audit,output):
    cases=load_json(inputs.artifact('public_cases',inputs.stage('07_model_output','public_cases.json')))
    cases=cases.get('cases',cases) if isinstance(cases,dict) else cases
    audit.truth('public_case_fields',all(all(k in c for k in ('case_id','study','arm','seed_or_ensemble','input_reference_and_sha256','target_origin','scenario','contract_state','expected_status','expected_reason','public_result_reference','independent_expected_reference','comparison_kind','tolerances','state_hash_before','state_hash_after')) for c in cases))
    ids=sorted({str(c['match_id']) for c in cases if c.get('match_id') is not None})
    lookup={str(mid):i for i,mid in enumerate(series.match_id)}
    ix=np.asarray([lookup[mid] for mid in ids],np.int64)
    query=np.load(inputs.artifact('public_query_inputs',inputs.stage('07_model_output','public_query_inputs.npz')),allow_pickle=False)
    for key in arrays:
        audit.compare('exported_public_input:'+key,query[key],arrays[key][ix])
    for c in CHANNELS:
        audit.compare('exported_public_sidecar:'+c,query['n_prior_'+c],support[c][ix])
    predictions=pd.read_parquet(inputs.artifact('public_predictions',inputs.stage('07_model_output','public_predictions.parquet')))
    result=[]
    for arm in ARMS:
        laws=[]
        for seed in SEEDS:
            state=load_state(inputs.artifact(f'{arm}_seed_{seed}_final_state',inputs.stage('06_models',f'{arm}_seed_{seed}_final_state.pt')))
            expected,logits,_=replay(state,arm,arrays,ix,stats,support,vocab)
            actual=predictions[(predictions.arm==arm)&(predictions.seed==seed)&(predictions.public_kind=='member')].reset_index(drop=True)
            series_order(actual,series.iloc[ix].reset_index(drop=True),audit,f'exported_public_ids:{arm}:{seed}')
            audit.compare(f'exported_public_member:{arm}:{seed}',probabilities(actual),expected,'probability')
            audit.compare(f'exported_public_logit:{arm}:{seed}',actual[[f'logit_{i}' for i in range(6)]].to_numpy(),logits,'logit')
            audit.compare(f'exported_public_nonmutation:{arm}:{seed}',actual.state_hash_before.to_numpy(),actual.state_hash_after.to_numpy())
            laws.append(expected)
        temperature=load_json(inputs.artifact(arm+'_temperature',inputs.stage('06_models',arm+'_temperature.json')))['temperature']
        expected=np.mean([temper(law,temperature) for law in laws],axis=0)
        actual=predictions[(predictions.arm==arm)&(predictions.public_kind=='ensemble')].reset_index(drop=True)
        series_order(actual,series.iloc[ix].reset_index(drop=True),audit,'exported_public_ensemble_ids:'+arm)
        audit.compare('exported_public_ensemble:'+arm,probabilities(actual),expected,'probability')
        for j,mid in enumerate(ids):
            result.append(dict(match_id=mid,arm=arm,score_probs=expected[j].tolist()))
    write_json(output/'independent_public_expected.json',result)
    # For each observed state the source-derived deterministic first ID must be
    # used. Absent states explicitly remain absence, never invented support.
    rows=pd.read_parquet(inputs.artifact('stage0_series',inputs.stage('04_feature','stage0_series.parquet')))
    for c in cases:
        name=c['contract_state']
        if name.startswith('BO') and name[2:].isdigit():
            mask=rows.best_of.to_numpy()==int(name[2:])
        elif name in rows:
            mask=rows[name].to_numpy(bool)
        elif name=='unknown_player':
            mask=rows.unknown_player_count.to_numpy()>0
        elif name.startswith('history_'):
            mask=rows.history_profile.to_numpy()==name[len('history_'):]
        elif name in ('UNKNOWN_rule','rule_UNKNOWN'):
            mask=rows.draft_mode.to_numpy()=='UNKNOWN'
        elif name.startswith('held'):
            held_codes=[code for identity,code in vocab['champion'].items() if held('champion',identity)]
            mask=(np.isin(arrays['champions'],held_codes)&arrays['mask']).any(axis=(1,2,3))
        else:
            audit.truth('public_case_unrecognized:'+name,False,'Membership rule not pinned in verifier')
            continue
        selected=rows.loc[mask].sort_values('match_id')
        expected_id=None if selected.empty else str(selected.iloc[0].match_id)
        audit.truth('public_case_deterministic:'+name,c.get('match_id')==expected_id)
        if expected_id is None:
            audit.truth('public_case_absent:'+name,c.get('synthetic_contract_case') is True)


def decision_checks(inputs,series,groups,ensembles,refs,audit,output,arrays):
    rows=pd.read_parquet(inputs.artifact('stage0_series',inputs.stage('04_feature','stage0_series.parquet')))
    assess=rows[rows.split=='ASSESS'].reset_index(drop=True)
    assess_mask=series.split.to_numpy()=='ASSESS'
    membership={k:v[assess_mask] for k,v in groups.items()}
    months=assess.date.astype(str).str[:7].to_numpy()
    laws={a:ensembles[a]['ASSESS'] for a in ARMS}|refs
    primary=load_json(inputs.artifact('comparisons',inputs.stage('08_reporting','comparisons.json')))
    guards=load_json(inputs.artifact('guards',inputs.stage('08_reporting','guards.json')))
    brier_receipt=load_json(inputs.artifact('brier_guards',inputs.stage('08_reporting','brier_guards.json')))
    proxy_receipt=load_json(inputs.artifact('proxy_sensitivity',inputs.stage('08_reporting','proxy_sensitivity.json')))
    audit.truth('four_primary_slots',len(primary)==4)
    audit.truth('nine_formal_guards',len(guards)==9)
    slot_status={}
    for slot,reference,natural in ((3,'D0',True),(4,'H',True),(5,'C',True),(6,'C0',False)):
        group='natural_champion_oov' if natural else 'all'
        key=f'HC-{reference}:{group}:winner_nll'
        mask=membership['natural_champion_oov'] if natural else np.ones(len(assess),bool)
        mask=mask & np.isfinite(assess.winner.to_numpy(float))
        delta=laws['HC']['winner_nll']-laws[reference]['winner_nll']
        expected,draw_arrays=bootstrap_expected(delta,months,mask,MONTHS,min_rows=200,min_blocks=6)
        compare_mapping(audit,f'primary:{slot}',primary[key],expected)
        brier=float((laws['HC']['winner_brier']-laws[reference]['winner_brier'])[mask].mean()) if mask.any() else None
        compare_mapping(audit,f'primary_brier:{slot}',brier_receipt['primary'][key],dict(point=brier,nonworse=brier is not None and brier<=0))
        passes=expected['status']=='ESTIMABLE' and expected['upper']<0 and expected['equal_block']['point']<=0 and brier is not None and brier<=0
        export_draw_check(inputs,audit,f'primary_slot_{slot}',draw_arrays,delta,mask,laws['HC']['winner_nll'],laws[reference]['winner_nll'],assess.match_id.astype(str).to_numpy(),months)
        np.savez_compressed(output/f'primary_slot_{slot}.npz',**draw_arrays,delta=delta,mask=mask)
        slot_status[str(slot)]='PASS' if passes else expected['status'] if expected['status']!='ESTIMABLE' else 'FAIL'
    group_masks={'BO3':assess.best_of.to_numpy()==3,'BO5':assess.best_of.to_numpy()==5,
                 'major':assess.major_state.to_numpy()=='major','player_source_unseen':assess.player_source_state.to_numpy()=='unseen',
                 'seen_champion_history':membership['seen_champion_history']}
    audit.compare('taxonomy_major_mask',assess.major.to_numpy(bool),group_masks['major'])
    audit.compare('taxonomy_source_unseen_mask',assess.player_source_unseen.to_numpy(bool),group_masks['player_source_unseen'])
    guard_status={}
    for name,reference in [(g,r) for g in ('BO3','BO5','major','player_source_unseen') for r in ('D0','C0')]+[('seen_champion_history','D0')]:
        mask=group_masks[name]&np.isfinite(assess.winner.to_numpy(float))
        delta=laws['HC']['winner_nll']-laws[reference]['winner_nll']
        expected,draw_arrays=bootstrap_expected(delta,months,mask,MONTHS,confidence=1-.05/9,one_sided=True,min_rows=100,min_blocks=4)
        entry=guards[f'HC-{reference}:{name}']
        compare_mapping(audit,f'guard:{name}:{reference}',entry,expected)
        passed=expected['status']=='ESTIMABLE' and expected['upper']<=.005
        audit.truth(f'guard_pass_claim:{name}:{reference}',entry['pass']==passed)
        export_draw_check(inputs,audit,f'guard_{name}_{reference}',draw_arrays,delta,mask,laws['HC']['winner_nll'],laws[reference]['winner_nll'],assess.match_id.astype(str).to_numpy(),months)
        np.savez_compressed(output/f'guard_{name}_{reference}.npz',**draw_arrays,delta=delta,mask=mask)
        guard_status[name+':'+reference]='PASS' if passed else 'FAIL'
    names=assess['tournament_source_name'].fillna('UNKNOWN').astype(str).to_numpy(dtype=str)
    proxy=np.char.add(np.char.add(names,'|'),assess.date.astype(str).str[:4].to_numpy(dtype=str))
    universe=sorted(set(proxy))
    for slot,reference,natural in ((3,'D0',True),(4,'H',True),(5,'C',True),(6,'C0',False)):
        group='natural_champion_oov' if natural else 'all'
        key=f'HC-{reference}:{group}:winner_nll'
        mask=(membership['natural_champion_oov'] if natural else np.ones(len(assess),bool))&np.isfinite(assess.winner.to_numpy(float))
        delta=laws['HC']['winner_nll']-laws[reference]['winner_nll']
        expected,draw_arrays=bootstrap_expected(delta,proxy,mask,universe,confidence=.95,min_rows=200,min_blocks=6)
        compare_mapping(audit,f'proxy95:{slot}',proxy_receipt[key],expected)
        export_draw_check(inputs,audit,f'proxy_slot_{slot}',draw_arrays,delta,mask,laws['HC']['winner_nll'],laws[reference]['winner_nll'],assess.match_id.astype(str).to_numpy(),proxy)
        np.savez_compressed(output/f'proxy_slot_{slot}.npz',**draw_arrays,proxy_labels=proxy)
    whole=float((laws['HC']['winner_brier']-laws['C0']['winner_brier']).mean())
    compare_mapping(audit,'whole_cohort_brier',brier_receipt['whole_HC_C0'],dict(point=whole,nonworse=whole<=0))
    diagnostics=load_json(inputs.artifact('diagnostics',inputs.stage('08_reporting','diagnostics.json')))
    membership['masked_statistics']=(~np.isfinite(arrays['history'][assess_mask,...,:12]) & arrays['mask'][assess_mask,...,None]).any(axis=(1,2,3,4))
    for group in ('cold_history','missing_champion_identity','patch_oov','competition_oov','masked_statistics'):
        for reference in ('D0','C0'):
            mask=membership[group]&np.isfinite(assess.winner.to_numpy(float))
            expected,_=bootstrap_expected(laws['HC']['winner_nll']-laws[reference]['winner_nll'],months,mask,MONTHS)
            compare_mapping(audit,f'diagnostic:{group}:{reference}',diagnostics[group][reference],expected)
    decision=load_json(inputs.artifact('decision',inputs.stage('08_reporting','decision.json')))
    may_advance=all(v=='PASS' for v in slot_status.values()) and all(v=='PASS' for v in guard_status.values()) and whole<=0
    audit.truth('decision_scientific_gates',decision['gates_passed']==may_advance)
    audit.truth('decision_claim',decision['status']==('DEVELOPMENT_CANDIDATE' if may_advance else 'NO_ADOPTION'))
    return slot_status,guard_status


def prepare_checks(inputs,series,arrays,stats,inventory,supported,audit,output):
    splits=series.split.to_numpy()
    for split,n in (('TRAIN',36205),('CAL',1328),('ASSESS',3103)):
        audit.compare('canonical_partition_count:'+split,np.count_nonzero(splits==split),n)
    audit.truth('all_partitions_explicit',np.isin(splits,('TRAIN','CAL','ASSESS','UNASSIGNED')).all())
    dates=series.date.astype(str).to_numpy()
    release=series.effective_release_day.astype(str).to_numpy()
    expected=np.where((dates<'2025-07-01')&(release<'2025-07-01'),'TRAIN',np.where((dates>='2025-07-01')&(dates<'2026-01-01')&(release<'2026-01-01'),'CAL',np.where((dates>='2026-01-01')&(dates<'2027-01-01'),'ASSESS','UNASSIGNED')))
    audit.compare('temporal_partitions',splits,expected)
    audit.compare('best_of_identity',arrays['best_of'],series.best_of.to_numpy())
    assessment=series[splits=='ASSESS']
    for bo,n in ((1,745),(3,1873),(5,485)):
        audit.compare('ASSESS_BO'+str(bo),np.count_nonzero(assessment.best_of.to_numpy()==bo),n)
    audit.compare('ASSESS_date_range',[assessment.date.min(),assessment.date.max()],['2026-01-08','2026-09-08'])
    audit.truth('component_support_not_receipt_echo',inventory['component_contract']['available']==bool((stats['history']['count'][list(NUMERIC)]>0).all()) and arrays['history'].shape[-1]==20 and arrays['role_probs'].shape[-1]==6)
    normalizer=load_json(inputs.artifact('composition_preprocessor',inputs.stage('04_feature','composition_preprocessor.json')))
    for key in stats:
        for field in ('mean','scale','count'):
            plural='counts' if field=='count' else field
            audit.compare('normalizer:'+key+':'+field,normalizer[plural][key],stats[key][field],'normalizer' if field!='count' else 'exact')
        sufficient = normalizer[key+'_sufficient_statistics']
        for field,saved in (('count','counts'),('sum','sums'),('sum_of_squares','sums_of_squares')):
            audit.compare('sufficient_statistics:'+key+':'+field,sufficient[saved],stats[key][field],'normalizer' if field!='count' else 'exact')
        audit.truth('normalizer_ddof:'+key,sufficient['ddof']==0)
    audit.compare('phi_numeric_indices',normalizer['source_indices'],NUMERIC)
    train=np.flatnonzero(splits=='TRAIN')
    np.savez_compressed(output/'normalizers.npz',**{key+'_'+field:value for key,s in stats.items() for field,value in s.items()},train_indices=train)
    gate=inventory.get('support_gate',inventory.get('gate',{}))
    status=gate.get('status',inventory.get('status'))
    audit.truth('independent_stage0_support_decision',(status=='SUPPORT_PASS')==supported)
    return not supported


def closure_checks(inputs,audit,supported,counts):
    partial = inputs.index['branch_status'] == 'FIT_FAILED_WITHIN_BUDGET'
    status = 'NOT_RUN_FIT_FAILED' if partial else 'NOT_RUN_BLOCKED'
    final_integrity_failure=False
    if partial:
        execution=load_json(inputs.artifact('operation_trace',inputs.stage('08_reporting','execution.json')))
        final_integrity_failure=execution.get('failure_phase')=='FINAL_INTEGRITY'
        if final_integrity_failure:
            status='NOT_AUTHORIZING_INTEGRITY_FAILED'
    if inputs.index['branch_status']=='CLOSED_BY_SUPPORT':
        audit.truth('closure_independent_support',not supported)
    if not partial:
        prohibited=('final_state','initial_state','optimizer','seed_raw_terminal','seed_tempered_terminal')
        for d in inputs.descriptors:
            if any(s in d.get('role','') for s in prohibited):
                audit.truth('closure_no_candidate_artifact:'+d['role'],d.get('absence_status') not in ('AVAILABLE','PRESENT',None) and d.get('absence_reason') is not None)
        for stage in ('06_models','07_model_output'):
            root=inputs.repo/inputs.stage(stage,'')
            if root.exists():
                audit.truth('closure_no_candidate_files:'+stage,not any(p.name.startswith(('H_','C_','HC_')) and p.suffix in ('.pt','.joblib','.parquet','.npz') for p in root.iterdir()))
    decision=load_json(inputs.artifact('decision',inputs.stage('08_reporting','decision.json')))
    audit.truth('closure_no_adoption',decision['status'] in ('CLOSED_BY_SUPPORT','CLOSED_NO_FITS','HARD_STOP_BEFORE_FIT','FIT_FAILED_WITHIN_BUDGET') and decision['decision']=='NO_ADOPTION')
    if partial:
        for field, key in (('neural_fits','neural'),('CAL_terminal_scalars','CAL_temperature'),('seed_epochs','seed_epochs')):
            audit.truth('closure_budget:' + field, decision.get(field) == counts[key])
    comparisons=load_json(inputs.artifact('comparisons',inputs.stage('08_reporting','comparisons.json')))
    expected_keys={f'HC-{ref}:{group}:winner_nll' for ref,group in (('D0','natural_champion_oov'),('H','natural_champion_oov'),('C','natural_champion_oov'),('C0','all'))}
    audit.truth('closure_reserved_slots_complete',set(comparisons)==expected_keys)
    if not final_integrity_failure:
        for name,entry in comparisons.items():
            audit.truth('closure_primary_NOT_RUN:'+name,entry['status']==status and entry.get('upper') is None and entry.get('point') is None)
    guard=load_json(inputs.artifact('guards',inputs.stage('08_reporting','guards.json')))
    keys={f'HC-{ref}:{group}' for group in ('BO3','BO5','major','player_source_unseen') for ref in ('D0','C0')}|{'HC-D0:seen_champion_history'}
    audit.truth('closure_nine_guards_retained',set(guard)==keys)
    if not final_integrity_failure:
        audit.truth('closure_guards_NOT_RUN',all(value['status']=='NOT_RUN' for value in guard.values()))
    return {str(s):status for s in (3,4,5,6)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--research-root',type=Path)
    parser.add_argument('--evidence-root',type=Path,help='Optional explicit root; must match frozen protocol evidence_root')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('prepare','completed'),default='completed')
    parser.add_argument('--verifier-lineage',type=Path,help='Exact archived/current verifier binding; cannot override any other frozen input')
    args=parser.parse_args(argv)
    repo=args.repo_root.resolve()
    research=(args.research_root or repo).resolve()
    sys.path.insert(0,str(repo))
    if not args.run_id or Path(args.run_id).name != args.run_id:
        parser.error('run-id must be one path component')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    audit=Audit()
    receipt=dict(schema_version=1,run_id=args.run_id,study='W1',status='FAIL',execution_created_at_utc=datetime.now(timezone.utc).isoformat(),
                 verifier_code_hashes={str(Path(__file__).resolve()):digest(__file__)},input_and_output_hashes={},implementation_hashes={},
                 environment_versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__),
                 hardware_and_threads=dict(device='CPU',threads=4),scope_exercised=[],scope_not_exercised=['independent_original_opponent_rating_numerical_reconstruction','training_or_new_parameter_fits'],
                 formal_operation_counts={},independent_calculation_counts=dict(neural_updates=0,verification_scalar_solves=0),raw_state_count=0,
                 population_counts={},membership_counts={},month_block_counts={},primary_slot_statuses={},guard_statuses={},missingness_statuses={},
                 fitted_prediction_oracle_status='NOT_RUN_ERROR',
                 checks=[],failures=[],comparisons_by_kind={},max_absolute_and_relative_differences={},protected_before_after={},
                 scientific_limitations=['Historical exposed development only; no causal OOV claim, authentic issuance, betting edge, promotion or deployment certification.',
                   'Unchanged low-level _read_source and release helpers are explicitly shared source parsing dependencies.',
                   'Opponent-strength mean/SD are pinned unchanged original feature-bank constants; old rating arithmetic is outside this oracle.'])
    inputs=None
    authorization=None
    try:
        inputs=Inputs(repo,research,args.run_id,audit)
        inputs.unavailable_comparisons=[]
        evidence=resolve_evidence(inputs,args.evidence_root)
        if args.verifier_lineage is not None:
            receipt['verifier_lineage']=inputs.set_verifier_lineage(args.verifier_lineage,evidence)
        inputs.validate_descriptors()
        receipt['evidence_root']=str(evidence)
        authorization=load_json(evidence/'authorization_and_initial_pins.json')
        contract=evidence/'oracle_contract.json'
        receipt['oracle_contract_sha256']=digest(contract)
        audit.truth('oracle_contract_binding',inputs.index['oracle_contract_sha256']==digest(contract))
        for relative,expected_sha in authorization['protected_before'].items():
            path=inputs.path(relative)
            actual=digest(path)
            audit.truth('protected_before:'+relative,actual==expected_sha)
            receipt['protected_before_after'][relative]=dict(expected=expected_sha,before=actual)
        if inputs.index['protocol_sha256'] is None:
            audit.truth('pending_protocol_only_prepare',args.mode=='prepare' and inputs.index['branch_status']=='SUPPORT_PASS_PENDING_FREEZE')
            stage0_pin=load_json(inputs.artifact('stage0',inputs.stage('08_reporting','stage0.json')))
            pins=stage0_pin['initial_pins']
        else:
            protocol=inputs.artifact('protocol',inputs.stage('05_model_input','protocol.json'))
            binding=inputs.artifact('binding',inputs.stage('05_model_input','binding.json'))
            audit.truth('protocol_binding',digest(protocol)==inputs.index['protocol_sha256'] and digest(binding)==inputs.index['binding_sha256'])
            pins=load_json(binding)['pins']
            frozen=load_json(protocol)
            if frozen['status']=='FROZEN_INVALID_NO_FITS':
                compare_mapping(audit,'invalid_protocol_zero_authorization',frozen,dict(study='W1',run_id=args.run_id,fit_authorization=False,allowed_neural_fits=0,allowed_CAL_terminal_scalars=0,reserved_primary_slots=[3,4,5,6]))
            else:
                compare_mapping(audit,'frozen_protocol_scientific_contract',frozen,dict(study='W1',run_id=args.run_id,arms=list(ARMS),seeds=list(SEEDS),primary_slots=[3,4,5,6],family_slots=10,confidence=.995,draws=10000,seed=20261009,month_universe=list(MONTHS),blocked_slot_reallocation=False,guard_count=9,guard_one_sided_confidence=1-.05/9,guard_upper_max=.005,guard_support=dict(series=100,months=4),budget=dict(neural=9,seed_epochs=108,terminal_CAL=3,prop_scales=0,C0=0,D0=0)))
            audit.truth('binding_never_amends_original',load_json(binding)['original_binding_amended'] is False)
        receipt['implementation_hashes']={p:s for p,s in pins.items() if p.endswith('.py')}
        receipt['input_and_output_hashes'].update(pins)
        for relative,sha in pins.items():
            audit.truth('bound_input:'+relative,digest(inputs.path(relative))==sha)
        historical_pin_checks(inputs,audit)
        branch=execution_branch(inputs,audit)
        terminal_integrity_checks(inputs,audit,authorization,branch)
        receipt['execution_branch']=branch
        audit.truth('requested_mode_matches_persisted_phase',args.mode=='completed' or branch in ('PREPARE','ZERO_FIT'))
        if audit.failures:
            raise ValueError('Bound input/integrity contract failure before input loading')
        series=pd.read_parquet(inputs.artifact('series'))
        series['match_id']=series.match_id.astype(str)
        audit.truth('series_unique_ids',series.match_id.is_unique)
        audit.compare('series_row_index',series.row_index.to_numpy(),np.arange(len(series)))
        for split in ('TRAIN','CAL','ASSESS'):
            audit.compare('index_population_ids:'+split,inputs.index['population_ids'][split],series.loc[series.split==split,'match_id'].astype(str).to_numpy())
        npz=np.load(inputs.artifact('arrays'),allow_pickle=False)
        arrays={k:npz[k] for k in npz.files}
        schema=load_json(inputs.artifact('schema'))
        train=np.flatnonzero(series.split.to_numpy()=='TRAIN')
        stats=normalizers(arrays,train)
        groups,support,supported,inventory,raw,vocab=verify_source(inputs,arrays,series,schema,audit,output)
        receipt['scope_exercised'] += ['all_raw_source_observations','all_canonical_token_lineage','strict_prior_unique_event_counts','closed_TRAIN_vocabularies','all_group_memberships','TRAIN_normalizers_and_phi_mapping']
        no_fit=prepare_checks(inputs,series,arrays,stats,inventory,supported,audit,output)
        audit.truth('persisted_support_matches_execution_branch',branch in ('ZERO_FIT','PREPARE') or not no_fit)
        receipt['formal_operation_counts']=trace_check(inputs,train,audit,branch)
        receipt['population_counts']={s:int(np.count_nonzero(series.split.to_numpy()==s)) for s in ('TRAIN','CAL','ASSESS')}
        receipt['population_counts']['source_events']=len(raw)
        receipt['population_counts']['history_tokens']=int(arrays['mask'].sum())
        receipt['membership_counts']={k:int(v.sum()) for k,v in groups.items()}
        receipt['missingness_statuses']=inventory['TRAIN_missing_identity']
        receipt['month_block_counts']={m:int(np.count_nonzero((series.date.astype(str).str[:7].to_numpy()==m)&(series.split.to_numpy()=='ASSESS'))) for m in MONTHS}
        refs=references(inputs,series,audit)
        stage0_descriptives(inputs,series,groups,refs,audit,output)
        receipt['scope_exercised'].append('old_D0_C0_pinned_law_scores')
        if branch=='ZERO_FIT':
            receipt['primary_slot_statuses']=closure_checks(inputs,audit,supported,receipt['formal_operation_counts'])
            receipt['guard_statuses']={f'HC-{ref}:{group}':'NOT_RUN_BLOCKED' for group in ('BO3','BO5','major','player_source_unseen') for ref in ('D0','C0')}|{'HC-D0:seen_champion_history':'NOT_RUN_BLOCKED'}
            receipt['fitted_prediction_oracle_status']='NOT_RUN_NO_FIT'
            receipt['scope_not_exercised'] += ['saved_candidate_model_replay','CAL_temperature','public_saved_candidate_inference','candidate_metrics_and_primary_guards']
            receipt['status']='PASS_CLOSURE_CONTRACT' if not audit.failures else 'FAIL'
        elif branch=='PREPARE':
            audit.truth('prepare_zero_operations',sum(receipt['formal_operation_counts'].get(k,0) for k in ('neural','CAL_temperature'))==0)
            receipt['fitted_prediction_oracle_status']='NOT_RUN_PREPARE'
            receipt['primary_slot_statuses']={str(slot):'NOT_RUN_PREPARE' for slot in (3,4,5,6)}
            receipt['guard_statuses']={f'HC-{ref}:{group}':'NOT_RUN_PREPARE' for group in ('BO3','BO5','major','player_source_unseen') for ref in ('D0','C0')}|{'HC-D0:seen_champion_history':'NOT_RUN_PREPARE'}
            receipt['scope_not_exercised'] += ['saved_candidate_model_replay','CAL_temperature','public_saved_candidate_inference','candidate_metrics_and_primary_guards']
            receipt['status']='PASS_PREPARE_CONTRACT' if not audit.failures else 'FAIL'
        elif branch=='PARTIAL_FAILED':
            # The failed attempt is consumed, not retried. Earlier complete
            # trajectories/scalars remain independently auditable.
            _,n_states,public=model_checks(inputs,arrays,series,support,vocab,stats,audit,output,partial=True)
            receipt['raw_state_count']=n_states+failed_export_state_checks(inputs,arrays,series,support,vocab,stats,audit)
            receipt['public_cases']=public
            receipt['primary_slot_statuses']=closure_checks(inputs,audit,supported,receipt['formal_operation_counts'])
            guard_status='NOT_AUTHORIZING_INTEGRITY_FAILED' if receipt['primary_slot_statuses']['3']=='NOT_AUTHORIZING_INTEGRITY_FAILED' else 'NOT_RUN_FIT_FAILED'
            receipt['guard_statuses']={f'HC-{ref}:{group}':guard_status for group in ('BO3','BO5','major','player_source_unseen') for ref in ('D0','C0')}|{'HC-D0:seen_champion_history':guard_status}
            receipt['unavailable_comparisons']={name:'NOT_RUN_FIT_FAILED' for name in inputs.unavailable_comparisons}
            receipt['fitted_prediction_oracle_status']='PARTIAL_RETAINED_STATES_VERIFIED' if not audit.failures else 'FAIL'
            receipt['scope_exercised'] += ['retained_completed_states_and_receipts','available_completed_CAL_scalars','available_retained_law_exports']
            receipt['scope_not_exercised'] += ['failed_attempt_numerical_replay','unavailable_law_exports','public_saved_candidate_inference','candidate_primary_guards']
            receipt['status']='PASS_PARTIAL_FAILURE_CLOSURE' if not audit.failures else 'FAIL'
        else:
            if audit.failures:
                raise ValueError('Raw/prefit contract failure; model checks must not hide invalid source state')
            ensembles,n_states,public=model_checks(inputs,arrays,series,support,vocab,stats,audit,output)
            receipt['raw_state_count']=n_states
            receipt['public_cases']=public
            receipt['scope_exercised'] += ['all_nine_saved_states_full_TRAIN_CAL_ASSESS','functional_torch_logits_and_laws','terminal_temperature_member_then_mixture_law','public_singleton_swap_cold_permutation_nonmutation','scores_two_layers','optimizer_and_shuffle_receipts']
            primary,guards=decision_checks(inputs,series,groups,ensembles,refs,audit,output,arrays)
            receipt['primary_slot_statuses']=primary
            receipt['guard_statuses']=guards
            receipt['scope_exercised'] += ['four_primary_995_intervals','nine_one_sided_guards','10000_full_universe_month_draws','source_name_year_proxy_95']
            receipt['fitted_prediction_oracle_status']='PASS' if not audit.failures else 'FAIL'
            receipt['status']='PASS' if not audit.failures else 'FAIL'
    except Exception as exc:
        audit.truth('execution_error',False,dict(type=type(exc).__name__,message=str(exc)))
        receipt['status']='FAIL'
    finally:
        if inputs is not None and authorization is not None:
            for relative,expected in authorization['protected_before'].items():
                try:
                    actual=digest(inputs.path(relative))
                    receipt['protected_before_after'].setdefault(relative,dict(expected=expected))['after']=actual
                    audit.truth('protected_after:'+relative,actual==expected)
                except OSError as exc:
                    audit.truth('protected_after:'+relative,False,str(exc))
            receipt['input_and_output_hashes'].update(inputs.hashes)
        receipt['checks']=audit.checks
        receipt['failures']=audit.failures
        if audit.failures:
            receipt['status']='FAIL'
        receipt['comparisons_by_kind']=dict(Counter(c['kind'] for c in audit.checks))
        for kind in TOLS:
            selected=[c for c in audit.checks if c['kind']==kind]
            receipt['max_absolute_and_relative_differences'][kind]=dict(absolute=max((c.get('maximum_absolute_difference') or 0 for c in selected),default=0),relative=max((c.get('maximum_relative_difference') or 0 for c in selected),default=0))
        for path in output.iterdir():
            if path.is_file():
                receipt['input_and_output_hashes'][str(path)]=digest(path)
        write_json(output/'verification.json',receipt)
    print(json.dumps(dict(status=receipt['status'],failures=len(audit.failures),receipt=str(output/'verification.json'))))
    return 0 if receipt['status'].startswith('PASS') else 1


if __name__=='__main__':
    raise SystemExit(main())
