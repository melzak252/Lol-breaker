"""Research-only annual native series inference, isolated from serving namespaces.

The parent process validates requests without importing archive code. One backend
at a time owns a persistent private subprocess; no historical relaxation touches serving C0.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from collections.abc import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANNUAL_RUN = 'model_reliability_20261001_055321'
_ANNUAL_COMPLETION_PINS = {
    2024: '446ad1bc23c12e39e1f43c6d80a411c6d5231c7f76b60ec9bbec2399cedc9e38',
    2025: 'ec8e68ed453e4fc708ef2a8674c43b8b9f366f191bd773eafc2a043bb0fa561c',
}
MODES = ('full', 'no_w20', 'no_organization')
_BANK_METADATA_SHA256 = '9956fb96aacf6ed58eb8a7dc249febb29a1bff853b60fdb7eb1eb149b536c129'


class NativeArtifactUnavailable(ValueError):
    """A requested annual native origin has no eligible checkpoint."""


def _clock(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a timezone-aware ISO timestamp')
    parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f'{label} must include an explicit timezone')
    return parsed.astimezone(timezone.utc)


def _id(value, label):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{label} requires an exact nonempty ID')
    if isinstance(value, int):
        result = str(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        result = str(int(value))
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise ValueError(f'{label} requires a scalar exact ID')
    if not result or len(result) > 200:
        raise ValueError(f'{label} requires an exact nonempty ID of at most 200 characters')
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def native_model_eligibility(event, model_kind, artifact_contract):
    """Pure metadata eligibility, not checkpoint or publication verification."""
    def unavailable(reason):
        return {'available': False, 'reason': reason, 'authenticated': False, 'publication_at': None}
    if model_kind not in ('c0', 'a0'):
        return unavailable(f'unsupported_native_model_kind:{model_kind}')
    if artifact_contract is None:
        return unavailable(f'missing_{model_kind}_annual_artifact:{event.get("year")}')
    try:
        cutoff = _clock(event['cutoff'], 'cutoff')
        year = cutoff.year
        if type(event['year']) is not int or event['year'] != year:
            return unavailable('event_year_differs_from_utc_cutoff')
        if artifact_contract.get('model_kind') != model_kind:
            return unavailable('artifact_model_kind_mismatch')
        if type(artifact_contract.get('model_year')) is not int or artifact_contract['model_year'] != year:
            return unavailable('artifact_annual_origin_mismatch')
        paths, hashes = artifact_contract['artifact_paths'], artifact_contract['artifact_sha256']
        if not isinstance(paths, Mapping) or not paths or not isinstance(hashes, Mapping) or set(paths) != set(hashes):
            return unavailable('missing_artifact_hash_inventory')
        if any(not isinstance(paths[k], str) or not paths[k] or not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for k, v in hashes.items()):
            return unavailable('invalid_artifact_hash_inventory')
        train = _clock(artifact_contract['train_end'], 'train_end')
        cal = _clock(artifact_contract['calibration_end'], 'calibration_end')
        train_boundary = datetime(year - 1, 7, 1, tzinfo=timezone.utc)
        cal_boundary = datetime(year, 1, 1, tzinfo=timezone.utc)
        if model_kind == 'a0':
            if 'train_end_exclusive' in artifact_contract:
                train_boundary = _clock(artifact_contract['train_end_exclusive'], 'train_end_exclusive')
            if 'cal_end_exclusive' in artifact_contract:
                cal_boundary = _clock(artifact_contract['cal_end_exclusive'], 'cal_end_exclusive')
        if train >= min(train_boundary, cutoff):
            return unavailable('train_labels_not_strictly_before_origin_and_event')
        if cal >= min(cal_boundary, cutoff):
            return unavailable('calibration_labels_not_strictly_before_origin_and_event')
        if train > cal or train_boundary > cal_boundary or cal_boundary > cutoff:
            return unavailable('invalid_annual_partition_chronology')
    except (KeyError, TypeError, ValueError) as exc:
        return unavailable(f'incomplete_native_artifact_contract:{exc}')
    return {'available': True, 'reason': None, 'authenticated': False, 'publication_at': None}


class ResearchNativePairProvider:
    """Isolated annual C0/A0 inference with one resident backend and no fallback."""
    def __init__(self, research_root=None, control_roots=None):
        self.research_root = str(Path(research_root).expanduser().resolve()) if research_root is not None else None
        self.control_roots = {str(k): str(Path(v).expanduser().resolve()) for k, v in (control_roots or {}).items()}
        self._workers = {}

    @staticmethod
    def validate_request(request):
        required = {'team1_id', 'team2_id', 'roster_a', 'roster_b', 'best_of', 'decision_at', 'competition_context', 'start_at', 'mode'}
        if not isinstance(request, Mapping) or set(request) != required:
            raise ValueError(f'research request must contain exactly {sorted(required)}')
        result = dict(request)
        for field in ('team1_id', 'team2_id'):
            result[field] = _id(request[field], field)
        if result['team1_id'] == result['team2_id']:
            raise ValueError('opponents require distinct exact IDs')
        for field in ('roster_a', 'roster_b'):
            roster = request[field]
            if not isinstance(roster, (tuple, list)) or len(roster) != 5:
                raise ValueError(f'{field} requires five explicit players')
            result[field] = sorted(_id(v, field) for v in roster)
            if len(set(result[field])) != 5:
                raise ValueError(f'{field} requires five unique players')
        if set(result['roster_a']) & set(result['roster_b']):
            raise ValueError('opposing rosters must be disjoint')
        if type(request['best_of']) is not int or request['best_of'] not in (1, 3, 5):
            raise ValueError('best_of must be exactly 1, 3, or 5')
        if request['mode'] not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        result['decision_at'] = _clock(request['decision_at'], 'decision_at')
        result['start_at'] = None if request['start_at'] is None else _clock(request['start_at'], 'start_at')
        if result['start_at'] is not None and result['start_at'] <= result['decision_at']:
            raise ValueError('start_at must be strictly after decision_at')
        context = request['competition_context']
        if context is not None and (not isinstance(context, str) or not context.strip() or len(context) > 300):
            raise ValueError('competition_context requires null or an exact nonempty string')
        result['competition_context'] = context.strip() if context else None
        return result

    def _call(self, kind, operation, **payload):
        if kind not in ('c0', 'a0'):
            raise ValueError('native model_kind must be c0 or a0')
        if kind in self._workers and self._workers[kind][0].poll() is not None:
            self.close()
        if kind not in self._workers:
            # Release the previous backend's raw history before starting another.
            self.close()
            log = tempfile.TemporaryFile(mode='w+t')
            env = {**os.environ, 'BETTING_APP_C0_WORKER': '1'}
            try:
                process = subprocess.Popen([sys.executable, '-m', 'src.models.tournament_native', '--worker', kind], cwd=PROJECT_ROOT,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
            except OSError:
                log.close()
                raise
            self._workers[kind] = process, log
        process, log = self._workers[kind]
        request = {'operation': operation, 'research_root': self.research_root, 'control_roots': self.control_roots, **payload}
        try:
            process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
            process.stdin.flush()
            line = process.stdout.readline()
        except OSError:
            self.close()
            raise
        if not line:
            log.seek(0)
            message = f'{kind} private native worker exited: {log.read()}'
            self.close()
            raise RuntimeError(message)
        response = json.loads(line)
        if 'error' in response:
            error = response['error']
            cls = {'NativeArtifactUnavailable': NativeArtifactUnavailable, 'FileNotFoundError': FileNotFoundError,
                   'ValueError': ValueError, 'ImportError': ImportError}.get(error['type'], RuntimeError)
            raise cls(error['message'])
        return response['result']

    def artifact_contract(self, year, model_kind='c0'):
        return self._call(model_kind, 'identity', year=year)

    def predict_pair(self, request, model_kind='c0', *, include_snapshot=False):
        normalized = self.validate_request(request)
        encoded = {k: v.isoformat() if isinstance(v, datetime) else v for k, v in normalized.items()}
        return self._call(model_kind, 'predict', request=encoded, include_snapshot=bool(include_snapshot))

    def predict_table(self, event, model_kind='c0', mode='full'):
        teams = [_id(t, 'team') for t in event['teams']]
        if len(teams) < 2 or len(set(teams)) != len(teams):
            raise ValueError('event requires unique entrants')
        from src.models.tournament_formats import required_best_ofs
        formats = required_best_ofs(event['spec'])
        if not formats or any(type(bo) is not int or bo not in (1, 3, 5) for bo in formats):
            raise ValueError('event spec must declare supported required best_of formats')
        contract = self.artifact_contract(event['year'], model_kind)
        eligibility = native_model_eligibility(event, model_kind, contract)
        if not eligibility['available']:
            raise NativeArtifactUnavailable(eligibility['reason'])
        rows = []
        for a, b in itertools.combinations(teams, 2):
            for bo in sorted(formats):
                result = self.predict_pair({'team1_id': a, 'team2_id': b, 'roster_a': event['rosters'][a], 'roster_b': event['rosters'][b],
                                            'best_of': bo, 'decision_at': event['cutoff'], 'start_at': event.get('start_at'),
                                            'competition_context': event.get('event_title'), 'mode': mode}, model_kind)
                rows.append({'A': a, 'B': b, 'BO': bo, 'p': result['prob_a'], 'provenance': result['provenance']})
        return {'rows': rows, 'probability_unit': 'series_win', 'model_kind': model_kind, 'artifact_contract': contract,
                'provenance': {'scope': event.get('scope'), 'roster_basis': event.get('roster_basis'), 'roster_evidence': event.get('roster_evidence'),
                               'cutoff': event['cutoff'], 'roster_sha256': _digest(event['rosters']), 'authenticated': False,
                               'pair_table_sha256': _digest(rows)}}

    def close(self):
        for process, log in self._workers.values():
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
            process.stdout.close()
            log.close()
        self._workers.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def prepare_2023_control(run_id, research_root=None, smoke=False):
    """Train the single authorized origin in a fresh archive-isolated process."""
    command = [sys.executable, '-m', 'src.models.tournament_native_training', '--run-id', run_id]
    if research_root is not None:
        command += ['--research-root', str(research_root)]
    if smoke:
        command.append('--smoke')
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=True, stdout=subprocess.PIPE, text=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


class _Worker:
    def __init__(self, kind):
        self.kind = kind
        self.adapters = {}
        self.history = None

    def _history_adapter(self, root):
        if self.history is None:
            from betting_app.core.models.native_c0 import NativeC0Adapter
            class ResearchHistory(NativeC0Adapter):
                _validate_request = staticmethod(ResearchNativePairProvider.validate_request)
            history = ResearchHistory()
            history.research_root = root
            # Only the pinned raw-history builder is reused; annual model identity
            # is set separately and no serving artifact root is relaxed.
            history._contract = {'input_sha256': {'a0-phase-walkforward-20260914/data/04_feature/release_corrected_bank/release_days.json': 'e54914ba79c5620a819a48a03bf877c373c6e53bd019219e43ffdd7db47c2620'}}
            history._identity = {'raw_feature_replay_source': {'implementation_path': str(PROJECT_ROOT / 'src/analysis/ev_asof_replay.py'),
                                  'implementation_sha256': 'f01fdc9eef9ff551cc1bc592628f44b0b17dacba84750dfb0328f21b9994e58d'}}
            self.history = history
        return self.history

    def identity(self, year, root, controls):
        if type(year) is not int:
            raise ValueError('model year must be an integer')
        if year in self.adapters:
            return self.adapters[year]['contract']
        from src.analysis import ev_asof_replay as replay
        from betting_app.core.models.native_c0 import _verify_hash, _read_json
        metadata_path = root / replay.BANK_RELATIVE / 'metadata.parquet'
        _verify_hash(metadata_path, _BANK_METADATA_SHA256, 'native row-aligned corrected metadata')
        if self.kind == 'a0':
            if year not in (2024, 2025, 2026):
                raise NativeArtifactUnavailable(f'missing_a0_annual_artifact:{year}')
            from betting_app.core.models.causal_a0 import CausalA0Adapter
            adapter = CausalA0Adapter(year, root)
            identity = adapter.identity
            archive = root / replay.CAUSAL_ARCHIVE_RELATIVE
            receipt = _read_json(archive / f'data/06_models/causal_a0/{year}/calibration.json', 'annual A0 calibration')
            metadata = __import__('pandas').read_parquet(metadata_path)
            contract = self._a0_contract(year, archive, identity, receipt, metadata)
            self.adapters[year] = {'adapter': adapter, 'identity': identity, 'contract': contract}
            return contract
        if year == 2026:
            original_root = PROJECT_ROOT / 'data/06_models/a0_data_focus/a0_data_focus_20260929_192443'
            if str(year) in controls and Path(controls[str(year)]).resolve() != original_root.resolve():
                raise ValueError('2026 native control must reuse the original C0 root')
            from betting_app.core.models.native_c0 import NativeC0Adapter
            adapter = NativeC0Adapter()
            adapter.research_root = root
            identity = adapter.load()
            metadata = __import__('pandas').read_parquet(metadata_path)
            dates = __import__('pandas').to_datetime(metadata.date)
            releases = __import__('pandas').to_datetime(metadata.result_day)
            train = metadata.loc[(dates < '2025-07-01') & (releases < '2025-07-01')]
            cal = metadata.loc[(dates >= '2025-07-01') & (dates < '2026-01-01') & (releases < '2026-01-01')]
            from src.models.a0_native_refit import _contract_digest
            from src.analysis.ev_approach_data import _research_artifact_path
            for key, digest in adapter._contract['input_sha256'].items():
                _verify_hash(_research_artifact_path(key, root), digest, f'original C0 input {key}')
            validation = metadata.loc[(dates >= '2026-01-01') & (dates < '2027-01-01')]
            memberships = {name: _contract_digest(rows.golgg_match_id.astype(str).tolist())
                           for name, rows in (('train', train), ('cal', cal), ('validation', validation))}
            if memberships != adapter._contract['split_ids_sha256']:
                raise ValueError('original C0 partition membership changed')
            identity['split_ids_sha256'] = memberships
            identity['metadata_sha256'] = _BANK_METADATA_SHA256
            contract = self._contract(year, identity['artifact_hashes'], train, cal, identity)
            self.adapters[year] = {'adapter': adapter, 'identity': identity, 'contract': contract}
            return contract
        model_root = Path(controls[str(year)]) if str(year) in controls else PROJECT_ROOT / f'data/06_models/model_reliability/{_ANNUAL_RUN}/neural/{year}'
        if not model_root.is_dir():
            raise NativeArtifactUnavailable(f'missing_c0_annual_artifact:{year}:{model_root}')
        relative = model_root.resolve().relative_to(PROJECT_ROOT)
        if year == 2023 and not relative.is_relative_to(Path('data/06_models/tournament_native')):
            raise ValueError('2023 native control must use its exclusive tournament_native namespace')
        report = PROJECT_ROOT / str(relative).replace('data/06_models/', 'data/08_reporting/', 1)
        completion_path = report / 'completed.json'
        if year in _ANNUAL_COMPLETION_PINS:
            expected_root = PROJECT_ROOT / f'data/06_models/model_reliability/{_ANNUAL_RUN}/neural/{year}'
            if model_root.resolve() != expected_root.resolve():
                raise ValueError('2024/25 native controls must reuse the original annual roots')
            _verify_hash(completion_path, _ANNUAL_COMPLETION_PINS[year], 'annual native completion')
        completion = _read_json(completion_path, 'native origin completion')
        execution = _read_json(model_root / 'execution.json', 'native origin execution')
        contract = execution['contract']
        if completion['contract'] != contract or contract['year'] != year or completion['status'] != 'TRAINING_COMPLETE_NOT_SCORED':
            raise ValueError('incomplete or mismatched native control origin')
        if year == 2023:
            from src.models.tournament_native_training import _verify_control_snapshot
            _verify_control_snapshot(model_root, contract)
        if year in _ANNUAL_COMPLETION_PINS:
            protocol_path = PROJECT_ROOT / f'data/05_model_input/model_reliability/{_ANNUAL_RUN}/neural_protocol.json'
            _verify_hash(protocol_path, contract['protocol_sha256'], 'annual control pre-fit protocol')
            protocol = _read_json(protocol_path, 'annual control protocol')
            _verify_hash(protocol_path.parent / 'neural_sources.zip', protocol['source_snapshot_sha256'], 'annual control source snapshot')
            if protocol['partitions'][str(year)]['membership_sha256'] != contract['split_ids_sha256']:
                raise ValueError('annual control protocol membership mismatch')
        for key, digest in contract['source_sha256'].items():
            path = root / key[9:] if key.startswith('research:') else PROJECT_ROOT / key.removeprefix('project:')
            _verify_hash(path, digest, f'control source {key}')
        for key, digest in contract['input_sha256'].items():
            _verify_hash(root / key, digest, f'control input {key}')
        selected = {}
        for key, digest in completion['artifacts'].items():
            path = PROJECT_ROOT / key
            if path.is_relative_to(model_root / 'control') or path.is_relative_to(model_root / 'shared') or path in (model_root / 'execution.json', model_root / 'protocol.json', model_root / 'sources.zip', report / 'partitions.parquet'):
                _verify_hash(path, digest, f'control asset {key}')
                selected[key] = digest
        import pandas as pd
        from src.models.a0_native_refit import _contract_digest
        members = pd.read_parquet(report / 'partitions.parquet')
        for part, digest in contract['split_ids_sha256'].items():
            if _contract_digest(members.loc[members.partition == part].golgg_match_id.tolist()) != digest:
                raise ValueError(f'control partition membership changed:{part}')
        train = members.loc[members.partition == 'train']
        cal = members.loc[members.partition == 'cal']
        for rows, boundary in ((train, f'{year-1}-07-01'), (cal, f'{year}-01-01')):
            if not (pd.to_datetime(rows.date).lt(boundary) & pd.to_datetime(rows.result_day).lt(boundary)).all():
                raise ValueError('native control membership exceeds its closed origin')
        if not pd.to_datetime(cal.date).ge(f'{year-1}-07-01').all():
            raise ValueError('native control CAL membership predates CAL window')
        if year == 2023:
            from src.models.tournament_native_training import load_control
            assembly = load_control(model_root, root)
        else:
            from src.models.a0_reliability import load_origin
            assembly = load_origin(model_root, 'control', root)
        result = self._contract(year, selected, train, cal, {'source_sha256': contract['source_sha256'], 'input_sha256': contract['input_sha256'],
                                                            'split_ids_sha256': contract['split_ids_sha256'], 'completion_sha256': replay._sha256(completion_path)})
        self.adapters[year] = {'assembly': assembly, 'identity': result['provenance'], 'contract': result}
        return result

    @staticmethod
    def _contract(year, assets, train, cal, provenance, kind='c0', **boundaries):
        if len(train) == 0 or len(cal) == 0:
            raise ValueError('native origin has empty closed TRAIN/CAL memberships')
        return {'model_kind': kind, 'model_year': year, 'train_end': str(train.result_day.max())[:10] + 'T00:00:00Z',
                'calibration_end': str(cal.result_day.max())[:10] + 'T00:00:00Z',
                'artifact_paths': {key: str(PROJECT_ROOT / key) if not Path(key).is_absolute() else key for key in assets},
                'artifact_sha256': assets, 'provenance': provenance, **boundaries}

    def _a0_contract(self, year, archive, identity, receipt, metadata):
        # Receipt IDs bind the actual original annual expert and two calibration
        # populations; never substitute native C0's July TRAIN membership.
        import pandas as pd
        ids = metadata.golgg_match_id.astype(str)
        if ids.duplicated().any():
            raise ValueError('duplicate native A0 input membership')
        train_ids = list(map(str, receipt['train_ids']))
        cal_ids = list(dict.fromkeys(map(str, receipt['annual_cal_ids'] + receipt['mixture_cal_ids'])))
        if not set(train_ids + cal_ids).issubset(set(ids)):
            raise ValueError('native A0 receipt contains unknown input IDs')
        train = metadata.loc[ids.isin(train_ids)]
        cal = metadata.loc[ids.isin(cal_ids)]
        dates, releases = pd.to_datetime(train.date), pd.to_datetime(train.result_day)
        if not ((dates < f'{year-1}-01-01') & (releases < f'{year-1}-01-01')).all():
            raise ValueError('native A0 TRAIN labels exceed original annual boundary')
        if not (pd.to_datetime(cal.result_day) < f'{year}-01-01').all():
            raise ValueError('native A0 CAL labels exceed original annual boundary')
        assets = {str(archive / key): value for key, value in identity['artifact_hashes'].items()}
        assets[str(archive / f'data/06_models/causal_a0/{year}/calibration.json')] = identity['calibration_sha256']
        provenance = {**identity, 'train_membership_sha256': _digest(train_ids), 'cal_membership_sha256': _digest(cal_ids)}
        return self._contract(year, assets, train, cal, provenance, kind='a0',
                              train_end_exclusive=f'{year-1}-01-01T00:00:00Z', cal_end_exclusive=f'{year}-01-01T00:00:00Z')

    def predict(self, value, root, controls, include_snapshot=False):
        request = ResearchNativePairProvider.validate_request(value)
        year = request['decision_at'].year
        contract = self.identity(year, root, controls)
        event = {'year': year, 'cutoff': request['decision_at'].isoformat()}
        eligibility = native_model_eligibility(event, self.kind, contract)
        if not eligibility['available']:
            raise NativeArtifactUnavailable(eligibility['reason'])
        history = self._history_adapter(root)
        target, raw, hist, provenance = history._raw_snapshot(request)
        from betting_app.core.models.native_c0 import encode_snapshot
        encoded_snapshot = encode_snapshot({'target': target, 'raw': raw, 'hist': hist})
        history._decode_snapshot(encoded_snapshot, request)
        before = history._array_digest({**raw, **hist})
        entry = self.adapters[year]
        if self.kind == 'a0':
            from src.analysis.ev_asof_replay import AsOfSnapshot
            prediction = entry['adapter'].predict_snapshot(AsOfSnapshot(target, raw, hist, True, None), mode=request['mode'])
            probability = float(prediction.prob_a)
            all_modes = prediction.diagnostics['mode_probabilities']
        else:
            import numpy as np
            from src.models.a0_data_focus import predict_models
            if 'adapter' in entry:
                adapter = entry['adapter']
                assembly = adapter._backend, adapter._shared, adapter._models, adapter._calibration
            else:
                assembly = entry['assembly']
            probabilities = predict_models(*assembly, {k: v[None] for k, v in raw.items()}, {k: v[None] for k, v in hist.items()}, np.asarray([request['best_of']]))['full']
            if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
                raise ValueError('native C0 returned malformed series probabilities')
            all_modes = dict(zip(MODES, map(float, probabilities[0])))
            probability = all_modes[request['mode']]
        if history._array_digest({**raw, **hist}) != before:
            raise ValueError('native inference mutated its tensors')
        proof = {'native_history_provenance': provenance, 'input_hashes': before, 'model_identity': entry['identity'],
                 'artifact_contract': contract, 'target': target, 'roster_sha256': _digest([request['roster_a'], request['roster_b']]),
                 'clock_sha256': _digest({'decision_at': value['decision_at'], 'start_at': value['start_at'], 'feature_source_max_day': target['feature_source_max_day']}),
                 'probability_unit': 'series_win', 'mode_probabilities': all_modes, 'authenticated': False}
        result = {'prob_a': probability, 'prob_b': 1 - probability, 'best_of': request['best_of'], 'provenance': proof}
        if include_snapshot:
            result['encoded_snapshot'] = encoded_snapshot
        return result


def _worker_main(kind):
    for name in ('C0_RAW_HISTORY_PATH', 'C0_RELEASE_MAP_PATH'):
        os.environ.pop(name, None)
    from contextlib import redirect_stdout
    worker = _Worker(kind)
    output = sys.stdout
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            with redirect_stdout(sys.stderr):
                from src.analysis.ev_asof_replay import _resolve_research_root
                root = _resolve_research_root(payload['research_root'])
                # Keep explicit-root precedence inside this private worker only.
                os.environ['C0_RESEARCH_ROOT'] = str(root)
                if payload['operation'] == 'identity':
                    result = worker.identity(payload['year'], root, payload['control_roots'])
                elif payload['operation'] == 'predict':
                    result = worker.predict(payload['request'], root, payload['control_roots'], payload.get('include_snapshot', False))
                else:
                    raise ValueError('unknown native worker operation')
            response = {'result': result}
        except Exception as exc:
            response = {'error': {'type': type(exc).__name__, 'message': str(exc)}}
        output.write(json.dumps(response, allow_nan=False) + '\n')
        output.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('c0', 'a0'), required=True)
    _worker_main(parser.parse_args().worker)
