"""R1 historical final-appearance / first-map-label study.

Only new isolated roots are written. Commands are explicit freeze, prepare,
run and saved predict. Native inference is ASSESS-only and partial-mass bounds
are algebraic under provisional side independence, NOT confidence intervals.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import ast
import re
import shutil
from typing import Mapping

import numpy as np
import pandas as pd

from src.analysis.ev_asof_replay import (
    _normalize_id, _as_day, _timestamp, _source_series_order,
    _target_competition_context, _event_roster, _raw_release_day,
    build_roster_events,
)
from src.analysis.sports_reliability_common import paired_block_contrast
from src.models.lineup_law_reliability import CLASS_ORDER, LineupLaw, fit_lineup_law
from src.utils import golgg_schema

STAGES = ('04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting')
MONTHS = tuple(f'2026-{m:02d}' for m in range(1, 10))
FLOORS = {'train_side_labels': 1000, 'train_previous_labels': 100, 'train_other_labels': 100,
          'assessment_series': 200, 'assessment_months': 6, 'cal_side_labels': 3,
          'cal_latest_labels': 1, 'cal_previous_labels': 1, 'cal_other_labels': 1}
LIMITATIONS = ['HISTORICAL_DEVELOPMENT_ONLY', 'final-map historical appearances predict first-map labels',
               'daily releases are not authenticated announcement clocks',
               'side independence is provisional and unvalidated',
               'bounds are conditional algebraic missing-roster-mass bounds, not confidence intervals',
               'no winner, betting, deployment or replacement claim',
               'G0, old10/10 and 2026-10-31/2026-11-01 forward dates unchanged']


class SourceValidationError(ValueError):
    """An identified defect in pinned source identities, schema or release law."""


class NativeIntegrityError(ValueError):
    """Captured native evidence contradicts its frozen request or provenance."""


def _source_validation(validate, *args, **kwargs):
    """Type only errors from designated input-validation boundaries."""
    try:
        return validate(*args, **kwargs)
    except SourceValidationError:
        raise
    except ValueError as error:
        raise SourceValidationError(str(error)) from error


def _id(value, label):
    result = _normalize_id(value, label)
    if result is None:
        raise ValueError('Missing ' + label)
    return result


def _roster(value, *, complete=False):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or (complete and len(value) != 5):
        raise ValueError('Roster requires five identity slots')
    result = _event_roster({'roster': value})
    if complete and any(p is None for p in result):
        raise ValueError('Roster requires five known unique player identities')
    return sorted((p for p in result if p is not None)) + [None] * result.count(None)


def _source_maps(raw, release_days):
    mid = _source_validation(_id, raw.get('match_id'), 'source match ID')
    native, release_source, reason = _source_validation(_raw_release_day, raw, release_days)
    games = golgg_schema.games(raw)
    formats = []
    for key in ('BoN', 'best_of'):
        if raw.get(key) is not None:
            value = raw[key]
            numeric = _source_validation(float, value)
            if isinstance(value, bool) or not np.isfinite(numeric):
                raise SourceValidationError('Invalid declared source format')
            integer = _source_validation(int, value)
            if numeric != integer or integer <= 0:
                raise SourceValidationError('Invalid declared source format')
            formats.append(integer)
    if len(set(formats)) > 1:
        raise SourceValidationError('Contradictory source format aliases')
    for keys in (('tid_1', 't1_id'), ('tid_2', 't2_id')):
        aliases = {_source_validation(_normalize_id, raw[k], 'source header team ID') for k in keys if k in raw}
        aliases.discard(None)
        if len(aliases) > 1:
            raise SourceValidationError('Contradictory source header team aliases')
    days = [_source_validation(_as_day, g.get('date'), 'source map date') for g in games]
    if any(b < a for a, b in zip(days, days[1:])):
        raise SourceValidationError('Map dates contradict source chronology ' + mid)
    teams = {_source_validation(_id, games[0].get(side + '_id'), 'source team ID') for side in ('t1', 't2')}
    if len(teams) != 2:
        raise SourceValidationError('Source opponents must have distinct IDs')
    opponents = [_source_validation(_id, getter(raw), 'source header team ID')
                 for getter in (golgg_schema.team1_id, golgg_schema.team2_id)]
    if opponents[0] == opponents[1]:
        raise SourceValidationError('Source header opponents must have distinct IDs')
    if set(opponents) != teams:
        raise SourceValidationError('Source header/map team mismatch')
    events = _source_validation(build_roster_events, [raw], release_days)
    gids = set()
    for index, game in enumerate(games, 1):
        if {_source_validation(_id, game.get(s + '_id'), 'map team ID') for s in ('t1', 't2')} != teams:
            raise SourceValidationError('Source map/series team mismatch')
        if game.get('match_id') is not None and _source_validation(_id, game['match_id'], 'map series ID') != mid:
            raise SourceValidationError('Source map/series identity mismatch')
        gid = _source_validation(_normalize_id, game.get('game_id'), 'map ID')
        if gid is not None:
            if gid in gids:
                raise SourceValidationError('Duplicate map identity ' + gid)
            gids.add(gid)
        if game.get('source_best_of') is not None:
            value = game['source_best_of']
            numeric = _source_validation(float, value)
            if isinstance(value, bool) or not np.isfinite(numeric):
                raise SourceValidationError('Source map/series format mismatch')
            integer = _source_validation(int, value)
            if numeric != integer or integer != golgg_schema.best_of(raw):
                raise SourceValidationError('Source map/series format mismatch')
        for side in ('t1', 't2'):
            for player in game[side + '_players'].values():
                aliases = {_source_validation(_normalize_id, player[k], 'player ID') for k in ('player_id', 'id') if k in player}
                aliases.discard(None)
                if len(aliases) > 1:
                    raise SourceValidationError('Contradictory source player identity aliases')
        own = [e for e in events if e['map_index'] == index]
        if set(p for p in own[0]['roster'] if p is not None) & set(p for p in own[1]['roster'] if p is not None):
            raise SourceValidationError('Cross-side player overlap')
        for event in own:
            event['roster'] = _source_validation(_roster, event['roster'])
    context = _source_validation(_target_competition_context, {
        'tournament_name': golgg_schema.match_tournament(raw),
        'competition_context': raw.get('competition_context')}, mid)
    for e in events:
        e.update(competition_context=context, tournament_name=context,
                 native_release_day=native.isoformat(), release_source=release_source,
                 effective_release_reason=reason)
    return events, gids


def build_series_appearances(raw_matches, release_days, *, canonical_release_days=None):
    if not isinstance(release_days, Mapping) or (canonical_release_days is not None and not isinstance(canonical_release_days, Mapping)):
        raise ValueError('Release inventories must be mappings')
    canonical_release_days = canonical_release_days or {}
    output, seen, map_ids = [], set(), set()
    for raw in raw_matches:
        if not isinstance(raw, Mapping):
            raise ValueError('Source series must be an object')
        mid = _id(raw.get('match_id'), 'source match ID')
        if mid in seen:
            raise ValueError('Duplicate source series identity ' + mid)
        seen.add(mid)
        events, gids = _source_maps(raw, release_days)
        if map_ids & gids:
            raise ValueError('Duplicate map identity across source series')
        map_ids.update(gids)
        canonical = _as_day(canonical_release_days[mid], 'canonical release') if mid in canonical_release_days else None
        final = max(e['map_index'] for e in events)
        for e in events:
            if e['map_index'] != final:
                continue
            native = _as_day(e['native_release_day'], 'native release')
            effective = max(native, canonical) if canonical is not None else native
            e.update(canonical_release_day=canonical.isoformat() if canonical else None,
                     effective_release_day=effective.isoformat(),
                     effective_release_reason='canonical_release_bound' if canonical and canonical > native else e['effective_release_reason'])
            output.append(e)
    return sorted(output, key=lambda e: (e['effective_release_day'], _source_series_order(e['source_match_id']), e['team_id']))


def candidate_state(events, *, team_id, competition_context, cutoff_day, target_match_id):
    team = _id(team_id, 'team ID')
    target = _id(target_match_id, 'target match ID')
    cutoff = _as_day(cutoff_day, 'cutoff day')
    context = _target_competition_context({'competition_context': competition_context}, target)
    selected, seen, clocks = [], set(), {}
    for event in events:
        mid = _id(event.get('source_match_id'), 'source match ID')
        eteam = _id(event.get('team_id'), 'appearance team ID')
        econtext = _target_competition_context(event, mid)
        key = (mid, eteam, econtext)
        if key in seen:
            raise ValueError('duplicate series appearance identity')
        seen.add(key)
        release = _as_day(event.get('effective_release_day'), 'appearance release')
        native = _as_day(event.get('native_release_day', event['effective_release_day']), 'native release')
        canonical = _as_day(event['canonical_release_day'], 'canonical release') if event.get('canonical_release_day') else None
        if release != (max(native, canonical) if canonical is not None else native):
            raise ValueError('Appearance release contradicts conservative native/canonical clock')
        clock = (native, canonical, release)
        if mid in clocks and clocks[mid] != clock:
            raise ValueError('Contradictory release dates within series')
        clocks[mid] = clock
        played = _as_day(event.get('original_map_day'), 'appearance played day')
        if played > release:
            raise ValueError('Played date later than appearance release')
        roster = _roster(event.get('roster'))
        if mid == target and (native < cutoff or (canonical is not None and canonical < cutoff)):
            raise ValueError('target already released before origin')
        if mid != target and release < cutoff and eteam == team and context is not None and econtext == context:
            selected.append((release, _source_series_order(mid), event, roster))
    selected.sort(key=lambda x: (x[0], x[1]))
    selected = selected[-10:]
    result = {'team_id': team, 'competition_context': context, 'cutoff_day': cutoff.isoformat(),
              'target_match_id': target, 'history_ids': [x[2]['source_match_id'] for x in selected],
              'incomplete_history_count': sum(None in x[3] for x in selected), 'candidates': [], 'absence': None}
    if context is None:
        result['absence'] = 'MISSING_EXACT_CONTEXT'
    elif not selected:
        result['absence'] = 'NO_EXACT_CONTEXT_HISTORY'
    elif None in selected[-1][3]:
        result['absence'] = 'INCOMPLETE_LATEST_APPEARANCE'
    else:
        unique = []
        for release, _, event, roster in reversed(selected):
            if None in roster or roster in unique:
                continue
            unique.append(roster)
            result['candidates'].append({'roster': roster, 'count': sum(x[3] == roster for x in selected),
                'age_days': (cutoff - release).days, 'last_source_match_id': event['source_match_id'],
                'last_played_day': event['original_map_day'], 'last_release_day': release.isoformat()})
            if len(unique) == 2:
                break
    return result


def native_clock_conflicts(events, *, cutoff_day):
    cutoff = _as_day(cutoff_day, 'cutoff day')
    clocks = {}
    for e in events:
        mid = _id(e.get('source_match_id'), 'source match ID')
        native = _as_day(e.get('native_release_day', e.get('effective_release_day')), 'native release')
        canonical = _as_day(e['canonical_release_day'], 'canonical release') if e.get('canonical_release_day') else None
        pair = (native, canonical)
        if mid in clocks and clocks[mid] != pair:
            raise ValueError('Contradictory release dates within series')
        clocks[mid] = pair
    return sorted((mid for mid, (n, c) in clocks.items() if c is not None and n < cutoff <= c), key=_source_series_order)


def released_target_label(raw, release_days, *, team_id, available_before):
    events, _ = _source_maps(raw, release_days)
    if golgg_schema.best_of(raw) not in (1, 3, 5):
        return None
    if _as_day(events[0]['native_release_day'], 'native release') >= _as_day(available_before, 'label availability'):
        return None
    team = _id(team_id, 'label team ID')
    matching = [e for e in events if e['map_index'] == 1 and e['team_id'] == team]
    if len(matching) != 1:
        raise ValueError('Label team does not bind to first map')
    roster = matching[0]['roster']
    return None if None in roster else roster


def classify_lineup(state, actual_five):
    actual = _roster(actual_five, complete=True)
    if state.get('absence') or not state.get('candidates'):
        raise ValueError('Unavailable candidate state cannot supply label')
    for index, c in enumerate(state['candidates']):
        if actual == _roster(c['roster'], complete=True):
            return CLASS_ORDER[index]
    return 'OTHER'


def _q(value):
    q = np.asarray(value, dtype=np.float64)
    if q.shape != (3,) or not np.isfinite(q).all() or (q < 0).any() or abs(float(q.sum()) - 1.) > 1e-12:
        raise ValueError('Law must be a normalized finite nonnegative three-vector')
    return q


def score_side_laws(rows):
    grouped, unscoreable, seen = defaultdict(list), defaultdict(list), set()
    for row in rows:
        key = (str(row['match_id']), row['side'])
        if key in seen:
            raise ValueError('duplicate side law key')
        seen.add(key)
        grouped.setdefault(key[0], [])
        if row.get('label') is None or row.get('q') is None:
            reason = row.get('absence') or row.get('label_absence')
            if not reason:
                raise ValueError('Unscoreable side requires explicit input/label absence')
            unscoreable[key[0]].append({'side': key[1], 'reason': reason})
            continue
        q = _q(row['q'])
        label = CLASS_ORDER.index(row['label'])
        if q[label] <= 0:
            raise ValueError('Zero true-label probability is an invalid observed score')
        grouped[key[0]].append((-float(np.log(q[label])), float((q[2] - (label == 2)) ** 2)))
    return [{'match_id': mid, 'lineup_nll': float(np.mean([v[0] for v in vals])) if vals else None,
             'other_brier': float(np.mean([v[1] for v in vals])) if vals else None,
             'scored_side_count': len(vals), 'unscoreable_sides': unscoreable[mid],
             'absence': None if vals else 'NO_ELIGIBLE_SIDE_LABEL'}
            for mid, vals in sorted(grouped.items())]


def support_gate(counts):
    if not isinstance(counts, Mapping):
        raise ValueError('Support counts must be a mapping')
    for key in FLOORS:
        value = counts.get(key)
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError('Support counts must be nonnegative integers or explicitly missing')
    failed = [{'field': k, 'required': v, 'observed': counts.get(k),
               'reason': 'MISSING_COUNT' if counts.get(k) is None else 'BELOW_FLOOR'}
              for k, v in FLOORS.items() if k not in counts or counts[k] is None or counts[k] < v]
    valid = counts.get('identity_release_valid') is True
    return {'status': 'BLOCKED_IDENTITY_RELEASE' if not valid else 'INSUFFICIENT_FIT_SUPPORT' if failed else 'READY_FOR_TWO_TRAIN_FITS',
            'failed_floors': failed, 'counts': counts, 'formal_fits': 0}


def build_native_pair_requests(target, state_a, state_b):
    context = _target_competition_context(target, str(target['match_id']))
    if context is None:
        raise ValueError('Missing exact target context')
    decision = _timestamp(target['decision_at'], 'decision_at')
    if pd.isna(decision) or decision.date() < date(2026, 1, 1):
        raise ValueError('Native requests require aware 2026 origin')
    cutoff = decision.date()
    for key in ('native_release_day', 'label_release_day'):
        if _as_day(target.get(key), key) < cutoff:
            raise ValueError('Target already released before native origin')
    start = _timestamp(target.get('start_at'), 'start_at')
    if not pd.isna(start) and start <= decision:
        raise ValueError('Target start_at must follow origin')
    if type(target['best_of']) is not int or target['best_of'] not in (1, 3, 5):
        raise ValueError('Unsupported target format')
    for state, key in ((state_a, 'team1_id'), (state_b, 'team2_id')):
        if state['team_id'] != _id(target[key], key) or state['competition_context'] != context or state['cutoff_day'] != cutoff.isoformat() or state['target_match_id'] != str(target['match_id']):
            raise ValueError('Candidate target identity/context/origin mismatch')
        if state.get('absence') or not state['candidates']:
            return {}
    if target['team1_id'] == target['team2_id']:
        raise ValueError('Opponents must have distinct IDs')
    requests = {}
    for i, a in enumerate(state_a['candidates']):
        for j, b in enumerate(state_b['candidates']):
            ra, rb = _roster(a['roster'], complete=True), _roster(b['roster'], complete=True)
            if set(ra) & set(rb):
                raise ValueError('Opposing candidate rosters overlap/share players')
            requests[i, j] = {'team1_id': target['team1_id'], 'team2_id': target['team2_id'],
                             'roster_a': ra, 'roster_b': rb, 'best_of': target['best_of'],
                             'decision_at': decision.isoformat(), 'start_at': None if pd.isna(start) else start.isoformat(),
                             'competition_context': context, 'mode': 'full'}
    if len(requests) > 4:
        raise ValueError('At most two candidates per side')
    return requests


def supported_roster_bounds(q_a3, q_b3, predictions, *, best_of):
    a, b = _q(q_a3), _q(q_b3)
    if best_of not in (1, 3, 5):
        raise ValueError('Unsupported native format')
    legal = {(i, j) for i in range(2) for j in range(2)}
    if any(k not in legal for k in predictions):
        raise ValueError('Unknown native candidate pair')
    weights = {(i, j): float(a[i] * b[j]) for i in range(2) for j in range(2)}
    lower = 0.
    for key, weight in weights.items():
        if key not in predictions:
            if weight > 0:
                raise ValueError('Missing positive-weight native candidate pair')
            continue
        p = predictions[key]
        if (p.best_of != best_of or p.model_name != 'Causal-C0' or p.model_version != 'c0-native-2026-w32-e12-v1'
                or p.feature_version != 'native-c0-history16-v1' or p.diagnostics.get('mode') != 'full'
                or p.diagnostics.get('probability_target') != 'series_win_already_projected'):
            raise ValueError('Native pair identity/format/full already-series law mismatch')
        if not np.isfinite([p.prob_a, p.prob_b]).all() or min(p.prob_a, p.prob_b) < 0 or abs(p.prob_a + p.prob_b - 1) > 1e-12:
            raise ValueError('Invalid native pair probability')
        lower += weight * p.prob_a
    supported = float(sum(weights.values()))
    # Algebraic complement is calculated without a scalar-coverage tolerance.
    missing = float(a[2] + b[2] - a[2] * b[2])
    return {'lower': lower, 'upper': lower + missing, 'supported_mass': supported,
            'missing_mass': missing, 'pair_weights': weights,
            'prob_a': lower if missing == 0 else None,
            'scalar_absence': None if missing == 0 else 'POSITIVE_UNSUPPORTED_ROSTER_MASS',
            'provisional_independence': True, 'algebraic_not_confidence_interval': True}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _safe(value):
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Nonfinite receipt statistic must have an explicit null reason')
    return value


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def _command_start(paths, command, invocation, value):
    """Exclusive invocation ownership begins at creation, before serialization."""
    marker = paths['08_reporting'] / (command + '_started.json')
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open('x', encoding='utf-8') as handle:
        invocation['started'] = True
        json.dump(_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')

def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _roots(run_id, repo_root=None, research_root=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Unsafe run_id')
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    if research_root is None:
        pointer = repo / 'data/research_root.txt'
        research_root = os.environ.get('ENSEMBLE_RESEARCH_ROOT') or (pointer.read_text().strip() if pointer.is_file() else repo)
    research = Path(research_root).expanduser().resolve()
    paths = {s: repo / 'data' / s / 'lineup_law_reliability' / run_id for s in STAGES}
    return repo, research, paths


def _resolve(roots, relative):
    for root in roots:
        path = root / relative
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Missing required pinned lineage: ' + str(relative))


def _partition(row):
    day, release = str(row['date']), str(row['release_day'])
    if day < '2025-07-01' and release < '2025-07-01':
        return 'TRAIN'
    if '2025-07-01' <= day < '2026-01-01' and release < '2026-01-01':
        return 'CAL'
    if '2026-01-01' <= day < '2027-01-01':
        return 'ASSESS'
    return 'EXCLUDED_LATE_RELEASE'


def _canonical(path):
    import pyarrow.parquet as pq
    required = ['match_id', 'date', 'release_day', 'best_of', 'team_a', 'team_b', 'roster_a', 'roster_b']
    available = pq.read_schema(path).names
    extra = [c for c in ('tournament', 'tournament_name', 'competition_context', 'winner', 'y',
                        'major', 'is_major', 'player_source_unseen', 'source_unseen') if c in available]
    frame = pd.read_parquet(path, columns=required + extra)
    frame['match_id'] = [_source_validation(_id, v, 'canonical match ID') for v in frame.match_id]
    if frame.match_id.duplicated().any() or len(frame) != 40636:
        raise SourceValidationError('Canonical target identities/count differ from frozen 40,636')
    for field in ('team_a', 'team_b'):
        frame[field] = [_source_validation(_id, v, field) for v in frame[field]]
    for field in ('date', 'release_day'):
        frame[field] = [_source_validation(_as_day, v, field).isoformat() for v in frame[field]]
    frame['partition'] = [_partition(r) for r in frame.to_dict('records')]
    expected = {'TRAIN': 36205, 'CAL': 1328, 'ASSESS': 3103}
    if dict(Counter(frame.partition)) != expected:
        raise SourceValidationError('Canonical partition membership changed')
    return frame


def freeze_run(run_id, *, repo_root=None, research_root=None, evidence_root=None):
    """Pin semantics, populations and import/source closure before any counts."""
    repo, research, paths = _roots(run_id, repo_root, research_root)
    evidence = Path(evidence_root).expanduser().resolve() if evidence_root else repo / 'data/08_reporting/sports_reliability_improvements' / run_id
    authorization = _load(evidence / 'authorization_and_initial_pins.json')
    if authorization['run_id'] != run_id:
        raise ValueError('Authorization run_id mismatch')
    contract = _load(evidence / 'r1_contract.json')
    manifest = _load(repo / 'conf/base/research_benchmark.json')
    native_config = _load(repo / 'conf/base/c0_serving.json')
    inputs = {
        'canonical_dataset': repo / contract['source_map']['canonical_target_dataset'],
        'canonical_protocol': repo / contract['source_map']['canonical_split_protocol'],
        'raw_source': repo / contract['source_map']['raw_source'],
        'native_release_map': _resolve((research, repo), native_config['release_map_relative']),
        'canonical_release_override': _resolve((research, repo), 'data/03_primary/ground_up_lol_20260927_run001/release_days.json'),
        'native_participants': _resolve((research, repo), manifest['input']['path']),
        'saved_c0_reference': repo / 'data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet',
        'native_config': repo / 'conf/base/c0_serving.json',
        'benchmark_config': repo / 'conf/base/research_benchmark.json',
    }
    expected = {'raw_source': contract['source_map']['raw_source_sha256'],
                'native_release_map': contract['source_map']['native_release_map_sha256_declared_in_replay'],
                'native_participants': manifest['input']['sha256']}
    pins = {}
    for role, path in inputs.items():
        digest = _sha(path)
        if role in expected and digest != expected[role]:
            raise ValueError('Frozen source identity changed: ' + role)
        pins[role] = {'path': str(path.resolve()), 'sha256': digest}
    for name in ('authorization_and_initial_pins.json', 'prefit_design_resolutions.json',
                 'r1_contract.json', 'common_contract.json', 'oracle_contract.json'):
        path = evidence / name
        pins[name] = {'path': str(path), 'sha256': _sha(path)}
    code_modules = ('src.analysis.lineup_law_reliability', 'src.models.lineup_law_reliability',
                    'src.analysis.sports_reliability_common', 'src.analysis.ev_asof_replay',
                    'src.features.champion_history_data', 'src.features.prop_distribution_data',
                    'src.models.champion_history_runner', 'src.utils.golgg_schema',
                    'src.models.competition_tiers',
                    'scripts.export_prospective_features', 'betting_app.services.c0_inference',
                    'betting_app.core.models.native_c0', 'betting_app.core.models.contract',
                    'betting_app.services.c0_worker')
    imports = _import_closure(repo, code_modules)
    from importlib.metadata import version
    library_versions = {}
    for module in ('numpy', 'scipy', 'pandas', 'joblib', 'torch', 'threadpoolctl'):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError('Required numerical library origin unavailable ' + module)
        imports[module] = {'path': str(Path(spec.origin).resolve()), 'sha256': _sha(spec.origin)}
        library_versions[module] = version(module)
    if imports['src.analysis.ev_asof_replay']['sha256'] != native_config['feature_replay_sha256']:
        raise ValueError('Native feature replay closure drift; never repin the frozen native identity')
    from src.analysis.ev_asof_replay import FEATURE_SOURCE_SHA256, NATIVE_SOURCE_SHA256
    for relative, digest in FEATURE_SOURCE_SHA256.items():
        path = _resolve((research, repo), Path('unified-ratings-20260911') / relative)
        if _sha(path) != digest:
            raise ValueError('Frozen feature archive source changed: ' + relative)
        pins['feature_archive:' + relative] = {'path': str(path), 'sha256': digest}
    for relative, digest in NATIVE_SOURCE_SHA256.items():
        path = _resolve((research, repo), Path('a0-phase-walkforward-20260914') / relative)
        if _sha(path) != digest:
            raise ValueError('Frozen model backend source changed: ' + relative)
        pins['native_archive:' + relative] = {'path': str(path), 'sha256': digest}
    checked = {str(Path(pin['path']).resolve()): pin['sha256'] for pin in pins.values()}
    for relative, digest in authorization['protected_before'].items():
        path = str((repo / relative).resolve())
        if path not in checked:
            checked[path] = _sha(path)
        if checked[path] != digest:
            raise ValueError('Protected original changed: ' + relative)
    cli = repo / 'scripts/train_lineup_law.py'
    verifier = repo / 'scripts/verify_lineup_law_reliability.py'
    pins['cli'] = {'path': str(cli), 'sha256': _sha(cli)}
    pins['independent_verifier'] = {'path': str(verifier), 'sha256': _sha(verifier)}
    fixture = repo / 'betting_app/tests/test_lineup_law_reliability.py'
    pins['synthetic_public_contract_tests'] = {'path': str(fixture), 'sha256': _sha(fixture)}
    canonical = _canonical(inputs['canonical_dataset'])
    populations = {p: canonical.loc[canonical.partition == p, 'match_id'].tolist() for p in ('TRAIN', 'CAL', 'ASSESS')}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Fresh output roots required; no overwrite/refreeze/restart')
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=False)
    protocol = {'schema': 'lineup_law_reliability_protocol_v1', 'run_id': run_id,
        'scope': 'HISTORICAL_DEVELOPMENT_ONLY', 'created_at': _now(), 'repo_root': str(repo),
        'research_root': str(research), 'evidence_root': str(evidence), 'resolved_semantics': contract,
        'prefit_resolutions': _load(evidence / 'prefit_design_resolutions.json'),
        'population_ids': populations, 'hashes': pins, 'import_origins': imports,
        'protected_before': authorization['protected_before'], 'partitions': contract['chronology_and_candidates']['partitions'],
        'fit_schedule': ['conditional_six_parameters', 'global_two_parameters'],
        'fit_budget': {'TRAIN': 2, 'CAL': 0, 'C0': 0, 'ASSESS': 0, 'restarts': 0},
        'support_floors': FLOORS, 'primary_slots': [7, 8], 'primary_confidence': .995,
        'draws': 10000, 'seed': 20261009, 'month_universe': MONTHS,
        'public_case_selection': 'lexically first ASSESS target for each prespecified BO/history/candidate/missingness state, at most32; immutable stage05/public_cases.json created during prepare before fits',
        'source_id_ties': 'ASCII decimal numeric before nonnumeric lexical',
        'membership_rules': {'major': 'classify_competition(exact source tournament,target UTC day).tier.value == major; unknown remains UNKNOWN',
            'player_source_unseen': 'pinned benchmark metadata unknown_players > 0; absent metadata UNKNOWN outside ASSESS, missing ASSESS hard stop',
            'history': 'count of final-map appearances; cold=0,partial=1..9,full=10; incomplete reported separately',
            'proxy': 'exact tournament source name within played year; absent name UNKNOWN'},
        'artifact_schema': _load(evidence / 'oracle_contract.json')['artifact_schema'],
        'output_paths': {k: str(v) for k, v in paths.items()}, 'limitations': LIMITATIONS}
    protocol['library_versions'] = library_versions
    protocol['hardware_contract'] = {'device': 'CPU', 'threads': 4, 'raw_source': 'streamed once during prepare',
                                     'native_source': 'one cached parse per resident worker session'}
    _json(paths['05_model_input'] / 'protocol.json', protocol)
    _json(paths['05_model_input'] / 'binding.json', {'schema_version': 1, 'study': 'R1',
        'run_id': run_id, 'created_at_utc': _now(), 'protocol_sha256': _sha(paths['05_model_input'] / 'protocol.json'),
        'pins': pins, 'import_origins': imports, 'protected_before': authorization['protected_before'],
        'oracle_schema': protocol['artifact_schema']})
    _json(paths['08_reporting'] / 'freeze_receipt.json', {'status': 'FROZEN_BEFORE_PREPARATION',
        'created_at_utc': _now(), 'formal_fits': 0, 'native_requests': 0})
    return protocol


def _frozen(run_id, repo_root=None, research_root=None):
    repo, research, paths = _roots(run_id, repo_root, research_root)
    protocol = _load(paths['05_model_input'] / 'protocol.json')
    binding = _load(paths['05_model_input'] / 'binding.json')
    if 'recovery' in protocol:
        _validate_saved_recovery(repo, paths, protocol)
    from importlib.metadata import version
    if any(version(module) != expected for module, expected in protocol['library_versions'].items()):
        raise ValueError('Frozen numerical-library version changed')
    if str(repo) != protocol['repo_root'] or str(research) != protocol['research_root']:
        raise ValueError('Frozen repository/research root binding differs')
    if _sha(paths['05_model_input'] / 'protocol.json') != binding['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    checked = {}
    def current_hash(path):
        key = str(Path(path).resolve())
        if key not in checked:
            checked[key] = _sha(path)
        return checked[key]
    for pin in (*binding['pins'].values(), *binding['import_origins'].values()):
        if current_hash(pin['path']) != pin['sha256']:
            raise ValueError('Frozen source/code/import pin changed: ' + pin['path'])
    for relative, digest in binding['protected_before'].items():
        if current_hash(repo / relative) != digest:
            raise ValueError('Protected source changed: ' + relative)
    return repo, research, paths, protocol, binding


def _table(path, records):
    if Path(path).exists():
        raise FileExistsError(str(path))
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if 'source_series_order' in frame:
        frame = frame.copy()
        frame['source_series_order'] = [json.dumps(_safe(v)) for v in frame['source_series_order']]
    frame.to_parquet(path, index=False)
    return frame


def _state_features(state):
    candidates = state['candidates']
    return {'counts': [candidates[i]['count'] if i < len(candidates) else 0 for i in range(2)],
            'ages': [candidates[i]['age_days'] if i < len(candidates) else 0 for i in range(2)],
            'previous_available': len(candidates) == 2}


def _membership_bool(row, fields):
    for key in fields:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            if value not in (True, False, 0, 1):
                raise ValueError('Malformed frozen taxonomy ' + key)
            return 'YES' if bool(value) else 'NO'
    return 'UNKNOWN'


def _prepare_run(run_id, *, repo_root=None, research_root=None, _invocation=None):
    """One streamed source parse, complete opportunity and clock ledgers; no fits."""
    repo, research, paths, protocol, binding = _frozen(run_id, repo_root, research_root)
    if 'recovery' in protocol:
        raise ValueError('Evaluation-only recovery cannot prepare or refit')
    _command_start(paths, 'prepare', {} if _invocation is None else _invocation,
                   {'started_at_utc': _now(), 'formal_fits': 0})
    pins = protocol['hashes']
    source = Path(pins['raw_source']['path'])
    native_overrides = _load(pins['native_release_map']['path'])
    canonical_overrides = _load(pins['canonical_release_override']['path'])
    frame = _canonical(pins['canonical_dataset']['path'])
    targets = {r['match_id']: r for r in frame.to_dict('records')}
    from scripts.export_prospective_features import iter_matches
    appearances, maps, clocks, source_targets = [], [], [], {}
    seen, seen_maps = set(), set()
    stream = iter_matches(source)
    try:
        for raw in stream:
            mid = _source_validation(_id, raw.get('match_id'), 'source ID')
            if mid in seen:
                raise SourceValidationError('Duplicate source series identity ' + mid)
            seen.add(mid)
            events, gids = _source_maps(raw, native_overrides)
            if seen_maps & gids:
                raise SourceValidationError('Duplicate map identity')
            seen_maps.update(gids)
            header = _source_validation(_as_day, raw.get('date'), 'source header day')
            map_floor = max(_source_validation(_as_day, e['original_map_day'], 'map day') for e in events)
            override = _source_validation(_as_day, canonical_overrides[mid], 'canonical override') if mid in canonical_overrides else None
            row = targets.get(mid)
            canonical_floor = max(header, map_floor, override or header)
            row_bound = _source_validation(_as_day, row['release_day'], 'canonical row release') if row else None
            if row_bound is not None and row_bound < canonical_floor:
                raise SourceValidationError('Canonical label release precedes canonical source floor ' + mid)
            canonical = max(canonical_floor, row_bound or canonical_floor)
            native = _as_day(events[0]['native_release_day'], 'native release')
            if row_bound is not None and row_bound < native:
                raise SourceValidationError('Canonical target label release precedes native whole-series source release ' + mid)
            effective = max(native, canonical)
            clocks.append({'source_match_id': mid, 'native_floor': map_floor.isoformat(),
                'native_override': native_overrides.get(mid), 'native_release_day': native.isoformat(),
                'canonical_header_day': header.isoformat(), 'canonical_maps_floor': map_floor.isoformat(),
                'canonical_override': canonical_overrides.get(mid), 'canonical_row_bound': row['release_day'] if row else None,
                'canonical_release_day': canonical.isoformat(), 'effective_release_day': effective.isoformat(),
                'disagreement': native != canonical, 'unknown_reason': None,
                'native_inventory_path': pins['native_release_map']['path'],
                'native_inventory_sha256': pins['native_release_map']['sha256'],
                'canonical_inventory_path': pins['canonical_release_override']['path'],
                'canonical_inventory_sha256': pins['canonical_release_override']['sha256']})
            final = max(e['map_index'] for e in events)
            for e in events:
                e.update(canonical_release_day=canonical.isoformat(), effective_release_day=effective.isoformat(),
                    effective_release_reason='canonical_release_bound' if canonical > native else e['effective_release_reason'],
                    best_of=golgg_schema.best_of(raw), source_sha256=pins['raw_source']['sha256'])
                maps.append(dict(e))
                if e['map_index'] == final:
                    appearances.append(dict(e))
            if row is not None:
                forecast = min(header, min(_as_day(e['original_map_day'], 'map day') for e in events))
                if forecast.isoformat() != row['date'] or golgg_schema.best_of(raw) != int(row['best_of']):
                    raise SourceValidationError('Canonical source chronology/format mismatch ' + mid)
                first = {e['team_id']: e['roster'] for e in events if e['map_index'] == 1}
                if set(first) != {row['team_a'], row['team_b']}:
                    raise SourceValidationError('Canonical source orientation identity mismatch ' + mid)
                for team, field in ((row['team_a'], 'roster_a'), (row['team_b'], 'roster_b')):
                    expected = _source_validation(_roster, row[field])
                    actual = first[team]
                    # Compare only evidenced identities: an unobserved first-map
                    # slot is an explicit label absence, not a contradictory ID.
                    known = {p for p in expected if p is not None} | {p for p in actual if p is not None}
                    if len(known) > 5:
                        raise SourceValidationError('Canonical first-map identity mismatch ' + mid)
                    if None not in actual and None not in expected and expected != actual:
                        raise SourceValidationError('Canonical exact first-map label mismatch ' + mid)
                wins = {row['team_a']: 0, row['team_b']: 0}
                for game in raw['games']:
                    flags = (game.get('t1_win'), game.get('t2_win'))
                    if flags[0] not in (True, False, 0, 1) or flags[1] not in (True, False, 0, 1) or flags[0] == flags[1]:
                        raise SourceValidationError('Canonical native winner lineage invalid ' + mid)
                    wins[str(game['t1_id'] if flags[0] else game['t2_id'])] += 1
                if wins[row['team_a']] == wins[row['team_b']]:
                    raise SourceValidationError('Canonical tied winner unsupported')
                winner = int(wins[row['team_a']] > wins[row['team_b']])
                source_targets[mid] = {'native_release_day': native.isoformat(),
                    'canonical_release_day': canonical.isoformat(), 'first_rosters': first,
                    'competition_context': events[0]['competition_context'], 'winner': winner}
    finally:
        if hasattr(stream, 'close'):
            stream.close()
    if set(targets) != set(source_targets):
        raise SourceValidationError('Canonical target missing raw source lineage')
    appearances.sort(key=lambda e: (e['effective_release_day'], _source_series_order(e['source_match_id']), e['team_id']))
    by_context = defaultdict(list)
    for e in appearances:
        by_context[e['team_id'], e['competition_context']].append(e)
    releases = {key: [e['effective_release_day'] for e in value] for key, value in by_context.items()}
    conflict_cache = _clock_conflict_inventory(clocks, sorted(set(frame.date)))
    native_order = sorted(clocks, key=lambda c: (c['native_release_day'], _source_series_order(c['source_match_id'])))
    native_days = [c['native_release_day'] for c in native_order]
    opportunities, origins, series = [], [], []
    for index, row in enumerate(frame.to_dict('records')):
        mid = row['match_id']
        source_target = source_targets[mid]
        day = row['date']
        if source_target['native_release_day'] < day or source_target['canonical_release_day'] < day:
            raise SourceValidationError('Target already released before origin ' + mid)
        context = source_target['competition_context']
        canonical_context = _source_validation(_target_competition_context,
            {'tournament_name': row.get('tournament_name', row.get('tournament')),
             'competition_context': row.get('competition_context')}, mid)
        if canonical_context is not None and context != canonical_context:
            raise SourceValidationError('Canonical source exact context disagreement ' + mid)
        series.append({'row_index': index, 'match_id': mid, 'source_match_id': mid, 'date': day,
            'target_origin_day': day, 'effective_release_day': source_target['canonical_release_day'],
            'native_release_day': source_target['native_release_day'], 'label_release_day': row['release_day'],
            'partition': row['partition'], 'best_of': int(row['best_of']), 'competition_context': context,
            'competition_context_id': context, 'tournament_source_name': context, 'month': day[:7],
            'team_a_id': row['team_a'], 'team_b_id': row['team_b'], 'winner_label': source_target['winner'],
            'roster_a_ids': source_target['first_rosters'][row['team_a']],
            'roster_b_ids': source_target['first_rosters'][row['team_b']],
            'canonical_roster_a_ids': _roster(row['roster_a']), 'canonical_roster_b_ids': _roster(row['roster_b']),
            'major': _membership_bool(row, ('is_major', 'major')),
            'player_source_unseen': _membership_bool(row, ('player_source_unseen', 'source_unseen')),
            'scenario': 'historical_final_appearance_next_first_map', 'terminal_score_label': None,
            'duration_minutes': None, 'label_availability': 'RELEASED_FOR_PARTITION',
            'label_absence_reason': None, 'max_used_release_day': None})
        origins.append({'match_id': mid, 'cutoff_day': day, 'conflict_source_ids': conflict_cache[day],
                        'native_consumed_prefix_length': bisect_left(native_days, day),
                        'native_absence': 'NATIVE_HISTORY_RELEASE_CONFLICT' if conflict_cache[day] else None})
        for side, team, opponent in (('A', row['team_a'], row['team_b']), ('B', row['team_b'], row['team_a'])):
            key = (team, context)
            pos = bisect_left(releases.get(key, []), day)
            prior = by_context.get(key, [])[max(0, pos - 10):pos]
            state = candidate_state(prior, team_id=team, competition_context=context, cutoff_day=day, target_match_id=mid)
            actual = source_target['first_rosters'][team]
            label_absence = 'INCOMPLETE_FIRST_MAP_TARGET' if None in actual else None
            end = '2025-07-01' if row['partition'] == 'TRAIN' else '2026-01-01' if row['partition'] == 'CAL' else '2027-01-01'
            if row['release_day'] >= end or source_target['native_release_day'] >= end:
                label_absence = 'UNRELEASED_LABEL'
            if int(row['best_of']) not in (1, 3, 5):
                label_absence = 'UNSUPPORTED_TARGET_FORMAT'
            absence = state['absence'] or label_absence
            label = None if absence else classify_lineup(state, actual)
            lineage = {'canonical_label_release_day': row['release_day'],
                'canonical_source_release_day': source_target['canonical_release_day'],
                'native_release_day': source_target['native_release_day'],
                'source_sha256': pins['raw_source']['sha256'],
                'native_release_map_sha256': pins['native_release_map']['sha256'],
                'canonical_release_override_sha256': pins['canonical_release_override']['sha256']}
            opportunities.append({'row_index': index, 'match_id': mid, 'side': side,
                'partition': row['partition'], 'date': day, 'month': day[:7], 'best_of': int(row['best_of']),
                'team_id': team, 'opponent_id': opponent, 'competition_context': context,
                'target_origin_day': day, 'native_release_day': source_target['native_release_day'],
                'label_release_day': row['release_day'], 'actual_roster': actual, 'label': label,
                'label_absence': label_absence, 'absence': absence,
                'history_ids': state['history_ids'], 'incomplete_history_count': state['incomplete_history_count'],
                'history_state': 'cold' if not state['history_ids'] else 'full' if len(state['history_ids']) == 10 else 'partial',
                'state': json.dumps(state, sort_keys=True), 'candidates': json.dumps(state['candidates'], sort_keys=True),
                'release_lineage': json.dumps(lineage, sort_keys=True), **_state_features(state)})
    # The original 2026 native reference is alignment evidence, never candidate inference.
    metadata = pd.read_parquet(pins['native_participants']['path'])
    saved = pd.read_parquet(pins['saved_c0_reference']['path'])
    assessment = pd.DataFrame([r for r in series if r['partition'] == 'ASSESS']).rename(columns={'team_a_id': 'team_a', 'team_b_id': 'team_b', 'winner_label': 'winner'})
    from src.models.champion_history_runner import _align_c0_reference
    _, reversal = _source_validation(_align_c0_reference, saved, metadata, assessment)
    native_metadata = metadata.assign(match_id=metadata.golgg_match_id.astype(str)).set_index('match_id')
    for row, reversed_ in zip(assessment.to_dict('records'), reversal):
        reference = native_metadata.loc[row['match_id']]
        for own, ref in (('roster_a_ids', 'roster_b' if reversed_ else 'roster_a'),
                         ('roster_b_ids', 'roster_a' if reversed_ else 'roster_b')):
            if _source_validation(_roster, reference[ref], complete=True) != row['canonical_' + own]:
                raise SourceValidationError('Canonical native first-map roster orientation mismatch')
    from src.models.competition_tiers import classify_competition
    if 'unknown_players' not in native_metadata:
        raise SourceValidationError('Frozen benchmark source-unseen metadata unavailable')
    for row in series:
        profile = classify_competition(row['tournament_source_name'], _as_day(row['date'], 'target day'))
        tier = profile.tier.value
        row['profile_tier'] = tier
        row['major'] = 'UNKNOWN' if tier == 'unknown' else 'YES' if tier == 'major' else 'NO'
        unknown = native_metadata.loc[row['match_id'], 'unknown_players'] if row['match_id'] in native_metadata.index else None
        if unknown is None or pd.isna(unknown):
            if row['partition'] == 'ASSESS':
                raise SourceValidationError('ASSESS benchmark unknown-player lineage unavailable')
            row['benchmark_unknown_players'] = None
            row['player_source_unseen'] = 'UNKNOWN'
        else:
            if not np.isfinite(float(unknown)) or float(unknown) < 0:
                raise SourceValidationError('Invalid benchmark unknown-player count')
            row['benchmark_unknown_players'] = float(unknown)
            row['player_source_unseen'] = 'YES' if unknown > 0 else 'NO'
    series_by_id = {r['match_id']: r for r in series}
    for row in opportunities:
        s = series_by_id[row['match_id']]
        row.update(major=s['major'], player_source_unseen=s['player_source_unseen'])
    sides_by_match = defaultdict(list)
    for r in opportunities:
        sides_by_match[r['match_id']].append(r)
    appearance_days = {(e['source_match_id'], e['team_id'], e['competition_context']): e['effective_release_day']
                       for e in appearances}
    for s in series:
        side_rows = sides_by_match[s['match_id']]
        # Complete side lineage is already available; no scoring outcome determines membership.
        used_days = [appearance_days[mid, r['team_id'], r['competition_context']]
                     for r in side_rows for mid in r['history_ids']]
        s['max_used_release_day'] = max(used_days) if used_days else None
        label_available = sum(r['label_absence'] is None for r in side_rows)
        s['lineup_label_availability'] = 'ALL_SIDES' if label_available == 2 else 'ONE_SIDE' if label_available else 'ABSENT'
        s['lineup_label_absence_reasons'] = [r['label_absence'] for r in side_rows]
        for r in side_rows:
            suffix = r['side'].lower()
            s['history_state_' + suffix] = r['history_state']
            s['candidate_count_' + suffix] = len(json.loads(r['candidates']))
            s['incomplete_history_' + suffix] = r['incomplete_history_count']
    _json(paths['08_reporting'] / 'opportunity_inventory.json', _opportunity_inventory(opportunities))
    _table(paths['04_feature'] / 'appearances.parquet', appearances)
    _table(paths['04_feature'] / 'source_maps.parquet', maps)
    _table(paths['04_feature'] / 'opportunities.parquet', opportunities)
    _table(paths['04_feature'] / 'series.parquet', series)
    _table(paths['04_feature'] / 'native_source_order.parquet',
           [{'native_rank': i, **c} for i, c in enumerate(native_order)])
    _table(paths['08_reporting'] / 'clock_comparison.parquet', clocks)
    _table(paths['08_reporting'] / 'origin_conflicts.parquet', origins)
    public_cases = _public_cases(paths, series, opportunities)
    _json(paths['05_model_input'] / 'public_cases.json', public_cases)
    public_contract = _public_case_contracts(paths, public_cases)
    _json(paths['05_model_input'] / 'public_case_contracts.json', public_contract)
    eligible = [r for r in opportunities if r['label'] is not None]
    assessment_ids = {r['match_id'] for r in eligible if r['partition'] == 'ASSESS'}
    counts = {'identity_release_valid': True,
        'train_side_labels': sum(r['partition'] == 'TRAIN' for r in eligible),
        'train_previous_labels': sum(r['partition'] == 'TRAIN' and r['label'] == 'PREVIOUS' for r in eligible),
        'train_other_labels': sum(r['partition'] == 'TRAIN' and r['label'] == 'OTHER' for r in eligible),
        'assessment_series': len(assessment_ids),
        'assessment_months': len({r['month'] for r in eligible if r['partition'] == 'ASSESS'}),
        'cal_side_labels': sum(r['partition'] == 'CAL' for r in eligible)}
    for label in CLASS_ORDER:
        counts['cal_' + label.lower() + '_labels'] = sum(r['partition'] == 'CAL' and r['label'] == label for r in eligible)
    support = {'schema': 'lineup_law_reliability_support_v1', 'created_at_utc': _now(),
        **support_gate(counts), 'canonical_series_by_partition': dict(Counter(r['partition'] for r in series)),
        'opportunity_sides_by_partition': dict(Counter(r['partition'] for r in opportunities)),
        'eligible_sides_by_partition': dict(Counter(r['partition'] for r in eligible)),
        'class_counts': {p: dict(Counter(r['label'] for r in eligible if r['partition'] == p)) for p in ('TRAIN', 'CAL', 'ASSESS')},
        'absence_counts': dict(Counter(r['absence'] or 'AVAILABLE' for r in opportunities)),
        'history_state_counts': dict(Counter(r['history_state'] for r in opportunities)),
        'source_parse_count': 1, 'source_series': len(seen), 'source_maps': len(seen_maps),
        'clock_disagreements': sum(c['disagreement'] for c in clocks), 'canonical_unknown_bounds': 0,
        'prepared_hashes': {str(p.resolve()): _sha(p) for s in ('04_feature', '08_reporting') for p in paths[s].glob('*.parquet')},
        'frozen_input_hashes': pins, 'protocol_sha256': binding['protocol_sha256'],
        'limits': LIMITATIONS}
    support['prepared_hashes'][str((paths['05_model_input'] / 'public_cases.json').resolve())] = _sha(paths['05_model_input'] / 'public_cases.json')
    support['prepared_hashes'][str((paths['05_model_input'] / 'public_case_contracts.json').resolve())] = _sha(paths['05_model_input'] / 'public_case_contracts.json')
    _json(paths['08_reporting'] / 'support.json', support)
    _json(paths['08_reporting'] / ('prepare_operation_trace.json' if support['status'] == 'READY_FOR_TWO_TRAIN_FITS' else 'operation_trace.json'), [])
    if support['status'] != 'READY_FOR_TWO_TRAIN_FITS':
        _decision(paths, protocol, support['status'], [], ['SUPPORT_GATE'], not_run='NOT_RUN_BLOCKED')
    _oracle_index(repo, paths, protocol, support['status'], [], public_cases)
    return support


def _protected_after(protocol, checked=None):
    repo = Path(protocol['repo_root'])
    checked = {} if checked is None else checked
    result = {}
    for relative in protocol['protected_before']:
        path = str((repo / relative).resolve())
        if path not in checked:
            checked[path] = _sha(path)
        result[relative] = checked[path]
    if result != protocol['protected_before']:
        raise ValueError('Protected artifacts changed during study')
    return result


def _decision(paths, protocol, status, operations, failures, *, not_run=None):
    if 'recovery' in protocol and not_run is None:
        _validate_saved_recovery(Path(protocol['repo_root']), paths, protocol)
    slots = {str(slot): {'status': not_run or status, 'reason': failures or None} for slot in (7, 8)}
    if not_run is None and (paths['08_reporting'] / 'metrics.json').exists():
        primary = _load(paths['08_reporting'] / 'metrics.json')['primary']
        for slot, metric in ((7, 'nll'), (8, 'brier')):
            contrast = primary[metric]
            gain = (contrast['status'] == 'ESTIMABLE' and contrast['upper'] is not None and contrast['upper'] < 0
                    and contrast['equal_block']['point'] is not None and contrast['equal_block']['point'] < 0)
            slots[str(slot)] = {'status': 'PASS_PRIMARY_CRITERION' if gain else 'NO_ESTABLISHED_GAIN',
                                'metric': metric, 'contrast': contrast}
    checked = {}
    protected_after = _protected_after(protocol, checked)
    for pin in (*protocol['hashes'].values(), *protocol['import_origins'].values()):
        path = str(Path(pin['path']).resolve())
        if path not in checked:
            checked[path] = _sha(path)
    receipt = {'schema': 'lineup_law_reliability_decision_v1', 'run_id': protocol['run_id'],
        'scope': protocol['scope'], 'created_at_utc': _now(), 'status': status, 'failures': failures,
        'formal_fits': len(operations), 'completed_fits': sum(o.get('status') == 'COMPLETED' for o in operations),
        'CAL_fits': 0, 'C0_fits': 0, 'ASSESS_fits': 0, 'unused_TRAIN_slots': 2 - len(operations),
        'primary_slots': slots, 'operation_trace': operations,
        'protected_before': protocol['protected_before'], 'protected_after': protected_after,
        'frozen_input_hashes': protocol['hashes'], 'limitations': LIMITATIONS}
    receipt['source_after'] = {role: checked[str(Path(pin['path']).resolve())]
                               for role, pin in protocol['hashes'].items()}
    receipt['code_after'] = {module: checked[str(Path(pin['path']).resolve())]
                             for module, pin in protocol['import_origins'].items()}
    if (any(receipt['source_after'][role] != pin['sha256'] for role, pin in protocol['hashes'].items())
            or any(receipt['code_after'][module] != pin['sha256'] for module, pin in protocol['import_origins'].items())):
        raise ValueError('Frozen source/code closure changed during study')
    if 'recovery' in protocol:
        receipt.update(inherited_TRAIN_fits=2, new_fit_counts=protocol['recovery']['new_fit_budget'],
                       recovery=protocol['recovery'], unused_TRAIN_slots=0)
    target_decision = paths['08_reporting'] / 'decision.json'
    if target_decision.exists():
        target_decision.unlink()
    _json(target_decision, receipt)
    return receipt


def _oracle_index(repo, paths, protocol, status, operations, public_cases=()):
    import pyarrow.parquet as pq
    descriptors = []
    for stage, directory in paths.items():
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.name.startswith('oracle_inputs'):
                continue
            shape, dtype = None, None
            if path.suffix == '.parquet':
                meta = pq.ParquetFile(path)
                shape = meta.metadata.num_rows
                dtype = {f.name: str(f.type) for f in meta.schema_arrow}
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as data:
                    shape = {k: list(data[k].shape) for k in data.files}
                    dtype = {k: str(data[k].dtype) for k in data.files}
            else:
                shape, dtype = 'JSON object or list', 'finite JSON'
                if path.parent.name == 'snapshots':
                    wire = _load(path)['encoded_snapshot']
                    shape = {family + '.' + key: array['shape'] for family in ('raw', 'hist')
                             for key, array in wire[family].items()}
                    dtype = {family + '.' + key: array['dtype'] for family in ('raw', 'hist')
                             for key, array in wire[family].items()}
            descriptors.append({'role': path.stem, 'repo_relative_path': str(path.relative_to(repo)),
                'sha256': _sha(path), 'format': path.suffix[1:], 'shape_or_row_count': shape,
                'dtype_or_columns': dtype, 'units': 'declared fields; release ages in UTC days',
                'ordered_axis_names': ['canonical_series', 'side'] if path.stem == 'opportunities' else ['record'],
                'ordered_id_axis_reference': str((paths['04_feature'] / 'series.parquet').relative_to(repo)),
                'absence_status': 'AVAILABLE', 'absence_reason': None})
    for role, pin in protocol['hashes'].items():
        path = Path(pin['path'])
        relative = str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
        descriptors.append({'role': role, 'repo_relative_path': relative, 'sha256': pin['sha256'],
            'format': path.suffix[1:], 'shape_or_row_count': 'pinned complete source, no altered subset',
            'dtype_or_columns': 'original source schema', 'units': 'original source fields',
            'ordered_axis_names': ['source_record'], 'ordered_id_axis_reference': None,
            'absence_status': 'AVAILABLE', 'absence_reason': None})
    closure_reason = status
    for command in ('prepare', 'run', 'recover_evaluation'):
        stop = paths['08_reporting'] / (command + '_hard_stop.json')
        if stop.exists():
            closure_reason = _load(stop)['failure_reason']
    for stage, name in (('04_feature', 'appearances.parquet'), ('04_feature', 'source_maps.parquet'),
                        ('04_feature', 'opportunities.parquet'), ('04_feature', 'series.parquet'),
                        ('04_feature', 'native_source_order.parquet'),
                        ('05_model_input', 'public_cases.json'), ('05_model_input', 'public_case_contracts.json'),
                        ('06_models', 'conditional.json'), ('06_models', 'global.json'),
                        ('07_model_output', 'side_laws.parquet'), ('07_model_output', 'series_scores.parquet'),
                        ('07_model_output', 'native_pairs.parquet'), ('07_model_output', 'native_blocked_pairs.parquet'),
                        ('07_model_output', 'roster_bounds.parquet'),
                        ('08_reporting', 'clock_comparison.parquet'), ('08_reporting', 'origin_conflicts.parquet'),
                        ('08_reporting', 'opportunity_inventory.json'), ('08_reporting', 'cal_audit.json'),
                        ('08_reporting', 'native_execution.json'), ('08_reporting', 'query_nonmutation.json'),
                        ('08_reporting', 'saved_output_binding.json'), ('08_reporting', 'public_cases.json'),
                        ('08_reporting', 'metrics.json'), ('08_reporting', 'contrasts.npz')):
        if not (paths[stage] / name).exists():
            descriptors.append({'role': Path(name).stem, 'repo_relative_path': str((paths[stage] / name).relative_to(repo)),
                'sha256': None, 'format': Path(name).suffix[1:], 'shape_or_row_count': None,
                'dtype_or_columns': None, 'units': None, 'ordered_axis_names': [],
                'ordered_id_axis_reference': None, 'absence_status': 'NOT_RUN',
                'absence_reason': closure_reason})
    index = {'schema_version': 1, 'study': 'R1', 'run_id': protocol['run_id'],
        'scenario': 'historical_final_appearance_next_first_map', 'branch_status': status,
        'created_at_utc': _now(), 'protocol_sha256': _sha(paths['05_model_input'] / 'protocol.json'),
        'binding_sha256': _sha(paths['05_model_input'] / 'binding.json'),
        'oracle_contract_sha256': protocol['hashes']['oracle_contract.json']['sha256'],
        'artifact_descriptors': descriptors, 'population_ids': protocol['population_ids'],
        'operation_trace': operations, 'missingness_ledger': str((paths['04_feature'] / 'opportunities.parquet').relative_to(repo)),
        'public_cases': public_cases, 'prefit_resolutions': protocol['prefit_resolutions'],
        'source_pins': protocol['hashes'], 'protected_before': protocol['protected_before'],
        'protected_after': _protected_after(protocol)}
    if 'recovery' in protocol:
        index.update(recovery=protocol['recovery'], inherited_TRAIN_fits=2,
                     new_fit_counts=protocol['recovery']['new_fit_budget'], new_operation_trace=[])
    name = 'oracle_inputs_prepared.json' if status == 'READY_FOR_TWO_TRAIN_FITS' else 'oracle_inputs.json'
    target_index = paths['08_reporting'] / name
    if target_index.exists():
        target_index.unlink()
    _json(target_index, index)
    return index


def _eligible_rows(frame):
    records = frame.to_dict('records')
    for row in records:
        if row.get('label') is None or pd.isna(row.get('label')):
            row['label'] = None
    return records


def _law_rows(opportunities, model, law, model_sha):
    eligible = [r for r in opportunities if not json.loads(r['state'])['absence']]
    q = model.predict(np.asarray([r['counts'] for r in eligible]),
                      np.asarray([r['ages'] for r in eligible]),
                      np.asarray([r['previous_available'] for r in eligible], dtype=bool))
    values = {(r['match_id'], r['side']): p for r, p in zip(eligible, q)}
    output = []
    for row in opportunities:
        p = values.get((row['match_id'], row['side']))
        label = CLASS_ORDER.index(row['label']) if row['label'] is not None else None
        output.append({**row, 'law': law, 'model_sha256': model_sha,
            'q_latest': float(p[0]) if p is not None else None,
            'q_previous': float(p[1]) if p is not None else None,
            'q_other': float(p[2]) if p is not None else None,
            'lineup_nll': -float(np.log(p[label])) if p is not None and label is not None else None,
            'other_brier': float((p[2] - (label == 2)) ** 2) if p is not None and label is not None else None})
    return output


def _diagnostics(law_rows, opportunities, bounds):
    result = {}
    for partition in ('TRAIN', 'CAL', 'ASSESS'):
        for law in ('conditional', 'global'):
            rows = [r for r in law_rows if r['partition'] == partition and r['law'] == law]
            eligible = [r for r in rows if r['label'] is not None and r.get('q_other') is not None and not pd.isna(r.get('q_other'))]
            reliability = []
            for bin_id in range(10):
                selected = [r for r in eligible if min(9, int(r['q_other'] * 10)) == bin_id]
                reliability.append({'bin': bin_id, 'n_sides': len(selected),
                    'mean_q_other': float(np.mean([r['q_other'] for r in selected])) if selected else None,
                    'realized_other': float(np.mean([r['label'] == 'OTHER' for r in selected])) if selected else None,
                    'absence_reason': None if selected else 'EMPTY_FIXED_BIN'})
            series_losses = score_side_laws([{'match_id': r['match_id'], 'side': r['side'], 'label': r['label'],
                'q': [r['q_latest'], r['q_previous'], r['q_other']]} for r in eligible])
            absence_counts = dict(Counter(('AVAILABLE' if r.get('absence') is None or pd.isna(r.get('absence')) else r['absence']) for r in rows))
            cand_counts = {str(k): v for k, v in Counter(len(json.loads(r['candidates']) if isinstance(r['candidates'], str) else r['candidates']) for r in rows).items()}
            hist_lengths = {str(k): v for k, v in Counter(len(json.loads(r['history_ids']) if isinstance(r['history_ids'], str) else r['history_ids']) for r in rows).items()}
            inc_counts = {str(k): v for k, v in Counter(r['incomplete_history_count'] for r in rows).items()}
            all_cands = [c for r in rows for c in (json.loads(r['candidates']) if isinstance(r['candidates'], str) else r['candidates'])]
            result[partition + ':' + law] = {
                'opportunity_sides': len(rows), 'eligible_sides': len(eligible),
                'eligible_series': len({r['match_id'] for r in eligible}),
                'class_counts': dict(Counter(r['label'] for r in eligible)),
                'absence_counts': absence_counts,
                'candidate_recall': float(np.mean([r['label'] != 'OTHER' for r in eligible])) if eligible else None,
                'realized_other_rate': float(np.mean([r['label'] == 'OTHER' for r in eligible])) if eligible else None,
                'mean_nll_side': float(np.mean([r['lineup_nll'] for r in eligible])) if eligible else None,
                'mean_other_brier_side': float(np.mean([r['other_brier'] for r in eligible])) if eligible else None,
                'other_reliability': reliability,
                'candidate_counts': cand_counts,
                'history_lengths': hist_lengths,
                'incomplete_history_counts': inc_counts,
                'release_age_days': _summary([c['age_days'] for c in all_cands]),
                'played_age_days': _summary([(_as_day(r['date'], 'origin') - _as_day(c['last_played_day'], 'played')).days
                                             for r in rows for c in (json.loads(r['candidates']) if isinstance(r['candidates'], str) else r['candidates'])])}
    for law in ('conditional', 'global', 'latest_point'):
        rows = [r for r in bounds if r['law'] == law]
        valid = [r for r in rows if r.get('missing_mass') is not None]
        result['bounds:' + law] = {'opportunities': len(rows), 'valid_series': len(valid),
            'absence_counts': dict(Counter(('AVAILABLE' if r.get('absence') is None or pd.isna(r.get('absence')) else r['absence']) for r in rows)),
            'missing_mass': _summary([r['missing_mass'] for r in valid]),
            'width': _summary([r['upper'] - r['lower'] for r in valid]),
            'scalar_available': sum(r.get('prob_a') is not None for r in rows),
            'scalar_absence_counts': dict(Counter(('AVAILABLE' if (r.get('scalar_absence') or r.get('absence')) is None or pd.isna(r.get('scalar_absence') or r.get('absence')) else (r.get('scalar_absence') or r.get('absence'))) for r in rows))}
        profiles = {}
        for dimension in ('month', 'best_of', 'competition_context', 'major', 'player_source_unseen',
                          'candidate_pattern', 'history_pattern', 'incomplete_pattern', 'absence'):
            grouped = defaultdict(list)
            for row in rows:
                grouped[str(row.get(dimension) or 'AVAILABLE_OR_UNKNOWN')].append(row)
            profiles[dimension] = {}
            for key, members in sorted(grouped.items()):
                available = [r for r in members if r.get('missing_mass') is not None]
                profiles[dimension][key] = {'series': len(members), 'valid_series': len(available),
                    'width': _summary([r['upper'] - r['lower'] for r in available]),
                    'missing_mass': _summary([r['missing_mass'] for r in available]),
                    'scalar_available': sum(r.get('prob_a') is not None for r in members),
                    'absences': dict(Counter(r.get('absence') or 'AVAILABLE' for r in members))}
        result['bounds:' + law]['profiles'] = profiles
    return result


def _summary(values):
    if not values:
        return {'n': 0, 'mean': None, 'min': None, 'max': None, 'quantiles': None, 'reason': 'NO_OBSERVATIONS'}
    a = np.asarray(values, dtype=np.float64)
    return {'n': len(a), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max()),
            'quantiles': dict(zip(('p0', 'p10', 'p25', 'p50', 'p75', 'p90', 'p100'),
                                  np.quantile(a, [0, .1, .25, .5, .75, .9, 1]).tolist())), 'reason': None}


def _saved_result(record):
    from betting_app.core.models.contract import UnifiedPredictionResult
    return UnifiedPredictionResult(**record)


def _capture_pairs(paths, protocol, series, opportunities, *, inherited=None):
    """One owned worker; sorted origins; exact captured wire and provenance."""
    from dataclasses import asdict
    from betting_app.services.c0_inference import C0InferenceClient, C0WorkerTimeout
    import time
    origin_ledger = {r['match_id']: r for r in pd.read_parquet(paths['08_reporting'] / 'origin_conflicts.parquet').to_dict('records')}
    conflicts = {mid: list(r['conflict_source_ids']) for mid, r in origin_ledger.items()}
    source_series_count = _load(paths['08_reporting'] / 'support.json')['source_series']
    sides = {(r['match_id'], r['side']): r for r in opportunities}
    pair_rows = [{**row, 'execution_scope': 'PARENT_INHERITED'} for row in inherited['pairs']] if inherited else []
    blocked_rows = [{**row, 'execution_scope': 'PARENT_INHERITED'} for row in inherited['blocked']] if inherited else []
    captured, statuses = {}, {}
    no_retry = set(inherited['no_retry_native_match_ids']) if inherited else set()
    worker_sessions, startup_attempts, cutoff_advances = [], [], set()
    confirmed_advances = set()
    active_session = None
    client = C0InferenceClient(protocol['hashes']['native_config']['path'])
    worker_environment = {'C0_RESEARCH_ROOT': protocol['research_root'], 'OMP_NUM_THREADS': '4',
                          'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': ''}
    previous_environment = {name: os.environ.get(name) for name in worker_environment}
    os.environ.update(worker_environment)
    try:
        for row in sorted((r for r in series if r['partition'] == 'ASSESS'), key=lambda r: (r['date'], _source_series_order(r['match_id']))):
            mid = row['match_id']
            if mid in no_retry:
                statuses[mid] = 'INHERITED_NATIVE_FAILURE_NO_RETRY'
                captured[mid] = {}
                continue
            target = {'match_id': mid, 'team1_id': row['team_a_id'], 'team2_id': row['team_b_id'],
                'best_of': int(row['best_of']), 'competition_context': row['competition_context'],
                'decision_at': row['date'] + 'T00:00:00+00:00', 'start_at': None,
                'native_release_day': row['native_release_day'], 'label_release_day': row['label_release_day']}
            a, b = json.loads(sides[mid, 'A']['state']), json.loads(sides[mid, 'B']['state'])
            if a['absence'] or b['absence']:
                statuses[mid] = a['absence'] or b['absence']
                continue
            if conflicts[mid]:
                statuses[mid] = 'NATIVE_HISTORY_RELEASE_CONFLICT'
                for i in range(len(a['candidates'])):
                    for j in range(len(b['candidates'])):
                        blocked_rows.append({'match_id': mid, 'candidate_a_index': i, 'candidate_b_index': j,
                            'reason': 'NATIVE_HISTORY_RELEASE_CONFLICT', 'conflict_source_ids': conflicts[mid],
                            'cutoff_day': row['date'], 'native_requested': False})
                continue
            requests = {}
            for i, ca in enumerate(a['candidates']):
                for j, cb in enumerate(b['candidates']):
                    single_a, single_b = {**a, 'candidates': [ca]}, {**b, 'candidates': [cb]}
                    try:
                        one = build_native_pair_requests(target, single_a, single_b)
                    except ValueError as error:
                        if 'overlap' not in str(error) and 'share' not in str(error):
                            raise
                        statuses[mid] = 'CROSS_PAIR_PLAYER_OVERLAP'
                        blocked_rows.append({'match_id': mid, 'candidate_a_index': i, 'candidate_b_index': j,
                            'reason': 'CROSS_PAIR_PLAYER_OVERLAP', 'conflict_source_ids': [],
                            'cutoff_day': row['date'], 'native_requested': False})
                    else:
                        requests[i, j] = one[0, 0]
            # Preserve resident state for the whole origin, reserving its maximum
            # request time as well as its at-most-four request-count allowance.
            expires_within_origin = (client._process is not None and
                time.monotonic() - client._started_at + len(requests) * (client.request_timeout + 1)
                >= client.max_worker_lifetime)
            if requests and (expires_within_origin or client._worker_requests + len(requests) > client.max_worker_requests):
                client.close()
                if active_session is not None and active_session['closed_at'] is None:
                    active_session.update(closed_at=_now(), close_reason='ORIGIN_BUDGET_RECYCLE')
                active_session = None
                client = C0InferenceClient(protocol['hashes']['native_config']['path'])
            matrix = {}
            for key, request in requests.items():
                started = _now()
                request_before = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                native_requested, phase, path = False, 'STARTUP', None
                startup = None
                path = paths['07_model_output'] / 'snapshots' / (hashlib.sha256(mid.encode()).hexdigest() + f'_{key[0]}_{key[1]}.json')
                if path.is_file():
                    try:
                        capture = _load(path)
                        if (capture.get('schema') == 'lineup_law_native_capture_v1'
                                and capture.get('match_id') == mid
                                and tuple(capture.get('pair', ())) == key
                                and 'prediction' in capture
                                and 'native_history_provenance' in capture):
                            pred = _saved_result(capture['prediction'])
                            provenance = capture['native_history_provenance']
                            if active_session is None:
                                client.identity
                                if client._process is not None:
                                    active_session = {'session_index': len(worker_sessions), 'pid': client._process.pid,
                                        'established_at': _now(), 'closed_at': None, 'close_reason': None}
                                    worker_sessions.append(active_session)
                                    startup_attempts.append({'attempt_index': len(startup_attempts), 'status': 'ESTABLISHED',
                                        'session_index': active_session['session_index'], 'failure_type': None, 'failure_reason': None})
                            sess_idx = active_session['session_index'] if active_session is not None else 0
                            pair_rows.append({'match_id': mid, 'candidate_a_index': key[0], 'candidate_b_index': key[1],
                                'status': 'AVAILABLE', 'failure_reason': None, 'failure_kind': None, 'failure_type': None,
                                'failure_phase': None, 'native_requested': True, 'session_index': sess_idx,
                                'roster_a': request['roster_a'], 'roster_b': request['roster_b'],
                                'competition_context': request['competition_context'],
                                'decision_at': request['decision_at'], 'cutoff_day': row['date'], 'mode': 'full',
                                'best_of': row['best_of'], 'prob_a': pred.prob_a, 'prob_b': pred.prob_b,
                                'model_name': pred.model_name, 'model_version': pred.model_version,
                                'feature_version': pred.feature_version,
                                'input_hashes': json.dumps(pred.diagnostics['input_hashes'], sort_keys=True),
                                'source_sha256': provenance['raw_history_sha256'], 'release_sha256': provenance['release_map_sha256'],
                                'history_series_consumed': provenance['history_series_consumed'],
                                'feature_source_max_day': provenance.get('feature_source_max_day'), 'requested_at': capture['generated_at'],
                                'generated_at': capture['generated_at'], 'received_at': capture.get('received_at', capture.get('parent_received_at')),
                                'evidence_path': str(path), 'evidence_sha256': _sha(path)})
                            matrix[key] = pred
                            cutoff_advances.add((sess_idx, row['date']))
                            confirmed_advances.add((sess_idx, row['date']))
                            continue
                    except Exception:
                        pass
                try:
                    # Identity establishes and validates a ready worker before any
                    # prediction. A first-request timeout must not erase its session.
                    needs_startup = (client._process is None or client._process.poll() is not None
                        or client._worker_requests >= client.max_worker_requests
                        or time.monotonic() - client._started_at >= client.max_worker_lifetime)
                    if needs_startup:
                        if active_session is not None and active_session['closed_at'] is None:
                            active_session.update(closed_at=_now(), close_reason='CLIENT_RECYCLE')
                        active_session = None
                        startup = {'attempt_index': len(startup_attempts), 'status': 'ATTEMPTED',
                                   'session_index': None, 'failure_type': None, 'failure_reason': None}
                        startup_attempts.append(startup)
                    client.identity
                    if client._process is None:
                        raise NativeIntegrityError('Verified worker establishment has no owned process')
                    if active_session is None:
                        active_session = {'session_index': len(worker_sessions), 'pid': client._process.pid,
                            'established_at': _now(), 'closed_at': None, 'close_reason': None}
                        worker_sessions.append(active_session)
                    if startup is not None:
                        startup.update(status='ESTABLISHED', session_index=active_session['session_index'])
                    phase, native_requested = 'PREDICTION', True
                    cutoff_advances.add((active_session['session_index'], row['date']))
                    evidence = client.predict_evidence(request)
                    path = paths['07_model_output'] / 'snapshots' / (hashlib.sha256(mid.encode()).hexdigest() + f'_{key[0]}_{key[1]}.json')
                    capture = {'schema': 'lineup_law_native_capture_v1', 'match_id': mid, 'pair': list(key),
                        'request': request, 'encoded_snapshot': evidence.encoded_snapshot,
                        'native_history_provenance': evidence.native_history_provenance, 'prediction': asdict(evidence.prediction),
                        'generated_at': evidence.generated_at.isoformat(), 'received_at': evidence.parent_received_at.isoformat(),
                        'source_pins': protocol['hashes'], 'repo_root': protocol['repo_root'],
                        'research_root': protocol['research_root']}
                    _json(path, capture)
                    if hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest() != request_before:
                        raise NativeIntegrityError('Native query mutated candidate request')
                    provenance = evidence.native_history_provenance
                    if (provenance.get('history_series_total') != source_series_count
                            or provenance.get('history_series_consumed') != origin_ledger[mid]['native_consumed_prefix_length']):
                        raise NativeIntegrityError('Native consumed source IDs do not bind to the complete frozen release-order prefix')
                    source_day = provenance.get('feature_source_max_day')
                    if source_day is not None and source_day >= row['date']:
                        raise NativeIntegrityError('Native captured history reached origin boundary')
                    if provenance.get('raw_history_sha256') != protocol['hashes']['raw_source']['sha256'] or provenance.get('release_map_sha256') != protocol['hashes']['native_release_map']['sha256']:
                        raise NativeIntegrityError('Native capture source/release provenance differs from frozen inputs')
                    if evidence.prediction.diagnostics.get('input_hashes') is None:
                        raise NativeIntegrityError('Native input nonmutation digest missing')
                    pair_rows.append({'match_id': mid, 'candidate_a_index': key[0], 'candidate_b_index': key[1],
                        'status': 'AVAILABLE', 'failure_reason': None, 'failure_kind': None, 'failure_type': None,
                        'failure_phase': None, 'native_requested': True, 'session_index': active_session['session_index'],
                        'roster_a': request['roster_a'], 'roster_b': request['roster_b'],
                        'competition_context': request['competition_context'],
                        'decision_at': request['decision_at'], 'cutoff_day': row['date'], 'mode': 'full',
                        'best_of': row['best_of'], 'prob_a': evidence.prediction.prob_a, 'prob_b': evidence.prediction.prob_b,
                        'model_name': evidence.prediction.model_name, 'model_version': evidence.prediction.model_version,
                        'feature_version': evidence.prediction.feature_version,
                        'input_hashes': json.dumps(evidence.prediction.diagnostics['input_hashes'], sort_keys=True),
                        'source_sha256': provenance['raw_history_sha256'], 'release_sha256': provenance['release_map_sha256'],
                        'history_series_consumed': provenance['history_series_consumed'],
                        'feature_source_max_day': source_day, 'requested_at': started,
                        'generated_at': evidence.generated_at.isoformat(), 'received_at': evidence.parent_received_at.isoformat(),
                        'evidence_path': str(path), 'evidence_sha256': _sha(path)})
                    matrix[key] = evidence.prediction
                    confirmed_advances.add((active_session['session_index'], row['date']))
                except BaseException as error:
                    if startup is not None and startup['status'] == 'ATTEMPTED':
                        startup.update(status='FAILED', failure_type=type(error).__name__, failure_reason=str(error))
                    recoverable = isinstance(error, C0WorkerTimeout)
                    status = 'NATIVE_PAIR_UNAVAILABLE' if recoverable else 'NATIVE_PAIR_FATAL'
                    pair_rows.append({'match_id': mid, 'candidate_a_index': key[0], 'candidate_b_index': key[1],
                        **request, 'cutoff_day': row['date'], 'status': status,
                        'failure_reason': str(error), 'failure_type': type(error).__name__,
                        'failure_kind': 'AVAILABILITY' if recoverable else _failure_kind(error),
                        'failure_phase': phase, 'native_requested': native_requested,
                        'session_index': active_session['session_index'] if active_session is not None else None,
                        'requested_at': started, 'received_at': _now(),
                        'evidence_path': str(path) if path is not None and path.exists() else None})
                    statuses[mid] = status
                    if active_session is not None and client._process is None and active_session['closed_at'] is None:
                        active_session.update(closed_at=_now(), close_reason='WORKER_REQUEST_FAILURE')
                    for uncalled in requests:
                        if uncalled > key:
                            blocked_rows.append({'match_id': mid, 'candidate_a_index': uncalled[0],
                                'candidate_b_index': uncalled[1], 'reason': 'NOT_REQUESTED_AFTER_NATIVE_PAIR_FAILURE',
                                'conflict_source_ids': [], 'cutoff_day': row['date'], 'native_requested': False})
                    if not recoverable:
                        raise
                    break
            captured[mid] = matrix
            statuses.setdefault(mid, None)
    finally:
        try:
            client.close()
            if active_session is not None and active_session['closed_at'] is None:
                active_session.update(closed_at=_now(), close_reason='OWNED_CLIENT_FINALIZATION')
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        # Failure evidence is durable even when the current pair is fatal.
        if inherited:
            for row in pair_rows:
                row.setdefault('execution_scope', 'RECOVERY_NEW')
            for row in blocked_rows:
                row.setdefault('execution_scope', 'RECOVERY_NEW')
        for p in (paths['07_model_output'] / 'native_pairs.parquet',
                  paths['07_model_output'] / 'native_blocked_pairs.parquet',
                  paths['08_reporting'] / 'native_execution.json'):
            if p.exists():
                p.unlink()
        _table(paths['07_model_output'] / 'native_pairs.parquet', pair_rows if pair_rows else
               pd.DataFrame(columns=['match_id', 'candidate_a_index', 'candidate_b_index', 'status',
                                     'failure_reason', 'failure_kind', 'failure_type', 'failure_phase',
                                     'native_requested', 'session_index', 'evidence_path']))
        _table(paths['07_model_output'] / 'native_blocked_pairs.parquet', blocked_rows if blocked_rows else
               pd.DataFrame(columns=['match_id', 'candidate_a_index', 'candidate_b_index', 'reason', 'conflict_source_ids',
                                     'cutoff_day', 'native_requested']))
        new_pairs = [r for r in pair_rows if r.get('execution_scope') != 'PARENT_INHERITED']
        new_blocked = [r for r in blocked_rows if r.get('execution_scope') != 'PARENT_INHERITED']
        requested = [r for r in new_pairs if r['native_requested']]
        _json(paths['08_reporting'] / 'native_execution.json', {'created_at_utc': _now(),
            'source_parse_count_preparation': 1, 'native_worker_sessions': len(worker_sessions),
            'native_worker_recycles': max(0, len(worker_sessions) - 1),
            'worker_pids': [s['pid'] for s in worker_sessions], 'worker_sessions': worker_sessions,
            'worker_startup_attempts': startup_attempts, 'native_worker_startup_attempts': len(startup_attempts),
            'native_worker_startup_failures': sum(a['status'] == 'FAILED' for a in startup_attempts),
            'state_advance_origin_count': len(confirmed_advances),
            'state_advance_origin_attempt_count': len(cutoff_advances),
            'state_advance_origin_unconfirmed_count': len(cutoff_advances - confirmed_advances),
            'state_advance_origin_count_semantics': 'verified evidence confirms advancement; failed-only origins remain unconfirmed',
            'native_request_count': len(requested), 'requests_by_series': dict(Counter(r['match_id'] for r in requested)),
            'native_valid_pairs': sum(r['status'] == 'AVAILABLE' for r in new_pairs),
            'native_failed_pairs': sum(r['status'] != 'AVAILABLE' for r in new_pairs),
            'native_blocked_pairs': len(new_blocked),
            'worker_closed': True, 'statuses': statuses,
            'raw_snapshots_persisted': sum(r.get('evidence_path') is not None for r in new_pairs),
            'native_nonmutation': 'frozen worker validates input digests before/after every query',
            'native_swap': 'NOT_RUN_PRODUCTION; independent oracle owns public swap checks',
            **({'new_native_request_count': len(requested),
                'inherited_request_count': inherited['execution']['native_request_count'],
                'new_source_parse_count_preparation': 0,
                'inherited_execution': inherited['execution'],
                'no_retry_native_match_ids': sorted(no_retry)} if inherited else {})})
    del pair_rows, blocked_rows, new_pairs, new_blocked, requested
    import gc; gc.collect()
    return captured, statuses


def _series_scores(series, law_rows):
    result = []
    scores = {}
    for law in ('conditional', 'global'):
        rows = [{'match_id': r['match_id'], 'side': r['side'], 'label': r['label'],
                 'q': [r['q_latest'], r['q_previous'], r['q_other']]} for r in law_rows if r['law'] == law and r['label'] is not None]
        scores[law] = {r['match_id']: r for r in score_side_laws(rows)}
    for row in series:
        c, g = scores['conditional'].get(row['match_id']), scores['global'].get(row['match_id'])
        if (c is None) != (g is None):
            raise ValueError('Arms changed primary series denominator')
        result.append({**row, 'eligible_side_count': c['scored_side_count'] if c else 0,
            'conditional_nll': c['lineup_nll'] if c else None, 'global_nll': g['lineup_nll'] if g else None,
            'conditional_brier': c['other_brier'] if c else None, 'global_brier': g['other_brier'] if g else None,
            'absence': None if c else 'NO_ELIGIBLE_SIDE_LABEL'})
    return result


def _bounds_rows(series, opportunities, models, captured, statuses):
    sides = {(r['match_id'], r['side']): r for r in opportunities}
    output = []
    for row in series:
        if row['partition'] != 'ASSESS':
            continue
        mid = row['match_id']
        side_rows = [sides[mid, s] for s in ('A', 'B')]
        states = [json.loads(r['state']) for r in side_rows]
        for law in ('conditional', 'global', 'latest_point'):
            base = {'match_id': mid, 'law': law, 'date': row['date'], 'month': row['month'],
                    'best_of': row['best_of'], 'provisional_independence': True, 'algebraic_not_confidence_interval': True}
            base.update(competition_context=row['competition_context'], major=row['major'],
                player_source_unseen=row['player_source_unseen'],
                candidate_pattern='/'.join(str(len(s['candidates'])) for s in states),
                history_pattern='/'.join(r['history_state'] for r in side_rows),
                incomplete_pattern='/'.join(str(r['incomplete_history_count']) for r in side_rows))
            absence = statuses.get(mid) or next((s['absence'] for s in states if s['absence']), None)
            if absence:
                output.append({**base, 'absence': absence, 'lower': None, 'upper': None,
                    'missing_mass': None, 'supported_mass': None, 'prob_a': None, 'pair_weights': None,
                    'scalar_absence': 'NATIVE_PAIR_UNAVAILABLE'})
                continue
            if law == 'latest_point':
                q = np.asarray([[1., 0., 0.], [1., 0., 0.]])
            else:
                q = models[law].predict(np.asarray([r['counts'] for r in side_rows]),
                    np.asarray([r['ages'] for r in side_rows]),
                    np.asarray([r['previous_available'] for r in side_rows], dtype=bool))
            bound = supported_roster_bounds(q[0], q[1], captured[mid], best_of=row['best_of'])
            bound['pair_weights'] = json.dumps({f'{i},{j}': w for (i, j), w in bound['pair_weights'].items()}, sort_keys=True)
            output.append({**base, **bound, 'absence': None,
                'interpretation': 'exact latest-roster conditional native control, not unconditional lineage forecast' if law == 'latest_point' else 'uncertain historical lineup law'})
    return output


def _contrasts(paths, series_scores):
    from src.analysis.sports_reliability_common import contrast_draw_arrays
    assessment = [r for r in series_scores if r['partition'] == 'ASSESS']
    labels = np.asarray([r['month'] for r in assessment])
    mask = np.asarray([r['eligible_side_count'] > 0 for r in assessment])
    proxies = np.asarray([r['date'][:4] + ':' + (r['tournament_source_name'] or 'UNKNOWN') for r in assessment])
    proxy_universe = sorted(set(proxies.tolist()))
    arrays = {'match_ids': np.asarray([r['match_id'] for r in assessment]), 'months': labels,
              'eligible': mask, 'proxy_blocks': proxies, 'proxy_universe': np.asarray(proxy_universe)}
    metrics, proxy_metrics = {}, {}
    for metric in ('nll', 'brier'):
        c = np.asarray([r['conditional_' + metric] if r['conditional_' + metric] is not None else np.nan for r in assessment])
        g = np.asarray([r['global_' + metric] if r['global_' + metric] is not None else np.nan for r in assessment])
        delta = c - g
        arrays['conditional_' + metric], arrays['global_' + metric], arrays['delta_' + metric] = c, g, delta
        metrics[metric] = paired_block_contrast(delta, labels, mask=mask, universe=MONTHS)
        proxy_metrics[metric] = paired_block_contrast(delta, proxies, mask=mask, universe=proxy_universe, confidence=.95)
        for name, universe, blocks in (('month', MONTHS, labels), ('proxy', proxy_universe, proxies)):
            raw = contrast_draw_arrays(delta, blocks, mask=mask, universe=universe)
            arrays.setdefault(name + '_draw_indices', raw['draw_indices'])
            for key, value in raw.items():
                if isinstance(value, np.ndarray):
                    arrays[name + '_' + metric + '_' + key] = value
    # Fixed prespecified support diagnostics; no new formal superiority slots.
    subgroup = {}
    groups = {'BO1': np.asarray([r['best_of'] == 1 for r in assessment]),
              'BO3': np.asarray([r['best_of'] == 3 for r in assessment]),
              'BO5': np.asarray([r['best_of'] == 5 for r in assessment]),
              'major': np.asarray([r['major'] == 'YES' for r in assessment]),
              'nonmajor': np.asarray([r['major'] == 'NO' for r in assessment]),
              'major_UNKNOWN': np.asarray([r['major'] == 'UNKNOWN' for r in assessment]),
              'source_unseen': np.asarray([r['player_source_unseen'] == 'YES' for r in assessment]),
              'source_seen': np.asarray([r['player_source_unseen'] == 'NO' for r in assessment]),
              'source_UNKNOWN': np.asarray([r['player_source_unseen'] == 'UNKNOWN' for r in assessment]),
              'unilateral': np.asarray([r['eligible_side_count'] == 1 for r in assessment]),
              'bilateral': np.asarray([r['eligible_side_count'] == 2 for r in assessment])}
    for history in ('full', 'partial', 'cold'):
        groups['history_any_' + history] = np.asarray([
            r['history_state_a'] == history or r['history_state_b'] == history for r in assessment])
    for pattern in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        groups['candidates_' + '_'.join(map(str, pattern))] = np.asarray([
            tuple(sorted((r['candidate_count_a'], r['candidate_count_b']))) == pattern for r in assessment])
    groups['incomplete_history'] = np.asarray([
        r['incomplete_history_a'] > 0 or r['incomplete_history_b'] > 0 for r in assessment])
    for name, group in groups.items():
        arrays['membership_' + name] = group
        subgroup[name] = {m: paired_block_contrast(arrays['delta_' + m], labels,
            mask=mask & group, universe=MONTHS, min_rows=200, min_blocks=6) for m in ('nll', 'brier')}
    np.savez_compressed(paths['08_reporting'] / 'contrasts.npz', **arrays)
    return metrics, proxy_metrics, subgroup


def _run_study(run_id, *, repo_root=None, research_root=None, _invocation=None):
    repo, research, paths, protocol, binding = _frozen(run_id, repo_root, research_root)
    if 'recovery' in protocol:
        raise ValueError('Evaluation-only recovery has no TRAIN fit slots')
    support = _load(paths['08_reporting'] / 'support.json')
    if support['status'] != 'READY_FOR_TWO_TRAIN_FITS':
        return _load(paths['08_reporting'] / 'decision.json')
    for path, digest in support['prepared_hashes'].items():
        if _sha(path) != digest:
            raise ValueError('Prepared causal ledger changed before fit')
    _command_start(paths, 'run', {} if _invocation is None else _invocation,
                   {'created_at_utc': _now(), 'restart_allowed': False})
    opportunities = _eligible_rows(pd.read_parquet(paths['04_feature'] / 'opportunities.parquet'))
    series = pd.read_parquet(paths['04_feature'] / 'series.parquet').to_dict('records')
    train = [r for r in opportunities if r['partition'] == 'TRAIN' and r['label'] is not None]
    operations, models = [], {}
    try:
        for arm, conditional in (('conditional', True), ('global', False)):
            operation = {'operation_id': 'R1_TRAIN_' + arm, 'study': 'R1', 'kind': 'bounded_TRAIN_lineup_law',
                'arm': arm, 'seed': None, 'partition': 'TRAIN', 'authorized_ceiling': 2,
                'counter_before': len(operations), 'counter_after': len(operations) + 1,
                'started_at_utc': _now(), 'finished_at_utc': None, 'status': 'ATTEMPTED',
                'parameters_sha256': None,
                'input_ids_sha256': hashlib.sha256(json.dumps([[r['match_id'], r['side']] for r in train]).encode()).hexdigest(),
                'iterations': None, 'function_evaluations': None, 'failure_reason': None}
            operations.append(operation)
            # Immutable attempt is durable BEFORE any optimizer invocation.
            _json(paths['08_reporting'] / (arm + '_attempt.json'), operation)
            artifact = fit_lineup_law(train, conditional=conditional)
            artifact.update(frozen_input_hashes=protocol['hashes'], code_hashes=protocol['import_origins'],
                            protocol_sha256=binding['protocol_sha256'])
            model_path = paths['06_models'] / (arm + '.json')
            _json(model_path, artifact)
            operation.update(finished_at_utc=_now(), status='COMPLETED' if artifact['accepted'] else 'INVALID_FIT',
                parameters_sha256=_sha(model_path), iterations=artifact['iterations'],
                function_evaluations=artifact['function_evaluations'],
                failure_reason=None if artifact['accepted'] else 'FIXED_OPTIMIZER_ACCEPTANCE_FAILED')
            _json(paths['08_reporting'] / (arm + '_result.json'), operation)
            if not artifact['accepted']:
                _json(paths['08_reporting'] / 'operation_trace.json', operations)
                decision = _decision(paths, protocol, 'INVALID_FIT', operations, ['FIXED_OPTIMIZER_ACCEPTANCE_FAILED'],
                                     not_run='NOT_RUN_FIT_FAILED')
                _oracle_index(repo, paths, protocol, decision['status'], operations)
                return decision
            models[arm] = LineupLaw(artifact)
    except BaseException as error:
        if operations and operations[-1]['status'] == 'ATTEMPTED':
            operations[-1].update(status='FAILED', finished_at_utc=_now(), failure_reason=type(error).__name__ + ': ' + str(error))
        if not (paths['08_reporting'] / 'operation_trace.json').exists():
            _json(paths['08_reporting'] / 'operation_trace.json', operations)
        raise
    _json(paths['08_reporting'] / 'operation_trace.json', operations)
    laws = []
    preassessment = [r for r in opportunities if r['partition'] != 'ASSESS']
    for arm, model in models.items():
        laws.extend(_law_rows(preassessment, model, arm, _sha(paths['06_models'] / (arm + '.json'))))
    cal = [r for r in laws if r['partition'] == 'CAL']
    cal_classes = Counter(r['label'] for r in cal if r['law'] == 'conditional' and r['label'] is not None)
    cal_pass = all(cal_classes[c] >= 1 for c in CLASS_ORDER)
    _json(paths['08_reporting'] / 'cal_audit.json', {'created_at_utc': _now(), 'audit_number': 1,
        'class_counts': dict(cal_classes), 'class_support_valid': cal_pass, 'formal_fits': 0,
        'diagnostics': _diagnostics(cal, opportunities, []), 'identity_release_valid': True,
        'status': 'PASS_CLASS_PRESENCE_NOT_POWER_CLAIM' if cal_pass else 'UNSUPPORTED_CAL_CLASSES'})
    if not cal_pass:
        _table(paths['07_model_output'] / 'side_laws.parquet', laws)
        decision = _decision(paths, protocol, 'INCONCLUSIVE', operations, ['UNSUPPORTED_CAL_CLASSES'], not_run='NOT_RUN_BLOCKED')
        _oracle_index(repo, paths, protocol, decision['status'], operations)
        return decision
    assessment_opportunities = [r for r in opportunities if r['partition'] == 'ASSESS']
    for arm, model in models.items():
        laws.extend(_law_rows(assessment_opportunities, model, arm, _sha(paths['06_models'] / (arm + '.json'))))
    _table(paths['07_model_output'] / 'side_laws.parquet', laws)
    scores = _series_scores(series, laws)
    _table(paths['07_model_output'] / 'series_scores.parquet', scores)
    del laws, scores
    import gc; gc.collect()
    return _finish_evaluation(repo, paths, protocol, operations, models, series, opportunities)


def _finish_evaluation(repo, paths, protocol, operations, models, series, opportunities, laws=None, scores=None, *, inherited=None):
    """Shared unchanged evaluation tail; never estimates parameters or scores."""
    import gc
    query_state_before = _query_state_hashes(paths, models)
    captured, native_statuses = _capture_pairs(paths, protocol, series, opportunities, inherited=inherited)
    bounds = _bounds_rows(series, opportunities, models, captured, native_statuses)
    del captured
    gc.collect()
    query_state_after = _query_state_hashes(paths, models)
    if query_state_before != query_state_after:
        raise ValueError('Native pair/bounds evaluation mutated candidate state or fitted weights')
    for p in (paths['08_reporting'] / 'query_nonmutation.json',
              paths['07_model_output'] / 'roster_bounds.parquet',
              paths['08_reporting'] / 'contrasts.npz',
              paths['08_reporting'] / 'metrics.json',
              paths['08_reporting'] / 'public_cases.json',
              paths['08_reporting'] / 'saved_output_binding.json'):
        if p.exists():
            p.unlink()
    _json(paths['08_reporting'] / 'query_nonmutation.json', {
        'created_at_utc': _now(), 'before': query_state_before, 'after': query_state_after, 'status': 'UNCHANGED'})
    _table(paths['07_model_output'] / 'roster_bounds.parquet', bounds)
    if scores is None:
        scores = pd.read_parquet(paths['07_model_output'] / 'series_scores.parquet',
                                 filters=[('partition', '==', 'ASSESS')]).to_dict('records')
    primary, proxy, subgroups = _contrasts(paths, scores)
    del scores
    gc.collect()
    diag_cols = ['partition', 'law', 'label', 'q_other', 'q_latest', 'q_previous',
                 'match_id', 'side', 'absence', 'lineup_nll', 'other_brier',
                 'candidates', 'history_ids', 'incomplete_history_count', 'date']
    if laws is None:
        laws = pd.read_parquet(paths['07_model_output'] / 'side_laws.parquet', columns=diag_cols).to_dict('records')
    diagnostics = _diagnostics(laws, opportunities, bounds)
    del laws, bounds
    gc.collect()
    passing = all(r['status'] == 'ESTIMABLE' and r['upper'] is not None and r['upper'] < 0
                  and r['equal_block']['point'] is not None and r['equal_block']['point'] < 0 for r in primary.values())
    estimable = all(r['status'] == 'ESTIMABLE' for r in primary.values())
    status = 'LINEUP_LAW_DEVELOPMENT_SUPPORTED_NOT_WINNER_MODEL' if passing else 'NO_LINEUP_LAW_GAIN' if estimable else 'INCONCLUSIVE'
    metrics = {'schema': 'lineup_law_reliability_metrics_v1', 'created_at_utc': _now(),
        'diagnostics': diagnostics, 'primary': primary, 'source_name_year_proxy': proxy,
        'subgroups': subgroups, 'portfolio_slots': 10, 'reserved_R1_slots': [7, 8],
        'native_status_counts': dict(Counter(v or 'AVAILABLE' for v in native_statuses.values())),
        'winner_scoring': 'NOT_ADVERTISED; no scalar loss on partial mixtures',
        'limitations': LIMITATIONS}
    _json(paths['08_reporting'] / 'metrics.json', metrics)
    public_cases = _load(paths['05_model_input'] / 'public_cases.json')
    _json(paths['08_reporting'] / 'public_cases.json', public_cases)
    _json(paths['08_reporting'] / 'saved_output_binding.json', {
        'created_at_utc': _now(), 'hashes': {str(p): _sha(p) for stage in ('06_models', '07_model_output')
                                          for p in sorted(paths[stage].rglob('*')) if p.is_file()}})
    decision = _decision(paths, protocol, status, operations, [] if passing else ['PRIMARY_GAIN_NOT_ESTABLISHED'])
    _oracle_index(repo, paths, protocol, status, operations, public_cases)
    return decision


def _public_cases(paths, series, opportunities):
    """Input-only lexical selection, <=32 targets, no selection by errors."""
    sides = defaultdict(list)
    for r in opportunities:
        sides[r['match_id']].append(r)
    selected = {}
    for row in sorted((r for r in series if r['partition'] == 'ASSESS'), key=lambda r: r['match_id']):
        states = sides[row['match_id']]
        cells = ['BO' + str(row['best_of'])]
        cells.extend('candidates_' + str(len(json.loads(r['candidates']))) for r in states)
        cells.extend('history_' + r['history_state'] for r in states)
        cells.extend('absence_' + (r['absence'] or 'AVAILABLE') for r in states)
        for cell in cells:
            selected.setdefault(cell, row)
    unique = {}
    for cell, row in sorted(selected.items()):
        unique.setdefault(row['match_id'], {'row': row, 'states': []})['states'].append(cell)
    output = []
    input_sha256 = _sha(paths['04_feature'] / 'opportunities.parquet')
    for mid, entry in sorted(unique.items())[:32]:
        row = entry['row']
        output.append({'case_id': 'R1_' + mid, 'study': 'R1', 'arm': ['conditional', 'global'],
            'seed_or_ensemble': None, 'match_id': mid,
            'input_reference_and_sha256': {'path': str(paths['04_feature'] / 'opportunities.parquet'),
                                         'sha256': input_sha256},
            'target_origin': row['date'], 'scenario': row['scenario'], 'contract_state': entry['states'],
            'expected_status': 'SAVED_CAUSAL_LAW_OR_EXPLICIT_ABSENCE', 'expected_reason': None,
            'public_result_reference': 'predict_run', 'independent_expected_reference': 'independent source and saved theta replay',
            'comparison_kind': 'laws/candidates/captures/bounds', 'tolerances': {'law': 1e-12, 'native': 2e-6},
            'state_hash_before': input_sha256,
            'state_hash_after': input_sha256})
    return output


def _public_case_contracts(paths, cases):
    """Disclose absent input states and existing synthetic behavioral cases."""
    required = ('BO1', 'BO3', 'BO5', 'candidates_0', 'candidates_1', 'candidates_2',
                'history_cold', 'history_partial', 'history_full', 'absence_AVAILABLE',
                'absence_MISSING_EXACT_CONTEXT', 'absence_NO_EXACT_CONTEXT_HISTORY',
                'absence_INCOMPLETE_LATEST_APPEARANCE')
    references = {
        'candidates_0': 'test_strict_prior_release_team_and_exact_context_cannot_be_swapped',
        'candidates_1': 'test_previous_and_third_distinct_roster_labels_and_single_candidate_mask',
        'candidates_2': 'test_last_ten_are_series_not_unique_rosters_and_repetitions_count',
        'history_cold': 'test_strict_prior_release_team_and_exact_context_cannot_be_swapped',
        'history_partial': 'test_older_incomplete_appearance_occupies_window_but_is_not_an_exact_candidate',
        'history_full': 'test_last_ten_are_series_not_unique_rosters_and_repetitions_count',
        'absence_AVAILABLE': 'test_final_map_is_one_series_appearance_but_target_label_is_first_map',
        'absence_MISSING_EXACT_CONTEXT': 'test_strict_prior_release_team_and_exact_context_cannot_be_swapped',
        'absence_NO_EXACT_CONTEXT_HISTORY': 'test_strict_prior_release_team_and_exact_context_cannot_be_swapped',
        'absence_INCOMPLETE_LATEST_APPEARANCE': 'test_latest_incomplete_appearance_withholds_instead_of_resurrecting_older_five'}
    observed = {state: case['case_id'] for case in reversed(cases) for state in case['contract_state']}
    fixture = Path(__file__).resolve().parents[2] / 'betting_app/tests/test_lineup_law_reliability.py'
    return {'schema': 'lineup_law_public_case_contracts_v1', 'created_at_utc': _now(),
        'selection': 'Input-only lexical ASSESS target per prespecified state; at most32 real targets',
        'real_cases_sha256': _sha(paths['05_model_input'] / 'public_cases.json'),
        'coverage': {state: {'status': 'AVAILABLE' if state in observed else 'ABSENT_IN_SOURCE',
            'case_id': observed.get(state),
            'synthetic_contract_reference': None if state in observed else (
                'bounds_format_inputs.best_of=' + state[2:] if state.startswith('BO') else references[state])}
                     for state in required},
        'synthetic_fixture_source': {'path': str(fixture), 'sha256': _sha(fixture),
            'status': 'INPUT_ONLY_NOT_EXECUTED', 'observed_population_support': False},
        'bounds_format_inputs': {'best_of': [1, 3, 5], 'q_a': [.5, .25, .25], 'q_b': [.5, .25, .25],
            'supported_native_probabilities': {'0,0': .75, '0,1': .75, '1,0': .75, '1,1': .75},
            'reversed_probabilities': {'0,0': .25, '0,1': .25, '1,0': .25, '1,1': .25},
            'scenario': 'EXPLICIT_SYNTHETIC_ALGEBRA_INPUT_ONLY_NOT_NATIVE_INFERENCE'},
        'additional_contract_references': [
            'test_previous_and_third_distinct_roster_labels_and_single_candidate_mask',
            'test_target_identity_is_excluded_even_with_erroneous_prior_release',
            'test_raw_source_corruption_fails_closed',
            'test_overlapping_native_pair_and_missing_target_context_cannot_be_inferred',
            'test_evidenced_complete_point_mass_retains_native_probability_without_shrinkage',
            'test_tiny_positive_other_mass_never_becomes_point_five_or_scalar',
            'test_side_reversal_complements_bounds_and_preserves_missing_mass'],
        'native_tensor_contract': 'Real predict_run captures with exact encoded full-mode wire; parent-owned native replay and swap only',
        'scope': 'Synthetic inputs do not imply any observed state, fitted-model verification, native output, or executed test'}


def replay_capture(path, *, config_path=None, expected_sha256=None):
    """Replay exact dtype-preserving saved inputs; never new source or fitting."""
    from betting_app.core.models.native_c0 import NativeC0Adapter
    from betting_app.services.c0_inference import C0InferenceClient
    if expected_sha256 is None:
        table = Path(path).resolve().parent.parent / 'native_pairs.parquet'
        if not table.is_file():
            raise ValueError('Public replay requires pinned native-pair evidence')
        records = pd.read_parquet(table, columns=['evidence_path', 'evidence_sha256'])
        selected = records.loc[records.evidence_path == str(Path(path).resolve())]
        if len(selected) != 1:
            raise ValueError('Capture is not a unique pinned native-pair case')
        expected_sha256 = selected.iloc[0]['evidence_sha256']
    if expected_sha256 is not None and _sha(path) != expected_sha256:
        raise ValueError('Captured native evidence hash mismatch')
    capture = _load(path)
    pinned_config = capture['source_pins']['native_config']
    config_path = Path(config_path) if config_path is not None else Path(pinned_config['path'])
    if _sha(config_path) != pinned_config['sha256']:
        raise ValueError('Native public replay configuration differs from captured identity')
    validator = object.__new__(NativeC0Adapter)
    request = validator._validate_request(capture['request'])
    target, raw, hist = validator._decode_snapshot(capture['encoded_snapshot'], request)
    before = validator._array_digest({**raw, **hist})
    if before != capture['prediction']['diagnostics']['input_hashes']:
        raise ValueError('Captured native tensor hashes differ from original evidence')
    client = C0InferenceClient(config_path)
    worker_environment = {'C0_RESEARCH_ROOT': capture['research_root'], 'OMP_NUM_THREADS': '4',
                          'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': ''}
    previous_environment = {name: os.environ.get(name) for name in worker_environment}
    os.environ.update(worker_environment)
    try:
        prediction = client.predict_snapshot({'target': target, 'raw': raw, 'hist': hist})
    finally:
        client.close()
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    if validator._array_digest({**raw, **hist}) != before:
        raise ValueError('Public replay mutated captured native tensors')
    return {'prediction': prediction, 'original_native_history_provenance': capture['native_history_provenance'],
            'original_capture_sha256': _sha(path)}


def predict_run(run_id, match_ids, *, law='conditional', repo_root=None, research_root=None):
    """Public saved forecast needs no target label and never invokes a fit."""
    if law not in ('conditional', 'global'):
        raise ValueError('Law must be conditional or global')
    repo, research, paths, protocol, binding = _frozen(run_id, repo_root, research_root)
    output_binding = _load(paths['08_reporting'] / 'saved_output_binding.json')
    for path, digest in output_binding['hashes'].items():
        if _sha(path) != digest:
            raise ValueError('Saved fitted output/capture changed ' + path)
    support = _load(paths['08_reporting'] / 'support.json')
    for path, digest in support['prepared_hashes'].items():
        if _sha(path) != digest:
            raise ValueError('Prepared causal inputs changed ' + path)
    model_path = paths['06_models'] / (law + '.json')
    model = LineupLaw.load(model_path, sha256=output_binding['hashes'][str(model_path)])
    # Read only candidate feature/state columns; actual lineup labels are not a prediction prerequisite.
    fields = ['match_id', 'side', 'state', 'counts', 'ages', 'previous_available']
    opportunities = pd.read_parquet(paths['04_feature'] / 'opportunities.parquet', columns=fields)
    pairs = pd.read_parquet(paths['07_model_output'] / 'native_pairs.parquet')
    bounds_frame = pd.read_parquet(paths['07_model_output'] / 'roster_bounds.parquet')
    output = []
    for value in match_ids:
        mid = _id(value, 'requested match ID')
        rows = opportunities.loc[opportunities.match_id == mid].sort_values('side').to_dict('records')
        if len(rows) != 2:
            raise ValueError('Unknown prepared canonical match ID ' + mid)
        laws = []
        for row in rows:
            state = json.loads(row['state'])
            q = None if state['absence'] else model.predict(np.asarray([row['counts']]), np.asarray([row['ages']]),
                                                           np.asarray([row['previous_available']], dtype=bool))[0].tolist()
            laws.append({'side': row['side'], 'state': state, 'q': q, 'absence': state['absence']})
        matrix, evidence_paths = {}, []
        for row in pairs.loc[pairs.match_id == mid].to_dict('records'):
            if row['status'] != 'AVAILABLE':
                continue
            capture = _load(row['evidence_path'])
            if _sha(row['evidence_path']) != row['evidence_sha256']:
                raise ValueError('Pinned native capture changed')
            key = (int(row['candidate_a_index']), int(row['candidate_b_index']))
            matrix[key] = _saved_result(capture['prediction'])
            evidence_paths.append({'pair': list(key), 'path': row['evidence_path'], 'sha256': row['evidence_sha256'],
                                   'encoded_snapshot': capture['encoded_snapshot'],
                                   'native_history_provenance': capture['native_history_provenance']})
        saved = bounds_frame.loc[(bounds_frame.match_id == mid) & (bounds_frame.law == law)]
        if not len(saved):
            bound = {'absence': 'NOT_ASSESS_NATIVE_CASE', 'prob_a': None, 'scalar_absence': 'NATIVE_PAIR_UNAVAILABLE'}
        elif saved.iloc[0]['absence'] is not None and not pd.isna(saved.iloc[0]['absence']):
            bound = {'absence': saved.iloc[0]['absence'], 'prob_a': None, 'scalar_absence': 'NATIVE_PAIR_UNAVAILABLE'}
        else:
            best_of = int(saved.iloc[0]['best_of'])
            bound = supported_roster_bounds(laws[0]['q'], laws[1]['q'], matrix, best_of=best_of)
            bound['pair_weights'] = {f'{a},{b}': w for (a, b), w in bound['pair_weights'].items()}
        output.append({'match_id': mid, 'law': law, 'side_laws': laws, 'supported_native_pairs': evidence_paths,
                       'bounds': bound, 'scope': 'HISTORICAL_DEVELOPMENT_ONLY', 'limitations': LIMITATIONS})
    return output


def _opportunity_inventory(opportunities):
    """Denominators by fixed input/missingness cells, including unavailable sides."""
    cells = {}
    dimensions = ('partition', 'month', 'best_of', 'competition_context', 'major',
                  'player_source_unseen', 'history_state', 'incomplete_history_count',
                  'previous_available', 'absence', 'label_absence')
    for dimension in dimensions:
        grouped = defaultdict(list)
        for row in opportunities:
            grouped[str(row[dimension]) if row.get(dimension) is not None else 'UNKNOWN_OR_NONE'].append(row)
        cells[dimension] = {key: {'opportunity_sides': len(rows),
            'parent_series': len({r['match_id'] for r in rows}),
            'eligible_sides': sum(r['label'] is not None for r in rows),
            'eligible_series': len({r['match_id'] for r in rows if r['label'] is not None}),
            'class_counts': dict(Counter(r['label'] for r in rows if r['label'] is not None)),
            'all_nine_month_counts': {m: sum(r['month'] == m for r in rows) for m in MONTHS},
            'candidate_recall': float(np.mean([r['label'] != 'OTHER' for r in rows if r['label'] is not None]))
                                if any(r['label'] is not None for r in rows) else None}
            for key, rows in sorted(grouped.items())}
    return {'schema': 'lineup_law_opportunity_inventory_v1', 'created_at_utc': _now(),
            'opportunity_sides': len(opportunities), 'cells': cells,
            'one_two_absent_candidates': dict(Counter(len(json.loads(r['candidates'])) for r in opportunities)),
            'window_appearance_counts': dict(Counter(len(r['history_ids']) for r in opportunities)),
            'latest_incomplete_withheld': sum(r['absence'] == 'INCOMPLETE_LATEST_APPEARANCE' for r in opportunities)}


def _failure_kind(error):
    from betting_app.services.c0_inference import C0InferenceError
    if isinstance(error, SourceValidationError):
        return 'SOURCE_VALIDATION'
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return 'INTERRUPTED'
    if isinstance(error, (C0InferenceError, NativeIntegrityError)):
        return 'NATIVE_INTEGRITY'
    if isinstance(error, OSError):
        return 'IO'
    return 'SOFTWARE'


def _failed_command(run_id, command, error, repo_root=None, research_root=None):
    """Close a scientific hard stop without authorization to retry any command."""
    repo, research, paths = _roots(run_id, repo_root, research_root)
    protocol = _load(paths['05_model_input'] / 'protocol.json')
    operations = []
    for arm in ('conditional', 'global'):
        result = paths['08_reporting'] / (arm + '_result.json')
        attempt = paths['08_reporting'] / (arm + '_attempt.json')
        if result.exists():
            operations.append(_load(result))
        elif attempt.exists():
            record = _load(attempt)
            record.update(status='INTERRUPTED_OR_FAILED', finished_at_utc=_now(),
                          failure_reason=type(error).__name__ + ': ' + str(error))
            operations.append(record)
    trace = paths['08_reporting'] / 'operation_trace.json'
    if trace.exists():
        operations = _load(trace)
    else:
        if 'recovery' in protocol:
            operations = _load(protocol['recovery']['parent_decision']['path'])['operation_trace']
        _json(trace, operations)
    kind = _failure_kind(error)
    source_defect = kind == 'SOURCE_VALIDATION'
    status = 'BLOCKED_IDENTITY_RELEASE' if source_defect else 'INCONCLUSIVE'
    support_path = paths['08_reporting'] / 'support.json'
    support = _load(support_path) if support_path.exists() else None
    validation = ('INVALID' if source_defect else 'VALID'
                  if support is not None and support.get('counts', {}).get('identity_release_valid') is True
                  else 'NOT_ESTABLISHED')
    _json(paths['08_reporting'] / (command + '_hard_stop.json'), {
        'created_at_utc': _now(), 'command': command, 'status': status,
        'failure_kind': kind, 'failure_type': type(error).__name__, 'failure_reason': str(error),
        'identity_release_validation_status': validation,
        'formal_fits': len(operations), 'completed_fits': sum(o.get('status') == 'COMPLETED' for o in operations),
        'unused_TRAIN_slots': 2 - len(operations), 'no_restart': True,
        **({'inherited_TRAIN_fits': 2, 'new_fit_counts': protocol['recovery']['new_fit_budget']}
           if 'recovery' in protocol else {})})
    if support is None:
        closure = {'schema': 'lineup_law_reliability_support_v1', 'created_at_utc': _now(),
            'status': status, 'formal_fits': 0, 'counts': {'identity_release_valid': False if source_defect else None},
            'identity_release_validation_status': validation, 'failure_kind': kind,
            'failed_floors': [], 'counts_unavailable_reason': 'IDENTITY_RELEASE_SOURCE_HARD_STOP' if source_defect else 'EXECUTION_HARD_STOP'}
        if source_defect:
            closure.update(support_gate(closure['counts']), source_backed_failure=str(error))
        _json(support_path, closure)
    if not (paths['08_reporting'] / 'decision.json').exists():
        _decision(paths, protocol, status, operations, [str(error)], not_run='NOT_RUN_BLOCKED' if source_defect else 'NOT_RUN_EXECUTION_FAILURE')
    if not (paths['08_reporting'] / 'oracle_inputs.json').exists():
        _oracle_index(repo, paths, protocol, status, operations)


def prepare_run(run_id, *, repo_root=None, research_root=None):
    invocation = {}
    try:
        return _prepare_run(run_id, repo_root=repo_root, research_root=research_root, _invocation=invocation)
    except BaseException as error:
        if invocation.get('started'):
            _failed_command(run_id, 'prepare', error, repo_root, research_root)
        raise


def run_study(run_id, *, repo_root=None, research_root=None):
    invocation = {}
    try:
        return _run_study(run_id, repo_root=repo_root, research_root=research_root, _invocation=invocation)
    except BaseException as error:
        if invocation.get('started'):
            _failed_command(run_id, 'run', error, repo_root, research_root)
        raise


def _import_closure(repo, seed_modules):
    """Static local import closure without importing native/model archives."""
    pending, result = list(seed_modules), {}
    for name in seed_modules:
        parts = name.split('.')
        pending.extend('.'.join(parts[:i]) for i in range(1, len(parts)))
    while pending:
        module = pending.pop()
        if module in result:
            continue
        relative = Path(*module.split('.'))
        candidates = (repo / relative.with_suffix('.py'), repo / relative / '__init__.py')
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        if module in seed_modules:
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None or Path(spec.origin).resolve() != path.resolve():
                raise ValueError('Actual import origin differs from frozen repo binding ' + module)
        result[module] = {'path': str(path.resolve()), 'sha256': _sha(path)}
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split('.')
                    prefix = '.'.join(parts[:len(parts) - node.level + 1])
                    base = '.'.join(p for p in (prefix, node.module) if p)
                else:
                    base = node.module or ''
                names = [base] + [base + '.' + a.name for a in node.names if a.name != '*']
            pending.extend(name for name in names if name.startswith(('src.', 'scripts.', 'betting_app.')))
    if any(module not in result for module in seed_modules):
        raise ValueError('Required local import closure is incomplete')
    return dict(sorted(result.items()))


def _clock_conflict_inventory(clocks, origin_days):
    """Sweep the complete native/canonical comparison once for all origins."""
    import heapq
    intervals = sorted((c['native_release_day'], c['canonical_release_day'], c['source_match_id'])
                       for c in clocks if c['canonical_release_day'] is not None
                       and c['canonical_release_day'] > c['native_release_day'])
    cursor, active, expiry, result = 0, set(), [], {}
    for day in origin_days:
        while cursor < len(intervals) and intervals[cursor][0] < day:
            _, canonical, mid = intervals[cursor]
            if canonical >= day:
                active.add(mid)
                heapq.heappush(expiry, (canonical, mid))
            cursor += 1
        while expiry and expiry[0][0] < day:
            _, mid = heapq.heappop(expiry)
            active.remove(mid)
        result[day] = sorted(active, key=_source_series_order)
    return result


def _query_state_hashes(paths, models):
    return {'candidate_ledger_sha256': _sha(paths['04_feature'] / 'opportunities.parquet'),
            'theta_sha256': {law: hashlib.sha256(model.theta.tobytes(order='C')).hexdigest()
                             for law, model in models.items()}}


_RECOVERY_RUNTIME = ('src/analysis/lineup_law_reliability.py',
                     'scripts/train_lineup_law.py', 'scripts/verify_lineup_law_reliability.py')
_ZERO_FITS = {'TRAIN': 0, 'CAL': 0, 'C0': 0, 'ASSESS': 0}


def _recovery_authorization(repo, path, run_id, parent_run_id):
    """An exact archived exception set, not permission to repin scientific code."""
    path = Path(path).expanduser().resolve()
    authorization = _load(path)
    if (authorization.get('schema') != 'lineup_law_technical_recovery_authorization_v1'
            or authorization.get('run_id') != run_id or authorization.get('parent_run_id') != parent_run_id
            or run_id == parent_run_id or authorization.get('inherited_TRAIN_fits') != 2
            or authorization.get('new_fit_budget') != _ZERO_FITS
            or set(authorization.get('allowed_source_changes', [])) != set(_RECOVERY_RUNTIME)
            or set(authorization.get('old_runtime_snapshots', {})) != set(_RECOVERY_RUNTIME)
            or authorization.get('scientific_contract_changed') is not False
            or authorization.get('extra_recipe_or_hyperparameter_choice') is not False
            or authorization.get('new_lineup_assessment_metrics_or_intervals_read') is not False):
        raise ValueError('Invalid explicit zero-fit recovery authorization')
    for relative, digest in authorization['parent_file_pins'].items():
        if _sha(repo / relative) != digest:
            raise ValueError('Authorized parent artifact changed: ' + relative)
    for relative, snapshot in authorization['old_runtime_snapshots'].items():
        if _sha(repo / snapshot['snapshot']) != snapshot['sha256']:
            raise ValueError('Archived failed runtime changed: ' + relative)
    for relative, digest in authorization['protected_before'].items():
        if _sha(repo / relative) != digest:
            raise ValueError('Protected recovery artifact changed: ' + relative)
    return authorization


def _validate_parent_artifact_index(repo, parent_paths, authorization):
    path = parent_paths['08_reporting'] / 'oracle_inputs.json'
    if authorization['parent_file_pins'].get(str(path.relative_to(repo))) != _sha(path):
        raise ValueError('Parent oracle index lacks exact authorization binding')
    descriptors = _load(path)['artifact_descriptors']
    for item in descriptors:
        artifact = (repo / item['repo_relative_path']).resolve()
        if item['absence_status'] == 'AVAILABLE' and any(
                artifact.is_relative_to(directory) for directory in parent_paths.values()):
            if _sha(artifact) != item['sha256']:
                raise ValueError('Parent indexed artifact changed: ' + str(artifact))
    blocked = parent_paths['07_model_output'] / 'native_blocked_pairs.parquet'
    if not any(item['repo_relative_path'] == str(blocked.relative_to(repo))
               and item['sha256'] == _sha(blocked) for item in descriptors):
        raise ValueError('Inherited blocked native evidence lacks an immutable parent index pin')


def _validate_saved_recovery(repo, paths, protocol):
    descriptor = _load(paths['05_model_input'] / 'recovery.json')
    if descriptor != protocol['recovery'] or descriptor['run_id'] != protocol['run_id']:
        raise ValueError('Recovery descriptor/protocol binding differs')
    pin = descriptor['authorization']
    if _sha(pin['path']) != pin['sha256']:
        raise ValueError('Recovery authorization changed')
    authorization = _recovery_authorization(repo, pin['path'], descriptor['run_id'], descriptor['parent_run_id'])
    _validate_parent_artifact_index(repo, _roots(descriptor['parent_run_id'], repo, protocol['research_root'])[2], authorization)
    if (descriptor['archived_runtime'] != authorization['old_runtime_snapshots']
            or descriptor['new_fit_budget'] != _ZERO_FITS
            or descriptor['no_retry_native_match_ids'] != authorization['no_retry_native_match_ids']):
        raise ValueError('Recovery archived-runtime/budget/no-retry lineage changed')
    for pin in (descriptor['parent_protocol'], descriptor['parent_binding'],
                descriptor['parent_hard_stop'], descriptor['parent_decision']):
        if _sha(pin['path']) != pin['sha256']:
            raise ValueError('Recovery parent lineage changed')
    for path, pin in descriptor['inherited_artifacts'].items():
        if _sha(path) != pin['sha256'] or _sha(pin['path']) != pin['sha256']:
            raise ValueError('Inherited immutable recovery artifact changed: ' + path)
    for pin in descriptor['current_runtime'].values():
        if _sha(pin['path']) != pin['sha256']:
            raise ValueError('Corrected recovery runtime changed')
    if _load(paths['08_reporting'] / 'new_operation_trace.json') != []:
        raise ValueError('Evaluation-only recovery consumed a new fit')


def _recovery_parent(repo, research, parent_paths, authorization):
    """Validate accepted inherited fits and causal lineage without any optimizer."""
    from importlib.metadata import version
    from src.models.lineup_law_reliability import lineup_objective, projected_gradient, rows_to_arrays
    protocol_path = parent_paths['05_model_input'] / 'protocol.json'
    binding_path = parent_paths['05_model_input'] / 'binding.json'
    protocol, binding = _load(protocol_path), _load(binding_path)
    if (protocol['run_id'] != authorization['parent_run_id']
            or protocol['repo_root'] != str(repo) or protocol['research_root'] != str(research)
            or binding['protocol_sha256'] != _sha(protocol_path)
            or binding['pins'] != protocol['hashes'] or binding['import_origins'] != protocol['import_origins']
            or binding['protected_before'] != protocol['protected_before']
            or protocol['protected_before'] != authorization['protected_before']
            or protocol['fit_budget'] != {**_ZERO_FITS, 'TRAIN': 2, 'restarts': 0}
            or any(version(name) != expected for name, expected in protocol['library_versions'].items())):
        raise ValueError('Parent protocol/input/import/budget binding differs')
    for pin in (*binding['pins'].values(), *binding['import_origins'].values()):
        path = Path(pin['path']).resolve()
        relative = str(path.relative_to(repo)) if path.is_relative_to(repo) else None
        archived = authorization['old_runtime_snapshots'].get(relative)
        if archived is not None:
            if pin['sha256'] != archived['sha256']:
                raise ValueError('Parent runtime is not the authorized archived runtime')
        elif _sha(path) != pin['sha256']:
            raise ValueError('Non-authorized scientific/source/import pin changed: ' + str(path))
    stop = _load(parent_paths['08_reporting'] / 'run_hard_stop.json')
    decision = _load(parent_paths['08_reporting'] / 'decision.json')
    if (stop.get('no_restart') is not True or stop.get('failure_kind') != 'SOFTWARE'
            or stop.get('failure_type') != 'AttributeError'
            or stop.get('failure_reason') != "'C0EvidenceResult' object has no attribute 'received_at'"
            or stop.get('identity_release_validation_status') != 'VALID'
            or stop.get('formal_fits') != 2 or stop.get('completed_fits') != 2
            or stop.get('status') != 'INCONCLUSIVE' or decision.get('status') != 'INCONCLUSIVE'
            or (parent_paths['08_reporting'] / 'metrics.json').exists()
            or (parent_paths['08_reporting'] / 'contrasts.npz').exists()):
        raise ValueError('Parent is not the authorized pre-metrics receipt-field hard stop')
    support = _load(parent_paths['08_reporting'] / 'support.json')
    approval = _load(parent_paths['05_model_input'] / 'prefit_approval.json')
    if (support['status'] != 'READY_FOR_TWO_TRAIN_FITS'
            or support['protocol_sha256'] != binding['protocol_sha256']
            or support['frozen_input_hashes'] != protocol['hashes']
            or approval['status'] != 'APPROVED' or approval['run_id'] != protocol['run_id']
            or approval['protocol_sha256'] != _sha(protocol_path)
            or approval['binding_sha256'] != _sha(binding_path)
            or approval['support_sha256'] != _sha(parent_paths['08_reporting'] / 'support.json')
            or approval['fit_budget'] != protocol['fit_budget']
            or approval['counts'] != support['counts']):
        raise ValueError('Parent preparation/support/approval binding differs')
    prepare_oracle = approval['independent_prepare_oracle']
    if (_sha(repo / prepare_oracle['path']) != prepare_oracle['sha256']
            or prepare_oracle['status'] != 'PASS_PREPARE_CONTRACT'):
        raise ValueError('Parent independent preparation approval changed')
    for path, digest in support['prepared_hashes'].items():
        if _sha(path) != digest:
            raise ValueError('Parent prepared causal ledger changed: ' + path)
    if _load(parent_paths['08_reporting'] / 'cal_audit.json')['class_support_valid'] is not True:
        raise ValueError('Parent CAL class presence did not pass')
    operations = _load(parent_paths['08_reporting'] / 'operation_trace.json')
    if len(operations) != 2 or decision['operation_trace'] != operations:
        raise ValueError('Exactly two inherited accepted fit operations required')
    opportunities = _eligible_rows(pd.read_parquet(parent_paths['04_feature'] / 'opportunities.parquet'))
    train = [row for row in opportunities if row['partition'] == 'TRAIN' and row['label'] is not None]
    arrays = rows_to_arrays(train)
    keys = [[row['match_id'], row['side']] for row in train]
    key_digest = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    models = {}
    for index, arm in enumerate(('conditional', 'global')):
        path = parent_paths['06_models'] / (arm + '.json')
        artifact, operation = _load(path), operations[index]
        model = LineupLaw(artifact)
        objective, gradient = lineup_objective(model.theta, *arrays, conditional=arm == 'conditional')
        residual = float(np.max(np.abs(projected_gradient(model.theta, gradient))))
        if (artifact['conditional'] != (arm == 'conditional') or artifact.get('success') is not True
                or artifact['protocol_sha256'] != binding['protocol_sha256']
                or artifact['frozen_input_hashes'] != protocol['hashes'] or artifact['code_hashes'] != protocol['import_origins']
                or artifact['training_keys'] != keys or artifact['iterations'] > 500
                or not np.isfinite(objective) or not np.isfinite(gradient).all() or residual > 1e-5
                or not np.isclose(objective, artifact['objective'], rtol=1e-12, atol=1e-12)
                or not np.allclose(gradient, artifact['gradient'], rtol=1e-12, atol=1e-12)
                or operation['arm'] != arm or operation['partition'] != 'TRAIN' or operation['status'] != 'COMPLETED'
                or operation['counter_before'] != index or operation['counter_after'] != index + 1
                or operation['parameters_sha256'] != _sha(path) or operation['input_ids_sha256'] != key_digest):
            raise ValueError('Inherited accepted fit/objective/KKT/operation binding differs: ' + arm)
        models[arm] = model
    del train, arrays
    import gc; gc.collect()
    _validate_parent_artifact_index(repo, parent_paths, authorization)
    pairs = pd.read_parquet(parent_paths['07_model_output'] / 'native_pairs.parquet').to_dict('records')
    blocked = pd.read_parquet(parent_paths['07_model_output'] / 'native_blocked_pairs.parquet').to_dict('records')
    execution = _load(parent_paths['08_reporting'] / 'native_execution.json')
    attempted = {row['match_id'] for row in pairs if row['native_requested']}
    if (attempted != set(authorization['no_retry_native_match_ids'])
            or sum(bool(row['native_requested']) for row in pairs) != authorization['original_failed_requests']
            or execution['native_request_count'] != authorization['original_failed_requests']
            or execution['native_valid_pairs'] != 0 or execution['raw_snapshots_persisted'] != 0
            or execution['worker_closed'] is not True or any(row['status'] == 'AVAILABLE' for row in pairs)):
        raise ValueError('Parent native attempt/failure/no-retry binding differs')
    return protocol, binding, operations, models, opportunities, {'pairs': pairs, 'blocked': blocked,
        'execution': execution, 'no_retry_native_match_ids': authorization['no_retry_native_match_ids']}


def recover_evaluation(run_id, *, parent_run_id, authorization_path, repo_root=None, research_root=None):
    """One authorized fresh evaluation-only namespace; exactly zero new fits."""
    repo, research, paths = _roots(run_id, repo_root, research_root)
    _, _, parent_paths = _roots(parent_run_id, repo, research)
    if run_id == parent_run_id or (paths['08_reporting'] / 'decision.json').exists():
        raise FileExistsError('Completed recovery namespace cannot be overwritten')
    authorization = _recovery_authorization(repo, authorization_path, run_id, parent_run_id)
    parent, parent_binding, operations, models, opportunities, inherited = _recovery_parent(
        repo, research, parent_paths, authorization)
    copies = {}
    for path, digest in _load(parent_paths['08_reporting'] / 'support.json')['prepared_hashes'].items():
        source = Path(path)
        stage = next((stage for stage, directory in parent_paths.items() if source.is_relative_to(directory)), None)
        if stage is None:
            raise ValueError('Prepared artifact outside explicit parent namespace')
        copies[str(paths[stage] / source.relative_to(parent_paths[stage]))] = {'path': str(source), 'sha256': digest}
    for stage, names in (('06_models', ('conditional.json', 'global.json')),
                         ('05_model_input', ('prefit_approval.json',)),
                         ('07_model_output', ('side_laws.parquet', 'series_scores.parquet')),
                         ('08_reporting', ('support.json', 'operation_trace.json', 'cal_audit.json', 'opportunity_inventory.json'))):
        for name in names:
            source = parent_paths[stage] / name
            if not source.exists() and name == 'opportunity_inventory.json':
                continue
            relative = str(source.relative_to(repo))
            if authorization['parent_file_pins'].get(relative) != _sha(source):
                raise ValueError('Required inherited artifact lacks exact authorization pin: ' + relative)
            copies[str(paths[stage] / name)] = {'path': str(source), 'sha256': _sha(source)}
    def parent_pin(stage, name):
        path = parent_paths[stage] / name
        if authorization['parent_file_pins'].get(str(path.relative_to(repo))) != _sha(path):
            raise ValueError('Required parent lineage lacks exact authorization pin: ' + name)
        return {'path': str(path), 'sha256': _sha(path)}
    descriptor = {'schema': 'lineup_law_evaluation_recovery_v1', 'run_id': run_id, 'parent_run_id': parent_run_id,
        'authorization': {'path': str(Path(authorization_path).expanduser().resolve()), 'sha256': _sha(authorization_path)},
        'parent_protocol': parent_pin('05_model_input', 'protocol.json'),
        'parent_binding': parent_pin('05_model_input', 'binding.json'),
        'parent_hard_stop': parent_pin('08_reporting', 'run_hard_stop.json'),
        'parent_decision': parent_pin('08_reporting', 'decision.json'),
        'inherited_artifacts': copies, 'archived_runtime': authorization['old_runtime_snapshots'],
        'current_runtime': {relative: {'path': str(repo / relative), 'sha256': _sha(repo / relative)}
                            for relative in _RECOVERY_RUNTIME},
        'inherited_TRAIN_fits': 2, 'new_fit_budget': dict(_ZERO_FITS),
        'no_retry_native_match_ids': authorization['no_retry_native_match_ids']}
    protocol = json.loads(json.dumps(parent))
    protocol.update(run_id=run_id, output_paths={stage: str(path) for stage, path in paths.items()}, recovery=descriptor)
    for group in ('hashes', 'import_origins'):
        for pin in protocol[group].values():
            path = Path(pin['path'])
            relative = str(path.relative_to(repo)) if path.is_relative_to(repo) else None
            if relative in descriptor['current_runtime']:
                pin['sha256'] = descriptor['current_runtime'][relative]['sha256']
    invocation = {}
    try:
        resuming = (paths['05_model_input'] / 'protocol.json').exists() and (paths['05_model_input'] / 'recovery.json').exists()
        if not resuming:
            for path in paths.values():
                path.mkdir(parents=True, exist_ok=True)
            _json(paths['05_model_input'] / 'protocol.json', protocol)
            _json(paths['05_model_input'] / 'recovery.json', descriptor)
            _json(paths['05_model_input'] / 'binding.json', {**parent_binding, 'run_id': run_id,
                'protocol_sha256': _sha(paths['05_model_input'] / 'protocol.json'),
                'pins': protocol['hashes'], 'import_origins': protocol['import_origins'],
                'recovery_sha256': _sha(paths['05_model_input'] / 'recovery.json')})
            _command_start(paths, 'recover_evaluation', invocation, {'created_at_utc': _now(),
                'parent_run_id': parent_run_id, 'inherited_TRAIN_fits': 2, 'new_fit_budget': _ZERO_FITS,
                'restart_allowed': False})
            for destination, pin in copies.items():
                if not Path(destination).exists():
                    shutil.copyfile(pin['path'], destination)
            if not (paths['08_reporting'] / 'new_operation_trace.json').exists():
                _json(paths['08_reporting'] / 'new_operation_trace.json', [])
        else:
            invocation['started'] = True
            protocol = _load(paths['05_model_input'] / 'protocol.json')
        _frozen(run_id, repo, research)
        series = pd.read_parquet(paths['04_feature'] / 'series.parquet').to_dict('records')
        return _finish_evaluation(repo, paths, protocol, operations, models, series, opportunities,
                                  inherited=inherited)
    except BaseException as error:
        if invocation.get('started'):
            _failed_command(run_id, 'recover_evaluation', error, repo, research)
        raise
