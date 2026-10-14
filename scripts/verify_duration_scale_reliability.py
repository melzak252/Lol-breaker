#!/usr/bin/env python3
"""Independent P1 saved-state oracle. Never trains or repairs a candidate.

Expected arithmetic is standalone NumPy/SciPy. Candidate imports occur only in
public-API checks, where they supply actual values. The frozen tree regressor is
an explicitly shared conditional baseline, not an independently fitted model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from scipy.stats import lognorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
COLUMNS = ('x_pace_player_duration_minutes_count_mean_level',
           'x_pace_player_duration_minutes_mean_gap',
           'x_pace_player_missing_duration_minutes_level')
MONTHS = tuple(f'2026-{m:02d}' for m in range(1, 10))
OPERATIONS = ('TRAIN_REFERENCE_SIGMA', 'TRAIN_HEAD', 'PG_CAL', 'PC_CAL')
LINES = (25.5, 28.5, 31.5, 34.5, 37.5)
QUANTILES = {'q025': .025, 'q05': .05, 'q10': .1, 'q25': .25, 'q50': .5,
             'q75': .75, 'q90': .9, 'q95': .95, 'q975': .975}
TOLERANCES = {'exact': (0., 0.), 'fp64': (1e-10, 1e-10),
              'mean': (0., 1e-10), 'cdf': (1e-8, 0.),
              'gradient': (2e-6, 2e-5), 'quadrature': (1e-7, 1e-8)}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Audit:
    def __init__(self):
        self.checks, self.failures = [], []
        self.hashes = {}
        self.calculations = Counter()
        self.not_run = []

    def record(self, name, passed, kind='exact', reason=None, **fields):
        entry = dict(check_id=name, kind=kind, scope='P1', input_reference=name,
                     independent_method='Standalone NumPy/SciPy or source identity replay',
                     expected_reference='independent reconstruction', observed_reference='saved/public actual',
                     compared_values=0, tolerance_kind=kind, maximum_absolute_difference=0.,
                     maximum_relative_difference=0., missing_mask_mismatches=0,
                     status='PASS' if passed else 'FAIL', reason=reason, failure_examples=[])
        entry.update(fields)
        self.checks.append(entry)
        if not passed:
            self.failures.append(entry)

    def compare(self, name, actual, expected, kind='fp64'):
        a, b = np.asarray(actual), np.asarray(expected)
        if a.shape != b.shape:
            self.record(name, False, kind, 'shape mismatch', failure_examples=[{'actual': list(a.shape), 'expected': list(b.shape)}])
            return
        if a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
            okay = a == b
            self.record(name, bool(okay.all()), 'exact', compared_values=a.size,
                        failure_examples=[{'index': i.tolist(), 'actual': str(a[tuple(i)]), 'expected': str(b[tuple(i)])}
                                          for i in np.argwhere(~okay)[:12]])
            return
        a, b = a.astype(float), b.astype(float)
        finite = np.isfinite(a) & np.isfinite(b)
        missing = np.isnan(a) != np.isnan(b)
        atol, rtol = TOLERANCES[kind]
        okay = (finite & (np.abs(a - b) <= atol + rtol * np.abs(b))) | (np.isnan(a) & np.isnan(b))
        difference = np.abs(a[finite] - b[finite])
        self.record(name, bool(okay.all()), kind, compared_values=a.size,
                    maximum_absolute_difference=float(difference.max()) if difference.size else 0.,
                    maximum_relative_difference=float(np.max(difference / np.maximum(np.abs(b[finite]), 1e-300))) if difference.size else 0.,
                    missing_mask_mismatches=int(missing.sum()), mismatch_count=int((~okay).sum()),
                    failure_examples=[{'index': i.tolist(), 'actual': clean(a[tuple(i)]), 'expected': clean(b[tuple(i)])}
                                      for i in np.argwhere(~okay)[:12]])

    def pin(self, path, expected):
        actual = digest(path)
        self.hashes[str(path)] = actual
        self.record('sha256:' + str(path), actual == expected, reason=None if actual == expected else 'pinned bytes differ')
        if actual != expected:
            raise ValueError('Source/artifact pin mismatch: ' + str(path))
        return actual


def partitions(frame):
    dates = pd.to_datetime(frame.date, errors='raise')
    release = pd.to_datetime(frame.release_day, errors='raise')
    return {'TRAIN': ((dates < '2025-07-01') & (release < '2025-07-01')).to_numpy(),
            'CAL': ((dates >= '2025-07-01') & (dates < '2026-01-01') & (release < '2026-01-01')).to_numpy(),
            'ASSESS': ((dates >= '2026-01-01') & (dates < '2027-01-01')).to_numpy()}


def normalize(frame, train):
    count, gap, missing = (frame[c].to_numpy(float) for c in COLUMNS)
    if not np.isfinite(count).all() or np.any(count < 0):
        raise ValueError('count finite nonnegative contract')
    if np.isinf(gap).any() or np.any(gap < 0):
        raise ValueError('gap nonnegative or NaN contract')
    if not np.isfinite(missing).all() or np.any((missing < 0) | (missing > 5)):
        raise ValueError('missing count [0,5] contract')
    observed = gap[train & np.isfinite(gap)]
    if not len(observed):
        raise ValueError('no observed TRAIN gap')
    median = float(np.median(observed))
    values = np.column_stack((np.log1p(count), np.where(np.isnan(gap), median, gap)))
    mean, scale = values[train].mean(0), values[train].std(0, ddof=0)
    if np.any(scale <= 0) or missing[train].var(ddof=0) <= 0:
        raise ValueError('TRAIN covariate variance is zero')
    variance = np.column_stack((values, missing))[train].var(0, ddof=0)
    z = np.column_stack((np.clip((values - mean) / scale, -3., 3.), missing))
    rank = int(np.linalg.matrix_rank(np.column_stack((np.ones(train.sum()), z[train])))) if np.isfinite(z[train]).all() else 0
    if np.any(variance <= 0) or rank != 4:
        raise ValueError(f'TRAIN variance/rank contract: variance={variance.tolist()}, rank={rank}')
    return z, dict(gap_median=median, mean=mean, scale=scale, variance=variance, rank=rank,
                   counts=np.isfinite(values[train]).sum(0), sums=values[train].sum(0),
                   sums_of_squares=np.square(values[train]).sum(0), ddof=0)


def objective(theta, design, mu0, y, anchor, ridge):
    """Differentiate a SciPy density, with explicit clipped-domain derivative."""
    theta = np.atleast_1d(theta).astype(float)
    raw_log_sigma = math.log(anchor) + design @ theta
    sigma = np.exp(np.clip(raw_log_sigma, math.log(1e-4), math.log(5.)))
    logmean = mu0 + anchor * anchor / 2
    mu = logmean - sigma * sigma / 2
    loss = -lognorm.logpdf(y, s=sigma, scale=np.exp(mu)).mean() + np.dot(ridge, theta * theta)
    residual = np.log(y) - logmean + sigma * sigma / 2
    active = (raw_log_sigma > math.log(1e-4)) & (raw_log_sigma < math.log(5.))
    derivative = np.where(active, 1 + residual - (residual / sigma) ** 2, 0.)
    gradient = design.T @ derivative / len(y) + 2 * ridge * theta
    return float(loss), gradient


def law(mu, sigma, y):
    """Single-lognormal CRPS from E|X-y| - E|X-X'|/2.

    The truncated first moment is m*Phi((log y-mu)/s-s); the
    independent pair term is m*(2*Phi(s/sqrt(2))-1).
    """
    mean = np.exp(mu + sigma * sigma / 2)
    z = (np.log(y) - mu) / sigma
    result = {'duration_log_mu': mu, 'duration_log_sigma': sigma, 'fixed_mean': mean,
              'nll': -lognorm.logpdf(y, s=sigma, scale=np.exp(mu)),
              'pit': lognorm.cdf(y, s=sigma, scale=np.exp(mu)),
              'crps': y * (2 * ndtr(z) - 1) - 2 * mean * (ndtr(z - sigma) + ndtr(sigma / math.sqrt(2)) - 1)}
    for name, probability in QUANTILES.items():
        result[name] = lognorm.ppf(probability, s=sigma, scale=np.exp(mu))
    for line in LINES:
        result[f'p_over_{line}'] = lognorm.sf(line, s=sigma, scale=np.exp(mu))
    return result


def block_contrast(delta, labels, universe, mask=None, confidence=.995, one_sided=False, min_rows=200, min_blocks=6):
    """Independent whole-block expansion, including unrepresented blocks."""
    delta, labels = np.asarray(delta, float), np.asarray(labels, str)
    mask = np.ones(len(delta), bool) if mask is None else np.asarray(mask, bool)
    universe = np.asarray(universe, str)
    counts = np.array([np.count_nonzero(mask & (labels == b)) for b in universe], np.int64)
    sums = np.array([delta[mask & (labels == b)].sum() for b in universe])
    means = np.divide(sums, counts, out=np.zeros(len(counts)), where=counts > 0)
    indices = np.random.default_rng(20261009).integers(0, len(universe), size=(10000, len(universe)), dtype=np.int64)
    denominator = counts[indices].sum(1)
    equal_denominator = (counts[indices] > 0).sum(1)
    weighted = np.divide(sums[indices].sum(1), denominator, out=np.full(10000, np.nan), where=denominator > 0)
    equal = np.divide(means[indices].sum(1), equal_denominator, out=np.full(10000, np.nan), where=equal_denominator > 0)
    nonfinite = int(np.count_nonzero(mask & ~np.isfinite(delta)))
    undefined = int(np.count_nonzero(~np.isfinite(weighted) | ~np.isfinite(equal)))
    status = 'NOT_ESTIMABLE' if undefined or nonfinite or not counts.sum() else ('INSUFFICIENT_SUPPORT' if counts.sum() < min_rows or np.count_nonzero(counts) < min_blocks else 'ESTIMABLE')
    reason = ('nonfinite selected loss differences' if nonfinite else 'empty selection' if not counts.sum() else
              'zero eligible denominator in bootstrap draws' if undefined else
              f'support below floors: rows {int(counts.sum())}/{min_rows}, occupied blocks {int(np.count_nonzero(counts))}/{min_blocks}'
              if status == 'INSUFFICIENT_SUPPORT' else None)
    def interval(samples):
        if undefined or nonfinite:
            return [None, None]
        if one_sided:
            return [None, float(np.quantile(samples, confidence, method='linear'))]
        return np.quantile(samples, [(1-confidence)/2, (1+confidence)/2], method='linear').tolist()
    lower, upper = interval(weighted)
    elower, eupper = interval(equal)
    summary = dict(status=status, reason=reason, n_rows=int(counts.sum()), n_blocks=int(np.count_nonzero(counts)),
                   n_universe_blocks=len(universe), point=float(sums.sum()/counts.sum()) if counts.sum() and not nonfinite else None,
                   lower=lower, upper=upper, equal_block=dict(point=float(means[counts > 0].mean()) if np.any(counts) and not nonfinite else None, lower=elower, upper=eupper),
                   undefined_draws=undefined, nonfinite_selected_rows=nonfinite, confidence=confidence, one_sided=one_sided,
                   draws=10000, seed=20261009, bootstrap_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                   block_summary=[dict(block=b, count=int(n), sum=float(s), mean_or_null=float(s/n) if n else None) for b,n,s in zip(universe, counts, sums)], min_rows=min_rows, min_blocks=min_blocks)
    arrays = dict(indices=indices, counts=counts, sums=sums, means=np.where(counts > 0, means, np.nan),
                  weighted=weighted, equal=equal, denominator=denominator, equal_denominator=equal_denominator,
                  undefined=~np.isfinite(weighted) | ~np.isfinite(equal))
    return summary, arrays


def compare_mapping(audit, name, actual, expected):
    for key, value in expected.items():
        if key not in actual:
            audit.record(name + '.' + key, False, reason='required field absent')
        elif isinstance(value, dict):
            compare_mapping(audit, name + '.' + key, actual[key], value)
        elif value is None or isinstance(value, (str, bool)):
            audit.record(name + '.' + key, actual[key] == value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            audit.record(name + '.' + key + '.length', len(actual[key]) == len(value))
            for i, (observed, wanted) in enumerate(zip(actual[key], value)):
                compare_mapping(audit, f'{name}.{key}[{i}]', observed, wanted)
        else:
            audit.compare(name + '.' + key, actual[key], value, 'exact' if isinstance(value, int) else 'fp64')


def source_objects(path):
    """Bounded streaming parser; no production source reader is imported."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as stream:
        buffer, done = '', False
        def refill():
            nonlocal buffer, done
            chunk = stream.read(1024 * 1024)
            done = not chunk
            buffer += chunk
        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not done:
                refill()
                buffer = buffer.lstrip()
        whitespace()
        if not buffer.startswith('['):
            raise ValueError('source must be a JSON array')
        buffer = buffer[1:]
        while True:
            whitespace()
            if buffer.startswith(']'):
                buffer = buffer[1:]
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if done:
                        raise ValueError('truncated source JSON')
                    refill()
            buffer = buffer[end:]
            yield value
            whitespace()
            if buffer.startswith(','):
                buffer = buffer[1:]
                whitespace()
                if buffer.startswith(']'):
                    raise ValueError('source trailing comma')
            elif not buffer.startswith(']'):
                raise ValueError('source delimiter')
        whitespace()
        if buffer:
            raise ValueError('trailing source bytes')


def identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('unusable explicit identity')
    value = str(value)
    if not value or value != value.strip() or value.lower() in ('none', 'null', 'nan'):
        raise ValueError('unusable explicit identity')
    return value


def day(value):
    return pd.Timestamp(value).date().isoformat()


def number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def excluded(value):
    return any(value.get(k) is True or value.get(k) == 1 for k in ('remake', 'is_remake', 'cancelled', 'canceled', 'forfeit', 'is_forfeit')) or str(value.get('status', '')).lower().strip() in ('remake', 'cancelled', 'canceled', 'forfeit')


def roster(game, side):
    payload = game.get(side + '_players')
    if not isinstance(payload, dict) or len(payload) != 5:
        raise ValueError('five explicit source player objects required')
    ids = []
    for player in payload.values():
        values = {identity(player[k]) for k in ('player_id', 'id') if player.get(k) is not None}
        if len(values) != 1:
            raise ValueError('missing/conflicting player identity')
        ids.append(values.pop())
    if len(set(ids)) != 5:
        raise ValueError('duplicate source roster identity')
    return tuple(ids)


def summaries(players, teams, player_fields, team_fields, prefix=''):
    """Enumerate channels independently by named statistic and side."""
    # The inherited feature contract promotes stored FP32 values before reduction.
    players, teams = (np.asarray(values, dtype=np.float64) for values in (players, teams))
    valid = np.isfinite(players)
    counts = valid.sum(2)
    average = np.divide(np.where(valid, players, 0.).sum(2), counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    variance = np.divide(np.where(valid, (players-average[:, :, None])**2, 0.).sum(2), counts,
                         out=np.full(counts.shape, np.nan), where=counts > 0)
    output = {}
    channels = [('player_mean', average, player_fields), ('player_std', np.sqrt(variance), player_fields),
                ('player_missing', (~valid).sum(2), player_fields), ('team', teams, team_fields),
                ('team_missing', (~np.isfinite(teams)).astype(float), team_fields)]
    for tag, values, names in channels:
        for j, field in enumerate(names):
            for stat in ('level', 'gap'):
                name = prefix + tag + '_' + str(field) + '_' + stat
                if name.startswith('pace_player_mean_'):
                    name = 'pace_player_' + str(field) + '_mean_' + stat
                output['x_' + name] = (values[:, 0, j]+values[:, 1, j])/2 if stat == 'level' else abs(values[:, 0, j]-values[:, 1, j])
    return output


def reconstruct_sources(repo, contract, frame, audit):
    raw = repo / contract['raw_feature_dir']
    metadata = pd.read_parquet(raw / 'metadata.parquet')
    audit.compare('raw_metadata_order', metadata.match_id.astype(str), frame.match_id.astype(str), 'exact')
    for column in ('date', 'release_day', 'best_of', 'team_a', 'team_b'):
        audit.compare('raw_metadata.' + column, metadata[column].astype(str), frame[column].astype(str), 'exact')
    with np.load(raw / 'arrays.npz', allow_pickle=False) as arrays:
        features = summaries(arrays['players'], arrays['teams'], range(arrays['players'].shape[-1]), range(arrays['teams'].shape[-1]))
        for j, name in enumerate(('bo1', 'bo3', 'bo5')):
            features['x_' + name] = arrays['context'][:, j]
            features['x_' + name + '_missing'] = (~np.isfinite(arrays['context'][:, j])).astype(float)
            audit.compare('raw_context.' + name, arrays['context'][:, j], (frame.best_of == (1,3,5)[j]).astype(float), 'exact')
        for j in range(2):
            features[f'x_baseline_{j}_distance_half'] = abs(arrays['baseline_series'][:, j] - .5)
            features[f'x_baseline_{j}_missing'] = (~np.isfinite(arrays['baseline_series'][:, j])).astype(float)
        for name, values in features.items():
            audit.compare('raw_feature.' + name, frame[name], values)
    rows = {identity(row.match_id): row for row in frame.itertuples(index=False)}
    events, seen_series, seen_games = [], set(), set()
    labels = {}
    for raw_series in source_objects(repo / contract['source']):
        mid = identity(raw_series.get('match_id'))
        if mid in seen_series:
            raise ValueError('duplicate raw match ID')
        seen_series.add(mid)
        games = raw_series.get('games')
        if not isinstance(games, list):
            raise ValueError('raw games missing')
        clocks = [day(raw_series['date'])] + [day(g.get('date', raw_series['date'])) for g in games]
        target = rows.get(mid)
        release = str(target.release_day) if target is not None else max(clocks)
        if release < max(clocks):
            raise ValueError('canonical release precedes source maps')
        if target is not None:
            modes = {int(raw_series[k]) for k in ('BoN', 'best_of') if raw_series.get(k) is not None}
            audit.record('source_identity:' + mid, str(target.date) == min(clocks) and modes == {int(target.best_of)})
        for i, game in enumerate(games):
            gid = identity(game.get('game_id'))
            if gid in seen_games:
                raise ValueError('duplicate raw game ID')
            seen_games.add(gid)
            teams = tuple(identity(game.get(s + '_id')) for s in ('t1','t2'))
            rosters = tuple(roster(game, s) for s in ('t1','t2'))
            if teams[0] == teams[1] or set(rosters[0]) & set(rosters[1]):
                raise ValueError('corrupt source sides')
            if game.get('match_id') is not None and identity(game['match_id']) != mid:
                raise ValueError('source game parent identity')
            if game.get('source_best_of') is not None and target is not None and int(game['source_best_of']) != target.best_of:
                raise ValueError('source map best_of mismatch')
            if i == 0 and target is not None:
                mapping = dict(zip(teams, rosters))
                audit.record('source_rosters:' + mid, set(mapping) == {str(target.team_a), str(target.team_b)} and
                             set(mapping.get(str(target.team_a), ())) == set(map(str, target.roster_a)) and
                             set(mapping.get(str(target.team_b), ())) == set(map(str, target.roster_b)))
            duration = number(game.get('game_duration'))
            duration = duration/60 if duration > 0 and not excluded(game) else math.nan
            if i == 0 and target is not None:
                labels[mid] = math.nan if excluded(raw_series) else duration
            if not excluded(game) and not excluded(raw_series):
                kills = [number((game.get(s + '_stats') or {}).get('kills')) for s in ('t1','t2')]
                kills = [x if x >= 0 and x.is_integer() else math.nan for x in kills]
                a,b = kills
                events.append((release, mid, i, teams, rosters, ((a,b,a+b,duration),(b,a,a+b,duration))))
    audit.compare('source_duration_labels', frame.duration_minutes, [labels.get(str(mid), math.nan) for mid in frame.match_id])
    events.sort(key=lambda e: e[:3])
    players, teams = {}, {}
    fields = ('kills_for','kills_against','total_kills','duration_minutes')
    fields += tuple(k + '_count' for k in fields)
    n = len(frame)
    player_values, team_values = np.empty((n,2,5,8)), np.empty((n,2,8))
    max_release, position, last = [None]*n, 0, None
    def history_values(history):
        if not history:
            return np.r_[np.full(4,np.nan), np.zeros(4)]
        values = np.asarray(history)
        mask = np.isfinite(values)
        counts = mask.sum(0)
        means = np.divide(np.where(mask,values,0).sum(0), counts, out=np.full(4,np.nan), where=counts > 0)
        return np.r_[means, counts]
    ordered_rows = list(frame.itertuples(index=False))
    for index in sorted(range(n), key=lambda i: (str(ordered_rows[i].date), i)):
        row = ordered_rows[index]
        while position < len(events) and events[position][0] < str(row.date):
            release, mid, _, tids, rosters, values = events[position]
            if mid == str(row.match_id):
                raise ValueError('target appearance entered its own history')
            for tid, rids, value in zip(tids, rosters, values):
                teams.setdefault(tid, deque(maxlen=20)).append(value)
                for pid in rids:
                    players.setdefault(pid, deque(maxlen=16)).append(value)
            if np.isfinite(values).any():
                last = release
            position += 1
        for side, (tid,rids) in enumerate(((row.team_a,row.roster_a),(row.team_b,row.roster_b))):
            team_values[index,side] = history_values(teams.get(str(tid)))
            for slot,pid in enumerate(rids):
                player_values[index,side,slot] = history_values(players.get(str(pid)))
        max_release[index] = last
    for name, values in summaries(player_values, team_values, fields, fields, 'pace_').items():
        audit.compare('source_pace.' + name, frame[name], values)
    audit.compare('source_pace_release', frame.max_pace_release_day.fillna('EMPTY').astype(str),
                  np.array([x or 'EMPTY' for x in max_release]), 'exact')
    audit.calculations['source_series'] = len(seen_series)
    audit.calculations['source_maps'] = len(seen_games)
    audit.record('future_and_target_withheld', all(x is None or x < str(d) for x,d in zip(max_release,frame.date)),
                 reason='Strict-prior event replay never ingests target/same-day/future labels; target perturbation is additionally checked through public unlabeled inference.')
    return len(events)


def replay_tree(tree, frame, feature_columns):
    """Call only pinned saved regressor; reconstruct its input transform."""
    x = frame[feature_columns].to_numpy(float)
    pre = tree.preprocessor
    if pre is not None:
        present = ~np.isnan(x) & pre.observed
        x = np.concatenate((np.where(present, (x-pre.mean)/pre.scale, 0.), (~present).astype(float)), axis=1)
    mu = np.clip(tree.duration_model.predict(x), -12., 12.)
    return mu, float(tree.duration_sigma)


def verify_operation_trace(audit, trace, completed):
    audit.record('operation_ceiling', len(trace) <= 4)
    for index, operation in enumerate(trace):
        audit.record(f'operation_order[{index}]', index < 4 and operation['operation_id'] == OPERATIONS[index])
        for key in ('study', 'kind', 'arm', 'seed', 'partition', 'authorized_ceiling',
                    'counter_before', 'counter_after', 'started_at_utc', 'finished_at_utc',
                    'status', 'parameters_sha256', 'input_ids_sha256', 'iterations',
                    'function_evaluations', 'failure_reason'):
            audit.record(f'operation_field[{index}].{key}', key in operation)
        audit.record(f'operation_counter[{index}]', operation['counter_before'] == index and operation['counter_after'] == index+1)
        audit.record(f'operation_partition[{index}]', operation['partition'] == ('TRAIN' if index < 2 else 'CAL'))
        if operation.get('finished_at_utc'):
            audit.record(f'operation_clock[{index}]',
                         pd.Timestamp(operation['started_at_utc']) <= pd.Timestamp(operation['finished_at_utc']))
        audit.record(f'operation_iterations[{index}]', 0 <= (operation.get('iterations') or 0) <= 500)
    if completed:
        audit.record('completed_operation_count', len(trace) == 4 and all(x['status'] == 'SUCCESS' for x in trace))


def verify_gradient(audit, name, theta, design, mu0, y, anchor, ridge, saved, bounds):
    theta = np.atleast_1d(theta).astype(float)
    value, gradient = objective(theta, design, mu0, y, anchor, ridge)
    audit.calculations['saved_objective_reconstructions'] += 1
    audit.compare(name+'.objective', saved['objective'], value)
    audit.compare(name+'.gradient', saved['gradient'], gradient)
    audit.compare(name+'.parameters', saved['parameters'], theta)
    lower, upper = bounds
    projected = gradient.copy()
    projected[((theta == lower) & (gradient > 0)) | ((theta == upper) & (gradient < 0))] = 0
    audit.compare(name+'.projected_gradient', saved['projected_gradient_inf'], np.max(abs(projected)))
    audit.record(name+'.accepted', saved['status'] == 'success' and np.isfinite(value) and
                 np.isfinite(theta).all() and np.isfinite(gradient).all() and
                 np.all((theta >= lower) & (theta <= upper)) and np.max(abs(projected)) <= 1e-5 and
                 0 <= saved['iterations'] <= 500 and saved['estimation_operations'] == 1)
    audit.compare(name+'.bounds', saved['bounds'], [lower, upper])
    audit.record(name+'.boundary', saved['boundary_hit'] == bool(np.any(np.isclose(theta, lower, atol=1e-8, rtol=0) | np.isclose(theta, upper, atol=1e-8, rtol=0))))
    for fixture_name, point in [('saved', theta), ('zero', np.zeros(len(theta))),
                                ('interior', np.full(len(theta), .071))]:
        _, analytic = objective(point, design, mu0, y, anchor, ridge)
        for j in range(len(point)):
            step = 1e-5 * max(1., abs(point[j]))
            direction = np.zeros(len(point)); direction[j] = step
            raw = math.log(anchor) + design @ point
            movement = abs(design[:, j] * step)
            kink = np.any((abs(raw-math.log(1e-4)) <= movement) | (abs(raw-math.log(5.)) <= movement))
            if kink or point[j]-step < lower or point[j]+step > upper:
                sign = -1 if point[j]+step > upper else 1
                moved = point + sign*direction
                fd = (objective(moved, design, mu0, y, anchor, ridge)[0] -
                      objective(point, design, mu0, y, anchor, ridge)[0])/(sign*step)
                # One-sided secant has O(h) curvature error; at a clipping kink
                # only the independently derived feasible directional slope is
                # defined, not a fictitious two-sided analytic gradient.
                audit.record(f'{name}.{fixture_name}.direction[{j}]', np.isfinite(fd),
                             reason='Feasible one-sided clipping/bound directional derivative disclosed',
                             directional_derivative=float(fd), central_difference_not_defined=kink)
            else:
                fd = (objective(point+direction, design, mu0, y, anchor, ridge)[0] -
                      objective(point-direction, design, mu0, y, anchor, ridge)[0])/(2*step)
                audit.compare(f'{name}.{fixture_name}.finite_difference[{j}]', analytic[j], fd, 'gradient')


def scalar_verification(audit, name, c, z, slopes, mu0, y, sigma0, saved):
    # Fixed slope offset is moved into an equivalent row-specific location
    # while retaining the original arithmetic mean anchor explicitly.
    offset = z @ slopes
    def calculation(value):
        sigma = np.exp(np.clip(math.log(sigma0) + float(value) + offset, math.log(1e-4), math.log(5.)))
        mu = mu0 + sigma0*sigma0/2 - sigma*sigma/2
        result = -lognorm.logpdf(y, s=sigma, scale=np.exp(mu)).mean() + .001*float(value)**2
        residual = np.log(y)-mu
        active = (math.log(sigma0)+float(value)+offset > math.log(1e-4)) & (math.log(sigma0)+float(value)+offset < math.log(5.))
        derivative = np.where(active, 1+residual-(residual/sigma)**2, 0.).mean()+.002*float(value)
        return float(result), float(derivative)
    loss, gradient = calculation(c)
    audit.calculations['saved_objective_reconstructions'] += 1
    audit.compare(name+'.objective', saved['objective'], loss)
    audit.compare(name+'.gradient', saved['gradient'], [gradient])
    audit.compare(name+'.parameters', saved['parameters'], [c])
    lower, upper = math.log(.25), math.log(4.)
    projected = 0. if (c == lower and gradient > 0) or (c == upper and gradient < 0) else gradient
    audit.compare(name+'.projected_gradient', saved['projected_gradient_inf'], abs(projected))
    audit.record(name+'.accepted', saved['status'] == 'success' and lower <= c <= upper and
                 np.isfinite([c,loss,gradient]).all() and abs(projected) <= 1e-5 and saved['iterations'] <= 500)
    for fixture_name, value in [('saved', c), ('zero', 0.), ('interior', .071)]:
        step = 1e-5*max(1.,abs(value))
        raw = math.log(sigma0)+value+offset
        kink = np.any((abs(raw-math.log(1e-4)) <= step) | (abs(raw-math.log(5.)) <= step))
        if not kink and lower < value-step < value+step < upper:
            audit.compare(name+'.'+fixture_name+'.finite_difference', calculation(value)[1],
                          (calculation(value+step)[0]-calculation(value-step)[0])/(2*step), 'gradient')
        else:
            audit.record(name+'.'+fixture_name+'.direction', True,
                         reason='One-sided derivative at bound/clipping crossing; central derivative unavailable',
                         directional_derivative=(calculation(value-step if value+step > upper else value+step)[0]-calculation(value)[0])/(-step if value+step > upper else step))
    result = minimize_scalar(lambda value: calculation(value)[0], bounds=(lower,upper), method='bounded',
                             options={'maxiter': 500, 'xatol': 1e-8})
    audit.calculations['saved_scalar_verification_solves'] += 1
    audit.record(name+'.independent_scalar_success', bool(result.success) and result.nit <= 500 and np.isfinite(result.fun),
                 verification_only=True, independently_solved_parameter=float(result.x))
    audit.record(name+'.independent_scalar_objective', abs(loss-float(result.fun)) <= 1e-8,
                 maximum_absolute_difference=abs(loss-float(result.fun)))


def public_checks(audit, model_path, frame, masks, expected, prepared_mu, sigma0, cases, output):
    from src.models import duration_scale_reliability as actual_api
    model = actual_api.load_duration_scale(model_path, expected_sha256=digest(model_path))
    before = joblib.hash(model)
    unlabeled = frame.drop(columns=[c for c in ('duration_minutes','kills','maps_played','duration_reason','kills_reason','maps_reason') if c in frame])
    transformed = model.preprocessor.transform(unlabeled)
    independent_z, _ = normalize(frame, masks['TRAIN'])
    audit.compare('public_normalizer', transformed, independent_z)
    observed_pre = actual_api.ScalePreprocessor.fit(frame.loc[masks['TRAIN']])
    audit.compare('public_preprocessor_fit.mean', observed_pre.mean, model.preprocessor.mean)
    audit.compare('public_preprocessor_fit.scale', observed_pre.scale, model.preprocessor.scale)
    audit.compare('public_sigma_T', actual_api.train_reference_sigma(prepared_mu[masks['TRAIN']], frame.duration_minutes.to_numpy(float)[masks['TRAIN']]),
                  model.train_reference_sigma)
    for arm in ('P0','PG','PC'):
        result = model.predict(unlabeled, arm=arm)
        audit.compare('public_full.'+arm+'.mu', result.duration_log_mu, expected[arm]['duration_log_mu'])
        audit.compare('public_full.'+arm+'.sigma', result.duration_log_sigma, expected[arm]['duration_log_sigma'])
        base = model.predict(unlabeled, arm='P0')
        for field in ('kills_mu','kills_alpha','length_probs','best_of'):
            audit.compare('public_unchanged.'+arm+'.'+field, getattr(result,field), getattr(base,field), 'exact')
    lexical = frame.assign(_id=frame.match_id.astype(str)).sort_values('_id')
    selected = set()
    states = {'BO1': frame.best_of.eq(1), 'BO3': frame.best_of.eq(3), 'BO5': frame.best_of.eq(5),
              'cold': frame[COLUMNS[0]].eq(0), 'partial': frame[COLUMNS[0]].between(0,16,inclusive='neither'),
              'full': frame[COLUMNS[0]].eq(16), 'missing_gap': frame[COLUMNS[1]].isna(),
              'missing_players': frame[COLUMNS[2]].gt(0), 'ordinary': frame[COLUMNS[2]].eq(0)}
    for state, membership in states.items():
        support = lexical.loc[membership.loc[lexical.index]]
        if len(support):
            selected.add(int(support.index[0]))
        else:
            audit.record('public_absent_state.'+state, True, reason='ABSENT_IN_SOURCE; synthetic input contract checked separately')
    saved_states = {f'BO{b}':frame.best_of.eq(b) for b in (1,3,5)}
    saved_states.update({f'history_{h}':frame.history.eq(h) for h in ('full','partial','cold')})
    saved_states.update(gap_missing=frame[COLUMNS[1]].isna(),gap_observed=frame[COLUMNS[1]].notna())
    expected_case_ids = set()
    for state,mask in saved_states.items():
        population = lexical.loc[(mask & frame.partition.ne('EXCLUDED')).loc[lexical.index]]
        if len(population):
            expected_case_ids.update(f'{state}/{arm}' for arm in ('P0','PG','PC'))
    audit.record('saved_case_complete_states',{record['case_id'] for record in cases} == expected_case_ids)
    for record in cases:
        for field in ('case_id','study','arm','seed_or_ensemble','input_reference_and_sha256','target_origin','scenario',
                      'contract_state','expected_status','expected_reason','public_result_reference','independent_expected_reference',
                      'comparison_kind','tolerances','state_hash_before','state_hash_after'):
            audit.record('public_case_field.'+record.get('case_id','UNKNOWN')+'.'+field, field in record)
        state_population = lexical.loc[(saved_states[record['contract_state']] & frame.partition.ne('EXCLUDED')).loc[lexical.index]]
        audit.record('saved_case_selection.'+record['case_id'],bool(len(state_population)) and str(state_population.match_id.iloc[0]) == record['match_id'])
        audit.record('saved_public_case_nonmutation.'+record['case_id'], record['state_hash_before'] == record['state_hash_after'])
        row_indices = np.flatnonzero(frame.match_id.astype(str).eq(record['match_id']).to_numpy())
        audit.record('saved_case_identity.'+record['case_id'],len(row_indices) == 1)
        if len(row_indices) == 1:
            row_index = int(row_indices[0])
            values = expected[record['arm']]
            compare_mapping(audit,'saved_case_law.'+record['case_id'],record['public_result_reference'],
                            {'duration_log_mu':float(values['duration_log_mu'][row_index]),
                             'duration_log_sigma':float(values['duration_log_sigma'][row_index]),
                             'predictive_mean':float(values['fixed_mean'][row_index]),
                             'p_over':{str(line):float(values[f'p_over_{line}'][row_index]) for line in LINES}})
    audit.record('saved_public_cases_present', bool(cases))
    audit.record('public_case_limit', len(selected) <= 32)
    for index in sorted(selected):
        one = unlabeled.iloc[[index]].copy()
        swapped = one.copy()
        for left,right in (('team_a','team_b'),('roster_a','roster_b')):
            swapped[left],swapped[right] = one[right].copy(),one[left].copy()
        poisoned = frame.iloc[[index]].copy()
        for field in ('duration_minutes','kills','maps_played'):
            if field in poisoned:
                poisoned[field] = 1e12
        original_hash = joblib.hash(one)
        for arm in ('P0','PG','PC'):
            result = model.predict(one,arm=arm)
            swapped_result = model.predict(swapped,arm=arm)
            poisoned_result = model.predict(poisoned,arm=arm)
            for name in ('duration_log_mu','duration_log_sigma'):
                audit.compare(f'public_singleton[{index}].{arm}.{name}', getattr(result,name), expected[arm][name][[index]])
                audit.compare(f'public_target_perturbation[{index}].{arm}.{name}', getattr(poisoned_result,name), getattr(result,name), 'exact')
            for line in LINES:
                audit.compare(f'public_swap[{index}].{arm}.{line}', swapped_result.duration_over(line), result.duration_over(line), 'cdf')
                audit.compare(f'public_cdf[{index}].{arm}.{line}', result.duration_over(line), expected[arm][f'p_over_{line}'][[index]], 'cdf')
            audit.compare(f'public_mean[{index}].{arm}', np.exp(result.duration_log_mu+result.duration_log_sigma**2/2),
                          np.exp(prepared_mu[[index]]+sigma0**2/2), 'mean')
        audit.record(f'public_input_nonmutation[{index}]', joblib.hash(one) == original_hash)
    altered = copy.deepcopy(model); altered.train_intercept += .125
    audit.compare('discarded_train_intercept', altered.predict(unlabeled).duration_log_mu, model.predict(unlabeled).duration_log_mu, 'exact')
    for name, bad in [('negative_count',-1.),('nonfinite_count',np.inf)]:
        invalid = unlabeled.iloc[[0]].copy(); invalid[COLUMNS[0]] = bad
        try:
            model.predict(invalid)
        except (ValueError, TypeError):
            audit.record('public_invalid.'+name, True)
        else:
            audit.record('public_invalid.'+name, False, reason='malformed scale input accepted')
    try:
        actual_api.load_duration_scale(model_path, expected_sha256='0'*64)
    except ValueError:
        audit.record('public_hash_mismatch', True)
    else:
        audit.record('public_hash_mismatch', False)
    roundtrip = output / 'public_roundtrip.joblib'
    model.save(roundtrip)
    decoded = actual_api.load_duration_scale(roundtrip, expected_sha256=digest(roundtrip))
    audit.compare('public_save_load', decoded.predict(unlabeled).duration_log_mu, expected['PC']['duration_log_mu'])
    try:
        model.save(roundtrip)
    except FileExistsError:
        audit.record('public_exclusive_save', True)
    else:
        audit.record('public_exclusive_save', False)
    public_contract_fixtures(audit,model,unlabeled)
    audit.record('public_state_nonmutation', joblib.hash(model) == before)
    audit.calculations['public_real_cases'] = len(selected)


def public_contract_fixtures(audit, model, frame):
    from types import SimpleNamespace
    from src.models import duration_scale_reliability as actual
    z = np.array([[-1.,.2,0.],[.4,-.3,1.],[.8,.6,.5],[-.2,-.5,2.]])
    mu0,y,sigma = np.log([22.,28.,35.,30.]),np.array([25.,29.,40.,24.]),.18
    design = np.column_stack((np.ones(4),z))
    for name,theta in [('zero',np.zeros(4)),('interior',np.array([.1,-.2,.05,.1]))]:
        wanted = objective(theta,design,mu0,y,sigma,np.array([0.,.1,.1,.1]))
        observed = actual.train_scale_objective(theta,z,mu0,y,sigma)
        audit.compare('public_objective.'+name+'.loss',observed[0],wanted[0])
        audit.compare('public_objective.'+name+'.gradient',observed[1],wanted[1],'gradient')
        c,slopes = float(theta[0]),theta[1:]
        sigmas = np.clip(sigma*np.exp(c+z@slopes),1e-4,5.)
        locations = mu0+sigma*sigma/2-sigmas*sigmas/2
        residual = np.log(y)-locations
        wanted_loss = -lognorm.logpdf(y,s=sigmas,scale=np.exp(locations)).mean()+.001*c*c
        wanted_gradient = (1+residual-(residual/sigmas)**2).mean()+.002*c
        observed_loss,observed_gradient = actual.cal_scale_objective(c,z,mu0,y,sigma,slopes)
        audit.compare('public_cal_objective.'+name+'.loss',observed_loss,wanted_loss)
        audit.compare('public_cal_objective.'+name+'.gradient',observed_gradient,wanted_gradient,'gradient')
    for label,changes in [('failure',{'success':False}),('nonfinite',{'fun':np.nan}),('bounds',{'x':np.array([1.01,0.,0.,0.])}),
                          ('iterations',{'nit':501}),('nonstationary',{'jac':np.array([0.,.01,0.,0.])})]:
        fields = dict(success=True,x=np.zeros(4),fun=3.,jac=np.zeros(4),nit=12,nfev=16,message='independent rejection fixture')
        fields.update(changes)
        try:
            actual.accept_fit_result(SimpleNamespace(**fields),stage='TRAIN_HEAD')
        except ValueError:
            audit.record('public_accept_rejection.'+label,True)
        else:
            audit.record('public_accept_rejection.'+label,False)
    for bad in (0.,-1.,np.nan,np.inf):
        invalid_y = y.copy(); invalid_y[0] = bad
        try:
            actual.train_reference_sigma(mu0,invalid_y)
        except ValueError:
            audit.record('public_invalid_duration.'+str(bad),True)
        else:
            audit.record('public_invalid_duration.'+str(bad),False)
    synthetic = frame.iloc[:3].copy()
    synthetic['best_of'] = [1,3,5]
    synthetic[COLUMNS[0]],synthetic[COLUMNS[1]],synthetic[COLUMNS[2]] = 0.,np.nan,5.
    for b in (1,3,5):
        if f'x_bo{b}' in synthetic:
            synthetic[f'x_bo{b}'] = synthetic.best_of.eq(b).astype(float)
    expected_z = np.column_stack((np.full(3,np.clip((0.-model.preprocessor.mean[0])/model.preprocessor.scale[0],-3,3)),
                                 np.full(3,np.clip((model.preprocessor.gap_median-model.preprocessor.mean[1])/model.preprocessor.scale[1],-3,3)),
                                 np.full(3,5.)))
    audit.compare('synthetic_cold_missing_normalizer',model.preprocessor.transform(synthetic),expected_z)
    mu0,sigma0 = replay_tree(model.tree.model,synthetic,model.tree.feature_columns)
    anchor = mu0+sigma0*sigma0/2
    for arm,offset in [('P0',np.zeros(3)),('PG',np.full(3,model.c_global)),('PC',model.c_conditional+expected_z@model.slopes)]:
        sigma = np.full(3,sigma0) if arm == 'P0' else np.clip(sigma0*np.exp(offset),1e-4,5.)
        mu = mu0 if arm == 'P0' else anchor-sigma*sigma/2
        observed = model.predict(synthetic,arm=arm)
        audit.compare('synthetic_BO135_cold.'+arm+'.mu',observed.duration_log_mu,mu)
        audit.compare('synthetic_BO135_cold.'+arm+'.sigma',observed.duration_log_sigma,sigma)
        for line in LINES:
            audit.compare('synthetic_BO135_cold.'+arm+'.cdf'+str(line),observed.duration_over(line),
                          lognorm.sf(line,s=sigma,scale=np.exp(mu)),'cdf')
    base = model.predict(frame.iloc[:3],arm='P0')
    for offset in (np.array([-.4,.3,.9]),np.array([-100.,100.,0.])):
        observed = actual.mean_preserving_prediction(base,offset)
        sigma = np.clip(base.duration_log_sigma*np.exp(offset),1e-4,5.)
        mu = base.duration_log_mu+base.duration_log_sigma**2/2-sigma*sigma/2
        audit.compare('public_mean_helper.mu.'+str(offset.tolist()),observed.duration_log_mu,mu)
        audit.compare('public_mean_helper.sigma.'+str(offset.tolist()),observed.duration_log_sigma,sigma)
    optional = frame.iloc[[0]].copy()
    for column in ('team_a','team_b'):
        optional[column] = None
    audit.compare('public_optional_organization_state',model.predict(optional).duration_log_mu,
                  model.predict(frame.iloc[[0]]).duration_log_mu,'exact')


def metric_summary(values, y, mask):
    observed = y[mask]
    mean = values['fixed_mean'][mask]
    result = dict(n=int(mask.sum()), nll=float(values['nll'][mask].mean()), crps=float(values['crps'][mask].mean()),
                  mean_mae=float(abs(mean-observed).mean()), mean_bias=float((mean-observed).mean()),
                  pit_mean=float(values['pit'][mask].mean()),
                  pit_histogram=np.histogram(values['pit'][mask], bins=np.linspace(0,1,11))[0].tolist())
    for level,lo,hi in ((50,'q25','q75'),(80,'q10','q90'),(90,'q05','q95'),(95,'q025','q975')):
        coverage = float(((observed >= values[lo][mask]) & (observed <= values[hi][mask])).mean())
        result.update({f'coverage{level}':coverage, f'coverage_error{level}':abs(coverage-level/100),
                       f'width{level}':float((values[hi][mask]-values[lo][mask]).mean())})
    for line in LINES:
        result[f'brier_{line}'] = float(((values[f'p_over_{line}'][mask]-(observed > line))**2).mean())
    return result


def verify_draw_export(audit, path, delta, labels, universe, arrays):
    names = {'indices':'indices','counts':'counts','sums':'sums','means':'means',
             'weighted_denominator':'denominator','equal_denominator':'equal_denominator',
             'weighted_delta':'weighted','equal_delta':'equal','undefined':'undefined'}
    with np.load(path, allow_pickle=False) as saved:
        for field,wanted in [('delta',delta),('blocks',labels),('universe',universe)]:
            audit.compare(str(path)+'.'+field,saved[field],wanted,'fp64' if field == 'delta' else 'exact')
        for field,key in names.items():
            audit.compare(str(path)+'.'+field,saved[field],arrays[key],
                          'exact' if field in ('indices','counts','weighted_denominator','equal_denominator','undefined') else 'fp64')


def quadrature_checks(audit, frame, laws):
    selected = []
    lexical = frame.assign(_id=frame.match_id.astype(str)).sort_values('_id')
    for _,cell in lexical.groupby(['best_of','history'],dropna=False,sort=True):
        for missing in (False,True):
            rows = cell.loc[cell[COLUMNS[1]].isna().eq(missing)]
            if len(rows):
                selected.append(int(rows.index[0]))
    fixtures = []
    for index in selected:
        for arm in ('P0','PG','PC'):
            if len(fixtures) < 30:
                fixtures.append((f'{arm}:{frame.match_id.iloc[index]}',float(frame.duration_minutes.iloc[index]),
                                 float(laws[arm]['duration_log_mu'][index]),float(laws[arm]['duration_log_sigma'][index]),
                                 float(laws[arm]['crps'][index])))
    for y,mu,sigma in ((30.,math.log(28.),.2),(1.,0.,.7)):
        fixtures.append(('synthetic',y,mu,sigma,float(law(np.array([mu]),np.array([sigma]),np.array([y]))['crps'][0])))
    for index,(name,y,mu,sigma,expected) in enumerate(fixtures):
        below,e1 = quad(lambda x: lognorm.cdf(x,s=sigma,scale=math.exp(mu))**2,
                        0.,y,epsabs=2e-10,epsrel=2e-12,limit=300)
        above,e2 = quad(lambda x: lognorm.sf(x,s=sigma,scale=math.exp(mu))**2,
                        y,np.inf,epsabs=2e-10,epsrel=2e-12,limit=300)
        audit.compare(f'crps_quadrature[{index}].{name}',expected,below+above,'quadrature')
        audit.record(f'quadrature_error[{index}]',e1+e2 <= 1e-8,quadrature_error=e1+e2)
    audit.calculations['crps_quadrature_rows'] = len(fixtures)


def verify_completed(audit, roots, frame, masks, z, mu0, sigma0, index, output):
    receipt = read_json(roots['06_models']/'receipt.json')
    fit = read_json(roots['08_reporting']/'fit_receipts.json')
    verify_operation_trace(audit,fit['operation_trace'],True)
    audit.compare('formal_count',fit['estimation_operations'],4,'exact')
    audit.compare('index_trace',[x['operation_id'] for x in index['operation_trace']],list(OPERATIONS),'exact')
    y = frame.duration_minutes.to_numpy(float)
    sigma_t = float(np.sqrt(np.mean((np.log(y[masks['TRAIN']])-mu0[masks['TRAIN']])**2)))
    audit.compare('sigma_T',receipt['sigma_T'],sigma_t)
    audit.record('sigma_T_positive',math.isfinite(sigma_t) and sigma_t > 0)
    receipts = {r['stage']:r for r in receipt['receipts']}
    audit.compare('sigma_T_receipt',receipts['TRAIN_REFERENCE_SIGMA']['parameters'],[sigma_t])
    theta = np.r_[receipt['a'],receipt['b']]
    verify_gradient(audit,'TRAIN_HEAD',theta,np.column_stack((np.ones(masks['TRAIN'].sum()),z[masks['TRAIN']])),
                    mu0[masks['TRAIN']],y[masks['TRAIN']],sigma_t,np.array([0.,.1,.1,.1]),receipts['TRAIN_HEAD'],(-1.,1.))
    for name,c,slopes in [('PG_CAL',receipt['c_G'],np.zeros(3)),('PC_CAL',receipt['c_C'],np.array(receipt['b']))]:
        scalar_verification(audit,name,c,z[masks['CAL']],slopes,mu0[masks['CAL']],y[masks['CAL']],sigma0,receipts[name])
    model_path = roots['06_models']/'duration_scale.joblib'
    audit.pin(model_path,receipt['artifact_sha256'])
    model = joblib.load(model_path)
    for name,actual,wanted in [('sigma_T',model.train_reference_sigma,sigma_t),('a',model.train_intercept,receipt['a']),
                               ('b',model.slopes,receipt['b']),('c_G',model.c_global,receipt['c_G']),('c_C',model.c_conditional,receipt['c_C'])]:
        audit.compare('saved_state.'+name,actual,wanted)
    anchor = mu0+sigma0*sigma0/2
    expected = {}
    for arm,offset in [('P0',np.zeros(len(frame))),('PG',np.full(len(frame),receipt['c_G'])),('PC',receipt['c_C']+z@np.array(receipt['b']))]:
        sigma = np.full(len(frame),sigma0) if arm == 'P0' else np.clip(sigma0*np.exp(offset),1e-4,5.)
        mu = mu0 if arm == 'P0' else anchor-sigma*sigma/2
        expected[arm] = law(mu,sigma,y)
        audit.compare('mean_preservation.'+arm,expected[arm]['fixed_mean'],np.exp(anchor),'mean')
    ts = np.clip(sigma_t*np.exp(theta[0]+z[masks['TRAIN']]@theta[1:]),1e-4,5.)
    tm = mu0[masks['TRAIN']]+sigma_t*sigma_t/2-ts*ts/2
    audit.compare('TRAIN_anchor_mean',np.exp(tm+ts*ts/2),np.exp(mu0[masks['TRAIN']]+sigma_t*sigma_t/2),'mean')
    saved = pd.read_parquet(roots['07_model_output']/'duration_predictions.parquet')
    retained = np.logical_or.reduce(list(masks.values()))
    audit.record('full_output_population',len(saved) == int(retained.sum())*3)
    for arm in expected:
        actual = saved.loc[saved.arm.eq(arm)].reset_index(drop=True)
        audit.compare('law_ids.'+arm,actual.match_id.astype(str),frame.loc[retained,'match_id'].astype(str),'exact')
        audit.compare('law_partitions.'+arm,actual.partition,frame.loc[retained,'partition'],'exact')
        audit.record('law_missingness.'+arm,bool(actual.eligible.all()) and actual.absence_reason.isna().all())
        audit.compare('law_baseline.'+arm,actual.mu0,mu0[retained])
        for field,values in expected[arm].items():
            audit.compare('independent_raw_law.'+arm+'.'+field,actual[field],values[retained],
                          'cdf' if field == 'pit' or field.startswith('p_over_') else 'mean' if field == 'fixed_mean' else 'fp64')
        layer = law(actual.duration_log_mu.to_numpy(float),actual.duration_log_sigma.to_numpy(float),actual.duration_minutes.to_numpy(float))
        for field,values in layer.items():
            audit.compare('exported_law_evaluator.'+arm+'.'+field,actual[field],values,'mean' if field == 'fixed_mean' else 'fp64')
        audit.record('legal_quantiles.'+arm,bool(np.all(np.diff(np.column_stack([actual[q] for q in QUANTILES]),axis=1)>0)) and actual.q025.gt(0).all())
    quadrature_checks(audit,frame,expected)
    metrics = read_json(roots['08_reporting']/'metrics.json')
    bypartition = {p:{arm:metric_summary(values,y,mask) for arm,values in expected.items()} for p,mask in masks.items()}
    compare_mapping(audit,'metrics.ASSESS',metrics['ASSESS'],bypartition['ASSESS'])
    for p in ('TRAIN','CAL'):
        compare_mapping(audit,'metrics.'+p,metrics['partitions'][p],bypartition[p])
    assess = masks['ASSESS']
    profile_columns = {column:frame[column] for column in ('month','best_of','history','major_state','source_unseen_state','source_present')}
    profile_columns.update(gap_observed=frame[COLUMNS[1]].notna(),missing_player_count=frame[COLUMNS[2]])
    for column,membership in profile_columns.items():
        for value in membership.loc[assess].unique():
            mask = assess & membership.eq(value).to_numpy()
            compare_mapping(audit,f'profile.{column}.{value}',metrics['profiles'][column][str(value)],
                            {arm:metric_summary(values,y,mask) for arm,values in expected.items()})
    contrast_rows = pd.read_parquet(roots['08_reporting']/'contrast_inputs.parquet')
    audit.compare('contrast_input_ids',contrast_rows.match_id.astype(str),frame.loc[assess,'match_id'].astype(str),'exact')
    audit.record('contrast_input_complete',bool(contrast_rows.eligible.all()) and not contrast_rows.paired_absent.any())
    for arm in ('P0','PG','PC'):
        for score in ('nll','crps'):
            audit.compare('contrast_input.'+arm+'.'+score,contrast_rows[f'{arm}_{score}'],expected[arm][score][assess])
        audit.compare('contrast_input_missing.'+arm,contrast_rows[f'{arm}_missing'],np.zeros(assess.sum(),bool),'exact')
        for level,lo,hi in ((80,'q10','q90'),(50,'q25','q75')):
            covered = (y >= expected[arm][lo]) & (y <= expected[arm][hi])
            audit.compare('contrast_input_coverage.'+arm+'.'+str(level),contrast_rows[f'{arm}_covered{level}'],covered[assess],'exact')
    comparisons = read_json(roots['08_reporting']/'comparisons.json')
    guards = read_json(roots['08_reporting']/'guards.json')
    monthly,proxy = (frame.loc[assess,c].to_numpy(str) for c in ('month','proxy_block'))
    primary_passes,guard_passes,cohort = [],[],[]
    overall = bypartition['ASSESS']
    for slot,reference in ((1,'P0'),(2,'PG')):
        delta = expected['PC']['nll'][assess]-expected[reference]['nll'][assess]
        result,arrays = block_contrast(delta,monthly,MONTHS)
        passes = result['status'] == 'ESTIMABLE' and result['upper'] is not None and result['upper'] < -.002 and result['equal_block']['point'] < 0
        result.update(slot=slot,candidate='PC',reference=reference,metric='duration_density_nll_minutes',passed=passes)
        compare_mapping(audit,'primary.PC-'+reference,comparisons['PC-'+reference],result)
        verify_draw_export(audit,roots['08_reporting']/f'contrast_PC_{reference}.npz',delta,monthly,MONTHS,arrays)
        primary_passes.append(passes)
        sensitivity,parray = block_contrast(delta,proxy,sorted(set(proxy)),confidence=.95,min_blocks=1)
        compare_mapping(audit,'proxy.PC-'+reference,comparisons['proxy_PC-'+reference],sensitivity)
        verify_draw_export(audit,roots['08_reporting']/f'proxy_PC_{reference}.npz',delta,proxy,sorted(set(proxy)),parray)
        crps_delta = expected['PC']['crps'][assess]-expected[reference]['crps'][assess]
        crps,crps_arrays = block_contrast(crps_delta,monthly,MONTHS)
        compare_mapping(audit,'crps.PC-'+reference,comparisons['crps_PC-'+reference],crps)
        verify_draw_export(audit,roots['08_reporting']/f'crps_PC_{reference}.npz',crps_delta,monthly,MONTHS,crps_arrays)
        cohort.extend([overall['PC']['crps'] <= overall[reference]['crps'],
                       overall['PC']['coverage_error50']-overall[reference]['coverage_error50'] <= .005])
    cohort.extend([overall['P0']['coverage_error80']-overall['PC']['coverage_error80'] >= .01,
                   overall['PC']['coverage_error80'] <= overall['PG']['coverage_error80']])
    compare_mapping(audit,'cohort_guard',metrics,{'cohort_guard_passed':all(cohort)})
    names = {f'{group}/{ref}' for group in ('BO1','BO3','BO5','history_full','history_partial') for ref in ('P0','PG')}
    audit.record('ten_guard_family',set(guards) == names)
    groups = {f'BO{b}':frame.best_of.eq(b).to_numpy() for b in (1,3,5)}
    groups.update(history_full=frame[COLUMNS[0]].eq(16).to_numpy(),
                  history_partial=(frame[COLUMNS[0]].gt(0)&frame[COLUMNS[0]].lt(16)).to_numpy())
    for group,membership in groups.items():
        mask = membership[assess]
        for reference in ('P0','PG'):
            delta = expected['PC']['nll'][assess]-expected[reference]['nll'][assess]
            result,arrays = block_contrast(delta,monthly,MONTHS,mask=mask,one_sided=True,min_rows=100,min_blocks=4)
            support = membership & assess
            worsening = (metric_summary(expected['PC'],y,support)['coverage_error80'] -
                         metric_summary(expected[reference],y,support)['coverage_error80']) if support.any() else None
            passes = result['status'] == 'ESTIMABLE' and result['upper'] is not None and result['upper'] <= .01 and worsening is not None and worsening <= .025
            result.update(candidate='PC',reference=reference,group=group,coverage_error80_worsening=worsening,passed=passes)
            compare_mapping(audit,'guard.'+group+'/'+reference,guards[group+'/'+reference],result)
            verify_draw_export(audit,roots['08_reporting']/f'guard_{group}_{reference}.npz',delta[mask],monthly[mask],MONTHS,arrays)
            guard_passes.append(passes)
    decision = read_json(roots['08_reporting']/'decision.json')
    audit.record('scientific_decision',decision['status'] == ('DEVELOPMENT_CANDIDATE' if all(primary_passes+guard_passes+cohort) else 'NO_ADOPTION'))
    audit.record('no_production_adoption',decision['production_adoption'] is False)
    cases = index['public_cases']
    if isinstance(cases,dict):
        cases = cases.get('cases',[])
    public_checks(audit,model_path,frame,masks,expected,mu0,sigma0,cases,output)
    audit.calculations['raw_saved_member_laws'] = int(retained.sum())*3
    return {'1':comparisons['PC-P0']['status'],'2':comparisons['PC-PG']['status']},{k:v['status'] for k,v in guards.items()}


def verify_prepared(audit, roots, source, masks, z, stats, mu0, sigma0, prepare):
    frame = pd.read_parquet(roots['04_feature']/'prepared.parquet')
    audit.pin(roots['04_feature']/'prepared.parquet',prepare['prepared_sha256'])
    audit.pin(roots['04_feature']/'normalizer.json',prepare['normalizer_sha256'])
    audit.compare('prepared_ids',frame.match_id.astype(str),source.match_id.astype(str),'exact')
    audit.compare('prepared_row_index',frame.row_index,np.arange(len(source)),'exact')
    partition = np.full(len(source),'EXCLUDED',dtype=object)
    for p,mask in masks.items():
        partition[mask] = p
        audit.compare('prepare_population.'+p,prepare['population_ids'][p],source.loc[mask,'match_id'].astype(str),'exact')
    audit.compare('prepared_partition',frame.partition,partition,'exact')
    for column in source:
        if column in ('roster_a','roster_b'):
            audit.compare('prepared_input.'+column,[json.dumps(list(map(str,x))) for x in frame[column]],
                          [json.dumps(list(map(str,x))) for x in source[column]],'exact')
        elif pd.api.types.is_numeric_dtype(source[column]):
            audit.compare('prepared_input.'+column,frame[column],source[column])
        else:
            audit.compare('prepared_input.'+column,frame[column].fillna('EMPTY').astype(str),source[column].fillna('EMPTY').astype(str),'exact')
    audit.compare('prepared_mu0',frame.mu0,mu0)
    audit.compare('prepared_sigma0',frame.sigma0,np.full(len(frame),sigma0))
    audit.compare('prepared_z',frame[['z0','z1','z2']],z)
    history = np.select([source[COLUMNS[0]].eq(16),source[COLUMNS[0]].between(0,16,inclusive='neither'),source[COLUMNS[0]].eq(0)],
                        ['full','partial','cold'],default='UNKNOWN')
    audit.compare('history_membership',frame.history,history,'exact')
    audit.compare('month_membership',frame.month,pd.to_datetime(source.date).dt.strftime('%Y-%m'),'exact')
    audit.compare('proxy_membership',frame.proxy_block,source.tournament.fillna('UNKNOWN').astype(str)+'/'+pd.to_datetime(source.date).dt.year.astype(str),'exact')
    protocol = read_json(roots['05_model_input']/'protocol.json')
    major,unseen = independent_taxonomy(source,protocol,roots['05_model_input'].parents[3],audit)
    audit.compare('major_state',frame.major_state,major,'exact')
    audit.compare('source_unseen_state',frame.source_unseen_state,unseen,'exact')
    raw = np.column_stack((np.log1p(source[COLUMNS[0]]),source[COLUMNS[1]].fillna(stats['gap_median']),source[COLUMNS[2]]))[masks['TRAIN']]
    normalizer = read_json(roots['04_feature']/'normalizer.json')
    wanted = dict(columns=list(COLUMNS),gap_median=stats['gap_median'],mean=stats['mean'],scale=stats['scale'],ddof=0,
                  train_ids_sha256=hashlib.sha256('\n'.join(source.loc[masks['TRAIN'],'match_id'].astype(str)).encode()).hexdigest(),
                  n_rows=int(masks['TRAIN'].sum()),counts=np.isfinite(raw).sum(0),sums=raw.sum(0),
                  sums_of_squares=(raw*raw).sum(0),rank=stats['rank'],raw_variance=raw.var(0),
                  observed_gap_count=int(source.loc[masks['TRAIN'],COLUMNS[1]].notna().sum()),dtype='float64',
                  clipping=[-3,3],missingness_unstandardized=True)
    observed_raw = raw.copy()
    observed_raw[:,1] = source.loc[masks['TRAIN'],COLUMNS[1]].to_numpy(float)
    wanted.update(observed_counts=np.isfinite(observed_raw).sum(0),observed_sums=np.nansum(observed_raw,axis=0),
                  observed_sums_of_squares=np.nansum(observed_raw**2,axis=0),
                  zero_variance_handling='HARD_STOP',rounding='float64 throughout')
    compare_mapping(audit,'normalizer',normalizer,wanted)
    memberships = pd.read_parquet(roots['04_feature']/'memberships.parquet')
    for column in memberships:
        if column in frame:
            audit.compare('membership_ledger.'+column,memberships[column].fillna('EMPTY').astype(str),frame[column].fillna('EMPTY').astype(str),'exact')
    rule_receipt = read_json(roots['04_feature']/'membership_rules.json')
    for column,definition in rule_receipt.items():
        audit.record('membership_rule_hash.'+column,definition['rule_sha256'] == hashlib.sha256(json.dumps(definition['rule'],sort_keys=True).encode()).hexdigest())
        audit.record('membership_rule_source.'+column,definition['source_snapshot_sha256'] == digest(roots['05_model_input'].parents[3]/read_json(roots['05_model_input']/'binding.json')['dataset_path']))
        unknown_count = int(frame[column].isna().sum()+frame[column].eq('UNKNOWN').sum())
        audit.compare('membership_unknown.'+column,definition['unknown_count'],unknown_count,'exact')
        for state,counts in definition['states'].items():
            membership = frame[column].astype(str).eq(state)
            compare_mapping(audit,'membership_counts.'+column+'.'+state,counts,
                            {'n_series':int(membership.sum()),'month_counts':{month:int((membership & frame.month.eq(month)).sum()) for month in MONTHS}})
    ledger = pd.read_parquet(roots['04_feature']/'series_ledger.parquet')
    audit.compare('series_ledger_ids',ledger.match_id.astype(str),source.match_id.astype(str),'exact')
    for column in ('source_match_id','target_origin_day','effective_release_day','partition','best_of','team_a_id','team_b_id','duration_minutes','scenario'):
        original = {'source_match_id':'match_id','target_origin_day':'date','effective_release_day':'release_day','team_a_id':'team_a','team_b_id':'team_b'}.get(column,column)
        wanted = frame[original] if original in frame else np.full(len(frame),'historical_actual_first_map_roster_daily_release')
        audit.compare('series_ledger.'+column,ledger[column].astype(str),pd.Series(wanted).astype(str),'exact')
    with np.load(roots['04_feature']/'train_design.npz',allow_pickle=False) as design:
        for key,wanted in [('match_id',source.loc[masks['TRAIN'],'match_id'].astype(str).to_numpy()),('z',z[masks['TRAIN']]),
                           ('mu0',mu0[masks['TRAIN']]),('y',source.duration_minutes.to_numpy(float)[masks['TRAIN']]),
                           ('gap_observed',source.loc[masks['TRAIN'],COLUMNS[1]].notna().to_numpy())]:
            audit.compare('train_design.'+key,design[key],wanted,'exact' if key in ('match_id','gap_observed') else 'fp64')
    return frame


def verify_index(audit, repo, index, oracle, protocol_path, binding_path):
    for field in oracle['artifact_schema']['index_required_fields']:
        audit.record('index_required.'+field,field in index)
    audit.compare('index_protocol',index['protocol_sha256'],digest(protocol_path),'exact')
    audit.compare('index_binding',index['binding_sha256'],digest(binding_path),'exact')
    for number,descriptor in enumerate(index['artifact_descriptors']):
        for field in oracle['artifact_schema']['artifact_descriptor_required_fields']:
            audit.record(f'descriptor[{number}].'+field,field in descriptor)
        if descriptor['absence_status'] is not None:
            audit.record(f'descriptor_absence[{number}]',bool(descriptor['absence_reason']) and descriptor['sha256'] is None)
            continue
        relative = descriptor['repo_relative_path']
        path = (repo/relative).resolve() if relative else Path(descriptor['external_absolute_path']).resolve()
        if relative and not path.is_relative_to(repo):
            raise ValueError('descriptor path leaves repo')
        audit.pin(path,descriptor['sha256'])
        if descriptor['format'] == 'parquet' and descriptor['shape_or_row_count'] is not None:
            import pyarrow.parquet as pq
            audit.compare(f'descriptor_rows[{number}]',descriptor['shape_or_row_count'],pq.read_metadata(path).num_rows,'exact')
            audit.compare(f'descriptor_columns[{number}]',descriptor['dtype_or_columns'],pq.read_schema(path).names,'exact')
        if descriptor['format'] == 'npz' and descriptor['shape_or_row_count'] is not None:
            with np.load(path,allow_pickle=False) as arrays:
                for key in arrays.files:
                    audit.compare(f'descriptor_npz_shape[{number}].'+key,descriptor['shape_or_row_count'][key],list(arrays[key].shape),'exact')
                    audit.compare(f'descriptor_npz_dtype[{number}].'+key,descriptor['dtype_or_columns'][key],str(arrays[key].dtype),'exact')


def verify_partial(audit, roots, frame, masks, z, mu0, sigma0, index):
    fit = read_json(roots['08_reporting']/'fit_receipts.json')
    trace = fit['operation_trace']
    verify_operation_trace(audit,trace,False)
    audit.compare('partial_count',fit['estimation_operations'],len(trace),'exact')
    audit.record('partial_failed_stop',bool(trace) and trace[-1]['status'] == 'FAILED' and all(r['status'] == 'SUCCESS' for r in trace[:-1]))
    successful = {r['stage']:r for r in fit['receipts']}
    y = frame.duration_minutes.to_numpy(float)
    sigma_t = float(np.sqrt(np.mean((np.log(y[masks['TRAIN']])-mu0[masks['TRAIN']])**2)))
    if 'TRAIN_REFERENCE_SIGMA' in successful:
        audit.compare('partial_sigma_T',successful['TRAIN_REFERENCE_SIGMA']['parameters'],[sigma_t])
    if 'TRAIN_HEAD' in successful:
        head = successful['TRAIN_HEAD']
        theta = np.array(head['parameters'])
        verify_gradient(audit,'partial_TRAIN_HEAD',theta,np.column_stack((np.ones(masks['TRAIN'].sum()),z[masks['TRAIN']])),
                        mu0[masks['TRAIN']],y[masks['TRAIN']],sigma_t,np.array([0.,.1,.1,.1]),head,(-1.,1.))
    if 'PG_CAL' in successful:
        scalar_verification(audit,'partial_PG_CAL',successful['PG_CAL']['parameters'][0],z[masks['CAL']],np.zeros(3),
                            mu0[masks['CAL']],y[masks['CAL']],sigma0,successful['PG_CAL'])
    if 'PC_CAL' in successful:
        scalar_verification(audit,'partial_PC_CAL',successful['PC_CAL']['parameters'][0],z[masks['CAL']],
                            np.array(successful['TRAIN_HEAD']['parameters'][1:]),mu0[masks['CAL']],y[masks['CAL']],sigma0,successful['PC_CAL'])
    decision = read_json(roots['08_reporting']/'decision.json')
    audit.record('partial_no_adoption',decision['status'] == 'NO_ADOPTION')
    audit.record('partial_slots_not_run',all(decision['primary_slots'].get(str(s)) in ('NOT_RUN','NOT_RUN_FIT_FAILED') for s in (1,2)))
    audit.not_run.append('Public fitted inference/assessment: incomplete fixed recipe; no fabricated forward proof')
    return {'1':'NOT_RUN_FIT_FAILED','2':'NOT_RUN_FIT_FAILED'},{}


def run(args, audit, output):
    repo = args.repo_root.resolve()
    roots = {stage:repo/'data'/stage/'duration_scale_reliability'/args.run_id for stage in ('04_feature','05_model_input','06_models','07_model_output','08_reporting')}
    protocol_path,binding_path = roots['05_model_input']/'protocol.json',roots['05_model_input']/'binding.json'
    protocol,binding = read_json(protocol_path),read_json(binding_path)
    evidence = repo/protocol['evidence_root']
    oracle_path = evidence/'oracle_contract.json'
    oracle = read_json(oracle_path)
    contract = read_json(evidence/'p1_contract.json')['resolved_inputs']
    audit.record('run_identity',protocol['run_id'] == args.run_id and binding['run_id'] == args.run_id and oracle['run_id'] == args.run_id)
    audit.record('research_root',Path(protocol['research_root']).resolve() == args.research_root.resolve())
    audit.pin(protocol_path,binding['protocol_sha256'])
    audit.record('verifier_bound_before_fit','scripts/verify_duration_scale_reliability.py' in binding['pins'])
    for path,pin in binding['pins'].items():
        audit.pin(repo/path,pin)
    compare_mapping(audit,'frozen_optimizer',protocol['optimizer'],
                    {'TRAIN':{'method':'L-BFGS-B','maxiter':500,'ftol':1e-12,'gtol':1e-8},
                     'CAL':{'method':'bounded','maxiter':500,'xatol':1e-8}})
    compare_mapping(audit,'frozen_uncertainty',protocol,{'month_universe':list(MONTHS),'draws':10000,'seed':20261009,'family_slots':10,'primary_slots':[1,2],'operation_trace':list(OPERATIONS)})
    authorization = read_json(evidence/'authorization_and_initial_pins.json')
    protected = {}
    for path,pin in authorization['protected_before'].items():
        actual = audit.pin(repo/path,pin)
        protected[path] = {'before':pin,'after':actual,'unchanged':pin == actual}
    frame = pd.read_parquet(repo/contract['dataset'])
    audit.record('input_identity_unique',not frame.match_id.isna().any() and not frame.match_id.astype(str).duplicated().any())
    masks = partitions(frame)
    populations = {p:int(mask.sum()) for p,mask in masks.items()}
    audit.compare('canonical_partition_counts',list(populations.values()),[36205,1328,3103],'exact')
    for column in ('max_history_release_day','max_pace_release_day'):
        previous = pd.to_datetime(frame[column],errors='raise')
        audit.record('exclusive_clock.'+column,bool((previous.isna() | (previous < pd.to_datetime(frame.date))).all()))
    audit.record('label_release_clock',bool((pd.to_datetime(frame.release_day) >= pd.to_datetime(frame.date)).all()))
    reconstruct_sources(repo,contract,frame,audit)
    prepare = read_json(roots['05_model_input']/'prepare_receipt.json')
    index_path = roots['08_reporting']/'oracle_inputs.json'
    index = read_json(index_path) if index_path.exists() else None
    if index is not None:
        verify_index(audit,repo,index,oracle,protocol_path,binding_path)
        audit.compare('oracle_contract_hash',index['oracle_contract_sha256'],digest(oracle_path),'exact')
    independent_invalid = None
    try:
        retained = np.logical_or.reduce(list(masks.values()))
        if not np.isfinite(frame.duration_minutes.to_numpy(float)[retained]).all() or np.any(frame.duration_minutes.to_numpy(float)[retained] <= 0):
            raise ValueError('positive finite duration labels')
        z,stats = normalize(frame,masks['TRAIN'])
    except ValueError as exc:
        independent_invalid = str(exc)
    if prepare['status'] == 'INVALID_PREPARE':
        audit.record('closure_independent_cause',independent_invalid is not None,
                     reason=independent_invalid or 'No independently reconstructed input/design failure')
        audit.compare('closure_counter',prepare['estimation_operations'],0,'exact')
        audit.record('closure_index_present',index is not None)
        fit = read_json(roots['08_reporting']/'fit_receipts.json')
        audit.record('closure_no_operations',fit['operation_trace'] == [] and fit['estimation_operations'] == 0)
        decision = read_json(roots['08_reporting']/'decision.json')
        audit.record('closure_no_adoption',decision['status'] == 'NO_ADOPTION')
        audit.record('closure_reserved_slots',all(decision['primary_slots'].get(str(s)) in ('NOT_RUN','NOT_RUN_BLOCKED') for s in (1,2)))
        audit.record('closure_no_candidate',not (roots['06_models']/'duration_scale.joblib').exists() and not (roots['07_model_output']/'duration_predictions.parquet').exists())
        if independent_invalid is not None:
            categories = ('variance','rank','positive','gap','count','missing')
            expected_categories = {key for key in categories if key in independent_invalid.lower()}
            audit.record('closure_reason_consistency',bool(expected_categories) and
                         any(key in str(decision['reason']).lower() for key in expected_categories))
        if index is not None:
            for p,mask in masks.items():
                audit.compare('closure_population.'+p,index['population_ids'].get(p,[]),frame.loc[mask,'match_id'].astype(str),'exact')
        inventory = read_json(roots['04_feature']/'raw_inventory.json')
        audit.pin(roots['04_feature']/'raw_closure.parquet',inventory['raw_closure_sha256'])
        compare_mapping(audit,'closure_inventory',inventory,{'counts':populations,'estimation_operations':0})
        raw_closure = pd.read_parquet(roots['04_feature']/'raw_closure.parquet')
        audit.compare('closure_inventory_ids',raw_closure.match_id.astype(str),frame.match_id.astype(str),'exact')
        for p,mask in masks.items():
            audit.compare('closure_inventory_partition.'+p,raw_closure.partition.eq(p),mask,'exact')
        for column in COLUMNS:
            audit.compare('closure_raw_feature.'+column,raw_closure[column],frame[column])
        audit.not_run.extend(['Fitted forward laws, objectives, scalar optimum, public saved model, score/CI gates: no-fit closure'])
        return dict(population_counts=populations,primary_slot_statuses={'1':'NOT_RUN_BLOCKED','2':'NOT_RUN_BLOCKED'},
                    guard_statuses={f'{group}/{ref}':'NOT_RUN_BLOCKED' for group in ('BO1','BO3','BO5','history_full','history_partial') for ref in ('P0','PG')},
                    protected_before_after=protected,oracle_contract_sha256=digest(oracle_path),
                    formal_operation_counts=0,closure=True)
    if independent_invalid:
        raise ValueError('Prepared invalid independent design: '+independent_invalid)
    bundle = joblib.load(repo/contract['selected_bundle'])
    audit.record('selected_tree_identity',bundle.selected['duration'] == 'tree_leaves_15' and list(bundle.feature_columns) == binding['feature_columns'])
    tree = bundle.models['tree_leaves_15']
    audit.record('selected_tree_family',tree.family == 'tree')
    mu0,sigma0 = replay_tree(tree,frame,binding['feature_columns'])
    audit.compare('sigma0',sigma0,contract['old_sigma0'])
    for p,key in [('CAL','old_cal_laws'),('ASSESS','old_assess_laws')]:
        old = pd.read_parquet(repo/contract[key])
        old = old.loc[old.candidate.eq('tree_leaves_15')]
        audit.compare('old_P0_ids.'+p,old.match_id.astype(str),frame.loc[masks[p],'match_id'].astype(str),'exact')
        audit.compare('old_P0_mu.'+p,old.duration_log_mu,mu0[masks[p]])
        audit.compare('old_P0_sigma.'+p,old.duration_log_sigma,np.full(masks[p].sum(),sigma0))
    old_scores = pd.read_parquet(repo/contract['old_tree_scores'])
    audit.compare('old_P0_score_ids',old_scores.match_id.astype(str),frame.loc[masks['ASSESS'],'match_id'].astype(str),'exact')
    baseline_scores = law(mu0[masks['ASSESS']],np.full(masks['ASSESS'].sum(),sigma0),frame.loc[masks['ASSESS'],'duration_minutes'].to_numpy(float))
    audit.compare('old_P0_nll',old_scores.duration_nll,baseline_scores['nll'])
    if 'duration_proper_secondary' in old_scores:
        audit.compare('old_P0_crps',old_scores.duration_proper_secondary,baseline_scores['crps'])
    prepared_frame = verify_prepared(audit,roots,frame,masks,z,stats,mu0,sigma0,prepare)
    partial = False
    if args.mode == 'prepare':
        audit.record('prepare_counter_zero',prepare['estimation_operations'] == 0)
        audit.record('prepare_no_formal_attempts',not (roots['08_reporting']/'operation_01_started.json').exists())
        audit.not_run.append('Completed candidate/public/score/CI verification: prepare-only mode')
        slots,guards,counter = {'1':'NOT_RUN_PREPARE','2':'NOT_RUN_PREPARE'},{},0
    else:
        if index is None:
            raise ValueError('completed oracle_inputs.json missing')
        fit = read_json(roots['08_reporting']/'fit_receipts.json')
        counter = fit['estimation_operations']
        for i,operation in enumerate(fit['operation_trace']):
            started = read_json(roots['08_reporting']/f'operation_{i+1:02d}_started.json')
            audit.record(f'count_before_invocation[{i}]',started['status'] == 'STARTED' and started['counter_after'] == i+1 and
                         started['finished_at_utc'] is None and started['parameters_sha256'] is None)
            audit.compare(f'operation_input_ids[{i}]',operation['input_ids_sha256'],
                          hashlib.sha256('\n'.join(frame.loc[masks[operation['partition']],'match_id'].astype(str)).encode()).hexdigest(),'exact')
        if any(row['status'] == 'FAILED' for row in fit['operation_trace']):
            partial = True
            slots,guards = verify_partial(audit,roots,prepared_frame,masks,z,mu0,sigma0,index)
        else:
            approval = read_json(roots['05_model_input']/'prefit_approval.json')
            audit.record('prefit_approval',approval['status'] == 'APPROVED' and approval['protocol_sha256'] == digest(protocol_path) and approval['binding_sha256'] == digest(binding_path))
            slots,guards = verify_completed(audit,roots,prepared_frame,masks,z,mu0,sigma0,index,output)
    return dict(population_counts=populations,primary_slot_statuses=slots,guard_statuses=guards,formal_operation_counts=counter,
                protected_before_after=protected,oracle_contract_sha256=digest(oracle_path),closure=False,partial=partial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--research-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh directory; exclusive creation')
    parser.add_argument('--mode',choices=('prepare','completed'),default='completed')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_id) or args.run_id in ('.','..'):
        parser.error('run-id must be a safe single component')
    args.output.mkdir(parents=True,exist_ok=False)
    audit = Audit()
    receipt = dict(schema_version=1,run_id=args.run_id,study='P1',execution_created_at_utc=datetime.now(timezone.utc).isoformat(),
                   verifier_code_hashes={str(Path(__file__).resolve()):digest(__file__)},
                   hardware_and_threads={'cpu_only':True,'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}},
                   environment_versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','pandas','joblib')},
                   scientific_limitations=['Exposed historical development, not authentic issuance, causal identification, betting edge or deployment readiness.',
                                           'Pinned source and saved selected tree are shared dependencies. No new base fit or candidate replacement.'])
    receipt.update(oracle_contract_sha256=None,formal_operation_counts=None,population_counts={},
                   primary_slot_statuses={'1':'NOT_VERIFIED','2':'NOT_VERIFIED'},guard_statuses={},
                   protected_before_after={})
    try:
        receipt.update(run(args,audit,args.output))
    except Exception as exc:
        audit.record('required_verification_exception',False,reason=f'{type(exc).__name__}: {exc}')
    receipt.update(status=('PASS_CLOSURE_CONTRACT' if receipt.get('closure') else 'PASS') if not audit.failures else 'FAIL',
                   fitted_prediction_oracle_status='NOT_RUN_NO_FIT' if receipt.get('closure') else 'NOT_RUN_PREPARE' if args.mode == 'prepare' else 'NOT_RUN_FIT_FAILED' if receipt.get('partial') else 'FAIL' if audit.failures else 'PASS',
                   input_and_output_hashes=audit.hashes,
                   implementation_hashes={p:h for p,h in audit.hashes.items() if '/src/' in p},
                   scope_exercised=sorted(set(check['kind'] for check in audit.checks)),
                   scope_not_exercised=audit.not_run,independent_calculation_counts=dict(audit.calculations),
                   raw_state_count=audit.calculations.get('raw_saved_member_laws',0),
                   membership_counts={check['check_id']:check.get('compared_values',0) for check in audit.checks if 'membership' in check['check_id']},
                   month_block_counts={'universe':list(MONTHS),'n_blocks':9},missingness_statuses={'required_mismatches':sum(c['missing_mask_mismatches'] for c in audit.checks)},
                   checks=audit.checks,failures=audit.failures,
                   comparisons_by_kind=dict(Counter(c['kind'] for c in audit.checks)),
                   max_absolute_and_relative_differences={kind:{'absolute':max((c['maximum_absolute_difference'] for c in audit.checks if c['kind'] == kind),default=0.),
                                                                  'relative':max((c['maximum_relative_difference'] for c in audit.checks if c['kind'] == kind),default=0.)} for kind in TOLERANCES})
    with (args.output/'verification.json').open('x') as handle:
        json.dump(clean(receipt),handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status':receipt['status'],'failures':len(audit.failures),'output':str(args.output/'verification.json')}))
    return 1 if audit.failures else 0




def independent_taxonomy(frame, protocol, repo, audit):
    """Read the frozen declarative rule table as data, never import its code."""
    import ast
    import unicodedata
    source = ast.parse((repo/'src/models/competition_tiers.py').read_text())
    definitions = []
    for node in source.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id == '_RULES':
            definitions = node.value.elts
            break
    if not definitions:
        raise ValueError('frozen declarative competition rules missing')
    rules = []
    for rule in definitions:
        pattern = ast.literal_eval(rule.args[0])
        tier = rule.args[2].attr.lower()
        clocks = {}
        for keyword in rule.keywords:
            if keyword.arg in ('not_before','before'):
                value = keyword.value
                clocks[keyword.arg] = datetime(*[ast.literal_eval(x) for x in value.args]).date() if isinstance(value,ast.Call) else None
        rules.append((re.compile(pattern),tier,clocks))
    major = []
    for name,date in zip(frame.tournament,frame.date):
        text = unicodedata.normalize('NFKD',str(name))
        normalized = re.sub('[^a-z0-9]+',' ',''.join(c for c in text if not unicodedata.combining(c)).casefold()).strip()
        tier = 'unknown'
        if pd.notna(name) and normalized not in ('','na','nan','nat','none','null'):
            origin = pd.Timestamp(date).date()
            for pattern,match_tier,clocks in rules:
                if clocks.get('not_before') is not None and origin < clocks['not_before']:
                    continue
                if clocks.get('before') is not None and origin >= clocks['before']:
                    continue
                if pattern.search(normalized):
                    tier = match_tier
                    break
        major.append('UNKNOWN' if tier == 'unknown' else 'major' if tier == 'major' else 'nonmajor')
    taxonomy_path = Path(protocol['taxonomy_metadata_path'])
    audit.pin(taxonomy_path,protocol['taxonomy_metadata_sha256'])
    taxonomy = pd.read_parquet(taxonomy_path,columns=['golgg_match_id','unknown_players'])
    audit.record('taxonomy_unique_ids',not taxonomy.golgg_match_id.astype(str).duplicated().any())
    by_id = dict(zip(taxonomy.golgg_match_id.astype(str),taxonomy.unknown_players))
    unseen = []
    for mid in frame.match_id.astype(str):
        value = by_id.get(mid,math.nan)
        unseen.append('UNKNOWN' if pd.isna(value) else 'unseen' if value > 0 else 'seen' if value == 0 else 'UNKNOWN')
    return np.array(major),np.array(unseen)


if __name__ == '__main__':
    raise SystemExit(main())
