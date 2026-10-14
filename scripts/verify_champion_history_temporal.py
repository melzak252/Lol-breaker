#!/usr/bin/env python3
"""Independent raw-source receipts for the champion-history research run.

The membership oracle retains every eligible observation, filters each player's
complete raw ledger by whole-series release, and then takes the last 16. It never
calls the production source reader or preparation function. The preparation
function is used only as the system under test on isolated physical mutations.
The established Gaussian rating replay is deliberately reused and identified.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import traceback

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_prospective_features import iter_matches
from src.models import prematch_features as rating

RUN_ID = 'champion_history_20261007_01'
SOURCE = ROOT / 'data/artifacts/golgg-database-recovery-20260909/matches.json'
BASE = ROOT / 'data/04_feature/prop_distribution/prop_distribution_20261007_01/dataset.parquet'
RELEASES = ROOT / 'data/03_primary/ground_up_lol_20260927_run001/release_days.json'
RULES = ROOT / 'data/05_model_input/draft_prop/draft_prop_20261007_01/rules.json'
TRAIN_END, CAL_END = '2025-07-01', '2026-01-01'
INPUTS = ('history', 'champions', 'patches', 'competitions', 'roles', 'mask',
          'players', 'role_probs', 'context', 'best_of')
STATS = ('kills', 'deaths', 'assists', 'csm', 'gpm', 'dpm', 'vspm', 'gd@15',
         'csd@15', 'xpd@15', 'kp%', 'dmg%')
STAT_UNITS = dict(zip(STATS, ('count', 'count', 'count', 'creep score/minute',
    'gold/minute', 'damage/minute', 'vision score/minute', 'signed gold difference at 15 minutes',
    'signed creep score difference at 15 minutes', 'signed experience difference at 15 minutes',
    'kill participation percent', 'damage share percent')))
ROLES = ('TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT')
ALIASES = {'top': 1, 'toplane': 1, 'jungle': 2, 'jungler': 2, 'jg': 2, 'jng': 2,
           'mid': 3, 'middle': 3, 'midlane': 3, 'adc': 4, 'bot': 4, 'bottom': 4,
           'adcarry': 4, 'marksman': 4, 'support': 5, 'sup': 5}
MISSING_TAGS = {'', '0', 'unknown', 'none', 'null', 'nan', '<na>', 'n/a', '?'}
FLAGS = ('remake', 'is_remake', 'cancelled', 'canceled', 'forfeit', 'is_forfeit')
CODE_FILES = ('scripts/verify_champion_history_temporal.py', 'src/features/champion_history_data.py',
              'src/models/champion_history_model.py', 'src/models/champion_history_runner.py',
              'src/models/prematch_features.py', 'scripts/export_prospective_features.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


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
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    return value


def save(path, receipt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(receipt), indent=2, allow_nan=False) + '\n')


class Checks:
    def __init__(self):
        self.items = {}

    def equal(self, name, actual, expected, where=None, atol=0.):
        a, e = np.asarray(actual), np.asarray(expected)
        if a.shape != e.shape:
            self.require(name, False, {'where': where, 'actual_shape': a.shape, 'expected_shape': e.shape})
            return
        numeric = a.dtype.kind in 'biuf' and e.dtype.kind in 'biuf'
        if numeric:
            missing = np.isnan(a) if a.dtype.kind == 'f' else np.zeros(a.shape, bool)
            expected_missing = np.isnan(e) if e.dtype.kind == 'f' else np.zeros(e.shape, bool)
            valid = ~missing & ~expected_missing
            error = np.abs(a.astype(np.float64)[valid] - e.astype(np.float64)[valid])
            bad = int(np.count_nonzero(missing != expected_missing) + np.count_nonzero(error > atol))
            maximum = float(error.max(initial=0.))
        else:
            bad = int(np.count_nonzero(a != e))
            maximum = 0.
        item = self.items.setdefault(name, {'compared_values': 0, 'mismatched_values': 0, 'max_abs_error': 0.})
        item['absolute_tolerance'] = max(item.get('absolute_tolerance', 0.), float(atol))
        item['compared_values'] += int(a.size)
        item['mismatched_values'] += bad
        item['max_abs_error'] = max(item['max_abs_error'], maximum)
        if bad and 'first_failure' not in item:
            item['first_failure'] = clean(where)

    def require(self, name, condition, where=None):
        self.equal(name, bool(condition), True, where)

    @property
    def passed(self):
        return all(item['mismatched_values'] == 0 for item in self.items.values())


def identity(value):
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, np.integer)):
        raise ValueError(f'Non-explicit identity: {value!r}')
    return str(value)


def tag(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    return None if text.lower() in MISSING_TAGS else text


def role(value):
    return ALIASES.get(''.join(c for c in str(value).lower() if c.isalnum()), 0)


def number(value):
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        result = float(value.strip().replace(',', '').replace('%', '') if isinstance(value, str) else value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) and abs(result) <= np.finfo(np.float32).max else math.nan


def label_number(value):
    if value is None or isinstance(value, str) and value.strip().lower() in {'', '-', '--', 'n/a', 'na', 'null', 'none', 'nan', '?'}:
        return math.nan, 'missing'
    if isinstance(value, bool):
        return math.nan, 'invalid_numeric'
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan, 'invalid_numeric'
    return (result, 'valid') if math.isfinite(result) else (math.nan, 'invalid_numeric')


def excluded(raw):
    result = [key for key in FLAGS if raw.get(key) is True or raw.get(key) == 1]
    status = str(raw.get('status', '')).strip().lower()
    if status in {'remake', 'cancelled', 'canceled', 'forfeit'}:
        result.append('status=' + status)
    return result


def side_kills(game, side):
    payload = game.get(side + '_stats')
    value, reason = label_number(payload.get('kills') if isinstance(payload, dict) else None)
    if reason == 'valid' and (value < 0 or not value.is_integer()):
        value, reason = math.nan, 'invalid_count'
    return value, 'valid' if reason == 'valid' else side + '_kills_' + reason


def map_labels(raw, game):
    flags = excluded(raw) or excluded(game)
    if flags:
        reason = ('series_excluded:' if excluded(raw) else 'map_excluded:') + ','.join(flags)
        return (math.nan, math.nan, reason, reason)
    a, ar = side_kills(game, 't1')
    b, br = side_kills(game, 't2')
    duration, dr = label_number(game.get('game_duration'))
    if dr != 'valid':
        dr = 'duration_' + dr
    elif duration <= 0:
        duration, dr = math.nan, 'duration_nonpositive'
    return a + b, duration / 60., ';'.join(r for r in (ar, br) if r != 'valid') or 'valid', dr


def integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean integer')
    parsed = float(value)
    if not math.isfinite(parsed) or parsed != math.floor(parsed):
        raise ValueError('Not integral')
    return int(parsed)


def declared_integer(raw, aliases):
    values = {integer(raw[key]) for key in aliases if raw.get(key) is not None}
    if len(values) > 1:
        raise ValueError('Conflicting integer aliases')
    return next(iter(values), None)


def score_label(raw, teams, bo):
    games = raw['games']
    if bo not in (1, 3, 5):
        return math.nan, 'unsupported_best_of'
    if not games:
        return math.nan, 'maps_missing'
    if excluded(raw) or any(excluded(game) for game in games):
        return math.nan, 'series_or_map_excluded_flag'
    try:
        initial = raw.get('initial_wins')
        if initial is not None and (not isinstance(initial, dict) or any(str(t) not in teams or integer(w) != 0 for t, w in initial.items())):
            return math.nan, 'map_advantage'
        for key in ('map_advantage', 'map_advantages', 'initial_map_advantage'):
            value = raw.get(key)
            if value not in (None, 0, '0', {}, []) and not (isinstance(value, dict) and all(integer(v) == 0 for v in value.values())):
                return math.nan, 'map_advantage'
    except (TypeError, ValueError):
        return math.nan, 'map_advantage'
    threshold = bo // 2 + 1
    if not threshold <= len(games) <= bo:
        return math.nan, 'illegal_terminal_map_count'
    if raw.get('games_played') is not None:
        try:
            if integer(raw['games_played']) != len(games):
                return math.nan, 'declared_map_count_mismatch'
        except (TypeError, ValueError):
            return math.nan, 'declared_map_count_invalid'
    wins = [0, 0]
    for game in games:
        a, b = game.get('t1_win'), game.get('t2_win')
        if a not in (True, False, 0, 1) or b not in (True, False, 0, 1) or a == b or game.get('draw'):
            return math.nan, 'map_winner_invalid'
        if max(wins) >= threshold:
            return math.nan, 'maps_after_deciding_win'
        own = a if str(game['t1_id']) == teams[0] else b
        wins[0 if own else 1] += 1
    if max(wins) != threshold or raw.get('draw'):
        return math.nan, 'series_incomplete_or_draw'
    header_a, header_b = raw.get('tid_1', raw.get('t1_id')), raw.get('tid_2', raw.get('t2_id'))
    if header_a is not None and header_b is not None:
        for aliases, team in ((('score_1', 't1_score'), str(header_a)), (('score_2', 't2_score'), str(header_b))):
            try:
                declared = declared_integer(raw, aliases)
            except (TypeError, ValueError):
                return math.nan, 'declared_series_score_invalid'
            if declared is not None and declared != wins[teams.index(team)]:
                return math.nan, 'declared_series_score_mismatch'
        for key, team in (('t1_win', str(header_a)), ('t2_win', str(header_b))):
            value = raw.get(key)
            if value is not None:
                if value not in (True, False, 0, 1):
                    return math.nan, 'declared_series_winner_invalid'
                if bool(value) != (wins[teams.index(team)] > wins[1 - teams.index(team)]):
                    return math.nan, 'declared_series_winner_mismatch'
    winner = int(wins[1] > wins[0])
    return float(winner * 3 + wins[1 - winner]), 'valid'


@dataclass(slots=True)
class Event:
    gid: str
    index: int
    played: int
    players: tuple
    champs: tuple
    status: np.ndarray
    roles: np.ndarray
    values: np.ndarray
    patch: str | None
    competition: str | None
    labels: tuple
    eligible: bool
    rosters: tuple
    ambiguous_slots: tuple = ()
    opponent: np.ndarray = field(default_factory=lambda: np.full((2, 2), np.nan, np.float32))


@dataclass(slots=True)
class Record:
    mid: str
    day: date
    release: date
    bo: int
    teams: tuple
    events: list
    score: float
    score_reason: str
    rating_record: object
    rank: int = 0


def key(mid):
    return (0, int(mid)) if mid.isascii() and mid.isdecimal() else (1, mid)


def parse_source(base, checks):
    canonical = {str(row.match_id): row for row in base.itertuples(index=False)}
    overrides = json.loads(RELEASES.read_text())
    records, counts, examples = [], Counter(), defaultdict(list)
    observed, minimum, maximum = Counter(), {}, {}
    def anomaly(name, payload):
        counts[name] += 1
        if len(examples[name]) < 8:
            examples[name].append(payload)
    for source_index, raw in enumerate(iter_matches(SOURCE)):
        mid = identity(raw['match_id'])
        header = date.fromisoformat(raw['date'])
        games = raw['games']
        days = [date.fromisoformat(game.get('date', str(header))) for game in games]
        for flag in excluded(raw):
            counts['declared_series_exclusion_' + flag] += 1
        day = min(header, min(days))
        release = max(header, max(days), date.fromisoformat(overrides.get(mid, str(header))))
        row = canonical.get(mid)
        if row is not None:
            checks.require('canonical_release_not_before_raw_whole_series', date.fromisoformat(str(row.release_day)) >= release, mid)
            release = date.fromisoformat(str(row.release_day))
        bo = declared_integer(raw, ('BoN', 'best_of'))
        first = (str(games[0]['t1_id']), str(games[0]['t2_id']))
        ha, hb = raw.get('tid_1', raw.get('t1_id')), raw.get('tid_2', raw.get('t2_id'))
        a = str(ha) if ha is not None else next(t for t in first if t != str(hb)) if hb is not None else first[0]
        b = str(hb) if hb is not None else next(t for t in first if t != a)
        teams = (a, b)
        comp = tag(next((raw[k] for k in ('tournament', 'tournament_name') if raw.get(k) is not None and raw[k] != ''), None))
        events, rating_maps = [], []
        for index, (game, played) in enumerate(zip(games, days), 1):
            gid = identity(game['game_id'])
            for flag in excluded(game):
                counts['declared_map_exclusion_' + flag] += 1
            sides = ('t1', 't2') if str(game['t1_id']) == a else ('t2', 't1')
            players, champs, statuses, roles, payloads, rosters = [], [], [], [], [], []
            values = np.full((10, 16), np.nan, np.float32)
            for side_index, side in enumerate(sides):
                roster, by_role = [], {}
                for offset, (raw_role, player) in enumerate(game[side + '_players'].items()):
                    ids = {identity(player[k]) for k in ('player_id', 'id') if player.get(k) is not None}
                    checks.require('explicit_player_id_alias_consistency', len(ids) <= 1, gid)
                    pid = next(iter(ids), None)
                    roster.append(pid)
                    players.append(pid)
                    champion = player.get('champion_id')
                    status = 1 if champion is None else 3 if isinstance(champion, (str, int)) and not isinstance(champion, bool) and str(champion).strip().lower() not in MISSING_TAGS else 2
                    champs.append(str(champion).strip() if status == 3 else None)
                    statuses.append(status)
                    r = role(raw_role) or role(player.get('role', player.get('position', '')))
                    roles.append(r)
                    if r in by_role:
                        by_role[r] = None
                    else:
                        by_role[r] = player
                    stats = player.get('stats') if isinstance(player.get('stats'), dict) else {}
                    for channel, stat in enumerate(STATS):
                        v = number(stats.get(stat))
                        values[side_index * 5 + offset, channel] = v
                        if not math.isfinite(v):
                            raw_value = stats.get(stat)
                            declared_missing = raw_value is None or isinstance(raw_value, str) and raw_value.strip().lower() in {'', '-', '--', 'n/a', 'na', 'null', 'none', 'nan', '?'}
                            counts['player_stat_' + stat + ('_declared_missing' if declared_missing else '_invalid_numeric')] += 1
                            if not declared_missing:
                                anomaly('invalid_player_stat_numeric', {'game_id': gid, 'player_id': pid, 'stat': stat, 'raw': raw_value})
                        if math.isfinite(v):
                            observed[stat] += 1
                            minimum[stat] = min(minimum.get(stat, v), v)
                            maximum[stat] = max(maximum.get(stat, v), v)
                            if channel < 7 and v < 0:
                                anomaly('negative_unsigned_player_stat', {'game_id': gid, 'player_id': pid, 'stat': stat, 'value': v})
                            if channel < 3 and v != math.floor(v):
                                anomaly('nonintegral_player_count', {'game_id': gid, 'player_id': pid, 'stat': stat, 'value': v})
                            if channel >= 10 and not 0 <= v <= 100:
                                anomaly('percent_outside_0_100', {'game_id': gid, 'player_id': pid, 'stat': stat, 'value': v})
                    counts['raw_player_objects'] += 1
                    counts['missing_player_identity'] += int(pid is None)
                    counts['missing_champion_identity'] += int(status == 1)
                missing = 5 - len(roster)
                players.extend([None] * missing)
                champs.extend([None] * missing)
                statuses.extend([1] * missing)
                roles.extend([0] * missing)
                roster.extend([None] * missing)
                rosters.append(tuple(roster))
                payloads.append(by_role)
                team_kills, team_reason = side_kills(game, side)
                if team_reason != 'valid':
                    anomaly('invalid_or_missing_team_kills', {'game_id': gid, 'side': side, 'reason': team_reason})
                kills = values[side_index * 5:(side_index + 1) * 5, 0]
                if len(game[side + '_players']) == 5 and np.isfinite(kills).all() and math.isfinite(team_kills):
                    counts['player_kill_sum_comparable_sides'] += 1
                    if float(kills.sum()) != team_kills:
                        anomaly('player_kill_sum_team_kills_mismatch', {'game_id': gid, 'side': side, 'player_sum': float(kills.sum()), 'team_kills': team_kills})
            ambiguous = {c for c, count in Counter(c for c in champs if c is not None).items() if count > 1}
            ambiguous_slots = tuple(i for i, c in enumerate(champs) if c in ambiguous)
            if ambiguous:
                anomaly('ambiguous_duplicate_champion_maps', {'match_id': mid, 'game_id': gid, 'champions': sorted(ambiguous), 'slots': ambiguous_slots})
                for slot in ambiguous_slots:
                    champs[slot], statuses[slot] = None, 2
            labels = map_labels(raw, game)
            kills_a, _ = side_kills(game, sides[0])
            kills_b, _ = side_kills(game, sides[1])
            values[:5, 12:] = kills_a, kills_b, labels[0], labels[1]
            values[5:, 12:] = kills_b, kills_a, labels[0], labels[1]
            duration, dr = label_number(game.get('game_duration'))
            if dr != 'valid' or duration <= 0:
                anomaly('invalid_or_nonpositive_duration_seconds', {'game_id': gid, 'raw': game.get('game_duration'), 'reason': dr})
            elif duration < 300 or duration > 7200:
                anomaly('duration_outside_5_120_minutes_diagnostic', {'game_id': gid, 'seconds': duration})
            counts['raw_map_patch_present'] += int(tag(game.get('patch')) is not None)
            patch = tag(game.get('patch')) or tag(raw.get('patch'))
            counts['effective_historical_patch_present'] += int(patch is not None)
            eligible = not excluded(raw) and not excluded(game)
            event = Event(gid, index, played.toordinal(), tuple(players), tuple(champs), np.asarray(statuses, np.uint8),
                np.asarray(roles, np.int64), values, patch,
                tag(game.get('tournament_name', game.get('tournament'))) or comp, labels, eligible, tuple(rosters), ambiguous_slots)
            events.append(event)
            counts['source_maps'] += 1
            counts['history_eligible_maps'] += int(eligible)
            counts['excluded_history_maps'] += int(not eligible)
            if eligible:
                try:
                    normalized = []
                    normalized_stats = []
                    for by_role in payloads:
                        if set(by_role) != set(range(1, 6)) or any(p is None for p in by_role.values()):
                            raise ValueError('Role coverage')
                        people = tuple(by_role[r] for r in range(1, 6))
                        roster = tuple(next(identity(p[k]) for k in ('player_id', 'id') if p.get(k) is not None) for p in people)
                        if None in roster or len(set(roster)) != 5:
                            raise ValueError('Rating identity')
                        normalized.append(roster)
                        # Rating-stat alias semantics belong to the reused rating recipe.
                        normalized_stats.append(rating._source_stats(people))
                    wa, wb = game.get(sides[0] + '_win'), game.get(sides[1] + '_win')
                    if wa not in (0, 1, False, True) or wb not in (0, 1, False, True) or wa == wb or game.get('draw'):
                        raise ValueError('Rating winner')
                    rating_maps.append(rating._Map(gid, *normalized, *normalized_stats, int(wa)))
                except (ValueError, StopIteration):
                    counts['rating_ineligible_maps'] += 1
        score, reason = score_label(raw, teams, bo)
        rr = rating._Series(source_index, mid, day, header, release, release > max(header, max(days)), day < header,
            bo, a, b, events[0].rosters[0], events[0].rosters[1], comp, None, 'independent_raw_parse', tuple(rating_maps))
        records.append(Record(mid, day, release, bo, teams, events, score, reason, rr))
        counts['source_series'] += 1
        counts['source_best_of_' + str(bo)] += 1
        counts['whole_series_delayed_release'] += int(release.toordinal() > min(e.played for e in events))
        counts['release_corrected_after_raw_maps'] += int(release > max(header, max(days)))
    records.sort(key=lambda r: (r.day, key(r.mid)))
    for rank, record in enumerate(records):
        record.rank = rank
    scalar = {'counts': counts, 'examples': examples,
        'player_channels': {s: {'source_field': 'game.t{1,2}_players[historic_role].stats[' + repr(s) + ']',
            'unit_interpretation': STAT_UNITS[s], 'observed': observed[s], 'missing': counts['raw_player_objects'] - observed[s],
            'minimum': minimum.get(s), 'maximum': maximum.get(s)} for s in STATS},
        'declared_label_flag_counts': {level: {flag: counts['declared_' + level + '_exclusion_' + flag] for flag in FLAGS}
                                       for level in ('series', 'map')},
        'player_numeric_parser': 'Direct source numeric fields; commas and percent signs removed, percentages remain 0..100 units. Nonfinite/boolean/float32-overflow invalid values become missing; GD/CSD/XPD@15 are signed differences.',
        'target_kills_definition': 't1_stats.kills + t2_stats.kills, nonnegative integral source counts; zero valid',
        'target_duration_definition': 'game_duration raw seconds divided by 60; strictly positive minutes',
        'anomaly_policy': 'Diagnostic only; no outcome-selected exclusions or newly changed eligibility',
        'rating_reused': ['prematch_features.ReplayState', '_consume_release_day', '_player_team_features', '_player_team_strength', '_source_stats rating-only aliases'],
        'container_reused': 'iter_matches only decodes/enumerates the existing JSON container; raw field interpretation, histories, labels, vocabulary and membership oracle are independently implemented here.',
        'rating_not_independent': 'Rating posterior mathematics and rating-only stat alias normalization are reused, not independently rederived.'}
    return records, scalar


def load_prepared(run_id):
    directory = ROOT / f'data/04_feature/champion_history/{run_id}'
    series = pd.read_parquet(directory / 'series.parquet')
    maps = pd.read_parquet(directory / 'maps.parquet')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    schema = json.loads((directory / 'schema.json').read_text())
    return directory, series, maps, arrays, schema


def rules_bindings():
    result = {}
    modes = ('UNKNOWN', 'STANDARD', 'HARD_FEARLESS', 'SOFT_FEARLESS')
    for binding in json.loads(RULES.read_text()).get('bindings', []):
        mode = binding.get('draft_mode', 'UNKNOWN')
        evidence = binding.get('evidence_key')
        if mode not in modes or mode != 'UNKNOWN' and (not isinstance(evidence, str) or not evidence.strip() or 'not verified' in evidence.lower()):
            mode = 'UNKNOWN'
        result[binding['competition_edition_phase']] = mode
    return result


def audit_data(run_id):
    directory, series, maps, arrays, schema = load_prepared(run_id)
    base = pd.read_parquet(BASE)
    checks = Checks()
    checks.equal('canonical_row_order', series.match_id.astype(str).to_numpy(), base.match_id.astype(str).to_numpy())
    checks.equal('canonical_row_index', series.row_index, np.arange(len(base)))
    checks.equal('allowed_model_input_keys', sorted(schema['model_input_keys']), sorted(INPUTS))
    checks.require('target_patch_not_input', schema['target_patch_input'] is False)
    checks.require('target_composition_not_input', schema['target_composition_input'] is False)
    records, scalar = parse_source(base, checks)
    by_id = {r.mid: r for r in records}
    player_ledger = defaultdict(list)
    pair_ledger = defaultdict(list)
    global_events = []
    vocab = ({}, {}, {})
    for record in records:
        for event in record.events:
            if not event.eligible:
                continue
            if str(record.day) < TRAIN_END and str(record.release) < TRAIN_END:
                for champ in sorted({c for c in event.champs if c is not None}):
                    vocab[0].setdefault(champ, len(vocab[0]) + 2)
                for v, identity_ in zip(vocab[1:], (event.patch, event.competition)):
                    if identity_ is not None:
                        v.setdefault(identity_, len(v) + 2)
            for slot, pid in enumerate(event.players):
                if pid is not None:
                    player_ledger[pid].append((event.played, record.rank, event.index, slot, record, event))
            for roster in event.rosters:
                known = [p for p in roster if p is not None]
                for j, pid in enumerate(known):
                    for other in known[j + 1:]:
                        pair_ledger[tuple(sorted((pid, other)))].append(record.release.toordinal())
            global_events.append((record.release.toordinal(), event.labels[:2]))
    for name, expected in zip(('champion', 'patch', 'competition'), vocab):
        checks.require('closed_train_' + name + '_vocabulary', schema[name + '_vocabulary'] == expected,
                       {'observed_size': len(schema[name + '_vocabulary']), 'expected_size': len(expected)})
    for ledger in player_ledger.values():
        ledger.sort(key=lambda e: e[:4])
    for ledger in pair_ledger.values():
        ledger.sort()
    global_events.sort(key=lambda e: e[0])
    global_days = [e[0] for e in global_events]
    gv = np.asarray([e[1] for e in global_events], np.float64)
    sums = np.vstack((np.zeros(2), np.cumsum(np.where(np.isfinite(gv), gv, 0), axis=0)))
    counts = np.vstack((np.zeros(2, np.int64), np.cumsum(np.isfinite(gv), axis=0)))
    # Replay reused Gaussian state on independently parsed, strictly released batches.
    release_batches, historical_days, target_days = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        release_batches[record.release].append(record.rating_record)
        historical_days[record.day].append(record)
    rows = list(series.itertuples(index=False))
    for row in rows:
        target_days[date.fromisoformat(str(row.date))].append(row)
    state, position = rating.ReplayState(), 0
    release_days = sorted(release_batches)
    bindings = rules_bindings()
    modes = ('UNKNOWN', 'STANDARD', 'HARD_FEARLESS', 'SOFT_FEARLESS')
    membership_digest = hashlib.sha256()
    membership_counts = Counter()
    ambiguity_proofs, boundary_rows = [], []
    role_cache = {}
    for day in sorted(set(historical_days) | set(target_days)):
        while position < len(release_days) and release_days[position] < day:
            release = release_days[position]
            batch = sorted((r for r in release_batches[release] if r.maps), key=lambda r: key(r.match_id))
            if batch:
                rating._consume_release_day(state, batch, release)
            position += 1
        for record in historical_days[day]:
            for event in record.events:
                for side in range(2):
                    mean, variance = rating._player_team_strength(state, event.rosters[1 - side], day, rating._PROCESS_SLOW)
                    event.opponent[side] = mean, math.sqrt(variance)
        role_cache.clear()
        for row in target_days[day]:
            i, ordinal = int(row.row_index), day.toordinal()
            record = by_id.get(str(row.match_id))
            rosters = (tuple(identity(p) for p in row.roster_a), tuple(identity(p) for p in row.roster_b))
            if record is not None:
                checks.equal('canonical_forecast_day', str(row.date), str(record.day), row.match_id)
                checks.equal('canonical_best_of', row.best_of, record.bo, row.match_id)
                for team, roster in zip((str(row.team_a), str(row.team_b)), rosters):
                    actual = record.events[0].rosters[record.teams.index(team)]
                    checks.require('first_map_identity_set_alignment', all(p is None or p in actual for p in roster), row.match_id)
                score = record.score
                if math.isfinite(score) and str(row.team_a) != record.teams[0]:
                    score = score + 3 if score < 3 else score - 3
                checks.equal('all_canonical_score_labels', row.score_index, score, row.match_id)
                checks.equal('all_canonical_winner_labels', row.winner, float(score < 3) if math.isfinite(score) else math.nan, row.match_id)
                checks.equal('all_canonical_score_absence_reasons', row.score_reason, record.score_reason, row.match_id)
            expected_split = ('TRAIN' if str(day) < TRAIN_END and str(row.effective_release_day) < TRAIN_END else
                'CAL' if TRAIN_END <= str(day) < CAL_END and str(row.effective_release_day) < CAL_END else
                'ASSESS' if CAL_END <= str(day) < '2027-01-01' else 'UNASSIGNED')
            checks.equal('release_partition_boundaries', row.split, expected_split, row.match_id)
            if str(day) in ('2025-06-30', TRAIN_END, '2025-12-31', CAL_END) or str(day) < TRAIN_END <= str(row.effective_release_day) or str(day) < CAL_END <= str(row.effective_release_day):
                boundary_rows.append({'match_id': row.match_id, 'forecast': str(day), 'release': row.effective_release_day, 'split': expected_split})
            expected_context = np.zeros(arrays['context'].shape[-1], np.float32)
            expected_context[{1: 0, 3: 1, 5: 2}[int(row.best_of)]] = 1
            mode = bindings.get(row.tournament, 'UNKNOWN')
            expected_context[3 + modes.index(mode)] = 1
            g = bisect_left(global_days, ordinal)
            expected_context[7:] = np.divide(sums[g], counts[g], out=np.full(2, np.nan), where=counts[g] > 0)
            checks.equal('historical_global_context_and_evidenced_rules', arrays['context'][i], expected_context, row.match_id)
            checks.equal('row_best_of_array', arrays['best_of'][i], row.best_of, row.match_id)
            history_counts, maximum_release = [], None
            for side, roster in enumerate(rosters):
                prior_players = rating._player_team_features(state, roster, day)
                checks.equal('reused_rating_player_features', arrays['players'][i, side, :, :-4], prior_players, row.match_id, atol=1e-5)
                for slot, pid in enumerate(roster):
                    if pid not in role_cache:
                        # Independent full-ledger selection, not bounded builder state.
                        released = [e for e in player_ledger.get(pid, ()) if e[4].release.toordinal() < ordinal] if pid is not None else []
                        role_cache[pid] = released, released[-16:]
                    released, selected = role_cache[pid]
                    history_counts.append(len(selected))
                    shape = (16, arrays['history'].shape[-1])
                    expected_history = np.full(shape, np.nan, np.float32)
                    categorical = {k: np.zeros(16, arrays[k].dtype) for k in ('champions', 'patches', 'competitions', 'roles', 'champion_status')}
                    expected_mask = np.zeros(16, bool)
                    for h, entry in enumerate(selected, 16 - len(selected)):
                        _, _, _, source_slot, source_record, event = entry
                        expected_history[h, :16] = event.values[source_slot]
                        expected_history[h, 16:18] = event.opponent[source_slot // 5]
                        expected_history[h, 18:] = ordinal - event.played, ordinal - source_record.release.toordinal()
                        expected_mask[h] = True
                        categorical['champions'][h] = 0 if event.champs[source_slot] is None else vocab[0].get(event.champs[source_slot], 1)
                        categorical['patches'][h] = 0 if event.patch is None else vocab[1].get(event.patch, 1)
                        categorical['competitions'][h] = 0 if event.competition is None else vocab[2].get(event.competition, 1)
                        categorical['roles'][h] = event.roles[source_slot]
                        categorical['champion_status'][h] = event.status[source_slot]
                        membership_digest.update(f'{i}|{side}|{slot}|{h}|{source_record.mid}|{event.gid}|{source_slot}\n'.encode())
                        membership_counts['observed_tokens'] += 1
                        membership_counts['bo2_tokens'] += int(source_record.bo == 2)
                        membership_counts['ambiguous_champion_tokens'] += int(source_slot in event.ambiguous_slots)
                        membership_counts['missing_champion_tokens'] += int(event.status[source_slot] == 1)
                        membership_counts['unknown_role_tokens'] += int(event.roles[source_slot] == 0)
                        maximum_release = max(maximum_release or source_record.release, source_record.release)
                        if source_slot in event.ambiguous_slots and len(ambiguity_proofs) < 8:
                            ambiguity_proofs.append({'row_index': i, 'target_match_id': row.match_id, 'historical_game_id': event.gid,
                                'player_id': pid, 'champion_code': int(arrays['champions'][i, side, slot, h]),
                                'player_stats_preserved': bool(np.array_equal(arrays['history'][i, side, slot, h, :12], event.values[source_slot, :12], equal_nan=True))})
                    where = {'row_index': i, 'player_id': pid, 'side': side, 'slot': slot}
                    checks.equal('all_history_membership_masks', arrays['mask'][i, side, slot], expected_mask, where)
                    checks.equal('all_history_numeric_alignment', arrays['history'][i, side, slot], expected_history, where)
                    for name, expected in categorical.items():
                        checks.equal('all_history_' + name + '_alignment', arrays[name][i, side, slot], expected, where)
                    rc = np.bincount([e[5].roles[e[3]] for e in released], minlength=6)
                    probs = (rc / rc.sum()).astype(np.float32) if rc.sum() else np.array([1, 0, 0, 0, 0, 0], np.float32)
                    checks.equal('historic_role_probability_without_target_roles', arrays['role_probs'][i, side, slot], probs, where)
                    pairs = [bisect_left(pair_ledger.get(tuple(sorted((pid, other))), ()), ordinal)
                             for other in roster if pid is not None and other is not None and other != pid]
                    extra = np.array([float(pid is not None), math.log1p(len(released)), math.log1p(float(np.mean(pairs))) if pairs else 0., math.log1p(min(pairs)) if pairs else 0.], np.float32)
                    checks.equal('identity_map_and_teammate_support', arrays['players'][i, side, slot, -4:], extra, where)
                    membership_counts['player_slots_checked'] += 1
            checks.equal('reported_history_min_mean_max', [row.history_min, row.history_mean, row.history_max],
                         [min(history_counts), np.mean(history_counts), max(history_counts)], row.match_id)
            checks.equal('reported_cold_player_count', row.cold_player_count, sum(c == 0 for c in history_counts), row.match_id)
            checks.equal('reported_unknown_player_count', row.unknown_player_count, sum(p is None for r in rosters for p in r), row.match_id)
            checks.equal('reported_latest_whole_series_release', identity(row.max_history_release_day), str(maximum_release) if maximum_release else None, row.match_id)
            membership_counts['canonical_rows_checked'] += 1
    event_lookup = {e.gid: (r, e) for r in records for e in r.events}
    checks.require('unique_raw_series_identities', len(by_id) == len(records))
    checks.require('unique_raw_game_identities', len(event_lookup) == sum(len(r.events) for r in records))
    expected_map_membership = [(int(row.row_index), str(row.match_id), event.gid, event.index)
                               for row in rows for event in by_id[str(row.match_id)].events]
    actual_map_membership = [(int(row.row_index), str(row.match_id), str(row.game_id), int(row.map_index))
                             for row in maps.itertuples(index=False)]
    checks.equal('full_canonical_map_membership_and_row_alignment', actual_map_membership, expected_map_membership)
    map_reason_counts = {k: Counter() for k in ('kills', 'duration')}
    for row in maps.itertuples(index=False):
        record, event = event_lookup[str(row.game_id)]
        checks.equal('canonical_map_match_id', str(row.match_id), record.mid, row.game_id)
        checks.equal('canonical_played_map_index', row.map_index, event.index, row.game_id)
        checks.equal('canonical_map_forecast_day', str(row.date), str(record.day), row.game_id)
        checks.equal('canonical_map_best_of', row.best_of, record.bo, row.game_id)
        checks.equal('all_played_map_kills_raw_total', row.kills, event.labels[0], row.game_id)
        checks.equal('all_played_map_duration_seconds_to_minutes', row.duration_minutes, event.labels[1], row.game_id)
        checks.equal('all_played_map_kills_absence_reason', row.kills_reason, event.labels[2], row.game_id)
        checks.equal('all_played_map_duration_absence_reason', row.duration_reason, event.labels[3], row.game_id)
        checks.equal('map_missing_champion_count', row.missing_champion_count, int((event.status == 1).sum()), row.game_id)
        checks.equal('map_unreliable_champion_count', row.unreliable_champion_count, int((event.status == 2).sum()), row.game_id)
        checks.equal('map_missing_stat_count', row.missing_stat_count, int(np.isnan(event.values[:, :12]).sum()), row.game_id)
        for task, reason in zip(('kills', 'duration'), event.labels[2:]):
            map_reason_counts[task][reason] += 1
    expected_maps = sum(len(by_id[str(row.match_id)].events) for row in rows if str(row.match_id) in by_id)
    checks.equal('all_canonical_recorded_maps_retained', len(maps), expected_maps)
    checks.equal('expected_raw_ambiguous_champion_map_count', scalar['counts']['ambiguous_duplicate_champion_maps'], 13)
    checks.require('bo2_history_actually_represented', membership_counts['bo2_tokens'] > 0)
    source_before = sha(SOURCE)
    original = json.loads((ROOT / f'data/08_reporting/champion_history/{run_id}/initial_integrity.json').read_text())
    checks.equal('immutable_source_original_hash', source_before, original['protected_hashes'][str(SOURCE.relative_to(ROOT))])
    return {'status': 'PASS' if checks.passed else 'FAIL', 'checks': checks.items, 'membership_counts': membership_counts,
        'membership_sha256': membership_digest.hexdigest(), 'canonical_maps': len(maps), 'all_map_label_reasons': map_reason_counts,
        'partition_counts': series.split.value_counts().to_dict(), 'closed_train_vocabulary_sizes': {k: len(v) for k, v in zip(('champion', 'patch', 'competition'), vocab)},
        'exact_rule_binding_counts': Counter(bindings.values()), 'ambiguous_channel_examples': ambiguity_proofs,
        'release_boundary_rows': boundary_rows, 'raw_scalar_audit': scalar,
        'scenario': 'Known first-map roster identity set; its supplied order is not a realized-role feature.',
        'limitations': ['Not a certified publication-time roster scenario.', 'Strict prior-day whole-series release; no intraday source publication timestamps.',
            'Exact rule applicability evidence is retrospective, not publication-time certification.', 'TRAIN-full vocabulary and numeric preprocessing are authorized; not rolling-origin model-dictionary fits.'],
        'input_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, BASE, RELEASES, RULES, *[directory / n for n in ('arrays.npz', 'series.parquet', 'maps.parquet', 'schema.json')])},
        'code_hashes': {p: sha(ROOT / p) for p in CODE_FILES}}


def audit_physical_perturbations(run_id):
    from src.features.champion_history_data import prepare_history_data
    _, series, _, _, _ = load_prepared(run_id)
    base = pd.read_parquet(BASE)
    checks = Checks()
    # Actual-source representative slices; no synthetic source records and no lineup mutation.
    choices = {}
    for name, selection in (
        ('BO1', (series.date >= CAL_END) & (series.best_of == 1)),
        ('BO3', (series.date >= CAL_END) & (series.best_of == 3)),
        ('BO5', (series.date >= CAL_END) & (series.best_of == 5)),
        ('unknown_identity', (series.date >= TRAIN_END) & (series.unknown_player_count > 0)),
        ('new_roster', (series.date >= TRAIN_END) & (series.roster_profile == 'new_roster')),
        ('partial_history', (series.date >= TRAIN_END) & (series.history_profile == 'partial'))):
        positions = np.flatnonzero(selection.to_numpy())
        if len(positions):
            choices[name] = int(positions[0])
    assessment = np.flatnonzero((series.date >= CAL_END).to_numpy())
    choices['future_target'] = int(max(assessment, key=lambda i: (str(series.iloc[i].date), key(str(series.iloc[i].match_id)))))
    wanted = {str(series.iloc[i].match_id) for i in choices.values()}
    wanted_players = {identity(p) for i in choices.values() for roster in (series.iloc[i].roster_a, series.iloc[i].roster_b) for p in roster if p is not None}
    raw_selected = []
    support_count = 0
    for raw in iter_matches(SOURCE):
        mid = str(raw['match_id'])
        actual_players = {identity(p[k]) for g in raw['games'] for side in ('t1', 't2') for p in g[side + '_players'].values()
                          for k in ('player_id', 'id') if p.get(k) is not None}
        if mid in wanted or actual_players & wanted_players:
            raw_selected.append(raw)
            support_count += int(mid not in wanted)
    canonical = base.loc[base.match_id.astype(str).isin(wanted)].copy().reset_index(drop=True)
    target_positions = {str(row.match_id): i for i, row in enumerate(canonical.itertuples(index=False))}
    rules = json.loads(RULES.read_text())
    receipts = []
    with tempfile.TemporaryDirectory(prefix='champion_history_temporal_') as temporary:
        folder = Path(temporary)
        def build(name, raw):
            path = folder / (name + '.json')
            path.write_text(json.dumps(raw))
            result = prepare_history_data(path, canonical, history_length=16, rules=rules, research_root=folder)
            return result, sha(path)
        original, original_hash = build('original_actual_subset', raw_selected)
        for mutation in ('future_fields', 'source_row_reverse'):
            changed = deepcopy(raw_selected)
            if mutation == 'source_row_reverse':
                changed.reverse()
                affected = list(wanted)
            else:
                affected = [str(series.iloc[choices['future_target']].match_id)]
                for raw in changed:
                    if str(raw['match_id']) not in affected:
                        continue
                    raw['patch'] = 'AUDIT_UNCERTIFIED_FUTURE_PATCH'
                    for game in raw['games']:
                        game['patch'] = 'AUDIT_UNCERTIFIED_FUTURE_PATCH'
                        game['t1_side'], game['t2_side'] = game.get('t2_side'), game.get('t1_side')
                        game['bans'] = ['AUDIT_FORBIDDEN_FUTURE_BAN']
                        game['t1_bans'] = ['AUDIT_FORBIDDEN_FUTURE_BAN_A']
                        game['t2_bans'] = ['AUDIT_FORBIDDEN_FUTURE_BAN_B']
                        game['game_duration'] = 9876
                        for side in ('t1', 't2'):
                            game.setdefault(side + '_stats', {})['kills'] = 87 if side == 't1' else 43
                            game[side + '_win'] = not bool(game.get(side + '_win'))
                            remapped = {}
                            for j, (_, player) in enumerate(game[side + '_players'].items()):
                                player['champion_id'] = 'AUDIT_TARGET_CHAMPION_' + side + str(j)
                                player['stats'] = {s: 7777 + j for s in STATS}
                                player['role'] = ROLES[(j + 2) % 5]
                                remapped[ROLES[(j + 2) % 5]] = player
                            game[side + '_players'] = remapped
            altered, mutated_hash = build(mutation, changed)
            compared_rows = []
            for mid, i in target_positions.items():
                for name in INPUTS:
                    checks.equal(mutation + '_' + name, altered[2][name][i], original[2][name][i], mid, atol=1e-5 if name == 'players' else 0.)
                compared_rows.append(mid)
            receipts.append({'mutation': mutation, 'changed_match_ids': affected, 'compared_match_ids': compared_rows,
                             'source_sha256': mutated_hash})
        # Each target gets its own physical copy, avoiding legitimate propagation
        # of other changed earlier targets into this target's historical snapshot.
        for mid in sorted(wanted):
            if str(canonical.iloc[target_positions[mid]].date) < TRAIN_END:
                continue
            changed = deepcopy(raw_selected)
            for raw in changed:
                if str(raw['match_id']) != mid:
                    continue
                raw['patch'] = 'AUDIT_UNCERTIFIED_TARGET_PATCH'
                for game in raw['games']:
                    game['patch'] = 'AUDIT_UNCERTIFIED_TARGET_PATCH'
                    game['t1_side'], game['t2_side'] = game.get('t2_side'), game.get('t1_side')
                    game['bans'] = ['AUDIT_FORBIDDEN_TARGET_BAN']
                    game['t1_bans'] = ['AUDIT_FORBIDDEN_TARGET_BAN_A']
                    game['t2_bans'] = ['AUDIT_FORBIDDEN_TARGET_BAN_B']
                    game['game_duration'] = 9876
                    for side in ('t1', 't2'):
                        game.setdefault(side + '_stats', {})['kills'] = 87
                        game[side + '_win'] = not bool(game.get(side + '_win'))
                        mapping = {}
                        for j, (_, player) in enumerate(game[side + '_players'].items()):
                            player['champion_id'] = 'AUDIT_TARGET_CHAMP_' + side + str(j)
                            player['stats'] = {s: 7000 + j for s in STATS}
                            player['role'] = ROLES[(j + 1) % 5]
                            mapping[ROLES[(j + 1) % 5]] = player
                        game[side + '_players'] = mapping
            altered, mutated_hash = build('target_' + mid, changed)
            i = target_positions[mid]
            for name in INPUTS:
                checks.equal('individual_target_fields_' + name, altered[2][name][i], original[2][name][i], mid, atol=1e-5 if name == 'players' else 0.)
            checks.require('physical_target_map_labels_actually_changed', not np.array_equal(
                original[1].loc[original[1].row_index == i, ['kills', 'duration_minutes']].to_numpy(),
                altered[1].loc[altered[1].row_index == i, ['kills', 'duration_minutes']].to_numpy(), equal_nan=True), mid)
            receipts.append({'mutation': 'individual_target_fields', 'changed_match_id': mid, 'compared_match_id': mid, 'source_sha256': mutated_hash})
        result = {'status': 'PASS' if checks.passed else 'FAIL', 'checks': checks.items, 'cases': choices,
            'actual_source_series': len(raw_selected), 'support_source_series': support_count,
            'canonical_target_series': len(canonical), 'original_subset_sha256': original_hash, 'attempts': receipts,
            'changed_fields': ['target/future patch', 'champion identities', 'historic-role-labelled payload keys on target maps',
                'player stats', 'team kills', 'duration seconds', 'map outcomes', 'target/future realized side',
                'target/future bans sentinel fields', 'source series row order'],
            'unchanged_scenario_inputs': ['target first-map player identity set', 'teams', 'best_of', 'forecast and release days', 'tournament'],
            'dictionary_caveat': 'All mutated targets are outside closed TRAIN. TRAIN-full vocabulary/preprocessing is allowed; arbitrary TRAIN dictionary mutation is not an earlier-origin fitted-model invariance.',
            'ban_fixture_limit': 'The inspected normalized game schema has no bans field; sentinel bans/t1_bans/t2_bans are added to actual isolated records to prove they cannot become model inputs.',
            'temporary_source_isolated': True, 'temporary_files_deleted': True, 'immutable_source_sha256_after': sha(SOURCE)}
    return result


def prediction_fields(prediction):
    result = {n: getattr(prediction, n) for n in ('score_probs', 'p_win', 'length_probs', 'play_probs')}
    if prediction.kills_mu is not None:
        result.update({n: getattr(prediction, n) for n in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma')})
    for i, component in enumerate(prediction.components):
        result.update({f'component_{i}_{k}': v for k, v in prediction_fields(component).items()})
    return result


def coherent(checks, prediction, name):
    p = prediction.score_probs
    bo = prediction.best_of
    legal = np.tile(np.arange(3), 2)[None] <= (bo // 2)[:, None]
    checks.require(name + '_finite_nonnegative_score', np.isfinite(p).all() and np.all(p >= 0))
    checks.equal(name + '_terminal_mass', p.sum(1), np.ones(len(p)), atol=1e-12)
    checks.equal(name + '_illegal_mass', p[~legal], np.zeros(np.count_nonzero(~legal)))
    checks.equal(name + '_winner_terminal_marginal', prediction.p_win, p[:, :3].sum(1))
    length = p[:, :3] + p[:, 3:]
    checks.equal(name + '_length_terminal_marginal', prediction.length_probs, length)
    expected_play = np.zeros((len(p), 5))
    for i, best_of in enumerate(bo):
        for loser in range(int(best_of) // 2 + 1):
            actual_length = int(best_of) // 2 + 1 + loser
            expected_play[i, :actual_length] += p[i, loser] + p[i, loser + 3]
    checks.equal(name + '_play_terminal_marginal', prediction.play_probs, expected_play, atol=1e-12)
    checks.require(name + '_play_nonincreasing', np.all(np.diff(prediction.play_probs, axis=1) <= 1e-12))


def audit_saved(run_id, *, include_cli=True):
    import torch
    torch.set_num_threads(1)
    _, series, _, arrays, _ = load_prepared(run_id)
    checks = Checks()
    model_dir = ROOT / f'data/06_models/champion_history/{run_id}'
    selected = set()
    categories = {}
    for name, condition in (
        ('BO1', (series.date >= CAL_END) & (series.best_of == 1)),
        ('BO3', (series.date >= CAL_END) & (series.best_of == 3)),
        ('BO5', (series.date >= CAL_END) & (series.best_of == 5)),
        ('cold', series.history_profile == 'cold'), ('new_roster', series.roster_profile == 'new_roster'),
        ('unknown_identity', series.unknown_player_count > 0), ('partial', series.history_profile == 'partial')):
        positions = np.flatnonzero(condition.to_numpy())
        categories[name] = int(positions[0]) if len(positions) else None
        if len(positions):
            selected.add(int(positions[0]))
    missing = np.isnan(arrays['history'][..., :12]) & arrays['mask'][..., None]
    positions = np.flatnonzero(missing.any(axis=(1, 2, 3, 4)))
    categories['actual_missing_player_stats'] = int(positions[0]) if len(positions) else None
    if len(positions):
        selected.add(int(positions[0]))
    ix = np.array(sorted(selected), np.int64)
    actual = {name: value[ix].copy() for name, value in arrays.items()}
    perm = np.array([4, 2, 0, 3, 1])
    team_keys = ('history', 'champions', 'patches', 'competitions', 'roles', 'mask', 'players', 'role_probs', 'champion_status')
    swapped = {k: v[:, ::-1].copy() if k in team_keys else v.copy() for k, v in actual.items()}
    permuted = {k: v[:, :, perm].copy() if k in team_keys else v.copy() for k, v in actual.items()}
    unknown = {k: v.copy() for k, v in actual.items()}
    unknown['champions'][unknown['mask']] = 10**9
    no_stats = {k: v.copy() for k, v in actual.items()}
    no_stats['history'][..., :12] = np.nan
    cold = {k: v.copy() for k, v in actual.items()}
    cold['mask'][:] = False
    cold['history'][:] = np.nan
    for name in ('champions', 'patches', 'competitions', 'roles', 'champion_status'):
        cold[name][:] = 0
    cold['role_probs'][:] = 0
    cold['role_probs'][..., 0] = 1
    cold['players'][..., :-4] = rating._player_team_features(rating.ReplayState(), (None,) * 5, date(2026, 1, 1))[None, None]
    cold['players'][..., -4:] = 0
    train = np.flatnonzero((series.date < TRAIN_END).to_numpy() & (series.effective_release_day < TRAIN_END).to_numpy())
    expected_preprocess = {}
    for name in ('history', 'players', 'context'):
        width = arrays[name].shape[-1]
        totals, squares, count = np.zeros(width), np.zeros(width), np.zeros(width, np.int64)
        for start in range(0, len(train), 512):
            batch = arrays[name][train[start:start + 512]].astype(np.float64)
            valid = np.isfinite(batch)
            if name == 'history':
                valid &= arrays['mask'][train[start:start + 512]][..., None]
            x = np.where(valid, batch, 0).reshape(-1, width)
            totals += x.sum(0)
            squares += np.square(x).sum(0)
            count += valid.reshape(-1, width).sum(0)
        mean = totals / np.maximum(count, 1)
        scale = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean * mean, 0))
        scale[scale < 1e-6] = 1
        expected_preprocess[name] = mean, scale, count
    models = []
    for variant in 'ABCD':
        for seed in (20261007, 20261008, 20261009):
            path = model_dir / f'{variant}_seed_{seed}.joblib'
            models.append((variant + '_' + str(seed), variant, path, joblib.load(path)))
        path = model_dir / f'{variant}_ensemble.joblib'
        models.append((variant + '_ensemble', variant, path, joblib.load(path)))
    model_receipts = []
    for name, variant, path, model in models:
        before = joblib.hash(model)
        input_scenarios = {'actual': actual, 'swapped': swapped, 'permuted': permuted,
                           'unknown_champion': unknown, 'missing_stats': no_stats, 'fully_cold': cold}
        input_before = {scenario: {k: array_sha(v) for k, v in values.items()}
                        for scenario, values in input_scenarios.items()}
        prediction = model.predict(actual)
        coherent(checks, prediction, name)
        swap_prediction = model.predict(swapped)
        permutation_prediction = model.predict(permuted)
        original_fields = prediction_fields(prediction)
        for field_name, value in original_fields.items():
            transformed = prediction_fields(swap_prediction)[field_name]
            expected = value[:, [3, 4, 5, 0, 1, 2]] if field_name.endswith('score_probs') else 1 - value if field_name.endswith('p_win') else value
            tolerance = 2e-4 if any(term in field_name for term in ('kills_mu', 'kills_alpha', 'duration_log')) else 2e-6
            checks.equal(name + '_team_swap_' + field_name, transformed, expected, atol=tolerance)
            checks.equal(name + '_player_permutation_' + field_name, prediction_fields(permutation_prediction)[field_name], value, atol=tolerance)
        for scenario, values in (('unknown_champion', unknown), ('missing_stats', no_stats), ('fully_cold', cold)):
            stress = model.predict(values)
            coherent(checks, stress, name + '_' + scenario)
            for field_name, value in prediction_fields(stress).items():
                checks.require(name + '_' + scenario + '_finite_' + field_name, np.isfinite(value).all())
            if scenario == 'unknown_champion' and variant in 'AC':
                for field_name, value in original_fields.items():
                    checks.equal(name + '_no_champion_sidechannel_' + field_name, prediction_fields(stress)[field_name], value)
        checks.equal(name + '_model_nonmutation', joblib.hash(model), before)
        for scenario, values in input_scenarios.items():
            for k, value in values.items():
                checks.equal(name + '_input_nonmutation_' + scenario + '_' + k, array_sha(value), input_before[scenario][k])
        if hasattr(model, 'preprocessing'):
            for key_, (mean, scale, count) in expected_preprocess.items():
                checks.equal(name + '_TRAIN_only_mean_' + key_, model.preprocessing.mean[key_], mean, atol=1e-8)
                checks.equal(name + '_TRAIN_only_scale_' + key_, model.preprocessing.scale[key_], scale, atol=1e-8)
                checks.equal(name + '_TRAIN_only_count_' + key_, model.preprocessing.counts[key_], count)
            checks.equal(name + '_train_index_binding', model.fit_receipt['train_indices_sha256'], hashlib.sha256(train.tobytes()).hexdigest())
            checks.equal(name + '_fixed_epochs', model.fit_receipt['epochs_completed'], 12)
            for key_ in ('champions', 'patches', 'competitions'):
                checks.equal(name + '_TRAIN_only_vocab_size_' + key_, model.vocab_sizes[key_], max(2, int(arrays[key_][train].max()) + 1))
        if variant in 'AB':
            try:
                prediction.map_prediction(1)
            except NotImplementedError:
                checks.require(name + '_score_only_props_raise', True)
            else:
                checks.require(name + '_score_only_props_raise', False)
        else:
            for m in range(1, 6):
                marginal = prediction.map_prediction(m)
                checks.require(name + f'_map_{m}_conditional_kills_probabilities', np.isfinite(marginal.kills_over(30.5)).all())
                checks.require(name + f'_map_{m}_conditional_duration_probabilities', np.isfinite(marginal.duration_over(30.)).all())
        model_receipts.append({'name': name, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'rows': len(ix)})
    # Exercise the saved CLI as an actual process, not the Python helper as proof.
    cli_receipts = []
    cli_rows = [categories[n] for n in ('BO1', 'BO3', 'BO5') if categories[n] is not None]
    ids = [str(series.iloc[i].match_id) for i in cli_rows]
    for variant in ('ABCD' if include_cli else ''):
        command = [sys.executable, 'scripts/train_champion_history.py', 'predict', '--run-id', run_id,
                   '--variant', variant, '--match-id', *ids]
        executed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        checks.equal('actual_saved_cli_' + variant + '_exit', executed.returncode, 0, executed.stderr[-2000:])
        if executed.returncode:
            cli_receipts.append({'variant': variant, 'exit': executed.returncode, 'stderr': executed.stderr})
            continue
        output = json.loads(executed.stdout)
        ensemble = joblib.load(model_dir / f'{variant}_ensemble.joblib')
        direct = ensemble.predict(arrays, np.array(cli_rows, np.int64))
        for j, row in enumerate(output['predictions']):
            checks.equal('actual_saved_cli_' + variant + '_terminal_probabilities', row['score_probabilities'], direct.score_probs[j], atol=1e-12)
            checks.equal('actual_saved_cli_' + variant + '_winner_marginal', row['winner_probability'], direct.p_win[j], atol=1e-12)
            checks.equal('actual_saved_cli_' + variant + '_map_play_marginal', row['map_play_probabilities'], direct.play_probs[j], atol=1e-12)
            checks.equal('actual_saved_cli_' + variant + '_scenario', row['scenario'], 'actual_first_map_roster_daily_release')
            checks.require('actual_saved_cli_' + variant + '_not_joint_kills_duration', row['joint_kills_duration'] is False)
            checks.equal('actual_saved_cli_' + variant + '_map_distribution_count', len(row['map_distributions']), int(row['best_of']) if variant in 'CD' else 0)
            for marginal in row['map_distributions']:
                m = marginal['map_index']
                checks.require('actual_saved_cli_' + variant + '_conditional_on_played', marginal['conditional_on_played'] is True)
                checks.equal('actual_saved_cli_' + variant + '_separate_play_probability', marginal['probability_played'], direct.play_probs[j, m - 1], atol=1e-12)
                distribution = direct.map_prediction(m)
                for line, probability in marginal['p_kills_over'].items():
                    checks.equal('actual_saved_cli_' + variant + '_conditional_kills_not_play_weighted', probability, distribution.kills_over(float(line))[j], atol=1e-12)
                for line, probability in marginal['p_duration_over_minutes'].items():
                    checks.equal('actual_saved_cli_' + variant + '_conditional_duration_not_play_weighted', probability, distribution.duration_over(float(line))[j], atol=1e-12)
        cli_receipts.append({'variant': variant, 'command': command, 'exit': executed.returncode,
                             'output_sha256': hashlib.sha256(executed.stdout.encode()).hexdigest(), 'predictions': output['predictions']})
    return {'status': 'PASS' if checks.passed else 'FAIL', 'checks': checks.items, 'models_exercised': model_receipts,
        'actual_source_case_categories': categories, 'canonical_case_rows': ix, 'saved_cli_attempts': cli_receipts,
        'verification_scope': 'saved_models_and_public_cli' if include_cli else 'typed_saved_models_only_public_cli_pending',
        'generated_stress_scenarios': ['actual source tensors with all champion IDs unknown', 'actual source tensors with player stats absent', 'symmetric completely cold unknown-player roster'],
        'preprocessing_oracle': 'Independent masked full-TRAIN numeric sums/squares/counts and TRAIN categorical maxima; no CAL/ASSESS fit',
        'limitations': ['No authenticated publication-time roster availability.', 'Conditional played-map marginal is not an unconditional forecast when map may not be played.', 'Separate kills and duration marginals do not claim their joint distribution.'],
        'torch_threads': torch.get_num_threads(), 'code_hashes': {p: sha(ROOT / p) for p in CODE_FILES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('data', 'perturb', 'saved', 'saved-typed'))
    parser.add_argument('--run-id', default=RUN_ID)
    args = parser.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Unsafe run ID')
    directory = ROOT / f'data/08_reporting/champion_history/{args.run_id}/verification'
    started = datetime.now(timezone.utc).isoformat()
    output = directory / ('temporal_' + args.action + '.json')
    attempt_path = directory / ('temporal_' + args.action + '_attempts.json')
    attempts = json.loads(attempt_path.read_text()) if attempt_path.exists() else []
    code_digest = sha(Path(__file__))
    source_archive = directory / ('temporal_verifier_' + code_digest + '.py')
    directory.mkdir(parents=True, exist_ok=True)
    if not source_archive.exists():
        shutil.copyfile(Path(__file__), source_archive)
    attempt_number = len(attempts) + 1
    try:
        receipt = {'data': audit_data, 'perturb': audit_physical_perturbations, 'saved': audit_saved,
                   'saved-typed': lambda run: audit_saved(run, include_cli=False)}[args.action](args.run_id)
        receipt.update(run_id=args.run_id, started_at=started, completed_at=datetime.now(timezone.utc).isoformat())
        receipt['code_hashes'] = {**receipt.get('code_hashes', {}), 'scripts/verify_champion_history_temporal.py': code_digest}
        receipt['archived_verifier'] = str(source_archive.relative_to(ROOT))
        save(output, receipt)
        attempt_receipt = directory / f'temporal_{args.action}_attempt_{attempt_number:03d}.json'
        save(attempt_receipt, receipt)
        attempts.append({'started_at': started, 'completed_at': receipt['completed_at'], 'status': receipt['status'],
                         'receipt_sha256': sha(output), 'retained_receipt': str(attempt_receipt.relative_to(ROOT)),
                         'verifier_sha256': code_digest, 'archived_verifier': str(source_archive.relative_to(ROOT))})
        save(attempt_path, attempts)
        print(json.dumps({'status': receipt['status'], 'receipt': str(output.relative_to(ROOT)), 'checks': len(receipt['checks'])}))
        return 0 if receipt['status'] == 'PASS' else 1
    except Exception as error:
        attempts.append({'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat(), 'status': 'ERROR',
            'error': repr(error), 'traceback': traceback.format_exc(), 'code_hashes': {p: sha(ROOT / p) for p in CODE_FILES}})
        failure_path = directory / f'temporal_{args.action}_attempt_{attempt_number:03d}.json'
        save(failure_path, attempts[-1])
        attempts[-1].update(retained_receipt=str(failure_path.relative_to(ROOT)),
                            verifier_sha256=code_digest, archived_verifier=str(source_archive.relative_to(ROOT)))
        save(attempt_path, attempts)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
