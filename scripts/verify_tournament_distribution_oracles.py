#!/usr/bin/env python3
"""Independent small-state distribution references; no historical accuracy claims."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_tournament_forecasts import monte_carlo_precision
from src.models.tournament_catalog import instantiate_profile
from src.models.tournament_formats import simulate_tournament
from src.models.tournament_prediction import file_digest


def pair(a, b):
    return json.dumps(sorted((a, b)), separators=(',', ':'))


def probability(table, a, b, bo):
    return table[a, b, bo] if (a, b, bo) in table else 1 - table[b, a, bo]


def lck_reference(table, *, after_first=False):
    """Enumerate the five stated match outcomes directly, without graph execution."""
    joint = {pair(a, b): 0. for a, b in combinations('ABCDEF', 2)}
    last_pair = dict(joint)
    for bits in product((0, 1), repeat=4 if after_first else 5):
        decisions = iter(bits)
        mass = 1.

        def play(a, b):
            nonlocal mass
            first_wins = next(decisions)
            p = probability(table, a, b, 5)
            mass *= p if first_wins else 1 - p
            return (a, b) if first_wins else (b, a)

        first = 'E' if after_first else play('E', 'F')[0]
        second = play('D', first)[0]
        challenger = play('C', second)[0]
        upper, upper_loser = play('A', 'B')
        second_qualifier = play(upper_loser, challenger)[0]
        joint[pair(upper, second_qualifier)] += mass
        last_pair[pair(upper_loser, challenger)] += mass
    advancement = {team: sum(p for category, p in joint.items() if team in json.loads(category))
                   for team in 'ABCDEF'}
    return joint, advancement, last_pair


def elimination_reference(table):
    joint = {pair(a, b): 0. for a, b in combinations('ABCD', 2)}
    champion = dict.fromkeys('ABCD', 0.)
    for first, second in product(('A', 'D'), ('B', 'C')):
        mass = (probability(table, 'A', 'D', 1) if first == 'A' else 1 - probability(table, 'A', 'D', 1))
        mass *= probability(table, 'B', 'C', 1) if second == 'B' else 1 - probability(table, 'B', 'C', 1)
        joint[pair(first, second)] += mass
        p = probability(table, first, second, 1)
        champion[first] += mass * p
        champion[second] += mass * (1 - p)
    return champion, joint


def round_robin_reference(table, *, score_tiebreak=False):
    champion = dict.fromkeys('ABC', 0.)
    joint = {pair(a, b): 0. for a, b in combinations('ABC', 2)}
    bo = 3 if score_tiebreak else 1
    for bits in product((0, 1), repeat=6 if score_tiebreak else 3):
        wins, maps, mass = dict.fromkeys('ABC', 0), dict.fromkeys('ABC', 0), 1.
        for index, ((a, b), first_wins) in enumerate(zip(combinations('ABC', 2), bits[:3])):
            p = probability(table, a, b, bo)
            mass *= p if first_wins else 1 - p
            wins[a if first_wins else b] += 1
            if score_tiebreak:
                differential = (2 - bits[3+index]) * (1 if first_wins else -1)
                maps[a] += differential
                maps[b] -= differential
                mass *= .5
        a, b = sorted('ABC', key=lambda team: (-wins[team], -maps[team], team))[:2]
        joint[pair(a, b)] += mass
        p = probability(table, a, b, bo)
        champion[a] += mass * p
        champion[b] += mass * (1 - p)
    return champion, joint


def fixtures():
    table = {(a, b, bo): .5 + .055 * (ord(b) - ord(a))
             for a, b in combinations('ABCDEF', 2) for bo in (1, 3, 5)}
    final = {'id': 'final', 'kind': 'single_elimination', 'entrants': list('ABCD'), 'best_of': 1}
    event = {'version': 1, 'id': 'four-team', 'teams': list('ABCD'), 'stages': [final],
             'champion': {'stage': 'final', 'group': 'all', 'rank': 1}}
    champion, joint = elimination_reference(table)
    yield 'four-team', event, table, {}, {'champion_prob': champion, 'joint_final_prob': joint}
    event = {'version': 1, 'id': 'rr-final', 'teams': list('ABC'), 'stages': [
        {'id': 'groups', 'kind': 'round_robin', 'entrants': list('ABC'), 'best_of': 1,
         'tiebreakers': ['wins', 'seed'], 'advance': 2},
        {'id': 'final', 'kind': 'single_elimination', 'best_of': 1,
         'entrants': [{'stage': 'groups', 'group': 'all', 'rank': rank} for rank in (1, 2)]}],
        'champion': {'stage': 'final', 'group': 'all', 'rank': 1}}
    champion, joint = round_robin_reference(table)
    yield 'rr-final', event, table, {}, {'champion_prob': champion, 'joint_final_prob': joint}
    score_event = json.loads(json.dumps(event))
    for stage in score_event['stages']:
        stage['best_of'] = 3
    score_event['stages'][0]['tiebreakers'] = ['wins', 'map_diff', 'seed']
    champion, joint = round_robin_reference(table, score_tiebreak=True)
    yield 'rr-score-tie-final', score_event, table, {'score_distributions': {3: [.5, .5]}}, {
        'champion_prob': champion, 'joint_final_prob': joint}
    event = instantiate_profile('lck-road-to-msi', list('ABCDEF'))
    for after_first in (False, True):
        joint, advancement, last_pair = lck_reference(table, after_first=after_first)
        kwargs = {'observables': [{'id': 'last-pair', 'kind': 'node_matchup',
                                  'node': {'stage': 'qualifier', 'round': 'last'},
                                  'categories': list(last_pair), 'absent_category': 'absent'}]}
        kwargs['observables'][0]['categories'].append('absent')
        if after_first:
            kwargs['state'] = {'version': 1, 'cutoff': '2026-01-02T00:00:00Z',
                               'temporal_policy': 'daily_rollforward', 'completed': [{
                'stage': 'qualifier', 'group': 'all', 'round': 'r1', 'team_a': 'E', 'team_b': 'F',
                'winner': 'E', 'loser': 'F', 'best_of': 5, 'score_a': 3, 'score_b': 1, 'weight': 1,
                'source_date': '2026-01-01'}]}
        yield 'lck-prefix' if after_first else 'lck-start', event, table, kwargs, {
            'joint_advance_prob/qualifier': joint, 'advance_prob/qualifier': advancement,
            'observable_prob/last-pair': {**last_pair, 'absent': 0.}}
    event = {'version': 1, 'id': 'reset', 'teams': list('AB'), 'stages': [
        {'id': 'event', 'kind': 'double_elimination', 'entrants': list('AB'), 'best_of': 1,
         'final_reset': 'if_lower_wins'}], 'champion': {'stage': 'event', 'group': 'all', 'rank': 1}}
    upper = {'stage': 'event', 'round': 'upper-1'}
    kwargs = {'observables': [
        {'id': 'series', 'kind': 'future_series_count', 'categories': ['2', '3']},
        {'id': 'reset', 'kind': 'node_matchup', 'categories': [pair('A', 'B'), 'absent'],
         'absent_category': 'absent', 'node': {'stage': 'event', 'round': 'grand-final-reset'}},
        {'id': 'path', 'kind': 'champion_path', 'categories': ['upper', 'lower'],
         'source': 'synthetic two-team reset bracket with declared upper match',
         'paths': {'upper': [{'node': upper, 'outcome': 'winner'}],
                   'lower': [{'node': upper, 'outcome': 'loser'}]}}]}
    yield 'double-elimination-reset', event, {('A', 'B', 1): .5}, kwargs, {
        'champion_prob': {'A': .5, 'B': .5}, 'observable_prob/series': {'2': .5, '3': .5},
        'observable_prob/reset': {pair('A', 'B'): .5, 'absent': .5},
        'observable_prob/path': {'upper': .75, 'lower': .25}}
    event = {'version': 1, 'id': 'swiss', 'teams': list('ABCD'), 'stages': [
        {'id': 'event', 'kind': 'swiss', 'entrants': list('ABCD'), 'best_of': 1,
         'rounds': 2, 'no_rematches': True, 'tiebreakers': ['wins', 'seed'], 'advance': 2,
         'draw': {'policy': 'uniform', 'avoid_same': ['pool'],
                  'labels': {team: {'pool': 'x' if team in 'AB' else 'y'} for team in 'ABCD'}}}]}
    kwargs = {'observables': [{'id': 'series', 'kind': 'future_series_count', 'categories': ['4']},
                              {'id': 'repeat', 'kind': 'repeat_encounter_count', 'categories': ['0']}]}
    yield 'constrained-swiss', event, table, kwargs, {
        'observable_prob/series': {'4': 1.}, 'observable_prob/repeat': {'0': 1.}}


def compare(exact, observed, samples, *, marginal=False, cells=None, looks=1):
    if set(observed) - set(exact):
        raise ValueError('simulation emits a category absent from the independent reference')
    observed = {key: observed.get(key, 0.) for key in exact}
    errors = [abs(observed[key] - p) for key, p in exact.items()]
    if marginal:
        precision = {key: monte_carlo_precision({'yes': observed[key], 'no': 1-observed[key]}, samples,
                     cells=cells, looks=looks, structural_zeros=['yes'] if p == 0 else ['no'] if p == 1 else [])
                     for key, p in exact.items()}
        return {'maximum_category_error': max(errors), 'total_variation': None,
                'reason': 'slot-conserving marginals are not one categorical vector', 'precision': precision}
    infinite = any(p > 0 and observed[key] == 0 for key, p in exact.items())
    kl = None if infinite else sum(p * math.log(p / observed[key]) for key, p in exact.items() if p > 0)
    return {'maximum_category_error': max(errors), 'total_variation': sum(errors)/2,
            'expected_log_score_excess': kl,
            'expected_log_score_status': 'infinite_unsampled_reference_support' if infinite else 'finite',
            'expected_brier_excess': sum((observed[key]-p)**2 for key, p in exact.items()),
            'precision': monte_carlo_precision(observed, samples, cells=cells, looks=looks,
                         structural_zeros=[key for key, p in exact.items() if p == 0])}


def run(output, samples, seeds):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/'src/models/tournament_formats.py', ROOT/'scripts/score_tournament_forecasts.py',
             ROOT/'conf/base/tournament_formats.json']
    hashes = {str(path): file_digest(path) for path in paths}
    protocol = {'version': 'tournament-exact-distribution-oracles-v1', 'samples': samples, 'seeds': seeds,
                'qualification': 'synthetic_mechanics_not_historical_accuracy', 'python': platform.python_version(),
                'inputs': hashes, 'reference': 'independent direct outcome enumeration',
                'stopping': 'all fixed sample sizes and seeds; no observed historical labels',
                'precision': 'simultaneous cells across named vectors and all fixed looks per fixture/seed'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for name, event, table, kwargs, expected in fixtures():
        relevant = {(a, b, bo): p for (a, b, bo), p in table.items()
                    if a in event['teams'] and b in event['teams'] and bo == event['stages'][0]['best_of']}
        cells = sum(2*len(values) if source.startswith('advance_prob/') else len(values)
                    for source, values in expected.items())
        for seed in seeds:
            for n in samples:
                result = simulate_tournament(event, relevant, simulations=n, seed=seed, **kwargs)
                for source, exact in expected.items():
                    observed = result
                    for component in source.split('/'):
                        observed = observed[component]
                    rows.append({'fixture': name, 'seed': seed, 'simulations': n, 'source': source,
                                 'exact': exact, 'estimate': observed,
                                 **compare(exact, observed, n, marginal=source.startswith('advance_prob/'),
                                           cells=cells, looks=len(samples))})
                print(json.dumps({'fixture': name, 'seed': seed, 'simulations': n}), flush=True)
    if any(file_digest(path) != value for path, value in hashes.items()):
        raise ValueError('oracle implementation or reference changed during experiment')
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    (output/'completed.json').write_text(json.dumps({'rows': len(rows), 'results_sha256': file_digest(output/'results.json'),
        'maximum_category_error': max(row['maximum_category_error'] for row in rows),
        'status': 'completed', 'production_qualified': False}, indent=2)+'\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, nargs='+', default=[20000, 40000, 80000, 160000, 200000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    args = parser.parse_args()
    if any(n < 1 for n in args.samples) or len(set(args.samples)) != len(args.samples):
        parser.error('samples must be distinct positive integers')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('seeds must be distinct')
    run(args.output, sorted(args.samples), args.seeds)


if __name__ == '__main__':
    main()
