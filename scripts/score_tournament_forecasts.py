#!/usr/bin/env python3
"""Score complete tournament forecasts without treating simulations as observations.

CLI: --bundle-manifest index.json --output-dir NEW [--allow-retrospective].
Index is a JSON list: {forecast: simulation.json, results: results.json,
bracket: bracket.json, snapshot: snapshot.json, protocol: protocol_directory}.
Snapshot and protocol are required unless retrospective; prestart forecasts must
be sealed simulation.json artifacts reproducible from the captured predictions.
Paths are relative to the index. Forecast is the existing simulation.json payload
(tournament_id, models -> simulator output). Bracket uses TournamentBracket and
BracketMatchNode. Results uses the existing cache format tournament_id + matches
(node IDs -> team1/team2/winner/score1/score2); every node must be complete.
Optional placement_prob is team -> rank-string -> probability for every rank
1..N; corresponding results.placements must be a complete, untied permutation.
Tied placement bands are deliberately not invented from elimination rounds.

``evaluate_ledger`` scores separately persisted forecast and outcome rows for
multi-phase/rolling evaluation; unlike the static-graph CLI it accepts official
tied placement bands, actual qualification targets and categorical joint targets.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from datetime import date, datetime, timedelta, timezone
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix
from scipy.special import expit
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from betting_app.services.tournament_service import BracketMatchNode, TournamentBracket
from src.models.frozen_tournament import validate_bracket

VERSION = 'tournament-forecast-scoring-v1'
TOLERANCE = 1e-8


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, payload):
    # JSON has no infinity; keep its mathematical meaning explicit, never clip.
    def safe(value):
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            return 'Infinity' if value == math.inf else '-Infinity' if value == -math.inf else None
        return value
    Path(path).write_text(json.dumps(safe(payload), indent=2, allow_nan=False) + '\n')


def load_bracket(payload):
    return TournamentBracket(**{**payload, 'matches': {
        key: BracketMatchNode(**node) for key, node in payload['matches'].items()}})


def _vector(mapping, teams, label, mass):
    if not isinstance(mapping, dict) or set(mapping) != set(teams):
        raise ValueError(f'{label}: probabilities must cover every exact entrant')
    if any(isinstance(value, (bool, str)) for value in mapping.values()):
        raise ValueError(f'{label}: probabilities must be numeric')
    values = np.asarray([mapping[team] for team in teams], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f'{label}: probabilities must be finite and within [0,1]')
    if not np.isclose(values.sum(), mass, atol=TOLERANCE, rtol=0):
        raise ValueError(f'{label}: inconsistent probability mass (expected {mass})')
    return values


def _log_score(probability):
    return -math.log(probability) if probability > 0 else math.inf


def observed_outcomes(bracket, results):
    """Follow the supplied pre-event graph; never derive its edges from outcomes."""
    order = validate_bracket(bracket)
    if results.get('tournament_id') != bracket.id or set(results.get('matches', {})) != set(order):
        raise ValueError('outcomes must identify the same tournament and every bracket node')
    pending, participants, winners = {}, {}, {}
    for node_id in order:
        node, observed = bracket.matches[node_id], results['matches'][node_id]
        slots = [pending.pop((node_id, slot), getattr(node, f'team{slot}')) for slot in (1, 2)]
        present = [team for team in slots if team is not None]
        if not present or len(set(present)) != len(present):
            raise ValueError(f'{node_id}: invalid routed participants')
        if [observed.get('team1'), observed.get('team2')] != slots:
            raise ValueError(f'{node_id}: observed participants disagree with frozen routing')
        winner = observed.get('winner')
        if winner not in present:
            raise ValueError(f'{node_id}: missing or invalid observed winner')
        loser = next((team for team in present if team != winner), None)
        if len(present) == 2:
            scores = [observed.get('score1'), observed.get('score2')]
            if any(type(value) is not int or value < 0 for value in scores):
                raise ValueError(f'{node_id}: complete nonnegative integer series scores required')
            winning_slot = slots.index(winner)
            if scores[winning_slot] != node.best_of // 2 + 1 or scores[1-winning_slot] >= scores[winning_slot]:
                raise ValueError(f'{node_id}: score contradicts winner or best_of')
        elif observed.get('score1') is not None or observed.get('score2') is not None:
            raise ValueError(f'{node_id}: a bye must not fabricate a played score')
        participants[node_id], winners[node_id] = present, winner
        for outcome, team in [('winner', winner), ('loser', loser)]:
            target = getattr(node, f'next_match_{outcome}_id')
            if target is not None:
                pending[(target, getattr(node, f'next_match_{outcome}_slot'))] = team
    return participants, winners


def _validate_probability_flow(bracket, reach, champion, terminal):
    """Require feasible winner/loser marginal flow through every incoming slot."""
    teams = bracket.teams
    width = len(teams)
    offsets = {node_id: index * width for index, node_id in enumerate(bracket.matches)}
    incoming = {node_id: [] for node_id in bracket.matches}
    for source, node in bracket.matches.items():
        for outcome in ('winner', 'loser'):
            target = getattr(node, f'next_match_{outcome}_id')
            if target is not None:
                incoming[target].append((source, outcome))
    rows, columns, coefficients, rhs, bounds = [], [], [], [], []
    for node_id, node in bracket.matches.items():
        row = len(rhs)
        rhs.append(1.0)
        for index in range(width):
            rows.append(row)
            columns.append(offsets[node_id] + index)
            coefficients.append(1.0)
            bounds.append((float(champion[index]), float(champion[index])) if node_id == terminal
                          else (0.0, float(reach[node_id][index])))
        for index, team in enumerate(teams):
            row = len(rhs)
            value = float(reach[node_id][index]) - int(team in (node.team1, node.team2))
            for source, outcome in incoming[node_id]:
                rows.append(row)
                columns.append(offsets[source] + index)
                coefficients.append(1.0 if outcome == 'winner' else -1.0)
                if outcome == 'loser':
                    value -= float(reach[source][index])
            rhs.append(value)
    variables = len(offsets) * width
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(len(rhs), variables)).tocsr()
    solution = linprog(np.zeros(variables), A_eq=matrix, b_eq=rhs, bounds=bounds,
                       method='highs', options={'primal_feasibility_tolerance': TOLERANCE})
    if not solution.success:
        raise ValueError('forecast probabilities violate routed winner/loser marginal flow')


def score_event(bracket, forecast, results):
    participants, winners = observed_outcomes(bracket, results)
    teams = bracket.teams
    terminal = next(key for key, node in bracket.matches.items() if node.next_match_winner_id is None)
    if forecast.get('championship_node_id') != terminal or forecast.get('probability_unit') != 'series_win':
        raise ValueError('forecast championship or SERIES probability unit disagrees with graph')
    champion = _vector(forecast.get('champion_prob'), teams, 'champion', 1)
    final = _vector(forecast.get('final_prob'), teams, 'final', len(participants[terminal]))
    node_maps = forecast.get('node_reach_prob')
    if not isinstance(node_maps, dict) or set(node_maps) != set(bracket.matches):
        raise ValueError('node reach forecasts must cover every bracket node')
    if (champion > final + TOLERANCE).any():
        raise ValueError('champion probability cannot exceed final reach')
    observed_champion = np.asarray([team == winners[terminal] for team in teams], dtype=float)
    scores = {'entrant_count': len(teams), 'node_count': len(participants),
              'champion_log_loss': _log_score(champion[teams.index(winners[terminal])]),
              'champion_brier': float(np.square(champion-observed_champion).sum())}
    calibration = []

    def binary(label, node_id, probability, observed):
        target = np.asarray([team in observed for team in teams], dtype=float)
        losses = [_log_score(float(p if y else 1-p)) for p, y in zip(probability, target)]
        calibration.extend({'target': label, 'node_id': node_id, 'team': team,
                            'probability': float(p), 'observed': int(y)}
                           for team, p, y in zip(teams, probability, target))
        return float(np.square(probability-target).mean()), float(np.mean(losses))

    binary('champion', terminal, champion, [winners[terminal]])
    scores['final_brier'], scores['final_log_loss'] = binary('final', terminal, final, participants[terminal])
    node_brier, node_log = [], []
    possible_outputs = {}
    node_probabilities = {}
    for node_id in validate_bracket(bracket):
        node = bracket.matches[node_id]
        possible = {team for team in (node.team1, node.team2) if team is not None}
        for parent_id, parent in bracket.matches.items():
            if node_id in (parent.next_match_winner_id, parent.next_match_loser_id):
                possible.update(possible_outputs[parent_id])
        possible_outputs[node_id] = possible
        probability = _vector(node_maps[node_id], teams, f'node {node_id}', len(participants[node_id]))
        node_probabilities[node_id] = probability
        for index, team in enumerate(teams):
            if team not in possible and probability[index] > TOLERANCE:
                raise ValueError(f'{node_id}: unreachable entrant has positive reach probability')
            if team in (node.team1, node.team2) and abs(probability[index]-1) > TOLERANCE:
                raise ValueError(f'{node_id}: a seeded entrant must reach its node')
        if node_id == terminal and not np.allclose(probability, final, atol=TOLERANCE, rtol=0):
            raise ValueError('final probability must equal championship node reach')
        brier, log_loss = binary('node_reach', node_id, probability, participants[node_id])
        node_brier.append(brier)
        node_log.append(log_loss)
    _validate_probability_flow(bracket, node_probabilities, champion, terminal)
    scores['node_reach_brier'], scores['node_reach_log_loss'] = float(np.mean(node_brier)), float(np.mean(node_log))
    # Missing optional forecasts are unavailable even when no ranks were recorded.
    scores['placement_scores_available'] = 'placement_prob' in forecast
    if 'placement_prob' in forecast:
        placements, distributions = results.get('placements'), forecast['placement_prob']
        if not isinstance(placements, dict) or set(placements) != set(teams) or sorted(placements.values()) != list(range(1, len(teams)+1)):
            raise ValueError('placement outcomes require every untied rank exactly once')
        if placements[winners[terminal]] != 1 or (len(teams) > 1 and {t for t in teams if placements[t] <= 2} != set(participants[terminal])):
            raise ValueError('placement outcomes disagree with final results')
        if not isinstance(distributions, dict) or set(distributions) != set(teams):
            raise ValueError('placement distributions require every entrant')
        ranks = [str(rank) for rank in range(1, len(teams)+1)]
        matrix = np.stack([_vector(distributions[team], ranks, 'placement', 1) for team in teams])
        if not np.allclose(matrix.sum(axis=0), 1, atol=TOLERANCE, rtol=0) or not np.allclose(matrix[:, 0], champion, atol=TOLERANCE, rtol=0):
            raise ValueError('placement mass or champion marginal disagrees')
        if len(teams) > 1 and not np.allclose(matrix[:, :2].sum(axis=1), final, atol=TOLERANCE, rtol=0):
            raise ValueError('placement top-two marginal disagrees with final')
        target = np.eye(len(teams))[[placements[team]-1 for team in teams]]
        scores['placement_brier'] = float(np.square(matrix-target).sum(axis=1).mean())
        scores['placement_log_loss'] = float(np.mean([_log_score(matrix[i, placements[team]-1]) for i, team in enumerate(teams)]))
        if len(teams) > 1:
            scores['placement_rps'] = float(np.square(np.cumsum(matrix-target, axis=1)[:, :-1]).mean())
    return scores, calibration


def summarize_scores(scores, replicates=5000, seed=82):
    if type(replicates) is not int or replicates < 1 or scores.empty:
        raise ValueError('nonempty scores and positive bootstrap replicates required')
    if scores.duplicated(['tournament_id', 'model']).any():
        raise ValueError('exactly one frozen forecast per event/model required')
    cohorts = [set(group.tournament_id) for _, group in scores.groupby('model')]
    if any(cohort != cohorts[0] for cohort in cohorts):
        raise ValueError('all compared models must cover the same events')
    models = sorted(scores.model.unique())
    events = sorted(scores.tournament_id.unique())
    indices_by_count = {}
    metrics = [column for column in scores if column.endswith(('_brier', '_log_loss', '_rps'))]
    summaries, paired = [], []
    for metric in metrics:
        table = scores.pivot(index='tournament_id', columns='model', values=metric).reindex(events)
        missing = table.isna()
        if missing.any().any():
            unavailable = missing.all(axis=1)
            if (not metric.startswith('placement_') or 'placement_scores_available' not in scores
                    or (missing.any(axis=1) != unavailable).any()):
                raise ValueError(f'{metric}: incomplete outcomes/distributions across compared models')
            availability = scores.pivot(index='tournament_id', columns='model',
                                        values='placement_scores_available').reindex(events)
            if not availability.loc[unavailable].eq(False).all().all():
                raise ValueError(f'{metric}: incomplete scores were not declared unavailable')
            table = table.loc[~unavailable]
        count = len(table)
        if not count:
            raise ValueError(f'{metric}: no eligible events')
        if count not in indices_by_count:
            indices_by_count[count] = np.random.default_rng(seed).integers(count, size=(replicates, count))
        indices = indices_by_count[count]
        values = table[models].to_numpy(dtype=float)
        if np.isnan(values).any() or (values < 0).any():
            raise ValueError(f'{metric}: invalid proper score')
        draws = values[indices].mean(axis=1)
        for index, model in enumerate(models):
            finite = bool(np.isfinite(values[:, index]).all())
            bounds = np.quantile(draws[:, index], [0.025, 0.975]) if count > 1 and finite else [None, None]
            summaries.append({'model': model, 'metric': metric, 'mean': float(values[:, index].mean()),
                              'ci_low': bounds[0], 'ci_high': bounds[1], 'events': count,
                              'input_events': len(events), 'excluded_unavailable_events': len(events)-count,
                              'uncertainty': 'event_cluster_percentile' if count > 1 and finite else 'unavailable_single_event_or_infinite_score'})
        for left, right in itertools.combinations(range(len(models)), 2):
            finite = bool(np.isfinite(values[:, [left, right]]).all())
            delta = values[:, left]-values[:, right] if finite else None
            bounds = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]) if finite and count > 1 else [None, None]
            paired.append({'model': models[left], 'reference': models[right], 'metric': metric,
                           'delta': float(delta.mean()) if finite else None,
                           'ci_low': bounds[0], 'ci_high': bounds[1], 'events': count,
                           'input_events': len(events), 'excluded_unavailable_events': len(events)-count})
    return pd.DataFrame(summaries), pd.DataFrame(paired)


def calibration_table(rows):
    frame = pd.DataFrame(rows)
    # Each event contributes equal total weight within a target, regardless of
    # entrant count or number of nodes. Fixed bins are not outcome-selected.
    frame['weight'] = 1 / frame.groupby(['model', 'target', 'tournament_id']).probability.transform('size')
    frame['bin'] = np.minimum((frame.probability * 10).astype(int), 9)
    output = []
    for (model, target, bin_id), group in frame.groupby(['model', 'target', 'bin']):
        output.append({'model': model, 'target': target, 'bin': int(bin_id), 'rows': len(group),
                       'events': group.tournament_id.nunique(), 'event_weight': float(group.weight.sum()),
                       'mean_probability': float(np.average(group.probability, weights=group.weight)),
                       'observed_frequency': float(np.average(group.observed, weights=group.weight))})
    return pd.DataFrame(output)


def _utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('explicit timezone required')
    return parsed.astimezone(timezone.utc)


LEDGER_VERSION = 'tournament-ledger-scoring-v3'
_LEDGER_ID = ('tournament_id', 'phase_id', 'origin_id', 'target_id')
LEDGER_COHORT = ('forecast_mode', 'axis', 'target_family', 'target_kind', 'forecast_scope')
_LEDGER_KINDS = {'champion', 'binary', 'placement', 'series', 'joint', 'categorical', 'count'}
_LEDGER_DIAGNOSTICS = ('format', 'family', 'tier', 'best_of', 'axis', 'horizon',
                       'phase_id', 'season', 'entrant_count', 'checkpoint_progress', 'forecast_scope')

_FORECAST_MODES = {'pre_draw', 'post_draw', 'daily_rollforward', 'post_round'}


def monte_carlo_precision(probabilities, samples, *, structural_zeros=(), looks=1,
                          cells=None, alpha=.05, absolute_half_width=.005,
                          log_half_width=.05):
    """Outcome-blind exact binomial bounds, simultaneous over cells and planned looks.

    ``structural_zeros`` names categories whose impossibility the caller proved
    independently of simulation hits. All other categories must meet both budgets.
    The returned intervals describe numerical error, never historical uncertainty.
    """
    if type(samples) is not int or samples < 1 or type(looks) is not int or looks < 1:
        raise ValueError('samples and planned looks must be positive integers')
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError('a complete category probability mapping is required')
    categories = list(probabilities)
    values = _vector(probabilities, categories, 'MC precision', 1)
    if any(isinstance(value, (bool, np.bool_)) for value in probabilities.values()):
        raise ValueError('MC probabilities must be numeric, not boolean')
    zeros = set(structural_zeros)
    if not zeros.issubset(categories) or any(probabilities[name] != 0 for name in zeros):
        raise ValueError('structural zeros must name zero-probability categories')
    if cells is None:
        cells = len(categories)
    if type(cells) is not int or cells < len(categories):
        raise ValueError('cells must cover all categories in every claimed target')
    if not (0 < alpha < 1 and math.isfinite(absolute_half_width) and absolute_half_width > 0
            and math.isfinite(log_half_width) and log_half_width > 0):
        raise ValueError('finite positive precision limits and alpha in (0,1) required')
    counts = np.rint(values * samples).astype(np.int64)
    if (not np.allclose(values * samples, counts, atol=1e-6, rtol=0)
            or int(counts.sum()) != samples):
        raise ValueError('MC probabilities must represent integer hits summing to samples')
    tail = alpha / (2 * cells * looks)
    intervals, unsampled, log_unresolved = {}, [], []
    for category, probability, hits in zip(categories, values, counts):
        structural = category in zeros
        low = 0. if hits == 0 else float(beta.ppf(tail, hits, samples - hits + 1))
        high = 1. if hits == samples else float(beta.isf(tail, hits + 1, samples - hits))
        if structural:
            low = high = 0.
        width = (high - low) / 2
        log_width = (math.log(high) - math.log(low)) / 2 if low > 0 else None
        absolute_ok = structural or width <= absolute_half_width
        log_ok = structural or (log_width is not None and log_width <= log_half_width)
        if not structural and hits == 0:
            unsampled.append(category)
        if not log_ok:
            log_unresolved.append(category)
        intervals[category] = {
            'hits': int(hits), 'probability': float(probability), 'low': low, 'high': high,
            'absolute_half_width': width, 'log_probability_half_width': log_width,
            'marginal_resolved': bool(absolute_ok), 'log_resolved': bool(log_ok),
            'support': 'structural_zero' if structural else 'possible_unsampled' if hits == 0 else 'sampled'}
    marginal = all(item['marginal_resolved'] for item in intervals.values())
    logarithmic = all(item['log_resolved'] for item in intervals.values())
    return {
        'resolved': marginal and logarithmic, 'marginal_resolved': marginal,
        'log_resolved': logarithmic, 'samples': samples,
        'status': 'resolved' if marginal and logarithmic else 'unresolved_at_current_sample_count',
        'method': 'exact_clopper_pearson_bonferroni_categories_and_planned_looks',
        'limits': {'alpha': alpha, 'simultaneous_cells': cells, 'planned_looks': looks,
                   'per_tail_alpha': tail, 'absolute_half_width': absolute_half_width,
                   'log_probability_half_width': log_half_width},
        'structural_zero_categories': sorted(zeros), 'unsampled_categories': unsampled,
        'log_unresolved_categories': log_unresolved, 'intervals': intervals}


def _ledger_safe(value):
    """Preserve nonfinite score meaning while producing strict JSON values."""
    if isinstance(value, dict):
        return {key: _ledger_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ledger_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            raise ValueError('undefined NaN is not a reportable score')
        return 'Infinity' if value == math.inf else '-Infinity' if value == -math.inf else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _ledger_identity(row):
    if not isinstance(row, dict):
        raise ValueError('ledger rows must be mappings')
    key = tuple(row.get(field) for field in _LEDGER_ID)
    if any(not isinstance(value, str) or not value for value in key):
        raise ValueError('ledger identity requires nonempty tournament_id/phase_id/origin_id/target_id strings')
    return key


def _ledger_time(value, field):
    if not isinstance(value, str):
        raise ValueError(f'{field}: timezone-aware timestamp string required')
    try:
        return _utc(value).isoformat()
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field}: invalid timezone-aware timestamp') from error


def _ledger_mode_metadata(row, key):
    if 'forecast_mode' not in row or 'observation_day_end' not in row:
        raise ValueError(f'{key}: forecast_mode and observation_day_end are required')
    mode = row['forecast_mode']
    if mode not in _FORECAST_MODES:
        raise ValueError(f'{key}: unsupported forecast_mode')
    observation_day_end = row['observation_day_end']
    cutoff = _utc(row['cutoff'])
    if mode != 'daily_rollforward':
        if observation_day_end is not None:
            raise ValueError(f'{key}: non-daily forecast modes require null observation_day_end')
        return
    if cutoff.time() != datetime.min.time():
        raise ValueError(f'{key}: daily_rollforward cutoff must be UTC midnight')
    if not isinstance(observation_day_end, str):
        raise ValueError(f'{key}: daily_rollforward requires ISO observation_day_end')
    try:
        day_end = date.fromisoformat(observation_day_end)
    except ValueError as error:
        raise ValueError(f'{key}: daily_rollforward requires ISO observation_day_end') from error
    if day_end != cutoff.date() - timedelta(days=1):
        raise ValueError(f'{key}: observation_day_end must be the UTC date before cutoff')


def _ledger_forecast(raw):
    key = _ledger_identity(raw)
    row = dict(raw)
    for field in ('model', 'phase_id', 'axis', 'target_kind'):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f'{key}: nonempty {field} required')
    if row['target_kind'] not in _LEDGER_KINDS:
        raise ValueError(f'{key}: unsupported target_kind')
    if row['axis'] not in ('axis1', 'axis2', 'axis3', 'axis4'):
        raise ValueError(f'{key}: axis must be axis1, axis2, axis3 or axis4')
    if type(row.get('resolved')) is not bool:
        raise ValueError(f'{key}: resolved must be explicitly boolean')
    row['cutoff'] = _ledger_time(row.get('cutoff'), 'cutoff')
    if row.get('edition_start_at') is None:
        if (row.get('edition_start_status') != 'unverified_whole_edition_boundary'
                or row.get('forecast_scope') not in ('phase_start', 'remaining_phase')):
            raise ValueError(f'{key}: unknown edition start requires explicitly unverified phase scope')
        row['edition_start_at'] = None
        row['phase_start_at'] = _ledger_time(row.get('phase_start_at'), 'phase_start_at')
    else:
        row['edition_start_at'] = _ledger_time(row['edition_start_at'], 'edition_start_at')
        row['edition_start_status'] = row.get('edition_start_status', 'verified')
        if row['edition_start_status'] == 'unverified_whole_edition_boundary':
            raise ValueError(f'{key}: unverified edition boundary must not supply an imputed timestamp')
        row['phase_start_at'] = (_ledger_time(row['phase_start_at'], 'phase_start_at')
                                 if row.get('phase_start_at') is not None else None)
    _ledger_mode_metadata(row, key)
    row['target_family'] = row.get('target_family', row['target_kind'])
    if not isinstance(row['target_family'], str) or not row['target_family']:
        raise ValueError(f'{key}: target_family must be a nonempty string')
    for field in _LEDGER_DIAGNOSTICS:
        value = row.get(field)
        if value is None:
            row[field] = 'unavailable'
        elif field == 'format' and isinstance(value, list):
            if not value or any(not isinstance(item, str) or not item for item in value) or len(set(value)) != len(value):
                raise ValueError(f'{key}: composite format must contain unique named formats')
            row[field] = sorted(value)
        elif isinstance(value, bool) or not isinstance(value, (str, int, float)) or (
                isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f'{key}: invalid diagnostic {field}')
    status = row.get('forecast_status', 'ok')
    if status not in ('ok', 'failed'):
        raise ValueError(f'{key}: forecast_status must be ok or failed')
    row['forecast_status'] = status
    if status == 'failed':
        if not isinstance(row.get('failure_reason'), str) or not row['failure_reason']:
            raise ValueError(f'{key}: failed forecasts require failure_reason')
        if row.get('probabilities') not in (None, {}):
            raise ValueError(f'{key}: a failed forecast cannot supply a probability distribution')
        return row
    probabilities = row.get('probabilities')
    if not isinstance(probabilities, dict) or not probabilities or any(
            not isinstance(category, str) or not category for category in probabilities):
        raise ValueError(f'{key}: probabilities must map exact category strings to numeric probabilities')
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
           for value in probabilities.values()):
        raise ValueError(f'{key}: probabilities must be numeric, not booleans or strings')
    categories = row.get('categories')
    if categories is None:
        if row['target_kind'] in ('placement', 'count'):
            raise ValueError(f'{key}: ordered targets require explicit ordered categories')
        categories = sorted(probabilities)
    if (not isinstance(categories, list) or any(not isinstance(item, str) for item in categories)
            or len(set(categories)) != len(categories) or set(categories) != set(probabilities)):
        raise ValueError(f'{key}: categories must match the full probability vector without duplicates')
    values = _vector(probabilities, categories, str(key), 1)
    if row['target_kind'] == 'count':
        try:
            numeric = [float(category) for category in categories]
        except ValueError as error:
            raise ValueError(f'{key}: count categories must be numeric') from error
        if (any(not math.isfinite(value) or value < 0 or not value.is_integer() for value in numeric)
                or any(left >= right for left, right in zip(numeric, numeric[1:]))):
            raise ValueError(f'{key}: count categories must be distinct ascending nonnegative integers')
    if row['target_kind'] in ('binary', 'series') and len(categories) != 2:
        raise ValueError(f'{key}: binary/series targets require exactly two categories')
    row['categories'] = list(categories)
    row['probabilities'] = dict(zip(categories, map(float, values)))
    if row['target_kind'] in ('binary', 'series'):
        row['positive_category'] = row.get('positive_category', categories[-1])
        if row['positive_category'] not in categories:
            raise ValueError(f'{key}: positive_category must be in categories')
    source = row.get('probability_source', 'unspecified')
    if source not in ('unspecified', 'exact', 'analytic', 'monte_carlo'):
        raise ValueError(f'{key}: unsupported probability_source')
    row['probability_source'] = source
    if source != 'monte_carlo':
        if any(field in row for field in ('mc_samples', 'mc_hits', 'mc_precision_resolved')):
            raise ValueError(f'{key}: MC fields require probability_source=monte_carlo')
        return row
    samples, hits = row.get('mc_samples'), row.get('mc_hits')
    if type(samples) is not int or samples < 1:
        raise ValueError(f'{key}: positive integer mc_samples required')
    if (not isinstance(hits, dict) or set(hits) != set(categories)
            or any(type(value) is not int or value < 0 for value in hits.values())
            or sum(hits.values()) != samples):
        raise ValueError(f'{key}: mc_hits must cover every category and sum to mc_samples')
    if not np.allclose(values, [hits[category] / samples for category in categories],
                       atol=TOLERANCE, rtol=0):
        raise ValueError(f'{key}: probability vector disagrees with raw MC hits')
    if type(row.get('mc_precision_resolved')) is not bool:
        raise ValueError(f'{key}: explicit mc_precision_resolved boolean required')
    zeros = row.get('structural_zero_categories', [])
    if (not isinstance(zeros, list) or any(not isinstance(item, str) for item in zeros)
            or len(set(zeros)) != len(zeros) or not set(zeros).issubset(categories)
            or any(hits[item] != 0 for item in zeros)):
        raise ValueError(f'{key}: structural_zero_categories must name only zero-hit categories')
    evidence = row.get('structural_zero_evidence', {})
    if not isinstance(evidence, dict) or any(
            not isinstance(evidence.get(category), str) or not evidence[category].strip() for category in zeros):
        raise ValueError(f'{key}: structural zeros require category-specific structural evidence')
    row['structural_zero_categories'] = zeros
    return row


def _ledger_join(forecasts, outcomes):
    indexed, cohorts, editions, origins = {}, defaultdict(lambda: defaultdict(set)), {}, {}
    for raw in forecasts:
        row = _ledger_forecast(raw)
        key = _ledger_identity(row)
        model_key = (*key, row['model'])
        if model_key in indexed:
            raise ValueError(f'duplicate forecast identity: {model_key}')
        indexed[model_key] = row
        group = tuple(row[field] for field in LEDGER_COHORT)
        cohorts[group][row['model']].add(key)
        for mapping, identity, value, label in (
                (editions, key[0], row['edition_start_at'], 'edition start'),
                (origins, key[:3], row['cutoff'], 'origin cutoff')):
            if identity in mapping and mapping[identity] != value:
                raise ValueError(f'{identity}: inconsistent {label}')
            mapping[identity] = value
    for group, models in cohorts.items():
        identities = list(models.values())
        if any(keys != identities[0] for keys in identities):
            raise ValueError(f'{group}: all compared models must cover the same origin/target cohort')
    by_target = defaultdict(list)
    for row in indexed.values():
        by_target[_ledger_identity(row)].append(row)
    common_fields = ('phase_id', 'axis', 'target_kind', 'target_family', 'cutoff',
                     'edition_start_at', 'edition_start_status', 'phase_start_at',
                     'forecast_mode', 'observation_day_end',
                     'resolved', *_LEDGER_DIAGNOSTICS)
    for key, rows in by_target.items():
        for field in common_fields:
            if any(row[field] != rows[0][field] for row in rows[1:]):
                raise ValueError(f'{key}: model mismatch in {field}')
        available = [row for row in rows if row['forecast_status'] == 'ok']
        if available:
            first = available[0]
            for row in available[1:]:
                if (row['categories'] != first['categories']
                        or row.get('positive_category') != first.get('positive_category')):
                    raise ValueError(f'{key}: compared models require identical categories and orientation')
    observed = {}
    for row in outcomes:
        key = _ledger_identity(row)
        if 'model' in row:
            raise ValueError('outcomes must be model-independent')
        if key in observed:
            raise ValueError(f'duplicate outcome identity: {key}')
        if key not in by_target:
            raise ValueError(f'outcome without matching forecast target: {key}')
        if 'observed' not in row:
            raise ValueError(f'{key}: outcome must explicitly contain observed (null means pending)')
        category = row['observed']
        if category is not None and (not isinstance(category, str) or not category):
            raise ValueError(f'{key}: observed must be an exact category string or null')
        for forecast in by_target[key]:
            if category is not None and forecast['forecast_status'] == 'ok' and category not in forecast['categories']:
                raise ValueError(f'{key}: outcome is absent from the complete probability vector')
        observed[key] = category
    return indexed, by_target, observed, editions


def _ledger_target_scores(indexed, by_target, observed):
    output = []
    fields = (*_LEDGER_ID, 'model', 'axis', 'edition_start_at', 'edition_start_status', 'phase_start_at', 'cutoff',
              'forecast_mode', 'observation_day_end', 'target_kind', 'target_family',
              *_LEDGER_DIAGNOSTICS)
    for row in indexed.values():
        key = _ledger_identity(row)
        actual = observed.get(key)
        result = {field: row[field] for field in fields}
        result.update(observed=actual, forecast_status=row['forecast_status'], resolved=row['resolved'])
        result.update({field: row[field] for field in (
            'probabilities', 'categories', 'probability_source', 'mc_samples', 'mc_hits',
            'mc_precision_resolved', 'mc_precision', 'structural_zero_categories',
            'structural_zero_evidence') if field in row})
        failed = [peer for peer in by_target[key] if peer['forecast_status'] == 'failed']
        if row['resolved']:
            status = 'excluded_resolved'
        elif row['forecast_status'] == 'failed':
            status = 'forecast_failed'
        elif failed:
            status = 'cohort_forecast_failure'
        elif actual is None:
            status = 'pending_outcome'
        else:
            status = 'scored'
        result['status'] = status
        result['failure_reasons'] = sorted({peer['failure_reason'] for peer in failed})
        metrics = ['brier', 'log_loss'] + (
            ['rps'] if row['target_kind'] == 'placement' else ['count_rps'] if row['target_kind'] == 'count' else [])
        scores, metric_status = dict.fromkeys(metrics), dict.fromkeys(metrics, status)
        if status == 'scored':
            categories = row['categories']
            probabilities = np.asarray([row['probabilities'][item] for item in categories])
            target = np.asarray([item == actual for item in categories], dtype=float)
            if row['target_kind'] in ('binary', 'series'):
                positive = row['positive_category']
                scores['brier'] = (row['probabilities'][positive] - int(actual == positive))**2
            else:
                scores['brier'] = float(np.square(probabilities - target).sum())
            scores['log_loss'] = _log_score(row['probabilities'][actual])
            if row['target_kind'] in ('placement', 'count'):
                rps = 'rps' if row['target_kind'] == 'placement' else 'count_rps'
                if len(categories) > 1:
                    scores[rps] = float(np.square(np.cumsum(probabilities - target)[:-1]).mean())
                else:
                    metric_status[rps] = 'unavailable_single_ordered_category'
            for metric, value in scores.items():
                if value is not None:
                    metric_status[metric] = 'infinite' if math.isinf(value) else 'scored'
            if row['probability_source'] == 'monte_carlo':
                if not row['mc_precision_resolved']:
                    scores = dict.fromkeys(metrics)
                    metric_status = dict.fromkeys(metrics, 'mc_precision_unresolved')
                    result['status'] = 'mc_precision_unresolved'
                elif row['mc_hits'][actual] == 0 and actual not in row['structural_zero_categories']:
                    scores['log_loss'] = None
                    metric_status['log_loss'] = 'mc_zero_hits_unresolved'
                    result['status'] = 'mc_log_unresolved'
        result.update(scores=scores, metric_status=metric_status)
        output.append(result)
    return output


def _ledger_counts(rows, metric=None):
    unique = lambda fields: len({tuple(row[field] for field in fields) for row in rows})
    eligible = [row for row in rows if not row['resolved'] and row['observed'] is not None]
    counts = {
        'forecast_rows': len(rows), 'editions': unique(('tournament_id',)),
        'phases': unique(('tournament_id', 'phase_id')),
        'origins': unique(_LEDGER_ID[:3]),
        'targets': unique(_LEDGER_ID),
        'series': len({_ledger_identity(row) for row in rows if row['target_kind'] == 'series'}),
        'resolved_targets': sum(row['resolved'] for row in rows),
        'pending_targets': sum(not row['resolved'] and row['observed'] is None for row in rows),
        'observed_targets': sum(row['observed'] is not None for row in rows),
        'eligible_targets': len(eligible),
        'forecast_failed_rows': sum(row['forecast_status'] == 'failed' for row in rows),
        'cohort_failed_targets': sum(row['status'] == 'cohort_forecast_failure' for row in rows),
        'failure_reason_counts': dict(Counter(reason for row in rows for reason in row['failure_reasons'])),
    }
    counts.update(_ledger_month_support(
        (row['tournament_id'] for row in rows),
        {row['tournament_id']: row['edition_start_at'] for row in rows}))
    if metric is not None:
        statuses = Counter(row['metric_status'][metric] for row in eligible)
        counts.update(scored_targets=statuses['scored'] + statuses['infinite'],
                      infinite_targets=statuses['infinite'],
                      unresolved_targets=sum(count for status, count in statuses.items()
                                             if status not in ('scored', 'infinite')),
                      score_status_counts=dict(sorted(statuses.items())))
    return counts


def _ledger_average(values):
    if not values or any(value is None for value in values):
        return None
    return math.inf if any(value == math.inf for value in values) else float(np.mean(values))


def _ledger_origin_scores(rows):
    groups = defaultdict(list)
    for row in rows:
        for metric in row['scores']:
            key = (*_ledger_identity(row)[:3], row['model'],
                   *(row[field] for field in LEDGER_COHORT), metric)
            groups[key].append(row)
    output = []
    for key, group in sorted(groups.items()):
        metric = key[-1]
        eligible = [row for row in group if not row['resolved'] and row['observed'] is not None]
        output.append(dict(zip((*_LEDGER_ID[:3], 'model', *LEDGER_COHORT, 'metric'), key)) | {
            'mean': _ledger_average([row['scores'][metric] for row in eligible]),
            'counts': _ledger_counts(group, metric)})
    return output


def _ledger_edition_values(rows, metric):
    origins = defaultdict(list)
    for row in rows:
        if not row['resolved'] and row['observed'] is not None:
            origins[_ledger_identity(row)[:3]].append(row['scores'][metric])
    editions = defaultdict(list)
    for (edition, _, _), values in origins.items():
        editions[edition].append(_ledger_average(values))
    return {edition: _ledger_average(values) for edition, values in sorted(editions.items())}


def _ledger_bootstrap(values, editions, replicates, seed, cache):
    """Moving calendar-month blocks; an edition always moves with every origin."""
    names = tuple(values)
    result = {'ci_low': None, 'ci_high': None, 'probability_non_improvement': None,
              'inference_status': 'inconclusive_no_observed_targets', 'sensitivity': []}
    if not names:
        return result
    if any(value is None or not math.isfinite(value) for value in values.values()):
        result['inference_status'] = 'inconclusive_unresolved_or_infinite'
        return result
    support = _ledger_month_support(names, editions)
    result.update(support)
    if support['editions_with_unknown_start']:
        result['inference_status'] = 'inconclusive_unknown_edition_months'
        return result
    months = [int(editions[name][:4]) * 12 + int(editions[name][5:7]) - 1 for name in names]
    occupied = len(set(months))
    if len(names) < 10 or occupied < 6:
        result['inference_status'] = 'inconclusive_sparse_blocks'
        return result
    if replicates < 5000:
        result['inference_status'] = 'inconclusive_insufficient_bootstrap_replicates'
        return result
    span = max(months) - min(months) + 1
    offsets = np.asarray(months) - min(months)
    vector = np.asarray(list(values.values()), dtype=float)
    for length in (1, 2, 3):
        if occupied < 6 * length:
            result['sensitivity'].append({
                'block_months': length, 'ci_low': None, 'ci_high': None,
                'probability_non_improvement': None, 'requested_replicates': replicates,
                'nonempty_replicates': None, 'status': 'inconclusive_sparse_blocks'})
            continue
        cache_key = (names, length)
        if cache_key not in cache:
            rng = np.random.default_rng(seed + length - 1)
            starts = rng.integers(span - length + 1, size=(replicates, math.ceil(span / length)))
            sampled = (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[:, :span]
            month_counts = np.zeros((replicates, span), dtype=np.int32)
            np.add.at(month_counts, (np.arange(replicates)[:, None], sampled), 1)
            weights = month_counts[:, offsets]
            totals = weights.sum(axis=1)
            cache[cache_key] = (weights, totals)
        weights, totals = cache[cache_key]
        valid = totals > 0
        draws = (weights[valid] @ vector) / totals[valid]
        # Empty-month resamples are never silently discarded to certify precision.
        complete = bool(valid.all())
        bounds = np.quantile(draws, [.025, .975]) if complete else [None, None]
        item = {'block_months': length, 'ci_low': bounds[0], 'ci_high': bounds[1],
                'probability_non_improvement': float(np.mean(draws >= 0)) if complete else None,
                'requested_replicates': replicates, 'nonempty_replicates': int(valid.sum()),
                'status': 'descriptive' if complete else 'inconclusive_empty_calendar_resamples'}
        result['sensitivity'].append(item)
        if length == 1:
            result.update(ci_low=item['ci_low'], ci_high=item['ci_high'],
                          probability_non_improvement=item['probability_non_improvement'],
                          inference_status='descriptive_month_block_bootstrap' if complete else item['status'])
    return result


def _ledger_summaries(rows, editions, replicates, seed, cache):
    groups = defaultdict(list)
    for row in rows:
        for metric in row['scores']:
            groups[(*(row[field] for field in LEDGER_COHORT), metric)].append(row)
    summaries, comparisons = [], []
    for key, group in sorted(groups.items()):
        identity = dict(zip((*LEDGER_COHORT, 'metric'), key))
        metric = key[-1]
        models = sorted({row['model'] for row in group})
        model_values = {}
        for model in models:
            selected = [row for row in group if row['model'] == model]
            values = _ledger_edition_values(selected, metric)
            model_values[model] = values
            uncertainty = _ledger_bootstrap(values, editions, replicates, seed, cache)
            uncertainty.pop('probability_non_improvement')
            for sensitivity in uncertainty['sensitivity']:
                sensitivity.pop('probability_non_improvement')
            summaries.append(identity | {
                'model': model, 'mean': _ledger_average(list(values.values())),
                'counts': _ledger_counts(selected, metric),
                'scored_editions': sum(value is not None for value in values.values()),
                'edition_values': [{'tournament_id': edition, 'mean': value} for edition, value in values.items()],
                'date_start': min(row['cutoff'] for row in selected),
                'edition_start_min': min((row['edition_start_at'] for row in selected
                                          if row['edition_start_at'] is not None), default=None),
                'edition_start_max': max((row['edition_start_at'] for row in selected
                                          if row['edition_start_at'] is not None), default=None),
                'date_end': max(row['cutoff'] for row in selected), **uncertainty})
        for model, reference in itertools.combinations(models, 2):
            left, right = model_values[model], model_values[reference]
            if set(left) != set(right):
                raise ValueError('paired edition score cohorts differ')
            delta = {
                edition: left[edition] - right[edition]
                if left[edition] is not None and right[edition] is not None
                and math.isfinite(left[edition]) and math.isfinite(right[edition]) else None
                for edition in left}
            selected = [row for row in group if row['model'] == model]
            reference_rows = [row for row in group if row['model'] == reference]
            reference_targets = {_ledger_identity(row): row for row in reference_rows}
            pairs = [(row, reference_targets[_ledger_identity(row)]) for row in selected
                     if not row['resolved'] and row['observed'] is not None]
            finite_pairs = sum(
                all(item['scores'][metric] is not None and math.isfinite(item['scores'][metric])
                    for item in pair) for pair in pairs)
            comparisons.append(identity | {
                'model': model, 'reference': reference, 'delta': _ledger_average(list(delta.values())),
                'model_mean': _ledger_average(list(left.values())),
                'reference_mean': _ledger_average(list(right.values())),
                'counts': _ledger_counts(selected, metric),
                'reference_counts': _ledger_counts(reference_rows, metric),
                'paired_eligible_targets': len(pairs), 'paired_finite_targets': finite_pairs,
                'paired_scored_targets': sum(all(item['scores'][metric] is not None for item in pair)
                                             for pair in pairs),
                **_ledger_bootstrap(delta, editions, replicates, seed, cache)})
    return summaries, comparisons


def _ledger_calibration_regression(probabilities, observed, weights, edition_count):
    unavailable = {'status': 'inconclusive_sparse_editions', 'intercept': None, 'slope': None,
                   'uncertainty': 'not_estimated; descriptive diagnostic, not a fitted production calibrator'}
    if edition_count < 10 or len(probabilities) < 20:
        return unavailable
    if len(set(observed)) < 2:
        return unavailable | {'status': 'unavailable_single_outcome_class'}
    if np.any((probabilities == 0) | (probabilities == 1)):
        return unavailable | {'status': 'unavailable_boundary_logits_no_clipping'}
    logits = np.log(probabilities) - np.log1p(-probabilities)
    if np.ptp(logits) <= TOLERANCE:
        return unavailable | {'status': 'unavailable_constant_probabilities'}
    positive, negative = logits[observed == 1], logits[observed == 0]
    if positive.min() >= negative.max() or negative.min() >= positive.max():
        return unavailable | {'status': 'unavailable_complete_or_quasi_separation'}
    design = np.column_stack((np.ones(len(logits)), logits))

    def objective(parameters):
        linear = design @ parameters
        loss = float(np.sum(weights * (np.logaddexp(0, linear) - observed * linear)))
        gradient = design.T @ (weights * (expit(linear) - observed))
        return loss, gradient

    fit = minimize(objective, np.array([0., 1.]), jac=True, method='BFGS',
                   options={'gtol': 1e-8, 'maxiter': 1000})
    if not fit.success or not np.isfinite(fit.x).all():
        return unavailable | {'status': 'unavailable_regression_nonconvergence'}
    fitted = expit(design @ fit.x)
    information = design.T @ ((weights * fitted * (1 - fitted))[:, None] * design)
    if not np.isfinite(information).all() or np.linalg.cond(information) > 1e10:
        return unavailable | {'status': 'unavailable_degenerate_information'}
    return unavailable | {'status': 'estimated', 'intercept': float(fit.x[0]),
                          'slope': float(fit.x[1])}


def _ledger_auc(probabilities, observed, weights):
    positive, negative = weights[observed == 1].sum(), weights[observed == 0].sum()
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(probabilities, kind='stable')
    p, y, w = probabilities[order], observed[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    positive_ties = np.add.reduceat(w * y, starts)
    negative_ties = np.add.reduceat(w * (1-y), starts)
    previous_negatives = np.cumsum(negative_ties) - negative_ties
    return float(np.sum(positive_ties * (previous_negatives + .5 * negative_ties)) / (positive * negative))


def _ledger_reliability(cells):
    probabilities = np.asarray([cell['probability'] for cell in cells])
    observed = np.asarray([cell['observed'] for cell in cells], dtype=float)
    weights = np.asarray([cell['weight'] for cell in cells])
    bins = np.minimum((probabilities * 10).astype(int), 9)
    total = float(weights.sum())
    table, ece, mce, reliability, resolution = [], 0., 0., 0., 0.
    prevalence = float(np.average(observed, weights=weights)) if total else None
    for bin_id in range(10):
        selected = bins == bin_id
        indices = np.flatnonzero(selected)
        mass = float(weights[selected].sum())
        probability = float(np.average(probabilities[selected], weights=weights[selected])) if mass else None
        frequency = float(np.average(observed[selected], weights=weights[selected])) if mass else None
        edition_count = len({cells[index]['tournament_id'] for index in indices})
        gap = abs(probability-frequency) if mass else None
        if mass:
            fraction = mass / total
            ece += fraction * gap
            mce = max(mce, gap)
            reliability += fraction * gap**2
            resolution += fraction * (frequency-prevalence)**2
        table.append({
            'bin': bin_id, 'lower': bin_id/10, 'upper': (bin_id+1)/10,
            'upper_inclusive': bin_id == 9, 'cells': len(indices), 'editions': edition_count,
            'origins': len({_ledger_identity(cells[index])[:3] for index in indices}),
            'targets': len({_ledger_identity(cells[index]) for index in indices}),
            'edition_weight': mass, 'weight_fraction': mass/total if total else 0.,
            'mean_probability': probability, 'observed_frequency': frequency,
            'absolute_error': gap,
            'support': 'unavailable' if not mass else 'sparse' if edition_count < 10 else 'descriptive_only',
            'ci_low': None, 'ci_high': None})
    return {
        'bins': table, 'ece_10bin': ece if total else None,
        'mce_10bin': mce if total else None,
        'prevalence': prevalence, 'reliability': reliability if total else None,
        'resolution': resolution if total else None,
    }, probabilities, observed, weights


def _ledger_calibration(indexed, scores, editions, replicates, seed, cache):
    score_index = {(*_ledger_identity(row), row['model']): row for row in scores}
    matched_status = defaultdict(list)
    for row in scores:
        matched_status[_ledger_identity(row)].append(row['metric_status']['brier'])
    groups = defaultdict(list)
    for row in indexed.values():
        groups[(*(row[field] for field in LEDGER_COHORT), row['model'])].append(row)
    calibration, series = [], []
    for key, rows in sorted(groups.items()):
        identity = dict(zip((*LEDGER_COHORT, 'model'), key))
        selected, exclusions = [], Counter()
        for row in rows:
            score = score_index[(*_ledger_identity(row), row['model'])]
            if score['resolved']:
                exclusions['resolved'] += 1
            elif score['observed'] is None:
                exclusions['pending_outcome'] += 1
            elif any(status != 'scored' for status in matched_status[_ledger_identity(row)]):
                exclusions['matched_cohort_unscorable_brier'] += 1
            else:
                selected.append(row)
        origin_sizes = Counter(_ledger_identity(row)[:3] for row in selected)
        edition_origins = Counter(edition for edition, phase, origin in origin_sizes)
        cells = []
        for row in selected:
            categories = [row['positive_category']] if row['target_kind'] in ('binary', 'series') else row['categories']
            weight = 1 / (origin_sizes[_ledger_identity(row)[:3]]
                          * edition_origins[row['tournament_id']] * len(categories))
            actual = score_index[(*_ledger_identity(row), row['model'])]['observed']
            cells.extend({
                **{field: row[field] for field in _LEDGER_ID}, 'category': category,
                'probability': row['probabilities'][category], 'observed': int(category == actual),
                'weight': weight} for category in categories)
        reliability, probabilities, observed, weights = _ledger_reliability(cells)
        edition_count = len(edition_origins)
        for bin_row in reliability['bins']:
            bin_cells = [cell | {'value': cell['observed']-cell['probability']} for cell in cells
                         if min(int(cell['probability']*10), 9) == bin_row['bin']]
            interval = _ledger_diagnostic_interval(
                bin_cells, editions, replicates, seed, cache, alpha=.05/(10*len(groups)))
            bin_row.update(
                **_ledger_month_support((cell['tournament_id'] for cell in bin_cells), editions),
                observed_minus_predicted=(bin_row['observed_frequency']-bin_row['mean_probability']
                                         if bin_cells else None),
                ci_low=interval['ci_low'], ci_high=interval['ci_high'],
                uncertainty=interval, band_estimand='observed_minus_predicted',
                uncertainty_family_cells=10*len(groups))
        calibration.append(identity | {
            'forecast_rows': len(rows), 'included_targets': len(selected), 'editions': edition_count,
            'origins': len(origin_sizes), 'cells': len(cells), 'exclusions': dict(exclusions),
            'bins': reliability['bins'], 'ece_10bin': reliability['ece_10bin'],
            'mce_10bin': reliability['mce_10bin'],
            'regression': _ledger_calibration_regression(probabilities, observed, weights, edition_count),
            'status': 'inconclusive_incomplete_cohort' if exclusions['matched_cohort_unscorable_brier']
                      else 'inconclusive_unknown_edition_months' if any(editions[name] is None for name in edition_origins)
                      else 'inconclusive_sparse_editions' if edition_count < 10
                      or _ledger_month_support(edition_origins, editions)['occupied_months'] < 6 else 'descriptive_only',
            'limitations': [
                'Equal editions, equal origins within edition, equal targets within origin; categorical cells additionally weighted by 1/category_count.',
                'Categorical calibration is pooled one-vs-rest, not a test of joint independence or full vector calibration.',
                'Bins are fixed equal-width intervals; empty bins are unavailable, never zero-error evidence.',
                'Bonferroni bands cover all displayed cohort/model/bin cells only where independent month/edition support and bootstrap tails suffice; degenerate cells remain inconclusive. No locked power analysis or calibration certification.']})
        if key[3] != 'series':
            continue
        if cells:
            errors = np.square(probabilities-observed)
            brier = float(np.average(errors, weights=weights))
            accuracy = np.where(probabilities == .5, .5, ((probabilities > .5) == observed).astype(float))
            uncertainty = reliability['prevalence'] * (1-reliability['prevalence'])
            decomposition = {
                'uncertainty': uncertainty, 'reliability': reliability['reliability'],
                'resolution': reliability['resolution'],
                'within_bin_residual': brier - (
                    uncertainty - reliability['resolution'] + reliability['reliability']),
                'estimator': 'edition_weighted_fixed_10bin_murphy_with_exact_residual',
                'limitations': 'Finite-sample binned plug-in estimates are biased; resolution/reliability are descriptive. The residual retains within-bin forecast variation; not an exact three-term decomposition of unbinned forecasts.'}
            log_scores = [score_index[(*_ledger_identity(row), row['model'])]['scores']['log_loss']
                          for row in selected]
            series.append(identity | {
                'editions': edition_count, 'origins': len(origin_sizes), 'series': len(cells),
                'accuracy': float(np.average(accuracy, weights=weights)),
                'auc': _ledger_auc(probabilities, observed, weights), 'brier': brier,
                'auc_status': 'descriptive_dependent_series' if len(set(observed)) == 2 else 'unavailable_single_class',
                'brier_decomposition': decomposition,
                'micro': {'accuracy': float(np.mean(accuracy)),
                          'auc': _ledger_auc(probabilities, observed, np.ones(len(cells))),
                          'brier': float(np.mean(errors)), 'log_loss': _ledger_average(log_scores),
                          'log_loss_status': 'unresolved' if any(value is None for value in log_scores)
                                             else 'infinite' if math.inf in log_scores else 'scored',
                          'series': len(cells)},
                'exclusions': dict(exclusions), 'inference_status': 'descriptive_only_no_series_pseudoreplication'})
        else:
            series.append(identity | {'editions': 0, 'origins': 0, 'series': 0,
                                      'accuracy': None, 'auc': None, 'brier': None,
                                      'brier_decomposition': None, 'micro': None,
                                      'exclusions': dict(exclusions),
                                      'inference_status': 'inconclusive_no_observed_series'})
    return calibration, series


def _ledger_month_support(names, editions):
    names = set(names)
    unknown = sorted(name for name in names if editions[name] is None)
    return {'occupied_months': len({editions[name][:7] for name in names if editions[name] is not None}),
            'editions_with_known_start': len(names)-len(unknown),
            'editions_with_unknown_start': len(unknown), 'unknown_start_editions': unknown}




def _ledger_diagnostic_interval(cells, editions, replicates, seed, cache, *, alpha=.05):
    """Calendar-block ratio bootstrap; phases and origins stay in their true edition."""
    unavailable = {'ci_low': None, 'ci_high': None, 'confidence_level': 1-alpha,
                   'status': 'inconclusive_sparse_blocks'}
    totals, sums = defaultdict(float), defaultdict(float)
    for cell in cells:
        edition = cell['tournament_id']
        totals[edition] += cell['weight']
        sums[edition] += cell['weight'] * cell['value']
    names = tuple(sorted(totals))
    support = _ledger_month_support(names, editions)
    unavailable.update(support)
    if support['editions_with_unknown_start']:
        return unavailable | {'status': 'inconclusive_unknown_edition_months'}
    if len(names) < 10 or support['occupied_months'] < 6:
        return unavailable
    if replicates < 5000 or replicates * alpha / 2 < 10:
        return unavailable | {'status': 'inconclusive_insufficient_tail_replicates'}
    # Reuse the same whole-edition calendar draws as score intervals.
    uncertainty = _ledger_bootstrap({name: 0. for name in names}, editions, replicates, seed, cache)
    if uncertainty['inference_status'] != 'descriptive_month_block_bootstrap':
        return unavailable | {'status': uncertainty['inference_status']}
    weights, _ = cache[(names, 1)]
    numerator = np.asarray([sums[name] for name in names])
    denominator = np.asarray([totals[name] for name in names])
    edition_means = numerator / denominator
    if np.ptp(edition_means) <= TOLERANCE:
        return unavailable | {'status': 'inconclusive_degenerate_edition_variation'}
    draws = (weights @ numerator) / (weights @ denominator)
    low, high = np.quantile(draws, [alpha/2, 1-alpha/2])
    return {'ci_low': float(low), 'ci_high': float(high), 'confidence_level': 1-alpha,
            'status': 'descriptive_month_block_bootstrap', 'replicates': replicates,
            'unit': 'whole_edition_in_edition_start_month'}


def _ledger_distribution_diagnostics(indexed, scores, editions, replicates, seed, cache):
    """Plot-ready discrete predictive sets, frequencies and genuinely ordered PIT."""
    score_index = {(*_ledger_identity(row), row['model']): row for row in scores}
    matched = defaultdict(list)
    for row in scores:
        matched[_ledger_identity(row)].append(row['metric_status']['brier'])
    groups = defaultdict(list)
    for row in indexed.values():
        groups[(*(row[field] for field in LEDGER_COHORT), row['model'])].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        if key[3] in ('binary', 'series'):
            continue
        selected, exclusions = [], Counter()
        for row in rows:
            scored = score_index[(*_ledger_identity(row), row['model'])]
            if scored['resolved']:
                exclusions['resolved'] += 1
            elif scored['observed'] is None:
                exclusions['pending_outcome'] += 1
            elif any(status != 'scored' for status in matched[_ledger_identity(row)]):
                exclusions['matched_cohort_unscorable_brier'] += 1
            else:
                selected.append(row)
        origin_sizes = Counter(_ledger_identity(row)[:3] for row in selected)
        edition_origins = Counter(edition for edition, phase, origin in origin_sizes)
        set_rows, pit_rows, frequencies = [], [], defaultdict(list)
        for row in sorted(selected, key=_ledger_identity):
            identity = {field: row[field] for field in _LEDGER_ID}
            actual = score_index[(*_ledger_identity(row), row['model'])]['observed']
            weight = 1 / (origin_sizes[_ledger_identity(row)[:3]] * edition_origins[row['tournament_id']])
            categories, probabilities = row['categories'], row['probabilities']
            # Category-string ordering is fixed and independent of observed labels.
            ranked = sorted(categories, key=lambda category: (-probabilities[category], category))
            for level in (.5, .8, .9, .95):
                included, mass = [], 0.
                for category in ranked:
                    included.append(category)
                    mass += probabilities[category]
                    if mass >= level:
                        break
                set_rows.append(identity | {
                    'nominal_level': level, 'included_categories': included, 'included_mass': mass,
                    'set_size': len(included), 'support_size': len(categories),
                    'covered': int(actual in included), 'coverage_minus_mass': int(actual in included)-mass,
                    'weight': weight})
            for category in categories:
                frequencies[category].append(identity | {
                    'probability': probabilities[category], 'observed': int(actual == category),
                    'value': int(actual == category)-probabilities[category], 'weight': weight})
            if row['target_kind'] in ('placement', 'count'):
                payload = json.dumps([seed, *_ledger_identity(row)], separators=(',', ':'))
                random_seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], 'big')
                uniform = float(np.random.default_rng(random_seed).random())
                position = categories.index(actual)
                lower = sum(probabilities[category] for category in categories[:position])
                pit_rows.append(identity | {'pit': lower + uniform * probabilities[actual],
                                            'weight': weight, 'observed': actual})
        def support(items):
            return {'targets': len(items),
                    'editions': len({item['tournament_id'] for item in items}),
                    'origins': len({_ledger_identity(item)[:3] for item in items}),
                    **_ledger_month_support((item['tournament_id'] for item in items), editions)}

        def average(items, field):
            return (float(np.average([item[field] for item in items],
                                     weights=[item['weight'] for item in items])) if items else None)

        sets = []
        for level in (.5, .8, .9, .95):
            items = [item for item in set_rows if item['nominal_level'] == level]
            interval = _ledger_diagnostic_interval(
                [item | {'value': item['coverage_minus_mass']} for item in items],
                editions, replicates, seed, cache, alpha=.05/4)
            sets.append({'nominal_level': level, **support(items),
                         'coverage': average(items, 'covered'),
                         'actual_included_mass': average(items, 'included_mass'),
                         'mean_set_size': average(items, 'set_size'),
                         'mean_support_size': average(items, 'support_size'),
                         'coverage_minus_mass': average(items, 'coverage_minus_mass'),
                         'coverage_minus_mass_interval': interval})
        category_frequencies = []
        for category, items in sorted(frequencies.items()):
            category_frequencies.append({
                'category': category, **support(items), 'mean_probability': average(items, 'probability'),
                'observed_frequency': average(items, 'observed'),
                'observed_minus_predicted_interval': _ledger_diagnostic_interval(
                    items, editions, replicates, seed, cache, alpha=.05/len(frequencies))})
        pit_bins = []
        for bin_id in range(10):
            items = [item | {'value': int(min(int(item['pit']*10), 9) == bin_id)} for item in pit_rows]
            pit_bins.append({
                'bin': bin_id, 'lower': bin_id/10, 'upper': (bin_id+1)/10,
                'forecast_count': sum(item['value'] for item in items), **support(items),
                'edition_weighted_fraction': average(items, 'value'),
                'interval': _ledger_diagnostic_interval(items, editions, replicates, seed, cache, alpha=.05/10)})
        output.append(dict(zip((*LEDGER_COHORT, 'model'), key)) | {
            'forecast_rows': len(rows), 'included_targets': len(selected),
            'editions': len(edition_origins), 'origins': len(origin_sizes), 'exclusions': dict(exclusions),
            'status': 'inconclusive_incomplete_cohort' if exclusions['matched_cohort_unscorable_brier']
                      else 'inconclusive_unknown_edition_months' if any(editions[name] is None for name in edition_origins)
                      else 'inconclusive_sparse_blocks' if len(edition_origins) < 10
                      or _ledger_month_support(edition_origins, editions)['occupied_months'] < 6 else 'descriptive_only',
            'predictive_sets': sets, 'predictive_set_rows': set_rows,
            'category_frequencies': category_frequencies,
            'pit': {'status': 'not_applicable_unordered_categories' if key[3] not in ('placement', 'count')
                    else 'inconclusive_no_scoreable_targets' if not pit_rows else 'descriptive_only',
                    'seed': seed, 'rows': pit_rows, 'bins': pit_bins if pit_rows else [],
                    'randomization': 'SHA256(seed, tournament_id, phase_id, origin_id, target_id); common uniforms across compared models'},
            'uncertainty': 'Whole-edition month-block descriptive bands; Bonferroni within each displayed diagnostic family, not across post-hoc strata; no calibration certification.'})
    return output


def evaluate_ledger(forecast_rows, outcome_rows, *, bootstrap=5000, seed=82):
    """Evaluate immutable forecasts joined one-to-one to independent outcomes.

    Required forecast fields: tournament_id, origin_id, target_id, model,
    phase_id, axis (axis1..axis4), edition_start_at, cutoff (aware ISO timestamps),
    Edition start may be null only with edition_start_status=
    unverified_whole_edition_boundary, an explicit aware phase_start_at and phase_start/
    remaining_phase scope. Such rows retain true edition weighting but cannot supply
    a month-block interval; phase timestamps never stand in for edition timestamps.
    forecast_mode (pre_draw/post_draw/daily_rollforward/post_round), observation_day_end
    (null except daily_rollforward, which requires the prior UTC date), target_kind
    (champion/binary/placement/series/joint/categorical/count), probabilities (complete
    category-string -> finite probability vector), resolved (bool). Placement requires
    categories in official best-to-worst band order; count requires explicit ascending
    nonnegative integer category strings and uses separately named count_rps. Tied teams retain
    the same observed band. Binary/series have exactly two categories and use
    positive_category, defaulting to the last supplied category, otherwise last
    lexicographically sorted category. Champion/joint/placement Brier is the
    categorical SUM, binary/series Brier is the single Bernoulli squared error.

    Optional target_family distinguishes finalist, destination-specific
    qualification, node reach etc.; default target_kind. Target semantics and
    observed outcomes must be supplied, never inferred from champion rankings.
    Optional diagnostic fields: format (string or composite list), family, tier,
    best_of (integer or mixed), horizon, season, entrant_count,
    checkpoint_progress, forecast_scope. Missing diagnostics explicitly become unavailable.
    Forecast scope also partitions every score and diagnostic cohort; whole-tournament
    and phase-start estimands are never averaged together.
    Optional probability_source is unspecified/exact/analytic/monte_carlo.
    Monte Carlo requires mc_samples, mc_hits and mc_precision_resolved; declared
    structural_zero_categories require category-specific structural_zero_evidence.
    Unresolved precision blocks all scores; nonstructural observed zero hits
    block LogLoss only. Unspecified zero probabilities incur infinite LogLoss.
    Failed forecasts use forecast_status='failed', failure_reason and no vector.

    Outcomes contain tournament_id/phase_id/origin_id/target_id and observed (category
    string or null). Absent rows/null are pending, not forecast failure. Model
    fields, duplicates, extra targets and incompatible vectors/cohorts raise
    ValueError. Cohorts match within forecast_mode/axis/target_family/target_kind/forecast_scope, permitting
    frozen/refreshed models only on axis3 and no flat model on joint families.
    Failed targets block scores for every model rather than improving a subset.
    Resolved targets are excluded identically, including from calibration.

    Return schema: version, protocol, coverage, target_scores, origin_scores,
    summaries, comparisons, calibration, series_diagnostics, diagnostics,
    distribution_diagnostics, verdict. Target scores retain full forecast vectors and
    numerical precision metadata for checkpoint trajectories. Distribution diagnostics
    contain predictive-set coverage/size/actual mass, category frequencies and seeded
    ordered PIT, with full denominators and support-qualified bands.
    Target scores carry scores/metric_status; summaries carry mean,
    counts, edition_values, dates, CI and block sensitivity. Comparisons use
    model minus reference (negative is better), exact matched targets and
    chronological calendar-month blocks of whole editions. Results are strict
    JSON: true infinite scores are "Infinity"; unavailable means null plus an
    explicit status/reason. No aggregate averages away missing/unresolved scores.
    """
    if type(bootstrap) is not int or bootstrap < 1 or type(seed) is not int or seed < 0:
        raise ValueError('positive integer bootstrap and nonnegative integer seed required')
    indexed, by_target, observed, editions = _ledger_join(forecast_rows, outcome_rows)
    scores = _ledger_target_scores(indexed, by_target, observed)
    cache = {}
    summaries, comparisons = _ledger_summaries(scores, editions, bootstrap, seed, cache)
    calibration, series = _ledger_calibration(indexed, scores, editions, bootstrap, seed, cache)
    distributions = _ledger_distribution_diagnostics(indexed, scores, editions, bootstrap, seed, cache)
    diagnostics = []
    for dimension in _LEDGER_DIAGNOSTICS:
        groups = defaultdict(list)
        for row in scores:
            values = row[dimension] if isinstance(row[dimension], list) else [row[dimension]]
            for value in values:
                groups[str(value)].append(row)
        for value, rows in sorted(groups.items()):
            diagnostic_summaries, diagnostic_comparisons = _ledger_summaries(
                rows, editions, bootstrap, seed, cache)
            diagnostics.append({'dimension': dimension, 'value': value,
                                'summaries': diagnostic_summaries, 'comparisons': diagnostic_comparisons})
    mode_horizon_breakdown = []
    mode_horizon_groups = defaultdict(list)
    for row in scores:
        mode_horizon_groups[(row['forecast_mode'], row['forecast_scope'], row['horizon'])].append(row)
    for (forecast_mode, forecast_scope, horizon), rows in sorted(mode_horizon_groups.items()):
        breakdown_summaries, breakdown_comparisons = _ledger_summaries(
            rows, editions, bootstrap, seed, cache)
        mode_horizon_breakdown.append({
            'forecast_mode': forecast_mode, 'forecast_scope': forecast_scope, 'horizon': horizon,
            'summaries': breakdown_summaries, 'comparisons': breakdown_comparisons})
    months = defaultdict(list)
    for edition, started in sorted(editions.items()):
        if started is not None:
            months[started[:7]].append(edition)
    unique_rows = [rows[0] for rows in by_target.values()]
    model_coverage = {
        model: _ledger_counts([row for row in scores if row['model'] == model])
        for model in sorted({row['model'] for row in scores})}
    reasons = []
    if not scores:
        reasons.append('empty_forecast_ledger')
    if any(row['forecast_status'] == 'failed' for row in scores):
        reasons.append('forecast_failures')
    if any(not row['resolved'] and row['observed'] is None for row in scores):
        reasons.append('pending_outcomes')
    if any(status.startswith('mc_') for row in scores for status in row['metric_status'].values()):
        reasons.append('monte_carlo_unresolved')
    if any(status == 'infinite' for row in scores for status in row['metric_status'].values()):
        reasons.append('infinite_proper_scores')
    if any(row['inference_status'].startswith('inconclusive') for row in summaries):
        reasons.append('insufficient_or_unscorable_independent_edition_blocks')
    reasons.append('no_locked_holdout_power_multiplicity_or_simultaneous_calibration_gate_in_ledger')
    return _ledger_safe({
        'version': LEDGER_VERSION,
        'protocol': {
            'identity': list(_LEDGER_ID), 'forecast_identity': [*_LEDGER_ID, 'model'],
            'cohort': list(LEDGER_COHORT),
            'predictive_sets': {'levels': [.5, .8, .9, .95],
                                'ties': 'descending probability, ascending exact category string',
                                'reference': 'actual included mass, never nominal mass'},
            'ordered_pit': {'kinds': ['placement', 'count'], 'seed': seed,
                            'unit': 'independent target uniforms; same uniform across compared models'},
            'weighting': 'equal targets within origin, equal origins within edition, equal editions within target family; forecast modes, scopes, axes and Brier definitions never pooled',
            'bootstrap': {'replicates': bootstrap, 'seed': seed,
                          'unit': 'whole_edition_with_all_phases_origins_in_edition_start_month',
                          'method': 'moving_calendar_month_block_percentile',
                          'block_months': [1, 2, 3], 'confidence_level': .95,
                          'minimum_descriptive_editions': 10, 'minimum_occupied_months': 6,
                          'minimum_adequacy_note': 'Conservative sparse-cell suppression, not a power calculation or certification threshold.',
                          'blocks': [{'month': month, 'editions': names} for month, names in sorted(months.items())]},
            'primary_metrics': {'axis1_champion': 'champion_log_loss',
                                'axis1_qualification_only': 'qualification_log_loss',
                                'axis2': 'series_log_loss',
                                'axis3': 'remaining_target_log_loss_refreshed_minus_frozen'},
            'corroborating_metrics': ['brier', 'placement_rps', 'count_rps', 'calibration'],
            'comparison_direction': 'model minus reference; negative favors model; pair labels sorted, never outcome-selected',
            'probability_non_improvement': 'fraction of paired bootstrap deltas >=0, not posterior probability or a calibrated significance p-value',
            'infinite_scores': 'Infinity string; no epsilon clipping and no finite-subset means',
            'pending_outcomes': 'excluded from observed denominators for every model; visible partial prospective cohort',
            'diagnostics': 'marginal slices, not pooled independent observations; composite format memberships may overlap'},
        'coverage': {
            **_ledger_month_support(editions, editions),
            'forecast_rows': len(scores), 'targets': len(by_target), 'outcome_rows': len(observed),
            'editions': len(editions),
            'origins': len({key[:3] for key in by_target}),
            'observed_targets': sum(observed.get(key) is not None for key in by_target),
            'pending_targets': sum(not rows[0]['resolved'] and observed.get(key) is None
                                   for key, rows in by_target.items()),
            'resolved_targets': sum(row['resolved'] for row in unique_rows),
            'forecast_failed_rows': sum(row['forecast_status'] == 'failed' for row in scores),
            'failure_reasons': dict(Counter(row['failure_reason'] for row in indexed.values()
                                           if row['forecast_status'] == 'failed')),
            'models': model_coverage},
        'target_scores': scores, 'origin_scores': _ledger_origin_scores(scores),
        'summaries': summaries, 'comparisons': comparisons, 'calibration': calibration,
        'series_diagnostics': series, 'diagnostics': diagnostics,
        'distribution_diagnostics': distributions,
        'mode_horizon_breakdown': mode_horizon_breakdown,
        'verdict': {'status': 'inconclusive', 'reasons': reasons, 'production_qualified': False,
                    'explanation': 'Proper scores and descriptive matched uncertainty are evidence, not automatic superiority, calibration certification or model promotion.'},
    })


def _verify_prestart_capture(snapshot_path, protocol_dir, forecast_path, bracket_path):
    from scripts.confirm_oddsfree_student import _read_sealed
    from scripts.tournament_research_snapshot import VERSION as SNAPSHOT_VERSION, replay_tournament

    if snapshot_path.name != 'snapshot.json' or forecast_path.name != 'simulation.json':
        raise ValueError('prestart scoring requires canonical sealed capture and replay artifacts')
    snapshot, _ = _read_sealed(snapshot_path.parent, 'snapshot', expected_version=SNAPSHOT_VERSION)
    forecast, _ = _read_sealed(forecast_path.parent, 'simulation', expected_version=None)
    if snapshot['input_sha256']['bracket.json'] != digest(bracket_path):
        raise ValueError('bracket differs from frozen capture')
    models = forecast.get('models')
    if not isinstance(models, dict) or not models:
        raise ValueError('captured replay must contain explicit model outputs')
    controls = {(model['simulations'], model['seed']) for model in models.values()}
    if len(controls) != 1:
        raise ValueError('captured replay models must share simulation controls')
    simulations, seed = controls.pop()
    # Existing replay validates protocol/models, source, inputs, saved prediction
    # hashes and recomputed inference. A newly forged seal is not sufficient.
    with TemporaryDirectory(prefix='tournament_score_verify_') as temporary:
        expected = replay_tournament(protocol_dir, snapshot_path.parent, Path(temporary)/'replay',
                                     simulations=simulations, seed=seed)
    if forecast != expected:
        raise ValueError('scored forecast differs from replay of captured predictions')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--allow-retrospective', action='store_true')
    parser.add_argument('--model', action='append', help='Explicit model subset, e.g. --model outcome; all selected models must cover every event')
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=82)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                'bootstrap': args.bootstrap, 'seed': args.seed, 'resampling_unit': 'tournament_id',
                'weighting': 'equal events; equal teams/nodes within event for marginal scores',
                'optional_placements': 'only events explicitly unavailable for every compared model may be excluded; per-metric input/eligible/excluded counts are reported',
                'primary_scores': ['champion_log_loss', 'champion_brier'],
                'selected_models': args.model,
                'log_zero': 'positive infinity, never epsilon clipping',
                'qualification': 'retrospective_not_certified' if args.allow_retrospective else 'prestart_local_capture_not_source_certified',
                'input_index_sha256': digest(args.bundle_manifest), 'source_sha256': digest(__file__)}
    save_json(args.output_dir/'protocol.json', protocol)
    entries = json.loads(args.bundle_manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('bundle manifest must be a nonempty list')
    rows, calibration, inventory = [], [], []
    for entry in entries:
        paths = {key: (args.bundle_manifest.parent/entry[key]).resolve() for key in ('forecast', 'results', 'bracket')}
        payloads = {key: json.loads(path.read_text()) for key, path in paths.items()}
        forecast = payloads['forecast']
        bracket = load_bracket(payloads['bracket'])
        if forecast.get('tournament_id') != bracket.id:
            raise ValueError('forecast tournament identity disagrees with bracket')
        if not args.allow_retrospective:
            snapshot_path = (args.bundle_manifest.parent/entry['snapshot']).resolve()
            snapshot = json.loads(snapshot_path.read_text())
            if digest(snapshot_path) != forecast.get('snapshot_sha256') or snapshot['tournament_id'] != bracket.id:
                raise ValueError('forecast does not refer to the supplied frozen snapshot')
            if not (_utc(snapshot['source_available_at']) <= _utc(snapshot['data_cutoff_at']) <= _utc(snapshot['predicted_at']) < _utc(snapshot['event_start_at'])):
                raise ValueError('snapshot must precede whole event; live updates cannot be pooled')
            if snapshot['input_sha256']['bracket.json'] != digest(paths['bracket']):
                raise ValueError('bracket differs from frozen snapshot')
            inventory.append({'kind': 'snapshot', 'path': str(snapshot_path), 'sha256': digest(snapshot_path)})
            protocol_dir = (args.bundle_manifest.parent/entry['protocol']).resolve()
            _verify_prestart_capture(snapshot_path, protocol_dir, paths['forecast'], paths['bracket'])
            inventory.append({'kind': 'protocol', 'path': str(protocol_dir/'protocol.json'),
                              'sha256': digest(protocol_dir/'protocol.json')})
        models = forecast.get('models')
        if not isinstance(models, dict) or not models:
            raise ValueError('forecast must contain explicit model outputs')
        selected_models = args.model if args.model is not None else sorted(models)
        if len(set(selected_models)) != len(selected_models) or not set(selected_models).issubset(models):
            raise ValueError('selected model names must be unique and present in every event')
        for model in selected_models:
            prediction = models[model]
            if any(word in model.lower() for word in ('market', 'benter', 'bookmaker')):
                raise ValueError('market/Benter students are not odds-free tournament candidates')
            scores, cells = score_event(bracket, prediction, payloads['results'])
            rows.append({'tournament_id': bracket.id, 'model': model, **scores})
            calibration.extend({'tournament_id': bracket.id, 'model': model, **cell} for cell in cells)
        inventory.extend({'kind': key, 'path': str(path), 'sha256': digest(path)} for key, path in paths.items())
    scores = pd.DataFrame(rows)
    summary, paired = summarize_scores(scores, args.bootstrap, args.seed)
    scores.to_csv(args.output_dir/'event_scores.csv', index=False)
    summary.to_csv(args.output_dir/'score_summary.csv', index=False)
    paired.to_csv(args.output_dir/'paired_event_differences.csv', index=False)
    pd.DataFrame(calibration).to_csv(args.output_dir/'calibration_cells.csv', index=False)
    calibration_table(calibration).to_csv(args.output_dir/'calibration.csv', index=False)
    pd.DataFrame(inventory).to_csv(args.output_dir/'provenance.csv', index=False)
    save_json(args.output_dir/'summary.json', {'protocol': protocol, 'events': scores.tournament_id.nunique(),
                                             'scores': summary.to_dict('records'), 'production_qualified': False})
    save_json(args.output_dir/'manifest.json', {'inputs': inventory, 'files': {
        path.name: digest(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}})
    print(json.dumps({'output_dir': str(args.output_dir), 'events': scores.tournament_id.nunique()}))


if __name__ == '__main__':
    main()
