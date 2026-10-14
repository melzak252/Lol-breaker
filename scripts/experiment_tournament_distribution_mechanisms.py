#!/usr/bin/env python3
"""Offline, append-only historical tournament mechanism ablations (no promotion).

Explicit manifest/base run, matched OOF predictions and local source are required.
All settings are frozen in protocol.json before fitting; already inspected history
remains retrospective development evidence, never untouched prospective proof.
Conditional score schema: probability_bins partitions [0,1]; distributions maps
BoN to one complete losing-map PMF per bin of the sampled winner's probability.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import tournament_evaluation as runner
from scripts.build_siamese_research_dataset import _history_helpers
from scripts.prospective_sports_features import _compact_series, _game_days, _order
from scripts.score_tournament_forecasts import evaluate_ledger, save_json
from scripts.train_and_tune_siamese_series import fit_platt_scaling
from src.models.tournament_formats import _conditional_scores, required_best_ofs, simulate_tournament
from src.models.tournament_prediction import file_digest, fit_score_distributions, json_digest, utc
from src.models.tournament_uncertainty import fit_team_strength_posterior
from src.utils import golgg_schema

VERSION = 'tournament-distribution-mechanisms-v1'
MECHANISMS = ('fixed', 'positive_calibration', 'pooled_earlier', 'conditional_earlier',
              'recent_pooled_earlier', 'posterior_mean', 'coherent_posterior',
              'independent_matched_marginals', 'zero_uncertainty')
REFERENCES = {name: 'fixed' for name in MECHANISMS if name != 'fixed'}
REFERENCES.update(conditional_earlier='pooled_earlier', recent_pooled_earlier='pooled_earlier',
                  coherent_posterior='independent_matched_marginals')
IDENTITY = ('tournament_id', 'phase_id', 'origin_id', 'target_id')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')


def _bin(p, bins):
    return min(bisect.bisect_right(bins, p) - 1, len(bins) - 2)


def fit_conditional_scores(rows, *, cutoff, best_ofs, bins):
    """Unsmoothed MLE with fixed bins; unsupported cells fail, never back off.

    p is a genuine prior series probability for score_a's team, not the observed
    winner label. The likelihood conditions on the realized winner; this selects
    p or 1-p, exactly as prediction does after independently sampling the winner.
    """
    cutoff = date.fromisoformat(cutoff)
    best_ofs = sorted(set(best_ofs))
    if any(type(bo) is not int or bo not in (1, 3, 5) for bo in best_ofs):
        raise ValueError('conditional score best_of must be 1, 3 or 5')
    # Validate the partition before any outcome is used.
    _conditional_scores({'probability_bins': bins, 'distributions': {}}, [])
    counts = {bo: [[0] * (bo // 2 + 1) for _ in range(len(bins) - 1)] for bo in best_ofs}
    seen, used, exclusions = set(), [], Counter()
    for row in rows:
        start, end = date.fromisoformat(row['date']), date.fromisoformat(row.get('end_date', row['date']))
        if end < start:
            raise ValueError('score completion precedes series start')
        if max(start, end) >= cutoff:
            exclusions['on_or_after_cutoff'] += 1
            continue
        bo = row['best_of']
        if bo not in counts:
            exclusions['unrequested_format'] += 1
            continue
        a, b = row['score_a'], row['score_b']
        if (type(bo) is not int or type(a) is not int or type(b) is not int or
                min(a, b) < 0 or max(a, b) != bo // 2 + 1 or min(a, b) > bo // 2):
            raise ValueError('invalid completed historical series score')
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
            raise ValueError('historical score IDs must be unique nonempty strings')
        seen.add(row['id'])
        p = row['p']
        if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('conditional scores require finite prior series probabilities')
        winner_p = p if row['score_a'] > row['score_b'] else 1 - p
        counts[bo][_bin(winner_p, bins)][min(row['score_a'], row['score_b'])] += 1
        used.append(row)
    unsupported = [(bo, i) for bo, cells in counts.items() if bo != 1
                   for i, cell in enumerate(cells) if not sum(cell)]
    if unsupported:
        raise ValueError(f'conditional score support missing in (best_of, bin) cells {unsupported}; no fallback or pseudocounts')
    distributions = {bo: [[1.] if bo == 1 else [n / sum(cell) for n in cell] for cell in cells]
                     for bo, cells in counts.items()}
    model = {'probability_bins': list(bins), 'distributions': distributions}
    _conditional_scores(model, best_ofs)
    return model, {'model': 'probability_conditional_losing_maps', 'cutoff_exclusive': cutoff.isoformat(),
                   'counts': counts, 'used_series_ids': sorted(seen), 'used_scores_sha256': json_digest(used),
                   'exclusions': dict(exclusions), 'smoothing': None, 'winner_marginal': 'unchanged',
                   'bin_rule': 'left inclusive, last right inclusive; sampled winner probability'}


def _load_evidence(predictions, source, column, test_end):
    path = predictions / 'predictions.csv' if predictions.is_dir() else predictions
    required = ['golgg_match_id', 'date', 'team1_id', 'team2_id', 'best_of', 'y_true', column,
                'feature_history_max_at', 'history_last_source_day']
    frame = pd.read_csv(path, usecols=required,
                        dtype={'golgg_match_id': str, 'team1_id': str, 'team2_id': str})
    if frame[required].isna().any().any() or frame.golgg_match_id.duplicated().any():
        raise ValueError('matched predictions require unique exact IDs and complete metadata')
    times = pd.to_datetime(frame.date, utc=True)
    if (not (pd.to_datetime(frame.history_last_source_day, utc=True) < times.dt.normalize()).all() or
            not (pd.to_datetime(frame.feature_history_max_at, utc=True) <= times.dt.normalize()).all()):
        raise ValueError('predictions must use strictly prior-day feature history')
    p = frame[column].to_numpy(float)
    if not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('posterior/calibration fitting requires interior finite OOF probabilities')
    if not frame.y_true.isin([0, 1]).all():
        raise ValueError('predictions require binary series outcomes')
    raw = _read(source)
    if not isinstance(raw, list):
        raise ValueError('source must be the explicit local GOLGG match JSON array')
    source_index = {str(row['match_id']): row for row in raw}
    if len(source_index) != len(raw):
        raise ValueError('source match IDs must be unique')
    scores, exclusions = [], []
    helpers, game_ids = _history_helpers(), set()
    end_cutoff = date.fromisoformat(test_end)
    # Parse only declared predictions, never fabricate scores for unmatched IDs.
    for row in frame.to_dict('records'):
        identity = row['golgg_match_id']
        match = source_index.get(identity)
        reason = None
        if match is None:
            reason = 'exact source match ID absent'
        elif {str(golgg_schema.team1_id(match)), str(golgg_schema.team2_id(match))} != {row['team1_id'], row['team2_id']}:
            reason = 'source/prediction exact team IDs mismatch'
        elif str(match['date']) != str(row['date'])[:10] or golgg_schema.best_of(match) != row['best_of']:
            reason = 'source/prediction date or best_of mismatch'
        else:
            try:
                days = _game_days(match, date.fromisoformat(match['date']))
                if (max(days) >= end_cutoff or golgg_schema.best_of(match) not in (1, 3, 5)
                        or golgg_schema.initial_wins(match)):
                    reason = 'no ordinary completed Bo1/3/5 score strictly before test_end'
                else:
                    compact = _compact_series(match, _order(identity), helpers, game_ids, days)
                    wins = sum(game.score for game in compact.games)
                    score = {'id': identity, 'date': match['date'], 'end_date': days[-1].isoformat(),
                             'best_of': golgg_schema.best_of(match), 'score_a': int(wins),
                             'score_b': int(len(compact.games) - wins)}
                    if str(golgg_schema.team1_id(match)) != row['team1_id']:
                        score['score_a'], score['score_b'] = score['score_b'], score['score_a']
                    if int(score['score_a'] > score['score_b']) != row['y_true']:
                        reason = 'source score and prediction label mismatch'
                    else:
                        scores.append({**score, 'p': float(row[column]), 'team1_id': row['team1_id'],
                                       'team2_id': row['team2_id'], 'tournament': golgg_schema.match_tournament(match)})
            except (ValueError, KeyError, TypeError) as error:
                reason = f'invalid source score: {error}'
        if reason:
            exclusions.append({'golgg_match_id': identity, 'reason': reason})
    index = {row['id']: row for row in scores}
    admitted = frame.loc[frame.golgg_match_id.isin(index)].copy()
    # Posterior chronology uses completion day, not a possibly earlier first map.
    admitted['date'] = admitted.golgg_match_id.map(lambda identity: index[identity].get('end_date', index[identity]['date']))
    return admitted, scores, exclusions


def _loss(p, y):
    p, y = np.asarray(p), np.asarray(y)
    observed = np.where(y == 1, p, 1 - p)
    with np.errstate(divide='ignore'):
        return float(np.mean(-np.log(observed)))


def _choose_prior(frame, column, args):
    selection = frame.loc[(frame.date >= args.selection_start) & (frame.date < args.calibration_start)]
    if selection.empty:
        raise ValueError('no earlier development selection rows')
    choices = []
    for prior in args.prior_grid:
        try:
            posterior = fit_team_strength_posterior(frame, column, cutoff=args.selection_start, prior_sd=prior)
            p = posterior.predict(selection.team1_id, selection.team2_id, selection[column])
            choices.append({'prior_sd': prior, 'status': 'ok', 'log_loss': _loss(p, selection.y_true),
                            'selection_rows': len(selection), 'fit': posterior.provenance})
        except (ValueError, RuntimeError) as error:
            choices.append({'prior_sd': prior, 'status': 'unsupported', 'reason': str(error)})
    valid = [row for row in choices if row['status'] == 'ok']
    if not valid:
        raise ValueError(f'all prior candidates unsupported: {choices}')
    chosen = min(valid, key=lambda row: (row['log_loss'], row['prior_sd']))['prior_sd']
    return chosen, {'candidates': choices, 'chosen_prior_sd': chosen,
                    'selection_ids': selection.golgg_match_id.tolist(),
                    'selection_start': args.selection_start, 'selection_end_exclusive': args.calibration_start}


def _fit_scores(scores, args):
    models, fits = {}, {}
    recent_start = (date.fromisoformat(args.test_start) - pd.DateOffset(years=args.score_window_years)).date().isoformat()
    for mechanism in ('pooled_earlier', 'conditional_earlier', 'recent_pooled_earlier'):
        models[mechanism], fits[mechanism] = {}, {}
        source = [row for row in scores if row['date'] >= recent_start] if mechanism == 'recent_pooled_earlier' else scores
        for bo in (1, 3, 5):
            try:
                if mechanism == 'conditional_earlier':
                    model, provenance = fit_conditional_scores(source, cutoff=args.test_start, best_ofs=[bo], bins=args.probability_bins)
                else:
                    model, provenance = fit_score_distributions(source, cutoff=date.fromisoformat(args.test_start), best_ofs=[bo])
                models[mechanism][bo] = model
                fits[mechanism][bo] = {'status': 'ok', 'provenance': provenance,
                                      'window_start': recent_start if mechanism == 'recent_pooled_earlier' else None}
            except (ValueError, KeyError, TypeError) as error:
                fits[mechanism][bo] = {'status': 'unsupported', 'reason': str(error)}
    return models, fits


def _score_kwargs(mechanism, models, formats):
    absent = sorted(set(formats) - set(models[mechanism]))
    if absent:
        raise ValueError(f'{mechanism} lacks earlier fitted BoN support {absent}; no fallback')
    if mechanism == 'conditional_earlier':
        first = models[mechanism][formats[0]]
        return {'score_distributions': None, 'conditional_score_distributions': {
            'probability_bins': first['probability_bins'], 'distributions': {
                bo: models[mechanism][bo]['distributions'][bo] for bo in formats}}}
    return {'score_distributions': {bo: models[mechanism][bo][bo] for bo in formats}}


def _score_diagnostics(scores, models, args):
    rows = []
    for row in scores:
        if not args.test_start <= row['date'] < args.test_end:
            continue
        p, bo = row['p'], row['best_of']
        k = bo // 2 + 1
        for mechanism in models:
            record = {'id': row['id'], 'mechanism': mechanism, 'best_of': bo, 'era': row['date'][:4],
                      'probability_bin': _bin(max(p, 1-p), args.probability_bins), 'tournament': row['tournament']}
            try:
                kwargs = _score_kwargs(mechanism, models, [bo])
                if mechanism == 'conditional_earlier':
                    cells = kwargs['conditional_score_distributions']['distributions'][bo]
                    qa, qb = cells[_bin(p, args.probability_bins)], cells[_bin(1-p, args.probability_bins)]
                else:
                    qa = qb = kwargs['score_distributions'][bo]
                mass = np.asarray([p * q for q in qa] + [(1-p) * q for q in qb])
                observed = min(row['score_a'], row['score_b']) + (0 if row['score_a'] > row['score_b'] else k)
                truth = np.zeros(2 * k)
                truth[observed] = 1
                record.update(status='ok', log_loss=-math.log(mass[observed]) if mass[observed] > 0 else math.inf,
                              brier=float(np.sum((mass-truth)**2)),
                              sweep_probability=float(mass[0] + mass[k]),
                              sweep_observed=int(min(row['score_a'], row['score_b']) == 0),
                              close_probability=float(mass[k-1] + mass[-1]),
                              close_observed=int(min(row['score_a'], row['score_b']) == k-1),
                              winner_marginal_error=abs(float(sum(mass[:k])) - p))
            except (ValueError, KeyError) as error:
                record.update(status='unsupported', reason=str(error))
            rows.append(record)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['mechanism'], row['best_of'], row['probability_bin'], row['era'])].append(row)
    summary = []
    for key, group in groups.items():
        supported = [row for row in group if row['status'] == 'ok']
        item = dict(zip(('mechanism', 'best_of', 'probability_bin', 'era'), key))
        item.update(denominator=len(group), supported=len(supported), unsupported=len(group)-len(supported),
                    source_tournaments=len({row['tournament'] for row in group}),
                    inference='retrospective descriptive; source tournament titles not certified independent editions')
        for metric in ('log_loss', 'brier', 'sweep_probability', 'sweep_observed', 'close_probability', 'close_observed'):
            item[metric] = float(np.mean([row[metric] for row in supported])) if supported else None
        item['winner_marginal_error_max'] = max((row['winner_marginal_error'] for row in supported), default=None)
        summary.append(item)
    return {'rows': rows, 'slices': summary, 'heldout_score_rows': len(rows) // len(models),
            'probability_bins': args.probability_bins, 'inference': 'no tuning on heldout score labels; no promotion'}


def _posterior_at_origin(frame, column, prior, args, event, origin, state, probabilities, cache=None):
    # A phase start is not proof of the whole edition boundary. Without that
    # boundary, an explicit COMPLETE source ID list is needed to exclude edition
    # outcomes from the frozen historical fit.
    earlier = frame.loc[frame.date < args.test_start].copy()
    edition_start = event.get('edition_start_at')
    ids = event.get('source_series_ids')
    if ids is not None and (not isinstance(ids, list) or not ids or
                            any(not isinstance(value, str) or not value for value in ids)):
        raise ValueError('whole-edition source_series_ids must be a complete nonempty string list')
    if edition_start is not None:
        if utc(edition_start).date().isoformat() < args.test_start:
            if not ids or event.get('source_series_ids_scope') != 'whole_edition':
                raise ValueError('target edition overlaps posterior fit; complete whole-edition source IDs required')
    elif not ids or event.get('source_series_ids_scope') != 'whole_edition':
        raise ValueError('unverified whole-edition boundary; cannot prove target-edition posterior exclusion')
    if ids:
        earlier = earlier.loc[~earlier.golgg_match_id.isin([str(value) for value in ids])]
    prefix = []
    for index, row in enumerate((state or {}).get('completed', [])):
        policy = (state or {}).get('temporal_policy', 'strict_timestamped')
        if policy in ('daily_rollforward', 'prior_day_reconstruction'):
            available = row['source_date' if policy == 'daily_rollforward' else 'completed_date']
            if date.fromisoformat(available) >= utc(origin['cutoff']).date():
                raise ValueError('posterior prefix source day must strictly precede origin day')
        else:
            available = row['available_at']
            if utc(available) >= utc(origin['cutoff']):
                raise ValueError('posterior prefix availability must strictly precede origin')
        a, b, bo = row['team_a'], row['team_b'], row['best_of']
        prefix.append({'golgg_match_id': f'prefix:{event["tournament_id"]}:{event["phase_id"]}:{index}',
                       'date': available, 'team1_id': a, 'team2_id': b,
                       'y_true': int(row['winner'] == a), column: runner._series_p(probabilities, a, b, bo)})
    evidence = pd.concat([earlier, pd.DataFrame(prefix)], ignore_index=True) if prefix else earlier
    fit_cutoff = origin['cutoff'] if prefix else args.test_start
    cache_key = json_digest([prior, ids, prefix, fit_cutoff])
    posterior = cache.get(cache_key) if cache is not None else None
    if posterior is None:
        # Avoid mixed ISO string parsing ambiguity without inventing availability:
        # calendar-only training days remain explicitly midnight calendar labels.
        evidence['date'] = evidence.date.map(lambda value: utc(value) if 'T' in value else
            datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=timezone.utc))
        posterior = fit_team_strength_posterior(evidence, column, cutoff=fit_cutoff, prior_sd=prior)
        if cache is not None:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[cache_key] = posterior
    return posterior, {'policy': 'frozen earlier development likelihood plus real available checkpoint prefix; joint Laplace refit',
                       'historical_fit_end_exclusive': args.test_start, 'prefix_count': len(prefix),
                       'prefix_sha256': json_digest(prefix), 'excluded_edition_source_ids': ids,
                       'prefix_temporal_policy': (state or {}).get('temporal_policy', 'strict_timestamped'),
                       'prefix_dates': 'raw source days retained for retrospective policies; not publication timestamps',
                       'prefix_offsets': 'frozen pre-prefix reference matrix; never post-result refreshed offsets',
                       'posterior': posterior.provenance}


def _transform(probabilities, posterior, integrate):
    keys = list(probabilities)
    values = posterior.predict([key[0] for key in keys], [key[1] for key in keys],
                               [probabilities[key] for key in keys], integrate=integrate)
    return dict(zip(keys, map(float, values)))


def _failed(template, name, reason):
    row = {key: value for key, value in template.items() if not key.startswith('mc_')}
    row.update(model=name, probabilities={}, probability_source='unspecified', forecast_status='failed',
               failure_reason=reason, provenance={'mechanism_exclusion': reason, 'production_qualified': False})
    return row


def run(args):
    paths = {name: Path(getattr(args, name)).resolve() for name in
             ('manifest', 'baseline_run', 'predictions', 'source', 'outcomes', 'output_dir')}
    output, base = paths['output_dir'], paths['manifest'].parent
    allowed = ROOT / 'data/artifacts/tournament-validation-roadmap-20260910'
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('output must be a NEW subdirectory under the ignored roadmap artifact root')
    if output.exists():
        raise ValueError('output already exists; never overwrite an experiment')
    if not args.selection_start < args.calibration_start < args.test_start < args.test_end:
        raise ValueError('require selection_start < calibration_start < test_start < test_end')
    for value in (args.selection_start, args.calibration_start, args.test_start, args.test_end):
        date.fromisoformat(value)
    if args.simulations < 1 or min(args.seed, args.latent_seed, args.bootstrap_seed) < 0 or args.bootstrap < 1:
        raise ValueError('positive simulations/bootstrap and nonnegative seeds required')
    if args.score_window_years < 1 or not args.prior_grid or any(not math.isfinite(p) or p <= 0 for p in args.prior_grid):
        raise ValueError('positive predeclared prior grid and score window required')
    _conditional_scores({'probability_bins': args.probability_bins, 'distributions': {}}, [])
    manifest = _read(paths['manifest'])
    runner._validate_manifest(manifest)
    if manifest['qualification'] != 'retrospective_reconstruction':
        raise ValueError('only explicitly eligible historical retrospective manifests are admitted; synthetic is not historical')
    baseline_protocol = _read(paths['baseline_run'] / 'protocol.json')
    baseline_status = _read(paths['baseline_run'] / 'status.json')
    baseline_ledger = paths['baseline_run'] / 'forecasts.jsonl'
    if baseline_status['status'] not in ('completed', 'no_eligible_origins'):
        raise ValueError('baseline run must be complete before mechanism fitting')
    if baseline_status.get('forecast_sha256') != file_digest(baseline_ledger):
        raise ValueError('baseline ledger changed after completion')
    if baseline_protocol['manifest_sha256'] != file_digest(paths['manifest']):
        raise ValueError('baseline and ablation must use the exact same manifest')
    input_hashes = {str(path): file_digest(path) for path in runner._source_paths(manifest, base)}
    if baseline_protocol['input_sha256'] != input_hashes:
        raise ValueError('referenced baseline inputs changed; no matched-mechanism comparison possible')
    models = [model for model in manifest['models'] if model['id'] == args.model]
    if len(models) != 1:
        raise ValueError('model must identify exactly one declared manifest model')
    model = models[0]
    prediction_path = paths['predictions'] / 'predictions.csv' if paths['predictions'].is_dir() else paths['predictions']
    sources = [paths['manifest'], prediction_path, paths['source'], paths['outcomes'], baseline_ledger,
               paths['baseline_run'] / 'protocol.json', paths['baseline_run'] / 'status.json']
    prediction_protocol = prediction_path.parent / 'protocol.json'
    if not prediction_protocol.exists():
        raise ValueError('matched predictions require sibling protocol.json for recipe/fold ancestry')
    sources.append(prediction_protocol)
    matched_protocol = _read(prediction_protocol)
    if args.probability_column not in matched_protocol.get('models', {}) or not args.probability_column.startswith('p_'):
        raise ValueError('probability column must identify an explicit matched benchmark recipe')
    recipe_files = sorted((prediction_path.parent / 'models').glob(f'*/{args.probability_column[2:]}.*'))
    if not recipe_files:
        raise ValueError('matched annual recipe artifacts are required to link OOF and hypothetical pair probabilities')
    recipe_hashes = {file_digest(path) for path in recipe_files}
    sources.extend(recipe_files)
    settings_source = Path(args.settings_evidence).resolve()
    if settings_source.is_file():
        sources.append(settings_source)
    elif args.settings_status == 'report_informed':
        raise ValueError('report-informed settings require an explicit existing --settings-evidence file')
    protocol = {'version': VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
                'arguments': vars(args), 'sources_sha256': {str(path): file_digest(path) for path in sources},
                'referenced_inputs_sha256': input_hashes,
                'code_sha256': {name: file_digest(ROOT / name) for name in (
                    'scripts/experiment_tournament_distribution_mechanisms.py', 'scripts/tournament_evaluation.py',
                    'scripts/score_tournament_forecasts.py', 'src/models/tournament_formats.py',
                    'src/models/tournament_prediction.py', 'src/models/tournament_uncertainty.py')},
                'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'mechanisms': MECHANISMS, 'paired_references': REFERENCES,
                'settings_status': args.settings_status, 'settings_evidence': args.settings_evidence,
                'fit_cutoffs': {'prior_candidate_fit_end_exclusive': args.selection_start,
                                'prior_selection_end_exclusive': args.calibration_start,
                                'calibration_start': args.calibration_start, 'calibration_end_exclusive': args.test_start,
                                'final_posterior_and_score_fit_end_exclusive': args.test_start,
                                'test_start': args.test_start, 'test_end_exclusive': args.test_end},
                'numerical_precision': {'looks': 1, 'simulations': args.simulations, 'outcome_adaptive_stopping': False},
                'posterior_policy': 'one frozen earlier historical fit plus explicitly reconditioned observed prefix; no other target outcomes',
                'alternative_score_model': 'predeclared recent-year-window pooled empirical PMF; no implied map dynamics',
                'independent_control': '32-node Gaussian-quadrature matched marginal SERIES Bernoulli outcomes, not independent arbitrary jitter',
                'production_qualified': False, 'qualification': 'retrospective_reconstruction',
                'limitations': ['Previously report-inspected history is not untouched prospective confirmation',
                                'Calendar completion establishes chronology, not publication availability',
                                'Laplace static-team residual posterior; no roster covariance or posterior coverage guarantee',
                                'Null, adverse, missing support and numerical-inconclusive cases are retained']}
    output.mkdir(parents=True)
    save_json(output / 'protocol.json', protocol)
    baseline_rows = [json.loads(line) for line in baseline_ledger.read_text().splitlines() if line]
    baseline = {tuple(row[key] for key in IDENTITY): row for row in baseline_rows
                if row['model'] == args.model or row['model'] == args.model + ':' + args.matrix_mode}
    frame, scores, exclusions = _load_evidence(paths['predictions'], paths['source'], args.probability_column, args.test_end)
    save_json(output / 'evidence_admission.json', {'prediction_rows': len(frame) + len(exclusions),
               'admitted_rows': len(frame), 'excluded_rows': len(exclusions), 'exclusions': exclusions,
               'admitted_ids_sha256': json_digest(frame.golgg_match_id.tolist())})
    # Selection and calibration never inspect heldout values.
    fit_failures, fit_report = {}, {}
    try:
        prior, fit_report['prior_selection'] = _choose_prior(frame, args.probability_column, args)
    except (ValueError, RuntimeError) as error:
        prior = None
        fit_failures['posterior'] = str(error)
    calibration = frame.loc[(frame.date >= args.calibration_start) & (frame.date < args.test_start)]
    try:
        if calibration.empty:
            raise ValueError('no earlier calibration rows')
        slope = fit_platt_scaling(logit(calibration[args.probability_column].to_numpy()), calibration.y_true.to_numpy())
        fit_report['calibration'] = {'positive_slope': slope, 'rows': len(calibration),
                                     'series_ids': calibration.golgg_match_id.tolist(), 'intercept': 0}
    except (ValueError, RuntimeError) as error:
        slope = None
        fit_failures['positive_calibration'] = str(error)
    score_models, fit_report['scores'] = _fit_scores(scores, args)
    fit_report['failures'] = fit_failures
    save_json(output / 'fits.json', fit_report)
    (output / 'posteriors').mkdir()
    forecasts, origin_reports = [], []
    posterior_cache = {}
    for event in manifest['events']:
        spec = _read(runner._path(base, event['spec']))
        formats = required_best_ofs(spec)
        initial = {}
        reference_models = {observable['reference_model'] for declared in event['origins']
                            for observable in declared.get('observables', [])
                            if observable.get('kind') == 'upset_count'}
        for origin in event['origins']:
            origin_key = (event['tournament_id'], event['phase_id'], origin['origin_id'])
            templates = [baseline[(*origin_key, target['target_id'])] for target in origin['targets']]
            report = {'tournament_id': origin_key[0], 'phase_id': origin_key[1], 'origin_id': origin_key[2],
                      'targets': len(templates), 'mechanisms': {}}
            origin_reports.append(report)
            try:
                if origin['axis'] == 'axis3' and args.matrix_mode == 'frozen':
                    probability, provenance = initial[origin['reference_origin_id']][args.model]
                else:
                    probability, provenance = runner._probabilities(
                        model, origin, event, spec, formats, base, manifest['qualification'])
                if origin['axis'] == 'axis1':
                    initial[origin['origin_id']] = {args.model: (probability, provenance)}
                    for reference in manifest['models']:
                        if reference['id'] in reference_models and reference['id'] != args.model:
                            initial[origin['origin_id']][reference['id']] = runner._probabilities(
                                reference, origin, event, spec, formats, base, manifest['qualification'])
                supplied = provenance.get('supplied_provenance', provenance)
                if supplied.get('model_artifact_sha256') not in recipe_hashes:
                    raise ValueError('hypothetical pair matrix does not use an exact artifact of the selected matched recipe')
                phase_start = event.get('phase_start_at') or event.get('edition_start_at')
                if not phase_start or utc(phase_start).date().isoformat() < args.test_start:
                    raise ValueError('phase starts before frozen mechanism fit cutoff or has no supported start boundary')
                if not args.test_start <= utc(origin['cutoff']).date().isoformat() < args.test_end:
                    raise ValueError('origin outside the predeclared heldout interval')
                state = _read(runner._path(base, origin['state'])) if origin.get('state') else None
                runner._validate_mode_state(origin, spec, state, manifest['qualification'])
                if state is not None and utc(state['cutoff']) != utc(origin['cutoff']):
                    raise ValueError('state cutoff differs from origin')
                base_kwargs = {'simulations': args.simulations, 'seed': args.seed, 'state': state,
                               'score_distributions': runner._score_model(origin, base),
                               'observables': runner._observable_specs(origin, initial)}
                fixed_result = simulate_tournament(spec, probability, **base_kwargs)
                projected = [(runner._condition_target(target, fixed_result), target) for target in origin['targets']]
                posterior = posterior_meta = posterior_error = None
                for mechanism in MECHANISMS:
                    try:
                        kwargs, changed, marginal = dict(base_kwargs), probability, probability
                        ancestry = {'baseline_probability_provenance': provenance, 'mechanism': mechanism,
                                    'protocol_sha256': file_digest(output / 'protocol.json'),
                                    'state_qualification': fixed_result['state_qualification']}
                        if mechanism == 'positive_calibration':
                            if slope is None:
                                raise ValueError(fit_failures[mechanism])
                            changed = {key: float(expit(slope * logit(p))) for key, p in probability.items()}
                            marginal = changed
                            ancestry['calibration'] = fit_report['calibration']
                        elif mechanism in score_models:
                            kwargs.update(_score_kwargs(mechanism, score_models, formats))
                            ancestry['score_fit'] = fit_report['scores'][mechanism]
                        elif mechanism == 'zero_uncertainty':
                            kwargs['team_strength_draws'] = {team: np.zeros(args.simulations) for team in spec['teams']}
                        elif mechanism in ('posterior_mean', 'coherent_posterior', 'independent_matched_marginals'):
                            if prior is None:
                                raise ValueError(fit_failures['posterior'])
                            if posterior is None and posterior_error is None:
                                try:
                                    prefix_probability = probability
                                    if (state or {}).get('completed'):
                                        reference_id = origin.get('reference_origin_id')
                                        if reference_id not in initial or args.model not in initial[reference_id]:
                                            raise ValueError('posterior conditioning needs an explicit pre-prefix reference matrix')
                                        reference_origin = next(row for row in event['origins'] if row['origin_id'] == reference_id)
                                        if utc(reference_origin['cutoff']) >= utc(origin['cutoff']):
                                            raise ValueError('posterior prefix matrix must strictly precede checkpoint')
                                        for fact in state['completed']:
                                            if 'completed_at' in fact:
                                                before_fact = utc(reference_origin['cutoff']) < utc(fact['completed_at'])
                                            else:
                                                fact_day = fact.get('source_date', fact.get('completed_date'))
                                                before_fact = utc(reference_origin['cutoff']).date() <= date.fromisoformat(fact_day)
                                            if not before_fact:
                                                raise ValueError('reference matrix postdates part of the observed prefix')
                                        prefix_probability = initial[reference_id][args.model][0]
                                    posterior, posterior_meta = _posterior_at_origin(frame, args.probability_column,
                                        prior, args, event, origin, state, prefix_probability, posterior_cache)
                                    identity = json_digest(list(origin_key))
                                    fit_id = json_digest(posterior.provenance)
                                    covariance_path = output / 'posteriors' / f'{fit_id}.npz'
                                    if not covariance_path.exists():
                                        np.savez_compressed(covariance_path, teams=np.asarray(posterior.teams),
                                                            mean=posterior.mean, covariance=posterior.covariance)
                                    posterior_meta['parameter_file'] = covariance_path.name
                                    posterior_meta['parameter_sha256'] = file_digest(covariance_path)
                                    save_json(output / 'posteriors' / f'{identity}.json', posterior_meta)
                                except (ValueError, RuntimeError, KeyError) as error:
                                    posterior_error = str(error)
                            if posterior_error:
                                raise ValueError(posterior_error)
                            ancestry['posterior'] = posterior_meta
                            marginal = _transform(probability, posterior, mechanism != 'posterior_mean')
                            if mechanism == 'coherent_posterior':
                                kwargs['team_strength_draws'] = posterior.sample(spec['teams'], args.simulations, args.latent_seed)
                            else:
                                changed = marginal
                        result = fixed_result if mechanism == 'fixed' else simulate_tournament(spec, changed, **kwargs)
                        if mechanism == 'zero_uncertainty' and result != fixed_result:
                            raise RuntimeError('zero uncertainty changed complete engine result or RNG stream')
                        vectors = [(conditioned, runner._target_vector(conditioned, result, marginal))
                                   for conditioned, _ in projected]
                        cells = max(1, sum(len(vector) for target, vector in vectors
                                           if not target['resolved'] and target['source'] != 'series_probability'))
                        method_rows = []
                        for (target, vector), template in zip(vectors, templates):
                            sampled = None if target['source'] == 'series_probability' else result
                            row = runner._row(event, origin, target, mechanism, vector, sampled, ancestry,
                                              precision_kwargs={'looks': 1, 'cells': cells})
                            if (row['resolved'] != template['resolved'] or
                                    (template.get('forecast_status', 'ok') == 'ok' and
                                     set(vector) != set(template['probabilities']))):
                                raise ValueError('mechanism changed baseline target eligibility or support')
                            row['predicted_at'] = datetime.now(timezone.utc).isoformat()
                            method_rows.append(row)
                        forecasts.extend(method_rows)
                        report['mechanisms'][mechanism] = {'status': 'ok', 'forecast_rows': len(method_rows),
                            'precision_unresolved_rows': sum(not row.get('mc_precision_resolved', True) for row in method_rows)}
                    except (ValueError, RuntimeError, KeyError, TypeError) as error:
                        reason = str(error)
                        forecasts.extend(_failed(template, mechanism, reason) for template in templates)
                        report['mechanisms'][mechanism] = {'status': 'unsupported_or_failed', 'reason': reason,
                                                          'forecast_rows': len(templates)}
            except (ValueError, RuntimeError, KeyError, TypeError) as error:
                for mechanism in MECHANISMS:
                    forecasts.extend(_failed(template, mechanism, str(error)) for template in templates)
                    report['mechanisms'][mechanism] = {'status': 'unsupported_or_failed', 'reason': str(error),
                                                      'forecast_rows': len(templates)}
    _jsonl(output / 'forecasts.jsonl', forecasts)
    save_json(output / 'origin_coverage.json', {'origins': origin_reports, 'forecast_rows': len(forecasts),
        'target_denominator': len(forecasts) // len(MECHANISMS),
        'edition_clusters': len({event['tournament_id'] for event in manifest['events']}),
        'phase_count': len(manifest['events']), 'production_qualified': False})
    # Outcomes are joined only AFTER all tournament forecasts have been written.
    outcomes = _read(paths['outcomes'])
    union = evaluate_ledger(forecasts, outcomes, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    save_json(output / 'scores_union.json', union)
    pairs = {}
    for mechanism, reference in REFERENCES.items():
        comparison = [row for row in forecasts if row['model'] in (mechanism, reference)]
        pairs[mechanism] = evaluate_ledger(comparison, outcomes, bootstrap=args.bootstrap, seed=args.bootstrap_seed)
    save_json(output / 'scores_paired.json', pairs)
    save_json(output / 'heldout_score_diagnostics.json', _score_diagnostics(scores, score_models, args))
    heldout = frame.loc[(frame.date >= args.test_start) & (frame.date < args.test_end)]
    calibration_effect = {'denominator': len(heldout), 'status': 'unsupported' if slope is None or heldout.empty else 'ok',
                          'selection': 'none on heldout; positive means worse LogLoss', 'production_qualified': False}
    if slope is not None and not heldout.empty:
        raw = heldout[args.probability_column].to_numpy()
        calibrated = expit(slope * logit(raw))
        calibration_effect.update(fixed_log_loss=_loss(raw, heldout.y_true), calibrated_log_loss=_loss(calibrated, heldout.y_true),
            delta_log_loss=_loss(calibrated, heldout.y_true)-_loss(raw, heldout.y_true),
            fixed_brier=float(np.mean((raw-heldout.y_true.to_numpy())**2)),
            calibrated_brier=float(np.mean((calibrated-heldout.y_true.to_numpy())**2)),
            heldout_ids_sha256=json_digest(heldout.golgg_match_id.tolist()))
    save_json(output / 'heldout_calibration_effect.json', calibration_effect)
    if {str(path): file_digest(path) for path in sources} != protocol['sources_sha256']:
        raise ValueError('source inputs changed during experiment; results not complete')
    if {str(path): file_digest(path) for path in runner._source_paths(manifest, base)} != input_hashes:
        raise ValueError('referenced tournament inputs changed during experiment')
    if {name: file_digest(ROOT / name) for name in protocol['code_sha256']} != protocol['code_sha256']:
        raise ValueError('implementation changed during experiment; results not complete')
    status = {'status': 'completed', 'production_qualified': False, 'forecast_rows': len(forecasts),
              'failed_rows': sum(row.get('forecast_status') == 'failed' for row in forecasts),
              'forecast_sha256': file_digest(output / 'forecasts.jsonl')}
    save_json(output / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('manifest', 'baseline-run', 'predictions', 'source', 'outcomes', 'output-dir', 'model', 'probability-column',
                 'selection-start', 'calibration-start', 'test-start', 'test-end', 'settings-evidence'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--settings-status', choices=('predeclared_new', 'report_informed'), required=True)
    parser.add_argument('--matrix-mode', choices=('frozen', 'refreshed'), default='frozen')
    parser.add_argument('--probability-bins', nargs='+', type=float, default=[0, .25, .5, .75, 1])
    parser.add_argument('--prior-grid', nargs='+', type=float, default=[.1, .3, .7, 1.5])
    parser.add_argument('--score-window-years', type=int, default=2)
    parser.add_argument('--simulations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=81)
    parser.add_argument('--latent-seed', type=int, default=20260910)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--bootstrap-seed', type=int, default=82)
    args = parser.parse_args()
    output_existed = Path(args.output_dir).exists()
    try:
        print(json.dumps(run(args), allow_nan=False))
    except Exception as error:
        output = Path(args.output_dir)
        if not output_existed and output.is_dir() and (output / 'protocol.json').exists() and not (output / 'status.json').exists():
            save_json(output / 'status.json', {'status': 'failed', 'error': str(error), 'production_qualified': False})
        raise


if __name__ == '__main__':
    main()
