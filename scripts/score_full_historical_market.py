"""Offline identity-validated historical odds alignment and strictly paired scoring.

Run without --predictions to materialize market-alignment. Then pass the chronological
predictions CSV to create market-comparison, preserving the alignment and all inputs.
Archived opening/closing labels have no quote timestamps: this is descriptive research.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_prospective_features import iter_matches
from scripts.prospective_sports_features import _series_team_ids

ROOT = Path('data/artifacts/full-historical-predictions-20260909')
INVENTORY = Path('data/artifacts/leaguepedia-tournament-rules-20260908')
PATHS = {
    'results': INVENTORY / 'series_results.csv',
    'mapping': INVENTORY / 'golgg_mapping.csv',
    'events': INVENTORY / 'events.csv',
    'source': Path('data/artifacts/golgg-header-repair-20260908-validated/matches.json'),
    'odds': Path('data/artifacts/exp081-full-audit-20260908/identity-only-v3/odds.csv'),
}
LIMITATIONS = [
    'Archived opening/closing odds have no quote timestamps; prematch availability is not certified.',
    'No live DB, scraping, production mutation, model tuning, promotion, or profitability claims.',
    'Inventory and odds coverage are selected subsets of all eligible model history.',
    'Equal-series weighting; phases and events overlap only in separate tables, never pooled twice.',
    'Calibration bins and ECE are descriptive, not calibration certification.',
    'Monthly block bootstrap does not remove all tournament, team, or inter-month dependence.',
]


def load_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def dump_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def dump_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def iso_day(value):
    try:
        return date.fromisoformat(value).isoformat()
    except (ValueError, TypeError):
        return None


def unique_index(rows, key):
    """Collapse identical records; reject all records in conflicting ID groups."""
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    index, duplicates = {}, []
    for identity, group in groups.items():
        same = all(row == group[0] for row in group)
        if same:
            index[identity] = group[0]
        if len(group) > 1:
            duplicates.append({'key': key, 'identity': identity, 'n': len(group),
                               'action': 'collapse_identical' if same else 'exclude_conflicting'})
    return index, duplicates


def no_vig(a, b):
    try:
        a, b = float(a), float(b)
    except (ValueError, TypeError):
        return None
    return b / (a + b) if all(math.isfinite(v) and v > 1 for v in (a, b)) else None


def phase_catalog(events):
    """Merge cross-family copies of the same actual event phase, preserving aliases."""
    groups = defaultdict(list)
    for event in events:
        groups[(event['title'], event['stage'], event['start_date'], event['end_date'])].append(event)
    phases = {}
    for group in groups.values():
        first = group[0]
        phase_id = min(e['phase_id'] for e in group)
        phases[phase_id] = {
            'phase_id': phase_id, 'catalogue_phase_ids': sorted(e['phase_id'] for e in group),
            'event_title': first['title'], 'year': int(first['year']), 'stage': first['stage'],
            'phase_name': first['phase_name'] or first['stage'],
            'start_date': first['start_date'], 'end_date': first['end_date'],
            'families': sorted(set(e['family'] for e in group)),
            'definition_source': 'events.csv', 'format_summary': first['format_summary'],
        }
    return phases


def resolve_phase(row, day, phases):
    candidates = [p for p in phases.values() if p['event_title'] == row['event_title'] and p['stage'] == row['stage']]
    if len(candidates) > 1:
        candidates = [p for p in candidates if day and p['start_date'] <= day <= p['end_date']]
    if len(candidates) == 1:
        return candidates[0]['phase_id'], 'title_stage_and_date_window' if len([
            p for p in phases.values() if p['event_title'] == row['event_title'] and p['stage'] == row['stage']]) > 1 else 'title_and_stage'
    if candidates:
        return None, 'ambiguous_phase_definition'
    # Do not replace a known but out-of-window same-stage definition with a guessed one.
    if any(p['event_title'] == row['event_title'] and p['stage'] == row['stage'] for p in phases.values()):
        return None, 'phase_date_window_unresolved'
    if not day:
        return None, 'phase_date_unresolved'
    phase_id = 'observed-' + hashlib.sha256((row['event_title'] + '\0' + row['stage']).encode()).hexdigest()[:20]
    phases[phase_id] = {
        'phase_id': phase_id, 'catalogue_phase_ids': [], 'event_title': row['event_title'],
        'year': int(day[:4]), 'stage': row['stage'], 'phase_name': row['stage'],
        'start_date': '', 'end_date': '', 'families': json.loads(row['families']),
        'definition_source': 'series_results.csv observed stage (catalogue event is broader)',
        'format_summary': '',
    }
    return phase_id, 'observed_title_and_stage'


def source_references(wanted):
    references, issues = {}, []
    for raw in iter_matches(PATHS['source']):
        mid = str(raw['match_id'])
        if mid not in wanted:
            continue
        try:
            t1, t2 = _series_team_ids(raw)
            names = defaultdict(set)
            wins = Counter()
            for game in raw['games']:
                for side in (1, 2):
                    tid = str(game[f't{side}_id'])
                    names[tid].add(game.get(f't{side}_name'))
                if game.get('draw') or game.get('t1_win') == game.get('t2_win'):
                    raise ValueError('nonbinary game outcome')
                wins[str(game['t1_id'] if game['t1_win'] else game['t2_id'])] += 1
            if any(len(names[tid]) != 1 or not next(iter(names[tid])) for tid in (t1, t2)):
                raise ValueError('nonunique exact full team names')
            reference = {'date': raw['date'], 'team1_id': t1, 'team2_id': t2,
                         'team1_name': next(iter(names[t1])), 'team2_name': next(iter(names[t2])),
                         'team1_score': wins[t1], 'team2_score': wins[t2],
                         'best_of': int(raw.get('best_of', raw.get('BoN')))}
            if mid in references:
                raise ValueError('duplicate source match ID')
            references[mid] = reference
        except (ValueError, KeyError, TypeError) as exc:
            issues.append({'golgg_match_id': mid, 'reason': str(exc)})
    for issue in issues:
        references.pop(issue['golgg_match_id'], None)
    return references, issues


def align(destination):
    results, result_duplicates = unique_index(load_csv(PATHS['results']), 'wiki_series_id')
    mapping, mapping_duplicates = unique_index(load_csv(PATHS['mapping']), 'wiki_series_id')
    odds, odds_duplicates = unique_index(load_csv(PATHS['odds']), 'golgg_match_id')
    phases = phase_catalog(load_csv(PATHS['events']))
    wanted = {m['golgg_match_id'] for m in mapping.values() if m['status'] == 'matched'} | set(odds)
    references, source_issues = source_references(wanted)
    inventory, accepted, excluded, seen = [], [], [], set()
    books = sorted(k[len('odds1_'):-len('_close')] for k in next(iter(odds.values())) if k.startswith('odds1_') and k.endswith('_close'))
    for wid, row in results.items():
        m = mapping.get(wid, {})
        mid = m.get('golgg_match_id', '')
        ref = references.get(mid)
        # Source date is authoritative only for an exact matched mapping with agreeing source.
        mapped_day = m.get('golgg_date') if m.get('status') == 'matched' and ref and m.get('golgg_date') == ref['date'] else None
        original_day = iso_day(row['source_date'])
        day = mapped_day or iso_day(row['scheduled_at_utc'][:10]) or original_day
        provenance = 'verified_golgg_mapping_and_source' if mapped_day else ('scheduled_at_utc' if iso_day(row['scheduled_at_utc'][:10]) else 'source_date')
        phase_id, phase_basis = resolve_phase(row, day, phases)
        year = int(day[:4]) if day else next((p['year'] for p in phases.values() if p['event_title'] == row['event_title']), None)
        if year is not None and year < 2020:
            continue
        item = {'wiki_series_id': wid, 'golgg_match_id': mid, 'event_title': row['event_title'],
                'phase_id': phase_id, 'stage': row['stage'], 'section_path': row['section_path'],
                'date': day, 'year': year, 'date_provenance': provenance,
                'source_date': row['source_date'], 'source_date_invalid': original_day is None,
                'phase_resolution': phase_basis, 'source_status': row['status'],
                'mapping_status': m.get('status', 'missing'), 'has_odds_id': mid in odds}
        reason = None
        o = odds.get(mid)
        if not phase_id:
            reason = phase_basis
        elif not day:
            reason = 'date_unresolved'
        elif row['status'] != 'completed' or row['winner_side'] not in ('1', '2'):
            reason = 'not_completed_binary_outcome'
        elif m.get('status') != 'matched' or m.get('side_swapped') not in ('True', 'False'):
            reason = 'mapping_not_unambiguous'
        elif ref is None:
            reason = 'source_identity_or_outcome_unavailable'
        elif mapped_day is None:
            reason = 'source_mapping_date_disagreement'
        elif mid in seen:
            reason = 'duplicate_underlying_series'
        elif o is None:
            reason = 'no_odds_match_id'
        elif o['match_type'] != 'identity_exact':
            reason = 'odds_identity_date_not_exact'
        elif o['odds_date'] != o['golgg_date'] or o['golgg_date'] != mapped_day:
            reason = 'odds_mapping_date_disagreement'
        else:
            source_names = (ref['team1_name'], ref['team2_name'])
            odds_names = (o['golgg_team1'], o['golgg_team2'])
            if source_names not in (odds_names, odds_names[::-1]):
                reason = 'source_team_names_not_exactly_aligned'
            else:
                odds_swapped = source_names == odds_names[::-1]
                wiki_swapped = m['side_swapped'] == 'True'
                try:
                    wiki_scores = [float(row['team1_score']), float(row['team2_score'])]
                    wiki_scores = wiki_scores[::-1] if wiki_swapped else wiki_scores
                    odds_scores = [float(o['t1_score']), float(o['t2_score'])]
                    odds_scores = odds_scores[::-1] if odds_swapped else odds_scores
                    source_scores = [ref['team1_score'], ref['team2_score']]
                    y = int((row['winner_side'] == '1') != wiki_swapped)
                    if wiki_scores != source_scores or odds_scores != source_scores or source_scores[0] == source_scores[1]:
                        reason = 'independent_outcome_score_disagreement'
                    elif y != int(source_scores[0] > source_scores[1]) or y != int((o['t1_win'] == 'True') != odds_swapped) or o['draw'] != 'False' or (o['t1_win'], o['t2_win']) not in (('True', 'False'), ('False', 'True')):
                        reason = 'independent_winner_disagreement'
                    elif int(row['best_of']) != ref['best_of']:
                        reason = 'independent_best_of_disagreement'
                except (ValueError, TypeError):
                    reason = 'invalid_outcome_or_format'
                if not reason:
                    item.update(ref)
                    item.update(y_true=y, odds_side_swapped=odds_swapped, wiki_side_swapped=wiki_swapped)
                    for mode, a, b in [('open', 'avg_open_home', 'avg_open_away'), ('close', 'avg_odds_home', 'avg_odds_away')]:
                        p = no_vig(o[a], o[b])
                        item['p_market_' + mode] = None if p is None else (1 - p if odds_swapped else p)
                        for book in books:
                            p = no_vig(o[f'odds1_{book}_{mode}'], o[f'odds2_{book}_{mode}'])
                            item[f'p_{book}_{mode}'] = None if p is None else (1 - p if odds_swapped else p)
                    if not any(item[f'p_market_{mode}'] is not None for mode in ('open', 'close')):
                        reason = 'no_valid_mean_two_sided_decimal_odds'
        item['alignment_status'] = reason or 'accepted'
        inventory.append(item)
        if reason:
            excluded.append(item)
        else:
            accepted.append(item)
            seen.add(mid)
    coverage = []
    for phase in sorted(phases.values(), key=lambda p: (p['year'], p['event_title'], p['start_date'], p['stage'])):
        group = [r for r in inventory if r['phase_id'] == phase['phase_id']]
        ok = [r for r in group if r['alignment_status'] == 'accepted']
        coverage.append({**phase, 'inventory_n': len(group), 'completed_n': sum(r['source_status'] == 'completed' for r in group),
                         'mapped_n': sum(r['mapping_status'] == 'matched' for r in group),
                         'odds_id_n': sum(r['has_odds_id'] for r in group), 'aligned_n': len(ok),
                         'opening_n': sum(r['p_market_open'] is not None for r in ok),
                         'closing_n': sum(r['p_market_close'] is not None for r in ok),
                         'paired_open_close_n': sum(r['p_market_open'] is not None and r['p_market_close'] is not None for r in ok),
                         'exclusion_counts': dict(Counter(r['alignment_status'] for r in group if r['alignment_status'] != 'accepted'))})
    report = {'qualification': 'retrospective_archived_odds_not_timestamp_certified',
              'input_hashes': {k: digest(v) for k, v in PATHS.items()}, 'input_paths': {k: str(v) for k, v in PATHS.items()},
              'inventory_n': len(inventory), 'aligned_n': len(accepted),
              'paired_open_close_n': sum(r['p_market_open'] is not None and r['p_market_close'] is not None for r in accepted),
              'source_date_invalid_n': sum(r['source_date_invalid'] for r in inventory),
              'date_provenance_counts': dict(Counter(r['date_provenance'] for r in inventory)),
              'exclusion_counts': dict(Counter(r['alignment_status'] for r in excluded)),
              'duplicates': result_duplicates + mapping_duplicates + odds_duplicates,
              'source_issues': source_issues, 'limitations': LIMITATIONS,
              'phase_policy': 'Unique event+stage; repeated same-stage phases use catalogue date windows. Cross-family identical event phases merge alias IDs. Broader catalogue main-event definitions remain zero-row metadata; observed stages receive deterministic IDs. Full-event aggregates are separate.',
              'normalization': 'p(team1)=decimal_odds_team2/(decimal_odds_team1+decimal_odds_team2), separately on archived opening and closing mean odds, then exact-identity side alignment.'}
    destination.mkdir(parents=True, exist_ok=False)
    dump_json(destination / 'alignment.json', report)
    dump_json(destination / 'aligned_rows.json', accepted)
    dump_json(destination / 'inventory.json', inventory)
    dump_json(destination / 'phase_coverage.json', coverage)
    dump_csv(destination / 'aligned_rows.csv', accepted)
    dump_csv(destination / 'inventory.csv', inventory)
    dump_csv(destination / 'exclusions.csv', excluded)
    for label, subset in [('2020_2024', [r for r in coverage if 2020 <= r['year'] <= 2024]), ('2025_plus', [r for r in coverage if r['year'] >= 2025])]:
        dump_csv(destination / f'phase_coverage_{label}.csv', subset)
    market_rows = [r for r in accepted if r['p_market_open'] is not None and r['p_market_close'] is not None]
    write_tables(destination, coverage, market_rows, ['p_market_open', 'p_market_close'], 'market_only')
    print(json.dumps({k: report[k] for k in ('inventory_n', 'aligned_n', 'paired_open_close_n', 'source_date_invalid_n', 'exclusion_counts')}, indent=2))


def metrics(rows, probability):
    if not rows:
        return {'n': 0, 'log_loss': None, 'brier': None, 'accuracy': None, 'ece_10_equal_width': None,
                'mean_prediction': None, 'observed_win_rate': None, 'calibration_bins': []}
    p = np.array([r[probability] for r in rows], dtype=float)
    y = np.array([r['y_true'] for r in rows], dtype=float)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    bins = []
    for i in range(10):
        selected = np.minimum((p * 10).astype(int), 9) == i
        bins.append({'low': i / 10, 'high': (i + 1) / 10, 'n': int(selected.sum()),
                     'predicted': float(p[selected].mean()) if selected.any() else None,
                     'observed': float(y[selected].mean()) if selected.any() else None})
    return {'n': len(rows), 'log_loss': float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            'brier': float(((p - y) ** 2).mean()),
            'accuracy': float(np.where(p == .5, .5, (p > .5) == y).mean()),
            'ece_10_equal_width': sum(b['n'] * abs(b['predicted'] - b['observed']) for b in bins if b['n']) / len(rows),
            'mean_prediction': float(p.mean()), 'observed_win_rate': float(y.mean()), 'calibration_bins': bins}


def write_tables(destination, coverage, rows, names, prefix):
    table = []
    groups = [('phase', p['phase_id'], p['event_title'], p['year'], p['stage'], [r for r in rows if r['phase_id'] == p['phase_id']]) for p in coverage]
    for title in sorted({p['event_title'] for p in coverage}):
        phases = [p for p in coverage if p['event_title'] == title]
        groups.append(('event', title, title, phases[0]['year'], 'full_event', [r for r in rows if r['event_title'] == title]))
    for year in range(2020, 2027):
        groups.append(('year', str(year), '', year, '', [r for r in rows if r['year'] == year]))
    groups.extend(('period', label, '', None, '', [r for r in rows if low <= r['year'] <= high]) for label, low, high in [('2020_2024', 2020, 2024), ('2025_plus', 2025, 2026), ('all_2020_plus', 2020, 2026)])
    for kind, identity, title, year, stage, group in groups:
        for name in names:
            table.append({'aggregation': kind, 'group_id': identity, 'event_title': title, 'year': year, 'stage': stage,
                          'forecaster': name, **metrics(group, name)})
    dump_json(destination / f'{prefix}_metrics.json', table)
    dump_csv(destination / f'{prefix}_metrics.csv', table)
    for label, low, high in [('2020_2024', 2020, 2024), ('2025_plus', 2025, 2026)]:
        dump_csv(destination / f'{prefix}_phases_{label}.csv', [r for r in table if r['aggregation'] == 'phase' and low <= r['year'] <= high])
    return table


def bootstrap(rows, a, b, seed):
    months = sorted({r['date'][:7] for r in rows})
    result = {'a': a, 'b': b, 'delta_direction': 'a_minus_b; negative favors a for LL and Brier',
              'n': len(rows), 'months': len(months), 'replicates': 5000, 'seed': seed, 'minimum_months': 6}
    if len(months) < 6:
        return {**result, 'status': 'insufficient_months', 'intervals': None}
    y = np.array([r['y_true'] for r in rows])
    pa, pb = (np.clip(np.array([r[name] for r in rows]), 1e-15, 1 - 1e-15) for name in (a, b))
    delta_ll = -(y * np.log(pa) + (1 - y) * np.log1p(-pa)) + (y * np.log(pb) + (1 - y) * np.log1p(-pb))
    delta_brier = (pa - y) ** 2 - (pb - y) ** 2
    cells = [np.array([r['date'][:7] == month for r in rows]) for month in months]
    n = np.array([cell.sum() for cell in cells])
    sums = np.array([[delta_ll[cell].sum(), delta_brier[cell].sum()] for cell in cells])
    draws = np.random.default_rng(seed).integers(0, len(months), size=(5000, len(months)))
    samples = sums[draws].sum(axis=1) / n[draws].sum(axis=1)[:, None]
    return {**result, 'status': 'descriptive_percentile_month_block_interval', 'intervals': {
        name: {'estimate': float(values.mean()), 'low_95': float(np.quantile(samples[:, i], .025)),
               'high_95': float(np.quantile(samples[:, i], .975)),
               'bootstrap_fraction_delta_ge_zero': float((samples[:, i] >= 0).mean())}
        for i, (name, values) in enumerate([('log_loss', delta_ll), ('brier', delta_brier)])}}


def score(predictions_path, alignment_dir, destination):
    alignment = json.loads((alignment_dir / 'alignment.json').read_text())
    coverage = json.loads((alignment_dir / 'phase_coverage.json').read_text())
    accepted = json.loads((alignment_dir / 'aligned_rows.json').read_text())
    predictions, duplicates = unique_index(load_csv(predictions_path), 'golgg_match_id')
    paired, exclusions = [], []
    for row in accepted:
        pred = predictions.get(row['golgg_match_id'])
        reason = None
        if pred is None:
            reason = 'no_unique_model_prediction'
        elif row['p_market_open'] is None or row['p_market_close'] is None:
            reason = 'missing_valid_opening_or_closing_mean'
        else:
            try:
                ids = (pred['team1_id'], pred['team2_id'])
                reference_ids = (row['team1_id'], row['team2_id'])
                if ids not in (reference_ids, reference_ids[::-1]):
                    reason = 'prediction_team_ids_disagree'
                else:
                    reverse = ids == reference_ids[::-1]
                    names = (pred['team1_name'], pred['team2_name'])
                    reference_names = (row['team1_name'], row['team2_name'])
                    history_bound = datetime.fromisoformat(pred['feature_history_max_at'])
                    target_midnight = datetime.fromisoformat(row['date']).replace(tzinfo=timezone.utc)
                    history_source_day = iso_day(pred['history_last_source_day'])
                    if names != (reference_names[::-1] if reverse else reference_names):
                        reason = 'prediction_exact_team_names_disagree'
                    elif pred['date'][:10] != row['date'] or int(pred['fold_year']) != row['year']:
                        reason = 'prediction_date_or_fold_disagrees'
                    elif history_source_day is None or history_source_day >= row['date'] or history_bound.tzinfo is None or history_bound > target_midnight:
                        reason = 'prediction_history_not_strictly_prior_day'
                    elif int(pred['best_of']) != row['best_of']:
                        reason = 'prediction_best_of_disagrees'
                    elif int(pred['y_true']) not in (0, 1) or (1 - int(pred['y_true']) if reverse else int(pred['y_true'])) != row['y_true']:
                        reason = 'prediction_outcome_disagrees'
                    else:
                        values = {k: float(pred[k]) for k in ('p_exp039', 'p_exp081')}
                        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values.values()):
                            reason = 'invalid_model_probability'
                        else:
                            paired.append({**row, **{k: 1 - p if reverse else p for k, p in values.items()},
                                           'prediction_side_swapped': reverse, 'fold_year': int(pred['fold_year']),
                                           'feature_history_max_at': pred['feature_history_max_at'],
                                           'history_last_source_day': history_source_day, 'p_fair': .5})
            except (ValueError, TypeError, KeyError):
                reason = 'invalid_prediction_contract'
        if reason:
            exclusions.append({'golgg_match_id': row['golgg_match_id'], 'phase_id': row['phase_id'], 'event_title': row['event_title'], 'year': row['year'], 'reason': reason})
    destination.mkdir(parents=True, exist_ok=False)
    names = ['p_exp039', 'p_exp081', 'p_market_open', 'p_market_close', 'p_fair']
    write_tables(destination, coverage, paired, names, 'paired')
    dump_csv(destination / 'paired_rows.csv', paired)
    dump_json(destination / 'paired_rows.json', paired)
    dump_csv(destination / 'prediction_exclusions.csv', exclusions)
    intervals = []
    for label, low, high in [('2020_2024', 2020, 2024), ('2025_plus', 2025, 2026), ('all_2020_plus', 2020, 2026)]:
        group = [r for r in paired if low <= r['year'] <= high]
        for a, b in [('p_exp039', 'p_market_open'), ('p_exp039', 'p_market_close'), ('p_exp081', 'p_market_open'), ('p_exp081', 'p_market_close'), ('p_exp081', 'p_exp039')]:
            intervals.append({'period': label, **bootstrap(group, a, b, 20260909)})
    dump_json(destination / 'monthly_block_bootstrap.json', intervals)
    report = {'prediction_path': str(predictions_path), 'prediction_sha256': digest(predictions_path),
              'alignment_sha256': digest(alignment_dir / 'alignment.json'), 'alignment_input_hashes': alignment['input_hashes'],
              'prediction_n': len(predictions), 'aligned_market_n': len(accepted), 'identical_paired_n': len(paired),
              'prediction_without_aligned_market_n': len(set(predictions) - {r['golgg_match_id'] for r in accepted}),
              'paired_n_by_year': dict(Counter(r['year'] for r in paired)),
              'exclusion_counts': dict(Counter(r['reason'] for r in exclusions)), 'duplicate_handling': duplicates,
              'cohort': 'All five forecasters evaluated on exactly the same rows with both models and valid opening AND closing archived mean odds.',
              'metrics': 'LL clipped at 1e-15; Brier raw probabilities; accuracy .5 credit for exact .5 ties; ten equal-width ECE bins, [0,.1),...,[.9,1].',
              'limitations': LIMITATIONS}
    dump_json(destination / 'comparison.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--alignment-dir', type=Path)
    parser.add_argument('--comparison-dir', type=Path)
    args = parser.parse_args()
    alignment_dir = args.alignment_dir or args.output_root / 'market-alignment'
    if args.predictions:
        score(args.predictions, alignment_dir, args.comparison_dir or args.output_root / 'market-comparison')
    else:
        align(alignment_dir)


if __name__ == '__main__':
    main()
