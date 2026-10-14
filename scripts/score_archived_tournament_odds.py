"""Offline market benchmark on archived tournament fixtures, not live betting evidence."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_prospective_features import iter_matches
from scripts.prospective_sports_features import _series_team_ids

ROOT = Path('data/artifacts/real-series-model-comparison-20260909')
PATHS = {
    'odds': Path('data/odds.csv'),
    'inventory': ROOT / 'exp081/series_rows.csv',
    'mapping': Path('data/artifacts/leaguepedia-tournament-rules-20260908/golgg_mapping.csv'),
    'results': Path('data/artifacts/leaguepedia-tournament-rules-20260908/series_results.csv'),
    'source': Path('data/artifacts/golgg-header-repair-20260908-validated/matches.json'),
}


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def no_vig(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b) and a > 1 and b > 1):
        return None
    return b / (a + b)


def metrics(rows, name):
    selected = [(r['probabilities'][name], r['y']) for r in rows if name in r['probabilities']]
    if not selected:
        return {'n': 0}
    n = len(selected)
    bins = []
    for i in range(10):
        cells = [(p, y) for p, y in selected if min(int(10*p), 9) == i]
        bins.append({'low': i/10, 'high': (i+1)/10, 'n': len(cells),
                     'predicted': sum(p for p, y in cells)/len(cells) if cells else None,
                     'observed': sum(y for p, y in cells)/len(cells) if cells else None})
    return {'n': n, 'log_loss': sum(-math.log(p if y else 1-p) for p, y in selected)/n,
            'brier': sum((p-y)**2 for p, y in selected)/n,
            'accuracy': sum(.5 if p == .5 else int((p > .5) == y) for p, y in selected)/n,
            'ece_10_equal_width': sum(b['n'] * abs(b['predicted']-b['observed']) for b in bins if b['n'])/n,
            'calibration_bins': bins}


def main():
    odds = load(PATHS['odds'])
    assert len({r['golgg_match_id'] for r in odds}) == len(odds)
    oi = {r['golgg_match_id']: r for r in odds}
    mi = {r['wiki_series_id']: r for r in load(PATHS['mapping'])}
    ri = {r['wiki_series_id']: r for r in load(PATHS['results'])}
    inventory = [r for r in load(PATHS['inventory']) if r['information_policy'] == 'refreshed_prior_day']
    books = sorted(k[len('odds1_'):-len('_close')] for k in odds[0] if k.startswith('odds1_') and k.endswith('_close'))
    accepted, exclusions, seen = [], [], set()
    references = {}
    for source in iter_matches(PATHS['source']):
        mid = str(source['match_id'])
        if mid not in oi:
            continue
        try:
            t1, t2 = _series_team_ids(source)
        except ValueError:
            continue  # Missing identity is explicitly excluded below, never inferred from scores.
        names = {str(g[f't{s}_id']): g.get(f't{s}_name') for g in source['games'] for s in (1, 2)}
        references[mid] = (names[t1], names[t2])
    for row in inventory:
        wid, mid = row['wiki_series_id'], row['golgg_match_id']
        reason = None
        o, m, r = oi.get(mid), mi.get(wid), ri[wid]
        if o is None:
            reason = 'no_odds_match_id'
        elif m is None or m['status'] != 'matched' or m['side_swapped'] not in ('True', 'False'):
            reason = 'mapping_not_unambiguous'
        elif mid in seen:
            reason = 'duplicate_underlying_series'
        elif r['status'] != 'completed' or r['winner_side'] not in ('1', '2'):
            reason = 'not_completed_binary_outcome'
        elif o['match_type'] != 'exact':
            reason = 'odds_match_not_exact'
        elif o['odds_date'] != o['golgg_date'] or o['golgg_date'] != m['golgg_date']:
            reason = 'odds_mapping_date_disagreement'
        elif references.get(mid) not in ((o['golgg_team1'], o['golgg_team2']), (o['golgg_team2'], o['golgg_team1'])):
            reason = 'source_team_names_not_exactly_aligned'
        else:
            reversed_odds = references[mid] == (o['golgg_team2'], o['golgg_team1'])
            swap = (m['side_swapped'] == 'True') != reversed_odds
            wiki_scores = [float(r['team1_score']), float(r['team2_score'])]
            expected = wiki_scores[::-1] if swap else wiki_scores
            if expected != [float(o['t1_score']), float(o['t2_score'])]:
                reason = 'independent_outcome_score_disagreement'
            else:
                y = int((r['winner_side'] == '1') != swap)
                if y != int(o['t1_win'] == 'True') or o['draw'] != 'False':
                    reason = 'independent_winner_disagreement'
        if reason:
            exclusions.append({'series_id': wid, 'golgg_match_id': mid, 'reason': reason})
            continue
        probabilities = {'fair': .5}
        for mode, a, b in [('close', 'avg_odds_home', 'avg_odds_away'), ('open', 'avg_open_home', 'avg_open_away')]:
            p = no_vig(o[a], o[b])
            if p is not None:
                probabilities['market_'+mode] = p
            for book in books:
                p = no_vig(o[f'odds1_{book}_{mode}'], o[f'odds2_{book}_{mode}'])
                if p is not None:
                    probabilities[book+'_'+mode] = p
        if len(probabilities) == 1:
            exclusions.append({'series_id': wid, 'golgg_match_id': mid, 'reason': 'no_valid_two_sided_decimal_odds'})
            continue
        seen.add(mid)
        accepted.append({'series_id': wid, 'golgg_match_id': mid, 'phase_id': row['phase_id'],
                         'tournament': r['event_title'], 'family': json.loads(r['families']),
                         'date': o['golgg_date'], 'round': r['section_path'],
                         'team_a': o['golgg_team1'], 'team_b': o['golgg_team2'],
                         'y': y, 'probabilities': probabilities})
    paired = [r for r in accepted if {'market_close', 'market_open'} <= r['probabilities'].keys()]
    groups = defaultdict(list)
    for r in paired:
        for family in r['family']:
            groups[family].append(r)
    names = ['market_close', 'market_open', 'fair']
    report = {'qualification': 'retrospective_archived_odds_diagnostic_not_timestamp_certified',
              'model_comparison': 'No shared eligible model predictions; not comparable with June61 scores',
              'normalization': 'proportional no-vig; normalize reciprocal archived mean decimal odds separately for opening and closing',
              'limitations': ['No quote timestamps, so closing/prematch availability not certified', 'Equal-series weighting; dependent matches, phases and months are not independent', 'ECE bins descriptive, no calibration certification or inferential intervals', 'Individual bookmakers have different covered cohorts; not a bookmaker ranking'],
              'input_hashes': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k,p in PATHS.items()},
              'inventory_n': len(inventory), 'id_overlap_n': sum(r['golgg_match_id'] in oi for r in inventory),
              'accepted_n': len(accepted), 'paired_open_close_n': len(paired),
              'paired_summary': {name: metrics(paired, name) for name in names},
              'by_family_paired': {k: {name: metrics(v, name) for name in names} for k,v in groups.items()},
              'by_phase_paired': {k: {name: metrics([r for r in paired if r['phase_id']==k], name) for name in names} for k in sorted({r['phase_id'] for r in paired})},
              'per_bookmaker_available': {book: {mode: metrics(accepted, book+'_'+mode) for mode in ('open','close')} for book in books},
              'exclusion_counts': dict(Counter(r['reason'] for r in exclusions)), 'exclusions': exclusions,
              'rows': accepted}
    destination = ROOT/'bookmaker-historical-benchmark-aligned.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('inventory_n','id_overlap_n','accepted_n','paired_open_close_n','paired_summary','exclusion_counts')}, indent=2))


if __name__ == '__main__':
    main()
