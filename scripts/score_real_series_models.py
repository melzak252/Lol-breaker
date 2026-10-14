"""Score the exact shared retrospective series cohort; no predictive trust claim."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path('data/artifacts/real-series-model-comparison-20260909')


def main():
    paths = [ROOT / 'exp039/matched-refreshed/predictions.csv', ROOT / 'exp081/series_rows.csv']
    with paths[0].open() as stream:
        left = list(csv.DictReader(stream))
    with paths[1].open() as stream:
        right = [r for r in csv.DictReader(stream) if r['information_policy'] == 'refreshed_prior_day' and r['probability_a']]
    def key(row):
        return tuple(row[k] for k in ('series_id', 'phase_id', 'team_a', 'team_b'))
    index = {key(r): r for r in right}
    assert len(index) == len(right) == len(left) == 61
    assert {key(r) for r in left} == set(index)
    paired, groups = [], defaultdict(list)
    for row in left:
        other = index[key(row)]
        assert row['source_date'] == other['source_date']
        assert row['forecast_cutoff'][:10] == other['forecast_cutoff'][:10]
        assert row['best_of'] == other['best_of']
        y = float(row['y_true'])
        assert y == float(other['y_true']) and y in (0, 1)
        item = {k: row[k] for k in ('series_id', 'phase_id', 'source_date', 'tournament', 'team_a', 'team_b', 'forecast_cutoff')}
        item['y_true'] = y
        for model, p in [('exp039', float(row['probability_a'])), ('exp081', float(other['probability_a'])), ('fair', .5)]:
            assert 0 < p < 1
            item[model] = {'probability_a': p, 'log_loss': -math.log(p if y else 1-p), 'brier': (p-y)**2}
        paired.append(item)
        groups['all'].append(item)
        groups[row['tournament']].append(item)
    summary = {name: {'n': len(rows), **{m: {metric: sum(r[m][metric] for r in rows)/len(rows) for metric in ('log_loss', 'brier')} for m in ('exp039', 'exp081', 'fair')}} for name, rows in groups.items()}
    result = {'qualification': 'inspected_retrospective_same_state_prior_roster_diagnostic', 'production_qualified': False, 'weighting': 'equal actual series; descriptive phase slices; not equal-edition inference', 'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'limitations': ['61 series concentrated in five phases and June 2026; no independent block inference', '309 globally missing recovered source series; publication and roster announcements uncertified', 'EXP039 original calibrated symmetry defect preserved; chronological EXP081 reconstruction not original production weights'], 'summary': summary, 'paired_rows': paired}
    with (ROOT / 'paired-real-series-scores.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
