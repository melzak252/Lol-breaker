#!/usr/bin/env python3
"""Exercise actual saved CLI and central scoring consumers; never fit models."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import champion_history_runner as runner


def main():
    paths, _, _, series, _, arrays = runner.load_frozen()
    destination = paths['08_reporting'] / 'public_inference'
    destination.mkdir(exist_ok=False)
    assessment = series.loc[series.date.astype(str).str.startswith('2026-')]
    selected = [str(assessment.loc[assessment.best_of == bo, 'match_id'].iloc[0]) for bo in (1, 3, 5)]
    indices = np.asarray([int(series.index[series.match_id.astype(str) == match_id][0]) for match_id in selected])
    processes = []
    errors = []
    tolerance = 2e-6  # FP32 batch-shape effects; all reported CDFs are consumer probabilities.
    for variant in ('B', 'C', 'D'):
        command = [str(ROOT / '.venv/bin/python'), 'scripts/train_champion_history.py', 'predict',
                   '--run-id', runner.RUN_ID, '--variant', variant, '--match-id', *selected]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        (destination / f'{variant}_stdout.json').write_text(result.stdout)
        (destination / f'{variant}_stderr.txt').write_text(result.stderr)
        processes.append({'command': command, 'returncode': result.returncode,
                          'stdout_sha256': hashlib.sha256(result.stdout.encode()).hexdigest()})
        if result.returncode:
            raise RuntimeError(f'Actual {variant} CLI failed: {result.stderr}')
        payload = json.loads(result.stdout)
        ensemble = joblib.load(paths['06_models'] / f'{variant}_ensemble.joblib')
        prediction = ensemble.predict(arrays, indices)
        for i, row in enumerate(payload['predictions']):
            errors.extend(np.abs(np.asarray(row['score_probabilities']) - prediction.score_probs[i]).tolist())
            errors.extend(np.abs(np.asarray(row['length_probabilities']) - prediction.length_probs[i]).tolist())
            errors.extend(np.abs(np.asarray(row['map_play_probabilities']) - prediction.play_probs[i]).tolist())
            errors.append(abs(row['winner_probability'] - prediction.p_win[i]))
            if row['match_id'] != selected[i] or row['best_of'] != int(prediction.best_of[i]):
                raise ValueError('Public identity or BO alignment failed')
            if variant == 'B' and row['map_distributions']:
                raise ValueError('Score-only CLI fabricated statistics')
            if variant != 'B' and len(row['map_distributions']) != row['best_of']:
                raise ValueError('Conditional hypothetical map distributions missing')
            for marginal in row['map_distributions']:
                m = marginal['map_index']
                if not marginal['conditional_on_played']:
                    raise ValueError('Public conditional distribution semantics lost')
                errors.append(abs(marginal['probability_played'] - prediction.play_probs[i, m - 1]))
                law = prediction.map_prediction(m)
                for line, value in marginal['p_kills_over'].items():
                    errors.append(abs(value - law.kills_over(float(line))[i]))
                for line, value in marginal['p_duration_over_minutes'].items():
                    errors.append(abs(value - law.duration_over(float(line))[i]))
    early_id = str(series.loc[~series.date.astype(str).str.startswith('2026-'), 'match_id'].iloc[0])
    command = [str(ROOT / '.venv/bin/python'), 'scripts/train_champion_history.py', 'predict',
               '--run-id', runner.RUN_ID, '--match-id', early_id]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    (destination / 'early_origin_stderr.txt').write_text(result.stderr)
    if result.returncode == 0 or 'not earlier-origin inference' not in result.stderr:
        raise ValueError('Actual CLI did not reject earlier-origin inference')
    processes.append({'command': command, 'returncode': result.returncode, 'expected_rejection': True})
    for flags, reason in ((['--doctor'], 'another evaluation mode'), (['--bootstraps', '100'], 'resampling settings')):
        command = [str(ROOT / '.venv/bin/python'), 'scripts/run_model_benchmark.py',
                   '--champion-history', runner.RUN_ID, '--output-dir', str(destination / 'must_not_exist'), *flags]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        if result.returncode != 2 or reason not in result.stderr:
            raise ValueError('Central CLI accepted a conflicting or changed evaluation contract')
        processes.append({'command': command, 'returncode': result.returncode, 'expected_rejection': True})
    maximum = max(errors, default=0.)
    if maximum > tolerance or (destination / 'must_not_exist').exists():
        raise ValueError('Public replay differs or invalid central CLI created artifacts')
    receipt = {'run_id': runner.RUN_ID, 'status': 'PASS', 'executed_at': datetime.now(timezone.utc).isoformat(),
               'actual_cli_predictions': 9, 'BO': [1, 3, 5], 'match_ids': selected,
               'probability_comparisons': len(errors), 'maximum_absolute_error': maximum,
               'tolerance': tolerance, 'expected_rejections': 3, 'processes': processes,
               'joint_kills_duration_claim': False, 'fits': 0,
               'code_sha256': runner.sha256(Path(__file__)),
               'outputs': {str(p.relative_to(ROOT)): runner.sha256(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    runner._json(destination / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
