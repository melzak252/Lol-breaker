"""Saved-model scoring through the common benchmark CLI; no fitting or promotion."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.analysis.champion_history_evaluation import evaluate_prediction, evaluate_first_map_baseline
from src.models import champion_history_runner as runner


def benchmark_saved(run_id, output_dir):
    paths, protocol, binding, series, maps, arrays = runner.load_frozen(run_id)
    decision_path = paths['08_reporting'] / 'decision.json'
    decision = json.loads(decision_path.read_text())
    if decision['status'] != 'COMPLETED':
        raise ValueError('Only completed frozen research can be benchmarked')
    for name, expected in decision['output_hashes'].items():
        if runner.sha256(runner.ROOT / name) != expected:
            raise ValueError(f'Executed model/output changed: {name}')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    indices = runner.partitions(series)['ASSESS']
    ss, mm = runner._subset(series, maps, indices)
    metrics, scores = {}, {}
    for variant in runner.VARIANTS:
        model = joblib.load(paths['06_models'] / f'{variant}_ensemble.joblib')
        prediction = model.predict(arrays, indices)
        metric, scored, map_scores, profiles = evaluate_prediction(prediction, ss, mm)
        metrics[variant], scores[variant] = metric, scored
        scored.to_parquet(output / f'{variant}_series_scores.parquet', index=False)
        map_scores.to_parquet(output / f'{variant}_map_scores.parquet', index=False)
        profiles.to_csv(output / f'{variant}_profiles.csv', index=False)
        if variant == 'D':
            metric, first_scores, _, _ = evaluate_prediction(prediction, ss, mm.loc[mm.map_index == 1])
            metrics['D_first'], scores['D_first'] = metric, first_scores
    c0 = runner._c0_prediction(ss)
    metric, scored, _, _ = evaluate_prediction(c0, ss, mm)
    metrics['C0_coherent_reference'], scores['C0'] = metric, scored
    base = pd.read_parquet(runner.ROOT / runner.BASE).iloc[indices].reset_index(drop=True)
    metric, scored, _, _ = evaluate_first_map_baseline(runner._old_prediction(base), ss, mm)
    metrics['old_props'], scores['old_props'] = metric, scored
    comparisons = {f'{a}-{b}:{metric}': runner._paired(scores[a], scores[b], metric)
                   for a, b, metric in protocol['primary_contrasts']}
    original_metrics = json.loads((paths['08_reporting'] / 'metrics.json').read_text())
    original_comparisons = json.loads((paths['08_reporting'] / 'comparisons.json').read_text())
    maximum = 0.
    numeric_values = 0

    def compare(actual, expected):
        nonlocal maximum, numeric_values
        if isinstance(expected, dict):
            if set(actual) != set(expected):
                raise ValueError('Benchmark metric/contrast schema differs from execution')
            for key in expected:
                compare(actual[key], expected[key])
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError('Benchmark metric/contrast support differs from execution')
            for a, b in zip(actual, expected):
                compare(a, b)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            difference = abs(float(actual) - expected)
            maximum = max(maximum, difference)
            numeric_values += 1
            if not np.isclose(actual, expected, rtol=0., atol=2e-7):
                raise ValueError(f'Saved benchmark differs from execution: {actual} versus {expected}')
        elif actual != expected:
            raise ValueError('Benchmark nonnumeric field differs from execution')

    compare(runner._clean(metrics), original_metrics)
    compare(runner._clean(comparisons), original_comparisons)
    runner._json(output / 'metrics.json', metrics)
    runner._json(output / 'comparisons.json', comparisons)
    receipt = {
        'run_id': run_id, 'status': 'DEVELOPMENT_ONLY', 'executed_at': runner._now(),
        'assessment_series': len(ss), 'played_maps': len(mm), 'new_fits': 0,
        'formal_fit_attempts_in_source_run': decision.get('formal_fit_attempts', decision['formal_fits']),
        'source_decision_sha256': runner.sha256(decision_path),
        'protocol_sha256': binding['protocol_sha256'], 'recomputed_primary_contrasts': len(comparisons),
        'bootstrap': {'unit': protocol['bootstrap_unit'], 'draws': protocol['bootstrap_draws'],
                      'confidence': protocol['confidence'], 'blocks': 9},
        'replay_numeric_values': numeric_values, 'replay_max_absolute_error': maximum,
        'independence': 'Uses the existing saved model and shared evaluator; independent numerical audit is separate',
        'source_availability': 'conditional first-map rosters/day-only releases, not authenticated issuance',
        'later_map_old_prop_reference': 'UNAVAILABLE; first-map comparison is separate',
        'no_production_or_betting_adoption': True,
        'code_hashes': {str(p.relative_to(runner.ROOT)): runner.sha256(p) for p in
                        (Path(__file__), runner.ROOT / 'scripts/run_model_benchmark.py')},
        'output_hashes': {str(p): runner.sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    runner._json(output / 'receipt.json', receipt)
    return receipt
