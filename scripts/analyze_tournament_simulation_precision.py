#!/usr/bin/env python3
"""Offline repeated-seed precision experiment; numerical uncertainty is not predictive CI.

Uses the unmodified tournament evaluator and proper-score ledger. Every invocation
requires a new output directory. Full-cohort scores retain failures; a separately
labelled available-method analysis removes whole failed methods within a cohort,
never cherry-picks targets or seeds. Explicit --existing-run inputs reuse complete,
hash-verified forecasts without mutating them. No fitting or future holdout access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_tournament_forecasts import LEDGER_COHORT, evaluate_ledger
from scripts.tournament_evaluation import _source_paths, run_evaluation, score_evaluation
from src.models.tournament_prediction import file_digest

IDENTITY = ('tournament_id', 'phase_id', 'origin_id', 'target_id')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _available_methods(rows):
    failed = defaultdict(list)
    for row in rows:
        if row.get('forecast_status') == 'failed':
            failed[tuple(row[field] for field in LEDGER_COHORT) + (row['model'],)].append(row)
    kept = [row for row in rows
            if tuple(row[field] for field in LEDGER_COHORT) + (row['model'],) not in failed]
    removed = []
    for key, failures in sorted(failed.items()):
        affected = [row for row in rows
                    if tuple(row[field] for field in LEDGER_COHORT) + (row['model'],) == key]
        removed.append(dict(zip((*LEDGER_COHORT, 'model'), key)) | {
            'removed_rows': len(affected), 'failed_rows': len(failures),
            'reasons': dict(Counter(row['failure_reason'] for row in failures))})
    return kept, removed


def _contrasts(scores, model):
    result = []
    for row in scores['comparisons']:
        wanted = model + ':frozen' if row['axis'] == 'axis3' else model
        if {row['model'], row['reference']} != {wanted, 'fair_series'}:
            continue
        if row['axis'] not in {'axis1', 'axis3'}:
            continue
        oriented = dict(row)
        if row['model'] != wanted:
            oriented.update(model=wanted, reference='fair_series',
                            delta=-row['delta'] if row['delta'] is not None else None,
                            model_mean=row['reference_mean'], reference_mean=row['model_mean'],
                            counts=row['reference_counts'], reference_counts=row['counts'])
            # Reversing a comparison also reverses its predictive interval endpoints.
            oriented.update(ci_low=-row['ci_high'] if row['ci_high'] is not None else None,
                            ci_high=-row['ci_low'] if row['ci_low'] is not None else None)
            oriented.pop('probability_non_improvement', None)
            oriented.pop('sensitivity', None)
        matching_targets = [target for target in scores['target_scores']
                            if all(target[field] == row[field] for field in LEDGER_COHORT)
                            and target['model'] in (row['model'], row['reference'])]
        oriented['matched_target_ids'] = sorted({
            tuple(target[field] for field in IDENTITY) for target in matching_targets})
        oriented['locked_primary'] = (
            row['forecast_mode'] == 'post_draw' and row['axis'] == 'axis1'
            and row['target_family'] == 'qualification' and row['target_kind'] == 'binary'
            and row['metric'] == 'log_loss' and bool(matching_targets)
            and all(target['forecast_scope'] == 'phase_start' for target in matching_targets))
        result.append(oriented)
    return result


def _mc_summary(runs, analysis):
    groups = defaultdict(list)
    for run in runs:
        for row in run[analysis]['contrasts']:
            key = tuple(row[field] for field in (*LEDGER_COHORT, 'metric', 'model', 'reference'))
            groups[key].append({'seed': run['seed'], **row})
    output = []
    for key, rows in sorted(groups.items()):
        finite = all(isinstance(row['delta'], (int, float)) and math.isfinite(row['delta']) for row in rows)
        same_cohort = all(row.get('matched_target_ids') == rows[0].get('matched_target_ids') for row in rows)
        complete = finite and len(rows) == len(runs) and len(rows) >= 3 and same_cohort
        values = [row['delta'] for row in rows] if complete else []
        sd = statistics.stdev(values) if complete else None
        half_width = float(student_t.ppf(.975, len(values)-1) * sd / math.sqrt(len(values))) if complete else None
        primary = all(row.get('locked_primary', False) for row in rows)
        output.append(dict(zip((*LEDGER_COHORT, 'metric', 'model', 'reference'), key)) | {
            'analysis': analysis, 'status': 'numerical_variation_only' if complete else 'unavailable_incomplete_or_unscorable',
            'seed_count': len(rows), 'required_seed_count': len(runs),
            'same_matched_targets_across_seeds': same_cohort,
            'matched_target_ids': rows[0].get('matched_target_ids'),
            'mean_paired_delta': statistics.mean(values) if complete else None,
            'minimum_paired_delta': min(values) if complete else None,
            'maximum_paired_delta': max(values) if complete else None,
            'paired_delta_spread': max(values) - min(values) if complete else None,
            'mc_single_run_delta_sd': sd,
            'mc_mean_delta_se': sd / math.sqrt(len(values)) if complete else None,
            'mc_mean_delta_95_half_width': half_width,
            'mc_single_run_delta_95_scale': float(student_t.ppf(.975, len(values)-1) * sd) if complete else None,
            'numerical_interval_method': 'Student-t across independent paired seed deltas; approximate seed-mean interval conditional on fixed editions/outcomes, not historical bootstrap.',
            'locked_primary_delta_ll_budget': {
                'applicable': primary, 'maximum_95_half_width': .001,
                'estimand': 'mean paired delta LL across the predeclared independent seeds',
                'observed_half_width': half_width if primary else None,
                'status': 'not_applicable_secondary_target' if not primary
                          else 'inconclusive_unscorable' if not complete
                          else 'pass' if half_width <= .001 else 'fail',
                'does_not_override_per_cell_precision': True},
            'mc_se_definition': 'Sample SD of paired seed deltas / sqrt(independent seeds); numerical SE of their mean, conditional on fixed inputs and outcomes.',
            'predictive_ci': None,
            'seed_results': rows})
    return output


def _evidence(scores):
    groups = defaultdict(dict)
    for row in scores['target_scores']:
        groups[tuple(row[field] for field in LEDGER_COHORT)][tuple(row[field] for field in IDENTITY)] = row
    return {
        'coverage': scores['coverage'], 'verdict': scores['verdict'],
        'target_coverage': [dict(zip(LEDGER_COHORT, key)) | {
            'targets': len(rows),
            'phases': len({(row['tournament_id'], row['phase_id']) for row in rows.values()}),
            'origins': len({tuple(row[field] for field in IDENTITY[:3]) for row in rows.values()}),
            'editions': len({row['tournament_id'] for row in rows.values()}),
            'unresolved_observed_targets': sum(not row['resolved'] and row['observed'] is not None for row in rows.values())}
            for key, rows in sorted(groups.items())],
        'series_targets_present': any(key[3] == 'series' for key in groups),
        'joint_targets_present': any(key[3] == 'joint' for key in groups),
        'calibration': scores['calibration'], 'series_diagnostics': scores['series_diagnostics'],
        'distribution_diagnostics': scores.get('distribution_diagnostics', []),
        'distribution_diagnostics_status': 'available' if 'distribution_diagnostics' in scores
                                           else 'unavailable_in_saved_legacy_report',
        'categorical_targets_present': any(key[3] == 'categorical' for key in groups),
        'count_targets_present': any(key[3] == 'count' for key in groups),
        'limitations': [
            'Binary qualification marginals do not validate the full joint distribution of qualifiers or tournament paths.',
            'Even pooled one-vs-rest calibration of explicit joint categories cannot establish full-vector calibration or dependence correctness.',
            'No series targets means no observed-series accuracy/AUC or series calibration evidence in this ledger.',
            'Fixed-bin ECE and calibration regression statuses remain descriptive; low ECE on a tiny cohort is not certification.']}


def _admit_existing_runs(paths, manifest_path, manifest, seeds, simulations):
    if not paths:
        return {}
    if len(paths) != len(seeds):
        raise ValueError('existing runs must contain exactly every declared seed')
    inputs = {str(path): file_digest(path) for path in _source_paths(manifest, manifest_path.parent)}
    code = {name: file_digest(ROOT / name) for name in (
        'scripts/tournament_evaluation.py', 'scripts/simulate_tournament.py',
        'src/models/tournament_formats.py', 'src/models/tournament_prediction.py',
        'scripts/prospective_sports_features.py', 'src/models/siamese_series.py',
        'scripts/score_tournament_forecasts.py')}
    manifest_hash = file_digest(manifest_path)
    admitted = {}
    for path in map(lambda value: Path(value).resolve(), paths):
        saved = _load(path / 'protocol.json')
        status = _load(path / 'status.json')
        seed = saved['seed']
        if seed not in seeds or seed in admitted:
            raise ValueError('existing run has an unexpected or duplicate seed')
        if saved['simulations'] != simulations or saved['max_simulations'] != simulations:
            raise ValueError('existing run simulation budget differs from the declared fixed budget')
        if saved['version'] != manifest['version'] or saved['manifest_sha256'] != manifest_hash:
            raise ValueError('existing run manifest differs from the declared cohort')
        if saved['qualification'] != manifest['qualification'] or saved['python'] != platform.python_version():
            raise ValueError('existing run qualification or Python runtime changed')
        if saved['input_sha256'] != inputs or saved['source_sha256'] != code:
            raise ValueError('existing run source or referenced input hashes changed')
        if _load(path / 'input_manifest.json') != manifest:
            raise ValueError('saved input manifest differs from the declared cohort')
        if status['status'] != 'completed' or status['completed_origins'] != sum(
                len(event['origins']) for event in manifest['events']):
            raise ValueError('existing seed run is not complete')
        ledger_hash = file_digest(path / 'forecasts.jsonl')
        if status['forecast_sha256'] != ledger_hash:
            raise ValueError('existing seed forecast ledger changed after completion')
        admitted[seed] = {'path': str(path), 'forecast_sha256': ledger_hash,
                          'protocol_sha256': file_digest(path / 'protocol.json'),
                          'status_sha256': file_digest(path / 'status.json')}
    return admitted


def _seed_evidence(directory, score_directory, status, outcomes_path, model, qualification, outcomes_hash):
    # Release the full report before loading forecasts, and all seed-local payloads
    # before the next simulation/scoring pass. Only the published summaries survive.
    complete = score_evaluation(directory, outcomes_path, score_directory / 'complete-cohort-scores.json')
    complete_evidence = {'contrasts': _contrasts(complete, model), **_evidence(complete)}
    del complete
    forecasts = []
    with (directory / 'forecasts.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            # The immutable source keeps provenance; no scoring/precision consumer
            # reads it. Do not retain its repeated feature/artifact payload in RAM.
            row.pop('provenance', None)
            forecasts.append(row)
    available, removed = _available_methods(forecasts)
    available_ids = {tuple(row[field] for field in IDENTITY) for row in available}
    available_outcomes = [row for row in _load(outcomes_path)
                          if tuple(row[field] for field in IDENTITY) in available_ids]
    available_scores = evaluate_ledger(available, available_outcomes)
    available_scores.update(analysis='available_methods_only_not_complete_cohort',
                            removed_methods=removed, forecast_sha256=status['forecast_sha256'],
                            outcomes_sha256=outcomes_hash, qualification=qualification, production_qualified=False)
    _new(score_directory / 'available-method-scores.json', available_scores)
    return {
        'status': status, 'actual_mc_sample_counts': sorted({
            row['mc_samples'] for row in forecasts if 'mc_samples' in row}),
        'target_numerical_precision': [{
            **{field: row[field] for field in (*IDENTITY, *LEDGER_COHORT, 'model')},
            'forecast_status': row.get('forecast_status', 'ok'),
            'resolved_at_origin': row['resolved'], 'mc_samples': row.get('mc_samples'),
            'precision': row.get('mc_precision'),
            'precision_status': ('not_applicable_non_mc' if row.get('probability_source') != 'monte_carlo'
                                 else 'unavailable_precision_payload' if 'mc_precision' not in row
                                 else 'resolved' if row['mc_precision_resolved']
                                 else 'cap_unresolved' if row['mc_precision'].get('cap_reached')
                                 else 'unresolved')}
            for row in forecasts],
        'protocol_sha256': file_digest(directory / 'protocol.json'),
        'complete_scores_sha256': file_digest(score_directory / 'complete-cohort-scores.json'),
        'available_scores_sha256': file_digest(score_directory / 'available-method-scores.json'),
        'complete_cohort': complete_evidence,
        'available_methods': {'contrasts': _contrasts(available_scores, model),
                              'removed_methods': removed, **_evidence(available_scores)}}


def _saved_evidence(path):
    scores = _load(path)
    return {'path': str(path), 'sha256': file_digest(path),
            'qualification': scores.get('qualification'), **_evidence(scores)}


def analyze(manifest_path, outcomes_path, output, *, seeds, simulations, model, existing_scores, existing_runs=()):
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('at least three distinct nonnegative seeds are required')
    if simulations < 1:
        raise ValueError('simulations must be positive')
    manifest_path, outcomes_path, output = map(lambda path: Path(path).resolve(), (manifest_path, outcomes_path, output))
    manifest = _load(manifest_path)
    if manifest.get('version') != 'tournament-evaluation-v3':
        raise ValueError('precision analysis requires an explicit tournament-evaluation-v3 manifest')
    if manifest['qualification'] != 'retrospective_reconstruction':
        raise ValueError('this inspected-development experiment requires retrospective_reconstruction, never future holdout')
    if model not in {item['id'] for item in manifest['models']}:
        raise ValueError('comparison model is absent from manifest')
    existing_scores = [Path(path).resolve() for path in existing_scores]
    reused = _admit_existing_runs(existing_runs, manifest_path, manifest, seeds, simulations)
    protocol = {
        'version': 'tournament-monte-carlo-precision-v3',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'manifest_path': str(manifest_path), 'manifest_sha256': file_digest(manifest_path),
        'outcomes_path': str(outcomes_path), 'outcomes_sha256': file_digest(outcomes_path),
        'source_sha256': {str(path.relative_to(ROOT)): file_digest(path) for path in
                          (Path(__file__).resolve(), ROOT / 'scripts/score_tournament_forecasts.py')},
        'existing_score_sha256': {str(path): file_digest(path) for path in existing_scores},
        'reused_runs': reused,
        'seeds': seeds, 'simulations_per_method_per_origin': simulations,
        'scorer_bootstrap_replicates': 5000, 'scorer_seed': 82, 'model': model,
        'locked_primary_numerical_budget': {
            'metric': 'edition_balanced_paired_qualification_log_loss',
            'forecast_mode': 'post_draw', 'forecast_scope': 'phase_start',
            'maximum_seed_mean_95_half_width': .001, 'registered_practical_effect': .01,
            'locked_before_seed_runs': True, 'per_cell_log_probability_half_width': .05,
            'per_cell_absolute_half_width': .005},
        'qualification': manifest['qualification'], 'data_role': 'inspected_development_pilot',
        'python': platform.python_version(),
        'packages': {name: version(name) for name in ('numpy', 'scipy')},
        'seed_design': 'Independent seeded simulator streams across runs; same seed across methods and origins within each run, preserving paired numerical dependence.',
        'available_method_rule': 'Drop an entire method within forecast_mode/axis/target_family/target_kind/forecast_scope if any of its rows failed; keep every target of retained methods. Never remove low-precision rows.',
        'weighting': 'Use scorer paired comparisons unchanged: equal targets within origin, equal origins within edition, equal editions. Never pool modes, scopes, axes, target families, frozen/refreshed, or qualification classes.',
        'interpretation': 'Across-seed variation is numerical Monte Carlo error conditional on the same real editions, not new predictive observations or a confidence interval for performance.',
        'production_qualified': False}
    output.mkdir(parents=True, exist_ok=False)
    _new(output / 'protocol.json', protocol)
    started = time.perf_counter()
    runs = []
    expected_inputs = None
    try:
        for seed in seeds:
            run_started = time.perf_counter()
            score_directory = output / f'seed-{seed}'
            directory = Path(reused[seed]['path']) if reused else score_directory
            if reused:
                score_directory.mkdir()
                status = _load(directory / 'status.json')
            else:
                status = run_evaluation(manifest_path, directory, simulations=simulations,
                                        max_simulations=simulations, seed=seed)
            run_protocol = _load(directory / 'protocol.json')
            signature = {key: run_protocol[key] for key in (
                'input_sha256', 'source_sha256', 'manifest_sha256', 'numerical_precision')}
            if expected_inputs is not None and signature != expected_inputs:
                raise ValueError('source or forecast inputs changed across seeds')
            expected_inputs = signature
            # The actual scorer reads outcomes only after each immutable forecast run.
            evidence = _seed_evidence(directory, score_directory, status, outcomes_path, model,
                                      manifest['qualification'], protocol['outcomes_sha256'])
            runs.append({'seed': seed, 'runtime_seconds': time.perf_counter() - run_started, **evidence})
            print(json.dumps({'seed': seed, 'runtime_seconds': runs[-1]['runtime_seconds'],
                              'forecast_sha256': status['forecast_sha256'], 'forecast_failures': status['forecast_failures']}), flush=True)
        for path, digest in protocol['existing_score_sha256'].items():
            if file_digest(path) != digest:
                raise ValueError('existing score evidence changed during experiment')
        if file_digest(outcomes_path) != protocol['outcomes_sha256']:
            raise ValueError('outcomes changed during experiment')
        for path, digest in protocol['source_sha256'].items():
            if file_digest(ROOT / path) != digest:
                raise ValueError('analysis or scorer source changed during experiment')
        for evidence in reused.values():
            directory = Path(evidence['path'])
            for name, key in (('forecasts.jsonl', 'forecast_sha256'), ('protocol.json', 'protocol_sha256'),
                              ('status.json', 'status_sha256')):
                if file_digest(directory / name) != evidence[key]:
                    raise ValueError('reused seed evidence changed during analysis')
        if reused:
            for path, digest in expected_inputs['input_sha256'].items():
                if file_digest(path) != digest:
                    raise ValueError('referenced seed inputs changed during analysis')
            for path, digest in expected_inputs['source_sha256'].items():
                if file_digest(ROOT / path) != digest:
                    raise ValueError('seed generation source changed during analysis')
        report = {'protocol': protocol, 'runtime_seconds': time.perf_counter() - started,
                  'input_and_runner_source_sha256': expected_inputs,
                  'complete_cohort_mc': _mc_summary(runs, 'complete_cohort'),
                  'available_method_mc': _mc_summary(runs, 'available_methods'),
                  'runs': runs,
                  'existing_evidence_separate_not_pooled': [_saved_evidence(path) for path in existing_scores],
                  'conclusion': 'Numerical sensitivity only. Model failures remain in the complete cohort. Available-method contrasts cannot establish refreshed-model performance, cross-edition uncertainty, calibration, full-joint validity, or future-holdout success.',
                  'future_holdout_accessed': False, 'production_qualified': False}
        _new(output / 'precision-report.json', report)
        _new(output / 'status.json', {'status': 'completed', 'seeds': seeds,
                                     'report_sha256': file_digest(output / 'precision-report.json'),
                                     'runtime_seconds': report['runtime_seconds']})
        return report
    except Exception as error:
        _new(output / 'status.json', {'status': 'failed', 'completed_seeds': [run['seed'] for run in runs],
                                     'error_type': type(error).__name__, 'error': str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Explicit tournament-evaluation-v3 manifest')
    parser.add_argument('--outcomes', type=Path, required=True, help='Independent phase-aware outcome ledger')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--simulations', type=int, default=40000)
    parser.add_argument('--model', default='exp081')
    parser.add_argument('--existing-score', type=Path, action='append', default=[])
    parser.add_argument('--existing-run', type=Path, action='append', default=[],
                        help='complete immutable run to reuse; supply exactly one per declared seed')
    args = parser.parse_args(argv)
    report = analyze(args.manifest, args.outcomes, args.output_dir, seeds=args.seeds,
                     simulations=args.simulations, model=args.model,
                     existing_scores=args.existing_score, existing_runs=args.existing_run)
    print(json.dumps({'output_dir': str(args.output_dir), 'runtime_seconds': report['runtime_seconds'],
                      'status': 'completed', 'production_qualified': False}))


if __name__ == '__main__':
    main()
