"""Unclipped paired scoring of frozen role-aware meta research forecasts.

Warmup forecasts are the unchanged C0 stream, not retrospectively fitted meta
forecasts. Only the 2026 assessment can contribute to a development decision;
no outcome from this module authorizes model promotion.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.model_benchmark import compute_benchmark_summary
from src.analysis.production_market_benchmark import paired_cluster_interval
from src.analysis.research_benchmark import IDENTITY, losses, safe, sha, validate_frame
from src.models.a0_target_experiment import resolve_research_root
from src.models.competition_tiers import classify_competition

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBABILITIES = ('p_c0', 'p_player', 'p_meta')
CONTRASTS = {
    'meta_minus_c0': ('p_meta', 'p_c0'),
    'meta_minus_player': ('p_meta', 'p_player'),
    'player_minus_c0': ('p_player', 'p_c0'),
}


def _summary(frame, column):
    if frame.empty:
        return {'sample_size': 0}
    y, p = frame.y.to_numpy(int), frame[column].to_numpy(float)
    result = asdict(compute_benchmark_summary(y, p, probability_epsilon=None))
    result.update(tail_ge_2_5=int((losses(y, p) >= 2.5).sum()),
                  min_probability=float(p.min()), max_probability=float(p.max()))
    return result


def _metrics(frame):
    return {column: _summary(frame, column) for column in PROBABILITIES}


def _contrast(frame, candidate, reference, blocks, draws, seed, confidence):
    y = frame.y.to_numpy(int)
    a, b = frame[candidate].to_numpy(float), frame[reference].to_numpy(float)
    return {
        'candidate': candidate, 'reference': reference,
        'log_loss': paired_cluster_interval(losses(y, a) - losses(y, b), blocks,
                                           draws=draws, seed=seed, confidence_level=confidence),
        'brier': paired_cluster_interval((a-y)**2 - (b-y)**2, blocks,
                                        draws=draws, seed=seed, confidence_level=.95),
    }


def _monthly_contrasts(frame, draws, seed, confidence=.95):
    months = pd.to_datetime(frame.date).dt.to_period('M').astype(str)
    result = {}
    for name, (candidate, reference) in CONTRASTS.items():
        contrast = _contrast(frame, candidate, reference, months, draws, seed, confidence)
        result[name] = {
            'candidate': candidate, 'reference': reference,
            'log_loss_monthly': contrast['log_loss'], 'brier_monthly': contrast['brier'],
        }
    return result


def _labels(values):
    return values.map(lambda value: 'unknown' if pd.isna(value) or str(value).strip().lower()
                      in ('', 'none', 'nan', 'null', 'unknown') else str(value))


def _available(frame, names):
    for name in names:
        if name in frame:
            return _labels(frame[name])
    return pd.Series('unknown', index=frame.index)


def _sport_profiles(frame, draws, seed):
    dates = pd.to_datetime(frame.date)
    names = _available(frame, ('tournament_name', 'competition'))
    identities = [classify_competition(name, day.date())
                  for name, day in zip(names, dates)]
    patch = _available(frame, ('target_patch', 'patch'))
    confidence = np.maximum(frame.p_c0, 1-frame.p_c0)
    dimensions = {
        'time': dates.dt.to_period('M').astype(str),
        'best_of': frame.best_of.astype(str),
        'tier': _available(frame, ('tier', 'competition_tier')) if any(
            name in frame for name in ('tier', 'competition_tier')) else pd.Series(
                [identity.tier.value for identity in identities], index=frame.index),
        'family': _available(frame, ('family', 'competition_family')) if any(
            name in frame for name in ('family', 'competition_family')) else pd.Series(
                [identity.family for identity in identities], index=frame.index),
        'unknown_players': _available(frame, ('unknown_players', 'unknown_player_count')),
        'new_organization': _available(frame, ('new_organization', 'new_org')),
        'confidence': pd.Series(np.select([confidence < .6, confidence < .75, confidence < .9],
                                ['[.5,.6)', '[.6,.75)', '[.75,.9)'], default='[.9,1)'), index=frame.index),
        'local_missingness': _available(frame, ('local_missingness',)),
        'support_missingness': _available(frame, ('support_missingness',)),
        'patch_known': pd.Series(np.where(patch.eq('unknown'), 'unknown', 'known'), index=frame.index),
        'patch_phase': _available(frame, ('patch_phase',)),
    }
    result = {}
    for dimension, labels in dimensions.items():
        result[dimension] = {}
        for bucket, group in frame.groupby(labels, sort=True, dropna=False):
            result[dimension][str(bucket)] = {
                'rows': len(group), 'metrics': _metrics(group),
                'contrasts': _monthly_contrasts(group, draws, seed),
                'descriptive_only': True,
            }
    return result


def _validate_partitions(frame):
    if 'partition' not in frame:
        raise ValueError('missing partition')
    years = pd.to_datetime(frame.date).dt.year
    expected = years.map({2024: 'train', 2025: 'cal', 2026: 'assessment'})
    if expected.isna().any():
        raise ValueError('forecast population must contain only 2024/2025/2026')
    excluded = frame.partition.eq('excluded_fit_labels')
    if excluded.any():
        if 'effective_label_release_day' not in frame:
            raise ValueError('excluded fit labels require effective release dates')
        reason = next((name for name in ('fit_exclusion_reason', 'partition_reason', 'exclusion_reason')
                       if name in frame), None)
        if reason is None or frame.loc[excluded, reason].isna().any() or (
                frame.loc[excluded, reason].astype(str).str.strip().eq('').any()):
            raise ValueError('excluded fit labels require an explicit reason')
        release = pd.to_datetime(frame.loc[excluded, 'effective_label_release_day'], errors='raise')
        boundary = pd.to_datetime((years.loc[excluded]+1).astype(str)+'-01-01')
        if years.loc[excluded].eq(2026).any() or release.isna().any() or not (release >= boundary).all():
            raise ValueError('invalid excluded fit label release boundary')
    if not frame.loc[~excluded, 'partition'].eq(expected.loc[~excluded]).all():
        raise ValueError('partition does not match fixed temporal cohorts')
    if not frame.partition.eq('assessment').any():
        raise ValueError('no primary assessment forecasts')
    warmup = years.lt(2026)
    for column in ('p_player', 'p_meta'):
        if not np.array_equal(frame.loc[warmup, column].to_numpy(float),
                              frame.loc[warmup, 'p_c0'].to_numpy(float)):
            raise ValueError('warmup main forecasts must remain exactly C0')
    if 'effective_label_release_day' in frame:
        for year, partition in ((2024, 'train'), (2025, 'cal')):
            release = pd.to_datetime(frame.loc[frame.partition.eq(partition), 'effective_label_release_day'], errors='raise')
            if release.isna().any() or not (release < pd.Timestamp(year+1, 1, 1)).all():
                raise ValueError('fit label not released strictly before fit boundary')


def evaluate_predictions(frame, *, draws=5000, seed=20261007) -> dict:
    """Score every supplied identity, with decisions restricted to assessment2026.

    Standalone scoring establishes statistical and coverage checks only. A formal
    evaluation additionally binds the saved artifacts and structural receipt.
    """
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws < 1:
        raise ValueError('draws must be a positive integer')
    development = [column for column in ('p_player_development', 'p_meta_development') if column in frame]
    validate_frame(frame, (*PROBABILITIES, *development), nullable=development)
    _validate_partitions(frame)
    primary = frame.loc[frame.partition.eq('assessment')]
    contrasts = _monthly_contrasts(primary, draws, seed, confidence=.975)
    checks = {'complete_coverage': True}
    for label, key in (('c0', 'meta_minus_c0'), ('player', 'meta_minus_player')):
        contrast = contrasts[key]
        upper = contrast['log_loss_monthly']['ci_upper']
        checks[f'meta_vs_{label}_ll_upper_negative'] = bool(
            upper is not None and np.isfinite(upper) and upper < 0)
        checks[f'meta_vs_{label}_brier_nonpositive'] = bool(contrast['brier_monthly']['delta'] <= 0)
    successful = all(checks.values())
    source = _available(primary, ('source', 'feature_source', 'source_name'))
    tournament = _available(primary, ('tournament_name', 'competition'))
    years = pd.to_datetime(primary.date).dt.year.astype(str)
    proxy_blocks = source + '|' + tournament + '|' + years
    proxy = {name: _contrast(primary, candidate, reference, proxy_blocks, draws, seed, .95)
             for name, (candidate, reference) in CONTRASTS.items()}
    diagnostics = {}
    for partition in ('train', 'cal'):
        subset = frame.loc[frame.partition.eq(partition)]
        diagnostics[partition] = {
            column: _summary(subset.loc[subset[column].notna()], column) for column in development
        }
    warmup = frame.loc[pd.to_datetime(frame.date).dt.year.lt(2026)]
    partition_metrics = {str(name): _metrics(group) for name, group in frame.groupby('partition', sort=False)}
    primary_metrics = partition_metrics['assessment']
    calibration_keys = ('calibration_slope', 'calibration_intercept', 'ece', 'mce',
                        'brier_reliability', 'brier_resolution', 'brier_uncertainty')
    equal_monthly = {
        name: {
            metric: {
                'delta': contrast[source]['equal_block_delta'],
                'ci_lower': contrast[source]['equal_block_ci_lower'],
                'ci_upper': contrast[source]['equal_block_ci_upper'],
                'blocks': contrast[source]['blocks'],
                'confidence_level': contrast[source]['confidence_level'],
            }
            for metric, source in (('log_loss', 'log_loss_monthly'), ('brier', 'brier_monthly'))
        }
        for name, contrast in contrasts.items()
    }
    result = {
        'status': 'DEVELOPMENT_SIGNAL_RETAINED' if successful else 'NOT_ADOPTED',
        'scope': 'DEVELOPMENT_ONLY_NOT_PROMOTION_EVIDENCE',
        'probability_epsilon': None,
        'bootstrap': {'draws': int(draws), 'seed': int(seed), 'primary_confidence': .975,
                      'family_confidence': .95, 'comparisons': 2,
                      'method': 'paired_monthly_blocks_two_sided_bonferroni'},
        'full_cohort': {'rows': len(frame), 'metrics': _metrics(frame),
                        'forecast_claim': 'complete_c0_warmup_plus_2026_assessment'},
        'partition_counts': {str(name): int(count) for name, count in frame.partition.value_counts(sort=False).items()},
        'partition_metrics': partition_metrics,
        'primary': {
            'partition': 'assessment', 'year': 2026, 'rows': len(primary),
            'metrics': primary_metrics, 'contrasts': contrasts,
            'calibration': {column: {key: metrics[key] for key in calibration_keys}
                            for column, metrics in primary_metrics.items()},
            'equal_monthly': {'descriptive_only': True, 'contrasts': equal_monthly},
            'source_name_year_proxy_sensitivity': {
                'certified_editions': False, 'descriptive_only': True,
                'unknown_sources_retained': True, 'contrasts': proxy,
                'block_definition': 'source/tournament_name/match_year; not certified editions'},
        },
        'warmup': {'rows': len(warmup), 'metrics': _metrics(warmup),
                   'forecast_claim': 'descriptive_exact_c0_only'},
        'development_diagnostics': diagnostics,
        'development_diagnostics_scope': 'fitted_TRAIN_and_CAL_not_out_of_fold_not_main_forecasts',
        'sport_profiles': _sport_profiles(primary, draws, seed),
        'decision': {'primary_success': successful, 'checks': checks,
                     'structural_verification': 'requires_separate_execution_receipt',
                     'model_promotion': False,
                     'retention_scope': 'development_candidate_only'},
    }
    return safe(result)


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _qualified_path(value, research_root):
    qualifier, separator, relative = value.partition(':')
    if not separator or qualifier not in ('project', 'research'):
        raise ValueError(f'unqualified source path: {value}')
    root = PROJECT_ROOT if qualifier == 'project' else research_root
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'invalid qualified source path: {value}')
    return root / path


def _verify_hash(path, expected, label):
    if not isinstance(expected, str) or sha(path) != expected:
        raise ValueError(f'{label} hash binding mismatch')


def _load_frame(path):
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix in ('.pkl', '.pickle'):
        return pd.read_pickle(path)
    if path.suffix == '.csv':
        return pd.read_csv(path, dtype={name: str for name in ('golgg_match_id', 'team1_id', 'team2_id')})
    raise ValueError(f'unsupported saved forecast format: {path.suffix}')


def evaluate_meta(protocol_path, output_dir, research_root=None) -> dict:
    """Verify a completed formal run, then exclusively save metrics and decision.

    No fitting, simulation of predictions, odds admission, or source acquisition
    occurs here. Bootstrap draws quantify paired losses on the saved forecasts.
    """
    output = _project_path(output_dir)
    if output.exists():
        raise FileExistsError(f'refusing existing output directory: {output}')
    protocol_path = _project_path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('schema') != 'c0_role_meta_v1' or protocol.get('status') != 'FROZEN_BEFORE_FITTING':
        raise ValueError('not a frozen formal role meta protocol')
    evaluation = protocol.get('evaluation', {})
    expected_evaluation = {'draws': 5000, 'seed': 20261007, 'primary_confidence': .975, 'probability_epsilon': None}
    if any(key not in evaluation or evaluation[key] != value for key, value in expected_evaluation.items()):
        raise ValueError('evaluation protocol differs from fixed paired scoring contract')
    execution_path = _project_path(protocol['execution_path'])
    execution = json.loads(execution_path.read_text())
    if execution.get('schema') != 'c0_meta_execution_v1' or execution.get('status') != 'COMPLETED':
        raise ValueError('only completed formal execution can be scored')
    _verify_hash(protocol_path, execution.get('protocol_sha256'), 'protocol')
    if execution.get('fits') != {'neural': 6, 'calibration': 2, 'base': 0}:
        raise ValueError('formal fit budget was not respected')
    sources = protocol.get('source_sha256')
    inputs = protocol.get('input_sha256')
    if not sources or not inputs or execution.get('source_sha256') != sources:
        raise ValueError('missing or different frozen source bindings')
    root = resolve_research_root(research_root)
    for label, bindings in (('source', sources), ('input', inputs)):
        for qualified, expected in bindings.items():
            _verify_hash(_qualified_path(qualified, root), expected, f'{label}:{qualified}')
    snapshot_path = _project_path(protocol['source_snapshot_path'])
    _verify_hash(snapshot_path, protocol['source_snapshot_sha256'], 'source snapshot')
    paths = {name: _project_path(protocol[name + '_path'])
             for name in ('features', 'frame', 'prepared_receipt', 'predictions')}
    for name, path in paths.items():
        _verify_hash(path, execution.get(name + '_sha256'), name)
    prepared = json.loads(paths['prepared_receipt'].read_text())
    if prepared.get('status') != 'PREPARED' or prepared.get('protocol_sha256') != execution['protocol_sha256']:
        raise ValueError('preparation receipt is not bound to formal protocol')
    for name in ('features', 'frame'):
        if prepared.get(name + '_sha256') != execution[name + '_sha256']:
            raise ValueError(f'preparation {name} binding differs')
    models = execution.get('models_sha256')
    if not models:
        raise ValueError('missing saved correction model bindings')
    for model, expected in models.items():
        _verify_hash(_project_path(model), expected, f'model:{model}')
    frame = _load_frame(paths['predictions'])
    canonical = _load_frame(paths['frame'])
    if len(frame) != 11550 or len(canonical) != 11550:
        raise ValueError('formal evaluation requires complete native11550 coverage')
    validate_frame(frame, PROBABILITIES)
    if canonical.golgg_match_id.astype(str).duplicated().any():
        raise ValueError('duplicate canonical coverage identity')
    aligned = canonical.assign(golgg_match_id=canonical.golgg_match_id.astype(str)).set_index('golgg_match_id')
    if set(aligned.index) != set(frame.golgg_match_id.astype(str)):
        raise ValueError('predictions do not cover exact canonical IDs')
    aligned = aligned.loc[frame.golgg_match_id.astype(str)].reset_index()
    for column in IDENTITY + ['result_day', 'feature_source_max_day']:
        dtype = int if column in ('best_of', 'y') else str
        if not np.array_equal(frame[column].to_numpy(dtype=dtype), aligned[column].to_numpy(dtype=dtype)):
            raise ValueError(f'canonical identity/chronology mismatch: {column}')
    if 'p_c0' not in aligned or not np.array_equal(frame.p_c0.to_numpy(float), aligned.p_c0.to_numpy(float)):
        raise ValueError('saved main C0 stream differs from prepared frozen baseline')
    if frame.partition.eq('assessment').sum() != 3103:
        raise ValueError('formal primary requires exact assessment2026 3103 coverage')
    result = evaluate_predictions(frame, draws=evaluation['draws'], seed=evaluation['seed'])
    structural = execution.get('structural', {})
    structural_passed = structural.get('passed') is True
    result['decision']['checks']['structural_receipt_passed'] = structural_passed
    result['decision']['primary_success'] = bool(result['decision']['primary_success'] and structural_passed)
    result['decision']['structural_verification'] = structural
    result['status'] = 'DEVELOPMENT_SIGNAL_RETAINED' if result['decision']['primary_success'] else 'NOT_ADOPTED'
    result['decision']['status'] = result['status']
    result['receipt'] = {'run_id': protocol['run_id'], 'protocol_sha256': sha(protocol_path),
                         'execution_sha256': sha(execution_path),
                         'predictions_sha256': execution['predictions_sha256'],
                         'complete_native_coverage_verified': True, 'source_bindings_verified': True}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('metrics.json', result), ('decision.json', result['decision'])):
        with (output / name).open('x') as stream:
            json.dump(safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    return result
