#!/usr/bin/env python3
"""Descriptive, predeclared support profiles of already-scored research outputs.

No fits, prediction changes, outcome-selected partitions or new advancement gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.champion_history_runner import ROOT, RUN_ID, _json, _paths, partitions, sha256

PROFILE_DEFINITION = {
    'inactivity': 'Maximum last-released-observation played-age across the ten current players; unsupported slots kept separate',
    'inactivity_bins_days': [7, 30, 90],
    'historical_stat_coverage': 'Observed cells among twelve performance channels on valid past tokens; padding excluded',
    'stat_coverage_bins': [.5, .9],
    'unfamiliar_player': 'Current roster has zero released history for at least one player, not missing identity',
    'competition': 'Exact supplied competition/edition/phase tag; descriptive source scenario',
    'champion_pool_breadth': 'Minimum resolved TRAIN-vocabulary distinct champions across ten last16 player histories; no-history and no-resolved-champion kept separate',
    'champion_pool_bins': [3, 6],
    'champion_pool_limit': 'Lower bound only: distinct out-of-TRAIN identities share UNK and are excluded; unknown-token fraction reported separately',
    'standard_absence': 'No verified STANDARD bindings in this source; UNKNOWN is not a standard-format control',
    'profile_use': 'Descriptive only; no re-selection or additional adoption gate',
}


def support_metadata(series, history, mask, champions):
    n = len(series)
    latest = np.where(mask[..., -1], history[..., -1, 18], np.nan).reshape(n, 10)
    supported = np.isfinite(latest)
    maximum = np.where(supported, latest, -1).max(axis=1)
    inactivity = np.select([~supported.any(1), ~supported.all(1), maximum <= 7,
                            maximum <= 30, maximum <= 90],
                           ['NO_SUPPORTED_PLAYER', 'AT_LEAST_ONE_UNSUPPORTED', '0-7', '8-30', '31-90'], default='91+')
    observed = (np.isfinite(history[..., :12]) & mask[..., None]).sum(axis=(1, 2, 3, 4))
    denominator = mask.sum(axis=(1, 2, 3)) * 12
    coverage = np.divide(observed, denominator, out=np.full(n, np.nan), where=denominator > 0)
    stat_profile = np.select([denominator == 0, coverage < .5, coverage < .9],
                            ['NO_HISTORY', 'BELOW_50_PERCENT', '50_TO_90_PERCENT'], default='AT_LEAST_90_PERCENT')
    resolved = np.sort(np.where(mask & (champions >= 2), champions, 0), axis=-1)
    distinct = ((resolved > 0) & (resolved != np.concatenate(
        (np.zeros_like(resolved[..., :1]), resolved[..., :-1]), axis=-1))).sum(axis=-1)
    minimum = distinct.min(axis=(1, 2))
    pool_profile = np.select([~mask.any(-1).all(axis=(1, 2)), minimum == 0, minimum <= 3, minimum <= 6],
                             ['AT_LEAST_ONE_NO_HISTORY', 'AT_LEAST_ONE_NO_RESOLVED_CHAMPION', 'MIN_1_TO_3', 'MIN_4_TO_6'],
                             default='MIN_7_PLUS')
    unknown_fraction = np.divide(((champions == 1) & mask).sum(axis=(1, 2, 3)), mask.sum(axis=(1, 2, 3)),
                                 out=np.full(n, np.nan), where=mask.sum(axis=(1, 2, 3)) > 0)
    return pd.DataFrame({'row_index': series.row_index.to_numpy(), 'inactivity_profile': inactivity,
                         'max_supported_inactivity_days': np.where(supported.any(1), maximum, np.nan),
                         'historical_stat_coverage': coverage, 'stat_mask_profile': stat_profile,
                         'unfamiliar_player_profile': np.where(supported.all(1), 'ALL_HISTORY_SUPPORTED', 'AT_LEAST_ONE_NO_HISTORY'),
                         'champion_pool_profile': pool_profile,
                         'min_resolved_champion_pool': minimum,
                         'mean_resolved_champion_pool': distinct.mean(axis=(1, 2)),
                         'unknown_champion_token_fraction': unknown_fraction,
                         'competition_profile': series.tournament.fillna('UNKNOWN').astype(str).to_numpy()})


def report(run_id=RUN_ID):
    paths = _paths(run_id)
    definition_path = paths['05_model_input'] / 'profile_protocol.json'
    definition = json.loads(definition_path.read_text())
    if definition['definition'] != PROFILE_DEFINITION or definition['script_sha256'] != sha256(Path(__file__)):
        raise ValueError('Descriptive profile definition/script differs from its pre-assessment binding')
    series = pd.read_parquet(paths['04_feature'] / 'series.parquet')
    indices = partitions(series)['ASSESS']
    series = series.iloc[indices].reset_index(drop=True)
    with np.load(paths['04_feature'] / 'arrays.npz') as saved:
        history, mask, champions = saved['history'][indices], saved['mask'][indices], saved['champions'][indices]
    support = support_metadata(series, history, mask, champions)
    support.to_parquet(paths['08_reporting'] / 'support_profiles.parquet', index=False)
    played = pd.read_parquet(paths['04_feature'] / 'maps.parquet')
    played = played.loc[played.row_index.isin(indices)]
    absence_rows = []
    for candidate in ('A', 'B', 'C0', 'old_props'):
        missing = played.loc[played.map_index > 1] if candidate == 'old_props' else played
        for task in ('kills', 'duration'):
            row = missing[['row_index', 'match_id', 'map_index']].copy()
            row['candidate'], row['task'] = candidate, task
            row['reason'] = ('EXISTING_REFERENCE_IS_MAP1_ONLY' if candidate == 'old_props' else
                             'STATISTICS_HEAD_DISABLED' if candidate in ('A', 'B') else 'NO_STATISTICS_REFERENCE')
            absence_rows.append(row)
    for task in ('winner', 'score'):
        row = series[['row_index', 'match_id']].copy()
        row['map_index'], row['candidate'], row['task'] = 0, 'old_props', task
        row['reason'] = 'PROP_BUNDLE_HAS_NO_WINNER_OR_TERMINAL_SCORE_HEAD'
        absence_rows.append(row)
    absences = pd.concat(absence_rows, ignore_index=True)
    absence_path = paths['07_model_output'] / 'reference_head_absences.parquet'
    absences.to_parquet(absence_path, index=False)
    summaries = []
    candidate_absences = {}
    dimensions = ('inactivity_profile', 'stat_mask_profile', 'unfamiliar_player_profile', 'competition_profile', 'champion_pool_profile')
    for candidate in ('A', 'B', 'C', 'D', 'C0', 'old_props'):
        for scope, suffix in (('series', 'series_scores'), ('played_map', 'map_scores')):
            filename = f'{candidate}_{suffix}.parquet'
            if candidate == 'old_props' and scope == 'played_map':
                filename = 'old_props_first_map_scores.parquet'
            if candidate == 'C0' and scope == 'played_map':
                continue
            scores = pd.read_parquet(paths['07_model_output'] / filename)
            if candidate in ('C', 'D') and scope == 'series':
                candidate_absences[candidate] = {task: int(scores[f'{task}_nll'].isna().sum())
                                                for task in ('winner', 'score', 'kills', 'duration')}
            joined = scores.merge(support, on='row_index', how='left', validate='many_to_one')
            for dimension in dimensions:
                for value, group in joined.groupby(dimension, sort=True, dropna=False):
                    for task in (('winner', 'score', 'length', 'kills', 'duration') if scope == 'series' else ('kills', 'duration')):
                        column = f'{task}_nll'
                        if column not in group:
                            continue
                        valid = group[column].notna()
                        row = {'candidate': candidate, 'scope': scope, 'dimension': dimension, 'value': str(value),
                               'task': task, 'total': len(group), 'n': int(valid.sum()), 'absent': int((~valid).sum())}
                        for metric in ('nll', 'brier', 'rps', 'crps', 'mae', 'coverage_50', 'width_50', 'coverage_80', 'width_80'):
                            key = f'{task}_{metric}'
                            row[metric] = float(group[key].mean()) if key in group and group[key].notna().any() else np.nan
                        summaries.append(row)
    frame = pd.DataFrame(summaries)
    frame.to_csv(paths['08_reporting'] / 'supplemental_profiles.csv', index=False)
    format_coverage = {mode: int(series.draft_mode.eq(mode).sum()) for mode in ('STANDARD', 'HARD_FEARLESS', 'SOFT_FEARLESS', 'UNKNOWN')}
    receipt = {'run_id': run_id, 'status': 'PASS', 'definition': PROFILE_DEFINITION, 'series': len(series),
               'profile_rows': len(frame), 'format_coverage': format_coverage,
               'explicit_reference_head_absences': len(absences),
               'C_D_label_or_prediction_absences': candidate_absences,
               'observed_numeric_missing_excludes_padding': True, 'prediction_changes': 0, 'fits': 0,
               'script_sha256': sha256(Path(__file__)), 'output_hashes': {
                   str(p.relative_to(ROOT)): sha256(p) for p in (paths['08_reporting'] / 'support_profiles.parquet',
                                                                 paths['08_reporting'] / 'supplemental_profiles.csv', absence_path)}}
    _json(paths['08_reporting'] / 'profile_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    args = parser.parse_args()
    print(json.dumps(report(args.run_id), indent=2))
