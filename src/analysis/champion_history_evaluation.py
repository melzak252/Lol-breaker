"""Coherent pre-series proper scores and whole-series monthly comparisons.

Recorded map rows are conditional played-map observations, never synthetic zero
kills/duration for unplayed maps. Bootstrap inputs are one row per series, with
map losses averaged within series. Missing-task months remain in the block
universe; draws with no paired labels are explicitly counted as undefined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.analysis.prop_distribution_evaluation import (
    _components, _mixture_quantile, lognormal_mixture_crps,
    negative_binomial_mixture_crps,
)
from src.models.prop_distribution import DistributionPrediction, DistributionMixture, distribution_nll


_STAT_COLUMNS = ('nll', 'crps', 'mae', 'coverage_50', 'width_50', 'coverage_80', 'width_80')
_PROFILE_ALIASES = {
    'patch': ('target_patch', 'patch'),
    'rule': ('draft_mode', 'rule_status', 'rule_profile', 'rule_known', 'rule'),
    'rule_unknown': ('rule_unknown',),
    'history': ('history_profile', 'history_support_profile', 'history_support', 'history_count'),
    'inactivity': ('inactivity_profile', 'inactivity_days', 'inactivity'),
    'new_roster': ('roster_profile', 'new_roster_profile', 'new_roster', 'roster_new'),
    'stat_mask': ('stat_mask_profile', 'stat_mask', 'stat_missingness'),
    'unknown_support': ('unknown_support_profile', 'unknown_champion_fraction', 'unknown_support', 'unknown_player_count'),
    'missing_champion_support': ('missing_champion_history',),
    'unseen_champion_support': ('unseen_champion_history',),
}


def _mean(values):
    values = np.asarray(values, float)
    observed = ~np.isnan(values)
    return float(values[observed].mean()) if observed.any() else None


def _labels(frame, name):
    values = frame[name].to_numpy(float) if name in frame else np.full(len(frame), np.nan)
    if np.isinf(values).any():
        raise ValueError(f'{name} labels cannot be infinite')
    return values


def _negative_log(probability):
    with np.errstate(divide='ignore'):
        return -np.log(probability)


def _stat_capability(prediction):
    components = tuple(getattr(prediction, 'components', ()))
    if components:
        capabilities = [_stat_capability(component) for component in components]
        if any(value is False for value in capabilities) and any(value is True for value in capabilities):
            raise ValueError('ensemble mixes score-only and multitask components')
        return False if all(value is False for value in capabilities) else None if any(value is None for value in capabilities) else True
    sentinel = object()
    value = getattr(prediction, 'kills_mu', sentinel)
    return None if value is sentinel else value is not None


def _inputs(prediction, series, maps):
    series = series.copy().reset_index(drop=True)
    maps = maps.copy().reset_index(drop=True)
    for column in ('row_index', 'match_id', 'date', 'best_of'):
        if column not in series:
            raise ValueError(f'series requires {column}')
    if series.row_index.isna().any() or series.row_index.duplicated().any() or series.match_id.duplicated().any():
        raise ValueError('series needs unique nonmissing row_index and match_id')
    bo = series.best_of.to_numpy(float)
    if not np.isin(bo, [1, 3, 5]).all():
        raise ValueError('best_of must be 1, 3 or 5')
    bo = bo.astype(int)
    if not np.array_equal(np.asarray(prediction.best_of), bo):
        raise ValueError('prediction best_of must match series in identical order')
    for column in ('row_index', 'map_index'):
        if column not in maps:
            raise ValueError(f'maps requires {column}')
    if not maps.row_index.isin(series.row_index).all():
        raise ValueError('map row_index is outside series population')
    if maps.duplicated(['row_index', 'map_index']).any():
        raise ValueError('duplicate canonical map within series')
    positions = pd.Series(np.arange(len(series)), index=series.row_index)
    position = maps.row_index.map(positions).to_numpy(int)
    indices = maps.map_index.to_numpy(float)
    if not np.isfinite(indices).all() or np.any(indices != np.floor(indices)) or np.any(indices < 1) or np.any(indices > bo[position]):
        raise ValueError('map_index is outside legal best_of support')
    metadata = {'match_id', 'date', 'best_of', 'tournament'}
    metadata.update(name for aliases in _PROFILE_ALIASES.values() for name in aliases)
    for column in series:
        if column in metadata and column not in maps:
            maps[column] = series[column].to_numpy()[position]
    for column in ('kills', 'duration_minutes'):
        if column not in maps:
            maps[column] = np.nan
    if 'game_id' not in maps:
        maps['game_id'] = 'UNKNOWN'
    return series, maps, position, bo


def _subset_distribution(prediction, positions):
    components, weights = _components(prediction)
    sliced = tuple(DistributionPrediction(
        c.kills_mu[positions], c.kills_alpha[positions],
        c.duration_log_mu[positions], c.duration_log_sigma[positions],
        c.length_probs[positions], c.best_of[positions],
    ) for c in components)
    return sliced[0] if len(sliced) == 1 else DistributionMixture(sliced, weights)


def _initialize_map_scores(maps):
    rows = maps.copy()
    for task, label in (('kills', 'kills'), ('duration', 'duration_minutes')):
        rows[f'{task}_label_observed'] = ~np.isnan(_labels(rows, label))
        reason = f'{task}_absence_reason'
        if reason not in rows:
            source = next((name for name in (f'{task}_absence', f'{task}_reason') if name in rows), None)
            rows[reason] = rows[source].fillna('').astype(str) if source else ''
        else:
            rows[reason] = rows[reason].fillna('').astype(str)
        missing = ~rows[f'{task}_label_observed'] & rows[reason].eq('')
        rows.loc[missing, reason] = 'missing_label'
        rows.loc[rows[f'{task}_label_observed'], reason] = ''
        for suffix in (*_STAT_COLUMNS, 'mean', 'q10', 'q25', 'q75', 'q90'):
            rows[f'{task}_{suffix}'] = np.nan
        rows[f'{task}_proper_secondary'] = np.nan
    return rows


def _score_map_distribution(prediction, rows, indices, positions):
    pred = _subset_distribution(prediction, positions)
    frame = rows.loc[indices]
    kills, duration = _labels(frame, 'kills'), _labels(frame, 'duration_minutes')
    losses = distribution_nll(pred, kills, duration, np.full(len(indices), np.nan))
    components, weights = _components(pred)
    tail_bound = None
    for task, labels in (('kills', kills), ('duration', duration)):
        valid = ~np.isnan(labels)
        mu = np.column_stack([getattr(c, 'kills_mu' if task == 'kills' else 'duration_log_mu') for c in components])
        spread = np.column_stack([getattr(c, 'kills_alpha' if task == 'kills' else 'duration_log_sigma') for c in components])
        mean = mu @ weights if task == 'kills' else np.exp(mu + .5 * spread**2) @ weights
        secondary = np.full(len(indices), np.nan)
        if valid.any():
            if task == 'kills':
                secondary[valid], tail_bound = negative_binomial_mixture_crps(labels[valid], mu[valid], spread[valid], weights)
            else:
                secondary[valid] = lognormal_mixture_crps(labels[valid], mu[valid], spread[valid], weights)
        rows.loc[indices, f'{task}_nll'] = losses[task]
        rows.loc[indices, f'{task}_crps'] = secondary
        rows.loc[indices, f'{task}_proper_secondary'] = secondary
        rows.loc[indices, f'{task}_mean'] = mean
        rows.loc[indices, f'{task}_mae'] = np.where(valid, np.abs(mean-labels), np.nan)
        for lower_p, upper_p, level in ((.25, .75, 50), (.1, .9, 80)):
            lower = _mixture_quantile(pred, lower_p, task)
            upper = _mixture_quantile(pred, upper_p, task)
            rows.loc[indices, f'{task}_q{int(lower_p*100)}'] = lower
            rows.loc[indices, f'{task}_q{int(upper_p*100)}'] = upper
            rows.loc[indices, f'{task}_coverage_{level}'] = np.where(valid, (labels >= lower) & (labels <= upper), np.nan)
            rows.loc[indices, f'{task}_width_{level}'] = np.where(valid, upper-lower, np.nan)
    return tail_bound


def _winner_metrics(rows, bins):
    valid = rows.winner.notna().to_numpy()
    y, p = rows.loc[valid, 'winner'].to_numpy(), rows.loc[valid, 'p_win'].to_numpy()
    result = {'n': len(y), 'absent_labels': int((~valid).sum()),
              'nll': _mean(rows.winner_nll), 'brier': _mean(rows.winner_brier),
              'auc': None, 'calibration_ece': None, 'calibration_bins': []}
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    if positive and negative:
        result['auc'] = float((rankdata(p)[y == 1].sum()-positive*(positive+1)/2)/(positive*negative))
    result['auc_absence_reason'] = None if positive and negative else 'requires_both_winner_classes'
    assignment = np.minimum((p*bins).astype(int), bins-1)
    ece = 0.
    for index in range(bins):
        chosen = assignment == index
        count = int(chosen.sum())
        forecast, outcome = _mean(p[chosen]), _mean(y[chosen])
        if count:
            ece += count/len(y)*abs(forecast-outcome)
        result['calibration_bins'].append({'lower': index/bins, 'upper': (index+1)/bins,
                                          'n': count, 'mean_prediction': forecast, 'observed_rate': outcome})
    result['calibration_ece'] = ece if len(y) else None
    return result


def _aggregate_stats(series_rows, map_rows, availability, tail_bound):
    metrics = {}
    for task in ('kills', 'duration'):
        for suffix in _STAT_COLUMNS:
            means = map_rows.groupby('row_index', sort=False)[f'{task}_{suffix}'].agg(_mean)
            series_rows[f'{task}_{suffix}'] = series_rows.row_index.map(means).astype(float)
        series_rows[f'{task}_proper_secondary'] = series_rows[f'{task}_crps']
        grouped = map_rows.groupby('row_index', sort=False)
        label_counts = grouped[f'{task}_label_observed'].sum()
        score_counts = grouped[f'{task}_nll'].count()
        series_rows[f'{task}_observed_maps'] = series_rows.row_index.map(label_counts).fillna(0).astype(int)
        series_rows[f'{task}_scored_maps'] = series_rows.row_index.map(score_counts).fillna(0).astype(int)
        series_rows[f'{task}_absence_reason'] = np.where(
            series_rows[f'{task}_scored_maps'] > 0, '',
            availability if availability == 'unavailable_score_only' else 'no_observed_supported_map_label',
        )
        label_count = int(map_rows[f'{task}_label_observed'].sum())
        score_count = int(map_rows[f'{task}_nll'].notna().sum())
        row = {'status': availability, 'n': score_count, 'observed_labels': label_count,
               'absent_labels': len(map_rows)-label_count, 'unscored_observed_labels': label_count-score_count,
               'series_n': int(series_rows[f'{task}_nll'].notna().sum()),
               'map_weighted': {}, 'series_weighted': {}}
        for suffix in _STAT_COLUMNS:
            row['map_weighted'][suffix] = _mean(map_rows[f'{task}_{suffix}'])
            row['series_weighted'][suffix] = _mean(series_rows[f'{task}_{suffix}'])
        for estimand in ('map_weighted', 'series_weighted'):
            stats = row[estimand]
            stats['mae_predictive_mean'] = stats['mae']
            for level in (50, 80):
                stats[f'central_{level}_coverage'] = stats[f'coverage_{level}']
                stats[f'central_{level}_mean_width'] = stats[f'width_{level}']
        row.update(row['map_weighted'])
        if task == 'kills':
            row['crps_tail_bound_max'] = tail_bound
        metrics[task] = row
    return metrics


def _profiles(series_rows, map_rows):
    result = []
    for scope, source in (('series', series_rows), ('played_map', map_rows)):
        working = source.copy()
        dates = pd.to_datetime(working['date'], errors='coerce')
        working['time'] = dates.dt.to_period('M').astype(str).replace('NaT', 'UNKNOWN')
        dimensions = ['best_of', 'time']
        if scope == 'played_map':
            dimensions.append('map_index')
        for dimension, aliases in _PROFILE_ALIASES.items():
            column = next((name for name in aliases if name in working), None)
            working[dimension] = working[column].fillna('UNKNOWN').astype(str) if column else 'UNKNOWN'
            dimensions.append(dimension)
        tasks = ['kills', 'duration'] if scope == 'played_map' else ['winner', 'score', 'length', 'kills', 'duration']
        if 'map_play_brier' in working:
            tasks.append('map_play')
        for dimension in dimensions:
            for value, group in working.groupby(dimension, dropna=False, sort=True):
                for task in tasks:
                    column = 'map_play_brier' if task == 'map_play' else f'{task}_nll'
                    n = int(group[column].notna().sum())
                    secondary = {'winner': 'winner_brier', 'length': 'length_rps',
                                 'kills': 'kills_crps', 'duration': 'duration_crps',
                                 'map_play': 'map_play_brier'}.get(task)
                    result.append({'scope': scope, 'dimension': dimension, 'value': str(value),
                                   'task': task, 'total': len(group), 'n': n, 'absent': len(group)-n,
                                   'nll': None if task == 'map_play' else _mean(group[column]),
                                   'proper_secondary': _mean(group[secondary]) if secondary else None})
    return pd.DataFrame(result, columns=['scope', 'dimension', 'value', 'task', 'total', 'n', 'absent', 'nll', 'proper_secondary'])


def evaluate_prediction(prediction, series, maps, *, calibration_bins=10):
    """Score every series and recorded map, retaining explicit label absences.

    Alignment is prediction order == series order, NOT numerical row_index.
    A score-only map_prediction must raise NotImplementedError; optional raw
    fields declare absent heads recursively through exact ensemble components.
    """
    if not isinstance(calibration_bins, int) or calibration_bins < 1:
        raise ValueError('calibration_bins must be a positive integer')
    rows, maps, positions, bo = _inputs(prediction, series, maps)
    probabilities = np.asarray(prediction.score_probs, float)
    n = len(rows)
    legal = np.tile(np.arange(3), 2)[None, :] <= (bo//2)[:, None]
    if probabilities.shape != (n, 6) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError('score_probs must be finite nonnegative [N,6]')
    if not np.allclose(probabilities.sum(axis=1), 1) or np.any(probabilities[~legal] != 0):
        raise ValueError('score_probs must be normalized with legal terminal support')
    winner_p = probabilities[:, :3].sum(axis=1)
    length_p = probabilities[:, :3] + probabilities[:, 3:]
    grid = (bo//2+1)[:, None] + np.arange(3)
    play_p = np.column_stack([(length_p*(grid >= m)).sum(axis=1) for m in range(1, 6)])
    for name, derived in (('p_win', winner_p), ('length_probs', length_p), ('play_probs', play_p)):
        given = np.asarray(getattr(prediction, name), float)
        if given.shape != derived.shape or not np.allclose(given, derived, rtol=1e-7, atol=1e-9):
            raise ValueError(f'{name} is incoherent with terminal scores')
    score, winner = _labels(rows, 'score_index'), _labels(rows, 'winner')
    observed_score, observed_winner = ~np.isnan(score), ~np.isnan(winner)
    if np.any(~np.isin(score[observed_score], np.arange(6))):
        raise ValueError('score_index must be an integer 0..5 or NaN')
    score_idx = score[observed_score].astype(int)
    if np.any(~legal[np.flatnonzero(observed_score), score_idx]):
        raise ValueError('score label outside legal terminal support')
    if np.any(~np.isin(winner[observed_winner], [0, 1])):
        raise ValueError('winner labels must be 0/1 or NaN')
    both = observed_score & observed_winner
    if np.any(winner[both] != (score[both] < 3)):
        raise ValueError('winner label contradicts terminal score')
    length = _labels(rows, 'maps_played')
    derived_length = np.full(n, np.nan)
    derived_length[observed_score] = bo[observed_score]//2+1+score_idx%3
    both_length = ~np.isnan(length) & observed_score
    if np.any(length[both_length] != derived_length[both_length]):
        raise ValueError('maps_played contradicts terminal score')
    length = np.where(np.isnan(length), derived_length, length)
    known_length = ~np.isnan(length)
    if np.any((length[known_length] != np.floor(length[known_length])) | (length[known_length] < bo[known_length]//2+1) | (length[known_length] > bo[known_length])):
        raise ValueError('maps_played outside legal best_of support')
    if np.any(maps.map_index.to_numpy()[known_length[positions]] > length[positions[known_length[positions]]]):
        raise ValueError('recorded map contradicts known terminal length')
    rows['winner'], rows['score_index'], rows['maps_played'] = winner, score, length
    for task, observed in (('winner', observed_winner), ('score', observed_score)):
        column = f'{task}_absence_reason'
        if column not in rows:
            source = next((name for name in (f'{task}_absence', f'{task}_reason') if name in rows), None)
            rows[column] = rows[source].fillna('').astype(str) if source else ''
        else:
            rows[column] = rows[column].fillna('').astype(str)
        rows.loc[~observed & rows[column].eq(''), column] = 'missing_label'
        rows.loc[observed, column] = ''
    rows['p_win'] = winner_p
    rows['winner_nll'] = np.nan
    rows.loc[observed_winner, 'winner_nll'] = _negative_log(np.where(winner[observed_winner] == 1, winner_p[observed_winner], 1-winner_p[observed_winner]))
    rows['winner_brier'] = (winner_p-winner)**2
    rows['score_nll'] = np.nan
    rows.loc[observed_score, 'score_nll'] = _negative_log(probabilities[np.flatnonzero(observed_score), score_idx])
    valid_length = known_length & (bo > 1)
    rows['length_nll'], rows['length_rps'] = np.nan, np.nan
    length_index = (length[valid_length]-(bo[valid_length]//2+1)).astype(int)
    rows.loc[valid_length, 'length_nll'] = _negative_log(length_p[np.flatnonzero(valid_length), length_index])
    thresholds = np.arange(3)[None, :] < (bo-bo//2-1)[:, None]
    rps = np.sum(np.where(thresholds, (np.cumsum(length_p, axis=1)-(length[:, None] <= grid))**2, 0), axis=1)
    rows.loc[valid_length, 'length_rps'] = rps[valid_length]
    rows['length_proper_secondary'] = rows.length_rps
    rows['length_absence_reason'] = np.where(bo == 1, 'excluded_bo1', np.where(known_length, '', 'missing_label'))
    rows['map_play_absence_reason'] = np.where(known_length, '', 'missing_terminal_length')
    play_losses = (play_p-(np.arange(1, 6)[None, :] <= length[:, None]))**2
    play_valid = known_length[:, None] & (np.arange(1, 6)[None, :] <= bo[:, None])
    play_losses = np.where(play_valid, play_losses, np.nan)
    rows['map_play_brier'] = np.where(known_length, np.nansum(play_losses, axis=1)/bo, np.nan)
    map_rows = _initialize_map_scores(maps)
    map_rows['play_probability'] = play_p[positions, maps.map_index.to_numpy(int)-1]
    map_rows['map_play_brier'] = (map_rows.play_probability-1)**2
    map_rows['map_play_outcome'] = 1
    # Absence is an explicit model capability, not a catch-all for broken heads.
    availability = 'available'
    if _stat_capability(prediction) is False:
        availability = 'unavailable_score_only'
    tail_bounds = []
    for m in sorted(map_rows.map_index.unique()):
        selected = np.flatnonzero(map_rows.map_index.to_numpy() == m)
        if availability == 'available':
            try:
                marginal = prediction.map_prediction(int(m))
            except NotImplementedError:
                availability = 'unavailable_score_only'
            else:
                bound = _score_map_distribution(marginal, map_rows, selected, positions[selected])
                if bound is not None:
                    tail_bounds.append(bound)
        if availability != 'available':
            for task in ('kills', 'duration'):
                map_rows.loc[selected, f'{task}_absence_reason'] = availability
    metrics = {'winner': _winner_metrics(rows, calibration_bins),
               'score': {'n': int(observed_score.sum()), 'absent_labels': int((~observed_score).sum()), 'nll': _mean(rows.score_nll)},
               'length': {'n': int(valid_length.sum()), 'absent_labels': int((~known_length).sum()),
                          'excluded_bo1': int((bo == 1).sum()), 'nll': _mean(rows.length_nll), 'rps': _mean(rows.length_rps)},
               'map_play': {'n': int(play_valid.sum()), 'absent_series_labels': int((~known_length).sum()),
                            'brier': _mean(play_losses.ravel()), 'series_weighted_brier': _mean(rows.map_play_brier),
                            'by_map_index': [{'map_index': m, 'n': int(play_valid[:, m-1].sum()),
                                              'brier': _mean(play_losses[:, m-1])} for m in range(1, 6)]},
               'population': {'series': n, 'recorded_maps': len(map_rows)},
               'props_estimand': 'conditional_on_played_map; primary_bootstrap_mean_within_series'}
    metrics.update(_aggregate_stats(rows, map_rows, availability, max(tail_bounds) if tail_bounds else None))
    return metrics, rows, map_rows, _profiles(rows, map_rows)


def evaluate_first_map_baseline(prediction, series, maps):
    """Score supplied per-series first-map DistributionPrediction/Mixture only.

    Returns the same four objects as evaluate_prediction; winner/terminal-score
    heads are not invented. Later map rows remain explicit unsupported rows.
    """
    rows, maps, positions, bo = _inputs(prediction, series, maps)
    scored = _initialize_map_scores(maps)
    first = np.flatnonzero(scored.map_index.to_numpy() == 1)
    bound = _score_map_distribution(prediction, scored, first, positions[first]) if len(first) else None
    later = scored.map_index.to_numpy() > 1
    for task in ('kills', 'duration'):
        scored.loc[later, f'{task}_absence_reason'] = 'unsupported_later_map_baseline'
    for column in ('winner_nll', 'winner_brier', 'score_nll', 'length_nll', 'length_rps'):
        rows[column] = np.nan
    length = _labels(rows, 'maps_played').copy()
    score = _labels(rows, 'score_index')
    known_score = ~np.isnan(score)
    legal = np.tile(np.arange(3), 2)[None, :] <= (bo//2)[:, None]
    if np.any(~np.isin(score[known_score], np.arange(6))):
        raise ValueError('score_index must be an integer 0..5 or NaN')
    if np.any(~legal[np.flatnonzero(known_score), score[known_score].astype(int)]):
        raise ValueError('score label outside legal terminal support')
    derived = np.full(len(rows), np.nan)
    derived[known_score] = bo[known_score]//2+1+score[known_score]%3
    both = known_score & ~np.isnan(length)
    if np.any(length[both] != derived[both]):
        raise ValueError('maps_played contradicts terminal score')
    length = np.where(np.isnan(length), derived, length)
    known_length = ~np.isnan(length)
    if np.any(maps.map_index.to_numpy()[known_length[positions]] > length[positions[known_length[positions]]]):
        raise ValueError('recorded map contradicts known terminal length')
    rows['maps_played'] = length
    valid = ~np.isnan(length) & (bo > 1)
    losses = distribution_nll(prediction, np.full(len(rows), np.nan), np.full(len(rows), np.nan), length)
    rows.loc[valid, 'length_nll'] = losses['length'][valid]
    comps, weights = _components(prediction)
    probs = sum(w*c.length_probs for w, c in zip(weights, comps))
    grid = (bo//2+1)[:, None]+np.arange(3)
    terms = (np.cumsum(probs, axis=1)-(length[:, None] <= grid))**2
    rps = np.where(np.arange(3)[None, :] < (bo-bo//2-1)[:, None], terms, 0).sum(axis=1)
    rows.loc[valid, 'length_rps'] = rps[valid]
    rows['length_proper_secondary'] = rows.length_rps
    rows['length_absence_reason'] = np.where(bo == 1, 'excluded_bo1', np.where(known_length, '', 'missing_label'))
    play_p = np.column_stack([(probs*(grid >= m)).sum(axis=1) for m in range(1, 6)])
    play_valid = known_length[:, None] & (np.arange(1, 6)[None, :] <= bo[:, None])
    play_losses = np.where(play_valid, (play_p-(np.arange(1, 6)[None, :] <= length[:, None]))**2, np.nan)
    rows['map_play_brier'] = np.where(known_length, np.nansum(play_losses, axis=1)/bo, np.nan)
    rows['map_play_absence_reason'] = np.where(known_length, '', 'missing_terminal_length')
    scored['play_probability'] = play_p[positions, maps.map_index.to_numpy(int)-1]
    scored['map_play_brier'] = (scored.play_probability-1)**2
    scored['map_play_outcome'] = 1
    metrics = _aggregate_stats(rows, scored, 'first_map_only', bound)
    for task in ('kills', 'duration'):
        metrics[task]['unsupported_later_maps'] = int(later.sum())
    metrics['length'] = {'n': int(valid.sum()), 'nll': _mean(rows.length_nll), 'rps': _mean(rows.length_rps),
                         'absent_labels': int(np.isnan(length).sum()), 'excluded_bo1': int((bo == 1).sum())}
    metrics['map_play'] = {'n': int(play_valid.sum()), 'brier': _mean(play_losses.ravel()),
                           'series_weighted_brier': _mean(rows.map_play_brier),
                           'absent_series_labels': int((~known_length).sum()),
                           'by_map_index': [{'map_index': m, 'n': int(play_valid[:, m-1].sum()),
                                             'brier': _mean(play_losses[:, m-1])} for m in range(1, 6)]}
    metrics['winner'] = {'status': 'unsupported_baseline_head', 'n': 0}
    metrics['score'] = {'status': 'unsupported_baseline_head', 'n': 0}
    metrics['later_map_baseline'] = 'unsupported; first-map predictions are never reused'
    return metrics, rows, scored, _profiles(rows, scored)


def paired_bootstrap(candidate_rows, control_rows, metrics=None, *, draws=5000, seed=20261007,
                     family_confidence=.95, family_size=None, confidence=None):
    """Paired whole-series calendar-month bootstrap for a metric family.

    The same block draws are used for every metric and both estimands. A task's
    paired finite-label intersection affects its denominator, not its month
    universe. Empty sampled intersections are undefined, not zero or redrawn.
    Confidence defaults to Bonferroni correction of the stated family level.
    """
    for column in ('row_index', 'match_id', 'date'):
        if column not in candidate_rows or column not in control_rows:
            raise ValueError(f'paired series scores require {column}')
        if not candidate_rows[column].reset_index(drop=True).equals(control_rows[column].reset_index(drop=True)):
            raise ValueError('paired predictions must use identical series population and date order')
    if candidate_rows.row_index.duplicated().any() or candidate_rows.match_id.duplicated().any():
        raise ValueError('bootstrap requires one aggregated row per whole series')
    if not isinstance(draws, int) or draws < 1:
        raise ValueError('draws must be a positive integer')
    metrics = list(metrics) if metrics is not None else [c for c in candidate_rows if c.endswith('_nll') and c in control_rows]
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError('metrics must be a nonempty distinct family')
    size = len(metrics) if family_size is None else family_size
    if not isinstance(size, int) or size < len(metrics) or not 0 < family_confidence < 1:
        raise ValueError('family_size must cover all metrics and family_confidence must be in (0,1)')
    confidence = 1-(1-family_confidence)/size if confidence is None else confidence
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0,1)')
    dates = pd.to_datetime(candidate_rows.date, errors='coerce')
    if dates.isna().any() or len(dates) == 0:
        raise ValueError('monthly bootstrap needs nonempty dated series population')
    row_months = dates.dt.to_period('M').astype(str)
    months = sorted(row_months.unique())
    codes = pd.Categorical(row_months, categories=months).codes
    sampled = np.random.default_rng(seed).integers(0, len(months), size=(draws, len(months)))
    results = {}
    tail = (1-confidence)/2
    for metric in metrics:
        a, b = candidate_rows[metric].to_numpy(float), control_rows[metric].to_numpy(float)
        if np.isinf(a).any() or np.isinf(b).any():
            raise ValueError(f'{metric}: infinite proper scores have no finite bootstrap delta')
        valid = np.isfinite(a) & np.isfinite(b)
        delta = a[valid]-b[valid]
        sums = np.bincount(codes[valid], weights=delta, minlength=len(months))
        counts = np.bincount(codes[valid], minlength=len(months))
        denominator = counts[sampled].sum(axis=1)
        defined = denominator > 0
        match_samples = sums[sampled].sum(axis=1)[defined]/denominator[defined]
        supported = counts > 0
        monthly_mean = np.divide(sums, counts, out=np.zeros(len(months)), where=supported)
        equal_denominator = supported[sampled].sum(axis=1)
        equal_samples = monthly_mean[sampled].sum(axis=1)[defined]/equal_denominator[defined]
        results[metric] = {'metric': metric, 'n': int(valid.sum()), 'population_n': len(a),
                           'candidate_missing': int(np.isnan(a).sum()), 'control_missing': int(np.isnan(b).sum()),
                           'paired_absent': int((~valid).sum()), 'blocks': len(months), 'months': months,
                           'supported_blocks': int(supported.sum()), 'draws': draws, 'seed': seed,
                           'confidence': confidence, 'undefined_draws': int((~defined).sum()),
                           'match_weighted_delta': _mean(delta),
                           'ci': np.quantile(match_samples, [tail, 1-tail]).tolist() if defined.any() else None,
                           'equal_month_delta': _mean(monthly_mean[supported]),
                           'equal_month_ci': np.quantile(equal_samples, [tail, 1-tail]).tolist() if defined.any() else None,
                           'status': 'available' if valid.any() else 'no_paired_labels'}
    return {'confidence': confidence, 'family_confidence': family_confidence, 'family_size': size,
            'months': months, 'draws': draws, 'seed': seed, 'metrics': results,
            'resampling_unit': 'whole_series_calendar_month_blocks',
            'missing_month_policy': 'retain_in_universe; undefined_draws_reported_not_imputed'}


def paired_comparison(candidate_rows, control_rows, metric, *, confidence=1-.05/12,
                      draws=5000, seed=20261007):
    """Single-metric adapter; task names resolve to their primary NLL column."""
    column = metric if metric in candidate_rows else f'{metric}_nll'
    return paired_bootstrap(candidate_rows, control_rows, [column], draws=draws, seed=seed,
                            confidence=confidence)['metrics'][column]
