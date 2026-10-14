"""Proper scores and paired historical comparisons for standalone prop marginals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import nbinom, norm

from src.analysis.bootstrap import monthly_block_bootstrap_delta, block_ratio_interval


def lognormal_mixture_crps(y, mus, sigmas, weights):
    """Exact CRPS for a finite lognormal mixture, in the target's units."""
    y = np.asarray(y, float)
    mus, sigmas, weights = np.asarray(mus, float), np.asarray(sigmas, float), np.asarray(weights, float)
    means = np.exp(mus + .5 * sigmas ** 2)
    z = (np.log(y)[:, None] - mus) / sigmas
    absolute = y[:, None] * (2 * ndtr(z) - 1) + means * (1 - 2 * ndtr(z - sigmas))
    pair_distance = np.zeros(len(y))
    for a, wa in enumerate(weights):
        for b, wb in enumerate(weights):
            scale = np.sqrt(sigmas[:, a] ** 2 + sigmas[:, b] ** 2)
            minimum = means[:, a] * ndtr((mus[:, b] - mus[:, a] - sigmas[:, a] ** 2) / scale)
            minimum += means[:, b] * ndtr((mus[:, a] - mus[:, b] - sigmas[:, b] ** 2) / scale)
            pair_distance += wa * wb * (means[:, a] + means[:, b] - 2 * minimum)
    return absolute @ weights - .5 * pair_distance


def negative_binomial_mixture_crps(y, mus, alphas, weights):
    """Discrete CRPS with an explicit upper bound on omitted squared tails.

    A finite CDF sum is not silently treated as the full score. The omitted
    sum of squared survival probabilities is <= the excess-count expectation.
    """
    y = np.asarray(y, float)
    mus, alphas, weights = np.asarray(mus, float), np.asarray(alphas, float), np.asarray(weights, float)
    r, p = 1 / alphas, 1 / (1 + alphas * mus)
    quantiles = nbinom.ppf(1 - 1e-10, r, p)
    if not np.isfinite(quantiles).all():
        raise ValueError('Negative-binomial quantile is not finite.')
    limit = int(max(np.max(y), np.max(quantiles)))
    result = np.zeros(len(y))
    # Bound memory independently of sample count and predicted tail length.
    for start in range(0, len(y), 256):
        stop = min(start + 256, len(y))
        for low in range(0, limit + 1, 256):
            grid = np.arange(low, min(low + 256, limit + 1))
            cdf = np.zeros((stop - start, len(grid)))
            for j, weight in enumerate(weights):
                cdf += weight * nbinom.cdf(grid[None, :], r[start:stop, j, None], p[start:stop, j, None])
            result[start:stop] += np.sum((cdf - (y[start:stop, None] <= grid)) ** 2, axis=1)
    threshold = limit + 1
    excess = mus * nbinom.sf(threshold - 1, r + 1, p) - threshold * nbinom.sf(threshold, r, p)
    tail_bound = float(np.max(np.maximum(excess, 0) @ weights))
    return result, tail_bound


def _components(pred):
    if hasattr(pred, 'components'):
        return tuple(pred.components), np.asarray(pred.weights, float)
    return (pred,), np.array([1.])


def _mixture_quantile(pred, probability, task):
    comps, weights = _components(pred)
    if task == 'duration':
        mus = np.column_stack([c.duration_log_mu for c in comps])
        sigmas = np.column_stack([c.duration_log_sigma for c in comps])
        low = np.zeros(len(mus))
        high = np.max(np.exp(mus + norm.ppf(probability) * sigmas), axis=1)
        for _ in range(55):
            mid = (low + high) / 2
            cdf = ndtr((np.log(mid)[:, None] - mus) / sigmas) @ weights
            low = np.where(cdf < probability, mid, low)
            high = np.where(cdf >= probability, mid, high)
        return (low + high) / 2
    mus = np.column_stack([c.kills_mu for c in comps])
    alphas = np.column_stack([c.kills_alpha for c in comps])
    r, p = 1 / alphas, 1 / (1 + alphas * mus)
    low = np.full(len(mus), -1, dtype=np.int64)
    high = np.max(nbinom.ppf(probability, r, p), axis=1).astype(np.int64)
    while np.any(high - low > 1):
        mid = (low + high) // 2
        cdf = nbinom.cdf(mid[:, None], r, p) @ weights
        low = np.where(cdf < probability, mid, low)
        high = np.where(cdf >= probability, mid, high)
    return high.astype(float)


def score_prediction(pred, frame, fixed_lines):
    """Return aggregate metrics, row proper scores and fixed-line diagnostics."""
    from src.models.prop_distribution import distribution_nll
    comps, weights = _components(pred)
    kills = frame['kills'].to_numpy(float)
    duration = frame['duration_minutes'].to_numpy(float)
    maps = frame['maps_played'].to_numpy(float)
    best_of = frame['best_of'].to_numpy(int)
    losses = distribution_nll(pred, kills, duration, maps)
    losses['length'] = np.where(best_of > 1, losses['length'], np.nan)
    rows = frame[['match_id', 'date', 'best_of', 'tournament']].copy().reset_index(drop=True)
    metrics, lines = {}, []
    for task, labels in [('kills', kills), ('duration', duration), ('length', maps)]:
        valid = np.isfinite(labels) & ((best_of > 1) if task == 'length' else True)
        if not valid.any() or not np.isfinite(losses[task][valid]).all():
            raise ValueError(f'{task}: empty support or nonfinite observed proper score')
        rows[f'{task}_nll'] = losses[task]
        metric = {'n': int(valid.sum()), 'nll': float(np.mean(losses[task][valid])), 'absent': int((~np.isfinite(labels)).sum())}
        secondary = np.full(len(frame), np.nan)
        if task == 'kills':
            mus = np.column_stack([c.kills_mu for c in comps])
            alphas = np.column_stack([c.kills_alpha for c in comps])
            secondary[valid], bound = negative_binomial_mixture_crps(kills[valid], mus[valid], alphas[valid], weights)
            mean = mus @ weights
            metric['crps_tail_bound_max'] = bound
        elif task == 'duration':
            mus = np.column_stack([c.duration_log_mu for c in comps])
            sigmas = np.column_stack([c.duration_log_sigma for c in comps])
            secondary[valid] = lognormal_mixture_crps(duration[valid], mus[valid], sigmas[valid], weights)
            mean = np.exp(mus + .5 * sigmas ** 2) @ weights
        else:
            probs = sum(w * c.length_probs for w, c in zip(weights, comps))
            needed = best_of // 2 + 1
            cumulative = np.cumsum(probs, axis=1)
            grid = needed[:, None] + np.arange(3)[None, :]
            legal_threshold = np.arange(3)[None, :] < (best_of - needed)[:, None]
            terms = (cumulative - (maps[:, None] <= grid)) ** 2
            secondary[valid] = np.sum(np.where(legal_threshold, terms, 0), axis=1)[valid]
            mean = np.sum(probs * grid, axis=1)
            rows['length_max_probability'] = np.max(probs, axis=1)
        metric['rps' if task == 'length' else 'crps'] = float(np.mean(secondary[valid]))
        metric['mae_predictive_mean'] = float(np.mean(np.abs(mean[valid] - labels[valid])))
        rows[f'{task}_proper_secondary'] = secondary
        rows[f'{task}_mean'] = mean
        if task != 'length':
            for lo, hi, name in [(.25, .75, '50'), (.1, .9, '80')]:
                lower, upper = _mixture_quantile(pred, lo, task), _mixture_quantile(pred, hi, task)
                metric[f'central_{name}_coverage'] = float(np.mean((labels[valid] >= lower[valid]) & (labels[valid] <= upper[valid])))
                metric[f'central_{name}_mean_width'] = float(np.mean((upper - lower)[valid]))
                rows[f'{task}_q{int(lo * 100)}'] = lower
                rows[f'{task}_q{int(hi * 100)}'] = upper
        task_lines = fixed_lines['duration_minutes' if task == 'duration' else task]
        probability_fn = getattr(pred, {'kills': 'kills_over', 'duration': 'duration_over', 'length': 'maps_over'}[task])
        for line in task_lines:
            eligible = valid.copy()
            if task == 'length':
                eligible &= (line >= best_of // 2 + 1) & (line < best_of)
            p_over = probability_fn(line)
            outcome = labels > line
            rows[f'{task}_over_{line}'] = p_over
            if eligible.any():
                lines.append({'task': task, 'line': line, 'n': int(eligible.sum()), 'mean_p_over': float(np.mean(p_over[eligible])), 'observed_over_rate': float(np.mean(outcome[eligible])), 'brier': float(np.mean((p_over[eligible] - outcome[eligible]) ** 2))})
        metrics[task] = metric
    return metrics, rows, lines


def paired_comparison(candidate_rows, control_rows, task, confidence=.9833333333333333, draws=5000, seed=20261007):
    """Reuse project monthly bootstrap; expose both estimands and proxy sensitivity."""
    if not candidate_rows['match_id'].equals(control_rows['match_id']):
        raise ValueError('Paired predictions must use identical ID order.')
    delta = candidate_rows[f'{task}_nll'] - control_rows[f'{task}_nll']
    working = pd.DataFrame({'date': candidate_rows['date'], 'delta': delta}).dropna()
    estimate, _, _, samples = monthly_block_bootstrap_delta(working, 'delta', n_bootstraps=draws, random_seed=seed)
    tail = (1 - confidence) / 2
    monthly = working.assign(month=pd.to_datetime(working['date']).dt.to_period('M').astype(str)).groupby('month')['delta'].mean()
    equal_frame = pd.DataFrame({'date': monthly.index + '-01', 'delta': monthly.to_numpy()})
    equal_estimate, _, _, equal_samples = monthly_block_bootstrap_delta(equal_frame, 'delta', n_bootstraps=draws, random_seed=seed)
    valid = delta.notna()
    proxies = candidate_rows.loc[valid, 'date'].astype(str).str[:4] + ':' + candidate_rows.loc[valid, 'tournament'].fillna('UNKNOWN').astype(str)
    sensitivity = block_ratio_interval(delta[valid].to_numpy(), np.ones(int(valid.sum())), proxies.to_numpy(), draws=draws, seed=seed)
    return {'n': len(working), 'blocks': len(monthly), 'draws': draws, 'seed': seed, 'confidence': confidence, 'match_weighted_delta': estimate, 'ci': np.quantile(samples, [tail, 1-tail]).tolist(), 'equal_month_delta': equal_estimate, 'equal_month_ci': np.quantile(equal_samples, [tail, 1-tail]).tolist(), 'source_name_year_proxy_sensitivity_95pct': sensitivity}


def profile_scores(candidate_id, rows, frame):
    """Descriptive slices; no outcome-selected exclusions or per-league recipes."""
    from betting_app.ml.props.feature_extractor import normalize_league_key
    working = rows.copy()
    working['month'] = working['date'].astype(str).str[:7]
    working['competition_family'] = working['tournament'].map(normalize_league_key)
    support_columns = [c for c in frame if c.startswith('x_pace_player_') and c.endswith('_count_mean_level')]
    if support_columns:
        support = frame[support_columns].mean(axis=1).to_numpy()
        working['historical_player_pace_support'] = pd.cut(support, [-np.inf, 1, 5, 10, np.inf], labels=['<1', '1-5', '5-10', '10+']).astype(str)
    profiles = []
    for dimension in ['month', 'best_of', 'competition_family', 'historical_player_pace_support']:
        if dimension not in working:
            continue
        for value, group in working.groupby(dimension, dropna=False):
            for task in ['kills', 'duration', 'length']:
                valid = group[f'{task}_nll'].notna()
                if valid.any():
                    profiles.append({'candidate': candidate_id, 'dimension': dimension, 'value': str(value), 'task': task, 'n': int(valid.sum()), 'nll': float(group.loc[valid, f'{task}_nll'].mean()), 'proper_secondary': float(group.loc[valid, f'{task}_proper_secondary'].mean())})
    for task, bounds in [('kills', [0, 10, 20, 30, np.inf]), ('duration', [0, 5, 10, 15, np.inf])]:
        widths = working[f'{task}_q90'] - working[f'{task}_q10']
        bins = pd.cut(widths, bounds).astype(str)
        for value in sorted(bins.unique()):
            subset = working.loc[bins == value]
            profiles.append({'candidate': candidate_id, 'dimension': 'predictive_80pct_width', 'value': value, 'task': task, 'n': int(subset[f'{task}_nll'].notna().sum()), 'nll': float(subset[f'{task}_nll'].mean()), 'proper_secondary': float(subset[f'{task}_proper_secondary'].mean())})
    confidence_bins = pd.cut(working['length_max_probability'], [0, .5, .7, .9, 1.0000001]).astype(str)
    for value in sorted(confidence_bins.unique()):
        subset = working.loc[(confidence_bins == value) & working['length_nll'].notna()]
        if len(subset):
            profiles.append({'candidate': candidate_id, 'dimension': 'length_max_probability', 'value': value, 'task': 'length', 'n': len(subset), 'nll': float(subset['length_nll'].mean()), 'proper_secondary': float(subset['length_proper_secondary'].mean())})
    return profiles
