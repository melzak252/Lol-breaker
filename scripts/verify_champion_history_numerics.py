#!/usr/bin/env python3
"""Independent saved-weight replay and numerical audit; never fit a network.

Candidate classes are loaded solely to decode joblib. No candidate prediction,
forward, preprocessing, ensemble, distribution or evaluation method is called.
Tolerances are a separate predeclared, run-owned input, not inferred from errors.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp, ndtr
from scipy.stats import nbinom, norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = 'champion_history_20261007_01'
SEEDS = (20261007, 20261008, 20261009)
FIELDS = ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma')
STATS = ('nll', 'crps', 'mae', 'coverage_50', 'width_50', 'coverage_80', 'width_80')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


class Audit:
    def __init__(self, tolerances):
        self.tolerances = tolerances
        self.checks = []
        self.failures = []

    def truth(self, name, condition, detail=None):
        item = {'name': name, 'passed': bool(condition), 'detail': detail}
        self.checks.append(item)
        if not condition:
            self.failures.append(item)

    def compare(self, name, actual, expected, kind='metric'):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        same_shape = a.shape == b.shape
        item = {'name': name, 'kind': kind, 'shape': list(a.shape), 'values': a.size}
        if same_shape:
            missing = np.isnan(a) == np.isnan(b)
            finite = np.isfinite(a) & np.isfinite(b)
            difference = np.abs(a[finite] - b[finite])
            atol, rtol = (self.tolerances[f'{kind}_{key}'] for key in ('atol', 'rtol'))
            matches = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
            item.update(max_absolute_difference=float(difference.max()) if difference.size else 0.,
                        max_relative_difference=float(np.max(difference / np.maximum(np.abs(b[finite]), 1e-300))) if difference.size else 0.,
                        missing_mask_mismatches=int((~missing).sum()),
                        mismatches=int((~matches).sum()), passed=bool(matches.all()))
            if not matches.all():
                indices = np.argwhere(~matches)[:12]
                item['retained_failures'] = [{'index': x.tolist(), 'independent': clean(a[tuple(x)]),
                                              'exported': clean(b[tuple(x)])} for x in indices]
        else:
            item.update(passed=False, expected_shape=list(b.shape))
        self.checks.append(item)
        if not item['passed']:
            self.failures.append(item)


def preprocess(arrays, train, batch_size):
    """TRAIN sufficient statistics, with no fitted preprocessor calls."""
    result = {}
    for key in ('history', 'players', 'context'):
        width = arrays[key].shape[-1]
        sums = np.zeros(width)
        squares = np.zeros(width)
        counts = np.zeros(width, np.int64)
        for begin in range(0, len(train), batch_size):
            ix = train[begin:begin + batch_size]
            values = arrays[key][ix].astype(np.float64)
            valid = np.isfinite(values)
            if key == 'history':
                valid &= arrays['mask'][ix][..., None]
            observed = np.where(valid, values, 0).reshape(-1, width)
            sums += observed.sum(axis=0)
            squares += np.square(observed).sum(axis=0)
            counts += valid.reshape(-1, width).sum(axis=0)
        mean = sums / np.maximum(counts, 1)
        scale = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - mean ** 2, 0))
        scale[scale < 1e-6] = 1
        result[key] = (mean, scale, counts)
    return result


def replay(model, arrays, indices, stats):
    """Explicit tensor arithmetic on saved tensors, not candidate modules."""
    state = model.state_dict
    size = model.config['batch_size']
    outputs = {'score': []}
    if model.multitask:
        outputs.update({name: [] for name in FIELDS})

    def linear(value, key):
        return torch.nn.functional.linear(value, state[key + '.weight'], state[key + '.bias'])

    def head(value, key):
        return linear(torch.tanh(linear(value, key + '.0')), key + '.2')

    with torch.inference_mode():
        for begin in range(0, len(indices), size):
            ix = indices[begin:begin + size]
            x = {}
            for key, (mean, scale, count) in stats.items():
                raw = arrays[key][ix]
                valid = np.isfinite(raw) & (count > 0)
                if key == 'history':
                    valid &= arrays['mask'][ix][..., None]
                # The model's two float32 rounding points are part of its API.
                normalized = (raw - mean).astype(np.float32)
                normalized = (normalized / scale).astype(np.float32)
                normalized[~valid] = 0
                x[key] = torch.from_numpy(np.concatenate((normalized, valid.astype(np.float32)), axis=-1))
            for key in ('champions', 'patches', 'competitions'):
                codes = arrays[key][ix]
                x[key] = torch.from_numpy(np.where(codes < model.vocab_sizes[key], codes, 1).astype(np.int64))
            mask = torch.from_numpy(arrays['mask'][ix])
            roles = torch.from_numpy(arrays['roles'][ix])
            if model.champion_enabled:
                champion = state['champion.weight'][x['champions']]
                status = (x['champions'].clamp_max(2)[..., None] == torch.arange(3)).float()
            else:
                champion = torch.zeros(mask.shape + (state['champion.weight'].shape[1],))
                status = torch.zeros(mask.shape + (3,))
            performance = torch.tanh(linear(x['history'], 'performance'))
            interaction = performance * (1 + torch.tanh(linear(champion, 'champion_performance')))
            token = torch.tanh(linear(torch.cat((interaction, champion, state['role.weight'][roles], status,
                                                state['patch.weight'][x['patches']],
                                                state['competition.weight'][x['competitions']]), dim=-1), 'token') + state['age'])
            attention = linear(token, 'attention').squeeze(-1).masked_fill(~mask, -torch.inf)
            supported = mask.any(dim=-1)
            attention = torch.where(supported[..., None], attention, torch.zeros_like(attention))
            weights = torch.softmax(attention, dim=-1) * mask
            history = (token * weights[..., None]).sum(dim=-2) + (~supported)[..., None] * state['cold']
            player = torch.cat((history, x['players']), dim=-1)
            role_probs = torch.from_numpy(arrays['role_probs'][ix].astype(np.float32))

            def roster(key):
                values = torch.tanh(linear(player, 'player_' + key + '.0'))
                weighted = torch.einsum('ntpr,ntpw->ntrw', role_probs, values)
                weighted /= role_probs.sum(dim=-2).clamp_min(1e-6)[..., None]
                return torch.tanh(linear(torch.cat((values.mean(dim=-2), weighted.flatten(-2)), dim=-1), 'roster_' + key + '.0'))

            strength, style = roster('strength'), roster('style')
            oriented = torch.stack((torch.cat((strength[:, 0], strength[:, 1], style[:, 0], style[:, 1], x['context']), -1),
                                    torch.cat((strength[:, 1], strength[:, 0], style[:, 1], style[:, 0], x['context']), -1)), 1)
            logits = head(oriented, 'score_head').flatten(1)
            legal = torch.tensor([0, 1, 2, 0, 1, 2])[None, :] <= torch.from_numpy(arrays['best_of'][ix] // 2)[:, None]
            outputs['score'].append(logits.masked_fill(~legal, -torch.inf).double().softmax(-1).numpy())
            if model.multitask:
                symmetric = torch.cat((strength.sum(1), (strength[:, 0] - strength[:, 1]).abs(), style.sum(1),
                                       (style[:, 0] - style[:, 1]).abs(), x['context']), -1)
                prop = head(torch.cat((symmetric[:, None, :].expand(-1, 5, -1),
                                       state['map_embedding.weight'][None, :, :].expand(len(ix), -1, -1)), -1), 'prop_head')
                bounds = ((-12, 12), (-12, 5), (-12, 12), (-7, 2))
                for j, (name, bound) in enumerate(zip(FIELDS, bounds)):
                    value = prop[..., j].clamp(*bound)
                    if j != 2:
                        value = value.exp()
                    outputs[name].append(value.double().numpy())
    return {name: np.concatenate(values) for name, values in outputs.items()}


def temper(probabilities, scalar):
    with np.errstate(divide='ignore'):
        log_probability = np.log(probabilities) / scalar
    return np.exp(log_probability - logsumexp(log_probability, axis=-1, keepdims=True))


def density(task, target, mu, spread):
    if task == 'kills':
        shape = 1 / spread
        product = spread * mu
        return (gammaln(target + shape) - gammaln(shape) - gammaln(target + 1)
                - shape * np.log1p(product) + target * (np.log(product) - np.log1p(product)))
    log_target = np.log(target)
    return -log_target - np.log(spread) - .5 * np.log(2 * np.pi) - .5 * ((log_target - mu) / spread) ** 2


def scalar_optimum(objective):
    bounds = (math.log(.25), math.log(4))
    def penalized(value):
        return objective(math.exp(value)) + .001 * value ** 2
    result = minimize_scalar(penalized, bounds=bounds, method='bounded', options={'xatol': 1e-7, 'maxiter': 300})
    options = (result.x, *bounds, 0.)
    chosen = min(options, key=penalized)
    return {'value': math.exp(chosen), 'nll': objective(math.exp(chosen)),
            'penalized_objective': penalized(chosen), 'success': bool(result.success), 'nfev': result.nfev}


def calibration(raw, series, maps, saved_receipt):
    valid = series.score_index.notna().to_numpy()
    target = series.loc[valid, 'score_index'].to_numpy(int)
    rows = np.flatnonzero(valid)
    def score_objective(value):
        probabilities = np.mean([temper(c['score'], value) for c in raw], axis=0)
        return float(-np.log(probabilities[rows, target]).mean())
    result = {'score_temperature': scalar_optimum(score_objective)}
    value = saved_receipt['score_temperature']['value']
    result['score_temperature']['saved_value_nll'] = score_objective(value)
    result['score_temperature']['saved_value_penalized_objective'] = score_objective(value) + .001 * math.log(value) ** 2
    if 'kills_mu' in raw[0]:
        location = pd.Series(np.arange(len(series)), index=series.row_index)
        for task, label, mean_key, spread_key in (('kills', 'kills', 'kills_mu', 'kills_alpha'),
                                                  ('duration', 'duration_minutes', 'duration_log_mu', 'duration_log_sigma')):
            observed = maps.loc[maps[label].notna()]
            r = observed.row_index.map(location).to_numpy(int)
            m = observed.map_index.to_numpy(int) - 1
            y = observed[label].to_numpy(float)
            mu = np.stack([c[mean_key][r, m] for c in raw])
            spread = np.stack([c[spread_key][r, m] for c in raw])
            _, inverse, counts = np.unique(r, return_inverse=True, return_counts=True)
            def objective(value):
                logs = density(task, y[None, :], mu, spread * value)
                nll = math.log(len(raw)) - logsumexp(logs, axis=0)
                return float(np.mean(np.bincount(inverse, weights=nll) / counts))
            name = 'kill_scale' if task == 'kills' else 'duration_scale'
            result[name] = scalar_optimum(objective)
            value = saved_receipt[name]['value']
            result[name]['saved_value_nll'] = objective(value)
            result[name]['saved_value_penalized_objective'] = objective(value) + .001 * math.log(value) ** 2
            result[name]['observed_maps'] = len(y)
            result[name]['observed_series'] = len(counts)
    return result


def quantile(task, mu, spread, weights, probability):
    if task == 'duration':
        low = np.zeros(len(mu))
        high = np.max(np.exp(mu + norm.ppf(probability) * spread), axis=1)
        for _ in range(55):
            mid = (low + high) / 2
            cdf = ndtr((np.log(mid)[:, None] - mu) / spread) @ weights
            low, high = np.where(cdf < probability, mid, low), np.where(cdf >= probability, mid, high)
        return (low + high) / 2
    r, p = 1 / spread, 1 / (1 + spread * mu)
    low = np.full(len(mu), -1, np.int64)
    high = np.max(nbinom.ppf(probability, r, p), axis=1).astype(np.int64)
    while np.any(high - low > 1):
        mid = (low + high) // 2
        cdf = nbinom.cdf(mid[:, None], r, p) @ weights
        low, high = np.where(cdf < probability, mid, low), np.where(cdf >= probability, mid, high)
    return high.astype(float)


def crps(task, y, mu, spread, weights):
    if task == 'duration':
        means = np.exp(mu + spread ** 2 / 2)
        z = (np.log(y)[:, None] - mu) / spread
        first = (y[:, None] * (2 * ndtr(z) - 1) + means * (1 - 2 * ndtr(z - spread))) @ weights
        # E|min(X_i,X_j)| from independently derived truncated moments.
        minimum = np.zeros(len(y))
        for i, wi in enumerate(weights):
            for j, wj in enumerate(weights):
                scale = np.hypot(spread[:, i], spread[:, j])
                minimum += wi * wj * (means[:, i] * ndtr((mu[:, j] - mu[:, i] - spread[:, i] ** 2) / scale)
                                         + means[:, j] * ndtr((mu[:, i] - mu[:, j] - spread[:, j] ** 2) / scale))
        return first - means @ weights + minimum, 0.
    r, p = 1 / spread, 1 / (1 + spread * mu)
    limit = int(max(np.max(y), np.max(nbinom.ppf(1 - 1e-10, r, p))))
    result = np.zeros(len(y))
    for begin in range(0, len(y), 128):
        end = min(begin + 128, len(y))
        for low in range(0, limit + 1, 128):
            grid = np.arange(low, min(low + 128, limit + 1))
            mixture = sum(w * nbinom.cdf(grid[None, :], r[begin:end, j, None], p[begin:end, j, None])
                          for j, w in enumerate(weights))
            result[begin:end] += np.square(mixture - (y[begin:end, None] <= grid)).sum(axis=1)
    threshold = limit + 1
    excess = mu * nbinom.sf(threshold - 1, r + 1, p) - threshold * nbinom.sf(threshold, r, p)
    return result, float(np.max(np.maximum(excess, 0) @ weights))


def score_rows(series, maps, score, parameters=None, weights=None, length_probs=None, first_only=False):
    ss, ms = series.copy().reset_index(drop=True), maps.copy().reset_index(drop=True)
    positions = pd.Series(np.arange(len(ss)), index=ss.row_index)
    map_positions = ms.row_index.map(positions).to_numpy(int)
    for column in ss:
        if column not in ms:
            ms[column] = ss[column].to_numpy()[map_positions]
    bo = ss.best_of.to_numpy(int)
    n = len(ss)
    terminal = ss.score_index.to_numpy(float)
    length = ss.maps_played.to_numpy(float) if 'maps_played' in ss else np.full(n, np.nan)
    known = np.isfinite(terminal)
    length = np.where(np.isnan(length), bo // 2 + 1 + terminal % 3, length)
    grid = (bo // 2 + 1)[:, None] + np.arange(3)
    if score is not None:
        p = score[:, :3].sum(axis=1)
        lp = score[:, :3] + score[:, 3:]
        winner = ss.winner.to_numpy(float)
        ss['p_win'] = p
        ss['winner_nll'] = -np.log(np.where(winner == 1, p, 1 - p))
        ss.loc[np.isnan(winner), 'winner_nll'] = np.nan
        ss['winner_brier'] = (p - winner) ** 2
        ss['score_nll'] = np.nan
        ss.loc[known, 'score_nll'] = -np.log(score[np.flatnonzero(known), terminal[known].astype(int)])
    else:
        lp = length_probs
        for column in ('winner_nll', 'winner_brier', 'score_nll'):
            ss[column] = np.nan
    valid = np.isfinite(length) & (bo > 1)
    ss['length_nll'], ss['length_rps'] = np.nan, np.nan
    ss.loc[valid, 'length_nll'] = -np.log(lp[np.flatnonzero(valid), (length[valid] - bo[valid] // 2 - 1).astype(int)])
    rps = np.where(np.arange(3)[None, :] < (bo - bo // 2 - 1)[:, None],
                   (np.cumsum(lp, axis=1) - (length[:, None] <= grid)) ** 2, 0).sum(axis=1)
    ss.loc[valid, 'length_rps'] = rps[valid]
    ss['length_proper_secondary'] = ss['length_rps']
    play = np.column_stack([(lp * (grid >= m)).sum(axis=1) for m in range(1, 6)])
    play_valid = np.isfinite(length)[:, None] & (np.arange(1, 6)[None, :] <= bo[:, None])
    play_loss = np.where(play_valid, (play - (np.arange(1, 6)[None, :] <= length[:, None])) ** 2, np.nan)
    ss['map_play_brier'] = np.where(np.isfinite(length), np.nansum(play_loss, axis=1) / bo, np.nan)
    lookup = pd.Series(np.arange(n), index=ss.row_index)
    positions = ms.row_index.map(lookup).to_numpy(int)
    map_ids = ms.map_index.to_numpy(int) - 1
    ms['play_probability'] = play[positions, map_ids]
    ms['map_play_brier'] = (ms.play_probability - 1) ** 2
    ms['map_play_outcome'] = 1
    tails = []
    for task, label, mean_key, spread_key in (('kills', 'kills', 'kills_mu', 'kills_alpha'),
                                              ('duration', 'duration_minutes', 'duration_log_mu', 'duration_log_sigma')):
        for suffix in (*STATS, 'mean', 'q10', 'q25', 'q75', 'q90'):
            ms[f'{task}_{suffix}'] = np.nan
        if parameters is not None:
            for m in np.unique(map_ids):
                selected = np.flatnonzero(map_ids == m)
                if first_only and m != 0:
                    continue
                rows = positions[selected]
                mu = np.column_stack([c[mean_key][rows, m] for c in parameters])
                spread = np.column_stack([c[spread_key][rows, m] for c in parameters])
                y = ms.loc[selected, label].to_numpy(float)
                observed = np.isfinite(y)
                mean = mu @ weights if task == 'kills' else np.exp(mu + spread ** 2 / 2) @ weights
                ms.loc[selected, f'{task}_mean'] = mean
                ms.loc[selected[observed], f'{task}_mae'] = np.abs(mean[observed] - y[observed])
                if observed.any():
                    log_p = density(task, y[observed, None], mu[observed], spread[observed])
                    ms.loc[selected[observed], f'{task}_nll'] = -logsumexp(log_p + np.log(weights), axis=1)
                    secondary, tail = crps(task, y[observed], mu[observed], spread[observed], weights)
                    ms.loc[selected[observed], f'{task}_crps'] = secondary
                    if task == 'kills':
                        tails.append(tail)
                for lower, upper, level in ((.25, .75, 50), (.1, .9, 80)):
                    lo, hi = (quantile(task, mu, spread, weights, prob) for prob in (lower, upper))
                    ms.loc[selected, f'{task}_q{int(lower * 100)}'] = lo
                    ms.loc[selected, f'{task}_q{int(upper * 100)}'] = hi
                    ms.loc[selected[observed], f'{task}_coverage_{level}'] = ((y >= lo) & (y <= hi))[observed].astype(float)
                    ms.loc[selected[observed], f'{task}_width_{level}'] = (hi - lo)[observed]
        ms[f'{task}_proper_secondary'] = ms[f'{task}_crps']
        ms[f'{task}_label_observed'] = ms[label].notna()
        for suffix in STATS:
            grouped = ms.groupby('row_index')[f'{task}_{suffix}'].mean()
            ss[f'{task}_{suffix}'] = ss.row_index.map(grouped)
        for count_suffix, observed in (('observed_maps', ms[label].notna()), ('scored_maps', ms[f'{task}_nll'].notna())):
            count = observed.groupby(ms.row_index).sum()
            ss[f'{task}_{count_suffix}'] = ss.row_index.map(count).fillna(0).astype(int)
        ss[f'{task}_proper_secondary'] = ss[f'{task}_crps']
    extra = {'length': length, 'play_losses': play_loss, 'play_valid': play_valid, 'tails': tails}
    return ss, ms, extra


def coherent_reference(probability, best_of):
    """Vectorized Markov path law from frozen C0 margins, independently coded."""
    result = np.zeros((len(probability), 6))
    for bo, slope in ((1, 1.), (3, 1.34), (5, 1.62)):
        selected = np.flatnonzero(best_of == bo)
        p = probability[selected]
        needed = bo // 2 + 1
        terminal = np.zeros((len(selected), 6))
        terminal[p == 1, 0] = 1
        terminal[p == 0, 3] = 1
        interior = (p > 0) & (p < 1)
        values = p[interior]
        logit = np.log(values / (1 - values)) / slope
        table = np.zeros((needed + 1, needed + 1, len(values)))
        table[0, 0] = 1
        for a in range(needed):
            for b in range(needed):
                next_probability = 1 / (1 + np.exp(-(logit + .35 * (a - b))))
                table[a + 1, b] += table[a, b] * next_probability
                table[a, b + 1] += table[a, b] * (1 - next_probability)
        a_mass, b_mass = table[needed, :needed].sum(0), table[:needed, needed].sum(0)
        for loser in range(needed):
            terminal[interior, loser] = values * table[needed, loser] / a_mass
            terminal[interior, 3 + loser] = (1 - values) * table[loser, needed] / b_mass
        result[selected] = terminal
    return result


def bootstrap(candidate, control, metric, confidence, seed, draws):
    months = pd.to_datetime(candidate.date).dt.to_period('M').astype(str)
    universe = sorted(months.unique())
    codes = pd.Categorical(months, categories=universe).codes
    a, b = candidate[metric].to_numpy(float), control[metric].to_numpy(float)
    valid = np.isfinite(a) & np.isfinite(b)
    delta = a[valid] - b[valid]
    sums = np.bincount(codes[valid], weights=delta, minlength=len(universe))
    counts = np.bincount(codes[valid], minlength=len(universe))
    sampled = np.random.default_rng(seed).integers(0, len(universe), size=(draws, len(universe)))
    denominator = counts[sampled].sum(axis=1)
    defined = denominator > 0
    match = sums[sampled].sum(axis=1)[defined] / denominator[defined]
    supported = counts > 0
    means = np.divide(sums, counts, out=np.zeros(len(universe)), where=supported)
    equal = means[sampled].sum(axis=1)[defined] / supported[sampled].sum(axis=1)[defined]
    tail = (1 - confidence) / 2
    return {'n': int(valid.sum()), 'population_n': len(a), 'candidate_missing': int(np.isnan(a).sum()),
            'control_missing': int(np.isnan(b).sum()), 'paired_absent': int((~valid).sum()), 'blocks': len(universe),
            'supported_blocks': int(supported.sum()), 'months': universe, 'draws': draws, 'seed': seed,
            'confidence': confidence, 'undefined_draws': int((~defined).sum()),
            'match_weighted_delta': float(delta.mean()) if len(delta) else None,
            'equal_month_delta': float(means[supported].mean()) if supported.any() else None,
            'ci': np.quantile(match, [tail, 1 - tail]).tolist() if len(match) else None,
            'equal_month_ci': np.quantile(equal, [tail, 1 - tail]).tolist() if len(equal) else None}


def compare_frame(audit, name, calculated, exported):
    audit.truth(name + ':row alignment', np.array_equal(calculated.row_index, exported.row_index))
    for column in ('match_id', 'game_id', 'map_index', 'date'):
        if column in calculated and column in exported:
            audit.truth(name + ':identity alignment:' + column,
                        np.array_equal(calculated[column].astype(str), exported[column].astype(str)))
    for column in calculated:
        if column in exported and (column.endswith(('_nll', '_brier', '_rps')) or
                any(column.endswith('_' + x) for x in (*STATS, 'mean', 'q10', 'q25', 'q75', 'q90', 'observed_maps', 'scored_maps', 'proper_secondary', 'label_observed')) or
                column in ('p_win', 'play_probability', 'map_play_outcome')):
            audit.compare(name + ':' + column, calculated[column], exported[column])


def compare_metrics(audit, name, ss, ms, extra, exported):
    def average(values):
        x = np.asarray(values, float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else None
    for task, columns in (('winner', {'nll': 'winner_nll', 'brier': 'winner_brier'}),
                          ('score', {'nll': 'score_nll'}), ('length', {'nll': 'length_nll', 'rps': 'length_rps'})):
        for key, column in columns.items():
            if key in exported[task]:
                audit.compare(f'{name}:aggregate:{task}:{key}', average(ss[column]), exported[task][key])
        audit.compare(f'{name}:aggregate:{task}:n', int(ss[columns[next(iter(columns))]].notna().sum()), exported[task]['n'], 'integer')
        absence = int(ss.winner.isna().sum()) if task == 'winner' else (int(ss.score_index.isna().sum()) if task == 'score' else int(np.isnan(extra['length']).sum()))
        if 'absent_labels' in exported[task]:
            audit.compare(f'{name}:aggregate:{task}:absent labels', absence, exported[task]['absent_labels'], 'integer')
        if task == 'length':
            audit.compare(name + ':BO1 length exclusion count', int((ss.best_of == 1).sum()), exported[task]['excluded_bo1'], 'integer')
    for task in ('kills', 'duration'):
        for suffix in STATS:
            for estimand, frame in (('map_weighted', ms), ('series_weighted', ss)):
                audit.compare(f'{name}:aggregate:{task}:{estimand}:{suffix}', average(frame[f'{task}_{suffix}']), exported[task][estimand][suffix])
                aliases = {'mae': ('mae_predictive_mean',), 'coverage_50': ('central_50_coverage',),
                           'coverage_80': ('central_80_coverage',), 'width_50': ('central_50_mean_width',),
                           'width_80': ('central_80_mean_width',)}.get(suffix, ())
                for alias in aliases:
                    audit.compare(f'{name}:aggregate:{task}:{estimand}:{alias}', average(frame[f'{task}_{suffix}']), exported[task][estimand][alias])
            audit.compare(f'{name}:aggregate:{task}:map top-level:{suffix}', average(ms[f'{task}_{suffix}']), exported[task][suffix])
        audit.compare(f'{name}:aggregate:{task}:n', int(ms[f'{task}_nll'].notna().sum()), exported[task]['n'], 'integer')
        audit.compare(f'{name}:aggregate:{task}:series_n', int(ss[f'{task}_nll'].notna().sum()), exported[task]['series_n'], 'integer')
        label = 'kills' if task == 'kills' else 'duration_minutes'
        observed = int(ms[label].notna().sum())
        scored = int(ms[f'{task}_nll'].notna().sum())
        for field, value in (('observed_labels', observed), ('absent_labels', len(ms) - observed), ('unscored_observed_labels', observed - scored)):
            audit.compare(f'{name}:aggregate:{task}:{field}', value, exported[task][field], 'integer')
        if 'unsupported_later_maps' in exported[task]:
            audit.compare(f'{name}:aggregate:{task}:unsupported later maps', int((ms.map_index > 1).sum()), exported[task]['unsupported_later_maps'], 'integer')
    audit.compare(name + ':mapplay aggregate', average(extra['play_losses']), exported['map_play']['brier'])
    audit.compare(name + ':mapplay series aggregate', average(ss.map_play_brier), exported['map_play']['series_weighted_brier'])
    audit.compare(name + ':mapplay n', int(extra['play_valid'].sum()), exported['map_play']['n'], 'integer')
    audit.compare(name + ':mapplay absent series labels', int(np.isnan(extra['length']).sum()), exported['map_play']['absent_series_labels'], 'integer')
    if 'population' in exported:
        audit.compare(name + ':population series', len(ss), exported['population']['series'], 'integer')
        audit.compare(name + ':population recorded maps', len(ms), exported['population']['recorded_maps'], 'integer')
    for item in exported['map_play']['by_map_index']:
        m = item['map_index'] - 1
        audit.compare(f'{name}:mapplay:{m + 1}:n', int(extra['play_valid'][:, m].sum()), item['n'], 'integer')
        audit.compare(f'{name}:mapplay:{m + 1}:brier', average(extra['play_losses'][:, m]), item['brier'])
    if 'calibration_ece' in exported['winner']:
        valid = ss.winner.notna()
        p, y = ss.loc[valid, 'p_win'].to_numpy(), ss.loc[valid, 'winner'].to_numpy()
        positive, negative = int((y == 1).sum()), int((y == 0).sum())
        auc = float((rankdata(p)[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)) if positive and negative else None
        audit.compare(name + ':winner auc', auc, exported['winner']['auc'])
        assignments = np.minimum((p * 10).astype(int), 9)
        ece = 0.
        for index, item in enumerate(exported['winner']['calibration_bins']):
            chosen = assignments == index
            forecast, outcome = average(p[chosen]), average(y[chosen])
            count = int(chosen.sum())
            if count:
                ece += count / len(y) * abs(forecast - outcome)
            audit.compare(f'{name}:calibrationbin:{index}:count', count, item['n'], 'integer')
            audit.compare(f'{name}:calibrationbin:{index}:forecast', forecast, item['mean_prediction'])
            audit.compare(f'{name}:calibrationbin:{index}:outcome', outcome, item['observed_rate'])
        audit.compare(name + ':winner ece', ece, exported['winner']['calibration_ece'])
    if extra['tails']:
        audit.truth(name + ':NB CRPS tail bound', max(extra['tails']) <= audit.tolerances['nb_crps_tail_max'], max(extra['tails']))
        audit.compare(name + ':NB CRPS exported tail bound', max(extra['tails']), exported['kills']['crps_tail_bound_max'])


def initial_state(model, arrays, train, maps):
    """Replay public torch initialization draw order, without a candidate class."""
    torch.manual_seed(model.seed)
    h, p, c = model.dimensions
    w, e = model.config['width'], model.config['embedding_dim']
    modules = {}
    for key, dimension in (('champion', e), ('patch', 4), ('competition', 4)):
        vocab_key = {'champion': 'champions', 'patch': 'patches', 'competition': 'competitions'}[key]
        modules[key] = torch.nn.Embedding(model.vocab_sizes[vocab_key], dimension)
    modules['role'] = torch.nn.Embedding(6, e)
    dimensions = (('performance', 2 * h, w), ('champion_performance', e, w),
                  ('token', w + 2 * e + 3 + 8, w), ('attention', w, 1),
                  ('player_strength.0', w + 2 * p, w), ('player_style.0', w + 2 * p, w),
                  ('roster_strength.0', 7 * w, w), ('roster_style.0', 7 * w, w),
                  ('score_head.0', 4 * w + 2 * c, w), ('score_head.2', w, 3))
    for key, inputs, outputs in dimensions:
        modules[key] = torch.nn.Linear(inputs, outputs)
    modules['map_embedding'] = torch.nn.Embedding(5, e)
    modules['prop_head.0'] = torch.nn.Linear(4 * w + 2 * c + e, w)
    modules['prop_head.2'] = torch.nn.Linear(w, 4)
    training_maps = maps[maps.row_index.isin(train)].copy()
    training_maps = training_maps.sort_values(['row_index', 'map_index'])
    # Match the TRAIN matrix's row-major reduction, including explicit NaNs.
    location = pd.Series(np.arange(len(train)), index=train)
    kills = np.full((len(train), 5), np.nan)
    duration = np.full((len(train), 5), np.nan)
    rows = training_maps.row_index.map(location).to_numpy(int)
    columns = training_maps.map_index.to_numpy(int) - 1
    kills[rows, columns] = training_maps.kills.to_numpy(float)
    duration[rows, columns] = training_maps.duration_minutes.to_numpy(float)
    with torch.no_grad():
        modules['prop_head.2'].bias.copy_(torch.tensor([math.log(max(float(np.nanmean(kills)), 1e-6)),
            math.log(.1), float(np.nanmean(np.log(duration))), math.log(.25)]))
    tensors = {'age': torch.zeros(16, w), 'cold': torch.zeros(w),
               'map_ids': torch.arange(5), 'loser_scores': torch.tensor([0, 1, 2, 0, 1, 2])}
    for name, module in modules.items():
        for field, value in module.state_dict().items():
            tensors[name + '.' + field] = value
    checksum = hashlib.sha256()
    for name in model.state_dict:
        checksum.update(name.encode())
        checksum.update(tensors[name].numpy().tobytes())
    return checksum.hexdigest(), tensors


def vocabulary_support(arrays, splits, sizes, batch_size):
    """Count only unmasked historical tokens; padded codes are not support."""
    result = {}
    for key in ('champions', 'patches', 'competitions'):
        counts = {}
        for split, indices in splits.items():
            count = np.zeros(sizes[key], np.int64)
            for begin in range(0, len(indices), batch_size):
                ix = indices[begin:begin + batch_size]
                codes = arrays[key][ix][arrays['mask'][ix]]
                codes = np.where(codes < sizes[key], codes, 1)
                count += np.bincount(codes, minlength=sizes[key])
            counts[split] = count
        result[key] = counts
    return result


def embedding_evidence(model, initial, support):
    """Bounded saved-state diagnostics, not a claim of training causality."""
    names = [name for name, _ in model.network.named_parameters()]
    ids = [parameter for group in model.optimizer_state['param_groups'] for parameter in group['params']]
    moments = dict(zip(names, (model.optimizer_state['state'].get(parameter) for parameter in ids)))
    result = {}

    def evidence(key, selector):
        before = initial[key][selector].detach().numpy()
        after = model.state_dict[key][selector].detach().numpy()
        state = moments.get(key)
        detail = {'initial_values': before.tolist(), 'saved_values': after.tolist(),
                  'max_absolute_change': float(np.max(np.abs(after - before))),
                  'optimizer_state_present': state is not None}
        if state is not None:
            detail.update(optimizer_steps=int(state['step'].item()),
                          exp_avg_max_absolute=float(state['exp_avg'][selector].abs().max()),
                          exp_avg_sq_max_absolute=float(state['exp_avg_sq'][selector].abs().max()))
        return detail

    for key, weight in (('champions', 'champion.weight'), ('patches', 'patch.weight'),
                        ('competitions', 'competition.weight')):
        counts = support[key]
        unsupported = np.flatnonzero(counts['TRAIN'] == 0)
        result[key] = {'vocabulary_size': len(counts['TRAIN']), 'zero_valid_TRAIN_codes_count': len(unsupported),
                       'zero_valid_TRAIN_codes_first_32': unsupported[:32].tolist(),
                       'unsupported_TRAIN_token_counts_in_CAL_ASSESS': {
                           split: int(counts[split][unsupported].sum()) for split in ('CAL', 'ASSESS')},
                       'valid_tokens_by_split': {split: int(count.sum()) for split, count in counts.items()},
                       'rows': {str(code): {'valid_token_counts': {split: int(count[code]) for split, count in counts.items()},
                                            'informative_channel_enabled': key != 'champions' or model.champion_enabled,
                                            **evidence(weight, code)} for code in (0, 1)}}
    # Token concatenation: performance(w), champion(e), role(e), status(3).
    unknown_column = model.config['width'] + 2 * model.config['embedding_dim'] + 1
    result['champion_UNK_status_token_column'] = {
        'column': unknown_column, 'valid_TRAIN_UNK_tokens': int(support['champions']['TRAIN'][1]),
        'informative_channel_enabled': model.champion_enabled,
        **evidence('token.weight', (slice(None), unknown_column))}
    result['interpretation_limit'] = (
        'Zero valid TRAIN frequency means no informative TRAIN-token support for that row/status; '
        'finite initialized or weight-decayed saved values are not evidence of data-learning. '
        'Final AdamW moments/parameter changes are evidence snapshots, not a reconstruction of every update. '
        'OOV representation support is a limitation, not an established cause of a negative result.')
    return result


def compare_profiles(audit, candidate, ss, ms, exported):
    """Recompute every profile's numerical aggregation from raw labels/PMFs."""
    aliases = {
        'patch': ('target_patch', 'patch'), 'rule': ('draft_mode', 'rule_status', 'rule_profile', 'rule_known', 'rule'),
        'rule_unknown': ('rule_unknown',), 'history': ('history_profile', 'history_support_profile', 'history_support', 'history_count'),
        'inactivity': ('inactivity_profile', 'inactivity_days', 'inactivity'),
        'new_roster': ('roster_profile', 'new_roster_profile', 'new_roster', 'roster_new'),
        'stat_mask': ('stat_mask_profile', 'stat_mask', 'stat_missingness'),
        'unknown_support': ('unknown_support_profile', 'unknown_champion_fraction', 'unknown_support', 'unknown_player_count'),
        'missing_champion_support': ('missing_champion_history',), 'unseen_champion_support': ('unseen_champion_history',)}
    records = []
    for scope, original in (('series', ss), ('played_map', ms)):
        frame = original.copy()
        frame['time'] = pd.to_datetime(frame.date).dt.to_period('M').astype(str)
        dimensions = ['best_of', 'time'] + (['map_index'] if scope == 'played_map' else [])
        for name, choices in aliases.items():
            field = next((choice for choice in choices if choice in frame), None)
            frame[name] = frame[field].fillna('UNKNOWN').astype(str) if field else 'UNKNOWN'
            dimensions.append(name)
        tasks = ['kills', 'duration'] if scope == 'played_map' else ['winner', 'score', 'length', 'kills', 'duration']
        tasks.append('map_play')
        for dimension in dimensions:
            for value, group in frame.groupby(dimension, sort=True, dropna=False):
                for task in tasks:
                    column = 'map_play_brier' if task == 'map_play' else task + '_nll'
                    count = int(group[column].notna().sum())
                    secondary = {'winner': 'winner_brier', 'length': 'length_rps', 'kills': 'kills_crps',
                                 'duration': 'duration_crps', 'map_play': 'map_play_brier'}.get(task)
                    records.append({'scope': scope, 'dimension': dimension, 'value': str(value), 'task': task,
                                    'total': len(group), 'n': count, 'absent': len(group) - count,
                                    'nll': group[column].mean() if task != 'map_play' else np.nan,
                                    'proper_secondary': group[secondary].mean() if secondary else np.nan})
    calculated = pd.DataFrame(records)
    keys = ['scope', 'dimension', 'value', 'task']
    expected = exported[exported.candidate == candidate].copy()
    expected['value'] = expected.value.astype(str)
    calculated = calculated.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    audit.truth(candidate + ':complete profile support', calculated.index.equals(expected.index))
    if calculated.index.equals(expected.index):
        for column in ('total', 'n', 'absent', 'nll', 'proper_secondary'):
            audit.compare(candidate + ':all profile aggregates:' + column, calculated[column], expected[column],
                          'integer' if column in ('total', 'n', 'absent') else 'metric')


def check_trajectory(audit, model, name, train, arrays, series, maps):
    receipt = model.fit_receipt
    size = model.config['batch_size']
    expected_steps = 12 * math.ceil(len(train) / size)
    for key, expected in (('epochs_completed', 12), ('optimizer_steps', expected_steps), ('train_series_count', len(train)),
                          ('software_smoke', False), ('threads', 4), ('history_length', 16)):
        audit.truth(name + ':trajectory:' + key, receipt[key] == expected, receipt[key])
    audit.truth(name + ':optimizer identity', receipt['optimizer'] == 'AdamW')
    audit.truth(name + ':preprocessing TRAIN only', receipt['preprocessing_split'] == 'TRAIN')
    audit.truth(name + ':CPU saved trajectory', receipt['device'] == 'cpu')
    audit.truth(name + ':receipt config correspondence', receipt['config'] == model.config)
    audit.truth(name + ':receipt identity correspondence',
                receipt['seed'] == model.seed and receipt['champion_enabled'] == model.champion_enabled
                and receipt['multitask'] == model.multitask)
    for group in model.optimizer_state['param_groups']:
        for key, expected in (('lr', model.config['lr']), ('weight_decay', model.config['weight_decay']),
                              ('betas', (.9, .999)), ('eps', 1e-8), ('amsgrad', False),
                              ('maximize', False)):
            audit.truth(name + ':saved AdamW group:' + key, group[key] == expected, group[key])
    audit.truth(name + ':TRAIN index hash', receipt['train_indices_sha256'] == hashlib.sha256(train.tobytes()).hexdigest())
    rng = np.random.default_rng(model.seed)
    orders = [hashlib.sha256(rng.permutation(train).tobytes()).hexdigest() for _ in range(12)]
    audit.truth(name + ':12 independently reproduced shuffle hashes', orders == receipt['epoch_order_sha256'])
    states = model.optimizer_state['state']
    steps = sorted({int(value['step'].item()) for value in states.values()})
    audit.truth(name + ':actual AdamW state steps', steps == [expected_steps], steps)
    for parameter, state in states.items():
        audit.truth(f'{name}:optimizer finite moments:{parameter}', bool(torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()))
    digest_state = hashlib.sha256()
    for key, value in model.state_dict.items():
        digest_state.update(key.encode())
        digest_state.update(value.detach().cpu().numpy().tobytes())
        audit.truth(name + ':network saved tensor:' + key, torch.equal(value, model.network.state_dict()[key]))
    audit.truth(name + ':final saved parameter hash', digest_state.hexdigest() == receipt['final_state_sha256'])
    train_maps = maps[maps.row_index.isin(train)]
    audit.compare(name + ':TRAIN score count', series.iloc[train].score_index.notna().sum(), receipt['train_score_count'], 'integer')
    for task, label in (('kills', 'kills'), ('duration', 'duration_minutes')):
        audit.compare(name + ':TRAIN map count:' + task, train_maps[label].notna().sum(), receipt['train_map_label_counts'][task], 'integer')
        audit.truth(name + ':TRAIN map support:' + task,
                    len(train_maps) == len(train_maps.drop_duplicates(['row_index', 'map_index'])))
    for key in ('champions', 'patches', 'competitions'):
        audit.compare(name + ':TRAIN vocabulary size:' + key, max(2, int(arrays[key][train].max()) + 1),
                      model.vocab_sizes[key], 'integer')
    for epoch in receipt['epoch_losses']:
        expected_observed = {'score': int(series.iloc[train].score_index.notna().sum())}
        expected_observed.update({task: int(train_maps.loc[train_maps[label].notna(), 'row_index'].nunique())
                                  if model.multitask else 0
                                  for task, label in (('kills', 'kills'), ('duration', 'duration_minutes'))})
        audit.truth(name + f":epoch {epoch['epoch']}:observed task populations",
                    epoch['observed_series'] == expected_observed, epoch['observed_series'])
        audit.truth(name + f":epoch {epoch['epoch']}:finite active losses",
                    all(epoch['losses'][task] is not None and np.isfinite(epoch['losses'][task])
                        for task, count in expected_observed.items() if count))
        audit.truth(name + f":epoch {epoch['epoch']}:all TRAIN traversed", epoch['series_seen'] == len(train))
    audit.truth(name + ':all twelve epoch receipts', [item['epoch'] for item in receipt['epoch_losses']] == list(range(1, 13)))


def verify(args):
    paths = {stage: ROOT / 'data' / stage / 'champion_history' / args.run_id
             for stage in ('04_feature', '05_model_input', '06_models', '07_model_output', '08_reporting')}
    reporting = paths['08_reporting']
    verification = reporting / 'verification'
    tolerances_path = verification / 'numerical_tolerances.json'
    tolerances = json.loads(tolerances_path.read_text())
    audit = Audit(tolerances)
    protocol = json.loads((paths['05_model_input'] / 'protocol.json').read_text())
    binding = json.loads((paths['05_model_input'] / 'binding.json').read_text())
    execution = json.loads((reporting / 'execution.json').read_text())
    decision = json.loads((reporting / 'decision.json').read_text())
    if execution['status'] != 'COMPLETED':
        raise RuntimeError('Real formal execution is not complete; numerical execution is not authorized')
    executed_at = datetime.now(timezone.utc)
    execution_id = executed_at.strftime('%Y%m%dT%H%M%S%fZ')
    archive = verification / 'code' / execution_id
    archive.mkdir(parents=True, exist_ok=True)
    code_hashes = {}
    continuation = json.loads((paths['05_model_input'] / 'scoring_continuation_binding.json').read_text())
    runner_relative = 'src/models/champion_history_runner.py'
    original_runner = ROOT / continuation['runner_change']['original_snapshot']
    audit.truth('scoring continuation original binding lineage',
                continuation['frozen_binding_sha256'] == digest(paths['05_model_input'] / 'binding.json'))
    audit.truth('scoring continuation preserves original runner binding',
                digest(original_runner) == continuation['runner_change']['original_sha256'] == binding['pins'][runner_relative])
    current_pins = {runner_relative: continuation['runner_change']['current_sha256'],
                    **continuation['additional_pins']}
    for relative, expected in current_pins.items():
        audit.truth('explicit scoring-only binding:' + relative, digest(ROOT / relative) == expected)
        if relative.endswith('.py'):
            destination = archive / 'scoring_continuation' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            code_hashes['scoring_continuation:' + relative] = digest(destination)
    def functions(source):
        return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    original_functions = functions(original_runner.read_text())
    current_functions = functions((ROOT / runner_relative).read_text())
    original_statements = original_functions['run_experiment'].body
    reconstructed_statements = (current_functions['run_experiment'].body[:-1]
                                + current_functions['_score_completed'].body[4:])
    audit.truth('independent AST identity of original training CAL scoring and gates',
                [ast.dump(node, include_attributes=False) for node in original_statements]
                == [ast.dump(node, include_attributes=False) for node in reconstructed_statements])
    for relative, expected in binding['pins'].items():
        source = original_runner if relative == runner_relative else ROOT / relative
        actual = digest(source)
        audit.truth('frozen source/input correspondence:' + relative, actual == expected)
        if relative.endswith('.py'):
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            code_hashes[relative] = digest(destination)
    shutil.copyfile(Path(__file__), archive / Path(__file__).name)
    code_hashes['independent_verifier'] = digest(archive / Path(__file__).name)
    series = pd.read_parquet(paths['04_feature'] / 'series.parquet')
    maps = pd.read_parquet(paths['04_feature'] / 'maps.parquet')
    with np.load(paths['04_feature'] / 'arrays.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    day = series.date.astype(str)
    release = series.effective_release_day.astype(str)
    splits = {'TRAIN': np.flatnonzero((day < '2025-07-01') & (release < '2025-07-01')),
              'CAL': np.flatnonzero((day >= '2025-07-01') & (day < '2026-01-01') & (release < '2026-01-01')),
              'ASSESS': np.flatnonzero((day >= '2026-01-01') & (day < '2027-01-01'))}
    for split, ix in splits.items():
        split_series = series.iloc[ix]
        split_maps = maps[maps.row_index.isin(ix)]
        counts = {'series': len(ix), 'recorded_maps': len(split_maps),
                  'winner_labels': int(split_series.winner.notna().sum()),
                  'score_labels': int(split_series.score_index.notna().sum()),
                  'kills_labels': int(split_maps.kills.notna().sum()),
                  'duration_labels': int(split_maps.duration_minutes.notna().sum()),
                  'first_map_kills': int(split_maps.loc[split_maps.map_index == 1, 'kills'].notna().sum()),
                  'first_map_duration': int(split_maps.loc[split_maps.map_index == 1, 'duration_minutes'].notna().sum())}
        audit.truth(split + ':complete frozen label population', counts == protocol['counts'][split], counts)
    torch.set_num_threads(1)
    stats = preprocess(arrays, splits['TRAIN'], protocol['config']['batch_size'])
    models = {}
    initial_hashes = {}
    capacity = {}
    all_scores = {}
    cal_scores = {}
    support = {}
    embedding_diagnostics = {}
    calibration_results = {}
    coverage = {}
    all_metrics = json.loads((reporting / 'metrics.json').read_text())
    profiles = pd.read_csv(reporting / 'profiles.csv', dtype={'value': str})
    for variant in ('A', 'B', 'C', 'D'):
        raw = {'CAL': [], 'ASSESS': []}
        models[variant] = []
        for seed in SEEDS:
            name = f'{variant}:{seed}'
            model = joblib.load(paths['06_models'] / f'{variant}_seed_{seed}.joblib')
            models[variant].append(model)
            check_trajectory(audit, model, name, splits['TRAIN'], arrays, series, maps)
            reproduced_hash, initialized = initial_state(model, arrays, splits['TRAIN'], maps)
            audit.truth(name + ':independently reproduced initial saved-state hash',
                        reproduced_hash == model.fit_receipt['initial_state_sha256'])
            if not support:
                support = vocabulary_support(arrays, splits, model.vocab_sizes, model.config['batch_size'])
            embedding_diagnostics[name] = embedding_evidence(model, initialized, support)
            audit.truth(name + ':saved config matches frozen recipe', model.config == protocol['config'])
            audit.truth(name + ':paired seed identity', model.seed == seed)
            audit.truth(name + ':variant flags', model.champion_enabled == (variant in ('B', 'D'))
                        and model.multitask == (variant in ('C', 'D')))
            audit.truth(name + ':actual allocated parameter count',
                        sum(value.numel() for key, value in model.state_dict.items()
                            if key not in ('map_ids', 'loser_scores')) == model.fit_receipt['parameter_count'])
            initial_hashes[name] = model.fit_receipt['initial_state_sha256']
            capacity[name] = model.fit_receipt['parameter_count']
            for key, (mean, scale, count) in stats.items():
                for field, expected in (('mean', mean), ('scale', scale), ('counts', count)):
                    audit.compare(name + ':TRAIN preprocessing:' + key + ':' + field,
                                  expected, getattr(model.preprocessing, field)[key], 'preprocessing' if field != 'counts' else 'integer')
            for split in raw:
                print(f'REPLAY {name} {split}', flush=True)
                raw[split].append(replay(model, arrays, splits[split], stats))
        ensemble = joblib.load(paths['06_models'] / f'{variant}_ensemble.joblib')
        audit.truth(variant + ':ensemble three embedded paired seeds', len(ensemble.models) == 3)
        for embedded, saved in zip(ensemble.models, models[variant]):
            audit.truth(variant + f':ensemble saved-weight correspondence:{saved.seed}',
                        embedded.seed == saved.seed and embedded.champion_enabled == saved.champion_enabled
                        and embedded.multitask == saved.multitask
                        and embedded.state_dict.keys() == saved.state_dict.keys()
                        and all(torch.equal(embedded.state_dict[key], saved.state_dict[key]) for key in saved.state_dict))
        cal_series = series.iloc[splits['CAL']].copy().reset_index(drop=True)
        cal_maps = maps[maps.row_index.isin(splits['CAL'])].copy().reset_index(drop=True)
        independent_cal = calibration(raw['CAL'], cal_series, cal_maps, ensemble.calibration_receipt)
        cal_receipt = ensemble.calibration_receipt
        audit.truth(variant + ':CAL split binding', cal_receipt['split'] == 'CAL')
        audit.truth(variant + ':CAL index binding',
                    cal_receipt['cal_indices_sha256'] == hashlib.sha256(splits['CAL'].tobytes()).hexdigest())
        audit.truth(variant + ':CAL frozen three-seed mixture',
                    cal_receipt['frozen_networks'] and cal_receipt['mixture_components'] == 3)
        audit.compare(variant + ':CAL series population', len(cal_series), cal_receipt['series_count'], 'integer')
        audit.compare(variant + ':CAL score population', cal_series.score_index.notna().sum(),
                      cal_receipt['score_observed_series'], 'integer')
        calibration_results[variant] = independent_cal
        for name, item in independent_cal.items():
            exported = ensemble.calibration_receipt[name]
            audit.truth(variant + ':saved CAL scalar bounded:' + name, .25 <= exported['value'] <= 4)
            audit.truth(variant + ':CAL scalar original bounds:' + name, exported['bounds'] == [.25, 4.])
            audit.truth(variant + ':CAL scalar original penalty:' + name, exported['regularization'] == .001)
            audit.truth(variant + ':CAL scalar matches deployed ensemble:' + name, exported['value'] == getattr(ensemble, name))
            audit.truth(variant + ':CAL optimizer success:' + name, item['success'])
            for field in ('value', 'nll', 'penalized_objective'):
                audit.compare(variant + ':CAL:' + name + ':' + field, item[field], exported[field],
                              'cal_scalar' if field == 'value' else 'cal_objective')
            for field in ('nll', 'penalized_objective'):
                audit.compare(variant + ':CAL saved scalar exact objective:' + name + ':' + field,
                              item['saved_value_' + field], exported[field], 'cal_objective')
            for field in ('observed_maps', 'observed_series'):
                if field in item:
                    audit.compare(variant + ':CAL population:' + name + ':' + field, item[field], exported[field], 'integer')
        for split in ('CAL', 'ASSESS'):
            ix = splits[split]
            sub_series = series.iloc[ix].copy().reset_index(drop=True)
            sub_maps = maps[maps.row_index.isin(ix)].copy().reset_index(drop=True)
            name = variant + ('_cal' if split == 'CAL' else '')
            # Replay the deployed saved scalar; its bounded optimum was verified
            # separately above, so optimizer stopping noise cannot contaminate
            # the comparison of saved-weight arithmetic with exported forecasts.
            temperature = ensemble.score_temperature
            components = []
            for component in raw[split]:
                c = {key: value.copy() for key, value in component.items()}
                c['score'] = temper(c['score'], temperature)
                if 'kills_mu' in c:
                    c['kills_alpha'] *= ensemble.kill_scale
                    c['duration_log_sigma'] *= ensemble.duration_scale
                components.append(c)
            score = np.mean([c['score'] for c in components], axis=0)
            terminal = pd.read_parquet(paths['07_model_output'] / f'{name}_terminal.parquet')
            audit.compare(name + ':complete terminal PMF', score, terminal[[f'score_p_{j}' for j in range(6)]], 'prediction')
            bo = sub_series.best_of.to_numpy(int)
            lp = score[:, :3] + score[:, 3:]
            audit.compare(name + ':winner probability', score[:, :3].sum(1), terminal.p_win, 'prediction')
            audit.compare(name + ':complete length PMF', lp, terminal[[f'length_p_{j}' for j in range(3)]], 'prediction')
            grid = bo[:, None] // 2 + 1 + np.arange(3)
            play = np.column_stack([(lp * (grid >= m)).sum(1) for m in range(1, 6)])
            audit.compare(name + ':all five play probabilities', play, terminal[[f'map_{m}_play_probability' for m in range(1, 6)]], 'prediction')
            legal = np.tile(np.arange(3), 2)[None, :] <= (bo // 2)[:, None]
            audit.truth(name + ':legal exact support', np.all(score[~legal] == 0))
            audit.compare(name + ':normalized PMF', score.sum(1), np.ones(len(score)), 'prediction')
            exported_components = pd.read_parquet(paths['07_model_output'] / f'{name}_components.parquet')
            lookup = pd.Series(np.arange(len(sub_series)), index=sub_series.row_index)
            for j, component in enumerate(components):
                rows = exported_components[exported_components.component == j]
                positions = rows.row_index.map(lookup).to_numpy(int)
                audit.compare(name + f':seed {j}:all exported PMFs', component['score'][positions], rows[[f'score_p_{k}' for k in range(6)]], 'prediction')
                audit.compare(name + f':seed {j}:weights', np.full(len(rows), 1 / 3), rows.weight, 'prediction')
                if 'kills_mu' in component:
                    m = rows.map_index.to_numpy(int) - 1
                    for field in FIELDS:
                        audit.compare(name + f':seed {j}:' + field, component[field][positions, m], rows[field], 'prediction')
                    audit.truth(name + f':seed {j}:all supported conditional maps exported', len(rows) == int(bo.sum()) and rows.conditional_on_played.all())
                    audit.truth(name + f':seed {j}:unique complete supported map rows',
                                not rows.duplicated(['row_index', 'map_index']).any()
                                and np.array_equal(np.sort(rows.row_index.unique()), np.sort(sub_series.row_index.to_numpy()))
                                and np.all((rows.map_index.to_numpy() >= 1) & (rows.map_index.to_numpy() <= rows.best_of.to_numpy())))
                else:
                    audit.truth(name + f':seed {j}:score-only no fake props', not rows.statistics_head_available.any() and len(rows) == len(sub_series))
            parameters = components if variant in ('C', 'D') else None
            ss, ms, extra = score_rows(sub_series, sub_maps, score, parameters, np.ones(3) / 3)
            scores_path = paths['07_model_output'] / f'{name}_series_scores.parquet'
            compare_frame(audit, name + ':series scores', ss, pd.read_parquet(scores_path))
            metrics = json.loads((reporting / f'{variant}_cal_metrics.json').read_text()) if split == 'CAL' else all_metrics[variant]
            compare_metrics(audit, name, ss, ms, extra, metrics)
            if split == 'ASSESS':
                all_scores[variant] = ss
                compare_profiles(audit, variant, ss, ms, profiles)
                compare_frame(audit, name + ':map scores', ms, pd.read_parquet(paths['07_model_output'] / f'{variant}_map_scores.parquet'))
                if variant == 'D':
                    first_maps = sub_maps[sub_maps.map_index == 1].reset_index(drop=True)
                    fs, fm, fe = score_rows(sub_series, first_maps, score, components, np.ones(3) / 3)
                    all_scores['D_first'] = fs
                    compare_frame(audit, 'D_first:series scores', fs, pd.read_parquet(paths['07_model_output'] / 'D_first_series_scores.parquet'))
                    compare_metrics(audit, 'D_first', fs, fm, fe, all_metrics['D_first'])
            else:
                cal_scores[variant] = ss
            coverage[name] = {'series': len(sub_series), 'maps': len(sub_maps), 'seed_pmfs': 3 * len(sub_series),
                              'exported_component_rows': len(exported_components), 'all_supported_prop_parameters': variant in ('C', 'D')}
    for seed in SEEDS:
        hashes = [initial_hashes[f'{v}:{seed}'] for v in ('A', 'B', 'C', 'D')]
        audit.truth(f'paired seed {seed}:identical starts', len(set(hashes)) == 1, hashes)
    audit.truth('distinct paired seed starts', len({initial_hashes[f'A:{seed}'] for seed in SEEDS}) == 3)
    audit.truth('same allocated topology capacity', len(set(capacity.values())) == 1, capacity)
    assess = series.iloc[splits['ASSESS']].copy().reset_index(drop=True)
    assess_maps = maps[maps.row_index.isin(splits['ASSESS'])].copy().reset_index(drop=True)
    c0 = pd.read_parquet(paths['07_model_output'] / 'C0_coherent_reference_terminal.parquet')
    frozen_c0 = pd.read_parquet(ROOT / 'data/07_model_output/a0_capacity/native_capacity_20261004_01/predictions.parquet')
    frozen_c0 = frozen_c0.assign(match_id=frozen_c0.golgg_match_id.astype(str)).set_index('match_id').loc[assess.match_id.astype(str)]
    manifest = json.loads((ROOT / 'conf/base/research_benchmark.json').read_text())
    metadata_paths = [Path(path) for path, value in continuation['additional_pins'].items()
                      if path.endswith(manifest['input']['path']) and value == manifest['input']['sha256']]
    audit.truth('C0 native participant metadata manifest binding', len(metadata_paths) == 1)
    metadata = pd.read_parquet(metadata_paths[0])
    audit.truth('C0 native metadata unique identities', not metadata.golgg_match_id.astype(str).duplicated().any())
    metadata = metadata.assign(match_id=metadata.golgg_match_id.astype(str)).set_index('match_id').loc[assess.match_id.astype(str)]
    a, b = assess.team_a.astype(str).to_numpy(), assess.team_b.astype(str).to_numpy()
    one, two = metadata.team1_id.astype(str).to_numpy(), metadata.team2_id.astype(str).to_numpy()
    direct, reversed_ = (a == one) & (b == two), (a == two) & (b == one)
    audit.truth('C0 ID-derived exact orientation partition', np.all(direct ^ reversed_))
    audit.compare('C0 native saved labels versus canonical metadata', frozen_c0.y, metadata.y, 'integer')
    audit.compare('C0 native saved best-of versus target', frozen_c0.best_of, assess.best_of, 'integer')
    aligned_y = np.where(reversed_, 1 - frozen_c0.y.to_numpy(float), frozen_c0.y.to_numpy(float))
    audit.compare('C0 ID-aligned labels versus target', aligned_y, assess.winner, 'integer')
    aligned_probability = np.where(reversed_, 1 - frozen_c0['w32_e12__full'].to_numpy(float),
                                   frozen_c0['w32_e12__full'].to_numpy(float))
    c0_alignment = {'direct': int(direct.sum()), 'reversed': int(reversed_.sum()),
                    'unmatched': int((~(direct | reversed_)).sum()),
                    'sign_decision_inputs': ['canonical team1_id', 'canonical team2_id', 'target team_a', 'target team_b'],
                    'labels_used_only_to_validate_lineage': True, 'native_metadata_sha256': digest(metadata_paths[0])}
    score = coherent_reference(aligned_probability, assess.best_of.to_numpy(int))
    audit.compare('C0 independent coherent terminal PMF', score, c0[[f'score_p_{j}' for j in range(6)]], 'prediction')
    css, cms, ce = score_rows(assess, assess_maps, score)
    all_scores['C0'] = css
    compare_frame(audit, 'C0:independent metrics', css, pd.read_parquet(paths['07_model_output'] / 'C0_series_scores.parquet'))
    compare_metrics(audit, 'C0', css, cms, ce, all_metrics['C0_coherent_reference'])
    compare_profiles(audit, 'C0_coherent_reference', css, cms, profiles)
    audit.compare('C0 frozen ID-aligned winner correspondence', css.p_win, aligned_probability, 'prediction')
    old = pd.read_parquet(paths['07_model_output'] / 'old_props_components.parquet')
    components, weights = [], []
    audit.truth('old_props:complete selected component identity', old.component.nunique() > 0)
    lp = np.zeros((len(assess), 3))
    for component, frame in old.groupby('component', sort=True):
        frame = frame.set_index('row_index').loc[assess.row_index]
        audit.truth(f'old_props:component {component}:unique complete ASSESS rows',
                    len(frame) == len(assess) and frame.index.is_unique
                    and np.array_equal(frame.match_id.astype(str), assess.match_id.astype(str)))
        audit.truth(f'old_props:component {component}:one positive fixed weight',
                    frame.weight.nunique() == 1 and float(frame.weight.iloc[0]) > 0)
        weight = float(frame.weight.iloc[0])
        weights.append(weight)
        parameters = {field: np.repeat(frame[field].to_numpy()[:, None], 5, axis=1) for field in FIELDS}
        components.append(parameters)
        lp += weight * frame[[f'length_p_{j}' for j in range(3)]].to_numpy()
    audit.compare('old_props:normalized complete mixture weights', np.sum(weights), 1., 'prediction')
    oss, oms, oe = score_rows(assess, assess_maps, None, components, np.asarray(weights), lp, first_only=True)
    all_scores['old_props'] = oss
    compare_frame(audit, 'old_props:series', oss, pd.read_parquet(paths['07_model_output'] / 'old_props_series_scores.parquet'))
    compare_frame(audit, 'old_props:maps', oms, pd.read_parquet(paths['07_model_output'] / 'old_props_first_map_scores.parquet'))
    compare_metrics(audit, 'old_props', oss, oms, oe, all_metrics['old_props'])
    compare_profiles(audit, 'old_props', oss, oms, profiles)
    exported_comparisons = json.loads((reporting / 'comparisons.json').read_text())
    independent_comparisons = {}
    for candidate, control, metric in protocol['primary_contrasts']:
        key = f'{candidate}-{control}:{metric}'
        for column in ('row_index', 'match_id', 'date'):
            audit.truth(key + ':whole-series paired alignment:' + column,
                        np.array_equal(all_scores[candidate][column].astype(str),
                                       all_scores[control][column].astype(str)))
        audit.truth(key + ':one row per whole series',
                    not all_scores[candidate].row_index.duplicated().any()
                    and not all_scores[control].row_index.duplicated().any())
        audit.truth(key + ':identical evaluated label availability',
                    np.array_equal(np.isfinite(all_scores[candidate][metric]), np.isfinite(all_scores[control][metric])))
        result = bootstrap(all_scores[candidate], all_scores[control], metric,
                           protocol['confidence'], protocol['bootstrap_seed'], protocol['bootstrap_draws'])
        independent_comparisons[key] = result
        exported = exported_comparisons[key]
        for field, value in result.items():
            if isinstance(value, list) and field == 'months':
                audit.truth(key + ':months', value == exported[field])
            else:
                audit.compare(key + ':bootstrap:' + field, value, exported[field],
                              'integer' if field in ('n', 'population_n', 'candidate_missing', 'control_missing', 'paired_absent', 'blocks', 'supported_blocks', 'draws', 'seed', 'undefined_draws') else 'bootstrap')
        for alias, point, interval in (('match_weighted', 'match_weighted_delta', 'ci'),
                                       ('equal_month', 'equal_month_delta', 'equal_month_ci')):
            audit.compare(key + ':nested alias:' + alias + ':point', result[point], exported[alias]['delta'], 'bootstrap')
            audit.compare(key + ':nested alias:' + alias + ':interval', result[interval], exported[alias]['interval'], 'bootstrap')
    diagnostics = json.loads((reporting / 'multitask_diagnostics.json').read_text())
    for candidate, control in (('C', 'A'), ('D', 'B')):
        for metric in ('winner_nll', 'length_nll'):
            key = f'{candidate}-{control}:{metric}'
            result = bootstrap(all_scores[candidate], all_scores[control], metric, .95,
                               protocol['bootstrap_seed'], protocol['bootstrap_draws'])
            for field in ('match_weighted_delta', 'equal_month_delta', 'ci', 'equal_month_ci'):
                audit.compare(key + ':diagnostic bootstrap:' + field, result[field], diagnostics[key][field], 'bootstrap')
    selection = {task: min(('B', 'D'), key=lambda v: cal_scores[v][task + '_nll'].mean()) for task in ('winner', 'length')}
    selection.update(kills='D', duration='D')
    audit.truth('CAL-only task selection', selection == decision['selection'], selection)
    def nonincrease(a, b, metric):
        x, y = all_scores[a][metric].to_numpy(), all_scores[b][metric].to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        return bool(valid.any() and (x[valid] - y[valid]).mean() <= 0)
    def primary_pass(keys):
        return all(independent_comparisons[k]['ci'] is not None and independent_comparisons[k]['ci'][1] < 0 for k in keys)
    gates = {}
    for task in ('winner', 'length'):
        chosen = selection[task]
        matched = 'A' if chosen == 'B' else 'C'
        reference = 'C0' if task == 'winner' else 'old_props'
        keys = [f'{chosen}-{matched}:{task}_nll', f'{chosen}-{reference}:{task}_nll']
        metric = 'winner_brier' if task == 'winner' else 'length_rps'
        guards = all(nonincrease(chosen, other, metric) for other in (matched, reference))
        passed = primary_pass(keys) and guards
        gates[task] = 'ADOPT' if passed else 'NOT_ADOPTED'
        audit.truth(task + ':decision gate', gates[task] == decision['tasks'][task]['decision'])
        audit.truth(task + ':point guard', guards == decision['tasks'][task]['point_guard_passed'])
    for task in ('kills', 'duration'):
        keys = [f'D-C:{task}_nll', f'D_first-old_props:{task}_nll']
        guards = nonincrease('D', 'C', task + '_crps') and nonincrease('D_first', 'old_props', task + '_crps')
        coverage_failed = False
        for a, b in (('D', 'C'), ('D_first', 'old_props')):
            for level, minimum in ((50, .45), (80, .75)):
                x, y = (all_scores[v][f'{task}_coverage_{level}'].to_numpy() for v in (a, b))
                valid = np.isfinite(x) & np.isfinite(y)
                if not valid.any():
                    coverage_failed = True
                else:
                    ca, cb = x[valid].mean(), y[valid].mean()
                    coverage_failed |= ca < minimum or abs(ca - level / 100) > abs(cb - level / 100) + .025
        gates[task] = 'ADOPT' if primary_pass(keys) and guards and not coverage_failed else 'NOT_ADOPTED'
        audit.truth(task + ':decision gate', gates[task] == decision['tasks'][task]['decision'])
        audit.truth(task + ':CRPS guard', guards == decision['tasks'][task]['crps_guard_passed'])
        audit.truth(task + ':coverage guard', bool(coverage_failed) == bool(decision['tasks'][task]['coverage_failures']))
    bundle = all(value == 'ADOPT' for value in gates.values()) and selection['winner'] == selection['length']
    audit.truth('overall bundle gate', bundle == decision['overall_bundle_gate_passed'])
    audit.truth('formal actual trajectory count', len(execution['formal_trajectories']) == 12)
    audit.truth('actual scalar count', sum(len(items) for items in calibration_results.values()) == 8)
    audit.compare('formal actual total epochs', sum(model.fit_receipt['epochs_completed']
                  for group in models.values() for model in group), execution['formal_seed_epochs'], 'integer')
    audit.truth('no C0 or assessment fitting', execution['c0_fits'] == 0 and execution['assessment_fits'] == 0)
    audit.compare('actual optimizer trajectory count', len(execution['formal_trajectories']), execution['formal_fit_count'], 'integer')
    audit.compare('actual ensemble scalar receipt count', sum(entry['receipt']['scalar_count']
                  for entry in execution['calibration_fits']), execution['ensemble_cal_scalar_fit_count'], 'integer')
    authorization_path = paths['05_model_input'] / 'resume_authorization.json'
    authorization = json.loads(authorization_path.read_text())
    original_path = reporting / 'execution_interrupted.json'
    original = json.loads(original_path.read_text())
    resumption = json.loads((reporting / 'resumption_completed.json').read_text())
    recovery_accounting = {}
    for key, expected in (('formal_completed_trajectories', 12), ('formal_fit_attempts', 13),
                          ('interrupted_fit_attempts', 1), ('restored_completed_checkpoints', 3),
                          ('resumed_actual_fit_attempts', 9), ('interrupted_partial_seed_epochs', 'UNKNOWN'),
                          ('no_recipe_or_gate_change', True)):
        audit.truth('authorized resumption accounting:' + key,
                    execution[key] == expected and decision[key] == expected and resumption[key] == expected)
        recovery_accounting[key] = execution[key]
    audit.truth('explicit identical-recipe recovery authorization',
                authorization['user_decision'] == 'Resume the identical recipe')
    audit.truth('original interrupted execution binding',
                digest(original_path) == authorization['interrupted_execution_sha256']
                == execution['original_execution_sha256'])
    audit.truth('recovery authorization binding', digest(authorization_path) == execution['authorization_sha256'])
    audit.truth('identical frozen protocol under recovery',
                digest(paths['05_model_input'] / 'protocol.json') == authorization['frozen_protocol_sha256'])
    audit.truth('recovery source binding',
                digest(ROOT / 'scripts/resume_champion_history.py') == authorization['recovery_code_sha256'])
    completed_original = [r for r in original['formal_trajectories'] if r['status'] == 'COMPLETED']
    lost_original = [r for r in original['formal_trajectories'] if r['status'] != 'COMPLETED']
    audit.truth('three completed originals and one lost partial',
                len(completed_original) == 3 and [(r['variant'], r['seed']) for r in lost_original] == [('D', SEEDS[0])])
    audit.truth('nine newly completed recovery records',
                sum(not record.get('checkpoint_restored_without_fit', False)
                    for record in execution['formal_trajectories']) == 9)
    audit.truth('three restored completed recovery records',
                sum(record.get('checkpoint_restored_without_fit', False)
                    for record in execution['formal_trajectories']) == 3)
    audit.truth('scoring-only repair adds no fits',
                execution['scoring_repair_new_fits'] == decision['scoring_repair_new_fits'] == resumption['scoring_repair_new_fits'] == 0)
    for record in completed_original:
        saved = next(model for model in models[record['variant']] if model.seed == record['seed'])
        audit.truth(f"restored original fit receipt:{record['variant']}:{record['seed']}",
                    saved.fit_receipt == record['fit_receipt'])
        audit.truth('archived original model bytes:' + record['model_path'],
                    digest(reporting / 'interrupted_model_checkpoints' / Path(record['model_path']).name) == record['sha256'])
    for relative, expected in decision['output_hashes'].items():
        audit.truth('actual sealed output correspondence:' + relative, digest(ROOT / relative) == expected)
    maxima = {}
    for item in audit.checks:
        if 'max_absolute_difference' in item:
            kind = item['kind']
            maxima[kind] = max(maxima.get(kind, 0.), item['max_absolute_difference'])
    evidence = {'run_id': args.run_id, 'status': 'PASS' if not audit.failures else 'FAIL',
                'executed_at': executed_at.isoformat(), 'execution_id': execution_id, 'independent': True,
                'candidate_methods_called': [], 'network_fits': 0, 'verification_scalar_optimizations': 8,
                'verification_threads': 1,
                'shared_numerical_primitives': ['torch linear/tanh/softmax/einsum and standard initialization draws',
                                                'numpy reductions and seeded Generator', 'scipy special functions/CDFs',
                                                'scipy bounded minimize_scalar algorithm; objectives independently reconstructed'],
                'precision_limits': tolerances,
                'formal_saved_models_replayed': 12, 'fixed_epochs_per_trajectory': 12,
                'recovery_accounting': recovery_accounting,
                'complete_CAL_and_ASSESS': coverage, 'primary_intervals': 12, 'estimands_per_interval': 2,
                'bootstrap_draws': 5000, 'tolerances_sha256': digest(tolerances_path),
                'source_archive_hashes': code_hashes, 'check_count': len(audit.checks),
                'compared_numeric_values': sum(item.get('values', 0) for item in audit.checks),
                'max_absolute_differences': maxima, 'failures': audit.failures, 'checks': audit.checks,
                'calibration': calibration_results, 'primary_comparisons': independent_comparisons,
                'selection': selection, 'task_gates': gates, 'capacity': capacity,
                'embedding_OOV_support': embedding_diagnostics,
                'C0_participant_alignment': c0_alignment,
                'scoring_continuation_binding_sha256': digest(paths['05_model_input'] / 'scoring_continuation_binding.json'),
                'capacity_note': 'Identical allocated topology; score-only prop parameters are inactive, and noChampion embedding paths receive no informative champion identity. Equal active capacity is not claimed.',
                'limitations': ['Old prop component predictions and C0 winner margins are treated as frozen reference inputs, not independent reimplementations of their original fitting algorithms; the C0 coherent terminal law is independently reconstructed.',
                                'Absence-text semantics are covered by the separate temporal audit; frozen profiles.csv numerical aggregates are reconstructed. Parent-owned supplemental descriptive profiles are not audited here.',
                                'No cryptographic execution-continuity claim; archived source hashes establish correspondence at verification time only.']}
    immutable_receipt = verification / f'numerical_receipt_{execution_id}.json'
    save(immutable_receipt, evidence)
    save(verification / 'numerical_receipt.json', evidence)
    return {key: evidence[key] for key in ('status', 'check_count', 'compared_numeric_values', 'max_absolute_differences', 'failures')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN)
    args = parser.parse_args()
    result = verify(args)
    print(json.dumps(clean(result), indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
