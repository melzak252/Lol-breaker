#!/usr/bin/env python3
"""Independent retained-parameter arithmetic and saved-model replay; never fits."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp, ndtr
from scipy.stats import nbinom
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify(run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('Invalid run identity')
    report = ROOT / f'data/08_reporting/draft_prop/{run_id}'
    inputs = ROOT / f'data/05_model_input/draft_prop/{run_id}'
    binding = json.loads((inputs / 'binding.json').read_text())
    decision = json.loads((report / 'decision.json').read_text())
    metrics = json.loads((report / 'metrics.json').read_text())
    contrasts = json.loads((report / 'comparisons.json').read_text())
    schema = json.loads((ROOT / binding['schema_path']).read_text())
    frame = pd.read_parquet(ROOT / binding['dataset_path'])
    assess = frame.loc[pd.to_datetime(frame.date) >= pd.Timestamp('2026-01-01')].reset_index(drop=True)
    exports = pd.read_parquet(ROOT / f'data/07_model_output/draft_prop/{run_id}/distribution_components.parquet')
    scored = pd.read_parquet(ROOT / f'data/07_model_output/draft_prop/{run_id}/row_scores.parquet')
    errors, counts, hash_count, reconstructed = {}, {}, 0, {}

    def check(name, actual, expected, tolerance=2e-10):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        np.testing.assert_allclose(a, b, rtol=0, atol=tolerance, equal_nan=True, err_msg=name)
        finite = np.isfinite(a) & np.isfinite(b)
        errors[name] = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        counts[name] = int(a.size)

    pins = {**binding['pins'], **binding['protected_hashes'], **decision['output_hashes']}
    for path, expected in pins.items():
        with (ROOT / path).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected, path
        hash_count += 1
    for candidate, parts in exports.groupby('candidate', sort=True):
        groups = [part.reset_index(drop=True) for _, part in parts.groupby('component', sort=True)]
        for group in groups:
            assert group.match_id.equals(assess.match_id)
            assert group.weight.nunique() == 1
        weights = np.array([g.weight.iloc[0] for g in groups])
        check(candidate + ':mixture_mass', weights.sum(), 1., 1e-14)
        mu = np.column_stack([g.kills_mu for g in groups])
        alpha = np.column_stack([g.kills_alpha for g in groups])
        lm = np.column_stack([g.duration_log_mu for g in groups])
        sigma = np.column_stack([g.duration_log_sigma for g in groups])
        y, duration = assess.kills.to_numpy(float), assess.duration_minutes.to_numpy(float)
        r = 1 / alpha
        p = 1 / (1 + alpha * mu)
        logpmf = gammaln(y[:, None] + r) - gammaln(r) - gammaln(y[:, None] + 1)
        logpmf += r * np.log(p) + y[:, None] * np.log1p(-p)
        logpdf = -np.log(duration[:, None] * sigma * np.sqrt(2 * np.pi))
        logpdf -= .5 * ((np.log(duration[:, None]) - lm) / sigma) ** 2
        knll, dnll = (-logsumexp(ll + np.log(weights), axis=1) for ll in (logpmf, logpdf))
        rows = scored.loc[scored.candidate == candidate].reset_index(drop=True)
        assert rows.match_id.equals(assess.match_id)
        check(candidate + ':kill_NLL', knll, rows.kills_nll)
        check(candidate + ':duration_NLL', dnll, rows.duration_nll)
        reconstructed[candidate] = {'kills': knll, 'duration': dnll}
        cdf = sum(w * nbinom.cdf(np.arange(1201)[None, :], r[:, j, None], p[:, j, None]) for j, w in enumerate(weights))
        check(candidate + ':kill_mass_through1200', cdf[:, -1], np.ones(len(assess)), 1e-12)
        kcrps = np.sum((cdf - (y[:, None] <= np.arange(1201))) ** 2, axis=1)
        check(candidate + ':kill_CRPS_grid1200', kcrps, rows.kills_proper_secondary, 2e-8)
        for q in (10, 25, 75, 90):
            check(candidate + f':kill_q{q}', np.argmax(cdf >= q / 100, axis=1), rows[f'kills_q{q}'], 0.)
            cut = rows[f'duration_q{q}'].to_numpy(float)
            check(candidate + f':duration_q{q}_cdf', ndtr((np.log(cut)[:, None] - lm) / sigma) @ weights, np.full(len(cut), q / 100), 2e-12)
        sample = np.unique(np.linspace(0, len(assess) - 1, 24, dtype=int))
        for i in sample:
            def F(x):
                return 0. if x <= 0 else float(ndtr((np.log(x) - lm[i]) / sigma[i]) @ weights)
            crps = quad(lambda x: F(x) ** 2, 0, duration[i], epsabs=1e-9)[0]
            crps += quad(lambda x: (1 - F(x)) ** 2, duration[i], np.inf, epsabs=1e-9)[0]
            check(candidate + f':duration_CRPS_integral_{i}', crps, rows.duration_proper_secondary.iloc[i], 2e-8)
        probs = sum(w * g[['length_p_0', 'length_p_1', 'length_p_2']].to_numpy() for w, g in zip(weights, groups))
        bo = assess.best_of.to_numpy(int)
        needed = bo // 2 + 1
        length = assess.maps_played.to_numpy(int)
        check(candidate + ':length_mass', probs.sum(axis=1), np.ones(len(bo)), 2e-7)
        assert (probs[np.arange(3)[None, :] > (bo // 2)[:, None]] == 0).all()
        keep = bo > 1
        lnll = -np.log(probs[np.arange(len(bo)), length - needed])
        lrps = np.sum(np.where(np.arange(3)[None, :] < (bo - needed)[:, None], (np.cumsum(probs, axis=1) - (length[:, None] <= needed[:, None] + np.arange(3))) ** 2, 0), axis=1)
        check(candidate + ':length_NLL', lnll[keep], rows.length_nll[keep])
        check(candidate + ':length_RPS', lrps[keep], rows.length_proper_secondary[keep])
        for task in ('kills', 'duration'):
            check(candidate + ':' + task + '_mean_NLL', reconstructed[candidate][task].mean(), metrics[candidate][task]['nll'])
            obs = y if task == 'kills' else duration
            for low, high, label in ((25, 75, '50'), (10, 90, '80')):
                lower, upper = rows[f'{task}_q{low}'].to_numpy(), rows[f'{task}_q{high}'].to_numpy()
                check(candidate + f':{task}_coverage{label}', np.mean((obs >= lower) & (obs <= upper)), metrics[candidate][task][f'central_{label}_coverage'])
                check(candidate + f':{task}_width{label}', np.mean(upper - lower), metrics[candidate][task][f'central_{label}_mean_width'])
        if candidate != 'existing_selected':
            model = joblib.load(ROOT / f'data/06_models/draft_prop/{run_id}/{candidate}.joblib')
            before = assess.copy(deep=True)
            replay = model.predict(assess)
            pd.testing.assert_frame_equal(assess, before)
            for component, group in zip(replay.components, groups):
                for name in ('kills_mu', 'kills_alpha', 'duration_log_mu', 'duration_log_sigma'):
                    check(candidate + ':replay_' + name, getattr(component, name), group[name], 0.)
                check(candidate + ':replay_length', component.length_probs, group[['length_p_0', 'length_p_1', 'length_p_2']], 2e-7)
            swapped = assess.copy(deep=True)
            for a, b in schema['side_swap'].items():
                swapped[a] = assess[b].to_numpy()
            reflected = model.predict(swapped)
            for task, fn, line in (('kills', 'kills_over', 25.5), ('duration', 'duration_over', 30.5)):
                check(candidate + ':swap_' + task, getattr(replay, fn)(line), getattr(reflected, fn)(line), 1e-14)
            cold = assess.iloc[:3].copy(deep=True)
            cold[model.feature_columns] = np.nan
            cold[model.categorical_features] = -1.
            unknown = model.predict(cold)
            assert np.isfinite(unknown.kills_over(25.5)).all() and np.isfinite(unknown.duration_over(30.5)).all()
            assert ((unknown.kills_over(25.5) >= 0) & (unknown.kills_over(25.5) <= 1)).all()
    months = pd.to_datetime(assess.date).dt.to_period('M').astype(str)
    for key, contrast in contrasts.items():
        task, reference = key.split('_oracle_minus_')
        d = reconstructed['actual_composition_oracle'][task] - reconstructed[reference][task]
        grouped = pd.DataFrame({'month': months, 'd': d}).groupby('month')['d'].agg(['sum', 'count', 'mean'])
        indices = np.random.default_rng(contrast['seed']).integers(0, len(grouped), size=(contrast['draws'], len(grouped)))
        draws = grouped['sum'].to_numpy()[indices].sum(axis=1) / grouped['count'].to_numpy()[indices].sum(axis=1)
        equal = grouped['mean'].to_numpy()[indices].mean(axis=1)
        tail = (1 - contrast['confidence']) / 2
        check(key + ':monthly_delta', d.mean(), contrast['match_weighted_delta'])
        check(key + ':monthly_CI', np.quantile(draws, [tail, 1 - tail]), contrast['ci'])
        check(key + ':equal_delta', grouped['mean'].mean(), contrast['equal_month_delta'])
        check(key + ':equal_CI', np.quantile(equal, [tail, 1 - tail]), contrast['equal_month_ci'])
        assert len(grouped) == contrast['blocks'] == 9 and contrast['draws'] == 5000
    gate = json.loads((report / 'stage2_gate.json').read_text())
    independent_pass = all(contrasts[f'{task}_oracle_minus_{ref}']['ci'][1] < 0 and metrics['actual_composition_oracle'][task]['crps'] <= metrics[ref][task]['crps'] and abs(metrics['actual_composition_oracle'][task]['central_80_coverage'] - .8) - abs(metrics[ref][task]['central_80_coverage'] - .8) <= .02 for task in ('kills', 'duration') for ref in ('matched_no_draft', 'existing_selected'))
    assert independent_pass == (gate['status'] == 'PASS')
    receipt = {'status': 'PASS', 'run_id': run_id, 'assessment_series': len(assess), 'hash_bindings_checked': hash_count, 'comparison_checks': len(counts), 'numerical_values_checked': sum(counts.values()), 'max_error': max(errors.values()), 'errors': errors, 'counts': counts, 'duration_quadrature_cases_per_model': 24, 'bootstrap_contrasts': 4, 'draws_per_contrast': 5000, 'independent_gate': 'PASS' if independent_pass else 'FAIL', 'nonmutation_side_swap_cold_saved_replay': 'PASS', 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'limits': ['Duration CRPS independently integrated for24 fixed rows per model, not every row.', 'No draft sampler or sequential simulator authorized by failed gate; legal picks/stopping/depletion not exercised.', 'Source proxy intervals and every profile slice are not independently rescored here.', 'Existing frozen NN length replay allows2e-7 float32 batch variation; no bitwise NN claim.']}
    output = report / 'independent_numerical_verification.json'
    if output.exists():
        raise FileExistsError('Independent receipt already retained.')
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('errors', 'counts')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    verify(parser.parse_args().run_id)
