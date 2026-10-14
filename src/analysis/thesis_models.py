"""Chronological thesis replay; no filesystem writes or archive imports.

All input rows, including rows without odds, are eligible for TRAIN. Input
features must already satisfy the caller's information-availability contract.
This module checks label availability, not the upstream feature construction.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models.team_order import pair_columns, swap_orientation, symmetrize_binary_probabilities

RATING_NAMES = ('elo', 'gl', 'ts', 'os', 'pl', 'tm')
RANK_PROBABILITIES = [f'{scope}_{name}' for scope in ('player', 'team') for name in RATING_NAMES]
EPSILON = 1e-6

# Frozen parameters from 06ab; CPU limits are execution settings, not candidates.
FROZEN_PARAMS = {
    'logistic': dict(C=0.03297234640536737, penalty='elasticnet', l1_ratio=0.9439657999531195,
                     solver='saga', max_iter=5000, random_state=42, mask_rate=0.0),
    'extra_trees': dict(n_estimators=797, max_depth=9, min_samples_leaf=23,
                        max_features=0.8, random_state=42, n_jobs=4, mask_rate=0.05),
    'hist_gradient_boosting': dict(max_iter=201, learning_rate=0.016660815547461415,
                                   max_leaf_nodes=12, min_samples_leaf=200,
                                   l2_regularization=0.0002892673145972703,
                                   random_state=42, mask_rate=0.10),
    'lightgbm': dict(max_depth=3, num_leaves=4, learning_rate=0.012805167280103086,
                    n_estimators=414, min_child_samples=147, subsample=0.942480859768491,
                    colsample_bytree=0.8133135296907867, reg_alpha=0.0013112363452929046,
                    reg_lambda=0.022766181282355566, random_state=42, verbosity=-1,
                    n_jobs=4, mask_rate=0.20),
    'mlp': dict(hidden_layer_sizes=(32,), activation='relu', alpha=0.0014014609810153243,
                learning_rate_init=0.004262402727044875, max_iter=300,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                random_state=42, mask_rate=0.0),
}


def make_model(family: str = 'logistic', params: dict | None = None) -> Pipeline:
    """Build one thesis family with TRAIN-only imputation and standardization."""
    effective = {**FROZEN_PARAMS[family], **(params or {})}
    effective.pop('mask_rate', None)
    constructors = {'logistic': LogisticRegression, 'extra_trees': ExtraTreesClassifier,
                    'hist_gradient_boosting': HistGradientBoostingClassifier, 'mlp': MLPClassifier}
    if family == 'lightgbm':
        from lightgbm import LGBMClassifier
        constructors['lightgbm'] = LGBMClassifier
    if family == 'mlp':
        effective['hidden_layer_sizes'] = tuple(effective['hidden_layer_sizes'])
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if family in ('logistic', 'mlp'):
        steps.append(('scaler', StandardScaler()))
    return Pipeline([*steps, ('model', constructors[family](**effective))])


def positive_slope_calibration(probability: np.ndarray, target: np.ndarray) -> float:
    """Fit sigmoid(a logit(p)), a in [0.05, 20], preserving side symmetry."""
    p = np.asarray(probability, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(p) != len(y) or not len(p) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid calibration inputs')
    z = logit(np.clip(p, EPSILON, 1-EPSILON))
    result = minimize_scalar(lambda a: float(np.mean(np.logaddexp(0, a*z)-y*a*z)),
                             bounds=(0.05, 20.0), method='bounded', options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError('Positive calibration slope optimization failed')
    return float(result.x)


def _prepare(data, features, id_column, date_column, release_column, target_column):
    missing = set([*features, id_column, date_column, release_column, target_column])-set(data.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    working = data.copy()
    working[id_column] = working[id_column].astype(str)
    if working[id_column].duplicated().any():
        raise ValueError('Input contains duplicate match IDs')
    for column in (date_column, release_column):
        working[column] = pd.to_datetime(working[column], errors='raise')
        if working[column].isna().any():
            raise ValueError(f'Missing dates in {column}')
    if not working[target_column].isin([0, 1]).all():
        raise ValueError('Targets must be binary')
    numeric = working[features].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError('Infinite features are invalid')
    recognized = set(RANK_PROBABILITIES)
    recognized.update(f for f in features if f.endswith('_binom_series'))
    recognized.update(f for pair in pair_columns(features) for f in pair)
    recognized.update(('BoN', 'best_of'))
    if set(features)-recognized:
        raise ValueError(f'Undefined side orientation for features: {sorted(set(features)-recognized)}')
    return working.sort_values([date_column, id_column], kind='stable').reset_index(drop=True)


def _membership(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(map(str, ids)), separators=(',', ':')).encode()).hexdigest()


def _fit_predict(train, test, features, family, params, target_column, seed):
    model = make_model(family, params)
    mask_rate = float({**FROZEN_PARAMS[family], **(params or {})}['mask_rate'])
    x = train[features].copy()
    if mask_rate:
        x = x.mask(np.random.default_rng(seed).random(x.shape) < mask_rate)
    swapped = swap_orientation(test, features, RANK_PROBABILITIES,
                               np.ones(len(test), dtype=bool), target_column=target_column)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(x, train[target_column].astype(int))
        original = model.predict_proba(test[features])[:, 1]
        opposite = model.predict_proba(swapped[features])[:, 1]
    messages = sorted(set(f'{w.category.__name__}: {w.message}' for w in captured))
    return original, opposite, messages


def replay_monthly(
    data: pd.DataFrame, features: list[str], family: str = 'logistic', params: dict | None = None,
    *, start: str = '2023-01-01', end: str | None = None,
    id_column: str = 'golgg_match_id', date_column: str = 'date',
    release_column: str = 'result_day', target_column: str = 'y_true',
    min_calibration_samples: int = 200, calibration_start: str = '2024-01-01',
    progress: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Train monthly on all strict-past rows, calibrate on strict-past OOS rows.

    End is exclusive. Defaults emit 2023 diagnostic warmup but exclude its
    selected-on-development predictions from calibration. No odds filtering or
    complete-case filtering occurs. Missing features receive TRAIN medians.
    Receipts retain membership IDs and hashes and all fitting warnings.
    """
    working = _prepare(data, features, id_column, date_column, release_column, target_column)
    first = pd.Timestamp(start)
    if first != first.to_period('M').start_time:
        raise ValueError('Replay start must be a calendar month boundary')
    stop = pd.Timestamp(end) if end else working[date_column].max().to_period('M').end_time + pd.Timedelta(nanoseconds=1)
    output, receipts = [], []
    effective = {**FROZEN_PARAMS[family], **(params or {})}
    for fold, origin in enumerate(pd.date_range(first, stop, freq='MS', inclusive='left'), 1):
        next_origin = min(origin + pd.offsets.MonthBegin(1), stop)
        train = working[(working[date_column] < origin) & (working[release_column] < origin)]
        test = working[(working[date_column] >= origin) & (working[date_column] < next_origin)]
        if test.empty:
            continue
        if train[target_column].nunique() != 2:
            raise ValueError(f'TRAIN lacks both classes at {origin.date()}')
        original, opposite, fit_warnings = _fit_predict(train, test, features, family, effective,
                                                       target_column, 42+fold)
        raw = np.clip(symmetrize_binary_probabilities(original, opposite), EPSILON, 1-EPSILON)
        prior = pd.concat(output, ignore_index=True) if output else pd.DataFrame()
        if not prior.empty:
            prior = prior[(pd.to_datetime(prior[date_column]) < origin)
                          & (pd.to_datetime(prior[release_column]) < origin)
                          & (pd.to_datetime(prior[date_column]) >= pd.Timestamp(calibration_start))]
        available = len(prior) >= min_calibration_samples and prior[target_column].nunique() == 2
        slope = positive_slope_calibration(prior.p_raw.to_numpy(), prior[target_column].to_numpy()) if available else 1.0
        out = test[[id_column, date_column, release_column, target_column]].copy()
        out['origin'] = origin.date().isoformat()
        out['family'] = family
        out['p_original'] = original
        out['p_swapped'] = opposite
        out['p_raw'] = raw
        out['p_calibrated'] = expit(slope * logit(raw))
        out['calibrator_available'] = bool(available)
        out['calibration_slope'] = slope
        output.append(out)
        cal_ids = prior[id_column].tolist() if len(prior) else []
        receipt = {'origin': origin.date().isoformat(), 'test_end_exclusive': next_origin.isoformat(),
                   'family': family, 'params': effective.copy(), 'features': list(features),
                   'train_rows': len(train), 'test_rows': len(test), 'cal_rows': len(prior),
                   'train_ids': train[id_column].tolist(), 'test_ids': test[id_column].tolist(), 'cal_ids': cal_ids,
                   'train_ids_sha256': _membership(train[id_column]), 'test_ids_sha256': _membership(test[id_column]),
                   'cal_ids_sha256': _membership(cal_ids),
                   'train_max_date': train[date_column].max().isoformat(),
                   'train_max_release': train[release_column].max().isoformat(),
                   'cal_max_date': prior[date_column].max().isoformat() if len(prior) else None,
                   'cal_max_release': prior[release_column].max().isoformat() if len(prior) else None,
                   'calibration_start': calibration_start, 'calibration_slope': slope,
                   'calibrator_available': bool(available), 'warnings': fit_warnings}
        receipts.append(receipt)
        if progress:
            progress(receipt)
    if not output:
        raise ValueError('No replay targets in the requested period')
    return pd.concat(output, ignore_index=True), receipts


def default_candidate_grid() -> dict[str, list[dict]]:
    """Frozen thesis recipe and two predeclared simpler controls per family."""
    controls = {
        'logistic': [dict(C=.01, l1_ratio=.5), dict(C=.1, l1_ratio=.5)],
        'extra_trees': [dict(n_estimators=300, max_depth=5, min_samples_leaf=50),
                        dict(n_estimators=300, max_depth=7, min_samples_leaf=100)],
        'hist_gradient_boosting': [dict(max_iter=100, max_leaf_nodes=5), dict(max_iter=150, max_leaf_nodes=8)],
        'lightgbm': [dict(n_estimators=150, max_depth=2, num_leaves=3),
                    dict(n_estimators=250, max_depth=2, num_leaves=3)],
        'mlp': [dict(hidden_layer_sizes=(8,), alpha=.01), dict(hidden_layer_sizes=(16,), alpha=.01)],
    }
    return {family: [params.copy(), *[{**params, **c} for c in controls[family]]]
            for family, params in FROZEN_PARAMS.items()}


def select_development_configuration(
    data: pd.DataFrame, features: list[str], *, candidates: dict[str, list[dict]] | None = None,
    development_start: str = '2023-01-01', evaluation_start: str = '2024-01-01',
    id_column: str = 'golgg_match_id', date_column: str = 'date',
    release_column: str = 'result_day', target_column: str = 'y_true',
) -> dict:
    """Fixed-origin development selection; never accesses evaluation labels.

    One fit per candidate before development_start, evaluated only on rows
    whose date and label release are before evaluation_start. Uses raw
    side-symmetrized log loss, no TEST-selected calibration. Fits all candidates
    on the identical full-history TRAIN and reports all scores, not just winner.
    """
    earlier = data[pd.to_datetime(data[date_column]) < pd.Timestamp(evaluation_start)].copy()
    earlier = earlier[pd.to_datetime(earlier[release_column]) < pd.Timestamp(evaluation_start)]
    working = _prepare(earlier, features, id_column, date_column, release_column, target_column)
    origin = pd.Timestamp(development_start)
    train = working[(working[date_column] < origin) & (working[release_column] < origin)]
    validation = working[working[date_column] >= origin]
    if train[target_column].nunique() != 2 or validation.empty:
        raise ValueError('Development split lacks training classes or validation rows')
    grid = candidates if candidates is not None else default_candidate_grid()
    records, selected = [], {}
    y = validation[target_column].to_numpy(dtype=float)
    for family, configs in grid.items():
        family_records = []
        for index, config in enumerate(configs):
            effective = {**FROZEN_PARAMS[family], **config}
            original, opposite, messages = _fit_predict(train, validation, features, family, effective, target_column, 42)
            p = np.clip(symmetrize_binary_probabilities(original, opposite), EPSILON, 1-EPSILON)
            record = dict(family=family, candidate=index, params=effective,
                          logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
                          brier=float(np.mean((p-y)**2)), warnings=messages)
            records.append(record)
            family_records.append(record)
        selected[family] = min(family_records, key=lambda r: (r['logloss'], r['candidate']))['params']
    return dict(selected=selected, candidates=records, development_start=development_start,
                evaluation_start=evaluation_start, train_rows=len(train), validation_rows=len(validation),
                train_ids_sha256=_membership(train[id_column]), validation_ids_sha256=_membership(validation[id_column]),
                train_ids=train[id_column].tolist(), validation_ids=validation[id_column].tolist(),
                train_max_release=train[release_column].max().isoformat(),
                validation_max_release=validation[release_column].max().isoformat(),
                interpretation='Historical development selection; not a new protected prospective holdout.')
