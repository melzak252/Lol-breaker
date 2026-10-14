"""TRAIN-only bounded three-class historical lineup laws (R1).

The OTHER class is not a roster and cannot supply a native winner forecast.
The fixed recipe has no calibration, restart or context-specific parameters.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

CLASS_ORDER = ('LATEST', 'PREVIOUS', 'OTHER')
CONDITIONAL_ORDER = ('a1', 'b1', 'd1', 'a2', 'b2', 'd2')
GLOBAL_ORDER = ('a1', 'a2')


def _design(theta, counts, ages, previous_available, conditional):
    theta = np.asarray(theta, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    ages = np.asarray(ages, dtype=np.float64)
    mask = np.asarray(previous_available)
    size = 6 if conditional else 2
    if theta.shape != (size,) or not np.isfinite(theta).all():
        raise ValueError('Invalid finite theta shape')
    if counts.ndim != 2 or counts.shape[1:] != (2,) or not len(counts) or ages.shape != counts.shape:
        raise ValueError('Nonempty Nx2 count and age inputs required')
    if mask.shape != (len(counts),) or mask.dtype.kind != 'b':
        raise ValueError('previous_available must be an N boolean mask')
    if not np.isfinite(counts).all() or not np.isfinite(ages).all() or (counts < 0).any() or (ages < 0).any():
        raise ValueError('Counts and ages must be finite nonnegative')
    if (counts[:, 0] <= 0).any() or (counts[mask, 1] <= 0).any():
        raise ValueError('Available candidates require positive count')
    if conditional:
        first = np.column_stack((np.ones(len(counts)), np.log1p(counts[:, 0]), np.log1p(ages[:, 0])))
        second = np.column_stack((np.ones(len(counts)), np.log1p(counts[:, 1]), np.log1p(ages[:, 1])))
        logits = np.column_stack((first @ theta[:3], second @ theta[3:], np.zeros(len(counts))))
    else:
        first = second = np.ones((len(counts), 1))
        logits = np.tile([theta[0], theta[1], 0.], (len(counts), 1))
    logits[~mask, 1] = -np.inf
    return theta, logits, first, second, mask


def lineup_probabilities(theta, counts, ages, previous_available, *, conditional):
    _, logits, _, _, _ = _design(theta, counts, ages, previous_available, conditional)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def lineup_objective(theta, counts, ages, previous_available, labels, *, conditional):
    theta, logits, first, second, mask = _design(theta, counts, ages, previous_available, conditional)
    labels = np.asarray(labels)
    if labels.shape != (len(logits),) or labels.dtype.kind not in 'iu' or ((labels < 0) | (labels > 2)).any():
        raise ValueError('Labels must be N integers in [0,2]')
    if ((labels == 1) & ~mask).any():
        raise ValueError('PREVIOUS label is unavailable under structural mask')
    logq = logits - logsumexp(logits, axis=1, keepdims=True)
    residual = np.exp(logq)
    residual[np.arange(len(labels)), labels] -= 1
    gradient = np.concatenate((first.T @ residual[:, 0], second.T @ residual[:, 1])) / len(labels)
    gradient += .2 * theta
    objective = -float(logq[np.arange(len(labels)), labels].mean()) + .1 * float(theta @ theta)
    return objective, gradient


def rows_to_arrays(rows):
    rows = list(rows)
    if not rows:
        raise ValueError('No eligible TRAIN side labels')
    counts = np.asarray([r['counts'] for r in rows], dtype=np.float64)
    ages = np.asarray([r['ages'] for r in rows], dtype=np.float64)
    mask = np.asarray([r['previous_available'] for r in rows], dtype=bool)
    labels = np.asarray([CLASS_ORDER.index(r['label']) if isinstance(r['label'], str) else r['label'] for r in rows], dtype=np.int64)
    return counts, ages, mask, labels


def projected_gradient(theta, gradient):
    theta, gradient = np.asarray(theta), np.asarray(gradient)
    return np.where(((theta <= -5) & (gradient > 0)) | ((theta >= 5) & (gradient < 0)), 0., gradient)


def fit_lineup_law(training_rows, *, conditional):
    """One optimizer invocation; caller MUST consume/persist its slot beforehand."""
    rows = list(training_rows)
    if any(r.get('partition', 'TRAIN') != 'TRAIN' for r in rows):
        raise ValueError('Only TRAIN rows may estimate lineup parameters')
    counts, ages, mask, labels = rows_to_arrays(rows)
    order = CONDITIONAL_ORDER if conditional else GLOBAL_ORDER
    initial = np.zeros(len(order), dtype=np.float64)
    started = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=4):
        result = minimize(
            lambda t: lineup_objective(t, counts, ages, mask, labels, conditional=conditional),
            initial, jac=True, method='L-BFGS-B', bounds=[(-5., 5.)] * len(order),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    objective, gradient = lineup_objective(result.x, counts, ages, mask, labels, conditional=conditional)
    residual = float(np.max(np.abs(projected_gradient(result.x, gradient))))
    finite = bool(np.isfinite(result.x).all() and np.isfinite(gradient).all() and np.isfinite(objective))
    within_bounds = bool(((result.x >= -5) & (result.x <= 5)).all())
    probabilities_finite = bool(np.isfinite(lineup_probabilities(result.x, counts, ages, mask, conditional=conditional)).all())
    accepted = finite and within_bounds and probabilities_finite and bool(result.success) and int(result.nit) <= 500 and residual <= 1e-5
    membership = [{'match_id': r['match_id'], 'side': r['side'], 'label': r['label'],
                   'release_lineage': r.get('release_lineage'), 'history_ids': r.get('history_ids')} for r in rows]
    receipt = {'schema': 'lineup_law_reliability_model_v1', 'conditional': conditional,
               'theta_order': list(order), 'theta': result.x.tolist(), 'initial_theta': initial.tolist(),
               'bounds': [[-5., 5.]] * len(order), 'ridge': .1, 'optimizer': 'L-BFGS-B',
               'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'iterations': int(result.nit),
               'function_evaluations': int(result.nfev), 'success': bool(result.success),
               'status': int(result.status), 'message': str(result.message), 'objective': objective,
               'mean_nll': objective - .1 * float(result.x @ result.x), 'gradient': gradient.tolist(),
               'projected_gradient_infinity': residual,
               'boundary_hit': ((result.x <= -5) | (result.x >= 5)).tolist(),
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'accepted': accepted, 'training_membership': membership,
               'training_keys': [[r['match_id'], r['side']] for r in rows]}
    return receipt


class LineupLaw:
    """A saved law needs candidate features, never the target's actual lineup."""
    def __init__(self, artifact):
        self.artifact = artifact
        if type(artifact.get('conditional')) is not bool:
            raise ValueError('Saved conditional flag must be boolean')
        self.conditional = artifact['conditional']
        self.theta = np.asarray(artifact['theta'], dtype=np.float64)
        if (artifact.get('accepted') is not True or self.theta.shape != (6 if self.conditional else 2,)
                or not np.isfinite(self.theta).all() or ((self.theta < -5) | (self.theta > 5)).any()):
            raise ValueError('Invalid saved fit cannot predict')

    def predict(self, counts, ages, previous_available):
        return lineup_probabilities(self.theta, counts, ages, previous_available, conditional=self.conditional)

    @classmethod
    def load(cls, path, *, sha256=None):
        content = Path(path).read_bytes()
        if sha256 is not None and hashlib.sha256(content).hexdigest() != sha256:
            raise ValueError('Saved lineup law hash mismatch')
        return cls(json.loads(content))
