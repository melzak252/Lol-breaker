"""Mean-preserving, separately frozen duration dispersion experiment."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from src.models.prop_distribution import DistributionPrediction

COLUMNS = ['x_pace_player_duration_minutes_count_mean_level', 'x_pace_player_duration_minutes_mean_gap', 'x_pace_player_missing_duration_minutes_level']

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()

def _vector(value, name, n=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (n is not None and len(result) != n) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite aligned vector')
    return result

@dataclass
class ScalePreprocessor:
    gap_median: float
    mean: np.ndarray
    scale: np.ndarray

    @staticmethod
    def _raw(frame, median=None):
        count = _vector(frame[COLUMNS[0]], 'count')
        gap = np.asarray(frame[COLUMNS[1]], dtype=float)
        missing = _vector(frame[COLUMNS[2]], 'missingness', len(count))
        if np.any(count < 0) or np.any(missing < 0) or np.any(missing > 5):
            raise ValueError('invalid support or missingness')
        if np.isinf(gap).any() or np.any(gap[np.isfinite(gap)] < 0):
            raise ValueError('invalid gap support')
        if median is None:
            if not np.isfinite(gap).any(): raise ValueError('no observed TRAIN gap')
            median = float(np.nanmedian(gap))
        return np.column_stack((np.log1p(count), np.where(np.isnan(gap), median, gap), missing)), median

    @classmethod
    def fit(cls, train_frame):
        raw, median = cls._raw(train_frame)
        if len(raw) == 0 or np.any(np.std(raw, axis=0) == 0): raise ValueError('TRAIN covariate variance is zero')
        pre = cls(median, raw[:, :2].mean(0), raw[:, :2].std(0))
        design = np.column_stack((np.ones(len(raw)), pre.transform(train_frame)))
        if np.linalg.matrix_rank(design) != 4: raise ValueError('TRAIN augmented design rank deficient')
        return pre

    def transform(self, frame):
        raw, _ = self._raw(frame, self.gap_median)
        raw[:, :2] = np.clip((raw[:, :2] - self.mean) / self.scale, -3, 3)
        return raw

def _observations(mu0, y):
    mu = _vector(mu0, 'mu0')
    obs = _vector(y, 'duration', len(mu))
    if not len(mu) or np.any(obs <= 0): raise ValueError('duration observations must be finite positive')
    return mu, obs

def train_reference_sigma(mu0, y):
    mu, obs = _observations(mu0, y)
    sigma = float(np.sqrt(np.mean((np.log(obs) - mu) ** 2)))
    if not np.isfinite(sigma) or sigma <= 0: raise ValueError('TRAIN residual sigma must be positive')
    return sigma

def mean_preserving_prediction(base, log_scale):
    offset = _vector(log_scale, 'log scale', len(base.duration_log_mu))
    sigma = np.clip(base.duration_log_sigma * np.exp(np.clip(offset, -700, 700)), 1e-4, 5.)
    mu = base.duration_log_mu + .5 * (base.duration_log_sigma ** 2 - sigma ** 2)
    return DistributionPrediction(base.kills_mu.copy(), base.kills_alpha.copy(), mu, sigma,
                                  base.length_probs.copy(), base.best_of.copy())

def _loss(offset, mu0, y, anchor):
    mu, obs = _observations(mu0, y)
    if not np.isfinite(anchor) or anchor <= 0: raise ValueError('anchor sigma must be positive finite')
    log_sigma = np.log(anchor) + offset
    sigma = np.exp(np.clip(log_sigma, np.log(1e-4), np.log(5.)))
    residual = np.log(obs) - mu - .5 * anchor ** 2 + .5 * sigma ** 2
    loss = np.log(obs) + np.log(sigma) + .5 * np.log(2 * np.pi) + .5 * (residual / sigma) ** 2
    derivative = 1 + residual - (residual / sigma) ** 2
    derivative *= (log_sigma > np.log(1e-4)) & (log_sigma < np.log(5.))
    return float(loss.mean()), derivative

def _design(z, n):
    z = np.asarray(z, dtype=float)
    if z.shape != (n, 3) or not np.isfinite(z).all(): raise ValueError('finite aligned three-column design required')
    return z

def train_scale_objective(theta, z, mu0, y, sigma_T):
    theta = _vector(theta, 'theta', 4)
    z = _design(z, len(mu0))
    loss, derivative = _loss(theta[0] + z @ theta[1:], mu0, y, sigma_T)
    gradient = np.r_[derivative.mean(), z.T @ derivative / len(z)]
    gradient[1:] += .2 * theta[1:]
    return loss + .1 * float(theta[1:] @ theta[1:]), gradient

def cal_scale_objective(c, z, mu0, y, sigma0, slopes):
    c = float(c)
    if not np.isfinite(c): raise ValueError('CAL parameter must be finite')
    z = _design(z, len(mu0)); slopes = _vector(slopes, 'slopes', 3)
    loss, derivative = _loss(c + z @ slopes, mu0, y, sigma0)
    return loss + .001 * c * c, float(derivative.mean() + .002 * c)

def accept_fit_result(result, *, stage):
    if stage not in ('TRAIN_HEAD', 'PG_CAL', 'PC_CAL'): raise ValueError('unknown fit stage')
    width = 4 if stage == 'TRAIN_HEAD' else 1
    lower, upper = (-1., 1.) if width == 4 else (np.log(.25), np.log(4.))
    x = _vector(np.atleast_1d(result.x), 'fit parameters', width)
    jac = _vector(np.atleast_1d(result.jac), 'fit gradient', width)
    if not result.success or not np.isfinite(result.fun): raise ValueError('fit failed or nonfinite objective')
    if np.any(x < lower) or np.any(x > upper): raise ValueError('fit parameter outside bound')
    if not 0 <= result.nit <= 500: raise ValueError('fit iteration budget exceeded')
    projected = jac.copy()
    projected[((x == lower) & (jac > 0)) | ((x == upper) & (jac < 0))] = 0
    norm = float(np.max(np.abs(projected)))
    if norm > 1e-5: raise ValueError('fit did not converge: projected gradient')
    return dict(status='success', stage=stage, parameters=x.tolist(), objective=float(result.fun), gradient=jac.tolist(),
                iterations=int(result.nit), nfev=int(result.nfev), message=str(result.message),
                boundary_hit=bool(np.any(np.isclose(x, lower, atol=1e-8, rtol=0) | np.isclose(x, upper, atol=1e-8, rtol=0))),
                projected_gradient_inf=norm, projected_gradient=projected.tolist(), estimation_operations=1, bounds=[lower, upper],
                optimizer='L-BFGS-B' if width == 4 else 'bounded', start=[0.] * 4 if width == 4 else None,
                options={'maxiter':500,'ftol':1e-12,'gtol':1e-8} if width == 4 else {'maxiter':500,'xatol':1e-8})

@dataclass
class FrozenDurationTree:
    model: object
    feature_columns: list
    selected_id: str
    source_sha256: str

    def predict(self, frame):
        return self.model.predict(frame[self.feature_columns].to_numpy(dtype=float), frame['best_of'].to_numpy())

def load_selected_tree(bundle_path, *, expected_sha256, selected_id='tree_leaves_15', feature_columns):
    if sha256(bundle_path) != expected_sha256: raise ValueError('selected bundle hash pin mismatch')
    bundle = joblib.load(bundle_path)
    if bundle.selected.get('duration') != selected_id or selected_id not in bundle.models:
        raise ValueError('selected tree identity mismatch')
    tree = bundle.models[selected_id]
    if tree.family != 'tree' or list(bundle.feature_columns) != list(feature_columns): raise ValueError('selected tree/schema mismatch')
    if not np.isfinite(tree.duration_sigma) or tree.duration_sigma <= 0: raise ValueError('invalid saved CAL sigma')
    return FrozenDurationTree(tree, list(feature_columns), selected_id, expected_sha256)

@dataclass(kw_only=True)
class DurationScaleModel:
    tree: FrozenDurationTree
    preprocessor: ScalePreprocessor
    slopes: np.ndarray
    c_global: float
    c_conditional: float
    train_reference_sigma: float
    train_intercept: float
    fit_receipt: dict
    def __post_init__(self):
        self.slopes = _vector(self.slopes, 'slopes', 3).copy()
        if np.any(np.abs(self.slopes) > 1) or not np.isfinite(self.train_intercept) or abs(self.train_intercept) > 1:
            raise ValueError('TRAIN head parameters outside frozen bounds')
        for c in [self.c_global, self.c_conditional]:
            if not np.isfinite(c) or not np.log(.25) <= c <= np.log(4):
                raise ValueError('CAL parameter outside frozen bounds')
        if not np.isfinite(self.train_reference_sigma) or self.train_reference_sigma <= 0:
            raise ValueError('TRAIN reference scale must be positive finite')


    def predict(self, frame, *, arm='PC'):
        base = self.tree.predict(frame)
        if arm == 'P0': return base
        if arm == 'PG': offset = np.full(len(frame), self.c_global)
        elif arm == 'PC': offset = self.c_conditional + self.preprocessor.transform(frame) @ self.slopes
        else: raise ValueError('unknown duration arm')
        return mean_preserving_prediction(base, offset)

    def save(self, path):
        with Path(path).open('xb') as stream: joblib.dump(self, stream)

def load_duration_scale(path, *, expected_sha256):
    if sha256(path) != expected_sha256: raise ValueError('duration artifact hash pin mismatch')
    model = joblib.load(path)
    if not isinstance(model, DurationScaleModel): raise ValueError('invalid duration artifact type')
    return model
