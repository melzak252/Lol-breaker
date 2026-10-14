"""EXP-083 frozen-stage series estimators, never a production model registry.

Columns are ordered odd differences first, then swap-invariant context. Callers
own chronological, date-grouped, disjoint train -> stop -> select -> calibration
partitions: arrays alone cannot establish their timestamps or provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

MODEL_VERSION = "exp083-series-estimators-v1"
_C_VALUES = (0.01, 0.1, 1.0)
_BLEND_WEIGHTS = (0.0, 0.1, 0.25, 0.5)
_SPLINE_INPUTS = (
    "d_rating_consensus_logit",
    "d_team_glicko_logit",
    "d_player_glicko_logit",
    "d_form_w20",
    "d_w20_win_rate_diff",
    "d_w20_gd15_diff",
)
_EPS = np.finfo(np.float64).eps


def _matrix(value: np.ndarray, name: str, n_features: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != n_features or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty matrix with {n_features} columns")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite real numeric features")
    return np.asarray(array, dtype=np.float64)


def _labels(value: np.ndarray, name: str, n_rows: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != n_rows or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional binary label array")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return np.asarray(array, dtype=np.float64)


def _swap(x: np.ndarray, n_odd: int) -> np.ndarray:
    swapped = x.copy()
    swapped[:, :n_odd] *= -1.0
    return swapped


def _open_probability(p: np.ndarray) -> np.ndarray:
    """Bound floating-point saturation only; never repair invalid input data."""
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("model produced invalid probabilities")
    return np.clip(p, _EPS, 1.0 - _EPS)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = _open_probability(p)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def _project_probability(q: np.ndarray, q_swap: np.ndarray) -> np.ndarray:
    # Algebraically (q + 1 - q_swap)/2; difference-first gives exactly .5
    # for neutral odd evidence, even when the tree itself has a side bias.
    return _open_probability(0.5 + 0.5 * (q - q_swap))


def _paired_projected_log_loss(
    y: np.ndarray, prediction: np.ndarray
) -> tuple[str, float, bool]:
    """LightGBM callback for contiguous [original, swapped] stopping rows."""
    n = len(y) // 2
    if n == 0 or len(y) != 2 * n or not np.array_equal(y[n:], 1.0 - y[:n]):
        raise ValueError(
            "projected stopping metric requires aligned original/swapped pairs"
        )
    return (
        "projected_log_loss",
        _log_loss(y[:n], _project_probability(prediction[:n], prediction[n:])),
        False,
    )


@dataclass
class _OddBasis:
    n_odd: int
    feature_names: tuple[str, ...]
    context_indices: tuple[int, ...] = ()
    spline_inputs: tuple[str, ...] = ()
    spline: SplineTransformer | None = None
    scaler: StandardScaler | None = None
    constant_spline_inputs: tuple[str, ...] = ()

    def _unscaled(self, x: np.ndarray) -> np.ndarray:
        odd = x[:, : self.n_odd]
        blocks = [odd]
        if self.spline is not None:
            indices = [self.feature_names.index(name) for name in self.spline_inputs]
            values = x[:, indices]
            blocks.append(
                (self.spline.transform(values) - self.spline.transform(-values)) * 0.5
            )
        evidence = np.concatenate(blocks, axis=1) if len(blocks) > 1 else odd
        context = x[:, self.context_indices]
        # Only declared format interactions; other context is available to trees.
        if context.shape[1]:
            interactions = (evidence[:, :, None] * context[:, None, :]).reshape(
                len(x), -1
            )
            evidence = np.concatenate((evidence, interactions), axis=1)
        if not np.isfinite(evidence).all():
            raise ValueError("odd feature interactions overflowed")
        return evidence

    def fit_transform(self, x: np.ndarray, *, use_splines: bool) -> np.ndarray:
        if use_splines:
            named = tuple(
                name
                for name in _SPLINE_INPUTS
                if name in self.feature_names[: self.n_odd]
            )[:4]
            if not named:
                raise ValueError(
                    "odd_spline requires at least one predeclared named odd input"
                )
            # A zero-only signal has no distinct knots and no estimable spline.
            self.constant_spline_inputs = tuple(
                name
                for name in named
                if np.all(x[:, self.feature_names.index(name)] == 0.0)
            )
            self.spline_inputs = tuple(
                name for name in named if name not in self.constant_spline_inputs
            )
            if self.spline_inputs:
                values = x[
                    :, [self.feature_names.index(name) for name in self.spline_inputs]
                ]
                self.spline = SplineTransformer(
                    n_knots=5,
                    degree=3,
                    knots="uniform",
                    extrapolation="constant",
                    include_bias=False,
                ).fit(np.concatenate((values, -values), axis=0))
        evidence = self._unscaled(x)
        self.scaler = StandardScaler(with_mean=False)
        result = self.scaler.fit_transform(evidence)
        if not np.isfinite(self.scaler.scale_).all() or not np.isfinite(result).all():
            raise ValueError(
                "training-only scaling produced non-finite scales or features"
            )
        return result

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("odd basis has not been fitted")
        result = self.scaler.transform(self._unscaled(x))
        if not np.isfinite(result).all():
            raise ValueError("scaling produced non-finite features")
        return result


@dataclass
class SeriesEstimator:
    """Frozen base predictor plus one final positive, zero-intercept slope.

    ``load`` accepts only trusted, locally generated joblib artifacts. Pickle-based
    joblib is executable serialization, not a format for external uploads.
    """

    feature_names: tuple[str, ...]
    n_odd: int
    kind: str
    base: object
    basis: _OddBasis | None = None
    slope: float = 1.0
    metadata: dict = field(default_factory=dict)
    blend_weight: float = 0.0

    def _raw(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "linear":
            z = self.base.decision_function(self.basis.transform(x))
            if not np.isfinite(z).all():
                raise ValueError("linear model produced non-finite logits")
            return _open_probability(expit(z))
        if self.kind == "tree":
            q = self.base.predict(x, num_threads=2)
            q_swap = self.base.predict(_swap(x, self.n_odd), num_threads=2)
            return _project_probability(q, q_swap)
        if self.kind == "blend":
            ridge, tree = self.base
            return _open_probability(
                (1.0 - self.blend_weight) * ridge._raw(x)
                + self.blend_weight * tree._raw(x)
            )
        raise ValueError(f"unsupported estimator kind: {self.kind}")

    def raw_probability(self, x: np.ndarray) -> np.ndarray:
        """Uncalibrated deployed probability, including projection/blending."""
        return self._raw(_matrix(x, "prediction data", len(self.feature_names)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.slope) or self.slope <= 0.0:
            raise ValueError("calibration slope must be finite and positive")
        return _open_probability(expit(self.slope * logit(self.raw_probability(x))))

    def save(self, path: Path) -> None:
        """Exclusively create a trusted local artifact; never overwrite a model."""
        path = Path(path)
        with path.open("xb") as handle:
            joblib.dump(self, handle, compress=3)

    @classmethod
    def load(cls, path: Path) -> SeriesEstimator:
        """Load a trusted local artifact created by ``save`` (never external pickle)."""
        with Path(path).open("rb") as handle:
            model = joblib.load(handle)
        if (
            not isinstance(model, cls)
            or model.metadata.get("model_version") != MODEL_VERSION
        ):
            raise ValueError("artifact is not a compatible EXP-083 series estimator")
        return model


def _calibrate(model: SeriesEstimator, x: np.ndarray, y: np.ndarray) -> None:
    # Reuse the established optimizer without moving or changing legacy training.
    from scripts.train_and_tune_siamese_series import fit_platt_scaling

    z = logit(model.raw_probability(x))
    model.slope = fit_platt_scaling(z, y)
    magnitude = float(np.max(np.abs(z)))
    normalized_log_slope = float(np.log(model.slope * magnitude)) if magnitude else None
    status = "interior"
    if magnitude == 0.0:
        status = "unidentifiable_zero_logits"
    elif normalized_log_slope <= -20.0 + 1e-4:
        status = "lower_numerical_boundary"
    elif normalized_log_slope >= 20.0 - 1e-4:
        status = "upper_numerical_boundary"
    model.metadata.update(
        {
            "slope": model.slope,
            "calibration": {
                "method": "positive_zero_intercept_logit_slope",
                "fit_stage": "calibration",
                "target": "actual_raw_probability",
                "status": status,
                "normalized_log_slope": normalized_log_slope,
                "normalized_log_slope_bounds": [-20.0, 20.0],
                "fit_log_loss": _log_loss(y, model.predict(x)),
            },
        }
    )


def fit_candidates(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xstop: np.ndarray,
    ystop: np.ndarray,
    Xselect: np.ndarray,
    yselect: np.ndarray,
    Xcal: np.ndarray,
    ycal: np.ndarray,
    *,
    n_odd: int,
    feature_names: list[str] | tuple[str, ...],
    seed: int = 83,
    max_rounds: int = 600,
) -> dict[str, SeriesEstimator]:
    """Fit the locked search; stop selects bases, select picks blend, cal fits slopes.

    No split/shuffle/refit is performed. The caller must supply disjoint whole-date
    partitions in chronological order and record their date/series boundaries.
    """
    names = tuple(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature_names must be nonempty names")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if (
        isinstance(n_odd, (bool, np.bool_))
        or not isinstance(n_odd, (int, np.integer))
        or not 1 <= n_odd <= len(names)
    ):
        raise ValueError("n_odd must identify a nonempty leading block of odd features")
    if any(not name.startswith("d_") for name in names[:n_odd]) or any(
        not name.startswith("c_") for name in names[n_odd:]
    ):
        raise ValueError("schema must order d_* odd features before c_* even context")
    if (
        isinstance(max_rounds, (bool, np.bool_))
        or not isinstance(max_rounds, (int, np.integer))
        or not 1 <= max_rounds <= 600
    ):
        raise ValueError("max_rounds must be an integer in [1, 600]")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer in [0, 2**32)")
    stages = {}
    for stage, x, y in (
        ("train", Xtrain, ytrain),
        ("stop", Xstop, ystop),
        ("select", Xselect, yselect),
        ("calibration", Xcal, ycal),
    ):
        matrix = _matrix(x, f"{stage} data", len(names))
        stages[stage] = matrix, _labels(y, f"{stage} labels", len(matrix))
    train, y_train = stages["train"]
    stop, y_stop = stages["stop"]
    select, y_select = stages["select"]
    cal, y_cal = stages["calibration"]
    if len(np.unique(y_train)) != 2:
        raise ValueError("training labels must contain both classes")

    import lightgbm as lgb

    common = {
        "model_version": MODEL_VERSION,
        "feature_names": list(names),
        "n_odd": int(n_odd),
        "schema": {
            "odd": list(names[:n_odd]),
            "even": list(names[n_odd:]),
            "swap": "(-d,c)",
        },
        "seed": int(seed),
        "loss": "binary_log_loss",
        "refit": False,
        "stage_rows": {stage: len(values[0]) for stage, values in stages.items()},
        "stage_contract": "caller supplies disjoint chronological whole-date partitions",
        "preprocessing_fit_stage": "train",
        "base_selection_stage": "stop",
        "probability_saturation_epsilon": float(_EPS),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "scikit-learn", "lightgbm", "joblib")
        },
    }
    models = {}
    format_indices = tuple(
        i for i, name in enumerate(names) if name in ("c_bo3", "c_bo5")
    )
    for family, use_splines in (
        ("fixed_ridge", False),
        ("ridge", False),
        ("odd_spline", True),
    ):
        basis = _OddBasis(
            int(n_odd),
            names,
            context_indices=() if family == "fixed_ridge" else format_indices,
        )
        transformed = basis.fit_transform(train, use_splines=use_splines)
        stop_transformed = basis.transform(stop)
        search, fitted = [], []
        for c in ((0.1,) if family == "fixed_ridge" else _C_VALUES):
            linear = LogisticRegression(
                C=c,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=10000,
                tol=1e-8,
                random_state=int(seed),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                linear.fit(transformed, y_train)
            stop_loss = _log_loss(
                y_stop, expit(linear.decision_function(stop_transformed))
            )
            search.append(
                {
                    "C": c,
                    "stop_log_loss": stop_loss,
                    "optimizer_iterations": int(linear.n_iter_[0]),
                }
            )
            fitted.append(linear)
        selected = min(
            range(len(search)), key=lambda index: search[index]["stop_log_loss"]
        )
        basis_metadata = {
            "basis": "odd_spline" if use_splines else "odd_linear",
            "center": False,
            "scales": basis.scaler.scale_.tolist(),
            "even_context_interactions": [names[i] for i in basis.context_indices],
            "spline_inputs": list(basis.spline_inputs),
            "zero_only_spline_inputs_omitted": list(basis.constant_spline_inputs),
            "spline": (
                {
                    "degree": 3,
                    "n_knots": 5,
                    "knots": "uniform",
                    "extrapolation": "constant",
                    "fit_rows": "train_and_negative_train",
                    "projection": "(B(d)-B(-d))/2",
                }
                if use_splines
                else None
            ),
        }
        models[family] = SeriesEstimator(
            names,
            int(n_odd),
            "linear",
            fitted[selected],
            basis,
            metadata={
                **common,
                **basis_metadata,
                "candidate": family,
                "search_configs": search,
                "chosen_config": search[selected],
                "stop_log_loss": search[selected]["stop_log_loss"],
                "chosen_iteration": None,
                "base_selection_stage": (
                    "fixed_predeclared" if family == "fixed_ridge" else "stop"
                ),
            },
        )

    augmented_train = np.concatenate((train, _swap(train, int(n_odd))), axis=0)
    augmented_y = np.concatenate((y_train, 1.0 - y_train))
    augmented_stop = np.concatenate((stop, _swap(stop, int(n_odd))), axis=0)
    augmented_stop_y = np.concatenate((y_stop, 1.0 - y_stop))
    pair_weights = np.full(len(augmented_train), 0.5)
    tree_search, trees = [], []
    for depth, leaves in ((2, 4), (3, 8)):
        for min_child in (100, 300):
            params = {
                "max_depth": depth,
                "num_leaves": leaves,
                "min_child_samples": min_child,
                "learning_rate": 0.03,
                "reg_lambda": 10.0,
                "reg_alpha": 0.0,
                "max_bin": 63,
                "n_estimators": int(max_rounds),
                "objective": "binary",
                "metric": "None",
                "boosting_type": "gbdt",
                "random_state": int(seed),
                "n_jobs": 2,
                "device_type": "cpu",
                "deterministic": True,
                "force_col_wise": True,
                "colsample_bytree": 1.0,
                "subsample": 1.0,
                "subsample_freq": 0,
                "verbosity": -1,
            }
            tree = lgb.LGBMClassifier(**params)
            tree.fit(
                augmented_train,
                augmented_y,
                sample_weight=pair_weights,
                eval_set=[(augmented_stop, augmented_stop_y)],
                eval_metric=_paired_projected_log_loss,
                callbacks=[
                    lgb.early_stopping(50, first_metric_only=True, verbose=False)
                ],
            )
            iteration = int(tree.best_iteration_)
            if iteration <= 0:
                raise ValueError("LightGBM did not select a stopping iteration")
            # Trim unused later rounds; inference cannot accidentally use a refit or
            # the patience tail. Booster-only artifacts avoid sklearn feature-name warnings.
            frozen = lgb.Booster(
                model_str=tree.booster_.model_to_string(num_iteration=iteration)
            )
            probability = _project_probability(
                frozen.predict(stop, num_threads=2),
                frozen.predict(_swap(stop, int(n_odd)), num_threads=2),
            )
            tree_search.append(
                {
                    **params,
                    "chosen_iteration": iteration,
                    "stop_log_loss": _log_loss(y_stop, probability),
                }
            )
            trees.append(frozen)
    selected = min(
        range(len(tree_search)), key=lambda index: tree_search[index]["stop_log_loss"]
    )
    models["lightgbm"] = SeriesEstimator(
        names,
        int(n_odd),
        "tree",
        trees[selected],
        metadata={
            **common,
            "candidate": "lightgbm",
            "search_configs": tree_search,
            "chosen_config": tree_search[selected],
            "stop_log_loss": tree_search[selected]["stop_log_loss"],
            "chosen_iteration": tree_search[selected]["chosen_iteration"],
            "early_stopping_metric": "paired_projected_log_loss",
            "patience": 50,
            "pair_weight": 0.5,
            "projection": "(q(x)+1-q(Sx))/2",
        },
    )
    ridge, tree = models["ridge"], models["lightgbm"]
    p_ridge, p_tree = ridge.raw_probability(select), tree.raw_probability(select)
    blend_search = [
        {
            "tree_weight": weight,
            "select_log_loss": _log_loss(
                y_select, (1.0 - weight) * p_ridge + weight * p_tree
            ),
        }
        for weight in _BLEND_WEIGHTS
    ]
    chosen_blend = min(blend_search, key=lambda config: config["select_log_loss"])
    models["stack"] = SeriesEstimator(
        names,
        int(n_odd),
        "blend",
        (ridge, tree),
        blend_weight=chosen_blend["tree_weight"],
        metadata={
            **common,
            "candidate": "stack",
            "search_configs": blend_search,
            "chosen_config": chosen_blend,
            "blend_weight": chosen_blend["tree_weight"],
            "blend_selection_stage": "select",
            "members": ["ridge", "lightgbm"],
            "members_are_uncalibrated": True,
            "chosen_iteration": None,
        },
    )
    for model in models.values():
        _calibrate(model, cal, y_cal)
    return models
