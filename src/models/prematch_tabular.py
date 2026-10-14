"""Fresh, side-symmetric tabular estimators for prematch series forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class _SidePreprocessor:
    """TRAIN-fitted shared transforms for a flattened collection of side fields."""

    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()

    def fit(self, sides: np.ndarray, names: tuple[str, ...]) -> "_SidePreprocessor":
        # sides is [N, 2, D]; a single set of parameters is estimated over both sides.
        stacked = sides.reshape(-1, sides.shape[-1]).astype(np.float64, copy=False)
        valid = ~np.isnan(stacked)
        counts = valid.sum(axis=0)
        means = np.divide(
            np.where(valid, stacked, 0.0).sum(axis=0),
            counts,
            out=np.zeros(stacked.shape[1], dtype=np.float64),
            where=counts != 0,
        )
        filled = np.where(np.isnan(stacked), means, stacked)
        scales = np.sqrt(np.mean((filled - means) ** 2, axis=0))
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        self.means = means
        self.scales = scales
        self.feature_names = names
        return self

    def transform(self, sides: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise RuntimeError("side preprocessor has not been fitted")
        missing = np.isnan(sides)
        filled = np.where(missing, self.means[None, None, :], sides)
        values = (filled - self.means[None, None, :]) / self.scales[None, None, :]
        indicators = missing.astype(np.float64)
        # Retain missingness while using the same scale on both sides. No centering
        # is applied to indicators, so identical side values remain identical.
        return np.concatenate((values, indicators), axis=-1)


def _validate_batch(batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("players", "teams", "context", "baseline_series")
    absent = [key for key in required if key not in batch]
    if absent:
        raise ValueError(f"batch is missing required arrays: {', '.join(absent)}")
    specs = {"players": (4, (2, 5)), "teams": (3, (2,)), "context": (2, (3,)), "baseline_series": (2, (2,))}
    arrays: dict[str, np.ndarray] = {}
    n_rows: int | None = None
    for name, (ndim, prefix) in specs.items():
        value = np.asarray(batch[name])
        if value.ndim != ndim or tuple(value.shape[1:1 + len(prefix)]) != prefix:
            expected = "[N," + ",".join(map(str, prefix)) + (",F]" if name == "players" else (",G]" if name == "teams" else "]"))
            raise ValueError(f"batch[{name!r}] must have shape {expected}; got {value.shape}")
        if name in ("players", "teams") and value.shape[-1] < 1:
            raise ValueError(f"batch[{name!r}] must contain at least one feature")
        if value.dtype.kind not in "biuf":
            raise ValueError(f"batch[{name!r}] must contain real numeric values")
        if np.isinf(value).any():
            raise ValueError(f"batch[{name!r}] must not contain infinities")
        if n_rows is None:
            n_rows = len(value)
        elif len(value) != n_rows:
            raise ValueError("batch arrays must have the same row count")
        arrays[name] = value.astype(np.float64, copy=False)
    if n_rows is None or n_rows == 0:
        raise ValueError("batch must contain at least one row")
    if np.isnan(arrays["context"]).any() or np.isnan(arrays["baseline_series"]).any():
        raise ValueError("context and baseline_series must be finite")
    if np.any((arrays["baseline_series"] < 0.0) | (arrays["baseline_series"] > 1.0)):
        raise ValueError("baseline_series probabilities must lie in [0, 1]")
    return arrays["players"], arrays["teams"], arrays["context"], arrays["baseline_series"]


class TabularModel:
    """TRAIN-only-preprocessed logistic or LightGBM series estimator.

    Both model families consume side-symmetric representations. The tree model is
    additionally trained on mirrored examples and projected at inference, ensuring
    exact complementary probabilities even if its raw classifier is asymmetric.
    """

    def __init__(self, kind: str, params: dict[str, Any], seed: int):
        if kind not in ("logistic", "lightgbm"):
            raise ValueError("kind must be 'logistic' or 'lightgbm'")
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        if not isinstance(params.get("pool_players", False), bool):
            raise ValueError("pool_players must be a boolean")
        self.kind = kind
        self.params = dict(params)
        self.seed = int(seed)
        self.preprocessor: _SidePreprocessor | None = None
        self.model: Any = None
        self.feature_names_: tuple[str, ...] = ()
        self.n_features_in_: int | None = None

    @staticmethod
    def _side_fields(players: np.ndarray, teams: np.ndarray, baseline: np.ndarray, *, pool_players: bool = False) -> np.ndarray:
        n = len(players)
        # Role remains explicit by flattening in stable side/role/feature order.
        # Baseline columns identify two models, not the two teams.
        side_controls = np.stack((baseline, 1.0 - baseline), axis=1)
        fields = [players.reshape(n, 2, -1), teams, side_controls]
        if pool_players:
            valid = np.isfinite(players)
            counts = valid.sum(axis=2)
            means = np.divide(
                np.sum(players, axis=2, where=valid, initial=0.0), counts,
                out=np.full(counts.shape, np.nan), where=counts > 0,
            )
            centered = np.subtract(players, means[:, :, None, :], out=np.zeros_like(players), where=valid)
            variance = np.divide(
                np.einsum("nsrf,nsrf->nsf", centered, centered), counts,
                out=np.full(counts.shape, np.nan), where=counts > 0,
            )
            fields.extend((means, np.sqrt(variance)))
        return np.concatenate(fields, axis=2)

    @staticmethod
    def _base_names(n_player: int, n_team: int, *, pool_players: bool = False) -> tuple[str, ...]:
        names = tuple(f"player_role{role}_feature{feature}" for role in range(5) for feature in range(n_player)) + tuple(
            f"team_feature{feature}" for feature in range(n_team)
        ) + ("team_bt_series", "player_bt_series")
        if pool_players:
            names += tuple(f"player_{stat}_feature{feature}" for stat in ("mean", "std") for feature in range(n_player))
        return names

    def _representations(self, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        players, teams, context, baseline = _validate_batch(batch)
        if self.preprocessor is None:
            raise RuntimeError("model has not been fitted")
        transformed = self.preprocessor.transform(self._side_fields(players, teams, baseline, pool_players=self.params.get("pool_players", False)))
        return transformed, context

    @staticmethod
    def _contrast(transformed: np.ndarray, context: np.ndarray) -> np.ndarray:
        odd = transformed[:, 0, :] - transformed[:, 1, :]
        # BO interactions are odd because the interaction uses only signed side
        # contrasts and the side-invariant one-hot best-of context.
        interactions = (odd[:, :, None] * context[:, None, :]).reshape(len(odd), -1)
        return np.concatenate((odd, interactions), axis=1)

    def _tree_features(self, transformed: np.ndarray, context: np.ndarray, first: int, second: int) -> np.ndarray:
        a, b = transformed[:, first, :], transformed[:, second, :]
        return np.concatenate((a, b, a - b, np.abs(a - b), context), axis=1)

    def fit(self, batch: dict[str, np.ndarray], y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TabularModel":
        players, teams, context, baseline = _validate_batch(batch)
        n = len(players)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != n or labels.dtype.kind not in "biuf":
            raise ValueError("y must be a one-dimensional binary array aligned with batch")
        if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError("y must contain only finite binary labels 0 and 1")
        labels = labels.astype(np.float64, copy=False)
        if np.unique(labels).size != 2:
            raise ValueError("training labels must contain both classes")
        if sample_weight is None:
            weights = None
        else:
            weights = np.asarray(sample_weight)
            if weights.ndim != 1 or len(weights) != n or weights.dtype.kind not in "biuf":
                raise ValueError("sample_weight must be a one-dimensional numeric array aligned with batch")
            if not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("sample_weight must contain finite positive values")
            weights = weights.astype(np.float64, copy=False)

        pool_players = self.params.get("pool_players", False)
        sides = self._side_fields(players, teams, baseline, pool_players=pool_players)
        names = self._base_names(players.shape[-1], teams.shape[-1], pool_players=pool_players)
        self.preprocessor = _SidePreprocessor().fit(sides, names)
        transformed = self.preprocessor.transform(sides)
        self.n_features_in_ = transformed.shape[-1]
        if self.kind == "logistic":
            x = self._contrast(transformed, context)
            c = float(self.params.get("C", 1.0))
            if not np.isfinite(c) or c <= 0:
                raise ValueError("logistic C must be finite and positive")
            self.model = LogisticRegression(
                C=c,
                fit_intercept=False,
                solver="lbfgs",
                max_iter=int(self.params.get("max_iter", 1000)),
                random_state=self.seed,
            )
            self.model.fit(x, labels, sample_weight=weights)
            raw_names = names + tuple(f"missing_{name}" for name in names)
            self.feature_names_ = tuple(raw_names) + tuple(
                f"{name}:bo{bo}" for name in raw_names for bo in range(3)
            )
        else:
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:
                raise ImportError("lightgbm is required for kind='lightgbm'") from exc
            x = self._tree_features(transformed, context, 0, 1)
            x_swapped = self._tree_features(transformed, context, 1, 0)
            mirrored = np.concatenate((x, x_swapped), axis=0)
            mirrored_y = np.concatenate((labels, 1.0 - labels))
            mirrored_weight = None if weights is None else np.concatenate((weights, weights))
            allowed = {
                "num_leaves", "max_depth", "learning_rate", "n_estimators",
                "subsample", "colsample_bytree", "min_child_samples", "reg_lambda",
                "reg_alpha", "max_bin", "min_split_gain",
            }
            params = {"learning_rate": 0.03, "colsample_bytree": 0.8}
            params.update({key: value for key, value in self.params.items() if key in allowed})
            params.update(
                objective="binary",
                n_jobs=2,
                random_state=self.seed,
                deterministic=True,
                force_col_wise=True,
                verbosity=-1,
            )
            self.model = LGBMClassifier(**params)
            self.model.fit(mirrored, mirrored_y, sample_weight=mirrored_weight)
            self.feature_names_ = (
                tuple(f"side_a_{name}" for name in names + tuple(f"missing_{v}" for v in names))
                + tuple(f"side_b_{name}" for name in names + tuple(f"missing_{v}" for v in names))
                + tuple(f"difference_{name}" for name in names + tuple(f"missing_{v}" for v in names))
                + tuple(f"absolute_difference_{name}" for name in names + tuple(f"missing_{v}" for v in names))
                + ("context_bo1", "context_bo3", "context_bo5")
            )
        return self

    def predict_proba(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        transformed, context = self._representations(batch)
        if self.model is None:
            raise RuntimeError("model has not been fitted")
        if self.kind == "logistic":
            result = np.asarray(self.model.predict_proba(self._contrast(transformed, context))[:, 1], dtype=np.float64)
        else:
            forward = self._tree_features(transformed, context, 0, 1)
            reverse = self._tree_features(transformed, context, 1, 0)
            # Shapes/schema were validated above; use the ndarray-native API.
            q_forward = self.model.booster_.predict(forward, num_threads=2)
            q_reverse = self.model.booster_.predict(reverse, num_threads=2)
            result = np.asarray(0.5 + 0.5 * (q_forward - q_reverse), dtype=np.float64)
        if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
            raise ValueError("model produced invalid probabilities")
        return result

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model is None:
            raise AttributeError("feature_importances_ is available after fitting")
        if self.kind == "logistic":
            return np.asarray(np.abs(self.model.coef_[0]), dtype=np.float64)
        return np.asarray(self.model.feature_importances_, dtype=np.float64)
