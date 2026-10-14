"""Fresh role-aware neural model for pre-draft LoL series probabilities."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


_ROLE_COUNT = 5
_CONTEXT_COUNT = 3
_BASELINE_COUNT = 2
_TRAIN_BATCH_SIZE = 512
_INFERENCE_BATCH_SIZE = 4096
_LEARNING_RATE = 0.001


class _TrainNormalizer:
    """Feature-wise imputation and scaling fitted only on supplied TRAIN rows."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, observed: np.ndarray):
        self.mean = mean
        self.scale = scale
        self.observed = observed

    @classmethod
    def fit(cls, values: np.ndarray, center: bool = True) -> _TrainNormalizer:
        feature_count = values.shape[-1]
        if feature_count == 0:
            empty = np.empty(0, dtype=np.float64)
            return cls(mean=empty, scale=empty.copy(), observed=empty.astype(bool))
        flat = values.reshape(-1, feature_count).astype(np.float64, copy=False)
        present = np.isfinite(flat)
        counts = present.sum(axis=0)
        observed = counts > 0
        safe_counts = np.maximum(counts, 1)
        if center:
            mean = np.where(observed, np.where(present, flat, 0.0).sum(axis=0) / safe_counts, 0.0)
            deviations = np.where(present, flat - mean, 0.0)
        else:
            mean = np.zeros(feature_count, dtype=np.float64)
            deviations = np.where(present, flat, 0.0)
        variance = np.where(observed, np.square(deviations).sum(axis=0) / safe_counts, 0.0)
        standard_deviation = np.sqrt(variance)
        scale = np.where(np.isfinite(standard_deviation) & (standard_deviation > 1e-6), standard_deviation, 1.0)
        return cls(mean=mean, scale=scale, observed=observed)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature width differs from fitted TRAIN data")
        if values.shape[-1] == 0:
            return np.empty((*values.shape[:-1], 0), dtype=np.float32)
        flat = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
        present = np.isfinite(flat) & self.observed
        filled = np.where(present, flat, self.mean)
        scaled = np.where(present, (filled - self.mean) / self.scale, 0.0)
        missing = (~present).astype(np.float32)
        result = np.concatenate((scaled.astype(np.float32), missing), axis=1)
        if not np.isfinite(result).all():
            raise ValueError("preprocessed features contain non-finite values")
        return result.reshape(*values.shape[:-1], 2 * values.shape[-1])


class _RoleAwarePairNetwork(nn.Module):
    """Shared player encoder and a shared ordered-pair scorer."""

    def __init__(
        self,
        player_feature_count: int,
        team_feature_count: int,
        hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.player_encoder = nn.Sequential(
            nn.Linear(2 * player_feature_count + _ROLE_COUNT, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Concatenating the five encoded roles is a role-preserving team pool.
        team_width = _ROLE_COUNT * hidden + 2 * team_feature_count + 2 * _BASELINE_COUNT
        pair_width = 2 * team_width + 2 * _CONTEXT_COUNT
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_width, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.team_width = team_width

    def forward(
        self,
        players: torch.Tensor,
        teams: torch.Tensor,
        context: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = players.shape[0]
        role_codes = torch.eye(_ROLE_COUNT, dtype=players.dtype, device=players.device)
        role_codes = role_codes.view(1, 1, _ROLE_COUNT, _ROLE_COUNT).expand(batch_size, 2, -1, -1)
        encoded_players = self.player_encoder(torch.cat((players, role_codes), dim=-1))
        role_pool = encoded_players.reshape(batch_size, 2, _ROLE_COUNT * encoded_players.shape[-1])
        sides = torch.cat((role_pool, teams, baseline), dim=-1)
        side_a, side_b = sides[:, 0], sides[:, 1]
        score_ab = self.pair_scorer(torch.cat((side_a, side_b, context), dim=-1)).squeeze(-1)
        score_ba = self.pair_scorer(torch.cat((side_b, side_a, context), dim=-1)).squeeze(-1)
        return (score_ab - score_ba) * 0.5


class NeuralModel:
    """CPU neural estimator with exact side-swap antisymmetry at inference."""

    def __init__(self, params: dict[str, Any], seed: int):
        self.params = dict(params)
        self.seed = int(seed)
        self.hidden = self._positive_int("hidden", 32)
        self.epochs = self._positive_int("epochs", 24)
        self.dropout = float(self.params.get("dropout", 0.1))
        self.weight_decay = float(self.params.get("weight_decay", 0.001))
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")

        self._player_normalizer: _TrainNormalizer | None = None
        self._team_normalizer: _TrainNormalizer | None = None
        self._context_normalizer: _TrainNormalizer | None = None
        self._baseline_normalizer: _TrainNormalizer | None = None
        self._player_feature_count: int | None = None
        self._team_feature_count: int | None = None
        self._network: _RoleAwarePairNetwork | None = None

    def _positive_int(self, name: str, default: int) -> int:
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _numeric_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be a numeric array")
        array = array.astype(np.float64, copy=False)
        if np.isinf(array).any():
            raise ValueError(f"{name} may contain NaN but not infinity")
        return array

    @classmethod
    def _validated_batch(cls, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping of feature arrays")
        required = ("players", "teams", "context", "baseline_series")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"batch is missing required arrays: {', '.join(missing)}")
        arrays = {key: cls._numeric_array(batch[key], key) for key in required}
        players, teams = arrays["players"], arrays["teams"]
        context, baseline = arrays["context"], arrays["baseline_series"]
        if players.ndim != 4 or players.shape[1:3] != (2, _ROLE_COUNT) or players.shape[-1] < 1:
            raise ValueError("players must have shape [N, 2, 5, F] with F >= 1")
        n = players.shape[0]
        if n < 1:
            raise ValueError("batch must contain at least one row")
        if teams.ndim != 3 or teams.shape[:2] != (n, 2):
            raise ValueError("teams must have shape [N, 2, G]")
        if context.shape != (n, _CONTEXT_COUNT):
            raise ValueError("context must have shape [N, 3]")
        if not np.isfinite(context).all():
            raise ValueError("context must contain a finite BO1/BO3/BO5 one-hot row")
        if not np.isin(context, (0.0, 1.0)).all() or not np.equal(context.sum(axis=1), 1.0).all():
            raise ValueError("context rows must be one-hot BO1/BO3/BO5 indicators")
        if baseline.shape != (n, _BASELINE_COUNT):
            raise ValueError("baseline_series must have shape [N, 2]")
        finite_baseline = baseline[np.isfinite(baseline)]
        if ((finite_baseline < 0.0) | (finite_baseline > 1.0)).any():
            raise ValueError("baseline_series values must be probabilities in [0, 1]")
        return arrays

    @staticmethod
    def _baseline_logits(baseline: np.ndarray) -> np.ndarray:
        result = np.full(baseline.shape, np.nan, dtype=np.float64)
        present = np.isfinite(baseline)
        clipped = np.clip(baseline[present], 1e-6, 1.0 - 1e-6)
        result[present] = np.log(clipped) - np.log1p(-clipped)
        return result

    def _transform(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._player_normalizer is not None
        assert self._team_normalizer is not None
        assert self._context_normalizer is not None
        assert self._baseline_normalizer is not None
        players = self._player_normalizer.transform(arrays["players"])
        teams = self._team_normalizer.transform(arrays["teams"])
        context = self._context_normalizer.transform(arrays["context"])
        # The controls are probabilities for side A. Logit contrasts are
        # normalized without centering, then represented with opposite signs.
        baseline = self._baseline_normalizer.transform(self._baseline_logits(arrays["baseline_series"]))
        baseline_values = baseline[..., :_BASELINE_COUNT]
        baseline_missing = baseline[..., _BASELINE_COUNT:]
        baseline_a = np.concatenate((baseline_values, baseline_missing), axis=-1)
        baseline_b = np.concatenate((-baseline_values, baseline_missing), axis=-1)
        baseline_sides = np.stack((baseline_a, baseline_b), axis=1)
        return players, teams, context, baseline_sides


    def fit(
        self,
        batch: Mapping[str, Any],
        y: Any,
        sample_weight: Any | None = None,
    ) -> NeuralModel:
        arrays = self._validated_batch(batch)
        players = arrays["players"]
        n = players.shape[0]
        labels = np.asarray(y, dtype=np.float64)
        if labels.shape != (n,) or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("y must be a finite binary vector with one label per row")
        if sample_weight is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (n,) or not np.isfinite(weights).all() or (weights <= 0.0).any():
                raise ValueError("sample_weight must be a finite positive vector with one weight per row")

        self._player_feature_count = players.shape[-1]
        self._team_feature_count = arrays["teams"].shape[-1]
        self._player_normalizer = _TrainNormalizer.fit(players)
        self._team_normalizer = _TrainNormalizer.fit(arrays["teams"])
        self._context_normalizer = _TrainNormalizer.fit(arrays["context"])
        self._baseline_normalizer = _TrainNormalizer.fit(
            self._baseline_logits(arrays["baseline_series"]), center=False
        )
        transformed = self._transform(arrays)
        player_tensor, team_tensor, context_tensor, baseline_tensor = (
            torch.from_numpy(value).to(dtype=torch.float32, device="cpu") for value in transformed
        )
        label_tensor = torch.from_numpy(labels.astype(np.float32)).to(device="cpu")
        weight_tensor = torch.from_numpy(weights).to(dtype=torch.float64, device="cpu")

        torch.set_num_threads(2)
        seed = self.seed % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = _RoleAwarePairNetwork(
                player_feature_count=self._player_feature_count,
                team_feature_count=self._team_feature_count,
                hidden=self.hidden,
                dropout=self.dropout,
            ).to(device="cpu")
            optimizer = torch.optim.AdamW(network.parameters(), lr=_LEARNING_RATE, weight_decay=self.weight_decay)
            shuffle_generator = torch.Generator(device="cpu")
            shuffle_generator.manual_seed(seed)
            network.train()
            for _ in range(self.epochs):
                order = torch.randperm(n, generator=shuffle_generator)
                for indices in order.split(_TRAIN_BATCH_SIZE):
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(
                        player_tensor[indices],
                        team_tensor[indices],
                        context_tensor[indices],
                        baseline_tensor[indices],
                    )
                    losses = nn.functional.binary_cross_entropy_with_logits(
                        logits, label_tensor[indices], reduction="none"
                    )
                    batch_weights = weight_tensor[indices]
                    batch_weights = (batch_weights / torch.max(batch_weights)).to(dtype=losses.dtype)
                    loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
                    loss.backward()
                    optimizer.step()
            network.eval()
        self._network = network
        return self

    def predict_proba(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("NeuralModel must be fitted before prediction")
        arrays = self._validated_batch(batch)
        if arrays["players"].shape[-1] != self._player_feature_count:
            raise ValueError("player feature width differs from fitted TRAIN data")
        if arrays["teams"].shape[-1] != self._team_feature_count:
            raise ValueError("team feature width differs from fitted TRAIN data")
        players, teams, context, baseline = self._transform(arrays)
        tensors = tuple(torch.from_numpy(value).to(dtype=torch.float32, device="cpu") for value in (players, teams, context, baseline))
        network = self._network
        network.eval()
        logits: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, arrays["players"].shape[0], _INFERENCE_BATCH_SIZE):
                stop = start + _INFERENCE_BATCH_SIZE
                logit = network(*(value[start:stop] for value in tensors))
                logits.append(logit.to(dtype=torch.float64, device="cpu").numpy())
        logit_array = np.concatenate(logits).astype(np.float64, copy=False)
        if not np.isfinite(logit_array).all():
            raise RuntimeError("NeuralModel produced a non-finite inference logit")
        probabilities = np.empty_like(logit_array, dtype=np.float64)
        nonnegative = logit_array >= 0.0
        probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-logit_array[nonnegative]))
        exponentials = np.exp(logit_array[~nonnegative])
        probabilities[~nonnegative] = exponentials / (1.0 + exponentials)
        return probabilities
