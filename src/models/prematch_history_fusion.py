"""Joint history/summary neural architectures for pre-draft series prediction."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.prematch_neural import (
    NeuralModel,
    _BASELINE_COUNT,
    _CONTEXT_COUNT,
    _ROLE_COUNT,
    _TrainNormalizer,
)


_HISTORY_MATCHES = 16
_TRAIN_BATCH_SIZE = 512
_INFERENCE_BATCH_SIZE = 512
_HISTORY_STAT_CHUNK_ROWS = 64
_LEARNING_RATE = 0.001


class _HistoryPairNetwork(nn.Module):
    """History attention, five-role pooling, and antisymmetric pair scoring."""

    def __init__(
        self,
        player_width: int,
        history_width: int,
        team_width: int,
        hidden: int,
        dropout: float,
        architecture: str,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.hidden = hidden
        self.history_projection = nn.Linear(2 * history_width, hidden)
        self.history_attention = nn.Linear(hidden, 1, bias=False)
        if architecture == "input":
            self.player_encoder = nn.Sequential(
                nn.Linear(2 * player_width + hidden + _ROLE_COUNT, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.history_encoder = None
            self.fusion = None
        else:
            self.player_encoder = nn.Sequential(
                nn.Linear(2 * player_width + _ROLE_COUNT, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.history_encoder = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
            self.fusion = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU()
            )
        team_feature_width = _ROLE_COUNT * hidden + 2 * team_width + 2 * _BASELINE_COUNT
        pair_width = 2 * team_feature_width + 2 * _CONTEXT_COUNT
        self.pair_scorer = nn.Sequential(
            nn.Linear(pair_width, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def _attend_history(self, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        shape = history.shape
        flattened = history.reshape(shape[0], 2, _ROLE_COUNT, _HISTORY_MATCHES, shape[-1])
        visible = mask.reshape(shape[0], 2, _ROLE_COUNT, _HISTORY_MATCHES)
        projected = torch.tanh(self.history_projection(flattened))
        scores = self.history_attention(projected).squeeze(-1)
        # Multiplicative masked softmax handles an entirely empty history without NaNs.
        weights = torch.softmax(scores.masked_fill(~visible, torch.finfo(scores.dtype).min), dim=-1)
        weights = weights * visible.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.sum(projected * weights.unsqueeze(-1), dim=-2)

    def forward(
        self,
        players: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        teams: torch.Tensor,
        context: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        n = players.shape[0]
        attended = self._attend_history(history, history_mask)
        role_codes = torch.eye(_ROLE_COUNT, dtype=players.dtype, device=players.device)
        role_codes = role_codes.view(1, 1, _ROLE_COUNT, _ROLE_COUNT).expand(n, 2, -1, -1)
        if self.architecture == "input":
            encoded = self.player_encoder(torch.cat((players, attended, role_codes), dim=-1))
        else:
            encoded_player = self.player_encoder(torch.cat((players, role_codes), dim=-1))
            encoded_history = self.history_encoder(attended)
            encoded = self.fusion(torch.cat((encoded_player, encoded_history), dim=-1))
        role_pool = encoded.reshape(n, 2, _ROLE_COUNT * self.hidden)
        sides = torch.cat((role_pool, teams, baseline), dim=-1)
        side_a, side_b = sides[:, 0], sides[:, 1]
        score_ab = self.pair_scorer(torch.cat((side_a, side_b, context), dim=-1)).squeeze(-1)
        score_ba = self.pair_scorer(torch.cat((side_b, side_a, context), dim=-1)).squeeze(-1)
        return (score_ab - score_ba) * 0.5


class HistoryFusionModel:
    """Fresh CPU neural model jointly learning static and masked history features.

    Empty histories require explicit masks; omitted inputs are an error.
    TRAIN-fitted history statistics remain available for architecture ablations.
    """

    def __init__(self, params: dict[str, Any], seed: int):
        self.params = dict(params)
        self.seed = int(seed)
        self.architecture = self.params.get("architecture", "representation")
        if self.architecture not in ("input", "representation"):
            raise ValueError("architecture must be 'input' or 'representation'")
        self.hidden = self._positive_int("hidden", 32)
        self.epochs = self._positive_int("epochs", 24)
        self.dropout = float(self.params.get("dropout", 0.1))
        self.weight_decay = float(self.params.get("weight_decay", 0.001))
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        self._player_normalizer: _TrainNormalizer | None = None
        self._team_normalizer: _TrainNormalizer | None = None
        self._context_normalizer: _TrainNormalizer | None = None
        self._baseline_normalizer: _TrainNormalizer | None = None
        self.history_normalizer: _TrainNormalizer | None = None
        self._player_feature_count: int | None = None
        self._team_feature_count: int | None = None
        self._history_feature_count: int | None = None
        self._network: _HistoryPairNetwork | None = None
        self.history_summary_: dict[str, Any] | None = None
        self.parameter_count_: int | None = None

    def _positive_int(self, name: str, default: int) -> int:
        value = self.params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    @staticmethod
    def _baseline_logits(baseline: np.ndarray) -> np.ndarray:
        result = np.full(baseline.shape, np.nan, dtype=np.float64)
        present = np.isfinite(baseline)
        clipped = np.clip(baseline[present], 1e-6, 1.0 - 1e-6)
        result[present] = np.log(clipped) - np.log1p(-clipped)
        return result

    @staticmethod
    def _history_arrays(batch: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray]:
        has_history = "history" in batch
        has_mask = "history_mask" in batch
        if not has_history or not has_mask:
            raise ValueError("history and history_mask must both be supplied")
        history = np.asarray(batch["history"])
        mask_raw = np.asarray(batch["history_mask"])
        if not np.issubdtype(history.dtype, np.number):
            raise TypeError("history must be a numeric array")
        expected = (n, 2, _ROLE_COUNT, _HISTORY_MATCHES)
        if history.ndim != 5 or history.shape[:4] != expected or history.shape[-1] < 1:
            raise ValueError("history must have shape [N, 2, 5, 16, H] with H >= 1")
        if mask_raw.shape != expected:
            raise ValueError("history_mask must have shape [N, 2, 5, 16]")
        if mask_raw.dtype != np.bool_:
            if not np.issubdtype(mask_raw.dtype, np.number) or not np.isfinite(mask_raw).all() or not np.isin(mask_raw, (0, 1)).all():
                raise ValueError("history_mask must contain only boolean values")
        mask = mask_raw.astype(bool, copy=False)
        # Inspect only visible tokens in bounded chunks; padding may contain any bit pattern.
        for start in range(0, n, _HISTORY_STAT_CHUNK_ROWS):
            stop = min(start + _HISTORY_STAT_CHUNK_ROWS, n)
            local_mask = mask[start:stop].reshape(-1)
            if local_mask.any() and not np.isfinite(history[start:stop].reshape(-1, history.shape[-1])[local_mask]).all():
                raise ValueError("observed history tokens must contain finite values")
        return history, mask

    @classmethod
    def _fit_history_normalizer(cls, history: np.ndarray, mask: np.ndarray, width: int) -> _TrainNormalizer:
        total = np.zeros(width, dtype=np.float64)
        count = 0
        n = history.shape[0]
        for start in range(0, n, _HISTORY_STAT_CHUNK_ROWS):
            stop = min(start + _HISTORY_STAT_CHUNK_ROWS, n)
            local_mask = mask[start:stop].reshape(-1)
            if local_mask.any():
                rows = history[start:stop].reshape(-1, width)[local_mask].astype(np.float64, copy=False)
                total += rows.sum(axis=0)
                count += rows.shape[0]
        if count == 0:
            return _TrainNormalizer(np.zeros(width), np.ones(width), np.zeros(width, dtype=bool))
        mean = total / count
        squared = np.zeros(width, dtype=np.float64)
        for start in range(0, n, _HISTORY_STAT_CHUNK_ROWS):
            stop = min(start + _HISTORY_STAT_CHUNK_ROWS, n)
            local_mask = mask[start:stop].reshape(-1)
            if local_mask.any():
                rows = history[start:stop].reshape(-1, width)[local_mask].astype(np.float64, copy=False)
                squared += np.square(rows - mean).sum(axis=0)
        std = np.sqrt(squared / count)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("history TRAIN statistics are not finite")
        scale = np.where(std > 1e-6, std, 1.0)
        return _TrainNormalizer(mean, scale, np.ones(width, dtype=bool))

    @staticmethod
    def _transform_history(
        history: np.ndarray,
        mask: np.ndarray,
        normalizer: _TrainNormalizer,
        indices: np.ndarray | slice,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_mask = np.ascontiguousarray(mask[indices])
        shape = local_mask.shape
        width = normalizer.mean.shape[0]
        result = np.zeros((*shape, 2 * width), dtype=np.float32)
        local = history[indices]
        visible = local_mask.reshape(-1)
        if visible.any():
            rows = local.reshape(-1, width)[visible].astype(np.float64, copy=False)
            present = np.isfinite(rows) & normalizer.observed
            scaled = np.where(present, (rows - normalizer.mean) / normalizer.scale, 0.0)
            encoded = np.concatenate((scaled.astype(np.float32), (~present).astype(np.float32)), axis=1)
            result.reshape(-1, 2 * width)[visible] = encoded
        return result, local_mask

    def _static_transform(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
        assert self._player_normalizer is not None and self._team_normalizer is not None
        assert self._context_normalizer is not None and self._baseline_normalizer is not None
        players = self._player_normalizer.transform(arrays["players"])
        teams = self._team_normalizer.transform(arrays["teams"])
        context = self._context_normalizer.transform(arrays["context"])
        baseline = self._baseline_normalizer.transform(self._baseline_logits(arrays["baseline_series"]))
        values, missing = baseline[..., :_BASELINE_COUNT], baseline[..., _BASELINE_COUNT:]
        sides = np.stack((np.concatenate((values, missing), axis=-1), np.concatenate((-values, missing), axis=-1)), axis=1)
        return players, teams, context, sides

    def fit(self, batch: Mapping[str, Any], y: Any, sample_weight: Any | None = None) -> HistoryFusionModel:
        arrays = NeuralModel._validated_batch(batch)
        n = arrays["players"].shape[0]
        labels = np.asarray(y, dtype=np.float64)
        if labels.shape != (n,) or not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("y must be a finite binary vector with one label per row")
        if sample_weight is None:
            weights = np.ones(n, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (n,) or not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("sample_weight must be a finite positive vector with one weight per row")
        history, mask = self._history_arrays(batch, n)
        self._player_feature_count = arrays["players"].shape[-1]
        self._team_feature_count = arrays["teams"].shape[-1]
        self._history_feature_count = history.shape[-1]
        self._player_normalizer = _TrainNormalizer.fit(arrays["players"])
        self._team_normalizer = _TrainNormalizer.fit(arrays["teams"])
        self._context_normalizer = _TrainNormalizer.fit(arrays["context"])
        self._baseline_normalizer = _TrainNormalizer.fit(self._baseline_logits(arrays["baseline_series"]), center=False)
        history_normalizer = self._fit_history_normalizer(history, mask, self._history_feature_count)
        self.history_normalizer = history_normalizer
        static = self._static_transform(arrays)
        tensors = tuple(torch.from_numpy(value).to(dtype=torch.float32, device="cpu") for value in static)
        label_tensor = torch.from_numpy(labels.astype(np.float32))
        weight_tensor = torch.from_numpy(weights)
        torch.set_num_threads(2)
        seed = self.seed % (2**63 - 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = _HistoryPairNetwork(
                self._player_feature_count, self._history_feature_count, self._team_feature_count,
                self.hidden, self.dropout, self.architecture,
            ).to(device="cpu")
            optimizer = torch.optim.AdamW(network.parameters(), lr=_LEARNING_RATE, weight_decay=self.weight_decay)
            shuffle = torch.Generator(device="cpu")
            shuffle.manual_seed(seed)
            network.train()
            for _ in range(self.epochs):
                order = torch.randperm(n, generator=shuffle)
                for ids_tensor in order.split(_TRAIN_BATCH_SIZE):
                    ids = ids_tensor.numpy()
                    history_values, history_mask = self._transform_history(history, mask, history_normalizer, ids)
                    player_batch, team_batch, context_batch, baseline_batch = (
                        value[ids_tensor] for value in tensors
                    )
                    optimizer.zero_grad(set_to_none=True)
                    logits = network(
                        player_batch,
                        torch.from_numpy(history_values),
                        torch.from_numpy(history_mask),
                        team_batch,
                        context_batch,
                        baseline_batch,
                    )
                    losses = nn.functional.binary_cross_entropy_with_logits(logits, label_tensor[ids_tensor], reduction="none")
                    batch_weights = weight_tensor[ids_tensor]
                    batch_weights = (batch_weights / batch_weights.max()).to(dtype=losses.dtype)
                    loss = torch.sum(losses * batch_weights) / batch_weights.sum()
                    loss.backward()
                    optimizer.step()
            network.eval()
        self._network = network
        self.parameter_count_ = sum(parameter.numel() for parameter in network.parameters())
        self.history_summary_ = {
            "observed_tokens": int(mask.sum()),
            "history_width": self._history_feature_count,
            "architecture": self.architecture,
        }
        return self

    def predict_proba(self, batch: Mapping[str, Any]) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("HistoryFusionModel must be fitted before prediction")
        arrays = NeuralModel._validated_batch(batch)
        n = arrays["players"].shape[0]
        if arrays["players"].shape[-1] != self._player_feature_count:
            raise ValueError("player feature width differs from fitted TRAIN data")
        if arrays["teams"].shape[-1] != self._team_feature_count:
            raise ValueError("team feature width differs from fitted TRAIN data")
        history, mask = self._history_arrays(batch, n)
        if history.shape[-1] != self._history_feature_count:
            raise ValueError("history feature width differs from fitted TRAIN data")
        assert self._player_normalizer is not None and self._team_normalizer is not None
        static = self._static_transform(arrays)
        network = self._network
        results: list[np.ndarray] = []
        network.eval()
        with torch.inference_mode():
            for start in range(0, n, _INFERENCE_BATCH_SIZE):
                stop = min(start + _INFERENCE_BATCH_SIZE, n)
                history_values, history_mask = self._transform_history(history, mask, self.history_normalizer, slice(start, stop))
                tensors = [torch.from_numpy(value[start:stop]).to(dtype=torch.float32) for value in static]
                logits = network(
                    tensors[0], torch.from_numpy(history_values), torch.from_numpy(history_mask),
                    tensors[1], tensors[2], tensors[3],
                ).to(dtype=torch.float64).numpy()
                results.append(logits)
        logit_array = np.concatenate(results)
        if not np.isfinite(logit_array).all():
            raise RuntimeError("HistoryFusionModel produced a non-finite inference logit")
        probabilities = np.empty_like(logit_array)
        positive = logit_array >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logit_array[positive]))
        exponential = np.exp(logit_array[~positive])
        probabilities[~positive] = exponential / (1.0 + exponential)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("HistoryFusionModel produced non-finite probabilities")
        return probabilities
