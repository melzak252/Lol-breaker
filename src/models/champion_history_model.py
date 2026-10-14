"""History-only, organization-independent score and played-map distributions.

The four recipes share every parameter and initialization. Hiding champions zeros
both their embedding and observability indicators, not just their identity lookup.
Only TRAIN fits numerical preprocessing; CAL changes three bounded global scalars
at most, and never the network. No current draft or target statistics are read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import math
import random
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import nbinom
import torch
from torch import nn
from torch.nn import functional as F

from src.models.prop_distribution import DistributionPrediction, DistributionMixture

DEFAULT_CONFIG = {"width": 32, "embedding_dim": 8, "epochs": 12,
                  "batch_size": 256, "lr": .001, "weight_decay": .0001,
                  "threads": 4, "kills_weight": .25, "duration_weight": .25}
CALIBRATION_BOUNDS = (.25, 4.)
CALIBRATION_REGULARIZATION = .001
_INPUT_KEYS = ("history", "champions", "patches", "competitions", "roles", "mask",
               "players", "role_probs", "context", "best_of")


def _configuration(config, software_smoke=False):
    if not isinstance(config, dict) or config.keys() - DEFAULT_CONFIG.keys():
        raise ValueError("unknown history-model configuration key")
    result = DEFAULT_CONFIG | config
    for key in ("width", "embedding_dim", "epochs", "threads"):
        expected = 1 if key == "epochs" and software_smoke else DEFAULT_CONFIG[key]
        if isinstance(result[key], bool) or result[key] != expected:
            raise ValueError(f"{key} is fixed at {expected}")
        result[key] = int(result[key])
    size = result["batch_size"]
    if isinstance(size, bool) or not np.isfinite(size) or size != int(size) or size <= 0:
        raise ValueError("batch_size must be a positive integer")
    result["batch_size"] = int(size)
    for key in ("lr", "weight_decay", "kills_weight", "duration_weight"):
        value = result[key]
        if isinstance(value, bool) or not np.isfinite(value) or value < 0 or (key == "lr" and value == 0):
            raise ValueError(f"{key} must be finite and nonnegative (lr strictly positive)")
        result[key] = float(value)
    return result


def _indices(value, n):
    if value is None:
        return np.arange(n, dtype=np.int64)
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype.kind not in "iu" or np.any((result < 0) | (result >= n)):
        raise ValueError("indices must be an in-range one-dimensional integer array")
    return result.astype(np.int64, copy=False)


def _legal(bo):
    return np.tile(np.arange(3), 2)[None, :] <= (bo // 2)[:, None]


def _validate_arrays(arrays, dimensions=None):
    if not isinstance(arrays, dict) or any(key not in arrays for key in _INPUT_KEYS):
        raise ValueError("missing required history input arrays")
    values = {key: np.asarray(arrays[key]) for key in _INPUT_KEYS}
    history = values["history"]
    if history.ndim != 5 or history.shape[1:4] != (2, 5, 16) or history.shape[-1] < 1:
        raise ValueError("history must have shape [N,2,5,16,F] with F positive")
    n = len(history)
    token_shape = history.shape[:-1]
    for key in ("champions", "patches", "competitions", "roles", "mask"):
        if values[key].shape != token_shape:
            raise ValueError(f"{key} must have shape [N,2,5,16]")
    if values["mask"].dtype.kind != "b":
        raise ValueError("mask must be boolean")
    for key in ("champions", "patches", "competitions", "roles"):
        x = values[key]
        if x.dtype.kind not in "iu" or np.any(x < 0) or (key == "roles" and np.any(x > 5)):
            raise ValueError(f"{key} must use nonnegative integer codes (roles <=5)")
    players = values["players"]
    if players.ndim != 4 or players.shape[:3] != (n, 2, 5):
        raise ValueError("players must have shape [N,2,5,P]")
    if values["role_probs"].shape != (n, 2, 5, 6):
        raise ValueError("role_probs must have shape [N,2,5,6]")
    context = values["context"]
    if context.ndim != 2 or len(context) != n:
        raise ValueError("context must have shape [N,C]")
    for key in ("history", "players", "context", "role_probs"):
        x = values[key]
        if x.dtype.kind not in "fiu" or np.isinf(x).any():
            raise ValueError(f"{key} must be numeric without infinity")
    probs = values["role_probs"]
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(-1), 1., atol=1e-6):
        raise ValueError("role_probs must be finite nonnegative normalized six-role probabilities")
    bo = values["best_of"]
    if bo.shape != (n,) or not np.isin(bo, (1, 3, 5)).all():
        raise ValueError("best_of must contain 1, 3 or 5 in shape [N]")
    actual = (history.shape[-1], players.shape[-1], context.shape[-1])
    if dimensions is not None and actual != dimensions:
        raise ValueError("input feature dimensions differ from TRAIN")
    return values, actual


@dataclass
class HistoryPrediction:
    score_probs: np.ndarray
    best_of: np.ndarray
    kills_mu: np.ndarray | None = None
    kills_alpha: np.ndarray | None = None
    duration_log_mu: np.ndarray | None = None
    duration_log_sigma: np.ndarray | None = None
    components: tuple[HistoryPrediction, ...] = field(default_factory=tuple, repr=False)
    p_win: np.ndarray = field(init=False)
    length_probs: np.ndarray = field(init=False)
    play_probs: np.ndarray = field(init=False)

    def __post_init__(self):
        bo = np.asarray(self.best_of)
        if bo.ndim != 1 or not np.isin(bo, (1, 3, 5)).all():
            raise ValueError("best_of must be one-dimensional with values 1, 3 or 5")
        self.best_of = bo.astype(np.int64, copy=True)
        p = np.array(self.score_probs, dtype=np.float64, copy=True)
        if p.shape != (len(bo), 6) or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("score_probs must be finite nonnegative [N,6]")
        if not np.allclose(p.sum(1), 1., atol=1e-9, rtol=1e-7) or np.any(p[~_legal(self.best_of)] != 0):
            raise ValueError("score_probs must be normalized with exactly zero illegal score mass")
        self.score_probs = p
        self.p_win = p[:, :3].sum(1)
        self.length_probs = p[:, :3] + p[:, 3:]
        lengths = (self.best_of // 2 + 1)[:, None] + np.arange(3)
        self.play_probs = (self.length_probs[:, :, None] *
                           (lengths[:, :, None] >= np.arange(1, 6))).sum(1)
        names = ("kills_mu", "kills_alpha", "duration_log_mu", "duration_log_sigma")
        present = [getattr(self, name) is not None for name in names]
        if any(present) and not all(present):
            raise ValueError("all four prop parameters must be supplied together")
        for name in names:
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                if value.shape != (len(bo), 5) or not np.isfinite(value).all() or (name != "duration_log_mu" and np.any(value <= 0)):
                    raise ValueError(f"{name} must be finite [N,5] and positive except log_mu")
                setattr(self, name, value)
        self.components = tuple(self.components)
        if self.components and (any(not isinstance(c, HistoryPrediction) for c in self.components) or
                                any(not np.array_equal(c.best_of, self.best_of) for c in self.components)):
            raise ValueError("ensemble components must preserve rows and best_of")

    def map_prediction(self, map_index):
        if isinstance(map_index, bool) or not isinstance(map_index, (int, np.integer)) or not 1 <= map_index <= 5:
            raise ValueError("map_index must be an integer in 1..5")
        if self.components:
            # Each seed keeps its full NB/lognormal density, including calibrated
            # dispersion. Shared ensemble length probabilities preserve coherence.
            marginals = []
            for component in self.components:
                marginal = component.map_prediction(map_index)
                marginals.append(DistributionPrediction(marginal.kills_mu, marginal.kills_alpha,
                                                        marginal.duration_log_mu, marginal.duration_log_sigma,
                                                        self.length_probs, self.best_of))
            return DistributionMixture(marginals)
        if self.kills_mu is None:
            raise NotImplementedError("score-only model has no fitted played-map prop distributions")
        i = map_index - 1
        return DistributionPrediction(self.kills_mu[:, i], self.kills_alpha[:, i],
                                      self.duration_log_mu[:, i], self.duration_log_sigma[:, i],
                                      self.length_probs, self.best_of)


@dataclass
class HistoryPreprocessor:
    """Public fitted arrays permit independent replay; zero imputation is masked."""
    mean: dict[str, np.ndarray]
    scale: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]

    @classmethod
    def fit(cls, arrays, indices, batch_size):
        means, scales, counts = {}, {}, {}
        for key in ("history", "players", "context"):
            width = arrays[key].shape[-1]
            total = np.zeros(width, dtype=np.float64)
            square = np.zeros(width, dtype=np.float64)
            count = np.zeros(width, dtype=np.int64)
            for start in range(0, len(indices), batch_size):
                ix = indices[start:start + batch_size]
                x = arrays[key][ix].astype(np.float64)
                valid = np.isfinite(x)
                if key == "history":
                    valid &= arrays["mask"][ix][..., None]
                x = np.where(valid, x, 0.).reshape(-1, width)
                total += x.sum(0)
                square += (x * x).sum(0)
                count += valid.reshape(-1, width).sum(0)
            denominator = np.maximum(count, 1)
            mean = total / denominator
            variance = np.maximum(square / denominator - mean * mean, 0.)
            scale = np.sqrt(variance)
            scale[scale < 1e-6] = 1.
            means[key], scales[key], counts[key] = mean, scale, count
        return cls(means, scales, counts)

    def batch(self, arrays, ix, vocab_sizes):
        result = {}
        for key in ("history", "players", "context"):
            x = arrays[key][ix]
            valid = np.isfinite(x) & (self.counts[key] > 0)
            if key == "history":
                valid &= arrays["mask"][ix][..., None]
            width = x.shape[-1]
            value = np.empty(x.shape[:-1] + (2 * width,), dtype=np.float32)
            normalized = value[..., :width]
            np.subtract(x, self.mean[key], out=normalized, casting="unsafe")
            np.divide(normalized, self.scale[key], out=normalized, casting="unsafe")
            normalized[~valid] = 0.
            value[..., width:] = valid
            result[key] = torch.from_numpy(value)
        for key in ("champions", "patches", "competitions"):
            codes = arrays[key][ix]
            result[key] = torch.from_numpy(np.where(codes < vocab_sizes[key], codes, 1).astype(np.int64, copy=False))
        result["roles"] = torch.from_numpy(arrays["roles"][ix].astype(np.int64, copy=False))
        result["mask"] = torch.from_numpy(arrays["mask"][ix])
        result["role_probs"] = torch.from_numpy(arrays["role_probs"][ix].astype(np.float32, copy=False))
        result["best_of"] = torch.from_numpy(arrays["best_of"][ix].astype(np.int64, copy=False))
        return result


class _HistoryNetwork(nn.Module):
    def __init__(self, dimensions, vocab_sizes, config):
        super().__init__()
        h, p, c = dimensions
        w, e = config["width"], config["embedding_dim"]
        self.champion = nn.Embedding(vocab_sizes["champions"], e)
        self.patch = nn.Embedding(vocab_sizes["patches"], 4)
        self.competition = nn.Embedding(vocab_sizes["competitions"], 4)
        self.role = nn.Embedding(6, e)
        self.age = nn.Parameter(torch.zeros(16, w))
        self.performance = nn.Linear(2 * h, w)
        self.champion_performance = nn.Linear(e, w)
        self.token = nn.Linear(w + 2 * e + 3 + 8, w)
        self.attention = nn.Linear(w, 1)
        self.cold = nn.Parameter(torch.zeros(w))
        self.player_strength = nn.Sequential(nn.Linear(w + 2 * p, w), nn.Tanh())
        self.player_style = nn.Sequential(nn.Linear(w + 2 * p, w), nn.Tanh())
        self.roster_strength = nn.Sequential(nn.Linear(7 * w, w), nn.Tanh())
        self.roster_style = nn.Sequential(nn.Linear(7 * w, w), nn.Tanh())
        self.score_head = nn.Sequential(nn.Linear(4 * w + 2 * c, w), nn.Tanh(), nn.Linear(w, 3))
        self.map_embedding = nn.Embedding(5, e)
        self.prop_head = nn.Sequential(nn.Linear(4 * w + 2 * c + e, w), nn.Tanh(), nn.Linear(w, 4))
        self.register_buffer("map_ids", torch.arange(5))
        self.register_buffer("loser_scores", torch.tensor([0, 1, 2, 0, 1, 2]))

    def forward(self, x, champion_enabled, compute_props=True):
        mask = x["mask"]
        # Includes missing/UNK/known status; all are removed for noChampion.
        if champion_enabled:
            champion = self.champion(x["champions"])
            status = F.one_hot(x["champions"].clamp_max(2), 3).to(champion.dtype)
        else:
            champion = x["history"].new_zeros(mask.shape + (self.champion.embedding_dim,))
            status = x["history"].new_zeros(mask.shape + (3,))
        performance = torch.tanh(self.performance(x["history"]))
        interaction = performance * (1. + torch.tanh(self.champion_performance(champion)))
        token = torch.tanh(self.token(torch.cat((interaction, champion, self.role(x["roles"]), status,
                                                self.patch(x["patches"]), self.competition(x["competitions"])), -1)) + self.age)
        logits = self.attention(token).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        has_history = mask.any(-1)
        # All-padding rows have a defined empty distribution, not softmax(-inf).
        logits = torch.where(has_history[..., None], logits, torch.zeros_like(logits))
        weights = logits.softmax(-1) * mask
        history = (token * weights[..., None]).sum(-2)
        history = history + (~has_history)[..., None] * self.cold
        player = torch.cat((history, x["players"]), -1)
        strength = self.player_strength(player)
        style = self.player_style(player)
        roles = x["role_probs"]
        def aggregate(values, head):
            weighted = torch.einsum("ntpr,ntpw->ntrw", roles, values)
            weighted = weighted / roles.sum(-2).clamp_min(1e-6)[..., None]
            return head(torch.cat((values.mean(-2), weighted.flatten(-2)), -1))
        strength = aggregate(strength, self.roster_strength)
        style = aggregate(style, self.roster_style)
        context = x["context"]
        a = torch.cat((strength[:, 0], strength[:, 1], style[:, 0], style[:, 1], context), -1)
        b = torch.cat((strength[:, 1], strength[:, 0], style[:, 1], style[:, 0], context), -1)
        oriented = self.score_head(torch.stack((a, b), 1)).flatten(1)
        legal = self.loser_scores[None, :] <= (x["best_of"] // 2)[:, None]
        score_logits = oriented.masked_fill(~legal, -torch.inf)
        if not compute_props:
            return score_logits, None
        symmetric = torch.cat((strength.sum(1), (strength[:, 0] - strength[:, 1]).abs(),
                               style.sum(1), (style[:, 0] - style[:, 1]).abs(), context), -1)
        prop_input = torch.cat((symmetric[:, None, :].expand(-1, 5, -1),
                                self.map_embedding(self.map_ids)[None, :, :].expand(len(a), -1, -1)), -1)
        raw = self.prop_head(prop_input)
        # Parametric support bounds match existing NB/lognormal classes. They
        # constrain network outputs only: no label or log likelihood is clipped.
        log_mu = raw[..., 0].clamp(-12., 12.)
        log_alpha = raw[..., 1].clamp(-12., 5.)
        log_duration = raw[..., 2].clamp(-12., 12.)
        log_sigma = raw[..., 3].clamp(-7., 2.)
        return score_logits, (log_mu.exp(), log_alpha.exp(), log_duration, log_sigma.exp())


class HistorySeedModel:
    def __init__(self, *, config, seed, champion_enabled, multitask, dimensions, vocab_sizes, preprocessing, network):
        self.config = dict(config)
        self.seed = seed
        self.champion_enabled = champion_enabled
        self.multitask = multitask
        self.dimensions = tuple(dimensions)
        self.vocab_sizes = dict(vocab_sizes)
        self.vocab_size = vocab_sizes["champions"]
        self.preprocessing = preprocessing
        self.network = network
        self.fit_receipt = {}
        self.optimizer_state = None
        self.state_dict = None

    def predict(self, arrays, indices=None):
        values, _ = _validate_arrays(arrays, self.dimensions)
        ix = _indices(indices, len(values["best_of"]))
        torch.set_num_threads(self.config["threads"])
        scores = []
        props = [[], [], [], []]
        # Fitted networks are already eval() and prediction mutates no model state.
        with torch.inference_mode():
            for start in range(0, len(ix), self.config["batch_size"]):
                batch = self.preprocessing.batch(values, ix[start:start + self.config["batch_size"]], self.vocab_sizes)
                logits, parameters = self.network(batch, self.champion_enabled, self.multitask)
                # Double softmax gives normalized coherent consumer probabilities.
                scores.append(logits.double().softmax(-1).numpy())
                if self.multitask:
                    for collected, parameter in zip(props, parameters):
                        collected.append(parameter.double().numpy())
        score_probs = np.concatenate(scores) if scores else np.empty((0, 6))
        optional = {}
        if self.multitask:
            optional = {name: np.concatenate(value) if value else np.empty((0, 5)) for name, value in
                        zip(("kills_mu", "kills_alpha", "duration_log_mu", "duration_log_sigma"), props)}
        return HistoryPrediction(score_probs, values["best_of"][ix], **optional)


def _labels(series, maps, values, indices):
    n = len(values["best_of"])
    if len(series) != n or "score_index" not in series:
        raise ValueError("series must align with arrays and contain score_index")
    scores = series["score_index"].to_numpy(dtype=np.float64)
    observed = np.isfinite(scores)
    if np.isinf(scores).any() or np.any((scores[observed] < 0) | (scores[observed] > 5) | (scores[observed] != np.floor(scores[observed]))):
        raise ValueError("score labels must be legal integer indices or NaN")
    rows = np.flatnonzero(observed)
    if np.any(~_legal(values["best_of"])[rows, scores[rows].astype(int)]):
        raise ValueError("score label outside legal best_of support")
    if not observed[indices].any():
        raise ValueError("split requires observed terminal score labels")
    required = ("row_index", "map_index", "kills", "duration_minutes")
    if any(key not in maps for key in required):
        raise ValueError("maps missing required label/index columns")
    row = maps["row_index"].to_numpy(dtype=np.float64)
    mi = maps["map_index"].to_numpy(dtype=np.float64)
    if not np.isfinite(row).all() or np.any((row != np.floor(row)) | (row < 0) | (row >= n)):
        raise ValueError("map row_index must be a valid series index")
    row = row.astype(np.int64)
    if not np.isfinite(mi).all() or np.any((mi != np.floor(mi)) | (mi < 1) | (mi > values["best_of"][row])):
        raise ValueError("map_index must be legal for its series")
    mi = mi.astype(np.int64) - 1
    if len(np.unique(row * 5 + mi)) != len(row):
        raise ValueError("duplicate recorded map index in a series")
    if np.any(observed[row] & (mi + 1 > (values["best_of"][row] // 2 + 1 + np.nan_to_num(scores[row], nan=0.).astype(int) % 3))):
        raise ValueError("recorded map beyond observed terminal series length")
    labels = {"score": scores}
    for task, column in (("kills", "kills"), ("duration", "duration_minutes")):
        y = maps[column].to_numpy(dtype=np.float64)
        if np.isinf(y).any():
            raise ValueError(f"{task} labels cannot contain infinity")
        seen = np.isfinite(y)
        if task == "kills" and np.any((y[seen] < 0) | (y[seen] != np.floor(y[seen]))):
            raise ValueError("kills labels must be nonnegative integer counts or NaN")
        if task == "duration" and np.any(y[seen] <= 0):
            raise ValueError("duration labels must be positive minutes or NaN")
        matrix = np.full((n, 5), np.nan, dtype=np.float64)
        matrix[row, mi] = y
        labels[task] = matrix
    return labels


def _prop_losses(parameters, labels):
    mu, alpha, log_mu, sigma = (p.double() for p in parameters)
    losses = {}
    for task in ("kills", "duration"):
        y = labels[task]
        valid = torch.isfinite(y)
        rows, columns = valid.nonzero(as_tuple=True)
        observed = y[rows, columns]
        if task == "kills":
            mean, dispersion = mu[rows, columns], alpha[rows, columns]
            r = dispersion.reciprocal()
            log_prob = (torch.lgamma(observed + r) - torch.lgamma(r) - torch.lgamma(observed + 1.)
                        - r * torch.log1p(dispersion * mean)
                        + observed * (torch.log(dispersion * mean) - torch.log1p(dispersion * mean)))
            nll = -log_prob
        else:
            spread = sigma[rows, columns]
            log_y = observed.log()
            nll = log_y + spread.log() + .5 * math.log(2. * math.pi) + .5 * ((log_y - log_mu[rows, columns]) / spread) ** 2
        total = torch.zeros(len(y), dtype=torch.float64).scatter_add(0, rows, nll)
        count = valid.sum(-1)
        active = count > 0
        series_loss = total[active] / count[active]
        losses[task] = (series_loss.sum(), int(active.sum()))
    return losses


def _parameter_digest(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_history_model(arrays, series, maps, train_indices, *, config, seed, champion_enabled, multitask, software_smoke=False):
    """Twelve formal epochs, or an explicit one-epoch TRAIN-only software smoke."""
    if not isinstance(software_smoke, bool):
        raise ValueError("software_smoke must be boolean")
    recipe = _configuration(config, software_smoke)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a nonnegative int32 integer")
    if not isinstance(champion_enabled, bool) or not isinstance(multitask, bool):
        raise ValueError("champion_enabled and multitask must be boolean")
    values, dimensions = _validate_arrays(arrays)
    indices = _indices(train_indices, len(values["best_of"]))
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("TRAIN indices must be nonempty and unique")
    if software_smoke and len(indices) > 256:
        raise ValueError("software smoke is limited to 256 TRAIN series")
    labels = _labels(series, maps, values, indices)
    if multitask and any(not np.isfinite(labels[task][indices]).any() for task in ("kills", "duration")):
        raise ValueError("multitask TRAIN requires observed kills and duration labels")
    torch.set_num_threads(recipe["threads"])
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    preprocessing = HistoryPreprocessor.fit(values, indices, recipe["batch_size"])
    vocab_sizes = {key: max(2, int(values[key][indices].max()) + 1)
                   for key in ("champions", "patches", "competitions")}
    network = _HistoryNetwork(dimensions, vocab_sizes, recipe)
    # Initialize all variants from the same TRAIN-derived baseline, even when
    # their prop heads are inactive. No CAL label is involved.
    kills = labels["kills"][indices]
    duration = labels["duration"][indices]
    mean_kills = float(np.nanmean(kills)) if np.isfinite(kills).any() else 1.
    mean_log_duration = float(np.nanmean(np.log(duration))) if np.isfinite(duration).any() else 0.
    with torch.no_grad():
        network.prop_head[-1].bias.copy_(torch.tensor([math.log(max(mean_kills, 1e-6)), math.log(.1),
                                                     mean_log_duration, math.log(.25)]))
    initial_digest = _parameter_digest(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    model = HistorySeedModel(config=recipe, seed=int(seed), champion_enabled=champion_enabled,
                             multitask=multitask, dimensions=dimensions, vocab_sizes=vocab_sizes,
                             preprocessing=preprocessing, network=network)
    losses, order_digests = [], []
    steps = 0
    for epoch in range(recipe["epochs"]):
        order = rng.permutation(indices)
        order_digests.append(hashlib.sha256(order.tobytes()).hexdigest())
        sums = {"score": 0., "kills": 0., "duration": 0.}
        counts = dict.fromkeys(sums, 0)
        for start in range(0, len(order), recipe["batch_size"]):
            ix = order[start:start + recipe["batch_size"]]
            batch = preprocessing.batch(values, ix, vocab_sizes)
            logits, parameters = network(batch, champion_enabled, multitask)
            target = torch.from_numpy(labels["score"][ix])
            valid = torch.isfinite(target)
            terms = {"score": (F.cross_entropy(logits[valid].double(), target[valid].long(), reduction="sum"), int(valid.sum()))}
            if multitask:
                terms.update(_prop_losses(parameters, {task: torch.from_numpy(labels[task][ix]) for task in ("kills", "duration")}))
            # Graph-connected zero permits complete TRAIN traversal even for a
            # batch whose outcomes are all absent, without inventing labels.
            loss = logits[:, 0].sum() * 0.
            for task, (total, count) in terms.items():
                if count:
                    weight = 1. if task == "score" else recipe[task + "_weight"]
                    loss = loss + weight * total / count
                    sums[task] += float(total.detach())
                    counts[task] += count
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite TRAIN likelihood at epoch {epoch + 1}, step {steps + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
        losses.append({"epoch": epoch + 1, "losses": {key: sums[key] / counts[key] if counts[key] else None for key in sums},
                       "observed_series": counts, "series_seen": len(order)})
    network.eval()
    model.optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.state_dict = copy.deepcopy(network.state_dict())
    optimizer_steps = sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})
    model.fit_receipt = {
        "seed": int(seed), "config": dict(recipe), "champion_enabled": champion_enabled, "multitask": multitask,
        "software_smoke": software_smoke,
        "epochs_completed": recipe["epochs"], "optimizer_steps": steps, "optimizer_parameter_steps": optimizer_steps,
        "train_series_count": len(indices), "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "train_score_count": int(np.isfinite(labels["score"][indices]).sum()),
        "train_map_label_counts": {task: int(np.isfinite(labels[task][indices]).sum()) for task in ("kills", "duration")},
        "parameter_count": sum(p.numel() for p in network.parameters()), "epoch_losses": losses,
        "epoch_order_sha256": order_digests, "initial_state_sha256": initial_digest,
        "final_state_sha256": _parameter_digest(network), "torch_version": str(torch.__version__),
        "device": "cpu", "threads": recipe["threads"], "optimizer": "AdamW", "preprocessing_split": "TRAIN",
        "history_length": 16, "vocab_sizes": dict(vocab_sizes),
    }
    return model


def _positive_scalar(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _temper(probabilities, temperature):
    if temperature == 1.:
        return probabilities.copy()
    with np.errstate(divide="ignore"):
        logs = np.log(probabilities) / temperature
    return np.exp(logs - logsumexp(logs, axis=1, keepdims=True))


def _scaled_prediction(prediction, temperature, kill_scale, duration_scale):
    optional = {}
    if prediction.kills_mu is not None:
        optional = {"kills_mu": prediction.kills_mu, "kills_alpha": prediction.kills_alpha * kill_scale,
                    "duration_log_mu": prediction.duration_log_mu,
                    "duration_log_sigma": prediction.duration_log_sigma * duration_scale}
    return HistoryPrediction(_temper(prediction.score_probs, temperature), prediction.best_of, **optional)


class HistoryEnsemble:
    """Three complete seed forecasts; never average NB/lognormal parameters."""
    def __init__(self, models, *, score_temperature=1., kill_scale=1., duration_scale=1.):
        self.models = tuple(models)
        if len(self.models) != 3 or not all(isinstance(m, HistorySeedModel) for m in self.models):
            raise ValueError("history ensemble requires exactly three fitted seed models")
        if any(m.multitask != self.models[0].multitask or m.dimensions != self.models[0].dimensions for m in self.models):
            raise ValueError("seed models must share tasks and input dimensions")
        self.multitask = self.models[0].multitask
        self.score_temperature = _positive_scalar(score_temperature, "score_temperature")
        self.kill_scale = _positive_scalar(kill_scale, "kill_scale")
        self.duration_scale = _positive_scalar(duration_scale, "duration_scale")
        if not self.multitask and (self.kill_scale != 1. or self.duration_scale != 1.):
            raise ValueError("score-only ensemble cannot calibrate untrained props")
        self.calibration_receipt = {}

    def predict(self, arrays, indices=None):
        components = tuple(_scaled_prediction(model.predict(arrays, indices), self.score_temperature,
                                              self.kill_scale, self.duration_scale) for model in self.models)
        scores = np.stack([component.score_probs for component in components]).mean(0)
        # Ensembles intentionally expose no averaged raw parametric fields.
        return HistoryPrediction(scores, components[0].best_of, components=components)


def _series_mean(loss, inverse, count):
    total = np.bincount(inverse, weights=loss, minlength=len(count))
    return float((total / count).mean())


def _calibrate_scalar(objective, name):
    bounds = tuple(np.log(CALIBRATION_BOUNDS))
    def penalized(log_value):
        return objective(float(np.exp(log_value))) + CALIBRATION_REGULARIZATION * log_value ** 2
    result = minimize_scalar(penalized, bounds=bounds, method="bounded", options={"xatol": 1e-7, "maxiter": 300})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"{name} CAL optimization failed: {result.message}")
    candidates = [float(result.x), *bounds, 0.]
    chosen = min(candidates, key=penalized)
    value = float(np.exp(chosen))
    return value, {"value": value, "bounds": list(CALIBRATION_BOUNDS), "regularization": CALIBRATION_REGULARIZATION,
                   "nll": float(objective(value)), "penalized_objective": float(penalized(chosen)),
                   "nfev": int(result.nfev), "boundary": chosen in bounds, "status": "success"}


def calibrate_history_ensemble(models, arrays, series, maps, cal_indices):
    """Independent CAL fits of score temperature and exact mixture dispersions."""
    ensemble = HistoryEnsemble(models)
    values, _ = _validate_arrays(arrays, ensemble.models[0].dimensions)
    indices = _indices(cal_indices, len(values["best_of"]))
    if len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("CAL indices must be nonempty and unique")
    labels = _labels(series, maps, values, indices)
    predictions = [model.predict(values, indices) for model in ensemble.models]
    target = labels["score"][indices]
    valid = np.isfinite(target)
    def score_loss(temperature):
        probabilities = np.stack([_temper(pred.score_probs, temperature) for pred in predictions]).mean(0)
        with np.errstate(divide="ignore"):
            return float(-np.log(probabilities[valid, target[valid].astype(int)]).mean())
    ensemble.score_temperature, score_receipt = _calibrate_scalar(score_loss, "score temperature")
    receipt = {"split": "CAL", "series_count": len(indices), "cal_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "score_temperature": score_receipt, "score_observed_series": int(valid.sum()), "scalar_count": 1,
               "mixture_components": len(predictions), "frozen_networks": True, "dispersion_objective": "per-series mean exact mixture NLL"}
    if ensemble.multitask:
        for task, parameter_name in (("kills", "kill_scale"), ("duration", "duration_scale")):
            matrix = labels[task][indices]
            rows, columns = np.nonzero(np.isfinite(matrix))
            if len(rows) == 0:
                raise ValueError(f"CAL requires observed {task} labels")
            y = matrix[rows, columns]
            _, inverse = np.unique(rows, return_inverse=True)
            count = np.bincount(inverse)
            if task == "kills":
                means = np.stack([p.kills_mu[rows, columns] for p in predictions])
                dispersion = np.stack([p.kills_alpha[rows, columns] for p in predictions])
                def objective(scale):
                    alpha = dispersion * scale
                    logs = nbinom.logpmf(y[None, :], 1. / alpha, 1. / (1. + alpha * means))
                    return _series_mean(-(logsumexp(logs, axis=0) - math.log(3.)), inverse, count)
            else:
                means = np.stack([p.duration_log_mu[rows, columns] for p in predictions])
                dispersion = np.stack([p.duration_log_sigma[rows, columns] for p in predictions])
                log_y = np.log(y)[None, :]
                def objective(scale):
                    sigma = dispersion * scale
                    logs = -log_y - np.log(sigma) - .5 * math.log(2. * math.pi) - .5 * ((log_y - means) / sigma) ** 2
                    return _series_mean(-(logsumexp(logs, axis=0) - math.log(3.)), inverse, count)
            scalar, detail = _calibrate_scalar(objective, parameter_name)
            setattr(ensemble, parameter_name, scalar)
            detail.update({"observed_maps": len(rows), "observed_series": len(np.unique(rows))})
            receipt[parameter_name] = detail
            receipt["scalar_count"] += 1
    ensemble.calibration_receipt = receipt
    return ensemble, receipt
