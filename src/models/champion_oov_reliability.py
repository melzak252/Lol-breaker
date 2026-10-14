"""Frozen W1 categorical adapters for the unchanged champion-history backbone.

Every prediction requires an explicit ``identity_state`` whose ``n_prior`` arrays
share the *current* input's target/side/player/token axes. Passing full arrays and
full sidecars together with ``indices`` is supported; constructor row positions
are never a prediction fallback. Source replay and execution authorization belong
to the study runner, not this module. No original model closure is modified.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.champion_history_model import (
    CALIBRATION_BOUNDS,
    CALIBRATION_REGULARIZATION,
    DEFAULT_CONFIG,
    HistoryEnsemble,
    HistoryPrediction,
    HistoryPreprocessor,
    HistorySeedModel,
    _HistoryNetwork,
    _calibrate_scalar,
    _configuration,
    _indices,
    _labels,
    _legal,
    _parameter_digest,
    _positive_scalar,
    _prop_losses,
    _scaled_prediction,
    _temper,
    _validate_arrays,
)

CHANNELS = ("champion", "patch", "competition")
ARRAY_CHANNELS = {"champion": "champions", "patch": "patches", "competition": "competitions"}
CHANNEL_WIDTHS = {"champion": 8, "patch": 4, "competition": 4}
PROJECTOR_SEED = 20261009
MODEL_SEEDS = (20261007, 20261008, 20261009)
HISTORY_FEATURE_NAMES = (
    "kills", "deaths", "assists", "csm", "gpm", "dpm", "vspm", "gd@15",
    "csd@15", "xpd@15", "kp%", "dmg%", "kills_for", "kills_against",
    "total_kills", "duration_minutes", "prior_opponent_strength_mean",
    "prior_opponent_strength_sd", "played_age_days", "release_age_days",
)
PHI_SOURCE_INDICES = tuple(range(17)) + (18, 19)
PHI_NAMES = tuple(HISTORY_FEATURE_NAMES[i] for i in PHI_SOURCE_INDICES) + tuple(
    "observed_" + name for name in HISTORY_FEATURE_NAMES[:12]
) + ("role_TOP", "role_JUNGLE", "role_MID", "role_ADC", "role_SUPPORT")


def _channel(channel):
    if channel not in CHANNELS:
        raise ValueError("channel must be champion, patch or competition")


def _missing(identity):
    return identity is None or (isinstance(identity, str) and identity == "") or (
        isinstance(identity, (float, np.floating)) and np.isnan(identity)
    )


def held_identity(channel, canonical_id):
    """Permanent entity partition; canonical strings are encoded verbatim."""
    _channel(channel)
    if not isinstance(canonical_id, str) or not canonical_id:
        raise ValueError("identity must be a nonmissing canonical string")
    digest = hashlib.sha256(f"{channel}:{canonical_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 5 == 0


def _vocabulary(channel, vocabulary):
    _channel(channel)
    if not isinstance(vocabulary, dict):
        raise ValueError("identity vocabulary must map canonical strings to integer codes")
    for identity, code in vocabulary.items():
        if not isinstance(identity, str) or not identity:
            raise ValueError("identity vocabulary contains a missing canonical identity")
        if isinstance(code, (bool, np.bool_)) or not isinstance(code, (int, np.integer)) or code < 2:
            raise ValueError("identity vocabulary codes must be integers >=2")
    if len(set(vocabulary.values())) != len(vocabulary):
        raise ValueError("identity vocabulary contains duplicate codes")
    return dict(vocabulary)


def _prior_counts(value, shape):
    counts = np.asarray(value)
    if counts.shape != shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("n_prior must be nonnegative integer counts on the current token axes")
    if counts.dtype.kind == "u" and np.any(counts > np.iinfo(np.int64).max):
        raise ValueError("n_prior exceeds int64 support")
    return counts.astype(np.int64, copy=False)


def held_channel_codes(channel, codes, vocabulary, mask):
    vocabulary = _vocabulary(channel, vocabulary)
    codes, mask = np.asarray(codes), np.asarray(mask)
    if codes.shape != mask.shape or codes.dtype.kind not in "iu" or np.any(codes < 0) or mask.dtype.kind != "b":
        raise ValueError("identity codes and boolean mask must have matching token shapes")
    result = codes.copy()
    held_codes = [code for identity, code in vocabulary.items() if held_identity(channel, identity)]
    if held_codes:
        result[mask & np.isin(codes, held_codes)] = 1
    return result


def identity_residual_weights(channel, canonical_ids, n_prior, mask, train_vocabulary, *, held_out):
    vocabulary = _vocabulary(channel, train_vocabulary)
    identities, mask = np.asarray(canonical_ids, dtype=object), np.asarray(mask)
    if identities.shape != mask.shape or mask.dtype.kind != "b" or not isinstance(held_out, bool):
        raise ValueError("identity and boolean mask shapes must align; held_out must be boolean")
    counts = _prior_counts(n_prior, identities.shape)
    allowed = np.zeros(identities.shape, dtype=bool)
    for identity in set(identities.ravel()):
        if _missing(identity):
            continue
        if not isinstance(identity, str):
            raise ValueError("identity must be a canonical string or missing")
        if identity in vocabulary and not (held_out and held_identity(channel, identity)):
            allowed |= identities == identity
    return np.where(mask & allowed, counts.astype(np.float64) / (counts + 20.), 0.)


def compose_identity(phi36, residuals, n_prior, mask, known, held, weight, bias, *, missing=None):
    """NumPy reference-facing composition API; no input is modified."""
    phi, residual = np.asarray(phi36), np.asarray(residuals)
    weight, bias = np.asarray(weight), np.asarray(bias)
    if phi.ndim < 1 or phi.shape[-1] != 36 or weight.ndim != 2 or weight.shape[1] != 36:
        raise ValueError("composition requires exactly 36 phi fields")
    width, shape = weight.shape[0], phi.shape[:-1]
    if width not in (8, 4) or bias.shape != (width,) or residual.shape != shape + (width,):
        raise ValueError("composition residual/projector width must be 8 or 4 with matching shapes")
    flags = [np.asarray(flag) for flag in (mask, known, held)]
    if missing is None:
        missing = np.zeros(shape, dtype=bool)
    flags.append(np.asarray(missing))
    if any(flag.shape != shape or flag.dtype.kind != "b" for flag in flags):
        raise ValueError("composition identity flags must be boolean and share token axes")
    mask, known, held, missing = flags
    if np.any(missing & (known | held)) or np.any(held & ~known):
        raise ValueError("composition identity flags are inconsistent")
    if any(x.dtype.kind not in "fiu" or not np.isfinite(x).all() for x in (phi, residual, weight, bias)):
        raise ValueError("composition numeric inputs must be finite")
    counts = _prior_counts(n_prior, shape)
    support = np.where(known & ~held, counts.astype(np.float64) / (counts + 20.), 0.)
    support = np.where(missing, 1., support)
    result = phi @ weight.T + bias + support[..., None] * residual
    return np.where(mask[..., None], result, 0.)


@dataclass
class CompositionPreprocessor:
    """TRAIN normalizers shared with the inherited forty-channel history stream."""
    mean: dict[str, np.ndarray]
    scale: dict[str, np.ndarray]
    counts: dict[str, np.ndarray]
    feature_names: tuple[str, ...] = HISTORY_FEATURE_NAMES
    source_indices: tuple[int, ...] = PHI_SOURCE_INDICES
    phi_names: tuple[str, ...] = PHI_NAMES

    @property
    def count(self):
        return self.counts

    @classmethod
    def from_history(cls, preprocessing, history_feature_names):
        if tuple(history_feature_names) != HISTORY_FEATURE_NAMES:
            raise ValueError("history feature names/order must match the twenty inherited fields")
        if preprocessing.mean["history"].shape != (20,):
            raise ValueError("history feature width must be exactly twenty")
        return cls(preprocessing.mean, preprocessing.scale, preprocessing.counts)

    @classmethod
    def fit(cls, arrays, train_indices, *, history_feature_names):
        values, _ = _validate_arrays(arrays)
        indices = _train_indices(train_indices, len(values["best_of"]))
        preprocessing = HistoryPreprocessor.fit(values, indices, DEFAULT_CONFIG["batch_size"])
        return cls.from_history(preprocessing, history_feature_names)

    def batch(self, arrays, ix, *, inherited_batch=None):
        if inherited_batch is None:
            # Reuse the original subtract-to-FP32 then divide-to-FP32 operations.
            preprocessing = HistoryPreprocessor(self.mean, self.scale, self.counts)
            vocab_sizes = {key: 2 for key in ARRAY_CHANNELS.values()}
            inherited_batch = preprocessing.batch(arrays, ix, vocab_sizes)
        history = inherited_batch["history"]
        roles, mask = inherited_batch["roles"], inherited_batch["mask"]
        numeric = history[..., list(self.source_indices)]
        observed = history[..., 20:32]
        role_bits = (roles[..., None] == torch.arange(1, 6)).to(torch.float32)
        return torch.cat((numeric, observed, role_bits), -1) * mask[..., None]

    def transform(self, arrays):
        values, dimensions = _validate_arrays(arrays)
        if dimensions[0] != 20:
            raise ValueError("history feature width must be exactly twenty")
        result = np.empty(values["mask"].shape + (36,), dtype=np.float32)
        for start in range(0, len(result), DEFAULT_CONFIG["batch_size"]):
            ix = np.arange(start, min(start + DEFAULT_CONFIG["batch_size"], len(result)))
            result[ix] = self.batch(values, ix).numpy()
        return result

    def to_dict(self):
        return {"mean": {key: value.tolist() for key, value in self.mean.items()},
                "scale": {key: value.tolist() for key, value in self.scale.items()},
                "counts": {key: value.tolist() for key, value in self.counts.items()},
                "feature_names": list(self.feature_names), "source_indices": list(self.source_indices),
                "phi_names": list(self.phi_names), "preprocessing_split": "TRAIN",
                "numeric_rounding": "inherited HistoryPreprocessor.batch subtract and divide outputs FP32"}


class _CompositionEmbedding(nn.Module):
    """Reuse an existing residual Parameter; register the projector only once."""
    def __init__(self, embedding, projector):
        super().__init__()
        self.weight = embedding.weight
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        object.__setattr__(self, "_projector", projector)
        self._batch_context = None

    def forward(self, codes):
        if self._batch_context is None:
            raise RuntimeError("composition embedding requires explicit current-batch support")
        phi, counts, mask = self._batch_context
        support = counts.to(phi.dtype)
        support = support / (support + 20.)
        support = torch.where(codes >= 2, support, torch.zeros_like(support))
        support = torch.where(codes == 0, torch.ones_like(support), support)
        residual = F.embedding(codes, self.weight)
        return (self._projector(phi) + support[..., None] * residual) * mask[..., None]


class _OovNetwork(_HistoryNetwork):
    """Only categorical embedding calls change; original forward is inherited."""
    def __init__(self, dimensions, vocab_sizes, config, arm):
        super().__init__(dimensions, vocab_sizes, config)
        self.arm = arm
        if arm in ("C", "HC"):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(PROJECTOR_SEED)
                self.projectors = nn.ModuleDict({channel: nn.Linear(36, CHANNEL_WIDTHS[channel]) for channel in CHANNELS})
            for channel in CHANNELS:
                setattr(self, channel, _CompositionEmbedding(getattr(self, channel), self.projectors[channel]))

    def forward(self, x, champion_enabled, compute_props=True):
        if self.arm == "H":
            return super().forward(x, champion_enabled, compute_props)
        try:
            for channel in CHANNELS:
                getattr(self, channel)._batch_context = (x["phi"], x["n_prior"][channel], x["mask"])
            return super().forward(x, champion_enabled, compute_props)
        finally:
            for channel in CHANNELS:
                getattr(self, channel)._batch_context = None


def _train_indices(indices, n):
    result = _indices(indices, n)
    if not len(result) or len(np.unique(result)) != len(result) or np.any(result[1:] <= result[:-1]):
        raise ValueError("TRAIN indices must be nonempty, unique and in ascending canonical order")
    return result


def _recipe(config):
    recipe = _configuration(config)
    if recipe != DEFAULT_CONFIG:
        raise ValueError("W1 uses the complete fixed original DEFAULT_CONFIG; recipe overrides are forbidden")
    return recipe


def _identity_state(values, identity_state, vocabularies):
    if not isinstance(identity_state, dict) or not isinstance(identity_state.get("n_prior"), dict):
        raise ValueError("explicit identity_state.n_prior sidecars are required on current input axes")
    if set(identity_state["n_prior"]) != set(CHANNELS):
        raise ValueError("n_prior sidecars require exactly champion, patch and competition")
    counts = {channel: _prior_counts(identity_state["n_prior"][channel], values["mask"].shape) for channel in CHANNELS}
    canonical = identity_state.get("canonical_identities")
    if canonical is not None:
        if not isinstance(canonical, dict) or set(canonical) != set(CHANNELS):
            raise ValueError("canonical identity sidecars require all three channels")
        for channel in CHANNELS:
            ids = np.asarray(canonical[channel], dtype=object)
            if ids.shape != values["mask"].shape:
                raise ValueError("canonical identity sidecars must share current token axes")
            vocabulary = vocabularies[channel]
            codes = values[ARRAY_CHANNELS[channel]]
            def code_for(identity):
                if _missing(identity):
                    return 0
                if not isinstance(identity, str):
                    raise ValueError("canonical identity must be a string or missing")
                return vocabulary.get(identity, 1)
            # One traversal, not one full token-array scan per canonical ID.
            observed_ids = ids[values["mask"]]
            expected = np.fromiter((code_for(identity) for identity in observed_ids), np.int64, len(observed_ids))
            if np.any(codes[values["mask"]] != expected):
                raise ValueError(f"{channel} canonical identity/code sidecars are inconsistent")
    return counts


class OovSeedModel(HistorySeedModel):
    """Serializable seed model with explicit, batch-only causal support sidecars."""
    def __init__(self, *, arm, vocabularies, composition_preprocessing, **kwargs):
        super().__init__(**kwargs)
        self.arm = arm
        self.vocabularies = vocabularies
        self.composition_preprocessing = composition_preprocessing
        self.initial_state = None
        self.final_state = None

    def _batch(self, values, ix, counts):
        batch = self.preprocessing.batch(values, ix, self.vocab_sizes)
        if self.arm in ("H", "HC"):
            for channel in CHANNELS:
                key = ARRAY_CHANNELS[channel]
                routed = held_channel_codes(channel, batch[key].numpy(), self.vocabularies[channel], batch["mask"].numpy())
                batch[key] = torch.from_numpy(routed)
        if self.arm in ("C", "HC"):
            batch["phi"] = self.composition_preprocessing.batch(values, ix, inherited_batch=batch)
            batch["n_prior"] = {channel: torch.from_numpy(counts[channel][ix]) for channel in CHANNELS}
        return batch

    def predict(self, arrays, indices=None, *, identity_state=None):
        values, _ = _validate_arrays(arrays, self.dimensions)
        counts = _identity_state(values, identity_state, self.vocabularies)
        ix = _indices(indices, len(values["best_of"]))
        torch.set_num_threads(self.config["threads"])
        scores, props = [], [[], [], [], []]
        raw_logits = []
        with torch.inference_mode():
            for start in range(0, len(ix), self.config["batch_size"]):
                batch = self._batch(values, ix[start:start + self.config["batch_size"]], counts)
                logits, parameters = self.network(batch, True, True)
                raw_logits.append(logits.numpy())
                scores.append(logits.double().softmax(-1).numpy())
                for collected, parameter in zip(props, parameters):
                    collected.append(parameter.double().numpy())
        optional = {name: np.concatenate(value) if value else np.empty((0, 5)) for name, value in zip(
            ("kills_mu", "kills_alpha", "duration_log_mu", "duration_log_sigma"), props)}
        prediction = HistoryPrediction(np.concatenate(scores) if scores else np.empty((0, 6)), values["best_of"][ix], **optional)
        # Audit-only export from the same forward, including masked illegal -inf.
        # Temperature-transformed predictions deliberately do not claim logits.
        prediction.score_logits = np.concatenate(raw_logits) if raw_logits else np.empty((0, 6), dtype=np.float32)
        return prediction


def build_initial_model(arrays, train_indices, *, seed, arm, schema, identity_state, config):
    """Initialize an unfitted fixture; no optimizer or fit attempt is invoked."""
    recipe = _recipe(config)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed <= np.iinfo(np.int32).max:
        raise ValueError("seed must be a nonnegative int32 integer")
    if arm not in ("H", "C", "HC"):
        raise ValueError("arm must be H, C or HC")
    values, dimensions = _validate_arrays(arrays)
    indices = _train_indices(train_indices, len(values["best_of"]))
    if not isinstance(schema, dict) or "history_feature_names" not in schema:
        raise ValueError("schema requires explicit history feature names and identity vocabularies")
    vocabularies = {channel: _vocabulary(channel, schema.get(f"{channel}_vocabulary")) for channel in CHANNELS}
    _identity_state(values, identity_state, vocabularies)
    preprocessing = HistoryPreprocessor.fit(values, indices, recipe["batch_size"])
    composition = CompositionPreprocessor.from_history(preprocessing, schema["history_feature_names"])
    # Keep the original constructor's exact residual-table shapes and capacity.
    vocab_sizes = {key: max(2, int(values[key][indices].max()) + 1) for key in ARRAY_CHANNELS.values()}
    for channel in CHANNELS:
        if any(code >= vocab_sizes[ARRAY_CHANNELS[channel]] for code in vocabularies[channel].values()):
            raise ValueError(f"{channel} frozen vocabulary exceeds the original TRAIN embedding shape")
        known = values[ARRAY_CHANNELS[channel]][values["mask"]]
        if np.any((known >= 2) & ~np.isin(known, list(vocabularies[channel].values()))):
            raise ValueError(f"{channel} known codes must belong to the frozen TRAIN vocabulary")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        network = _OovNetwork(dimensions, vocab_sizes, recipe, arm)
    network.eval()
    return OovSeedModel(arm=arm, vocabularies=vocabularies, composition_preprocessing=composition,
                        config=recipe, seed=int(seed), champion_enabled=True, multitask=True,
                        dimensions=dimensions, vocab_sizes=vocab_sizes, preprocessing=preprocessing, network=network)


def fit_oov_model(arrays, series, maps, train_indices, *, seed, arm, schema, identity_state, config,
                  attempt_callback=None):
    """One inherited twelve-epoch trajectory, with no retry or smoke branch.

    The runner journals authorization/attempt *before* calling this function.
    An optional callback can perform that journal operation for direct callers;
    it is invoked before construction/training and never triggers a restart.
    """
    if seed not in MODEL_SEEDS or isinstance(seed, (bool, np.bool_)):
        raise ValueError("formal W1 seed must be one of the three frozen seeds")
    recipe = _recipe(config)
    values, _ = _validate_arrays(arrays)
    indices = _train_indices(train_indices, len(values["best_of"]))
    attempt = {"operation": "W1_NEURAL", "arm": arm, "seed": int(seed),
               "attempted_at": datetime.now(timezone.utc).isoformat(), "epochs": recipe["epochs"],
               "train_series_count": len(indices), "steps_per_epoch": math.ceil(len(indices) / recipe["batch_size"])}
    if attempt_callback is not None:
        attempt_callback(dict(attempt))
    labels = _labels(series, maps, values, indices)
    if any(not np.isfinite(labels[task][indices]).any() for task in ("kills", "duration")):
        raise ValueError("multitask TRAIN requires observed kills and duration labels")
    torch.set_num_threads(recipe["threads"])
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(seed)
    model = build_initial_model(values, indices, seed=seed, arm=arm, schema=schema,
                                identity_state=identity_state, config=recipe)
    counts = _identity_state(values, identity_state, model.vocabularies)
    network = model.network
    kills, duration = labels["kills"][indices], labels["duration"][indices]
    mean_kills = float(np.nanmean(kills)) if np.isfinite(kills).any() else 1.
    mean_log_duration = float(np.nanmean(np.log(duration))) if np.isfinite(duration).any() else 0.
    with torch.no_grad():
        network.prop_head[-1].bias.copy_(torch.tensor([math.log(max(mean_kills, 1e-6)), math.log(.1),
                                                      mean_log_duration, math.log(.25)]))
    model.initial_state = copy.deepcopy(network.state_dict())
    initial_digest = _parameter_digest(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    network.train()
    losses, order_digests, update_trace = [], [], []
    steps = 0
    for epoch in range(recipe["epochs"]):
        order = rng.permutation(indices)
        order_digests.append(hashlib.sha256(order.tobytes()).hexdigest())
        sums = {"score": 0., "kills": 0., "duration": 0.}
        observed_counts = dict.fromkeys(sums, 0)
        for start in range(0, len(order), recipe["batch_size"]):
            ix = order[start:start + recipe["batch_size"]]
            logits, parameters = network(model._batch(values, ix, counts), True, True)
            target = torch.from_numpy(labels["score"][ix])
            valid = torch.isfinite(target)
            terms = {"score": (F.cross_entropy(logits[valid].double(), target[valid].long(), reduction="sum"), int(valid.sum()))}
            terms.update(_prop_losses(parameters, {task: torch.from_numpy(labels[task][ix]) for task in ("kills", "duration")}))
            loss = logits[:, 0].sum() * 0.
            for task, (total, count) in terms.items():
                if count:
                    loss = loss + (1. if task == "score" else recipe[task + "_weight"]) * total / count
                    sums[task] += float(total.detach())
                    observed_counts[task] += count
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite TRAIN likelihood at epoch {epoch + 1}, step {steps + 1}; no retry")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            update_trace.append({"epoch": epoch + 1, "step": steps, "batch_series": len(ix),
                                 "indices_sha256": hashlib.sha256(ix.tobytes()).hexdigest(),
                                 "observed_series": {task: count for task, (_, count) in terms.items()},
                                 "objective": float(loss.detach())})
        losses.append({"epoch": epoch + 1, "losses": {key: sums[key] / observed_counts[key] if observed_counts[key] else None for key in sums},
                       "observed_series": observed_counts, "series_seen": len(order)})
    if any(not torch.isfinite(parameter).all() for parameter in network.parameters()):
        raise RuntimeError("nonfinite final W1 parameters; study closes without retry")
    network.eval()
    model.optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.final_state = copy.deepcopy(network.state_dict())
    model.state_dict = model.final_state
    missing_support = {}
    for channel in CHANNELS:
        missing = values["mask"][indices] & (values[ARRAY_CHANNELS[channel]][indices] == 0)
        missing_support[channel] = {"train_window_tokens": int(missing.sum()),
                                    "train_target_rows": int(missing.reshape(len(indices), -1).any(1).sum()),
                                    "source_event_counts": "reported separately by Stage0 source replay; not inferred from windows",
                                    "legacy_missing_has_training_examples": bool(missing.any()),
                                    "zero_support_state": "UNTRAINED_LEGACY_CODE0" if not missing.any() else None}
    model.fit_receipt = {
        **attempt, "config": dict(recipe), "champion_enabled": True, "multitask": True,
        "software_smoke": False, "epochs_completed": recipe["epochs"], "optimizer_steps": steps,
        "optimizer_parameter_steps": sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state}),
        "train_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "train_score_count": int(np.isfinite(labels["score"][indices]).sum()),
        "train_map_label_counts": {task: int(np.isfinite(labels[task][indices]).sum()) for task in ("kills", "duration")},
        "parameter_count": sum(p.numel() for p in network.parameters()), "added_parameters": 0 if arm == "H" else 592,
        "epoch_losses": losses, "epoch_order_sha256": order_digests, "optimizer_update_trace": update_trace,
        "initial_state_sha256": initial_digest, "final_state_sha256": _parameter_digest(network),
        "torch_version": str(torch.__version__), "device": "cpu", "threads": recipe["threads"],
        "optimizer": "AdamW", "preprocessing_split": "TRAIN", "history_length": 16,
        "vocab_sizes": dict(model.vocab_sizes), "projector_seed": None if arm == "H" else PROJECTOR_SEED,
        "missing_identity_support": missing_support, "prediction_requires_explicit_identity_state": True,
    }
    return model


def assemble_terminal_ensemble(members, *, temperature):
    temperature = _positive_scalar(temperature, "temperature")
    members = tuple(members)
    if len(members) != 3 or any(not isinstance(member, HistoryPrediction) for member in members):
        raise ValueError("terminal ensemble requires exactly three HistoryPrediction members")
    if any(not np.array_equal(member.best_of, members[0].best_of) for member in members):
        raise ValueError("terminal ensemble members must preserve row and best_of alignment")
    components = tuple(_scaled_prediction(member, temperature, 1., 1.) for member in members)
    return HistoryPrediction(np.stack([member.score_probs for member in components]).mean(0),
                             members[0].best_of, components=components)


class OovEnsemble(HistoryEnsemble):
    """One terminal temperature only; inherited prop laws are not calibrated."""
    def __init__(self, models, *, temperature=1.):
        super().__init__(models, score_temperature=temperature, kill_scale=1., duration_scale=1.)
        if any(not isinstance(model, OovSeedModel) for model in self.models):
            raise ValueError("W1 ensemble requires three OovSeedModel members")
        if len({model.arm for model in self.models}) != 1 or len({model.seed for model in self.models}) != 3:
            raise ValueError("W1 ensemble requires one arm and three distinct seeds")
        self.arm = self.models[0].arm

    @property
    def temperature(self):
        return self.score_temperature

    def predict(self, arrays, indices=None, *, identity_state=None):
        members = [model.predict(arrays, indices, identity_state=identity_state) for model in self.models]
        return assemble_terminal_ensemble(members, temperature=self.temperature)


def calibrate_oov_ensemble(models, arrays, series, cal_indices, *, identity_state=None, attempt_callback=None):
    """Fit exactly one terminal scalar for one three-seed arm; no prop fits."""
    ensemble = OovEnsemble(models)
    if any(model.fit_receipt.get("epochs_completed") != DEFAULT_CONFIG["epochs"] for model in ensemble.models):
        raise ValueError("CAL requires three completed frozen twelve-epoch models")
    values, _ = _validate_arrays(arrays, ensemble.models[0].dimensions)
    indices = _indices(cal_indices, len(values["best_of"]))
    if not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("CAL indices must be nonempty and unique")
    if len(series) != len(values["best_of"]) or "score_index" not in series:
        raise ValueError("CAL series labels must align with the complete arrays")
    target = series["score_index"].to_numpy(dtype=np.float64)[indices]
    valid = np.isfinite(target)
    if np.isinf(target).any() or not valid.any() or np.any((target[valid] != np.floor(target[valid])) | (target[valid] < 0) | (target[valid] > 5)):
        raise ValueError("CAL requires observed legal terminal score indices or NaN")
    rows = np.flatnonzero(valid)
    scores = target[valid].astype(np.int64)
    if np.any(~_legal(values["best_of"][indices])[rows, scores]):
        raise ValueError("CAL score label outside legal best_of support")
    predictions = [model.predict(values, indices, identity_state=identity_state) for model in ensemble.models]
    attempt = {"operation": "W1_TERMINAL_CAL", "arm": ensemble.arm,
               "attempted_at": datetime.now(timezone.utc).isoformat(), "scalar_count": 1}
    if attempt_callback is not None:
        attempt_callback(dict(attempt))

    objective_evaluations = []
    def score_loss(temperature):
        probabilities = np.stack([_temper(pred.score_probs, temperature) for pred in predictions]).mean(0)
        with np.errstate(divide="ignore"):
            nll = float(-np.log(probabilities[rows, scores]).mean())
        objective_evaluations.append({"temperature": float(temperature), "nll": nll,
                                      "penalized_objective": nll + CALIBRATION_REGULARIZATION * math.log(temperature) ** 2})
        return nll

    temperature, detail = _calibrate_scalar(score_loss, "W1 terminal temperature")
    log_temperature = math.log(temperature)
    true_probabilities, true_derivatives = [], []
    for prediction in predictions:
        tempered = _temper(prediction.score_probs, temperature)
        logp = np.zeros_like(prediction.score_probs)
        np.log(prediction.score_probs, out=logp, where=prediction.score_probs > 0)
        expected_logp = (tempered * logp).sum(1)
        true = tempered[rows, scores]
        true_probabilities.append(true)
        true_derivatives.append(true * (expected_logp[rows] - logp[rows, scores]) / temperature)
    gradient = float(-(np.mean(true_derivatives, axis=0) / np.mean(true_probabilities, axis=0)).mean()
                     + 2. * CALIBRATION_REGULARIZATION * log_temperature)
    lower, upper = CALIBRATION_BOUNDS
    projected = min(gradient, 0.) if temperature == lower else max(gradient, 0.) if temperature == upper else gradient
    if not np.isfinite(gradient) or abs(projected) > 1e-5:
        raise RuntimeError("W1 terminal CAL failed fixed projected-gradient acceptance; no retry")
    candidates = []
    for value in (temperature, lower, upper, 1.):
        nll = score_loss(value)
        candidates.append({"temperature": value, "nll": nll,
                           "penalized_objective": nll + CALIBRATION_REGULARIZATION * math.log(value) ** 2})
    detail.update({"temperature": temperature, "log_temperature": log_temperature,
                   "penalty": CALIBRATION_REGULARIZATION * log_temperature ** 2,
                   "objective": detail["penalized_objective"], "raw_gradient_log_temperature": gradient,
                   "projected_gradient": projected, "projected_gradient_infinity": abs(projected),
                   "projected_gradient_limit": 1e-5, "candidate_evaluations": candidates,
                   "objective_evaluations": objective_evaluations,
                   "method": "bounded", "xatol": 1e-7, "maxiter": 300,
                   "converged": True, "convergence_success": True})
    ensemble.score_temperature = temperature
    receipt = {**attempt, "split": "CAL", "series_count": len(indices),
               "cal_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
               "label_row_indices": indices[valid].tolist(), "label_score_indices": scores.tolist(),
               "score_observed_series": int(valid.sum()), "score_temperature": detail,
               "temperature": temperature, "mixture_components": 3, "frozen_networks": True,
               "prop_scale_fits": 0, "kill_scale": 1., "duration_scale": 1.,
               "objective": "mean terminal-score mixture NLL + .001*log(T)^2",
               "transform": "member full terminal PMF temperature then equal mean; no best-of reconversion"}
    ensemble.calibration_receipt = receipt
    return ensemble, receipt
