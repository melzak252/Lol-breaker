"""TRAIN-only pretext learning for the unchanged native A0 history encoder."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


STAT_NAMES = ("csd@15", "xpd@15")
STAT_SLICE = slice(8, 10)
PRESENCE_SLICE = slice(20, 22)
RNG_NAMESPACE = 20261001


def mask_history_fields(inputs, selected_history):
    """Return corrupt inputs, safe targets, native presence, and artificial mask.

    One selection hides both fields across a player's valid history. A measured
    zero is still observed. Native absence and padding never become targets.
    """
    history, valid = inputs["history"], inputs["mask"]
    if (history.ndim != 5 or history.shape[-1] != 33
            or history.shape[:-1] != valid.shape
            or selected_history.shape != valid.shape[:-1]):
        raise ValueError("history/player corruption mask shape mismatch")
    if valid.dtype != torch.bool or selected_history.dtype != torch.bool:
        raise ValueError("native and artificial masks must be boolean")
    flags = history[..., PRESENCE_SLICE]
    observed = valid[..., None] & (flags > 0)
    if not torch.isfinite(flags[valid]).all() or not torch.isfinite(history[..., STAT_SLICE][observed]).all():
        raise ValueError("nonfinite observed history statistics or presence")
    targets = torch.where(observed, history[..., STAT_SLICE], 0.)
    artificial = observed & selected_history[..., None, None]
    visible = observed & ~artificial
    corrupt = torch.where(valid[..., None], history, 0.)
    corrupt[..., STAT_SLICE] = torch.where(visible, targets, 0.)
    corrupt[..., PRESENCE_SLICE] = visible.to(history.dtype)
    return {**inputs, "history": corrupt}, targets, observed, artificial


def reconstruction_loss(prediction, targets, artificial):
    """Cell-weighted SmoothL1 only on artificially hidden observed values."""
    if prediction.shape != targets.shape or targets.shape != artificial.shape:
        raise ValueError("reconstruction shapes must align")
    if artificial.dtype != torch.bool or not artificial.any():
        raise ValueError("nonempty hidden observed targets are required")
    if not torch.isfinite(targets[artificial]).all() or not torch.isfinite(prediction[artificial]).all():
        raise ValueError("nonfinite hidden observed reconstruction")
    return F.smooth_l1_loss(prediction[artificial], targets[artificial], beta=1.)


def reconstruct_history(network, inputs, decoder):
    """Decode tokens and the native pooled player representation, sharing weights."""
    mask = inputs["mask"].reshape(-1, 16)
    values = inputs["history"].reshape(-1, 16, 33)
    encoded = network.history_encoder(torch.where(mask[..., None], values, 0.))
    all_encoded = torch.cat([network.empty.expand(len(encoded), -1, -1), encoded], dim=1)
    valid = torch.cat([torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], dim=1)
    weights = network.attention(all_encoded).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
    pooled = (all_encoded * weights[..., None]).sum(1)
    player = torch.tanh(network.player(inputs["players"]).reshape(-1, 32) + pooled)
    features = torch.cat([encoded, player[:, None, :].expand(-1, 16, -1)], dim=-1)
    return decoder(features).reshape(*inputs["mask"].shape, 2)


def pretrain_history(model, raw, data, indices, bo, *, epoch_equivalents=6,
                     probability=.25, device="cpu", progress=None):
    """Spend exactly E*ceil(full TRAIN/128) nonempty reconstruction updates.

    Naturally unobserved rows cannot supply a reconstruction target. Cycle the
    eligible TRAIN rows chronologically to meet the fixed update budget; never
    turn an empty loss/weight-decay step into a claimed encoder update. No label
    array is accepted. The native input scaler was fitted on the same TRAIN.
    """
    if epoch_equivalents < 1 or int(epoch_equivalents) != epoch_equivalents or not 0 < probability <= 1:
        raise ValueError("positive integer epochs and corruption probability in (0,1] required")
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("unique nonempty TRAIN indices required")
    started = time.monotonic()
    eligible_parts = []
    for begin in range(0, len(indices), 256):
        ix = indices[begin:begin + 256]
        values = np.asarray(data["history"][ix])
        present = np.asarray(data["mask"][ix], dtype=bool)[..., None] & (values[..., PRESENCE_SLICE] > 0)
        eligible_parts.append(ix[present.any(axis=(1, 2, 3, 4))])
    eligible = np.concatenate(eligible_parts)
    if not len(eligible):
        raise ValueError("TRAIN has no observed reconstruction statistics")
    batches_per_equivalent = math.ceil(len(indices) / 128)
    steps = int(epoch_equivalents) * batches_per_equivalent
    rng = np.random.default_rng(np.random.SeedSequence([model.seed, RNG_NAMESPACE]))
    # Decoder initialization neither changes paired native initialization nor
    # consumes the native gate-dropout random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model.seed + RNG_NAMESPACE)
        decoder = nn.Linear(64, 2).to(device)
    network = model.network.to(device)
    network.train()
    parameters = [*network.history_encoder.parameters(), *network.attention.parameters(),
                  network.empty, *network.player.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.01)
    records = []
    cursor = 0
    totals = {"observed_cells": 0, "hidden_cells": 0, "eligible_histories": 0,
              "selected_histories": 0, "corruption_draws": 0, "training_row_visits": 0}
    loss_sum, cell_count, maximum_gradient = 0., 0, 0.
    for step in range(steps):
        ix = eligible[cursor:cursor + 128]
        cursor += len(ix)
        if cursor == len(eligible):
            cursor = 0
        inputs, _ = model.batch(raw, data, None, ix, bo, rng=None)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        native = inputs["mask"][..., None] & (inputs["history"][..., PRESENCE_SLICE] > 0)
        # Paired sides share the role-position draw: swapping A/B with the same
        # RNG state commutes exactly, even when their native missingness differs.
        while True:
            selected = torch.as_tensor(rng.random((len(ix), 1, 5)) < probability, device=device).expand(-1, 2, -1)
            totals["corruption_draws"] += 1
            if (native & selected[..., None, None]).any():
                break
        masked, targets, native, artificial = mask_history_fields(inputs, selected)
        prediction = reconstruct_history(network, masked, decoder)
        loss = reconstruction_loss(prediction, targets, artificial)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        # A zero-target batch must not silently inflate matched update counts.
        if any(parameter.grad is None for parameter in parameters):
            raise ValueError("reconstruction failed to update the entire shared encoder")
        optimizer.step()
        hidden = int(artificial.sum())
        loss_sum += float(loss.detach()) * hidden
        cell_count += hidden
        maximum_gradient = max(maximum_gradient, float(norm))
        eligible_histories = native.any(-1).any(-1)
        totals["observed_cells"] += int(native.sum())
        totals["hidden_cells"] += hidden
        totals["eligible_histories"] += int(eligible_histories.sum())
        totals["selected_histories"] += int((eligible_histories & selected).sum())
        totals["training_row_visits"] += len(ix)
        if (step + 1) % batches_per_equivalent == 0:
            record = {"epoch_equivalent": (step + 1) // batches_per_equivalent,
                      "optimizer_updates": step + 1, "loss": loss_sum / cell_count,
                      "hidden_cells": cell_count, "max_gradient_norm_before_clip": maximum_gradient}
            records.append(record)
            if progress:
                progress(record)
            loss_sum, cell_count, maximum_gradient = 0., 0, 0.
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise ValueError("nonfinite pretrained parameters")
    network.eval().cpu()
    return {"objective": "hidden_observed_cell_mean_smooth_l1_beta_1",
            "decoder": "Linear(64,2): token32 + native pooled player32; discarded after pretraining",
            "stats": list(STAT_NAMES), "scaling": "shared full-TRAIN native HistoryNormalizer, clipped +/-10",
            "eligible_train_rows": len(eligible), "train_rows": len(indices),
            "epoch_equivalents": int(epoch_equivalents), "optimizer_updates": steps,
            "encoder_parameters": sum(p.numel() for p in parameters[:-2]),
            "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
            "probability": probability, "rng_namespace": RNG_NAMESPACE,
            "rng_final_state": rng.bit_generator.state, "epochs": records,
            "counts": totals, "runtime_seconds": time.monotonic() - started}
