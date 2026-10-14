"""Device-aware execution of the frozen A0 optimizer loops.

Architectures, preprocessing, losses, iteration order and optimization settings
come from the genuine research backend. Only minibatches and parameters move
to the requested device. Models and caches return on CPU, so the native
inference methods and portable joblib checkpoints do not change.

Load the verified research backend before calling these functions. The CPU
execution path is also used to check equivalence to the archived fit methods.
"""

from __future__ import annotations

import numpy as np
import torch


def fit_beta_on_device(
    model, raw, data, ids, ix, bo, a, b, targets, mask, progress=None, *, device="cuda"
):
    from src.models.auxiliary_player import TargetScaler, masked_auxiliary_loss
    from src.models.beta_training_torch import beta_terminal_nll

    device = torch.device(device)
    model.target_scaler = TargetScaler().fit(targets, mask, ix)
    model.network.to(device)
    model.eta = torch.nn.Parameter(model.eta.detach().to(device))
    rng = np.random.default_rng(model.seed)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.network.parameters()), "weight_decay": 0.01},
            {"params": [model.eta], "weight_decay": 0.0},
        ],
        lr=0.001,
    )
    parameters = [*model.network.parameters(), model.eta]
    model.train_losses = []
    model.auxiliary_losses = []
    model.epoch_diagnostics = []
    model.network.train()
    for epoch in range(model.epochs):
        total = 0.0
        auxiliary_total = 0.0
        maximum_gradient = 0.0
        clipped_batches = 0
        for begin in range(0, len(ix), 128):
            selected = ix[begin:begin + 128]
            inputs, offset = model.batch(raw, data, ids, selected, bo, rng=rng)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            offset = offset.to(device)
            residual, prediction = model.network(inputs, return_aux=True)
            logits = (offset + residual).double()
            main = beta_terminal_nll(
                logits,
                torch.as_tensor(2 * bo[selected] + 1, device=device),
                torch.as_tensor(a[selected], device=device),
                torch.as_tensor(b[selected], device=device),
                model.eta,
            ).mean()
            target = torch.from_numpy(model.target_scaler.transform(targets[selected], mask[selected])).to(device)
            auxiliary = masked_auxiliary_loss(
                prediction, target,
                torch.from_numpy(np.asarray(mask[selected], bool)).to(device),
            )
            loss = main + model.coefficient * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("nonfinite beta training loss")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
            norm_value = float(norm)
            maximum_gradient = max(maximum_gradient, norm_value)
            clipped_batches += int(norm_value > 5)
            optimizer.step()
            if not torch.isfinite(model.eta):
                raise ValueError("nonfinite training dependence")
            total += main.item() * len(selected)
            auxiliary_total += auxiliary.item() * len(selected)
        if not all(torch.isfinite(parameter).all() for parameter in parameters):
            raise ValueError("nonfinite trained weights")
        model.train_losses.append(total / len(ix))
        model.auxiliary_losses.append(auxiliary_total / len(ix))
        eta = float(model.eta.detach())
        rho = float(0.5 * torch.sigmoid(model.eta.detach()))
        record = {
            "epoch": epoch + 1, "eta": eta, "rho": rho,
            "max_gradient_norm_before_clip": maximum_gradient,
            "clipped_batches": clipped_batches, "finite_gradients": True,
            "near_rho_boundary": bool(min(rho, 0.5 - rho) < 1e-4),
        }
        model.epoch_diagnostics.append(record)
        if progress:
            progress(epoch + 1, model.train_losses[-1], model.auxiliary_losses[-1], record)
    model.dropout_rng_final_state = rng.bit_generator.state
    model.network.eval().cpu()
    model.eta = torch.nn.Parameter(model.eta.detach().cpu())
    model.training_device = str(device)
    return model


def fit_unified_on_device(
    model, raw, data, ix, y, bo, initial=None, progress=None, *, device="cuda"
):
    if model.kind != "mlp" and not (model.kind == "raw" and not model.full):
        raise ValueError("A0 acceleration supports MLP experts and the CPU raw-rating baseline only")
    from src.models.roster_optional import Preprocessor, subset
    from src.models.temporal_training import HistoryNormalizer

    # The raw-rating expert is an sklearn fit, not a neural optimizer.
    if model.kind == "raw":
        result = model.fit(raw, data, ix, y, bo, initial=initial, progress=progress)
        model.training_device = "cpu"
        return result
    device = torch.device(device)
    model.normalizer = Preprocessor().fit(subset(raw, ix))
    if model.full:
        model.history_normalizer = HistoryNormalizer().fit(data, ix)
    if initial is not None:
        if initial.full or initial.kind != model.kind:
            raise ValueError("incompatible pretrained compressor")
        model.network.player_rating.load_state_dict(initial.network.player_rating.state_dict())
        model.network.team_rating.load_state_dict(initial.network.team_rating.state_dict())
        with torch.no_grad():
            model.network.log_gain.copy_(initial.network.log_gain)
            gain = model.network.log_gain.clamp(-2, 2).exp()
            model.network.linear_head.weight[:, 1] = gain
            model.network.linear_head.weight[:, 48] = gain
    cache = model.prepare(raw, data)
    model.network.to(device)
    rng = np.random.default_rng(model.seed)
    optimizer = torch.optim.AdamW(
        model.network.parameters(), lr=0.001 if model.full else 0.003,
        weight_decay=0.01,
    )
    batch_size = 128 if model.full else 256
    model.network.train()
    for epoch in range(model.epochs):
        total = 0.0
        for start in range(0, len(ix), batch_size):
            selected = ix[start:start + batch_size]
            inputs = model._batch(cache, selected, rng=rng)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model.network(inputs, torch.as_tensor(bo[selected], dtype=torch.long, device=device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(y[selected], dtype=torch.float32, device=device)
            )
            _, correction = model.network.ratings(inputs)
            loss = loss + 0.05 * correction.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.network.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        model.train_losses.append(total / len(ix))
        if progress is not None and (epoch + 1) % 4 == 0:
            progress(epoch + 1, model.train_losses[-1])
    model.network.eval().cpu()
    model.training_device = str(device)
    return cache
