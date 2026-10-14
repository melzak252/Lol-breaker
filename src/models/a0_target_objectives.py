"""Differentiable stopped-series Beta objectives for the native A0 head."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


_OBJECTIVES = ("terminal", "winner")
_BEST_OF = (1, 3, 5)


def _inputs(logits, best_of, wins_a, wins_b, eta):
    if not torch.is_tensor(logits) or logits.ndim != 1 or not logits.numel():
        raise ValueError("finite aligned nonempty map logits are required")
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise ValueError("finite floating map logits are required")
    device = logits.device
    z = logits.to(dtype=torch.float64)
    bo = torch.as_tensor(best_of, device=device)
    a = torch.as_tensor(wins_a, device=device)
    b = torch.as_tensor(wins_b, device=device)
    if bo.shape != z.shape or a.shape != z.shape or b.shape != z.shape:
        raise ValueError("best-of and stopped scores must align with logits")
    for name, values in (("best_of", bo), ("wins_a", a), ("wins_b", b)):
        if values.is_floating_point():
            if not torch.isfinite(values).all().item() or not torch.equal(values, values.round()):
                raise ValueError(f"{name} must contain finite integers")
        elif values.dtype == torch.bool or values.is_complex():
            raise ValueError(f"{name} must contain integers")
    bo, a, b = bo.to(torch.int64), a.to(torch.int64), b.to(torch.int64)
    if not torch.isin(bo, torch.tensor(_BEST_OF, device=device)).all().item():
        raise ValueError("best_of must contain only 1, 3, or 5")
    if torch.any(a < 0).item() or torch.any(b < 0).item():
        raise ValueError("stopped scores cannot be negative")
    target = (bo + 1) // 2
    if not torch.equal(torch.maximum(a, b), target) or torch.any(torch.minimum(a, b) >= target).item():
        raise ValueError("scores must be legal terminal stopped outcomes")
    rho_parameter = torch.as_tensor(eta, dtype=torch.float64, device=device)
    if rho_parameter.ndim != 0 or not torch.isfinite(rho_parameter).item():
        raise ValueError("eta must be one finite scalar")
    return z, bo, a, b, target, rho_parameter


def _log_stopped_score(logits, wins_a, wins_b, log_rho, rho, log_one_minus_rho):
    """Log probability of one legal stopped score, including path multiplicity."""
    log_p_a = F.logsigmoid(logits)
    log_p_b = F.logsigmoid(-logits)
    result = torch.zeros_like(logits)
    for count, log_probability in ((wins_a, log_p_a), (wins_b, log_p_b)):
        for prior_wins in range(3):
            active = count > prior_wins
            if prior_wins == 0:
                term = log_probability + log_one_minus_rho
            else:
                term = torch.logaddexp(
                    log_probability + log_one_minus_rho,
                    log_rho + math.log(prior_wins),
                )
            result = result + torch.where(active, term, torch.zeros_like(term))
    games = wins_a + wins_b
    for prior_games in range(5):
        active = games > prior_games
        factor = 1.0 + (prior_games - 1) * rho
        result = result - torch.where(active, torch.log(factor), torch.zeros_like(result))
    path_count_log = (
        torch.lgamma(games.to(torch.float64))
        - torch.lgamma(torch.maximum(wins_a, wins_b).to(torch.float64))
        - torch.lgamma(torch.minimum(wins_a, wins_b).to(torch.float64) + 1.0)
    )
    return result + path_count_log


def beta_objective(logits, best_of, wins_a, wins_b, eta, objective="terminal"):
    """Return per-row terminal-score or marginal winner negative log likelihood.

    The shared latent map-win probability is ``sigmoid(logits)`` and the
    Beta-Bernoulli dependence parameter is ``rho = 0.5 * sigmoid(eta)``.
    ``winner`` sums the probabilities of every legal terminal score for the
    observed series winner; it does not project a map probability to a series
    probability or use a second Beta head.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}")
    z, bo, a, b, target, eta_value = _inputs(logits, best_of, wins_a, wins_b, eta)
    rho = 0.5 * torch.sigmoid(eta_value)
    log_rho = math.log(0.5) + F.logsigmoid(eta_value)
    log_one_minus_rho = torch.log1p(-rho)
    if objective == "terminal":
        log_mass = _log_stopped_score(z, a, b, log_rho, rho, log_one_minus_rho)
        return -log_mass

    winner_a = a == target
    log_winner_mass = torch.empty_like(z)
    for rounds in (1, 2, 3):
        selected = target == rounds
        if not selected.any().item():
            continue
        row_logits = z[selected]
        row_logits = torch.where(winner_a[selected], row_logits, -row_logits)
        target_score = torch.full_like(row_logits, rounds, dtype=torch.int64)
        legal_logs = []
        for loser_wins in range(rounds):
            loser_score = torch.full_like(target_score, loser_wins)
            legal_logs.append(_log_stopped_score(
                row_logits, target_score, loser_score, log_rho, rho, log_one_minus_rho,
            ))
        log_winner_mass[selected] = torch.logsumexp(torch.stack(legal_logs, dim=1), dim=1)
    return -log_winner_mass



def weighted_native_losses(main_rows, prediction, target, mask, row_weights):
    """Apply row weights to native main and hierarchical auxiliary losses.

    Weights are used as supplied: callers normalize them over the complete
    training set, while this helper intentionally does not renormalize batches.
    """
    valid = mask.bool()
    if (
        prediction.shape != target.shape
        or target.shape != valid.shape
        or prediction.ndim != 4
        or main_rows.ndim != 1
        or main_rows.shape[0] != prediction.shape[0]
    ):
        raise ValueError("native loss batch shapes must align")
    if not torch.isfinite(target[valid]).all().item():
        raise ValueError("nonfinite observed auxiliary targets")
    weights = torch.as_tensor(
        row_weights, dtype=main_rows.dtype, device=main_rows.device,
    )
    if weights.ndim != 1 or weights.shape != main_rows.shape:
        raise ValueError("row weights must align with native loss rows")
    if not torch.isfinite(weights).all().item() or torch.any(weights <= 0).item():
        raise ValueError("row weights must be finite and strictly positive")

    safe_target = torch.where(valid, target, torch.zeros_like(target))
    error = F.smooth_l1_loss(prediction, safe_target, reduction="none", beta=1)
    per_player = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    present = valid.any(-1)
    per_series = (per_player * present).sum((1, 2)) / present.sum((1, 2)).clamp_min(1)
    return (main_rows * weights).mean(), (per_series * weights).mean()


def fit_beta_target_on_device(
    model, raw, data, ids, ix, bo, a, b, targets, mask, progress=None, *,
    device="cuda", objective="terminal", row_weights=None,
):
    """Fit the native beta head with the experiment-selected main objective.

    This intentionally mirrors the established accelerator training loop while
    keeping this experiment's loss selection out of the frozen accelerator
    source and its provenance hash.
    """
    import numpy as np

    from src.models.auxiliary_player import TargetScaler, masked_auxiliary_loss
    from src.models.beta_training_torch import beta_terminal_nll

    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}")
    device = torch.device(device)
    train_row_weights = None
    if row_weights is not None:
        train_row_weights = torch.as_tensor(
            row_weights, dtype=torch.float64, device=device,
        )
        if train_row_weights.ndim != 1 or train_row_weights.numel() != len(ix):
            raise ValueError("row weights must align with the complete training indices")
        if (
            not torch.isfinite(train_row_weights).all().item()
            or torch.any(train_row_weights <= 0).item()
        ):
            raise ValueError("row weights must be finite and strictly positive")
        if not torch.isclose(
            train_row_weights.mean(),
            torch.ones((), dtype=train_row_weights.dtype, device=device),
            rtol=1e-5, atol=1e-6,
        ).item():
            raise ValueError("full-training row weights must have mean 1")
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
            best_of = torch.as_tensor(2 * bo[selected] + 1, device=device)
            wins_a = torch.as_tensor(a[selected], device=device)
            wins_b = torch.as_tensor(b[selected], device=device)
            if objective == "winner":
                main_rows = beta_objective(
                    logits, best_of, wins_a, wins_b, model.eta, objective="winner",
                )
            else:
                main_rows = beta_terminal_nll(
                    logits, best_of, wins_a, wins_b, model.eta,
                )
            target = torch.from_numpy(
                model.target_scaler.transform(targets[selected], mask[selected])
            ).to(device)
            batch_mask = torch.from_numpy(np.asarray(mask[selected], bool)).to(device)
            if train_row_weights is None:
                main = main_rows.mean()
                auxiliary = masked_auxiliary_loss(prediction, target, batch_mask)
            else:
                batch_weights = train_row_weights[begin:begin + len(selected)]
                main, auxiliary = weighted_native_losses(
                    main_rows, prediction, target, batch_mask, batch_weights,
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
