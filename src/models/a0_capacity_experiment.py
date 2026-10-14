"""Width-only native attention and uninterrupted terminal-Beta trajectories.

Archive activation remains explicit at construction: importing this module does
not replace the repository's model namespace. Width 32 retains native RNG draws,
state keys and arithmetic; final assembly belongs to the runner.
"""
from __future__ import annotations

import copy
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.a0_target_objectives import beta_objective


@lru_cache(maxsize=1)
def _backend():
    names = ('nonlinear_history', 'coherent_series', 'temporal_training',
             'auxiliary_player', 'beta_training_torch')
    if all(f'src.models.{name}' in sys.modules for name in names):
        return {name: sys.modules[f'src.models.{name}'] for name in names}
    from src.models.a0_uncertainty import _activate_archive_backend, _resolve_research_root
    return _activate_archive_backend(_resolve_research_root(None))[0]


class ParametricTemporalResidual(nn.Module):
    """Native attention representation with only its hidden width changed."""

    def __init__(self, width=32):
        super().__init__()
        if width not in (16, 32, 64):
            raise ValueError('width must be 16, 32, or 64')
        self.width = width
        self.kind = 'attention'
        self.has_history = True
        self.has_champions = self.has_graph = False
        self.player = nn.Sequential(nn.Linear(21, width), nn.Tanh())
        self.history_encoder = nn.Linear(33, width)
        self.empty = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.Linear(width, 1)
        # Preserve the native three-output initialization draws. CommonMap below
        # keeps its first row; drawing a fresh single-output normal breaks C0.
        self.head = nn.Sequential(nn.Linear(70 + 2 * width, width), nn.Tanh(),
                                  nn.Linear(width, 3, bias=False))
        nn.init.normal_(self.head[-1].weight, std=.01)

    def _representation(self, x):
        h = self.player(x['players'])
        n = h.shape[0]
        mask = x['mask'].reshape(-1, 16).bool()
        values = x['history'].reshape(-1, 16, 33)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        encoded = self.history_encoder(values)
        encoded = torch.cat([self.empty.expand(len(encoded), -1, -1), encoded], 1)
        valid = torch.cat([torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], 1)
        weight = self.attention(encoded).squeeze(-1).masked_fill(~valid, -torch.inf).softmax(1)
        out = (encoded * weight[..., None]).sum(1)
        h = h + out.reshape(n, 2, 5, self.width)
        return torch.tanh(h)

    def forward(self, x, best_of_index):
        h = self._representation(x)
        team = torch.cat([x['context'], h.mean(2), h.std(2, unbiased=False)], -1)
        utility = self.head(team)
        rows = torch.arange(len(h), device=h.device)
        return utility[rows, 0, best_of_index] - utility[rows, 1, best_of_index]


class ParametricAuxiliaryResidual(ParametricTemporalResidual):
    def __init__(self, width=32):
        super().__init__(width)
        self.auxiliary_head = nn.Linear(width, 12)

    def forward(self, x, best_of_index, return_aux=False):
        h = self._representation(x)
        team = torch.cat([x['context'], h.mean(2), h.std(2, unbiased=False)], -1)
        utility = self.head(team)
        rows = torch.arange(len(h), device=h.device)
        z = utility[rows, 0, best_of_index] - utility[rows, 1, best_of_index]
        return (z, self.auxiliary_head(h)) if return_aux else z


class ParametricCommonMapResidual(ParametricAuxiliaryResidual):
    def __init__(self, width=32):
        super().__init__(width)
        weight = self.head[-1].weight.detach()[:1].clone()
        self.head[-1] = nn.Linear(width, 1, bias=False)
        with torch.no_grad():
            self.head[-1].weight.copy_(weight)
        self.history_encoder = _backend()['nonlinear_history'].TanhHistoryEncoder(self.history_encoder)
        if sum(p.numel() for p in self.parameters()) != 2 * width**2 + 142 * width + 13:
            raise AssertionError('native parameter count differs')

    def forward(self, x, best_of_index=None, return_aux=False):
        index = torch.zeros(len(x['players']), dtype=torch.long, device=x['players'].device)
        return super().forward(x, index, return_aux=return_aux)


def soft_terminal_loss(logits, best_of, eta, teacher_probabilities):
    """Per-row cross entropy over legal stopped scores, with a detached teacher."""
    bo = torch.as_tensor(best_of, device=logits.device, dtype=torch.int64)
    teacher = torch.as_tensor(teacher_probabilities, dtype=torch.float64, device=logits.device).detach()
    if teacher.shape != (len(logits), 6) or bo.shape != logits.shape:
        raise ValueError('teacher terminal mass must align with map logits')
    if not torch.isfinite(teacher).all() or torch.any(teacher < 0):
        raise ValueError('teacher mass must be finite and nonnegative')
    if not torch.isin(bo, torch.tensor([1, 3, 5], device=bo.device)).all():
        raise ValueError('teacher best_of must be 1, 3, or 5')
    target = (bo + 1) // 2
    legal = torch.arange(3, device=bo.device)[None, :] < target[:, None]
    legal = torch.cat([legal, legal], 1)
    if torch.any(teacher[~legal] != 0) or not torch.allclose(teacher.sum(1), torch.ones(len(logits), dtype=teacher.dtype, device=bo.device), atol=1e-10, rtol=0):
        raise ValueError('teacher must have normalized legal terminal mass')
    losses = torch.zeros_like(logits, dtype=torch.float64)
    for rounds in (1, 2, 3):
        selected = target == rounds
        if not selected.any():
            continue
        row_z = logits[selected]
        row_bo = bo[selected]
        for side in (0, 1):
            for loser in range(rounds):
                a, b = (rounds, loser) if side == 0 else (loser, rounds)
                hard = beta_objective(row_z, row_bo, torch.full_like(row_bo, a),
                                      torch.full_like(row_bo, b), eta)
                losses[selected] = losses[selected] + teacher[selected, side * 3 + loser] * hard
    return losses


class ParametricNonlinearHistoryModel:
    output_kind = 'uncalibrated_map_logit'

    def __init__(self, baseline, normalizer, width=32, seed=20260910, epochs=24, coefficient=.1):
        if epochs < 1 or not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError('positive epochs and nonnegative auxiliary coefficient required')
        torch.manual_seed(seed)
        self.network = ParametricCommonMapResidual(width)
        self.baseline, self.normalizer = baseline, normalizer
        self.width, self.seed, self.epochs = width, seed, epochs
        self.coefficient, self.kind = float(coefficient), 'attention'
        self.eta = nn.Parameter(torch.tensor(math.log(1 / 9), dtype=torch.double))
        self.main_objective = 'terminal'

    def batch(self, raw, data, ids, ix, bo, w20=True, org=True, rng=None):
        return _backend()['coherent_series'].CoherentModel.batch(self, raw, data, ids, ix, bo, w20, org, rng)

    def map_logits(self, raw, data, ids, ix, bo, w20=True, org=True):
        return _backend()['temporal_training'].TemporalModel.logits(self, raw, data, ids, ix, bo, w20, org)

    def fit_trajectory(self, raw, data, ids, ix, bo, a, b, targets, mask, *,
                       checkpoint_dir, checkpoint_epochs=(12, 24), device='cpu',
                       teacher_probabilities=None, distillation_weight=0., progress=None):
        backend = _backend()
        TargetScaler = backend['auxiliary_player'].TargetScaler
        auxiliary_loss = backend['auxiliary_player'].masked_auxiliary_loss
        terminal_nll = backend['beta_training_torch'].beta_terminal_nll
        ix = np.asarray(ix, dtype=np.int64)
        if len(ix) == 0 or len(np.unique(ix)) != len(ix):
            raise ValueError('nonempty unique TRAIN indices required')
        checkpoints = tuple(checkpoint_epochs)
        if not checkpoints or any(epoch < 1 or epoch > self.epochs for epoch in checkpoints):
            raise ValueError('checkpoint epochs must belong to the trajectory')
        if not 0 <= distillation_weight <= 1:
            raise ValueError('distillation weight must be in [0,1]')
        if distillation_weight and (teacher_probabilities is None or np.shape(teacher_probabilities) != (len(ix), 6)):
            raise ValueError('TRAIN-only aligned teacher probabilities required')
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch in checkpoints:
            if (checkpoint_dir / f'checkpoint_epoch{epoch}.pt').exists():
                raise FileExistsError(checkpoint_dir / f'checkpoint_epoch{epoch}.pt')
        device = torch.device(device)
        self.target_scaler = TargetScaler().fit(targets, mask, ix)
        self.network.to(device)
        self.eta = nn.Parameter(self.eta.detach().to(device))
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.AdamW([
            {'params': list(self.network.parameters()), 'weight_decay': .01},
            {'params': [self.eta], 'weight_decay': 0.},
        ], lr=.001)
        parameters = [*self.network.parameters(), self.eta]
        self.train_losses, self.auxiliary_losses, self.epoch_diagnostics = [], [], []
        self.network.train()
        snapshots = {}
        for epoch in range(1, self.epochs + 1):
            total = auxiliary_total = maximum_gradient = 0.
            clipped_batches = 0
            for begin in range(0, len(ix), 128):
                selected = ix[begin:begin + 128]
                inputs, offset = self.batch(raw, data, ids, selected, bo, rng=rng)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                residual, prediction = self.network(inputs, return_aux=True)
                logits = (offset.to(device) + residual).double()
                best_of = torch.as_tensor(2 * bo[selected] + 1, device=device)
                main_rows = terminal_nll(logits, best_of, torch.as_tensor(a[selected], device=device),
                                         torch.as_tensor(b[selected], device=device), self.eta)
                if distillation_weight:
                    soft = soft_terminal_loss(logits, best_of, self.eta,
                                              teacher_probabilities[begin:begin + len(selected)])
                    main_rows = (1 - distillation_weight) * main_rows + distillation_weight * soft
                main = main_rows.mean()
                target = torch.from_numpy(self.target_scaler.transform(targets[selected], mask[selected])).to(device)
                batch_mask = torch.from_numpy(np.asarray(mask[selected], bool)).to(device)
                auxiliary = auxiliary_loss(prediction, target, batch_mask)
                loss = main + self.coefficient * auxiliary
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite beta training loss')
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                norm_value = float(norm)
                maximum_gradient = max(maximum_gradient, norm_value)
                clipped_batches += int(norm_value > 5)
                optimizer.step()
                if not torch.isfinite(self.eta):
                    raise ValueError('nonfinite training dependence')
                total += main.item() * len(selected)
                auxiliary_total += auxiliary.item() * len(selected)
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError('nonfinite trained weights')
            self.train_losses.append(total / len(ix))
            self.auxiliary_losses.append(auxiliary_total / len(ix))
            rho = float(.5 * torch.sigmoid(self.eta.detach()))
            record = {'epoch': epoch, 'eta': float(self.eta.detach()), 'rho': rho,
                      'train_loss': self.train_losses[-1], 'auxiliary_loss': self.auxiliary_losses[-1],
                      'max_gradient_norm_before_clip': maximum_gradient,
                      'clipped_batches': clipped_batches, 'finite_gradients': True,
                      'near_rho_boundary': min(rho, .5 - rho) < 1e-4}
            self.epoch_diagnostics.append(record)
            self.dropout_rng_final_state = copy.deepcopy(rng.bit_generator.state)
            if epoch in checkpoints:
                snapshot = copy.deepcopy(self)
                snapshot.network.eval().cpu()
                snapshot.eta = nn.Parameter(snapshot.eta.detach().cpu())
                snapshot.epochs = epoch
                snapshots[epoch] = snapshot
                state = {'epoch': epoch, 'width': self.width, 'seed': self.seed,
                         'network_state_dict': {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()},
                         'eta': self.eta.detach().cpu().clone(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'numpy_rng_state': copy.deepcopy(rng.bit_generator.state),
                         'torch_rng_state': torch.get_rng_state(),
                         'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                         'target_scaler': self.target_scaler,
                         'epoch_diagnostics': copy.deepcopy(self.epoch_diagnostics)}
                with (checkpoint_dir / f'checkpoint_epoch{epoch}.pt').open('xb') as stream:
                    torch.save(state, stream)
            if progress:
                progress(epoch, self.train_losses[-1], self.auxiliary_losses[-1], record)
        self.network.eval().cpu()
        self.eta = nn.Parameter(self.eta.detach().cpu())
        self.training_device = str(device)
        return snapshots
