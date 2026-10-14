"""Small retrospective graph controls with TRAIN scaling and STOP-only fitting.

Only supplied node and current-roster edge features enter these models. The
caller owns chronological feature construction, frozen baseline fitting, and
any subsequent calibration. This module does no work at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TypedMessageLayer(nn.Module):
    """Mean messages over separate teammate and opponent relation types."""

    def __init__(self, edge_channels: int) -> None:
        super().__init__()
        self.sources = nn.ModuleList([nn.Linear(16, 16) for _ in range(2)])
        history_channels = edge_channels - 2
        self.gates = nn.ModuleList(
            [nn.Linear(history_channels, 16) for _ in range(2)]
            if history_channels
            else []
        )
        self.edge_values = nn.ModuleList(
            [nn.Linear(history_channels, 16, bias=False) for _ in range(2)]
            if history_channels
            else []
        )
        self.update = nn.Linear(48, 16)

    def forward(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        history = edges[..., 2:]
        aggregates = []
        for relation in range(2):
            mask = edges[..., relation : relation + 1]
            # Edge[i, j] carries the historical relation from receiver i to j.
            messages = self.sources[relation](h).unsqueeze(1)
            if self.gates:
                messages = torch.sigmoid(
                    self.gates[relation](history)
                ) * messages + torch.tanh(self.edge_values[relation](history))
            # Sum in float64 so changing neighbor order does not accumulate
            # float32 rounding error before the strict invariance guard.
            aggregate = (messages * mask).sum(dim=2, dtype=torch.float64)
            aggregate = aggregate / mask.sum(dim=2).clamp_min(1.0)
            aggregates.append(aggregate.to(h.dtype))
        return torch.tanh(h + self.update(torch.cat([h, *aggregates], dim=-1)))


class GraphResidualModel(nn.Module):
    """Shared player encoding and shared team scoring, without identity tables.

    Roles stay attached to node features under permutation. Team membership is
    the first/last five nodes; neither encoder nor scorer receives a side ID.
    The unbounded linear correction is initially exactly zero.
    """

    def __init__(self, kind: str, node_channels: int, edge_channels: int) -> None:
        super().__init__()
        if kind not in {"deepsets", "player_gnn"}:
            raise ValueError(f"unknown graph variant: {kind}")
        self.node_encoder = nn.Sequential(
            nn.Linear(node_channels, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh()
        )
        self.team_scorer = nn.Sequential(
            nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1, bias=False)
        )
        nn.init.zeros_(self.team_scorer[-1].weight)
        # Create the common encoder/readout first so their seeded initial
        # parameters are identical across the graph and no-edge controls.
        self.messages = (
            TypedMessageLayer(edge_channels) if kind == "player_gnn" else None
        )

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, baseline_logits: torch.Tensor
    ) -> torch.Tensor:
        h = self.node_encoder(nodes)
        if self.messages is not None:
            h = self.messages(h, edges)
        sides = torch.stack(
            (
                h[:, :5].mean(dim=1, dtype=torch.float64),
                h[:, 5:].mean(dim=1, dtype=torch.float64),
            ),
            dim=1,
        ).to(h.dtype)
        scores = self.team_scorer(sides).squeeze(-1).to(torch.float64)
        correction = scores[:, 0] - scores[:, 1]
        # Float64 addition preserves the supplied frozen offset and keeps
        # roundoff in counterfactual comparisons independent of offset size.
        return baseline_logits.to(torch.float64) + correction.to(torch.float64)


def _fit_scaler(values: np.ndarray, protected: int) -> dict[str, np.ndarray]:
    """Fit on TRAIN observations only; binary indicators remain literal masks."""
    binary = np.all((values == 0.0) | (values == 1.0), axis=0)
    binary[:protected] = True
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    mean[binary] = 0.0
    scale[binary] = 1.0
    return {
        "mean": mean.astype(np.float32),
        "scale": scale.astype(np.float32),
        "binary": binary,
    }


def _standardize(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    result = (values - scaler["mean"]) / scaler["scale"]
    np.clip(result, -10.0, 10.0, out=result)
    return result.astype(np.float32, copy=False)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _predict(
    model: GraphResidualModel,
    nodes: torch.Tensor,
    edges: torch.Tensor,
    offsets: torch.Tensor,
    indices: np.ndarray,
) -> np.ndarray:
    predictions = np.empty(len(indices), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            predictions[start : start + len(batch)] = model(
                nodes[batch], edges[batch], offsets[batch]
            ).numpy()
    if not np.isfinite(predictions).all():
        raise FloatingPointError("graph model produced non-finite logits")
    return predictions


def _log_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def _counterfactual_errors(
    model: GraphResidualModel,
    nodes: torch.Tensor,
    edges: torch.Tensor,
    offsets: torch.Tensor,
    original: np.ndarray,
) -> tuple[float, float]:
    """Exercise full forward calls, permuting both edge axes with the nodes."""
    swap = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    permute = torch.tensor([2, 4, 0, 1, 3, 8, 5, 9, 7, 6])
    swap_error = 0.0
    permutation_error = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(nodes), 256):
            x = nodes[start : start + 256]
            e = edges[start : start + 256]
            b = offsets[start : start + 256]
            reference = original[start : start + len(x)]
            swapped = model(x[:, swap], e[:, swap][:, :, swap], -b).numpy()
            permuted = model(x[:, permute], e[:, permute][:, :, permute], b).numpy()
            if not np.isfinite(swapped).all() or not np.isfinite(permuted).all():
                raise FloatingPointError(
                    "counterfactual forward produced non-finite logits"
                )
            swap_error = max(swap_error, float(np.max(np.abs(swapped + reference))))
            permutation_error = max(
                permutation_error, float(np.max(np.abs(permuted - reference)))
            )
    if swap_error > 1e-6 or permutation_error > 1e-6:
        raise RuntimeError(
            f"graph invariance violation: swap={swap_error}, permutation={permutation_error}"
        )
    return swap_error, permutation_error


def fit_graph_variant(
    kind: str,
    nodes: np.ndarray,
    edges: np.ndarray,
    baseline_logits: np.ndarray,
    y: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    seed: int = 42,
    max_epochs: int = 60,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Fit one CPU graph residual; never use SELECT/CALIBRATION/TEST labels.

    Scaler binary-channel detection also uses TRAIN only. The first five node
    channels are role indicators and the first two edge channels are current
    teammate/opponent masks. Edge scaling excludes absent edges and diagonals.
    The returned state contains reconstruction arguments, best weights and
    TRAIN scalers, all compatible with torch.save (and weights-only loading).
    """
    if kind not in {"deepsets", "player_gnn"}:
        raise ValueError(f"unknown graph variant: {kind}")
    if isinstance(max_epochs, (bool, np.bool_)) or not isinstance(
        max_epochs, (int, np.integer)
    ):
        raise TypeError("max_epochs must be an integer between 1 and 60")
    if not 1 <= max_epochs <= 60:
        raise ValueError("max_epochs must be between 1 and 60")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    for name, values in (
        ("nodes", nodes),
        ("edges", edges),
        ("baseline_logits", baseline_logits),
        ("y", y),
    ):
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must contain real values")
    nodes = np.asarray(nodes, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.float32)
    offsets = np.asarray(baseline_logits, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if nodes.ndim != 3 or nodes.shape[1] != 10 or nodes.shape[2] < 5:
        raise ValueError("nodes must have shape [N, 10, F>=5] with roles first")
    n = len(nodes)
    if edges.ndim != 4 or edges.shape[:3] != (n, 10, 10) or edges.shape[3] < 2:
        raise ValueError("edges must have shape [N, 10, 10, E>=2]")
    if offsets.shape != (n,) or labels.shape != (n,):
        raise ValueError("baseline_logits and y must have shape [N]")
    if not all(np.isfinite(v).all() for v in (nodes, edges, offsets, labels)):
        raise ValueError("all graph inputs must be finite")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise ValueError("y must contain binary outcomes")
    if not np.all((nodes[..., :5] == 0.0) | (nodes[..., :5] == 1.0)):
        raise ValueError("first five node channels must be role indicators")
    if not np.all(nodes[..., :5].sum(axis=-1) == 1.0):
        raise ValueError("each node must have exactly one role indicator")
    relation_masks = edges[..., :2]
    if not np.all((relation_masks == 0.0) | (relation_masks == 1.0)):
        raise ValueError("first two edge channels must be binary relation masks")
    if np.any(relation_masks.sum(axis=-1) > 1.0):
        raise ValueError("teammate and opponent masks must be disjoint")
    diagonal = np.arange(10)
    if np.any(edges[:, diagonal, diagonal] != 0.0):
        raise ValueError("edge diagonals must be zero")
    split_indices: dict[str, np.ndarray] = {}
    assigned = np.zeros(n, dtype=bool)
    for name in ("train", "stop", "select", "calibration", "test"):
        mask = np.asarray(masks[name])
        if mask.shape != (n,) or mask.dtype != np.bool_:
            raise ValueError(f"{name} mask must be a boolean array of shape [N]")
        if np.any(assigned & mask):
            raise ValueError("graph split masks must be disjoint")
        assigned |= mask
        split_indices[name] = np.flatnonzero(mask)
    train, stop = split_indices["train"], split_indices["stop"]
    if not len(train) or not len(stop):
        raise ValueError("TRAIN and STOP must both contain rows")

    node_scaler = _fit_scaler(nodes[train].reshape(-1, nodes.shape[-1]), protected=5)
    train_edges = edges[train]
    valid_train_edges = train_edges[..., :2].sum(axis=-1) > 0.0
    if not np.any(valid_train_edges):
        raise ValueError("TRAIN must contain typed roster edges")
    edge_scaler = _fit_scaler(train_edges[valid_train_edges], protected=2)
    scaled_nodes = torch.from_numpy(_standardize(nodes, node_scaler))
    scaled_edges = _standardize(edges, edge_scaler)
    scaled_edges[relation_masks.sum(axis=-1) == 0.0] = 0.0
    scaled_edges[:, diagonal, diagonal] = 0.0
    edge_tensor = torch.from_numpy(scaled_edges)
    offset_tensor = torch.from_numpy(offsets)
    label_tensor = torch.from_numpy(labels)
    model_config = {
        "kind": kind,
        "node_channels": nodes.shape[-1],
        "edge_channels": edges.shape[-1],
    }
    baseline_stop_loss = _log_loss(offsets[stop], labels[stop])
    torch.set_num_threads(2)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = GraphResidualModel(**model_config)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.003, weight_decay=0.01
            )
            best_state = _snapshot(model)
            best_loss, best_epoch, stale = baseline_stop_loss, 0, 0
            stop_history = []
            for epoch in range(1, max_epochs + 1):
                model.train()
                order = rng.permutation(train)
                for start in range(0, len(order), 256):
                    batch = order[start : start + 256]
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(
                        scaled_nodes[batch], edge_tensor[batch], offset_tensor[batch]
                    )
                    loss = F.binary_cross_entropy_with_logits(
                        prediction, label_tensor[batch]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite graph training loss")
                    loss.backward()
                    optimizer.step()
                stop_logits = _predict(
                    model, scaled_nodes, edge_tensor, offset_tensor, stop
                )
                stop_loss = _log_loss(stop_logits, labels[stop])
                stop_history.append(stop_loss)
                if stop_loss < best_loss:
                    best_loss, best_epoch, stale = stop_loss, epoch, 0
                    best_state = _snapshot(model)
                else:
                    stale += 1
                    if stale >= 8:
                        break
            model.load_state_dict(best_state)
            raw_logits = _predict(
                model, scaled_nodes, edge_tensor, offset_tensor, np.arange(n)
            )
            swap_error, permutation_error = _counterfactual_errors(
                model, scaled_nodes, edge_tensor, offset_tensor, raw_logits
            )
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic, warn_only=previous_warn_only
        )

    metadata = {
        "kind": kind,
        "seed": seed,
        "hidden_size": 16,
        "message_layers": int(kind == "player_gnn"),
        "zero_initialized_correction": True,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "batch_size": 256,
        "max_epochs": int(max_epochs),
        "patience": 8,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "baseline_stop_log_loss": baseline_stop_loss,
        "best_stop_log_loss": best_loss,
        "stop_log_loss_history": stop_history,
        "split_counts": {name: len(indices) for name, indices in split_indices.items()},
        "max_team_swap_symmetry_error": swap_error,
        "max_within_roster_permutation_error": permutation_error,
        "counterfactual_rows": n,
        "node_binary_channels": np.flatnonzero(node_scaler["binary"]).tolist(),
        "edge_binary_channels": np.flatnonzero(edge_scaler["binary"]).tolist(),
        "scaler_fit_split": "train",
        "checkpoint_selection_split": "stop",
        "calibrated": False,
        "cpu_threads": 2,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
    }
    state = {
        "model_config": model_config,
        "model_state_dict": best_state,
        "node_scaler": {
            name: torch.from_numpy(value.copy()) for name, value in node_scaler.items()
        },
        "edge_scaler": {
            name: torch.from_numpy(value.copy()) for name, value in edge_scaler.items()
        },
        "scaler_clip": 10.0,
        "metadata": metadata,
    }
    return raw_logits, metadata, state
