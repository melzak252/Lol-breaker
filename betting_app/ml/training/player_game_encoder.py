"""Training utilities for EXP-049 PlayerGameEncoder.

The module is intentionally importable without PyTorch so regular backend tests
can run on lightweight local environments.  PyTorch is imported lazily only when
``train_player_game_encoder`` is executed, which should normally happen on the
CUDA server.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from betting_app.ml.training.player_game_dataset import PlayerGameDataset


@dataclass(frozen=True)
class PlayerGameEncoderConfig:
    """Configuration for the supervised+denoising player-game encoder."""

    experiment_id: str = "EXP-049"
    model_name: str = "PlayerGameEncoder"
    model_version: str = "exp-049"
    embedding_dim: int = 16
    hidden_dim: int = 128
    latent_dim: int = 64
    dropout: float = 0.15
    batch_size: int = 2048
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15
    random_state: int = 42
    reconstruction_weight: float = 0.50
    match_win_weight: float = 1.00
    game_win_weight: float = 0.50
    device: str = "auto"
    num_workers: int = 0


@dataclass(frozen=True)
class PlayerGameEncoderTrainingResult:
    artifact_path: Path
    metadata: dict[str, Any]
    history: list[dict[str, float]]


def require_torch() -> Any:
    """Import PyTorch lazily and raise a clear setup error when unavailable."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required for PlayerGameEncoder training. Install GPU build on the server, "
            "e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu129` "
            "or use the CUDA wheel appropriate for the installed driver."
        ) from exc
    return torch


def build_vocabulary(values: pd.Series) -> dict[str, int]:
    """Build stable categorical vocabulary with 0 reserved for unknown/missing."""

    unique = sorted({str(value) for value in values.dropna().astype(str) if str(value)})
    return {value: idx + 1 for idx, value in enumerate(unique)}


def build_vocabularies(dataset: PlayerGameDataset) -> dict[str, dict[str, int]]:
    """Build vocabularies for all categorical columns used by the encoder."""

    return {name: build_vocabulary(dataset.frame[name]) for name in dataset.categorical_names}


def encode_categoricals(frame: pd.DataFrame, vocabularies: dict[str, dict[str, int]]) -> dict[str, np.ndarray]:
    """Encode categorical columns into integer arrays."""

    encoded: dict[str, np.ndarray] = {}
    for name, vocab in vocabularies.items():
        values = frame[name].astype(str).map(vocab).fillna(0).astype("int64").to_numpy()
        encoded[name] = values
    return encoded


def prepare_numeric_matrix(dataset: PlayerGameDataset) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Median-impute and standardize numeric features for neural training."""

    numeric = dataset.frame[dataset.feature_names].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = numeric.median(axis=0, skipna=True).fillna(0.0)
    filled = numeric.fillna(medians)
    means = filled.mean(axis=0)
    stds = filled.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    scaled = ((filled - means) / stds).astype("float32")
    stats = {
        "feature_names": list(dataset.feature_names),
        "median": [float(medians[name]) for name in dataset.feature_names],
        "mean": [float(means[name]) for name in dataset.feature_names],
        "std": [float(stds[name]) for name in dataset.feature_names],
    }
    return scaled.to_numpy(dtype=np.float32), stats


def _valid_binary_target(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(values)
    values = np.nan_to_num(values, nan=0.0).astype(np.float32)
    return values, mask.astype(np.float32)


def _chronological_split(frame: pd.DataFrame, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if frame.empty:
        raise ValueError("Cannot train PlayerGameEncoder on an empty dataset")
    order = np.arange(len(frame), dtype=np.int64)
    val_size = max(1, int(math.ceil(len(order) * validation_fraction)))
    if len(order) - val_size < 1:
        raise ValueError("Dataset is too small for chronological train/validation split")
    return order[:-val_size], order[-val_size:]


def _build_model(torch: Any, *, numeric_dim: int, vocab_sizes: dict[str, int], cfg: PlayerGameEncoderConfig) -> Any:
    nn = torch.nn

    class PlayerGameEncoderModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding_names = list(vocab_sizes.keys())
            self.embeddings = nn.ModuleDict(
                {
                    name: nn.Embedding(size + 1, cfg.embedding_dim, padding_idx=0)
                    for name, size in vocab_sizes.items()
                }
            )
            input_dim = numeric_dim + cfg.embedding_dim * len(vocab_sizes)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.latent_dim),
                nn.ReLU(),
            )
            self.reconstruction_head = nn.Linear(cfg.latent_dim, numeric_dim)
            self.match_win_head = nn.Linear(cfg.latent_dim, 1)
            self.game_win_head = nn.Linear(cfg.latent_dim, 1)

        def forward(self, numeric_x: Any, categorical_x: dict[str, Any]) -> dict[str, Any]:
            embedded = [self.embeddings[name](categorical_x[name]) for name in self.embedding_names]
            joined = torch.cat([numeric_x, *embedded], dim=1) if embedded else numeric_x
            latent = self.encoder(joined)
            return {
                "latent": latent,
                "reconstruction": self.reconstruction_head(latent),
                "match_win_logit": self.match_win_head(latent).squeeze(1),
                "game_win_logit": self.game_win_head(latent).squeeze(1),
            }

    return PlayerGameEncoderModule()


def train_player_game_encoder(
    dataset: PlayerGameDataset,
    config: PlayerGameEncoderConfig | None = None,
    *,
    artifact_root: Path | str = Path("betting_app/models/ml"),
) -> PlayerGameEncoderTrainingResult:
    """Train a supervised+denoising encoder and write versioned artifacts.

    The objective combines numeric feature reconstruction with game/match win
    heads.  The trained latent layer is later intended to be aggregated with a
    strict ``player_game.date < predicted_match.date`` filter.
    """

    cfg = config or PlayerGameEncoderConfig()
    torch = require_torch()
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    device_name = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else cfg.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    frame = dataset.frame.sort_values(["date", "game_id", "side", "role_index"]).reset_index(drop=True)
    dataset = PlayerGameDataset(frame, dataset.feature_names, dataset.categorical_names, dataset.target_names, dataset.metadata)
    x_num, numeric_stats = prepare_numeric_matrix(dataset)
    vocabularies = build_vocabularies(dataset)
    cat_arrays = encode_categoricals(dataset.frame, vocabularies)
    y_match, y_match_mask = _valid_binary_target(dataset.frame, "match_win")
    y_game, y_game_mask = _valid_binary_target(dataset.frame, "game_win")

    train_idx, val_idx = _chronological_split(dataset.frame, cfg.validation_fraction)
    model = _build_model(torch, numeric_dim=x_num.shape[1], vocab_sizes={k: len(v) for k, v in vocabularies.items()}, cfg=cfg)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    mse = torch.nn.MSELoss()

    tensors = {
        "x_num": torch.from_numpy(np.array(x_num, copy=True)),
        "y_match": torch.from_numpy(y_match),
        "y_match_mask": torch.from_numpy(y_match_mask),
        "y_game": torch.from_numpy(y_game),
        "y_game_mask": torch.from_numpy(y_game_mask),
        **{f"cat_{name}": torch.from_numpy(np.array(values, copy=True)) for name, values in cat_arrays.items()},
    }
    rng = np.random.default_rng(cfg.random_state)

    def run_epoch(indices: np.ndarray, *, training: bool) -> dict[str, float]:
        if training:
            model.train()
            shuffled = indices.copy()
            rng.shuffle(shuffled)
        else:
            model.eval()
            shuffled = indices
        losses: list[float] = []
        match_correct = 0.0
        match_total = 0.0
        for start in range(0, len(shuffled), cfg.batch_size):
            batch_idx = shuffled[start : start + cfg.batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long)
            numeric_x = tensors["x_num"][idx].to(device)
            categorical_x = {name: tensors[f"cat_{name}"][idx].to(device) for name in dataset.categorical_names}
            target_match = tensors["y_match"][idx].to(device)
            mask_match = tensors["y_match_mask"][idx].to(device)
            target_game = tensors["y_game"][idx].to(device)
            mask_game = tensors["y_game_mask"][idx].to(device)
            with torch.set_grad_enabled(training):
                out = model(numeric_x, categorical_x)
                recon_loss = mse(out["reconstruction"], numeric_x)
                match_loss_vec = bce(out["match_win_logit"], target_match) * mask_match
                game_loss_vec = bce(out["game_win_logit"], target_game) * mask_game
                match_loss = match_loss_vec.sum() / mask_match.sum().clamp_min(1.0)
                game_loss = game_loss_vec.sum() / mask_game.sum().clamp_min(1.0)
                loss = (
                    cfg.reconstruction_weight * recon_loss
                    + cfg.match_win_weight * match_loss
                    + cfg.game_win_weight * game_loss
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            losses.append(float(loss.detach().cpu()))
            prob = torch.sigmoid(out["match_win_logit"]).detach()
            pred = (prob >= 0.5).float()
            match_correct += float(((pred == target_match) * mask_match).sum().cpu())
            match_total += float(mask_match.sum().cpu())
        return {
            "loss": float(np.mean(losses)) if losses else math.nan,
            "match_accuracy": float(match_correct / match_total) if match_total else math.nan,
        }

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(train_idx, training=True)
        val_metrics = run_epoch(val_idx, training=False)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_match_accuracy": train_metrics["match_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_match_accuracy": val_metrics["match_accuracy"],
            }
        )

    artifact_path = Path(artifact_root) / cfg.model_name / cfg.model_version
    artifact_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_id": cfg.experiment_id,
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "config": asdict(cfg),
        "dataset_metadata": dataset.metadata,
        "numeric_feature_count": len(dataset.feature_names),
        "categorical_names": dataset.categorical_names,
        "vocab_sizes": {name: len(vocab) for name, vocab in vocabularies.items()},
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
        "final_metrics": history[-1] if history else {},
        "leakage_note": "Use exported embeddings only from games with date < predicted match date.",
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "numeric_feature_names": dataset.feature_names,
            "categorical_names": dataset.categorical_names,
            "vocabularies": vocabularies,
            "numeric_stats": numeric_stats,
            "config": asdict(cfg),
            "metadata": metadata,
        },
        artifact_path / "model.pt",
    )
    (artifact_path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (artifact_path / "numeric_stats.json").write_text(json.dumps(numeric_stats, indent=2, sort_keys=True) + "\n")
    (artifact_path / "vocabularies.json").write_text(json.dumps(vocabularies, indent=2, sort_keys=True) + "\n")
    (artifact_path / "training_history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")
    return PlayerGameEncoderTrainingResult(artifact_path=artifact_path, metadata=metadata, history=history)
