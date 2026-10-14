"""Leakage-safe match dataset from PlayerGameEncoder embeddings.

EXP-049 trains a one-row-per-player-game encoder on completed games.  This
module performs the next leakage boundary: for a match at date ``T`` it builds
team features only from player-game embeddings observed before ``T``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from betting_app.core.matching import normalize_team_name
from betting_app.ml.training.player_game_dataset import PlayerGameDataset
from betting_app.ml.training.player_game_encoder import _build_model, require_torch


@dataclass(frozen=True)
class PlayerEmbeddingMatchDatasetConfig:
    """Configuration for leakage-safe match-level embedding aggregation."""

    experiment_id: str = "EXP-050"
    history_size: int = 250
    min_prior_player_games: int = 50
    include_std_features: bool = True


@dataclass(frozen=True)
class PlayerEmbeddingMatchDataset:
    """Materialized match-level dataset compatible with strength_model trainer."""

    frame: pd.DataFrame
    feature_names: list[str]
    metadata: dict[str, Any]


def _team_key(team_id: Any, team_name: Any) -> str:
    raw_id = str(team_id or "").strip()
    if raw_id and raw_id.lower() != "nan":
        return f"id:{raw_id}"
    normalized = normalize_team_name(str(team_name or ""))
    return f"name:{normalized}" if normalized else ""


def _target_from_match(record: dict[str, Any]) -> int | None:
    if _to_float(record.get("team1_win")) == 1.0:
        return 1
    if _to_float(record.get("team2_win")) == 1.0:
        return 0
    winner = str(record.get("winner_name") or "").strip()
    if winner:
        team1 = str(record.get("team1_name") or "").strip()
        team2 = str(record.get("team2_name") or "").strip()
        if winner == team1:
            return 1
        if winner == team2:
            return 0
    return None


def _to_float(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if not math.isnan(converted) else default


def _numeric_from_saved_stats(frame: pd.DataFrame, stats: dict[str, Any]) -> np.ndarray:
    names = list(stats["feature_names"])
    numeric = frame[names].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(stats["median"], index=names, dtype=float)
    means = pd.Series(stats["mean"], index=names, dtype=float)
    stds = pd.Series(stats["std"], index=names, dtype=float).replace(0.0, 1.0).fillna(1.0)
    scaled = ((numeric.fillna(medians) - means) / stds).astype("float32")
    return scaled.to_numpy(dtype=np.float32)


def _encode_with_vocab(frame: pd.DataFrame, vocabularies: dict[str, dict[str, int]], categorical_names: list[str]) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for name in categorical_names:
        vocab = vocabularies.get(name, {})
        encoded[name] = frame[name].astype(str).map(vocab).fillna(0).astype("int64").to_numpy()
    return encoded


def encode_player_game_embeddings(
    dataset: PlayerGameDataset,
    *,
    encoder_artifact: Path | str,
    device: str = "auto",
    batch_size: int = 8192,
) -> pd.DataFrame:
    """Run a trained EXP-049 encoder and return one latent row per player-game."""

    torch = require_torch()
    artifact_path = Path(encoder_artifact)
    checkpoint = torch.load(artifact_path / "model.pt", map_location="cpu")
    config = checkpoint["config"]
    numeric_stats = checkpoint["numeric_stats"]
    vocabularies = checkpoint["vocabularies"]
    categorical_names = list(checkpoint["categorical_names"])
    numeric_feature_names = list(checkpoint["numeric_feature_names"])

    frame = dataset.frame.sort_values(["date", "game_id", "side", "role_index"]).reset_index(drop=True)
    x_num = _numeric_from_saved_stats(frame, numeric_stats)
    cat_arrays = _encode_with_vocab(frame, vocabularies, categorical_names)

    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if device_name == "auto":
        device_name = "cpu"
    torch_device = torch.device(device_name)

    cfg_obj = type("EncoderConfig", (), config)
    model = _build_model(
        torch,
        numeric_dim=len(numeric_feature_names),
        vocab_sizes={name: len(vocabularies[name]) for name in categorical_names},
        cfg=cfg_obj,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch_device)
    model.eval()

    latents: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            end = start + batch_size
            numeric_x = torch.from_numpy(np.array(x_num[start:end], copy=True)).to(torch_device)
            categorical_x = {
                name: torch.from_numpy(np.array(values[start:end], copy=True)).to(torch_device)
                for name, values in cat_arrays.items()
            }
            out = model(numeric_x, categorical_x)
            latents.append(out["latent"].detach().cpu().numpy().astype(np.float32))
    latent = np.vstack(latents) if latents else np.empty((0, int(config["latent_dim"])), dtype=np.float32)

    meta_cols = [
        "date",
        "match_id",
        "game_id",
        "team_id",
        "team_name",
        "opponent_team_id",
        "player_id",
        "role",
        "champion_id",
    ]
    result = frame[meta_cols].copy()
    for idx in range(latent.shape[1]):
        result[f"embedding_{idx}"] = latent[:, idx]
    return result


def build_match_dataset_from_embeddings(
    matches: pd.DataFrame,
    player_embeddings: pd.DataFrame,
    config: PlayerEmbeddingMatchDatasetConfig | None = None,
) -> PlayerEmbeddingMatchDataset:
    """Aggregate prior team embeddings into leakage-safe match examples."""

    cfg = config or PlayerEmbeddingMatchDatasetConfig()
    if matches.empty or player_embeddings.empty:
        return PlayerEmbeddingMatchDataset(pd.DataFrame(), [], {"experiment_id": cfg.experiment_id, "rows": 0})

    ordered_matches = matches.copy()
    ordered_matches["date"] = pd.to_datetime(ordered_matches["date"], errors="coerce", utc=True)
    ordered_matches["match_id"] = ordered_matches["match_id"].astype(str)
    ordered_matches = ordered_matches.dropna(subset=["date"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    embeddings = player_embeddings.copy()
    embeddings["match_id"] = embeddings["match_id"].astype(str)
    embedding_cols = [name for name in embeddings.columns if name.startswith("embedding_")]
    if not embedding_cols:
        return PlayerEmbeddingMatchDataset(pd.DataFrame(), [], {"experiment_id": cfg.experiment_id, "rows": 0, "skipped": {"no_embedding_columns": len(ordered_matches)}})

    by_match = {match_id: group for match_id, group in embeddings.groupby("match_id", sort=False)}
    histories: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=cfg.history_size))
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = defaultdict(int)

    def team_stats(key: str) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        values = list(histories.get(key, ()))
        count = len(values)
        if count < cfg.min_prior_player_games:
            return None, None, count
        matrix = np.vstack(values[-cfg.history_size:])
        return matrix.mean(axis=0), matrix.std(axis=0), count

    for _, date_group in ordered_matches.groupby("date", sort=True):
        pending_update_match_ids: list[str] = []
        for record in date_group.to_dict(orient="records"):
            match_id = str(record.get("match_id"))
            pending_update_match_ids.append(match_id)
            target = _target_from_match(record)
            if target is None:
                skipped["no_binary_target"] += 1
                continue
            key1 = _team_key(record.get("team1_id"), record.get("team1_name"))
            key2 = _team_key(record.get("team2_id"), record.get("team2_name"))
            if not key1 or not key2 or key1 == key2:
                skipped["bad_team_key"] += 1
                continue
            mean1, std1, count1 = team_stats(key1)
            mean2, std2, count2 = team_stats(key2)
            if mean1 is None or mean2 is None or std1 is None or std2 is None:
                skipped["min_prior_player_games"] += 1
                continue
            row: dict[str, Any] = {
                "match_id": match_id,
                "date": record.get("date"),
                "team1_name": record.get("team1_name"),
                "team2_name": record.get("team2_name"),
                "team1_id": record.get("team1_id"),
                "team2_id": record.get("team2_id"),
                "target": int(target),
                "team1_embedding_count": float(count1),
                "team2_embedding_count": float(count2),
                "embedding_count_diff": float(count1 - count2),
            }
            for idx, _ in enumerate(embedding_cols):
                row[f"team1_embedding_mean_{idx}"] = float(mean1[idx])
                row[f"team2_embedding_mean_{idx}"] = float(mean2[idx])
                row[f"embedding_mean_diff_{idx}"] = float(mean1[idx] - mean2[idx])
                if cfg.include_std_features:
                    row[f"team1_embedding_std_{idx}"] = float(std1[idx])
                    row[f"team2_embedding_std_{idx}"] = float(std2[idx])
                    row[f"embedding_std_diff_{idx}"] = float(std1[idx] - std2[idx])
            rows.append(row)

        # Strict date<T boundary: same-date matches cannot update histories used
        # by other matches from this date group.
        for match_id in pending_update_match_ids:
            current = by_match.get(match_id)
            if current is None:
                continue
            for team_id, team_rows in current.groupby("team_id", sort=False):
                key = _team_key(team_id, team_rows["team_name"].iloc[0] if "team_name" in team_rows else None)
                if not key:
                    continue
                for values in team_rows[embedding_cols].to_numpy(dtype=np.float32):
                    histories[key].append(values)

    frame = pd.DataFrame(rows)
    feature_names = [
        name
        for name in frame.columns
        if name not in {"match_id", "date", "team1_name", "team2_name", "team1_id", "team2_id", "target"}
    ]
    metadata = {
        "experiment_id": cfg.experiment_id,
        "purpose": "Leakage-safe match-level aggregation of EXP-049 player-game embeddings.",
        "leakage_note": "For match date T, histories are updated only after all matches at T are emitted; features use date < T only.",
        "config": asdict(cfg),
        "raw_matches": int(len(matches)),
        "player_embedding_rows": int(len(player_embeddings)),
        "embedding_dim": int(len(embedding_cols)),
        "rows": int(len(frame)),
        "feature_count": int(len(feature_names)),
        "skipped": {key: int(value) for key, value in skipped.items() if value},
    }
    if not frame.empty:
        metadata.update(
            {
                "date_min": pd.to_datetime(frame["date"], errors="coerce").min().isoformat(),
                "date_max": pd.to_datetime(frame["date"], errors="coerce").max().isoformat(),
            }
        )
    return PlayerEmbeddingMatchDataset(frame=frame, feature_names=feature_names, metadata=metadata)


def iter_match_embedding_rows(dataset: PlayerEmbeddingMatchDataset) -> Iterable[dict[str, Any]]:
    """Yield JSON-serializable match embedding rows."""

    for row in dataset.frame.replace({np.nan: None}).to_dict(orient="records"):
        if hasattr(row.get("date"), "isoformat"):
            row["date"] = row["date"].isoformat()
        yield row


def save_match_embedding_dataset(dataset: PlayerEmbeddingMatchDataset, path: Path | str) -> Path:
    """Persist a match embedding dataset as JSONL plus metadata."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2, default=str) + "\n")
    (root / "feature_names.json").write_text(json.dumps(dataset.feature_names, indent=2) + "\n")
    with (root / "match_dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in iter_match_embedding_rows(dataset):
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return root
