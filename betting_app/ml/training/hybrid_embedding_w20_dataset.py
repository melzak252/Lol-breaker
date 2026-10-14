"""EXP-051 hybrid dataset: player-game embeddings plus W20 rating context.

EXP-050 proved that leakage-safe PlayerGameEncoder embeddings add signal over
the pure team-strength baseline, but they did not beat the older W20/player
rating feature space from EXP-039/EXP-045.  This module merges both sources at
the same leakage boundary: every match row uses only pre-match embedding
histories and pre-match W20/rating predictions.
"""

from __future__ import annotations

import json
import gzip
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from betting_app.ml.training.player_embedding_match_dataset import PlayerEmbeddingMatchDataset
from src.utils.golgg_schema import games, team1_id, team2_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TARGET = "target"
LEGACY_TARGET = "y_true"
MATCH_KEY = "golgg_match_id"

PLAYER_RANK_PROB_FEATURES = ["player_elo", "player_gl", "player_ts", "player_os", "player_pl", "player_tm"]
TEAM_RANK_PROB_FEATURES = ["team_elo", "team_gl", "team_ts", "team_os", "team_pl", "team_tm"]

OPTUNA_BASE_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
    "player_elo_min1",
    "player_elo_min2",
    "player_gl_max1",
    "player_gl_max2",
    "player_gl_rd_avg1",
    "player_gl_rd_avg2",
    "player_ts_sigma_avg1",
    "player_ts_sigma_avg2",
    "player_os_sigma_avg1",
    "player_os_sigma_avg2",
    "player_pl_sigma_avg1",
    "player_pl_sigma_avg2",
    "player_tm_sigma_avg1",
    "player_tm_sigma_avg2",
]

ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate",
    "t2_rolling_win_rate",
    "t1_rolling_kills",
    "t2_rolling_kills",
    "t1_rolling_deaths",
    "t2_rolling_deaths",
    "t1_rolling_gd15",
    "t2_rolling_gd15",
    "t1_rolling_dpm",
    "t2_rolling_dpm",
    "t1_rolling_vspm",
    "t2_rolling_vspm",
    "t1_rolling_towers",
    "t2_rolling_towers",
    "t1_rolling_nashors",
    "t2_rolling_nashors",
    "t1_rolling_gold",
    "t2_rolling_gold",
    "t1_rolling_duration",
    "t2_rolling_duration",
]

TEAM_RATING_CONTEXT_FEATURES = [
    "team_elo_r1",
    "team_elo_r2",
    "team_gl_r1",
    "team_gl_rd1",
    "team_gl_r2",
    "team_gl_rd2",
    "team_ts_mu1",
    "team_ts_sigma1",
    "team_ts_mu2",
    "team_ts_sigma2",
    "team_os_mu1",
    "team_os_sigma1",
    "team_os_mu2",
    "team_os_sigma2",
    "team_pl_mu1",
    "team_pl_sigma1",
    "team_pl_mu2",
    "team_pl_sigma2",
    "team_tm_mu1",
    "team_tm_sigma1",
    "team_tm_mu2",
    "team_tm_sigma2",
]

DAYS_CONTEXT_FEATURES = ["days_since_last_1", "days_since_last_2", "days_diff"]


@dataclass(frozen=True)
class HybridEmbeddingW20Config:
    """Configuration for EXP-051 hybrid dataset construction."""

    experiment_id: str = "EXP-051"
    data_dir: str = "data"
    rolling_window: int = 20
    min_date: str = "2020-01-01"
    max_date: str | None = None
    market_common_only: bool = False
    include_team_features: bool = True
    include_days_features: bool = True
    require_target_agreement: bool = True


@dataclass(frozen=True)
class HybridEmbeddingW20Dataset:
    """Materialized EXP-051 dataset compatible with strength_model."""

    frame: pd.DataFrame
    feature_names: list[str]
    metadata: dict[str, Any]


def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    """Convert map win probability to best-of-series probability.

    Mirrors the W20-Binomial transform from EXP-039/045 scripts without pulling
    in notebook/report dependencies.
    """

    probabilities = np.clip(map_probability.astype(float), 0.001, 0.999)
    best_of_int = np.maximum(best_of.astype(float), 1).astype(int)
    result = np.zeros_like(probabilities, dtype=float)
    for idx, (probability, best_of_value) in enumerate(zip(probabilities, best_of_int, strict=False)):
        wins_needed = best_of_value // 2 + 1
        total = 0.0
        for wins in range(wins_needed, best_of_value + 1):
            total += math.comb(best_of_value, wins) * (probability**wins) * ((1.0 - probability) ** (best_of_value - wins))
        result[idx] = total
    return result


def add_binomial_features(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add `<probability>_binom_series` features using the `BoN` column."""

    out = data.copy()
    created: list[str] = []
    best_of = out.get("BoN", pd.Series(1, index=out.index)).fillna(1).to_numpy(dtype=float)
    for feature in features:
        if feature not in out.columns:
            continue
        name = f"{feature}_binom_series"
        out[name] = series_probability(out[feature].to_numpy(dtype=float), best_of)
        created.append(name)
    return out, created


def load_legacy_w20_features(config: HybridEmbeddingW20Config | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load EXP-039/045-style rating/W20 features from local data files."""

    cfg = config or HybridEmbeddingW20Config()
    data_dir = Path(cfg.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    predictions = pd.read_csv(data_dir / "golgg_y_predicts.csv")
    predictions[MATCH_KEY] = predictions[MATCH_KEY].astype(str)
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce", utc=True)
    predictions = predictions.dropna(subset=["date", MATCH_KEY]).copy()

    if cfg.market_common_only:
        odds = pd.read_csv(data_dir / "odds.csv", usecols=[MATCH_KEY])
        odds[MATCH_KEY] = odds[MATCH_KEY].astype(str)
        predictions = predictions.merge(odds.drop_duplicates(), on=MATCH_KEY, how="inner")

    rolling_source = data_dir / "golgg_matches.json"
    if not rolling_source.exists():
        compressed_source = data_dir / "golgg_matches.json.gz"
        if compressed_source.exists():
            rolling_source = compressed_source
    rolling = generate_rolling_features_from_json(rolling_source, cfg.rolling_window)
    rolling[MATCH_KEY] = rolling[MATCH_KEY].astype(str)
    data = predictions.merge(rolling.drop(columns=["context_window"], errors="ignore"), on=MATCH_KEY, how="left")

    if cfg.min_date:
        data = data[data["date"] >= pd.Timestamp(cfg.min_date, tz="UTC")].copy()
    if cfg.max_date:
        data = data[data["date"] <= pd.Timestamp(cfg.max_date, tz="UTC")].copy()

    rank_features = list(PLAYER_RANK_PROB_FEATURES)
    if cfg.include_team_features:
        rank_features += TEAM_RANK_PROB_FEATURES
    data, binomial_features = add_binomial_features(data, rank_features)

    feature_names = [name for name in OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES if name in data.columns]
    if cfg.include_team_features:
        feature_names += [name for name in TEAM_RANK_PROB_FEATURES + TEAM_RATING_CONTEXT_FEATURES if name in data.columns]
    if cfg.include_days_features:
        feature_names += [name for name in DAYS_CONTEXT_FEATURES if name in data.columns]
    feature_names += binomial_features

    keep_cols = [MATCH_KEY, "date", LEGACY_TARGET, "BoN", "team1_id", "team2_id", "team1_name", "team2_name"]
    keep_cols = [name for name in keep_cols if name in data.columns]
    data = data[keep_cols + feature_names].sort_values(["date", MATCH_KEY]).reset_index(drop=True)
    return data, feature_names


def build_hybrid_embedding_w20_dataset(
    embedding_dataset: PlayerEmbeddingMatchDataset,
    legacy_features: pd.DataFrame,
    legacy_feature_names: list[str],
    config: HybridEmbeddingW20Config | None = None,
) -> HybridEmbeddingW20Dataset:
    """Merge EXP-050 embedding features with EXP-039/045 W20 features."""

    cfg = config or HybridEmbeddingW20Config()
    if embedding_dataset.frame.empty or legacy_features.empty:
        return HybridEmbeddingW20Dataset(pd.DataFrame(), [], {"experiment_id": cfg.experiment_id, "rows": 0})

    embeddings = embedding_dataset.frame.copy()
    embeddings["match_id"] = embeddings["match_id"].astype(str)
    legacy = legacy_features.copy()
    legacy[MATCH_KEY] = legacy[MATCH_KEY].astype(str)

    merged = embeddings.merge(legacy, left_on="match_id", right_on=MATCH_KEY, how="inner", suffixes=("", "_legacy"))
    skipped: dict[str, int] = {"missing_legacy_features": int(len(embeddings) - len(merged))}

    if cfg.require_target_agreement and LEGACY_TARGET in merged.columns:
        legacy_target = pd.to_numeric(merged[LEGACY_TARGET], errors="coerce")
        agree_mask = legacy_target.isna() | (legacy_target.astype("Int64") == merged["target"].astype("Int64"))
        skipped["target_disagreement"] = int((~agree_mask).sum())
        merged = merged[agree_mask].copy()

    if cfg.min_date:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce", utc=True)
        before = len(merged)
        merged = merged[merged["date"] >= pd.Timestamp(cfg.min_date, tz="UTC")].copy()
        skipped["before_min_date"] = int(before - len(merged))
    if cfg.max_date:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce", utc=True)
        before = len(merged)
        merged = merged[merged["date"] <= pd.Timestamp(cfg.max_date, tz="UTC")].copy()
        skipped["after_max_date"] = int(before - len(merged))

    feature_names = list(embedding_dataset.feature_names) + [name for name in legacy_feature_names if name in merged.columns]
    merged = merged.dropna(subset=["target"]).sort_values(["date", "match_id"]).reset_index(drop=True)

    metadata = {
        "experiment_id": cfg.experiment_id,
        "purpose": "Hybrid EXP-050 PlayerGameEncoder embeddings plus EXP-039/045 W20 rating features.",
        "leakage_note": "Embedding features use date<T histories; W20/rating features are pre-match predictions from historical EXP-039/045 pipelines.",
        "config": asdict(cfg),
        "embedding_rows": int(len(embedding_dataset.frame)),
        "legacy_rows": int(len(legacy_features)),
        "rows": int(len(merged)),
        "embedding_feature_count": int(len(embedding_dataset.feature_names)),
        "legacy_feature_count": int(len([name for name in legacy_feature_names if name in merged.columns])),
        "feature_count": int(len(feature_names)),
        "skipped": {key: int(value) for key, value in skipped.items() if value},
    }
    if not merged.empty:
        metadata.update(
            {
                "date_min": pd.to_datetime(merged["date"], errors="coerce").min().isoformat(),
                "date_max": pd.to_datetime(merged["date"], errors="coerce").max().isoformat(),
            }
        )
    return HybridEmbeddingW20Dataset(frame=merged, feature_names=feature_names, metadata=metadata)


def generate_rolling_features_from_json(path: Path | str, window_size: int) -> pd.DataFrame:
    """Generate W20 rolling team context from raw GOL.GG JSON."""

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8") as fh:
        matches = json.load(fh)
    matches.sort(key=lambda item: str(item.get("date") or ""))

    team_history: dict[str, deque[dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []
    for match in matches:
        team_1 = str(team1_id(match) or "")
        team_2 = str(team2_id(match) or "")
        t1_stats = _average_history(team_history.get(team_1))
        t2_stats = _average_history(team_history.get(team_2))
        row: dict[str, Any] = {MATCH_KEY: str(match.get("match_id")), "context_window": int(window_size)}
        for stat, value in t1_stats.items():
            row[f"t1_rolling_{stat}"] = value
        for stat, value in t2_stats.items():
            row[f"t2_rolling_{stat}"] = value
        rows.append(row)

        for game in games(match):
            _update_team_history(team_history, team_1, game, window_size)
            _update_team_history(team_history, team_2, game, window_size)
    return pd.DataFrame(rows)


def iter_hybrid_rows(dataset: HybridEmbeddingW20Dataset) -> Iterable[dict[str, Any]]:
    """Yield JSON-serializable hybrid rows."""

    for row in dataset.frame.replace({np.nan: None}).to_dict(orient="records"):
        if hasattr(row.get("date"), "isoformat"):
            row["date"] = row["date"].isoformat()
        yield row


def save_hybrid_embedding_w20_dataset(dataset: HybridEmbeddingW20Dataset, path: Path | str) -> Path:
    """Persist EXP-051 dataset snapshot."""

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (root / "feature_names.json").write_text(json.dumps(dataset.feature_names, indent=2) + "\n", encoding="utf-8")
    with (root / "hybrid_dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in iter_hybrid_rows(dataset):
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return root


def _default_team_stats() -> dict[str, float]:
    return {
        "win_rate": 0.5,
        "kills": 12.0,
        "deaths": 12.0,
        "gd15": 0.0,
        "dpm": 1800.0,
        "vspm": 7.0,
        "towers": 5.0,
        "nashors": 0.5,
        "gold": 55000.0,
        "duration": 1800.0,
    }


def _average_history(history: deque[dict[str, float]] | None) -> dict[str, float]:
    if not history:
        return _default_team_stats()
    rows = list(history)
    return {key: float(np.mean([row[key] for row in rows])) for key in _default_team_stats()}


def _update_team_history(
    team_history: dict[str, deque[dict[str, float]]],
    team_id: str,
    game: dict[str, Any],
    window_size: int,
) -> None:
    if not team_id:
        return
    is_team_1 = str(game.get("t1_id")) == str(team_id)
    players_key = "t1_players" if is_team_1 else "t2_players"
    stats_key = "t1_stats" if is_team_1 else "t2_stats"
    players = game.get(players_key, {}) or {}
    game_stats = {
        "win_rate": float(bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))),
        "kills": sum(_safe_player_stat(player, "kills") for player in players.values()),
        "deaths": sum(_safe_player_stat(player, "deaths") for player in players.values()),
        "dpm": sum(_safe_player_stat(player, "dpm") for player in players.values()),
        "vspm": sum(_safe_player_stat(player, "vspm") for player in players.values()),
        "gd15": sum(_safe_player_stat(player, "gd@15") for player in players.values()),
        "towers": _safe_team_stat(game, stats_key, "towers"),
        "nashors": _safe_team_stat(game, stats_key, "nashors"),
        "gold": _safe_team_stat(game, stats_key, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    if team_id not in team_history:
        team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)


def _safe_player_stat(player: dict[str, Any], key: str) -> float:
    value = (player.get("stats", {}) or {}).get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_team_stat(game: dict[str, Any], stats_key: str, key: str) -> float:
    value = (game.get(stats_key, {}) or {}).get(key, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
