"""EXP-056 champion-role embedding artifact builder.

Builds leakage-safe champion-role aggregate vectors from completed GOL.GG
player-game rows.  The default artifact uses a reference date equal to the last
available GOL.GG game date and chooses the first sufficient history window per
champion-role pair: 90d -> 180d -> 365d -> all history.  The resulting vectors
are standardised numeric aggregates that can be consumed by future
context-aware PlayerGameEncoder and match-level pipelines.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db


@dataclass(frozen=True)
class ChampionRoleEmbeddingConfig:
    experiment_id: str = "EXP-056"
    model_name: str = "ChampionRoleEmbeddings"
    model_version: str = "exp-056"
    min_date: str = "2020-01-01"
    max_date: str | None = None
    reference_date: str | None = None
    min_recent_games: int = 20
    windows_days: tuple[int, ...] = (90, 180, 365)
    decay_half_life_days: float = 180.0
    shrinkage_prior_games: float = 20.0
    output_dir: str = "betting_app/models/ml/champion_role_embeddings/exp-056"


BASE_FEATURES: tuple[str, ...] = (
    "game_win",
    "side_blue",
    "game_duration_seconds",
    "stat_kills",
    "stat_deaths",
    "stat_assists",
    "stat_cs",
    "stat_csm",
    "stat_golds",
    "stat_gpm",
    "stat_gold%",
    "stat_total_damage_to_champion",
    "stat_dpm",
    "stat_dmg%",
    "stat_kp%",
    "stat_wards_placed",
    "stat_wards_destroyed",
    "stat_control_wards_purchased",
    "stat_vision_score",
    "stat_vspm",
    "stat_gd@15",
    "stat_csd@15",
    "stat_xpd@15",
    "stat_lvld@15",
    "team_kills",
    "opp_team_kills",
    "team_gold",
    "opp_team_gold",
    "team_gold_diff",
    "team_kill_diff",
)

DERIVED_FEATURES: tuple[str, ...] = (
    "kda",
    "kill_participation",
    "damage_share",
    "gold_share",
    "damage_per_gold",
    "deaths_per_min",
)

ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MID", "ADC", "SUPPORT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default=ChampionRoleEmbeddingConfig.min_date)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--reference-date", default=None, help="Leakage cutoff date. Defaults to last loaded game date.")
    parser.add_argument("--min-recent-games", type=int, default=ChampionRoleEmbeddingConfig.min_recent_games)
    parser.add_argument("--windows-days", default="90,180,365")
    parser.add_argument("--decay-half-life-days", type=float, default=ChampionRoleEmbeddingConfig.decay_half_life_days)
    parser.add_argument("--shrinkage-prior-games", type=float, default=ChampionRoleEmbeddingConfig.shrinkage_prior_games, help="Empirical-Bayes prior strength toward role default for sparse champion-role pairs.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke-test row limit.")
    parser.add_argument("--output-dir", default=ChampionRoleEmbeddingConfig.output_dir)
    parser.add_argument("--walk-forward", action="store_true", help="Also build leakage-safe historical snapshots.")
    parser.add_argument("--snapshot-start", default=None, help="First snapshot reference date, e.g. 2025-01-01. Defaults to last 18 months.")
    parser.add_argument("--snapshot-end", default=None, help="Last snapshot reference date. Defaults to latest available date + 1 day.")
    parser.add_argument("--snapshot-frequency", choices=("MS", "W-MON"), default="MS", help="Snapshot cadence: month-start or weekly Monday.")
    return parser.parse_args()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def _prepare_frame(frame: pd.DataFrame, *, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df[df["date"].notna() & (df["date"] < reference_date)].copy()
    df["role"] = df["role"].astype(str).str.upper()
    df = df[df["role"].isin(ROLES)]
    df["champion_key"] = df["champion_id"].fillna("").astype(str)
    missing_id = df["champion_key"].isin(["", "None", "nan"])
    df.loc[missing_id, "champion_key"] = df.loc[missing_id, "champion_name"].fillna("UNKNOWN").astype(str)
    df["champion_name"] = df["champion_name"].fillna(df["champion_key"])

    # Derived champion-style proxies.  These are per-player-game because GOL.GG
    # champion rows are player-role observations, not draft-level rows.
    df["kda"] = (df["stat_kills"] + df["stat_assists"]) / df["stat_deaths"].clip(lower=1)
    df["kill_participation"] = _safe_div(df["stat_kills"] + df["stat_assists"], df["team_kills"])
    df["damage_share"] = df["stat_dmg%"]
    df["gold_share"] = df["stat_gold%"]
    df["damage_per_gold"] = _safe_div(df["stat_total_damage_to_champion"], df["stat_golds"])
    minutes = df["game_duration_seconds"] / 60.0
    df["deaths_per_min"] = _safe_div(df["stat_deaths"], minutes)

    for col in (*BASE_FEATURES, *DERIVED_FEATURES):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return math.nan
    return float(np.average(values[mask].to_numpy(dtype=float), weights=weights[mask].to_numpy(dtype=float)))


def _aggregate_group(
    group: pd.DataFrame,
    *,
    reference_date: pd.Timestamp,
    all_history_decay: bool,
    decay_half_life_days: float = 180.0,
) -> dict[str, Any]:
    ages_days = (reference_date - group["date"]).dt.total_seconds() / 86400.0
    if all_history_decay:
        # Half-life weighting makes old games available but naturally weaker.
        weights = np.power(0.5, ages_days / max(decay_half_life_days, 1e-6))
    else:
        weights = np.ones(len(group), dtype=float)
    w = pd.Series(weights, index=group.index)

    out: dict[str, Any] = {
        "n_games": int(len(group)),
        "date_min": group["date"].min().isoformat(),
        "date_max": group["date"].max().isoformat(),
        "age_days_mean": float(ages_days.mean()),
        "age_days_max": float(ages_days.max()),
        "distinct_players": int(group["player_id"].nunique()),
        "distinct_teams": int(group["team_id"].nunique()),
        "distinct_opponents": int(group["opponent_team_id"].nunique()),
        "distinct_tournaments": int(group["tournament_name"].nunique()),
    }
    for col in (*BASE_FEATURES, *DERIVED_FEATURES):
        out[f"mean_{col}"] = _weighted_mean(group[col], w)
    out["win_rate"] = out.get("mean_game_win", math.nan)
    out["blue_side_rate"] = out.get("mean_side_blue", math.nan)
    return out


def _role_default(role_frame: pd.DataFrame, *, role: str, reference_date: pd.Timestamp) -> dict[str, Any]:
    group = role_frame[role_frame["role"] == role]
    if group.empty:
        group = role_frame
    agg = _aggregate_group(group, reference_date=reference_date, all_history_decay=True)
    agg.update({"fallback_level": "role_default", "window_days": None})
    return agg


def _apply_role_shrinkage(agg: dict[str, Any], role_default: dict[str, Any], *, prior_games: float) -> dict[str, Any]:
    """Shrink sparse champion-role aggregate means toward role defaults.

    Rare off-role champion observations can be extremely noisy (e.g. one ADC
    game on a top-lane champion).  Keeping n_games as metadata but shrinking
    mean features gives downstream models a stable vector and an explicit
    count/fallback signal.
    """

    n_games = float(agg.get("n_games") or 0.0)
    if prior_games <= 0 or not role_default:
        agg["shrinkage_weight_observed"] = 1.0
        return agg
    weight = n_games / (n_games + prior_games) if n_games > 0 else 0.0
    shrink_cols = [
        key
        for key in agg.keys()
        if key.startswith("mean_") or key in {"win_rate", "blue_side_rate"}
    ]
    for key in shrink_cols:
        observed = agg.get(key)
        default = role_default.get(key)
        if observed is None or pd.isna(observed):
            agg[key] = default
        elif default is not None and not pd.isna(default):
            agg[key] = float(weight * float(observed) + (1.0 - weight) * float(default))
    agg["shrinkage_weight_observed"] = float(weight)
    return agg


def build_embeddings(raw_dataset: Any, cfg: ChampionRoleEmbeddingConfig) -> dict[str, Any]:
    reference_date = pd.Timestamp(cfg.reference_date, tz="UTC") if cfg.reference_date else None
    loaded = raw_dataset.frame.copy()
    loaded["date"] = pd.to_datetime(loaded["date"], utc=True, errors="coerce")
    if reference_date is None:
        reference_date = loaded["date"].max() + pd.Timedelta(days=1)
    df = _prepare_frame(loaded, reference_date=reference_date)
    if df.empty:
        raise RuntimeError("No champion-role rows available before reference_date")

    role_defaults = {
        role: _aggregate_group(
            df[df["role"] == role],
            reference_date=reference_date,
            all_history_decay=True,
            decay_half_life_days=cfg.decay_half_life_days,
        )
        for role in ROLES
        if not df[df["role"] == role].empty
    }

    champion_meta = (
        df.sort_values("date")
        .groupby(["champion_key", "role"], as_index=False)
        .agg(champion_name=("champion_name", "last"))
    )

    rows: list[dict[str, Any]] = []
    for item in champion_meta.itertuples(index=False):
        champion_key = str(item.champion_key)
        role = str(item.role)
        champion_name = str(item.champion_name)
        pair = df[(df["champion_key"] == champion_key) & (df["role"] == role)]

        recent_window_days = int(cfg.windows_days[0]) if cfg.windows_days else 90
        recent_cutoff = reference_date - pd.Timedelta(days=recent_window_days)
        recent_pair = pair[pair["date"] >= recent_cutoff]
        recent_games = int(len(recent_pair))
        recent_date_max = recent_pair["date"].max().isoformat() if not recent_pair.empty else None

        selected: pd.DataFrame | None = None
        fallback = "all_history_decay"
        selected_window: int | None = None
        for window_days in cfg.windows_days:
            cutoff = reference_date - pd.Timedelta(days=int(window_days))
            candidate = pair[pair["date"] >= cutoff]
            if len(candidate) >= cfg.min_recent_games:
                selected = candidate
                fallback = f"{window_days}d"
                selected_window = int(window_days)
                break
        if selected is None:
            selected = pair

        if selected.empty:
            agg = _role_default(df, role=role, reference_date=reference_date)
        else:
            agg = _aggregate_group(
                selected,
                reference_date=reference_date,
                all_history_decay=(fallback == "all_history_decay"),
                decay_half_life_days=cfg.decay_half_life_days,
            )
            agg.update({"fallback_level": fallback, "window_days": selected_window})

        agg = _apply_role_shrinkage(agg, role_defaults.get(role, {}), prior_games=cfg.shrinkage_prior_games)
        agg.update(
            {
                "champion_id": champion_key,
                "champion_name": champion_name,
                "role": role,
                "recent_window_days": recent_window_days,
                "recent_games": recent_games,
                "recent_date_max": recent_date_max,
            }
        )
        rows.append(agg)

    emb = pd.DataFrame(rows).sort_values(["role", "champion_name", "champion_id"]).reset_index(drop=True)

    feature_cols = [
        c
        for c in emb.columns
        if c.startswith("mean_")
        or c
        in {
            "n_games",
            "age_days_mean",
            "age_days_max",
            "distinct_players",
            "distinct_teams",
            "distinct_opponents",
            "distinct_tournaments",
            "win_rate",
            "blue_side_rate",
            "shrinkage_weight_observed",
            "recent_window_days",
            "recent_games",
        }
    ]
    matrix = emb[feature_cols].replace([np.inf, -np.inf], np.nan)
    fill_values = matrix.median(numeric_only=True).fillna(0.0)
    matrix_filled = matrix.fillna(fill_values)
    scaler = StandardScaler()
    vectors = scaler.fit_transform(matrix_filled.to_numpy(dtype=float))
    vector_cols = [f"emb_{i:03d}" for i in range(vectors.shape[1])]
    vector_df = pd.DataFrame(vectors, columns=vector_cols)
    result = pd.concat([emb, vector_df], axis=1)

    diagnostics = {
        "experiment_id": cfg.experiment_id,
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "config": asdict(cfg),
        "reference_date": reference_date.isoformat(),
        "source_rows": int(len(df)),
        "source_date_min": df["date"].min().isoformat(),
        "source_date_max": df["date"].max().isoformat(),
        "champion_role_rows": int(len(result)),
        "distinct_champions": int(result["champion_id"].nunique()),
        "roles": sorted(result["role"].dropna().unique().tolist()),
        "feature_count": int(len(feature_cols)),
        "embedding_dim": int(len(vector_cols)),
        "feature_columns": feature_cols,
        "vector_columns": vector_cols,
        "fallback_counts": {str(k): int(v) for k, v in result["fallback_level"].value_counts().sort_index().items()},
        "median_games_per_pair": float(result["n_games"].median()),
        "min_games_per_pair": int(result["n_games"].min()),
        "max_games_per_pair": int(result["n_games"].max()),
        "recent_window_days": int(cfg.windows_days[0]) if cfg.windows_days else 90,
        "median_recent_games_per_pair": float(result["recent_games"].median()),
        "max_recent_games_per_pair": int(result["recent_games"].max()),
        "stale_pairs_no_recent_games": int((result["recent_games"] == 0).sum()),
        "sparse_pairs_lt_min_recent": int((result["recent_games"] < cfg.min_recent_games).sum()),
        "shrinkage_prior_games": float(cfg.shrinkage_prior_games),
        "mean_shrinkage_weight_observed": float(result["shrinkage_weight_observed"].mean()),
        "dataset_metadata": raw_dataset.metadata,
    }
    return {"embeddings": result, "diagnostics": diagnostics, "fill_values": fill_values.to_dict()}


def _write_artifact(artifact: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = artifact["embeddings"]
    try:
        embeddings.to_parquet(output_dir / "champion_role_embeddings.parquet", index=False)
        parquet_written = True
    except Exception as exc:  # optional dependency / platform fallback
        parquet_written = False
        print(f"WARN: parquet export skipped for {output_dir}: {exc}")
    embeddings.to_csv(output_dir / "champion_role_embeddings.csv", index=False)
    artifact["diagnostics"]["parquet_written"] = parquet_written
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(artifact["diagnostics"], fh, indent=2, ensure_ascii=False, default=_json_default)
    with (output_dir / "feature_fill_values.json").open("w", encoding="utf-8") as fh:
        json.dump(artifact["fill_values"], fh, indent=2, ensure_ascii=False, default=_json_default)
    return artifact["diagnostics"]


def _snapshot_dates(dataset_frame: pd.DataFrame, args: argparse.Namespace) -> list[pd.Timestamp]:
    dates = pd.to_datetime(dataset_frame["date"], utc=True, errors="coerce").dropna()
    if dates.empty:
        return []
    latest_ref = dates.max().normalize() + pd.Timedelta(days=1)
    start = pd.Timestamp(args.snapshot_start, tz="UTC") if args.snapshot_start else latest_ref - pd.DateOffset(months=18)
    end = pd.Timestamp(args.snapshot_end, tz="UTC") if args.snapshot_end else latest_ref
    start = start.normalize()
    end = end.normalize()
    if start > end:
        return []
    refs = pd.date_range(start=start, end=end, freq=args.snapshot_frequency, tz="UTC").to_list()
    if latest_ref not in refs:
        refs.append(latest_ref)
    return sorted({pd.Timestamp(ref).normalize() for ref in refs})


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def main() -> None:
    args = parse_args()
    windows_days = tuple(int(x.strip()) for x in args.windows_days.split(",") if x.strip())
    cfg = ChampionRoleEmbeddingConfig(
        min_date=args.min_date,
        max_date=args.max_date,
        reference_date=args.reference_date,
        min_recent_games=args.min_recent_games,
        windows_days=windows_days,
        decay_half_life_days=args.decay_half_life_days,
        shrinkage_prior_games=args.shrinkage_prior_games,
        output_dir=args.output_dir,
    )
    dataset = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(
            min_date=cfg.min_date,
            max_date=cfg.max_date,
            limit_rows=args.limit_rows,
            require_core_stats=True,
        )
    )
    artifact = build_embeddings(dataset, cfg)
    output_dir = Path(cfg.output_dir)
    latest_metadata = _write_artifact(artifact, output_dir)

    if args.walk_forward:
        snapshots_root = output_dir / "snapshots"
        manifest: dict[str, Any] = {
            "mode": "walk_forward_snapshots",
            "cadence": args.snapshot_frequency,
            "snapshot_start": args.snapshot_start,
            "snapshot_end": args.snapshot_end,
            "snapshots": [],
        }
        for ref_date in _snapshot_dates(dataset.frame, args):
            ref_str = ref_date.strftime("%Y-%m-%d")
            snapshot_cfg = replace(cfg, reference_date=ref_date.isoformat())
            try:
                snapshot_artifact = build_embeddings(dataset, snapshot_cfg)
            except RuntimeError as exc:
                print(f"WARN: snapshot {ref_str} skipped: {exc}")
                continue
            snapshot_dir = snapshots_root / ref_str
            snapshot_metadata = _write_artifact(snapshot_artifact, snapshot_dir)
            manifest["snapshots"].append(
                {
                    "snapshot": ref_str,
                    "reference_date": snapshot_metadata.get("reference_date"),
                    "champion_role_rows": snapshot_metadata.get("champion_role_rows"),
                    "distinct_champions": snapshot_metadata.get("distinct_champions"),
                    "source_rows": snapshot_metadata.get("source_rows"),
                    "source_date_min": snapshot_metadata.get("source_date_min"),
                    "source_date_max": snapshot_metadata.get("source_date_max"),
                    "fallback_counts": snapshot_metadata.get("fallback_counts", {}),
                }
            )
            print(f"WROTE snapshot {ref_str}: {snapshot_metadata.get('champion_role_rows')} champion-role rows")
        manifest["snapshots"].sort(key=lambda item: item["snapshot"])
        with (output_dir / "walk_forward_manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, default=_json_default)
        latest_metadata["walk_forward_manifest"] = str(output_dir / "walk_forward_manifest.json")
        latest_metadata["walk_forward_snapshot_count"] = len(manifest["snapshots"])

    print(json.dumps(latest_metadata, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
