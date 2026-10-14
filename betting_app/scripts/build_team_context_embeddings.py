"""
Builds leakage-safe team/opponent context embeddings from completed GOL.GG
player-game rows.

EXP-057 mirrors the champion-role embedding pipeline, but the unit is a team at
reference date T.  For every team, only games with date < T are used.  The first
sufficient recent window is selected (90d -> 180d -> 365d), otherwise all prior
history is used with exponential time decay.  Sparse teams are shrunk toward a
global default so that rare/renamed teams do not produce extreme vectors.
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
class TeamContextEmbeddingConfig:
    experiment_id: str = "EXP-057"
    model_name: str = "TeamContextEmbeddings"
    model_version: str = "exp-057"
    min_date: str = "2020-01-01"
    max_date: str | None = None
    reference_date: str | None = None
    min_recent_games: int = 10
    windows_days: tuple[int, ...] = (90, 180, 365)
    decay_half_life_days: float = 180.0
    shrinkage_prior_games: float = 12.0
    output_dir: str = "betting_app/models/ml/team_context_embeddings/exp-057"


TEAM_GAME_FEATURES: tuple[str, ...] = (
    "game_win",
    "side_blue",
    "game_duration_seconds",
    "team_kills",
    "opp_team_kills",
    "team_towers",
    "opp_team_towers",
    "team_dragons",
    "opp_team_dragons",
    "team_nashors",
    "opp_team_nashors",
    "team_gold",
    "opp_team_gold",
    "team_gold_diff",
    "team_kill_diff",
    "tower_diff",
    "dragon_diff",
    "nashor_diff",
    "kills_per_min",
    "deaths_per_min",
    "gold_per_min",
    "gold_diff_per_min",
    "kill_diff_per_min",
    "avg_player_kda",
    "avg_player_kp",
    "avg_player_damage_share",
    "avg_player_gold_share",
    "avg_player_cs_per_min",
    "avg_player_damage_per_min",
    "avg_player_vision_score",
    "avg_player_vspm",
    "avg_player_gd15",
    "avg_player_csd15",
    "avg_player_xpd15",
    "total_wards_placed",
    "total_wards_destroyed",
    "total_control_wards",
    "total_damage_to_champions",
    "champion_pool_size",
    "player_roster_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default=TeamContextEmbeddingConfig.min_date)
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--reference-date", default=None, help="Leakage cutoff date. Defaults to last loaded game date + 1 day.")
    parser.add_argument("--min-recent-games", type=int, default=TeamContextEmbeddingConfig.min_recent_games)
    parser.add_argument("--windows-days", default="90,180,365")
    parser.add_argument("--decay-half-life-days", type=float, default=TeamContextEmbeddingConfig.decay_half_life_days)
    parser.add_argument("--shrinkage-prior-games", type=float, default=TeamContextEmbeddingConfig.shrinkage_prior_games)
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke-test player-row limit.")
    parser.add_argument("--output-dir", default=TeamContextEmbeddingConfig.output_dir)
    parser.add_argument("--walk-forward", action="store_true", help="Also build leakage-safe historical snapshots.")
    parser.add_argument("--snapshot-start", default=None, help="First snapshot reference date, e.g. 2026-01-01. Defaults to last 18 months.")
    parser.add_argument("--snapshot-end", default=None, help="Last snapshot reference date. Defaults to latest available date + 1 day.")
    parser.add_argument("--snapshot-frequency", choices=("MS", "W-MON"), default="MS")
    return parser.parse_args()


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def _team_game_frame(frame: pd.DataFrame, *, reference_date: pd.Timestamp) -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df[df["date"].notna() & (df["date"] < reference_date)].copy()
    df = df[df["team_id"].notna() & (df["team_id"].astype(str) != "")].copy()
    if df.empty:
        return pd.DataFrame()

    for col in [
        "game_win",
        "side_blue",
        "game_duration_seconds",
        "team_kills",
        "opp_team_kills",
        "team_towers",
        "opp_team_towers",
        "team_dragons",
        "opp_team_dragons",
        "team_nashors",
        "opp_team_nashors",
        "team_gold",
        "opp_team_gold",
        "team_gold_diff",
        "team_kill_diff",
        "stat_kills",
        "stat_deaths",
        "stat_assists",
        "stat_cs",
        "stat_csm",
        "stat_total_damage_to_champion",
        "stat_dpm",
        "stat_kp%",
        "stat_dmg%",
        "stat_gold%",
        "stat_vision_score",
        "stat_vspm",
        "stat_gd@15",
        "stat_csd@15",
        "stat_xpd@15",
        "stat_wards_placed",
        "stat_wards_destroyed",
        "stat_control_wards_purchased",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    minutes = df["game_duration_seconds"] / 60.0
    df["player_kda"] = (df["stat_kills"] + df["stat_assists"]) / df["stat_deaths"].clip(lower=1)
    df["player_kp"] = _safe_div(df["stat_kills"] + df["stat_assists"], df["team_kills"])

    grouped = df.groupby(["game_id", "team_id"], dropna=False)
    team_games = grouped.agg(
        match_id=("match_id", "first"),
        date=("date", "first"),
        tournament_name=("tournament_name", "first"),
        team_name=("team_name", "last"),
        opponent_team_id=("opponent_team_id", "first"),
        side_blue=("side_blue", "first"),
        game_win=("game_win", "first"),
        game_duration_seconds=("game_duration_seconds", "first"),
        team_kills=("team_kills", "first"),
        opp_team_kills=("opp_team_kills", "first"),
        team_towers=("team_towers", "first"),
        opp_team_towers=("opp_team_towers", "first"),
        team_dragons=("team_dragons", "first"),
        opp_team_dragons=("opp_team_dragons", "first"),
        team_nashors=("team_nashors", "first"),
        opp_team_nashors=("opp_team_nashors", "first"),
        team_gold=("team_gold", "first"),
        opp_team_gold=("opp_team_gold", "first"),
        team_gold_diff=("team_gold_diff", "first"),
        team_kill_diff=("team_kill_diff", "first"),
        avg_player_kda=("player_kda", "mean"),
        avg_player_kp=("player_kp", "mean"),
        avg_player_damage_share=("stat_dmg%", "mean"),
        avg_player_gold_share=("stat_gold%", "mean"),
        avg_player_cs_per_min=("stat_csm", "mean"),
        avg_player_damage_per_min=("stat_dpm", "mean"),
        avg_player_vision_score=("stat_vision_score", "mean"),
        avg_player_vspm=("stat_vspm", "mean"),
        avg_player_gd15=("stat_gd@15", "mean"),
        avg_player_csd15=("stat_csd@15", "mean"),
        avg_player_xpd15=("stat_xpd@15", "mean"),
        total_wards_placed=("stat_wards_placed", "sum"),
        total_wards_destroyed=("stat_wards_destroyed", "sum"),
        total_control_wards=("stat_control_wards_purchased", "sum"),
        total_damage_to_champions=("stat_total_damage_to_champion", "sum"),
        champion_pool_size=("champion_id", "nunique"),
        player_roster_size=("player_id", "nunique"),
    ).reset_index()

    minutes_tg = team_games["game_duration_seconds"] / 60.0
    team_games["tower_diff"] = team_games["team_towers"] - team_games["opp_team_towers"]
    team_games["dragon_diff"] = team_games["team_dragons"] - team_games["opp_team_dragons"]
    team_games["nashor_diff"] = team_games["team_nashors"] - team_games["opp_team_nashors"]
    team_games["kills_per_min"] = _safe_div(team_games["team_kills"], minutes_tg)
    team_games["deaths_per_min"] = _safe_div(team_games["opp_team_kills"], minutes_tg)
    team_games["gold_per_min"] = _safe_div(team_games["team_gold"], minutes_tg)
    team_games["gold_diff_per_min"] = _safe_div(team_games["team_gold_diff"], minutes_tg)
    team_games["kill_diff_per_min"] = _safe_div(team_games["team_kill_diff"], minutes_tg)

    return team_games.sort_values(["date", "game_id", "team_id"]).reset_index(drop=True)


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
        "distinct_opponents": int(group["opponent_team_id"].nunique()),
        "distinct_tournaments": int(group["tournament_name"].nunique()),
        "distinct_players": int(group["player_roster_size"].sum()),
        "distinct_champions_sum": int(group["champion_pool_size"].sum()),
    }
    for col in TEAM_GAME_FEATURES:
        out[f"mean_{col}"] = _weighted_mean(group[col], w) if col in group.columns else math.nan
    out["win_rate"] = out.get("mean_game_win", math.nan)
    out["blue_side_rate"] = out.get("mean_side_blue", math.nan)
    return out


def _apply_global_shrinkage(agg: dict[str, Any], default: dict[str, Any], *, prior_games: float) -> dict[str, Any]:
    n_games = float(agg.get("n_games") or 0.0)
    if prior_games <= 0 or not default:
        agg["shrinkage_weight_observed"] = 1.0
        return agg
    weight = n_games / (n_games + prior_games) if n_games > 0 else 0.0
    for key in [k for k in agg if k.startswith("mean_") or k in {"win_rate", "blue_side_rate"}]:
        observed = agg.get(key)
        fallback = default.get(key)
        if observed is None or pd.isna(observed):
            agg[key] = fallback
        elif fallback is not None and not pd.isna(fallback):
            agg[key] = float(weight * float(observed) + (1.0 - weight) * float(fallback))
    agg["shrinkage_weight_observed"] = float(weight)
    return agg


def build_embeddings(raw_dataset: Any, cfg: TeamContextEmbeddingConfig) -> dict[str, Any]:
    loaded = raw_dataset.frame.copy()
    loaded["date"] = pd.to_datetime(loaded["date"], utc=True, errors="coerce")
    reference_date = pd.Timestamp(cfg.reference_date, tz="UTC") if cfg.reference_date else loaded["date"].max() + pd.Timedelta(days=1)
    team_games = _team_game_frame(loaded, reference_date=reference_date)
    if team_games.empty:
        raise RuntimeError("No team-game rows available before reference_date")

    global_default = _aggregate_group(
        team_games,
        reference_date=reference_date,
        all_history_decay=True,
        decay_half_life_days=cfg.decay_half_life_days,
    )

    team_meta = (
        team_games.sort_values("date")
        .groupby("team_id", as_index=False)
        .agg(team_name=("team_name", "last"), latest_date=("date", "last"))
    )

    rows: list[dict[str, Any]] = []
    recent_window_days = int(cfg.windows_days[0]) if cfg.windows_days else 90
    recent_cutoff = reference_date - pd.Timedelta(days=recent_window_days)
    for item in team_meta.itertuples(index=False):
        team_id = str(item.team_id)
        team_name = str(item.team_name)
        team = team_games[team_games["team_id"].astype(str) == team_id]
        recent = team[team["date"] >= recent_cutoff]
        recent_games = int(len(recent))
        recent_date_max = recent["date"].max().isoformat() if not recent.empty else None

        selected: pd.DataFrame | None = None
        fallback = "all_history_decay"
        selected_window: int | None = None
        for window_days in cfg.windows_days:
            cutoff = reference_date - pd.Timedelta(days=int(window_days))
            candidate = team[team["date"] >= cutoff]
            if len(candidate) >= cfg.min_recent_games:
                selected = candidate
                fallback = f"{window_days}d"
                selected_window = int(window_days)
                break
        if selected is None:
            selected = team

        agg = _aggregate_group(
            selected,
            reference_date=reference_date,
            all_history_decay=(fallback == "all_history_decay"),
            decay_half_life_days=cfg.decay_half_life_days,
        )
        agg.update({"fallback_level": fallback, "window_days": selected_window})
        agg = _apply_global_shrinkage(agg, global_default, prior_games=cfg.shrinkage_prior_games)
        agg.update(
            {
                "team_id": team_id,
                "team_name": team_name,
                "recent_window_days": recent_window_days,
                "recent_games": recent_games,
                "recent_date_max": recent_date_max,
            }
        )
        rows.append(agg)

    emb = pd.DataFrame(rows).sort_values(["team_name", "team_id"]).reset_index(drop=True)
    feature_cols = [
        c
        for c in emb.columns
        if c.startswith("mean_")
        or c
        in {
            "n_games",
            "age_days_mean",
            "age_days_max",
            "distinct_opponents",
            "distinct_tournaments",
            "distinct_players",
            "distinct_champions_sum",
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
        "source_player_rows": int(len(loaded[loaded["date"].notna() & (loaded["date"] < reference_date)])),
        "source_team_games": int(len(team_games)),
        "source_date_min": team_games["date"].min().isoformat(),
        "source_date_max": team_games["date"].max().isoformat(),
        "team_rows": int(len(result)),
        "feature_count": int(len(feature_cols)),
        "embedding_dim": int(len(vector_cols)),
        "feature_columns": feature_cols,
        "vector_columns": vector_cols,
        "fallback_counts": {str(k): int(v) for k, v in result["fallback_level"].value_counts().sort_index().items()},
        "median_games_per_team": float(result["n_games"].median()),
        "min_games_per_team": int(result["n_games"].min()),
        "max_games_per_team": int(result["n_games"].max()),
        "recent_window_days": recent_window_days,
        "median_recent_games_per_team": float(result["recent_games"].median()),
        "max_recent_games_per_team": int(result["recent_games"].max()),
        "stale_teams_no_recent_games": int((result["recent_games"] == 0).sum()),
        "sparse_teams_lt_min_recent": int((result["recent_games"] < cfg.min_recent_games).sum()),
        "shrinkage_prior_games": float(cfg.shrinkage_prior_games),
        "mean_shrinkage_weight_observed": float(result["shrinkage_weight_observed"].mean()),
        "dataset_metadata": raw_dataset.metadata,
    }
    return {"embeddings": result, "diagnostics": diagnostics, "fill_values": fill_values.to_dict()}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _write_artifact(artifact: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = artifact["embeddings"]
    try:
        embeddings.to_parquet(output_dir / "team_context_embeddings.parquet", index=False)
        parquet_written = True
    except Exception as exc:
        parquet_written = False
        print(f"WARN: parquet export skipped for {output_dir}: {exc}")
    embeddings.to_csv(output_dir / "team_context_embeddings.csv", index=False)
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


def main() -> None:
    args = parse_args()
    windows_days = tuple(int(x.strip()) for x in args.windows_days.split(",") if x.strip())
    cfg = TeamContextEmbeddingConfig(
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
                    "team_rows": snapshot_metadata.get("team_rows"),
                    "source_team_games": snapshot_metadata.get("source_team_games"),
                    "source_date_min": snapshot_metadata.get("source_date_min"),
                    "source_date_max": snapshot_metadata.get("source_date_max"),
                    "fallback_counts": snapshot_metadata.get("fallback_counts", {}),
                }
            )
            print(f"WROTE snapshot {ref_str}: {snapshot_metadata.get('team_rows')} team rows")
        manifest["snapshots"].sort(key=lambda item: item["snapshot"])
        with (output_dir / "walk_forward_manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False, default=_json_default)
        latest_metadata["walk_forward_manifest"] = str(output_dir / "walk_forward_manifest.json")
        latest_metadata["walk_forward_snapshot_count"] = len(manifest["snapshots"])

    print(json.dumps(latest_metadata, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
