"""Evaluate match-level model with champion-pool + team/opponent context embeddings.

EXP-058/059 diagnostic: before training a larger neural PlayerGameEncoder, test
whether the new walk-forward context artifacts add predictive signal at match
level.  Each match uses only the latest snapshot with reference_date <= match
 date, so team/champion vectors are computed from games before that snapshot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db, query_df
from betting_app.core.matching import normalize_team_name
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.ml.training.strength_dataset import StrengthDataset, StrengthDatasetConfig, build_strength_dataset, load_golgg_match_results
from betting_app.ml.training.strength_model import StrengthModelConfig, train_strength_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default="2026-01-01")
    parser.add_argument("--max-date", default=None)
    parser.add_argument("--limit-matches", type=int, default=None)
    parser.add_argument("--limit-player-rows", type=int, default=None)
    parser.add_argument("--team-artifact-dir", default="betting_app/models/ml/team_context_embeddings/exp-057")
    parser.add_argument("--champion-artifact-dir", default="betting_app/models/ml/champion_role_embeddings/exp-056")
    parser.add_argument("--champion-pool-days", type=int, default=90)
    parser.add_argument("--initial-train-size", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--step-size", type=int, default=60)
    parser.add_argument("--min-fold-train-size", type=int, default=100)
    parser.add_argument("--logistic-c", type=float, default=0.10)
    parser.add_argument("--l1-ratio", type=float, default=0.50)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "auc": float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)) if len(y) else None,
    }


def _read_manifest(root: Path) -> list[str]:
    manifest = root / "walk_forward_manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [str(item["snapshot"]) for item in payload.get("snapshots", [])]


def _snapshot_for_date(match_date: pd.Timestamp, snapshots: list[str]) -> str | None:
    if pd.isna(match_date):
        return None
    d = match_date.normalize()
    eligible = [s for s in snapshots if pd.Timestamp(s, tz="UTC") <= d]
    return eligible[-1] if eligible else None


def _load_embedding_table(root: Path, snapshot: str, filename: str) -> pd.DataFrame:
    path = root / "snapshots" / snapshot / filename
    if not path.exists():
        path = root / filename
    return pd.read_csv(path)


def _build_team_lookup(team_root: Path, snapshots: list[str]) -> tuple[dict[tuple[str, str], np.ndarray], list[str], dict[str, Any]]:
    lookup: dict[tuple[str, str], np.ndarray] = {}
    emb_cols: list[str] | None = None
    rows_per_snapshot: dict[str, int] = {}
    for snap in snapshots:
        df = _load_embedding_table(team_root, snap, "team_context_embeddings.csv")
        cols = [c for c in df.columns if c.startswith("emb_")]
        if emb_cols is None:
            emb_cols = cols
        rows_per_snapshot[snap] = int(len(df))
        for row in df.itertuples(index=False):
            team_id = str(getattr(row, "team_id"))
            lookup[(snap, team_id)] = np.array([float(getattr(row, c)) for c in cols], dtype=float)
    return lookup, emb_cols or [], {"rows_per_snapshot": rows_per_snapshot}


def _build_champion_pool_lookup(
    player_frame: pd.DataFrame,
    champion_root: Path,
    snapshots: list[str],
    *,
    pool_days: int,
) -> tuple[dict[tuple[str, str], np.ndarray], list[str], dict[str, Any]]:
    frame = player_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["team_id"] = frame["team_id"].astype(str)
    frame["champion_id"] = frame["champion_id"].astype(str)
    frame["role"] = frame["role"].astype(str).str.upper()
    lookup: dict[tuple[str, str], np.ndarray] = {}
    emb_cols: list[str] | None = None
    rows_per_snapshot: dict[str, int] = {}
    teams_per_snapshot: dict[str, int] = {}
    for snap in snapshots:
        ref = pd.Timestamp(snap, tz="UTC")
        champ = _load_embedding_table(champion_root, snap, "champion_role_embeddings.csv")
        cols = [c for c in champ.columns if c.startswith("emb_")]
        if emb_cols is None:
            emb_cols = cols
        champ_key = champ[["champion_id", "role", *cols]].copy()
        champ_key["champion_id"] = champ_key["champion_id"].astype(str)
        champ_key["role"] = champ_key["role"].astype(str).str.upper()
        recent = frame[(frame["date"] < ref) & (frame["date"] >= ref - pd.Timedelta(days=pool_days))].copy()
        if recent.empty:
            rows_per_snapshot[snap] = 0
            teams_per_snapshot[snap] = 0
            continue
        merged = recent.merge(champ_key, on=["champion_id", "role"], how="inner")
        rows_per_snapshot[snap] = int(len(merged))
        for team_id, group in merged.groupby("team_id"):
            lookup[(snap, str(team_id))] = group[cols].mean(axis=0).to_numpy(dtype=float)
        teams_per_snapshot[snap] = int(len({key[1] for key in lookup if key[0] == snap}))
    return lookup, emb_cols or [], {"rows_per_snapshot": rows_per_snapshot, "teams_per_snapshot": teams_per_snapshot}


def _add_pair_features(
    row: dict[str, Any],
    prefix: str,
    team1_vec: np.ndarray | None,
    team2_vec: np.ndarray | None,
    dim: int,
) -> None:
    if team1_vec is None:
        team1_vec = np.full(dim, np.nan)
    if team2_vec is None:
        team2_vec = np.full(dim, np.nan)
    diff = team1_vec - team2_vec
    absdiff = np.abs(diff)
    for i in range(dim):
        row[f"{prefix}_diff_{i:03d}"] = float(diff[i])
        row[f"{prefix}_absdiff_{i:03d}"] = float(absdiff[i])
    row[f"{prefix}_team1_missing"] = float(np.isnan(team1_vec).all())
    row[f"{prefix}_team2_missing"] = float(np.isnan(team2_vec).all())


def _build_context_dataset(args: argparse.Namespace) -> tuple[StrengthDataset, StrengthDataset, dict[str, Any]]:
    raw_matches = load_golgg_match_results(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_matches)
    strength = build_strength_dataset(raw_matches, StrengthDatasetConfig(min_date=args.min_date, max_date=args.max_date, limit_rows=args.limit_matches))
    if strength.frame.empty:
        raise RuntimeError("No strength dataset rows")

    raw_meta = raw_matches[["match_id", "date", "team1_id", "team2_id"]].copy()
    raw_meta["match_id"] = raw_meta["match_id"].astype(str)
    raw_meta["date"] = pd.to_datetime(raw_meta["date"], utc=True, errors="coerce")
    frame = strength.frame.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.merge(raw_meta[["match_id", "team1_id", "team2_id"]], on="match_id", how="left")
    frame["team1_id"] = frame["team1_id"].astype(str)
    frame["team2_id"] = frame["team2_id"].astype(str)

    team_root = Path(args.team_artifact_dir)
    champion_root = Path(args.champion_artifact_dir)
    snapshots = sorted(set(_read_manifest(team_root)) & set(_read_manifest(champion_root)))
    if not snapshots:
        raise RuntimeError("No common walk-forward snapshots for team/champion artifacts")
    frame["context_snapshot"] = frame["date"].apply(lambda d: _snapshot_for_date(pd.Timestamp(d), snapshots))
    frame = frame[frame["context_snapshot"].notna()].copy()

    team_lookup, team_cols, team_diag = _build_team_lookup(team_root, snapshots)
    player_ds = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(min_date="2020-01-01", max_date=args.max_date, limit_rows=args.limit_player_rows)
    )
    champ_lookup, champ_cols, champ_diag = _build_champion_pool_lookup(
        player_ds.frame,
        champion_root,
        snapshots,
        pool_days=args.champion_pool_days,
    )

    rows: list[dict[str, Any]] = []
    team_dim = len(team_cols)
    champ_dim = len(champ_cols)
    coverage = {"team1_team": 0, "team2_team": 0, "team1_champion_pool": 0, "team2_champion_pool": 0}
    for rec in frame.to_dict(orient="records"):
        snap = str(rec["context_snapshot"])
        t1 = str(rec.get("team1_id"))
        t2 = str(rec.get("team2_id"))
        out = dict(rec)
        v1 = team_lookup.get((snap, t1))
        v2 = team_lookup.get((snap, t2))
        c1 = champ_lookup.get((snap, t1))
        c2 = champ_lookup.get((snap, t2))
        coverage["team1_team"] += int(v1 is not None)
        coverage["team2_team"] += int(v2 is not None)
        coverage["team1_champion_pool"] += int(c1 is not None)
        coverage["team2_champion_pool"] += int(c2 is not None)
        _add_pair_features(out, "team_ctx", v1, v2, team_dim)
        _add_pair_features(out, "champ_pool", c1, c2, champ_dim)
        rows.append(out)

    context_frame = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    context_features = [c for c in context_frame.columns if c.startswith("team_ctx_") or c.startswith("champ_pool_")]
    context_features += ["context_snapshot_code"]
    snapshot_order = {snap: i for i, snap in enumerate(snapshots)}
    context_frame["context_snapshot_code"] = context_frame["context_snapshot"].map(snapshot_order).astype(float)
    feature_names = list(strength.feature_names) + context_features

    metadata = {
        "experiment_id": "EXP-058-059-context-eval",
        "description": "Strength features plus walk-forward team/opponent context and champion-pool context embeddings.",
        "rows": int(len(context_frame)),
        "feature_count": int(len(feature_names)),
        "base_strength_feature_count": int(len(strength.feature_names)),
        "context_feature_count": int(len(context_features)),
        "team_embedding_dim": int(team_dim),
        "champion_pool_embedding_dim": int(champ_dim),
        "snapshots": snapshots,
        "coverage": {k: float(v / max(len(context_frame), 1)) for k, v in coverage.items()},
        "team_lookup": team_diag,
        "champion_pool_lookup": champ_diag,
        "strength_metadata": strength.metadata,
        "player_dataset_metadata": player_ds.metadata,
    }
    base_frame = context_frame[[*strength.frame.columns]].copy() if all(c in context_frame.columns for c in strength.frame.columns) else strength.frame
    base = StrengthDataset(frame=base_frame, feature_names=strength.feature_names, metadata=strength.metadata)
    context = StrengthDataset(frame=context_frame, feature_names=feature_names, metadata=metadata)
    return base, context, metadata


def _train(ds: StrengthDataset, *, name: str, args: argparse.Namespace) -> Any:
    cfg = StrengthModelConfig(
        model_name=name,
        model_version="exp-058-059-oof",
        initial_train_size=args.initial_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        min_fold_train_size=args.min_fold_train_size,
        logistic_c=args.logistic_c,
        l1_ratio=args.l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=42,
        use_order_augmentation=True,
        calibrate=True,
        collect_oof=True,
    )
    return train_strength_model(ds, cfg)


def _load_market_alignment() -> pd.DataFrame:
    sql = """
    WITH latest_odds AS (
      SELECT DISTINCT ON (os.canonical_match_id, os.bookmaker_id)
        os.canonical_match_id, os.bookmaker_id, os.odds_a, os.odds_b, os.scraped_at
      FROM odds_snapshots os
      JOIN canonical_matches cm ON cm.id = os.canonical_match_id
      WHERE os.odds_a > 1 AND os.odds_b > 1
        AND os.scraped_at <= cm.start_time_normalized::timestamptz + interval '30 minutes'
      ORDER BY os.canonical_match_id, os.bookmaker_id, os.scraped_at DESC
    ), market AS (
      SELECT canonical_match_id, COUNT(*) AS books,
             AVG((1.0/odds_a) / ((1.0/odds_a) + (1.0/odds_b))) AS market_prob_a_novig,
             AVG((1.0/odds_a) + (1.0/odds_b) - 1.0) AS avg_margin
      FROM latest_odds GROUP BY canonical_match_id
    ), latest_pred AS (
      SELECT DISTINCT ON (canonical_match_id, model_name, model_version)
        canonical_match_id, model_name, model_version, prob_a, predicted_at
      FROM canonical_predictions
      WHERE (model_name, model_version) IN (
        ('Hybrid-Thesis-Market', 'a0.05-t0.80'),
        ('Hybrid-Thesis-Market', 'a0.50-t0.80'),
        ('Sym-Cal LR-ElasticNet-W20-Binomial', 'exp-039')
      )
      ORDER BY canonical_match_id, model_name, model_version, predicted_at DESC
    )
    SELECT gmm.golgg_match_id::text AS match_id, gm.team1_name AS golgg_team1, gm.team2_name AS golgg_team2,
           cm.id AS canonical_match_id, cm.team_a_name, cm.team_b_name, cm.normalized_team_a, cm.normalized_team_b,
           cm.start_time_normalized::timestamptz AS start_at, cm.winner_side,
           m.books, m.market_prob_a_novig, m.avg_margin,
           MAX(lp.prob_a) FILTER (WHERE lp.model_name='Hybrid-Thesis-Market' AND lp.model_version='a0.05-t0.80') AS hybrid_new_prob_a,
           MAX(lp.prob_a) FILTER (WHERE lp.model_name='Hybrid-Thesis-Market' AND lp.model_version='a0.50-t0.80') AS hybrid_old_prob_a,
           MAX(lp.prob_a) FILTER (WHERE lp.model_name='Sym-Cal LR-ElasticNet-W20-Binomial') AS exp039_prob_a
    FROM golgg_match_mappings gmm
    JOIN golgg_matches gm ON gm.match_id::text = gmm.golgg_match_id::text
    JOIN canonical_matches cm ON cm.id = gmm.canonical_match_id
    LEFT JOIN market m ON m.canonical_match_id = cm.id
    LEFT JOIN latest_pred lp ON lp.canonical_match_id = cm.id
    WHERE cm.status='finished' AND cm.winner_side IN ('team_a','team_b','A','B')
    GROUP BY gmm.golgg_match_id, gm.team1_name, gm.team2_name, cm.id, cm.team_a_name, cm.team_b_name,
             cm.normalized_team_a, cm.normalized_team_b, cm.start_time_normalized, cm.winner_side,
             m.books, m.market_prob_a_novig, m.avg_margin
    """
    return query_df(sql)


def _align_prob(row: pd.Series, col: str) -> float:
    p = row.get(col)
    if pd.isna(p):
        return np.nan
    g1 = normalize_team_name(str(row.get("golgg_team1") or ""))
    ca = normalize_team_name(str(row.get("team_a_name") or row.get("normalized_team_a") or ""))
    cb = normalize_team_name(str(row.get("team_b_name") or row.get("normalized_team_b") or ""))
    if g1 and ca and g1 == ca:
        return float(p)
    if g1 and cb and g1 == cb:
        return 1.0 - float(p)
    return np.nan


def _market_compare(oof: pd.DataFrame, label: str) -> dict[str, Any]:
    market = _load_market_alignment()
    merged = oof.copy()
    merged["match_id"] = merged["match_id"].astype(str)
    merged = merged.merge(market, on="match_id", how="inner")
    for src in ["market_prob_a_novig", "hybrid_new_prob_a", "hybrid_old_prob_a", "exp039_prob_a"]:
        merged[f"{src}_team1"] = merged.apply(lambda r: _align_prob(r, src), axis=1)
    out: dict[str, Any] = {
        "label": label,
        "common_mapped_rows": int(len(merged)),
        "date_min": str(pd.to_datetime(merged["date"], utc=True, errors="coerce").min()) if len(merged) else None,
        "date_max": str(pd.to_datetime(merged["date"], utc=True, errors="coerce").max()) if len(merged) else None,
        "metrics": {},
    }
    y = merged["target"].astype(int).to_numpy()
    for name, col in [
        (f"{label}_oof_calibrated", "oof_prob_calibrated"),
        (f"{label}_oof_raw", "oof_prob_raw"),
        ("market_no_vig", "market_prob_a_novig_team1"),
        ("hybrid_new_a0.05", "hybrid_new_prob_a_team1"),
        ("hybrid_old_a0.50", "hybrid_old_prob_a_team1"),
        ("exp039", "exp039_prob_a_team1"),
    ]:
        mask = merged[col].notna()
        if int(mask.sum()) > 0:
            out["metrics"][name] = _metrics(y[mask.to_numpy()], merged.loc[mask, col].to_numpy(dtype=float))
    return out


def main() -> None:
    args = parse_args()
    init_db()
    base_ds, context_ds, context_meta = _build_context_dataset(args)
    base_result = _train(base_ds, name="Strength-Baseline-LR", args=args)
    context_result = _train(context_ds, name="ContextEmbedding-LR", args=args)
    if base_result.oof_frame is None or context_result.oof_frame is None:
        raise RuntimeError("OOF collection failed")

    payload = {
        "experiment_id": "EXP-058-059",
        "description": "Walk-forward match-level evaluation of champion-pool + team/opponent context embeddings.",
        "args": vars(args),
        "context_dataset_metadata": context_meta,
        "base_strength": {
            "metrics": base_result.metrics,
            "folds": [asdict(f) for f in base_result.folds],
            "market_comparison": _market_compare(base_result.oof_frame, "base_strength"),
        },
        "context_embedding_model": {
            "metrics": context_result.metrics,
            "folds": [asdict(f) for f in context_result.folds],
            "market_comparison": _market_compare(context_result.oof_frame, "context_embedding"),
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
