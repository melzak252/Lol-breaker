"""EXP-063: EXP-039-style model with embedding context replacing W20 averages.

This is a read-only evaluation script.  It keeps the EXP-039 rating-probability
and binomial series features, but compares three leakage-safe feature sets on
the same chronological rows:

* ``exp039_w20``: original 46 EXP-039 features.
* ``ratings_binomial``: EXP-039 rating/binomial block only, no rolling W20.
* ``ratings_context``: EXP-039 rating/binomial block plus walk-forward
  team-context and champion-pool embedding features.

The context snapshots are selected with ``reference_date <= match_date`` so a
match never sees future games.  The goal is to answer whether embedding context
captures player/team form better than simple W20 averages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from betting_app.core.db import init_db
from betting_app.ml.training.player_game_dataset import PlayerGameDatasetConfig, build_player_game_dataset_from_db
from betting_app.scripts.evaluate_context_embedding_model import (
    _build_champion_pool_lookup,
    _build_team_lookup,
    _load_market_alignment,
    _read_manifest,
    _snapshot_for_date,
)
from betting_app.ml.pipelines.exp039_weekly_retrain import build_exp039_training_frame
from betting_app.scripts.train_thesis_model import build_logistic_regression
from betting_app.services.thesis_inference_service import (
    ALL_FEATURES,
    BINOMIAL_FEATURES,
    EPSILON,
    OPTUNA_BASE_FEATURES,
    RANK_PROB_FEATURES,
    ROLLING_FULL_FEATURES,
    _logit,
    _symmetrize,
)


PAIR_SWAP_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_min1", "_min2"),
    ("_max1", "_max2"),
    ("_avg1", "_avg2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-date", default="2020-01-01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--initial-train-before", default="2026-03-01")
    parser.add_argument("--update-interval", type=int, default=500)
    parser.add_argument("--team-artifact-dir", default="betting_app/models/ml/team_context_embeddings/exp-057")
    parser.add_argument("--champion-artifact-dir", default="betting_app/models/ml/champion_role_embeddings/exp-056")
    parser.add_argument("--champion-pool-days", type=int, default=90)
    parser.add_argument("--limit-player-rows", type=int, default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), EPSILON, 1.0 - EPSILON)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)) if len(y) else None,
        "mean_prob": float(np.mean(p)) if len(y) else None,
        "target_rate": float(np.mean(y)) if len(y) else None,
    }
    try:
        out["auc"] = float(roc_auc_score(y, p)) if len(set(y)) >= 2 else None
    except Exception:
        out["auc"] = None
    return out


def _swap_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Swap team orientation for EXP-039+context features."""
    out = features.copy()
    cols = set(out.columns)

    # Probability-like rating features and binomial series invert under side swap.
    for col in [*RANK_PROB_FEATURES, *BINOMIAL_FEATURES]:
        if col in out.columns:
            out[col] = 1.0 - out[col]

    # Uncertainty/summary pairs use suffix 1/2, not t1/t2.
    for left_suffix, right_suffix in PAIR_SWAP_SUFFIXES:
        for col in list(out.columns):
            if not col.endswith(left_suffix):
                continue
            other = col[: -len(left_suffix)] + right_suffix
            if other in cols:
                tmp = out[col].copy()
                out[col] = out[other]
                out[other] = tmp

    # Original W20 team columns.
    for col in list(out.columns):
        if not col.startswith("t1_"):
            continue
        other = "t2_" + col[len("t1_") :]
        if other in cols:
            tmp = out[col].copy()
            out[col] = out[other]
            out[other] = tmp

    # Context vectors are represented as team1-team2 diff and absdiff.
    for col in out.columns:
        if "_diff_" in col and "_absdiff_" not in col:
            out[col] = -out[col]

    # Missing flags should swap side.
    for prefix in ("team_ctx", "champ_pool"):
        left = f"{prefix}_team1_missing"
        right = f"{prefix}_team2_missing"
        if left in cols and right in cols:
            tmp = out[left].copy()
            out[left] = out[right]
            out[right] = tmp
    return out


def _predict_symmetric(model: Any, frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    x = frame[feature_names]
    original = np.clip(model.predict_proba(x)[:, 1], EPSILON, 1.0 - EPSILON)
    swapped = _swap_feature_frame(x)
    swapped_prob = np.clip(model.predict_proba(swapped)[:, 1], EPSILON, 1.0 - EPSILON)
    return np.array([_symmetrize(o, s) for o, s in zip(original, swapped_prob)], dtype=float)


def _fit_augmented(train: pd.DataFrame, feature_names: list[str]) -> Any:
    model = build_logistic_regression()
    x = train[feature_names]
    x_swap = _swap_feature_frame(x)
    y = train["y_true"].astype(int).to_numpy()
    x_fit = pd.concat([x, x_swap], ignore_index=True)
    y_fit = np.concatenate([y, 1 - y])
    model.fit(x_fit, y_fit)
    return model


def _train_oof(frame: pd.DataFrame, feature_names: list[str], *, initial_train_before: str, update_interval: int) -> dict[str, Any]:
    # Do not drop rows with missing context vectors.  Missingness is expected for
    # sparse teams/champion pools and is represented both by *_missing flags and
    # by NaNs handled by the model pipeline's median imputer.  Dropping feature
    # NaNs would bias the comparison toward only well-covered teams.
    data = frame.dropna(subset=["date", "y_true"]).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    cutoff = pd.Timestamp(initial_train_before)
    train_df = data[data["date"] < cutoff].copy()
    test_pool = data[data["date"] >= cutoff].copy()
    if len(train_df) < 100 or len(test_pool) < 20:
        raise ValueError(f"Not enough rows after context filter: train={len(train_df)} test={len(test_pool)} cutoff={initial_train_before}")

    oof_rows: list[pd.DataFrame] = []
    oof_probs: list[np.ndarray] = []
    oof_y: list[np.ndarray] = []
    folds: list[dict[str, Any]] = []

    for start in range(0, len(test_pool), update_interval):
        chunk = test_pool.iloc[start : start + update_interval].copy()
        if chunk.empty:
            continue
        model = _fit_augmented(train_df, feature_names)
        p = _predict_symmetric(model, chunk, feature_names)
        y = chunk["y_true"].astype(int).to_numpy()
        oof_probs.append(p)
        oof_y.append(y)
        part = chunk[["golgg_match_id", "date", "team1_id", "team2_id", "team1_name", "team2_name", "y_true"]].copy()
        part["oof_prob_raw"] = p
        oof_rows.append(part)
        folds.append({
            "train_size": int(len(train_df)),
            "test_size": int(len(chunk)),
            "start_date": str(chunk["date"].min().date()),
            "end_date": str(chunk["date"].max().date()),
            **_safe_metrics(y, p),
        })
        train_df = pd.concat([train_df, chunk], ignore_index=True)

    raw = np.concatenate(oof_probs)
    y_all = np.concatenate(oof_y)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    calibrator.fit(_logit(raw), y_all)
    calibrated = np.clip(calibrator.predict_proba(_logit(raw))[:, 1], EPSILON, 1.0 - EPSILON)
    oof_frame = pd.concat(oof_rows, ignore_index=True)
    oof_frame["oof_prob_calibrated"] = calibrated
    return {
        "rows_available": int(len(data)),
        "train_rows_initial": int((data["date"] < cutoff).sum()),
        "oof_count": int(len(y_all)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "oof_raw": _safe_metrics(y_all, raw),
        "oof_calibrated": _safe_metrics(y_all, calibrated),
        "folds": folds,
        "oof_frame": oof_frame,
    }


def _add_pair_features(row: dict[str, Any], prefix: str, v1: np.ndarray | None, v2: np.ndarray | None, dim: int) -> None:
    if v1 is None:
        v1 = np.full(dim, np.nan)
    if v2 is None:
        v2 = np.full(dim, np.nan)
    diff = v1 - v2
    absdiff = np.abs(diff)
    for i in range(dim):
        row[f"{prefix}_diff_{i:03d}"] = float(diff[i])
        row[f"{prefix}_absdiff_{i:03d}"] = float(absdiff[i])
    row[f"{prefix}_team1_missing"] = float(np.isnan(v1).all())
    row[f"{prefix}_team2_missing"] = float(np.isnan(v2).all())


def _attach_context(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    team_root = Path(args.team_artifact_dir)
    champion_root = Path(args.champion_artifact_dir)
    snapshots = sorted(set(_read_manifest(team_root)) & set(_read_manifest(champion_root)))
    if not snapshots:
        raise RuntimeError(f"No common walk-forward snapshots in {team_root} and {champion_root}")

    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].dt.tz is not None:
        out["date"] = out["date"].dt.tz_convert(None)
    out["context_snapshot"] = out["date"].apply(lambda d: _snapshot_for_date(pd.Timestamp(d, tz="UTC"), snapshots))
    out = out[out["context_snapshot"].notna()].copy()

    team_lookup, team_cols, team_diag = _build_team_lookup(team_root, snapshots)
    player_ds = build_player_game_dataset_from_db(
        PlayerGameDatasetConfig(min_date=args.min_date, limit_rows=args.limit_player_rows)
    )
    champ_lookup, champ_cols, champ_diag = _build_champion_pool_lookup(
        player_ds.frame,
        champion_root,
        snapshots,
        pool_days=args.champion_pool_days,
    )
    team_dim = len(team_cols)
    champ_dim = len(champ_cols)

    rows: list[dict[str, Any]] = []
    coverage = {"team1_team": 0, "team2_team": 0, "team1_champion_pool": 0, "team2_champion_pool": 0}
    for rec in out.to_dict(orient="records"):
        snap = str(rec["context_snapshot"])
        t1 = str(rec["team1_id"])
        t2 = str(rec["team2_id"])
        row = dict(rec)
        tv1 = team_lookup.get((snap, t1))
        tv2 = team_lookup.get((snap, t2))
        cv1 = champ_lookup.get((snap, t1))
        cv2 = champ_lookup.get((snap, t2))
        coverage["team1_team"] += int(tv1 is not None)
        coverage["team2_team"] += int(tv2 is not None)
        coverage["team1_champion_pool"] += int(cv1 is not None)
        coverage["team2_champion_pool"] += int(cv2 is not None)
        _add_pair_features(row, "team_ctx", tv1, tv2, team_dim)
        _add_pair_features(row, "champ_pool", cv1, cv2, champ_dim)
        rows.append(row)
    if not rows:
        raise RuntimeError(
            "No rows matched the available context snapshots. "
            "Use a min-date/cutoff covered by walk-forward embedding snapshots."
        )
    ctx = pd.DataFrame(rows).sort_values(["date", "golgg_match_id"]).reset_index(drop=True)
    context_features = [c for c in ctx.columns if c.startswith("team_ctx_") or c.startswith("champ_pool_")]
    metadata = {
        "snapshots": snapshots,
        "team_embedding_dim": int(team_dim),
        "champion_pool_embedding_dim": int(champ_dim),
        "coverage": {k: float(v / max(len(ctx), 1)) for k, v in coverage.items()},
        "team_lookup": team_diag,
        "champion_pool_lookup": champ_diag,
    }
    return ctx, context_features, metadata


def _market_compare(oof_frame: pd.DataFrame, label: str) -> dict[str, Any]:
    market = _load_market_alignment()
    if market.empty or oof_frame.empty:
        return {"label": label, "common_mapped_rows": 0, "metrics": {}}
    merged = oof_frame.copy()
    merged["match_id"] = merged["golgg_match_id"].astype(str)
    merged = merged.merge(market, on="match_id", how="inner")
    if merged.empty:
        return {"label": label, "common_mapped_rows": 0, "metrics": {}}
    y = merged["y_true"].astype(int).to_numpy()
    out: dict[str, Any] = {
        "label": label,
        "common_mapped_rows": int(len(merged)),
        "date_min": str(pd.to_datetime(merged["date"], errors="coerce").min()),
        "date_max": str(pd.to_datetime(merged["date"], errors="coerce").max()),
        "metrics": {
            f"{label}_oof_calibrated": _safe_metrics(y, merged["oof_prob_calibrated"].to_numpy(dtype=float)),
            f"{label}_oof_raw": _safe_metrics(y, merged["oof_prob_raw"].to_numpy(dtype=float)),
        },
    }
    # This helper's market probability is aligned to GOL.GG team1 already by prior script conventions only if
    # canonical team A matches GOL.GG team1.  To avoid silently wrong orientation, report market only when present
    # but leave production-quality open/mid/close market comparison to EXP-060/061 scripts.
    return out


def main() -> None:
    args = parse_args()
    init_db()

    base = build_exp039_training_frame(min_date=args.min_date, limit=args.limit)
    ctx, context_features, context_meta = _attach_context(base, args)

    ratings_binomial_features = [*OPTUNA_BASE_FEATURES, *BINOMIAL_FEATURES]
    exp039_w20_features = list(ALL_FEATURES)
    ratings_context_features = [*OPTUNA_BASE_FEATURES, *BINOMIAL_FEATURES, *context_features]

    results: dict[str, Any] = {
        "experiment_id": "EXP-063",
        "description": "EXP-039-style ElasticNet LR where W20 rolling averages are replaced by walk-forward team/champion embedding context.",
        "args": vars(args),
        "dataset": {
            "base_rows": int(len(base)),
            "context_rows": int(len(ctx)),
            "date_min": str(ctx["date"].min()) if len(ctx) else None,
            "date_max": str(ctx["date"].max()) if len(ctx) else None,
        },
        "context_metadata": context_meta,
        "models": {},
    }

    for label, features in [
        ("exp039_w20", exp039_w20_features),
        ("ratings_binomial", ratings_binomial_features),
        ("ratings_context", ratings_context_features),
    ]:
        res = _train_oof(ctx, features, initial_train_before=args.initial_train_before, update_interval=args.update_interval)
        oof = res.pop("oof_frame")
        results["models"][label] = res
        results["models"][label]["market_common"] = _market_compare(oof, label)

    text = json.dumps(results, indent=2, sort_keys=True, default=str)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
