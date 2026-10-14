"""EXP-053: walk-forward market/meta-model evaluation on production DB.

The goal is to check whether deployed model probabilities add predictive value
over bookmaker no-vig probabilities on finished canonical matches.

It evaluates three families in chronological walk-forward folds:

* ``market_novig``: average latest bookmaker no-vig probability.
* ``static_blend_*``: train-fold-selected blend between market and one model.
* ``meta_logistic``: logistic regression on market + deployed model logits and
  disagreement/coverage features.

This script is read-only: it does not write predictions to the database.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from betting_app.core.db import get_session, init_db


EPS = 1e-6
MODEL_KEYS = {
    "hybrid": ("Hybrid-Thesis-Market", "a0.50-t0.80"),
    "exp039": ("Sym-Cal LR-ElasticNet-W20-Binomial", "exp-039"),
}


@dataclass(frozen=True)
class MetricRow:
    name: str
    n: int
    log_loss: float
    brier: float
    auc: float | None
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-back", type=int, default=365)
    parser.add_argument("--initial-train", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def _logit(values: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> MetricRow:
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y_true, dtype=int)
    auc: float | None
    try:
        auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) >= 2 else None
    except ValueError:
        auc = None
    return MetricRow(
        name=name,
        n=int(len(y)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        auc=auc,
        accuracy=float(accuracy_score(y, p >= 0.5)),
    )


def load_frame(days_back: int) -> pd.DataFrame:
    init_db()
    session = get_session()
    query = text(
        """
        WITH finished AS (
          SELECT id,
                 team_a_name,
                 team_b_name,
                 league,
                 start_time_normalized::timestamptz AS start_at,
                 CASE
                   WHEN winner_side = 'team_a' THEN 1
                   WHEN winner_side = 'team_b' THEN 0
                   ELSE NULL
                 END AS y
          FROM canonical_matches
          WHERE status = 'finished'
            AND winner_side IN ('team_a', 'team_b')
            AND start_time_normalized::timestamptz >= now() - (:days_back || ' days')::interval
        ), pred_ranked AS (
          SELECT cp.*,
                 row_number() OVER (
                   PARTITION BY cp.canonical_match_id, cp.model_name, cp.model_version
                   ORDER BY cp.predicted_at DESC NULLS LAST, cp.id DESC
                 ) AS rn
          FROM canonical_predictions cp
          JOIN finished f ON f.id = cp.canonical_match_id
        ), latest_book_odds AS (
          SELECT os.canonical_match_id,
                 os.bookmaker_id,
                 os.odds_a,
                 os.odds_b,
                 row_number() OVER (
                   PARTITION BY os.canonical_match_id, os.bookmaker_id
                   ORDER BY os.scraped_at DESC, os.id DESC
                 ) AS rn
          FROM odds_snapshots os
          JOIN finished f ON f.id = os.canonical_match_id
          WHERE os.market_type = 'match_winner'
            AND os.odds_a > 1
            AND os.odds_b > 1
            AND os.scraped_at <= f.start_at + interval '30 minutes'
        ), market AS (
          SELECT canonical_match_id,
                 count(*) AS books,
                 avg((1.0 / odds_a) / ((1.0 / odds_a) + (1.0 / odds_b)))::float AS p_market,
                 avg((1.0 / odds_a) + (1.0 / odds_b) - 1.0)::float AS margin
          FROM latest_book_odds
          WHERE rn = 1
          GROUP BY canonical_match_id
        )
        SELECT f.id,
               f.team_a_name,
               f.team_b_name,
               f.league,
               f.start_at,
               f.y,
               m.books,
               m.p_market,
               m.margin,
               p.model_name,
               p.model_version,
               p.prob_a::float AS p_model
        FROM finished f
        JOIN market m ON m.canonical_match_id = f.id
        JOIN pred_ranked p ON p.canonical_match_id = f.id AND p.rn = 1
        ORDER BY f.start_at, f.id, p.model_name, p.model_version
        """
    )
    raw = pd.read_sql(query, session.bind, params={"days_back": days_back})
    session.close()
    if raw.empty:
        return pd.DataFrame()

    base_cols = ["id", "team_a_name", "team_b_name", "league", "start_at", "y", "books", "p_market", "margin"]
    frame = raw[base_cols].drop_duplicates("id").copy()
    for key, (model_name, version) in MODEL_KEYS.items():
        part = raw[(raw["model_name"] == model_name) & (raw["model_version"] == version)][["id", "p_model"]]
        frame = frame.merge(part.rename(columns={"p_model": f"p_{key}"}), on="id", how="left")

    frame = frame.sort_values(["start_at", "id"]).reset_index(drop=True)
    return frame


def add_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    out["logit_market"] = _logit(out["p_market"])
    features = ["logit_market", "books", "margin"]
    for key in MODEL_KEYS:
        col = f"p_{key}"
        if col not in out.columns:
            out[col] = np.nan
        out[f"has_{key}"] = out[col].notna().astype(float)
        out[f"logit_{key}"] = np.where(out[col].notna(), _logit(out[col].fillna(0.5)), 0.0)
        out[f"abs_market_{key}"] = (out[col] - out["p_market"]).abs().fillna(0.0)
        features.extend([f"has_{key}", f"logit_{key}", f"abs_market_{key}"])
    return out, features


def best_static_blend(train: pd.DataFrame, model_col: str) -> float:
    valid = train[["y", "p_market", model_col]].dropna()
    if len(valid) < 50:
        return 1.0
    best_alpha = 1.0
    best_loss = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        p = alpha * valid["p_market"].to_numpy() + (1.0 - alpha) * valid[model_col].to_numpy()
        loss = log_loss(valid["y"].astype(int), np.clip(p, EPS, 1.0 - EPS), labels=[0, 1])
        if loss < best_loss:
            best_loss = float(loss)
            best_alpha = float(alpha)
    return best_alpha


def walk_forward(frame: pd.DataFrame, features: list[str], initial_train: int, test_size: int) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for train_end in range(initial_train, len(frame), test_size):
        test_end = min(train_end + test_size, len(frame))
        train = frame.iloc[:train_end].copy()
        test = frame.iloc[train_end:test_end].copy()
        if len(test) == 0 or train["y"].nunique() < 2:
            continue

        estimator = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.25, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
            ]
        )
        estimator.fit(train[features], train["y"].astype(int))

        chunk = test[["id", "start_at", "team_a_name", "team_b_name", "league", "y", "p_market"]].copy()
        chunk["p_meta_logistic"] = estimator.predict_proba(test[features])[:, 1]
        for key in MODEL_KEYS:
            model_col = f"p_{key}"
            alpha = best_static_blend(train, model_col)
            chunk[f"alpha_market_{key}"] = alpha
            chunk[f"p_blend_{key}"] = np.where(
                test[model_col].notna(),
                alpha * test["p_market"] + (1.0 - alpha) * test[model_col],
                test["p_market"],
            )
        predictions.append(chunk)
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def main() -> None:
    args = parse_args()
    frame = load_frame(args.days_back)
    if frame.empty:
        raise SystemExit("No finished matches with market odds and predictions found.")
    frame, features = add_features(frame)
    preds = walk_forward(frame, features, args.initial_train, args.test_size)
    if preds.empty:
        raise SystemExit("No walk-forward predictions produced; lower --initial-train.")

    rows = [_metrics("market_novig", preds["y"].to_numpy(), preds["p_market"].to_numpy())]
    rows.append(_metrics("meta_logistic", preds["y"].to_numpy(), preds["p_meta_logistic"].to_numpy()))
    for key in MODEL_KEYS:
        rows.append(_metrics(f"static_blend_{key}", preds["y"].to_numpy(), preds[f"p_blend_{key}"].to_numpy()))

    payload: dict[str, Any] = {
        "experiment_id": "EXP-053",
        "days_back": args.days_back,
        "initial_train": args.initial_train,
        "test_size": args.test_size,
        "input_matches": int(len(frame)),
        "evaluated_matches": int(len(preds)),
        "date_min": str(frame["start_at"].min()),
        "date_max": str(frame["start_at"].max()),
        "feature_names": features,
        "metrics": [asdict(row) for row in sorted(rows, key=lambda item: item.log_loss)],
        "mean_selected_alpha": {
            key: float(preds[f"alpha_market_{key}"].mean()) for key in MODEL_KEYS
        },
    }

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
