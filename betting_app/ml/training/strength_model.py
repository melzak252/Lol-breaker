"""EXP-047 calibrated strength model training utilities.

This module trains on the EXP-046 point-in-time strength dataset.  It keeps the
model intentionally tabular and interpretable because the main improvement over
the existing weekly retrain is the leakage-safe dataset, not model capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betting_app.ml.metrics import accuracy_from_prob, binary_log_loss, brier_score, clip_probability
from betting_app.ml.training.strength_dataset import StrengthDataset, iter_feature_rows


@dataclass(frozen=True)
class StrengthModelConfig:
    """Configuration for EXP-047."""

    model_name: str = "Strength-Calibrated-LR"
    model_version: str = "exp-047"
    initial_train_size: int = 3000
    test_size: int = 1000
    step_size: int = 1000
    min_fold_train_size: int = 500
    logistic_c: float = 0.20
    l1_ratio: float = 0.25
    max_iter: int = 1000
    tol: float = 1e-3
    random_state: int = 42
    use_order_augmentation: bool = True
    calibrate: bool = True
    collect_oof: bool = False


@dataclass(frozen=True)
class StrengthFoldResult:
    fold: int
    train_size: int
    test_size: int
    test_start_at: str
    test_end_at: str
    log_loss: float
    brier: float
    accuracy: float
    auc: float | None


@dataclass(frozen=True)
class StrengthTrainingResult:
    model: Pipeline
    calibrator: LogisticRegression | None
    feature_names: list[str]
    folds: list[StrengthFoldResult]
    metrics: dict[str, Any]
    oof_frame: pd.DataFrame | None = None


def build_strength_estimator(config: StrengthModelConfig | None = None) -> Pipeline:
    """Build the EXP-047 base estimator."""

    cfg = config or StrengthModelConfig()
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=cfg.logistic_c,
                    penalty="elasticnet",
                    l1_ratio=cfg.l1_ratio,
                    solver="saga",
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )


def train_strength_model(
    dataset: StrengthDataset,
    config: StrengthModelConfig | None = None,
) -> StrengthTrainingResult:
    """Evaluate EXP-047 with walk-forward folds and train a final model."""

    cfg = config or StrengthModelConfig()
    if dataset.frame.empty:
        raise ValueError("Cannot train EXP-047: EXP-046 dataset is empty")
    if not dataset.feature_names:
        raise ValueError("Cannot train EXP-047: no feature columns")

    frame = dataset.frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["date", "target"]).sort_values(["date", "match_id"]).reset_index(drop=True)
    if len(frame) < cfg.min_fold_train_size:
        raise ValueError(f"Need at least {cfg.min_fold_train_size} rows, got {len(frame)}")

    feature_names = list(dataset.feature_names)
    folds: list[StrengthFoldResult] = []
    oof_prob = np.full(len(frame), np.nan, dtype=float)
    elo_baseline_prob = np.full(len(frame), np.nan, dtype=float)
    starts = _fold_starts(len(frame), cfg)
    for fold_idx, train_end in enumerate(starts, start=1):
        test_end = min(train_end + cfg.test_size, len(frame))
        if test_end <= train_end:
            continue
        train_df = frame.iloc[:train_end]
        test_df = frame.iloc[train_end:test_end]
        y_train = train_df["target"].astype(int).to_numpy()
        y_test = test_df["target"].astype(int).to_numpy()
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        estimator = build_strength_estimator(cfg)
        x_train, y_train_fit = _training_matrix(train_df, feature_names, cfg)
        estimator.fit(x_train, y_train_fit)
        probs = estimator.predict_proba(test_df[feature_names].to_numpy(dtype=float))[:, 1]
        probs = np.array([clip_probability(float(p)) for p in probs])
        oof_prob[train_end:test_end] = probs
        if "elo_expected_team1" in test_df.columns:
            elo_baseline_prob[train_end:test_end] = np.array(
                [clip_probability(float(p)) for p in test_df["elo_expected_team1"].to_numpy(dtype=float)]
            )
        folds.append(
            StrengthFoldResult(
                fold=fold_idx,
                train_size=int(len(train_df)),
                test_size=int(len(test_df)),
                test_start_at=test_df["date"].min().isoformat(),
                test_end_at=test_df["date"].max().isoformat(),
                log_loss=binary_log_loss(y_test, probs),
                brier=brier_score(y_test, probs),
                accuracy=accuracy_from_prob(y_test, probs),
                auc=_safe_auc(y_test, probs),
            )
        )

    valid_oof = ~np.isnan(oof_prob)
    calibrator: LogisticRegression | None = None
    if cfg.calibrate and valid_oof.sum() >= max(200, cfg.test_size) and len(set(frame.loc[valid_oof, "target"].astype(int))) == 2:
        logits = _logit(oof_prob[valid_oof]).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=cfg.random_state)
        calibrator.fit(logits, frame.loc[valid_oof, "target"].astype(int).to_numpy())

    final_model = build_strength_estimator(cfg)
    x_all, y_all = _training_matrix(frame, feature_names, cfg)
    final_model.fit(x_all, y_all)

    raw_final_oof = oof_prob[valid_oof]
    y_oof = frame.loc[valid_oof, "target"].astype(int).to_numpy()
    calibrated_oof = _apply_calibrator(raw_final_oof, calibrator)
    valid_elo = valid_oof & ~np.isnan(elo_baseline_prob)
    y_elo = frame.loc[valid_elo, "target"].astype(int).to_numpy()
    elo_oof = elo_baseline_prob[valid_elo]
    metrics = {
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "rows": int(len(frame)),
        "feature_count": int(len(feature_names)),
        "fold_count": int(len(folds)),
        "oof_count": int(valid_oof.sum()),
        "mean_log_loss": float(np.mean([f.log_loss for f in folds])) if folds else None,
        "mean_brier": float(np.mean([f.brier for f in folds])) if folds else None,
        "mean_accuracy": float(np.mean([f.accuracy for f in folds])) if folds else None,
        "mean_auc": _mean_optional([f.auc for f in folds]),
        "oof_log_loss_raw": binary_log_loss(y_oof, raw_final_oof) if len(y_oof) else None,
        "oof_brier_raw": brier_score(y_oof, raw_final_oof) if len(y_oof) else None,
        "oof_accuracy_raw": accuracy_from_prob(y_oof, raw_final_oof) if len(y_oof) else None,
        "oof_auc_raw": _safe_auc(y_oof, raw_final_oof) if len(y_oof) else None,
        "oof_log_loss_calibrated": binary_log_loss(y_oof, calibrated_oof) if len(y_oof) else None,
        "oof_brier_calibrated": brier_score(y_oof, calibrated_oof) if len(y_oof) else None,
        "oof_accuracy_calibrated": accuracy_from_prob(y_oof, calibrated_oof) if len(y_oof) else None,
        "oof_auc_calibrated": _safe_auc(y_oof, calibrated_oof) if len(y_oof) else None,
        "elo_baseline_oof_count": int(valid_elo.sum()),
        "elo_baseline_log_loss": binary_log_loss(y_elo, elo_oof) if len(y_elo) else None,
        "elo_baseline_brier": brier_score(y_elo, elo_oof) if len(y_elo) else None,
        "elo_baseline_accuracy": accuracy_from_prob(y_elo, elo_oof) if len(y_elo) else None,
        "elo_baseline_auc": _safe_auc(y_elo, elo_oof) if len(y_elo) else None,
        "calibrated": calibrator is not None,
        "config": asdict(cfg),
    }
    oof_frame: pd.DataFrame | None = None
    if cfg.collect_oof and valid_oof.sum() > 0:
        oof_data: dict[str, Any] = {
            "match_id": frame.loc[valid_oof, "match_id"].to_numpy(),
            "date": frame.loc[valid_oof, "date"].to_numpy(),
            "target": frame.loc[valid_oof, "target"].astype(int).to_numpy(),
            "oof_prob_raw": raw_final_oof,
            "oof_prob_calibrated": calibrated_oof,
        }
        for col in ("team1_name", "team2_name"):
            if col in frame.columns:
                oof_data[col] = frame.loc[valid_oof, col].to_numpy()
        oof_frame = pd.DataFrame(oof_data)

    return StrengthTrainingResult(final_model, calibrator, feature_names, folds, metrics, oof_frame=oof_frame)


def save_strength_artifacts(
    *,
    dataset: StrengthDataset,
    training: StrengthTrainingResult,
    config: StrengthModelConfig,
    artifact_root: Path | str = Path("betting_app/models/ml"),
) -> Path:
    """Persist EXP-046 dataset snapshot and EXP-047 model bundle."""

    root = Path(artifact_root) / config.model_name / config.model_version
    root.mkdir(parents=True, exist_ok=True)

    import json

    (root / "feature_names.json").write_text(json.dumps(training.feature_names, indent=2), encoding="utf-8")
    (root / "dataset_metadata.json").write_text(json.dumps(dataset.metadata, indent=2, default=str), encoding="utf-8")
    with (root / "train_dataset.jsonl").open("w", encoding="utf-8") as fh:
        for row in iter_feature_rows(dataset):
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    bundle = {
        "model": training.model,
        "calibrator": training.calibrator,
        "feature_names": training.feature_names,
        "model_name": config.model_name,
        "model_version": config.model_version,
    }
    joblib.dump(bundle, root / "model.joblib")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "EXP-047",
        "model_name": config.model_name,
        "model_version": config.model_version,
        "metrics": training.metrics,
        "folds": [asdict(f) for f in training.folds],
        "dataset_metadata": dataset.metadata,
        "artifact_files": ["model.joblib", "metadata.json", "feature_names.json", "dataset_metadata.json", "train_dataset.jsonl"],
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return root


def _fold_starts(n_rows: int, cfg: StrengthModelConfig) -> list[int]:
    start = max(cfg.initial_train_size, cfg.min_fold_train_size)
    starts: list[int] = []
    while start < n_rows:
        starts.append(start)
        start += cfg.step_size
    return starts


def _training_matrix(frame: pd.DataFrame, feature_names: list[str], cfg: StrengthModelConfig) -> tuple[np.ndarray, np.ndarray]:
    x = frame[feature_names].to_numpy(dtype=float)
    y = frame["target"].astype(int).to_numpy()
    if not cfg.use_order_augmentation:
        return x, y
    swapped = swap_feature_frame(frame[feature_names]).to_numpy(dtype=float)
    return np.vstack([x, swapped]), np.concatenate([y, 1 - y])


def swap_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Swap team1/team2 feature orientation for order augmentation."""

    out = features.copy()
    columns = set(out.columns)
    processed: set[str] = set()
    for col in list(out.columns):
        if col in processed:
            continue
        if col.startswith("team1_"):
            other = "team2_" + col[len("team1_") :]
            if other in columns:
                left = out[col].copy()
                out[col] = out[other]
                out[other] = left
                processed.update({col, other})
        elif col.startswith("team2_"):
            other = "team1_" + col[len("team2_") :]
            if other in columns:
                left = out[col].copy()
                out[col] = out[other]
                out[other] = left
                processed.update({col, other})

    diff_columns = {"elo_diff", "prior_matches_diff", "career_win_rate_diff", "days_since_last_diff"}
    for col in out.columns:
        if col in diff_columns or "_diff" in col:
            out[col] = -out[col]
    if "elo_expected_team1" in out.columns:
        out["elo_expected_team1"] = 1.0 - out["elo_expected_team1"]
    if "h2h_team1_win_rate" in out.columns:
        out["h2h_team1_win_rate"] = 1.0 - out["h2h_team1_win_rate"]
    return out


def predict_strength_proba(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Predict calibrated EXP-047 probabilities from a saved bundle."""

    feature_names = list(bundle["feature_names"])
    raw = bundle["model"].predict_proba(frame[feature_names].to_numpy(dtype=float))[:, 1]
    return _apply_calibrator(raw, bundle.get("calibrator"))


def _apply_calibrator(probabilities: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    probs = np.array([clip_probability(float(p)) for p in probabilities], dtype=float)
    if calibrator is None:
        return probs
    return calibrator.predict_proba(_logit(probs).reshape(-1, 1))[:, 1]


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probs = np.array([clip_probability(float(p)) for p in probabilities], dtype=float)
    return np.log(probs / (1.0 - probs))


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    try:
        if len(set(np.asarray(y_true, dtype=int))) < 2:
            return None
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return None


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None
