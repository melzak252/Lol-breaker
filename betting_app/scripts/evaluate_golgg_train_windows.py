"""Evaluate EXP-039 thesis model with expanding vs rolling training windows.

This script uses the same full GOL.GG + OddsPortal dataset and feature pipeline
as ``betting_app.scripts.train_thesis_model``. It does not use the production
``upcoming_match_features`` table.

Usage:
    python -m betting_app.scripts.evaluate_golgg_train_windows
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

from betting_app.scripts.train_thesis_model import (
    CONTEXT_WINDOW,
    EPSILON,
    OPTUNA_BASE_FEATURES,
    RANK_PROB_FEATURES,
    ROLLING_FULL_FEATURES,
    TARGET,
    UPDATE_INTERVAL,
    add_binomial_features,
    build_logistic_regression,
    generate_rolling_features,
    load_base_data,
    logit,
)
from src.models.team_order import swap_orientation, symmetrize_binary_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"
ASSET_DIR = PROJECT_ROOT / "docs" / "assets" / "golgg_train_window_experiment"
DOCS_DIR = PROJECT_ROOT / "docs" / "04_experiments"
for directory in (REPORT_DIR, ASSET_DIR, DOCS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class WindowResult:
    label: str
    train_window_days: int | None
    n_oof: int
    folds: int
    skipped_folds: int
    min_train: int
    avg_train: float
    max_train: int
    logloss_raw: float
    logloss_calibrated: float
    auc_calibrated: float
    brier_calibrated: float
    accuracy_calibrated: float


def prepare_clean_dataset() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the same clean modelling frame as EXP-039 final training."""
    base = load_base_data()
    print(f"Base rows after GOL.GG × odds join: {len(base):,}")

    rolling = generate_rolling_features(CONTEXT_WINDOW)
    print(f"Rolling rows: {len(rolling):,}")

    data = base.merge(rolling, on="golgg_match_id", how="inner")
    data = data.sort_values("date").reset_index(drop=True)
    data, binomial_features = add_binomial_features(data)

    all_features = OPTUNA_BASE_FEATURES + ROLLING_FULL_FEATURES + binomial_features
    clean = data.dropna(subset=all_features + [TARGET]).copy()
    clean = clean[clean["date"] >= pd.Timestamp("2020-01-01")].copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    print(
        "Clean rows: "
        f"{len(clean):,}; date range: {clean['date'].min()} → {clean['date'].max()}; "
        f"features: {len(all_features)}"
    )
    return clean, all_features, binomial_features


def _predict_symmetric(model, chunk: pd.DataFrame, all_features: list[str]) -> np.ndarray:
    original_prob = np.clip(
        model.predict_proba(chunk[all_features])[:, 1],
        EPSILON,
        1.0 - EPSILON,
    )
    swapped_chunk = swap_orientation(
        chunk,
        all_features,
        RANK_PROB_FEATURES,
        np.ones(len(chunk), dtype=bool),
    )
    swapped_prob = np.clip(
        model.predict_proba(swapped_chunk[all_features])[:, 1],
        EPSILON,
        1.0 - EPSILON,
    )
    return symmetrize_binary_probabilities(original_prob, swapped_prob)


def evaluate_window(
    clean: pd.DataFrame,
    all_features: list[str],
    *,
    train_window_days: int | None,
    update_interval: int = UPDATE_INTERVAL,
    min_train_size: int = 500,
) -> tuple[WindowResult, pd.DataFrame]:
    """Walk-forward evaluation for an expanding or time-limited train window."""
    label = "full_history" if train_window_days is None else f"rolling_{train_window_days}d"
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()

    oof_probs: list[np.ndarray] = []
    oof_true: list[np.ndarray] = []
    oof_ids: list[pd.Series] = []
    train_sizes: list[int] = []
    skipped_folds = 0

    for start in tqdm(range(0, len(test_pool), update_interval), desc=label):
        chunk = test_pool.iloc[start : start + update_interval].copy()
        chunk_start = chunk["date"].min()

        train_mask = clean["date"] < chunk_start
        if train_window_days is not None:
            cutoff = chunk_start - pd.Timedelta(days=train_window_days)
            train_mask &= clean["date"] >= cutoff
        train_df = clean.loc[train_mask].copy()

        if len(train_df) < min_train_size:
            skipped_folds += 1
            continue

        model = build_logistic_regression()
        model.fit(train_df[all_features], train_df[TARGET].astype(int))
        p_sym = _predict_symmetric(model, chunk, all_features)

        oof_probs.append(p_sym)
        oof_true.append(chunk[TARGET].astype(int).to_numpy())
        oof_ids.append(chunk["golgg_match_id"].astype(str))
        train_sizes.append(len(train_df))

    if not oof_probs:
        raise RuntimeError(f"No folds evaluated for {label}; lower min_train_size.")

    oof_probs_all = np.concatenate(oof_probs)
    oof_true_all = np.concatenate(oof_true)
    match_ids = pd.concat(oof_ids, ignore_index=True)

    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    platt.fit(logit(oof_probs_all), oof_true_all)
    calibrated = np.clip(
        platt.predict_proba(logit(oof_probs_all))[:, 1],
        EPSILON,
        1.0 - EPSILON,
    )

    result = WindowResult(
        label=label,
        train_window_days=train_window_days,
        n_oof=int(len(oof_true_all)),
        folds=int(len(train_sizes)),
        skipped_folds=int(skipped_folds),
        min_train=int(min(train_sizes)),
        avg_train=float(np.mean(train_sizes)),
        max_train=int(max(train_sizes)),
        logloss_raw=float(log_loss(oof_true_all, oof_probs_all)),
        logloss_calibrated=float(log_loss(oof_true_all, calibrated)),
        auc_calibrated=float(roc_auc_score(oof_true_all, calibrated)),
        brier_calibrated=float(brier_score_loss(oof_true_all, calibrated)),
        accuracy_calibrated=float(accuracy_score(oof_true_all, calibrated >= 0.5)),
    )
    predictions = pd.DataFrame(
        {
            "label": label,
            "golgg_match_id": match_ids,
            "y_true": oof_true_all,
            "prob_raw": oof_probs_all,
            "prob_calibrated": calibrated,
        }
    )
    return result, predictions


def write_experiment_report(results: list[WindowResult], metadata: dict) -> Path:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    best = min(results, key=lambda item: item.logloss_calibrated)
    table_rows = "\n".join(
        "| {label} | {window} | {n_oof} | {folds} | {min_train} | {avg_train:.1f} | "
        "{max_train} | {ll:.6f} | {auc:.6f} | {brier:.6f} | {acc:.6f} |".format(
            label=r.label,
            window="full" if r.train_window_days is None else r.train_window_days,
            n_oof=r.n_oof,
            folds=r.folds,
            min_train=r.min_train,
            avg_train=r.avg_train,
            max_train=r.max_train,
            ll=r.logloss_calibrated,
            auc=r.auc_calibrated,
            brier=r.brier_calibrated,
            acc=r.accuracy_calibrated,
        )
        for r in results
    )
    report = f"""# EXP-045 — Rolling train window on full GOL.GG thesis dataset

> [!abstract]
> Celem eksperymentu było sprawdzenie, czy model trenowany tak jak finalny model EXP-039 (`Sym-Cal LR-ElasticNet-W20-Binomial`) powinien w walk-forward używać całej historii GOL.GG, czy tylko ograniczonego okna czasowego. W przeciwieństwie do EXP-044 eksperyment używa pełnych danych `data/golgg_y_predicts.csv` + `data/odds.csv` + `data/golgg_matches.json`, a nie produkcyjnej tabeli `upcoming_match_features`.

## Metadata

- **Experiment ID**: EXP-045
- **Date & Time**: {now}
- **Tags**: #golgg #walk-forward #rolling-window #exp-039 #thesis-model
- **Model**: `Sym-Cal LR-ElasticNet-W20-Binomial`
- **Seed**: `42`
- **Update interval**: `{metadata['update_interval']}` matches per fold
- **Context window**: W`{metadata['context_window']}` team rolling features
- **Rows after cleaning**: `{metadata['clean_rows']}`
- **Date range**: `{metadata['date_min']}` → `{metadata['date_max']}`
- **Features**: `{metadata['n_features']}`

## Setup

- Base data: `data/golgg_y_predicts.csv` joined with `data/odds.csv` on `golgg_match_id`, filtered to `date >= 2020-01-01`.
- Rolling context: W20 features generated leakage-safely from `data/golgg_matches.json`.
- Additional features: binomial best-of-series transforms for ranking probabilities.
- Walk-forward test period: `date >= 2021-01-01`.
- Compared train windows: full expanding history, 1095d, 730d, 365d.
- Prediction: original orientation + swapped orientation, combined by `symmetrize_binary_probabilities`.
- Calibration: Platt calibration on logit-transformed OOF probabilities, as in EXP-039.

## Results

| Variant | Window days | OOF n | Folds | Min train | Avg train | Max train | Cal. LogLoss | AUC | Brier | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

> [!check]
> Best calibrated LogLoss: `{best.logloss_calibrated:.6f}` for `{best.label}`.

## Interpretation

- This experiment answers the corrected question: it evaluates train-window policy on the same full GOL.GG basis as the currently used thesis model, not on the small live feature table.
- If a rolling window wins, the likely explanation is concept drift: older LoL seasons/meta and roster structures become less representative.
- If full history wins, the additional sample size is more valuable than recency filtering for the fixed EXP-039 logistic model.

## Artifacts

- JSON summary: `reports/golgg_train_window_experiment.json`
- CSV summary: `docs/assets/golgg_train_window_experiment/results.csv`
- OOF predictions: `docs/assets/golgg_train_window_experiment/oof_predictions.csv`
"""
    path = DOCS_DIR / "EXP-045_golgg_rolling_train_window.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    clean, all_features, binomial_features = prepare_clean_dataset()
    windows = [None, 1095, 730, 365]

    results: list[WindowResult] = []
    predictions: list[pd.DataFrame] = []
    for window in windows:
        result, pred = evaluate_window(clean, all_features, train_window_days=window)
        results.append(result)
        predictions.append(pred)
        print(result)

    results_df = pd.DataFrame([asdict(r) for r in results])
    results_df.to_csv(ASSET_DIR / "results.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(ASSET_DIR / "oof_predictions.csv", index=False)

    metadata = {
        "clean_rows": int(len(clean)),
        "date_min": clean["date"].min().isoformat(),
        "date_max": clean["date"].max().isoformat(),
        "n_features": len(all_features),
        "features": all_features,
        "binomial_features": binomial_features,
        "context_window": CONTEXT_WINDOW,
        "update_interval": UPDATE_INTERVAL,
    }
    summary = {"metadata": metadata, "results": [asdict(r) for r in results]}
    (REPORT_DIR / "golgg_train_window_experiment.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report_path = write_experiment_report(results, metadata)

    print("\nSaved:")
    print(f"  {REPORT_DIR / 'golgg_train_window_experiment.json'}")
    print(f"  {ASSET_DIR / 'results.csv'}")
    print(f"  {ASSET_DIR / 'oof_predictions.csv'}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
