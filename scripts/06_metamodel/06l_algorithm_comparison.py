"""Compare metamodel algorithm families in walk-forward validation.

The experiment is restricted to the odds-mapped sample used in the thesis. It
uses the same player-rating and W50 rolling-context features as the final
metamodel search, but changes the supervised learning algorithm. The goal is to
justify the final choice of LightGBM against simpler and alternative gradient
boosting baselines.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "metamodel_algorithm_comparison"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42


class ProbabilisticClassifier(Protocol):
    """Protocol for binary classifiers exposing class probabilities."""

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Fit the classifier on a training frame."""

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for a test frame."""


def load_best_config_module() -> object:
    """Load helper functions from the best-configuration script.

    Returns:
        Imported module object containing data loading and feature definitions.
    """

    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def build_models(lightgbm_params: dict[str, object]) -> dict[str, ProbabilisticClassifier]:
    """Construct compared metamodel algorithms.

    Args:
        lightgbm_params: Optimized parameters from the current Optuna run.

    Returns:
        Mapping from readable model name to estimator.
    """

    return {
        "LR": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=80,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(**lightgbm_params),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.70,
            colsample_bytree=0.90,
            reg_alpha=0.20,
            reg_lambda=1.00,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=250,
            depth=3,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def walk_forward_predict(
    data: pd.DataFrame,
    features: list[str],
    model_name: str,
    model: ProbabilisticClassifier,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Generate walk-forward probabilities for one algorithm.

    Args:
        data: Chronologically sorted modeling frame.
        features: Feature column names.
        model_name: Readable name for progress reporting.
        model: Classifier exposing ``fit`` and ``predict_proba``.

    Returns:
        Tuple of labels, probabilities, and prediction dates.
    """

    clean = data.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if initial_train.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced empty train or test set.")

    train_df = initial_train.copy()
    y_true_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    date_parts: list[pd.Series] = []

    steps = range(0, len(test_pool), UPDATE_INTERVAL)
    for start in tqdm(steps, desc=f"Walk-forward {model_name}"):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL]
        model.fit(train_df[features], train_df[TARGET].astype(int))
        probabilities = model.predict_proba(test_chunk[features])[:, 1]
        y_true_parts.append(test_chunk[TARGET].astype(int).to_numpy())
        prob_parts.append(probabilities)
        date_parts.append(test_chunk["date"])
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    return np.concatenate(y_true_parts), np.concatenate(prob_parts), pd.concat(date_parts, ignore_index=True)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dates: pd.Series,
    n_features: int,
) -> dict[str, object]:
    """Evaluate one model's probability vector.

    Args:
        model_name: Readable algorithm name.
        y_true: Binary ground-truth labels.
        y_prob: Positive-class probabilities.
        dates: Prediction dates.
        n_features: Number of input features.

    Returns:
        Metric row ready for a DataFrame.
    """

    clipped = np.clip(y_prob, 0.001, 0.999)
    return {
        "model": model_name,
        "sample_size": len(y_true),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "n_features": n_features,
        "auc": roc_auc_score(y_true, clipped),
        "logloss": log_loss(y_true, clipped),
        "brier": brier_score_loss(y_true, clipped),
        "ece": calculate_ece(y_true, clipped),
        "accuracy_0_5": accuracy_score(y_true, clipped >= 0.5),
    }


def save_metric_plot(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a thesis-friendly bar chart for one metric.

    Args:
        results: Evaluation table.
        metric: Metric column to visualize.
        output_path: Destination image path.
    """

    ascending = metric in {"logloss", "brier", "ece"}
    data = results.sort_values(metric, ascending=ascending)
    palette = ["#6B7A8F", "#A68A64", "#6F8F72", "#8C6D8C", "#8A8A8A"]
    plt.figure(figsize=(10.5, 6.2))
    ax = sns.barplot(data=data, x="model", y=metric, palette=palette, hue="model", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=11, padding=4)
    values = data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.30, 0.0015)
    if metric in {"logloss", "auc", "brier"}:
        ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_title(f"Porównanie algorytmów metamodelu — {metric.upper()}", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=15)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the algorithm-family comparison experiment."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    helper = load_best_config_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    models = build_models(helper.optuna_params())

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, model in models.items():
        y_true, y_prob, dates = walk_forward_predict(data, features, model_name, model)
        rows.append(evaluate_predictions(model_name, y_true, y_prob, dates, len(features)))
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "date": dates,
                    "y_true": y_true,
                    "y_prob": np.clip(y_prob, 0.001, 0.999),
                }
            )
        )

    results = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(ASSETS_DIR / "metamodel_algorithm_comparison_metrics.csv", index=False)
    predictions.to_csv(ASSETS_DIR / "metamodel_algorithm_comparison_predictions.csv", index=False)
    save_metric_plot(results, "logloss", ASSETS_DIR / "metamodel_algorithm_logloss.png")
    save_metric_plot(results, "auc", ASSETS_DIR / "metamodel_algorithm_auc.png")
    save_metric_plot(results, "ece", ASSETS_DIR / "metamodel_algorithm_ece.png")

    print("\n=== METAMODEL ALGORITHM COMPARISON ===")
    print(results.to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
