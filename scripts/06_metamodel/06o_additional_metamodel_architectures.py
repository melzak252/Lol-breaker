"""Compare additional metamodel architectures before final model selection.

The experiment intentionally excludes attention/Transformer-style sequence
aggregation. It focuses on classical tabular alternatives that are easy to
justify in an engineering thesis: ElasticNet logistic regression, small neural
networks, histogram gradient boosting and ExtraTrees. All models use the same
odds-mapped walk-forward sample and the same full W50 feature set as the current
best logistic-regression experiment.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "metamodel_additional_architectures"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42


class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers returning binary class probabilities."""

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Fit the estimator on a training frame."""

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for a test frame."""


def load_best_config_module() -> object:
    """Load data and feature helpers from the best-configuration script.

    Returns:
        Imported helper module.
    """

    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def build_models() -> dict[str, ProbabilisticClassifier]:
    """Construct additional tabular metamodel architectures.

    Returns:
        Mapping from readable model name to estimator.
    """

    return {
        "LR-EN C=0.3 α=0.25": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.25,
                        C=0.3,
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "LR-EN C=0.3 α=0.50": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.50,
                        C=0.3,
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "LR-EN C=0.1 α=0.50": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        l1_ratio=0.50,
                        C=0.1,
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "MLP shallow 8": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(8,),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "MLP small 16": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16,),
                        activation="relu",
                        alpha=0.01,
                        learning_rate_init=0.001,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "MLP medium 32-16": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        alpha=0.02,
                        learning_rate_init=0.001,
                        max_iter=250,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=15,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.035,
                        max_leaf_nodes=8,
                        min_samples_leaf=80,
                        l2_regularization=0.10,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=6,
                        min_samples_leaf=80,
                        max_features="sqrt",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def prepare_modeling_data() -> tuple[pd.DataFrame, list[str]]:
    """Load odds-mapped data and final W50 feature set.

    Returns:
        Modeling frame and feature list.
    """

    helper = load_best_config_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    return data.reset_index(drop=True), features


def walk_forward_predict(
    data: pd.DataFrame,
    features: list[str],
    model_name: str,
    model: ProbabilisticClassifier,
) -> pd.DataFrame:
    """Generate walk-forward predictions for one architecture.

    Args:
        data: Chronologically sorted modeling frame.
        features: Input feature names.
        model_name: Readable model name.
        model: Classifier with ``fit`` and ``predict_proba``.

    Returns:
        Prediction table for the test period.
    """

    clean = data.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if initial_train.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced empty train or test set.")

    train_df = initial_train.copy()
    prediction_parts: list[pd.DataFrame] = []
    steps = range(0, len(test_pool), UPDATE_INTERVAL)
    for fold, start in enumerate(tqdm(steps, desc=f"Walk-forward {model_name}"), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model.fit(train_df[features], train_df[TARGET].astype(int))
        probabilities = model.predict_proba(test_chunk[features])[:, 1]
        prediction_parts.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold": fold,
                    "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                    "date": test_chunk["date"].to_numpy(),
                    "y_true": test_chunk[TARGET].astype(int).to_numpy(),
                    "y_prob": np.clip(probabilities, 0.001, 0.999),
                }
            )
        )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    return pd.concat(prediction_parts, ignore_index=True)


def evaluate_predictions(predictions: pd.DataFrame, n_features: int) -> dict[str, object]:
    """Evaluate one model's prediction table.

    Args:
        predictions: Prediction table with labels and probabilities.
        n_features: Number of model input features.

    Returns:
        Metric row.
    """

    y_true = predictions["y_true"].astype(int).to_numpy()
    y_prob = predictions["y_prob"].to_numpy()
    return {
        "model": predictions["model"].iloc[0],
        "sample_size": len(predictions),
        "date_min": pd.to_datetime(predictions["date"]).min().date().isoformat(),
        "date_max": pd.to_datetime(predictions["date"]).max().date().isoformat(),
        "n_features": n_features,
        "auc": roc_auc_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": calculate_ece(y_true, y_prob),
        "accuracy_0_5": accuracy_score(y_true, y_prob >= 0.5),
    }


def save_metric_plot(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a thesis-friendly bar chart for one metric."""

    ascending = metric in {"logloss", "brier", "ece"}
    data = results.sort_values(metric, ascending=ascending)
    colors = ["#6B7A8F", "#A68A64", "#6F8F72", "#8C6D8C", "#8A8A8A", "#A6766E", "#6E8A8A", "#9A8F6A"]
    plt.figure(figsize=(12.5, 6.8))
    ax = sns.barplot(data=data, x="model", y=metric, hue="model", legend=False, palette=colors[: len(data)])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=10, padding=4)
    values = data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.28, 0.0015)
    if metric in {"auc", "logloss", "brier"}:
        ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_title(f"Dodatkowe architektury metamodelu — {metric.upper()}")
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the additional architecture comparison."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    data, features = prepare_modeling_data()
    models = build_models()

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, model in models.items():
        predictions = walk_forward_predict(data, features, model_name, model)
        rows.append(evaluate_predictions(predictions, len(features)))
        prediction_frames.append(predictions)

    results = pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    results.to_csv(ASSETS_DIR / "additional_architecture_metrics.csv", index=False)
    all_predictions.to_csv(ASSETS_DIR / "additional_architecture_predictions.csv", index=False)

    save_metric_plot(results, "logloss", ASSETS_DIR / "additional_architecture_logloss.png")
    save_metric_plot(results, "auc", ASSETS_DIR / "additional_architecture_auc.png")
    save_metric_plot(results, "ece", ASSETS_DIR / "additional_architecture_ece.png")

    print("\n=== ADDITIONAL METAMODEL ARCHITECTURES ===")
    print(results.to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
