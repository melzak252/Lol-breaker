"""Evaluate logistic-regression robustness and feature importance.

This experiment is restricted to the odds-mapped thesis sample. It tests whether
the strong result of logistic regression observed in the algorithm-comparison
experiment is stable across feature sets and regularization strengths. It also
exports coefficient-based feature importance, calibration diagnostics and
temporal/format segment metrics for the best variant.
"""

from __future__ import annotations

import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "logistic_regression_robustness"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
REGULARIZATION_GRID = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 10.00]


@dataclass(frozen=True)
class LogisticVariant:
    """Configuration of one logistic-regression experiment variant."""

    feature_set: str
    features: list[str]
    penalty: str
    c_value: float

    @property
    def variant_name(self) -> str:
        """Return a compact, readable variant label."""

        return f"{self.feature_set} | {self.penalty.upper()} | C={self.c_value:g}"


def load_best_config_module() -> object:
    """Load helper functions from the best-configuration script.

    Returns:
        Imported module with data-loading and rolling-feature helpers.
    """

    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def build_estimator(penalty: str, c_value: float) -> Pipeline:
    """Create a logistic-regression pipeline.

    Args:
        penalty: Regularization penalty, currently ``l1`` or ``l2``.
        c_value: Inverse regularization strength.

    Returns:
        Pipeline with median imputation, standardization and logistic regression.
    """

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=c_value,
                    l1_ratio=l1_ratio,
                    solver=solver,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def prepare_modeling_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the odds-mapped modeling sample and define feature sets.

    Returns:
        Chronologically sorted modeling frame and feature-set mapping.
    """

    helper = load_best_config_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    data["year"] = pd.to_datetime(data["date"]).dt.year

    feature_sets = {
        "player_only": helper.OPTUNA_BASE_FEATURES,
        "player_core_context_w50": helper.OPTUNA_BASE_FEATURES + helper.ROLLING_CORE_FEATURES,
        "player_full_context_w50": helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES,
    }
    return data.reset_index(drop=True), feature_sets


def walk_forward_predict(
    data: pd.DataFrame,
    variant: LogisticVariant,
    collect_coefficients: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward logistic-regression prediction.

    Args:
        data: Chronologically sorted modeling frame.
        variant: Logistic-regression configuration.
        collect_coefficients: Whether to store standardized coefficients per fold.

    Returns:
        Tuple of prediction rows and coefficient rows.
    """

    clean = data.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if initial_train.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced empty train or test set.")

    train_df = initial_train.copy()
    prediction_parts: list[pd.DataFrame] = []
    coefficient_parts: list[pd.DataFrame] = []

    steps = range(0, len(test_pool), UPDATE_INTERVAL)
    for fold, start in enumerate(tqdm(steps, desc=f"Logistic {variant.variant_name}"), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model = build_estimator(variant.penalty, variant.c_value)
        model.fit(train_df[variant.features], train_df[TARGET].astype(int))
        probabilities = model.predict_proba(test_chunk[variant.features])[:, 1]

        prediction_parts.append(
            pd.DataFrame(
                {
                    "variant": variant.variant_name,
                    "feature_set": variant.feature_set,
                    "penalty": variant.penalty,
                    "c_value": variant.c_value,
                    "fold": fold,
                    "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                    "date": test_chunk["date"].to_numpy(),
                    "year": test_chunk["year"].to_numpy(),
                    "BoN": test_chunk["BoN"].to_numpy(),
                    "y_true": test_chunk[TARGET].astype(int).to_numpy(),
                    "y_prob": np.clip(probabilities, 0.001, 0.999),
                }
            )
        )

        if collect_coefficients:
            estimator = model.named_steps["model"]
            coefficients = estimator.coef_[0]
            coefficient_parts.append(
                pd.DataFrame(
                    {
                        "variant": variant.variant_name,
                        "feature_set": variant.feature_set,
                        "penalty": variant.penalty,
                        "c_value": variant.c_value,
                        "fold": fold,
                        "feature": variant.features,
                        "coefficient": coefficients,
                    }
                )
            )

        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else pd.DataFrame()
    return predictions, coefficients


def evaluate_prediction_frame(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Evaluate predictions globally or by segment.

    Args:
        predictions: Prediction rows containing ``y_true`` and ``y_prob``.
        group_cols: Columns used for grouping. Empty list means one global row.

    Returns:
        Evaluation table with probabilistic metrics.
    """

    groups = [("overall", predictions)] if not group_cols else predictions.groupby(group_cols, dropna=False)
    rows: list[dict[str, object]] = []
    for key, frame in groups:
        y_true = frame["y_true"].astype(int).to_numpy()
        y_prob = frame["y_prob"].to_numpy()
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true, y_prob)
        row: dict[str, object] = {
            "sample_size": len(frame),
            "date_min": pd.to_datetime(frame["date"]).min().date().isoformat(),
            "date_max": pd.to_datetime(frame["date"]).max().date().isoformat(),
            "auc": auc,
            "logloss": log_loss(y_true, y_prob),
            "brier": brier_score_loss(y_true, y_prob),
            "ece": calculate_ece(y_true, y_prob),
            "accuracy_0_5": accuracy_score(y_true, y_prob >= 0.5),
        }
        if group_cols:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(group_cols, key_tuple)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_coefficients(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Aggregate standardized logistic-regression coefficients across folds.

    Args:
        coefficients: Fold-level coefficient table.

    Returns:
        Feature-importance table sorted by mean absolute coefficient.
    """

    if coefficients.empty:
        return coefficients
    summary = (
        coefficients.groupby("feature")
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", lambda values: float(np.mean(np.abs(values)))),
            std_coefficient=("coefficient", "std"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
            positive_share=("coefficient", lambda values: float(np.mean(np.asarray(values) > 0))),
        )
        .reset_index()
        .sort_values("mean_abs_coefficient", ascending=False)
    )
    return summary


def save_metric_plot(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a metric plot over feature sets and regularization strengths."""

    plt.figure(figsize=(10.5, 6.4))
    ax = sns.lineplot(
        data=results[results["penalty"] == "l2"],
        x="c_value",
        y=metric,
        hue="feature_set",
        marker="o",
        linewidth=2.2,
        palette=["#6B7A8F", "#A68A64", "#6F8F72"],
    )
    ax.set_xscale("log")
    ax.set_title(f"Regresja logistyczna — {metric.upper()} względem regularyzacji")
    ax.set_xlabel("C, odwrotność siły regularyzacji L2")
    ax.set_ylabel(metric.upper())
    ax.legend(title="Zestaw cech")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_feature_importance_plot(importance: pd.DataFrame, output_path: Path, top_n: int = 18) -> None:
    """Save coefficient-based feature-importance chart."""

    data = importance.head(top_n).sort_values("mean_abs_coefficient", ascending=True)
    plt.figure(figsize=(10.5, 7.8))
    ax = sns.barplot(data=data, x="mean_abs_coefficient", y="feature", color="#6B7A8F")
    ax.set_title("Regresja logistyczna — najważniejsze cechy wg |współczynnika|")
    ax.set_xlabel("Średnia bezwzględna wartość standaryzowanego współczynnika")
    ax.set_ylabel("")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_calibration_plot(predictions: pd.DataFrame, output_path: Path, n_bins: int = 10) -> None:
    """Save reliability curve for the selected logistic-regression variant."""

    data = predictions.copy()
    data["bin"] = pd.cut(data["y_prob"], bins=np.linspace(0.0, 1.0, n_bins + 1), include_lowest=True)
    calibration = (
        data.groupby("bin", observed=True)
        .agg(mean_prob=("y_prob", "mean"), observed_rate=("y_true", "mean"), sample_size=("y_true", "size"))
        .reset_index()
    )
    calibration.to_csv(ASSETS_DIR / "logistic_best_calibration_bins.csv", index=False)

    plt.figure(figsize=(7.5, 7.0))
    ax = sns.lineplot(data=calibration, x="mean_prob", y="observed_rate", marker="o", color="#6B7A8F")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Regresja logistyczna — krzywa kalibracji")
    ax.set_xlabel("Średnie przewidywane prawdopodobieństwo")
    ax.set_ylabel("Rzeczywisty udział zwycięstw")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run logistic-regression robustness and feature-importance analysis."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    data, feature_sets = prepare_modeling_data()

    variants = [
        LogisticVariant(feature_set=name, features=features, penalty="l2", c_value=c_value)
        for name, features in feature_sets.items()
        for c_value in REGULARIZATION_GRID
    ]
    variants.extend(
        [
            LogisticVariant(
                feature_set="player_full_context_w50",
                features=feature_sets["player_full_context_w50"],
                penalty="l1",
                c_value=c_value,
            )
            for c_value in [0.03, 0.10, 0.30, 1.00]
        ]
    )

    metric_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for variant in variants:
        predictions, _ = walk_forward_predict(data, variant, collect_coefficients=False)
        metrics = evaluate_prediction_frame(predictions, [])
        metrics["variant"] = variant.variant_name
        metrics["feature_set"] = variant.feature_set
        metrics["penalty"] = variant.penalty
        metrics["c_value"] = variant.c_value
        metrics["n_features"] = len(variant.features)
        metric_rows.append(metrics)
        prediction_frames.append(predictions)

    metrics_all = pd.concat(metric_rows, ignore_index=True).sort_values("logloss").reset_index(drop=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    metrics_all.to_csv(ASSETS_DIR / "logistic_regression_robustness_metrics.csv", index=False)
    predictions_all.to_csv(ASSETS_DIR / "logistic_regression_robustness_predictions.csv", index=False)

    save_metric_plot(metrics_all, "logloss", ASSETS_DIR / "logistic_regularization_logloss.png")
    save_metric_plot(metrics_all, "auc", ASSETS_DIR / "logistic_regularization_auc.png")
    save_metric_plot(metrics_all, "ece", ASSETS_DIR / "logistic_regularization_ece.png")

    best_row = metrics_all.iloc[0]
    best_variant = next(variant for variant in variants if variant.variant_name == best_row["variant"])
    best_predictions, coefficients = walk_forward_predict(data, best_variant, collect_coefficients=True)
    best_predictions.to_csv(ASSETS_DIR / "logistic_best_predictions.csv", index=False)
    coefficients.to_csv(ASSETS_DIR / "logistic_best_coefficients_by_fold.csv", index=False)
    importance = summarize_coefficients(coefficients)
    importance.to_csv(ASSETS_DIR / "logistic_best_feature_importance.csv", index=False)
    save_feature_importance_plot(importance, ASSETS_DIR / "logistic_best_feature_importance.png")
    save_calibration_plot(best_predictions, ASSETS_DIR / "logistic_best_calibration.png")

    segment_year = evaluate_prediction_frame(best_predictions, ["year"]).sort_values("year")
    segment_bon = evaluate_prediction_frame(best_predictions, ["BoN"]).sort_values("BoN")
    segment_year.to_csv(ASSETS_DIR / "logistic_best_metrics_by_year.csv", index=False)
    segment_bon.to_csv(ASSETS_DIR / "logistic_best_metrics_by_bon.csv", index=False)

    print("\n=== LOGISTIC REGRESSION ROBUSTNESS ===")
    print(metrics_all.head(12).to_string(index=False))
    print("\n=== BEST VARIANT ===")
    print(best_row.to_string())
    print("\n=== TOP FEATURE IMPORTANCE ===")
    print(importance.head(20).to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
