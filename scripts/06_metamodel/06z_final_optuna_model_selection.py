"""Tune final LR-W20-Binomial model family candidates with Optuna.

This diagnostic uses the final planned feature space: player-rating
probabilities, rating uncertainty features, rolling W20 context, and binomial
series-adjusted rating probabilities. Hyperparameters are optimized on a
chronological validation split to support thesis-level model-family selection.
"""

from __future__ import annotations

import importlib.util
import sys
from math import comb
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece
HELPER_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "final_optuna_model_selection"
TARGET = "y_true"
CONTEXT_WINDOW = 20
RANDOM_SEED = 42
N_TRIALS_PER_FAMILY = 35

RANK_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]


class ProbabilisticClassifier(Protocol):
    """Protocol for classifiers returning binary probabilities."""

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Fit the classifier."""

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        """Return class-probability matrix."""


def load_helper_module() -> object:
    """Load the current metamodel helper module.

    Returns:
        Imported helper module with data loading and rolling-context functions.
    """

    spec = importlib.util.spec_from_file_location("best_config", HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    """Convert map-win probabilities to best-of-series probabilities.

    Args:
        map_probability: Approximate probability of winning one map.
        best_of: Series length encoded as 1, 3, or 5.

    Returns:
        Probability of winning the full series.
    """

    probability = np.clip(map_probability.astype(float), 0.001, 0.999)
    best_of_int = best_of.astype(int)
    result = probability.copy()
    for n_maps in (3, 5):
        needed = n_maps // 2 + 1
        series_prob = np.zeros_like(probability)
        for wins in range(needed, n_maps + 1):
            series_prob += (
                comb(n_maps, wins)
                * np.power(probability, wins)
                * np.power(1.0 - probability, n_maps - wins)
            )
        result = np.where(best_of_int == n_maps, series_prob, result)
    return np.clip(result, 0.001, 0.999)


def add_binomial_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add binomial series-adjusted ranking probability features.

    Args:
        data: Modeling frame containing `BoN` and ranking probabilities.

    Returns:
        Enriched frame and generated feature names.
    """

    enriched = data.copy()
    generated: list[str] = []
    best_of = enriched["BoN"].fillna(1).astype(int).to_numpy()
    for feature in RANK_PROB_FEATURES:
        column = f"{feature}_binom_series"
        enriched[column] = series_probability(enriched[feature].to_numpy(dtype=float), best_of)
        generated.append(column)
    return enriched, generated


def prepare_data() -> tuple[pd.DataFrame, list[str]]:
    """Prepare final W20-binomial modeling data.

    Returns:
        Chronologically sorted modeling frame and final feature list.
    """

    helper = load_helper_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner")
    data = data.sort_values("date").reset_index(drop=True)
    data, binomial_features = add_binomial_features(data)
    features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES + binomial_features
    clean = data.dropna(subset=features + [TARGET]).copy()
    clean = clean[clean["date"] >= pd.Timestamp("2020-01-01")].copy()
    return clean.reset_index(drop=True), features


def split_train_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation split for Optuna.

    Args:
        data: Chronologically sorted modeling frame.

    Returns:
        Training and validation frames.
    """

    train = data[data["date"] < pd.Timestamp("2023-01-01")].copy()
    validation = data[
        (data["date"] >= pd.Timestamp("2023-01-01"))
        & (data["date"] < pd.Timestamp("2024-01-01"))
    ].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological train/validation split produced an empty subset.")
    return train, validation


def maybe_mask_training_frame(
    x_train: pd.DataFrame,
    mask_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Randomly mask training features for robustness regularization.

    Args:
        x_train: Training feature matrix.
        mask_rate: Probability of masking each cell.
        seed: Random seed.

    Returns:
        Feature matrix with optional missing values.
    """

    if mask_rate <= 0.0:
        return x_train.copy()
    rng = np.random.default_rng(seed)
    return x_train.mask(rng.random(x_train.shape) < mask_rate)


def build_logistic_model(trial: optuna.Trial) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Build a logistic-regression candidate from Optuna suggestions."""

    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    params: dict[str, Any] = {
        "C": trial.suggest_float("C", 0.03, 3.0, log=True),
        "penalty": penalty,
        "solver": "saga",
        "max_iter": 5000,
        "random_state": RANDOM_SEED,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.05, 0.95)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params)),
        ]
    )
    return model, params


def build_lightgbm_model(trial: optuna.Trial) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Build a LightGBM candidate from Optuna suggestions."""

    max_depth = trial.suggest_int("max_depth", 2, 6)
    params: dict[str, Any] = {
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 3, min(63, 2**max_depth - 1)),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 80, 450),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 250),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "verbosity": -1,
    }
    return LGBMClassifier(**params), params


def build_hist_gradient_boosting_model(
    trial: optuna.Trial,
) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Build a histogram gradient boosting candidate."""

    params: dict[str, Any] = {
        "max_iter": trial.suggest_int("max_iter", 80, 350),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 31),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 40, 250),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
        "random_state": RANDOM_SEED,
    }
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(**params)),
        ]
    )
    return model, params


def build_extra_trees_model(trial: optuna.Trial) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Build an ExtraTrees candidate."""

    params: dict[str, Any] = {
        "n_estimators": trial.suggest_int("n_estimators", 250, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(**params)),
        ]
    )
    return model, params


def build_mlp_model(trial: optuna.Trial) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Build an MLP candidate from Optuna suggestions."""

    architecture = trial.suggest_categorical("architecture", ["8", "16", "32", "32_16"])
    hidden_layers = {
        "8": (8,),
        "16": (16,),
        "32": (32,),
        "32_16": (32, 16),
    }[architecture]
    params: dict[str, Any] = {
        "hidden_layer_sizes": hidden_layers,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-4, 0.2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True),
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
        "random_state": RANDOM_SEED,
    }
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(**params)),
        ]
    )
    return model, params


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_model,
    "lightgbm": build_lightgbm_model,
    "hist_gradient_boosting": build_hist_gradient_boosting_model,
    "extra_trees": build_extra_trees_model,
    "mlp": build_mlp_model,
}


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate predicted probabilities.

    Args:
        y_true: Binary labels.
        y_prob: Positive-class probabilities.

    Returns:
        Metric dictionary.
    """

    clipped = np.clip(y_prob, 0.001, 0.999)
    return {
        "logloss": float(log_loss(y_true, clipped)),
        "auc": float(roc_auc_score(y_true, clipped)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "ece": calculate_ece(y_true, clipped),
        "accuracy_0_5": float(accuracy_score(y_true, clipped >= 0.5)),
    }


def optimize_family(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
) -> tuple[optuna.Study, dict[str, Any]]:
    """Optimize one model family.

    Args:
        family: Model-family key.
        train: Training frame.
        validation: Validation frame.
        features: Input feature names.

    Returns:
        Completed Optuna study and best result summary.
    """

    builder = MODEL_BUILDERS[family]
    x_validation = validation[features]
    y_validation = validation[TARGET].astype(int).to_numpy()

    def objective(trial: optuna.Trial) -> float:
        """Evaluate one hyperparameter configuration."""

        mask_rate = trial.suggest_categorical("mask_rate", [0.0, 0.05, 0.10, 0.20])
        model, params = builder(trial)
        x_train = maybe_mask_training_frame(train[features], mask_rate, RANDOM_SEED + trial.number)
        y_train = train[TARGET].astype(int)
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_validation)[:, 1]
        metrics = evaluate_probabilities(y_validation, probability)
        for metric_name, metric_value in metrics.items():
            trial.set_user_attr(metric_name, metric_value)
        trial.set_user_attr("model_params", params)
        return metrics["logloss"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(
        direction="minimize",
        study_name=f"final_w20_binomial_{family}",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=N_TRIALS_PER_FAMILY, show_progress_bar=True)

    best = study.best_trial
    summary = {
        "family": family,
        "best_logloss": best.value,
        "best_auc": best.user_attrs["auc"],
        "best_brier": best.user_attrs["brier"],
        "best_ece": best.user_attrs["ece"],
        "best_accuracy_0_5": best.user_attrs["accuracy_0_5"],
        "best_params": best.params,
        "model_params": best.user_attrs["model_params"],
    }
    return study, summary


def main() -> None:
    """Run Optuna optimization for final model-family selection."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, features = prepare_data()
    train, validation = split_train_validation(data)
    print("=== FINAL W20-BINOMIAL OPTUNA MODEL SELECTION ===")
    print(f"Rows: {len(data)} | Train: {len(train)} | Validation: {len(validation)}")
    print(f"Features: {len(features)}")

    summaries: list[dict[str, Any]] = []
    all_trials: list[pd.DataFrame] = []
    for family in MODEL_BUILDERS:
        print(f"\nOptimizing family: {family}")
        study, summary = optimize_family(family, train, validation, features)
        summaries.append(summary)
        trials = study.trials_dataframe()
        trials.insert(0, "family", family)
        all_trials.append(trials)
        print(
            f"Best {family}: LogLoss={summary['best_logloss']:.6f}, "
            f"AUC={summary['best_auc']:.6f}, ECE={summary['best_ece']:.6f}"
        )

    summary_df = pd.DataFrame(summaries).sort_values("best_logloss").reset_index(drop=True)
    trials_df = pd.concat(all_trials, ignore_index=True)
    summary_df.to_csv(OUTPUT_DIR / "final_w20_binomial_optuna_summary.csv", index=False)
    trials_df.to_csv(OUTPUT_DIR / "final_w20_binomial_optuna_trials.csv", index=False)
    feature_df = pd.DataFrame({"feature": features})
    feature_df.to_csv(OUTPUT_DIR / "final_w20_binomial_features.csv", index=False)

    print("\n=== BEST FAMILY SUMMARY ===")
    print(summary_df[["family", "best_logloss", "best_auc", "best_brier", "best_ece"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
