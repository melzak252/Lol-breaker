"""Compare tuned W20-Binomial model families with block bootstrap.

This script treats Optuna as the hyperparameter-selection stage and evaluates
the tuned model families in a leakage-safe expanding walk-forward protocol.
The final comparison is reported as LogLoss deltas against Logistic Regression
ElasticNet with monthly block-bootstrap confidence intervals.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from math import comb
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.visualization.thesis_style import (
    DARK_TEXT,
    PASTEL_BLUE,
    PASTEL_RED,
    apply_thesis_style,
    clean_axis,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import binary_log_loss_vector as log_loss_vector, calculate_ece
HELPER_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "w20_binomial_all_models_bootstrap"
TARGET = "y_true"
CONTEXT_WINDOW = 20
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
N_BOOTSTRAPS = 10000
BASELINE_VARIANT = "Logistic Regression ElasticNet W20-Binomial"

RANK_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]


class ProbabilisticClassifier(Protocol):
    """Protocol for binary classifiers exposing ``fit`` and ``predict_proba``."""

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Fit a classifier."""

    def predict_proba(self, x_test: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""


def load_helper_module() -> object:
    """Load the current metamodel helper module.

    Returns:
        Imported helper module with data-loading and rolling-context functions.
    """

    spec = importlib.util.spec_from_file_location("best_config", HELPER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {HELPER_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def series_probability(map_probability: np.ndarray, best_of: np.ndarray) -> np.ndarray:
    """Convert approximate map-win probabilities to best-of-series probabilities.

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
    """Add binomial series-adjusted ranking features.

    Args:
        data: Modeling frame with ``BoN`` and ranking probability columns.

    Returns:
        Enriched frame and names of generated features.
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
    """Prepare W20-Binomial data for all model-family comparisons.

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
    clean["year"] = pd.to_datetime(clean["date"]).dt.year
    return clean.reset_index(drop=True), features


def maybe_mask_training_frame(x_train: pd.DataFrame, mask_rate: float, seed: int) -> pd.DataFrame:
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


def build_logistic_regression() -> Pipeline:
    """Build the tuned ElasticNet logistic-regression candidate.

    Returns:
        Median-imputed and standardized logistic-regression pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.03297234640536737,
                    penalty="elasticnet",
                    l1_ratio=0.9439657999531195,
                    solver="saga",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_extra_trees() -> Pipeline:
    """Build the tuned ExtraTrees candidate.

    Returns:
        Median-imputed ExtraTrees pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=797,
                    max_depth=9,
                    min_samples_leaf=23,
                    max_features=0.8,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_hist_gradient_boosting() -> Pipeline:
    """Build the tuned histogram-gradient-boosting candidate.

    Returns:
        Median-imputed HistGradientBoosting pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=201,
                    learning_rate=0.016660815547461415,
                    max_leaf_nodes=12,
                    min_samples_leaf=200,
                    l2_regularization=0.0002892673145972703,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_lightgbm() -> Pipeline:
    """Build the tuned LightGBM candidate.

    Returns:
        Median-imputed LightGBM pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    max_depth=3,
                    num_leaves=4,
                    learning_rate=0.012805167280103086,
                    n_estimators=414,
                    min_child_samples=147,
                    subsample=0.942480859768491,
                    colsample_bytree=0.8133135296907867,
                    reg_alpha=0.0013112363452929046,
                    reg_lambda=0.022766181282355566,
                    random_state=RANDOM_SEED,
                    verbosity=-1,
                ),
            ),
        ]
    )


def build_mlp() -> Pipeline:
    """Build the tuned MLP candidate.

    Returns:
        Median-imputed and standardized MLP pipeline.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    alpha=0.0014014609810153243,
                    learning_rate_init=0.004262402727044875,
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def model_registry() -> list[tuple[str, Callable[[], ProbabilisticClassifier], float]]:
    """Return tuned model families and training mask rates.

    Returns:
        Tuples of variant name, model builder, and mask rate.
    """

    return [
        (BASELINE_VARIANT, build_logistic_regression, 0.0),
        ("ExtraTrees W20-Binomial", build_extra_trees, 0.05),
        ("HistGradientBoosting W20-Binomial", build_hist_gradient_boosting, 0.10),
        ("LightGBM W20-Binomial", build_lightgbm, 0.20),
        ("MLP W20-Binomial", build_mlp, 0.0),
    ]


def walk_forward_model(
    data: pd.DataFrame,
    features: list[str],
    name: str,
    model_builder: Callable[[], ProbabilisticClassifier],
    mask_rate: float,
) -> pd.DataFrame:
    """Run expanding-window walk-forward prediction for one model.

    Args:
        data: Modeling frame.
        features: Feature column names.
        name: Variant name.
        model_builder: Zero-argument callable returning a classifier.
        mask_rate: Training feature mask rate.

    Returns:
        Out-of-time predictions for all walk-forward chunks.
    """

    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if train_df.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced an empty train or test subset.")

    parts: list[pd.DataFrame] = []
    for fold, start in enumerate(tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc=name), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model = model_builder()
        x_train = maybe_mask_training_frame(
            train_df[features],
            mask_rate=mask_rate,
            seed=RANDOM_SEED + fold,
        )
        model.fit(x_train, train_df[TARGET].astype(int))
        probabilities = np.clip(model.predict_proba(test_chunk[features])[:, 1], 0.001, 0.999)
        parts.append(
            pd.DataFrame(
                {
                    "variant": name,
                    "fold": fold,
                    "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                    "date": test_chunk["date"].to_numpy(),
                    "year": test_chunk["year"].to_numpy(),
                    "BoN": test_chunk["BoN"].to_numpy(),
                    "y_true": test_chunk[TARGET].astype(int).to_numpy(),
                    "y_prob": probabilities,
                }
            )
        )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate variants with thesis metrics.

    Args:
        predictions: Long prediction table.

    Returns:
        Metric table sorted by LogLoss.
    """

    rows: list[dict[str, object]] = []
    for variant, group in predictions.groupby("variant"):
        y_true = group["y_true"].astype(int).to_numpy()
        y_prob = group["y_prob"].to_numpy(dtype=float)
        rows.append(
            {
                "variant": variant,
                "sample_size": int(len(group)),
                "auc": float(roc_auc_score(y_true, y_prob)),
                "logloss": float(log_loss(y_true, y_prob)),
                "brier": float(brier_score_loss(y_true, y_prob)),
                "ece": calculate_ece(y_true, y_prob),
                "accuracy_0_5": float(accuracy_score(y_true, y_prob >= 0.5)),
            }
        )
    return pd.DataFrame(rows).sort_values("logloss")


def monthly_block_bootstrap(predictions: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Compare variants against a baseline with monthly block bootstrap.

    Args:
        predictions: Long prediction table.
        baseline: Variant used as the reference. Delta is variant minus baseline.

    Returns:
        Bootstrap confidence intervals for LogLoss deltas.
    """

    wide = predictions.pivot_table(
        index=["golgg_match_id", "date", "y_true"],
        columns="variant",
        values="y_prob",
        aggfunc="first",
    ).reset_index()
    wide["month"] = pd.to_datetime(wide["date"]).dt.to_period("M").astype(str)
    months = sorted(wide["month"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    baseline_loss = log_loss_vector(wide["y_true"].to_numpy(), wide[baseline].to_numpy())
    rows: list[dict[str, object]] = []
    excluded_columns = {"golgg_match_id", "date", "y_true", "month", baseline}
    for variant in [column for column in wide.columns if column not in excluded_columns]:
        variant_loss = log_loss_vector(wide["y_true"].to_numpy(), wide[variant].to_numpy())
        wide["delta"] = variant_loss - baseline_loss
        observed = float(wide["delta"].mean())
        month_stats = wide.groupby("month")["delta"].agg(delta_sum="sum", n="size").loc[months]
        delta_sums = month_stats["delta_sum"].to_numpy(dtype=float)
        counts = month_stats["n"].to_numpy(dtype=float)
        samples = []
        for _ in range(N_BOOTSTRAPS):
            sampled_idx = rng.integers(0, len(months), size=len(months))
            samples.append(float(delta_sums[sampled_idx].sum() / counts[sampled_idx].sum()))
        sample_array = np.asarray(samples)
        rows.append(
            {
                "comparison": f"{variant} vs {baseline}",
                "variant": variant,
                "baseline": baseline,
                "observed_delta_logloss": observed,
                "ci_lower_95": float(np.quantile(sample_array, 0.025)),
                "ci_upper_95": float(np.quantile(sample_array, 0.975)),
                "p_one_sided_variant_worse": float((np.sum(sample_array <= 0.0) + 1) / (len(sample_array) + 1)),
                "significantly_worse": bool(np.quantile(sample_array, 0.025) > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("observed_delta_logloss")


def plot_bootstrap_intervals(bootstrap: pd.DataFrame, output_path: Path) -> None:
    """Create a horizontal confidence-interval plot for LogLoss deltas.

    Args:
        bootstrap: Bootstrap comparison table.
        output_path: Target PNG path.
    """

    plot_data = bootstrap.sort_values("observed_delta_logloss", ascending=True).copy()
    labels = plot_data["variant"].str.replace(" W20-Binomial", "", regex=False)
    y_pos = np.arange(len(plot_data))
    lower_error = plot_data["observed_delta_logloss"] - plot_data["ci_lower_95"]
    upper_error = plot_data["ci_upper_95"] - plot_data["observed_delta_logloss"]
    apply_thesis_style(context="paper")
    colors = np.where(plot_data["significantly_worse"], PASTEL_RED, PASTEL_BLUE)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axvline(0.0, color=DARK_TEXT, linewidth=1.1, linestyle="--", label="brak różnicy")
    ax.errorbar(
        plot_data["observed_delta_logloss"],
        y_pos,
        xerr=np.vstack([lower_error, upper_error]),
        fmt="none",
        ecolor=DARK_TEXT,
        elinewidth=1.4,
        capsize=4,
        zorder=1,
    )
    ax.scatter(plot_data["observed_delta_logloss"], y_pos, s=70, c=colors, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ LogLoss względem Logistic Regression ElasticNet")
    ax.set_title("Bootstrap blokowy modeli W20-Binomial")
    clean_axis(ax, grid_axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run all tuned W20-Binomial model-family comparisons and bootstrap."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, features = prepare_data()
    prediction_frames = []
    for name, builder, mask_rate in model_registry():
        prediction_frames.append(
            walk_forward_model(
                data=data,
                features=features,
                name=name,
                model_builder=builder,
                mask_rate=mask_rate,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = evaluate_predictions(predictions)
    bootstrap = monthly_block_bootstrap(predictions, baseline=BASELINE_VARIANT)

    predictions.to_csv(OUTPUT_DIR / "all_models_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "all_models_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "all_models_block_bootstrap.csv", index=False)
    plot_bootstrap_intervals(bootstrap, OUTPUT_DIR / "all_models_bootstrap_delta_logloss.png")

    print("\n=== W20-BINOMIAL TUNED MODEL-FAMILY METRICS ===")
    print(metrics.to_string(index=False))
    print("\n=== MONTHLY BLOCK BOOTSTRAP VS LOGISTIC REGRESSION ===")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
