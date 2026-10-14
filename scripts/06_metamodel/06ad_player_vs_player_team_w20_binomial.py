"""Compare player-only and player-plus-team W20-Binomial inputs.

The final thesis model uses player-based ranking probabilities, player-rating
uncertainty, W20 historical team context, and binomial series-adjusted player
signals. This diagnostic tests whether adding team-based rating predictions and
their uncertainty descriptors improves the same tuned ElasticNet estimator.
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
from src.analysis.bootstrap import monthly_block_bootstrap_delta
HELPER_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "player_vs_player_team_w20_binomial"
TARGET = "y_true"
CONTEXT_WINDOW = 20
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
N_BOOTSTRAPS = 10000
BASELINE_VARIANT = "Player-based W20-Binomial"

PLAYER_RANK_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]

TEAM_RANK_PROB_FEATURES = [
    "team_elo",
    "team_gl",
    "team_ts",
    "team_os",
    "team_pl",
    "team_tm",
]

TEAM_RATING_CONTEXT_FEATURES = [
    "team_elo_r1",
    "team_elo_r2",
    "team_gl_r1",
    "team_gl_rd1",
    "team_gl_r2",
    "team_gl_rd2",
    "team_ts_mu1",
    "team_ts_sigma1",
    "team_ts_mu2",
    "team_ts_sigma2",
    "team_os_mu1",
    "team_os_sigma1",
    "team_os_mu2",
    "team_os_sigma2",
    "team_pl_mu1",
    "team_pl_sigma1",
    "team_pl_mu2",
    "team_pl_sigma2",
    "team_tm_mu1",
    "team_tm_sigma1",
    "team_tm_mu2",
    "team_tm_sigma2",
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


def add_binomial_features(
    data: pd.DataFrame,
    probability_features: list[str],
    suffix: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Add binomial series-adjusted probability features.

    Args:
        data: Modeling frame with ``BoN`` and probability columns.
        probability_features: Probability columns to transform.
        suffix: Suffix used in generated column names.

    Returns:
        Enriched frame and generated feature names.
    """

    enriched = data.copy()
    generated: list[str] = []
    best_of = enriched["BoN"].fillna(1).astype(int).to_numpy()
    for feature in probability_features:
        column = f"{feature}_{suffix}"
        enriched[column] = series_probability(enriched[feature].to_numpy(dtype=float), best_of)
        generated.append(column)
    return enriched, generated


def prepare_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare player-only and player-plus-team W20-Binomial feature sets.

    Returns:
        Modeling frame, player-only features, and augmented player-plus-team features.
    """

    helper = load_helper_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner")
    data = data.sort_values("date").reset_index(drop=True)
    data, player_binomial = add_binomial_features(
        data, PLAYER_RANK_PROB_FEATURES, "binom_series"
    )
    data, team_binomial = add_binomial_features(
        data, TEAM_RANK_PROB_FEATURES, "binom_series"
    )

    player_features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES + player_binomial
    team_features = TEAM_RANK_PROB_FEATURES + TEAM_RATING_CONTEXT_FEATURES + team_binomial
    augmented_features = player_features + team_features

    clean = data.dropna(subset=augmented_features + [TARGET]).copy()
    clean = clean[clean["date"] >= pd.Timestamp("2020-01-01")].copy()
    clean["month"] = pd.to_datetime(clean["date"]).dt.to_period("M").astype(str)
    return clean.reset_index(drop=True), player_features, augmented_features


def build_logistic_regression() -> Pipeline:
    """Build the tuned ElasticNet logistic-regression estimator.

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


def walk_forward_predict(
    data: pd.DataFrame,
    features: list[str],
    build_model: Callable[[], ProbabilisticClassifier],
    variant: str,
) -> pd.DataFrame:
    """Generate leakage-safe walk-forward predictions.

    Args:
        data: Chronologically sorted modeling frame.
        features: Feature names used by the estimator.
        build_model: Factory returning a fresh classifier.
        variant: Human-readable variant name.

    Returns:
        Frame with out-of-time predictions and identifiers.
    """

    train_df = data[data["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = data[data["date"] >= pd.Timestamp("2021-01-01")].copy()
    predictions: list[pd.DataFrame] = []

    for start in tqdm(
        range(0, len(test_pool), UPDATE_INTERVAL),
        desc=f"Walk-forward {variant}",
    ):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        if test_chunk.empty:
            continue

        model = build_model()
        model.fit(train_df[features], train_df[TARGET])
        probability = np.clip(model.predict_proba(test_chunk[features])[:, 1], 0.001, 0.999)

        chunk_predictions = test_chunk[
            ["golgg_match_id", "date", "month", "team1_name", "team2_name", TARGET]
        ].copy()
        chunk_predictions["variant"] = variant
        chunk_predictions["probability"] = probability
        predictions.append(chunk_predictions)

        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    return pd.concat(predictions, ignore_index=True)


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float | str | int]:
    """Calculate probabilistic classification metrics for one variant.

    Args:
        predictions: Frame with ``y_true`` and ``probability``.

    Returns:
        Metric dictionary.
    """

    y_true = predictions[TARGET].to_numpy()
    y_prob = predictions["probability"].to_numpy()
    return {
        "variant": str(predictions["variant"].iloc[0]),
        "n": int(len(predictions)),
        "auc": roc_auc_score(y_true, y_prob),
        "logloss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": calculate_ece(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_prob >= 0.5),
    }


def monthly_block_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap LogLoss differences for augmented features versus player-only.

    Args:
        predictions: Long prediction frame containing both compared variants.

    Returns:
        Bootstrap summary where positive delta means augmented model is worse.
    """

    pivot = predictions.pivot_table(
        index=["golgg_match_id", "month", TARGET],
        columns="variant",
        values="probability",
        aggfunc="first",
    ).reset_index()
    baseline_loss = -(
        pivot[TARGET] * np.log(np.clip(pivot[BASELINE_VARIANT], 1e-15, 1.0))
        + (1 - pivot[TARGET]) * np.log(np.clip(1 - pivot[BASELINE_VARIANT], 1e-15, 1.0))
    )

    summaries: list[dict[str, float | str | bool]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    month_values = pivot["month"].unique()
    for variant in [col for col in pivot.columns if col not in {"golgg_match_id", "month", TARGET, BASELINE_VARIANT}]:
        variant_loss = -(
            pivot[TARGET] * np.log(np.clip(pivot[variant], 1e-15, 1.0))
            + (1 - pivot[TARGET]) * np.log(np.clip(1 - pivot[variant], 1e-15, 1.0))
        )
        pivot["delta"] = variant_loss - baseline_loss
        # Resample complete monthly blocks and retain the match-weighted estimand.
        pivot["date"] = pd.to_datetime(pivot["month"].astype(str))
        observed, ci_low, ci_high, boot = monthly_block_bootstrap_delta(
            pivot, "delta", n_bootstraps=N_BOOTSTRAPS, random_generator=rng,
        )
        sign_fraction = float(np.mean(boot <= 0.0))
        summaries.append(
            {
                "variant": variant,
                "baseline": BASELINE_VARIANT,
                "delta_logloss_variant_minus_baseline": observed,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_fraction_delta_le_zero": sign_fraction,
                "significantly_better": bool(ci_high < 0.0),
                "significantly_worse": bool(ci_low > 0.0),
            }
        )
    return pd.DataFrame(summaries)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    """Plot bootstrap LogLoss delta confidence intervals.

    Args:
        bootstrap: Bootstrap summary frame.
    """

    if bootstrap.empty:
        return
    ordered = bootstrap.sort_values("delta_logloss_variant_minus_baseline")
    y_pos = np.arange(len(ordered))
    values = ordered["delta_logloss_variant_minus_baseline"].to_numpy()
    lower = values - ordered["ci_low"].to_numpy()
    upper = ordered["ci_high"].to_numpy() - values

    plt.figure(figsize=(8.5, 3.0))
    plt.errorbar(values, y_pos, xerr=[lower, upper], fmt="o", color="#1f77b4", capsize=4)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.yticks(y_pos, ordered["variant"])
    plt.xlabel("Δ LogLoss względem player-only (wariant - baseline)")
    plt.title("Czy team-based ratingi poprawiają finalny input?")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "player_vs_player_team_bootstrap_delta_logloss.png", dpi=200)
    plt.close()


def main() -> None:
    """Run player-only versus player-plus-team W20-Binomial comparison."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, player_features, augmented_features = prepare_data()
    variants = {
        BASELINE_VARIANT: player_features,
        "Player+Team W20-Binomial": augmented_features,
    }

    prediction_frames = []
    metric_rows = []
    for variant, features in variants.items():
        predictions = walk_forward_predict(data, features, build_logistic_regression, variant)
        prediction_frames.append(predictions)
        metric_rows.append(calculate_metrics(predictions))

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("logloss")
    bootstrap = monthly_block_bootstrap(all_predictions)
    plot_bootstrap(bootstrap)

    all_predictions.to_csv(OUTPUT_DIR / "player_vs_player_team_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "player_vs_player_team_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "player_vs_player_team_bootstrap.csv", index=False)

    print("\n=== Player-only vs Player+Team W20-Binomial ===")
    print(metrics.to_string(index=False))
    print("\n=== Monthly block bootstrap ===")
    print(bootstrap.to_string(index=False))
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
