"""Evaluate rating-disagreement and ensemble features for LR metamodel.

The experiment checks whether rating families other than Glicko-2 add useful
information when represented as disagreement/consensus features instead of raw,
highly correlated probability columns. All variants are evaluated with the same
chronological walk-forward protocol used by the thesis metamodel experiments.
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece, binary_log_loss_vector as log_loss_vector
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "rating_disagreement_features"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
N_BOOTSTRAPS = 10000

PLAYER_RATING_PROBS = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]
G2_FEATURES = [
    "player_gl",
    "player_gl_max1",
    "player_gl_max2",
    "player_gl_rd_avg1",
    "player_gl_rd_avg2",
]


@dataclass(frozen=True)
class FeatureVariant:
    """One feature-engineering variant for the disagreement experiment."""

    name: str
    features: list[str]
    use_pca: bool = False
    n_components: int = 0


def load_best_config_module() -> object:
    """Load helper functions from the best-configuration script."""
    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def add_disagreement_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add rating consensus and disagreement features.

    Args:
        data: Modeling frame with player rating probability columns.

    Returns:
        Copy of the input frame with engineered disagreement columns.
    """
    enriched = data.copy()
    rating_matrix = enriched[PLAYER_RATING_PROBS]
    enriched["rating_mean"] = rating_matrix.mean(axis=1)
    enriched["rating_std"] = rating_matrix.std(axis=1)
    enriched["rating_min"] = rating_matrix.min(axis=1)
    enriched["rating_max"] = rating_matrix.max(axis=1)
    enriched["rating_range"] = enriched["rating_max"] - enriched["rating_min"]
    for column in PLAYER_RATING_PROBS:
        if column == "player_gl":
            continue
        suffix = column.replace("player_", "")
        enriched[f"gl_minus_{suffix}"] = enriched["player_gl"] - enriched[column]
    enriched["non_g2_mean"] = enriched[[c for c in PLAYER_RATING_PROBS if c != "player_gl"]].mean(axis=1)
    enriched["gl_minus_non_g2_mean"] = enriched["player_gl"] - enriched["non_g2_mean"]
    return enriched


def prepare_modeling_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load odds-mapped modeling data and define feature variants."""
    helper = load_best_config_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    data["year"] = pd.to_datetime(data["date"]).dt.year
    data = add_disagreement_features(data)

    context_features = helper.ROLLING_FULL_FEATURES
    stats_features = ["rating_mean", "rating_std", "rating_min", "rating_max", "rating_range"]
    diff_features = [
        "gl_minus_elo",
        "gl_minus_ts",
        "gl_minus_os",
        "gl_minus_pl",
        "gl_minus_tm",
        "non_g2_mean",
        "gl_minus_non_g2_mean",
    ]
    variants = {
        "LR-W50": helper.OPTUNA_BASE_FEATURES + context_features,
        "G2 + context": G2_FEATURES + context_features,
        "Consensus + context": stats_features + context_features,
        "G2 + disagreement": G2_FEATURES + stats_features + diff_features + context_features,
        "All ratings + disagreement": helper.OPTUNA_BASE_FEATURES + stats_features + diff_features + context_features,
    }
    return data.reset_index(drop=True), variants


def build_lr_pipeline() -> Pipeline:
    """Build the main L1-regularized logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=0.30,
                    l1_ratio=1.0,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_pca_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    context_features: list[str],
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create fold-local PCA rating features without leakage.

    Args:
        train_df: Current walk-forward training frame.
        test_df: Current test chunk.
        context_features: Context features appended after PCA features.
        n_components: Number of rating principal components.

    Returns:
        Train features, test features and feature names.
    """
    rating_imputer = SimpleImputer(strategy="median")
    rating_scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)

    train_rating = rating_imputer.fit_transform(train_df[PLAYER_RATING_PROBS])
    train_rating = rating_scaler.fit_transform(train_rating)
    train_pca = pca.fit_transform(train_rating)

    test_rating = rating_imputer.transform(test_df[PLAYER_RATING_PROBS])
    test_rating = rating_scaler.transform(test_rating)
    test_pca = pca.transform(test_rating)

    pca_columns = [f"rating_pc{i}" for i in range(1, n_components + 1)]
    train_features = pd.DataFrame(train_pca, columns=pca_columns, index=train_df.index)
    test_features = pd.DataFrame(test_pca, columns=pca_columns, index=test_df.index)
    for feature in context_features:
        train_features[feature] = train_df[feature].to_numpy()
        test_features[feature] = test_df[feature].to_numpy()
    return train_features, test_features, pca_columns + context_features


def walk_forward_predict(
    data: pd.DataFrame,
    variant: FeatureVariant,
    context_features: list[str],
) -> pd.DataFrame:
    """Run walk-forward prediction for one feature variant."""
    clean = data.dropna(subset=[TARGET]).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    train_df = initial_train.copy()
    prediction_parts: list[pd.DataFrame] = []

    for fold, start in enumerate(tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc=variant.name), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model = build_lr_pipeline()
        if variant.use_pca:
            x_train, x_test, _ = build_pca_train_test(
                train_df, test_chunk, context_features, variant.n_components
            )
            model.fit(x_train, train_df[TARGET].astype(int))
            probabilities = model.predict_proba(x_test)[:, 1]
        else:
            model.fit(train_df[variant.features], train_df[TARGET].astype(int))
            probabilities = model.predict_proba(test_chunk[variant.features])[:, 1]

        prediction_parts.append(
            pd.DataFrame(
                {
                    "variant": variant.name,
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
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(prediction_parts, ignore_index=True)


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all prediction variants."""
    rows: list[dict[str, object]] = []
    for variant, group in predictions.groupby("variant"):
        y_true = group["y_true"].astype(int).to_numpy()
        y_prob = group["y_prob"].to_numpy()
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


def monthly_block_bootstrap(predictions: pd.DataFrame, baseline: str = "LR-W50") -> pd.DataFrame:
    """Run monthly block bootstrap for feature variants versus baseline."""
    wide = predictions.pivot_table(
        index=["golgg_match_id", "date", "y_true"],
        columns="variant",
        values="y_prob",
        aggfunc="first",
    ).reset_index()
    wide["month"] = pd.to_datetime(wide["date"]).dt.to_period("M").astype(str)
    months = sorted(wide["month"].unique())
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    baseline_loss = log_loss_vector(wide["y_true"].to_numpy(), wide[baseline].to_numpy())
    for variant in [column for column in wide.columns if column not in {"golgg_match_id", "date", "y_true", "month", baseline}]:
        variant_loss = log_loss_vector(wide["y_true"].to_numpy(), wide[variant].to_numpy())
        wide["delta"] = variant_loss - baseline_loss
        observed = float(wide["delta"].mean())
        month_stats = (
            wide.groupby("month", as_index=False)["delta"]
            .agg(delta_sum="sum", n="size")
            .set_index("month")
            .loc[months]
        )
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
                "observed_delta_logloss": observed,
                "ci_lower_95": float(np.quantile(sample_array, 0.025)),
                "ci_upper_95": float(np.quantile(sample_array, 0.975)),
                "p_one_sided_variant_worse": float((np.sum(sample_array <= 0.0) + 1) / (len(sample_array) + 1)),
                "significantly_worse": bool(np.quantile(sample_array, 0.025) > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("observed_delta_logloss")



def plot_metrics(metrics: pd.DataFrame, metric: str, file_name: str, ylabel: str) -> None:
    """Save a bar chart for one metric."""
    plot_data = metrics.sort_values(metric, ascending=(metric != "auc")).copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(plot_data["variant"], plot_data[metric], color="#6B7A8F", edgecolor="white")
    ax.set_title(f"Cechy disagreement/ensemble — {metric.upper()}", fontsize=15, pad=12)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    values = plot_data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.35, 0.002)
    if metric == "auc":
        ax.set_ylim(values.min() - margin, values.max() + margin)
    else:
        ax.set_ylim(values.min() - margin, values.max() + margin)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / file_name, dpi=180)
    plt.close(fig)


def plot_bootstrap(bootstrap: pd.DataFrame) -> None:
    """Save confidence intervals for deltas versus LR-W50."""
    plot_data = bootstrap.sort_values("observed_delta_logloss").copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y_positions = np.arange(len(plot_data))
    ax.errorbar(
        plot_data["observed_delta_logloss"],
        y_positions,
        xerr=[
            plot_data["observed_delta_logloss"] - plot_data["ci_lower_95"],
            plot_data["ci_upper_95"] - plot_data["observed_delta_logloss"],
        ],
        fmt="o",
        color="#4F6D7A",
        ecolor="#A68A64",
        capsize=4,
    )
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_data["comparison"])
    ax.set_xlabel("Delta LogLoss względem LR-W50")
    ax.set_title("Block bootstrap — cechy disagreement vs LR-W50")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rating_disagreement_block_bootstrap_ci.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the disagreement-feature experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    data, feature_sets = prepare_modeling_data()
    helper = load_best_config_module()
    context_features = helper.ROLLING_FULL_FEATURES
    variants = [FeatureVariant(name=name, features=features) for name, features in feature_sets.items()]
    variants.extend(
        [
            FeatureVariant(name="PCA2 + context", features=[], use_pca=True, n_components=2),
            FeatureVariant(name="PCA3 + context", features=[], use_pca=True, n_components=3),
        ]
    )

    predictions = pd.concat(
        [walk_forward_predict(data, variant, context_features) for variant in variants],
        ignore_index=True,
    )
    metrics = evaluate_predictions(predictions)
    predictions.to_csv(OUTPUT_DIR / "rating_disagreement_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "rating_disagreement_metrics.csv", index=False)
    bootstrap = monthly_block_bootstrap(predictions)

    bootstrap.to_csv(OUTPUT_DIR / "rating_disagreement_block_bootstrap.csv", index=False)
    plot_metrics(metrics, "logloss", "rating_disagreement_logloss.png", "LogLoss (niżej = lepiej)")
    plot_metrics(metrics, "auc", "rating_disagreement_auc.png", "AUC (wyżej = lepiej)")
    plot_bootstrap(bootstrap)

    print("\n=== RATING DISAGREEMENT FEATURE RESULTS ===")
    print(metrics.to_string(index=False))
    print("\n=== BLOCK BOOTSTRAP VS LR-W50 ===")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
