"""Test masking and ablation of dominant rating signals in LR-W50.

The experiment checks whether the main logistic-regression metamodel depends too
strongly on Player Glicko-2. It compares the baseline LR-W50 configuration with
training-time masking of rating features, targeted masking of ``player_gl``, and
feature-ablation variants.
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
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "logistic_masking_ablation"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
BEST_C = 0.30


@dataclass(frozen=True)
class MaskingVariant:
    """Configuration of one masking or ablation variant."""

    variant: str
    features: list[str]
    mask_strategy: str
    mask_rate: float
    mask_columns: list[str]


def load_best_config_module() -> object:
    """Load helper functions from the best-configuration script.

    Returns:
        Imported helper module used for data loading and rolling features.
    """

    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def build_estimator() -> Pipeline:
    """Create the final L1 logistic-regression estimator.

    Returns:
        Pipeline with median imputation, standardization and L1 logistic model.
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    C=BEST_C,
                    l1_ratio=1.0,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def prepare_modeling_data() -> tuple[pd.DataFrame, object]:
    """Load the odds-mapped sample with W50 rolling features.

    Returns:
        Chronological modeling frame and imported helper module.
    """

    helper = load_best_config_module()
    base = helper.load_base_data()
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data.reset_index(drop=True), helper


def apply_training_mask(
    train_df: pd.DataFrame,
    variant: MaskingVariant,
    fold: int,
) -> pd.DataFrame:
    """Apply deterministic training-time feature masking.

    Args:
        train_df: Training fold before modification.
        variant: Masking configuration.
        fold: Walk-forward fold index used to vary the random seed.

    Returns:
        Training frame with selected cells replaced by ``NaN``.
    """

    if variant.mask_strategy == "none" or variant.mask_rate <= 0:
        return train_df

    masked = train_df.copy()
    available_columns = [column for column in variant.mask_columns if column in masked.columns]
    if not available_columns:
        return masked

    rng = np.random.default_rng(RANDOM_SEED + fold * 1009 + int(variant.mask_rate * 1000))
    mask = rng.random((len(masked), len(available_columns))) < variant.mask_rate
    values = masked[available_columns].to_numpy(dtype=float, copy=True)
    values[mask] = np.nan
    masked.loc[:, available_columns] = values
    return masked


def walk_forward_predict(
    data: pd.DataFrame,
    variant: MaskingVariant,
    collect_coefficients: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward LR prediction for one masking variant.

    Args:
        data: Chronologically sorted modeling frame.
        variant: Masking or ablation variant.
        collect_coefficients: Whether to export fold-level coefficients.

    Returns:
        Prediction rows and optional coefficient rows.
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

    for fold, start in enumerate(tqdm(steps, desc=f"Masking {variant.variant}"), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        masked_train = apply_training_mask(train_df, variant, fold)
        model = build_estimator()
        model.fit(masked_train[variant.features], masked_train[TARGET].astype(int))
        probabilities = model.predict_proba(test_chunk[variant.features])[:, 1]

        prediction_parts.append(
            pd.DataFrame(
                {
                    "variant": variant.variant,
                    "mask_strategy": variant.mask_strategy,
                    "mask_rate": variant.mask_rate,
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
            coefficient_parts.append(
                pd.DataFrame(
                    {
                        "variant": variant.variant,
                        "mask_strategy": variant.mask_strategy,
                        "mask_rate": variant.mask_rate,
                        "fold": fold,
                        "feature": variant.features,
                        "coefficient": estimator.coef_[0],
                    }
                )
            )

        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    coefficients = pd.concat(coefficient_parts, ignore_index=True) if coefficient_parts else pd.DataFrame()
    return predictions, coefficients


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all variants in a prediction table.

    Args:
        predictions: Prediction rows with ``variant``, ``y_true`` and ``y_prob``.

    Returns:
        Metric table sorted by LogLoss.
    """

    rows: list[dict[str, object]] = []
    for variant, frame in predictions.groupby("variant", sort=False):
        y_true = frame["y_true"].astype(int).to_numpy()
        y_prob = frame["y_prob"].to_numpy()
        rows.append(
            {
                "variant": variant,
                "sample_size": len(frame),
                "date_min": pd.to_datetime(frame["date"]).min().date().isoformat(),
                "date_max": pd.to_datetime(frame["date"]).max().date().isoformat(),
                "auc": roc_auc_score(y_true, y_prob),
                "logloss": log_loss(y_true, y_prob),
                "brier": brier_score_loss(y_true, y_prob),
                "ece": calculate_ece(y_true, y_prob),
                "accuracy_0_5": accuracy_score(y_true, y_prob >= 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def feature_group(feature: str) -> str:
    """Map a feature name to a compact semantic group."""

    if feature.startswith("t1_rolling") or feature.startswith("t2_rolling"):
        return "context"
    if "gl" in feature:
        return "glicko2"
    if "elo" in feature:
        return "elo"
    if "ts" in feature:
        return "trueskill"
    if "os" in feature:
        return "openskill"
    if "pl" in feature:
        return "plackett_luce"
    if "tm" in feature:
        return "thurstone_mosteller"
    return "other"


def summarize_coefficients(coefficients: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fold-level coefficients by feature and by feature group."""

    feature_summary = (
        coefficients.groupby(["variant", "feature"])
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", lambda values: float(np.mean(np.abs(values)))),
            positive_share=("coefficient", lambda values: float(np.mean(np.asarray(values) > 0))),
        )
        .reset_index()
        .sort_values(["variant", "mean_abs_coefficient"], ascending=[True, False])
    )
    feature_summary["feature_group"] = feature_summary["feature"].map(feature_group)
    group_summary = (
        feature_summary.groupby(["variant", "feature_group"])
        .agg(total_abs_coefficient=("mean_abs_coefficient", "sum"), n_features=("feature", "nunique"))
        .reset_index()
        .sort_values(["variant", "total_abs_coefficient"], ascending=[True, False])
    )
    return feature_summary, group_summary


def plot_metric_bars(metrics: pd.DataFrame, metric: str, output_path: Path) -> None:
    """Save a bar chart for one metric."""

    data = metrics.sort_values(metric, ascending=(metric != "auc"))
    plt.figure(figsize=(11.5, 6.4))
    ax = sns.barplot(data=data, x="variant", y=metric, color="#6B7A8F", edgecolor="white")
    ax.set_title(f"LR-W50 masking/ablacja — {metric.upper()}")
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    for patch in ax.patches:
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_group_importance(group_summary: pd.DataFrame, output_path: Path) -> None:
    """Save feature-group coefficient importance plot."""

    selected = group_summary[group_summary["variant"].isin(["LR-W50", "Mask all 30%", "Mask G2 30%", "No player_gl"])]
    plt.figure(figsize=(11.5, 6.8))
    ax = sns.barplot(
        data=selected,
        x="variant",
        y="total_abs_coefficient",
        hue="feature_group",
        palette="muted",
    )
    ax.set_title("Suma |współczynników| według grup cech")
    ax.set_xlabel("")
    ax.set_ylabel("Suma średnich |współczynników|")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.legend(title="Grupa cech", bbox_to_anchor=(1.02, 1), loc="upper left")
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def create_variants(helper: object) -> list[MaskingVariant]:
    """Create masking and ablation variants.

    Args:
        helper: Imported best-config module defining feature lists.

    Returns:
        Ordered list of experiment variants.
    """

    full_features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    no_player_gl = [feature for feature in full_features if feature != "player_gl"]
    glicko2_family = ["player_gl", "player_gl_max1", "player_gl_max2", "player_gl_rd_avg1", "player_gl_rd_avg2"]
    no_glicko2_family = [feature for feature in full_features if feature not in glicko2_family]
    glicko2_context_only = glicko2_family + helper.ROLLING_FULL_FEATURES

    variants = [
        MaskingVariant("LR-W50", full_features, "none", 0.0, []),
        MaskingVariant("Mask all 10%", full_features, "random_all_ratings", 0.10, helper.OPTUNA_BASE_FEATURES),
        MaskingVariant("Mask all 20%", full_features, "random_all_ratings", 0.20, helper.OPTUNA_BASE_FEATURES),
        MaskingVariant("Mask all 30%", full_features, "random_all_ratings", 0.30, helper.OPTUNA_BASE_FEATURES),
        MaskingVariant("Mask G2 10%", full_features, "player_gl", 0.10, ["player_gl"]),
        MaskingVariant("Mask G2 20%", full_features, "player_gl", 0.20, ["player_gl"]),
        MaskingVariant("Mask G2 30%", full_features, "player_gl", 0.30, ["player_gl"]),
        MaskingVariant("No player_gl", no_player_gl, "ablation", 0.0, []),
        MaskingVariant("No G2 family", no_glicko2_family, "ablation", 0.0, []),
        MaskingVariant("G2 + context", glicko2_context_only, "minimal", 0.0, []),
    ]
    return variants


def main() -> None:
    """Run masking and ablation experiments for LR-W50."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    data, helper = prepare_modeling_data()
    variants = create_variants(helper)

    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    for variant in variants:
        predictions, coefficients = walk_forward_predict(data, variant, collect_coefficients=True)
        prediction_frames.append(predictions)
        coefficient_frames.append(coefficients)

    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    coefficients_all = pd.concat(coefficient_frames, ignore_index=True)
    metrics = evaluate_predictions(predictions_all)
    baseline = float(metrics.loc[metrics["variant"] == "LR-W50", "logloss"].iloc[0])
    metrics["delta_logloss_vs_lr_w50"] = metrics["logloss"] - baseline

    feature_summary, group_summary = summarize_coefficients(coefficients_all)

    metrics.to_csv(ASSETS_DIR / "logistic_masking_ablation_metrics.csv", index=False)
    predictions_all.to_csv(ASSETS_DIR / "logistic_masking_ablation_predictions.csv", index=False)
    coefficients_all.to_csv(ASSETS_DIR / "logistic_masking_ablation_coefficients_by_fold.csv", index=False)
    feature_summary.to_csv(ASSETS_DIR / "logistic_masking_feature_importance.csv", index=False)
    group_summary.to_csv(ASSETS_DIR / "logistic_masking_group_importance.csv", index=False)

    plot_metric_bars(metrics, "logloss", ASSETS_DIR / "logistic_masking_logloss.png")
    plot_metric_bars(metrics, "auc", ASSETS_DIR / "logistic_masking_auc.png")
    plot_metric_bars(metrics, "ece", ASSETS_DIR / "logistic_masking_ece.png")
    plot_group_importance(group_summary, ASSETS_DIR / "logistic_masking_group_importance.png")

    print("\n=== LOGISTIC MASKING / ABLATION ===")
    print(metrics.to_string(index=False))
    print("\n=== GROUP IMPORTANCE (selected top rows) ===")
    print(group_summary.head(30).to_string(index=False))
    print("\nSaved artefacts to:", ASSETS_DIR)


if __name__ == "__main__":
    main()
