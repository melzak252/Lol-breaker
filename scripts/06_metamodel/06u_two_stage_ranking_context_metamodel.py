"""Evaluate a two-stage ranking-context metamodel.

Stage 1 learns a ranking-only metamodel that consolidates rating-family
probabilities into one probability. Stage 2 uses this Stage-1 probability plus
historical W50 context features. The experiment checks whether explicit
separation of ranking aggregation and context modeling improves over the current
one-stage LR-W50 baseline.
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
from src.analysis.probability_metrics import calculate_ece, binary_log_loss_vector as log_loss_vector
SOURCE_SCRIPT = PROJECT_ROOT / "scripts" / "06_metamodel" / "06i_best_metamodel_config_search.py"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "two_stage_ranking_context"
TARGET = "y_true"
UPDATE_INTERVAL = 1000
RANDOM_SEED = 42
N_BOOTSTRAPS = 10000

RANK_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]
RANK_UNCERTAINTY_FEATURES = [
    "player_elo_min1",
    "player_elo_min2",
    "player_gl_max1",
    "player_gl_max2",
    "player_gl_rd_avg1",
    "player_gl_rd_avg2",
    "player_ts_sigma_avg1",
    "player_ts_sigma_avg2",
    "player_os_sigma_avg1",
    "player_os_sigma_avg2",
    "player_pl_sigma_avg1",
    "player_pl_sigma_avg2",
    "player_tm_sigma_avg1",
    "player_tm_sigma_avg2",
]
G2_FEATURES = [
    "player_gl",
    "player_gl_max1",
    "player_gl_max2",
    "player_gl_rd_avg1",
    "player_gl_rd_avg2",
]
TEAM_LEVEL_SOURCE_COLUMNS = {
    "elo": ("team_elo_r1", "team_elo_r2"),
    "glicko2": ("team_gl_r1", "team_gl_r2"),
    "trueskill": ("team_ts_mu1", "team_ts_mu2"),
    "openskill": ("team_os_mu1", "team_os_mu2"),
    "plackett_luce": ("team_pl_mu1", "team_pl_mu2"),
    "thurstone": ("team_tm_mu1", "team_tm_mu2"),
}
TEAM_LEVEL_FEATURES = [
    "team_level_elo_avg",
    "team_level_elo_abs_gap",
    "team_level_glicko2_avg",
    "team_level_glicko2_abs_gap",
    "team_level_trueskill_avg",
    "team_level_trueskill_abs_gap",
    "team_level_openskill_avg",
    "team_level_openskill_abs_gap",
    "team_level_plackett_luce_avg",
    "team_level_plackett_luce_abs_gap",
    "team_level_thurstone_avg",
    "team_level_thurstone_abs_gap",
]


@dataclass(frozen=True)
class TwoStageVariant:
    """Configuration of one two-stage experiment variant."""

    name: str
    stage1_features: list[str]
    stage1_penalty: str
    stage1_c: float
    stage2_features: list[str]
    stage2_penalty: str
    stage2_c: float


def load_best_config_module() -> object:
    """Load helper functions from the best-configuration script."""
    spec = importlib.util.spec_from_file_location("best_config", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper module from {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def build_lr(penalty: str, c_value: float) -> Pipeline:
    """Build a median-imputed, standardized logistic-regression pipeline.

    Args:
        penalty: Either ``l1`` or ``l2``.
        c_value: Inverse regularization strength.

    Returns:
        Scikit-learn pipeline.
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


def prepare_data() -> tuple[pd.DataFrame, object]:
    """Load odds-mapped rating data and W50 context features."""
    helper = load_best_config_module()
    base = helper.load_base_data()
    base = add_team_level_features(base)
    rolling = helper.generate_rolling_features(helper.CONTEXT_WINDOW)
    data = base.merge(rolling, on="golgg_match_id", how="inner").sort_values("date")
    data["year"] = pd.to_datetime(data["date"]).dt.year
    return data.reset_index(drop=True), helper


def add_team_level_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add match-level absolute team-strength features.

    The raw rating scales differ across rating families. The logistic-regression
    pipelines standardize all included features with ``StandardScaler`` fitted
    only on the current training fold, so these level descriptors are converted
    to fold-local z-score scale before model fitting.

    Args:
        data: Rating prediction table.

    Returns:
        Data frame with average team level and absolute team-level gap features
        for every rating family with available raw team-rating columns.
    """
    enriched = data.copy()
    for family, (team1_col, team2_col) in TEAM_LEVEL_SOURCE_COLUMNS.items():
        avg_col = f"team_level_{family}_avg"
        gap_col = f"team_level_{family}_abs_gap"
        enriched[avg_col] = (enriched[team1_col].astype(float) + enriched[team2_col].astype(float)) / 2.0
        enriched[gap_col] = (enriched[team1_col].astype(float) - enriched[team2_col].astype(float)).abs()
    return enriched


def fit_stage1_probability(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    penalty: str,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Stage 1 on current history and return train/test probabilities.

    The test probabilities are strictly out-of-time. Train probabilities are
    in-sample Stage-1 outputs used to train Stage 2 on the same historical
    distribution available at the current walk-forward step.
    """
    stage1 = build_lr(penalty, c_value)
    stage1.fit(train_df[features], train_df[TARGET].astype(int))
    train_prob = np.clip(stage1.predict_proba(train_df[features])[:, 1], 0.001, 0.999)
    test_prob = np.clip(stage1.predict_proba(test_df[features])[:, 1], 0.001, 0.999)
    return train_prob, test_prob


def walk_forward_two_stage(data: pd.DataFrame, variant: TwoStageVariant) -> pd.DataFrame:
    """Run two-stage walk-forward prediction for one variant."""
    required = variant.stage1_features + variant.stage2_features + [TARGET]
    clean = data.dropna(subset=required).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if initial_train.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced empty train or test set.")

    train_df = initial_train.copy()
    parts: list[pd.DataFrame] = []
    for fold, start in enumerate(tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc=variant.name), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        train_rank_prob, test_rank_prob = fit_stage1_probability(
            train_df,
            test_chunk,
            variant.stage1_features,
            variant.stage1_penalty,
            variant.stage1_c,
        )
        stage2_train = train_df[variant.stage2_features].copy()
        stage2_test = test_chunk[variant.stage2_features].copy()
        stage2_train.insert(0, "p_rank_meta", train_rank_prob)
        stage2_test.insert(0, "p_rank_meta", test_rank_prob)

        stage2 = build_lr(variant.stage2_penalty, variant.stage2_c)
        stage2.fit(stage2_train, train_df[TARGET].astype(int))
        probabilities = np.clip(stage2.predict_proba(stage2_test)[:, 1], 0.001, 0.999)
        parts.append(
            pd.DataFrame(
                {
                    "variant": variant.name,
                    "fold": fold,
                    "golgg_match_id": test_chunk["golgg_match_id"].astype(str).to_numpy(),
                    "date": test_chunk["date"].to_numpy(),
                    "year": test_chunk["year"].to_numpy(),
                    "BoN": test_chunk["BoN"].to_numpy(),
                    "y_true": test_chunk[TARGET].astype(int).to_numpy(),
                    "p_rank_meta": test_rank_prob,
                    "y_prob": probabilities,
                }
            )
        )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def walk_forward_one_stage(
    data: pd.DataFrame,
    name: str,
    features: list[str],
    penalty: str = "l1",
    c_value: float = 0.30,
) -> pd.DataFrame:
    """Run one-stage LR walk-forward baseline for direct comparison."""
    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    train_df = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    parts: list[pd.DataFrame] = []
    for fold, start in enumerate(tqdm(range(0, len(test_pool), UPDATE_INTERVAL), desc=name), start=1):
        test_chunk = test_pool.iloc[start : start + UPDATE_INTERVAL].copy()
        model = build_lr(penalty, c_value)
        model.fit(train_df[features], train_df[TARGET].astype(int))
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
                    "p_rank_meta": np.nan,
                    "y_prob": probabilities,
                }
            )
        )
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)
    return pd.concat(parts, ignore_index=True)


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all variants with thesis metrics."""
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
    """Compare every variant against a baseline with monthly block bootstrap."""
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
    for variant in [c for c in wide.columns if c not in {"golgg_match_id", "date", "y_true", "month", baseline}]:
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
                "observed_delta_logloss": observed,
                "ci_lower_95": float(np.quantile(sample_array, 0.025)),
                "ci_upper_95": float(np.quantile(sample_array, 0.975)),
                "p_one_sided_variant_worse": float((np.sum(sample_array <= 0.0) + 1) / (len(sample_array) + 1)),
                "significantly_worse": bool(np.quantile(sample_array, 0.025) > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("observed_delta_logloss")


def plot_metric(metrics: pd.DataFrame, metric: str, file_name: str, ylabel: str) -> None:
    """Save a metric bar plot."""
    plot_data = metrics.sort_values(metric, ascending=(metric != "auc")).copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(plot_data["variant"], plot_data[metric], color="#6B7A8F", edgecolor="white")
    ax.set_title(f"Two-stage ranking-context — {metric.upper()}", fontsize=15, pad=12)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    values = plot_data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.35, 0.002)
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
    """Save confidence-interval plot for deltas vs LR-W50."""
    plot_data = bootstrap.sort_values("observed_delta_logloss").copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
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
    ax.set_title("Monthly block bootstrap — two-stage vs LR-W50")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "two_stage_block_bootstrap_ci.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the two-stage metamodel experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    data, helper = prepare_data()
    full_features = helper.OPTUNA_BASE_FEATURES + helper.ROLLING_FULL_FEATURES
    context_features = helper.ROLLING_FULL_FEATURES

    variants = [
        TwoStageVariant(
            "TwoStage rank-prob L2 + W50 L2",
            RANK_PROB_FEATURES,
            "l2",
            0.30,
            context_features,
            "l2",
            0.30,
        ),
        TwoStageVariant(
            "TwoStage rank-prob L1 + W50 L2",
            RANK_PROB_FEATURES,
            "l1",
            0.30,
            context_features,
            "l2",
            0.30,
        ),
        TwoStageVariant(
            "TwoStage rank+unc L2 + W50 L2",
            RANK_PROB_FEATURES + RANK_UNCERTAINTY_FEATURES,
            "l2",
            0.30,
            context_features,
            "l2",
            0.30,
        ),
        TwoStageVariant(
            "TwoStage rank+unc L1 + W50 L2",
            RANK_PROB_FEATURES + RANK_UNCERTAINTY_FEATURES,
            "l1",
            0.30,
            context_features,
            "l2",
            0.30,
        ),
        TwoStageVariant(
            "TwoStage rank+unc L1 + W50 L1",
            RANK_PROB_FEATURES + RANK_UNCERTAINTY_FEATURES,
            "l1",
            0.30,
            context_features,
            "l1",
            0.30,
        ),
        TwoStageVariant(
            "TwoStage rank+unc+level L1 + W50 L1",
            RANK_PROB_FEATURES + RANK_UNCERTAINTY_FEATURES + TEAM_LEVEL_FEATURES,
            "l1",
            0.30,
            context_features,
            "l1",
            0.30,
        ),
    ]

    prediction_parts = [walk_forward_two_stage(data, variant) for variant in variants]
    prediction_parts.append(walk_forward_one_stage(data, "LR-W50", full_features, "l1", 0.30))
    prediction_parts.append(walk_forward_one_stage(data, "G2 + context", G2_FEATURES + context_features, "l1", 0.30))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = evaluate_predictions(predictions)
    bootstrap = monthly_block_bootstrap(predictions, baseline="LR-W50")

    predictions.to_csv(OUTPUT_DIR / "two_stage_predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "two_stage_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "two_stage_block_bootstrap.csv", index=False)
    plot_metric(metrics, "logloss", "two_stage_logloss.png", "LogLoss (niżej = lepiej)")
    plot_metric(metrics, "auc", "two_stage_auc.png", "AUC (wyżej = lepiej)")
    plot_bootstrap(bootstrap)

    print("\n=== TWO-STAGE RANKING-CONTEXT RESULTS ===")
    print(metrics.to_string(index=False))
    print("\n=== BLOCK BOOTSTRAP VS LR-W50 ===")
    print(bootstrap.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
