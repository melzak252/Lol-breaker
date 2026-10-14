"""Run metamodel ablation experiments on the odds-mapped sample only.

The experiments in this script are designed for whitepaper point 6. They test:

1. Player-based rating features vs player+team rating features.
2. Different walk-forward model update intervals.
3. Different rolling context windows.
4. No masking vs input masking.
5. Additional context-feature ablations.

All reported evaluation rows are restricted to matches present in ``odds.csv``.
Rolling features are nevertheless computed chronologically from the full
``golgg_matches.json`` history, because historical non-odds matches can be known
before the evaluated odds match and therefore are legitimate context history.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece

from src.utils.golgg_schema import games, team1_id, team2_id

ASSETS_DIR = PROJECT_ROOT / "docs" / "assets" / "metamodel_experiments_point6"

CONTEXT_WINDOWS = [3, 5, 10, 20, 50]
DEFAULT_CONTEXT_WINDOW = 10
DEFAULT_UPDATE_INTERVAL = 1000
DEFAULT_MASK_RATE = 0.20

TARGET = "y_true"

PLAYER_PROB_FEATURES = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]
PLAYER_UNCERTAINTY_FEATURES = [
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
TEAM_PROB_FEATURES = [
    "team_elo",
    "team_gl",
    "team_ts",
    "team_os",
    "team_pl",
    "team_tm",
]
MATCH_CONTEXT_FEATURES = ["days_since_last_1", "days_since_last_2", "days_diff", "BoN"]
ROLLING_CORE_FEATURES = [
    "t1_rolling_win_rate",
    "t2_rolling_win_rate",
    "t1_rolling_gd15",
    "t2_rolling_gd15",
]
ROLLING_FULL_FEATURES = [
    "t1_rolling_win_rate",
    "t2_rolling_win_rate",
    "t1_rolling_kills",
    "t2_rolling_kills",
    "t1_rolling_deaths",
    "t2_rolling_deaths",
    "t1_rolling_gd15",
    "t2_rolling_gd15",
    "t1_rolling_dpm",
    "t2_rolling_dpm",
    "t1_rolling_vspm",
    "t2_rolling_vspm",
    "t1_rolling_towers",
    "t2_rolling_towers",
    "t1_rolling_nashors",
    "t2_rolling_nashors",
    "t1_rolling_gold",
    "t2_rolling_gold",
    "t1_rolling_duration",
    "t2_rolling_duration",
]


def configure_plot_style() -> None:
    """Configure compact, thesis-friendly charts."""

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )



def safe_stat(player: dict[str, Any], key: str) -> float:
    """Read a numeric player statistic and convert missing values to zero."""

    value = player.get("stats", {}).get(key, 0.0)
    return float(value or 0.0)


def safe_team_stat(game: dict[str, Any], stats_key: str, key: str) -> float:
    """Read a numeric team-level statistic from a game record.

    Args:
        game: Single game dictionary from the GOL.GG match archive.
        stats_key: Team stats key, usually ``t1_stats`` or ``t2_stats``.
        key: Statistic name inside the team stats dictionary.

    Returns:
        Numeric statistic value or zero when the field is missing.
    """

    value = (game.get(stats_key, {}) or {}).get(key, 0.0)
    return float(value or 0.0)


def default_team_stats() -> dict[str, float]:
    """Return neutral defaults for teams with no historical games."""

    return {
        "win_rate": 0.5,
        "kills": 12.0,
        "deaths": 12.0,
        "gd15": 0.0,
        "dpm": 1800.0,
        "vspm": 7.0,
        "towers": 5.0,
        "nashors": 0.5,
        "gold": 55000.0,
        "duration": 1800.0,
    }


def average_history(history: deque[dict[str, float]] | None) -> dict[str, float]:
    """Average a team's rolling game history."""

    if not history:
        return default_team_stats()
    rows = list(history)
    return {
        "win_rate": float(np.mean([row["win"] for row in rows])),
        "kills": float(np.mean([row["kills"] for row in rows])),
        "deaths": float(np.mean([row["deaths"] for row in rows])),
        "gd15": float(np.mean([row["gd15"] for row in rows])),
        "dpm": float(np.mean([row["dpm"] for row in rows])),
        "vspm": float(np.mean([row["vspm"] for row in rows])),
        "towers": float(np.mean([row["towers"] for row in rows])),
        "nashors": float(np.mean([row["nashors"] for row in rows])),
        "gold": float(np.mean([row["gold"] for row in rows])),
        "duration": float(np.mean([row["duration"] for row in rows])),
    }


def update_team_history(
    team_history: dict[str, deque[dict[str, float]]],
    team_id: str,
    game: dict[str, Any],
    window_size: int,
) -> None:
    """Update a team's rolling history with one already-played game."""

    is_team_1 = str(game.get("t1_id")) == str(team_id)
    win = bool(game.get("t1_win")) if is_team_1 else bool(game.get("t2_win"))
    players_key = "t1_players" if is_team_1 else "t2_players"
    stats_key = "t1_stats" if is_team_1 else "t2_stats"
    players = game.get(players_key, {}) or {}
    game_stats = {
        "win": float(win),
        "kills": sum(safe_stat(player, "kills") for player in players.values()),
        "deaths": sum(safe_stat(player, "deaths") for player in players.values()),
        "dpm": sum(safe_stat(player, "dpm") for player in players.values()),
        "vspm": sum(safe_stat(player, "vspm") for player in players.values()),
        "gd15": sum(safe_stat(player, "gd@15") for player in players.values()),
        "towers": safe_team_stat(game, stats_key, "towers"),
        "nashors": safe_team_stat(game, stats_key, "nashors"),
        "gold": safe_team_stat(game, stats_key, "gold"),
        "duration": float(game.get("game_duration") or 0.0),
    }
    if team_id not in team_history:
        team_history[team_id] = deque(maxlen=window_size)
    team_history[team_id].append(game_stats)


def generate_rolling_features(windows: list[int]) -> dict[int, pd.DataFrame]:
    """Generate rolling context features for multiple window sizes.

    Args:
        windows: Rolling window sizes in games.

    Returns:
        Mapping from window size to feature DataFrame.
    """

    with open(PROJECT_ROOT / "data" / "golgg_matches.json", "r", encoding="utf-8") as file:
        matches = json.load(file)
    matches.sort(key=lambda item: item["date"])

    histories: dict[int, dict[str, deque[dict[str, float]]]] = {window: {} for window in windows}
    results: dict[int, list[dict[str, object]]] = {window: [] for window in windows}

    for match in tqdm(matches, desc="Rolling windows"):
        match_id = str(match["match_id"])
        team_1 = team1_id(match)
        team_2 = team2_id(match)
        for window in windows:
            team_history = histories[window]
            t1_stats = average_history(team_history.get(team_1))
            t2_stats = average_history(team_history.get(team_2))
            row: dict[str, object] = {"golgg_match_id": match_id, "context_window": window}
            for stat, value in t1_stats.items():
                row[f"t1_rolling_{stat}"] = value
            for stat, value in t2_stats.items():
                row[f"t2_rolling_{stat}"] = value
            results[window].append(row)

        for game in games(match):
            for window in windows:
                update_team_history(histories[window], team_1, game, window)
                update_team_history(histories[window], team_2, game, window)

    return {window: pd.DataFrame(rows) for window, rows in results.items()}


def load_base_predictions() -> pd.DataFrame:
    """Load rating predictions restricted to the odds-mapped sample."""

    predictions = pd.read_csv(PROJECT_ROOT / "data" / "golgg_y_predicts.csv")
    odds = pd.read_csv(PROJECT_ROOT / "data" / "odds.csv", usecols=["golgg_match_id"])
    predictions["golgg_match_id"] = predictions["golgg_match_id"].astype(str)
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    data = predictions.merge(odds.drop_duplicates(), on="golgg_match_id", how="inner")
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


def lgbm_params() -> dict[str, object]:
    """Return LightGBM parameters used across ablation experiments."""

    return {
        "max_depth": 6,
        "num_leaves": 6,
        "learning_rate": 0.024,
        "n_estimators": 486,
        "min_child_samples": 60,
        "subsample": 0.727,
        "colsample_bytree": 0.859,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "verbosity": -1,
        "random_state": 42,
    }


def walk_forward_predict(
    data: pd.DataFrame,
    features: list[str],
    update_interval: int,
    mask_rate: float,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Generate walk-forward predictions on odds-mapped matches.

    Args:
        data: Chronologically sorted dataset.
        features: Feature columns.
        update_interval: Number of test matches between model refits.
        mask_rate: Probability of masking each training feature value.
        random_seed: Seed for deterministic masking.

    Returns:
        Tuple of labels, probabilities and test dates.
    """

    clean = data.dropna(subset=features + [TARGET]).copy().sort_values("date").reset_index(drop=True)
    initial_train = clean[clean["date"] < pd.Timestamp("2021-01-01")].copy()
    test_pool = clean[clean["date"] >= pd.Timestamp("2021-01-01")].copy()
    if initial_train.empty or test_pool.empty:
        raise ValueError("Walk-forward split produced empty train or test set.")

    rng = np.random.default_rng(random_seed)
    train_df = initial_train.copy()
    y_true_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    date_parts: list[pd.Series] = []

    for start in range(0, len(test_pool), update_interval):
        test_chunk = test_pool.iloc[start : start + update_interval]
        x_train = train_df[features].copy()
        if mask_rate > 0:
            mask = rng.random(x_train.shape) < mask_rate
            x_train = x_train.mask(mask)
        model = LGBMClassifier(**lgbm_params())
        model.fit(x_train, train_df[TARGET].astype(int))
        probabilities = model.predict_proba(test_chunk[features])[:, 1]
        y_true_parts.append(test_chunk[TARGET].astype(int).to_numpy())
        prob_parts.append(probabilities)
        date_parts.append(test_chunk["date"])
        train_df = pd.concat([train_df, test_chunk], ignore_index=True)

    return np.concatenate(y_true_parts), np.concatenate(prob_parts), pd.concat(date_parts, ignore_index=True)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dates: pd.Series,
    experiment_group: str,
    variant: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Calculate metrics for one experiment variant."""

    clipped = np.clip(y_prob, 0.001, 0.999)
    row = {
        "experiment_group": experiment_group,
        "variant": variant,
        "sample_size": len(y_true),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "auc": roc_auc_score(y_true, clipped),
        "logloss": log_loss(y_true, clipped),
        "brier": brier_score_loss(y_true, clipped),
        "ece": calculate_ece(y_true, clipped),
        "accuracy_0_5": accuracy_score(y_true, clipped >= 0.5),
    }
    row.update(metadata)
    return row


def run_variant(
    data: pd.DataFrame,
    features: list[str],
    experiment_group: str,
    variant: str,
    update_interval: int = DEFAULT_UPDATE_INTERVAL,
    mask_rate: float = DEFAULT_MASK_RATE,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run and evaluate a single ablation variant."""

    y_true, y_prob, dates = walk_forward_predict(data, features, update_interval, mask_rate)
    return evaluate_predictions(
        y_true,
        y_prob,
        dates,
        experiment_group,
        variant,
        metadata or {},
    )


def get_feature_sets() -> dict[str, list[str]]:
    """Create named feature sets for ablation experiments."""

    player_base = PLAYER_PROB_FEATURES + PLAYER_UNCERTAINTY_FEATURES + MATCH_CONTEXT_FEATURES
    player_team = player_base + TEAM_PROB_FEATURES
    context_core = player_team + ROLLING_CORE_FEATURES
    context_full = player_team + ROLLING_FULL_FEATURES
    return {
        "player_base": player_base,
        "player_team_base": player_team,
        "context_core": context_core,
        "context_full": context_full,
    }


def save_group_plot(results: pd.DataFrame, group: str, metric: str, output_path: Path, title: str) -> None:
    """Save one compact comparison plot for an experiment group."""

    data = results[results["experiment_group"] == group].copy()
    ascending = metric == "logloss"
    data = data.sort_values(metric, ascending=ascending)
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=data, x="variant", y=metric, palette="mako", hue="variant", legend=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8, padding=3)
    values = data[metric].to_numpy()
    margin = max((values.max() - values.min()) * 0.25, 0.002)
    ax.set_ylim(values.min() - margin, values.max() + margin)
    ax.set_title(title, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel(metric.upper())
    ax.tick_params(axis="x", rotation=25)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def write_markdown_summary(results: pd.DataFrame) -> None:
    """Write an autogenerated Markdown summary for quick inspection."""

    output_path = PROJECT_ROOT / "docs" / "whitepaper" / "06g_metamodel_ablation_autogenerated.md"
    lines = [
        "---",
        "type: experiment-report",
        "tags: [whitepaper, metamodel, ablation, odds-sample]",
        "project: inzynierka",
        "date: 2026-04-30",
        "status: autogenerated",
        "---",
        "",
        "# 06g. Autogenerated metamodel ablation experiments",
        "",
        "> [!abstract]",
        "> Wszystkie warianty są trenowane i oceniane wyłącznie na meczach zmapowanych w `odds.csv`. Rolling context jest liczony historycznie z pełnego `golgg_matches.json`, ponieważ wcześniejsze mecze bez kursów mogą być legalną historią sportową.",
        "",
    ]
    for group in results["experiment_group"].unique():
        lines.extend([f"## {group}", ""])
        table = results[results["experiment_group"] == group][
            ["variant", "sample_size", "auc", "logloss", "brier", "ece", "accuracy_0_5"]
        ].copy()
        lines.append(table.to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run all ablation experiments and write artefacts."""

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    base_predictions = load_base_predictions()
    rolling_by_window = generate_rolling_features(CONTEXT_WINDOWS)
    feature_sets = get_feature_sets()
    results: list[dict[str, object]] = []

    default_data = base_predictions.merge(
        rolling_by_window[DEFAULT_CONTEXT_WINDOW], on="golgg_match_id", how="inner"
    ).sort_values("date")

    experiments = [
        (
            "player_vs_team_features",
            "Player-base",
            default_data,
            feature_sets["player_base"],
            DEFAULT_UPDATE_INTERVAL,
            DEFAULT_MASK_RATE,
            {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
        ),
        (
            "player_vs_team_features",
            "Player-base + Team-base",
            default_data,
            feature_sets["player_team_base"],
            DEFAULT_UPDATE_INTERVAL,
            DEFAULT_MASK_RATE,
            {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
        ),
        (
            "context_feature_set",
            "Ratings only",
            default_data,
            feature_sets["player_team_base"],
            DEFAULT_UPDATE_INTERVAL,
            DEFAULT_MASK_RATE,
            {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
        ),
        (
            "context_feature_set",
            "Ratings + core context",
            default_data,
            feature_sets["context_core"],
            DEFAULT_UPDATE_INTERVAL,
            DEFAULT_MASK_RATE,
            {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
        ),
        (
            "context_feature_set",
            "Ratings + full context",
            default_data,
            feature_sets["context_full"],
            DEFAULT_UPDATE_INTERVAL,
            DEFAULT_MASK_RATE,
            {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
        ),
    ]

    for interval in [250, 500, 1000, 2000, 4000]:
        experiments.append(
            (
                "update_interval",
                f"update={interval}",
                default_data,
                feature_sets["context_full"],
                interval,
                DEFAULT_MASK_RATE,
                {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": DEFAULT_MASK_RATE},
            )
        )

    for mask_rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        experiments.append(
            (
                "masking_rate",
                f"mask={mask_rate:.1f}",
                default_data,
                feature_sets["context_full"],
                DEFAULT_UPDATE_INTERVAL,
                mask_rate,
                {"context_window": DEFAULT_CONTEXT_WINDOW, "mask_rate": mask_rate},
            )
        )

    for window in CONTEXT_WINDOWS:
        data = base_predictions.merge(rolling_by_window[window], on="golgg_match_id", how="inner").sort_values("date")
        experiments.append(
            (
                "context_window",
                f"window={window}",
                data,
                feature_sets["context_full"],
                DEFAULT_UPDATE_INTERVAL,
                DEFAULT_MASK_RATE,
                {"context_window": window, "mask_rate": DEFAULT_MASK_RATE},
            )
        )

    for group, variant, data, features, interval, mask_rate, metadata in tqdm(experiments, desc="Experiments"):
        row = run_variant(
            data=data,
            features=features,
            experiment_group=group,
            variant=variant,
            update_interval=interval,
            mask_rate=mask_rate,
            metadata={**metadata, "update_interval": interval, "feature_count": len(features)},
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(ASSETS_DIR / "metamodel_ablation_results.csv", index=False)
    for group in results_df["experiment_group"].unique():
        save_group_plot(
            results_df,
            group,
            "auc",
            ASSETS_DIR / f"{group}_auc.png",
            f"{group} — AUC",
        )
        save_group_plot(
            results_df,
            group,
            "logloss",
            ASSETS_DIR / f"{group}_logloss.png",
            f"{group} — LogLoss",
        )
    write_markdown_summary(results_df)
    print(f"Saved ablation results to: {ASSETS_DIR}")
    print(results_df.sort_values(["experiment_group", "logloss"]).to_string(index=False))


if __name__ == "__main__":
    main()
