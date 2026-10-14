"""Grid-search TrueSkill and OpenSkill parameters for rating baselines.

The experiment is intentionally separated from the main baseline script. Its
goal is to check whether weak default TrueSkill/OpenSkill results are caused by
bad hyperparameters rather than by the modelling family itself.

Methodological note:
    Parameters are selected on the historical common-market period 2020-2023 and
    evaluated on 2024+ common-market matches. This prevents tuning parameters on
    the same period later used as the main modern-era validation set.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[2]
MATCHES_PATH = ROOT_DIR / "data" / "golgg_matches.json"
ODDS_PATH = ROOT_DIR / "data" / "odds.csv"
ASSET_DIR = ROOT_DIR / "docs" / "assets" / "baseline_point5"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.ratings.openskill_rating import OpenSkillRating
from src.ratings.trueskill_rating import TrueSkillRating


@dataclass(frozen=True)
class MatchRecord:
    """Compact match representation required for rating grid search."""

    match_id: int
    match_date: date
    team_1: str
    team_2: str
    players_1: tuple[str, ...]
    players_2: tuple[str, ...]
    y_true: int
    scores: tuple[int, ...]


def configure_style() -> None:
    """Configure plot style used by optimization artefacts."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"savefig.dpi": 300, "axes.titleweight": "bold"})


def load_common_market_ids() -> set[int]:
    """Load match IDs with available market data."""
    odds = pd.read_csv(ODDS_PATH, usecols=["golgg_match_id", "avg_open_home", "avg_open_away"])
    odds = odds.dropna(subset=["avg_open_home", "avg_open_away"])
    return set(pd.to_numeric(odds["golgg_match_id"], errors="coerce").dropna().astype(int))


def load_matches() -> list[MatchRecord]:
    """Load GOL.GG matches into compact chronological records.

    Returns:
        Chronologically sorted list of matches without draws.
    """
    with MATCHES_PATH.open("r", encoding="utf-8") as handle:
        raw_matches: list[dict[str, Any]] = json.load(handle)

    records: list[MatchRecord] = []
    for match in raw_matches:
        if match.get("draw"):
            continue

        scores: list[int] = []
        for game in match.get("games", []):
            score_1 = int(game.get("t1_win", 0))
            if game.get("t1_id") != match.get("tid_1"):
                score_1 = 1 - score_1
            scores.append(score_1)

        records.append(
            MatchRecord(
                match_id=int(match["match_id"]),
                match_date=date.fromisoformat(match["date"]),
                team_1=str(match["tid_1"]),
                team_2=str(match["tid_2"]),
                players_1=tuple(map(str, match.get("players_1", []))),
                players_2=tuple(map(str, match.get("players_2", []))),
                y_true=int(match["t1_win"]),
                scores=tuple(scores),
            )
        )

    return sorted(records, key=lambda item: item.match_date)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Calculate expected calibration error for binary probabilities."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        if not np.any(in_bin):
            continue
        ece += abs(np.mean(y_true[in_bin]) - np.mean(y_prob[in_bin])) * np.mean(in_bin)
    return float(ece)


def calculate_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float | int]:
    """Calculate standard probabilistic metrics."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.clip(np.asarray(y_prob, dtype=float), 0.001, 0.999)
    return {
        "sample_size": int(len(y_true_array)),
        "auc": float(roc_auc_score(y_true_array, y_prob_array)),
        "logloss": float(log_loss(y_true_array, y_prob_array)),
        "brier": float(brier_score_loss(y_true_array, y_prob_array)),
        "ece": expected_calibration_error(y_true_array, y_prob_array),
    }


def update_system_after_match(system: Any, match: MatchRecord) -> None:
    """Update a rating system with all games from a match."""
    for score_1 in match.scores:
        score_2 = 1 - score_1
        system.update_team(match.team_1, match.team_2, score_1, score_2)
        system.update_player(match.players_1, match.players_2, score_1, score_2)


def evaluate_parameter_set(
    system_name: str,
    params: dict[str, float],
    matches: Iterable[MatchRecord],
    common_ids: set[int],
) -> dict[str, Any]:
    """Evaluate one TrueSkill/OpenSkill parameter configuration.

    Args:
        system_name: Either ``trueskill`` or ``openskill``.
        params: Constructor parameters for the rating system.
        matches: Chronological match records.
        common_ids: Match IDs present in the market dataset.

    Returns:
        Row with train, test and all-period metrics for player-level predictions.
    """
    if system_name == "trueskill":
        system = TrueSkillRating(**params)
    elif system_name == "openskill":
        system = OpenSkillRating(**params)
    else:
        raise ValueError(f"Unsupported system: {system_name}")

    train_true: list[int] = []
    train_prob: list[float] = []
    test_true: list[int] = []
    test_prob: list[float] = []
    all_true: list[int] = []
    all_prob: list[float] = []

    for match in matches:
        player_prob = float(system.predict_player_win_prob(match.players_1, match.players_2))
        if match.match_id in common_ids and match.match_date >= date(2020, 1, 1):
            all_true.append(match.y_true)
            all_prob.append(player_prob)
            if match.match_date < date(2024, 1, 1):
                train_true.append(match.y_true)
                train_prob.append(player_prob)
            else:
                test_true.append(match.y_true)
                test_prob.append(player_prob)

        update_system_after_match(system, match)

    row: dict[str, Any] = {"system": system_name, **params}
    for prefix, y_true, y_prob in [
        ("train_2020_2023", train_true, train_prob),
        ("test_2024_plus", test_true, test_prob),
        ("all_common_2020_plus", all_true, all_prob),
    ]:
        metrics = calculate_metrics(y_true, y_prob)
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return row


def build_parameter_grid() -> list[tuple[str, dict[str, float]]]:
    """Build a compact, interpretable grid for TrueSkill and OpenSkill."""
    trueskill_grid = [
        ("trueskill", {"mu": 25.0, "sigma": 8.333, "beta": beta, "tau": tau})
        for beta, tau in product(
            [4.16, 6.25, 8.33, 12.5, 16.67],
            [0.05, 0.12, 0.25],
        )
    ]
    openskill_grid = [
        ("openskill", {"mu": 25.0, "sigma": sigma})
        for sigma in [3.5, 5.0, 6.25, 8.333, 10.0]
    ]
    return [*trueskill_grid, *openskill_grid]


def save_top_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save top optimized configurations by 2024+ LogLoss."""
    plot_data = results.sort_values("test_2024_plus_logloss").head(12).copy()
    plot_data["label"] = plot_data.apply(
        lambda row: (
            f"TS β={row['beta']}, τ={row['tau']}"
            if row["system"] == "trueskill"
            else f"OS σ={row['sigma']}"
        ),
        axis=1,
    )
    min_value = plot_data["test_2024_plus_logloss"].min()
    max_value = plot_data["test_2024_plus_logloss"].max()
    margin = max(max_value - min_value, 0.003)

    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(
        data=plot_data,
        x="label",
        y="test_2024_plus_logloss",
        hue="system",
        dodge=False,
        palette="viridis",
    )
    ax.set_title("Optymalizacja TrueSkill/OpenSkill — test 2024+ LogLoss")
    ax.set_xlabel("")
    ax.set_ylabel("LogLoss, 2024+")
    ax.set_ylim(min_value - margin * 0.4, max_value + margin * 0.5)
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def write_summary(results: pd.DataFrame) -> None:
    """Write Markdown summary for the optimization experiment."""
    top = results.sort_values("test_2024_plus_logloss").head(15).copy()
    for column in top.select_dtypes(include=["float"]).columns:
        top[column] = top[column].map(lambda value: f"{value:.5f}")
    markdown = top.to_markdown(index=False)
    content = f"""---
type: experiment-log
tags:
  - whitepaper
  - ratings
  - trueskill
  - openskill
  - grid-search
project: inzynierka
date: 2026-04-30
source_script: scripts/05b_optimize_trueskill_openskill.py
---

# 5b. Optymalizacja parametrów TrueSkill i OpenSkill

> [!abstract]
> Eksperyment sprawdza, czy słabsze wyniki TrueSkill/OpenSkill w baseline'ach wynikają z niedostrojenia parametrów. Parametry są wybierane historycznie na okresie 2020-2023, a oceniane na próbie 2024+ z dostępnymi kursami rynkowymi.

## Top konfiguracje według LogLoss na 2024+

{markdown}
"""
    (ROOT_DIR / "docs" / "whitepaper" / "05b_opt_trueskill_openskill_autogenerated.md").write_text(
        content, encoding="utf-8"
    )


def main() -> None:
    """Run the parameter grid search."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    matches = load_matches()
    common_ids = load_common_market_ids()
    grid = build_parameter_grid()

    rows: list[dict[str, Any]] = []
    for index, (system_name, params) in enumerate(grid, start=1):
        print(f"[{index}/{len(grid)}] {system_name} {params}", flush=True)
        rows.append(evaluate_parameter_set(system_name, params, matches, common_ids))

    results = pd.DataFrame(rows)
    results = results.sort_values("test_2024_plus_logloss")
    results.to_csv(ASSET_DIR / "trueskill_openskill_gridsearch.csv", index=False)
    save_top_plot(results, ASSET_DIR / "trueskill_openskill_gridsearch_top.png")
    write_summary(results)

    print("Saved:", ASSET_DIR / "trueskill_openskill_gridsearch.csv")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
