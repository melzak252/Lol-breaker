"""Run sanity checks for rating-family implementations.

The script verifies two things:

1. Toy orientation/update checks: after repeated wins by team A, every rating
   family should assign a higher win probability to A than to B, and swapped
   probabilities should approximately sum to one.
2. Calibration diagnostics: weak raw LogLoss for some families may be caused by
   overconfident probability scaling rather than wrong ordering. A simple Platt
   and isotonic calibration check is run on 2021-2023 and evaluated on 2024+.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.analysis.probability_metrics import calculate_ece

from src.ratings.elo import EloRating
from src.ratings.glicko import GlickoRating
from src.ratings.openskill_rating import OpenSkillRating
from src.ratings.plackett_luce import PlackettLuceRating
from src.ratings.thurstone import ThurstoneRating
from src.ratings.trueskill_rating import TrueSkillRating

OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "rating_implementation_sanity"
RATING_COLUMNS = [
    "player_elo",
    "player_gl",
    "player_ts",
    "player_os",
    "player_pl",
    "player_tm",
]


def toy_orientation_checks() -> pd.DataFrame:
    """Check that rating updates move probabilities in the correct direction.

    Returns:
        DataFrame with before/after probabilities for each family.
    """
    systems = {
        "Elo": EloRating(),
        "Glicko-2": GlickoRating(),
        "TrueSkill": TrueSkillRating(draw_probability=0.0),
        "OpenSkill": OpenSkillRating(),
        "Plackett-Luce": PlackettLuceRating(),
        "Thurstone-Mosteller": ThurstoneRating(),
    }
    players_a = [f"a{i}" for i in range(5)]
    players_b = [f"b{i}" for i in range(5)]
    rows: list[dict[str, float | str | bool]] = []
    for family, system in systems.items():
        if family == "Glicko-2":
            system.update_rd_before_match("A", "B", players_a, players_b, date(2020, 1, 1))
        before = float(system.predict_player_win_prob(players_a, players_b))
        for _ in range(20):
            system.update_player(players_a, players_b, 1, 0)
        after_ab = float(system.predict_player_win_prob(players_a, players_b))
        after_ba = float(system.predict_player_win_prob(players_b, players_a))
        rows.append(
            {
                "family": family,
                "before_a_vs_b": before,
                "after_a_vs_b": after_ab,
                "after_b_vs_a": after_ba,
                "swapped_sum": after_ab + after_ba,
                "passes_direction_check": after_ab > before and after_ba < before,
            }
        )
    return pd.DataFrame(rows)


def load_rating_common_sample() -> pd.DataFrame:
    """Load rating predictions restricted to the odds-mapped thesis sample."""
    ratings = pd.read_csv(PROJECT_ROOT / "data" / "golgg_y_predicts.csv")
    ratings["golgg_match_id"] = ratings["golgg_match_id"].astype(str)
    ratings["date"] = pd.to_datetime(ratings["date"])
    odds = pd.read_csv(
        PROJECT_ROOT / "data" / "odds.csv",
        usecols=["golgg_match_id", "avg_open_home", "avg_open_away"],
    ).dropna(subset=["avg_open_home", "avg_open_away"])
    odds["golgg_match_id"] = odds["golgg_match_id"].astype(str)
    data = ratings.merge(odds[["golgg_match_id"]], on="golgg_match_id", how="inner")
    return data[data["date"] >= pd.Timestamp("2021-01-01")].sort_values("date").reset_index(drop=True)


def evaluate_probability(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Evaluate binary probabilities with thesis metrics."""
    clipped = np.clip(y_prob.astype(float), 0.001, 0.999)
    return {
        "auc": float(roc_auc_score(y_true, clipped)),
        "logloss": float(log_loss(y_true, clipped)),
        "brier": float(brier_score_loss(y_true, clipped)),
        "ece": calculate_ece(y_true, clipped),
        "mean_prob": float(np.mean(clipped)),
        "std_prob": float(np.std(clipped)),
    }


def calibration_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Check whether weak rating families are mainly miscalibrated.

    Args:
        data: Odds-mapped rating prediction frame.

    Returns:
        Metrics for raw, Platt-calibrated and isotonic-calibrated probabilities.
    """
    train = data[data["date"] < pd.Timestamp("2024-01-01")]
    test = data[data["date"] >= pd.Timestamp("2024-01-01")]
    rows: list[dict[str, float | str | int]] = []
    for column in RATING_COLUMNS:
        train_subset = train[[column, "y_true"]].dropna()
        test_subset = test[[column, "y_true"]].dropna()
        y_train = train_subset["y_true"].astype(int).to_numpy()
        y_test = test_subset["y_true"].astype(int).to_numpy()
        x_train = train_subset[[column]].to_numpy()
        x_test = test_subset[[column]].to_numpy()
        raw_prob = test_subset[column].to_numpy()

        platt = LogisticRegression(max_iter=1000)
        platt.fit(x_train, y_train)
        platt_prob = platt.predict_proba(x_test)[:, 1]

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(train_subset[column].to_numpy(), y_train)
        isotonic_prob = isotonic.predict(test_subset[column].to_numpy())

        for method, probabilities in [
            ("raw", raw_prob),
            ("platt", platt_prob),
            ("isotonic", isotonic_prob),
        ]:
            row = {
                "rating_column": column,
                "method": method,
                "train_size": int(len(train_subset)),
                "test_size": int(len(test_subset)),
            }
            row.update(evaluate_probability(y_test, probabilities))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["rating_column", "method"])


def main() -> None:
    """Run rating implementation sanity checks and write artefacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    toy = toy_orientation_checks()
    data = load_rating_common_sample()
    calibration = calibration_diagnostics(data)
    toy.to_csv(OUTPUT_DIR / "rating_toy_orientation_checks.csv", index=False)
    calibration.to_csv(OUTPUT_DIR / "rating_calibration_diagnostics.csv", index=False)

    print("\n=== TOY ORIENTATION CHECKS ===")
    print(toy.to_string(index=False))
    print("\n=== CALIBRATION DIAGNOSTICS 2024+ ===")
    print(calibration.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
