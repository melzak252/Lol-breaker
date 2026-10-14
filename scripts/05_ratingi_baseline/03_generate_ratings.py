"""Generate pre-match rating predictions for all GOL.GG matches."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ratings.manager import RatingManager
from src.utils.golgg_schema import (
    best_of,
    game_score_for_match_team1,
    games,
    players1,
    players2,
    team1_name,
    team1_id,
    team2_name,
    team2_id,
)


def add_to_y_predicts(
    y_predicts: dict[str, list[object]],
    predictions: dict[str, object],
    bon: int | None,
    match_date: date,
    match_id: object,
) -> None:
    """Append one match prediction row to a column-oriented dictionary.

    Args:
        y_predicts: Column-oriented accumulator for prediction rows.
        predictions: Rating predictions and contextual features for the match.
        bon: Best-of-N format value.
        match_date: Match date.
        match_id: GOL.GG match identifier.
    """

    predictions["BoN"] = bon
    predictions["date"] = match_date
    predictions["golgg_match_id"] = match_id

    current_len = 0
    if y_predicts:
        current_len = len(next(iter(y_predicts.values())))

    for key, value in predictions.items():
        if key not in y_predicts:
            y_predicts[key] = [None] * current_len
        y_predicts[key].append(value)

    for key in y_predicts:
        if key not in predictions:
            y_predicts[key].append(None)


def create_rating_manager() -> RatingManager:
    """Family parameters used to produce the legacy canonical79 rating inputs."""
    return RatingManager({
        "elo": {"k_player": 48, "k_team": 64},
        "ts": {"mu": 25.0, "sigma": 8.333, "beta": 4.16, "tau": 0.25},
        "os": {"mu": 25.0, "sigma": 3.5},
        "pl": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
        "tm": {"mu": 25.0, "sigma": 8.333, "beta": 18.75, "tau": 0.05},
    })


def main() -> None:
    """Generate and save leakage-safe pre-match rating predictions."""

    print("Running 03_generate_ratings.py...")
    print("Loading golgg_matches.json...")
    with (PROJECT_ROOT / "data" / "golgg_matches.json").open(
        "r", encoding="utf-8"
    ) as handle:
        matches = json.load(handle)

    matches = [match for match in matches if not match["draw"]]
    matches = sorted(
        matches,
        key=lambda match: (
            date.fromisoformat(match["date"]),
            int(match.get("match_id") or 0),
        ),
    )
    print(f"Matches without draws: {len(matches)}")

    # Initialize RatingManager with the best family-level parameters found in
    # scripts/05_ratingi_baseline/05c_optimize_rating_families.py.
    # Glicko-2 defaults correspond to the best grid-search setting:
    # RD=350, volatility=0.06, period_days=7.
    manager = create_rating_manager()

    y_predicts: dict[str, list[object]] = {}
    y_trues = []

    print("Processing matches and calculating ratings...")
    for match in tqdm(matches, desc="Processing matches"):
        t1 = team1_id(match)
        t2 = team2_id(match)
        match_date = date.fromisoformat(match["date"])
        players_1 = players1(match)
        players_2 = players2(match)
        scores = []

        if not players_1 or not players_2:
            continue

        match_info = manager.update_before_match(
            t1, t2, players_1, players_2, match_date
        )
        predictions = manager.predict_match(t1, t2, players_1, players_2)

        # Add contextual features
        predictions["days_since_last_1"] = match_info["days_since_last_1"]
        predictions["days_since_last_2"] = match_info["days_since_last_2"]
        predictions["days_diff"] = (
            match_info["days_since_last_1"] - match_info["days_since_last_2"]
        )
        predictions["team1_id"] = t1
        predictions["team2_id"] = t2
        predictions["team1_name"] = team1_name(match)
        predictions["team2_name"] = team2_name(match)

        for game in games(match):
            score_1 = game_score_for_match_team1(match, game)
            score_2 = 1 - score_1

            scores.append(score_1)
            manager.update_after_game(t1, t2, players_1, players_2, score_1, score_2)

        if not scores:
            continue

        add_to_y_predicts(
            y_predicts,
            predictions,
            best_of(match),
            match_date,
            match["match_id"],
        )
        y_trues.append(int(sum(scores) > len(scores) / 2))

        manager.update_after_match(t1, t2, players_1, players_2, scores)

    print("Saving results to golgg_y_predicts.csv...")
    df = pd.DataFrame(y_predicts)
    df["y_true"] = y_trues
    df.to_csv(PROJECT_ROOT / "data" / "golgg_y_predicts.csv", index=False)
    print("03_generate_ratings.py completed successfully.")


if __name__ == "__main__":
    main()
