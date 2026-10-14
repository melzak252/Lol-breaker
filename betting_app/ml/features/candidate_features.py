"""Candidate features for EXP-040: side advantage, patch decay, and roster continuity."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any

import numpy as np


def compute_side_advantage(
    game1_blue_team: str | None,
    team_a: str,
    team_b: str,
) -> float:
    """Compute Game 1 side advantage for team A relative to team B.

    Returns:
        +1.0 if team_a has Blue side in Game 1,
        -1.0 if team_b has Blue side in Game 1,
         0.0 if unknown or neither matches.
    """
    if not game1_blue_team:
        return 0.0

    blue_norm = game1_blue_team.strip().casefold()
    a_norm = team_a.strip().casefold()
    b_norm = team_b.strip().casefold()

    if a_norm == b_norm:
        return 0.0

    if blue_norm == a_norm:
        return 1.0
    if blue_norm == b_norm:
        return -1.0
    return 0.0


def compute_series_side_priority(
    higher_seed_team: str | None,
    team_a: str,
    team_b: str,
) -> float:
    """Compute series side selection priority for team A relative to team B.

    In multi-game series (Bo3/Bo5), the higher seed chooses side in Game 1
    (and decider game where applicable).

    Returns:
        +1.0 if team_a has side selection priority (higher seed),
        -1.0 if team_b has side selection priority,
         0.0 if unknown or neither matches.
    """
    if not higher_seed_team:
        return 0.0

    seed_norm = higher_seed_team.strip().casefold()
    a_norm = team_a.strip().casefold()
    b_norm = team_b.strip().casefold()

    if a_norm == b_norm:
        return 0.0

    if seed_norm == a_norm:
        return 1.0
    if seed_norm == b_norm:
        return -1.0
    return 0.0


def _parse_patch_version(patch_str: str) -> tuple[int, int]:
    """Parse major and minor version numbers from a patch string.

    Examples:
        "14.10" -> (14, 10)
        "14.9.1" -> (14, 9)
        "v14.10" -> (14, 10)
    """
    match = re.search(r"(\d+)\.(\d+)", str(patch_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    nums = re.findall(r"\d+", str(patch_str))
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    if len(nums) == 1:
        return int(nums[0]), 0
    return (0, 0)


def _is_older_patch(game_patch: str, target_patch: str) -> bool:
    """Determine whether game_patch represents an older major/minor patch than target_patch."""
    game_ver = _parse_patch_version(game_patch)
    target_ver = _parse_patch_version(target_patch)

    if game_ver != (0, 0) and target_ver != (0, 0):
        return game_ver < target_ver

    # Fallback to string inequality if version cannot be parsed
    return str(game_patch).strip() != str(target_patch).strip()


def compute_patch_decay_weights(
    game_dates: list[datetime],
    game_patches: list[str],
    target_date: datetime,
    target_patch: str,
    half_life_days: float = 21.0,
    old_patch_multiplier: float = 0.4,
) -> np.ndarray:
    """Compute sample weights combining exponential time-decay and patch adaptation lag penalty.

    Args:
        game_dates: Historical game kickoff timestamps.
        game_patches: Historical game patch versions (e.g. "14.10").
        target_date: Kickoff timestamp of the upcoming target match.
        target_patch: Target game patch version.
        half_life_days: Exponential decay half-life in days (default 21.0).
        old_patch_multiplier: Weight multiplier for games played on an older major/minor patch.

    Returns:
        1D numpy array of float weights corresponding to each historical game.
    """
    if len(game_dates) != len(game_patches):
        raise ValueError(
            f"game_dates ({len(game_dates)}) and game_patches ({len(game_patches)}) must have the same length"
        )

    if not game_dates:
        return np.array([], dtype=float)

    if half_life_days <= 0.0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")

    decay_constant = math.log(2.0) / half_life_days

    # Ensure target_date has timezone handling consistent with game_dates
    target_is_aware = target_date.tzinfo is not None

    weights: list[float] = []
    for g_date, g_patch in zip(game_dates, game_patches, strict=True):
        # Timezone reconciliation
        dt = g_date
        if target_is_aware and dt.tzinfo is None:
            dt = dt.replace(tzinfo=target_date.tzinfo)
        elif not target_is_aware and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        elapsed_seconds = (target_date - dt).total_seconds()
        elapsed_days = max(0.0, elapsed_seconds / 86400.0)

        time_decay = math.exp(-decay_constant * elapsed_days)

        # Patch penalty
        patch_multiplier = (
            old_patch_multiplier if _is_older_patch(g_patch, target_patch) else 1.0
        )

        weights.append(time_decay * patch_multiplier)

    return np.array(weights, dtype=float)


def _normalize_player_id(player: int | str) -> str:
    """Normalize player identifier for reliable comparison across representations."""
    return str(player).strip().casefold()


def compute_roster_continuity(
    current_lineup: list[int | str],
    past_lineups: list[list[int | str]],
    max_games: int = 20,
) -> dict[str, float]:
    """Compute roster continuity metrics for the current lineup given historical lineups.

    Args:
        current_lineup: 5-player lineup starting the target match.
        past_lineups: Historical lineups (ordered list of past match lineups).
        max_games: Number of recent games to consider for the rolling window.

    Returns:
        Dictionary containing:
        - `lineup_cohesion`: average pairwise Jaccard similarity across the last max_games.
        - `substitute_count`: count of players in current_lineup who played fewer than 3 games in past_lineups.
        - `games_together`: average number of past games played together by pairs in the roster.
    """
    if not current_lineup:
        return {
            "lineup_cohesion": 0.0,
            "substitute_count": 0.0,
            "games_together": 0.0,
        }

    norm_current = [_normalize_player_id(p) for p in current_lineup]
    curr_set = set(norm_current)

    # Normalize past lineups
    norm_past_lineups = [
        {_normalize_player_id(p) for p in lineup}
        for lineup in past_lineups
    ]

    # 1. Lineup cohesion: average pairwise Jaccard similarity with current lineup across last max_games
    recent_past = (
        norm_past_lineups[-max_games:]
        if max_games > 0 and len(norm_past_lineups) > max_games
        else norm_past_lineups
    )

    if not recent_past:
        lineup_cohesion = 0.0
    else:
        jaccards: list[float] = []
        for past_set in recent_past:
            union_len = len(curr_set | past_set)
            if union_len == 0:
                jaccards.append(1.0 if not curr_set else 0.0)
            else:
                jaccards.append(len(curr_set & past_set) / union_len)
        lineup_cohesion = float(np.mean(jaccards))

    # 2. Substitute count: count of players in current_lineup who played fewer than 3 games in past_lineups
    player_appearances: dict[str, int] = {p: 0 for p in curr_set}
    for past_set in norm_past_lineups:
        for p in curr_set:
            if p in past_set:
                player_appearances[p] += 1

    # Count distinct players in current lineup with < 3 appearances
    substitute_count = float(sum(1 for p in curr_set if player_appearances[p] < 3))

    # 3. Games together: average number of past games played together by pairs in the roster
    curr_unique = list(curr_set)
    num_players = len(curr_unique)
    if num_players < 2:
        games_together = 0.0
    else:
        pair_games: list[float] = []
        for i in range(num_players):
            for j in range(i + 1, num_players):
                p1, p2 = curr_unique[i], curr_unique[j]
                count = sum(1 for past_set in norm_past_lineups if p1 in past_set and p2 in past_set)
                pair_games.append(float(count))
        games_together = float(np.mean(pair_games))

    return {
        "lineup_cohesion": round(lineup_cohesion, 6),
        "substitute_count": float(substitute_count),
        "games_together": round(games_together, 6),
    }


def _clean_zero(val: float) -> float:
    """Normalize -0.0 to 0.0 for strict float equality in test assertions."""
    return 0.0 if abs(val) < 1e-12 else val


def assemble_symmetric_candidate_features(
    team_a_stats: dict[str, float],
    team_b_stats: dict[str, float],
    context: dict[str, Any],
) -> dict[str, float]:
    """Assemble symmetric candidate feature vector for EXP-040.

    Guarantees exact antisymmetry under team swapping:
    features(team_b_stats, team_a_stats, swapped_context) == -features(team_a_stats, team_b_stats, context).

    Args:
        team_a_stats: Numerical performance stats for team A.
        team_b_stats: Numerical performance stats for team B.
        context: Context dictionary containing match metadata such as:
            - `team_a` / `team_a_name`: str
            - `team_b` / `team_b_name`: str
            - `game1_blue_team`: str | None
            - `higher_seed_team`: str | None
            - `team_a_current_lineup` or `team_a_lineup`: list[int | str]
            - `team_b_current_lineup` or `team_b_lineup`: list[int | str]
            - `team_a_past_lineups` or `past_lineups_a`: list[list[int | str]]
            - `team_b_past_lineups` or `past_lineups_b`: list[list[int | str]]
            - `team_a_game_dates`, `team_a_game_patches`: list
            - `team_b_game_dates`, `team_b_game_patches`: list
            - `target_date`: datetime
            - `target_patch`: str

    Returns:
        Dictionary mapping feature names to their float values.
    """
    features: dict[str, float] = {}

    # 1. Team stats differentials: Delta_X = X_A - X_B
    all_stat_keys = sorted(set(team_a_stats.keys()) | set(team_b_stats.keys()))
    for key in all_stat_keys:
        val_a = float(team_a_stats.get(key, 0.0))
        val_b = float(team_b_stats.get(key, 0.0))
        features[f"delta_{key}"] = _clean_zero(val_a - val_b)

    # 2. Side advantage and series side priority
    team_a_name = str(context.get("team_a") or context.get("team_a_name") or "team_a")
    team_b_name = str(context.get("team_b") or context.get("team_b_name") or "team_b")
    game1_blue_team = context.get("game1_blue_team")
    higher_seed_team = context.get("higher_seed_team")

    side_adv = compute_side_advantage(game1_blue_team, team_a_name, team_b_name)
    side_prio = compute_series_side_priority(higher_seed_team, team_a_name, team_b_name)

    features["side_advantage"] = _clean_zero(side_adv)
    features["series_side_priority"] = _clean_zero(side_prio)

    # 3. Roster continuity features
    lineup_a = context.get("team_a_current_lineup") or context.get("team_a_lineup")
    past_a = context.get("team_a_past_lineups") or context.get("past_lineups_a") or []
    lineup_b = context.get("team_b_current_lineup") or context.get("team_b_lineup")
    past_b = context.get("team_b_past_lineups") or context.get("past_lineups_b") or []

    if lineup_a is not None or lineup_b is not None:
        cont_a = compute_roster_continuity(lineup_a or [], past_a)
        cont_b = compute_roster_continuity(lineup_b or [], past_b)

        features["delta_lineup_cohesion"] = _clean_zero(
            cont_a["lineup_cohesion"] - cont_b["lineup_cohesion"]
        )
        # substitute_count difference: A having more substitutes is a disruption,
        # so Delta_substitutes = sub_A - sub_B is antisymmetric.
        features["delta_substitute_count"] = _clean_zero(
            cont_a["substitute_count"] - cont_b["substitute_count"]
        )
        features["delta_games_together"] = _clean_zero(
            cont_a["games_together"] - cont_b["games_together"]
        )

    # 4. Patch decay weights features (if provided in context)
    target_date = context.get("target_date")
    target_patch = context.get("target_patch")
    dates_a = context.get("team_a_game_dates")
    patches_a = context.get("team_a_game_patches")
    dates_b = context.get("team_b_game_dates")
    patches_b = context.get("team_b_game_patches")

    if target_date is not None and target_patch is not None and (dates_a is not None or dates_b is not None):
        weights_a = (
            compute_patch_decay_weights(dates_a, patches_a or [], target_date, target_patch)
            if dates_a
            else np.array([], dtype=float)
        )
        weights_b = (
            compute_patch_decay_weights(dates_b, patches_b or [], target_date, target_patch)
            if dates_b
            else np.array([], dtype=float)
        )

        sum_w_a = float(np.sum(weights_a)) if len(weights_a) > 0 else 0.0
        sum_w_b = float(np.sum(weights_b)) if len(weights_b) > 0 else 0.0
        mean_w_a = float(np.mean(weights_a)) if len(weights_a) > 0 else 0.0
        mean_w_b = float(np.mean(weights_b)) if len(weights_b) > 0 else 0.0

        features["delta_effective_sample_size"] = _clean_zero(sum_w_a - sum_w_b)
        features["delta_mean_patch_decay_weight"] = _clean_zero(mean_w_a - mean_w_b)

    return features
