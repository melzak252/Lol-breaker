"""Hierarchical Markov Series Simulator for League of Legends match prediction.

Models non-independent games in Bo1, Bo3, and Bo5 series with dynamic rotating
side selection:
- Game 1: Side selection is chosen by the higher-seeded team (priority team),
  who selects Blue side.
- Game 2: Side selection is chosen by the LOSER of Game 1 (who selects Blue side).
- Game 3 (for Bo3 decider): Side selection is chosen by the LOSER of Game 2
  (or tournament priority rule if configured).
- Game 4/5: Continued alternating loser-picks-side dynamic.

Maintains strict binary symmetry: P(A wins) + P(B wins) == 1.0.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.special import expit, logit

__all__ = [
    "MarkovSeriesSimulator",
    "predict_series_proba",
    "predict_score_distribution",
    "compute_state_probabilities",
    "single_game_proba",
    "series_expected_games",
]


def _safe_game_prob(
    p_neutral: np.ndarray,
    a_is_blue: bool | np.ndarray,
    blue_side_bonus: float,
) -> np.ndarray:
    """Compute single-game win probability for Team A given side assignment.

    logit(p_game) = logit(p_neutral) + (side_bonus if Team A is Blue else -side_bonus).

    Handles boundary values (0.0, 1.0) and NaNs gracefully without warnings.
    """
    p = np.asarray(p_neutral, dtype=float)
    bonus = np.where(a_is_blue, blue_side_bonus, -blue_side_bonus)

    out = np.empty_like(p, dtype=float)
    finite = np.isfinite(p)
    zero_mask = finite & (p <= 0.0)
    one_mask = finite & (p >= 1.0)
    nan_mask = ~finite
    valid_mask = finite & (p > 0.0) & (p < 1.0)

    out[zero_mask] = 0.0
    out[one_mask] = 1.0
    out[nan_mask] = np.nan

    if np.any(valid_mask):
        pv = p[valid_mask]
        bv = bonus if np.ndim(bonus) == 0 else bonus[valid_mask]
        out[valid_mask] = expit(logit(pv) + bv)

    return out


def _generate_series_paths(
    best_of: int,
    priority: bool,
    decider_rule: str = "loser_picks",
) -> list[tuple[str, bool, list[tuple[str, bool]]]]:
    """Generate all distinct game outcome paths in the series Markov tree.

    Returns a list of tuples:
        (score_str, team_a_won_series, factors)
    where factors is a list of (winner_str, team_a_is_blue).
    """
    needed = (best_of + 1) // 2
    leaves: list[tuple[str, bool, list[tuple[str, bool]]]] = []

    def dfs(
        history: tuple[str, ...],
        factors: list[tuple[str, bool]],
    ) -> None:
        wins_a = history.count("A")
        wins_b = history.count("B")
        if wins_a == needed or wins_b == needed:
            leaves.append((f"{wins_a}-{wins_b}", wins_a == needed, factors))
            return

        k = len(history) + 1
        if k == 1:
            a_is_blue = priority
        elif decider_rule == "priority_picks" and k == best_of:
            a_is_blue = priority
        else:
            # Loser of previous game selects Blue side
            # If last winner was A -> loser was B -> B selects Blue -> A is Red
            # If last winner was B -> loser was A -> A selects Blue -> A is Blue
            a_is_blue = (history[-1] == "B")

        dfs(history + ("A",), factors + [("A", a_is_blue)])
        dfs(history + ("B",), factors + [("B", a_is_blue)])

    dfs((), [])
    return leaves


def _compute_raw_series_proba(
    p_arr: np.ndarray,
    priority: bool,
    best_of: int,
    blue_side_bonus: float,
    decider_rule: str,
) -> np.ndarray:
    """Compute raw series win probability for Team A over an array of neutral probabilities."""
    leaves = _generate_series_paths(best_of, priority, decider_rule)
    p_blue = _safe_game_prob(p_arr, True, blue_side_bonus)
    p_red = _safe_game_prob(p_arr, False, blue_side_bonus)
    q_blue = 1.0 - p_blue
    q_red = 1.0 - p_red

    p_win_series = np.zeros_like(p_arr, dtype=float)
    nan_mask = ~np.isfinite(p_arr)

    for _score, a_wins, factors in leaves:
        if not a_wins:
            continue
        p_path = np.ones_like(p_arr, dtype=float)
        for winner, a_is_blue in factors:
            pg = p_blue if a_is_blue else p_red
            qg = q_blue if a_is_blue else q_red
            p_path = p_path * (pg if winner == "A" else qg)
        p_win_series = p_win_series + p_path

    return np.where(nan_mask, np.nan, p_win_series)


def _predict_series_proba_core(
    p_neutral_a: float | np.ndarray,
    team_a_has_game1_priority: bool | np.ndarray,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> np.ndarray:
    """Vectorized and symmetrized series win probability calculation."""
    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError(f"best_of must be an odd positive integer (e.g. 1, 3, 5), got {best_of}")
    if decider_rule not in ("loser_picks", "priority_picks"):
        raise ValueError(f"decider_rule must be 'loser_picks' or 'priority_picks', got '{decider_rule}'")
    if not np.isfinite(blue_side_bonus) or blue_side_bonus < 0.0:
        raise ValueError(f"blue_side_bonus must be a finite non-negative float, got {blue_side_bonus}")

    p_b, prio_b = np.broadcast_arrays(p_neutral_a, team_a_has_game1_priority)
    p_f = np.asarray(p_b, dtype=float)
    prio_bool = np.asarray(prio_b, dtype=bool)

    # Canonical orientation: evaluate uniquely from the perspective where
    # (p > 0.5) or (p == 0.5 and priority), ensuring exact bit-level binary symmetry.
    is_canonical = (p_f > 0.5) | ((p_f == 0.5) & prio_bool)
    p_canon = np.where(is_canonical, p_f, 1.0 - p_f)
    prio_canon = np.where(is_canonical, prio_bool, ~prio_bool)

    raw_true = _compute_raw_series_proba(p_canon, True, best_of, blue_side_bonus, decider_rule)
    raw_false = _compute_raw_series_proba(p_canon, False, best_of, blue_side_bonus, decider_rule)
    raw_canon = np.where(prio_canon, raw_true, raw_false)

    p_canon_opp = 1.0 - p_canon
    raw_true_opp = _compute_raw_series_proba(p_canon_opp, True, best_of, blue_side_bonus, decider_rule)
    raw_false_opp = _compute_raw_series_proba(p_canon_opp, False, best_of, blue_side_bonus, decider_rule)
    raw_canon_opp = np.where(~prio_canon, raw_true_opp, raw_false_opp)

    val_canon = 0.5 * (raw_canon + (1.0 - raw_canon_opp))
    val_final = np.where(is_canonical, val_canon, 1.0 - val_canon)
    val_final = np.where(~np.isfinite(p_f), np.nan, val_final)
    return np.asarray(val_final, dtype=float)


def _predict_score_distribution_core(
    p_neutral_a: float,
    team_a_has_game1_priority: bool,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> dict[str, float]:
    """Compute exact analytic score distribution summing to 1.0 with binary symmetry."""
    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError(f"best_of must be an odd positive integer (e.g. 1, 3, 5), got {best_of}")
    if decider_rule not in ("loser_picks", "priority_picks"):
        raise ValueError(f"decider_rule must be 'loser_picks' or 'priority_picks', got '{decider_rule}'")
    if not np.isfinite(blue_side_bonus) or blue_side_bonus < 0.0:
        raise ValueError(f"blue_side_bonus must be a finite non-negative float, got {blue_side_bonus}")

    p_val = float(p_neutral_a)
    if not np.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
        raise ValueError(f"p_neutral_a must be a valid probability in [0.0, 1.0], got {p_neutral_a}")

    def _leaf_scores(p_in: float, prio_in: bool) -> dict[str, float]:
        leaves = _generate_series_paths(best_of, prio_in, decider_rule)
        pb = float(_safe_game_prob(np.array(p_in), True, blue_side_bonus))
        pr = float(_safe_game_prob(np.array(p_in), False, blue_side_bonus))
        qb = 1.0 - pb
        qr = 1.0 - pr

        scores: dict[str, float] = {}
        for score, _a_wins, factors in leaves:
            prob = 1.0
            for winner, a_is_blue in factors:
                pg = pb if a_is_blue else pr
                qg = qb if a_is_blue else qr
                prob *= pg if winner == "A" else qg
            scores[score] = scores.get(score, 0.0) + prob
        return scores

    # Compute raw leaf scores for Team A and opposing Team B
    scores_a = _leaf_scores(p_val, bool(team_a_has_game1_priority))
    scores_b = _leaf_scores(1.0 - p_val, not bool(team_a_has_game1_priority))

    needed = (best_of + 1) // 2
    # Canonical score ordering: Team A sweeps down to Team B sweeps
    ordered_keys: list[str] = []
    # Team A wins: (needed, 0) up to (needed, needed - 1)
    for wb in range(needed):
        ordered_keys.append(f"{needed}-{wb}")
    # Team B wins: (needed - 1, needed) down to (0, needed)
    for wa in range(needed - 1, -1, -1):
        ordered_keys.append(f"{wa}-{needed}")

    # Symmetrize score probabilities
    symmetrized: dict[str, float] = {}
    for score in ordered_keys:
        wa, wb = score.split("-")
        rev_score = f"{wb}-{wa}"
        prob_a = scores_a.get(score, 0.0)
        prob_b = scores_b.get(rev_score, 0.0)
        symmetrized[score] = 0.5 * (prob_a + prob_b)

    # Normalize to 1.0
    total = sum(symmetrized.values())
    if total > 0.0:
        symmetrized = {k: float(v / total) for k, v in symmetrized.items()}

    return symmetrized


def _compute_state_probabilities_core(
    p_neutral_a: float,
    team_a_has_game1_priority: bool,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> dict[str, float]:
    """Compute reach probabilities for all score states across the series Markov tree.

    For Bo1: '0-0', '1-0', '0-1'
    For Bo3: '0-0', '1-0', '0-1', '2-0', '1-1', '0-2', '2-1', '1-2'
    For Bo5: Full state tree with dynamic side conditions.
    """
    if best_of <= 0 or best_of % 2 == 0:
        raise ValueError(f"best_of must be an odd positive integer, got {best_of}")
    p_val = float(p_neutral_a)
    if not np.isfinite(p_val) or p_val < 0.0 or p_val > 1.0:
        raise ValueError(f"p_neutral_a must be in [0.0, 1.0], got {p_neutral_a}")

    pb = float(_safe_game_prob(np.array(p_val), True, blue_side_bonus))
    pr = float(_safe_game_prob(np.array(p_val), False, blue_side_bonus))

    needed = (best_of + 1) // 2
    state_probs: dict[str, float] = {"0-0": 1.0}

    # Track active nodes at current series stage: (wins_a, wins_b, last_winner) -> reach_prob
    current_nodes: dict[tuple[int, int, str | None], float] = {(0, 0, None): 1.0}

    while current_nodes:
        next_nodes: dict[tuple[int, int, str | None], float] = {}
        for (wa, wb, last_winner), prob in current_nodes.items():
            if wa == needed or wb == needed:
                # Terminal state, no further transitions
                continue

            k = wa + wb + 1
            if k == 1:
                a_is_blue = bool(team_a_has_game1_priority)
            elif decider_rule == "priority_picks" and k == best_of:
                a_is_blue = bool(team_a_has_game1_priority)
            else:
                a_is_blue = (last_winner == "B")

            pg_a = pb if a_is_blue else pr
            pg_b = 1.0 - pg_a

            node_a = (wa + 1, wb, "A")
            next_nodes[node_a] = next_nodes.get(node_a, 0.0) + prob * pg_a

            node_b = (wa, wb + 1, "B")
            next_nodes[node_b] = next_nodes.get(node_b, 0.0) + prob * pg_b

        for (wa, wb, _), prob in next_nodes.items():
            score = f"{wa}-{wb}"
            state_probs[score] = state_probs.get(score, 0.0) + prob

        current_nodes = next_nodes

    return {k: float(v) for k, v in state_probs.items()}


class _DualSimulatorMethod:
    """Descriptor enabling methods to be invoked on instances, classes, or as standalone callers."""

    def __init__(self, core_fn: Any) -> None:
        self.core_fn = core_fn

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            # Called as MarkovSeriesSimulator.method(...)
            def _class_call(
                p_neutral_a: Any,
                team_a_has_game1_priority: Any,
                best_of: int = 3,
                blue_side_bonus: float = 0.22,
                decider_rule: str = "loser_picks",
            ) -> Any:
                return self.core_fn(
                    p_neutral_a,
                    team_a_has_game1_priority,
                    best_of,
                    blue_side_bonus,
                    decider_rule,
                )

            return _class_call
        else:
            # Called on instance sim.method(...)
            def _instance_call(
                p_neutral_a: Any,
                team_a_has_game1_priority: Any,
                best_of: int | None = None,
                blue_side_bonus: float | None = None,
                decider_rule: str | None = None,
            ) -> Any:
                bo = best_of if best_of is not None else instance.default_best_of
                bonus = blue_side_bonus if blue_side_bonus is not None else instance.default_blue_side_bonus
                rule = decider_rule if decider_rule is not None else instance.default_decider_rule
                return self.core_fn(
                    p_neutral_a,
                    team_a_has_game1_priority,
                    bo,
                    bonus,
                    rule,
                )

            return _instance_call


class MarkovSeriesSimulator:
    """Markov chain series simulator with dynamic rotating side selection.

    Parameters
    ----------
    default_best_of : int, default=3
        Default series length (must be odd positive integer, e.g. 1, 3, 5).
    default_blue_side_bonus : float, default=0.22
        Default log-odds boost for Blue side (+0.22 ~ 55.5% winrate on neutral).
    default_decider_rule : str, default="loser_picks"
        Side selection rule for decider game ("loser_picks" or "priority_picks").
    """

    def __init__(
        self,
        default_best_of: int = 3,
        default_blue_side_bonus: float = 0.22,
        default_decider_rule: str = "loser_picks",
    ) -> None:
        if default_best_of <= 0 or default_best_of % 2 == 0:
            raise ValueError(f"default_best_of must be odd positive integer, got {default_best_of}")
        if default_decider_rule not in ("loser_picks", "priority_picks"):
            raise ValueError(f"default_decider_rule must be 'loser_picks' or 'priority_picks', got '{default_decider_rule}'")
        self.default_best_of = default_best_of
        self.default_blue_side_bonus = float(default_blue_side_bonus)
        self.default_decider_rule = default_decider_rule

    predict_series_proba = _DualSimulatorMethod(_predict_series_proba_core)
    predict_score_distribution = _DualSimulatorMethod(_predict_score_distribution_core)
    compute_state_probabilities = _DualSimulatorMethod(_compute_state_probabilities_core)

    @staticmethod
    def single_game_proba(
        p_neutral_a: float | np.ndarray,
        a_is_blue: bool | np.ndarray,
        blue_side_bonus: float = 0.22,
    ) -> np.ndarray:
        """Compute single-game win probability for Team A given side assignment."""
        return _safe_game_prob(np.asarray(p_neutral_a), a_is_blue, blue_side_bonus)

    @staticmethod
    def series_expected_games(
        p_neutral_a: float,
        team_a_has_game1_priority: bool,
        best_of: int = 3,
        blue_side_bonus: float = 0.22,
        decider_rule: str = "loser_picks",
    ) -> float:
        """Calculate expected number of games played in the series."""
        scores = _predict_score_distribution_core(
            p_neutral_a,
            team_a_has_game1_priority,
            best_of,
            blue_side_bonus,
            decider_rule,
        )
        expected = 0.0
        for score, prob in scores.items():
            wa, wb = map(int, score.split("-"))
            expected += (wa + wb) * prob
        return float(expected)

    @staticmethod
    def simulate_series(
        p_neutral_a: float,
        team_a_has_game1_priority: bool,
        best_of: int = 3,
        blue_side_bonus: float = 0.22,
        decider_rule: str = "loser_picks",
        rng: np.random.Generator | int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Perform a single Monte Carlo simulation of a series.

        Returns
        -------
        final_score : str
            e.g. "2-1" or "0-2"
        game_history : list of dict
            List of game-by-game records containing game index, side assignments,
            and winner.
        """
        if rng is None or isinstance(rng, int):
            generator = np.random.default_rng(rng)
        else:
            generator = rng

        needed = (best_of + 1) // 2
        wa, wb = 0, 0
        history: list[dict[str, Any]] = []
        last_winner: str | None = None

        while wa < needed and wb < needed:
            k = wa + wb + 1
            if k == 1:
                a_is_blue = bool(team_a_has_game1_priority)
            elif decider_rule == "priority_picks" and k == best_of:
                a_is_blue = bool(team_a_has_game1_priority)
            else:
                a_is_blue = (last_winner == "B")

            p_a = float(_safe_game_prob(np.array(p_neutral_a), a_is_blue, blue_side_bonus))
            u = generator.random()
            winner = "A" if u < p_a else "B"

            if winner == "A":
                wa += 1
            else:
                wb += 1
            last_winner = winner

            history.append({
                "game": k,
                "team_a_side": "Blue" if a_is_blue else "Red",
                "team_b_side": "Red" if a_is_blue else "Blue",
                "p_game_a": p_a,
                "winner": winner,
                "score_after": f"{wa}-{wb}",
            })

        return f"{wa}-{wb}", history


def predict_series_proba(
    p_neutral_a: float | np.ndarray,
    team_a_has_game1_priority: bool | np.ndarray,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> np.ndarray:
    """Predict series win probability for Team A under dynamic rotating side selection.

    Parameters
    ----------
    p_neutral_a : float or np.ndarray
        Neutral match win probability for Team A (on neutral map).
    team_a_has_game1_priority : bool or np.ndarray
        True if Team A has Game 1 side selection priority (chooses Blue side).
    best_of : int, default=3
        Series length (odd positive integer: 1, 3, 5, etc.).
    blue_side_bonus : float, default=0.22
        Log-odds boost for Blue side (+0.22 corresponds to ~55.5% Blue winrate).
    decider_rule : str, default="loser_picks"
        Side selection rule for final decider game ("loser_picks" or "priority_picks").

    Returns
    -------
    np.ndarray
        Exact series win probability for Team A, satisfying strict binary symmetry.
    """
    return _predict_series_proba_core(
        p_neutral_a,
        team_a_has_game1_priority,
        best_of,
        blue_side_bonus,
        decider_rule,
    )


def predict_score_distribution(
    p_neutral_a: float,
    team_a_has_game1_priority: bool,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> dict[str, float]:
    """Compute exact analytic probability distribution across all possible series scores.

    Parameters
    ----------
    p_neutral_a : float
        Neutral match win probability for Team A.
    team_a_has_game1_priority : bool
        True if Team A has Game 1 side selection priority.
    best_of : int, default=3
        Series length (1, 3, 5).
    blue_side_bonus : float, default=0.22
        Log-odds boost for Blue side.
    decider_rule : str, default="loser_picks"
        Side selection rule for decider game.

    Returns
    -------
    dict[str, float]
        Dictionary mapping score strings (e.g. "2-0", "2-1", "1-2", "0-2")
        to probabilities summing to 1.0.
    """
    return _predict_score_distribution_core(
        p_neutral_a,
        team_a_has_game1_priority,
        best_of,
        blue_side_bonus,
        decider_rule,
    )


def compute_state_probabilities(
    p_neutral_a: float,
    team_a_has_game1_priority: bool,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> dict[str, float]:
    """Compute exact reach probabilities for all intermediate and terminal states in the Markov tree."""
    return _compute_state_probabilities_core(
        p_neutral_a,
        team_a_has_game1_priority,
        best_of,
        blue_side_bonus,
        decider_rule,
    )


def single_game_proba(
    p_neutral_a: float | np.ndarray,
    a_is_blue: bool | np.ndarray,
    blue_side_bonus: float = 0.22,
) -> np.ndarray:
    """Compute single-game win probability for Team A given side assignment."""
    return _safe_game_prob(np.asarray(p_neutral_a), a_is_blue, blue_side_bonus)


def series_expected_games(
    p_neutral_a: float,
    team_a_has_game1_priority: bool,
    best_of: int = 3,
    blue_side_bonus: float = 0.22,
    decider_rule: str = "loser_picks",
) -> float:
    """Calculate expected number of games played in the series."""
    return MarkovSeriesSimulator.series_expected_games(
        p_neutral_a,
        team_a_has_game1_priority,
        best_of,
        blue_side_bonus,
        decider_rule,
    )
