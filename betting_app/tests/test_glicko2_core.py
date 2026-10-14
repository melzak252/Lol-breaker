from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from src.ratings.glicko2_core import (
    Glicko2Observation,
    Glicko2State,
    expected_score,
    inflate,
    update,
)


def test_official_glicko2_worked_example() -> None:
    result = update(
        Glicko2State(rating=1500, rd=200, volatility=0.06),
        [
            Glicko2Observation(1400, 30, 1),
            Glicko2Observation(1550, 100, 0),
            Glicko2Observation(1700, 300, 0),
        ],
    )

    assert result.rating == pytest.approx(1464.06, abs=0.01)
    assert result.rd == pytest.approx(151.52, abs=0.01)
    assert result.volatility == pytest.approx(0.059996, abs=0.000001)


def test_volatility_objective_uses_phi_squared_at_high_precision() -> None:
    result = update(
        Glicko2State(rating=1720, rd=65, volatility=0.12),
        [
            Glicko2Observation(2100, 45, 1, weight=2.75),
            Glicko2Observation(1300, 80, 0, weight=0.4),
        ],
        tau=1.1,
        convergence_tolerance=1e-14,
    )

    # Independently evaluated at 80 decimal digits from the published
    # Glicko-2 equations.  Replacing phi**2 with phi changes volatility by
    # about 2.5e-4 in this deliberately sensitive case.
    assert result.rating == pytest.approx(1773.8722403894805, abs=1e-10)
    assert result.rd == pytest.approx(66.93988496297521, abs=1e-10)
    assert result.volatility == pytest.approx(0.12203967782610317, abs=1e-12)


def test_empty_and_all_zero_weight_batches_are_exact_no_ops() -> None:
    state = Glicko2State(1612, 87, 0.055)

    assert update(state, []) is state
    assert update(
        state,
        [
            Glicko2Observation(1400, 30, 1, weight=0),
            Glicko2Observation(1750, 120, 0, weight=0),
        ],
    ) is state


def test_zero_weight_observations_are_removed_from_mixed_batch() -> None:
    state = Glicko2State(1575, 110, 0.07)
    evidence = Glicko2Observation(1650, 90, 1, weight=0.75)

    without_zero = update(state, [evidence])
    with_zero = update(
        state,
        [
            Glicko2Observation(-9000, 0, 0, weight=0),
            evidence,
            Glicko2Observation(9000, 350, 1, weight=0),
        ],
    )

    assert with_zero == without_zero


def test_integer_weight_matches_replicating_the_observation() -> None:
    state = Glicko2State(1510, 140, 0.08)
    observation = Glicko2Observation(1580, 75, 1)

    weighted = update(
        state,
        [Glicko2Observation(1580, 75, 1, weight=4)],
        convergence_tolerance=1e-12,
    )
    replicated = update(
        state,
        [observation, observation, observation, observation],
        convergence_tolerance=1e-12,
    )

    assert weighted.rating == pytest.approx(replicated.rating, abs=1e-12)
    assert weighted.rd == pytest.approx(replicated.rd, abs=1e-12)
    assert weighted.volatility == pytest.approx(replicated.volatility, abs=1e-15)


def test_equal_uncertainty_head_to_head_update_is_side_symmetric() -> None:
    stronger = Glicko2State(1600, 100, 0.06)
    weaker = Glicko2State(1400, 100, 0.06)

    stronger_after_loss = update(
        stronger,
        [Glicko2Observation(weaker.rating, weaker.rd, 0)],
    )
    weaker_after_win = update(
        weaker,
        [Glicko2Observation(stronger.rating, stronger.rd, 1)],
    )

    assert stronger_after_loss.rating - stronger.rating == pytest.approx(
        -(weaker_after_win.rating - weaker.rating),
        abs=1e-12,
    )
    assert stronger_after_loss.rd == pytest.approx(weaker_after_win.rd, abs=1e-12)
    assert stronger_after_loss.volatility == pytest.approx(
        weaker_after_win.volatility,
        abs=1e-15,
    )


def test_batch_update_is_permutation_invariant() -> None:
    state = Glicko2State(1490, 185, 0.065)
    observations = [
        Glicko2Observation(1420, 45, 1, weight=0.25),
        Glicko2Observation(1710, 210, 0, weight=2),
        Glicko2Observation(1515, 65, 0.5, weight=1.5),
        Glicko2Observation(1380, 300, 0, weight=0.8),
    ]

    assert update(state, observations) == update(state, list(reversed(observations)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rating": math.nan},
        {"rating": math.inf},
        {"rd": -0.01},
        {"rd": math.inf},
        {"volatility": 0},
        {"volatility": -0.01},
        {"volatility": math.nan},
    ],
)
def test_state_rejects_invalid_domains(kwargs: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Glicko2State(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opponent_rating": math.nan, "opponent_rd": 10, "score": 1},
        {"opponent_rating": 1500, "opponent_rd": -1, "score": 1},
        {"opponent_rating": 1500, "opponent_rd": 10, "score": -0.01},
        {"opponent_rating": 1500, "opponent_rd": 10, "score": 1.01},
        {"opponent_rating": 1500, "opponent_rd": 10, "score": 1, "weight": -1},
        {"opponent_rating": 1500, "opponent_rd": 10, "score": 1, "weight": math.inf},
    ],
)
def test_observation_rejects_invalid_domains(kwargs: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        Glicko2Observation(**kwargs)


def test_states_and_observations_are_immutable_finite_values() -> None:
    state = Glicko2State(1500, 50, 0.06)
    observation = Glicko2Observation(1600, 70, 0.5, 0.25)

    assert all(math.isfinite(value) for value in (state.rating, state.rd, state.volatility))
    assert all(
        math.isfinite(value)
        for value in (
            observation.opponent_rating,
            observation.opponent_rd,
            observation.score,
            observation.weight,
        )
    )
    with pytest.raises(FrozenInstanceError):
        state.rating = 1600  # type: ignore[misc]


def test_update_requires_list_like_typed_evidence_and_valid_parameters() -> None:
    state = Glicko2State()
    observation = Glicko2Observation(1500, 100, 1)

    assert update(state, (observation,)) == update(state, [observation])
    with pytest.raises(TypeError):
        update(state, (item for item in [observation]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        update(state, [object()])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        update(state, [observation], tau=0)
    with pytest.raises(ValueError):
        update(state, [observation], convergence_tolerance=math.nan)


def test_prediction_is_complementary_with_unequal_deviations() -> None:
    first = expected_score(1675, 45, 1520, 230)
    second = expected_score(1520, 230, 1675, 45)

    assert 0 < first < 1
    assert 0 < second < 1
    assert first + second == pytest.approx(1.0, abs=1e-15)
    assert expected_score(1500, 20, 1500, 300) == 0.5


def test_prediction_validates_all_inputs() -> None:
    with pytest.raises(ValueError):
        expected_score(1500, -1, 1500, 50)
    with pytest.raises(ValueError):
        expected_score(1500, 50, math.inf, 50)
    with pytest.raises(TypeError):
        expected_score(1500, 50, "1500", 50)  # type: ignore[arg-type]


def test_inactivity_inflates_deviation_by_elapsed_periods_and_caps_it() -> None:
    state = Glicko2State(1540, 80, 0.06)
    periods = 9

    result = inflate(state, periods)
    expected_rd = 173.7178 * math.sqrt(
        (state.rd / 173.7178) ** 2 + periods * state.volatility**2
    )

    assert result.rating == state.rating
    assert result.rd == pytest.approx(expected_rd, abs=1e-12)
    assert result.volatility == state.volatility
    assert inflate(state, 0) is state
    assert inflate(state, 100_000, max_rd=125).rd == 125


def test_inactivity_rejects_invalid_periods_and_cap() -> None:
    state = Glicko2State()

    with pytest.raises(TypeError):
        inflate(state, 1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        inflate(state, -1)
    with pytest.raises(ValueError):
        inflate(state, 1, max_rd=math.inf)
