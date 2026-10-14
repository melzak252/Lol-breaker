from collections import defaultdict

import pytest
import torch

from src.models.a0_target_objectives import beta_objective


def _enumerate_terminal_probabilities(logit, best_of, eta):
    """Independent path enumeration of the stopped beta-Bernoulli series."""
    p_a = 1.0 / (1.0 + torch.exp(-torch.tensor(logit, dtype=torch.float64)))
    rho = 0.5 / (1.0 + torch.exp(-torch.tensor(eta, dtype=torch.float64)))
    concentration = (1.0 - rho) / rho
    alpha_a = float((p_a * concentration).item())
    alpha_b = float(((1.0 - p_a) * concentration).item())
    target = (best_of + 1) // 2
    probabilities = defaultdict(float)

    def visit(wins_a, wins_b, path_probability):
        if wins_a == target or wins_b == target:
            probabilities[(wins_a, wins_b)] += path_probability
            return
        games = wins_a + wins_b
        next_a = (alpha_a + wins_a) / (alpha_a + alpha_b + games)
        visit(wins_a + 1, wins_b, path_probability * next_a)
        visit(wins_a, wins_b + 1, path_probability * (1.0 - next_a))

    visit(0, 0, 1.0)
    return probabilities


def _objective(logits, best_of, wins_a, wins_b, eta, objective):
    return beta_objective(
        torch.tensor(logits, dtype=torch.float64),
        torch.tensor(best_of, dtype=torch.long),
        torch.tensor(wins_a, dtype=torch.long),
        torch.tensor(wins_b, dtype=torch.long),
        torch.tensor(eta, dtype=torch.float64),
        objective=objective,
    )


@pytest.mark.parametrize("wins_a,wins_b", [(1, 0), (0, 1)])
def test_bo1_terminal_and_winner_objectives_are_identical(wins_a, wins_b):
    terminal = _objective([0.37], [1], [wins_a], [wins_b], -0.42, "terminal")
    winner = _objective([0.37], [1], [wins_a], [wins_b], -0.42, "winner")

    torch.testing.assert_close(terminal, winner, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("winner", ["A", "B"])
def test_winner_objective_marginalizes_every_legal_stopped_score(best_of, winner):
    target = (best_of + 1) // 2
    scores = {
        3: [(2, 0), (2, 1), (0, 2), (1, 2)],
        5: [
            (3, 0), (3, 1), (3, 2),
            (0, 3), (1, 3), (2, 3),
        ],
    }[best_of]
    paths = _enumerate_terminal_probabilities(0.61, best_of, -0.17)
    assert set(paths) == set(scores)

    for score_a, score_b in scores:
        observed_winner = "A" if score_a == target else "B"
        if observed_winner != winner:
            continue
        expected_mass = sum(
            probability
            for (stopped_a, stopped_b), probability in paths.items()
            if (stopped_a == target) == (winner == "A")
        )
        actual = _objective(
            [0.61], [best_of], [score_a], [score_b], -0.17, "winner"
        )
        assert actual.shape == (1,)
        assert actual.item() == pytest.approx(-torch.log(torch.tensor(expected_mass, dtype=torch.float64)).item(), abs=1e-10)


def test_terminal_objective_matches_independent_stopped_path_probability():
    best_of, score_a, score_b = 5, 3, 2
    paths = _enumerate_terminal_probabilities(-0.28, best_of, 0.31)
    actual = _objective([-0.28], [best_of], [score_a], [score_b], 0.31, "terminal")

    assert actual.item() == pytest.approx(-torch.log(torch.tensor(paths[(score_a, score_b)], dtype=torch.float64)).item(), abs=1e-10)


@pytest.mark.parametrize("objective", ["terminal", "winner"])
@pytest.mark.parametrize("best_of,score_a,score_b", [(1, 1, 0), (3, 2, 1), (5, 1, 3)])
def test_beta_objective_is_symmetric_when_teams_and_score_are_swapped(
    objective, best_of, score_a, score_b
):
    original = _objective([0.44], [best_of], [score_a], [score_b], -0.23, objective)
    swapped = _objective([-0.44], [best_of], [score_b], [score_a], -0.23, objective)

    torch.testing.assert_close(original, swapped, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("objective", ["terminal", "winner"])
def test_beta_objective_has_finite_gradients_for_logits_and_eta(objective):
    logits = torch.tensor([-0.8, 0.15, 1.1], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor(-0.35, dtype=torch.float64, requires_grad=True)
    loss = beta_objective(
        logits,
        torch.tensor([1, 3, 5]),
        torch.tensor([1, 2, 1]),
        torch.tensor([0, 1, 3]),
        eta,
        objective=objective,
    ).sum()
    grad_logits, grad_eta = torch.autograd.grad(loss, (logits, eta))

    assert torch.isfinite(loss)
    assert torch.isfinite(grad_logits).all()
    assert torch.isfinite(grad_eta)
    assert grad_logits.abs().sum() > 0
    assert grad_eta.abs() > 0


@pytest.mark.parametrize("bad_best_of", [0, 2, 4, 7])
def test_beta_objective_rejects_unsupported_best_of(bad_best_of):
    with pytest.raises(ValueError):
        _objective([0.2], [bad_best_of], [1], [0], -0.1, "terminal")


@pytest.mark.parametrize("best_of,score_a,score_b", [(3, 0, 0), (3, 1, 1), (3, 2, 2), (5, 4, 0), (5, 0, 0)])
def test_terminal_objective_rejects_nonterminal_or_impossible_scores(best_of, score_a, score_b):
    with pytest.raises(ValueError):
        _objective([0.2], [best_of], [score_a], [score_b], -0.1, "terminal")


def test_winner_objective_rejects_scores_that_do_not_identify_a_winner():
    with pytest.raises(ValueError):
        _objective([0.2], [3], [1], [1], -0.1, "winner")
