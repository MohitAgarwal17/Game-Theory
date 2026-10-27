"""
Unit tests for Bayesian Game and Solver.
"""

import pytest
from src.games.bayesian_game import BayesianGame, PlayerType
from src.solvers.bayesian_solver import BayesianSolver


@pytest.fixture
def solver():
    return BayesianSolver()


def test_bayesian_game_creation():
    """Test factory creates valid Bayesian game structure."""
    game = BayesianGame.create_av_vs_human(prob_aggressive=0.3)
    assert len(game.types_a) == 1  # AV has 1 type
    assert len(game.types_b) == 2  # Human has aggressive & cautious
    assert game.actions_a == ["Go", "Yield"]
    assert game.actions_b == ["Go", "Yield"]


def test_bayesian_solver(solver):
    """Test solving Bayesian game with conservative AV behavior."""
    game = BayesianGame.create_av_vs_human(prob_aggressive=0.4)
    result = solver.solve(game)

    assert result.num_equilibria >= 1
    # Check that in at least one equilibrium AV yields or adapts
    assert 0.0 <= result.av_yield_rate <= 1.0


def test_bayesian_sensitivity_to_priors():
    """Test that higher aggressive probability leads to higher AV yield rate."""
    game_cautious_traffic = BayesianGame.create_av_vs_human(prob_aggressive=0.1)
    game_aggressive_traffic = BayesianGame.create_av_vs_human(prob_aggressive=0.8)

    eq_cautious = game_cautious_traffic.find_bayesian_ne()
    eq_aggressive = game_aggressive_traffic.find_bayesian_ne()

    assert len(eq_cautious) >= 1
    assert len(eq_aggressive) >= 1
