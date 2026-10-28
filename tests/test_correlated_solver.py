"""
Unit tests for Correlated Equilibrium Solver.
"""

import pytest
import numpy as np

from src.games.normal_form import NormalFormGame
from src.solvers.correlated_solver import CorrelatedSolver
from src.utils.payoff_matrices import chicken_game


@pytest.fixture
def solver():
    return CorrelatedSolver()


def test_correlated_equilibrium_chicken(solver):
    """Test that CE on Chicken Game achieves optimal social welfare without collisions."""
    payoff_a, payoff_b = chicken_game()
    game = NormalFormGame(
        payoff_a,
        payoff_b,
        strategy_labels_a=["Go", "Yield"],
        strategy_labels_b=["Go", "Yield"],
    )

    ce_result = solver.solve(game)

    assert ce_result.is_valid
    # Social welfare under optimal CE is 12.0
    assert np.isclose(ce_result.social_welfare, 12.0, atol=1e-3)
    # The probability of mutual Go (collision) must be zero
    assert np.isclose(ce_result.distribution[0, 0], 0.0, atol=1e-5)


def test_correlated_traffic_light_signal(solver):
    """Test that the traffic light equilibrium evenly coordinates turns."""
    payoff_a, payoff_b = chicken_game()
    game = NormalFormGame(
        payoff_a,
        payoff_b,
        strategy_labels_a=["Go", "Yield"],
        strategy_labels_b=["Go", "Yield"],
    )

    tl_result = solver.traffic_light_equilibrium(game)

    assert tl_result.is_valid
    # In symmetric chicken, traffic light splits probability between (0, 1) and (1, 0)
    p_go_yield = tl_result.distribution[0, 1]
    p_yield_go = tl_result.distribution[1, 0]
    assert np.isclose(p_go_yield, 0.5, atol=0.05)
    assert np.isclose(p_yield_go, 0.5, atol=0.05)
    assert np.isclose(tl_result.social_welfare, 12.0, atol=1e-3)
