"""
Unit tests for Stackelberg Equilibrium Solver.
"""

import pytest
import numpy as np

from src.games.normal_form import NormalFormGame
from src.games.extensive_form import ExtensiveFormGame
from src.solvers.stackelberg_solver import StackelbergSolver
from src.utils.payoff_matrices import chicken_game


@pytest.fixture
def solver():
    return StackelbergSolver()


def test_stackelberg_chicken_game(solver):
    """Test that in the Chicken game, Leader commits to Go (0), Follower responds Yield (1)."""
    game = chicken_game()
    eq = solver.solve_from_normal_form(game)

    # Leader chooses Go (action 0)
    assert eq.leader_action == 0
    # Follower best-response to Go (0) is Yield (1)
    assert eq.follower_response[0] == 1
    # Payoffs: Leader gets 10, Follower gets 2
    assert eq.leader_payoff == 10.0
    assert eq.follower_payoff == 2.0


def test_stackelberg_extensive_tree(solver):
    """Test backward induction on extensive form game tree."""
    tree = ExtensiveFormGame.create_intersection_tree()
    eq = solver.solve_extensive_game(tree)

    assert eq is not None
    assert eq.leader_payoff > eq.follower_payoff
