"""
Unit tests for Nash Equilibrium Solver.
"""

import pytest
import numpy as np

from src.games.normal_form import NormalFormGame
from src.solvers.nash_solver import NashSolver
from src.utils.payoff_matrices import (
    chicken_game,
    prisoners_dilemma,
    coordination_game,
    asymmetric_chicken_game,
)


@pytest.fixture
def solver():
    return NashSolver()


def test_chicken_game_equilibria(solver):
    """Test Chicken Game has 2 pure NE and 1 mixed NE."""
    game = chicken_game()
    equilibria = solver.find_all_equilibria(game)

    pure = equilibria["pure"]
    mixed = equilibria["mixed"]

    # Chicken game should have exactly 2 pure equilibria: (Go, Yield) and (Yield, Go)
    assert len(pure) == 2
    assert (0, 1) in pure  # Player A Go, Player B Yield
    assert (1, 0) in pure  # Player A Yield, Player B Go

    # Should have 1 mixed equilibrium
    assert len(mixed) == 1
    p, q = mixed[0]
    # In symmetric chicken with payoffs [-100, 10; 2, 0],
    # q*(-100) + (1-q)*10 = q*(2) + (1-q)*0 => -100q + 10 - 10q = 2q => 112q = 10 => q ~ 0.089
    assert 0.05 < p[0] < 0.15
    assert 0.05 < q[0] < 0.15


def test_prisoners_dilemma(solver):
    """Test Prisoner's Dilemma has unique strictly dominant NE."""
    game = prisoners_dilemma()
    equilibria = solver.find_all_equilibria(game)

    pure = equilibria["pure"]
    # Defect (Go) is dominant: (0, 0)
    assert len(pure) == 1
    assert pure[0] == (0, 0)


def test_coordination_game(solver):
    """Test Coordination Game has 2 pure NE."""
    game = coordination_game()
    equilibria = solver.find_all_equilibria(game)

    pure = equilibria["pure"]
    assert len(pure) == 2
    assert (0, 0) in pure
    assert (1, 1) in pure


def test_asymmetric_chicken(solver):
    """Test asymmetric chicken game calculation."""
    game = asymmetric_chicken_game()
    equilibria = solver.find_all_equilibria(game)

    pure = equilibria["pure"]
    assert len(pure) >= 1
