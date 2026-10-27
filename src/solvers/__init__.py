"""Solvers module — equilibrium computation algorithms."""

from src.solvers.nash_solver import NashSolver
from src.solvers.stackelberg_solver import StackelbergSolver, StackelbergEquilibrium
from src.solvers.bayesian_solver import BayesianSolver, BayesianNEResult
from src.solvers.correlated_solver import CorrelatedSolver, CorrelatedEquilibriumResult

# Alias for backwards compatibility
CorrelatedEquilibriumSolver = CorrelatedSolver

__all__ = [
    "NashSolver",
    "StackelbergSolver",
    "StackelbergEquilibrium",
    "BayesianSolver",
    "BayesianNEResult",
    "CorrelatedSolver",
    "CorrelatedEquilibriumSolver",
    "CorrelatedEquilibriumResult",
]
