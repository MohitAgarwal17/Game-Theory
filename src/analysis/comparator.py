"""
Comparator module for cross-mechanism analysis.

Runs all intersection management strategies (Nash, Stackelberg, Auction,
Correlated Equilibrium) on the same scenarios and produces comparative
results tables and visualizations.
"""

import numpy as np
from dataclasses import dataclass

from src.games.normal_form import NormalFormGame
from src.games.repeated_game import RepeatedGame, StrategyType
from src.solvers.nash_solver import NashSolver
from src.solvers.stackelberg_solver import StackelbergSolver
from src.solvers.correlated_solver import CorrelatedSolver
from src.mechanisms.auction import (
    VickreyAuction, FirstPriceAuction, AuctionBid
)


class MechanismComparator:
    """
    Compares different intersection management mechanisms.

    Runs Nash, Stackelberg, Correlated Equilibrium, and Auction-based
    approaches on the same game and produces side-by-side results.

    Examples
    --------
    >>> from src.utils.payoff_matrices import chicken_game
    >>> game = NormalFormGame(*chicken_game(),
    ...     strategy_labels_a=["Go", "Yield"],
    ...     strategy_labels_b=["Go", "Yield"])
    >>> comp = MechanismComparator()
    >>> results = comp.compare_all(game)
    """

    def __init__(self):
        self.nash_solver = NashSolver()
        self.stackelberg_solver = StackelbergSolver()
        self.correlated_solver = CorrelatedSolver()
        self.vickrey_auction = VickreyAuction()

    def compare_all(self, game: NormalFormGame) -> dict:
        """
        Run all mechanisms and compile comparison results.

        Parameters
        ----------
        game : NormalFormGame
            The intersection game to analyze.

        Returns
        -------
        dict
            Comparison results for each mechanism.
        """
        results = {}

        # 1. Nash Equilibrium
        nash_eq = self.nash_solver.solve(game)
        if nash_eq:
            # Use the best NE (highest social welfare)
            best_ne = max(nash_eq, key=lambda e: e.social_welfare)
            results["Nash Equilibrium"] = {
                "social_welfare": best_ne.social_welfare,
                "payoff_a": best_ne.payoff_a,
                "payoff_b": best_ne.payoff_b,
                "is_pure": best_ne.is_pure,
                "num_equilibria": len(nash_eq),
            }

            # Also track worst NE for Price of Anarchy
            worst_ne = min(nash_eq, key=lambda e: e.social_welfare)
            results["Nash (Worst)"] = {
                "social_welfare": worst_ne.social_welfare,
                "payoff_a": worst_ne.payoff_a,
                "payoff_b": worst_ne.payoff_b,
            }

        # 2. Stackelberg Equilibrium
        stack_eq = self.stackelberg_solver.solve_from_normal_form(game)
        results["Stackelberg"] = {
            "social_welfare": stack_eq.leader_payoff + stack_eq.follower_payoff,
            "payoff_a": stack_eq.leader_payoff,
            "payoff_b": stack_eq.follower_payoff,
            "outcome": stack_eq.outcome,
        }

        # 3. Correlated Equilibrium (max welfare)
        ce = self.correlated_solver.solve(game)
        results["Correlated Equilibrium"] = {
            "social_welfare": ce.social_welfare,
            "payoff_a": ce.expected_payoff_a,
            "payoff_b": ce.expected_payoff_b,
            "is_valid": ce.is_valid,
        }

        # 4. Traffic Light (specific CE)
        tl = self.correlated_solver.traffic_light_equilibrium(game)
        results["Traffic Light"] = {
            "social_welfare": tl.social_welfare,
            "payoff_a": tl.expected_payoff_a,
            "payoff_b": tl.expected_payoff_b,
            "is_valid_ce": tl.is_valid,
        }

        # 5. Social Optimum (best possible outcome)
        pareto = game.pareto_optimal_outcomes()
        if pareto:
            best_pareto = max(pareto, key=lambda p: game.social_welfare(*p))
            so_a, so_b = game.payoff(*best_pareto)
            results["Social Optimum"] = {
                "social_welfare": so_a + so_b,
                "payoff_a": so_a,
                "payoff_b": so_b,
            }

        # 6. Price of Anarchy
        poa_data = self.nash_solver.compute_price_of_anarchy(game)
        results["_price_of_anarchy"] = poa_data

        return results

    def compare_repeated_strategies(
        self,
        game: NormalFormGame,
        num_rounds: int = 100,
    ) -> dict:
        """
        Compare all repeated game strategies head-to-head.

        Parameters
        ----------
        game : NormalFormGame
            The stage game.
        num_rounds : int
            Number of rounds per match.

        Returns
        -------
        dict
            Tournament results with rankings.
        """
        return RepeatedGame.run_tournament(
            game, num_rounds=num_rounds
        )

    def generate_summary_table(self, comparison: dict) -> str:
        """
        Generate a formatted summary table from comparison results.

        Parameters
        ----------
        comparison : dict
            Output from compare_all().

        Returns
        -------
        str
            Formatted table string.
        """
        lines = [
            f"{'Mechanism':<25} {'Welfare':>10} {'Payoff A':>10} {'Payoff B':>10}",
            "-" * 60,
        ]

        for name, data in comparison.items():
            if name.startswith("_"):
                continue
            sw = data.get("social_welfare", 0)
            pa = data.get("payoff_a", 0)
            pb = data.get("payoff_b", 0)
            lines.append(f"{name:<25} {sw:>10.2f} {pa:>10.2f} {pb:>10.2f}")

        return "\n".join(lines)
