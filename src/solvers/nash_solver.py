"""
Nash Equilibrium Solver.

Computes pure-strategy, mixed-strategy, and support-enumeration
Nash Equilibria for 2-player normal-form games. Integrates with
the Nashpy library for robust computation and provides custom
implementations for educational demonstration.
"""

import numpy as np
from dataclasses import dataclass

try:
    import nashpy
    HAS_NASHPY = True
except ImportError:
    HAS_NASHPY = False

from src.games.normal_form import NormalFormGame


@dataclass
class NashEquilibrium:
    """
    Represents a Nash Equilibrium.

    Attributes
    ----------
    strategy_a : np.ndarray
        Player A's strategy (probability distribution over actions).
    strategy_b : np.ndarray
        Player B's strategy (probability distribution over actions).
    payoff_a : float
        Player A's expected payoff at this NE.
    payoff_b : float
        Player B's expected payoff at this NE.
    is_pure : bool
        True if this is a pure-strategy NE.
    is_pareto_optimal : bool
        True if this NE is Pareto optimal.
    social_welfare : float
        Sum of both players' payoffs.
    """
    strategy_a: np.ndarray
    strategy_b: np.ndarray
    payoff_a: float
    payoff_b: float
    is_pure: bool
    is_pareto_optimal: bool = False
    social_welfare: float = 0.0

    def __post_init__(self):
        self.social_welfare = self.payoff_a + self.payoff_b

    def strategy_label(self, game: NormalFormGame) -> str:
        """Human-readable strategy labels."""
        if self.is_pure:
            idx_a = int(np.argmax(self.strategy_a))
            idx_b = int(np.argmax(self.strategy_b))
            return (
                f"({game.strategy_labels_a[idx_a]}, "
                f"{game.strategy_labels_b[idx_b]})"
            )
        else:
            parts_a = [
                f"{game.strategy_labels_a[i]}:{self.strategy_a[i]:.3f}"
                for i in range(len(self.strategy_a))
                if self.strategy_a[i] > 1e-6
            ]
            parts_b = [
                f"{game.strategy_labels_b[j]}:{self.strategy_b[j]:.3f}"
                for j in range(len(self.strategy_b))
                if self.strategy_b[j] > 1e-6
            ]
            return f"({', '.join(parts_a)}) × ({', '.join(parts_b)})"


class NashSolver:
    """
    Solver for Nash Equilibria in 2-player normal-form games.

    Provides multiple solution methods:
    1. Pure-strategy NE via best-response enumeration
    2. Mixed-strategy NE via indifference conditions (2×2 games)
    3. Support enumeration via Nashpy (general NxM games)

    Examples
    --------
    >>> from src.utils.payoff_matrices import chicken_game
    >>> game = NormalFormGame(*chicken_game(),
    ...     strategy_labels_a=["Go", "Yield"],
    ...     strategy_labels_b=["Go", "Yield"])
    >>> solver = NashSolver()
    >>> equilibria = solver.solve(game)
    >>> for eq in equilibria:
    ...     print(eq.strategy_label(game), eq.payoff_a, eq.payoff_b)
    """

    def _ensure_game(self, game: NormalFormGame | tuple) -> NormalFormGame:
        if isinstance(game, tuple) and len(game) == 2:
            labels_a = ["Go", "Yield"] if game[0].shape[0] == 2 else [f"S{i}" for i in range(game[0].shape[0])]
            labels_b = ["Go", "Yield"] if game[0].shape[1] == 2 else [f"S{j}" for j in range(game[0].shape[1])]
            return NormalFormGame(game[0], game[1], strategy_labels_a=labels_a, strategy_labels_b=labels_b)
        return game

    def solve(self, game: NormalFormGame | tuple) -> list[NashEquilibrium]:
        """
        Find all Nash Equilibria (pure and mixed).

        Parameters
        ----------
        game : NormalFormGame or tuple[np.ndarray, np.ndarray]
            The game to solve.

        Returns
        -------
        list[NashEquilibrium]
            All found Nash Equilibria.
        """
        game = self._ensure_game(game)
        equilibria = []

        # Step 1: Find pure-strategy NE
        pure_ne = self.find_pure_ne(game)
        equilibria.extend(pure_ne)

        # Step 2: Find mixed-strategy NE
        mixed_ne = self.find_mixed_ne(game)
        equilibria.extend(mixed_ne)

        # Step 3: Tag Pareto optimality
        pareto_outcomes = game.pareto_optimal_outcomes()
        for eq in equilibria:
            if eq.is_pure:
                idx_a = int(np.argmax(eq.strategy_a))
                idx_b = int(np.argmax(eq.strategy_b))
                eq.is_pareto_optimal = (idx_a, idx_b) in pareto_outcomes

        return equilibria

    def find_all_equilibria(self, game: NormalFormGame | tuple) -> dict:
        """
        Convenience method returning pure and mixed equilibria in dictionary format.
        """
        game = self._ensure_game(game)
        all_eq = self.solve(game)
        pure_profiles = [
            (int(np.argmax(eq.strategy_a)), int(np.argmax(eq.strategy_b)))
            for eq in all_eq
            if eq.is_pure
        ]
        mixed_profiles = [
            (eq.strategy_a, eq.strategy_b)
            for eq in all_eq
            if not eq.is_pure
        ]
        return {
            "pure": pure_profiles,
            "mixed": mixed_profiles,
            "all": all_eq,
        }


    def find_pure_ne(self, game: NormalFormGame) -> list[NashEquilibrium]:
        """
        Find all pure-strategy Nash Equilibria.

        A pure strategy profile (i, j) is a NE if:
        - i is a best response to j for Player A
        - j is a best response to i for Player B

        Parameters
        ----------
        game : NormalFormGame
            The game to analyze.

        Returns
        -------
        list[NashEquilibrium]
            List of pure-strategy NE.
        """
        equilibria = []

        for i in range(game.num_strategies_a):
            for j in range(game.num_strategies_b):
                # Check if i is best response to j
                br_a = game.best_response_a(j)
                # Check if j is best response to i
                br_b = game.best_response_b(i)

                if i in br_a and j in br_b:
                    strategy_a = np.zeros(game.num_strategies_a)
                    strategy_a[i] = 1.0
                    strategy_b = np.zeros(game.num_strategies_b)
                    strategy_b[j] = 1.0

                    payoff_a, payoff_b = game.payoff(i, j)

                    equilibria.append(NashEquilibrium(
                        strategy_a=strategy_a,
                        strategy_b=strategy_b,
                        payoff_a=payoff_a,
                        payoff_b=payoff_b,
                        is_pure=True,
                    ))

        return equilibria

    def find_mixed_ne(self, game: NormalFormGame) -> list[NashEquilibrium]:
        """
        Find mixed-strategy Nash Equilibria.

        For 2×2 games, uses the indifference condition.
        For general NxM games, uses Nashpy's support enumeration.

        Parameters
        ----------
        game : NormalFormGame
            The game to analyze.

        Returns
        -------
        list[NashEquilibrium]
            List of mixed-strategy NE (excluding pure NE).
        """
        equilibria = []

        if game.shape == (2, 2):
            # Analytical solution for 2×2 games
            mixed = self._solve_2x2_mixed(game)
            if mixed is not None:
                equilibria.append(mixed)

        # Also try Nashpy if available
        if HAS_NASHPY:
            nashpy_results = self._solve_with_nashpy(game)
            # Filter out pure strategies and duplicates
            for ne in nashpy_results:
                if not ne.is_pure and not self._is_duplicate(ne, equilibria):
                    equilibria.append(ne)

        return equilibria

    def _solve_2x2_mixed(self, game: NormalFormGame) -> NashEquilibrium | None:
        """
        Solve a 2×2 game for the mixed-strategy NE using indifference conditions.

        Player B mixes to make Player A indifferent:
        q·u_A(0,0) + (1-q)·u_A(0,1) = q·u_A(1,0) + (1-q)·u_A(1,1)

        Player A mixes to make Player B indifferent:
        p·u_B(0,0) + (1-p)·u_B(1,0) = p·u_B(0,1) + (1-p)·u_B(1,1)
        """
        a = game.payoff_a
        b = game.payoff_b

        # Solve for q (B's mixing probability for action 0)
        # A indifferent: q*a[0,0] + (1-q)*a[0,1] = q*a[1,0] + (1-q)*a[1,1]
        denom_q = (a[0, 0] - a[0, 1] - a[1, 0] + a[1, 1])
        if abs(denom_q) < 1e-10:
            return None  # No interior mixed NE

        q = (a[1, 1] - a[0, 1]) / denom_q

        # Solve for p (A's mixing probability for action 0)
        # B indifferent: p*b[0,0] + (1-p)*b[1,0] = p*b[0,1] + (1-p)*b[1,1]
        denom_p = (b[0, 0] - b[0, 1] - b[1, 0] + b[1, 1])
        if abs(denom_p) < 1e-10:
            return None

        p = (b[1, 1] - b[1, 0]) / denom_p

        # Check if probabilities are valid (strictly between 0 and 1)
        if not (0 < p < 1 and 0 < q < 1):
            return None

        strategy_a = np.array([p, 1 - p])
        strategy_b = np.array([q, 1 - q])

        payoff_a, payoff_b = game.expected_payoff_mixed(strategy_a, strategy_b)

        return NashEquilibrium(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            payoff_a=payoff_a,
            payoff_b=payoff_b,
            is_pure=False,
        )

    def _solve_with_nashpy(self, game: NormalFormGame) -> list[NashEquilibrium]:
        """Use the Nashpy library for robust NE computation."""
        if not HAS_NASHPY:
            return []

        nashpy_game = nashpy.Game(game.payoff_a, game.payoff_b)
        equilibria = []

        try:
            for eq in nashpy_game.support_enumeration():
                strategy_a, strategy_b = eq

                # Skip invalid results
                if np.any(np.isnan(strategy_a)) or np.any(np.isnan(strategy_b)):
                    continue
                if np.any(strategy_a < -1e-6) or np.any(strategy_b < -1e-6):
                    continue

                # Clip small negatives to zero
                strategy_a = np.maximum(strategy_a, 0)
                strategy_b = np.maximum(strategy_b, 0)

                # Normalize
                strategy_a /= strategy_a.sum()
                strategy_b /= strategy_b.sum()

                is_pure = (
                    np.count_nonzero(strategy_a > 1e-6) == 1 and
                    np.count_nonzero(strategy_b > 1e-6) == 1
                )

                payoff_a, payoff_b = game.expected_payoff_mixed(
                    strategy_a, strategy_b
                )

                equilibria.append(NashEquilibrium(
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    payoff_a=payoff_a,
                    payoff_b=payoff_b,
                    is_pure=is_pure,
                ))
        except Exception:
            pass  # Nashpy may fail on degenerate games

        return equilibria

    def _is_duplicate(
        self, ne: NashEquilibrium, existing: list[NashEquilibrium]
    ) -> bool:
        """Check if an NE is already in the list."""
        for other in existing:
            if (np.allclose(ne.strategy_a, other.strategy_a, atol=1e-6) and
                    np.allclose(ne.strategy_b, other.strategy_b, atol=1e-6)):
                return True
        return False

    def compute_price_of_anarchy(
        self, game: NormalFormGame
    ) -> dict:
        """
        Compute the Price of Anarchy and Price of Stability.

        Price of Anarchy = Social Optimum / Worst NE Social Welfare
        Price of Stability = Social Optimum / Best NE Social Welfare

        Parameters
        ----------
        game : NormalFormGame
            The game to analyze.

        Returns
        -------
        dict
            Contains 'price_of_anarchy', 'price_of_stability',
            'social_optimum', 'worst_ne_welfare', 'best_ne_welfare'.
        """
        # Find social optimum
        best_sw = float("-inf")
        for i in range(game.num_strategies_a):
            for j in range(game.num_strategies_b):
                sw = game.social_welfare(i, j)
                if sw > best_sw:
                    best_sw = sw

        # Find NE welfare values
        equilibria = self.solve(game)
        if not equilibria:
            return {
                "price_of_anarchy": float("inf"),
                "price_of_stability": float("inf"),
                "social_optimum": best_sw,
            }

        ne_welfares = [eq.social_welfare for eq in equilibria]
        worst_ne = min(ne_welfares)
        best_ne = max(ne_welfares)

        # Avoid division by zero or negative
        poa = best_sw / worst_ne if worst_ne > 0 else float("inf")
        pos = best_sw / best_ne if best_ne > 0 else float("inf")

        return {
            "price_of_anarchy": poa,
            "price_of_stability": pos,
            "social_optimum": best_sw,
            "worst_ne_welfare": worst_ne,
            "best_ne_welfare": best_ne,
            "num_equilibria": len(equilibria),
        }
