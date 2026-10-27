"""
Normal-Form (Strategic-Form) Game representation.

A normal-form game is defined by:
- A set of players
- A set of strategies (actions) for each player
- A payoff function mapping strategy profiles to utilities

This module provides the NormalFormGame class used to represent
simultaneous-move intersection games (e.g., the Chicken Game).
"""

import numpy as np
from itertools import product


class NormalFormGame:
    """
    Represents a finite normal-form (matrix) game.

    Supports 2-player games with arbitrary numbers of strategies.
    Stores payoff matrices for each player and provides utility
    methods for best-response analysis, dominance checking, and
    Pareto optimality evaluation.

    Parameters
    ----------
    payoff_a : np.ndarray
        Payoff matrix for Player A (row player). Shape: (m, n)
        where m = number of A's strategies, n = number of B's strategies.
    payoff_b : np.ndarray
        Payoff matrix for Player B (column player). Shape: (m, n).
    strategy_labels_a : list[str], optional
        Human-readable labels for Player A's strategies.
    strategy_labels_b : list[str], optional
        Human-readable labels for Player B's strategies.
    player_names : tuple[str, str], optional
        Names for the two players.

    Examples
    --------
    >>> # Create the Chicken Game
    >>> payoff_a = np.array([[-100, 10], [2, 0]])
    >>> payoff_b = np.array([[-100, 2], [10, 0]])
    >>> game = NormalFormGame(payoff_a, payoff_b,
    ...     strategy_labels_a=["Go", "Yield"],
    ...     strategy_labels_b=["Go", "Yield"],
    ...     player_names=("Vehicle A", "Vehicle B"))
    >>> print(game)
    """

    def __init__(
        self,
        payoff_a: np.ndarray,
        payoff_b: np.ndarray,
        strategy_labels_a: list[str] | None = None,
        strategy_labels_b: list[str] | None = None,
        player_names: tuple[str, str] = ("Player A", "Player B"),
    ):
        self.payoff_a = np.array(payoff_a, dtype=float)
        self.payoff_b = np.array(payoff_b, dtype=float)

        if self.payoff_a.shape != self.payoff_b.shape:
            raise ValueError(
                f"Payoff matrices must have the same shape. "
                f"Got {self.payoff_a.shape} and {self.payoff_b.shape}."
            )

        self.num_strategies_a, self.num_strategies_b = self.payoff_a.shape
        self.player_names = player_names

        # Default strategy labels: S0, S1, S2, ...
        self.strategy_labels_a = strategy_labels_a or [
            f"S{i}" for i in range(self.num_strategies_a)
        ]
        self.strategy_labels_b = strategy_labels_b or [
            f"S{j}" for j in range(self.num_strategies_b)
        ]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the strategy dimensions (m × n)."""
        return self.payoff_a.shape

    @property
    def is_symmetric(self) -> bool:
        """Check if the game is symmetric (payoff_a == payoff_b^T)."""
        return np.allclose(self.payoff_a, self.payoff_b.T)

    @property
    def is_zero_sum(self) -> bool:
        """Check if the game is zero-sum (payoffs sum to zero everywhere)."""
        return np.allclose(self.payoff_a + self.payoff_b, 0)

    def payoff(self, i: int, j: int) -> tuple[float, float]:
        """
        Get the payoff pair for strategy profile (i, j).

        Parameters
        ----------
        i : int
            Player A's strategy index.
        j : int
            Player B's strategy index.

        Returns
        -------
        tuple[float, float]
            (payoff_A, payoff_B) for the given strategy profile.
        """
        return float(self.payoff_a[i, j]), float(self.payoff_b[i, j])

    def best_response_a(self, j: int) -> list[int]:
        """
        Find Player A's best response(s) to Player B playing strategy j.

        Parameters
        ----------
        j : int
            Player B's pure strategy index.

        Returns
        -------
        list[int]
            Indices of Player A's best-response strategies.
        """
        col = self.payoff_a[:, j]
        max_val = col.max()
        return [i for i in range(self.num_strategies_a) if col[i] == max_val]

    def best_response_b(self, i: int) -> list[int]:
        """
        Find Player B's best response(s) to Player A playing strategy i.

        Parameters
        ----------
        i : int
            Player A's pure strategy index.

        Returns
        -------
        list[int]
            Indices of Player B's best-response strategies.
        """
        row = self.payoff_b[i, :]
        max_val = row.max()
        return [j for j in range(self.num_strategies_b) if row[j] == max_val]

    def best_response_a_mixed(self, q: np.ndarray) -> list[int]:
        """
        Find Player A's best response(s) when Player B plays mixed strategy q.

        Parameters
        ----------
        q : np.ndarray
            Player B's mixed strategy (probability distribution over strategies).

        Returns
        -------
        list[int]
            Indices of Player A's best-response pure strategies.
        """
        expected = self.payoff_a @ q
        max_val = expected.max()
        return [i for i in range(self.num_strategies_a)
                if np.isclose(expected[i], max_val)]

    def best_response_b_mixed(self, p: np.ndarray) -> list[int]:
        """
        Find Player B's best response(s) when Player A plays mixed strategy p.

        Parameters
        ----------
        p : np.ndarray
            Player A's mixed strategy (probability distribution over strategies).

        Returns
        -------
        list[int]
            Indices of Player B's best-response pure strategies.
        """
        expected = p @ self.payoff_b
        max_val = expected.max()
        return [j for j in range(self.num_strategies_b)
                if np.isclose(expected[j], max_val)]

    def is_dominant_strategy(self, player: str, strategy: int) -> bool:
        """
        Check if a strategy is strictly dominant for a given player.

        A strategy is strictly dominant if it yields a higher payoff
        than every other strategy regardless of the opponent's choice.

        Parameters
        ----------
        player : str
            "A" or "B".
        strategy : int
            Strategy index to check.

        Returns
        -------
        bool
            True if the strategy strictly dominates all others.
        """
        if player.upper() == "A":
            payoff_matrix = self.payoff_a
            n_strategies = self.num_strategies_a
            axis = 0  # Compare rows
        else:
            payoff_matrix = self.payoff_b.T
            n_strategies = self.num_strategies_b
            axis = 0

        for other in range(n_strategies):
            if other == strategy:
                continue
            if player.upper() == "A":
                if not np.all(payoff_matrix[strategy, :] > payoff_matrix[other, :]):
                    return False
            else:
                if not np.all(self.payoff_b[:, strategy] > self.payoff_b[:, other]):
                    return False
        return True

    def find_dominated_strategies(self, player: str) -> list[int]:
        """
        Find all strictly dominated strategies for a player.

        Parameters
        ----------
        player : str
            "A" or "B".

        Returns
        -------
        list[int]
            Indices of dominated strategies.
        """
        dominated = []
        if player.upper() == "A":
            for i in range(self.num_strategies_a):
                for other in range(self.num_strategies_a):
                    if other == i:
                        continue
                    if np.all(self.payoff_a[other, :] > self.payoff_a[i, :]):
                        dominated.append(i)
                        break
        else:
            for j in range(self.num_strategies_b):
                for other in range(self.num_strategies_b):
                    if other == j:
                        continue
                    if np.all(self.payoff_b[:, other] > self.payoff_b[:, j]):
                        dominated.append(j)
                        break
        return dominated

    def social_welfare(self, i: int, j: int) -> float:
        """
        Compute the social welfare (sum of payoffs) for a strategy profile.

        Parameters
        ----------
        i : int
            Player A's strategy index.
        j : int
            Player B's strategy index.

        Returns
        -------
        float
            Sum of both players' payoffs.
        """
        return float(self.payoff_a[i, j] + self.payoff_b[i, j])

    def pareto_optimal_outcomes(self) -> list[tuple[int, int]]:
        """
        Find all Pareto optimal strategy profiles.

        A profile (i, j) is Pareto optimal if no other profile
        makes one player better off without making the other worse off.

        Returns
        -------
        list[tuple[int, int]]
            List of Pareto optimal strategy profile indices.
        """
        profiles = list(product(
            range(self.num_strategies_a),
            range(self.num_strategies_b),
        ))
        pareto = []

        for i, j in profiles:
            is_dominated = False
            for i2, j2 in profiles:
                if (i2, j2) == (i, j):
                    continue
                # Check if (i2, j2) Pareto-dominates (i, j)
                if (self.payoff_a[i2, j2] >= self.payoff_a[i, j] and
                    self.payoff_b[i2, j2] >= self.payoff_b[i, j] and
                    (self.payoff_a[i2, j2] > self.payoff_a[i, j] or
                     self.payoff_b[i2, j2] > self.payoff_b[i, j])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append((i, j))

        return pareto

    def expected_payoff_mixed(
        self, p: np.ndarray, q: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute expected payoffs under mixed strategy profiles.

        Parameters
        ----------
        p : np.ndarray
            Player A's mixed strategy.
        q : np.ndarray
            Player B's mixed strategy.

        Returns
        -------
        tuple[float, float]
            Expected payoffs (E[u_A], E[u_B]).
        """
        eu_a = float(p @ self.payoff_a @ q)
        eu_b = float(p @ self.payoff_b @ q)
        return eu_a, eu_b

    def __repr__(self) -> str:
        return (
            f"NormalFormGame(players={self.player_names}, "
            f"shape={self.shape}, "
            f"strategies_A={self.strategy_labels_a}, "
            f"strategies_B={self.strategy_labels_b})"
        )

    def __str__(self) -> str:
        """Pretty-print the game in a tabular format."""
        lines = [f"Game: {self.player_names[0]} vs {self.player_names[1]}"]
        lines.append(f"Strategies: {self.shape[0]} × {self.shape[1]}")
        lines.append("")

        # Header
        header = f"{'':>12}"
        for j, label in enumerate(self.strategy_labels_b):
            header += f"  {label:>14}"
        lines.append(header)
        lines.append("  " + "-" * (14 * self.num_strategies_b + 12))

        # Rows
        for i in range(self.num_strategies_a):
            row = f"{self.strategy_labels_a[i]:>12}"
            for j in range(self.num_strategies_b):
                a_val = self.payoff_a[i, j]
                b_val = self.payoff_b[i, j]
                row += f"  ({a_val:>5.0f},{b_val:>5.0f})"
            lines.append(row)

        return "\n".join(lines)
