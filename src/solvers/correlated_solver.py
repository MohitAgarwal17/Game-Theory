"""
Correlated Equilibrium Solver.

A correlated equilibrium generalizes Nash equilibrium by allowing players
to coordinate via a shared signal (e.g., a traffic light). This module
uses linear programming to compute correlated equilibria that maximize
social welfare.

In the intersection context, a traffic light acts as a correlating device:
- Green/Red signals tell each vehicle whether to Go or Yield
- Players voluntarily follow the signal because deviating is worse
"""

import numpy as np
from scipy.optimize import linprog
from dataclasses import dataclass

from src.games.normal_form import NormalFormGame


@dataclass
class CorrelatedEquilibriumResult:
    """
    Result of correlated equilibrium computation.

    Attributes
    ----------
    distribution : np.ndarray
        Joint probability distribution over action profiles.
        Shape: (m, n) where distribution[i,j] = Pr(A plays i, B plays j).
    expected_payoff_a : float
        Player A's expected payoff under the CE.
    expected_payoff_b : float
        Player B's expected payoff under the CE.
    social_welfare : float
        Sum of expected payoffs.
    is_valid : bool
        Whether the solution satisfies all CE constraints.
    signal_interpretation : dict
        Human-readable interpretation of the correlating device.
    """
    distribution: np.ndarray
    expected_payoff_a: float
    expected_payoff_b: float
    social_welfare: float
    is_valid: bool
    signal_interpretation: dict


class CorrelatedSolver:
    """
    Computes correlated equilibria using linear programming.

    The CE is found by maximizing social welfare subject to:
    1. Incentive compatibility constraints (no player wants to deviate)
    2. Probability constraints (non-negative, sums to 1)

    Examples
    --------
    >>> from src.utils.payoff_matrices import chicken_game
    >>> game = NormalFormGame(*chicken_game(),
    ...     strategy_labels_a=["Go", "Yield"],
    ...     strategy_labels_b=["Go", "Yield"])
    >>> solver = CorrelatedSolver()
    >>> ce = solver.solve(game)
    >>> print(f"Social Welfare: {ce.social_welfare}")
    """


    def solve(
        self,
        game: NormalFormGame,
        objective: str = "max_welfare",
    ) -> CorrelatedEquilibriumResult:
        """
        Compute a correlated equilibrium using linear programming.

        Parameters
        ----------
        game : NormalFormGame
            The game to solve.
        objective : str
            Optimization objective:
            - "max_welfare": maximize social welfare (default)
            - "max_fairness": minimize payoff difference
            - "max_min_payoff": maximize the minimum payoff

        Returns
        -------
        CorrelatedEquilibriumResult
            The optimal correlated equilibrium.
        """
        m, n = game.shape
        num_vars = m * n  # Variables: p[i,j] for each action pair

        # Decision variables are flattened: x[i*n + j] = p(i, j)

        # Objective: maximize social welfare = sum of expected payoffs
        # linprog minimizes, so negate for maximization
        c = np.zeros(num_vars)
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                if objective == "max_welfare":
                    c[idx] = -(game.payoff_a[i, j] + game.payoff_b[i, j])
                elif objective == "max_min_payoff":
                    c[idx] = -min(game.payoff_a[i, j], game.payoff_b[i, j])
                else:
                    c[idx] = -(game.payoff_a[i, j] + game.payoff_b[i, j])

        # Inequality constraints: incentive compatibility
        # For Player A: for each pair of actions (i, i'),
        #   sum_j p(i,j) * [u_A(i,j) - u_A(i',j)] >= 0
        # Equivalently: sum_j p(i,j) * [u_A(i',j) - u_A(i,j)] <= 0

        A_ub_list = []
        b_ub_list = []

        # Player A incentive constraints
        for i in range(m):
            for i_prime in range(m):
                if i == i_prime:
                    continue
                row = np.zeros(num_vars)
                for j in range(n):
                    idx = i * n + j
                    # Deviation gain: u_A(i', j) - u_A(i, j)
                    row[idx] = game.payoff_a[i_prime, j] - game.payoff_a[i, j]
                A_ub_list.append(row)
                b_ub_list.append(0.0)

        # Player B incentive constraints
        for j in range(n):
            for j_prime in range(n):
                if j == j_prime:
                    continue
                row = np.zeros(num_vars)
                for i in range(m):
                    idx = i * n + j
                    # Deviation gain: u_B(i, j') - u_B(i, j)
                    row[idx] = game.payoff_b[i, j_prime] - game.payoff_b[i, j]
                A_ub_list.append(row)
                b_ub_list.append(0.0)

        A_ub = np.array(A_ub_list) if A_ub_list else None
        b_ub = np.array(b_ub_list) if b_ub_list else None

        # Equality constraint: probabilities sum to 1
        A_eq = np.ones((1, num_vars))
        b_eq = np.array([1.0])

        # Bounds: all probabilities >= 0
        bounds = [(0, 1) for _ in range(num_vars)]

        # Solve LP
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs'
        )

        if result.success:
            distribution = result.x.reshape(m, n)

            # Compute expected payoffs
            eu_a = float(np.sum(distribution * game.payoff_a))
            eu_b = float(np.sum(distribution * game.payoff_b))

            # Generate signal interpretation
            signals = {}
            for i in range(m):
                for j in range(n):
                    if distribution[i, j] > 1e-6:
                        label_a = game.strategy_labels_a[i]
                        label_b = game.strategy_labels_b[j]
                        prob = distribution[i, j]
                        signals[f"Signal: A={label_a}, B={label_b}"] = (
                            f"Probability {prob:.4f}"
                        )

            return CorrelatedEquilibriumResult(
                distribution=distribution,
                expected_payoff_a=eu_a,
                expected_payoff_b=eu_b,
                social_welfare=eu_a + eu_b,
                is_valid=True,
                signal_interpretation=signals,
            )
        else:
            # Return uniform distribution as fallback
            distribution = np.ones((m, n)) / num_vars
            eu_a = float(np.sum(distribution * game.payoff_a))
            eu_b = float(np.sum(distribution * game.payoff_b))

            return CorrelatedEquilibriumResult(
                distribution=distribution,
                expected_payoff_a=eu_a,
                expected_payoff_b=eu_b,
                social_welfare=eu_a + eu_b,
                is_valid=False,
                signal_interpretation={"error": "LP did not converge"},
            )

    def compare_with_nash(
        self, game: NormalFormGame, nash_equilibria: list
    ) -> dict:
        """
        Compare the correlated equilibrium with Nash equilibria.

        Parameters
        ----------
        game : NormalFormGame
            The game.
        nash_equilibria : list
            List of NashEquilibrium objects.

        Returns
        -------
        dict
            Comparison results showing welfare improvements.
        """
        ce = self.solve(game)

        ne_welfares = [ne.social_welfare for ne in nash_equilibria]
        best_ne = max(ne_welfares) if ne_welfares else 0
        worst_ne = min(ne_welfares) if ne_welfares else 0
        avg_ne = np.mean(ne_welfares) if ne_welfares else 0

        return {
            "ce_welfare": ce.social_welfare,
            "best_ne_welfare": best_ne,
            "worst_ne_welfare": worst_ne,
            "avg_ne_welfare": avg_ne,
            "improvement_over_worst_ne": (
                (ce.social_welfare - worst_ne) / abs(worst_ne) * 100
                if worst_ne != 0 else float("inf")
            ),
            "improvement_over_avg_ne": (
                (ce.social_welfare - avg_ne) / abs(avg_ne) * 100
                if avg_ne != 0 else float("inf")
            ),
            "ce_distribution": ce.distribution,
            "ce_signals": ce.signal_interpretation,
        }

    def traffic_light_equilibrium(
        self, game: NormalFormGame
    ) -> CorrelatedEquilibriumResult:
        """
        Compute the specific CE that corresponds to a fair traffic light.

        A traffic light alternates: 50% of the time A has green (Go),
        50% of the time B has green (Go). When one has green, the other
        has red (Yield).

        Parameters
        ----------
        game : NormalFormGame
            The 2×2 intersection game.

        Returns
        -------
        CorrelatedEquilibriumResult
            The traffic-light CE.
        """
        # Traffic light: p(Go, Yield) = 0.5, p(Yield, Go) = 0.5
        distribution = np.zeros(game.shape)
        distribution[0, 1] = 0.5  # A goes, B yields
        distribution[1, 0] = 0.5  # B goes, A yields

        eu_a = float(np.sum(distribution * game.payoff_a))
        eu_b = float(np.sum(distribution * game.payoff_b))

        signals = {
            "Green for A, Red for B": "50% — A proceeds, B waits",
            "Red for A, Green for B": "50% — B proceeds, A waits",
        }

        # Verify incentive compatibility
        is_valid = self._verify_ic(game, distribution)

        return CorrelatedEquilibriumResult(
            distribution=distribution,
            expected_payoff_a=eu_a,
            expected_payoff_b=eu_b,
            social_welfare=eu_a + eu_b,
            is_valid=is_valid,
            signal_interpretation=signals,
        )

    def _verify_ic(
        self, game: NormalFormGame, distribution: np.ndarray
    ) -> bool:
        """Verify incentive compatibility constraints."""
        m, n = game.shape

        # Check Player A's IC
        for i in range(m):
            for i_prime in range(m):
                if i == i_prime:
                    continue
                gain = sum(
                    distribution[i, j] * (
                        game.payoff_a[i_prime, j] - game.payoff_a[i, j]
                    )
                    for j in range(n)
                )
                if gain > 1e-6:
                    return False

        # Check Player B's IC
        for j in range(n):
            for j_prime in range(n):
                if j == j_prime:
                    continue
                gain = sum(
                    distribution[i, j] * (
                        game.payoff_b[i, j_prime] - game.payoff_b[i, j]
                    )
                    for i in range(m)
                )
                if gain > 1e-6:
                    return False

        return True


# Alias for backward compatibility
CorrelatedEquilibriumSolver = CorrelatedSolver
