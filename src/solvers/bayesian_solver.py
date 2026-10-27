"""
Bayesian Nash Equilibrium Solver.

Thin wrapper around BayesianGame.find_bayesian_ne() with additional
analysis utilities: sensitivity analysis across prior beliefs, expected
collision probability, and optimal AV conservatism level.
"""

import numpy as np
from dataclasses import dataclass

from src.games.bayesian_game import BayesianGame


@dataclass
class BayesianNEResult:
    """
    Extended result of Bayesian NE analysis.

    Attributes
    ----------
    equilibria : list[dict]
        List of BNE found.
    num_equilibria : int
        Number of BNE.
    av_yield_rate : float
        How often the AV yields across all equilibria.
    expected_collision_rate : float
        Expected collision probability.
    """
    equilibria: list[dict]
    num_equilibria: int
    av_yield_rate: float
    expected_collision_rate: float


class BayesianSolver:
    """
    Solver for Bayesian Nash Equilibria with analysis utilities.

    Examples
    --------
    >>> game = BayesianGame.create_av_vs_human(prob_aggressive=0.3)
    >>> solver = BayesianSolver()
    >>> result = solver.solve(game)
    >>> print(f"AV Yield Rate: {result.av_yield_rate:.1%}")
    """

    def solve(self, game: BayesianGame) -> BayesianNEResult:
        """
        Find all BNE and compute summary statistics.

        Parameters
        ----------
        game : BayesianGame
            The Bayesian game to solve.

        Returns
        -------
        BayesianNEResult
            Equilibria with analysis metrics.
        """
        equilibria = game.find_bayesian_ne()

        # Compute AV yield rate
        yield_count = 0
        total = 0
        collision_rates = []

        for eq in equilibria:
            for type_name, action_idx in eq["strategy_a"].items():
                total += 1
                if action_idx == 1:  # Yield
                    yield_count += 1

            # Estimate collision probability
            coll = 0.0
            for ta in game.types_a:
                for tb in game.types_b:
                    if (eq["strategy_a"][ta.name] == 0 and
                            eq["strategy_b"][tb.name] == 0):
                        coll += ta.probability * tb.probability
            collision_rates.append(coll)

        av_yield = yield_count / total if total > 0 else 0.0
        avg_collision = np.mean(collision_rates) if collision_rates else 0.0

        return BayesianNEResult(
            equilibria=equilibria,
            num_equilibria=len(equilibria),
            av_yield_rate=av_yield,
            expected_collision_rate=avg_collision,
        )

    def sensitivity_analysis(
        self,
        prob_range: np.ndarray | None = None,
    ) -> dict:
        """
        Analyze how BNE changes as the prior belief about aggressive
        drivers varies.

        Parameters
        ----------
        prob_range : np.ndarray, optional
            Range of p(aggressive) values to test. Default: 0.0 to 1.0
            in steps of 0.05.

        Returns
        -------
        dict
            Contains arrays of:
            - 'prob_aggressive': tested probabilities
            - 'num_equilibria': number of BNE at each probability
            - 'av_yield_rates': AV yield rate at each probability
            - 'collision_rates': expected collision rate at each probability
        """
        if prob_range is None:
            prob_range = np.arange(0.0, 1.05, 0.05)

        results = {
            "prob_aggressive": [],
            "num_equilibria": [],
            "av_yield_rates": [],
            "collision_rates": [],
        }

        for p in prob_range:
            p_clamped = np.clip(p, 0.01, 0.99)  # Avoid degenerate cases
            game = BayesianGame.create_av_vs_human(prob_aggressive=p_clamped)
            result = self.solve(game)

            results["prob_aggressive"].append(float(p_clamped))
            results["num_equilibria"].append(result.num_equilibria)
            results["av_yield_rates"].append(result.av_yield_rate)
            results["collision_rates"].append(result.expected_collision_rate)

        return results
