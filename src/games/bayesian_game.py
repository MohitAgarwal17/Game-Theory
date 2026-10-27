"""
Bayesian Game representation for incomplete information scenarios.

In mixed traffic, an autonomous vehicle may not know whether the other
driver is aggressive or cautious. A Bayesian game models this uncertainty
by introducing player types with associated probability distributions.

Key concepts:
- Types (θ): Private information about a player's characteristics
- Beliefs (p): Prior probability distribution over opponent's types
- Type-contingent strategies: Action choices that depend on own type
"""

import numpy as np
from dataclasses import dataclass, field
from itertools import product


@dataclass
class PlayerType:
    """
    Represents a player type in a Bayesian game.

    Attributes
    ----------
    name : str
        Type identifier (e.g., "aggressive", "cautious").
    probability : float
        Prior probability of this type (common knowledge).
    description : str
        Human-readable description of the type's behavior.
    """
    name: str
    probability: float
    description: str = ""


class BayesianGame:
    """
    Represents a 2-player Bayesian game with discrete types.

    Each player has a set of types, and payoffs depend on the type
    profile as well as the action profile. Players know their own
    type but have only beliefs (priors) about the opponent's type.

    Parameters
    ----------
    types_a : list[PlayerType]
        Player A's possible types.
    types_b : list[PlayerType]
        Player B's possible types.
    actions_a : list[str]
        Player A's available actions.
    actions_b : list[str]
        Player B's available actions.
    payoffs : dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]
        Mapping from (type_a_name, type_b_name) to (payoff_A, payoff_B)
        matrices of shape (|actions_a|, |actions_b|).
    player_names : tuple[str, str]
        Names for the two players.

    Examples
    --------
    >>> # AV vs. unknown human driver
    >>> types_a = [PlayerType("av", 1.0, "Autonomous Vehicle")]
    >>> types_b = [
    ...     PlayerType("aggressive", 0.3, "Aggressive human driver"),
    ...     PlayerType("cautious", 0.7, "Cautious human driver"),
    ... ]
    >>> game = BayesianGame(types_a, types_b, ["Go", "Yield"], ["Go", "Yield"], payoffs)
    """

    def __init__(
        self,
        types_a: list[PlayerType],
        types_b: list[PlayerType],
        actions_a: list[str],
        actions_b: list[str],
        payoffs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
        player_names: tuple[str, str] = ("Player A", "Player B"),
    ):
        self.types_a = types_a
        self.types_b = types_b
        self.actions_a = actions_a
        self.actions_b = actions_b
        self.payoffs = payoffs
        self.player_names = player_names

        # Validate probabilities sum to 1
        prob_a = sum(t.probability for t in types_a)
        prob_b = sum(t.probability for t in types_b)
        if not np.isclose(prob_a, 1.0):
            raise ValueError(f"Player A type probabilities sum to {prob_a}, not 1.0")
        if not np.isclose(prob_b, 1.0):
            raise ValueError(f"Player B type probabilities sum to {prob_b}, not 1.0")

        # Validate that payoffs exist for all type combinations
        for ta in types_a:
            for tb in types_b:
                key = (ta.name, tb.name)
                if key not in payoffs:
                    raise ValueError(f"Missing payoff matrix for type pair {key}")

    def expected_payoff_a(
        self,
        type_a: str,
        action_a: int,
        strategy_b: dict[str, int],
    ) -> float:
        """
        Compute Player A's expected payoff given their type and action,
        and Player B's type-contingent strategy.

        Parameters
        ----------
        type_a : str
            Player A's type name.
        action_a : int
            Player A's action index.
        strategy_b : dict[str, int]
            Player B's strategy: type_name → action_index.

        Returns
        -------
        float
            Expected payoff for Player A.
        """
        expected = 0.0
        for tb in self.types_b:
            action_b = strategy_b[tb.name]
            payoff_matrix_a = self.payoffs[(type_a, tb.name)][0]
            expected += tb.probability * payoff_matrix_a[action_a, action_b]
        return expected

    def expected_payoff_b(
        self,
        type_b: str,
        action_b: int,
        strategy_a: dict[str, int],
    ) -> float:
        """
        Compute Player B's expected payoff given their type and action,
        and Player A's type-contingent strategy.

        Parameters
        ----------
        type_b : str
            Player B's type name.
        action_b : int
            Player B's action index.
        strategy_a : dict[str, int]
            Player A's strategy: type_name → action_index.

        Returns
        -------
        float
            Expected payoff for Player B.
        """
        expected = 0.0
        for ta in self.types_a:
            action_a = strategy_a[ta.name]
            payoff_matrix_b = self.payoffs[(ta.name, type_b)][1]
            expected += ta.probability * payoff_matrix_b[action_a, action_b]
        return expected

    def find_bayesian_ne(self) -> list[dict]:
        """
        Find all pure-strategy Bayesian Nash Equilibria by exhaustive search.

        A Bayesian NE is a type-contingent strategy profile where each type
        of each player is best-responding to the other player's strategy,
        given their beliefs about the opponent's type.

        Returns
        -------
        list[dict]
            List of BNE, each containing:
            - 'strategy_a': {type_name: action_index}
            - 'strategy_b': {type_name: action_index}
            - 'expected_payoffs': {type_name: expected_payoff}
        """
        equilibria = []
        n_a = len(self.actions_a)
        n_b = len(self.actions_b)

        # Generate all possible type-contingent strategies
        # Player A: for each type, choose an action
        a_strategies = list(product(range(n_a), repeat=len(self.types_a)))
        b_strategies = list(product(range(n_b), repeat=len(self.types_b)))

        for sa_tuple in a_strategies:
            for sb_tuple in b_strategies:
                # Convert to dict
                strategy_a = {
                    self.types_a[i].name: sa_tuple[i]
                    for i in range(len(self.types_a))
                }
                strategy_b = {
                    self.types_b[j].name: sb_tuple[j]
                    for j in range(len(self.types_b))
                }

                is_equilibrium = True

                # Check if each type of A is best-responding
                for ta in self.types_a:
                    current_payoff = self.expected_payoff_a(
                        ta.name, strategy_a[ta.name], strategy_b
                    )
                    for alt_action in range(n_a):
                        alt_payoff = self.expected_payoff_a(
                            ta.name, alt_action, strategy_b
                        )
                        if alt_payoff > current_payoff + 1e-9:
                            is_equilibrium = False
                            break
                    if not is_equilibrium:
                        break

                if not is_equilibrium:
                    continue

                # Check if each type of B is best-responding
                for tb in self.types_b:
                    current_payoff = self.expected_payoff_b(
                        tb.name, strategy_b[tb.name], strategy_a
                    )
                    for alt_action in range(n_b):
                        alt_payoff = self.expected_payoff_b(
                            tb.name, alt_action, strategy_a
                        )
                        if alt_payoff > current_payoff + 1e-9:
                            is_equilibrium = False
                            break
                    if not is_equilibrium:
                        break

                if is_equilibrium:
                    # Compute expected payoffs for each type
                    payoffs = {}
                    for ta in self.types_a:
                        payoffs[f"A_{ta.name}"] = self.expected_payoff_a(
                            ta.name, strategy_a[ta.name], strategy_b
                        )
                    for tb in self.types_b:
                        payoffs[f"B_{tb.name}"] = self.expected_payoff_b(
                            tb.name, strategy_b[tb.name], strategy_a
                        )

                    equilibria.append({
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "expected_payoffs": payoffs,
                        "strategy_a_labels": {
                            t: self.actions_a[a]
                            for t, a in strategy_a.items()
                        },
                        "strategy_b_labels": {
                            t: self.actions_b[a]
                            for t, a in strategy_b.items()
                        },
                    })

        return equilibria

    @staticmethod
    def create_av_vs_human(
        prob_aggressive: float = 0.3,
    ) -> "BayesianGame":
        """
        Factory: Create an AV vs. unknown human driver Bayesian game.

        The AV knows it's rational (one type). The human driver may be
        aggressive (probability prob_aggressive) or cautious.

        Parameters
        ----------
        prob_aggressive : float
            Prior probability that the human is aggressive.

        Returns
        -------
        BayesianGame
            The constructed Bayesian game.
        """
        types_a = [PlayerType("av", 1.0, "Autonomous Vehicle (rational)")]
        types_b = [
            PlayerType(
                "aggressive", prob_aggressive,
                "Aggressive: high Go bias, underestimates collision risk"
            ),
            PlayerType(
                "cautious", 1.0 - prob_aggressive,
                "Cautious: high Yield bias, safety-first"
            ),
        ]

        payoffs = {
            ("av", "aggressive"): (
                np.array([[-80, 10], [2, 0]], dtype=float),
                np.array([[-80, 1], [15, -2]], dtype=float),
            ),
            ("av", "cautious"): (
                np.array([[-120, 10], [2, 0]], dtype=float),
                np.array([[-120, 2], [8, 0]], dtype=float),
            ),
        }

        return BayesianGame(
            types_a=types_a,
            types_b=types_b,
            actions_a=["Go", "Yield"],
            actions_b=["Go", "Yield"],
            payoffs=payoffs,
            player_names=("AV", "Human Driver"),
        )

    def __str__(self) -> str:
        lines = [f"Bayesian Game: {self.player_names[0]} vs {self.player_names[1]}"]
        lines.append(f"Actions A: {self.actions_a}")
        lines.append(f"Actions B: {self.actions_b}")
        lines.append("")

        lines.append(f"{self.player_names[0]} Types:")
        for t in self.types_a:
            lines.append(f"  {t.name} (p={t.probability:.2f}): {t.description}")

        lines.append(f"\n{self.player_names[1]} Types:")
        for t in self.types_b:
            lines.append(f"  {t.name} (p={t.probability:.2f}): {t.description}")

        return "\n".join(lines)
