"""
Stackelberg Equilibrium Solver.

Computes subgame-perfect equilibria for sequential (extensive-form) games
using backward induction. In the AV context, the leader vehicle commits
to an action first, and the follower observes and best-responds.
"""

import numpy as np
from dataclasses import dataclass

from src.games.normal_form import NormalFormGame
from src.games.extensive_form import ExtensiveFormGame


@dataclass
class StackelbergEquilibrium:
    """
    Represents a Stackelberg (subgame-perfect) equilibrium.

    Attributes
    ----------
    leader_action : int
        The leader's optimal action index.
    follower_response : dict[int, int]
        Mapping from leader's action to follower's best response.
    leader_payoff : float
        Leader's equilibrium payoff.
    follower_payoff : float
        Follower's equilibrium payoff.
    equilibrium_path : list[tuple[str, str]]
        Sequence of (player, action) on the equilibrium path.
    outcome : str
        Description of the equilibrium outcome.
    """
    leader_action: int
    follower_response: dict[int, int]
    leader_payoff: float
    follower_payoff: float
    equilibrium_path: list[tuple[str, str]]
    outcome: str = ""


class StackelbergSolver:
    """
    Solver for Stackelberg equilibria in leader-follower games.

    Supports two approaches:
    1. Direct computation from a normal-form game (converting to sequential)
    2. Backward induction on an explicit extensive-form game tree

    Examples
    --------
    >>> from src.utils.payoff_matrices import chicken_game
    >>> game = NormalFormGame(*chicken_game(),
    ...     strategy_labels_a=["Go", "Yield"],
    ...     strategy_labels_b=["Go", "Yield"],
    ...     player_names=("Leader", "Follower"))
    >>> solver = StackelbergSolver()
    >>> eq = solver.solve_from_normal_form(game)
    >>> print(f"Leader: {eq.leader_action}, Payoff: {eq.leader_payoff}")
    """

    def solve_from_normal_form(
        self, game: NormalFormGame | tuple
    ) -> StackelbergEquilibrium:
        """
        Compute the Stackelberg equilibrium from a normal-form game.

        Player A is the leader, Player B is the follower.
        For each possible leader action, the follower best-responds.
        The leader then picks the action that maximizes their payoff
        given the follower's rational response.

        Parameters
        ----------
        game : NormalFormGame or tuple[np.ndarray, np.ndarray]
            The stage game (Player A = leader).

        Returns
        -------
        StackelbergEquilibrium
            The computed Stackelberg equilibrium.
        """
        if isinstance(game, tuple) and len(game) == 2:
            labels_a = ["Go", "Yield"] if game[0].shape[0] == 2 else [f"S{i}" for i in range(game[0].shape[0])]
            labels_b = ["Go", "Yield"] if game[0].shape[1] == 2 else [f"S{j}" for j in range(game[0].shape[1])]
            game = NormalFormGame(game[0], game[1], strategy_labels_a=labels_a, strategy_labels_b=labels_b)

        best_leader_payoff = float("-inf")

        best_leader_action = 0
        follower_responses = {}

        for i in range(game.num_strategies_a):
            # Follower's best response to leader action i
            br = game.best_response_b(i)
            # If ties, follower breaks in favor of leader (optimistic Stackelberg)
            best_j = max(br, key=lambda j: game.payoff_a[i, j])
            follower_responses[i] = best_j

            leader_payoff = game.payoff_a[i, best_j]
            if leader_payoff > best_leader_payoff:
                best_leader_payoff = leader_payoff
                best_leader_action = i

        best_j = follower_responses[best_leader_action]
        follower_payoff = game.payoff_b[best_leader_action, best_j]

        path = [
            (game.player_names[0], game.strategy_labels_a[best_leader_action]),
            (game.player_names[1], game.strategy_labels_b[best_j]),
        ]

        return StackelbergEquilibrium(
            leader_action=best_leader_action,
            follower_response=follower_responses,
            leader_payoff=best_leader_payoff,
            follower_payoff=follower_payoff,
            equilibrium_path=path,
            outcome=f"Leader plays {game.strategy_labels_a[best_leader_action]}, "
                    f"Follower responds with {game.strategy_labels_b[best_j]}",
        )

    def solve_from_extensive_form(
        self, game: ExtensiveFormGame
    ) -> StackelbergEquilibrium:
        """
        Solve a Stackelberg game from its extensive-form representation
        using backward induction.

        Parameters
        ----------
        game : ExtensiveFormGame
            The game tree to solve.

        Returns
        -------
        StackelbergEquilibrium
            The computed equilibrium.
        """
        result = game.backward_induction()

        leader_action = -1
        follower_responses = {}

        if result["equilibrium_path"]:
            leader_name, leader_act = result["equilibrium_path"][0]
            # Map action label to index (Go=0, Yield=1)
            action_map = {"Go": 0, "Yield": 1}
            leader_action = action_map.get(leader_act, 0)

        return StackelbergEquilibrium(
            leader_action=leader_action,
            follower_response=follower_responses,
            leader_payoff=result["equilibrium_payoffs"].get(
                game.players[0], 0
            ),
            follower_payoff=result["equilibrium_payoffs"].get(
                game.players[1], 0
            ),
            equilibrium_path=result["equilibrium_path"],
            outcome=result.get("outcome", ""),
        )

    # Alias
    solve_extensive_game = solve_from_extensive_form

    def compare_leader_advantage(

        self, game: NormalFormGame
    ) -> dict:
        """
        Compare outcomes when each player is the leader.

        Computes the Stackelberg equilibrium with A as leader and
        with B as leader, showing the first-mover advantage.

        Parameters
        ----------
        game : NormalFormGame
            The base game.

        Returns
        -------
        dict
            Comparison results showing payoffs for each leadership scenario.
        """
        # A leads
        eq_a_leads = self.solve_from_normal_form(game)

        # B leads: swap payoff matrices
        swapped_game = NormalFormGame(
            payoff_a=game.payoff_b.T,
            payoff_b=game.payoff_a.T,
            strategy_labels_a=game.strategy_labels_b,
            strategy_labels_b=game.strategy_labels_a,
            player_names=(game.player_names[1], game.player_names[0]),
        )
        eq_b_leads = self.solve_from_normal_form(swapped_game)

        return {
            "a_leads": {
                "leader_payoff": eq_a_leads.leader_payoff,
                "follower_payoff": eq_a_leads.follower_payoff,
                "outcome": eq_a_leads.outcome,
                "social_welfare": eq_a_leads.leader_payoff + eq_a_leads.follower_payoff,
            },
            "b_leads": {
                "leader_payoff": eq_b_leads.leader_payoff,
                "follower_payoff": eq_b_leads.follower_payoff,
                "outcome": eq_b_leads.outcome,
                "social_welfare": eq_b_leads.leader_payoff + eq_b_leads.follower_payoff,
            },
            "first_mover_advantage_a": (
                eq_a_leads.leader_payoff - eq_a_leads.follower_payoff
            ),
            "first_mover_advantage_b": (
                eq_b_leads.leader_payoff - eq_b_leads.follower_payoff
            ),
        }
