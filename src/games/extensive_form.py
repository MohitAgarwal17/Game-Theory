"""
Extensive-Form (Sequential) Game representation.

An extensive-form game captures sequential decision-making via a game tree.
This is used to model Stackelberg (leader-follower) interactions where one
vehicle commits to an action before the other responds.

Key components:
- Game tree with decision nodes and terminal nodes
- Information sets (for imperfect information)
- Backward induction for subgame-perfect equilibrium
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TerminalNode:
    """
    A leaf node in the game tree representing an outcome.

    Attributes
    ----------
    name : str
        Human-readable description of the outcome.
    payoffs : dict[str, float]
        Mapping from player name to payoff value.
    """
    name: str
    payoffs: dict[str, float]

    @property
    def is_terminal(self) -> bool:
        return True


@dataclass
class DecisionNode:
    """
    An internal node where a player makes a decision.

    Attributes
    ----------
    player : str
        The player who decides at this node.
    actions : dict[str, DecisionNode | TerminalNode]
        Mapping from action label to the resulting subtree.
    info_set : str
        Information set label (for imperfect information games).
    """
    player: str
    actions: dict[str, "DecisionNode | TerminalNode"] = field(default_factory=dict)
    info_set: str = ""

    @property
    def is_terminal(self) -> bool:
        return False

    def add_action(self, action: str, child: "DecisionNode | TerminalNode"):
        """Add an action leading to a child node."""
        self.actions[action] = child


class ExtensiveFormGame:
    """
    Represents a finite extensive-form game (game tree).

    Supports sequential games with perfect or imperfect information.
    Provides backward induction for computing subgame-perfect equilibria
    (Stackelberg equilibria in the 2-player case).

    Parameters
    ----------
    root : DecisionNode
        The root of the game tree (first mover's decision).
    players : list[str]
        List of player names.
    name : str
        Descriptive name for the game.

    Examples
    --------
    >>> # Create a simple Stackelberg intersection game
    >>> # Vehicle A (leader) moves first, Vehicle B (follower) responds
    >>> game = ExtensiveFormGame.create_stackelberg_intersection()
    >>> eq = game.backward_induction()
    >>> print(eq)
    """

    def __init__(
        self,
        root: DecisionNode,
        players: list[str],
        name: str = "Extensive-Form Game",
    ):
        self.root = root
        self.players = players
        self.name = name

    @staticmethod
    def create_stackelberg_intersection(
        collision=-100,
        proceed=10,
        yield_to=2,
        deadlock=0,
        leader="Vehicle A",
        follower="Vehicle B",
    ) -> "ExtensiveFormGame":
        """
        Create a Stackelberg intersection game.

        The leader commits to Go or Yield. The follower observes
        the leader's action and best-responds.

        Parameters
        ----------
        collision, proceed, yield_to, deadlock : float
            Payoff values for different outcomes.
        leader, follower : str
            Player names.

        Returns
        -------
        ExtensiveFormGame
            The complete game tree.
        """
        # Terminal nodes after leader Goes
        t_go_go = TerminalNode(
            "Both Go → Collision",
            {leader: collision, follower: collision},
        )
        t_go_yield = TerminalNode(
            "Leader Goes, Follower Yields",
            {leader: proceed, follower: yield_to},
        )

        # Terminal nodes after leader Yields
        t_yield_go = TerminalNode(
            "Leader Yields, Follower Goes",
            {leader: yield_to, follower: proceed},
        )
        t_yield_yield = TerminalNode(
            "Both Yield → Deadlock",
            {leader: deadlock, follower: deadlock},
        )

        # Follower's decision nodes
        follower_after_go = DecisionNode(player=follower)
        follower_after_go.add_action("Go", t_go_go)
        follower_after_go.add_action("Yield", t_go_yield)

        follower_after_yield = DecisionNode(player=follower)
        follower_after_yield.add_action("Go", t_yield_go)
        follower_after_yield.add_action("Yield", t_yield_yield)

        # Leader's decision node (root)
        root = DecisionNode(player=leader)
        root.add_action("Go", follower_after_go)
        root.add_action("Yield", follower_after_yield)

        return ExtensiveFormGame(
            root=root,
            players=[leader, follower],
            name="Stackelberg Intersection Game",
        )

    # Alias
    create_intersection_tree = create_stackelberg_intersection


    def backward_induction(self) -> dict:
        """
        Solve the game using backward induction.

        Starting from terminal nodes and working upward, at each
        decision node the current player selects the action that
        maximizes their payoff.

        Returns
        -------
        dict
            Solution containing:
            - 'equilibrium_path': list of (player, action) tuples
            - 'equilibrium_payoffs': dict of player → payoff
            - 'strategies': dict of player → {info_set: action}
            - 'all_node_values': complete backward induction values
        """
        strategies = {p: {} for p in self.players}
        path = []

        def _solve(node, depth=0):
            """Recursively solve subtree, returning payoffs."""
            if node.is_terminal:
                return node.payoffs

            best_action = None
            best_payoffs = None

            for action, child in node.actions.items():
                child_payoffs = _solve(child, depth + 1)

                if (best_payoffs is None or
                        child_payoffs[node.player] > best_payoffs[node.player]):
                    best_action = action
                    best_payoffs = child_payoffs

            # Record this player's optimal action at this node
            info = node.info_set or f"depth_{depth}"
            strategies[node.player][info] = best_action

            return best_payoffs

        eq_payoffs = _solve(self.root)

        # Trace the equilibrium path
        node = self.root
        while not node.is_terminal:
            info = node.info_set or f"depth_{len(path)}"
            action = strategies[node.player][info]
            path.append((node.player, action))
            node = node.actions[action]

        return {
            "equilibrium_path": path,
            "equilibrium_payoffs": eq_payoffs,
            "strategies": strategies,
            "outcome": node.name if node.is_terminal else None,
        }

    def get_all_terminal_nodes(self) -> list[TerminalNode]:
        """
        Collect all terminal nodes in the game tree.

        Returns
        -------
        list[TerminalNode]
            All terminal (leaf) nodes.
        """
        terminals = []

        def _collect(node):
            if node.is_terminal:
                terminals.append(node)
            else:
                for child in node.actions.values():
                    _collect(child)

        _collect(self.root)
        return terminals

    def tree_to_string(self, node=None, prefix="", is_last=True) -> str:
        """
        Generate a string representation of the game tree.

        Returns
        -------
        str
            ASCII art tree representation.
        """
        if node is None:
            node = self.root

        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if node.is_terminal:
            payoff_str = ", ".join(
                f"{p}: {v}" for p, v in node.payoffs.items()
            )
            return f"{prefix}{connector}🏁 {node.name} [{payoff_str}]\n"

        result = f"{prefix}{connector}🎮 {node.player}'s turn\n"

        actions = list(node.actions.items())
        for idx, (action, child) in enumerate(actions):
            is_child_last = idx == len(actions) - 1
            action_prefix = prefix + extension
            result += f"{action_prefix}{'└── ' if is_child_last else '├── '}→ {action}\n"
            child_prefix = action_prefix + ("    " if is_child_last else "│   ")
            result += self.tree_to_string(
                child, child_prefix, True
            )

        return result

    def __str__(self) -> str:
        header = f"=== {self.name} ===\n"
        header += f"Players: {', '.join(self.players)}\n\n"
        header += "Game Tree:\n"
        return header + self.tree_to_string()
