"""
Repeated Game framework for intersection interactions over time.

When vehicles encounter the same intersection repeatedly (e.g., daily
commuters), they can build reputations and learn cooperation strategies.
This module implements repeated game simulation with configurable
strategies including Tit-for-Tat, Grim Trigger, and Win-Stay Lose-Shift.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from src.games.normal_form import NormalFormGame


class StrategyType(Enum):
    """Available strategies for repeated games."""
    ALWAYS_GO = "always_go"
    ALWAYS_YIELD = "always_yield"
    TIT_FOR_TAT = "tit_for_tat"
    GRIM_TRIGGER = "grim_trigger"
    WIN_STAY_LOSE_SHIFT = "win_stay_lose_shift"
    RANDOM = "random"
    PAVLOV = "pavlov"


@dataclass
class RoundResult:
    """
    Result of a single round of the repeated game.

    Attributes
    ----------
    round_num : int
        Round number (0-indexed).
    action_a : int
        Player A's action index.
    action_b : int
        Player B's action index.
    payoff_a : float
        Player A's payoff this round.
    payoff_b : float
        Player B's payoff this round.
    """
    round_num: int
    action_a: int
    action_b: int
    payoff_a: float
    payoff_b: float

    @property
    def both_cooperated(self) -> bool:
        """Check if both players yielded (cooperative outcome)."""
        # In the intersection context: Yield=1 is "cooperation"
        # Go=0 is "defection"
        return self.action_a == 1 and self.action_b == 1

    @property
    def collision(self) -> bool:
        """Check if both chose Go (collision)."""
        return self.action_a == 0 and self.action_b == 0


class RepeatedGameStrategy:
    """
    Implements various strategies for the repeated intersection game.

    In the intersection context:
    - Action 0 = Go (defect / aggressive)
    - Action 1 = Yield (cooperate / cautious)
    """

    def __init__(self, strategy_type: StrategyType, seed: int | None = None):
        self.strategy_type = strategy_type
        self.rng = np.random.default_rng(seed)
        self._has_been_defected = False  # For Grim Trigger

    def choose_action(
        self, history: list[RoundResult], player: str
    ) -> int:
        """
        Choose an action based on the strategy and game history.

        Parameters
        ----------
        history : list[RoundResult]
            List of past round results.
        player : str
            "A" or "B" — which player this strategy controls.

        Returns
        -------
        int
            Action index (0=Go, 1=Yield).
        """
        match self.strategy_type:
            case StrategyType.ALWAYS_GO:
                return 0

            case StrategyType.ALWAYS_YIELD:
                return 1

            case StrategyType.TIT_FOR_TAT:
                return self._tit_for_tat(history, player)

            case StrategyType.GRIM_TRIGGER:
                return self._grim_trigger(history, player)

            case StrategyType.WIN_STAY_LOSE_SHIFT:
                return self._wsls(history, player)

            case StrategyType.RANDOM:
                return int(self.rng.random() < 0.5)

            case StrategyType.PAVLOV:
                return self._pavlov(history, player)

            case _:
                return 1  # Default: Yield

    def _tit_for_tat(self, history: list[RoundResult], player: str) -> int:
        """Start cooperative, then copy opponent's last action."""
        if not history:
            return 1  # Cooperate (Yield) first

        last = history[-1]
        # Copy what the opponent did last round
        if player == "A":
            return last.action_b  # Mirror B's last action
        else:
            return last.action_a  # Mirror A's last action

    def _grim_trigger(self, history: list[RoundResult], player: str) -> int:
        """Cooperate until opponent defects, then defect forever."""
        if not history:
            self._has_been_defected = False
            return 1  # Start cooperative

        if self._has_been_defected:
            return 0  # Permanent retaliation

        last = history[-1]
        opponent_action = last.action_b if player == "A" else last.action_a
        if opponent_action == 0:  # Opponent chose Go (defected)
            self._has_been_defected = True
            return 0

        return 1  # Continue cooperating

    def _wsls(self, history: list[RoundResult], player: str) -> int:
        """Win-Stay, Lose-Shift: Repeat action if payoff was good, switch otherwise."""
        if not history:
            return 1  # Start cooperative

        last = history[-1]
        my_payoff = last.payoff_a if player == "A" else last.payoff_b
        my_action = last.action_a if player == "A" else last.action_b

        # "Win" threshold: positive payoff
        if my_payoff > 0:
            return my_action  # Stay with current strategy
        else:
            return 1 - my_action  # Shift

    def _pavlov(self, history: list[RoundResult], player: str) -> int:
        """Pavlov: Cooperate if last outcome was mutual cooperation or mutual defection."""
        if not history:
            return 1

        last = history[-1]
        # Cooperate if both did the same thing last round
        if last.action_a == last.action_b:
            return 1  # Cooperate
        else:
            return 0  # Defect


class RepeatedGame:
    """
    Simulates a repeated game between two players with configurable strategies.

    Parameters
    ----------
    stage_game : NormalFormGame
        The one-shot game played each round.
    num_rounds : int
        Number of rounds to play.
    discount_factor : float
        Discount factor δ ∈ (0, 1] for future payoffs.
    strategy_a : StrategyType
        Player A's repeated game strategy.
    strategy_b : StrategyType
        Player B's repeated game strategy.

    Examples
    --------
    >>> from src.utils.payoff_matrices import chicken_game
    >>> stage = NormalFormGame(*chicken_game())
    >>> rg = RepeatedGame(stage, 100, 0.95,
    ...     StrategyType.TIT_FOR_TAT, StrategyType.GRIM_TRIGGER)
    >>> results = rg.play()
    """

    def __init__(
        self,
        stage_game: NormalFormGame,
        num_rounds: int = 100,
        discount_factor: float = 0.95,
        strategy_a: StrategyType = StrategyType.TIT_FOR_TAT,
        strategy_b: StrategyType = StrategyType.TIT_FOR_TAT,
        seed: int | None = None,
    ):
        self.stage_game = stage_game
        self.num_rounds = num_rounds
        self.discount_factor = discount_factor
        self.strategy_a_obj = RepeatedGameStrategy(strategy_a, seed)
        self.strategy_b_obj = RepeatedGameStrategy(strategy_b, seed)
        self.strategy_a_type = strategy_a
        self.strategy_b_type = strategy_b
        self.history: list[RoundResult] = []

    def play(self) -> dict:
        """
        Play the full repeated game.

        Returns
        -------
        dict
            Results containing:
            - 'history': list of RoundResult
            - 'total_payoff_a': discounted sum of A's payoffs
            - 'total_payoff_b': discounted sum of B's payoffs
            - 'cooperation_rate_a': fraction of rounds A yielded
            - 'cooperation_rate_b': fraction of rounds B yielded
            - 'collision_count': number of mutual Go outcomes
            - 'avg_social_welfare': average per-round social welfare
        """
        self.history = []
        total_a = 0.0
        total_b = 0.0
        collisions = 0
        cooperation_a = 0
        cooperation_b = 0

        for t in range(self.num_rounds):
            action_a = self.strategy_a_obj.choose_action(self.history, "A")
            action_b = self.strategy_b_obj.choose_action(self.history, "B")

            payoff_a, payoff_b = self.stage_game.payoff(action_a, action_b)

            result = RoundResult(
                round_num=t,
                action_a=action_a,
                action_b=action_b,
                payoff_a=payoff_a,
                payoff_b=payoff_b,
            )
            self.history.append(result)

            discount = self.discount_factor ** t
            total_a += discount * payoff_a
            total_b += discount * payoff_b

            if result.collision:
                collisions += 1
            if action_a == 1:
                cooperation_a += 1
            if action_b == 1:
                cooperation_b += 1

        social_welfare = sum(
            r.payoff_a + r.payoff_b for r in self.history
        ) / self.num_rounds

        return {
            "history": self.history,
            "total_payoff_a": total_a,
            "total_payoff_b": total_b,
            "cooperation_rate_a": cooperation_a / self.num_rounds,
            "cooperation_rate_b": cooperation_b / self.num_rounds,
            "collision_count": collisions,
            "collision_rate": collisions / self.num_rounds,
            "avg_social_welfare": social_welfare,
            "strategy_a": self.strategy_a_type.value,
            "strategy_b": self.strategy_b_type.value,
        }

    @staticmethod
    def run_tournament(
        stage_game: NormalFormGame,
        num_rounds: int = 100,
        discount_factor: float = 0.95,
    ) -> dict:
        """
        Run a round-robin tournament of all strategies against each other.

        Parameters
        ----------
        stage_game : NormalFormGame
            The stage game to play.
        num_rounds : int
            Rounds per match.
        discount_factor : float
            Discount factor.

        Returns
        -------
        dict
            Tournament results with total scores and head-to-head results.
        """
        strategies = list(StrategyType)
        scores = {s.value: 0.0 for s in strategies}
        matchups = {}

        for sa in strategies:
            for sb in strategies:
                game = RepeatedGame(
                    stage_game, num_rounds, discount_factor, sa, sb, seed=42
                )
                result = game.play()

                scores[sa.value] += result["total_payoff_a"]
                scores[sb.value] += result["total_payoff_b"]

                matchups[(sa.value, sb.value)] = {
                    "payoff_a": result["total_payoff_a"],
                    "payoff_b": result["total_payoff_b"],
                    "cooperation_rate_a": result["cooperation_rate_a"],
                    "cooperation_rate_b": result["cooperation_rate_b"],
                    "collisions": result["collision_count"],
                }

        # Rank strategies
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "scores": scores,
            "ranking": ranking,
            "matchups": matchups,
        }
