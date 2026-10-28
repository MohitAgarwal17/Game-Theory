"""
Unit tests for Repeated Game Framework.
"""

import pytest
import numpy as np

from src.games.normal_form import NormalFormGame
from src.games.repeated_game import RepeatedGame, StrategyType
from src.utils.payoff_matrices import chicken_game


@pytest.fixture
def chicken():
    payoff_a, payoff_b = chicken_game()
    return NormalFormGame(
        payoff_a,
        payoff_b,
        strategy_labels_a=["Go", "Yield"],
        strategy_labels_b=["Go", "Yield"],
        player_names=("Vehicle A", "Vehicle B"),
    )


def test_tit_for_tat_mutual_cooperation(chicken):
    """Test that two Tit-for-Tat agents achieve high cooperation."""
    rep_game = RepeatedGame(
        stage_game=chicken,
        num_rounds=50,
        discount_factor=0.95,
        strategy_a=StrategyType.TIT_FOR_TAT,
        strategy_b=StrategyType.TIT_FOR_TAT,
    )
    results = rep_game.play()

    assert len(results["history"]) == 50
    assert results["cooperation_rate_a"] > 0.8
    assert results["cooperation_rate_b"] > 0.8
    assert results["collision_count"] == 0


def test_grim_trigger_retaliation(chicken):
    """Test that Grim Trigger permanently defects after opponent defects."""
    # Player A is Always Go (Defect), Player B is Grim Trigger
    rep_game = RepeatedGame(
        stage_game=chicken,
        num_rounds=30,
        discount_factor=1.0,
        strategy_a=StrategyType.ALWAYS_GO,
        strategy_b=StrategyType.GRIM_TRIGGER,
    )
    results = rep_game.play()

    # Grim trigger starts by cooperating (Yield=1) on round 0
    # After experiencing Player A's Go (0), B triggers and plays Go forever
    assert results["history"][0].action_b == 1
    for r in results["history"][1:]:
        assert r.action_b == 0


def test_always_yield_vs_always_go(chicken):
    """Test Always Yield vs Always Go yields zero collisions and high payoff for Go."""
    rep_game = RepeatedGame(
        stage_game=chicken,
        num_rounds=20,
        discount_factor=1.0,
        strategy_a=StrategyType.ALWAYS_GO,
        strategy_b=StrategyType.ALWAYS_YIELD,
    )
    results = rep_game.play()

    assert results["collision_count"] == 0
    assert len(results["history"]) == 20
    # Player A plays Go (0), Player B plays Yield (1) every round
    for r in results["history"]:
        assert r.action_a == 0
        assert r.action_b == 1
        assert r.payoff_a == 10.0
        assert r.payoff_b == 2.0
